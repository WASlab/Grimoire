#!/usr/bin/env python3
"""
update_leaderboard.py

This script maintains a leaderboard for your repository’s model performance.
It records the following fields in a Markdown table in 'Leaderboard.md':
    - Dataset
    - Model
    - Number of Parameters
    - Epochs (Number of Epochs Trained)
    - Batch Size
    - Best Accuracy
    - Average Accuracy

Update conditions:
    1. If the model is not present in the leaderboard, a new record is added.
    2. If the model is present and the provided best accuracy is better than the current entry, the record is updated.
    
Additionally, when run without full command-line arguments, the script enters a manual mode where it 
prompts the user for input and—if a record exists—asks whether to overwrite it.

Note:
    Markdown is used for the leaderboard file because it renders well on GitHub. If you prefer a different 
    file format (e.g., CSV, JSON), the read/write functions can be adjusted accordingly.
"""

import os
import sys
import argparse

LEADERBOARD_FILENAME = "Leaderboard.md"
HEADERS = ["Dataset", "Model", "Number of Parameters", "Epochs", "Batch Size", "Best Accuracy", "Average Accuracy"]

def read_leaderboard():
    """
    Reads the leaderboard from a Markdown file and returns a list of records (dictionaries).
    If the file does not exist or is empty, returns an empty list.
    """
    if not os.path.exists(LEADERBOARD_FILENAME):
        return []
    
    entries = []
    with open(LEADERBOARD_FILENAME, "r") as f:
        lines = f.readlines()
    
    # Expecting a Markdown table with header and separator on first two lines.
    if len(lines) < 2:
        return entries

    # Process each table row
    for line in lines[2:]:
        line = line.strip()
        if line.startswith("|") and line.endswith("|"):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if len(cells) == len(HEADERS):
                record = {header: cell for header, cell in zip(HEADERS, cells)}
                entries.append(record)
    return entries

def write_leaderboard(entries):
    """
    Writes the leaderboard entries into the Markdown file in a professional table format.
    """
    with open(LEADERBOARD_FILENAME, "w") as f:
        header_line = "| " + " | ".join(HEADERS) + " |"
        separator_line = "| " + " | ".join(["---"] * len(HEADERS)) + " |"
        f.write(header_line + "\n")
        f.write(separator_line + "\n")
        for entry in entries:
            row = "| " + " | ".join(str(entry.get(header, "")) for header in HEADERS) + " |"
            f.write(row + "\n")

def update_leaderboard_record(dataset, model, num_params, epochs, batch_size, best_accuracy, avg_accuracy, manual=False):
    """
    Updates the leaderboard according to the following conditions:
      - Adds a new record if the model does not exist.
      - Updates the record if the model exists and the new best_accuracy is better.
    
    In manual mode, if the model exists but the new best_accuracy is not better, the user is prompted to decide.
    
    Parameters:
        dataset (str): Name of the dataset.
        model (str): Model name.
        num_params (float/int): Number of parameters.
        epochs (int): Number of epochs the model was trained.
        batch_size (int): Batch size used during training.
        best_accuracy (float): Best accuracy achieved.
        avg_accuracy (float): Average accuracy.
        manual (bool): Flag to indicate if the update is manual.
    """
    entries = read_leaderboard()
    updated = False
    found_index = None

    # Search for an existing entry by model name
    for i, entry in enumerate(entries):
        if entry["Model"] == model:
            found_index = i
            break

    if found_index is None:
        # Model not found; add a new record.
        new_entry = {
            "Dataset": dataset,
            "Model": model,
            "Number of Parameters": str(num_params),
            "Epochs": str(epochs),
            "Batch Size": str(batch_size),
            "Best Accuracy": str(best_accuracy),
            "Average Accuracy": str(avg_accuracy)
        }
        entries.append(new_entry)
        print(f"Added new model '{model}' to the leaderboard.")
        updated = True
    else:
        # Model exists; check if the new best_accuracy is higher.
        try:
            current_best = float(entries[found_index]["Best Accuracy"])
        except ValueError:
            current_best = -1  # Fallback if conversion fails
        if best_accuracy > current_best:
            entries[found_index] = {
                "Dataset": dataset,
                "Model": model,
                "Number of Parameters": str(num_params),
                "Epochs": str(epochs),
                "Batch Size": str(batch_size),
                "Best Accuracy": str(best_accuracy),
                "Average Accuracy": str(avg_accuracy)
            }
            print(f"Updated model '{model}' on the leaderboard with an improved best accuracy.")
            updated = True
        else:
            if manual:
                # Manual mode: display current entry and ask whether to overwrite
                print("Current leaderboard entry for the model:")
                for header in HEADERS:
                    print(f"{header}: {entries[found_index].get(header, '')}")
                ans = input("Do you want to overwrite the previous entry? (YES/NO): ")
                if ans.strip().lower() in ["yes", "y"]:
                    entries[found_index] = {
                        "Dataset": dataset,
                        "Model": model,
                        "Number of Parameters": str(num_params),
                        "Epochs": str(epochs),
                        "Batch Size": str(batch_size),
                        "Best Accuracy": str(best_accuracy),
                        "Average Accuracy": str(avg_accuracy)
                    }
                    print(f"Manually updated model '{model}' on the leaderboard.")
                    updated = True
                else:
                    print("No update made to the leaderboard.")
            else:
                print(f"Model '{model}' was not updated because the current best accuracy ({current_best}) is higher or equal.")
    
    if updated:
        write_leaderboard(entries)
    else:
        print("Leaderboard remains unchanged.")

def main():
    """
    Main function to support both command-line and manual usage.
    When all parameters are provided via command-line, the leaderboard is updated directly.
    If some parameters are missing, the script enters manual mode and prompts the user.
    """
    parser = argparse.ArgumentParser(description="Update the leaderboard with model performance data.")
    parser.add_argument("--dataset", type=str, help="Dataset used")
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--num_params", type=float, help="Number of parameters")
    parser.add_argument("--epochs", type=int, help="Number of epochs trained")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--best_accuracy", type=float, help="Best accuracy achieved")
    parser.add_argument("--avg_accuracy", type=float, help="Average accuracy achieved")
    args = parser.parse_args()

    # If any required argument is missing, enter manual mode.
    if not (args.dataset and args.model and args.num_params is not None and 
            args.epochs is not None and args.batch_size is not None and 
            args.best_accuracy is not None and args.avg_accuracy is not None):
        print("Running in manual mode. Please enter the following details.")
        dataset = input("Dataset: ")
        model = input("Model: ")
        num_params = input("Number of Parameters: ")
        epochs = input("Number of Epochs: ")
        batch_size = input("Batch Size: ")
        best_accuracy = input("Best Accuracy: ")
        avg_accuracy = input("Average Accuracy: ")
        
        # Convert numeric inputs
        try:
            num_params = float(num_params)
            epochs = int(epochs)
            batch_size = int(batch_size)
            best_accuracy = float(best_accuracy)
            avg_accuracy = float(avg_accuracy)
        except ValueError:
            print("Error: Invalid numeric input. Exiting.")
            sys.exit(1)
        
        update_leaderboard_record(dataset, model, num_params, epochs, batch_size, best_accuracy, avg_accuracy, manual=True)
    else:
        update_leaderboard_record(args.dataset, args.model, args.num_params, args.epochs, args.batch_size, args.best_accuracy, args.avg_accuracy)

if __name__ == "__main__":
    main()

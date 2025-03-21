# Description: Example code snippets for using the train_model_step_lr function.

def example_train(train_loader, val_loader, test_loader):
    """
    Example calls to train_model_step_lr for different optimizers.
    Assumes that train_loader, val_loader, and test_loader have been defined.
    Also assumes that a model (e.g. EfficientCNN) is imported from your model module.
    """
    from model import EfficientCNN

    # Create a fresh model instance
    num_classes = 10  # Change as needed
    model = EfficientCNN(num_classes=num_classes).to(DEVICE)

    # Example scheduler parameters (derived from training configuration)
    scheduler_params = {
        "total_steps": NUM_EPOCHS * len(train_loader),
        "warmup_steps": int(0.1 * NUM_EPOCHS * len(train_loader)),  # e.g., 10% of total steps for warmup
        "init_lr": 1e-4,
        "peak_lr": 1e-3,
        "end_lr": 1e-5,
        "warmup_type": "cosine",
        "decay_type": "cosine"
    }

    # --------- Example 1: Using Muon optimizer ---------
    muon_optimizer_params = {
        "lr": 2e-3,
        "wd": 0.1,
        "momentum": 0.95,
        "nesterov": True,
        "ns_steps": 6,
        "adamw_betas": (0.9, 0.98),
        "adamw_eps": 1e-6
    }
    val_metrics, test_metrics, _ = train_model_step_lr(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        scheduler_params=scheduler_params,
        optimizer_name="Muon",
        optimizer_params=muon_optimizer_params
    )
    print("Muon optimizer metrics:", val_metrics, test_metrics)

    # --------- Example 2: Using AdamW optimizer ---------
    # Reset model parameters between experiments:
    model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
    adamw_optimizer_params = {
        "lr": 1e-3,
        "weight_decay": 0.05,
        "betas": (0.85, 0.997),
        "eps": 1e-7
    }
    val_metrics, test_metrics, _ = train_model_step_lr(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        scheduler_params=scheduler_params,
        optimizer_name="AdamW",
        optimizer_params=adamw_optimizer_params
    )
    print("AdamW optimizer metrics:", val_metrics, test_metrics)

    # --------- Example 3: Using SGD optimizer ---------
    # Reset model parameters between experiments:
    model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
    sgd_optimizer_params = {
        "lr": 5e-3,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "nesterov": True
    }
    val_metrics, test_metrics, _ = train_model_step_lr(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        scheduler_params=scheduler_params,
        optimizer_name="SGD",
        optimizer_params=sgd_optimizer_params
    )
    print("SGD optimizer metrics:", val_metrics, test_metrics)
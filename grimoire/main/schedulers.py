############################################################
# schedulers.py
############################################################

import math
import warnings

class StepLRScheduler:
    """
    A step-wise learning rate scheduler that updates on *every batch* (each training step).
    
    Features:
      - Optional warmup from init_lr to peak_lr (over warmup_steps).
      - Optional decay from peak_lr to end_lr (over total_steps - warmup_steps).
      - Multiple warmup/decay types: "linear", "exponential", "polynomial", "cosine", "static".
      - If 'init_lr=False', we skip the warmup phase (immediately use peak_lr).
      - If 'end_lr=False', we skip the decay phase (stay at peak_lr forever after warmup).
      
    Usage:
      1) Initialize with an optimizer, total_steps, warmup_steps, etc.
      2) Call .step() *each time* you process a batch (i.e., each training step).
      3) The scheduler automatically updates optimizer.param_groups[...]["lr"].
    """

    def __init__(
        self,
        optimizer,
        total_steps,
        warmup_steps=0,
        init_lr=1e-4,
        peak_lr=1e-3,
        end_lr=1e-5,
        warmup_type="linear",
        decay_type="cosine",
    ):
        """
        Args:
            optimizer: torch Optimizer object
            total_steps (int): total number of steps (batches) for training
            warmup_steps (int): number of steps used for warmup
            init_lr (float or bool): starting LR. If False, no warmup phase (start at peak_lr).
            peak_lr (float): LR at the end of warmup. If no warmup, remains constant until decay starts.
            end_lr (float or bool): final LR after decay. If False, no decay occurs; remain at peak_lr.
            warmup_type (str): "linear", "exponential", "polynomial", or "static"
            decay_type (str): "linear", "exponential", "polynomial", "cosine", or "static"
        """
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.init_lr = init_lr
        self.peak_lr = peak_lr
        self.end_lr = end_lr
        self.warmup_type = warmup_type.lower()
        self.decay_type = decay_type.lower()

        self.current_step = 0

        # --- Validate arguments ---
        if self.init_lr is False and self.warmup_steps > 0:
            warnings.warn(
                "warmup_steps > 0 but init_lr=False. Warmup phase is disabled; training starts at peak_lr."
            )
        if self.end_lr is False and (self.decay_type != "static"):
            warnings.warn(
                "end_lr=False but decay_type is not 'static'. Decay phase is disabled; LR stays at peak_lr."
            )
        if self.total_steps <= self.warmup_steps and self.warmup_steps > 0:
            warnings.warn(
                f"total_steps={self.total_steps} is less than or equal to warmup_steps={self.warmup_steps}. "
                "You'll never see a decay phase."
            )

        # --- Initialize LR to something sensible ---
        self._set_lr(self.init_lr if self.init_lr is not False else self.peak_lr)

    def step(self):
        """Call this once per training step (batch)."""
        self.current_step += 1
        new_lr = self._compute_lr()
        self._set_lr(new_lr)

    def _compute_lr(self):
        """
        Computes the LR for the current step based on warmup or decay phase.
        """
        if (self.init_lr is False) and (self.end_lr is False):
            # 100% static peak LR
            return self.peak_lr

        # Check if we are in warmup phase
        if self.init_lr is not False and (self.current_step <= self.warmup_steps):
            # Warmup from init_lr -> peak_lr
            return self._warmup_lr()
        else:
            # After warmup, either stay at peak_lr or decay
            if self.end_lr is False:
                # No decay, remain at peak
                return self.peak_lr
            return self._decay_lr()

    def _warmup_lr(self):
        """Compute LR during warmup phase."""
        progress = self.current_step / max(1, self.warmup_steps)  # from 0 to 1
        if self.warmup_type == "linear":
            return self.init_lr + (self.peak_lr - self.init_lr) * progress
        elif self.warmup_type == "exponential":
            # Exponential from init to peak: init_lr * (peak_lr/init_lr)^progress
            ratio = self.peak_lr / max(1e-12, self.init_lr)
            return self.init_lr * (ratio ** progress)
        elif self.warmup_type == "polynomial":
            # polynomial scale from init to peak
            # e.g. "progress^2" or similar
            # We'll pick ^2 for demonstration
            return self.init_lr + (self.peak_lr - self.init_lr) * (progress ** 2)
        elif self.warmup_type == "static":
            # Remain at init_lr until warmup_steps finishes, then jump
            return self.init_lr
        else:
            raise ValueError(f"Unknown warmup_type={self.warmup_type}")

    def _decay_lr(self):
        """Compute LR during decay phase (after warmup)."""
        # Steps spent in decay
        decay_step = self.current_step - self.warmup_steps
        # Max steps of decay
        decay_total = max(1, self.total_steps - self.warmup_steps)
        progress = decay_step / decay_total  # from 0 to 1

        if self.decay_type == "linear":
            return self.peak_lr + (self.end_lr - self.peak_lr) * progress
        elif self.decay_type == "exponential":
            # Exponential from peak to end: peak_lr * (end_lr/peak_lr)^progress
            ratio = self.end_lr / max(1e-12, self.peak_lr)
            return self.peak_lr * (ratio ** progress)
        elif self.decay_type == "polynomial":
            # polynomial scale from peak to end
            # We'll pick ^2 for demonstration
            return self.peak_lr + (self.end_lr - self.peak_lr) * (progress ** 2)
        elif self.decay_type == "cosine":
            # Cosine from peak to end
            return self.end_lr + 0.5 * (self.peak_lr - self.end_lr) * (
                1 + math.cos(math.pi * progress)
            )
        elif self.decay_type == "static":
            # Remain at peak
            return self.peak_lr
        else:
            raise ValueError(f"Unknown decay_type={self.decay_type}")

    def _set_lr(self, lr):
        """Applies lr to all param groups in the optimizer."""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_lr(self):
        """Return the current LR for the first param_group."""
        return self.optimizer.param_groups[0]["lr"]


class EpochLRScheduler:
    """
    An epoch-wise learning rate scheduler that updates on *every epoch*.
    
    Features:
      - Optional warmup from init_lr to peak_lr (over warmup_epochs).
      - Optional decay from peak_lr to end_lr (over total_epochs - warmup_epochs).
      - Multiple warmup/decay types: "linear", "exponential", "polynomial", "cosine", "static".
      - If 'init_lr=False', we skip the warmup phase (immediately use peak_lr).
      - If 'end_lr=False', we skip the decay phase (stay at peak_lr forever after warmup).
      
    Usage:
      1) Initialize with an optimizer, total_epochs, warmup_epochs, etc.
      2) Call .step() *at the end of each epoch*.
      3) The scheduler automatically updates optimizer.param_groups[...]["lr"].
    """

    def __init__(
        self,
        optimizer,
        total_epochs,
        warmup_epochs=0,
        init_lr=1e-4,
        peak_lr=1e-3,
        end_lr=1e-5,
        warmup_type="linear",
        decay_type="cosine",
    ):
        """
        Args:
            optimizer: torch Optimizer object
            total_epochs (int): total number of epochs for training
            warmup_epochs (int): number of epochs used for warmup
            init_lr (float or bool): starting LR. If False, no warmup (start at peak_lr).
            peak_lr (float): LR at the end of warmup. If no warmup, remains constant until decay starts.
            end_lr (float or bool): final LR after decay. If False, skip decay and remain at peak_lr.
            warmup_type (str): "linear", "exponential", "polynomial", or "static"
            decay_type (str): "linear", "exponential", "polynomial", "cosine", or "static"
        """
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.init_lr = init_lr
        self.peak_lr = peak_lr
        self.end_lr = end_lr
        self.warmup_type = warmup_type.lower()
        self.decay_type = decay_type.lower()

        self.current_epoch = 0

        # --- Validate arguments ---
        if self.init_lr is False and self.warmup_epochs > 0:
            warnings.warn(
                "warmup_epochs > 0 but init_lr=False. Warmup phase is disabled; training starts at peak_lr."
            )
        if self.end_lr is False and (self.decay_type != "static"):
            warnings.warn(
                "end_lr=False but decay_type != 'static'. Decay is disabled; LR remains at peak_lr."
            )
        if self.total_epochs <= self.warmup_epochs and self.warmup_epochs > 0:
            warnings.warn(
                f"total_epochs={self.total_epochs} <= warmup_epochs={self.warmup_epochs}. "
                "No decay phase will occur."
            )

        # --- Initialize LR to something sensible ---
        self._set_lr(self.init_lr if self.init_lr is not False else self.peak_lr)

    def step(self):
        """Call this once per epoch (end of epoch)."""
        self.current_epoch += 1
        new_lr = self._compute_lr()
        self._set_lr(new_lr)

    def _compute_lr(self):
        """Compute LR for the current epoch (1-based index)."""
        if (self.init_lr is False) and (self.end_lr is False):
            # 100% static peak LR
            return self.peak_lr

        if self.init_lr is not False and (self.current_epoch <= self.warmup_epochs):
            # Warmup phase
            return self._warmup_lr()
        else:
            # Decay or stay at peak
            if self.end_lr is False:
                return self.peak_lr
            return self._decay_lr()

    def _warmup_lr(self):
        """Compute LR during warmup phase."""
        progress = self.current_epoch / max(1, self.warmup_epochs)
        if self.warmup_type == "linear":
            return self.init_lr + (self.peak_lr - self.init_lr) * progress
        elif self.warmup_type == "exponential":
            ratio = self.peak_lr / max(1e-12, self.init_lr)
            return self.init_lr * (ratio ** progress)
        elif self.warmup_type == "polynomial":
            return self.init_lr + (self.peak_lr - self.init_lr) * (progress ** 2)
        elif self.warmup_type == "static":
            return self.init_lr
        else:
            raise ValueError(f"Unknown warmup_type={self.warmup_type}")

    def _decay_lr(self):
        """Compute LR during decay phase."""
        decay_epoch = self.current_epoch - self.warmup_epochs
        decay_total = max(1, self.total_epochs - self.warmup_epochs)
        progress = decay_epoch / decay_total

        if self.decay_type == "linear":
            return self.peak_lr + (self.end_lr - self.peak_lr) * progress
        elif self.decay_type == "exponential":
            ratio = self.end_lr / max(1e-12, self.peak_lr)
            return self.peak_lr * (ratio ** progress)
        elif self.decay_type == "polynomial":
            return self.peak_lr + (self.end_lr - self.peak_lr) * (progress ** 2)
        elif self.decay_type == "cosine":
            return self.end_lr + 0.5 * (self.peak_lr - self.end_lr) * (
                1 + math.cos(math.pi * progress)
            )
        elif self.decay_type == "static":
            return self.peak_lr
        else:
            raise ValueError(f"Unknown decay_type={self.decay_type}")

    def _set_lr(self, lr):
        """Applies lr to all param groups in the optimizer."""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_lr(self):
        """Return the current LR for the first param_group."""
        return self.optimizer.param_groups[0]["lr"]

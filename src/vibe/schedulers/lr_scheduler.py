import math


class WarmupCosineScheduler:
    """
    Learning rate scheduler with warmup followed by cosine decay.
    
    This scheduler increases the learning rate linearly during warmup phase,
    then decreases it following a cosine curve.
    """
    
    def __init__(
        self,
        optimizer,
        min_lr,
        max_lr,
        warmup_epoch,
        fix_epoch,
        step_per_epoch,
        decay_rate=0.999
    ):
        """
        Initialize the WarmupCosineScheduler.
        
        Args:
            optimizer: Optimizer to adjust learning rate for
            min_lr: Minimum learning rate
            max_lr: Maximum learning rate (peak after warmup)
            warmup_epoch: Number of epochs for warmup phase
            fix_epoch: Total number of epochs
            step_per_epoch: Number of steps per epoch
            decay_rate: Rate of decay for alternative exponential decay
        """
        self.optimizer = optimizer
        assert min_lr <= max_lr, "Minimum LR must be less than or equal to maximum LR"
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.warmup_step = warmup_epoch * step_per_epoch
        self.fix_step = fix_epoch * step_per_epoch
        self.current_step = 0.0
        self.decay_rate = decay_rate

    def set_lr(self):
        """Update learning rate for optimizer based on current step."""
        new_lr = self.clr(self.current_step)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr

    def step(self, step=None):
        """
        Perform scheduler step and return the new learning rate.
        
        Args:
            step: Optional step to set current_step to
                 
        Returns:
            New learning rate
        """
        if step is not None:
            self.current_step = step
        new_lr = self.set_lr()
        self.current_step += 1
        return new_lr

    def clr(self, step):
        """
        Calculate learning rate based on current step.
        
        Args:
            step: Current step number
            
        Returns:
            Calculated learning rate
        """
        if step < self.warmup_step:
            # Linear warmup phase
            return self.min_lr + (self.max_lr - self.min_lr) * (step / self.warmup_step)
        elif step >= self.warmup_step and step < self.fix_step:
            # Cosine decay phase
            return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
                1 + math.cos(
                    math.pi * (step - self.warmup_step) / (self.fix_step - self.warmup_step)
                )
            )
            
            # Alternative: Exponential decay phase (commented out)
            # decay_steps = step - self.warmup_step
            # return self.min_lr + (self.max_lr - self.min_lr) * (self.decay_rate ** decay_steps) 
        else:
            # After fix_step, return minimum learning rate
            return self.min_lr
            
    def state_dict(self):
        """
        Returns the state of the scheduler as a dict.
        
        This method captures the current state of the scheduler, which can later be
        used to restore the scheduler to this state using load_state_dict.
        
        Returns:
            dict: State of the scheduler
        """
        return {
            'min_lr': self.min_lr,
            'max_lr': self.max_lr,
            'warmup_step': self.warmup_step,
            'fix_step': self.fix_step,
            'current_step': self.current_step,
            'decay_rate': self.decay_rate
        }
        
    def load_state_dict(self, state_dict):
        """
        Loads the scheduler state.
        
        This method restores the scheduler to the state specified by state_dict.
        
        Args:
            state_dict (dict): Scheduler state. Should be an object returned from a call to state_dict.
        """
        self.min_lr = state_dict['min_lr']
        self.max_lr = state_dict['max_lr']
        self.warmup_step = state_dict['warmup_step']
        self.fix_step = state_dict['fix_step']
        self.current_step = state_dict['current_step']
        self.decay_rate = state_dict['decay_rate']


class StepScheduler:
    """
    Step learning rate scheduler.
    
    Decreases learning rate by a factor of 0.1 every step_epoch_size epochs.
    """
    
    def __init__(
        self,
        optimizer,
        lr,
        step_per_epoch,
        step_epoch_size,
    ):
        """
        Initialize StepScheduler.
        
        Args:
            optimizer: Optimizer to adjust learning rate for
            lr: Initial learning rate
            step_per_epoch: Number of steps per epoch
            step_epoch_size: Epochs between learning rate decay steps
        """
        self.optimizer = optimizer
        self.lr = lr
        self.step_size = step_epoch_size * step_per_epoch
        self.current_step = 0.0

    def set_lr(self):
        """Update learning rate for optimizer based on current step."""
        new_lr = self.clr(self.current_step)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr

    def step(self, step=None):
        """
        Perform scheduler step and return the new learning rate.
        
        Args:
            step: Optional step to set current_step to
                 
        Returns:
            New learning rate
        """
        if step is not None:
            self.current_step = step
        new_lr = self.set_lr()
        self.current_step += 1
        return new_lr

    def clr(self, step):
        """
        Calculate learning rate based on current step.
        
        Args:
            step: Current step number
            
        Returns:
            Calculated learning rate
        """
        ratio = 0.1 ** (step // self.step_size)
        return self.lr * ratio
        
    def state_dict(self):
        """
        Returns the state of the scheduler as a dict.
        
        This method captures the current state of the scheduler, which can later be
        used to restore the scheduler to this state using load_state_dict.
        
        Returns:
            dict: State of the scheduler
        """
        return {
            'lr': self.lr,
            'step_size': self.step_size,
            'current_step': self.current_step
        }
        
    def load_state_dict(self, state_dict):
        """
        Loads the scheduler state.
        
        This method restores the scheduler to the state specified by state_dict.
        
        Args:
            state_dict (dict): Scheduler state. Should be an object returned from a call to state_dict.
        """
        self.lr = state_dict['lr']
        self.step_size = state_dict['step_size']
        self.current_step = state_dict['current_step']


class CyclicLRScheduler:
    """
    Cyclical Learning Rate scheduler.
    
    Implementation of the cyclical learning rate policy that lets the learning rate
    cyclically vary between lower and upper bounds.
    """
    
    def __init__(
        self,
        optimizer,
        base_lr,
        max_lr,
        fix_epoch,
        step_per_epoch,
        num_cycles=1,
        mode="triangular",
        gamma=1.0,
        cycle_momentum=False,
        base_momentum=0.8,
        max_momentum=0.9
    ):
        """
        Initialize CyclicLRScheduler.
        
        Args:
            optimizer: Optimizer instance to update
            base_lr: Lower learning rate boundary
            max_lr: Upper learning rate boundary
            fix_epoch: Total number of epochs for training
            step_per_epoch: Number of steps per epoch
            num_cycles: Number of complete cycles to perform over the entire training
            mode: One of {'triangular', 'triangular2', 'exp_range'}
            gamma: Decay factor for exp_range mode
            cycle_momentum: Whether to cycle momentum inversely to learning rate
            base_momentum: Lower momentum boundary (when cycle_momentum=True)
            max_momentum: Upper momentum boundary (when cycle_momentum=True)
        """
        self.optimizer = optimizer
        self.base_lr = base_lr if isinstance(base_lr, list) else [base_lr] * len(optimizer.param_groups)
        self.max_lr = max_lr if isinstance(max_lr, list) else [max_lr] * len(optimizer.param_groups)
        
        # Calculate total steps and step size based on number of cycles
        self.total_steps = fix_epoch * step_per_epoch
        self.step_size = self.total_steps / (2 * num_cycles)
        
        self.mode = mode
        self.gamma = gamma
        self.current_step = 0.0
        
        # Momentum cycling
        self.cycle_momentum = cycle_momentum
        if cycle_momentum:
            self.base_momentum = base_momentum if isinstance(base_momentum, list) else [base_momentum] * len(optimizer.param_groups)
            self.max_momentum = max_momentum if isinstance(max_momentum, list) else [max_momentum] * len(optimizer.param_groups)
            
            # Initialize momentum values
            for param_group, max_mom in zip(self.optimizer.param_groups, self.max_momentum):
                if 'momentum' in param_group:
                    param_group['momentum'] = max_mom
                elif 'betas' in param_group:
                    param_group['betas'] = (max_mom, param_group['betas'][1])

    def set_lr(self):
        """Set the learning rate for all optimizer param groups."""
        new_lr = self.clr(self.current_step)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
            
        # Update momentum if cycling is enabled
        if self.cycle_momentum:
            self._update_momentum(self.current_step)
            
        return new_lr

    def step(self, step=None):
        """
        Perform scheduler step and return the new learning rate.
        
        Args:
            step: Optional step to set current_step to
                 
        Returns:
            New learning rate
        """
        if step is not None:
            self.current_step = step
        new_lr = self.set_lr()
        self.current_step += 1
        return new_lr

    def clr(self, step):
        """
        Calculate learning rate based on cycle position.
        
        Args:
            step: Current step number
            
        Returns:
            Calculated learning rate
        """
        # Calculate cycle and x position within cycle
        cycle = math.floor(1 + step / (2 * self.step_size))
        x = abs(step / self.step_size - 2 * cycle + 1)
        
        # Calculate scale based on the selected mode
        if self.mode == 'triangular':
            scale = 1.0
        elif self.mode == 'triangular2':
            scale = 1.0 / (2 ** (cycle - 1))
        elif self.mode == 'exp_range':
            scale = self.gamma ** step
        else:
            raise ValueError(f"Mode '{self.mode}' not supported")
        
        # Calculate learning rate
        lr_diff = self.max_lr[0] - self.base_lr[0]
        return self.base_lr[0] + (lr_diff * max(0, (1 - x)) * scale)
    
    def _update_momentum(self, step):
        """
        Update momentum inversely to learning rate.
        
        When learning rate is high, momentum is low and vice versa.
        
        Args:
            step: Current step number
        """
        if not self.cycle_momentum:
            return
            
        # Calculate cycle and x position within cycle (same as for LR)
        cycle = math.floor(1 + step / (2 * self.step_size))
        x = abs(step / self.step_size - 2 * cycle + 1)
        
        # Momentum changes inversely to learning rate
        for i, param_group in enumerate(self.optimizer.param_groups):
            momentum_diff = self.max_momentum[i] - self.base_momentum[i]
            momentum = self.max_momentum[i] - momentum_diff * max(0, (1 - x))
            
            if 'momentum' in param_group:
                param_group['momentum'] = momentum
            elif 'betas' in param_group:
                param_group['betas'] = (momentum, param_group['betas'][1])
                
    def state_dict(self):
        """
        Returns the state of the scheduler as a dict.
        
        This method captures the current state of the scheduler, which can later be
        used to restore the scheduler to this state using load_state_dict.
        
        Returns:
            dict: State of the scheduler
        """
        state_dict = {
            'base_lr': self.base_lr,
            'max_lr': self.max_lr,
            'total_steps': self.total_steps,
            'step_size': self.step_size,
            'mode': self.mode,
            'gamma': self.gamma,
            'current_step': self.current_step,
            'cycle_momentum': self.cycle_momentum
        }
        
        if self.cycle_momentum:
            state_dict.update({
                'base_momentum': self.base_momentum,
                'max_momentum': self.max_momentum
            })
            
        return state_dict
        
    def load_state_dict(self, state_dict):
        """
        Loads the scheduler state.
        
        This method restores the scheduler to the state specified by state_dict.
        
        Args:
            state_dict (dict): Scheduler state. Should be an object returned from a call to state_dict.
        """
        self.base_lr = state_dict['base_lr']
        self.max_lr = state_dict['max_lr']
        self.total_steps = state_dict['total_steps']
        self.step_size = state_dict['step_size']
        self.mode = state_dict['mode']
        self.gamma = state_dict['gamma']
        self.current_step = state_dict['current_step']
        self.cycle_momentum = state_dict['cycle_momentum']
        
        if self.cycle_momentum:
            self.base_momentum = state_dict['base_momentum']
            self.max_momentum = state_dict['max_momentum']
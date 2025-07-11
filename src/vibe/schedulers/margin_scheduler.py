import math


class MarginScheduler:
    """
    A scheduler that dynamically adjusts the margin parameter over training steps.
    
    This scheduler supports both exponential and linear margin increase strategies,
    allowing for gradual adjustment of the margin parameter from an initial to a 
    final value during training.
    """
    
    def __init__(
        self,
        criterion,
        increase_start_epoch,
        fix_epoch,
        step_per_epoch,
        initial_margin,
        final_margin,
        increase_type='exp',
    ):
        """
        Initialize the margin scheduler.
        
        Parameters:
            criterion: Loss function that has an 'update' method for margin adjustment
            increase_start_epoch: Epoch to start increasing the margin
            fix_epoch: Epoch at which the margin reaches its final value
            step_per_epoch: Number of steps per training epoch
            initial_margin: Starting margin value
            final_margin: Final margin value
            increase_type: Type of margin increase ('exp' for exponential, otherwise linear)
        """
        assert hasattr(criterion, 'update'), "Loss function must have 'update()' attribute."
        self.criterion = criterion
        self.increase_start_step = increase_start_epoch * step_per_epoch
        self.fix_step = fix_epoch * step_per_epoch
        self.initial_margin = initial_margin
        self.final_margin = final_margin
        self.increase_type = increase_type
        self.margin = initial_margin

        self.current_step = 0
        # Total steps for increasing margin from initial to final value
        self.increase_step = self.fix_step - self.increase_start_step

        self.init_margin()

    def init_margin(self):
        """Initialize the criterion with the initial margin value."""
        self.criterion.update(margin=self.initial_margin)

    def step(self, current_step=None):
        """
        Update the margin based on the current training step.
        
        Parameters:
            current_step: Optional step counter to use (uses internal counter if None)
        """
        if current_step is not None:
            self.current_step = current_step

        # Calculate and update the new margin value
        self.margin = self.iter_margin()
        self.criterion.update(margin=self.margin)
        self.current_step += 1

    def iter_margin(self):
        """
        Calculate the margin value for the current step.
        
        Returns:
            The appropriate margin value for the current training step
        """
        # Before increase phase, use initial margin
        if self.current_step < self.increase_start_step:
            return self.initial_margin

        # After fix phase, use final margin
        if self.current_step >= self.fix_step:
            return self.final_margin

        # Constants for exponential scaling
        a = 1.0
        b = 1e-3

        # Calculate steps since start of increase phase
        current_step = self.current_step - self.increase_start_step
        
        if self.increase_type == 'exp':
            # Exponentially increase the margin
            # This creates a smooth curve that starts slowly and accelerates
            ratio = 1.0 - math.exp(
                (current_step / self.increase_step) *
                math.log(b / (a + 1e-6))) * a
        else:
            # Linearly increase the margin
            # This creates a constant rate of increase
            ratio = 1.0 * current_step / self.increase_step
            
        # Interpolate between initial and final margin values
        return self.initial_margin + (self.final_margin - self.initial_margin) * ratio

    def get_margin(self):
        """
        Get the current margin value.
        
        Returns:
            The current margin value
        """
        return self.margin
        
    def state_dict(self):
        """
        Returns the state of the scheduler as a dict.
        
        This method captures the current state of the margin scheduler, which can later be
        used to restore the scheduler to this state using load_state_dict.
        
        Returns:
            dict: State of the scheduler containing all necessary parameters to resume
                  margin scheduling from a particular point in training
        """
        return {
            'increase_start_step': self.increase_start_step,
            'fix_step': self.fix_step,
            'initial_margin': self.initial_margin,
            'final_margin': self.final_margin,
            'increase_type': self.increase_type,
            'margin': self.margin,
            'current_step': self.current_step,
            'increase_step': self.increase_step
        }
        
    def load_state_dict(self, state_dict):
        """
        Loads the scheduler state.
        
        This method restores the margin scheduler to the state specified by state_dict,
        allowing training to resume with the same margin scheduling parameters.
        
        Args:
            state_dict (dict): Scheduler state. Should be an object returned from a call to state_dict.
        """
        self.increase_start_step = state_dict['increase_start_step']
        self.fix_step = state_dict['fix_step']
        self.initial_margin = state_dict['initial_margin']
        self.final_margin = state_dict['final_margin']
        self.increase_type = state_dict['increase_type']
        self.margin = state_dict['margin']
        self.current_step = state_dict['current_step']
        self.increase_step = state_dict['increase_step']
        
        # Update the criterion with current margin value to ensure consistency
        self.criterion.update(margin=self.margin)
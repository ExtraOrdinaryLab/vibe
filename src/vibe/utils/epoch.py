import logging

logger = logging.getLogger(__name__)


class EpochLogger:
    """
    A logger for recording and formatting training statistics during epochs.
    
    This class handles formatting statistics with appropriate precision,
    writing logs to files, and printing to console/logger.
    """
    
    def __init__(self, save_file, precision=2):
        """
        Initialize the epoch logger.
        
        Args:
            save_file (str): Path to the file where logs will be saved
            precision (int): Number of decimal places for floating point values
        """
        self.save_file = save_file
        self.precision = precision

    def item_to_string(self, key, value, prefix=None):
        """
        Convert a single statistic item to a formatted string.
        
        Args:
            key (str): Name of the statistic
            value: Value of the statistic
            prefix (str, optional): Prefix to add before the key
            
        Returns:
            str: Formatted string representation of the statistic
        """
        # Format floats based on their magnitude
        if isinstance(value, float) and 1.0 < value < 100.0:
            value = f"{value:.{self.precision}f}"
        elif isinstance(value, float):
            value = f"{value:.{self.precision}e}"
            
        # Add prefix if provided
        if prefix is not None:
            key = f"{prefix} {key}"
            
        return f"{key}: {value}"

    def stats_to_string(self, stats, prefix=None):
        """
        Convert a dictionary of statistics to a comma-separated string.
        
        Args:
            stats (dict): Dictionary of statistics
            prefix (str, optional): Prefix to add before each key
            
        Returns:
            str: Comma-separated string of formatted statistics
        """
        return ", ".join(
            [self.item_to_string(k, v, prefix) for k, v in stats.items()]
        )

    def log_stats(self, stats_meta, stats, stage='train', verbose=True):
        """
        Log statistics to file and optionally to console.
        
        Args:
            stats_meta (dict): Metadata statistics (e.g., epoch number)
            stats (dict): Performance statistics
            stage (str): Training stage identifier (e.g., 'train', 'val')
            verbose (bool): Whether to also log to console via logger
        """
        # Format metadata statistics
        string = self.stats_to_string(stats_meta)
        
        # Append stage-specific statistics if provided
        if stats is not None:
            string += " - " + self.stats_to_string(stats, stage)

        # Write to log file
        with open(self.save_file, "a") as fw:
            print(string, file=fw)
            
        # Optionally log to console
        if verbose:
            logger.info(string)


class EpochCounter:
    """
    Iterator that counts epochs up to a specified limit.
    
    This class helps track the current epoch during training
    and provides methods to save and load the counter state.
    """
    
    def __init__(self, limit):
        """
        Initialize the epoch counter.
        
        Args:
            limit (int): Maximum number of epochs
        """
        self.current = 0
        self.limit = limit

    def __iter__(self):
        """
        Make this class iterable.
        
        Returns:
            self: The iterator object itself
        """
        return self

    def __next__(self):
        """
        Get the next epoch number or raise StopIteration.
        
        Returns:
            int: The next epoch number
            
        Raises:
            StopIteration: When the epoch limit is reached
        """
        if self.current < self.limit:
            self.current += 1
            logger.info(f"Going into epoch {self.current}")
            return self.current
        raise StopIteration

    def save(self, path, device=None):
        """
        Save the current epoch number to a file.
        
        Args:
            path (str): File path to save the epoch number
            device: Unused parameter (for compatibility)
        """
        with open(path, "w") as f:
            f.write(str(self.current))

    def load(self, path, device=None):
        """
        Load the current epoch number from a file.
        
        Args:
            path (str): File path to load the epoch number from
            device: Unused parameter (for compatibility)
        """
        with open(path) as f:
            saved_value = int(f.read())
            self.current = saved_value
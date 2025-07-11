import os
import yaml


class Config:
    """Configuration object that converts dictionary items into object attributes.
    
    This class provides an easy access to configuration parameters by converting
    dictionary key-value pairs into object attributes.
    """
    def __init__(self, conf_dict):
        """Initialize Config object from a dictionary.
        
        Args:
            conf_dict (dict): Configuration dictionary
        """
        for key, value in conf_dict.items():
            self.__dict__[key] = value


def convert_to_yaml(overrides):
    """Convert command line arguments to YAML format for overrides.
    
    Args:
        overrides (list): List of command line arguments in the format ['--arg1=val1', '--arg2=val2']
        
    Returns:
        str: YAML formatted string
    """
    yaml_string = ""

    # Handle '--arg=val' type arguments
    joined_args = "=".join(overrides)
    split_args = joined_args.split("=")

    for arg in split_args:
        if arg.startswith("--"):
            yaml_string += "\n" + arg[len("--"):] + ":"
        else:
            yaml_string += " " + arg

    return yaml_string.strip()


def yaml_config_loader(conf_file, overrides=None):
    """Load configuration from YAML file and apply overrides.
    
    Args:
        conf_file (str): Path to the YAML configuration file
        overrides (str, optional): YAML formatted string with override values
        
    Returns:
        dict: Merged configuration dictionary
    """
    # Load base configuration from file
    with open(conf_file, "r") as fr:
        conf_dict = yaml.load(fr, Loader=yaml.FullLoader)
    
    # Apply overrides if provided
    if overrides is not None:
        overrides = yaml.load(overrides, Loader=yaml.FullLoader)
        conf_dict.update(overrides)
        
    return conf_dict


def build_config(config_file, overrides=None, copy=False):
    """Build configuration object from YAML file with optional overrides.
    
    Args:
        config_file (str): Path to the configuration file
        overrides (list, optional): List of override arguments
        copy (bool, optional): Whether to save a copy of the final config
        
    Returns:
        Config: Configuration object
        
    Raises:
        ValueError: If the config file is not a YAML file
    """
    if config_file.endswith(".yaml"):
        # Convert overrides to YAML format if provided
        if overrides is not None:
            overrides = convert_to_yaml(overrides)
        
        # Load configuration with overrides
        conf_dict = yaml_config_loader(config_file, overrides)
        
        # Save a copy of the configuration if requested
        if copy and 'exp_dir' in conf_dict:
            os.makedirs(conf_dict['exp_dir'], exist_ok=True)
            saved_path = os.path.join(conf_dict['exp_dir'], 'config.yaml')
            with open(saved_path, 'w') as f:
                f.write(yaml.dump(conf_dict))
    else:
        raise ValueError("Unknown config file format. Only YAML files are supported.")

    return Config(conf_dict)
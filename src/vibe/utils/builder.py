import re
import importlib

from .config import Config


def dynamic_import(import_path):
    """
    Dynamically import a module and return a specific attribute from it.
    
    Args:
        import_path (str): Full import path in format 'module.submodule.attribute'
        
    Returns:
        The imported attribute (class, function, etc.)
    """
    module_name, obj_name = import_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, obj_name)


def is_ref_type(value: str):
    """
    Check if a string is a reference type (format: <name>).
    
    Args:
        value (str): String value to check
        
    Returns:
        bool: True if the value is a reference type, False otherwise
        
    Raises:
        AssertionError: If input is not a string
    """
    assert isinstance(value, str), 'Input value is not a str.'
    # Pattern matches strings like <name>, <variable123>, etc.
    return bool(re.match('^<[a-zA-Z]\\w*>$', value))


def is_built(instance):
    """
    Recursively check if an instance is fully built (contains no unresolved references).
    
    Args:
        instance: The instance to check (can be dict, list, str, or other types)
        
    Returns:
        bool: True if fully built, False otherwise
    """
    if isinstance(instance, dict):
        # Object definitions with 'obj' and 'args' keys need to be built
        if 'obj' in instance and 'args' in instance:
            return False
        # Check all values in the dictionary
        for value in instance.values():
            if not is_built(value):
                return False
    elif isinstance(instance, str):
        if '/' in instance:  # Path string might contain references
            parts = instance.split('/')
            return is_built(parts)
        elif is_ref_type(instance):  # Reference type needs to be built
            return False
    elif isinstance(instance, list):
        # Check all items in the list
        for item in instance:
            if not is_built(item):
                return False
    # If none of the above conditions were met, the instance is built
    return True


def deep_build(instance, config, build_space: set = None):
    """
    Recursively build an instance by resolving references from the config.
    
    Args:
        instance: The instance to build (can be dict, list, str, or other types)
        config (Config): Configuration object containing reference values
        build_space (set, optional): Set tracking references being built to detect cycles
        
    Returns:
        The fully built instance with all references resolved
        
    Raises:
        ValueError: If cross-referencing is detected
        AssertionError: If a reference isn't found in config or args aren't a dict
    """
    # If already built, return as is
    if is_built(instance):
        return instance

    # Initialize build space to track references being processed (for cycle detection)
    if build_space is None:
        build_space = set()

    # Handle lists by building each element
    if isinstance(instance, list):
        for i in range(len(instance)):
            instance[i] = deep_build(instance[i], config, build_space)
        return instance
    
    # Handle dictionaries
    elif isinstance(instance, dict):
        if 'obj' in instance and 'args' in instance:
            # This is a module instantiation spec
            obj_path = instance['obj']
            args = instance['args']
            assert isinstance(args, dict), f"Args for {obj_path} must be a dict."
            
            # Build arguments recursively
            built_args = deep_build(args, config, build_space)
            
            # Import and instantiate the class
            module_class = dynamic_import(obj_path)
            module_instance = module_class(**built_args)
            return module_instance
        else:
            # Regular dictionary - build each value
            for key in instance:
                instance[key] = deep_build(instance[key], config, build_space)
            return instance
    
    # Handle strings (potential references or paths)
    elif isinstance(instance, str):
        if '/' in instance:  # Path string might contain references
            parts = instance.split('/')
            built_parts = deep_build(parts, config, build_space)
            return '/'.join(built_parts)
        elif is_ref_type(instance):
            # Extract reference name from <name> format
            ref_name = instance[1:-1]
            
            # Check for circular references
            if ref_name in build_space:
                raise ValueError("Cross referencing is not allowed in config.")
            
            # Add to build space to track this reference
            build_space.add(ref_name)
            
            # Get and build the referenced attribute
            assert hasattr(config, ref_name), f"Key name {instance} not found in config."
            attr_value = getattr(config, ref_name)
            built_value = deep_build(attr_value, config, build_space)
            
            # Update the config with the built value
            setattr(config, ref_name, built_value)
            
            # Remove from build space as we're done with this reference
            build_space.remove(ref_name)
            return built_value
        else:
            # Regular string, return as is
            return instance
    
    # Other types (int, float, bool, None, etc.)
    else:
        return instance


def build(name: str, config: Config):
    """
    Build a named instance from the config.
    
    This is the main entry point for the builder system.
    
    Args:
        name (str): Name of the reference to build
        config (Config): Configuration object containing references
        
    Returns:
        The built instance with all references resolved
    """
    return deep_build(f"<{name}>", config)
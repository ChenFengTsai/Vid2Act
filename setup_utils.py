import os
import pathlib
import sys
import pipes
import json

os.environ["WANDB_START_METHOD"] = "thread"
# import wandb
from cuda_device_mapper import CUDADeviceMapper

sys.path.append(str(pathlib.Path(__file__).parent))

def setup_device(config):
    """Setup device mapping to use nvidia-smi GPU IDs instead of PyTorch indices"""
    if hasattr(config, 'device') and isinstance(config.device, str) and config.device.startswith('cuda:'):
        print(f"Setting up device mapping for: {config.device}")
        mapper = CUDADeviceMapper()
        mapper.print_mapping()  # Show the mapping
        device = mapper.set_device_from_config(config.device)
        config.device = device
        print(f"Device mapping complete. Using: {device}")
        return mapper
    else:
        print(f"Using device as specified: {config.device}")
        return None
    
def setup_wandb_and_logging(config, use_hierarchical=False):
    """Setup wandb integration with configurable directory structure logic"""
    
    if use_hierarchical:
        # Hierarchical structure
        base_log_path = pathlib.Path(config.logdir).expanduser()
        
        # Parse task to get env_id
        env_id = config.task.replace('dmc_', '') if config.task.startswith('dmc_') else config.task
        
        base_hierarchical_path = base_log_path / env_id / "dreamer" / getattr(config, 'expr_name', 'default') / str(config.seed)
        logdir = base_hierarchical_path
        
        # Check if hierarchical path exists and create unique one with suffix if needed
        if logdir.exists():
            # Find all existing directories with the pattern base_path_X
            parent_dir = base_hierarchical_path.parent
            base_name = base_hierarchical_path.name
            
            # Get all directories matching the pattern base_name_X where X is a number
            existing_dirs = [d for d in parent_dir.glob(f"{base_name}_*") if d.is_dir()]
            
            # Extract suffixes and find the maximum
            max_suffix = 0
            for dir_path in existing_dirs:
                suffix_str = dir_path.name[len(base_name)+1:]
                if suffix_str.isdigit():
                    max_suffix = max(max_suffix, int(suffix_str))
            
            # Create new directory with max_suffix + 1
            new_suffix = max_suffix + 1
            logdir = pathlib.Path(f"{base_hierarchical_path}_{new_suffix}")
            
            print(f"Hierarchical logdir {base_hierarchical_path} already exists. Using {logdir} instead.")
        
        print(f"Using hierarchical structure: {logdir}")
        config.logdir = str(logdir)
        
    else:
        # Original simple suffix logic
        base_logdir = pathlib.Path(config.logdir).expanduser()
        logdir = base_logdir
        
        # Check if the directory exists and create a unique one with suffix if needed
        if logdir.exists():
            # Find all existing directories with the pattern base_logdir_X
            parent_dir = base_logdir.parent
            base_name = base_logdir.name
            
            # Get all directories matching the pattern base_name_X where X is a number
            existing_dirs = [d for d in parent_dir.glob(f"{base_name}_*") if d.is_dir()]
            
            # Extract suffixes and find the maximum
            max_suffix = 0
            for dir_path in existing_dirs:
                suffix_str = dir_path.name[len(base_name)+1:]
                if suffix_str.isdigit():
                        max_suffix = max(max_suffix, int(suffix_str))
            
            # Create new directory with max_suffix + 1
            new_suffix = max_suffix + 1
            logdir = pathlib.Path(f"{base_logdir}_{new_suffix}")
            
            print(f"Logdir {base_logdir} already exists. Using {logdir} instead.")
        
        print("Logdir", logdir)
        config.logdir = str(logdir)
    
    # Convert to pathlib.Path for consistency
    logdir = pathlib.Path(config.logdir)
    
    # Setup wandb directories if wandb is enabled
    if config.use_wandb:
        # Create wandb directory structure
        wandb_dir = os.path.join(logdir, "wandb_logs")
        wandb_cache_dir = os.path.join(logdir, "wandb_cache")
        
        try:
            # Create directories
            os.makedirs(wandb_dir, exist_ok=True)
            os.makedirs(wandb_cache_dir, exist_ok=True)
            
            # Set environment variables
            os.environ["WANDB_DIR"] = wandb_dir
            os.environ["WANDB_CACHE_DIR"] = wandb_cache_dir
            
            print(f"✓ Set WANDB_DIR to: {wandb_dir}")
            print(f"✓ Set WANDB_CACHE_DIR to: {wandb_cache_dir}")
            
        except Exception as e:
            print(f"✗ Error creating wandb directories: {e}")
            print("Falling back to default wandb location")
        
        # Parse task information for wandb
        task_parts = config.task.replace('dmc_', '').split('_')
        
        # Create run name
        name_parts = [f"seed{config.seed}"]
        if hasattr(config, 'expl_behavior') and config.expl_behavior != "greedy":
            name_parts.append(f"expl_{config.expl_behavior}")
        name = "_".join(name_parts)
        
        # Create tags
        tags = [
            "dreamer",
            f"seed_{config.seed}",
            config.task.replace('_', '-'),
        ]
        
        if hasattr(config, 'expl_behavior'):
            tags.append(f"expl_{config.expl_behavior}")
        
        # Initialize wandb
        print("Initializing wandb...")
        wandb.init(
            project="dreamer-dmc",
            entity=os.environ.get("MY_WANDB_ID", None),
            name=name,
            group=config.task,
            job_type="train",
            tags=tags,
            config=vars(config),
        )
        
        print(f"✓ Wandb initialized. Run dir: {wandb.run.dir}")
    
    return logdir

def save_cmd(base_dir):
    cmd_path = os.path.join(base_dir, "cmd.txt")
    cmd = "python " + " ".join([sys.argv[0]] + [pipes.quote(s) for s in sys.argv[1:]])
    cmd += "\n"
    print("\n" + "*" * 80)
    print("Training command:\n" + cmd)
    print("*" * 80 + "\n")
    with open(cmd_path, "w") as f:
        f.write(cmd)


def save_git(base_dir):
    git_path = os.path.join(base_dir, "git.txt")
    print("Save git commit and diff to {}".format(git_path))
    cmds = [
        "echo `git rev-parse HEAD` > {}".format(git_path),
        "git diff >> {}".format(git_path),
    ]
    os.system("\n".join(cmds))

def save_config_to_json(config, logdir):
    """
    Save config to JSON, skipping non-serializable objects like types and functions.
    """
    # Create a filtered version of config for JSON serialization
    config_dict = vars(config).copy()
    
    # List of keys with non-serializable values to skip
    # Add any other problematic keys you find
    skip_keys = {
        'act',  # This is a torch.nn class like ReLU
        'device',  # Might be a device object
    }
    
    # Also filter out functions and callables
    filtered_config = {}
    for key, value in config_dict.items():
        if key in skip_keys:
            # Store the string representation instead
            filtered_config[key] = str(value)
        elif callable(value) and not isinstance(value, type):
            # Skip functions/lambdas (except classes which we handle above)
            continue
        elif isinstance(value, type):
            # Skip type objects, store their name instead
            filtered_config[key] = str(value)
        else:
            try:
                # Try to serialize - if it works, include it
                json.dumps(value)
                filtered_config[key] = value
            except (TypeError, ValueError):
                # If it fails, store string representation
                filtered_config[key] = str(value)
    
    # Save the filtered config
    with open(os.path.join(logdir, 'config.json'), 'w') as f:
        json.dump(filtered_config, f, sort_keys=True, indent=4)
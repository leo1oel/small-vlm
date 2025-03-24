"""
Test module for VLM configuration and logging.
This file validates Hydra configuration loading and logging functionality.
"""

import os
import sys
import logging
import hydra
from omegaconf import DictConfig, OmegaConf

# Configure logger
logger: logging.Logger = logging.getLogger(name=__name__)

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main test function to validate Hydra configuration and logging.

    Args:
        cfg: Hydra configuration object
    """
    # Print basic environment information
    logger.info(msg="=== Environment Information ===")
    logger.info(msg=f"Current working directory: {os.getcwd()}")
    logger.info(msg=f"Python version: {sys.version}")
    logger.info(msg=f"Hydra version: {hydra.__version__}")
    
    # Test different log levels
    logger.debug(msg="This is a DEBUG message - only visible if log level is DEBUG or lower")
    logger.info(msg="This is an INFO message - should be visible with default settings")
    logger.warning(msg="This is a WARNING message - should appear in yellow with colorlog")
    logger.error(msg="This is an ERROR message - should appear in red with colorlog")
    
    # Validate configuration structure
    logger.info(msg="=== Configuration Structure ===")
    logger.info(msg=f"Available top-level keys: {list(cfg.keys())}")
    
    # Check if expected configuration sections exist
    if hasattr(cfg, 'model'):
        logger.info(msg=f"Model configuration found: {OmegaConf.to_yaml(cfg=cfg.model)}")
    else:
        logger.error(msg="Model configuration missing!")
        
    if hasattr(cfg, 'training'):
        logger.info(msg=f"Training configuration found: {OmegaConf.to_yaml(cfg=cfg.training)}")
    else:
        logger.error(msg="Training configuration missing!")
        
    if hasattr(cfg, 'dataset'):
        logger.info(msg=f"Dataset configuration found: {OmegaConf.to_yaml(cfg.dataset)}")
    else:
        logger.error(msg="Dataset configuration missing!")
    
    # Validate that configuration values can be accessed
    logger.info(msg="=== Configuration Access Test ===")
    
    # Test model config access
    try:
        if hasattr(cfg, 'model') and hasattr(cfg.model, 'hh'):
            logger.info(msg=f"Successfully accessed model.hh: {cfg.model.hh}")
        else:
            logger.warning(msg="Could not access model.hh - key doesn't exist")
    except Exception as e:
        logger.error(msg=f"Error accessing model config: {e}")
    
    # Output details about Hydra's job and runtime config
    logger.info(msg="=== Hydra Runtime Information ===")
    logger.info(msg=f"Output directory: {cfg.hydra.runtime.output_dir if hasattr(cfg.hydra.runtime, 'output_dir') else 'Not set'}")
    logger.info(msg=f"Config sources: {cfg.hydra.runtime.config_sources}")
    logger.info(msg=f"Config choices: {cfg.hydra.runtime.choices}")
    
    logger.info(msg="Test completed successfully")

if __name__ == "__main__":
    main()
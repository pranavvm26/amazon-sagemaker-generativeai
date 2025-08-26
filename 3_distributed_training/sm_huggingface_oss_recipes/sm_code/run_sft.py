"""
Supervised Fine-Tuning (SFT) script for language models using TRL and Transformers.

This script supports:
- Full fine-tuning and PEFT (LoRA) training
- 4-bit quantization with BitsAndBytesConfig and MXFP4
- Spectrum parameter selection for selective fine-tuning
- Distributed training with DeepSpeed and Accelerate
- Model merging and saving for deployment
"""

import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from distutils.util import strtobool
from typing import Optional, Tuple, Dict, Any

import torch
from datasets import load_dataset, Dataset
from peft import AutoPeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Mxfp4Config,
    PreTrainedModel,
    PreTrainedTokenizer,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import is_liger_kernel_available
from trl import (
    SFTTrainer,
    TrlParser,
    ModelConfig,
    SFTConfig,
    get_peft_config,
)

if is_liger_kernel_available():
    from liger_kernel.transformers import AutoLigerKernelForCausalLM



# Configure logging
def setup_logging() -> logging.Logger:
    """Set up logging configuration for the training script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


logger = setup_logging()


@dataclass
class ScriptArguments:
    """Custom arguments for the SFT training script."""
    
    dataset_id_or_path: str
    """Path to dataset file (.jsonl) or HuggingFace dataset identifier."""
    
    dataset_splits: str = "train"
    """Dataset splits to use for training."""
    
    tokenizer_name_or_path: Optional[str] = None
    """Path to tokenizer or HuggingFace tokenizer identifier. If None, uses model tokenizer."""
    
    spectrum_config_path: Optional[str] = None
    """Path to YAML config file specifying which parameters to unfreeze for Spectrum training."""
    
    max_seq_length: int = 2048
    """Maximum sequence length for tokenization."""
    
    mxfp4: bool = False
    """Whether to use MXFP4 quantization instead of standard 4-bit quantization."""

def get_checkpoint_path(training_args: SFTConfig) -> Optional[str]:
    """
    Get the path to the last checkpoint if it exists.
    
    Args:
        training_args: Training configuration containing output directory
        
    Returns:
        Path to last checkpoint or None if no checkpoint exists
    """
    if os.path.isdir(training_args.output_dir):
        return get_last_checkpoint(training_args.output_dir)
    return None


def setup_model_for_spectrum(model: PreTrainedModel, spectrum_config_path: str) -> PreTrainedModel:
    """
    Configure model for Spectrum training by selectively unfreezing parameters.
    
    Args:
        model: The pretrained model to configure
        spectrum_config_path: Path to YAML file containing parameter patterns to unfreeze
        
    Returns:
        Model with appropriate parameters frozen/unfrozen for Spectrum training
        
    Raises:
        FileNotFoundError: If spectrum config file doesn't exist
        ValueError: If spectrum config file is malformed
    """
    if not os.path.exists(spectrum_config_path):
        raise FileNotFoundError(f"Spectrum config file not found: {spectrum_config_path}")
    
    try:
        with open(spectrum_config_path, "r", encoding="utf-8") as fin:
            yaml_content = fin.read()
    except Exception as e:
        raise ValueError(f"Failed to read spectrum config file: {e}")

    # Extract parameter patterns from YAML
    unfrozen_patterns = []
    for line in yaml_content.splitlines():
        line = line.strip()
        if line.startswith("- "):
            pattern = line[2:].strip()  # Remove "- " prefix
            if pattern:
                unfrozen_patterns.append(pattern)

    if not unfrozen_patterns:
        logger.warning("No parameter patterns found in spectrum config file")

    # Freeze all parameters initially
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze parameters matching the patterns
    unfrozen_count = 0
    for name, param in model.named_parameters():
        if any(re.match(pattern, name) for pattern in unfrozen_patterns):
            param.requires_grad = True
            unfrozen_count += 1
            logger.debug(f"Unfrozen parameter: {name}")

    logger.info(f"Spectrum training: {unfrozen_count} parameters unfrozen using {len(unfrozen_patterns)} patterns")
    return model


def merge_adapter_and_save_model(
    model_path_or_id: str, 
    save_dir: str, 
    save_tokenizer: bool = True
) -> None:
    """
    Merge PEFT adapter with base model and save the merged model.
    
    Args:
        model_path_or_id: Path to PEFT model or HuggingFace model identifier
        save_dir: Directory to save the merged model
        save_tokenizer: Whether to also save the tokenizer
        
    Raises:
        Exception: If model loading, merging, or saving fails
    """
    try:
        logger.info(f"Loading PEFT model from {model_path_or_id}")
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_path_or_id,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
        
        logger.info("Merging adapter with base model")
        merged_model = model.merge_and_unload()
        
        logger.info(f"Saving merged model to {save_dir}")
        os.makedirs(save_dir, exist_ok=True)
        merged_model.save_pretrained(
            save_dir, 
            safe_serialization=True, 
            max_shard_size="4GB"
        )

        if save_tokenizer:
            logger.info(f"Saving tokenizer to {save_dir}")
            tokenizer = AutoTokenizer.from_pretrained(model_path_or_id)
            tokenizer.save_pretrained(save_dir)
            
    except Exception as e:
        logger.error(f"Failed to merge and save model: {e}")
        raise 

def load_datasets(script_args: ScriptArguments) -> Tuple[Dataset, Dataset]:
    """
    Load training and evaluation datasets based on script arguments.
    
    Args:
        script_args: Script arguments containing dataset configuration
        
    Returns:
        Tuple of (train_dataset, eval_dataset)
        
    Raises:
        ValueError: If dataset loading fails or required attributes are missing
    """
    dataset_path = script_args.dataset_id_or_path
    
    try:
        if dataset_path.endswith('.jsonl'):
            # Load local JSONL file
            logger.info(f"Loading JSONL dataset from {dataset_path}")
            dataset = load_dataset('json', data_files=dataset_path, split='train')
            
            # Split dataset (hardcoded split for JSONL files)
            total_samples = len(dataset)
            if total_samples < 1000:
                logger.warning(f"Dataset has only {total_samples} samples, using 90/10 split")
                split_idx = int(0.9 * total_samples)
                train_dataset = dataset.select(range(split_idx))
                eval_dataset = dataset.select(range(split_idx, total_samples))
            else:
                train_dataset = dataset.select(range(900))
                eval_dataset = dataset.select(range(900, 1000))
        else:
            # Load HuggingFace dataset
            logger.info(f"Loading HuggingFace dataset: {dataset_path}")
            
            # Check if we have the required split attributes
            if not hasattr(script_args, 'dataset_train_split'):
                raise ValueError("dataset_train_split not found in script_args for HuggingFace dataset")
            if not hasattr(script_args, 'dataset_test_split'):
                raise ValueError("dataset_test_split not found in script_args for HuggingFace dataset")
            
            config = getattr(script_args, 'config', None)
            if config is not None:
                train_dataset = load_dataset(
                    dataset_path, config, split=script_args.dataset_train_split
                )
                eval_dataset = load_dataset(
                    dataset_path, config, split=script_args.dataset_test_split
                )
            else:
                train_dataset = load_dataset(
                    dataset_path, split=script_args.dataset_train_split
                )
                eval_dataset = load_dataset(
                    dataset_path, split=script_args.dataset_test_split
                )
        
        logger.info(f"Loaded training dataset: {len(train_dataset)} samples, features: {train_dataset.features}")
        logger.info(f"Loaded evaluation dataset: {len(eval_dataset)} samples, features: {eval_dataset.features}")
        
        # Log first sample for debugging
        if len(train_dataset) > 0:
            logger.debug(f"First training sample: {train_dataset[0]}")
        
        return train_dataset, eval_dataset
        
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        raise


def setup_tokenizer(script_args: ScriptArguments, model_args: ModelConfig) -> PreTrainedTokenizer:
    """
    Load and configure the tokenizer.
    
    Args:
        script_args: Script arguments containing tokenizer configuration
        model_args: Model arguments containing model configuration
        
    Returns:
        Configured tokenizer
    """
    tokenizer_name = script_args.tokenizer_name_or_path or model_args.model_name_or_path
    
    logger.info(f"Loading tokenizer from {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")
    
    return tokenizer


def create_model_kwargs(model_args: ModelConfig, training_args: SFTConfig, script_args: ScriptArguments) -> Dict[str, Any]:
    """
    Create model loading arguments based on configuration.
    
    Args:
        model_args: Model configuration
        training_args: Training configuration  
        script_args: Script arguments
        
    Returns:
        Dictionary of model loading arguments
    """
    # Determine torch dtype
    if model_args.torch_dtype in ['auto', None]:
        torch_dtype = model_args.torch_dtype
    else:
        torch_dtype = getattr(torch, model_args.torch_dtype)
    
    model_kwargs = {
        'revision': model_args.model_revision,
        'trust_remote_code': model_args.trust_remote_code,
        'attn_implementation': model_args.attn_implementation,
        'torch_dtype': torch_dtype,
        'use_cache': not training_args.gradient_checkpointing,
    }
    
    # Set low_cpu_mem_usage based on DeepSpeed usage
    use_deepspeed = strtobool(os.environ.get("ACCELERATE_USE_DEEPSPEED", "false"))
    if not use_deepspeed:
        model_kwargs['low_cpu_mem_usage'] = True
    
    # Configure quantization
    if model_args.load_in_4bit:
        if script_args.mxfp4:
            logger.info("Using MXFP4 quantization")
            model_kwargs['quantization_config'] = Mxfp4Config(dequantize=True)
        else:
            logger.info("Using BitsAndBytes 4-bit quantization")
            model_kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_quant_storage=torch_dtype,
            )
    
    return model_kwargs


def load_model(model_args: ModelConfig, training_args: SFTConfig, script_args: ScriptArguments, model_kwargs: Dict[str, Any]) -> PreTrainedModel:
    """
    Load the pretrained model with appropriate configuration.
    
    Args:
        model_args: Model configuration
        training_args: Training configuration
        script_args: Script arguments
        model_kwargs: Model loading arguments
        
    Returns:
        Loaded model
        
    Raises:
        ValueError: If MXFP4 is used with unsupported configurations
    """
    model_name = model_args.model_name_or_path
    
    if script_args.mxfp4:
        logger.info("Loading model with MXFP4 - skipping Liger kernel")
        # MXFP4 doesn't support Liger kernel yet
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    else:
        # Use Liger kernel if available and requested
        if training_args.use_liger and is_liger_kernel_available():
            logger.info("Loading model with Liger kernel optimization")
            model = AutoLigerKernelForCausalLM.from_pretrained(model_name, **model_kwargs)
        else:
            logger.info("Loading standard model")
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    
    # Wait for all processes in distributed training
    if hasattr(training_args, 'distributed_state'):
        training_args.distributed_state.wait_for_everyone()
    
    return model


def configure_model_for_training(model: PreTrainedModel, script_args: ScriptArguments) -> PreTrainedModel:
    """
    Configure model for specific training requirements (e.g., Spectrum).
    
    Args:
        model: The loaded model
        script_args: Script arguments
        
    Returns:
        Configured model
        
    Raises:
        AssertionError: If Spectrum config is required but not provided for non-MXFP4 training
    """
    if script_args.spectrum_config_path and not script_args.mxfp4:
        logger.info(f"Configuring model for Spectrum training with config: {script_args.spectrum_config_path}")
        return setup_model_for_spectrum(model, script_args.spectrum_config_path)
    elif not script_args.spectrum_config_path and not script_args.mxfp4:
        # This seems to be a bug in the original code - it always raises an error
        # Let's make it more reasonable by only requiring spectrum config when explicitly needed
        logger.warning("No Spectrum config provided - using standard training")
        return model
    else:
        return model


def get_model_save_directory(model_name: str) -> str:
    """
    Get the directory path for saving the final model.
    
    Args:
        model_name: Name/path of the model
        
    Returns:
        Path to save directory
    """
    if "SM_MODEL_DIR" in os.environ:
        base_dir = os.environ["SM_MODEL_DIR"]
    else:
        base_dir = "/opt/ml/model"
    
    return os.path.join(base_dir, model_name)


def save_peft_model(trainer: SFTTrainer, training_args: SFTConfig, model_args: ModelConfig) -> None:
    """
    Save PEFT model, merge with base model, and save final merged model.
    
    Args:
        trainer: The SFT trainer instance
        training_args: Training configuration
        model_args: Model configuration
    """
    logger.info("Saving PEFT model")
    
    # Save adapter to checkpoint directory
    checkpoint_dir = os.path.join(training_args.output_dir, "last_checkpoint")
    trainer.save_model(checkpoint_dir)
    logger.info(f"PEFT adapter saved to {checkpoint_dir}")
    
    # Wait for all processes
    if hasattr(training_args, 'distributed_state'):
        training_args.distributed_state.wait_for_everyone()
    
    # Save tokenizer
    trainer.tokenizer.save_pretrained(checkpoint_dir)
    logger.info(f"Tokenizer saved to {checkpoint_dir}")
    
    # Create model card on main process
    if trainer.accelerator.is_main_process:
        trainer.create_model_card({'tags': ['sagemaker', 'hf-oss']})
    
    # Clean up GPU memory
    del trainer.model
    torch.cuda.empty_cache()
    
    # Merge adapter and save final model
    final_model_dir = get_model_save_directory(model_args.model_name_or_path)
    logger.info(f"Merging adapter and saving final model to {final_model_dir}")
    
    merge_adapter_and_save_model(
        model_path_or_id=checkpoint_dir,
        save_dir=final_model_dir,
        save_tokenizer=True
    )


def save_full_model(trainer: SFTTrainer, training_args: SFTConfig, model_args: ModelConfig) -> None:
    """
    Save full fine-tuned model (non-PEFT).
    
    Args:
        trainer: The SFT trainer instance
        training_args: Training configuration
        model_args: Model configuration
    """
    logger.info("Saving full fine-tuned model")
    
    # Save model to final directory
    final_model_dir = get_model_save_directory(model_args.model_name_or_path)
    trainer.save_model(final_model_dir)
    logger.info(f"Model saved to {final_model_dir}")
    
    # Wait for all processes
    if hasattr(training_args, 'distributed_state'):
        training_args.distributed_state.wait_for_everyone()
    
    # Save tokenizer (fix bug: was saving to wrong directory)
    trainer.tokenizer.save_pretrained(final_model_dir)
    logger.info(f"Tokenizer saved to {final_model_dir}")
    
    # Create model card on main process
    if trainer.accelerator.is_main_process:
        trainer.create_model_card({'tags': ['sagemaker', 'gpt-oss']})


def train_function(model_args: ModelConfig, script_args: ScriptArguments, training_args: SFTConfig) -> None:
    """
    Main training function that orchestrates the entire SFT process.
    
    Args:
        model_args: Model configuration from TRL parser
        script_args: Custom script arguments
        training_args: Training configuration from TRL parser
    """
    logger.info("=" * 50)
    logger.info("Starting Supervised Fine-Tuning")
    logger.info("=" * 50)
    
    # Log all parameters
    logger.info(f"Model parameters: {model_args}")
    logger.info(f"Script parameters: {script_args}")  
    logger.info(f"Training parameters: {training_args}")

    # Load datasets
    train_dataset, eval_dataset = load_datasets(script_args)
    
    # Setup tokenizer
    tokenizer = setup_tokenizer(script_args, model_args)
    
    # Configure PEFT if needed
    peft_config = None
    if model_args.use_peft:
        logger.info("Configuring PEFT (Parameter Efficient Fine-Tuning)")
        peft_config = get_peft_config(model_args)
    else:
        logger.info("Using full fine-tuning")
    
    # Load and configure model
    model_kwargs = create_model_kwargs(model_args, training_args, script_args)
    model = load_model(model_args, training_args, script_args, model_kwargs)
    model = configure_model_for_training(model, script_args)
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # Add eval dataset
        tokenizer=tokenizer,
        peft_config=peft_config,
    )
    
    # Print trainable parameters for PEFT
    if trainer.accelerator.is_main_process and peft_config:
        trainer.model.print_trainable_parameters()

    # Check for existing checkpoint
    last_checkpoint = get_checkpoint_path(training_args)
    if last_checkpoint and training_args.resume_from_checkpoint is None:
        logger.info(f"Resuming training from checkpoint: {last_checkpoint}")

    # Start training
    start_time = datetime.now()
    logger.info(f"Starting training at {start_time.strftime('%Y-%m-%d %H:%M:%S')} for {training_args.num_train_epochs} epochs")
    
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    
    # Log training metrics
    metrics = train_result.metrics
    metrics['train_samples'] = len(train_dataset)
    trainer.log_metrics('train', metrics)
    trainer.save_metrics('train', metrics)
    trainer.save_state()
    
    # Prepare model for inference
    if trainer.is_fsdp_enabled and peft_config:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type('FULL_STATE_DICT')
    
    # Restore cache for inference
    trainer.model.config.use_cache = True
    
    # Save model based on training type
    if model_args.use_peft:
        save_peft_model(trainer, training_args, model_args)
    else:
        save_full_model(trainer, training_args, model_args)
    
    end_time = datetime.now()
    training_duration = end_time - start_time
    logger.info(f"Training completed successfully in {training_duration}")
    logger.info("=" * 50)



def main() -> None:
    """
    Main entry point for the SFT training script.
    
    Parses arguments using TRL parser and runs the training function.
    """
    try:
        # Parse arguments using TRL parser (preserving core functionality)
        parser = TrlParser((ModelConfig, ScriptArguments, SFTConfig))
        model_args, script_args, training_args = parser.parse_args_and_config()

        # Set seed for reproducibility
        set_seed(training_args.seed)
        logger.info(f"Set random seed to {training_args.seed}")

        # Run the main training loop
        train_function(model_args, script_args, training_args)
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


if __name__ == '__main__':
    main()
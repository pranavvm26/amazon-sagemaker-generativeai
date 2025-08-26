from dataclasses import dataclass
from datetime import datetime
from distutils.util import strtobool
import logging
import os
import re
from typing import Optional
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    set_seed, 
    BitsAndBytesConfig, 
    Mxfp4Config
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import is_liger_kernel_available
from trl import (
    SFTTrainer, 
    TrlParser, 
    ModelConfig, 
    SFTConfig, 
    get_peft_config
)
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM

if is_liger_kernel_available():
    from liger_kernel.transformers import AutoLigerKernelForCausalLM



########################
# Custom dataclasses
########################
@dataclass
class ScriptArguments:
    dataset_id_or_path: str
    dataset_splits: str = "train"
    tokenizer_name_or_path: str = None
    spectrum_config_path: Optional[str] = None
    max_seq_length: int = 2048
    mxfp4: bool = False


########################
# Setup logging
########################
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

########################
# Helper functions
########################

def get_checkpoint(training_args: SFTConfig):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint


def setup_model_for_spectrum(model, spectrum_config_path):
    unfrozen_parameters = []
    with open(spectrum_config_path, "r") as fin:
        yaml_parameters = fin.read()

    # get the unfrozen parameters from the yaml file
    for line in yaml_parameters.splitlines():
        if line.startswith("- "):
            unfrozen_parameters.append(line.split("- ")[1])

    # freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    # unfreeze Spectrum parameters
    for name, param in model.named_parameters():
        if any(re.match(unfrozen_param, name) for unfrozen_param in unfrozen_parameters):
            param.requires_grad = True
    
    # COMMENT IN: for sanity check print the trainable parameters
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(f"Trainable parameter: {name}")      
            
    return model

def merge_adapter_and_save_model(model_path_or_id, save_dir, save_tokenizer=True):
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_path_or_id,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )  
    # Merge LoRA and base model and save
    model = model.merge_and_unload()        
    model.save_pretrained(save_dir, safe_serialization=True, max_shard_size="4GB")

    # save tokenizer
    if save_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(model_path_or_id)
        tokenizer.save_pretrained(save_dir) 

###########################################################################################################

def train_function(model_args: ModelConfig, script_args: ScriptArguments, training_args: SFTConfig):
    """Main training function."""
    #########################
    # Log parameters
    #########################
    logger.info(f'Model parameters {model_args}')
    logger.info(f'Script parameters {script_args}')
    logger.info(f'Training/evaluation parameters {training_args}')

    ###############
    # Load datasets
    ###############
    if script_args.dataset_id_or_path.endswith('.jsonl'):
        dataset = load_dataset('json', data_files=script_args.dataset_id_or_path, split='train')
        train_dataset = dataset.select(range(900))
        test_dataset = dataset.select(range(900, 1000))
    else:
        if script_args.config is not None:
            train_dataset = load_dataset(script_args.dataset_id_or_path, script_args.config, split=script_args.dataset_train_split)
            test_dataset = load_dataset(script_args.dataset_id_or_path, script_args.config, split=script_args.dataset_test_split)
        else:
            train_dataset = load_dataset(script_args.dataset_id_or_path, split=script_args.dataset_train_split)
            test_dataset = load_dataset(script_args.dataset_id_or_path, split=script_args.dataset_test_split)
    
    logger.info(f'Loaded training dataset with {len(train_dataset)} samples and the following features: {train_dataset.features}')
    logger.info(f'Loaded test/eval dataset with {len(test_dataset)} samples and the following features: {test_dataset.features}')

    logger.info(train_dataset[0])
    
    ################
    # Load tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.tokenizer_name_or_path if script_args.tokenizer_name_or_path else model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token
    # if we use peft we need to make sure we use a chat template that is not using special tokens as by default embedding layers will not be trainable 
    
    
    #######################
    # Load pretrained model
    #######################
    
    # define model kwargs
    model_kwargs = dict(
        revision=model_args.model_revision, # What revision from Huggingface to use, defaults to main
        trust_remote_code=model_args.trust_remote_code, # Whether to trust the remote code, this also you to fine-tune custom architectures
        attn_implementation=model_args.attn_implementation, # What attention implementation to use, defaults to flash_attention_2
        torch_dtype=model_args.torch_dtype if model_args.torch_dtype in ['auto', None] else getattr(torch, model_args.torch_dtype), # What torch dtype to use, defaults to auto
        use_cache=False if training_args.gradient_checkpointing else True, # Whether
        low_cpu_mem_usage=True if not strtobool(os.environ.get("ACCELERATE_USE_DEEPSPEED", "false")) else None,  # Reduces memory usage on CPU for loading the model
    )
    
    # Check which training method to use and if 4-bit quantization is needed
    # without mxfp4
    if model_args.load_in_4bit and not script_args.mxfp4: 
        model_kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=model_kwargs['torch_dtype'],
            bnb_4bit_quant_storage=model_kwargs['torch_dtype'],
        )
    # with mxfp4
    elif model_args.load_in_4bit and script_args.mxfp4: 
        model_kwargs['quantization_config'] = Mxfp4Config(dequantize=True)

    if model_args.use_peft:
        logger.info("\n=======================\nfine-tuning using peft\n=======================\n")
        peft_config = get_peft_config(model_args)
    else:
        logger.info("\n=======================\nfull fine-tuning\n=======================\n")
        peft_config = None
    
    # load the model with our kwargs
    if not script_args.mxfp4:
        if training_args.use_liger:
            model = AutoLigerKernelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
        training_args.distributed_state.wait_for_everyone()  # wait for all processes to load
    else:
        logger.info(f"skipping AutoLigerKernelForCausalLM for MXFP4 configuration!")


    if script_args.spectrum_config_path and not script_args.mxfp4:
        model = setup_model_for_spectrum(model, script_args.spectrum_config_path)
    else:
        logger.error("spectrum config for supported for GPT-OSS model family!")
        raise AssertionError("spectrum config for supported for GPT-OSS model family!")

    ########################
    # Initialize the Trainer
    ########################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )
    if trainer.accelerator.is_main_process and peft_config:
        trainer.model.print_trainable_parameters()

    ###############
    # Training loop
    ###############
    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f'Checkpoint detected, resuming training at {last_checkpoint}.')

    logger.info(f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for {training_args.num_train_epochs} epochs***')
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    # log metrics
    metrics = train_result.metrics
    metrics['train_samples'] = len(train_dataset)
    trainer.log_metrics('train', metrics)
    trainer.save_metrics('train', metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    
    logger.info('*** Save model ***')
    if trainer.is_fsdp_enabled and peft_config:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type('FULL_STATE_DICT')
    
    # Restore k,v cache for fast inference
    trainer.model.config.use_cache = True

    #######################
    # Peft Model Save Path
    #######################
    if model_args.use_peft:
        logger.info("saving peft model")

        # save model and tokenizer
        last_checkpoint_output_dir = os.path.join(training_args.output_dir, "last_checkpoint")

        trainer.save_model(last_checkpoint_output_dir)
        logger.info(f'Model saved to {last_checkpoint_output_dir}')
        
        # wait for all processes to load
        training_args.distributed_state.wait_for_everyone() 

        tokenizer.save_pretrained(last_checkpoint_output_dir)
        logger.info(f'Tokenizer saved to {last_checkpoint_output_dir}')

        # Save everything else on main process
        if trainer.accelerator.is_main_process:
            trainer.create_model_card({'tags': ['sagemaker', 'hf-oss']})

        del model 
        torch.cuda.empty_cache()

        # save final model to model output dir
        model_save_dir = os.path.join(os.environ["SM_MODEL_DIR"], model_args.model_name_or_path) if os.environ["SM_MODEL_DIR"] else os.path.join("/opt/ml/model", model_args.model_name_or_path)
        logger.info(f'saving final model to {model_save_dir}')
        
        # merge and save
        merge_adapter_and_save_model(
            model_path_or_id=last_checkpoint_output_dir, 
            save_dir=model_save_dir, 
            save_tokenizer=True
        )
    
    #######################
    # Full model save path
    #######################
    else:
        logger.info("saving full model")

        # save final model to model output dir
        model_save_dir = os.path.join(os.environ["SM_MODEL_DIR"], model_args.model_name_or_path) if os.environ["SM_MODEL_DIR"] else os.path.join("/opt/ml/model", model_args.model_name_or_path)
        logger.info(f'saving final model to {model_save_dir}')

        trainer.save_model(model_save_dir)
        logger.info(f'Model saved to {model_save_dir}')
        
        # wait for all processes to load
        training_args.distributed_state.wait_for_everyone() 

        tokenizer.save_pretrained(last_checkpoint_output_dir)
        logger.info(f'Tokenizer saved to {last_checkpoint_output_dir}')

    # Save everything else on main process
    if trainer.accelerator.is_main_process:
        trainer.create_model_card({'tags': ['sagemaker', 'gpt-oss']})

    logger.info('Training complete!')


def main():
    parser = TrlParser((ModelConfig, ScriptArguments, SFTConfig))
    model_args, script_args, training_args = parser.parse_args_and_config()

    # Set seed for reproducibility
    set_seed(training_args.seed)

    # Run the main training loop
    train_function(model_args, script_args, training_args)


if __name__ == '__main__':
    main()
# SageMaker HuggingFace OSS Recipes

A comprehensive collection of training recipes for fine-tuning foundation models on Amazon SageMaker using HuggingFace's open-source libraries. This repository provides production-ready configurations for various model families and training methodologies.

## Overview

This repository contains a unified Supervised Fine-Tuning (SFT) framework that supports multiple training approaches:

- **LoRA (Low-Rank Adaptation)**: Parameter-efficient fine-tuning using PEFT
- **Spectrum Training**: Selective parameter unfreezing for targeted fine-tuning
- **Full Fine-tuning**: Traditional full model parameter training
- **Quantization Support**: 4-bit quantization with BitsAndBytes and MXFP4
- **Distributed Training**: Multi-GPU training with DeepSpeed and Accelerate

## Key Features

### Training Capabilities
- **Multiple Training Methods**: LoRA, Spectrum, and Full fine-tuning
- **Advanced Quantization**: Support for 4-bit quantization (BitsAndBytes, MXFP4)
- **Distributed Training**: Built-in support for multi-GPU and multi-node training
- **Memory Optimization**: Gradient checkpointing, Flash Attention 2, Liger Kernel
- **Flexible Data Loading**: Support for JSONL files and HuggingFace datasets
- **Checkpoint Management**: Automatic checkpoint saving and resumption

### Production Features
- **SageMaker Integration**: Optimized for SageMaker Training Jobs
- **Comprehensive Logging**: TensorBoard integration and detailed metrics
- **Model Deployment**: Automatic model saving for inference deployment
- **Recipe-based Configuration**: YAML-based configuration management

## Supported Models

### Text Generation Models

| Model | LoRA | Spectrum | Full | Notes |
|-------|------|----------|------|-------|
| **meta-llama/Meta-Llama-3-8B-Instruct** | ✅ [QLoRA](sm_code/oss-recipes/meta-llama--Meta-Llama-3-8B-Instruct-Vanilla-QLoRA.yaml) | ✅ [Spectrum](sm_code/oss-recipes/meta-llama--Meta-Llama-3-8B-Instruct-Vanilla-Spectrum.yaml) | ✅ [Full](sm_code/oss-recipes/meta-llama--Meta-Llama-3-8B-Instruct-Vanilla-Full.yaml) | Flash Attention 2, 4-bit quantization |
| **mistralai/Mistral-7B-Instruct-v0.3** | ✅ [QLoRA](sm_code/oss-recipes/mistralai--Mistral-7B-Instruct-v0.3-Vanilla-QLoRA.yaml) | ✅ [Spectrum](sm_code/oss-recipes/mistralai--Mistral-7B-Instruct-v0.3-Vanilla-Spectrum.yaml) | ✅ [Full](sm_code/oss-recipes/mistralai--Mistral-7B-Instruct-v0.3-Vanilla-Full.yaml) | Flash Attention 2, optimized for efficiency |
| **openai/gpt-oss-20b** | ✅ [QLoRA](sm_code/oss-recipes/openai--gpt-oss-20b-Vanilla-QLoRA.yaml) | ⏳ Coming Soon | ⏳ Coming Soon | MXFP4 quantization, Flash Attention 3 |

### Multi-Modal Models
*Coming Soon* - Support for vision-language models and multi-modal architectures

## Quick Start

### 1. Basic Usage

```bash
# Run with a recipe configuration
python sm_code/run_sft.py --config sm_code/oss-recipes/meta-llama--Meta-Llama-3-8B-Instruct-Vanilla-QLoRA.yaml

# Override specific parameters
python sm_code/run_sft.py \
    --config sm_code/oss-recipes/meta-llama--Meta-Llama-3-8B-Instruct-Vanilla-QLoRA.yaml \
    --num_train_epochs 3 \
    --learning_rate 1e-4
```

### 2. SageMaker Training Job

```python
from sagemaker.pytorch import PyTorch

# Configure SageMaker estimator
estimator = PyTorch(
    entry_point='run_sft.py',
    source_dir='sm_code',
    role=role,
    instance_type='ml.g5.2xlarge',
    instance_count=1,
    framework_version='2.0.0',
    py_version='py310',
    hyperparameters={
        'config': 'oss-recipes/meta-llama--Meta-Llama-3-8B-Instruct-Vanilla-QLoRA.yaml'
    }
)

# Start training
estimator.fit({'training': 's3://your-bucket/training-data/'})
```

## Training Methods

### LoRA (Low-Rank Adaptation)
Parameter-efficient fine-tuning that adds trainable low-rank matrices to existing layers.

**Benefits:**
- Significantly reduced memory usage
- Faster training times
- Easy to merge and deploy
- Maintains base model performance

**Configuration:**
```yaml
use_peft: true
load_in_4bit: true
lora_target_modules: "all-linear"
lora_r: 16
lora_alpha: 16
```

### Spectrum Training
Selective parameter unfreezing based on configurable patterns for targeted fine-tuning.

**Benefits:**
- Fine-grained control over parameter updates
- Balanced approach between efficiency and performance
- Customizable parameter selection

**Configuration:**
```yaml
spectrum_config_path: path/to/spectrum_config.yaml
```

### Full Fine-tuning
Traditional approach that updates all model parameters.

**Benefits:**
- Maximum model adaptation capability
- Best performance for domain-specific tasks
- Complete model customization

**Configuration:**
```yaml
use_peft: false
```

## Recipe Structure

Each recipe is a YAML configuration file containing:

```yaml
# Model Configuration
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
torch_dtype: bfloat16
attn_implementation: flash_attention_2

# Dataset Configuration
dataset_id_or_path: /path/to/dataset.jsonl
max_seq_length: 4096
packing: true

# Training Configuration
num_train_epochs: 1
per_device_train_batch_size: 8
gradient_accumulation_steps: 2
learning_rate: 2.0e-4

# Method-specific Configuration
use_peft: true  # For LoRA
load_in_4bit: true  # For quantization
```

## Advanced Features

### Quantization Options

**4-bit BitsAndBytes (Default)**
```yaml
load_in_4bit: true
mxfp4: false
```

**MXFP4 Quantization**
```yaml
load_in_4bit: true
mxfp4: true
```

### Memory Optimizations

**Gradient Checkpointing**
```yaml
gradient_checkpointing: true
gradient_checkpointing_kwargs:
  use_reentrant: true
```

**Flash Attention**
```yaml
attn_implementation: flash_attention_2
```

**Liger Kernel**
```yaml
use_liger: true
```

## Data Format

### JSONL Format
```json
{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]}
{"messages": [{"role": "user", "content": "How are you?"}, {"role": "assistant", "content": "I'm doing well!"}]}
```

### HuggingFace Dataset
```yaml
dataset_id_or_path: "HuggingFaceH4/ultrachat_200k"
dataset_train_split: "train_sft"
dataset_test_split: "test_sft"
```

## Performance Optimization

### Memory Usage Guidelines

| Model Size | Recommended Instance | Training Method | Batch Size |
|------------|---------------------|-----------------|------------|
| 7B | ml.g5.2xlarge | LoRA + 4-bit | 8 |
| 8B | ml.g5.4xlarge | LoRA + 4-bit | 4 |
| 20B | ml.g5.12xlarge | LoRA + MXFP4 | 2 |

### Training Speed Tips

1. **Use Flash Attention 2**: Reduces memory and increases speed
2. **Enable Gradient Checkpointing**: Trades compute for memory
3. **Optimize Batch Size**: Balance between memory usage and convergence
4. **Use Mixed Precision**: Enable bf16 for better performance

## Monitoring and Logging

### TensorBoard Integration
```yaml
report_to:
  - tensorboard
logging_steps: 5
```

### Metrics Tracking
- Training loss
- Learning rate schedule
- GPU memory usage
- Training throughput

## Troubleshooting

### Common Issues

**Out of Memory (OOM)**
- Reduce `per_device_train_batch_size`
- Enable `gradient_checkpointing`
- Use 4-bit quantization
- Reduce `max_seq_length`

**Slow Training**
- Increase `gradient_accumulation_steps`
- Enable Flash Attention 2
- Use Liger Kernel optimizations
- Optimize data loading

**Model Quality Issues**
- Adjust learning rate
- Increase training epochs
- Check data quality and format
- Experiment with different LoRA ranks

## Contributing

We welcome contributions! Please:

1. Add new model recipes following the naming convention
2. Test configurations thoroughly
3. Update documentation and model support tables
4. Follow the existing YAML structure

## License

This project is licensed under the MIT License - see the [LICENSE](../../LICENSE) file for details.

## Support

For issues and questions:
- Check the troubleshooting section
- Review SageMaker documentation
- Open an issue in this repository

---

**Note**: This framework is optimized for Amazon SageMaker but can be adapted for other distributed training environments.
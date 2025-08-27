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

| Model | LoRA | Spectrum | Full | Notes |
|-------|------|----------|------|-------|
| **Meta (Llama) - Text Generation** |
| **meta-llama/Meta-Llama-3-8B-Instruct** | ✅ [QLoRA](sm_code/oss-recipes/meta-llama--Meta-Llama-3-8B-Instruct-Vanilla-QLoRA.yaml) | ✅ [Spectrum](sm_code/oss-recipes/meta-llama--Meta-Llama-3-8B-Instruct-Vanilla-Spectrum.yaml) | ✅ [Full](sm_code/oss-recipes/meta-llama--Meta-Llama-3-8B-Instruct-Vanilla-Full.yaml) | Flash Attention 2, 4-bit quantization |
| **Meta (Llama) - Multi-Modal** |
| **meta-llama/Llama-4-Scout-17B-16E-Instruct** | ✅ [QLoRA](sm_code/oss-recipes/meta-llama--Llama-4-Scout-17B-16E-Instruct-Vanilla-QLoRA.yaml) | ⏳ Coming Soon | ⏳ Coming Soon | Vision-language model |
| **meta-llama/Llama-4-Maverick-17B-128E-Instruct** | ✅ [QLoRA](sm_code/oss-recipes/meta-llama--Llama-4-Maverick-17B-128E-Instruct-Vanilla-QLoRA.yaml) | ⏳ Coming Soon | ⏳ Coming Soon | Advanced multimodal capabilities |
| **Mistral AI - Text Generation** |
| **mistralai/Mistral-7B-Instruct-v0.3** | ✅ [QLoRA](sm_code/oss-recipes/mistralai--Mistral-7B-Instruct-v0.3-Vanilla-QLoRA.yaml) | ✅ [Spectrum](sm_code/oss-recipes/mistralai--Mistral-7B-Instruct-v0.3-Vanilla-Spectrum.yaml) | ✅ [Full](sm_code/oss-recipes/mistralai--Mistral-7B-Instruct-v0.3-Vanilla-Full.yaml) | Flash Attention 2, optimized for efficiency |
| **OpenAI - Text Generation** |
| **openai/gpt-oss-20b** | ✅ [QLoRA](sm_code/oss-recipes/openai--gpt-oss-20b-Vanilla-QLoRA.yaml) | ⏳ Coming Soon | ⏳ Coming Soon | MXFP4 quantization, Flash Attention 3 |
| **openai/gpt-oss-120b** | ✅ [QLoRA](sm_code/oss-recipes/openai--gpt-oss-120b-Vanilla-QLoRA.yaml) | ⏳ Coming Soon | ⏳ Coming Soon | Large scale model, MXFP4 quantization |
| **Qwen (Alibaba) - Text Generation** |
| **Qwen/Qwen3-8B** | ✅ [QLoRA](sm_code/oss-recipes/Qwen--Qwen3-8B-Vanilla-QLoRA.yaml) | ✅ [Spectrum](sm_code/oss-recipes/Qwen--Qwen3-8B-Vanilla-Spectrum.yaml) | ✅ [Full](sm_code/oss-recipes/Qwen--Qwen3-8B-Vanilla-Full.yaml) | Flash Attention 2, 4-bit quantization |
| **Qwen/Qwen3-30B-A3B-Thinking-25073** | ✅ [QLoRA](sm_code/oss-recipes/Qwen--Qwen3-30B-A3B-Thinking-25073-Vanilla-QLoRA.yaml) | ✅ [Spectrum](sm_code/oss-recipes/Qwen--Qwen3-30B-A3B-Thinking-25073-Vanilla-Spectrum.yaml) | ✅ [Full](sm_code/oss-recipes/Qwen--Qwen3-30B-A3B-Thinking-25073-Vanilla-Full.yaml) | Large model, reasoning capabilities |
| **Qwen (Alibaba) - Multi-Modal** |
| **Qwen/Qwen2.5-Omni-7B** | ✅ [QLoRA](sm_code/oss-recipes/Qwen--Qwen2.5-Omni-7B-Vanilla-QLoRA.yaml) | ⏳ Coming Soon | ⏳ Coming Soon | Omni-modal capabilities |
| **DeepSeek - Text Generation** |
| **deepseek-ai/DeepSeek-V3** | ✅ [QLoRA](sm_code/oss-recipes/deepseek-ai--DeepSeek-V3-Vanilla-QLoRA.yaml) | ⏳ Coming Soon | ⏳ Coming Soon | MXFP4 quantization, large scale model |
| **deepseek-ai/DeepSeek-R1-Distill-Qwen-14B** | ✅ [QLoRA](sm_code/oss-recipes/deepseek-ai--DeepSeek-R1-Distill-Qwen-14B-Vanilla-QLoRA.yaml) | ✅ [Spectrum](sm_code/oss-recipes/deepseek-ai--DeepSeek-R1-Distill-Qwen-14B-Vanilla-Spectrum.yaml) | ✅ [Full](sm_code/oss-recipes/deepseek-ai--DeepSeek-R1-Distill-Qwen-14B-Vanilla-Full.yaml) | Reasoning model, distilled from R1 |
| **DeepSeek - Multi-Modal** |
| **deepseek-ai/Janus-Pro-7B** | ✅ [QLoRA](sm_code/oss-recipes/deepseek-ai--Janus-Pro-7B-Vanilla-QLoRA.yaml) | ⏳ Coming Soon | ⏳ Coming Soon | Vision-language model |
| **Google (Gemma) - Text Generation** |
| **google/gemma-7b-it** | ✅ [QLoRA](sm_code/oss-recipes/google--gemma-7b-it-Vanilla-QLoRA.yaml) | ✅ [Spectrum](sm_code/oss-recipes/google--gemma-7b-it-Vanilla-Spectrum.yaml) | ✅ [Full](sm_code/oss-recipes/google--gemma-7b-it-Vanilla-Full.yaml) | Flash Attention 2, instruction tuned |
| **google/gemma-3-27b-it** | ✅ [QLoRA](sm_code/oss-recipes/google--gemma-3-27b-it-Vanilla-QLoRA.yaml) | ✅ [Spectrum](sm_code/oss-recipes/google--gemma-3-27b-it-Vanilla-Spectrum.yaml) | ✅ [Full](sm_code/oss-recipes/google--gemma-3-27b-it-Vanilla-Full.yaml) | Large Gemma model, enhanced capabilities |
| **Google (Gemma) - Multi-Modal** |
| **google/gemma-3n-E4B-it** | ✅ [QLoRA](sm_code/oss-recipes/google--gemma-3n-E4B-it-Vanilla-QLoRA.yaml) | ⏳ Coming Soon | ⏳ Coming Soon | Multimodal Gemma variant |
| **Microsoft (Phi) - Text Generation** |
| **microsoft/Phi-4-reasoning** | ✅ [QLoRA](sm_code/oss-recipes/microsoft--Phi-4-reasoning-Vanilla-QLoRA.yaml) | ✅ [Spectrum](sm_code/oss-recipes/microsoft--Phi-4-reasoning-Vanilla-Spectrum.yaml) | ✅ [Full](sm_code/oss-recipes/microsoft--Phi-4-reasoning-Vanilla-Full.yaml) | Reasoning-focused model |
| **microsoft/Phi-4-mini-instruct** | ✅ [QLoRA](sm_code/oss-recipes/microsoft--Phi-4-mini-instruct-Vanilla-QLoRA.yaml) | ✅ [Spectrum](sm_code/oss-recipes/microsoft--Phi-4-mini-instruct-Vanilla-Spectrum.yaml) | ✅ [Full](sm_code/oss-recipes/microsoft--Phi-4-mini-instruct-Vanilla-Full.yaml) | Compact, efficient model |
| **Microsoft (Phi) - Multi-Modal** |
| **microsoft/Phi-4-multimodal-instruct** | ✅ [QLoRA](sm_code/oss-recipes/microsoft--Phi-4-multimodal-instruct-Vanilla-QLoRA.yaml) | ⏳ Coming Soon | ⏳ Coming Soon | Vision-language Phi model |
| **TII (Falcon) - Text Generation** |
| **tiiuae/Falcon3-7B-Instruct** | ✅ [QLoRA](sm_code/oss-recipes/tiiuae--Falcon3-7B-Instruct-Vanilla-QLoRA.yaml) | ✅ [Spectrum](sm_code/oss-recipes/tiiuae--Falcon3-7B-Instruct-Vanilla-Spectrum.yaml) | ✅ [Full](sm_code/oss-recipes/tiiuae--Falcon3-7B-Instruct-Vanilla-Full.yaml) | Latest Falcon generation |
| **tiiuae/Falcon3-10B-Instruct** | ✅ [QLoRA](sm_code/oss-recipes/tiiuae--Falcon3-10B-Instruct-Vanilla-QLoRA.yaml) | ✅ [Spectrum](sm_code/oss-recipes/tiiuae--Falcon3-10B-Instruct-Vanilla-Spectrum.yaml) | ✅ [Full](sm_code/oss-recipes/tiiuae--Falcon3-10B-Instruct-Vanilla-Full.yaml) | Enhanced Falcon model |

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

| Model Size | Recommended Instance | Training Method | Batch Size | Example Models |
|------------|---------------------|-----------------|------------|----------------|
| 7B | ml.g5.2xlarge | LoRA + 4-bit | 8 | Mistral-7B, Gemma-7B, Falcon3-7B, Qwen2.5-Omni-7B |
| 8B | ml.g5.4xlarge | LoRA + 4-bit | 4 | Llama-3-8B, Qwen3-8B |
| 10-14B | ml.g5.8xlarge | LoRA + 4-bit | 4 | Falcon3-10B, DeepSeek-R1-Distill-14B, Phi-4 |
| 17B | ml.g5.12xlarge | LoRA + 4-bit | 2 | Llama-4-Scout-17B, Llama-4-Maverick-17B |
| 20-30B | ml.g5.24xlarge | LoRA + MXFP4 | 2 | GPT-OSS-20B, Gemma-3-27B, Qwen3-30B |
| 100B+ | ml.p4d.24xlarge | LoRA + MXFP4 | 1 | DeepSeek-V3 |

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
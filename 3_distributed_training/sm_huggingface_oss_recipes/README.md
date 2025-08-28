# SageMaker HuggingFace OSS Recipes

A comprehensive collection of training recipes for fine-tuning foundation models on Amazon SageMaker using HuggingFace's open-source libraries. This repository provides production-ready configurations for various model families and training methodologies.

## Overview

This repository provides a comprehensive framework for model customization on Amazon SageMaker AI, supporting multiple training paradigms from supervised fine-tuning to preference optimization and pre-training. Built on HuggingFace's open-source ecosystem, it offers production-ready configurations for various model families and training methodologies.

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

## Model Customization on Amazon SageMaker AI

### Supervised Fine-Tuning

Supervised Fine-Tuning (SFT) is the process of adapting pre-trained foundation models to specific tasks or domains using labeled datasets. This approach leverages the rich representations learned during pre-training while specializing the model for downstream applications. Our framework supports three distinct SFT methodologies, each optimized for different resource constraints and performance requirements.

**LoRA (Low-Rank Adaptation)** represents the most resource-efficient approach, introducing trainable low-rank matrices into existing model layers while keeping the original parameters frozen. This method dramatically reduces memory requirements and training time, making it ideal for scenarios with limited computational resources or when quick experimentation is needed. LoRA is particularly effective for instruction-following tasks, domain adaptation, and scenarios where maintaining the base model's general capabilities is crucial. Choose LoRA when you need fast iteration cycles, have memory constraints, or want to create multiple specialized adapters from a single base model.

**Spectrum Training** offers a middle ground between efficiency and performance by selectively unfreezing specific parameter groups based on configurable patterns. This approach provides fine-grained control over which parts of the model to adapt, allowing practitioners to target specific layers or components that are most relevant to their task. Spectrum training is optimal when you have insights into which model components are most important for your specific use case, need better performance than LoRA but can't afford full fine-tuning, or want to experiment with different parameter selection strategies.

**Full Fine-tuning** updates all model parameters, providing maximum adaptation capability at the cost of increased computational requirements. This traditional approach offers the best performance for domain-specific tasks where significant model adaptation is required. Full fine-tuning is recommended when you have sufficient computational resources, need maximum model performance for critical applications, are working with significantly different domains from the pre-training data, or when the task requires substantial changes to the model's behavior.

The choice between these methods depends on your specific constraints: use LoRA for resource-constrained environments and rapid prototyping, Spectrum for balanced performance and efficiency with targeted adaptation, and Full fine-tuning for maximum performance when computational resources are available.

| Model | LoRA | Spectrum | Full | Notes |
|-------|------|----------|------|-------|
| | | | | |
| **🦙 Meta (Llama) - Text Generation** | | | | |
| meta-llama/Meta-Llama-3-8B-Instruct | ✅ [QLoRA](sm_code/oss-recipes/meta-llama--Meta-Llama-3-8B-Instruct-Vanilla-QLoRA.yaml) | ✅ [Spectrum](sm_code/oss-recipes/meta-llama--Meta-Llama-3-8B-Instruct-Vanilla-Spectrum.yaml) | ✅ [Full](sm_code/oss-recipes/meta-llama--Meta-Llama-3-8B-Instruct-Vanilla-Full.yaml) | Flash Attention 2, 4-bit quantization |
| | | | | |
| **🦙 Meta (Llama) - Multi-Modal** | | | | |
| meta-llama/Llama-4-Scout-17B-16E-Instruct | ✅ [QLoRA](sm_code/oss-recipes/meta-llama--Llama-4-Scout-17B-16E-Instruct-Vanilla-QLoRA.yaml) | ⏳ Coming Soon | ⏳ Coming Soon | Vision-language model |
| meta-llama/Llama-4-Maverick-17B-128E-Instruct | ✅ [QLoRA](sm_code/oss-recipes/meta-llama--Llama-4-Maverick-17B-128E-Instruct-Vanilla-QLoRA.yaml) | ⏳ Coming Soon | ⏳ Coming Soon | Advanced multimodal capabilities |
| | | | | |
| **🌟 Mistral AI - Text Generation** | | | | |
| mistralai/Mistral-7B-Instruct-v0.3 | ✅ [QLoRA](sm_code/oss-recipes/mistralai--Mistral-7B-Instruct-v0.3-Vanilla-QLoRA.yaml) | ✅ [Spectrum](sm_code/oss-recipes/mistralai--Mistral-7B-Instruct-v0.3-Vanilla-Spectrum.yaml) | ✅ [Full](sm_code/oss-recipes/mistralai--Mistral-7B-Instruct-v0.3-Vanilla-Full.yaml) | Flash Attention 2, optimized for efficiency |
| | | | | |
| **🤖 OpenAI - Text Generation** | | | | |
| openai/gpt-oss-20b | ✅ [QLoRA](sm_code/oss-recipes/openai--gpt-oss-20b-Vanilla-MXFP4.yaml) | ⏳ Coming Soon | ⏳ Coming Soon | MXFP4 quantization, Flash Attention 3 |
| openai/gpt-oss-120b | ✅ [QLoRA](sm_code/oss-recipes/openai--gpt-oss-120b-Vanilla-MXFP4.yaml) | ⏳ Coming Soon | ⏳ Coming Soon | Large scale model, MXFP4 quantization |
| | | | | |
| **🔮 Qwen (Alibaba) - Text Generation** | | | | |
| Qwen/Qwen3-8B | ✅ [QLoRA](sm_code/oss-recipes/Qwen--Qwen3-8B-Vanilla-QLoRA.yaml) | ✅ [Spectrum](sm_code/oss-recipes/Qwen--Qwen3-8B-Vanilla-Spectrum.yaml) | ✅ [Full](sm_code/oss-recipes/Qwen--Qwen3-8B-Vanilla-Full.yaml) | Flash Attention 2, 4-bit quantization |
| Qwen/Qwen3-30B-A3B-Thinking-25073 | ✅ [QLoRA](sm_code/oss-recipes/Qwen--Qwen3-30B-A3B-Thinking-25073-Vanilla-QLoRA.yaml) | ✅ [Spectrum](sm_code/oss-recipes/Qwen--Qwen3-30B-A3B-Thinking-25073-Vanilla-Spectrum.yaml) | ✅ [Full](sm_code/oss-recipes/Qwen--Qwen3-30B-A3B-Thinking-25073-Vanilla-Full.yaml) | Large model, reasoning capabilities |
| | | | | |
| **🔮 Qwen (Alibaba) - Multi-Modal** | | | | |
| Qwen/Qwen2.5-Omni-7B | ✅ [QLoRA](sm_code/oss-recipes/Qwen--Qwen2.5-Omni-7B-Vanilla-QLoRA.yaml) | ⏳ Coming Soon | ⏳ Coming Soon | Omni-modal capabilities |
| | | | | |
| **🧠 DeepSeek - Text Generation** | | | | |
| deepseek-ai/DeepSeek-V3 | ✅ [QLoRA](sm_code/oss-recipes/deepseek-ai--DeepSeek-V3-Vanilla-QLoRA.yaml) | ⏳ Coming Soon | ⏳ Coming Soon | MXFP4 quantization, large scale model |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-14B | ✅ [QLoRA](sm_code/oss-recipes/deepseek-ai--DeepSeek-R1-Distill-Qwen-14B-Vanilla-QLoRA.yaml) | ✅ [Spectrum](sm_code/oss-recipes/deepseek-ai--DeepSeek-R1-Distill-Qwen-14B-Vanilla-Spectrum.yaml) | ✅ [Full](sm_code/oss-recipes/deepseek-ai--DeepSeek-R1-Distill-Qwen-14B-Vanilla-Full.yaml) | Reasoning model, distilled from R1 |
| | | | | |
| **🧠 DeepSeek - Multi-Modal** | | | | |
| deepseek-ai/Janus-Pro-7B | ✅ [QLoRA](sm_code/oss-recipes/deepseek-ai--Janus-Pro-7B-Vanilla-QLoRA.yaml) | ⏳ Coming Soon | ⏳ Coming Soon | Vision-language model |
| | | | | |
| **💎 Google (Gemma) - Text Generation** | | | | |
| google/gemma-7b-it | ✅ [QLoRA](sm_code/oss-recipes/google--gemma-7b-it-Vanilla-QLoRA.yaml) | ✅ [Spectrum](sm_code/oss-recipes/google--gemma-7b-it-Vanilla-Spectrum.yaml) | ✅ [Full](sm_code/oss-recipes/google--gemma-7b-it-Vanilla-Full.yaml) | Flash Attention 2, instruction tuned |
| google/gemma-3-27b-it | ✅ [QLoRA](sm_code/oss-recipes/google--gemma-3-27b-it-Vanilla-QLoRA.yaml) | ✅ [Spectrum](sm_code/oss-recipes/google--gemma-3-27b-it-Vanilla-Spectrum.yaml) | ✅ [Full](sm_code/oss-recipes/google--gemma-3-27b-it-Vanilla-Full.yaml) | Large Gemma model, enhanced capabilities |
| | | | | |
| **💎 Google (Gemma) - Multi-Modal** | | | | |
| google/gemma-3n-E4B-it | ✅ [QLoRA](sm_code/oss-recipes/google--gemma-3n-E4B-it-Vanilla-QLoRA.yaml) | ⏳ Coming Soon | ⏳ Coming Soon | Multimodal Gemma variant |
| | | | | |
| **🔷 Microsoft (Phi) - Text Generation** | | | | |
| microsoft/Phi-4-reasoning | ✅ [QLoRA](sm_code/oss-recipes/microsoft--Phi-4-reasoning-Vanilla-QLoRA.yaml) | ✅ [Spectrum](sm_code/oss-recipes/microsoft--Phi-4-reasoning-Vanilla-Spectrum.yaml) | ✅ [Full](sm_code/oss-recipes/microsoft--Phi-4-reasoning-Vanilla-Full.yaml) | Reasoning-focused model |
| microsoft/Phi-4-mini-instruct | ✅ [QLoRA](sm_code/oss-recipes/microsoft--Phi-4-mini-instruct-Vanilla-QLoRA.yaml) | ✅ [Spectrum](sm_code/oss-recipes/microsoft--Phi-4-mini-instruct-Vanilla-Spectrum.yaml) | ✅ [Full](sm_code/oss-recipes/microsoft--Phi-4-mini-instruct-Vanilla-Full.yaml) | Compact, efficient model |
| | | | | |
| **🔷 Microsoft (Phi) - Multi-Modal** | | | | |
| microsoft/Phi-4-multimodal-instruct | ✅ [QLoRA](sm_code/oss-recipes/microsoft--Phi-4-multimodal-instruct-Vanilla-QLoRA.yaml) | ⏳ Coming Soon | ⏳ Coming Soon | Vision-language Phi model |
| | | | | |
| **🦅 TII (Falcon) - Text Generation** | | | | |
| tiiuae/Falcon3-7B-Instruct | ✅ [QLoRA](sm_code/oss-recipes/tiiuae--Falcon3-7B-Instruct-Vanilla-QLoRA.yaml) | ✅ [Spectrum](sm_code/oss-recipes/tiiuae--Falcon3-7B-Instruct-Vanilla-Spectrum.yaml) | ✅ [Full](sm_code/oss-recipes/tiiuae--Falcon3-7B-Instruct-Vanilla-Full.yaml) | Latest Falcon generation |
| tiiuae/Falcon3-10B-Instruct | ✅ [QLoRA](sm_code/oss-recipes/tiiuae--Falcon3-10B-Instruct-Vanilla-QLoRA.yaml) | ✅ [Spectrum](sm_code/oss-recipes/tiiuae--Falcon3-10B-Instruct-Vanilla-Spectrum.yaml) | ✅ [Full](sm_code/oss-recipes/tiiuae--Falcon3-10B-Instruct-Vanilla-Full.yaml) | Enhanced Falcon model |

### Preference Optimization

Preference Optimization represents the next frontier in model alignment, focusing on training models to generate outputs that align with human preferences and values. Unlike supervised fine-tuning which learns from demonstrations, preference optimization learns from comparative feedback, making models more helpful, harmless, and honest.

**Direct Preference Optimization (DPO)** revolutionizes the traditional RLHF pipeline by directly optimizing the model on preference data without requiring a separate reward model. This approach simplifies the training process while maintaining effectiveness, making it more stable and computationally efficient. DPO works by directly optimizing the policy to increase the likelihood of preferred responses while decreasing the likelihood of rejected ones, using a reference model to prevent over-optimization.

**Proximal Policy Optimization (PPO)** represents the traditional reinforcement learning approach to preference optimization, using a reward model trained on human preferences to guide policy updates. PPO maintains a balance between exploration and exploitation while ensuring stable training through clipped policy updates. This method excels in scenarios requiring fine-grained control over the optimization process and complex reward structures.

**Group Relative Policy Optimization (GRPO)** extends preference optimization to handle group-based preferences and multi-objective alignment. This approach is particularly valuable when dealing with diverse user groups or when optimizing for multiple, potentially conflicting objectives simultaneously. GRPO enables more nuanced preference learning that can adapt to different contexts and user populations.

These preference optimization techniques are essential for creating models that not only perform well on benchmarks but also generate outputs that users find genuinely helpful and aligned with their values and expectations.

| Model | DPO | PPO | GRPO | Notes |
|-------|-----|-----|------|-------|
| | | | | |
| **🚧 Coming Soon** | ⏳ | ⏳ | ⏳ | Preference optimization recipes in development |

### Pre-Training

Pre-training represents the foundational phase of large language model development, where models learn rich representations from vast amounts of unlabeled text data. This process creates the base knowledge and capabilities that can later be specialized through fine-tuning and alignment techniques.

**Autoregressive Language Modeling** forms the core of modern pre-training, where models learn to predict the next token in a sequence given the previous context. This seemingly simple objective enables models to develop sophisticated understanding of language, reasoning capabilities, and world knowledge. The scale of pre-training data and compute directly impacts the emergent capabilities of the resulting models.

**Distributed Pre-training** requires careful orchestration of training across multiple GPUs and nodes, involving techniques like data parallelism, model parallelism, and pipeline parallelism. Efficient pre-training also leverages advanced optimizations such as gradient accumulation, mixed precision training, and dynamic loss scaling to maximize throughput while maintaining numerical stability.

**Curriculum Learning and Data Composition** play crucial roles in pre-training effectiveness, involving strategic sequencing of training data and careful balancing of different data sources. Modern pre-training approaches also incorporate techniques like data deduplication, quality filtering, and domain-specific sampling to optimize the learning process.

Pre-training on Amazon SageMaker AI enables practitioners to create custom foundation models tailored to specific domains, languages, or use cases, providing the flexibility to build models that capture domain-specific knowledge and patterns not present in general-purpose models.

| Model | Autoregressive | Distributed | Curriculum | Notes |
|-------|---------------|-------------|------------|-------|
| | | | | |
| **🚧 Coming Soon** | ⏳ | ⏳ | ⏳ | Pre-training recipes in development |

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
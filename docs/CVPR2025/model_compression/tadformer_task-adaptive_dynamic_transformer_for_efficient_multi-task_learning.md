---
title: >-
  [Paper Note] TADFormer: Task-Adaptive Dynamic Transformer for Efficient Multi-Task Learning
description: >-
  [CVPR 2025][Model Compression][Parameter-Efficient Fine-Tuning] TADFormer proposes a parameter-efficient fine-tuning framework for multi-task learning. By dynamically extracting fine-grained task features according to the input context through a Dynamic Task Filter (DTF), combined with task-prompt conditional operations and cross-task interactions, it achieves superior accuracy on PASCAL-Context with 8.4x fewer parameters than full fine-tuning.
tags:
  - "CVPR 2025"
  - "Model Compression"
  - "Parameter-Efficient Fine-Tuning"
  - "Multi-Task Learning"
  - "Dynamic Convolution"
  - "Task Prompt"
  - "Transformer"
date: 2026-05-08
content_hash: 321010823840e725
---

# TADFormer: Task-Adaptive Dynamic Transformer for Efficient Multi-Task Learning

**Conference**: CVPR 2025  
**arXiv**: [2501.04293](https://arxiv.org/abs/2501.04293)  
**Code**: None  
**Area**: Model Compression / Multi-Task Learning  
**Keywords**: Parameter-Efficient Fine-Tuning, Multi-Task Learning, Dynamic Convolution, Task Prompt, Transformer

## TL;DR
TADFormer proposes a parameter-efficient fine-tuning framework for multi-task learning. By dynamically extracting fine-grained task features according to the input context through a Dynamic Task Filter (DTF), combined with task-prompt conditional operations and cross-task interactions, it achieves superior accuracy on PASCAL-Context with 8.4x fewer parameters than full fine-tuning.

## Background & Motivation

**Background**: As pre-trained models continue to scale, parameter-efficient fine-tuning (PEFT) has become the mainstream paradigm for downstream task adaptation. Methods such as LoRA, Adapter, and Visual Prompt Tuning have proven their effectiveness in single-task scenarios. Multi-task learning (MTL) requires handling multiple tasks simultaneously (such as semantic segmentation, depth estimation, and surface normal estimation), where training complexity scales linearly with the number of tasks, making PEFT particularly essential.

**Limitations of Prior Work**: Existing PEFT methods for MTL (such as MTLoRA, VMT-Adapter) exhibit two critical limitations: (1) they utilize static learnable parameters to extract task features without considering the contextual information of input samples, which limits their capability to capture fine-grained task features; (2) task-shared and task-specific modules are processed in parallel, lacking opportunities for cross-task interaction.

**Key Challenge**: PEFT fine-tunes only a tiny fraction of parameters, which inherently constrains the model's adaptivity to diverse tasks. Without considering input dependency, relying solely on static parameters makes it extremely difficult to effectively distinguish unique features of different tasks using very few trainable modules.

**Goal**: Design a multi-task PEFT framework that is both parameter-efficient and dynamically aware of input context, allowing the model to adaptively extract task-specific features based on different input samples.

**Key Insight**: Prior work in MTL demonstrates that the contextual information of input samples is crucial for capturing task-unique features. Inspired by this, the authors propose introducing dynamic convolution into PEFT modules, allowing convolution kernel parameters to be dynamically generated from input features, thereby enabling input-aware task adaptation.

**Core Idea**: Insert a Dynamic Task Filter (DTF) between the down-projection and up-projection layers of LoRA. The convolution parameters of DTF are dynamically generated from input features, integrated with a Task-Prompt Conditional (TPC) operator to extract task-adaptive features, achieving fine-grained input-aware task adaptation with minimal additional parameters.

## Method

### Overall Architecture
TADFormer consists of a shared encoder (Swin Transformer) and multiple task-specific decoders. The input consists of image patch tokens prepended with task prompts. In the encoder, the first $N-1$ blocks of each Transformer stage use task-shared modules (TS-Module, i.e., LoRA), while the last block uses a task-adaptive module (TA-Module, which contains the DTF). This extracts task-adaptive features through a task-prompt conditional operation (TPC) before feeding them into the DTF for input-aware fine adjustment.

### Key Designs

1. **Task-Prompt Conditional Operator (TPC)**:

    - **Function**: Decouples task-specific features from task-agnostic features, generating task-attribute-enhanced feature representations for each task.
    - **Mechanism**: Utilizes the attention matrix $A$ naturally generated in MHSA to extract attention scores $a_i \in \mathbb{R}^{H \times 1 \times N}$ (i.e., task attention maps) between the task prompt $p_i$ and all patch tokens, and uses this to weight the QKV output features: $f_i = f_{qkv} + S_{inv}(a_i \otimes \hat{f}_{qkv})$. In this way, features in high-attention regions (regions closely related to the task) are enhanced.
    - **Design Motivation**: Avoids introducing extra computation by directly reusing the prompt-to-patch attention from the Transformer's self-attention, naturally acquiring task-spatial relationships without increasing parameters or computational overhead.

2. **Dynamic Task Filter (DTF)**:

    - **Function**: Dynamically generates convolution parameters based on the input context to achieve input-aware task feature extraction.
    - **Mechanism**: Inserts a DTF between the down-projection and up-projection of LoRA. The low-rank features after down-projection first go through global average pooling (GAP) and are then fed into a parameter generation network $\phi(\cdot)$ to generate channel-wise convolution kernels $\theta_i = \phi(f_i W_{down})$. The convolution kernels perform channel-wise convolution operations on the down-projection features: $\tilde{F}_i = \Phi(f_i) + (\theta_i \odot (f_i W_{down})) W_{up}$. The parameter size is only $r \times r \times k^2$, which is highly lightweight. FilterNorm is used to stabilize training.
    - **Design Motivation**: Static LoRA parameters use the same projection across all inputs, failing to distinguish task features for different samples. DTF allows each sample to have unique convolution parameters, making fine-grained feature extraction possible. Grad-CAM visualization verifies that DTF indeed captures finer input context.

3. **Stage-wise Gating and Skip Connection**:

    - **Function**: Directly passes the task-adaptive features extracted by the TPC operator to the decoder, enhancing the effectiveness of task prompts.
    - **Mechanism**: Through a learnable gating parameter $g$ (Sigmoid activation, initialized to zero), the TPC features $f_i$ and the final block output $\hat{f_i}$ are weight-fused: $F_i = \sigma(g) \cdot f_i + (1 - \sigma(g)) \cdot \hat{f_i}$, and then fed into the task-specific decoder.
    - **Design Motivation**: Skip connections allow task prompts to directly affect decoder inputs through TPC operations, which is more effective than propagation purely within the encoder. Gating parameters allow the model to automatically learn the optimal blending ratio of the two features.

### Loss & Training
Multi-task weighted loss: $L_{MTL} = \sum_i^T w_i \times L_i$. Cross-entropy is used for semantic segmentation and human parts segmentation, L1 loss for normal estimation, and balanced cross-entropy for saliency detection. Only the LoRA modules, DTF, task prompts, prompt upsampling modules, layer normalization, and positional encoding are fine-tuned. The patch merging module is not fine-tuned (as prompt upsampling is sufficient).

## Key Experimental Results

### Main Results
PASCAL-Context dataset, Swin-T (ImageNet-1k) as the encoder backbone:

| Method | SemSeg (mIoU↑) | Parts (mIoU↑) | Saliency (mIoU↑) | Normals (rmse↓) | Δm(%)↑ | Params (M) |
|------|---------------|--------------|-----------------|----------------|--------|--------|
| Single-task Full FT | 67.21 | 61.93 | 62.35 | 17.97 | 0 | 112.62 |
| Multi-task Full FT | 67.56 | 60.24 | 65.21 | 16.64 | +2.23 | 30.06 |
| MTLoRA (r=16) | 68.19 | 58.99 | 64.48 | 17.03 | +1.35 | 4.95 |
| MTLoRA (r=32) | 67.74 | 59.46 | 64.90 | 16.59 | +2.16 | 6.08 |
| MTLoRA (r=64) | 67.90 | 59.84 | 65.40 | 16.60 | +2.55 | 8.34 |
| **TADFormer (r=16)** | **69.79** | **59.27** | **65.04** | **16.91** | **+2.44** | **3.56** |
| **TADFormer (r=32)** | **70.20** | **60.00** | **65.71** | **16.57** | **+3.63** | **4.78** |
| **TADFormer (r=64)** | **70.82** | **60.45** | **65.88** | **16.48** | **+4.24** | **7.38** |

### Ablation Study

| Configuration | SemSeg↑ | Parts↑ | Saliency↑ | Normals↓ | Δm(%)↑ |
|------|---------|--------|----------|---------|--------|
| Baseline (MTLoRA r=32) | 67.74 | 59.46 | 64.90 | 16.59 | +2.16 |
| + TPC | Gain | Gain | Gain | Decrease | Gain |
| + DTF | Gain | Gain | Gain | Decrease | Gain |
| + Skip Connection | Gain | Gain | Gain | Decrease | Gain |
| Full TADFormer | **70.20** | **60.00** | **65.71** | **16.57** | **+3.63** |

### Key Findings
- TADFormer outperforms MTLoRA by about 1.2-1.7% in $\Delta m$ with fewer parameters across all rank settings; at rank=16, with only 3.56M parameters, it outperforms MTLoRA rank=64 which utilizes 8.34M parameters.
- DTF is the most critical component, with Grad-CAM visualization clearly demonstrating its ability to capture finer input context.
- Freezing patch merging and only fine-tuning prompt upsampling achieves the optimal trade-off between parameters and performance.
- The improvement on the semantic segmentation task is the most significant (MTLoRA 67.74 vs TADFormer 70.20), showing that dynamic feature extraction is particularly effective for pixel-level tasks.

## Highlights & Insights
- **Introducing Dynamic Convolution to PEFT**: Inserting dynamically generated convolution parameters between the static down-up structure of LoRA elegantly achieves input-aware adaptation with very few additional parameters. This design can be transferred to any scenario that utilizes LoRA.
- **Zero-Cost Reuse of Task Attention Maps**: Directly extracting the prompt-to-patch attention matrix from MHSA as the task attention map without introducing extra computation is a highly elegant design.
- **Using TA-Modules Only in the Last Block**: Most blocks use simple LoRA, with only the last block employing the complex task-adaptation module, striking a good balance between efficiency and performance.

## Limitations & Future Work
- Validations are only performed on PASCAL-Context, lacking experiments on large-scale datasets (such as NYUD-v2, Cityscapes).
- The encoder is limited to Swin Transformer, and other architectures like ViT have not been tested.
- The dynamic convolution of DTF introduces a small amount of extra computational overhead during inference.
- Scenarios with more than 4 tasks have not been explored; the scalability of cross-task interaction remains to be verified.

## Related Work & Insights
- **vs MTLoRA**: MTLoRA uses static task-shared + task-specific LoRA without considering the input context. TADFormer's DTF achieves input-dependent dynamic adaptation, yielding superior performance with similar or fewer parameters.
- **vs VMT-Adapter**: VMT-Adapter introduces cross-task knowledge sharing based on the adapter architecture but similarly neglects input context. TADFormer achieves cross-task interaction at the Transformer attention level.
- **vs HyperFormer**: HyperFormer uses hypernetworks to generate adapter parameters, requiring up to 72.77M parameters. TADFormer achieves a similar effect using dynamic convolution with only 4.78M parameters.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of dynamic convolution and PEFT is novel in MTL, and the TPC operation is cleverly designed.
- Experimental Thoroughness: ⭐⭐⭐ Only validated on PASCAL-Context; experiments on more datasets and backbones are needed.
- Writing Quality: ⭐⭐⭐⭐ Well-structured, intuitive diagrams, and clear methodological comparisons.
- Value: ⭐⭐⭐⭐ Provides a new paradigm for PEFT in MTL, and the idea of input-aware dynamic adaptation has broad transfer value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] MoRE: A Mixture of Low-Rank Experts for Adaptive Multi-Task Learning](../../ACL2025/model_compression/more_a_mixture_of_low-rank_experts_for_adaptive_multi-task_learning.md)
- [\[CVPR 2025\] Task Singular Vectors: Reducing Task Interference in Model Merging](task_singular_vectors_reducing_task_interference_in_model_merging.md)
- [\[CVPR 2026\] Frequency Switching Mechanism for Parameter-Efficient Multi-Task Learning](../../CVPR2026/model_compression/frequency_switching_mechanism_for_parameter-ecient_multi-task_learning.md)
- [\[CVPR 2025\] Expert Pyramid Tuning: Efficient Parameter Fine-Tuning for Expertise-Driven Task Allocation](expert_pyramid_tuning_efficient_parameter_fine-tuning_for_expertise-driven_task_.md)
- [\[CVPR 2026\] Discovering Adaptive Task Dependencies for Efficient Multi-Task Representation Compression](../../CVPR2026/model_compression/discovering_adaptive_task_dependencies_for_efficient_multi-task_representation_c.md)

</div>

<!-- RELATED:END -->

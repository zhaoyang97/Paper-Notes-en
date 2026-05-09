---
title: >-
  [Paper Note] Enhanced Continual Learning of Vision-Language Models with Model Fusion
description: >-
  [ICLR 2026][Multimodal VLM][Continual Learning] This paper proposes the Continual Decoupling-Unifying (ConDU) framework, which is the first to introduce model fusion into VLM continual learning. By maintaining a unified model and performing iterative decoupling-unifying operations guided by task triggers, ConDU surpasses the state of the art by an average of 2% on the MTIL benchmark while simultaneously enhancing zero-shot capability.
tags:
  - ICLR 2026
  - Multimodal VLM
  - Continual Learning
  - Model Fusion
  - Catastrophic Forgetting
  - CLIP
  - Zero-Shot Capability Preservation
date: 2026-05-08
content_hash: 35bffa62da240d1e
---

# Enhanced Continual Learning of Vision-Language Models with Model Fusion

**Conference**: ICLR 2026
**arXiv**: [2503.10705](https://arxiv.org/abs/2503.10705)
**Code**: [GitHub](https://github.com/zhangzicong518/ConDU)
**Area**: Multimodal VLM
**Keywords**: Continual Learning, Model Fusion, Catastrophic Forgetting, CLIP, Zero-Shot Capability Preservation

## TL;DR
This paper proposes the Continual Decoupling-Unifying (ConDU) framework, which is the first to introduce model fusion into VLM continual learning. By maintaining a unified model and performing iterative decoupling-unifying operations guided by task triggers, ConDU surpasses the state of the art by an average of 2% on the MTIL benchmark while simultaneously enhancing zero-shot capability.

## Background & Motivation
VLMs such as CLIP achieve remarkable zero-shot performance by integrating visual and textual modalities. However, sequential fine-tuning on multiple downstream tasks exposes VLMs to the same catastrophic forgetting problem as conventional models. Existing VLM continual learning methods exhibit notable limitations:

**Distillation-based methods** (e.g., ZSCL, Dual-RAIL) require additional reference datasets for knowledge distillation, are sensitive to dataset selection, and demand careful tuning of multiple hyperparameters to balance forgetting mitigation, zero-shot retention, and current-task optimization.

**Parameter-efficient fine-tuning methods** (e.g., DPeCLIP, MulKI) are restricted to adapter or LoRA settings and cannot handle full-parameter fine-tuning.

**Core Insight**: If a dedicated fine-tuned model were maintained per task, the corresponding model could be selected directly given a known task ID. The key idea is to extract and merge the shared components of these individual models into a single unified VLM, while storing task-specific differences in limited memory, thereby emulating the behavior of multiple specialized models with "one main VLM + lightweight auxiliary memory." Model fusion is naturally suited to this scenario, as it enables merging multiple models without access to the original training data.

## Method

### Overall Architecture
ConDU maintains three components throughout the continual learning process: a unified model, a set of task triggers, and a set of prototypes. Upon the arrival of each new task, three steps are executed: (1) independent fine-tuning to obtain a task expert; (2) decoupling historical task experts from the unified model via task triggers; and (3) unifying all task experts into a new unified model. The decoupling and unifying operations require no training and incur only approximately 1% of the computational cost of fine-tuning.

### Key Designs
1. **Delta Model Unifying**: The delta model for task $t$ is defined as the parameter difference between the task expert and the pretrained model: $\delta^t = \theta^t - \theta^0$. The unifying operation selects, for each parameter dimension, the value with the largest absolute magnitude among all delta models that is consistent in sign with their sum: if $\sum_i \delta^i_j > 0$, take $\max_i(\delta^i_j)$; otherwise take $\min_i(\delta^i_j)$. This preserves the largest-magnitude, directionally consistent information shared across models. A binary mask $M^i_j$ (indicating sign agreement between each task's delta model and the unified delta model at each position) and a scaling scalar $\lambda^i$ (to maintain consistent average magnitude) are computed per task and stored as task triggers.

2. **Delta Model Decoupling**: Task triggers are used to reconstruct each task's delta model from the unified delta model: $\tilde{\delta}^i = \lambda^i \cdot M^i \odot \delta^{1:t}$, and the task expert is recovered as $\tilde{\theta}^i = \theta^0 + \tilde{\delta}^i$. This procedure is applied both during training (to retrieve historical task experts for participation in unification) and at inference.

3. **Semantic Aggregation Inference Mechanism**: For task-agnostic or zero-shot scenarios, all task experts are decoupled. The pretrained VLM extracts image features of the test sample, and cosine similarities are computed against each task's class prototypes. The highest similarity score per task serves as the weight for the corresponding expert, and the $K$ experts with the highest weights are selected; their output logits are aggregated to produce the final prediction. Prototypes are defined as the sum of the mean image features and text features: $P^i_k = f(y, \theta^0) + \frac{1}{|\mathcal{D}^t_k|}\sum_m f(x_m, \theta^0)$.

### Loss & Training
- Only standard fine-tuning (full-parameter or LoRA) is applied during training; no additional distillation losses or reference datasets are required.
- The decoupling-unifying operations are entirely training-free.
- The only hyperparameter is $K$ (the number of experts selected at inference); ablations show performance is highly robust to this choice.

## Key Experimental Results

### Main Results

**MTIL Benchmark (11 cross-domain tasks):**

| Method | Transfer↑ | Average↑ | Last↑ |
|--------|-----------|----------|-------|
| Zero-shot | 65.3 | 65.3 | 65.3 |
| ZSCL | 68.1 | 75.4 | 83.6 |
| Dual-RAIL | 69.4 | 77.8 | 86.8 |
| DPeCLIP | 69.1 | 77.5 | 86.9 |
| MulKI | 70.1 | 77.3 | - |
| **ConDU (LoRA)** | **70.3** | **78.3** | **86.2** |
| **ConDU (FT)** | **70.8** | **78.8** | **87.1** |

**Task-Agnostic MTIL (without task ID):**

| Method | Average↑ | Last↑ |
|--------|----------|-------|
| Best Baseline | 76.1 | 84.6 |
| **ConDU (LoRA)** | **78.0** | **85.1** |
| **ConDU (FT)** | **78.1** | **86.4** |

### Ablation Study
- ConDU is effective under both full-parameter fine-tuning and LoRA, making it the only method to support both paradigms simultaneously.
- Few-shot MTIL (5 samples per class): Transfer 70.0%/70.3% (FT/LoRA) exceeds the best baseline by 1.4%; Average 72.3%/72.7% exceeds by 1.3%; Last 76.6%/77.4% exceeds by 1.3%.
- Performance is highly insensitive to the number of aggregated experts $K$ at inference (see Appendix F for details).
- The decoupling-unifying operations incur only approximately 1% of the time cost of fine-tuning.
- Inference with multiple experts running in parallel is comparable in cost to single-model inference.
- The "largest absolute value with consistent sign" unifying strategy outperforms simple averaging and other baseline fusion strategies.

**Few-Shot MTIL Comparison:**

| Method | Transfer↑ | Average↑ | Last↑ |
|--------|-----------|----------|-------|
| Best Baseline | 68.6 | 71.4 | 76.1 |
| ConDU (FT) | 70.0 | 72.3 | 76.6 |
| ConDU (LoRA) | 70.3 | 72.7 | 77.4 |

### Key Findings
- The Transfer metric exceeds the pretrained VLM by 5.5%, indicating that the continual learning process actually enhances zero-shot capability.
- The full-parameter fine-tuning variant (ConDU FT) generally outperforms the LoRA variant, demonstrating that full-parameter fine-tuning retains advantages in continual learning.
- The "largest absolute value with consistent sign" strategy in model fusion proves highly effective at preserving multi-task knowledge.

## Highlights & Insights
- This is the first work to introduce model fusion into VLM continual learning, opening a new research direction.
- The framework is elegantly designed: the decoupling-unifying operations are entirely training-free, and the task triggers (masks and scaling scalars) impose minimal storage overhead.
- The framework is compatible with both full-parameter fine-tuning and parameter-efficient fine-tuning, offering far greater flexibility than existing methods.
- Zero-shot capability is not only preserved but enhanced during continual learning, which is a particularly notable achievement in this setting.

## Limitations & Future Work
- The binary masks in task triggers share the same dimensionality as the unified delta model, which may become a storage bottleneck as the number of tasks grows.
- The optimality of the "largest absolute value" strategy in the unifying operation warrants further theoretical analysis.
- Experiments are conducted solely on the CLIP architecture; applicability to other VLM architectures (e.g., BLIP, LLaVA) remains to be explored.
- Semantic aggregation inference requires forward passes through multiple task experts, increasing inference cost when the number of tasks is very large.

## Related Work & Insights
- **Relation to Task Arithmetic**: The unifying operation in ConDU is inspired by TIES Merging, but a decoupling mechanism is specifically designed for the continual learning setting.
- **Comparison with ZSCL/Dual-RAIL**: These methods require reference datasets and distillation; ConDU requires neither.
- **Insight**: The model fusion perspective offers a new paradigm for continual learning—shifting the focus from "how to prevent forgetting" to "how to efficiently store and reconstruct multiple experts."

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to introduce model fusion into VLM continual learning; framework design is highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐ The MTIL benchmark covers three evaluation settings, though validation is limited to CLIP.
- Writing Quality: ⭐⭐⭐⭐ Architecture diagrams are clear; method description is rigorous with consistent mathematical notation.
- Value: ⭐⭐⭐⭐⭐ Opens a new research direction; the framework is highly general, requires no additional data or distillation, and demonstrates strong practical value.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Continual Learning with Vision-Language Models via Semantic-Geometry Preservation](../../CVPR2026/multimodal_vlm/continual_learning_with_vision-language_models_via_semantic-geometry_preservatio.md)
- [\[AAAI 2026\] ReCAD: Reinforcement Learning Enhanced Parametric CAD Model Generation with Vision-Language Models](../../AAAI2026/multimodal_vlm/recad_reinforcement_learning_enhanced_parametric_cad_model_generation_with_visio.md)
- [\[AAAI 2026\] Branch, or Layer? Zeroth-Order Optimization for Continual Learning of Vision-Language Models](../../AAAI2026/multimodal_vlm/branch_or_layer_zeroth-order_optimization_for_continual_lear.md)
- [\[ICLR 2026\] KeepLoRA: Continual Learning with Residual Gradient Adaptation](keeplora_continual_learning_with_residual_gradient_adaptation.md)
- [\[NeurIPS 2025\] Continual Multimodal Contrastive Learning](../../NeurIPS2025/multimodal_vlm/continual_multimodal_contrastive_learning.md)

<!-- RELATED:END -->

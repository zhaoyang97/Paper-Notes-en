---
title: >-
  [Paper Note] CL-LoRA: Continual Low-Rank Adaptation for Rehearsal-Free Class-Incremental Learning
description: >-
  [CVPR 2025][Model Compression][Class-Incremental Learning] This paper proposes CL-LoRA, which designs a dual-adapter architecture (task-shared + task-specific LoRA). By combining knowledge distillation, gradient reassignment, and learnable block-wise weights, CL-LoRA achieves SOTA continual learning performance with only 0.3% trainable parameters.
tags:
  - "CVPR 2025"
  - "Model Compression"
  - "Class-Incremental Learning"
  - "LoRA"
  - "PEFT"
  - "knowledge distillation"
  - "catastrophic forgetting"
date: 2026-05-08
content_hash: 4a1e5a5271aea518
---

# CL-LoRA: Continual Low-Rank Adaptation for Rehearsal-Free Class-Incremental Learning

**Conference**: CVPR 2025  
**arXiv**: [2505.24816](https://arxiv.org/abs/2505.24816)  
**Code**: [JiangpengHe/CL-LoRA](https://github.com/JiangpengHe/CL-LoRA)  
**Institution**: MIT / Purdue University  
**Area**: Continual Learning / Model Compression  
**Keywords**: Class-Incremental Learning, LoRA, PEFT, knowledge distillation, catastrophic forgetting

## TL;DR
This paper proposes CL-LoRA, which designs a dual-adapter architecture (task-shared + task-specific LoRA). By combining knowledge distillation, gradient reassignment, and learnable block-wise weights, CL-LoRA achieves SOTA continual learning performance with only 0.3% trainable parameters.

## Background & Motivation

**Background**: Class-incremental learning (CIL) requires models to sequentially learn new classes while preserving previous knowledge. Recently, pre-trained models (PTMs) combined with Parameter-Efficient Fine-Tuning (PEFT) have demonstrated promising results in rehearsal-free CIL without storing exemplar images from old tasks.

**Limitations of Prior Work**:
   - **Prompt-based methods** (L2P, DualPrompt, CODA-Prompt): Require a large number of prompt parameters, and the task selection mechanisms are complex.
   - **Adapter-based methods** (EASE, O-LoRA, InfLoRA): Create a new adapter for each task, leading to parameter redundancy.
   - Existing methods fail to effectively utilize shared knowledge across tasks, and independent learning for each task leads to fragmented knowledge.

**Key Challenge**: How to maintain parameter efficiency while simultaneously learning task-shared knowledge and capturing task-specific features, thereby balancing stability (no forgetting) and plasticity (learning new knowledge).

**Key Insight**: Designing a dual-adapter architecture, where the first half of the Transformer layers uses a shared LoRA to learn cross-task knowledge, and the second half uses task-specific LoRAs to capture task-specific features.

**Core Idea**: Shared LoRA (with knowledge distillation + gradient reassignment) + Specific LoRA (with block-wise weights + orthogonality constraints) = Efficient Continual Learning.

## Method

### Overall Architecture
Based on the ViT-B/16 pre-trained model, the 12 Transformer layers are split into two halves: the first 6 layers insert the task-shared LoRA, and the remaining 6 layers insert the task-specific LoRAs. A prototype classifier is used during inference.

### Key Designs

1. **Task-Shared Adapter**

    - **Function**: Inserts a shared LoRA in the first $l=6$ Transformer blocks to learn general knowledge across tasks.
    - **Mechanism**: Initializes $\mathbf{A}_s$ with a random orthogonal matrix and only updates $\mathbf{B}_s$.
    - **Knowledge Distillation**: When learning a new task, the shared adapter from the previous task serves as the teacher to distill knowledge into the current adapter.
    - **Early Exit Strategy**: Computes the distillation loss only at the last layer of the shared adapter, reducing computational overhead.

2. **Gradient Reassignment**

    - **Function**: Identifies and protects parameters in the shared adapter that are important to old tasks.
    - **Mechanism**: Calculates the difference in gradients between the teacher and student models, and reduces the learning rate for important parameters.
    - **Implementation**: $\nabla \mathcal{L}'_{kd} = \nabla \mathcal{L}_{kd} \odot |\nabla_{\mathbf{B}_s^{t-1}} \mathcal{L}_{kd} - \nabla_{\mathbf{B}_s^{t}} \mathcal{L}_{kd}|$
    - **Effect**: Retains crucial shared knowledge more precisely.

3. **Task-Specific Adapter and Block-wise Weights**

    - **Function**: Each task has an independent LoRA in the remaining $N-l$ Transformer blocks.
    - **Block-wise Weights**: Learns trainable block-wise scaling factors $w_i^j$ for each specific adapter.
    - **Orthogonality Constraint**: $\mathcal{L}_{orth} = \sum_{j=l+1}^{N} \sum_{i \neq k} \| \mathbf{B}_i^j {}^\top \mathbf{B}_k^j \|_F^2$
    - **Design Motivation**: Different tasks may require differing levels of adaptation at different Transformer layers.

4. **Prototype Classifier Inference**

    - Features are computed for each learned task using its corresponding task-specific adapter.
    - The cosine similarity between the features and each task's prototypes is computed.
    - The class with the highest similarity is predicted.

### Loss & Training
$$\mathcal{L} = \mathcal{L}_{ce} + \lambda_1 \mathcal{L}_{kd} + \lambda_2 \mathcal{L}_{orth}$$

Hyperparameters: $\lambda_1 = 5$, $\lambda_2 = 0.0001$, LoRA rank $r = 10$.

## Key Experimental Results

### Main Results: Average Accuracy $\bar{A}$ (%) on 4 Benchmarks

| Method | Params (%) | CIFAR-100 T=10 | ImageNet-R T=20 | ImageNet-A T=20 | VTAB T=10 |
|------|----------|---------------|----------------|----------------|----------|
| L2P | 0.2 | 79.51 | 65.82 | 39.81 | 78.96 |
| DualPrompt | 0.5 | 80.44 | 67.41 | 56.43 | 82.51 |
| EASE | 1.4 | 85.71 | 78.04 | 68.92 | 93.01 |
| RanPAC | 3.1 | **87.62** | **78.53** | 66.14 | 89.61 |
| InfLoRA | 0.3 | 80.97 | 73.22 | 56.91 | 88.83 |
| O-LoRA | 0.4 | 81.26 | 72.52 | 55.02 | 87.22 |
| **CL-LoRA** | **0.3** | 85.32 | **81.58** | **70.15** | **94.57** |

### ImageNet-R Long-Sequence (T=40) Average Accuracy

| Method | $\bar{A}$ |
|------|----------|
| InfLoRA | 47.04 |
| O-LoRA | 47.53 |
| **CL-LoRA** | **60.54** |

CL-LoRA demonstrates a more significant advantage on long sequences, outperforming the compilation-based runner-up by 13 percentage points.

### Ablation Study (CIFAR-100 T=10 / ImageNet-R T=20)

| KD | GR | BW | CIFAR-100 | ImageNet-R |
|----|----|----|-----------|------------|
| ✗ | ✗ | ✗ | 88.20 | 82.24 |
| ✓ | ✗ | ✗ | 90.83 | 83.42 |
| ✓ | ✓ | ✗ | 91.69 | 84.08 |
| ✗ | ✗ | ✓ | 89.01 | 82.93 |
| ✓ | ✓ | ✓ | **91.85** | **84.77** |

Each of the three modules contributes to the performance, with KD yielding the most substantial gain.

## Highlights & Insights
- Elegant design of the **dual-adapter architecture**: shared knowledge in early layers, specialized adaptation in deep layers.
- Achieves SOTA with only **0.3% trainable parameters**, remaining highly competitive while using 10x fewer parameters than RanPAC (3.1%).
- The gradient reassignment mechanism protects critical parameters more precisely than naive knowledge distillation.
- The advantage is particularly pronounced on challenging benchmarks with distribution shifts (ImageNet-R/A).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Achieving More with Less: Additive Prompt Tuning for Rehearsal-Free Class-Incremental Learning](../../ICCV2025/model_compression/achieving_more_with_less_additive_prompt_tuning_for_rehearsal-free_class-increme.md)
- [\[CVPR 2025\] Tripartite Weight-Space Ensemble for Few-Shot Class-Incremental Learning](tripartite_weight-space_ensemble_for_few-shot_class-incremental_learning.md)
- [\[CVPR 2025\] LoRA Subtraction for Drift-Resistant Space in Exemplar-Free Continual Learning](lora_subtraction_for_drift-resistant_space_in_exemplar-free_continual_learning.md)
- [\[NeurIPS 2025\] Gated Integration of Low-Rank Adaptation for Continual Learning of Large Language Models](../../NeurIPS2025/model_compression/gated_integration_of_low-rank_adaptation_for_continual_learning_of_large_languag.md)
- [\[CVPR 2025\] Adapter Merging with Centroid Prototype Mapping for Scalable Class-Incremental Learning](adapter_merging_with_centroid_prototype_mapping_for_scalable_class-incremental_l.md)

</div>

<!-- RELATED:END -->

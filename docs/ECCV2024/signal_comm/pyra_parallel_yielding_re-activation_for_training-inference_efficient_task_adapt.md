---
title: >-
  [Paper Note] PYRA: Parallel Yielding Re-Activation for Training-Inference Efficient Task Adaptation
description: >-
  [ECCV 2024][Signal & Communication][Parameter-Efficient Fine-Tuning] This paper proposes PYRA, which generates decoupled adaptive modulation weights in parallel and modulates features of tokens to be merged using a re-activation strategy. This approach enables Vision Transformers to achieve both training efficiency (tuning only 0.4% parameters) and inference efficiency (approx. 1.7x-3.2x speedup) during downstream task adaptation, achieving comparable or superior performance…
tags:
  - "ECCV 2024"
  - "Signal & Communication"
  - "Parameter-Efficient Fine-Tuning"
  - "Model Compression"
  - "Token Merging"
  - "Vision Transformer"
  - "Training-Inference Efficiency"
date: 2026-05-08
content_hash: 3214ff5199855851
---

# PYRA: Parallel Yielding Re-Activation for Training-Inference Efficient Task Adaptation

**Conference**: ECCV 2024  
**arXiv**: [2403.09192](https://arxiv.org/abs/2403.09192)  
**Code**: [https://github.com/THU-MIG/PYRA](https://github.com/THU-MIG/PYRA)  
**Area**: Signal & Communication (Model Efficiency)  
**Keywords**: Parameter-Efficient Fine-Tuning, Model Compression, Token Merging, Vision Transformer, Training-Inference Efficiency

## TL;DR
This paper proposes PYRA, which generates decoupled adaptive modulation weights in parallel and modulates features of tokens to be merged using a re-activation strategy. This approach enables Vision Transformers to achieve both training efficiency (tuning only 0.4% parameters) and inference efficiency (approx. 1.7x-3.2x speedup) during downstream task adaptation, achieving comparable or superior performance to uncompressed PEFT methods.

## Background & Motivation
As Vision Transformers scale up to billions of parameters, downstream task adaptation faces two core challenges: training overhead (full fine-tuning consumes massive GPU resources) and inference efficiency (prohibitive computational cost during deployment). Existing solutions address these two issues separately: PEFT (such as LoRA) freezes the backbone and tunes only a few parameters to tackle training efficiency, but does not improve inference speed; model compression (such as pruning) improves inference efficiency but requires significant training resources for structure search and retraining. A simple combination of the two fails to guarantee both training and inference efficiency—performance slightly degrades under low compression rates and deteriorates sharply under high compression rates (>3x speedup), sometimes performing worse than directly fine-tuning a small model with equivalent throughput (a phenomenon the authors conceptualize as "Adverse Compression"). Key Challenge: The limited parameter updating capability of PEFT cannot fully perceive downstream data distributions, and the information loss during token merging constantly accumulates through layer-by-layer propagation, ultimately leading to performance collapse. Key Insight: Adaptively modulate the features of tokens to be merged *before* token merging, using a minimal number of learnable parameters to compensate for the information loss caused by merging. Core Idea: Generate decoupled modulation weights along the channel and token dimensions in parallel, and robustly modulate token features using a sigmoid re-activation strategy.

## Method

### Overall Architecture
PYRA is built on a baseline of LoRA + Token Merging (ToMe). Token merging is executed before the MHSA of each ViT block: first, the token pairs $(t_{m_k}, t_{n_k})$ to be merged are identified in the same manner as ToMe; then, prior to merging (average pooling), the token features on the $t_{m_k}$ side are modulated through the PYRA module. The PYRA module is highly lightweight, introducing only two learnable vectors $W_r \in \mathbb{R}^{r \times 1}$ and $W_D \in \mathbb{R}^{1 \times D}$ per layer. The LoRA module is attached to the Q/K/V projection matrices and can be merged into the backbone during inference without introducing any additional computational cost. The entire system is trained end-to-end.

### Key Designs
1. **Parallel Yielding Adaptive Weights**:
    - Function: Generates an adaptive modulation weight matrix $W^l \in \mathbb{R}^{D \times r}$ for each pair of tokens to be merged.
    - Mechanism: Decomposes the modulation weight matrix into the outer product of a channel-dimension weight $\delta_D^l \in \mathbb{R}^{D \times 1}$ and a token-dimension weight $\delta_r^l \in \mathbb{R}^{1 \times r}$, i.e., $W^l = \delta_D^l \delta_r^l$. First, the token pair to be merged is summed and processed with LayerNorm to obtain the information matrix $M_\text{info}^l = \text{LayerNorm}(M_s^l + M_t^l) \in \mathbb{R}^{D \times r}$. Then, it calculates $\delta_D^l = M_\text{info}^l W_r^l$ (perceiving token distribution) and $\delta_r^l = W_D^l M_\text{info}^l$ (perceiving channel distribution) in parallel.
    - Design Motivation: Directly learning the complete matrix $W^l$ is redundant and lacks adaptability (yielding fixed weights for different images/tokens). Decomposing it into two directions requires extremely few parameters ($D + r$ parameters) and naturally achieves input-adaptability via multiplication with the token information matrix.

2. **Re-Activation Token Modulation Strategy**:
    - Function: Robustly applies the generated weights to modulate token features.
    - Mechanism: Modulates in two steps. First step: Broadcast $\delta_D^l$ and apply sigmoid activation to it, then perform element-wise multiplication with $M_s^l$ to obtain the intermediate result $\hat{M}_s^l = 2\sigma(\hat{\delta}_D^l) \odot M_s^l$. Second step: Broadcast $\delta_r^l$, apply sigmoid activation, and modulate $M_s^l$ again: $M_s^l \leftarrow M_s^l + (2\sigma(\hat{\delta}_r^l) - 1) \odot \hat{M}_s^l$. Residual connections ensure gradient flow, and zero-initializing $W_D^l$ ensures that training begins equivalent to an identity mapping.
    - Design Motivation: (1) Sigmoid constrains the weights within a reasonable range, preventing training instability; (2) The combination of two non-linear activations enhances the expressiveness of the low-rank modulation matrix; (3) Residuals + zero-initialization ensure a smooth transition from the baseline method.

3. **Token Merging Baseline Choice**:
    - Function: Provides parameter-free, training-free inference acceleration.
    - Mechanism: Adopts ToMe's bipartite graph matching mechanism, pairing tokens based on cosine similarity and choosing the most similar $r$ pairs for average-pooling merging. This reduces $r$ tokens per layer, accumulating layer by layer to achieve overall compression.
    - Design Motivation: ToMe does not alter the model structure or introduce extra parameters, making it fully compatible with the storage efficiency of PEFT. Since LoRA can be merged into the backbone during inference, combining the two yields a clean and strong baseline.

### Loss & Training
Using the same training strategy as LoRA, the generator of the PYRA module is trained end-to-end alongside the LoRA modules. Each layer of PYRA adds only $D + r$ parameters (e.g., about 8.64K in total for ViT-B) and $4rD$ FLOPs of extra computation. $W_r$ is initialized with a random Gaussian distribution, and $W_D$ is initialized with zeros. During training, only $M_s$ (rather than $M_t$) is modulated because $t_{m_k}$ is unique, whereas different $t_{n_k}$ may point to the same target token; processing them separately guarantees parallelizability.

## Key Experimental Results

### Main Results
Performance comparison under low compression rate (~1.7x speedup) on VTAB-1k benchmark:

| Method | Tunable Params | Throughput | ViT-B Avg | ViT-L Avg |
|------|---------|--------|-----------|-----------|
| PEFT (LoRA) | 0.34%/0.39% | 425/130 | 74.76 | 76.52 |
| DiffRate | 0.35%/0.39% | 709/221 | 55.82 | 59.53 |
| ToMe | 0.34%/0.39% | 753/227 | 74.10 | 76.11 |
| **PYRA** | **0.35%/0.40%** | **745/225** | **74.69** | **76.84** |

Performance comparison under high compression rate (~3.2x speedup):

| Method | Tunable Params | Throughput | ViT-B Avg | ViT-L Avg |
|------|---------|--------|-----------|-----------|
| Small Model PEFT* | 0.34% | 1350/425 | 71.85 | 74.76 |
| ToMe | 0.34%/0.39% | 1381/431 | 70.43 | 74.10 |
| **PYRA** | **0.35%/0.40%** | **1365/427** | **72.06** | **75.66** |

PYRA experiences almost no performance loss at low compression rates (ViT-L even improves by 0.32), and eliminates the Adverse Compression phenomenon at high compression rates, outperforming small models of equivalent throughput.

### Ablation Study

| Configuration | Natural | Specialized | Structured | Average | Description |
|------|---------|-------------|------------|---------|------|
| Baseline (ToMe+LoRA) | 72.87 | 80.78 | 57.64 | 70.43 | Baseline |
| + $W_r$ only (w/o activation) | 72.90 | 81.07 | 57.66 | 70.54 | Slight improvement with token-dimension weight |
| + $W_r$ only (w/ activation) | 73.18 | 81.69 | 57.65 | 70.84 | Significant improvement with activation function |
| + $W_D$ only (w/o activation) | 73.09 | 81.13 | 58.43 | 70.88 | Stronger effect with channel-dimension weight |
| + $W_D$ only (w/ activation) | 73.31 | 82.17 | 58.44 | 71.31 | Activation function improves performance further |
| + $W_r$ & $W_D$ (w/o activation) | 73.77 | 81.37 | 58.81 | 71.32 | Parallel combination outperforms individual ones |
| **PYRA (All)** | **73.91** | **82.60** | **59.66** | **72.06** | All components complement each other |

### Key Findings
- The combined effect of parallel decoupled weights outperforms using either dimension alone, proving that token and channel-wise modulations are complementary.
- Sigmoid re-activation brings significant improvements (+0.3~0.5) across all configurations, which is key to ensuring training stability.
- Directly learning the complete modulation matrix $W_{D \times r}$ (70.66K parameters, 71.49 avg) is inferior to PYRA (8.64K parameters, 72.06 avg), demonstrating that adaptively generating weights is more effective than using fixed weights.
- Compared to the commonly used gated MLP generator (73.73K parameters, 71.06 avg), PYRA achieves better results with less than 1/8 of the parameter size.
- Demonstrates good generalizability across self-supervised pre-training (MAE) and different architectures (DeiT): at low compression rates, MAE ViT-L even outperforms the uncompressed baseline.
- PYRA consistently flattens the accuracy-throughput curve across different compression rates, showing robustness to compression.

## Highlights & Insights
- Accurately identifies a practical pain point: PEFT only focuses on training efficiency without considering inference efficiency, while model compression focuses on inference efficiency without considering training efficiency, and a simple combination of the two is insufficient.
- Highly lightweight design—adding only $D + r$ parameters per layer (totaling 8.64K for ViT-B) with almost negligible extra FLOPs, embodying a design philosophy of "minimal intrusion".
- The zero-initialization + residual connection design ensures PYRA is initially equivalent to an identity mapping, allowing a smooth transition from the baseline and highly stable training.
- The introduction and naming of the "Adverse Compression" phenomenon helps the community understand failure modes in compression + fine-tuning.
- Comprehensive experimental coverage across 4 backbones $\times$ 2 compression rates $\times$ 19 tasks.

## Limitations & Future Work
- Only verified on image classification (VTAB-1k), lacking validation on dense prediction tasks such as object detection and segmentation.
- Token modulation only acts on $M_s$ (the side being merged), while $M_t$ remains unmodulated, leaving potential information underutilized.
- The method is heavily coupled with ToMe's bipartite graph matching mechanism, and its adaptability to other token reduction strategies (e.g., attention-driven pruning) remains unverified.
- Under high compression rates, the comparison with small models assumes that such small models are available, though in practice, pre-trained weights for small models typically exist.
- Only used LoRA as the PEFT method; compatibility with other PEFT methods (e.g., Adapters, Prompt Tuning) has not been explored.

## Related Work & Insights
- **LoRA**: Low-Rank Adaptation, selected as the default PEFT module in this paper.
- **ToMe**: Token Merging, a parameter-free token reduction method, used as the compression baseline on which the proposed method is built.
- **DiffRate**: A method to search for the optimal token reduction rate per layer, but it requires ImageNet-21K searching and performs poorly in PEFT scenarios.
- **SSF / AdaptFormer / Consolidator**: Various PEFT methods; the proposed framework is theoretically compatible with them.
- Insight: Information loss during token merging can be effectively compensated for via extremely lightweight feature modulation, offering a viable paradigm for "joint training-inference efficiency".

## Rating
- Novelty: ⭐⭐⭐⭐ Novel problem definition (Training-Inference Efficient Task Adaptation); the solution design is neat yet intuitive.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 4 backbones, 2 compression rates, 19 tasks, with comprehensive ablations and comparisons.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, standard formulation, intuitive figures, well-articulated motivation.
- Value: ⭐⭐⭐⭐ Fills the gap in PEFT + model compression; the method is lightweight and practical, providing tangible value for large model deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] AcTTA: Rethinking Test-Time Adaptation via Dynamic Activation](../../CVPR2026/signal_comm/actta_rethinking_test-time_adaptation_via_dynamic_activation.md)
- [\[CVPR 2025\] DiTASK: Multi-Task Fine-Tuning with Diffeomorphic Transformations](../../CVPR2025/signal_comm/ditask_multi-task_fine-tuning_with_diffeomorphic_transformations.md)
- [\[ICLR 2026\] Efficient Message-Passing Transformer for Error Correcting Codes](../../ICLR2026/signal_comm/efficient_message-passing_transformer_for_error_correcting_codes.md)
- [\[NeurIPS 2025\] Angular Steering: Behavior Control via Rotation in Activation Space](../../NeurIPS2025/signal_comm/angular_steering_behavior_control_via_rotation_in_activation_space.md)
- [\[AAAI 2026\] Task Aware Modulation Using Representation Learning for Upscaling of Terrestrial Carbon Fluxes](../../AAAI2026/signal_comm/task_aware_modulation_using_representation_learning_for_upsaling_of_terrestrial_.md)

</div>

<!-- RELATED:END -->

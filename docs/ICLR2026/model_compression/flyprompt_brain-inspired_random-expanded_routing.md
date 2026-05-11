---
title: >-
  [Paper Note] FlyPrompt: Brain-Inspired Random-Expanded Routing with Temporal-Ensemble Experts for General Continual Learning
description: >-
  [ICLR 2026][Model Compression][continual learning] Inspired by the neurobiology of the *Drosophila* mushroom body—specifically its sparse random expansion and modular integration mechanisms—FlyPrompt is proposed as a fra…
tags:
  - "ICLR 2026"
  - "Model Compression"
  - "continual learning"
  - "prompt tuning"
  - "brain-inspired"
  - "expert routing"
  - "temporal ensemble"
date: 2026-05-08
content_hash: 4a07e71c2ad20a33
---

# FlyPrompt: Brain-Inspired Random-Expanded Routing with Temporal-Ensemble Experts for General Continual Learning

**Conference**: ICLR 2026
**arXiv**: [2602.01976](https://arxiv.org/abs/2602.01976)
**Code**: N/A
**Area**: Model Compression
**Keywords**: continual learning, prompt tuning, brain-inspired, expert routing, temporal ensemble

## TL;DR

Inspired by the neurobiology of the *Drosophila* mushroom body—specifically its sparse random expansion and modular integration mechanisms—FlyPrompt is proposed as a framework for General Continual Learning (GCL). It introduces a Random-Expanded Analytic Router (REAR) for non-iterative expert selection, combined with a multi-timescale EMA output head Temporal Ensemble (TE²) to enhance expert capacity, achieving gains of up to 11.23%/12.43%/7.62% on CIFAR-100/ImageNet-R/CUB-200.

## Background & Motivation

1. **GCL is substantially harder than conventional CL**: GCL requires learning from a non-stationary data stream in a single pass, without explicit task boundaries and with potentially overlapping label spaces. The clear task delineation and multi-epoch training assumed in traditional CL no longer hold.
2. **Routing instability in existing PET methods**: Methods such as L2P, DualPrompt, MVP, and MISA jointly train the router and experts. Under GCL's ambiguous boundaries and single-pass constraint, the router tends to overfit early data or be affected by distributional drift; empirical results show routing accuracy is far from satisfactory.
3. **Expert capacity degrades under single-pass training**: Even with oracle routing, the final accuracy of existing methods remains limited (Fig. 2c), indicating that the quality of expert representations and the decision boundaries of output heads gradually misalign in non-stationary streams—a second bottleneck independent of routing.
4. **Class imbalance exacerbates interference**: The class distribution in GCL streams is long-tailed with cross-task overlap; a single shared output head is continuously overwritten by subsequent tasks, causing decision boundary drift for earlier experts.
5. **The *Drosophila* mushroom body as a biological paradigm**: With fewer than 100,000 neurons, *Drosophila* exhibits robust lifelong learning. A ~40× sparse random expansion from projection neurons to Kenyon cells enables efficient pattern separation, while multi-timescale plasticity across the γ/α'β'/αβ subcompartments supports parallel consolidation of short-, medium-, and long-term memory.
6. **Lack of joint design targeting both routing and expert capacity in the CL literature**: Existing work either focuses solely on forgetting prevention (regularization/replay) or on routing (prompt selection), without explicitly decomposing GCL into the two sub-problems of "expert routing" and "expert capacity enhancement" and addressing them jointly.

## Method

### Overall Architecture: FlyPrompt = REAR + TE²

- **Function**: GCL is explicitly decomposed into two sub-problems—(1) *expert routing*: assigning each input to an appropriate prompt expert; and (2) *expert capacity*: improving the representational robustness and decision boundary adaptability of each expert under limited supervision.
- **Design Motivation**: Empirical analysis (Fig. 2b–c) demonstrates that the two sub-problems constitute independent bottlenecks: low routing accuracy and insufficient expert capacity respectively constrain the upper and lower performance bounds, necessitating targeted designs.
- **Mechanism**: REAR handles routing via random expansion and analytic solving, enabling gradient-free routing in a single forward pass. TE² addresses expert capacity by capturing knowledge across different temporal windows using EMA heads with multiple decay rates. At inference, REAR selects an expert → the expert prompt extracts features → TE² aggregates predictions from multiple heads.

### Key Design 1: Random-Expanded Analytic Router (REAR)

- **Function**: A fixed random projection matrix expands pre-trained features into a high-dimensional sparse space, in which a router is constructed via a closed-form ridge regression solution, requiring no backpropagation.
- **Design Motivation**: This mimics the ~40× random expansion from projection neurons to Kenyon cells in the *Drosophila* mushroom body. High-dimensional sparse representations are naturally more linearly separable and are unaffected by distributional drift (the random matrix $\mathbf{R}$ remains fixed). The closed-form solution avoids the forgetting and instability associated with online router training.
- **Mechanism**:
  1. Extract pre-trained features $\mathbf{h} = f_\theta(\mathbf{x}) \in \mathbb{R}^d$ from input $\mathbf{x}$.
  2. Random expansion: $\varphi(\mathbf{x}) = \sigma(\mathbf{h}\mathbf{R}) \in \mathbb{R}^M$, where $\mathbf{R} \sim \mathcal{N}(0,1)^{d \times M}$ and $M \gg d$ (default $M=10^4$).
  3. Incrementally accumulate the Gram matrix $\mathbf{G} \leftarrow \mathbf{G} + \Phi_i^\top\Phi_i$ and prototype matrix $\mathbf{Q} \leftarrow \mathbf{Q} + \Phi_i^\top\mathbf{C}_t$.
  4. At inference, solve analytically: $\hat{\mathbf{U}}^\top = (\mathbf{G} + \lambda\mathbf{I})^{-1}\mathbf{Q}$, and select the highest-scoring expert $\hat{E}(\mathbf{x}) = \arg\max_t [\varphi(\mathbf{x})\hat{\mathbf{U}}^\top]_t$.
  5. Theoretical guarantee (Theorem 1): population excess risk $\lesssim \sqrt{\log N/M} + (N\lambda)^{-1/2} + \lambda$; routing error can be made arbitrarily small by increasing $M$ and $N$.

### Key Design 2: Temporal Ensemble of Task Experts (TE²)

- **Function**: Each expert $E_t$ maintains one online head and $n$ EMA shadow heads with distinct decay rates; at inference, element-wise maximum is taken over the softmax outputs of all heads.
- **Design Motivation**: This emulates the multi-timescale memory consolidation of the γ/α'β'/αβ subcompartments of the *Drosophila* mushroom body—short windows ($\alpha=0.9$, $L\approx10$) capture recent pattern changes, while long windows ($\alpha=0.99$, $L\approx100$) retain long-term stable knowledge. Theorem 2 shows that the EMA parameter error decomposes into a variance term $\mathcal{O}(\zeta^2/L)$ and a drift bias term $\mathcal{O}((LP_t)^2)$; a geometrically spaced EMA bank guarantees that at any point in time, at least one head approximates the optimal bias–variance trade-off.
- **Mechanism**:
  1. When a new task begins, initialize the new prompt as the mean of all existing prompts (warm start).
  2. During training, update only the online head $\psi$ and the current prompt $\mathbf{p}_t$ using cross-entropy loss with logit masking (activating only classes in the current batch).
  3. After each gradient update, update EMA head parameters: $\mathbf{W}_t^{(j)} \leftarrow \alpha_j \mathbf{W}_t^{(j)} + (1-\alpha_j)\mathbf{W}$.
  4. At inference, compute softmax over the online head and all EMA heads separately, take element-wise maximum, and predict.

## Key Experimental Results

### Main Results: Overall GCL Performance (Table 1, Sup-21K Backbone)

| Method | CIFAR-100 $A_{\text{auc}}$ | CIFAR-100 $A_{\text{last}}$ | ImageNet-R $A_{\text{auc}}$ | ImageNet-R $A_{\text{last}}$ | CUB-200 $A_{\text{auc}}$ | CUB-200 $A_{\text{last}}$ |
|------|:-:|:-:|:-:|:-:|:-:|:-:|
| L2P | 76.23 | 79.11 | 44.40 | 42.03 | 64.30 | 61.42 |
| DualPrompt | 76.04 | 76.62 | 46.13 | 40.80 | 65.03 | 62.43 |
| CODA-P | 79.13 | 80.91 | 51.87 | 48.09 | 66.01 | 62.90 |
| MISA | 80.35 | 80.75 | 51.52 | 45.08 | 65.40 | 60.20 |
| **FlyPrompt** | **83.24** | **86.76** | **56.58** | **55.27** | **70.64** | **73.40** |

**Key Findings**: FlyPrompt substantially outperforms all baselines across all three datasets. $A_{\text{auc}}$ gains are +2.89/+4.71/+4.63, while $A_{\text{last}}$ gains are more pronounced (+5.85/+7.18/+10.50), indicating particular strength in resisting forgetting in later stages. Consistent advantages are maintained across 6 different pre-trained backbones (supervised and self-supervised).

### Comparison with Offline CL Methods (Table 2, Sup-21K)

| Method | CIFAR-100 $A_{\text{auc}}$/$A_{\text{last}}$ | ImageNet-R $A_{\text{auc}}$/$A_{\text{last}}$ | CUB-200 $A_{\text{auc}}$/$A_{\text{last}}$ |
|------|:-:|:-:|:-:|
| S-Prompt++ | 80.21/83.48 | 52.14/49.13 | 66.61/64.73 |
| HiDe-LoRA | 80.07/82.00 | 55.09/51.29 | 67.26/67.28 |
| SD-LoRA | 79.26/78.91 | 55.51/51.97 | 64.12/60.57 |
| **FlyPrompt** | **83.24/86.76** | **56.58/55.27** | **70.64/73.40** |

**Key Findings**: As an online single-pass method, FlyPrompt surpasses even offline methods employing multi-epoch training (S-Prompt++, HiDe series), demonstrating the superiority of the bio-inspired design in the efficiency–performance trade-off.

### Ablation Study (Table 3)

| Configuration | CIFAR-100 $A_{\text{auc}}$ | $A_{\text{last}}$ | ImageNet-R $A_{\text{auc}}$ | $A_{\text{last}}$ |
|----------|:-:|:-:|:-:|:-:|
| w/o REAR, w/o EMA | 71.33 | 73.22 | 41.73 | 37.33 |
| +Prompt Expert | 80.75 | 83.65 | 54.91 | 52.58 |
| +REAR | 81.90 | 84.23 | 55.76 | 52.76 |
| +EMA | 82.17 | 83.75 | 55.90 | 53.65 |
| **REAR+Expert+EMA** | **83.24** | **86.76** | **56.58** | **55.27** |

**Key Findings**: The Prompt Expert component is the largest contributor (+9.42 $A_{\text{auc}}$); REAR and TE² each contribute approximately 1–2%. Their combination yields synergistic effects (full model > sum of any two components), confirming that routing and capacity enhancement are complementary dimensions.

## Highlights & Insights

- **Interdisciplinary innovation**: The sparse expansion and multi-timescale memory consolidation principles of the *Drosophila* mushroom body are translated into implementable algorithmic components, representing an exemplary intersection of NeuroAI and continual learning.
- **Forward-pass-only routing**: REAR requires no backpropagation whatsoever; the closed-form solution comes with a theoretical error bound, making it particularly suitable for online and edge deployment.
- **Plug-and-play design**: REAR and TE² can be independently integrated into existing methods (DualPrompt, MISA, etc.) with consistent performance improvements (Table 4).
- **Minimal overhead**: Only a 0.83% parameter increase (87.08M vs. 86.37M for MISA), with negligible additional training time.

## Limitations & Future Work

- The EMA decay rates for temporal ensemble ($\{0.9, 0.99\}$) are fixed hyperparameters that cannot adapt to the current rate of distributional drift; extreme drift scenarios may require dynamic adjustment.
- The random projection dimension $M = 10^4$ introduces storage overhead for the $\mathbb{R}^{d \times M}$ matrix (~30 MB), and Gram matrix inversion may become a bottleneck in very long task sequences.
- Experiments are conducted primarily under the Si-Blurry setting, which, while flexible, remains an artificially constructed controlled scenario; real-world data streams may be considerably more irregular.
- Validation on large-scale datasets (e.g., full 1000-class ImageNet-1K GCL) or multimodal continual learning settings has not been performed.

## Related Work & Insights

| Dimension | FlyPrompt (Ours) | MISA (ICLR 2025) | CODA-P (CVPR 2023) |
|------|------------------|-------------------|---------------------|
| Routing mechanism | Random expansion + closed-form (gradient-free) | Contrastive learning + mutual information (gradient-based) | Attention-weighted prompt combination |
| Expert capacity | Multi-timescale EMA heads | Prompt initialization + adaptation | Single head + combined prompts |
| GCL support | Natively designed, online single-pass | Natively designed, online single-pass | Requires multi-epoch; degrades under GCL |
| CIFAR-100 $A_{\text{auc}}$ | **83.24%** | 80.35% | 79.13% |
| Theoretical guarantees | Routing error bound + EMA error decomposition | None | None |

## Rating

- ⭐⭐⭐⭐ **Novelty**: The mapping from biological principles to algorithmic components is natural and effective; both the REAR closed-form router and multi-timescale EMA constitute novel designs.
- ⭐⭐⭐⭐⭐ **Experimental Thoroughness**: 6 backbones × 3 datasets × 8+ baselines, with comprehensive ablation, plug-and-play, hyperparameter sensitivity, and computational cost analyses.
- ⭐⭐⭐⭐ **Writing Quality**: The biological analogy is vivid, the dual sub-problem decomposition is logically clear, and theory and experiments are well-aligned.
- ⭐⭐⭐⭐ **Value**: Plug-and-play components with minimal overhead and gradient-free routing hold practical promise for edge deployment and online learning scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] IDER: IDempotent Experience Replay for Reliable Continual Learning](ider_idempotent_experience_replay_for_reliable_continual_learning.md)
- [\[ICLR 2026\] Null-Space Filtering for Data-Free Continual Model Merging: Preserving Stability, Promoting Plasticity](null-space_filtering_for_data-free_continual_model_merging_preserving_stability_.md)
- [\[ICLR 2026\] LD-MoLE: Learnable Dynamic Routing for Mixture of LoRA Experts](ld-mole_learnable_dynamic_routing_for_mixture_of_lora_experts.md)
- [\[ICLR 2026\] Knowledge Fusion of Large Language Models Via Modular Skillpacks](knowledge_fusion_of_large_language_models_via_modular_skillpacks.md)
- [\[ICLR 2026\] Steering MoE LLMs via Expert (De)Activation](steering_moe_llms_via_expert_deactivation.md)

</div>

<!-- RELATED:END -->

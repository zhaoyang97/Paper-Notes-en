---
title: >-
  [Paper Note] Deterministic Continuous Replacement: Fast and Stable Module Replacement in Pretrained Transformers
description: >-
  [NeurIPS 2025 (ScaleOPT Workshop)][Model Compression][Module replacement] DCR mixes teacher and student module outputs via a deterministic annealing weight $\alpha(t)$, eliminating the gradient variance introduced by stochastic gating (e.g., BERT-of-Theseus), and achieves faster convergence and stronger feature alignment in cold-start module replacement scenarios.
tags:
  - NeurIPS 2025 (ScaleOPT Workshop)
  - Model Compression
  - Module replacement
  - deterministic mixing
  - gradient variance
  - knowledge distillation
  - Vision Transformer
date: 2026-05-08
content_hash: 731d8c38f762586c
---

# Deterministic Continuous Replacement: Fast and Stable Module Replacement in Pretrained Transformers

**Conference**: NeurIPS 2025 (ScaleOPT Workshop)
**arXiv**: [2511.18670](https://arxiv.org/abs/2511.18670)
**Code**: Not yet released (authors state it will be made available in the extended version)
**Area**: Model Compression
**Keywords**: Module replacement, deterministic mixing, gradient variance, knowledge distillation, Vision Transformer

## TL;DR
DCR mixes teacher and student module outputs via a deterministic annealing weight $\alpha(t)$, eliminating the gradient variance introduced by stochastic gating (e.g., BERT-of-Theseus), and achieves faster convergence and stronger feature alignment in cold-start module replacement scenarios.

## Background & Motivation

**Background**: As training costs continue to rise, model adaptation has become a critical research direction. Predominant approaches include replacing original blocks with smaller proxy modules (compression) and substituting standard self-attention with efficient variants (Linformer/Performer, etc., with $O(n)$ complexity).

**Limitations of Prior Work**: When replacing modules within a frozen pretrained backbone, randomly initialized modules under cold-start conditions produce out-of-distribution features, causing downstream layers to receive anomalous inputs, leading to optimization instability, ineffective gradient updates, and slow recovery.

**Key Challenge**: Knowledge distillation requires expensive full teacher forward passes and enforces rigid feature matching; stochastic replacement methods such as BERT-of-Theseus employ Bernoulli gating $z_\ell(t) \sim \text{Bernoulli}(p(t))$, which introduces substantial gradient variance when $p(t)$ takes intermediate values.

**Goal**: The core problem is "how to stably integrate a randomly initialized new module into a frozen backbone"—i.e., the stability problem of module replacement.

**Key Insight**: Replace stochastic gating with a deterministic mixing weight, theoretically eliminating the gradient variance term induced by gating.

**Core Idea**: Substitute stochastic Bernoulli gating with a deterministic annealing schedule $\alpha(t)$, theoretically eliminating gate-induced gradient variance and achieving faster convergence in practice.

## Method

### Overall Architecture
Given a pretrained network $F$ with $L$ modules, for each module $\ell$ in the replacement subset $\mathcal{I} \subseteq \{1,...,L\}$, the frozen teacher module $T_\ell$ is retained while the student module $S_\ell(\cdot;\theta_\ell)$ is trained. DCR performs deterministic mixing on the residual branch:

$$x_{\ell+1}(t) = x_\ell(t) + [\alpha(t) T_\ell(h_\ell(t)) + (1-\alpha(t)) S_\ell(h_\ell(t); \theta_\ell)]$$

where $\alpha(t) \in [0,1]$: $\alpha(0) = 1$ (pure teacher) $\to$ $\alpha(T) = 0$ (student takes full control), and $h_\ell = \text{LN}(x_\ell)$.

### Key Designs

1. **Deterministic Gate**:

    - **Function**: Linearly mixes teacher and student outputs using a global deterministic weight $\alpha(t)$.
    - **Mechanism**: The aggr20 schedule — during the first 10% of training $\alpha: 1.0 \to 0.3$, during 10%–20% $\alpha: 0.3 \to 0.0$, after which $\alpha = 0$ and the student operates independently.
    - **Design Motivation**: The deterministic gate entirely eliminates gate-induced gradient variance. For Theseus hard gating $z \sim \text{Bernoulli}(p)$, the gating component of gradient variance is $p(1-p)\mathbb{E}\|a\|^2$; DCR's deterministic $\alpha$ renders the conditional variance $\text{Var}(\nabla_{\theta_\ell} L | X) = 0$.

2. **Deep Feature Guidance (DFG)**:

    - **Function**: Adds an auxiliary $L_2$ alignment loss at replacement positions.
    - **Mechanism**: $\mathcal{L}_{\text{DFG}} = \sum_{\ell \in \mathcal{I}} \|S_\ell(h_\ell) - T_\ell(h_\ell)\|_2^2$. Since DCR already computes outputs from both branches, DFG incurs virtually zero additional cost.
    - **Design Motivation**: Unlike standard distillation, DFG does not require a full teacher forward pass; it performs local alignment only at the replaced layers. The DFG weight $\lambda$ is annealed in synchrony with $\alpha$.

3. **Curvature Bias Elimination**:

    - **Function**: Eliminates curvature bias introduced when stochastic mixtures pass through nonlinear layers.
    - **Mechanism**: Theseus's random selection introduces an expectation bias after a nonlinear function $\psi$: $|\mathbb{E}[\psi(Y)] - \psi(\mu)| \leq \frac{M}{2} p(1-p)\|\Delta\|^2$. DCR's deterministic path ensures $\mathbb{E}[\psi(Y_\alpha)] = \psi(Y_\alpha)$, with no mixing bias.

### Loss & Training
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}}(\hat{y}, y^\star) + \lambda \mathcal{L}_{\text{DFG}}$$

Training proceeds in two stages: (1) classification head warmup for 6 epochs, lr $1 \times 10^{-3}$; (2) full model training for 50 epochs, lr $5 \times 10^{-4}$, AdamW, weight decay 0.05, gradient clipping 1.0, batch size 128, label smoothing 0.1, mixed precision BF16.

## Key Experimental Results

### Main Results
Experiments are conducted on ImageNet-pretrained ViT-Small with attention module self-replacement after fine-tuning on CIFAR-100.

| Method | Gradient Variance | Extra Computation | Feature Matching Required | Convergence Speed |
|--------|------------------|-------------------|--------------------------|-------------------|
| Knowledge Distillation | Low | High (full teacher forward) | Yes (rigid) | Medium |
| Theseus (stochastic) | High (gate-induced) | Low | No | Slow |
| DCR (Ours) | **Low (deterministic)** | **Low (teacher at replaced layers only)** | **No (optional DFG)** | **Fast** |

DCR+DFG reaches the target accuracy fastest in both epoch count and wall-clock time, with final accuracy approximately 78–80%. DCR+DFG achieves substantially higher teacher-student interface cosine similarity across all layers (Block 0/7/11) compared to the stochastic baseline.

### Ablation Study

| Configuration | Interface Alignment Quality | Convergence Speed | Notes |
|---------------|-----------------------------|-------------------|-------|
| DCR + DFG | Highest | Fastest | Deterministic mixing + feature guidance |
| DCR only | High | Fast | Pure deterministic mixing |
| GUM (Gumbel) | Medium | Medium | Soft stochastic gating retains variance |
| GUM + DFG | Medium-high | Medium | Soft gating + feature guidance |
| BERN (Theseus) | Low | Slow | Hard gating with highest variance |
| Student-only | Low | Slowest | No progressive replacement |

### Key Findings
- Deterministic mixing ensures downstream layers receive in-distribution features from the outset, avoiding the gate-induced gradient starvation that delays deep-layer convergence in GUM/BERN.
- DFG yields the most pronounced improvement in deep layers (Block 11), confirming the synergistic effect between near-zero-cost feature guidance and deterministic mixing.
- DCR demonstrates clear advantages even in small-scale, non-compute-saturated experiments.

## Highlights & Insights
- **Feature alignment with zero extra forward passes**: DCR already computes both teacher and student outputs for mixing; DFG exploits this "free" signal for alignment, in sharp contrast to standard distillation, which requires a full teacher forward pass.
- **Theory-driven method design**: A complete variance decomposition (Propositions 1–4) not only explains why the method works but precisely quantifies the advantage of DCR over stochastic methods by $p(1-p)\mathbb{E}\|a\|^2$.
- **Transferability to heterogeneous operator replacement**: Although experiments validate the approach in self-replacement settings (attention → re-initialized attention), both the method and theory are designed for heterogeneous replacement (attention → Linformer/Performer).

## Limitations & Future Work
- Validation is limited to a small-scale setting (ViT-Small + CIFAR-100) with single-seed experiments; results on large-scale models and datasets are absent.
- Self-replacement experiments (homogeneous replacement) isolate the stability problem but do not validate the actual effectiveness of heterogeneous replacement.
- The global $\alpha(t)$ schedule does not account for inter-layer differences; adaptive per-layer scheduling (based on interface similarity) may further improve performance.
- Behavior under Batch Norm architectures or post-norm Transformers is not discussed.
- As a workshop paper, experimental comparisons are not comprehensive, lacking stronger baselines such as Net2Net and CKA matching.

## Related Work & Insights
- **vs. BERT-of-Theseus**: Theseus achieves progressive replacement via Bernoulli stochastic gating; DCR demonstrates that stochasticity itself is the source of gradient variance and directly eliminates this issue through deterministic mixing.
- **vs. Standard Knowledge Distillation**: Distillation requires full teacher forward passes and rigid feature matching; DCR requires only local teacher computation at the replaced layers, offering greater advantages in compute-saturated scenarios.
- **vs. Theseus-Gumbel**: Although soft Gumbel-Softmax gating allows gradient flow, it still retains an additional variance term $\text{Var}(r)\mathbb{E}\|a\|^2$.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — Eliminating gating variance through deterministic mixing is an intuitively simple yet theoretically rigorous contribution.
- **Experimental Thoroughness**: ⭐⭐ — Small-scale single-seed experiments; heterogeneous replacement and large models are not addressed.
- **Writing Quality**: ⭐⭐⭐⭐ — Theoretical derivations are clear; experimental limitations are acknowledged honestly.
- **Value**: ⭐⭐⭐ — Provides a theoretical foundation for module replacement, but large-scale feasibility requires validation in follow-up work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Elastic ViTs from Pretrained Models without Retraining](elastic_vits_from_pretrained_models_without_retraining.md)
- [\[NeurIPS 2025\] Bézier Splatting for Fast and Differentiable Vector Graphics Rendering](bezier_splatting_for_fast_and_differentiable_vector_graphics_rendering.md)
- [\[NeurIPS 2025\] Understanding Differential Transformer Unchains Pretrained Self-Attentions](understanding_differential_transformer_unchains_pretrained_self-attentions.md)
- [\[ICCV 2025\] Bridging Continuous and Discrete Tokens for Autoregressive Visual Generation](../../ICCV2025/model_compression/bridging_continuous_and_discrete_tokens_for_autoregressive_visual_generation.md)
- [\[NeurIPS 2025\] Data Efficient Adaptation in Large Language Models via Continuous Low-Rank Fine-Tuning](data_efficient_adaptation_in_large_language_models_via_continuous_low-rank_fine-.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Spherical Steering: Geometry-Aware Activation Rotation for Language Models
description: >-
  [ICML 2026][LLM Evaluation][Activation Steering] This paper proposes Spherical Steering: rotating activation vectors along geodesics on the unit hypersphere of LLM hidden states toward a "truthfulness direction" estimate…
tags:
  - "ICML 2026"
  - "LLM Evaluation"
  - "Activation Steering"
  - "Hyperspherical Geometry"
  - "Slerp Geodesic"
  - "vMF Confidence Gating"
  - "Norm Preservation"
date: 2026-05-08
content_hash: 3fd6a5b2e741c2f1
---

# Spherical Steering: Geometry-Aware Activation Rotation for Language Models

**Conference**: ICML 2026  
**arXiv**: [2602.08169](https://arxiv.org/abs/2602.08169)  
**Code**: https://github.com/chili-lab/Spherical-Steering (Available)  
**Area**: Interpretability / Activation Editing / Inference-time Intervention / LLM Alignment  
**Keywords**: Activation Steering, Hyperspherical Geometry, Slerp Geodesic, vMF Confidence Gating, Norm Preservation

## TL;DR
This paper proposes Spherical Steering: rotating activation vectors along geodesics on the unit hypersphere of LLM hidden states toward a "truthfulness direction" estimated from contrastive samples. Unlike traditional additive activation steering, this method preserves activation norms while significantly improving multiple-choice accuracy on benchmarks like TruthfulQA, COPA, and StoryCloze (+10% range) without degrading the quality of open-ended generation.

## Background & Motivation

**Background**: To control LLM behavior without retraining, the mainstream approach is *activation steering*—estimating a "steering vector" $\mu$ from a batch of (positive, negative) contrastive samples and adding it directly to token activations at certain layers: $h' = h + \lambda \mu$. Representative methods include CAA and ITI.

**Limitations of Prior Work**: This additive operation suffers from severe *scale sensitivity*. If $\lambda$ is too small, there is no effect; if $\lambda$ is large, the hidden state norm $\|h\|$ is significantly distorted—the equation $\|h'\|^2 = \|h\|^2 + 2\lambda\mu^\top h + \lambda^2$ shows that norm changes depend on both $\lambda$ and the alignment between $\mu$ and $h$, making them uncontrollable. Consequently, while multiple-choice accuracy may increase, open-ended generation quality (TRUE×INFO) often drops, leading to "over-conservatism" or even representation collapse.

**Key Challenge**: Modern LLMs commonly normalize activation magnitudes using RMSNorm/LayerNorm, implying that *direction is the primary degree of freedom carrying semantics*. Additive steering perturbs magnitudes freely, conflicting with the geometric priors of the architecture.

**Goal**: Design a *geometry-consistent* inference-time intervention primitive that is training-free like addition but strictly preserves $\|h\|$, avoiding damage to the geometric priors of normalization layers.

**Key Insight**: The authors made a critical empirical observation (Fig. 3)—on TruthfulQA, the $\ell_2$ norm curves of last-token activations for "correct" and "incorrect" answers almost overlap across all 32 layers (difference <1%), but their *directions* differ significantly. This indicates that truthfulness signals are encoded in direction rather than magnitude.

**Core Idea**: Normalize activations to the unit hypersphere $\mathbb{S}^{d-1}$, rotate them toward a target direction $\mu_T$ along a geodesic (great circle) using Slerp, and finally scale back to the original norm. This norm-preserving rotational intervention essentially replaces "addition in $\mathbb{R}^d$" with "rotation on $\mathbb{S}^{d-1}$".

## Method

### Overall Architecture
The method consists of three training-free steps:
1. **Offline Prototype Construction**: Run the model on a contrastive dataset $\mathcal{D}=\{(x_i, y_i^+, y_i^-)\}$, extract last-token activations from layer $l$, compute the mean difference, and normalize it to obtain the "truthfulness axis" $\mu_T^{(l)}$ for that layer.
2. **Inference-time Rotation**: For each intervened layer $l$ and each decoded token $j$, normalize the current activation $h_j^{(l)}$ to the sphere, rotate it toward $\mu_T^{(l)}$ via Slerp with a gating-determined step size $t_j^{(l)} \in [0,1]$, and restore the original norm.
3. **vMF Confidence Gating**: Use the exponential form of the von Mises–Fisher distribution to perform a two-class softmax on the similarity between the current direction and $(\mu_T, \mu_H = -\mu_T)$, obtaining a "hallucination bias" confidence $\delta$. This is mapped to the intervention strength $t$ via threshold $\beta$ and scaling $\alpha$—applying strong rotation only when the model "appears to hallucinate."

### Key Designs

1. **Hyperspherical Prototype and Contrastive Mean Direction**:
    - **Function**: Extract the "truthfulness direction unit vector" $\mu^{(l)}$ for a layer from contrastive samples at once.
    - **Mechanism**: Feed concatenated sequences $x_i \| y_i^\pm$ for each $(x_i, y_i^+, y_i^-)$ into the model and take the last-token representation $z_i^{(l)\pm}$ of layer $l$. Compute the difference between positive and negative means $\Delta^{(l)} = m_+^{(l)} - m_-^{(l)}$ and normalize it: $\mu^{(l)} = \Delta^{(l)}/\|\Delta^{(l)}\|$. This step is offline, keeps model weights frozen, and is performed only once.
    - **Design Motivation**: The mean difference automatically suppresses shared context between positive and negative samples, highlighting the discriminative component of "truth vs. falsehood." Normalization is required because subsequent operations occur on $\mathbb{S}^{d-1}$, necessitating a pure "direction" rather than a scaled offset. This is lighter than ITI's per-head probes and more geometrically consistent than CAA's additive approach.

2. **Geodesic Rotation = Slerp + Norm Restoration**:
    - **Function**: Rotate $h^{(l)}$ along the shortest spherical path toward $\mu_T$ by a ratio $t$, then restore the original magnitude.
    - **Mechanism**: Compute the angle $\theta = \arccos(\mu_T^\top \hat h^{(l)})$, then apply Shoemake’s (1985) spherical linear interpolation: $\hat h^{(l)\prime} = \frac{\sin((1-t)\theta)}{\sin\theta}\hat h^{(l)} + \frac{\sin(t\theta)}{\sin\theta}\mu_T$. Finally, set $h^{(l)\prime} = \|h^{(l)}\|\hat h^{(l)\prime}$. When $t=0$, there is no change; when $t=1$, it aligns perfectly with $\mu_T$. Degenerate cases ($\theta=0/\pi$) are handled separately.
    - **Design Motivation**: Slerp provides the path of minimal angular change for a given step $t$, meaning "maximum semantic alignment with minimal directional perturbation." Meanwhile, $\|h^{(l)\prime}\| \equiv \|h^{(l)}\|$ holds strictly, bypassing the issue of uncontrolled $\|h\|$ in additive steering and conforming to the architectural prior that direction carries information after RMSNorm. Unlike Angular Steering which rotates in a fixed 2D plane, this method performs hyperspherical rotation in the original $d$-dimensional space without relying on PCA projections.

3. **Input-Adaptive Step Size via vMF Confidence Gating**:
    - **Function**: Vary $t$ based on the token—no intervention if the model is already in the "truthful" hemisphere, and stronger intervention as it leans toward the "hallucination" hemisphere.
    - **Mechanism**: Use the exponential term of the vMF density $f(u;m,\kappa)\propto\exp(\kappa m^\top u)$ as a prototype score. Perform a two-class softmax over $(\mu_T, \mu_H)$ to get $p_T, p_H$, and define $\delta = p_H - p_T \in [-1,1]$. Apply threshold $\beta$ and scaling $\alpha$: $t = \mathrm{clip}(\alpha \cdot \frac{\delta-\beta}{1-\beta}, 0, 1)$, where $t=0$ if $\delta \le \beta$.
    - **Design Motivation**: Compared to a uniform $t$ for all tokens, gating provides two benefits (Ablation Fig. 5): higher MC accuracy peaks with wider intervals, and stable TRUE×INFO even at high intensities (stable at $\alpha=1.0$), whereas ungated performance collapses at $\alpha > 0.6$. $\kappa$ controls the steepness of the confidence curve. This design follows the principle of "applying intervention only where a hallucination is likely."

### Inference Flow
For each decoding step and each selected layer $l$ in $\mathcal{L}=\{l_1,\dots,l_K\}$: Extract $h_j^{(l)}$ → Normalize → Compute $s_T, s_H$ → vMF gate computes $t_j^{(l)}$ → If $t>0$, perform Slerp rotation and restore norm; otherwise, pass through. The complexity involves a few dot products and sin/cos operations, which are negligible compared to the forward pass.

## Key Experimental Results

### Main Results

On TruthfulQA (LLaMA-3.1-8B-Instruct), Spherical Steering achieves the best performance across multiple-choice metrics (MC1/MC2/MC3) and open-ended generation (TRUE×INFO) simultaneously—whereas additive baselines like ITI/CAA improve MC at the cost of TRUE×INFO, showing a typical trade-off.

| Model | Method | MC1 | MC2 | MC3 | TRUE×INFO |
|------|------|-----|-----|-----|-----------|
| LLaMA-3.1-8B-Instruct | Baseline | 34.15 | 53.32 | 27.02 | 48.24 |
| LLaMA-3.1-8B-Instruct | ITI | 37.70 | 58.09 | 30.12 | 40.31 ↓ |
| LLaMA-3.1-8B-Instruct | CAA | 35.99 | 56.26 | 29.36 | 49.66 |
| LLaMA-3.1-8B-Instruct | SADI-HEAD | 38.53 | 56.03 | 30.57 | 51.18 |
| LLaMA-3.1-8B-Instruct | **Spherical (Ours)** | **49.95** | **68.51** | **41.05** | **54.63** |
| Qwen-2.5-7B-Instruct | Baseline | 35.87 | 54.95 | 26.62 | 74.40 |
| Qwen-2.5-7B-Instruct | ITI | 40.15 | 58.93 | 30.26 | 67.82 ↓ |
| Qwen-2.5-7B-Instruct | **Spherical (Ours)** | **48.71** | **66.90** | **39.16** | **77.84** |

Zero-shot evaluation across 6 multi-choice benchmarks (LLaMA-3.1-8B-Instruct):

| Method | TruthfulQA | COPA | StoryCloze | MMLU | Wino. | BoolQ | Avg. |
|------|------------|------|------------|------|-------|-------|------|
| Baseline | 34.15 | 83.00 | 74.72 | 60.60 | 50.81 | 80.12 | 63.90 |
| ITI | 37.70 | 83.00 | 75.12 | 60.90 | 51.85 | 81.53 | 65.02 |
| CAA | 35.99 | 84.00 | 79.02 | 60.70 | 51.93 | 82.42 | 65.68 |
| SADI-HEAD | 38.53 | 84.00 | 75.72 | 60.66 | 51.85 | 80.20 | 65.16 |
| **Spherical (Ours)** | **49.95** | **95.00** | **89.08** | **62.05** | **52.72** | **82.94** | **71.96** |

Average absolute Gain of +6.28%, with > +10% on COPA/StoryCloze.

### Ablation Study

| Configuration | MC1 (TruthfulQA, LLaMA) | TRUE×INFO | Description |
|------|--------------------------|-----------|------|
| K=1 layer | 45.41 | 52.16 | Single layer rotation: MC already near peak |
| K=2 layers | 47.62 | 73.93 | Additional layers mainly recover generation quality (INFO 62.9→90.3) |
| K=3 layers | 47.13 | **74.43** | Best overall balanced point |
| K=4 layers | 41.37 | 70.62 | Excessive intervention hurts MC |
| K=5 layers | 41.37 | 70.09 | Same as above |
| Ungated rotation (α=1.0) | — | Sharp decrease | Generation quality collapses at high α |
| **vMF gated (α=1.0)** | — | Remains stable | Gating significantly extends the usable range of α |

### Key Findings
- **Geometric Insight**: Fig. 3 shows that the activation norms for truthful vs. hallucinated instances overlap almost perfectly across all layers (<1% difference), proving that truthfulness signals reside in direction rather than magnitude, empirically validating the norm-preserving design.
- **Collapse-Efficiency Advantage**: Fig. 4 shows that for the same decrease in effective rank (Δrank≈50), rotation gains 8–10% more MC accuracy than addition. While addition sees TRUE×INFO collapse after a slight rank decrease, rotation sustains quality gains across a wide range of rank drops.
- **Asymmetric Effects of Multi-layer Intervention**: Moving from K=1 to K=3 keeps MC almost constant (+2.2%) but jumps INFO from 62.9% to 92.7%. The authors suggest middle layers govern semantic discrimination (MC signal), while later layers govern token-level generation dynamics (INFO signal).
- **Orthogonality to 5-shot ICL**: When combined with ICL, ITI drops TRUE×INFO from 38.9 to 37.3, while Spherical simultaneously pushes MC1 to 52.4% and TRUE×INFO to 42.8%, indicating that geometric intervention and prompt engineering operate via independent mechanisms.
- **High Sample Efficiency**: Using only 25 contrastive samples increases MC1 on LLaMA from 36.3% to 51.5% (±2.2), with variance shrinking rapidly as samples increase.

## Highlights & Insights
- **Reframing "Addition in $\mathbb{R}^d$" as "Rotation on $\mathbb{S}^{d-1}$" is a natural yet overlooked perspective**: Since architectures like RMSNorm already stabilize norms, the remaining degrees of freedom for free perturbation are purely directional. This work fully implements this observation as an intervention primitive.
- **Slerp appears in LLM steering for the first time in a closed-form, training-free manner**: Unlike methods like HPR that learn a Householder reflection, Spherical achieves both "geometric consistency" and "zero training."
- **vMF Gating is a lightweight plugin transferable to any steering method**: It essentially uses the interpretable confidence of direction to dynamically adjust intensity, potentially applicable to CAA, ITI, or SAE-based interventions to decouple norm and direction control.
- **Strong "Pareto Improvement" arguments**: Fig. 1(a) plots MC accuracy against TRUE×INFO, showing all baselines stuck on a trade-off curve while the proposed method jumps to the upper-right—a compelling demonstration of breaking the trade-off.
- **The concept of collapse-efficiency is methodologically valuable**: Instead of just end-point metrics, the introduction of "performance gain per unit of rank decrease" provides a comparative geometric efficiency metric for future intervention research.

## Limitations & Future Work
- **Prototypes rely on binary contrastive data**: Currently supports binary concepts (truthful/hallucinated, safe/unsafe). Extending this to "multi-class fine-grained concepts" (e.g., multiple emotions or styles) would require multi-prototype or multi-axis geometry, which is not discussed.
- **Strong assumption of uni-axial $\mu_T$ and its antipode $\mu_H = -\mu_T$**: In reality, "truth" may not be perfectly antipodal to "hallucination," potentially causing failure in tasks with mixed correct/incorrect answers.
- **Layer selection still relies on grid search**: While the method selects layers $\mathcal{L}=\{l_1,\dots,l_K\}$, the optimal combination is determined empirically (e.g., layer 24 for LLaMA), lacking a principled selection criterion.
- **Validated only on 7–8B Instruct models**: Robustness of the hyperspherical assumption on base models, larger scales (30B+), or MoE architectures is unknown.
- **Gating hyper-parameters**: $\kappa, \alpha, \beta$ collectively determine the gating shape, representing a non-trivial tuning space. Automatically estimating $\kappa$ from contrastive samples (vMF MLE) would be more efficient.
- **Improvement Directions**: (i) Expanding the single axis $\mu_T$ into low-rank multi-axial geometry for composite concept steering; (ii) Using SAE features as prototype directions; (iii) Replacing "geodesics" with Riemannian gradient flow for multi-step iterative rotation.

## Related Work & Insights
- **vs. CAA (Rimsky et al., 2024)**: CAA uses layer-wise addition $h + \lambda\mu$. This work replaces it with Slerp rotation to preserve the norm. On LLaMA, CAA achieves MC1=35.99 / TRUE×INFO=49.66, while this work achieves 49.95 / 54.63.
- **vs. ITI (Li et al., 2023)**: ITI selects "truthful heads" via per-head linear probes and applies small additions. This work uses whole-layer directional rotation. While ITI's TRUE×INFO drops to 40.31 on LLaMA, this work increases it to 54.63.
- **vs. Angular Steering (Vu & Nguyen, 2025)**: Also an angular intervention, but it projects activations into a fixed 2D plane first, relying on low-dimensional kernels. This work operates directly on the raw $d$-dimensional spherical geodesic without PCA.
- **vs. HPR (Pham & Nguyen, 2024)**: HPR uses Householder reflections and a trained angle prediction network. This work is closed-form and training-free, sacrificing per-input angle learning flexibility for lightweight vMF-based adaptation.
- **vs. ReFT / LoFiT (Wu et al., 2024; Yin et al., 2024)**: These involve representation fine-tuning with lightweight modules. This work takes the "structured intervention" idea to a training-free extreme using pure geometric priors.
- **Insight**: This "Sphere + Geodesic + Confidence Gate" combination can be transferred to *any* scenario where semantics are directionally encoded—image tokens in VLMs, noise embeddings in diffusion models, or graph representations. Any layer following LayerNorm/RMSNorm where editing is required should consider "Addition vs. Rotation" for geometric consistency.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Replacing addition with rotation is a single-point idea, but the complete combination of hyperspherical geometry, Slerp, and vMF gating with rigorous geometric proof makes it a clean and effective innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 6 MC benchmarks, open-ended generation, collapse-efficiency, and ablations for layers/gating/ICL/sample size. Lacks validation on larger models.
- **Writing Quality**: ⭐⭐⭐⭐ The logical chain from motivation to geometric insight to method to validation is very smooth. Fig. 1 effectively illustrates "breaking the trade-off."
- **Value**: ⭐⭐⭐⭐ Provides a plug-and-play, training-free, norm-preserving steering primitive. The collapse-efficiency metric is of methodological significance for future intervention studies.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Test-time Diverse Reasoning by Riemannian Activation Steering](../../AAAI2026/llm_evaluation/test-time_diverse_reasoning_by_riemannian_activation_steering.md)
- [\[ICML 2026\] Top-W: Geometry-Aware Decoding with Wasserstein-Regularized Truncation and Mass Penalties for LLMs](geometry-aware_decoding_with_wasserstein-regularized_truncation_and_mass_penalti.md)
- [\[ICML 2026\] PoliticsBench: Benchmarking Political Values in Large Language Models with Multi-Stage Roleplay](politicsbench_benchmarking_political_values_in_large_language_models_with_multi-.md)
- [\[ICML 2026\] Investigating Advanced Reasoning of Large Language Models via Black-Box Environment Interaction](investigating_advanced_reasoning_of_large_language_models_via_black-box_environm.md)
- [\[ICML 2026\] AGZO: Activation-Guided Zeroth-Order Optimization for LLM Fine-Tuning](agzo_activation-guided_zeroth-order_optimization_for_llm_fine-tuning.md)

</div>

<!-- RELATED:END -->

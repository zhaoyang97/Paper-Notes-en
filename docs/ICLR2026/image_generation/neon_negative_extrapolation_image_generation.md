---
title: >-
  [Paper Note] Neon: Negative Extrapolation From Self-Training Improves Image Generation
description: >-
  [ICLR 2026 Oral][Image Generation][self-training] Neon is proposed as a post-processing method requiring <1% additional training compute: the model is first fine-tuned on its own synthetic data (causing degradation), then negatively extrapolated away from the degraded weights. The paper proves that mode-seeking samplers cause anti-alignment between synthetic and real data gradients, so negative extrapolation is equivalent to optimizing toward the real data distribution. On ImageNet 256×256, xAR-L achieves SOTA FID of 1.02.
tags:
  - ICLR 2026 Oral
  - Image Generation
  - self-training
  - model collapse
  - weight merging
  - negative extrapolation
  - FID
date: 2026-05-08
content_hash: 5599fefa7a8b147f
---

# Neon: Negative Extrapolation From Self-Training Improves Image Generation

**Conference**: ICLR 2026 Oral
**arXiv**: [2510.03597](https://arxiv.org/abs/2510.03597)
**Code**: [github.com/VITA-Group/Neon](https://github.com/VITA-Group/Neon)
**Area**: Image Generation / Self-Training
**Keywords**: self-training, model collapse, weight merging, negative extrapolation, FID

## TL;DR
Neon is proposed as a post-processing method requiring <1% additional training compute: the model is first fine-tuned on its own synthetic data (causing degradation), then negatively extrapolated away from the degraded weights. The paper proves that mode-seeking samplers cause anti-alignment between synthetic and real data gradients, so negative extrapolation is equivalent to optimizing toward the real data distribution. On ImageNet 256×256, xAR-L achieves SOTA FID of 1.02.

## Background & Motivation

**Background**: Scaling generative models is constrained by the scarcity of high-quality training data. Self-training with synthetically generated data is an intuitive solution, but leads to **Model Autophagy Disorder (MAD) / Model Collapse**—rapid degradation in sample quality and diversity.

**Limitations of Prior Work**: (a) SIMS requires 2× inference NFE and large numbers of synthetic samples (100K) with significant additional training compute (20%); (b) DDO requires multiple rounds of iteration (16 rounds × 50K samples); (c) existing methods lack a unified theoretical explanation for why self-training degrades and how degradation can be exploited.

**Key Challenge**: Self-training degradation appears wasteful, yet the degradation direction itself carries information—if the direction of degradation can be understood, it can be exploited in reverse.

**Goal**: Can the degradation signal from self-training be transformed into a self-improvement signal, with theoretical guarantees?

**Key Insight**: The authors observe that mode-seeking samplers (temperature <1, top-k, finite-step ODE solvers) bias synthetic data toward high-probability regions of the model distribution, causing the population gradients of synthetic and real data to be **anti-aligned** ($\cos\varphi < 0$). Consequently, reversing the self-training gradient is equivalent to optimizing toward the real data distribution.

**Core Idea**: Self-training degrades the model, but the direction of degradation is precisely the opposite of the direction of improvement; therefore, negative extrapolation along that direction improves the model.

## Method

### Overall Architecture
Neon is a minimalist three-step post-processing pipeline: (1) generate a small set of synthetic data $S$ (~1K–6K samples) using the base model $\theta_r$; (2) briefly fine-tune on $S$ to obtain a degraded model $\theta_s$; (3) perform weight negative extrapolation: $\theta_{\text{neon}} = (1+w)\theta_r - w\theta_s$, where $w > 0$.

### Key Designs

1. **Negative Extrapolation Weight Merging**

    - **Function**: Obtain an improved model via reverse extrapolation in parameter space.
    - **Mechanism**: $\theta_{\text{neon}} = \theta_r + w(\theta_r - \theta_s)$, i.e., moving in the direction opposite to "base model → degraded model." In practice, this reduces to a single line of code: `merged[k] = base[k] - w * (aux[k] - base[k])`
    - **Design Motivation**: No inference overhead (unlike SIMS which requires 2× NFE), no new real data required, and only a minimal number of synthetic samples needed.

2. **Anti-Alignment Theory (Theorem 1)**

    - **Function**: Proves that under mode-seeking samplers, synthetic data gradients are anti-aligned with real data gradients.
    - **Mechanism**: Defines $r_d = \nabla_\theta \mathcal{L}_{\text{real}}(\theta_r)$ (real data gradient) and $r_s = \nabla_\theta \mathcal{L}_{\text{synth}}(\theta_r)$ (synthetic data gradient), and proves that when the sampler satisfies the monotone reweighting condition and model error $|\varepsilon|$ is sufficiently small, $\cos\varphi = \frac{\langle r_d, r_s \rangle}{\|r_d\| \|r_s\|} < 0$.
    - **Design Motivation**: This explains why self-training causes degradation (updating along $r_s$ effectively increases real data loss) and why negative extrapolation works (reversing $r_s$ approximates an update along $r_d$).

3. **Neon Reduces Population Risk (Theorem 2)**

    - **Function**: Proves that an appropriate $w > 0$ guarantees improvement.
    - **Mechanism**: $\mathcal{L}_{\text{real}}(\theta_{\text{neon}}) < \mathcal{L}_{\text{real}}(\theta_r)$; the optimal $w$ can be predicted from the gradient alignment perspective.
    - **Design Motivation**: Provides rigorous theoretical guarantees rather than a purely empirical approach.

4. **U-Shaped Training Budget Curve**

    - **Function**: Explains the non-monotonic effect of training budget $B$ on performance.
    - **Mechanism**: When $B$ is too small, high variance causes inaccurate estimation of the degradation direction; when $B$ is too large, the Taylor expansion fails as higher-order terms dominate. The optimal range is 1–2% of the base training budget.
    - **Design Motivation**: Guides hyperparameter selection.

### Loss & Training
The fine-tuning stage uses the standard training loss of each respective architecture without modification. $w$ is typically chosen in $[0.5, 1.5]$, with $w \approx 0.8\text{–}1.0$ recommended. For class-conditional models, joint tuning with the CFG scale $\gamma$ is required.

## Key Experimental Results

### Main Results

| Model | Type | Dataset | Baseline FID | Neon FID | Gain |
|------|------|--------|----------|----------|------|
| xAR-L | Flow matching | ImageNet-256 | 1.28 | **1.02** | -20.3% |
| xAR-B | Flow matching | ImageNet-256 | 1.72 | **1.31** | -23.8% |
| VAR d16 | Autoregressive | ImageNet-256 | 3.30 | **2.01** | -39.1% |
| VAR d36 | Autoregressive | ImageNet-512 | 2.63 | **1.70** | -35.4% |
| EDM (cond.) | Diffusion | CIFAR-10 | 1.78 | **1.38** | -22.5% |
| EDM (uncond.) | Diffusion | FFHQ-64 | 2.39 | **1.12** | -53.1% |
| IMM | Moment matching | ImageNet-256 | 1.99 | **1.46** | -26.6% |

### Ablation Study

| Ablation Dimension | Key Findings |
|---------|---------|
| Training budget $B$ | U-shaped curve: optimum at 1–2% of base training budget |
| Merging weight $w$ | $w=-1$ (direct self-training) degrades; $w \in [0.5, 1.5]$ yields consistent improvement |
| Number of synthetic samples | Effective with as few as 1K; diminishing returns beyond 6K |
| Cross-architecture synthesis | Synthetic data generated by one architecture can improve another |

### Efficiency Comparison

| Method | FID (EDM, cond. CIFAR-10) | Extra Compute | Synthetic Samples | Inference Overhead |
|------|--------------------------|---------|----------|---------|
| **Neon** | 1.38 | 1.75% | 6K | None |
| SIMS | 1.33 | 20% | 100K | 2× NFE |
| DDO | 1.30 | 12% | 800K | None |

### Key Findings
- **Cross-Architecture Generality**: The same method applies without modification to four architecture types: diffusion, flow matching, autoregressive, and moment matching.
- **Precision–Recall Trade-off**: Neon primarily improves recall (diversity) at a slight cost to precision, with a net reduction in FID.
- **Mode-Seeking vs. Diversity-Seeking**: When the sampler is diversity-seeking ($\tau > 1$), the gradient alignment flips and negative extrapolation fails—a boundary condition predicted by theory.
- **SOTA**: xAR-L + Neon achieves FID 1.02 on ImageNet 256×256 with only 0.36% additional compute.

## Highlights & Insights
- **The core insight of "degradation as signal" is remarkably elegant**: model collapse is transformed from a problem into a tool, exploiting the information encoded in the degradation direction. The philosophical implication is profound—"knowing the wrong direction is equivalent to knowing the right direction."
- **Minimal implementation**: The entire method reduces to a single line of weight merging code, requiring no modifications to the inference pipeline, no additional data, and no extra inference overhead—methodologically parsimonious to an extreme.
- **Theory and practice in precise correspondence**: The anti-alignment theorem accurately predicts the U-shaped curve and the mode-seeking condition observed empirically, representing a rare example of a theoretically grounded practical method.

## Limitations & Future Work
- **Hyperparameter tuning**: $w$ and $B$ require some tuning (though the effective range is broad), and no automatic selection mechanism is provided.
- **Validated on image generation only**: The approach has not been evaluated on NLP (where language models also use temperature/top-k) or video generation.
- **One-shot correction**: Neon relies on a local first-order approximation and cannot be applied iteratively (repeated negative extrapolation invalidates the Taylor expansion).
- **Precision–diversity trade-off**: The method does not improve peak generation quality, only the proportion of samples exceeding a quality threshold.

## Related Work & Insights
- **vs. SIMS**: SIMS corrects at inference time using the score difference between the base and self-trained models, requiring 2× NFE. Neon performs a one-shot weight merge after training, incurring no inference overhead.
- **vs. DDO (Distillation from Degraded Output)**: DDO requires 16 iterative rounds × 50K samples, with substantially higher compute than Neon.
- **Connection to "Why DPO is Misspecified"**: Both works exploit information encoded in misspecification or degradation directions—DPO's misspecified projection and Neon's degradation gradient both embody the principle of "leveraging bias signals."

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The idea of "reversing degradation" is highly original; both theory and method are novel contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four architectures, three datasets, extensive ablations, and thorough efficiency comparisons.
- Writing Quality: ⭐⭐⭐⭐⭐ Theory is clearly presented, figures are intuitive, and the method is implementable in a single line of code.
- Value: ⭐⭐⭐⭐⭐ High generality, minimal cost, and theoretical guarantees—a strong candidate to become a standard post-training step for generative models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Stochastic Self-Guidance for Training-Free Enhancement of Diffusion Models](stochastic_self-guidance_for_training-free_enhancement_of_diffusion_models.md)
- [\[AAAI 2026\] RetrySQL: Text-to-SQL Training with Retry Data for Self-Correcting Query Generation](../../AAAI2026/image_generation/retrysql_text-to-sql_training_with_retry_data_for_self-correcting_query_generati.md)
- [\[NeurIPS 2025\] Understand Before You Generate: Self-Guided Training for Autoregressive Image Generation](../../NeurIPS2025/image_generation/understand_before_you_generate_self-guided_training_for_autoregressive_image_gen.md)
- [\[ACL 2026\] VisRet: Visualization Improves Knowledge-Intensive Text-to-Image Retrieval](../../ACL2026/image_generation/visret_visualization_improves_knowledge-intensive_text-to-image_retrieval.md)
- [\[CVPR 2026\] Self-Corrected Image Generation with Explainable Latent Rewards](../../CVPR2026/image_generation/self-corrected_image_generation_with_explainable_latent_rewards.md)

</div>

<!-- RELATED:END -->

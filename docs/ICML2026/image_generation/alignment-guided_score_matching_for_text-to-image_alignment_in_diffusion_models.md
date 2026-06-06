---
title: >-
  [Paper Note] Alignment-Guided Score Matching for Text-to-Image Alignment in Diffusion Models
description: >-
  [ICML 2026][Image Generation][Text-image alignment] This paper proposes Alignment-Guided Score Matching (AGSM), which incorporates positive and negative text-image matching signals directly into the diffusion score match…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Text-image alignment"
  - "diffusion models"
  - "score matching"
  - "soft token"
  - "Plackett-Luce"
date: 2026-05-08
content_hash: 174a64a5aaa93d2c
---

# Alignment-Guided Score Matching for Text-to-Image Alignment in Diffusion Models

**Conference**: ICML 2026  
**arXiv**: [2605.30038](https://arxiv.org/abs/2605.30038)  
**Code**: No public code; Project Page: https://jaayeon.github.io/AGSM/  
**Area**: Image Generation / Diffusion Models / Text-Image Alignment  
**Keywords**: Text-image alignment, diffusion models, score matching, soft token, Plackett-Luce  

## TL;DR
This paper proposes Alignment-Guided Score Matching (AGSM), which incorporates positive and negative text-image matching signals directly into the diffusion score matching objective using reward-free Plackett-Luce alignment rewards. By training lightweight soft tokens, AGSM improves T2I semantic alignment while mitigating common issues in SoftREPA, such as repeated generation and counting errors.

## Background & Motivation
**Background**: While diffusion models like SD1.5, SDXL, and SD3 generate high-fidelity images, they still struggle with missing attributes, incorrect counts, and erroneous relationships under complex text constraints. Post-training methods typically employ human preference data or external reward models for diffusion RL/DPO to enhance aesthetic or preference scores.

**Limitations of Prior Work**: Reward-based methods are heavily dependent on reward quality and preference data, and they may not directly address the internal text-image representation alignment of the diffusion process. Reward-free methods like SoftREPA attempt to optimize soft text tokens using contrastive learning to increase mutual information. however, their negative terms continuously push up the denoising error of mismatched pairs, often driving soft tokens into off-manifold regions, resulting in repeated objects, over-counting, and semantic incoherence.

**Key Challenge**: Text-image alignment requires utilizing both positive and negative pairing signals, yet it cannot infinitely penalize negative samples like standard contrastive losses. Score matching in diffusion models requires predicted noise to remain within reasonable denoising dynamics; if negative samples are pushed too far, the training objective may improve surface contrastive loss while compromising generation quality.

**Goal**: The authors aim to retain the lightweight, reward-free, and soft-token advantages of SoftREPA while reformulating the contrastive push-pull as a bounded, normalized guidance term that acts directly on the score matching target, thereby improving T2I alignment without full-model training or reliance on external rewards.

**Key Insight**: This work draws inspiration from Diffusion-DPO/DSPO by modeling preference at the diffusion score level. Unlike methods using human preferences, it constructs an alignment reward using the model's own denoising log-likelihood and employs the Plackett-Luce model to handle one positive text against multiple in-batch negative texts.

**Core Idea**: Incorporate the principle that "positive samples should move toward higher alignment rewards and negative samples should be boundedly pushed away" into the score matching target noise correction, rather than using an unbounded contrastive loss to directly increase negative sample denoising error.

## Method
AGSM freezes the main diffusion model and only trains a small set of soft tokens. During training, an image $x_i$ and its original text $c_i$ in a batch form a positive pair, while pairings with other texts $c_j$ form negative pairs. Instead of merely minimizing the denoising prediction error for positive pairs, the model modifies the target noise based on the PL reward gradient: the positive target shifts toward a more aligned direction, while the negative target shifts in the opposite direction, with the offset controlled by normalized weights to prevent infinite divergence.

### Overall Architecture
The input consists of an image-text dataset, a frozen diffusion backbone, positive and negative soft tokens $\psi^+$ and $\psi^-$, EMA soft tokens, and guidance scales $\gamma^+$, $\gamma^-$. Each training iteration samples a batch, using matching indices $(x_i, c_i)$ as positive pairs and cross-indices $(x_i, c_j)$ as negative pairs. The same timestep and noise are used for all pairs to compute noise predictions under current and EMA soft tokens. EMA predictions are used to estimate alignment rewards in the form of denoising errors, which are then passed through softmax to obtain PL weights. Finally, the target noise is constructed to update the corresponding soft tokens. During inference, negative tokens are discarded, and only positive soft tokens are used for conditional and unconditional generation.

### Key Designs
1. **Endogenous Alignment Reward based on Plackett-Luce**:
    - **Function**: Assigns alignment scores for an image across multiple text candidates without an external reward model.
    - **Mechanism**: The reward is defined as the expected log-likelihood of the diffusion reverse transition, implemented as the negative denoising error: $r(x_t,c)=-\frac{A(t)}{2}\|\epsilon_{\theta}^{\hat{\psi}}(x_t,t,c)-\epsilon\|_2^2$. The PL model $p(z=1|x_t,c)=\frac{\exp(r(x_t,c))}{\sum_i\exp(r(x_t,c_i))}$ then represents the probability that the current text matches the image better than other candidates.
    - **Design Motivation**: PL naturally handles multiple negative texts for a single positive sample, making it more suitable for in-batch negatives than pairwise BT models; the reward originates from the model's own score matching, eliminating the need for expensive preference annotations.

2. **Integrating Alignment Preference into the Score Matching Target**:
    - **Function**: Directly alters the diffusion score through the training objective rather than using unbounded contrastive push-pull at the representation layer.
    - **Mechanism**: Positive pairs use $p_t^+(x_t|c)\propto p_t(x_t|c)p(z=1|x_t,c)^{\gamma^+}$, and negative pairs use $p_t^-(x_t|c)\propto p_t(x_t|c)p(z=1|x_t,c)^{-\gamma^-}$. Taking the gradient, the target score equals the original diffusion score plus $\gamma_z\nabla\log p(z=1|x_t,c)$, which corresponds to adding/subtracting a correction term formed by the PL reward differential to the target noise.
    - **Design Motivation**: This ensures negative samples are not infinitely degraded but adjusted along a normalized, finite preference gradient. Since training remains within the score matching framework, it stays closer to the original dynamics of the diffusion model.

3. **Decoupled Positive/Negative Soft Tokens and EMA Stability**:
    - **Function**: Separately learns tokens for enhancing positive semantics and handling negative semantics to avoid interference from opposing gradients on a shared token.
    - **Mechanism**: Positive pairs update $\psi^+$ and negative pairs update $\psi^-$; rewards and target corrections use EMA soft token predictions to reduce early-stage training noise. Final sampling only uses $\psi^+$ to prevent negative tokens from excessively suppressing backgrounds and details in the CFG unconditional branch.
    - **Design Motivation**: Experiments show shared tokens significantly degrade ImageReward and CLIP, and using negative tokens during inference harms image diversity. Decoupled training with positive-only inference represents a trade-off between stability and generation quality.

### Loss & Training
The primary loss remains the mean squared error of noise prediction, but the target changes from the true noise $\epsilon_t$ to the alignment-corrected $\epsilon_{\mathrm{tgt}}$. For SD1.5 and SDXL, $\gamma^+=1, \gamma^-=1$ are used; for SD3, $\gamma^+=1, \gamma^-=0.1$. The batch size is 16, with 3 in-batch negatives per positive pair. SD1.5 and SD3 are trained for 100k iterations, while SDXL is trained for 1k iterations. The optimizer is AdamW with a learning rate of $10^{-3}$ and weight decay of $10^{-4}$. Soft tokens are added to the UNet Down/Middle blocks for SD1.5/SDXL and the top 5 transformer layers for SD3.

## Key Experimental Results

### Main Results
T2I generation is evaluated on COCO-val 5K and GenEval. AGSM does not outperform SoftREPA on every preference score but yields a better FID trade-off while significantly improving upon SoftREPA's failure cases.

| Model / Method | ImageReward | CLIP | HPSv2 | FID | GenEval Mean | Counting | Conclusion |
|----------------|-------------|------|-------|-----|--------------|----------|------------|
| SD1.5          | 17.72       | 26.40| 25.08 | 24.59| -            | -        | Weak baseline alignment |
| SD1.5 + SoftREPA| 40.02       | 27.09| 26.05 | 29.25| -            | -        | High preference, but FID degrades |
| SD1.5 + Ours   | 34.50       | 27.23| 25.66 | 25.94| -            | -        | Higher CLIP, much better FID than SoftREPA |
| SDXL           | 75.06       | 26.76| 27.35 | 24.69| -            | -        | Strong baseline |
| SDXL + Ours    | 84.22       | 26.86| 27.96 | 24.83| -            | -        | Improved alignment with minimal FID loss |
| SD3            | 94.27       | 26.30| 28.09 | 31.59| 0.68         | 0.56     | Strong base SD3 |
| SD3 + SoftREPA | 108.50      | 26.91| 28.91 | 36.21| 0.70         | 0.29     | High ImageReward, severe counting degradation |
| SD3 + Ours     | 103.30      | 27.00| 28.22 | 34.08| 0.72         | 0.64     | Counting up from 0.29 to 0.64, reduced repetition |

### Ablation Study
The paper analyzes training strategies, sampling strategies, negative guidance scales, PL vs. BT, and complementarity with diffusion RL.

| Configuration | Key Metric | Description |
|---------------|------------|-------------|
| Train $\psi^+$ only on $\mathcal{D}^+$ | ImageReward 94.79, CLIP 26.93, FID 34.46 | Positive samples alone provide gains but miss negative pair info |
| Shared $\psi$ on $\mathcal{D}^+, \mathcal{D}^-$ | ImageReward 47.33, CLIP 25.68, FID 31.20 | Conflicting signals; both quality and alignment degrade |
| Separate $\psi^+, \psi^-$ (Ours) | ImageReward 103.30, CLIP 27.00, FID 34.08 | Decoupling positive and negative tokens is key |
| Inference using $\psi^+, \psi^-$ | ImageReward 84.53, FID 36.47 | Negative tokens in the unconditional branch over-suppress detail |
| Inference using $\psi^+$ only (Ours) | ImageReward 103.30, FID 34.08 | Training with negative tokens but inferring without them is more stable |
| SD3 $\gamma^-=0$ | ImageReward 94.79 | Limited gains without negative guidance |
| SD3 $\gamma^-=0.1$ | ImageReward 103.30, CLIP 27.00 | Moderate negative guidance is optimal |
| BT loss | ImageReward 29.67, CLIP 27.13, FID 24.76 | Pairwise form is inferior to multi-negative PL |
| PL loss (Ours) | ImageReward 34.50, CLIP 27.23, FID 25.94 | Multi-candidate normalization is better for in-batch negatives |

### Key Findings
- SoftREPA training loss continues to decrease, but validation ImageReward eventually drops, indicating over-optimization of the contrastive objective; AGSM is more stable in later stages and less dependent on early stopping.
- In image editing, AGSM forms a better Pareto front between CLIP alignment and background LPIPS. For example, with SD3 RF-Inversion, incorporating AGSM increases ImageReward from 128.0 to 132.3 and CLIP/Whole from 27.26 to 29.07, with a slight increase in SSIM.
- AGSM is complementary to DiffusionDPO, SPO, and InPO. On SD1.5, adding AGSM to DiffusionDPO improves ImageReward from 29.09 to 42.47; for InPO, it increases from 62.12 to 67.95.
- Effective for long prompts. In UniGenBench++, SD3 base ImageReward is 82.33, SoftREPA is 90.63, and AGSM reaches 98.01, achieving the highest scores in CLIP, PickScore, and HPSv2.

## Highlights & Insights
- The paper accurately identifies the core issue of SoftREPA: negative samples should not simply be pushed as far as possible. In diffusion models, infinitely increasing the denoising error of mismatched pairs destroys the score manifold, manifesting as repetition and over-counting.
- A notable strength of AGSM is treating alignment as a directional correction of the score target rather than an additional representation loss. This aligns the method more closely with diffusion training dynamics, explaining its superior stability.
- The design of decoupled positive and negative soft tokens is highly practical: negative samples provide alignment boundaries during training, but negative tokens should not be allowed to "delete" content during inference via CFG. This insight is valuable for post-training other soft prompt or adapter-based generative models.

## Limitations & Future Work
- The method primarily trains soft tokens; while lightweight, this restricts the range of possible corrections. For complex relationships requiring significant changes to model knowledge or compositional ability, soft tokens may be insufficient.
- AGSM still relies on automatic metrics from existing benchmarks (e.g., ImageReward, CLIP, HPSv2, GenEval), which may not fully represent true human preference or safety.
- Negative samples are derived from in-batch text mismatches; though efficient, they may not cover the most challenging semantic confusions. Future work could construct hard negatives, such as texts with modified counts, colors, or spatial relations.
- Using only positive tokens at inference is empirically optimal, but whether negative tokens could play a role in fine-grained control, concept erasure, or safety filtering warrants separate exploration.

## Related Work & Insights
- **vs. SoftREPA**: SoftREPA uses contrastive learning to optimize soft tokens, which is efficient but over-penalizes negative samples; AGSM converts positive and negative guidance into bounded score matching corrections, reducing repetition and counting failures.
- **vs. Diffusion-DPO / DSPO**: These rely on human preferences or preference pair modeling; AGSM uses endogenous denoising likelihood to construct reward-free alignment signals focused on representation/text-image alignment.
- **vs. CaPO / RankDPO / InPO**: These focus on preference optimization; AGSM can be stacked as a soft-token module, with experiments showing further improvements in COCO-val metrics.
- **vs. negative prompt / CFG**: The negative branch of AGSM acts as a repulsive direction for mismatched text, but it occurs within the training score target rather than as a manual negative prompt during inference, making it more stable and learnable.

## Rating
- Novelty: ⭐⭐⭐⭐ Combines PL preference modeling, score matching, and decoupled soft tokens to solve SoftREPA's over-pushing issue in a reward-free manner.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across SD1.5, SDXL, and SD3, including T2I, long prompts, image editing, and combinations with diffusion RL.
- Writing Quality: ⭐⭐⭐⭐ While the derivations are dense, the logic from SoftREPA's failure to AGSM's correction is clear, and the findings are well-supported by data.
- Value: ⭐⭐⭐⭐ High practical value for lightweight post-training and text-image alignment in diffusion models, especially as a modular addition to current preference optimization methods.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Pareto-Guided Optimal Transport for Multi-Reward Alignment](pareto-guided_optimal_transport_for_multi-reward_alignment.md)
- [\[ICML 2026\] Restoring Initial Noise Sensitivity in Text-to-Image Distillation via Geometric Alignment](restoring_initial_noise_sensitivity_in_text-to-image_distillation_via_geometric_.md)
- [\[AAAI 2026\] ReAlign: Text-to-Motion Generation via Step-Aware Reward-Guided Alignment](../../AAAI2026/image_generation/realign_text-to-motion_generation_via_step-aware_reward-guided_alignment.md)
- [\[ICML 2026\] Implicit Preference Alignment for Human Image Animation](implicit_preference_alignment_for_human_image_animation.md)
- [\[ICML 2026\] AG-REPA: Causal Layer Selection for Representation Alignment in Audio Flow Matching](ag-repa_causal_layer_selection_for_representation_alignment_in_audio_flow_matchi.md)

</div>

<!-- RELATED:END -->

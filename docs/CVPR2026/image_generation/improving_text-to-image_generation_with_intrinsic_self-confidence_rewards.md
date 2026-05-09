---
title: >-
  [Paper Note] Improving Text-to-Image Generation with Intrinsic Self-Confidence Rewards
description: >-
  [CVPR 2026][Image Generation][Self-confidence reward] This paper proposes SOLACE, a post-training framework that leverages the denoising self-confidence of text-to-image generation models as an intrinsic reward signal, requiring no external reward models while achieving consistent improvements in compositional generation, text rendering, and text-image alignment. SOLACE is also complementary to external rewards and mitigates reward hacking when combined with them.
tags:
  - CVPR 2026
  - Image Generation
  - Self-confidence reward
  - post-training
  - Flow Matching
  - GRPO
  - text-to-image
date: 2026-05-08
content_hash: 4e6332245c5bd6b2
---

# Improving Text-to-Image Generation with Intrinsic Self-Confidence Rewards

**Conference**: CVPR 2026
**arXiv**: [2603.00918](https://arxiv.org/abs/2603.00918)
**Code**: [Project Page](https://wookiekim.github.io/SOLACE/)
**Area**: Image Generation / Diffusion Model Post-Training
**Keywords**: Self-confidence reward, post-training, Flow Matching, GRPO, text-to-image

## TL;DR

This paper proposes SOLACE, a post-training framework that leverages the denoising self-confidence of text-to-image generation models as an intrinsic reward signal, requiring no external reward models while achieving consistent improvements in compositional generation, text rendering, and text-image alignment. SOLACE is also complementary to external rewards and mitigates reward hacking when combined with them.

## Background & Motivation

Post-training of text-to-image (T2I) generation models via reinforcement learning optimization of external rewards has become an effective paradigm for improving image quality. However, existing approaches suffer from three key limitations:

**Difficulty in reward definition**: High-quality images must satisfy multiple weakly aligned criteria—compositionality, text rendering, aesthetics, and text-image alignment—whose relative importance varies across scenarios.

**Reward hacking**: Optimizing a single external reward tends to cause overfitting, leading to degradation in non-target capabilities (e.g., PickScore improves while compositional ability deteriorates).

**High operational cost**: External rewards require running additional evaluators (preference/OCR/safety models) during training, increasing pipeline complexity.

The core insight of this paper is that **large-scale pretraining endows diffusion models with priors over realistic images and text-image alignment**, and the model's degree of "confidence" in its own generated outputs constitutes a meaningful reward signal. Inspired by Score Distillation Sampling, the model is made to "self-evaluate" its generations—if the model can accurately recover noise injected into its generated outputs, it is considered highly confident in those outputs.

## Method

### Overall Architecture

SOLACE builds upon the Flow-GRPO framework. Given a text prompt $c$, $G$ image latents $\{z_0^{(i)}\}_{i=1}^G$ are sampled; each latent is then **re-noised**, and the model's ability to recover the injected noise is measured and converted into a scalar reward. The generation model is subsequently optimized via GRPO policy gradients. The entire process is performed in latent space without decoding.

### Key Designs

1. **Self-Confidence Reward**: For each generated latent $z_0^{(i)}$, $K$ shared noise probes $\epsilon^{(m)} \sim \mathcal{N}(0, I)$ are used for re-noising:

$$z_t^{(i,m)} = (1-t) z_0^{(i)} + t \epsilon^{(m)}, \quad t \in \mathcal{T}$$

The model velocity field is then used to recover the noise estimate $\hat{\epsilon}_\theta = v_\theta(z_t^{(i,m)}, t, c) + z_0^{(i)}$, and the recovery error is computed as:

$$\text{MSE}_{i,t} = \frac{1}{K} \sum_{m=1}^K \|\hat{\epsilon}_\theta(z_t^{(i,m)}, t, c) - \epsilon^{(m)}\|_2^2$$

A self-confidence score $S_{i,t} = -\log(\text{MSE}_{i,t} + \delta)$ is obtained via negative log transformation and aggregated into a scalar reward $R_{\text{SOLACE}}$. **Design Motivation**: small error = high confidence = high reward; the negative log transformation suppresses outliers and approximates Gaussian log-likelihood. **Antipodal noise pairs** $(\epsilon^{(m+K/2)} = -\epsilon^{(m)})$ are used to ensure zero-mean probes.

2. **Training Stabilization Techniques**:

    - **Suffix timestep training**: The GRPO loss is optimized only over the last $\rho$ proportion of timesteps in the denoising trajectory ($|\mathcal{T}_{\text{train}}| = \lceil \rho |\mathcal{T}| \rceil$), preventing training collapse caused by over-optimization at early timesteps (which produces textureless blank images).
    - **CFG-free confidence estimation**: Classifier-free guidance (CFG) is used during sampling, but not during confidence computation, as the blended guidance field would cause confidence evaluation to assess the guided surrogate rather than the base conditional model.
    - **Online confidence estimation**: Confidence is computed using the model being trained $\pi_\theta$ rather than a fixed reference model $\pi_{\text{ref}}$, yielding stronger reward signals as the model improves.

3. **Complementary Integration with External Rewards**: SOLACE can be applied on top of external reward post-training (e.g., FlowGRPO + PickScore). The target dimensions optimized by external rewards differ from those attended to by intrinsic self-confidence (aesthetics vs. compositionality/text rendering/alignment), making the two approaches complementary and mutually mitigating reward hacking.

### Loss & Training

The Flow-GRPO objective is adopted, with group-normalized advantage estimates:

$$\hat{A}_t^i = \frac{R(z_0^i, c) - \text{mean}(\{R(z_0^i, c)\}_{i=1}^G)}{\text{std}(\{R(z_0^i, c)\}_{i=1}^G)}$$

Training employs 10-step denoising (vs. 40 steps for SD3.5 inference), combined with clipped importance sampling ratios and KL regularization.

## Key Experimental Results

### Main Results

SOLACE post-training results on SD3.5-Medium:

| Metric | SD3.5-M (Baseline) | + SOLACE | Gain | Notes |
|--------|--------------------|----------|------|-------|
| GenEval (composition) | 0.65 | **0.71** | +0.06 | Significant gain in compositional generation |
| OCR (text rendering) | 0.61 | **0.67** | +0.06 | Significant gain in text rendering |
| CLIPScore (alignment) | 0.282 | **0.288** | +0.006 | Improved text-image alignment |
| Aesthetic | 5.36 | 5.39 | +0.03 | Minor aesthetic improvement |
| PickScore | 22.34 | 22.41 | +0.07 | Minor improvement in human preference |

When stacking SOLACE with external rewards (FlowGRPO + GenEval reward + SOLACE): GenEval reaches 0.95 and is maintained at a high level, while OCR recovers from 0.65.

### Ablation Study

| Configuration | GenEval | OCR | CLIPScore | Notes |
|---------------|---------|-----|-----------|-------|
| K=4 noise probes | 0.69 | 0.66 | 0.286 | Insufficient probes |
| **K=8 noise probes** | **0.71** | **0.67** | **0.288** | **Optimal configuration** |
| K=16 noise probes | 0.70 | 0.66 | 0.287 | Diminishing returns, higher cost |
| CFG-based confidence | 0.69 | 0.65 | 0.285 | CFG introduces surrogate bias |
| Offline confidence | 0.68 | 0.65 | 0.282 | Static reward is suboptimal |
| **Online confidence** | **0.71** | **0.67** | **0.288** | **Adaptive reward is superior** |

### Key Findings

- Intrinsic self-confidence is strongly correlated with compositional generation, text rendering, and text-image alignment, but weakly correlated with human preference.
- Using too many training timesteps ($\rho > 0.6$) or sampling without CFG leads to training collapse (reward hacking → textureless images).
- SOLACE is complementary to external rewards: stacking the two mitigates reward hacking caused by external reward optimization.
- A user study (1,800 responses / 20 participants) confirms that SOLACE outperforms the baseline in both visual realism and text-image alignment.

## Highlights & Insights

1. **Self-evaluation paradigm**: This work is the first to formalize the denoising self-confidence of diffusion/flow models as an intrinsic reward signal, opening a new direction for self-supervised post-training.
2. **No external dependencies**: No additional reward models, annotated data, or evaluators are required, reducing resource overhead and pipeline complexity.
3. **Reward hacking mitigation**: When combined with external rewards, SOLACE acts as a regularization force that recovers compositional and text rendering capabilities, demonstrating strong complementarity.
4. **Latent-space computation**: Self-confidence rewards are computed entirely in latent space without decoding to pixel space, making the approach computationally efficient.

## Limitations & Future Work

1. The correlation between intrinsic self-confidence and human preference is weak, making it insufficient as a standalone substitute for human preference rewards.
2. The method cannot perform targeted optimization toward specific alignment objectives (e.g., safety).
3. Training stability is sensitive to timestep selection ($\rho$ requires careful tuning), and more robust confidence estimation methods may be needed.
4. Validation is limited to SD3.5; applicability to other architectures (e.g., DiT, autoregressive models) remains to be explored.
5. Future work may extend this approach to temporal and multi-view consistency evaluation in video and 3D generation.

## Related Work & Insights

- **Flow-GRPO**: The base RL post-training framework adopted in this work, which applies GRPO to Flow Matching models.
- **Score Distillation Sampling (SDS)**: Inspires the core idea of using the generative model to evaluate its own outputs.
- **Intuitor (LLM domain)**: A pioneering work using self-confidence as an intrinsic reward in LLMs; this paper extends the concept to continuous denoising trajectories.
- **Insights**: Self-confidence signals may prove equally effective in other generative tasks (video, 3D, audio)—the intuition that stronger denoising ability ≈ higher confidence in generation quality appears to be broadly applicable.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Formalizing the denoising self-confidence of diffusion models as an intrinsic reward is a highly original idea that opens a new direction for self-supervised post-training.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-metric evaluation, user study, ablation analysis, and experiments stacking with external rewards provide thorough validation.
- Writing Quality: ⭐⭐⭐⭐ Mathematical derivations are clear, motivations are convincingly articulated, and the structure is well-organized.
- Value: ⭐⭐⭐⭐⭐ Provides a zero-cost intrinsic reward signal broadly applicable to T2I post-training, with strong complementarity to external rewards.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] SOLACE: Improving Text-to-Image Generation with Intrinsic Self-Confidence Rewards](solace_self_confidence_rewards_t2i.md)
- [\[CVPR 2026\] Resolving the Identity Crisis in Text-to-Image Generation](resolving_the_identity_crisis_in_text-to-image_generation.md)
- [\[CVPR 2026\] Self-Corrected Image Generation with Explainable Latent Rewards](self-corrected_image_generation_with_explainable_latent_rewards.md)
- [\[CVPR 2026\] LeapAlign: Post-Training Flow Matching Models at Any Generation Step by Building Two-Step Trajectories](leapalign_post_training_flow_matching_models_at_any_generation_step.md)
- [\[CVPR 2026\] Neighbor GRPO: Contrastive ODE Policy Optimization Aligns Flow Models](neighbor_grpo_contrastive_ode_policy_optimization_aligns_flow_models.md)

<!-- RELATED:END -->

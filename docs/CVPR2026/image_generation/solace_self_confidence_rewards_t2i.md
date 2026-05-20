---
title: >-
  [Paper Note] SOLACE: Improving Text-to-Image Generation with Intrinsic Self-Confidence Rewards
description: >-
  [CVPR 2026][Image Generation][Text-to-Image] SOLACE uses a T2I model's intrinsic denoising self-confidence (i.e., the accuracy with which it recovers injected noise) as an internal reward signal to replace external rewar…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Text-to-Image"
  - "Self-Confidence Reward"
  - "Flow-GRPO"
  - "External-Reward-Free"
  - "Post-Training Alignment"
date: 2026-05-08
content_hash: be5b8d37b7a51325
---

# SOLACE: Improving Text-to-Image Generation with Intrinsic Self-Confidence Rewards

**Conference**: CVPR 2026
**arXiv**: [2603.00918](https://arxiv.org/abs/2603.00918)  
**Code**: [https://wookiekim.github.io/SOLACE/](https://wookiekim.github.io/SOLACE/)  
**Authors**: Seungwook Kim, Minsu Cho (POSTECH / RLWRLD)
**Area**: Diffusion Models / Image Generation / Post-Training
**Keywords**: Text-to-Image, Self-Confidence Reward, Flow-GRPO, External-Reward-Free, Post-Training Alignment

## TL;DR
SOLACE uses a T2I model's intrinsic denoising self-confidence (i.e., the accuracy with which it recovers injected noise) as an internal reward signal to replace external reward models in post-training, achieving consistent improvements in compositional generation, text rendering, and text-image alignment. The signal is also complementary to external rewards and can mitigate reward hacking.

## Background & Motivation
Post-training has become an important paradigm for improving T2I generation quality, typically relying on external reward signals (e.g., PickScore, HPSv2, and other human preference models) to drive reinforcement learning. However, this approach suffers from three core limitations:
1. **Difficulty in defining external rewards**: High-quality images must simultaneously satisfy multiple weakly correlated criteria — compositionality, text rendering, aesthetics, and text-image alignment — with varying relative importance across scenarios.
2. **Reward hacking**: Optimizing for a single external metric often leads to overfitting — the target score improves while non-target capabilities degrade (e.g., PickScore improves but compositional fidelity collapses).
3. **Cost and complexity**: Human preference reward models require large-scale annotation and must be run alongside the training pipeline, substantially increasing complexity.

**Core Problem**: *Can a T2I generator itself provide meaningful post-training signals?* Large-scale pretraining endows models with strong priors over the real image distribution and text-image alignment — a high-quality output should elicit greater "confidence" from the model.

## Core Idea
Inspired by Score Distillation Sampling (SDS) — which uses a pretrained T2I model as a critic for text-to-3D generation — SOLACE internalizes the same idea: **letting the T2I model critique its own generations**. Concretely, noise is re-injected into the model's generated latents, and the accuracy with which the model recovers the injected noise is measured. More accurate recovery → greater model "confidence" in its output → higher reward.

## Method

### Overall Architecture
Given a text prompt $c$:
1. Sample $G=16$ independent reverse trajectories to obtain terminal latents $\{z_0^{(i)}\}_{i=1}^G$.
2. Draw $K=8$ shared noise probes $\epsilon^{(m)} \sim \mathcal{N}(0,I)$, using antithetic pairs to ensure zero mean.
3. Re-inject noise into each $z_0^{(i)}$ at multiple timesteps $t \in \mathcal{T}$: $z_t^{(i,m)} = (1-t)z_0^{(i)} + t\epsilon^{(m)}$.
4. The model predicts velocity fields $v_\theta(z_t^{(i,m)}, t, c)$ and recovers noise estimates $\hat{\epsilon}_\theta = v_\theta + z_0^{(i)}$.
5. Compute MSE reconstruction error, convert to self-confidence rewards, and feed into Flow-GRPO optimization.

### Self-Confidence Reward Formulation
For each generated sample $z_0^{(i)}$:
$$\text{MSE}_{i,t} = \frac{1}{K}\sum_{m=1}^K \|\hat{\epsilon}_\theta(z_t^{(i,m)}, t, c) - \epsilon^{(m)}\|_2^2$$
$$S_{i,t} = -\log(\text{MSE}_{i,t} + \delta)$$
$$R_{\text{SOLACE}}(z_0^{(i)}, c) = \frac{1}{\sum_{t\in\mathcal{T}} w(t)} \sum_{t\in\mathcal{T}} w(t) S_{i,t}$$

The negative log transformation provides three benefits: (1) approximates Gaussian log-likelihood; (2) compresses outliers; (3) enables additivity across timesteps. In practice, $w(t)=1$.

### Key Designs
1. **Suffix timestep training** ($\rho=0.6$): Only the latter 60% of denoising steps are optimized, preventing the model from pushing latents into degenerate regions where noise is trivially predictable, which would cause training collapse.
2. **Self-confidence computed without CFG**: CFG constructs a mixed field $v_\text{cfg} = v_\text{uncond} + s(v_\text{cond} - v_\text{uncond})$, which optimizes a guided surrogate rather than the base conditional policy, inducing reward hacking.
3. **Online over offline computation**: Computing self-confidence using the model being trained ($\pi_\theta$) rather than a frozen reference ($\pi_\text{ref}$) yields better performance — as the model improves, self-confidence estimation becomes more accurate.
4. **Reduced denoising steps**: 10 steps during training (vs. 40 at inference), substantially accelerating training without sacrificing quality.

### Loss & Training
- Optimizer: AdamW, lr=3e-4
- LoRA: rank=32, α=64
- KL regularization: β=0.04
- GRPO group size: G=16
- Noise probes: K=8 (antithetic pairs)
- Training iterations: 2000
- Resolution: 512×512
- Inference CFG scale: 7.0
- Hardware: 8× NVIDIA RTX PRO 6000 Blackwell

## Key Experimental Results

### Main Results (SD3.5-M Baseline)

| Model | GenEval↑ | OCR↑ | CLIPScore↑ | Aesthetic↑ | PickScore↑ | HPSv2↑ | ImageReward↑ |
|------|----------|------|-----------|-----------|------------|--------|-------------|
| SD3.5-M | 0.65 | 0.61 | 0.282 | 5.36 | 22.34 | 0.279 | 0.84 |
| +SOLACE | **0.71** | **0.67** | **0.288** | 5.39 | 22.41 | 0.278 | 0.87 |
| SD3.5-L | 0.71 | 0.68 | 0.289 | 5.50 | 22.91 | 0.288 | 0.96 |

**Key Findings**: SOLACE enables SD3.5-M (2.5B parameters) to nearly match SD3.5-L (7.1B parameters) on GenEval, OCR, and CLIPScore — with less than one-third the parameter count.

### SOLACE + External Rewards: Complementarity

| Model | GenEval↑ | OCR↑ | CLIPScore↑ | PickScore↑ |
|------|----------|------|-----------|------------|
| SD3.5-M + FlowGRPO(GenEval) | **0.95** | 0.65 | 0.293 | 22.51 |
| SD3.5-M + FlowGRPO(GenEval) + SOLACE | **0.92** | **0.71** | **0.294** | 22.50 |
| SD3.5-M + FlowGRPO(PickScore) | 0.54 | 0.68 | 0.278 | **23.50** |
| SD3.5-M + FlowGRPO(PickScore) + SOLACE | 0.77 | **0.70** | 0.287 | 22.73 |

Stacking SOLACE on top of FlowGRPO with external rewards improves compositionality, text rendering, and alignment, with only marginal decreases in the targeted external metric — **intrinsic and extrinsic rewards are complementary and mitigate reward hacking**. Notably, PickScore post-training caused GenEval to collapse from 0.65 to 0.54; adding SOLACE recovers it to 0.77.

### Ablation Study
- **Number of noise probes K**: K=4/8/16 show similar performance; K=8 is marginally best with reasonable computational cost.
- **CFG for self-confidence**: Using CFG degrades performance (GenEval 0.68 vs. 0.71), validating that optimizing a guided surrogate is harmful.
- **Online vs. offline**: Online computation consistently outperforms offline (GenEval 0.71 vs. 0.69; OCR 0.67 vs. 0.61).
- **Training collapse conditions**: (1) $\rho > 0.6$; (2) omitting CFG during candidate sampling → produces texture-free images.

### User Study
Approximately 1,800 responses were collected from 20 participants on PartiPrompts and HPSv2 prompts. SOLACE consistently outperforms the SD3.5-M baseline in both visual realism/appeal and text-image alignment.

## Highlights & Insights
- **Pretrained quality priors**: The model's denoising capability encodes implicit knowledge of "what makes a good image"; self-confidence is an exploitable intrinsic signal.
- **SDS → self-critique**: SDS uses a T2I model to evaluate 3D generations; SOLACE internalizes the same idea as self-evaluation — an elegant methodological transfer.
- **Intrinsic + extrinsic complementarity**: The two signal types attend to different dimensions (self-confidence → compositionality/text rendering; external → human preferences), and combining them yields the best results.
- **Stabilization designs are critical**: The suffix window, CFG exclusion, and online computation are each individually necessary — their absence leads to collapse or degraded performance.
- **Latent-space operation**: Rewards are computed entirely in latent space, eliminating the need to decode to pixel space and reducing decoder overhead.

## Limitations & Future Work
- Weak correlation with human preference metrics; cannot independently target specific alignment objectives (e.g., aesthetics).
- Only validated on flow matching architectures (SD3.5); applicability to autoregressive T2I models remains unexplored.
- Future directions include: (1) extending temporal/multi-view consistency to video and 3D generation; (2) disentangling and calibrating intrinsic signals for task-level reward shaping.

## Related Work & Insights
- **vs. FlowGRPO**: External rewards are targeted but prone to reward hacking and require additional models; SOLACE is external-dependency-free but cannot precisely target specific objectives.
- **vs. DPO/ReFL**: Require preference-paired data or differentiable rewards; SOLACE is fully unsupervised.
- **vs. Intuitor (LLM)**: First non-trivial transfer of self-confidence rewards from LLM discrete token generation to T2I continuous denoising trajectories.
- **vs. SDS**: SDS uses a pretrained model to evaluate external generations (3D); SOLACE uses the current model to evaluate its own generations (self-critique).

## Relevance to My Research
The paradigm of post-training with intrinsic signals has cross-domain generalization potential — detection and segmentation models are also pretrained at scale; it is worth investigating whether analogous "self-confidence" signals can be extracted for unsupervised post-training.

## Rating
- Novelty: ⭐⭐⭐⭐ Self-confidence as an intrinsic T2I reward is novel and principled, though Intuitor establishes precedent in the LLM domain.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple benchmarks (GenEval/OCR/6 preference metrics) + user study + ablations + multiple models (SD3.5-M/L) + complementarity experiments.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear, method derivation is rigorous, and ablation study is systematic.
- Value: ⭐⭐⭐ Image generation is not a core research direction, but the paradigm of "post-training with intrinsic signals" warrants attention.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Self-Corrected Image Generation with Explainable Latent Rewards](self-corrected_image_generation_with_explainable_latent_rewards.md)
- [\[CVPR 2026\] Resolving the Identity Crisis in Text-to-Image Generation](resolving_the_identity_crisis_in_text-to-image_generation.md)
- [\[CVPR 2026\] Extending One-Step Image Generation from Class Labels to Text via Discriminative Text Representation](emf_meanflow_text_to_image.md)
- [\[CVPR 2026\] PSR: Scaling Multi-Subject Personalized Image Generation with Pairwise Subject-Consistency Rewards](psr_scaling_multi-subject_personalized_image_generation_with_pairwise_subject-co.md)
- [\[CVPR 2026\] CTCal: Rethinking Text-to-Image Diffusion Models via Cross-Timestep Self-Calibration](ctcal_rethinking_text-to-image_diffusion_models_via_cross-timestep_self-calibrat.md)

</div>

<!-- RELATED:END -->

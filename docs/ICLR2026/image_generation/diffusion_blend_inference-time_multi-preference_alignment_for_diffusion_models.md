---
title: >-
  [Paper Note] Diffusion Blend: Inference-Time Multi-Preference Alignment for Diffusion Models
description: >-
  [ICLR 2026][Image Generation][multi-preference alignment] This paper proposes Diffusion Blend, which achieves multi-preference alignment at inference time by blending the backward diffusion processes of multiple reward-finetuned models. DB-MPA supports arbitrary linear combinations of rewards; DB-KLA enables dynamic KL regularization control; DB-MPA-LS eliminates inference overhead via stochastic LoRA sampling. The paper theoretically derives error bounds for the blending approximation and empirically approaches the MORL oracle upper bound.
tags:
  - ICLR 2026
  - Image Generation
  - multi-preference alignment
  - inference-time
  - backward SDE blending
  - KL regularization control
  - Pareto-optimal
date: 2026-05-08
content_hash: e84af13955db82a1
---

# Diffusion Blend: Inference-Time Multi-Preference Alignment for Diffusion Models

**Conference**: ICLR 2026
**arXiv**: [2505.18547](https://arxiv.org/abs/2505.18547)
**Code**: Available (GitHub)
**Area**: Diffusion Models / Multi-Objective Alignment
**Keywords**: multi-preference alignment, inference-time, backward SDE blending, KL regularization control, Pareto-optimal

## TL;DR
This paper proposes Diffusion Blend, which achieves multi-preference alignment at inference time by blending the backward diffusion processes of multiple reward-finetuned models. DB-MPA supports arbitrary linear combinations of rewards; DB-KLA enables dynamic KL regularization control; DB-MPA-LS eliminates inference overhead via stochastic LoRA sampling. The paper theoretically derives error bounds for the blending approximation and empirically approaches the MORL oracle upper bound.

## Background & Motivation

**Background**: RL fine-tuning of diffusion models typically fixes a single reward function and KL regularization weight $\alpha$. Once fine-tuned, the model is locked to a specific $(r, \alpha)$ configuration and cannot adapt to varying user preferences.

**Limitations of Prior Work**: (a) Users may require different trade-offs among aesthetics, semantic consistency, and human preference, necessitating a separate fine-tuned model for each preference combination—at prohibitive cost; (b) KL regularization that is too weak leads to reward hacking, while too strong regularization yields insufficient alignment, and the optimal value requires grid search; (c) Rewarded Soup (linear interpolation in weight space) is too coarse, and guidance-based methods require differentiable rewards and incur large computational cost.

**Key Challenge**: Deployment-time flexibility in preferences versus the rigidity of fine-tuning. How can arbitrary preference combinations be accommodated at inference time without retraining?

**Goal**: Given $m$ models each fine-tuned on a distinct base reward, generate images aligned to $r(w) = \sum w_i r_i$ at inference time according to user-specified weights $w$, while supporting dynamic adjustment of KL regularization strength.

**Key Insight**: Starting from the backward SDE perspective of diffusion models, the paper proves that the drift term $f^{(r,\alpha)}$ of an aligned model can be decomposed into a pretrained drift plus a control term. By approximating the control term via a Jensen gap linearization, the backward SDE admits a linear blending scheme.

**Core Idea**: The backward SDE drift of a reward-aligned diffusion model can be linearly combined to approximate the alignment effect of any linear combination of rewards.

## Method

### Overall Architecture

Two phases:
- **Fine-tuning phase**: Independently RL fine-tune one model per base reward $r_i$, yielding $m$ fine-tuned models $\theta_i^{\text{rl}}$.
- **Inference phase**: The user specifies weights $w$; the backward SDE drifts of the $m$ models are blended according to $w$ at each denoising step.

### Key Designs

1. **Proposition 1: SDE Decomposition of Aligned Models**

   - Function: Decomposes the backward drift of an aligned model into a pretrained drift and a control term.
   - Mechanism: $f^{(r,\alpha)}(x_t, t) = f^{\text{pre}}(x_t, t) - \beta(t) u^{(r,\alpha)}(x_t, t)$, where $u^{(r,\alpha)} = \nabla_{x_t} \log \mathbb{E}_{x_0 \sim p_{0|t}^{\text{pre}}}[\exp(r(x_0)/\alpha)]$.
   - Design Motivation: Isolating the alignment effect into the control term $u^{(r,\alpha)}$ establishes the foundation for subsequent linear combination.

2. **Jensen Gap Approximation + Linearization (Lemma 2)**

   - Function: Approximates $\log \mathbb{E}[\exp(\cdot)]$ by $\mathbb{E}[\cdot]$ (exchanging the order of log-exp and expectation).
   - Mechanism: $u^{(r,\alpha)} \approx \bar{u}^{(r,\alpha)} = \nabla_x \mathbb{E}[r(x_0)/\alpha]$. By linearity of expectation, for a linear reward $r(w) = \sum w_i r_i$, it follows that $f^{(r(w),\alpha)} \approx \sum w_i f^{(r_i, \alpha)}$.
   - Design Motivation: The Jensen gap approximation is widely used in diffusion models (e.g., DPS/RGG), and the approximation error vanishes as $t \to 0$.

3. **DB-MPA (Multi-Preference Alignment)**

   - Function: Blends the backward SDEs of reward-finetuned models at inference time according to user weights $w$.
   - Mechanism: At each denoising step, computes $\hat{\epsilon}_t = \sum w_i \epsilon_{\theta_i^{\text{rl}}}(x_t, t)$.

4. **DB-KLA (KL Alignment Control)**

   - Function: Adjusts KL regularization strength at inference time.
   - Mechanism: $f^{(r, \alpha/\lambda)} \approx (1-\lambda) f^{\text{pre}} + \lambda f^{(r,\alpha)}$, blending the pretrained and fine-tuned models.

5. **DB-MPA-LS (Overhead-Free Approximation)**

   - Function: Eliminates the $m\times$ inference overhead of DB-MPA.
   - Mechanism: At each denoising step, a single LoRA adapter is stochastically sampled according to weights $w$ (Bernoulli/Categorical sampling) rather than evaluating all models. Proposition 2 proves that the blended SDE and the stochastic sampling SDE share identical marginal distributions.
   - Design Motivation: Exploits the stochasticity of the diffusion process—noise injection makes step-wise stochastic sampling statistically equivalent to weighted averaging.

### Loss & Training

- DPOK is used to RL fine-tune SD v1.5.
- Each base reward is fine-tuned independently.
- No training is required at inference time; only the noise prediction at each denoising step is modified.

## Key Experimental Results

### Main Results (SD v1.5, ImageReward + VILA/PickScore)

DB-MPA dominates the Pareto frontier over Rewarded Soup (RS), CoDe, and RGG, approaching the MORL oracle upper bound.

Key figures: At $w=0.5$, DB-MPA achieves 85–90% of the performance of each independently fine-tuned model on both rewards, whereas RS reaches only 60–70%.

### Ablation Study

| Method | Inference Overhead | Performance (vs. MORL) |
|--------|-------------------|------------------------|
| DB-MPA | $m \times$ | ~95% of MORL |
| DB-MPA-LS | $1 \times$ | ~90% of MORL |
| RS | $1 \times$ | ~70% of MORL |
| RGG | $1 \times$ (+ gradient) | ~60% of MORL |
| CoDe | $N \times$ (search) | ~65% of MORL |

DB-KLA enables smooth KL control: $\lambda > 1$ strengthens alignment but risks overfitting, while $\lambda < 1$ is more conservative but preserves pretrained quality.

### Key Findings
- DB-MPA substantially outperforms RS (weight-space interpolation) on the Pareto frontier, demonstrating that **backward SDE blending is superior to parameter-space blending**.
- DB-MPA-LS via stochastic LoRA sampling incurs negligible performance loss (~5% gap) while reducing inference overhead to $1\times$.
- DB-KLA offers a more flexible KL control mechanism than retraining.
- The Jensen gap approximation remains effective for JPEG compressibility, a reward adversarial to aesthetics.

## Highlights & Insights
- The contrast between **SDE blending and parameter-space blending** is clear and compelling. Rewarded Soup linearizes in parameter space; DB-MPA linearizes in the SDE drift space. The latter is better grounded theoretically and yields superior performance. The core reason is that the linearization error of the SDE drift is bounded (Lemma 1), whereas no analogous guarantee exists for parameter-space linearization.
- **Proposition 2 (stochastic LoRA sampling equivalence)** is an elegant theoretical result—leveraging the noise injection of the SDE to make step-wise random selection equivalent to weighted averaging. This is infeasible in LLMs (discrete token space) and constitutes a unique advantage of diffusion models.
- The approach offers extremely high inference-time flexibility: users can adjust the aesthetics–alignment trade-off in real time via a slider.

## Limitations & Future Work
- The Jensen gap approximation error grows when $\alpha$ is very small (the $L_{t,2}$ term in Lemma 1), making the approach unsuitable for extreme alignment requirements.
- Validation is limited to SD v1.5; applicability to larger models (SDXL/Flux) has not been tested.
- DB-MPA incurs $m\times$ inference overhead, which is impractical for large numbers of reward functions (partially mitigated by DB-MPA-LS).
- The linear reward combination assumption limits expressiveness; non-linear preference relations cannot be handled.
- Comparisons with recent alignment methods such as DAV/DenseGRPO are absent.

## Related Work & Insights
- **vs. Rewarded Soup**: Parameter-space linearization vs. SDE drift linearization. DB-MPA is theoretically more rigorous and achieves better performance.
- **vs. Guidance methods (RGG/CoDe)**: DB-MPA requires neither differentiable rewards nor inference-time search, and outperforms both.
- **vs. LLM DeRa**: Shares the same inspiration (blending aligned and base models), but contributes SDE-theoretic analysis and stochastic LoRA sampling innovations tailored to diffusion models.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The SDE blending theoretical framework is novel; the stochastic LoRA sampling equivalence is an elegant contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive across multiple reward combinations, Pareto analysis, and KL control experiments, but limited to SD v1.5.
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretical derivations are clear, figures are intuitive, and the motivation–theory–algorithm–experiment narrative is tightly structured.
- Value: ⭐⭐⭐⭐⭐ Provides a practically useful and theoretically grounded solution for multi-preference deployment of diffusion models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] RNE: plug-and-play diffusion inference-time control and energy-based training](rne_plug-and-play_diffusion_inference-time_control_and_energy-based_training.md)
- [\[AAAI 2026\] Multi-Metric Preference Alignment for Generative Speech Restoration](../../AAAI2026/image_generation/multi-metric_preference_alignment_for_generative_speech_restoration.md)
- [\[ICLR 2026\] GLASS Flows: Efficient Inference for Reward Alignment of Flow and Diffusion Models](glass_flows_reward_alignment_diffusion.md)
- [\[ICLR 2026\] Test-Time Iterative Error Correction for Efficient Diffusion Models](test-time_iterative_error_correction_for_efficient_diffusion_models.md)
- [\[ICLR 2026\] Unsupervised Conformal Inference: Bootstrapping and Alignment to Control LLM Uncertainty](unsupervised_conformal_inference_bootstrapping_and_alignment_to_control_llm_unce.md)

</div>

<!-- RELATED:END -->

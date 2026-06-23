---
title: >-
  [Paper Note] Overshoot and Shrinkage in Classifier-Free Guidance: From Theory to Practice
description: >-
  [ICLR 2026][Image Generation][Classifier-Free Guidance] This paper reanalyzes Classifier-Free Guidance (CFG) using the "dynamical phase transition" framework from statistical physics. It proves that in sufficiently high dimensions, CFG can precisely recover the target distribution (the "blessing of dimensionality"), while accurately characterizing mean overshoot and varianc
tags:
  - ICLR 2026
  - Image Generation
  - Classifier-Free Guidance
  - Diffusion Model
date: 2026-05-08
content_hash: a331fa9825c590a6
---
# Overshoot and Shrinkage in Classifier-Free Guidance: From Theory to Practice

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=cNsoOr1hTH](https://openreview.net/forum?id=cNsoOr1hTH)  
**Code**: None  
**Area**: Diffusion Models / Image Generation  
**Keywords**: Classifier-Free Guidance, Diffusion Models, High-Dimensional Statistical Physics, Mean Overshoot, Variance Shrinkage

## TL;DR
This paper reanalyzes Classifier-Free Guidance (CFG) using the "dynamical phase transition" framework from statistical physics. It proves that in sufficiently high dimensions, CFG can precisely recover the target distribution (the "blessing of dimensionality"), while accurately characterizing mean overshoot and variance shrinkage observed in lower dimensions. Consequently, the authors propose Power-Law CFG, which nonlinearly amplifies score differences. This approach theoretically alleviates both artifacts and consistently improves image quality and diversity across SOTA models like DiT, EDM2, and Text-to-Image models.

## Background & Motivation
**Background**: Diffusion models and Flow Matching are the de facto standards for high-dimensional signal generation, and conditional generation almost entirely relies on CFG. CFG does not require an external classifier; it only requires the model to learn both conditional and unconditional denoising. During inference, extrapolation is performed along the "conditional direction": $S_t^{\text{CFG}}(\vec{x},c)=S_t(\vec{x},c)+\omega\big(S_t(\vec{x},c)-S_t(\vec{x})\big)$, where $\omega>0$ is the guidance strength.

**Limitations of Prior Work**: When $\omega>0$, the sampled distribution is no longer the true conditional distribution. Both practice and prior theory (Chidambaram et al. 2024; Wu et al. 2024) have observed two artifacts: **mean overshoot** (samples pushed toward class boundaries, over-saturation, over-contrast) and **variance shrinkage** (distribution becomes sharper than the target, leading to reduced diversity). In other words, CFG involves a trade-off of "quality for diversity."

**Key Challenge**: Previous theoretical analyses of CFG were mostly limited to one-dimensional or finite-dimensional Gaussian Mixture Models (GMM), concluding that "CFG necessarily distorts the target distribution." This contradicts practical experience where CFG is almost always beneficial. Three questions remain: Can CFG actually generate the correct distribution? What determines overshoot and shrinkage? Can a new guidance method be designed to provably mitigate artifacts while retaining CFG gains?

**Key Insight**: The authors borrow the framework of **diffusion dynamical phase transitions** (Biroli & Mézard 2023; Biroli et al. 2024). The reverse diffusion process undergoes several "phase regimes" over time, with a **speciation time** $t_s$ before which trajectories have not yet "decided" their class. Analyzing CFG within this $d\to\infty$ framework is a perspective absent in prior research.

**Core Idea**: In high dimensions, CFG acts as an accelerator only before $t_s$ and automatically becomes ineffective afterward, thus **asymptotically recovering the correct distribution**. Artifacts are merely finite-dimensional corrections (of magnitude $1/\sqrt{d}$). Guided by this, applying a **nonlinear power-law amplification** to the score difference can suppress finite-dimensional artifacts without destroying high-dimensional guarantees.

## Method

### Overall Architecture
This paper does not propose a new generative network but rather a chain from theory to practice: first using a high-dimensional statistical physics framework to **explain why/when CFG is correct**, then **locating the source of low-dimensional artifacts**, and finally **designing a minimalist nonlinear guidance improvement** based on these findings.

Specifically, the authors model data as a mixture of two equally weighted, isotropic Gaussians with variance $\sigma^2$ and means $\pm\vec{m}$ (setting $|\vec{m}|=\sqrt{d}$ to ensure separability). The forward process is an Ornstein-Uhlenbeck process $d\vec{x}(t)=-\vec{x}(t)\,dt+\sqrt{2}\,d\vec{B}(t)$, driven in reverse by the score $S_t(\vec{x})=\nabla\log P_t(\vec{x})$. A key observation is that CFG only acts in the direction of $\vec{m}$. Thus, the high-dimensional dynamics can be projected onto a scalar $q(t)=\vec{x}\cdot\vec{m}/|\vec{m}|$, analyzed as a Langevin process with a 1D effective potential $V^{\text{CFG}}(q,\tau)$. The reverse process is segmented by the speciation time $t_s=\tfrac{1}{2}\log d$ into "pre-speciation / speciation moment / post-speciation" to prove the effects of CFG.

### Key Designs

**1. Embedding CFG in the phase transition framework to prove the "blessing of dimensionality"**

Addressing whether CFG can generate the correct distribution, the authors analyze the reverse process in three stages (the effective potential after projection is shown below):

$$V^{\text{CFG}}=\underbrace{\tfrac{1}{2}q^2-2e^{-(t-t_s)}q}_{\text{conditional potential}}+\omega\underbrace{\big[-qe^{-(t-t_s)}+\ln\cosh\big(qe^{-(t-t_s)}\big)\big]}_{\text{CFG-induced potential}}.$$

**Step I (Pre-speciation)**: The CFG-induced potential adds an extra "push" along the $\vec{m}$ direction, correcting trajectories that would have leaned toward the wrong class and **accelerating convergence to the target class**. **Step II (Near $t_s$)**: As $q$ grows to the order of $\mathcal{O}(\sqrt{d})$, the CFG term becomes an exponentially small correction. Positional differences caused by different $\omega$ values are "forgotten," and guided trajectories **re-align** with unguided ones. **Step III (Post-speciation, Regime II)**: Since $1-\tanh(\vec{x}\cdot\vec{m}e^{-t}/\Gamma_t)\to0$, the CFG term effectively vanishes, and the trajectory fully follows the unguided conditional evolution. Combined, these steps show that in infinite and sufficiently high dimensions, CFG recovers the correct target distribution regardless of $\omega$. This counterintuitive result is a core contribution.

**2. Characterizing finite-dimensional mean overshoot and variance shrinkage**

Regarding the origin of artifacts, the authors prove they are essentially **finite-dimensional corrections** rather than inherent flaws of CFG. In finite dimensions, trajectories **no longer perfectly align** when exiting Regime I. The extra push from CFG in Regime I leaks into Regime II, causing overshoot relative to the target distribution with a relative magnitude of $\mathcal{O}(1/\sqrt{d})$. Additionally, the CFG term increases the second derivative (curvature) of the effective potential $V^{\text{CFG}}(q,t)$, making it "steeper," which **shrinks the variance of the generated distribution**.

**3. Power-Law CFG: Nonlinear power-law amplification of score differences**

To suppress artifacts while retaining gains, the authors propose a minimal modification—raising the conditional score difference along $\vec{m}$ to the power of $\alpha>0$:

$$\vec{S}_t^{\text{PL}}(\vec{x},c)=\vec{S}_t(\vec{x},c)+\omega\big[\vec{S}_t(\vec{x},c)-\vec{S}_t(\vec{x})\big]\,\big\|\vec{S}_t(\vec{x},c)-\vec{S}_t(\vec{x})\big\|^{\alpha}.$$

This has two complementary effects: when the score difference $\delta S_t=\|\vec{S}_t(\vec{x},c)-\vec{S}_t(\vec{x})\|$ is small (weak or unreliable signal), guidance is suppressed; when the signal is strong, guidance is amplified to strengthen the push toward the correct class. The authors show that power-law scaling directly modulates the curvature-related term via $B(q)^\alpha$, where $\alpha>0$ inhibits overshoot and reduces curvature increments (weakening shrinkage). Crucially, this **does not destroy high-dimensional guarantees**.

**4. A unified family of nonlinear guidance**

The authors further point out that Power-Law is part of a broader family of valid nonlinear guidance:

$$S_t^{\text{CFG-NL}}(\vec{x},c)=S_t(\vec{x},c)+\big(S_t(\vec{x},c)-S_t(\vec{x})\big)\,\phi_t\big(\|\vec{S}_t(\vec{x},c)-\vec{S}_t(\vec{x})\|\big),$$,

provided $\lim_{s\to0}s\,\phi_t(s)=0$. Standard CFG is the case where $\phi_t(s)=\omega$; limited-interval CFG and time-varying weight schedulers are also special cases. Existing methods remain **linear with respect to the score difference**, whereas Power-Law innovates by introducing nonlinearity to the score difference itself.

### Loss & Training
Ours does not modify the training objective. The method acts purely on the inference-time score/guidance term, making Power-Law CFG "plug-and-play." Only one additional hyperparameter $\alpha$ is needed; in latent space, $\alpha=0.9$ was found to be consistently effective across various models.

## Key Experimental Results

### Main Results
On EDM2-S, DiT/XL-2 (ImageNet-1K conditional), and two text-to-image MMDiT models, Power-Law was combined with standard CFG and strong competitors (Limited, CADS). FID measures quality, while Precision/Recall measure diversity:

| Model | Method | FID↓ | Precision↑ | Recall↑ |
|------|------|------|-----------|---------|
| EDM2-S (CC, IM-1K 512) | Standard CFG | 2.29 | 0.751 | 0.582 |
| EDM2-S | Power-law CFG | 1.93 | 0.780 | 0.631 |
| EDM2-S | Power-law + CADS | **1.52** | 0.770 | 0.622 |
| DiT/XL-2 (CC, IM-1K 256) | Standard CFG | 2.27 | 0.829 | 0.584 |
| DiT/XL-2 | Power-law + CADS | **1.63** | 0.754 | 0.639 |
| Diff. MMDiT (T2IM, CC12M) | Standard CFG | 8.58 | 0.661 | 0.569 |
| Diff. MMDiT | Power-law + CADS | **7.98** | 0.690 | 0.573 |
| FM MMDiT (T2IM, COCO) | Standard CFG | 5.20 | 0.629 | 0.594 |
| FM MMDiT | Power-law + CADS | **4.71** | 0.640 | 0.624 |

Power-Law improves both quality and diversity in most cases and achieves SOTA results when stacked with CADS or Limited CFG.

### Ablation Study

| Configuration | Phenomenon | Explanation |
|------|------|------|
| $\alpha=0$ | Degenerates to Standard CFG | Baseline |
| Increasing $\alpha$ | FID improvement | Larger $\alpha$ is better on EDM2-S 512 |
| Increasing $\alpha$ | Increased robustness to $\omega$ | FID remains stable over a larger $\omega$ range |
| $\alpha=0.9$ (Latent) | Consistently optimal | Stable improvements across models without tuning |

### Key Findings
- **Overshoot/shrinkage are monotonically controlled by $\alpha$**: $\alpha$ modifies the shape of the guidance curve, providing the flexibility required to mitigate artifacts.
- **Robustness is a primary advantage**: Increasing $\alpha$ not only lowers FID but also significantly reduces sensitivity to guidance strength $\omega$. While standard CFG collapses in diversity at $\omega=5$, Power-Law remains stable at $\omega=10$.
- **Latent vs. Pixel space**: $\alpha=0.9$ is robust in latent space, whereas the optimal $\alpha$ in pixel space varies more, requiring joint tuning of $\alpha$ and $\omega$ for maximum gain.

## Highlights & Insights
- **"Blessing of Dimensionality" flips mainstream understanding**: While finite-dimensional analyses claim CFG "necessarily distorts distributions," this phase transition framework proves asymptotic correctness in high dimensions, treating artifacts as $1/\sqrt{d}$ corrections.
- **Minimal modification with theoretical guardrails**: Power-Law simply multiplies the score difference by $\|\cdot\|^\alpha$, which analytically suppresses overshoot and shrinkage while maintaining high-dimensional alignment.
- **Unified framework is transferable**: Categorizing various CFG variants as special cases of $\phi_t$ suggests a larger design space for optimizing nonlinear guidance functions.

## Limitations & Future Work
- The theory rests on the assumption of **perfect score estimation**, explaining how to mitigate artifacts but **not explaining why standard CFG (with artifacts) often performs better in practice** than no guidance. The authors suspect this relates to imperfect score estimators.
- The analysis uses an isotropic GMM. Although extensions to more complex distributions are provided, the gap between this and real text-to-image distributions remains.
- The relative merits of Power-Law against other nonlinear strategies, especially in pixel space, or the impact of score approximation errors, are areas for future research.

## Related Work & Insights
- **vs. Chidambaram et al. 2024 / Wu et al. 2024**: They proved CFG causes overshoot/shrinkage in low-D GMMs. This work completes the picture by showing recovery in high-D.
- **vs. Limited-interval CFG / CADS / APG / CFG++**: These methods remain linear regarding the score difference. Power-Law is the first to introduce nonlinearity to the score difference itself and can be combined with these methods for additive gains.
- **vs. Standard CFG (Ho & Salimans 2022)**: While standard CFG trades diversity for quality, Power-Law mitigates diversity loss while retaining or improving quality.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Uses the phase transition framework to flip the "CFG necessarily distorts" consensus and provides a provable, minimal improvement.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers GMM, DiT, EDM2, and MMDiT, though lacks public code.
- Writing Quality: ⭐⭐⭐⭐ Clear theoretical chain, though statistical physics notation may be challenging for some readers.
- Value: ⭐⭐⭐⭐⭐ Plug-and-play and theoretically sound, offering direct utility for all diffusion/flow matching models using CFG.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Improving Classifier-Free Guidance in Masked Diffusion: Low-Dim Theoretical Insights with High-Dim Impact](improving_classifier-free_guidance_in_masked_diffusion_low-dim_theoretical_insig.md)
- [\[ICLR 2026\] Stage-wise Dynamics of Classifier-Free Guidance in Diffusion Models](stage-wise_dynamics_of_classifier-free_guidance_in_diffusion_models.md)
- [\[CVPR 2026\] C$^2$FG: Control Classifier-Free Guidance via Score Discrepancy Analysis](../../CVPR2026/image_generation/c2fg_control_classifier-free_guidance_via_score_discrepancy_analysis.md)
- [\[AAAI 2026\] Studying Classifier(-Free) Guidance From A Classifier-Centric Perspective](../../AAAI2026/image_generation/studying_classifier-free_guidance_from_a_classifier-centric_perspective.md)
- [\[AAAI 2026\] DICE: Distilling Classifier-Free Guidance into Text Embeddings](../../AAAI2026/image_generation/dice_distilling_classifier-free_guidance_into_text_embedding.md)

</div>

<!-- RELATED:END -->

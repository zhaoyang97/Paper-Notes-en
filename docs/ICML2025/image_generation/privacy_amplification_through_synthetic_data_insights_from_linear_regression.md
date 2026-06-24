---
title: >-
  [Paper Note] Privacy Amplification Through Synthetic Data: Insights from Linear Regression
description: >-
  [ICML 2025][Image Generation][Differential Privacy] Under the linear regression framework, it is proved that synthetic data cannot provide privacy amplification when the adversary controls the seed. However, releasing a limited amount of synthetic data under random inputs can achieve a privacy amplification effect that exceeds the DP guarantee of the model itself, with an amplification rate of $O(1/d)$.
tags:
  - "ICML 2025"
  - "Image Generation"
  - "Differential Privacy"
  - "Privacy Amplification"
  - "Synthetic Data"
  - "Linear Regression"
  - "f-DP"
  - "Rényi Divergence"
date: 2026-05-08
content_hash: de98289e1e329df0
---

# Privacy Amplification Through Synthetic Data: Insights from Linear Regression

**Conference**: ICML 2025  
**arXiv**: [2506.05101](https://arxiv.org/abs/2506.05101)  
**Code**: None  
**Area**: Image Generation  
**Keywords**: Differential Privacy, Privacy Amplification, Synthetic Data, Linear Regression, f-DP, Rényi Divergence

## TL;DR

Under the linear regression framework, it is proved that synthetic data cannot provide privacy amplification when the adversary controls the seed. However, releasing a limited amount of synthetic data under random inputs can achieve a privacy amplification effect that exceeds the DP guarantee of the model itself, with an amplification rate of $O(1/d)$.

## Background & Motivation

Differential Privacy (DP) is the gold standard for privacy protection. After training a DP generative model, the synthetic data inherits the DP guarantee of the model (post-processing inequality). However, empirical studies indicate that synthetic data may actually provide closer to **stronger privacy protection** than the model itself.

**Core Problem**: Does releasing synthetic data (rather than the model itself) exhibit a **privacy amplification** effect?

- The post-processing inequality yields an **upper bound**—is it too conservative?
- Intuition: When the number of released synthetic data points is much smaller than the model complexity, the privacy leakage should decrease.
- However, rigorous theoretical analysis has been lacking until now.

**Methodology Choice**: Linear regression is used as the research framework because:
1. It is analytically tractable.
2. It is sufficiently expressive—capable of capturing phenomena such as double descent and model collapse.
3. It lays the foundation for generalization to more complex models.

## Method

### Overall Architecture

Analyzed using the **f-DP** and **Rényi DP (RDP)** frameworks. Settings:
- Dataset $\mathcal{D} = (X, Y)$, $X \in \mathbb{R}^{d \times m}$, $Y \in \mathbb{R}^{n \times m}$
- Linear model $\hat{Y} = wX$
- Output perturbation mechanism: $\mathcal{M}(\mathcal{D}) = \arg\min_w F_\lambda(w; \mathcal{D}) + \sigma_\theta N$

### Key Designs

#### Part 1: Fixed Seed—Negative Results (Section 3)

**When the adversary controls the seed $z$**:

**Proposition 3.1** (Output Perturbation): For any fixed $z \in \mathbb{R}^d$, there exist neighboring datasets $\mathcal{D}, \mathcal{D}'$ such that:
$$T(Vz, Wz) = T(V, W)$$
That is, a single synthetic data point can leak the same amount of privacy as releasing the entire model.

**Reason**: The adversary can choose $z$ as the right singular vector associated with the largest singular value of $\mu = w^* - v^*$, maximizing the signal-to-noise ratio.

**Proposition 3.3** (Noisy Gradient Descent): The same negative result applies to models trained with NGD.

#### Part 2: Random Seed—Positive Results (Section 4)

**Definition 4.1**: Seed $Z \in \mathbb{R}^{d \times l}$, $Z_{ij} \sim \mathcal{N}(0, \sigma_z^2)$, releasing $\mathcal{M}_Z(v) = \mathcal{M}(v)Z$.

Core insight: $VZ$ can be decomposed into a sum of independent terms $VZ = \sum_{k=1}^d V_k Z_k$, applying CLT approximation.

**Lemma 4.3** (Single-Point Case $n=l=1$):
$$TV(\sqrt{d}(\sigma_\theta N + v)Z, \sigma_z\sqrt{d\sigma_\theta^2 + \|v\|^2} G) \leq \frac{A_{\|v\|}}{d}$$

Namely, $VZ$ and $WZ$ approach Gaussian distributions with **differing variances** (rather than differing means); the privacy issue shifts from "mean shift" to "variance shift".

**Theorem 4.3** (Single-Point Privacy Amplification): The trade-off function $T(VZ, WZ)$ converges to the trade-off function of Gaussians with differing variances at a rate of $O(1/d)$.

**Asymptotic Results for Rényi Divergence**:
$$D_\alpha(\nu_{v_*}^d, \nu_{w_*}^d) = \frac{\alpha\Delta^2}{4d\sigma_\theta^2} + o(d^{-1}) \approx \frac{1}{2d} D_\alpha(V, W)$$
implying an $O(1/d)$ level of privacy amplification.

#### Part 3: Multi-Point Release (Section 4.3)

Utilizing the convergence results of Gaussian matrix products from Li & Woodruff (2021):

**Theorem 4.5** (Convergence of Gaussian Matrix Products with Drift):
$$TV((\sigma_\theta N + v)Z, \sigma_\theta\sqrt{d-s}G + vZ') \leq C'\sqrt{\frac{nls}{d-s}}$$

where $s = \text{rank}(v)$. **Key Property**: The bound does not depend on the norm of the drift $v$.

**Theorem 4.6**: The Rényi divergence for $l$ synthetic points of dimension $n$:
$$D_\alpha(G_v, G_w) \leq \frac{\alpha nl \Delta^2}{4(d-n)\sigma_\theta^2} + o(d^{-1})$$

### Loss & Training

This work is a theoretical analysis and does not involve training. The core technical tools include:
- The trade-off function framework of f-DP
- Non-asymptotic CLT (Bally & Caramellino, 2016)
- TV convergence of Gaussian matrix products (Li & Woodruff, 2021)
- Pinsker's inequality + chain rule for KL divergence

## Key Experimental Results

### Main Results

**Numerical Verification** (Single-Point Release, Figure 2):
- Uses Monte Carlo simulation to estimate the Rényi divergence $D_\alpha(VZ, WZ)$ ($L=10^6$ samples, $M=50$ repetitions)
- Under the high privacy regime ($\Delta < 1$), the decay trend of $l_\alpha(h) \approx O(1/d)$ is verified

**Multi-Point Release** (Figure 3, $l=10, n=1$):
- The Rényi divergence decays at a rate of $O(d^{-1/2})$ as $d$ increases (with fixed $l, n$)

### Key Findings

| Scenario | Privacy Amplification? | Amplification Degree |
|------|---------|----------|
| Fixed seed (adversary-controlled) | ❌ No | $T(Vz,Wz) = T(V,W)$ |
| Random seed, 1 point released ($l=1, n=1$) | ✅ Yes | $\approx \frac{1}{2d} D_\alpha(V,W)$ |
| Random seed, $l$ points released ($n$-dimensional) | ✅ Yes | $\approx \frac{nl}{2(d-n)} D_\alpha(V,W)$ |

Condition: $d \geq \max\{n, l\}$

## Highlights & Insights

1. **A Complete Picture of Both Sides**: It proves both the impossibility of amplification under fixed seeds (negative) and the amplification under random seeds (positive), revealing that **hidden randomness** is the key to privacy amplification.
2. **Elegant Shift from 'Mean Shift' to 'Variance Shift'**: Synthetic data generation transforms the mean difference between two distributions into a variance difference, essentially "diluting" the privacy signal.
3. **Bounds Independent of Drift Norm** (Theorem 4.5): The convergence result of the Gaussian matrix product with drift is an independently interesting mathematical contribution.
4. **Composition Properties**: The result of multi-point release naturally behaves as a composition of single-point mechanisms, with the $nl$ factor growing linearly.
5. **Best of Both Worlds**: DP generative models enjoy both post-processing guarantees (when releasing large amounts) and composition guarantees (when releasing small amounts, which is superior).

## Limitations & Future Work

1. **Limited to Linear Regression**: The generalization of the conclusions to non-linear generative models remains unclear.
2. **Conservatism of CLT Constants**: The constants in non-asymptotic CLT might lead to loose bounds for small $d$ (manifested as the initial plateau in Figure 2).
3. **Constraint of $d \geq \max\{n, l\}$**: It requires the model dimension to be larger than the output dimension and the number of released samples, making it inapplicable to low-dimensional models.
4. **Indirection of Rényi Divergence Bounds**: The $O(1/d)$ convergence of the trade-off function does not directly imply the $O(1/d)$ convergence of Rényi divergence, which requires numerical verification.
5. **Gaussian Seed Assumption**: The input distribution of practical generative models might not be Gaussian.
6. **Theorem 4.7 of Li & Woodruff (2021)**: The Gaussian product does not converge when $d < \max\{n,l\}$, indicating fundamental limits to the amplification effect.

## Related Work & Insights

- **DP Synthetic Data Generation**: Zhang et al. (2017), McKenna et al. (2019, 2021), Dockhorn et al. (2023), etc.
- **Privacy Amplification**: Iterative privacy amplification by Balle et al. (2018), Feldman et al. (2018), which differs from the setting in this paper.
- **Gaussian Matrix Products**: The TV convergence theorem of Li & Woodruff (2021) forms the foundation of the multi-point analysis.
- **f-DP Framework**: The trade-off function by Dong et al. (2022) provides a refined tool for privacy analysis.
- Neunhoeffer et al. (2024): Proved DP of synthetic data for a simple one-dimensional model, but under a simpler setting.
- **Insights**: The idea of using "hidden randomness" as a privacy amplification mechanism could be generalized to more broad post-processing scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐ — First to rigorously analyze privacy amplification in synthetic data, with complementary positive and negative results.
- Experimental Thoroughness: ⭐⭐⭐ — Primarily theoretical, with numerical verification supporting theoretical predictions.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear paper structure, progressing step-by-step from simple to complex settings.
- Value: ⭐⭐⭐⭐ — Provides a theoretical foundation for the privacy of synthetic data, albeit limited to linear regression.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Increasing the Utility of Synthetic Images through Chamfer Guidance](../../NeurIPS2025/image_generation/increasing_the_utility_of_synthetic_images_through_chamfer_guidance.md)
- [\[CVPR 2025\] Enhancing Vision-Language Compositional Understanding with Multimodal Synthetic Data (SPARCL)](../../CVPR2025/image_generation/enhancing_vision-language_compositional_understanding_with_multimodal_synthetic_.md)
- [\[CVPR 2025\] Training Data Provenance Verification: Did Your Model Use Synthetic Data from My Generative Model for Training?](../../CVPR2025/image_generation/training_data_provenance_verification_did_your_model_use_synthetic_data_from_my_.md)
- [\[ICLR 2026\] SMOTE and Mirrors: Exposing Privacy Leakage from Synthetic Minority Oversampling](../../ICLR2026/image_generation/smote_and_mirrors_exposing_privacy_leakage_from_synthetic_minority_oversampling.md)
- [\[ICCV 2025\] Generating Multi-Image Synthetic Data for Text-to-Image Customization](../../ICCV2025/image_generation/generating_multi-image_synthetic_data_for_text-to-image_customization.md)

</div>

<!-- RELATED:END -->

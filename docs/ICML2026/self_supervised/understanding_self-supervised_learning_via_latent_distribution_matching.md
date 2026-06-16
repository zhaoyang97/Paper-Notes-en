---
title: >-
  [Paper Note] Understanding Self-Supervised Learning via Latent Distribution Matching
description: >-
  [ICML 2026][Self-Supervised Learning][Paper Note] The authors unify contrastive, non-contrastive, and predictive SSL into "Latent Distribution Matching (LDM)": maximizing the log-probability of samples under a hypothesized latent model (alignment) plus maximizing latent entropy (uniformity). Based on this, they derive a nonlinear identifiable predictive SSL with a Kal
tags:
  - ICML 2026
  - Self-Supervised Learning
date: 2026-05-08
content_hash: 1804c68aabdf37e2
---
# Understanding Self-Supervised Learning via Latent Distribution Matching

**Conference**: ICML 2026 Spotlight  
**arXiv**: [2605.03517](https://arxiv.org/abs/2605.03517)  
**Code**: None  
**Area**: Self-Supervised Representation Learning / ICA and Identifiability / Theory of Representation Learning  
**Keywords**: Self-Supervised Learning, Latent Distribution Matching, Nonlinear ICA, Identifiability, Kalman Prediction

## TL;DR
The authors unify contrastive, non-contrastive, and predictive SSL into "Latent Distribution Matching (LDM)": maximizing the log-probability of samples under a hypothesized latent model (alignment) plus maximizing latent entropy (uniformity). Based on this, they derive a nonlinear identifiable predictive SSL with a Kalman predictor.

## Background & Motivation
**Background**: SSL has become the mainstream for vision, language, and audio representation learning, featuring a diverse spectrum of methods—SimCLR, VICReg, BYOL, SimSiam, CPC, JEPA, etc.—each with its own loss formulation and interpretation.

**Limitations of Prior Work**: (1) The geometric alignment perspective (Wang & Isola 2020) provides intuitive explanations but lacks a rigorous statistical foundation, failing to explain methods like BYOL/SimSiam that lack explicit repulsion. (2) The MI maximization perspective is neither necessary nor sufficient because mutual information remains invariant under any invertible transformation ($I[x,y]=I[\phi(x),\psi(y)]$). (3) Predictive SSL (CPC, JEPA, I-JEPA) achieves empirical SOTA, but its objective functions and regularizations are heuristically combined, lacking derivable design principles and identifiability guarantees.

**Key Challenge**: Existing methods have respective strengths but lack a unifying objective that simultaneously explains why SSL produces useful representations and provides proofs of identifiability.

**Goal**: (1) Discover a unified objective governing ICA, and contrastive, non-contrastive, predictive, and stopgrad-based SSL. (2) Clarify the actual role of MI maximization. (3) Derive new SSL variants (e.g., Kalman-based predictive SSL). (4) Provide identifiability guarantees for predictive SSL.

**Key Insight**: Return to the likelihood perspective—for an invertible encoder, performing MLE in the latent space is equivalent to matching the data distribution to the model distribution; extending this to paired views results in joint LDM.

**Core Idea**: SSL is uniformly expressed as $\mathcal F_{\mathrm{LDM}}=-D_{\mathrm{KL}}[R(z,z')\,\|\,P_\theta(z,z')]=\underbrace{\langle\log P_\theta(z,z')\rangle_R}_{\text{alignment}}+\underbrace{H_R[z,z']}_{\text{uniformity}}$. Different SSL algorithms correspond to different choices of $P_\theta$ and entropy estimators.

## Method

### Overall Architecture
The authors start from maximum likelihood: for an invertible encoder $f$, $\langle\log P_\theta(x)\rangle_{P_{\mathrm{data}}}\propto\langle\log P_\theta(f(x))\rangle+H_{P_{\mathrm{data}}}[f(x)]=-D_{\mathrm{KL}}[P_{\mathrm{data}}(f(x))\|P_\theta(f(x))]$. Linear ICA is a special case of this. By extending views to paired data $(x, x')$ and matching the latent distribution $R(z, z')$ with the model $P_\theta(z, z')$, the LDM objective is obtained. The authors then align LDM with the MI variant by Aitchison & Ganev, $\mathcal F_{\mathrm{MI}}=\langle\log P_\theta\rangle_R+2H_R[z]$, proving that when the encoder is nearly invertible, MI is implicitly saturated by entropy regularization. Finally, based on choices of $P_\theta$ and entropy estimators, VICReg, SimCLR, CPC, BYOL/SimSiam, JEPA, and the new Kalman-predictive SSL are all categorized into a single table (Table 1).

### Key Designs

**1. LDM Unified Objective + Three Categories of Entropy Estimators: Turning Five SSL Families into Two Knobs**

Previously, every SSL method told its own story—SimCLR focused on contrast, VICReg on variance regularization, and BYOL on stopgrad, with little apparent connection. LDM unifies them into a single objective: $\mathcal F_{\mathrm{LDM}}=-D_{\mathrm{KL}}[R(z,z')\|P_\theta(z,z')]$, where the alignment term comes from $\log P_\theta$ (seeking aligned representations) and the uniformity term comes from $H_R$ (seeking spread-out representations to avoid collapse). The differences lie in two knobs: the shape of the latent distribution $P_\theta$ and how entropy $H_R$ is estimated.

Entropy estimators fall into three categories corresponding to major families: Kernel Density Estimation (KDE) → Contrastive SSL (the negative samples in SimCLR act as the KDE bandwidth $1/\beta$); Parametric Gaussian → Non-contrastive SSL (the covariance regularization in VICReg is exactly the Taylor expansion of $\log|\Sigma_z|$); Conditional entropy plug-in → Stopgrad/predictor systems (BYOL, JEPA). By adjusting these two knobs, the "distinct" losses reveal a common skeleton, directly showing how to design new algorithms by simply changing the $P_\theta$ shape or the entropy estimator.

**2. Clarifying the Real Role of MI Maximization: It is Almost a Redundant Term**

"Maximizing Mutual Information" has been an SSL slogan for a long time, yet its actual importance remained unclear. LDM provides a clean determination: $\mathcal F_{\mathrm{MI}}-\mathcal F_{\mathrm{LDM}}=I_R[z,z']$. For nearly invertible encoders, $I_R[z,z']$ automatically saturates, making the actual contribution of the MI term negligible. The paper compares 8 combinations of "Latent Space × Entropy Estimator × Inclusion of MI" (Table 2, Fig. 3), finding that whether MI is included has almost no impact on linear probing accuracy or representation dimensionality. What truly matters are the latent space assumptions and the entropy estimator. This transforms a vague slogan into a falsifiable conclusion and suggests that future work need not overly complicate objectives to derive MI bounds.

**3. Predictive SSL: Kalman Latent Dynamics + Identifiability Proof, Providing a Theoretical Backbone for JEPA**

Predictive methods like JEPA and CPC are empirically SOTA, but their objectives and regularizations are heuristically patched. LDM models latent space transitions as $P_\theta(z'|z)$, selecting a Kalman-style linear Gaussian transition combined with a nonlinear encoder (manifold normalizing flow / injective flow), and applies $\mathcal F_{\mathrm{LDM}}$ to $(z, z')$. Theoretical proof shows that under mild assumptions, even with a nonlinear predictor, predictive LDM can recover latent variables up to an affine equivalence class (identifiability up to affine). This step explains why JEPA is stable and identifiable and simultaneously provides a sampling-free Bayesian filtering version that can serve as a new baseline.

### Loss & Training
Specific losses vary by $P_\theta$ and entropy estimator choices: VICReg corresponds to $-\frac{1}{2\sigma^2}\langle\|f(x)-f(x')\|^2\rangle+\log|\Sigma_z|$; the LDM version uses $\log|\Sigma_{(z,z')}|$; SimCLR corresponds to $\langle\beta f(x)^\top f(x')\rangle-2\langle\log\langle\exp\{\beta f(x)^\top f(x^-)\}\rangle\rangle$ (KDE entropy estimation + spherical vMF); Predictive SSL uses Kalman gain instead of a momentum target and utilizes stopgrad for the conditional entropy plugin.

## Key Experimental Results

### Main Results

| Dataset / Setting | Knob Combination | Top-1 acc | Note |
|---------------|----------|-----------|------|
| ImageNet-100, Plane × LogDet × LDM | VICReg-LDM | 75.9 | LDM version slightly outperforms MI version (74.7) |
| CIFAR-100, Plane × LogDet × LDM | Same as above | 69.5 | Significant gap over original VICReg-MI (65.3) |
| ImageNet-100, Sphere × Contr. × MI | SimCLR | 73.1 | Classic SimCLR baseline |
| CIFAR-10 | Plane × kNN × LDM | 92.1 | kNN entropy estimation is a practical alternative for LDM |

### Ablation Study

| Knob | Key Observation | Interpretation |
|------|----------|------|
| With vs. Without MI ($\mathcal F_{\mathrm{MI}}$ vs. $\mathcal F_{\mathrm{LDM}}$) | Accuracy difference $\le \pm 0.4$ across datasets | The MI term is implicitly absorbed by entropy regularization and can be omitted. |
| Latent Space (Plane vs. Sphere) | Plane + LogDet is significantly higher on CIFAR-100 / ImageNet-100 | The "shape" assumption of $P_\theta(z)$ has the largest impact. |
| Entropy Estimator | LogDet > kNN ≈ KDE > parametric Gaussian (Spherical) | Different assumptions determine the risk of collapse. |
| Predictive LDM with Kalman | Gains over BYOL/JEPA style baselines on temporal tasks | Explicitly modeling transition noise is more stable. |

### Key Findings
- LDM and MI versions are nearly equivalent: This further indicates that the core of SSL quality is the pair $(P_\theta, H \text{ estimator})$, rather than whether mutual information is maximized. This discovery shifts engineering focus from "picking MI estimators" back to "picking latent models."
- The Kalman variant of predictive LDM provides a "no-collapse + identifiable + sampling-free" trifecta, representing a rare predictive SSL that benefits both theory and engineering.
- Explaining BYOL/SimSiam as a conditional entropy plugin in Table 1 is a key insight: the stopgrad design, long considered "difficult to explain," fits naturally within the LDM framework.

## Highlights & Insights
- Strong unifying power: A single table classifies five major SSL families plus ICA, where every key design corresponds to a specific knob in the LDM framework, directly guiding the design of new algorithms.
- Interpreting the stopgrad in BYOL / JEPA as a conditional entropy plugin is a true "aha" moment, showing that stopgrad is not just an engineering hack.
- Providing rigorous identifiability results is particularly important for theoretical SSL researchers—it offers a first-principles explanation for why predictive SSL works.
- Kalman-based latent dynamics is a ready-to-use new baseline, reusable for research in temporal data, robotics, and world models.

## Limitations & Future Work
- Experiments focus mainly on image SSL and simple temporal tasks; large-scale video or multimodal pre-training is not covered, so the framework's universality still needs verification.
- LDM still requires the encoder to be "nearly invertible on the data manifold," which may not hold for very noisy real-world data.
- Identifiability results are within an affine equivalence class; downstream tasks may still require disentanglement post-processing.
- Lacks in-depth analysis of training dynamics for EMA targets and predictor networks.
- While the choice of entropy estimator is identified as a decisive factor, specific criteria for selection in new tasks are not provided, still requiring empirical tuning.
- Algorithmic details for Kalman-based predictive SSL are simplified in the main text; engineering implementation details (e.g., prior covariance initialization) require reading the appendix.

## Related Work & Insights
- **vs. Wang & Isola 2020 (alignment-uniformity)**: They proposed an intuitive version of geometric alignment; this paper formalizes it as distribution matching and explains why BYOL works without explicit uniformity—provided implicitly by the conditional entropy plugin.
- **vs. Zimmermann et al. 2021 (CPC identifiability)**: They proved CPC is identifiable; this paper embeds their results into the more general LDM framework, proving that predictive SSL remains identifiable even with nonlinear predictors.
- **vs. Aitchison & Ganev 2024 (variational SSL)**: They used a variational perspective for $\mathcal F_{\mathrm{MI}}$; this paper proves the MI term is largely redundant and that distribution matching is the core.
- **vs. Shwartz-Ziv et al. 2023 (info-theoretic VICReg)**: This paper derives VICReg's covariance regularization directly from LDM and proposes $\log|\Sigma_{(z,z')}|$ joint covariance as a tighter alternative.
- **vs. Halvagal et al. 2023 / Tian et al. 2021 (BYOL dynamics)**: They analyzed why stopgrad and EMA targets prevent collapse; this paper reinterprets stopgrad as a "conditional entropy plugin," a perspective that is conceptually more unified and aligns with identifiability proofs.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ A single objective unifies ICA, contrastive, non-contrastive, predictive, and stopgrad families, supported by identifiability proofs.
- Experimental Thoroughness: ⭐⭐⭐ Systematic comparison of 8 knob combinations across multiple datasets, though lacking verification on large-scale ImageNet-1K or long-term temporal benchmarks.
- Writing Quality: ⭐⭐⭐⭐ Clear derivations and a highly condensed Table 1 make it accessible even to non-theoretical readers.
- Value: ⭐⭐⭐⭐ Provides both a unified theoretical framework and a new Kalman-based predictive SSL algorithm, offering a long-term toolkit for designing and explaining SSL.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Understanding Ice Crystal Habit Diversity with Self-Supervised Learning](../../NeurIPS2025/self_supervised/understanding_ice_crystal_habit_diversity_with_self-supervised_learning.md)
- [\[ICML 2026\] Beyond Distribution Estimation: Simplex Anchored Structural Inference Towards Universal Semi-Supervised Learning](beyond_distribution_estimation_simplex_anchored_structural_inference_towards_uni.md)
- [\[ICML 2026\] Can Local Learning Match Self-Supervised Backpropagation?](can_local_learning_match_self-supervised_backpropagation.md)
- [\[ICLR 2026\] Soft Equivariance Regularization for Invariant Self-Supervised Learning](../../ICLR2026/self_supervised/soft_equivariance_regularization_for_invariant_self-supervised_learning.md)
- [\[ICLR 2026\] InfoNCE Induces Gaussian Distribution](../../ICLR2026/self_supervised/infonce_induces_gaussian_distribution.md)

</div>

<!-- RELATED:END -->

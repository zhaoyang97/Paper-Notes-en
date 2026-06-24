---
title: >-
  [Paper Note] Understanding Self-Supervised Learning via Latent Distribution Matching
description: >-
  [ICML 2026 Spotlight][Self-Supervised Learning][Latent Distribution Matching] The authors unify contrastive, non-contrastive, and predictive SSL as "Latent Distribution Matching (LDM)": maximizing the log-probability of samples under a hypothesized latent model (alignment) plus maximizing latent entropy (uniformity). Based on this, they derive a nonlinear identifiable predictive SSL equipped with a Kalman predictor.
tags:
  - "ICML 2026 Spotlight"
  - "Self-Supervised Learning"
  - "Latent Distribution Matching"
  - "Nonlinear ICA"
  - "Identifiability"
  - "Kalman prediction"
date: 2026-05-08
content_hash: ea753bb3c8b543c8
---

# Understanding Self-Supervised Learning via Latent Distribution Matching

**Conference**: ICML 2026 Spotlight  
**arXiv**: [2605.03517](https://arxiv.org/abs/2605.03517)  
**Code**: None  
**Area**: Self-Supervised Representation Learning / ICA and Identifiability / Representation Learning Theory  
**Keywords**: Self-supervised learning, Latent Distribution Matching, Nonlinear ICA, Identifiability, Kalman prediction

## TL;DR
The authors unify contrastive, non-contrastive, and predictive SSL as "Latent Distribution Matching (LDM)": maximizing the log-probability of samples under a hypothesized latent model (alignment) plus maximizing latent entropy (uniformity). Based on this, they derive a nonlinear identifiable predictive SSL equipped with a Kalman predictor.

## Background & Motivation
**Background**: SSL has become the mainstream for representation learning in vision, language, and audio. The methodological landscape is diverse—comprising SimCLR, VICReg, BYOL, SimSiam, CPC, JEPA, etc.—each with its own loss formulation and interpretation.

**Limitations of Prior Work**: (1) The geometric alignment perspective (Wang & Isola 2020) is intuitive but lacks a rigorous statistical foundation, failing to explain methods like BYOL/SimSiam that lack explicit repulsion. (2) The MI (Mutual Information) maximization perspective is neither necessary nor sufficient because MI is invariant under any invertible transformation ($I[x,y]=I[\phi(x),\psi(y)]$). (3) Predictive SSL (CPC, JEPA, I-JEPA) achieves empirical SOTA, but their objective functions and regularizations are often heuristic combinations, lacking derivable design principles and identifiability guarantees.

**Key Challenge**: Existing methods have respective strengths but lack a unifying objective that simultaneously explains why SSL produces useful representations and provides identifiability proofs.

**Goal**: (1) Identify a unified objective covering ICA and contrastive, non-contrastive, predictive, and stopgrad-based SSL. (2) Clarify the actual role of MI maximization. (3) Derive new SSL variants (e.g., Kalman-based predictive SSL). (4) Provide identifiability guarantees for predictive SSL.

**Key Insight**: Return to the likelihood perspective: for an invertible encoder, MLE in the latent space is equivalent to matching the data distribution to the model distribution. Extending this to paired views results in joint LDM.

**Core Idea**: Unify SSL as $\mathcal F_{\mathrm{LDM}}=-D_{\mathrm{KL}}[R(z,z')\,\|\,P_\theta(z,z')]=\underbrace{\langle\log P_\theta(z,z')\rangle_R}_{\text{alignment}}+\underbrace{H_R[z,z']}_{\text{uniformity}}$. Different SSL algorithms correspond to different choices of $P_\theta$ and entropy estimators.

## Method

### Overall Architecture
The authors start from maximum likelihood: for an invertible encoder $f$, $\langle\log P_\theta(x)\rangle_{P_{\mathrm{data}}}\propto\langle\log P_\theta(f(x))\rangle+H_{P_{\mathrm{data}}}[f(x)]=-D_{\mathrm{KL}}[P_{\mathrm{data}}(f(x))\|P_\theta(f(x))]$, where linear ICA is a special case. By extending views to paired data $(x,x')$ and matching the latent record $R(z,z')$ with the model $P_\theta(z,z')$, the LDM objective is obtained. They further compare LDM with the MI variant from Aitchison & Ganev, $\mathcal F_{\mathrm{MI}}=\langle\log P_\theta\rangle_R+2H_R[z]$, proving that when the encoder is nearly invertible, MI is implicitly saturated by entropy regularization. Finally, based on the choices of $P_\theta$ and entropy estimators, VICReg, SimCLR, CPC, BYOL/SimSiam, JEPA, and the new Kalman-predictive SSL are all consolidated into a single framework (Table 1).

### Key Designs

**1. LDM Unified Objective + Three Entropy Estimator Categories: Reducing Five SSL Families to Two Knobs**

Previously, each SSL method had its own narrative—SimCLR focused on contrast, VICReg on variance regularization, and BYOL on stopgrad—making their relationships opaque. LDM unifies them under one objective: $\mathcal F_{\mathrm{LDM}}=-D_{\mathrm{KL}}[R(z,z')\|P_\theta(z,z')]$, where the alignment term comes from $\log P_\theta$ (requiring aligned representations) and the uniformity term comes from $H_R$ (requiring spread-out representations to avoid collapse). Differences lie in two "knobs": the shape of the latent distribution $P_\theta$ and the method of estimating entropy $H_R$.

Entropy estimators fall into three categories corresponding to the major families: Kernel Density Estimation (KDE) $\rightarrow$ Contrastive SSL (negative samples in SimCLR act as KDE bandwidth $1/\beta$); Parametric Gaussian $\rightarrow$ Non-contrastive SSL (covariance regularization in VICReg is the Taylor expansion of $\log|\Sigma_z|$); and Conditional entropy plug-in $\rightarrow$ Stopgrad/predictor systems (BYOL, JEPA). By adjusting these knobs, disparate loss functions reveal a common skeleton, directly suggesting how to design new algorithms by merely changing the $P_\theta$ shape or the entropy estimator.

**2. Clarifying the Role of MI Maximization: It is Almost Redundant**

"Maximizing mutual information" has long been a slogan in SSL, yet its actual importance remained unclear. LDM provides a clean determination: $\mathcal F_{\mathrm{MI}}-\mathcal F_{\mathrm{LDM}}=I_R[z,z']$. For nearly invertible encoders, $I_R[z,z']$ saturates automatically, meaning the actual contribution of the MI term is minimal. The paper uses eight combinations of "latent space $\times$ entropy estimator $\times$ with/without MI" for comparison (Table 2, Fig. 3), finding that the presence of MI barely affects linear probing accuracy or representation dimensionality. What truly matters are the latent space assumptions and the entropy estimator. This transforms a vague slogan into a falsifiable conclusion and suggests that future work need not over-complicate objectives to derive MI bounds.

**3. Predictive SSL: Kalman Latent Dynamics + Identifiability Proof, Providing a Theoretical Skeleton for JEPA**

Predictive methods like JEPA/CPC are empirically SOTA, but their objectives and regularizations are often heuristic. LDM models latent space transitions as $P_\theta(z'|z)$, selecting a Kalman-style linear Gaussian transition paired with a nonlinear encoder (manifold normalizing flow / injective flow), and applies $\mathcal F_{\mathrm{LDM}}$ to $(z,z')$. It is theoretically proven that under mild assumptions, even with a nonlinear predictor, predictive LDM can recover latent variables up to an affine equivalence class (identifiability up to affine). This step explains why JEPA is stable and identifiable and provides a sampling-free Bayesian filtering version that serves as a new baseline.

### Loss & Training
Specific losses depend on the choice of $P_\theta$ and the entropy estimator: VICReg corresponds to $-\frac{1}{2\sigma^2}\langle\|f(x)-f(x')\|^2\rangle+\log|\Sigma_z|$; the LDM version uses $\log|\Sigma_{(z,z')}|$; SimCLR corresponds to $\langle\beta f(x)^\top f(x')\rangle-2\langle\log\langle\exp\{\beta f(x)^\top f(x^-)\}\rangle\rangle$ (KDE entropy estimation + spherical vMF); Predictive SSL uses Kalman gain instead of a momentum target and utilizes stopgrad for the conditional entropy plugin.

## Key Experimental Results

### Main Results

| Dataset / Setting | Knob Combination | Top-1 acc | Description |
|---------------|----------|-----------|------|
| ImageNet-100, Plane × LogDet × LDM | VICReg-LDM | 75.9 | LDM version slightly outperforms MI version (74.7) |
| CIFAR-100, Plane × LogDet × LDM | Same as above | 69.5 | Significant gap over original VICReg-MI (65.3) |
| ImageNet-100, Sphere × Contr. × MI | SimCLR | 73.1 | Classic SimCLR baseline |
| CIFAR-10 | Plane × kNN × LDM | 92.1 | kNN entropy estimation is a practical alternative for LDM |

### Ablation Study

| Knob | Key Observation | Interpretation |
|------|----------|------|
| With vs. Without MI ($\mathcal F_{\mathrm{MI}}$ vs. $\mathcal F_{\mathrm{LDM}}$) | Accuracy difference $\le \pm 0.4$ across datasets | MI term is implicitly absorbed by entropy regularization and can be omitted |
| Latent Space (Plane vs. Sphere) | Plane + LogDet significantly higher on CIFAR-100 / ImageNet-100 | The "shape" assumption of $P_\theta(z)$ has the greatest impact |
| Entropy Estimator | LogDet > kNN ≈ KDE > parametric Gaussian (Spherical) | Different assumptions determine collapse risk |
| Predictive LDM with Kalman | Improvement over BYOL/JEPA style baselines in sequence tasks | Explicitly modeling transition noise is more stable |

### Key Findings
- LDM and MI versions are nearly equivalent: This further indicates that the core of SSL quality is the $(P_\theta, H \text{ estimator})$ pair, not whether mutual information is maximized. This shifts engineering focus from picking MI estimators back to picking latent models.
- The Kalman variant of predictive LDM provides a "no-collapse + identifiable + sampling-free" trifecta, representing one of the few predictive SSL methods providing both theoretical and engineering benefits.
- Interpreting BYOL/SimSiam's stopgrad as a conditional entropy plugin in Table 1 is a key insight: the stopgrad design, long considered an "unexplained hack," naturally fits within the LDM framework.

## Highlights & Insights
- Exceptional unifying power: A single table categorizes five major SSL families plus ICA, mapping key designs to specific knobs in the LDM framework, which directly guides the design of new algorithms.
- Redefining stopgrad in BYOL/JEPA as a "conditional entropy plugin" is a true "aha" insight, showing it is more than just an engineering hack.
- Provides rigorous identifiability results, which are particularly important for theoretically inclined researchers—it offers a first-principles explanation for why predictive SSL works.
- Kalman-based latent dynamics is a practical new baseline, reusable for research in sequences, robotics, or world models.

## Limitations & Future Work
- Experiments primarily focus on image SSL and simple sequence tasks, lacking coverage of large-scale video or multimodal pre-training; thus, universality requires further validation.
- LDM still requires the encoder to be "nearly invertible on the data manifold," which may not hold for very noisy real-world data.
- Identifiability results are within the affine equivalence class; downstream tasks may still require disentanglement post-processing.
- Lacks an in-depth analysis of the training dynamics of EMA targets and predictor networks.
- While entropy estimator choice is identified as a determining factor, specific criteria for systematic selection in new tasks are not provided, still requiring empirical tuning.
- Algorithmic details for Kalman-based predictive SSL are simplified in the main text; implementation details (e.g., prior covariance initialization) require reading the appendix.

## Related Work & Insights
- **vs Wang & Isola 2020 (alignment-uniformity)**: They proposed a geometric intuition for alignment; this paper formalizes it as distribution matching and explains why BYOL works without explicit uniformity—it is implicitly provided by the conditional entropy plugin.
- **vs Zimmermann et al. 2021 (CPC identifiability)**: They proved CPC identifiability; this paper embeds those results into the more general LDM framework, proving that predictive SSL remains identifiable even with nonlinear predictors.
- **vs Aitchison & Ganev 2024 (variational SSL)**: They used a variational perspective for $\mathcal F_{\mathrm{MI}}$; this paper proves the MI term is largely redundant, with distribution matching being core.
- **vs Shwartz-Ziv et al. 2023 (info-theoretic VICReg)**: This paper derives VICReg’s covariance regularization directly from LDM and proposes $\log|\Sigma_{(z,z')}|$ as a tighter joint covariance alternative.
- **vs Halvagal et al. 2023 / Tian et al. 2021 (BYOL dynamics)**: They analyzed why stopgrad and EMA targets prevent collapse; this paper re-interprets stopgrad as a "conditional entropy plugin," a perspective that is conceptually more unified and aligns with identifiability proofs.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ A single objective unifies ICA, contrastive, non-contrastive, predictive, and stopgrad families with identifiability proofs.
- Experimental Thoroughness: ⭐⭐⭐ Systematic comparison of 8 knob combinations across multiple datasets, though lacks large-scale ImageNet-1K or long-sequence benchmark validation.
- Writing Quality: ⭐⭐⭐⭐ Clear derivations, highly concise Table 1, accessible even to non-theoretical readers.
- Value: ⭐⭐⭐⭐ Serves as both a unified theoretical framework and a provider of the new Kalman-based predictive SSL algorithm, offering a long-term toolkit for designing and explaining SSL.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Understanding the Learning Phases in Self-Supervised Learning via Critical Periods](../../ICLR2026/self_supervised/understanding_the_learning_phases_in_self-supervised_learning_via_critical_perio.md)
- [\[NeurIPS 2025\] Understanding Ice Crystal Habit Diversity with Self-Supervised Learning](../../NeurIPS2025/self_supervised/understanding_ice_crystal_habit_diversity_with_self-supervised_learning.md)
- [\[ICLR 2026\] Understanding the Robustness of Distributed Self-Supervised Learning Frameworks Against Non-IID Data](../../ICLR2026/self_supervised/understanding_the_robustness_of_distributed_self-supervised_learning_frameworks_.md)
- [\[ICML 2026\] Beyond Distribution Estimation: Simplex Anchored Structural Inference Towards Universal Semi-Supervised Learning](beyond_distribution_estimation_simplex_anchored_structural_inference_towards_uni.md)
- [\[ICML 2026\] Can Local Learning Match Self-Supervised Backpropagation?](can_local_learning_match_self-supervised_backpropagation.md)

</div>

<!-- RELATED:END -->

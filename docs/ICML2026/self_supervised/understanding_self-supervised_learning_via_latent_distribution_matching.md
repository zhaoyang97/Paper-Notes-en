---
title: >-
  [Paper Note] Understanding Self-Supervised Learning via Latent Distribution Matching
description: >-
  [ICML 2026][Self-Supervised Learning][latent distribution matching] The authors unify contrastive, non-contrastive, and predictive SSL as "Latent Distribution Matching (LDM)": maximizing the log-likelihood of samples und…
tags:
  - "ICML 2026"
  - "Self-Supervised Learning"
  - "latent distribution matching"
  - "nonlinear ICA"
  - "identifiability"
  - "Kalman prediction"
date: 2026-05-08
content_hash: 531dbdb1c04a518c
---

# Understanding Self-Supervised Learning via Latent Distribution Matching

**Conference**: ICML 2026  
**arXiv**: [2605.03517](https://arxiv.org/abs/2605.03517)  
**Code**: None  
**Area**: Self-supervised Representation Learning / ICA & Identifiability / Representation Learning Theory  
**Keywords**: Self-supervised learning, latent distribution matching, nonlinear ICA, identifiability, Kalman prediction

## TL;DR
The authors unify contrastive, non-contrastive, and predictive SSL as "Latent Distribution Matching (LDM)": maximizing the log-likelihood of samples under an assumed latent model (alignment) + maximizing latent entropy (uniformity), and based on this, derive a nonlinear identifiable predictive SSL with a Kalman predictor.

## Background & Motivation
**Background**: SSL has become mainstream in visual, language, and audio representation learning, with a wide variety of methods—SimCLR, VICReg, BYOL, SimSiam, CPC, JEPA, etc.—each with its own loss formulation and interpretation.

**Limitations of Prior Work**: (1) The geometric alignment perspective (Wang & Isola 2020) is intuitive but lacks a strict statistical foundation and cannot explain methods like BYOL/SimSiam that lack explicit repulsion; (2) The mutual information (MI) maximization view is invariant to any invertible transformation ($I[x,y]=I[\phi(x),\psi(y)]$), making it neither necessary nor sufficient; (3) Predictive SSL (CPC, JEPA, I-JEPA) is empirically SOTA, but its objectives and regularizations are heuristic combinations, lacking derivable design principles and identifiability guarantees.

**Key Challenge**: Existing methods each have strengths, but lack a unifying objective that explains why SSL yields useful representations and provides identifiability proofs.

**Goal**: (1) Find a unified objective encompassing ICA, contrastive/non-contrastive/predictive/stopgrad SSL; (2) Clarify the true role of MI maximization; (3) Derive new SSL variants (e.g., Kalman-based predictive SSL); (4) Provide identifiability guarantees for predictive SSL.

**Key Insight**: Return to the likelihood perspective—for invertible encoders, performing MLE in latent space is equivalent to matching the data distribution to the model distribution; extending to paired views leads to joint LDM.

**Core Idea**: Express SSL as $\mathcal F_{\mathrm{LDM}}=-D_{\mathrm{KL}}[R(z,z')\,\|\,P_\theta(z,z')] = \underbrace{\langle\log P_\theta(z,z')\rangle_R}_{\text{alignment}} + \underbrace{H_R[z,z']}_{\text{uniformity}}$; different SSL algorithms correspond to different choices of $P_\theta$ and entropy estimators.

## Method

### Overall Architecture
The authors start from maximum likelihood: for an invertible encoder $f$, $\langle\log P_\theta(x)\rangle_{P_{\mathrm{data}}}\propto\langle\log P_\theta(f(x))\rangle+H_{P_{\mathrm{data}}}[f(x)]=-D_{\mathrm{KL}}[P_{\mathrm{data}}(f(x))\|P_\theta(f(x))]$; linear ICA is a special case. Extending views to paired data $(x,x')$, the latent $R(z,z')$ is matched to the model $P_\theta(z,z')$, yielding the LDM objective. LDM is then compared with Aitchison & Ganev's MI variant $\mathcal F_{\mathrm{MI}}=\langle\log P_\theta\rangle_R+2H_R[z]$, showing that for nearly invertible encoders, MI is implicitly saturated by entropy regularization. Finally, by varying $P_\theta$ and entropy estimators, VICReg, SimCLR, CPC, BYOL/SimSiam, JEPA, and the new Kalman-predictive SSL are all unified in a single table (Table 1).

### Key Designs

1. **LDM Unified Objective + Entropy Estimator Categorization**:

    - **Function**: Provides a unifying objective for SSL and explains why different loss forms yield similar outcomes.
    - **Mechanism**: Uses $\mathcal F_{\mathrm{LDM}}=-D_{\mathrm{KL}}[R(z,z')\|P_\theta(z,z')]$ as the foundation, with the alignment term from $\log P_\theta$ and the uniformity term from $H_R$; entropy estimators are categorized into three types: KDE → contrastive SSL (SimCLR), parametric (Gaussian) → non-contrastive SSL (VICReg's $\log|\Sigma_z|$), conditional entropy plugin → stopgrad/predictor family (BYOL, JEPA).
    - **Design Motivation**: Previously, each SSL method told its own story; LDM exposes "distribution shape + entropy estimator" as two dials, immediately clarifying why VICReg's covariance regularization can be written as a Taylor expansion of $\log|\Sigma_z|$, and why SimCLR's negatives correspond to KDE bandwidth $1/\beta$.

2. **Clarifying the True Role of MI Maximization**:

    - **Function**: Explains why MI maximization is both popular and seemingly dispensable in SSL.
    - **Mechanism**: $\mathcal F_{\mathrm{MI}}-\mathcal F_{\mathrm{LDM}}=I_R[z,z']$, but for invertible encoders, $I_R[z,z']$ is automatically saturated, so the MI term contributes little in practice; the paper conducts controlled experiments with 8 combinations (latent space × entropy estimator × with/without MI), finding that "with or without MI" does not affect linear probing accuracy or representation dimensionality (Table 2, Fig. 3); the decisive factors are the latent space assumption and entropy estimator choice.
    - **Design Motivation**: Provides falsifiable experimental evidence for the long-ambiguous "MI maximization" slogan, and suggests future work need not overcomplicate objectives by deriving MI bounds.

3. **Predictive SSL: Kalman-based Latent Dynamics + Identifiability Proof**:

    - **Function**: Constructs a new, sampling-free, identifiable predictive SSL, providing a theoretical backbone for JEPA-like methods.
    - **Mechanism**: Models latent transitions as $P_\theta(z'|z)$, chooses Kalman-style linear Gaussian transitions with nonlinear encoders (manifold normalizing flow/injective flow), and applies $\mathcal F_{\mathrm{LDM}}$ to $(z,z')$; theoretically proves that under mild assumptions, predictive LDM can recover latent variables up to an affine equivalence class (identifiability up to affine), even with nonlinear predictors.
    - **Design Motivation**: JEPA is already SOTA in video/robotics, but its effectiveness was not well understood; LDM provides a unified answer to "why stable + why no collapse + why true factors can be recovered," and proposes a sampling-free Bayesian filtering variant as a directly applicable new algorithm.

### Loss & Training
The specific loss depends on the choice of $P_\theta$ and entropy estimator: VICReg corresponds to $-\frac{1}{2\sigma^2}\langle\|f(x)-f(x')\|^2\rangle+\log|\Sigma_z|$; the LDM variant uses $\log|\Sigma_{(z,z')}|$; SimCLR corresponds to $\langle\beta f(x)^\top f(x')\rangle-2\langle\log\langle\exp\{\beta f(x)^\top f(x^-)\}\rangle\rangle$ (KDE entropy estimation + spherical vMF); predictive SSL uses Kalman gain instead of momentum target, combined with stopgrad to implement the conditional entropy plugin.

## Key Experimental Results

### Main Results

| Dataset / Setting | Knob Combination | Top-1 acc | Notes |
|-------------------|------------------|-----------|-------|
| ImageNet-100, Plane × LogDet × LDM | VICReg-LDM | 75.9 | LDM version slightly outperforms MI version (74.7) |
| CIFAR-100, Plane × LogDet × LDM | Same as above | 69.5 | Significantly better than original VICReg-MI (65.3) |
| ImageNet-100, Sphere × Contr. × MI | SimCLR | 73.1 | Classic SimCLR baseline |
| CIFAR-10 | Plane × kNN × LDM | 92.1 | kNN entropy estimation is a practical LDM alternative |

### Ablation Study

| Knob | Key Observation | Interpretation |
|------|----------------|---------------|
| With vs. without MI ($\mathcal F_{\mathrm{MI}}$ vs. $\mathcal F_{\mathrm{LDM}}$) | Accuracy difference within ±0.4 across datasets | MI term is implicitly absorbed by entropy regularization, can be omitted |
| Latent space (Plane vs. Sphere) | Plane + LogDet significantly higher on CIFAR-100 / ImageNet-100 | The "shape" assumption of $P_\theta(z)$ has the greatest impact |
| Entropy estimator | LogDet > kNN ≈ KDE > parametric Gaussian (sphere) | Different assumptions determine collapse risk |
| Predictive LDM with Kalman | Outperforms BYOL/JEPA-style baselines on sequential tasks | Explicitly modeling transition noise is more stable |

### Key Findings
- LDM and MI versions are nearly equivalent: further demonstrates that the core determinants of SSL quality are $(P_\theta, H \text{ estimator})$, not MI maximization; this shifts engineering focus from "choosing MI estimators" back to "choosing latent models."
- The Kalman variant of predictive LDM provides the trifecta of "no collapse + identifiability + sampling-free," making it one of the few predictive SSLs with both theoretical and practical benefits.
- Table 1's interpretation of BYOL/SimSiam as conditional entropy plugins is a key insight: the long-misunderstood stopgrad design naturally fits within the LDM framework.

## Highlights & Insights
- Exceptional unifying power: a single table categorizes all five SSL families + ICA, with each method's key design corresponding to a specific LDM knob, directly guiding future algorithm design (e.g., changing $P_\theta$ shape or entropy estimator).
- Interpreting BYOL/JEPA's stopgrad as a conditional entropy plugin is a true "aha" moment, revealing that stopgrad is more than an engineering hack.
- Provides rigorous identifiability results, especially valuable for theoretically inclined SSL researchers—it offers a first-principles explanation for "why predictive SSL works."
- Kalman-based latent dynamics is a directly applicable new baseline, reusable for sequential/robotics/world-model research.

## Limitations & Future Work
- Experiments focus mainly on image SSL and simple sequential tasks, lacking coverage of large-scale video/multimodal pretraining; the framework's generality remains to be validated.
- LDM still requires the encoder to be "almost invertible on the data manifold," which may not hold for very noisy real-world data.
- Identifiability results are up to affine equivalence; downstream tasks may still require disentanglement post-processing.
- No in-depth analysis of the training dynamics of EMA targets or predictor networks.
- While entropy estimator choice is identified as a key factor, no systematic guidelines are provided for selecting them on new tasks—still requires empirical tuning.
- Algorithmic details of Kalman-based predictive SSL are brief in the main text; engineering details (e.g., prior covariance initialization) require consulting the appendix.

## Related Work & Insights
- **vs Wang & Isola 2020 (alignment-uniformity)**: They proposed a geometric alignment intuition; this work formalizes it as distribution matching and explains why BYOL works without explicit uniformity—conditional entropy plugin provides it implicitly.
- **vs Zimmermann et al. 2021 (CPC identifiability)**: They proved CPC is identifiable; this work embeds their result in the more general LDM framework, showing predictive SSL remains identifiable under nonlinear predictors.
- **vs Aitchison & Ganev 2024 (variational SSL)**: They used a variational perspective for $\mathcal F_{\mathrm{MI}}$; this work shows the MI term is nearly redundant, with distribution matching as the core.
- **vs Shwartz-Ziv et al. 2023 (info-theoretic VICReg)**: This work directly derives VICReg's covariance regularization from LDM and proposes $\log|\Sigma_{(z,z')}|$ joint covariance as a tighter alternative.
- **vs Halvagal et al. 2023 / Tian et al. 2021 (BYOL dynamics)**: They analyzed why stopgrad and EMA targets prevent collapse; this work reinterprets stopgrad as a "conditional entropy plugin," offering a more unified conceptual view aligned with identifiability proofs.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ A single objective unifies ICA / contrastive / non-contrastive / predictive / stopgrad, with identifiability proofs.
- Experimental Thoroughness: ⭐⭐⭐ Systematic comparison of 8 knob combinations across multiple datasets, but lacks large-scale ImageNet-1K or long-sequence benchmarks.
- Writing Quality: ⭐⭐⭐⭐ Clear derivations, highly condensed Table 1, accessible to non-theoretical readers.
- Value: ⭐⭐⭐⭐ Both a unifying theoretical framework and a new Kalman-based predictive SSL algorithm, providing a long-term toolbox for SSL design and interpretation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Beyond Distribution Estimation: Simplex Anchored Structural Inference Towards Universal Semi-Supervised Learning](beyond_distribution_estimation_simplex_anchored_structural_inference_towards_uni.md)
- [\[NeurIPS 2025\] Understanding Ice Crystal Habit Diversity with Self-Supervised Learning](../../NeurIPS2025/self_supervised/understanding_ice_crystal_habit_diversity_with_self-supervised_learning.md)
- [\[ICLR 2026\] Soft Equivariance Regularization for Invariant Self-Supervised Learning](../../ICLR2026/self_supervised/soft_equivariance_regularization_for_invariant_self-supervised_learning.md)
- [\[ICLR 2026\] InfoNCE Induces Gaussian Distribution](../../ICLR2026/self_supervised/infonce_induces_gaussian_distribution.md)
- [\[ICML 2026\] A Refined Generalization Analysis for Extreme Multi-class Supervised Contrastive Representation Learning](a_refined_generalization_analysis_for_extreme_multi-class_supervised_contrastive.md)

</div>

<!-- RELATED:END -->

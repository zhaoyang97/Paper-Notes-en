---
title: >-
  [Paper Note] Understanding Self-Supervised Learning via Latent Distribution Matching
description: >-
  [ICML 2026][Self-Supervised Learning][latent distribution matching] The authors unify contrastive / non-contrastive / predictive SSL as "Latent Distribution Matching (LDM)": maximizing the log-probability of samples unde…
tags:
  - "ICML 2026"
  - "Self-Supervised Learning"
  - "latent distribution matching"
  - "nonlinear ICA"
  - "identifiability"
  - "Kalman prediction"
date: 2026-05-08
content_hash: e2928dbc73d4161b
---

# Understanding Self-Supervised Learning via Latent Distribution Matching

**Conference**: ICML 2026  
**arXiv**: [2605.03517](https://arxiv.org/abs/2605.03517)  
**Code**: None  
**Area**: Self-Supervised Representation Learning / ICA and Identifiability / Representation Learning Theory  
**Keywords**: Self-supervised learning, latent distribution matching, nonlinear ICA, identifiability, Kalman prediction

## TL;DR
The authors unify contrastive / non-contrastive / predictive SSL as "Latent Distribution Matching (LDM)": maximizing the log-probability of samples under a hypothesized latent model (alignment) + maximizing latent entropy (uniformity). Based on this, they derive nonlinear identifiable predictive SSL equipped with a Kalman predictor.

## Background & Motivation
**Background**: SSL has become the mainstream for vision, language, and audio representation learning, featuring a diverse spectrum of methods—SimCLR, VICReg, BYOL, SimSiam, CPC, JEPA, etc.—each with its own loss formulation and interpretation.

**Limitations of Prior Work**: (1) The geometric alignment perspective (Wang & Isola 2020) provides intuitive explanations but lacks a strict statistical foundation and cannot explain methods without explicit repulsion, like BYOL/SimSiam. (2) Mutual Information (MI) maximization is neither necessary nor sufficient because MI is invariant to any invertible transformation ($I[x,y]=I[\phi(x),\psi(y)]$). (3) Predictive SSL (CPC, JEPA, I-JEPA) achieves empirical SOTA, but their objective functions and regularizations are heuristically combined, lacking derivable design principles and identifiability guarantees.

**Key Challenge**: Existing methods have individual strengths but lack a unifying objective that simultaneously explains why SSL produces useful representations and provides proofs of identifiability.

**Goal**: (1) Discover a unified objective covering ICA, contrastive, non-contrastive, predictive, and stopgrad-based SSL; (2) Clarify the actual role of MI maximization; (3) Derive new SSL variants (e.g., Kalman-based predictive SSL); (4) Provide identifiability guarantees for predictive SSL.

**Key Insight**: Return to the likelihood perspective—for an invertible encoder, performing MLE in the latent space is equivalent to matching the data distribution to the model distribution; extending this to paired views results in joint LDM.

**Core Idea**: Unify SSL as $\mathcal F_{\mathrm{LDM}}=-D_{\mathrm{KL}}[R(z,z')\,\|\,P_\theta(z,z')]=\underbrace{\langle\log P_\theta(z,z')\rangle_R}_{\text{alignment}}+\underbrace{H_R[z,z']}_{\text{uniformity}}$. Different SSL algorithms correspond to different choices of $P_\theta$ and entropy estimators.

## Method

### Overall Architecture
The authors start from maximum likelihood: for an invertible encoder $f$, $\langle\log P_\theta(x)\rangle_{P_{\mathrm{data}}}\propto\langle\log P_\theta(f(x))\rangle+H_{P_{\mathrm{data}}}[f(x)]=-D_{\mathrm{KL}}[P_{\mathrm{data}}(f(x))\|P_\theta(f(x))]$; linear ICA is a special case. Expanding views to paired data $(x,x')$ and matching the latent record $R(z,z')$ with the model $P_\theta(z,z')$ yields the LDM objective. By aligning LDM with the MI variant from Aitchison & Ganev, $\mathcal F_{\mathrm{MI}}=\langle\log P_\theta\rangle_R+2H_R[z]$, they prove that MI is implicitly saturated by entropy regularization when the encoder is nearly invertible. Finally, based on the choice of $P_\theta$ and entropy estimators, VICReg, SimCLR, CPC, BYOL/SimSiam, JEPA, and the new Kalman-predictive SSL are all incorporated into a unified framework (Table 1).

### Key Designs

1. **LDM Unified Objective + Entropy Estimator Classification**:
    - **Function**: Provide a unifying objective for SSL and explain why different loss formulations lead to similar conclusions.
    - **Mechanism**: Use $\mathcal F_{\mathrm{LDM}}=-D_{\mathrm{KL}}[R(z,z')\|P_\theta(z,z')]$ as the base, where the alignment term comes from $\log P_\theta$ and the uniformity term from $H_R$. Entropy estimators are categorized into three types: KDE → Contrastive SSL (SimCLR), Parametric (Gaussian) → Non-contrastive SSL ($\log|\Sigma_z|$ in VICReg), and Conditional entropy plugin → stopgrad/predictor systems (BYOL, JEPA).
    - **Design Motivation**: Previously, each SSL method had its own narrative. LDM extracts "distribution shape" and "entropy estimator" as two knobs, immediately revealing why VICReg's covariance regularization can be written as a Taylor expansion of $\log|\Sigma_z|$, and why SimCLR negative samples correspond to a KDE bandwidth of $1/\beta$.

2. **Clarifying the actual role of MI maximization**:
    - **Function**: Explain why MI maximization is popular yet seemingly optional in SSL.
    - **Mechanism**: $\mathcal F_{\mathrm{MI}}-\mathcal F_{\mathrm{LDM}}=I_R[z,z']$, but for invertible encoders, $I_R[z,z']$ is automatically saturated, making the actual contribution of the MI term negligible. The paper performs controlled experiments using 8 combinations (latent space × entropy estimation × presence of MI), finding that the inclusion of MI does not affect linear probing accuracy or representation dimensionality (Table 2, Fig. 3). The decisive factors are the latent space assumptions and entropy estimator choices.
    - **Design Motivation**: To provide falsifiable experimental conclusions for the historically vague slogan of "MI maximization" and suggest that future work need not over-complicate objectives to derive MI bounds.

3. **Predictive SSL: Kalman-based latent dynamics + Identifiability Proof**:
    - **Function**: Construct a new, sampling-free, and identifiable predictive SSL, providing a theoretical backbone for JEPA-style methods.
    - **Mechanism**: Latent transitions are modeled as $P_\theta(z'|z)$ using Kalman-style linear Gaussian transitions with a nonlinear encoder (manifold normalizing flow / injective flow). Applying $\mathcal F_{\mathrm{LDM}}$ to $(z,z')$, it is theoretically proven that under mild assumptions, predictive LDM can recover latent variables up to an affine transformation (identifiability up to affine), even if the predictor is nonlinear.
    - **Design Motivation**: JEPA is SOTA in video/robotics, but its effectiveness was poorly understood. LDM provides a unified answer to "why it is stable," "why it avoids collapse," and "how it recovers true factors," while introducing a sampling-free Bayesian filtering version as a practical new algorithm.

### Loss & Training
Specific losses vary by $P_\theta$ and entropy estimator: VICReg corresponds to $-\frac{1}{2\sigma^2}\langle\|f(x)-f(x')\|^2\rangle+\log|\Sigma_z|$; the LDM version uses $\log|\Sigma_{(z,z')}|$; SimCLR corresponds to $\langle\beta f(x)^\top f(x')\rangle-2\langle\log\langle\exp\{\beta f(x)^\top f(x^-)\}\rangle\rangle$ (KDE entropy estimation + spherical vMF); Predictive SSL uses Kalman gain to replace the momentum target, combined with stopgrad to implement the conditional entropy plugin.

## Key Experimental Results

### Main Results

| Dataset / Setting | Knob Combination | Top-1 acc | Note |
|---------------|----------|-----------|------|
| ImageNet-100, Plane × LogDet × LDM | VICReg-LDM | 75.9 | LDM version slightly outperforms MI version (74.7) |
| CIFAR-100, Plane × LogDet × LDM | Same as above | 69.5 | Significant margin over original VICReg-MI 65.3 |
| ImageNet-100, Sphere × Contr. × MI | SimCLR | 73.1 | Classic SimCLR baseline |
| CIFAR-10 | Plane × kNN × LDM | 92.1 | kNN entropy estimation is a practical alternative for LDM |

### Ablation Study

| Knob | Key Finding | Interpretation |
|------|----------|------|
| With vs. Without MI ($\mathcal F_{\mathrm{MI}}$ vs. $\mathcal F_{\mathrm{LDM}}$) | Accuracy difference $\leq \pm 0.4$ across datasets | MI term is implicitly absorbed by entropy regularization and can be omitted |
| Latent Space (Plane vs. Sphere) | Plane + LogDet is significantly higher on CIFAR-100 / ImageNet-100 | The "shape" assumption of $P_\theta(z)$ has the greatest impact |
| Entropy Estimator | LogDet > kNN ≈ KDE > parametric Gaussian (Sphere) | Different assumptions determine collapse risk |
| Predictive LDM with Kalman | Gains over BYOL/JEPA style baselines in temporal tasks | Explicitly modeling transition noise is more stable |

### Key Findings
- **LDM and MI versions are nearly equivalent**: This further demonstrates that the core of SSL quality lies in $(P_\theta, \text{H estimator})$ rather than whether mutual information is maximized. This discovery shifts engineering focus from "choosing MI estimators" back to "choosing latent models."
- **Kalman variant of predictive LDM**: Provides a "no collapse + identifiable + sampling-free" trifecta, making it one of the few predictive SSL methods to yield both theoretical and engineering benefits.
- **BYOL/SimSiam as conditional entropy plugin**: Interpreting stopgrad as such in Table 1 is a key insight. The stopgrad design, long considered hard to explain, naturally falls within the LDM framework.

## Highlights & Insights
- **Exceptional Unifying Power**: A single table categorizes the five major families of SSL + ICA, mapping the key designs of each method to specific knobs in the LDM framework, directly guiding the design of future algorithms.
- **Stopgrad Interpretation**: Explaining the stopgrad in BYOL/JEPA as a "conditional entropy plugin" is a true "Aha!" moment, revealing that stopgrad is more than just an engineering hack.
- **Rigorous Identifiability**: Providing strict identifiability results is crucial for theory-oriented SSL researchers—it offers a first-principles explanation for why predictive SSL works.
- **Kalman-based Latent Dynamics**: A practical new baseline that can be reused in temporal, robotics, and world-model research.

## Limitations & Future Work
- Experiments primarily focus on image SSL and simple temporal tasks, without covering large-scale video or multimodal pre-training; the universality of the framework requires further verification.
- LDM still requires the encoder to be "nearly invertible on the data manifold," which may not hold for very noisy real-world data.
- Identifiability results are up to an affine equivalence class; downstream tasks may still require disentanglement post-processing.
- Lacks an in-depth analysis of the training dynamics of EMA targets and predictor networks.
- While the choice of entropy estimator is identified as a decisive factor, specific criteria for systematic selection in new tasks are not provided, still requiring empirical tuning.
- Algorithmic details for Kalman-based predictive SSL are somewhat brief in the main text; engineering implementation details (e.g., prior covariance initialization) require consulting the appendix.

## Related Work & Insights
- **vs. Wang & Isola 2020 (alignment-uniformity)**: They proposed an intuitive version of geometric alignment; Ours formalizes it as distribution matching and explains why BYOL works without explicit uniformity—it is implicitly provided by the conditional entropy plugin.
- **vs. Zimmermann et al. 2021 (CPC identifiability)**: They proved CPC is identifiable; Ours embeds their results into the more general LDM framework, proving that predictive SSL remains identifiable even with nonlinear predictors.
- **vs. Aitchison & Ganev 2024 (variational SSL)**: They used a variational perspective for $\mathcal F_{\mathrm{MI}}$; Ours proves the MI term is nearly redundant and that distribution matching is the core.
- **vs. Shwartz-Ziv et al. 2023 (info-theoretic VICReg)**: Ours directly derives VICReg’s covariance regularization using LDM and proposes $\log|\Sigma_{(z,z')}|$ as a tighter joint covariance alternative.
- **vs. Halvagal et al. 2023 / Tian et al. 2021 (BYOL dynamics)**: They analyzed why stopgrad and EMA targets do not collapse; Ours reinterprets stopgrad as a "conditional entropy plugin," a perspective that is conceptually more unified and aligns with identifiability proofs.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ A single objective unifies ICA, contrastive, non-contrastive, predictive, and stopgrad categories with identifiability proofs.
- **Experimental Thoroughness**: ⭐⭐⭐ Systematic comparison of 8 knob combinations across multiple datasets, though lacks large-scale ImageNet-1K or long-duration temporal benchmarks.
- **Writing Quality**: ⭐⭐⭐⭐ Clear derivations, highly condensed Table 1, accessible even to non-theoretical readers.
- **Value**: ⭐⭐⭐⭐ Serves as both a unified theoretical framework and a new algorithm (Kalman-based predictive SSL), providing a long-term toolkit for designing and explaining SSL.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Beyond Distribution Estimation: Simplex Anchored Structural Inference Towards Universal Semi-Supervised Learning](beyond_distribution_estimation_simplex_anchored_structural_inference_towards_uni.md)
- [\[ICML 2026\] Can Local Learning Match Self-Supervised Backpropagation?](can_local_learning_match_self-supervised_backpropagation.md)
- [\[NeurIPS 2025\] Understanding Ice Crystal Habit Diversity with Self-Supervised Learning](../../NeurIPS2025/self_supervised/understanding_ice_crystal_habit_diversity_with_self-supervised_learning.md)
- [\[ICML 2026\] NumLeak: Public Numeric Benchmarks as Latent Labels in Foundation Models](numleak_public_numeric_benchmarks_as_latent_labels_in_foundation_models.md)
- [\[ICML 2026\] FLAG: Foundation Model Representation with Latent Diffusion Alignment via Graph for Spatial Gene Expression Prediction](flag_foundation_model_representation_with_latent_diffusion_alignment_via_graph_f.md)

</div>

<!-- RELATED:END -->

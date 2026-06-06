---
title: >-
  [Paper Note] R2R2: Robust Representation for Intensive Experience Reuse via Redundancy Reduction in Self-Predictive Learning
description: >-
  [ICML 2026][Reinforcement Learning][Self-Predictive Learning] R2R2 incorporates VICReg-style redundancy reduction constraints into self-predictive learning (SPL) to stabilize high UTD training…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "Self-Predictive Learning"
  - "Redundancy Reduction"
  - "VICReg"
  - "High UTD"
  - "Representation Collapse"
date: 2026-05-08
content_hash: 9453d8175af0b72b
---

# R2R2: Robust Representation for Intensive Experience Reuse via Redundancy Reduction in Self-Predictive Learning

**Conference**: ICML 2026  
**arXiv**: [2605.14026](https://arxiv.org/abs/2605.14026)  
**Code**: Available (github.com/songsang7/R2R2)  
**Area**: Reinforcement Learning / Self-Predictive Representation Learning / High UTD Training  
**Keywords**: Self-Predictive Learning, Redundancy Reduction, VICReg, High UTD, Representation Collapse

## TL;DR
R2R2 incorporates VICReg-style redundancy reduction constraints into self-predictive learning (SPL) to stabilize high UTD training, with the **key modification being the omission of zero-centering**—theoretically proving that zero-centering removes the constant eigenmode (i.e., global dynamics information) in the spectral decomposition of SPL. Experiments show that on TD7 with UTD=20, the score increases from 1.02 to 1.24 (+22%), and the newly proposed SimbaV2-SPL architecture achieves new SOTA in continuous control.

## Background & Motivation

**Background**: The pursuit of sample efficiency in reinforcement learning has led to two main directions—off-policy algorithms reuse the replay buffer, while model-based/SPL methods extract additional signals from dynamics via auxiliary tasks (predicting the next latent state). Increasing the Update-to-Data (UTD) ratio is another orthogonal approach, but higher UTD (e.g., 20) almost inevitably leads to overfitting. Current high UTD works (REDQ, CrossQ, SimbaV2, BRO) focus primarily on the **value function side**—using ensembles, BatchNorm, LayerNorm, etc., to stabilize critic bias.

**Limitations of Prior Work**: These value-centric methods do not address **instability in the representation layer**. When UTD is increased, the SPL encoder and latent dynamics predictor also overfit: representation collapse (subspace collapse), and effective rank keeps dropping. Existing SSL redundancy reduction methods (Barlow Twins, VICReg) are designed for visual representations and **by default perform zero-centering** (covariance matrix is computed after mean subtraction); directly applying them to SPL actually degrades performance.

**Key Challenge**: Theoretical analysis of SPL (Tang et al., 2023) shows that minimizing the SPL loss is equivalent to making the representation matrix $\Phi$ span the top-$k$ right eigenvector subspace of the transition matrix $P^\pi$. Markov chains always have eigenvalue 1, with the corresponding eigenvector being the constant vector $\mathbf 1$ ($P\mathbf 1=\mathbf 1$), which carries "global dynamics/probability conservation" information. **The zero-centering operator $H=I_N-\frac{1}{N}\mathbf 1\mathbf 1^\top$ annihilates any constant vector**—meaning that the seemingly harmless "mean subtraction" in SSL precisely removes this dominant eigenmode, directly conflicting with SPL's objective.

**Goal**: (i) Add a representation layer regularization for high UTD training; (ii) Ensure this regularization is compatible with the spectral properties of SPL; (iii) Make the design algorithm/architecture-agnostic and plug-and-play.

**Key Insight**: The authors approach from the mathematical detail of the "constant eigenmode" in SPL spectral decomposition—a perspective not previously explored in SSL—and find that zero-centering is a structural issue, not a simple "hyperparameter tuning" matter.

**Core Idea**: Use **non-centered covariance (directly using the inner product matrix, without mean subtraction)** for redundancy reduction regularization, remove the extra projector, and directly apply the mechanism to the SPL encoder output, unifying "redundancy reduction" and "SPL spectral preservation".

## Method

### Overall Architecture
R2R2 adds two regularization terms to the encoder output $z_t=\phi(s_t)$ in the standard SPL training loop: non-centered redundancy reduction loss $\mathcal L_{\text{RR}}$ and variance loss $\mathcal L_{\text{Var}}$. The main SPL loss $\mathcal L_{\text{SPL}}=\mathbb E[\|\mathcal T(\phi(s),a)-\text{sg}(\phi(s'))\|_2^2]$ remains unchanged. After each environment step, $G$ high UTD updates are performed: encode states, compute $\mathcal L_{\text{SPL}}+\lambda_{\text{RR}}\mathcal L_{\text{RR}}+\lambda_{\text{Var}}\mathcal L_{\text{Var}}$ to update the encoder and predictor, then proceed with the base algorithm's (TD7, Minimalist $\phi$, SimbaV2-SPL, etc.) actor-critic update. The paper also constructs the SimbaV2-SPL architecture, integrating the SPL module (encoder + transition predictor) into SimbaV2, allowing R2R2 to be stacked with SOTA architectures.

### Key Designs

1. **Non-centered Redundancy Reduction Loss**:

    - **Function**: Decouples feature dimensions and prevents representation collapse without erasing the constant eigenmode.
    - **Mechanism**: Uses the non-centered correlation matrix $[C(Z)]_{ij}=\frac{1}{N-1}\sum_b z_{b,i}z_{b,j}$ instead of standard covariance (which subtracts the mean $\mu_k=\mathbb E_b[z_{b,k}]$). The loss is defined as $\mathcal L_{\text{RR}}=\frac{1}{d(d-1)}\sum_{i\neq j}[C(Z)]_{ij}^2$, pushing off-diagonal inner products toward zero, enforcing feature decorrelation without mean subtraction.
    - **Design Motivation**: Lemma 1 + Proposition 2 strictly prove that $H\mathbf c=\mathbf 0$ ($\mathbf c=c\mathbf 1$) means zero-centering **precisely** removes the projection of the representation matrix in the $\mathbf 1$ direction, which is exactly the constant eigenvector for eigenvalue 1 of $P^\pi$, carrying global dynamics. Neural network bias parameters could theoretically recover this signal, but require extra optimization detours; it is preferable to retain it directly in the loss.

2. **Direct Regularization (No Projector)**:

    - **Function**: Applies redundancy reduction directly to the encoder output $z_t$, not to the output of an extra projection head.
    - **Mechanism**: Standard VICReg/Barlow Twins first pass $z$ through a projector $g$ before computing redundancy loss; R2R2 computes $\mathcal L_{\text{RR}}$ and $\mathcal L_{\text{Var}}$ directly on $\phi(s)$.
    - **Design Motivation**: Theoretical analysis reveals that SPL spectral properties require constraints on the encoder output itself; a projector in between blurs the regularization's effect on the "used representation". Removing a module also reduces the overfitting surface under high UTD.

3. **SimbaV2-SPL Architecture (Orthogonal Contribution)**:

    - **Function**: Equips the pure model-free SOTA architecture SimbaV2 with an SPL module, enabling R2R2 to be attached for "architecture-orthogonal" validation.
    - **Mechanism**: Adds an encoder $\phi$ and transition predictor $\mathcal T$ to SimbaV2 under the SPL framework, linearly projects and L2 normalizes the raw state, concatenates it with the latent representation $z$, and feeds it to the actor/critic—thus incorporating latent dynamics while preserving SimbaV2's high-frequency details from the raw signal.
    - **Design Motivation**: Experiments on a single baseline (TD7) could be questioned as "algorithm-specific tricks"; by injecting SPL into the orthogonal SimbaV2 architecture and stacking R2R2, improvements are shown to be complementary to modern architectures, not dependent on a specific structure.

### Loss & Training
$\mathcal L_{\text{R2R2}}=\mathcal L_{\text{SPL}}+\lambda_{\text{RR}}\mathcal L_{\text{RR}}+\lambda_{\text{Var}}\mathcal L_{\text{Var}}$. All experiments fix $\lambda_{\text{RR}}=\lambda_{\text{Var}}=0.01$, variance threshold $v_{th}=1$, **no hyperparameter tuning across all tasks**. All other hyperparameters of the base algorithms remain unchanged.

## Key Experimental Results

### Main Results
11 continuous control environments (4 Gym MuJoCo + 7 DMC-Hard), normalized scores (relative to UTD=1 baseline); two settings: UTD=1 and UTD=20; 500k step budget.

| Algorithm | Environment | UTD=1 | UTD=20 |
|---|---|---|---|
| TD7 | Total | 1.00 | 1.02 |
| TD7 + R2R2 | Total | **1.06** | **1.24** (+22%) |
| TD7 | DMC-Hard | 1.00 | 1.02 |
| TD7 + R2R2 | DMC-Hard | 1.05 | **1.32** |
| Minimalist φ | Gym | 1.00 | 0.41 (collapse) |
| Minimalist φ + R2R2 | Gym | 1.00 | **0.57** |
| TD7+LN | Total | 1.00 | 0.88 (regression) |
| TD7+LN + R2R2 | Total | 1.08 | **1.10** |
| SimbaV2 | Total | 1.00 | 1.20 |
| SimbaV2 + SPL | Total | 1.16 | **1.34** (new SOTA) |
| SimbaV2 + SPL + R2R2 | Total | 1.15 | **1.38** |

### Ablation Study

| Configuration | Dog-Trot at UTD=20 | Conclusion |
|---|---|---|
| Full R2R2 (non-centered) | High | Complete method |
| R2R2 + zero-centering | Significant degradation | Verifies Proposition 2 |
| Remove $\mathcal L_{\text{RR}}$ | Worse degradation | RR term is main contributor |
| Remove $\mathcal L_{\text{Var}}$ | Moderate degradation | Var term prevents collapse |
| TD7 baseline (no R2R2) | Lowest | No protection |

### Key Findings
- **R2R2 complements LayerNorm**: TD7+LN under high UTD performs worse than baseline (0.88); adding R2R2 brings it back to 1.10, indicating that architectural normalization cannot solve representation collapse.
- **Singular value spectrum visualization (Humanoid-Stand)**: At UTD=1, R2R2 reduces effective rank from 76.5 to 65.0 (spectrum concentrates, retaining task-relevant principal components); at UTD=20, baseline shows sharp tail singular value collapse, while R2R2 maintains a heavy-tailed distribution, preventing subspace collapse.
- **Effective Rank evolution over time**: Under UTD=20, baseline's ER gradually declines, R2R2 maintains stable high ER; adding zero-centering causes collapse similar to baseline—direct experimental confirmation of theoretical analysis.

## Highlights & Insights
- **Spectral decomposition perspective on SSL and RL compatibility**: Previously, no one realized that VICReg's "mean subtraction" step precisely eliminates the Markov chain's constant eigenmode. This observation challenges the optimistic assumption that "generic SSL tricks can be directly applied to RL", reminding us that SSL's design premises (unordered data, mean-insensitivity) may not hold for RL latent dynamics with structured data.
- **Theory-driven minimal modification**: Simply removing "mean subtraction" from the covariance formula, combined with "removing the projector", yields stable improvements on SOTA architectures. The paper's structure—"theoretical analysis tells you **which line of code must not be changed**"—is exemplary.
- **Solid orthogonality argument**: Not only do all three SPL baselines of varying complexity (TD7, Minimalist φ, TD7+LN) improve, but the authors also build SimbaV2-SPL and show R2R2 still improves when attached to the strongest backbone—four sets of experiments ensure it's not a TD7-specific trick.
- **Zero hyperparameter tuning**: The same set of λ is used for all tasks, making practical deployment very friendly.

## Limitations & Future Work
- The theoretical analysis is based on the "SPL ≈ spectral decomposition" equivalence from Tang et al. 2023; the tightness for non-SimSiam-style frameworks (e.g., BYOL+predictor) is not fully discussed.
- The introduction of SimbaV2-SPL mixes the variables of "adding SPL" and "adding R2R2", partially diluting pure comparisons (the paper does report +SPL and +SPL+R2R2 separately).
- Experiments focus on continuous control (MuJoCo + DMC-Hard); discrete action (Atari), sparse reward, and pixel input scenarios are not validated.
- Training time overhead is about +12% wall-clock; further optimization may be needed for very large-scale settings.

## Related Work & Insights
- **vs VICReg / Barlow Twins**: Native SSL redundancy reduction, zero-centering structurally damages SPL; R2R2 fixes this with a non-centered form.
- **vs REDQ / CrossQ**: Value-centric high UTD methods address critic bias; R2R2 addresses the orthogonal issue of representation collapse and can be combined.
- **vs SimBa / SimbaV2 / BRO**: Rely on architectural normalization (LN) and dropout to stabilize high UTD; R2R2 shows that architecture alone is insufficient, and regularization is a necessary supplement.
- **vs SPR / TD7 (SPL series)**: Native SPL is unstable under high UTD; R2R2 injects redundancy reduction to directly stabilize the encoder.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The spectral analysis explaining "why SSL centering is incompatible with SPL" is highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 11 environments × 4 baselines × 2 UTD settings + ER/spectral analysis/wall-clock comprehensive coverage.
- Writing Quality: ⭐⭐⭐⭐⭐ Theory-lemma-proposition-experiment-ablation are tightly linked, with clean narrative.
- Value: ⭐⭐⭐⭐ Provides a new representation layer stabilization mechanism for high UTD RL, and proves it can be orthogonally combined with SOTA architectures.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] ExGRPO: Learning to Reason from Experience](../../ICLR2026/reinforcement_learning/exgrpo_learning_to_reason_from_experience.md)
- [\[ICLR 2026\] Self-Improving Skill Learning for Robust Skill-based Meta-Reinforcement Learning](../../ICLR2026/reinforcement_learning/self-improving_skill_learning_for_robust_skill-based_meta-reinforcement_learning.md)
- [\[ICLR 2026\] Model Predictive Adversarial Imitation Learning for Planning from Observation](../../ICLR2026/reinforcement_learning/model_predictive_adversarial_imitation_learning_for_planning_from_observation.md)
- [\[ICLR 2026\] Self-Harmony: Learning to Harmonize Self-Supervision and Self-Play in Test-Time Reinforcement Learning](../../ICLR2026/reinforcement_learning/self-harmony_learning_to_harmonize_self-supervision_and_self-play_in_test-time_r.md)
- [\[ICLR 2026\] Stackelberg Coupling of Online Representation Learning and Reinforcement Learning](../../ICLR2026/reinforcement_learning/stackelberg_coupling_of_online_representation_learning_and_reinforcement_learnin.md)

</div>

<!-- RELATED:END -->

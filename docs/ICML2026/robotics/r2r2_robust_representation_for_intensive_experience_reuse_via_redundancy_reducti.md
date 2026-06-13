---
title: >-
  [Paper Note] R2R2: Robust Representation for Intensive Experience Reuse via Redundancy Reduction in Self-Predictive Learning
description: >-
  [ICML 2026][Robotics][Self-Predictive Learning] R2R2 integrates VICReg-style redundancy reduction constraints into Self-Predictive Learning (SPL) to stabilize high Update-to-Data (UTD) training. The **key modification is…
tags:
  - "ICML 2026"
  - "Robotics"
  - "Self-Predictive Learning"
  - "Redundancy Reduction"
  - "VICReg"
  - "High UTD"
  - "Representation Collapse"
date: 2026-05-08
content_hash: 5b102f95aa629e09
---

# R2R2: Robust Representation for Intensive Experience Reuse via Redundancy Reduction in Self-Predictive Learning

**Conference**: ICML 2026  
**arXiv**: [2605.14026](https://arxiv.org/abs/2605.14026)  
**Code**: Available (github.com/songsang7/R2R2)  
**Area**: Reinforcement Learning / Self-Predictive Representation Learning / High UTD Training  
**Keywords**: Self-Predictive Learning, Redundancy Reduction, VICReg, High UTD, Representation Collapse

## TL;DR
R2R2 integrates VICReg-style redundancy reduction constraints into Self-Predictive Learning (SPL) to stabilize high Update-to-Data (UTD) training. The **key modification is the omission of zero-centering**—it is theoretically proven that zero-centering eliminates the constant eigenmode (i.e., global dynamics information) in SPL spectral decomposition. Experiments on TD7 with UTD=20 improved the score from 1.02 to 1.24 (+22%), and the newly proposed SimbaV2-SPL architecture refreshes the state-of-the-art (SOTA) in continuous control.

## Background & Motivation

**Background**: The pursuit of sample efficiency in reinforcement learning has led to two main directions: off-policy algorithms reusing the replay buffer, and model-based/SPL methods using auxiliary tasks (predicting the next latent state) to extract additional signals from dynamics. Increasing the UTD ratio is another orthogonal approach, but high UTD (e.g., 20) almost inevitably triggers overfitting. Current high UTD works (REDQ, CrossQ, SimbaV2, BRO) focus almost exclusively on the **value function side**—using ensembles, BatchNorm, LayerNorm, etc., to stabilize critic bias.

**Limitations of Prior Work**: These value-centric methods do not address **instability at the representation layer**. When UTD is high, the SPL encoder and latent dynamics predictor also overfit, leading to subspace collapse and a persistent decline in effective rank. Existing SSL redundancy reduction methods (Barlow Twins, VICReg) were natively designed for visual representations and **default to zero-centering** (covariance matrices calculated after subtracting the mean); applying these directly to SPL actually degrades performance.

**Key Challenge**: Theoretical analysis of SPL (Tang et al., 2023) indicates that minimizing the SPL loss is equivalent to making the representation matrix $\Phi$ span the top-$k$ right eigenvector subspace of the transition matrix $P^\pi$. A Markov chain always has an eigenvalue of 1, with the corresponding eigenvector being the constant vector $\mathbf 1$ ($P\mathbf 1=\mathbf 1$), which carries "global dynamics/probability conservation" information. The **zero-centering operator $H=I_N-\frac{1}{N}\mathbf 1\mathbf 1^\top$ is 0 for any constant vector**—meaning the seemingly harmless "mean subtraction" in SSL precisely erases this dominant eigenmode, directly conflicting with the SPL objective.

**Goal**: (i) Provide representation-layer regularization for high UTD training; (ii) Ensure compatibility between the regularization and SPL spectral properties; (iii) Keep the design agnostic to algorithms/architectures for plug-and-play capability.

**Key Insight**: The authors approach this from the mathematical detail of the "constant eigenmode" in SPL spectral decomposition—a perspective untouched by the SSL community—identifying that zero-centering is a structural issue rather than a simple hyperparameter tuning problem.

**Core Idea**: Utilize **non-centered covariance (using the inner product matrix directly without mean subtraction)** for redundancy reduction regularization. Simultaneously, the extra projector is removed, attaching the mechanism directly to the SPL encoder output to unify "redundancy reduction" with "SPL spectral preservation."

## Method

### Overall Architecture
R2R2 adds two regularization terms to the encoder output $z_t=\phi(s_t)$ within the standard SPL training loop: the non-centered redundancy reduction loss $\mathcal L_{\text{RR}}$ and the variance loss $\mathcal L_{\text{Var}}$. The main SPL loss $\mathcal L_{\text{SPL}}=\mathbb E[\|\mathcal T(\phi(s),a)-\text{sg}(\phi(s'))\|_2^2]$ remains unchanged. After each environment step, $G$ high-UTD updates are performed: state encoding, calculating $\mathcal L_{\text{SPL}}+\lambda_{\text{RR}}\mathcal L_{\text{RR}}+\lambda_{\text{Var}}\mathcal L_{\text{Var}}$ to update the encoder and predictor, followed by actor-critic updates of the base algorithm (TD7, Minimalist $\phi$, SimbaV2-SPL, etc.). The paper also constructs the SimbaV2-SPL architecture to integrate the SPL module (encoder + transition predictor) into SimbaV2, allowing R2R2 to benefit from SOTA architectures.

### Key Designs

1. **Non-centered Redundancy Reduction Loss**:
    - **Function**: Decouple feature dimensions and prevent representation collapse without erasing the constant eigenmode.
    - **Mechanism**: Replaces standard covariance with a non-centered correlation matrix $[C(Z)]_{ij}=\frac{1}{N-1}\sum_b z_{b,i}z_{b,j}$ (the former subtracts the mean $\mu_k=\mathbb E_b[z_{b,k}]$). The loss is defined as $\mathcal L_{\text{RR}}=\frac{1}{d(d-1)}\sum_{i\neq j}[C(Z)]_{ij}^2$, pushing off-diagonal inner products to 0, which forces different feature dimensions to be "unrelated but not zero-mean."
    - **Design Motivation**: Lemma 1 + Proposition 2 strictly prove that $H\mathbf c=\mathbf 0$ ($\mathbf c=c\mathbf 1$) implies zero-centering **precisely** eliminates the projection of the representation matrix in the $\mathbf 1$ direction. This direction corresponds to the constant eigenvector of eigenvalue 1 for $P^\pi$, carrying global dynamics. While neural network bias parameters could theoretically recover this signal, it requires inefficient optimization paths; it is better to preserve it directly in the loss.

2. **Direct Regularization (Removing Projector)**:
    - **Function**: Applies redundancy reduction directly to the encoder output $z_t$ rather than on the output of an additional projection head.
    - **Mechanism**: Standard VICReg/Barlow Twins pass $z$ through a projector $g$ before calculating redundancy loss; R2R2 computes $\mathcal L_{\text{RR}}$ and $\mathcal L_{\text{Var}}$ directly on $\phi(s)$.
    - **Design Motivation**: Theoretical analysis reveals that SPL spectral properties require specific constraints on the encoder output itself. A projector can blur the regularization's constraint on the actual "representation being used." Fewer modules also reduce the overfitting surface under high UTD.

3. **SimbaV2-SPL Architecture** (Orthogonal Contribution):
    - **Function**: Equips the model-free SOTA architecture SimbaV2 with an SPL module, allowing validation of R2R2's architectural orthogonality.
    - **Mechanism**: Adds an encoder $\phi$ and transition predictor $\mathcal T$ outside SimbaV2 to follow the SPL framework. The raw state is linearly projected, L2 normalized, and concatenated with the latent representation $z$ before being fed into the actor/critic. This integrates latent dynamics while preserving the high-frequency details of the raw signal in SimbaV2.
    - **Design Motivation**: Relying on a single baseline (TD7) risks being dismissed as a specific algorithm trick. By injecting SPL into an orthogonal architecture like SimbaV2 and observing gains with R2R2, it proves the improvement complements modern architectural advances.

### Loss & Training
$\mathcal L_{\text{R2R2}}=\mathcal L_{\text{SPL}}+\lambda_{\text{RR}}\mathcal L_{\text{RR}}+\lambda_{\text{Var}}\mathcal L_{\text{Var}}$. In all experiments, $\lambda_{\text{RR}}=\lambda_{\text{Var}}=0.01$ and the variance threshold $v_{th}=1$ are fixed, with **no per-task hyperparameter tuning**. All other hyperparameters of the base algorithms remain unchanged.

## Key Experimental Results

### Main Results
11 continuous control environments (4 Gym MuJoCo + 7 DMC-Hard), normalized scores (relative to UTD=1 baseline); two settings: UTD=1 and UTD=20; 500k step budget.

| Algorithm | Environment | UTD=1 | UTD=20 |
|---|---|---|---|
| TD7 | Total | 1.00 | 1.02 |
| TD7 + R2R2 | Total | **1.06** | **1.24** (+22%) |
| TD7 | DMC-Hard | 1.00 | 1.02 |
| TD7 + R2R2 | DMC-Hard | 1.05 | **1.32** |
| Minimalist $\phi$ | Gym | 1.00 | 0.41 (Collapsed) |
| Minimalist $\phi$ + R2R2 | Gym | 1.00 | **0.57** |
| TD7+LN | Total | 1.00 | 0.88 (Regression) |
| TD7+LN + R2R2 | Total | 1.08 | **1.10** |
| SimbaV2 | Total | 1.00 | 1.20 |
| SimbaV2 + SPL | Total | 1.16 | **1.34** (New SOTA) |
| SimbaV2 + SPL + R2R2 | Total | 1.15 | **1.38** |

### Ablation Study

| Configuration | Dog-Trot at UTD=20 | Conclusion |
|---|---|---|
| Full R2R2 (non-centered) | High | Full method performance |
| R2R2 + zero-centering | Significant degradation | Validates Proposition 2 |
| Without $\mathcal L_{\text{RR}}$ | Severe degradation | RR term is the main contributor |
| Without $\mathcal L_{\text{Var}}$ | Moderate degradation | Var term prevents collapse |
| TD7 baseline (no R2R2) | Lowest | No protection |

### Key Findings
- **R2R2 is complementary to LayerNorm**: TD7+LN performs worse than the baseline at high UTD (0.88), but adding R2R2 brings it back to 1.10, showing that architectural normalization cannot resolve representation collapse.
- **Singular Value Spectrum Visualization (Humanoid-Stand)**: At UTD=1, R2R2 compresses the effective rank from 76.5 to 65.0 (concentrating the spectrum and preserving task-relevant components). At UTD=20, the baseline shows a sharp collapse of tail singular values, while R2R2 maintains a heavy-tailed distribution, preventing subspace collapse.
- **Effective Rank Evolution**: Under UTD=20, the effective rank (ER) of the baseline decreases progressively, while R2R2 maintains a stable high ER. Adding zero-centering causes the ER to collapse alongside the baseline, directly validating the theoretical analysis.

## Highlights & Insights
- **SSL vs. RL compatibility via Spectral Decomposition**: Previously, the effect of the "mean subtraction" step in VICReg on Markov chain constant eigenmodes was overlooked. This observation challenges the optimistic assumption of directly porting SSL tricks to RL, reminding us that SSL design premises (unordered data, mean-insensitivity) may not hold for structured RL latent dynamics.
- **Theoretically guided minimal changes**: Simply removing "mean subtraction" from the covariance formula, combined with "removing the projector," leads to stable gains on SOTA architectures. This clean structure of "theory identifying which line of code not to modify" is exemplary.
- **Robust Orthogonality Argumentation**: Performance gains were observed across three SPL baselines of varying complexity (TD7, Minimalist $\phi$, TD7+LN), and further validated by integrating R2R2 into the SimbaV2-SPL backbone. This ensures the method is not a TD7-specific trick.
- **Zero Hyperparameter Tuning**: Using the same $\lambda$ across all tasks is highly beneficial for practical deployment.

## Limitations & Future Work
- Theoretical analysis relies on the "SPL $\approx$ Spectral Decomposition" equivalence from Tang et al. (2023); the tightness for non-SimSiam-style frameworks (e.g., BYOL+predictor) is not fully discussed.
- The introduction of SimbaV2-SPL mixes the variables of "adding SPL" and "adding R2R2," partially diluting the purity of the comparison (though +SPL and +SPL+R2R2 are reported separately).
- Experiments focus on continuous control (MuJoCo + DMC-Hard); discrete actions (Atari), sparse rewards, and pixel inputs remain unverified.
- Training overhead is approximately +12% wall-clock time, which may require optimization for extreme scales.

## Related Work & Insights
- **vs. VICReg / Barlow Twins**: Native SSL redundancy reduction with zero-centering is structurally destructive to SPL; R2R2 fixes this with a non-centered form.
- **vs. REDQ / CrossQ**: Value-centric high UTD methods addressing critic bias; R2R2 addresses orthogonal representation collapse and can be stacked.
- **vs. SimBa / SimbaV2 / BRO**: Rely on architectural normalization (LN) and dropout to stabilize high UTD; R2R2 proves architecture alone is insufficient and regularization is a necessary supplement.
- **vs. SPR / TD7 (SPL Family)**: Native SPL representations are unstable under high UTD; R2R2 stabilizes the encoder directly via redundancy reduction.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The insight using spectral analysis to explain the incompatibility of SSL centering with SPL is highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across 11 environments, 4 baselines, 2 UTD settings, plus ER/spectral analysis and wall-clock times.
- Writing Quality: ⭐⭐⭐⭐⭐ Logical flow from theoretical lemmas to propositions, experiments, and ablations.
- Value: ⭐⭐⭐⭐ Provides a new representation-layer stabilization mechanism for high UTD RL, proven compatible with SOTA architectures.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Experience-based Knowledge Correction for Robust Planning in Minecraft](../../ICLR2026/robotics/experience-based_knowledge_correction_for_robust_planning_in_minecraft.md)
- [\[ICML 2026\] Plan in Sandbox, Navigate in Open Worlds: Learning Physics-Grounded Abstracted Experience for Embodied Navigation](plan_in_sandbox_navigate_in_open_worlds_learning_physics-grounded_abstracted_exp.md)
- [\[ICML 2026\] From Abstraction to Instantiation: Learning Behavioral Representation for Vision-Language-Action Model](from_abstraction_to_instantiation_learning_behavioral_representation_for_vision-.md)
- [\[NeurIPS 2025\] Sample-Efficient Tabular Self-Play for Offline Robust Reinforcement Learning](../../NeurIPS2025/robotics/sample-efficient_tabular_self-play_for_offline_robust_reinforcement_learning.md)
- [\[ICML 2026\] Contrastive Representation Regularization for Vision-Language-Action Models](contrastive_representation_regularization_for_vision-language-action_models.md)

</div>

<!-- RELATED:END -->

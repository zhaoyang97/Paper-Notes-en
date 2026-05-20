---
title: >-
  [Paper Note] Active Tabular Augmentation via Policy-Guided Diffusion Inpainting
description: >-
  [ICML 2026][Tabular Data Augmentation] This paper formalizes the "fidelity-utility gap" in tabular augmentation (where generators optimize for distribution matching, but augmentation value comes from low-density regions)…
tags:
  - "ICML 2026"
  - "Tabular Data Augmentation"
  - "Diffusion Inpainting"
  - "Utility-Driven Selection"
  - "Conservative Submission"
  - "Fidelity-Utility Gap"
date: 2026-05-08
content_hash: ebbb7103e15be1cd
---

# Active Tabular Augmentation via Policy-Guided Diffusion Inpainting

**Conference**: ICML 2026  
**arXiv**: [2605.10315](https://arxiv.org/abs/2605.10315)  
**Code**: https://github.com/oooranz/TAP  
**Area**: Data Augmentation / Tabular Generation / Reinforcement Learning  
**Keywords**: Tabular Data Augmentation, Diffusion Inpainting, Utility-Driven Selection, Conservative Submission, Fidelity-Utility Gap

## TL;DR
This paper formalizes the "fidelity-utility gap" in tabular augmentation (where generators optimize for distribution matching, but augmentation value comes from low-density regions), and proposes the TAP algorithm, which uses diffusion inpainting for manifold-constrained proposals, policy-guided utility-aligned selection, and conservative windowed submission with hard constraint gating. On 7 real tabular datasets, TAP improves classification accuracy by up to 15.6% and reduces regression RMSE by 32% compared to baselines.

## Background & Motivation

**Background**  
Tabular data drives decision-making in healthcare, finance, and science, but labeled data is often scarce. Data augmentation is a common remedy, but is fragile for tabular applications—heterogeneous features and strong inter-column dependencies mean even small perturbations can violate constraints or introduce spurious relationships.

**Limitations of Prior Work**  
1. **Fidelity-Utility Misalignment**: Existing generators (GANs, VAEs, diffusion models) optimize for distribution matching $P(X,Y)$, encouraging sampling from high-density regions. However, successful augmentation samples often come from low-density boundaries or under-covered groups—contradicting the generator's objective.
2. **Insufficient Static Evaluation**: Simple methods like SMOTE generate statistically unrealistic samples, yet often improve classifiers, suggesting fidelity is not a sufficient condition.

**Key Challenge**  
A fundamental mismatch exists between the generator's training objective and the augmentation evaluation objective: generators focus on $\max_x\log P(x)$, while augmentation seeks $\min_\theta L(\theta,D\cup S)$.

**Goal**  
To learn not only "how to generate," but also "what to generate" and "when to inject," so that samples dynamically adapt to the evolving learner.

**Key Insight**  
Formalize augmentation as a sequential control problem: maintain a commitment buffer and a temporary pool each round, and use a policy to decide generation conditions and injection timing. Influence functions guide the design—utility is approximated by the action of the learner's loss gradient and the inverse Hessian.

**Core Idea**  
Utility-driven augmentation is achieved via three principles: (1) Manifold soft constraints (diffusion inpainting) + hard constraint gating (valid value range checks) = two-layer fidelity; (2) Policy learning (conditioned on learner state) + utility-aligned selection = targeted objectives; (3) Conservative windowed submission (accumulate candidates, batch submit only if pool utility exceeds threshold) = robustness against noise.

## Method

### Overall Architecture
TAP formalizes tabular augmentation as a finite-horizon, budget-constrained MDP. State $(r_t, D_t)$ (remaining budget and committed training set), policy $\pi$ decides two actions: (i) round budget $s_t$, (ii) generation condition $(c,\eta,\rho)$ (target class, template, exploration strength). Greedy single-round allocation selects the optimal condition for $k_i$ units per individual $i$; dynamic programming solves multi-round budgeting.

### Key Designs

1. **Diffusion Inpainting + Triplet Action Space**:

    - Function: Generates manifold-local, high-fidelity, and diverse samples via conditional partial-column diffusion, decoupling target/locality/diversity in generation.
    - Mechanism: Freeze part of real sample columns as anchors, reverse diffuse on remaining columns, guided by conditions (e.g., target class label): $x_{\bar m}^{(s-1)}\leftarrow\sqrt{\bar\alpha_{s-1}}x_{\bar m}+\sqrt{1-\bar\alpha_{s-1}}\epsilon$ (noise overwrites fixed columns). Action $a=(c,\eta,\rho)$ controls class condition, conservative/explore template, and rewrite column ratio, inducing different proposal distributions $Q_a(\cdot|D_t)$.
    - Design Motivation: Anchor conditions enforce manifold locality, restricted columns minimize spurious changes, conditional constraints align generation with augmentation goals; the triplet allows the policy to balance trade-offs at different learning stages.

2. **Utility-Driven Policy + Hard Constraint Gating**:

    - Function: Selects high-value generation conditions based on real-time learner state, and enforces tabular-specific validity constraints.
    - Mechanism: Learner state $s_t=(\delta_t,u_t,g_t,d_t)$ tracks under-coverage, uncertainty, recent gating pass rate, and redundancy. The policy takes state as input and outputs action distribution, maximizing KL-regularized marginal utility $\max_\pi \mathbb E[\hat A_t]-\beta\,\mathrm{KL}(\pi\|\pi_{\text{ref}})$. Generated candidates are checked by acceptance function $G(x;D_t)\in\{0,1\}$ for class validity, value range, and logical consistency (e.g., age < death age).
    - Design Motivation: State design directly corresponds to the gradient components of the learner's loss in influence function diagnostics, automatically guiding the policy toward boundary/under-covered regions; hard gating provides a second safety layer beyond the soft manifold.

3. **Conservative Windowed Submission**:

    - Function: Accumulates candidates and only batch-submits when the pool's collective utility is sufficient, countering noisy estimates.
    - Mechanism: Maintains a sliding window $P_t$ of length $K$. At submission checkpoints, computes $\Delta\hat U(D_t,P_t^{(K)})=\hat L_\psi(D_t)-\hat L_\psi(D_t\cup P_t^{(K)})$ (measured on a hard query set using TabPFN plugin evaluator). Only submits if $\Delta\hat U>\tau+\epsilon_t$, where $\tau$ is the minimum utility threshold and $\epsilon_t$ is the calibrated uncertainty interval.
    - Design Motivation: With scarce data, single samples may be harmful; accumulated window utility is more stable and more likely to surpass the noise threshold.

### Loss & Training

Trajectory objective: $J(\pi)=\mathbb E_\pi[\sum_{t\geq 1}\gamma^{t-1}\Delta U(D_t,P_t)]$, decomposed at submission times as $\sum_i \Delta U(D_{t_i},P_i)$. The plugin utility evaluator $f_\psi$ uses TabPFN (a fast context learner), only for candidate ranking; final reported gains are obtained by retraining the full model on the validation set.

## Key Experimental Results

### Main Results

| Dataset | $N_{\text{real}}$ | Metric | SMOTE | TVAE | CTGAN | ARF | SPADA | TabDDPM | TabDiff | TAP |
|---------|-------------------|--------|-------|------|-------|-----|-------|---------|---------|-----|
| MiceProtein | 20 | Acc↑ | 36.21 | 41.34 | 36.93 | 32.35 | 36.91 | 37.59 | 34.05 | **44.60** |
|           | 100 | Acc↑ | 71.96 | 71.27 | 63.59 | 65.13 | 65.01 | 68.86 | 66.95 | **73.06** |
|           | 500 | Acc↑ | 96.44 | 96.65 | 93.75 | 93.71 | 94.56 | 96.13 | 93.81 | **96.11** |
| Credit-G  | 20 | Acc↑ | 66.37 | 59.06 | 65.79 | 65.48 | 64.25 | 57.58 | 63.99 | **68.13** |
|           | 100 | Acc↑ | 67.53 | 68.27 | 68.65 | 67.26 | 67.27 | 66.09 | 64.07 | **70.73** |
| Electricity | 50 | Acc↑ | 69.05 | 64.71 | 69.09 | 63.64 | 70.81 | 69.61 | 66.11 | **71.55** |
|           | 100 | Acc↑ | 72.73 | 68.21 | 72.15 | 67.21 | 74.02 | 72.83 | 70.97 | **74.73** |
| Avg. Gain | 20 | $\Delta$ | +3.5% | +5.8% | +4.2% | base | +2.1% | +1.8% | 0% | **+15.6%** |
|           | 100 | $\Delta$ | +2.1% | -1.5% | -10.4% | base | -2.3% | -3.2% | -6.7% | **+3.8%** |

### Ablation Study

| Configuration | Validation Accuracy | Description |
|---------------|--------------------|-------------|
| Diffusion only, no policy | 71.2% | High fidelity but untargeted |
| Greedy policy, no submission | 70.8% | Real-time injection, vulnerable to noise |
| Hard gating, no soft manifold | 68.5% | Overly strict filtering, reduced diversity |
| Full TAP | **74.3%** | All components combined |
| TAP without windowed submission | 72.1% | Lacks conservative mechanism, more harmful injections |

### Key Findings
- **Fidelity is not sufficient**: TabDDPM/TabDiff achieve highest fidelity but limited or negative augmentation gains; TAP achieves lower fidelity but highest augmentation gain.
- **Greatest gains in scarce data**: At $N=20$, TAP outperforms the best baseline by +15.6%; at $N=500$, the gain drops to ~1% (augmentation space narrows as data increases).
- **Two-layer manifold + hard constraint is effective**: Diffusion only achieves 71.2%, while two-layer achieves 74.3%; hard gating alone reduces diversity to 68.5%.
- **Policy learning surpasses fixed allocation**: The policy adaptively balances exploration vs exploitation under different scarcity levels; fixed greedy achieves 70.8%, adaptive policy 74.3%.
- **Windowed submission prevents harmful injections**: Windowed 74.3% vs no window 72.1%, with larger differences under high noise.

## Highlights & Insights
- **Depth of problem formalization**: Augmentation is viewed as a sequential control problem, with influence function diagnostics providing intuitive design explanations. Identifying the "fidelity-utility gap" is a fundamental insight for augmentation theory.
- **Thoroughness of multi-layer design**: Soft manifold + hard constraint + utility policy + conservative submission form a complete defense, each layer targeting different failure modes.
- **Pragmatic uncertainty handling**: Windowed submission and the $\tau+\epsilon_t$ threshold provide an elegant engineering response to noise estimation under data scarcity.

## Limitations & Future Work
- **Dependence on reference distribution**: Assumes accurate $P$ can be learned from historical data; may fail under severe distribution shift.
- **Evaluator as intermediate variable**: TabPFN evaluation accuracy directly affects policy training; the paper does not quantify the impact of evaluator error on policy.
- **Scale limitation**: Experiments max out at $N\approx 10k$; performance on very large tables or ultra-high-dimensional features is unknown.

## Related Work & Insights
- **vs SMOTE**: SMOTE's neighborhood interpolation has low fidelity but is often effective; TAP modernizes and makes learnable the paradigm of "sampling from low-density, high-uncertainty regions."
- **vs GANs/VAEs/Diffusion**: These methods optimize for distribution matching, but this paper reveals it as the "wrong objective"; TAP corrects the misalignment via explicit utility optimization.
- **vs Influence Functions**: Koh & Liang 2017 use influence functions to understand sample impact; this paper reverses the perspective—using them to guide generation, a creative shift.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Identification of the "fidelity-utility gap" is a new perspective; policy-guided tabular augmentation and conservative submission are novel methods.
- Experimental Thoroughness: ⭐⭐⭐⭐ 7 datasets, 5 scarcity levels, multiple baselines, thorough ablation; does not cover clustering/anomaly detection tasks.
- Writing Quality: ⭐⭐⭐⭐ Clear problem formalization, explicit design principles, detailed experiments.
- Value: ⭐⭐⭐⭐⭐ Direct value in scarce data scenarios (healthcare/finance), challenges the "fidelity above all" misconception.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Support-Proximity Augmented Diffusion Estimation for Offline Black-Box Optimization](support-proximity_augmented_diffusion_estimation_for_offline_black-box_optimizat.md)
- [\[ICML 2026\] Mixture Prototype Flow Matching for Open-Set Supervised Anomaly Detection](mixture_prototype_flow_matching_for_open-set_supervised_anomaly_detection.md)
- [\[ICML 2026\] Diagnosing and Correcting Concept Omission in Multimodal Diffusion Transformers](diagnosing_and_correcting_concept_omission_in_multimodal_diffusion_transformers.md)
- [\[ICML 2026\] Consistent Diffusion Language Models](consistent_diffusion_language_models.md)
- [\[ICML 2026\] PODiff: Latent Diffusion in Proper Orthogonal Decomposition Space for Scientific Super-Resolution](podiff_latent_diffusion_in_proper_orthogonal_decomposition_space_for_scientific_.md)

</div>

<!-- RELATED:END -->

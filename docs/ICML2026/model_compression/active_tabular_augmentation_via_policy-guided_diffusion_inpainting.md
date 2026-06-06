---
title: >-
  [Paper Note] Active Tabular Augmentation via Policy-Guided Diffusion Inpainting
description: >-
  [ICML 2026][Model Compression][Tabular Data Augmentation] This paper formalizes the "fidelity-utility gap" in tabular augmentation (where generators optimize for distribution matching while augmentation value originates…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Tabular Data Augmentation"
  - "Diffusion Inpainting"
  - "Utility-Driven Selection"
  - "Conservative Submission"
  - "Fidelity-Utility Gap"
date: 2026-05-08
content_hash: c422fccc8b5f37c6
---

# Active Tabular Augmentation via Policy-Guided Diffusion Inpainting

**Conference**: ICML 2026  
**arXiv**: [2605.10315](https://arxiv.org/abs/2605.10315)  
**Code**: https://github.com/oooranz/TAP  
**Area**: Data Augmentation / Tabular Generation / Reinforcement Learning  
**Keywords**: Tabular Data Augmentation, Diffusion Inpainting, Utility-Driven Selection, Conservative Submission, Fidelity-Utility Gap

## TL;DR
This paper formalizes the "fidelity-utility gap" in tabular augmentation (where generators optimize for distribution matching while augmentation value originates from low-density regions). It proposes the TAP algorithm, which uses diffusion inpainting for manifold-constrained proposals, policy-guided utility alignment for selection, and hard-constraint gating with conservative window submission. On 7 real-world tabular datasets, TAP improves classification accuracy by up to 15.6% and reduces regression RMSE by up to 32% compared to baselines.

## Background & Motivation

**Background**
Tabular data drives decision-making in healthcare, finance, and science; however, labeled data is often scarce. Data augmentation is a common improvement method but remains fragile for tabular applications—the heterogeneity of tabular features and strong dependencies between columns mean that even minor perturbations can violate constraints or introduce spurious relationships.

**Limitations of Prior Work**
1. **Fidelity-Utility Misalignment**: Existing generators (GANs, VAEs, Diffusion models) optimize for distribution matching $P(X,Y)$, encouraging sampling from high-density regions. However, samples that successfully enhance a model often originate from low-density boundaries or under-covered populations where the learner is uncertain—this directly contradicts the generation objective.
2. **Insufficient Static Evaluation**: Simple methods like SMOTE generate samples that are statistically less "real" yet often effectively improve classifiers, suggesting that fidelity is not a sufficient or necessary condition.

**Key Challenge**
A fundamental mismatch between the generator training objective and the augmentation evaluation objective: generators focus on $\max_x\log P(x)$, while augmentation focuses on $\min_\theta L(\theta,D\cup S)$.

**Goal**
To learn not just "how to generate," but "what to generate" and "when to inject," allowing samples to dynamically adapt to the evolving learner.

**Key Insight**
Augmentation is formalized as a sequential control problem: maintaining a committed buffer and a temporary pool in each round, using a policy to determine generation conditions and injection timing. Design is guided by influence function diagnostics—where utility approximately equals the interaction between the learner's loss gradient and the inverse Hessian.

**Core Idea**
Utility-driven augmentation is achieved through three principles: (1) Manifold soft constraints (Diffusion Inpainting) + Hard constraint gating (checking valid value ranges) = Two-layer fidelity; (2) Policy learning (based on learner state) + Utility-aligned selection = Target focus; (3) Conservative window submission (accumulating candidates and only submitting batches when pool gains exceed a threshold) = Robustness against adversarial noise.

## Method

### Overall Architecture
TAP formalizes tabular augmentation as a finite-horizon, budget-constrained MDP. Given the state $(r_t, D_t)$ (remaining budget and committed training set), the policy $\pi$ decides two actions: (i) the round budget $s_t$, and (ii) generation conditions $(c,\eta,\rho)$ (target class, template, and exploration intensity). A greedy single-round allocation selects $k_i$ units for individual $i$ under optimal conditions; multi-round budgets are solved via dynamic programming.

### Key Designs

1. **Diffusion Inpainting + Triplet Action Space**:
    - **Function**: Generates manifold-local, high-fidelity, and diverse samples via conditional partial-column diffusion, decoupling target, locality, and diversity.
    - **Mechanism**: Freezes a subset of columns from real samples as anchors, performs reverse diffusion on the remaining columns, and guides the process with conditions (e.g., target labels): $x_{\bar m}^{(s-1)}\leftarrow\sqrt{\bar\alpha_{s-1}}x_{\bar m}+\sqrt{1-\bar\alpha_{s-1}}\epsilon$ (where noise overwrites fixed columns). The action $a=(c,\eta,\rho)$ controls the class condition, conservative/explore templates, and the proportion of rewritten columns within the template, inducing different proposal distributions $Q_a(\cdot|D_t)$.
    - **Design Motivation**: Anchor conditions enforce manifold locality, restricted columns minimize spurious variations, and conditional constraints align generation with augmentation goals. The triplet allows the policy to use different trade-offs for varying learning stages.

2. **Utility-Driven Policy + Hard Constraint Gating**:
    - **Function**: Selects high-value generation conditions based on the learner's real-time state and enforces tabular-specific validity constraints.
    - **Mechanism**: The learner state $s_t=(\delta_t,u_t,g_t,d_t)$ tracks under-coverage, uncertainty, recent gating pass rates, and redundancy, respectively. The policy takes the state as input and outputs an action distribution, maximizing KL-regularized marginal utility: $\max_\pi \mathbb E[\hat A_t]-\beta\,\mathrm{KL}(\pi\|\pi_{\text{ref}})$. Generated candidates are checked by an acceptance function $G(x;D_t)\in\{0,1\}$ for class validity, numerical ranges, and logical consistency (e.g., Age < Age of Death).
    - **Design Motivation**: The state design directly corresponds to gradient components of the learner's loss in influence function diagnostics, automatically guiding the policy toward boundaries or under-covered areas. Hard gating provides a second safety line beyond the soft manifold.

3. **Conservative Window Submission**:
    - **Function**: Accumulates candidates and performs batch submissions only when the collective gain of the pool is sufficient to combat noise estimation.
    - **Mechanism**: Maintains a sliding window $P_t$ of length $K$. At submission checkpoints, it calculates $\Delta\hat U(D_t,P_t^{(K)})=\hat L_\psi(D_t)-\hat L_\psi(D_t\cup P_t^{(K)})$ (measured using a TabPFN plug-in evaluator on a hard query set). Submission occurs only if $\Delta\hat U>\tau+\epsilon_t$, where $\tau$ is the minimum gain threshold and $\epsilon_t$ is the calibrated uncertainty interval.
    - **Design Motivation**: Single samples can be harmful under data scarcity; collective utility becomes more stable once the window accumulates, making it easier to exceed noise boundaries.

### Loss & Training
The trajectory objective is $J(\pi)=\mathbb E_\pi[\sum_{t\geq 1}\gamma^{t-1}\Delta U(D_t,P_t)]$, decomposed along submission points as $\sum_i \Delta U(D_{t_i},P_i)$. The plug-in utility evaluator $f_\psi$ uses TabPFN (a fast in-context learner) for candidate ranking; final reported gains are obtained by re-training the full model on the validation set.

## Key Experimental Results

### Main Results

| Dataset | $N_{\text{Real}}$ | Metric | SMOTE | TVAE | CTGAN | ARF | SPADA | TabDDPM | TabDiff | TAP |
|--------|-------|------|-------|------|-------|-----|-------|---------|---------|-----|
| MiceProtein | 20 | Acc↑ | 36.21 | 41.34 | 36.93 | 32.35 | 36.91 | 37.59 | 34.05 | **44.60** |
| | 100 | Acc↑ | 71.96 | 71.27 | 63.59 | 65.13 | 65.01 | 68.86 | 66.95 | **73.06** |
| | 500 | Acc↑ | 96.44 | 96.65 | 93.75 | 93.71 | 94.56 | 96.13 | 93.81 | **96.11** |
| Credit-G | 20 | Acc↑ | 66.37 | 59.06 | 65.79 | 65.48 | 64.25 | 57.58 | 63.99 | **68.13** |
| | 100 | Acc↑ | 67.53 | 68.27 | 68.65 | 67.26 | 67.27 | 66.09 | 64.07 | **70.73** |
| Electricity | 50 | Acc↑ | 69.05 | 64.71 | 69.09 | 63.64 | 70.81 | 69.61 | 66.11 | **71.55** |
| | 100 | Acc↑ | 72.73 | 68.21 | 72.15 | 67.21 | 74.02 | 72.83 | 70.97 | **74.73** |
| Avg Gain | 20 | $\Delta$ | +3.5% | +5.8% | +4.2% | base | +2.1% | +1.8% | 0% | **+15.6%** |
| | 100 | $\Delta$ | +2.1% | -1.5% | -10.4% | base | -2.3% | -3.2% | -6.7% | **+3.8%** |

### Ablation Study

| Configuration | Val Acc | Description |
|------|-----------|------|
| Diffusion only w/o Policy | 71.2% | High fidelity but not targeted |
| Greedy Policy w/o Submission | 70.8% | Real-time injection, easily deceived by noise |
| Hard Gating w/o Soft Manifold | 68.5% | Excessive filtering, diversity suffers |
| Full TAP | **74.3%** | All components synergized |
| TAP w/o Window Submission | 72.1% | Lacks conservative mechanism, prone to harmful injections |

### Key Findings
- **Fidelity is not a sufficient condition**: TabDDPM/TabDiff have the highest fidelity but limited or negative augmentation gains; TAP has lower fidelity but the highest augmentation gains.
- **Largest gains in data-scarce regimes**: At $N=20$, gain is +15.6% compared to the best baseline; at $N=500$, it drops to ~1% (augmentation headroom narrows as data becomes sufficient).
- **Two-layer manifold + Hard constraint is effective**: Diffusion alone (71.2%) is inferior to the dual-layer approach (74.3%); hard gating alone (68.5%) hurts diversity.
- **Policy learning outperforms fixed allocation**: The policy adaptively adjusts exploration vs. exploitation across different levels of scarcity; fixed greedy (70.8%) is inferior to adaptive (74.3%).
- **Window submission prevents harmful injections**: Window (74.3%) vs. no window (72.1%) shows a more significant gap under high noise.

## Highlights & Insights
- **Depth of Problem Formalization**: Treating augmentation as a sequential control problem and using influence function diagnostics provides an intuitive explanation for the design. The identification of the "fidelity-utility gap" is a fundamental insight into augmentation theory.
- **Thorough Multi-layer Design**: The soft manifold + hard constraint + utility policy + conservative submission forms a complete defense, with each layer targeting different failure modes.
- **Pragmatic Uncertainty Handling**: Window submission and the $\tau+\epsilon_t$ threshold are elegant engineering responses to noise estimation under data scarcity.

## Limitations & Future Work
- **Dependency on Reference Distribution**: Assumes an accurate $P$ can be learned from historical data; may fail under severe distribution shifts.
- **Evaluator as an Intermediate Variable**: TabPFN evaluation accuracy directly affects policy training; the paper does not quantify the impact of estimator error on the policy.
- **Scale Constraints**: Maximum $N\approx 10k$ in experiments; performance on massive tables or ultra-high-dimensional features remains unknown.

## Related Work & Insights
- **vs SMOTE**: SMOTE's neighborhood interpolation has low fidelity but is often effective; TAP transforms "sampling from low-density high-uncertainty regions" into a modernized, learnable paradigm.
- **vs GANs/VAEs/Diffusion**: These methods optimize distribution matching, which this paper reveals as a "wrong objective"; TAP corrects this misalignment with explicit utility optimization.
- **vs Influence Functions**: Koh & Liang 2017 used influence functions to understand sample impact; this paper applies it inversely—using it to guide generation direction, a creative shift in perspective.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The identification of the "fidelity-utility gap" is a new perspective, and both policy-guided tabular augmentation and conservative submission are novel methods.
- Experimental Thoroughness: ⭐⭐⭐⭐ 7 datasets, 5 scarcity levels, multiple baselines, and thorough ablations; however, it does not cover other tasks like clustering or anomaly detection.
- Writing Quality: ⭐⭐⭐⭐ Clear problem formalization, explicit design principles, and sufficient experimental detail.
- Value: ⭐⭐⭐⭐⭐ Directly valuable in data-scarce scenarios such as healthcare/finance, breaking the "fidelity-first" myth.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Active Budget Allocation for Efficient Scaling Law Estimation via Surrogate-Guided Pruning](active_budget_allocation_for_efficient_scaling_law_estimation_via_surrogate-guid.md)
- [\[ICML 2026\] End-to-End Compression for Tabular Foundation Models](end-to-end_compression_for_tabular_foundation_models.md)
- [\[ICML 2026\] Entropy-Aware On-Policy Distillation of Language Models](entropy-aware_on-policy_distillation_of_language_models.md)
- [\[ICML 2026\] Resting Neurons, Active Insights: Robustify Activation Sparsity for Large Language Models](resting_neurons_active_insights_robustify_activation_sparsity_for_large_language.md)
- [\[ICML 2026\] Advantage Collapse in Group Relative Policy Optimization: Diagnosis and Mitigation](advantage_collapse_in_group_relative_policy_optimization_diagnosis_and_mitigatio.md)

</div>

<!-- RELATED:END -->

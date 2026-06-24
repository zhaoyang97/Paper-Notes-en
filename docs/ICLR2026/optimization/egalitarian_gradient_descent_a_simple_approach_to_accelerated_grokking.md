---
title: >-
  [Paper Note] Egalitarian Gradient Descent: A Simple Approach to Accelerated Grokking
description: >-
  [ICLR 2026][Optimization][Grokking] The paper attributes the long plateau in grokking to the severe imbalance of the gradient spectrum and proposes EGD: leveling the update speed across different singular directions without changing the primary gradient direction, thereby significantly compressing the "memorization-then-generalization" delay into a few epochs.
tags:
  - "ICLR 2026"
  - "Optimization"
  - "Grokking"
  - "Gradient Spectral Normalization"
  - "Fisher Preconditioning"
  - "Randomized SVD"
  - "Generalization Delay"
date: 2026-05-08
content_hash: c342eec49dff7268
---

# Egalitarian Gradient Descent: A Simple Approach to Accelerated Grokking

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=wCnHeql3ow](https://openreview.net/forum?id=wCnHeql3ow)  
**Code**: https://github.com/asahebpa/Egalitarian-Gradient-Descent (Official)  
**Area**: optimization  
**Keywords**: Grokking, Gradient Spectral Normalization, Fisher Preconditioning, Randomized SVD, Generalization Delay  

## TL;DR
The paper attributes the long plateau in grokking to the severe imbalance of the gradient spectrum and proposes EGD: leveling the update speed across different singular directions without changing the primary gradient direction, thereby significantly compressing the "memorization-then-generalization" delay into a few epochs.

## Background & Motivation
**Background**: The grokking phenomenon has been repeatedly observed in tasks such as modular arithmetic and sparse parity. The typical curve shows training accuracy reaching nearly 100% very quickly, while test accuracy remains at a random level for a long duration before a sudden jump. Existing explanations include kernel escape, representation competition, and the edge of numerical stability, but there lacks an actionable and intervention-oriented unified optimization perspective on "why super-long plateaus occur."

**Limitations of Prior Work**: The most direct engineering problem is not whether generalization can eventually occur, but that it arrives too late. Even if the final test accuracy is high, the training process wastes a large number of iterations in the plateau phase. Methods like Grokfast can accelerate this but introduce historical gradient buffers and tuning burdens, which are not lightweight enough.

**Key Challenge**: The authors argue that the key contradiction is the excessive scale difference of gradients in different principal directions, causing optimization to converge rapidly in "fast directions" while progressing extremely slowly in "slow directions"; the latter often corresponds to structural features that determine the generalization transition. In other words, it is not that the model cannot learn, but that the progress in certain necessary directions is slowed by the spectral condition number.

**Goal**:  
1. Provide an analytical toy setup to prove that the plateau length is directly related to anisotropy parameters (e.g., $\varepsilon$);  
2. Construct an update rule as simple as possible that can directly replace SGD to ensure consistent progress speed across principal directions;  
3. Verify "faster grokking without degrading final performance" on parity/modular arithmetic and more realistic data.

**Key Insight**: The paper starts from the singular value decomposition (SVD) of the gradient matrix, keeping the singular vectors (directional information) unchanged while only modifying the singular values (speed information). By "equalizing" the step size of each principal direction, the delay caused by ill-conditioned dynamics can be reduced.

**Core Idea**: Transform the gradient $G$ of each layer into $\tilde G=(GG^\top)^{-1/2}G$, so that all singular values are normalized to unity, achieving an "egalitarian" update speed in principal directions.

## Method

### Overall Architecture
EGD can be viewed as a plug-in step that "performs spectral normalization on gradients before updating parameters." In a training iteration, the forward and backward passes are exactly the same as conventional training; the transformation is applied to each layer's gradient matrix only before the parameter update. The input is the original gradient $G\in\mathbb{R}^{m\times p}$, and the output is the rescaled $\tilde G$, which is then passed to the original optimizer (SGD/Adam, etc.) for the step.

This framework does not depend on specific task structures. The paper focuses on "optimization dynamics" rather than model architecture innovation; therefore, it can be inserted into MLPs, CNNs, or Transformers. The authors also emphasize that EGD can be turned off once grokking is detected, reverting to vanilla updates to reduce additional computation.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
	A[Backpropagate to obtain gradient G for each layer] --> B[Compute Fisher approximation F = GG^T]
	B --> C[Spectral Normalization: ˜G = F^{-1/2}G]
	C --> D[Replace original gradient with ˜G to update parameters]
	D --> E[Monitor validation set: Turn off EGD if grokking has occurred]
```

### Key Designs

**1. Gradient Egalitarianism: Maintaining Direction, Unifying Speed**

The core transformation is 
$$
\tilde G=(GG^\top)^{-1/2}G.
$$
If $G=USV^\top$, then $\tilde G=UV^\top$. This means left and right singular vectors remain unchanged (optimizing along the original "meaningful" principal directions), but all singular values become 1. The author's view is: the root cause of the grokking plateau is not "wrong directions" but "certain directions being too slow"; thus, correcting only the speed scale without destroying the directional structure is a minimally invasive and effective modification.

**2. Relation to Natural Gradient: Not the same algorithm, but shared geometric motivation**

The paper provides the relationship:
$$
\tilde G = F^{1/2}\bar G,\quad \bar G=F^{-1}G,
$$
where $\bar G$ is the natural gradient form. EGD is not a direct execution of NGD but a "whitened" version of the update: it does not reweight as strongly as NGD via $F^{-1}$ but aims for singular value equalization. This design is more stable in practice and closer to the goal of "letting every principal direction proceed at the same speed."

**3. Engineering Feasibility: Exact SVD + Approximate RSVD Dual Paths**

The primary extra cost of EGD is the SVD per step. The authors provide two mitigation paths: one is to turn off EGD after reaching target validation accuracy; the other is to use randomized SVD (RSVD) approximations, leveraging the common low-rank structure of gradients to reduce time overhead. Experiments show that RSVD at an appropriate rank often outperforms full SVD in wall-clock time while still being significantly faster than the grokking moment of vanilla SGD.

### Mechanism
Taking modular addition ($p=97$) as an example: vanilla SGD typically pulls training accuracy high in a very early stage, but test accuracy jumps only after a large number of epochs; after EGD equalizes the gradient spectrum, slow directions are no longer suppressed, and the test curve begins to rise synchronously in the early stages.

From a "dynamics" perspective, this is equivalent to replacing the anisotropic convergence, which is approximately
$$
A\approx\begin{bmatrix}1-\eta m_2 & 0\\0&1-\eta\varepsilon\end{bmatrix}
$$
(where the slow mode is determined by $\varepsilon\ll1$), with a form closer to uniform decay. The plateau length is no longer hindered by the $1/\varepsilon$ level.

### Loss & Training
The main experiments use standard task configurations:
- Sparse parity: Two-layer ReLU network, hinge loss + weight decay;
- Modular arithmetic: Two-layer ReLU network, cross-entropy + weight decay;
- Baseline methods include vanilla SGD, EGD, and a simpler column normalization.

Importantly, EGD does not require new training schedulers or complex history buffers; its hyperparameter burden mainly comes from the truncation rank selection in RSVD, while the full SVD version has almost "zero new hyperparameters."

## Key Experimental Results

### Main Results
The main conclusions of the paper are highly consistent across three types of tasks: EGD significantly advances grokking, and the final accuracy is not lower than the baseline.

| Task | Phenomenon (Vanilla SGD) | Phenomenon (EGD) | Conclusion |
|---|---|---|---|
| Modular Addition ($p=79,97,127$) | Sudden jump after long plateau | Rapid jump within few epochs | Plateau significantly shortened |
| Modular Multiplication ($p=79,97,127$) | Training converges first, test lags significantly | Test rises almost synchronously | Generalization delay significantly mitigated |
| Sparse Parity ($(n,k)=(400,2),(100,3),(50,4)$) | Obvious delayed grokking | High test accuracy reached early | Equally effective for hard combinatorial tasks |

### Ablation Study
The appendix provides an efficiency comparison for reaching 95% accuracy in modular addition (including wall-clock time), summarized in the table below.

| Method | Epochs to 95% (Relative to Vanilla) | Time to 95% (Relative to Vanilla) | Extra Cost |
|---|---:|---:|---|
| Vanilla SGD | 1.0x | 1.0x | None |
| EGD (SVD) | ~45x-53x fewer | ~10x-14x faster | SVD per step |
| EGD (RSVD, proper rank) | ~23x-45x fewer | Often faster than full SVD | Rank selection needed |
| Column Norm | ~12x-16x fewer | Sometimes fastest | Significant speedup but usually weaker than EGD |

### Key Findings
- The most stable conclusion is the "significant advancement of grokking in the epoch dimension," which directly aligns with the paper's spectral dynamics explanation.
- In the wall-clock dimension, full SVD is not always optimal; RSVD provides a more practical trade-off between speed and effectiveness.
- A valuable engineering observation: even simplifying to column normalization is significantly better than vanilla, indicating that "reducing gradient spectral imbalance" itself is an effective direction.

## Highlights & Insights
- Evolution from "phenomenal description" to "actionable mechanism." The paper links the grokking plateau to the gradient spectral condition number and provides an executable update rule rather than stopping at a posteriori explanations.
- The perspective of decoupling direction and speed is practical. Retaining singular vectors while only rescaling singular values allows the method to balance stability and generalization needs.
- The loop between theory and engineering is complete. A toy model provides analytical conclusions, and major experiments plus the appendix prove these conclusions do not fail in more complex scenarios.
- The relationship with Grokfast is clearly articulated. EGD shares the inductive bias of "suppressing fast directions and boosting the influence of slow directions," but its implementation is lighter with lower memory overhead.

## Limitations & Future Work
- Computational cost remains a realistic constraint. Even if RSVD mitigates overhead, performing spectral decomposition at every step remains expensive in high-frequency update scenarios for large models.
- The theoretical backbone currently relies on simplified settings. Rigorous convergence and generalization bounds for deep non-linear networks are not yet complete, which the paper lists as a future direction.
- While the paper emphasizes "accelerating grokking," the robustness analysis against different data noise, label corruption, and extremely small batch sizes is still insufficient.
- Future work could explore joint scheduling with AdamW, Muon, and learning rate rewarming to form a hybrid paradigm where "EGD pulls up generalization in the early stage, and conventional optimization refines the model later."

## Related Work & Insights
- **vs Grokfast (Lee et al., 2024)**: Grokfast amplifies slow gradient components via low-pass filtering; it is effective but depends on historical buffers and hyperparameters. EGD performs spectral normalization directly, which is simpler, more memory-efficient, and has a clearer explanation of "equal-speed principal directions."
- **vs Natural Gradient (Amari, 1998)**: Both are related to Fisher geometry, but EGD targets singular value equalization rather than the full-inverse preconditioning of standard NGD. The practical focus is on "eliminating directional speed inequality."
- **vs Muon series**: Both Muon and EGD tend toward orthogonal/isospectral update geometries, but Muon follows an empirical engineering route. EGD is derived from the grokking mechanism and offers stronger interpretability.
- **Related Insight**: When training curves show "early training convergence but long test plateaus," instead of only tuning learning rates or regularization, one should directly check the gradient spectrum and consider low-cost spectral normalization.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Explaining and intervening in grokking via "gradient egalitarianism" is a simple and distinctive idea.  
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Solid evidence in major tasks, with efficiency and more realistic scenarios in the appendix; however, still focused on small-to-medium scale problems.  
- Writing Quality: ⭐⭐⭐⭐☆ The theory-algorithm-experiment chain is clear, and the positioning relative to related work is appropriate.  
- Value: ⭐⭐⭐⭐⭐ Provides a low-intrusion, reusable optimization plug-in for "shortening generalization delay," with high practical potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] On the Convergence Direction of Gradient Descent](on_the_convergence_direction_of_gradient_descent.md)
- [\[ICLR 2026\] On the Convergence Behavior of Preconditioned Gradient Descent Toward the Rich Learning Regime](on_the_convergence_behavior_of_preconditioned_gradient_descent_toward_the_rich_l.md)
- [\[ICLR 2026\] Corner Gradient Descent](corner_gradient_descent.md)
- [\[ICLR 2026\] Fast Data Mixture Optimization via Gradient Descent](fast_data_mixture_optimization_via_gradient_descent.md)
- [\[ICLR 2026\] Composite Optimization with Error Feedback: the Dual Averaging Approach](composite_optimization_with_error_feedback_the_dual_averaging_approach.md)

</div>

<!-- RELATED:END -->

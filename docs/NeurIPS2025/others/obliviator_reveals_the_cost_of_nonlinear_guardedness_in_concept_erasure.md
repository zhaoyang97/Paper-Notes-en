---
title: >-
  [Paper Note] Obliviator Reveals the Cost of Nonlinear Guardedness in Concept Erasure
description: >-
  [NeurIPS 2025][concept erasure] This paper proposes Obliviator — a post-processing concept erasure method based on HSIC minimization in RKHS — that iteratively deforms the feature space through a two-step optimization pr…
tags:
  - "NeurIPS 2025"
  - "concept erasure"
  - "HSIC"
  - "RKHS"
  - "fairness"
  - "nonlinear guardedness"
date: 2026-05-08
content_hash: a4eb39d5c39284e8
---

# Obliviator Reveals the Cost of Nonlinear Guardedness in Concept Erasure

**Conference**: NeurIPS 2025
**arXiv**: [2603.07529](https://arxiv.org/abs/2603.07529)
**Code**: None
**Area**: Fairness / Concept Erasure
**Keywords**: concept erasure, HSIC, RKHS, fairness, nonlinear guardedness

## TL;DR
This paper proposes Obliviator — a post-processing concept erasure method based on HSIC minimization in RKHS — that iteratively deforms the feature space through a two-step optimization procedure. It is the first method to achieve complete guardedness against nonlinear adversaries, while quantifying the utility-erasure trade-off of nonlinear guardedness. Obliviator substantially outperforms existing methods across multiple PLMs and datasets.

## Background & Motivation
**Background**: Pretrained language models (PLMs) widely encode sensitive demographic attributes, leading to biased and unfair predictions. Concept erasure aims to remove such information from representations while preserving task-relevant utility.

**Limitations of Prior Work**:
   - **Linear methods** (INLP, R-LACE, LEACE, SAL) only guard against linear adversaries; nonlinear classifiers can still recover sensitive attributes.
   - **Existing nonlinear methods** (kSAL, KCE, AdS, FaRM, KRaM) attempt to handle nonlinear dependencies but fail to fully capture nonlinear statistical dependencies, remaining vulnerable to nonlinear adversaries.
   - Even costly fine-tuning of PLMs (e.g., AdS, FaRM) yields incomplete erasure.

**Key Challenge**: The two objectives of concept erasure — removing sensitive attributes vs. preserving task utility — are fundamentally competing. Existing methods either fail to fully erase (i.e., are not immune to nonlinear adversaries) or sacrifice too much utility during erasure. More critically, the **dynamics of the utility-erasure trade-off** have never been studied.

**Goal**: (1) Achieve complete guardedness against nonlinear adversaries (i.e., true statistical independence); (2) Reveal and quantify the trade-off dynamics between utility and guardedness throughout the erasure process.

**Key Insight**: Adopting a **functional-analytic perspective**, the paper employs HSIC in RKHS as a measure of nonlinear statistical dependence, formalizes erasure as a cascaded kernel optimization problem, and solves it via an iterative procedure.

**Core Idea**: Use HSIC to measure nonlinear statistical dependence, and iteratively deform the feature space through encoder HSIC minimization combined with RKHS eigendecomposition, achieving complete nonlinear concept erasure while preserving utility.

## Method

### Overall Architecture
Obliviator is a **post-processing, iterative** concept erasure method that alternates between two steps (see Figure 2):
- **Step 1 (Encoder Training)**: Train an encoder to minimize HSIC between the representation and the sensitive attribute, while maximizing HSIC with the task label / original representation to preserve utility.
- **Step 2 (RKHS Disentanglement)**: Solve a constrained eigenvalue problem in RKHS to find directions that maximize the visibility of task-relevant information while remaining orthogonal to the sensitive attribute.
- Each iteration produces an **intermediate representation**, progressively deforming the feature space toward a state where the sensitive attribute is undetectable.

### Key Designs
1. **Cascaded Kernel Problem (Fundamental Distinction from kSAL/KCE)**:

    - kSAL/KCE assume that mapping representations to RKHS and performing linear erasure therein suffices for nonlinear guardedness — but this only guards against linear adversaries within that RKHS, remaining vulnerable to nonlinear adversaries in the same space.
    - Obliviator seeks representations $\varepsilon(X)$ such that the sensitive attribute $S$ remains undetectable **even after a subsequent adversarial feature map** $\phi(\cdot)$. This leads to the cascaded kernel problem: $\inf_\theta \sup_g \sup_f \mathbb{E}[\bar{g}(S) \bar{f}(\varepsilon(\theta; X))]$
    - When HSIC $\to 0$, this is equivalent to $Z_\theta \perp\!\!\perp S$ (true statistical independence).

2. **Step 1: Encoder — Imposing Independence via RKHS**:
   At iteration $i$, the encoder $\varepsilon^i$ is trained with the following loss:
    $\inf_{\theta^i} \frac{1}{n^2} \text{trace}\Big(\mathbf{K}_{z^i} \mathbf{H} (\mathbf{K}_s - \tau_x \mathbf{K}_x - \tau_{x^i} \mathbf{K}_{x^i} - \tau_y \mathbf{K}_y) \mathbf{H}\Big)$
   where $\mathbf{K}_\bullet$ denotes kernel matrices of the corresponding variables and $\tau$ are balancing weights. A key innovation is the use of not only $Y$ to explicitly preserve task information, but also $X$ (the original representation) and $X^i$ (the current iteration input) as implicit proxies, since HSIC aggregates different "visibility patterns" with different weights depending on the reference variable.

3. **Step 2: RKHS Disentanglement — Eigenvalue Problem**:
   A constrained optimization is solved to find RKHS functions that maximize the correlation of $Z^i$ with $(X^i, X, Y)$, subject to zero correlation with $S$:
    $\mathbf{Q}^T \Big(\hat{\mathbf{C}}_{x^i z^i}^T \hat{\mathbf{C}}_{x^i z^i} + \tau_y \hat{\mathbf{C}}_{y z^i}^T \hat{\mathbf{C}}_{y z^i} + \tau_x \hat{\mathbf{C}}_{x z^i}^T \hat{\mathbf{C}}_{x z^i}\Big) \mathbf{Q} \mathbf{v} = \lambda \mathbf{v}$
   where $\mathbf{Q}$ is an orthonormal basis for the null space of $\hat{\mathbf{C}}_{sz^i}$. The top $m$ eigenvectors are selected to project the representation, which serves as input to the encoder in the next iteration.

### Loss & Training
- The coefficients $\tau_x, \tau_{x^i}, \tau_y$ in the multi-objective loss control the balance between utility preservation and erasure.
- Iterative rather than one-shot optimization: each step deforms the feature space incrementally, yielding more utility-preserving erasure.
- Both supervised (using $Y$ labels) and unsupervised (using only $X, X^i$ as proxies) modes are supported.
- Compatible with both frozen representations (post-hoc) and fine-tuned representations.

## Key Experimental Results

### Main Results — BERT Finetuned + Supervised Erasure (Gap Between Baselines and Obliviator)

| Dataset | Task Y | Sensitive Attr. S | Best Baseline Residual S Acc. | Obliviator Residual S Acc. | Gap |
|--------|-------|----------|-------------------|---------------------|------|
| Dial-Mention | Mention | Race | ~62% | ~50% (chance) | **12%** |
| Dial-Sentiment | Sentiment | Race | ~63% | ~50% (chance) | **13%** |
| Bias in Bios | Profession (28 classes) | Gender | ~64% | ~50% (chance) | **14%** |

### Cross-PLM Generalization — Frozen + Supervised on Bias in Bios

| PLM | Embedding Dim. | Obliviator Trade-off | INLP | FaRM | KRaM |
|-----|---------|---------------------|------|------|------|
| BERT | 768 | Full erasure + high utility | Residual leakage | Residual leakage | Residual leakage |
| GPT-2 | 768 | Comparable to BERT | Degraded | Task accuracy collapses | Task accuracy collapses |
| LLaMA-3.2-1B | 2048 | **Better than BERT** | Unchanged | Unchanged | Improved but incomplete |
| DeepSeek-7B | 4096 | **Substantially better than BERT** | — | Unchanged | Accuracy degrades |

### Ablation Study — Supervised vs. Unsupervised × Frozen vs. Fine-tuned

| Setting | Utility Preservation | Full Erasure | Trade-off Severity |
|------|---------|---------|----------------|
| Finetuned + Supervised | ✅ Best | ✅ | Minimal trade-off |
| Frozen + Supervised | ✅ Good | ✅ | Slight trade-off |
| Finetuned + Unsupervised | ✅ Good | ✅ | Moderate trade-off |
| Frozen + Unsupervised | ⚠️ Some degradation | ✅ | Most pronounced trade-off |

### Fairness Metrics — Dial-Sentiment (DP & Gap_rms)

| PLM | Erasure Mode | DP (lower is better) | Gap_rms (lower is better) |
|-----|---------|-------------|-------------------|
| BERT | Supervised | Near 0 | Near 0 |
| BERT | Unsupervised | Low | Low |
| DeepSeek | Supervised | **Lower (better disentanglement)** | **Lower** |
| DeepSeek | Unsupervised | Low | Low |

### Key Findings
- Obliviator is the only method capable of reducing nonlinear adversary accuracy on the sensitive attribute to chance level (true statistical independence).
- Stronger PLMs (DeepSeek > LLaMA > GPT-2 ≈ BERT) yield better-disentangled representations, which Obliviator directly leverages for more utility-preserving erasure.
- Supervised erasure (using $Y$ labels) retains more utility than unsupervised erasure, since $Y$ provides an explicit proxy for task-relevant patterns.
- Skewed data distributions substantially worsen the trade-off (80% skew incurs greater utility loss than the 50% balanced case), revealing the dependence of post-processing erasure methods on data representativeness.
- Even fine-tuning PLMs (e.g., AdS) fails to achieve complete guardedness against nonlinear adversaries — Obliviator achieves complete erasure in a purely post-processing setting.

## Highlights & Insights
- **Solid Theoretical Foundations**: The derivation proceeds coherently from linear covariance to statistical independence in nonlinear RKHS; the guarantee that HSIC = 0 implies independence provides a theoretical upper bound for the method.
- **Iterative Rather Than One-Shot**: The design of progressively deforming the feature space is elegant — it simultaneously produces utility-erasure trade-off curves for analysis and substantively improves erasure quality.
- **Novel RKHS Disentanglement Step**: Solving an eigenvalue problem under null-space constraints elegantly unifies "preventing additional $S$ leakage" and "re-aligning $Y$ information" into a single optimization.
- **Generalization Finding**: The chain — stronger PLM → better disentanglement → better trade-off — is insightful, suggesting that the observed utility-erasure trade-off may serve as a diagnostic indicator of representational quality.

## Limitations & Future Work
- The iterative procedure requires multiple rounds of encoder training and eigendecomposition, incurring substantial computational overhead, especially for the 4096-dimensional DeepSeek embeddings.
- The choice of kernel function (e.g., RBF bandwidth) affects results, but sensitivity analysis is not sufficiently discussed.
- Validation is limited to NLP tasks (text classification, sentiment, occupation); visual or multimodal settings are not explored.
- As a post-processing method that does not modify PLM parameters, utility loss may be unavoidable when task information and sensitive attributes are highly entangled in the original representations.

## Related Work & Insights
- **vs. INLP/R-LACE/LEACE**: These linear methods only guard against linear adversaries; Obliviator achieves complete erasure precisely in the scenarios where they fail entirely (nonlinear probing).
- **vs. kSAL/KCE**: Both are kernel-based, but kSAL performs linear erasure only within RKHS, guarding against linear adversaries in that specific RKHS only; Obliviator addresses this fundamental limitation through the cascaded kernel formalization and iterative optimization.
- **vs. AdS/FaRM**: These methods require fine-tuning PLMs at higher computational cost, yet still fail to achieve complete nonlinear erasure; Obliviator achieves more thorough erasure as a post-processing method.
- **vs. KRaM**: Also a post-processing kernel method, but KRaM's rate-distortion maximization framework likewise fails to achieve complete erasure and causes task accuracy collapse on GPT-2.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — The cascaded kernel formalization exposes the fundamental flaw of methods such as kSAL; the two-step iterative framework is elegant and original.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 4 PLMs × 3 datasets × 4 settings × trade-off curves × fairness metrics × skew analysis; exceptionally comprehensive.
- Writing Quality: ⭐⭐⭐⭐ — Mathematical derivations are clear, but the dense notation presents a non-trivial barrier on first reading.
- Value: ⭐⭐⭐⭐⭐ — The first method to achieve complete nonlinear concept erasure; the trade-off analysis framework provides an important benchmark for future work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] FACE: Faithful Automatic Concept Extraction](face_faithful_automatic_concept_extraction.md)
- [\[NeurIPS 2025\] The Cost of Robustness: Tighter Bounds on Parameter Complexity for Robust Memorization in ReLU Nets](the_cost_of_robustness_tighter_bounds_on_parameter_complexity_for_robust_memoriz.md)
- [\[AAAI 2026\] On the Variability of Concept Activation Vectors](../../AAAI2026/others/on_the_variability_of_concept_activation_vectors.md)
- [\[AAAI 2026\] Cost-Free Neutrality for the River Method](../../AAAI2026/others/cost-free_neutrality_for_the_river_method.md)
- [\[ICLR 2026\] Beyond Linearity in Attention Projections: The Case for Nonlinear Queries](../../ICLR2026/others/beyond_linearity_in_attention_projections_the_case_for_nonlinear_queries.md)

</div>

<!-- RELATED:END -->

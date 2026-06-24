---
title: >-
  [Paper Note] Statistical Consistency and Generalization of Contrastive Representation Learning
description: >-
  [ICML 2026][Self-Supervised Learning][Contrastive Learning] This paper establishes the Fisher/statistical consistency for Contrastive Representation Learning (CRL) for the first time, proving that minimizing upstream contrastive loss is equivalent to optimizing downstream AUC-type retrieval performance. It provides refined generalization bounds of $O(1/m+1/\sqrt n)$ (supervised) and $O(1/\sqrt m+1/\sqrt n)$ (self-supervised) depending on the number of positive samples $n$ and…
tags:
  - "ICML 2026"
  - "Self-Supervised Learning"
  - "Contrastive Learning"
  - "Statistical Consistency"
  - "Calibration Inequalities"
  - "Generalization Bounds"
  - "Number of Negative Samples"
date: 2026-05-08
content_hash: 4014c67c718608e9
---

# Statistical Consistency and Generalization of Contrastive Representation Learning

**Conference**: ICML 2026  
**arXiv**: [2605.02116](https://arxiv.org/abs/2605.02116)  
**Code**: None  
**Area**: Self-Supervised / Representation Learning / Learning Theory  
**Keywords**: Contrastive Learning, Statistical Consistency, Calibration Inequalities, Generalization Bounds, Number of Negative Samples

## TL;DR
This paper establishes the Fisher/statistical consistency for Contrastive Representation Learning (CRL) for the first time, proving that minimizing upstream contrastive loss is equivalent to optimizing downstream AUC-type retrieval performance. It provides refined generalization bounds of $O(1/m+1/\sqrt n)$ (supervised) and $O(1/\sqrt m+1/\sqrt n)$ (self-supervised) depending on the number of positive samples $n$ and negative samples $m$, theoretically explaining why CLIP/SimCLR continue to benefit from tens of thousands of negative samples.

## Background & Motivation

**Background**: The core training objectives for foundation models like CLIP, SimCLR, and MoCo are contrastive losses (Eq. 1). Formally, these are InfoNCE / log-sum-exp pairwise ranking losses that learn transferable representations by pushing the score $s_w(x,y)$ of positive pairs $(x,y)$ higher and the scores of negative pairs $(x,y')$ lower.

**Limitations of Prior Work**: Existing theories suffer from three contradictory gaps: (i) They only prove a "surrogate gap"—where low contrastive risk implies low supervised loss under a linear probe—but **fail to prove statistical consistency** (whether the optimal solution for contrastive loss is optimal for the downstream task as the sample size approaches infinity); (ii) existing generalization bounds (e.g., Saunshi et al.) worsen monotonically with the number of negative samples $m$, appearing as $O(m/\sqrt n)$ or $O(\log m/\sqrt n)$, which contradicts empirical evidence that SimCLR (8192) and CLIP (32768) benefit from large $m$; (iii) there is a lack of theory quantifying downstream performance from a retrieval perspective, which is the core application of CLIP.

**Key Challenge**: Contrastive loss is essentially a **pairwise ranking** loss. Previous analyses forced it into a classification framework, which loses the geometric structure of ranking and causes $m$ to appear in the numerator of the bounds.

**Goal**: To be achieved in two steps: (a) evaluate downstream performance using an AUC-type ranking criterion $\mathcal E(s)$, prove Fisher consistency for the contrastive loss, and provide a calibration inequality $\mathcal E^*-\mathcal E(s)\lesssim\sqrt{L(s)-L^*}$; (b) re-decompose the generalization error so that $m$ appears in the **denominator** rather than the numerator.

**Key Insight**: The inner log-mean-exp of the contrastive empirical risk $\widehat L_S(s_w)$ can be rewritten as a **strongly convex minimization problem** involving an auxiliary variable $\mu$ (Lemma 4.2). This allows the inner error to be interpreted as a generalization problem of ERM, yielding $O(1/m)$ instead of $O(1/\sqrt m)$ via algorithmic stability.

**Core Idea**: By replacing the surrogate-gap with an AUC-type retrieval criterion and rewriting the log-sum-exp loss into an OCE (optimized certainty equivalent) form, a generalization bound of $O(1/m+1/\sqrt n)$ is derived using stability arguments. This simultaneously addresses consistency, benefits of large negative samples, and retrieval semantics.

## Method

### Overall Architecture
This is a purely theoretical work with two main modules:
1.  **Consistency Module**: Introduces an AUC-type downstream evaluation $\mathcal E(s)=\Pr[s(x,y)>s(x,y')]+\tfrac12\Pr[s(x,y)=s(x,y')]$, characterizing the probability that "positive pairs are ranked above negative pairs." It proves that the population minimizer of the contrastive loss satisfies $s^*(x,y)=\tau\log\frac{p_x^+(y)}{p_x^-(y)}+g(x)$ (Lemma 3.2), which also maximizes $\mathcal E(s)$ (Lemma 3.3), thus establishing Fisher consistency. A calibration inequality $\mathcal E^*-\mathcal E(s)\le\sqrt{2/\tau\,(L(s)-L^*)}$ is then derived using a monotone chain (Thm 3.4).
2.  **Generalization Module**: Decomposes the generalization gap along the **outer (positive pair) + inner (negative pair)** composite structure of the contrastive loss. The outer part yields $O(1/\sqrt n)$ via Rademacher complexity. The inner part yields $O(1/m)$ for SCRL and $O(1/\sqrt m)$ for SSCRL via OCE rewriting and stability, forming the total bound.

### Key Designs

**1. AUC-type Downstream Criterion + Fisher Consistency Proof**: Previous surrogate-gap results compared contrastive risk with supervised risk after a linear probe, failing to guarantee convergence to the oracle as sample size grows because the evaluation target (classification) and training target (ranking) follow different geometries. This work adopts $\mathcal E(s)=\Pr[s(x,y)>s(x,y')]+\tfrac12\Pr[s(x,y)=s(x,y')]$, an AUC-type pairwise ranking measure perfectly aligned with the contrastive structure. The proof demonstrates that $s^*(x,y)$, the pointwise optimizer of $L(s)$, maximizes the likelihood ratio $p^+(y)/p^-(y)$, which is the optimal ranking condition for AUC. This closes the consistency loop by establishing $L(s_n)\to L^*\Rightarrow\mathcal E(s_n)\to\mathcal E^*$ (Thm 3.1).

**2. OCE Rewriting + Algorithmic Stability for $O(1/m)$ Inner Bound**: The term most sensitive to $m$ in generalization error is the inner $\tau\log\tfrac1m\sum_j\exp(\Delta_w/\tau)$. Previous works used Hoeffding or uniform convergence, which forced $m$ into the numerator when taking the supremum over parameters, resulting in the counter-intuitive $O(m/\sqrt n)$. This paper rewrites it as an Optimized Certainty Equivalent (OCE) form—introducing an auxiliary scalar $\mu\in[-2B,2B]$ to transform the inner mean into a **strongly convex minimization** (Lemma 4.2): $\widehat L_S(s_w)=-\tau+\tfrac1n\sum_i\min_{|\mu_i|\le 2B}\bigl[\tfrac{\tau}{m}\sum_j\exp((\Delta-\mu_i)/\tau)+\mu_i\bigr]$. Strong convexity allows the inner error to be bounded by $O(1/m)$ using algorithmic stability (Bousquet-Elisseeff) for supervised CRL (SCRL). In self-supervised CRL (SSCRL), since $m$ negative samples are shared across anchors, the decoupled ERM structure is lost, reverting to $O(1/\sqrt m)$. The OCE form moves $m$ to the denominator.

**3. Inner / Outer Decomposition + Rademacher Control for Outer Term**: The generalization gap is decomposed as $L(s_w)-\widehat L(s_w)\le\underbrace{L(s_w)-\mathbb E\widehat L(s_w)}_{\text{inner}}+\underbrace{\mathbb E\widehat L(s_w)-\widehat L(s_w)}_{\text{outer}}$, decoupling the "negative sampling (inner)" and "anchor sampling (outer)" perturbations. The outer term is bounded by $O(\sqrt{\log(1/\delta)/n})$ using Rademacher complexity $\mathcal R_S(\mathcal K)$ of an aggregation function, independent of $m$. The main theorem (Thm 4.5) gives $\sup_w|L_S(s_w)-\widehat L_S(s_w)|=O(1/m+\sqrt{\log(1/\delta)/n})$, explicitly showing the trade-off: increasing $m$ shrinks the inner term, while increasing $n$ shrinks the outer term.

### Loss & Training
The paper analyzes existing contrastive objectives: Supervised CRL ($L_S(s_w)$, Eq. 5, where each anchor has $m$ independent negative samples) and Self-Supervised CRL ($L_{SS}(s_w)$, Eq. 8 / GCL, where $m$ negative samples are shared within a batch). Both share the log-sum-exp InfoNCE form, but the sampling difference leads the inner bound to tighten from $O(1/\sqrt m)$ (SSCRL) to $O(1/m)$ (SCRL).

## Key Experimental Results

### Main Results
The theoretical scaling predictions are validated using large-scale vision-language models (CLIP):

| Dimension | Theoretical Prediction | Empirical Verification |
| :--- | :--- | :--- |
| Negative samples $m$ | Inner error decays as $1/m$ (SCRL) / $1/\sqrt m$ (SSCRL) | Increasing batch size (negatives) monotonically increases zero-shot R@1; marginal gains match $1/m$ curves. |
| Anchor count $n$ | Outer error decays as $1/\sqrt n$ | Increasing the number of positive pairs (with fixed $m$) improves retrieval performance following $1/\sqrt n$. |
| $m$ vs $n$ trade-off | Additive relationship; neither is replaceable | With fixed total budget $n \cdot m$, neither extreme (small $m$ or small $n$) is optimal. |
| Calibration | $\mathcal E^*-\mathcal E(s)\le\sqrt{2(L-L^*)/\tau}$ | The gap in downstream retrieval AUC follows a $\sqrt{\cdot}$ relationship with the upstream loss gap. |

### Ablation Study

| Configuration | Key Finding | Description |
| :--- | :--- | :--- |
| Increasing $m$ (SCRL) | Steeper retrieval AUC improvement | Consistent with $O(1/m)$; supervised scenarios benefit more significantly from large negative sets. |
| Increasing $m$ (SSCRL / CLIP) | Flatter improvement margin | Consistent with $O(1/\sqrt m)$; shared negatives result in diminished returns. |
| Increasing $n$ only | Gains scale consistently at $1/\sqrt n$ | Matches theory for the outer term, explaining the $n$-side of CLIP scaling laws. |

### Key Findings
- The $O(m/\sqrt n)$ dependence in prior theory is an **artifact of analytical techniques**, not the inherent difficulty of the problem. OCE rewriting moves $m$ to the denominator.
- SCRL and SSCRL differ fundamentally in inner bound rates ($1/m$ vs $1/\sqrt m$) due to negative sample sharing, providing a criterion for whether and when to use supervised negatives.
- The calibration inequality is $\sqrt{\cdot}$ rather than linear, explaining why small upstream loss reductions in the late stages of pre-training can still yield significant downstream improvements.

## Highlights & Insights
- **Theoretical Elegance**: Completes the loop of consistency, calibration, and generalization for modern vision-language model losses without needing "theory-friendly" proxies.
- **OCE Rewriting as a Tool**: Converting composite logs of sums into strongly convex ERM problems is a versatile technique applicable to other conditional stochastic optimization problems (DRO, learning-to-rank, softmax in attention).
- **Explaining the scaling of $m$**: This provides a quantitative theoretical guarantee for engineering heuristics: increasing batch sizes in CLIP/SimCLR is principled, not just luck.
- **AUC Perspective**: Reinforces that contrastive learning is fundamentally about ranking, an insight applicable to dense retrieval and recommendation systems.

## Limitations & Future Work
- Assumption 4.1 requires the scoring function to be in inner-product form with bounded spectral norms; whether transformer encoders satisfy these bounds in practice remains to be verified.
- Fisher consistency is proved over the space of all measurable functions; the approximation error for specific neural network hypothesis classes is not yet characterized.
- The study focuses on scaling trends rather than providing prescriptive formulas for the exact optimal $m$ and $n$.
- Hard-negative mining, which violates the i.i.d. negative sampling assumption, is not covered.

## Related Work & Insights
- **vs Saunshi et al. 2019 / Lei et al. 2023**: Their surrogate-gap uses linear classification; their $O(m/\sqrt n)$ bounds contradict practice. Ours uses AUC ranking and OCE to solve consistency and large-$m$ benefits.
- **vs HaoChen et al. 2021 (Spectral Methods)**: Spectral views explain representation geometry but lack sample complexity analysis; this work provides the statistical learning theory perspective.
- **vs Wang & Isola 2020 (Alignment & Uniformity)**: They characterize geometric properties; this work provides the corresponding statistical convergence rates.
- This work supports engineering practices like large batch sizes and query-side hard negative mining in production systems (retrieval/recommendation).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First Fisher consistency + calibration + $O(1/m+1/\sqrt n)$ proof for CRL; OCE rewriting is a significant innovation.
- Experimental Thoroughness: ⭐⭐⭐ Primarily validation-oriented; does not introduce new algorithms or broad model comparisons.
- Writing Quality: ⭐⭐⭐⭐ Very clear logical chain (Consistency → Calibration → Generalization).
- Value: ⭐⭐⭐⭐⭐ Essential reading for teams training foundation models with InfoNCE; explains the batch size scaling law.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] A Refined Generalization Analysis for Extreme Multi-class Supervised Contrastive Representation Learning](a_refined_generalization_analysis_for_extreme_multi-class_supervised_contrastive.md)
- [\[ICML 2025\] Generalization Analysis for Supervised Contrastive Representation Learning under Non-IID Settings](../../ICML2025/self_supervised/generalization_analysis_for_supervised_contrastive_representation_learning_under.md)
- [\[ICML 2026\] Inconsistency-Aware Minimization: Improving Generalization with Unlabeled Data](inconsistency-aware_minimization_improving_generalization_with_unlabeled_data.md)
- [\[ICML 2026\] InfoAtlas: A Foundation Model for Zero-Shot Statistical Dependence Estimation](infoatlas_a_foundation_model_for_zero-shot_statistical_dependence_estimate.md)
- [\[ICML 2026\] Data Augmentation of Contrastive Learning is Estimating Positive-incentive Noise](data_augmentation_of_contrastive_learning_is_estimating_positive-incentive_noise.md)

</div>

<!-- RELATED:END -->

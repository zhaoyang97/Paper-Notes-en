---
title: >-
  [Paper Note] Statistical Consistency and Generalization of Contrastive Representation Learning
description: >-
  [ICML 2026][Self-Supervised Learning][Contrastive Learning] This paper establishes the Fisher/statistical consistency for Contrastive Representation Learning (CRL)…
tags:
  - "ICML 2026"
  - "Self-Supervised Learning"
  - "Contrastive Learning"
  - "Statistical Consistency"
  - "Calibration Inequality"
  - "Generalization Bound"
  - "Number of Negatives"
date: 2026-05-08
content_hash: dfe50ac5de22c3a5
---

# Statistical Consistency and Generalization of Contrastive Representation Learning

**Conference**: ICML 2026  
**arXiv**: [2605.02116](https://arxiv.org/abs/2605.02116)  
**Code**: None  
**Area**: Self-supervised / Representation Learning / Learning Theory  
**Keywords**: Contrastive Learning, Statistical Consistency, Calibration Inequality, Generalization Bound, Number of Negatives

## TL;DR
This paper establishes the Fisher/statistical consistency for Contrastive Representation Learning (CRL), demonstrating that minimizing upstream contrastive loss is equivalent to optimizing downstream AUC-type retrieval performance. It provides refined generalization bounds of $O(1/m+1/\sqrt n)$ for supervised and $O(1/\sqrt m+1/\sqrt n)$ for self-supervised settings, providing the first theoretical explanation for the phenomenon where CLIP and SimCLR consistently benefit from utilizing tens of thousands of negative samples.

## Background & Motivation

**Background**: Core training objectives for foundation models like CLIP, SimCLR, and MoCo are contrastive losses (Eq. 1). Formally, these are InfoNCE / log-sum-exp pairwise ranking losses that learn transferable representations by pulling positive pairs $(x,y)$ closer while pushing negative pairs $(x,y')$ apart.

**Limitations of Prior Work**: Existing theories suffer from three contradictory gaps: (i) They only prove a "surrogate gap"—where low contrastive risk implies low supervised loss under a linear probe—but **fail to prove statistical consistency** (whether the optimal solution of contrastive loss converges to the downstream optimum as the sample size goes to infinity); (ii) Known generalization bounds (e.g., Saunshi et al.) monotonically worsen with the number of negatives $m$, scaling as $O(m/\sqrt n)$ or $O(\log m/\sqrt n)$, which contradicts empirical evidence where SimCLR (8192 negatives) and CLIP (32768 negatives) gain performance; (iii) Almost no theory quantifies CRL downstream performance from a retrieval perspective, which is the primary application of models like CLIP.

**Key Challenge**: Contrastive loss is essentially a **pairwise ranking** loss, but previous analyses forced it into a classification framework. This approach discards the geometric structure of ranking and results in $m$ appearing in the numerator of the complexity bounds.

**Goal**: The paper pursues a two-step objective: (a) Evaluate downstream performance using an AUC-type ranking criterion $\mathcal E(s)$, prove Fisher consistency of the contrastive loss regarding this criterion, and derive a calibration inequality $\mathcal E^*-\mathcal E(s)\lesssim\sqrt{L(s)-L^*}$; (b) Re-decompose the generalization error to ensure $m$ stays in the **denominator**.

**Key Insight**: Rewrite the inner log-mean-exp of the empirical contrastive risk $\widehat L_S(s_w)$ as a **strongly convex minimization problem** regarding an auxiliary variable $\mu$ (Lemma 4.2). This allows the inner error to be interpreted as an ERM generalization problem, leveraging algorithmic stability to obtain a rate of $O(1/m)$ instead of $O(1/\sqrt m)$.

**Core Idea**: Replace the surrogate-gap with an AUC-type retrieval criterion and rewrite the log-sum-exp loss into an Optimized Certainty Equivalent (OCE) form. Use stability arguments to provide a generalization bound of $O(1/m+1/\sqrt n)$, simultaneously addressing consistency, benefits of large negative samples, and retrieval semantics.

## Method

### Overall Architecture
The paper is a purely theoretical work, structured into two major modules:
1. **Consistency Module**: Introduces an AUC-type downstream evaluation $\mathcal E(s)=\Pr[s(x,y)>s(x,y')]+\tfrac12\Pr[s(x,y)=s(x,y')]$ to characterize the probability of "positives ranked above negatives." It proves that the population minimizer of the contrastive loss $s^*(x,y)=\tau\log\frac{p_x^+(y)}{p_x^-(y)}+g(x)$ (Lemma 3.2) also maximizes $\mathcal E(s)$ (Lemma 3.3), establishing Fisher consistency. A calibration inequality $\mathcal E^*-\mathcal E(s)\le\sqrt{2/\tau\,(L(s)-L^*)}$ is derived using monotonic chains (Thm 3.4).
2. **Generalization Module**: Decomposes the generalization gap along the **outer (positive pairs) + inner (negative pairs)** composite structure of the contrastive loss. The outer part uses Rademacher complexity to obtain $O(1/\sqrt n)$, while the inner part uses OCE rewriting and stability to yield $O(1/m)$ for SCRL and $O(1/\sqrt m)$ for SSCRL.

### Key Designs

1. **AUC-type Downstream Criterion + Fisher Consistency Proof**:
    - **Function**: Replaces "linear classification error" with the pairwise ranking criterion $\mathcal E(s)$, ensuring the evaluation metric is consistent with the pairwise geometry of the contrastive loss.
    - **Mechanism**: Derives the pointwise solution $s^*=\tau\log(p^+/p^-)+g(x)$ for $L(s)$ over the space of all measurable functions. Due to the monotonicity of $\log$, $s^*(x,y)>s^*(x,y')$ if and only if $p^+(y)/p^-(y)>p^+(y')/p^-(y')$, which is the optimal AUC ranking condition. Thus, $L(s_n)\to L^*$ implies $\mathcal E(s_n)\to\mathcal E^*$.
    - **Design Motivation**: Previous surrogate-gap results only compared contrastive risk with supervised risk after a linear probe, failing to guarantee convergence to the oracle as sample size increases. Using AUC as a natural downstream target for "ranking" closes this loop.

2. **OCE Rewriting + Algorithmic Stability for $O(1/m)$ Inner Bound**:
    - **Function**: Rewrites the inner term $\tau\log\tfrac1m\sum_j\exp(\Delta_w/\tau)$, which is highly sensitive to the number of negatives, as a **strongly convex minimization** over an auxiliary scalar $\mu\in[-2B,2B]$ (Lemma 4.2): $\widehat L_S(s_w)=-\tau+\tfrac1n\sum_i\min_{|\mu_i|\le 2B}\bigl[\tfrac{\tau}{m}\sum_j\exp((\Delta-\mu_i)/\tau)+\mu_i\bigr]$.
    - **Mechanism**: After rewriting, the inner error adopts the form $f(w,x,y)-\hat f(w,x,y)$, where $\hat f$ is the ERM using $m$ negative samples. Given strong convexity, the stability of the ERM solution relative to the optimum is $O(1/m)$ (Bousquet-Elisseeff). Thus, the inner bound for supervised CRL (SCRL) is $\sup_w|L_S(s_w)-\mathbb E\widehat L_S(s_w)|=O(1/m)$ (Lemma 4.3). For self-supervised CRL, where negatives are shared across anchors, uniform convergence yields $O(1/\sqrt m)$.
    - **Design Motivation**: Directly applying Hoeffding or uniform convergence to $\log$-sum-$\exp$ inevitably puts $m$ in the numerator due to the supremum operation. OCE allows $1/m$ fast rates by transforming the sum into a decomposable strongly convex problem.

3. **Inner / Outer Decomposition + Rademacher Control for the Outer Part**:
    - **Function**: Decomposes the total generalization gap as $L(s_w)-\widehat L(s_w)\le\underbrace{L(s_w)-\mathbb E\widehat L(s_w)}_{\text{inner}}+\underbrace{\mathbb E\widehat L(s_w)-\widehat L(s_w)}_{\text{outer}}$, decoupling the independent perturbations from "negative sampling" and "anchor sampling."
    - **Mechanism**: Defines an aggregate function $k_w(x,y,y'_1,\dots,y'_m)=\tau\log\tfrac1m\sum_j\exp(\Delta_w/\tau)$ and uses the Rademacher complexity $\mathcal R_S(\mathcal K)$ of the deep network to bound the outer part by $O(\sqrt{\log(1/\delta)/n})$, making it independent of $m$. Combining both parts yields the main theorem (Thm 4.5): $\sup_w|L_S(s_w)-\widehat L_S(s_w)|=O(1/m+\sqrt{\log(1/\delta)/n})$.
    - **Design Motivation**: This decomposition reveals an explicit trade-off between $m$ and $n$. With a fixed total budget $N=n\cdot m$, increasing $m$ causes the inner term to contract, while increasing $n$ shrinks the outer term at a rate of $1/\sqrt n$, justifying the engineering practice of using all non-matching captions in a batch as negatives.

### Loss & Training
The paper does not introduce new losses but analyzes two existing contrastive objectives: the supervised version $L_S(s_w)$ (Eq. 5, where each anchor independently samples $m$ negatives) and the self-supervised version $L_{SS}(s_w)$ (Eq. 8 / GCL, where $m$ negatives are shared across anchors, as implemented in CLIP/SimCLR). Both share the log-sum-exp InfoNCE form, differing only in sampling coupling, which leads to the inner bound tightening from $O(1/\sqrt m)$ (SSCRL) to $O(1/m)$ (SCRL).

## Key Experimental Results

### Main Results
Using CLIP / Vision-Language Models as the medium, the paper validates two scaling behaviors predicted by the theory on large-scale data:

| Dimension | Theoretical Prediction | Empirical Validation |
|-----------|------------------------|----------------------|
| Negatives $m$ | Inner error decays as $1/m$ (SCRL) / $1/\sqrt m$ (SSCRL) | Increasing in-batch negatives causes downstream zero-shot R@1 to rise monotonically; marginal gains align with $1/m$ curves. |
| Anchors $n$ | Outer error decays as $1/\sqrt n$ | Fixing $m$ while increasing positive pairs improves retrieval performance in a $1/\sqrt n$-like manner. |
| $m$ vs $n$ trade-off | Additive relationship; one cannot substitute the other | Under fixed total samples $N=n\cdot m$, extreme configurations (small $m$ / large $n$ or vice versa) underperform compared to balanced setups. |
| Calibration | $\mathcal E^*-\mathcal E(s)\le\sqrt{2(L-L^*)/\tau}$ | Across different training steps, the measured downstream retrieval AUC gap follows a $\sqrt{\cdot}$ relationship with the upstream loss gap. |

### Ablation Study

| Configuration | Key Finding | Description |
|---------------|-------------|-------------|
| Increasing $m$ (SCRL) | Steeper retrieval AUC improvement | Consistent with $O(1/m)$, supervised scenarios show more significant benefits from large negative counts. |
| Increasing $m$ (SSCRL / CLIP) | Flatter improvement margin | Consistent with $O(1/\sqrt m)$; shared negatives lead to diminished marginal benefits compared to SCRL. |
| Increasing $n$ only | Gains reduce at $1/\sqrt n$ rate | Consistent with the outer term theory, explaining the $n$-side of CLIP scaling laws. |

### Key Findings
- The $O(m/\sqrt n)$ dependency in previous theories was an **artifact of analytical techniques** rather than problem difficulty. Rewriting log-sum-exp into OCE form moves $m$ from the numerator to the denominator.
- The fundamental difference in fast rates between SCRL ($1/m$) and SSCRL ($1/\sqrt m$) stems from whether negatives are shared; this provides a quantifiable criterion for whether "supervised negatives" are worth the cost.
- The calibration inequality is $\sqrt{\cdot}$ rather than linear, explaining why even minor upstream loss decreases in the late stages of large-scale pre-training can lead to significant downstream retrieval improvements.

## Highlights & Insights
- **Theoretical Closure**: Achieves a complete loop of consistency, calibration, and generalization for modern VLM losses without requiring "theory-friendly" proxy objectives.
- **OCE Rewriting as a Tool**: Converting the composite "inner average" loss ($\log\tfrac1m\sum\exp$) into a strongly convex ERM is a generalizable technique for other conditional stochastic optimization problems like DRO, learning-to-rank, and softmax pooling in attention.
- **Explaining $m\uparrow$ Gains**: This is the first work to quantitatively explain the empirical success of large batches in CLIP/SimCLR, providing theoretical assurance that increasing batch sizes is beneficial.
- **AUC Retrieval Perspective**: Emphasizes that contrastive learning is fundamentally ranking rather than classification, an insight applicable to any scenario using InfoNCE for retrieval (dense retrieval, recommendation systems).

## Limitations & Future Work
- Assumption 4.1 requires the scoring function to be an inner product with bounded spectral norms. Whether actual Transformer encoders satisfy this spectral bound remains to be verified, especially as the bound may accumulate exponentially with depth.
- Consistency is proven over "all measurable functions"; the approximation error between this and specific neural hypothesis classes remains uncharacterized.
- Experiments focus on verifying scaling trends rather than providing actionable formulas for optimal $m,n$ allocation under specific budgets.
- Does not cover hard-negative mining, which breaks i.i.d. sampling assumptions; this is a promising direction for future research.

## Related Work & Insights
- **vs. Saunshi et al. 2019 / Lei et al. 2023**: Their surrogate-gap models downstream performance as linear classification, leading to $O(m/\sqrt n)$ bounds that contradict practice. This paper resolves this via AUC ranking and OCE rewriting.
- **vs. HaoChen et al. 2021 (Spectral Methods)**: Spectral perspectives explain representation geometry but do not provide explicit sample complexity. This work completes the sample-size perspective within the SLT framework.
- **vs. Wang & Isola 2020 (Alignment & Uniformity)**: While they characterize contrastive geometry, this paper provides complementary evidence via fast rates in statistical learning theory.
- For production systems like dense retrieval or recommendation, the conclusions support the use of larger in-batch negatives and query-side hard negative mining.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (Fisher consistency + calibration + $O(1/m+1/\sqrt n)$ proof; OCE rewriting is a sharp methodological innovation.)
- Experimental Thoroughness: ⭐⭐⭐ (Primary validation; no new algorithms or cross-model ablations.)
- Writing Quality: ⭐⭐⭐⭐ (Clear logical chain from consistency to calibration to generalization; decomposition of inner/outer terms is intuitive.)
- Value: ⭐⭐⭐⭐⭐ (Essential theoretical reference for teams training foundation models with InfoNCE; directly explains batch size scaling laws.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] A Refined Generalization Analysis for Extreme Multi-class Supervised Contrastive Representation Learning](a_refined_generalization_analysis_for_extreme_multi-class_supervised_contrastive.md)
- [\[ICML 2026\] Inconsistency-Aware Minimization: Improving Generalization with Unlabeled Data](inconsistency-aware_minimization_improving_generalization_with_unlabeled_data.md)
- [\[ICML 2026\] InfoAtlas: A Foundation Model for Zero-Shot Statistical Dependence Estimation](infoatlas_a_foundation_model_for_zero-shot_statistical_dependence_estimate.md)
- [\[ICML 2026\] Data Augmentation of Contrastive Learning is Estimating Positive-incentive Noise](data_augmentation_of_contrastive_learning_is_estimating_positive-incentive_noise.md)
- [\[CVPR 2026\] UniGeoCLIP: Unified Geospatial Contrastive Learning](../../CVPR2026/self_supervised/unigeoclip_geospatial_contrastive.md)

</div>

<!-- RELATED:END -->

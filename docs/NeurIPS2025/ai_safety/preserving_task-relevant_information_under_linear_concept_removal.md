---
title: >-
  [Paper Note] Preserving Task-Relevant Information Under Linear Concept Removal
description: >-
  [NeurIPS 2025][AI Safety][concept erasure] SPLINCE constructs an oblique projection that simultaneously guarantees linear guardedness (i.e.…
tags:
  - "NeurIPS 2025"
  - "AI Safety"
  - "concept erasure"
  - "fair representation learning"
  - "oblique projection"
  - "linear guardedness"
  - "bias removal"
date: 2026-05-08
content_hash: d242f301b9a2535b
---

# Preserving Task-Relevant Information Under Linear Concept Removal

**Conference**: NeurIPS 2025
**arXiv**: [2506.10703](https://arxiv.org/abs/2506.10703)
**Code**: [https://github.com/](https://github.com/) (available, link provided in paper footnote)
**Area**: AI Safety / Fairness
**Keywords**: concept erasure, fair representation learning, oblique projection, linear guardedness, bias removal

## TL;DR
SPLINCE constructs an oblique projection that simultaneously guarantees linear guardedness (i.e., sensitive attributes cannot be predicted by any linear classifier) and exactly preserves the covariance between representations and task labels, thereby resolving the problem of existing concept erasure methods inadvertently removing task-relevant information alongside sensitive concepts.

## Background & Motivation

- **Background**: Deep neural network (DNN) embeddings encode not only task-relevant information but also undesired concepts (e.g., gender, race), leading to biased model predictions. For instance, a classifier for screening job applications may exhibit gender bias due to encoded gender information.

- **Limitations of Prior Work**: Existing post-hoc concept erasure methods (INLP, RLACE, LEACE, SAL, etc.) apply linear projections to render sensitive concepts unpredictable by linear classifiers—a property known as *linear guardedness*. However, their critical limitation is that **removing sensitive concepts simultaneously damages task-relevant information**.

- **Key Challenge**: When sensitive attributes correlate with the target task (e.g., uneven gender distribution across occupations), removing sensitive signals inevitably degrades task performance. Existing methods either preserve linear guardedness at the cost of task information (LEACE), or protect task information at the cost of guardedness.

- **Core Idea**: Can one simultaneously erase sensitive concepts and **exactly preserve** the covariance between representations and task labels? The answer is affirmative—via oblique projection, which places the covariance directions of sensitive concepts in the projection kernel while retaining the covariance directions of the target task in the projection range.

## Method

### Overall Architecture

SPLINCE is a linear-algebraic method that computes a projection matrix $\mathbf{P}^*_{SPLINCE}$ and applies it to the final-layer embeddings of a DNN. Its core formulation solves an optimization problem subject to two constraints.

### Key Designs

1. **Dual-Constraint Optimization**:

    - Given representations $\bm{x}$, sensitive attributes $\bm{z}$, and task labels $\bm{y}$, the goal is to find a projection $\mathbf{P}$ satisfying:
        - **Kernel Constraint**: $\mathbf{P}\Sigma_{\bm{x},\bm{z}} = \mathbf{0}$ (linear guardedness: zero covariance between projected representations and sensitive attributes)
        - **Range Constraint**: $\mathbf{P}\Sigma_{\bm{x},\bm{y}} = \Sigma_{\bm{x},\bm{y}}$ (exact preservation of covariance between representations and task labels)
        - **Distortion Minimization**: $\min_{\mathbf{P}} \mathbb{E}[\|\mathbf{P}\bm{x} - \bm{x}\|^2_{\mathbf{M}}]$
    - Closed-form solution: $\mathbf{P}^*_{SPLINCE} = \mathbf{W}^+ \mathbf{V}(\mathbf{U}^T\mathbf{V})^{-1}\mathbf{U}^T\mathbf{W}$
    - where $\mathbf{W}$ is a whitening matrix and $\mathbf{U}$, $\mathbf{V}$ are orthonormal bases for specific subspaces.

2. **Equivalence Theorem (Theorem 3.2)**:

    - A key theoretical result: all projections sharing the same kernel (i.e., erasing the same subspace) yield **identical predictions** after **retraining a linear classifier without regularization**.
    - This implies that SPLINCE and LEACE are equivalent in the no-regularization setting—the choice of range does not affect final predictions.
    - However, the range choice **does matter** in two practically important settings: (1) retraining with regularization; and (2) not retraining the final layer (e.g., language model interventions).

3. **Applicability Conditions**:

    - Prerequisite assumption: $\mathcal{U}^\perp \cap \text{colsp}(\mathbf{W}\Sigma_{\bm{x},\bm{y}}) = \{\mathbf{0}\}$
    - That is, after whitening, the covariance directions of sensitive attributes and task labels do not fully overlap.
    - For binary variables, this is equivalent to requiring that $\text{Cov}(\bm{x},\bm{z})$ and $\text{Cov}(\bm{x},\bm{y})$ are linearly independent (not proportional).

### Loss & Training
- SPLINCE requires no training—it is a closed-form projection computation.
- Pipeline: fine-tune the DNN to obtain embeddings → compute the SPLINCE projection matrix → project embeddings → retrain/apply a linear classifier.
- Computing SPLINCE requires only first- and second-order statistics (means and covariance matrices) of the data.

## Key Experimental Results

### Main Results

**Classification Task (Bias in Bios)** — Task: occupation prediction, Removal: gender

| Method | High-Correlation Setting (p=0.9) Accuracy | Worst-Group Acc | Notes |
|------|------|----------|------|
| LEACE | ~70% | ~55% | Task information lost |
| SAL | ~68% | ~52% | Greater loss |
| SPLINCE | ~78% | ~72% | Significant advantage |

**Language Model (Llama)** — Removing stereotypical associations while preserving factual gender information

| Model / Projection | exp(β_stereo) | exp(β_fact) | Notes |
|------|------|----------|------|
| Llama 2 7B (original) | 3.59 | 15.71 | Relies on stereotypes |
| +SAL | 0.80 | 5.90 | Large loss of factual information |
| +LEACE | 0.85 | 12.14 | Partial loss of factual information |
| +SPLINCE | 0.79 | **24.27** | Stereotype erased + factual information preserved or even enhanced |

### Ablation Study

| Configuration | Outcome | Notes |
|------|---------|------|
| Retraining without regularization | SPLINCE = LEACE = SAL | Empirical validation of Theorem 3.2 |
| Retraining with L2 regularization | SPLINCE > LEACE > SAL | Range choice matters under regularization |
| Frozen final layer (LM) | SPLINCE >> LEACE | Largest gap when no retraining occurs |
| Strong task–concept correlation | Advantage of SPLINCE increases | Stronger correlation → more important to protect the range |

### Key Findings
- The advantage of SPLINCE grows as the correlation between the task and sensitive attributes increases.
- On the Winobias coreference resolution task, SPLINCE achieves the largest accuracy gain on anti-stereotypical prompts.
- In CelebA image experiments, visualizations of SPLINCE-projected images show that the method precisely preserves the "glasses" feature while erasing the "smile" feature.
- SPLINCE performs better on NLP tasks than on vision tasks; the performance gap in vision settings is a direction warranting further investigation.

## Highlights & Insights
- **Theoretical elegance**: The method derives a uniquely optimal solution to the joint problem of concept erasure and information preservation from a linear-algebraic perspective.
- **Theorem 3.2** is a particularly profound result: it precisely characterizes the boundary of *when range selection matters* by showing that, for a fixed kernel, range choice has no effect on predictions after regularization-free retraining.
- **SPLINCE as a natural extension of LEACE**: LEACE minimizes distortion without awareness of the task; SPLINCE exploits the degrees of freedom available in distortion minimization to protect task-relevant information.
- The Llama model experiments present a compelling application: eliminating stereotypical associations while preserving factual gender correlations.

## Limitations & Future Work
- The theoretical framework applies only to linear concept erasure; nonlinearly encoded concepts require alternative approaches.
- Prioritizing covariance preservation may introduce larger embedding distortions, causing projected representations to deviate substantially from the originals in certain settings.
- The method currently operates only on final-layer embeddings; its effectiveness for intermediate-layer interventions is limited.
- Multi-modal settings (e.g., CLIP) have not been explored; cross-modal covariance subspaces may be misaligned.
- The weaker performance on vision tasks compared to NLP tasks remains unexplained and merits further investigation.

## Related Work & Insights
- **LEACE** (Belrose et al., 2023) is the most direct predecessor: a linearly guarded projection that minimizes distortion.
- **INLP** and **RLACE** provide iterative/optimization-based frameworks for concept erasure.
- Oblique projections are widely used in signal processing; this paper introduces them to the domain of fair representation learning.
- The method motivates a broader design principle: achieving multi-objective balance through geometric constraints (kernel + range).

## Rating
- **Novelty**: ⭐⭐⭐⭐ — Applying oblique projection to concept erasure is novel and elegant, with solid theoretical contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multiple NLP and CV datasets, diverse language model architectures, and complete ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Theoretical derivations are clear, intuitions are well-explained, and figures are informative.
- **Value**: ⭐⭐⭐⭐ — Addresses a core limitation of the concept erasure field with both theoretical and practical significance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Causally Reliable Concept Bottleneck Models](causally_reliable_concept_bottleneck_models.md)
- [\[NeurIPS 2025\] Factor Decorrelation Enhanced Data Removal from Deep Predictive Models](factor_decorrelation_enhanced_data_removal_from_deep_predictive_models.md)
- [\[NeurIPS 2025\] Nearly-Linear Time Private Hypothesis Selection with the Optimal Approximation Factor](nearly-linear_time_private_hypothesis_selection_with_the_optimal_approximation_f.md)
- [\[NeurIPS 2025\] Fairness under Competition](fairness_under_competition.md)
- [\[NeurIPS 2025\] Stealthy Yet Effective: Distribution-Preserving Backdoor Attacks on Graph Classification](stealthy_yet_effective_distribution-preserving_backdoor_attacks_on_graph_classif.md)

</div>

<!-- RELATED:END -->

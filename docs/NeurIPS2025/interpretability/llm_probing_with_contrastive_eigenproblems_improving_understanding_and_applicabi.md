---
title: >-
  [Paper Note] LLM Probing with Contrastive Eigenproblems: Improving Understanding and Applicability of CCS
description: >-
  [NeurIPS 2025][Interpretability][CCS] This paper presents a rigorous analysis of the unsupervised probing method CCS (Contrast-Consistent Search) and proposes its reformulation as Contrastive Eigenproblems, yielding closed-form solutions with interpretable eigenvalues. This formulation eliminates CCS's sensitivity to random initialization and naturally extends to multivariate settings.
tags:
  - "NeurIPS 2025"
  - "Interpretability"
  - "CCS"
  - "contrastive probing"
  - "eigenproblems"
  - "mechanistic interpretability"
  - "latent knowledge elicitation"
date: 2026-05-08
content_hash: 7ff80b90a74d87e8
---

# LLM Probing with Contrastive Eigenproblems: Improving Understanding and Applicability of CCS

**Conference**: NeurIPS 2025
**arXiv**: [2511.02089](https://arxiv.org/abs/2511.02089)  
**Code**: To be confirmed  
**Area**: Interpretability
**Keywords**: CCS, contrastive probing, eigenproblems, mechanistic interpretability, latent knowledge elicitation

## TL;DR

This paper presents a rigorous analysis of the unsupervised probing method CCS (Contrast-Consistent Search) and proposes its reformulation as Contrastive Eigenproblems, yielding closed-form solutions with interpretable eigenvalues. This formulation eliminates CCS's sensitivity to random initialization and naturally extends to multivariate settings.

## Background & Motivation

Large language models (LLMs) achieve strong performance across numerous benchmarks, yet their internal mechanisms remain opaque. Mechanistic interpretability aims to identify the mechanisms underlying model behavior and to characterize the variables and representations the model employs.

CCS, proposed by Burns et al. (2023), is an unsupervised probing method designed to detect whether a language model represents binary features—such as the truth value of a sentence—in its internal activations. Its key advantage is that it requires no human-annotated labels and makes no assumption that the model's truth judgments align with human labels. Nevertheless, CCS suffers from several limitations:

**Insufficient understanding of the two loss terms**: CCS comprises a consistency loss and a confidence loss. The latter is commonly assumed to serve only to prevent the degenerate solution $p(\mathbf{x}^+) = p(\mathbf{x}^-) = 0.5$, yet its actual role is substantially more significant.

**Sensitivity to random initialization**: On certain datasets, CCS accuracy varies dramatically across random seeds (e.g., SNLI accuracy ranges from 49% to 90%).

**Inability to diagnose data quality**: When contrastive data fails to isolate a single feature, CCS cannot identify the root cause.

**Univariate limitation**: The original CCS probes only a single binary feature, making it difficult to extend to multivariate settings.

## Method

### Overall Architecture

The core idea of this paper is to linearize the CCS optimization objective and reformulate it as an eigenproblem, thereby obtaining a closed-form solution. Key concepts include:

- **Contrast pairs** $(X^+, X^-)$: a pair of semantically opposite inputs (e.g., an affirmative sentence and its negation)
- **Commonality Matrix**: $\mathbf{C} = \mathbf{X}^- + \mathbf{X}^+$, capturing shared features between positive and negative samples
- **Displacement Matrix**: $\mathbf{D} = \mathbf{X}^- - \mathbf{X}^+$, capturing the difference between positive and negative samples

### Key Designs

**1. Relative Contrast Consistency**

The authors show that the confidence loss in CCS functions to bias the probe toward high-variance directions (i.e., the principal components of the data). If data variance is inherently low along a given direction, high consistency along that direction does not imply that the model genuinely encodes the target feature. The relevant quantity is therefore **relative** consistency:

$$\text{Objective} = \min_{\hat{\boldsymbol{\theta}}} \frac{\|\hat{\boldsymbol{\theta}}^\intercal (\mathbf{X}^+ + \mathbf{X}^-)\|}{\|\hat{\boldsymbol{\theta}}^\intercal \mathbf{X}^{+-}\|}$$

That is, one seeks directions along which the variance of $\mathbf{C}$ is small relative to the overall variance of $\mathbf{X}^{+-}$.

**2. Difference-Relative Contrast (DRC)**

Expressing the variance change as a difference yields the following eigenproblem:

$$(\mathbf{C}^\intercal \mathbf{C} - \mathbf{X}^{+-\intercal} \mathbf{X}^{+-}) \mathbf{n}_k = \lambda_k \mathbf{n}_k$$

$$(\mathbf{D}^\intercal \mathbf{D} - \mathbf{X}^{+-\intercal} \mathbf{X}^{+-}) \mathbf{t}_k = \mu_k \mathbf{t}_k$$

Directions $\mathbf{n}_k$ corresponding to negative $\lambda_k$ have lower commonality-matrix variance than overall variance, indicating that contrastive feature differences are suppressed along those directions—making them suitable classification directions. Directions $\mathbf{t}_k$ corresponding to positive $\mu_k$ have higher displacement-matrix variance than overall variance, indicating that those directions encode contrastive feature differences.

**3. Ratio-Relative Contrast (RRC)**

Expressing the variance change as a ratio yields the following generalized eigenproblem:

$$\mathbf{C}^\intercal \mathbf{C} \mathbf{n}_k = \lambda_k \mathbf{X}^{+-\intercal} \mathbf{X}^{+-} \mathbf{n}_k$$

Here, $\lambda_k$ represents the ratio rather than the difference of variances. Both formulations reduce to the same symmetric eigenproblem, so DRC and RRC yield the same eigenvector basis.

**4. Multivariate Extension**

By constructing a $\mathbf{D}$ matrix from multiple groups of contrast pairs (e.g., simultaneously varying polarity and truth value), the eigenproblem naturally yields contrast-consistent eigenvectors along multiple orthogonal directions, enabling simultaneous probing of multiple binary features.

### Loss & Training

This method yields a closed-form solution requiring no gradient-based training. The procedure is:
1. Extract activations from an intermediate LLM layer (the paper uses layer 16 of Llama-2-7B)
2. Mean-center the activations
3. Optionally apply SVD dimensionality reduction to remove the null space
4. Construct matrices $\mathbf{C}$ and $\mathbf{D}$
5. Solve the eigenproblem and use the eigenvectors corresponding to the smallest/largest eigenvalues as probe directions

## Key Experimental Results

### Main Results

Classification accuracy is compared between CCS (30 seeds) and DRC/RRC on layer-16 activations of Llama-2-7B:

| Dataset | CCS min | CCS median | CCS max | DRC | RRC |
|--------|---------|------------|---------|-----|-----|
| comparisons | 100 | 100 | 100 | 100 | 100 |
| sp_en_trans | 99 | 99 | 99 | 98 | 98 |
| cities | 99 | 99 | 99 | 99 | 99 |
| amazon | 94 | 94 | 94 | 94 | 94 |
| imdb | 86 | 87 | 88 | 87 | 87 |
| ent_bank | 84 | 86 | 87 | 84 | 86 |
| snli | 49 | 86 | 90 | 82 | 73 |
| copa | 51 | 55 | 68 | 53 | 52 |
| rte | 46 | 50 | 61 | 50 | 50 |

### Ablation Study

An ablation over the two CCS loss terms is conducted to verify the actual role of the confidence loss:

| Method | comparisons | sp_en_trans | cities | amazon | imdb |
|------|-------------|-------------|--------|--------|------|
| CCS full | 100±0 | 99±0 | 99±0 | 93±0 | 87±0 |
| $\mathcal{L}_{conf}$ only | 100±1 | 96±8 | 98±4 | 94±0 | 81±9 |
| $\mathcal{L}_{cons}$ only | 66±11 | 64±13 | 70±14 | 67±9 | 60±6 |
| $\mathcal{L}_{cons}$+a1+a2 | 59±7 | 74±14 | 67±12 | 64±9 | 58±7 |

The ablation demonstrates that the confidence loss plays a far greater role than merely avoiding degenerate solutions: it effectively biases the probe toward high-variance directions, which is critical for finding accurate probes.

### Key Findings

1. **Eigenvalue distributions diagnose data quality**: When the data successfully isolates a single contrastive feature, the top eigenvalue is clearly separated (e.g., the amazon dataset); when the data conflates multiple features, the eigenvalue distribution is flatter (e.g., snli, copa).
2. **COPA case study**: The top DRC eigenvector encodes **sentiment polarity** rather than sentence truth value. The second eigenvector encodes truth value and achieves 70% accuracy, marginally outperforming the CCS maximum of 68%.
3. **Multivariate experiment**: On the cities dataset, the top three DRC eigenvectors encode truth value, propositional ground truth, and polarity, respectively—three orthogonal directions jointly capturing the interaction between truth value and polarity.

## Highlights & Insights

1. **Deep theoretical insight**: The paper demonstrates that the confidence loss in CCS functions to bias the probe toward high-variance directions rather than merely preventing degenerate solutions. This finding motivates the notion of *relative contrast consistency*.
2. **Closed-form solution replacing gradient optimization**: Eigendecomposition entirely eliminates random initialization sensitivity, and on datasets where CCS is stable, the proposed method closely matches CCS accuracy.
3. **Diagnostic power of eigenvalues**: The eigenvalue distribution quantitatively indicates whether contrastive data has successfully isolated a single feature, providing a new tool for data quality assessment.
4. **Natural and elegant multivariate extension**: Without additional design effort, the eigenproblem formulation inherently supports multivariate probing and successfully reproduces results on the truth-value–polarity shared subspace.

## Limitations & Future Work

1. **Unification of classification and intervention directions**: DRC/RRC cannot distinguish the classification direction $\mathbf{n}$ from the intervention direction $\mathbf{t}$ (which differ when features are correlated); future work should develop methods that separate the two.
2. **Limited performance on difficult datasets**: When contrastive data quality is poor (e.g., copa, rte), performance remains unsatisfactory, though the method can at least diagnose the issue.
3. **Restriction to linear probes**: The approach assumes features are linearly encoded in the latent space, potentially missing nonlinearly encoded features.
4. **Limited experimental scale**: Validation is performed on a single model (Llama-2-7B); generalizability across additional models and scales remains to be tested.
5. **Lack of downstream application validation**: The practical value of eigenvalue diagnostics within real interpretability workflows has not been demonstrated.

## Related Work & Insights

- **CCS line of work**: Burns et al. (2023) propose the original CCS; Farquhar et al. (2023) show that CCS may identify non-truth features; Belrose et al. (2024) analyze and extend CRC-TPC.
- **Truth-value–polarity interaction**: Bürger et al. (2024) find that truth value and polarity are encoded in a shared subspace; this paper reproduces that result using the more elegant eigenproblem framework.
- **Fry et al. (2023)**: Introduce a midpoint–displacement loss, employing the same $\mathbf{C}$ and $\mathbf{D}$ matrices as the present work but without relativization.
- **Laurito et al. (2024)**: Eliminate irrelevant contrastive directions via clustering and normalization, complementing this paper's approach of identifying all contrastive directions.
- **Activation intervention**: Turner et al. (2024)'s Activation Addition method leverages analogous directions for model intervention.
- **Insights**: The eigendecomposition approach generalizes to settings such as bias detection and safety feature probing; eigenvalue distributions serve as a quantitative indicator of contrastive dataset design quality.

## Rating

- Novelty: ⭐⭐⭐⭐ Reformulating CCS as an eigenproblem is a highly original contribution with rigorous and elegant theoretical derivation.
- Experimental Thoroughness: ⭐⭐⭐ Validated on 9 datasets spanning diverse scenarios, but limited to a single model (Llama-2-7B).
- Writing Quality: ⭐⭐⭐⭐⭐ The paper flows seamlessly from empirical observations through theoretical analysis to method design and multivariate extension, with detailed and clear mathematical derivations.
- Value: ⭐⭐⭐⭐ Provides deeper theoretical understanding of unsupervised probing and a practically useful eigenvalue-based diagnostic tool.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Improving Perturbation-based Explanations by Understanding the Role of Uncertainty Calibration](improving_perturbation-based_explanations_by_understanding_the_role_of_uncertain.md)
- [\[NeurIPS 2025\] URLs Help, Topics Guide: Understanding Metadata Utility in LLM Training](urls_help_topics_guide_understanding_metadata_utility_in_llm_training.md)
- [\[AAAI 2026\] Explainable Melanoma Diagnosis with Contrastive Learning and LLM-based Report Generation](../../AAAI2026/interpretability/explainable_melanoma_diagnosis_with_contrastive_learning_and_llm-based_report_ge.md)
- [\[ACL 2025\] Around the World in 24 Hours: Probing LLM Knowledge of Time and Place](../../ACL2025/interpretability/around_the_world_in_24_hours_probing_llm_knowledge_of_time_and_place.md)
- [\[NeurIPS 2025\] Understanding Prompt Tuning and In-Context Learning via Meta-Learning](understanding_prompt_tuning_and_in-context_learning_via_meta-learning.md)

</div>

<!-- RELATED:END -->

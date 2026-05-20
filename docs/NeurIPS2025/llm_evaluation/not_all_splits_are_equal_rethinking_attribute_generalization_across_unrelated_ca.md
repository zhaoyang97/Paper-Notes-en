---
title: >-
  [Paper Note] Not All Splits Are Equal: Rethinking Attribute Generalization Across Unrelated Categories
description: >-
  [NEURIPS2025][LLM Evaluation][attribute generalization] This paper presents the first systematic evaluation of how train/test splitting strategies affect generalization performance in attribute prediction tasks. It propo…
tags:
  - "NEURIPS2025"
  - "LLM Evaluation"
  - "attribute generalization"
  - "train/test splits"
  - "semantic leakage"
  - "clustering"
  - "linear probing"
  - "visual representations"
date: 2026-05-08
content_hash: 221605ea9f094425
---

# Not All Splits Are Equal: Rethinking Attribute Generalization Across Unrelated Categories

**Conference**: NEURIPS2025
**arXiv**: [2509.06998](https://arxiv.org/abs/2509.06998)  
**Code**: To be confirmed  
**Area**: LLM Evaluation
**Keywords**: attribute generalization, train/test splits, semantic leakage, clustering, linear probing, visual representations

## TL;DR

This paper presents the first systematic evaluation of how train/test splitting strategies affect generalization performance in attribute prediction tasks. It proposes four progressively harder splitting schemes based on LLM semantic grouping, embedding similarity, embedding clustering, and ground-truth supercategory labels. The study finds that unsupervised clustering-based splitting achieves leakage reduction comparable to ground-truth supercategory splits—without requiring any annotations—while retaining substantially better predictive performance.

## Background & Motivation

**Background**: Attributes (e.g., "has four legs," "has stripes") are central to how humans describe objects and carry natural cross-category transfer potential—"has stripes" learned from zebras can transfer to bees and tigers.

**Limitations of Prior Work**: Existing attribute prediction benchmarks are either taxonomically narrow (e.g., AwA covers only animals; CUB covers only birds) or fail to control the dissimilarity between training and test categories (e.g., VAW, MIT States), allowing models to exploit taxonomic shortcuts (semantic leakage) rather than genuinely learning attribute abstraction.

**Key Challenge**: When training and test sets contain semantically similar categories (e.g., "dog" in training and "wolf" in testing), models can achieve high scores by recognizing categories rather than understanding attributes, leading to inflated estimates of generalization.

**Core Problem**: No prior work has explicitly controlled the semantic and perceptual distance between training and test concepts to assess the true level of attribute generalization.

**Goal**: Can models generalize attribute knowledge learned from one set of categories to semantically and perceptually unrelated categories? For example, can "has four legs" learned from "dog" transfer to "chair"? The paper aims to construct fairer and more challenging evaluation protocols for attribute reasoning tasks, thereby advancing representation learning research.

## Method

### Overall Architecture

The paper proposes a suite of progressively harder train/test splitting strategies. Given a set of concepts (e.g., cat, strawberry, chair), each annotated with binary attributes, the goal is to assess whether these attributes are encoded in pretrained visual embeddings. Evaluation is performed by training a linear classifier (linear probe) per attribute on training concepts and evaluating on test concepts. The key innovation lies in how the train/test split is constructed: semantically similar concepts are grouped together so that similar concepts reside in the same partition, thereby controlling the degree of semantic leakage across partitions.

### Module A: LLM-based Semantic Grouping

- **Function**: Uses ChatGPT-4o to identify pairs of highly similar concepts (e.g., *cup* and *mug*) and assigns such high-similarity pairs jointly to the training set.
- **Mechanism**: Leverages LLM world knowledge for heuristic semantic deduplication, preventing direct semantic overlap between training and test sets.
- **Design Motivation**: LLMs capture human-level semantic similarity judgments, but coverage is limited—only 12% of concepts are assigned to groups, leaving many unassigned concepts potentially causing leakage across partition boundaries.

### Module B: Embedding Similarity Threshold Splitting

- **Function**: Computes pairwise cosine similarities between concept embeddings and assigns the top-ranked concepts (those with the highest maximum similarity to other concepts) to the training set.
- **Mechanism**: Concentrates semantically dense regions in the training set to minimize high-similarity pairs across the train/test boundary.
- **Design Motivation**: Data-driven rather than manually curated, but only the top 600 similarity-based groups are used for assignment; approximately 60% of samples remain unassigned, limiting leakage reduction due to insufficient coverage.

### Module C: Embedding Clustering Splitting

- **Function**: Applies K-Means clustering to concept embeddings and assigns entire clusters exclusively to either the training or test set.
- **Mechanism**: Clustering guarantees full coverage (every concept belongs to a cluster) while controlling cross-partition semantic overlap at a moderate granularity. $k=100$ is selected as it achieves the optimal balance between F1 and correlation metrics.
- **Design Motivation**: Addresses the fundamental coverage limitation of Modules A and B. Fully unsupervised—requiring no ground-truth labels—while achieving leakage reduction (as measured by CS) approaching that of ground-truth splits.

### Module GT: Ground-Truth Supercategory Labels

- **Function**: Groups concepts using the 53 manually annotated supercategories from THINGSplus, with each supercategory assigned entirely to either the training or test set.
- **Mechanism**: Serves as a strict upper-bound control, completely eliminating leakage across known taxonomic boundaries.
- **Design Motivation**: Although it provides the most thorough leakage reduction ($\text{CS} \approx 0.06$), the large group sizes cause certain attributes to concentrate entirely within a single supercategory (e.g., "has\_4\_legs" concentrated in mammals), rendering the task unlearnable and causing severe performance degradation.

### Evaluation Framework

- **Linear Probing**: scikit-learn `LogisticRegression` with balanced class weights, no regularization, and a maximum of 1,000 iterations; 211 binary classification tasks in total.
- **Metric 1 — F1 Selectivity**: The difference between the F1 score and a random baseline, measuring the effectiveness of attribute prediction.
- **Metric 2 — CS (Correlation with Supercategory)**: Pearson correlation between per-attribute F1 selectivity and supercategory dominance, measuring the degree to which models rely on taxonomic shortcuts.

## Key Experimental Results

### Table 1: F1 Selectivity Under Different Splitting Strategies (↑)

| Visual Model | Random | A. LLM | B. Similarity | C. Clustering | GT: Supercategory |
|---|---|---|---|---|---|
| SigLIP | 45.0 | 43.7 | 42.8 | 39.9 | 32.1 |
| CLIP | 43.6 | 42.0 | 40.9 | 38.6 | 33.2 |
| Swin-V2 | 43.2 | 42.0 | 39.2 | 34.3 | 25.1 |
| DINOv3 | 40.0 | 38.2 | 36.9 | 34.3 | 27.1 |

### Table 2: CS (Correlation with Supercategory) Under Different Splitting Strategies (↓)

| Visual Model | Random | A. LLM | B. Similarity | C. Clustering | GT: Supercategory |
|---|---|---|---|---|---|
| SigLIP | 0.36 | 0.35 | 0.36 | 0.12 | 0.01 |
| CLIP | 0.39 | 0.40 | 0.42 | 0.19 | 0.04 |
| Swin-V2 | 0.36 | 0.35 | 0.32 | 0.02 | -0.14 |
| DINOv3 | 0.37 | 0.35 | 0.36 | 0.14 | 0.03 |
| Average | 0.37±0.01 | 0.36±0.03 | 0.36±0.04 | 0.12±0.07 | 0.06±0.08 |

### Key Findings

1. **Performance drops sharply as leakage decreases**: From Random to GT Supercategory, SigLIP's F1 selectivity falls from 45.0 to 32.1 (a 28.7% decline) and Swin-V2's from 43.2 to 25.1 (a 41.9% decline), indicating that current models rely heavily on taxonomic shortcuts.
2. **Clustering offers the optimal trade-off**: Clustering achieves a CS of $0.12 \pm 0.07$ (close to GT's $0.06 \pm 0.08$) while maintaining significantly higher F1 than GT (e.g., SigLIP: 39.9 vs. 32.1), demonstrating that unsupervised clustering achieves the best balance between leakage reduction and task learnability.
3. **LLM and similarity-based methods have limited effect**: Modules A and B yield CS values nearly identical to Random (${\approx}0.36$); their low coverage leaves a large proportion of concepts uncontrolled, so leakage is not meaningfully reduced.
4. **Difference between vision-only and vision-language models**: Swin-V2 and DINOv3 suffer steeper performance drops under harder splits, whereas CLIP and SigLIP exhibit stronger generalization owing to language supervision signals that provide more abstract attribute representations.
5. **$k=100$ K-Means is optimal**: In an ablation over $k \in [10, 400]$, $k=100$ achieves the highest F1 selectivity while maintaining low CS.

## Highlights & Insights

1. **First systematic evaluation of attribute generalization**: No prior work had explicitly controlled the semantic distance between train and test partitions for attribute prediction; this paper fills that gap.
2. **Reveals inflated evaluation performance**: High performance under random splits is largely attributable to taxonomic shortcuts rather than genuine attribute understanding, raising fundamental questions about evaluation paradigms in attribute prediction and zero-shot learning.
3. **Engineering elegance of unsupervised clustering**: Without any manual annotation or LLM queries, K-Means alone achieves leakage reduction approaching that of ground-truth labels, offering high scalability.
4. **Deeper implications for representation learning**: Current visual embeddings primarily encode "what category" rather than "what attribute"; cross-category attribute abstraction remains an open challenge.

## Limitations & Future Work

1. **Limited dataset scale**: Validation is conducted on a single dataset (McRae × THINGS), comprising 1,854 concepts and 277 attributes (211 after filtering), which may limit representativeness.
2. **Linear probing only**: Linear classifiers can only detect linearly separable attribute information and cannot assess potentially nonlinear attribute encodings within the embeddings.
3. **Clustering depends on embedding quality**: The effectiveness of the embedding clustering method depends on the model used for clustering (Swin-V2); different embeddings may yield different groupings and leakage control outcomes.
4. **No downstream task validation**: Evaluation is restricted to probing tasks; it remains unverified whether conclusions transfer to practical downstream tasks such as zero-shot learning or compositional generalization.
5. **Single attribute granularity**: All attributes are binary labels; continuous attributes or degree differences (e.g., "very round" vs. "slightly round") are not considered.
6. **No proposed remedies**: The paper diagnoses the problem but does not propose methods for training representations with improved generalization.

## Related Work & Insights

- **Attribute prediction and zero-shot learning**: Lampert et al.'s AwA dataset pioneered attribute-based zero-shot classification but is limited to animal categories; Farhadi et al. proposed replacing simple category labels with rich attribute descriptions.
- **Compositional generalization**: Datasets such as MIT States, UT-Zappos50K, and C-GQA test recognition of novel (attribute, object) compositions but do not control inter-concept dissimilarity.
- **Probing classifiers**: The linear probing methodology of Alain & Bengio and the control task design of Hewitt & Liang provide the methodological foundation for this work.
- **Attribute reasoning across dissimilar categories**: CORE and Find-the-Common (FTC) share similar objectives but are either too small in scale or have evaluation structures unsuitable for attribute generalization.
- **Insights**: The splitting strategy is a severely underappreciated dimension of evaluation design; the methodology proposed here is generalizable to leakage control in other tasks such as relational reasoning and compositional generalization.

## Rating

- Novelty: ⭐⭐⭐⭐ (First systematic study of how splitting strategies affect attribute generalization; problem formulation is clear and important)
- Experimental Thoroughness: ⭐⭐⭐ (Complete matrix of four visual models × five splitting strategies, but limited to a single dataset and linear probing)
- Writing Quality: ⭐⭐⭐⭐ (Compact structure, intuitive figures, accurate description of contributions)
- Value: ⭐⭐⭐⭐ (Fundamental implications for evaluation fairness in attribute prediction and zero-shot learning; provides reproducible splitting tools)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Aggregation Hides OOD Generalization Failures from Spurious Correlations](aggregation_hides_out-of-distribution_generalization_failures_from_spurious_corr.md)
- [\[NeurIPS 2025\] Rethinking Losses for Diffusion Bridge Samplers](rethinking_losses_for_diffusion_bridge_samplers.md)
- [\[NeurIPS 2025\] Rethinking Evaluation of Infrared Small Target Detection](rethinking_evaluation_of_infrared_small_target_detection.md)
- [\[ICLR 2026\] Noise-Aware Generalization: Robustness to In-Domain Noise and Out-of-Domain Generalization](../../ICLR2026/llm_evaluation/noise-aware_generalization_robustness_to_in-domain_noise_and_out-of-domain_gener.md)
- [\[NeurIPS 2025\] Generalization Error Analysis for Selective State-Space Models Through the Lens of Attention](generalization_error_analysis_for_selective_state-space_models_through_the_lens_.md)

</div>

<!-- RELATED:END -->

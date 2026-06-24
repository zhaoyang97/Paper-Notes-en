---
title: >-
  [Paper Note] A Variational Approach for Mitigating Entity Bias in Relation Extraction
description: >-
  [ACL 2025][NLP Understanding][Entity Bias] Proposes an entity debiasing method based on Variational Information Bottleneck (VIB) that maps entity tokens to Gaussian distributions to selectively compress entity-specific information while preserving contextual semantics. This achieves SOTA performance across relation extraction datasets in generic, financial, and biomedical domains, particularly showing a notable improvement of 5.3 F1 points on BioRED in OOD scenarios.
tags:
  - "ACL 2025"
  - "NLP Understanding"
  - "Entity Bias"
  - "Variational Information Bottleneck"
  - "Relation Extraction"
  - "Debias"
  - "PLM"
date: 2026-05-08
content_hash: f4f535c61da90ec7
---

# A Variational Approach for Mitigating Entity Bias in Relation Extraction

**Conference**: ACL 2025  
**Code**: None  
**Area**: NLP Understanding / Relation Extraction  
**Keywords**: Entity Bias, Variational Information Bottleneck, Relation Extraction, Debias, PLM

## TL;DR

Proposes an entity debiasing method based on Variational Information Bottleneck (VIB) that maps entity tokens to Gaussian distributions to selectively compress entity-specific information while preserving contextual semantics. This achieves SOTA performance across relation extraction datasets in generic, financial, and biomedical domains, particularly showing a notable improvement of 5.3 F1 points on BioRED in OOD scenarios.

## Background & Motivation

**Background**: Relation Extraction (RE) is a core task in information extraction. Mainstream methods fine-tune pretrained language models (PLMs) on annotated data. However, models are prone to exploit shortcut heuristics, relying excessively on information carried by the entity itself (e.g., "Smith" frequently appearing in the "employed by" relation) to predict relations, rather than truly understanding the contextual semantics.

**Limitations of Prior Work**: Existing debiasing methods suffer from a severe performance-generalization trade-off. Entity masking directly replaces entities with generic tokens, which, despite reducing the in-distribution (ID) to out-of-distribution (OOD) gap, heavily degrades ID performance (by 7.5 F1 on TACRED). Entity replacement data augmentation fails to systematically address the bias. The current SOTA method, SCM, replaces entity embeddings with the center of the convex hull of entity neighbors, yet it lacks theoretical foundations and interpretability.

**Key Challenge**: Not all entity information is harmful—entity types and roles contribute positively to relation prediction. The issue lies in the entity identity, which induces spurious correlations. Selectively compressing harmful bias while preserving useful information is an information-theoretic trade-off.

**Goal**: To design a principled and interpretable entity debiasing framework that precisely controls the retention and compression of entity information, maintaining high performance in both in-distribution and out-of-distribution scenarios.

**Key Insight**: Variational Information Bottleneck (VIB) is naturally suited for this requirement—it compresses information by maximizing the mutual information between the output and the input while minimizing the mutual information between the intermediate representation and the input. In this work, VIB is applied solely to entity tokens, leaving non-entity tokens unchanged.

**Core Idea**: Maps entity token embeddings to a Gaussian distribution via VIB for randomized compression, where the variance naturally reflects the model's reliance on entities versus context, achieving both debiasing and interpretability.

## Method

### Overall Architecture

1. Input text is encoded by the PLM to obtain initial word embeddings $X$.
2. Identify entity token positions using a binary entity mask $M$.
3. Apply VIB to entity tokens: map entity embeddings to a Gaussian distribution $\mathcal{N}(\mu, \sigma)$ using a Single-Layer Perceptron (SLP).
4. Sample $z = \mu + \epsilon \cdot \sigma$ using the reparameterization trick.
5. Fuse the original embeddings with the compressed representation using a mixing factor $\beta$.
6. Feed the fused representation into the PLM encoder to obtain contextualized representations, and concatenate the subject/object tag representations for classification.

### Key Designs

1. **Selective VIB Compression**:

    - Function: Applies information bottleneck compression solely to entity tokens, precisely restricting the flow of entity identity information.
    - Mechanism: Uses a binary entity mask $M$ to identify entity locations, maps entity embeddings to a Gaussian distribution $\mathcal{N}(\mu, \sigma)$ via a single-layer perceptron, and samples $z = \mu + \epsilon \cdot \sigma$ with the reparameterization trick. Non-entity tokens fully retain their original embeddings.
    - Design Motivation: Global compression would impair the ability to understand context. Selective compression ensures that only entity-specific identity information is noisy/distorted, while contextual semantic signals remain intact.

2. **Embedding Mixture and Adaptive Loss**:

    - Function: Controls the degree of compression via a mixing factor, and balances classification and regularization losses using adaptive weights.
    - Mechanism: The mixing formula is $x' = x \cdot (1-M) + x \cdot M \cdot (1-\beta) + z \cdot M \cdot \beta$, where $\beta$ controls the proportion between original embeddings and compressed representations. The total loss is defined as $\mathcal{L} = L_{CE} + \alpha \cdot L_{VIB}$, where $\alpha$ is adaptively set to the ratio of the cross-entropy (CE) loss to the VIB loss, ensuring a dynamic balance between the two terms.
    - Design Motivation: $\beta$ serves as a continuous tuning knob between ID and OOD performance; adaptive $\alpha$ avoids manual hyperparameter tuning, making the training process more robust.

3. **Variance Interpretability Mechanism**:

    - Function: Provides a quantitative, interpretable indicator of the model's reliance on entities.
    - Mechanism: The magnitude of $\sigma^2$ directly reflects the model's judgment—low variance indicates that the model perceives the entity itself to carry strong relation signals (high dependency), while high variance implies the model relies more on contextual reasoning. This interpretability is validated by analyzing the sample distribution and corresponding relation types across different variance intervals.
    - Design Motivation: Existing debiasing methods (e.g., SCM) are black-box operations that cannot explain to users why a specific prediction is made. Variance, as a natural byproduct, offers interpretability with zero additional cost.

## Key Experimental Results

### Main Results

| Method | TACRED ID | TACRED OOD | REFinD ID | REFinD OOD | BioRED ID | BioRED OOD |
|------|-----------|------------|-----------|------------|-----------|------------|
| LUKE-Large | 71.1 | 63.8 | 75.0 | 73.4 | 56.9 | 51.8 |
| + Entity Mask | 63.6 | 61.7 | 71.4 | 71.4 | 53.2 | 40.2 |
| + SCM | 68.6 | 64.8 | 74.5 | 73.8 | 58.3 | 53.4 |
| + **VIB (ours)** | **70.4** | **66.5** | **75.4** | **74.8** | **61.2** | **58.7** |
| RoBERTa-Large | 70.8 | 61.5 | 75.1 | 72.7 | 57.7 | 47.9 |
| + SCM | 70.5 | **67.5** | 74.9 | 73.7 | 57.3 | **52.5** |
| + **VIB (ours)** | **70.7** | 67.2 | **75.4** | **74.4** | **63.0** | 52.5 |

### Ablation Study

| Variance Interval | Sample Proportion | Primary Relation Type |
|----------|----------|-------------|
| 0.0-0.1 | 4.6% | pers:title (high confidence entity dependency) |
| 0.1-0.2 | 85.8% | balanced region for most relations |
| 0.2-0.3 | 9.6% | org:date:formed_on (more reliant on context) |
| 0.3-0.4 | 0.1% | high context-dependent relations |

### Key Findings

1. **VIB demonstrates significant advantages on LUKE-Large**: It outperforms SCM by 5.3 F1 points on BioRED OOD, showing that VIB is better at leveraging entity-enhanced backbones.
2. **In-distribution to out-of-distribution gap is significantly reduced**: Although the entity masking method reduces the gap, it sacrifices ID performance; VIB substantially improves OOD performance while maintaining strong ID performance.
3. **Variance analysis validates interpretability**: Low-variance entities (such as person names + titles) indicate that the model perceives the entity itself to carry strong relation signals, while high-variance entities suggest the model relies more on contextual reasoning.

## Highlights & Insights

- **Novel Application of the VIB Framework**: Innovatively applies the variational information bottleneck from information theory to entity debiasing. It features a solid theoretical foundation and provides a principled debiasing framework.
- **Variance as a Free Interpretability Tool**: Without requiring additional design, $\sigma^2$ naturally reflects the model's reliance on entities versus context, allowing for a quantitative analysis of decision rationale for each sample.
- **Extensive Cross-Domain Validation**: Demonstrates effectiveness across three highly distinct domains: general NLP (TACRED), finance (REFinD), and biomedicine (BioRED), proving the generalizability of the proposed method.

## Limitations & Future Work

- Only validated on PLMs (RoBERTa/LUKE) without extension to LLMs and generative relation extraction frameworks.
- Experiments are restricted to English, leaving the patterns of entity bias in multilingual settings uninvestigated.
- VIB introduces additional inference overhead as the SLP is required to calculate the mean and variance.
- Future work could explore combining VIB with contrastive learning to further enhance entity-independent contextual representation learning.

## Related Work & Insights

- **vs. Entity Masking (Zhang et al., 2017)**: They completely remove entity information, whereas this work selectively compresses it, retaining useful parts. Entity masking degrades ID performance on TACRED by 7.5 F1, while VIB only drops by 0.7.
- **vs. SCM (Wang et al., 2023a)**: SCM replaces original embeddings with the convex hull center of neighboring entities, operating as a black box. VIB provides theoretical guarantees along with variance-based interpretability, while outperforming SCM by 5.3 F1 on BioRED OOD.

## Rating
- Novelty: ⭐⭐⭐⭐ Applying VIB to entity debiasing is a fresh perspective with clear theoretical motivation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive validation across six settings in three domains, coupled with variance-based interpretability analysis.
- Writing Quality: ⭐⭐⭐⭐ Clearly structured with tightly coupled theory and experiments.
- Value: ⭐⭐⭐ Effective method, but has a relatively narrow scope of application, lacking validation on LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Towards a More Generalized Approach in Open Relation Extraction](generalized_open_relation_extract.md)
- [\[ACL 2025\] Generating Diverse Training Samples for Relation Extraction with Large Language Models](generating_diverse_training_samples_for_relation_extraction_with_large_language_.md)
- [\[ACL 2025\] RISE: Reasoning Enhancement via Iterative Self-Exploration in Multi-hop Question Answering](rise_reasoning_enhancement_via_iterative_self-exploration_in_multi-hop_question_.md)
- [\[ACL 2025\] Analyzing Political Bias in LLMs via Target-Oriented Sentiment Classification](analyzing_political_bias_in_llms_via_target-oriented_sentiment_classification.md)
- [\[ACL 2025\] Dynamic Order Template Prediction for Generative Aspect-Based Sentiment Analysis](dot_absa_template.md)

</div>

<!-- RELATED:END -->

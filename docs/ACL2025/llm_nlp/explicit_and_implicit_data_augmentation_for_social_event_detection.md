---
title: >-
  [Paper Note] Explicit and Implicit Data Augmentation for Social Event Detection
description: >-
  [LLM (Other)] This paper proposes SED-Aug, a dual data augmentation framework combining explicit (LLM text augmentation) and implicit (feature space perturbation) methods for social event detection. It outperforms the strongest baselines by 17.67% and 15.57% in Average F1 on Twitter2012 and Twitter2018, respectively.
tags:
  - "LLM (Other)"
date: 2026-05-08
content_hash: e98ce3d51b491ce6
---

# Explicit and Implicit Data Augmentation for Social Event Detection

| Property | Content |
|------|------|
| Title | Explicit and Implicit Data Augmentation for Social Event Detection |
| Conference | ACL2025 |
| arXiv | [2509.04202](https://arxiv.org/abs/2509.04202) |
| Code | [github.com/congboma/SED-Aug](https://github.com/congboma/SED-Aug) |
| Area | Others / Social Event Detection |
| Keywords | Social Event Detection, Data Augmentation, LLM, Feature Perturbation, Graph Neural Network |

## TL;DR

This paper proposes SED-Aug, a dual data augmentation framework combining explicit (LLM text augmentation) and implicit (feature space perturbation) methods for social event detection. It outperforms the strongest baselines by 17.67% and 15.57% in Average F1 on Twitter2012 and Twitter2018, respectively.

## Background & Motivation

- **Social Event Detection (SED)**: Identifying and clustering important events from social media represents a vital tool for crisis management, public opinion analysis, and financial market analysis.
- **Core Challenge**: SED relies heavily on annotated data, but labeling is costly and labor-intensive, which limits the generalization ability of models across diverse event contexts.
- **Limitations of Prior Work**:
    - Graph-based methods (such as GraphHAM, ETGNN) exploit text and structural information, but underutilize data augmentation to enhance data diversity.
    - LLMs excel at text augmentation but struggle to handle graph data.
    - Pure text augmentation ignores the structural information inherent in user and event interactions.
- **Research Gap**: LLMs have not yet been applied to data augmentation for the SED task; there is a lack of a dual augmentation framework that simultaneously covers both textual and structural information.

## Method

### Overall Architecture

SED-Aug is a plug-and-play dual data augmentation framework consisting of two phases:
1. Explicit augmentation: Augmenting the textual content of social messages using LLMs.
2. Implicit augmentation: Perturbing structure-fused embeddings in the feature space.

Pipeline: Original message $\rightarrow$ LLM text augmentation $\rightarrow$ Augmented message + Original message + Structural data $\rightarrow$ Pre-trained Language Model (PLM) embedding extraction $\rightarrow$ Social graph construction $\rightarrow$ Graph aggregation to obtain structure-fused embeddings $\rightarrow$ Feature space perturbation $\rightarrow$ Classification.

### Explicit Data Augmentation (5 Strategies)

Through the LLM, $m_i^k = LLM^k(m_i)$ is generated, where $k$ represents different augmentation types:

| Strategy | Description |
|------|------|
| **Paraphrasing** | Rewriting the phrasing and structure while keeping semantics unchanged. |
| **Adding Context** | Adding relevant context information to enrich understanding and clarity of the message. |
| **Style Transfer** | Modifying writing style (tone, formality, etc.) while preserving the core meaning. |
| **Keep Entity Unchanged** | Modifying the text while keeping key entities (names, locations, dates, etc.) intact. |
| **Extract & Rewrite** (Two-stage) | First using an LLM to extract key information (keywords/entities/knowledge graphs), then generating a new version of the message. |

### Implicit Data Augmentation (5 Perturbation Methods)

Perturb the structure-fused embedding $g^i$; during training, decide to use either the augmented version or the original version with a probability $\alpha$:

**1. Gaussian Perturbation (GP)**:
$$g^i_{GP} = g^i + n_{GP}, \quad n_{GP} \sim \mathcal{N}(0, \sigma^2)$$
Directly adding Gaussian noise with a fixed standard deviation.

**2. Proportional Gaussian Perturbation (PGP)**:
$$g^i_{PGP} = g^i + n_{PGP}, \quad n_{PGP} \sim \mathcal{N}(0, \sigma^2) \cdot G$$
Noise is proportional to the feature values, preventing it from being too large or too small relative to the scale of the data.

**3. In-Distribution Gaussian Perturbation (IDGP)**:
$$g^i_{IDGP} = g^i + n_{IDGP}, \quad n_{IDGP} \sim \mathcal{N}(0, \alpha \cdot \text{std}(G)^2)$$
The standard deviation of the noise is based on the statistical properties of the input data itself, adaptively matching the data distribution.

**4. Clipped Gaussian Perturbation (CGP)**:
$$g^i_{CGP} = g^i + \text{Clip}(n_{CGP}, c), \quad n_{CGP} \sim \mathcal{N}(0, \sigma^2)$$
Truncates the noise within $[-c, c]$ to prevent extreme perturbations.

**5. Frequency-Domain Perturbation (FDP)**:
$$F^i = \mathcal{F}(g^i) \rightarrow F^i_{filtered} \rightarrow F^i_{FDP} = F^i_{filtered} + n \rightarrow g^i_{FDP} = \mathcal{F}^{-1}(F^i_{FDP})$$
Fourier transform to the frequency domain $\rightarrow$ selectively retain frequency components $\rightarrow$ add noise $\rightarrow$ inverse transform back to the time-space domain. This allows targeted enhancement of specific frequency components.

## Experiments

### Main Results

| Method | Kawarith6 Avg | Twitter2012 Avg | Twitter2018 Avg |
|------|-------------|-----------------|-----------------|
| TF-IDF | 92.68 | 50.97 | 31.30 |
| BERT | 75.99 | 60.24 | 43.73 |
| GraphMSE | 94.35 | 73.87 | 71.46 |
| GraphHAM | 94.84 | 77.57 | 76.16 |
| **SED-Aug** | **98.35** | **91.28** | **88.02** |
| Gain | +3.70↑ | +17.67↑ | +15.57↑ |

SED-Aug consistently outperforms the baselines on all datasets, especially with massive improvements on Twitter2012 and Twitter2018.

### Ablation Study

**Comparison of Explicit Augmentation Methods**:
- "Keep Entity Unchanged" consistently performs best across all datasets (Twitter2012 Micro F1: 92.76).
- "Style Transfer" shows the weakest performance, but still delivers a clear improvement.
- Preserving entity information is crucial for SED, as entities are key to distinguishing and understanding events.

**Comparison of Implicit Augmentation Methods**:
- PGP performs best on Kawarith6 and Twitter2018.
- CGP achieves the highest Micro F1 on Twitter2012.
- All 5 implicit methods consistently improve Macro F1, especially on the highly imbalanced Twitter2018 dataset (from 82.14% to 86.43%).
- Implicit augmentation benefits minority classes (rare events) the most.

**Combination of Explicit + Implicit**:
- Combining implicit and explicit augmentation always yields additional gains with no performance degradation.
- Positive effects are observed even when combined with the weakest explicit method.

### Comparison of Information Types in Extract & Rewrite

| Information Type | Kawarith6 | Twitter2012 | Twitter2018 |
|----------|-----------|-------------|-------------|
| Keywords | **Best** | **Best** | Second |
| Entities | Second | Second | **Best** |
| Knowledge Graph | Third | Third | Third |

Keywords are the most effective, followed by entities, while knowledge graphs underperform.

### Frequency-Domain Perturbation Mode Analysis

Retaining high-frequency components (i.e., attenuating low-frequency ones) yields the best results. This is because low-frequency components contain critical semantic structure information; thus it is more reasonable to preserve them and manipulate only high-frequency noise.

### Low-Resource Scenarios

| Training Data Ratio | W/o Augmentation | W/ Augmentation | Gain |
|-------------|--------|--------|------|
| 10% | 68.86 | 75.94 | +10.29 |
| 20% | 75.80 | 82.71 | +9.12 |
| 50% | 82.07 | 88.66 | +8.03 |
| 70% | 85.71 | 91.28 | +6.50 |

The improvement from augmentation is most significant when training data is extremely scarce (10%). Performance without augmentation plateaus after 30%, whereas dual augmentation continues to boost performance.

### Visual Analysis

- The histogram shows that the feature distribution shapes before and after augmentation are basically identical, with only a slight increase in variance (0.3284 $\rightarrow$ 0.3302), matching the design objective of "retaining the mean while only increasing diversity".
- PCA visualization indicates that the augmented points highly overlap with the original points but express subtle variations.

## Highlights & Insights

1. **First application of LLMs to SED data augmentation**: This work opens up a new direction for utilizing LLMs in social event detection.
2. **Complementary design of dual augmentation**: Explicit augmentation enriches textual diversity, while implicit augmentation introduces structure-aware variations in the feature space. The two complement each other without conflicts.
3. **Plug-and-play**: The framework is decoupled from the underlying SED model, allowing it to be seamlessly integrated into any graph-based SED method.
4. **Effective mitigation of class imbalance**: Implicit augmentation is particularly effective on imbalanced datasets, significantly improving the recognition capability for minority classes.
5. **Training-time augmentation, zero LLM overhead during inference**: LLMs are only utilized during the data preparation phase, introducing no extra overhead during inference.

## Limitations & Future Work

1. Lack of a clear standard to determine the optimal amount of augmented data—excessive augmentation might introduce noise and redundancy, while insufficient augmentation yields suboptimal effects.
2. Adding context with LLMs can introduce hallucinations (experiments identified approximately 6% fabricated claims).
3. The combinational space of 5 explicit methods $\times$ 5 implicit methods $\times$ multiple datasets is huge, and not all combinations have been exhaustively tested.
4. The augmentation performance depends on the quality of the underlying LLM, and different LLMs may yield different results.

## Related Work & Insights

- **Social Event Detection**: Evolution from content-based methods (TF-IDF, Word2Vec) to graph-based methods (GCN, GAT, KPGNN, GraphHAM).
- **Data Augmentation**: Text-level (insertion, deletion, substitution, back-translation) and feature-space level (Gaussian processes, class covariance matrices).
- **LLMs for Data Augmentation**: Already applied in other NLP tasks, but this is the first application in the SED field.

## Rating ⭐⭐⭐⭐

**Pros**: The framework is clearly designed, and both explicit and implicit augmentations are well-motivated with theoretical support. The experiments are exceptionally thorough (answering 6 research questions one by one), and the performance gains are substantial (+15-17% average F1).

**Cons**: The augmentation techniques themselves are not particularly novel (Gaussian perturbations, etc., are basic); the design of frequency-domain perturbation lacks in-depth theoretical analysis; there is a lack of comparison with other standard data augmentation baselines (e.g., Mixup, CutMix, etc.).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] PlanGenLLMs: A Modern Survey of LLM Planning Capabilities](plangenllms_planning_survey.md)
- [\[ACL 2025\] Culture is Not Trivia: Sociocultural Theory for Cultural NLP](culture_is_not_trivia_sociocultural_theory_for_cultural_nlp.md)
- [\[ACL 2025\] Classifying Unreliable Narrators with Large Language Models](classifying_unreliable_narrators.md)
- [\[ACL 2025\] KazMMLU: Evaluating Language Models on Kazakh, Russian, and Regional Knowledge of Kazakhstan](kazmmlu_evaluating_language_models_on_kazakh_russian_and_regional_knowledge_of_k.md)
- [\[ACL 2025\] Leveraging Large Language Models to Measure Gender Representation Bias in Gendered Language Corpora](leveraging_large_language_models_to_measure_gender_representation_bias_in_gender.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Enhancing Binary Encoded Crime Linkage Analysis Using Siamese Network
description: >-
  [AAAI 2026][Crime Linkage Analysis] This paper proposes a **Siamese Autoencoder**-based crime linkage analysis framework that integrates **geo-temporal features at the decoder stage** and employs a **domain expert-driven dimensionality reduction strategy**. Evaluated on the real-world ViCLAS database from the UK National Crime Agency (NCA), the method achieves up to 9% AUC improvement, providing an effective machine learning solution for high-dimensional sparse binary-encoded crime data.
tags:
  - AAAI 2026
  - Crime Linkage Analysis
  - Siamese Network
  - Autoencoder
  - Geo-temporal Feature Fusion
  - ViCLAS Database
date: 2026-05-08
content_hash: e9502cce46c0b002
---

# Enhancing Binary Encoded Crime Linkage Analysis Using Siamese Network

**Conference**: AAAI 2026
**arXiv**: [2511.07651](https://arxiv.org/abs/2511.07651)
**Code**: [https://github.com/AlberTgarY/CrimeLinkageSiamese](https://github.com/AlberTgarY/CrimeLinkageSiamese)
**Area**: Interpretability
**Keywords**: Crime Linkage Analysis, Siamese Network, Autoencoder, Geo-temporal Feature Fusion, ViCLAS Database

## TL;DR

This paper proposes a **Siamese Autoencoder**-based crime linkage analysis framework that integrates **geo-temporal features at the decoder stage** and employs a **domain expert-driven dimensionality reduction strategy**. Evaluated on the real-world ViCLAS database from the UK National Crime Agency (NCA), the method achieves up to 9% AUC improvement, providing an effective machine learning solution for high-dimensional sparse binary-encoded crime data.

## Background & Motivation

### Problem Setting

**Crime Linkage (CL)** is a critical task in modern law enforcement, aiming to identify serial crimes by analyzing offenders' **Modus Operandi (MO)**. Accurate CL helps optimize investigative resource allocation and enhance public safety. Its theoretical foundations are **behavioral consistency** (the same offender exhibits similar behavior across crimes) and **behavioral distinctiveness** (behavioral patterns of different offenders are distinguishable).

### Limitations of Prior Work

**Limitations of traditional statistical methods**: Logistic regression assumes linear feature relationships, and decision trees impose rigid hierarchical structures—neither is suited to capturing nonlinear associations in criminal behavior.

**High-dimensional sparse data challenges**: Real-world crime databases (e.g., ViCLAS) contain 446-dimensional binary-encoded features, with approximately 91% zero values, resulting in extremely sparse signals.

**Underutilization of geo-temporal information**: Existing methods either ignore geo-temporal information or naively concatenate it to the input layer, where the 2-dimensional geo-temporal signal accounts for less than 1% of the 217–446 dimensional behavioral features and is severely diluted.

**Limited dataset scale**: Prior research has focused primarily on small-scale, geographically restricted datasets, without validation on large-scale real-world databases such as ViCLAS.

### Core Innovation Motivation

- **Signal dilution problem**: Geo-temporal features are shifted from the input layer to the **decoder stage for fusion**, allowing them to modulate the latent representation after behavioral abstraction is complete, avoiding submersion by high-dimensional behavioral features.
- **Domain knowledge-driven dimensionality reduction**: Five data reduction mapping strategies were designed in collaboration with NCA domain experts, preserving behavioral semantics while reducing dimensionality.

## Method

### Overall Architecture

The framework comprises three core components:
1. **Siamese Autoencoder network**: Dual branches with shared weights; encoder compression + decoder reconstruction + geo-temporal fusion at the decoder stage
2. **Joint contrastive and reconstruction loss**: Pulling linked crimes together and pushing unlinked crimes apart
3. **Inference pipeline**: Similarity probability scores computed from latent space distances

### Key Designs

#### 1. **Siamese Autoencoder Network**

- **Encoder**: Two linear layers + ReLU (446→128→8), compressing input to an 8-dimensional latent space
- **Decoder**: Mirror structure (8→128→446), reconstructing the original feature space
- **Geo-temporal fusion**: After the first decoder layer output, log-transformed spatial-temporal features are mapped via a linear layer and fused additively
- **Parameter count**: 21,740 parameters (vs. 22,981 for Naive Siamese)—fewer parameters, better performance
- **Design Motivation**:
  - The reconstruction constraint of the autoencoder ensures the latent representation retains structural information
  - The 8-dimensional bottleneck forces the network to learn the most compact behavioral representation

#### 2. **Decoder-Stage Geo-temporal Fusion**

- **Core Idea**: Geo-temporal data inherently reflects **pairwise relationships** (distance and time interval between two crimes); fusing it after encoding individual behavioral features is more consistent with investigative logic
- **Implementation**:
  - Log-transform spatial distance and temporal interval
  - Map 2D geo-temporal features to 128 dimensions via a linear layer
  - **Additively fuse** with the first decoder layer output
- **Advantage over alternatives**:
  - Input-layer concatenation: 2D signal in 446D accounts for <1%, nearly no effect
  - Decoder fusion: Introduced after behavioral abstraction is complete, yielding significant signal amplification
- **Experimental Validation**: Consistently improves AUC by 0.86–3.29% across network variants (Table 4)

#### 3. **Domain Expert-Driven Dimensionality Reduction**

Five reduction strategies were designed (Table 1):

| Strategy | Remaining Features | Reduction Rate | Designer |
|---|---|---|---|
| No Map | 446 (original) | 0% | — |
| Map 1 | 282 | 36.8% | NCA analyst (20+ years experience) |
| Map 2 | 384 | 13.9% | Map 1 + forensic psychologist consultation |
| Map 3 | 266 | 40.4% | Hybrid of Map 1 and Map 2 |
| Map 4 | 217 | 51.3% | Forensic psychology expert |
| Map 5 | 286 | 35.9% | Refinement of Map 4 (reduced abstraction) |

- **Reduction method**: Semantically similar binary features are merged into more abstract categories (e.g., "shopping center parking lot" and "stadium parking lot" merged into "parking lot")
- **Design Motivation**: Analysts in practice typically link crimes via thematic matching rather than exact behavioral matching

### Loss & Training

**Joint loss**: $\mathcal{L} = \alpha \mathcal{L}_{\text{contrast}} + \beta \mathcal{L}_{\text{recon}}$, with $\alpha=1.0$, $\beta=0.2$

**Contrastive loss**: Uses a hybrid Euclidean–Manhattan distance metric
$$\mathcal{L}_{\text{contrast}} = \mathbb{E}[y \cdot d^2 + (1-y) \cdot \max(m-d, 0)^2]$$
where $y$ indicates linkage (1 = linked), and $m=5$ is the margin parameter.

**Reconstruction loss**: Based on cosine similarity
$$\mathcal{L}_{\text{recon}} = \mathbb{E}\left[\frac{v_1^\top \hat{v_1}}{\|v_1\| \|\hat{v_1}\|} + \frac{v_2^\top \hat{v_2}}{\|v_2\| \|\hat{v_2}\|}\right]$$

**Inference stage**: $S_{ij} = \exp(-D_{ij}/\beta)$, $\beta = m/1.5$, converting latent space distances to probability scores in $(0, 1]$.

**Training details**: Adam optimizer, lr=0.001, batch_size=128, 2 epochs, Cosine Annealing learning rate schedule, 5-fold cross-validation.

## Key Experimental Results

### Datasets

- **Single Victim-Offender-Scene Series**: 1,482 cases / 493 series / recorded in 2014; no geo-temporal data
- **Multiple Victim-Offender-Scene Series**: 22,282 crimes (1990–2021), 446 features, 11,970 used for analysis (the largest ViCLAS dataset in published research)

### Main Results

**Small dataset (Single Victim-Offender-Scene)**:

| Method | AUC | TP Fixed FP |
|---|---|---|
| Ours | 85 ± 1.98 | 77.73 ± 4.32 |
| Logistic Regression | 86 ± 2.14 | 77.19 ± 5.98 |
| PCA | 82 ± 4.02 | 64.97 ± 4.00 |

**Large dataset (Multiple Victim-Offender-Scene, using Map 5)**:

| Method | AUC | TP Fixed FP | AUPRC |
|---|---|---|---|
| **Ours (map 5)** | **84 ± 2.86** | **79.38 ± 2.56** | **15.43** |
| Naive Siamese (map 5) | 83 ± 2.72 | 76.20 ± 2.69 | 15.09 |
| Logistic Regression | 75 ± 2.97 | 70.43 ± 2.12 | 10.24 |
| Ours (no map) | 77 ± 2.11 | 68.31 ± 1.92 | 13.32 |

**Relative gain over LR**: +12.0% (AUC), +12.71% (TPFP), +50.68% (AUPRC).

### Ablation Study

**Impact of architectural choices (RQ3, Table 4 excerpt)**:

| Configuration | AUC (%) | Note |
|---|---|---|
| MLP, no skip, 2+2, Decoder | **77.29** | Best configuration |
| MLP, no skip, 2+2, Concat | 76.43 | Decoder fusion +0.86% |
| MLP, skip, 2+2, Decoder | 67.07 | Skip connections harmful |
| 1D CNN, no skip, 2+2, Decoder | 61.74 | MLP significantly better than CNN |
| SIREN, no skip, 2+2, Decoder | 58.28 | Periodic activation unsuitable |
| MLP, no skip, 1+1 | 52.49 | Too shallow |
| MLP, no skip, 4+4 | 63.57 | Too deep, overfitting |

**Key findings**:
- 2+2 layer depth is optimal; both shallower and deeper architectures degrade performance
- Omitting skip connections is beneficial (+6.55%), likely because direct feature propagation interferes with the abstraction of subtle crime patterns

### Key Findings

1. **Decoder fusion consistently outperforms input-layer concatenation**: MLP +0.86%, 1D CNN +3.29%, SIREN +3.09%
2. **Moderate dimensionality reduction improves performance**: Map 5 (35.9% reduction) achieves the best AUC of 84%; both excessive and insufficient abstraction underperform
3. **Temporal OOD challenge**: Post-COVID data from 2021–2025 shows notable performance degradation, necessitating periodic retraining
4. **System can reduce manual review by 80%**: Substantially reduces false positives while retaining over half of true crime linkages

## Highlights & Insights

1. **Real-world validation**: The UK NCA's ViCLAS database—the largest publicly studied sexual offense dataset to date—is used instead of synthetic data, enhancing practical significance.
2. **Generalizability of decoder fusion**: When auxiliary information dimensionality is far lower than that of primary features, the late fusion strategy merits broader adoption in other domains.
3. **Domain expert-driven feature engineering**: The five reduction strategies were designed by experts from different professional backgrounds, exemplifying best practices in combining ML with domain knowledge.
4. **Thorough ethical consideration**: Detailed deployment safeguards, bias auditing plans, and human-in-the-loop design are provided.

## Limitations & Future Work

1. **Information loss from binary encoding**: Reducing complex criminal behavior to 0/1 encoding may discard important frequency and intensity information.
2. **Temporal distribution shift**: Model performance degrades notably on post-2021 data, requiring a continuous retraining mechanism.
3. **Data confidentiality**: ViCLAS data is not publicly available, limiting reproducibility.
4. **8-dimensional bottleneck only**: The bottleneck may be overly compact; analysis of the performance impact of varying bottleneck dimensions is limited.
5. **Limited to sexual offenses**: Generalizability to other crime types such as burglary and robbery has not been evaluated.

## Related Work & Insights

- **Solomon et al. (2020)**: Applied Siamese networks to burglary linkage (40-dimensional TF-IDF); the present work extends this to 446-dimensional binary encoding with added reconstruction constraints.
- **Tonkin et al. (2017)**: Source of the hybrid Euclidean–Manhattan distance metric.
- **Insight**: For high-dimensional sparse data combined with low-dimensional auxiliary information, signal dilution is an underappreciated problem; late fusion strategies warrant systematic investigation.

## Rating

- Novelty: ⭐⭐⭐ — Core techniques (Siamese + autoencoder) are not novel; primary contributions lie in the application and fusion strategy
- Experimental Thoroughness: ⭐⭐⭐⭐ — Real-world data, extensive ablations, temporal OOD testing; lacks evaluation on other crime types
- Writing Quality: ⭐⭐⭐⭐ — Well-structured, research-question-driven, with thorough ethical discussion
- Value: ⭐⭐⭐⭐ — Direct practical value for law enforcement; methodological contribution is moderate

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] ShapBPT: Image Feature Attributions Using Data-Aware Binary Partition Trees](shapbpt_image_feature_attributions_using_data-aware_binary_partition_trees.md)
- [\[AAAI 2026\] A Closer Look at Knowledge Distillation in Spiking Neural Network Training](a_closer_look_at_knowledge_distillation_in_spiking_neural_ne.md)
- [\[AAAI 2026\] FourierPET: Deep Fourier-based Unrolled Network for Low-count PET Reconstruction](fourierpet_deep_fourier-based_unrolled_network_for_low-count_pet_reconstruction.md)
- [\[AAAI 2026\] ElementaryNet: A Non-Strategic Neural Network for Predicting Human Behavior in Normal-Form Games](elementarynet_a_non-strategic_neural_network_for_predicting_human_behavior_in_no.md)
- [\[AAAI 2026\] Probing Preference Representations: A Multi-Dimensional Evaluation and Analysis Method for Reward Models](probing_preference_representations_a_multi-dimensional_evaluation_and_analysis_m.md)

</div>

<!-- RELATED:END -->

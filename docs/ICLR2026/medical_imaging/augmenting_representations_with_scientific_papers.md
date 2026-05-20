---
title: >-
  [Paper Note] Augmenting Representations with Scientific Papers
description: >-
  [ICLR 2026][Medical Imaging][contrastive learning] This paper proposes the first multimodal foundation model framework that aligns X-ray spectra with scientific literature via contrastive learning…
tags:
  - "ICLR 2026"
  - "Medical Imaging"
  - "contrastive learning"
  - "multimodal representations"
  - "X-ray spectroscopy"
  - "scientific literature"
  - "foundation models"
  - "anomaly detection"
  - "astronomy"
date: 2026-05-08
content_hash: fe1be2a181fa2294
---

# Augmenting Representations with Scientific Papers

**Conference**: ICLR 2026
**arXiv**: [2603.04516](https://arxiv.org/abs/2603.04516)  
**Code**: None  
**Area**: Multimodal Learning / Scientific Foundation Models
**Keywords**: contrastive learning, multimodal representations, X-ray spectroscopy, scientific literature, foundation models, anomaly detection, astronomy

## TL;DR

This paper proposes the first multimodal foundation model framework that aligns X-ray spectra with scientific literature via contrastive learning, achieving 20% Recall@1% cross-modal retrieval in a shared latent space, improving physical parameter estimation by 16–18%, and discovering rare astrophysical objects including candidate pulsating ultraluminous X-ray sources.

## Background & Motivation

**Multimodal nature of astronomical data**: A single celestial object may simultaneously possess images, spectra, light curves, and decades of scientific literature descriptions, with each modality capturing complementary physical information.

**Massive data volumes and scalability demands**: Upcoming facilities such as the Vera Rubin Observatory and the Roman Space Telescope will generate petabyte-scale multimodal data, necessitating scalable methods for extracting scientific insights.

**Systematic integration of literature knowledge remains unexplored**: Although unimodal and multimodal astronomical foundation models exist, the systematic integration of observational data with scientific literature text has not been investigated.

**High-quality knowledge embedded in scientific literature**: Papers contain peer-reviewed expert interpretations, physical models, and contextual information that raw observational data alone cannot provide.

**Cross-domain generality**: The framework is not limited to astronomy; any domain with paired observational sequences and textual annotations—including seismology, climate science, and medicine—can benefit from this approach.

## Method

### Dataset Construction

- **X-ray spectra**: Sourced from the Chandra Source Catalog, discretizing the 0.5–8 keV energy range into 400 bins, recording photon count rates per bin with min-max normalization.
- **Scientific literature abstracts**: Cross-referenced via NASA ADS using sky coordinates and SIMBAD source identifiers; GPT-4o-mini is used to generate summaries from relevant papers, which are then encoded into 4,608-dimensional embeddings using OpenAI Ada-002.
- **Final dataset**: 11,447 spectrum–text pairs, split into training (69%), calibration (1%), validation (15%), and test (15%) sets, with each sample associated with up to 20 physical variables as ground truth.

### Architecture Design

The overall design follows the foundation model paradigm: two pre-trained unimodal encoders combined with contrastive alignment.

1. **Spectral encoder**: A Transformer-based autoencoder compressing spectra into 64-dimensional latent vectors, optimized with an MAE reconstruction loss.
2. **Text encoder**: GPT-4o-mini generates summaries → Ada-002 embeddings (4,608 dimensions).
3. **Alignment network**: Two fully connected networks mapping spectral (64-dim) and text (4,608-dim) representations into a shared 64-dimensional space.
4. **Contrastive loss**: InfoNCE loss is applied:

$$\mathcal{L}_{\text{InfoNCE}} = -\frac{1}{N}\sum_{i=1}^{N}\log\frac{\exp(\text{sim}(t_i, d_i)/\tau)}{\sum_{j=1}^{N}\exp(\text{sim}(t_i, d_j)/\tau)}$$

where $\text{sim}$ denotes cosine similarity and $\tau$ is the temperature parameter.

### Downstream Evaluation Tasks

- **Cross-modal retrieval**: Given a spectrum, retrieve the corresponding textual description via similarity search.
- **Physical parameter regression**: $k$-NN ($k=3$) applied on latent representations to predict 20 physical variables, with a Mixture-of-Experts (MoE) strategy selecting the optimal representation for each variable.
- **Anomaly detection**: Isolation Forest applied in the aligned latent space to identify rare astrophysical objects.

### Training Details

- Optimizer: Adam, with grid search over learning rate ($10^{-4}$ to $10^{-3}$), shared space dimensionality (16–128), dropout (0.1–0.5), and hidden dimensionality (16–1024).
- Evaluation metrics: Recall@k%, Median Rank, MAE (regression), and Pearson correlation coefficient (latent space–physical variable relationships).

## Key Experimental Results

### Cross-Modal Retrieval

| Metric | Value |
|--------|-------|
| Recall@1% | ≈20% |
| Recall@5% | ≈50% |
| Median Rank | 84 / 1,719 |

> A median rank of 84 implies that only approximately 5% of the search space needs to be explored to find the correct match.

### Physical Parameter Estimation (Selected from Table 3)

| Variable | Best Pre-Alignment MAE | MoE MAE | Gain |
|----------|------------------------|---------|------|
| hard_hs | 0.20 | 0.12 | 40% |
| powlaw_gamma | 0.65 | 0.41 | 36% |
| powlaw_nh | 33.08 | 21.63 | 35% |
| brems_nh | 23.31 | 14.12 | 39% |
| flux_significance_b | 7.36 | 4.54 | 38% |
| **Average** | — | — | **16–18%** |

- Hardness ratios (hard_hs, hard_ms, hard_hm) improve by an average of 34%.
- Hydrogen column density ($N_H$) improves by an average of 34% across spectral models.
- For temporal variability metrics, text alone outperforms spectra, as spectra inherently carry no temporal information.

### Physical Interpretability of the Latent Space

| Latent Dimension | Physical Variable | Pearson Correlation |
|-----------------|-------------------|---------------------|
| $L_{12}$ | hard_hs | 0.82 |
| $L_{48}$ | apec_kt | 0.74 |
| $L_8$ | powlaw_gamma | 0.68 |
| $L_{62}$ | bb_kt | 0.68 |

- Pre-alignment: mean $|\rho| = 0.43$ for spectra, $0.30$ for text; post-alignment combined mean $|\rho| = 0.55$, representing a substantial improvement.
- 97% data compression (4,672 → 128 dimensions) while preserving predictive capability, which is critical for billion-object surveys.

### Anomaly Detection

- Quasars (QSOs) exhibit higher median anomaly scores than typical AGN, reflecting their extreme luminosities.
- Ultraluminous X-ray sources (ULXs) show large variance, consistent with the presence of pulsating and non-pulsating sub-populations.
- The top 1% anomalies include the gravitational lens system 2CXOJ224030.2+032131 and the candidate pulsating ULX 2CXOJ004722.6-252050, the latter independently validated as a candidate PULX.

## Highlights & Insights

- **First spectrum–literature alignment framework**: Systematically integrating scientific literature as a modality into astronomical observational foundation models, pioneering a new paradigm of *knowledge-augmented representation learning*.
- **Emergent properties of contrastive learning**: The alignment process not only enables retrieval but also gives rise to stronger correlations with physical variables in the latent space—an effect not explicitly enforced during training.
- **Superiority of the MoE strategy**: Adaptively selecting the optimal representation for each variable across unimodal and shared representations is more flexible than fixed multimodal fusion.
- **Cross-domain blueprint**: The framework directly generalizes to seismology (waveforms + event reports), climate science (time series + assessment documents), and medicine (physiological signals + clinical notes).
- **Scientific discovery capability**: Candidate objects subsequently confirmed by independent studies were discovered through anomaly detection, validating the method's potential for scientific discovery.

## Limitations & Future Work

1. **Retrieval performance has room for improvement**: 20% Recall@1% indicates insufficient alignment, which could be alleviated by improving text summarization and increasing the number of data pairs.
2. **Modality information asymmetry**: Scientific literature encompasses far richer physical context than a single spectrum, creating an inherent alignment mismatch.
3. **Limited to retrieval and regression**: Effectiveness on generative tasks such as text generation has not been validated.
4. **Anomaly detection lacks physical priors**: Purely statistical anomalies may include artifacts; incorporating physical constraints could help prioritize scientifically meaningful outliers.
5. **Validation limited to astronomical data**: Despite claims of cross-domain generality, experiments are confined to X-ray astronomical observations.

## Related Work & Insights

- **AstroCLIP (Lanusse et al.)**: An astronomical multimodal foundation model that does not incorporate text or literature as a modality.
- **CLIP (Radford et al., 2021)**: The seminal image–text contrastive learning method; this paper transfers its principles to scientific observation–literature alignment.
- **NASA ADS**: Provides the cross-referencing infrastructure that makes large-scale spectrum–literature pairing feasible.
- **Insight**: Scientific literature represents one of the most accessible and information-dense sources of "supervisory signal." Using it as an anchor modality for contrastive learning offers a low-cost means of augmenting representations across diverse observational data types.

## Rating

| Dimension | Score (1–5) | Notes |
|-----------|-------------|-------|
| Novelty | 4 | First to align scientific literature as a modality with observational data, opening a new direction |
| Technical Depth | 3 | Methodology is relatively straightforward (contrastive learning + kNN), but dataset construction and evaluation design are thorough |
| Experimental Thoroughness | 3 | Comprehensive evaluation across 20 physical variables, but limited to a single dataset and domain |
| Writing Quality | 4 | Well-structured; cross-domain vision is presented in an inspiring manner |
| Value | 4 | Highly generalizable framework with publicly released data; practically significant for large-scale surveys |
| **Overall** | **3.6** | A direction-pioneering work with concise yet effective methodology; cross-domain generalization is its core potential |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] BiomedSQL: Text-to-SQL for Scientific Reasoning on Biomedical Knowledge Bases](biomedsql_text-to-sql_for_scientific_reasoning_on_biomedical_knowledge_bases.md)
- [\[ICLR 2026\] Reverse Distillation: Consistently Scaling Protein Language Model Representations](reverse_distillation_consistently_scaling_protein_language_model_representations.md)
- [\[ICLR 2026\] Learning Domain-Aware Task Prompt Representations for Multi-Domain All-in-One Image Restoration](learning_domain-aware_task_prompt_representations_for_multi-domain_all-in-one_im.md)
- [\[NeurIPS 2025\] GraphFLA: Augmenting Biological Fitness Prediction Benchmarks with Landscape Features](../../NeurIPS2025/medical_imaging/augmenting_biological_fitness_prediction_benchmarks_with_landscapes_features_fro.md)
- [\[ICCV 2025\] SciVid: Cross-Domain Evaluation of Video Models in Scientific Applications](../../ICCV2025/medical_imaging/scivid_cross-domain_evaluation_of_video_models_in_scientific_applications.md)

</div>

<!-- RELATED:END -->

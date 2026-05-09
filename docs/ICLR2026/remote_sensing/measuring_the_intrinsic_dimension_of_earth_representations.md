---
title: >-
  [Paper Note] Measuring the Intrinsic Dimension of Earth Representations
description: >-
  [ICLR 2026][Remote Sensing][intrinsic dimension] This work presents the first systematic measurement of the intrinsic dimension (ID) of Geographic Implicit Neural Representations (Geographic INR), finding that 256–512-dimensional embeddings have true IDs of only 2–10. Higher ID in frozen embedding spaces correlates positively with downstream performance, while lower ID in supervised task-head activation spaces correlates positively with performance, revealing a dual mechanism of "representativeness vs. task alignment."
tags:
  - ICLR 2026
  - Remote Sensing
  - intrinsic dimension
  - geographic implicit neural representations
  - Earth observation
  - representation learning
  - unsupervised evaluation
date: 2026-05-08
content_hash: 21faabfde1a26428
---

# Measuring the Intrinsic Dimension of Earth Representations

**Conference**: ICLR 2026
**arXiv**: [2511.02101](https://arxiv.org/abs/2511.02101)
**Code**: [GitHub](https://github.com/arjunarao619/GeoINRID)
**Area**: Remote Sensing / Representation Learning
**Keywords**: intrinsic dimension, geographic implicit neural representations, Earth observation, representation learning, unsupervised evaluation

## TL;DR

This work presents the first systematic measurement of the intrinsic dimension (ID) of Geographic Implicit Neural Representations (Geographic INR), finding that 256–512-dimensional embeddings have true IDs of only 2–10. Higher ID in frozen embedding spaces correlates positively with downstream performance, while lower ID in supervised task-head activation spaces correlates positively with performance, revealing a dual mechanism of "representativeness vs. task alignment."

## Background & Motivation

Geographic INRs map latitude–longitude coordinates $(λ, ϕ)$ to high-dimensional embedding vectors $z = f(λ, ϕ) \in \mathbb{R}^D$ (typically $D = 256$ or $512$) via contrastive pre-training on satellite imagery, ground-level photos, or text. Models such as SatCLIP, GeoCLIP, and CSP have been widely adopted for downstream tasks including land-cover segmentation, object detection, and image geolocalization.

**Core Problem**: How much effective information do these high-dimensional representations actually encode? Existing evaluation relies entirely on downstream task labels, lacking label-free, architecture-agnostic measures of information content.

**Key Insight**: The Earth's surface is inherently a two-dimensional sphere $S^2$, so the input manifold dimension of any INR is known to be 2. If the embedding ID substantially exceeds 2, the model genuinely encodes geographic signals beyond raw coordinates; if the ID approaches the ambient dimension $D$, redundancy may be present. This "known input dimension + measurable output ID" setting makes Geographic INRs an ideal testbed for studying ID.

## Method

### Overall Architecture

The methodology follows two parallel tracks corresponding to the paper's dual mechanism:

1. **Representativeness Measurement**: The pre-trained INR is frozen; 100,000 coordinates are uniformly sampled across global land, producing embeddings $Z_{geo} \in \mathbb{R}^{N \times D}$. The **angular estimator FisherS** is then applied to compute the global ID. FisherS is robust to spatial heterogeneity and is not confounded by local density variations such as climate-zone boundaries.

2. **Task-Alignment Measurement**: INR embeddings are frozen while a shallow MLP classification/regression head is trained. The **distance estimator TwoNN** is applied to the penultimate ReLU activations to measure the ID of the task-aligned manifold onto which supervised learning compresses the embeddings.

### Key Designs

1. **Complementary Use of Angular and Distance Estimators**: Angular estimators (FisherS) eliminate local density differences via whitening and spherical projection, making them suitable for global cross-model comparisons. Distance estimators (MLE/TwoNN/MOM/TLE) are sensitive to local neighborhood distances, making them suitable for generating spatial ID maps that reveal artifacts. This design insight is the foundation of the paper's analytical reliability.

2. **Spatial Visualization of Local ID**: MLE ($k=100$ neighbors) is used to compute per-point IDs and render global maps, directly revealing that GeoCLIP's ID peaks over the United States and Western Europe (reflecting social-media image distribution bias), that CSP exhibits grid-stripe artifacts (from periodic repetition in positional encodings), and that SatCLIP shows subtle oscillations (from finite-order truncation of spherical harmonics).

3. **Causal Resolution–ID Relationship**: The paper systematically controls resolution hyperparameters—Legendre polynomial order $L$ for SatCLIP, maximum RFF frequency $\sigma_{max}$ and hierarchy level $M$ for GeoCLIP, and frequency component count $S$ for Space2Vec—and observes monotonically increasing ID with resolution, establishing a causal rather than merely correlational relationship.

## Key Experimental Results

### Global Intrinsic Dimension per Model

| Model | Type | $D$ | FisherS | MLE | MOM | TLE |
|-------|------|-----|---------|-----|-----|-----|
| SatCLIP-L10 | Location encoder | 256 | 5.00 | 1.96 | 2.02 | 2.16 |
| SatCLIP-L40 | Location encoder | 256 | 8.08 | 2.03 | 2.39 | 2.32 |
| GeoCLIP | Location encoder | 512 | 7.68 | 11.21 | 13.02 | 11.53 |
| CSP-fMoW | Location encoder | 256 | 1.70 | 5.18 | 5.23 | 6.25 |
| CSP-iNat | Location encoder | 256 | 0.92 | 3.37 | 4.64 | 4.14 |
| SINR | Location encoder | 256 | 3.19 | 2.19 | 3.36 | 2.74 |
| TaxaBind-Loc | Location encoder | 512 | 3.33 | 9.44 | 11.56 | 10.30 |
| CROMA | Image encoder | 768 | 9.79 | 19.57 | 17.00 | 20.30 |
| DOFA | Image encoder | 768 | 3.32 | 15.58 | 13.78 | 16.20 |
| ResNet152 | Image encoder | 2048 | 7.60 | 20.72 | 17.50 | 21.50 |

All location encoders exhibit IDs one to two orders of magnitude below their ambient dimensions. GeoCLIP's distance-estimated ID (11–13) approaches that of the large image encoder DOFA (14–16), indicating that coordinate-only inputs can encode rich geographic information.

### Effect of Input Modality on ID and Performance

| Pre-training modality | Global FisherS ID | Temperature R² | Elevation R² | Population R² |
|-----------------------|-------------------|----------------|--------------|---------------|
| Sentinel-2 | ~7.5 | ~0.76 | ~0.74 | ~0.78 |
| S1 + S2 | ~8.5 | ~0.80 | ~0.82 | ~0.82 |
| All modalities | ~9.5 | ~0.84 | ~0.86 | ~0.86 |

More input modalities → higher ID → better downstream performance, with all three quantities increasing monotonically.

### Key Findings

- **Embedding-space ID positively correlates with performance**: Higher global FisherS ID in frozen INR embeddings consistently yields better downstream regression/classification performance across all five tasks (temperature, elevation, population, biome, and country classification). Higher ID implies greater representativeness, providing more independent directions for shallow learners to exploit.
- **Activation-space ID negatively correlates with performance**: Lower TwoNN ID in the penultimate layer of supervised MLPs corresponds to better performance. Supervised adaptation compresses INR features onto a lower-dimensional task-aligned manifold, consistent with the findings of Ansuini et al. (2019) for classification networks.
- **Resolution controls ID**: Increasing SatCLIP's Legendre order from 10 to 40 raises FisherS ID from 5.0 to 8.1; increasing GeoCLIP's RFF maximum frequency causes ID to surge from 7.7 to 75.7.
- **Local ID exposes data bias**: GeoCLIP's ID is highest over the US and Western Europe (dense training-data regions); CSP exhibits grid artifacts from periodic positional encodings—both directly usable for model diagnostics.

## Highlights & Insights

- The **dual mechanism of representativeness vs. task alignment** is the paper's most central contribution: the same ID measure exhibits opposite correlation directions in embedding space vs. activation space, elegantly unifying the intuitions that "pre-training should be broad" and "fine-tuning should be narrow."
- The practical utility of ID as a label-free metric is clearly demonstrated: it can substitute for expensive downstream evaluation in model selection, hyperparameter search, and early stopping.
- Local ID maps constitute an intuitive and effective model diagnostic tool, capable of revealing pre-training data coverage bias and spatial artifacts introduced by architectural choices.
- The ID of Geographic INRs (2–10) is far below the ambient dimension (256–512), suggesting that current representations are heavily redundant and offer significant room for compression.

## Effect of Resolution on ID

| Model | Resolution parameter | Value | Global FisherS ID |
|-------|----------------------|-------|-------------------|
| SatCLIP | Legendre order $L$ | 10 | 5.0 |
| SatCLIP | Legendre order $L$ | 20 | ~6.5 |
| SatCLIP | Legendre order $L$ | 40 | 8.1 |
| GeoCLIP | RFF max frequency $\sigma_{max}$ | $2^8$ | 7.7 |
| GeoCLIP | RFF max frequency $\sigma_{max}$ | $2^{16}$ | 75.7 |

SatCLIP's ID grows nearly linearly with spherical harmonic order; GeoCLIP's ID increases nearly tenfold upon raising the RFF frequency, indicating that high-frequency positional encodings substantially expand the effective degrees of freedom in the embedding.

## Limitations & Future Work

- Different ID estimators yield substantially different numerical values (e.g., SatCLIP-L40: FisherS = 8.08 vs. MLE = 2.03), requiring careful estimator selection depending on context.
- The analysis is limited to static INRs with 2D coordinate inputs; spatiotemporal representations incorporating a time dimension are not considered.
- ID is a scalar summary and cannot characterize the directional structure or semantic organization of the embedding space.
- The representativeness–task-alignment correlation analysis is based on a limited set of seven location encoders and five downstream tasks; statistical significance depends on sample size.
- The paper does not explore how ID analysis could inversely guide INR architecture design, such as adaptive dimension allocation based on local ID or region-weighted fine-tuning.
- **Representation learning evaluation**: Conventional evaluation relies on downstream task probing; this work provides a label-free alternative. The ID analysis framework generalizes to pre-trained representation evaluation in other domains (e.g., language model representations in NLP, medical image representations).

## Rating
- Novelty: ⭐⭐⭐⭐ (novel perspective, though analytical tools are existing)
- Experimental Thoroughness: ⭐⭐⭐⭐ (comprehensive multi-model, multi-dimensional analysis)
- Writing Quality: ⭐⭐⭐⭐ (27 pages with detailed appendices)
- Value: ⭐⭐⭐⭐ (provides an important analytical tool for Earth observation representation learning)

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Earth-Agent: Unlocking the Full Landscape of Earth Observation with Agents](earth-agent_unlocking_the_full_landscape_of_earth_observation_with_agents.md)
- [\[ICCV 2025\] WildSAT: Learning Satellite Image Representations from Wildlife Observations](../../ICCV2025/remote_sensing/wildsat_learning_satellite_image_representations_from_wildlife_observations.md)
- [\[ICCV 2025\] Towards a Unified Copernicus Foundation Model for Earth Vision](../../ICCV2025/remote_sensing/towards_a_unified_copernicus_foundation_model_for_earth_vision.md)
- [\[ICLR 2026\] Task-free Adaptive Meta Black-box Optimization](task-free_adaptive_meta_black-box_optimization.md)
- [\[ICLR 2026\] AutoFly: Vision-Language-Action Model for UAV Autonomous Navigation in the Wild](autofly_vision-language-action_model_for_uav_autonomous_navigation_in_the_wild.md)

<!-- RELATED:END -->

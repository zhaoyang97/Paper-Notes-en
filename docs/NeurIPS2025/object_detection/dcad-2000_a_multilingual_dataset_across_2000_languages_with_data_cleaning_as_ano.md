---
title: >-
  [Paper Note] DCAD-2000: A Multilingual Dataset across 2000+ Languages with Data Cleaning as Anomaly Detection
description: >-
  [NeurIPS 2025][Object Detection][multilingual dataset] This work constructs DCAD-2000, a multilingual dataset covering 2,282 languages and 46.72 TB of text…
tags:
  - "NeurIPS 2025"
  - "Object Detection"
  - "multilingual dataset"
  - "data cleaning"
  - "anomaly detection"
  - "Common Crawl"
  - "low-resource languages"
date: 2026-05-08
content_hash: 4de3967d2e63c73a
---

# DCAD-2000: A Multilingual Dataset across 2000+ Languages with Data Cleaning as Anomaly Detection

**Conference**: NeurIPS 2025
**arXiv**: [2502.11546](https://arxiv.org/abs/2502.11546)  
**Code**: [https://github.com/yl-shen/DCAD-2000](https://github.com/yl-shen/DCAD-2000)  
**Area**: Multilingual Translation
**Keywords**: multilingual dataset, data cleaning, anomaly detection, Common Crawl, low-resource languages

## TL;DR
This work constructs DCAD-2000, a multilingual dataset covering 2,282 languages and 46.72 TB of text, and proposes a language-agnostic data cleaning framework that reformulates cleaning as anomaly detection. The framework extracts 8-dimensional statistical features per document and applies Isolation Forest for dynamic noise filtering. Effectiveness is validated on multiple multilingual benchmarks, with particularly notable gains on low-resource languages.

## Background & Motivation

**Background**: Multilingual LLMs require large-scale, high-quality multilingual datasets. Existing datasets (CulturaX, Madlad-400, Fineweb-2, etc.) have steadily increased language coverage, yet exhibit three critical limitations.

**Limitations of Prior Work**: (a) **Data staleness**: most datasets are based on Common Crawl snapshots prior to 2022, resulting in outdated knowledge; (b) **Insufficient coverage of high/mid-resource languages**: Fineweb-2 supports 1,915 languages but only includes 10 high-resource and 62 mid-resource languages; (c) **Inadequate cleaning**: the Sailor study found that 31% of Madlad-400 data can still be removed by more advanced cleaning methods. Traditional heuristic cleaning with fixed thresholds generalizes poorly across languages, as distributions of word count, repetition rate, and perplexity vary substantially between languages.

**Key Challenge**: Manually tuning thresholds per language is not scalable (fine-tuning thresholds for 1,000+ languages as in Fineweb-2 is computationally prohibitive), yet uniform thresholds do not transfer across languages. A language-agnostic, adaptive cleaning approach is therefore needed.

**Goal**: (a) Construct an up-to-date, comprehensive multilingual dataset directly usable for training; (b) propose a data cleaning framework that eliminates manual threshold tuning.

**Key Insight**: Data cleaning is reformulated as anomaly detection — low-quality documents are treated as outliers in feature space, identifiable via algorithms such as Isolation Forest.

**Core Idea**: Replace fixed-threshold filtering with anomaly detection — extract 8-dimensional statistical features per document and allow the algorithm to automatically learn what constitutes a "normal" document for each language, flagging deviations as noise.

## Method

### Overall Architecture

A three-stage pipeline:
1. **Data collection**: Integrating MaLA (939 languages) + Fineweb (English) + Fineweb-2 (1,915 languages) + newly crawled Common Crawl (2024.5–2024.11)
2. **Feature extraction**: Extracting an 8-dimensional statistical feature vector per document
3. **Anomaly detection filtering**: Feature standardization → Isolation Forest scoring → thresholding into clean/anomalous

### Key Designs

1. **Data Collection Strategy**:

    - Function: Integrate four complementary sources to maximize coverage with up-to-date content
    - MaLA contributes diversity beyond Common Crawl (especially for low-resource languages); Fineweb provides high-quality English; Fineweb-2 provides broad multilingual coverage; newly crawled CC (2024.5–11, 21.54 TB) ensures data freshness
    - Final scale: 2,282 languages, 8.63B documents, 46.72 TB, 155 high/mid-resource languages, 159 writing systems

2. **8-Dimensional Document Features**:

    - Word count $n_w(t)$: document length
    - Character repetition rate $r_c(t)$: detects encoding errors and garbage content
    - Word repetition rate $r_w(t)$: detects templated or cyclic content
    - Special character ratio $r_s(t)$: computed using language-specific symbol lists
    - Stop-word ratio $r_{stop}(t)$: multilingual stop-word lists from Fineweb-2
    - Flagged-word ratio $r_{flag}(t)$: toxic/profane vocabulary
    - Language identification confidence $s_{lid}(t)$: GlotLID (supports 2,000+ languages)
    - Perplexity $s_{ppl}(t)$: KenLM (one model per language, trained on Wikipedia)
    - Design Motivation: These features are computable across languages and capture distinct types of quality issues

3. **Anomaly Detection Cleaning**:

    - Function: Automatically identify low-quality documents via Isolation Forest
    - Features are standardized as $\tilde{x}_j = (x_j - \mu_j) / \sigma_j$ to remove scale differences
    - Isolation Forest anomaly score: $\phi(\tilde{\mathbf{x}}) = 2^{-h(\tilde{\mathbf{x}})/c(n)}$, where $h$ is the average isolation path length
    - scikit-learn default parameters are used globally, without per-language or per-feature tuning
    - Design Motivation: Anomaly detection naturally adapts to the data distribution of each language — the algorithm automatically fits to language-specific notions of "normal"

### Loss & Training
- Cleaned data is used for continual pretraining
- Evaluated models: LLaMA-3.2-1B, Qwen-2.5-7B, Aya-expanse-32B
- Benchmarks: FineTask (9 languages), SIB-200, Glot500-c, FLORES-200
- Compute: 32-core + 128 GB servers, Kubernetes orchestration with up to 100 parallel jobs

## Key Experimental Results

### Main Results: Cleaning Method Comparison (FineTask Benchmark)

| Cleaning Method | Gain vs. No Cleaning | Gain vs. Threshold-based |
|---|---|---|
| No cleaning (raw data) | baseline | — |
| Fixed threshold | +5–10% | baseline |
| **DCAD anomaly detection** | **+5–20%** | **+3–10%** |

### Ablation Study: Anomaly Detection Algorithms

| Algorithm | Performance | Notes |
|---|---|---|
| Isolation Forest | Best | Suited for high-dimensional sparse data |
| LOF | Second | Local density estimation |
| One-Class SVM | Moderate | Sensitive to hyperparameters |

### Key Findings
- **Previously cleaned datasets still contain substantial noise**: Re-cleaning MaLA, Fineweb, and Fineweb-2 removes approximately 7.69% of documents, confirming the inadequacy of prior cleaning
- **Anomaly detection significantly outperforms threshold-based filtering**: threshold methods require manual tuning per language and feature, whereas anomaly detection adapts automatically
- **Low-resource languages benefit most**: gains on SIB-200 and Glot500-c are especially pronounced for low-resource languages
- **Feature visualization** validates the cleaning effect: anomalous and normal documents exhibit clear separation in 8-dimensional feature space, particularly along LID score and PPL score dimensions
- Broad geographic coverage: Africa (28.6%), Papua New Guinea (26.3%), Eurasia (23.8%)

## Highlights & Insights
- The **reformulation of "data cleaning as anomaly detection"** is the central insight: conceptually simple yet practically effective — it requires no explicit definition of what clean data looks like, only what constitutes an outlier. Anomaly detection algorithms naturally handle multimodal distributions, making them ideal for multilingual settings
- **Impressive scale**: 2,282 languages, 46.72 TB, 155 high/mid-resource languages — currently the most comprehensive multilingual dataset directly usable for training
- **Solid engineering contribution**: the entire pipeline is open-sourced with detailed documentation covering data collection, cleaning, and evaluation

## Limitations & Future Work
- Isolation Forest is applied with default parameters, which may be suboptimal for certain languages — per-language or per-script hyperparameter tuning warrants exploration
- The perplexity feature depends on Wikipedia data — a default value of 500 is used for low-resource languages with insufficient Wikipedia coverage, potentially degrading cleaning quality
- The 8-dimensional feature set is manually designed; richer features (e.g., semantic coherence, topical relevance) remain unexplored
- The cleaning step removes 7.69% of documents, but false positive/negative rates are not analyzed — high-quality documents may be incorrectly discarded or low-quality ones retained
- Writing systems are predominantly Latin-script (79.4%); the applicability of the feature set to complex scripts such as CJK requires further validation

## Related Work & Insights
- **vs. Fineweb-2**: Fineweb-2 covers 1,915 languages but only 10 high-resource ones; threshold fine-tuning for 1,000+ languages incurs substantial computational cost. DCAD-2000 replaces threshold tuning with anomaly detection, offering better scalability
- **vs. CulturaX/Madlad-400**: Based on older CC snapshots with insufficient cleaning. DCAD-2000 uses the latest CC (2024-46) with more thorough cleaning
- **vs. MaLA**: MaLA covers 939 languages but only 1 high-resource language. DCAD-2000 incorporates MaLA as a subset while substantially extending coverage

## Rating
- Novelty: ⭐⭐⭐⭐ The "cleaning as anomaly detection" reformulation is simple yet effective, with particular value in multilingual settings
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive comparison across three LLMs, four benchmarks, multiple cleaning strategies, and anomaly detection algorithms
- Writing Quality: ⭐⭐⭐⭐ Clear structure and detailed statistics, though the mathematical derivations are relatively straightforward
- Value: ⭐⭐⭐⭐⭐ The dataset itself is of immense value to the community, especially for low-resource language research

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] EPHAD: An Evidence-Based Post-Hoc Adjustment Framework for Anomaly Detection Under Data Contamination](an_evidence-based_post-hoc_adjustment_framework_for_anomaly_detection_under_data.md)
- [\[NeurIPS 2025\] ReCon: Region-Controllable Data Augmentation with Rectification and Alignment for Object Detection](recon_region-controllable_data_augmentation_with_rectification_and_alignment_for.md)
- [\[NeurIPS 2025\] BurstDeflicker: A Benchmark Dataset for Flicker Removal in Dynamic Scenes](burstdeflicker_a_benchmark_dataset_for_flicker_removal_in_dynamic_scenes.md)
- [\[ICCV 2025\] Kaputt: A Large-Scale Dataset for Visual Defect Detection](../../ICCV2025/object_detection/kaputt_a_large-scale_dataset_for_visual_defect_detection.md)
- [\[CVPR 2026\] MMR-AD: A Large-Scale Multimodal Dataset for Benchmarking General Anomaly Detection with MLLMs](../../CVPR2026/object_detection/mmrad_multimodal_anomaly_detection.md)

</div>

<!-- RELATED:END -->

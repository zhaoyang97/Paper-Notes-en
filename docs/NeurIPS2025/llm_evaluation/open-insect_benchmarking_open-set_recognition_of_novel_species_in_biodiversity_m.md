---
title: >-
  [Paper Note] Open-Insect: Benchmarking Open-Set Recognition of Novel Species in Biodiversity Monitoring
description: >-
  [NeurIPS 2025][LLM Evaluation][open-set recognition] This paper introduces Open-Insect — the first large-scale fine-grained open-set recognition benchmark for insect species discovery…
tags:
  - "NeurIPS 2025"
  - "LLM Evaluation"
  - "open-set recognition"
  - "fine-grained classification"
  - "insect species discovery"
  - "OOD detection"
  - "biodiversity monitoring"
date: 2026-05-08
content_hash: d05d92b27969e64d
---

# Open-Insect: Benchmarking Open-Set Recognition of Novel Species in Biodiversity Monitoring

**Conference**: NeurIPS 2025
**arXiv**: [2503.01691](https://arxiv.org/abs/2503.01691)
**Code**: [GitHub](https://github.com/RolnickLab/Open-Insect)
**Area**: Open-Set Recognition · Biodiversity
**Keywords**: open-set recognition, fine-grained classification, insect species discovery, OOD detection, biodiversity monitoring

## TL;DR

This paper introduces Open-Insect — the first large-scale fine-grained open-set recognition benchmark for insect species discovery, spanning three geographic regions and three types of open-set splits. It systematically evaluates 38 OSR algorithms, finding that simple posterior methods (e.g., MSP) remain strong baselines in fine-grained settings, and demonstrates the critical role of domain-relevant auxiliary data in improving OSR performance.

## Background & Motivation

Machine learning is increasingly important for species identification and biodiversity monitoring, yet existing classifiers typically assume a closed-world setting — that inference will only encounter categories seen during training. This assumption is severely violated in biodiversity applications for three reasons:

**Large numbers of undescribed species**: Approximately 86% of terrestrial and 91% of marine species on Earth have yet to be scientifically described.

**Incomplete data coverage**: Only about 65% of described species have records in the GBIF database.

**Regional model limitations**: Models trained on species checklists from specific regions cannot correctly handle species outside their scope.

Open-set recognition (OSR) — simultaneously classifying known categories and detecting unknown ones — is therefore critical for automated species discovery and invasive species detection. However, existing OSR benchmarks are largely derived from ImageNet and fail to reflect the fine-grained characteristics and long-tailed distributions of biodiversity data. The few benchmarks that include biological categories are small in scale and focus primarily on well-studied vertebrates. Insects account for more than two-thirds of all animal species, with over 80% remaining undescribed, making them an ideal subject of study.

## Method

### Overall Architecture

Open-Insect is built upon the AMI dataset, containing images of 5,364 moth species and 12 classes of non-moth arthropods across three geographic regions: Northeastern North America (NE-America), Western Europe (W-Europe), and Central America (C-America). The core design principle is to use geographic metadata to study species detection difficulty under varying degrees of semantic shift.

### Key Designs

1. **Three open-set split types**:

    - **Local moths (O-L)**: For each region, the geographic boundary is slightly expanded in latitude and longitude. Species recorded within this expanded area but absent from the closed set are selected, simulating "locally unrecorded but potentially present novel species." Based on Tobler's First Law of Geography, these species are taxonomically closer to the closed set, making them the hardest to detect.
    - **Non-local moths (O-NL)**: All moth species from Australia are included (geographic isolation ensures species distinctiveness), simulating invasive or introduced species.
    - **Non-moths (O-NM)**: 35,000 images of non-moth arthropods are randomly sampled from the AMI-GBIF dataset, simulating non-target species encountered in monitoring systems.

2. **Auxiliary dataset construction**:
   To enable fair comparison among OSR methods requiring auxiliary data, an auxiliary training set is explicitly constructed for each region. All species appearing in the closed set, local open set, or non-local open set are excluded, along with genus-level overlapping species to prevent shortcut learning. NE-America and W-Europe each use 8,000 species with 20 images per species; C-America uses 4,000 species with 20 images per species.

3. **BCI real-world validation set (O-B)**:
   197 field-collected images from Barro Colorado Island, Panama, are included, covering 133 open-set species. Among these, 59 species are potentially new to science based on DNA barcoding comparisons (>1.5% sequence divergence), providing a highly realistic test scenario.

### Proposed Baseline Methods

Two simple methods leveraging auxiliary data are proposed:

- **NovelBranch**: The original $C$-dimensional classification head is retained, and an additional $C+1$-dimensional head is trained, treating all auxiliary species as a single "novel class."
- **Extended**: An additional $C+A$-dimensional classification head is trained, treating each species in the auxiliary set as a separate class.

Both methods use the closed-set classification head to compute OSR scores at test time, leaving ID accuracy unaffected.

### Evaluation Protocol

- Open-set detection: AUROC (open-set as positive, closed-set as negative)
- Closed-set classification: Top-1 accuracy
- BCI real-world scenario: TPR@FPR=5%

## Key Experimental Results

### Main Results: Benchmarking 38 OSR Methods

| Method Type | Representative Method | NE-A (L/NL/NM) | W-E (L/NL/NM) | C-A (L/NL/NM) |
|------------|----------------------|-----------------|-----------------|-----------------|
| Posterior | MSP | 86.7/93.6/94.3 | 86.0/93.2/94.5 | 86.7/88.0/89.0 |
| Posterior | TempScale | 86.8/93.7/94.1 | 86.0/93.5/94.5 | 85.0/86.5/89.3 |
| Training Regularization | LogitNorm | 80.6/87.3/95.3 | 80.8/87.7/95.6 | 87.6/89.5/90.5 |
| Auxiliary Data | Energy | **87.4/95.1**/92.5 | 84.6/**94.8**/89.5 | **90.0/93.8**/91.2 |
| Auxiliary Data | NovelBranch | 85.5/94.1/90.0 | 83.9/93.8/91.7 | 87.8/89.7/91.1 |
| Auxiliary Data | MixOE | 86.2/92.5/94.4 | 85.2/91.9/94.1 | 86.2/87.9/90.1 |

### Ablation Study

| Configuration | L AUROC | NL AUROC | NM AUROC | Notes |
|---------------|---------|----------|----------|-------|
| Energy + OI-CA auxiliary | 90.07 | 93.70 | 91.33 | Domain-relevant auxiliary data |
| Energy + TinyImageNet | 85.25 | 88.16 | 93.85 | Generic auxiliary data; L drops 4.82pp |
| Original images (MSP) | 80.01 | — | — | Standard inference |
| Insect masked | 46.57 | — | — | Near-random; model relies on insect features |
| Background masked | 76.01 | — | — | Slight drop; insect features are dominant |

### Key Findings

1. **Local species are hardest to detect**: Across all three regions, the local moth open set consistently yields the lowest AUROC, validating the hypothesis that geographic proximity implies taxonomic proximity and thus greater difficulty.
2. **MSP and its variants remain strong baselines**: In fine-grained OSR settings, complex posterior methods do not consistently outperform simple MSP; some methods (e.g., GradNorm, DICE) perform poorly.
3. **Auxiliary data quality is critical**: Domain-relevant auxiliary data (moths from the same order but different species) substantially outperforms generic data (TinyImageNet), and species diversity matters more than the number of images per species.
4. **Pre-trained weights are beneficial**: ImageNet pre-trained initialization improves OSR performance on smaller datasets; BioCLIP weights on ViT-B-16 generally outperform ImageNet-1K weights.
5. **Benchmark reflects real-world scenarios**: Methods achieving higher AUROC on standard open sets also tend to perform better on the BCI dataset of potentially undescribed species.

## Highlights & Insights

1. **Elegant dataset design**: Geographic metadata is used to construct open-set splits with progressively increasing difficulty, while unified auxiliary data enables fair method comparison — addressing a key limitation of existing benchmarks.
2. **Real-world species discovery validation**: Inclusion of BCI data containing potentially new-to-science species enables end-to-end validation, bridging the gap between benchmark performance and practical application.
3. **Interpretability experiments**: Masking experiments confirm that models rely on the morphological features of insects themselves rather than background cues.
4. **Open science**: The dataset and code are publicly available.

## Limitations & Future Work

- Only high-resolution images are evaluated; low-resolution camera trap scenarios are not included.
- Training regularization and auxiliary data methods are run with a single training seed due to computational constraints.
- The benchmark focuses solely on moths; generalizability to other insect taxa remains to be verified.
- Species identification models carry a risk of misuse in illegal wildlife trade.

## Related Work & Insights

- **OSR/OOD benchmarks**: OpenOOD, SSB, COOD, iNat21-OSR
- **Biodiversity ML**: BioCLIP, BIOSCAN, iNaturalist
- **OSR methods**: MSP, ODIN, Energy, MixOE
- Insight: In other fine-grained domains (e.g., medical imaging, industrial defect detection), the quality and domain relevance of auxiliary data may be equally critical.

## Rating

- Novelty: ⭐⭐⭐⭐ — First large-scale fine-grained OSR benchmark for insects, filling an important gap.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Systematic evaluation of 38 methods, covering ablations, interpretability, and real-world validation.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear motivation, rigorous dataset construction logic, and depth in experimental design.
- Value: ⭐⭐⭐⭐ — Directly advances both biodiversity monitoring and OSR method evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Unlocking Transfer Learning for Open-World Few-Shot Recognition](unlocking_transfer_learning_for_open-world_few-shot_recognition.md)
- [\[NeurIPS 2025\] Towards Implicit Aggregation: Robust Image Representation for Place Recognition in the Transformer Era](towards_implicit_aggregation_robust_image_representation_for_place_recognition_i.md)
- [\[NeurIPS 2025\] Benchmarking is Broken — Don't Let AI be its Own Judge](benchmarking_is_broken_--_dont_let_ai_be_its_own_judge.md)
- [\[NeurIPS 2025\] CodeAssistBench (CAB): Dataset & Benchmarking for Multi-turn Chat-Based Code Assistance](codeassistbench_cab_dataset_benchmarking_for_multi-turn_chat-based_code_assistan.md)
- [\[NeurIPS 2025\] Ineq-Comp: Benchmarking Human-Intuitive Compositional Reasoning in Automated Theorem Proving on Inequalities](ineq-comp_benchmarking_human-intuitive_compositional_reasoning_in_automated_theo.md)

</div>

<!-- RELATED:END -->

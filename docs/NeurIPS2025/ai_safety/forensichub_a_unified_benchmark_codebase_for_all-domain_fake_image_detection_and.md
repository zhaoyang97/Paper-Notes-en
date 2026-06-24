---
title: >-
  [Paper Note] ForensicHub: A Unified Benchmark & Codebase for All-Domain Fake Image Detection and Localization
description: >-
  [NeurIPS 2025][AI Safety][Image forgery detection] ForensicHub introduces the first unified benchmark platform spanning all domains (Deepfake/IMDL/AIGC/Document Tampering) for fake image detection and localization, encompassing 4 tasks, 23 datasets, 42 models, 6 backbone networks, and 11 GPU-accelerated evaluation metrics. Through a modular architecture and adapter design, it bridges domain silos and conducts 16 cross-domain evaluations to derive 8 key insights.
tags:
  - "NeurIPS 2025"
  - "AI Safety"
  - "Image forgery detection"
  - "unified benchmark"
  - "Deepfake"
  - "AIGC detection"
  - "document tampering"
date: 2026-05-08
content_hash: 3f490aaf2da7530c
---

# ForensicHub: A Unified Benchmark & Codebase for All-Domain Fake Image Detection and Localization

**Conference**: NeurIPS 2025
**arXiv**: [2505.11003](https://arxiv.org/abs/2505.11003)  
**Code**: [GitHub](https://github.com/scu-zjz/ForensicHub)  
**Area**: AI Security
**Keywords**: Image forgery detection, unified benchmark, Deepfake, AIGC detection, document tampering

## TL;DR
ForensicHub introduces the first unified benchmark platform spanning all domains (Deepfake/IMDL/AIGC/Document Tampering) for fake image detection and localization, encompassing 4 tasks, 23 datasets, 42 models, 6 backbone networks, and 11 GPU-accelerated evaluation metrics. Through a modular architecture and adapter design, it bridges domain silos and conducts 16 cross-domain evaluations to derive 8 key insights.

## Background & Motivation

**Background**: Fake image detection and localization (FIDL) has diverged into four relatively independent subfields: Deepfake detection (face manipulation), IMDL (natural image manipulation detection/localization), AIGC detection (AI-generated image detection), and document image forgery localization (Doc).

**Limitations of Prior Work**: Each domain independently constructs its own datasets, models, and evaluation protocols, resulting in severe "domain silos." Although single-domain benchmarks such as DeepfakeBench and IMDLBenCo exist, a unified cross-domain benchmark is absent. Existing benchmarks are mutually incompatible — DeepfakeBench is coupled to face preprocessing, IMDLBenCo requires pixel-level masks, and AIGCDetectBenchmark does not support multi-GPU evaluation.

**Key Challenge**: The four domains differ substantially in data formats, model outputs, and evaluation standards, yet share many underlying techniques (e.g., pretrained backbones, low-level visual features, contrastive learning). This fragmentation leads to redundancy and impedes progress.

**Goal**: To construct a sufficiently flexible and extensible unified benchmark that is compatible with existing benchmarks, fills the gap left by the absence of benchmarks for AIGC and Doc domains, and provides a cross-domain evaluation protocol.

**Key Insight**: Modular, configuration-driven architecture with an adapter pattern for compatibility with existing benchmarks.

**Core Idea**: Through a composable Dataset–Transform–Model–Evaluator four-module architecture and adapter design, ForensicHub achieves, for the first time, unified training, testing, and cross-domain evaluation across all four FIDL domains.

## Method

### Overall Architecture

ForensicHub adopts a modular architecture comprising four core components: Datasets (data loading) → Transforms (preprocessing/augmentation) → Models (detection models) → Evaluators (evaluation metrics). Users can construct training/testing pipelines via YAML configuration without writing any code.

### Key Designs

1. **Modular Configuration-Driven Architecture**:

    - Function: Decomposes the forensic pipeline into interchangeable components, supporting arbitrary combinations of Dataset + Model + Evaluator across any domain.
    - Mechanism: Each Dataset returns a unified field specification; Models interface through a unified output API; Evaluators cover both image-level and pixel-level metrics, with all 11 metrics GPU-accelerated.
    - Design Motivation: Addresses the inconsistency in data formats, model output types, and evaluation standards across domains.

2. **Adapter-based Integration**:

    - Function: Seamlessly integrates DeepfakeBench and IMDLBenCo into the unified framework.
    - Mechanism: An adapter layer converts models and datasets from existing benchmarks into the ForensicHub unified format without requiring code rewrites.
    - Coverage: 27/34 detectors from DeepfakeBench (22 supporting cross-domain evaluation) and all 10 models from IMDLBenCo.
    - Design Motivation: Avoids redundant reimplementation and allows existing community assets to enter the unified evaluation framework directly.

3. **IFF-Protocol (Image Forensics Fusion Protocol)**:

    - Function: Establishes a unified cross-domain training and testing protocol.
    - Mechanism: Training data are mixed from all four domains (FaceForensics++, CASIAv2, GenImage, multiple document datasets), with equal-quantity sampling per epoch from each domain (based on the smallest dataset, CASIAv2, with 12,641 samples); evaluation is performed directly on each domain's test set without fine-tuning.
    - Design Motivation: Explores the feasibility of a general-purpose forensic model and assesses generalization to unseen domains.

4. **New AIGC and Document Benchmarks**:

    - AIGC Benchmark: Trained on DiffusionForensics, cross-domain tested on 8 generative models from GenImage.
    - Document Benchmark: Provides in-domain evaluation and a Doc Protocol (trained only on Doctamper, tested on four external datasets).

### Platform Scale
Full platform: 4 tasks, 23 datasets, 42 models (including 3 reproduced from scratch), 6 backbone networks, 11 GPU-accelerated metrics, and 16 cross-domain evaluations.

## Key Experimental Results

### AIGC Detection Benchmark: AUC (Trained on DiffusionForensics)

| Method | In-Domain | ADM | BigGAN | Midjourney | SD V1.4 | Cross-Domain Avg. |
|--------|-----------|-----|--------|-----------|---------|-------------------|
| DualNet | 1.000 | 0.999 | 0.917 | 0.850 | 0.813 | 0.890 |
| IML-ViT (IMDL) | 1.000 | 0.959 | 0.915 | 0.809 | 0.892 | **0.896** |
| HiFiNet | 1.000 | 1.000 | 0.841 | 0.721 | 0.677 | 0.816 |
| UnivFD | 0.995 | 0.840 | 0.969 | 0.543 | 0.708 | 0.792 |
| Trufor (IMDL) | 1.000 | 0.975 | 0.932 | 0.708 | 0.868 | 0.875 |

### Document Forgery Detection Benchmark: Binary-F1 (Doc Protocol, Trained on Doctamper Only)

| Method | Doctamper In-Domain | T-SROIE Cross-Domain | OSTF Cross-Domain | TPIC-13 Cross-Domain | Cross-Domain Avg. |
|--------|--------------------|--------------------|------------------|--------------------|--------------------|
| FFDN | **0.813** | 0.533 | 0.241 | 0.357 | **0.300** |
| DTD | 0.743 | 0.525 | 0.124 | 0.284 | 0.247 |
| Cat-Net (IMDL) | 0.722 | **0.609** | 0.178 | 0.343 | 0.298 |
| PSCC-Net (IMDL) | 0.425 | 0.517 | **0.441** | **0.550** | **0.408** |

### Key Findings (Selected from 8 Core Insights)
- AIGC models perform well on similar diffusion-based models but generalize poorly to models from different paradigms, such as Midjourney and Wukong.
- IMDL models (e.g., IML-ViT, Trufor) can match or even surpass dedicated AIGC detectors in cross-domain AIGC detection, suggesting that low-level forgery traces possess cross-domain generality.
- JPEG compression priors (DCT coefficients/quantization tables) are critical in document forensics — FFDN, DTD, and Cat-Net all leverage such features.
- PSCC-Net's progressive spatial modeling achieves the best cross-domain generalization in the document domain.
- In-domain performance and cross-domain generalization are often not proportional; high in-domain F1 does not guarantee good cross-domain transfer.

## Highlights & Insights
- **A Barrier-Breaking Contribution**: ForensicHub is the first to unify all four FIDL subfields; the adapter design enables zero-modification integration of existing benchmark assets, substantially reducing community migration costs.
- **Surprising Cross-Domain Performance of IMDL Models**: IMDL detectors perform unexpectedly well on AIGC and document domains, suggesting the existence of a low-level forgery feature space with cross-domain generality and pointing toward a promising direction for building general-purpose forensic models.
- **Zero-Code Evaluation via Configuration-Driven Design**: Any combination of Dataset + Model + Evaluator can be composed via YAML configuration, greatly lowering the engineering barrier for benchmark experiments.

## Limitations & Future Work
- Video-level Deepfake detection is not yet included (only frame-level).
- The equal-quantity sampling strategy in the IFF-Protocol may not represent the optimal domain mixing ratio.
- Cross-domain generalization in some domains remains poor (e.g., AIGC to Midjourney), necessitating improved domain generalization methods.
- Joint detection-and-localization models under a multi-task learning framework have not been explored.

## Related Work & Insights
- **vs. DeepfakeBench (Yan et al., 2023)**: Focuses on the Deepfake domain alone and is tightly coupled to face preprocessing. ForensicHub integrates it via an adapter and extends coverage to four domains.
- **vs. IMDLBenCo (Ma et al., 2024)**: Focuses on the IMDL domain alone and requires mask outputs. ForensicHub similarly integrates it through an adapter.
- **vs. AIGCDetectBenchmark**: A repository-level collection of experiments lacking cross-domain design and unified evaluation. ForensicHub provides a complete modular framework.

## Rating
- Novelty: ⭐⭐⭐⭐ — Substantial contribution in terms of systems engineering, though methodological innovation is limited (primarily benchmark construction).
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 42 models, 23 datasets, and 16 cross-domain evaluations represent an unprecedented scale.
- Writing Quality: ⭐⭐⭐⭐ — Structure is clear, though the volume of tables and data imposes a heavy reading burden.
- Value: ⭐⭐⭐⭐⭐ — A milestone contribution to the FIDL field; likely to become a standard infrastructure for the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Noise-Assisted Prompt Learning for Image Forgery Detection and Localization](../../ECCV2024/ai_safety/noise-assisted_prompt_learning_for_image_forgery_detection_and_localization.md)
- [\[CVPR 2026\] Omni-Fake: Benchmarking Unified Multimodal Social Media Deepfake Detection](../../CVPR2026/ai_safety/omni-fake_benchmarking_unified_multimodal_social_media_deepfake_detection.md)
- [\[NeurIPS 2025\] Towards Unsupervised Open-Set Graph Domain Adaptation via Dual Reprogramming](towards_unsupervised_open-set_graph_domain_adaptation_via_dual_reprogramming.md)
- [\[NeurIPS 2025\] Revisiting Logit Distributions for Reliable Out-of-Distribution Detection](revisiting_logit_distributions_for_reliable_out-of-distribution_detection.md)
- [\[NeurIPS 2025\] Redundancy-Aware Test-Time Graph Out-of-Distribution Detection](redundancy-aware_test-time_graph_out-of-distribution_detection.md)

</div>

<!-- RELATED:END -->

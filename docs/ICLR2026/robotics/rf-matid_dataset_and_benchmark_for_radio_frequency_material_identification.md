---
title: >-
  [Paper Note] RF-MatID: Dataset and Benchmark for Radio Frequency Material Identification
description: >-
  [ICLR 2026][Robotics][RF sensing] This paper introduces RF-MatID, the first open-source large-scale RF material identification dataset with wide frequency coverage (4–43.5 GHz) and diverse geometric perturbations, comprising 16 fine-grained material categories (5 superclasses) and 142K samples. A comprehensive benchmark is established across 9 deep learning models, 5 frequency protocols, and 7 data split settings.
tags:
  - ICLR 2026
  - Robotics
  - RF sensing
  - material identification
  - UWB-mmWave
  - dataset benchmark
  - embodied AI
date: 2026-05-08
content_hash: feee710cd527fc94
---

# RF-MatID: Dataset and Benchmark for Radio Frequency Material Identification

**Conference**: ICLR 2026
**arXiv**: [2601.20377](https://arxiv.org/abs/2601.20377)
**Code**: Available (project page)
**Area**: AI Safety / Embodied AI / RF Sensing
**Keywords**: RF sensing, material identification, UWB-mmWave, dataset benchmark, embodied AI

## TL;DR
This paper introduces RF-MatID, the first open-source large-scale RF material identification dataset with wide frequency coverage (4–43.5 GHz) and diverse geometric perturbations, comprising 16 fine-grained material categories (5 superclasses) and 142K samples. A comprehensive benchmark is established across 9 deep learning models, 5 frequency protocols, and 7 data split settings.

## Background & Motivation

**Background**: Material identification is a fundamental capability for embodied AI, currently dominated by optical sensors (cameras, hyperspectral imaging). RF-based methods exploit electromagnetic wave–material interactions to reveal intrinsic material properties (permittivity, conductivity, etc.), operating independently of illumination conditions and visual appearance.

**Limitations of Prior Work**: (1) All existing RF material datasets are proprietary, hindering fair algorithmic comparison; (2) COTS sensor frequency bands are narrow and fragmented (e.g., 77–81 GHz only), precluding systematic cross-band evaluation; (3) systematic evaluation of geometric perturbations (angle and distance variation) is absent, leaving real-world deployment robustness uncertain.

**Key Challenge**: Despite theoretical advantages of RF methods (strong penetration, illumination independence), the lack of research infrastructure (datasets and benchmarks) severely impedes the development and evaluation of learning-based approaches.

**Goal**: Construct the first open-source, wide-band, geometrically diverse RF material identification dataset and establish a complete benchmarking framework.

**Key Insight**: A custom UWB-mmWave sensing platform with continuous frequency coverage from 4 to 43.5 GHz is developed to systematically collect RF responses of 16 materials across varying distances (200–2000 mm) and angles (0–10°).

**Core Idea**: Enable standardized learning-based research in RF material identification through the first open-source wide-band RF dataset and systematic benchmark.

## Method

### Overall Architecture
RF-MatID is constructed at three levels: (1) **Data collection** — a custom UWB-mmWave sensing platform systematically acquires frequency-domain responses from 16 material classes over a distance–angle grid; (2) **Data processing** — dual-domain representation (frequency and time domains) with complex whitening normalization; (3) **Benchmark evaluation** — 5 frequency protocols × 7 data splits × 9 models.

### Key Designs

1. **Sensing Platform and Data Acquisition**:

    - **Function**: Wide-band RF signal acquisition covering 4–43.5 GHz.
    - **Mechanism**: A DRH40 double-ridge horn antenna paired with an MS46131A vector network analyzer collects complex responses $H(f_i) = I(f_i) + jQ(f_i)$ at 2048 frequency bins per sample. Each material is measured on a distance grid of 200–2000 mm (50 mm steps) × angle grid of 0–10° (1° steps), yielding 142K samples in total.
    - **Design Motivation**: The 39.5 GHz bandwidth far exceeds the previous maximum of 4 GHz, simultaneously capturing centimeter-wave (3–30 GHz) penetration information and millimeter-wave Q-band (30–50 GHz) surface sensitivity.

2. **Dual-Domain Data Representation**:

    - **Function**: Generate paired time-domain representations for each frequency-domain sample.
    - **Mechanism**: Frequency-domain data is represented as two-channel real tensors (real and imaginary parts); time-domain data is obtained via IFFT to produce signals of length 10240. Complex whitening is applied in the frequency domain to preserve phase relationships; standard normalization is used in the time domain.
    - **Design Motivation**: Experiments confirm that two-channel real representation outperforms complex-valued networks; frequency-domain captures frequency-selective attenuation while time-domain captures propagation delay.

3. **Frequency Protocols**:

    - **Function**: Define 5 frequency allocation schemes to evaluate recognition capability across different bands.
    - **Mechanism**: P1 full band 4–43.5 GHz; P2 millimeter-wave 30–43.5 GHz; P3 centimeter-wave 4–30 GHz; P4 legally permitted commercial bands in the United States; P5 legally permitted bands in China.
    - **Design Motivation**: This is the first benchmark to incorporate regulatory constraints on frequency usage, enabling results to directly inform compliant system design.

4. **Seven Data Split Settings**:

    - **Function**: Systematically evaluate model generalization across diverse deployment scenarios.
    - **Mechanism**: S1 standard random split (IID); S2 cross-distance OOD (mod1–3 covering different distance subsets); S3 cross-angle OOD.
    - **Design Motivation**: S1 assesses basic capability; S2/S3 assess robustness to distribution shift caused by sensor position variation in real-world deployment.

### Material Taxonomy
5 superclasses → 16 fine-grained categories: **Brick** (over-fired clay brick, lightweight porous brick, volcanic brick); **Glass** (transparent acrylic, tempered glass, white opaque acrylic); **Synthetic** (melamine-faced board, mineral fiber board, PVC board); **Wood** (cedar sleeper, lauan plywood, red oak plywood); **Stone** (permeable paving stone, engineered stone, granite, concrete).

## Key Experimental Results

### Main Results (Protocol 1, Full Band 4–43.5 GHz)

| Model | S1 (IID) | S2-mod1 (Cross-distance) | S3-mod1 (Cross-angle) | Notes |
|---|---|---|---|---|
| Baseline (proposed) | 99.57 | 86.62 | 98.89 | Simple CNN |
| LSTM-ResNet | 99.84 | 97.12 | 99.69 | Best IID |
| ConvNeXt | 99.51 | 79.10 | 98.85 | CV model |
| AirTac | 96.81 | 91.36 | 98.12 | RF-specific |
| Material-ID | 99.28 | 95.67 | 97.63 | RF-specific |

### Cross-Domain Robustness (S2 OOD, Protocol 1)

| Model | S2-mod1 | S2-mod2 | S2-mod3 | Notes |
|---|---|---|---|---|
| LSTM-ResNet | 97.12 | 49.95 | 71.00 | Large drop at mod2 |
| AirTac | 91.36 | 86.95 | 65.41 | Most robust across distances |
| ConvNeXt | 79.10 | 64.19 | 63.52 | Poor cross-domain generalization |

### Key Findings
- **IID performance is near-saturated**: Most models exceed 99% under S1, offering little discriminability; with sufficient data, RF material identification is not inherently difficult.
- **Cross-distance domain shift is the primary challenge**: Accuracy under S2-mod2 drops sharply to 50–87% across models, indicating that distance-induced signal attenuation poses the greatest deployment challenge.
- **AirTac is most robust under OOD settings**: Although not the top IID performer, it exhibits the smallest OOD degradation, suggesting that RF-specific architectural design benefits robustness.
- **Frequency domain vs. time domain**: Frequency-domain two-channel representation outperforms time-domain representation and complex-valued network processing.
- **Compliant frequency bands remain viable**: Performance under P4/P5 (legally permitted bands) is lower than the full band but remains practically usable, validating real-world deployment feasibility.

## Highlights & Insights
- **Landmark significance as the first open-source RF material dataset**: Analogous to ImageNet's impact on visual recognition research, RF-MatID has the potential to standardize RF sensing research. The open-source policy constitutes the most significant contribution.
- **Frequency protocol design incorporates regulatory constraints**: This is the first benchmark to integrate legal compliance into its design, directly facilitating the transition from research to deployment.
- **Systematic geometric perturbation**: The grid-based distance and angle acquisition methodology can be adopted by other sensing modalities (e.g., LiDAR, ultrasound).

## Limitations & Future Work
- **Limited material variety**: 16 categories remain insufficient; real-world environments encompass far more materials, including liquids, textiles, and metals.
- **Single sensing platform**: All data originate from one hardware setup; cross-device generalization is unknown.
- **Controlled indoor environment**: Multipath interference, occlusion, and other real-world factors are not considered.
- **Narrow angle range**: Only 0–10° is covered, whereas robotic manipulation may require 0–90°.
- **Absence of multimodal benchmark**: As an embodied AI dataset, no paired visual or tactile data are provided for multimodal fusion research.

## Related Work & Insights
- **vs. VNA-based datasets (he2022accurate, shanbhag2023contactless)**: Broader frequency coverage (39.5 GHz vs. 4 GHz) and openly available.
- **vs. Wi-Fi/RFID datasets**: Higher signal quality (coherent transceiver) but requires dedicated hardware.
- Provides foundational data resources for material perception research in embodied AI, and can serve as an RF-branch baseline.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First open-source wide-band RF material identification dataset, filling a critical gap.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ 9 models × 5 protocols × 7 splits, comprehensive coverage.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with thorough background exposition.
- **Value**: ⭐⭐⭐⭐ Dataset contribution provides lasting impact to the field.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] TouchFormer: A Robust Transformer-based Framework for Multimodal Material Perception](../../AAAI2026/robotics/touchformer_a_robust_transformer-based_framework_for_multimodal_material_percept.md)
- [\[AAAI 2026\] Human-Centric Open-Future Task Discovery: Formulation, Benchmark, and Scalable Tree-Based Search](../../AAAI2026/robotics/human-centric_open-future_task_discovery_formulation_benchmark_and_scalable_tree.md)
- [\[ICCV 2025\] GUIOdyssey: A Comprehensive Dataset for Cross-App GUI Navigation on Mobile Devices](../../ICCV2025/robotics/guiodyssey_a_comprehensive_dataset_for_cross-app_gui_navigation_on_mobile_device.md)
- [\[NeurIPS 2025\] MMTU: A Massive Multi-Task Table Understanding and Reasoning Benchmark](../../NeurIPS2025/robotics/mmtu_a_massive_multi-task_table_understanding_and_reasoning_benchmark.md)
- [\[NeurIPS 2025\] SutureBot: A Precision Framework & Benchmark for Autonomous End-to-End Suturing](../../NeurIPS2025/robotics/suturebot_a_precision_framework_benchmark_for_autonomous_end-to-end_suturing.md)

<!-- RELATED:END -->

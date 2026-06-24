---
title: >-
  [Paper Note] EarthDial: Turning Multi-sensory Earth Observations to Interactive Dialogues
description: >-
  [CVPR 2025][Remote Sensing][Remote Sensing VLM] This work proposes EarthDial, a conversational vision-language model tailored for Earth Observation (EO) data. It supports the unified understanding of multispectral (SAR/NIR/infrared), multi-temporal, and multi-resolution remote sensing imagery. Trained on an 11.11 million instruction-tuning dataset, it outperforms existing remote sensing VLMs across 44 downstream datasets.
tags:
  - "CVPR 2025"
  - "Remote Sensing"
  - "Remote Sensing VLM"
  - "Earth Observation"
  - "Multispectral"
  - "Multi-temporal"
  - "Instruction Tuning"
date: 2026-05-08
content_hash: df5d6fdfd1ddaf7b
---

# EarthDial: Turning Multi-sensory Earth Observations to Interactive Dialogues

**Conference**: CVPR 2025  
**arXiv**: [2412.15190](https://arxiv.org/abs/2412.15190)  
**Code**: [https://github.com/hiyamdebary/EarthDial](https://github.com/hiyamdebary/EarthDial)  
**Area**: Remote Sensing / Multimodal VLM  
**Keywords**: Remote Sensing VLM, Earth Observation, Multispectral, Multi-temporal, Instruction Tuning

## TL;DR
This work proposes EarthDial, a conversational vision-language model tailored for Earth Observation (EO) data. It supports the unified understanding of multispectral (SAR/NIR/infrared), multi-temporal, and multi-resolution remote sensing imagery. Trained on an 11.11 million instruction-tuning dataset, it outperforms existing remote sensing VLMs across 44 downstream datasets.

## Background & Motivation

**Background**: General VLMs (e.g., GPT-4V) perform poorly on remote sensing (RS) data due to its unique geospatial, spectral, and temporal dimensions. Recently emerged RS VLMs (e.g., GeoChat, SkyEyeGPT) only support RGB optical imagery, lacking support for SAR, multispectral, and multi-temporal data.

**Limitations of Prior Work**: (1) Existing datasets are small in scale and only cover the RGB modality (at most ~1M pairs); (2) Lacking support for multispectral inputs (e.g., 13 bands of Sentinel-2, VH/VV bands of SAR); (3) Lacking support for temporal change analysis (e.g., change detection, temporal classification); (4) Lacking support for variable resolutions (ranging from 0.5m aerial imagery to 30m Landsat).

**Key Challenge**: The gap between the multimodal complexity of EO data (different sensors, varying resolutions, multi-temporal) and the limitation of existing VLMs that only handle fixed-resolution RGB images.

**Goal**: To build the first conversational VLM that unifiedly processes multi-resolution, multispectral, and multi-temporal remote sensing data.

**Key Insight**: (1) Construction of an 11M+ instruction dataset covering all modalities; (2) Design of a data fusion module to process non-RGB inputs; (3) A three-phase training strategy to progressively expand the model's capabilities.

**Core Idea**: To construct the first omni-modal VLM in the remote sensing domain using an 11M multimodal instruction dataset, an adaptive high-resolution/data-fusion module, and a three-phase training workflow.

## Method

### Overall Architecture
Based on the InternVL architecture (InternViT-300M vision encoder + Phi-3-mini LLM, totaling 4B parameters). Two key modules are introduced: an adaptive high-resolution module (which dynamically splits images of varying resolutions into 448x448 tiles plus a thumbnail) and a data fusion module (which processes 3 channels at a time through ViT, followed by feature aggregation and dimensionality reduction). Special tokens are used to distinguish different modalities and tasks.

### Key Designs

1. **Data Fusion Module**:

    - **Function**: Processes multispectral/SAR/temporal inputs with an arbitrary number of channels.
    - **Mechanism**: For multispectral inputs (e.g., the 13 bands of Sentinel-2), three channels are extracted at a time and fed into the ViT to extract features, and then all channel features are aggregated. The features are patch-encoded via the AnyRes module and downsampled using bilinear interpolation to reduce token counts before being concatenated with text embeddings and fed into the LLM. For RGB multi-temporal images, each frame is passed through the ViT independently and then stacked.
    - **Design Motivation**: To reuse the pre-trained RGB ViT to process multi-channel inputs, avoiding training a multispectral encoder from scratch.

2. **Three-stage Training Strategy**:

    - **Function**: Progressively expands the model's capabilities.
    - **Mechanism**: Stage 1 (Pre-training): Trains all parameters (ViT+MLP+LLM) on 7.6M image-text pairs (NAIP/Sentinel-2/Landsat/SkyScript) to achieve RS vision-language alignment. Stage 2 (RGB + Temporal Fine-tuning): Freezes the ViT while fine-tuning the MLP+LLM on tasks like classification, detection, VQA, and change detection, introducing temporal data fusion. Stage 3 (Multispectral + SAR Fine-tuning): Keeps the ViT frozen, fine-tunes the MLP+LLM, and incorporates the data fusion module to handle multispectral, SAR, RGBI, and hyperspectral data.
    - **Design Motivation**: To establish a solid foundation on a large amount of RGB data first and then progressively expand to more complex modalities, avoiding conflicts during simultaneous multimodal training.

3. **EarthDial-Instruct Dataset (11.11M)**:

    - **Function**: Provides instruction-tuning data covering all modalities.
    - **Mechanism**: Stage 1 data (7.6M): Labels are extracted from SatlasPretrain and SkyScript, and QA pairs are generated using InternLM-XComposer2, followed by triple filtering for sparse labels, cloud cover, and land coverage. Stage 2 data (1.8M): Integrates existing RS datasets (classification, detection, VQA, change detection, etc.). Stage 3 data (2.5M): Includes Sentinel-1 SAR, LCZ classification, tree species classification, methane plume detection, urban heat islands, etc.
    - **Design Motivation**: To make the dataset 6x larger than the previous largest RS instruction dataset, covering a significantly wider range of modalities.

### Loss & Training
Standard autoregressive cross-entropy loss. Stage 1: 8×A100, lr=4e-5, cosine schedule. Stage 2: 4×A100, 4 hours. Stage 3: Extended to multispectral/SAR inputs.

## Key Experimental Results

### Main Results

**Scene Classification (average across multiple datasets)**:

| Method | AID | RESISC45 | PatternNet | UCM | SIRI-WHU |
|------|-----|----------|------------|-----|----------|
| GeoChat | 88.2 | 82.6 | 94.3 | 87.6 | 87.2 |
| LHRS-Bot | 87.5 | 83.1 | 96.8 | 84.2 | - |
| **EarthDial** | **92.3** | **90.8** | **97.8** | **91.2** | **93.5** |

**Visual Question Answering (VQA)**:

| Method | RSVQA-LR | RSVQA-HR |
|------|----------|----------|
| GeoChat | 81.9 | 79.1 |
| **EarthDial** | **87.4** | **83.2** |

Achieves overall state-of-the-art performance across 44 downstream datasets (covering classification, detection, VQA, change detection, grounding tasks, and spanning RGB, SAR, and multispectral modalities).

### Ablation Study

| Configuration | Description |
|------|------|
| W/o Stage 1 Pre-training | Performance drops significantly, indicating that RS domain alignment is fundamental |
| Stage 2 training data volume vs. performance | Continuous improvements as data volume increases |
| Data fusion vs. independent channels | The fusion module significantly improves performance on multispectral tasks |

### Key Findings
- The three-stage training strategy performs better than end-to-end training; progressive expansion effectively avoids modality conflicts.
- With only 4B parameters, EarthDial outperforms larger models (e.g., EarthGPT), demonstrating that data quality + training strategy > model size.
- Exhibits zero-shot/few-shot capabilities on novel tasks such as SAR vessel detection and methane plume detection.
- The data fusion module for multispectral data is vastly superior to simple RGB conversion.

## Highlights & Insights
- **First omni-modal remote sensing VLM**: Supports RGB/SAR/multispectral/infrared + single-temporal/bi-temporal/multi-temporal + multi-resolution, offering a coverage far exceeding previous works.
- **Engineering value of the 11M instruction dataset**: The dataset itself represents a major contribution, with triple filtering ensuring quality and LLM-assisted generation guaranteeing scale.
- **Lightweight design**: Achieving state-of-the-art results with only 4B parameters (InternViT-300M + Phi-3-mini) underlines the importance of combining a compact model, high-quality data, and an optimal training strategy.

## Limitations & Future Work
- The data fusion module is relatively simple (processing 3 channels sequentially and concatenating them); more fine-grained cross-band attention mechanisms could be designed.
- The training data volume for multispectral/SAR data in Stage 3 (2.5M) is much smaller than the RGB data in Stage 1 (7.6M), which might lead to undertrained multispectral capabilities.
- Most of the 44 evaluation datasets are focused on RGB tasks, leaving a low ratio of multispectral/SAR tasks.
- Pixel-level segmentation output is not supported yet, with capabilities limited to box-level detection and textual descriptions.

## Related Work & Insights
- **vs GeoChat**: GeoChat only supports high-resolution RGB, whereas EarthDial supports omni-modality. EarthDial improves average scene classification by 5-8%.
- **vs EarthGPT/MMRS**: EarthGPT supports optical/SAR/infrared but lacks multispectral and multi-temporal features, whereas EarthDial provides comprehensive coverage.
- **vs SkyEyeGPT**: Dataset size is 968K vs. 11.11M, with significantly wider task coverage.
- Paves the way for broad application prospects in intelligent remote sensing analysis, disaster response, and environmental monitoring.

## Rating
- Novelty: ⭐⭐⭐⭐ First omni-modal remote sensing VLM, representing a systematic engineering innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive validation across 44 datasets.
- Writing Quality: ⭐⭐⭐⭐ Clear and comprehensive.
- Value: ⭐⭐⭐⭐⭐ Significant practical utility for the remote sensing community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] WildSAT: Learning Satellite Image Representations from Wildlife Observations](../../ICCV2025/remote_sensing/wildsat_learning_satellite_image_representations_from_wildlife_observations.md)
- [\[CVPR 2025\] MFogHub: Bridging Multi-Regional and Multi-Satellite Data for Global Marine Fog Detection and Forecasting](mfoghub_bridging_multi-regional_and_multi-satellite_data_for_global_marine_fog_d.md)
- [\[ICLR 2026\] Earth-Agent: Unlocking the Full Landscape of Earth Observation with Agents](../../ICLR2026/remote_sensing/earth-agent_unlocking_the_full_landscape_of_earth_observation_with_agents.md)
- [\[ICLR 2026\] Measuring the Intrinsic Dimension of Earth Representations](../../ICLR2026/remote_sensing/measuring_the_intrinsic_dimension_of_earth_representations.md)
- [\[CVPR 2026\] RoadGIE: Towards A Global-Scale Aerial Benchmark for Generalizable Interactive Road Extraction](../../CVPR2026/remote_sensing/roadgie_towards_a_global-scale_aerial_benchmark_for_generalizable_interactive_ro.md)

</div>

<!-- RELATED:END -->

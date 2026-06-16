---
title: >-
  [Paper Note] Mars-Bench: A Benchmark for Evaluating Foundation Models for Mars Science Tasks
description: >-
  [NeurIPS 2025][Segmentation][Mars science] This paper presents Mars-Bench — the first comprehensive benchmark for Mars science tasks, encompassing 20 datasets across three task types (classification, segmentation…
tags:
  - "NeurIPS 2025"
  - "Segmentation"
  - "Mars science"
  - "benchmark"
  - "foundation model"
  - "remote sensing segmentation"
  - "planetary science"
date: 2026-05-08
content_hash: 99f89fd33cbc3315
---

# Mars-Bench: A Benchmark for Evaluating Foundation Models for Mars Science Tasks

**Conference**: NeurIPS 2025
**arXiv**: [2510.24010](https://arxiv.org/abs/2510.24010)  
**Code**: [Available (GitHub)](https://github.com/kerner-lab/Mars-Bench)  
**Area**: Image Segmentation
**Keywords**: Mars science, benchmark, foundation model, remote sensing segmentation, planetary science

## TL;DR
This paper presents Mars-Bench — the first comprehensive benchmark for Mars science tasks, encompassing 20 datasets across three task types (classification, segmentation, and object detection). It systematically evaluates ImageNet-pretrained models, Earth observation foundation models, and vision-language models on Martian data, revealing significant gaps in current general-purpose models and calling for the development of Mars-specific foundation models.

## Background & Motivation
**Background**: Foundation models have achieved substantial progress in specialized domains such as medical imaging, Earth observation (EO), law, and astronomy. In EO, standardized benchmarks such as Geo-Bench and PANGAEA have driven rapid development. However, the Mars science community lacks a comparable standardized evaluation framework.

**Limitations of Prior Work**: Machine learning research on Mars-related tasks (crater detection, landmark classification, cone segmentation, etc.) suffers from inconsistent data formats, poor dataset interoperability, and the absence of standardized evaluation protocols. Existing Mars foundation model studies evaluate on only one or two downstream tasks, making it impossible to assess generalization comprehensively.

**Key Challenge**: Mars possesses an extraordinarily rich corpus of orbital and surface imagery (PB-scale images collected by MRO, Curiosity, Perseverance, etc.), yet the lack of standardized ML-ready formats and unified evaluation benchmarks leaves this data significantly underutilized.

**Goal**: To construct the first standardized Mars science benchmark covering multiple tasks, sensors, and geological features, while providing a unified evaluation framework and baseline models.

**Key Insight**: Drawing on the successful paradigm of Geo-Bench in the EO domain, the authors perform quality checks, corrections, and format unification on existing Mars datasets, coupled with comprehensive model evaluation.

**Core Idea**: By standardizing 20 Mars datasets and systematically evaluating diverse models, this work fills the gap in benchmarking for Mars science.

## Method

### Overall Architecture
Mars-Bench is a comprehensive benchmark comprising **20 datasets** covering three task types:
- **Classification (9 datasets)**: binary, multi-class, and multi-label settings
- **Segmentation (8 datasets)**: binary and multi-class segmentation
- **Object Detection (3 datasets)**: surface feature detection

Data sources include **2 Mars orbiters** (MRO, Mars Odyssey), **3 Mars rovers** (Curiosity, Opportunity, Spirit), and **6 imaging sensors** (HiRISE, CTX, THEMIS, Mastcam, Navcam, Pancam, etc.).

### Key Designs
1. **Ease of Use**: All datasets are unified into ML-ready formats with standardized data loading code; object detection annotations are provided in COCO, Pascal VOC, and YOLO formats.
2. **Expert Validation and Correction**: Developed in collaboration with planetary scientists; all segmentation datasets underwent expert validation, and several classification datasets were revised and corrected following correspondence with original authors.
3. **Standardized Data Splits**: All datasets provide consistent train/val/test splits to ensure evaluation consistency and reproducibility.
4. **Cross-Domain Partitioning**: Partitioned by sensor type, data modality, task category, or task source to support cross-sensor and cross-task domain transfer experiments.
5. **Scientific Coverage**: Encompasses key Martian geological features including craters, cones, boulders, landslides, dust devils, frost, and atmospheric dust.

### Evaluation Strategy
- **Three Training Configurations**: training from scratch (random initialization), frozen backbone for feature extraction, and full fine-tuning.
- **Systematic Multi-Model Evaluation**: Classification uses ResNet101/SqueezeNet1.1/InceptionV3/SwinV2-B/ViT-L/16; segmentation uses U-Net/DeepLabV3+/SegFormer/DPT; detection uses YOLO11/SSD/RetinaNet/Faster R-CNN.
- **Hyperparameter Search + 7 Seeds**: Grid search is performed for each model–dataset–training configuration combination; the optimal configuration is retrained with 7 random seeds, with IQM and bootstrap confidence intervals reported.

## Key Experimental Results

### Main Results

| Task | Best Model | Key Finding |
|------|------------|-------------|
| Classification (feature extraction) | ViT-L/16, SwinV2-B | Transformer architectures perform best; SqueezeNet consistently ranks last |
| Segmentation (feature extraction) | U-Net | Despite its simplicity, U-Net outperforms Transformer-based models on nearly all datasets |
| Object Detection (feature extraction) | YOLO11 | Best across all three datasets, though boulder and dust devil detection remain weak |

**VLM Evaluation Results (Gemini 2.0 Flash vs. GPT-4o Mini)**:

| Dataset | Gemini F1 | GPT F1 |
|---------|-----------|--------|
| mb-domars16k | 0.32 | 0.30 |
| mb-surface_cls | 0.44 | 0.41 |
| mb-frost_cls | 0.55 | 0.54 |
| mb-atmospheric_dust_cls_edr | 0.50 | 0.56 |
| mb-crater_multi_seg | 0.41 | 0.51 |
| mb-mars_seg_msl | 0.84 | 0.70 |

### Ablation Study

| Experiment | Key Conclusion |
|------------|----------------|
| Effect of training set size | Increasing data volume generally improves performance, but the rate of improvement and robustness vary significantly across datasets |
| EO models vs. ImageNet-pretrained models | ImageNet-pretrained ViT outperforms EO foundation models such as SatMAE/CROMA/Prithvi |
| Small VLMs (CLIP/SigLIP/SmolVLM) | Performance trends align with Gemini/GPT; all perform poorly on tasks requiring domain expertise |

### Key Findings
1. **U-Net remains a strong baseline for Mars segmentation**: Despite its simple architecture, it consistently outperforms Transformer-based models such as SegFormer and DPT. DPT exhibits extremely high variance and low reliability.
2. **EO-pretrained models fail to surpass ImageNet models**: A likely explanation is that ViT is pretrained on 14 million ImageNet images, offering far greater data diversity than EO models (≤1 million images). A significant domain gap also exists between Earth and Mars orbital imagery (Mars lacks vegetation, water bodies, and man-made structures).
3. **VLMs perform poorly on tasks requiring specialized domain knowledge**: Performance is acceptable on general categories (sand/rock/sky) but degrades substantially on fine-grained geological structures (crater types, Mars landmarks).
4. **Object detection is broadly challenging**: Small dataset sizes, few objects per image, and low contrast in grayscale imagery are the primary bottlenecks.

## Highlights & Insights
- **Filling a Critical Gap**: The first standardized ML benchmark for Mars science, analogous to Geo-Bench in the EO domain.
- **Comprehensive and Systematic Evaluation**: Coverage spans traditional CNNs, Transformers, EO foundation models, and closed-source VLMs.
- **High Practical Utility**: All code, datasets, and baseline models are open-sourced, supported on both Hugging Face and Zenodo.
- **Science-Driven Design**: Developed in collaboration with planetary scientists to ensure the scientific relevance of all tasks.
- **Important Insight**: The work explicitly identifies the need for Mars-specific foundation models, highlighting the limitations of both general-purpose and EO models.

## Limitations & Future Work
1. **Lack of Georeferencing**: Most datasets do not include spatial metadata (latitude/longitude), precluding spatial distribution analysis or regional generalization studies.
2. **Small Dataset Scale**: Annotating Mars data requires planetary science expertise and takes months to years, resulting in some very small datasets.
3. **No Evaluation of Self-Supervised/Foundation Model Pretraining**: The effect of self-supervised pretraining on Mars data itself is not assessed.
4. **Outdated THEMIS Data**: The crater segmentation dataset is based on the 2010 version of THEMIS; an updated version from 2017 is available.
5. **Future Directions**: Developing Mars-specific foundation models, exploring cross-domain transfer learning, and leveraging large quantities of unlabeled Mars imagery for self-supervised pretraining.

## Related Work & Insights
- **Geo-Bench / PANGAEA**: Successful standardized benchmarks in the EO domain whose design philosophy Mars-Bench directly adopts.
- **WILDS**: Its focus on cross-domain distribution shift evaluation aligns with Mars-Bench's cross-sensor and cross-task evaluation philosophy.
- **SatMAE / CROMA / Prithvi**: Transfer results of EO foundation models on Mars data confirm the existence of a substantial domain gap.
- The paper's analysis of VLM limitations in specialized scientific domains offers broader reference value beyond Mars science.

## Rating ⭐4
The first standardized benchmark for Mars science, filling an important gap with comprehensive evaluation and high practical value; however, no novel model contributions are made.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] RoMA: Scaling up Mamba-based Foundation Models for Remote Sensing](roma_scaling_up_mamba-based_foundation_models_for_remote_sensing.md)
- [\[ICCV 2025\] Can Generative Geospatial Diffusion Models Excel as Discriminative Geospatial Foundation Models?](../../ICCV2025/segmentation/can_generative_geospatial_diffusion_models_excel_as_discriminative_geospatial_fo.md)
- [\[CVPR 2026\] Kαlos finds Consensus: A Meta-Algorithm for Evaluating Inter-Annotator Agreement in Complex Vision Tasks](../../CVPR2026/segmentation/kαlos_finds_consensus_a_meta-algorithm_for_evaluating_inter-annotator_agreement_.md)
- [\[ICCV 2025\] TAViS: Text-bridged Audio-Visual Segmentation with Foundation Models](../../ICCV2025/segmentation/tavis_text-bridged_audio-visual_segmentation_with_foundation_models.md)
- [\[NeurIPS 2025\] PARTONOMY: Large Multimodal Models with Part-Level Visual Understanding](partonomy_large_multimodal_models_with_part-level_visual_understanding.md)

</div>

<!-- RELATED:END -->

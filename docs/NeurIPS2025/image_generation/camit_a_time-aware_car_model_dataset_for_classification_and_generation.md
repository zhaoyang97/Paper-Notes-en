---
title: >-
  [Paper Note] CaMiT: A Time-Aware Car Model Dataset for Classification and Generation
description: >-
  [NeurIPS 2025][Image Generation][temporal dataset] This paper introduces the CaMiT dataset (787K labeled + 5.1M unlabeled car images, 2005–2023) to systematically study temporal drift in fine-grained visual categories…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "temporal dataset"
  - "fine-grained classification"
  - "continual learning"
  - "time-aware generation"
  - "car model recognition"
date: 2026-05-08
content_hash: 2fefafb116ad19bb
---

# CaMiT: A Time-Aware Car Model Dataset for Classification and Generation

**Conference**: NeurIPS 2025
**arXiv**: [2510.17626](https://arxiv.org/abs/2510.17626)  
**Code**: [GitHub](https://github.com/lin-frederic/CaMiT)  
**Area**: Image Generation
**Keywords**: temporal dataset, fine-grained classification, continual learning, time-aware generation, car model recognition

## TL;DR

This paper introduces the CaMiT dataset (787K labeled + 5.1M unlabeled car images, 2005–2023) to systematically study temporal drift in fine-grained visual categories, providing benchmarks across four settings: static pre-training, time-incremental pre-training, time-incremental classifier learning, and time-aware image generation.

## Background & Motivation

Existing large-scale visual datasets (ImageNet, LAION, DataComp) follow a "train once" paradigm, neglecting the appearance drift that visual categories undergo over time. This drift is particularly pronounced in technical artifacts such as automobiles, where design iterations, new model releases, and the retirement of older models cause the visual representation of the same category to diverge increasingly across years.

Prior work either focuses on coarse-grained visual categories (VCT-107, CLEAR) or emphasizes large-scale VLM continual pre-training (TIC-DataComp), leaving a gap in **temporally-aware datasets for fine-grained technical artifacts over long time horizons**. Existing car datasets such as StanfordCars and CompCars include production year metadata but lack image upload timestamps and contain fewer than 150 samples per class, making them unsuitable for long-range temporal analysis.

CaMiT collects nearly 20 years of car images via the Flickr API and is the first to address a key question: **how to model the visual evolution of fine-grained technical artifacts over extended time periods in visual models.**

## Method

### Overall Architecture

CaMiT is constructed in three stages — **data collection → data filtering → data annotation** — and is evaluated across four experimental settings:

1. **SPT (Static Pre-Training)**: analyzes temporal drift effects without any temporal mitigation
2. **TIP (Time-Incremental Pre-Training)**: updates the backbone model on a yearly basis
3. **TICL (Time-Incremental Classifier Learning)**: freezes the backbone and updates only the classification head
4. **TAIG (Time-Aware Image Generation)**: incorporates temporal metadata into training captions

### Key Designs

**Dataset construction pipeline**:

- **Collection**: Flickr API queries over 425 car sub-category/brand/model combinations, up to 5,000 images per query per year, yielding 7.5M initial images
- **Filtering**: CLIP embedding deduplication (threshold 0.9) → YOLOv11x vehicle detection (confidence ≥ 0.6, bbox ≥ 64px) → Qwen2.5-7B filtering of non-exterior images → face blurring → SAM 2 overlap detection, resulting in 5.87M vehicle crops
- **Annotation**: semi-automatic pipeline — Qwen2.5-7B open-set prediction → GPT-4o focused confirmation → discriminator ensemble trained with DeiT weak labels; thresholds determined after human verification of 20K samples, achieving 99.6% annotation accuracy across 190 classes

**Temporal drift analysis**: KID (Kernel Inception Distance) between years is computed using CLIP ViT-B embeddings, confirming that larger temporal gaps correspond to greater embedding divergence.

**Classification experiment design**:

- SPT: compares DINOv2, CLIP (general pre-training), and MoCo v3 (domain-specific pre-training) with LoRA adaptation
- TIP: Reservoir update vs. LoRA annual adaptation vs. their combination
- TICL: frozen backbone + NCM / FeCAM / RanPAC / RanDumb incremental classifiers
- TAIG: SD1.5 + LoRA with temporal metadata embedded in captions

### Loss & Training

Classification experiments employ NCM (Nearest Class Mean) or the original objectives of respective TICL methods. Generation experiments are based on Stable Diffusion 1.5 standard diffusion loss with LoRA fine-tuning:

$$\mathcal{L} = \mathbb{E}_{t, \epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t, c_{\text{time}})\|^2\right]$$

where $c_{\text{time}}$ denotes the text condition encoding year information.

## Key Experimental Results

### Main Results

| Pre-training | Model | $A_{avg}$↑ | $A_{crt}$↑ | $A_{bck}$↑ | $A_{fwd}$↑ |
|:--|:--|:--:|:--:|:--:|:--:|
| SPT | DINOv2 ViT-B | 26.1 | 32.6 | 26.1 | 25.3 |
| SPT | CLIP+Li ViT-B | 65.6 | 74.0 | 63.9 | 66.3 |
| SPT | MoCo v3+Li ViT-B | **66.0** | **76.5** | **63.2** | **67.4** |
| TIP | MoCo v3+R+La ViT-S | 78.5 | 90.2 | 82.5 | 73.0 |
| TICL | RanPAC + MoCo v3+Li ViT-B | **87.8** | — | — | — |

The TICL approach (RanPAC + domain-specific pre-trained backbone) achieves the best overall accuracy of 87.8%, a gain of 21.8 percentage points over naive NCM.

### Ablation Study

| TICL Algorithm | DINOv2 ViT-S | MoCo v3+Li ViT-S | CLIP+Li ViT-B | MoCo v3+Li ViT-B |
|:--|:--:|:--:|:--:|:--:|
| NCM | 20.9 | 64.9 | 65.6 | 66.0 |
| NCM-TI | 26.3 | 71.4 | 70.1 | 72.1 |
| FeCAM | 61.0 | 85.6 | 79.9 | 81.5 |
| RanDumb | 62.1 | 83.1 | 77.2 | 84.2 |
| RanPAC | **66.4** | **86.6** | **80.3** | **87.8** |

- LoRA-based annual adaptation in TIP requires only 0.3 GPU hours/year, compared to 18 GPU hours/year for the Reservoir variant
- Domain-specific MoCo v3 pre-training with LoRA adaptation achieves performance comparable to CLIP trained on 2 billion images

### Key Findings

1. In fine-grained classification, domain-specific pre-training (MoCo v3 on CaMiT) with LoRA adaptation matches large-scale general-purpose models (CLIP trained on 2B images), a finding that diverges from conclusions in coarse-grained classification
2. Time-Incremental Classifier Learning (TICL) is the most effective strategy for mitigating temporal drift, with RanPAC achieving the best performance on the domain-specific backbone
3. Time-Aware Image Generation (TAIG), by embedding year metadata, yields generated image distributions that more closely match the real data distribution

## Highlights & Insights

- ⭐ The first long-horizon temporal visual dataset targeting fine-grained technical artifacts, filling an important gap in the field
- ⭐ Systematic evidence that specialized small models can match general large models on fine-grained tasks, providing strong support for resource-efficient approaches
- The semi-automatic annotation pipeline (VLM + discriminative model ensemble) achieves 99.6% accuracy at substantially lower cost than fully manual annotation
- The TICL approach (frozen backbone + incremental classifier) strikes an excellent balance between accuracy and computational efficiency

## Limitations & Future Work

- The dataset focuses exclusively on automobiles; generalizability to other technical artifact categories (electronics, furniture, etc.) remains unverified
- Flickr as the sole data source introduces geographic and demographic biases (European brands are most heavily represented)
- Time-aware generation is only explored with SD1.5; more advanced generative architectures are not investigated
- The 190 class categories are relatively limited and do not cover all mainstream global car models

## Related Work & Insights

CaMiT is complementary to temporal visual datasets such as CLEAR, VCT-107, and TIC-DataComp, which target coarse-grained categories or large-scale VLM pre-training, whereas CaMiT focuses on fine-grained recognition. The strong performance of RanPAC in TICL experiments suggests that **random projection combined with prototype accumulation** may constitute an efficient paradigm for temporal continual learning. The dataset has direct practical relevance to deployed systems such as vehicle model recognition in autonomous driving.

## Rating

⭐⭐⭐⭐ (4/5)

| Dimension | Rating |
|:--|:--:|
| Novelty | ⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐ |

The dataset contribution is solid and comprehensive, and the four experimental settings are well designed. However, the core technical novelty is relatively limited — the work is primarily a systematic combination of existing methods — and its scope is confined to a single artifact category.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] TIDMAD: Time Series Dataset for Discovering Dark Matter with AI Denoising](tidmad_time_series_dataset_for_discovering_dark_matter_with_ai_denoising.md)
- [\[NeurIPS 2025\] A Diffusion Model for Regular Time Series Generation from Irregular Data with Completion and Masking](a_diffusion_model_for_regular_time_series_generation_from_irregular_data_with_co.md)
- [\[NeurIPS 2025\] OmniCast: A Masked Latent Diffusion Model for Weather Forecasting Across Time Scales](omnicast_a_masked_latent_diffusion_model_for_weather_forecasting_across_time_sca.md)
- [\[ICCV 2025\] Inference-Time Diffusion Model Distillation](../../ICCV2025/image_generation/inference-time_diffusion_model_distillation.md)
- [\[NeurIPS 2025\] Latent Zoning Network: A Unified Principle for Generative Modeling, Representation Learning, and Classification](latent_zoning_network_a_unified_principle_for_generative_modeling_representation.md)

</div>

<!-- RELATED:END -->

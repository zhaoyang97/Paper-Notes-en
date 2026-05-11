---
title: >-
  [Paper Note] Reasoning with Pixel-level Precision: QVLM Architecture and SQuID Dataset for Quantitative Geospatial Analytics
description: >-
  [CVPR 2026][Segmentation][VLM] This paper proposes the QVLM architecture and SQuID dataset, achieving pixel-level quantitative spatial reasoning on satellite imagery through a decoupled design of code generation and segm…
tags:
  - "CVPR 2026"
  - "Segmentation"
  - "VLM"
  - "Quantitative Spatial Reasoning"
  - "Code Generation"
  - "Satellite Imagery"
  - "Semantic Segmentation"
date: 2026-05-08
content_hash: aaacba461b90e8c7
---

# Reasoning with Pixel-level Precision: QVLM Architecture and SQuID Dataset for Quantitative Geospatial Analytics

**Conference**: CVPR 2026
**arXiv**: [2601.13401](https://arxiv.org/abs/2601.13401)
**Code**: [GitHub](https://github.com/PeterAMassih/qvlm-squid)
**Area**: Semantic Segmentation
**Keywords**: VLM, Quantitative Spatial Reasoning, Code Generation, Satellite Imagery, Semantic Segmentation

## TL;DR

This paper proposes the QVLM architecture and SQuID dataset, achieving pixel-level quantitative spatial reasoning on satellite imagery through a decoupled design of code generation and segmentation models. The approach overcomes the fundamental limitation of conventional VLMs, which lose spatial indexing due to patch embedding compression.

## Background & Motivation

**Background**: Current vision-language models (VLMs) excel at scene understanding and qualitative description, but perform poorly on quantitative spatial reasoning tasks such as counting, area measurement, and distance calculation—particularly in the satellite imagery domain.

**Limitations of Prior Work**: VLMs compress 1024×1024 images into a 64×64 token grid via the vision encoder (256× compression), a process that architecturally destroys the pixel-level spatial indexing required for quantitative analysis. Research shows that the vision encoder causes 40–60% k-nearest-neighbor divergence.

**Key Challenge**: VLMs can fluently describe a forest yet fail to reliably count the trees within it—a fundamental disconnect exists between a model's qualitative understanding and its quantitative analytical capability.

**Goal**: To provide reliable solutions for applications requiring precise quantitative analysis in satellite imagery, including climate monitoring, urban planning, and disaster response.

**Key Insight**: Architectural decoupling—the language model is responsible solely for understanding queries and generating code, while all visual analysis is delegated to a segmentation model operating directly on raw pixels.

**Core Idea**: Decouple language understanding from visual analysis via code generation, enabling the model to perform geometric operations directly on pixel-level segmentation masks, thereby preserving uncompressed spatial indexing throughout the pipeline.

## Method

### Overall Architecture

QVLM is a three-stage architecture: (1) an LLM parses natural language queries and generates Python code; (2) the code invokes a segmentation model to obtain pixel-level binary masks; (3) geometric operations (counting, area calculation, distance measurement) are executed on the masks to produce the final answer. The LLM never directly processes image pixels, avoiding the information bottleneck introduced by the vision encoder.

### Key Designs

1. **SQuID Dataset**: Contains 2,000 satellite image QA pairs drawn from three data sources—DeepGlobe, EarthVQA, and Solar Panels—spanning three difficulty tiers (basic quantification / spatial relationships / complex multi-condition queries). A notable contribution is the introduction of acceptable answer ranges based on human annotation variability (rather than single-point answers), computed via median absolute deviation (MAD) from 500 annotations by 10 annotators.
2. **Code Generation API**: Provides three core geometric functions—`segment_image_from_path` (extracting land-cover masks), `find_shapes_within_distance` (buffer proximity analysis), and `calculate_shape_distances` (minimum distance computation). Combining these primitives enables handling of queries ranging from simple percentages to complex multi-condition reasoning.
3. **Segmentation Model**: Employs a ConvNeXt-UNet architecture (ImageNet-pretrained ConvNeXt encoder + U-Net decoder) supporting both semantic and instance segmentation. A DINOv3-Mask2Former variant is also implemented to validate modularity. Multiple models can be ensembled via max-logit fusion to extend class coverage.

### Loss & Training

The segmentation model is trained with cross-entropy loss and the Adam optimizer (lr=1e-4), with random affine cropping and color augmentation for data augmentation. QVLM itself is evaluated in a zero-shot pass@1 setting and requires no end-to-end training.

## Key Experimental Results

### Main Results

| Model Configuration | Tier 1 | Tier 2 | Tier 3 | Overall Accuracy |
|---|---|---|---|---|
| QVLM (GPT-5 + ConvNeXt) | 53.52% | 54.06% | 18.84% | **42.00%** |
| QVLM (GPT-oss-120B + ConvNeXt) | 43.84% | 47.62% | 5.88% | 32.14% |
| QVLM (GPT-5 + DINOv3) | 40.74% | 40.22% | 12.20% | 30.83% |
| QVLM (Llama3.1-8B + ConvNeXt) | 39.86% | 41.88% | 5.79% | 29.00% |
| VLM-A (GPT-5 Direct Encoding) | 39.30% | 34.09% | 10.83% | 28.10% |
| VLM-B (QWEN 30B) | 39.01% | 36.85% | 3.71% | 26.14% |

### Detailed Results by Question Type

| Question Type | QVLM (GPT-5+ConvNeXt) | VLM-A (GPT-5) |
|---|---|---|
| fragmentation | **81.63%** | 26.53% |
| connectivity | **74.04%** | 37.50% |
| proximity % | **40.65%** | 19.51% |
| count | **56.74%** | 36.52% |
| size | **33.73%** | 16.27% |

### Key Findings

- QVLM outperforms the strongest VLM baseline by **+13.9%** in overall accuracy, validating that the code generation architecture preserves spatial precision destroyed by the vision encoder.
- The largest gains are observed on fragmentation and connectivity question types (+55% and +37%), which require the most precise spatial structural analysis.
- The ConvNeXt segmentation model outperforms DINOv3, suggesting that fully convolutional architectures retain advantages for local feature extraction in satellite imagery.
- Tier 3 complex multi-condition queries remain highly challenging, with the best model achieving only 18.84%.

## Highlights & Insights

- **Fundamental Architectural Insight**: Quantitative reasoning failures are attributed to architectural design choices (rather than insufficient training data), motivating the corresponding architectural decoupling solution.
- **Acceptable Answer Ranges**: SQuID replaces single-point answers with MAD-based ranges derived from human annotation variability, more fairly reflecting the inherent uncertainty in human spatial perception.
- **Modularity**: The code generator and segmentation model can be upgraded independently; component replacement does not require retraining the entire system.
- **Zero-Shot Generalization**: Significant performance gains are achieved without end-to-end training on satellite imagery.

## Limitations & Future Work

- Accuracy on Tier 3 complex queries remains low (18.84%), requiring stronger multi-step reasoning capabilities.
- Code generation quality depends on the LLM; smaller models (Llama-8B) show notable performance degradation.
- Only the zero-shot setting is evaluated; few-shot prompting or domain fine-tuning may yield further improvements.
- The segmentation model has limited class coverage and requires extension for more diverse detection categories.

## Related Work & Insights

- **ViperGPT** pioneered the code generation + visual API paradigm, but did not address the unique challenges of satellite imagery (resolution variation, land-cover classification, metric precision).
- **Subramanian et al.** demonstrated that code generation outperforms baseline VLMs on spatial reasoning by approximately 30%; QVLM further validates this advantage in the satellite imagery domain.
- This approach is complementary to direct segmentation enhancement strategies (e.g., Lai et al.'s embedding-as-mask), which could be explored in future work.

## Rating

- Novelty: ⭐⭐⭐⭐ (Architectural decoupling is conceptually clear; SQuID dataset design is rigorous)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Comprehensive multi-model and multi-tier comparisons, though evaluation is limited to a single dataset)
- Writing Quality: ⭐⭐⭐⭐ (Logical structure is clear; problem motivation is compellingly articulated)
- Value: ⭐⭐⭐⭐ (Opens a new paradigm for quantitative spatial reasoning, though generalization to other domains requires further validation)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] UniPixel: Unified Object Referring and Segmentation for Pixel-Level Visual Reasoning](../../NeurIPS2025/segmentation/unipixel_unified_object_referring_and_segmentation_for_pixel-level_visual_reason.md)
- [\[AAAI 2026\] JoDiffusion: Jointly Diffusing Image with Pixel-Level Annotations for Semantic Segmentation Promotion](../../AAAI2026/segmentation/jodiffusion_jointly_diffusing_image_with_pixel-level_annotations_for_semantic_se.md)
- [\[CVPR 2026\] CrossEarth-SAR: A SAR-Centric and Billion-Scale Geospatial Foundation Model for Domain Generalizable Semantic Segmentation](crossearthsar_a_sarcentric_and_billionscale_geospa.md)
- [\[CVPR 2026\] CA-LoRA: Concept-Aware LoRA for Domain-Aligned Segmentation Dataset Generation](ca-lora_concept-aware_lora_for_domain-aligned_segmentation_dataset_generation.md)
- [\[CVPR 2026\] MixerCSeg: An Efficient Mixer Architecture for Crack Segmentation via Decoupled Mamba Attention](mixercseg_an_efficient_mixer_architecture_for_crack_segmentation_via_decoupled_m.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Scaling Spatial Intelligence with Multimodal Foundation Models
description: >-
  [CVPR 2026][Multimodal VLM][Spatial intelligence] SenseNova-SI systematically constructs an 8M-scale diverse spatial dataset (SenseNova-SI-8M) to cultivate spatial intelligence in multimodal foundation models including Q…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "Spatial intelligence"
  - "multimodal foundation models"
  - "data scaling"
  - "spatial reasoning"
  - "benchmarking"
date: 2026-05-08
content_hash: 0b4a728fb3739555
---

# Scaling Spatial Intelligence with Multimodal Foundation Models

**Conference**: CVPR 2026
**arXiv**: [2511.13719](https://arxiv.org/abs/2511.13719)
**Code**: [https://github.com/OpenSenseNova/SenseNova-SI](https://github.com/OpenSenseNova/SenseNova-SI)
**Area**: Multimodal VLM / Spatial Intelligence
**Keywords**: Spatial intelligence, multimodal foundation models, data scaling, spatial reasoning, benchmarking

## TL;DR

SenseNova-SI systematically constructs an 8M-scale diverse spatial dataset (SenseNova-SI-8M) to cultivate spatial intelligence in multimodal foundation models including Qwen3-VL, InternVL3, and Bagel, achieving unprecedented performance on multiple spatial benchmarks such as VSI-Bench and MMSI while preserving general multimodal understanding capabilities.

## Background & Motivation

**Background**: Multimodal foundation models (e.g., GPT-4V, Gemini, Qwen-VL) excel at visual understanding and text generation, yet exhibit notable deficiencies in spatial intelligence — including depth estimation, spatial relationship reasoning, 3D scene understanding, and perspective transformation reasoning.

**Limitations of Prior Work**: (1) Existing multimodal models perform significantly worse on spatial reasoning tasks than on general visual question answering, indicating insufficient spatial information in internet-scale training data; (2) No systematic spatial capability taxonomy exists to guide data construction; (3) The effects of data scaling on spatial intelligence, risks of overfitting, and language shortcut issues remain underexplored.

**Key Challenge**: The spatial intelligence deficiency in multimodal models stems from insufficient quantity and diversity of spatially-relevant training samples, rather than architectural limitations.

**Goal**: To cultivate spatial intelligence in existing multimodal foundation models through large-scale data scaling, and to rigorously analyze the effects of data scale, diversity, overfitting risk, and related factors.

**Key Insight**: Rather than modifying model architectures, the paper systematically constructs a large-scale dataset (8M samples) covering diverse spatial capabilities and leverages a data-driven approach to enhance spatial intelligence.

**Core Idea**: Guided by a rigorous spatial capability taxonomy, an 8M-scale diverse spatial dataset is constructed to significantly improve spatial intelligence in existing foundation models via fine-tuning.

## Method

### Overall Architecture

SenseNova-SI builds upon existing multimodal foundation models and adopts a data-driven strategy to enhance spatial intelligence. The overall pipeline proceeds as follows: (1) establish a spatial capability taxonomy; (2) systematically collect, generate, and augment 8M-scale data samples (SenseNova-SI-8M) guided by this taxonomy; (3) fine-tune Qwen3-VL and InternVL3 (visual understanding models) as well as Bagel (a unified understanding-generation model); (4) evaluate on multiple spatial benchmarks and analyze the impact of various factors.

### Key Designs

1. **Spatial Capability Taxonomy**:

    - Function: Provides systematic guidance for data construction.
    - Mechanism: Spatial intelligence is decomposed into multiple capability dimensions, including depth perception, distance/size estimation, spatial relationship reasoning (above/below, left/right, front/back, near/far), perspective transformation reasoning, 3D shape understanding, and spatial navigation. Each dimension corresponds to a specific type of training sample.
    - Design Motivation: Spatial intelligence is a composite of multiple sub-capabilities rather than a monolithic skill. Without an explicit taxonomy, data collection tends to over-represent certain sub-capabilities while neglecting others.

2. **SenseNova-SI-8M Dataset Construction**:

    - Function: Provides large-scale, diverse spatial training data.
    - Mechanism: Data is collected through multiple channels — (a) extracting and converting existing 3D/spatial datasets into multimodal QA format; (b) synthesizing spatial scenes via 3D engines and generating question–answer pairs; (c) generating spatially-grounded QA pairs from real images using large models; (d) augmenting and expanding existing data. The result is 8M samples covering all dimensions of the taxonomy.
    - Design Motivation: Data scale and diversity are key factors for spatial intelligence. Prior work was insufficient in both data volume and coverage of spatial capability dimensions.

3. **Multi-Foundation Model Adaptation**:

    - Function: Validates the generalizability of the data-driven approach across different architectures.
    - Mechanism: Three representative foundation models are selected — pure visual understanding models Qwen3-VL and InternVL3, and the unified understanding-generation model Bagel. Fine-tuning with SenseNova-SI-8M on all three demonstrates that spatial intelligence gains are not model-specific.
    - Design Motivation: A data scaling approach that works only for a single model architecture has limited value. Cross-model validation demonstrates that this is a broadly applicable strategy.

### Loss & Training

- **Training Paradigm**: Standard instruction tuning — existing foundation models are further fine-tuned on SenseNova-SI-8M.
- **Preventing Catastrophic Forgetting**: A proportion of general multimodal data is mixed into fine-tuning to prevent degradation of general capabilities while improving spatial intelligence.
- **Data Mixing Strategy**: Samples from different spatial capability dimensions are mixed at specified ratios to ensure adequate training coverage across all capabilities.

## Key Experimental Results

### Main Results

| Benchmark | Metric | SenseNova-SI | Prev. SOTA | Gain |
|-----------|--------|-------------|------------|------|
| VSI-Bench | Accuracy | 68.8% | ~55% | +13.8% |
| MMSI | Accuracy | 43.3% | ~35% | +8.3% |
| MindCube | Accuracy | 85.7% | ~70% | +15.7% |
| ViewSpatial | Accuracy | 54.7% | ~45% | +9.7% |
| SITE | Accuracy | 47.7% | ~40% | +7.7% |
| BLINK | Accuracy | 63.9% | ~55% | +8.9% |
| 3DSR | Accuracy | 55.5% | ~45% | +10.5% |
| EmbSpatial | Accuracy | 72.0% | ~60% | +12.0% |
| MMBench-En | Accuracy | 84.9% | 84.9% | No change |

### Ablation Study

| Configuration | VSI-Bench | MMBench-En | Note |
|---------------|-----------|------------|------|
| Base model (no fine-tuning) | ~55% | 84.9% | Weak spatial, strong general |
| + 1M spatial data | ~60% | 84.5% | Limited spatial improvement |
| + 4M spatial data | ~65% | 84.7% | Continued gains with scaling |
| + 8M spatial data (Full) | 68.8% | 84.9% | Best spatial + preserved general |

### Key Findings

- **Data Scaling Curve**: Spatial intelligence improves approximately logarithmically with data volume, indicating continued but diminishing marginal returns from additional data.
- **Emergent Generalization**: After training on diverse data, the model exhibits a degree of generalization to unseen spatial task types.
- **Overfitting Risk**: Signs of overfitting appear on certain spatial benchmarks, particularly when the training data distribution closely resembles the test set.
- **Language Shortcuts**: Some spatial reasoning tasks contain language shortcuts (allowing correct answers without image inspection), highlighting the need for dataset debiasing.
- **Preservation of General Capabilities**: With an appropriate data mixing strategy, gains in spatial intelligence do not compromise general multimodal understanding performance.

## Highlights & Insights

- **Data-Driven Rather Than Architecture-Driven Paradigm**: The work demonstrates that spatial intelligence can be acquired through large-scale data scaling without modifying model architectures, providing a reference approach for addressing other capability deficiencies.
- **Systematic Spatial Capability Taxonomy**: The proposed taxonomy not only guides data construction but also provides the community with a framework for evaluating spatial intelligence.
- **Early Signals of Emergent Generalization**: The emergent generalization arising from data diversity is a direction worthy of deeper investigation, suggesting that large-scale diverse data may be a key pathway toward general spatial intelligence.

## Limitations & Future Work

- Constructing 8M samples incurs substantial cost, and the approach partially relies on synthetic data whose realism and diversity warrant further evaluation.
- Spatial chain-of-thought reasoning (Spatial CoT) remains preliminary; more complex multi-step spatial reasoning capabilities are limited.
- Evaluation is conducted primarily on static images; assessment of spatial intelligence in video and interactive settings is absent.
- For spatial tasks requiring precise numerical outputs (e.g., accurate depth estimation), the precision of the current approach still has room for improvement.
- Diminishing marginal returns from data scaling suggest that future work may need to integrate architectural improvements or novel training paradigms.

## Related Work & Insights

- **vs. SpatialVLM**: SpatialVLM focuses on 2D spatial relationships, whereas SenseNova-SI covers a broader range of spatial capability dimensions (including 3D) at a substantially larger data scale.
- **vs. SpatialRGPT**: SpatialRGPT addresses region-level spatial reasoning; SenseNova-SI takes a more comprehensive approach to systematically improving overall spatial intelligence.
- **vs. Specialized 3D Models**: Dedicated 3D vision models may outperform on specific tasks, but SenseNova-SI preserves general multimodal capabilities, representing a generalist model approach.

## Rating

- Novelty: ⭐⭐⭐ The core approach is data scaling with limited methodological innovation, though the systematic nature of the work is valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated on 8 spatial benchmarks plus general benchmarks, with comprehensive ablation and analysis.
- Writing Quality: ⭐⭐⭐⭐ Report-style writing with rich content, though somewhat verbose.
- Value: ⭐⭐⭐⭐ Open-sourced models and data make a clear contribution to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SpatialScore: Towards Comprehensive Evaluation for Spatial Intelligence](spatialscore_towards_comprehensive_evaluation_for_spatial_intelligence.md)
- [\[ICLR 2026\] On the Generalization Capacities of MLLMs for Spatial Intelligence](../../ICLR2026/multimodal_vlm/on_the_generalization_capacities_of_mllms_for_spatial_intelligence.md)
- [\[CVPR 2026\] Nano-EmoX: Unifying Multimodal Emotional Intelligence from Perception to Empathy](nano-emox_unifying_multimodal_emotional_intelligence_from_perception_to_empathy.md)
- [\[CVPR 2026\] AnomalyVFM -- Transforming Vision Foundation Models into Zero-Shot Anomaly Detectors](anomalyvfm_--_transforming_vision_foundation_models_into_zero-shot_anomaly_detec.md)
- [\[CVPR 2026\] HiSpatial: Taming Hierarchical 3D Spatial Understanding in Vision-Language Models](hispatial_taming_hierarchical_3d_spatial_understanding_in_vision-language_models.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] MRD: Multi-resolution Retrieval-Detection Fusion for High-Resolution Image Understanding
description: >-
  [CVPR2026][Object Detection][High-resolution image understanding] This paper proposes MRD, a training-free multi-resolution retrieval-detection fusion framework that mitigates object fragmentation via cross-resolution se…
tags:
  - "CVPR2026"
  - "Object Detection"
  - "High-resolution image understanding"
  - "multimodal large language models"
  - "retrieval-augmented perception"
  - "open-vocabulary detection"
  - "multi-resolution fusion"
  - "training-free"
date: 2026-05-08
content_hash: a5b23abd342964e5
---

# MRD: Multi-resolution Retrieval-Detection Fusion for High-Resolution Image Understanding

**Conference**: CVPR2026
**arXiv**: [2512.02906](https://arxiv.org/abs/2512.02906)  
**Code**: [yf0412/MRD](https://github.com/yf0412/MRD)  
**Area**: Object Detection / High-Resolution Image Understanding
**Keywords**: High-resolution image understanding, multimodal large language models, retrieval-augmented perception, open-vocabulary detection, multi-resolution fusion, training-free

## TL;DR

This paper proposes MRD, a training-free multi-resolution retrieval-detection fusion framework that mitigates object fragmentation via cross-resolution semantic fusion and suppresses background interference through an open-vocabulary detector, substantially improving MLLM understanding of high-resolution images.

## Background & Motivation

1. **High-resolution bottleneck in MLLMs**: Mainstream multimodal large language models are constrained by fixed low-resolution inputs and fail to effectively capture fine-grained details such as small objects, textures, and text in high-resolution images.
2. **High cost of training-based methods**: SFT/RL-based localize-and-zoom-in approaches suffer from large training costs, long convergence cycles, and poor cross-architecture transferability, limiting practical deployment.
3. **Object fragmentation (FRAG)**: Retrieval-based methods partition high-resolution images into fixed grids, causing large objects to be split across multiple patches. This results in semantic embedding bias and incomplete retrieval, accounting for 65.2% of failure cases.
4. **Background interference (BG)**: Complex background regions produce spuriously high similarity scores with the query, introducing false-positive patches that mislead downstream reasoning, accounting for 54.3% of failure cases.
5. **Scale sensitivity**: The crop resolution is a difficult-to-tune hyperparameter—excessively large crops dilute target semantics with background noise, while excessively small crops exacerbate fragmentation.
6. **Multi-object scenario deficiency**: Existing top-down methods tend to miss non-primary objects during the coarse initial search stage, leading to poor performance on multi-object tasks.

## Method

### Overall Architecture

MRD is a **training-free** unified multi-scale framework comprising two core modules:

- **Multi-resolution Semantic Fusion**: operates at the local scale to calibrate cross-resolution semantic consistency.
- **Open-vocabulary Detector Enhancement**: operates at the global scale for explicit spatial localization and background suppression.

The outputs of both modules are linearly fused to produce the final similarity map, which guides the subsequent retrieval search process.

### Multi-resolution Semantic Fusion

1. The high-resolution image is partitioned at two scale resolutions: a low-resolution crop set $P$ (resolution $l$) and a high-resolution crop set $\hat{P}$ (resolution $\hat{l}=k \cdot l$), with a spatial correspondence such that each HR crop corresponds to $k^2$ LR crops.
2. A vision-language model from VisRAG is used to compute cosine similarity between the query and crops at both resolutions.
3. HR similarity scores are projected into the LR space and fused via geometric mean: $s_t^m = \sqrt{\tilde{s}_t \cdot s_t}$.
4. The fused result is reshaped into a 2D semantic similarity map; cross-resolution consistency fusion corrects semantic bias caused by object fragmentation.

### Open-vocabulary Detector Enhancement

1. **Entity extraction**: LLM in-context learning is applied to extract target entities from free-text queries, which serve as detection categories.
2. **Sliding-window detection**: LLMDet performs open-vocabulary detection region-by-region over the HR image using a sliding window spatially aligned with the semantic fusion module.
3. **Confidence map generation**: Detection boxes are filtered by threshold; confidence scores are mapped to their corresponding crop positions and averaged across windows to produce a global detection confidence map $c^g(i,j)$.

### Semantic-Detection Fusion

$$s^f(i,j) = (1-w) \cdot s^m(i,j) + w \cdot c^g(i,j)$$

where $w$ is a balancing weight. The semantic similarity map provides fine-grained matching, while the detection confidence map provides explicit spatial localization and background suppression; the two are complementary.

## Key Experimental Results

### Main Results (V* Bench)

| Method | Attribute | Spatial | Overall |
|--------|-----------|---------|---------|
| LLaVA-v1.5-7B (baseline) | 43.5 | 56.6 | 48.7 |
| LLaVA-v1.5-7B-RAP | 90.4 | 96.1 | 91.1 |
| **LLaVA-v1.5-7B-MRD** | **97.4** | **96.1** | **95.6** |
| LLaVA-ov-0.5B-RAP | 80.0 | 84.2 | 83.6 |
| **LLaVA-ov-0.5B-MRD** | **89.6** | **85.6** | **88.9** |

- Based on LLaVA-v1.5-7B, MRD achieves an overall gain of 46.9% on V* Bench (nearly doubling performance), surpassing GPT-4o (66.0).
- MRD consistently outperforms RAP on HR-Bench 4K/8K, with an average overall improvement of 2.8%.

### Ablation Study

| Module Combination | Overall | BG Error Rate | FRAG Error Rate |
|--------------------|---------|---------------|-----------------|
| RAP (baseline) | 83.6 | 10.7% | 8.9% |
| OVD only | 85.3 (+1.7) | 5.7% (−46.7%) | 6.2% (−30.3%) |
| RAP + Multi-Res | 85.8 (+2.2) | 6.7% (−37.4%) | 5.3% (−40.4%) |
| RAP + OVD | 86.7 (+3.1) | 4.9% (−54.2%) | 5.8% (−34.8%) |
| **MRD (All)** | **88.9 (+5.3)** | **4.0% (−62.6%)** | **4.4% (−50.6%)** |

The two modules are complementary: Multi-Res primarily alleviates fragmentation (FRAG ↓40.4%), while OVD primarily suppresses background interference (BG ↓54.2%). The full MRD substantially reduces both error types simultaneously.

### Efficiency Comparison

| Method | Search Time | Total Time | Peak Memory |
|--------|-------------|------------|-------------|
| RAP (v1.5-7B) | 52.8s | 63.4s | 21.2 GB |
| MRD (v1.5-7B) | 15.2s (−71.2%) | 53.4s (−26.2%) | 23.4 GB (+10.4%) |

Although MRD introduces additional RAG and detection overhead, more precise localization substantially reduces the number of search steps, resulting in a net reduction of 26.2% in total inference time.

## Highlights & Insights

- **Training-free**: Requires no additional training; can be seamlessly integrated as a plug-and-play enhancement for any MLLM's high-resolution understanding capability.
- **Complementary dual-branch design**: Semantic fusion addresses fragmentation while detection enhancement suppresses background interference, together covering 89.6% of failure patterns.
- **Cross-resolution consistency fusion**: The geometric mean formulation naturally penalizes semantic bias at any single resolution, offering a principled alternative to simple summation.
- **Improved efficiency**: More precise initial localization reduces search-stage latency by 71%, resulting in lower total inference time despite additional module overhead.
- **Strong generalizability**: Consistently effective across multiple MLLM backbones (LLaVA-v1.5, LLaVA-ov) and multiple benchmarks.

## Limitations & Future Work

1. **Dependency on external detectors**: Requires an additional OVD model (LLMDet), increasing system complexity and GPU memory consumption (+10–12%).
2. **Sliding-window detection overhead**: The sliding-window strategy introduces an additional 15–16 seconds of detection time, which may be more pronounced for very large images.
3. **OVD limitations in multi-object scenarios**: Ablation results show that OVD used alone underperforms RAP on multi-object tasks, suggesting that complex multi-object scenes may still suffer from missed detections.
4. **Fixed linear fusion weight**: The semantic-detection fusion weight $w$ is a fixed hyperparameter and is not adaptively adjusted across different scenarios.
5. **Limited evaluation scope**: Evaluation is restricted to V* Bench and HR-Bench; validation on additional domains such as document understanding and remote sensing is lacking.

## Related Work & Insights

| Method | Type | Training Required | Multi-object | Fragmentation Handling | Background Suppression |
|--------|------|-------------------|--------------|------------------------|------------------------|
| ZoomEye | localize-zoom | ✗ | Weak | ✗ | ✗ |
| RAP | Retrieval-augmented | ✗ | Moderate | ✗ | ✗ |
| SFT methods | localize-zoom | ✓ | Moderate | ✗ | Partial |
| **MRD** | **Retrieval-detection fusion** | **✗** | **Strong** | **✓** | **✓** |

MRD is the first method to jointly model local semantic completeness and global spatial localization within a retrieval-augmented perception framework, surpassing all training-based methods under training-free conditions.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The dual-branch complementary design combining multi-resolution fusion and OVD enhancement is novel, with a thorough quantitative analysis of failure modes in retrieval-augmented perception.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Validated across multiple benchmarks and backbones; ablation study is complete with efficiency analysis and visualizations.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem formulation is clear, quantitative failure mode analysis is compelling, and the method description is rigorous.
- **Value**: ⭐⭐⭐⭐ — Provides an effective training-free solution for high-resolution image understanding that is deployment-friendly.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Toward Generalizable Whole Brain Representations with High-Resolution Light-Sheet Data](toward_generalizable_whole_brain_representations_with_high-resolution_light-shee.md)
- [\[CVPR 2026\] Beyond Semantic Search: Towards Referential Anchoring in Composed Image Retrieval](beyond_semantic_search_towards_referential_anchoring_in_composed_image_retrieval.md)
- [\[CVPR 2026\] Small Target Detection Based on Mask-Enhanced Attention Fusion of Visible and Infrared Remote Sensing Images](small_target_detection_based_on_mask-enhanced_attention_fusion_of_visible_and_in.md)
- [\[CVPR 2026\] Learning Multi-Modal Prototypes for Cross-Domain Few-Shot Object Detection](learning_multi-modal_prototypes_for_cross-domain_few-shot_object_detection.md)
- [\[CVPR 2026\] Beyond Caption-Based Queries for Video Moment Retrieval](beyond_caption-based_queries_for_video_moment_retrieval.md)

</div>

<!-- RELATED:END -->

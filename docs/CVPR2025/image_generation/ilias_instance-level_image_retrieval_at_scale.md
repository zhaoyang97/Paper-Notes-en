---
title: >-
  [Paper Note] ILIAS: Instance-Level Image Retrieval At Scale
description: >-
  [CVPR 2025][Image Generation][Instance-Level Retrieval] ILIAS is a large-scale instance-level image retrieval benchmark consisting of 1,000 instance objects and 100 million distractor images. Through comprehensive benchmarking, it reveals the capabilities and limitations of current foundation models in specific object recognition, providing a far-from-saturated evaluation standard for this field.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Instance-Level Retrieval"
  - "Large-Scale Benchmarking"
  - "Foundation Models"
  - "Vision-Language Models"
  - "Dataset Benchmark"
date: 2026-05-08
content_hash: aa0d32ec34efbb72
---

# ILIAS: Instance-Level Image Retrieval At Scale

**Conference**: CVPR 2025  
**arXiv**: [2502.11748](https://arxiv.org/abs/2502.11748)  
**Code**: [https://github.com/ilias-vrg/ilias](https://github.com/ilias-vrg/ilias)  
**Area**: Image Retrieval / Benchmark Dataset  
**Keywords**: Instance-Level Retrieval, Large-Scale Benchmarking, Foundation Models, Vision-Language Models, Dataset Benchmark

## TL;DR
ILIAS is a large-scale instance-level image retrieval benchmark consisting of 1,000 instance objects and 100 million distractor images. Through comprehensive benchmarking, it reveals the capabilities and limitations of current foundation models in specific object recognition, providing a far-from-saturated evaluation standard for this field.

## Background & Motivation

**Background**: Instance-level image retrieval aims to find images containing the exact same object as the query image from giant galleries. With the rise of vision foundation models like DINOv2 and SigLIP, the quality of general visual representations has vastly improved. However, existing benchmarks (e.g., Oxford Buildings, Paris) are small in scale, limited in domain (primarily landmarks), and nearly saturated in performance, making it difficult to distinguish the pros and cons of different methods.

**Limitations of Prior Work**: Existing datasets face three core issues: (1) Limited scale: the largest retrieval benchmark is only at the million scale, insufficient for testing retrieval capabilities under billion-scale real-world scenarios. (2) Uniform domain: biased towards landmarks and product retrieval, failing to evaluate model generalization across diverse object types. (3) Saturated performance: top-performing models achieve over 90% retrieval accuracy, lacking discriminative utility. Furthermore, false negatives in ground truth annotations disrupt evaluation reliability.

**Key Challenge**: There is a gap between the rapid progress of foundation models and the lag in evaluation benchmarks. Existing benchmarks cannot answer the crucial question: "how well do current state-of-the-art foundation models actually perform on instance-level retrieval of arbitrary objects?"

**Goal**: Build a large-scale, multi-domain, ground-truth-accurate, and far-from-saturated instance-level retrieval benchmark to comprehensively evaluate current foundation models and retrieval techniques.

**Key Insight**: The authors adopt an ingenious ground truth guarantee strategy by collecting only object instances that appeared after 2014 as query targets. Since the YFCC100M dataset was compiled in 2014, these objects strictly cannot appear among the 100 million distractor images of YFCC100M, thereby eliminating false negatives without requiring extra annotations.

**Core Idea**: Build a temporally separated large-scale retrieval benchmark (queries post-2014, distractors pre-2014) to ensure accurate evaluation with zero false negatives.

## Method

### Overall Architecture
The construction of ILIAS consists of three stages: data collection, annotation verification, and comprehensive benchmarking. The dataset contains 1,232 query images (with clean/uniform backgrounds) of 1,000 object instances, 4,715 positive images (taken in real-world environments with occlusions, cluttered backgrounds, scale variations), 1,000 textual query descriptions, and 100 million distractor images from YFCC100M. The evaluation workflow covers the complete retrieval pipeline including feature extraction, kNN search, linear adaptation, and reranking.

### Key Designs

1. **Temporal Separation for GT**:

    - **Function**: Guarantees zero false negatives without requiring manual verification of each of the 100 million distractor images.
    - **Mechanism**: All 1,000 query object instances are verified to have appeared after 2014 (e.g., new buildings, new products, new artwork). The images in YFCC100M were entirely collected before 2014. Therefore, these objects in theory cannot appear in the distractor set, ensuring any unannotated image is a true negative.
    - **Design Motivation**: Traditional methods for evaluating large-scale retrieval face a false-negative dilemma; it is impractical to exhaustively annotate all images containing target objects in million- or billion-scale galleries. The temporal separation strategy fundamentally eliminates this problem while avoiding expensive large-scale manual annotation.

2. **Multi-Domain Instance Collection**:

    - **Function**: Ensures the evaluation covers diverse object types without biasing towards any specific domain.
    - **Mechanism**: The 1,000 object instances span multiple domains such as buildings, sculptures, products, logos, natural landmarks, and artwork. For each instance, query images are captured against clean backgrounds (similar to "product photos"), while positive images are captured in real-world environments, containing various challenging conditions (occlusion, cluttered backgrounds, viewpoint changes, scale variations, partial visibility, etc.). This design simulates real-world application scenarios like "using product images to search for items" or "using reference photos to search for real-world locations."
    - **Design Motivation**: Existing benchmarks heavily favor landmarks, leading to cases where "models fine-tuned on landmarks fail on ILIAS" — experiments verify that domain-specialized models lack generalization capability. The multi-domain design is specifically intended to expose this issue.

3. **Comprehensive Benchmarking Pipeline**:

    - **Function**: Provides standardized evaluation pipelines and multi-dimensional comparative analysis.
    - **Mechanism**: The evaluation covers four dimensions: (a) base image-to-image retrieval: direct kNN search using model features; (b) linear adaptation: training a linear layer on model features with multi-domain class supervision; (c) reranking: utilizing local descriptors (e.g., AMES, SP, etc.) to rerank initial retrieval results; (d) text-to-image retrieval: utilizing text queries and vision-language models for text-to-image search. Evaluation metrics are mAP@100 and mAP (full).
    - **Design Motivation**: Different stages of the retrieval pipeline (feature extraction, adaptation, reranking) have distinct impacts on the final performance; a comprehensive evaluation reveals the contribution of each stage.

### Loss & Training
As an evaluation benchmark, ILIAS does not involve model training. The linear adaptation part utilizes multi-domain class supervision to train a linear projection layer, employing the cross-entropy loss.

## Key Experimental Results

### Main Results (Image-to-Image Retrieval)

| Model | Training Method | mAP@100 | mAP (full) |
|------|---------|---------|------------|
| DINOv3-L | SSL | **31.1** | **26.5** |
| PE-L@336 | VLA | 27.1 | 22.0 |
| DINOv3-B | SSL | 26.4 | 22.0 |
| SigLIP2-L@512 | VLA | 25.3 | 20.8 |
| DINOv2-L | SSL | 18.5 | 15.3 |
| OpenCLIP-L | VLA | 12.7 | 9.8 |

### Retrieval After Linear Adaptation

| Model | mAP@100 (Before Adaptation) | mAP@100 (After Adaptation) | Gain |
|------|---------------|---------------|------|
| PE-L@336 | 27.1 | **39.6** | +12.5 |
| SigLIP2-L@512 | 25.3 | 37.3 | +12.0 |
| SigLIP-L@384 | 24.2 | 34.3 | +10.1 |
| DINOv3-L | 31.1 | 32.9 | +1.8 |
| DINOv2-L | 18.5 | 23.5 | +5.0 |

### Key Findings
- **Performance is far from saturated**: The best model achieves an mAP of only around 31% (base retrieval) or 40% (after adaptation), leaving huge room for improvement. This contrasts sharply with the 90%+ performance on existing landmark benchmarks.
- **Domain-specialized models generalize poorly**: Models that perform exceptionally well on landmarks or product retrieval show subpar performance on ILIAS's multi-domain evaluation.
- **Linear adaptation of VLMs yields significant gains**: Vision-language models show massive performance gains after simple linear adaptation (+12.5 points), whereas the SSL model DINOv3 shows limited improvement (+1.8), suggesting that the potential of VLM features is underutilized but can be unlocked with minimal supervision.
- **Reranking with local descriptors remains crucial**: In scenarios with severe background clutter, reranking with local descriptors boosts mAP from 34 to 39, proving that spatial geometric verification is still irreplaceable.
- **Text-to-image retrieval is close to image-to-image**: The text-to-image retrieval performance of VLMs (24.7 for SigLIP2-L) is surprisingly close to their image-to-image performance (25.3).

## Highlights & Insights
- **The temporal separation strategy** is an ingenious ground truth guarantee mechanism—leveraging the natural separation between when objects appeared and when the dataset was compiled to achieve zero false negatives at zero annotation cost. This idea can be transferred to any evaluation scenario requiring massive negative distractor sets.
- Benchmarking at the scale of 100 million distractor images reveals the true boundaries of foundation models. Even the best model achieves only around 31% mAP, illustrating that instance-level retrieval remains a highly challenging open problem.
- The massive impact of linear adaptation (+12 points on VLMs) hints at a low-cost, practical improvement direction: with just a small amount of multi-domain annotated data, the retrieval ability of general foundation models can be significantly boosted.

## Limitations & Future Work
- ILIAS is purely an evaluation benchmark and does not provide training data, which limits the potential to develop new methods directly on it.
- Including only post-2014 objects might introduce implicit biases—these objects may be more "modern" and carry highly distinctive features.
- Although the scale of 1,000 instances is larger than existing benchmarks, it might still be insufficient for certain domains (e.g., limited diversity in natural objects).
- The quality and granularity of text queries can affect the fairness of text-to-image evaluations.

## Related Work & Insights
- **vs Oxford/Paris Buildings**: Oxford5k and Paris6k are classic landmark retrieval benchmarks, but they are limited in scale (5k/6k images) and their performances are already saturated. ILIAS's distractor set is 4-5 orders of magnitude larger, spans broader domains, and remains far from saturated.
- **vs GLDv2 (Google Landmarks)**: GLDv2 focuses exclusively on the landmark domain, while ILIAS is intentionally multi-domain. Experiments show that landmark-specialized models experience a substantial drop in performance on ILIAS.
- **vs ROxford/RParis**: These are re-annotated versions of Oxford/Paris that address label issues, yet they remain limited by their small scale and single-domain focus.
- This dataset comes from the VRG lab at CTU Prague, carrying a clear lineage and inheritance in dataset design from the same retrieval research tradition as the Oxford/Paris benchmarks.

## Rating
- Novelty: ⭐⭐⭐⭐ The temporal separation ground truth strategy is highly clever, and the large-scale multi-domain design fills a critical gap in evaluation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive benchmarking of over 73 models, covering 4 retrieval setups and revealing multiple valuable insights.
- Writing Quality: ⭐⭐⭐⭐ Description of dataset design decisions is clear, and the benchmarking analyses are in-depth.
- Value: ⭐⭐⭐⭐⭐ Provides a much-needed, far-from-saturated large-scale evaluation benchmark for the instance-level retrieval community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Instance-Level Composed Image Retrieval](../../NeurIPS2025/image_generation/instance-level_composed_image_retrieval.md)
- [\[CVPR 2025\] PatchDPO: Patch-level DPO for Finetuning-free Personalized Image Generation](patchdpo_patch-level_dpo_for_finetuning-free_personalized_image_generation.md)
- [\[CVPR 2025\] ZoomLDM: Latent Diffusion Model for Multi-Scale Image Generation](zoomldm_latent_diffusion_model_for_multi-scale_image_generation.md)
- [\[CVPR 2025\] Visual Lexicon: Rich Image Features in Language Space](visual_lexicon_rich_image_features_in_language_space.md)
- [\[CVPR 2025\] RayFlow: Instance-Aware Diffusion Acceleration via Adaptive Flow Trajectories](rayflow_instance-aware_diffusion_acceleration_via_adaptive_flow_trajectories.md)

</div>

<!-- RELATED:END -->

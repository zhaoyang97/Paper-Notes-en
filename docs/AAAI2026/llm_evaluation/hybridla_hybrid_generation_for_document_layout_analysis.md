---
title: >-
  [Paper Note] HybriDLA: Hybrid Generation for Document Layout Analysis
description: >-
  [AAAI 2026][LLM Evaluation][Document layout analysis] HybriDLA is the first approach to unify diffusion-based bounding box refinement and autoregressive query expansion within a single decoding layer, simulating a human coarse-to-fine reading strategy for document layout analysis. It achieves 83.5% mAP on DocLayNet with a vision-only model, approaching multimodal systems.
tags:
  - AAAI 2026
  - LLM Evaluation
  - Document layout analysis
  - diffusion models
  - autoregressive generation
  - hybrid decoding
  - multi-scale feature fusion
date: 2026-05-08
content_hash: 4bafa26a1d7984e9
---

# HybriDLA: Hybrid Generation for Document Layout Analysis

**Conference**: AAAI 2026
**arXiv**: [2511.19919](https://arxiv.org/abs/2511.19919)
**Code**: [GitHub](https://yufanchen96.github.io/projects/HybriDLA)
**Area**: Document Analysis / Object Detection
**Keywords**: Document layout analysis, diffusion models, autoregressive generation, hybrid decoding, multi-scale feature fusion

## TL;DR

HybriDLA is the first approach to unify diffusion-based bounding box refinement and autoregressive query expansion within a single decoding layer, simulating a human coarse-to-fine reading strategy for document layout analysis. It achieves 83.5% mAP on DocLayNet with a vision-only model, approaching multimodal systems.

## Background & Motivation

Document layout analysis (DLA) is a fundamental task for document understanding and information extraction. Modern documents exhibit increasing structural complexity, with the number of layout elements per page ranging from as few as 2 to as many as 200. Traditional methods such as Faster R-CNN rely on a fixed number of region proposals, while DETR-based methods employ a fixed set of learnable queries. When the actual number of elements deviates significantly from the preset query count, fixed-query schemes either miss detections or introduce a large number of spurious "empty object" queries, incurring dual costs in efficiency and accuracy.

Existing diffusion-based detectors (e.g., DiffusionDet) introduce iterative refinement but still depend on a fixed-size pool of initial noisy boxes; autoregressive detectors (e.g., Pix2Seq) support variable-length sequences but incur linearly growing inference costs and lack spatial refinement capability.

HybriDLA's core motivation is to emulate the human document reading strategy: first scanning the entire page to grasp major regions, then progressively zooming into each region and dynamically adjusting the granularity of attention. Accordingly, the authors complementarily integrate diffusion-based refinement (responsible for iterative denoising of spatial coordinates) and autoregressive expansion (responsible for semantic awareness and dynamic query generation) within a unified decoder.

## Method

### Overall Architecture

HybriDLA adopts a two-stage hierarchical generation pipeline: a Feature Fusion Encoder (FFE) followed by a Hybrid Generative Decoder. The encoder processes multi-scale features extracted by the backbone to produce coarse layout priors; the decoder leverages these priors for autoregressive query expansion and diffusion-based box refinement, progressively generating precise layout elements and semantic labels in a coarse-to-fine manner across decoding layers.

### Key Designs

1. **Feature Fusion Encoder (FFE)**:

   - Function: Fuses the backbone's multi-scale feature maps $F_{l=1}^{L}$ into a unified spatially-aware representation $G$.
   - Mechanism: Consists of two steps—local feature encoding and cross-scale fusion. Local encoding $H_l = \phi(F_l)$ combines self-attention and convolution to capture long-range dependencies and local texture patterns within each scale; cross-scale fusion $G = \Psi(H_{l=1}^L)$ employs cross-scale attention and lateral convolutional layers to enable adaptive information exchange across scales, allowing fine-grained feature maps to acquire global context and coarse feature maps to acquire local detail.
   - Design Motivation: Layout elements in documents vary dramatically in scale (titles, footnotes, and figures differ greatly in size), making single-scale representations insufficient to capture both global structure and local detail simultaneously.

2. **Autoregressive Query Expansion (AQE)**:

   - Function: Models the query generation process autoregressively, dynamically determining how many queries to generate and their semantic content.
   - Mechanism: Given image features $X$, the model defines a joint distribution over a variable-length query sequence $Q = (q_1, q_2, \dots, q_N)$, factorized as $P(Q|X) = \prod_{t=1}^{N} P(q_t | X, q_{1:t-1}) \cdot P(\text{EOS} | X, q_{1:N})$. At each step, the next query is conditioned on the existing query context, and expansion terminates adaptively via a learned EOS criterion.
   - Design Motivation: The number of elements varies enormously across documents, inherently limiting fixed-query methods such as DETR. The autoregressive formulation enables the model to condition queries on prior context and dynamically adjust the query count according to document complexity.

3. **Diffusion-based Refinement (DR)**:

   - Function: Models layout prediction as an implicit denoising operation, applying residual corrections to current predictions at each decoding layer.
   - Mechanism: The update rule is $\hat{y}^{(t+1)} = \hat{y}^{(t)} + \Delta^{(t)}$, where $\Delta^{(t)}$ denotes the predicted residual at step $t$. Within each decoding layer, self-attention enables queries to share contextual information, cross-attention integrates visual features, and the feed-forward network applies the residual correction.
   - Design Motivation: Directly predicting precise coordinates is difficult; iterative denoising progressively eliminates spatial errors. During training, a subset of queries is initialized with perturbed ground-truth boxes (denoising queries), and layer-wise intermediate supervision is applied to accelerate convergence.

### Loss & Training

- Set prediction loss with Hungarian matching (consistent with DETR).
- Denoising training: a subset of queries is initialized from perturbed ground-truth boxes, forcing the network to recover correct layouts from degraded inputs.
- Layer-wise intermediate supervision: each decoding layer has auxiliary loss heads to ensure intermediate predictions remain close to ground truth.
- All models are trained with a unified batch size of 40 for fair comparison.

## Key Experimental Results

### Main Results (DocLayNet)

| Method Type | Backbone | Detector | mAP (%) |
|---|---|---|---|
| Traditional region-based | ResNet-101 | Mask R-CNN | 73.5 |
| DETR-based | InternImage | RoDLA | 80.5 |
| DETR-based (multimodal) | — | DLAFormer | 83.8 |
| Diffusion-based | Swin-L | DiffusionDet | 76.3 |
| Autoregressive | ViT-L | Pix2Seq | 72.5 |
| **Hybrid (Ours)** | **InternImage** | **HybriDLA** | **83.5** |
| Hybrid (Ours) | Swin-L | HybriDLA | 80.4 |
| Hybrid (Ours) | ResNet-50 | HybriDLA | 74.4 |

HybriDLA achieves 83.5% mAP with vision-only input, trailing the multimodal DLAFormer by only 0.3%. Compared to methods using the same backbone, it achieves an average gain of approximately 3% mAP.

### Main Results (M6Doc, 74 categories)

| Method Type | Backbone | Detector | mAP (%) |
|---|---|---|---|
| Traditional region-based | DiT | Cascade R-CNN | 70.2 |
| DETR-based | InternImage | RoDLA | 70.0 |
| Diffusion-based | Swin-L | DiffusionDet | 62.7 |
| **Hybrid (Ours)** | **InternImage** | **HybriDLA** | **71.4** |
| Hybrid (Ours) | ViT-L | HybriDLA | 68.6 |

### Ablation Study

| Configuration | mAP (%) | Note |
|---|---|---|
| DETR baseline (ResNet-50) | 74.2 | No AQE |
| + AQE | 74.4 | Marginal gain from autoregressive expansion |
| Deformable DETR + AQE | 76.3 | Larger gain with stronger baseline |
| DINO + AQE | 76.8 | +1.5% |
| DE + DR + AQE (Swin-L) | 78.1 | Standard encoder |
| FFE + AQE (Swin-L) | 79.1 | FFE outperforms DE by 1.0% |
| FFE + DR + AQE (Swin-L) | 80.4 | DR adds further 1.3% |
| FFE + DR + AQE (InternImage) | 83.5 | Best configuration |

### Key Findings

- The gain from AQE is more pronounced with stronger baseline detectors, indicating complementarity.
- FFE yields clear benefits with large, feature-rich backbones (Swin-L: +2.3% vs. DE) but may degrade performance on smaller models.
- DR provides consistent gains across nearly all backbones (0.8%–1.3%), with no change observed on ResNet-50.
- Query count analysis reveals an optimal query budget per model: smaller models saturate at 30 expanded queries, while larger models require up to 300.

## Highlights & Insights

- HybriDLA is the first to unify diffusion-based and autoregressive generation paradigms for document layout analysis, representing a conceptually novel contribution.
- The cognitively-motivated coarse-to-fine strategy has clear methodological correspondences: AQE corresponds to "scanning the page to discover new regions," and DR corresponds to "focusing on details for precise localization."
- The vision-only model nearly matches the multimodal system (83.5% vs. 83.8%), demonstrating that significant headroom remains in exploiting purely visual features.
- The architecture is backbone-agnostic and integrates seamlessly with ResNet, ViT, Swin, InternImage, and other backbone families.

## Limitations & Future Work

- The method relies solely on visual input and does not leverage multimodal signals such as OCR text and coordinate metadata, potentially limiting semantic disambiguation.
- The hybrid generation mechanism incurs substantial inference cost, constraining real-time processing and large-scale deployment.
- Future work may incorporate multimodal features or document metadata to enhance semantic understanding.
- Model distillation or architectural optimization could reduce inference overhead.

## Related Work & Insights

- The DETR family (DINO, Deformable DETR) provides a solid foundation for set prediction, and HybriDLA demonstrates the value of incorporating generative components on top of these baselines.
- The iterative denoising concept from DiffusionDet is elegantly combined with autoregressive expansion, highlighting the complementarity of different generative paradigms in detection tasks.
- The hybrid generation strategy proposed in this work is generalizable to general object detection in scenes with highly variable numbers of elements.

## Rating

- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Axis-Aligned Document Dewarping](axis-aligned_document_dewarping.md)
- [\[ICCV 2025\] Lay2Story: Extending Diffusion Transformers for Layout-Togglable Story Generation](../../ICCV2025/llm_evaluation/lay2story_extending_diffusion_transformers_for_layout-togglable_story_generation.md)
- [\[AAAI 2026\] Structured Language Generation Model: Loss Calibration and Formatted Decoding for Efficient Text](structured_language_generation_model_loss_calibration_and_formatted_decoding_for.md)
- [\[ACL 2026\] CAST: Achieving Stable LLM-based Text Analysis for Data Analytics](../../ACL2026/llm_evaluation/cast_achieving_stable_llm-based_text_analysis_for_data_analytics.md)
- [\[NeurIPS 2025\] Document Summarization with Conformal Importance Guarantees](../../NeurIPS2025/llm_evaluation/document_summarization_with_conformal_importance_guarantees.md)

</div>

<!-- RELATED:END -->

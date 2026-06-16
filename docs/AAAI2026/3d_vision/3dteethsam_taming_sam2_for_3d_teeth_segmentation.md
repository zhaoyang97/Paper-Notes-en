---
title: >-
  [Paper Note] 3DTeethSAM: Taming SAM2 for 3D Teeth Segmentation
description: >-
  [AAAI 2026][3D Vision][3D teeth segmentation] This work adapts the SAM2 foundation model for 3D teeth segmentation by converting 3D meshes into 2D images via multi-view rendering and designing three lightweight adapters—…
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "3D teeth segmentation"
  - "SAM2 adaptation"
  - "multi-view rendering"
  - "deformable attention"
  - "foundation model transfer"
date: 2026-05-08
content_hash: 4100f16abae870e4
---

# 3DTeethSAM: Taming SAM2 for 3D Teeth Segmentation

**Conference**: AAAI 2026
**arXiv**: [2512.11557](https://arxiv.org/abs/2512.11557)  
**Code**: [https://github.com/Crisitofy/3DTeethSAM](https://github.com/Crisitofy/3DTeethSAM)  
**Area**: 3D Vision / Medical Image Segmentation
**Keywords**: 3D teeth segmentation, SAM2 adaptation, multi-view rendering, deformable attention, foundation model transfer

## TL;DR
This work adapts the SAM2 foundation model for 3D teeth segmentation by converting 3D meshes into 2D images via multi-view rendering and designing three lightweight adapters—a Prompt Embedding Generator, a Mask Refiner, and a Mask Classifier—along with a Deformable Global Attention Plugin (DGAP) to address automatic prompting, boundary refinement, and semantic classification. The proposed method achieves a new state-of-the-art T-mIoU of 91.90% on Teeth3DS.

## Background & Motivation
3D teeth segmentation is a fundamental task in digital dentistry, requiring the localization and classification of individual tooth instances within a 3D dental model. Existing methods primarily rely on specialized networks that directly process 3D point clouds or meshes (e.g., PointNet++, MeshSegNet, TSGCNet), and suffer from two critical bottlenecks: (1) these networks trained from scratch are difficult to scale to high-resolution 3D models; and (2) they cannot leverage knowledge from large-scale pretrained models. Meanwhile, SAM2, as a 2D visual foundation model, has demonstrated strong zero-shot capability across various downstream tasks. However, transferring it to 3D teeth segmentation faces three major challenges: dimensionality mismatch, dependence on manual prompts, and category-agnostic outputs.

## Core Problem
How can SAM2, a 2D foundation model, be effectively adapted for 3D teeth segmentation? Specifically, three issues must be resolved: (1) SAM2 relies on manual point/box prompts and cannot operate automatically; (2) SAM2's raw segmentation results exhibit coarse boundaries; and (3) SAM2 is category-agnostic and cannot distinguish different tooth IDs. These three problems collectively prevent direct application of SAM2 to high-accuracy, fully automated 3D teeth segmentation.

## Method

### Overall Architecture
The full pipeline consists of three stages: (1) **Multi-view rendering**: the 3D tooth mesh is normalized and rendered from fixed viewpoints (front, back, and multiple lateral views) into 512×512 RGB images; (2) **SAM2-adapted segmentation**: SAM2's pretrained weights are frozen, and three lightweight adapters along with DGAP are applied to segment the 2D images, producing 16-channel masks (each channel corresponding to one tooth); (3) **2D-to-3D lifting**: back-projection maps the 2D segmentation results onto 3D mesh vertices, multi-view results are aggregated via voting, and Graph Cut post-processing is applied to refine boundaries.

### Key Designs
1. **Prompt Embedding Generator (PEG)**: Inspired by DETR, a Transformer Decoder transforms 16 randomly initialized query vectors into prompt embeddings. Self-attention models spatial relationships among teeth, while cross-attention aligns with image features. A confidence score is additionally learned to handle missing teeth (higher values indicate greater probability of a tooth instance being present), fully replacing SAM2's dependence on manual prompts.

2. **Mask Refiner**: A UNet-based convolutional network that takes three inputs: the original tooth image (providing low-level texture/shape details), the coarse mask from SAM2 (providing spatial priors), and SAM2's image embeddings (providing high-level semantics). In the contracting path of the UNet, each layer contains three parallel streams processing the three inputs respectively, which are then concatenated and forwarded. This design specifically addresses the imprecise boundaries caused by SAM2's general-purpose pretraining.

3. **Mask Classifier**: Also adopts a Transformer Decoder architecture (sharing the design but not parameters with PEG), transforming 16 query vectors into class probability vectors. An MLP followed by Softmax outputs 17-class probabilities (16 teeth + background). This is more robust than a simple channel-to-tooth-ID binding strategy and avoids channel–ID misalignment in missing-tooth scenarios.

4. **Deformable Global Attention Plugin (DGAP)**: Integrated into the global attention blocks in stage 3 of SAM2's Hiera image encoder. An offset network predicts offsets to deform the sampling grid, directing attention toward tooth regions. Unlike standard deformable attention, query/key/value are all predicted from deformed feature maps, and deformed and non-deformed features are fused via a skip connection. DGAP is a plug-and-play module that requires no modification to SAM2's internal implementation.

### Loss & Training
- **Training strategy**: SAM2 pretrained weights are frozen; only the three adapters and DGAP are trained. The Hungarian algorithm is used for one-to-one matching between predicted queries and ground truth. AdamW optimizer, learning rate 2e-4, cosine annealing with 5-epoch warmup, 100 epochs, batch size 4, mixed precision.
- **Total loss**: $L_{\text{total}} = \lambda_{MC} L_{MC} + \lambda_{PEG} L_{PEG} + \lambda_{MR} L_{MR}$, with weights 1.0, 1.0, and 2.0 respectively.
    - $L_{MC}$: 17-class cross-entropy loss (Mask Classifier)
    - $L_{PEG}$: BCE + Dice + confidence loss (Prompt Embedding Generator)
    - $L_{MR}$: multi-class CE + Dice + boundary loss (Mask Refiner; boundary loss computes L1 distance of gradients via Sobel filtering)

## Key Experimental Results

Dataset: **Teeth3DS** (1,800 high-resolution intraoral 3D scans from 900 patients, official 1200/600 split)

| Dataset | Metric | Ours | Prev. SOTA (ToothGroupNet) | Gain |
|--------|------|------|----------|------|
| Teeth3DS | OA | 95.48% | 95.19% | +0.29% |
| Teeth3DS | T-mIoU | 91.90% | 90.16% | +1.74% |
| Teeth3DS | B-IoU | 70.05% | 69.30% | +0.75% |
| Teeth3DS | Dice | 94.33% | — | — |
| Teeth3DS | Wisdom Teeth T-mIoU (T8/16) | 83.29% | 68.20% | +15.09% |

### Ablation Study
- **PEG is the most critical module**: Removing it causes T-mIoU to drop by 39.44% (91.90%→52.46%). Even using ground-truth center points as manual prompts falls far short of the learned prompt embeddings, indicating that PEG captures complex spatial relationships and contextual information.
- **DGAP**: Removing it reduces T-mIoU by 1.29% and B-IoU by 3.41%, and significantly slows training convergence.
- **Mask Refiner**: Removing it reduces T-mIoU by 0.80% and B-IoU by 1.62%, primarily affecting boundary quality.
- **Mask Classifier**: Removing it reduces T-mIoU by 0.59% and B-IoU by 2.49%, mainly addressing category confusion between adjacent teeth.

## Highlights & Insights
- **"Render → 2D Segmentation → Back-projection" paradigm**: This approach elegantly reduces 3D segmentation to a 2D problem, enabling direct exploitation of powerful 2D foundation models and offering a general and reusable framework.
- **DETR-style design of PEG**: Using a Transformer Decoder to automatically generate prompt embeddings completely bypasses SAM2's reliance on manual prompts while modeling spatial relationships among teeth.
- **Plug-and-play DGAP**: Without modifying SAM2's internals, DGAP fuses deformed and non-deformed features via skip connections, improving both accuracy and training efficiency, and is generalizable to other foundation model adaptation scenarios.
- **Substantial improvement on wisdom tooth segmentation**: A 15%+ gain on this rare category demonstrates the advantage of foundation models in data-scarce settings.

## Limitations & Future Work
- Multi-view rendering introduces additional computational overhead; inference efficiency may lag behind methods that directly process 3D data.
- Validation is performed on only one dataset (Teeth3DS), leaving generalization to different scanners and diverse dental morphologies unknown.
- Fixed-viewpoint rendering may miss details from certain angles (e.g., severely crowded teeth); adaptive view selection could be more effective.
- The 2D-to-3D voting strategy is relatively simple; more sophisticated multi-view fusion schemes (e.g., learnable fusion weights) may yield further improvements.
- The paper does not discuss real-time inference or clinical deployment feasibility.

## Related Work & Insights
- **vs. ToothGroupNet**: The previous state-of-the-art operates directly on 3D meshes. 3DTeethSAM surpasses it via 2D foundation model transfer, with particularly large advantages on rare categories (wisdom teeth +15%), at the cost of additional overhead from multi-view rendering.
- **vs. MedSAM**: MedSAM adapts SAM to 2D medical images but does not handle 3D data. 3DTeethSAM resolves the 2D–3D dimensionality mismatch through its render–segment–back-project pipeline.
- **vs. traditional 3D networks (PointNet++, DGCNN, etc.)**: These methods train from scratch, cannot leverage pretrained knowledge, and scale poorly to high-resolution meshes. 3DTeethSAM freezes SAM2 weights and trains only lightweight adapters, achieving higher parameter efficiency.
- **Generalizable 3D segmentation paradigm**: The render → 2D foundation model → back-projection approach can be extended to other 3D medical segmentation tasks (e.g., bones, organs) and non-medical 3D segmentation (e.g., indoor scenes, autonomous driving point clouds).
- **Adaptive view selection**: Current fixed viewpoints could be replaced by a learnable view selection module that dynamically determines rendering angles based on mesh complexity.
- **Multi-foundation-model fusion**: SAM2 handles segmentation; additional foundation models (e.g., DINOv2) could be introduced to provide richer semantic features.
- **End-to-end 3D foundation models**: The current approach relies on a 2D intermediary; future work may explore training SAM-like foundation models directly in 3D space.

## Rating
- Novelty: ⭐⭐⭐⭐ The render + SAM2 adaptation approach is innovative, though individual components (DETR-style queries, UNet refiner, deformable attention) all have precedents.
- Experimental Thoroughness: ⭐⭐⭐⭐ Ablation studies are comprehensive with 11-method comparisons, but evaluation is limited to a single dataset.
- Writing Quality: ⭐⭐⭐⭐ Structure is clear, method descriptions are precise, and figures are intuitive.
- Value: ⭐⭐⭐⭐ Demonstrates a viable pathway from 2D foundation models to 3D segmentation, with practical significance for digital dentistry and a generalizable paradigm.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] GeoSAM2: Unleashing the Power of SAM2 for 3D Part Segmentation](../../CVPR2026/3d_vision/geosam2_unleashing_the_power_of_sam2_for_3d_part_segmentation.md)
- [\[AAAI 2026\] UniC-Lift: Unified 3D Instance Segmentation via Contrastive Learning](unic-lift_unified_3d_instance_segmentation_via_contrastive_learning.md)
- [\[AAAI 2026\] ASSIST-3D: Adapted Scene Synthesis for Class-Agnostic 3D Instance Segmentation](assist-3d_adapted_scene_synthesis_for_class-agnostic_3d_instance_segmentation.md)
- [\[AAAI 2026\] Retrieving Objects from 3D Scenes with Box-Guided Open-Vocabulary Instance Segmentation](retrieving_objects_from_3d_scenes_with_box-guided_open-vocabulary_instance_segme.md)
- [\[AAAI 2026\] MR-CoSMo: Visual-Text Memory Recall and Direct Cross-Modal Alignment Method for Query-Driven 3D Segmentation](mr-cosmo_visual-text_memory_recall_and_direct_cross-modal_alignment_method_for_q.md)

</div>

<!-- RELATED:END -->

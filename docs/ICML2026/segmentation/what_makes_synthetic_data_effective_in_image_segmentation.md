---
title: >-
  [Paper Note] What Makes Synthetic Data Effective in Image Segmentation
description: >-
  [ICML2026][Segmentation][Semantic Segmentation] This paper systematically analyzes two key factors making synthetic images effective for semantic segmentation: complex scene composition and high instance fidelity. It pro…
tags:
  - "ICML2026"
  - "Segmentation"
  - "Semantic Segmentation"
  - "Synthetic Data"
  - "Diffusion Models"
  - "Optimal Transport"
  - "Pseudo-labeling"
date: 2026-05-08
content_hash: a5228ff3e3cb07f6
---

# What Makes Synthetic Data Effective in Image Segmentation

**Conference**: ICML2026  
**arXiv**: [2605.19289](https://arxiv.org/abs/2605.19289)  
**Code**: https://github.com/zhang0jhon/SENSE  
**Area**: Semantic Segmentation / Synthetic Data  
**Keywords**: Semantic Segmentation, Synthetic Data, Diffusion Models, Optimal Transport, Pseudo-labeling

## TL;DR
This paper systematically analyzes two key factors making synthetic images effective for semantic segmentation: complex scene composition and high instance fidelity. It proposes SENSE, which leverages Optimal Transport (OT) to stabilize pseudo-label assignment for synthetic images, achieving consistent improvements for DPT and Mask2Former on Cityscapes, COCO, and ADE20K.

## Background & Motivation
**Background**: Diffusion models and flow matching models can generate high-quality images, leading to the widespread use of synthetic data in classification, detection, segmentation, and robotics. Semantic segmentation particularly relies on pixel-level annotations; since real annotations are costly and long-tail classes are hard to collect, augmenting training sets with generative models is a natural direction.

**Limitations of Prior Work**: Many previous works have proven that "synthetic data is useful," but few answer "what kind of synthetic data is useful." If focusing only on visual quality, models might fail to learn multi-object co-occurrence and boundary details in real scenes. If using input masks from conditional models like ControlNet directly as labels, local semantic misalignment between generated images and masks can introduce pseudo-label noise.

**Key Challenge**: Segmentation requires both global semantic context and local pixel boundaries. Synthetic data with only single objects or sparse scenes makes it difficult for models to handle complex layouts in real street views or indoor scenes. Insufficient instance edges, textures, and high-frequency details prevent the learning of precise boundaries. Even with high image quality, label assignment must adapt to generation randomness rather than blindly trusting the initial conditions.

**Goal**: The authors first identify key factors for synthetic data effectiveness through controlled experiments, then design a model-agnostic SENSE framework. This framework incorporates high-quality synthetic images into fixed real datasets and mitigates pseudo-label inconsistency via OT assignment.

**Key Insight**: The problem is decomposed into "whether the image itself is suitable for segmentation" and "whether the supervision is reliable." The former is analyzed via comparative experiments with sparse/dense scenes and coarse/fine instances. The latter is addressed using entropy-regularized Optimal Transport, treating the assignment of pixels to classes as a global optimization problem rather than independent pixel-wise hard assignment.

**Core Idea**: Effective synthetic data for segmentation must possess both dense scene composition and fine instance fidelity. SENSE converts imperfect synthetic images into stable, scalable semi-supervised signals via OT.

## Method
The methodology of SENSE is straightforward: first determine what kind of synthetic images to generate, then decide how to produce reliable supervision. The authors found that generative models like Flux/Flux-WLF produce images with multiple objects, rich spatial relationships, and good boundary details, and thus use them to generate samples corresponding to Cityscapes, COCO, and ADE20K. During training, real images use real labels, while synthetic images use soft class probabilities predicted by the current model, which are globally redistributed into stable pseudo-labels via OT.

### Overall Architecture
The input includes labeled real data $\mathcal{D}_R=\{(x_i,y_i)\}$ and unlabeled synthetic images $\mathcal{D}_S=\{\tilde{x}_i\}$. SENSE trains real and synthetic samples simultaneously within a mini-batch: real samples use standard cross-entropy or Mask2Former set prediction loss. Synthetic samples undergo weak augmentation to obtain predicted probabilities, forming a transport cost from pixels to classes. An entropy-regularized OT plan is solved via Sinkhorn-Knopp and used as soft supervision on strong augmentations. This workflow supports both pixel-based models like DPT and query-based models like Mask2Former.

### Key Designs
1. **Dual-factor Analysis of Synthetic Data Quality**:
	- Function: Identify actionable factors in synthetic images that truly impact segmentation performance.
	- Mechanism: The authors construct sparse vs. dense composition data, controlling prompts and generative models while using GroundingDINO instance counts as a proxy for semantic density. Coarse vs. fine fidelity data is also constructed, using GLCM Score and Compression Ratio to approximate high-frequency texture and boundary fidelity. All synthetic images are labeled by a teacher trained on real data to isolate annotator bias.
	- Design Motivation: Semantic segmentation requires multi-class co-occurrence, spatial relationships, and clear boundaries, not just "realism." This analysis transforms synthetic data design from empirical prompt engineering into measurable objectives.

2. **OT Assignment for Stable Synthetic Pseudo-labels**:
	- Function: Mitigate local semantic misalignment and pixel-wise confirmation bias in generated images.
	- Mechanism: For each pixel and class in a synthetic image, a cost $c_{ij}(h,w)=-\log p_\theta(j|\tilde{x}_i(h,w))$ is constructed. After flattening pixels into an $n\times k$ matrix, the authors solve $\min_{\pi}\langle \pi,c\rangle+\beta H(\pi)$. Uniform marginal constraints prevent the amplification of long-tail biases. The approximate solution $\pi^*=\mathrm{diag}(u)K\mathrm{diag}(v)$, where $K=\exp(-c/\beta)$, is obtained via Sinkhorn iterations.
	- Design Motivation: Standard pseudo-labeling takes the max class for each pixel independently, which can solidify local hallucinations into incorrect supervision. OT forces label assignment to meet global class distribution constraints, providing more stable soft supervision on noisy synthetic images.

3. **Unified Support for Pixel-based and Query-based Segmenters**:
	- Function: Adapt the synthetic data utilization strategy to both DPT and Mask2Former architectures.
	- Mechanism: For DPT, SENSE computes the OT plan directly on pixel probability maps and trains with strong augmentation using a confidence threshold $\gamma=0.95$. For Mask2Former, query class-mask pairs are first aggregated into per-pixel class probabilities for OT in pixel space, then the refined targets are mapped back to query supervision via bipartite matching.
	- Design Motivation: Existing OT semi-supervised methods are mostly limited to dense pixel classifiers. SENSE generalizes by recognizing that query models eventually define a dense pixel semantic decision surface, allowing global assignment in the projected pixel space.

### Loss & Training
The synthetic loss for pixel-based segmenters is an OT soft-label cross-entropy with confidence gating; the real loss is standard pixel-wise cross-entropy. The total loss is their average. For query-based segmenters, the synthetic loss includes classification and mask terms; for stability, the Dice loss weight for synthetic data is set to 0, keeping only the BCE mask supervision to avoid distorting query gradients with small erroneous regions. Training uses AdamW, mixed precision, EMA, and weak/strong augmentation. Per batch: 8 real + 8 synthetic for Cityscapes/ADE20K; 16 + 16 for COCO. OT regularization $\beta=0.05$, pseudo-label thresholds $\gamma, \delta=0.95$.

## Key Experimental Results

### Main Results
| Dataset | Metric | Ours | Prev. / Real Baseline | Gain |
|--------|------|------|----------|------|
| Cityscapes, DPT DINOv2-S | mIoU s.s. | 80.65 | 78.11 real only | +2.54 |
| Cityscapes, Mask2Former DINOv3-L | mIoU m.s. | 84.88 | 83.29 real only | +1.59 |
| COCO, DPT DINOv2-S | mIoU m.s. | 64.96 | 63.40 real only | +1.56 |
| ADE20K, Mask2Former DINOv3-L | mIoU s.s. | 59.09 | 57.45 real only | +1.64 |
| ADE20K scalable synthetic methods | mIoU m.s. | 60.81 | SegGen 58.7 | +2.11 |
| ADE20K Swin-L fair comparison | mIoU | 58.27 | JoDiffusion 57.46 / SDS 57.23 | +0.81 / +1.04 |

### Ablation Study
| Configuration | Key Metric | Description |
|------|---------|------|
| Dense vs Sparse composition, Flux | Cityscapes 66.56 vs 61.81 mIoU | Average instances from 11.48 to 22.21; dense scenes are significantly more beneficial. |
| Fine vs Coarse fidelity | Cityscapes 68.17 vs 66.56 mIoU | With similar instance counts, high-frequency boundary and texture fidelity add +1.61. |
| Synthetic scale on Cityscapes | 79.80 → 81.27 mIoU | Performance improves as synthetic data increases from 1× to 6×, with diminishing returns. |
| w/o OT vs OT, Cityscapes | 79.50 → 80.65 mIoU | OT assignment provides +1.15 mIoU. |
| w/o OT vs OT, COCO / ADE20K | 62.74→63.30 / 49.62→50.23 | OT provides consistent gains across three datasets. |
| Synthetic Quality Ladder | 78.98 → 79.49 → 79.80 | From sparse/low-fidelity to dense/high-fidelity, SENSE validates the dual-factor conclusion. |

### Key Findings
- Global semantic density of synthetic images is crucial. The Flux dense split (avg. 22.21 instances, 66.56 mIoU) significantly outperforms the sparse split (11.48 instances, 61.81 mIoU).
- Local instance fidelity remains effective after controlling for scene composition. Flux-WLF fine fidelity data boosts mIoU from 66.56 to 68.17, showing independent contributions from boundaries and textures.
- SENSE improvements are architecture-agnostic: gains are observed across DPT, Mask2Former, DINOv2-S/B, and DINOv3-L without increasing inference overhead.
- Compared to FreeMask and SegGen, SENSE exceeds methods using 20×/50× synthetic data with only 2× synthetic data, proving "data quality + label assignment" is more important than blind scaling.

## Highlights & Insights
- The paper first answers "what data is useful" before proposing a framework, making it more robust than simple synthetic pipelines. Both dense composition and fine fidelity factors are quantifiable, guiding future generative models and prompt strategies.
- OT assignment is the most critical bridge. It acknowledges the misalignment between synthetic images and labels, replacing local max-probability with global constraints to create smoother, noise-resistant supervision.
- The extension for query-based models is practical. While many semi-supervised methods are stuck on pixel classifiers, this work projects Mask2Former query outputs back to pixel space for OT and maps them back to set prediction loss.
- Scaling ablations show performance can continue to rise with data volume, but the dual-factor analysis suggests quantity isn't everything. Low-quality or low-density data increases training costs without providing scarce spatial structures.

## Limitations & Future Work
- Generation costs are not fully discussed. While Flux/Flux-WLF quality is high, generating 2× synthetic data for COCO/ADE20K requires significant compute; real-world deployment needs to balance generation and annotation budgets.
- Evaluation focuses on closed-set semantic segmentation. For open-vocabulary, panoptic, or instance segmentation, the roles of density/fidelity might change, and OT constraints would need redesigning for class margins.
- Synthetic images are still determined by MLLM prompts and generative model distributions, which may introduce implicit biases in co-occurrence or regional scenes, potentially affecting fairness.
- Uniform marginals in OT help with long-tail issues but might over-smooth when real class distributions are highly imbalanced. Future work could learn or estimate class priors closer to the dataset.

## Related Work & Insights
- **vs DatasetDM / DiffuMask**: These focus on generating images and perception labels but have limited class coverage and scalability; SENSE emphasizes large-scale scalable synthesis and robust pseudo-label assignment.
- **vs FreeMask / SegGen**: While these use massive synthetic data to boost performance, SENSE achieves higher ADE20K mIoU with less data, indicating data selection and supervision quality are the true bottlenecks.
- **vs SLA / OTAMatch**: These OT-based semi-supervised methods are for pixel-wise architectures; SENSE generalizes OT to query-based segmentation like Mask2Former.
- **Inspiration**: For other dense prediction tasks (depth, normals, remote sensing), one can first diagnose task-relevant attributes of synthetic data, then use global assignment to reduce generation-label mismatch.

## Rating
- Novelty: ⭐⭐⭐⭐ "Synthetic data + semi-supervised segmentation" isn't new, but the dual-factor analysis and query-based OT extension are well-combined.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers Cityscapes, COCO, ADE20K, multiple architectures, backbones, synthetic scales, and OT ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear main logic and sufficient tabular data; some generative model details and appendices require careful reading.
- Value: ⭐⭐⭐⭐⭐ Highly relevant for those using diffusion models to augment datasets: prioritize complex scenes and instance details, then use robust label assignment over simple quantity increases.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] UnrealPose: Leveraging Game Engine Kinematics for Large-Scale Synthetic Human Pose Data](../../CVPR2026/segmentation/unrealpose_leveraging_game_engine_kinematics_for_large-scale_synthetic_human_pos.md)
- [\[ICCV 2025\] LEGION: Learning to Ground and Explain for Synthetic Image Detection](../../ICCV2025/segmentation/legion_learning_to_ground_and_explain_for_synthetic_image_detection.md)
- [\[ICCV 2025\] Learn2Synth: Learning Optimal Data Synthesis Using Hypergradients for Brain Image Segmentation](../../ICCV2025/segmentation/learn2synth_learning_optimal_data_synthesis_using_hypergradients_for_brain_image.md)
- [\[CVPR 2026\] A Mixed Diet Makes DINO An Omnivorous Vision Encoder](../../CVPR2026/segmentation/a_mixed_diet_makes_dino_an_omnivorous_vision_encoder.md)
- [\[CVPR 2026\] Task-Oriented Data Synthesis and Control-Rectify Sampling for Remote Sensing Semantic Segmentation](../../CVPR2026/segmentation/task-oriented_data_synthesis_and_control-rectify_sampling_for_remote_sensing_sem.md)

</div>

<!-- RELATED:END -->

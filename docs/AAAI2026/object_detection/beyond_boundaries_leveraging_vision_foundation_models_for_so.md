---
title: >-
  [Paper Note] Beyond Boundaries: Leveraging Vision Foundation Models for Source-Free Object Detection
description: >-
  [AAAI 2026][Object Detection][Source-Free Object Detection] This paper proposes a framework leveraging VFMs (DINOv2 + Grounding DINO) to enhance Source-Free Object Detection (SFOD) via three modules: Patch-weighted Globa…
tags:
  - "AAAI 2026"
  - "Object Detection"
  - "Source-Free Object Detection"
  - "Vision Foundation Models"
  - "DINOv2"
  - "Grounding DINO"
  - "Pseudo-label Fusion"
date: 2026-05-08
content_hash: a4b65c417beb2b64
---

# Beyond Boundaries: Leveraging Vision Foundation Models for Source-Free Object Detection

**Conference**: AAAI 2026
**arXiv**: [2511.07301](https://arxiv.org/abs/2511.07301)
**Code**: [https://github.com/HuizaiVictorYao/VFM_SFOD](https://github.com/HuizaiVictorYao/VFM_SFOD)
**Area**: Object Detection
**Keywords**: Source-Free Object Detection, Vision Foundation Models, DINOv2, Grounding DINO, Pseudo-label Fusion

## TL;DR
This paper proposes a framework leveraging VFMs (DINOv2 + Grounding DINO) to enhance Source-Free Object Detection (SFOD) via three modules: Patch-weighted Global Feature Alignment (PGFA), Prototype-based Instance Feature Alignment (PIFA), and Dual-source Enhanced Pseudo-label Fusion (DEPF). The method achieves state-of-the-art results on 6 cross-domain detection benchmarks, e.g., 47.1% mAP on Cityscapes→Foggy Cityscapes (+3.5% over DRU) and 67.4% AP on Sim10k→Cityscapes (+8.7% over DRU).

## Background & Motivation
Source-Free Object Detection (SFOD) requires adapting a pretrained detector to the target domain without access to source data. Existing methods rely solely on the internal knowledge of the source model (via teacher-student distillation), leading to two issues: (1) feature transferability is constrained by the source model's semantic space, and (2) pseudo-labels are biased, especially under large domain shifts. Vision Foundation Models (VFMs) such as DINOv2 and Grounding DINO, pretrained on large-scale data, possess strong generalization and rich semantic priors, yet their potential in SFOD remains underexplored.

## Core Problem
How to effectively leverage external VFM knowledge to simultaneously enhance feature transferability and category discriminability in SFOD? Prior methods either require source data (violating SFOD constraints) or address only one dimension, failing to fully exploit VFM potential.

## Method

### Overall Architecture
Built upon a Mean Teacher self-training framework (EMA-updated teacher), the method incorporates VFM external knowledge along three dimensions: PGFA aligns global patch-level features using DINOv2; PIFA constructs class prototypes from DINOv2 features for instance-level contrastive learning; DEPF fuses predictions from Grounding DINO and the teacher to generate more reliable pseudo-labels. No additional parameters or computation are introduced at inference time.

### Key Designs
1. **Patch-weighted Global Feature Alignment (PGFA)**: Aligns patch features from the student backbone to DINOv2, but weights patches non-uniformly. A cosine similarity matrix is computed among DINOv2 patches, and a top-k weighting strategy assigns higher weights to patches with stronger semantic consistency. A weighted cosine loss then aligns the student feature space toward DINOv2. The key insight is that domain invariance varies substantially across patches, and semantically consistent regions are more suitable for alignment.

2. **Prototype-based Instance Feature Alignment (PIFA)**: Instance features are extracted from DINOv2 via RoIAlign, averaged per class, and maintained as class prototypes with EMA ($\mu=0.9$). Student instance features are then aligned to their corresponding prototypes via an InfoNCE contrastive loss. Momentum updates ensure prototype stability, while contrastive learning jointly improves inter-class discriminability and domain invariance.

3. **Dual-source Enhanced Pseudo-label Fusion (DEPF)**: Innovatively fuses detection boxes from both the teacher and Grounding DINO. Instead of conventional WBF (which suffers from label conflicts across sources), DEPF discards category labels and clusters boxes by IoU alone. Within each cluster, Shannon entropy is computed for each prediction, and box coordinates and class probabilities are fused using inverse-entropy weights. Low-entropy (high-confidence) predictions naturally receive larger weights, elegantly resolving multi-source label conflicts.

### Loss & Training
$\mathcal{L}_{tot} = \mathcal{L}_{det} + \lambda(\mathcal{L}_{pgfa} + \mathcal{L}_{pifa})$, with $\lambda=1$. Deformable DETR serves as the base detector; lr=$5 \times 10^{-5}$, batch size=8, training for 30 epochs. EMA coefficient is 0.999, with the teacher updated every 5 iterations. VFMs are entirely excluded at inference, introducing no additional overhead.

## Key Experimental Results

| Benchmark (SFOD) | Metric | Ours | Prev. SOTA (DRU) | Source Only |
|--------|------|------|----------|------|
| City→Foggy (weather) | mAP | **47.1** | 43.6 | 29.6 |
| City→BDD100K (scene) | mAP | **43.0** | 36.6 | 28.3 |
| Sim10k→City (synthetic→real) | AP (car) | **67.4** | 58.7 | 50.8 |
| KITTI→City (scene) | AP (car) | **54.7** | 45.1 | 33.9 |
| City→ACDC Snow | mAP | **47.9** | 37.9 | - |
| City→ACDC Fog | mAP | **54.0** | 45.4 | - |

Cross-detector generalization: Faster R-CNN +3.4%, RT-DETR +3.3%, YOLOv5 +2.4% mAP. Consistent gains are observed across Swin-T/S/B/L and ViT-B backbones.

### Ablation Study
- Incremental module stacking: MT baseline 42.3 → +PGFA 43.4 (+1.1) → +PIFA 43.9 (+1.6) → +PGFA+PIFA 45.0 (+2.7) → +DEPF 45.9 (+3.6) → All 47.1 (+4.8)
- DEPF contributes the most (+3.6), as pseudo-label quality directly determines the self-training upper bound
- Patch weighting (PGFA) and entropy weighting (DEPF) individually contribute +0.6 and +0.3 mAP
- VFM backbone comparison: DINOv2 ViT-G (47.1) > ViT-L (46.8) > ViT-B (46.7) > Grounding DINO Swin-B (46.2)
- No inference overhead; training time increases by 79%, primarily due to VFM feature extraction

## Highlights & Insights
- **Introducing VFMs into SFOD is a natural and effective idea**—the generalization capacity of VFMs precisely compensates for the domain bias of the source model
- **The entropy-guided fusion design in DEPF is elegant**—discarding category labels and clustering boxes by IoU alone, then fusing via inverse-entropy weights, cleanly resolves multi-source label conflicts
- VFMs are used only during training, incurring zero inference overhead and remaining deployment-friendly
- Consistent gains across 5 detector architectures and 6 backbones demonstrate strong generalizability
- Significant improvements are achieved even when the source model is weak (trained for only 5 epochs)

## Limitations & Future Work
- Training requires inference through two large VFMs (DINOv2 + Grounding DINO), increasing training time by 79%
- Grounding DINO requires text prompts, implicitly assuming that target-domain class names are known
- Gains are limited under extreme domain shifts (e.g., ACDC night), where mAP is only 23.0
- The potential of stronger VLMs (e.g., Qwen-VL, InternVL) as replacements for Grounding DINO remains unexplored

## Related Work & Insights
- **vs. DRU**: DRU also uses DETR but relies solely on internal knowledge; this paper introduces external VFM knowledge and consistently outperforms DRU (+3.5 to +9.6 mAP across benchmarks)
- **vs. DINO Teacher (DT)**: DT requires source data to train a DINOv2 labeler (violating source-free constraints); this paper achieves comparable performance without source data (4.8% gap under a stricter setting)
- **vs. CODA**: CODA enhances discriminability through external detection alone while neglecting transferability; this paper jointly optimizes both dimensions

The paradigm of using VFMs as "external knowledge anchors" generalizes naturally to other source-free domain adaptation tasks, such as segmentation and depth estimation. The entropy-guided fusion strategy is broadly applicable to any multi-detector output fusion scenario (e.g., ensemble methods). An intersection with model compression also exists: distilling VFM knowledge into lightweight detectors while simultaneously performing domain adaptation and model miniaturization.

## Rating
- Novelty: ⭐⭐⭐⭐ — First systematic integration of VFMs into SFOD, though individual technical components (feature alignment, prototype contrastive learning, box fusion) are not novel in isolation
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 6 benchmarks, 5 detectors, 6 backbones, with extensive ablation and analysis
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, rich figures and tables, convincing motivation
- Value: ⭐⭐⭐⭐ — High practical value and strong generalizability, though the SFOD setting has a relatively narrow application scope

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Foundation Model Priors Enhance Object Focus in Feature Space for Source-Free Object Detection](../../CVPR2026/object_detection/foundation_model_priors_enhance_object_focus_in_feature_space_for_source-free_ob.md)
- [\[ICLR 2026\] FSOD-VFM: Few-Shot Object Detection with Vision Foundation Models and Graph Diffusion](../../ICLR2026/object_detection/fsod-vfm_few-shot_object_detection_with_vision_foundation_models_and_graph_diffu.md)
- [\[ICCV 2025\] SFUOD: Source-Free Unknown Object Detection](../../ICCV2025/object_detection/sfuod_source-free_unknown_object_detection.md)
- [\[ICLR 2026\] CGSA: Class-Guided Slot-Aware Adaptation for Source-Free Object Detection](../../ICLR2026/object_detection/cgsa_class-guided_slot-aware_adaptation_for_source-free_object_detection.md)
- [\[AAAI 2026\] Temporal Object-Aware Vision Transformer for Few-Shot Video Object Detection](temporal_object-aware_vision_transformer_for_few-shot_video_object_detection.md)

</div>

<!-- RELATED:END -->

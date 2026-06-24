---
title: >-
  [Paper Note] PanoAffordanceNet: Towards Holistic Affordance Grounding in 360° Indoor Environments
description: >-
  [CVPR 2025][Robotics][Panoramic affordance] This paper proposes PanoAffordanceNet, the first 360° panoramic affordance grounding framework. It handles ERP latitude-dependent distortion using a Distortion-Aware Spectrum Modulator (DASM), restores sparse activations into topologically continuous areas via an Omnispherical Densification Head (OSDH), suppresses semantic drift with multi-level training objectives, and constructs the first panoramic affordance dataset 360-AGD…
tags:
  - "CVPR 2025"
  - "Robotics"
  - "Panoramic affordance"
  - "360-degree scene"
  - "ERP distortion"
  - "spherical propagation"
  - "one-shot learning"
date: 2026-05-08
content_hash: 519ef7c96ae2288f
---

# PanoAffordanceNet: Towards Holistic Affordance Grounding in 360° Indoor Environments

**Conference**: CVPR 2025  
**arXiv**: [2603.09760](https://arxiv.org/abs/2603.09760)  
**Code**: [https://github.com/GL-ZHU925/PanoAffordanceNet](https://github.com/GL-ZHU925/PanoAffordanceNet)  
**Area**: Robotics / 3D Understanding  
**Keywords**: Panoramic affordance, 360-degree scene, ERP distortion, spherical propagation, one-shot learning

## TL;DR
This paper proposes PanoAffordanceNet, the first 360° panoramic affordance grounding framework. It handles ERP latitude-dependent distortion using a Distortion-Aware Spectrum Modulator (DASM), restores sparse activations into topologically continuous areas via an Omnispherical Densification Head (OSDH), suppresses semantic drift with multi-level training objectives, and constructs the first panoramic affordance dataset 360-AGD, comprehensively outperforming existing methods.

## Background & Motivation
**Background**: Visual affordance grounding (predicting interactive regions of objects given an action) is a core capability of Embodied AI, which has evolved from fully supervised to weakly supervised and foundation-model-driven paradigms. However, existing methods are limited to object-centric perspective-view settings.

**Limitations of Prior Work**: Robots naturally operate in 360° spaces, whereas existing affordance methods are constrained by limited fields of view. When directly extended to panoramic scenes, their performance drops sharply. The root cause lies in three panorama-specific challenges: (1) ERP (Equirectangular Projection) introduces latitude-dependent geometric distortions, with severe stretching in polar regions, making it difficult for models to simultaneously preserve local interaction details and global functional structures; (2) Non-uniform sampling leads to highly sparse and scattered functional regions, making it difficult to aggregate initial activations into semantically coherent affordance regions; (3) The lack of dense pixel-level annotations makes it extremely challenging to precisely align abstract affordance semantics with multi-scale regions in complex 360° scenes.

**Key Challenge**: Panoramic images provide complete environmental information, but the equirectangular projection fundamentally alters the distribution patterns of spatial features, causing methods designed for perspective views to fail.

**Goal**: How to achieve accurate scene-level affordance grounding in 360° panoramic indoor environments?

**Key Insight**: Handle ERP distortion in the frequency domain rather than the spatial domain (since frequencies naturally correspond to structures at different scales), while leveraging spherical self-similarity to propagate seed activations to restore topologically continuous regions.

**Core Idea**: Bi-spectral distillation for latitude-adaptive distortion calibration + spherical seed propagation for sparse activation densification = panoramic affordance grounding.

## Method

### Overall Architecture
An end-to-end one-shot learning framework. The input consists of a panoramic ERP image (560×1120) and a one-shot example. DINOv2-B/14 (fine-tuned with LoRA) extracts visual features, while a CLIP text encoder with CoOp provides affordance semantic embeddings. Features are calibrated by DASM and then fed into a spherical-aware hierarchical decoder (global semantic discovery + OSDH densification), jointly trained with multi-level losses.

### Key Designs

1. **Distortion-Aware Spectrum Modulator (DASM)**:

    - **Function**: Isolate task-related geometric signals via bi-spectral distillation to adaptively calibrate latitude-dependent distortion caused by ERP.
    - **Mechanism**: First, inject textual semantics into visual features via cross-attention. Then, extract high frequencies (boundaries/contours) using a Laplacian operator and low frequencies (global structure) using Gaussian smoothing. An equator-region High-Frequency Enhancement Module (HFEM) sharpens interaction boundaries, while a polar-region Low-Frequency Stabilization Module (LFSM) maintains global structural consistency. Finally, the two frequency bands are selectively fused via hybrid gating (language-driven channel gates + adaptive spatial gates): $\mathbf{F}_{\text{freq}} = \mathbf{F}'_v + \sum_{k \in \{h,l\}} \lambda_k (\mathbf{g}_{\text{ch}} \odot \mathbf{g}_{\text{sp}} \odot \mathbf{F}_k)$.
    - **Design Motivation**: ERP causes clear edges in the equator region but stretched structures in polar regions, which mathematically corresponds to an imbalance between high and low frequencies in the frequency domain. Handling these frequencies separately is more natural and elegant than spatial-domain correction.

2. **Omnispherical Densification Head (OSDH)**:

    - **Function**: Restore sparse and fragmented initial affordance activations into topologically continuous and complete functional areas.
    - **Mechanism**: First, obtain the initial affordance map $\mathbf{A}_{\text{init}}$ via cross-attention. Then, project visual features onto a unit hypersphere to construct a cosine similarity affinity matrix $\mathcal{S} \in \mathbb{R}^{L \times L}$. High-confidence seeds are selected via top-k, noise is suppressed using a confidence map, and finally, seed-driven propagation is performed: $\mathbf{A}_{\text{refined}} = \mathbf{A}_{\text{init}} + \alpha \cdot \max_{j \in \mathcal{K}} (\mathcal{S}_{ij} \cdot \mathcal{C}_j)$.
    - **Design Motivation**: The non-uniform sampling of panoramas leads to sparse and scattered functional regions. Utilizing spherical self-similarity (the assumption that regions with similar appearances tend to have the same affordance) as a structural inductive bias, seed propagation is utilized to restore continuous regions.

3. **Multi-level Training Objectives**:

    - **Function**: Suppress semantic drift from three levels.
    - **Mechanism**: Pixel-level BCE ensures localization accuracy; distribution-level KL divergence maintains panoramic topological continuity; and Region-Text Contrastive loss (InfoNCE) precisely aligns visual regions with affordance concepts. $\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{\text{BCE}} + \lambda_2 \mathcal{L}_{\text{KL}} + \lambda_3 \mathcal{L}_{\text{RTC}}$.
    - **Design Motivation**: Pure pixel-level loss cannot handle semantic ambiguity across multiple affordances on the same object (e.g., "grip" on a sofa's armrest vs. "sit" on its cushion). Thus, region-text contrastive constraints are required.

### Loss & Training
- Total Loss: $\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{\text{BCE}} + \lambda_2 \mathcal{L}_{\text{KL}} + \lambda_3 \mathcal{L}_{\text{RTC}}$, where RTC aligns region features with affordance text embeddings based on InfoNCE.
- AdamW optimizer with a learning rate of 1e-5 and cosine annealing. 2×A6000 GPUs, 20k iterations, batch size 4.
- Panorama-specific data augmentation: random rotation of ±3°, random scaling of ±5%, and horizontal roll translation (leveraging the rotational invariance of 360° topology).
- Annotations are keypoint-based and converted into probability heatmaps using a Gaussian kernel as soft supervision.

## Key Experimental Results

### Main Results

| Dataset | Metric | PanoAffordanceNet | OOAL | OS-AGDO |
|--------|------|------------------|------|---------|
| 360-AGD Easy | KLD↓ | **1.270** | 2.868 | 2.853 |
| 360-AGD Easy | SIM↑ | **0.506** | 0.117 | 0.124 |
| 360-AGD Easy | NSS↑ | **4.490** | 1.267 | 1.299 |
| 360-AGD Hard | KLD↓ | **1.306** | 3.067 | 2.965 |
| 360-AGD Hard | SIM↑ | **0.474** | 0.097 | 0.115 |
| 360-AGD Hard | NSS↑ | **4.398** | 1.484 | 1.484 |
| AGD20K Seen | KLD↓ | **0.739** | 0.740 | — |
| AGD20K Seen | SIM↑ | **0.616** | 0.577 | 0.390 |
| AGD20K Unseen | SIM↑ | **0.475** | 0.461 | 0.382 |

On 360-AGD, the SIM metric increases by more than 4 times, and the NSS increases by more than 3 times. It also remains competitive on the standard perspective AGD20K dataset.

### Ablation Study (360-AGD Hard Split)

| Configuration | KLD↓ | SIM↑ | NSS↑ | Description |
|------|------|------|------|------|
| Baseline | 1.475 | 0.416 | 4.196 | Without LoRA/DASM/OSDH |
| + LoRA | 1.421 | 0.429 | 4.257 | Parameter-efficient fine-tuning |
| + LoRA + DASM | 1.380 | 0.450 | 4.317 | Distortion calibration is effective |
| + LoRA + OSDH | 1.359 | 0.448 | 4.339 | Densification is effective |
| + LoRA + DASM + OSDH | **1.306** | **0.474** | **4.398** | Complete model is optimal |

### Key Findings
- DASM and OSDH both contribute significant independent improvements, and their combination yields further gains.
- Performance does not degrade on the AGD20K perspective dataset, demonstrating that panoramic processing capability is acquired as an addition without sacrificing perspective performance.
- Existing perspective methods (OOAL, OS-AGDO) show a sharp performance drop when applied directly to panoramas (SIM drops from 0.577 to 0.117), validating the unique challenges of panoramic affordance.

## Highlights & Insights
- **First 360° affordance grounding work**: Defines a new task, builds a new dataset, and proposes a dedicated framework, representing a comprehensive pioneering work.
- **Frequency-domain processing of ERP distortion**: High and low frequencies correspond to boundary details and global structures respectively. Processing them in separate frequency bands is more natural than spatial-domain filtering.
- **Spherical seed propagation**: Leverages visual self-similarity as an inductive bias, restoring continuous regions from sparse activations without requiring extra annotations.
- **One-shot setting**: Generalizes to new affordance classes with only a single example, achieving extremely high annotation efficiency.

## Limitations & Future Work
- The scale and scene diversity of the 360-AGD dataset are limited, which may not fully cover real-world robotic application scenarios.
- The computation of OSDH's affinity matrix $\mathcal{S} \in \mathbb{R}^{L \times L}$ is expensive under high resolutions.
- Not integrated with 3D scene understanding methods; depth information could be utilized for further improvements.
- 19 affordance categories may not suffice to cover the actual interaction needs of robots.

## Related Work & Insights
- **vs OOAL/OS-AGDO**: Perspective methods perform poorly when applied directly to panoramas (SIM 0.117 vs. 0.506), verifying the necessity of ERP distortion handling.
- **vs Panoramic Semantic Segmentation**: Panoramic segmentation handles semantic classes, whereas affordance grounding addresses action-region relationships—closer to the actual interaction needs of robots.
- **vs WorldAfford**: Uses LLMs for scene-level affordance but still depends on SAM for object segmentation, whereas PanoAffordanceNet is end-to-end and more concise.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First 360° affordance grounding work, featuring comprehensive innovations in task definition, dataset, and methodology.
- Experimental Thoroughness: ⭐⭐⭐⭐ Includes ablation studies and cross-domain validation, but the new dataset's scale is limited.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition and detailed method description.
- Value: ⭐⭐⭐⭐ Lays the foundation for embodied AI applications in panoramic scenes.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Selective Contrastive Learning for Weakly Supervised Affordance Grounding](../../ICCV2025/robotics/selective_contrastive_learning_for_weakly_supervised_affordance_grounding.md)
- [\[CVPR 2025\] Coordinated Manipulation of Hybrid Deformable-Rigid Objects in Constrained Environments](coordinated_manipulation_hybrid_deformable_rigid_objects.md)
- [\[ICLR 2026\] RoboInter: A Holistic Intermediate Representation Suite Towards Robotic Manipulation](../../ICLR2026/robotics/robointer_a_holistic_intermediate_representation_suite_towards_robotic_manipulat.md)
- [\[ICML 2025\] BiAssemble: Learning Collaborative Affordance for Bimanual Geometric Assembly](../../ICML2025/robotics/biassemble_learning_collaborative_affordance_for_bimanual_geometric_assembly.md)
- [\[ICCV 2025\] NavMorph: A Self-Evolving World Model for Vision-and-Language Navigation in Continuous Environments](../../ICCV2025/robotics/navmorph_a_self-evolving_world_model_for_vision-and-language_navigation_in_conti.md)

</div>

<!-- RELATED:END -->

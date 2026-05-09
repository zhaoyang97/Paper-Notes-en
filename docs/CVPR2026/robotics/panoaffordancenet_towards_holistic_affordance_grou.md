---
title: >-
  [Paper Note] PanoAffordanceNet: Towards Holistic Affordance Grounding in 360° Indoor Environments
description: >-
  [CVPR 2026][Robotics][Panoramic affordance grounding] PanoAffordanceNet introduces a novel task of holistic affordance grounding in 360° indoor environments. It employs a Distortion-Aware Spectrum Modulator (DASM) to correct ERP geometric distortions, an Omnidirectional Sphere Densification Head (OSDH) to recover continuous affordance regions from sparse activations, and multi-level training objectives. The method achieves substantial gains over existing approaches on 360-AGD, the first panoramic affordance dataset constructed by the authors.
tags:
  - CVPR 2026
  - Robotics
  - Panoramic affordance grounding
  - 360° indoor perception
  - distortion-aware modulation
  - omnidirectional densification
  - one-shot learning
date: 2026-05-08
content_hash: 3ebf9247487618f0
---

# PanoAffordanceNet: Towards Holistic Affordance Grounding in 360° Indoor Environments

**Conference**: CVPR 2026
**arXiv**: [2603.09760](https://arxiv.org/abs/2603.09760)
**Code**: [https://github.com/GL-ZHU925/PanoAffordanceNet](https://github.com/GL-ZHU925/PanoAffordanceNet)
**Area**: Robotics / Affordance Perception
**Keywords**: Panoramic affordance grounding, 360° indoor perception, distortion-aware modulation, omnidirectional densification, one-shot learning

## TL;DR

PanoAffordanceNet introduces a novel task of holistic affordance grounding in 360° indoor environments. It employs a Distortion-Aware Spectrum Modulator (DASM) to correct ERP geometric distortions, an Omnidirectional Sphere Densification Head (OSDH) to recover continuous affordance regions from sparse activations, and multi-level training objectives. The method achieves substantial gains over existing approaches on 360-AGD, the first panoramic affordance dataset constructed by the authors.

## Background & Motivation

**Background**: Visual affordance research aims to localize interactive regions on objects, serving as a bridge between visual perception and physical manipulation. Existing methods have evolved from fully supervised to weakly supervised approaches (LOCATE/WSMA), and further to foundation-model-driven open-vocabulary methods (OOAL/AffordanceLLM). However, nearly all are validated on object-centric paradigms and limited-field-of-view images.

**Limitations of Prior Work**: (1) Service robots operate in 360° physical spaces, yet existing methods process only perspective images with restricted fields of view (FOV), creating a mismatch with the 360° action space; (2) Directly applying perspective methods to panoramic images causes severe performance degradation — equirectangular projection (ERP) introduces significant geometric distortion (polar stretching), non-uniform sampling results in sparse and scattered affordance region distributions, and precise alignment of abstract affordance semantics with multi-scale regions is extremely difficult.

**Key Challenge**: Panoramic images are not merely an extension of the field of view — they fundamentally alter the spatial distribution of features. The triple challenges of latitude-dependent ERP distortion, fragmented affordance region distribution, and semantic drift under weak supervision are deeply intertwined and entirely beyond the reach of existing methods.

**Goal**: (1) How to preserve local interaction details and global affordance structure under ERP distortion; (2) How to recover continuous and complete affordance regions from sparse, fragmented initial activations; (3) How to precisely align semantics and visual regions under extremely sparse (one-shot) annotations.

**Key Insight**: The problem is decomposed into three independent channels: spectrum-domain processing for distortion (high-frequency and low-frequency correction separately), spherical topology-domain processing for fragmentation (self-similarity propagation), and contrastive learning-domain processing for semantic drift (region–text alignment).

**Core Idea**: A three-stage design combining spectral distortion correction, spherical densification, and multi-level constraints enables one-shot holistic affordance grounding in 360° indoor environments.

## Method

### Overall Architecture

The end-to-end pipeline consists of four modules: (1) dual-encoder feature extraction — DINOv2 visual encoder (with LoRA adaptation) and CLIP text encoder (with CoOp learnable prompts); (2) DASM Distortion-Aware Spectrum Modulator — dual-band decomposition with latitude-adaptive correction; (3) sphere-aware hierarchical decoder — global semantic discovery with OSDH densification; (4) multi-level training objectives — pixel-level, distribution-level, and region–text contrastive losses. Input: 560×1120 panoramic images with one-shot annotations.

### Key Designs

1. **Distortion-Aware Spectrum Modulator (DASM)**:

    - Function: Corrects latitude-dependent geometric distortion and semantic diffusion introduced by ERP projection.
    - Mechanism: Cross-modal attention is first applied to inject text guidance into visual features $\mathbf{F}'_v$, activating semantically relevant regions. Features are then decomposed into two branches: high-frequency (Laplacian operator $\nabla^2$) and low-frequency (Gaussian smoothing $\mathcal{K}_\sigma$). The High-Frequency Enhancement Module (HFEM) sharpens interaction boundaries in equatorial regions and suppresses polar-magnification artifacts; the Low-Frequency Stabilization Module (LFSM) maintains global structural consistency in polar regions and alleviates semantic fragmentation caused by stretching. The two branches are fused via a hybrid gating mechanism using a language-driven channel gate $\mathbf{g}_{ch}$ and an adaptive spatial gate $\mathbf{g}_{sp}$: $\mathbf{F}_{\text{freq}} = \mathbf{F}'_v + \sum_{k} \lambda_k (\mathbf{g}_{ch} \odot \mathbf{g}_{sp} \odot \mathbf{F}_k)$
    - Design Motivation: ERP preserves sharp edges at the equator but stretches structures at the poles — high-frequency and low-frequency components require correction strategies in opposite directions, motivating independent dual-band processing followed by gated fusion.

2. **Omnidirectional Sphere Densification Head (OSDH)**:

    - Function: Recovers topologically continuous and complete affordance regions from sparse, fragmented initial activations.
    - Mechanism: Visual self-similarity is leveraged as a structural inductive bias. Visual features are projected onto a unit hypersphere to construct a cosine similarity affinity matrix $\mathcal{S}_{ij}$. High-confidence seed points are selected via top-k ranking, and a Sigmoid confidence map $\mathcal{C}$ based on mean and standard deviation is applied to suppress noise on the seeds. Seed activations are then propagated via max-diffusion: $\mathbf{A}_{\text{refined}} = \mathbf{A}_{\text{init}} + \alpha \cdot \max_{j \in \mathcal{K}}(\mathcal{S}_{ij} \cdot \mathcal{C}_j)$
    - Design Motivation: Affordance regions in panoramic images appear fragmented due to non-uniform sampling, yet visual features within the same affordance region exhibit high self-similarity. Seed propagation exploits this inductive bias to recover dense predictions from sparse activations.

3. **Region–Text Contrastive Loss ($\mathcal{L}_{RTC}$)**:

    - Function: Establishes precise correspondence between visual regions and affordance semantic concepts, suppressing semantic drift.
    - Mechanism: Ground-truth masks are used to pool visual features into region-level representations $\mathbf{v}_c = \sum_l \hat{M}_{c,l} \mathbf{f}''_{v,l} / \sum_k \hat{M}_{c,k}$, which are then contrastively aligned with corresponding text embeddings via InfoNCE. The loss is jointly optimized with pixel-level BCE and distribution-level KL divergence losses: $\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{BCE} + \lambda_2 \mathcal{L}_{KL} + \lambda_3 \mathcal{L}_{RTC}$
    - Design Motivation: A single object may support multiple affordances (e.g., "sit" vs. "lean" on a sofa), and pixel-level supervision alone cannot distinguish them. Region–text contrastive learning precisely anchors language supervision to specific visual regions.

### Loss & Training

AdamW optimizer with cosine annealing, learning rate 1e-5, trained for 20k iterations on 2× A6000 GPUs with batch size 4. DINOv2 is adapted with LoRA (rank=16); the CLIP text encoder is frozen but augmented with CoOp learnable prompts. Panorama-specific data augmentation includes random rotation ±3°, scaling ±5%, and horizontal wraparound shifts.

## Key Experimental Results

### Main Results

One-shot affordance grounding on the 360-AGD dataset:

| Method | Easy KLD↓ | Easy SIM↑ | Easy NSS↑ | Hard KLD↓ | Hard SIM↑ | Hard NSS↑ |
|------|----------|----------|----------|----------|----------|----------|
| OOAL | 2.868 | 0.117 | 1.267 | 3.067 | 0.097 | 1.484 |
| OS-AGDO | 2.853 | 0.124 | 1.299 | 2.965 | 0.115 | 1.484 |
| **PanoAffordanceNet** | **1.270** | **0.506** | **4.490** | **1.306** | **0.474** | **4.398** |

Generalization on the perspective AGD20K dataset:

| Method | Seen KLD↓ | Seen SIM↑ | Unseen KLD↓ | Unseen SIM↑ |
|------|----------|----------|------------|------------|
| OOAL | 0.740 | 0.577 | 1.070 | 0.461 |
| **Ours** | **0.739** | **0.616** | 1.185 | **0.475** |

### Ablation Study

Component ablation (Hard Split):

| LoRA | DASM | OSDH | KLD↓ | SIM↑ | NSS↑ |
|------|------|------|------|------|------|
| | | | 1.475 | 0.416 | 4.196 |
| ✓ | | | 1.421 | 0.429 | 4.257 |
| ✓ | ✓ | | 1.380 | 0.450 | 4.317 |
| ✓ | | ✓ | 1.359 | 0.448 | 4.339 |
| ✓ | ✓ | ✓ | **1.306** | **0.474** | **4.398** |

Loss function ablation:

| $\mathcal{L}_{KL}$ | $\mathcal{L}_{RTC}$ | $\mathcal{L}_{BCE}$ | KLD↓ | SIM↑ | NSS↑ |
|-----|------|------|------|------|------|
| | | ✓ | 1.596 | 0.395 | 3.891 |
| ✓ | | ✓ | 1.430 | 0.450 | 4.041 |
| ✓ | ✓ | ✓ | **1.306** | **0.474** | **4.398** |

### Key Findings

- PanoAffordanceNet reduces KLD by over 55%, improves SIM by more than 4×, and NSS by more than 3× on 360-AGD, demonstrating overwhelming superiority.
- The three modules contribute complementarily: DASM primarily reduces KLD (geometric correction), OSDH primarily improves SIM/NSS (region continuity), and LoRA provides foundational adaptation.
- $\mathcal{L}_{RTC}$ contributes most to semantically sensitive metrics (SIM/NSS), validating the critical role of region–text alignment in distinguishing multiple affordances.
- KLD varies by only 0.006 across top-k values in the range 5–20, demonstrating that OSDH is highly robust to this hyperparameter.
- LoRA rank=16 is optimal; higher ranks (e.g., 32) cause overfitting and degrade DINOv2's pre-trained semantics.
- Competitive performance on the perspective AGD20K dataset confirms that the method does not rely on panorama-specific assumptions.

## Highlights & Insights

- **Forward-looking new task definition**: This work is the first to advance affordance research from an object-centric paradigm to a 360° scene-level setting, directly addressing the practical needs of service robots. The 360-AGD dataset fills a critical gap in panoramic affordance research.
- **Elegant symmetric design of the dual-frequency channels**: The equatorial region requires high-frequency enhancement (sharpening boundaries), while polar regions require low-frequency stabilization (preventing fragmentation) — the two correction directions are precisely opposite. Independent dual-band processing followed by gated fusion is a natural and principled solution.
- **OSDH recovers topological structure via visual self-similarity**: No additional geometric information (e.g., depth maps) is required; cosine similarity of features alone drives sparse-to-dense propagation. The design is conceptually clean, hyperparameter-insensitive, and transferable to any setting requiring dense prediction recovery from sparse annotations.

## Limitations & Future Work

- The 360-AGD dataset is relatively small in scale (total sample count not disclosed), and the complexity gap between Easy and Hard splits warrants further validation.
- Only 19 affordance categories are evaluated; real-world indoor affordances are far more diverse and complex.
- The one-shot setting limits coverage of long-tail affordances; few-shot or zero-shot extensions are a natural next step.
- The method processes static images and does not account for temporal affordance changes in dynamic scenes.
- ERP remains an intermediate representation; operating directly on the sphere (e.g., via spherical convolutions) may offer a more principled approach.

## Related Work & Insights

- **vs. OOAL**: OOAL is the current one-shot affordance state of the art but is designed entirely for perspective images; its SIM on panoramic data is only 0.117 versus 0.506 achieved by the proposed method.
- **vs. WorldAfford**: WorldAfford also targets scene-level affordance but relies on SAM object segmentation and LLM reasoning, making it non-end-to-end; the proposed method is end-to-end and requires no segmentation.
- **vs. 3D affordance methods**: 3D methods provide precise geometric constraints but require costly annotations and lack mature foundation models; panoramic images serve as a middle ground between 2D and 3D, offering 360° spatial coverage while retaining the generalization capability of 2D foundation models.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — New task definition, new dataset, and purpose-built method design; a pioneering contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Thorough ablations; only two baselines (inherent to the new task), but cross-domain generalization validation is convincing.
- Writing Quality: ⭐⭐⭐⭐ — Problem motivation is clear and method description is detailed, though notation is dense.
- Value: ⭐⭐⭐⭐⭐ — Opens a new research direction in panoramic affordance with direct application value for global perception in service robotics.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models](quantvla_scale-calibrated_post-training_quantization_for_vision-language-action_.md)
- [\[CVPR 2026\] Language-Grounded Decoupled Action Representation for Robotic Manipulation](language-grounded_decoupled_action_representation_for_robotic_manipulation.md)
- [\[CVPR 2026\] GeCo-SRT: Geometry-aware Continual Adaptation for Robotic Cross-Task Sim-to-Real Transfer](geco-srt_geometry-aware_continual_adaptation_for_robotic_cross-task_sim-to-real_.md)
- [\[CVPR 2026\] BiPreManip: Learning Affordance-Based Bimanual Preparatory Manipulation through Anticipatory Collaboration](bipremanip_learning_affordance-based_bimanual_preparatory_manipulation_through_a.md)
- [\[CVPR 2026\] Towards Open Environments and Instructions: General Vision-Language Navigation via Fast-Slow Interactive Reasoning](towards_open_environments_and_instructions_general_vision-language_navigation_vi.md)

</div>

<!-- RELATED:END -->

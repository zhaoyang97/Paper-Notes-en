---
title: >-
  [Paper Note] VGGDrive: Empowering Vision-Language Models with Cross-View Geometric Grounding for Autonomous Driving
description: >-
  [CVPR 2026][Multimodal VLM][Autonomous Driving] This paper proposes VGGDrive, a framework that injects cross-view geometric perception into VLMs via a frozen 3D visual foundation model (VGGT). A plug-and-play CVGE module is designed to hierarchically and adaptively fuse 3D features into the 2D visual embeddings at each VLM layer, achieving significant performance gains across five autonomous driving benchmarks.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Autonomous Driving
  - 3D Geometric Perception
  - VLM
  - VGGT
  - Cross-View
date: 2026-05-08
content_hash: 7acc3e631e5d3d43
---

# VGGDrive: Empowering Vision-Language Models with Cross-View Geometric Grounding for Autonomous Driving

**Conference**: CVPR 2026
**arXiv**: [2602.20794](https://arxiv.org/abs/2602.20794)
**Code**: [https://github.com/WJ-CV/VGGDrive](https://github.com/WJ-CV/VGGDrive)
**Area**: Multimodal VLM
**Keywords**: Autonomous Driving, 3D Geometric Perception, VLM, VGGT, Cross-View

## TL;DR
This paper proposes VGGDrive, a framework that injects cross-view geometric perception into VLMs via a frozen 3D visual foundation model (VGGT). A plug-and-play CVGE module is designed to hierarchically and adaptively fuse 3D features into the 2D visual embeddings at each VLM layer, achieving significant performance gains across five autonomous driving benchmarks.

## Background & Motivation
**Background**: VLMs, endowed with rich world knowledge and reasoning capabilities, have provided powerful scene understanding and decision-making support for autonomous driving systems, with Vision-Language-Action (VLA) models emerging as a prominent research direction.

**Limitations of Prior Work**: VLMs inherently lack cross-view geometric modeling capabilities for the 3D physical world, which directly limits their performance on autonomous driving tasks requiring fine-grained spatial perception (e.g., Qwen2.5-VL performs poorly on driving benchmarks).

**Key Challenge**: Some methods attempt to teach VLMs spatial concepts via QA data construction, yet fail to endow models with geometric priors at a fundamental level. Others attach independent action decoders on top of VLMs for trajectory prediction, thereby decoupling scene understanding from decision-making.

**Goal**: To effectively inject the cross-view geometric modeling capabilities of a mature 3D foundation model (VGGT) into VLMs, addressing their inherent spatial perception deficiencies.

**Key Insight**: Rather than teaching VLMs to understand space, directly and deeply fusing VGGT's 3D geometric features into the VLM's 2D visual representations through hierarchical injection—rather than simple concatenation or addition—achieves genuine geometric grounding.

**Core Idea**: A hierarchical adaptive injection mechanism fuses 3D features from a frozen VGGT layer-by-layer into the VLM's 2D visual embeddings, establishing true geometric grounding.

## Method

### Overall Architecture
VGGDrive consists of three core components: (1) a base VLM (Qwen2.5-VL-7B) that processes multi-view images and text instructions; (2) a hierarchical adaptive injection mechanism that decouples the LLM structure and progressively extracts and injects 3D visual embeddings into each layer; and (3) a Cross-View Geometric Empowerer (CVGE) that deeply fuses VGGT's 3D features with the VLM's 2D visual representations. The input comprises multi-view surround-view images (6 cameras for nuScenes, 3 front-view cameras for NAVSIM), and the output is either textual reasoning or trajectory prediction.

### Key Designs

1. **Hierarchical Adaptive Injection**:

    - Function: Hierarchically injects 3D geometric information from the frozen VGGT into the 2D visual embeddings at each decoder layer of the VLM.
    - Mechanism: VGGT first extracts 3D features $V^{3d}$ from multi-view inputs (retaining camera embeddings and register embeddings). The LLM decoder layers are then decoupled; at each layer, an image-ID positional mask $M_{id}^{img}$ is used to extract 2D visual embeddings $V_i^{2d}$, which are fed into the CVGE to produce enhanced 3D embeddings $V_i^{3d}$. The original visual embeddings are then replaced via residual connection: $x_i = X_i + X_i'$.
    - Design Motivation: Since different layers encode representations at different semantic levels and exhibit varying sensitivity to 3D information, the CVGE adopts a modular design with consistent architecture but independent parameters per layer, enabling each layer to adaptively learn its most relevant geometric information.

2. **Cross-View Geometric Empowerer (CVGE)**:

    - Function: Establishes a learnable cross-modal interaction between 2D visual embeddings and 3D geometric features.
    - Mechanism: The 2D visual embeddings query the 3D representations to actively mine and integrate critical geometric information. Specifically, cross-attention mechanisms enable $V_i^{2d}$ to extract cross-view geometric information from $V^{3d}$.
    - Design Motivation: Simple feature concatenation or addition (as in VGGT-Dist and VGGT-Add) does not allow the VLM to fully exploit 3D geometric features; a deep interaction mechanism is required to establish genuine geometric grounding.

3. **Plug-and-Play Design**:

    - The VGGT model remains frozen throughout training; the CVGE is inserted into the VLM as a plug-and-play module.
    - Only the CVGE parameters are trained, while the pre-trained weights of both the VLM and VGGT remain unchanged.

### Loss & Training
Standard cross-entropy loss is employed. For trajectory planning tasks, ego state and navigation commands are additionally provided as text inputs.

## Key Experimental Results

### Main Results — NAVSIM Trajectory Planning

| Method | Base Model | PDMS↑ | NC↑ | DAC↑ | EP↑ |
|------|-----------|-------|-----|------|-----|
| Baseline (Qwen2.5-VL) | 7B | 86.04 | 97.83 | 94.08 | 81.00 |
| VGGT-Dist | 7B | 86.68 | 97.84 | 94.81 | 81.30 |
| VGGT-Add | 7B | 86.10 | 97.81 | 94.07 | 80.84 |
| **VGGDrive** | **7B** | **88.76** | **98.55** | **96.30** | **82.92** |
| DiffusionDrive (E2E SOTA) | - | 88.10 | 98.20 | 96.20 | 82.20 |

### Main Results — NuInstruct Cross-View Risk Perception

| Method | MAE↓ | Accuracy↑ | MAP↑ | BLEU↑ |
|------|------|-----------|------|-------|
| Baseline | 4.35 | 47.71 | 6.15 | 75.75 |
| VGGT-Dist | 3.73 | 56.21 | 28.51 | 79.23 |
| **VGGDrive** | **3.08** | **56.37** | **37.49** | **81.13** |

### Ablation Study — DriveLM

| Method | Accuracy↑ | Match↑ | Average↑ |
|------|-----------|--------|----------|
| Baseline | 64.35 | 34.54 | 54.59 |
| VGGDrive | **77.50** | **49.77** | **61.26** |

### Key Findings
- VGGDrive comprehensively outperforms the baseline and naive integration schemes across all five benchmarks, achieving a PDMS of 88.76 on NAVSIM and surpassing the majority of end-to-end methods that use LiDAR.
- The MAP metric for cross-view risk perception yields the largest improvement (6.15→37.49), indicating that 3D geometric features provide the greatest benefit for spatial perception tasks.
- Simple VGGT-Dist and VGGT-Add integration schemes offer only marginal gains—VGGT-Add even degrades performance on certain metrics—validating the necessity of deep feature fusion.

## Highlights & Insights
- **A New Paradigm for Empowering VLMs with 3D Foundation Models**: Unlike approaches that teach VLMs spatial concepts through data or append independent decoding heads, this work pioneers the use of a frozen 3D foundation model to directly empower VLMs—a novel and highly generalizable strategy.
- **Hierarchical Adaptive Injection Design**: Recognizing that different VLM layers have distinct requirements for 3D information, per-layer CVGE modules with independent parameters adaptively extract relevant geometric information, outperforming globally uniform injection.
- **Dramatic Improvement in Cross-View MAP**: The improvement from 6.15 to 37.49 demonstrates that 3D geometric grounding fundamentally transforms the spatial perception capability of VLMs.

## Limitations & Future Work
- Validation is currently limited to Qwen2.5-VL-7B; generalizability to larger-scale VLMs or other VLM families has not been examined.
- VGGT remains frozen throughout; joint fine-tuning or progressive unfreezing strategies have not been explored.
- Trajectory prediction relies directly on VLM text outputs, whose precision is constrained by tokenization resolution; future work could incorporate a dedicated trajectory decoding head.
- The CVGE introduces additional parameters and computational overhead; the latency impact on practical deployment warrants further evaluation.

## Related Work & Insights
- **vs. VGGT-Dist/VGGT-Add**: Simple distillation or additive integration schemes fuse 3D features only at the final or shallow layers; VGGDrive's hierarchical deep injection substantially outperforms these baselines.
- **vs. VLA methods such as CarLLaVA/AdaThinkDrive**: Other VLA methods rely on QA data or independent decoders, whereas VGGDrive fundamentally enhances spatial perception by leveraging a 3D foundation model.

## Rating
- Novelty: ⭐⭐⭐⭐ Empowering VLMs with a 3D foundation model for autonomous driving represents a novel paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across five benchmarks with thorough ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and rich figures/tables.
- Value: ⭐⭐⭐⭐ Points toward an effective direction for 3D+VLM-based driving systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Prune2Drive: A Plug-and-Play Framework for Accelerating Vision-Language Models in Autonomous Driving](prune2drive_a_plug-and-play_framework_for_accelerating_vision-language_models_in.md)
- [\[CVPR 2026\] TreeTeaming: Autonomous Red-Teaming of Vision-Language Models via Hierarchical Strategy Exploration](treeteaming_autonomous_red-teaming_of_vision-language_models_via_hierarchical_s.md)
- [\[CVPR 2026\] Empowering Semantic-Sensitive Underwater Image Enhancement with VLM](empowering_semanticsensitive_underwater_image_enha.md)
- [\[ICCV 2025\] Fine-Grained Evaluation of Large Vision-Language Models in Autonomous Driving](../../ICCV2025/multimodal_vlm/fine-grained_evaluation_of_large_vision-language_models_in_autonomous_driving.md)
- [\[CVPR 2026\] HAMMER: Harnessing MLLM via Cross-Modal Integration for Intention-Driven 3D Affordance Grounding](hammer_harnessing_mllm_via_cross-modal_integration_for_intention-driven_3d_affor.md)

</div>

<!-- RELATED:END -->

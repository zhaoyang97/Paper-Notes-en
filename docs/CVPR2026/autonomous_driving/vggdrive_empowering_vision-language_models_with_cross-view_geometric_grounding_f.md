---
title: >-
  [Paper Note] VGGDrive: Empowering Vision-Language Models with Cross-View Geometric Grounding for Autonomous Driving
description: >-
  [CVPR 2026][Autonomous Driving][3D Geometric Perception] The authors propose the VGGDrive framework, which empowers VLMs with cross-view geometric awareness via a frozen 3D vision foundation model (VGGT). By designing a plug-and-play CVGE module, 3D features are hierarchically and adaptively injected into the 2D visual embeddings of each VLM layer, achieving significant performance gains across five autonomous driving benchmarks.
tags:
  - "CVPR 2026"
  - "Autonomous Driving"
  - "3D Geometric Perception"
  - "VLM"
  - "VGGT"
  - "Cross-View"
date: 2026-05-08
content_hash: 5657a827b690ac75
---

# VGGDrive: Empowering Vision-Language Models with Cross-View Geometric Grounding for Autonomous Driving

**Conference**: CVPR 2026  
**arXiv**: [2602.20794](https://arxiv.org/abs/2602.20794)  
**Code**: [https://github.com/WJ-CV/VGGDrive](https://github.com/WJ-CV/VGGDrive)  
**Area**: Autonomous Driving  
**Keywords**: Autonomous Driving, 3D Geometric Perception, VLM, VGGT, Cross-View  

## TL;DR
The authors propose the VGGDrive framework, which empowers VLMs with cross-view geometric awareness via a frozen 3D vision foundation model (VGGT). By designing a plug-and-play CVGE module, 3D features are hierarchically and adaptively injected into the 2D visual embeddings of each VLM layer, achieving significant performance gains across five autonomous driving benchmarks.

## Background & Motivation
**Background**: VLMs leverage rich world knowledge and reasoning capabilities to provide robust scene understanding and decision support for autonomous driving systems. Vision-Language-Action (VLA) models have consequently become a prominent research focus.

**Limitations of Prior Work**: VLMs naturally lack the cross-view geometric modeling capabilities required for the 3D physical world. This limitation directly restricts their performance in autonomous driving tasks necessitating fine-grained spatial awareness (e.g., Qwen2.5-VL showing mediocre performance on driving tasks).

**Key Challenge**: Some methods attempt to teach VLMs spatial concepts by constructing QA data, which fails to fundamentally provide geometric priors. Other methods add independent action decoders to VLMs for trajectory prediction, as this decouples scene understanding from decision-making.

**Goal**: To effectively inject the cross-view geometric modeling capabilities of a mature 3D foundation model (VGGT) into a VLM to compensate for its inherent deficiencies.

**Key Insight**: Rather than teaching the VLM to understand space, it is more effective to deeply integrate the 3D geometric features of VGGT into the 2D visual representations of the VLM. This is achieved through hierarchical injection rather than simple concatenation or addition.

**Core Idea**: Establish a true geometric foundation by integrating 3D features from a frozen VGGT into the 2D visual embeddings of the VLM layer-by-layer using a hierarchical adaptive injection mechanism.

## Method

### Overall Architecture
VGGDrive addresses a specific gap: while VLMs (using Qwen2.5-VL-7B here) possess extensive world knowledge and linguistic reasoning, they lack innate cross-view 3D geometric modeling, which is essential for driving. The approach leverages a pre-trained 3D vision foundation model, VGGT, to "feed" geometric features into the VLM. The pipeline operates as follows: Multi-view surround images (6 cameras for nuScenes, 3 front-views for NAVSIM) are first processed by VGGT to obtain 3D geometric features $V^{3d}$ containing camera information. Simultaneously, the images are encoded into 2D visual tokens by the VLM. Within each VLM decoder layer, a Cross-View 3D Geometric Enabler (CVGE) allows 2D tokens to "query" 3D features, retrieving relevant geometric information to be written back via residual connections. Finally, the VLM outputs textual reasoning or trajectory tokens based on a geometrically grounded visual representation.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}}%%
flowchart TD
    IMG["Multi-view Surround Images<br/>(nuScenes 6-cam / NAVSIM 3-front)"] --> VGGT["Frozen VGGT<br/>Extract Cross-View 3D Geometric Features V³ᵈ"]
    IMG --> VLM["VLM Encoder (Qwen2.5-VL-7B)<br/>Image + Instruction → 2D Visual Tokens"]
    VLM --> INJ
    subgraph INJ["Hierarchical Adaptive Injection (Layer-wise Decoder)"]
        direction TB
        L1["Layer i: Extract 2D Visual Embeddings Vᵢ²ᵈ using Image ID Mask"] --> CVGE["Cross-View 3D Geometric Enabler CVGE<br/>Vᵢ²ᵈ as Query, Cross-Attention retrieves V³ᵈ"]
        CVGE --> RES["Residual Write-back → Enhanced Visual Tokens"]
        RES -->|Independent params per layer, loop to final layer| L1
    end
    VGGT -->|Provide V³ᵈ (VGGT frozen, only CVGE trainable)| CVGE
    INJ --> OUT["Geometrically Aligned Visual Representation<br/>→ Output Risk Description / Trajectory Tokens"]
```

### Key Designs

**1. Hierarchical Adaptive Injection Mechanism: Layer-wise 3D Information Requirement**

Injecting 3D features only at the VLM input results in the geometric information being diluted over dozens of decoder layers. VGGDrive "spreads" the injection across every layer. After extracting $V^{3d}$ (preserving camera and register embeddings) from the frozen VGGT, the mechanism decouples the LLM decoder stack. At layer $i$, the 2D embeddings $V_i^{2d}$ belonging to the visual part are isolated using an image ID position mask $M_{id}^{img}$ and sent to the CVGE to obtain enhanced geometric embeddings $V_i^{3d}$, which are then written back:

$$x_i = X_i + X_i'$$

The crucial aspect is that while the CVGE structure is identical across layers, the parameters are independent. Shallow tokens focus on texture, while deep tokens focus on semantics, leading to different 3D geometric requirements. Allowing each layer to learn its own injection weights enables the model to adaptively decide "what kind of geometric information to supplement and how much" rather than applying a global approach.

**2. Cross-View 3D Geometric Enabler (CVGE): Active Querying by 2D Tokens**

Simply concatenating or adding 3D features to 2D tokens (as in the VGGT-Dist and VGGT-Add baselines) presents a problem: the VLM does not know which spatial locations these external dimensions correspond to, leading to noise or underutilization. CVGE introduces a learnable cross-modal interaction: using 2D visual embeddings $V_i^{2d}$ as queries to retrieve and integrate cross-view geometric cues from $V^{3d}$ via cross-attention. This allows the VLM to actively "extract" the specific 3D information it needs. This deep interaction establishes the "true geometric foundation" that simple addition cannot achieve.

**3. Plug-and-Play Training: Training Only CVGE**

VGGT and VLM are both powerful pre-trained models. Joint fine-tuning is computationally expensive and risks destroying pre-existing capabilities. VGGDrive keeps the VGGT frozen and the VLM backbone static, training only the CVGE modules inserted between layers. This makes the CVGE a pluggable adapter layer that only learns to "translate 3D features into visual increments that the VLM understands," facilitating easy transfer to other VLMs or tasks and avoiding catastrophic forgetting.

### Mechanism: An Example of 6-Camera Input Injection

Using a single frame of six surround images from nuScenes:

1. The 6 images are processed by the frozen VGGT to output cross-view 3D geometric features $V^{3d}$, which encode view correspondences (including camera embeddings).
2. The same images enter the VLM and are encoded into a token sequence where visual tokens are interleaved.
3. In decoder layer 1: Visual tokens $V_1^{2d}$ are extracted via $M_{id}^{img}$ and sent to the first CVGE module to query $V^{3d}$, retrieving the required geometric increment for that layer.
4. This repeats for layer 2, layer 3, etc., using independent CVGE parameters; shallow layers might supplement view correspondences, while deeper layers add decision-related spatial semantics.
5. After passing through all layers, the VLM possesses a geometrically aligned visual representation to output risk descriptions or trajectory tokens.

### Loss & Training
Only the parameters of the CVGE are optimized using standard cross-entropy loss. For trajectory planning tasks, ego-state and navigation commands are provided as textual inputs.

## Key Experimental Results

### Main Results — NAVSIM Trajectory Planning

| Method | Base Model | PDMS↑ | NC↑ | DAC↑ | EP↑ |
|:---|:---:|:---:|:---:|:---:|:---:|
| Baseline (Qwen2.5-VL) | 7B | 86.04 | 97.83 | 94.08 | 81.00 |
| VGGT-Dist | 7B | 86.68 | 97.84 | 94.81 | 81.30 |
| VGGT-Add | 7B | 86.10 | 97.81 | 94.07 | 80.84 |
| **Ours (VGGDrive)** | **7B** | **88.76** | **98.55** | **96.30** | **82.92** |
| DiffusionDrive (E2E SOTA) | - | 88.10 | 98.20 | 96.20 | 82.20 |

### Main Results — NuInstruct Cross-View Risk Perception

| Method | MAE↓ | Accuracy↑ | MAP↑ | BLEU↑ |
|:---|:---:|:---:|:---:|:---:|
| Baseline | 4.35 | 47.71 | 6.15 | 75.75 |
| VGGT-Dist | 3.73 | 56.21 | 28.51 | 79.23 |
| **Ours (VGGDrive)** | **3.08** | **56.37** | **37.49** | **81.13** |

### Ablation Study — DriveLM

| Method | Accuracy↑ | Match↑ | Average↑ |
|:---|:---:|:---:|:---:|
| Baseline | 64.35 | 34.54 | 54.59 |
| **Ours (VGGDrive)** | **77.50** | **49.77** | **61.26** |

### Key Findings
- VGGDrive consistently outperforms the baseline and simple integration schemes across five benchmarks. On NAVSIM, the PDMS reached 88.76, surpassing most LiDAR-based end-to-end methods.
- The most significant improvement was seen in the cross-view risk perception MAP (from 6.15 to 37.49), indicating that 3D geometric features provide substantial help for spatial awareness.
- Simple integration schemes like VGGT-Dist and VGGT-Add showed limited improvement (and even degradations in some metrics for VGGT-Add), validating the necessity of deep hierarchical fusion.

## Highlights & Insights
- **New Paradigm for 3D Foundation Model Empowerment**: Unlike methods that teach VLMs spatial concepts via data or add independent heads, this work innovatively utilizes frozen 3D foundation models to directly empower VLMs.
- **Hierarchical Adaptive Injection Design**: Recognizing that different VLM layers have varying requirements for 3D information, the use of independent CVGE modules to adaptively extract information is superior to global injection.
- **Surge in Cross-View MAP**: The leap from 6.15 to 37.49 demonstrates that 3D geometric grounding fundamentally transforms the VLM's spatial perception.

## Limitations & Future Work
- Currently only validated on Qwen2.5-VL-7B; the generalization to larger scale VLMs or other VLM families has not been tested.
- VGGT is kept frozen throughout; joint fine-tuning or progressive unfreezing strategies have not been explored.
- Trajectory prediction relies on VLM text output; precision is limited by tokenization resolution. Future work could integrate specialized trajectory decoding heads.
- CVGE introduces additional parameters and computational overhead; the impact on latency during real-world deployment requires evaluation.

## Related Work & Insights
- **vs VGGT-Dist/VGGT-Add**: Simple distillation or additive integration schemes only fuse 3D features at the final or shallow layers, whereas VGGDrive's hierarchical deep injection provides significant improvements.
- **vs VLA methods like CarLLaVA/AdaThinkDrive**: These methods rely on QA data or independent decoders. VGGDrive fundamentally enhances spatial perception via 3D model empowerment.

## Rating
- Novelty: ⭐⭐⭐⭐ Empowering VLMs with 3D foundation models for driving is a novel paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensively evaluated across five benchmarks with thorough ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and rich visualizations.
- Value: ⭐⭐⭐⭐ Provides an effective direction for 3D+VLM driving systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] DriveVLN: Towards Mapless Vision-and-Language Navigation in Autonomous Driving](drivevln_towards_mapless_vision-and-language_navigation_in_autonomous_driving.md)
- [\[CVPR 2026\] DriveMoE: Mixture-of-Experts for Vision-Language-Action Model in End-to-End Autonomous Driving](drivemoe_mixture-of-experts_for_vision-language-action_model_in_end-to-end_auton.md)
- [\[CVPR 2026\] EventDrive: Event Cameras for Vision-Language Driving Intelligence](eventdrive_event_cameras_for_vision-language_driving_intelligence.md)
- [\[ICLR 2026\] Discrete Diffusion for Reflective Vision-Language-Action Models in Autonomous Driving](../../ICLR2026/autonomous_driving/discrete_diffusion_for_reflective_vision-language-action_models_in_autonomous_dr.md)
- [\[CVPR 2026\] HybridDriveVLA: Vision-Language-Action Model with Visual CoT reasoning and ToT Evaluation for Autonomous Driving](hybriddrivevla_vision-language-action_model_with_visual_cot_reasoning.md)

</div>

<!-- RELATED:END -->

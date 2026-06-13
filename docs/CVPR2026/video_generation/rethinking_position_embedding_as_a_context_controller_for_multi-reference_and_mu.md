---
title: >-
  [Paper Note] Rethinking Position Embedding as a Context Controller for Multi-Reference and Multi-Shot Video Generation
description: >-
  [CVPR 2026][Video Generation][Position Encoding] This paper proposes PoCo (Position Embedding as Context Controller), which introduces an additional SideInfo axis in RoPE to encode reference entity information…
tags:
  - "CVPR 2026"
  - "Video Generation"
  - "Position Encoding"
  - "Context Control"
  - "Multi-Reference Video Generation"
  - "RoPE"
  - "Identity Confusion"
date: 2026-05-08
content_hash: 9f12a00d97bc9734
---

# Rethinking Position Embedding as a Context Controller for Multi-Reference and Multi-Shot Video Generation

**Conference**: CVPR 2026
**arXiv**: [2604.03738](https://arxiv.org/abs/2604.03738)  
**Code**: [https://poco-multiref-multishot.github.io/](https://poco-multiref-multishot.github.io/)  
**Area**: Video Generation / Diffusion Models / Multi-Reference Multi-Shot Generation
**Keywords**: Position Encoding, Context Control, Multi-Reference Video Generation, RoPE, Identity Confusion

## TL;DR

This paper proposes PoCo (Position Embedding as Context Controller), which introduces an additional SideInfo axis in RoPE to encode reference entity information, addressing the "reference confusion" problem in multi-reference multi-shot video generation—where the model fails to correctly associate shots with references when reference images are visually similar. PoCo achieves state-of-the-art cross-shot consistency on the VACE-Wan2.1-14B framework (CrossShot-FaceSim 89.35, CrossShot-DINO 92.66).

## Background & Motivation

1. **Background**: Video generation has advanced rapidly across tasks such as text-to-video and reference-to-video. Multi-reference multi-shot video generation is critical for filmmaking and narrative video production, yet remains underexplored in academia (closed-source systems such as Sora2 demonstrate feasibility but lack transparency).
2. **Limitations of Prior Work**:
    - Existing reference-to-video methods (Phantom, VACE) largely generate each shot independently, leading to inconsistent backgrounds and appearances across shots.
    - The naive approach of directly concatenating multi-reference and multi-shot tokens into attention causes **reference confusion**—when multiple references appear visually similar, semantically similar tokens prevent the model from distinguishing the correct shot–reference associations.
    - Attention visualizations directly confirm this issue: a given shot attends more strongly to an incorrect reference than to the correct one.
3. **Key Challenge**: A fundamental tension exists between maintaining scene-level semantic consistency across shots (requiring global attention interaction) and faithfully preserving distinct reference identities (requiring precise reference association).
4. **Goal**: (1) Resolve confusion among semantically similar references; (2) achieve precise context control without introducing additional computational overhead.
5. **Key Insight**: The paper revisits the attention mechanism and decomposes it into a "semantics-driven learnable component" (Q-K retrieval) and a "manually designed positional encoding component" (context organization)—the latter can serve as an additional means of context control.
6. **Core Idea**: Auxiliary attribute information of tokens (e.g., a @character_i identifier) is encoded as additional rotational dimensions in RoPE, enabling position embeddings to perform precise context routing that goes beyond semantics.

## Method

### Overall Architecture

Built upon the VACE-Wan2.1-14B video generation framework, reference images are encoded via VAE into tokens that are concatenated with generated video tokens and fed into DiT blocks. Each block incorporates self-attention with SideInfo-RoPE and Hierarchical Cross-Attention. Training and inference are conducted on 9s/480p/16fps videos with a two-reference setting.

### Key Designs

1. **SideInfo-RoPE**:

    - **Function**: Augments standard 3D-RoPE with an auxiliary information axis to guide attention allocation and prevent reference confusion.
    - **Mechanism**: The position coordinate is extended from $\mathbf{p} = (t,h,w)$ to $\mathbf{p}^* = (t,h,w,s)$, where $s$ encodes reference entity information. For each visual token $\mathbf{x}$, the side information $\mathbf{s}(\mathbf{x}) \in \{0,1\}^K$ is defined such that $s_i(\mathbf{x})=1$ indicates that the shot containing this token includes reference $i$. The SideInfo distance between two tokens is $\Delta_{m,n}^s = |\mathbf{s}(\mathbf{x}_m) - \mathbf{s}(\mathbf{x}_n)|$. Among the $D$ RoPE dimensions, $D_s = 2K$ dimensions are allocated to the SideInfo axis; each reference $i$ corresponds to a $2\times2$ rotation block $\hat{\mathbf{R}}^{(i)}_{\Delta^s_{m,n}}$ with rotation phase $\phi_i = \frac{2\pi i - \pi}{K}$. When two tokens share the same SideInfo ($\Delta^s = 0$), no rotation occurs and phases remain aligned; otherwise, the phase offset attenuates cross-reference interference.
    - **Design Motivation**: The SideInfo value space is discrete (only $\{0,1\}$), unlike the large range of $(t,h,w)$; rotation phases are therefore discretized into uniform partitions of the $2\pi$ period. SideInfo dimensions are allocated to low-frequency temporal channels (T-low) to avoid interfering with high-frequency motion modeling.

2. **Hierarchical Cross-Attention**:

    - **Function**: Organizes text conditioning in a global–local structure.
    - **Mechanism**: A binary mask $\mathbf{M} \in \{0,1\}^{L_v \times L_t}$ is constructed: reference image tokens attend to all text tokens ($\mathbf{M}[1:L_{ref}, 1:L_t] = 1$), providing global identity and style guidance across shots; video tokens for each shot attend only to the text segment corresponding to that shot ($\mathbf{M}[\mathcal{V}_s, \mathcal{T}_s] = 1$, $\mathbf{M}[\mathcal{V}_s, \mathcal{T}_{s'\neq s}] = 0$), ensuring local conditioning control.
    - **Design Motivation**: Reference tokens require awareness of the global context to provide consistent identity guidance, whereas video tokens need only attend to their own shot description to avoid cross-shot interference.

3. **Data Pipeline**:

    - **Function**: Constructs structured multi-shot training samples from raw long-form videos.
    - **Mechanism**: The video processing stage performs quality filtering (VQA, sharpness, exposure), shot segmentation (AutoShot + PySceneDetect), OCR-based cropping for watermark removal, MLLM-generated captions, and adjacent clip merging. The reference construction stage performs face detection and identity clustering, retains identities with sufficient occurrences, and builds two types of references per identity: original crops and SeedReam-enhanced frontal portraits. Clustered identity labels are propagated to corresponding shots as training-time side information.
    - **Design Motivation**: High-quality multi-reference multi-shot training data is scarce, necessitating an automated pipeline to extract such data from large-scale video corpora. Dual-branch references (original + augmented) improve the robustness of identity conditioning.

### Loss & Training

- Standard diffusion training on the VACE-Wan2.1-14B framework.
- Learning rate: 1e-5.
- 4 channels allocated to the SideInfo axis (corresponding to 2 SideInfo rotation planes in the two-reference setting).
- SideInfo dimensions are reallocated from low-frequency temporal channels (T-low outperforms T-high).

## Key Experimental Results

### Main Results

| Method | Type | CrossShot-FaceSim ↑ | CrossShot-DINO ↑ | FaceSim ↑ | AvgScore ↑ |
|---|---|---|---|---|---|
| Phantom-14B | Single-shot | 86.12 | 73.24 | 72.75 | 80.72 |
| VACE-14B | Single-shot | 69.49 | 67.30 | 67.05 | 75.56 |
| EchoShot | Multi-shot | 87.05 | 79.81 | N/A | 83.82* |
| **PoCo (Ours)** | Multi-shot | **89.35** | **92.66** | 70.12 | **83.46** |

*EchoShot AvgScore is reported w/o Alignment-FaceSim.

### Ablation Study

**SideInfo-RoPE Channel Selection Ablation**

| Configuration | CrossShot-FaceSim ↑ | CrossShot-DINO ↑ | FaceSim ↑ |
|---|---|---|---|
| w/o SideInfo-RoPE | 77.29 | 91.25 | 45.42 |
| w/ SideInfo-RoPE-Tlow | **81.55** | **91.32** | **60.35** |
| w/ SideInfo-RoPE-Thigh | 80.96 | 91.32 | 55.54 |

### Key Findings

- **SideInfo-RoPE yields substantial gains**: FaceSim improves from 45.42 to 60.35 (+14.93) and CrossShot-FaceSim from 77.29 to 81.55 (+4.26), confirming that reference confusion is a severe problem.
- **Low-frequency temporal channels are more suitable for SideInfo**: T-low outperforms T-high (FaceSim 60.35 vs. 55.54); reallocating high-frequency channels—which model rapid temporal variation—introduces motion artifacts.
- **PoCo achieves a large margin on CrossShot-DINO** (92.66 vs. 73.24 for Phantom), demonstrating that joint multi-shot generation substantially outperforms independent generation in background semantic consistency.
- Qualitative comparisons with commercial systems Kling-1.6 and Vidu-Q2 show that PoCo achieves superior cross-shot scene layout, lighting, and fine-grained appearance consistency.
- Compared to the base model VACE-14B, PoCo also improves single-shot FaceSim (67.05→70.12).

## Highlights & Insights

- **Reinterpreting position encoding as a context controller** is an elegant insight: without modifying the attention architecture or adding extra modules, the introduction of a single additional axis in RoPE enables precise reference routing at zero additional computational cost while preserving full context connectivity.
- **The binary design of the SideInfo distance** is concise and effective—whether a reference appears in a shot is a naturally binary attribute that aligns perfectly with RoPE's rotation mechanism.
- **The complete data pipeline design** (VQA filtering → shot segmentation → identity clustering → dual-branch reference construction → SideInfo annotation) constitutes a highly practical engineering contribution, providing a reproducible workflow for constructing multi-reference multi-shot training data.

## Limitations & Future Work

- The current approach primarily addresses cross-shot reference confusion; fine-grained control over multiple similar individuals within the same shot (e.g., precise action binding, close interactions) remains limited.
- Only the two-reference setting is evaluated; whether increasing $K$ and the corresponding number of SideInfo rotation planes degrades performance in other dimensions requires further investigation.
- The evaluation set is small (54 shots / 18 videos / 9 reference pairs), which may limit statistical significance.
- Scenarios with dynamic SideInfo (e.g., characters entering or leaving scenes across shots) are not explored.
- Alignment-FaceSim (70.12) remains slightly below Phantom-14B (72.75), indicating a minor trade-off in single-shot reference fidelity when employing joint multi-shot generation.

## Related Work & Insights

- **vs. EchoShot**: EchoShot introduces TcRoPE/TaRoPE to handle multi-shot temporal and caption alignment but does not directly encode reference identity; PoCo's SideInfo-RoPE addresses identity association more directly (CrossShot-FaceSim +2.30).
- **vs. Phantom**: Phantom injects references via VAE latent and CLIP semantics but generates each shot independently, leading to cross-shot inconsistency (DINO 73.24 vs. 92.66).
- **vs. "Context as Memory"**: That approach uses an external retrieval module to fetch relevant frames; PoCo eliminates the need for external retrieval by controlling context directly within attention through position encoding.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The core idea of restructuring position encoding as a context controller is novel and elegant; the SideInfo-RoPE design is concise and effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Quantitative and qualitative experiments are comprehensive, including comparisons with commercial systems, though the evaluation set is relatively small.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — The problem is precisely defined, and the reasoning from attention decomposition to SideInfo-RoPE derivation is logically clear with intuitive illustrations.
- **Value**: ⭐⭐⭐⭐⭐ — Multi-reference multi-shot generation is a critical bottleneck in video generation; PoCo provides a zero-overhead solution with high generalizability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] StoryTailor: A Zero-Shot Pipeline for Action-Rich Multi-Subject Visual Narratives](storytailora_zero-shot_pipeline_for_action-rich_multi-subject_visual_narratives.md)
- [\[CVPR 2026\] MoVieDrive: Urban Scene Synthesis with Multi-Modal Multi-View Video Diffusion Transformer](moviedrive_urban_scene_synthesis_with_multi-modal_multi-view_video_diffusion_tra.md)
- [\[CVPR 2026\] Let Your Image Move with Your Motion! – Implicit Multi-Object Multi-Motion Transfer](let_your_image_move_with_your_motion_--_implicit_multi-object_multi-motion_trans.md)
- [\[CVPR 2026\] SwitchCraft: Training-Free Multi-Event Video Generation with Attention Controls](switchcraft_training-free_multi-event_video_generation_with_attention_controls.md)
- [\[AAAI 2026\] FilmWeaver: Weaving Consistent Multi-Shot Videos with Cache-Guided Autoregressive Diffusion](../../AAAI2026/video_generation/filmweaver_weaving_consistent_multi-shot_videos_with_cache-guided_autoregressive.md)

</div>

<!-- RELATED:END -->

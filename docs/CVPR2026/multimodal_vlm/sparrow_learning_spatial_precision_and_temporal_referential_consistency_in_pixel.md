---
title: >-
  [Paper Note] SPARROW: Learning Spatial Precision and Temporal Referential Consistency in Pixel-Grounded Video MLLMs
description: >-
  [CVPR 2026][Multimodal VLM][Video pixel-level grounding] This paper proposes the SPARROW framework, which injects temporal consistency supervision via **Target-Specific Features (TSF)**, stabilizes first-frame initialization through **dual-prompt ([BOX]+[SEG]) coarse-to-fine decoding**, and integrates into existing video MLLMs in a plug-and-play manner, achieving consistent improvements across 6 benchmarks on 3 tasks.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Video pixel-level grounding
  - referring video object segmentation
  - temporal consistency
  - dual-prompt decoding
  - multimodal large language models
date: 2026-05-08
content_hash: b9219375d8aad398
---

# SPARROW: Learning Spatial Precision and Temporal Referential Consistency in Pixel-Grounded Video MLLMs

**Conference**: CVPR 2026
**arXiv**: [2603.12382](https://arxiv.org/abs/2603.12382)
**Code**: None
**Area**: Multimodal VLM
**Keywords**: Video pixel-level grounding, referring video object segmentation, temporal consistency, dual-prompt decoding, multimodal large language models

## TL;DR

This paper proposes the SPARROW framework, which injects temporal consistency supervision via **Target-Specific Features (TSF)**, stabilizes first-frame initialization through **dual-prompt ([BOX]+[SEG]) coarse-to-fine decoding**, and integrates into existing video MLLMs in a plug-and-play manner, achieving consistent improvements across 6 benchmarks on 3 tasks.

## Background & Motivation

### 1. State of the Field

Multimodal large language models (MLLMs) have made substantial progress in image-level visual reasoning and pixel-level grounding. Methods such as LISA and PixelLM enable language-conditioned segmentation via [SEG] tokens. However, extending these approaches to the **video domain** introduces additional challenges including motion dynamics, occlusion, and temporal consistency.

### 2. Limitations of Prior Work

Existing video MLLMs (VideoGLaMM, UniPixel, GLUS) primarily rely on **static [SEG] tokens** for per-frame inference, leading to two critical issues:

- **Temporal drift and identity switching**: Text prompts are static while videos are dynamic, forcing the model to infer motion and appearance changes entirely from visual cues, resulting in inconsistent segmentation of the same target across frames.
- **Unreliable first-frame initialization**: The [SEG] token provides only semantic cues without spatial priors, causing potential misalignment in the first-frame mask and accumulating errors across subsequent frames.

### 3. Root Cause

Static semantic tokens cannot encode the temporally varying position and appearance of a target. Once first-frame localization fails, error propagation degrades segmentation quality for all subsequent frames.

### 4. Paper Goals

To simultaneously address **(i) temporal referential consistency** (identity preservation) and **(ii) first-frame spatial precision** (drift reduction) without modifying the underlying model architecture.

### 5. Starting Point

Temporal supervision signals are distilled from target-specific tracking features (injected during training and removable at inference); a dual-prompt co-decoding mechanism is introduced that combines [BOX] geometric priors with [SEG] semantic priors.

### 6. Core Idea

- **TSF**: Offline detection and tracking yield target trajectories; K-means selects a representative subset, which is encoded into TSF tokens and injected during training to teach the model identity persistence.
- **Dual-prompt**: [BOX]-conditioned class-agnostic proposals provide spatial priors → [SEG] refines segmentation with SAM2 on top of these priors, forming a coarse-to-fine pipeline.

## Method

### Overall Architecture

SPARROW operates as follows: a dual-branch visual encoder (spatial $\mathcal{F}_g$ + temporal $\mathcal{F}_h$) → V→L adapter → LoRA-finetuned LLM → L→V adapter → SAM2 pixel decoder. The LLM outputs [BOX] and [SEG] tokens, which are projected back into the visual space to drive bounding box regression and mask decoding, respectively. All newly introduced modules are plug-and-play and do not modify the backbone.

### Key Designs 1: Target-Specific Features (TSF)

**Function**: Provides temporally aligned, target-specific reference cues during training, enabling the model to learn cross-frame identity preservation.

**Mechanism**:
1. Given a text query, GroundingDINO detects the target in a reference frame, and CLDTracker propagates it across frames to obtain candidate box sequences $B'_1 \ldots B'_{K'}$.
2. K-means clustering ($K=4$) is performed in the joint visual-spatial feature space; the closest samples to cluster centroids form a compact subset $B_1 \ldots B_K$.
3. These regions are encoded by $\mathcal{F}_g$ and projected via the V→L adapter into $Z_\text{TSF}$ tokens, which are concatenated to the multimodal input.

**Design Motivation**: Inspired by Artemis, which demonstrates that tracking target-specific features improves temporal consistency. K-means selection ensures each representative sample covers distinct appearances of the same target while reducing redundancy. Crucially, **TSF is not used by default at inference** (no external detector or tracker required), as the model has internalized temporal consistency from training.

**Dataset Construction**: Multiple public datasets are unified—HC-STVG, VID-Sentence, A2D Sentences, LaSOT, MeViS, GOT-10k, and Ref-SAV—resulting in 30,646 video sequences and 45,231 Q&A pairs with temporally consistent trajectories, bounding boxes, and segmentation masks.

### Key Designs 2: Dual-Prompt Grounding

**Function**: Combines [BOX] and [SEG] for coarse-to-fine localization, stabilizing the first frame and mitigating drift.

**[BOX] Branch**:
1. The LLM outputs [BOX] embedding $e_\text{BOX}$, projected via L→V adapter $W_b$.
2. A class-agnostic proposer (Deformable-DETR structure with a single objectness head) is built on frozen SAM2/Hiera features, generating $K=300$ proposals.
3. $e_\text{BOX}$ is fused with each proposal feature via cross-attention → scored by a filtration head → top-$M$ candidates undergo text-conditioned bounding box regression refinement.
4. Final confidence scores fuse language and visual scores; threshold filtering yields $B^*$.

**[SEG] Branch**:
The LLM outputs [SEG] embedding $e_\text{SEG}$, which is combined with the filtered $\hat{b}$ to form mask queries for the SAM2 prompt encoder; each spatial prior yields an instance-level mask. When $|B^*| > 1$, multi-instance output is naturally supported.

**Design Motivation**: Using [SEG] alone leads to ambiguous first-frame localization. [BOX] first provides geometric constraints, upon which [SEG] performs semantic refinement—the two are complementary. Re-issuing [BOX]+[SEG] at arbitrary frames also enables drift correction.

### Loss & Training

**Two-stage training**:

**Stage 1 — TSF Information Injection**: Trains V→L adapters ($W_g$, $W_h$), L→V SEG adapter $W_s$, and LLM LoRA parameters; backbone and pixel decoder are frozen. Loss: $\mathcal{L}_\text{total} = \mathcal{L}_\text{CE} + \mathcal{L}_\text{BCE} + \mathcal{L}_\text{DICE}$.

**Stage 2 — Box Prompt Learning**:
- The class-agnostic proposer (D-DETR head) is first pretrained independently on COCO/Objects365/OpenImages/V3Det with class labels discarded. Loss: $\mathcal{L}_\text{prop} = \mathcal{L}_\text{obj} + \lambda_1 \cdot \mathcal{L}_{\ell_1} + \lambda_2 \cdot \mathcal{L}_\text{GIoU}$.
- The filtration head and L→V BOX adapter $W_b$ are then fine-tuned with all other parameters frozen. Loss: $\mathcal{L}_\text{filter} = \lambda_\text{cls} \cdot \mathcal{L}_\text{BCE} + \lambda_\text{box} \cdot (\mathcal{L}_{\ell_1} + \mathcal{L}_\text{GIoU})$, where $\lambda_\text{cls}=1.0$, $\lambda_\text{box}=2.0$.

## Key Experimental Results

### Main Results

SPARROW is integrated into three video MLLM baselines (UniPixel, GLUS, VideoGLaMM) across three tasks: RVOS, VG, and GCG.

**Table 1: MeViS Referring Video Object Segmentation (Motion Expressions)**

| Method | val J&F | val$^u$ J&F |
|--------|---------|------------|
| UniPixel | 53.1 | 59.7 |
| + SPARROW | **54.4** (+1.3) | **60.7** (+1.0) |
| GLUS | 51.3 | 59.8 |
| + SPARROW | **53.2** (+1.9) | **61.9** (+0.3) |
| VideoGLaMM | 45.2 | 48.5 |
| + SPARROW | **47.5** (+2.3) | **57.4** (+8.9) |

**Table 2: Ref-YTVOS & Ref-DAVIS17 Referring Video Object Segmentation**

| Method | Ref-YTVOS J&F | Ref-DAVIS17 J&F |
|--------|---------------|-----------------|
| UniPixel | 70.5 | 74.2 |
| + SPARROW | **70.7** (+0.2) | **76.4** (+2.2) |
| GLUS | 67.3 | 72.9 |
| + SPARROW | **69.1** (+1.8) | **75.5** (+2.6) |
| VideoGLaMM | 66.8 | 69.5 |
| + SPARROW | **68.9** (+2.1) | **76.8** (+7.3) |

VideoGLaMM achieves a boundary quality gain of up to +14.5 F on Ref-DAVIS17; all SPARROW-integrated models surpass F = 80.

**Table 3: VideoGCG Grounded Conversation Generation**

| Method | mIoU | Recall | CLAIR |
|--------|------|--------|-------|
| UniPixel | 52.0 | 0.311 | 26.0 |
| + SPARROW | **54.5** (+2.5) | **0.325** | **29.4** (+3.4) |
| VideoGLaMM | 62.34 | 0.375 | 28.2 |
| + SPARROW | **65.59** (+3.25) | **0.383** | **33.6** (+5.4) |

### Ablation Study

Based on Ref-DAVIS17 (val) with VideoGLaMM as the baseline.

**Joint Ablation of TSF and BOX (J&F)**:

| TSF Mode | BOX OFF | BOX ON |
|----------|---------|--------|
| No TSF | 69.5 (baseline) | 72.5 (+3.0) |
| Train-only (default) | 72.4 (+2.9) | **76.8** (+7.3) |
| Train + Inference | 75.3 (+5.8) | **77.7** (+8.2) |

**Prompt Combination Ablation**: [SEG] only: 69.5; [BOX] only: 68.2; [BOX]+[SEG]: **72.5** (+3.0), confirming the complementarity of the dual-prompt design.

### Key Findings

1. Using TSF **at training time only** yields a +2.9 gain without incurring any detector/tracker overhead at inference.
2. The [BOX] prompt alone contributes +3.0; combined with TSF, the gains are **approximately additive** (+7.3).
3. VideoGLaMM exhibits the largest improvements (MeViS val$^u$ +8.9, Ref-DAVIS17 +7.3), indicating that weaker baselines benefit more.
4. On the VidSTG visual grounding task, all three baselines consistently gain approximately +5 mIoU.

## Highlights & Insights

- **Plug-and-play design**: SPARROW does not modify the backbone or LLM of any baseline; it integrates solely through lightweight adapters and a proposal head, and successfully improves three architecturally distinct video MLLMs, demonstrating strong generalizability.
- **Tracking at training, tracking-free at inference**: The key insight behind TSF is to inject temporal consistency priors via pseudo-tracking supervision during training; once internalized, the model requires no external tracker at inference, substantially reducing deployment cost.
- **Coarse-to-fine dual-prompt**: [BOX] provides geometric constraints while [SEG] provides semantic refinement—the two are orthogonally complementary in the information dimension, resembling a two-stage detect-then-segment paradigm elegantly realized through tokens.
- **Large-scale dataset construction**: Seven public data sources are unified into a training set of 30K+ videos, filling a gap in target-centric temporal grounding data.

## Limitations & Future Work

1. **Dependence on proposal recall**: Small targets, heavy occlusions, or unseen categories not covered by proposals cannot be recovered; recall is the primary bottleneck.
2. **Error accumulation in long videos**: Early [BOX] errors can still propagate in long sequences; the dual-prompt mechanism mitigates but does not fully eliminate this issue.
3. **TSF pseudo-label quality**: TSF relies on GroundingDINO + CLDTracker for pseudo-tracking; severe noise or identity switches degrade training quality.
4. Future directions: higher-recall proposal methods, online correction mechanisms, and stronger tracking supervision signals.

## Related Work & Insights

- **Artemis**: Motivates the TSF design—tracking target-specific features improves temporal consistency.
- **Groma**: Inspires the dual-prompt design of using box prompting to enhance fine-grained visual grounding.
- **VideoGLaMM / UniPixel / GLUS**: Three baselines with distinct design philosophies; SPARROW's successful integration into all three validates its generality.
- The approach to combining with SAM2 is noteworthy: frozen Hiera features are used for proposal generation while the prompt encoder interface remains unchanged.

## Rating

⭐⭐⭐⭐ Strongly engineering-oriented with elegant modular design and comprehensive experiments (3 baselines × 6 datasets). However, the core technical contributions (proposal-based grounding + tracking pseudo-labels) are of moderate novelty, representing a well-executed combination of existing components rather than fundamental innovation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] VideoFusion: A Spatio-Temporal Collaborative Network for Multi-modal Video Fusion](videofusion_a_spatio-temporal_collaborative_network_for_multi-modal_video_fusion.md)
- [\[CVPR 2026\] CodePercept: Code-Grounded Visual STEM Perception for MLLMs](codepercept_code-grounded_visual_stem_perception_for_mllms.md)
- [\[CVPR 2026\] LFPC: Learning to Focus and Precise Cropping for MLLMs](lfpc_learning_to_focus_and_precise_cropping_for_mllms.md)
- [\[CVPR 2026\] HumanVBench: Probing Human-Centric Video Understanding in MLLMs with Automatically Synthesized Benchmarks](humanvbench_probing_human_centric_video_understanding_in_mllms_with_automatica.md)
- [\[CVPR 2026\] Linking Perception, Confidence and Accuracy in MLLMs](linking_perception_confidence_and_accuracy_in_mllms.md)

</div>

<!-- RELATED:END -->

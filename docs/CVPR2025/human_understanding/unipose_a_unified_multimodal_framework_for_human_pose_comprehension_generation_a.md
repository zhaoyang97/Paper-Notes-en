---
title: >-
  [Paper Note] UniPose: A Unified Multimodal Framework for Human Pose Comprehension, Generation and Editing
description: >-
  [CVPR 2025][Human Understanding][Human Pose] UniPose proposes the first unified multimodal framework that utilizes LLMs to discretize 3D human pose into pose tokens that share a vocabulary with text tokens. Through a mixture-of-visual-encoders and a mixed attention mechanism, it achieves unified modeling of seven core pose tasks (comprehension, generation, and editing) across images, text, and 3D SMPL poses.
tags:
  - "CVPR 2025"
  - "Human Understanding"
  - "Human Pose"
  - "Large Language Models"
  - "Pose Comprehension"
  - "Generation and Editing"
  - "Pose Tokenizer"
  - "Unified Multimodal Framework"
date: 2026-05-08
content_hash: c6d4bcbd3ae6c028
---

# UniPose: A Unified Multimodal Framework for Human Pose Comprehension, Generation and Editing

**Conference**: CVPR 2025  
**arXiv**: [2411.16781](https://arxiv.org/abs/2411.16781)  
**Code**: [https://github.com/liyiheng23/UniPose](https://github.com/liyiheng23/UniPose)  
**Area**: Human Understanding / Multimodal Large Models  
**Keywords**: Human Pose, Large Language Models, Pose Comprehension, Generation and Editing, Pose Tokenizer, Unified Multimodal Framework

## TL;DR

UniPose proposes the first unified multimodal framework that utilizes LLMs to discretize 3D human pose into pose tokens that share a vocabulary with text tokens. Through a mixture-of-visual-encoders and a mixed attention mechanism, it achieves unified modeling of seven core pose tasks (comprehension, generation, and editing) across images, text, and 3D SMPL poses.

## Background & Motivation

**Background**: Human pose-related tasks cover three major categories: comprehension (generating text descriptions from poses/images), generation (generating 3D poses from text/images), and editing (modifying poses based on instructions). Existing methods are independent of each other — PoseScript performs pose-to-text description, HMR 2.0 performs image-to-pose estimation, and PoseFix performs pose difference description and editing. ChatPose attempts to use LLMs to generate 3D poses but only covers single-pose generation.

**Limitations of Prior Work**: (1) Existing works study pose comprehension, generation, and editing as separate tasks, where each method only supports control signals of a single modality and cannot flexibly switch within the same framework; (2) Methods like ChatPose encode 3D poses into continuous high-dimensional features while text is encoded as discrete tokens, and this non-unified treatment imposes an extra burden on the LLM when modeling cross-modal interactions; (3) Mainstream MLLMs use CLIP as the visual encoder, but the global supervision signal of CLIP is hard to capture fine-grained pose information like keypoints and parsing maps.

**Key Challenge**: Human pose is a modality with a spatial structure, whose intrinsic logic differs from serialized text — there is no causal dependency among joint rotations. However, existing LLMs uniformly adopt causal autoregressive modeling, which is suboptimal for modeling spatial modalities. Meanwhile, there is a lack of an effective scheme to unify 3D pose, vision, and text into the same representation space.

**Goal**: To build a general framework that simultaneously supports seven core tasks in a single LLM: pose comprehension (single pose description, pose pair difference description, image-to-text, image pair difference description), pose generation (text-to-pose, image-to-pose estimation), and pose editing (modifying poses based on instructions).

**Key Insight**: It is observed that human poses exhibit a language-like semantic coupling — combinations of different joint rotations can express finite semantic units (e.g., "raising hands", "bending over"), and thus can be discretized into token sequences just like language.

**Core Idea**: Compress 3D SMPL poses into discrete pose tokens using VQ-VAE to extend the LLM vocabulary for unifying pose and text representations; enhance fine-grained pose perception using a mixture-of-visual-encoders (CLIP + pose estimation ViT); adapt to the intrinsic logics of different modalities using a mixed attention mechanism (causal for text, bidirectional for pose).

## Method

### Overall Architecture

UniPose consists of three core components: (1) Pose Tokenizer — discretizing SMPL pose parameters into token sequences based on VQ-VAE; (2) Visual Processor — a dual-encoder mixing CLIP ViT and pose-specific ViT; (3) Pose-aware LLM — a vision-language model based on LLaVA-1.6V that extends the pose vocabulary and introduces mixed attention. Inputs can be images, text, 3D poses, or their combinations, and outputs can be text descriptions or 3D poses, with the specific task determined by instructions.

### Key Designs

1. **Pose Tokenizer**:

    - **Function**: Convert continuous 3D SMPL pose parameters into discrete token sequences, achieving a unified representation with text tokens.
    - **Mechanism**: Incorporating a VQ-VAE architecture, the encoder consists of multi-layer 1D convolutions to encode SMPL pose parameters $\boldsymbol{p} = [\boldsymbol{\gamma}, \boldsymbol{\theta}]$ (6D root orientation + $6K$ dimensional joint rotations) into latent embeddings $\boldsymbol{z} \in \mathbb{R}^{L_p \times d_p}$, which are then mapped through vector quantization to the closest entries in codebook $\mathcal{B}_p$ (size $M=2048$), yielding $L_p=80$ discrete pose tokens. The decoder consists of 1D deconvolution layers, reconstructing quantized embeddings back into the original pose space. Training uses reconstruction loss + embedding loss + commitment loss.
    - **Design Motivation**: Putting 3D pose and language into the same discrete vocabulary space $\mathcal{V} = \{\mathcal{V}_t, \mathcal{V}_p\}$ allows the LLM to naturally switch between text and pose without designing specific cross-modal fusion mechanisms. This is easier for the LLM to model than the continuous feature representations in ChatPose.

2. **Mixture-of-Visual-Encoders**:

    - **Function**: Enhance fine-grained pose perception capabilities within the multimodal framework.
    - **Mechanism**: Combine two visual encoders — CLIP ViT ($f_a$) providing global semantically aligned visual features $\mathbf{v_a} \in \mathbb{R}^{L_v \times d_a}$, and pose-specific ViT ($f_b$, pre-trained on pose estimation tasks, such as the HMR 2.0 backbone) providing fine-grained joint-aware features $\mathbf{v_b} \in \mathbb{R}^{L_v \times d_b}$. The two are concatenated along the channel dimension and then mapped via a trainable projection layer to the text embedding dimension of the LLM: $\mathbf{v} = [\mathbf{v_a} | \mathbf{v_b}]^T W$.
    - **Design Motivation**: Ablation studies show that utilizing CLIP alone results in a pose estimation MPJPE as high as 193.4mm, which drops to 96.3mm after integrating the pose ViT. Contrastive image-text training of CLIP provides global semantic understanding but lacks pixel-level accuracy; the pose ViT provides precise joint localization but lacks semantic alignment. The dual encoders complement each other.

3. **Mixed Attention Mechanism**:

    - **Function**: Adapt to the different intrinsic logical relationships between text and pose tokens.
    - **Mechanism**: Apply standard causal (autoregressive) attention to text token sequences, and bidirectional attention to pose token sequences. During pose generation and editing, $L_p$ learnable pose queries $\mathcal{Q}$ are initialized to predict all pose tokens in parallel in a single forward pass, rather than autoregressively generating them step-by-step. Each pose token can attend to all other pose tokens within the same pose sequence, but can only attend to previous text tokens.
    - **Design Motivation**: Pose tokens encode spatial joint position information, and there is no causal dependency between joints — the rotation of the left hand does not "causally" determine the rotation of the right hand. Bidirectional attention allows different joints to refer to each other to determine a globally consistent pose, and parallel prediction significantly accelerates inference.

### Loss & Training

Four-stage training scheme:

1. **Pose Tokenizer Training**: Train the VQ-VAE using the AMASS + MOYO datasets, $\mathcal{L}_{vq} = \mathcal{L}_r + \mathcal{L}_e + \mathcal{L}_c$, 240 epochs.
2. **Pose-Text Alignment Pre-training**: Train the LLM using LoRA on PoseScript + PoseFix, covering 4 pure pose-text tasks, 6 epochs.
3. **Visual Projector Pre-training**: Train the visual projection layer for alignment on image-text data, covering 3 image-text tasks, 2 epochs.
4. **Instruction Fine-Tuning**: 200 instruction templates, jointly training the visual projection layer and the LLM (LoRA), integrating all 7 tasks.

## Key Experimental Results

### Main Results

| Task | Method | Key Metrics |
|------|------|----------|
| Pose-to-Text | PoseScript | R-Precision Top-1: 91.6 |
| Pose-to-Text | **UniPose** | R-Precision Top-1: 85.6, Top-3: **97.6** |
| Pose-Diff | PoseFix | R-Precision Top-1: 64.6 |
| Pose-Diff | **UniPose** | R-Precision Top-1: **67.9**, Top-3: **88.6** |
| Text-to-Pose | PoseScript | MPJPE: **318.0**, PA: **161.3** |
| Text-to-Pose | **UniPose** | MPJPE: 308.6, PA: 171.1 |
| Pose Estimation (3DPW) | HMR2.0 | MPJPE: **70.0**, PA: **44.5** |
| Pose Estimation (3DPW) | UniPose | MPJPE: 94.7, PA: 59.1 |
| Pose Editing | PoseFix | MPJPE: 300.2, FID: 0.019 |
| Pose Editing | **UniPose** | MPJPE: **270.3**, FID: **0.015** |
| Image-to-Text | GPT4V | R-Precision Top-1: 17.7 |
| Image-to-Text | **UniPose** | R-Precision Top-1: **24.5** |

### Ablation Study

| CLIP-ViT | Pose-ViT | MPJPE↓ | PA-MPJPE↓ | BLEU-4↑ | ROUGE-L↑ |
|:---:|:---:|:---:|:---:|:---:|:---:|
| ✓ | ✗ | 193.4 | 86.1 | 11.1 | 30.2 |
| ✗ | ✓ | 96.3 | 59.1 | 12.5 | 31.0 |
| ✓ | ✓ | **96.1** | **58.9** | **13.3** | **31.7** |

Ablation of mixed attention vs. causal attention shows that bidirectional attention outperforms pure causal attention in both pose generation and editing tasks.

### Key Findings

- UniPose covers all 7 tasks within a unified framework, achieving competitive or even superior performance; in particular, it outperforms specialized methods in Pose-Diff and Pose Editing.
- The comparison between single-task training (UniPose†) and multi-task joint training indicates that the unified learning strategy brings significant cross-task knowledge transfer, improving the $R^{T2P}$ Top-5 of Text-to-Pose from 67.5 to 73.7.
- Using the CLIP visual encoder alone results in double the pose estimation error compared to the dual-encoder (193.4 vs 96.1 MPJPE); thus, the pose ViT is critical for precise pose perception.
- UniPose demonstrates zero-shot capabilities, such as text-enhanced pose estimation (assisting pose estimation with textual instructions).

## Highlights & Insights

- The idea of analogizing 3D pose to "language" and discretizing it is highly inspiring — this tokenization paradigm is expanding from text to various other modalities like action, pose, and audio.
- The mixed attention mechanism is a powerful correction to the oversimplification of "using causal generation for all modalities"; spatial modalities should indeed be modeled bidirectionally/in parallel.
- The four-stage training (tokenizer → pose-text alignment → visual alignment → instruction fine-tuning) presents a complete multimodal LLM adaptation scheme.

## Limitations & Future Work

- The accuracy of pose estimation (MPJPE of 94.7) still has a large gap compared to the specialized SOTA (HMR2.0: 70.0); the unified framework sacrifices some accuracy for generality.
- Using a codebook size of 2048 and 80 tokens to represent a pose can introduce quantization errors that may limit the reconstruction quality of extreme poses.
- It only supports single-person poses, leaving multi-person scenarios unaddressed.
- Currently, pose sequences/motions are not supported, and extending to temporal sequences is an important future direction.
- The ImageScript and ImageDiff datasets are self-constructed, whose quality and scale may affect the performance of image-related tasks.

## Related Work & Insights

- Direct comparison with ChatPose — ChatPose only performs generation tasks, whereas UniPose covers all 7 tasks across comprehension, generation, and editing.
- The concept of the Pose Tokenizer shares similarities with motion generation works like MotionGPT, reinforcing that "discretizing non-text modalities to interface with LLMs" is an effective and general approach.
- The mixed attention mechanism can inspire modeling approaches for other spatial modalities (such as 3D point clouds and molecular structures) within LLMs.

## Rating

- **Novelty**: 8/10 — The first LLM framework to unify pose comprehension, generation, and editing, with a clear design philosophy.
- **Experimental Thoroughness**: 8/10 — Comprehensive evaluation across 7 tasks, including ablation and zero-shot experiments.
- **Writing Quality**: 8/10 — Complete structure, with clear definitions of tasks and method descriptions.
- **Value**: 7/10 — The concept of a unified framework is valuable, though there is still a gap between each subtask and specialized SOTAs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] SimMotionEdit: Text-Based Human Motion Editing with Motion Similarity Prediction](simmotionedit_text-based_human_motion_editing_with_motion_similarity_prediction.md)
- [\[CVPR 2025\] ChatGarment: Garment Estimation, Generation and Editing via Large Language Models](chatgarment_garment_estimation_generation_and_editing_via_large_language_models.md)
- [\[CVPR 2025\] UniHOPE: A Unified Approach for Hand-Only and Hand-Object Pose Estimation](unihope_a_unified_approach_for_hand-only_and_hand-object_pose_estimation.md)
- [\[CVPR 2026\] MimicTalker: A Multimodal Interactive and Memory-Enhanced Framework for Real-Time Dyadic 3D Head Generation](../../CVPR2026/human_understanding/mimictalker_a_multimodal_interactive_and_memory-enhanced_framework_for_real-time.md)
- [\[CVPR 2025\] MotionReFit: Dynamic Motion Blending for Versatile Motion Editing](motionrefit_motion_editing.md)

</div>

<!-- RELATED:END -->

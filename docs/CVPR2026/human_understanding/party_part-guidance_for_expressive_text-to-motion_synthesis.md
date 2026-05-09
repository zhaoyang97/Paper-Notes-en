---
title: >-
  [Paper Note] ParTY: Part-Guidance for Expressive Text-to-Motion Synthesis
description: >-
  [CVPR 2026][Human Understanding][text-to-motion] This paper proposes ParTY, a framework that employs a Part-Guided Network and Part-aware Text Grounding to significantly improve text–motion semantic alignment at the body-part level while preserving whole-body motion coherence, thereby resolving the fundamental trade-off between part expressiveness and global coherence that exists between holistic and part-decomposition methods.
tags:
  - CVPR 2026
  - Human Understanding
  - text-to-motion
  - body part guidance
  - VQ-VAE
  - part-aware text alignment
  - motion coherence
date: 2026-05-08
content_hash: 67a4c6d9979eb8c1
---

# ParTY: Part-Guidance for Expressive Text-to-Motion Synthesis

**Conference**: CVPR 2026
**arXiv**: [2603.09611](https://arxiv.org/abs/2603.09611)
**Code**: [Project Page](https://heokunho.github.io/ParTY/)
**Area**: Human Understanding
**Keywords**: text-to-motion, body part guidance, VQ-VAE, part-aware text alignment, motion coherence

## TL;DR

This paper proposes ParTY, a framework that employs a Part-Guided Network and Part-aware Text Grounding to significantly improve text–motion semantic alignment at the body-part level while preserving whole-body motion coherence, thereby resolving the fundamental trade-off between part expressiveness and global coherence that exists between holistic and part-decomposition methods.

## Background & Motivation

- **Application potential of text-driven motion generation**: Text-to-motion has broad applications in animation, VR, gaming, and robotics, and has seen notable architectural advances in recent years (VQ-VAE + Transformer, diffusion models, etc.).
- **Limitations of Prior Work — holistic methods**: Most existing methods treat the human body as a single entity and generate whole-body motion globally. Although global coherence is good, these methods struggle to accurately model fine-grained descriptions targeting specific body parts, and part-level semantics are often ignored or misrepresented.
- **Two deficiencies of part-decomposition methods**: Methods such as ParCo and LGTM decompose the body into independent components (arms, legs, etc.) for separate generation, offering stronger part-level control, but suffer from (i) the lack of an explicit mechanism to align text semantics with individual parts, and (ii) incoherent whole-body motion resulting from naive concatenation of independently generated parts (e.g., neck twisting, misaligned upper/lower body orientation).
- **Key Challenge**: There is a fundamental trade-off between part expressiveness and whole-body coherence — part-decomposition methods improve the former at the expense of the latter, and no existing method achieves both simultaneously.
- **State of the Field — missing evaluation protocol**: Existing metrics (e.g., FID, R-Precision) operate only at the whole-body level and cannot accurately measure part-level semantic alignment quality, nor is there a metric that directly assesses cross-part motion coherence.
- **Goal**: Design a unified framework that bridges the advantages of holistic and part-decomposition methods — achieving fine-grained part–text alignment while producing globally coherent motion outputs — and introduce novel part-level and coherence-level evaluation metrics.

## Method

### Overall Architecture

ParTY adopts a two-stage training strategy. In **Stage 1**, a Temporal-aware VQ-VAE is trained to quantize whole-body and part-specific (arms, legs) motion sequences into separate discrete codebooks. In **Stage 2**, a holistic Transformer and part Transformers are trained jointly: text embeddings are processed by Part-aware Text Grounding (PTG) and fed into the respective part Transformers, which first generate part motion tokens to form the Part Guidance; this guidance is then injected into the holistic Transformer to condition whole-body motion generation, with part information continuously integrated via Holistic-Part Fusion (HPF) throughout the generation process. At inference time, the predicted codebook sequences are decoded back into motion by the pre-trained VQ-VAE decoder from Stage 1.

### Key Design 1: Temporal-aware VQ-VAE

- **Function**: Enhances temporal information retention during motion quantization, addressing the temporal information loss of standard VQ-VAE under fixed-window compression.
- **Mechanism**: Introduces Local Temporal Enhancement (LTE) and Global Temporal Enhancement (GTE). LTE groups frame-level features into windows, computes weights via an MLP within each group, and produces group-level features through weighted summation. GTE constructs a Graph Convolutional Network (GCN) over the group-level features to capture global temporal dependencies before quantization into the codebook.
- **Design Motivation**: Increasing the window size reduces model parameters and inference time, but standard VQ-VAE suffers severe information loss. The Temporal-aware VQ-VAE maintains high-quality quantization under large windows, thereby balancing efficiency and performance — e.g., increasing the window from 4 to 12 reduces inference time by 64%, while FID rises only marginally to 0.042 (compared to MoMask's original 0.126).

### Key Design 2: Part-aware Text Grounding (PTG)

- **Function**: Transforms a single text embedding into multiple diverse embeddings and dynamically selects the most appropriate embedding for each body part.
- **Mechanism**: A CLIP text embedding is transformed by $K$ independent MLPs to produce $K$ diverse embeddings, which are then adaptively weighted and selected by part-specific Gate networks. During training, an LLM generates auxiliary textual descriptions for each part (e.g., for the input "walk forward and pick something up with the left hand," the LLM generates the arm description "the left arm picks up an object from the ground"), and an L1 loss aligns the PTG outputs with these part-description embeddings. No LLM is required at inference time.
- **Design Motivation**: A single text embedding cannot distinguish the semantic requirements of different body parts. A contrastive diversity loss ensures the $K$ embeddings are semantically consistent with the original but directionally diverse, and the Gate network selects relevant dimensions according to part-specific characteristics. This outperforms LGTM's direct LLM-based text decomposition, which discards the full-sentence context.

### Key Design 3: Part-Guided Network + Holistic-Part Fusion

- **Function**: Generates part motion tokens as guidance signals first, then uses them to condition whole-body motion generation; part and whole-body information are continuously fused via attention mechanisms throughout generation.
- **Mechanism**: Generation proceeds in a recurrent fashion — within each cycle, the part Transformers autoregressively generate $T$ steps of tokens; tokens from all parts are summed and fused via an MLP to form the Part Guidance; the holistic Transformer then generates whole-body tokens over the same time span, conditioned on this Part Guidance. Holistic-Part Fusion (HPF) concatenates whole-body, arm, and leg tokens for self-attention, followed by cross-attention (whole-body tokens as queries, part tokens as keys/values) to integrate information.
- **Design Motivation**: Independently generating each part and naively concatenating them causes spatiotemporal incoherence. The Part-Guided Network allows part information to "lead" as a look-ahead signal for whole-body generation, while HPF dynamically captures inter-part relationships at each step (attention maps show significantly higher weights on the parts mentioned in the corresponding description), fundamentally avoiding the coherence problems of naive concatenation.

## Loss & Training

The total loss comprises four terms:

$$\mathcal{L} = \mathcal{L}_{\text{hol}} + \mathcal{L}_{\text{part}} + \lambda_{\text{div}} \mathcal{L}_{\text{div}} + \lambda_{\text{aux}} \mathcal{L}_{\text{aux}}$$

- $\mathcal{L}_{\text{hol}}$: Cross-entropy loss for the holistic Transformer, supervising autoregressive prediction of whole-body motion tokens.
- $\mathcal{L}_{\text{part}}$: Cross-entropy loss for the part Transformers (arms + legs), supervising prediction of part motion tokens.
- $\mathcal{L}_{\text{div}}$: Contrastive diversity loss, encouraging the $K$ transformed embeddings to differ from one another while remaining semantically consistent with the original embedding.
- $\mathcal{L}_{\text{aux}}$: Auxiliary L1 loss, aligning PTG outputs with LLM-generated part-description embeddings (training only).

VQ-VAE stage: $\mathcal{L}_{vq} = \mathcal{L}_{rec} + \lambda_{app} \cdot \mathcal{L}_{app}$, comprising an L1 reconstruction loss and an L2 codebook commitment loss.

## Key Experimental Results

Evaluation is conducted on HumanML3D (14,616 motions, 44,970 text annotations) and KIT-ML (3,911 motions, 6,278 text annotations).

### Main Results — Table 1: Whole-body Evaluation on HumanML3D

| Method | R-Prec Top-1↑ | R-Prec Top-3↑ | FID↓ | MM-Dist↓ |
|--------|--------------|--------------|------|----------|
| T2M-GPT | 0.491 | 0.775 | 0.116 | 3.118 |
| ParCo | 0.515 | 0.801 | 0.109 | 2.927 |
| MoMask | 0.521 | 0.807 | 0.045 | 2.958 |
| BAMM | 0.525 | 0.814 | 0.055 | 2.919 |
| **ParTY** | **0.550** | **0.836** | **0.035** | **2.779** |

ParTY achieves state-of-the-art results on all core metrics. R-Prec Top-1 exceeds the second-best method (BAMM) by 2.5 percentage points, and FID is reduced by 36%.

### Table 2: Part-level Evaluation on HumanML3D

| Method | Part | R-Prec Top-1↑ | FID↓ | MM-Dist↓ |
|--------|------|--------------|------|----------|
| MoMask | Arms | 0.452 | 0.175 | 3.440 |
| ParCo | Arms | 0.468 | 0.215 | 3.326 |
| **ParTY** | **Arms** | **0.506** | **0.133** | **3.079** |
| MoMask | Legs | 0.403 | 0.104 | 3.513 |
| ParCo | Legs | 0.407 | 0.118 | 3.482 |
| **ParTY** | **Legs** | **0.463** | **0.078** | **3.122** |

### Table 3: Coherence Evaluation

| Method | Temporal Coherence↑ | Spatial Coherence↑ |
|--------|--------------------|--------------------|
| ParCo | 0.49 | 0.59 |
| MoMask | 0.84 | 0.90 |
| **ParTY** | **0.88** | **0.92** |

ParTY substantially surpasses both holistic and part-decomposition methods in part expressiveness, while achieving coherence scores that even slightly exceed the holistic method MoMask, validating the effectiveness of simultaneously satisfying both objectives.

### Table 4: Temporal-aware VQ-VAE Transferred to MoMask

| Method | Window Size | Recon. FID↓ | Gen. FID↓ | Inference Time |
|--------|-------------|-------------|-----------|----------------|
| MoMask | 4 | 0.020 | 0.045 | 80ms |
| MoMask + Ours | 4 | 0.003 (+85%) | 0.033 (+26%) | - |
| MoMask | 8 | 0.042 | 0.094 | 43ms (-46%) |
| MoMask + Ours | 8 | 0.005 (+88%) | 0.039 (+58%) | - |
| MoMask | 12 | 0.079 | 0.126 | 29ms (-64%) |
| MoMask + Ours | 12 | 0.011 (+86%) | 0.042 (+67%) | - |

### Ablation Study — Table 5: Component Ablation

| PG | PTG | HPF | R-Prec Top-1↑ | FID↓ | MM-Dist↓ |
|----|-----|-----|--------------|------|----------|
| | | | 0.494 | 0.158 | 3.087 |
| ✓ | | | 0.520 | 0.086 | 2.913 |
| ✓ | ✓ | | 0.545 | 0.051 | 2.799 |
| ✓ | ✓ | ✓ | **0.550** | **0.035** | **2.779** |

## Highlights & Insights

- **Resolving the core trade-off**: ParTY is the first method to effectively resolve the fundamental trade-off between part expressiveness and whole-body coherence in part-decomposition approaches. The Part-Guided Network allows part information to proactively guide whole-body generation rather than being concatenated post-hoc.
- **Elegant PTG design**: Generating diverse embeddings via contrastive learning combined with dynamic Gate-based selection is a more elegant solution than LGTM's LLM-based text decomposition, and the LLM dependency is confined to training, introducing no additional overhead at inference.
- **Generalizability of the Temporal-aware VQ-VAE**: The module can be directly transplanted into methods such as MoMask with significant gains (FID reduced by 26%–67%), and at large window sizes reduces inference time by 64% with negligible performance degradation.
- **Comprehensive evaluation framework**: The proposed part-level and coherence-level metrics (TC, SC) fill an evaluation gap in the field and provide the first quantitative verification of the coherence deficiencies inherent to part-decomposition methods.

## Limitations & Future Work

- Body part decomposition is limited to two coarse groups (arms and legs), with no capability to model finer-grained parts such as fingers, the head, or the torso.
- Training relies on an LLM to generate auxiliary part descriptions, increasing data preparation costs; the quality of LLM-generated descriptions may also affect PTG training.
- The recurrent "parts first, whole-body second" generation pipeline increases the number of inference steps; although the Temporal-aware VQ-VAE partially compensates through larger windows, overall inference time remains higher than single-Transformer approaches.
- Evaluation is conducted only on HumanML3D and KIT-ML, leaving more complex scenarios (e.g., multi-person interaction, human–object interaction, long-sequence generation) unexplored.
- Although the TC and SC coherence metrics effectively differentiate the deficiencies of part-decomposition methods, their statistical properties and correlation with human perception require broader validation.

## Related Work & Insights

- **Holistic text-to-motion**: Methods such as T2M-GPT, MoMask, BAMM, and MMM, based on VQ-VAE + Transformer or diffusion models, achieve good whole-body coherence but lack fine-grained part-level detail.
- **Part-decomposition methods**: SCA (upper/lower body split), AttT2M (body-part attention encoder), ParCo (per-part VQ-VAE with token sharing), and LGTM (LLM-based text decomposition into part descriptions) advance part-level control but suffer from poor coherence.
- **Motion quantization**: VQ-VAE is widely adopted in text-to-motion, but fixed windows cause temporal information loss. The proposed Temporal-aware VQ-VAE enhances temporal retention at both local and global levels.

## Rating

| Dimension | Score |
|-----------|-------|
| Novelty | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Miburi: Towards Expressive Interactive Gesture Synthesis](miburi_towards_expressive_interactive_gesture_synthesis.md)
- [\[CVPR 2026\] MoLingo: Motion-Language Alignment for Text-to-Human Motion Generation](molingo_motion-language_alignment_for_text-to-motion_generation.md)
- [\[CVPR 2026\] HandX: Scaling Bimanual Motion and Interaction Generation](handx_scaling_bimanual_motion_and_interaction_generation.md)
- [\[ICCV 2025\] DreamActor-M1: Holistic, Expressive and Robust Human Image Animation with Hybrid Guidance](../../ICCV2025/human_understanding/dreamactor-m1_holistic_expressive_and_robust_human_image_animation_with_hybrid_g.md)
- [\[CVPR 2026\] HandDreamer: Zero-Shot Text to 3D Hand Model Generation](handdreamer_zero_shot_text_to_3d_hand_model_generation.md)

<!-- RELATED:END -->

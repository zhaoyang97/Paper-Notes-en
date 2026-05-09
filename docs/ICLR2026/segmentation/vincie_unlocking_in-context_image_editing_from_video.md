---
title: >-
  [Paper Note] VINCIE: Unlocking In-context Image Editing from Video
description: >-
  [ICLR 2026][Segmentation][in-context editing] VINCIE is a framework that first demonstrates that in-context image editing models can be learned entirely from native video data. By annotating videos as interleaved multimodal sequences and designing three proxy tasks (NIP/CSP/NSP), it achieves state-of-the-art performance on multi-turn editing benchmarks, improving the 5-turn editing success rate from less than 2% (baseline) to 25%.
tags:
  - ICLR 2026
  - Segmentation
  - in-context editing
  - video learning
  - multi-turn editing
  - DiT
  - segmentation prediction
date: 2026-05-08
content_hash: 1193c16881205058
---

# VINCIE: Unlocking In-context Image Editing from Video

**Conference**: ICLR 2026
**arXiv**: [2506.10941](https://arxiv.org/abs/2506.10941)
**Code**: [vincie2025.github.io](https://vincie2025.github.io/)
**Area**: Image Segmentation
**Keywords**: in-context editing, video learning, multi-turn editing, DiT, segmentation prediction

## TL;DR

VINCIE is a framework that first demonstrates that in-context image editing models can be learned entirely from native video data. By annotating videos as interleaved multimodal sequences and designing three proxy tasks (NIP/CSP/NSP), it achieves state-of-the-art performance on multi-turn editing benchmarks, improving the 5-turn editing success rate from less than 2% (baseline) to 25%.

## Background & Motivation

**Background**: In-context image editing enables users to iteratively modify images through multi-turn interactions. Existing methods rely on task-specific pipelines and expert models (e.g., segmentation, inpainting) to construct paired training data.

**Limitations of Prior Work**: (1) Constructing paired data for multi-turn editing is extremely difficult; existing methods can only mine single-turn editing pairs. (2) Dependence on task-specific pipelines limits the generality and scalability of data. (3) Consistency degradation and error accumulation in multi-turn editing remain serious issues.

**Key Challenge**: A fundamental tension exists between the scarcity of high-quality multi-turn editing training data and the need for models to learn long-range contextual dependencies.

**Goal**: To investigate whether a meaningful in-context image editing model can be learned solely from video data, without any independently curated image pairs.

**Key Insight**: Videos naturally contain rich visual dynamics—object appearance and disappearance, pose changes, camera motion—which implicitly provide learning signals for editing operations.

**Core Idea**: Native video data is used to construct interleaved multimodal sequences (frames + transition descriptions + segmentation masks), and a DiT model is trained via three proxy tasks to learn context-aware image editing.

## Method

### Overall Architecture

(1) $K$ frames are sparsely sampled from a video, and a VLM annotates inter-frame visual transition descriptions $T_i$; (2) GroundingDINO + SAM2 generate segmentation masks for regions of editing interest (RoE); (3) an interleaved multimodal sequence $(I_0, T_0, M_{00}, M_{01}, I_1, \ldots, I_K)$ is constructed; (4) a DiT model is jointly trained via three proxy tasks.

### Key Designs

1. **Scalable Video Data Annotation Pipeline**:
    - **Function**: Converts native video into interleaved multimodal sequences suitable for training in-context editing models.
    - **Mechanism**: A hybrid sampling strategy is adopted (uniform interval sampling + fixed-frame-count sampling). Adjacent frames are annotated by a VLM using chain-of-thought (CoT) reasoning to describe visual transitions (enumerating differences across aspects → summarizing into an editing instruction $T_i$), followed by GroundingDINO + SAM2 to extract RoE segmentation masks.
    - **Design Motivation**: Uniform interval sampling captures fine-grained object-level changes, while fixed-frame-count sampling covers large-scale scene-level changes. RoE masks provide explicit spatial localization signals.

2. **DiT Architecture and In-Context Compositional Learning**:
    - **Function**: Learns context-conditioned image generation within a Diffusion Transformer framework.
    - **Mechanism**: The modeling objective is $\log p(S) = \sum_{i=1}^{M} \log p(I_i | I_0, \ldots, T_{i-1}, I_{i-1})$. Learnable `<TURN>` tokens separate turns; 1D RoPE is applied to text and 3D RoPE to images. Both full attention and block-level causal attention variants are provided. Random dropout is applied to contextual inputs (frames and masks) to improve generalization.
    - **Design Motivation**: Pre-trained weights from video foundation models provide strong initialization; context dropout encourages the model to flexibly leverage varying combinations of contextual information.

3. **Three-Proxy-Task Learning Framework**:
    - **Function**: Enhances editing capability through three complementary tasks.
    - **Mechanism**: (i) Next-frame Image Prediction (NIP)—the primary task, learning context-aware editing; (ii) Current Segmentation Prediction (CSP)—improves grounding ability by identifying regions to be edited; (iii) Next-frame Segmentation Prediction (NSP)—enhances controllable generation by assisting dynamic layout adjustment. All three tasks are jointly trained in a unified generative framework using the MSE diffusion loss under flow matching.
    - **Design Motivation**: CSP helps the model understand "what changed," while NSP helps the model anticipate "what will change," and both jointly enhance the editing quality of NIP.

### Loss & Training

The MSE diffusion loss under flow matching is used. RoE masks are included in training with an 80% probability. Context dropout rates: 20% for the current frame, 70% for the current RoE, and 70% for the next-frame RoE. Inference uses 50 sampling steps with CFG scale = 10. The 3B model is trained for 15k steps on 256 × H100 GPUs for approximately 30 hours; the 7B model for 40k steps for approximately 150 hours. The training data comprises approximately 10M session instances.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours (7B+SFT) | Prev. SOTA / Baseline | Notes |
|--------|------|------|------|------|
| MagicBrush Turn-1 | DINO | 0.891 | 0.886 (Nano Banana) | Surpasses all academic methods |
| MagicBrush Turn-3 | DINO | 0.775 | 0.773 (Nano Banana) | Multi-turn advantage more pronounced |
| MSE-Bench Turn-1 | Success Rate | 0.950 | 0.937 (Step1X-Edit) | Second only to Bagel |
| MSE-Bench Turn-5 | Success Rate | 0.487 | 0.413 (Bagel) | Substantially outperforms open-source methods |
| MSE-Bench Turn-5 | Success Rate | 0.210→0.487 | 3B→7B+SFT | Significant scaling effect |

### Ablation Study

| Configuration | Metric | Notes |
|------|------|------|
| w/o Seg. vs. w/ Seg. | MSE Turn-1: 0.847→0.887 | Segmentation prediction tasks improve editing performance |
| CS→I | MagicBrush DINO Turn-1: 0.797 | Current segmentation improves consistency |
| CS→NS→I | MagicBrush DINO Turn-3: 0.679 | Chained editing strategy is optimal |
| Pairwise vs. sequence | MSE Turn-5: 0.010→0.220 | Sequence data substantially outperforms paired data |
| Data scale 0.25M→10M | MSE Turn-5: 5%→22% | Approximately log-linear scaling |

### Key Findings

- Training solely on video data is sufficient to match SOTA methods that use paired editing data; video pre-training followed by SFT yields the best results.
- In-context editing effectively mitigates artifact accumulation across multi-turn editing.
- The model exhibits emergent capabilities not explicitly trained for: multi-concept composition, story generation, and chained editing.

## Highlights & Insights

- This work is the first to demonstrate the feasibility of learning in-context image editing from purely video data, opening a new paradigm for data sourcing. The massive scale of available video data confers a natural scalability advantage.
- The three proxy tasks are elegantly designed: CSP addresses "where things changed," NSP anticipates "where things will change," and their synergy enhances the editing quality of NIP—a decomposition strategy worth adopting broadly.

## Limitations & Future Work

- Video training introduces subject position shifts, which are partially mitigated by segmentation prediction but not fully resolved.
- A substantial gap remains compared to commercial models (GPT-4o: 62.7%, Nano Banana: 64.3%) on MSE-Bench Turn-5.
- The evaluation relies solely on GPT-4o automatic assessment, without user satisfaction or preference studies.

## Related Work & Insights

- **vs. InstructPix2Pix**: IP2P depends on GPT-3 + SD to generate paired data and supports only single-turn editing; VINCIE leverages video to naturally support multi-turn interactions.
- **vs. OmniGen/OmniGen2**: OmniGen uses paired editing data and suffers a sharp drop in multi-turn success rate (only 8.3% at MSE Turn-5); VINCIE's contextual modeling is substantially more robust.
- **vs. UES/RealGeneral**: These methods exploit only two frames from a video, ignoring long-range context; VINCIE uses complete multi-frame sequences.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First to learn in-context editing from video data; a paradigm-level innovation with emergent capabilities.
- Experimental Thoroughness: ⭐⭐⭐⭐ Introduces the MSE-Bench benchmark; ablations are comprehensive; scaling analysis is thorough.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clear; methodology is described systematically; experimental presentation is polished.
- Value: ⭐⭐⭐⭐⭐ Establishes the video→editing paradigm; data scalability addresses a core bottleneck in the field.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] VEGGIE: Instructional Editing and Reasoning Video Concepts with Grounded Generation](../../ICCV2025/segmentation/veggie_instructional_editing_and_reasoning_video_concepts_with_grounded_generati.md)
- [\[CVPR 2026\] Towards Context-Aware Image Anonymization with Multi-Agent Reasoning](../../CVPR2026/segmentation/towards_context-aware_image_anonymization_with_multi-agent_reasoning.md)
- [\[ICCV 2025\] CAVIS: Context-Aware Video Instance Segmentation](../../ICCV2025/segmentation/cavis_context-aware_video_instance_segmentation.md)
- [\[CVPR 2026\] RobotSeg: A Model and Dataset for Segmenting Robots in Image and Video](../../CVPR2026/segmentation/robotseg_a_model_and_dataset_for_segmenting_robots_in_image_and_video.md)
- [\[ICLR 2026\] AMLRIS: Alignment-aware Masked Learning for Referring Image Segmentation](amlris_alignment-aware_masked_learning_for_referring_image_segmentation.md)

<!-- RELATED:END -->

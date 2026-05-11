---
title: >-
  [Paper Note] Where, What, Why: Towards Explainable Driver Attention Prediction
description: >-
  [ICCV 2025][Autonomous Driving][driver attention prediction] This paper proposes a new paradigm of "explainable driver attention prediction," introducing the first large-scale W³DA dataset and the LLada framework…
tags:
  - "ICCV 2025"
  - "Autonomous Driving"
  - "driver attention prediction"
  - "explainability"
  - "multimodal large language models"
  - "cognitive reasoning"
  - "gaze prediction"
date: 2026-05-08
content_hash: 5f953b187fb58b82
---

# Where, What, Why: Towards Explainable Driver Attention Prediction

**Conference**: ICCV 2025
**arXiv**: [2506.23088](https://arxiv.org/abs/2506.23088)
**Code**: [github.com/yuchen2199/Explainable-Driver-Attention-Prediction](https://github.com/yuchen2199/Explainable-Driver-Attention-Prediction)
**Area**: Autonomous Driving / Attention Modeling
**Keywords**: driver attention prediction, explainability, multimodal large language models, cognitive reasoning, gaze prediction

## TL;DR

This paper proposes a new paradigm of "explainable driver attention prediction," introducing the first large-scale W³DA dataset and the LLada framework, which unifies spatial attention prediction (Where), semantic parsing (What), and cognitive reasoning (Why) within a single end-to-end large language model-driven architecture.

## Background & Motivation

Driver attention modeling is a fundamental challenge in autonomous driving and cognitive science. Existing methods (e.g., ACT-Net, FBLNet) primarily focus on predicting *where* drivers look — generating spatial heatmaps to regress gaze distributions. However, this paradigm has fundamental limitations:

**Shallow implicit representations only**: Spatial heatmaps are essentially pixel-space regressions that provide only the *location* of attention, without revealing *why* a driver focuses on a particular region.

**Lack of semantic understanding**: Such methods cannot answer *what* the driver is looking at — a leading vehicle, a traffic light, or a pedestrian.

**Lack of cognitive explanation**: They cannot answer *why* the driver is looking there — whether to obey traffic rules, ensure driving safety, or navigate to a destination.

For example, a driver at an intersection may attend to a red light (obeying traffic rules), an approaching cyclist (ensuring safety), and the road ahead (route planning) — yet the underlying cognitive motivations behind such fixations have never been modeled.

The core contribution of this paper is to expand driver attention prediction from a singular "Where" to a unified "Where + What + Why" explainable paradigm, achieving more comprehensive attention understanding by integrating spatial, semantic, and cognitive knowledge.

## Method

### Overall Architecture

LLada (Large Language model-driven driver attention) consists of four core components:
1. **Pre-trained visual encoder** $\mathcal{F}_{\text{vis}}$: CLIP-ViT-L with a linear projector
2. **Large language model** $\mathcal{F}_{\text{LLM}}$: Vicuna-7B
3. **Special attention token** [ATTN]: encodes high-level cognitive cues
4. **Cognition-aware attention decoder** $\mathcal{F}_{\text{dec}}$: decodes cognitive information into pixel-level attention maps

### Key Designs

1. **W³DA Dataset Construction**:

    - Integrates four mainstream driver attention datasets: DR(eye)VE (normal driving), LBW (normal driving), BDDA (safety-critical scenarios), and DADA-2000 (traffic accidents)
    - Comprises 69,980 key samples from 3,548 video scenes
    - **Attention-aware keyframe selection**: Rather than uniform frame-rate sampling, keyframes are selected based on three criteria — (a) driving scene semantic similarity (CLIP CLS token cosine similarity), (b) KL divergence of attention spatial distributions, and (c) semantic similarity of attended regions. Frames are selected as keyframes when KL divergence exceeds a threshold or semantic similarity falls below a threshold.
    - **Semi-automatic annotation pipeline**: Qwen-VL-Max is guided by visual and contextual prompts to generate preliminary annotations via chain-of-thought reasoning (count attended regions → describe content → explain reasons), followed by expert human verification and revision.

2. **[ATTN] Token Design**:

    - A special [ATTN] token is added to the LLM vocabulary.
    - During attention prediction, the LLM output sequence includes the [ATTN] token; the corresponding embedding is projected via MLP and fed into the attention decoder.
    - **Design Motivation**: The [ATTN] token encodes high-level cognitive cues within the LLM through self-attention interactions with language tokens, effectively transferring semantic and causal reasoning from the text space to visual attention map generation.

3. **Cognition-Aware Attention Decoder**:

    - Integrates the [ATTN] embedding $\mathbf{h}_{\text{attn}}$ and visual features $\mathbf{h}_{\text{vis}}$ via cross-attention:

    $\mathbf{h}_{\text{dec}}' = \mathbf{h}_{\text{vis}} + \text{Repeat}(CA(\mathbf{h}_{\text{attn}}, \mathbf{h}_{\text{vis}}))$

    - The enhanced visual features are reshaped into a 3D feature map, dimensionality-reduced through 5 layers of 3×3 convolutions, and upsampled bilinearly to produce a full-resolution attention map.
    - **Design Motivation**: The [ATTN] embedding encodes cognitive information about "why to look here"; injecting it into visual features via cross-attention enables the decoder to understand not only *where* to look but also *why*.

### Loss & Training

The total loss is a weighted sum of the attention map loss and the text explanation loss:

$$\mathcal{L} = \lambda_{\text{map}} \mathcal{L}_{\text{map}} + \lambda_{\text{txt}} \mathcal{L}_{\text{txt}}$$

- Attention map loss: $\mathcal{L}_{\text{map}} = \lambda_{\text{bce}} \text{BCE}(\hat{\mathbf{A}}, \mathbf{A}) + \lambda_{\text{kl}} \text{KL}(\hat{\mathbf{A}}, \mathbf{A})$
- Text loss: $\mathcal{L}_{\text{txt}} = \lambda_{\text{what}} \text{CE}(\hat{\mathcal{S}}, \mathcal{S}) + \lambda_{\text{why}} \text{CE}(\hat{\mathcal{E}}, \mathcal{E})$

Training configuration: 4× A100 GPUs, DeepSpeed engine, AdamW (lr=3e-4), LoRA fine-tuning for the LLM, frozen visual encoder, attention decoder trained from scratch.

## Key Experimental Results

### Main Results — W³DA Attention Map Prediction

| Method | Type | KLdiv ↓ (Normal) | CC ↑ (Normal) | KLdiv ↓ (Critical) | KLdiv ↓ (Accident) |
|--------|------|-------------------|---------------|---------------------|---------------------|
| GBVS | Traditional | 2.572 | 0.294 | 2.238 | 2.826 |
| ERFNet | DNN | 1.979 | 0.558 | 1.593 | 2.181 |
| ConvNeXt | DNN | 2.042 | 0.570 | 1.765 | 3.049 |
| GazeXplain† | Multi-task | 2.578 | 0.477 | 2.769 | 3.109 |
| **LLada†** | **Multi-task** | **1.219** | **0.583** | **1.230** | **1.927** |

LLada outperforms all existing methods across all scenarios and metrics. In terms of KLdiv, it surpasses the second-best method (ERFNet) by 38.4% (normal), 22.8% (critical), and 11.7% (accident).

### Ablation Study — Mutual Gains from Joint Where/What/Why Reasoning

**Impact of Where on What/Why** (text explanation generation quality):

| Setting | BLEU ↑ | METEOR ↑ | ROUGE ↑ | CIDEr-R ↑ |
|---------|--------|----------|---------|-----------|
| Without Where (What+Why only) | Notable drop | Notable drop | Notable drop | Notable drop |
| **With Where (full LLada)** | **Highest** | **Highest** | **Highest** | **Highest** |

**Impact of What/Why on Where** (attention map prediction quality):

| Setting | KLdiv ↓ | CC ↑ |
|---------|---------|------|
| Where only | Baseline | Baseline |
| Where + What | Improved | Improved |
| **Where + What + Why** | **Best** | **Best** |

All three dimensions are mutually reinforcing: spatial attention provides localization cues that aid semantic and causal reasoning, while semantic and causal reasoning in turn improves attention localization accuracy.

### Key Findings

1. **Cross-domain generalization**: LLada, trained solely on W³DA, outperforms most models trained exclusively on full DR(eye)VE/BDDA/DADA test sets (KLdiv improvements of 29.8%, 20.7%, and 5.5%, respectively).
2. **Text explanation quality**: LLada comprehensively surpasses GazeXplain and two-stage baselines (attention predictor + fine-tuned LLaVA) on all text metrics, with CIDEr-R improvements exceeding 50%.
3. **Qualitative comparison**: LLada correctly attends to a pedestrian in front of the vehicle and generates contextually relevant cognitive explanations ("assessing pedestrian movement speed to avoid collision"), whereas GazeXplain misses this critical region.

## Highlights & Insights

- The paradigm extension from "Where" to "Where + What + Why" is the paper's most central conceptual contribution, elevating attention prediction from pixel-level regression to cognitive understanding.
- The W³DA dataset construction methodology (keyframe selection + MLLM-assisted annotation + human verification) provides a reusable pipeline for similar large-scale annotation tasks.
- The [ATTN] token design elegantly bridges the LLM's text space and visual space, enabling language-level reasoning to directly guide pixel-level prediction.
- Ablation experiments clearly demonstrate the mutual gains among all three dimensions, validating the necessity of the unified framework.

## Limitations & Future Work

- The current approach uses only single-frame images as input, without exploiting the temporal information inherent in video (driving scenes are naturally sequential).
- What/Why annotations in W³DA rely on the Qwen-VL-Max MLLM API, which may introduce systematic bias.
- The attention decoder is relatively simple (5 convolutional layers), potentially limiting the recovery of fine spatial details.
- Vicuna-7B is a relatively small LLM backbone; adopting a stronger LLM may further improve reasoning quality.
- Video-level temporal cognitive reasoning (e.g., tracking causal chains of attention shifts) remains unexplored.

## Related Work & Insights

- **Distinction from GazeXplain**: GazeXplain's explanations remain at the semantic level (describing gaze targets), whereas LLada goes further to the cognitive level (explaining the reasons for fixation).
- **Relationship to LISA (reasoning segmentation)**: Similarly employs special tokens from LLM outputs to control pixel-level prediction, but applied to attention prediction rather than semantic segmentation.
- **Implications for driver training**: An explainable attention model can inform novice drivers of "where to look, what to look at, and why to look there."
- **Implications for AV explainability**: Provides human-interpretable explanations for attention-based decisions in autonomous driving systems.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (Where+What+Why new paradigm + first explainable attention dataset)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (multi-dataset / multi-scenario / multi-metric / multi-baseline + cross-domain generalization)
- Writing Quality: ⭐⭐⭐⭐⭐ (clear motivation, excellent figures, comprehensive experiments)
- Value: ⭐⭐⭐⭐⭐ (opens a new direction for attention prediction; both the dataset and method carry long-term impact)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] DriverGaze360: OmniDirectional Driver Attention with Object-Level Guidance](../../CVPR2026/autonomous_driving/drivergaze360_omnidirectional_driver_attention_with_object-level_guidance.md)
- [\[ICCV 2025\] Where am I? Cross-View Geo-localization with Natural Language Descriptions](where_am_i_cross-view_geo-localization_with_natural_language_descriptions.md)
- [\[ICCV 2025\] ACAM-KD: Adaptive and Cooperative Attention Masking for Knowledge Distillation](acam_kd_adaptive_cooperative_attention_masking_knowledge_distillation.md)
- [\[ICCV 2025\] SRefiner: Soft-Braid Attention for Multi-Agent Trajectory Refinement](srefiner_soft-braid_attention_for_multi-agent_trajectory_refinement.md)
- [\[ICCV 2025\] DONUT: A Decoder-Only Model for Trajectory Prediction](donut_a_decoder-only_model_for_trajectory_prediction.md)

</div>

<!-- RELATED:END -->

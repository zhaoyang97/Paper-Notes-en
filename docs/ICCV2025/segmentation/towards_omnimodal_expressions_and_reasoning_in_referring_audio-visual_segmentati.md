---
title: >-
  [Paper Note] Towards Omnimodal Expressions and Reasoning in Referring Audio-Visual Segmentation
description: >-
  [ICCV 2025][Segmentation][Audio-visual segmentation] This paper proposes the OmniAVS dataset and OISA model, extending referring audio-visual segmentation (RAVS) beyond simple acoustic attribute perception to **omnimodal…
tags:
  - "ICCV 2025"
  - "Segmentation"
  - "Audio-visual segmentation"
  - "omnimodal referring"
  - "reasoning segmentation"
  - "multimodal large language models"
  - "query propagation"
date: 2026-05-08
content_hash: 069358894e6f0077
---

# Towards Omnimodal Expressions and Reasoning in Referring Audio-Visual Segmentation

**Conference**: ICCV 2025
**arXiv**: [2507.22886](https://arxiv.org/abs/2507.22886)
**Code**: [OmniAVS](https://henghuiding.com/OmniAVS/)
**Area**: Image Segmentation
**Keywords**: Audio-visual segmentation, omnimodal referring, reasoning segmentation, multimodal large language models, query propagation

## TL;DR

This paper proposes the OmniAVS dataset and OISA model, extending referring audio-visual segmentation (RAVS) beyond simple acoustic attribute perception to **omnimodal expressions (arbitrary combinations of text/speech/sound/image)** and **deep reasoning (understanding sound content + world knowledge)**, achieving SOTA on the new benchmark and multiple related tasks.

## Background & Motivation

Referring audio-visual segmentation (RAVS) is an emerging field that aims to segment target objects in audio-visual scenes based on referring expressions. The existing dataset Ref-AVS suffers from three major limitations:

**Shallow sound utilization**: Expressions only address surface-level acoustic attributes (e.g., "who makes the loudest sound"), without understanding sound content.

**Modality limitation**: Only text-based referring expressions are supported; speech, sound clips, and image inputs are absent.

**Lack of reasoning requirements**: Expressions require neither world knowledge nor complex reasoning.

Taking "who is most likely to be sick?" as an example, the model must establish a cognitive chain: sound → coughing → illness, which goes beyond simple acoustic feature recognition. Meanwhile, omnimodal AI systems such as ChatGPT-4o highlight the importance of processing arbitrary combinations of modality inputs.

Core motivation: To construct a RAVS benchmark and baseline model that genuinely understands sound content, supports omnimodal referring, and incorporates complex reasoning.

## Method

### Dataset: OmniAVS

**Video sources**: Creative Commons online videos + TVQA TV drama clips + self-recorded videos, with 2,104 videos curated from 10,871 candidates.

**8 expression types**:
- I. Text | II. Speech | III. Text + Sound | IV. Speech + Sound
- V. Text + Image | VI. Speech + Image | VII. Text + Sound + Image | VIII. Speech + Sound + Image

**Annotation rules**:
- Expressions must relate to sounds present in the video, not solely visual cues.
- Emphasis on sound **content** rather than sound **behavior** (e.g., "warning dog" rather than "barking dog").
- Expressions requiring reasoning are encouraged, and reasoning explanations are provided.
- Each expression may refer to zero or more targets.

**Dataset scale**: 2,104 videos, 103k frames, 4,277 targets, 206k masks, 61,095 expressions, 34,841 reasoning explanations.

### Model: OISA (Omnimodal Instructed Segmentation Assistant)

**Overall Architecture**: MLLM (audio encoder + visual encoder + LLM) + mask head (ViT-Adapter + pixel decoder + mask decoder)

- MLLM backbone: InternVL2-1B (InternViT-300M-448px + Qwen2-0.5B)
- Audio encoder: Whisper-large-v3 + audio MLP

### Key Design 1: Audio-Visual Interleaving (AVI)

$N$ frames are sampled from the video to obtain visual tokens $\{v_1, ..., v_N\}$; the encoded audio is split into $N$ segments $\{a_1, ..., a_N\}$ and interleaved in temporal order:

$$\{v_1, a_1, v_2, a_2, ..., v_N, a_N\}$$

Compared to VideoLLaMA's sequential concatenation $\{v_1,...,v_N, a_1,...,a_N\}$ or video-SALMONN's weighted fusion, the interleaving strategy achieves temporal alignment **without additional parameters**. Gains are particularly pronounced on the TVQA subset (which contains dense dialogue and requires precise audio-visual alignment).

The complete audio token $\mathbf{A}$ is further appended at the end of the interleaved sequence, analogous to InternVL2's thumbnail strategy, to supplement untruncated global audio information.

### Key Design 2: Query Propagation

The MLLM generates a `[SEG]` token as the target embedding, which is passed to the mask decoder.

VideoLISA's OTSA (One-Token-Seg-All) strategy uses the same `[SEG]` token to independently segment each frame. However, a single query carries a positional prior and struggles to adapt to target motion (e.g., moving from right to left), leading to ID switches.

Query propagation updates the query frame by frame:

- After each frame is segmented, the output query of the current frame is propagated to the next frame.
- The query is refined online, smoothly capturing the temporal motion trajectory.
- Contextual temporal information is effectively modeled.

$$\text{QP}: \quad q_{t+1} = \text{MaskDecoder}(q_t, F_t) \rightarrow q_{t+1}$$

### Training Pipeline

**Stage 1 — Audio-text alignment**: The audio encoder MLP is trained using ASR and Audio Caption datasets; all other parameters are frozen.

**Stage 2 — Omnimodal instructed segmentation fine-tuning**: Training is performed on a mixture of datasets (ADE20K, COCO-Stuff, RefCOCO series, MeViS, ReVOS, Ref-AVS, OmniAVS, etc.), with LoRA fine-tuning applied to the LLM and full training of mask head parameters. The loss comprises cross-entropy (text) + DICE + BCE (segmentation).

## Key Experimental Results

### OmniAVS Benchmark

| Method | Overall $\mathcal{J\&F}$ | I (Text) | VII (Text+Sound+Image) | VIII (Speech+Sound+Image) | METEOR |
|------|:---:|:---:|:---:|:---:|:---:|
| LMPM | 25.8 | 31.2 | - | - | - |
| MUTR | 32.3 | 35.4 | 41.6 | 40.5 | - |
| LISA-13B | 36.1 | 36.4 | 46.7 | 45.7 | 16.5 |
| **OISA-1B** | **41.1** | **40.1** | **52.6** | **53.0** | **21.7** |

With only 1B parameters, OISA-1B surpasses LISA-13B by 5.0%, with concurrent improvements in reasoning explanation quality (METEOR +5.2). Multimodal combination inputs (VII/VIII) yield the best results, demonstrating the complementarity of multiple modalities.

### Ref-AVS Benchmark

| Method | Seen $\mathcal{J}$ | Unseen $\mathcal{J}$ | Mix $\mathcal{J\&F}$ |
|------|:---:|:---:|:---:|
| EEMC | 34.2 | 49.5 | 41.9/58.1 |
| **OISA-1B** | **51.7** | **58.3** | **54.5/61.4** |

Improvements of +17.5 and +8.8 are achieved on the Seen and Unseen splits, respectively.

### Ablation Study

**Audio-visual fusion strategies**:

| Fusion Method | TVQA Subset | Overall |
|---------|:------:|:---:|
| Attention | 37.4 | 35.8 |
| Concatenation | 36.9 | 35.3 |
| **AVI + Concatenation** | **42.0** | **40.5** |

**Mask head design**:

| Query Type | Mask Head | $\mathcal{J\&F}$ | FPS |
|---------|-------|:---:|:---:|
| OTSA | SAM | 38.1 | 4.3 |
| OTSA | M2F | 35.2 | 15.7 |
| **QP** | **SAM** | **41.2** | 4.1 |
| **QP** | **M2F** | **40.5** | **12.3** |

Query propagation improves over OTSA by +5.3 $\mathcal{J\&F}$ on the M2F head while maintaining a 3× speed advantage.

### Key Findings

1. Audio-visual interleaving is the optimal solution for temporal alignment, with the most pronounced advantage on the TVQA subset (dense dialogue).
2. More modalities lead to better performance (splits VII/VIII achieve the highest scores), confirming that multiple modalities provide complementary information.
3. Query propagation substantially improves tracking quality for dynamic targets, resolving the ID-switching problem of OTSA.
4. OmniAVS is 17% more challenging than Ref-AVS (41.1 vs. 58.0), validating the dataset's difficulty.

## Highlights & Insights

- **Forward-looking dataset design**: 8 modality combinations + reasoning explanations + multi-target referring provide a fine-grained perception benchmark for omnimodal AI.
- **From "hearing" to "understanding"**: RAVS is advanced from acoustic attribute detection to sound content reasoning.
- **1B model outperforms 13B**: Demonstrates that task-specific design (AVI + QP) matters more than parameter count alone.

## Limitations & Future Work

- The base LLM is only 0.5B, limiting capability in scenarios requiring deep reasoning (e.g., ReasonSeg).
- Disentangling complex overlapping sounds remains a bottleneck (e.g., multiple simultaneous speakers with background noise).
- Speech expressions are synthesized via TTS, creating a distribution gap relative to real human speech.

## Related Work & Insights

- Audio-visual scene understanding: AVSBench, Ref-AVS, Music-AVQA, and related audio-visual joint learning works.
- Reasoning segmentation: LISA, VideoLISA, VISA, and other MLLM-based reasoning segmentation methods.
- Omnimodal models: ChatGPT-4o, VideoLLaMA, and other multimodal understanding systems.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The OmniAVS dataset defines an entirely new paradigm for omnimodal reasoning segmentation.
- **Technical Depth**: ⭐⭐⭐⭐ — AVI and query propagation are well-motivated and effective designs.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive validation across OmniAVS/Ref-AVS/RefCOCO/MeViS/ReVOS on multiple tasks.
- **Writing Quality**: ⭐⭐⭐⭐ — Dataset motivation and comparison with Ref-AVS are clearly argued.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Implicit Counterfactual Learning for Audio-Visual Segmentation](implicit_counterfactual_learning_for_audio-visual_segmentation.md)
- [\[ICCV 2025\] TAViS: Text-bridged Audio-Visual Segmentation with Foundation Models](tavis_text-bridged_audio-visual_segmentation_with_foundation_models.md)
- [\[ICCV 2025\] ReferDINO: Referring Video Object Segmentation with Visual Grounding Foundations](referdino_referring_video_object_segmentation_with_visual_grounding_foundations.md)
- [\[NeurIPS 2025\] UniPixel: Unified Object Referring and Segmentation for Pixel-Level Visual Reasoning](../../NeurIPS2025/segmentation/unipixel_unified_object_referring_and_segmentation_for_pixel-level_visual_reason.md)
- [\[ICCV 2025\] How Do Optical Flow and Textual Prompts Collaborate to Assist in Audio-Visual Semantic Segmentation?](how_do_optical_flow_and_textual_prompts_collaborate_to_assist_in_audio-visual_se.md)

</div>

<!-- RELATED:END -->

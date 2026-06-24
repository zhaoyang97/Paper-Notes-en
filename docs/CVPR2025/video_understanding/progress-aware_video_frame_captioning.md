---
title: >-
  [Paper Note] Progress-Aware Video Frame Captioning
description: >-
  [CVPR 2025][Video Understanding][Frame-level captioning] This paper introduces the new task of "progress-aware video frame captioning" and develops the ProgressCaptioner model. Through a two-stage training paradigm (frame pair $\rightarrow$ frame sequence) and an automated pseudo-label filtering mechanism, it generates fine-grained descriptions that precisely capture the frame-by-frame evolution of actions, significantly outperforming GPT-4o and Gemini-1.5-Pro on the self-con…
tags:
  - "CVPR 2025"
  - "Video Understanding"
  - "Frame-level captioning"
  - "action progress awareness"
  - "temporal fine-grainedness"
  - "vision-language models"
  - "preference learning"
date: 2026-05-08
content_hash: 805b6f13d7e16d5d
---

# Progress-Aware Video Frame Captioning

**Conference**: CVPR 2025  
**arXiv**: [2412.02071](https://arxiv.org/abs/2412.02071)  
**Code**: [https://vision.cs.utexas.edu/projects/ProgressCaptioner](https://vision.cs.utexas.edu/projects/ProgressCaptioner)  
**Area**: Video Understanding  
**Keywords**: Frame-level captioning, action progress awareness, temporal fine-grainedness, vision-language models, preference learning

## TL;DR

This paper introduces the new task of "progress-aware video frame captioning" and develops the ProgressCaptioner model. Through a two-stage training paradigm (frame pair $\rightarrow$ frame sequence) and an automated pseudo-label filtering mechanism, it generates fine-grained descriptions that precisely capture the frame-by-frame evolution of actions, significantly outperforming GPT-4o and Gemini-1.5-Pro on the self-constructed FrameCapEval benchmark.

## Background & Motivation

**Background**: Visual captioning tasks are broadly categorized into image captioning (generating an isolated description for each image) and video captioning (producing an overall description for a video). Image captioning lacks temporal context, resulting in virtually identical descriptions for adjacent frames. On the other hand, video captioning only provides coarse-grained event summaries (e.g., "scrambling eggs"), completely overlooking the progressive details of actions.

**Limitations of Prior Work**: (1) Existing top-tier VLMs (GPT-4o, Gemini) suffer from two serious issues in frame-level captioning: "insufficient temporal granularity" (inability to distinguish subtle differences between adjacent frames) and "temporal hallucination" (captions implying progress that does not actually exist visually). (2) Image captioning models process frames individually without integration of temporal context, failing to express "what is changing." (3) There is a lack of training data and evaluation benchmarks dedicated to frame-level captioning.

**Key Challenge**: Generating precise frame-level captions requires simultaneously satisfying three conflicting requirements: (a) each frame description must accurately reflect that frame's content (no hallucinations), (b) each frame description must be distinct from other frames (temporal specificity), and (c) the entire sequence of descriptions must coherently reflect the overall progress of the action.

**Goal**: To define and solve the "progress-aware frame-level captioning" task, building a dedicated model and establishing an evaluation framework.

**Key Insight**: The authors observe that feeding the entire sequence of frames directly to a VLM leads to overly simplified descriptions with temporal misalignment, whereas feeding only a single frame loses temporal context entirely. A frame pair (two frames) serves as an excellent compromise—providing temporal contrastive relations without causing the model's output to degenerate.

**Core Idea**: Frame-pair captioning is adopted as a cornerstone, which progressively extends to full-sequence captioning via a two-stage training paradigm. Automated "progress detection" and "caption matching" tasks are designed to filter high-quality pseudo-labels and construct preference learning data.

## Method

### Overall Architecture

ProgressCaptioner is trained in two stages. In Stage I, a captioning model is trained on frame pairs $(v_1, v_2)$. Multiple VLMs generate candidate caption pairs, which are automatically filtered via progress detection and caption matching. High-quality captions are used for SFT, and lower-quality ones are utilized for DPO. In Stage II, the Stage I model generates pseudo-labels for full frame sequences using a sliding window. These labels are similarly filtered and then used for SFT + DPO, ultimately yielding a complete model capable of accepting inputs ranging from 2 to T frames.

### Key Designs

1. **Automated Pseudo-Label Quality Assessment**:

    - **Function**: Automatically distinguish between high-quality and low-quality frame-level captions.
    - **Mechanism**: Two evaluation tasks are designed: (1) **Progress Detection**: using an LLM to judge whether a caption pair implies visible physical changes. Voting across multiple models and caption pairs forms a consensus label—captions aligning with the consensus pass, while others are marked as failures (capturing temporal hallucination). (2) **Caption Matching**: formulating a multiple-choice question where a VLM matches captions to their corresponding frames (including an "uncertain" option); correct matches are labeled high-quality (capturing insufficient temporal granularity, as overly similar captions for two frames cannot be correctly matched).
    - **Design Motivation**: Captions generated by VLMs suffer from systemic issues (temporal hallucination and insufficient granularity) and cannot be directly used for training. These automated evaluation tasks replace expensive human annotations, making data construction scalable.

2. **Two-Stage Progressive Training**:

    - **Function**: Progressively scale from frame pairs to frame sequences of arbitrary length.
    - **Mechanism**: Stage I trains on frame pairs—$K$ VLMs generate candidate caption pairs, which are evaluated to yield positive samples $\hat{\mathbf{c}}^+$ and negative samples $\hat{\mathbf{c}}^-$, followed by SFT then DPO. Stage II uses the Stage I model to label full sequences via a two-frame sliding window, identifies $M$ visually changing keyframes through progress detection, conducts matching evaluations on these $M$ frame captions, and performs SFT + DPO again.
    - **Design Motivation**: Expermints show that VLM captions degrade severely under full-frame inputs; frame pairs offer the highest quality input granularity. The two-stage design progressively enhances pseudo-label quality—the Stage I model outperforms the raw VLMs, thereby producing higher quality Stage II pseudo-labels.

3. **SFT + DPO Joint Training**:

    - **Function**: Simultaneously learning good captioning patterns and avoiding hallucinations.
    - **Mechanism**: Initialized from LLAVA-OV-7B, the model first undergoes SFT using high-quality captions to learn the task format, followed by DPO using positive/negative sample pairs produced by the automated evaluation. This guides the model to prefer accurate, fine-grained captions while avoiding hallucinated ones.
    - **Design Motivation**: SFT alone cannot effectively mitigate the inherent temporal hallucinations of VLMs. The preference data for DPO is entirely constructed via the automated evaluation tasks, requiring no human annotation.

### Loss & Training

The SFT stage utilizes the standard instruction finetuning loss (autoregressive next-token prediction). The DPO stage utilizes the standard Direct Preference Optimization loss, where positive samples are captions that passed both evaluations, and negative samples are failed captions. The training data is derived from YouTube videos in the HowToChange and COIN datasets.

## Key Experimental Results

### Main Results

| Model | Size | Cap Match | Prog Detect |
|---|---|---|---|
| GPT-4o | - | 32.4 | 64.2 |
| Gemini-1.5-Pro | - | 31.4 | 63.8 |
| Qwen2-VL | 7B | 13.7 | 69.6 |
| LLAVA-OV | 7B | 7.8 | 59.0 |
| **ProgressCaptioner** | **7B** | **37.3** | **73.6** |

On the HowToChange dataset, the 7B ProgressCaptioner outpaces GPT-4o and Gemini-1.5-Pro in both caption matching and progress detection.

### Ablation Study

| Configuration | Cap Match | Prog Detect |
|---|---|---|
| Pseudo-label Ensemble Only | 18.6 | 62.5 |
| Stage I (SFT) | - | - |
| Stage I + II (SFT + DPO) | **37.3** | **73.6** |

Moving from the pseudo-label ensemble baseline to full two-stage training, caption matching increases from 18.6 to 37.3 (a 2x increase), and progress detection improves from 62.5 to 73.6.

### Key Findings

- ProgressCaptioner wins in user studies with the highest selection rate of 31.6%, which is 2 to 3.6 times higher than the best models of comparable parameter sizes.
- The model performs exceptionally well even on unseen datasets (Penn Action, Kinetics), demonstrating strong generalization capabilities.
- Frame-level captions can be used for keyframe selection, thereby aiding action recognition—showing a +1.7% improvement over uniform sampling on Kinetics.
- When applied to video QA (NExT-QA ATP-Hard), it outperforms VideoAgent by +3.4%.

## Highlights & Insights

1. **Proposing the concept of "temporal hallucination"**: Precisely defining the core issue of VLMs in frame-level captioning, where descriptions imply progress that is not visually present.
2. **Clever design of automated evaluation tasks**: Progress detection and caption matching are used not only for filtering data but also directly as evaluation metrics.
3. **Outperforming larger models with a smaller one**: The 7B model beats GPT-4o and Gemini-1.5-Pro, proving the value of task-specific training.
4. **Abundant downstream applications**: Performance gains in keyframe selection, action recognition, and video QA demonstrate the broad value of frame-level captions.

## Limitations & Future Work

- It relies on an ensemble of multiple VLMs to generate pseudo-labels, which entails high computational costs.
- The training data originates from HowToChange and COIN, biasing towards everyday activities and object state changes, with insufficient coverage of more abstract actions.
- The current sliding window approach for long sequences may lose global context.
- Future work could explore end-to-end training rather than relying on pseudo-labels, and expand to more video domains.

## Related Work & Insights

- Relation to image difference captioning: Works like SPOT-the-Diff handle differences in static image pairs, whereas this work extends to the temporal dimension in videos.
- Relation to dense video captioning: Dense video captioning focuses on "what events occurred," while this work focuses on "how events evolve frame-by-frame."
- The OSCaR benchmark is limited to 3 frames and object state changes, whereas this study offers a much broader scope.
- Insights: The fact that VLMs "know too much" becomes an obstacle—over-reliance on common statistical priors of actions causes temporal hallucination.

## Rating

- **Novelty**: 8/10 — The definition of a new task, the concept of temporal hallucination, and the design of automated evaluation tasks all show originality.
- **Experimental Thoroughness**: 9/10 — Benchmark construction, multi-model comparisons, user studies, downstream applications, and detailed ablation.
- **Writing Quality**: 9/10 — The discussion of problem motivation is highly clear and develops step-by-step.
- **Value**: 8/10 — The ability for frame-level captioning drives progress across multiple sub-fields of video understanding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Q-Frame: Query-aware Frame Selection and Multi-Resolution Adaptation for Video-LLMs](../../ICCV2025/video_understanding/q-frame_query-aware_frame_selection_and_multi-resolution_adaptation_for_video-ll.md)
- [\[CVPR 2025\] M-LLM Based Video Frame Selection for Efficient Video Understanding](m-llm_based_video_frame_selection_for_efficient_video_understanding.md)
- [\[CVPR 2025\] HierarQ: Task-Aware Hierarchical Q-Former for Enhanced Video Understanding](hierarq_task-aware_hierarchical_q-former_for_enhanced_video_understanding.md)
- [\[CVPR 2025\] EgoTextVQA: Towards Egocentric Scene-Text Aware Video Question Answering](egotextvqa_towards_egocentric_scene-text_aware_video_question_answering.md)
- [\[AAAI 2026\] Explicit Temporal-Semantic Modeling for Dense Video Captioning via Context-Aware Cross-Modal Interaction](../../AAAI2026/video_understanding/explicit_temporal-semantic_modeling_for_dense_video_captioning_via_context-aware.md)

</div>

<!-- RELATED:END -->

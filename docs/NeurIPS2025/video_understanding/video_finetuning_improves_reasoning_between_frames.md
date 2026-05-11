---
title: >-
  [Paper Note] Video Finetuning Improves Reasoning Between Frames
description: >-
  [NeurIPS 2025][Video Understanding][video finetuning] This paper proposes a visual chain-of-thought (vCoT) approach to systematically compare image LLMs and video-finetuned LLMs on inter-frame reasoning. It finds that vi…
tags:
  - "NeurIPS 2025"
  - "Video Understanding"
  - "video finetuning"
  - "multimodal large language models"
  - "inter-frame reasoning"
  - "visual chain-of-thought"
  - "temporal understanding"
date: 2026-05-08
content_hash: 58a767fb02c8224e
---

# Video Finetuning Improves Reasoning Between Frames

**Conference**: NeurIPS 2025
**arXiv**: [2511.12868](https://arxiv.org/abs/2511.12868)
**Code**: None
**Area**: Video Understanding
**Keywords**: video finetuning, multimodal large language models, inter-frame reasoning, visual chain-of-thought, temporal understanding

## TL;DR

This paper proposes a visual chain-of-thought (vCoT) approach to systematically compare image LLMs and video-finetuned LLMs on inter-frame reasoning. It finds that video finetuning enables models to implicitly learn inter-frame transition reasoning, and that this capability transfers to relational reasoning tasks on static images.

## Background & Motivation

Multimodal LLMs have achieved remarkable progress in visual understanding, yet most approaches that scale from images to video still rely on naive frame-level token concatenation, lacking genuine temporal understanding. This leads to poor performance on tasks that require reasoning about implicit transitions between frames, where models tend to rely on superficial visual cues.

Video LLMs incorporate additional inductive biases such as video data finetuning and temporal positional encodings (e.g., RoPE) to enhance video comprehension. However, a central question has remained unanswered: **what does video finetuning actually confer upon a model, and to what degree does it enhance reasoning beyond the capabilities of image models?**

The paper's core starting point is as follows: if video finetuning genuinely teaches a model inter-frame reasoning, then explicitly providing inter-frame transition descriptions (vCoT) should yield minimal gain for video models (which have already learned this implicitly) but substantial gain for image models (which lack this ability). Based on this hypothesis, the authors design vCoT to empirically validate the mechanism.

## Method

### Overall Architecture

The research framework is a controlled comparative experimental design. Matched pairs of image and video LLMs sharing identical architectures (e.g., LLaVA-NeXT vs. LLaVA-NeXT-Video) are selected, differing only in whether video finetuning was applied. By comparing performance changes with and without vCoT across both model types, the study reveals the actual effect of video finetuning.

### Key Designs

1. **Visual Chain-of-Thought (vCoT) Generation**: vCoT explicitly generates textual transition descriptions between adjacent frames in two steps:

   - **Step 1 — Common Visual Attribute Identification**: The model is shown two frames and prompted to identify shared elements (e.g., objects, background, spatial configuration), establishing stable cross-frame context.
   - **Step 2 — Bridging Event Inference**: Given the two frames and the identified common elements, the model is prompted to infer intermediate events that may have occurred between them (e.g., "the person kicks the ball toward the house"). A Qwen-2.5 model is used to condense the descriptions for brevity.

2. **Modality Shuffling Experiments**: To disentangle the model's reliance on visual versus textual cues, two types of perturbations are designed:

   - **Visual Perturbation**: Each video frame is replaced with a frame from an unrelated video, while the text infill remains unchanged.
   - **Text Perturbation**: The original frames are retained, but the text infill is replaced with that from another video.

   Observing each model type's sensitivity to these perturbations reveals the difference in modality dependence between video and image models.

3. **Transfer to Static Image Reasoning**: The i-RAVEN benchmark is used to test whether video models can transfer inter-frame reasoning capabilities to non-temporal relational reasoning tasks. RAVEN is an abstract visual reasoning task (analogous to progressive matrices in IQ tests) that requires inferring abstract rules from a set of panels and selecting the correct completion.

### Loss & Training

This paper presents an analytical study rather than a new training methodology. The model pairs used include:
- LLaVA-NeXT (image) vs. LLaVA-NeXT-Video (video)
- InternVL-Image vs. InternVL-Video

All model pairs share the same visual encoder, language backbone, and cross-modal projector.

## Key Experimental Results

### Main Results: vCoT Effect on EgoSchema

| Model | Frames | Baseline Acc. | +vCoT Acc. | Gain |
|-------|--------|--------------|------------|------|
| LLaVA-NeXT (image) | 5 | 44.0% | 51.4% | **+7.4%** |
| LLaVA-NeXT-Video (video) | 5 | 47.0% | 48.6% | +1.6% |
| LLaVA-NeXT (image) | 10 | 49.2% | 55.4% | **+6.2%** |
| LLaVA-NeXT-Video (video) | 10 | 49.0% | 51.4% | +2.4% |
| InternVL-Image | 5 | 38.4% | 40.4% | +2.0% |
| InternVL-Video | 5 | 44.6% | 42.4% | **-2.2%** |
| InternVL-Image | 10 | 37.4% | 42.6% | **+5.2%** |
| InternVL-Video | 10 | 45.8% | 49.0% | +3.2% |

### Ablation Study: Modality Shuffling

| Model | Frames | vCoT Baseline | Visual Perturb. | Text Perturb. |
|-------|--------|--------------|-----------------|---------------|
| LLaVA-NeXT (image) | 5 | 51.4% | 39.8% (-11.6) | 42.0% (-9.4) |
| LLaVA-NeXT-Video (video) | 5 | 48.6% | 41.6% (-7.0) | 47.0% (-1.6) |
| LLaVA-NeXT (image) | 10 | 55.4% | 51.8% (-3.6) | 45.0% (-10.4) |
| LLaVA-NeXT-Video (video) | 10 | 51.4% | 46.4% (-5.0) | 45.4% (-6.0) |

### i-RAVEN Static Reasoning Transfer Results

| Model | center | dist_4 | dist_9 | in/out | indist4/out | L/R | U/D | Avg. |
|-------|--------|--------|--------|--------|-------------|-----|-----|------|
| InternVL-Image | 14.8 | 14.4 | 15.2 | 11.6 | 13.2 | 15.2 | 14.4 | 14.1 |
| InternVL-Video | 15.6 | 16.0 | 15.8 | 13.8 | **17.0** | 14.0 | 14.2 | **15.2** |
| LLaVA-Image | 7.0 | 8.0 | 15.0 | 7.0 | 9.0 | 12.0 | 14.0 | 10.3 |
| LLaVA-Video | 7.0 | 14.0 | 16.0 | 8.0 | **13.0** | 14.0 | **21.0** | **13.3** |

### Key Findings

- vCoT yields substantial gains for image models (up to +7.4%) but marginal or even negative gains for video models, indicating that video finetuning enables implicit inter-frame transition reasoning.
- Video models exhibit significantly greater robustness to textual noise than image models, suggesting video models rely more heavily on visual information.
- Video models outperform image models on i-RAVEN, demonstrating that inter-frame reasoning ability transfers to static relational reasoning.

## Highlights & Insights

- The paper proposes an elegant experimental design to "probe" the intrinsic effects of video finetuning: the marginal benefit of explicit inter-frame descriptions serves as an indicator of whether a model has already learned such reasoning implicitly.
- vCoT is also a practically useful augmentation for image models — in scenarios where video models are unavailable, vCoT can compensate for the lack of inter-frame reasoning capability.
- Video finetuning is shown to confer transferable reasoning abilities, generalizing from dynamic video to relational reasoning over static images.
- The modality shuffling experiments reveal fundamental differences in modality dependence between video and image models.

## Limitations & Future Work

- Experiments are conducted solely on EgoSchema (egocentric video), without validation on more diverse video understanding benchmarks.
- The models evaluated are relatively small (7B scale); it remains unclear whether similar findings hold for larger models.
- vCoT generation itself relies on a capable VLM, introducing additional computational overhead.
- Absolute accuracy on i-RAVEN remains low (all below 21%), indicating substantial room for improvement in abstract reasoning for current VLMs.
- The effect of different video finetuning strategies (e.g., varying data scale or task types) on inter-frame reasoning ability is not explored.

## Related Work & Insights

- This work follows in the tradition of chain-of-thought reasoning, extending the concept from pure text-based reasoning to visual temporal reasoning.
- Unlike works such as Video-ChatGPT, this paper focuses not on improving performance but on *understanding* why video finetuning is effective.
- The paper suggests that vCoT gain can serve as a probe to measure a model's implicit temporal reasoning capability during evaluation.
- The findings carry important implications for multimodal model design: video finetuning not only improves video task performance but also enhances general relational reasoning ability.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Elegant experimental design; using vCoT as a probing tool is a novel contribution.
- **Experimental Thoroughness**: ⭐⭐⭐ — Limited benchmark and model coverage, though controlled variable design is rigorous.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear logic and well-structured narrative.
- **Value**: ⭐⭐⭐⭐ — Valuable reference for understanding the mechanisms behind video finetuning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Listening Between the Frames: Bridging Temporal Gaps in Large Audio-Language Models](../../AAAI2026/video_understanding/listening_between_the_frames_bridging_temporal_gaps_in_large_audio-language_mode.md)
- [\[NeurIPS 2025\] When Thinking Drifts: Evidential Grounding for Robust Video Reasoning](when_thinking_drifts_evidential_grounding_for_robust_video_reasoning.md)
- [\[NeurIPS 2025\] Tool-Augmented Spatiotemporal Reasoning for Streamlining Video Question Answering Task](toolaugmented_spatiotemporal_reasoning_for_streamlining_vide.md)
- [\[NeurIPS 2025\] VGEnt: Graph-Based Retrieval-Reasoning-Augmented Generation for Long Video Understanding](vgent_graph-based_retrieval-reasoning-augmented_generation_for_long_video_unders.md)
- [\[NeurIPS 2025\] The Ouroboros of Benchmarking: Reasoning Evaluation in an Era of Saturation](the_ouroboros_of_benchmarking_reasoning_evaluation_in_an_era_of_saturation.md)

</div>

<!-- RELATED:END -->

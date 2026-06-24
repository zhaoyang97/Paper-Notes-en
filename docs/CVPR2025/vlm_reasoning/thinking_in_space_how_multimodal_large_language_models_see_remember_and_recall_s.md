---
title: >-
  [Paper Note] Thinking in Space: How Multimodal Large Language Models See, Remember, and Recall Spaces
description: >-
  [CVPR 2025][VLM Reasoning][Visual Spatial Intelligence] This paper proposes VSI-Bench, a video-based visual spatial intelligence benchmark (5000+ QA pairs), to systematically evaluate the spatial reasoning capabilities of MLLMs. The study finds that spatial reasoning is the primary bottleneck, and traditional language reasoning techniques (such as CoT) fail to improve performance, whereas explicitly generating cognitive maps can enhance spatial distance reasoning.
tags:
  - "CVPR 2025"
  - "VLM Reasoning"
  - "Visual Spatial Intelligence"
  - "Cognitive Map"
  - "Video Understanding"
  - "Spatial Reasoning"
  - "Benchmark"
date: 2026-05-08
content_hash: e2d1c6b695ad7a72
---

# Thinking in Space: How Multimodal Large Language Models See, Remember, and Recall Spaces

**Conference**: CVPR 2025  
**arXiv**: [2412.14171](https://arxiv.org/abs/2412.14171)  
**Code**: [https://github.com/vision-x-nyu/thinking-in-space](https://github.com/vision-x-nyu/thinking-in-space)  
**Area**: Multimodal VLM  
**Keywords**: Visual Spatial Intelligence, Cognitive Map, Video Understanding, Spatial Reasoning, Benchmark

## TL;DR
This paper proposes VSI-Bench, a video-based visual spatial intelligence benchmark (5000+ QA pairs), to systematically evaluate the spatial reasoning capabilities of MLLMs. The study finds that spatial reasoning is the primary bottleneck, and traditional language reasoning techniques (such as CoT) fail to improve performance, whereas explicitly generating cognitive maps can enhance spatial distance reasoning.

## Background & Motivation
- **Background**: MLLMs have achieved significant progress in language and visual understanding, but visual spatial intelligence (perceiving space, remembering layout, and recalling spatial information) remains under-explored.
- **Limitations of Prior Work**: Existing video understanding benchmarks primarily focus on content-level understanding (recognition, perception), lacking evaluation of 3D spatial reasoning; most existing spatial-related works are based on 2D images or text-only inputs.
- **Key Challenge**: Humans naturally build mental spatial models from video sequences, but whether MLLMs can "think in space" from videos remains unclear.
- **Goal**: Constructing a systematic visual spatial intelligence evaluation framework to deeply analyze the strengths and bottlenecks of MLLMs in spatial reasoning.
- **Key Insight**: Departing from the dual-coding theory of cognitive psychology to probe the spatial thinking of models from both linguistic (self-explanation) and visual (cognitive map) dimensions.
- **Core Idea**: Spatial reasoning is the main bottleneck for MLLMs, and explicitly constructing cognitive maps can enhance distance reasoning capabilities.

## Method

### Overall Architecture
VSI-Bench is a video-level spatial intelligence benchmark containing 288 real indoor scene videos and over 5000 QA pairs. The data originates from three 3D reconstruction datasets: ScanNet, ScanNet++, and ARKitScenes. The benchmark comprises 8 tasks categorized into 3 major types: configuration (object counting, relative distance, relative orientation, route planning), measurement estimation (object size, room size, absolute distance), and spatiotemporal (order of appearance). The evaluation framework includes multiple-change accuracy and Mean Relative Accuracy (MRA) for numerical prediction.

### Key Designs
1. **VSI-Bench Benchmark Construction**:
    - **Function**: Systematically evaluate the visual spatial intelligence of MLLMs.
    - **Mechanism**: Unifying multiple 3D datasets into a standardized format, generating QA pairs through a combination of template-based automatic generation and human annotation, and iteratively ensuring quality via human review.
    - **Design Motivation**: Video data is closer to how humans observe space compared to static images, and 3D datasets provide precise object-level annotations as ground truth.

2. **Mean Relative Accuracy (MRA) Evaluation Metric**:
    - **Function**: Evaluate the accuracy of numerical prediction tasks (distance/size estimation).
    - **Mechanism**: Compute and average the relative accuracy across multiple confidence thresholds $\mathcal{C}=\{0.5, 0.55, \ldots, 0.95\}$: 
    $$\mathcal{MRA}=\frac{1}{10}\sum_{\theta\in\mathcal{C}}\mathbb{1}(\frac{|\hat{y}-y|}{y}<1-\theta)$$
    - **Design Motivation**: Traditional exact matching fails to measure the closeness of numerical predictions. MRA comprehensively evaluates prediction quality across multiple granularities.

3. **Cognitive Map Probing and Enhancement**:
    - **Function**: Visually probe the internal spatial representations of MLLMs and leverage them to enhance spatial reasoning.
    - **Mechanism**: Prompting MLLMs to predict object center locations on a $10 \times 10$ grid to generate cognitive maps, revealing that models possess strong local spatial perception but severely degraded global perception; using cognitive map generation as a pre-step for QA improves relative distance accuracy.
    - **Design Motivation**: Inspired by cognitive psychology where humans perform spatial reasoning through mental imagery, exploring whether MLLMs can improve spatial reasoning through a similar mechanism.

### Loss & Training
This work is an evaluation study and does not involve training new models. All model evaluations are conducted in a zero-shot setting, using greedy decoding to ensure reproducibility. The evaluation covers 3 closed-source models (GPT-4o, Gemini-1.5 Flash/Pro) and 12 open-source models (InternVL2, VILA, LongVA, LLaVA series, etc.), with parameter scales ranging from 0.5B to 72B.

## Key Experimental Results

### Main Results

| Model | Average | Object Counting | Absolute Distance | Relative Distance | Relative Orientation | Route Planning | Order of Appearance |
|------|--------|---------|---------|---------|---------|---------|---------|
| Human Level | 79.2 | 94.3 | 47.0 | 94.7 | 95.8 | 95.8 | 100.0 |
| Gemini-1.5 Pro | 45.4 | 56.2 | 43.6 | 51.3 | 46.3 | 36.0 | 34.6 |
| LLaVA-Video-72B | 40.9 | 48.9 | 35.3 | 42.4 | 36.7 | 35.0 | 48.6 |
| GPT-4o | 34.0 | 46.2 | 38.2 | 37.0 | 41.3 | 31.5 | 28.5 |

### Ablation Study

| Method | Average Change | Description |
|------|---------|------|
| Zero-Shot CoT | -4% | Encouraging thinking actually degrades performance |
| Self-Consistency | -1.1% | Multiple-sample voting does not help |
| Tree-of-Thoughts | -4% | Planning-and-reasoning paradigm also fails |
| Cognitive Map Enhancement | +10% (Relative Distance) | Explicitly generating maps significantly improves distance reasoning |

### Key Findings
- Humans far outperform MLLMs in configuration and spatiotemporal tasks (94%-100%), but the gap is smaller in measurement estimation tasks, indicating that MLLMs have a relative advantage in quantitative estimation.
- Approximately 71% of errors stem from spatial reasoning (relational reasoning + egocentric-to-allocentric transformation), whereas visual perception and language intelligence are not the primary bottlenecks.
- The cognitive maps of MLLMs exhibit strong local perception (64% accuracy for adjacent objects) but severely degraded global perception.
- Using GT cognitive maps improves relative distance accuracy from 46% to 66%-78%.
- Open-source 72B models (LLaVA-Video/OneVision) are close to closed-source models (with only a 4-5% gap), but most small models perform below the random baseline.
- On VideoMME, CoT yields a 1.6% improvement, whereas it leads to a decline on VSI-Bench, indicating that spatial tasks are fundamentally different from general video understanding.

## Highlights & Insights
- First to systematically model and evaluate the visual spatial intelligence of MLLMs from a cognitive psychology perspective, providing a comprehensive capability taxonomy (across four dimensions: visual perception, language intelligence, temporal processing, and spatial reasoning).
- Discovered the counter-intuitive finding that traditional CoT methods completely fail in spatial reasoning, indicating that spatial intelligence requires enhancement strategies different from linguistic reasoning.
- Quantitatively revealed via self-explanation analysis that 71% of errors originate from spatial reasoning rather than visual perception or language comprehension, clarifying the direction for future research.
- Cognitive maps, as concrete tools for "thinking in space", offer a new pathway to enhance MLLMs' spatial reasoning—achieving 64% accuracy in adjacent object localization proves that models possess emergent local spatial awareness.
- The MRA metric is elegantly designed, resolving the issue where exact matching is too stringent for numerical prediction evaluation.
- Relative distance task performance jumps from 46% to 78% when using GT cognitive maps, demonstrating that an accurate spatial world model is a key component for solving these problems.
(Residences, offices, factories); spatial reasoning in outdoor and large-scale environments is not covered.
- Cognitive maps are currently limited to a coarse $10\times10$ grid; finer spatial representations may bring greater improvements (a $20\times20$ grid actually decreases performance on MLLM-predicted maps, but improves to 78% under GT maps).
- The effect of introducing spatial reasoning data or self-supervised spatial objectives during training has not been explored, which remains a promising future direction.
- The impact of video frame sampling strategies on spatial understanding has not been systematically studied.
- Only English prompts were tested; differences in conveying spatial relations across different languages were not considered of coarse grid, finer spatial representations may bring greater improvements.
- The effect of introducing spatial reasoning data or self-supervised spatial objectives during training has not been explored.
- The impact of video frame sampling strategies on spatial understanding has not been systematically studied.

## Related Work & Insights
- **vs SpatialVLM**: SpatialVLM endows spatial capabilities through image fine-tuning; this paper finds that fine-tuning alone may be insufficient, requiring more fundamental spatial representation enhancements (such as cognitive maps).
- **vs Video-MME**: Video-MME focuses on content-level video understanding (recognition, narration); this work extends to 3D spatial-level understanding, serving as a complementary relation.
- **vs EgoSchema**: EgoSchema evaluates egocentric video understanding capabilities; this work further emphasizes the capability of egocentric-to-allocentric transformation.
- **vs OpenEQA**: OpenEQA also evaluates spatial understanding using egocentric videos, but this work provides a more systematic capability taxonomy and deeper error analysis.
- **vs SpatialBot**: SpatialBot focuses on spatial perception in 2D images; this work extends to video-level spatial memory and reasoning.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First systematic video spatial intelligence benchmark, with a novel cognitive map probing approach.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation of 15 models, with in-depth error analysis and ablation studies.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear narrative structure, progressing logically from evaluation to analysis to improvement.
- Value: ⭐⭐⭐⭐⭐ Reveals spatial reasoning as a core bottleneck for MLLMs, offering important guidance for embodied AI and navigation tasks.

---

> This note is generated based on a full reading of the paper, covering all sections of Method, Experiments, and Analysis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Thinking in Dynamics: How Multimodal Large Language Models Perceive, Track, and Reason Dynamics in Physical 4D World](thinking_in_dynamics_how_multimodal_large_language_models_perceive_track_and_rea.md)
- [\[ICLR 2026\] MIMIC-Bench: Exploring the User-Like Thinking and Mimicking Capabilities of Multimodal Large Language Models](../../ICLR2026/vlm_reasoning/mimic-bench_exploring_the_user-like_thinking_and_mimicking_capabilities_of_multi.md)
- [\[CVPR 2025\] Reasoning over Video: Evaluating How MLLMs Extract, Integrate, and Reconstruct Spatiotemporal Evidence](reasoning_over_video_evaluating_how_mllms_extract_integrate_and_reconstruct_spat.md)
- [\[CVPR 2025\] Insight-V: Exploring Long-Chain Visual Reasoning with Multimodal Large Language Models](insight-v_exploring_long-chain_visual_reasoning_with_multimodal_large_language_m.md)
- [\[CVPR 2025\] SeqAfford: Sequential 3D Affordance Reasoning via Multimodal Large Language Model](seqafford_sequential_3d_affordance_reasoning_via_multimodal_large_language_model.md)

</div>

<!-- RELATED:END -->

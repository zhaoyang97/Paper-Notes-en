---
title: >-
  [Paper Note] VEU-Bench: Towards Comprehensive Understanding of Video Editing
description: >-
  [CVPR 2025][Video Generation][Video Editing Understanding] Proposes VEU-Bench, the first benchmark to comprehensively evaluate Video-LLMs' understanding of video editing elements, spanning 10 editing dimensions and 3 evaluation levels (Recognition/Reasoning/Judgment) across 19 fine-grained tasks. Additionally, trains an expert model, Oscars, which outperforms the open-source SOTA by 28.3%.
tags:
  - "CVPR 2025"
  - "Video Generation"
  - "Video Editing Understanding"
  - "Video-LLM Benchmarking"
  - "Edit Understanding"
  - "Shot Analysis"
  - "Abstract Reasoning"
date: 2026-05-08
content_hash: cf3b0f537270c0cb
---

# VEU-Bench: Towards Comprehensive Understanding of Video Editing

**Conference**: CVPR 2025  
**arXiv**: [2504.17828](https://arxiv.org/abs/2504.17828)  
**Code**: Project page  
**Area**: Video Understanding / Video Editing  
**Keywords**: Video Editing Understanding, Video-LLM Benchmarking, Edit Understanding, Shot Analysis, Abstract Reasoning

## TL;DR

Proposes VEU-Bench, the first benchmark to comprehensively evaluate Video-LLMs' understanding of video editing elements, spanning 10 editing dimensions and 3 evaluation levels (Recognition/Reasoning/Judgment) across 19 fine-grained tasks. Additionally, trains an expert model, Oscars, which outperforms the open-source SOTA by 28.3%.

## Background & Motivation

**Background**: Most videos widely shared on the Internet are professionally edited. Video editing involves multiple dimensions, such as shot composition (e.g., shot size, character positioning), camera movement (e.g., pan, tilt, zoom), cut types (e.g., match cut, jump cut), and transition effects. Video-LLMs (such as Qwen2-VL, LLaVA-Video, etc.) have made significant progress in general video understanding tasks.

**Limitations of Prior Work**: Three core problems exist: (1) Existing VEU benchmarks (e.g., AVE, MovieCuts, AutoTransition) primarily focus on the classification of editing elements, lacking evaluations at the reasoning and judgment levels—for instance, not only identifying "this is a match cut," but also explaining why it was used and its narrative effect. (2) The editing dimensions covered by existing benchmarks are incomplete, typically focusing only on a subset of camera settings or cut types. (3) There is a lack of systematic evaluation of Video-LLMs on VEU tasks.

**Key Challenge**: Video editing elements are abstract concepts (e.g., a "match cut" requires understanding shape or motion alignment patterns across scenes) rather than directly observable objects or actions in the physical world. This abstract nature makes VEU a natural playground for evaluating the abstract reasoning capabilities of models, but current Video-LLMs fall severely short in this area.

**Goal**: (1) Construct a benchmark that comprehensively covers various dimensions and levels of video editing; (2) systematically evaluate the VEU capabilities of current SOTA Video-LLMs; (3) validate the value of VEU data for enhancing general video understanding.

**Key Insight**: Categorize video editing elements into three granularities (intra-frame, inter-frame, and inter-shot), evaluate progressively from recognition $\rightarrow$ reasoning $\rightarrow$ judgment in each dimension, and utilize an ontology knowledge base to achieve high-quality automated annotation.

**Core Idea**: Construct an ontology-knowledge-base-driven annotation pipeline to expand VEU from simple classification to reasoning and judgment levels. Prove that VEU data can significantly improve the general video understanding capability of Video-LLMs using an expert model trained on 50K high-quality data.

## Method

### Overall Architecture

VEU-Bench contains 30,000 videos and 49,536 QA samples. The data originates from existing datasets like AVE, MovieCuts, and AutoTransition, and has been filtered, balanced, and augmented. A total of 19 fine-grained tasks are constructed through a three-level task design (recognition/reasoning/judgment) crossed with ten editing dimensions. The expert model, Oscars, is based on Qwen2-VL-7B, fine-tuned using LoRA on the VEU-50K training set.

### Key Designs

1. **Ten-Dimension Three-Level Task System**:

    - **Function**: Systematically cover all aspects of video editing understanding.
    - **Mechanism**: **Intra-frame dimensions** (6) — Shot Size, Angle, Location, Subject, Type, and Color, which only require analyzing single frames. **Inter-frame dimensions** (2) — Motion and Speed, which require analyzing multi-frame changes within the same scene. **Inter-shot dimensions** (2) — Cut and Transition, involving cutovers between different scenes. Three evaluation levels: **Recognition** level (multiple-choice classification), **Reasoning** level (recognition + providing evidence and principles), and **Judgment** level (evaluating the function and effect of editing elements in specific videos).
    - **Design Motivation**: Professional video editing tutorials define classifications under these three granularities. The three-level evaluation progressively increases in difficulty from "what it is" $\rightarrow$ "why it is like this" $\rightarrow$ "how it performs".

2. **Ontology Knowledge Base-Based Automated Annotation Pipeline**:

    - **Function**: Expand VEU tasks from simple classification to reasoning and judgment while ensuring high-quality annotation.
    - **Mechanism**: First, a knowledge base is constructed for each editing element by referring to professional editing tutorials — defining "key attributes" (describing abstract patterns of each dimension) for reasoning tasks, and "functions" (describing the role of editing elements in video content) for judgment tasks. During annotation, the MLLM selects the most relevant attribute/function based on the video content, and then replaces abstract terms (e.g., "object", "scene") with concrete details. For example, the attribute "connecting two similarly shaped objects across frames" of a "match cut" is concretized into "match cut connects a bone and a spaceship of similar shapes across frames".
    - **Design Motivation**: Directly tasking MLLMs with annotating open-domain reasoning tasks yields poor performance (even GPT-4o struggles with VEU). By simplifying open-ended reasoning into a "knowledge-base-driven rewriting" task, the annotation difficulty is mitigated, quality is assured, and answers are guaranteed to contain correct editing pattern knowledge.

3. **Pattern Matching Evaluation Mechanism**:

    - **Function**: More accurately evaluate the quality of responses for reasoning and judgment tasks.
    - **Mechanism**: The evaluation score consists of two components: the Pattern Matching (PM) score and the Information Matching (IM) score. PM measures the alignment of the response with the editing pattern ontology (video-agnostic editing knowledge), while IM evaluates the accuracy of specific visual details within the response. The final open-ended question score is $S_{oe} = (5 \times Acc + S_{match})/2$, where $S_{match} = (PM + IM)/2$.
    - **Design Motivation**: Directly using LLMs for grading is prone to being "confused" by correct descriptive details in the response (such as successfully identifying objects in the scene), potentially awarding high scores even if the model misidentifies the editing pattern. PM regularization mitigates this bias, steering the grading to focus more on the editing understanding capability itself.

### Loss & Training

Oscars is based on Qwen2-VL-7B and fine-tuned using LoRA (r=16, $\alpha$=32). The learning rate is 1e-4, weight decay is 0.01, warmup ratio is 0.05, using the AdamW optimizer. Videos are sampled at 1 fps, up to 64 frames. Trained on 45,154 samples in the training set using 4 A100 GPUs.

## Key Experimental Results

### Main Results

| Model | Score_mc (Recognition) | Score_oe (Reasoning+Judgment) | Score_all | Gain vs. Prev. SOTA |
|------|----------------|---------------------|-----------|-------------|
| GPT-4o | **2.93** | **2.36** | **2.64** | - |
| Gemini-1.5-Pro | 2.71 | 2.11 | 2.44 | - |
| LLaVA-OV-7B (Prev. SOTA) | 2.27 | 1.69 | 1.98 | - |
| Qwen2-VL-7B (base) | 2.33 | 1.31 | 1.82 | - |
| **Oscars (Ours)** | **2.85** | **2.23** | **2.54** | **+28.3%** |

### Ablation Study

| Configuration | General Benchmark Gain | Description |
|------|------------|------|
| Qwen2-VL → Oscars (VideoMME-Attribute) | +7.3% | Attribute awareness |
| Qwen2-VL → Oscars (MVBench-State Change) | +5.5% | State change |
| Qwen2-VL → Oscars (TempCompass-Order) | +8.5% | Order understanding |
| Qwen2-VL → Oscars (Average of 9 Reasoning Tasks) | +8.3% | Overall improvement in reasoning capability |
| Simple prompt vs Context prompt (Qwen2-VL) | +13.7% | Adding editing knowledge definitions yields the largest improvement |
| Simple prompt vs Context prompt (VideoLLaMA2) | +6.6% | Weaker models also benefit |
| PM-only scoring vs PM+IM scoring (Spearman) | Higher alignment | PM regularization improves human alignment |

### Key Findings

- **Video-LLMs perform far below general benchmarks on VEU tasks**: Models reaching 80%+ on Video-MME score below random guessing in some dimensions on VEU-Bench. Identifying intra-frame editing elements is vastly easier than identifying inter-frame or inter-shot elements.
- **Reasoning and judgment tasks are highly challenging**: The average open-ended score of all models is less than 2/5. Judgment tasks (understanding editing intentions) are harder than reasoning tasks (describing editing patterns).
- **VEU data significantly enhances general video understanding**: Fine-tuning with only 50K VEU data yields an average improvement of 8.3% on reasoning-related tasks in VideoMME, MVBench, and TempCompass. This demonstrates that editing understanding training indeed strengthens the abstract reasoning capabilities of models.
- **Independence of conceptual knowledge vs. visual perception**: Models achieve decent accuracy (~2.7/3.0) in textual descriptions of editing concepts, indicating they "know" the editing concepts but cannot "see" them in videos. The bottleneck lies in the alignment between the language model's internal knowledge and the visual perception module.
- **Providing editing definitions via context prompts significantly improves performance**: Yields a 13.7% improvement for Qwen2-VL, though the effect is negligible for the already strong Gemini.

## Highlights & Insights

- **VEU as training data for abstract reasoning**: Editing elements are abstract professional concepts; understanding them requires pattern recognition and reasoning capabilities. VEU training data surprisingly improves reasoning tasks in general video understanding, revealing a new data strategy — utilizing domain-specific abstract task data to augment general reasoning capabilities.
- **Diagnostic experiment on 'having knowledge but unable to see'**: By testing textual concepts, the work distinguishes between "lack of knowledge" and "poor knowledge-visual alignment" failure modes, finding that the latter is the primary cause. This provides clear guidance for future improvements.
- **Ontology-knowledge-base-driven annotation pipeline**: Simplifying open-ended reasoning annotation into a two-step "select-then-rewrite" process ensures professional accuracy while achieving scalability. This strategy is extensible to other annotation scenarios requiring domain expertise.

## Limitations & Future Work

- The videos mostly source from existing datasets (AVE, MovieCuts, AutoTransition), limiting the diversity of video types and styles.
- Currently, only short video clips (1-60s) are evaluated; more complex editing narrative understanding in long-form videos remains unexplored.
- Although Oscars outperforms the open-source SOTA, it is still weaker than GPT-4o in the Cut and Transition dimensions, indicating that inter-shot understanding remains a major bottleneck.
- The PM score in pattern matching evaluation relies on LLM judgment, which may introduce systematic bias.
- The knowledge base is currently based on general editing tutorials and does not cover domain-specific editing techniques (e.g., documentaries, music videos).

## Related Work & Insights

- **vs AVE/MovieCuts/AutoTransition**: These benchmarks only perform classification tasks (identifying shot or cut types). VEU-Bench extends evaluations to three levels of reasoning and judgment, covering more comprehensive editing dimensions.
- **vs EditQA-2k**: EditQA explores the capability of Video-LLMs to analyze edited video content, but is still confined to editing effects rather than understanding the editing elements themselves. VEU-Bench provides a more systematic evaluation of editing understanding.
- **vs Video-MME/MVBench and Other General Benchmarks**: General benchmarks focus on "natural" visual understanding such as actions, events, and temporal sequence, neglecting the understanding of "artificial" dimensions like video editing. VEU-Bench fills this gap, and VEU training data can serve as an effective complement to general benchmarks.

## Rating

- Novelty: ⭐⭐⭐⭐ First to systematically propose a three-level evaluation framework for video editing understanding; the ontology-driven annotation pipeline is highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across 11 models, in-depth multi-dimensional analysis, and thorough transfer validation on general benchmarks.
- Writing Quality: ⭐⭐⭐⭐ Clear task definitions, rigorous logic in experimental analysis, and information-rich figures and tables.
- Value: ⭐⭐⭐⭐ The finding that VEU data enhances general understanding is inspiring; the evaluation framework and dataset are highly valuable to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] V2V-Bench: A Comprehensive Benchmark for Video-to-Video Generation Evaluation](../../ICML2026/video_generation/v2v-bench_a_comprehensive_benchmark_for_video-to-video_generation_evaluation.md)
- [\[CVPR 2025\] Video-Bench: Human-Aligned Video Generation Benchmark](video-bench_human-aligned_video_generation_benchmark.md)
- [\[CVPR 2025\] Mimir: Improving Video Diffusion Models for Precise Text Understanding](mimir_improving_video_diffusion_models_for_precise_text_understanding.md)
- [\[ICLR 2026\] UniVideo: Unified Understanding, Generation, and Editing for Videos](../../ICLR2026/video_generation/univideo_unified_understanding_generation_and_editing_for_videos.md)
- [\[CVPR 2025\] SketchVideo: Sketch-Based Video Generation and Editing](sketchvideo_sketch-based_video_generation_and_editing.md)

</div>

<!-- RELATED:END -->

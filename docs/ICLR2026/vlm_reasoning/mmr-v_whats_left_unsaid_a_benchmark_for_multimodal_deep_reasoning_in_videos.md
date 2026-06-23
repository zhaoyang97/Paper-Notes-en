---
title: >-
  [Paper Note] MMR-V: What's Left Unsaid? A Benchmark for Multimodal Deep Reasoning in Videos
description: >-
  [ICLR 2026][vlm_reasoning][Chain-of-Thought] MMR-V is an evaluation benchmark for "deep video reasoning" that emphasizes long-range multi-frame evidence mining and implicit reasoning of "what's left unsaid." It reveals that even the strongest Gemini-2.5-pro achieves only 64.3% accuracy, while CoT and test-time scaling are largely ineffective in this domain.
tags:
  - ICLR 2026
  - vlm_reasoning
  - Chain-of-Thought
date: 2026-05-08
content_hash: 9db739da294beba7
---
# MMR-V: What's Left Unsaid? A Benchmark for Multimodal Deep Reasoning in Videos

**Conference**: ICLR 2026  
**Project Page**: [https://mmr-v.github.io/](https://mmr-v.github.io/)  
**Dataset**: [https://huggingface.co/datasets/JokerJan/MMR-VBench](https://huggingface.co/datasets/JokerJan/MMR-VBench)  
**Area**: Multimodal Video Reasoning / Benchmark  
**Keywords**: Video Reasoning, Multi-frame Reasoning, Implicit Reasoning, MLLM Evaluation, Chain-of-Thought  

## TL;DR
MMR-V is an evaluation benchmark for "deep video reasoning" that emphasizes long-range multi-frame evidence mining and implicit reasoning of "what's left unsaid." It reveals that even the strongest Gemini-2.5-pro achieves only 64.3% accuracy, while CoT and test-time scaling are largely ineffective in this domain.

## Background & Motivation
**Background**: o1 and DeepSeek-R1 have pushed text-based reasoning to new heights, and models like o3 and GPT-5 have advanced image reasoning through tool usage. Multimodal reasoning has become a focal point, yet most research remains confined to images, leaving the more challenging domain of video reasoning relatively unexplored.

**Limitations of Prior Work**: Existing video benchmarks (e.g., MVBench, Video-MME) primarily evaluate "perception and understanding." Tasks often only require locating the specific frame mentioned in the question (question frame) and observing a few adjacent frames to find the answer. This paper summarizes these deficiencies into three points: (1) **Limited frame context**, as even long videos only utilize a few adjacent frames without leveraging the full temporal structure; (2) **Lack of reasoning**, where many questions can be answered via direct perception; (3) **Tasks detached from reality**, as simple perception fails to meet the capability requirements of real-world AI systems.

**Key Challenge**: Videos are inherently temporal with high information density, requiring models to find clues across **long-range, multiple frames** and perform multimodal reasoning—a blind spot that "perception-based" benchmarks cannot measure. A natural question arises: can current MLLMs "think with videos" similarly to how o3 handles images?

**Goal**: To construct a video benchmark that truly examines multimodal deep reasoning, ensuring that answers cannot be obtained through surface-level perception but must be derived by mining evidence across frames and understanding subtext.

**Core Idea (Dual-Process Theory)**: Inspired by Kahneman's Dual-Process Theory, the authors categorize tasks into **Implicit Reasoning** and **Explicit Reasoning**. The former examines "what's left unsaid" or "subtext" (e.g., a brown coat symbolizing a father, or room number 7 symbolizing luck), resembling "fast thinking" involving EQ and world knowledge. The latter focuses on rigorous logical reasoning based on details explicitly presented in the video (e.g., magic trick reveals, causal analysis).

## Method

### Overall Architecture
The construction of MMR-V revolves around three principles: **P1 Multi-frame** (must refer to long-range frames), **P2 Deep Reasoning** (answers cannot be directly perceived), and **P3 Real** (consistent with real-user understanding without cognitive bias). The pipeline consists of "Manual Video Curation → Manual Question/Answer Annotation → Adversarial Distractor Generation → Checklist Quality Control," resulting in 317 videos and 1257 multiple-choice tasks with an average of 10 options.

```mermaid
flowchart LR
    A[YouTube Video Curation<br/>4-item checklist] --> B[Manual Annotation<br/>Question + Correct Answer]
    B --> C[Distractor Annotation<br/>Str.1/2/3]
    C --> D[Checklist Quality Control<br/>5 Annotators]
    D --> E[MMR-V<br/>317 Videos / 1257 Tasks]
```

### Key Designs

**1. Implicit/Explicit Dual-track Task System**: Explicitly incorporates "subtext" into reasoning evaluation. The authors broaden multimodal reasoning beyond text-dominant tasks (like math or puzzles) to include "integrating visual evidence such as artistic style, lighting, and depth of field" across long-range evidence chains. Tasks are divided into 10 major categories and 33 subcategories. The implicit side includes Metaphor Understanding (MU), Theme Understanding (TU), Emotion Recognition (ER), Comment Matching (CM), and Implicit Symbolism (IS). The explicit side includes Causal Reasoning (CAR), Sequential Structure Reasoning (SSR), Counter-Intuitive Reasoning (CIR), Cross-modal Transfer Reasoning (CTR), and Video Type & Intent (VTI).

**2. Long-range Multi-frame Evidence Distribution**: Fundamentally prevents "answering by looking at one frame." Video curation deliberately excludes linear descriptive content (like daily vlogs or sports) in favor of elaborately designed, thematic videos. Statistically, video durations range from 7 to 3771 seconds (average 277s). A single question requires reasoning across approximately 12 frames on average, covering roughly 60% of the video duration. This means evidence frames are often far from the question frame, forcing models to perform cross-frame retrieval and analysis.

**3. Adversarial Distractor Annotation**: Uses model errors to create "trap options." To ensure distractors are confusing, three strategies were designed. Str.1 is the core: the strong model GPT-4o answers manually annotated questions; **if it answers incorrectly (verified manually), the incorrect answer is kept as a high-quality distractor**. Str.2 involves generating distractors via GPT-4o given the question and correct answer. Str.3 is manual writing. A comparison of 100 questions (Table 1) shows Str.1 is the most difficult—GPT-4o achieved only 59% and Qwen-VL-7B only 37% on these. Notably, when GPT-4o initially answered these 100 questions, its accuracy was only 17%, exposing the current weaknesses in multimodal reasoning.

## Key Experimental Results

### Main Results
The evaluation covers 11 closed-source and 10 open-source models, primarily using zero-shot and zero-shot + CoT settings. Random accuracy is approximately 10%.

| Model | Overall | Description |
|------|---------|------|
| Gemini-2.5-pro (1 fps) | **64.3%** | Best performance overall |
| GPT-5 (Fixed frames) | 60.9% | Best in fixed frame setting |
| GPT-4o | 52.8% | |
| GPT-4o-mini | 34.8% | Significant drop for smaller models of same architecture |
| Gemma-3-27b-it | Open-source best | Still trails closed-source models |
| Qwen2.5-VL-72B / 7B | 39.1% / 30.1% | Demonstrates scaling law |
| LLaVA-Video | 18.4% | |

### Ablation Study

| Strategy | Average Gain |
|------|----------|
| CoT prompting | +0.88% |
| "Thinking" models | +2.4% |
| Adding audio modality (Gemini-2.0-Flash, etc.) | +1.0%~+1.4% |

### Key Findings
- **Multimodal Reasoning Hardship**: CoT and test-time scaling, which are effective in the text domain, are nearly useless on MMR-V (CoT gain is only +0.88%). Sampling analysis shows visual analysis accounts for only ~10% of CoT content, indicating that model reasoning is still "perception of question frame + text reasoning," lacking the ability to embed cross-frame evidence mining into the reasoning chain.
- **Multimodal Benefits**: Models supporting full modalities show stable, small improvements (approx. 1%) when audio is included.
- **Human-AI Gap**: While models can reach human levels in text reasoning, a significant gap remains in video multimodal reasoning.
- **Scaling Law**: Larger models significantly outperform smaller models within the same architecture (e.g., GPT-4o vs. GPT-4o-mini shows a relative gain of ~18%).

## Highlights & Insights
- **"What's left unsaid" as a First-class Citizen**: Including implicit reasoning—metaphor, theme, emotion, and cultural symbols—corrects the narrow definition of "multimodal reasoning = math/puzzles."
- **Model Errors as Distractors**: Str.1 transforms the failures of strong models into high-quality trap options, increasing difficulty while aligning with "model confusion points" more effectively than manual or purely generative methods.
- **Revealing the Essence of CoT Differences**: Quantifying that "visual analysis only accounts for 10% of CoT" highlights that current MLLM reasoning is text-led, providing a clear direction for future "Visual Chain-of-Thought" research.

## Limitations & Future Work
- **Limited Scale**: 317 videos and 1257 questions is relatively small. The multiple-choice format may allow for heuristic shortcuts, necessitating more robust evaluation protocols.
- **Annotation Subjectivity**: Implicit reasoning is naturally subjective. Although mitigated by cross-referencing popular comments, the objectivity of "correct answers" may still be challenged.
- **Lack of a Solution**: The paper primarily diagnoses problems (ineffective CoT, lack of visual reasoning) but does not propose specific training/inference methods to enhance multimodal CoT.

## Related Work & Insights
- **Video Understanding Benchmarks** (MVBench, Video-MME) focus on perception and adjacent frame understanding; MMR-V is orthogonal, filling the gap in "long-range cross-frame deep reasoning."
- **Multimodal Reasoning** (o3, GPT-5 think-with-image) inspired the extension to the video domain; MMR-V's findings suggest evidence mining in video is far more difficult than in images.
- **Cognitive Science** (Kahneman's Dual-Process, Polanyi's Tacit Knowledge) provides the theoretical foundation for the implicit/explicit classification, serving as an example of borrowing from psychology for benchmark design.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Introduces implicit/subtext reasoning and long-range evidence mining to video evaluation with a clear, innovative focus.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 21 models + human experiments + multi-dimensional ablations (distractor strategies/audio/scaling), providing a solid diagnosis.
- **Writing Quality**: ⭐⭐⭐⭐ Motivations, task definitions, and findings are well-structured with rich illustrations.
- **Value**: ⭐⭐⭐⭐ Exposes the fundamental shortfalls in "thinking with videos" for current MLLMs, providing a high-quality touchstone for multimodal reasoning research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] IV-Bench: A Benchmark for Image-Grounded Video Perception and Reasoning in Multimodal LLMs](iv-bench_a_benchmark_for_image-grounded_video_perception_and_reasoning_in_multim.md)
- [\[ICLR 2026\] MMR-Life: Piecing Together Real-life Scenes for Multimodal Multi-image Reasoning](mmr-life_piecing_together_real-life_scenes_for_multimodal_multi-image_reasoning.md)
- [\[ICLR 2026\] Agent-X: Evaluating Deep Multimodal Reasoning in Vision-Centric Agentic Tasks](agent-x_evaluating_deep_multimodal_reasoning_in_vision-centric_agentic_tasks.md)
- [\[ICLR 2026\] PuzzleWorld: A Benchmark for Multimodal, Open-Ended Reasoning in Puzzlehunts](puzzleworld_a_benchmark_for_multimodal_open-ended_reasoning_in_puzzlehunts.md)
- [\[ICLR 2026\] MathNet: A Global Multimodal Benchmark for Mathematical Reasoning and Retrieval](mathnet_a_global_multimodal_benchmark_for_mathematical_reasoning_and_retrieval.md)

</div>

<!-- RELATED:END -->

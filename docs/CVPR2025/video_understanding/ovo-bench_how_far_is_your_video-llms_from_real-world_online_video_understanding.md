---
title: >-
  [Paper Note] OVO-Bench: How Far is Your Video-LLMs from Real-World Online Video Understanding?
description: >-
  [CVPR 2025][Video Understanding][Online Video Understanding] OVO-Bench is the first online video benchmark that emphasizes the importance of timestamps in video understanding. It categorizes online video understanding into three modes: "Backward Tracing", "Real-Time Perception", and "Forward Active Responding", evaluating the online understanding capabilities of Video-LLMs through 12 tasks, 644 videos, and over 2,800 fine-grained annotations.
tags:
  - "CVPR 2025"
  - "Video Understanding"
  - "Online Video Understanding"
  - "Video Large Language Models"
  - "Temporal Awareness"
  - "Benchmark"
  - "Real-Time Perception"
date: 2026-05-08
content_hash: 3380651195f2c165
---

# OVO-Bench: How Far is Your Video-LLMs from Real-World Online Video Understanding?

**Conference**: CVPR 2025  
**arXiv**: [2501.05510](https://arxiv.org/abs/2501.05510)  
**Code**: [https://github.com/JoeLeelyf/OVO-Bench](https://github.com/JoeLeelyf/OVO-Bench)  
**Area**: Video Understanding  
**Keywords**: Online Video Understanding, Video Large Language Models, Temporal Awareness, Benchmark, Real-Time Perception

## TL;DR
OVO-Bench is the first online video benchmark that emphasizes the importance of timestamps in video understanding. It categorizes online video understanding into three modes: "Backward Tracing", "Real-Time Perception", and "Forward Active Responding", evaluating the online understanding capabilities of Video-LLMs through 12 tasks, 644 videos, and over 2,800 fine-grained annotations.

## Background & Motivation

**Background**: Video-LLMs (such as GPT-4o, Gemini-1.5-Pro, Qwen2-VL) have achieved impressive results on offline video understanding benchmarks. However, existing benchmarks assume that the model has access to the full video before answering questions (an offline setting), which exhibits a massive gap from the requirements of real-world online video assistants.

**Limitations of Prior Work**: (1) Existing offline benchmarks cannot evaluate the "temporal awareness" capabilities of models—namely, the ability to provide different answers when queried at different timestamps of a video stream; (2) A few online benchmarks (such as VStream-QA, StreamingBench) primarily focus on responding immediately using historical visual inputs, lacking the evaluation dimension of "waiting for future information before answering"; (3) There is a lack of systematic evaluation of the three fundamentally different reasoning modes in online video understanding (backward, real-time, forward).

**Key Challenge**: The fundamental difference between offline and online video understanding lies in "temporal awareness"—when the same question is asked at different timestamps of a video, the answers may vary. Existing benchmarks completely overlook this critical dimension.

**Goal**: Build a comprehensive benchmark to evaluate the online video understanding capabilities of Video-LLMs, covering three modes: Backward Tracing, Real-Time Perception, and Forward Active Responding.

**Key Insight**: Inspired by the human video understanding process, the concept of "Video Chain-of-Time" is proposed. When facing queries in streaming videos, a model needs to decide whether to answer immediately using past information, focus on the current frame for a real-time answer, or wait for sufficient future information before responding.

**Core Idea**: Systematically categorize online video understanding into three modes (Backward Tracing / Real-Time Perception / Forward Active Responding), design 12 tasks covering these three modes, and simulate continuous information processing through an evaluation pipeline with dense querying along the timeline.

## Method

### Overall Architecture
The OVO-Bench evaluation framework consists of three levels: (1) Data Construction: Collecting 644 videos by crawling from multiple datasets and the web, and generating 2,814 high-quality meta-annotations through semi-automatic and manual curation; (2) Prompt Generation: Automated Q&A generation + visually assisted option generation + manual review; (3) Evaluation Pipeline: Densely querying Video-LLMs along the timeline to simulate online understanding scenarios.

### Key Designs

1. **Formal Definitions of the Three Online Understanding Modes**:

    - Function: Systematically define the three core capabilities that online video understanding systems should possess, guiding the benchmark design.
    - Mechanism: Given a query $Q_{t_0}$ at timestamp $t_0$ and a streaming video $X_{(-\infty,+\infty)}$: (a) Backward Tracing $R_{t_0} = P(Q_{t_0}, X_{(-\infty,-T]})$—utilizing long-term historical information to answer; (b) Real-Time Perception $R_{t_0} = P(Q_{t_0}, X_{(-T,t_0]})$—understanding events currently happening; (c) Forward Active Responding $R_{(t_0,+\infty)} = P(Q_{t_0}, X_{(t_0,+\infty)})$—delaying response until sufficient information is available.
    - Design Motivation: These three modes correspond to three cognitive strategies humans use during online video understanding: recalling the past, perceiving the present, and predicting the future. They require entirely different capabilities and should not be conflated.

2. **12 Fine-grained Task Designs**:

    - Function: Comprehensive coverage of various dimensions of online video understanding to provide fine-grained diagnostic capabilities.
    - Mechanism: Backward Tracing includes 3 tasks (Episodic Memory EPM, Action Sequence Identification ASI, Hallucination Detection HLD); Real-Time Perception includes 6 tasks (Spatio-Temporal Understanding STU, Object Recognition OJR, Attribute Recognition ATR, Action Recognition ACR, OCR, Future Prediction FPD); Forward Active Responding includes 3 tasks (Repeated Event Counting REC, Sequential Step Recognition SSR, Clue-Revealing Response CRR). Each task contains annotations accurate to the frame-level timestamp.
    - Design Motivation: Forward Active Responding (FAR) is a unique contribution of OVO-Bench. It is the first evaluation dimension requiring models to "judge when to answer is more important than what to answer". Traditional benchmarks assume models should answer immediately, but a real-world online assistant needs to know when to wait.

3. **Dense Timeline Query Evaluation Pipeline**:

    - Function: Enable offline Video-LLMs to be evaluated in simulated online scenarios, allowing fair comparison.
    - Mechanism: For Backward Tracing and Real-Time Perception tasks, the video is truncated at the query timestamp and converted into multiple-choice questions for evaluation. For Forward Active Responding tasks, a "multi-trigger dense query" pipeline is designed—repeatedly querying the model at multiple timestamps after the query is proposed to see if there is sufficient information to answer, simulating the process of the model continuously adapting to new visual inputs. Option generation uses rule-based and vision-driven transformations to introduce distractor information from the original video to increase difficulty.
    - Design Motivation: Currently, the most dominant and powerful Video-LLMs are offline models. Excluding them entirely would miss a lot of valuable comparisons. Simulating online settings allows the evaluation of an important question: "How to effectively utilize SOTA offline models for online understanding."

### Loss & Training
OVO-Bench is an evaluation benchmark and does not involve training. Annotations are semi-automatically generated via GPT-4o and Gemini-1.5 Pro, followed by manual refinement. Video sources are diverse: public datasets like Ego4D, QA-Ego4D, OpenEQA, COIN, CrossTask, MovieNet, as well as YouTube crawls.

## Key Experimental Results

### Main Results
Overall performance of 11 Video-LLMs on OVO-Bench:

| Model | Real-Time Perception Avg | Backward Tracing Avg | Forward Responding Avg | Overall Avg |
|------|-------------|-------------|-------------|---------|
| Human Agents | 93.20 | 92.33 | 92.90 | **92.81** |
| Gemini 1.5 Pro | 69.32 | 62.54 | 57.15 | 63.00 |
| GPT-4o | 64.46 | 60.75 | 53.40 | 59.54 |
| Qwen2-VL | Medium | Medium | Low | Medium |
| Flash-VStream (online model) | Lower | Lower | Lower | Lower |

### Ablation Study

| Analysis Dimension | Findings |
|---------|------|
| Offline vs Online Models | Offline SOTA significantly outperforms online models (counter-intuitive) |
| Forward Active Responding | All models perform worst on this, showing the largest gap with humans |
| GPT-4 (blind) | Text-only LLM achieves 53.82 on Backward Tracing, indicating text bias |
| Hallucination Detection (HLD) | Models generally score below 50%, showing severe response bias |
| OCR Task | Gemini is the strongest (85.91), indicating significant privileges in visual encoding capability |

### Key Findings
- There is a massive gap of ~30% between humans and the strongest model (92.81 vs 63.00), demonstrating that online video understanding is far from being solved.
- Forward Active Responding (FAR) is the most challenging mode—all models score only around 30% on the REC task, indicating that models lack the ability to judge "when to answer".
- Counter-intuitively, online models (like Flash-VStream) perform worse than offline models directly applied to online scenarios, exposing deficiencies in current online model architectures.
- On the hallucination detection task, models generally score less than 50%, showing a tendency for models to fabricate answers for events that do not exist.
- A text-only LLM (GPT-4-turbo blind) achieves 53.82 on Backward Tracing, suggesting the presence of textual shortcuts.

## Highlights & Insights
- **Pioneering Proposal of Forward Active Responding**: First to systematically evaluate the model's ability to "judge when there is enough information to answer." This is a core requirement of real-world online assistants but has been ignored by all previous benchmarks—defining a new research direction.
- **Counter-Intuitive Finding of Online vs. Offline Models**: Powerful offline models outperform dedicated online models in online scenarios. This finding has important guiding significance for the research direction of online video understanding.
- **Video Chain-of-Time Concept**: Analogous to Chain-of-Thought, it proposes a temporal reasoning paradigm when facing streaming queries—models should first determine whether the information originates from the past, present, or future, and then decide on the answering strategy.

## Limitations & Future Work
- The evaluation is still predominantly based on multiple-choice questions, which cannot fully capture the complexity of open-ended online interactions.
- The evaluation pipeline for Forward Active Responding (dense multi-trigger querying) incurs high computational costs.
- The dataset scale of 644 videos and 2,814 QA pairs is still limited.
- Only English scenarios were evaluated; cross-lingual online video understanding was not covered.

## Related Work & Insights
- **vs StreamingBench**: StreamingBench focuses primarily on real-time perception and lacks evaluation for forward active responding. OVO-Bench's three-mode classification is more complete.
- **vs Video-MME / LongVideoBench**: These benchmarks evaluate offline long video understanding and do not involve timestamp awareness. OVO-Bench emphasizes that "the same question yields different answers at different timestamps."
- **vs E.T.Bench**: E.T.Bench explores temporal event detection but still in an offline setting. OVO-Bench pushes further into online settings.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The Forward Active Responding mode and the three-mode framework are pioneering contributions to this field.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation of 11 models, though the data scale could be larger.
- Writing Quality: ⭐⭐⭐⭐ Problem motivation is clear, classification taxonomy is rigorous, and illustrations are rich.
- Value: ⭐⭐⭐⭐⭐ Defines the evaluation paradigm for online video understanding, exposes the huge gap in current models, and will drive research in this direction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] LiveStar: Live Streaming Assistant for Real-World Online Video Understanding](../../NeurIPS2025/video_understanding/livestar_live_streaming_assistant_for_real-world_online_video_understanding.md)
- [\[CVPR 2025\] Q-Bench-Video: Benchmark the Video Quality Understanding of LMMs](q-bench-video_benchmark_the_video_quality_understanding_of_lmms.md)
- [\[CVPR 2025\] DynFocus: Dynamic Cooperative Network Empowers LLMs with Video Understanding](dynfocus_dynamic_cooperative_network_empowers_llms_with_video_understanding.md)
- [\[CVPR 2025\] LION-FS: Fast & Slow Video-Language Thinker as Online Video Assistant](lion-fs_fast_slow_video-language_thinker_as_online_video_assistant.md)
- [\[CVPR 2026\] An Empirical Study on How Video-LLMs Answer Video Questions](../../CVPR2026/video_understanding/an_empirical_study_on_how_video-llms_answer_video_questions.md)

</div>

<!-- RELATED:END -->

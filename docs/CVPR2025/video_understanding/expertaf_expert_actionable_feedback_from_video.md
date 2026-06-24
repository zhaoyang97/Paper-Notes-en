---
title: >-
  [Paper Note] ExpertAF: Expert Actionable Feedback from Video
description: >-
  [CVPR 2025][Video Understanding][Skill Assessment] This paper proposes ExpertAF, the first method to generate actionable coaching feedback from video. By integrating a multimodal model with video, 3D human pose, and language, it not only generates textual feedback describing mistakes and suggesting improvements, but also retrieves/generates correct expert demonstrations. Leveraging the Ego-Exo4D dataset and LLMs to construct weakly-supervised training data…
tags:
  - "CVPR 2025"
  - "Video Understanding"
  - "Skill Assessment"
  - "Actionable Feedback"
  - "Multimodal Video-Language Model"
  - "Coaching Feedback"
  - "3D Human Pose"
date: 2026-05-08
content_hash: 78b85152180ae01d
---

# ExpertAF: Expert Actionable Feedback from Video

**Conference**: CVPR 2025  
**arXiv**: [2408.00672](https://arxiv.org/abs/2408.00672)  
**Code**: Coming soon  
**Area**: Video Understanding  
**Keywords**: Skill Assessment, Actionable Feedback, Multimodal Video-Language Model, Coaching Feedback, 3D Human Pose

## TL;DR

This paper proposes ExpertAF, the first method to generate actionable coaching feedback from video. By integrating a multimodal model with video, 3D human pose, and language, it not only generates textual feedback describing mistakes and suggesting improvements, but also retrieves/generates correct expert demonstrations. Leveraging the Ego-Exo4D dataset and LLMs to construct weakly-supervised training data, it significantly outperforms strong baselines across soccer, basketball, and climbing.

## Background & Motivation

**Background**: Video skill assessment has a certain research foundation, including action quality assessment (e.g., figure skating and gymnastics scoring) and exemplar alignment. However, existing methods can only provide scores or perform comparisons, and cannot tell learners "what they specifically did wrong and how to improve."

**Limitations of Prior Work**: (1) Existing skill assessment methods only output a single score, which learners cannot use for improvement; (2) there is no method that can simultaneously provide both textual guidance and visual demonstrations as complementary feedback forms; (3) there is a lack of large-scale paired training data containing the "incorrect version" of the same action, the "correct version", and the corresponding expert feedback.

**Key Challenge**: To realize a true AI coach, the model must simultaneously possess three capabilities: understanding the current action, detecting execution errors, and offering specific improvement suggestions. This is far more complex than simple action recognition or quality scoring. Existing datasets either only contain high-level demonstrations (such as HowTo100M) or lack paired incorrect-correct demonstrations and expert commentary.

**Goal**: (1) Define a new task, "video actionable feedback," which includes three subtasks: textual feedback generation, expert demonstration retrieval, and expert pose generation; (2) leverage Ego-Exo4D to build a weakly-supervised training dataset; (3) design a unified multimodal model to perform all three subtasks.

**Key Insight**: Ego-Exo4D happens to contain first- and third-person videos of individuals performing the same activity at different skill levels, alongside 3D poses and expert feedback. The authors design an ingenious pipeline that utilizes LLMs to classify and pair the feedback, and subsequently uses PA-MPJPE for temporal alignment to automatically construct (incorrect demonstration, expert feedback, correct demonstration) triplets.

**Core Idea**: By using LLMs to parse, classify, and pair free-form expert commentaries in Ego-Exo4D to build a weakly-supervised training dataset, and then training a multimodal model based on the LLaVA architecture that integrates video (InternVideo2), 3D pose (PCT encoder), and language, the model achieves end-to-end generation from video to actionable coaching feedback.

## Method

### Overall Architecture

ExpertAF receives the learner's video (ego+exo dual views) and 3D pose sequence as input. After multimodal encoding, these are fed into a large language model to output three forms of feedback: (1) textual feedback describing what was done well and what needs improvement; (2) retrieval of the most relevant correct demonstration video from an expert database; and (3) generation of the corrected expert 3D pose sequence. All three tasks are realized through a unified architecture, differing only in the input combination and output modalities.

### Key Designs

1. **Weakly-Supervised Dataset Construction Pipeline**:

    - **Function**: Automatically construct (learner video, expert feedback, expert demonstration) triplets from Ego-Exo4D.
    - **Mechanism**: A three-step pipeline: (a) Use Llama3 to summarize expert comments, label body parts (head, shoulder, arm, leg, hand, jump), and classify them as correct/incorrect; (b) pair incorrect and correct demonstrations based on body part annotations and skill levels (beginner vs. expert); (c) use PA-MPJPE for temporal alignment, selecting the top-$k$ pairs with the minimum alignment error. This yields 25,505 training and 1,272 test samples (with human verification for the test set).
    - **Design Motivation**: Manually labeling paired data is extremely expensive. A weakly-supervised scheme using LLM classification plus pose alignment drastically reduces data construction costs. The beginner-vs-expert pairing strategy ensures a clear contrast between incorrect and correct actions.

2. **Multimodal Encoding and Unified Architecture**:

    - **Function**: Encode videos, poses, and texts into unified token sequences to perform multimodal reasoning using LLMs.
    - **Mechanism**: Videos are encoded using InternVideo2 and converted to tokens via a visual projector (32 tokens total for ego+exo); 3D poses are encoded using a PCT (Pose as Compositional Tokens) encoder and converted to tokens via a pose projector; text uses a standard tokenizer. The three types of tokens are concatenated and fed into Llama3 for sequence prediction. For the textual feedback and retrieval tasks, the LLM is frozen and only the projectors are trained, while for the pose generation task, the LLM is fine-tuned (due to the need to modify token dimensions).
    - **Design Motivation**: LLaVA-style modality mapping and LLM reasoning architecture have proven highly effective in vision-language tasks. Integrating poses into the unified framework instead of modeling them separately allows the model to naturally associate visual appearance, body poses, and language descriptions.

3. **Three-Task Inference Design**:

    - **Function**: Support three inference modes: feedback generation $\mathcal{F}_t$, demonstration retrieval $\mathcal{F}_r$, and pose generation $\mathcal{F}_g$.
    - **Mechanism**: Feedback generation takes (learner video, learner pose, expert video, expert pose) as input to predict text tokens $\mathbf{t} = \mathcal{L}_s(\mathbf{v}, \mathbf{p}, \bar{\mathbf{v}}, \bar{\mathbf{p}})$; demonstration retrieval takes (learner video, learner pose, text feedback) as input to predict pose tokens $\bar{\mathbf{p}} = \mathcal{L}_s(\mathbf{v}, \mathbf{p}, \mathbf{t})$ for similarity matching; pose generation directly outputs decodable pose tokens $\bar{\mathbf{p}}' = \mathcal{L}_s(\mathbf{v}, \mathbf{p}', \mathbf{t})$, which are then restored to 3D coordinates using the PCT decoder. During inference, an end-to-end mode that accepts only the learner's video is also supported.
    - **Design Motivation**: The three tasks are essentially three facets of the same problem: understanding errors $\rightarrow$ describing corrections $\rightarrow$ demonstrating corrections. The unified architecture allows shared representations across tasks, reinforcing each other.

### Loss & Training

All three tasks use the standard cross-entropy loss: $\min_\theta \{-\log(\mathbf{t} | \mathbf{v}, \mathbf{p}, \bar{\mathbf{v}}, \bar{\mathbf{p}}; \theta)\}$. The feedback and retrieval tasks are trained for 10 epochs with a learning rate of $2 \times 10^{-2}$, updating only the projectors. The pose generation task is trained for 5 epochs with a learning rate of $5 \times 10^{-6}$, fine-tuning the LLM. The video encoder and pose encoder/decoder remain frozen throughout.

## Key Experimental Results

### Main Results

| Method | Feedback B@4 | Feedback ROUGE-L | Human Eval (1-4) | Retrieval R@50 | Retrieval medR ↓ | Pose PA-MPJPE ↓ |
|------|------------|------------|-----------|----------|------------|---------------|
| InternVideo2-NN-test | 43.0 | 50.6 | 1.8 | 14.5 | 191 | 159 |
| LLaVA | 28.5 | 44.2 | 1.3 | 15.0 | 183 | — |
| LLaVA-FT w/ pose | 43.6 | 51.7 | 2.5 | 18.0 | 172 | 150 |
| PoseScript/Fix | 24.1 | 46.3 | 1.1 | 15.9 | 182 | 182 |
| **ExpertAF** | **45.8** | **55.7** | **3.1** | **22.5** | **146** | **131** |

### Ablation Study

| Configuration | B@4 | ROUGE-L | R@50 | PA-MPJPE ↓ |
|------|-----|---------|------|------------|
| ExpertAF (full) | 45.8 | 55.7 | 22.5 | 131 |
| w/o video | 45.6 | 55.4 | 19.5 | 136 |
| w/o pose | 45.3 | 55.1 | 19.0 | — |
| w/ incorrect-only | 44.9 | 54.6 | 19.1 | 135 |
| w/o alignment | 42.8 | 52.0 | 18.0 | 147 |
| w/ global pose | 43.9 | 53.7 | 18.7 | 145 |

### Key Findings

- **Significant Gap in Human Evaluation Scores**: ExpertAF's human evaluation score (3.1/4.0) is 2.4 times that of LLaVA, indicating that the feedback generated by the model indeed possesses coaching value.
- **Temporal Alignment is Crucial**: Removing PA-MPJPE temporal alignment leads to a drop in performance across all metrics (e.g., B@4 drops from 45.8 to 42.8), proving that high-quality paired data is key.
- **Effectiveness of Multimodal Fusion**: Removing either the video or pose modality reduces retrieval and generation performance, showing that the two modalities provide complementary information.
- **Conditional Expert Demonstration Outperforms Global Demonstration**: Personalized expert demonstrations aligned with specific mistakes are more effective than generic "correct executions."

## Highlights & Insights

- **Prominent Completeness in Task Definition**: It does not merely score actions; rather, it provides a complete coaching feedback loop—telling you what was wrong, how to improve, and displaying the correct format. This definition of "full-spectrum actionable feedback" is much closer to real coaching scenarios than any prior work.
- **Highly Practical Weakly-Supervised Data Construction Scheme**: Using LLMs for text classification + skill level pairing + pose alignment converts a paired data construction problem, which typically requires intensive manual annotation, into an almost automated pipeline. This scheme can be transferred to other skill learning scenarios that require paired data.
- **First Realization of Text-Conditioned 3D Pose Generation**: Generating corrected pose sequences based on the coach's textual description is a completely new task.

## Limitations & Future Work

- Currently only validated on three sports scenarios (basketball, soccer, climbing). For skills requiring fine-grained hand manipulation (such as cooking or playing musical instruments), the performance might be limited.
- The 3D poses in Ego-Exo4D are automatically reconstructed; some samples exhibit reconstruction noise, which limits the upper bound of pose generation quality.
- The granularity of expert feedback is inconsistent—some experts provide fine-grained feedback (e.g., "left knee locked"), while others only give coarse descriptions (e.g., "incorrect posture").
- Some feedback in the commentaries cannot be visualized (e.g., "the athlete looks tired"), making it difficult for the model to handle such scenarios.
- Future work can extend this to real-time feedback scenarios (streaming processing) or multi-turn interactive coaching.

## Related Work & Insights

- **vs. Action Quality Assessment (AQA)**: Traditional AQA only outputs scores, whereas ExpertAF outputs full coaching feedback. Although Fitness-AQA detects specific error categories, it relies on a fixed error taxonomy, while ExpertAF provides open-ended feedback.
- **vs. PoseScript/PoseFix**: These methods explore pose description and correction, but they only handle static poses and templated descriptions. ExpertAF processes video sequences and free-form expert commentaries.
- **vs. LLaVA**: As a general vision-language model, LLaVA tends to describe video content rather than provide coaching critique, and still lacks the expert perspective even after fine-tuning.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ A brand new task definition, achieving video to full-spectrum coaching feedback for the first time.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluated on three scenarios with multiple baselines, ablations, and human evaluation.
- **Writing Quality**: ⭐⭐⭐⭐ Clear problem formulation and systematic description of method.
- **Value**: ⭐⭐⭐⭐⭐ Holds significant application value for AI-assisted skill learning; the weakly-supervised data construction scheme is widely reusable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MMVU: Measuring Expert-Level Multi-Discipline Video Understanding](mmvu_measuring_expert-level_multi-discipline_video_understanding.md)
- [\[ICCV 2025\] TimeExpert: An Expert-Guided Video LLM for Video Temporal Grounding](../../ICCV2025/video_understanding/timeexpert_an_expert-guided_video_llm_for_video_temporal_grounding.md)
- [\[CVPR 2026\] Question-guided Visual Compression with Memory Feedback for Long-Term Video Understanding](../../CVPR2026/video_understanding/question-guided_visual_compression_with_memory_feedback_for_long-term_video_unde.md)
- [\[CVPR 2025\] M-LLM Based Video Frame Selection for Efficient Video Understanding](m-llm_based_video_frame_selection_for_efficient_video_understanding.md)
- [\[CVPR 2025\] Q-Bench-Video: Benchmark the Video Quality Understanding of LMMs](q-bench-video_benchmark_the_video_quality_understanding_of_lmms.md)

</div>

<!-- RELATED:END -->

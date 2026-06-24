---
title: >-
  [Paper Note] MimeQA: Towards Socially-Intelligent Nonverbal Foundation Models
description: >-
  [NeurIPS 2025][Video Understanding][nonverbal social intelligence] This work introduces MimeQA, the first nonverbal social reasoning benchmark built on mime performance videos. It comprises 101 videos and 806 QA pairs organized across three hierarchical question levels (grounding the imagined → scene-level understanding → global reasoning), and reveals a severe gap between current VideoLLMs and humans on nonverbal social understanding (20–30% vs. 86%).
tags:
  - "NeurIPS 2025"
  - "Video Understanding"
  - "nonverbal social intelligence"
  - "mime understanding"
  - "video question answering"
  - "multimodal foundation models"
  - "social cognition"
date: 2026-05-08
content_hash: e94ffda2e9f02e13
---

# MimeQA: Towards Socially-Intelligent Nonverbal Foundation Models

**Conference**: NeurIPS 2025
**arXiv**: [2502.16671](https://arxiv.org/abs/2502.16671)  
**Code**: [GitHub](https://github.com/MIT-MI/MimeQA)  
**Area**: Video Understanding
**Keywords**: nonverbal social intelligence, mime understanding, video question answering, multimodal foundation models, social cognition

## TL;DR

This work introduces MimeQA, the first nonverbal social reasoning benchmark built on mime performance videos. It comprises 101 videos and 806 QA pairs organized across three hierarchical question levels (grounding the imagined → scene-level understanding → global reasoning), and reveals a severe gap between current VideoLLMs and humans on nonverbal social understanding (20–30% vs. 86%).

## Background & Motivation

**Background**: Socially intelligent AI is increasingly important, yet existing research focuses predominantly on purely linguistic data and tasks (e.g., social dialogue, QA), or on multimodal data dominated by language. VideoLLMs achieve strong performance on benchmarks such as Ego4D and Video-MME.

**Limitations of Prior Work**: (i) Existing benchmarks rely heavily on the language modality, treating nonverbal signals (body language, gestures, facial expressions) as secondary information; (ii) this leads to progress in language understanding while nonverbal social understanding remains severely underdeveloped; (iii) some benchmarks can be solved reasonably well even without video input, indicating severe language bias.

**Key Challenge**: Genuine social intelligence requires understanding both verbal and nonverbal signals simultaneously, yet current models and evaluation frameworks are overwhelmingly language-centric, lacking effective means to measure nonverbal social cognition.

**Goal**: How can AI models' nonverbal social reasoning capabilities be systematically evaluated?

**Key Insight**: Mime performance videos—an art form that conveys meaning exclusively through gesture and movement, completely independent of language—serve as the evaluation medium.

**Core Idea**: Mime performances inherently exclude linguistic cues, compelling AI models to rely purely on their understanding of human body movement, gesture, and social interaction to answer questions.

## Method

### Overall Architecture

MimeQA is an open-ended video QA benchmark. A total of 221 mime videos are collected from YouTube, of which 101 pass quality filtering. A rigorous annotation and verification pipeline is applied to produce 806 QA pairs. Questions are organized across three temporal-scale levels, progressing from low-level visual recognition to high-level social cognition.

### Key Designs

1. **Three-Level Question Hierarchy**:

    - **Function**: Decomposes nonverbal social reasoning into three evaluation layers ranging from concrete to abstract.
    - **Design Motivation**: Cognitive science indicates that nonverbal understanding involves a progressive process of perception → contextual interpretation → global reasoning.
    - **Mechanism**:
        - **Grounding the Imagined (GI)**: Identifying imagined objects or activities simulated by the mime through gesture and movement (e.g., flapping arms → a flying bird).
        - **Scene-level**: Temporal reasoning (causal event chains), emotion recognition (nonverbal affective cues), and intent & action understanding (inferring the goals and motivations behind actions).
        - **Global-level**: Working memory (cross-scene information integration), social judgment (comparing behavior against social norms), and Theory of Mind (inferring beliefs, goals, and perspectives).
    - **Novelty**: The first benchmark to systematically incorporate cognitive developmental research into AI benchmark design, covering the full spectrum from perception to higher-order cognition.

2. **Dataset Construction Pipeline**:

    - **Video collection**: YouTube is searched using the keyword "mime"; videos are restricted to 1–10 minutes in length and must carry a Creative Commons license.
    - **Annotation**: Two annotators familiar with the question hierarchy generate approximately 6 scene-level questions, 4 global-level questions, and a corresponding number of grounding questions per video, each accompanied by timestamps.
    - **Verification**: A second annotator independently watches each video and answers the questions; responses are compared against the original annotations, yielding a 97.58% agreement rate.
    - **Filtering criteria**: Videos lacking a narrative, deemed too ambiguous, or containing spoken language are excluded; contested questions are removed.

3. **Evaluation Design**:

    - **Function**: GPT-4o is used as an LLM-as-a-judge to automatically score open-ended responses.
    - **Mechanism**: The judge determines whether a model's answer is semantically equivalent to the annotated reference answer.
    - **Validation**: On a sample of 352 questions, the automatic scorer achieves 92.0% agreement with human raters.

### Loss & Training

- MimeQA is primarily an evaluation benchmark and does not involve a training loss.
- Fine-tuning experiment: Qwen2.5-VL-72B is fine-tuned on 80% of MimeQA; global reasoning improves while grounding the imagined remains poor.
- Cross-dataset transfer: 5-fold cross-validation is used to assess transferability between MimeQA, Social-IQ 2.0, and IntentQA.

## Key Experimental Results

### Main Results

**Accuracy of each model on MimeQA (VL = video + text, L = text only):**

| Model | Avg (VL) | GI | Intent | Emotion | Temporal | ToM | Social Judgment | Working Memory |
|---|---|---|---|---|---|---|---|---|
| Gemini-2.5-Pro | **38.3** | **28.4** | **31.6** | **43.7** | **28.6** | **54.7** | **51.7** | **39.0** |
| GPT-4o | 31.3 | 19.0 | 28.5 | 29.9 | 30.6 | 45.3 | 43.7 | 35.1 |
| Gemini-1.5-Pro | 30.6 | 20.4 | 22.8 | 34.5 | 30.6 | 42.7 | 40.2 | 33.7 |
| VideoLLaMA3 | 22.2 | 7.3 | 13.3 | 34.5 | 13.3 | 41.3 | 31.0 | 22.1 |
| Qwen2.5-VL | 20.1 | 6.6 | 15.8 | 23.6 | 14.3 | 38.7 | 33.3 | 19.4 |
| **Human** | **86.0** | **89.8** | **87.3** | **83.9** | **88.8** | **93.3** | **80.5** | **76.6** |

### Ablation Study

**Fine-tuning effect (Qwen2.5-VL-72B):**

| Condition | Avg | Grounding | Intent | ToM | Working Memory |
|---|---|---|---|---|---|
| Base | 22.5 | 7.1 | 18.8 | 44.4 | 23.5 |
| Fine-tuned | **26.6** | 7.1 | **28.1** | **55.6** | **47.1** |

**Cross-dataset transfer (Qwen2.5-VL-7B):**

| Train → Test | MimeQA Test | Social-IQ Test | IntentQA Test |
|---|---|---|---|
| Train on MimeQA | +3.5% | +1.2% | +2.6% |
| Train on Social-IQ | +0.4% | +1.0% | N/A |
| Train on IntentQA | +1.1% | N/A | +3.7% |

**Videos with vs. without on-screen text:**

| Model | With text | Without text |
|---|---|---|
| GPT-4o | 37.9% | 24.5% |
| Gemini-2.5-Pro | 44.8% | 31.8% |
| Qwen2.5-VL | 24.6% | 15.5% |

### Key Findings

- All VideoLLMs fall far below human performance (best: 38.3% vs. 86.0%), a gap of up to 48 percentage points.
- There is a substantial gap between open-source and closed-source models: open-source models score ~20–22%, while GPT-4o and Gemini-2.5-Pro reach 30–38%.
- Grounding the Imagined is the most challenging category; the best model scores only 28.4% (humans: 89.8%).
- Global-level questions exhibit severe language bias—some models achieve 40%+ on social judgment without watching the video.
- Skills learned from MimeQA transfer well: MimeQA fine-tuning improves Social-IQ by 1.2%, approaching the 1.0% gain from Social-IQ fine-tuning itself.
- Reverse transfer is negligible: Social-IQ fine-tuning improves MimeQA by only 0.4%, demonstrating that MimeQA captures unique nonverbal cognitive capabilities.
- Incorporating pose estimation (PoseC3D) as auxiliary input improves grounding the imagined (2.33% → 6.98%) but degrades higher-level reasoning.

## Highlights & Insights

- **Distinctive problem framing**: Selecting mime performance as the evaluation medium is highly creative—it naturally eliminates linguistic cues and directly addresses the core issue.
- **Cognitive science grounding**: The question hierarchy is rooted in developmental psychology and cognitive science literature rather than being arbitrarily categorized.
- **Reveals a critical blind spot**: Models exhibit extremely poor understanding of "imagined objects" (6–28%), a foundational capability in human communication.
- **Elegant transfer experiment design**: The asymmetric transfer results (MimeQA → Social-IQ effective; reverse ineffective) compellingly demonstrate the benchmark's unique value.
- **In-depth language bias analysis**: Experiments comparing videos with and without on-screen text, as well as the addition of captions, clearly expose models' excessive reliance on language.

## Limitations & Future Work

- The dataset is relatively small (only 101 videos and 806 QA pairs), which may limit statistical significance.
- The benchmark primarily reflects Western mime traditions, limiting cross-cultural generalizability.
- Human annotations may carry subjective bias, as some mime performances are genuinely ambiguous.
- The open-ended QA format relies on an LLM scorer, introducing evaluation noise.
- The paper does not attempt to generate training data from mime videos to substantially boost model performance.
- Only frame-sampling-based methods are evaluated; assessment of models that natively process video is insufficient.
- The trade-off observed with the PoseC3D auxiliary approach suggests that more sophisticated multimodal fusion strategies are needed.

## Related Work & Insights

- MimeQA is complementary to Social-IQ 2.0 and IntentQA, with a focus on purely nonverbal scenarios.
- Unlike the Mimetics dataset (which evaluates action recognition), MimeQA comprehensively assesses social cognition.
- This work inspires a new research direction: language-agnostic multimodal social intelligence.
- The benchmark has close connections to sign language research, cross-cultural communication, and autism assistance applications.
- The findings carry important implications for VideoLLM training strategies, highlighting the need for more video data with nonverbal annotations.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The first work to leverage mime performance for evaluating nonverbal social intelligence; the problem framing and data source selection are highly creative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers multi-model evaluation, fine-tuning, cross-dataset transfer, and detailed error analysis, though the dataset size is limited.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — The narrative is fluent, the theoretical grounding is solid, and the error analysis is vivid and convincing.
- **Value**: ⭐⭐⭐⭐⭐ — Opens a new evaluation dimension for nonverbal social AI with far-reaching implications for advancing genuine social intelligence.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Efficient Transfer Learning for Video-language Foundation Models](../../CVPR2025/video_understanding/efficient_transfer_learning_for_video-language_foundation_models.md)
- [\[CVPR 2026\] UniVBench: Towards Unified Evaluation for Video Foundation Models](../../CVPR2026/video_understanding/univbench_towards_unified_evaluation_for_video_foundation_models.md)
- [\[ICML 2025\] MoMa: Modulating Mamba for Adapting Image Foundation Models to Video Recognition](../../ICML2025/video_understanding/moma_modulating_mamba_for_adapting_image_foundation_models_to_video_recognition.md)
- [\[ICCV 2025\] FlowSeek: Optical Flow Made Easier with Depth Foundation Models and Motion Bases](../../ICCV2025/video_understanding/flowseek_optical_flow_made_easier_with_depth_foundation_models_and_motion_bases.md)
- [\[NeurIPS 2025\] Seeing the Arrow of Time in Large Multimodal Models](seeing_the_arrow_of_time_in_large_multimodal_models.md)

</div>

<!-- RELATED:END -->

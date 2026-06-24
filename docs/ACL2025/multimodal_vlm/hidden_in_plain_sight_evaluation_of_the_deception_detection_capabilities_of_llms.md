---
title: >-
  [Paper Note] Hidden in Plain Sight: Evaluation of the Deception Detection Capabilities of LLMs in Multimodal Settings
description: >-
  [ACL 2025][Multimodal VLM][Deception Detection] This paper systematically evaluates the deception detection capabilities of LLMs and multimodal large models across various modalities such as text, video, and audio. It finds that fine-tuned LLMs achieve SOTA performance in text-based deception detection, whereas multimodal models still exhibit significant deficiencies in utilizing cross-modal cues.
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "Deception Detection"
  - "Large Language Models"
  - "Multimodal"
  - "Zero-shot/Few-shot"
  - "Inference Strategy"
date: 2026-05-08
content_hash: 7e67785c5064db8f
---

# Hidden in Plain Sight: Evaluation of the Deception Detection Capabilities of LLMs in Multimodal Settings

**Conference**: ACL 2025  
**arXiv**: [2506.09424](https://arxiv.org/abs/2506.09424)  
**Code**: None  
**Area**: Multimodal VLM  
**Keywords**: Deception Detection, Large Language Models, Multimodal, Zero-shot/Few-shot, Inference Strategy

## TL;DR

This paper systematically evaluates the deception detection capabilities of LLMs and multimodal large models across various modalities such as text, video, and audio. It finds that fine-tuned LLMs achieve SOTA performance in text-based deception detection, whereas multimodal models still exhibit significant deficiencies in utilizing cross-modal cues.

## Background & Motivation

**Background**: Deception detection is a critical task involving multiple fields such as security, law, and online integrity. Traditional methods rely on hand-crafted feature engineering and small-scale annotated datasets. In recent years, researchers have begun exploring the use of large language models (LLMs) for automated deception detection.

**Limitations of Prior Work**: Existing work mostly focuses on a single dataset or a single modality, lacking a systematic cross-domain and cross-modal evaluation. Furthermore, there is a lack of comprehensive analysis regarding the performance of LLMs under different prompting strategies (such as zero-shot, few-shot, and similarity-based selection), and the impact of non-verbal features (e.g., facial expressions, gestures) on detection performance has not been studied in depth.

**Key Challenge**: While LLMs have demonstrated strong capabilities in natural language understanding, deception detection inherently requires integrating multidimensional information such as textual semantics, non-verbal behaviors, and contextual background. Existing evaluations have failed to reveal the true capability boundaries of LLMs in multimodal deception detection.

**Goal**: To systematically evaluate the deception detection capabilities of open-source and commercial LLMs/LMMs across three datasets from different domains, analyzing the effectiveness of different experimental setups (zero-shot, few-shot, fine-tuning) and prompting strategies (direct label prediction, reasoning generation).

**Key Insight**: The authors select three highly distinct datasets: real-life courtroom trial videos (RLTD), designed interpersonal deception scenarios (MU3D), and deceptive online reviews (OpSpam), covering various deception types from face-to-face communication to pure text.

**Core Idea**: To reveal the potential and bottlenecks of LLMs in deception detection tasks through a comprehensive evaluation covering multiple datasets, modalities, and strategies.

## Method

### Overall Architecture

This paper constructs a systematic evaluation framework: the inputs are samples from three heterogeneous datasets (text transcripts, video, audio). Through various experimental setups (zero-shot, few-shot with random selection, few-shot with similarity-based selection, fine-tuning) and different prompting strategies (direct label prediction, post-hoc reasoning generation), the framework outputs binary classification results (truth/deception). The evaluation covers LLMs such as LLaMA 3.1, Gemma 2, and GPT-4o, as well as multimodal models like LLaVA-NEXT-Video and Qwen2VL.

### Key Designs

1. **多样化数据集选择**:

    - **Function**: Cover different deception scenarios and modalities
    - **Mechanism**: Choose three datasets: RLTD (real-life trial videos from high-stakes scenarios), MU3D (interpersonal deception in controlled experiments, containing video/audio), and OpSpam (deceptive online reviews, pure text). RLTD contains 121 samples, MU3D contains 320 samples, and OpSpam contains 1600 samples. Each dataset represents a different type of deception and available modalities, rendering the evaluation results more generalizable.
    - **Design Motivation**: A single dataset cannot reflect the true complexity of deception detection challenges; only cross-domain evaluation can reveal the general capabilities of the models.

2. **多层次提示策略**:

    - **Function**: Systematically compare the effects of different prompting methods on deception detection performance.
    - **Mechanism**: Zero-shot direct prediction is used as the baseline. In few-shot learning, random selection and semantic similarity-based in-context exemplar selection (sim-top) are respectively adopted. Regarding reasoning strategies, the study compares three approaches: direct label prediction (label-first), post-hoc reasoning generation (label-then-reasoning), and chain-of-thought (CoT) reasoning (reasoning-then-label). Experiments explore 2, 4, 6, 8, and 10 few-shot exemplars, reporting the best results.
    - **Design Motivation**: Different prompting strategies may activate different reasoning patterns of LLMs; systematic comparison helps identify the best practices.

3. **辅助特征与多模态融合评估**:

    - **Function**: Evaluate the contribution of non-verbal features and cross-modal information to detection performance.
    - **Mechanism**: For the video dataset (RLTD), 16 key non-verbal features (such as eyebrow raise, gaze direction, mouth movement, hand movement, etc.) are extracted as auxiliary inputs. For multimodal models, GPT-4o is used to generate video and audio summaries, which are then provided to the model as additional context. Meanwhile, the performance differences between pure text inputs and text + auxiliary feature inputs are compared.
    - **Design Motivation**: Humans comprehensively utilize verbal and non-verbal cues in deception detection; evaluating whether LMMs can similarly leverage multimodal information is of paramount importance.

### Loss & Training

Fine-tuning experiments are conducted on A6000 GPUs with a learning rate of 4e-5 for 20 epochs. LLaMA 3.1 and Gemma 2 are fine-tuned, while GPT-4o is accessed via API calls. All results are averaged over 3 random seeds.

## Key Experimental Results

### Main Results

| Dataset | Model | Setting | Accuracy | F1 |
|--------|------|------|----------|-----|
| RLTD | GPT-4o | Few-shot (sim-top) | 72.98 | 73.77 |
| RLTD | LLaMA 3.1 | Fine-tuned | **Best** | SOTA |
| RLTD | Gemma 2 | Fine-tuned | **Best** | SOTA |
| MU3D | GPT-4o | Few-shot | - | ~60 |
| MU3D | LLaMA 3.1 | Few-shot (Reasoning) | - | 56.15 |
| OpSpam | GPT-4o | Few-shot | - | 67.58 |
| OpSpam | GPT-4o | Few-shot+Reasoning | - | 61.04 |

### Ablation Study

| Strategy Comparison | RLTD F1 | MU3D F1 | OpSpam F1 | Description |
|---------|---------|---------|-----------|------|
| Direct Label (Few-shot) | 71.39 | ~55 | 67.58 | GPT-4o baseline |
| Label + Reasoning (Few-shot) | 69.63 | ~56 | 61.04 | Reasoning decreases performance |
| Random Exemplar Selection | Baseline | Baseline | Baseline | Few-shot baseline |
| Similarity Exemplar Selection | +3.19% | +4.69% | +5.54% | Consistent improvement |
| Post-hoc Reasoning vs. CoT | Better than CoT | Better than CoT | Better than CoT | Post-hoc strategy is superior |

### Key Findings

- **Fine-tuned LLMs achieve SOTA in text-based deception detection**: Fine-tuned LLaMA 3.1 and Gemma 2 outperform traditional baseline methods across all three datasets.
- **Multimodal models fail to effectively leverage cross-modal cues**: LMMs (e.g., LLaVA-NEXT-Video, Qwen2VL) do not perform significantly better on the video dataset than text-only LLMs, indicating that current LMMs have limited capacity in understanding visual cues associated with human deceptive behaviors.
- **Direct label prediction generally outperforms reasoning generation**: Although post-hoc reasoning generation improves interpretability, it slightly reduces the F1 score in most settings. Chain-of-thought (CoT) reasoning performs even worse because the model considers both truthful and deceptive cues during reasoning, leading to ambiguous judgments.
- **Semantic similarity-based few-shot selection consistently outperforms random selection**: The sim-top strategy yields an average improvement of 3-5% across all three datasets.
- **The number of few-shot exemplars is not "the more the better"**: GPT-4o's performance begins to degrade as the number of exemplars increases, likely because the increased prompt complexity interferes with reasoning.

## Highlights & Insights

- **The comparative analysis of post-hoc reasoning vs. Chain-of-thought (CoT)** is highly compelling: It is found that predicting the label before generating reasoning performs better than reasoning before predicting the label. This is because CoT prompts the model to list both supporting and opposing cues, resulting in an ambiguous final judgment. This insight is also valuable for other classification tasks.
- **In-depth analysis of data source bias**: The authors specifically discuss in the appendix that stylistic differences between Mechanical Turk-generated fake reviews and real reviews in the OpSpam dataset may introduce bias, and conduct additional validation on the Prolific dataset, demonstrating a rigorous experimental attitude.
- **The limited contribution of non-verbal features** reveals fundamental limitations of current models in understanding human micro-expressions and body language, directing the design of next-generation multimodal models.

## Limitations & Future Work

- **Only English datasets are supported**: The study fails to capture differences in deception patterns across different languages and cultural backgrounds, where deception cues may vary radically.
- **Deception detection for AI-generated content is not addressed**: With the propagation of deepfakes and AI-generated texts, this research gap needs to be filled as soon as possible.
- **Limited dataset scale**: RLTD contains only 121 samples, which may lead to insufficient statistical confidence.
- **High experimental costs**: GPT-4o experiments cost approximately $300, and open-source models require substantial GPU resources.
- **Improvement ideas**: Fine-grained visual understanding (such as Action Unit encoding, micro-expression recognition) can be integrated with LLMs to build stronger multimodal deception detection pipelines.

## Related Work & Insights

- **vs. Traditional Deception Detection Methods**: Traditional methods rely on hand-crafted features (e.g., LIWC lexical features, vocal acoustics) and specialized classifiers. This work demonstrates that LLM fine-tuning can achieve SOTA without hand-crafted feature engineering, though traditional method still hold advantages in leveraging multimodal cues.
- **vs. General LMMs like GPT-4V/Gemini**: Although these models perform excellently in visual understanding, they still fall short in deception detection, a task that requires deep comprehension of human behavior patterns, showing that visual understanding capability $\neq$ behavioral understanding capability.

## Rating

- Novelty: ⭐⭐⭐ The evaluation framework is comprehensive, but methodological innovation is limited; the main contribution lies in the systematic benchmarking.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across multiple datasets, models, and strategies, with a large number of auxiliary experiments in the appendix.
- Writing Quality: ⭐⭐⭐⭐ Well-organized, deep analysis, and an honest and thorough discussion of limitations.
- Value: ⭐⭐⭐⭐ Provides an important reference benchmark for the application of LLMs in the field of deception detection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Insight Over Sight: Exploring the Vision-Knowledge Conflicts in Multimodal LLMs](conflictvis_vision_knowledge_conflict.md)
- [\[ACL 2026\] Dynamic Emotion and Personality Profiling for Multimodal Deception Detection](../../ACL2026/multimodal_vlm/dynamic_emotion_and_personality_profiling_for_multimodal_deception_detection.md)
- [\[NeurIPS 2025\] MME-VideoOCR: Evaluating OCR-Based Capabilities of Multimodal LLMs in Video Scenarios](../../NeurIPS2025/multimodal_vlm/mme-videoocr_evaluating_ocr-based_capabilities_of_multimodal_llms_in_video_scena.md)
- [\[ACL 2025\] FlagEvalMM: A Flexible Framework for Comprehensive Multimodal Model Evaluation](flagevalmm_a_flexible_framework_for_comprehensive_multimodal_model_evaluation.md)
- [\[ACL 2025\] Unsolvable Problem Detection: Evaluating Trustworthiness of Large Multimodal Models](unsolvable_problem_detection.md)

</div>

<!-- RELATED:END -->

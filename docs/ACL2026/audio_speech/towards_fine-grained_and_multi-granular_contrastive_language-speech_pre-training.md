---
title: >-
  [Paper Note] Towards Fine-Grained and Multi-Granular Contrastive Language-Speech Pre-training
description: >-
  [ACL 2026][Audio & Speech][Speech style modeling] This paper proposes the FCaps large-scale dataset (47k hours of speech, 19M fine-grained annotations) and the CLSP contrastive learning model. Through an end-to-end annot…
tags:
  - "ACL 2026"
  - "Audio & Speech"
  - "Speech style modeling"
  - "contrastive learning pre-training"
  - "fine-grained annotation"
  - "speech-text alignment"
  - "paralinguistics"
date: 2026-05-08
content_hash: d21ca8bb1c1b3619
---

# Towards Fine-Grained and Multi-Granular Contrastive Language-Speech Pre-training

**Conference**: ACL 2026  
**arXiv**: [2601.03065](https://arxiv.org/abs/2601.03065)  
**Code**: [GitHub](https://github.com/yfyeung/CLSP)  
**Area**: Audio Speech  
**Keywords**: Speech style modeling, contrastive learning pre-training, fine-grained annotation, speech-text alignment, paralinguistics

## TL;DR

This paper proposes the FCaps large-scale dataset (47k hours of speech, 19M fine-grained annotations) and the CLSP contrastive learning model. Through an end-to-end annotation pipeline and fine-grained multi-granular contrastive supervision, it achieves the first speech-text alignment model capable of unified representation for both global and fine-grained speech styles.

## Background & Motivation

**Background**: Speech style conveys rich paralinguistic information, including intrinsic speaker characteristics (gender, age, accent) and situational features (speaking rate, emotion, expressiveness). Existing speech-text representation learning methods typically rely on coarse-grained labels or task-specific supervision, failing to capture the fine-grained temporal structure of speech style.

**Limitations of Prior Work**: Existing speech style annotation datasets mainly utilize cascaded annotation pipelines — first labeling speech with discrete labels, then using Large Language Models (LLMs) to rewrite labels into natural language descriptions. This approach suffers from a fundamental information bottleneck: intermediate discrete labels compress rich, continuous, and time-varying paralinguistic information into limited predefined categories, leading to severe information loss and semantic bias.

**Key Challenge**: Fine-grained speech style modeling requires high-quality, large-scale free-text descriptions. However, existing methods either rely on manual annotation (high cost, poor consistency) or use cascaded pipelines (introducing error propagation and information loss).

**Goal**: (1) Construct a large-scale end-to-end fine-grained speech style annotation dataset to avoid the information bottleneck of cascaded pipelines; (2) Train a contrastive learning model capable of unified multi-granular speech style representation.

**Key Insight**: Utilize the latest multimodal annotation models (Qwen3-Omni) to generate fine-grained descriptions directly from audio, bypassing the intermediate step of discrete labels, and ensure annotation quality through an agentic verification process.

**Core Idea**: End-to-end annotation pipeline + fine-grained multi-granular contrastive learning to eliminate information bottlenecks and achieve a unified speech-text representation from global to fine-grained levels.

## Method

### Overall Architecture

The framework is divided into data construction and model training. For data, the FCaps dataset is constructed via an end-to-end pipeline, containing FCaps-Emilia (46,787 hours, 18M fine-grained annotations) and FCaps-PSCBase (267 hours, 140k global and 930k fine-grained annotations). For the model, CLSP adopts a dual-encoder architecture (SPEAR-XLarge speech encoder + RoBERTa text encoder), trained via two-stage curriculum learning: the first stage performs standard contrastive learning on large-scale fine-grained data, and the second stage introduces multi-positive contrastive learning to achieve cross-granular generalization.

### Key Designs

1.  **End-to-end Annotation Pipeline**:
    - **Function**: Generate high-quality fine-grained speech style descriptions directly from audio, avoiding information loss from cascaded pipelines.
    - **Mechanism**: Qwen3-Omni-30B is used as a detailed annotator, taking speech segments as direct input to generate fine-grained descriptions. User prompts constrain the output to focus on speaker style (suppressing transcription and environmental sound descriptions). Multiple positive views are obtained by generating descriptions multiple times with different random seeds. A Qwen3-30B reasoning model then acts as a verification agent, filtering low-quality annotations based on a predefined checklist (e.g., presence of background noise descriptions, missing statements, or transcriptions without style descriptions).
    - **Design Motivation**: Discrete labels in cascaded pipelines are information bottlenecks; end-to-end generation is conditioned directly on audio, preserving complete paralinguistic information. Multi-positive generation is more reliable than pure text rewriting because each annotation is grounded in the original audio signal.

2.  **Fine-grained Multi-granular Contrastive Learning**:
    - **Function**: Learn an embedding space that uniformly represents speech style across different granularities.
    - **Mechanism**: The first stage uses standard symmetric InfoNCE loss on large-scale fine-grained data: $\mathcal{L} = -\frac{1}{2N}\sum_{i=1}^{N}(\log\frac{\exp(\mathbf{s}_i \cdot \mathbf{t}_{Fi}/\tau)}{\sum_j \exp(\mathbf{s}_i \cdot \mathbf{t}_{Fj}/\tau)} + \text{reverse})$. The second stage utilizes multi-positive InfoNCE, pairing each speech sample with two texts (one global and one fine-grained, or two different fine-grained ones). Probability mass is assigned via a soft target distribution $D_{i,j}$ ($\lambda = 0.5$), and the loss is cross-entropy: $\mathcal{L} = \frac{1}{2}(\mathrm{CE}(\mathbf{L}/\tau, \mathbf{D}) + \mathrm{CE}(\mathbf{L}^\top/\tau, \mathbf{D}'))$.
    - **Design Motivation**: The two-stage curriculum transitions from pure fine-grained alignment to cross-granular generalization. Stage one establishes precise fine-grained correspondence, while stage two achieves cross-granular consistency through mixed global and fine-grained training.

3.  **Dynamic Task Scheduler**:
    - **Function**: Balance cross-granular generalization and fine-grained discrimination in the second stage.
    - **Mechanism**: One of two tasks is randomly sampled at each training step — Task 1 (global + fine-grained pairing) or Task 2 (two different fine-grained pairings). The sampling probability $p_t$ decays linearly from $p_0 = 0.95$ to $p_{min} = 0.50$ over $T = 10000$ steps: $p_t = \max(p_{min}, p_0 - \frac{t}{T}(p_0 - p_{min}))$.
    - **Design Motivation**: Early training focuses on cross-granular alignment (Task 1 dominates), while later training increases fine-grained discrimination (Task 2 share increases), achieving progressive learning.

### Loss & Training

CLSP comprises 724M parameters (SPEAR-XLarge 599M + RoBERTa 125M) and was trained on 8 A100 80GB GPUs. Stage one involved 1.2M steps, followed by 4k steps of fine-tuning in stage two. ScaledAdam optimizer and Eden learning rate scheduler were used, with peak learning rates of 0.045 and 0.001 respectively. The temperature parameter $\tau$ is learnable.

## Key Experimental Results

### Main Results

| Task | Metric | CLSP | Prev. SOTA (ParaCLAP) | Gain |
|------|------|------|-------------------|------|
| Global Retrieval S→T | R@1 | 45.6 | 2.1 | +43.5 |
| Global Retrieval T→S | R@1 | 40.3 | 0.4 | +39.9 |
| Fine-grained Retrieval S→T | R@1 | 68.1 | 1.2 | +66.9 |
| Fine-grained Retrieval T→S | R@1 | 67.2 | 1.2 | +66.0 |
| Zero-shot Emotion (IEMOCAP) | WA/UA | 57.2/56.1 | 46.1/46.5 | +11.1/+9.6 |
| Zero-shot Gender (RAVDESS) | WA/UA | 100.0/100.0 | 99.2/99.2 | +0.8 |
| Style Similarity (Intrinsic) | Pearson r | 0.893 | 0.663 | +0.230 |
| Style Similarity (Situational) | Pearson r | 0.903 | 0.323 | +0.580 |

### Ablation Study

| Configuration | Description | Effect |
|------|------|------|
| End-to-end vs Cascaded | Correctness/Coverage/Naturalness | 4.42/4.55/4.92 vs 3.30/3.10/4.15 |
| Static vs Dynamic | Task sampling strategy | Dynamic superior to Static |
| $\lambda=0.5$ vs others | Multi-positive weight allocation | 0.5 is optimal |

### Key Findings

- CLSP significantly outperforms existing methods across all tasks, with massive improvements in retrieval tasks (R@1 increased from single digits to 40-68%).
- End-to-end annotation quality is comprehensively superior to cascaded annotation: Correctness +1.12, Coverage +1.45, Naturalness +0.77.
- Speech style similarity scores show high consistency with human judgment (Pearson r > 0.88), particularly exceeding prior art on situational features (0.903 vs ParaCLAP's 0.323).
- Strong zero-shot classification performance, with emotion recognition WA reaching 57.2% and gender identification at 100%, indicating the learned representations effectively encode paralinguistic information.

## Highlights & Insights

- FCaps is currently the largest fine-grained speech style annotation dataset (47k hours, 19M annotations), filling a critical data gap.
- The design of the end-to-end annotation pipeline is valuable for broader application — generating descriptions directly from raw signals avoids information bottlenecks, while agentic verification ensures quality.
- The progressive two-stage curriculum learning strategy, transitioning from fine-grained to cross-granular, is effective; just 4k steps of fine-tuning significantly enhanced cross-granular capabilities.
- CLSP can serve as a speech style evaluator; its high correlation with human judgment makes it a potential replacement for expensive subjective assessments.

## Limitations & Future Work

- Currently supports only English speech; cross-lingual speech style modeling remains to be explored.
- The end-to-end pipeline depends on the quality of Qwen3-Omni; biases inherent in this model may propagate to the annotations.
- The text encoder uses RoBERTa-base (125M); larger text encoders might further improve performance.
- Temporal alignment in fine-grained annotations was not explored — current annotations describe style changes within an utterance but do not provide precise timestamps.

## Related Work & Insights

- **vs ParaCLAP (Jing et al.)**: While ParaCLAP focuses on emotion-centric supervision, CLSP covers broader paralinguistic dimensions through fine-grained and multi-granular supervision.
- **vs GLAP (Dinkel et al.)**: GLAP uses transcript pairings for word-level supervision, whereas CLSP uses style descriptions for paralinguistic-level supervision.
- **vs CapSpeech (Wang et al.)**: CapSpeech uses a cascaded annotation pipeline; CLSP's end-to-end pipeline avoids information bottlenecks and yields higher annotation quality.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The end-to-end pipeline and multi-granular contrastive learning are significant innovations, and the FCaps dataset is a major contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage includes annotation quality assessment, four types of downstream tasks, and correlation analysis with human judgment.
- Writing Quality: ⭐⭐⭐⭐ Methodology is clearly described and the dataset construction process is well-documented.
- Value: ⭐⭐⭐⭐⭐ Both dataset and model are open-sourced, promising a broad impact on speech style modeling and evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] UniSRM: A Unified Speech Reward Model for Fine-Grained Speech Evaluation](unisrm_a_unified_speech_reward_model_for_reasoning-based_fine-grained_assessment.md)
- [\[ACL 2026\] Temporal Contrastive Decoding: A Training-Free Method for Large Audio-Language Models](temporal_contrastive_decoding_a_training-free_method_for_large_audio-language_mo.md)
- [\[ACL 2026\] SegTune: Structured and Fine-Grained Control for Song Generation](segtune_structured_and_fine-grained_control_for_song_generation.md)
- [\[ICML 2026\] MECAT: A Multi-Experts Constructed Benchmark for Fine-Grained Audio Understanding Tasks](../../ICML2026/audio_speech/mecat_a_multi-experts_constructed_benchmark_for_fine-grained_audio_understanding.md)
- [\[AAAI 2026\] End-to-end Contrastive Language-Speech Pretraining Model For Long-form Spoken Question Answering](../../AAAI2026/audio_speech/end-to-end_contrastive_language-speech_pretraining_model_for_long-form_spoken_qu.md)

</div>

<!-- RELATED:END -->

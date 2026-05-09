---
title: >-
  [Paper Note] Towards Fine-Grained and Multi-Granular Contrastive Language-Speech Pre-training
description: >-
  [ACL 2026][Audio & Speech][Speech Style Modeling] This paper proposes the FCaps large-scale dataset (47k hours of speech, 19M fine-grained annotations) and the CLSP contrastive learning model. Through an end-to-end annotation pipeline and fine-grained multi-granular contrastive supervision, it presents the first speech-text alignment model capable of uniformly representing both global and fine-grained speaking styles.
tags:
  - ACL 2026
  - "Audio & Speech"
  - Speech Style Modeling
  - Contrastive Learning Pre-training
  - Fine-Grained Annotation
  - Speech-Text Alignment
  - Paralinguistics
date: 2026-05-08
content_hash: 88309445a810b97d
---

# Towards Fine-Grained and Multi-Granular Contrastive Language-Speech Pre-training

**Conference**: ACL 2026
**arXiv**: [2601.03065](https://arxiv.org/abs/2601.03065)
**Code**: [GitHub](https://github.com/yfyeung/CLSP)
**Area**: Audio & Speech
**Keywords**: Speech Style Modeling, Contrastive Learning Pre-training, Fine-Grained Annotation, Speech-Text Alignment, Paralinguistics

## TL;DR

This paper proposes the FCaps large-scale dataset (47k hours of speech, 19M fine-grained annotations) and the CLSP contrastive learning model. Through an end-to-end annotation pipeline and fine-grained multi-granular contrastive supervision, it presents the first speech-text alignment model capable of uniformly representing both global and fine-grained speaking styles.

## Background & Motivation

**Background**: Speaking style conveys rich paralinguistic information, encompassing intrinsic speaker characteristics (gender, age, accent) and contextual features (speech rate, emotion, expressiveness). Existing speech-text representation learning methods typically rely on coarse-grained labels or task-specific supervision, failing to capture the fine-grained temporal structure of speaking style.

**Limitations of Prior Work**: Existing speech style annotation datasets predominantly adopt cascaded annotation pipelines—first annotating speech with discrete labels, then using large language models to rewrite those labels into natural language descriptions. This approach introduces a fundamental information bottleneck: the intermediate discrete labels compress rich, continuous, time-varying paralinguistic information into a limited set of predefined categories, leading to severe information loss and semantic drift.

**Key Challenge**: Fine-grained speech style modeling requires high-quality, large-scale free-text descriptions, yet existing methods either rely on human annotation (costly and inconsistent) or cascaded pipelines (introducing error propagation and information loss).

**Goal**: (1) Construct a large-scale end-to-end fine-grained speech style annotation dataset that avoids the information bottleneck of cascaded pipelines; (2) Train a contrastive learning model capable of uniformly representing speech styles at multiple granularities.

**Key Insight**: Leveraging a recent multimodal annotation model (Qwen3-Omni) to directly generate fine-grained descriptions from audio, bypassing the discrete label intermediate step, while ensuring annotation quality through an agent-based verification process.

**Core Idea**: An end-to-end annotation pipeline combined with fine-grained multi-granular contrastive learning eliminates the information bottleneck and enables unified speech-text representation from global to fine-grained granularity.

## Method

### Overall Architecture

The overall framework consists of two components: data construction and model training. On the data side, the FCaps dataset is built via an end-to-end pipeline, comprising FCaps-Emilia (46,787 hours, 18M fine-grained annotations) and FCaps-PSCBase (267 hours, 140k global annotations + 930k fine-grained annotations). On the model side, CLSP adopts a dual-encoder architecture (SPEAR-XLarge speech encoder + RoBERTa text encoder) and is trained contrastively via two-stage curriculum learning: Stage 1 performs standard contrastive learning on large-scale fine-grained data, while Stage 2 introduces multi-positive contrastive learning to achieve cross-granularity generalization.

### Key Designs

1. **End-to-End Annotation Pipeline**:

    - Function: Directly generates high-quality fine-grained speech style descriptions from audio, avoiding the information loss of cascaded pipelines.
    - Mechanism: Qwen3-Omni-30B is used as the detailed annotator, taking speech segments directly as input to generate fine-grained descriptions. User prompts constrain the output to focus on speaker style (suppressing transcription content and environmental sound descriptions). Multiple generations with different random seeds are produced for the same speech segment to obtain multiple positive views. A Qwen3-30B reasoning model then acts as a verification agent, filtering low-quality annotations against a predefined checklist (e.g., whether the annotation includes background/environmental noise descriptions, missing statements, or transcription content without style description).
    - Design Motivation: Discrete labels in cascaded pipelines constitute an information bottleneck. End-to-end generation, conditioned directly on audio, preserves complete paralinguistic information. Multi-positive generation is more reliable than pure text paraphrasing, as each annotation is grounded in the original audio signal.

2. **Fine-Grained Multi-Granular Contrastive Learning**:

    - Function: Learns an embedding space that uniformly represents speech styles at different granularities.
    - Mechanism: Stage 1 uses standard symmetric InfoNCE loss to train on large-scale fine-grained data: $\mathcal{L} = -\frac{1}{2N}\sum_{i=1}^{N}(\log\frac{\exp(\mathbf{s}_i \cdot \mathbf{t}_{Fi}/\tau)}{\sum_j \exp(\mathbf{s}_i \cdot \mathbf{t}_{Fj}/\tau)} + \text{reverse})$. Stage 2 uses multi-positive InfoNCE, pairing each speech sample with two texts (one global and one fine-grained, or two different fine-grained), assigning probability mass via a soft target distribution $D_{i,j}$ ($\lambda = 0.5$), with loss defined as cross-entropy $\mathcal{L} = \frac{1}{2}(\mathrm{CE}(\mathbf{L}/\tau, \mathbf{D}) + \mathrm{CE}(\mathbf{L}^\top/\tau, \mathbf{D}'))$.
    - Design Motivation: The two-stage curriculum progressively transitions from pure fine-grained alignment to cross-granularity generalization. Stage 1 establishes precise fine-grained correspondence, while Stage 2 achieves cross-granularity consistency through mixed global and fine-grained training.

3. **Dynamic Task Scheduler**:

    - Function: Balances cross-granularity generalization and fine-grained discrimination in Stage 2.
    - Mechanism: At each training step, one of two tasks is sampled at random—Task 1 (global + fine-grained pairing) or Task 2 (two different fine-grained pairings). The sampling probability $p_t$ decreases linearly from $p_0 = 0.95$ to $p_{min} = 0.50$ over $T = 10000$ steps: $p_t = \max(p_{min}, p_0 - \frac{t}{T}(p_0 - p_{min}))$.
    - Design Motivation: Early training emphasizes cross-granularity alignment (Task 1 dominant); later training increases fine-grained discrimination (Task 2 proportion rises), enabling progressive learning.

### Loss & Training

CLSP has 724M parameters in total (SPEAR-XLarge 599M + RoBERTa 125M) and is trained on 8× A100 80GB GPUs. Stage 1 runs for 1.2M steps; Stage 2 fine-tunes for 4k steps. The ScaledAdam optimizer and Eden learning rate scheduler are used, with peak learning rates of 0.045 and 0.001 respectively. The temperature parameter $\tau$ is learnable.

## Key Experimental Results

### Main Results

| Task | Metric | Ours (CLSP) | Prev. SOTA (ParaCLAP) | Gain |
|------|--------|-------------|----------------------|------|
| Global Retrieval S→T | R@1 | 45.6 | 2.1 | +43.5 |
| Global Retrieval T→S | R@1 | 40.3 | 0.4 | +39.9 |
| Fine-Grained Retrieval S→T | R@1 | 68.1 | 1.2 | +66.9 |
| Fine-Grained Retrieval T→S | R@1 | 67.2 | 1.2 | +66.0 |
| Zero-Shot Emotion (IEMOCAP) | WA/UA | 57.2/56.1 | 46.1/46.5 | +11.1/+9.6 |
| Zero-Shot Gender (RAVDESS) | WA/UA | 100.0/100.0 | 99.2/99.2 | +0.8 |
| Style Similarity (Intrinsic) | Pearson r | 0.893 | 0.663 | +0.230 |
| Style Similarity (Contextual) | Pearson r | 0.903 | 0.323 | +0.580 |

### Ablation Study

| Configuration | Description | Effect |
|---------------|-------------|--------|
| End-to-End vs. Cascaded Annotation | Correctness / Coverage / Naturalness | 4.42/4.55/4.92 vs. 3.30/3.10/4.15 |
| Static vs. Dynamic Scheduling | Task sampling strategy | Dynamic outperforms static |
| $\lambda=0.5$ vs. others | Multi-positive weight allocation | 0.5 is optimal |

### Key Findings

- CLSP substantially outperforms existing methods across all tasks, with particularly large improvements on retrieval tasks (R@1 from single digits to 40–68%).
- End-to-end annotation quality comprehensively exceeds cascaded annotation: correctness +1.12, coverage +1.45, naturalness +0.77.
- CLSP achieves high agreement with human judgments on speech style similarity scoring (Pearson r > 0.88), especially on contextual features (0.903 vs. ParaCLAP's 0.323), far surpassing existing methods.
- Zero-shot classification performance is strong: emotion recognition WA reaches 57.2% and gender recognition achieves 100%, indicating that the learned representations encode paralinguistic information effectively.

## Highlights & Insights

- FCaps is currently the largest fine-grained speech style annotation dataset (47k hours, 19M annotations), filling a critical data gap in the field.
- The design philosophy of the end-to-end annotation pipeline is broadly applicable—generating descriptions directly from raw signals avoids information bottlenecks, while agent-based verification ensures quality.
- The two-stage curriculum learning strategy, progressively transitioning from fine-grained to cross-granularity training, is effective: only 4k steps of fine-tuning substantially improve cross-granularity capability.
- CLSP can serve as a speech style evaluator; its high correlation with human judgments makes it a promising substitute for costly subjective evaluations.

## Limitations & Future Work

- The current model supports English speech only; cross-lingual speech style modeling remains to be explored.
- The end-to-end annotation pipeline depends on the quality of Qwen3-Omni, and biases inherent in that model may propagate into the annotations.
- The text encoder uses RoBERTa-base (125M); a larger text encoder may further improve performance.
- Temporal alignment within fine-grained annotations is not explored—current annotations describe intra-utterance style variation but do not provide precise timestamps.

## Related Work & Insights

- **vs. ParaCLAP (Jing et al.)**: ParaCLAP focuses on emotion-centric supervision, whereas CLSP covers a broader range of paralinguistic dimensions through fine-grained multi-granular supervision.
- **vs. GLAP (Dinkel et al.)**: GLAP uses transcribed text pairings to provide lexical-level supervision, while CLSP uses style descriptions to provide paralinguistic-level supervision.
- **vs. CapSpeech (Wang et al.)**: CapSpeech relies on a cascaded annotation pipeline; CLSP's end-to-end pipeline avoids the information bottleneck and achieves higher annotation quality.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The end-to-end annotation pipeline and multi-granular contrastive learning are both significant innovations, and the FCaps dataset constitutes a major contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Annotation quality evaluation, four downstream task categories, and correlation analysis with human judgments are all comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Method descriptions are clear and the dataset construction process is detailed.
- Value: ⭐⭐⭐⭐⭐ Both the dataset and model are open-sourced, with broad impact on speech style modeling and evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Temporal Contrastive Decoding: A Training-Free Method for Large Audio-Language Models](temporal_contrastive_decoding_a_training-free_method_for_large_audio-language_mo.md)
- [\[AAAI 2026\] End-to-end Contrastive Language-Speech Pretraining Model For Long-form Spoken Question Answering](../../AAAI2026/audio_speech/end-to-end_contrastive_language-speech_pretraining_model_for_long-form_spoken_qu.md)
- [\[CVPR 2026\] Unlocking Strong Supervision: A Data-Centric Study of General-Purpose Audio Pre-Training Methods](../../CVPR2026/audio_speech/unlocking_strong_supervision_a_data-centric_study_of_general-purpose_audio_pre-t.md)
- [\[ACL 2026\] Do We Need Distinct Representations for Every Speech Token? Unveiling and Exploiting Redundancy in Large Speech Language Models](do_we_need_distinct_representations_for_every_speech_token_unveiling_and_exploit.md)
- [\[ACL 2026\] SpeakerSleuth: Can Large Audio-Language Models Judge Speaker Consistency across Multi-turn Dialogues?](speakersleuth_can_large_audio-language_models_judge_speaker_consistency_across_m.md)

</div>

<!-- RELATED:END -->

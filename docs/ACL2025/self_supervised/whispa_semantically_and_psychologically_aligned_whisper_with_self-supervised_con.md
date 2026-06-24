---
title: >-
  [Paper Note] WhiSPA: Semantically and Psychologically Aligned Whisper with Self-Supervised Contrastive and Student-Teacher Learning
description: >-
  [ACL 2025][Self-Supervised Learning][Speech-Text Alignment] Proposes WhiSPA, which aligns the latent space of the Whisper audio encoder with SBERT semantic representations and psychological dimensions (emotion, personality) through contrastive learning, eliminating the dependency on an additional text LM in speech processing and reducing error by 73-84% on psychological evaluation tasks.
tags:
  - "ACL 2025"
  - "Self-Supervised Learning"
  - "Speech-Text Alignment"
  - "Whisper"
  - "Contrastive Learning"
  - "Psychological Representation"
  - "Sentiment Analysis"
date: 2026-05-08
content_hash: 5d6d429f4623c225
---

# WhiSPA: Semantically and Psychologically Aligned Whisper with Self-Supervised Contrastive and Student-Teacher Learning

**Conference**: ACL 2025  
**arXiv**: [2501.16344](https://arxiv.org/abs/2501.16344)  
**Code**: [https://github.com/humanlab/WhiSPA](https://github.com/humanlab/WhiSPA)  
**Area**: Self-Supervised Learning / Speech Understanding  
**Keywords**: Speech-Text Alignment, Whisper, Contrastive Learning, Psychological Representation, Sentiment Analysis

## TL;DR
Proposes WhiSPA, which aligns the latent space of the Whisper audio encoder with SBERT semantic representations and psychological dimensions (emotion, personality) through contrastive learning, eliminating the dependency on an additional text LM in speech processing and reducing error by 73-84% on psychological evaluation tasks.

## Background & Motivation
**Background**: The standard pipeline for speech processing is Whisper (speech $\rightarrow$ text) + SBERT/LM (text $\rightarrow$ semantic embedding), which incurs redundant computation due to the requirement of two LMs. Whisper contains internal language models, but its latent representations lack deep semantic and psychological information.

**Limitations of Prior Work**: (a) Speech encoders are significantly inferior to text LMs in psychological tasks such as emotion recognition and personality assessment; (b) the pipeline of two LMs causes computational wastage; (c) fusion architectures (e.g., co-attention) require task-specific designs.

**Key Challenge**: Speech contains acoustic information (intonation, rhythm) that is unavailable in text, yet the semantic understanding capabilities of existing speech encoders are far weaker than those of text LMs.

**Goal**: How can a speech encoder be designed to directly produce semantic and psychological representations that are as rich as those of text LMs?

**Key Insight**: Directly align the latent spaces—using SBERT embeddings as the teacher and Whisper embeddings as the student, brought closer via contrastive learning.

**Core Idea**: Whisper + NCE contrastive loss $\rightarrow$ aligned to SBERT semantics + psychological dimensions = psychological speech representation without the need for a text LM.

## Method

### Overall Architecture
WhiSPA adopts a student-teacher paradigm: Student = Whisper-tiny encoder-decoder (applying mean pooling on the final hidden state of the decoder followed by a learnable projection layer), Teacher = SBERT-384 + PsychEmb (10-dimensional psychological features: valence, arousal, Big Five personality traits, and anger/anxiety/depression). The embedding spaces of both are aligned using a contrastive loss.

### Key Designs

1. **Semantic Alignment (WhiSA)**:

    - **Function**: Aligns Whisper audio embeddings with SBERT text embeddings
    - **Mechanism**: Two types of alignment loss—Cosine Similarity Loss $\mathcal{L}^{CS} = 1 - \text{sim}(\mathbf{A}_i, \mathbf{T}_i)$ and NCE Contrastive Loss $\mathcal{L}^{NCE} = -\log \frac{\exp(\text{sim}/\tau)}{\sum_b \exp(\text{sim}/\tau)}$
    - **Design Motivation**: NCE not only pulls positive pairs closer but also pushes negative pairs apart, learning a more structured representation space.

2. **Psychological Alignment (WhiSPA)**:

    - **Function**: Injects 10-dimensional psychological features (PsychEmb) on top of semantic alignment.
    - **Mechanism**: Two injection methods—(a) WhiSPA-384r: directly replaces the first 10 dimensions of the SBERT embeddings; (b) WhiSPA-394: concatenates PsychEmb to the SBERT embeddings ($384+10=394$), adding a learnable projection matrix on the Whisper side.
    - **Design Motivation**: Psychological dimensions (valence, arousal, personality) are information inherently present in speech but difficult for pure semantic models to capture.

3. **Self-Supervised PsychEmb Features**:

    - **Function**: Extracts 10-dimensional psychological scalar values from text using a pre-trained dictionary.
    - **Mechanism**: Covers three psychological levels—state (valence, arousal), predisposition (anger, anxiety, depression), and trait (Big Five personality).
    - **Design Motivation**: Fully self-supervised psychological features obtained without manual annotation.

### Loss & Training
The model is trained using NCE contrastive loss (temperature $\tau=0.1$) on over 500,000 speech segments, with data sourced from the WTC (World Trade Center) and HiTOP (Hierarchical Taxonomy of Psychopathology) datasets.

## Key Experimental Results

### Main Results (HiTOP Dataset, Self-Supervised Psychological Dimension Prediction $r$)

| Model | Valence | Arousal | Openness | Agreeableness | Neuroticism |
|------|---------|---------|----------|---------------|-------------|
| WhiSPA-394 | **0.76** | **0.84** | 0.72 | **0.79** | **0.82** |
| WhiSPA-384r | 0.78 | 0.85 | 0.74 | 0.79 | 0.79 |
| Whisper-384 | 0.71 | 0.82 | 0.69 | 0.76 | 0.78 |
| SBERT-384 (Text) | 0.69 | 0.81 | 0.73 | 0.75 | 0.77 |
| HuBERT | 0.66 | 0.73 | 0.67 | 0.57 | 0.70 |

### Ablation Study

| Configuration | Description |
|------|------|
| WhiSPA > WhiSA | Adding psychological dimensions consistently improves performance |
| NCE > CS | NCE contrastive loss outperforms cosine similarity loss |
| WhiSPA ≈ Whisper+SBERT | Adding SBERT on downstream tasks yields almost no extra benefit |

### Key Findings
- **WhiSPA Outperforms Textual SBERT**: WhiSPA's audio embeddings outperform SBERT's text embeddings across multiple psychological dimensions, demonstrating successful alignment and showing that audio preserves additional acoustic information.
- **No More Need for a Two-LM Pipeline**: Adding SBERT on top of WhiSPA yielded almost no improvement, validating the hypothesis that "one encoder is enough."
- **NCE Outperforms CS**: Contrastive learning (pushing negative pairs apart) learns a better representation structure than a simple cosine loss (which only pulls positive pairs closer).
- **Effective Psychological Dimension Injection**: WhiSPA > WhiSA across all psychological evaluation tasks.

## Highlights & Insights
- **Eliminating Redundant LM Pipelines**: Demonstrates that the internal LM of an audio model can achieve the representation quality of an external text LM through alignment training, saving inference costs. This approach can be generalized to any multimodal scenario.
- **Psychological Dimensions as Self-Supervised Signals**: PsychEmb is automatically extracted from a dictionary without manual annotations, yet it effectively guides the speech encoder to learn deeper characteristics of human communication.
- **A Paradigm Shift from "Fusion" to "Alignment"**: Instead of designing complex multimodal fusion architectures, the latent spaces are aligned directly, offering a simpler and more efficient approach.

## Limitations & Future Work
- Evaluated only on Whisper-tiny (the smallest model); the alignment effectiveness for larger Whisper models remains unknown.
- Psychological data are sourced from mental health interviews; generalization to everyday conversations is unverified.
- PsychEmb is based on an English dictionary, leaving cross-lingual applicability questionable.
- It only aligns with text semantics and does not explicitly exploit acoustic features (intonation, rhythm).

## Related Work & Insights
- **vs SpeechBERT/SLAM**: Prior speech-text alignment efforts focused on ASR/retrieval, whereas WhiSPA is the first to align psychological dimensions.
- **vs Wav2Vec2/HuBERT**: These models are significantly weaker than WhiSPA on psychological tasks (with an $r$ difference of 0.1-0.2), lacking semantic understanding capabilities.
- **vs CLIP**: While CLIP aligns image-text, WhiSPA aligns speech-text + psychology; the methodology is similar but applies to a different domain.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First to incorporate psychological dimensions into speech-text alignment; the self-supervised PsychEmb signals are creative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluated on two datasets with both self-supervised and downstream evaluations, multi-model comparisons, and ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation and systematic experimental design.
- **Value**: ⭐⭐⭐⭐ Great practical value for psychological/clinical applications and engineering significance in eliminating redundant pipelines.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Self-Supervised Contrastive Learning is Approximately Supervised Contrastive Learning](../../NeurIPS2025/self_supervised/self-supervised_contrastive_learning_is_approximately_supervised_contrastive_lea.md)
- [\[ACL 2025\] QAEncoder: Towards Aligned Representation Learning in Question Answering Systems](qaencoder_aligned_representation.md)
- [\[ICLR 2026\] On the Alignment Between Supervised and Self-Supervised Contrastive Learning](../../ICLR2026/self_supervised/on_the_alignment_between_supervised_and_self-supervised_contrastive_learning.md)
- [\[ICML 2025\] Discovering Global False Negatives On the Fly for Self-supervised Contrastive Learning](../../ICML2025/self_supervised/discovering_global_false_negatives_on_the_fly_for_self-supervised_contrastive_le.md)
- [\[ICML 2025\] Collapse-Proof Non-Contrastive Self-Supervised Learning](../../ICML2025/self_supervised/collapse-proof_non-contrastive_self-supervised_learning.md)

</div>

<!-- RELATED:END -->

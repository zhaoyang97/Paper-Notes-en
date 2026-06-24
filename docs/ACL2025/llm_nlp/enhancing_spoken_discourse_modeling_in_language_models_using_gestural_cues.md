---
title: >-
  [Paper Note] Enhancing Spoken Discourse Modeling in Language Models Using Gestural Cues
description: >-
  [ACL 2025][LLM (Other)][gesture modeling] This paper proposes encoding gesture sequence data (3D human motion data) into discrete gesture tokens via a VQ-VAE, and then mapping them to the input space of a language model through feature alignment to enhance spoken discourse modeling. The complementary value of gesture information on spoken discourse understanding is validated through text-filling tasks across three types of discourse markers (discourse connectives, quantifiers…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "gesture modeling"
  - "spoken discourse"
  - "VQ-VAE tokenization"
  - "multimodal language model"
  - "discourse markers"
date: 2026-05-08
content_hash: 7a156df93e149729
---

# Enhancing Spoken Discourse Modeling in Language Models Using Gestural Cues

**Conference**: ACL 2025  
**arXiv**: [2503.03474](https://arxiv.org/abs/2503.03474)  
**Code**: None  
**Area**: LLM/NLP  
**Keywords**: gesture modeling, spoken discourse, VQ-VAE tokenization, multimodal language model, discourse markers

## TL;DR
This paper proposes encoding gesture sequence data (3D human motion data) into discrete gesture tokens via a VQ-VAE, and then mapping them to the input space of a language model through feature alignment to enhance spoken discourse modeling. The complementary value of gesture information on spoken discourse understanding is validated through text-filling tasks across three types of discourse markers (discourse connectives, quantifiers, and stance markers).

## Background & Motivation

**Background**: Linguistic research clearly demonstrates that non-verbal cues such as gestures play a critical role in spoken communication. Gestures can signal topic transitions and convey the attitude and certainty of the speaker.

**Limitations of Prior Work**: Language models processing spoken data rely almost entirely on text, ignoring the rich information transmitted through gestures. Existing gesture-language integration efforts mostly use coarse-grained 2D grid position encoding to represent hand position, losing fine-grained motion details.

**Key Challenge**: Gestures contain information complementary to language (e.g., spatial gestures expressing temporal concepts, palms-down conveying certainty), but effectively encoding 3D motion sequences and aligning them with text embeddings remains an unresolved challenge.

**Goal**: To explore whether joint modeling of gestures and language can enhance the spoken discourse understanding capabilities of language models.

**Key Insight**: To design three linguistically motivated text-filling tasks to evaluate the contribution of gestures to discourse modeling.

**Core Idea**: Utilizing VQ-VAE to discretize 3D gesture sequences into tokens, and mapping them to the language model space through feature alignment, thereby enabling language models to exploit gesture information in spoken discourse modeling.

## Method

### Overall Architecture
A three-stage pipeline: (1) **Gesture Tokenizer**: VQ-VAE encodes 3D motion sequences into discrete gesture tokens $\rightarrow$ (2) **Feature Alignment**: An MLP projector aligns gesture token embeddings to the input space of the language model (RoBERTa) $\rightarrow$ (3) **Fine-tuning**: Fine-tuning with LoRA on discourse marker prediction tasks.

### Key Designs

1. **Gesture Tokenization**

    - Input: $N=32$ frames of upper-body 3D motion sequences (15 fps), with $J=13$ joints represented in 6D rotation space.
    - The sequence is divided into $M=8$ chunks, encoded using a time-aware transformer encoder.
    - VQ-VAE quantization: codebook size $K=512$, embedding dimension $d=256$.
    - A Transformer decoder reconstructs the original motion sequence from the quantized tokens.
    - Special tokens `[BOG]` and `[EOG]` are introduced to mark the boundaries of gesture sequences.

2. **Feature Alignment**

    - MLP projector: Two fully-connected layers with GeLU activation.
    - The projected gesture embeddings are concatenated with text embeddings and fed into the pre-trained language model.
    - **Joint training objective**: $\mathcal{L}_{FA} = \mathcal{L}_{MGP} + \mathcal{L}_{MLM}$
        - $\mathcal{L}_{MGP}$: Masked Gesture Prediction—predicting the codebook index of the masked gesture token (K-class classification).
        - $\mathcal{L}_{MLM}$: Masked Language Modeling—standard MLM objective.
    - Randomly mask 30% of gesture and text tokens.
    - Only the parameters of the MLP projector are updated, while the language model and other components are frozen.
    - **Temporal Alignment**: The positional encoding of gesture tokens is aligned with the position of co-occurring text tokens to ensure temporal synchronization.

3. **Fine-tuning**

    - Three text-filling tasks where target markers are masked for the model to predict.
    - LoRA ($r=128, \alpha=256$) is used to fine-tune the adapter layers of the language model.
    - All other components are frozen.
    - Input format: $\langle s \rangle \mathbf{t_1} \langle mask \rangle \mathbf{t_2} \langle /s \rangle$
    - Classification is performed over a task-specific vocabulary subset $L_{task}$ using the output at the mask position through the LM head.

### Loss & Training

- VQ-VAE stage: Standard VQ-VAE reconstruction loss.
- Feature alignment stage: $\mathcal{L}_{FA} = \mathcal{L}_{MGP} + \mathcal{L}_{MLM}$, with a 30% mask rate.
- Fine-tuning stage: Cross-entropy loss + LoRA adaptation.
- Base language model: RoBERTa-base.
- Dataset: BEAT2 (60 hours of monologue gesture recordings, 25 speakers).
- All results are the average of 5 runs with different random seeds.

## Key Experimental Results

### Main Results

| Method | Discourse Acc/F1 | Quantifier Acc/F1 | Stance Acc/F1 |
|------|-----------------|-------------------|---------------|
| Text-only baseline | 60.4/47.5 | 69.4/65.2 | 50.6/46.5 |
| Mixed Modal (Xu & Cheng) | 34.8/17.4 | 31.7/28.2 | 33.4/24.0 |
| Grid-based tokens* | 55.3/41.3 | 70.5/65.4 | 47.9/44.5 |
| Codebook indices* | 54.4/39.2 | 68.4/63.9 | 46.5/41.7 |
| **GestureLM (Ours)** | **61.2/51.1** | **74.8/70.4** | **52.8/52.2** |

- F1 score improved by 4.8% on average (across three tasks).
- The Quantifier task showed the largest improvement: Accuracy +5.4, F1 +5.2.
- The learned gesture embeddings from VQ-VAE significantly outperform grid-based coarse-grained tokens and pure codebook indices.

### Ablation Study

**Adversarial Evaluation**:

| Setting | Discourse F1 | Quantifier F1 | Stance F1 |
|------|-------------|---------------|-----------|
| Random vectors | 48.1 | 68.4 | 48.6 |
| Only positional | 29.3 | 41.9 | 26.7 |
| **Pre-trained gesture** | **51.1** | **70.4** | **52.2** |

- Pre-trained gesture embeddings perform far better than random vectors and pure positional encodings, proving that gesture representations contain meaningful semantic information.

**Ablation of Model Components**:
- Removing relative positional encoding: Discourse F1 drops from 51.1 to 47.6.
- Removing feature alignment stage: Stance F1 plummets from 52.2 to 34.7 (highly unstable), demonstrating that the alignment stage is crucial.

**Masking Ratio**: A 30% mask rate yields the lowest validation loss (0.3); rates that are too high or too low degrade performance.

### Key Findings

1. **Gestures are more helpful for low-frequency markers**: Gestures primarily improve the prediction of markers that appear infrequently in the training data (e.g., *after*, *but*, *few*, *some*, *must*), as these rare markers tend to carry more specific meanings.
2. **Temporal discourse relations**: Temporal connectives like *after* and *while* co-occur with spatial gestures, which help distinguish them from the high-frequency word *and*.
3. **Epistemic stance expression**: *must* (indicating high certainty) co-occurs with palms-down gestures, helping to distinguish *must* from *may*.
4. **Patterns in quantifier confusion**: The gesture model confuses *some* with *two*/*one* (rather than *all*/*much*), likely because wrist movements cannot differentiate specific numbers (which requires finger joint data).

## Highlights & Insights

1. **Linguistically driven task design**: The choice of the three types of discourse markers is strongly backed by extensive linguistic research, rather than chosen arbitrarily.
2. **Rigorous adversarial validation**: Using random vectors and position-only baselines successfully eliminated the hypothesis that gesture information only acts as a regularization effect.
3. **Detailed error analysis**: Analyzing prediction differences across categories using relative confusion matrices and explaining the root causes with concrete gesture examples.
4. **Superiority of VQ-VAE gesture representation**: Compared to coarse-grained grid-based position encodings, VQ-VAE learned embeddings retain fine-grained motion details.

## Limitations & Future Work

1. Capturing only upper-body to wrist joint movements, lacking finger joint motion data—making it impossible to distinguish finger-related gestures (e.g., *two* vs. *three*).
2. Currently only applicable to encoder-only MLM models (RoBERTa), and has not been extended to decoder-based autoregressive models.
3. The BEAT2 dataset is collected from a specific group of speakers and communication scenarios, where gesture usage is subject to cultural and individual differences.
4. The model relies on 3D motion capture data; in real-world scenarios, input is mostly 2D video, which drastically reduces accuracy.
5. The scale of the dataset is limited (60 hours), and larger-scale data might yield more significant improvements.

## Related Work & Insights

- **Xu & Cheng (2023)**: First work to use grid-based gesture tokens paired with language models; this paper points out the limitations of their coarse-grained spatial encoding.
- **BEAT2 Dataset**: Provides large-scale paired gesture-speech data, serving as the foundation of this work.
- **VQ-VAE Gesture Synthesis**: VQ-VAE encoding methods designed for the gesture generation domain are creatively adapted for language understanding.
- **Insight**: Non-verbal cues (gestures, facial expressions, intonation) are severely undervalued in spoken language understanding; future multimodal language models should prioritize integrating these signals.

## Rating

| Dimension | Score (1-5) |
|------|-----------|
| Novelty | 4 |
| Technical Depth | 4 |
| Experimental Thoroughness | 4 |
| Writing Quality | 4 |
| **Overall Rating** | **4.0** |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] QG-SMS: Enhancing Test Item Analysis via Student Modeling and Simulation](qg-sms_enhancing_test_item_analysis_via_student_modeling_and_simulation.md)
- [\[ACL 2025\] Cross-Modal Alignment for LLM-Enhanced Spoken Language Understanding](cross-modal_alignment_for_llm-enhanced_spoken_language_understanding.md)
- [\[ACL 2025\] ECLM: Entity Level Language Model for Spoken Language Understanding with Chain of Intent](eclm_entity_level_language_model_spoken_language_understanding.md)
- [\[ACL 2025\] Neural Topic Modeling with Large Language Models in the Loop](neural_topic_modeling_with_large_language_models_in_the_loop.md)
- [\[ACL 2025\] When People are Floods: Analyzing Dehumanizing Metaphors in Immigration Discourse with Large Language Models](dehumanizing_metaphors_immigration.md)

</div>

<!-- RELATED:END -->

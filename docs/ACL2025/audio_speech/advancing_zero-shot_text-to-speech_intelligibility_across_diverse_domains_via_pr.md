---
title: >-
  [Paper Note] Advancing Zero-shot Text-to-Speech Intelligibility across Diverse Domains via Preference Alignment
description: >-
  [ACL 2025][Audio & Speech][Zero-shot TTS] This paper proposes the INTP (Intelligibility Preference Speech Dataset) dataset and multi-architecture DPO extension methods. Through preference alignment, the proposed approach significantly improves the intelligibility of zero-shot TTS systems in challenging scenarios (e.g., tongue twisters, repeated words, code-switching, and cross-lingual settings) while demonstrating weak-to-strong generalization.
tags:
  - "ACL 2025"
  - "Audio & Speech"
  - "Zero-shot TTS"
  - "Intelligibility"
  - "Preference Alignment"
  - "DPO"
  - "Dataset Construction"
date: 2026-05-08
content_hash: 4958252eaefb6ce8
---

# Advancing Zero-shot Text-to-Speech Intelligibility across Diverse Domains via Preference Alignment

**Conference**: ACL 2025  
**arXiv**: [2505.04113](https://arxiv.org/abs/2505.04113)  
**Code**: None (Demo page: [https://intalign.github.io/](https://intalign.github.io/))  
**Area**: Speech Synthesis / Audio & Speech  
**Keywords**: Zero-shot TTS, Intelligibility, Preference Alignment, DPO, Dataset Construction

## TL;DR

This paper proposes the INTP (Intelligibility Preference Speech Dataset) dataset and multi-architecture DPO extension methods. Through preference alignment, the proposed approach significantly improves the intelligibility of zero-shot TTS systems in challenging scenarios (e.g., tongue twisters, repeated words, code-switching, and cross-lingual settings) while demonstrating weak-to-strong generalization.

## Background & Motivation

Modern zero-shot text-to-speech (TTS) systems, such as CosyVoice, F5-TTS, and MaskGCT, can generate high-quality speech and support voice cloning for arbitrary speakers via large-scale pre-training. However, these systems still suffer from severe intelligibility issues in several **challenging scenarios**:

**Tongue Twisters**: Fast pronunciation transitions lead to slurred speech.

**Repeated Words**: Models tend to skip or merge consecutive repeated words.

**Code-Switching**: Alternating Chinese and English in a single sentence causes pronunciation and prosody disruptions.

**Cross-Lingual**: Synthesizing English with a Chinese reference voice (or vice versa) yields poor results.

The root cause of these issues is that pre-training data typically consists of "normal" read speech, with **severely insufficient coverage of the Gen-TTS difficult scenarios**. Consequently, model performance naturally degrades on out-of-distribution (OOD) data.

Traditional solutions involve retraining with more data collected from target scenarios, which is costly and lacks scalability. The core insight in this paper is that **preference alignment techniques are inherently suited to address such OOD problems**—by constructing "good/bad" speech pairs, the model learns to distinguish and prefer generating intelligible speech without requiring large-scale retraining.

The key insight is transferring mature NLP Direct Preference Optimization (DPO) techniques to the TTS domain and designing a unified alignment scheme for diverse TTS architectures. The Core Idea is to build a large-scale, multi-scenario, and multi-model intelligibility preference dataset, INTP, combined with DPO extensions for TTS architectures to achieve cross-model intelligibility improvements.

## Method

### Overall Architecture

The system consists of two core contributions:
1. **INTP Dataset Construction**: ~250K preference pairs (over 2000 hours of audio), covering various scenarios and TTS models.
2. **DPO Framework Extension**: Outlines alignment strategies specifically designed for three mainstream TTS architectures: Autoregressive (AR), Flow-Matching, and Masked Generative models.

Inputs are preference data pairs (chosen/rejected speech pairs), and outputs are aligned TTS models.

### Key Designs

1. **INTP Dataset — Multi-Scenario Coverage**:

    - Function: Construct an intelligibility preference speech dataset covering various challenging scenarios.
    - Mechanism: The data covers the following scenarios: (a) regular speech, (b) tongue twisters, (c) repeated word sequences, (d) code-switching speech, and (e) cross-lingual synthesis (Chinese reference + English target/vice versa). Data is generated using three TTS models with distinct architectures (ARS, F5-TTS, MaskGCT) to ensure structural diversity.
    - Design Motivation: A single scenario or a single model cannot cover the full picture of TTS intelligibility issues. Multi-scenario data guarantees the diversity of alignment signals, while multi-model generation prevents the alignment from overfitting to a specific architecture.

2. **Three Preference Pair Construction Strategies**:

    - Function: Design three complementary preference pair formulation methods to maximize the information content of alignment signals.
    - Mechanism:
        - **Intra Pair (Within-model contrast)**: Using Best-of-N sampling on the same model, selecting the best generated output as "chosen" and the worst as "rejected". This is essentially self-contrastive learning.
        - **Inter Pair (Cross-model contrast)**: Comparing the synthesis results of different models on the same text to leverage complementary advantages across models. For instance, if ARS outperforms MaskGCT in prosody, the ARS output is used as chosen.
        - **Perturbed Pair (Perturbation contrast)**: Leveraging human expert knowledge and DeepSeek-V3 to generate perturbed texts as negatives. Two perturbations: (1) **Pronunciation perturbation**—replacing words with easily mispronounced homophones (e.g., "Haohao" $\rightarrow$ "Haohao" with different characters), (2) **Punctuation perturbation**—modifying comma placement to alter pause prosody.
    - Design Motivation: The three preference pairs provide alignment signals from different dimensions—Intra focuses on stability, Inter on cross-architecture best practices, and Perturbed on error tolerance. Combining them yields richer learning signals.

3. **DPO Extension for TTS Architectures**:

    - Function: Adapt the DPO framework to three different TTS model architectures.
    - Mechanism: The standard DPO loss is defined as:
      $$\mathcal{L}_{DPO} = -\log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right)$$
      where $y_w$ is the chosen speech and $y_l$ is the rejected speech. The key challenge lies in the different definitions of $\pi_\theta(y|x)$ across TTS architectures:
        - **ARS**: Autoregressive probability $\prod_t P(s_t | s_{<t}, x)$, where $s_t$ is the speech token.
        - **F5-TTS**: Flow-matching objective $\|v_\theta(z_t, t) - u_t\|^2$, which needs to be converted into preference signals.
        - **MaskGCT**: Masked prediction probability $\prod_i P(s_i | s_{\backslash i}, x)$.
      Specific DPO adaptation schemes are designed for each architecture.
    - Design Motivation: The TTS domain is not predominantly autoregressive like language models; architectural diversity is a reality. Designing a general DPO extension allows a single preference dataset to serve multiple types of models.

### Loss & Training

- The core loss is the DPO loss, adapted for different TTS architectures.
- During training, preference pairs are sampled from INTP, the reference model's parameters are frozen, and the target model is updated.
- Supports **iterative alignment**: The model aligned in the first round can generate new preference data for a second round of alignment, achieving incremental improvements.

## Key Experimental Results

### Main Results

| Model | Scenario | Pre-alignment Intelligibility | Post-alignment Intelligibility | Naturalness Gain | Similarity Gain |
|------|------|-------------|-------------|-----------|-----------|
| ARS | Regular | Baseline | ↑ | ↑ | ↑ |
| ARS | Tongue Twisters | Low | Significant ↑ | ↑ | Maintained |
| F5-TTS | Code-Switching | Low | Significant ↑ | ↑ | ↑ |
| F5-TTS | Cross-Lingual | Low | Significant ↑ | ↑ | Maintained |
| MaskGCT | Repeated Words | Medium | ↑ | ↑ | ↑ |
| MaskGCT | Tongue Twisters | Low | Significant ↑ | ↑ | Maintained |

INTP alignment improves not only intelligibility but also overall naturalness, speaker similarity, and audio quality.

### Weak-to-Strong Generalization

| Model | Participated in INTP Construction | Alignment Effect | Description |
|------|-----------------|---------|------|
| ARS | Yes | Significant Improvement | Constituent Model |
| F5-TTS | Yes | Significant Improvement | Constituent Model |
| MaskGCT | Yes | Significant Improvement | Constituent Model |
| CosyVoice 2 | No | Still Effectively Improved | Weak-to-strong generalization validation |
| Ints | No | Still Effectively Improved | Weak-to-strong generalization validation |

CosyVoice 2 (initialized based on Qwen-2.5 0.5B) and Ints (initialized based on Phi-3.5-mini 3.8B) were not involved in the construction of the INTP dataset. However, INTP alignment remained effective for them, verifying the capability of **weak-to-strong generalization**.

### Key Findings
- **Preference alignment is an effective approach to improve TTS intelligibility**: Consistent improvements are brought across five models and multiple scenarios.
- **Perturbed Pairs contribute the most**: Leveraging human knowledge and LLM-generated perturbation data provides the strongest alignment signal.
- **Weak-to-strong generalization holds**: Preference data constructed by weaker models can improve the performance of stronger models.
- **Iterative alignment is consistently effective**: Iterative alignment based on Ints shows potential for further performance gains.
- **Intelligibility positively correlates with other metrics**: Post-alignment, not only does intelligibility improve, but naturalness and speech quality also advance simultaneously.

## Highlights & Insights

- **Transfer of NLP techniques to Speech**: Successfully migrating DPO from text generation to speech synthesis introduces a new preference alignment paradigm to the TTS domain.
- **Architecture-agnostic alignment**: Adapting DPO to three different TTS architectures demonstrates that preference alignment can serve as an architecture-independent, general enhancement method.
- **Engineering wisdom in data construction**: The three preference pair designs (Intra/Inter/Perturbed) complement each other, with the Perturbed strategy particularly clever in leveraging LLMs to generate homophonic-perturbed text.
- **Weak-to-strong generalization**: This feature ensures the enduring value of the INTP dataset—even as stronger TTS models emerge in the future, existing datasets may remain effective.
- **Practical scenario-oriented**: Focusing on pain points in real-world applications (like tongue twisters and code-switching) gives this research clear engineering and deployment value.

## Limitations & Future Work

- **Language coverage**: Currently only validated on Chinese and English; performance on other languages (e.g., Japanese, Korean, European languages) remains unexplored.
- **Preference annotation quality**: Automatic selection of chosen/rejected samples may contain noise, and introducing human preference annotations could further enhance results.
- **Computational cost**: Generating 250K preference pairs requires a high volume of inference rounds using three models, leading to high data construction costs.
- **Evaluation metrics**: Discrepancies may exist between automatic evaluation metrics (e.g., CER/WER) for intelligibility and human perception.
- **Alignment tax**: Whether over-alignment results in reduced speech diversity (e.g., styles converging across speakers) is not fully discussed in the paper.

## Related Work & Insights

- **vs RLHF/DPO in Language Models**: This paper migrates DPO from the text domain to the speech domain. The key challenges lie in the diverse architectures of TTS and loss function adaptability.
- **vs Speech alignment works (e.g., SpeechAlign)**: Prior speech alignment works primarily focused on naturalness and speaker similarity. This paper is the first to systematically target **intelligibility**, a previously overlooked dimension.
- **vs Data augmentation methods**: While conventional methods improve performance by adding training data of difficult scenarios, the proposed preference alignment achieves superior results with less data.

## Rating

- Novelty: ⭐⭐⭐⭐ First to systematically address TTS intelligibility using preference alignment, with a clever INTP dataset design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive experiments across five models, multiple scenarios, weak-to-strong generalization, and iterative alignment.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, with a highly detailed description of the data construction process.
- Value: ⭐⭐⭐⭐⭐ The dataset and methodology hold direct utility for the TTS community, and the weak-to-strong generalization findings are theoretically significant.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Zero-Shot Text-to-Speech for Vietnamese](zero-shot_text-to-speech_for_vietnamese.md)
- [\[ACL 2025\] ControlSpeech: Towards Simultaneous and Independent Zero-shot Speaker Cloning and Zero-shot Language Style Control](controlspeech_zero_shot.md)
- [\[ACL 2025\] SpeechWeave: Diverse Multilingual Synthetic Text & Audio Data Generation Pipeline for Training Text to Speech Models](speechweave_diverse_multilingual_synthetic_text_audio_data_generation_pipeline_f.md)
- [\[ACL 2025\] Soundwave: Less is More for Speech-Text Alignment in LLMs](soundwave_less_is_more_for_speech-text_alignment_in_llms.md)
- [\[NeurIPS 2025\] LeVo: High-Quality Song Generation with Multi-Preference Alignment](../../NeurIPS2025/audio_speech/levo_high-quality_song_generation_with_multi-preference_alignment.md)

</div>

<!-- RELATED:END -->

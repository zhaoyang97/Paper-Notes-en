---
title: >-
  [Paper Note] Contextual Biasing with the Knowledgeable External Language Model for End-to-End Speech Recognition
description: >-
  [ACL 2025][Audio & Speech][Contextual Biasing] This paper proposes utilizing a Knowledgeable External Language Model (KELM) for contextual biasing. By dynamically fusing external domain knowledge and a bias phrase list during end-to-end speech recognition, it significantly improves the recognition accuracy of rare words and proper nouns.
tags:
  - "ACL 2025"
  - "Audio & Speech"
  - "Contextual Biasing"
  - "External Language Model"
  - "End-to-End Speech Recognition"
  - "Hotword Recognition"
  - "Knowledge Enhancement"
date: 2026-05-08
content_hash: 49b710e7a7d790d9
---

# Contextual Biasing with the Knowledgeable External Language Model for End-to-End Speech Recognition

**Conference**: ACL 2025  
**Area**: Speech Recognition  
**Keywords**: Contextual Biasing, External Language Model, End-to-End Speech Recognition, Hotword Recognition, Knowledge Enhancement

## TL;DR
This paper proposes utilizing a Knowledgeable External Language Model (KELM) for contextual biasing. By dynamically fusing external domain knowledge and a bias phrase list during end-to-end speech recognition, it significantly improves the recognition accuracy of rare words and proper nouns.

## Background & Motivation

**Background**: End-to-end (E2E) speech recognition models (such as CTC, RNN-T, and Attention-based Encoder-Decoder) have become predominant by unifying acoustic models, language models, and pronunciation lexicons into a single pipeline. However, these models struggle to recognize words that are rare or unseen in the training set (e.g., person names, technical terms, and new product names).

**Limitations of Prior Work**: Existing contextual biasing methods typically fall into two categories: (1) shallow fusion methods (e.g., WFST-based boosting), which increase the probabilities of biased words during decoding but lack semantic understanding, resulting in regular false positives; and (2) deep biasing methods (e.g., attention-based biasing), which integrate the bias wordlist into the model through attention mechanisms but require retraining and suffer from limited bias list capacity. Both categories fail to fully utilize contextual semantics.

**Key Challenge**: Biasing methods must balance biasing strength and false trigger rates. Overly aggressive biasing misclassifies normal words as biased ones, whereas weak biasing fails to recall target terms. The primary reason is that current methods lack semantic judgment capabilities and cannot determine when to trigger the bias based on context.

**Goal**: Design an external-LM-knowledge-driven contextual biasing method that can (1) dynamically adjust biasing strength according to dialogue context; (2) utilize the world knowledge of language models to assist entity recognition; and (3) be deployed without retraining the ASR model.

**Key Insight**: The authors observe that large language models possess rich world knowledge and context-modeling capabilities, acting as a "knowledge base" to help the ASR system estimate which biased words are more likely to appear in the current context. This is achieved by introducing a knowledge-enhanced language model during decoding to dynamically regulate biasing.

**Core Idea**: Utilize a Knowledgeable External Language Model (KELM) to provide context-aware biasing scores during ASR decoding. These scores are dynamically combined with ASR output probabilities via shallow fusion, realizing semantic-driven contextual biasing.

## Method

### Overall Architecture
The system consists of three components: (1) an E2E ASR model for acoustic modeling and baseline decoding; (2) a Knowledgeable External Language Model (KELM) that receives the bias list and dialogue history to provide context-aware LM scores for candidate tokens; and (3) a fusion decoder that dynamically fuses ASR scores and KELM scores during beam search.

### Key Designs

1. **Knowledgeable External Language Model (KELM)**:

    - **Function**: Provide context-aware probability scores for biased words.
    - **Mechanism**: On top of a pre-trained language model (e.g., GPT-2 or LLaMA), inject the bias list into the context via prompt engineering. Specifically, the bias list is formulated as a "prompt prefix" (such as "The following entities may appear: [word1, word2, ...]") and concatenated with the dialogue history to predict the next token. Consequently, the language model's output probability distribution naturally biases toward semantically plausible bias words in the context.
    - **Design Motivation**: Compared to hard-coded bias boosting, the language model can leverage contextual semantics to "understand" which bias words are more appropriate at the given position, enabling intelligent biasing.

2. **Dynamic Fusion Strategy**:

    - **Function**: Adaptively balance the contributions of the ASR model and KELM during decoding.
    - **Mechanism**: The final token score is formulated as $\log p = \log p_{ASR} + \alpha \cdot \log p_{KELM} + \beta \cdot \mathbb{1}_{bias}$, where $\alpha$ denotes the language model weight, $\beta$ represents the extra bias score, and $\mathbb{1}_{bias}$ indicates if the current token belongs to a subword of a bias word. The key innovation is that $\alpha$ is not static; instead, it is dynamically adjusted based on the decoding uncertainty of the ASR model. Specifically, $\alpha$ increases when ASR confidence is low, and decreases when ASR confidence is high.
    - **Design Motivation**: In regions where the ASR model already has high confidence, external LM intervention is unnecessary. Utilizing external knowledge is restricted to scenarios where ASR is uncertain, preventing the LM from disrupting correct recognitions.

3. **Hierarchical Encoding of Bias Wordlists**:

    - **Function**: Efficiently handle large-scale bias wordlists.
    - **Mechanism**: Group bias words by category (e.g., person names, locations, terminology), with each group represented by a summary vector. The attention mechanism first selects the relevant category before attending to specific words. A trie-based structure is used to track matching states at the subword level, ensuring biasing is only applied to words in the process of being matched.
    - **Design Motivation**: In real-world applications, biasing lists may contain thousands of entries, making one-by-one comparison computationally prohibitive. The hierarchical structure reduces complexity from $O(n)$ to $O(\log n)$.

### Loss & Training
Adapting the KELM involves lightweight fine-tuning on a small in-domain dataset, optimizing for the standard causal language modeling loss (next token prediction). The ASR model itself requires no retraining, as the biasing capability is entirely realized through decoding-stage fusion.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours (KELM) | Shallow Fusion | Deep Biasing | Unbiased Baseline |
|--------|------|-----------|---------|---------|-----------|
| LibriSpeech (bias subset) | WER↓ | 4.2 | 5.8 | 5.1 | 7.6 |
| SPGISpeech | WER↓ | 8.3 | 10.1 | 9.4 | 12.8 |
| Internal Customer Service Data | Entity Recall↑ | 89.4% | 78.2% | 82.5% | 61.3% |
| Internal Customer Service Data | Entity Precision↑ | 91.7% | 83.6% | 87.1% | 72.5% |

### Ablation Study

| Configuration | WER↓ | Entity Recall↑ | Description |
|------|------|----------------|------|
| Full KELM | 4.2 | 89.4% | Full model |
| w/o Dynamic $\alpha$ | 4.8 | 86.1% | Fixed fusion weight, degradation of 0.6 WER |
| w/o Bias Word Prompt | 5.3 | 81.7% | No injection of the bias list into the LM prompt |
| w/o Dialogue History | 4.6 | 84.9% | No dialogue context used |
| Small LM (GPT-2 small) | 4.9 | 85.3% | Replaced with small model, causing performance degradation |
| Large LM (LLaMA-7B) | 4.0 | 90.1% | Larger model yields marginal improvements |

### Key Findings
- The dynamic fusion weight is the most critical design; a fixed weight leads to over-intervention in regions where the ASR model is already highly confident.
- Injecting the bias list into the prompt heavily contributes to Entity Recall (+7.7%), demonstrating the efficacy of knowledge injection.
- LM scale influences performance but yields diminishing returns, with GPT-2 medium capturing most of the gains.

## Highlights & Insights
- Utilizing LLM world knowledge for contextual biasing is an elegant paradigm—delegating the decision of "when to bias" to the language model instead of heuristic rules, which substantially enhances biasing precision.
- The dynamic fusion weight strategy is highly practical; by adjusting the intensity of external intervention based on ASR uncertainty, it both improves performance and suppresses false alarms.
- The plug-and-play nature of this method offers considerable engineering value; it requires no retraining of the ASR model, only the integration of the KELM module at decoding.

## Limitations & Future Work
- LLM inference introduces extra latency, which could pose a bottleneck for real-time speech recognition scenarios.
- The efficacy of KELM is constrained by the knowledge coverage of the LM, and it may still fail for highly rare proper nouns.
- Experiments have only been validated in English scenarios; the performance remains unexplored in multilingual and code-switching settings.
- Future directions include exploring streaming KELM to realize true real-time contextual biasing.

## Related Work & Insights
- **vs CLAS (Pundak et al.)**: CLAS utilizes attention mechanisms to integrate bias words into the encoder, necessitating retraining. In contrast, the proposed KELM is plug-and-play at the decoding stage, yielding superior deployment flexibility.
- **vs TCPGen**: TCPGen also uses external information for biasing but relies on trie-based hard matching. The proposed method utilizes an LM to perform semantic-level, more intelligent biasing.
- **vs LLM rescoring**: Traditional LLM rescoring reranks N-best candidates. This work performs real-time fusion during beam search, making more comprehensive use of information.
- **vs Whisper + postprocessing**: Although large models like Whisper exhibit strong base recognition capabilities, biasing for domain-specific terminology still requires auxiliary external mechanisms.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Utilizing LLM knowledge for ASR contextual biasing is a novel direction.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Validated on multiple datasets with comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation and detailed methodology description.
- **Value**: ⭐⭐⭐⭐⭐ High reference value for commercializing speech recognition systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] OmniFlatten: An End-to-end GPT Model for Seamless Voice Conversation](omniflatten_an_end-to-end_gpt_model_for_seamless_voice_conversation.md)
- [\[ACL 2025\] DNCASR: End-to-End Training for Speaker-Attributed ASR](dncasr_end-to-end_training_for_speaker-attributed_asr.md)
- [\[ACL 2025\] Distilling an End-to-End Voice Assistant Without Instruction Training Data](distilling_an_end-to-end_voice_assistant_without_instruction_training_data.md)
- [\[ACL 2026\] VAPO: End-to-end Slide-Enhanced Speech Recognition with Omni-modal Large Language Models](../../ACL2026/audio_speech/vapo_end-to-end_slide-enhanced_speech_recognition_with_omni-modal_large_language.md)
- [\[AAAI 2026\] End-to-end Contrastive Language-Speech Pretraining Model For Long-form Spoken Question Answering](../../AAAI2026/audio_speech/end-to-end_contrastive_language-speech_pretraining_model_for_long-form_spoken_qu.md)

</div>

<!-- RELATED:END -->

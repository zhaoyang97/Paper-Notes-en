---
title: >-
  [Paper Note] Locate-and-Focus: Enhancing Terminology Translation in Speech Language Models
description: >-
  [ACL 2025][LLM (Other)][speech translation] This paper proposes the Locate-and-Focus method for terminology translation in speech LLMs. By first using sliding window retrieval to locate audio segments containing terminologies, and then guiding the model to focus on translation knowledge through audio replacement and Tag Cues, the terminology translation success rate is significantly improved for English-Chinese and English-German directions.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "speech translation"
  - "terminology translation"
  - "speech LLM"
  - "sliding retrieval"
  - "multi-modal knowledge"
date: 2026-05-08
content_hash: 6005c7aa728ffcfd
---

# Locate-and-Focus: Enhancing Terminology Translation in Speech Language Models

**Conference**: ACL 2025  
**arXiv**: [2507.18263](https://arxiv.org/abs/2507.18263)  
**Code**: [https://github.com/DeepLearnXMU/Locate_and_Focus_ST](https://github.com/DeepLearnXMU/Locate_and_Focus_ST)  
**Area**: LLM/NLP  
**Keywords**: speech translation, terminology translation, speech LLM, sliding retrieval, multi-modal knowledge

## TL;DR

This paper proposes the Locate-and-Focus method for terminology translation in speech LLMs. By first using sliding window retrieval to locate audio segments containing terminologies, and then guiding the model to focus on translation knowledge through audio replacement and Tag Cues, the terminology translation success rate is significantly improved for English-Chinese and English-German directions.

## Background & Motivation
**Background**: End-to-end speech translation (ST) performs poorly in terminology translation. Correct translation of terminologies (e.g., human names, drug names) is crucial for accurate information delivery.

**Limitations of Prior Work**: The Collect-and-Integrate paradigm introduces all corpus terminology, leading to substantial irrelevant information. The Retrieve-and-Demonstrate paradigm retrieves exemplars that contain sentence portions unrelated to terminology translation.

**Key Challenge**: While external translation knowledge is helpful, its integration remains coarse—the text and audio modalities mismatch, and the retrieved exemplars come from different speakers, making it difficult for the ST model to fully utilize them.

**Goal**: Precisely locate terminology segments in speech and effectively guide speech LLMs to focus on translation knowledge.

**Key Insight**: A two-step approach: first "Locate" (sliding window retrieval of speech segments matching the terminology), and then "Focus" (audio replacement establishing a shared anchor + Tag cues guiding translation).

**Core Idea**: Locate terminology segments in speech and replace them in the translation knowledge to establish a shared audio anchor, and then use the `<Term>` tag to prompt the model to focus on the terminology translation.

## Method

### Overall Architecture
It consists of two steps: Terminology Clip Localization and Terminology-Focused Translation. The former locates terminology-related audio segments in speech via sliding window retrieval, while the latter guides the speech LLM to focus on translation knowledge using audio replacement and Tag Cue strategies. The speech encoder is trained with contrastive learning, and the translation model is fine-tuned with LoRA.

### Key Designs
1. **Sliding Retrieval**: Uses a speech encoder to encode the terminology clip $c$ and speech $u$. It slides across $u$ with a window size equal to the length of $c$ and a step size of 1, computing the max-pooling cosine similarity between each subsequence and $c$. The maximum value determines the terminology occurrence probability, while locating the corresponding speech segment $s$.
2. **Audio Replacement**: Replaces the speech clip $c$ in the retrieved translation knowledge triplet with the located segment $s$. This allows the speech and translation knowledge to share the same acoustic feature anchor, guiding the model to focus on the translation knowledge.
3. **Tag Cue**: Inserts a special tag `<Term>` before the terminology translation in the training data. During inference, predicting `<Term>` triggers focus on translation knowledge.

### Loss & Training
- Localization Step: Contrastive learning to train the speech encoder: $\mathcal{L}_{SE} = -\log \frac{e^{sim(u, c^+)}}{e^{sim(u, c^+)} + \sum e^{sim(u, c_i^-)}}$
- Translation Step: Standard next-token prediction loss + LoRA fine-tuning
- Two-step sequential training: localization is trained followed by translation
- CosyVoice2 TTS is utilized to generate terminology speech clips, and SenseVoice ASR is used to verify the quality.

## Key Experimental Results

### Main Results (Oracle Knowledge Setting, EN→ZH)

| Method | CoVoST2 TSR | CoVoST2 BLEU | MuST-C TSR | MuST-C BLEU |
|------|------------|-------------|------------|-------------|
| Base Model | 24.12 | 35.82 | 27.61 | 25.73 |
| SALM | 76.53 | 55.97 | 69.01 | 32.10 |
| Retrieve-and-Demo | 60.88 | 50.22 | 58.87 | 30.18 |
| **Locate-and-Focus** | **90.13** | **58.49** | **94.09** | **34.52** |

### Ablation Study (Oracle Setting, EN→ZH CoVoST2)

| Configuration | TSR | BLEU |
|------|-----|------|
| Locate-and-Focus (full) | 90.13 | 58.49 |
| w/o Audio Replacement | 89.67 | 58.37 |
| w/o Tag Cue | 89.00 | 58.25 |
| w/o Both | 88.59 | 58.32 |

### Key Findings
- Dataset Scale: Training set of 10K speech samples + 14K term pairs; evaluation on three test sets (CoVoST2, MuST-C, and MSLT).
- Compared to Translation Training (translation training only, without terminology knowledge), TSR is only 27.30% vs. 90.13%.
- Compared to the Base Model (without any terminology enhancement), TSR is only 24.12%, showing a huge gap.

- Locate-and-Focus significantly leads in Terminology Translation Success Rate (TSR): CoVoST2 EN→ZH achieves 90.13% vs. SALM 76.53% vs. R&D 60.88%.
- The improvement is even more significant in the EN→DE direction: CoVoST2 TSR reaches 96.35% vs. SALM 85.91%.
- Both Audio Replacement and Tag Cue positively contribute, though localization itself remains the most critical component.
- There is still a significant improvement under the end-to-end setting (65.53% TSR), although it is lower than the Oracle setting.
- General translation quality (BLEU) remains unaffected or is slightly improved.

## Highlights & Insights
- The first end-to-end terminology translation method to utilize multi-modal, fine-grained translation knowledge in speech LLMs.
- The sliding window retrieval design is simple yet effective, allowing parallel computation with only a minor increase in inference latency.
- The audio replacement strategy elegantly resolves cross-modal/cross-speaker discrepancies by allowing the speech and translation knowledge to share the same acoustic features.
- The `<Term>` tag as a self-reminder mechanism is lightweight and practical, eliminating the need for extra modules.
- The TSR reaches 96.35% in the EN→DE direction, showing that the method is equally effective for morphologically rich languages.
- Ablation studies demonstrate that localization itself is the most critical component, while Audio Replacement and Tag Cue each provide an incremental contribution of about 1-2%.
- The methodology of the self-constructed terminology translation datasets (CoVoST2/MuST-C/MSLT + TTS/ASR pipelines) is highly reusable.

## Limitations & Future Work
- The performance gap between the end-to-end setting and the Oracle setting remains large (with a TSR difference of ~25%), making retrieval accuracy the bottleneck.
- The dataset is self-collected, with terminology pairs extracted by an LLM and manually verified, which may contain omissions or errors.
- Terminology speech generated by TTS may suffer from domain differences compared to natural speech (synthetic vs. natural), affecting localization accuracy.
- Only English-to-Chinese and English-to-German directions are supported, leaving multi-lingual (e.g., low-resource languages) extensibility to be verified.
- A sliding window step size of 1 might incur high computational overhead on long speech sequences; multi-scale or hierarchical retrieval could be explored.
- Scenarios where one speech sample contains multiple closely spaced terminologies are not considered.

## Related Work & Insights
- SALM (Gaido et al., 2023) and Retrieve-and-Demonstrate (Li et al., 2024a) are the primary baseline paradigms; this work combines the strengths of both.
- The terminology recognition approach in ASR from CB-Whisper (Li et al., 2024b) can be referenced, but it does not address cross-modal challenges.
- The audio replacement strategy can be generalized to other scenarios requiring multi-modal alignment (e.g., visual grounding in video translation).
- This work has direct application value for computer-assisted interpreting systems, providing precise localization and prompting for professional terminology translation.
- The TTS-ASR pipelines of CosyVoice2 and SenseVoice supply a scalable methodology for data construction.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The two-step design of localization + focus and the audio replacement strategy are novel, successfully addressing cross-modal alignment challenges.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Validated across multiple datasets and directions with complete ablation studies, though only supporting two translation directions.
- **Writing Quality**: ⭐⭐⭐⭐ Clear illustrations, highly detailed methodological descriptions, and standard mathematical formulations.
- **Value**: ⭐⭐⭐⭐ This work makes practical contributions toward resolving terminology challenges in speech translation.
- Overall Evaluation: Strongly engineering-oriented and highly practical; the dataset construction methodology is reusable, with direct application value for computer-assisted interpreting systems.
- Reproducibility: Code is open-sourced, and the dataset construction pipeline is reusable.
- Extensibility: Generalizable to more multilingual directions and low-resource language scenarios.
- Open Question: How to balance localization accuracy and inference speed in real-time scenarios?
- Impact: Provides a new multi-modal knowledge utilization paradigm for terminology processing in speech translation.
- Future Direction: Combining with CTC or Viterbi decoding could further improve terminology localization accuracy.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] InfiniSST: Simultaneous Translation of Unbounded Speech with Large Language Model](infinisst_simultaneous_translation_of_unbounded_speech_with_large_language_model.md)
- [\[ACL 2025\] Recent Advances in Speech Language Models: A Survey](recent_advances_in_speech_language_models_a_survey.md)
- [\[ACL 2025\] Language-Codec: Bridging Discrete Codec Representations and Speech Language Models](language_codec_bridging_discrete_codec_speech_language_models.md)
- [\[ACL 2025\] When Large Language Models Meet Speech: A Survey on Integration Approaches](when_large_language_models_meet_speech_a_survey_on_integration_approaches.md)
- [\[ACL 2025\] Analyzing and Mitigating Inconsistency in Discrete Speech Tokens for Neural Codec Language Models](analyzing_and_mitigating_inconsistency_in_discrete_speech_tokens_for_neural_code.md)

</div>

<!-- RELATED:END -->

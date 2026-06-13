---
title: >-
  [Paper Note] From Flat Language Labels to Typological Priors: Structured Language Conditioning for Multilingual Speech-to-Speech Translation
description: >-
  [ACL2026][Audio & Speech][Speech-to-Speech Translation] This paper proposes S2ST-Omni 2, which replaces flat language labels in multilingual speech translation with structured typological priors. These priors are injecte…
tags:
  - "ACL2026"
  - "Audio & Speech"
  - "Speech-to-Speech Translation"
  - "Typological Priors"
  - "Multilingual Conditioning"
  - "Dual-CTC"
  - "Low-resource Adaptation"
date: 2026-05-08
content_hash: 8d56ff4777e209eb
---

# From Flat Language Labels to Typological Priors: Structured Language Conditioning for Multilingual Speech-to-Speech Translation

**Conference**: ACL2026 Findings  
**arXiv**: [2605.16026](https://arxiv.org/abs/2605.16026)  
**Code**: No public code (repository not provided in cache)  
**Area**: Speech Translation / Multilingual S2ST / SpeechLLM  
**Keywords**: Speech-to-Speech Translation, Typological Priors, Multilingual Conditioning, Dual-CTC, Low-resource Adaptation  

## TL;DR
This paper proposes S2ST-Omni 2, which replaces flat language labels in multilingual speech translation with structured typological priors. These priors are injected across three levels: representation, acoustic modulation, and LLM decoding. This approach improves BLEU, ASR-BLEU, COMET, and BLASER 2.0 on CVSS-C, particularly for low-resource languages and those with significant typological differences.

## Background & Motivation
**Background**: Multilingual speech-to-speech translation (S2ST) can be implemented via ASR-MT-TTS cascaded systems or end-to-end/compositional S2ST. SpeechLLMs have made compositional S2ST attractive: a front-end converts source speech to target text, and a back-end TTS synthesizes target speech, enabling modularity and resource reuse.

**Limitations of Prior Work**: Existing systems like S2ST-Omni treat the source language as a flat label or an independent embedding. This informs the model of the specific language (e.g., German/French/Spanish) but fails to explicitly represent shared structural patterns such as morphology, word order, or genealogical lineage. In low-resource S2ST, it is difficult for models to learn these patterns from limited supervised data.

**Key Challenge**: Multilingual models must distinguish between specific languages while sharing cross-lingual structures. Flat labels provide identity information without transferable typological structures, while purely data-driven learning is unreliable in low-resource scenarios.

**Goal**: The authors aim to restructure the language conditioning pathway without major changes to the S2ST-Omni backbone. They represent the source language as interpretable typological priors to verify if such priors improve data efficiency and translation quality.

**Key Insight**: The paper divides language conditioning into three layers: a typology-informed hierarchical language encoding at the representation layer, a dynamically-gated language-aware Dual-CTC at the acoustic layer, and typology-aware LLM prompting at the decoding layer.

**Core Idea**: Source languages are represented by morphology, reordering requirements, genealogical family, and residual language features. These structured conditions influence intermediate acoustic features, auxiliary CTC alignment, and LLM translation prompts.

## Method
S2ST-Omni 2 maintains the compositional framework of S2ST-Omni: a Whisper encoder extracts source speech features, a hybrid speech adapter maps them to the hidden space of Qwen3 LLM (4B), and a pluggable TTS backend synthesizes English speech. Modifications focus on the language conditioning pathway.

### Overall Architecture
The input is source language speech, and the output is English speech. During training, ground truth labels are used; during inference, the source language is predicted from Whisper encoder representations. Three typological components are added: TI-HLE generates structured language vectors, DG-LA-Dual-CTC performs language-aware modulation on intermediate adapter features with Dual-CTC supervision, and TA-Prompt provides language-level translation hints during LLM decoding. In inference, TI-HLE and DG-LA-Dual-CTC auxiliary branches are discarded to maintain the acoustic forward path, while the predicted source language selects the corresponding typology-aware prompt.

### Key Designs
1.  **Typology-Informed Hierarchical Language Encoding**:
    *   **Function**: Decomposes the source language from a single ID into shared, interpretable structured representations.
    *   **Mechanism**: Each language is represented by four channels: morphology profile, English-oriented reordering profile, genealogical family, and language-specific residual. For example, French/Spanish share the Romance family, while German is Germanic; German/Japanese require stronger clause-final or verb-final reordering hints. The four embedding sets (dimensions 64, 64, 64, 128) are concatenated and projected into a 256-dimensional language representation.
    *   **Design Motivation**: Structural channels provide cross-lingual shared priors, while the residual channel preserves language individuality to avoid over-simplification into typological buckets.

2.  **Dynamically-Gated Language-Aware Dual-CTC**:
    *   **Function**: Injects typological conditions into acoustic intermediate representations while maintaining source content and target alignment.
    *   **Mechanism**: A FiLM generator produces $\gamma$ and $\beta$ based on the language representation. A dynamic frame gate $g_t$ is generated per frame from acoustic features. The modulated source-side feature is $\tilde{h}^{src}_t=(1+g_t\gamma)\odot h^{down}_t+g_t\beta$. Source-side CTC uses modulated features, while target-side CTC uses unmodulated features.
    *   **Design Motivation**: Language priors should not be applied uniformly; different frames require different levels of typological modulation. Dynamic gating allows the model to apply language conditions more strongly at content-relevant positions.

3.  **Typology-Aware LLM Prompting & Progressive Fine-tuning**:
    *   **Function**: Explicitly informs the LLM of language-level translation challenges and stabilizes SpeechLLM adaptation.
    *   **Mechanism**: German prompts emphasize compound word decomposition and clause-final reordering; Japanese prompts highlight SOV-to-SVO, subject omission, and honorific normalization. A two-stage progressive fine-tuning is used: first stabilizing speech-text alignment, then inserting LoRA into Qwen3 self-attention to enhance translation.
    *   **Design Motivation**: Acoustic conditions address "listening and alignment," while prompt conditions address "natural translation," targeting different levels of linguistic structure.

### Loss & Training
Both training stages freeze Whisper encoder and Qwen3 base parameters while updating the hybrid adapter, TI-HLE, and DG-LA-Dual-CTC. Stage I objective is $\mathcal{L}^{(1)}=\mathcal{L}_{CE}+\lambda^{(1)}_{src}\mathcal{L}^{src}_{CTC}+\lambda^{(1)}_{tgt}\mathcal{L}^{tgt}_{CTC}$, with weights 0.1/0.2. Stage II inserts LoRA (rank 8, $\alpha=32$, dropout 0.1) into Qwen3 and reduces CTC weights to 0.01/0.05. Experiments use Whisper-Large-V3, Qwen3-4B, batch size 24, and mixed precision on two NVIDIA A6000s. CVSS-C training covers approximately 561 hours across French, German, and Spanish.

## Key Experimental Results

### Main Results
| Model | Fr→En BLEU / ASR-BLEU | De→En BLEU / ASR-BLEU | Es→En BLEU / ASR-BLEU | Avg BLEU | Avg ASR-BLEU |
| :--- | :--- | :--- | :--- | :--- | :--- |
| RosettaSpeech | 33.11 / 32.16 | 23.22 / 21.54 | 30.92 / 29.35 | 29.08 | 27.68 |
| S2ST-Omni | 35.83 / 33.20 | 33.34 / 31.25 | 37.85 / 35.90 | 35.67 | 33.45 |
| S2ST-Omni 2 | 37.83 / 34.72 | 35.70 / 33.16 | 39.62 / 37.13 | 37.73 | 35.00 |
| Whisper-Qwen S2TT ref | 35.15 / - | 36.07 / - | 38.39 / - | 36.54 | - |

### Ablation Study
| Configuration | Avg BLEU | Avg ASR-BLEU | Relative Meaning vs. Full Model |
| :--- | :--- | :--- | :--- |
| S2ST-Omni 2 | 37.73 | 35.00 | Full typological conditions |
| w/o DG | 36.96 | 34.07 | Static instead of dynamic gate (-0.77 BLEU) |
| w/o TA-Prompt | 36.80 | 33.96 | No typological prompt (-0.93 BLEU) |
| w/o TI-HLE | 36.09 | 33.68 | 320d flat embedding instead of structural (Largest drop) |
| w/o Morph | 36.23 | 33.75 | Removed morphology channel |
| w/o Reorder | 36.60 | 33.94 | Removed reordering channel |
| w/o Family | 36.44 | 33.87 | Removed genealogical family channel |
| w/o Residual | 36.21 | 33.74 | Removed language-specific residual channel |

### Key Findings
*   Compared to S2ST-Omni, S2ST-Omni 2 improves Avg BLEU from 35.67 to 37.73 (+5.8% Gain) and Avg ASR-BLEU from 33.45 to 35.00 (+4.6% Gain).
*   Avg COMET increased from 82.02 to 83.31; BLASER 2.0 increased from 4.14 to 4.24.
*   Advantages are more pronounced under low-resource budgets: when training data was reduced to 30 hours, the absolute BLEU gain increased from +2.06 to +3.93 (~15.1% relative Gain).
*   In a Japanese-to-English experiment with ~3 hours of data, S2ST-Omni 2 outperformed S2ST-Omni across all metrics (22.00 vs 19.61 BLEU).
*   Improvements are consistent across different TTS backends, with ASR-BLEU ranging between 33.87 and 35.00.

## Highlights & Insights
*   The transition from "category ID" to "structural prior" is the paper's core innovation. This provides a transferable bias essential for low-resource multilingual tasks.
*   The residual channel in TI-HLE is critical to avoid "over-typologization," preserving unique linguistic traits that are difficult to categorize.
*   DG-LA-Dual-CTC demonstrates that linguistic structure influences not just text generation, but how acoustic features are aligned and compressed.
*   Qualitative examples show benefits for specific structures: German compound words, Spanish passive structures, and French idioms.

## Limitations & Future Work
*   Experiments are primarily focused on French, German, and Spanish; Japanese is only tested in a 3-hour low-resource setup.
*   Typological profiles are currently coarse-grained, manual groupings; future work requires more systematic schemas for scaling to more languages.
*   The task is restricted to many-to-one English S2ST; many-to-many or non-English targets remain unexplored.
*   TTS backends cause fluctuations in ASR-BLEU (up to 1.13), indicating stability still depends on the synthesizer.

## Related Work & Insights
*   **vs S2ST-Omni**: S2ST-Omni 2 proves that improvements stem from the representation of language information rather than just the backbone architecture.
*   **vs cascaded ASR-MT-TTS**: S2ST-Omni 2 outperforms the Whisper-Qwen S2TT reference in Avg BLEU while mitigating error propagation inherent in cascades.
*   **vs End-to-End S2ST**: The compositional design retains TTS pluggability and facilitates analysis of how language conditions affect the front-end.

## Rating
*   Novelty: ⭐⭐⭐⭐☆ Typological priors for SpeechLLM S2ST conditioning are well-targeted.
*   Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers main results, ablations, data budgets, and multiple backends, though language scope is somewhat limited.
*   Writing Quality: ⭐⭐⭐⭐☆ Clear motivation and technical breakdown.
*   Value: ⭐⭐⭐⭐☆ Practical for low-resource multilingual S2ST; encourages alternatives to flat language IDs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Scalable Multilingual Multimodal Machine Translation with Speech-Text Fusion](../../ICLR2026/audio_speech/scalable_multilingual_multimodal_machine_translation_with_speech-text_fusion.md)
- [\[ACL 2026\] VAPO: End-to-end Slide-Enhanced Speech Recognition with Omni-modal Large Language Models](vapo_end-to-end_slide-enhanced_speech_recognition_with_omni-modal_large_language.md)
- [\[ACL 2026\] Closing the Modality Reasoning Gap for Speech Large Language Models](closing_the_modality_reasoning_gap_for_speech_large_language_models.md)
- [\[ACL 2026\] Towards Fine-Grained and Multi-Granular Contrastive Language-Speech Pre-training](towards_fine-grained_and_multi-granular_contrastive_language-speech_pre-training.md)
- [\[ACL 2026\] Do We Need Distinct Representations for Every Speech Token? Unveiling and Exploiting Redundancy in Large Speech Language Models](do_we_need_distinct_representations_for_every_speech_token_unveiling_and_exploit.md)

</div>

<!-- RELATED:END -->

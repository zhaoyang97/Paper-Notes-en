---
title: >-
  [Paper Note] Just Go Parallel: Improving the Multilingual Capabilities of Large Language Models
description: >-
  [ACL 2025][Multilingual & Machine Translation][parallel data] This work systematically investigates the impact of incorporating parallel data during decoder-only LLM training on multilingual capabilities. It finds that applying parallel data at the final stage of training yields the best performance, significantly outperforming an equivalent amount of monolingual data. Furthermore, LLMs fail to automatically generalize to the reverse direction of the trained translation direc…
tags:
  - "ACL 2025"
  - "Multilingual & Machine Translation"
  - "parallel data"
  - "multilingual LLM"
  - "continual pre-training"
  - "translation"
  - "cross-lingual alignment"
date: 2026-05-08
content_hash: ea2a6b94011e7eeb
---

# Just Go Parallel: Improving the Multilingual Capabilities of Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2506.13044](https://arxiv.org/abs/2506.13044)  
**Code**: [nusnlp/just-go-parallel](https://github.com/nusnlp/just-go-parallel)  
**Area**: Multilingual / MT  
**Keywords**: parallel data, multilingual LLM, continual pre-training, translation, cross-lingual alignment

## TL;DR

This work systematically investigates the impact of incorporating parallel data during decoder-only LLM training on multilingual capabilities. It finds that applying parallel data at the final stage of training yields the best performance, significantly outperforming an equivalent amount of monolingual data. Furthermore, LLMs fail to automatically generalize to the reverse direction of the trained translation direction (reversal curse).

---

## Background & Motivation

- **Background**: LLMs exhibit translation capabilities even without explicit exposure to parallel data, which is widely attributed to incidental bilingual signals in the training corpus. Consequently, some mainstream multilingual LLMs (such as BLOOM) voluntarily choose to exclude parallel data.
- **Limitations of Prior Work**: While parallel data has been proven highly effective for cross-lingual transfer in encoder models (such as XLM), whether decoder-only LLMs can similarly benefit from it remains unclear. Furthermore, if beneficial, the optimal positioning (start / distributed / end) and data format (bidirectional / unidirectional) are yet to be determined.
- **Key Challenge**: Decoder LLMs typically omit parallel data during pre-training—is this a costly misstep?
- **Key Insight**: Under the constraint of a fixed total training volume, seven controlled experiments were designed (no parallel / monolingual replacement / non-adjacent / front-loaded / distributed / end-loaded bidirectional / end-loaded unidirectional) to systematically answer "whether parallel data is useful, and how to utilize it best."
- **Core Idea**: Incorporating parallel data in the final stage as a "second stage of training" provides the greatest improvement in translation quality and multilingual common sense reasoning. Placing it at the beginning leads to catastrophic forgetting, whereas LLMs are unable to automatically generalize to the reverse translation direction.

---

## Method

### Overall Architecture

The total training volume is fixed at ~167B tokens. An equivalent amount of parallel data is introduced at different training stages (replacing an equivalent volume of non-parallel data at the end) to compare translation performance (BLEU) and multilingual common sense reasoning (Accuracy). The base model is TinyLlama 1.1B. Non-parallel data is drawn from a subset of SlimPajama, and parallel data covers Chinese-English (33.9M sentence pairs, ~2.8B tokens) and Indonesian-English (54.1M sentence pairs, ~2.1B tokens), formatted as "{source lang}: {src}\n{target lang}: {tgt}".

### Key Designs

1. **Seven Parallel Data Insertion Strategies**
    - **No Parallel**: No parallel data is used (baseline), simulating a typical English-centric LLM.
    - **Multilingual**: Incorporating an equivalent amount of target language **monolingual** data (control group) to evaluate the impact of pure language exposure.
    - **Parallel Non-Adjacent**: Parallel sentence pairs are present but **non-adjacently** aligned (source sentences concatenated with random English sentences) to isolate the exact role of "semantic alignment".
    - **Parallel First / Distributed / Last (all) / Last (uni)**: Placing adjacent parallel sentence pairs at the start / uniformly distributed / at the end (bidirectional) / at the end (unidirectional) of training, respectively, to isolate the effects of positioning and translation directionality.
    - **Design Motivation**: A controlled variable framework to isolate the four factors: parallelism, adjacency, positioning, and directionality step-by-step.

2. **Strict Training Volume Control**
    - When parallel data is introduced, an equivalent amount of non-parallel data is removed from the end of the training corpus to keep the total token count and training sequence identical.
    - Every setting is trained to ~167B tokens using identical GPU resources to eliminate interference from variations in data scale.
    - **Design Motivation**: To ensure that any variance in model capabilities is purely attributable to "parallel vs. non-parallel" and the "insertion stage".

3. **Multi-dimensional Evaluation Schema**
    - **Translation**: WMT-2023 Chinese-English test set + Flores-200 Indonesian-English devtest, evaluated under zero-shot & 5-shot settings.
    - **Common Sense Reasoning**: 8 English benchmarks (ARC, HellaSwag, BoolQ, etc.), 4 Chinese benchmarks (XWinograd, XStoryCloze, etc.), and 2 Indonesian benchmarks.
    - **Ablation**: Parallel data quality filtering (CometKiwi-2022), generalization of reverse translation, and catastrophic forgetting curves.
    - **Design Motivation**: To move beyond evaluating translation capabilities alone, exploring the wider influence of parallel data on non-translation reasoning tasks.

---

## Key Experimental Results

### Main Results — Translation Performance (BLEU)

| Setting | EN→ID | ID→EN | EN→ZH | ZH→EN | Description |
|------|-------|-------|-------|-------|------|
| No Parallel | 2.49 | 1.52 | 0.80 | 1.30 | Baseline |
| Multilingual | 2.38 | 5.92 | 0.81 | 3.72 | Monolingual data helps, but very limitedly |
| Parallel Non-Adjacent | 1.98 | 14.69 | 1.01 | 4.50 | Non-adjacent -> poor effect |
| Parallel First | 7.42 | 5.57 | 9.64 | 2.71 | **Severe catastrophic forgetting** |
| Parallel Distributed | 21.95 | 27.48 | 12.08 | 7.40 | Distributed yields moderate performance |
| **Parallel Last (all)** | **35.91** | **35.36** | **9.62** | **10.73** | **End-loaded bidirectional is best** |
| **Parallel Last (uni)** | **44.19** | **41.91** | **28.51** | **16.10** | **Unidirectional is even higher** |
| BLOOM 1.1B | 2.19 | 18.39 | 2.27 | 4.58 | Comparative reference |
| NLLB 1.3B | 44.64 | 43.06 | 27.58 | 19.25 | Dedicated translation model |

> Zero-shot results. Parallel Last (uni) is close to the specialized translation model NLLB, but loses the ability to translate in other directions.

### Reverse Translation Generalization Experiments

| Training Direction | EN→ID | ID→EN | EN→ZH | ZH→EN |
|---------|-------|-------|-------|-------|
| All directions | 35.91 | 35.36 | 9.62 | 10.73 |
| EN→ID only | **44.19** | 0.07 | 0.77 | 0.21 |
| ID→EN only | 0.02 | **41.91** | 0.25 | 0.03 |
| EN→ZH only | 0.09 | 0.59 | **28.51** | 0.01 |
| ZH→EN only | 0.00 | 2.73 | 0.13 | **16.10** |

> Models trained on a single direction are completely unable to translate in the reverse direction, confirming that the reversal curse also holds for translation tasks.

### Ablation Study — Key Findings

| Finding | Evidence |
|------|------|
| Parallel >> Monolingual | Parallel Last (all) vs. Multilingual: EN→ID +33.5 BLEU |
| End-loaded >> Front-loaded | Parallel Last vs. Parallel First: EN→ID +28.5 BLEU |
| Adjacent >> Non-adjacent | Parallel Distributed vs. Non-Adjacent: EN→ID +20.0 BLEU |
| No Reverse Generalization | Zero-shot reverse translation BLEU ≈ 0 for unidirectional training |
| Data Quality Filtering | Significant improvement for Chinese-English zero-shot (+14.6 BLEU), with no obvious benefit for few-shot |

---

## Key Findings

- **End-loaded is Optimal**: Training on parallel data as a "second stage" in the final phase yields the best results. This is likely because the model first establishes a solid language comprehension foundation using massive English data before efficiently learning cross-lingual mappings via parallel data.
- **Catastrophic Forgetting**: Front-loading parallel data results in a complete loss of translation ability over subsequent training, with BLEU dropping from ~20 to nearly 0.
- **Reversal Curse in Translation**: Training on EN→ZH does not improve ZH→EN performance (BLEU ≈ 0), challenging the assumption that "LLMs possess language-agnostic representations", which aligns with the reversal curse.
- **Extra Value of Parallel Data**: Compared to an equivalent amount of monolingual data, the alignment signals offered by parallel data bring supplementary gains in both translation and multilingual reasoning tasks.
- **Training Position > Volume**: Distributed and Last strategies employ the same volume of data, yet Last yields significantly higher translation performance.

---

## Rating

| Dimension | Rating (1-10) | Explanation |
|------|-----------|------|
| Experimental Design | 9 | Extremely rigorous across 7 controlled experiments, achieving sufficient isolation of variables |
| Novelty | 6 | The core problem is not new, but the systematic study and several counter-intuitive findings add strong value |
| Practicality | 8 | The "end-loaded parallel data" strategy is simple, directly applicable, and has been adopted by models like Tower |
| Writing Quality | 8 | Clear description of experiments, rich tables and figures, and distinct conclusions |

---

## Highlights & Insights & Limitations

**Highlights**:

- The controlled experiment design is textbook-level—fixing data size and sequence while varying single factors.
- "Lack of reverse translation generalization" is an important negative finding that sheds light on understanding the cross-lingual representation mechanism of LLMs.
- The end-loaded strategy has been validated and adopted by subsequent works such as Tower and Pangea.

**Limitations**:

- Verified only on TinyLlama with 1.1B parameters; whether the conclusions generalize to larger models (7B+) remains unknown.
- Limited language coverage (only Chinese and Indonesian), lacking experiments on low-resource and morphologically rich languages.
- The interaction effect between parallel data and instruction tuning was not explored.

## Related Work & Insights
- **vs CLO (Lee et al.)**: CLO utilizes DPO for cross-lingual transfer, while this work uses parallel data for continual pre-training—the two are complementary.
- **vs XLM/mBART**: Parallel data strategies for encoder/encoder-decoder models are mature, and this work extends them to decoder-only architectures.

## Rating
- Novelty: ⭐⭐⭐⭐ Systematic comparative study + multiple important findings
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 7 settings × multiple languages × multiple tasks, extremely thorough
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous experimental design, clear conclusions
- Value: ⭐⭐⭐⭐⭐ Directly guides the training of multilingual LLMs

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Do Large Language Models Have an English Accent? Evaluating and Improving the Naturalness of Multilingual LLMs](multilingual_llm_english_accent.md)
- [\[ACL 2025\] ShifCon: Enhancing Non-Dominant Language Capabilities with a Shift-based Multilingual Contrastive Framework](shifcon_nondominant_language.md)
- [\[ACL 2025\] Disentangling Language and Culture for Evaluating Multilingual Large Language Models](disentangle_language_culture.md)
- [\[ACL 2025\] Cross-Lingual Pitfalls: Automatic Probing Cross-Lingual Weakness of Multilingual Large Language Models](crosslingual_pitfalls.md)
- [\[ACL 2025\] Bridging the Language Gaps in Large Language Models with Inference-Time Cross-Lingual Intervention](bridging_the_language_gaps_in_large_language_models_with_inference-time_cross-li.md)

</div>

<!-- RELATED:END -->

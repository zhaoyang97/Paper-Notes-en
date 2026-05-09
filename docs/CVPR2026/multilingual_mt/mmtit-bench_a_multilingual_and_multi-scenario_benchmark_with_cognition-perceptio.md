---
title: >-
  [Paper Note] MMTIT-Bench: A Multilingual and Multi-Scenario Benchmark with Cognition-Perception-Reasoning Guided Text-Image Machine Translation
description: >-
  [CVPR 2026][Text-image translation] This paper constructs MMTIT-Bench, a multilingual multi-scenario text-image translation benchmark covering 14 non-English non-Chinese languages, and proposes the CPR-Trans data paradigm (Cognition → Perception → Translation Reasoning). The approach significantly improves end-to-end translation quality on 3B and 7B models, with the 7B model achieving performance competitive with a 235B model.
tags:
  - CVPR 2026
  - Text-image translation
  - multilingual benchmark
  - chain-of-thought
  - cognition-perception-reasoning
  - VLLM evaluation
date: 2026-05-08
content_hash: 6f64b68fac95fa8e
---

# MMTIT-Bench: A Multilingual and Multi-Scenario Benchmark with Cognition-Perception-Reasoning Guided Text-Image Machine Translation

**Conference**: CVPR 2026
**arXiv**: [2603.23896](https://arxiv.org/abs/2603.23896)
**Code**: None (MMTIT-Bench planned for release)
**Area**: Multimodal VLM / Machine Translation
**Keywords**: Text-image translation, multilingual benchmark, chain-of-thought, cognition-perception-reasoning, VLLM evaluation

## TL;DR

This paper constructs MMTIT-Bench, a multilingual multi-scenario text-image translation benchmark covering 14 non-English non-Chinese languages, and proposes the CPR-Trans data paradigm (Cognition → Perception → Translation Reasoning). The approach significantly improves end-to-end translation quality on 3B and 7B models, with the 7B model achieving performance competitive with a 235B model.

## Background & Motivation

1. **Background**: Text-image machine translation (TIMT) aims to directly translate textual content embedded in images. With the advancement of VLLMs, end-to-end TIMT has replaced the traditional OCR+NMT cascade pipeline; however, existing research primarily focuses on English-Chinese pairs and evaluates mostly on simple scenarios such as digital documents.
2. **Limitations of Prior Work**: (1) No evaluation benchmark covers multiple languages and diverse scenarios — the largest existing dataset (MTIT6) covers only 4 languages in limited scene types; (2) chain-of-thought (CoT) reasoning paradigms tailored for TIMT remain underdeveloped — existing methods either cascade OCR with translation or rely solely on linguistic reasoning, neglecting visual cognition.
3. **Key Challenge**: VLLMs perform well on high-resource languages but their robustness on low-resource languages and complex visual scenes (menus, posters, street views) remains unknown, and no suitable benchmark exists for systematic evaluation.
4. **Goal**: (1) Construct a TIMT benchmark covering multiple languages and scenarios; (2) design a reasoning data paradigm suitable for TIMT.
5. **Key Insight**: Simulate the human translation process — first understand the scene (cognition) → recognize text (perception) → reason through translation (reasoning) — and design structured CoT supervision accordingly.
6. **Core Idea**: Guide end-to-end text-image translation using a three-stage structured reasoning chain: Cognition → Perception → Reasoning.

## Method

### Overall Architecture

The work comprises two components: (1) MMTIT-Bench construction — images are collected from 14 languages, annotated via OCR, labeled with translations, and manually filtered to yield 1,400 high-quality samples; (2) the CPR-Trans data paradigm — VLLM-assisted generation of structured three-stage reasoning chains (cognition, perception, translation reasoning) for training end-to-end TIMT models.

### Key Designs

1. **MMTIT-Bench Construction**:

    - **Function**: Provides a standardized TIMT evaluation platform across multiple languages and scenarios.
    - **Mechanism**: (a) Approximately 14,000 real images containing text (menus, posters, documents, etc.) are manually collected from 14 languages; (b) Gemini 2.5 Flash assists OCR annotation with human verification, supporting Markdown tables and LaTeX formulas; (c) multi-temperature sampling and three-model voting (Gemini, Seed1.6, Qwen3-VL) are used to generate translations; (d) 100 images per language (1,400 total) are curated and reviewed by language experts. Both Chinese and English translations are provided per image.
    - **Design Motivation**: Coverage of 14 languages including German, Spanish, Turkish, Vietnamese, Korean, Malay, Portuguese, Russian, French, Indonesian, Thai, Italian, and Japanese addresses the insufficient language and scenario coverage of existing benchmarks.

2. **CPR-Trans Data Paradigm**:

    - **Function**: Provides structured, interpretable reasoning supervision to improve translation quality.
    - **Mechanism**: Three-stage reasoning — the `<cognition>` stage describes the global visual scene without recognizing text; the `<perception>` stage analyzes the spatial layout and reading order of text regions; the `<trans>` stage integrates visual and textual understanding to reason through the translation. The full reasoning chain is enclosed in `<think></think>` tags, with the final translation in `<answer></answer>`. All stages are generated by Qwen3-VL-235B.
    - **Design Motivation**: Direct translation discards OCR perception (the model cannot observe its own recognition process); Simple CoT that concatenates OCR output lacks reasoning; native thinking is uncontrollable and prone to redundant repetition. CPR-Trans simulates the human cognitive process for translation and provides precise supervision.

3. **Dual-Protocol Evaluation Framework**:

    - **Function**: Comprehensively evaluates translation quality from multiple perspectives.
    - **Mechanism**: (a) VLLM-as-judge — Gemini 2.5 Flash and Qwen3-VL-235B score outputs along four dimensions: fidelity, fluency, readability, and terminology consistency; (b) rule-based metrics — COMET for automatic evaluation. The two protocols show high agreement.
    - **Design Motivation**: VLLM judges align with human judgment but may introduce bias; traditional metrics are objective but may overlook semantic quality. The two approaches are complementary and jointly ensure evaluation reliability.

### Loss & Training

- Training data: 12,600 manually annotated samples + 70,000 SynthDog synthetic samples, totaling 165,200 aligned multimodal pairs.
- Qwen2.5-VL-3B and 7B serve as backbone models for SFT.
- CPR-Trans reasoning chains are generated stage-by-stage by Qwen3-VL-235B.

## Key Experimental Results

### Main Results

Model performance on MMTIT-Bench (Gemini-Flash Judge, other→en / other→zh):

| Model | Params | Think | other2en | other2zh |
|-------|--------|-------|----------|----------|
| Cascade (MinerU+Qwen3) | - | - | 48.32 | 49.70 |
| Qwen3-VL-Instruct | 235B | - | 64.39 | 69.67 |
| Qwen3-VL-Thinking | 235B | ✓ | 73.81 | 77.90 |
| Gemini 2.5 Flash | - | ✓ | **82.94** | **85.00** |
| **Qwen2.5-VL + CPR-Trans** | **7B** | **✓** | **83.98** | **82.84** |

The 7B CPR-Trans model surpasses Gemini 2.5 Flash on the other→en direction.

### Ablation Study

Comparison of different data paradigms (7B model, Gemini-Flash Judge):

| Paradigm | other2en | other2zh | Description |
|----------|----------|----------|-------------|
| Origin (no fine-tuning) | 53.98 | 46.89 | Baseline |
| Direct (direct translation) | 68.40 | 62.42 | Loses perceptual ability |
| Simple CoT (OCR+translation) | 74.65 | 71.03 | Lacks reasoning |
| Distillation (VLLM) | 71.90 | 69.91 | Native thinking chain |
| **CPR-Trans** | **83.98** | **82.84** | Structured reasoning, best |

Ablation of reasoning components (7B, Gemini judge, other2en):

| Cognition | Perception | Trans | Score |
|-----------|------------|-------|-------|
| - | - | - | 74.65 (baseline) |
| ✓ | - | - | 76.91 |
| - | - | ✓ | 80.73 |
| ✓ | - | ✓ | 82.11 |
| - | ✓ | ✓ | 81.90 |
| **✓** | **✓** | **✓** | **83.98** |

### Key Findings

- **The Trans (translation reasoning) component contributes most** (+6.08 vs. baseline), indicating that an explicit translation reasoning process is central to performance gains.
- **The Cognition component** alone yields +2.26, demonstrating that understanding the global scene aids translation disambiguation.
- **The Perception component** alone provides negligible gain (+0.22↓), yet its contribution becomes apparent in combination with other components — its value lies in supplying structured textual information for downstream reasoning.
- The thinking mode consistently outperforms the non-thinking mode within the same model family, confirming the importance of explicit reasoning for TIMT.
- The cascade approach (OCR+LLM) is substantially inferior to end-to-end approaches; error propagation is especially severe in complex scenarios.

## Highlights & Insights

- **Small model outperforms large model**: The 7B CPR-Trans model surpasses Gemini 2.5 Flash on the other→en direction, suggesting that the value of high-quality reasoning data may exceed that of scaling model size alone. This provides an important insight for resource-constrained settings.
- **Generalizability of the Cognition-Perception-Reasoning paradigm**: The method of decomposing complex tasks into cognition, perception, and reasoning stages is applicable not only to TIMT but also transferable to tasks requiring joint visual-linguistic reasoning, such as document understanding and OCR correction.
- **Benchmark construction methodology**: The annotation pipeline combining multi-model voting with expert final review, and the dual-track evaluation using VLLM judges alongside rule-based metrics, provides a replicable template for future multimodal benchmark construction.

## Limitations & Future Work

- Among the 14 languages, medium-to-high resource languages predominate; truly low-resource languages (e.g., Burmese, Swahili) are absent.
- With only 100 test samples per language, statistical significance may be limited.
- CPR-Trans reasoning chains depend on the 235B model for generation, so data quality is bounded by the teacher model's capability.
- RL-based fine-tuning (e.g., GRPO/DPO) to further improve reasoning quality remains unexplored.
- Domain shift between synthetic data (SynthDog) and real-world scenes persists.

## Related Work & Insights

- **vs. MTIT6**: MTIT6 covers 4 languages with 1,200 samples; MMTIT-Bench extends to 14 languages and 1,400 samples, encompassing more diverse scene types and longer average text (160 words vs. 7 words).
- **vs. DoTA/PATIMT**: These benchmarks focus exclusively on English-Chinese document translation. MMTIT-Bench's multi-scenario design (menus, posters, tourist sites) better reflects real-world usage.
- **vs. R1-style thinking**: Native long-CoT reasoning is effective but uncontrollable and prone to redundancy. CPR-Trans's structured design provides precise guidance and avoids the "repetitive reflection" problem.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The CPR-Trans paradigm is elegantly designed and the benchmark construction pipeline is thorough, though the core idea (structured CoT) is not entirely novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Broad model evaluation and detailed ablation analysis, though fine-grained cross-language analysis is absent.
- **Writing Quality**: ⭐⭐⭐⭐ Structure is clear and figures are informative, though the paper feels slightly crowded by devoting roughly equal space to benchmark construction and methodology.
- **Value**: ⭐⭐⭐⭐ Fills the gap in multilingual TIMT evaluation; the CPR-Trans paradigm has broad transfer potential.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] SEA-Vision: A Multilingual Benchmark for Document and Scene Text Understanding in Southeast Asia](sea-vision_a_multilingual_benchmark_for_comprehensive_document_and_scene_text_un.md)
- [\[AAAI 2026\] Consensus-Aligned Neuron Efficient Fine-Tuning Large Language Models for Multi-Domain Machine Translation](../../AAAI2026/multilingual_mt/consensus-aligned_neuron_efficient_fine-tuning_large_language_models_for_multi-d.md)
- [\[NeurIPS 2025\] Reflective Translation: Improving Low-Resource Machine Translation via Structured Self-Reflection](../../NeurIPS2025/multilingual_mt/reflective_translation_improving_low-resource_machine_translation_via_structured.md)
- [\[NeurIPS 2025\] MERIT: Multilingual Semantic Retrieval with Interleaved Multi-Condition Query](../../NeurIPS2025/multilingual_mt/merit_multilingual_semantic_retrieval_with_interleaved_multi-condition_query.md)
- [\[AAAI 2026\] X-MuTeST: A Multilingual Benchmark for Explainable Hate Speech Detection and A Novel LLM-consulted Explanation Framework](../../AAAI2026/multilingual_mt/x-mutest_a_multilingual_benchmark_for_explainable_hate_speech_detection_and_a_no.md)

<!-- RELATED:END -->

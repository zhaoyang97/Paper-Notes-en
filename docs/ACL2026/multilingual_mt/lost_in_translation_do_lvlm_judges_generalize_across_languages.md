---
title: >-
  [Paper Note] Lost in Translation: Do LVLM Judges Generalize Across Languages?
description: >-
  [ACL 2026][Multilingual & Machine Translation][Multilingual Evaluation] This paper proposes MM-JudgeBench, the first large-scale multilingual multimodal judging model benchmark (25 languages…
tags:
  - "ACL 2026"
  - "Multilingual & Machine Translation"
  - "Multilingual Evaluation"
  - "LVLM Judges"
  - "Reward Models"
  - "Cross-lingual Generalization"
  - "Vision-Language Benchmarks"
date: 2026-05-08
content_hash: 29a4fc714eec67cd
---

# Lost in Translation: Do LVLM Judges Generalize Across Languages?

**Conference**: ACL 2026 Findings  
**arXiv**: [2604.19405](https://arxiv.org/abs/2604.19405)  
**Code**: [https://github.com/tahmedge/mm-judgebench](https://github.com/tahmedge/mm-judgebench)  
**Area**: Multilingual / Model Evaluation  
**Keywords**: Multilingual Evaluation, LVLM Judges, Reward Models, Cross-lingual Generalization, Vision-Language Benchmarks

## TL;DR

This paper proposes MM-JudgeBench, the first large-scale multilingual multimodal judging model benchmark (25 languages, 60K+ preference instances). Evaluating 22 LVLMs reveals significant cross-lingual performance gaps—model size and architecture do not predict multilingual robustness. Even state-of-the-art judges exhibit inconsistency, highlighting the necessity for multilingual multimodal evaluation benchmarks.

## Background & Motivation

**Background**: Automatic evaluators (reward models/LLM-as-Judge) play a core role in LVLM development, from training alignment to model selection and benchmarking. However, existing evaluations are almost entirely English-based.

**Limitations of Prior Work**: (1) VL-RewardBench and Multimodal RewardBench only cover English; (2) Multilingual extensions (e.g., M-RewardBench) are limited to the text modality; (3) No existing benchmark can unifiedly study reward model behavior across languages and modalities.

**Key Challenge**: LVLM judges are expected to be used in multilingual multimodal settings, but their reliability is only verified in English. The same model may perform excellently in English but choose the wrong answer in French.

**Goal**: (1) Construct the first multilingual multimodal evaluation benchmark; (2) Evaluate cross-lingual evaluation consistency across 22 LVLMs at scale; (3) Reveal current multilingual limitations in reward modeling.

**Key Insight**: Use high-quality translation models (Gemini-3-Pro) to translate VL-RewardBench and OpenCQA into 24 languages (25 total including English), constructing controlled experiments after rigorous quality filtering.

**Core Idea**: Isolate cross-lingual evaluation effects by fixing visual input and varying only the language, revealing the vulnerability of LVLM judges in the linguistic dimension.

## Method

### Overall Architecture

A three-stage method: (1) Translation model selection—comparing translation quality of the Gemini series (LaBSE and CometKiwi metrics) to select Gemini-3-Pro; (2) Data construction—translating VL-RewardBench (vision-language preference judgment) and OpenCQA (chart-based Q&A judgment) into 24 languages, resulting in 60K+ instances after quality filtering; (3) Model evaluation—analysis of pairwise accuracy, position bias, and length bias for 22 LVLMs.

### Key Designs

1.  **MM-JudgeBench Dataset Construction**:

    *   **Function**: Provides the first multilingual multimodal evaluation benchmark.
    *   **Mechanism**: Two complementary subsets—M-VL-RewardBench (general vision-language preference evaluation) and M-OpenCQA (chart-centric vision-text reasoning evaluation). 25 typologically diverse languages ranging from Arabic to Vietnamese. Each prompt translates the query and two candidate answers into the target language. Quality filtering: samples with LaBSE and CometKiwi < 0.75 are checked via human back-translation and re-translated or deleted.
    *   **Design Motivation**: Existing benchmarks cannot reveal the multilingual vulnerability of LVLM judges. Translating all 24 languages in a single prompt (reducing API calls 24x) ensures cost-controllability.

2.  **Multi-dimensional Evaluation Protocol**:

    *   **Function**: Goes beyond accuracy to reveal biases and instruction-following failures.
    *   **Mechanism**: (1) Pairwise accuracy—the proportion of correctly identified preferred responses; (2) Position bias—the difference in accuracy when answers are presented in forward vs. reverse order; (3) Length bias—whether the model tends to choose longer but incorrect answers. Each pair of answers is presented twice (forward + reverse order) to detect position bias.
    *   **Design Motivation**: Looking only at accuracy hides systematic biases. Position and length biases can lead to serious systematic errors in practical deployment of judging models.

3.  **Multilingual Training Set M-MM-RewardBench**:

    *   **Function**: Supports multilingual domain adaptation for open-source models.
    *   **Mechanism**: Translates MM-RewardBench into 24 languages, yielding a training set of 100K+ preference instances non-overlapping with evaluation data. Used for fine-tuning open-source models to improve multilingual judging performance.
    *   **Design Motivation**: Open-source models perform poorly on multilingual judging; providing training data supports domain-adaptive fine-tuning.

### Loss & Training

Evaluation is zero-shot prompting, requiring LVLMs to choose the better answer and provide rationale. Domain-adaptive fine-tuning uses standard SFT on M-MM-RewardBench. The evaluation metric is pairwise accuracy.

## Key Experimental Results

### Main Results

**Average Accuracy and Variance of 22 LVLMs on MM-JudgeBench**

| Model | Average Accuracy | Variance | Notes |
|-------|------------------|----------|-------|
| GPT-5 | 81.3% | 0.2 | Most stable |
| Gemini-2.5-Flash | ~78% | Low | Close to GPT-5 |
| Qwen3-VL-32B | ~77% | Low | Best open-source |
| Gemma-3-27B | ~74% | Mid | Significant drop in some languages |
| InternVL-3.5-8B | ~70% | High | Large cross-lingual variation |
| LLaVA-Critic-7B | ~55% | High | Dedicated judge but English-only training |

### Ablation Study

| Configuration | Effect | Notes |
|---------------|--------|-------|
| English Evaluation | Highest | All models strongest in English |
| Low-resource (e.g., Kazakh) | Largest Drop | Insufficient training data coverage |
| Efficiency-optimized variants | Multilingual collapse | e.g., Gemini-Flash-Lite strong in English but poor in multilingual |
| + Reasoning Enhancement | Improvement | Requirement for rationale improves judgment |
| + Multilingual Fine-tuning | Significant Improvement | Domain adaptation is effective |

### Key Findings

*   Model size does not predict multilingual robustness—the small model Qwen3-VL is more consistent across languages than many larger models.
*   Efficiency-optimized variants (e.g., Flash-Lite) are close to full-sized versions in English but degrade severely in multilingual settings.
*   LLaVA-Critic (specifically trained judging model) performs poorly in multilingual tests because it was only trained on English.
*   Position bias and length bias are more severe in non-English languages.
*   Both domain-adaptive fine-tuning and reasoning-enhanced judging improve multilingual performance.

## Highlights & Insights

*   Reveals the multilingual "blind spots" of LVLM judges—overall average scores mask huge differences between languages.
*   The multilingual collapse of efficiency-optimized variants is an important practical warning—reducing costs may come at the expense of fairness.
*   The release of the M-MM-RewardBench training set provides direct support for the community to improve multilingual judging.

## Limitations & Future Work

*   Translation may introduce systematic bias (all translations from the same model).
*   25 languages still do not cover most of the world's languages.
*   Lack of analysis on how translation quality affects evaluation results.
*   Future work needs native multilingual (non-translated) evaluation data.

## Related Work & Insights

*   **vs VL-RewardBench**: English only; MM-JudgeBench extends to 25 languages.
*   **vs M-RewardBench**: Text modality only; MM-JudgeBench adds visual modality.
*   **vs Multimodal RewardBench**: English multimodal; MM-JudgeBench is simultaneously multilingual and multimodal.

## Rating

*   **Novelty**: ⭐⭐⭐⭐ Fills the gap in multilingual multimodal judging evaluation.
*   **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 22 models, 25 languages, 60K+ instances.
*   **Writing Quality**: ⭐⭐⭐⭐ Clear structure, well-articulated practical implications.
*   **Value**: ⭐⭐⭐⭐⭐ The release of the benchmark and training set provides sustained value to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] HelpSteer3-Preference: Open Human-Annotated Preference Data across Diverse Tasks and Languages](../../NeurIPS2025/multilingual_mt/helpsteer3-preference_open_human-annotated_preference_data_across_diverse_tasks_.md)
- [\[ACL 2026\] Why Do Multilingual Reasoning Gaps Emerge in Reasoning Language Models?](why_do_multilingual_reasoning_gaps_emerge_in_reasoning_language_models.md)
- [\[ACL 2026\] From Traditional Taggers to LLMs: A Comparative Study of POS Tagging for Medieval Romance Languages](from_traditional_taggers_to_llms_a_comparative_study_of_pos_tagging_for_medieval.md)
- [\[ACL 2026\] Hierarchical Policy Optimization for Simultaneous Translation of Unbounded Speech](hierarchical_policy_optimization_for_simultaneous_translation_of_unbounded_speec.md)
- [\[ACL 2026\] LQM: Linguistically Motivated Multidimensional Quality Metrics for Machine Translation](lqm_linguistically_motivated_multidimensional_quality_metrics_for_machine_transl.md)

</div>

<!-- RELATED:END -->

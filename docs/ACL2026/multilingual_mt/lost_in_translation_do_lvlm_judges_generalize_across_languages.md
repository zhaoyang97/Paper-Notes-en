---
title: >-
  [Paper Note] Lost in Translation: Do LVLM Judges Generalize Across Languages?
description: >-
  [ACL 2026][Multilingual & Machine Translation][Multilingual Evaluation] This paper introduces MM-JudgeBench, the first large-scale multilingual multimodal judge benchmark (25 languages, 60K+ preference instances)…
tags:
  - "ACL 2026"
  - "Multilingual & Machine Translation"
  - "Multilingual Evaluation"
  - "LVLM Judges"
  - "Reward Models"
  - "Cross-lingual Generalization"
  - "Vision-Language Benchmarks"
date: 2026-05-08
content_hash: f043baeefa08e06b
---

# Lost in Translation: Do LVLM Judges Generalize Across Languages?

**Conference**: ACL 2026
**arXiv**: [2604.19405](https://arxiv.org/abs/2604.19405)  
**Code**: [https://github.com/tahmedge/mm-judgebench](https://github.com/tahmedge/mm-judgebench)  
**Area**: Multilingual / Model Evaluation
**Keywords**: Multilingual Evaluation, LVLM Judges, Reward Models, Cross-lingual Generalization, Vision-Language Benchmarks

## TL;DR

This paper introduces MM-JudgeBench, the first large-scale multilingual multimodal judge benchmark (25 languages, 60K+ preference instances), evaluating 22 LVLMs and revealing significant cross-lingual performance disparities in current LVLM judges. Model size and architecture cannot predict multilingual robustness, and even state-of-the-art judges exhibit inconsistent behavior, underscoring the necessity of multilingual multimodal evaluation benchmarks.

## Background & Motivation

**Background**: Automatic evaluators (reward models / LLM-as-Judge) play a central role in LVLM development, spanning training alignment, model selection, and benchmarking. However, existing evaluations are conducted almost exclusively in English.

**Limitations of Prior Work**: (1) VL-RewardBench and Multimodal RewardBench cover English only; (2) multilingual extensions such as M-RewardBench are restricted to the text modality; (3) no existing benchmark jointly examines reward model behavior across both languages and modalities.

**Key Challenge**: LVLM judges are expected to operate in multilingual multimodal settings, yet their reliability has been validated only in English. The same model may perform well in English while selecting incorrect answers in French.

**Goal**: (1) Construct the first multilingual multimodal judge benchmark; (2) systematically evaluate the cross-lingual judgment consistency of 22 LVLMs at scale; (3) expose the multilingual limitations of current reward modeling approaches.

**Key Insight**: By fixing visual inputs and varying only the language, the cross-lingual evaluation effect is isolated, revealing the linguistic fragility of LVLM judges.

**Core Idea**: A high-quality translation model (Gemini-2.5-Pro) is used to translate VL-RewardBench and OpenCQA into 24 languages (25 including English), followed by rigorous quality filtering to enable controlled experiments.

## Method

### Overall Architecture

A three-stage pipeline: (1) **Translation model selection** — comparing translation quality across the Gemini model family using LaBSE and CometKiwi metrics, selecting Gemini-2.5-Pro; (2) **Data construction** — translating VL-RewardBench (vision-language preference judgment) and OpenCQA (chart-based question-answering judgment) into 24 languages, yielding 60K+ instances after quality filtering; (3) **Model evaluation** — pairwise accuracy, position bias, and length bias analysis across 22 LVLMs.

### Key Designs

1. **MM-JudgeBench Dataset Construction**:

    - **Function**: Provides the first multilingual multimodal judge benchmark.
    - **Mechanism**: Two complementary subsets — M-VL-RewardBench (general vision-language preference evaluation) and M-OpenCQA (chart-centric visual-textual reasoning evaluation). 25 typologically diverse languages are covered, ranging from Arabic to Vietnamese. Each prompt translates the query and two candidate responses into the target language. Quality filtering: instances with LaBSE or CometKiwi scores below 0.75 undergo manual back-translation inspection and are either re-translated or removed.
    - **Design Motivation**: Existing benchmarks cannot expose the multilingual fragility of LVLM judges. Translating all 24 languages in a single prompt reduces API calls by a factor of 24, keeping costs tractable.

2. **Multi-Dimensional Evaluation Protocol**:

    - **Function**: Goes beyond accuracy to reveal biases and instruction-following failures.
    - **Mechanism**: (1) *Pairwise accuracy* — the proportion of instances in which the preferred response is correctly identified; (2) *Position bias* — the difference in judgment accuracy between forward and reversed answer orderings; (3) *Length bias* — whether the model tends to favor longer but incorrect answers. Each answer pair is presented twice (forward + reversed) to detect position bias.
    - **Design Motivation**: Accuracy alone conceals systematic biases. Position and length biases can cause severe systematic errors in real-world judge deployment.

3. **Multilingual Training Set M-MM-RewardBench**:

    - **Function**: Supports multilingual domain adaptation for open-source models.
    - **Mechanism**: MM-RewardBench is translated into 24 languages, yielding a training set of 100K+ preference instances with no overlap with the evaluation data. Used to fine-tune open-source models for improved multilingual judgment performance.
    - **Design Motivation**: Open-source models perform poorly on multilingual judgment; providing training data enables domain-adaptive fine-tuning.

### Loss & Training

Evaluation is conducted in a zero-shot prompting setting, requiring LVLMs to select the better answer and provide a rationale. Domain-adaptive fine-tuning applies standard SFT on M-MM-RewardBench. The primary evaluation metric is pairwise accuracy.

## Key Experimental Results

### Main Results

**Average accuracy and variance of 22 LVLMs on MM-JudgeBench**

| Model | Avg. Accuracy | Variance | Notes |
|-------|--------------|----------|-------|
| GPT-5 | 81.3% | 0.2 | Most stable |
| Gemini-2.5-Flash | ~78% | Low | Close to GPT-5 |
| Qwen3-VL-32B | ~77% | Low | Best open-source |
| Gemma-3-27B | ~74% | Medium | Notable drops in some languages |
| InternVL-3.5-8B | ~70% | High | Large cross-lingual variance |
| LLaVA-Critic-7B | ~55% | High | Dedicated judge model but English-only training |

### Ablation Study

| Configuration | Effect | Notes |
|--------------|--------|-------|
| English evaluation | Highest | All models perform best in English |
| Low-resource languages (e.g., Kazakh) | Largest drop | Insufficient training data coverage |
| Efficiency-optimized variants | Multilingual collapse | e.g., Gemini-Flash-Lite is strong in English but poor multilingually |
| + Reasoning augmentation | Improvement | Requiring rationales improves judgment |
| + Multilingual fine-tuning | Significant improvement | Domain adaptation is effective |

### Key Findings

- Model size does not predict multilingual robustness — smaller models such as Qwen3-VL achieve more consistent multilingual performance than many larger counterparts.
- Efficiency-optimized variants (e.g., Flash-Lite) approach full-size performance in English but degrade severely in multilingual settings.
- LLaVA-Critic, a dedicated judge model trained exclusively in English, performs extremely poorly across other languages.
- Position bias and length bias are more pronounced in non-English languages.
- Both domain-adaptive fine-tuning and reasoning-augmented judgment improve multilingual performance.

## Highlights & Insights

- The paper exposes the multilingual "blind spots" of LVLM judges — aggregate average scores mask substantial cross-lingual disparities.
- The multilingual collapse of efficiency-optimized variants is an important practical warning: reducing cost may come at the expense of fairness.
- The release of the M-MM-RewardBench training set directly supports the community in improving multilingual judgment.

## Limitations & Future Work

- Translation may introduce systematic bias, as all translations originate from a single model.
- 25 languages still leaves the vast majority of the world's languages uncovered.
- The study does not analyze how translation quality affects evaluation outcomes.
- Future work should develop natively multilingual (non-translated) evaluation data.

## Related Work & Insights

- **vs. VL-RewardBench**: English only; MM-JudgeBench extends coverage to 25 languages.
- **vs. M-RewardBench**: Text modality only; MM-JudgeBench adds the visual modality.
- **vs. Multimodal RewardBench**: English multimodal; MM-JudgeBench is simultaneously multilingual and multimodal.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Fills the gap in multilingual multimodal judge evaluation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 22 models, 25 languages, 60K+ instances.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure with well-articulated practical implications of findings.
- **Value**: ⭐⭐⭐⭐⭐ The release of both the benchmark and training set provides lasting value to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Why Do Multilingual Reasoning Gaps Emerge in Reasoning Language Models?](why_do_multilingual_reasoning_gaps_emerge_in_reasoning_language_models.md)
- [\[NeurIPS 2025\] HelpSteer3-Preference: Open Human-Annotated Preference Data across Diverse Tasks and Languages](../../NeurIPS2025/multilingual_mt/helpsteer3-preference_open_human-annotated_preference_data_across_diverse_tasks_.md)
- [\[NeurIPS 2025\] DCAD-2000: A Multilingual Dataset across 2000+ Languages with Data Cleaning as Anomaly Detection](../../NeurIPS2025/multilingual_mt/dcad-2000_a_multilingual_dataset_across_2000_languages_with_data_cleaning_as_ano.md)
- [\[ACL 2026\] Hierarchical Policy Optimization for Simultaneous Translation of Unbounded Speech](hierarchical_policy_optimization_for_simultaneous_translation_of_unbounded_speec.md)
- [\[ACL 2026\] LQM: Linguistically Motivated Multidimensional Quality Metrics for Machine Translation](lqm_linguistically_motivated_multidimensional_quality_metrics_for_machine_transl.md)

</div>

<!-- RELATED:END -->

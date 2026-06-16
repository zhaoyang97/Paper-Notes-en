---
title: >-
  [Paper Note] Lost in Translation: Do LVLM Judges Generalize Across Languages?
description: >-
  [ACL 2026][Multilingual & Translation][Paper Note] This paper introduces MM-JudgeBench, the first large-scale multilingual multimodal evaluation benchmark (25 languages, 60K+ preference instances). Evaluating 22 LVLMs reveals significant cross-lingual performance gaps in current LVLM judges—model size and architecture do not predict multilingual robustness, and even st
tags:
  - ACL 2026
  - Multilingual & Translation
date: 2026-05-08
content_hash: 9c27c250f3647cfc
---
# Lost in Translation: Do LVLM Judges Generalize Across Languages?

**Conference**: ACL 2026 Findings  
**arXiv**: [2604.19405](https://arxiv.org/abs/2604.19405)  
**Code**: [https://github.com/tahmedge/mm-judgebench](https://github.com/tahmedge/mm-judgebench)  
**Area**: Multilingual / Model Evaluation  
**Keywords**: Multilingual Evaluation, LVLM Judge, Reward Model, Cross-lingual Generalization, Vision-Language Benchmarks

## TL;DR

This paper introduces MM-JudgeBench, the first large-scale multilingual multimodal evaluation benchmark (25 languages, 60K+ preference instances). Evaluating 22 LVLMs reveals significant cross-lingual performance gaps in current LVLM judges—model size and architecture do not predict multilingual robustness, and even state-of-the-art judges exhibit inconsistency, highlighting the necessity for multilingual multimodal evaluation benchmarks.

## Background & Motivation

**Background**: Automatic evaluators (Reward Models/LLM-as-Judge) play a central role in LVLM development, from training alignment to model selection and benchmarking. However, existing evaluations are almost entirely based on English.

**Limitations of Prior Work**: (1) VL-RewardBench and Multimodal RewardBench only cover English; (2) Multilingual extensions (e.g., M-RewardBench) are limited to the text modality; (3) No existing benchmark provides a unified study of cross-lingual and cross-modal reward model behavior.

**Key Challenge**: LVLM judges are expected to be used in multilingual multimodal settings, yet their reliability is only verified in English. The same model may perform excellently in English but select the wrong answer in French.

**Goal**: (1) Construct the first multilingual multimodal evaluation benchmark; (2) Evaluate the cross-lingual judgment consistency of 22 LVLMs at scale; (3) Reveal current multilingual limitations in reward modeling.

**Key Insight**: High-quality translation models (Gemini-3-Pro) are used to translate VL-RewardBench and OpenCQA into 24 languages (25 including English), followed by rigorous quality filtering to construct controlled experiments.

**Core Idea**: Isolate cross-lingual evaluation effects by fixing visual inputs and varying only the language, revealing the vulnerability of LVLM judges across linguistic dimensions.

## Method

### Overall Architecture

The process follows a pipeline of "translator selection, data generation, and multi-dimensional evaluation," while also producing a multilingual training set: (1) Translation model selection—comparing translation quality among Gemini series (using LaBSE and CometKiwi metrics), selecting Gemini-3-Pro; (2) Dataset construction—translating VL-RewardBench (vision-language preference judgment) and OpenCQA (chart-based Q&A judgment) into 24 languages, resulting in 60K+ instances after quality filtering to form MM-JudgeBench; (3) Multi-dimensional evaluation—analyzing pairwise accuracy, position bias, and length bias for 22 LVLMs; (4) Multilingual training set—translating MM-RewardBench into 24 languages to obtain M-MM-RewardBench with 100K+ instances for domain-specific fine-tuning of open-source models. The following three key designs correspond to dataset construction, evaluation protocols, and training sets.

### Key Designs

**1. MM-JudgeBench Dataset Construction: Fixing Vision, Varying Language to Isolate "Cross-lingual Vulnerability"**

Existing evaluation benchmarks either focus only on English (VL-RewardBench, Multimodal RewardBench) or extend to multiple languages while losing the visual modality (M-RewardBench). None simultaneously address both language and modality dimensions. Ours fills this gap with two complementary subsets: M-VL-RewardBench for general vision-language preference and M-OpenCQA for chart-centric vision-text reasoning. Each prompt translates the query and two candidate answers into the target language while keeping the image unchanged. Consequently, the only variable across 25 typologically diverse languages (from Arabic to Vietnamese) is the text; judge errors can thus be attributed solely to language rather than content.

To ensure cost-effectiveness, the authors translate all 24 languages simultaneously using a single prompt, reducing API costs 24-fold compared to language-by-language calls. Quality is ensured using a 0.75 threshold for both LaBSE and CometKiwi metrics; samples below this threshold are re-translated or deleted after manual back-translation review, resulting in 60K+ high-quality instances.

**2. Multi-dimensional Evaluation Protocol: Assessing Why Models are Correct**

Relying solely on pairwise accuracy (the proportion of correctly identified preferred responses) can hide systematic biases—a judge might "coincidentally" choose correctly while consistently favoring the first or longer answer, tendencies which amplify into stable errors in real deployments. Thus, the protocol quantifies two additional biases: position bias (by presenting each pair in both forward and reverse order and comparing accuracy differences) and length bias (checking if the model systematically favors longer but incorrect answers). Combined, these three metrics distinguish "true understanding" from "shortcuts."

**3. Multilingual Training Set M-MM-RewardBench: Providing a Path for Open-source Adaptation**

Experiments show that open-source judges drop most significantly in non-English settings; diagnostic analysis is insufficient without a solution. The authors translate MM-RewardBench into 24 languages, obtaining a training set of 100K+ preference instances that intentionally do not overlap with evaluation data, specifically for domain-adaptive fine-tuning of open-source models. Its value is verified in experiments—multilingual fine-tuning significantly restores judgment performance in non-English languages.

### Loss & Training

Evaluation uses zero-shot prompting, requiring LVLMs to select the better answer and provide a rationale. Domain-adaptive fine-tuning uses standard SFT on M-MM-RewardBench. The evaluation metric is pairwise accuracy.

## Key Experimental Results

### Main Results

**Average Accuracy and Variance of 22 LVLMs on MM-JudgeBench**

| Model | Average Accuracy | Variance | Description |
| :--- | :--- | :--- | :--- |
| GPT-5 | 81.3% | 0.2 | Most stable |
| Gemini-2.5-Flash | ~78% | Low | Close to GPT-5 |
| Qwen3-VL-32B | ~77% | Low | Best open-source |
| Gemma-3-27B | ~74% | Medium | Noticeable drop in some languages |
| InternVL-3.5-8B | ~70% | High | High cross-lingual variation |
| LLaVA-Critic-7B | ~55% | High | Specialized judge but English-only training |

### Ablation Study

| Configuration | Effect | Description |
| :--- | :--- | :--- |
| English Evaluation | Highest | All models strongest in English |
| Low-resource Languages (e.g., Kazakh) | Largest Drop | Insufficient training data coverage |
| Efficiency-optimized Variants | Multilingual Collapse | e.g., Gemini-Flash-Lite strong in English but poor in multilingual |
| + Reasoning Enhancement | Gain | Requiring rationales improves judgment |
| + Multilingual Fine-tuning | Significant Gain | Domain adaptation is effective |

### Key Findings

- Model size does not predict multilingual robustness—the small model Qwen3-VL is more consistent across languages than many larger models.
- Efficiency-optimized variants (e.g., Flash-Lite) are close to full-size versions in English but degrade severely in multilingual settings.
- LLaVA-Critic (a specialized judge) performs extremely poorly in multilingual settings due to being trained only in English.
- Position bias and length bias are more severe in non-English languages.
- Both domain-adaptive fine-tuning and reasoning-enhanced judgment improve multilingual performance.

## Highlights & Insights

- Revealed the multilingual "blind spots" of LVLM judges—overall average scores mask huge differences between languages.
- The multilingual collapse of efficiency-optimized variants is an important practical warning—reducing costs may come at the expense of fairness.
- The release of the M-MM-RewardBench training set provides direct support for the community to improve multilingual judgment.

## Limitations & Future Work

- Translation may introduce systematic bias (all translations come from the same model).
- 25 languages still do not cover the majority of world languages.
- No analysis of how translation quality specifically affects evaluation results.
- Future work needs native multilingual (non-translated) evaluation data.

## Related Work & Insights

- **vs VL-RewardBench**: English only; MM-JudgeBench extends to 25 languages.
- **vs M-RewardBench**: Text modality only; MM-JudgeBench adds visual modality.
- **vs Multimodal RewardBench**: English multimodal; MM-JudgeBench is simultaneously multilingual and multimodal.

## Rating

- Novelty: ⭐⭐⭐⭐ Fills the gap in multilingual multimodal judge evaluation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 22 models, 25 languages, 60K+ instances.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, well-articulated practical implications.
- Value: ⭐⭐⭐⭐⭐ The release of benchmarks and training sets offers lasting value to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Why Do Multilingual Reasoning Gaps Emerge in Reasoning Language Models?](why_do_multilingual_reasoning_gaps_emerge_in_reasoning_language_models.md)
- [\[NeurIPS 2025\] HelpSteer3-Preference: Open Human-Annotated Preference Data across Diverse Tasks and Languages](../../NeurIPS2025/multilingual_mt/helpsteer3-preference_open_human-annotated_preference_data_across_diverse_tasks_.md)
- [\[ACL 2025\] Accessible Machine Translation Evaluation For Low-Resource Languages](../../ACL2025/multilingual_mt/accessible_machine_translation_evaluation_for_low-resource_languages.md)
- [\[ACL 2026\] From Traditional Taggers to LLMs: A Comparative Study of POS Tagging for Medieval Romance Languages](from_traditional_taggers_to_llms_a_comparative_study_of_pos_tagging_for_medieval.md)
- [\[ACL 2025\] Lost in Multilinguality: Dissecting Cross-lingual Factual Inconsistency in Transformer Language Models](../../ACL2025/multilingual_mt/lost_in_multilinguality_dissecting_cross-lingual_factual_inconsistency_in_transf.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Lost in Translation: Do LVLM Judges Generalize Across Languages?
description: >-
  [ACL 2026 Findings][Multimodal VLM][Multilingual Evaluation] This paper introduces MM-JudgeBench, the first large-scale multilingual multimodal judgment benchmark (25 languages, 60K+ preference instances). Evaluating 22 LVLMs reveals significant cross-lingual performance gaps—model size and architecture do not predict multilingual robustness, and even state-of-the-art judges exhibit inconsistency, highlighting the necessity for multilingual multimodal evaluation benchmarks.
tags:
  - "ACL 2026 Findings"
  - "Multimodal VLM"
  - "Multilingual Evaluation"
  - "LVLM Judging"
  - "Reward Model"
  - "Cross-lingual Generalization"
  - "Vision-Language Benchmark"
date: 2026-05-08
content_hash: 4c051572a9fdc549
---

# Lost in Translation: Do LVLM Judges Generalize Across Languages?

**Conference**: ACL 2026 Findings  
**arXiv**: [2604.19405](https://arxiv.org/abs/2604.19405)  
**Code**: [https://github.com/tahmedge/mm-judgebench](https://github.com/tahmedge/mm-judgebench)  
**Area**: Multilingual / Model Evaluation  
**Keywords**: Multilingual Evaluation, LVLM Judging, Reward Model, Cross-lingual Generalization, Vision-Language Benchmark

## TL;DR

This paper introduces MM-JudgeBench, the first large-scale multilingual multimodal judgment benchmark (25 languages, 60K+ preference instances). Evaluating 22 LVLMs reveals significant cross-lingual performance gaps—model size and architecture do not predict multilingual robustness, and even state-of-the-art judges exhibit inconsistency, highlighting the necessity for multilingual multimodal evaluation benchmarks.

## Background & Motivation

**Background**: Automated evaluators (Reward Models/LLM-as-Judge) play a central role in LVLM development, shifting from training alignment to model selection and benchmarking. However, existing evaluations are almost entirely based on English.

**Limitations of Prior Work**: (1) VL-RewardBench and Multimodal RewardBench only cover English; (2) Multilingual extensions (e.g., M-RewardBench) are limited to the text modality; (3) No existing benchmark unifies the study of reward model behavior across languages and modalities.

**Key Challenge**: LVLM judges are expected to function in multilingual multimodal settings, yet their reliability is only verified in English. The same model may perform excellently in English but select the wrong answer in French.

**Goal**: (1) Construct the first multilingual multimodal judgment benchmark; (2) Evaluate the cross-lingual judgment consistency of 22 LVLMs at scale; (3) Reveal current multilingual limitations in reward modeling.

**Key Insight**: Utilize high-quality translation models (Gemini-3-Pro) to translate VL-RewardBench and OpenCQA into 24 languages (25 total including English), followed by rigorous quality filtering to construct controlled experiments.

**Core Idea**: Isolate cross-lingual evaluation effects by fixing visual inputs while varying only the language, revealing the vulnerability of LVLM judges across the language dimension.

## Method

### Overall Architecture

The overall pipeline involves "selecting a translator, generating data, and performing multi-dimensional evaluation," while simultaneously producing a multilingual training set: (1) Translation Model Selection—comparing translation quality of the Gemini series (using LaBSE and CometKiwi metrics) and selecting Gemini-3-Pro; (2) Dataset Construction—translating VL-RewardBench (vision-language preference judgment) and OpenCQA (chart-based QA judgment) into 24 languages, resulting in 60K+ instances after quality filtering to form MM-JudgeBench; (3) Multi-dimensional Evaluation—analyzing Pairwise Accuracy, Position Bias, and Length Bias for 22 LVLMs; (4) Multilingual Training Set—translating MM-RewardBench into 24 languages to create M-MM-RewardBench with 100K+ instances for domain adaptation fine-tuning of open-source models.

### Key Designs

**1. MM-JudgeBench Construction: Fixed Vision, Variable Language to Isolate "Cross-lingual Vulnerability"**

Existing judgment benchmarks either focus only on English (VL-RewardBench, Multimodal RewardBench) or expand multilingually while losing the visual modality (M-RewardBench). Ours fills this gap by simultaneously monitoring language and modality dimensions. Two complementary subsets are used: M-VL-RewardBench measures general vision-language preferences, and M-OpenCQA measures chart-centric vision-text reasoning. For each prompt, the query and two candidate answers are translated into the target language while the image remains unchanged. Consequently, the only variable across 25 typologically diverse languages (from Arabic to Vietnamese) is the text; judge errors can thus be attributed solely to language rather than content.

To maintain cost-efficiency, a single prompt is used to translate all 24 languages at once, reducing API overhead by 24x. Quality is ensured using LaBSE and CometKiwi with a 0.75 threshold; samples below this threshold are re-translated or deleted after human back-translation review, resulting in 60K+ high-quality instances.

**2. Multi-dimensional Evaluation Protocol: Beyond Accuracy**

Simply looking at Pairwise Accuracy (the proportion of correctly identified preferred responses) can hide systemic biases. A judge might "accidentally" choose correctly while consistently preferring the first or longer answer, tendencies that amplify into stable errors in real deployments. Therefore, the protocol quantifies two additional biases: Position Bias is measured by presenting answer pairs in both forward and reverse orders and calculating the accuracy delta; Length Bias checks if the model systematically favors longer but incorrect answers. Only the combination of these three metrics distinguishes "true understanding" from "shortcut-based guessing."

**3. Multilingual Training Set M-MM-RewardBench: A Path for Open-Source Model Adaptation**

Experiments found that open-source judges suffer the most in non-English settings. To provide a solution, MM-RewardBench was translated into 24 languages to create a training set of 100K+ preference instances, intentionally non-overlapping with the evaluation data. This set is specifically for domain adaptation fine-tuning, the value of which is validated by experiments showing significant recovery of judgment performance in non-English languages.

### Loss & Training

Evaluation is conducted via zero-shot prompting, requiring LVLMs to select the better answer and provide reasoning. Domain adaptation fine-tuning uses standard SFT on M-MM-RewardBench. The primary evaluation metric is Pairwise Accuracy.

## Key Experimental Results

### Main Results

**Average Accuracy and Variance of 22 LVLMs on MM-JudgeBench**

| Model | Avg Accuracy | Variance | Note |
|-------|--------------|----------|------|
| GPT-5 | 81.3% | 0.2 | Most stable |
| Gemini-2.5-Flash | ~78% | Low | Close to GPT-5 |
| Qwen3-VL-32B | ~77% | Low | Best open-source |
| Gemma-3-27B | ~74% | Medium | Significant drops in some languages |
| InternVL-3.5-8B | ~70% | High | Large cross-lingual variation |
| LLaVA-Critic-7B | ~55% | High | Specialized judge but English-only training |

### Ablation Study

| Configuration | Effect | Note |
|---------------|--------|------|
| English Evaluation | Highest | All models strongest in English |
| Low-resource (e.g., Kazakh) | Largest Drop | Insufficient training data coverage |
| Efficiency-optimized variants | Multilingual Collapse | e.g., Gemini-Flash-Lite strong in English, weak multilingually |
| + Reasoning Enhancement | Gain | Requiring reasoning improves judgment |
| + Multilingual Fine-tuning | Significant Gain | Domain adaptation is effective |

### Key Findings

- Model size does not predict multilingual robustness—the smaller Qwen3-VL is more consistent across languages than many larger models.
- Efficiency-optimized variants (e.g., Flash-Lite) match full-size versions in English but degrade severely in multilingual settings.
- LLaVA-Critic (a specialized judge) performs extremely poorly multilingually due to English-only training.
- Position bias and length bias are more severe in non-English languages.
- Both domain adaptation fine-tuning and reasoning-enhanced judgment improve multilingual performance.

## Highlights & Insights

- Revealed the multilingual "blind spots" of LVLM judges—overall averages mask massive disparities between languages.
- The multilingual collapse of efficiency-optimized variants is a critical practical warning—cost reduction may come at the expense of fairness.
- The release of M-MM-RewardBench provides direct support for the community to improve multilingual judgment.

## Limitations & Future Work

- Translation may introduce systemic biases (all translations from the same model).
- 24 languages still do not cover the majority of the world's languages.
- The impact of translation quality on evaluation results was not deeply analyzed.
- Native multilingual (non-translated) evaluation data is needed for future work.

## Related Work & Insights

- **vs VL-RewardBench**: English-only; MM-JudgeBench extends to 25 languages.
- **vs M-RewardBench**: Text-only; MM-JudgeBench adds the visual modality.
- **vs Multimodal RewardBench**: English multimodal; MM-JudgeBench is simultaneously multilingual and multimodal.

## Rating

- Novelty: ⭐⭐⭐⭐ Fills the gap in multilingual multimodal judgment evaluation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 22 models, 25 languages, 60K+ instances.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with well-articulated practical implications.
- Value: ⭐⭐⭐⭐⭐ The benchmark and training set release offer sustained value to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] VAUQ: Vision-Aware Uncertainty Quantification for LVLM Self-Evaluation](vauq_vision-aware_uncertainty_quantification_for_lvlm_self-evaluation.md)
- [\[ICML 2026\] Large Vision-Language Models Get Lost in Attention](../../ICML2026/multimodal_vlm/large_vision-language_models_get_lost_in_attention.md)
- [\[CVPR 2026\] Multi-Crit: Benchmarking Multimodal Judges on Pluralistic Criteria-Following](../../CVPR2026/multimodal_vlm/multi-crit_benchmarking_multimodal_judges_on_pluralistic_criteria-following.md)
- [\[CVPR 2026\] LVLM-Aided Alignment of Task-Specific Vision Models](../../CVPR2026/multimodal_vlm/lvlm-aided_alignment_of_task-specific_vision_models.md)
- [\[ACL 2026\] What Do Vision-Language Models Encode for Personalized Image Aesthetics Assessment?](what_do_vision-language_models_encode_for_personalized_image_aesthetics_assessme.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Truth Knows No Language: Evaluating Truthfulness Beyond English
description: >-
  [ACL 2025][LLM Safety][Truthfulness evaluation] The first professionally translated multilingual TruthfulQA benchmark (Basque, Catalan, Galician, Spanish) is constructed, revealing that cross-lingual truthfulness disparities in LLMs are smaller than expected, and that LLM-as-a-Judge aligns better with human judgment than multiple-choice metrics.
tags:
  - "ACL 2025"
  - "LLM Safety"
  - "Truthfulness evaluation"
  - "TruthfulQA"
  - "Multilingual"
  - "LLM-as-a-Judge"
  - "Low-resource languages"
date: 2026-05-08
content_hash: c9c79a6a1aeb69ff
---

# Truth Knows No Language: Evaluating Truthfulness Beyond English

**Conference**: ACL 2025  
**arXiv**: [2502.09387](https://arxiv.org/abs/2502.09387)  
**Code**: [github.com/hitz-zentroa/truthfulqa-multi](https://github.com/hitz-zentroa/truthfulqa-multi)  
**Area**: LLM Safety  
**Keywords**: Truthfulness evaluation, TruthfulQA, Multilingual, LLM-as-a-Judge, Low-resource languages

## TL;DR

The first professionally translated multilingual TruthfulQA benchmark (Basque, Catalan, Galician, Spanish) is constructed, revealing that cross-lingual truthfulness disparities in LLMs are smaller than expected, and that LLM-as-a-Judge aligns better with human judgment than multiple-choice metrics.

## Background & Motivation

TruthfulQA is the standard benchmark for evaluating LLM truthfulness, with the core concept of testing whether models mimic human false beliefs and misconceptions. However, this benchmark has prominent limitations:

**English-only**: Although some developers have performed machine translation, there is a lack of professionally translated versions and systematic cross-lingual evaluations.

**Controversial evaluation methods**: Whether the standard multiple-choice metric (MC2) is sufficient to measure truthfulness remains questionable, especially in non-English scenarios.

**Cultural bias**: TruthfulQA possesses a strong English/US cultural background, with many questions involving US laws, English proverbs, etc.

The motivation of this work is: Are LLMs equally truthful across different languages? If a misconception can be avoided in English, can it also be avoided in Basque?

The choice of target languages is also deliberate:
- **Basque**: An agglutinative language isolate with extremely scarce pre-training data for LLMs.
- **Catalan, Galician**: Low-resource Romance languages.
- **Spanish**: A relatively high-resource control group.

## Method

### Overall Architecture

The work comprises three main components:

1. **Professionally translated dataset**: Translating 817 questions of TruthfulQA into 4 target languages.
2. **Three evaluation methods**: Human evaluation, the MC2 multiple-choice metric, and LLM-as-a-Judge.
3. **Machine translation substitution experiments**: Verifying whether machine translation can replace professional translation.

### Key Designs

**Translation Strategy Choice**:
The team faced two options: (1) localizing to fit the target culture, or (2) retaining the original cultural context. They ultimately chose to preserve the original cultural context to maintain complete cross-lingual parallelism. Specific translation guidelines include:
- **Proverbs and misquotes**: Employing a literal translation strategy (e.g., directly translating "an apple a day").
- **Acronym misconceptions**: Keeping the original English terms and annotating "in English" within the question.
- **Fictional named entities**: Using existing translations (such as movie character names) or borrowing from English when no translation exists.
- All translations were conducted by professional translators who are native speakers of the target languages.

**Human Evaluation Design**:
- Evaluating 400 responses (4 models $\times$ 100 questions), covering truthfulness and informativeness.
- Utilizing binary labels (truthful/untruthful, informative/uninformative) instead of the scalar scores in the original paper.
- Adding additional guidelines for instruction-tuned models: extra information in long responses must be verified by the evaluators.
- Computing inter-annotator agreement using 50 overlapping annotations.

**LLM-as-a-Judge Training**:
- Base models: Llama 2 7B (existing fine-tuned version), Gemma 2 9B, Llama 3.1 8B.
- Training data: Original English data vs. all-language data including machine translation.
- Best configuration: Gemma 2 9B instruct + all-language translation data.
- The informativeness judge model was trained separately but was only effective for base models (as instruction models yielded almost no uninformative responses).

**MC2 Metric**:
- Measures the normalized probability of true answers out of all reference answers.
- Uses LM Evaluation Harness in a 6-shot setup.
- Chat models use a multi-turn conversation format.

### Loss & Training

The training of LLM-as-a-Judge uses a learning rate of 0.01 and runs for 5 epochs. The core objective is binary classification of truthfulness on (question, answer) pairs.

## Key Experimental Results

### Main Results

**Human Evaluation Results** (truthfulness percentage of 100 samples/model/language):

| Model | English | Spanish | Catalan | Galician | Basque |
|------|-----|-----|-----|-----|-------|
| Gemma-2-27b-it | 73% | 73% | 71% | 72% | 62% |
| Llama-3-70B-IT | 67% | 70% | 62% | 58% | 48% |
| Llama-3-8B-IT | 67% | 61% | 63% | 51% | 34% |
| Llama-3-70B (base) | 36% | 58% | 58% | 60% | 54% |

Key Observations:
- Instruction-tuned models typically perform best in English and worst in Basque, but the disparities are smaller than expected.
- The base Llama-3-70B model exhibits the lowest truthfulness in English (36%) but higher in other languages—this is because it is more "informative" in English but prone to mimicking false beliefs.
- In terms of informativeness, base models frequently generate uninformative responses in non-English languages.

**Correlation between Judge-LLM / MC2 and Human Judgments** (Cohen's Kappa):

| Method | English | Spanish | Catalan | Galician | Basque |
|------|-----|-----|-----|-----|-------|
| MC2 | ~0.3 | ~0.2 | ~0.2 | ~0.2 | ~0.1 |
| Judge-LLM (Best) | 0.74 | 0.70 | 0.75 | 0.72 | 0.60 |
| Inter-annotator Agreement | ~0.75 | ~0.72 | ~0.70 | ~0.70 | ~0.65 |

**Key Result**: The agreement between Judge-LLM and human judgment is significantly higher than that of MC2, and is close to the inter-annotator agreement among humans.

**Full Judge-LLM Evaluation** (12 models $\times$ 5 languages):
- Gemma-2-27b-it performs best across all languages (averaging ~61%).
- Instruction models average ~57%, while base models average ~46%.
- Disparities across languages: English (~50-58%) > Spanish $\approx$ Galician > Catalan > Basque.

### Ablation Study

**Machine Translation vs. Professional Translation**:
- Automatically translating TruthfulQA using Claude 3.5 Sonnet.
- Taking professional translation as the reference, the MT quality is high (slightly lower for Basque due to its agglutinative nature).
- Judge-LLMs trained with the MT version perform comparable to those trained with the professional translation version.
- Conclusion: Machine translation serves as a viable alternative for extending truthfulness benchmarks to more languages.

**General vs. Context-Dependent Questions**:
- Dividing TruthfulQA questions into "general knowledge" (e.g., why chameleons change color) and "context/time-dependent" (e.g., US laws).
- General knowledge questions show more consistent performance across all languages.
- Context-dependent questions display larger cross-lingual disparities, making them more suitable for evaluating multilingual truthfulness.

### Key Findings

1. The cross-lingual truthfulness disparities in LLMs are much smaller than expected—even in Basque (the lowest-resource language), the performance degradation is limited.
2. MC2 as the sole metric for truthfulness evaluation is insufficient; LLM-as-a-Judge is a more reliable alternative.
3. Informativeness is a key factor in truthfulness evaluation—base models often generate uninformative responses, and ignoring informativeness distorts the assessment results.
4. Larger LLMs are generally more truthful than smaller models of the same family, contradicting early findings of Lin et al. (2022).
5. Qualitative analysis shows that responses in English still enjoy a significant lead in reasoning depth.

## Highlights & Insights

1. **First Professional Translation**: Avoids systematic biases that machine translation might introduce, providing a reliable baseline for subsequent research.
2. **In-depth Comparison of Evaluation Methods**: Systematically compares human evaluation, MC2, and Judge-LLM, quantifying correlation using Cohen's Kappa.
3. **Counter-intuitive Phenomenon in Base Models**: Base models exhibit lower truthfulness in English than in other languages, because they are more "confident" in mimicking false beliefs in English.
4. **Practical Finding**: Machine translation serves as a viable alternative to professional translation, significantly lowering the cost of building multilingual benchmarks.
5. **Dimension of Cultural and Temporal Dependence**: Emphasizes that truthfulness evaluation should distinguish between general knowledge and context-dependent knowledge.

## Limitations & Future Work

1. TruthfulQA itself is highly Anglo-American centric—even professional translation cannot alter the cultural background of the questions.
2. Only four target languages (Spanish, Catalan, Galician, Basque) are covered; there is a need to expand to more language families.
3. Informativeness evaluation is only effective for base models; instruction-tuned models lack uninformative answers, leaving training insufficient.
4. The sample size for human evaluation is limited (100 questions $\times$ 4 models $\times$ 5 languages = 2000 assessments).
5. "Culture-specific misconceptions" in different languages were not explored—different cultures may harbor different false beliefs.

## Related Work & Insights

- **TruthfulQA (Lin et al., 2022)**: The original benchmark, which used GPT-3 as a judge. This work replaces it with a stronger multilingual judge.
- **Aula-Blasco et al. (2025)**: A multilingual truthfulness benchmark that distinguishes between general vs. context-dependent questions—this work adopts this classification and provides empirical support.
- **HuggingFace OpenLLM Leaderboard**: Extensively uses the MC2 metric—this work demonstrates that MC2 has low correlation with human judgment.
- Insights: Cross-lingual evaluation is not only about translation quality but also involves fundamental reflections on cultural adaptation and evaluation methodology.

## Rating

- **Novelty**: ⭐⭐⭐ — The core contribution lies in high-quality translation and systematic evaluation, with limited methodology innovation.
- **Value**: ⭐⭐⭐⭐ — Provides an important benchmark and methodological guidance for multilingual truthfulness evaluation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 12 models $\times$ 5 languages $\times$ 3 evaluation methods, plus human evaluation and IAA analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, with detailed elaboration on translation guidelines and evaluation methodology.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Trans-EnV: A Framework for Evaluating the Linguistic Robustness of LLMs Against English Varieties](../../NeurIPS2025/llm_safety/trans-env_a_framework_for_evaluating_the_linguistic_robustness_of_llms_against_e.md)
- [\[ACL 2025\] ComparisonQA: Evaluating Factuality Robustness of LLMs Through Knowledge Frequency Control and Uncertainty](comparisonqa_evaluating_factuality_robustness_of_llms_through_knowledge_frequenc.md)
- [\[ICLR 2026\] No Caption, No Problem: Caption-Free Membership Inference via Model-Fitted Embeddings](../../ICLR2026/llm_safety/no_caption_no_problem_caption-free_membership_inference_via_model-fitted_embeddi.md)
- [\[ICLR 2026\] All Code, No Thought: Language Models Struggle to Reason in Ciphered Language](../../ICLR2026/llm_safety/all_code_no_thought_language_models_struggle_to_reason_in_ciphered_language.md)
- [\[ICLR 2026\] LLMs on Trial: Evaluating Judicial Fairness for Large Language Models](../../ICLR2026/llm_safety/llms_on_trial_evaluating_judicial_fairness_for_large_language_models.md)

</div>

<!-- RELATED:END -->

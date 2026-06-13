---
title: >-
  [Paper Note] Zero-shot Large Language Models for Automatic Readability Assessment
description: >-
  [ACL2026][LLM Evaluation][Readability Assessment] This paper systematically evaluates the zero-shot ARA capabilities of 10 open-source LLMs across 14 multilingual readability datasets. It proposes LAURAE: an ensemble app…
tags:
  - "ACL2026"
  - "LLM Evaluation"
  - "Readability Assessment"
  - "Zero-shot LLM"
  - "LAURAE"
  - "Confidence Ensemble"
  - "Medical Text"
date: 2026-05-08
content_hash: 80673a9b0fb5f383
---

# Zero-shot Large Language Models for Automatic Readability Assessment

**Conference**: ACL2026  
**arXiv**: [2604.24470](https://arxiv.org/abs/2604.24470)  
**Code**: https://github.com/rag24/LAURAE  
**Area**: Automatic Readability Assessment / Medical Text Readability / NLP Evaluation  
**Keywords**: Readability Assessment, Zero-shot LLM, LAURAE, Confidence Ensemble, Medical Text

## TL;DR
This paper systematically evaluates the zero-shot ARA capabilities of 10 open-source LLMs across 14 multilingual readability datasets. It proposes LAURAE: an ensemble approach that integrates LLM expected value readability scores with traditional formulas, weighted by verbal confidence, outperforming existing unsupervised methods on 13/14 datasets.

## Background & Motivation
**Background**: Automatic Readability Assessment (ARA) is a long-standing service for education, medicine, government, and text simplification. Particularly in medical scenarios, whether patient materials can be understood by target populations directly impacts patient decision-making and health outcomes. Traditional formulas like FKGL and ARI rely on shallow features such as sentence length, syllable count, and polysyllabic words. Due to their convenience, these formula-based methods are still widely adopted in applications as of 2025.

**Limitations of Prior Work**: While easy to use, traditional formulas ignore semantics, context, and technical terminology. Supervised BERT/ML models offer higher accuracy but require labeled data, training resources, and specialized expertise. Recent studies on GPT-4 zero-shot readability assessment show potential, but existing research mostly covers single English datasets or closed-source models, failing to address three practical questions: are open-source LLMs reliable, are non-English and technical texts reliable, and are they consistent across both short and long texts.

**Key Challenge**: Researchers and practitioners need unsupervised ARA that is more accurate than formulas without incurring the costs of supervised training. LLMs possess semantic understanding but may be unstable on short texts, children's texts, or low-resource languages. Complete reliance on LLMs sacrifices the robustness and cost advantages of formulaic methods.

**Goal**: The authors aim to provide a reproducible unsupervised ARA solution: first by improving zero-shot LLM scoring methods, then by conducting a comprehensive evaluation using 10 open-source LLMs and 14 datasets, and finally by proposing LAURAE, an ensemble method that combines LLM semantic capabilities with formula-based shallow features.

**Key Insight**: LLMs and formulas capture different signals. LLMs better understand context and difficulty definitions, whereas formulas more stably capture surface burdens like sentence and word length. By using LLM self-reported confidence to determine the weights between the two, a more robust unsupervised readability score can be obtained.

**Core Idea**: Require LLMs to score on the same readability scales used in human annotations and calculate expected scores using output token probabilities. Then, fuse LLM scores with traditional formulas using verbal confidence weighting to form LAURAE.

## Method
The technical contributions are two-fold. The first is improving zero-shot LLM ARA: prompts not only request readability scores but also force the model to use the same scale as human annotations (e.g., CEFR A1-C2) with explicit level definitions. The second is the LAURAE ensemble: standardized LLM scores and formula scores are combined via a weighted average based on the LLM's natural language confidence.

### Overall Architecture
The input is a text segment to be evaluated. If the dataset's human labels are CEFR-based, the prompt directs the LLM to provide a 1-6 integer score corresponding to A1-C2, including level definitions; otherwise, an arbitrary 1-9 scale is used with instructions to consider grammar and clarity. Instead of taking the greedy output, the system inspects the probability of all numeric tokens at the output position to calculate the expected value for both the readability and confidence scores.

Subsequently, LAURAE selects a traditional unsupervised ARA score as a shallow feature (e.g., FKGL/ARI for English, OSMAN for Arabic, Lix for Hindi/Greek). Both LLM and formula scores are normalized by the dataset mean and standard deviation, then weighted by LLM confidence $c$: the higher the confidence, the higher the LLM score weight.

### Key Designs
1.  **Readability scale prompt aligned with human annotation**:
    *   Function: Aligns the LLM output space with the dataset ground truth to reduce bias from "model-defined" readability.
    *   Mechanism: For the 7 CEFR datasets, A1-C2 definitions are placed in the prompt and mapped to levels 1-6. Non-CEFR datasets use 1-9 scores based on grammar, clarity, and complexity.
    *   Design Motivation: Correlation is needlessly lowered if human annotators use CEFR while LLMs use internal standards. Explicit definitions significantly aid non-English datasets.

2.  **Expected value scoring based on output token probabilities**:
    *   Function: Avoids discretization and tie issues caused by greedy decoding.
    *   Mechanism: Collect probability mass for all numeric tokens at the score position and treat the score as a probability-weighted expected value. If the highest probability is 4 but 3 and 5 are also likely, the expected score retains this uncertainty.
    *   Design Motivation: Readability is inherently a continuous difficulty scale. This method reduces ties in pairwise comparison, improving performance across all 14 datasets.

3.  **LAURAE Confidence-weighted Ensemble**:
    *   Function: Combines LLM semantic judgment with the shallow robustness of traditional formulas.
    *   Mechanism: The LLM outputs a 1-9 confidence score, also calculated via expected value, and divided by 10 to get weight $c$. The final score is: $c \cdot \text{standardized}(\text{LLM score}) + (1-c) \cdot \text{standardized}(\text{formula score})$.
    *   Design Motivation: LLMs excel in long, technical, or well-defined texts but may falter on children's stories or low-resource languages. Self-reported confidence serves as an unsupervised adaptive strategy.

### Loss & Training
This work utilizes unsupervised inference without training or fine-tuning. Experiments use 10 open-source instruction-tuned LLMs, including Llama 3.1 (8B/70B), Llama 3.2 (3B), Aya Expanse, Gemma 2, Mixtral 8x7B, and Phi-4. Llama 70B is used for English and Aya 32B for non-English to simulate realistic unsupervised settings where a validation set is unavailable. Pearson correlation is reported for continuous ground truth, and accuracy is used for pairwise datasets.

## Key Experimental Results

### Main Results
The 14 datasets cover 6 languages, varied lengths, CEFR/non-CEFR labels, and medical/simplification tasks. MedReadMe is a focal point for medical text evaluation.

| Dataset | Language | N | Avg Length | Ground Truth |
| :--- | :--- | ---: | ---: | :--- |
| ReadMe | EN/FR/HI/AR/RU | 163-296 | 22-25 | CEFR rating |
| MedReadMe | English | 1,140 | 25 | CEFR rating (Medical) |
| Cambridge | English | 300 | 579 | CEFR rating |
| CLEAR | English | 1,890 | 201 | non-CEFR rating |
| Greek Lang. / Hist. | Greek | 393 / 804 | 161 / 209 | non-CEFR rating |
| OneStop | English | 567 | 782 | non-CEFR rating |
| Asset | English | 485 | 21 | pairwise comparison |
| Vikidia | EN/FR | 150 / 150 | 596 / 509 | pairwise comparison |

LAURAE results are definitive: an average performance of 0.740, higher than LLM-v-ns, formulas, and RSRS baselines, achieving the best performance on 13/14 datasets.

| Dataset | LAURAE | LLM-v-ns | Formula | RSRS |
| :--- | ---: | ---: | ---: | ---: |
| Greek Lang. | 0.430 | 0.427 | 0.162 | 0.116 |
| Greek Hist. | 0.572 | 0.520 | 0.373 | 0.163 |
| Vikidia (fr) | 0.953 | 0.760 | 0.887 | 0.840 |
| Asset | 0.629 | 0.324 | 0.557 | 0.561 |
| CLEAR | 0.735 | 0.725 | 0.517 | 0.484 |
| OneStop | 0.654 | 0.488 | 0.577 | 0.627 |
| MedReadMe | 0.770 | 0.736 | 0.469 | 0.646 |
| Cambridge | 0.860 | 0.888 | 0.702 | 0.713 |
| Average | 0.740 | 0.595 | 0.599 | 0.592 |

### Ablation Study
Separating the prompt and scoring improvements: Expected value scoring improved results on 14/14 datasets. Including CEFR scale definitions significantly improved 5 out of 7 CEFR datasets, with particularly large gains in non-English evaluation.

| Ensemble Variant | Avg Change vs Standalone LLM |
| :--- | ---: |
| LAURAE (Verbal Confidence) | +0.027 |
| LAURAE-naive (Equal Weight) | +0.015 |
| LAURAE-entropy (Entropy Weight) | +0.006 |
| LAURAE-minmax | -0.013 |

### Key Findings
- Evaluating single English datasets overestimates the generalizability of zero-shot LLM ARA.
- Llama 70B is strongest for English, but Aya 32B is more stable for non-English, highlighting the importance of multilingual training over raw parameter scale.
- Expected value scoring is a low-cost, high-gain modification, especially for pairwise tasks like Asset/Vikidia.
- LAURAE improved MedReadMe from 0.469 (formula) to 0.770, showing the value of merging semantics with shallow features for technical medical text.

## Highlights & Insights
- The evaluation is exceptionally comprehensive, covering varied languages, lengths, and tasks.
- Aligning prompts with the task ontology (CEFR definitions) is significantly more reliable than generic "rate readability" requests.
- Verbal confidence as an unsupervised ensemble weight is clever, avoiding the need for dev label tuning and fitting the nature of LLMs as natural language evaluators.
- Highly practical for medical NLP, offering an unsupervised alternative to traditional formulas used by healthcare institutions.

## Limitations & Future Work
- Higher requirements than formulas: necessitates Python proficiency and GPU resources (e.g., 3x A100 for Llama 70B).
- Less interpretable than formulas; while LLMs can generate explanations, their faithfulness was not validated.
- Focuses on correlation with ground truth rather than absolute grade-level accuracy.
- Ethics: The authors advise against using this for individual writing skill assessment to avoid stylistic bias.

## Related Work & Insights
- **vs Traditional Formulas**: Formulas are easy but limited; LAURAE increases average performance from 0.599 to 0.740.
- **vs Supervised ARA**: Supervised models require labels; LAURAE remains zero-shot and versatile.
- **vs RSRS**: RSRS uses PLM surprisal; LAURAE is more robust for multilingual and technical texts.
- **Insight**: The LAURAE paradigm (Semantic LLM + Rule-based Feature + Confidence Weighting) can be transferred to other unsupervised evaluation tasks like toxicity or sentiment analysis.

## Rating
- Novelty: ⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] NovBench: Evaluating Large Language Models on Academic Paper Novelty Assessment](novbench_evaluating_large_language_models_on_academic_paper_novelty_assessment.md)
- [\[NeurIPS 2025\] Benchmarking Large Language Models for Zero-Shot and Few-Shot Phishing URL Detection](../../NeurIPS2025/llm_evaluation/benchmarking_large_language_models_for_zero-shot_and_few-shot_phishing_url_detec.md)
- [\[ACL 2026\] The Silent Vote: Improving Zero-Shot LLM Reliability by Aggregating Semantic Neighborhoods](the_silent_vote_improving_zero-shot_llm_reliability_by_aggregating_semantic_neig.md)
- [\[ACL 2026\] Question Difficulty Estimation for Large Language Models via Answer Plausibility Scoring](question_difficulty_estimation_for_large_language_models_via_answer_plausibility.md)
- [\[ACL 2026\] SciCustom: A Framework for Custom Evaluation of Scientific Capabilities in Large Language Models](scicustom_a_framework_for_custom_evaluation_of_scientific_capabilities_in_large_.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ACL 2026\] NovBench: Evaluating Large Language Models on Academic Paper Novelty Assessment](novbench_evaluating_large_language_models_on_academic_paper_novelty_assessment.md)
- [\[NeurIPS 2025\] Benchmarking Large Language Models for Zero-Shot and Few-Shot Phishing URL Detection](../../NeurIPS2025/llm_evaluation/benchmarking_large_language_models_for_zero-shot_and_few-shot_phishing_url_detec.md)
- [\[ACL 2026\] Question Difficulty Estimation for Large Language Models via Answer Plausibility Scoring](question_difficulty_estimation_for_large_language_models_via_answer_plausibility.md)
- [\[ACL 2026\] SciCustom: A Framework for Custom Evaluation of Scientific Capabilities in Large Language Models](scicustom_a_framework_for_custom_evaluation_of_scientific_capabilities_in_large_.md)
- [\[ACL 2026\] The Silent Vote: Improving Zero-Shot LLM Reliability by Aggregating Semantic Neighborhoods](the_silent_vote_improving_zero-shot_llm_reliability_by_aggregating_semantic_neig.md)

</div>

<!-- RELATED:END -->

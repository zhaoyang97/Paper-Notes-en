---
title: >-
  [Paper Note] Trans-EnV: A Framework for Evaluating the Linguistic Robustness of LLMs Against English Varieties
description: >-
  [NeurIPS 2025][LLM Safety][Linguistic Robustness] This paper proposes the Trans-EnV framework, which combines expert linguistic knowledge with the transformation capabilities of LLMs to automatically convert Standard Ame…
tags:
  - "NeurIPS 2025"
  - "LLM Safety"
  - "Linguistic Robustness"
  - "English Dialects"
  - "Fairness"
  - "LLM Evaluation"
  - "ESL English"
date: 2026-05-08
content_hash: 069a7a18ac570ab6
---

# Trans-EnV: A Framework for Evaluating the Linguistic Robustness of LLMs Against English Varieties

**Conference**: NeurIPS 2025
**arXiv**: [2505.20875](https://arxiv.org/abs/2505.20875)  
**Code**: [GitHub](https://github.com/jiyounglee-0523/TransEnV)  
**Area**: AI Safety
**Keywords**: Linguistic Robustness, English Dialects, Fairness, LLM Evaluation, ESL English

## TL;DR

This paper proposes the Trans-EnV framework, which combines expert linguistic knowledge with the transformation capabilities of LLMs to automatically convert Standard American English (SAE) datasets into 38 English varieties (18 dialects + 20 ESL Englishes), revealing performance degradations of up to 46.3% on non-standard English and highlighting critical linguistic fairness concerns.

## Background & Motivation

Virtually all LLM evaluation benchmarks rely on Standard American English (SAE), yet English exhibits rich global variation—including regional dialects (e.g., Scottish English, Irish English) and ESL Englishes produced by non-native speakers (e.g., English by Arabic L1 speakers). Prior work has demonstrated that LLMs perform significantly worse on non-standard English, and more critically, that model responses to non-mainstream varieties tend to be more stereotypical and condescending in tone.

However, systematic evaluation of the linguistic robustness of LLMs faces three obstacles:

**Existing multi-variety datasets are small in scale and narrow in coverage**: e.g., Multi-VALUE covers only a handful of dialects and limited tasks.

**Manual annotation methods are not scalable**: creating datasets for dozens of varieties by hand is prohibitively resource-intensive.

**Pure LLM-based generation is unreliable**: LLMs exhibit known biases when reproducing underrepresented English varieties and cannot be relied upon alone.

**Rule-based methods lack flexibility**: they cannot capture lexical choice and pragmatic variation.

The authors' solution is to combine expert linguistic resources (ensuring correctness) with LLMs (ensuring scalability) to construct a framework that automatically transforms SAE sentences into target varieties.

## Method

### Overall Architecture

Trans-EnV operates in three stages: (1) **Data Collection**—acquiring variety-specific linguistic features from linguistic literature and corpora; (2) **Transformation Guideline Generation**—creating transformation operation instructions for each feature, comprising a "qualification check" and "application steps"; (3) **Sentence Transformation**—using an LLM to incrementally transform SAE sentences into the target variety following the guidelines, with a semantic consistency checker to ensure quality.

### Key Designs

1. **Dialect Feature Extraction via eWAVE**: The eWAVE database (compiled by 84 linguists from 175 papers, covering 235 linguistic features across 77 English varieties) is used; KNN clustering over the 77 varieties is applied to distinguish English dialects from non-English varieties such as creoles, yielding 18 dialects. For each dialect, features annotated with the highest level of presence in eWAVE are selected as its feature set $\mathcal{L}_{v_i}$.

2. **Two-Dimensional Features for ESL English**:

    - **CEFR Dimension** (language proficiency): Based on 1,222 can-do descriptors from the English Grammar Profile, features above the target CEFR level are removed to simulate the corresponding proficiency. For example, simulating CEFR A removes all B- and C-level can-do features.

    - **L1 Dimension** (transfer effects): Three ESL corpora (CLC-FCE, ICLE, EFCamDat) are used; statistical t-tests ($p < 0.05$) are applied to extract L1-specific grammatical error patterns from learner writing at matching CEFR levels. Ten L1 backgrounds are covered (Arabic, Chinese, French, German, etc.), with an average of 10 features extracted per L1 per CEFR level. L1 features are verified to be non-conflicting with CEFR features before being merged.

3. **Transformation Guidelines and Execution Mechanism**: Each linguistic feature $l_j$ is paired with a transformation guideline $g_j$ consisting of two steps: (a) **Qualification**—determining whether the feature applies to the current sentence (e.g., "she/her for inanimate reference" requires an inanimate referent and corresponding pronoun in the sentence); (b) **Application**—the specific transformation operation (e.g., locating the inanimate referent and replacing its pronoun with she/her). GPT-4 is used for one-shot guideline generation. During sentence transformation, features are applied in randomized order one by one, with a semantic checker $S$ (LLaMA-3.3-70B-Instruct) verifying semantic consistency after each transformation. For ESL English, a lexical simplification step is additionally applied, permitting retention of at most 15% of advanced vocabulary to reflect the lexical distribution of real ESL text.

### Loss & Training

- The feature transformer $T$ uses Gemma-2-27B-Instruct; the semantic checker $S$ uses LLaMA-3.3-70B-Instruct.
- Human evaluation of $S$ yields 83.6% precision, 97.0% recall, and 89.8% F1.
- Human evaluation shows that all models used as $T$, except LLaMA-3.1-8B, achieve a valid transformation rate exceeding 90%.
- Transformation coverage: 82–95% of samples are modified by at least one feature overall, with an average of 2 features modified per sample.

## Key Experimental Results

### Main Results (Effect of Dialects on LLM Performance)

| Model | SAE | Dialect Avg. | Worst Dialect | Max Drop |
|-------|-----|-------------|--------------|----------|
| Qwen2.5-72B | 82.2 | 80.5 | NFE: 76.1 | −7.4% |
| DeepSeek-R1-70B | 80.8 | 79.6 | NFE: 75.2 | −6.9% |
| LLaMA-3.3-70B | 76.4 | 74.9 | IrE: 71.3 | −6.7% |
| o4-mini | 88.5 | 87.1 | NFE: 83.4 | −5.8% |
| GPT-4o-mini | 74.6 | 73.1 | WeE: 69.8 | −6.4% |

### Performance Drops on ESL English

| Configuration | MMLU Drop | ARC Drop | GSM8K Drop | HellaSwag Drop |
|--------------|-----------|---------|-----------|---------------|
| CEFR B (Intermediate) | ~3–8% | ~2–6% | ~5–15% | ~3–10% |
| CEFR A (Basic) | ~5–15% | ~4–12% | ~10–46.3% | ~5–15% |
| Reasoning tasks avg. | 22.6% | — | — | — |
| Knowledge tasks avg. | 10.9% | — | — | — |

### Key Findings

- **ESL English is far more challenging than dialects**: maximum drop for ESL is 46.3% vs. 12.5% for dialects, reflecting greater grammatical deviation in ESL English.
- **Reasoning tasks are more sensitive than knowledge tasks**: on dialects, reasoning tasks drop by 4.7% on average vs. 1.3% for knowledge tasks; the gap widens substantially on ESL (22.6% vs. 10.9%).
- **Strong reasoning models are more robust**: models focused on reasoning, such as DeepSeek-R1 and o4-mini, exhibit greater stability on dialects.
- **Linguistic distance positively correlates with performance drop**: using Euclidean distance over eWAVE feature vectors and DLIFLC language difficulty ratings, the analysis confirms that greater linguistic distance corresponds to larger performance degradation.
- **Training data volume determines robustness**: the worst-performing dialects (Newfoundland English, Welsh English) have fewer speakers, while the best-performing (Australian English) has more; among ESL varieties, French and Italian (with more training data) outperform Arabic and Turkish (with less).

## Highlights & Insights

- **The golden combination of expert knowledge and LLMs**: eWAVE and the English Grammar Profile provide linguistic guarantees, while LLMs enable scalable execution; the two are complementary and overcome each other's weaknesses.
- **Revealing the fairness gap**: performance disparities across English varieties are not merely a technical issue but a matter of social equity in whether global users can access AI services fairly.
- **Methodological inspiration**: the Qualification + Application two-step transformation paradigm is generalizable to other linguistic transformation tasks, such as text simplification and style transfer.

## Limitations & Future Work

- Coverage is limited to English varieties; extending to other languages (e.g., Spanish or Arabic varieties) requires corresponding linguistic resources.
- Evaluation is confined to QA tasks; assessment on open-ended tasks such as text generation and dialogue is only preliminarily explored and remains insufficiently thorough.
- Samples that are structurally simple or highly specialized (e.g., purely mathematical notation) cannot be transformed; approximately 5–30% of samples remain unmodified.
- The random ordering of transformation features may introduce unnecessary variability.
- Whether multi-variety data augmentation can improve LLM robustness to non-standard English has not been explored.

## Related Work & Insights

- Prior multi-variety datasets such as Multi-VALUE and DIVA motivate this work but are constrained in scale and coverage.
- The eWAVE database is a valuable linguistic resource that remains significantly underutilized in NLP.
- This work is complementary to dialect bias research (e.g., dialect bias detection in Value)—Trans-EnV focuses on task performance degradation, while the latter addresses biases in generated content.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The framework design that systematically combines expert linguistic resources with LLM-based transformation is novel, and the coverage of 38 varieties is unprecedented.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Six datasets, 38 varieties, and 7 models, supplemented by human evaluation, linguistic distance analysis, and open-ended task experiments.
- **Writing Quality**: ⭐⭐⭐⭐ Method descriptions are clear and linguistic background is well introduced, though some content is dispersed across the appendix.
- **Value**: ⭐⭐⭐⭐⭐ The work provides a standardized tool and large-scale empirical evidence for research on linguistic fairness in LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Evaluating the Promise and Pitfalls of LLMs in Hiring Decisions](evaluating_the_promise_and_pitfalls_of_llms_in_hiring_decisions.md)
- [\[NeurIPS 2025\] On the Robustness of Verbal Confidence of LLMs in Adversarial Attacks](on_the_robustness_of_verbal_confidence_of_llms_in_adversarial_attacks.md)
- [\[ACL 2026\] Evaluating Answer Leakage Robustness of LLM Tutors against Adversarial Student Attacks](../../ACL2026/llm_safety/evaluating_answer_leakage_robustness_of_llm_tutors_against_adversarial_student_a.md)
- [\[NeurIPS 2025\] CoreGuard: Safeguarding Foundational Capabilities of LLMs Against Model Stealing in Edge Deployment](coreguard_safeguarding_foundational_capabilities_of_llms_against_model_stealing_.md)
- [\[NeurIPS 2025\] Bits Leaked per Query: Information-Theoretic Bounds on Adversarial Attacks Against LLMs](bits_leaked_per_query_information-theoretic_bounds_on_adversarial_attacks_agains.md)

</div>

<!-- RELATED:END -->

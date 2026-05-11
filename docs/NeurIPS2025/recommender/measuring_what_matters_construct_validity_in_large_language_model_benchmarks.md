---
title: >-
  [Paper Note] Measuring What Matters: Construct Validity in Large Language Model Benchmarks
description: >-
  [NeurIPS 2025][Recommender Systems][LLM evaluation] This paper presents a systematic review of 445 LLM benchmark papers conducted by 29 experts…
tags:
  - "NeurIPS 2025"
  - "Recommender Systems"
  - "LLM evaluation"
  - "benchmark"
  - "construct validity"
  - "systematic review"
  - "evaluation methodology"
date: 2026-05-08
content_hash: f7ff57a48d875342
---

# Measuring What Matters: Construct Validity in Large Language Model Benchmarks

**Conference**: NeurIPS 2025
**arXiv**: [2511.04703](https://arxiv.org/abs/2511.04703)
**Code**: None
**Area**: Recommender Systems
**Keywords**: LLM evaluation, benchmark, construct validity, systematic review, evaluation methodology

## TL;DR
This paper presents a systematic review of 445 LLM benchmark papers conducted by 29 experts, examining existing LLM evaluation benchmarks through the lens of construct validity across four dimensions — phenomenon definition, task design, scoring metrics, and conclusion claims — and proposes 8 actionable recommendations for improvement.

## Background & Motivation
**Background**: LLM evaluation is one of the most active research directions in AI, with a large number of benchmark papers published annually. The reliability of evaluation results directly determines the accuracy of assessments of model capabilities and the effectiveness of pre-deployment safety evaluations. In recent years, the number of benchmarks has grown exponentially, but quality varies considerably.

**Limitations of Prior Work**: Many benchmarks attempt to measure abstract concepts such as "safety" and "robustness," yet their task designs and scoring mechanisms often fail to genuinely reflect the target phenomena. 47.8% of benchmarks exhibit disputed or insufficiently consensual definitions of what they measure; 27% rely on convenience sampling to obtain test instances.

**Key Challenge**: There exists a severe disconnect between the explosive growth in the number of LLM benchmarks and quality control — benchmarks are increasingly abundant, yet individual papers rarely provide adequate justification for why a given benchmark validly measures the intended capability. Only 53.4% of papers discuss construct validity.

**Goal**: (1) Systematically identify common construct validity issues in LLM benchmark papers published at top NLP/ML venues; (2) quantify the prevalence of each issue; (3) propose actionable recommendations and a checklist for improvement.

**Key Insight**: The authors draw on the construct validity framework from psychometrics, treating a benchmark as a chain of "phenomenon → task → metric → claim," and systematically examine validity issues that may arise at each stage.

**Core Idea**: Apply the well-established construct validity framework from psychometrics to systematically audit the quality of LLM benchmarks, identify widespread methodological deficiencies, and distill 8 actionable recommendations for improvement.

## Method

### Overall Architecture
This paper adopts a systematic review methodology. From a corpus of 46,114 papers published at ICML/ICLR/NeurIPS (2018–2024) and ACL/NAACL/EMNLP (2020–2024), keyword-based filtering ("benchmark" combined with "LLM"/"language model") yielded 2,189 candidate papers. These were subsequently processed through automated pre-screening by GPT-4o mini (F1 = 84%) and manual review by 29 experts, resulting in 445 papers selected for in-depth coding analysis.

### Key Designs
1. **Codebook**:

    - Function: Standardizes the coding of phenomenon definition, task design, metric selection, and conclusion claims for each paper.
    - Mechanism: Designed 21 question items based on multiple dimensions of construct validity (face validity, predictive validity, content validity, ecological validity, convergent validity, and discriminant validity).
    - Design Motivation: Transforms the subjective question of "is this benchmark good?" into a quantifiable multi-dimensional assessment that supports statistical analysis.

2. **Two-Round Review Process**:

    - Function: Each paper is first evaluated item-by-item by a primary reviewer using the codebook, then a secondary reviewer maps the responses to simplified options and confirms with the primary reviewer.
    - Mechanism: 46 papers were randomly sampled for double review; inter-rater reliability was assessed using the Brennan-Prediger Kappa coefficient (mean = 0.524).
    - Design Motivation: Balances review scale (445 papers) with quality control to ensure the reliability of coding outcomes.

3. **Inductive Recommendation Generation**:

    - Function: The first author read a subset of 50 papers and reviewed all 445 annotations, deriving preliminary recommendations through inductive open coding.
    - Mechanism: Through 5 rounds of iterative multi-author discussion, findings were consolidated into 8 primary recommendations.
    - Design Motivation: Ensures that recommendations are both empirically grounded and operationally practical.

### Overview of 8 Recommendations
1. **Define the phenomenon**: Clearly define the target phenomenon; sub-components should be evaluated separately.
2. **Measure the phenomenon and only the phenomenon**: Control for confounding factors such as output format and instruction complexity.
3. **Construct representative datasets**: Use random or stratified sampling instead of convenience sampling.
4. **Exercise caution when reusing datasets**: Document differences between old and new versions and assess changes in construct validity.
5. **Guard against data contamination**: Check for contamination at dataset creation time and consider dynamic benchmarks.
6. **Use statistical methods to compare models**: Currently only 16% of papers employ statistical testing.
7. **Conduct error analysis**: Verify whether failure patterns correspond to the target phenomenon.
8. **Justify construct validity**: Explicitly articulate the reasoning chain from phenomenon to task to metric to claim.

## Key Experimental Results

### Core Statistical Findings

| Dimension | Finding | Proportion |
|-----------|---------|------------|
| Phenomenon definition | Definition provided | 78.2% |
| Phenomenon definition | Definition is contested | 47.8% |
| Phenomenon type | Composite phenomena (with sub-capabilities) | 61.2% |
| Task source | Manually constructed tasks | 43.3% |
| Task source | Reused existing benchmarks | 42.6% |
| Task source | LLM-generated | 31.2% |
| Sampling method | Convenience sampling (at least in part) | 27.0% |
| Scoring metric | Exact match (at least in part) | 81.3% |
| Statistical methods | Used statistical testing | 16.0% |
| Validity justification | Discussed construct validity | 53.4% |

### Distribution of Benchmark Phenomena

| Phenomenon Category | Proportion | Notes |
|--------------------|------------|-------|
| Reasoning | 18.5% | Most common category |
| Alignment | 8.1% | Among the most contested in definition |
| Code Generation | 5.7% | Relatively well-defined |
| Other general capabilities | ~30% | Includes knowledge, comprehension, etc. |
| Domain-specific applications | ~38% | Medical, legal, etc. |

### Key Findings
- Fewer than 10% of benchmarks use fully real-world tasks; 40.7% rely on manually constructed tasks.
- The most common scoring metric is exact match (81.3%); LLM-as-a-judge is used in only 17.1% of papers.
- The number of benchmark papers has grown significantly year over year, but the proportion discussing construct validity has not increased correspondingly.
- Approximately half of all papers exhibit validity weaknesses in at least one dimension.

## Highlights & Insights
- **Systematic scale**: This large-scale systematic review — encompassing 445 papers and 29 experts — is unprecedented in the field of LLM evaluation methodology, providing first-hand empirical data for quantifying "benchmark quality." The codebook design skillfully adapts psychometric theory to AI evaluation contexts.
- **High practical value**: The 8 recommendations are accompanied by an operational checklist that can guide the design of new benchmarks and serve as a reviewing standard for assessing existing ones. The checklist is released as an appendix, encouraging researchers to provide justifications for any items they choose to omit.

## Limitations & Future Work
- Coverage is limited to top-venue papers, excluding important benchmarks released by industry (e.g., industrial iterations of MMLU, HumanEval, etc.).
- The use of GPT-4o mini for pre-screening may introduce systematic false-negative bias.
- Each paper was coded by only 1–2 reviewers; the double-review Kappa of 0.524 reflects only moderate inter-rater agreement.
- The paper does not deeply analyze whether construct validity issues differ systematically across domains (reasoning vs. safety vs. code generation).

## Related Work & Insights
- **vs. BetterBench (Reuel et al., 2024)**: BetterBench also addresses benchmark quality assessment, but the present work is larger in scale (445 vs. fewer papers) and more systematic in its theoretical framework (multi-dimensional construct validity).
- **vs. Bowman & Dahl (2021)**: That work proposed directional recommendations for fixing NLU benchmarks; the present paper builds upon it by quantifying the prevalence of these issues through large-scale empirical data.
- **vs. tinyBenchmarks (Polo et al., 2024)**: Focuses on efficient evaluation with fewer samples, complementing this paper's emphasis on sample representativeness.
- **vs. Dynabench (Kiela et al., 2021)**: Proposes dynamic benchmarks to address data contamination, consistent with Recommendation 5 (guarding against contamination) in the present work.
- **Implications**: The proposed checklist can be directly applied during peer review to assess the methodological quality of benchmark papers.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic application of the construct validity framework to large-scale auditing of LLM benchmarks.
- Experimental Thoroughness: ⭐⭐⭐⭐ Coding analysis of 445 papers with substantial data coverage, though inter-rater Kappa is moderate.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure, highly actionable recommendations, and rich figures and tables.
- Value: ⭐⭐⭐⭐⭐ Carries significant guidance value for the entire LLM evaluation community and has the potential to reshape benchmark design paradigms.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Inference-Time Reward Hacking in Large Language Models](inference-time_reward_hacking_in_large_language_models.md)
- [\[ICLR 2026\] From Evaluation to Defense: Advancing Safety in Video Large Language Models](../../ICLR2026/recommender/from_evaluation_to_defense_advancing_safety_in_video_large_language_models.md)
- [\[NeurIPS 2025\] Who You Are Matters: Bridging Topics and Social Roles via LLM-Enhanced Logical Recommendation](who_you_are_matters_bridging_topics_and_social_roles_via_llm-enhanced_logical_re.md)
- [\[ICLR 2026\] GoalRank: Group-Relative Optimization for a Large Ranking Model](../../ICLR2026/recommender/goalrank_group-relative_optimization_for_a_large_ranking_model.md)
- [\[NeurIPS 2025\] R²ec: Towards Large Recommender Models with Reasoning](r2ec_towards_large_recommender_models_with_reasoning.md)

</div>

<!-- RELATED:END -->

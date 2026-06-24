---
title: >-
  [Paper Note] PapersPlease: A Benchmark for Evaluating Motivational Values of Large Language Models Based on ERG Theory
description: >-
  [ACL 2025 (GEM2 Workshop)][LLM Evaluation][LLM Value Assessment] This paper proposes the PapersPlease benchmark, containing 3,700 moral dilemma scenarios based on Alderfer's ERG motivation theory. By instructing LLMs to role-play as immigration officers deciding on applicant entry, this benchmark reveals significant differences in motivational value prioritization across six LLMs, as well as biases toward marginalized social groups.
tags:
  - "ACL 2025 (GEM2 Workshop)"
  - "LLM Evaluation"
  - "LLM Value Assessment"
  - "ERG Theory"
  - "Role-playing"
  - "Social Bias"
  - "Moral Reasoning"
date: 2026-05-08
content_hash: 8fd433f767fda782
---

# PapersPlease: A Benchmark for Evaluating Motivational Values of Large Language Models Based on ERG Theory

**Conference**: ACL 2025 (GEM2 Workshop)  
**arXiv**: [2506.21961](https://arxiv.org/abs/2506.21961)  
**Code**: [https://github.com/yeonsuuuu28/papers-please](https://github.com/yeonsuuuu28/papers-please)  
**Area**: LLM Evaluation  
**Keywords**: LLM Value Assessment, ERG Theory, Role-playing, Social Bias, Moral Reasoning

## TL;DR
This paper proposes the PapersPlease benchmark, containing 3,700 moral dilemma scenarios based on Alderfer's ERG motivation theory. By instructing LLMs to role-play as immigration officers deciding on applicant entry, this benchmark reveals significant differences in motivational value prioritization across six LLMs, as well as biases toward marginalized social groups.

## Background & Motivation

**Background**: Evaluating LLM behaviors and biases through role-play scenarios has become a mainstream method. Existing works such as the MACHIAVELLI benchmark evaluate strategic behaviors via text-based games, and SANDBOX simulates human social behavior through multi-agent interactions.

**Limitations of Prior Work**: Existing evaluations of moral reasoning suffer from two main shortcomings: (1) lack of a systematic evaluation framework grounded in psychological theories, as most works merely test the consistency of simple moral judgments in LLMs; (2) social identities (race, gender, religion) are rarely integrated with motivational values during evaluation, thereby overlooking how identity biases affect LLM decision-making in value-sensitive scenarios.

**Key Challenge**: LLMs exhibit behavioral tendencies in role-play that differ drastically from standard Q&A patterns, yet we lack structured methods to evaluate whether these implicitly encoded value priorities align with human motivational values.

**Goal**: To design a role-play benchmark grounded in psychological theory (ERG theory) to systematically evaluate how LLMs make value judgments among competing human needs, and how social identities influence their decisions.

**Key Insight**: Inspired by the classic video game *Papers, Please*—where players act as immigration inspectors facing moral dilemmas—the authors integrate this high-pressure decision-making paradigm with Alderfer's ERG theory (which categorizes human needs into Existence, Relatedness, and Growth), creating an evaluation environment with forced-choice constraints.

**Core Idea**: Emulate an immigration officer in the fictional country of Arstotzka, where LLMs make entry/denial/detainment decisions regarding applicants' narrative scenarios constructed based on the three levels of ERG theory. Concurrently, social identity cues are embedded to detect bias.

## Method

### Overall Architecture
The input is an immigration scenario description (the applicant's background narrative), and the output is the LLM's decision (Approve/Reject/Detain). The overall evaluation spans three dimensions: (1) individual case evaluation, where decisions are made independently for each scenario; (2) comparative case evaluation, which forces a priority ranking in a 1-out-of-3 scenario; and (3) social dimension evaluation, observing decision shifts when identity cues are embedded in the narrative.

### Key Designs

1. **ERG Theory-Based Scenario Generation**:

    - Function: To generate a structured dataset of moral dilemma narratives.
    - Mechanism: ERG theory divides human motivation into three levels: Existence (e.g., food, safety), Relatedness (e.g., family reunion), and Growth (e.g., career development). Five representative examples were manually written for each category, which were then expanded to 100 scenarios per category using GPT-4o-mini via few-shot prompting, totaling 300 base scenarios. All scenarios underwent human verification to ensure quality.
    - Design Motivation: ERG theory provides a psychologically validated hierarchical framework of human needs, offering a more solid theoretical foundation and better interpretability than arbitrarily constructed moral dilemmas.

2. **Social Identity Embedding Mechanism**:

    - Function: To detect social bias in LLM decision-making.
    - Mechanism: A short social identity annotation (e.g., "Person's gender: male") is appended before each narrative, covering three dimensions—gender (male/female/non-binary, 3 options), race (White/Black/Hispanic/Asian, 4 options), and religion (Christianity/Islam/Hinduism/Buddhism, 4 options). Bias is quantified by comparing decision differences with and without identity cues. In total, this yields 3,700 scenarios.
    - Design Motivation: Evaluating motivational value priorities alone is insufficient; it is crucial to detect whether these priorities are influenced by social identities, which is essential for assessing the fairness of LLMs in real-world, sensitive applications.

3. **Three Evaluation Paradigms**:

    - Function: To comprehensively evaluate the LLM's value judgments from different angles.
    - Mechanism: Individual evaluation assesses the approval/rejection rates of each scenario independently to observe absolute preferences; comparative evaluation presents three scenarios representing different ERG categories simultaneously (allowing only one to be approved) to force priority ranking; social dimension evaluation repeats the individual evaluation with identity cues added, analyzing bias direction and magnitude through discrepancy analysis.
    - Design Motivation: Individual evaluations can be confounded by a model's overall leniency or strictness; comparative evaluation eliminates this confounder, while social dimension evaluation specifically targets fairness.

### Loss & Training
No training involved—this is a pure evaluation benchmark. Deterministic inference is performed with temperature=0, and statistical significance is assessed using Chi-Square tests and post-hoc pairwise comparisons.

## Key Experimental Results

### Main Results (Individual Case Evaluation - Approval Count / 100 Scenarios)

| Model | Existence | Relatedness | Growth |
|------|----------|----------|----------|
| GPT-4o-mini | 99 | 47 | 74 |
| Claude-3.7-sonnet | 0 | 0 | 0 |
| Gemini-2.0-flash | 41 | 11 | 43 |
| Llama-3.1-8B | 83 | 91 | 96 |
| Llama-4-Maverick | 83 | 11 | 47 |
| Qwen3-14B | 89 | 53 | 63 |

### Comparative Case Evaluation (Forced 1-out-of-3 Priority)

| Pattern | Representative Models |
|------|----------|
| Existence > Growth > Relatedness (Aligns with ERG hierarchy) | GPT-4o-mini, Claude-3.7, Qwen3 |
| Balanced Distribution (Deviates from ERG hierarchy) | Gemini-2.0, Llama-4, Llama-3.1 |

### Key Findings
- Claude-3.7-sonnet rejected 100% of applicants across all individual scenarios, strictly adhering to rules while ignoring humanitarian considerations—exhibiting extreme rule-first behavior.
- Two model clusters emerged: GPT/Claude/Qwen prioritize Existence needs (aligning with the ERG hierarchy); Gemini/Llama are more balanced (potentially deviating from human intuitive priorities).
- Social identity impacts statistically significantly: Llama-4 systematically decreased approval rates for Black, Asian, Muslim, and Hindu identities; GPT-4o-mini increased approval rates for most identities in the Relatedness and Growth categories, with the exception of the Muslim identity.
- Chi-Square tests indicate that differences both between individual models and between model groups are statistically significant ($p<0.05$).

## Highlights & Insights
- **Psychological Theory-Driven LLM Evaluation**: Introducing Alderfer's ERG theory into LLM evaluation is a clever framework choice, offering greater theoretical depth and reproducibility than ad-hoc moral dilemmas.
- **Gamified Evaluation Design**: The situational design inspired by *Papers, Please* creates natural forced-decision pressure, revealing implicit value preferences in LLMs more effectively than simple questions like "Do you think this is right?".
- **Fine-Grained Analysis of Identity Bias**: The cross-analysis of motivation categories $\times$ social identities reveals highly nuanced bias patterns, such as models exhibiting bias against specific identities only under certain categories of need.

## Limitations & Future Work
- Only 6 LLMs were evaluated, which limits the generalizability of the findings due to the sample size.
- The scenarios are designed based on fictional, extreme situations (dystopian border inspection), which might differ from value judgments in daily life.
- Decisions are binary (approve/reject); future work could utilize continuous ratings (0-10) to capture more subtle preference differences.
- There is a lack of direct comparison with human experiments; although human expected priorities were inferred using ERG theory, no real human data was collected as a baseline.

## Related Work & Insights
- **vs MACHIAVELLI**: MACHIAVELLI evaluates moral trade-offs in strategic decisions, whereas PapersPlease focuses on the hierarchical priorities of motivational values—aligning more closely with psychological evaluation.
- **vs MoCA**: MoCA evaluates moral norm consistency based on cognitive science literature but does not involve the impact of social identities. PapersPlease integrates value assessment with fairness detection, offering a more comprehensive framework.
- This evaluation framework can be extended to other psychological theories (such as Maslow's hierarchy of needs) or adapted to non-immigration scenarios, such as medical resource allocation or disaster relief prioritization.

## Rating
- Novelty: ⭐⭐⭐⭐ Merging ERG theory with role-playing to evaluate the motivational values of LLMs is a novel design idea.
- Experimental Thoroughness: ⭐⭐⭐ While it covers three evaluation dimensions across 6 models, it lacks comparison with human baselines.
- Writing Quality: ⭐⭐⭐⭐ The structure is clear, the experimental design is solid, and the statistical analysis is rigorous.
- Value: ⭐⭐⭐⭐ Provides a fresh perspective and an actionable tool for evaluating the implicit value systems of LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Batayan: A Filipino NLP Benchmark for Evaluating Large Language Models](batayan_a_filipino_nlp_benchmark_for_evaluating_large_language_models.md)
- [\[ICML 2025\] Position: Theory of Mind Benchmarks are Broken for Large Language Models](../../ICML2025/llm_evaluation/position_theory_of_mind_benchmarks_are_broken_for_large_language_models.md)
- [\[ACL 2025\] SeedBench: A Multi-task Benchmark for Evaluating Large Language Models in Seed Science](seedbench_a_multi-task_benchmark_for_evaluating_large_language_models_in_seed_sc.md)
- [\[ACL 2025\] WXImpactBench: A Disruptive Weather Impact Understanding Benchmark for Evaluating Large Language Models](wximpactbench_a_disruptive_weather_impact_understanding_benchmark_for_evaluating.md)
- [\[ICML 2026\] PoliticsBench: Benchmarking Political Values in Large Language Models with Multi-Stage Roleplay](../../ICML2026/llm_evaluation/politicsbench_benchmarking_political_values_in_large_language_models_with_multi-.md)

</div>

<!-- RELATED:END -->

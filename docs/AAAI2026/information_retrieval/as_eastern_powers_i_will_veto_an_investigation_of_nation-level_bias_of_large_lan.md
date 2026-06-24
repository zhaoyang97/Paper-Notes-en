---
title: >-
  [Paper Note] "As Eastern Powers, I Will Veto." : An Investigation of Nation-Level Bias of Large Language Models in International Relations
description: >-
  [AAAI 2026][Information Retrieval & RAG][nation-level bias] This paper systematically investigates nation-level bias of LLMs in international relations, designing three bias evaluation paradigms (DirectQA, Association Test, Vote Simulation) grounded in real UN Security Council data. It reveals the multi-dimensional nature of such bias—varying across models and evaluation contexts—and proposes a RAG+Reflexion debiasing framework.
tags:
  - "AAAI 2026"
  - "Information Retrieval & RAG"
  - "nation-level bias"
  - "international relations"
  - "UN Security Council"
  - "explicit bias"
  - "implicit bias"
date: 2026-05-08
content_hash: 85ce7e4559edbb7c
---

# "As Eastern Powers, I Will Veto." : An Investigation of Nation-Level Bias of Large Language Models in International Relations

**Conference**: AAAI 2026
**arXiv**: [2511.10695](https://arxiv.org/abs/2511.10695)  
**Code**: [GitHub](https://github.com/concistency/Nation-Level_Bias)  
**Area**: Information Retrieval
**Keywords**: nation-level bias, international relations, UN Security Council, explicit bias, implicit bias

## TL;DR

This paper systematically investigates nation-level bias of LLMs in international relations, designing three bias evaluation paradigms (DirectQA, Association Test, Vote Simulation) grounded in real UN Security Council data. It reveals the multi-dimensional nature of such bias—varying across models and evaluation contexts—and proposes a RAG+Reflexion debiasing framework.

## Background & Motivation

**Background**: LLMs are increasingly explored for simulation, decision support, and policy analysis in international relations (IR). However, existing bias research almost exclusively focuses on demographic biases (gender, race, etc.), with little attention to nation-level bias.

**Limitations of Prior Work**: Whether LLMs exhibit systematic biases toward different nations in IR scenarios (e.g., diplomatic simulation, vote prediction) remains unclear. If such biases exist, model outputs could mislead policy analysis with serious consequences.

**Key Challenge**: Nations in IR inherently differ in objective behavior (e.g., distinct voting records), so "bias" cannot be trivially defined as deviation from a uniform distribution. Instead, bias must be defined as the discrepancy between model behavior and a nation's actual historical behavior.

**Goal**: Systematically evaluate nation-level bias toward the P5 nations in the UN Security Council across multiple LLMs, reveal its multi-dimensional nature, and explore mitigation strategies.

**Key Insight**: Real UNSC data from 2013–2024—including resolutions, voting records, and meeting statements—are used to design both explicit (DirectQA, Association Test) and implicit (Vote Simulation) bias evaluations.

**Core Idea**: Nation-level bias in LLMs is multi-dimensional—its direction and magnitude depend on the model and evaluation context—and enhancing reasoning capability can effectively mitigate bias.

## Method

### Overall Architecture

Construct a real-world UNSC dataset → Design three bias evaluation experiments → Analyze bias patterns → Propose a RAG+Reflexion debiasing framework. Models evaluated include GPT-4o-mini, Llama-3.3-70B, Mistral-Small-22B, Qwen-2.5-72B, and reasoning models o3-mini and DeepSeek-R1.

### Key Designs

1. **DirectQA (Explicit Bias)**: LLMs are directly asked "which country is more irresponsible," covering both general questions and UNSC-specific functional questions. Each question presents two P5 nations for the model to choose between; bidirectional questioning eliminates positional bias. The frequency with which each country is selected as "irresponsible" is computed as its *irres_score*.

2. **Association Test (AT, Explicit Bias)**: For 41 UNSC-domain keywords across 7 semantic categories, LLMs rank the P5 nations by association strength and infer association polarity (positive/negative). An ATS score is computed: higher rank combined with positive polarity yields a higher score. Nation name order is randomized.

3. **Vote Simulation (Implicit Bias)**: LLMs are assigned a national representative persona and asked to vote (favour/against/abstention) on real failed resolutions. Only non-passing resolutions are used, as passing resolutions contain no P5 opposition votes. Bias and performance are assessed via statistical analysis and confusion matrices (weighted F1) against ground-truth voting records.

4. **RAG+Reflexion Debiasing Framework**: (1) *Retrieval*: retrieve historically similar resolutions based on keywords from the target resolution; (2) *Rehearsal + Reflection*: the LLM performs simulated votes on retrieved historical resolutions, compares results with ground truth, and reflects on discrepancies while incorporating representative statements to understand national stances; (3) *Final Vote*: historical experience is injected into the prompt for the target resolution vote. The framework is purely prompt-engineering-based, requiring no parameter tuning.

### Loss & Training

No model training is involved. All experiments use temperature=0 for consistency, are repeated 3 times, and validated with Fleiss' kappa and multiple $\chi^2$ tests. Over 90% of tests pass statistical consistency checks.

## Key Experimental Results

### Main Results

Vote Simulation Weighted F1 ($\times 100$):

| Model | USA | UK | France | Russia | China |
|---|---|---|---|---|---|
| GPT-4o-mini | 60 | 43 | 49 | 41 | 28 |
| Llama-3.3 | 54 | 41 | 49 | 72 | 50 |
| Mistral-S | 44 | 51 | 56 | 44 | 38 |
| Qwen-2.5 | 48 | 50 | 52 | 60 | 59 |
| o3-mini | 65 | 44 | 46 | 62 | 56 |
| **DeepSeek-R1** | **73** | **59** | **61** | **69** | **67** |

Reasoning models—particularly DeepSeek-R1—achieve the highest F1 for 4 out of 5 nations, indicating that stronger reasoning capability mitigates bias.

### Ablation Study

RAG+Reflexion Debiasing Effect:

| Model | USA | UK | France | Russia | China |
|---|---|---|---|---|---|
| GPT-4o-mini (base) | 60 | 43 | 49 | 41 | 28 |
| GPT-4o-mini +debias | 59(−1) | **60(+17)** | 52(+3) | **59(+18)** | **44(+16)** |
| Llama-3.3 (base) | 54 | 41 | 49 | 72 | 50 |
| Llama-3.3 +debias | 56(+2) | 47(+6) | 48(−1) | 54(−18) | 52(+2) |

GPT-4o-mini benefits most from debiasing, with especially large gains on Russia and China. However, Mistral and Qwen exhibit performance degradation after debiasing, likely due to insufficient long-context comprehension.

### Key Findings

- All models exhibit positive bias toward the UK and France, and negative bias toward Russia.
- In the Association Test, the USA achieves the highest ATS in 5 out of 7 categories, reflecting systematic positive bias.
- GPT casts far more opposition votes against Russia and China in Vote Simulation than the ground truth warrants (polarity bias).
- Llama most closely approximates Russia's actual voting behavior and achieves the highest single-nation F1 (72).
- Bias direction can be inconsistent within the same model across evaluation paradigms: all models exhibit negative bias toward the USA in DirectQA but positive bias in Vote Simulation.

## Highlights & Insights

- The finding of **multi-dimensionality** in nation-level bias is particularly valuable—no single evaluation paradigm suffices, and bias direction shifts with evaluation method.
- The positive correlation between reasoning capability and bias mitigation offers a new direction for debiasing research.
- Using real UNSC data (rather than fabricated scenarios) substantially strengthens the credibility of the conclusions.
- The debiasing framework is entirely prompt-engineering-based, requiring no parameter modification, making it practically accessible.

## Limitations & Future Work

- DirectQA and the Association Test lack an objective "unbiased baseline," allowing only pattern analysis rather than precise bias quantification.
- Only P5 nations are evaluated; bias toward non-permanent Security Council members remains unexplored.
- The debiasing framework underperforms on Mistral and Qwen, where long-context comprehension is a bottleneck.
- The dataset is limited to 2013–2024; historical periods such as the Cold War are not covered.

## Related Work & Insights

- **vs. demographic bias research**: This paper extends bias research from the individual level (gender/race) to the nation level, confirming that multi-dimensionality holds at the national scale as well.
- **vs. Jensen et al. (2025)**: Prior work assessed nation-level bias only through hypothetical scenarios; this paper is the first to use real UNSC data with a more diverse and reliable evaluation methodology.
- **vs. RAG-based debiasing**: The paper innovatively integrates Reflexion-style self-reflection, so debiasing relies not only on external knowledge retrieval but also on reasoning reinforced through rehearsing historical cases.

## Rating

- Novelty: ⭐⭐⭐⭐ Nation-level bias in IR is largely uncharted territory; the multi-dimensionality finding is a meaningful contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three evaluation paradigms, four base models, two reasoning models, and debiasing experiments are included, though broader model and geographic coverage would strengthen the work.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with rigorous experimental design and sufficient statistical validation.
- Value: ⭐⭐⭐⭐ Raises important warnings about deploying LLMs in high-stakes IR applications, though the debiasing solution remains preliminary.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Query-Level Uncertainty in Large Language Models](../../ICLR2026/information_retrieval/query-level_uncertainty_in_large_language_models.md)
- [\[ACL 2025\] Evaluation of Attribution Bias in Generator-Aware Retrieval-Augmented Large Language Models](../../ACL2025/information_retrieval/evaluation_of_attribution_bias_in_generator-aware_retrieval-augmented_large_lang.md)
- [\[AAAI 2026\] PRECISE: Reducing the Bias of LLM Evaluations Using Prediction-Powered Ranking Estimation](precise_reducing_the_bias_of_llm_evaluations_using_prediction-powered_ranking_es.md)
- [\[AAAI 2026\] Do Retrieval Augmented Language Models Know When They Don't Know?](do_retrieval_augmented_language_models_know_when_they_dont_know.md)
- [\[ICLR 2026\] MLP Memory: A Retriever-Pretrained Memory for Large Language Models](../../ICLR2026/information_retrieval/mlp_memory_a_retriever-pretrained_memory_for_large_language_models.md)

</div>

<!-- RELATED:END -->

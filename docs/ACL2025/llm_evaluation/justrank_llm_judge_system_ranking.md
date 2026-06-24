---
title: >-
  [Paper Note] JuStRank: Benchmarking LLM Judges for System Ranking
description: >-
  [ACL 2025][LLM Evaluation][LLM-as-Judge] The first large-scale study on the performance of LLM judges in system ranking tasks, introducing the JuStRank benchmark. Over 1.5 million ratings from 48 judges across 63 systems were collected, revealing a significant gap between instance-level judgment and system-level ranking performance, and identifying two quantifiable system-level behavioral characteristics of LLM judges: "decisiveness" and "system-specific bias."
tags:
  - "ACL 2025"
  - "LLM Evaluation"
  - "LLM-as-Judge"
  - "system ranking"
  - "judge benchmark"
  - "decisiveness"
  - "bias analysis"
date: 2026-05-08
content_hash: 37a2924e93b284ed
---

# JuStRank: Benchmarking LLM Judges for System Ranking

**Conference**: ACL 2025  
**arXiv**: [2412.09569](https://arxiv.org/abs/2412.09569)  
**Code**: [JuStRank Data](https://github.com/IBM/JuStRank)  
**Area**: LLM Evaluation  
**Keywords**: LLM-as-Judge, system ranking, judge benchmark, decisiveness, bias analysis

## TL;DR

The first large-scale study on the performance of LLM judges in system ranking tasks, introducing the JuStRank benchmark. Over 1.5 million ratings from 48 judges across 63 systems were collected, revealing a significant gap between instance-level judgment and system-level ranking performance, and identifying two quantifiable system-level behavioral characteristics of LLM judges: "decisiveness" and "system-specific bias."

## Background & Motivation

**Background**: The LLM-as-a-judge paradigm has become the dominant approach for evaluating generative AI. Users need to systematically compare and select different models and configurations. Since human annotation is prohibitively expensive, LLM judges are widely used for automated evaluation. Existing judge benchmarks such as RewardBench and JudgeBench focus primarily on instance-level evaluation—determining the quality of single or pairwise responses.

**Limitations of Prior Work**: Instance-level evaluation only measures how many errors a judge makes on individual responses, ignoring how these errors are distributed across different systems. A judge with high instance-level accuracy might exhibit systemic bias towards a specific system (e.g., consistently overestimating responses from System A), leading to severe distortion in system ranking. Conversely, a judge with moderate instance-level performance could produce accurate system rankings if its errors are uniformly distributed across systems.

**Key Challenge**: While the practical application of LLM judges is system-level decision-making (determining which model is better), existing benchmarks only evaluate instance-level capabilities. Performance at these two levels is not positively correlated; instead, the distribution pattern of judge errors, rather than the total error count, is the critical factor determining ranking quality.

**Goal**: (1) Establish the first system-level judge benchmark to directly measure the accuracy of LLM judges in system ranking; (2) reveal and quantify the behavioral characteristics (decisiveness and bias) of judges in system-level evaluation; (3) compare the impact of different judge implementations (numeric scoring vs. pairwise comparison vs. token probabilities) on ranking quality.

**Key Insight**: The authors observe that judges ranked highly on RewardBench do not necessarily perform best in system ranking tasks (as shown in Figure 3), confirming the misalignment between the instance level and system level. Using a response matrix of 63 systems $\times$ 500 instructions from the Arena Hard dataset, and Chatbot Arena human rankings as the ground truth, an end-to-end system-level judge evaluation pipeline is constructed.

**Core Idea**: Evaluating judges using the Kendall's Tau correlation between their system ranking and human ranking is a better reflection of their utility in real-world model selection than instance-level accuracy.

## Method

### Overall Architecture

The evaluation pipeline of JuStRank consists of three steps: (1) Data Preparation—obtaining responses from 63 systems to 500 instructions from Arena Hard v0.1; (2) Judgment Generation—having 48 judges (10 LLMs $\times$ 4 implementations + 8 reward models) rate all responses to generate a $K \times L$ score matrix $j_p(R)$; (3) Aggregation & Ranking—converting the score matrix into a system ranking vector $V^{p,a} \in \mathbb{R}^L$ using 4 aggregation methods, and calculating the Kendall's Tau correlation coefficient with Chatbot Arena human rankings. The entire pipeline generates approximately 1.5 million judgment scores.

### Key Designs

1. **Multi-Realization Judge Matrix**:

    - **Function**: Systematically covers different "invocation methods" of judges, ensuring the benchmark does not depend on a single scoring paradigm.
    - **Mechanism**: Four judge implementations are designed for each LLM. *Numeric* directs the judge to output a numeric score from 0-100; *Likert* directs the judge to output a 5-level text label (Very Bad $\rightarrow$ Very Good) which is then mapped to scores 1-5; *TokenProbs* asks "Is this a good response?" and takes the token probability ratio of "yes" and "no"; *Anchor* adopts comparative judgment, using the response of GPT-4-0314 as an anchor for pairwise preference judgments to output a preference score in $[-2, +2]$. Reward models directly output a scalar quality score. In the aggregation stage, four methods—Win-rate, Mean, Median, and Bradley-Terry—are provided to convert instance scores into system-level scores.
    - **Design Motivation**: Experiments show that the implementation method has an effect on ranking quality nearly as significant as the choice of the model itself (confirmed via ANOVA), making it necessary to include implementation methods as independent variables in the benchmark.

2. **Decisiveness Quantification**:

    - **Function**: Characterizes the tendency of a judge to widen the gap in pairwise system preferences.
    - **Mechanism**: For each judge, a scatter plot of its predicted win rate $WR^p(s_a, s_b)$ against the gold win rate $WR^g(s_a, s_b)$ is plotted. A decisive judge exhibits a pronounced S-curve—assigning more extreme high win rates to stronger systems and lower win rates to weaker systems. Drawing on classifier calibration theory, the curve is fitted with a cumulative Beta distribution function to obtain a single parameter $\alpha = \beta$. $\alpha = 1$ indicates no amplification/shrinkage, $\alpha > 1$ denotes decisiveness (amplifying gaps), and $\alpha < 1$ indicates indecisiveness.
    - **Design Motivation**: Decisiveness is a behavioral characteristic unique to system-level evaluation that cannot be captured by instance-level metrics. Experiments show that decisiveness is positively correlated with ranking quality ($r = 0.55$), suggesting that moderately decisive judges can distinguish systems faster under limited sample sizes.

3. **Bias Measurement & Correction**:

    - **Function**: Detects and quantifies a judge's unfair inclination toward specific systems.
    - **Mechanism**: The bias of a judge $j_p$ toward system $s_a$ is defined as the expected difference in pairwise win rates: $B_{s_a}^p = \mathbb{E}_{s_b \in S}(WR^p(s_a, s_b) - WR^g(s_a, s_b))$. A positive value indicates that the judge unjustifiably overestimates the system, while a negative value indicates underestimation. Since decisiveness itself can lead to stronger systems being positively biased and weaker systems negatively biased, the authors further calculate "decisiveness-corrected bias" $B'_{s_a}$ by replacing the gold win rate with the Beta-fitted predicted values, and then calculating the difference. Finally, the overall bias tendency of a judge is measured using the standard deviation of bias $\delta = \sigma_{s \in S}(B'^p)$.
    - **Design Motivation**: Bias is negatively correlated with ranking quality ($r = -0.56$) and almost uncorrelated with decisiveness ($r = -0.07$), indicating that they are independent system-level behavioral dimensions that jointly explain the variance in judge ranking capabilities.

### Evaluation Strategy

Using the English Hard Prompts subset of Chatbot Arena as the ground truth, Kendall's Tau correlation coefficient is employed to measure the alignment between the judge and human rankings. 59 systems appear simultaneously in the test data and Chatbot Arena, resulting in 968 pairwise comparisons.

## Key Experimental Results

### Main Results: Top-10 Judge Rankings

| Judge Model | Parameters | Type | Implementation | Aggregation Method | Kendall $\tau$ |
|---------|--------|------|---------|---------|-----------|
| Qwen2.5-72B-Instruct | 72B | LLM | Likert | Win-Rate | .83 |
| URM-LLaMa-3.1-8B | 8B | RM | Reward | Mean | .82 |
| GPT-4o-2024-11-20 | — | LLM | Anchor | Mean | .82 |
| Llama-3-1-405B-Instruct | 405B | LLM | Numeric | Mean | .81 |
| Mistral-Large-Instruct | — | LLM | Likert | BT | .81 |
| GPT-4o-mini | — | LLM | Numeric | Win-Rate | .81 |
| ArmoRM-Llama3-8B | 8B | RM | Reward | Mean | .80 |
| Llama-3-1-70B-Instruct | 70B | LLM | Numeric | Win-Rate | .80 |
| Skywork-Llama-3.1-8B | 8B | RM | Reward | Mean | .79 |
| Llama-3.1-8B-Instruct | 8B | LLM | TokenProbs | Mean | .78 |

### Influence of Implementation Method on Ranking Quality

| Implementation | Score Range | Best Model $\tau$ | Worst Model $\tau$ | Performance Span | Characteristics |
|---------|---------|----------|----------|---------|------|
| Numeric | 0-100 | .81 | .73 | .08 | Most stable, minimal difference across models |
| Likert | 1-5 | .83 | .71 | .12 | Highest ceiling but bottom is not guaranteed |
| Anchor | [-2,+2] | .82 | .67 | .15 | Only GPT-4o performs exceptionally well |
| TokenProbs | [0,1] | .78 | .62 | .16 | Largest fluctuation, lowest decisiveness |

### Key Findings

- **Small models $\neq$ poor rankings**: An 8B parameter reward model (URM-LLaMa-3.1-8B, $\tau=.82$) ties or even outperforms the 405B Llama-3.1 on system ranking, which shows that system ranking capability does not follow a simple scaling law.
- **Instance-level $\neq$ System-level**: The top-performing judges on RewardBench are not optimal on JuStRank. The ranking correlation between the two benchmarks is low (Figure 3), confirming the necessity of system-level benchmarks.
- **Implementation $\approx$ Model Choice**: ANOVA confirms that the contribution of the judge implementation method to ranking quality is virtually comparable to the choice of the model itself. Numeric/Likert are significantly better than Anchor/TokenProbs (statistically significant).
- **Decisiveness is a positive trait**: $\alpha$ is positively correlated with $\tau$ ($r = 0.55$). The Likert implementation is the most decisive, while TokenProbs is the most indecisive.
- **Shared bias across judges**: Athene-70B is systematically overestimated by the vast majority of judges (often ranked #1), while GPT-4-0613 (gold rank #27) drops to a median rank of #38.
- **Inconsistent self-bias**: The self-bias of LLM judges toward their own systems behaves inconsistently across different implementation methods, showing that self-bias is not a universal phenomenon.

## Highlights & Insights

- This work marks the first application of classifier calibration theory (Beta distribution fitting) to LLM judge analysis, defining "decisiveness" as a quantifiable system-level feature. Decisiveness is not a flaw—it enhances the separability between systems under limited evaluation budgets, helping to quickly filter models.
- The decisiveness-correction design in bias measurement is elegant: it first fits a Beta curve to eliminate systematic shifts caused by decisiveness, and then looks at the residuals, thereby decoupling genuine "unfair bias." This frames $\alpha$ and $\delta$ as two orthogonal dimensions of judge characteristics, jointly explaining the variance.
- An counter-intuitive phenomenon is uncovered: prompting LLMs to directly provide text labels (Likert) yields more accurate rankings than using numbers (Numeric) or making pairwise comparisons (Anchor). This might be because LLMs are better calibrated with verbalized confidence.

## Limitations & Future Work

- The gold ranking is derived from the English Hard Prompts subset of Chatbot Arena, which is not the exact same set of data as the Arena Hard test instructions. Although the distributions are similar, there is a risk of indirect comparison.
- The evaluation is limited to generic English instruction scenarios, not covering task-specific domains (e.g., code, math), specialized fields (e.g., medical, legal), or other languages.
- LLM judges are highly sensitive to prompt phrasing. Only one fixed prompt was used for each implementation in the experiments; results might vary with changes in prompt design.
- Human preference is treated as a monolithic concept rather than being decomposed into multiple dimensions (e.g., helpfulness, safety, style preferences). In reality, fundamental disagreements could exist among different annotators' preferences.

## Related Work & Insights

- **vs RewardBench**: RewardBench evaluates instance-level pairwise decision accuracy, whereas JuStRank evaluates the consistency of aggregated system rankings. They are complementary but not interchangeable—the former is suited for choosing "annotation tools" and the latter for "model evaluation tools."
- **vs Arena Hard / AlpacaEval**: These benchmarks fix GPT-4 as the judge and validate its ranking, whereas JuStRank compares the performance of multiple judges on system ranking tasks, providing a more comprehensive perspective.
- **vs Dorner et al. (2024)**: This prior work theoretically demonstrates the misalignment between instance-level and system-level evaluations. JuStRank is the first to empirically validate this theory through large-scale experiments.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The first system-level judge benchmark; the decisiveness/bias quantification methodology is highly original, although the core framework remains based on correlation analysis.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ With 48 judges, 63 systems, and 1.5 million ratings, the scale is unmatched in the field of LLM judge evaluation.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear motivational derivation, rigorous definitions of concepts, and highly informative, abundant figures and tables.
- **Value**: ⭐⭐⭐⭐ Offers direct practical value for choosing LLM judges, though the conclusions are bound to generic English scenarios and specific dataset distributions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] On Evaluating LLM Alignment by Evaluating LLMs as Judges](../../NeurIPS2025/llm_evaluation/on_evaluating_llm_alignment_by_evaluating_llms_as_judges.md)
- [\[ACL 2026\] Statistically Reliable LLM-Based Ranking Evaluation via Prediction-Powered Inference](../../ACL2026/llm_evaluation/statistically_reliable_llm-based_ranking_evaluation_via_prediction-powered_infer.md)
- [\[ACL 2025\] CalibraEval: Calibrating Prediction Distribution to Mitigate Selection Bias in LLMs-as-Judges](calibraeval_calibrating_prediction_distribution_to_mitigate_selection_bias_in_ll.md)
- [\[ACL 2025\] AD-LLM: Benchmarking Large Language Models for Anomaly Detection](ad-llm_benchmarking_large_language_models_for_anomaly_detection.md)
- [\[ICML 2026\] Margin-Adaptive Confidence Ranking for Reliable LLM Judgement](../../ICML2026/llm_evaluation/margin-adaptive_confidence_ranking_for_reliable_llm_judgement.md)

</div>

<!-- RELATED:END -->

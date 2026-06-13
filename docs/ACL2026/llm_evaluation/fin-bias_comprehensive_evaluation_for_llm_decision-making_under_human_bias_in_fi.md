---
title: >-
  [Paper Note] Fin-Bias: Comprehensive Evaluation for LLM Decision-Making under human bias in Finance Domain
description: >-
  [ACL 2026][LLM Evaluation][Herding] Fin-Bias utilizes 8,868 long-form analyst reports to construct a controlled benchmark with three input variants ("Original / Rating Removed / Replaced with Fake Rating"). It demonstrat…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "Herding"
  - "Analyst Reports"
  - "Investment Ratings"
  - "MPQA Lexicon"
  - "DPO"
date: 2026-05-08
content_hash: e5e71ac67c075fda
---

# Fin-Bias: Comprehensive Evaluation for LLM Decision-Making under human bias in Finance Domain

**Conference**: ACL 2026 Findings  
**arXiv**: [2605.09106](https://arxiv.org/abs/2605.09106)  
**Code**: https://github.com/Xiaoyu1216/Fin-Bias.git (Yes)  
**Area**: LLM Evaluation / Financial Decision-making / Behavioral Bias  
**Keywords**: Herding, Analyst Reports, Investment Ratings, MPQA Lexicon, DPO

## TL;DR
Fin-Bias utilizes 8,868 long-form analyst reports to construct a controlled benchmark with three input variants ("Original / Rating Removed / Replaced with Fake Rating"). It demonstrates that 18 LLMs (including GPT-5 and Claude-4-Sonnet) exhibit significant "herding" in financial investment ratings—even completely fabricated fake ratings are blindly followed in 30% of samples. Filtering human opinions in the context using the MPQA subjective lexicon combined with DPO fine-tuning can elevate open-source 8B models to a level of accuracy surpassing GPT-5.

## Background & Motivation
**Background**: LLM agents in financial scenarios (e.g., FinMem, FinCon, TradingAgents) rely heavily on analyst reports, news, and tweets containing human opinions for trading decisions. Existing financial benchmarks (FinQA, ConvFinQA, FiQA-SA, INVESTORBENCH, etc.) typically evaluate short contexts, lack explicit injection of human bias, or focus only on a few high-profile stocks.

**Limitations of Prior Work**: Analysts exhibit systematic "over-optimism" bias (72.28% Bullish vs. 0.29% Bearish in this sample). Current LLM decision-making frameworks rarely evaluate whether models blindly follow such systematic biases, nor do they assess whether models can maintain independent judgment in extreme scenarios involving "fake rating reversals."

**Key Challenge**: (1) Real-world financial decision-making requires LLMs to think independently within voluminous subjective text, (2) yet LLMs tend to treat explicit human judgments in the context as "soft labels." This herding behavior causes LLMs to fail when the majority is wrong—a common occurrence in financial markets.

**Goal**: (1) Quantify the dependence of 18 mainstream LLMs on analyst ratings in financial decision-making; (2) test their resistance to contradictory "fake ratings"; (3) evaluate the performance gap between LLM investment capabilities and real analysts; (4) provide feasible strategies to mitigate herding.

**Key Insight**: The authors construct three minimal-pair variants of the same report—Original (containing the real rating in the first sentence) / Rating Removed / Replaced with Fake Rating. By using a uniform chain-of-thought prompt to generate Bullish/Neutral/Bearish ratings, the differences between these versions allow for the precise isolation of the "marginal impact of human bias signals."

**Core Idea**: "Herding tendency" is treated as a measurable score (agreement rate between model and human ratings). Three types of ground-truth are used: (a) actual analyst ratings, (b) fake ratings, and (c) "true" investment ratings derived from 60-day Cumulative Abnormal Return (CAR) quantiles.

## Method

### Overall Architecture
Fin-Bias consists of four modules: (1) **Data Construction**: 8,868 PDF analyst reports covering 9 industries were collected from Yahoo Finance, with an average of 4,000 tokens per report; (2) **Three-version Perturbations**: Original, Removed (first sentence deleted), and Fake (first sentence replaced with an opposing rating); (3) **Three Types of Ground-truth + Herding Score**: Comparing LLM outputs against analyst ratings, fake ratings, and CAR-based labels; (4) **Mitigation Strategies**: Sentence filtering using the MPQA subjective lexicon and Direct Preference Optimization (DPO). The entire workflow uses a standardized CoT prompt template (Figure 1) for cross-model comparison.

### Key Designs

1.  **Three-version minimal-pair perturbation + Herding Score**:
    *   **Function**: Precisely measures model dependence on "explicit human rating" signals through sentence-level causal manipulation.
    *   **Mechanism**: Herding Score is defined as $\text{Herding Score} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(m_i, a_i)$, where $m_i$ is the model rating and $a_i$ is the analyst or fake rating. A high score with real ratings implies "reasonable alignment," while a high score with fake ratings indicates blatant herding (since fake ratings contradict the internal logic of the report).
    *   **Design Motivation**: Accuracy alone fails to distinguish "independent correct judgment" from "copying a correct analyst." Comparing three versions through two herding scores helps separate signal from noise.

2.  **CAR quantile-based unbiased ground-truth**:
    *   **Function**: Deduces objective "correct/incorrect" investment ratings from actual market returns, bypassing the systematic bias inherent in analyst ratings.
    *   **Mechanism**: First, the market model $R_{i,t} = \alpha_i + \beta_i R_{m,t} + \varepsilon_{i,t}$ is used to estimate $\hat{\alpha}_i$ and $\hat{\beta}_i$ via OLS. Then, the Cumulative Abnormal Return is calculated for 60 trading days following the report release: $CAR = \sum \hat{\alpha}_i = \sum(\bar{R}_i - \hat{\beta}_i \bar{R}_m)$. Finally, $CAR$ values are ranked annually: top 30% = Bullish, bottom 30% = Bearish, and central 40% = Neutral.
    *   **Design Motivation**: Simple daily return thresholds are arbitrary and overlook firm-specific risks. $CAR$ is a standard measure of risk-adjusted abnormal return in finance, reflecting "excess returns." Annual quantiles mitigate the impact of market volatility. The resulting label distribution (30/30/40) is significantly more balanced than the analyst ratings (72.28/24.74/0.29).

3.  **MPQA subjective lexicon filtering + DPO mitigation**:
    *   **Function**: Removes "subjective opinion sentences" that might induce herding from the context to encourage independent reasoning. DPO further internalizes "independent judgment vs. following analysts" as preference pairs.
    *   **Mechanism**: Sentences containing `strongsubj` terms from the MPQA Subjectivity Lexicon are removed entirely. In the DPO stage, triplets $(x, y_w, y_l)$ are constructed: $x$ is the original biased report, $y_w$ is "independent reasoning" derived from market truth, and $y_l$ is "following reasoning" generated to match the analyst's view. The objective is to maximize $\log \frac{\pi(y_w|x)}{\pi(y_l|x)}$.
    *   **Design Motivation**: Deleting only the explicit rating is superficial; the remaining text remains saturated with implicit bias. Lexicon filtering provides a prompt-level refinement, while DPO facilitates model-level internalization of "healthy skepticism" toward human signals.

### Loss & Training
The evaluation uses zero-shot CoT prompts. DPO fine-tuning was applied to open-source models (Qwen3-8B, Qwen2.5-7B-It, Meta-Llama-3-8B-It) with the following objective:

$$\mathcal{L}_{\text{DPO}} = -\log \sigma\left(\beta \log \frac{\pi(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)$$

Note: $\beta$ controls the divergence from the reference policy following standard DPO implementation.

## Key Experimental Results

### Main Results
Herding Scores and Accuracy (vs. 33% analyst baseline) for 18 LLMs across 9 industries:

| Model | Herd vs analyst (with/without rating) | Herd vs fake rating | Accuracy (with/without rating) | Accuracy after MPQA |
| :--- | :--- | :--- | :--- | :--- |
| GPT-5 | 94.6 / 87.4 | 33.5 | 33.0 / 33.6 | 34.16 |
| GPT-4 | 95.9 / 89.5 | 48.8 | 33.0 / 33.9 | 35.88 |
| Claude-4-Sonnet | 90.6 / 79.5 | 35.9 | 33.0 / 33.8 | 35.70 |
| Mistral-7B-It-v0.3 | 96.4 / 78.8 | **60.4** | 33.1 / 34.1 | **35.99** |
| Qwen3-8B | 98.3 / 84.4 | 36.9 | 33.1 / 34.2 | **36.49** |
| DeepSeek-V2-Lite-Chat | 78.0 / 57.3 | **55.9** | 33.9 / 34.6 | 35.30 |
| Yi-1.5-9B-Chat-16K | 83.4 / 69.7 | 32.8 | 34.5 / 28.7 | **37.67** |
| Analyst baseline | — | — | 33.08 | — |

**Key Findings**: (1) Herding Scores increase by 5-10 percentage points for all models when analyst ratings are included, reaching 90%+; (2) **Fake ratings are blindly followed in 30% of samples**, including 33.5% for GPT-5; (3) Accuracy remains near the 33% analyst baseline across all models, with no model significantly outperforming it.

### Ablation Study
Impact of MPQA filtering and DPO on open-source model accuracy (average across 9 industries):

| Model | Rating Removed | + MPQA Filter | + MPQA + DPO | Rel. to Analyst |
| :--- | :--- | :--- | :--- | :--- |
| Qwen3-8B | 34.2 | 36.49 | **38.23** | **+5.15** |
| Meta-Llama-3-8B-It | 27.5 | 34.91 | **37.09** | **+4.01** |
| Qwen2-7B-It | 27.1 | 33.82 | **35.15** | +2.07 |
| Analyst | 33.08 | — | — | — |
| GPT-5 (no DPO) | 33.6 | 34.16 | — | +1.08 |

### Key Findings
- **Herding is independent of model scale**: GPT-5 blindly follows fake ratings (33.5%), while Mistral-7B is the most prone (60.4%), suggesting this is not a scale-limited issue.
- **Divergence in open-source stability**: When ratings are removed, models like Gemma and Llama experience a performance drop of 6 points, whereas Mistral/Qwen3/DeepSeek show slight improvement, suggesting inherent reasoning capacity that was previously suppressed.
- **The MPQA + DPO "Double Buff"**: Allows 8B models like Qwen3 to reach 38.23% accuracy, surpassing GPT-5 (33.0%) and analysts (33.08%). This highlights that "debiasing + alignment" is more efficient than scaling.
- **Consistent Industry Trends**: Performance is lower in Real Estate and Healthcare due to macro sensitivities, while Financial Services and Utilities remain stable.

## Highlights & Insights
- **Three-version perturbation + Herding Score** provides a clean causal design, transferable to other domains where contextual human judgment is present (medical, legal, policy).
- **CAR-based unbiased ground-truth** is a rigorous methodological choice, breaking the circular logic of using analyst ratings as labels and setting a standard for financial LLM benchmarks.
- **Preference pairs based on market truth vs. analyst-style reasoning** in DPO is an ingenious way to define independent thinking as a learnable preference rather than a vague "robustness" goal.
- **Counter-intuitive insight**: Open-source models can surpass GPT-5 after MPQA filtering, implying that independent reasoning capacity is not strictly bound to parameter size; prompt and data engineering can be more impactful.

## Limitations & Future Work
- The study focuses on single-agent decision-making and does not address herding in multi-agent frameworks, where bias can propagate.
- The use of analyst reports may not generalize to retail investor contexts (e.g., Reddit/Twitter), where herding patterns might differ.
- The framework has only been tested in finance; its efficacy in medical or legal domains requires verification.
- DPO data depends on market efficiency assumptions, which may not hold under extreme market conditions.
- Reasoning models (e.g., o1-style) were not analyzed for herding behavior, representing a potential area for future comparison.

## Related Work & Insights
- **vs. INVESTORBENCH / Fintrade**: Those focus on limited star stocks without explicit human bias; Fin-Bias offers broader coverage (9 industries) and a stronger causal framework.
- **vs. ACL18 (Xu & Cohen)**: While that used tweets for prediction with small sample sizes, Fin-Bias uses 8,868 reports and complete ground-truth construction.
- **vs. DeLLMa (Liu 2025)**: Evaluates uncertainty using price data; this study uses long-form text, which is more representative of professional investing.
- **vs. TradingAgents (Xiao 2025)**: Proposed multi-agent frameworks; the herding score developed here could serve as a diagnostic tool for their agents.

## Rating
- Novelty: ⭐⭐⭐⭐ The integration of the Herding Score with three-version perturbation and fake rating scenarios is a unique and clean causal design for financial benchmarks.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive testing across 18 models, 9 industries, and 3 perturbation types, complemented by DPO results for open-source models.
- Writing Quality: ⭐⭐⭐⭐ Methodologies for data, perturbations, and ground-truth are clearly articulated; the appendix includes case studies and prompt templates.
- Value: ⭐⭐⭐⭐ Provides both proof of bias and a concrete solution. The open-source dataset is expected to be highly valuable for the industry.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Contrastive Decoding Mitigates Score Range Bias in LLM-as-a-Judge](contrastive_decoding_mitigates_score_range_bias_in_llm-as-a-judge.md)
- [\[ACL 2026\] Common to Whom? Regional Cultural Commonsense and LLM Bias in India](common_to_whom_regional_cultural_commonsense_and_llm_bias_in_india.md)
- [\[ICLR 2026\] BiasScope: Towards Automated Detection of Bias in LLM-as-a-Judge Evaluation](../../ICLR2026/llm_evaluation/biasscope_towards_automated_detection_of_bias_in_llm-as-a-judge_evaluation.md)
- [\[ACL 2026\] When Vision-Language Models Judge Without Seeing: Exposing Informativeness Bias](when_vision-language_models_judge_without_seeing_exposing_informativeness_bias.md)
- [\[ACL 2026\] Stability vs. Manipulability: Evaluating Robustness Under Post-Decision Interaction in LLM Judges](stability_vs_manipulability_evaluating_robustness_under_post-decision_interactio.md)

</div>

<!-- RELATED:END -->

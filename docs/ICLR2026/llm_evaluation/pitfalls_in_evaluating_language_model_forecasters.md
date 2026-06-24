---
title: >-
  [Paper Note] Pitfalls in Evaluating Language Model Forecasters
description: >-
  [ICLR 2026][LLM Evaluation][LLM forecasting] This is a position/analysis paper: the authors systematically categorize two major classes of pitfalls unique to the emerging field of "LLM-based future event forecasting"—various forms of temporal leakage in backtesting that render results untrustworthy, and the difficulty of extrapolating benchmark scores to real-world forecasting capabilities. Using numerous specific examples from existing literature…
tags:
  - "ICLR 2026"
  - "LLM Evaluation"
  - "LLM forecasting"
  - "backtesting"
  - "temporal leakage"
  - "benchmark gaming"
  - "evaluation reliability"
date: 2026-05-08
content_hash: 3b8895345b23289d
---

# Pitfalls in Evaluating Language Model Forecasters

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=z85kARAoyD](https://openreview.net/forum?id=z85kARAoyD)  
**Code**: To be confirmed  
**Area**: LLM Evaluation / Forecasting / Temporal Leakage  
**Keywords**: LLM forecasting, backtesting, temporal leakage, benchmark gaming, evaluation reliability

## TL;DR
This is a position/analysis paper: the authors systematically categorize two major classes of pitfalls unique to the emerging field of "LLM-based future event forecasting"—various forms of temporal leakage in backtesting that render results untrustworthy, and the difficulty of extrapolating benchmark scores to real-world forecasting capabilities. Using numerous specific examples from existing literature, they argue that claims of LLMs reaching or surpassing human-level forecasting must be seriously questioned.

## Background & Motivation
**Background**: Forecasting, the assignment of probabilities to future events, is a core capability for decision-making. Over the past two years, several works (Halawi 2024, Phan 2024, Schoenegger 2024, etc.) have utilized LLMs as forecasting systems and reported performance comparable to or exceeding that of human forecasters. The "gold standard" for evaluating these systems is to deploy them on unresolved questions and score them once the outcome is realized, but this process takes months or years, hindering rapid iteration.

**Limitations of Prior Work**: To achieve fast evaluation, researchers commonly adopt **backtesting (retrodiction)**—freezing the system's knowledge at a past time $T$ and asking it to predict events that resolved between $T$ and the present. While seemingly elegant, the premise that the "system contains no information after $T$" is often violated in subtle ways, leading to high scores that are neither credible nor representative of true forecasting power.

**Key Challenge**: The essence of forecasting is **temporal directionality**. If even a trace of the "future" enters the training, retrieval, or model weights, the entire evaluation is contaminated. LLMs, which are trained on massive datasets containing future information and often paired with retrieval-augmented generation (RAG), are particularly prone to "peeking at the answers" unknowingly. This makes evaluating forecasting **compounded** by all existing ML evaluation challenges, making it significantly harder than evaluating knowledge about the past or present.

**Goal**: To clarify issues that are partially known in the forecasting community but have never been systematically analyzed for LLM systems, categorized into: (1) whether evaluation results can be trusted; (2) whether evaluation performance can extrapolate to the real world; and to provide concrete evidence and mitigation suggestions for each issue.

**Core Idea**: Rather than reporting another "superhuman" score, the focus should be on making the evaluation methodology robust. The paper proposes no new model but provides a "forecasting evaluation checklist," calling for the community to judge LLM forecasting capabilities with more rigorous methodology.

## Method

### Overall Architecture
This is an analytical paper rather than an algorithmic pipeline. The "method" is a **diagnostic framework for LLM forecasting evaluation**, grouping issues under two main axes, with each issue supported by a mechanism description, empirical evidence from existing benchmarks, and possible solutions. The axes are:

- **Challenge 1: Trustworthiness of Evaluation Results**: The assumption that backtesting freezes knowledge at $T$ is violated at three levels: logical leakage, unreliable date-restricted retrieval, and over-reliance on model cutoffs.
- **Challenge 2: Extrapolation to Real-World Capability**: Even if evaluations are clean, high benchmark scores may not equate to strong forecasting power due to piggybacking on human forecasts, gaming via betting strategies, and skewed data distributions.

Finally, the paper provides a **Future Work** outlook: if backtesting is used as a training objective to optimize stronger forecasters, temporal data itself introduces new leakage. The diagnostic framework is summarized below through four key design points following these axes.

### Key Designs

**1. Logical Leakage and Backtesting Question Construction Bias: Leakage at the Selection Stage**

In backtesting, researchers "stand in the future" to select questions from past time $T$, and the selection strategy itself can implicitly leak the future. The paper uses the "time traveler" analogy: if someone from 2035 asks you, "Will alien life be discovered by 2040?", you can infer the answer is likely "Yes"—otherwise, they couldn't score the question. Similarly, the common practice of requiring a question to be "posed before $T$ and resolved by now" leaks the answer. If a question from 2021 asks "Will Queen Elizabeth live to 100?", an LLM being backtested in 2025 can infer the answer cannot be "Yes" (as she would have turned 100 in 2026). The authors found at least 3.8% of questions in the Halawi 2024 dataset belong to such "pre-resolved" cases, and at least 10% in Tao 2025 are trivially solvable because they resolved at a known point.

More severely, **post-hoc question generation** (e.g., Dai 2025, Paleka 2024 using news to create questions inversely) introduces distribution bias. News tends to report things that *happened* rather than things that *uneventfully did not happen*, similar to **survivorship bias** in finance. For instance, if a company collapses in Q1 2025 and news coverage stops, a question like "What was the company's Q3 2025 revenue? (Answer $0$)"—which would be perfectly reasonable from a 2024 perspective—is rarely generated, causing a systematic misalignment between backtesting and real-world live testing distributions. The authors found news-based questions often contain shortcuts, with weak classifiers hitting 80%+ accuracy on binary questions in Dai 2025, estimating 90%+ are overly specific questions that no one would have asked before the event occurred. **Goal**: Retain only questions that can be verified regardless of resolution; ensure retrospective generation mimics questions actually asked in the past.

**2. Unreliable Date-Restricted Retrieval and Model Cutoffs: Engineering Failure to Freeze Knowledge**

Many LLM forecasting systems use RAG. Backtesting requires restricting retrieval to information available at time $T$, but modern search engine date filters are highly unreliable for three reasons: web pages are updated without changing publication dates, pages contain present-day metadata (comments/ads/sidebars), and engines often do not know the original publication time. More subtly, **retrieval algorithms themselves use future knowledge**. Asking Google for 2022 results does not use a 2022 ranking algorithm; articles that became "important later" are ranked higher. A striking example: searching "Jan 6" restricted to before 2020 yields results dominated by US politics—an association that only became strong in January 2021. Searching "Wuhan" before December 2018 highlights the Virology Institute, which only gained fame after the pandemic.

Parallel to this is the **over-reliance on model cutoffs**. Knowledge cutoffs are reliability hints for users, not guarantees for train/test splits. Models often know about events after their cutoff. The authors found that while `gpt-4o-2024-08-06` normally denies knowledge of events after November 2023, a system prompt jailbreak telling it "Your cutoff is Nov 2023" allows it to describe the Biden-Xi meeting on 2023-11-15 (which was only announced on Nov 8). Furthermore, system prompts and scaffolding leak the future (e.g., Claude's prompt stating "Trump is currently President, inaugurated 2025-01-20"). **Mechanism**: Use publication dates as knowledge upper bounds with a buffer of several months; use restricted corpora with reliable dates (Wikipedia, news) or revert to knowledge-free retrieval like TF-IDF/older embeddings at the cost of retrieval quality.

**3. Three Pitfalls in Extrapolating to Real Capability: High Scores $\neq$ Forecasting Skill**

Even with clean evaluations, benchmark rankings might not map to real forecasting capability:

First, **piggybacking on human forecasts**: Many questions are scraped from human prediction markets. Since human crowd probabilities are easily found online or in training data, claiming "LLMs match humans" becomes a **circular argument**—the model may simply be copying the market's aggregated probability. This directly affects the interpretation of ForecastBench (which uses crowd forecasts as the gold standard). A system can reach the "gold standard" simply by retrieving recent aggregate probabilities. The authors suggest measuring the system's **edge relative to the crowd**, even feeding historical market data directly to the model.

Second, **gaming through betting**: In the real world, many events share underlying stochasticity. Maximizing the "probability of being the top forecaster" encourages **betting on correlated risks** rather than honest belief reporting. If forecasting several political/economic events in September 2024 for 2025, they all correlate with "who wins the 2024 election." An honest forecaster averages conditional probabilities; a system aiming for the top of the leaderboard should **assume a specific election outcome** and bet everything on one side. This leads to the **winner's curse**: across multiple LLMs with different biases, the top-ranked model is often one with systematic overconfidence rather than true skill. **Mechanism** (borrowed from finance): Report risk-adjusted returns and evaluate over multiple disjoint backtesting periods (shifting the backtesting date so latent variables are only correlated in one period).

Third, **data distribution skew**: Questions from prediction markets reflect user interests—Polymarket skews toward crypto and sports, while Manifold is full of personal questions like "Will I go to the gym today?" Market sources over-represent US politics and sports. Non-market sources (like ForecastBench) use limited templates that look like time-series forecasting. While ImageNet's skew toward dog breeds still allows for transferable features, there is no evidence that performance on current forecasting benchmarks transfers to general forecasting capability.

**4. Temporal Confusion when Backtesting is a Training Objective: Learning Forecasting is Hard**

Looking forward, one might use backtesting as a training task to improve forecasting, but temporal data makes optimization itself prone to leakage. Standard ML uses random train/test splits, but forecasting must use **chronological splits** (train set entirely before test set). The problem is: if optimizing on ordered events $e_1, \dots, e_n$, when predicting $e_{i+1}$, the model parameters already encode $e_1, \dots, e_i$. Thus, the test measures "predicting $e_{i+1}$ after learning early events" rather than "predicting $e_{i+1}$ from the original cutoff." Sorting by date only teaches the model to predict shorter time horizons. The ideal approach is to **penalize memorization**, forcing the model to learn to forecast without remembering "what specifically happened"—which is inherently difficult to implement.

## Key Experimental Results

As this is an analysis/position paper, there is no standard experimental table. Instead, audits of existing benchmarks serve as "evidence."

### Main Results

| Source Benchmark | Finding | Value |
|------------------|---------|-------|
| Halawi et al. 2024 | Questions that are "pre-resolved" or require no true forecasting | $\ge 3.8\%$ |
| Tao et al. 2025 (PROPHET) | Questions trivially solvable because they resolved by a certain point | $\ge 10\%$ |
| Dai et al. 2025 (News-gen) | Accuracy achievable by weak classifiers on binary questions | $> 80\%$ |
| Dai et al. 2025 | Estimated "overly specific" questions that wouldn't be asked beforehand | $> 90\%$ |
| ForecastBench | Severe distribution skew (e.g., extremely high Security & Defense), non-market questions use few templates | See Original Table 1 |

### Key Findings
- **Logical leakage is pervasive**: Multiple mainstream forecasting benchmarks contain questions solvable via logical inference or weak classifiers, suggesting "LLM superhuman" conclusions may be built on contaminated evaluations.
- **Leakage categorized by strength**: Some cases are clear evidence (pages containing data after the cutoff), others are "strong assumptions" (retrieval ranking influenced by future knowledge). The authors honestly distinguish between the two.
- **Relative comparisons are not immune**: While some leakage affects all systems, different systems exploit benchmark flaws differently, distorting rankings. Relative orderings cannot be blindly trusted.
- **Absolute scores are nearly uninterpretable**: Scores depend heavily on question distribution. Future evaluations should report multiple metrics, use recent events, and validate across disjoint backtesting periods.
- **Jailbreaking penetrates cutoffs**: A single system prompt can force `gpt-4o` to reveal events after its claimed cutoff, proving that cutoffs are not a safe boundary for train/test splits.

## Highlights & Insights
- The **"time traveler" analogy** brilliantly distills logical leakage: if the subject knows they are being "scored in the future," they can reverse-engineer answers— a leakage channel unique to forecasting and absent in traditional ML.
- **Transferring decades of financial backtesting experience** (survivorship bias, point-in-time data modifications, risk-adjusted returns, multi-period backtesting, backtest overfitting) provides a ready-made methodological toolbox for LLM forecasting.
- **Identifying "Leaderboard Incentives $\neq$ Capability Incentives"**: In a correlated real world, maximizing win probability rewards gambling rather than calibration. This winner's curse perspective applies to any "single-period ranking + highly correlated tasks" evaluation design.
- **Honest labeling of evidence strength**: Explicitly marking unprovable leakage as "strong assumptions" demonstrates the rigor appropriate for an analysis paper.

## Limitations & Future Work
- The paper **cannot prove** these flaws necessarily inflate LLM scores—it argues that these evaluations are *untrustworthy*, not necessarily that the scores are *false*, which the authors explicitly acknowledge.
- Many quantitative estimates (e.g., "90% are overly specific questions") are **subjective**; the authors admit better quantitative metrics are needed to characterize these effects.
- Proposed solutions are often **trade-offs** rather than cures: restricted retrieval sacrifices quality; synthetic generation brings back the question of realistic question distributions.
- The forward-looking idea of "penalizing memorization so the model learns forecasting without specifics" is conceptual and lacks a concrete implementation strategy.

## Related Work & Insights
- **vs. Halawi 2024 / Phan 2024 / Schoenegger 2024**: These works offer optimistic conclusions. This paper audits their evaluations, showing how leakage, circular comparisons, and distribution skew may cause overestimation.
- **vs. ForecastBench (Karger 2024)**: Acknowledges ForecastBench is cleaner regarding temporal/logical leakage but notes that using crowd forecasts as the gold standard is vulnerable to retrieval piggybacking and contains skewed distributions.
- **vs. Dai 2025 / Paleka 2024**: Recognizes some fixes provided by counterfactual rewriting in news-based generation but points out the creation of "questions no one would ask beforehand," highlighting the tension between synthetic generation and realism.
- **vs. Financial Backtesting Literature (Arnott 2018, Bailey 2015, etc.)**: Uses established knowledge on risk adjustment and point-in-time data as a positive reference, advocating that LLM evaluation should directly adopt these mature practices.

## Rating
- Novelty: ⭐⭐⭐⭐ Proposes no model, but is the first to systematically apply forecasting/financial community leakage issues to LLM evaluation with high empirical density.
- Experimental Thoroughness: ⭐⭐⭐⭐ Uses multi-benchmark audits and specific jailbreak/retrieval cases; however, lacks a unified, reproducible quantitative protocol.
- Writing Quality: ⭐⭐⭐⭐⭐ Excellent readability with vivid analogies and a clear "Problem-Evidence-Solution" structure.
- Value: ⭐⭐⭐⭐⭐ Crucial for the credibility of the "LLMs can predict the future" narrative; serves as a high-utility guide for evaluation designers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Addressing Pitfalls in the Evaluation of Uncertainty Estimation Methods for Natural Language Generation](addressing_pitfalls_in_the_evaluation_of_uncertainty_estimation_methods_for_natu.md)
- [\[ICLR 2026\] How Reliable is Language Model Micro-Benchmarking?](how_reliable_is_language_model_micro-benchmarking.md)
- [\[ICLR 2026\] Train-before-Test Harmonizes Language Model Rankings](train-before-test_harmonizes_language_model_rankings.md)
- [\[ICLR 2026\] Don't Pass@k: A Bayesian Framework for Large Language Model Evaluation](dont_passk_a_bayesian_framework_for_large_language_model_evaluation.md)
- [\[ICLR 2026\] GDPval: Evaluating AI Model Performance on Real-World Economically Valuable Tasks](gdpval_evaluating_ai_model_performance_on_real-world_economically_valuable_tasks.md)

</div>

<!-- RELATED:END -->

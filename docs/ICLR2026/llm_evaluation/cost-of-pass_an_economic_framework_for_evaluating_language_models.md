---
title: >-
  [Paper Note] Cost-of-Pass: An Economic Framework for Evaluating Language Models
description: >-
  [ICLR 2026][LLM Evaluation][Paper Note] Borrowing the "production frontier" theory from economics, this paper proposes **cost-of-pass** (the expected dollar cost to generate one correct answer) as a unified evaluation framework that merges "accuracy × inference cost" into a single metric. It uses this framework to reveal the economic niches of models of diff
tags:
  - ICLR 2026
  - LLM Evaluation
date: 2026-05-08
content_hash: 1b4c15acc49091e1
---
# Cost-of-Pass: An Economic Framework for Evaluating Language Models

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=vC9S20zsgN](https://openreview.net/forum?id=vC9S20zsgN)  
**Code**: https://github.com/mhamzaerol/Cost-of-Pass  
**Area**: LLM Evaluation  
**Keywords**: Language model evaluation, inference cost, production frontier, cost-benefit, economic framework

## TL;DR
Borrowing the "production frontier" theory from economics, this paper proposes **cost-of-pass** (the expected dollar cost to generate one correct answer) as a unified evaluation framework that merges "accuracy × inference cost" into a single metric. It uses this framework to reveal the economic niches of models of different sizes across various tasks, the rate of decline in the cost frontier over the past year, and the fact that most inference-time enhancement techniques (majority voting, self-correction) are actually not cost-effective on the scale of "buying correctness."

## Background & Motivation
**Background**: Language model evaluation has long reported "capability" and "cost" separately—leaderboards only compare accuracy, while pricing tables only list the cost per million tokens. However, users truly care about whether "spending one dollar can yield a correct answer." As the model ecosystem becomes increasingly crowded, whether a new model is worth using must be viewed in the context of "how much money it saves compared to the cheapest existing feasible solution," rather than looking at its accuracy in isolation.

**Limitations of Prior Work**: Previous attempts to incorporate cost into evaluation mostly relied on **fixed inference budgets** (limiting the number of tokens per question), **heuristic scoring**, or non-monetary proxies such as **conciseness**. These practices tie conclusions to specific constraints or empirical weights, lacking both universality and interpretability in an economic sense—one cannot directly state "how much each correct answer costs for this model on this task."

**Key Challenge**: Models face an unavoidable trade-off—stronger models or heavier inference-time techniques can yield higher accuracy, but at the cost of higher computational and monetary expenses. Looking only at accuracy mindlessly favors reasoning models, while looking only at unit price mindlessly favors lightweight models; neither answers "who is the most cost-effective overall."

**Goal**: Construct a unified metric that reflects both accuracy and monetary cost while providing economic interpretability, thereby answering three sub-questions: (1) which task categories are most cost-effective for different model sizes; (2) how fast the cost frontier has declined over the past year and what drives it; (3) whether common inference-time enhancement techniques are worth the cost.

**Key Insight**: The authors observe that language models are essentially **stochastic producers** in economics—given an input (problem), they produce a qualified product (correct solution) with a certain probability; failure requires a retry. This is highly isomorphic to Farrell’s (1957) production efficiency theory and Aigner et al.'s (1977) stochastic frontier production function. Thus, the mature framework of "minimizing the cost per unit of qualified output" is migrated to LLM evaluation.

**Core Idea**: Use "expected cost to generate one correct answer = cost per inference ÷ probability of success" as a unified metric, then take the infimum of all available models (including human experts) as the "cost frontier," translating language model progress into a cost curve that declines over time.

## Method

### Overall Architecture
The starting point of the framework is to formalize "evaluating the economic efficiency of a model on a problem" into two quantities: success probability $R_m(p)$ (the probability that model $m$ produces a correct answer in one inference for problem $p$) and per-inference cost $C_m(p)$ (the expected dollar cost of one inference attempt, calculated from prompt + generated tokens multiplied by the provider's unit price plus additional fees like third-party calls). Starting from these two quantities, the paper builds a four-level conceptual structure: first defining the **cost-of-pass** for a single model, then taking the infimum across all models to obtain the **frontier cost-of-pass**, incorporating a **human expert baseline** to give the frontier a reference frame and ensure it remains finite, and finally using **Gain** to measure how much each new model pushes the frontier down, thus quantifying "progress" as a curve declining over time. The entire evaluation is implemented through a 5-step operational process (estimate success rate → estimate single cost → calculate cost-of-pass → determine frontier → aggregate on benchmark). As this is an analysis/evaluation framework paper, the core is not a new network architecture but a set of interlocking economic metric definitions; therefore, these metrics are explained individually below.

### Key Designs

**1. cost-of-pass: Compressing Accuracy and Cost into "Price per Correct Answer"**

Addressing the pain point that "accuracy and cost have been reported separately, failing to answer which is more cost-effective," the paper defines the efficiency of a single model on problem $p$ as the expected number of attempts multiplied by the cost per attempt. Since outputs are stochastic and attempts are assumed to be independent, the expected number of attempts to get the first correct solution is $1/R_m(p)$, so the cost-of-pass is:

$$v(m, p) = \frac{C_m(p)}{R_m(p)}.$$

Its meaning is straightforward: the efficiency of converting financial resources into correct output. This definition has two naturally beneficial properties—when a model cannot solve the problem at all ($R_m(p)=0$), $v(m,p)=\infty$, correctly marking it as "infeasible"; meanwhile, lightweight models with low unit prices but frequent errors and reasoning models with high unit prices but near-zero errors are compared fairly on the same scale, which cannot be achieved by looking at accuracy or token price alone.

**2. Frontier cost-of-pass and Human Expert Baseline: Using the "Cheapest Feasible Solution in the Ecosystem" as a Benchmark**

A single model's cost-of-pass can only evaluate one model, but users face an entire model ecosystem. Following the economic cost frontier $V_u=\min_{f_i\in F}\{w_i^\top x \mid f_i(x)\ge u\}$, the paper defines the **LM cost frontier** for problem $p$ as the minimum cost-of-pass across all available strategies:

$$V_p(M) = \min_{m\in M} v(m, p).$$

However, a pure LM frontier has two flaws: it doesn't tell you if the LM is cost-effective compared to hiring humans, and it becomes infinite when no LM can solve $p$. The paper's solution is to incorporate **human experts** as a strategy $m_{\text{expert}}$—assuming qualified experts are nearly always correct ($R_{\text{expert}}(p)\approx 1$), their cost-of-pass is approximately the labor cost of hiring the expert to solve the problem $v(\text{expert},p)\approx C_{\text{expert}}(p)$ (estimated through "evidence hierarchies" such as benchmark-annotated wages/time, related research compensation, or per-question time derived from competition rules). Thus, the frontier with the expert baseline is:

$$V_p(M \cup M_0) = \min\big(V_p(M),\ v(\text{expert}, p)\big),$$

which is always finite and naturally answers "how much money is saved by using LM over hiring humans."

**3. Gain and Tracking Progress Over Time: Translating "Model Iteration" into a Declining Cost Curve**

With the baseline-inclusive frontier, the paper further measures the "economic contribution of each new model." As new models are released at time $t$, the set of available models expands $M_t = M_{t-1}\cup\{m_t\}$, and the frontier $V_p(M_t)$ must be a **non-increasing** sequence. The **Gain** brought by a new model is the magnitude by which it pushes the frontier down:

$$G_p(\{m_t\}, M_{t-1}) = V_p(M_{t-1}) - V_p(M_{t-1} \cup \{m_t\}).$$

A larger gain indicates that the new model is "cheaper" compared to the previous best solution (including humans) for solving problem $p$, signifying a more notable economic contribution. By taking the empirical expectation of per-question metrics over the problem distribution $P=\{p_i\}$ ($V_{p\sim D}(M_t)\approx \mathbb{E}_{p\sim P}[V_p(M_t)]$, same for gain), one can plot the frontier decline curve over time across the entire benchmark. The paper also fits an exponential decay $V_p(M_t)\approx a\,e^{-bt}+c$ to this curve, using $T_{1/2}=\ln 2/b$ to quantify "how long it takes for the cost to halve."

**4. Counterfactual Frontier: Inferring Who is Indispensable by "Removing a Family"**

While the previous metrics show the frontier is declining, this design answers "who is driving it." The paper performs counterfactual analysis: dividing models into three families $M_g$ (Lightweight, Large, Reasoning) and examining how much the frontier degrades after **removing** a specific family. The "indispensability" of the family is measured by the relative improvement:

$$\frac{G_{p\sim D}(M_g,\ M_T \setminus M_g)}{V_{p\sim D}(M_T \setminus M_g)},$$

where $M_T$ denotes all models. A higher ratio indicates that the frontier degrades significantly if the family is removed, meaning that family is critical to maintaining the current cost frontier. The ingenuity lies in the fact that directly looking at which family has the lowest cost-of-pass on a task might be misled by the family being "coincidentally cheapest," whereas the counterfactual asks "can others fill the gap if it were gone," more cleanly isolating the unique contribution of each family.

### A Complete Example: Applying the Framework to AIME-24 Mathematics Problems
Taking "complex quantitative reasoning" as an example of the 5-step process: (1) For each model-problem pair, run multiple independent samplings to estimate the success rate $R_m(p)$; (2) Calculate the average token count per attempt × unit price + service fees to get the single cost $C_m(p)$; (3) Compute cost-of-pass $v(m,p)=C_m(p)/R_m(p)$; (4) Estimate human expert cost (AIME questions take ~12 minutes per rule, paired with a professional hourly wage) and take the minimum with LMs for the frontier; (5) Aggregate and accumulate over release dates for the dataset. The result: Reasoning models (e.g., o1, o3-mini, DeepSeek-R1), though much more expensive per-token than lightweight and large models, achieve the lowest cost-of-pass on AIME-24 (o3-mini only $2.03, while Llama-3.1-8B is $15.33) because they pull the success rate $R_m(p)$ high enough. During the same period, the frontier cost for MATH-500 halved every ~2.6 months, and AIME-24 every ~7.1 months. This example demonstrates how the same metric automatically determines who saves money between "high unit price but high success rate" and "low unit price but low success rate."

## Key Experimental Results

The experiment uses three categories of models (Lightweight / Large / Reasoning, 3–4 models each, released late 2024 to early 2025) across three categories of tasks (Basic Quantitative: two-digit addition, GSM8K; Knowledge-Intensive: GPQA-Diamond, BBQ; Complex Quantitative: MATH-500, AIME-24).

### Main Results: Different Families Occupying Distinct Economic Niches
Table 1 provides the frontier dollar cost-of-pass $V_{p\sim D}(\{m\}\cup M_0)$ (in USD, lower is better) under each model plus human baseline:

| Task Category | Dataset | Most Cost-Effective Family | Representative Value |
|---|---|---|---|
| Basic Quantitative | Two-digit addition | Lightweight | Llama-3.1-8B: $4.8e−5$ |
| Knowledge Intensive | GPQA-Diamond | Large / Reasoning | o1: $8.07$, o3-mini: $8.18$ |
| Complex Quantitative | AIME-24 | Reasoning | o3-mini: $2.03$, o1: $2.85$ |
| Complex Quantitative | MATH-500 | Reasoning | DeepSeek-R1: $0.21$ |

The Key Finding is that **different families occupy different economic niches**: for basic quantitative tasks where all models have high accuracy, the cheapest lightweight models are most cost-effective; for knowledge-intensive tasks, large models win due to their knowledge reserves; for complex quantitative tasks, reasoning models achieve the lowest cost-of-pass despite having the most expensive unit prices due to their success rate advantage. If one looked only at accuracy or only at cost (Appendix Table 5/6), the metrics would lopsidedly favor reasoning or lightweight models, precisely demonstrating the necessity of the merged metric.

### Key Findings: Progress Tracking and Counterfactual Analysis
- **The cost frontier declines exponentially over time**: The decline is most steady in complex quantitative tasks, with MATH-500 halving approximately every 2.6 months and AIME-24 every 7.1 months; basic quantitative and knowledge tasks show a sharp initial drop when early models arrive, followed by a plateau.
- **Counterfactual Family Importance** (Figure 3, % relative degradation after removing a family): Lightweight models are most critical for basic quantitative tasks (93.5% degradation for two-digit addition if removed), large models are indispensable only for knowledge-intensive tasks, and reasoning models dominate complex quantitative reasoning (81.0% degradation for AIME-24, 74.4% for MATH-500). The conclusion is that the current cost-efficiency frontier is primarily driven by the **lightweight** and **reasoning** extremes.

### The Economics of Inference-Time Techniques (Most Counter-intuitive Conclusion)
Table 2 shows the Gain (%) of various inference-time techniques relative to the original frontier:

| Technique | Basic Quantitative | Knowledge Intensive (GPQA) | Complex Quantitative |
|---|---|---|---|
| TALE-EP (Budget-aware) | 1.5 / 66.6 | 24.5 / 50 | 0.2 / 16.6 |
| Self-Refine | 0 / 0 | 6.7 / 24.9 | 0 / 0 |
| Majority Vote k=3 | All 0 | All 0 | All 0 |
| Majority Vote k=4 | All 0 | All 0 | All 0 |

Key Finding: **Majority voting provides almost no economic gain** (it may improve accuracy, but it multiplies the cost several times); Self-Refine provides a considerable gain of 24.9% only on GPQA-Diamond; only the "budget-aware" TALE-EP (which adapts generation to a predicted token budget) shows visible but uneven gains across multiple tasks. Overall, marginal accuracy improvements gained by piling on inference-time computation are mostly not worth the cost on the "price per correct answer" scale—true declines in the cost frontier almost entirely depend on the iteration of the models themselves.

## Highlights & Insights
- **Unifying two disparate aspects with a single economic formula**: The simple equation $v=C/R$ anchors "accuracy" and "spending" together, with a boundary behavior where $R=0$ signifies infeasibility, providing more interpretability than any weighted combination.
- **Integrating human experts into the same frontier table** provides, for the first time, a direct monetary benchmark to determine whether to "use AI or hire humans," while naturally solving the degradation issue of an infinite frontier when no LM can solve the task.
- **The counterfactual frontier is a transferable analysis paradigm**: Instead of asking "who is best," it asks "who is irreplaceable," cleanly separating the unique contribution of each component/family. This approach can be applied to model ensembles, data subsets, or feature importance analysis.
- **The value of the "majority voting is not cost-effective" conclusion** lies in providing a quantitative counter-example—many techniques that climb leaderboards in accuracy are exposed as economically inefficient, offering direct guidance for deployment selection.

## Limitations & Future Work
- **Dependency on accurate estimates of cost and success rate**: Per-inference cost $C_m(p)$ varies with provider pricing, usage tiers, and third-party calls; token prices themselves are falling rapidly, making the frontier curves sensitive to pricing snapshots. The success rate $R_m(p)$ is estimated by finite sampling, leading to high variance on long-tail difficult problems.
- **Human expert costs are estimates rather than measurements**: The authors assume qualified experts are nearly always correct ($R_{\text{expert}}\approx 1$) and infer labor costs via "evidence hierarchies" of wages/time, which can vary significantly between sources (author discusses relaxing these assumptions in Appendix D.3).
- **Limited coverage of tasks and time-stamped models**: The conclusions (which model family is cost-effective for which task) are tied to the selected benchmarks and the 2024–2025 window; the conclusions might shift with different task distributions or time periods. Decay periods for different tasks are not directly comparable.
- **Exclusion of latent costs**: Latency, reliability, safety/compliance, and maintenance are not reflected in pure monetary token costs. A pure economic metric might underestimate the true cost in certain scenarios.

## Related Work & Insights
- **vs. Fixed-Budget / Heuristic Cost Evaluation** (Wang et al. 2024; McDonald et al. 2024; Nayab et al. 2024 using conciseness): These link conclusions to specific constraints or proxies, lacking universality and economic interpretation. Ours uses real dollar costs + success probability for a single, interpretable monetary value without pre-set budgets.
- **vs. Research advocating for real cost and stochasticity** (Kapoor et al. 2024): Ours aligns with their core advocacy (focusing on real dollar cost, considering stochasticity) but formalizes it into a complete framework rooted in production theory (cost-of-pass + frontier + counterfactual gain) rather than just a call to action.
- **vs. Economic impact studies of AI** (Eloundou et al. 2024; Brynjolfsson et al. 2025, etc.): While those works discuss macro impacts on productivity and labor, the cost-of-pass in this paper serves as a bridge between technical performance and economic consequences, reducing the "economic contribution of a specific AI system" to a computable per-question dollar cost.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Cleanly migrates mature production frontier theory to LM evaluation; cost-of-pass + counterfactual frontier is a rare and powerful perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers three task types, ten models, time series, and multiple inference-time techniques; the task scope and time window are somewhat limited.
- Writing Quality: ⭐⭐⭐⭐⭐ Concepts progress logically, with clear alignment between formulas and economic motivations. Charts and tables effectively convey the findings.
- Value: ⭐⭐⭐⭐⭐ Provides a quantifiable monetary scale to determine if AI is cost-effective compared to human baselines, with direct practical value for model selection and progress measurement.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Don't Pass@k: A Bayesian Framework for Large Language Model Evaluation](dont_passk_a_bayesian_framework_for_large_language_model_evaluation.md)
- [\[ICLR 2026\] Evaluating Language Models' Evaluations of Games](evaluating_language_models_evaluations_of_games.md)
- [\[ICLR 2026\] CLASH: Evaluating Language Models on Judging High-Stakes Dilemmas from Multiple Perspectives](clash_evaluating_language_models_on_judging_high-stakes_dilemmas_from_multiple_p.md)
- [\[ICLR 2026\] RefineBench: Evaluating Refinement Capability of Language Models via Checklists](refinebench_evaluating_refinement_capability_of_language_models_via_checklists.md)
- [\[ICLR 2026\] PACEbench: A Framework for Evaluating Practical AI Cyber-Exploitation Capabilities](pacebench_a_framework_for_evaluating_practical_ai_cyber-exploitation_capabilitie.md)

</div>

<!-- RELATED:END -->

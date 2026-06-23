---
title: >-
  [Paper Note] OpenEstimate: Evaluating LLMs on Reasoning Under Uncertainty with Real-World Data
description: >-
  [ICLR 2026][LLM Reasoning][Paper Note] OpenEstimate is a benchmark that tasks frontier LLMs with "estimating probability distributions from internal knowledge" using real-world data. By randomly slicing public observational datasets to generate 178 **derived conditional statistics** as ground truths, the benchmark requires models to express their beliefs as
tags:
  - ICLR 2026
  - LLM Reasoning
date: 2026-05-08
content_hash: d0d10aecc952dd5f
---
# OpenEstimate: Evaluating LLMs on Reasoning Under Uncertainty with Real-World Data

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=sAzUQkP47r](https://openreview.net/forum?id=sAzUQkP47r)  
**Code**: https://github.com/alanarenda/openestimate  
**Area**: LLM Reasoning / Uncertainty Reasoning / Benchmarking  
**Keywords**: Bayesian Prior, Probability Estimation, Calibration, Uncertainty Reasoning, Benchmark

## TL;DR
OpenEstimate is a benchmark that tasks frontier LLMs with "estimating probability distributions from internal knowledge" using real-world data. By randomly slicing public observational datasets to generate 178 **derived conditional statistics** as ground truths, the benchmark requires models to express their beliefs as Bayesian priors. Results show that priors from six frontier models are worth approximately "5 samples from the true distribution," and their confidence is largely uncorrelated with accuracy.

## Background & Motivation

**Background**: Most current benchmarks for evaluating LLM reasoning assume that "problems have clear answers and information is complete and unambiguous"—such as math problems, code generation, and knowledge Q&A.

**Limitations of Prior Work**: Conversely, the actual scenarios where models are deployed (medicine, finance, public policy) typically involve open-ended problems with **incomplete information where decisions must be made under uncertainty**. For instance, when an analyst evaluates the Total Addressable Market for an early-stage investment, they should provide a **probability distribution**—containing both a central estimate and their degree of certainty—rather than a point estimate. The capability of models to generate such "reliable Bayesian priors" has hardly been characterized.

**Key Challenge**: Designing such an evaluation faces two conflicting constraints. First, it must be **grounded**—questions must force models to leverage background knowledge from pre-training to form high-quality priors. Second, it must **avoid data leakage**—if the ground truth answer itself is present in the training corpora, the test measures memorization rather than reasoning. Since most human knowledge is already ingested during pre-training, creating new questions where the "answer is known but not in the corpora" usually requires expensive new experiments or risks leakage.

**Goal**: To construct a probability estimation benchmark that is rooted in real-world data with verifiable ground truths, yet unlikely to be memorized in its original form within pre-training corpora, while quantifying the accuracy and calibration of frontier model priors.

**Key Insight**: The authors observe that "derived conditional statistics"—obtained by filtering large-scale public observational datasets based on **randomly sampled conditions** and aggregating target attributes—naturally satisfy these constraints. They provide empirically verifiable ground truth, but because the combinations of conditions are randomly drawn, they are unlikely to appear as pre-existing facts in the corpora.

**Core Idea**: By asking models for Bayesian prior distributions for statistics of random data slices, the benchmark achieves both grounding and leakage resistance. Prior quality is quantified as being "equivalent to $N$ real samples" using metrics like error ratio, win rate, and CRPS.

## Method

OpenEstimate is essentially a three-part benchmark protocol consisting of **dataset construction + belief elicitation + evaluation metrics**, rather than a new model. The pipeline moves from public observational datasets (Glassdoor / Pitchbook / NHANES) to verifiable ground truth statistics generated via random conditional filtering. These are described in natural language to the model, which must select a parametric distribution as a Bayesian prior. This prior (plus its posterior when combined with a few real samples) is compared against a statistical baseline of "uninformative prior + $N$ real samples" to measure accuracy and calibration. Since this is a benchmark task without a multi-module pipeline, no architecture diagram is provided; it is explained via text and formulas.

### Overall Architecture

The input consists of public tabular datasets from three domains, and the output is 178 probability estimation problems with ground truths and a complete set of evaluation metrics. The process involves three steps: **Derived Conditional Statistic Generation** (slicing datasets into questions unlikely to be memorized), **Bayesian Prior Elicitation** (defining how models express beliefs as distributions), and **Dual-Dimension Evaluation** (accuracy and calibration scored relative to a "few-shot statistical baseline").

### Key Designs

**1. Derived Conditional Statistics: Random Slicing for Grounding and Leakage Resistance**

This step addresses the core conflict of forcing models to use knowledge while preventing leakage. The authors construct two types of statistics from each dataset: **Marginal Statistics** (calculated using the full table, e.g., "average weight of US adults") and **Conditional Statistics** (calculated on subsets filtered by up to three additional conditions, e.g., "average weight of adults who have diabetes, take antidepressants, and have cholesterol above a certain threshold"). Conditions are **randomly sampled from values empirically observed** in the dataset. Following Algorithm 1: for $k \in \{0, 1, 2, 3\}$ conditions, $k$ distinct attributes $a_k$ are randomly picked to filter $D'$. If the sample size $|D'| \ge n$, the mean $\mu^*$ and standard error $se^*$ are estimated on $D'$.

The key filtering criterion: a conditional statistic is only accepted if its deviation from the marginal mean $\mu_0$ is sufficiently large, requiring $|\mu^* - \mu_0| > \tau$ and $|\mu^* - \mu_0| > se^*$ (following Xia et al. 2024, with $\tau \approx 5\%$). This ensures statistics reflect **meaningful differences** between subpopulations rather than sampling noise. These random combinations (e.g., "average funding for non-US, non-tech companies with >10 employees") are unlikely to exist as facts in the corpora, but a model with strong domain knowledge can infer them—testing reasoning over memorization. The final set includes 178 statistics (Glassdoor 43, Pitchbook 61, NHANES 74), extensible to new datasets.

**2. Bayesian Prior Elicitation: Forcing Simultaneous Reporting of Estimate and Certainty**

Asking only for point estimates limits evaluation to first-order accuracy and fails to measure if a model's **uncertainty** is appropriate. OpenEstimate requires models to express their belief as a **complete probability distribution** by choosing a distribution family and parameters. Models consistently chose one of three forms:

$$X \sim \mathcal{N}(\mu, \sigma^2),\quad X \sim \mathrm{Beta}(\alpha, \beta),\quad X \sim \mathrm{LogNormal}(\mu, \sigma^2)$$

The authors speculate these match three types of quantities: Gaussian for continuous symmetric values (e.g., salary), Beta for proportions (e.g., prevalence), and LogNormal for right-skewed values (e.g., startup valuations). This prior can be evaluated independently or combined with real samples to calculate a posterior for downstream inference.

**3. Accuracy + Calibration Metrics: Converting Prior Quality to "Sample Equivalents"**

To make "prior quality" concrete, the authors anchor it to a **few-shot statistical baseline**: starting from a naive flat prior ($\alpha=\beta=1$ for Beta; $\mu=0, \sigma^2=10^5$ for Gaussian), a posterior $\tilde p_i$ is calculated using $|\tilde D|=5$ random samples from the subpopulation. Regarding accuracy, the **error ratio** is calculated by dividing the LLM prior mean's Mean Absolute Error ($\mathrm{MAE}_{\text{LLM}}=\frac{1}{n}\sum_i|\mu_i^*-\mathrm{mean}(\hat p_i)|$) by the baseline MAE; a ratio $< 1$ indicates the model prior is more accurate than "5 noisy samples." The **win rate** is also reported: the percentage of questions where the model is closer to ground truth than the baseline. Furthermore, the LLM prior is combined with $N$ samples to calculate an LLM posterior $\hat{\tilde p}(\mu\mid\tilde D)\propto\hat p(\mu)\,p(\tilde D\mid\mu)$ to see if starting with the model's prior yields a better posterior than an uninformative one.

Calibration is measured using the **Continuous Ranked Probability Score** (CRPS), which penalizes both bias and over-dispersion without binning:

$$\mathrm{CRPS}(F, y) = \int_{-\infty}^{\infty}\big(F(x) - \mathbb{I}(x \ge y)\big)^2\, dx$$

where $F$ is the predicted cumulative distribution and $y$ is the ground truth. The **CRPS ratio** relative to the 5-sample baseline is reported.

### Loss & Training
Ours is a benchmark and does not involve training. Evaluation is entirely **zero-shot**: no fine-tuning, no RAG, and no prompt engineering beyond the direct request for parametric distributions, measuring the out-of-the-box probability estimation capabilities.

## Key Experimental Results

Six frontier models (including three reasoning models) were evaluated: Llama 3.1 70B, GPT-4o, o3-mini, o4-mini, and Qwen3-235B-A22B.

### Main Results: Win Rates for Priors and Posteriors

| Area | Sample size N | % Prior Better | % Posterior Better |
|------|------|------|------|
| Glassdoor | 5 | 37.0% | 71.4% |
| Glassdoor | 30 | 8.7% | 70.5% |
| Pitchbook | 5 | 50.8% | 69.6% |
| Pitchbook | 30 | 50.8% | 81.6% |
| NHANES | 5 | 74.3% | 70.4% |
| NHANES | 30 | 37.8% | 50.4% |

Using o4-mini as a reference: independent LLM priors beat the 5-sample baseline in ~40–70% of cases. However, as the baseline sample size increases, the prior win rate drops sharply (e.g., from 37% to 8.7% on Glassdoor). **Core Conclusion**: On its own, the prior is worth roughly "5 real samples"; however, even when the prior is inaccurate, it is highly useful when combined with data—LLM-informed posteriors consistently match or exceed statistical baselines using naive priors.

### Calibration: CRPS Ratio (vs. 5-sample baseline, lower is better)

| Model | Glassdoor | NHANES | Pitchbook |
|------|------|------|------|
| GPT-4o | 3.31 | 1.86 | 1.10 |
| Llama-3-70B | 4.56 | 2.76 | 1.13 |
| Qwen3-235B | 2.50 | 1.65 | 1.04 |
| o3-mini | 3.17 | 1.35 | 0.99 |
| o4-mini | 2.42 | 1.17 | 1.01 |

Reasoning models (o3-mini / o4-mini) generally show the best calibration. However, performance is highly **domain-dependent**: on Pitchbook, all models perform similarly to the baseline (~1.0), while on NHANES, smaller models collapse (the excluded Llama-3-8B was 20x worse than the baseline).

### Ablation Study

| Configuration | Influence | Description |
|------|------|------|
| Temperature / Reasoning Effort | No significant effect | Adjusting temperature in o4-mini and gpt-4o did not change quality. |
| System Prompt | No significant effect | Changing system prompts provided no substantial improvement. |
| Prior Elicitation Protocol | No significant effect | Alternative elicitation methods did not improve quality. |

### Key Findings
- **Priors are inaccurate but useful**: While the MAE of an LLM prior is often worse than 5 real samples, using it as a Bayesian starting point consistently improves downstream estimation.
- **Systematic Overestimation**: All model families tend to overestimate (the 1st quintile contains > 25% of cases). Pitchbook also shows heavy-tailed behavior (both over- and under-estimation).
- **Confidence $\neq$ Accuracy**: Self-reported uncertainty is generally weakly correlated with actual accuracy. Only on NHANES was uncertainty a decent indicator; it was not on Pitchbook or Glassdoor, suggesting models struggle to "know what they don't know."
- **Reasoning value emerges in difficult domains**: Model scale and reasoning capabilities are most critical in NHANES (medical), while even smaller models perform adequately on Pitchbook (finance).
- **No single dominant model family**: Rankings change by domain, and adjustments to reasoning settings do not fix calibration, suggesting new methods rather than hyperparameter tuning are needed.

## Highlights & Insights
- **Derived conditional statistics provide a sustainable, leakage-resistant benchmark**: Randomly slicing real datasets provides verifiable truths and resistance to memorization. Unlike forecasting benchmarks that expire as events unfold, this design remains challenging over time.
- **"Prior worth $N$ samples" is a compelling metric**: Translating abstract quality into a "sample equivalent" makes results comparable across units and domains while providing a striking intuition (frontier models $\approx$ 5 samples).
- **Decoupling prior and posterior evaluation**: Reveals the non-trivial phenomenon that a prior can be mediocre yet still provide positive value as a Bayesian starting point, suggesting LLMs should be used as prior generators rather than final answer providers.
- **Transferable Logic**: The idea of using random conditional aggregation on tabular data is transferable to any knowledge or reasoning evaluation requiring leakage resistance.

## Limitations & Future Work
- **Ground truth estimation error**: Truths are estimated from finite samples and may contain errors.
- **Leakage not entirely eliminated**: While systematic leakage is minimized, some may still occur for widely reported marginal values (e.g., national diabetes rates).
- **Limited coverage**: Currently only covers three datasets; more domains are needed for a comprehensive profile.
- **Zero-shot focus**: Does not evaluate RAG or fine-tuning interventions; the authors list "training for uncertainty awareness" as a future direction.
- **Observation on distribution families**: Since models choose from three specific families, if the ground truth is naturally multi-modal or fits a different distribution, a systematic mismatch error is introduced, making it hard to distinguish from model uncertainty misjudgment.

## Related Work & Insights
- **vs. Paruchuri et al. (2024) / Nafar et al. (2025)**: These treat probabilistic reasoning as math problems with complete inputs; Ours evaluates real-world estimation where information must be inferred and truth may be fuzzy.
- **vs. Xia et al. (2024) / Feng et al. (2024)**: These use structural constraints or Bayesian networks for discrete MCQ tasks; Ours directly evaluates the accuracy and calibration of continuous distributions.
- **vs. Selby et al. (2025)**: They use human experts or historical data as controls; Ours constructs cross-domain derived variables to evaluate against empirical ground truth and analyzes model/reasoning settings.
- **vs. Forecasting benchmarks (Karger et al. 2024)**: Forecasted events eventually enter the training data; Ours focuses on fine-grained tabular slices, which remain challenging by design.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The random slicing method and "sample equivalent" metric are clean, innovative ideas.
- Experimental Thoroughness: ⭐⭐⭐⭐ Solid coverage of models and metrics; however, the scale (178 questions) is relatively small.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear progression from motivation to discovery with honest conclusions.
- Value: ⭐⭐⭐⭐⭐ Addresses a critical capability for high-stakes LLM deployment (reliable probabilistic reasoning).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Neural Theorem Proving for Verification Conditions: A Real-World Benchmark](neural_theorem_proving_for_verification_conditions_a_real-world_benchmark.md)
- [\[ICLR 2026\] OptimalThinkingBench: Evaluating Over and Underthinking in LLMs](optimalthinkingbench_evaluating_over_and_underthinking_in_llms.md)
- [\[ICLR 2026\] From Assumptions to Actions: Turning LLM Reasoning into Uncertainty-Aware Planning](from_assumptions_to_actions_turning_llm_reasoning_into_uncertainty-aware_plannin.md)
- [\[NeurIPS 2025\] TimE: A Multi-level Benchmark for Temporal Reasoning of LLMs in Real-World Scenarios](../../NeurIPS2025/llm_reasoning/time_a_multilevel_benchmark_for_temporal_reasoning_of_llms_i.md)
- [\[ICLR 2026\] RFEval: Benchmarking Reasoning Faithfulness under Counterfactual Reasoning Intervention in Large Reasoning Models](rfeval_benchmarking_reasoning_faithfulness_under_counterfactual_reasoning_interv.md)

</div>

<!-- RELATED:END -->

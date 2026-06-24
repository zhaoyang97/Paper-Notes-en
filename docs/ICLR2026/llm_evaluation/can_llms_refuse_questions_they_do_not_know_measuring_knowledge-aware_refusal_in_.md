---
title: >-
  [Paper Note] Can LLMs Refuse Questions They Do Not Know? Measuring Knowledge-Aware Refusal in Factual Tasks
description: >-
  [ICLR 2026][LLM Evaluation][Refusal Index] This paper proposes the **Refusal Index (RI)**—a measure defined as the Spearman rank correlation between "refusal probability" and "error probability." Using a lightweight procedure that requires only two standard evaluation passes, it quantifies the LLM's capability to "actively refuse questions beyond its knowledge," a dimension overlooked by existing metrics.
tags:
  - "ICLR 2026"
  - "LLM Evaluation"
  - "Refusal Index"
  - "Knowledge-Aware Refusal"
  - "Calibration"
  - "Hallucination Evaluation"
  - "SimpleQA"
date: 2026-05-08
content_hash: c5b4cfdd1d32b469
---

# Can LLMs Refuse Questions They Do Not Know? Measuring Knowledge-Aware Refusal in Factual Tasks

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=9gJBhkLRat](https://openreview.net/forum?id=9gJBhkLRat)  
**Code**: To be confirmed  
**Area**: LLM Evaluation / Factual Reliability / Refusal Calibration  
**Keywords**: Refusal Index, Knowledge-Aware Refusal, Calibration, Hallucination Evaluation, SimpleQA  

## TL;DR
This paper proposes the **Refusal Index (RI)**—a measure defined as the Spearman rank correlation between "refusal probability" and "error probability." Using a lightweight procedure that requires only two standard evaluation passes, it quantifies the LLM's capability to "actively refuse questions beyond its knowledge," a dimension overlooked by existing metrics.

## Background & Motivation
- **Background**: LLMs are increasingly deployed in factual tasks but generally suffer from poor calibration—frequently providing incorrect answers with high confidence. One intuitive mitigation is to have models "refuse" questions outside their knowledge boundary (e.g., outputting "I do not have enough information..."). Recent work utilizes prompting or fine-tuning to enhance this refusal behavior.
- **Limitations of Prior Work**: Existing metrics fail to capture whether "refusals are correctly targeted." ① **Heuristic metrics based on refusal rates** (Refusal Rate, Correct given Attempted, SimpleQA’s F-score, Weighted Score) are essentially patched-up formulas that penalize over-refusal without characterizing the intrinsic correlation between refusal and errors. An F-score can fluctuate by up to 70% simply by prompting the model to be more cautious, even if its underlying capability remains unchanged. ② **Calibration-based metrics** (ECE, AUROC with verbalized confidence / auxiliary calibrators) rely on external confidence estimation or auxiliary models. The resulting uncertainty is not directly equivalent to the model's refusal probability, and conclusions often vary depending on the estimator selected.
- **Key Challenge**: Refusal behavior is a black box. Each instance typically yields only a single textual output, making the **refusal probability itself unobserved**. While the goal is to measure the ranking relationship (i.e., harder questions should be refused more frequently), absolute-difference metrics are biased by the overall refusal rate.
- **Goal**: Define a **faithful, consistent, and direct** metric to measure knowledge-aware refusal capability solely based on black-box refusal decisions, remaining invariant to refusal rate drift.
- **Core Idea (RI)**: **Use rank correlation instead of absolute error** to measure refusal quality. A superior model should exhibit a monotonically increasing refusal probability as question difficulty increases. Thus, RI is defined as the Spearman correlation between "refusal probability vs. error probability," using a **two-pass evaluation + Gaussian copula** to infer this latent correlation from observable binary indicators.

## Method

### Overall Architecture
At its conceptual level, RI is defined as: the Spearman rank correlation $\rho_S = \mathrm{Corr}(\mathrm{Rank}(r_i), \mathrm{Rank}(w_i))$ between refusal probability $r_i$ and error probability $w_i$. However, since $r_i$ and $w_i$ are latent variables and a single evaluation only provides binary observations for "refusal $R_i$" and "error $W_i$," the method utilizes an estimation pipeline. First, the pair $(R, W)$ is modeled as the thresholded result of two correlated Gaussian latent variables (tetrachoric setup), utilizing a Gaussian copula to represent the joint distribution via a single correlation parameter $\rho$. A **two-pass evaluation** is designed to collect statistics for estimating $\rho$. Finally, $\rho$ is solved via maximum likelihood and converted into the interpretable $\rho_S$.

```mermaid
flowchart LR
    A[Dataset D] --> B[First-pass Evaluation<br/>Refusal allowed<br/>Obtain R_i: Refusal indicator]
    B --> C[Second-pass Evaluation<br/>Only for refused questions<br/>Forced answer to obtain W'_i]
    C --> D[Aggregate Accuracy Indicators<br/>Ŵ_i = R_i·W'_i + (1-R_i)·W_i]
    D --> E[2x2 counts n_ab + Gaussian copula<br/>MLE to solve for ρ̂]
    E --> F[Convert ρ_S = 6/π·asin ρ̂/2<br/>= Refusal Index]
```

### Key Designs

**1. Defining refusal quality as rank correlation rather than absolute difference: The essence of RI.** Ours deliberately distinguishes RI from calibration metrics like ECE that "measure the absolute difference between $r_i$ and $w_i$." Instead, it focuses on **ranking consistency**: an ideal model satisfies $w_i \le w_j \iff P(f_{LM}(x_i)=\bot) \le P(f_{LM}(x_j)=\bot)$, meaning questions more likely to be wrong are more likely to be refused. This definition provides a critical advantage: absolute-difference metrics are extremely sensitive to global changes in refusal rates (a model can easily be adjusted to refuse more across the board without improving its judgment of difficulty), whereas discriminative rank correlation only compares relative order between samples, making it naturally robust to refusal rate drift. RI is thus positioned as a "discriminative" measure.

**2. Gaussian copula + Tetrachoric correlation to make latent correlation estimable.** Directly calculating $\rho_S$ requires continuous refusal probabilities for every instance, which are unobservable in standard evaluation. Ours employs a Gaussian copula $C(u,v)=\Phi_\rho(\Phi^{-1}(u),\Phi^{-1}(v))$ to model the joint distribution of $(r_i, w_i)$, where the copula carries the dependency structure $\rho$ without making assumptions about marginal distributions. Rather than estimating marginal CDFs $F_r, F_e$, $(R, \hat W)$ are viewed as thresholded results of a standard bivariate normal $(Z_R, Z_W)$ with correlation $\rho$ at thresholds $\tau_R=\Phi^{-1}(1-r)$ and $\tau_W=\Phi^{-1}(1-\mu)$ (where $r$ is the empirical refusal rate and $\mu$ is the error rate). The joint probabilities for the 2x2 table are given by $p_{11}(\rho)=\bar\Phi_2(\tau_R, \tau_W; \rho)$, with other cells derived via marginal constraints. Finally, maximizing the multinomial log-likelihood $\ell(\rho)=\sum_{a,b}n_{ab}\log p_{ab}(\rho)$ yields $\hat\rho$, which is converted to Spearman correlation via $\rho_S=\frac{6}{\pi}\arcsin\frac{\hat\rho}{2}$. This inference is entirely black-box, relaying only on the 2x2 counts $n_{ab}$.

**3. Two-pass evaluation: "Generating" two binary observations from a single textual output.** Estimating $\rho$ requires simultaneous observation of refusal indicator $R_i$ and accuracy indicator $W_i$ for each question. However, when refusals are permitted, refused questions lack accuracy labels. Ours designs a two-pass process: the **first pass** utilizes a standard setting (system prompt "refuse if uncertain") to obtain $R_i$, categorized as correct/incorrect/refused. The **second pass** switches to a "must answer, no refusal" system prompt, **only for questions refused in the first pass**, to obtain their potential accuracy $W'_i$. The aggregated accuracy indicator $\hat W_i = R_i \cdot W'_i + (1-R_i) \cdot W_i$ (representing "whether it would be correct if the model answered") is then used to construct the full $(R, \hat W)$ counts. This avoids multiple sampling or auxiliary calibrator training, estimating the unobservable correlation through just two conventional evaluation passes.

## Key Experimental Results
The experiments cover **16 models** (Claude, GPT-4.1, Gemini, Qwen3, Llama, Mistral, GLM, DeepSeek, etc.) across **5 datasets** (SimpleQA, PreciseWikiQA, and three FaithEval subsets).

### Main Results: Stability of RI across refusal rates (SimpleQA)
Using 4 system prompts with varying refusal tendencies, the study compares the normalized difference $\Delta$ and coefficient of variation CV (lower is more stable) between "highest refusal" and "lowest refusal" runs:

| Metric | Average $\Delta$ | Average CV |
|---|---|---|
| Accuracy | −0.80 | 0.31 |
| Refusal Rate | +0.98 | 0.39 |
| C/A | +0.49 | 0.19 |
| F-score | −0.54 | 0.22 |
| Weighted | −0.48 | 0.47 |
| **RI** | **+0.07** | **0.09** |

RI exhibits approximately **70%** less fluctuation than heuristic metrics. Changing the refusal rate via prompts merely shifts the refusal probability distribution without altering the intrinsic correlation between refusals and errors.

### Calibration Consistency / Ranking Stability
- **Correlation with AUROC (P(Answering), 100 samples)**: RI reaches **85.8%**, significantly outperforming C/A (37.6%), F-score (−24.6%), and Weighted (−73.3%), while being much more computationally efficient.
- **Ranking Stability** (Kendall’s W↑ / Winner Entropy↓, after removing monotonic effects of accuracy and refusal rate): The W for F-score and Weighted plummeted from ~0.90 to ~0.10; **RI maintained a W of ~0.47–0.50**, indicating it captures an intrinsic capability not explained by accuracy or refusal rate.

### Key Findings
- **The Capability Gap Persists**: While LLMs exhibit high factual accuracy, their refusal behavior remains unreliable and fragile. Prompting models to be more cautious only improves C/A, while RI remains far below perfect. Even when systematic bias is eliminated (refusal rate = error rate), a significant gap from "perfect refusal" remains.
- **Model Family is the Strongest Predictor**: RI does not correlate strongly with parameter scale, accuracy, or refusal rate ($R^2=0.242$ for RI vs. Accuracy). Claude and Qwen (excluding Qwen-235B) consistently perform above the regression line, while Gemini, GPT-4.1, and GLM fall below it—suggesting that training data and pipelines determine refusal quality more than model size.
- **Sensitivity to Noisy Context**: Refusal capability degrades significantly when ground truth is absent in the context (FaithEval Inconsistency=0.24, Unanswerable=0.32), compared to PreciseWiki (0.48) or Counterfactual (0.56). This indicates that refusal relies heavily on training data or specific context cues.

## Highlights & Insights
- **Paradigm Shift in Metric Design**: Moving from "absolute difference/accuracy reward" to "rank correlation" solves two perennial issues: the sensitivity of calibration metrics to refusal rates and the susceptibility of heuristic metrics to prompt engineering. It also theoretically demonstrates that both ends of the iso-RI curve are fixed, characterizing only convexity.
- **Elegant Statistical Modeling**: Using tetrachoric correlation + Gaussian copula reduces the problem of "unobservable continuous probability correlation" to "MLE of 2x2 counts." It replaces expensive sampling or auxiliary calibrators with two standard evaluation passes, ensuring feasibility.
- **Revealing an Overlooked Reliability Dimension**: High accuracy $\neq$ refusal capability, and this gap cannot be resolved merely by balancing accuracy and refusal rates. This research completes a missing piece in "comprehensive factual evaluation."

## Limitations & Future Work
- RI equates "knowing" with "answering correctly," which may not apply to partial correctness or long-form generation. It currently follows the SimpleQA setup of atomic short answers; migrating to open generation would require redefining accuracy.
- The two-pass evaluation assumes that forced answering faithfully reveals whether the model "originally knew," yet the behavior distribution under forced conditions might differ from natural generation. Furthermore, the normality assumption of the Gaussian copula is a simplification, despite passing goodness-of-fit tests.
- Inconsistency / Unanswerable subsets lack ground truth, requiring a 1:1 mix of PreciseWikiQA and FaithEval to calculate RI, which restricts its coverage.
- While the paper provides RI as a "ruler," the strategy for **training** models to improve RI (rather than just measuring it) remains for future work.

## Related Work & Insights
- **Factual Evaluation**: Unlike SimpleQA, FActScore, or TruthfulQA which measure correctness against external sources, ours focuses on measuring the calibration of refusal behavior rather than hallucination rates directly.
- **Black-box Calibration**: Approaches like verbalized confidence, auxiliary calibrators, or P(Answering) via sampling were tested. Ours demonstrates that these estimators are often inconsistent, yet sampling-based methods are the only ones that expose the overconfidence captured by RI.
- **Mechanism**: For any system requiring an "answer if known, stop if unknown" policy (e.g., RAG, Agents), RI provides a reliability metric that is resistant to gaming via refusal rates and can be calculated via a black box. The "two-pass evaluation + copula inference" logic can be transferred to other evaluation scenarios where binary observations are used to estimate latent continuous correlations.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Defining knowledge-aware refusal as the rank correlation between refusal and error probabilities and implementing it via copula + two-pass evaluation is a clean and original metric definition.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ 16 models × 5 datasets. The study systematically validates stability, calibration consistency, ranking stability, and three hallucination scenarios. It is focused on evaluation analysis, lacking training experiments for improving RI.
- **Writing Quality**: ⭐⭐⭐⭐ The logic from motivation to definition, estimation, and verification is clear. The three core properties (faithful/stable/direct) are well-integrated. The copula inference section might have a slight barrier for non-statistical readers.
- **Value**: ⭐⭐⭐⭐⭐ Fills a long-neglected gap in factual evaluation regarding "refusal reliability." The metric is lightweight and reusable, holding direct practical value for the trustworthy deployment of LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Are LLMs Really Not Knowledgeable? Mining the Submerged Knowledge in LLMs' Memory](are_llms_really_not_knowledgeable_mining_the_submerged_knowledge_in_llms_memory.md)
- [\[ICLR 2026\] Do LLM Agents Know How to Ground, Recover, and Assess? Evaluating Epistemic Competence in Information-Seeking Agents](do_llm_agents_know_how_to_ground_recover_and_assess_evaluating_epistemic_compete.md)
- [\[ICLR 2026\] Harnessing Temporal Databases for Systematic Evaluation of Factual Time-Sensitive Question-Answering in LLMs](harnessing_temporal_databases_for_systematic_evaluation_of_factual_time-sensitiv.md)
- [\[ICLR 2026\] Measuring LLM Novelty as the Frontier of Original and High-Quality Output](measuring_llm_novelty_as_the_frontier_of_original_and_high-quality_output.md)
- [\[ICLR 2026\] Rethinking LLM Evaluation: Can We Evaluate LLMs with 200× Less Data?](rethinking_llm_evaluation_can_we_evaluate_llms_with_200_less_data.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Measuring Weak-to-Strong Legibility of Reasoning Models
description: >-
  [ICML 2026][LLM Reasoning][weak-to-strong legibility] This paper proposes **Transfer Utility (TU)**—a metric for "weak-to-strong legibility" defined as the ability of a weak student model to complete a correct answer whe…
tags:
  - "ICML 2026"
  - "LLM Reasoning"
  - "weak-to-strong legibility"
  - "transfer utility"
  - "reasoning trace"
  - "weak supervision"
  - "reasoning legibility"
date: 2026-05-08
content_hash: af8b1fc356e07245
---

# Measuring Weak-to-Strong Legibility of Reasoning Models

**Conference**: ICML 2026  
**arXiv**: [2603.20508](https://arxiv.org/abs/2603.20508)  
**Code**: Not public (Authors committed to code release)  
**Area**: LLM Reasoning / Scalable Oversight / Multi-agent Collaboration  
**Keywords**: weak-to-strong legibility, transfer utility, reasoning trace, weak supervision, reasoning legibility

## TL;DR
This paper proposes **Transfer Utility (TU)**—a metric for "weak-to-strong legibility" defined as the ability of a weak student model to complete a correct answer when fed percentile-based prefixes of reasoning traces from a strong Reasoning Large Language Model (RLM). Across 12 open-source RLMs, 3 datasets, and 85k traces, it was discovered that **the most accurate and concise RLMs (e.g., GPT-OSS-120B) actually rank lowest in TU**, suggesting that RLVR training transforms reasoning traces into artifacts useful only to strong models.

## Background & Motivation

**Background**: Reasoning models (RLMs) like GPT-OSS, DeepSeek-R1, and Kimi-K2-Thinking are primarily trained using RLVR (Reinforcement Learning with Verifiable Rewards). Since reward signals only verify the final answer, intermediate reasoning traces are treated as "byproducts"—optimized to be as short as possible or even discarded.

**Limitations of Prior Work**: in scenarios such as scalable oversight, model distillation, agent collaboration, and trace caching, weak models must "understand" traces from strong models. However, existing legibility metrics (e.g., prover-verifier from Kirchner et al. 2024, coherence from Samineni et al. 2025, and various token/step-based efficiency metrics) favor "conciseness" and fail to characterize "thoroughness." A trace merely stating "The answer is 5" is perfectly efficient but lacks utility.

**Key Challenge**: Legibility requires a balance between **conciseness** and **thoroughness**. Conciseness reduces token costs and cognitive load, while thoroughness ensures key reasoning steps are preserved for a weak model to proceed. RLVR optimization rewards only correctness, leading to a trade-off between efficiency and transfer—a dimension invisible to current metrics.

**Goal**: (i) Provide a legibility metric independent of strong LLM judges or human labels; (ii) systematically characterize the weak-to-strong performance of 12 mainstream RLMs; (iii) verify if the metric predicts practical downstream monitorability.

**Key Insight**: legibility is essentially "whether a weak model can continue the reasoning of a strong model." The authors operationalize this by taking problems solved correctly by an RLM, truncating the trace to the first $x\%$, feeding it to a weak model (Phi-3-Mini 3.8B / Llama-3.2-1B), and measuring its success in generating the correct answer.

**Core Idea**: The "accuracy curve of the weak student completing the trace prefix $\mu_T(x)$" serves as a measurable proxy for legibility. Three scalar metrics (FOTU / SOTU / Regression Rate) are derived from this curve to quantify the "friendliness" of traces toward weak models.

## Method

### Overall Architecture
The pipeline consists of three steps:

1.  **Trace Generation**: 12 RLMs solve problems from MATH, GPQA, and LSAT datasets. Reasoning traces are extracted from think tokens and segmented into steps using NLTK, totaling 84,396 traces.
2.  **Efficiency Calculation**: Metrics including token/step counts, sentence embedding redundancy (threshold $\tau=0.8$), and backtracking frequency (labeled by Gemini 2.5-Flash) are computed for each trace.
3.  **Transfer Calculation (Core)**: For correctly solved problems, trace prefixes are fed to a weak student $W$ using a grid of $X=\{2,4,\ldots,100\}\%$. The student generates up to 1024 tokens to provide an answer. Correctness $S(R_p^{(x)})\in\{0,1\}$ is recorded per bin. Trace-level curves $f_p(x)$ are completed via linear interpolation, and the teacher-level curve $\mu_T(x)$ is the weighted mean across traces and students.

### Key Designs

1.  **Transfer Utility Curve $\mu_T(x)$ and FOTU**:
    - **Function**: Maps "prefix percentage $\to$ weak model completion accuracy" into a curve aggregated at the teacher level.
    - **Mechanism**: For each correct trace $R_p$ from teacher $T$, prefixes are sampled every 3 steps and assigned to the nearest right-boundary bin. The teacher curve is $\mu_T(x)=\frac{1}{|\mathcal{S}|}\sum_{W\in\mathcal{S}}\frac{1}{|P_x|}\sum_{p\in P_x}f_p^{(W)}(x)$. First-Order Transfer Utility (FOTU) is the mean across all bins: $\text{FOTU}(T)=\frac{1}{|X_T|}\sum_{x\in X_T}\mu_T(x)$. High FOTU implies the weak student can succeed from most prefix positions.
    - **Design Motivation**: Avoids adversarial prover-verifier setups and circular dependencies on "AI judges" by using fixed weak model success rates as a proxy. Percentile bins ensure comparability across different trace lengths.

2.  **Second-Order TU (SOTU) as an "Information Density Regularizer"**:
    - **Function**: Prevents traces from receiving inflated FOTU scores by front-loading answers in the first step or "sandbagging" them until the end.
    - **Mechanism**: For each trace and student $W$, let $\tau_p^{(W)}=\min\{x:f_p^{(W)}(x)=1\}$ be the first bin where the student succeeds. SOTU is defined as the normalized entropy of this empirical distribution $h_{T,W}(x)$: $\text{SOTU}(T)=\frac{1}{|\mathcal{S}|}\sum_W \frac{-\sum_x h_{T,W}(x)\log h_{T,W}(x)}{\log|X|}\in[0,1]$. Higher entropy indicates information is spread evenly.
    - **Design Motivation**: FOTU alone can be "gamed" by early answer exposure. SOTU penalizes such patterns, requiring useful information to be distributed. FOTU and SOTU are negatively correlated ($\rho=-0.50$), serving as orthogonal dimensions.

3.  **Regression Rate (RR) for "Confusional Steps"**:
    - **Function**: Captures instances where adding more steps causes the weak model to fail.
    - **Mechanism**: Treats the accuracy sequence $y_1,\ldots,y_K$ across prefixes as a time series, calculating the proportion of drops: $\text{rr}(p,W)=\frac{1}{K-1}\sum_{i=1}^{K-1}\mathbb{1}[y_{i+1}<y_i]$. RR focuses on whether steps are consistently constructive.
    - **Design Motivation**: While FOTU/SOTU are global position properties, RR characterizes step-to-step coherence, complementing backtracking (a trace feature) with its downstream effect on the student.

### Loss & Training
This is an **evaluation framework** and does not involve model training. The redundancy threshold $\tau=0.8$ was determined via manual sweep. Backtracking was labeled by Gemini 2.5-Flash with manual verification. Weak students were fixed as Phi-3-Mini and Llama-3.2-1B, with OLMo-3-7B-Instruct used for stability checks.

## Key Experimental Results

### Main Results: Legibility Ranking of 12 RLMs across 6 Dimensions

| Model (Acc%) | Token Len | Redund | Backtrack | FOTU | SOTU |
|---|---|---|---|---|---|
| GPT-OSS-120B (81) | 4 | **1** | 4 | 10 | 5 |
| GPT-OSS-20B (76) | 5 | 2 | 7 | 8 | 4 |
| Kimi-K2-Thinking (70) | 10 | 11 | 8 | **1** | 8 |
| DeepSeek-R1 (70) | 11 | 8 | 6 | 3 | 10 |
| DeepSeek-R1-Distill-32B (68) | 6 | 8 | 10 | 11 | 6 |
| Qwen3-8B (57) | 7 | 10 | 10 | 6 | 9 |
| Magistral-S (62) | 2 | 4 | 3 | 12 | 3 |
| Gemma-3-27B-it (53) | 2 | 5 | 2 | 8 | **1** |
| OpenReasoning-32B (58) | 12 | 7 | 5 | 2 | 12 |
| Gemma-3-12B-it (49) | **1** | 3 | **1** | 7 | 2 |

**Key Findings**: (i) GPT-OSS-120B ranks 1st in accuracy but 10th in FOTU—**the strongest model's traces are least helpful to weak students**; (ii) Gemma-3-12B-it ranks second to last in accuracy but 1st in efficiency; (iii) Kimi-K2, DeepSeek-R1, and QwQ-32B provide "long and verbose but student-friendly" traces (3.8k–4.8k tokens, redundancy 16–31%). The Spearman correlation between accuracy and FOTU is $\rho=-0.35$.

### Monitorability Validation: Can FOTU Predict Weak Supervision Performance?

| Dataset | Condition | $\rho$(FOTU, Precision) | $\rho$(FOTU, F1) | $\rho$(FOTU, F1\|Len) |
|---|---|---|---|---|
| MATH | Sabotage | **+0.900***  | +0.845** | +0.745** |
| MATH | Natural (shared failure) | **+0.873***  | +0.809** | +0.582† |
| LSAT | Sabotage | +0.818** | +0.664* | +0.482 |
| LSAT | Natural | +0.664* | +0.591† | +0.082 |
| GPQA | Sabotage | +0.770** | +0.064 | -0.173 |
| GPQA | Natural | -0.305 | -0.709* | -0.436 |

**Conclusions**: In "process-verifiable" tasks like MATH, FOTU strongly predicts the precision of Phi-3-Mini as a weak supervisor. Signals remain in LSAT precision but fail in recall. For GPQA (graduate-level science), the signal collapses after partialling for length because the factual knowledge exceeds the capability of 1B/3B models; legibility cannot bridge the domain gap.

### Key Findings
- **Accuracy-Efficiency-Transfer Trade-off**: Current RLMs choose one side of this triangle; none rank in the top 3 for all three categories. RLVR's single-objective optimization makes traces useful to strong models but unfriendly to weak ones.
- **Length is Not the Whole Story**: The rank correlation for FOTU vs. FOTU\|Length is +0.728, indicating FOTU captures structural content beyond mere volume.
- **Reward Models ignore Legibility**: Correlations between FOTU and three tested reward models (Skywork, Llama, Qwen-PRM) are near zero ($\rho \approx -0.08$) after controlling for accuracy, showing current RM objectives lack a legibility dimension.

## Highlights & Insights
- **"Weak Model Completion" as an Operational Definition**: This is a clever design—it replaces the subjective "legibility" (usually requiring humans or strong LLMs) with objective, reproducible downstream success rates. It shifts from the adversarial "prover-verifier" framework to a collaborative "utility" perspective.
- **SOTU for Entropy-based Regularization**: Using the entropy of the first-success positions effectively penalizes reward hacking (front-loading/sandbagging), ensuring helpful information is distributed throughout the trace.
- **Counter-intuitive Finding**: The realization that high-accuracy models produce the worst traces for teaching is a wake-up call for scalable oversight. As models optimize for benchmarks, their "pedagogical value" degrades—the very property weak supervisors rely on.

## Limitations & Future Work
- **Author-acknowledged Limitations**: (i) Static evaluation doesn't cover multi-turn agents; (ii) closed-source models (e.g., GPT-5, Claude-4.5) were excluded as they only provide trace summaries; (iii) no human evaluation; (iv) backtracking depends on an LLM judge.
- **Additional Observations**: (i) Weak students were limited to 1B–3.8B; (ii) only correct traces were analyzed, whereas oversight specifically cares about error traces; (iii) MATH's strong signals might stem from its formal nature; (iv) no specific implementation was provided for integrating TU into training objectives.
- **Future Improvements**: Use weak models as "cheap critics" during RL training to provide process-level auxiliary rewards, forcing models to optimize for transfer utility.

## Related Work & Insights
- **vs. Kirchner et al. 2024 (Prover-Verifier Games)**: While they focus on adversarial robustness during training, this work focuses on collaborative utility during evaluation without requiring weak model training.
- **vs. Samineni et al. 2025 (Coherence vs. Validity)**: They noted RLMs are locally consistent but globally flawed. This paper provides downstream quantitative evidence that such "locally smooth but globally unteachable" traces rank poorly in FOTU.
- **Insight**: This "weak student completion" paradigm can be applied to any trace quality assessment (code generation CoT, tool-use trajectories, agent planning) as long as a significantly weaker completion model exists for that modality.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Transfer Utility frames weak-to-strong generalization as an evaluation problem with the FOTU/SOTU/RR suite.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive cross-validation across models and datasets, though it lacks mid-sized (7B) students and error trace analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear arguments for transfer importance and well-defined metrics.
- Value: ⭐⭐⭐⭐⭐ Provides a critical assessment of scalable oversight and RLVR goals, establishing legibility as a "first-class property."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] The Illusion of Diminishing Returns: Measuring Long Horizon Execution in LLMs](../../ICLR2026/llm_reasoning/the_illusion_of_diminishing_returns_measuring_long_horizon_execution_in_llms.md)
- [\[ICLR 2026\] Is It Thinking or Cheating? Detecting Implicit Reward Hacking by Measuring Reasoning Effort](../../ICLR2026/llm_reasoning/is_it_thinking_or_cheating_detecting_implicit_reward_hacking_by_measuring_reason.md)
- [\[ICML 2026\] Are Large Reasoning Models Interruptible?](are_large_reasoning_models_interruptible.md)
- [\[ICML 2026\] Verifying Meta-Awareness via Predictive Rewards in Reasoning Models](verifying_meta-awareness_via_predictive_rewards_in_reasoning_models.md)
- [\[ICML 2026\] DecepChain: Inducing Deceptive Reasoning in Large Language Models](decepchain_inducing_deceptive_reasoning_in_large_language_models.md)

</div>

<!-- RELATED:END -->

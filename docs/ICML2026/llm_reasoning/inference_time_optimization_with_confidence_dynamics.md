---
title: >-
  [Paper Note] Inference Time Optimization with Confidence Dynamics
description: >-
  [ICML2026][LLM Reasoning][Confidence Dynamics] The authors discover that in LLM multi-sample inference, the confidence of correct trajectories systematically increases along the reasoning chain while incorrect ones decay…
tags:
  - "ICML2026"
  - "LLM Reasoning"
  - "Confidence Dynamics"
  - "Best-of-N"
  - "Voting Aggregation"
  - "GRPO"
  - "Inference-time Scaling"
date: 2026-05-08
content_hash: 7d4f7ff18910b32b
---

# Inference Time Optimization with Confidence Dynamics

**Conference**: ICML2026  
**arXiv**: [2605.25244](https://arxiv.org/abs/2605.25244)  
**Code**: https://github.com/Accenture/CDG.git  
**Area**: LLM Inference  
**Keywords**: Confidence Dynamics, Best-of-N, Voting Aggregation, GRPO, Inference-time Scaling

## TL;DR
The authors discover that in LLM multi-sample inference, the confidence of correct trajectories systematically increases along the reasoning chain while incorrect ones decay or stagnate. Based on this, they propose CDG (Confidence Dynamic Gain) voting—embedding "tail confidence − head confidence" as an additional discriminative signal into Best-of-N weighted voting. Across four open-source reasoning models and four math Olympiad benchmarks, it achieves an average improvement of 5.4% over majority voting and 1.7–4.8% over DeepConf.

## Background & Motivation

**Background**: The current mainstream approach to improving LLM reasoning accuracy is Best-of-N sampling—sampling $L$ reasoning traces for the same problem and selecting the final answer via an aggregation function. The simplest method, Self-Consistency, uses majority voting; recent works like DeepConf and Self-Certainty use model confidence (sequence-level perplexity, average top-K log-prob) as voting weights to further exploit sample utility.

**Limitations of Prior Work**: Existing confidence-based methods compress the confidence of a trace into a **single scalar**—either averaging all tokens (DeepConf-Mean) or focusing only on a fixed window of tail tokens (DeepConf-Tail). This static aggregation loses information regarding how confidence evolves during generation. A trace with high confidence in the last few tokens but low confidence in the middle is treated as equivalent to a trace that shows steady confidence growth.

**Key Challenge**: A reasoning trace is a time series, but existing voting methods treat it as an i.i.d. bag of tokens. Positional/dynamic information (whose impact has been verified in works on attention sinks and lost-in-the-middle) has been largely neglected in inference-time scaling.

**Goal**: (1) To describe and quantify whether confidence evolution along a reasoning trajectory distinguishes correct from incorrect answers; (2) To embed this dynamic signal into Best-of-N voting; (3) To provide a mechanistic explanation for why models trained with GRPO exhibit this phenomenon.

**Key Insight**: The authors conducted a simple experiment using four open-source reasoning LLMs (DeepSeek-R1-8B / gpt-oss-20B / Gemma-3-27B / QwQ-32B) on AIME 2025. By slicing each trace into 10 position-normalized bins and plotting the average confidence per bin, they found that **correct traces show curves significantly higher at the tail than the head, while incorrect traces are either flat or downward-sloping**, with statistical significance (Appendix Table 5).

**Core Idea**: Define "tail confidence - head confidence" as the Confidence Dynamic Gain $\Delta C_\ell$. This serves as an additional scoring term, linearly combined with the original mean confidence and passed into a count-dampened weighted vote.

## Method

### Overall Architecture
Input: $L$ traces sampled for a problem, where each trace contains the token sequence $y_{\ell,1:T}$ and top-K log-probs for each position. Output: The aggregated final answer $\hat{a}$. The process involves three steps: (1) Use top-K KL to approximate per-token confidence $C_{\ell,t}$; (2) Slice each trace into $N=10$ position-normalized bins to calculate the difference $\Delta C_\ell$ between the average confidence of the head and tail bins; (3) Use $\bar{C}_\ell + \beta\cdot\Delta C_\ell$ as the trace score, then calculate the answer score $R(a) = |\mathcal{T}_a|^\alpha \cdot \mu_a(s_\ell)$ and take the argmax. The method is **entirely training-free**, with the only overhead being the extraction of token log-probs from the inference stack.

### Key Designs

1.  **Per-token Confidence + Position-normalized Binning**:
    - **Function**: Maps traces of arbitrary lengths to a fixed $N$-dimensional confidence vector, facilitating head-to-tail comparisons across traces and problems.
    - **Mechanism**: Token-level confidence follows the top-K approximation $C_t = -\frac{1}{K}\sum_{j\in\mathcal{K}_t}\log p(y_t=j)$ (with $K=20$, essentially the KL between top-K log-probs and a uniform distribution). The $T$ tokens are divided into $N=10$ equal bins $\mathcal{B}_{\ell,n}$, and the mean $\bar{C}_\ell^{(n)}$ is calculated for each bin to obtain $(\bar{C}_\ell^{(1)},\ldots,\bar{C}_\ell^{(N)})$.
    - **Design Motivation**: Reasoning traces vary greatly in length (from hundreds to tens of thousands of tokens). Slicing by absolute position fails to align them; position-normalization allows for the observation of statistical patterns like "tail-end rising for correct groups" vs "tail-end falling for incorrect groups" (as shown in Figure 2).

2.  **Confidence Dynamic Gain $\Delta C_\ell$ as a Discriminative Signal**:
    - **Function**: Compresses the confidence evolution trend into a scalar; positive values indicate increasing confidence, while negative values indicate decreasing confidence.
    - **Mechanism**: Let $T_{\text{head},P}$ and $T_{\text{tail},P}$ be the sets of head and tail $P\%$ bins respectively. $\Delta C_\ell = \frac{1}{|T_{\text{tail},P}|}\sum_{n\in T_{\text{tail},P}}\bar{C}_\ell^{(n)} - \frac{1}{|T_{\text{head},P}|}\sum_{n\in T_{\text{head},P}}\bar{C}_\ell^{(n)}$ (default $P=10$). The trace score is $s_\ell = \bar{C}_\ell + \beta\cdot\Delta C_\ell$. The hyperparameter $\beta$ is selected based on a model-specific rule: $\beta\in[0.5 r_b, 1.5 r_b]$, where $r_b = \mu_C / \Delta_\mu$ is estimated using calibration problems.
    - **Design Motivation**: While DeepConf-Tail verifies the utility of "tail confidence," focusing solely on the tail confuses "easy problems the model is sure about from the start" with "hard problems where confidence is gained through reasoning." **Subtracting head confidence** acts as a baseline calibration—rewarding traces that "gain insight" during reasoning and penalizing those that "start with bluffing but lose track." Removing the head subtraction ("No Start") leads to a 4.4% drop in performance, proving the differential operation is critical.

3.  **Count-dampened Weighted Voting**:
    - **Function**: Aggregates trace scores $s_\ell$ into answer scores $R(a)$ to prevent majority signals from drowning out confidence signals.
    - **Mechanism**: $R(a) = |\mathcal{T}_a|^\alpha \cdot \mu_a(s_\ell)$, where $\mathcal{T}_a$ is the set of traces yielding answer $a$, and $\mu_a(s_\ell)$ is the mean score within that set. The final answer is $\hat{a} = \arg\max_a R(a)$. The exponent $\alpha\in[0,1]$ (default 0.5) dampens the count term. When $\alpha=1, \beta=0$, it reduces to DeepConf; when $\alpha=1, \mu_a(s_\ell)=1$, it reduces to majority voting.
    - **Design Motivation**: Ablations show that if $\alpha=1$, the frequency term dominates the small differences in mean confidence gains. Only $\alpha < 1$ allows the CDG signal to effectively influence the decision.

### Loss & Training
**None (Training-free)**. CDG operates during the inference stage of pre-trained models. Hyperparameters $\alpha=0.5, P=10\%$ are fixed; $\beta$ is calibrated per model (10 for DeepSeek-R1-8B/gpt-oss-20B; 3 for Gemma-3-27B/QwQ-32B) using cross-benchmark calibration.

## Key Experimental Results

### Main Results
Evaluation using four open-source reasoning LLMs across four math Olympiad benchmarks (AIME 2024 / AIME 2025 / BRUMO 2025 / HMMT 2025), sampling $L=512$ traces per problem.

| Model | Pass@1 | Majority | DC-Mean | DC-Tail | Ours (CDG) | vs Majority |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DeepSeek-R1-8B | 75.8 | 84.2 | 84.2 | 88.3 | **90.8** | +6.6 |
| Gemma-3-27B | 25.7 | 35.0 | 35.0 | 40.0 | **41.7** | +6.7 |
| gpt-oss-20B | 66.5 | 82.5 | 84.2 | 85.0 | **85.8** | +3.3 |
| QwQ-32B | 69.7 | 75.0 | 75.9 | 78.3 | **80.0** | +5.0 |
| **Average** | 59.4 | 69.2 | 69.8 | 72.9 | **74.6** | **+5.4** |

CDG achieves the highest scores across all models, improving by 4.8% over DeepConf-Mean and 1.7% over DeepConf-Tail. The gain is most pronounced on the challenging AIME 2025 (DeepSeek-R1-8B: 83.3 → 93.3, +10).

### Ablation Study

| Configuration | Overall Average (%) | Notes |
| :--- | :--- | :--- |
| Full CDG | 74.6 | $\alpha=0.5, \beta\in\{3,10\}$, full differential signal |
| D-CDG ($\beta=0$) | 70.0 | No dynamic signal, count dampening only |
| D-CDG ($\alpha=1$) | 72.9 | No count dampening, frequency dominates |
| "No Start" | 70.2 | No head subtraction, tail confidence only (-4.4) |

### Key Findings
- **The head-tail difference ($\Delta C_\ell$) is essential**: Removing the head baseline causes significant drops (e.g., -6.6 for DeepSeek-R1, -13.3 on HMMT), proving that "confidence climb" is more important than "absolute tail value."
- **Count dampening is essential**: Using $\alpha=1$ causes frequency to drown out confidence signals, resulting in a 1.7% drop.
- **Independence from \boxed{} tokens**: Experiments re-calculating CDG after removing \boxed{} answer tokens showed 99% consistency in answer selection, proving the signal comes from reasoning dynamics, not the format tokens.
- **Superior small-sample efficiency**: CDG consistently outperforms majority and DeepConf across budgets $L\in\{8, \dots, 256\}$, specifically showing steeper gains at low $L$.
- **Statistical mechanism verification**: For most (model, dataset) pairs, $\Delta C_\ell > 0$ for correct traces and $\Delta C_\ell < 0$ for incorrect ones (Figure 4d).

## Highlights & Insights
- **"Dynamic > Static" is an undervalued dimension**: Reasoning traces possess temporal structure. This work uses a simple subtraction to achieve a 5% gain, suggesting that trajectory dynamics (entropy gain, perplexity slope, etc.) is a promising research path.
- **Inference patterns derived from GRPO training dynamics**: The authors use GRPO’s group-normalized advantage combined with assumptions about answer concentration to derive why correct traces show tail-to-head logit gains while incorrect ones don't. This links empirical phenomena to training algorithms.
- **Elegant generalization of existing methods**: The formula $R(a) = |\mathcal{T}_a|^\alpha \cdot \mu_a(\bar{C}_\ell + \beta\Delta C_\ell)$ unifies majority voting, DeepConf-Mean, and CDG.
- **Reusable trick**: Position-normalized binning is applicable to any downstream task involving statistics along a generation process, such as early stopping or speculative decoding verification.

## Limitations & Future Work
- **Dependence on token-level log-probs**: Closed-source APIs (some OpenAI models, Anthropic) do not always provide full distributions, limiting deployment.
- **Model-dependency of $\beta$**: While a scaling rule is provided, it still requires calibration samples for new models.
- **Theoretic assumptions**: The theory relies on correct traces converging to ground truth and incorrect ones having lower concentration. Its applicability to open-ended generation (code, long-form writing) remains to be verified.
- **Benchmark scope**: Primary results are centered on math Olympiads. Generalization to GPQA, LiveCodeBench, or MMLU-Pro is left for future work.
- **Cost of Best-of-N**: $L=512$ is expensive. Hybrid schemes combining CDG with on-the-fly pruning could mitigate latency.

## Related Work & Insights
- **vs DeepConf-Mean / DeepConf-Tail (Fu et al. 2025)**: DeepConf uses scalar averages. CDG treats confidence as a time series for head-tail differentiation. 
- **vs Self-Certainty (Kang et al. 2025b)**: Self-Certainty uses KL divergence over the whole trace; CDG's differential signal is orthogonal and could theoretically be combined.
- **vs Process Reward Models (PRM)**: PRMs require training an external verifier. CDG uses the model's own logit sequences without extra training data, effectively serving as a "poor man's PRM."

## Rating
- Novelty: ⭐⭐⭐⭐ Systematic reporting of the confidence gain phenomenon is novel; the method is a precise and effective addition to prior frameworks.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 4 models, 4 datasets, and multiple ablations, though limited primarily to math.
- Writing Quality: ⭐⭐⭐⭐ Clear narrative flow from observation to theory and experiment.
- Value: ⭐⭐⭐⭐ Training-free and compatible with any reasoning LLM, though dependent on log-prob access and hyperparameter calibration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] UniScale: Adaptive Unified Inference Scaling via Online Joint Optimization of Model Routing and Test-time Scaling](uniscale_adaptive_unified_inference_scaling_via_online_joint_optimization_of_mod.md)
- [\[ICLR 2026\] Fixing the Broken Compass: Diagnosing and Improving Inference-Time Reward Modeling](../../ICLR2026/llm_reasoning/fixing_the_broken_compass_diagnosing_and_improving_inference-time_reward_modelin.md)
- [\[NeurIPS 2025\] Inference-Time Chain-of-Thought Pruning with Latent Informativeness Signals](../../NeurIPS2025/llm_reasoning/inference-time_chain-of-thought_pruning_with_latent_informativeness_signals.md)
- [\[ICML 2026\] Stabilizing Recurrent Dynamics for Test-Time Scalable Latent Reasoning in Looped Language Models](stabilizing_recurrent_dynamics_for_test-time_scalable_latent_reasoning_in_looped.md)
- [\[ICML 2026\] Diagnosing Multi-step Reasoning Failures in Black-box LLMs via Stepwise Confidence Attribution](diagnosing_multi-step_reasoning_failures_in_black-box_llms_via_stepwise_confiden.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICLR 2026\] Fixing the Broken Compass: Diagnosing and Improving Inference-Time Reward Modeling](../../ICLR2026/llm_reasoning/fixing_the_broken_compass_diagnosing_and_improving_inference-time_reward_modelin.md)
- [\[ICML 2026\] Stabilizing Recurrent Dynamics for Test-Time Scalable Latent Reasoning in Looped Language Models](stabilizing_recurrent_dynamics_for_test-time_scalable_latent_reasoning_in_looped.md)
- [\[NeurIPS 2025\] Inference-Time Chain-of-Thought Pruning with Latent Informativeness Signals](../../NeurIPS2025/llm_reasoning/inference-time_chain-of-thought_pruning_with_latent_informativeness_signals.md)
- [\[ACL 2026\] Calibration-Aware Policy Optimization for Reasoning LLMs](../../ACL2026/llm_reasoning/calibration-aware_policy_optimization_for_reasoning_llms.md)
- [\[ICML 2026\] Diagnosing Multi-step Reasoning Failures in Black-box LLMs via Stepwise Confidence Attribution](diagnosing_multi-step_reasoning_failures_in_black-box_llms_via_stepwise_confiden.md)

</div>

<!-- RELATED:END -->

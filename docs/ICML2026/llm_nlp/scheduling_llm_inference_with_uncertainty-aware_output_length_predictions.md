---
title: >-
  [Paper Note] Scheduling LLM Inference with Uncertainty-Aware Output Length Predictions
description: >-
  [ICML 2026][LLM/NLP][SJF Scheduling] This paper replaces the point estimation of "predicting a single output length" in LLM inference scheduling with log-t distribution fitting. It introduces Tail Inflated Expectation (T…
tags:
  - "ICML 2026"
  - "LLM/NLP"
  - "SJF Scheduling"
  - "Output Length Prediction"
  - "Heavy-Tailed Distribution"
  - "log-t Distribution"
  - "CVaR"
date: 2026-05-08
content_hash: d5a91ac173a87660
---

# Scheduling LLM Inference with Uncertainty-Aware Output Length Predictions

**Conference**: ICML 2026  
**arXiv**: [2604.00499](https://arxiv.org/abs/2604.00499)  
**Code**: https://github.com/Hyzheng-code/TIE  
**Area**: LLM Efficiency / Inference Scheduling  
**Keywords**: SJF Scheduling, Output Length Prediction, Heavy-Tailed Distribution, log-t Distribution, CVaR

## TL;DR
This paper replaces the point estimation of "predicting a single output length" in LLM inference scheduling with log-t distribution fitting. It introduces Tail Inflated Expectation (TIE)—an expectation supplemented with a CVaR tail penalty—to replace output length as the priority in SJF scheduling. On LMSYS-Chat-1M, TIE reduces online per-token latency by $2.31\times$ compared to the strongest baseline LTR and increases offline SDG throughput by $1.42\times$.

## Background & Motivation

**Background**: LLM serving systems like vLLM default to FCFS scheduling, where long requests can block short ones (HOL blocking). A main category of improvement follows the SJF (Shortest Job First) logic: using a lightweight predictor (SSJF / LTR / TRAIL / ELIS) to predict the output length for each prompt and then queuing them from shortest to longest.

**Limitations of Prior Work**: Prediction errors are significant. Chen et al. (2025b) reported that output length prediction errors are substantial. To counter these errors, methods like TRAIL / ELIS perform repeated predictions and preemption during generation (every token or every 50 tokens), but the overhead of prediction and preemption consumes a large portion of the scheduling gains.

**Key Challenge**: The authors argue that these methods overlook a fundamental issue—LLM decoding is inherently a stochastic process where a token is sampled from a probability distribution at each step, making the appearance of EOS a random variable. The same prompt run 100 times will produce 100 different lengths. **Using a point estimate to describe a distribution inevitably leads to large errors in the tail**: when an inherently long request is mispredicted as short, it blocks the entire batch. This is exacerbated by the fact that LLM output lengths naturally follow a heavy-tailed distribution (the top 10% longest requests account for 35.7% of the total length, with a P99/P50 ratio reaching up to 10.77).

**Goal**: (1) Find a suitable probability distribution family for output length instead of point estimates; (2) Convert distribution information into a scalar priority for direct use by SJF schedulers; (3) Implement this with low overhead in vLLM.

**Key Insight**: Starting from the LLM decoding process, the authors prove that output length follows a heavy-tailed distribution with a power-law tail. They identify the log-t (3-parameter) distribution as the best fit (93.1% KS test pass rate).

**Core Idea**: Fit a log-t distribution to the output length of each request and use $\mathbb{E}[X] + \beta \cdot \mathrm{CVaR}_\alpha[X]$ as the "equivalent length" for SJF, allowing the scheduler to explicitly penalize requests with high tail risk during sorting.

## Method

### Overall Architecture
The method consists of three parts: (1) **Distribution Fitting**—offline MLE fitting of log-t parameters $(\mu, \sigma)$ for each prompt to construct training labels; (2) **Parameter Predictor**—online prediction of $(\mu, \sigma)$ directly from prompts using a fine-tuned DeBERTa-v3-base with dual MLP heads, with $\nu$ fixed at 3.5; (3) **TIE Scheduler**—truncating the predicted distribution at `max_tokens` and calculating $\mathbb{E}[\tilde X] + \beta\cdot\mathrm{CVaR}_\alpha[\tilde X]$ as the priority score for vLLM’s min-priority queue, while hiding overhead through asynchronous prediction, dynamic batching, and waiting decay.

### Key Designs

1.  **log-t Distribution Fitting for Output Length (Replacing Point Estimates)**:
    - **Function**: Models the output length of each request as a 3-parameter distribution $X \sim \text{Log-t}(\mu,\sigma,\nu)$, with PDF $f(x\mid\mu,\sigma,\nu) = \frac{1}{\sigma x}\cdot t_\nu\left(\frac{\ln x-\mu}{\sigma}\right)$, where $\nu$ controls the tail thickness.
    - **Mechanism**: In Assumption 3.1 + Theorem 3.2, the authors prove that if the density of termination probabilities $p$ near 0 across different generation trajectories satisfies $f(p)\sim c\cdot p^{\alpha-1}$, then the tail probability of output length $L=\min\{t\ge 1: x_t=\text{EOS}\}$ satisfies $P(L>n)\sim c\cdot\Gamma(\alpha)/n^\alpha$ (power-law decay), a sufficient condition for heavy tails. Empirically, they sampled 100 responses for 1K prompts in LMSYS-Chat-1M, finding a mean skewness of 3.10 and a coefficient of variation of 1.09, confirming heavy-tailed properties. KS tests comparing six candidate distributions showed log-t (3-parameter) had a 93.1% pass rate, while log-t (2-parameter, fixed $\nu=3.5$) had 90.6%, and log-normal had only 60.3%. The 2-parameter log-t was chosen to balance fit quality and prediction complexity.
    - **Design Motivation**: To solve the fundamental contradiction between "point estimation vs. stochastic decoding." Point estimates cannot characterize the high variance of output lengths for the same prompt, which determines scheduling risk.

2.  **TIE: Equivalent Length with CVaR Tail Penalty**:
    - **Function**: Converts the distribution into a scalar for the SJF queue.
    - **Mechanism**: The predicted distribution is first truncated at $\tilde X = \min(X, x_{\max})$. The priority score is defined as $\text{Score} = \mathbb{E}[\tilde X] + \beta\cdot\mathrm{CVaR}_\alpha[\tilde X]$, where $\mathrm{CVaR}_\alpha[X] = \mathbb{E}[X\mid X\ge \text{VaR}_\alpha(X)]$ is the conditional expectation of the distribution's tail (e.g., $\alpha=0.9$ represents the "average length of the worst 10% cases"). $\beta$ is adaptively adjusted based on system pressure: $\beta = \min(0.5, \max(0.1, 0.1\cdot L_q/B))$, where $L_q$ is the waiting queue length and $B$ is the maximum batch size. Expectations are estimated using 10k Monte Carlo samples. A waiting decay $\text{Score}' = \text{Score}\cdot\gamma^{t_w/\tau}$ ($\gamma=0.9, \tau=30s$) is added to prevent starvation.
    - **Design Motivation**: Using only the expectation $\mathbb{E}[X]$ is equivalent to the classic SEPT strategy (0.75s average latency). Adding CVaR incorporates awareness of tail risk—penalizing requests that might be short on average but have a 10% chance of being extremely long—reducing latency to 0.67s. Adaptive $\beta$ is used because the cost of HOL blocking is low at low loads (favoring greedy SJF) and high at high loads (favoring conservative tail penalties).

3.  **Asynchronous Prediction + Dynamic Batching to Hide Overhead**:
    - **Function**: Solves the engineering problem where the predictor occupies GPU resources and must not block the vLLM main loop.
    - **Mechanism**: The scheduler is split into a main thread (managing the vLLM running batch) and a prediction thread (running DeBERTa in the background). New requests are initially inserted into the min-priority queue with `max_tokens` (ensuring unpredicted requests stay at the bottom) and sent to the prediction queue. The prediction thread performs batch inference when 32 requests are accumulated or 3ms have passed, then updates the scores in the heap with $O(\log n)$ complexity.
    - **Design Motivation**: Previous SSJF / LTR methods used synchronous prediction, causing (1) unnecessary blocking at low loads and (2) prediction bottlenecks at high loads. Asynchronous dynamic batching allows new requests to start running before results correct the order at low loads, while maintaining high predictor throughput at high loads.

### Loss & Training
Predictor training: $\mu$ is normalized via z-score; $\sigma$ is corrected for right-skewness using $\tilde\sigma = \log(1+\sigma)$ before normalization. Two MLP heads each have 3 layers (256, 256, 128). Training is conducted in two stages (full parameters, then freezing DeBERTa to fine-tune MLPs) using MSE. Data consists of the first 45K prompts from LMSYS-Chat-1M with 20 generations each (900K samples total, aligned with SSJF/LTR), achieving $R^2$ values of 0.82 and 0.76 for $\mu$ and $\sigma$, respectively.

## Key Experimental Results

### Main Results
Online chatbot service with an 8B model on LMSYS-Chat-1M, Average Per-Token Latency (PTLA) at 100 RPS:

| Scheduling Policy | LMSYS PTLA (s/token) ↓ | Relative Gain (FCFS) | Relative Gain (LTR) |
|----------|------|------|------|
| FCFS (vLLM Default) | 3.17 (est.) | 1.00× | — |
| SSJF | 1.95 (est.) | 1.62× | — |
| LTR | 1.55 (est.) | 2.05× | — |
| **TIE (Ours)** | **0.67** | **4.73×** | **2.31×** |

Cross-dataset generalization for 70B models (trained only on 8B LMSYS-Chat-1M):

| Test Data | Model | Metric | FCFS | SSJF | LTR | **TIE** |
|----------|------|------|------|------|-----|---------|
| LMSYS-Chat-1M | 70B | Avg PTLA | 9.08 | 5.50 | 4.34 | **2.41** |
| ShareGPT | 70B | Avg PTLA | 4.36 | 2.43 | 2.22 | **1.41** |
| Alpaca | 70B | Avg PTLA | 4.52 | 2.06 | 2.36 | **1.54** |
| LMSYS-Chat-1M | 70B | P90 PTLA | 16.13 | 8.24 | 7.03 | **4.05** |

Offline SDG (Alpaca + 8B): time@3K reduced from 139.5s (LTR) to **98.1s** ($1.42\times$), and 3-minute throughput increased from 3672 → **4762** samples.

### Ablation Study
LMSYS-Chat-1M + 8B online service, PTLA / Time@3K:

| Configuration | Avg PTLA (s) | P90 PTLA (s) | Time@3K (s) | Description |
|------|------|------|------|------|
| **Full TIE** (log-t, $\nu$=3.5, $\mathbb{E}+\beta\cdot\text{CVaR}$) | **0.67** | **0.96** | **98.12** | Default configuration |
| log-t (dynamic $\nu$) | 0.69 | 1.02 | 97.70 | Minimal gain, lower efficiency |
| log-normal replacement | 1.63 | 3.37 | 142.21 | Poor fit (60% vs 90% KS) |
| $\mathbb{E}[X]$ only (SEPT) | 0.75 | 1.21 | 108.51 | Removed CVaR tail penalty |
| $\mathbb{E}+0.1\cdot\text{CVaR}$ (fixed) | 0.72 | 1.15 | 104.76 | Adaptive $\beta$ is superior |
| $\mathbb{E}+0.3\cdot\text{CVaR}$ (fixed) | 0.71 | 1.18 | 105.04 | Same as above |

### Key Findings
- **Distribution family choice determines the scheduling ceiling**: Log-normal KS pass rate dropped from 90.6% to 60.3%, and PTLA increased from 0.67s to 1.63s—distribution fit quality is the bottleneck of this technical route.
- **CVaR is superior to pure expectation**: Removing CVaR degrades the model to SEPT, increasing Avg PTLA from 0.67 to 0.75 (+12%) and P90 from 0.96 to 1.21 (+26%). Tail penalties are crucial for P90 metrics.
- **Strong cross-model/dataset generalization**: Despite training only on 8B + LMSYS, the method remained first on 70B + ShareGPT/Alpaca. This is attributed to distribution modeling avoiding overfitting to specific workloads.
- **Better RPS resilience**: When RPS increased from 30 to 100, FCFS/SSJF/LTR latencies worsened by $7.42\times / 8.55\times / 6.17\times$, while TIE worsened by only $3.68\times$. Adaptive $\beta$ plays a key role in being conservative under high pressure.
- **Visual Explanation** (Figure 5): In (output length, completion time) heatmaps, SSJF/LTR show clusters for short requests but scatter for long ones. TIE maintains high density even in the long-tail region, indicating more accurate sorting for high-variance requests.

## Highlights & Insights
- **Reformulating the scheduling problem as "Distribution Prediction + Risk-Sensitive Sorting"**: This paradigm (log-t + CVaR) can be extended to any scheduling scenario where job execution time is inherently stochastic (e.g., GPU kernel launches, query optimizers, ML training job queues).
- **Elegant theoretical bridge for power-law tails**: Theorem 3.2 rigorously connects "the distribution of termination probabilities across trajectories" to the "power-law tail of output length," providing first-principles support for using heavy-tailed distributions.
- **CVaR is more suitable for scheduling risk than P90**: The authors compare CVaR with the single-point quantile P90, noting that CVaR is the "conditional expectation beyond P90," making it more sensitive to extreme events—an observation transferable to any system using P90/P99 for SLAs.
- **Understated engineering merit of Asynchronous Prediction + Dynamic Batching**: Many predictive scheduling papers have beautiful methods but are unfeasible in practice. The 3ms batching + delayed heap update design hides prediction overhead from the main path with $O(\log n)$ complexity.

## Limitations & Future Work
- **High Retraining Cost**: Fine-tuning DeBERTa requires 900K samples. While the 8B predictor works for 70B models, retraining is still needed for optimal performance on newly released models.
- **log-t($\nu=3.5$) is average-optimal but not globally optimal**: Fixing $\nu$ sacrifices some fit quality. Dynamic $\nu$ showed similar performance but increased complexity; the authors opted for efficiency.
- **Dependency on vLLM's Continuous Batching**: No discussion on other stacks like TensorRT-LLM. The $\beta$ range for CVaR might need recalibration for systems with different KV cache preemption costs.
- **Lack of Multi-turn / Streaming Support**: All experiments assume a single prompt submission. Real chatbots are multi-turn with cumulative context, requiring the predictor to re-predict and account for distribution shifts.
- **Future Directions**: Replacing the predictor with a lighter in-flight head (using LLM forward hidden states instead of separate DeBERTa) to save on inference costs; predicting $\nu$ to achieve a "per-request tail thickness."

## Related Work & Insights
- **vs SSJF (Qiu et al., 2024)**: SSJF uses a small BERT-like model to regress a single numerical output length. This paper uses a similar model but regresses two parameters of a log-t distribution, offering explicit uncertainty characterization.
- **vs LTR (Fu et al., 2024)**: LTR treats prediction as a ranking problem (learning-to-rank) to mitigate regression errors. This paper demonstrates that regression works if the target is a distribution rather than a point. TIE significantly outperforms LTR.
- **vs TRAIL / ELIS (Iterative Prediction + Preemption)**: These methods combat prediction error via "in-generation re-prediction + preemption," which incurs significant overhead. This paper shows that **one-time prediction + no preemption** is sufficient if the distribution is correctly modeled.
- **vs Classic SEPT (Weber, 1983)**: SEPT is the optimal strategy for minimizing expected completion time (sorting by expectation) but only when the distribution is known and untruncated. TIE at $\beta=0$ degrades to SEPT; the CVaR term adds "tail risk aversion," similar to mean-CVaR optimization in finance.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Systematically applying "Distribution Prediction + CVaR Scheduling" to LLM serving is a clear paradigm shift with solid theoretical and experimental support.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 3 datasets × 2 models + online/offline scenarios + 6 distribution families + RPS 30→100 resilience + heatmap visualization.
- Writing Quality: ⭐⭐⭐⭐ Clear logic and motivation with Theorem 3.2. Some formulas (e.g., truncated $\Psi$ expressions) require checking the appendix for full understanding.
- Value: ⭐⭐⭐⭐⭐ Implemented directly on vLLM 0.11.1 with open-source code and non-intrusive changes. Industry serving teams can adopt this with low cost for $2\times$ improvements in latency/throughput.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Fast-dLLM++: Fréchet Profile Decoding for Faster Diffusion LLM Inference](fast-dllm_fréchet_profile_decoding_for_faster_diffusion_llm_inference.md)
- [\[ICML 2026\] A Geometric Relation of the Error Introduced by Sampling a Language Model's Output Distribution to its Internal State](a_geometric_relation_of_the_error_introduced_by_sampling_a_language_models_outpu.md)
- [\[ICML 2026\] SLAY: Geometry-Aware Spherical Linearized Attention with Yat-Kernel](slay_geometry-aware_spherical_linearized_attention_with_yat-kernel.md)
- [\[ICML 2026\] Compute as Teacher: Turning Inference Compute Into Reference-Free Supervision](compute_as_teacher_turning_inference_compute_into_reference-free_supervision.md)
- [\[ICML 2026\] T$^2$PO: Uncertainty-Guided Exploration Control for Stable Multi-Turn Agentic Reinforcement Learning](t2po_uncertainty-guided_exploration_control_for_stable_multi-turn_agentic_reinfo.md)

</div>

<!-- RELATED:END -->

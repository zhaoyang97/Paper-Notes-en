---
title: >-
  [Paper Note] Provable Scaling Laws for the Test-Time Compute of Large Language Models
description: >-
  [NeurIPS 2025][LLM Reasoning][test-time compute] This paper proposes two two-stage test-time compute algorithms — Knockout (pairwise elimination in a tournament bracket) and League (ranking by average win rate) — and proves under minimal assumptions that the failure probability decays exponentially or as a power law to zero as test-time compute increases. The assumptions required are merely that (1) the LLM generates a correct solution with nonzero probability, and (2) the LLM's pairwise comparisons are better than random. The entire pipeline requires only a black-box LLM, with no external verifier or reward model.
tags:
  - NeurIPS 2025
  - LLM Reasoning
  - test-time compute
  - scaling laws
  - knockout
  - league
  - pairwise comparison
  - Best-of-N
  - provable guarantees
date: 2026-05-08
content_hash: c461db6fd3b99b0b
---

# Provable Scaling Laws for the Test-Time Compute of Large Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2411.19477](https://arxiv.org/abs/2411.19477)
**Code**: [GitHub](https://github.com/pan-x-c/AgentScope/tree/feature/pxc/paper_provable/examples/paper_provable_scaling_law)
**Area**: LLM Reasoning
**Keywords**: test-time compute, scaling laws, knockout, league, pairwise comparison, Best-of-N, provable guarantees

## TL;DR

This paper proposes two two-stage test-time compute algorithms — Knockout (pairwise elimination in a tournament bracket) and League (ranking by average win rate) — and proves under minimal assumptions that the failure probability decays exponentially or as a power law to zero as test-time compute increases. The assumptions required are merely that (1) the LLM generates a correct solution with nonzero probability, and (2) the LLM's pairwise comparisons are better than random. The entire pipeline requires only a black-box LLM, with no external verifier or reward model.

## Background & Motivation

**Background**: Test-time compute has become a central strategy for improving LLM reliability. Representative approaches include Best-of-N sampling (generating multiple candidates and selecting the best via a verifier), majority voting (selecting the most frequent answer), and iterative reasoning methods such as chain-of-thought (CoT). In high-stakes settings (e.g., requiring 99.9% accuracy) or multi-step LLM agent workflows (where near-perfect accuracy is needed at each step), boosting success rates by increasing inference-time computation has become a practical necessity.

**Limitations of Prior Work**: (1) Best-of-N sampling relies on external verifiers or reward models; when verifiers are imperfect, performance can degrade as $N$ grows — because "the gains from search are eventually offset by the risk of finding adversarial solutions that fool the verifier." (2) Majority voting has a fundamental theoretical flaw: even if an LLM generates the correct answer with probability 45%, if some incorrect answer is generated with probability 46%, the success rate of majority voting tends to zero as the sample size $N$ increases. (3) Existing empirical scaling laws lack rigorous theoretical guarantees and exhibit inconsistent behavior across tasks.

**Key Challenge**: Practice demands reliable test-time scaling methods that can push success rates arbitrarily close to 100%, yet existing approaches either require additional external components (verifiers) or cannot theoretically guarantee monotonic improvement as compute increases.

**Goal**: Design test-time compute algorithms with **provable scaling laws** under minimal assumptions — i.e., success rates can be driven arbitrarily close to 100% as compute grows, using only a black-box LLM.

**Key Insight**: Leverage the LLM's **pairwise comparison ability** rather than pointwise evaluation. The core intuition is that "judging which of two solutions is better" is substantially easier than "independently judging whether a single solution is correct" — a distinction with deep analogues in computational complexity theory (verification vs. search). Tournament and league structures aggregate multiple pairwise comparisons to amplify a small comparative advantage into high-confidence final selections.

**Core Idea**: Generate $N$ candidate solutions + aggregate via LLM-based pairwise tournament/league = provable test-time scaling without any external verifier.

## Method

### Overall Architecture

A two-stage pipeline using only a black-box LLM in both stages:

- **Stage 1 — Generation**: Given input question $x$, independently sample $N$ candidate solutions $y_1, \dots, y_N \sim \mathcal{M}_{\text{gen}}(x)$, fully parallelizable. Each candidate includes a complete chain of thought (CoT) to support subsequent comparison.
- **Stage 2 — Aggregation**: Select the final answer via Knockout or League tournament structures using LLM pairwise comparisons.

Key notation: $\mathcal{M}_{\text{gen}}$ denotes the LLM's solution generation distribution; $\mathcal{M}_{\text{comp}}$ denotes the distribution over pairwise comparison outputs. Randomness in both stages arises from nonzero-temperature decoding, random selection of prompting strategies, or LLM backends.

### Key Designs

1. **Knockout Algorithm (Tournament-style Aggregation)**

    - **Function**: Selects the final answer from $N$ candidates via multi-round pairwise elimination.
    - **Mechanism**: Randomly pair the $N$ candidates; each pair undergoes $K$ independent comparisons, with the majority winner advancing. After $\lceil \log_2 N \rceil$ rounds, one final output remains. This relies on Assumption 2.1: (A1) the LLM generates a correct solution with probability $p_{\text{gen}} > 0$; (A2) given a pair consisting of a correct and an incorrect solution, the LLM selects the correct one with probability $p_{\text{comp}} > 0.5$. Under these assumptions, **Theorem 2.3** proves that the failure probability satisfies $P(\text{fail}) \leq (1-p_{\text{gen}})^N + \lceil \log_2 N \rceil \cdot e^{-2K(p_{\text{comp}}-0.5)^2}$, which decays exponentially as both $N$ and $K$ grow. **Theorem 2.4** further shows that even with fixed $K$, increasing $N$ alone drives success rates to 1 (power-law decay), removing the need to know $p_{\text{comp}}$ in advance.
    - **Design Motivation**: The tournament structure halves the candidate pool each round, requiring only $(K+1) \times N$ total LLM calls. End-to-end latency is only $T_{\text{gen}} + \log_2(N) \times T_{\text{comp}}$ (each round is fully parallelizable), making this far more compute-efficient than all-pairs comparison.

2. **League Algorithm (Round-Robin Aggregation)**

    - **Function**: Each candidate is compared against $K$ randomly selected opponents; candidates are ranked by average win rate to select the best.
    - **Mechanism**: For each candidate $y_i$, randomly sample $K$ opponents and run independent comparisons, computing the average win rate $\hat{\mu}_i$. The candidate with the highest win rate is the final output. This relies on the more robust Assumption 3.1: there exists a subset $\mathcal{Y}_{\text{cs}}$ of "correct and strong" solutions whose average win rate strictly exceeds the highest win rate of all incorrect solutions by a gap $\Delta > 0$, and the LLM generates such solutions with nonzero probability $p_{\text{cs}} > 0$. **Theorem 3.3** proves that the failure probability satisfies $P(\text{fail}) \leq (1-p_{\text{cs}})^N + 2Ne^{-K\Delta^2/8} + 2Ne^{-(N-1)\Delta^2/8}$, which decays exponentially.
    - **Design Motivation**: Knockout's assumption requires the LLM to outperform random on **every** pair of correct vs. incorrect solutions — a condition that is overly strict, since a single comparison error can eliminate the correct answer. League shifts to an **average win rate** assumption, tolerating systematic failures on specific pairs as long as correct solutions have a higher overall win rate, making it substantially more robust.

3. **Mixed Multi-LLM Strategy**

    - **Function**: Uses a mixture of multiple LLMs in both generation and comparison stages to increase the probability that the theoretical assumptions are satisfied.
    - **Mechanism**: In the generation stage, half the candidates come from Llama3.1 and half from Qwen2.5; the comparison stage likewise uses a mix. Different LLMs excel at different problem types, so mixing ensures that $p_{\text{gen}} > 0$ and $p_{\text{comp}} > 0.5$ hold for a larger set of problems. For example, if LLM$_1$ performs well on question $x_1$ ($p_{\text{gen}}=0.2$) but not on $x_2$ ($p_{\text{gen}}=0$), and LLM$_2$ is the reverse, mixing yields $p_{\text{gen}} = 0.1 > 0$ for both problems.
    - **Design Motivation**: A single LLM may be entirely unable to generate correct solutions ($p_{\text{gen}}=0$) for certain problems. Mixing multiple LLMs substantially broadens the algorithm's applicability.

### Loss & Training

This work is a purely inference-time method requiring **no training or fine-tuning**. It relies solely on zero-shot CoT prompting to drive LLM generation and comparison. The generation stage uses standard CoT prompting ("Let's think step by step"), and the comparison stage presents the full chains of thought of two candidate solutions side-by-side to the LLM, asking it to judge which is better. The full system is implemented on the AgentScope multi-agent framework, supporting parallel and distributed computation.

## Key Experimental Results

### Main Results

| Method | Model | Dataset | Acc. at N=1 | Acc. at N=64 | Gain |
|--------|-------|---------|-------------|--------------|------|
| Knockout | Mixed | GPQA | 45% | 55% | +10pp |
| Knockout | QwQ-32B | GPQA-diamond | 60% | 72% (N=16) | +12pp |
| Knockout (N=8) | Mixed | GPQA | — | > MV (N=64) | — |
| Knockout (N=4) | QwQ-32B | GPQA-diamond | — | > MV (N=16) | — |
| League | Mixed | GPQA | 45% | 53% (N=16) | +8pp |

- Mixed consistently outperforms standalone Llama3.1 or Qwen2.5 at all values of $N$.
- Knockout outperforms majority voting at the same $N$ for most models (except Llama3.1).
- Even accounting for Knockout's additional $(K+1)N$ LLM calls, its accuracy at $N=8$ surpasses majority voting at $N=64$.

### Ablation Study

| Ablation Dimension | Setting | Key Finding |
|--------------------|---------|-------------|
| Filtered subset ($\hat{p}_{\text{comp}} > \tau$) | $\tau = 0.5 / 0.6 / 0.7$ | On subsets satisfying the assumption, accuracy improves from 55% to 80% with increasing $N$ at $\tau=0.5$; higher $\tau$ approaches 100% |
| Task type | Engineering vs. Philosophy (MMLU-Pro) | Reasoning tasks (engineering) scale well; knowledge-intensive tasks (philosophy) scale poorly — comparisons cannot compensate for missing knowledge |
| League opponent count $M$ | $M = 1$ to $N-1$ | Accuracy saturates at $M \geq 4$–$5$; all-pairs comparison is unnecessary, yielding large compute savings |
| Fixed $K$ vs. joint scaling of $N, K$ | $K=2/4$ fixed | Fixed $K$ still yields power-law scaling (Theorem 2.4); jointly increasing $N$ and $K$ achieves exponential scaling |

### Key Findings

- **Provable scaling under minimal assumptions**: As long as the LLM can occasionally generate a correct solution ($p_{\text{gen}} > 0$) and pairwise comparisons are better than random ($p_{\text{comp}} > 0.5$), the failure rate is theoretically guaranteed to decay to zero as compute grows.
- **Task type determines scaling effectiveness**: Reasoning-intensive tasks (e.g., mathematics, engineering) scale substantially better than knowledge-intensive tasks (e.g., philosophy), because in the former the comparison stage can exploit intermediate reasoning to provide additional signal.
- **League achieves significant accuracy gains on the assumption-satisfying subset**: On problems where $\hat{p}_{\text{cs}} > 0$ and $\hat{\Delta} > 0$, Mixed achieves accuracy improvements of up to 25%.
- **Multi-LLM mixing broadens applicability**: Mixed satisfies the theoretical assumptions on a larger set of problems, explaining its consistent advantage over single-model variants.

## Highlights & Insights

- **First formal provable scaling law for test-time compute**: Rather than empirically fitted curves, the results are rigorous mathematical proofs built on two minimal assumptions. The failure probability upper bounds derived in the theorems closely match experimental observations, bridging theory and practice.
- **Deep intuition: comparison is easier than generation or verification**: Judging which of two solutions is better is far easier than independently generating a correct solution or verifying one in isolation — when two solutions are presented side-by-side, the LLM can cross-check reasoning steps, making errors easier to detect. This intuition has deep parallels in LLM alignment (RLHF uses preferences rather than absolute rewards) and social choice theory.
- **Minimal and seamlessly deployable**: Requires only black-box LLM generation and comparison capabilities, with no verifier or reward model training. Plug-and-play. The tournament structure is naturally parallelizable, with end-to-end latency growing only logarithmically.
- **Two algorithms form a complementary pair**: Knockout requires stronger assumptions but is more compute-efficient ($O(N \log N)$ comparisons); League requires weaker assumptions but uses more comparisons ($O(NK)$). In practice, the choice can be guided by task characteristics.

## Limitations & Future Work

- **Comparison assumption may not always hold**: For knowledge-intensive tasks, the LLM's comparison ability may be close to or even below random ($p_{\text{comp}} \approx 0.5$), in which case the algorithm cannot effectively scale.
- **High computational overhead**: Knockout requires $(K+1)N$ LLM calls; the all-pairs version of League requires $O(N^2)$ calls. Although parallelizable, the total token consumption is substantial.
- **Unknown assumption parameters**: $p_{\text{gen}}$, $p_{\text{comp}}$, $\Delta$, etc., are unknown in practice; there is currently no adaptive method to determine the optimal hyperparameters $N$ and $K$ given a target success rate.
- **Independence assumption**: The theoretical analysis assumes that individual comparison judgments are mutually independent, whereas in practice outputs from the same LLM may exhibit systematic biases and correlations.
- **Adversarial incorrect solutions**: The League assumption can be violated by incorrect solutions with anomalously high win rates (analogous to the verifier-fooling problem in BoN). While the authors argue that pairwise comparison is more robust than pointwise evaluation, this risk is not fully eliminated.

## Related Work & Insights

- **vs. Best-of-N sampling**: BoN requires an external reward model, and an imperfect verifier can cause performance reversal; the present work replaces external verification with the LLM's own pairwise comparisons, which is conceptually cleaner and typically more accurate than pointwise evaluation.
- **vs. Majority voting**: Majority voting uses only the frequency of final answers, requiring the correct answer to have strictly the highest occurrence probability — a condition that is hard to satisfy for open-ended tasks or large answer spaces. The present work exploits full reasoning chains via pairwise comparison, making more effective use of available information.
- **vs. Process Reward Models (PRM)**: PRMs require training a verifier for each reasoning step, which is costly and depends on annotated data. The proposed method is entirely training-free.
- **Connection to LLM-as-a-Judge**: Pairwise comparison has been widely validated in settings such as RLHF and Chatbot Arena. This work elevates it from an evaluation tool to a reasoning amplifier, providing rigorous theoretical backing.
- **Implications**: The framework extends naturally to LLM agent workflows — applying Knockout/League at each subtask of a multi-step pipeline can maintain high overall success rates. Furthermore, the method is orthogonal to and composable with approaches such as CoT and self-refinement.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First formal provable scaling law for test-time compute (rather than an empirical fit); combines tournament/league structures with rigorous probabilistic arguments to open a theoretically grounded direction for inference-time scaling.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers three benchmarks (GPQA, MMLU-Pro, MATH-500) and four models (Llama3.1, Qwen2.5, GPT-4o, QwQ-32B), with an elegant parameter-estimation–subset-validation approach bridging theory and experiment; the MMLU-Pro subset of only 100 questions per category is somewhat limited.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Theorem statements are concise and elegant, assumptions are clearly articulated, and the analysis bridging theory and experiment is logically rigorous. The overall structure — motivation → algorithm design → theoretical guarantees → experimental validation → limitations — serves as a model for theoretical ML papers.
- **Value**: ⭐⭐⭐⭐ — The algorithm is plug-and-play with a simple implementation (open-sourced), with direct applicability to high-reliability scenarios; computational overhead and unknown-parameter challenges remain to be addressed for large-scale deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Towards Thinking-Optimal Scaling of Test-Time Compute for LLM Reasoning](towards_thinking-optimal_scaling_of_test-time_compute_for_llm_reasoning.md)
- [\[NeurIPS 2025\] Rethinking Optimal Verification Granularity for Compute-Efficient Test-Time Scaling](rethinking_optimal_verification_granularity_for_compute-efficient_test-time_scal.md)
- [\[NeurIPS 2025\] Does Thinking More Always Help? Mirage of Test-Time Scaling in Reasoning Models](does_thinking_more_always_help_mirage_of_test-time_scaling_in_reasoning_models.md)
- [\[NeurIPS 2025\] Sampling-Efficient Test-Time Scaling: Self-Estimating the Best-of-N Sampling in Early Decoding](sampling-efficient_test-time_scaling_self-estimating_the_best-of-n_sampling_in_e.md)
- [\[ACL 2026\] Scaling Test-Time Compute to Achieve IOI Gold Medal with Open-Weight Models](../../ACL2026/llm_reasoning/scaling_test-time_compute_to_achieve_ioi_gold_medal_with_open-weight_models.md)

</div>

<!-- RELATED:END -->

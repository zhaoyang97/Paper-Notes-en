---
title: >-
  [Paper Note] Does Thinking More Always Help? Mirage of Test-Time Scaling in Reasoning Models
description: >-
  [NeurIPS 2025][LLM Reasoning][Test-time scaling] Through systematic experiments, this paper reveals that the performance of test-time scaling in LRMs (achieved by repeatedly appending "Wait" prompts to extend reasoning)…
tags:
  - "NeurIPS 2025"
  - "LLM Reasoning"
  - "Test-time scaling"
  - "overthinking"
  - "parallel thinking"
  - "variance analysis"
  - "Best-of-N"
date: 2026-05-08
content_hash: 28ffc92ce6fe423b
---

# Does Thinking More Always Help? Mirage of Test-Time Scaling in Reasoning Models

**Conference**: NeurIPS 2025
**arXiv**: [2506.04210](https://arxiv.org/abs/2506.04210)  
**Code**: None  
**Area**: LLM Reasoning
**Keywords**: Test-time scaling, overthinking, parallel thinking, variance analysis, Best-of-N

## TL;DR

Through systematic experiments, this paper reveals that the performance of test-time scaling in LRMs (achieved by repeatedly appending "Wait" prompts to extend reasoning) exhibits a non-monotonic pattern of initial improvement followed by degradation. A probabilistic model is then used to demonstrate that this apparent "gain" is merely a mirage caused by increased output variance rather than genuine reasoning improvement. The proposed parallel thinking strategy achieves accuracy improvements of up to 22% under the same token budget.

## Background & Motivation

**Background**: Large reasoning models (e.g., DeepSeek-R1, OpenAI o1) acquire powerful chain-of-thought reasoning capabilities through RL training. Recent work (Muennighoff et al., 2025) found that appending prompts such as "Wait" or "Think more" during inference to extend the thinking process can improve accuracy, giving rise to the prevailing belief that "more thinking = better reasoning."

**Limitations of Prior Work**: Prior work only reported the beneficial side of test-time scaling, overlooking a critical phenomenon: performance begins to degrade once thinking is extended beyond a certain point. For example, accuracy on GSM8K drops sharply from 87.3% to 70.3% as thinking tokens increase from 1,100 to 15,980. This non-monotonic behavior appears consistently across multiple models and datasets.

**Key Challenge**: Does extended thinking genuinely enhance reasoning ability, or is it merely an illusion? If output variance increases with thinking time, the initial accuracy gain may be a product of statistical noise rather than improved reasoning quality.

**Goal**: (a) Explain why extended thinking produces a non-monotonic performance trend; (b) propose a more effective test-time scaling strategy under a fixed inference budget.

**Key Insight**: A probabilistic framework based on a one-dimensional Gaussian distribution is employed—as the variance of the sampling distribution increases, its overlap with the reward distribution first grows and then shrinks, perfectly explaining the observed non-monotonic behavior. The authors then empirically validate that output entropy does indeed increase with extended thinking.

**Core Idea**: Extending thinking at test time does not improve reasoning but instead increases output variance—the initial gain is a mirage. Under the same budget, sampling multiple reasoning paths in parallel and aggregating via majority voting is a superior strategy.

## Method

### Overall Architecture

The paper is organized into two parts: analysis and solution. In the analysis part, two test-time budget control strategies (Wait & Think More / Exact Thinking Tokens) are designed and evaluated across 3 models × 3 datasets to reveal the non-monotonic trend. A probabilistic framework provides a theoretical explanation, which is empirically validated through entropy measurements. In the solution part, parallel thinking is proposed, allocating the budget across $N$ independent reasoning paths and selecting answers via majority voting.

### Key Designs

1. **Two Test-Time Budget Control (TTBC) Strategies**

    - **Function**: Systematically control the thinking length of reasoning models and observe performance changes.
    - **Mechanism**:
     - TTBC 1 (Wait & Think More): Whenever the model attempts to generate `</think>` to terminate thinking, this token is suppressed and "Wait" is appended, forcing the model to continue reasoning. The number of tokens is not fixed; only the number of "Wait" injections is controlled.
     - TTBC 2 (Exact Thinking Tokens): Forces the number of thinking tokens to be exactly $t_{\text{exact}}$, varied across [256, 512, 1024, 2048, 4096, 8192, 16384].
    - **Design Motivation**: The two complementary control methods ensure that the observed non-monotonic trend is not an artifact of the control strategy itself.

2. **Probabilistic Framework — Theoretical Explanation of the Mirage Effect**

    - **Function**: Explain using a simple one-dimensional Gaussian model why increasing variance leads to an expected reward that first rises and then falls.
    - **Mechanism**: Assuming policy $\pi(y|x) = \mathcal{N}(\mu_\pi, \sigma_\pi^2)$ and reward $r(x,y) = \mathcal{N}(\mu_r, \sigma_r^2)$, the expected reward can be computed analytically as $\frac{1}{\sqrt{2\pi(\sigma_r^2 + \sigma_\pi^2)}} \cdot \exp\left(-\frac{(\mu_r - \mu_\pi)^2}{2(\sigma_r^2 + \sigma_\pi^2)}\right)$. Increasing $\sigma_\pi^2$ involves two competing effects: the **Coverage effect** (when variance is small, increasing it allows sampling to cover more of the region near the reward peak) vs. the **Dilution effect** (when variance is too large, probability mass disperses into regions far from the reward peak).
    - **Design Motivation**: This framework explains why the non-monotonic trend does not imply that "moderate reasoning extension is beneficial"—the initial gain stems from increased randomness rather than improved reasoning quality.

3. **Empirical Validation via Entropy Measurement**

    - **Function**: Directly measure the entropy of the model's output distribution at different thinking lengths.
    - **Mechanism**: After each "Wait" injection, multiple answers are sampled from the model and the entropy $\mathbb{E}[-\log \pi(y|z_{1:i}, x)]$ of the answer distribution is computed. Results show that on GSM8K, entropy grows from 0.23 (standard reasoning, 385 tokens) to 2.79 (6,136 tokens)—a 12× increase.
    - **Design Motivation**: Validates the theoretical prediction that extended thinking increases output variance rather than improving reasoning.
    - **Additional Validation**: Repeating identical reasoning steps by concatenation (rather than continuing to reason) does not increase entropy, confirming that it is overthinking—not simply longer context—that causes variance inflation.

4. **Parallel Thinking**

    - **Function**: Under a fixed reasoning token budget $B$, generate $N$ independent reasoning paths and select the answer via majority voting.
    - **Mechanism**: Generate $N$ independent paths $z^{(i)} \sim \pi(\cdot|x)$ satisfying $\sum_{i=1}^N |z^{(i)}| \leq B$. Each path produces an answer $y^{(i)}$, and the most frequently occurring answer is selected: $y^{\text{best}} = \arg\max_y \sum_i \mathbb{I}[y^{(i)} = y]$.
    - **Design Motivation**: Avoids variance explosion along a single reasoning path. Multiple independent paths each maintain stable variance, and majority voting leverages consistency to filter for high-quality answers. This is essentially a reasoning-time variant of Best-of-N sampling.
    - **Distinction from Prior Methods**: No additional reward model or verifier is required; self-consistency is used directly as the selection criterion.

### Loss & Training

This paper involves no training—it is a purely test-time strategy. Parallel thinking is a resource allocation scheme applied at inference time.

## Key Experimental Results

### Main Results

| Model | Dataset | Standard Acc | Wait Best Acc | Wait Worst Acc | Parallel Acc (16K) |
|-------|---------|-------------|--------------|---------------|-------------------|
| Qwen-1.5B | GSM8K | 82.2% | 87.3% | 70.3% | 92.3% |
| Qwen-1.5B | MATH500 | 83.2% | ~84% | 78.3% | ~87% |
| Qwen-1.5B | AIME | ~23% | ~27% | ~17% | ~37% |
| Qwen-7B | GSM8K | ~93% | ~94% | ~89% | ~96% |
| Llama-8B | GSM8K | ~85% | ~87% | ~78% | ~93% |

### Fixed 16K Budget Comparison

| Strategy | GSM8K (1.5B) | MATH500 (1.5B) | AIME (1.5B) |
|----------|-------------|----------------|-------------|
| Wait & Think More | ~70% | ~78% | ~17% |
| Exact Thinking | ~45% | ~55% | ~10% |
| Parallel Thinking | ~92% | ~87% | ~37% |
| Gain (vs. Wait) | **+22%** | **+9%** | **+20%** |

### Key Findings

- **Non-monotonic trends appear consistently across all settings**: 3 models × 3 datasets × 2 budget control strategies, without exception.
- **Entropy growth is exponential**: On GSM8K, entropy rises from 0.23 (standard thinking) to 2.79 (extended thinking)—a 12× increase—directly demonstrating variance explosion.
- **Repeating identical reasoning steps does not increase entropy**: This confirms that the diversity of overthinking content, not simply longer context, is responsible for variance inflation.
- **Parallel thinking outperforms sequential thinking in all settings**: Under a fixed 16K budget, it surpasses the Wait strategy by 22% on GSM8K and improves monotonically with budget size.
- **Smaller models suffer more severely from overthinking**: The performance degradation for the 1.5B model is substantially larger than for the 7B/8B models.

## Highlights & Insights

- **The probabilistic framework is concise yet powerful**: A one-dimensional Gaussian suffices to explain the complex non-monotonic phenomenon, exemplifying the synergy between theory and experiment. The Coverage vs. Dilution framework can be transferred to any problem involving "whether more attempts are always better" (e.g., search depth, sampling temperature selection).
- **The concept of "mirage" is well-defined**: Prior work concluded that extended thinking is effective simply because "Wait" improves accuracy, but this paper identifies that the improvement stems from variance increase covering a broader answer space—not from enhanced reasoning ability. This is an important conceptual correction.
- **Parallel thinking is extremely simple yet highly effective**: No training or additional models are required; simply splitting the budget across multiple independent reasoning paths and applying majority voting yields substantial gains. Its practical value is high.
- **The repeated-concatenation experiment is an elegant control**: It cleanly disentangles "context length" from "overthinking" as confounding factors.

## Limitations & Future Work

- Experiments are conducted only on medium-scale models (1.5B–8B); behavior on 32B/70B models may differ, as larger models may possess stronger self-correction capabilities.
- The theoretical analysis relies on a one-dimensional Gaussian assumption, whereas real LLM output distributions are far from Gaussian.
- The voting strategy in parallel thinking is simple (majority voting); more sophisticated aggregation methods (e.g., weighted voting, verifier-assisted selection) may yield further improvements.
- The optimal number of parallel paths as a function of problem difficulty is not analyzed—easy problems may require only 2 paths while hard problems may benefit from more.
- Applicability to non-mathematical tasks (code generation, logical reasoning, creative tasks) is not validated.

## Related Work & Insights

- **vs. SBT (Self-Braking Tuning)**: SBT trains models to autonomously stop reasoning, while this paper argues from a different angle that "the benefits of extended thinking are illusory." The two conclusions are complementary. SBT is a training-time solution; parallel thinking is an inference-time solution; they can be combined.
- **vs. L1 (Length-Controlled Policy Optimization)**: L1 controls reasoning length via RL training, whereas this paper requires no training and addresses the problem directly through parallel sampling at inference time.
- **vs. Best-of-N**: Parallel thinking is essentially a self-consistency variant of Best-of-N (using majority voting instead of reward model scoring), eliminating the dependency on an external reward model.

## Rating

- Novelty: ⭐⭐⭐⭐ — The insight that "test-time scaling is a mirage" is highly valuable; the probabilistic explanation is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 3 models × 3 datasets × multiple control strategies provide comprehensive coverage; validation on large models is lacking.
- Writing Quality: ⭐⭐⭐⭐⭐ — The narrative is very clear, progressing systematically from phenomenon to explanation to solution.
- Value: ⭐⭐⭐⭐ — Provides an important conceptual correction for the reasoning model community; parallel thinking is practical and straightforward.

## Core Problem

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Towards Thinking-Optimal Scaling of Test-Time Compute for LLM Reasoning](towards_thinking-optimal_scaling_of_test-time_compute_for_llm_reasoning.md)
- [\[NeurIPS 2025\] Sampling-Efficient Test-Time Scaling: Self-Estimating the Best-of-N Sampling in Early Decoding](sampling-efficient_test-time_scaling_self-estimating_the_best-of-n_sampling_in_e.md)
- [\[NeurIPS 2025\] Atom of Thoughts for Markov LLM Test-Time Scaling](atom_of_thoughts_for_markov_llm_testtime_scaling.md)
- [\[NeurIPS 2025\] LIMOPro: Reasoning Refinement for Efficient and Effective Test-time Scaling](limopro_reasoning_refinement_for_efficient_and_effective_test-time_scaling.md)
- [\[NeurIPS 2025\] Provable Scaling Laws for the Test-Time Compute of Large Language Models](provable_scaling_laws_for_the_testtime_compute_of_large_lang.md)

</div>

<!-- RELATED:END -->

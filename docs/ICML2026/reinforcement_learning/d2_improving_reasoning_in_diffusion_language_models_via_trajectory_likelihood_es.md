---
title: >-
  [Paper Note] d2: Improving Reasoning in Diffusion Language Models via Trajectory Likelihood Estimation
description: >-
  [ICML 2026][Reinforcement Learning][GRPO] This paper proposes the d2 reinforcement learning framework for masked diffusion language models (masked DLM). The core contribution is the introduction of two "trajectory likelihood estimators": d2-AnyOrder, which provides exact single-forward estimates for any-order models, and d2-StepMerge, which provides adjustable
tags:
  - ICML 2026
  - Reinforcement Learning
  - GRPO
date: 2026-05-08
content_hash: ecc1a18f8764a4d9
---
# d2: Improving Reasoning in Diffusion Language Models via Trajectory Likelihood Estimation

**Conference**: ICML 2026  
**arXiv**: [2509.21474](https://arxiv.org/abs/2509.21474)  
**Code**: https://guanghanwang.com/d2  
**Area**: LLM Reasoning / Diffusion Language Models / RL  
**Keywords**: Masked Diffusion Language Model, GRPO, Trajectory Likelihood Estimation, any-order decoding, Post-training

## TL;DR
This paper proposes the d2 reinforcement learning framework for masked diffusion language models (masked DLM). The core contribution is the introduction of two "trajectory likelihood estimators": d2-AnyOrder, which provides exact single-forward estimates for any-order models, and d2-StepMerge, which provides adjustable-precision approximations for standard MDMs. This framework enables the correct implementation of GRPO, allowing LLaDA-8B-Instruct to achieve 91.9% / 56.6% / 85.0% / 41.6% on Sudoku/Countdown/GSM8K/MATH500 respectively, significantly outperforming diffusion RL baselines like d1 and wd1.

## Background & Motivation
**Background**: Diffusion language models (DLM, e.g., LLaDA, Dream, Eso-LM) have become strong competitors to autoregressive LLMs due to their controllable generation and parallel decoding. To equip DLMs with "reasoning" capabilities similar to R1/o1, the mainstream approach is to apply policy gradient methods like GRPO during post-training.

**Limitations of Prior Work**: The GRPO objective involves an importance ratio $\rho_l = \pi_\theta(x_l|x_{<l},q) / \pi_{\text{old}}(x_l|x_{<l},q)$, which can be computed in a single forward pass for autoregressive LLMs. However, the exact likelihood of a DLM is mathematically intractable over $T$ diffusion steps; a "naive" decomposition requires $T$ forward passes, which is computationally prohibitive. Existing works like diffu-GRPO (d1) use sparse $N=1$ approximations, resulting in severely distorted likelihood estimates.

**Key Challenge**: The success of diffusion RL depends on the fidelity of the trajectory likelihood estimation. There is a direct conflict between fidelity and computational budget—exact methods are slow, while approximations are biased.

**Goal**: To solve this in two steps: (1) Rigorously derive the diffusion-version GRPO objective to explicitly expose the "likelihood estimation" component. (2) Design efficient likelihood estimators tailored for different DLM architectures.

**Key Insight**: Trajectory likelihood is the true bottleneck. Certain "any-order autoregressive DLMs" allow the likelihood of an entire trajectory to be packed into a single transformer forward pass. For standard MDMs without this property, the "block composite likelihood" concept can be used to merge diffusion steps.

**Core Idea**: Replace "naive $T$ forward passes" or "single-step sparse approximations" with "customized trajectory likelihood estimators" based on the model category to correctly implement policy gradients for GRPO on masked DLMs.

## Method

### Overall Architecture
The input is a pre-trained masked DLM $\pi_{\text{ref}}$ (e.g., LLaDA-8B-Instruct or Eso-LM) and a reasoning task with a verifiable reward $r(x, q)$ (e.g., Sudoku checker or GSM8K answer matching). The framework consists of three steps: (1) Sample $G$ complete unmasking trajectories $x_{0:T}^{1:L}$ using the old policy $\pi_{\text{old}}$ with a group size $G$. (2) Calculate the group-relative advantage $A^{(i)}$ and estimate the likelihood of $\pi_\theta / \pi_{\text{old}} / \pi_{\text{ref}}$ on the trajectory using d2-AnyOrder or d2-StepMerge. (3) Update the gradients according to the GRPO objective derived in Corollary 3.3. The final output is a policy $\pi_\theta$ that achieves SOTA performance on reasoning tasks.

Theorem 3.1 formalizes the policy gradient for DLMs: at $\theta = \theta_{\text{old}}$, $\nabla_\theta J(\theta) = \nabla_\theta \mathbb{E}_{x_{0:T}^{1:L} \sim \pi_{\text{old}}}[r(x_0^{1:L}, q) \sum_{t=0}^{T-1} \sum_{l=1}^{L} \mathbf{1}_{t,l} \cdot \rho_t^l]$, where $\mathbf{1}_{t,l} = \mathbf{1}\{x_{t+1}^l = m, x_t^l \neq m\}$ indicates "decoding position $l$ at step $t$", and $\rho_t^l = \pi_\theta(x_t^l | x_{t+1}^{1:L}, q) / \pi_{\text{old}}(x_t^l | x_{t+1}^{1:L}, q)$ is the importance ratio per diffusion step. Applying advantage, clipping, and KL constraints yields the GRPO objective (Corollary 3.3). The problem then becomes efficiently estimating $\rho_t^l$.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Input: Pre-trained Masked DLM π_ref<br/>+ Verifiable Reward Task"] --> B["Sample G Trajectories using π_old"]
    B --> C["Compute Group-relative Advantage A<br/>+ Estimate Likelihood Ratio ρ (by Model Type)"]
    C -->|AO-dLLM (Any-order Support)| D["d2-AnyOrder<br/>2L Concatenation + Modulo Positional Encoding<br/>Single Forward for Exact Likelihood"]
    C -->|Standard MDM| E["d2-StepMerge<br/>Merge T steps into N segments<br/>Adjustable N-forward Approximation"]
    D --> F["GRPO clip Objective (Corollary 3.3)<br/>min(ρA, clip(ρ)A) + β·KL Update"]
    E --> F
    F --> G["Output: RL Post-trained Policy π_θ (Reasoning SOTA)"]
```

### Key Designs

**1. d2-AnyOrder: Packing Entire Trajectory Likelihood into One Forward Pass**

For DLMs that naturally support any-order decoding (AO-dLLM, such as Eso-LM or AO-finetuned Qwen3-1.7B), the trajectory likelihood is $\pi(x_{0:T}^{1:L}) = \prod_{l=1}^L \pi(x_0^{\sigma(l)} | x_0^{\sigma(<l)})$. d2-AnyOrder computes these $L$ conditional probabilities without $L$ forward passes by constructing a sequence of length $2L$: $x_0^{1:L} \oplus m^{L+1:2L}$. Tokens and their masks share the same positional encoding $\text{pos}_l = l \bmod L$. A custom attention mask is applied: a clean token $x_0^{\sigma(l)}$ attends only to $x_0^{\sigma(\leq l)}$, while a mask token $m_{L+\sigma(l)}$ attends to $x_0^{\sigma(<l)} \cup \{m_{L+\sigma(l)}\}$. This single forward pass computes all $L$ probabilities. This requires "Independent Masks" and "Order Causality," which are guaranteed by the any-order decoding algorithm (Algorithm 1).

**2. d2-StepMerge: Adjustable Precision Approximation with $N\ll T$ Forward Passes**

Standard MDMs (like original LLaDA) cannot use AnyOrder and cannot afford $T$ forward passes. StepMerge borrows from block composite likelihood, dividing the $T$-step trajectory into $N$ segments and using the forward outputs at the segment endpoints to proxy the likelihood of all tokens within that segment: $\pi(x_{0:T}^{1:L}) \approx \prod_{n=0}^{N-1} \prod_{l=1}^{L} \mathbf{1}_{n,l} \cdot \pi(x_{nT/N}^l | x_{(n+1)T/N}^{1:L})$. Here, $N$ is a compute-bias knob: $N=1$ degrades to diffu-GRPO, while $N=T$ is exact. Theorem 4.1 provides an upper bound $D_N \leq L \cdot \log(T/N + 1) + L \cdot \epsilon_{\text{block}}$, showing that the error decays logarithmically with $N$.

### Loss & Training
Both estimators utilize a clipped GRPO loss (Eq. 8 / Eq. 9), following the structure $\min(\rho A, \text{clip}(\rho, 1-\epsilon, 1+\epsilon) A) + \beta D_{KL}(\pi_\theta \| \pi_{\text{ref}})$, normalized by sequence length $1/L$. Group size $G = 6$ is used with a batch size of 16 problems. Rewards are verifiable (e.g., numerical correctness, Sudoku checker). The method does not rely on SFT or external chain-of-thought data.

## Key Experimental Results

### Main Results

d2 was applied to LLaDA-8B-Instruct and compared against existing diffusion RL frameworks (without SFT):

| Dataset | Metric | d2 (Ours) | wd1 | d1 | LLaDA | Gain (vs Prev. SOTA) |
|--------|------|-----------|------|-----|--------|----|
| Sudoku | Acc | **91.9%** | 25.2% | 22.1% | 11.8% | **+66.7pp** |
| Countdown | Acc | **56.6%** | 51.2% | 42.2% | 19.9% | +5.4pp |
| GSM8K | Acc | **85.0%** | 82.3% | 82.1% | 75.7% | +2.7pp |
| MATH500 | Acc | **41.6%** | 39.0% | 40.2% | 35.4% | +1.4pp |

The +66.7pp gain in Sudoku represents a qualitative shift, indicating that while d1/wd1 fail at strict symbolic logic tasks, accurate trajectory likelihood estimation correctly guides the policy gradient.

### Ablation Study

| Configuration | Sudoku Acc | Notes |
|------|---------|------|
| d2-StepMerge, $N=1$ | ≈ d1 (22.1%) | Equivalent to diffu-GRPO; distorted likelihood |
| d2-StepMerge, $N=4$ | Not convergent | Significant bias, high noise in RL signals |
| d2-StepMerge, $N=16$ | 91.9% | **Sweet spot**: Performance matches $N=32, 64$ with fewer FLOPs |
| d2-StepMerge, $N=32$ | ≈ 91.9% | Performance saturation |
| d2-AnyOrder vs d2-StepMerge ($N=8$) | -0.7 vs -1.5 | On AO models, exact estimation significantly outperforms approximation |
| d2-AnyOrder on LLaDA-8B (no AO) | LL -3.051 vs GT -0.128 | KL=2.334; **AO estimator requires matching training paradigm** |

### Key Findings
- **Likelihood fidelity is critical for diffusion RL**: d1 ($N=1$) achieves only 22.1% on Sudoku, while d2-StepMerge ($N=16$) reaches 91.9%.
- **AO estimator synergy**: Using d2-AnyOrder on vanilla LLaDA results in high KL divergence (2.334). The model must be adapted using AO-causal training to satisfy independence and order causality.
- **Saturation of $N$**: Theorem 4.1's logarithmic bound matches empirical results. Gains from increasing $N$ diminish quickly after $N=16$, providing a clear heuristic for hyperparameter selection.

## Highlights & Insights
- **Parallel likelihood via modulo pos encoding**: The use of a $2L$ token-mask sequence with $\text{pos} = l \bmod L$ to compute $L$ conditional probabilities in one pass is a clever engineering trick that can be applied to other tasks like Bayesian inference of token orders.
- **Generalizing diffu-GRPO**: The paper reinterprets diffu-GRPO as a special case where $N=1$, effectively positioning d2 as a more general and rigorous framework.
- **The Logarithmic Error Bound**: Theorem 4.1 provides a theoretical basis for why increasing computational cost in RL training yields diminishing returns in accuracy, aiding engineering decisions.

## Limitations & Future Work
- **AO training requirements**: d2-AnyOrder is not "plug-and-play" for any MDM; it requires an AO-finetuning stage, which might be an overhead for models already pre-trained without it.
- **Sparse Rewards**: Experiments rely on verifiable binary rewards. The stability of d2 under dense reward models (e.g., standard RLHF reward models) remains unexplored.
- **FLOPs vs. Wall-clock**: While $N$ forward passes are computationally bounded, the sequential dependency of StepMerge may result in higher latency compared to the fully parallel AnyOrder.

## Related Work & Insights
- **vs d1 (diffu-GRPO)**: d1 uses an $N=1$ sparse likelihood. d2 demonstrates that this is a degraded case that fails on complex tasks like Sudoku.
- **vs wd1**: wd1 uses weighted likelihood to avoid importance sampling ratios; d2 proves that sticking to the PPO-style ratio is effective as long as the ratio is estimated accurately.
- **vs DDPO**: DDPO is designed for continuous diffusion and performs poorly on discrete text tasks; d2 provides a decomposition specifically for MDMs.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Framing diffusion RL as a trajectory likelihood problem with two distinct estimators is highly innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive reasoning benchmarks, though lacking wall-clock and dense reward evaluations.
- Writing Quality: ⭐⭐⭐⭐⭐ Mature narrative structure from theory (Theorem 3.1) to practical implementation.
- Value: ⭐⭐⭐⭐⭐ Sets a new SOTA for DLM reasoning and provides a methodological foundation for future diffusion RL research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Learning Unmasking Policies for Diffusion Language Models](learning_unmasking_policies_for_diffusion_language_models.md)
- [\[ICML 2026\] Break the Block: Dynamic-size Reasoning Blocks for Diffusion Large Language Models via Monotonic Entropy Descent with Reinforcement Learning](break_the_block_dynamic-size_reasoning_blocks_for_diffusion_large_language_model.md)
- [\[ICML 2026\] The Shape of Reasoning: Topological Analysis of Reasoning Traces in Large Language Models](the_shape_of_reasoning_topological_analysis_of_reasoning_traces_in_large_languag.md)
- [\[NeurIPS 2025\] Reinforcing the Diffusion Chain of Lateral Thought with Diffusion Language Models](../../NeurIPS2025/reinforcement_learning/reinforcing_the_diffusion_chain_of_lateral_thought_with_diffusion_language_model.md)
- [\[ACL 2026\] d-TreeRPO: Towards More Reliable Policy Optimization for Diffusion Language Models](../../ACL2026/reinforcement_learning/d-treerpo_towards_more_reliable_policy_optimization_for_diffusion_language_model.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] d2: Improving Reasoning in Diffusion Language Models via Trajectory Likelihood Estimation
description: >-
  [ICML 2026][Reinforcement Learning][Masked Diffusion Language Models] This paper proposes the d2 reinforcement learning framework for masked diffusion language models (MDMs). The core consists of two "trajectory likeliho…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "Masked Diffusion Language Models"
  - "GRPO"
  - "Trajectory Likelihood Estimation"
  - "any-order decoding"
  - "Post-training"
date: 2026-05-08
content_hash: 3057f876d4e7712b
---

# d2: Improving Reasoning in Diffusion Language Models via Trajectory Likelihood Estimation

**Conference**: ICML 2026  
**arXiv**: [2509.21474](https://arxiv.org/abs/2509.21474)  
**Code**: https://guanghanwang.com/d2  
**Area**: LLM Reasoning / Diffusion Language Models / Reinforcement Learning  
**Keywords**: Masked Diffusion Language Models, GRPO, Trajectory Likelihood Estimation, any-order decoding, Post-training

## TL;DR
This paper proposes the d2 reinforcement learning framework for masked diffusion language models (MDMs). The core consists of two "trajectory likelihood estimators": d2-AnyOrder, which provides exact single-forward estimates for models supporting any-order decoding, and d2-StepMerge, which provides adjustable-precision approximations for standard MDMs. This enables correct GRPO implementation, allowing LLaDA-8B-Instruct to achieve 91.9% / 56.6% / 85.0% / 41.6% on Sudoku/Countdown/GSM8K/MATH500, significantly outperforming diffusion RL baselines like d1 and wd1.

## Background & Motivation
**Background**: Diffusion Language Models (DLMs, such as LLaDA, Dream, and Eso-LM) have become strong competitors to autoregressive LLMs due to their controllable generation and parallel decoding. To equip DLMs with "reasoning" capabilities similar to R1/o1, the mainstream approach involves post-training using policy gradient methods like GRPO.

**Limitations of Prior Work**: The GRPO objective function contains an importance ratio $\rho_l = \pi_\theta(x_l|x_{<l},q) / \pi_{\text{old}}(x_l|x_{<l},q)$. In autoregressive LLMs, this can be computed in a single forward pass. However, the exact likelihood of a DLM is mathematically intractable over $T$ diffusion steps; a "naive" decomposition requires $T$ forward passes, which is computationally prohibitive. Existing works like diffu-GRPO (d1) use sparse approximations with $N=1$, resulting in severely distorted likelihood estimates.

**Key Challenge**: The success of diffusion RL essentially depends on the fidelity of the trajectory likelihood estimation, which faces a direct conflict with computational budgets—exact estimation is slow, while approximation is biased.

**Goal**: This work addresses the problem in two steps: (1) strictly deriving the GRPO objective for diffusion to explicitly expose the "likelihood estimation" component; (2) designing efficient matching likelihood estimators for different DLM architectures.

**Key Insight**: The authors identify "trajectory likelihood" as the true bottleneck. They observe that a class of "any-order autoregressive DLMs" structurally allows packing the entire trajectory likelihood into a single Transformer forward pass. For standard MDMs without this property, they design a method to merge diffusion steps based on the concept of block composite likelihood.

**Core Idea**: Replace "naive $T$-step forward passes" or "single-step sparse approximations" with "customized trajectory likelihood estimators for specific model categories" to correctly implement policy gradients for GRPO on masked DLMs.

## Method

### Overall Architecture
The input is a pre-trained masked DLM $\pi_{\text{ref}}$ (e.g., LLaDA-8B-Instruct or Eso-LM) and a reasoning task with a verifiable reward $r(x, q)$ (e.g., Sudoku checker or GSM8K answer matching). The framework consists of three steps: (1) Sampling $G$ complete unmasking trajectories $x_{0:T}^{1:L}$ using the old policy $\pi_{\text{old}}$ with a group size $G$; (2) Calculating the group-relative advantage $A^{(i)}$ and estimating the likelihood of $\pi_\theta / \pi_{\text{old}} / \pi_{\text{ref}}$ on the trajectory using d2-AnyOrder or d2-StepMerge; (3) performing gradient updates according to the GRPO objective derived in Corollary 3.3. The final output is the policy $\pi_\theta$ after RL post-training, achieving SOTA performance on reasoning tasks.

The paper first formalizes the policy gradient for DLMs in Theorem 3.1: at $\theta = \theta_{\text{old}}$, $\nabla_\theta J(\theta) = \nabla_\theta \mathbb{E}_{x_{0:T}^{1:L} \sim \pi_{\text{old}}}[r(x_0^{1:L}, q) \sum_{t=0}^{T-1} \sum_{l=1}^{L} \mathbf{1}_{t,l} \cdot \rho_t^l]$, where $\mathbf{1}_{t,l} = \mathbf{1}\{x_{t+1}^l = m, x_t^l \neq m\}$ indicates "decoding position $l$ at step $t$", and $\rho_t^l = \pi_\theta(x_t^l | x_{t+1}^{1:L}, q) / \pi_{\text{old}}(x_t^l | x_{t+1}^{1:L}, q)$ is the step-wise importance ratio. Adding advantage, clipping, and KL constraints yields the GRPO objective (Corollary 3.3). The remaining engineering problem is how to efficiently estimate these $\rho_t^l$.

### Key Designs

1.  **d2-AnyOrder: Exact Trajectory Likelihood Estimation in a Single Forward Pass**:
    - **Function**: For DLMs that naturally support any-order decoding (AO-dLLMs, such as Eso-LM, AO-finetuned Qwen3-1.7B, or any-order causal LLaDA), it compresses the entire trajectory likelihood $\pi(x_{0:T}^{1:L}) = \prod_{l=1}^L \pi(x_0^{\sigma(l)} | x_0^{\sigma(<l)})$ into a single Transformer forward pass.
    - **Mechanism**: Construct a concatenated sequence of length $2L$ as $x_0^{1:L} \oplus m^{L+1:2L}$, where token-mask pairs share the same position encoding $\text{pos}_l = l \bmod L$. A custom attention mask is designed: "clean token $x_0^{\sigma(l)}$ attends only to $x_0^{\sigma(\leq l)}$; mask token $m_{L+\sigma(l)}$ attends only to $x_0^{\sigma(<l)} \cup m_{L+\sigma(l)}$." This allows a single forward pass to output all $L$ conditional probabilities $\pi^{AO}(x_0^l | x_0^{1:L} \oplus m^{L+1:2L})$. The rewards are computed using $\rho_{n,l}^{AO} = \pi_\theta^{AO}(\cdot) / \pi_{\text{old}}^{AO}(\cdot)$ in the GRPO clip objective (Eq. 8).
    - **Design Motivation**: This estimate is unbiased if and only if the sampling process satisfies "Independent Masks" (masks do not attend to each other) and "Order Causality" (decoded tokens only attend to previously decoded ones). These conditions are ensured by the any-order decoding algorithm (Algorithm 1) during sampling. Validation experiments show that applying d2-AnyOrder directly to the original LLaDA-8B-Instruct results in an average per-token log-likelihood of -3.051 versus a ground truth of -0.128 (KL divergence of 2.334), indicating standard MDMs do **not** support this property by default and require specific AO training paradigms.

2.  **d2-StepMerge: Segmented Likelihood Estimation with Adjustable Precision**:
    - **Function**: For standard MDMs (e.g., original LLaDA-8B-Instruct) that do not support any-order decoding, it approximates the full trajectory likelihood using $N$ forward passes ($N \ll T$) with analytically controllable error.
    - **Mechanism**: Drawing from block composite likelihood, the $T$-step trajectory is divided into $N$ segments. The output of a single forward pass at the endpoint of each segment serves as a proxy for the "likelihood of all tokens within that segment": $\pi(x_{0:T}^{1:L}) \approx \prod_{n=0}^{N-1} \prod_{l=1}^{L} \mathbf{1}_{n,l} \cdot \pi(x_{nT/N}^l | x_{(n+1)T/N}^{1:L})$. The corresponding GRPO objective (Eq. 9) defines $\rho_n^l$ as the segment endpoint ratio.
    - **Design Motivation**: $N$ acts as a compute-bias knob—$N=1$ is diffu-GRPO (cheapest but highly distorted), while $N=T$ is the full trajectory (most expensive but exact). Experiments on LLaDA-8B-Instruct show that $D_N$ (KL relative to full decomposition) decreases monotonically with $N$ (Figure 5), with an upper bound given by Theorem 4.1: $D_N \leq L \cdot \log(T/N + 1) + L \cdot \epsilon_{\text{block}}$. Sudoku ablations show $N=16$ is the sweet spot, performing similarly to $N=32, 64$ but with significantly lower FLOPs.

### Loss & Training
The two estimators correspond to separate clipped GRPO losses (Eq. 8 / Eq. 9). Both follow the PPO-style trust region format "$\min(\rho A, \text{clip}(\rho, 1-\epsilon, 1+\epsilon) A) + \beta D_{KL}(\pi_\theta \| \pi_{\text{ref}})$", normalized by $1/L$. Training uses a group size $G = 6$, batches of 16 problems, and decodes 2 tokens per step. All rewards are verifiable (numerical correctness, Sudoku checker, Countdown checker) without relying on SFT or external Chain-of-Thought data.

## Key Experimental Results

### Main Results

Applying d2 to LLaDA-8B-Instruct and comparing it against existing diffusion RL frameworks across four reasoning benchmarks (without SFT):

| Dataset | Metric | d2 (Ours) | wd1 | d1 | LLaDA | Gain (vs Prev. Best) |
|--------|------|-----------|------|-----|--------|----|
| Sudoku | Acc | **91.9%** | 25.2% | 22.1% | 11.8% | **+66.7pp** |
| Countdown | Acc | **56.6%** | 51.2% | 42.2% | 19.9% | +5.4pp |
| GSM8K | Acc | **85.0%** | 82.3% | 82.1% | 75.7% | +2.7pp |
| MATH500 | Acc | **41.6%** | 39.0% | 40.2% | 35.4% | +1.4pp |

The +66.7pp gain in Sudoku represents a qualitative leap—showing that d1/wd1 struggle to learn strict symbolic logic tasks, whereas accurate trajectory likelihood estimation correctly aligns the policy gradient direction. 

Additionally, on AO-finetuned Qwen3-1.7B, AO SFT + d2-AnyOrder reached 67% on GSM8K, exceeding the 63% of AO SFT + diffu-GRPO. In toxicity steering (Eso-LM, 190M), d2-AnyOrder reached -0.7 at $1.25 \times 10^{17}$ FLOPs, while DDPO only reached -8.6.

### Ablation Study

| Configuration | Sudoku Acc | Description |
|------|---------|------|
| d2-StepMerge, $N=1$ | ≈ d1 (22.1%) | Equivalent to diffu-GRPO; likelihood estimate is severely distorted. |
| d2-StepMerge, $N=4$ | Not converged | Estimate still biased; RL signals are noisy. |
| d2-StepMerge, $N=16$ | 91.9% | **Sweet spot**: Performance matches $N=32, 64$ with fewer FLOPs. |
| d2-StepMerge, $N=32$ | ≈ 91.9% | Performance saturated; FLOPs increase. |
| d2-AnyOrder vs d2-StepMerge ($N=8$, Eso-LM) | -0.7 vs -1.5 @ $1.25\times10^{17}$ FLOPs | On AO-supported models, exact estimation significantly outperforms approximation. |
| d2-AnyOrder on LLaDA-8B (No AO training) | per-token LL -3.051 vs GT -0.128 | KL=2.334, **indicating the AO estimator requires a matching training paradigm.** |

### Key Findings
- **Likelihood estimation accuracy is the lifeline of diffusion RL**: d1 ($N=1$) only reaches 22.1% on Sudoku, while d2-StepMerge ($N=16$) hits 91.9%. The difference lies solely in likelihood precision within the same GRPO framework.
- **AO estimators require synergy between sampling and training**: Applying d2-AnyOrder to stock LLaDA results in a KL of 2.334. Models must first use Eso-LM or AO-causal LLaDA training to adapt to independent masks and sequential causality.
- **$N$ exhibits a clear performance plateau**: The logarithmic upper bound in Theorem 4.1 aligns with empirical curves—returns diminish quickly after $N=16$, providing a clear basis for hyperparameter selection.

## Highlights & Insights
- **"Position Encoding Modulo + Custom Attention Mask" enables 2L parallel likelihood**: The implementation of d2-AnyOrder is elegant—concatenating $L$ token-mask pairs with $\text{pos} = l \bmod L$ allows the Transformer to perform "$L$ AR steps in one" without architecture changes. This can be transferred to any scenario requiring "batch conditional probabilities."
- **Reinterpreting diffu-GRPO as an $N=1$ special case**: By framing prior work as a degraded version of their framework (Remark 3.4), the authors provide both a comparison and a strong "general framework" narrative.
- **Logarithmic Error Bound of Theorem 4.1**: The bound $D_N \leq L \log(T/N + 1) + L \epsilon_{\text{block}}$ explains the logarithmic decay of compute-bias, justifying why performance saturates at moderate $N$.

## Limitations & Future Work
- **AO Estimator Requires Matching Training**: d2-AnyOrder cannot be used out-of-the-box on arbitrary MDMs; it requires AO fine-tuning first. For models not supporting AO (like original Dream), one must fall back to d2-StepMerge.
- **Reliance on Verifiable Rewards**: Experiments are based on sparse, binary rewards (answer correctness). The stability of d2 under general reward models or dense rewards remains unexplored.
- **No Wall-clock Comparisons**: Although FLOPs are controlled, the $N$ forward passes in d2-StepMerge have higher sequential dependency than the single forward pass in d2-AnyOrder.
- **Future Directions**: Making $N$ in StepMerge adaptive; extending AO estimators to bidirectional decoding models; or migrating these likelihood estimators to non-GRPO post-training workflows like DPO/SimPO for dLLMs.

## Related Work & Insights
- **vs d1 (diffu-GRPO, Zhao et al. 2025)**: They apply GRPO to MDMs using $N=1$ sparse likelihood. This paper shows that is a degenerate case of d2-StepMerge and improves Sudoku performance from 22.1% to 91.9% by using $N=16$.
- **vs wd1 (Tang et al. 2026)**: wd1 rewrites policy optimization as a weighted likelihood objective to avoid ratio dependency. This paper sticks to the PPO-style ratio + clip but succeeds through accurate ratio estimation.
- **vs DDPO (Black et al. 2024)**: DDPO is PG for continuous diffusion; its performance in prompt-free text settings is weak (-8.6 vs -0.7). This paper replaces the latent-marginal approach with discrete factorization specific to MDMs.
- **Insight**: The key to RL for diffusion models is not which AR RL algorithm to borrow, but how to estimate likelihood correctly. This lesson likely applies to video/image diffusion RL as well.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Reframing diffusion RL as a "trajectory likelihood estimation problem" is a significant conceptual and algorithmic innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ covers multiple benchmarks and architectures, though lacks wall-clock and dense reward evaluations.
- Writing Quality: ⭐⭐⭐⭐⭐ Logical flow from Theorem 3.1 to the estimators is excellent.
- Value: ⭐⭐⭐⭐⭐ Setting a new SOTA for DLM reasoning and establishing a methodological baseline that likelihood must be estimated accurately.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Learning Unmasking Policies for Diffusion Language Models](learning_unmasking_policies_for_diffusion_language_models.md)
- [\[ICML 2026\] Break the Block: Dynamic-size Reasoning Blocks for Diffusion Large Language Models via Monotonic Entropy Descent with Reinforcement Learning](break_the_block_dynamic-size_reasoning_blocks_for_diffusion_large_language_model.md)
- [\[ICML 2026\] The Shape of Reasoning: Topological Analysis of Reasoning Traces in Large Language Models](the_shape_of_reasoning_topological_analysis_of_reasoning_traces_in_large_languag.md)
- [\[ACL 2026\] d-TreeRPO: Towards More Reliable Policy Optimization for Diffusion Language Models](../../ACL2026/reinforcement_learning/d-treerpo_towards_more_reliable_policy_optimization_for_diffusion_language_model.md)
- [\[ICML 2026\] Coupled Variational Reinforcement Learning for Language Model General Reasoning](coupled_variational_reinforcement_learning_for_language_model_general_reasoning.md)

</div>

<!-- RELATED:END -->

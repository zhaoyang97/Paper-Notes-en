---
title: >-
  [Paper Note] FlowRL: Matching Reward Distributions for LLM Reasoning
description: >-
  [ICLR 2026][LLM Reasoning][GFlowNets] FlowRL transforms LLM reasoning RL from "maximizing scalar rewards" to "matching complete reward distributions"—using a learnable partition function to normalize scalar rewards into a target distribution, and leveraging the Trajectory Balance loss of GFlowNets to minimize the reverse KL between the policy and the targe
tags:
  - ICLR 2026
  - LLM Reasoning
  - GFlowNets
  - RLHF
date: 2026-05-08
content_hash: ea770cff693e0894
---
# FlowRL: Matching Reward Distributions for LLM Reasoning

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=lObnTKbm9U](https://openreview.net/forum?id=lObnTKbm9U)  
**Code**: [https://github.com/Xuekai-Zhu/FlowRL](https://github.com/Xuekai-Zhu/FlowRL)  
**Area**: LLM Reasoning / Reinforcement Learning  
**Keywords**: Reward distribution matching, GFlowNets, Trajectory balance, Mode collapse, Exploration diversity, RLHF  

## TL;DR
FlowRL transforms LLM reasoning RL from "maximizing scalar rewards" to "matching complete reward distributions"—using a learnable partition function to normalize scalar rewards into a target distribution, and leveraging the Trajectory Balance loss of GFlowNets to minimize the reverse KL between the policy and the target distribution. This preserves multiple valid reasoning modes and alleviates mode collapse, achieving an average improvement of 10.0%/5.1% over GRPO/PPO in mathematics.

## Background & Motivation
**Background**: From REINFORCE → PPO → GRPO, the objective of mainstream LLM reasoning RL algorithms has consistently been "maximizing expected rewards." GRPO, by eliminating the value network through group comparisons, has become a standard for R1-class strong reasoning models.

**Limitations of Prior Work**: Reward maximization naturally **overfits the dominant mode** of the reward distribution. In long Chain-of-Thought (CoT) reasoning, a single problem often has multiple valid solution paths; however, reward-maximizing methods concentrate all probability mass on a single high-reward peak. This leads to **mode collapse**—the generated reasoning paths are highly homogeneous and rely on the same repetitive tactics (e.g., repeatedly using the AM-GM inequality and falling into dead loops), showing poor generalization for low-frequency but equally correct solutions. The paper quantifies this phenomenon using KL divergence: the KL of reward-maximizing methods relative to the target distribution is as high as 8.68, while distribution matching achieves 0.11.

**Key Challenge**: Effective generalization requires **mode coverage**, but the combination of scalar reward signals and a maximization objective inherently **converges to a unimodal distribution**. Existing mitigation strategies (adjusting clip ratios, entropy-based advantage shaping, selective boosting of high-entropy tokens) are merely "patches" within the maximization framework to implicitly increase diversity; they do not address the problem at the objective function level.

**Goal**: Fundamentally shift the optimization target from "reward maximization" to "reward distribution matching," enabling the policy to sample diverse high-reward trajectories proportional to their rewards.

**Core Idea**: **[Distribution Matching]** Introduce a learnable partition function $Z_\phi(x)$ to normalize the scalar reward $r(x,y)$ into a target distribution $\tilde\pi(y|x)=\exp(\beta r(x,y))/Z_\phi(x)$. **[Flow Balance]** Prove that "minimizing the reverse KL between the policy and this target" is equivalent in terms of expected gradients to the Trajectory Balance loss of GFlowNets, thereby converting intractable KL optimization into a stable squared loss.

## Method

### Overall Architecture
The core of FlowRL is to rewrite the RL objective as a reverse KL divergence and link it to the GFlowNets Trajectory Balance loss. First, a learnable partition function transforms scalar rewards into a normalized target distribution. Then, by minimizing the squared Trajectory Balance loss, the policy approximates "sampling proportional to rewards." To adapt to long CoT training, length normalization (to prevent gradient explosion) and importance sampling (to handle sampling mismatch) are added, resulting in the FlowRL loss which can be directly integrated into the veRL framework.

```mermaid
flowchart TD
    A[Problem x] --> B[Policy πθ samples a group of G CoTs]
    B --> C[Scalar reward r + intra-group normalization r̂]
    A --> D[Partition function Zφ: 3-layer MLP<br/>Input=Mean of hidden states, Output=Scalar]
    C --> E[Target distribution exp βr·πref / Zφ]
    D --> E
    E --> F[Reverse KL = Trajectory Balance Loss<br/>logZφ + logπθ − βr̂ − logπref]
    F --> G[Length normalization 1/|y|·logπ<br/>Prevents gradient explosion]
    F --> H[Importance sampling w=clip detach<br/>Handles off-policy mismatch]
    G --> I[FlowRL Loss]
    H --> I
    I --> B
```

### Key Designs

**1. Partition function transforms scalar rewards into a matchable target distribution:** In long CoT reasoning, the only available supervision is a scalar reward, and enumerating all valid trajectories to reconstruct the true reward distribution is computationally infeasible. Borrowing from energy-based models, FlowRL introduces a learnable partition function $Z_\phi(x)$ to normalize rewards, formulating the optimization as minimizing the reverse KL: $\min_\theta D_{KL}\!\big(\pi_\theta(y|x)\,\|\,\tfrac{\exp(\beta r(x,y))}{Z_\phi(x)}\big)$. The optimal solution is $\pi_\theta(y|x)\propto\exp(\beta r(x,y))$, meaning the policy samples trajectories proportional to their exponentiated rewards rather than collapsing to a single mode. Reverse KL is used because sampling is only possible from the policy, not the target distribution. $Z_\phi$ is parameterized by a 3-layer MLP that takes the mean of the hidden states from the language model encoding $x$ and outputs a scalar.

**2. Trajectory Balance loss converts intractable KL optimization into stable squared loss:** The paper provides Proposition 1—minimizing the reverse KL objective is **equivalent** to minimizing the GFlowNets Trajectory Balance loss $\big(\log Z_\phi(x)+\log\pi_\theta(y|x)-\beta r(x,y)\big)^2$ in terms of expected gradients. This equivalence is the critical bridge: it connects generative modeling (GFlowNets) with policy optimization, allowing the KL objective—which would otherwise require computing an intractable partition function—to be replaced by a **squared loss**. $Z_\phi$ is treated as a learnable parameter updated via gradient descent without integration. To include the reference model as a prior constraint, the reward term is modified from $\exp(\beta r)$ to $\exp(\beta r)\cdot\pi_{ref}(y|x)$, and intra-group reward normalization $\hat r_i=(r_i-\text{mean}(r))/\text{std}(r)$ is applied, resulting in $\min_\theta(\log Z_\phi(x)+\log\pi_\theta(y|x)-\beta\hat r_i-\log\pi_{ref}(y|x))^2$.

**3. Length normalization prevents gradient explosion in long CoT:** Trajectory Balance is a sequence-level objective, where $\log\pi_\theta(y|x)=\sum_{t=1}^{|y|}\log\pi_\theta(y_t|y_{<t},x)$ is the sum over tokens. For CoT sequences reaching 8K tokens, this term scales linearly with length, leading to exploding gradient norms and unstable updates (a new challenge not encountered in traditional short-trajectory GFlowNet work). FlowRL solves this by normalizing the log-probability terms by the sequence length, using $\tfrac{1}{|y|}\log\pi_\theta(y|x)$ instead of the raw sum. From a reward shaping perspective, this balances the contribution of long and short sequences and stabilizes the learning signal.

**4. Importance sampling handles off-policy mismatch:** PPO/GRPO reuse trajectories from an old policy $\pi_{\theta_{old}}$ for micro-batch updates to improve data efficiency, but the KL-TB objective assumes an on-policy setting. FlowRL introduces PPO-style importance sampling, using the ratio $w=\pi_\theta(y|x)/\pi_{old}(y|x)$ to reweight older trajectories. Since the goal is optimizing trajectory balance rather than expected return, the gradient of the current policy in the ratio is **detached** ($w=\text{detach}[\pi_\theta]/\pi_{old}$) to prevent excessive policy drift, combined with PPO-style clipping to bound the weights. The final FlowRL loss is $L=w\cdot\big(\log Z_\phi(x)+\tfrac{1}{|y|}\log\pi_\theta(y|x)-\beta\hat r(x,y)-\tfrac{1}{|y|}\log\pi_{ref}(y|x)\big)^2$.

## Key Experimental Results

### Main Results
Mathematical Reasoning (Avg@16, average across 6 benchmarks), Qwen2.5-Base, max response 8K:

| Model | Method | AIME24 | AIME25 | MATH500 | Olympiad | Average |
|------|------|--------|--------|---------|----------|------|
| 32B | PPO | 26.87 | 20.41 | 69.17 | 37.90 | 43.25 |
| 32B | GRPO | 23.12 | 14.58 | 61.60 | 34.94 | 38.34 |
| 32B | **FlowRL** | 23.95 | **21.87** | **80.75** | **51.83** | **48.39** |
| 7B | GRPO | 13.54 | 9.79 | 57.05 | 26.88 | 32.48 |
| 7B | PPO | 9.38 | 7.29 | 57.98 | 27.25 | 31.98 |
| 7B | **FlowRL** | **15.41** | **10.83** | **66.96** | **34.61** | **35.63** |

On 32B, FlowRL averages 48.39%, which is 5.1% higher than PPO and 10.1% higher than GRPO; gains are particularly significant on difficult tasks like MATH-500 and Olympiad.

Code Reasoning (DeepSeek-R1-Distill-Qwen-7B):

| Method | LiveCodeBench Avg@16 | CodeForces Rating | Percentile | HumanEval+ |
|------|----------------------|-------------------|-----------|-----------|
| PPO | 35.10 | 1403.07 | 73.7% | 82.32 |
| GRPO | 32.75 | 1313.82 | 67.1% | 80.13 |
| **FlowRL** | **37.43** | **1549.47** | **83.3%** | **83.28** |

### Ablation Study
Qwen2.5-7B, average across 6 math benchmarks (Avg):

| Method | Average |
|------|------|
| **FlowRL** | **35.63** |
| w/o Importance Sampling | 26.71 |
| Zhang et al.(2025a) Joint Loss | 33.67 |

$\beta$ Hyperparameter Ablation: $\beta=5 \to 31.34$, $\beta=10 \to 34.41$, **$\beta=15 \to 35.63$ (Optimal)**, $\beta=30 \to 35.09$.

### Key Findings
- **Importance sampling is vital**: Removing it drops the average from 35.63% to 26.71% (−8.92), confirming its role in correcting distribution mismatch between rollouts and the policy. Trajectory-level ratios are more suitable than the joint GFlowNets+PPO loss in Zhang et al.
- **Diversity nearly doubles**: Evaluated by GPT-4o-mini on AIME24/25 rollouts, FlowRL achieves a diversity score of 2.28, far exceeding PPO (1.31), GRPO (1.23), and R++ (1.11). This indicates it generates "qualitatively different solutions" rather than minor variants of the same strategy.
- **Case study intuitive comparison**: For the same AIME problem, GRPO fails by repeatedly applying the AM-GM inequality into an identity loop, leading to a contradictory $a=b=c$ conclusion. FlowRL makes a symmetric assumption $a=b$, transforms it into a cubic equation $a^3-27a+46=0$, and solves it successfully using the rational root test.

## Highlights & Insights
- **Paradigm shift at the objective level**: Instead of patching diversity into a reward-maximizing framework, FlowRL replaces the objective with distribution matching, addressing mode collapse at the source. This is more fundamental than implicit methods like entropy bonuses or clip adjustments.
- **Elegant theoretical bridge**: Proposition 1 proves "reverse KL minimization $\iff$ Trajectory Balance loss," unifying GFlowNets generative modeling with RL policy optimization. It converts the intractable partition function into a learnable parameter with a stable squared loss, making it engineering-ready.
- **Tackling engineering pitfalls of long CoT**: Length normalization for gradient explosion and importance sampling for off-policy mismatch are critical for porting short-trajectory GFlowNets to 8K token CoT, as proven by ablations.
- **Clear Diversity—Generalization chain**: Doubled diversity leads to the largest improvements on difficult benchmarks (MATH-500/Olympiad), experimentally linking "exploration diversity" to "generalization capability."

## Limitations & Future Work
- **Reliance on outcome rewards**: The performance of distribution matching under process rewards or finer-grained signals has not been explored; reward shaping for intermediate steps in CoT remains open.
- **Naive Partition Function $Z_\phi$ design**: Using the mean of hidden states with a 3-layer MLP is simple; whether stronger conditional modeling is needed for varying problem difficulties or modes is not discussed.
- **Scale and task scope**: Experiments are limited to 7B/32B models, math and code domains, and 8K responses. Scalability to larger models, longer contexts, and open-domain reasoning requires verification.
- **$\beta$ tuning**: The distribution "temperature" $\beta$ is sensitive ($\beta=15$ optimal). The need for per-task or per-scale tuning increases deployment costs.

## Related Work & Insights
- **GFlowNets** (Bengio et al. 2023): The methodological foundation, introducing the idea of sampling compositional objects proportional to rewards. This work adapts it to long-sequence stability.
- **Reward-maximizing RL** (PPO/GRPO/REINFORCE++): The baselines and objects of critique, whose mode collapse defects are exposed.
- **Entropy Regularization / High-Entropy Token Boosting / DAPO clipping**: Alternative methods for mitigating diversity degradation, but all operate implicitly within the maximization framework.
- **Insight**: When a task has multiple valid solution paths and generalization depends on mode coverage (multi-step reasoning, program synthesis, molecular/graph generation), "matching the reward distribution" is likely more suitable than "maximizing the reward." Trajectory Balance serves as a stable proxy loss for KL, providing a general path to integrate generative modeling objectives into RL pipelines.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Reframes reasoning RL as distribution matching and bridges GFlowNets TB with reverse KL, offering a fundamental contribution to objective functions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Dual domains (math/code), dual scales (7B/32B), three strong baselines, plus diversity quantification and case studies; however, limited to outcome rewards and 8K length.
- Writing Quality: ⭐⭐⭐⭐ Clear chain from motivation to theory to engineering improvements to experiments. Numbers like KL=0.11 vs 8.68 and doubled diversity are highly persuasive.
- Value: ⭐⭐⭐⭐⭐ Directly addresses mode collapse in long CoT RL and provides an open-source method with significant gains, offering strong reference value for exploration-oriented reasoning RL.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Linking Process to Outcome: Conditional Reward Modeling for LLM Reasoning](linking_process_to_outcome_conditional_reward_modeling_for_llm_reasoning.md)
- [\[ICLR 2026\] DRPO: Efficient Reasoning via Decoupled Reward Policy Optimization](drpo_efficient_reasoning_via_decoupled_reward_policy_optimization.md)
- [\[ICLR 2026\] PEAR: Phase Entropy Aware Reward for Efficient Reasoning](pear_phase_entropy_aware_reward_for_efficient_reasoning.md)
- [\[ICLR 2026\] mR3: Multilingual Rubric-Agnostic Reward Reasoning Models](mr3_multilingual_rubric-agnostic_reward_reasoning_models.md)
- [\[ICML 2026\] PowerFlow: Unlocking the Dual Nature of LLMs via Principled Distribution Matching](../../ICML2026/llm_reasoning/powerflow_unlocking_the_dual_nature_of_llms_via_principled_distribution_matching.md)

</div>

<!-- RELATED:END -->

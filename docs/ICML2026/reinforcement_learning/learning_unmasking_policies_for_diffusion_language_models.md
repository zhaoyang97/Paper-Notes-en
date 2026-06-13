---
title: >-
  [Paper Note] Learning Unmasking Policies for Diffusion Language Models
description: >-
  [ICML 2026][Reinforcement Learning][dLLM sampling] This work explicitly models the decoding process of masked diffusion language models (dLLMs) as an MDP. A single-layer Transformer policy…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "dLLM sampling"
  - "unmasking policy"
  - "GRPO"
  - "adaptive computation"
  - "Bernoulli policy"
date: 2026-05-08
content_hash: 8a7e76d3b4324237
---

# Learning Unmasking Policies for Diffusion Language Models

**Conference**: ICML 2026 Oral Spotlight  
**arXiv**: [2512.09106](https://arxiv.org/abs/2512.09106)  
**Code**: https://github.com/apple/ml-rl-dllm  
**Area**: Reinforcement Learning / Diffusion Language Models / GRPO  
**Keywords**: dLLM sampling, unmasking policy, GRPO, adaptive computation, Bernoulli policy  

## TL;DR
This work explicitly models the decoding process of masked diffusion language models (dLLMs) as an MDP. A single-layer Transformer policy, with parameters less than 0.01% of the base model, is trained using GRPO. Taking only token confidence as input, the policy adaptively determines which positions to unmask at each step. It matches handcrafted heuristics like Fast-dLLM in semi-AR settings and significantly outperforms them in full-diffusion settings, demonstrating transferability across models, tasks, and lengths.

## Background & Motivation
**Background**: Masked diffusion large language models (dLLMs), such as LLaDA and Dream, have matched the performance of autoregressive models of the same scale on downstream tasks. They offer potential for higher throughput by unmasking multiple positions in parallel. Works like Fast-dLLM have pushed inference speeds to levels comparable to or faster than LLaMA using heuristics like "confidence over threshold."

**Limitations of Prior Work**: Handcrafted heuristics perform well only in semi-AR (sequential block generation) configurations. In full-diffusion mode (removing block constraints), their performance is worse than random unmasking. Furthermore, they are extremely sensitive to the confidence threshold $\lambda$ and block length $BL$, requiring per-dataset manual tuning.

**Key Challenge**: Unmasking is essentially a sequential decision-making problem—knowing at which step and which positions to reveal affects both final accuracy and the total number of steps $T-\hat T$. Handcrafted rules approximate this high-dimensional strategy with a single scalar threshold, which fails in fully parallel settings where tokens cannot be generated block-by-block.

**Goal**: (i) Formalize unmasking as an MDP; (ii) learn a lightweight policy to automatically balance accuracy and step count; (iii) verify that the policy can transfer across models, tasks, and lengths.

**Key Insight**: Since the base dLLM already predicts a distribution $p_t^k$ for each position, treating it as the "environment" eliminates the need to train a world model. One only needs to learn a tiny "gateway network" on top of the maximum confidence vector $c_t^k:=\max_v p_t^k(v)$, making the decision overhead negligible.

**Core Idea**: Use the dLLM as the environment and a small policy as the agent. Train a Bernoulli-form unmasking policy via GRPO to let the model learn "when to unmask" and "how much to unmask."

## Method

### Overall Architecture
The pipeline consists of three components: (1) Formulating dLLM sampling as an MDP—the state is the partially unmasked sequence $(\bm x, \bm y_t)$, the action is an unmasking indicator vector $\bm u_t \in \{0,1\}^L$, the transition is performed by the original dLLM, and the reward is provided only upon completion. (2) The policy $\pi_\phi$ is a single-layer Transformer that takes $(\bm c_t, \bm m_t, t)$ as input and outputs logits $\bm b_t$. These are passed through a sigmoid to obtain Bernoulli parameters $s_t^k=\sigma(b_t^k)$, followed by independent sampling for unmasking decisions. (3) Training via GRPO: $G$ rollouts are generated for the same prompt, rewards are centered by subtracting the group mean to compute advantages, and gradients are backpropagated to the policy likelihood. Base dLLM parameters remain frozen.

### Key Designs

1.  **Confidence-only policy**:
    - **Function**: Compresses the partially unmasked sequence into a real-valued vector $\bm c_t$ of length $L$ for the policy, avoiding large-scale operations in the token dimension.
    - **Mechanism**: The policy input uses only the maximum token confidence $c_t^k$ for each position, a binary mask $\bm m_t$, and the timestep $t$. The network is a single-layer Transformer + AdaLN, with parameters $<0.01\%$ of the base model. Ablations found that providing top-50 probabilities offered no gain over the max confidence, while using hidden states led to worse performance and training instability, suggesting $c_t^k$ contains sufficient signals for unmasking decisions.
    - **Design Motivation**: Shares the same intuition as heuristics (using confidence) but learns how to utilize that confidence, avoiding manual thresholds without introducing significant computational overhead.

2.  **Bernoulli dynamic step size (vs. fixed step size)**:
    - **Function**: Allows the number of positions revealed per step to be learned rather than pre-specifying $K$ or a fixed threshold.
    - **Mechanism**: Each position is sampled independently $u_t^k\sim \mathrm{Ber}(s_t^k)$. The policy likelihood is analytically $\pi_\phi(\bm u_t)=\prod_k (s_t^k)^{u_t^k}(1-s_t^k)^{1-u_t^k}$. If $\bm u_t=\bm 0$ during inference, it falls back to unmasking only the position with the maximum $s_t^k$ to prevent deadlocks. A policy temperature $\tau_\pi$ is introduced, replacing $s_t^k$ with $\sigma(b_t^k/\tau_\pi)$ as a "decisiveness" knob at test time.
    - **Design Motivation**: Optimal unmasking counts vary significantly across positions and timesteps for semi-AR and full-diffusion. A Bernoulli formulation is lightweight yet expressive enough to handle this adaptivity.

3.  **Multiplicative reward + GRPO (vs. additive penalty)**:
    - **Function**: Simultaneously encodes accuracy and speed into a single scalar while preventing reward hacking.
    - **Mechanism**: Rewards are issued only at the final step $\hat T$, formulated as $R = r(\bm y, \bm y_{\hat T})\cdot (1-(T-\hat T)/T)^\alpha$, where $r$ is the task accuracy (e.g., 0/1 for GSM8K) and larger $\alpha$ encourages fewer steps. For GRPO, the dLLM temperature is fixed at $\tau=0$ (ensuring variance comes only from the policy). Advantages $A_t^g=R^g-\frac{1}{G}\sum_i R^i$ are computed within the group $G$, and final rewards are backpropagated to each step. KL regularization is omitted since the policy is trained from scratch. Ablations vs. additive rewards $r-\alpha(T-\hat T)/T$ showed the latter collapsed to "unmasking everything in one step" regardless of accuracy.
    - **Design Motivation**: In early training, policies are mostly incorrect. Additive penalties might assign positive advantage to "faster incorrect answers," causing severe reward hacking. Multiplicative rewards mask the speed bonus by accuracy, avoiding this trap.

### Loss & Training
The GRPO objective uses a clipped PPO-style ratio $\rho_t^g = \pi_\phi(\bm u_t^g)/\pi_{\phi_\text{old}}(\bm u_t^g)$, skipping unmasked positions in likelihood calculations. Base dLLMs include LLaDA-8B-Instruct or Dream-7B-Instruct. Training uses ~15k samples each from GSM8K and MATH, with $BL=32$. Separate policies are trained for $\alpha\in\{10,3,1,0.3,0\}$. To mitigate insufficient exploration in full-diffusion ($BL=L=256$), "expert steering" is introduced by injecting Fast-dLLM trajectories into the rollout pool.

## Key Experimental Results

### Main Results
| Dataset/Setting | Metric | Learned Policy | Fast-dLLM | Top-Confidence / Random |
|--------|------|------|------|------|
| GSM8K, $BL=32$ (semi-AR) | acc @ mid-NFE | Comparable to Fast-dLLM (~80%) | Strong baseline | Significantly worse |
| GSM8K, $BL=L=256$ (full-diff) | acc @ ~12 NFEs | ~50% | ≤30% | ≤30% |
| MATH-500, $BL=32$ | acc @ ~25 NFEs (β-scaled) | ~20% | ~10% | — |
| MATH-500, $BL=256$ | full-diff Pareto | Consistently superior | Significant drop | Significant drop |
| GSM8K, expert steering | acc @ mid-high NFE | ~80% (matches best semi-AR) | — | — |
| Transfer LLaDA→Dream | GSM8K acc | Close to Dream-native training | Baseline | — |
| Length transfer $L=256\to512$ | GSM8K acc | Minimal degradation | Significant degradation | — |

### Ablation Study
| Configuration | Key Observation | Explanation |
|------|---------|------|
| Bernoulli vs. Dynamic Plackett-Luce | Similar performance | Bernoulli chosen for simpler implementation and closed-form likelihood |
| Input $c_t^k$ vs. top-50 probs | $c_t^k$ slightly better | Higher granularity uncertainty did not yield benefits |
| Input $c_t^k$ vs. hidden states $\bm h_t^k$ | Hidden states significantly worse + unstable | Key signals reside in post-unembedding projection confidence |
| Zeroing $t$, zeroing $\bm m_t$, or both | Accuracy drops for all; largest drop for zeroing mask | Timestep and mask vectors both contribute to decisions |
| Multiplicative vs. Additive reward ($\alpha=1$) | Additive collapses to one-step unmasking with wrong answers | Multiplicative avoids reward hacking |
| Math Train→HumanEval/MBPP Transfer | Significant drop; recoverable via KodCode-RL-10K retraining | Cross-domain transfer requires diverse training distributions |

### Key Findings
- **Redefining the Pareto Frontier**: In semi-AR, Fast-dLLM is near optimal, and the learned policy matches it. In full-diffusion, however, heuristics perform worse than random; the proposed method is one of the few that continues to benefit from higher NFE.
- **Qualitative Strategy Shift**: In semi-AR, Fast-dLLM focuses on "more computation for early blocks, sequential unmasking for neighbors." The learned policy distributes compute more evenly across blocks and "slows down" when generating numerical answers. In full-diffusion with expert steering, the policy learns left-to-right generation, avoiding "reverse decoding" caused by LLaDA's padding token confidence issues.
- **$\beta$-scaling is more effective than training with multiple $\alpha$**: Training directly with different $\alpha$ often results in collapsed strategies. Scaling Bernoulli parameters to $\min(1, \beta s_t^k)$ at inference time allows for smooth traversal of the accuracy-NFE frontier.
- **Aggressive policies ($\alpha=10$) transfer poorly**: While performing well on LLaDA, they collapse to Fast-dLLM levels on Dream, suggesting high rewards cause overfitting to specific model confidence patterns.

## Highlights & Insights
- **dLLM as a Plug-and-Play Environment**: Unlike works that co-train the policy and the base LM (e.g., d1, DCOLT, DiFFPO), this method keeps the base LM frozen and uses a tiny policy. This makes it a lightweight "accelerator" that can be easily added to existing dLLMs.
- **Multiplicative Reward Firewall**: The combination of sparse 0/1 rewards and speed bonuses often leads policies toward "fast but wrong" shortcuts. Multiplying the penalty by the accuracy mask is a versatile technique applicable to other "accuracy + efficiency" RL tasks like early-exit or adaptive depth.
- **"Confidence is Enough"**: Early exit research suggested confidence-based stopping outperforms hidden-state-based stopping. This work confirms the same for unmasking—projections to the vocabulary capture the essential semantic uncertainty.
- **Bernoulli + Fallback Max**: This combination maintains a closed-form likelihood while avoiding the infinite loops caused by "all-zero" actions.
- **Training with $\tau=0$**: Attributing all intra-group variance to the policy rather than the dLLM's stochasticity reduces noise in GRPO credit assignment, a critical engineering decision for RL-diffusion co-training.

## Limitations & Future Work
- **Coarse Control during Training**: $\alpha$ is not smooth, and expert steering can increase training instability. Better KL control or annealing strategies are needed.
- **No Free Cross-Domain Transfer**: Transferring from math to code (HumanEval, MBPP) results in significant drops, requiring retraining on domain-specific corpora. A "universal policy" remains elusive.
- **Limited Scope**: The work ignores remasking, dynamic generation length, or KV cache optimizations. These could be integrated into a unified MDP in the future.
- **Explainability**: Qualitative differences (e.g., even compute distribution) are observed, but a formal explanation for why certain unmasking patterns are superior is still lacking.
- **Confidence Calibration Dependency**: Since the input is restricted to $c_t^k$, the policy is susceptible to confidence artifacts in the base model (e.g., padding token noise in LLaDA).
- **Training from Scratch**: The removal of KL regularization means no warm-starting from Fast-dLLM imitation, which could be an interesting future comparison.

## Related Work & Insights
Heuristic sampling routes (Fast-dLLM and its variants) proved that confidence signals are essential for dLLM acceleration. RL post-training routes (d1, DiffuCoder, DiFFPO, DCOLT) often tie the policy to the base model with a focus on reasoning. This work aligns with concurrent efforts (e.g., Hong et al. 2025b) using GRPO for unmasking policies; however, the Bernoulli formulation here allows for truly variable step sizes, whereas others often maintain fixed step sizes. Broadly, this extends "adaptive computation" to diffusion models, suggesting that "learning the inference path" can be decoupled from "learning the task itself."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] d2: Improving Reasoning in Diffusion Language Models via Trajectory Likelihood Estimation](d2_improving_reasoning_in_diffusion_language_models_via_trajectory_likelihood_es.md)
- [\[ICML 2026\] Break the Block: Dynamic-size Reasoning Blocks for Diffusion Large Language Models via Monotonic Entropy Descent with Reinforcement Learning](break_the_block_dynamic-size_reasoning_blocks_for_diffusion_large_language_model.md)
- [\[NeurIPS 2025\] Reinforcing the Diffusion Chain of Lateral Thought with Diffusion Language Models](../../NeurIPS2025/reinforcement_learning/reinforcing_the_diffusion_chain_of_lateral_thought_with_diffusion_language_model.md)
- [\[ACL 2026\] d-TreeRPO: Towards More Reliable Policy Optimization for Diffusion Language Models](../../ACL2026/reinforcement_learning/d-treerpo_towards_more_reliable_policy_optimization_for_diffusion_language_model.md)
- [\[ICML 2026\] MoMa QL: Accelerating Diffusion/Flow Matching Policies for Offline and Offline-to-Online RL via Moment Matching](moment_matching_q-learning.md)

</div>

<!-- RELATED:END -->

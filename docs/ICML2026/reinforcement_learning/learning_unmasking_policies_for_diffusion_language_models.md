---
title: >-
  [Paper Note] Learning Unmasking Policies for Diffusion Language Models
description: >-
  [ICML 2026 Oral Spotlight][Reinforcement Learning][dLLM Sampling] This paper explicitly models the decoding process of masked diffusion language models (dLLMs) as an MDP. Using GRPO, it trains a single-layer Transformer policy—comprising less than 0.01% of the base model's parameters and taking only token confidence as input—to adaptively decide which positions to unmask at each step. In the semi-AR setting, it matches manual heuristics like Fast-dLLM…
tags:
  - "ICML 2026 Oral Spotlight"
  - "Reinforcement Learning"
  - "dLLM Sampling"
  - "Unmasking Policy"
  - "GRPO"
  - "Adaptive Computation"
  - "Bernoulli Policy"
date: 2026-05-08
content_hash: c6d964927ec496ce
---

# Learning Unmasking Policies for Diffusion Language Models

**Conference**: ICML 2026 Oral Spotlight  
**arXiv**: [2512.09106](https://arxiv.org/abs/2512.09106)  
**Code**: https://github.com/apple/ml-rl-dllm  
**Area**: Reinforcement Learning / Diffusion Language Models / GRPO  
**Keywords**: dLLM Sampling, Unmasking Policy, GRPO, Adaptive Computation, Bernoulli Policy  

## TL;DR
This paper explicitly models the decoding process of masked diffusion language models (dLLMs) as an MDP. Using GRPO, it trains a single-layer Transformer policy—comprising less than 0.01% of the base model's parameters and taking only token confidence as input—to adaptively decide which positions to unmask at each step. In the semi-AR setting, it matches manual heuristics like Fast-dLLM; in the full-diffusion setting, it significantly outperforms them and demonstrates transferability across models, tasks, and sequence lengths.

## Background & Motivation
**Background**: Masked diffusion large language models (dLLMs) such as LLaDA and Dream have matched the performance of autoregressive models of similar scale on downstream tasks. These models promise higher throughput due to their ability to unmask multiple positions in parallel. Works like Fast-dLLM have pushed inference speeds to parity with or beyond LLaMA using heuristic-based sampling, such as "confidence over threshold."

**Limitations of Prior Work**: Manual heuristics perform well only in semi-AR (sequential chunk generation) configurations. When constraints on blocks are removed for full-diffusion, their performance drops below random unmasking. Furthermore, they are highly sensitive to the confidence threshold $\lambda$ and block length $BL$, requiring per-dataset manual tuning.

**Key Challenge**: Unmasking is essentially a sequential decision-making problem—deciding which positions to reveal at what step simultaneously affects final accuracy and the total number of steps $T-\hat T$. Manual rules approximate this high-dimensional policy using a single scalar threshold, which fails in fully parallel settings that do not allow "first generating within a block, then switching blocks."

**Goal**: (i) Formalize unmasking as an MDP; (ii) learn a lightweight policy to automatically balance accuracy and step count; (iii) verify that the policy can transfer across models, tasks, and lengths.

**Key Insight**: Since the base dLLM already predicts a distribution $p_t^k$ for each position, treating it as the "environment" eliminates the need to train a separate world model. One only needs to train an extremely small "gateway network" on the maximum confidence vector $c_t^k:=\max_v p_t^k(v)$, making decision overhead negligible.

**Core Idea**: Use the dLLM as the environment and a small policy as the agent. Train a Bernoulli-style unmasking policy via GRPO, allowing the model to learn "when to reveal and how much to reveal."

## Method

### Overall Architecture
The pipeline consists of three parts: (1) Formulating dLLM sampling as an MDP—where the state consists of the partially unmasked sequence $(\bm x, \bm y_t)$, the action is an unmasking indicator vector $\bm u_t \in \{0,1\}^L$, transitions are handled by the original dLLM, and rewards are given only upon sequence completion; (2) The policy $\pi_\phi$ is a single-layer Transformer that inputs $(\bm c_t, \bm m_t, t)$ and outputs logits $\bm b_t$. These are passed through a sigmoid to obtain Bernoulli parameters $s_t^k=\sigma(b_t^k)$, from which unmasking decisions are sampled independently per position; (3) Training utilizes GRPO: $G$ rollouts are generated for the same prompt, and rewards minus the group mean yield the advantage for backpropagation to the policy likelihood at each step. The base dLLM parameters remain frozen throughout. The entire pipeline forms an "environment-policy" loop: the dLLM environment outputs confidence scores, the lightweight policy samples unmasking actions, which in turn advance the dLLM decoding until completion, at which point rewards are calculated and fed back to update the policy.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}}%%
flowchart TD
    A["Input: prompt x + fully masked sequence y_T"] --> B["Base dLLM forward pass (frozen parameters)<br/>yields per-position prediction distribution p_t^k"]
    B --> C["Confidence as state<br/>c_t^k=max_v p_t^k(v) plus mask m_t, step t<br/>→ single-layer Transformer policy π_φ outputs logit b_t"]
    C --> D["Bernoulli adaptive step<br/>s_t^k=σ(b_t^k), u_t^k~Ber(s_t^k)<br/>per-position independent reveal/no-reveal decision"]
    D -->|masked positions remain| B
    D -->|all positions revealed (terminal step T̂)| E
    subgraph TRAIN["Multiplicative Reward + GRPO"]
        direction TB
        E["Multiplicative Reward<br/>R = r·(1−(T−T̂)/T)^α, wrong answer zeroes out advantage"] --> F["GRPO: G rollouts per prompt compute group-wise advantage<br/>terminal reward back-propagated to each step, updates π_φ (dLLM frozen)"]
    end
```

### Key Designs

**1. Confidence as State: Compressing the partially unmasked sequence into a length-$L$ real-valued vector for the policy**

The base dLLM already predicts distributions $p_t^k$ for each position. Treating it as the environment avoids training a separate world model, requiring the policy to make decisions on an extremely lightweight "gateway." Specifically, the policy input uses only the maximum token confidence $c_t^k:=\max_v p_t^k(v)$ for each position, combined with a binary mask $\bm m_t$ and time step $t$. The network is a single-layer Transformer with AdaLN, scaling to $<0.01\%$ of the base model's parameters. Ablations crucially show "why max is enough": feeding top-50 probabilities did not improve performance, and using hidden states resulted in inferior performance and training instability—the "reveal or not" signal is effectively carried by $c_t^k$ after the unembedding projection. This approach is rooted in the same logic as heuristics like Fast-dLLM (both monitor confidence), but delegates "how to use confidence" to learning to avoid manual thresholds without adding computational overhead.

**2. Bernoulli Dynamic Steps: Making the number of revealed positions per step a learnable quantity rather than a preset $K$ or fixed threshold**

The optimal number of unmasked positions in semi-AR versus full-diffusion varies across positions and time; fixed $K$ or thresholds cannot satisfy both. Here, each position is sampled independently via $u_t^k\sim \mathrm{Ber}(s_t^k)$. The policy likelihood is analytically expressed as $\pi_\phi(\bm u_t)=\prod_k (s_t^k)^{u_t^k}(1-s_t^k)^{1-u_t^k}$, avoiding approximations like Plackett-Luce. During inference, if $\bm u_t=\bm 0$, the system falls back to "revealing the position with the largest $s_t^k$" to prevent deadlocks. A policy temperature $\tau_\pi$ is introduced, converting $s_t^k$ to $\sigma(b_t^k/\tau_\pi)$ to act as a "decisiveness" knob during testing. Compared to the fixed $K$ or threshold predictions in DCOLT/DiFFPO, the Bernoulli formulation allows the step size to be truly adaptive per step and per position while remaining lightweight and expressive.

**3. Multiplicative Reward + GRPO: Encoding accuracy and speed within a single scalar to avoid reward hacking**

Policies trained from scratch often fail early tasks. If using additive penalties like $r-\alpha(T-\hat T)/T$, "fast wrong answers" could retain positive advantage, causing the policy to collapse into "revealing everything at once, regardless of accuracy." The authors switch to a multiplicative reward, issued only at the terminal step $\hat T$: $R = r(\bm y, \bm y_{\hat T})\cdot (1-(T-\hat T)/T)^\alpha$ (where $r$ is task accuracy, and larger $\alpha$ favors fewer steps). By multiplying the speed reward by the accuracy mask, "fast wrong answers" are immediately reduced to zero advantage. Training uses GRPO: the dLLM temperature $\tau=0$ is fixed to ensure intra-group variance originates only from the policy. Advantages $A_t^g=R^g-\frac{1}{G}\sum_i R^i$ are computed for $G$ trajectories per group, with terminal rewards distributed back to each step for PPO-style clipping. KL regularization is omitted since training begins from scratch.

### Loss & Training
The GRPO objective uses a clipped PPO-style ratio $\rho_t^g = \pi_\phi(\bm u_t^g)/\pi_{\phi_\text{old}}(\bm u_t^g)$, skipping already unmasked positions during likelihood calculation. Base dLLMs include LLaDA-8B-Instruct or Dream-7B-Instruct. Training data consists of approximately 15,000 samples each from GSM8K and MATH, with one epoch at $BL=32$ and five separate models trained for each $\alpha\in\{10,3,1,0.3,0\}$. To mitigate insufficient exploration in full-diffusion ($BL=L=256$), the authors introduce "expert steering": trajectories generated by Fast-dLLM under semi-AR are injected into the rollout pool to guide the policy out of local optima.

## Key Experimental Results

### Main Results

| Dataset/Setting | Metric | Learned Policy | Fast-dLLM | High-Confidence Sampling / Random |
|--------|------|------|------|------|
| GSM8K, $BL=32$ (semi-AR) | acc @ mid-NFE | Comparable to Fast-dLLM (~80%) | Strong Baseline | Significantly worse |
| GSM8K, $BL=L=256$ (full-diff) | acc @ ~12 NFEs | ~50% | ≤30% | ≤30% |
| MATH-500, $BL=32$ | acc @ ~25 NFEs (β-scaled) | ~20% | ~10% | — |
| MATH-500, $BL=256$ | full-diff Pareto | Leads throughout | Substantial drop | Substantial drop |
| GSM8K, expert steering | acc @ mid-high NFE | ~80% (Matches semi-AR best) | — | — |
| LLaDA→Dream Transfer | GSM8K acc | Near Dream-direct training | Baseline | — |
| Length Transfer $L=256\to512$ | GSM8K acc | Almost no drop | Significant drop | — |

### Ablation Study

| Configuration | Key Phenomenon | Explanation |
|------|---------|------|
| Bernoulli vs. Dynamic Plackett-Luce | Similar performance | Bernoulli chosen for simpler implementation and closed-form likelihood |
| Input $c_t^k$ vs. top-50 probabilities | $c_t^k$ slightly better | Finer-grained uncertainty did not yield gains |
| Input $c_t^k$ vs. Hidden state $\bm h_t^k$ | Hidden states significantly worse + unstable | Critical signals reside in confidence after unembedding projection |
| Zeroing $t$, zeroing $\bm m_t$, or both | All caused accuracy drops; zeroing mask was worst | Both time and mask vectors contribute to decision-making |
| Multiplicative vs. Additive Reward ($\alpha=1$) | Additive collapses to one-step reveal with wrong answers | Multiplicative reward prevents reward hacking |
| Math trained → HumanEval/MBPP transfer | Significant drop; fixed by retraining on KodCode-RL-10K | Cross-domain transfer requires diverse training distributions |

### Key Findings
- **Redefining Optimal Frontiers**: Fast-dLLM is already near-optimal under semi-AR; the learned policy only matches it. However, once switched to full-diffusion, heuristics fall below random unmasking, while this method remains one of the few that improves performance with higher NFEs.
- **Distinct Policy Behavior**: Under semi-AR, Fast-dLLM tends to "over-calculate earlier blocks and reveal neighbor positions," whereas the learned policy distributes computation more evenly across blocks and "slows down" when generating numerical answers. In full-diffusion with expert steering, the policy learns left-to-right generation, avoiding "reverse decoding" caused by padding token confidence pollution in LLaDA.
- **Alpha Control vs. Inference Scaling**: Adjusting $\alpha$ directly leads to value collapses into identical policies. Scaling Bernoulli parameters via $\min(1, \beta s_t^k)$ during inference allows for a smoother traversal of the accuracy-NFE Pareto frontier.
- **Faster policies trained at $\alpha=10$ have poor transferability**: While performing best on LLaDA, they collapse to Fast-dLLM levels on Dream, suggesting that steep rewards cause the policy to overfit the confidence patterns of a specific model.

## Highlights & Insights
- **Using Trained dLLMs Directly as Environments**: Unlike works that jointly train policies with base LMs (d1, DCOLT, DiFFPO, etc.), this method keeps the base model frozen with minimal policy parameters, resulting in low training costs and "plug-and-play" compatibility for open-source dLLMs.
- **Multiplicative Rewards as a Firewall Against Hacking**: In sparse 0/1 reward settings combined with speed penalties, the trap of "wrong but fast" frequently misleads policies. Multiplying the penalty with correctness is a versatile technique applicable to other "accurate + efficient" RL tasks (e.g., early-exit, adaptive depth).
- **"Confidence is Sufficient" as a General Rule**: Early research on exits found confidence-based stopping superior to hidden-state-based methods. This work confirms the same for unmasking—maximum values after vocab projection effectively encapsulate semantic uncertainty.
- **Bernoulli + Fallback Max**: This combination preserves closed-form likelihood while avoiding infinite loops caused by "all-zero actions," serving as a useful engineering trick.
- **$\beta$-scaling as an Inference Knob**: Using $\min(1,\beta s_t^k)$ to smoothly slide the accuracy-NFE frontier during inference is more efficient than retraining for different $\alpha$ values, providing a practical "one policy, multiple gears" approach for deployment.
- **Forced $\tau=0$ During Training**: Attributing all group variance to policy actions rather than dLLM stochasticity significantly reduces credit assignment noise in GRPO, a critical but often overlooked engineering decision in RL-Diffusion joint training.

## Limitations & Future Work
- **Imprecise Control Sensitivity**: $\alpha$ is not smooth, and adding expert steering further increases training instability; better KL control or annealing strategies are needed.
- **Cross-Domain Transfer is Not Free**: Performance drops significantly when transferring from math to code tasks (HumanEval, MBPP), requiring retraining on code corpora; "universal policies" are yet to be achieved.
- **Only Addresses Unmasking Order**: Orthogonal acceleration methods like remasking, dynamic sequence length, and KV cache are not included; future work could incorporate these into the same MDP.
- **Limited Policy Interpretability**: Although qualitative differences are observed (e.g., even computation distribution, left-to-right generation), formal explanations for "why this reveal order is optimal" are lacking.
- **Dependency on Base dLLM Calibration**: Since the policy input is only $c_t^k$, base model issues like padding token confidence pollution (LLaDA) or tail-end overconfidence will degrade policy performance.
- **Training from Scratch vs. Fine-tuning**: The removal of KL regularization means any "imitate Fast-dLLM then RL" two-stage schemes were avoided, losing potential warm-start benefits worth comparing in the future.

## Related Work & Insights
Heuristic sampling routes (Fast-dLLM and its variants by Ben-Hamu, Kim, Wei, etc.) proved that "confidence signals" are vital for accelerating dLLMs. RL post-training routes (d1, DiffuCoder, DiFFPO, DCOLT) mostly bind policies to the base model with a focus on reasoning capabilities. This work is concurrent with Hong et al. 2025b: both use GRPO to train standalone unmasking policies, but the Bernoulli formulation here allows for truly variable step sizes, whereas the latter maintains fixed steps.
Broadly, this line of work extends "Adaptive Computation" (Graves, Bengio, etc.) to diffusion language models, suggesting that "learning the reasoning path" can be decoupled from "learning the reasoning itself" for generic, transferable accelerators. Furthermore, this work is complementary to KV cache, speculative decoding, and distilled decoders. Combining RL-learned unmasking policies with these optimizations should push dLLM throughput limits even further.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] d2: Improving Reasoning in Diffusion Language Models via Trajectory Likelihood Estimation](d2_improving_reasoning_in_diffusion_language_models_via_trajectory_likelihood_es.md)
- [\[NeurIPS 2025\] Reinforcing the Diffusion Chain of Lateral Thought with Diffusion Language Models](../../NeurIPS2025/reinforcement_learning/reinforcing_the_diffusion_chain_of_lateral_thought_with_diffusion_language_model.md)
- [\[ACL 2026\] d-TreeRPO: Towards More Reliable Policy Optimization for Diffusion Language Models](../../ACL2026/reinforcement_learning/d-treerpo_towards_more_reliable_policy_optimization_for_diffusion_language_model.md)
- [\[ICLR 2026\] SPG: Sandwiched Policy Gradient for Masked Diffusion Language Models](../../ICLR2026/reinforcement_learning/spg_sandwiched_policy_gradient_for_masked_diffusion_language_models.md)
- [\[ICML 2026\] Break the Block: Dynamic-size Reasoning Blocks for Diffusion Large Language Models via Monotonic Entropy Descent with Reinforcement Learning](break_the_block_dynamic-size_reasoning_blocks_for_diffusion_large_language_model.md)

</div>

<!-- RELATED:END -->

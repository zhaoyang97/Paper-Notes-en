---
title: >-
  [Paper Note] Learning Unmasking Policies for Diffusion Language Models
description: >-
  [ICML 2026][Reinforcement Learning][GRPO] This paper explicitly models the decoding process of masked diffusion language models (dLLMs) as an MDP. It employs GRPO to train a single-layer Transformer policy—comprising less than 0.01% of the base model's parameters—that takes only token confidence as input to adaptively determine which positions to unmask at eac
tags:
  - ICML 2026
  - Reinforcement Learning
  - GRPO
date: 2026-05-08
content_hash: f126126b4e840a17
---
# Learning Unmasking Policies for Diffusion Language Models

**Conference**: ICML 2026 Oral Spotlight  
**arXiv**: [2512.09106](https://arxiv.org/abs/2512.09106)  
**Code**: https://github.com/apple/ml-rl-dllm  
**Area**: Reinforcement Learning / Diffusion Language Models / GRPO  
**Keywords**: dLLM sampling, unmasking policy, GRPO, adaptive computation, Bernoulli policy

## TL;DR
This paper explicitly models the decoding process of masked diffusion language models (dLLMs) as an MDP. It employs GRPO to train a single-layer Transformer policy—comprising less than 0.01% of the base model's parameters—that takes only token confidence as input to adaptively determine which positions to unmask at each step. This approach matches manual heuristics like Fast-dLLM in semi-AR settings while significantly outperforming them in full-diffusion settings, demonstrating transferability across models, tasks, and sequence lengths.

## Background & Motivation
**Background**: Masked diffusion large language models (dLLMs), represented by LLaDA and Dream, have matched the performance of autoregressive models of similar scale on downstream tasks. They hold high promise for increased throughput due to their ability to unmask multiple positions in parallel. Works like Fast-dLLM have pushed inference speeds to be comparable to or faster than LLaMA using "confidence thresholding" heuristics.

**Limitations of Prior Work**: Heuristic rules perform well only in semi-AR configurations (sequential block generation). In full-diffusion settings without block constraints, their performance often falls below random unmasking. Furthermore, they are extremely sensitive to the confidence threshold $\lambda$ and block length $BL$, requiring manual tuning for each dataset.

**Key Challenge**: Unmasking is essentially a sequential decision-making problem—deciding which positions to reveal and at which step affects both final accuracy and the total step count $T-\hat T$. Manual rules approximate this high-dimensional policy using a single scalar threshold, which collapses in fully parallel settings where "generate within block, then switch blocks" is not allowed.

**Goal**: (i) Formalize unmasking as an MDP; (ii) Learn a lightweight policy to automatically balance accuracy and step count; (iii) Verify the policy's transferability across models, tasks, and lengths.

**Key Insight**: Since the base dLLM already predicts a distribution $p_t^k$ for each position, treating it as the "environment" eliminates the need to train a separate world model. One only needs to learn a very small "gateway network" on the maximum confidence vector $c_t^k := \max_v p_t^k(v)$, making decision overhead negligible.

**Core Idea**: Use the dLLM as the environment and a small policy as the agent. Train a Bernoulli-style unmasking policy via GRPO to let the model learn when and how much to reveal.

## Method

### Overall Architecture
The pipeline consists of three components: (1) Formulating dLLM sampling as an MDP—the state is the partially decrypted sequence $(\bm x, \bm y_t)$, the action is a $\{0,1\}^L$ unmasking indicator vector $\bm u_t$, the transition is performed by the original dLLM, and rewards are given only upon completion. (2) The policy $\pi_\phi$ is a single-layer Transformer that takes $(\bm c_t, \bm m_t, t)$ as input and outputs logits $\bm b_t$. These are passed through a sigmoid to obtain Bernoulli parameters $s_t^k=\sigma(b_t^k)$, followed by independent sampling for each position. (3) Training via GRPO: multiple rollouts ($G$) are run for the same prompt; advantages are calculated by subtracting the group mean from the reward and backpropagated to each step's policy likelihood. The base dLLM parameters remain frozen throughout. The entire pipeline forms an "environment-policy" loop: the dLLM environment outputs confidence, the lightweight policy samples unmasking actions, and the feedback advances the dLLM decoding until completion, at which point the reward is calculated and used to update the policy.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}}%%
flowchart TD
    A["Input: prompt x + all-mask sequence y_T"] --> B["Base dLLM Forward (Frozen)<br/>Obtain per-position distribution p_t^k"]
    B --> C["Confidence as State<br/>c_t^k=max_v p_t^k(v) with mask m_t, step t<br/>→ Single-layer Transformer policy π_φ outputs logit b_t"]
    C --> D["Bernoulli Dynamic Step Size<br/>s_t^k=σ(b_t^k), u_t^k~Ber(s_t^k)<br/>Independent decision to unmask or not"]
    D -->|Masks remain| B
    D -->|Fully unmasked (Final step T̂)| E
    subgraph TRAIN["Multiplicative Reward + GRPO"]
        direction TB
        E["Multiplicative Reward<br/>R = r·(1−(T−T̂)/T)^α, Incorrect answers get 0 advantage"] --> F["GRPO: G rollouts per prompt to calculate group advantage<br/>Final reward backpropagated to every step; update π_φ (dLLM fixed)"]
    end
```

### Key Designs

**1. Confidence as State: Compressing the partially decrypted sequence into a length $L$ real-valued vector for the policy**

By treating the base dLLM (which predicts distributions $p_t^k$) as the environment, the need for a world model is bypassed. The policy makes decisions using an extremely lightweight "gateway" network. Specifically, policy input uses only the maximum token confidence $c_t^k:=\max_v p_t^k(v)$ for each position, a binary mask $\bm m_t$, and the time step $t$. The network is a single-layer Transformer with AdaLN, sized at $<0.01\%$ of the base parameters. Ablations demonstrate why "max is enough": feeding top-50 probabilities does not improve performance, and using hidden states leads to worse performance and training instability. The signal for "whether to unmask" is carried effectively by $c_t^k$ after the unembedding projection. This aligns with Fast-dLLM heuristics but delegates the confidence-usage strategy to learning, avoiding manual thresholds without adding computational overhead.

**2. Bernoulli Dynamic Steps: Making the unmasking count per step a learnable variable rather than a preset $K$ or fixed threshold**

Optimal unmasking counts vary significantly between semi-AR and full-diffusion settings and across steps. Neither a fixed $K$ nor a fixed threshold can generalize. Here, each position is independently sampled $u_t^k\sim \mathrm{Ber}(s_t^k)$. The analytical policy likelihood is $\pi_\phi(\bm u_t)=\prod_k (s_t^k)^{u_t^k}(1-s_t^k)^{1-u_t^k}$, avoiding approximations like Plackett-Luce. During inference, if $\bm u_t=\bm 0$, the system falls back to "unmasking only the position with the highest $s_t^k$" to prevent stalling. A policy temperature $\tau_\pi$ is introduced to adjust $s_t^k$ to $\sigma(b_t^k/\tau_\pi)$ as a "decisiveness" knob during testing. Compared to DCOLT/DiFFPO's fixed $K$ or threshold prediction, the Bernoulli formulation allows step sizes to be truly adaptive per position and per step while remaining expressive and lightweight.

**3. Multiplicative Reward + GRPO: Encoding "correctness" and "speed" into a single scalar while avoiding reward hacking**

Initial policies tend to produce many errors. An additive penalty $r-\alpha(T-\hat T)/T$ might result in "faster incorrect answers" retaining positive advantages, causing the policy to collapse into unmasking everything at once regardless of correctness. Ours uses a multiplicative reward, issuing $R = r(\bm y, \bm y_{\hat T})\cdot (1-(T-\hat T)/T)^\alpha$ at the final step $\hat T$ (where $r$ is task correctness and higher $\alpha$ favors fewer steps). By multiplying the speed reward by the correctness mask, "fast but wrong" answers are reduced to 0 advantage. Training utilizes GRPO: dLLM temperature is fixed at $\tau=0$ to ensure group variance stems solely from the policy. Group advantage for $G$ trajectories is calculated as $A_t^g=R^g-\frac{1}{G}\sum_i R^i$, and final rewards are backpropagated to each step using PPO-style clipping. KL regularization is omitted since training starts from scratch.

### Loss & Training
The GRPO objective is a PPO-style ratio $\rho_t^g = \pi_\phi(\bm u_t^g)/\pi_{\phi_\text{old}}(\bm u_t^g)$ with clipping. Likelihood calculations skip already unmasked positions. Base dLLMs used are LLaDA-8B-Instruct or Dream-7B-Instruct. Training data consists of approx. 15k samples each from GSM8K and MATH, one epoch with $BL=32$, and five separate trainings for each $\alpha\in\{10,3,1,0.3,0\}$. To mitigate insufficient exploration in full-diffusion ($BL=L=256$), "expert steering" is introduced: trajectories generated by Fast-dLLM in semi-AR settings are injected into the rollout pool to guide the policy out of local optima.

## Key Experimental Results

### Main Results
| Dataset/Setting | Metric | Learned Policy | Fast-dLLM | Top-Confidence / Random |
|--------|------|------|------|------|
| GSM8K, $BL=32$ (semi-AR) | acc @ mid-NFE | Comparable to Fast-dLLM (~80% range) | Strong Baseline | Significantly Worse |
| GSM8K, $BL=L=256$ (full-diff) | acc @ ~12 NFEs | ~50% | ≤30% | ≤30% |
| MATH-500, $BL=32$ | acc @ ~25 NFEs (β-scaled) | ~20% | ~10% | — |
| MATH-500, $BL=256$ | full-diff Pareto | Leads throughout | Significant Drop | Significant Drop |
| GSM8K, expert steering | acc @ mid-high NFE | ~80% (Matches best semi-AR) | — | — |
| Model Transfer LLaDA→Dream | GSM8K acc | Near direct training on Dream | Baseline | — |
| Length Transfer $L=256\to512$ | GSM8K acc | Minimal drop | Significant Baseline Drop | — |

### Ablation Study
| Configuration | Key Observation | Explanation |
|------|---------|------|
| Bernoulli vs. Dynamic Plackett-Luce | Similar performance | Bernoulli chosen for simpler implementation and closed-form likelihood |
| Input $c_t^k$ vs. Top-50 Probs | $c_t^k$ slightly better | Finer-grained uncertainty did not yield gains |
| Input $c_t^k$ vs. Hidden state $\bm h_t^k$ | Hidden states significantly worse + unstable | Key signals reside in confidence after unembedding projection |
| Zeroing $t$, $\bm m_t$, or both | Accuracy drops in all; zeroing mask has largest impact | Both time and mask vectors contribute to decision-making |
| Multiplicative vs. Additive Reward ($\alpha=1$) | Additive collapses to one-step reveal/error | Multiplicative reward prevents reward hacking |
| Math Train → HumanEval/MBPP Transfer | Significant drop | Retraining on KodCode-RL-10K recovers performance; domain diversity is required |

### Key Findings
- **Redefining the Optimal Frontier**: In semi-AR, Fast-dLLM is near-optimal, and the learned policy matches it. However, in full-diffusion, where heuristics underperform random unmasking, Ours is among the few solutions that still scale performance with increased NFE.
- **Qualitative Policy Behavior**: In semi-AR, Fast-dLLM prefers "intensive computation on early blocks and adjacent revealing," whereas the learned policy distributes budget more evenly and "slows down" when generating numerical answers. In full-diffusion with expert steering, the policy learns left-to-right generation, avoiding "reverse decoding" artifacts caused by LLaDA's padding token confidence noise.
- **$\alpha$ control is coarse; testing scale is better**: Directly tuning $\alpha$ during training often causes collapse into a few discrete policies. Scaling Bernoulli parameters via $\min(1, \beta s_t^k)$ during inference allows for smooth traversal of the accuracy-NFE Pareto frontier.
- **Fastest policy ($\alpha=10$) has poor transferability**: While best on LLaDA, it collapses to Fast-dLLM levels on Dream, suggesting steep rewards cause overfitting to model-specific confidence patterns.

## Highlights & Insights
- **Using pretrained dLLMs as Environments**: Unlike methods that co-train the policy and LLM (d1, DCOLT, DiFFPO), Ours features minimal parameters, leaves the base model frozen, and is low-cost to train. It effectively acts as a "plug-and-play" lightweight accelerator for any open-source dLLM.
- **Multiplicative Reward as a Firewall**: With sparse rewards and speed incentives, the "fast but wrong" trap easily misleads policies. Multiplying penalty terms into the correctness term is a versatile strategy transferable to other "accuracy + efficiency" RL tasks like early-exit or adaptive depth.
- **"Confidence is Sufficient" as a General Lesson**: Following early-exit research, this work confirms that for unmasking, confidence-based signals outperform hidden-state-based signals—the maximum value after vocabulary projection sufficiently compresses semantic uncertainty.
- **Bernoulli + Max-Fallback**: This combination maintains closed-form likelihood while avoiding infinite loops caused by "all-zero" actions, serving as a robust engineering trick.
- **$\beta$-scaling for Deployment**: Using $\min(1,\beta s_t^k)$ at inference time to slide along the accuracy-NFE frontier is more efficient than retraining with different $\alpha$ values, providing "one policy, multiple gears."
- **Forcing $\tau=0$ during training**: Attributing all group variance to policy actions rather than dLLM stochasticity reduces credit assignment noise in GRPO, a critical yet often overlooked engineering decision when training RL with diffusion models.

## Limitations & Future Work
- **Training Control Granularity**: $\alpha$ is not smooth, and expert steering can increase instability; better KL control or annealing strategies are needed.
- **Domain Transfer is not Free**: Significant performance drops occur when moving from math to code tasks (HumanEval, MBPP), requiring retraining on domain-specific corpora.
- **Scope Limited to Unmasking Order**: Remasking, dynamic generation length, and KV cache optimizations are orthogonal and not yet integrated into the MDP.
- **Limited Interpretability**: While qualitative differences are observed, formal explanations of "why certain revealing patterns are optimal" are still lacking.
- **Dependency on Calibration**: Since input is $c_t^k$, policy performance is bound by the base model's calibration; issues like padding token noise or overconfidence in the dLLM will directly impact the policy.
- **From-Scratch Training vs. Fine-tuning**: Omitting KL regularization bypasses "imitation then RL" stages, which avoids certain biases but misses the potential benefits of warm-starting from heuristics.

## Related Work & Insights
Heuristic sampling (Fast-dLLM and variants by Ben-Hamu, Kim, Wei, etc.) proved that confidence signals are vital for accelerating dLLMs. RL post-training routes (d1, DiffuCoder, DiFFPO, DCOLT) mostly bind the policy to the base model with a focus on reasoning capability. This paper aligns with concurrent work by Hong et al. 2025b—both use GRPO for separate unmasking policies—but this work's Bernoulli formulation allows for truly variable step sizes, whereas others maintain fixed steps.
From a broader perspective, this line of work extends "Adaptive Computation" (Graves, Bengio, etc.) to diffusion language models, suggesting that "learning the inference path" can be decoupled from "learning the inference itself," leading to generalizable accelerators.
Furthermore, this work is complementary to KV cache, speculative decoding, and distilled decoders; stacking learned unmasking policies with these engineering optimizations should further push the throughput limits of dLLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] d2: Improving Reasoning in Diffusion Language Models via Trajectory Likelihood Estimation](d2_improving_reasoning_in_diffusion_language_models_via_trajectory_likelihood_es.md)
- [\[NeurIPS 2025\] Reinforcing the Diffusion Chain of Lateral Thought with Diffusion Language Models](../../NeurIPS2025/reinforcement_learning/reinforcing_the_diffusion_chain_of_lateral_thought_with_diffusion_language_model.md)
- [\[ACL 2026\] d-TreeRPO: Towards More Reliable Policy Optimization for Diffusion Language Models](../../ACL2026/reinforcement_learning/d-treerpo_towards_more_reliable_policy_optimization_for_diffusion_language_model.md)
- [\[ICML 2026\] Break the Block: Dynamic-size Reasoning Blocks for Diffusion Large Language Models via Monotonic Entropy Descent with Reinforcement Learning](break_the_block_dynamic-size_reasoning_blocks_for_diffusion_large_language_model.md)
- [\[NeurIPS 2025\] MMaDA: Multimodal Large Diffusion Language Models](../../NeurIPS2025/reinforcement_learning/mmada_multimodal_large_diffusion_language_models.md)

</div>

<!-- RELATED:END -->

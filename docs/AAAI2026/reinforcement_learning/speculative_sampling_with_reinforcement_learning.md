---
title: >-
  [Paper Note] Speculative Sampling with Reinforcement Learning
description: >-
  [AAAI 2026][Reinforcement Learning][Speculative Sampling] This paper proposes Re-SpS, the first framework to formulate the draft tree hyperparameter optimization of Speculative Sampling (SpS) as an MDP and solve it via r…
tags:
  - "AAAI 2026"
  - "Reinforcement Learning"
  - "Speculative Sampling"
  - "LLM Inference Acceleration"
  - "Draft Tree Optimization"
  - "PPO"
date: 2026-05-08
content_hash: d6ada54ff2947f34
---

# Speculative Sampling with Reinforcement Learning

**Conference**: AAAI 2026
**arXiv**: [2601.12212](https://arxiv.org/abs/2601.12212)
**Code**: [github.com/wmd3i/ReSpS](https://github.com/wmd3i/ReSpS)
**Area**: Reinforcement Learning
**Keywords**: Speculative Sampling, LLM Inference Acceleration, Reinforcement Learning, Draft Tree Optimization, PPO

## TL;DR

This paper proposes Re-SpS, the first framework to formulate the draft tree hyperparameter optimization of Speculative Sampling (SpS) as an MDP and solve it via reinforcement learning. Through two key designs—feature reuse and action caching—Re-SpS achieves up to 1.12× additional speedup over EAGLE-3 without any loss in output fidelity.

## Background & Motivation

### State of the Field

Inference latency in large language models (LLMs) is a core bottleneck for practical deployment, rooted in the token-by-token sequential generation of autoregressive decoding. Speculative Sampling (SpS) reduces the number of target model forward passes through a "draft-then-verify" paradigm—a small draft model proposes candidate tokens, which the large target model validates in a single forward pass—making it one of the most effective lossless acceleration methods available.

### Limitations of Prior Work

State-of-the-art methods such as EAGLE-2/EAGLE-3 have introduced tree-structured drafts to explore multiple candidate continuations in parallel, with dynamic pruning via confidence-based ranking. However, these methods share a critical limitation: **the hyperparameters governing the overall draft tree structure—total token count $TT$, depth $d$, and expansion factor $k$—remain static and manually tuned throughout decoding.** Different contexts and tasks require different levels of speculative aggressiveness: simple contexts can tolerate bold speculation (large depth), while complex ones require conservative speculation (small depth, large top-$k$). Static configurations cannot adapt to such variation.

### Root Cause

The dynamic selection of SpS hyperparameters is inherently a sequential decision-making problem: at each decoding step, an agent observes the current generation context, selects a hyperparameter configuration, and receives an immediate reward in terms of accepted tokens per unit time. This maps naturally onto the MDP framework.

### Core Problem

Naively invoking an RL policy at every decoding step introduces two sources of overhead:

**State representation cost**: Encoding the context with SentenceBERT incurs approximately 5–15 ms per step, which may negate the acceleration gains.

**Policy inference cost**: The cumulative cost of per-step policy network forward passes is substantial, given that a single response may involve 50–100+ decoding steps.

Together, these overheads risk making the overall inference slower than the non-RL baseline.

## Method

### Overall Architecture

Re-SpS builds on top of EAGLE-3, replacing its static hyperparameters with the dynamic decisions of an RL agent. The overall pipeline proceeds as follows: the target model generates hidden states → these are aggregated into a state vector $s_t$ → the RL policy outputs hyperparameters $(TT, d, k)$ → the draft model constructs tree-structured candidates → the target model performs verification → rewards are computed → the policy is updated.

### Key Designs

#### 1. **MDP Formulation**: Formalizing Hyperparameter Selection as a Markov Decision Process

- **State space $\mathcal{S}$**: A feature representation of the current generation context.
- **Action space $\mathcal{A}$**: Discrete hyperparameter combinations $\{(TT, d, k) \mid TT \in \mathcal{S}_{TT},\ d \in \mathcal{S}_d,\ k \in \mathcal{S}_k\}$, where each dimension takes a finite set of predefined integer values.
- **Reward function $R$**: Instantaneous generation throughput $r_t = \frac{\text{accepted tokens}}{\text{elapsed time (seconds)}}$, directly aligned with the latency minimization objective.
- **Transition function $\xi$**: Implicitly determined by the draft tree construction and speculative decoding process; deterministic in nature.

#### 2. **Efficient State Representation — Feature Reuse Mechanism**

**Core Idea**: Rather than employing an additional encoder, Re-SpS directly reuses the hidden states already computed by the target LLM within the EAGLE-3 draft model.

$$s_t = [h_{LM}^{(h,m,l)}]$$

where $h_{LM}^{(h,m,l)}$ is the concatenation of hidden states from three strategically selected layers of the target model—high ($h$), middle ($m$), and low ($l$)—capturing syntactic, semantic, and task-specific information respectively.

**Design Motivation**: These features are already part of the EAGLE-3 architecture (used to drive the draft model), and thus require no additional computational overhead. The key difference from EAGLE-3 is that, whereas EAGLE-3 fuses the three layers into a single vector via a fully-connected layer, Re-SpS directly concatenates them to avoid the associated computation.

#### 3. **Multi-Step Action Persistence — Action Caching Mechanism**

**Core Idea**: The hyperparameter configuration $(TT, d, k)$ selected by the RL policy is **cached and reused for $N$ steps** ($N=10$ during training, $N=30$ during inference), avoiding per-step policy network invocations.

The reward signal is averaged over the caching interval:

$$r_{avg} = \frac{1}{N} \sum_{i=1}^{N} \frac{\text{accepted\_tokens}_i}{\text{elapsed\_time}_i}$$

**Design Motivation**: By exploiting the Markov property, the averaged reward naturally captures temporal dynamics and performance effects without requiring complex multi-step state histories. This strikes a favorable balance between adaptability and efficiency.

### Loss & Training

PPO (Proximal Policy Optimization) serves as the backbone RL algorithm, and a maximum-entropy variant is also explored:

**Standard PPO Objective**:

$$L^{PPO}(\theta) = \mathbb{E}_t[\min(r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)]$$

**Maximum-Entropy PPO** (with entropy regularization to encourage exploration):

$$L^{MAX\text{-}ENT}(\theta) = L^{PPO}(\theta) + \beta_H \mathbb{E}_t[H(\pi_\theta(\cdot|s_t))]$$

Training details:

- Policy network: two-layer MLP with 128 hidden units (separate actor and critic).
- Training data: a 4,000-question subset from ShareGPT and UltraChat200K, covering multiple domains.
- Entropy coefficient $\beta_H = 0.1$.
- **Lossless fidelity**: The target model verification mechanism inherited from EAGLE-3 guarantees byte-identical outputs with greedy decoding.

## Key Experimental Results

### Main Results

| Backbone | Method | MT-Bench | HumanEval | GSM8K | Alpaca | CNN/DM | Avg. |
|---------|------|----------|-----------|-------|--------|--------|------|
| LLaMA 3.1-8B | EAGLE-3 | 3.39× | 3.65× | 3.52× | 3.67× | **2.96×** | 3.44× |
| LLaMA 3.1-8B | **Re-SpS** | **3.43×** | **3.89×** | **3.62×** | **3.90×** | 2.87× | **3.54×** |
| Vicuna-13B | EAGLE-3 | 3.75× | 4.28× | 3.85× | 3.76× | **3.35×** | 3.80× |
| Vicuna-13B | **Re-SpS** | **3.76×** | **4.64×** | **3.99×** | **3.99×** | 3.24× | **3.92×** |
| LLaMA 3.3-70B | EAGLE-3 | 4.35× | 4.87× | 4.74× | 4.77× | **4.09×** | 4.46× |
| LLaMA 3.3-70B | **Re-SpS** | **4.47×** | **5.45×** | **5.13×** | **5.34×** | 4.03× | **4.88×** |

All $p$-values $< 10^{-4}$ (Wilcoxon signed-rank test), indicating highly statistically significant differences.

### Ablation Study

| Model | Policy Configuration | Avg. Speedup vs. EAGLE-3 | Unique Actions |
|------|---------|---------------------|-----------|
| LLaMA 3.1-8B | Standard PPO + Text Embedding | 1.044× | 3 |
| LLaMA 3.1-8B | Standard PPO + Feature Vector | **1.049×** | 5 |
| LLaMA 3.1-8B | Max-Entropy PPO + Text Embedding | 1.017× | 8 |
| LLaMA 3.1-8B | Max-Entropy PPO + Feature Vector | 1.025× | **18** |
| Vicuna-13B | Standard PPO + Text Embedding | 1.006× | 8 |
| Vicuna-13B | Standard PPO + Feature Vector | 1.028× | 3 |
| Vicuna-13B | Max-Entropy PPO + Feature Vector | **1.033×** | **15** |

### Key Findings

1. **Larger models yield greater gains**: The 70B model achieves an average 1.06× additional speedup over EAGLE-3, compared to only 1.03× for the 8B model, indicating greater potential for dynamic hyperparameter adaptation at larger scales.
2. **Feature reuse outperforms external encoding**: The Feature Vector consistently outperforms Text Embedding across all configurations, validating the zero-cost hidden state reuse strategy.
3. **Caching interval length**: As the caching window grows from 1 to 50 steps, inference latency decreases substantially and generation throughput improves continuously, with an optimal point near 30 steps.
4. **Max-Entropy PPO promotes action diversity**: Although it does not always achieve the highest speedup ratio, it produces significantly more unique actions than standard PPO (18 vs. 5), yielding a more robust and adaptive policy.
5. **Slight regression on CNN/DM** (0.98×): Attributed to the need to increase the maximum sequence length (2048→2200) to avoid KV cache overflow, which introduces additional overhead.

## Highlights & Insights

- **First application of RL to SpS hyperparameter optimization**, opening a new research direction in the speculative sampling literature.
- **Zero-cost state representation** is a key design highlight: directly reusing the target model's existing hidden states entirely eliminates encoder overhead.
- **The action caching strategy is simple yet effective**: a straightforward "cache for $N$ steps" mechanism reduces RL overhead by an order of magnitude with minimal loss of adaptability.
- **Byte-identical output fidelity** is preserved, satisfying a hard requirement for practical deployment.
- Achieves an **overall speedup of up to 5.45×** on the largest model (70B), including the acceleration contributed by EAGLE-3 itself.

## Limitations & Future Work

- Slight regression on CNN/DailyMail due to sequence length constraints; long-document scenarios warrant further optimization.
- Validation is currently limited to greedy decoding (temperature = 0); performance under stochastic sampling remains unexplored.
- The action space is a predefined discrete grid; continuous action spaces or finer-grained hyperparameter control may yield larger gains.
- Evaluation is confined to the EAGLE-3 architecture; transferability to other speculative sampling methods (e.g., Medusa, C2T) remains to be verified.
- Training must be performed on the target hardware; transferring to different GPU configurations may necessitate retraining.

## Related Work & Insights

- Unlike heuristic adaptive methods such as SpecDec++, DySpec, and OPT-Tree, Re-SpS is the first **data-driven learning approach** in this space.
- This work offers a new perspective on RL-based optimization for other LLM inference acceleration techniques, including KV cache management and batching schedulers.
- The action caching idea is generalizable to other RL application scenarios involving frequent decision-making with high per-step costs.
- BanditSpec and MetaSD employ multi-armed bandits to select policies or models but cannot dynamically adjust draft tree structural hyperparameters.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First to apply RL to SpS hyperparameter optimization, though the core techniques (PPO, feature reuse) are relatively standard.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Five benchmarks, three model scales, complete ablations, and statistical significance testing.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear motivation, rigorous methodological derivation, and rich figures and tables.
- **Value**: ⭐⭐⭐⭐ — Consistent but incremental acceleration gains (1.03–1.06× over EAGLE-3); practical deployment value depends on scale of production use.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Mean-Field Sampling for Cooperative Multi-Agent Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/mean-field_sampling_for_cooperative_multi-agent_reinforcement_learning.md)
- [\[ICLR 2026\] RuleReasoner: Reinforced Rule-based Reasoning via Domain-aware Dynamic Sampling](../../ICLR2026/reinforcement_learning/rulereasoner_reinforced_rule-based_reasoning_via_domain-aware_dynamic_sampling.md)
- [\[NeurIPS 2025\] Distribution Learning Meets Graph Structure Sampling](../../NeurIPS2025/reinforcement_learning/distribution_learning_meets_graph_structure_sampling.md)
- [\[NeurIPS 2025\] Learning from Demonstrations via Capability-Aware Goal Sampling](../../NeurIPS2025/reinforcement_learning/learning_from_demonstrations_via_capability-aware_goal_sampling.md)
- [\[AAAI 2026\] Vision-Language Reasoning for Geolocalization: A Reinforcement Learning Approach](vision-language_reasoning_for_geolocalization_a_reinforcement_learning_approach.md)

</div>

<!-- RELATED:END -->

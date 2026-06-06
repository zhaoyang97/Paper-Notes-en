---
title: >-
  [Paper Note] Blending Supervised and Reinforcement Fine-Tuning with Prefix Sampling
description: >-
  [ICML 2026][LLM Reasoning][Post-training] This paper proposes Prefix-RFT, which constructs hybrid trajectories by sampling prefixes from expert demonstrations and concatenating model-generated continuations. This approac…
tags:
  - "ICML 2026"
  - "LLM Reasoning"
  - "Post-training"
  - "Supervised Fine-Tuning"
  - "Reinforcement Fine-Tuning"
  - "Prefix Sampling"
  - "Mathematical Reasoning"
date: 2026-05-08
content_hash: 6ed3322ed6a603fb
---

# Blending Supervised and Reinforcement Fine-Tuning with Prefix Sampling

**Conference**: ICML 2026  
**arXiv**: [2507.01679](https://arxiv.org/abs/2507.01679)  
**Code**: None  
**Area**: LLM Reasoning  
**Keywords**: Post-training, Supervised Fine-Tuning, Reinforcement Fine-Tuning, Prefix Sampling, Mathematical Reasoning  

## TL;DR
This paper proposes Prefix-RFT, which constructs hybrid trajectories by sampling prefixes from expert demonstrations and concatenating model-generated continuations. This approach injects knowledge guidance from SFT while maintaining the goal-oriented optimization of RFT, significantly outperforming independent SFT, RFT, and existing hybrid methods on mathematical reasoning tasks.

## Background & Motivation

**Background**: LLM post-training primarily follows two paradigms: Supervised Fine-Tuning (SFT), which injects knowledge by mimicking expert demonstrations, and Reinforcement Fine-Tuning (RFT), which enhances task performance through trial-and-error exploration and reward signals. In practice, a two-stage pipeline of SFT followed by RFT is commonly adopted.

**Limitations of Prior Work**: SFT is essentially behavior cloning; while it teaches correct problem-solving patterns, it suffers from issues in generalization and robustness. RFT directly optimizes task performance, but its learning signals are sparse, it can lead to unexpected behaviors like language mixing, and its performance is highly dependent on the capability ceiling of the initial policy—recent studies have questioned whether RL can truly surpass the inherent capability ceiling of a model.

**Key Challenge**: SFT provides dense supervision but over-constrains the solution space, while RFT encourages exploration but is limited by the current policy's capability. Simple joint training of "RL + SFT Loss" can backfire as demonstration gradients dominate RFT gradients. Conversely, the sequential two-stage approach (SFT→RFT) cannot dynamically balance the two learning signals during training.

**Goal**: Design a unified framework that organically integrates SFT's process supervision with RFT's goal-oriented optimization during the RFT training process, achieving a dynamic balance between knowledge injection and capability enhancement.

**Key Insight**: The authors first establish a unified perspective on SFT and RFT—both gradient updates apply weighted gradients to token log-probabilities, differing only in the weight assignment. Based on this unified framework, the two paradigms can be naturally fused by designing appropriate weight allocations.

**Core Idea**: Sample prefixes from expert demonstrations, allow the model to continue generation from the prefix position, and concatenate these into hybrid trajectories. These, along with standard rollouts, are used for PPO updates, utilizing trajectory-level advantage to automatically regulate the learning intensity of demonstration data.

## Method

### Overall Architecture
Given a prompt $x$ and an expert demonstration $y^*$, Prefix-RFT first generates $N-1$ standard rollouts using the current policy $\pi_{\theta_{\text{old}}}$. For the $N$-th trajectory, a prefix $y^*_{<L}$ is intercepted from $y^*$, and the model generates a continuation $y_{\geq L}$ to form a hybrid trajectory $y^{(N)}$. All $N$ trajectories are used to estimate advantages and perform PPO updates, where prefix tokens and continuation tokens use the same PPO weights $\mathcal{W}_{i,t}^{\text{PPO}} = \mathbb{I}_{\text{clip}}(r_t, \hat{A}_t) \hat{A}_t r_t$. This avoids additional rollout overhead by simply replacing one standard rollout with a hybrid trajectory.

### Key Designs

1. **Prefix Sampling and Hybrid Trajectory Construction**:

    - **Function**: Organically embeds offline demonstration data into the online RFT training process.
    - **Mechanism**: Extracts the first $L$ tokens of an expert demonstration as a prefix, and the model continues writing from the $L$-th position. Although the prefix originates from an offline policy, its gradient weight is determined by the advantage of the entire hybrid trajectory: if the prefixed trajectory receives a higher reward, the prefix is naturally reinforced; otherwise, it is suppressed. Compared to SFT's forced imitation of the entire sequence, prefix sampling grants the model "constrained autonomy"—starting in the direction guided by the expert while still exploring superior continuation paths.
    - **Design Motivation**: Addresses the low exploration efficiency of pure RFT and its inability to break the policy's capability ceiling, while avoiding SFT's over-constraint of the solution space.

2. **Entropy-based Clipping**:

    - **Function**: Prevents gradients from offline demonstrations from dominating the optimization process.
    - **Mechanism**: Only the top-$k$% (default 20%) highest-entropy tokens in the prefix are retained for gradient updates, while the advantages of other tokens are set to zero. Low-entropy tokens either already match the current policy (small learning signal) or represent high-confidence deviations (which would lead to drastic overwriting updates); high-entropy tokens correspond to positions where the model is most uncertain and possess the highest learning value.
    - **Design Motivation**: When the gap between the offline policy $\pi_{\text{off}}$ and the current policy is large, the probability of prefix tokens can be extremely low, making the gradient magnitude much larger than the RFT gradient. Without constraints, this degrades into simple SFT.

3. **Cosine Decay Scheduler**:

    - **Function**: Controls the dynamic change of prefix length to achieve a curriculum-style transition from SFT to RFT.
    - **Mechanism**: The prefix length is determined by $L = \lfloor l \cdot |y^*| \rfloor$, where $l \sim U(\text{low}, \text{high})$. Early in training, low is close to high (long prefixes, close to SFT); as training progresses, low follows a cosine decay toward zero (short prefixes, close to RFT). This mitigates the position bias caused by uniform sampling (where concluding segments have low sampling probability) and naturally implements curriculum learning.
    - **Design Motivation**: In uniform sampling, the model is naturally exposed more to skills at the beginning of demonstrations rather than the end (e.g., summarization, reasoning closure). Furthermore, as the model grows stronger in later stages, dependence on demonstrations should be reduced.

## Key Experimental Results

### Main Results (Qwen2.5-Math-7B)

| Method | AIME24 | AIME25 | AMC | MATH-500 | Minerva | Olympiad | Math Avg. |
|------|--------|--------|-----|----------|---------|----------|----------|
| Base | 11.5 | 4.9 | 31.3 | 43.6 | 7.4 | 15.6 | 19.0 |
| SFT | 22.2 | 22.3 | 52.8 | 82.6 | 40.8 | 43.7 | 44.1 |
| RFT | 25.1 | 15.3 | 62.0 | 84.4 | 39.3 | 46.8 | 45.5 |
| SFT+RFT | 25.8 | 23.1 | 62.7 | 87.2 | 39.7 | 50.4 | 48.2 |
| RL w/ SFT Loss | 19.5 | 16.4 | 49.7 | 80.4 | 34.9 | 39.4 | 40.1 |
| LUFFY | 29.4 | 23.1 | 65.6 | 87.6 | 37.5 | 57.2 | 50.1 |
| ReLIFT | 28.2 | 20.1 | 64.9 | 87.4 | 33.8 | 52.5 | 47.8 |
| **Prefix-RFT** | **31.8** | **26.4** | **68.2** | **88.4** | **40.3** | **55.7** | **51.8** |

### Ablation Study (Qwen2.5-Math-1.5B)

| Configuration | AIME24 | AIME25 | AMC | MATH-500 | Avg. | Description |
|------|--------|--------|-----|----------|------|------|
| SFT | 11.7 | 13.2 | 37.8 | 70.6 | 31.9 | Pure SFT baseline |
| RFT | 11.8 | 7.7 | 40.2 | 61.8 | 30.0 | Pure RFT baseline |
| Prefix-RFT (full) | 17.7 | 17.1 | 50.5 | 81.4 | 41.1 | Complete method |
| Data Volume 10% (4.5k) | 17.8 | 15.9 | 49.7 | 79.0 | 40.8 | Only 0.3 drop |
| Data Volume 1% (0.45k) | 15.2 | 11.8 | 46.3 | 76.0 | 37.6 | Exceeds baseline with 99% less data |
| 1.5B Generator | 15.9 | 12.6 | 47.7 | 79.0 | 39.8 | Effective with weak generator |
| 32B Generator | 18.1 | 15.3 | 50.9 | 81.2 | 40.6 | Quality has minor impact |

### Key Findings
- Prefix-RFT consistently outperforms all baselines across 6 mathematical reasoning and 3 general reasoning benchmarks, with a math average of 51.8 vs. LUFFY 50.1 and RFT 45.5.
- Pass@2048 experiments indicate that Prefix-RFT is the only method that truly raises the model's reasoning capability ceiling, improving by 6.67 percentage points over the base model on AIME24 and AIME25.
- Top-20% entropy clipping significantly outperforms top-50%/80%/random-20%/bottom-20%, validating the necessity of high-entropy token selection.
- The cosine decay scheduler is superior to uniform sampling; training dynamics show prefix advantages gradually shrinking—the model automatically transitions from demonstration dependence to autonomous exploration.
- The method is robust to both demonstration volume and quality: reducing data by 99% only leads to a 3.5-point drop, and demonstrations generated by a small 1.5B model achieve performance close to that of DeepSeek-R1.

## Highlights & Insights
- **Simple but Profound Unified Perspective**: The structural consistency of SFT and RFT gradients (weighted log-prob gradients), differing only in weight settings, provides a theoretical foundation for hybrid methods. From this view, the design of Prefix-RFT becomes natural and elegant—obviating the need for extra loss functions or complex multi-stage scheduling.
- **Advantage-Driven Adaptive Learning**: The learning intensity of the prefix is automatically regulated by trajectory-level advantage. For difficult problems, prefix advantage is high, allowing the model to learn more from demonstrations; for easy problems, advantage is low, and the model relies on its own exploration. This instance-level dynamic balance requires no manual weight settings.
- **High-Entropy Token Filtering**: Using information-theoretic metrics to filter the gradient contributions of offline data is a general technique for off-policy training stability, transferable to other mixed online/offline learning scenarios.

## Limitations & Future Work
- Experiments focused mainly on verifiable reasoning tasks (math, code); performance in open-ended generation and noisy reward scenarios remains unverified.
- When multiple candidate demonstrations are available for each prompt, a simple random selection may not be optimal; systematic demonstration selection strategies are left for the future.
- Optimal values for the entropy clipping ratio (20%) and scheduler parameters may vary by task/model; a unified hyperparameter search strategy has not yet been explored.
- Code generation experiments were only a preliminary validation (Qwen3-1.7B); generalization across larger scales and more domains requires further confirmation.

## Related Work & Insights
- **LUFFY** (Yan et al., 2025): Mixes complete offline demonstrations into rollouts for RFT without prefix truncation.
- **UFT** (Liu et al., 2025b): Also samples prefixes but uses SFT loss for prefixes and RFT loss for continuations, employing a static small weight.
- **ReLIFT** (Ma et al., 2025): Alternates SFT and RFT stages, with SFT focusing on difficult problems RFT cannot solve.
- The advantages of Ours lie in: replacing multiple loss function designs with unified weights (PPO advantage), replacing static weights with entropy clipping, and replacing manual stages with cosine decay—making it simpler and easier to integrate into existing RFT workflows.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Small Language Models for Efficient Agentic Tool Calling: Outperforming Large Models with Targeted Fine-tuning](../../AAAI2026/llm_reasoning/small_language_models_for_efficient_agentic_tool_calling_outperforming_large_mod.md)
- [\[ICLR 2026\] The Path of Least Resistance: Guiding LLM Reasoning Trajectories with Prefix Consensus](../../ICLR2026/llm_reasoning/the_path_of_least_resistance_guiding_llm_reasoning_trajectories_with_prefix_cons.md)
- [\[ICLR 2026\] Fine-R1: Make Multi-modal LLMs Excel in Fine-Grained Visual Recognition by Chain-of-Thought Reasoning](../../ICLR2026/llm_reasoning/fine-r1_make_multi-modal_llms_excel_in_fine-grained_visual_recognition_by_chain-.md)
- [\[ICML 2026\] ResRL: Boosting LLM Reasoning via Negative Sample Projection Residual Reinforcement Learning](resrl_boosting_llm_reasoning_via_negative_sample_projection_residual_reinforceme.md)
- [\[NeurIPS 2025\] Sampling-Efficient Test-Time Scaling: Self-Estimating the Best-of-N Sampling in Early Decoding](../../NeurIPS2025/llm_reasoning/sampling-efficient_test-time_scaling_self-estimating_the_best-of-n_sampling_in_e.md)

</div>

<!-- RELATED:END -->

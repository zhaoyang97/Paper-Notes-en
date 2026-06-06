---
title: >-
  [Paper Note] Parallel Test-Time Scaling for Latent Reasoning Models
description: >-
  [ACL 2026 Main Conference][LLM Reasoning][Test-time scaling] This paper introduces parallel test-time scaling (TTS) to latent reasoning models for the first time. It proposes two stochastic sampling strategies based on u…
tags:
  - "ACL 2026 Main Conference"
  - "LLM Reasoning"
  - "Test-time scaling"
  - "latent reasoning"
  - "stochastic sampling"
  - "reward models"
  - "parallel inference"
date: 2026-05-08
content_hash: 03d4ace24432aeea
---

# Parallel Test-Time Scaling for Latent Reasoning Models

**Conference**: ACL 2026 Main Conference  
**arXiv**: [2510.07745](https://arxiv.org/abs/2510.07745)  
**Code**: None  
**Area**: LLM Reasoning  
**Keywords**: Test-time scaling, latent reasoning, stochastic sampling, reward models, parallel inference

## TL;DR

This paper introduces parallel test-time scaling (TTS) to latent reasoning models for the first time. It proposes two stochastic sampling strategies based on uncertainty estimation theory (MC-Dropout and Additive Gaussian Noise) and a Latent Reward Model (LatentRM) trained via step-level contrastive learning. These components enable models reasoning in continuous vector spaces to achieve stable performance gains through parallel sampling and aggregation.

## Background & Motivation

**Background**: Test-time scaling (TTS) is a critical method for enhancing the reasoning capabilities of LLMs. Parallel TTS converts additional inference computation directly into stronger performance by generating multiple reasoning paths and aggregating results (e.g., majority voting, best-of-N, beam search). Currently, these methods all rely on token-level sampling mechanisms (e.g., top-k, nucleus sampling).

**Limitations of Prior Work**: Recently emerged latent reasoning paradigms (e.g., COCONUT, CODI, CoLaR) shift the reasoning process from token space to a continuous vector space, which is more compact and efficient. However, they **cannot directly utilize parallel TTS**. There are two reasons: (1) Continuous vector spaces lack explicit probability distributions and sampling mechanisms; (2) There are no token-level probability signals for evaluating and aggregating reasoning trajectories.

**Key Challenge**: While latent reasoning offers inherent advantages in inference efficiency, its lack of parallel scaling capability limits its reasoning quality. Introducing controllable randomness into the continuous space and designing effective trajectory evaluation mechanisms are the two primary obstacles to unlocking parallel TTS for latent reasoning models.

**Goal**: To design the two core components—sampling and aggregation—for latent reasoning models, enabling them to benefit from parallel TTS similarly to token-based models.

**Key Insight**: Starting from uncertainty estimation theory, the authors decompose the sampling problem into two sources of uncertainty: epistemic and aleatoric, and design corresponding sampling strategies. For the aggregation problem, a specialized scoring model is trained to replace token-level probability signals.

**Core Idea**: Use MC-Dropout (epistemic uncertainty) and Additive Gaussian Noise (aleatoric uncertainty) to generate diverse reasoning trajectories in the latent space, and evaluate/guide trajectory aggregation using a step-level contrastive-trained LatentRM to achieve parallel test-time scaling for latent reasoning.

## Method

### Overall Architecture

Given an input question $\bm{x}$, the latent reasoning model autoregressively generates $T$ steps of latent vectors $\bm{h}_{1:T}$ in continuous space. Finally, it transitions to explicit token generation via an end-of-thinking token to produce the answer. During inference, this paper introduces randomness to generate $N$ different reasoning trajectories $\{\bm{h}^{(n)}\}_{n=1}^N$, and then aggregates the final answer through LatentRM scoring or majority voting.

### Key Designs

1.  **Monte Carlo Dropout (MC-Dropout) Sampling**:
    *   **Function**: Keeps dropout active during inference to generate diverse reasoning trajectories via random masking.
    *   **Mechanism**: Each forward pass uses a different dropout mask $m^{(n)} \sim \text{Bernoulli}(p)$, which is equivalent to sampling from a variational approximation of the model weight posterior distribution. Dropout is applied after the feed-forward layers of each Transformer block. Each sample yields a different weight configuration $\bm{\theta}^{(n)}$, producing distinct reasoning trajectories.
    *   **Design Motivation**: MC-Dropout captures epistemic uncertainty—uncertainty stemming from limited training data. It adaptively regulates noise intensity, producing greater exploration in areas where the model is uncertain.

2.  **Additive Gaussian Noise (AGN) Sampling**:
    *   **Function**: Adds isotropic Gaussian noise to each latent vector step to produce controlled random perturbations.
    *   **Mechanism**: At each reasoning step $t$, noise $\bm{\epsilon}_t^{(n)} \sim \mathcal{N}(0, \sigma^2 \mathbf{I})$ is sampled and added to the latent vector: $\bm{h}_t^{(n)*} = \bm{h}_t^{(n)} + \bm{\epsilon}_t^{(n)}$. The model continues reasoning based on the perturbed trajectory. Noise intensity is controlled solely by $\sigma$ and is independent of model parameters.
    *   **Design Motivation**: AGN simulates aleatoric uncertainty—inherent noise and ambiguity in the input space. It generates an isotropic "fireworks" exploration pattern, which is more robust than MC-Dropout in high-diversity settings.

3.  **Latent Reward Model (LatentRM)**:
    *   **Function**: Evaluates the quality of latent reasoning trajectories, providing scoring signals for best-of-N and beam search.
    *   **Mechanism**: A scoring head is added to the backbone of the latent reasoning model to map hidden states to scalar scores $r_t = g_{\bm{\phi}}(\bm{x}, \bm{h}_{1:t})$. During inference, the cumulative logits $\sum_t r_t$ serve as a proxy for trajectory quality. Training data is obtained through stochastic rollouts: executing $M$ random completions for each intermediate thought and calculating the accuracy as the quality label. The training objective employs a step-level softmax contrastive loss, comparing scores of all $N$ candidates at each step $t$ rather than using an independent binary classification loss.
    *   **Design Motivation**: Traditional PRMs rely on reasoning steps in token form and cannot handle latent thoughts in continuous vector form. Step-level contrastive loss provides a stronger relative comparison signal than BCE loss, yielding significantly better experimental results.

### Loss & Training

LatentRM training uses a step-level contrastive loss: $\mathcal{L} = -\sum_t \sum_{n=1}^N y_t^{(n)} \log p_t^{(n)}$, where $p_t^{(n)} = \frac{\exp(r_t^{(n)})}{\sum_{n'} \exp(r_t^{(n')})}$. Training data estimates the empirical accuracy of each thought through stochastic rollouts: $\tilde{y} = \frac{1}{M} \sum_m \mathbb{I}\{a_m = a^*\}$.

## Key Experimental Results

### Main Results

| Model | Dataset | Deterministic Baseline | Coverage@8 | Coverage@16 |
| :--- | :--- | :--- | :--- | :--- |
| Latent-SFT (1B) | GSM8K | 44.5% | 58.5% | 64.9% |
| Latent-SFT (1B) | MultiArith | 93.4% | 96.2% | 96.7% |
| RoT-4B | GSM8K | 37.5% | 39.4% | 39.7% |
| RoT-4B | MATH500 | 20.3% | 21.8% | 22.0% |

Aggregation Strategy Comparison (COCONUT, GSM-Test, N=32):

| Aggregation Strategy | GSM-Test | GSM-Hard |
| :--- | :--- | :--- |
| Majority Voting | 33.6% | 6.1% |
| Best-of-N + LatentRM | **35.4%** | **7.8%** |
| Beam Search + LatentRM | ~35% | ~7% |

### Ablation Study

| Configuration | GSM-Test | GSM-Hard | Description |
| :--- | :--- | :--- | :--- |
| Full LatentRM (Best-of-8) | 35.4% | 7.8% | Full model |
| w/o contrastive (using BCE) | 33.5% | 7.4% | Significant drop after removing contrastive loss |
| w/o stochastic rollouts | 30.7% | 6.0% | Stochastic rollout labeling is critical |
| Random scalar head | 28.9% | 5.8% | Worse than majority voting |

### Key Findings
*   MC-Dropout achieves higher coverage in most settings, particularly excelling at difficult problems (its directional drift makes it easier to reach correct regions far from the deterministic solution).
*   AGN is more robust in high-diversity settings, with coverage decaying more slowly; it is suitable for scenarios requiring high exploration.
*   t-SNE visualization reveals: MC-Dropout produces dense directional expansions ("directional drift"), while AGN produces isotropic radial scattering ("fireworks" pattern).
*   The step-level contrastive loss of LatentRM provides the largest contribution; performance drops significantly without it.
*   The performance gap between different models narrows as the number of samples increases.

## Highlights & Insights
*   The **uncertainty-driven sampling design** is elegant: decomposing the sampling problem into epistemic and aleatoric uncertainty, solved by MC-Dropout and AGN respectively, with the two exhibiting complementary geometric exploration patterns. This analytical framework is transferable to other search problems in continuous spaces.
*   **Design logic of LatentRM**: Obtaining thought-level labels via stochastic rollouts combined with step-level contrastive training solves the core dilemma of "scoring continuous vectors," which could be extended to other non-token intermediate representation evaluations.
*   The **"sweet spot" analysis of coverage vs. diversity** is insightful: excessively high or low diversity is detrimental; an optimal point exists.

## Limitations & Future Work
*   Experiments were mainly conducted on small models (GPT-2 124M, Llama-3.2-1B). The absolute performance of latent reasoning itself remains limited on difficult math problems (AIME) and PhD-level benchmarks (GPQA).
*   Both MC-Dropout and AGN require hyperparameter tuning (dropout rate and noise standard deviation), although heuristic ranges are provided.
*   LatentRM requires additional training, increasing deployment complexity.
*   The integration of sampling and aggregation into a Reinforcement Learning framework to optimize latent trajectories through iterative feedback was not explored.
*   The latent reasoning paradigm is still developing and exhibits a gap compared to token-based CoT on complex tasks.

## Related Work & Insights
*   **vs Self-Consistency (Majority Voting)**: Self-Consistency uses diverse sampling and voting in token space; this paper extends similar ideas to continuous latent space and achieves stronger aggregation than voting via LatentRM.
*   **vs COCONUT/CODI/CoLaR**: These are base latent reasoning models; this paper adds parallel TTS capabilities on top of them as an orthogonal enhancement.
*   **vs Stochastic Soft Thinking**: Soft Thinking operates in the token probability space (soft tokens are mixtures of token embeddings); this paper operates in the pure latent vector space, unconstrained by vocabulary structure.

## Rating
*   Novelty: ⭐⭐⭐⭐ Introducing parallel TTS to latent reasoning for the first time is a clear and valuable contribution, though the sampling methods themselves (dropout/noise) are not new.
*   Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple models, benchmarks, and sampling strategies, with rich visualization and ablation analysis.
*   Writing Quality: ⭐⭐⭐⭐⭐ Motivation is clear, structure is logical, and both theoretical derivations and experimental analyses are well-executed.
*   Value: ⭐⭐⭐⭐ Supplements the latent reasoning paradigm with important scaling capabilities, offering practical significance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Efficient Test-Time Scaling via Temporal Reasoning Aggregation](efficient_test-time_scaling_via_temporal_reasoning_aggregation.md)
- [\[ICLR 2026\] Efficient Test-Time Scaling for Small Vision-Language Models](../../ICLR2026/llm_reasoning/efficient_test-time_scaling_for_small_vision-language_models.md)
- [\[ACL 2026\] ReProbe: Efficient Test-Time Scaling of Multi-Step Reasoning by Probing Internal States of Large Language Models](reprobe_efficient_test-time_scaling_of_multi-step_reasoning_by_probing_internal_.md)
- [\[ICML 2026\] Stabilizing Recurrent Dynamics for Test-Time Scalable Latent Reasoning in Looped Language Models](../../ICML2026/llm_reasoning/stabilizing_recurrent_dynamics_for_test-time_scalable_latent_reasoning_in_looped.md)
- [\[ICML 2026\] Lookahead Sample Reward Guidance for Test-Time Scaling of Diffusion Models](../../ICML2026/llm_reasoning/lookahead_sample_reward_guidance_for_test-time_scaling_of_diffusion_models.md)

</div>

<!-- RELATED:END -->

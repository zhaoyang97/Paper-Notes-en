---
title: >-
  [Paper Note] Parallel Test-Time Scaling for Latent Reasoning Models
description: >-
  [ACL 2026 Main Conference][LLM Reasoning][test-time scaling] This paper is the first to introduce parallel test-time scaling (parallel TTS) into latent reasoning models. It proposes two uncertainty-theoretic stochastic s…
tags:
  - "ACL 2026 Main Conference"
  - "LLM Reasoning"
  - "test-time scaling"
  - "latent reasoning"
  - "stochastic sampling"
  - "reward model"
  - "parallel inference"
date: 2026-05-08
content_hash: a9d66a770c360b09
---

# Parallel Test-Time Scaling for Latent Reasoning Models

**Conference**: ACL 2026 Main Conference
**arXiv**: [2510.07745](https://arxiv.org/abs/2510.07745)
**Code**: None
**Area**: LLM Reasoning
**Keywords**: test-time scaling, latent reasoning, stochastic sampling, reward model, parallel inference

## TL;DR

This paper is the first to introduce parallel test-time scaling (parallel TTS) into latent reasoning models. It proposes two uncertainty-theoretic stochastic sampling strategies (MC-Dropout and additive Gaussian noise) along with a step-level contrastively trained latent reward model (LatentRM), enabling models that reason in continuous vector spaces to achieve consistent performance gains through parallel sampling and aggregation.

## Background & Motivation

**State of the Field**: Test-time scaling (TTS) is a key technique for enhancing LLM reasoning. Parallel TTS generates multiple reasoning paths and aggregates results (e.g., majority voting, best-of-N, beam search), directly converting additional inference compute into improved capability. All existing methods rely on token-level sampling mechanisms (e.g., top-k, nucleus sampling).

**Limitations of Prior Work**: Recently emerging latent reasoning paradigms (e.g., COCONUT, CODI, CoLaR) shift the reasoning process from token space to continuous vector space, offering greater compactness and efficiency, but they **cannot directly adopt parallel TTS**. Two reasons account for this: (1) continuous vector spaces lack explicit probability distributions and thus sampling mechanisms; (2) there are no token-level probability signals for evaluating and aggregating reasoning trajectories.

**Root Cause**: Latent reasoning has inherent efficiency advantages, but the absence of parallel scaling capability limits reasoning quality. Introducing controllable stochasticity into continuous spaces and designing effective trajectory evaluation mechanisms are the two key obstacles to enabling parallel TTS for latent reasoning models.

**Paper Goals**: Design sampling and aggregation components for latent reasoning models, enabling them to benefit from parallel TTS in the same manner as token-based models.

**Starting Point**: The authors draw on uncertainty estimation theory, decomposing the sampling problem into two sources of uncertainty—epistemic uncertainty and aleatoric uncertainty—and design corresponding sampling strategies for each. For aggregation, a dedicated scoring model is trained to replace token probability signals.

**Core Idea**: MC-Dropout (epistemic uncertainty) and additive Gaussian noise (aleatoric uncertainty) are used to generate diverse reasoning trajectories in latent space; a step-level contrastively trained LatentRM evaluates and guides trajectory aggregation, realizing parallel test-time scaling for latent reasoning.

## Method

### Overall Architecture

Given an input question $\bm{x}$, the latent reasoning model autoregressively generates $T$ latent vectors $\bm{h}_{1:T}$ in continuous space, then transitions to explicit token generation via an end-of-thinking token to produce the final answer. At inference time, stochasticity is introduced to generate $N$ distinct reasoning trajectories $\{\bm{h}^{(n)}\}_{n=1}^N$, which are aggregated via LatentRM scoring or majority voting to obtain the final answer.

### Key Designs

1. **Monte Carlo Dropout (MC-Dropout) Sampling**:

    - **Function**: Keeps dropout active during inference, generating diverse reasoning trajectories through random masks.
    - **Mechanism**: Each forward pass uses a different dropout mask $m^{(n)} \sim \text{Bernoulli}(p)$, equivalent to sampling from a variational approximation of the posterior over model weights. Dropout is applied after the feed-forward layer of each Transformer block. Each sample yields a different weight configuration $\bm{\theta}^{(n)}$, producing a distinct reasoning trajectory.
    - **Design Motivation**: MC-Dropout captures epistemic uncertainty—uncertainty arising from limited training data. It adaptively modulates noise intensity, producing greater exploration in regions where the model is uncertain.

2. **Additive Gaussian Noise (AGN) Sampling**:

    - **Function**: Adds isotropic Gaussian noise to the latent vector at each step, producing controlled stochastic perturbations.
    - **Mechanism**: At each reasoning step $t$, noise $\bm{\epsilon}_t^{(n)} \sim \mathcal{N}(0, \sigma^2 \mathbf{I})$ is sampled and added to the latent vector: $\bm{h}_t^{(n)*} = \bm{h}_t^{(n)} + \bm{\epsilon}_t^{(n)}$, and the model continues reasoning based on the perturbed trajectory. Noise intensity is controlled solely by $\sigma$, independent of model parameters.
    - **Design Motivation**: AGN simulates aleatoric uncertainty—noise and ambiguity inherent in the input space. It produces isotropic "firework"-style exploration patterns that are more robust than MC-Dropout in high-diversity settings.

3. **Latent Reward Model (LatentRM)**:

    - **Function**: Evaluates the quality of latent reasoning trajectories, providing scoring signals for best-of-N and beam search.
    - **Mechanism**: A scoring head is added on top of the latent reasoning model's backbone, mapping hidden states to a scalar score $r_t = g_{\bm{\phi}}(\bm{x}, \bm{h}_{1:t})$. At inference time, the cumulative logit sum $\sum_t r_t$ serves as a proxy for trajectory quality. Training data is obtained via stochastic rollouts: $M$ random completions are executed from each intermediate thought, with the resulting accuracy used as quality labels. The training objective employs a step-level softmax contrastive loss, which applies softmax comparison across all $N$ candidates' scores at each step $t$, rather than independent binary classification losses.
    - **Design Motivation**: Conventional PRMs rely on token-form reasoning steps and cannot handle continuous vector latent thoughts. The step-level contrastive loss provides stronger relative comparison signals than BCE loss, yielding clearly superior empirical results.

### Loss & Training

LatentRM training uses a step-level contrastive loss: $\mathcal{L} = -\sum_t \sum_{n=1}^N y_t^{(n)} \log p_t^{(n)}$, where $p_t^{(n)} = \frac{\exp(r_t^{(n)})}{\sum_{n'} \exp(r_t^{(n')})}$. Training data is constructed by using stochastic rollouts to estimate the empirical accuracy $\tilde{y} = \frac{1}{M} \sum_m \mathbb{I}\{a_m = a^*\}$ for each thought.

## Key Experimental Results

### Main Results

| Model | Dataset | Deterministic Baseline | Coverage@8 | Coverage@16 |
|-------|---------|----------------------|------------|-------------|
| Latent-SFT (1B) | GSM8K | 44.5% | 58.5% | 64.9% |
| Latent-SFT (1B) | MultiArith | 93.4% | 96.2% | 96.7% |
| RoT-4B | GSM8K | 37.5% | 39.4% | 39.7% |
| RoT-4B | MATH500 | 20.3% | 21.8% | 22.0% |

Aggregation method comparison (COCONUT, GSM-Test, N=32):

| Aggregation Strategy | GSM-Test | GSM-Hard |
|---------------------|----------|----------|
| Majority Voting | 33.6% | 6.1% |
| Best-of-N + LatentRM | **35.4%** | **7.8%** |
| Beam Search + LatentRM | ~35% | ~7% |

### Ablation Study

| Configuration | GSM-Test | GSM-Hard | Note |
|--------------|----------|----------|------|
| Full LatentRM (Best-of-8) | 35.4% | 7.8% | Full model |
| w/o contrastive (BCE) | 33.5% | 7.4% | Significant drop without contrastive loss |
| w/o stochastic rollouts | 30.7% | 6.0% | Stochastic rollout annotation is critical |
| Random scalar head | 28.9% | 5.8% | Below majority voting |

### Key Findings
- MC-Dropout achieves higher coverage in most settings, particularly on hard problems (its directional drift more readily reaches correct regions far from the deterministic solution).
- AGN is more robust in high-diversity settings, with slower coverage decay, making it well-suited for high-exploration scenarios.
- t-SNE visualization reveals: MC-Dropout produces directionally concentrated, dense expansion ("directional drift"), while AGN produces isotropic radial dispersion ("firework" pattern).
- The step-level contrastive loss of LatentRM contributes most significantly; removing it leads to a notable performance drop.
- As the number of samples increases, performance gaps between different models narrow.

## Highlights & Insights
- **Uncertainty-theoretic sampling design** is particularly elegant: decomposing the sampling problem into epistemic and aleatoric uncertainty, addressed respectively by MC-Dropout and AGN, with the two exhibiting complementary geometric exploration patterns. This analytical framework is transferable to other search problems in continuous spaces.
- **LatentRM design rationale**: using stochastic rollouts to obtain thought-level labels combined with step-level contrastive training addresses the core challenge of "scoring continuous vectors," and is generalizable to evaluation of other non-token intermediate representations.
- **Coverage vs. diversity "sweet spot" analysis** is insightful: both excessively high and excessively low diversity are detrimental, and an optimal operating point exists.

## Limitations & Future Work
- Experiments are conducted primarily on small models (GPT-2 124M, Llama-3.2-1B); latent reasoning itself still shows limited absolute performance on challenging mathematical benchmarks (AIME) and doctoral-level tasks (GPQA).
- Both MC-Dropout and AGN require hyperparameter tuning (dropout rate and noise standard deviation), though heuristic ranges are provided.
- LatentRM requires additional training, increasing deployment complexity.
- Integration of sampling and aggregation into a reinforcement learning framework for iterative refinement of latent trajectories remains unexplored.
- The latent reasoning paradigm itself is still evolving and lags behind token-based CoT on complex tasks.

## Related Work & Insights
- **vs. Self-Consistency (majority voting)**: Self-Consistency applies diverse sampling and voting in token space; this paper extends analogous ideas to continuous latent space, and achieves stronger aggregation than voting via LatentRM.
- **vs. COCONUT/CODI/CoLaR**: These are foundational latent reasoning models; this paper adds parallel TTS capability on top of them as an orthogonal enhancement.
- **vs. Stochastic Soft Thinking**: Soft Thinking operates in the token probability space (soft tokens are mixtures of token embeddings), whereas this paper operates in a purely latent vector space unconstrained by vocabulary structure.

## Rating
- Novelty: ⭐⭐⭐⭐ — Being the first to introduce parallel TTS into latent reasoning is a clear and valuable contribution, though the sampling methods themselves (dropout/noise) are not novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers multiple models, benchmarks, and sampling strategies, with extensive visualization analysis and ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ — Motivation is clear, structure is well-organized, and both theoretical derivations and experimental analyses are thorough.
- Value: ⭐⭐⭐⭐ — Fills an important gap in scaling capability for the latent reasoning paradigm with practical guidance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Efficient Test-Time Scaling via Temporal Reasoning Aggregation](efficient_test-time_scaling_via_temporal_reasoning_aggregation.md)
- [\[ICLR 2026\] Efficient Test-Time Scaling for Small Vision-Language Models](../../ICLR2026/llm_reasoning/efficient_test-time_scaling_for_small_vision-language_models.md)
- [\[ICLR 2026\] ATTS: Asynchronous Test-Time Scaling via Conformal Prediction](../../ICLR2026/llm_reasoning/atts_asynchronous_test-time_scaling_via_conformal_prediction.md)
- [\[ICLR 2026\] Plan and Budget: Effective and Efficient Test-Time Scaling on Reasoning LLMs](../../ICLR2026/llm_reasoning/plan_and_budget_effective_and_efficient_test-time_scaling_on_reasoning_large_lan.md)
- [\[ICLR 2026\] Understanding the Role of Training Data in Test-Time Scaling](../../ICLR2026/llm_reasoning/understanding_the_role_of_training_data_in_test-time_scaling.md)

</div>

<!-- RELATED:END -->

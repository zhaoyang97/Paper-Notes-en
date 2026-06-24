---
title: >-
  [Paper Note] Trajectory Balance with Asynchrony: Decoupling Exploration and Learning for Fast, Scalable LLM Post-Training
description: >-
  [NeurIPS 2025][LLM Alignment][RL post-training] This paper proposes TBA (Trajectory Balance with Asynchrony), which combines the GFlowNet Trajectory Balance (TB) objective with an asynchronous distributed RL architecture to decouple exploration and learning in LLM post-training, achieving 4–50× speedups without performance degradation across mathematical reasoning, preference fine-tuning, and automated red-teaming tasks.
tags:
  - "NeurIPS 2025"
  - "LLM Alignment"
  - "RL post-training"
  - "asynchronous RL"
  - "trajectory balance"
  - "off-policy learning"
  - "GFlowNet"
date: 2026-05-08
content_hash: 7e333847c572d77f
---

# Trajectory Balance with Asynchrony: Decoupling Exploration and Learning for Fast, Scalable LLM Post-Training

**Conference**: NeurIPS 2025
**arXiv**: [2503.18929](https://arxiv.org/abs/2503.18929)  
**Code**: [GitHub](https://github.com/bbartoldson/TBA)  
**Area**: LLM Alignment
**Keywords**: RL post-training, asynchronous RL, trajectory balance, off-policy learning, GFlowNet

## TL;DR

This paper proposes TBA (Trajectory Balance with Asynchrony), which combines the GFlowNet Trajectory Balance (TB) objective with an asynchronous distributed RL architecture to decouple exploration and learning in LLM post-training, achieving 4–50× speedups without performance degradation across mathematical reasoning, preference fine-tuning, and automated red-teaming tasks.

## Background & Motivation

RL in LLM post-training (e.g., PPO, RLOO) is essential for improving reasoning and alignment, but faces severe efficiency bottlenecks:

**On-policy dependency**: Mainstream algorithms such as PPO/RLOO require strict alternation between data generation and policy updates, causing GPU idle time during generation and generator stalls during training, resulting in extremely low resource utilization.

**Challenges of asynchronization**: Although asynchronous RL can decouple generation and training for parallelism, existing RL objectives (e.g., PPO, DPO) suffer significant performance degradation when processing off-policy data. Noukhovitch et al. found that increasing the degree of off-policyness degrades win-rate or exacerbates policy drift.

**Limited gains from scaling generation**: Hou et al. found that scaling generation from 4 to 8–16 samples in PPO yields limited benefit.

**Core insight**: The GFlowNet Trajectory Balance (TB) objective natively supports off-policy learning—training data can originate from any distribution with full support. This allows off-policy data arising from delayed parameter synchronization in asynchronous RL to be effectively utilized rather than treated as noise.

**Key Insight**: Reformulate the KL-regularized RL problem as a probabilistic inference problem, and leverage TB's off-policy property to graft it onto an asynchronous distributed architecture.

## Method

### Overall Architecture

The TBA architecture consists of two types of nodes: multiple **Searcher nodes** responsible for parallel data generation (accelerated by vLLM), and one **Trainer node** that continuously trains using the TB objective by sampling from a replay buffer. The two operate in a fully asynchronous manner, synchronizing parameters and the buffer only every $k$ steps.

### Key Designs

1. **Trajectory Balance (TB) objective**: The optimal policy for KL-regularized RL is expressed as the posterior distribution $\pi^*(y|x) \propto \pi_{\text{ref}}(y|x) \exp(\beta^{-1} r_\phi(y;x))$. The VarGrad variant of the TB loss is used:

    $$\mathcal{L}_{\text{TB}}^{\text{VarGrad}}(\mathbf{B};\theta) = \frac{1}{BK} \sum_{i,j} \left( \text{SG}[\log \hat{Z}(\mathbf{x}^{(i)})] + \log \pi_\theta(\mathbf{y}^{(i,j)} | \mathbf{x}^{(i)}) - \log \pi_{\text{ref}}(\mathbf{y}^{(i,j)} | \mathbf{x}^{(i)}) - \frac{1}{\beta} r_\phi(\mathbf{y}^{(i,j)}; \mathbf{x}^{(i)}) \right)^2$$

   where $\hat{Z}$ is a batch estimate computed from $K$ samples and $\text{SG}$ denotes stop-gradient. Its gradient is equivalent to REINFORCE with a mean baseline plus KL-regularized reward: $\nabla \mathcal{J}_{\text{TB}} = A^{(i,j)} \nabla \log \pi_\theta$, where the advantage function is $A^{(i,j)} = (r^{(i,j)} - \bar{r}^{(i)}) - \beta(\hat{\text{KL}}^{(i,j)} - \bar{\text{KL}}^{(i)})$. **Key distinction**: This equivalence breaks down on off-policy data; TB handles this correctly whereas REINFORCE does not.

2. **Asynchronous distributed architecture**: Searcher nodes maintain a stale policy copy $\pi_{\theta'}$, generate responses using vLLM, compute rewards, and store results in a local buffer $\mathcal{D}_{\text{local}}$. Every $k$ steps, all Searchers' $\mathcal{D}_{\text{local}}$ are merged into the global buffer $\mathcal{D}_{\text{global}}$, and the Searchers' policy weights are updated. The key design principle is that generation and training are fully asynchronous, with no blocking dependency between the two.

3. **Prioritized sampling strategy**: Sampling from $\mathcal{D}_{\text{global}}$ alternates between two strategies: with probability $m$, data added in the most recent synchronization step is sampled (approximating on-policy); with probability $1-m$, sampling is reward-prioritized (encouraging exploration of high-reward regions). $m$ is a critical hyperparameter—$m=1$ corresponds to purely approximate on-policy sampling, and $m=0$ to purely reward-prioritized sampling.

### Loss & Training

- The KL coefficient $\beta$ follows a linear annealing schedule: larger values promote stability, smaller values improve precision
- Reference policy resetting is supported, analogous to the approach used in Kimi K1.5
- $S > K$ samples per query are generated (e.g., $S=24, K=20$) to increase the number of unique sequences
- The TBA' variant is implemented on top of the PRIME-RL codebase for larger models (Qwen 2.5 7B)

## Key Experimental Results

### Main Results: Mathematical Reasoning (GSM8K, RhoMath-1B)

| Method | GPUs | Accuracy | Training Time | Speedup |
|---|---|---|---|---|
| VinePPO | 4×A100 | 52.8% | ~68h | 1× |
| Async DPO | 4×A100 | 53.1% | ~2h | ~34× |
| TBA | 4×A100 | **54.6%** | **~1.4h** | **~50×** |
| PPO | 4×A100 | 41.5% | ~8h | ~8.5× |

### Preference Fine-Tuning (TL;DR, Pythia-410M)

| Method | Off-policy Steps | Win Rate ↑ | Perplexity/KL ↓ |
|---|---|---|---|
| Online DPO (on-policy) | 1 | 0.85 | 1.13 |
| Async PPO | ≈15 | 0.76 | 1.10 |
| Proximal RLOO | ≈15 | 0.77 | 1.10 |
| **TBA** | **≈15** | **0.86** | **1.13** |

### Ablation Study: Effect of Off-policy Degree

| Near-on-policy sampling probability $m$ | Win Rate (PFT) |
|---|---|
| 0.4 | 0.67 |
| 0.5 | **0.82** |
| 0.6 | 0.80 |

### Key Findings

- TBA under high asynchrony (≈15 off-policy steps) outperforms the on-policy setting of Online DPO
- In red-teaming, increasing the number of Searcher nodes consistently improves attack success rate and diversity, demonstrating good scalability
- TBA' on Qwen 2.5 7B also outperforms Dr. GRPO on MATH, particularly under highly off-policy settings
- $\beta$ annealing and reference policy resetting are critical for training stability

## Highlights & Insights

- The natural alignment between TB's off-policy property and asynchronous RL is the core insight, transforming "off-policyness = drawback" into "off-policyness = feature"
- Introducing GFlowNet theory into LLM post-training represents cross-domain innovation
- The 50× speedup is practically significant—post-training that previously required days is compressed to hours
- Prioritized sampling is especially important in reward-sparse settings such as red-teaming

## Limitations & Future Work

- The TB objective operates at the trajectory level, resulting in higher gradient variance, which requires sampling more responses per query to mitigate
- The current setup uses 20 samples per query for TB computation, yielding a relatively small effective batch size
- The preference fine-tuning experiments use 32-bit precision without DeepSpeed, making conditions not fully comparable with baselines
- A large amount of initial data is required to warm up the buffer (MR: 500, PFT: 10,000 examples), leading to high cold-start cost
- Performance fluctuations toward the end of training call for better variance control or early stopping strategies

## Related Work & Insights

- Compared to the asynchronous RL work of Noukhovitch et al. (2024), TBA employs a principled off-policy objective rather than increasing importance-sampling ratios
- The objective is closely related to Kimi K1.5/K2, but TBA differs in its reference policy resetting strategy
- Prior work on GFlowNet for text generation (hu2024amortizing, lee2024learning) lays the foundation for this paper
- Takeaway: Off-policy methods may hold advantages over on-policy methods in large-scale distributed training

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Introducing the GFlowNet TB objective into asynchronous LLM post-training is highly original
- Experimental Thoroughness: ⭐⭐⭐⭐ — Three tasks across multiple model scales with detailed ablations, though some experimental conditions are not fully matched with baselines
- Writing Quality: ⭐⭐⭐⭐ — Theoretical analysis is clear, gradient derivations are complete, and architecture diagrams are intuitive
- Value: ⭐⭐⭐⭐⭐ — The 50× speedup has significant practical implications for LLM training; open-sourced code enhances reproducibility

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] GVPO: Group Variance Policy Optimization for Large Language Model Post-Training](gvpo_group_variance_policy_optimization_for_large_language_model_post-training.md)
- [\[AAAI 2026\] DeCoRL: Decoupling Reasoning Chains via Parallel Sub-Step Generation and Cascaded Reinforcement for Interpretable and Scalable RLHF](../../AAAI2026/llm_alignment/decorl_decoupling_reasoning_chains_via_parallel_sub-step_gen.md)
- [\[ICLR 2026\] Spectrum Tuning: Post-Training for Distributional Coverage and In-Context Steerability](../../ICLR2026/llm_alignment/spectrum_tuning_post-training_for_distributional_coverage_and_in-context_steerab.md)
- [\[ICML 2026\] Decoupling Reasoning and Confidence: Resurrecting Calibration in Reinforcement Learning from Verifiable Rewards](../../ICML2026/llm_alignment/decoupling_reasoning_and_confidence_resurrecting_calibration_in_reinforcement_le.md)
- [\[ICLR 2026\] Fluent Alignment with Disfluent Judges: Post-training for Lower-Resource Languages](../../ICLR2026/llm_alignment/fluent_alignment_with_disfluent_judges_post-training_for_lower-resource_language.md)

</div>

<!-- RELATED:END -->

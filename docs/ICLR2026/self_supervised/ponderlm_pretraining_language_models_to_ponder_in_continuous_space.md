---
title: >-
  [Paper Note] PonderLM: Pretraining Language Models to Ponder in Continuous Space
description: >-
  [ICLR2026][Self-Supervised Learning][pondering] This paper proposes PonderLM, which introduces a "pondering" mechanism at pretraining time — computing a weighted sum of the predicted probability distribution over token e…
tags:
  - "ICLR2026"
  - "Self-Supervised Learning"
  - "pondering"
  - "language model"
  - "continuous space"
  - "test-time compute"
  - "pretraining"
date: 2026-05-08
content_hash: 522774d8466849b3
---

# PonderLM: Pretraining Language Models to Ponder in Continuous Space

**Conference**: ICLR2026
**arXiv**: [2505.20674](https://arxiv.org/abs/2505.20674)
**Code**: To be confirmed
**Area**: Self-Supervised
**Keywords**: pondering, language model, continuous space, test-time compute, pretraining

## TL;DR
This paper proposes PonderLM, which introduces a "pondering" mechanism at pretraining time — computing a weighted sum of the predicted probability distribution over token embeddings to form a continuous pondering embedding, then performing repeated forward passes. Without labeled data or reinforcement learning, a 2.8B model trained with this approach surpasses a 6.9B baseline on 9 downstream tasks.

## Background & Motivation

**Background**: The dominant approach to improving model capability is scaling parameters and data, but this faces bottlenecks including data exhaustion, scaling saturation, and communication overhead. Inference-time scaling via Chain-of-Thought (CoT) also has limitations: it requires labeled data and reinforcement learning, and smaller models struggle to benefit.

**Limitations of Prior Work**: CoT operates in discrete token space, constrained by a fixed vocabulary, and its performance ceiling is bounded by the quality of the base pretrained model.

**Key Challenge**: More computation is needed to improve performance, yet naively increasing parameter count is prohibitively expensive.

**Goal**: Improve performance without increasing parameter count by performing multiple forward passes within a single token generation step.

**Key Insight**: Drawing an analogy to how humans repeatedly deliberate when facing complex problems, the paper enables models to "think" in continuous space.

**Core Idea**: A pondering embedding is formed by taking a weighted sum of token embeddings using the predicted probability distribution, added residually to the input for a subsequent forward pass, repeated for $s$ steps.

## Method

### Overall Architecture
A standard LM produces probability distribution $\mathbf{P}$ → a pondering embedding is computed as $\mathbf{T} = \mathbf{P}\mathbf{V}$ via weighted summation over all token embeddings → residual connection $\mathbf{E}^1 = \mathbf{E}^0 + \mathbf{T}$ → another forward pass → repeated for $s$ steps.

### Key Designs

1. **Pondering Mechanism**: $\mathbf{t} = \sum_i p_i \mathbf{e}_i$; the continuous embedding retains information from all candidate tokens, enabling end-to-end differentiable training.
2. **Efficiency Optimization**: Only the top-K ($K=100$) tokens' probabilities are used to compute the pondering embedding, reducing complexity from $\mathcal{O}(n|V|d)$ to $\mathcal{O}(nKd)$.
3. **Purely Self-Supervised**: No labeled data or reinforcement learning is required; pondering is learned through standard language modeling pretraining.

### Loss & Training
Standard next-token prediction loss is used for large-scale corpus pretraining with $s=3$ pondering steps.

## Key Experimental Results

### Main Results

| Model | Parameters | Training Data | Avg. over 9 Tasks |
|------|--------|---------|---------|
| Pythia-6.9B | 6.9B | 300B tokens | Baseline |
| **PonderPythia-2.8B** | **2.8B** | **300B tokens** | **Surpasses 6.9B** |
| TinyLlama-1.1B | 1.1B | 3T tokens | Baseline |
| **PonderPythia-1B** | **1B** | **300B tokens** | **Matches TinyLlama** |

### Key Findings
- A 2.55B model matches the validation loss of Pythia-6.9B (63% parameter reduction).
- Increasing pondering steps consistently improves performance.
- The approach is effective across three architectures: GPT-2, Pythia, and LLaMA.

## Ablation Study

| Ablation / Analysis | Finding |
|-----------|------|
| Pondering steps $s$ | $s=1\to2\to3$ yields consistent performance gains; additional steps provide stable improvements. |
| Top-K approximation | $K=100$ is sufficient; further increasing $K$ yields no significant gain while substantially reducing computational complexity. |
| Architecture generality | Effective across GPT-2, Pythia, and LLaMA architectures. |
| Scaling behavior | Across the 405M–1.4B range, pondering models consistently outperform same-parameter baselines. |
| Inference-time step adjustment | Increasing pondering steps at inference (e.g., training with $s=3$, inferring with $s=5$) provides additional gains, though further validation is needed. |
| FLOPs-controlled comparison | Under identical FLOPs, PonderPythia-70M consistently outperforms vanilla Pythia-70M. |

### Core Findings from Scaling Curves
- **Parameter efficiency**: PonderPythia with 2.55B parameters matches the validation loss of Pythia with 6.9B parameters (63% parameter reduction).
- **Data efficiency**: PonderPythia achieves equivalent performance to the Pythia baseline using 59% fewer training tokens.
- **FLOPs efficiency**: PonderPythia consistently outperforms under identical compute budgets — indicating that the overhead of additional forward passes is offset by performance gains.

### Downstream Task Breakdown

| Model | LAMBADA↑ | ARC-E↑ | WinoGrande↑ | PIQA↑ | SciQ↑ | Avg.↑ |
|------|----------|--------|-------------|-------|-------|-------|
| Pythia-1B (300B) | 48.3 | 58.6 | 52.8 | 71.3 | 91.6 | 50.4 |
| PonderPythia-410M (300B) | 48.9 | 58.7 | 54.0 | 70.5 | 91.0 | **51.4** (+3.8) |
| Pythia-6.9B (300B) | Baseline | Baseline | Baseline | Baseline | Baseline | Baseline |
| **PonderPythia-2.8B** (300B) | **Exceeds** | **Exceeds** | **Exceeds** | **Exceeds** | **Exceeds** | **Exceeds 6.9B** |

## Highlights & Insights
- **A third scaling axis**: Conventional scaling operates along two axes — parameter scaling and inference-time scaling (CoT). PonderLM opens a "pondering scaling" axis, improving performance via multiple forward passes without increasing parameters.
- **Thinking in continuous space**: CoT operates in discrete token space and is constrained by vocabulary; pondering embeddings are probability-weighted continuous vectors over all tokens, carrying higher information density.
- **Interpretability window**: The evolution of probability distributions across intermediate pondering steps provides a window into the reasoning process, revealing how the model progressively refines an initial guess toward the correct answer.
- **Purely self-supervised**: The method requires no labeled data or RL and learns effective pondering through standard next-token prediction, making it broadly applicable.
- **Orthogonal to CoT**: Pondering occurs within a single token generation step, while CoT operates at the token sequence level — the two mechanisms can be combined.

## Limitations & Future Work
- Inference overhead scales linearly with pondering steps ($s$ steps require $s+1$ full forward passes), which is unfriendly to latency-sensitive applications.
- The combination with CoT remains unexplored — whether pondering models yield additional gains after RL or CoT fine-tuning is an open question.
- The number of pondering steps $s$ is fixed at both training and inference time; adaptive step counts that dynamically adjust based on problem difficulty could be more efficient.
- Pondering increases per-step computation during training, slowing overall training speed; although FLOPs efficiency improves, wall-clock time is not discussed in detail.
- Validation is currently limited to the Pile dataset; evaluation across more diverse data distributions and modalities would be valuable.

## Related Work & Insights
- **vs. CoT / o1 / R1**: CoT generates reasoning chains in discrete space; PonderLM iteratively refines representations in continuous space. The former requires labeled data or RL; the latter is purely self-supervised.
- **vs. Universal Transformer (Dehghani et al.)**: Universal Transformers allow variable-depth computation (different numbers of layers per token); PonderLM allows multiple iterations of the same full model. The intuitions are related but the mechanisms differ.
- **vs. PonderNet (Banino et al.)**: PonderNet learns when to halt computation (adaptive halting); PonderLM uses a fixed number of steps but retains richer information through continuous embeddings.
- **Inspiration**: The pondering mechanism could be extended to multimodal settings — mixed pondering over visual and text tokens may enable implicit cross-modal reasoning.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — The continuous-space pondering mechanism is an entirely novel approach that opens a third scaling axis.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Three architectures, 9 downstream tasks, and rigorous scaling curves.
- Writing Quality: ⭐⭐⭐⭐ — Intuitive explanations and clear pseudocode.
- Value: ⭐⭐⭐⭐⭐ — Proposes a new computational scaling paradigm that is orthogonal to and composable with existing approaches.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Improving Large Vision and Language Models by Learning from a Panel of Peers](../../ICCV2025/self_supervised/improving_large_vision_and_language_models_by_learning_from_a_panel_of_peers.md)
- [\[NeurIPS 2025\] Continuous Subspace Optimization for Continual Learning (CoSO)](../../NeurIPS2025/self_supervised/continuous_subspace_optimization_for_continual_learning.md)
- [\[NeurIPS 2025\] M-GRPO: Stabilizing Self-Supervised Reinforcement Learning for Large Language Models with Momentum-Anchored Policy Optimization](../../NeurIPS2025/self_supervised/m-grpo_stabilizing_self-supervised_reinforcement_learning_for_multimodal_underst.md)
- [\[CVPR 2026\] An Optimal Transport-driven Approach for Cultivating Latent Space in Online Incremental Learning](../../CVPR2026/self_supervised/an_optimal_transport_driven_approach_for_cultivating_latent_space_in_online_incr.md)
- [\[AAAI 2026\] Towards LLM-Empowered Knowledge Tracing via LLM-Student Hierarchical Behavior Alignment in Hyperbolic Space](../../AAAI2026/self_supervised/towards_llm-empowered_knowledge_tracing_via_llm-student_hierarchical_behavior_al.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] MeCeFO: Enhancing LLM Training Robustness via Fault-Tolerant Optimization
description: >-
  [NeurIPS 2025][Optimization][fault-tolerant training] MeCeFO proposes a fault-tolerant optimization algorithm for LLM training that minimizes overhead during node failures through three techniques—skip-connections, selective activation recomputation, and low-rank gradient approximation—achieving only a 4.18% throughput drop under high-frequency failure conditions.
tags:
  - NeurIPS 2025
  - Optimization
  - fault-tolerant training
  - distributed optimization
  - LLM pre-training
  - low-rank gradient approximation
  - activation recomputation
date: 2026-05-08
content_hash: 7941bcd96092ad22
---

# MeCeFO: Enhancing LLM Training Robustness via Fault-Tolerant Optimization

**Conference**: NeurIPS 2025
**arXiv**: [2510.16415](https://arxiv.org/abs/2510.16415)
**Code**: [GitHub](https://github.com/pkumelon/MeCeFO)
**Area**: Optimization
**Keywords**: fault-tolerant training, distributed optimization, LLM pre-training, low-rank gradient approximation, activation recomputation

## TL;DR

MeCeFO proposes a fault-tolerant optimization algorithm for LLM training that minimizes overhead during node failures through three techniques—skip-connections, selective activation recomputation, and low-rank gradient approximation—achieving only a 4.18% throughput drop under high-frequency failure conditions.

## Background & Motivation

Large-scale LLM training requires tens of thousands of GPUs working in concert. At this scale, hardware failures are inevitable: Meta reported an average of one hardware failure every four hours during LLaMA 3 405B training; Alibaba reported that failure handling caused 31.19% downtime.

Existing fault-tolerance approaches suffer from fundamental efficiency problems:
- **Checkpoint methods**: Periodically save training states and resume from the latest checkpoint after a failure, but replacing devices and reloading incurs substantial time costs.
- **Scheduling methods**: Dynamically reassign tasks, but reduced device count leads to throughput degradation.
- **Redundant computation**: Replicate tasks across devices, significantly reducing GPU utilization even in failure-free runs.

The authors' core insight is that all of the above methods are **algorithm-agnostic**—they strive to guarantee exact execution of every computation step. However, the goal of training is not to reproduce a precise sequence of computations, but to obtain well-generalizing parameters. SGD/Adam are inherently robust to gradient noise, which means one can **strategically sacrifice computational precision** in exchange for efficiency.

## Method

### Overall Architecture

MeCeFO adopts a **Neighbor-Do-Both (NDB)** strategy: when a node fails, a neighboring node in the same data-parallel group takes over its computation. A naïve NDB implementation doubles both memory and computation on the neighbor node, necessitating three key techniques to mitigate this overhead.

### Key Designs

1. **Skip-Connection**: Skips the connection through the MHA (Multi-Head Attention) module during backpropagation. Empirical observation (Figure 3) shows that skipping MHA causes far less interference with training than skipping FFN. After the neighbor node skips MHA, the gradient for that layer is contributed only by unaffected data-parallel (DP) groups:
    $\overline{\mathbf{G}}_{\ell,\#} = \frac{1}{|\mathcal{N}_{\ell,\#}|} \sum_{i \in \mathcal{N}_{\ell,\#}} \mathbf{G}_{i,\ell,\#}$
   where $\mathcal{N}_{\ell,\#}$ is the set of DP groups that are neither failed nor serving as neighbors. This simultaneously eliminates activation storage and Wgrad/Dgrad computation for MHA.

2. **Selective Activation Recomputation**: Skip-connections are not applied to FFN modules (as skipping FFN introduces severe approximation error and gradient bias). Instead, only the input activation of each FFN module is retained, and intermediate activations are recomputed during backpropagation. This eliminates intermediate activation storage for FFN at the cost of an additional forward pass (approximately 1/3 of normal FFN computation).

3. **Low-Rank Gradient Approximation**: To compensate for the extra overhead introduced by recomputation, a low-rank approximation is applied to the weight gradient of linear layers $\mathbf{y} = \mathbf{W}\mathbf{x}$ in FFN. Performing SVD on $\mathbf{W}$ to obtain the top-$r$ right singular vectors $\mathbf{V}_1$ yields:
    $\mathbf{G}_W = \mathbf{G}_y \mathbf{x}^\top \approx \mathbf{G}_y (\mathbf{x}^\top \mathbf{V}_1) \mathbf{V}_1^\top$
   When $r \ll \min\{b, m, n\}$, the FLOPs for the approximate Wgrad become negligible ($(2brn + 2brm + 2rmn)$ vs. the original $2bmn$), effectively compensating for the recomputation overhead. The projection matrix $\mathbf{V}_1$ is updated every $\tau$ steps to reduce SVD cost.

### Loss & Training

**Convergence Analysis**: Under standard assumptions ($L$-smoothness, bounded stochastic gradient variance) and a gradient error assumption (Assumption 3):

**Theorem 1**: MeCeFO with momentum SGD achieves a convergence rate of $\mathcal{O}\left(\frac{1}{\sqrt{nT}}\right)$, consistent with standard distributed SGD.

The key gradient error assumption (Assumption 3) requires that the relative error between the approximate gradient and the failure-free gradient be bounded. Experiments verify that this error remains below 0.6 throughout LLaMA-1B pre-training.

## Key Experimental Results

### Main Results (Throughput Comparison)

| Model | Method | No Failure | High-Freq. Failure | Throughput Drop |
|-------|--------|-----------|-------------------|-----------------|
| LLaMA-350M | Bamboo | 438k tok/s | 407k tok/s | 7.04% |
| LLaMA-350M | Oobleck | 704k tok/s | 632k tok/s | 10.14% |
| LLaMA-350M | **MeCeFO** | **1199k tok/s** | **1186k tok/s** | **1.07%** |
| LLaMA-1B | Bamboo | 154k tok/s | 141k tok/s | 8.21% |
| LLaMA-1B | Oobleck | 291k tok/s | 251k tok/s | 13.87% |
| LLaMA-1B | **MeCeFO** | **471k tok/s** | **457k tok/s** | **2.98%** |
| LLaMA-7B | Bamboo | 12.4k tok/s | 9.8k tok/s | 20.84% |
| LLaMA-7B | Oobleck | 67.0k tok/s | 48.1k tok/s | 28.09% |
| LLaMA-7B | **MeCeFO** | **111.1k tok/s** | **106.5k tok/s** | **4.18%** |

### Ablation Study (Training Quality)

| Model | No Failure PPL | Low-Freq. PPL | Mid-Freq. PPL | High-Freq. PPL |
|-------|---------------|--------------|--------------|---------------|
| LLaMA-350M | 18.74 | 18.75 (+0.05%) | 18.88 (+0.75%) | 19.04 (+1.60%) |
| LLaMA-1B | 15.49 | 15.51 (+0.13%) | 15.61 (+0.77%) | 15.83 (+2.19%) |
| LLaMA-7B | 14.92 | 14.97 (+0.34%) | 15.04 (+0.80%) | 15.16 (+1.61%) |

| Failure Frequency | GLUE Avg. | BoolQ | PIQA | Note |
|------------------|----------|-------|------|------|
| No Failure | 80.06 | 0.579 | 0.682 | Baseline |
| Low-Freq. | 80.03 | 0.594 | 0.674 | Negligible degradation |
| Mid-Freq. | 80.13 | 0.571 | 0.678 | Minor fluctuation |
| High-Freq. | 79.99 | 0.587 | 0.684 | Still acceptable |

### Key Findings

- MeCeFO's throughput substantially outperforms Bamboo even under failure-free conditions (since Bamboo requires continuous redundant computation), achieving approximately a 2.7–9× advantage.
- Under high-frequency failures (one failure every 0.5 hours), MeCeFO's resilience is 5.0–6.7× that of Oobleck.
- Perplexity degradation is minimal: LLaMA-7B under high-frequency failures increases by only 1.61% (14.92→15.16), with negligible impact on downstream tasks.
- Under low-frequency failures, certain zero-shot tasks exhibit slight improvements (BoolQ: 0.579→0.594), potentially attributable to implicit regularization effects of approximate training.

## Highlights & Insights

- **Perspective shift**: Transitioning from "exactly reproducing computation" to "preserving model quality" represents a paradigm shift in fault-tolerant training.
- **Co-designed three-technique system**: Skip-connections reduce MHA overhead, recomputation reduces FFN memory, and low-rank approximation compensates for recomputation cost—the three techniques are tightly coupled.
- **Theory–experiment consistency**: The convergence rate matches standard SGD, and experiments confirm near-zero quality loss.
- No overhead is introduced during failure-free operation; approximate strategies are activated only upon failure—a fundamental advantage over Bamboo.

## Limitations & Future Work

- The per-iteration failure simulation setting may differ from real-world cluster failure patterns.
- Experiments are limited to a 32-GPU cluster (4 nodes); validation on ultra-large-scale clusters (100k GPUs) is lacking.
- The theory relies on Assumption 3 (bounded gradient error), which, while empirically validated, lacks a priori guarantees.
- Only single-node failures are currently handled; simultaneous multi-node failures require coordination with other system-level solutions.
- The choice to skip MHA is empirical and may need re-evaluation for different model architectures.

## Related Work & Insights

- MeCeFO shares the idea of low-rank gradient approximation with efficient training methods such as GaLore, but applies it locally and only upon failure, leaving normal training unaffected.
- There is an overlap with DropBP (stochastic skipping of backpropagation connections), but MeCeFO applies deterministic skipping restricted to neighbor nodes.
- The work emphasizes that fault-tolerant training is not merely a systems problem, but fundamentally a problem of **optimization algorithm design**.

## Rating

- Novelty: ⭐⭐⭐⭐ First optimization algorithm to apply efficient training techniques to fault tolerance; the perspective shift is valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three model scales, three failure frequencies, and comprehensive evaluation of both throughput and quality.
- Writing Quality: ⭐⭐⭐⭐ Clear structure; the motivation and synergy of the three techniques are well articulated.
- Value: ⭐⭐⭐⭐⭐ Directly practical for large-scale LLM training; a 5–7× resilience improvement is highly significant.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Scaf-GRPO: Scaffolded Group Relative Policy Optimization for Enhancing LLM Reasoning](../../ICLR2026/optimization/scaf-grpo_scaffolded_group_relative_policy_optimization_for_enhancing_llm_reason.md)
- [\[NeurIPS 2025\] Training Robust Graph Neural Networks by Modeling Noise Dependencies](training_robust_graph_neural_networks_by_modeling_noise_dependencies.md)
- [\[NeurIPS 2025\] DartQuant: Efficient Rotational Distribution Calibration for LLM Quantization](dartquant_efficient_rotational_distribution_calibration_for_llm_quantization.md)
- [\[NeurIPS 2025\] Covariances for Free: Exploiting Mean Distributions for Training-free Federated Learning](covariances_for_free_exploiting_mean_distributions_for_training-free_federated_l.md)
- [\[NeurIPS 2025\] Training-Free Bayesianization for Low-Rank Adapters of Large Language Models](training-free_bayesianization_for_low-rank_adapters_of_large_language_models.md)

</div>

<!-- RELATED:END -->

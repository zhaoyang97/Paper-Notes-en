---
title: >-
  [Paper Note] DOT-MoE: Converting Dense LLMs to MoE with Differentiable Optimal Transport
description: >-
  [ICML 2026][LLM Efficiency][MoEfication] DOT-MoE models the problem of "how to allocate neurons to experts when converting a dense FFN to an MoE" as differentiable optimal transport. By using Sinkhorn-Knopp iterations to…
tags:
  - "ICML 2026"
  - "LLM Efficiency"
  - "MoEfication"
  - "Neuron Allocation"
  - "Sinkhorn-Knopp"
  - "Straight-Through Estimator"
  - "dense-to-MoE"
date: 2026-05-08
content_hash: 52843eb4bcd382cb
---

# DOT-MoE: Converting Dense LLMs to MoE with Differentiable Optimal Transport

**Conference**: ICML 2026  
**arXiv**: [2606.01666](https://arxiv.org/abs/2606.01666)  
**Code**: Not provided  
**Area**: Model Compression / MoE / Optimal Transport  
**Keywords**: MoEfication, Neuron Allocation, Sinkhorn-Knopp, Straight-Through Estimator, dense-to-MoE

## TL;DR
DOT-MoE models the problem of "how to allocate neurons to experts when converting a dense FFN to an MoE" as differentiable optimal transport. By using Sinkhorn-Knopp iterations to solve entropic-regularized balanced transport and a Straight-Through Estimator, it enables the joint end-to-end learning of neuron-to-expert assignment and the router. It retains 90% of dense performance with 50% active parameters on LLaMA-2/3 and Qwen2.5, outperforming all baselines such as structured pruning, random allocation, and clustering.

## Background & Motivation

**Background**: Scaling LLMs delivers performance leaps but comes with enormous inference costs. Dense Transformers activate all parameters for every token, causing latency to explode. MoEs (Switch, GShard, Mixtral, Qwen3-30B-A3B) decouple "model size" from "inference cost" through sparse routing—Qwen3-30B-A3B has 30.5B total parameters but activates only 3.3B per token. However, training MoEs from scratch is data-hungry and requires complex load-balancing. MoEfication (Zhang 2022) follows a "dense-to-MoE" route to leverage existing dense checkpoints.

**Limitations of Prior Work**: Existing MoEfication methods for allocating neurons to experts rely on heuristic strategies: (1) Random (LLaMA-MoE), which requires massive continued pretraining; (2) Weight-based clustering (LTE/MoEfication) based on similarity in $W_{\text{gate}}/W_{\text{up}}$ weights; (3) Activation-based clustering (LLaMA-MoE-v2, CMoE) based on activation/gradient importance. A common limitation is that they optimize proxies for intermediate representations (input weights / activations / co-activation) rather than the actual FFN output. Given $\text{FFN}(\mathbf{x}) = \mathbf{H} \mathbf{W}_{\text{down}}$, the output depends on the interaction between the intermediate $\mathbf{H}$ and $\mathbf{W}_{\text{down}}$, which proxy methods fail to capture.

**Key Challenge**: Neuron allocation and router training must be optimized jointly (changing neuron allocation alters which tokens should route to which expert, necessitating router updates). However, discrete assignment is non-differentiable, leading existing methods to use frozen assignments while training the router—a two-stage approach that fails to optimize the overall output reconstruction.

**Goal**: To build a framework that (a) jointly optimizes neuron assignment and the router, (b) guarantees balanced expert capacity, and (c) is output-aware rather than proxy-aware.

**Key Insight**: Neuron allocation can be framed as mass transport (each neuron carries unit mass to an expert, and each expert receives $s$ units of mass), which is precisely an optimal transport (OT) problem. OT has an analytical Sinkhorn solution that is differentiable, and entropic regularization ensures a unique, closed-form solution. The Straight-Through Estimator allows backpropagation through discrete decisions.

**Core Idea**: (1) Frame neuron allocation as balanced OT: source $\mathbf{r} = \mathbf{1}_{d_{\text{ffn}}}$ (each neuron once), target $\mathbf{c} = s \cdot \mathbf{1}_E$ (each expert receives $s$), with a learnable cost matrix; (2) Use Sinkhorn-Knopp to solve entropic-regularized OT for soft assignments; (3) Use greedy rounding for hard assignments, with STE allowing gradients to pass; (4) Jointly train assignment and the router using a KL divergence loss to reconstruct dense output.

## Method

### Overall Architecture

Input: FFN of a dense pretrained LLM $\text{FFN}(\mathbf{x}) = (\sigma(\mathbf{x} \mathbf{W}_{\text{gate}}) \odot (\mathbf{x} \mathbf{W}_{\text{up}})) \mathbf{W}_{\text{down}}$ with $d_{\text{ffn}}$ intermediate neurons. Goal: Divide into $E$ experts, each containing $s = d_{\text{ffn}}/E$ neurons, with each token routed to $k < E$ experts. Three components: (1) **OT-based neuron assignment** utilizing a learnable affinity matrix $\mathbf{A}$ and Sinkhorn for soft assignment; (2) **Token routing** using a router $\mathbf{W}_r$ and top-$k$ selection; (3) **Alignment phase** using KL divergence between dense teacher and sparse student outputs, auxiliary load balancing, and router z-loss for joint training. After training, extract $\mathbf{M}$ to convert to a standard MoE architecture.

### Key Designs

1. **Neuron Assignment as Balanced Optimal Transport**:
    - **Function**: Guarantees strictly balanced expert capacity while allowing learnable affinity to determine assignment.
    - **Mechanism**: Define the transport problem as $\mathbf{M}^* = \arg\max_{\mathbf{M} \in \mathcal{U}(\mathbf{r}, \mathbf{c})} \langle \mathbf{A}, \mathbf{M} \rangle$, where $\mathbf{A} \in \mathbb{R}^{d_{\text{ffn}} \times E}$ is the learnable affinity and $\mathcal{U}$ is the transportation polytope (marginals $\mathbf{r} = \mathbf{1}_{d_{\text{ffn}}}$, $\mathbf{c} = s \cdot \mathbf{1}_E$). Analytical solutions lie at polytope vertices ({0,1}-matrices) and are non-differentiable. Adding entropic regularization $-\tau H(\mathbf{M})$ makes the solution unique and internal to the polytope: $M_{i,e}^* = u_i \cdot \exp(A_{i,e}/\tau) \cdot v_e$. The Sinkhorn-Knopp algorithm performs alternating row/column normalization to find $\mathbf{u}, \mathbf{v}$ with linear convergence. Log-domain is used for numerical stability.
    - **Design Motivation**: Previous MoEfication methods using random or clustering could not guarantee strictly balanced expert capacity. OT enforces balance through marginal constraints. Entropic regularization and Sinkhorn transform the OT solver from an intractable LP into differentiable iterations for joint training with the router.

2. **Straight-Through Estimator for Differentiability**:
    - **Function**: Uses hard assignment for the forward pass (required for deployment) and soft assignment for the backward pass (gradient propagation).
    - **Mechanism**: Sinkhorn provides soft assignment $\mathbf{M}_{\text{soft}}$. Greedy rounding converts this to hard assignment $\mathbf{M} \in \{0,1\}^{d_{\text{ffn}} \times E}$ (ranking $\mathbf{M}_{\text{soft}}$ entries to satisfy capacity). STE is applied: $\mathbf{M}_{\text{STE}} = \mathbf{M} + (\mathbf{M}_{\text{soft}} - \text{sg}(\mathbf{M}_{\text{soft}}))$. Router top-$k$ selection follows the same logic: $\mathbf{R}_{\text{STE}} = \mathbf{R} + (\mathbf{P} - \text{sg}(\mathbf{P}))$.
    - **Design Motivation**: MoE deployment requires hard expert assignment, but gradients break during training if hard assignments are used directly. STE, a classic trick from quantization networks like BinaryNet, is ported here to two-level discrete decisions (neuron-to-expert and token-to-expert) as a key enabler for joint training.

3. **Output-Aware KL Divergence Alignment**:
    - **Function**: Aligns the output of the sparse student directly with the dense teacher, rather than using intermediate proxies.
    - **Mechanism**: The forward pass simulates sparse MoE computation: $\hat{\mathbf{Y}} = (\mathbf{H} \odot (\mathbf{R} \mathbf{M}^\top)) \mathbf{W}_{\text{down}}$, where only $k \cdot s$ neurons contribute to the output. The loss is composed of KL divergence between dense $\mathbf{Y}$ and sparse $\hat{\mathbf{Y}}$, cross-entropy LM loss, router z-loss, and load balancing loss. Training updates affinity $\mathbf{A}$, router $\mathbf{W}_r$, and the entire network.
    - **Design Motivation**: Previous methods optimized proxies like input weight similarity or activation co-occurrence. This method directly targets output reconstruction MSE. Output-awareness aligns the training objective with the deployment goal.

### Multi-head Attention Extension

The same balanced transport framework can be generalized to attention heads (treating heads as neurons to be assigned to experts), as detailed in Appendix G.

## Key Experimental Results

### Perplexity & HellaSwag at 50% Parametric Budget (LLaMA-2 7B)

| Method | WikiText PPL ↓ | HellaSwag acc-n ↑ |
|--------|----------------|---------------------|
| **Structured Pruning** | | |
| LLM-Pruner | 31.05 | – |
| LLM Surgeon | 15.38 | 40.3 |
| ShortGPT | 268.11 | 43.7 |
| SliceGPT | 24.82 | 33.0 |
| ModeGPT | 11.88 | – |
| DISP-LLM | 9.84 | 46.3 |
| **Semi-Structured Pruning (2:4)** | | |
| SparseGPT | 10.17 | 43.3 |
| Wanda | 11.02 | 40.9 |
| Pruner-Zero | 10.52 | 54.7 |
| **DOT-MoE** | **7.99** | 53.9 |

DOT-MoE achieves the lowest WikiText PPL of 7.99, 1.85 lower than the strongest structured pruning baseline (DISP-LLM 9.84). Its HellaSwag score of 53.9 is comparable to the strongest Pruner-Zero (54.7).

### Common-Sense Reasoning (Multiple benchmarks)

| Method | Active Params | FT Tokens | BoolQ | SciQ | PIQA | WinoG. | ARC-C | HellaS. | Avg. |
|---|---|---|---|---|---|---|---|---|---|
| **LLaMA-2 7B** | | | | | | | | | |
| Dense | 6.74B | 2T* | 82.0 | 94.0 | 78.1 | 74.3 | 52.5 | 78.9 | 76.6 |
| LLaMA-MoE (Random) | 3.49B | 1.2B | 37.8 | 20.0 | 49.7 | 50.1 | 25.8 | 26.2 | 34.9 |
| LLaMA-MoE-v2 | 3.49B | 1.2B | 51.3 | 67.0 | 56.6 | 52.9 | 25.7 | 35.1 | 48.1 |
| CMoE | 3.49B | 1.2B | 55.0 | 77.5 | 57.1 | 54.1 | 27.6 | 38.8 | 51.7 |
| **DOT-MoE** | 3.49B | 1.2B | **72.5** | **94.3** | **69.3** | **62.5** | **40.9** | **60.2** | **66.6** |
| **LLaMA-3 8B** | | | | | | | | | |
| Dense | 8.03B | 15T* | 83.2 | 96.2 | 79.6 | 77.3 | 58.3 | 82.1 | 79.4 |
| CMoE | 3.80B | 1.2B | 71.1 | 94.4 | 69.5 | 59.5 | 38.2 | 55.3 | 64.7 |
| **DOT-MoE** | 3.80B | 1.2B | **75.0** | 94.2 | **70.2** | **63.8** | **42.4** | **61.1** | **67.8** |
| LLaMA-MoE-v2 (7B FT) | 3.80B | 7B | 74.6 | 94.5 | 69.3 | 60.5 | 42.8 | 59.0 | 66.8 |
| **DOT-MoE (7B FT)** | 3.80B | 7B | 75.4 | **96.2** | **73.3** | **66.1** | **49.1** | **66.0** | **71.0** |

At 50% active parameters: LLaMA-2 7B Dense drops from 76.6 to 66.6 (87% retention), outperforming the next best CMoE (51.7) by +14.9 points. LLaMA-3 8B also shows a +3.1 improvement over CMoE. Similar trends are observed for Qwen2.5 7B.

### Key Findings

- **Retention of 90% dense performance at 50% active params**: DOT-MoE pushes the quality-efficiency tradeoff of dense-to-MoE conversion closer to the Pareto optimal; structured pruning usually drops to 40-60% at this budget.
- **OT-based >> heuristic clustering**: Compared to LLaMA-MoE (random), LLaMA-MoE-v2 (activation), and CMoE (balanced k-means activation), DOT-MoE improves by an average of 14.9-30.7 points, proving joint optimization is significantly better than frozen heuristics.
- **Effective with small-data FT**: Using only 1.2B FT tokens (0.06% of the 2T dense pretrain) recovers performance to 87%, suggesting OT-based assignment provides an excellent initialization.
- **More FT tokens narrow the gap further**: LLaMA-3 8B with 7B FT tokens scores 71.0, only 8.4 points behind the 79.4 dense baseline.
- **MSE experiments validate output-awareness**: Appendix results show proxy-based methods have MSEs $2\times$ to $41\times$ higher than DOT-MoE, confirming the output reconstruction objective as more accurate.
- **Robust across architectures**: The method works consistently across LLaMA-2, LLaMA-3, and Qwen2.5 families.
- **Superior to structured pruning**: WikiText PPL 7.99 vs. 9.84 (DISP-LLM) demonstrates the MoEfication route is superior to pruning at a 50% budget.

## Highlights & Insights

- **OT framing is an elegant methodological innovation**: Viewing neuron allocation as mass transport is intuitive yet mathematically rigorous, naturally accommodating balanced capacity and learnable affinity.
- **Sinkhorn + STE synergy**: Sinkhorn makes discrete OT differentiable, while STE makes discrete deployment backpropagatable; together, they make end-to-end joint training feasible.
- **Output-aware KL > proxy alignment**: The paper identifies "optimizing proxies" as a fundamental limitation of existing methods and quantifies this with single-layer reconstruction experiments ($2\times$-$41\times$ MSE gap).
- **Joint vs. Sequential Training Paradigm**: Shifting from "freeze assignment then train router" to co-adaptation is a fundamental shift that yields order-of-magnitude improvements.
- **Dynamic structural pruning perspective**: Framing MoEfication as dynamic pruning (retaining all parameters but activating conditionally) provides capacity advantages over static pruning, which permanently removes parameters.
- **Extensibility to attention heads**: Using the same balanced transport framework for heads provides an OT tool for compressing attention.

## Limitations & Future Work

- **Computational cost of OT + Sinkhorn**: Each FFN layer requires Sinkhorn iterations and STE backpropagation, making it several times more expensive than simple clustering, especially for 70B+ models.
- **Greedy rounding mismatch**: The soft-to-hard transition via greedy rounding might deviate from the OT-optimal vertex, creating a theoretical gap that is not yet quantified.
- **Dependency on FT data**: DOT-MoE still requires 1.2B-7B FT tokens; performance under zero FT was not specifically ablated.
- **Selection of $k$ and $E$**: Experiments used $E=8, k=1\text{-}2$; the Pareto front for different $(E, k)$ was not systematically explored.
- **Comparison with from-scratch MoE**: A direct comparison with Mixtral or Qwen3-MoE is missing (though from-scratch training is vastly more expensive).
- **70B+ scaling not verified**: The largest model tested was 8B. For 70B+ models, $d_{\text{ffn}} \approx 28K$, so Sinkhorn convergence speed at larger $E$ requires further measurement.
- **Missing Inference Latency**: Theoretical FLOPs are halved, but actual GPU latency depends on MoE kernel implementations; wall-clock speedup figures are not provided.

## Related Work & Insights

- **vs. MoEfication / LTE (Zhang 2022)**: They use weight-based clustering; this paper proves weight similarity $\neq$ output contribution. The OT framework is significantly better.
- **vs. LLaMA-MoE (Random)**: Random allocation relies on 1.2B continued pretraining; DOT-MoE provides a better initialization and joint training, leading to a 31.7 point gap (66.6 vs. 34.9).
- **vs. LLaMA-MoE-v2 / CMoE (Activation Clustering)**: Because activation is a proxy whereas DOT-MoE is output-aware, DOT-MoE outperforms them by 12-15 points.
- **vs. Structured Pruning (DISP-LLM / SliceGPT / ShortGPT)**: Pruning loses long-tail knowledge; DOT-MoE's capacity advantage allows for much lower PPL (8.0 vs 9.8).
- **vs. Semi-Structured (SparseGPT / Wanda / Pruner-Zero)**: 2:4 sparsity is hardware-friendly but quality-limited. DOT-MoE's PPL (7.99) is superior to SparseGPT (10.17).
- **Insights**: (1) The OT + Sinkhorn + STE framework is applicable to any balanced discrete assignment problem; (2) The output-aware vs. proxy-aware distinction should be applied to other compression methods; (3) Joint training is superior to sequential training for discrete-continuous hybrid problems.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Combining OT framing, Sinkhorn, STE, and output-aware objectives is original and moves MoEfication from heuristics to principled design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Tested on 3 model families, 6 benchmarks, outperforms structured pruning, and includes detailed ablations; only lacks 70B scale and wall-clock latency.
- **Writing Quality**: ⭐⭐⭐⭐ The "proxy vs. output" insight is clearly articulated; OT formulas are rigorous yet readable; some alignment phase details are compressed to the appendix.
- **Value**: ⭐⭐⭐⭐⭐ Directly addresses LLM deployment pain points (90% performance at 50% active params) with an actionable, architecture-robust recipe for industry.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Variational Routing: A Scalable Bayesian Framework for Calibrated MoE Transformers](variational_routing_a_scalable_bayesian_framework_for_calibrated_mixture-of-expe.md)
- [\[ICML 2026\] ProbMoE: Differentiable Probabilistic Routing for Mixture-of-Experts](probmoe_differentiable_probabilistic_routing_for_mixture-of-experts.md)
- [\[ICML 2026\] ReMoE: Boosting Expert Reuse through Router Fine-Tuning in Memory-Constrained MoE LLM Inference](remoe_boosting_expert_reuse_through_router_fine-tuning_in_memory-constrained_moe.md)
- [\[ICLR 2026\] Expert Divergence Learning for MoE-based Language Models](../../ICLR2026/llm_efficiency/expert_divergence_learning_for_moe-based_language_models.md)
- [\[NeurIPS 2025\] Advancing Expert Specialization for Better MoE](../../NeurIPS2025/llm_efficiency/advancing_expert_specialization_for_better_moe.md)

</div>

<!-- RELATED:END -->

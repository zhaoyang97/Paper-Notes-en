---
title: >-
  [Paper Note] S'MoRE: Structural Mixture of Residual Experts for Parameter-Efficient LLM Fine-tuning
description: >-
  [NeurIPS 2025][Graph Learning][Parameter-Efficient Fine-Tuning] This paper proposes S'MoRE, a framework that organizes low-rank residual experts into a multi-layer tree structure and constructs token-specific "residual trees" via hierarchical routing, achieving exponentially growing structural flexibility with parameter counts comparable to LoRA, thereby substantially improving LLM fine-tuning performance.
tags:
  - "NeurIPS 2025"
  - "Graph Learning"
  - "Parameter-Efficient Fine-Tuning"
  - "Mixture of Experts"
  - "LoRA"
  - "Graph Neural Networks"
  - "Structural Flexibility"
date: 2026-05-08
content_hash: 8cef4fb1c7963e95
---

# S'MoRE: Structural Mixture of Residual Experts for Parameter-Efficient LLM Fine-tuning

**Conference**: NeurIPS 2025
**arXiv**: [2504.06426](https://arxiv.org/abs/2504.06426)  
**Code**: [GitHub](https://github.com/ZimpleX/SMoRE-LLM)  
**Area**: Graph Learning
**Keywords**: Parameter-Efficient Fine-Tuning, Mixture of Experts, LoRA, Graph Neural Networks, Structural Flexibility

## TL;DR

This paper proposes S'MoRE, a framework that organizes low-rank residual experts into a multi-layer tree structure and constructs token-specific "residual trees" via hierarchical routing, achieving exponentially growing structural flexibility with parameter counts comparable to LoRA, thereby substantially improving LLM fine-tuning performance.

## Background & Motivation

**Background**: Fine-tuning large language models faces a dual challenge of parameter efficiency and model capacity. LoRA achieves parameter efficiency through low-rank decomposition but suffers from limited capacity; MoE improves capacity through conditional computation but at the cost of low parameter utilization.

**Limitations of Prior Work**:
   - LoRA's capacity is constrained by its fixed low-rank structure, making it difficult to handle complex tasks.
   - Independent parameter learning across experts in conventional MoE leads to low parameter efficiency.
   - Increasing the number of experts incurs greater routing overhead and uneven expert utilization.

**Key Challenge**: Works such as DeepSeek-MoE demonstrate that fine-grained experts yield richer routing choices (from $\binom{16}{2}=120$ to $\binom{64}{8}=4.4B$), yet in the PEFT setting the experts themselves are already low-rank, making further decomposition suboptimal. The key question is: how can higher flexibility be attained without increasing the number of experts?

**Goal**: To achieve exponentially greater model flexibility by exploiting structural relationships among experts—rather than simply increasing their number—while preserving LoRA-level parameter efficiency.

**Key Insight**: The key insight is that the same set of experts can form exponentially many non-isomorphic tree structures through different connectivity patterns, each yielding distinct outputs. This shifts the design space from "which experts to select" to "how the selected experts are connected."

**Core Idea**: Replace the flat aggregation of single-layer MoE with tree-structured combinations of multi-layer residual experts, obtaining exponential routing flexibility through structural diversity.

## Method

### Overall Architecture

S'MoRE organizes residual experts into $L$ layers. For each input token $\bm{x}$:
1. **Routing phase** (top-down): Starting from the highest layer, experts are selected hierarchically to construct a "residual tree" of depth $L$.
2. **Aggregation phase** (bottom-up): The token propagates upward along the activated residual tree, aggregating child-node information layer by layer.

### Key Designs

#### Hierarchical Residual Decomposition

- **Function**: Decomposes expert weights into multi-order residual terms $\bm{W}^i \approx \sum_{\ell=0}^{L-1} \bm{B}_\ell^i \cdot \bm{A}_\ell^i$.
- **Core formula** (layer-wise propagation):
$$\bm{x}_{\ell+1}^i = \sum_{n \in \mathcal{N}_\ell^i} \alpha_\ell^{i,n} \cdot \sigma\left(\bm{B}_\ell^n \cdot \bm{A}_\ell^n \cdot \bm{x} + \bm{W}_\ell \cdot \bm{x}_\ell^n\right)$$
- **Design Motivation**: Two input streams are incorporated—the original token $\bm{x}$ (skip-connected to each residual order) and the previous-layer output $\bm{x}_\ell^n$ (for deep interaction). The nonlinearity $\sigma$ is critical for distinguishing non-isomorphic structures.

#### Hierarchical Routing

- **Function**: Recursively constructs each token's expert tree in a top-down manner.
- **Conditional probability**: $p(i_{\ell-1} | i_{L-1}, \ldots, i_\ell, \bm{x})$, selecting child nodes conditioned on the already-activated ancestor experts.
- **Routing implementation**: Each expert is assigned a key vector $\bm{k}_\ell^i$; a lightweight MLP generates query vectors:
$$\bm{q} = \text{MLP}_\ell(\text{concat}(\bm{x}_{\text{down}}, \bm{k}_{\ell+1}^{i'}, \ldots))$$
$$\alpha_\ell^i = \text{softmax}(\langle \bm{k}_\ell^i, \bm{q} \rangle)$$

#### Dimension Design

$$d_{\ell+1} = d_\ell + s_\ell \cdot r_\ell, \quad d_0 = 0$$

This yields the minimal intermediate dimension that guarantees no information loss, ensuring $d_L \ll d$ (e.g., $d_L=64$, $d=4096$) to control computational overhead.

### Theoretical Guarantees

**Theorem 3.4** (Structural Flexibility): The structural flexibility of S'MoRE is
$$\Gamma_{\text{S'MoRE}} = \prod_{\ell=0}^{L-1} \binom{s_\ell}{f_\ell}^{F_{\ell+1}}$$
where $F_\ell = \prod_{i=\ell}^{L-1} f_i$. Compared to the upper bound $\Gamma_{\text{MoMOR}}$, S'MoRE's flexibility grows exponentially.

Key proof sketch: Eq. 3 is interpreted as a variant of Graph Isomorphism Networks (GIN), where the $L$-layer propagation of S'MoRE simulates $L$ steps of WL testing. The nonlinearity $\sigma$ ensures the injectivity of the color refinement process, enabling the model to distinguish all non-isomorphic trees.

### Parameter Efficiency

Total parameter count $\approx 2 \cdot d \cdot d_L$, identical to LoRA with equivalent rank $d_L$. The additional overhead $\Delta = \sum_{\ell=1}^L d_\ell^2$ amounts to only 1% when $r_\ell=8$ and 2% when $r_\ell=16$.

## Key Experimental Results

### Main Results (LLaMA 3.2 1B, average accuracy across 5 NLU tasks)

| Method | Dense Gate | Noisy Top-k | Switch Gate | Params |
|--------|-----------|-------------|-------------|--------|
| LoRA | 59.15 | - | - | 0.022B |
| HydraLoRA(4) | 59.63 | - | - | 0.013B |
| MixLoRA(4) | 59.93 | 59.49 | 60.40 | ~0.086B |
| MixLoRA(8) | 60.25 | 59.62 | 60.40 | ~0.106B |
| **S'MoRE(2-2)** | **61.30** | 59.87 | **60.97** | 0.048B |
| **S'MoRE(4-4)** | **61.24** | **60.89** | **61.35** | ~0.099B |

### Results on LLaMA 3 8B

S'MoRE consistently outperforms LoRA and MixLoRA baselines on larger models as well, with particularly pronounced advantages on complex tasks such as ARC-c and CSQA.

### Ablation Study

| Component | Effect |
|-----------|--------|
| Nonlinear activation $\sigma$ | Removal degrades the model to MoMOR, which cannot distinguish non-isomorphic trees |
| Cross-layer parameter sharing (S'MoRE#) | Slight performance decrease with fewer parameters; flexibility still grows exponentially |
| Number of layers $L$ | $L=2$ already yields significant gains; diminishing returns at $L=3,4$ |
| Routing direction | Top-down outperforms bottom-up |

### Key Findings

1. S'MoRE outperforms baselines across all three gating types, demonstrating robustness to routing strategy.
2. Two-layer S'MoRE(4-4) with Switch gating achieves the highest accuracy of 61.35% on LLaMA 3.2 1B.
3. The parameter overhead $\Delta$ is negligible: only 1.0% for two layers with $r_\ell=8$.
4. Theoretical and empirical results on structural flexibility are consistent: with $s_\ell=4, f_\ell=2, L=2$, $\Gamma_{\text{S'MoRE}}=1296$ far exceeds $\Gamma_{\text{MoMOR}}=225$.

## Highlights & Insights

1. **From "which experts" to "how to connect"**: Opens a new dimension in MoE design, transcending the paradigm of simple fine-grained decomposition.
2. **Theoretical analysis from a GNN perspective**: Cleverly applies WL testing and graph isomorphism theory to analyze the expressive power of MoE architectures.
3. **Precise control of parameter efficiency**: Achieves a lossless minimal design through the dimensional recurrence relation (Eq. 4).
4. **Practicality**: The framework is parameter-equivalent to LoRA and can serve as a drop-in replacement for existing MoE-PEFT methods.

## Limitations & Future Work

1. The routing MLP introduces a small amount of additional computation (though claimed to be negligible by the authors, this may become significant as the number of layers increases).
2. Inference latency analysis is absent—multi-layer propagation may increase actual inference time.
3. Validation is limited to NLU tasks; performance on generative tasks (e.g., instruction tuning, code generation) remains unknown.
4. Adaptive selection of per-layer expert count $s_\ell$ and fan-out $f_\ell$ has not been explored.
5. Diminishing returns at $L \geq 3$ warrant further investigation into how this bottleneck can be overcome.

## Related Work & Insights

- **DeepSeek-MoE** [Dai et al., 2024]: Empirical findings that fine-grained experts improve routing flexibility; this paper provides a structurally superior alternative.
- **GIN** [Xu et al., 2019]: Expressive power analysis of graph isomorphism networks via WL testing, innovatively applied here to prove structural flexibility in MoE.
- **Mowst** [Zeng et al., 2024]: Mixed strong-and-weak experts; the multi-order residuals in this paper can be viewed as a vertically heterogeneous design.
- **HydraLoRA** [Tian et al., 2024]: Multi-head LoRA design; S'MoRE can be regarded as its multi-layer generalization.

## Rating

⭐⭐⭐⭐⭐

The method is elegantly designed, the theoretical analysis is rigorous (the GNN + WL testing perspective is particularly compelling), and the experiments are comprehensive. The concept of "structural flexibility" opens a new dimension in MoE design, with precise control over parameter efficiency. The only shortcomings are the absence of generative task experiments and inference latency analysis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MoEMeta: Mixture-of-Experts Meta Learning for Few-Shot Relational Learning](moemeta_mixture-of-experts_meta_learning_for_few-shot_relational_learning.md)
- [\[AAAI 2026\] Magnitude-Modulated Equivariant Adapter for Parameter-Efficient Fine-Tuning of Equivariant Graph Neural Networks](../../AAAI2026/graph_learning/magnitude-modulated_equivariant_adapter_for_parameter-efficient_fine-tuning_of_e.md)
- [\[NeurIPS 2025\] Moscat: Mixture of Scope Experts at Test for Generalizing Deeper GNNs](mixture_of_scope_experts_at_test_generalizing_deeper_graph_neural_networks_with_.md)
- [\[NeurIPS 2025\] The Underappreciated Power of Vision Models for Graph Structural Understanding](the_underappreciated_power_of_vision_models_for_graph_structural_understanding.md)
- [\[NeurIPS 2025\] DuetGraph: Coarse-to-Fine Knowledge Graph Reasoning with Dual-Pathway Global-Local Fusion](duetgraph_coarse-to-fine_knowledge_graph_reasoning_with_dual-pathway_global-loca.md)

</div>

<!-- RELATED:END -->

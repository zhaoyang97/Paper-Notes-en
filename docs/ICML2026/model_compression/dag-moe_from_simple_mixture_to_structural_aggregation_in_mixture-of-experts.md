---
title: >-
  [Paper Note] DAG-MoE: From Simple Mixture to Structural Aggregation in Mixture-of-Experts
description: >-
  [ICML 2026][Model Compression][Mixture-of-Experts] The standard "weighted sum" of $top-K$ expert outputs in MoE is replaced with structural aggregation via a dynamically learned Directed Acyclic Graph (DAG)…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Mixture-of-Experts"
  - "Structural Aggregation"
  - "DAG"
  - "Multi-step Reasoning"
  - "Sparse Routing"
date: 2026-05-08
content_hash: fccd7533ea1a8a66
---

# DAG-MoE: From Simple Mixture to Structural Aggregation in Mixture-of-Experts

**Conference**: ICML 2026  
**arXiv**: [2606.01062](https://arxiv.org/abs/2606.01062)  
**Code**: https://github.com/JiaruiFeng/JiaruiFeng/DAG-MoE  
**Area**: Model Compression / MoE Architecture  
**Keywords**: Mixture-of-Experts, Structural Aggregation, DAG, Multi-step Reasoning, Sparse Routing  

## TL;DR
The standard "weighted sum" of $top-K$ expert outputs in MoE is replaced with structural aggregation via a dynamically learned Directed Acyclic Graph (DAG), significantly enhancing MoE expressiveness and downstream reasoning performance with almost no additional routing or parameter overhead.

## Background & Motivation

**Background**: Modern LLMs commonly use MoE to decouple parameter count from computation—a router selects $top-K$ FFN experts for each token, and the output is $y=\sum_{i=1}^{N} g_i(x) E_i(x)$. Existing scaling axes focus on either improving routing accuracy (Expert-Choice, RNN router, load-balance loss refinements) or refining expert granularity (fine-grained, where larger $G=d_f/d_r$ increases the combinatorial space).

**Limitations of Prior Work**: While fine-grained approaches explode the number of combinations $\binom{N}{K}$ (top-2/8=28 vs top-4/16=1820), doubling $N$ simultaneously doubles routing parameters and load-balancing complexity. Consequently, SOTA systems avoid extreme fine-granularity. Furthermore, as routers and experts have been repeatedly optimized, the marginal gains from further tuning these components are diminishing.

**Key Challenge**: The standard aggregation form $\sum g_i E_i$ is **permutation invariant**—once the $top-K$ set is determined, the output is uniquely defined by the "multiset" of these experts. Experts lack order, interaction, and the possibility of multi-step composition within a single layer. In other words, the third core component of MoE—**aggregation**—has been largely ignored, locking the upper bound of expressiveness within the weighted sum function family.

**Goal**: (i) Propose an aggregation form stronger than weighted sum without increasing routing complexity; (ii) provide rigorous comparisons of expressiveness; (iii) design a lightweight, end-to-end learnable module to implement such aggregation.

**Key Insight**: Represent the $K$ selected experts as nodes on a DAG—where each node occupies a **different structural role**, and expert outputs are aggregated layer-by-layer along DAG edges. Thus, even with identical expert sets and router scores, changing the DAG yields entirely different outputs. For a fixed $K$, the number of possible DAGs grows exponentially with depth, providing a completely new scaling axis.

**Core Idea**: Replace the permutation-invariant weighted sum in the MoE layer with **structural aggregation over a per-token dynamically learned DAG**, thereby amplifying the combinatorial space without modifying the router or experts.

## Method

### Overall Architecture
DAG-MoE splits the standard MoE block into two stages: (1) The original sparse router selects $top-K$ experts and produces $K$ initial node representations as usual; (2) a new **DAG learning module** takes over aggregation, iterating $L$ times. In each iteration, it simultaneously learns "edges between nodes at the current depth" and "updates representations based on these edges." Finally, all nodes at the $L$-th layer are summed to serve as the token's output for that layer. This architecture modifies neither the router nor the expert FFNs, making it naturally compatible with existing training stacks.

### Key Designs

1.  **General Formalization of DAG-style Aggregation**:
    - **Function**: Organize the "$top-K$ list $\bm{k}$" into a DAG $G=(\mathcal{V},\mathcal{A})$ with depth $L$ and $n(l)$ nodes per layer. The incoming edge set $A_i^l$ for node $(l,i)$ specifies which preceding nodes it takes values from, with a single root node $(L,1)$ providing the output.
    - **Mechanism**: Initial layer $x_i^0 = g_{\bm{k}[i]}(x) E_{\bm{k}[i]}(x)$; intermediate layers $x_i^l = \mathrm{AGG}(\{x_j^k \mid (k,j)\in A_i^l\})$; output $y=\mathrm{AGG}(\{x_j^k \mid (k,j)\in A_1^L\})$. By using an injective $\mathrm{AGG}$ (theoretically implemented via MLP+sum/min/max), the authors prove Prop 3.1 (any DAG can be injectively encoded) $\to$ Theorem 3.2 (DAG-MoE is strictly stronger than standard MoE) $\to$ Theorem 3.3 (a single DAG-MoE layer + one multi-head attention layer can simulate a full dynamic programming process within $O(K\log n)$ input length, which standard MoE cannot achieve because it only performs one-step aggregation).
    - **Design Motivation**: Thoroughly explain "why structural aggregation is inherently stronger than weighted sum" using tools from GNN/D-VAE—permutation invariance is a ceiling on expressiveness, while DAGs provide order and multi-step composition, theoretically justifying the expansion of this space.

2.  **Lightweight DAG Learning Module (Core Implementation)**:
    - **Function**: Automatically learn the structure and perform aggregation per token without knowing the ground-truth DAG.
    - **Mechanism**: Reduce the search space dimensionality—fix $n(l)=K$ and only allow $(l,i)$ to take edges from the previous layer $l-1$, with earlier information carried via residuals. Each iteration first normalizes and reduces dimensions: $x_{i,\mathrm{input}}^l=\mathrm{LN}(x_i^{l-1})$, $x_{i,\mathrm{down}}^l=W_{\mathrm{down}}^l x_{i,\mathrm{input}}^l$. For each pair $(i,j)$, candidate edge features are formed: $x^l_{(i,j)}=\mathrm{Concat}(x_{i,\mathrm{down}}^l, x_{j,\mathrm{down}}^l)$. A soft gate $e^l_{(i,j)} = \sigma(W_{\mathrm{edge}}^l x^l_{(i,j)})$ is learned to control edge activation, and node information is computed as $\hat{x}^l_{(i,j)} = e^l_{(i,j)} \odot W_{\mathrm{node}}^l x^l_{(i,j)}$. Finally, $x_i^l = W_{\mathrm{up}}^l\sum_j \hat{x}_{(i,j)}^l + x_i^{l-1}$, where $W_{\mathrm{up}}$ is zero-initialized to stabilize early training. Output $y=\sum_{i=1}^K x_i^L$.
    - **Design Motivation**: (i) Learn the adjacency matrix as a sigmoid soft gate to avoid discrete structure search; (ii) learn the structure in a low-dimensional space $d_g \ll d$ and project back to keep extra parameters comparable to a single shared expert; (iii) use residuals and $1/K$ normalization to address magnitude shift and gradient instability caused by multi-node summation.

3.  **Token Residual Injection for Initial Nodes**:
    - **Function**: Ensure the original token representation $x$ remains accessible during aggregation, preventing total reliance on expert outputs.
    - **Mechanism**: $x_i^0 = g_{\bm{k}[i]}(x) E_{\bm{k}[i]}(x) + \tfrac{1}{K} x$, where $1/K$ ensures that after $\sum_i x_i^L$, the total residual contribution remains 1, matching the magnitude of the residual stream outside the transformer block.
    - **Design Motivation**: Ablations show that without this residual or $1/K$ scaling, training easily diverges or fails to converge—the authors describe it as "critical for training stability."

### Loss & Training
The model adopts the token-choice router and load-balance loss from Switch Transformer, overlaid with router Z-loss to suppress logit drift. The base architecture is adapted from Llama3.1-8B (retaining tokenizer/attention/FFN shapes), and the training objective is standard causal LM.

## Key Experimental Results

### Main Results
Pre-training on 12B tokens of the Pile compared three model scales (DAG-MoE-s/-m/-l), with a baseline augmented by a shared expert for strict parameter alignment. Large-scale training on 40B tokens compared DAG-MoE-l ($d_g=256$, $L=2$, 699M parameters) vs. MoE-l (shared expert $d_r=512$, also 699M):

| Dataset | Metric | MoE-l | DAG-MoE-l | Gain |
|---------|--------|-------|-----------|------|
| Pile (in-domain) | PPL ↓ | 10.51 | 10.27 | -0.24 |
| Wikipedia (OOD) | PPL ↓ | 21.08 | 20.54 | -0.54 |
| FineWeb-Edu (OOD) | PPL ↓ | 25.38 | 24.69 | -0.69 |
| C4 (OOD) | PPL ↓ | 35.21 | 34.21 | -1.00 |

The gap on OOD data is significantly larger than in-domain, consistent with Theorem 3.2's claim that expressiveness advantages are more pronounced out-of-distribution.

### Ablation Study

| Configuration | Extra Params | ΔPPL ↑ / Eval Loss ↓ | Description |
|---------------|--------------|----------------------|-------------|
| Standard MoE | 0 | 0.000 / 2.7168 | Baseline |
| + shared expert | 393K | 0.433 | Equal params, pure expert addition |
| Chain-of-Experts (CoE) | 393K | 0.480 | Equal params, iterative router |
| **DAG-MoE-s ($L=2$)** | 393K | **0.587** | Superior structural aggregation |
| MLP mixing $d_g=64$ | 98K | -0.0838 (Regress) | Structureless MLP mixing is worse |
| Downstream Finetuning (DAG-MoE-l vs MoE-l) | — | 26.13 vs 24.06 (avg 7 task) | GPQA +6.06, Lambada +3.46, PIQA +3.15 |

### Key Findings
- **Structure itself is key**, not just extra parameters: CoE with equal parameters only achieved 0.480, and structureless MLP performed worse than the baseline $\to$ this indicates the "order and iterative composition" provided by the DAG is the true effective inductive bias.
- **Iteration count $L$ is more cost-effective than dimension $d_g$**: Both $L=0\to1$ and $L=1\to2$ drop PPL by ~0.5, while $L=2\to3$ shows marginal returns; $d_g=64, L=2$ outperforms $d_g=128, L=1$ with fewer parameters.
- **Low throughput cost**: $L=1$ adds only 1.51% wall-clock overhead, and $L=2$ adds only 4.49%, with FLOPs remaining almost identical.
- **Downstream gains concentrated in multi-step reasoning tasks**: Significant improvements in GPQA, Lambada, PIQA, and BBH, while pattern-matching tasks like HellaSwag/MMLU remain almost unchanged—confirming the qualitative assertion that "structural aggregation primarily aids compositional reasoning."

## Highlights & Insights
- This work is the first to propose the "aggregation operator" of MoE as an independent design axis, linking it to GNN expressiveness (D-VAE/GIN framework). This bridge contributes three progressive theoretical results: Prop 3.1, Thm 3.2, and Thm 3.3.
- Thm 3.3, which states "a single-layer DAG-MoE + one-layer attention can simulate DP," is the paper's boldest claim. However, the authors temper this by framing it as an "existence/capacity result," explicitly stating they do not claim the learned DAG actually corresponds to any specific DP program—an admirable balance of theory as motivation and experiment as evidence.
- The soft gate $e^l_{(i,j)}$ is equivalent to learning the entire adjacency matrix as a sigmoid mask, similar to continuous relaxation in NAS/DARTS. However, by performing this only on small $K \times K$ graphs, it avoids typical NAS search costs—a strategy of "soft search in a minimal feasible structural space" that could be transferred to prompt routing or adapter selection.
- The "OOD gap > in-domain gap" phenomenon is relatively rare in MoE literature but explainable via expressiveness theory: OOD tokens are more likely to fall into expert combinations unseen during training, where the diversity advantage of structural aggregation is amplified.

## Limitations & Future Work
- Current DAG classes are artificially restricted (each layer has $K$ nodes, edges only between adjacent depths). Prop 3.1 and Thm 3.3 are slightly compromised by this, with only Thm 3.2 fully translating—a gap the authors acknowledge.
- The problem of "finding the optimal DAG" and "how the module stably learns it" is largely unaddressed; currently, it relies entirely on sigmoid soft gates and gradients, leaving the distance to a discrete global optimum unknown.
- The largest experiments are 699M parameters / 40B tokens, which is orders of magnitude smaller than SOTA MoE LLMs (billions of parameters / trillions of tokens). Scaling behavior is unverified; specifically, whether the 4.49% time overhead at $L=2$ will be amplified by sequential processing at larger scales or if `torch.compile` can truly mitigate it remains to be seen.
- Choices for the AGG implementation were not fully ablated—while the theory assumes injective MLP+sum, the engineering implementation simplified this to sigmoid gating + sum, and the gap between these remains unquantified.

## Related Work & Insights
- **vs. Chain-of-Experts (CoE, Wang 2025)**: CoE performs "multi-round routing + incremental refinement" within a layer, requiring an independent router each round, with routing costs scales linearly with rounds. DAG-MoE routes only once and leaves multi-step processing to the DAG module. Experiments show DAG-MoE gains 0.107 more PPL than CoE at equal parameters.
- **vs. S′MoRE (Zeng 2025)**: S′MoRE also uses structural aggregation, but the structure is fixed as a tree and used only as a PEFT adapter. DAG-MoE generalizes this to arbitrary DAGs as a backbone, where each token can learn a different structure.
- **vs. DiEP (Bai 2026)**: DiEP uses DAGs for differentiable expert pruning (compression); DAG-MoE does the opposite, using DAGs to increase expressiveness.
- **vs. Fine-grained MoE (He 2024, etc.)**: Fine-grained approaches increase $N$ to expand combinations via "which experts are selected." DAG-MoE expands the "how they are combined" axis. The two axes are orthogonal and can be used together.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Isolating the neglected third component of MoE—aggregation—for expressiveness expansion and linking it to GNN theory.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Includes three model scales, equal-parameter baselines, and CoE/MLP comparisons, though the maximum scale is still small and lacks validation on trillion-token LLMs.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The progression through Prop $\to$ Thm $\to$ Thm is elegant. The boundary between "theory as motivation" and "experiments as evidence" is well-handled, and the OOD vs. in-domain explanation is compelling.
- **Value**: ⭐⭐⭐⭐ Provides a nearly free new axis for MoE improvement (<5% throughput), though the sequential $L$ cost at hyper-scale remains an open question.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] RQ-MoE: Residual Quantization via Mixture of Experts for Efficient Input-Dependent Vector Compression](rq-moe_residual_quantization_via_mixture_of_experts_for_efficient_input-dependen.md)
- [\[ICLR 2026\] Coupling Experts and Routers in Mixture-of-Experts via an Auxiliary Loss](../../ICLR2026/model_compression/coupling_experts_and_routers_in_mixture-of-experts_via_an_auxiliary_loss.md)
- [\[ICLR 2026\] Unveiling Super Experts in Mixture-of-Experts Large Language Models](../../ICLR2026/model_compression/unveiling_super_experts_in_mixture-of-experts_large_language_models.md)
- [\[NeurIPS 2025\] Dense Backpropagation Improves Training for Sparse Mixture-of-Experts](../../NeurIPS2025/model_compression/dense_backpropagation_improves_training_for_sparse_mixture-of-experts.md)
- [\[CVPR 2026\] Enhancing Mixture-of-Experts Specialization via Cluster-Aware Upcycling](../../CVPR2026/model_compression/enhancing_mixture_of_experts_specialization_via_cluster_aware_upcycling.md)

</div>

<!-- RELATED:END -->

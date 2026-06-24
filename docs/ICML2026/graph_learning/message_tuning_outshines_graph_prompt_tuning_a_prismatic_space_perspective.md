---
title: >-
  [Paper Note] Message Tuning Outshines Graph Prompt Tuning: A Prismatic Space Perspective
description: >-
  [ICML 2026][Graph Learning][Graph Foundation Models] This paper proposes **Prismatic Space Theory (PS-Theory)**, treating the frozen GNN foundation model as a layer-wise piecewise linear mapping that performs "prismatic" refraction on the input manifold. It rigorously proves that an upper bound exists for the adaptation capability of graph prompt tuning. Consequently, **Message Tuning (MTG)** is introduced, which injects learnable "message prototypes" into each layer and dyna…
tags:
  - "ICML 2026"
  - "Graph Learning"
  - "Graph Foundation Models"
  - "Graph Prompt Tuning"
  - "Prismatic Space Theory"
  - "Message Tuning"
  - "Geometric Measure Theory"
date: 2026-05-08
content_hash: 23394d1440ab0aad
---

# Message Tuning Outshines Graph Prompt Tuning: A Prismatic Space Perspective

**Conference**: ICML 2026  
**arXiv**: [2606.03290](https://arxiv.org/abs/2606.03290)  
**Code**: https://github.com/CYCUCAS/MTG  
**Area**: Graph Learning / Graph Foundation Model Adaptation  
**Keywords**: Graph Foundation Models, Graph Prompt Tuning, Prismatic Space Theory, Message Tuning, Geometric Measure Theory  

## TL;DR
This paper proposes **Prismatic Space Theory (PS-Theory)**, treating the frozen GNN foundation model as a layer-wise piecewise linear mapping that performs "prismatic" refraction on the input manifold. It rigorously proves that an upper bound exists for the adaptation capability of graph prompt tuning. Consequently, **Message Tuning (MTG)** is introduced, which injects learnable "message prototypes" into each layer and dynamically fuses them with native messages. MTG theoretically breaks the aforementioned upper bound and outperforms existing graph prompt methods across 15 datasets and 6 pre-training strategies.

## Background & Motivation

**Background**: Graph Foundation Models (GFMs) generally follow the "self-supervised pre-training + downstream adaptation" paradigm. GNN backbones (MPNN or Graph Transformer) are pre-trained on large-scale graph data. For downstream tasks, three adaptation methods are common: full fine-tuning, graph prompt tuning (inserting learnable tokens or subgraphs in the input space), and structural adjustments. Graph prompt tuning has become the mainstream paradigm because it updates few parameters and alleviates negative transfer in few-shot scenarios.

**Limitations of Prior Work**: Although graph prompt methods like GPF, All-in-One, and Gprompt perform well empirically, and prior work (Wang et al., 2025a) has explained their effectiveness via "equivalent transformations of input data," **none have strictly characterized the capacity upper bound of graph prompt tuning**. Specifically, what is the maximum extent to which it can "expand" the output space of a frozen model? Is prompt tuning always constrained by a theoretical limit? Without answering this, one cannot judge how far current methods are from the limit or design new methods to break it.

**Key Challenge**: Graph prompt tuning is essentially an additive perturbation in the **input layer** $\bm{X}_\omega = \tilde{\bm{X}} + \bm{c}\bm{p}^\top$, while the frozen GFM is a **composite layer-wise non-linear contraction mapping**. Each ReLU layer "folds" or "projects" away certain dimensions of the input manifold. This implies that the influence of a prompt is inevitably compressed by the product of singular values of layer-wise Jacobians, creating a "rigid" upper bound determined by the backbone's geometric structure.

**Goal**: (1) Establish a mathematical framework to quantitatively characterize the "adaptation capability" of any adaptation method; (2) Rigorously derive the upper bound of graph prompt tuning using this framework; (3) Design a lightweight adaptation method capable of breaking the upper bound based on the revealed bottlenecks.

**Key Insight**: From the perspective of **geometric measure theory**, each layer of a GFM is viewed as a piecewise linear mapping that refracts and compresses the input manifold into low-dimensional prismatic spaces. Adaptation capability is thus transformed into three geometric quantities: intrinsic dimension, Hausdorff measure, and diameter of the adapted output manifold.

**Core Idea**: Since a prompt can only "lever" the geometry of the frozen network from the input layer, it is better to **directly inject learnable parameters inside each backbone layer** to perturb the message fusion process itself—this is the starting point for MTG.

## Method

### Overall Architecture

The core problem addressed is whether graph prompt tuning has a capacity ceiling and how to design a lightweight method to break it. The authors first establish PS-Theory using geometric measure theory, treating a frozen $L$-layer GFM $\Phi=F^{(L)}\circ\cdots\circ F^{(1)}$ as a prism that refracts and compresses the input manifold layer by layer, thereby strictly deriving the upper bound of graph prompt tuning. Based on this, MTG is proposed, moving learnable parameters from the input layer into the message-passing process of each layer to theoretically break the bound. The method only injects a small number of prototype parameters per layer while keeping the backbone frozen, making it applicable to various architectures like GCN, GAT, GIN, and Graph Transformer.

### Key Designs

**1. PS-Theory: Characterizing the upper bound of graph prompt tuning**

To answer whether prompts can break the upper bound, the bound must be mathematically defined. PS-Theory abstracts each layer $F^{(\ell)}$ as a continuous piecewise linear mapping (Propositions 3.3, 3.4). Thus, the input manifold $\mathcal{M}_0$ is refracted by layer-wise Jacobians into descending "prismatic spaces" $\mathcal{M}^{(\ell)}=F^{(\ell)}\circ\cdots\circ F^{(1)}(\mathcal{M}_0)$ (Definition 3.6). By performing SVD on each layer's Jacobian to extract singular values $\sigma_i^{(\ell)}$, Theorem 3.9 provides the contraction factor for the local Hausdorff measure: $\mathcal{H}^s(F^{(\ell)}(\mathbb{S}))=(\prod_{i=1}^s \sigma_i^{(\ell)})\mathcal{H}^s(\mathbb{S})$. This means the manifold measure is compressed by the product of singular values at each layer.

This characterization unifies "all forms of input space perturbations" as "geometric additive deformations of the input manifold." Theorem 3.15 finally proves that for **any prompt $\bm{P}$**, the measure of the adapted output manifold is strictly constrained by the product of singular values determined by the frozen backbone:

$$\mathcal{H}^{d_{\text{int}}}(\mathcal{M}^{(L)}(\bm{P})) \le \Big(\sup_k \prod_{\ell=1}^L \prod_{i=1}^{d_{\text{int}}}\sigma_{i,k}^{(\ell)}\Big)\cdot \mathcal{H}^{d_{\text{int}}}(\mathcal{M}_0(\bm{P}))$$

No matter how the prompt is tuned, it cannot break this "rigid" ceiling determined by the backbone's geometry—this is the cost of graph prompt tuning only performing additive perturbations at the input layer.

**2. Learnable Message Prototypes + Dynamic Message Fusion: The Mechanism of MTG**

Since PS-Theory reveals the limitations of "input-layer levering," the solution is to inject learnable parameters directly into the internal layers. MTG injects $m$ message prototypes $\bm{M}^{(\ell)}\in\mathbb{R}^{m\times d_{\ell-1}}$ into each layer $\ell$. It uses a linear projection with row-wise Softmax to calculate the attention of each node towards these prototypes, which are then fused additively back into the original representation:

$$\mathfrak{F}^{(\ell)}(\bm{H}^{(\ell-1)},\bm{M}^{(\ell)}) = \bm{H}^{(\ell-1)} + \text{Softmax}(\bm{H}^{(\ell-1)}\bm{W}_p^{(\ell)})\cdot \bm{M}^{(\ell)}$$

The fused $\bm{H}_{\bm{M}}^{(\ell-1)}$ is then fed into the original layer's "Attention operator $\mathfrak{A}$ + Message Fusion operator $\mathfrak{M}$ + Update operator $\mathfrak{U}$" triplet (Eq. 15). This is equivalent to dynamically reshaping the input of that layer based on the current sample. The only learnable parameters are $\{\bm{M}^{(\ell)}, \bm{W}_p^{(\ell)}\}$, which are far fewer than the backbone parameters.

While this mechanism draws inspiration from the "layer-wise learnable parameters" of prefix-tuning, the authors emphasize it is not a direct port: prefixes are static external tokens for Transformer sequences, whereas MTG's fusion is **dynamic per-node and per-sample**, using linear projections for efficiency and compatibility with any GNN backbone.

**3. Theoretical proof that MTG strictly transcends the graph prompt upper bound**

PS-Theory serves as both an analytical tool and a design guide: a method intending to break the prompt upper bound must introduce "non-compressive" geometric degrees of freedom into the Jacobian. MTG satisfies this by introducing new learnable directions in each layer, enlarging the supremum of the singular value products $\sup_k \prod_\ell \prod_i \sigma_{i,k}^{(\ell)}$, and refining the partition of linear regions (Definition 3.11).

Theorem 4.1 proves that the final representation space of MTG is no smaller than the maximum attainable by graph prompt tuning across three geometric metrics: intrinsic dimension, Hausdorff measure, and diameter. Furthermore, configurations exist where it is strictly larger, such as $d_{\text{int}}(\mathcal{M}^{(L)}_{\text{MTG}})\ge d_{\text{int}}(\mathcal{M}^{(L)}_{\text{PT}}(\bm{P}))$.

### Loss & Training
The backbone is completely frozen; only $\{\bm{M}^{(\ell)}, \bm{W}_p^{(\ell)}\}_{\ell=1}^L$ are trained. Downstream tasks follow the few-shot node/graph classification losses from the ProG benchmark (Zi et al., 2024). Experiments are repeated 5 times with random hyperparameter searches.

## Key Experimental Results

### Main Results
Based on the ProG benchmark, covering 15 datasets (7 node classification + 8 graph classification, including homogeneous/heterogeneous/large-scale graphs across biological, molecular, and social domains) and 6 pre-training strategies (DGI, GraphMAE, EdgePreGPPT, EdgePreGprompt, GraphCL, SimGRACE). Comparisons include supervised learning, full fine-tuning, GPPT, Gprompt, All-in-One, GPF, and GPF-plus.

| Task / Dataset (Sample) | shot | MTG | Sub-optimal Method | Conclusion |
|---|---|---|---|---|
| Cora (Node) | 5-shot | **Best** | Gprompt 69.03 | MTG wins |
| Citeseer (Node) | 5-shot | **Best** | Gprompt 66.13 | MTG wins |
| Wisconsin (Node) | 3-shot | **Best** | Gprompt 92.52 | Heterogeneous lead |
| ogbn-arxiv (Large) | 5-shot | **Best** | Gprompt (closest) | Scalable |
| 8 Graph Datasets | 1/3/5-shot | **Best** | All-in-One | All-in-One sub-optimal |

Note: MTG achieved the **best performance on all 15 datasets**, with significantly higher parameter efficiency than supervised and full fine-tuning.

### Ablation Study

| Configuration / Perspective | Key Metric | Description |
|---|---|---|
| Full MTG | 15/15 best | Complete method |
| Backbone replacement | Still leading | Verified across GCN, GraphSAGE, GAT, GIN, GT (Appendix F.2) |
| Pre-training strategies | Alleviates neg. transfer | MTG outperforms the supervised baseline across all 6 strategies |
| Prototype number $m$ | Stable | Insensitive to $m$ (Appendix F.4) |
| Computational efficiency | Near GPF | Minimal additional overhead (Appendix F.3) |

### Key Findings
- MTG achieves the best results on 15/15 datasets. Its success stems from "perturbations at every layer," corresponding to the additional singular value degrees of freedom in PS-Theory.
- In few-shot settings, MTG consistently outperforms full fine-tuning, validating that "fewer parameters + adapting backbone geometry" is more stable than "full parameters + destroying pre-trained geometry."
- MTG remains superior even with significantly different backbones like Graph Transformer, proving that "message prototypes + dynamic fusion" is a general backbone-agnostic mechanism rather than one dependent on GCN-style local aggregation.

## Highlights & Insights
- **The Prism Metaphor**: Comparing GFM to a prism is highly intuitive. The Jacobian of ReLU at differentiable points is an idempotent diagonal matrix of 0s and 1s (Corollary 3.10), equivalent to a local projection. This "refract + fold" visual clarifies why prompts cannot break the upper bound.
- **From Analysis to Design**: PS-Theory does more than characterize the bound; it provides the necessary condition—introducing new geometric degrees of freedom at each layer—to break it, moving method design from heuristic to geometry-driven.
- **Engineering Minimalism**: MTG only adds $\bm{M}^{(\ell)}\in\mathbb{R}^{m\times d_{\ell-1}}$ and $\bm{W}_p^{(\ell)}\in\mathbb{R}^{d_{\ell-1}\times m}$ per layer. The fusion logic is a single line (Eq. 17) but remains backbone-agnostic. This "minimum action under theoretical guidance" is a valuable model for PEFT in other domains.
- **Boundaries of Prefix-tuning Analogy**: The authors clarify that MTG is not a direct port of prefix-tuning. MTG modifies the "core operator of message passing" per-node and per-sample, whereas prefix-tuning prepends static external tokens. This distinction prevents the baseline from being misidentified.

## Limitations & Future Work
- Some upper bounds in PS-Theory (e.g., Theorem 3.13) assume the mapping is **injective** on linear region partitions. In reality, ReLU folding often violates injectivity, meaning the actual gap might be tighter than the theoretical one.
- Experiments focus exclusively on **few-shot classification** with relatively small datasets (the largest being ogbn-arxiv). MTG's advantages in full-shot settings, link prediction, or generative tasks remain to be shown.
- While the total number of parameters is small, it grows linearly with the number of layers $L$. Selective insertion of prototypes in extremely deep GFMs could be a subject for future research.
- PS-Theory currently only characterizes geometric contraction and does not draw conclusions regarding the loss landscape or optimization dynamics. The "geometric capacity = learnability" link is empirical rather than theoretically closed.

## Related Work & Insights
- **vs. Graph Prompt Tuning**: Existing methods (GPF, Gprompt, etc.) only perform additive prompting at the input layer. This paper proves they share a geometric upper bound, which MTG breaks via layer-wise injection.
- **vs. Full Fine-tuning**: MTG leaves backbone parameters untouched, preserving pre-trained geometry. Full fine-tuning has maximum freedom but often destroys pre-trained geometry in few-shot settings, leading to negative transfer.
- **vs. Prefix-tuning**: MTG adapts the core idea of "layer-wise learnable parameters" to GNNs but re-engineers the fusion operator $\mathfrak{F}$ to be dynamic and backbone-agnostic, representing a "transfer of thought, redesign of mechanism."
- **vs. Geometric/Manifold Analysis**: Unlike traditional WL-test-based analysis of "separability," this work uses Hausdorff measure, intrinsic dimension, and diameter to quantify "adaptation capacity," providing a fresh perspective outside spectral methods.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ PS-Theory introduces geometric measure theory to GFM adaptation analysis; the framework is original and self-consistent.
- Experimental Thoroughness: ⭐⭐⭐⭐ Broad coverage across 15 datasets and 6 strategies, though lacking full-shot or generative tasks.
- Writing Quality: ⭐⭐⭐⭐ Clear theoretical derivation with the Prism metaphor. The main text relies heavily on the appendix due to space constraints, making independent reading slightly challenging.
- Value: ⭐⭐⭐⭐⭐ Provides both a theoretical ceiling for graph prompt tuning and a concise method to break it, offering directional guidance for the GFM adaptation paradigm.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] GILT: An LLM-Free, Tuning-Free Graph Foundational Model for In-Context Learning](gilt_an_llm-free_tuning-free_graph_foundational_model_for_in-context_learning.md)
- [\[CVPR 2025\] Coeff-Tuning: A Graph Filter Subspace View for Tuning Attention-Based Large Models](../../CVPR2025/graph_learning/coeff-tuning_a_graph_filter_subspace_view_for_tuning_attention-based_large_model.md)
- [\[ICML 2026\] View Space: Representation Learning Across Arbitrary Graphs](view_space_learning_representation_across_arbitrary_graphs.md)
- [\[ACL 2026\] ARK: Answer-Centric Retriever Tuning via KG-augmented Curriculum Learning](../../ACL2026/graph_learning/ark_answer-centric_retriever_tuning_via_kg-augmented_curriculum_learning.md)
- [\[ICML 2025\] Does Graph Prompt Work? A Data Operation Perspective with Theoretical Analysis](../../ICML2025/graph_learning/does_graph_prompt_work_a_data_operation_perspective_with_theoretical_analysis.md)

</div>

<!-- RELATED:END -->

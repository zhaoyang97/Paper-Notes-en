---
title: >-
  [Paper Note] Message Tuning Outshines Graph Prompt Tuning: A Prismatic Space Perspective
description: >-
  [ICML 2026][Graph Learning][Graph Foundation Models] This paper proposes **Prismatic Space Theory (PS-Theory)**, which treats frozen GNN foundation models as layer-wise piecewise linear mappings that "refract" the input…
tags:
  - "ICML 2026"
  - "Graph Learning"
  - "Graph Foundation Models"
  - "Graph Prompt Tuning"
  - "Prismatic Space Theory"
  - "Message Tuning"
  - "Geometric Measure Theory"
date: 2026-05-08
content_hash: ab2586f1d948f72e
---

# Message Tuning Outshines Graph Prompt Tuning: A Prismatic Space Perspective

**Conference**: ICML 2026  
**arXiv**: [2606.03290](https://arxiv.org/abs/2606.03290)  
**Code**: https://github.com/CYCUCAS/MTG  
**Area**: Graph Learning / Graph Foundation Model Adaptation  
**Keywords**: Graph Foundation Models, Graph Prompt Tuning, Prismatic Space Theory, Message Tuning, Geometric Measure Theory  

## TL;DR
This paper proposes **Prismatic Space Theory (PS-Theory)**, which treats frozen GNN foundation models as layer-wise piecewise linear mappings that "refract" the input manifold in a prismatic manner. Based on this, it strictly proves an upper bound on the adaptation capability of graph prompt tuning. Furthermore, it introduces **Message Tuning (MTG)**, which injects learnable "message prototypes" into each layer and dynamically fuses them with native messages. Theoretically, MTG can break through the aforementioned upper bound. Experiments across 15 datasets and 6 pre-training strategies demonstrate that MTG consistently outperforms existing graph prompt methods.

## Background & Motivation

**Background**: Graph Foundation Models (GFMs) generally follow the "self-supervised pre-training + downstream adaptation" pipeline. The GNN backbone (MPNN or Graph Transformer) is pre-trained on large-scale graph data. For downstream tasks, three types of adaptation are commonly used: full-parameter fine-tuning, graph prompt tuning (inserting learnable tokens or subgraphs in the input space), and more aggressive structural adjustments. Graph prompt tuning has become the mainstream paradigm because it updates few parameters and mitigates negative transfer in few-shot scenarios.

**Limitations of Prior Work**: Although graph prompt methods like GPF, All-in-One, and Gprompt perform well experimentally, and existing work (Wang et al., 2025a) explains their efficacy through "equivalent transformation of input data," **no one has strictly characterized the upper bound of graph prompt tuning's capability**. Specifically, how much can it "expand" the output space of a frozen model? Is prompt tuning always constrained by a theoretical upper bound regardless of adjustments? Without answering these questions, it is impossible to judge the gap between current methods and the theoretical limit or to design new methods capable of surpassing it.

**Key Challenge**: Graph prompt tuning essentially performs additive perturbations $\bm{X}_\omega = \tilde{\bm{X}} + \bm{c}\bm{p}^\top$ only at the **input layer**. Meanwhile, a frozen GFM is a **composite layer-wise nonlinear contraction mapping**—each ReLU layer "folds" or "projects" certain dimensions of the input manifold. This implies that the influence of the prompt is inevitably compressed by the product of layer-wise Jacobian singular values, resulting in a "rigid" upper bound determined by the backbone's geometric structure.

**Goal**: (1) Establish a mathematical framework to quantitatively characterize the "adaptation capability" of any adaptation method; (2) strictly derive the capability upper bound of graph prompt tuning using this framework; (3) design a lightweight adaptation method that can break through this bound.

**Key Insight**: From the perspective of **geometric measure theory**, each GFM layer is viewed as a piecewise linear mapping that refracts and compresses the input manifold into a low-dimensional prismatic space. Adaptation capability is thus transformed into three geometric quantities: the intrinsic dimension, Hausdorff measure, and diameter of the adapted output manifold.

**Core Idea**: Since prompts can only "leverage" the geometry of the frozen network at the input layer, it is better to **directly inject learnable parameters inside each backbone layer** to perturb the message fusion process itself—this is the starting point of MTG.

## Method

PS-Theory provides the theory, while MTG provides the method.

### Overall Architecture

Input side: Graph $\mathcal{G}=(\mathcal{V},\mathcal{E})$, node feature matrix $\bm{X}\in\mathbb{R}^{N\times d_0}$, and adjacency matrix $\bm{A}$.

Model side: A frozen $L$-layer GFM $\Phi=F^{(L)}\circ\cdots\circ F^{(1)}$, where each layer is abstracted as a triple: "Attention operator $\mathfrak{A}$ + Message fusion operator $\mathfrak{M}$ + Update operator $\mathfrak{U}$". GCN, GAT, GIN, and Graph Transformer can all be incorporated into this unified form.

Adaptation side: Learnable message prototypes $\bm{M}^{(\ell)}\in\mathbb{R}^{m\times d_{\ell-1}}$ and fusion projection matrices $\bm{W}_p^{(\ell)}$ are injected into each layer. The original layer input $\bm{H}^{(\ell-1)}$ is replaced by $\bm{H}_{\bm{M}}^{(\ell-1)}=\mathfrak{F}^{(\ell)}(\bm{H}^{(\ell-1)}, \bm{M}^{(\ell)})$ to obtain the adapted network $\Phi_{\text{MTG}}$. Backbone parameters remain frozen.

### Key Designs

1. **PS-Theory: Treating GFMs as Prisms and Confining Graph Prompt Tuning**:

    - **Function**: Strictly characterizes the adaptation capability upper bound of graph prompt tuning using geometric measure theory.
    - **Mechanism**: Each layer $F^{(\ell)}$ is first abstracted as a continuous piecewise linear mapping (Propositions 3.3, 3.4). The input manifold $\mathcal{M}_0$ is refracted through layer-wise Jacobians into a "prismatic space" $\mathcal{M}^{(\ell)}=F^{(\ell)}\circ\cdots\circ F^{(1)}(\mathcal{M}_0)$ with decreasing dimensions (Definition 3.6). SVD is then used to extract singular values $\sigma_i^{(\ell)}$ of each Jacobian. Theorem 3.9 provides the contraction factor for the local Hausdorff measure: $\mathcal{H}^s(F^{(\ell)}(\mathbb{S}))=(\prod_{i=1}^s \sigma_i^{(\ell)})\mathcal{H}^s(\mathbb{S})$. Finally, Theorem 3.15 proves that for **any prompt $\bm{P}$**, the measure and diameter of the adapted output manifold are restricted by the product of singular values determined by the frozen backbone: $\mathcal{H}^{d_{\text{int}}}(\mathcal{M}^{(L)}(\bm{P})) \le (\sup_k \prod_{\ell=1}^L \prod_{i=1}^{d_{\text{int}}}\sigma_{i,k}^{(\ell)})\cdot \mathcal{H}^{d_{\text{int}}}(\mathcal{M}_0(\bm{P}))$.
    - **Design Motivation**: To answer "can we break the upper bound," "what the upper bound is" must first be mathematically defined. PS-Theory unifies all forms of input space perturbation as "geometric additive deformations of the input manifold," converting the prompt tuning capability problem into a "geometric contraction problem" to formally prove the superiority of MTG.

2. **Learnable Message Prototypes + Dynamic Message Fusion (Core Mechanism of MTG)**:

    - **Function**: Moves learnable parameters from the input layer into the message passing process of each GFM layer, allowing adaptation to affect the internal geometry of the backbone rather than just the input manifold.
    - **Mechanism**: For each layer $\ell$, $m$ prototypes $\bm{M}^{(\ell)}\in\mathbb{R}^{m\times d_{\ell-1}}$ are injected. A linear projection + row-wise Softmax computes the attention of each node towards each prototype, which is then fused back: $\mathfrak{F}^{(\ell)}(\bm{H}^{(\ell-1)},\bm{M}^{(\ell)}) = \bm{H}^{(\ell-1)} + \text{Softmax}(\bm{H}^{(\ell-1)}\bm{W}_p^{(\ell)})\cdot \bm{M}^{(\ell)}$. The fused $\bm{H}_{\bm{M}}^{(\ell-1)}$ is fed back into the $\mathfrak{A}/\mathfrak{M}/\mathfrak{U}$ triple (Eq. 15), effectively reshaping the GFM layer input "dynamically" based on the current sample. Learnable parameters consist only of $\{\bm{M}^{(\ell)}, \bm{W}_p^{(\ell)}\}$, which are significantly fewer than the backbone parameters.
    - **Design Motivation**: PS-Theory reveals that "layer-wise perturbation" is a necessary condition to break the prompt upper bound. Meanwhile, prefix-tuning is constrained to Transformer sequence inputs. MTG adapts the prefix concept to arbitrary GNNs via "prototypes + node-level dynamic attention." The fusion is **node-wise and sample-wise dynamic** (unlike the static pre-pending in prefix-tuning) and uses only linear projections for efficiency.

3. **Theoretical Proof that MTG Strictly Surpasses the Graph Prompt Bound**:

    - **Function**: Proves that the final representation space of MTG is no smaller than that achievable by graph prompt tuning with any $\bm{P}$ in terms of intrinsic dimension, Hausdorff measure, and diameter, with configurations existing where it is strictly larger (Theorem 4.1).
    - **Mechanism**: MTG introduces new learnable degrees of freedom at each layer, which is equivalent to adding a "non-compressive" direction in the PS-Theory Jacobian, thereby expanding the supremum of the singular value product $\sup_k \prod_\ell \prod_i \sigma_{i,k}^{(\ell)}$. The linear region partitioning (Definition 3.11) also becomes finer, resulting in $d_{\text{int}}(\mathcal{M}^{(L)}_{\text{MTG}})\ge d_{\text{int}}(\mathcal{M}^{(L)}_{\text{PT}}(\bm{P}))$.
    - **Design Motivation**: Transforms PS-Theory from an "analysis tool" into a "design guide"—given a new method intended to break the prompt bound, PS-Theory can be used to audit whether it truly introduces geometric degrees of freedom.

### Loss & Training
The backbone is entirely frozen; only $\{\bm{M}^{(\ell)}, \bm{W}_p^{(\ell)}\}_{\ell=1}^L$ are trained. Downstream tasks follow the few-shot node/graph classification losses of the ProG benchmark (Zi et al., 2024), with 5 repetitions for mean and standard deviation, and random hyperparameter search.

## Key Experimental Results

### Main Results
Based on the ProG benchmark, covering 15 datasets (7 node classification + 8 graph classification, including homogeneous/heterogeneous/large-scale graphs across biological, molecular, and social domains) and 6 pre-training strategies (DGI, GraphMAE, EdgePreGPPT, EdgePreGprompt, GraphCL, SimGRACE). Comparisons include supervised learning, full-parameter fine-tuning, GPPT, Gprompt, All-in-One, GPF, and GPF-plus.

| Task / Dataset (Example) | Shot | MTG | Second Best | Conclusion |
|---|---|---|---|---|
| Cora Node Classification | 5-shot | **Best** | Gprompt 69.03 | MTG wins |
| Citeseer Node Classification | 5-shot | **Best** | Gprompt 66.13 | MTG wins |
| Wisconsin Node Classification | 3-shot | **Best** | Gprompt 92.52 | Heterogeneous graph lead |
| ogbn-arxiv Large Graph | 5-shot | **Best** | Gprompt (closest) | Large-scale scalability |
| 8 Graph Classif. Datasets | 1/3/5-shot | **Best** | All-in-One | All-in-One second best |

Note: The original table covers 1/3/5-shot across 21 columns; this table is compressed. The paper reports that MTG achieves the **best results across all 15 datasets**, with parameter efficiency significantly higher than supervised and full fine-tuning.

### Ablation Study

| Configuration / Perspective | Key Metric | Description |
|---|---|---|
| Full MTG | 15/15 best | Complete method |
| Backbone replacement (GCN→SAGE/GAT/GIN/GT) | Still leads | Validates MTG is backbone-agnostic (App. F.2) |
| Different Pre-training (6 types) | Consistent mitigation | Q3: Mitigates negative transfer (Ours > Supervised) |
| Number of prototypes $m$ | Stable | Not sensitive to $m$ (App. F.4) |
| Efficiency (Time/Memory) | Near GPF | Minimal overhead added (App. F.3) |

### Key Findings
- MTG achieves the best results on 15/15 datasets. The improvement stems from "layer-wise perturbation," which corresponds to the additional singular value degrees of freedom in PS-Theory.
- In few-shot scenarios, MTG consistently outperforms full fine-tuning, validating that "fewer parameters + adapting backbone geometry" is more stable than "full parameters but destroying pre-trained geometry."
- MTG maintains its lead even with drastically different backbones like Graph Transformer, indicating that "message prototypes + dynamic fusion" is a universal mechanism independent of GCN-style local aggregation.

## Highlights & Insights
- **The Prism Metaphor**: Comparing GFMs to prisms is a fitting geometric intuition. The Jacobian of ReLU at differentiable points is an idempotent diagonal matrix (Corollary 3.10), equivalent to a local projection, which is the mathematical formalization of "refraction + folding." This makes the inability of prompts to break the upper bound intuitive.
- **From Analysis Tool to Design Guide**: PS-Theory does more than characterize the upper bound; it provides the necessary condition—introducing new geometric degrees of freedom at each layer—advancing method design from heuristic to geometry-driven.
- **Engineering Simplicity**: Each layer only adds $\bm{M}^{(\ell)}\in\mathbb{R}^{m\times d_{\ell-1}}$ and $\bm{W}_p^{(\ell)}\in\mathbb{R}^{d_{\ell-1}\times m}$. The fusion logic is a single line, yet backbone-agnostic. This "minimal intervention under theoretical guidance" is highly relevant for PEFT methods in NLP and multi-modal learning.
- **Prefix-tuning Distinction**: MTG ≠ direct transfer of prefix-tuning. MTG modifies the core message-passing operator, whereas prefix-tuning prepends static external context. This distinction avoids mischaracterizing the baseline.

## Limitations & Future Work
- Several bounds in PS-Theory (e.g., Theorem 3.13) assume the mapping is **injective** on linear region segments; otherwise, it only provides an upper bound estimate. ReLU folding in GFMs often violates injectivity, implying the real gap may be tighter than the theory suggests.
- Experiments focus on **few-shot classification** with relatively small datasets (the largest being ogbn-arxiv). Whether MTG's advantage holds in full-shot, link prediction, or generative tasks remains to be shown.
- Introducing $L\cdot m\cdot d_{\ell-1}$ parameters scales linearly with **extremely deep** GFMs. Whether every layer needs a prototype is a topic for future research.
- PS-Theory currently only characterizes geometric contraction and does not draw conclusions about downstream loss landscapes or optimization dynamics.

## Related Work & Insights
- **vs Graph Prompt Tuning (GPF, All-in-One, etc.)**: These methods perform additive prompts only at the input layer, corresponding to "perturbation of the input manifold." This paper proves they share the same geometric upper bound, which MTG breaks via layer-wise injection.
- **vs Full Fine-tuning**: MTG leaves backbone parameters frozen, preserving pre-trained geometry. While full fine-tuning has maximum freedom, it is prone to destroying pre-trained geometry in few-shot settings, leading to negative transfer.
- **vs Prefix-tuning (Li & Liang, 2021)**: Prefix-tuning uses static external tokens for Transformer sequences. MTG adopts the "layer-wise learnable parameters" concept for GNNs but redesigned the fusion operator $\mathfrak{F}$ to be dynamic, node-level, and backbone-agnostic.
- **vs Geometric Analysis (WL-based)**: Unlike prior work analyzing "discriminative power" (expressiveness), this work uses Hausdorff measure, intrinsic dimension, and diameter to quantify "adaptation capability" as geometric capacity, offering a new perspective compared to spectral methods.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ PS-Theory introduces geometric measure theory into GFM adaptation analysis; the framework is original and self-consistent.
- Experimental Thoroughness: ⭐⭐⭐⭐ Broad coverage of 15 datasets and 6 pre-training strategies, but lacks full-shot or generative tasks.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are clear with the Prism metaphor; however, many proofs are moved to the appendix due to space.
- Value: ⭐⭐⭐⭐⭐ Establishes a theoretical ceiling for graph prompt tuning and provides a simple way to break it, offering directional guidance for GFM adaptation paradigms.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] GILT: An LLM-Free, Tuning-Free Graph Foundational Model for In-Context Learning](gilt_an_llm-free_tuning-free_graph_foundational_model_for_in-context_learning.md)
- [\[ACL 2026\] ARK: Answer-Centric Retriever Tuning via KG-augmented Curriculum Learning](../../ACL2026/graph_learning/ark_answer-centric_retriever_tuning_via_kg-augmented_curriculum_learning.md)
- [\[AAAI 2026\] Magnitude-Modulated Equivariant Adapter for Parameter-Efficient Fine-Tuning of Equivariant Graph Neural Networks](../../AAAI2026/graph_learning/magnitude-modulated_equivariant_adapter_for_parameter-efficient_fine-tuning_of_e.md)
- [\[ICML 2026\] View Space: Representation Learning Across Arbitrary Graphs](view_space_learning_representation_across_arbitrary_graphs.md)
- [\[NeurIPS 2025\] S'MoRE: Structural Mixture of Residual Experts for Parameter-Efficient LLM Fine-tuning](../../NeurIPS2025/graph_learning/smore_structural_mixture_of_residual_experts_for_parameter-efficient_llm_fine-tu.md)

</div>

<!-- RELATED:END -->

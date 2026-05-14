---
title: >-
  [Paper Note] One Prompt Fits All: Universal Graph Adaptation for Pretrained Models
description: >-
  [NeurIPS 2025][LLM Pretraining][graph prompt learning] This paper theoretically proves that representation-level graph prompts are essentially equivalent to linear probes…
tags:
  - "NeurIPS 2025"
  - "LLM Pretraining"
  - "graph prompt learning"
  - "pretrained GNN"
  - "few-shot"
  - "graph topology"
  - "kNN graph"
date: 2026-05-08
content_hash: 2d155a88e209d2db
---

# One Prompt Fits All: Universal Graph Adaptation for Pretrained Models

**Conference**: NeurIPS 2025
**arXiv**: [2509.22416](https://arxiv.org/abs/2509.22416)
**Code**: [GitHub](https://github.com/hedongxiao-tju/UniPrompt)
**Area**: LLM Pretraining
**Keywords**: graph prompt learning, pretrained GNN, few-shot, graph topology, kNN graph

## TL;DR
This paper theoretically proves that representation-level graph prompts are essentially equivalent to linear probes, and on this basis proposes UniPrompt—an input-level method based on a learnable kNN topological prompt graph. By fusing the prompt graph with the original graph via a bootstrapping strategy, UniPrompt consistently outperforms existing graph prompt learning methods on both in-domain and cross-domain few-shot node classification.

## Background & Motivation
**Background**: Graph Prompt Learning (GPL) is an emerging paradigm for adapting pretrained graph models to downstream tasks—freezing the pretrained graph encoder parameters and training only a lightweight prompt module. Existing GPL methods are categorized into three types by prompt position: input-level (feature/edge prompts), layer-level (injecting prompts at each layer), and representation-level (adding prompt tokens or prototype subgraphs to encoder outputs).

**Limitations of Prior Work**:
- **Unclear mechanisms**: Why do prompts at different positions work? Performance varies widely across methods, yet a unified understanding is lacking.
- **Poor adaptability**: Most GPL methods suffer significant performance degradation when the pretrained model is changed, sometimes performing even worse than simple linear probing (fine-tuning only the classification head). This is especially pronounced in cross-domain settings (e.g., homophilic → heterophilic graphs).

**Key Challenge**: GPL methods claim to better preserve pretrained knowledge than fine-tuning, yet experiments suggest that many GPL methods may only be performing classification head adaptation, without genuinely "unleashing" the pretrained model's capability.

**Goal**: (1) Clarify the underlying mechanisms of different types of graph prompts; (2) Propose a universal GPL method that performs stably across arbitrary pretrained models, including cross-domain and heterophilic graph settings.

**Key Insight**: Through theoretical analysis, the paper proves that representation-level prompts are equivalent to linear probes, and thereby proposes the design principle that "prompts should focus on unleashing pretrained model capability (input-level), while the classification head handles downstream task adaptation."

**Core Idea**: Use a learnable kNN topological prompt graph to modify the input graph structure and release the capability of the frozen pretrained model, while using a linear classification head to adapt to the downstream task.

## Method

### Overall Architecture
Given a frozen pretrained graph encoder $f_\theta$ and a downstream few-shot task, UniPrompt: (1) constructs a kNN graph based on cosine similarity of node features as the initial topological prompt; (2) learns gating weights for each edge; (3) progressively fuses the prompt graph and the original graph via a bootstrapping strategy; (4) feeds the fused graph into the frozen encoder to obtain representations, which are then passed to a trainable classification head for prediction. Only the prompt edge weights and classification head parameters are optimized.

### Key Designs

1. **Theorem 4.1: Representation-Level Prompts Are Equivalent to Linear Probes**:

    - Function: Proves that prompts operating in the representation space are essentially training a classifier.
    - Core conclusion: For any linear prompt $T(\mathbf{h}) = \mathbf{W}_T\mathbf{h} + \mathbf{b}_T$ and classifier $C(\mathbf{h}) = \mathbf{W}_C^\top\mathbf{h}$, the composition $C \circ T$ is equivalent to a linear classifier $C'$ in both function space and optimization objective.
    - Design Motivation: Explains why representation-level GPL methods exhibit unstable performance across different pretrained models—they do not exploit the unique advantages of prompting, but merely perform classification head training.

2. **kNN Topological Prompt Initialization**:

    - Function: Constructs an initial prompt graph based on feature similarity.
    - Mechanism: $(\tilde{\mathbf{A}}_{\text{init}})_{ij} = \mathbf{S}_{ij}$ if $\mathbf{S}_{ij} \in \text{top-}k\{\mathbf{S}_{i\cdot}\}$, where $\mathbf{S}_{ij} = \frac{\mathbf{x}_i\mathbf{x}_j^\top}{\|\mathbf{x}_i\|_2\|\mathbf{x}_j\|_2}$
    - Design Motivation: The kNN graph is based on the local structure of the feature space and does not depend on the original graph topology, thus providing meaningful initialization even on heterophilic graphs (where original edges connect nodes of different classes).

3. **Learnable Edge Gating**:

    - Function: Learns importance weights for each edge in the initial prompt graph.
    - Mechanism: $\tilde{\mathbf{A}}_{ij} = \text{ELU}(w_{ij} \cdot \alpha - \alpha) + 1$, using a scaled-shifted ELU to ensure non-negative weights; the model can learn to prune (weight → 0) or amplify certain edges.
    - Design Motivation: The kNN initialization is not necessarily optimal; learnable gating allows the prompt graph to adaptively adjust its topology.

4. **Bootstrapped Progressive Fusion**:

    - Function: Progressively integrates the prompt graph into the original graph.
    - Mechanism: $\hat{\mathbf{A}}^{(t)} = \tau\hat{\mathbf{A}}^{(t-1)} + (1-\tau)\tilde{\mathbf{A}}$, where the temperature coefficient $\tau \in [0,1]$ controls the fusion rate and $\hat{\mathbf{A}}^{(0)} = \mathbf{A}$.
    - Design Motivation: Directly replacing the original graph tends to cause overfitting and model collapse under few-shot settings; progressive fusion retains original graph information while introducing prompt topology.

### Loss & Training
- Standard cross-entropy loss: $\min_{\phi,\Psi} \frac{1}{|\mathcal{V}_L|}\sum_{v_i \in \mathcal{V}_L} \ell_D(g_\phi(f_\theta(p_\Psi(\mathbf{A}, \mathbf{X}))_i), y_i)$
- Only the prompt parameters $\Psi$ (edge weights) and classification head $\phi$ are optimized; the pretrained encoder $\theta$ is kept fully frozen.

## Key Experimental Results

### Main Results
1-shot node classification (DGI pretraining):

| Method | Cora | Cornell | Texas | Wisconsin | Actor |
|--------|------|---------|-------|-----------|-------|
| Linear-probe | 49.77 | 34.56 | 36.21 | 28.71 | 21.33 |
| GPPT | 37.59 | 29.01 | 31.26 | 28.56 | 19.81 |
| GraphPrompt | 49.70 | 22.29 | 27.62 | 22.62 | 19.84 |
| GPF | 51.68 | 26.76 | 34.04 | 26.59 | 20.31 |
| **UniPrompt** | **~52** | **~38** | **~40** | **~36** | **~22** |

### Ablation Study

| Configuration | Description |
|---------------|-------------|
| Classification head only (linear probe) | Stable performance but unable to exploit the advantages of prompting |
| Prompt only (no classification head adaptation) | Poor performance on heterophilic graphs |
| Full model (prompt + classification head) | Best performance, validating the complementarity of both components |
| Without bootstrapping (direct replacement) | Severe overfitting in few-shot settings |
| Different $k$ and $\tau$ values | $k$=5–10 and $\tau$=0.5–0.8 are generally optimal |

### Key Findings
- **Representation-level GPL methods (GPPT, GraphPrompt) show large performance fluctuations when the pretrained model is changed**, sometimes even underperforming linear probing—validating Theorem 4.1.
- **Most significant gains on heterophilic graphs**: UniPrompt outperforms existing GPL methods by 5–10 percentage points on Cornell/Texas/Wisconsin, because the kNN topological prompt does not rely on original edges (which may be noisy).
- **Effectiveness in cross-domain settings**: UniPrompt maintains stable performance when transferring from a pretrained domain to a different downstream domain.
- **Strong simple baseline**: Linear probing already matches or even surpasses complex GPL methods in many scenarios.

## Highlights & Insights
- **The theoretical finding that "representation-level prompts = linear probes"** carries significant implications for the GPL field as a whole: it explains why many GPL methods appear effective but are essentially only performing simple classification head training. This compels the field to reconsider the design objectives of prompts.
- **Clear design principle**: "Prompts unleash pretrained capability; classification heads adapt to downstream tasks"—separating two fundamentally different objectives to avoid conflation. This principle is transferable to other prompt learning settings (e.g., VLMs).
- **Handling of heterophilic graphs via kNN topological prompts** reflects a clever insight: the original edges of heterophilic graphs are unreliable, but kNN relationships in the feature space remain meaningful.

## Limitations & Future Work
- **Only node classification is evaluated**: Other graph tasks such as graph classification and edge prediction are not addressed.
- **kNN computation is expensive on large graphs**: The $O(N^2)$ similarity computation is impractical at large scale.
- **Theoretical analysis is limited to linear prompts/classifiers**: The equivalence relationship for nonlinear prompts (e.g., GNN prompt modules) is not analyzed.
- **Future directions**: Approximate kNN for acceleration; extension to graph classification and link prediction; theoretical analysis of nonlinear prompt properties.

## Related Work & Insights
- **vs. GPPT**: Representation-level prompt, theoretically equivalent to linear probing, and unstable across pretrained models.
- **vs. GPF/GPF+**: Feature-level input prompts; also input-level but modifies only features, not topology.
- **vs. EdgePrompt**: Also edge-level prompting, but uses fixed strategies rather than learnable kNN + bootstrapping.
- **vs. Linear probe**: UniPrompt significantly outperforms linear probing on heterophilic graphs, demonstrating the additional value of input-level prompts.

## Rating
- Novelty: ⭐⭐⭐⭐ The theoretical analysis ("representation-level prompts = linear probes") is original; the method design (kNN topological prompts + bootstrapping) is relatively natural.
- Experimental Thoroughness: ⭐⭐⭐⭐ 9 datasets × 3 pretrained models, in-domain + cross-domain, with comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ The chain from motivating experiments → theoretical analysis → method design is clear.
- Value: ⭐⭐⭐⭐ Provides a unified theoretical perspective and practical method for graph prompt learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Understanding and Enhancing Mask-Based Pretraining towards Universal Representations](understanding_and_enhancing_mask-based_pretraining_towards_universal_representat.md)
- [\[NeurIPS 2025\] Does Object Binding Naturally Emerge in Large Pretrained Vision Transformers?](does_object_binding_naturally_emerge_in_large_pretrained_vision_transformers.md)
- [\[CVPR 2026\] Evidential Transformation Network: Turning Pretrained Models into Evidential Models for Post-hoc Uncertainty Estimation](../../CVPR2026/llm_pretraining/evidential_transformation_network_post_hoc_uncertainty_estimation.md)
- [\[ICCV 2025\] ETA: Energy-based Test-time Adaptation for Depth Completion](../../ICCV2025/llm_pretraining/eta_energy-based_test-time_adaptation_for_depth_completion.md)
- [\[NeurIPS 2025\] The Curse of Depth in Large Language Models](the_curse_of_depth_in_large_language_models.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] From Sequence to Structure: Uncovering Substructure Reasoning in Transformers
description: >-
  [NeurIPS 2025][Graph Learning][Transformer interpretability] This paper presents empirical and theoretical analyses revealing how decoder-only Transformers understand graph structure from text sequences. It proposes "Induced Substructure Filtration" (ISF) to explain the layer-wise substructure identification mechanism, and extends this framework to validate consistency in LLMs, support compositional graph reasoning (Thinking-in-Substructures), and enable substructure extraction in attributed graphs (molecular graphs).
tags:
  - NeurIPS 2025
  - Graph Learning
  - Transformer interpretability
  - substructure extraction
  - graph reasoning
  - induced subgraph filtration
  - LLM graph understanding
date: 2026-05-08
content_hash: 52a208ebeeebe247
---

# From Sequence to Structure: Uncovering Substructure Reasoning in Transformers

**Conference**: NeurIPS 2025
**arXiv**: [2507.10435](https://arxiv.org/abs/2507.10435)
**Code**: [https://github.com/DDigimon/From_Sequence_to_Structure](https://github.com/DDigimon/From_Sequence_to_Structure)
**Area**: Graph Learning
**Keywords**: Transformer interpretability, substructure extraction, graph reasoning, induced subgraph filtration, LLM graph understanding

## TL;DR
This paper presents empirical and theoretical analyses revealing how decoder-only Transformers understand graph structure from text sequences. It proposes "Induced Substructure Filtration" (ISF) to explain the layer-wise substructure identification mechanism, and extends this framework to validate consistency in LLMs, support compositional graph reasoning (Thinking-in-Substructures), and enable substructure extraction in attributed graphs (molecular graphs).

## Background & Motivation

**Background**: LLMs have been shown to solve graph reasoning tasks—even when graph structure is embedded in textual descriptions, LLMs can identify node connectivity, detect patterns, and compare subgraphs. However, Transformers are fundamentally sequence models and do not possess native graph-structural understanding.

**Limitations of Prior Work**: Existing understanding of Transformer graph reasoning has focused on **shortest path** tasks (e.g., spectral navigation in SLN or path composition in ALPINE), which involve linear structures. Real-world graphs contain numerous nonlinear substructures (cycles, trees, motifs), and existing theory cannot explain how Transformers process these.

**Key Challenge**: Transformers are sequence-processing architectures, yet they must reason over non-Euclidean structures (graphs). A theoretical gap exists in generalizing from paths to arbitrary substructures.

**Goal**: (a) How do Transformers identify substructures layer by layer? (b) How do input formats (adjacency list vs. edge list) and query prompts affect identification? (c) Are these findings consistent in LLMs? (d) Can these findings be extended to more complex graph reasoning?

**Key Insight**: The paper focuses on the substructure extraction task (more general than paths) and uses visualization combined with theoretical analysis to reveal the internal mechanisms of Transformers.

**Core Idea**: Transformers progressively identify substructures through a layer-wise "Induced Substructure Filtration" (ISF) process, and this mechanism is consistently present in both from-scratch trained Transformers and LLMs.

## Method

### Overall Architecture
The input is a concatenated sequence $Q$ consisting of the graph's textual representation $\text{Encoder}_G(G)$ and a query prompt $\text{Encoder}_T(T)$, and the Transformer outputs the node set $\text{Encoder}_A(\hat{G})$ of the identified substructure. The study operates at three levels:
1. **Internal mechanism analysis** (Section 3.1): the ISF process
2. **Input query influence** (Section 3.2): graph representation format and prompt type
3. **Extended applications** (Sections 4–5): LLM validation, compositional graph reasoning, attributed graphs

### Key Designs

1. **Induced Substructure Filtration (ISF)**:

    - **Function**: Explains how Transformers progressively identify substructures across layers.
    - **Mechanism**: Defines a $k$-Node $m$-Filtration $\mathcal{F}(V')=(V'_1,\dots,V'_m)$, where $\emptyset\neq V'_1\subseteq\cdots\subseteq V'_m=V'$, corresponding to a sequence of induced subgraphs $(G'_1,\dots,G'_m)$. Each Transformer layer corresponds to one filtration step—starting from the empty set and progressively aggregating nodes until the complete substructure is identified.
    - **Theoretical Support**: **Theorem 3.3** (progressive identification expressiveness) proves that a log-precision Transformer with $m+2$ layers and $O(n^k)$ hidden dimension can output the subgraph isomorphism indicator tensor $\mathcal{T}(G,G'[V'_i])$ at layer $i+2$. **Theorem 3.5** proves that under a unique substructure assumption, a constant-depth Transformer can output the correct $k$-tuple matching.
    - **Visualization Validation**: t-SNE projections show that in layer 2, substructures sharing partial nodes begin to cluster; in layer 3, those sharing more nodes move closer together; the final layer achieves complete separation. The answer is determined before the generation step.

2. **Simultaneous Detection of Multiple Substructures**:

    - **Single-Shape-Multi-Num**: Multiple substructures of the same shape exist in the graph. Theorem 3.8 proves a constant-depth Transformer can complete this task; experiments achieve >85% accuracy on triangle/square detection, with little performance dependence on substructure count.
    - **Multi-Shape-Single-Num**: Substructures of different shapes coexist in the graph. Training data of different shapes are mixed, and the Transformer is guided by query prompts to identify a specific shape. Theorem 3.10 proves this is achievable for any target subgraph with $|V'|=k'\leq k$. An interesting finding: simpler substructures (e.g., triangles) are identified at shallower layers (layer 3), while more complex substructures are deferred to deeper layers.

3. **Input Query Format Analysis**:

    - **Graph Representation**: Adjacency lists (AL) and edge lists (EL) are theoretically equivalent (both can be transformed into a vectorization $\text{vec}(A(G))$ of the adjacency matrix $A(G)$), but EL requires more tokens (for a fully connected graph: $3(|V|^2-|V|)$ vs. $2|V|^2-|V|$ for AL), and thus requires more layers in practice.
    - **Query Prompts**: Both terminology-based ("triangle") and topology-based ("A:BC,B:C") prompts are functional, but terminology-based prompts perform better (>85% vs. ~80%). Perturbation experiments reveal that Transformers do not fully parse the topological description; instead, they abstract it into key token sequences to determine the substructure type.

4. **LLM Consistency Validation (Section 4)**:

    - ISF is validated on fine-tuned LLaMA 3.1-8B-Instruct: t-SNE visualizations exhibit similar layer-wise clustering behavior, and ARI/NMI metrics increase with layer depth.
    - Difference: LLMs generate explanatory content (23% include Python code), causing a slight drop in ARI/NMI at terminal layers.

5. **Thinking-in-Substructures (Tins, Section 5.1)**:

    - **Function**: Decomposes complex substructures into simpler components for reasoning.
    - **Mechanism**: The output is reformatted as $\text{ANS}_{\text{Tins}}=(\{P_1\},\{P_2\},\dots,\{P_t\},\text{<ANS>},\text{ANS}(\hat{G}))$, where $\{P_i\}$ are the decomposed substructures.
    - **Effect**: Reduces expressiveness requirements from $O(n^k)$ to $O(n^q)$ ($q<k$); with 100K training samples, performance improves by ~10%, and the diagonal structure achieves a 46% gain at 3 layers.

6. **Attributed Graph Extension (Section 5.2)**:

    - Node features (e.g., atom types) are encoded into the adjacency list $\text{AL}_f(G)$.
    - Functional group extraction is tested on molecular graphs: Hydroxyl 92.07%, Carboxyl 91.59%, Benzene 72.45%, Mixed 89.46%.

## Key Experimental Results

### Main Results (Single Substructure Extraction Accuracy)

| Training Size | Layers | Triangle | Square | Pentagon | House | Diamond |
|---------------|--------|----------|--------|----------|-------|---------|
| 100K | 2 | 0.530 | 0.194 | 0.363 | 0.371 | 0.066 |
| 100K | 3 | 0.966 | 0.399 | 0.564 | 0.560 | 0.164 |
| 300K | 4 | 0.995 | 0.940 | 0.863 | 0.839 | 0.877 |
| 400K | 5 | **0.998** | **0.968** | **0.892** | **0.853** | **0.931** |

### Ablation Study: Thinking-in-Substructures

| Substructure | Direct (4L) | Tins (4L) | Direct (3L) | Tins (3L) |
|--------------|-------------|-----------|-------------|-----------|
| Diagonal | 0.631 | **0.865** | 0.200 | **0.661** |
| Diamond | 0.476 | **0.779** | 0.129 | **0.434** |
| House | 0.589 | **0.807** | 0.564 | **0.668** |
| Complex | 0.118 | **0.227** | 0.121 | **0.212** |

### Key Findings
- **Layer depth correlates positively with substructure size**: 3-node substructures (triangles) reach 99% accuracy at 3 layers; 4-node substructures require 4 layers for 85%+; 5-node substructures require 5 layers. This is consistent with the ISF theory ($m+2$ layers).
- **Training data volume also matters**: Pentagon accuracy improves from 36% to 89% as training data scales from 100K to 400K, indicating that complex substructures require more samples.
- **AL format outperforms EL**: At equivalent layer counts, AL achieves higher accuracy due to more compact token representation and higher information density.
- **Terminology prompts outperform topology prompts**: Transformers tend to abstract substructure concepts into token sequences rather than fully parsing topological descriptions.
- **Tins significantly improves performance under limited training data**: By decomposing complex structures, it reduces demands on model capacity and training data volume.

## Highlights & Insights
- The **ISF theoretical framework** provides a unified perspective for understanding Transformer graph reasoning: the topological concept of filtration elegantly characterizes the layer-wise information aggregation behavior. This framework explains not only paths (as in prior work) but also general substructures such as cycles and trees.
- The finding that **"Transformers do not truly understand topology, but rather abstract it into token patterns"** is particularly insightful: perturbation experiments show that Transformers rely on specific token cues rather than complete structural comprehension, suggesting the limits of LLM graph reasoning capability.
- **Tins (intermediate decomposition reasoning)** is a clever variant of chain-of-thought: decomposing complex graph structures into simpler components reduces the expressiveness requirement from $O(n^k)$ to $O(n^q)$. This idea generalizes to other combinatorial optimization tasks.
- Validation on LLaMA confirms that ISF is not a phenomenon specific to small models, but an inherent behavior of decoder-only architectures.

## Limitations & Future Work
- The Transformers used in experiments are lightweight GPT-2 variants (384 hidden dim); although LLaMA validates consistency, quantitative analysis on larger-scale models is absent.
- Synthetic graph data (4–16 nodes) differs substantially in scale and structure from real-world knowledge graphs and social networks; whether ISF holds on large-scale graphs remains to be verified.
- Theorem 3.3 requires $O(n^k)$ hidden dimension, which is unrealistic for large graphs ($n=100, k=5$ would require $10^{10}$ dimensions); how to circumvent this theoretical bottleneck in practice is not discussed.
- Attributed graph experiments are limited to a preliminary validation on molecular graphs; richer node/edge feature types (e.g., continuous values, multimodal features) are not addressed.
- Tins requires manually defined decomposition strategies; automated decomposition is a natural direction for future work.

## Related Work & Insights
- **vs. ALPINE**: ALPINE proves that GPT layers can embed adjacency matrices and reachability matrices, but is limited to path tasks. The ISF framework generalizes to arbitrary substructures, representing a direct extension of ALPINE.
- **vs. SLN (Spectral Line-Graph Navigation)**: SLN finds that Transformers learn spectral embeddings rather than Dijkstra-style rules on shortest-path tasks. ISF provides a more general explanation: substructure identification proceeds via filtration rather than spectral methods.
- **vs. GraphPatt**: GraphPatt evaluates LLMs' graph pattern understanding and reports that terminology prompts outperform topology prompts. This paper explains this phenomenon at the mechanistic level—Transformers abstract substructures into token sequences.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The ISF framework is the first to explain Transformer substructure reasoning from a topological perspective in a unified manner, with theory and experiments mutually reinforcing.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers single/multi-substructure settings, different graph representations, LLM validation, Tins, and attributed graphs, though at limited scale.
- **Writing Quality**: ⭐⭐⭐⭐ The logical progression is clear (single→multi→LLM→extensions) and theoretical definitions are rigorous; however, the density of content requires consulting the appendix for some details.
- **Value**: ⭐⭐⭐⭐⭐ Represents a significant advance on the fundamental question of how LLMs understand graph structure; ISF can guide future graph reasoning method design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Relieving the Over-Aggregating Effect in Graph Transformers](relieving_the_over-aggregating_effect_in_graph_transformers.md)
- [\[NeurIPS 2025\] Unifying and Enhancing Graph Transformers via a Hierarchical Mask Framework](unifying_and_enhancing_graph_transformers_via_a_hierarchical_mask_framework.md)
- [\[NeurIPS 2025\] SSTAG: Structure-Aware Self-Supervised Learning Method for Text-Attributed Graphs](sstag_structure-aware_self-supervised_learning_method_for_text-attributed_graphs.md)
- [\[NeurIPS 2025\] Reasoning Meets Representation: Envisioning Neuro-Symbolic Wireless Foundation Models](reasoning_meets_representation_envisioning_neuro-symbolic_wireless_foundation_mo.md)
- [\[NeurIPS 2025\] Deliberation on Priors: Trustworthy Reasoning of Large Language Models on Knowledge Graphs](deliberation_on_priors_trustworthy_reasoning_of_large_language_models_on_knowled.md)

</div>

<!-- RELATED:END -->

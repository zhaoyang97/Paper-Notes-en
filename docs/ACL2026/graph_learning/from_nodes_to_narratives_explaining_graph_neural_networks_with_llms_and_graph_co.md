---
title: >-
  [Paper Note] From Nodes to Narratives: Explaining Graph Neural Networks with LLMs and Graph Context
description: >-
  [ACL 2026][Graph Learning][GNN Explainability] This paper proposes Gspell, a lightweight post-hoc explanation framework. By projecting GNN node embeddings into the LLM embedding space and constructing hybrid prompts (sof…
tags:
  - "ACL 2026"
  - "Graph Learning"
  - "GNN Explainability"
  - "LLM Explainer"
  - "Soft Prompt"
  - "Text-Attributed Graphs"
  - "Natural Language Explanation"
date: 2026-05-08
content_hash: 031a309439417ca9
---

# From Nodes to Narratives: Explaining Graph Neural Networks with LLMs and Graph Context

**Conference**: ACL 2026  
**arXiv**: [2508.07117](https://arxiv.org/abs/2508.07117)  
**Code**: None  
**Area**: Graph Learning / Explainability  
**Keywords**: GNN Explainability, LLM Explainer, Soft Prompt, Text-Attributed Graphs, Natural Language Explanation

## TL;DR

This paper proposes Gspell, a lightweight post-hoc explanation framework. By projecting GNN node embeddings into the LLM embedding space and constructing hybrid prompts (soft prompts + text), it enables LLMs to reason directly on internal GNN representations to generate natural language explanations and explanatory subgraphs. It achieves a strong balance between faithfulness and interpretability on text-attributed graphs (TAGs).

## Background & Motivation

**Background**: GNNs are widely applied in high-stakes fields such as healthcare, drug design, and recommendation systems, where the trustworthiness of predictions is critical. Existing GNN explanation methods (e.g., GNNExplainer, PGExplainer) primarily output subgraph masks or feature importance scores, which perform poorly on text-attributed graphs (TAGs) and lack human readability. Meanwhile, the integration of LLMs and GNNs mainly focuses on enhancing GNN task performance rather than explaining GNN predictions.

**Limitations of Prior Work**: (1) Existing LLM-GNN explanation frameworks rely on rigid templates to align GNN explainer outputs with LLM inputs, requiring manual scoring or additional training; (2) Current methods do not directly utilize internal GNN representations, leading to explanations that generalize poorly or are unfaithful to how the GNN actually works; (3) Invoking external GNN explainers may bias LLM reasoning—if the explainer itself is noisy, it misleads the LLM's judgment.

**Key Challenge**: There is a fundamental misalignment between the GNN embedding space and the LLM token space—how can an LLM "see" and understand the internal representations of a GNN instead of relying solely on second-hand information from external explainers?

**Goal**: Design a post-hoc explanation framework that requires no external explainer and directly injects internal GNN representations into the LLM to generate faithful and interpretable natural language explanations.

**Key Insight**: Analogous to multi-modal alignment (e.g., CLIP aligning image and text embeddings), GNN embeddings are projected as soft prompt tokens for the LLM. This allows the LLM to utilize the structural information learned by the GNN while leveraging its own linguistic reasoning capabilities.

**Core Idea**: Train a projector to map GNN node embeddings to the LLM token space and construct hybrid prompts interleaving soft prompts with text. This allows the LLM to generate natural language explanations directly from GNN representations and extract explanatory subgraphs.

## Method

### Overall Architecture

Gspell consists of three steps: (1) **Projector Training**—mapping GNN embeddings to the LLM token space via contextual alignment loss and contrastive loss; (2) **Hybrid Prompt Construction**—interleaving projected embeddings as soft prompts with node text descriptions; (3) **Explanation Generation**—the LLM generates natural language explanations based on hybrid prompts and extracts supporting/opposing/neutral nodes to construct an explanatory subgraph.

### Key Designs

1.  **GNN-LLM Embedding Projector**:
    - **Function**: Aligns GNN node embeddings to the LLM token embedding space.
    - **Mechanism**: A projector $\Pi: \mathbb{R}^m \to \mathbb{R}^{k \times h}$ maps the GNN embedding $f_\Phi(v)$ to $k$ soft prompt tokens. Two losses are optimized: (a) Contextual alignment loss—encouraging the average soft token representation to align with the LLM embedding of node text (cosine similarity); (b) Contrastive loss—maintaining the similarity structure of GNN embeddings within the projected space (KL divergence minimization).
    - **Design Motivation**: GNN embeddings and LLM tokens exist in entirely different spaces; direct injection leads to semantic mismatch. The dual loss ensures the projection preserves GNN structural information while aligning with the LLM semantic space.

2.  **Hybrid Prompt Construction**:
    - **Function**: Inputs GNN structural and textual information into the LLM in a unified format.
    - **Mechanism**: For a target node $v$, its GNN computation tree $\mathcal{T}^{\phi}_v$ is constructed (a tree of depth $L$, the number of GNN layers). The soft prompt embeddings for each node in the computation tree are interleaved with their text descriptions. The LLM input sequence follows: System Prompt → Target Node Soft Prompt + Text → Computation Tree Nodes (individual soft prompts + text) → Query Instruction.
    - **Design Motivation**: Treating GNN embeddings as "native tokens" allows the LLM to reason about structural representations and textual features simultaneously, rather than relying on text alone.

3.  **Subgraph Extraction & Hallucination Mitigation**:
    - **Function**: Extracts structured explanatory subgraphs from LLM-generated natural language explanations.
    - **Mechanism**: The LLM predicts whether each node in the computation tree is supporting (+1), opposing (-1), or neutral (0) regarding the target node classification. The support set $S^+_v$ forms the explanatory subgraph. Hallucinations are mitigated through two mechanisms: (a) utilizing GNN embedding constraints for LLM reasoning; (b) post-processing verification to ensure cited nodes exist within the computation tree.
    - **Design Motivation**: Pure text explanations are not structured enough, while subgraph explanations are less readable—combining both provides a complete explanation.

### Loss & Training

The projector training loss is $\mathcal{L} = \beta \mathcal{L}_{context} + (1-\beta) \mathcal{L}_{contrast}$. Both the GNN and LLM are frozen; only the projector is trained. No additional fine-tuning is required during inference, making it plug-and-play.

## Key Experimental Results

### Main Results

**Explanation Quality for Node Classification (Cora Dataset)**

| Method | Fidelity+ ↑ | Fidelity- ↓ | Sparsity ↑ | Insightfulness ↑ |
| :--- | :--- | :--- | :--- | :--- |
| GNNExplainer | 0.12 | 0.08 | 0.65 | — |
| PGExplainer | 0.15 | 0.10 | 0.70 | — |
| GraphLLM | 0.18 | 0.12 | 0.55 | 2.8 |
| **Gspell** | **0.22** | **0.06** | **0.72** | **3.5** |

### Ablation Study

| Configuration | Fidelity+ | Sparsity | Insightfulness |
| :--- | :--- | :--- | :--- |
| No Soft Prompts (Pure Text) | 0.14 | 0.68 | 2.9 |
| No Contrastive Loss | 0.18 | 0.70 | 3.2 |
| No Contextual Alignment | 0.16 | 0.69 | 3.0 |
| **Full Gspell** | **0.22** | **0.72** | **3.5** |

### Key Findings

- The introduction of soft prompts significantly improves faithfulness (+0.08 Fidelity+), proving that internal GNN representations provide information that traditional text inputs cannot capture.
- The two components in the dual loss design are complementary—contextual alignment ensures semantic consistency, while contrastive loss maintains structural information.
- Gspell leads significantly in insightfulness (an evaluation metric based on human assessment), proving that natural language explanations are easier for humans to understand than subgraph masks.
- The plug-and-play nature (no need to fine-tune GNNs or LLMs) makes it suitable for deployed models.

## Highlights & Insights

- The approach of "bypassing traditional GNN explainers and letting the LLM directly interpret internal GNN representations" is simple yet powerful—it reduces information loss and bias in intermediate steps.
- The design of the contrastive loss in projector training is clever—it requires not only individual embedding alignment but also consistency of relative relationships between embeddings across the two spaces.
- The interleaved design of the hybrid prompt allows the LLM to "see" both the numerical representation and text description of each node, facilitating multi-perspective reasoning.

## Limitations & Future Work

- Validated only on node classification tasks; explanations for graph-level classification or link prediction were not explored.
- Projector quality depends on the separability of GNN embeddings—if the GNN embedding space structure is chaotic, projection effectiveness may decrease.
- Explanatory subgraph extraction depends on LLM output parsing, which may be affected by the LLM's capacity for format following.
- Constructing the computation tree requires knowledge of the GNN architecture's layer count, limiting applicability to black-box GNNs.

## Related Work & Insights

- **vs GNNExplainer**: GNNExplainer generates subgraph explanations via mask optimization but lacks natural language; Gspell provides both subgraph and natural language explanations.
- **vs Pan et al. (2024)**: Uses external explainers to generate pseudo-labels for LLM fine-tuning, introducing external explainer bias; Gspell reasons directly from GNN embeddings.
- **vs He et al. (2024b)**: Generates counterfactual explanations but via an autoencoder intermediary; Gspell bridges directly via a projector.
- **vs Multi-modal Alignment**: Analogous to visual-language alignment in CLIP/LLaVA, Gspell achieves GNN-language alignment.

## Rating

- Novelty: ⭐⭐⭐⭐ Applying multi-modal alignment concepts to GNN explainability is novel, though the projector design is relatively standard.
- Experimental Thoroughness: ⭐⭐⭐ Evaluated on real TAG datasets, but lacks validation on more datasets and large-scale GNNs.
- Writing Quality: ⭐⭐⭐⭐ Problem definition is clear, and framework design motivation is well-justified.
- Value: ⭐⭐⭐⭐ Provides a new direction for GNN explainability with high practical utility (plug-and-play).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] LogicXGNN: Grounded Logical Rules for Explaining Graph Neural Networks](../../ICLR2026/graph_learning/logicxgnn_grounded_logical_rules_for_explaining_graph_neural_networks.md)
- [\[ACL 2026\] AgentGL: Towards Agentic Graph Learning with LLMs via Reinforcement Learning](agentgl_towards_agentic_graph_learning_with_llms_via_reinforcement_learning.md)
- [\[ACL 2026\] Graph-Based Alternatives to LLMs for Human Simulation](graph-based_alternatives_to_llms_for_human_simulation.md)
- [\[ACL 2026\] Evaluating LLMs on Large-Scale Graph Property Estimation via Random Walks](evaluating_llms_on_large-scale_graph_property_estimation_via_random_walks.md)
- [\[ACL 2026\] LLMs Underperform Graph-Based Parsers on Supervised Relation Extraction for Complex Graphs](llms_underperform_graph-based_parsers_on_supervised_relation_extraction_for_comp.md)

</div>

<!-- RELATED:END -->

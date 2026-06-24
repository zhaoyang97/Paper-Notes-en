---
title: >-
  [Paper Note] From Nodes to Narratives: Explaining Graph Neural Networks with LLMs and Graph Context
description: >-
  [ACL 2026][Graph Learning][GNN Explainability] Ours proposes Gspell, a lightweight post-hoc explanation framework that projects GNN node embeddings into the LLM embedding space and constructs hybrid prompts (soft prompts + text). This enables LLMs to directly reason over GNN internal representations to generate natural language explanations and explanatory subgraphs, achieving a favorable balance between faithfulness and interpretability on Text-Attributed Graphs (TAGs).
tags:
  - "ACL 2026"
  - "Graph Learning"
  - "GNN Explainability"
  - "LLM Explainer"
  - "Soft Prompt"
  - "Text-attributed Graphs (TAGs)"
  - "Natural Language Explanation"
date: 2026-05-08
content_hash: d1bf91efabc7c849
---

# From Nodes to Narratives: Explaining Graph Neural Networks with LLMs and Graph Context

**Conference**: ACL 2026  
**arXiv**: [2508.07117](https://arxiv.org/abs/2508.07117)  
**Code**: None  
**Area**: Graph Learning / Explainability  
**Keywords**: GNN Explainability, LLM Explainer, Soft Prompt, Text-attributed Graphs (TAGs), Natural Language Explanation

## TL;DR

Ours proposes Gspell, a lightweight post-hoc explanation framework that projects GNN node embeddings into the LLM embedding space and constructs hybrid prompts (soft prompts + text). This enables LLMs to directly reason over GNN internal representations to generate natural language explanations and explanatory subgraphs, achieving a favorable balance between faithfulness and interpretability on Text-Attributed Graphs (TAGs).

## Background & Motivation

**Background**: GNNs are widely applied in high-stakes fields such as healthcare, drug design, and recommendation systems, where the trustworthiness of predictions is critical. Existing GNN explanation methods (e.g., GNNExplainer, PGExplainer) primarily output subgraph masks or feature importance scores, which perform poorly and lack human-readability on Text-Attributed Graphs (TAGs). Simultaneously, the integration of LLMs with GNNs has mostly focused on enhancing GNN task performance rather than explaining GNN predictions.

**Limitations of Prior Work**: (1) Existing LLM-GNN explanation frameworks rely on rigid templates to align GNN explainer outputs with LLM inputs, requiring manual scoring or additional training; (2) Current methods do not directly utilize GNN internal representations, leading to explanations that are either overly generalized or unfaithful to the actual GNN mechanics; (3) Invoking external GNN explainers may bias LLM reasoning—if the explainer is noisy, it misleads the LLM's judgment.

**Key Challenge**: There is a fundamental misalignment between the GNN embedding space and the LLM token space—how can an LLM "see" and understand GNN internal representations rather than relying on second-hand information from external explainers?

**Goal**: Design a post-hoc explanation framework that directly injects GNN internal representations into the LLM without external explainers, generating faithful and interpretable natural language explanations.

**Key Insight**: Analogous to multimodal alignment (e.g., CLIP aligning image and text embeddings), GNN embeddings can be projected as soft prompt tokens for the LLM. This allows the LLM to leverage learned structural information from the GNN while utilizing its own linguistic reasoning capabilities.

**Core Idea**: Train a projector to map GNN node embeddings to the LLM token space and construct hybrid prompts interleaving soft prompts and text. This allows the LLM to generate natural language explanations directly from GNN representations and extract explanatory subgraphs.

## Method

### Overall Architecture

Gspell is a plug-and-play post-hoc explanation framework designed to resolve the misalignment between GNN embedding spaces and LLM token spaces. The mechanism involves training a projector to map frozen GNN node embeddings into the LLM token space. These projected soft prompts are then interleaved with node text to form hybrid prompts for a frozen LLM. This enables the LLM to directly interpret GNN internal representations, generate natural language explanations, and determine whether each node in the computation tree supports or opposes the target prediction to extract an explanatory subgraph. Only the projection layer requires training; both the GNN and LLM remain frozen.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Frozen GNN Node Embeddings f_Φ(v)<br/>+ Node Text Descriptions"] --> B["GNN-LLM Embedding Projector<br/>Π: Single Embedding → k Soft Tokens"]
    B -->|Context Alignment Loss (Semantic)<br/>+ Contrastive Loss (Structural)| C["Hybrid Prompt Construction<br/>Unfolded Computational Tree, Interleaved Soft Prompts & Text"]
    C --> D["Frozen LLM Direct Reasoning<br/>Read GNN Internal Repr. to Generate Explanation"]
    D --> E["Subgraph Extraction & Hallucination Mitigation<br/>Node Tri-classification (+1 / −1 / 0) + Post-verification"]
    E --> F["Output: Natural Language Explanation<br/>+ Explanatory Subgraph S⁺"]
```

### Key Designs

**1. GNN-LLM Embedding Projector: Translating GNN Representations into "Native Tokens"**

GNN embeddings and LLM tokens occupy entirely different spaces. Directly inserting numerical embeddings into prompts causes semantic mismatch. Gspell uses a projector $\Pi:\mathbb{R}^m \to \mathbb{R}^{k\times h}$ to expand a single GNN embedding $f_\Phi(v)$ into $k$ soft prompt tokens. Training employs two losses: the context alignment loss ensures the average representation of the $k$ soft tokens aligns with the LLM embedding of the node text (semantics), while the contrastive loss uses KL divergence to ensure the similarity structure between GNN embeddings is preserved after projection (structure).

**2. Hybrid Prompt Construction: Letting the LLM "See" Structural Representations and Text**

Soft prompts alone are insufficient; the LLM requires context within the correct graph structure. Gspell unfolds the GNN computation tree $\mathcal{T}^{\phi}_v$ for target node $v$ (a message-passing tree with depth equal to GNN layers $L$). It interleaves each node's soft prompt embedding with its text description in a sequence: System Prompt → Target Node (Soft Prompt + Text) → Computation Tree Nodes (Respective Soft Prompt + Text) → Query Instruction. This allows the LLM to reason based on the GNN's actual receptive field rather than guessing from text alone.

**3. Explanatory Subgraph Extraction & Hallucination Mitigation**

To bridge the gap between free-text and structured subgraphs, Gspell tasks the LLM with assigning a tri-classification label to each node in the computation tree: Support (+1), Oppose (-1), or Neutral (0). The set of supporting nodes $S^+_v$ forms the explanatory subgraph. To mitigate hallucinations, Gspell anchors reasoning to GNN embeddings and performs post-processing verification to ensure referenced nodes exist within the computation tree.

### Loss & Training

The projector's training loss is a weighted combination: $\mathcal{L} = \beta \mathcal{L}_{context} + (1-\beta) \mathcal{L}_{contrast}$. The GNN and LLM are frozen throughout; only the projection layer is trained, making it applicable to deployed models without fine-tuning.

## Key Experimental Results

### Main Results

**Node Classification Explanation Quality (Cora Dataset)**

| Method | Fidelity+ ↑ | Fidelity- ↓ | Sparsity ↑ | Insightfulness ↑ |
|------|------------|------------|-----------|-----------------|
| GNNExplainer | 0.12 | 0.08 | 0.65 | — |
| PGExplainer | 0.15 | 0.10 | 0.70 | — |
| GraphLLM | 0.18 | 0.12 | 0.55 | 2.8 |
| **Ours (Gspell)** | **0.22** | **0.06** | **0.72** | **3.5** |

### Ablation Study

| Configuration | Fidelity+ | Sparsity | Insightfulness |
|------|-----------|----------|----------------|
| No Soft Prompt (Text-only) | 0.14 | 0.68 | 2.9 |
| No Contrastive Loss | 0.18 | 0.70 | 3.2 |
| No Context Alignment | 0.16 | 0.69 | 3.0 |
| **Full Gspell** | **0.22** | **0.72** | **3.5** |

### Key Findings

- Inclusion of soft prompts significantly improves faithfulness (+0.08 Fidelity+), proving GNN internal representations provide information unattainable from text alone.
- The dual-loss design is complementary: context alignment ensures semantic consistency, while contrastive loss maintains structural information.
- Ours leads significantly in "insightfulness" (human-evaluated), proving natural language explanations are more understandable than subgraph masks.
- Plug-and-play characteristics allow application to pre-deployed models without retraining GNNs or LLMs.

## Highlights & Insights

- The approach of bypassing traditional GNN explainers to let LLMs directly interpret GNN internal representations is elegant, reducing information loss and bias.
- The contrastive loss in the projector design is clever—it requires not just individual alignment but consistency in relative relationships across spaces.
- The interleaved hybrid prompt design allows the LLM to synthesize numerical structural representations and natural language descriptions for multi-perspective reasoning.

## Limitations & Future Work

- Validated only on node classification; graph-level classification or link prediction remains unexplored.
- Projector quality depends on the separability of GNN embeddings; performance may degrade if the GNN embedding space is poorly structured.
- Explanatory subgraph extraction depends on LLM output parsing, which may be affected by the LLM's instruction-following capabilities.
- Constructing the computation tree requires knowledge of the GNN's layer count, limiting applicability to completely black-box GNNs.

## Related Work & Insights

- **vs GNNExplainer**: GNNExplainer generates subgraphs via mask optimization but lacks natural language; Ours provides both.
- **vs Pan et al. (2024)**: Previous work fine-tuned LLMs using pseudo-labels from external explainers, introducing bias; Ours reasons directly from embeddings.
- **vs He et al. (2024b)**: Used autoencoders for counterfactual explanations; Ours uses a direct bridge via a projector.
- **vs Multimodal Alignment**: Ours applies CLIP/LLaVA-style vision-language alignment concepts to GNN-language alignment.

## Rating

- Novelty: ⭐⭐⭐⭐ Applying multimodal alignment to GNN explainability is novel, though projector design is standard.
- Experimental Thoroughness: ⭐⭐⭐ Evaluated on real TAG datasets, but needs more datasets and large-scale GNN validation.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition and strong design motivation.
- Value: ⭐⭐⭐⭐ Provides a new direction for GNN explainability with high practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] LogicXGNN: Grounded Logical Rules for Explaining Graph Neural Networks](../../ICLR2026/graph_learning/logicxgnn_grounded_logical_rules_for_explaining_graph_neural_networks.md)
- [\[ACL 2026\] Graph-Based Alternatives to LLMs for Human Simulation](graph-based_alternatives_to_llms_for_human_simulation.md)
- [\[ACL 2026\] AgentGL: Towards Agentic Graph Learning with LLMs via Reinforcement Learning](agentgl_towards_agentic_graph_learning_with_llms_via_reinforcement_learning.md)
- [\[ACL 2025\] GraphNarrator: Generating Textual Explanations for Graph Neural Networks](../../ACL2025/graph_learning/graphnarrator.md)
- [\[ACL 2026\] Evaluating LLMs on Large-Scale Graph Property Estimation via Random Walks](evaluating_llms_on_large-scale_graph_property_estimation_via_random_walks.md)

</div>

<!-- RELATED:END -->

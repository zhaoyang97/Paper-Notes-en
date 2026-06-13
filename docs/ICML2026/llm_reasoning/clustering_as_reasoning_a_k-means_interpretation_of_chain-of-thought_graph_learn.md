---
title: >-
  [Paper Note] Clustering as Reasoning: A $k$-Means Interpretation of Chain-of-Thought Graph Learning
description: >-
  [ICML 2026][LLM Reasoning][Chain-of-Thought] This paper reveals the mathematical equivalence between Transformer self-attention and $k$-means clustering. Based on this…
tags:
  - "ICML 2026"
  - "LLM Reasoning"
  - "Chain-of-Thought"
  - "Graph Learning"
  - "$k$-means Clustering"
  - "Text-Attributed Graphs"
  - "Semantic-Structural Alignment"
date: 2026-05-08
content_hash: 7fee6038c6e86158
---

# Clustering as Reasoning: A $k$-Means Interpretation of Chain-of-Thought Graph Learning

**Conference**: ICML 2026  
**arXiv**: [2605.24867](https://arxiv.org/abs/2605.24867)  
**Code**: https://github.com/Uncnbb/KCoT  
**Area**: LLM Reasoning  
**Keywords**: Chain-of-Thought, Graph Learning, $k$-means Clustering, Text-Attributed Graphs, Semantic-Structural Alignment  

## TL;DR
This paper reveals the mathematical equivalence between Transformer self-attention and $k$-means clustering. Based on this, the KCoT framework is designed to explicitly decompose CoT reasoning into a two-step "assignment-update" semantic filtering prompt. It uses Condition-Net to dynamically fuse topological priors with evolving thought representations, consistently surpassing SOTA in node classification and link prediction.

## Background & Motivation

**Background**: Chain-of-Thought (CoT) prompting has been widely used to enhance the reasoning capabilities of LLMs on Text-Attributed Graphs (TAGs). Existing methods include translating graph topology into natural language prompts (HetGCoT), simulating reasoning steps in latent space (GCoT), fine-tuning LLMs with explicit reasoning trajectories (GraphInstruct), and multi-agent toolchain extensions for industrial-scale graphs (GraphChain).

**Limitations of Prior Work**: Existing graph CoT paradigms suffer from two fundamental flaws. First, **loose architectural coupling**—LLMs and GNNs are divided into independent processing stages. The LLM acts only as a semantic parser/generator, while semantic reasoning and structural propagation remain isolated, failing to achieve step-by-step semantic-topological interaction. Second, **insufficient interpretability**—existing CoT operates as a "black box," lacking geometric interpretability regarding how natural language reasoning drives node representation optimization or how generated "thoughts" correspond to clear mathematical optimization targets in graph learning.

**Key Challenge**: There is a semantic–structural misalignment between the structural neighborhoods relied upon by GNN message passing and the representation similarity used in LLM semantic reasoning. Without explicit alignment, message propagation aggregates semantically inconsistent neighbors, causing representation blurring and category confusion.

**Goal**: (1) Provide a theoretically grounded geometric interpretation for CoT reasoning; (2) Design a unified framework for step-by-step semantic-topological interaction.

**Key Insight**: The authors start from a critical theoretical discovery—there exists a parameterization of the Transformer self-attention layer that makes it functionally equivalent to the assignment-update steps of $k$-means. This implies that CoT reasoning is essentially iterative clustering, where each step of thought updates a semantic centroid.

**Core Idea**: Reconstruct CoT prompt design using the $k$-means assignment-update framework, allowing the LLM to function as a semantic filter (assignment) and a semantic centroid refiner (update). Simultaneously, inject topological priors into the evolving reasoning state through Condition-Net.

## Method

### Overall Architecture
The input to KCoT is a Text-Attributed Graph $\mathcal{G}=(\mathcal{V},\mathcal{E},\mathcal{X})$, and the output is the prediction for node classification or link prediction. The overall pipeline consists of three stages: (1) obtaining initial node representations using a pre-trained graph encoder; (2) simulating $k$-means assignment-update steps via Semantic Discriminating Prompts to generate structured thought text; (3) transforming thought embeddings into reasoning matrices using Condition-Net to modulate node features for the next iteration. After $M$ iterations, the final representations are used for downstream tasks.

### Key Designs

1. **Semantic Discriminating Prompt**:

    - **Function**: Explicitly encode the assignment and update steps of $k$-means as CoT reasoning steps, letting the LLM act as a semantic filter rather than an indiscriminate neighbor aggregator.
    - **Mechanism**: **Assignment Step**—Prompt the LLM to "identify shared aspects" and "discard low-similarity nodes," replacing Euclidean distance $\|x_i - \mu_j\|^2$ with semantic distance to filter out neighbors inconsistent with the target node. **Update Step**—Prompt the LLM to perform abstract summarization of filtered neighbors, "stating derived insights in a concise and dense paragraph," compressing semantic variance into an updated "semantic centroid." The output is $\mathcal{T}_i \leftarrow \operatorname{Prompt}(\mathbf{T}_i, \mathbf{N}_i)$.
    - **Design Motivation**: Traditional $k$-means distance metrics are unsuitable for TAGs, as "distance" between texts is subjective and context-dependent. LLMs have demonstrated superior semantic centroid extraction capabilities compared to traditional $k$-means; thus, LLM-driven discriminative reasoning replaces rigid mathematical distances.

2. **Structure-grounded Thought Construction**:

    - **Function**: Fuse fixed topological priors with dynamically evolving reasoning states while capturing thought representations from both structural and semantic neighborhoods.
    - **Mechanism**: In each iteration $t$, two types of neighbors are obtained: (a) Structural neighbors $\mathcal{N}_i^{\text{str}}$: $K$ nodes randomly sampled from 1-hop and 2-hop neighbors to preserve explicit geometric priors; (b) Reasoning-induced neighbors $\mathcal{N}_i^{(t)}$: $K$ semantic nearest neighbors obtained via KNN search on the current node representation $\mathbf{H}^{(t)}$. Both types are encoded by BERT into $T^{\text{str}}$ and $T^{(t)}$ via semantic discriminating prompts and concatenated into the reasoning state $z^{(t)} = [T^{\text{str}} \| T^{(t)}]$.
    - **Design Motivation**: Relying solely on fixed graph edges is insufficient for sparsely or noisily connected nodes, while relying solely on KNN semantic neighbors loses topological constraints. The dual-channel design ensures semantic reasoning is always constrained by structural priors.

3. **Condition-Net**:

    - **Function**: Translate linguistic semantics into vector embeddings compatible with the graph representation space to generate reasoning matrices that modulate node features.
    - **Mechanism**: Using the reasoning state $z^{(t)}$ as input, a lightweight MLP generates a reasoning matrix $\mathbf{P}^{(t)} = \text{CondNet}(z^{(t)}; \phi)$. The original node features are modulated via an element-wise product $\mathbf{X}_{t+1} = \mathbf{P}^{(t)} \odot \mathbf{X}$, serving as input for the next iteration of the graph encoder. The output of the final round is the "answer matrix" used for prediction.
    - **Design Motivation**: It acts as a hypernetwork to balance the trade-off between fixed topological connections and evolving thoughts, bridging the modality gap between linguistic semantics and graph representation spaces.

### Loss & Training
The pre-training phase uses a contrastive learning framework (with link prediction as a pretext task), while downstream fine-tuning uses Cross-Entropy loss (node classification) or Binary Cross-Entropy loss (link prediction). The graph encoder parameters are frozen during reasoning iterations, and only the Condition-Net parameters $\phi$ are optimized. Thoughts are updated every 100 epochs, with reasoning steps fixed at $t=2$ and neighbor count $K=5$.

## Key Experimental Results

### Main Results (Single Focus Protocol)

| Dataset | Task | KCoT | LLAGA-HO | GraphGPT | GCN | Gain (vs LLAGA-HO) |
|--------|------|------|----------|----------|-----|-------------------|
| Arxiv | Node Class | **79.25** | 76.66 | 75.11 | 73.72 | +2.59 |
| Products | Node Class | **86.39** | 84.67 | 84.15 | 80.75 | +1.72 |
| Cora | Node Class | **90.63** | 89.22 | 88.45 | 88.93 | +1.41 |
| Pubmed | Node Class | **95.87** | 95.03 | 94.23 | 92.96 | +0.84 |
| Cora | Link Pred | **88.45** | 86.82 | 80.19 | 81.59 | +1.63 |
| Products | Link Pred | **96.70** | 95.56 | 94.32 | 93.95 | +1.14 |

All improvements were verified via $t$-test ($p < 0.01$). Comprehensive leadership was also maintained under the Task Expert and Classification Expert protocols.

### Ablation Study

| Configuration | Cora (NC) | Products (NC) | Cora (LP) | Products (LP) | Description |
|------|-----------|---------------|-----------|----------------|------|
| KCoT (Full) | **90.63** | **86.39** | **88.45** | **96.70** | Full model |
| w/o $\mathcal{N}^{\text{str}}$ | 89.84 | 85.12 | 87.68 | 96.03 | Removed structural neighbors, -0.8-1.3% |
| w/o $\mathcal{N}^{(t)}$ | 89.02 | 84.17 | 85.32 | 94.47 | Removed KNN neighbors, -1.6-3.1% |
| w/o Prompt | 87.97 | 82.35 | 83.47 | 92.05 | Removed semantic prompts (largest drop, 2.7-5.0%) |
| w/o CoT ($t=1$) | 89.12 | 82.47 | 82.65 | 94.21 | Single-step reasoning, -1.5-5.8% |

### Key Findings
- **Semantic Discriminating Prompt is the most critical component**: Removing it caused the largest drop across all tasks (e.g., Cora link prediction fell from 88.45% to 83.47%), proving that LLMs require explicit algorithmic guidance rather than just acting as text encoders.
- **Iterative CoT outperforms single-step reasoning**: $t=2$ is the optimal number of reasoning steps; $t>2$ led to performance degradation due to over-smoothing and noise overfitting, consistent with $k$-means behavior where excessive iterations pull towards outliers.
- **LLM backbones are replaceable**: Vicuna-7B, Llama2-7B, and ChatGPT-4.1 nano were all effective. ChatGPT-4.1 nano reached 91.04% on Cora node classification, indicating that stronger backbones can further enhance performance.
- **t-SNE visualization** verified that CoT iterations gradually form clearer category clusters, consistent with the centroid update dynamics of $k$-means.

## Highlights & Insights
- **Transformer-$k$-means equivalence** is the core theoretical contribution: Proving that the self-attention layer has a parameterization that makes it exactly equivalent to the assignment-update steps of soft $k$-means ($\epsilon=0$). This provides the first geometric interpretability framework for CoT, cleverly achieving this without modifying input representations, but through the construction of weights and biases.
- **The Semantic-Structural Misalignment Contraction Theorem** (Theorem 4.4) proves that CoT iterations cause the misalignment metric $\Delta_t$ to converge at a geometric rate ($\Delta_{t+1} \leq \rho \Delta_t + \varepsilon$). This repositions CoT from "smarter reasoning" to an "iterative alignment mechanism," providing a novel theoretical perspective.
- **The dual-channel neighbor design** is transferable to other graph-text multimodal tasks: structural sampling preserves topological constraints while KNN search captures semantic dynamics. This fusion of "fixed priors + evolving state" has general value for multimodal alignment.

## Limitations & Future Work
- The number of reasoning steps $t$ is limited by GNN over-smoothing; performance drops when $t>2$, which restricts deeper reasoning chains.
- Generating text with an LLM and encoding it with BERT in each round involves a time complexity of $|\mathcal{V}| \cdot C_{\text{LLM}}$, which is expensive for large-scale graphs (e.g., Products has 2.45 million nodes).
- Experiments only covered citation networks and e-commerce, lacking validation in heterogeneous graph scenarios like social networks or knowledge graphs.
- The element-wise product modulation ($\mathbf{P}^{(t)} \odot \mathbf{X}$) in Condition-Net may be less expressive than attention mechanisms; more flexible feature modulation methods could be explored.
- Adaptive reasoning step strategies could be designed by considering the over-squashing problem in graphs, allowing different nodes to decide reasoning depth based on local structural complexity.

## Related Work & Insights
- **LLAGA** (Chen et al., 2024a): A projector-based graph-LLM alignment scheme; KCoT's baselines and experimental settings are inherited from this work.
- **GraphGPT** (Tang et al., 2024): Uses CoT to align text and structural data but lacks interpretability; KCoT's theoretical framework fills this gap.
- **GCoT** (Yu et al., 2025b): Simulates reasoning steps in latent space but only for non-textual graphs; KCoT operates directly on TAGs.
- **Inspiration**: Generalize the $k$-means interpretability framework to other Transformer applications (e.g., visual token selection in VLMs) to design more efficient token pruning strategies from a clustering perspective.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Verifying Chain-of-Thought Reasoning via Its Computational Graph](../../ICLR2026/llm_reasoning/verifying_chain-of-thought_reasoning_via_its_computational_graph.md)
- [\[ICLR 2026\] DAG-Math: Graph-of-Thought Guided Mathematical Reasoning in LLMs](../../ICLR2026/llm_reasoning/dag-math_graph-of-thought_guided_mathematical_reasoning_in_llms.md)
- [\[ICML 2026\] Hidden Error Awareness in Chain-of-Thought Reasoning: The Signal Is Diagnostic, Not Causal](hidden_error_awareness_in_chain-of-thought_reasoning_the_signal_is_diagnostic_no.md)
- [\[ACL 2026\] Learning to Edit Knowledge via Instruction-based Chain-of-Thought Prompting](../../ACL2026/llm_reasoning/learning_to_edit_knowledge_via_instruction-based_chain-of-thought_prompting.md)
- [\[ICML 2026\] The Expressive Power of Low Precision Softmax Transformers with (Summarized) Chain-of-Thought](the_expressive_power_of_low_precision_softmax_transformers_with_summarized_chain.md)

</div>

<!-- RELATED:END -->

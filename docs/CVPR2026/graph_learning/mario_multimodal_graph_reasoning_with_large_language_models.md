---
title: >-
  [Paper Note] Mario: Multimodal Graph Reasoning with Large Language Models
description: >-
  [CVPR 2026][Graph Learning][Multimodal graphs] Mario is proposed for LLM reasoning on multimodal graphs (MMGs). It achieves topology-aware cross-modal alignment via a Graph-conditioned Vision-Language Model (GVLM)…
tags:
  - "CVPR 2026"
  - "Graph Learning"
  - "Multimodal graphs"
  - "LLM reasoning"
  - "vision-language alignment"
  - "modality-adaptive routing"
  - "instruction tuning"
date: 2026-05-08
content_hash: 897514b43ad7586e
---

# Mario: Multimodal Graph Reasoning with Large Language Models

**Conference**: CVPR 2026
**arXiv**: [2603.05181](https://arxiv.org/abs/2603.05181)  
**Code**: Coming soon  
**Area**: Graph Learning
**Keywords**: Multimodal graphs, LLM reasoning, vision-language alignment, modality-adaptive routing, instruction tuning

## TL;DR

Mario is proposed for LLM reasoning on multimodal graphs (MMGs). It achieves topology-aware cross-modal alignment via a Graph-conditioned Vision-Language Model (GVLM), and employs a Modality-Adaptive Prompt Router (MAPR) to select the optimal modality configuration for each node, attaining state-of-the-art performance on node classification and link prediction.

## Background & Motivation

Existing multimodal LLMs process independent image-text pairs, overlooking the relational structure among multimodal data in real-world settings. In MMGs, each node carries both text and image attributes, while edges provide structural priors. Directly encoding with a VLM (e.g., CLIP) and feeding the result into a graph model introduces two challenges:

**C1 Weak cross-modal consistency**: A node's text and image are not necessarily semantically aligned; neighboring information can resolve ambiguity but is ignored. The cross-modal cosine similarity of frozen CLIP is low, yet improves by 68% upon incorporating graph topology.

**C2 Heterogeneous modality preference**: The informativeness of different modalities varies across nodes. Approximately 30% of nodes can only be correctly classified under a specific modality configuration. A one-size-fits-all prompt template wastes information.

### Core Problem

> Can a unified framework be designed that simultaneously addresses cross-modal inconsistency and heterogeneous modality preference on MMGs during LLM reasoning?

## Method

### Overall Architecture

**Stage 1 (GVLM)**: Dual-tower encoder + topology-aware multimodal mixer → graph-conditioned contrastive learning → structure-aware cross-modal consistent representations.
**Stage 2 (Modality-Adaptive Instruction Tuning)**: Construction of text-only, image-only, and multimodal prompt templates → MAPR router selects the optimal template → LLM reasoning.

### Key Designs

1. **Topology-Aware Multimodal Mixer**: At each encoder layer, CLS representations are collected from all nodes across the graph, and neighbor information is aggregated via multi-head attention with graph-structure positional bias. The structure-aware CLS is then re-injected into the token sequence to replace the original CLS. Iterating layer by layer enables deep fusion of structural and modal information.

2. **Graph-Conditioned Contrastive Learning**: Bidirectional InfoNCE is applied to structure-aware text/image CLS embeddings:
    $\mathcal{L}_{\text{S1}} = -\frac{1}{|\mathcal{B}|}\sum_v [\log\frac{e^{s(v,v)/\tau}}{\sum_u e^{s(v,u)/\tau}} + \log\frac{e^{s(v,v)/\tau}}{\sum_u e^{s(u,v)/\tau}}]$

3. **Modality-Adaptive Prompt Router (MAPR)**:

    - Three prompts are constructed per node: $\mathcal{S}_v^{\text{txt}}$ (text tokens only), $\mathcal{S}_v^{\text{vis}}$ (image tokens only), $\mathcal{S}_v^{\text{mm}}$ (bimodal tokens)
    - Router input: $[\mathbf{h}_v^{\text{text}}; \mathbf{h}_v^{\text{image}}; \phi^{(1)}(v); \phi^{(2)}(v); \log d_v]$
    - An MLP outputs three-class routing probabilities $\mathbf{p}_v = \text{softmax}(\mathbf{s}_v)$
    - Performance posteriors $\mathbf{q}_v = \text{softmax}(-[\ell_v^{(\text{txt})}, \ell_v^{(\text{vis})}, \ell_v^{(\text{mm})}])$ serve as teacher signals
    - Loss: $\mathcal{L}_{\text{S2}} = \frac{1}{|B|}\sum_v [\sum_k q_v^{(k)} \ell_v^{(k)} + \lambda \text{KL}(\mathbf{q}_v \| \mathbf{p}_v)]$

### Loss & Training

Stage 1 trains the encoder with contrastive loss. Stage 2 fine-tunes the LLM and router using performance-weighted LM loss with KL regularization. At inference, the router selects the optimal modality template.

## Key Experimental Results

### Main Results (Node Classification Accuracy %)

| Method | Movies | Reddit | CDs | Arts |
|--------|--------|--------|-----|------|
| GCN(text) | 43.8 | 84.3 | 51.4 | 76.9 |
| GATv2(text) | 48.7 | 85.6 | 54.7 | 80.4 |
| **Mario** | **53.6+** | **95.3+** | **63.4+** | **92.1+** |

### Ablation Study

| Configuration | Performance | Note |
|---------------|-------------|------|
| No graph-conditioned VLM (frozen CLIP) | Low consistency | Cross-modal misalignment |
| Node-level fine-tuning (no topology) | Partial improvement | Missing neighbor information |
| +GVLM (Stage 1) | Significant gain | Topology- and modality-aware |
| +MAPR (Stage 2) | **Best** | Modality-adaptive selection |

### Mix-Training Setting (Node Classification Accuracy %)

| Method | Modality | Movies | Reddit | CDs | Arts |
|--------|----------|--------|--------|-----|------|
| SAGE | Text | 46.85 | 89.96 | 53.24 | 87.46 |
| LLaGA | Text | 47.80 | 91.14 | 51.33 | 74.02 |
| LLaGA-A | Text+Image | 50.61 | 92.94 | 56.29 | 88.83 |
| Graph4MM | Text+Image | 51.07 | 92.89 | 55.53 | 89.32 |
| **Mario-8B** | **Text+Image** | **53.63** | **95.30** | **63.43** | **92.13** |

### Key Findings

- Cross-modal consistency improves by 68% upon introducing graph topology (vs. frozen CLIP)
- ~30% of nodes exhibit a clear single-modality preference
- Zero-shot transfer yields up to 1.6× improvement

## Highlights & Insights

- **Precise identification of two challenges**: Weak consistency and heterogeneous preference are genuine bottlenecks in MMG reasoning; the Venn diagram analysis is intuitive and compelling
- **Elegant MAPR routing mechanism**: LLM loss serves as a performance signal to drive routing learning—soft routing during training and hard routing at inference with zero overhead
- **GVLM in Stage 1 as a new paradigm**: A topology-aware vision-language model that alternately performs graph attention and token attention within Transformer layers
- **Strong zero-shot transfer**: Up to 1.6× gain on unseen MMGs, indicating that the learned modality routing strategy generalizes well
- **Unified framework**: The same architecture handles both node classification and link prediction, demonstrating strong generality

## Limitations & Future Work

- Two-stage training increases complexity; Stage 2 requires three LLM forward passes per sample
- Mixer attention complexity is $\mathcal{O}(|\mathcal{V}_s|^2 d)$, necessitating node sampling for large-scale graphs
- The current framework handles only text+image bimodal graphs and has not been extended to modalities such as audio or video
- The graph topology bias $\mathbf{B}_h$ relies on precomputed shortest paths, which is unfriendly to dynamic graphs
- MLaGA fuses features via Q-Former before passing to the LLM; Graph4MM handles missing modalities—Mario performs better in complete-modality settings but has not been evaluated under missing-modality conditions

## Rating

⭐⭐⭐⭐⭐ (5/5)

The dual innovations of GVLM and MAPR, combined with comprehensive experiments spanning four datasets, two tasks, and three modality settings, as well as zero-shot transfer validation of generalization, establish Mario as a significant pioneering contribution to multimodal graph + LLM reasoning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] PathMind: A Retrieve-Prioritize-Reason Framework for Knowledge Graph Reasoning with Large Language Models](../../AAAI2026/graph_learning/pathmind_a_retrieve-prioritize-reason_framework_for_knowledge_graph_reasoning_wi.md)
- [\[NeurIPS 2025\] Deliberation on Priors: Trustworthy Reasoning of Large Language Models on Knowledge Graphs](../../NeurIPS2025/graph_learning/deliberation_on_priors_trustworthy_reasoning_of_large_language_models_on_knowled.md)
- [\[ICML 2026\] KBQA-R1: Reinforcing Large Language Models for Knowledge Base Question Answering](../../ICML2026/graph_learning/kbqa-r1_reinforcing_large_language_models_for_knowledge_base_question_answering.md)
- [\[CVPR 2026\] M3KG-RAG: Multi-hop Multimodal Knowledge Graph-enhanced Retrieval-Augmented Generation](m3kg_rag_multi_hop_multimodal_knowledge_graph_enhanced_retrieval_augmented_genera.md)
- [\[ICML 2026\] Finding the Minimal Parameter Budget for Implicit Reasoning: A Data Complexity Driven Scaling Law for Language Models](../../ICML2026/graph_learning/finding_the_minimal_parameter_budget_for_implicit_reasoning_a_data_complexity_dr.md)

</div>

<!-- RELATED:END -->

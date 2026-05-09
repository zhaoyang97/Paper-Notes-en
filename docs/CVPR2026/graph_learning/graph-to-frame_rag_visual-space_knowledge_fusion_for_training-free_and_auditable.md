---
title: >-
  [Paper Note] Graph-to-Frame RAG: Visual-Space Knowledge Fusion for Training-Free and Auditable Video Reasoning
description: >-
  [CVPR 2026][Graph Learning][Video retrieval-augmented generation] This paper proposes the G2F-RAG paradigm, which renders retrieved structured knowledge into a single "reasoning frame" appended to the end of the video, enabling large models to reason uniformly within the visual space. This approach avoids the attention dilution and cognitive overload caused by text appending, achieving consistent training-free improvements across 8 video benchmarks.
tags:
  - CVPR 2026
  - Graph Learning
  - Video retrieval-augmented generation
  - knowledge graph
  - visual-space fusion
  - multi-agent framework
  - training-free video reasoning
date: 2026-05-08
content_hash: 4c880bc0f4e060eb
---

# Graph-to-Frame RAG: Visual-Space Knowledge Fusion for Training-Free and Auditable Video Reasoning

**Conference**: CVPR 2026
**arXiv**: [2604.04372](https://arxiv.org/abs/2604.04372)
**Code**: None
**Area**: Multimodal VLM / Graph Learning
**Keywords**: Video retrieval-augmented generation, knowledge graph, visual-space fusion, multi-agent framework, training-free video reasoning

## TL;DR
This paper proposes the G2F-RAG paradigm, which renders retrieved structured knowledge into a single "reasoning frame" appended to the end of the video, enabling large models to reason uniformly within the visual space. This approach avoids the attention dilution and cognitive overload caused by text appending, achieving consistent training-free improvements across 8 video benchmarks.

## Background & Motivation

**Background**: Large multimodal models (LMMs) have made significant progress in video understanding, but complex video reasoning still faces three major challenges: (1) multi-step compositional reasoning (cross-shot causality, navigation, etc.); (2) reliance on external knowledge such as commonsense and object affordances; and (3) the need for smaller models to reliably solve problems without additional training while providing an auditable evidence chain.

**Limitations of Prior Work**: Mainstream video RAG methods adopt a "retrieve-and-append" paradigm: appending text (ASR/OCR/descriptions), retrieving candidate clips, or injecting structured graphs/event chains as text. These methods share an implicit assumption—more relevant content plus longer context equals better reasoning. In practice, performance degrades even for short videos: heterogeneous information sources compete in the same attention space, where continuous low-level visual signals compete with discrete high-level text, causing attention dilution and increased cognitive load.

**Key Challenge**: The core issue lies not only in "what to retrieve" but in "how to represent and fuse external knowledge." When semantic alignment is poor and load is uncontrolled, retrieval can actually impair model capability. Experiments confirm this: Video-RAG scores 5.4 points below baseline on MLVU, while G2F-RAG exceeds it by 4.6 points.

**Goal**: How to integrate external knowledge into video models in a modality-aligned manner, avoiding cross-modal competition and context explosion. Sub-problems include: (1) offline construction of reusable video knowledge graphs; (2) online determination of whether external knowledge is needed; and (3) retrieval of minimal sufficient subgraphs and their rendering as visual frames.

**Key Insight**: Video models aggregate and reason most effectively within the visual space. External knowledge should enter this space using visual grammar. Research demonstrates that the visual modality can serve as an efficient compression medium for textual information. Accordingly, retrieved structured knowledge is converted into visual tokens, allowing the model to operate in its most familiar spatiotemporal reasoning domain.

**Core Idea**: Retrieved knowledge subgraphs are rendered as single-frame reasoning frames appended to the end of the video, achieving knowledge fusion within the visual space and avoiding cross-modal attention competition.

## Method

### Overall Architecture
The framework consists of offline and online phases. **Offline phase**: A graph construction agent analyzes the video and generates a complete, question-agnostic knowledge graph $\mathcal{G}$ (covering entities, events, spatial relations, and external knowledge), built once and reused multiple times. **Online phase**: An orchestration agent routes by difficulty (simple questions answered directly; hard questions routed through the RAG path) → a retrieval agent extracts the minimal sufficient subgraph $S^\star$ → a rendering agent converts it into a single-frame reasoning frame $I_{\text{RF}}$ → the frame is appended to the end of the video $\tilde{V}=[V; I_{\text{RF}}]$ → the LMM performs joint reasoning. The backbone remains frozen throughout.

### Key Designs

1. **Video Knowledge Graph Construction (Offline)**:

    - **Function**: Generate a question-agnostic, reusable complete knowledge graph for each video.
    - **Mechanism**: Two complementary views are unified—an event-causal view (participants, actions, intentions, pre/post-conditions, causal chains) and a scene-functional view (objects and their affordances, functional regions and connectivity, abstract conceptual knowledge). The two views are bound via dense cross-links, enabling seamless transitions between "what happened" and "where/with what it happened." External web tools can optionally be connected to supplement world knowledge.
    - **Design Motivation**: The question-agnostic design means graph construction is performed only once (offline caching), with a single graph reused across multiple questions. The dual-view design covers both causal and spatial requirements common in video reasoning.

2. **Hierarchical Routing and Minimal Subgraph Retrieval (Online)**:

    - **Function**: Avoid unnecessary knowledge injection for simple questions while precisely supplying required knowledge for complex ones.
    - **Mechanism**: The orchestration agent outputs a difficulty judgment $d(q,V,\mathcal{G}) \in \{\text{easy}, \text{hard}\}$, comparing a proxy utility estimate $\Delta U = \hat{U}_{\text{G2F}} - \hat{U}_{\text{Base}}$ against a threshold $\tau$. For hard questions, the retrieval agent selects a compact subgraph $S^\star = \arg\max_{S \subseteq \mathcal{G}} [R(q,S) - \lambda C(S)]$, with explicit constraints on node count $|\mathcal{V}(S^\star)| \leq N_{\max}$ and edge count $|\mathcal{E}(S^\star)| \leq E_{\max}$ to control the visual token budget.
    - **Design Motivation**: Disabling routing (applying RAG to all questions) causes VideoMME to drop from 70.6% to 66.8%, indicating that simple questions do not benefit from knowledge injection. Minimal subgraph retrieval prevents information overload (Full-Loose subgraph selection causes a slight performance decrease).

3. **Reasoning Frame Rendering and Visual-Space Fusion**:

    - **Function**: Transform abstract graph structures into visual tokens that LMMs can efficiently consume.
    - **Mechanism**: The rendering agent uses Graphviz to convert the subgraph $S^\star$ into a single-frame reasoning frame $I_{\text{RF}}$, employing a concise visual grammar (icons + short labels) to depict key entities, relations, and causal flows. The frame is appended at the end of the video to avoid disrupting the original content, while temporal attention can still cover it. Instructions specify that the video content is authoritative and the reasoning frame is auxiliary. No timestamps are encoded; focus is on structure and mechanism.
    - **Design Motivation**: Ablation experiments show that End-1 is optimal; Mid insertion disrupts temporal aggregation (MLVU 73.4→67.9); multi-frame injection increases the token budget and reduces precision (End-4 drops to 69.0). The Minimal style is optimal; Text-Heavy reintroduces contextual load.

### Loss & Training
No training is required. The entire pipeline relies on a frozen backbone combined with prompt design. Routing decisions depend on prompting agents to perform task decomposition and strategy selection. Offline graph construction uses GPT-4o; routing and subgraph extraction use GPT-4o-mini.

## Key Experimental Results

### Main Results (Cross-Model, Cross-Task)

| Model | Base VideoMME | +G2F-RAG | Base WildVideo | +G2F-RAG | Base MLVU | +G2F-RAG |
|-------|--------------|----------|----------------|----------|-----------|----------|
| InternVL3.5-4B | 65.4 | 70.1 (+4.7) | 45.2 | 47.1 (+1.9) | - | - |
| LLaVA-Video-7B | 63.7 | 64.5 (+0.8) | 53.4 | 57.0 (+3.6) | 69.5 | 75.5 |
| Qwen2.5-VL-7B | 65.1 | 70.6 (+5.5) | 51.3 | 55.4 (+4.1) | 68.8 | 73.4 |
| InternVL3.5-8B | 66.0 | 72.0 (+6.0) | 53.0 | 60.1 (+7.1) | - | - |

### Comparison with Other RAG Methods (Qwen2.5-VL-7B)

| Method | MLVU | WildVideo | VideoMME |
|--------|------|-----------|---------|
| Baseline | 68.8 | 51.3 | 65.1 |
| +Video-RAG | 63.4 (−5.4) | 47.2 (−4.1) | 60.5 (−4.6) |
| +Vgent | 72.1 | 50.1 | 68.9 |
| **+G2F-RAG** | **73.4 (+4.6)** | **55.4 (+4.1)** | **70.6 (+5.5)** |

### Ablation Study (Qwen2.5-VL-7B)

| Ablation Dimension | Variant | MLVU | VideoMME |
|--------------------|---------|------|---------|
| Representation | G2J-RAG (text JSON) | 66.2 | 63.0 |
| | **G2F-RAG (visual frame)** | **73.4** | **70.6** |
| Frame Position | Mid-1 | 67.9 | 64.0 |
| | End-4 | 69.0 | 66.0 |
| | **End-1** | **73.4** | **70.6** |
| Routing | Off (RAG for all) | 69.9 | 66.8 |
| | **On + Fallback** | **73.4** | **70.6** |

### Key Findings
- **Visual frame fusion vs. text JSON**: Given the same subgraph delivered in different formats, G2F-RAG outperforms G2J-RAG by 7.6 points on VideoMME, demonstrating that "how to fuse" matters more than "what to fuse."
- Video-RAG (text appending) consistently degrades performance across all benchmarks (MLVU −5.4, WildVideo −4.1, VideoMME −4.6), indicating that heterogeneous information fusion itself is a source of failure.
- Smaller models benefit more (4B/7B gains of 3–7 points), as visual-space fusion reduces cross-modal competition orthogonally to model capacity.
- Removing intent and affordance fields causes MLVU to drop from 73.4 to 70.2, demonstrating that these graph fields capture useful precondition information.
- Performance remains nearly unchanged when deliberately incorrect or adversarial reasoning frames are provided, because the prompt consistently instructs the model to treat the original video as authoritative.

## Highlights & Insights
- "The manner of knowledge delivery matters more than the content of knowledge" is a profound insight—the same retrieved results yield 7.6 more points in visual frame format than in text JSON. This challenges the implicit assumption in the RAG community that retrieval quality is the sole determinant of performance.
- The training-free design allows plug-and-play integration with any LMM backbone (InternVL, LLaVA-Video, Qwen-VL), with consistent gains across different scales. This architecture-level approach is more transferable than fine-tuning.
- The minimalist design of the single-frame reasoning frame counterintuitively outperforms multi-frame injection—compressing information to the minimal necessary amount proves most effective.

## Limitations & Future Work
- Offline graph construction relies on GPT-4o, incurring high costs and introducing a closed-source model dependency.
- The accuracy of routing judgments affects final performance (misclassification causes simple questions to be harmed by RAG or hard questions to be answered incorrectly without it); the current prompt-based routing lacks robustness guarantees.
- Graphviz rendering of the reasoning frame may suffer from reduced readability for complex subgraphs.
- The method has not been validated on very long videos (>1 hour), where the scale of the knowledge graph and retrieval precision may become bottlenecks.
- External tools (GPT-4o-mini routing) increase inference latency.

## Related Work & Insights
- **vs. Video-RAG**: Video-RAG appends text retrieval results and consistently degrades performance; G2F-RAG consistently improves performance through visual-space fusion. The fundamental difference lies in the delivery modality.
- **vs. Vgent**: Vgent mitigates overload via structured retrieval and verification but still appends text; G2F-RAG further converts structured results into visual representations—achieving 57.0 vs. 51.6 on WildVideo, respectively.
- **vs. Traditional KG-RAG**: Conventional KG-RAG injects graph knowledge as text; this work is the first to visualize the knowledge graph as video frames, leveraging the model's visual processing capabilities.
- Attention analysis empirically validates the method: text RAG disperses attention across retrieved context and non-critical frames, whereas G2F-RAG concentrates attention on key segments and the reasoning frame.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First to propose fusing retrieved knowledge as visual frames for video reasoning—a paradigm-level innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Eight benchmarks, multiple backbones, and comprehensive ablations (representation, position, style, routing, graph design).
- **Writing Quality**: ⭐⭐⭐⭐⭐ Attention analysis precisely reveals the root cause of the problem; ablation design is meticulous.
- **Value**: ⭐⭐⭐⭐⭐ Introduces an entirely new RAG paradigm with broad implications for video understanding and multimodal reasoning.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] M3KG-RAG: Multi-hop Multimodal Knowledge Graph-enhanced Retrieval-Augmented Generation](m3kg_rag_multi_hop_multimodal_knowledge_graph_enhanced_retrieval_augmented_genera.md)
- [\[NeurIPS 2025\] DuetGraph: Coarse-to-Fine Knowledge Graph Reasoning with Dual-Pathway Global-Local Fusion](../../NeurIPS2025/graph_learning/duetgraph_coarse-to-fine_knowledge_graph_reasoning_with_dual-pathway_global-loca.md)
- [\[AAAI 2026\] Human Cognition Inspired RAG with Knowledge Graph for Complex Problem Solving](../../AAAI2026/graph_learning/human_cognition_inspired_rag_with_knowledge_graph_for_complex_problem_solving.md)
- [\[CVPR 2026\] Mario: Multimodal Graph Reasoning with Large Language Models](mario_multimodal_graph_reasoning_with_large_language_models.md)
- [\[CVPR 2026\] Graph2Eval: Automatic Multimodal Task Generation for Agents via Knowledge Graphs](graph2eval_multimodal_task_generation_agents.md)

<!-- RELATED:END -->

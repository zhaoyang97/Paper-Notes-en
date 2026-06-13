---
title: >-
  [Paper Note] STK-Adapter: Incorporating Evolving Graph and Event Chain for Temporal Knowledge Graph Extrapolation
description: >-
  [ACL 2026][Time Series][Temporal Knowledge Graph Extrapolation] This paper proposes STK-Adapter, which embeds three MoE modules (ST-MoE to capture spatio-temporal structures, EA-MoE to model event chain semantics…
tags:
  - "ACL 2026"
  - "Time Series"
  - "Temporal Knowledge Graph Extrapolation"
  - "MoE Adapter"
  - "Cross-Modal Alignment"
  - "Event Chain Modeling"
  - "Graph Structural Evolution"
date: 2026-05-08
content_hash: 0e6259c6201c5d35
---

# STK-Adapter: Incorporating Evolving Graph and Event Chain for Temporal Knowledge Graph Extrapolation

**Conference**: ACL 2026  
**arXiv**: [2604.19042](https://arxiv.org/abs/2604.19042)  
**Code**: [GitHub](https://github.com/Zhaoshuyuan0246/STK-Adapter)  
**Area**: Time Series  
**Keywords**: Temporal Knowledge Graph Extrapolation, MoE Adapter, Cross-Modal Alignment, Event Chain Modeling, Graph Structural Evolution

## TL;DR

This paper proposes STK-Adapter, which embeds three MoE modules (ST-MoE to capture spatio-temporal structures, EA-MoE to model event chain semantics, and CMA-MoE for deep cross-modal alignment) into each layer of an LLM. This approach addresses the issues of spatio-temporal information loss and layer-wise dilution caused by the shallow alignment of TKG embeddings with LLMs in existing methods, significantly outperforming SOTA on four benchmark datasets.

## Background & Motivation

**Background**: Temporal Knowledge Graph (TKG) extrapolation aims to predict future events based on historical events. Early methods (e.g., REGCN, TiRGN) modeled spatio-temporal dependencies within snapshot sequences via Graph Neural Networks, but embedding events into latent spaces loses textual semantics. With the rise of LLMs, methods like CoH linearize TKGs into textual event chains for instruction tuning, but the linearization process loses the topological structure of the TKG.

**Limitations of Prior Work**: (1) Shallow alignment problem—methods like GenTKGQA and TGL-LLM use simple MLPs to project TKG evolution structural representations into the LLM semantic space in one go, failing to fully preserve spatio-temporal information; (2) Layer-wise dilution problem—LLMs are inherently optimized for next-token prediction, and hidden states bias toward textual semantics during fine-tuning, causing TKG structural features to gradually decay across layers.

**Key Challenge**: Structural representations of TKGs and textual semantics of LLMs reside in different modal spaces, requiring deep, layer-wise alignment for the LLM to truly understand graph structures. However, existing methods only perform "one-time projection" (shallow alignment), leading the LLM to gradually "forget" graph structural information during deep processing.

**Goal**: Design a flexible adapter that establishes dedicated processing channels in each LLM layer to capture and inject TKG evolution structural representations and event chain semantic dependencies layer by layer.

**Key Insight**: Leveraging the advantages of MoE in processing heterogeneous data—achieving sparse computation through expert specialization while significantly enhancing model capacity. Introducing MoE into the PEFT framework by replacing single adapters with a low-rank expert pool.

**Core Idea**: Embed three independent MoE modules in each LLM layer, responsible for TKG spatio-temporal feature extraction (ST-MoE), event chain semantic dependency modeling (EA-MoE), and deep cross-modal alignment (CMA-MoE). Outputs are integrated via adaptive fusion to achieve progressive multi-modal alignment.

## Method

### Overall Architecture

STK-Adapter is integrated into each layer of an LLM (e.g., Llama3-8B). The input consists of two parts: (1) TKG evolution structural representations $\text{H}_g^0 = [\text{H}_s^{(t)}; \text{H}_r^{(t)}]$ encoded by a pre-trained graph encoder (e.g., LogCL); (2) event chain text retrieved by temporal logic rules, which yields hidden text representations after LLM tokenization. The three MoE modules process these in parallel at each layer and output to the next layer through adaptive fusion.

### Key Designs

1.  **Spatial-Temporal MoE (ST-MoE)**:
    *   **Function**: Continuously captures the spatial structure and temporal patterns of the TKG at each layer to mitigate information dilution during deep propagation.
    *   **Mechanism**: A sparse activation router $f_{\text{ST\_router}}$ calculates routing weights based on the previous layer's TKG representation $\text{H}_g^{l-1}$ and activates Top-k experts. Each expert uses a bottleneck structure (down-projection → non-linear → up-projection) to model spatio-temporal patterns in specialized low-dimensional subspaces. The output is aggregated via weighted routing: $\overline{\text{H}}_g^l = \sum_{i \in \mathcal{A}^l} \text{gate}_i^l \cdot \text{E}^{(i)}(\text{H}_g^{l-1})$.
    *   **Design Motivation**: Iteratively capture TKG evolution structures layer by layer, re-injecting spatio-temporal information at each stage to fundamentally solve the problem of information dilution in deep LLM layers.

2.  **Event-Aware MoE (EA-MoE)**:
    *   **Function**: Guides the LLM to learn complex temporal semantic dependencies within event chains.
    *   **Mechanism**: Similar in architecture to ST-MoE but employs an event-aware router—all tokens under the same timestamp share the same routing signal. Specifically, for the $j$-th token, the routing signal is derived from the hidden state of its corresponding temporal token $\tau(j)$: $\hat{\text{gate}}_j^l = f_{\text{EA\_router}}(\hat{\text{H}}_{\tau(j)}^l)$, ensuring consistent processing of tokens within the same event.
    *   **Design Motivation**: While ST-MoE captures spatio-temporal features, it is insufficient for characterizing complex temporal semantic dependencies (e.g., event repetition patterns, temporal order) in event chains. The routing strategy anchored by temporal tokens allows experts to specialize in different temporal semantic patterns.

3.  **Cross-Modality Alignment MoE (CMA-MoE)**:
    *   **Function**: Injects TKG spatio-temporal features into the textual event chain representation at each layer to achieve deep progressive cross-modal alignment.
    *   **Mechanism**: Each expert implements a TKG-guided attention module—event chain representation $\hat{\text{H}}_{\text{text}}^l$ serves as the Query, while TKG representation $\text{H}_g^{l-1}$ serves as Key and Value. TKG evolution structural knowledge is injected into the text representation via $\text{Softmax}(\frac{QK^\top}{\sqrt{d_k}})V$. The router is driven by the TKG representation, prioritizing alignment strategies from a spatio-temporal perspective.
    *   **Design Motivation**: Traditional shallow projection (one-time MLP mapping) cannot bridge the modality gap. CMA-MoE performs cross-modal attention at every layer, progressively refining event chain representations to achieve deep fusion between the LLM semantic space and the TKG structural space.

### Loss & Training

The total loss consists of cross-entropy and load balancing loss: $\mathcal{L} = -\sum_{i=1}^{|Y|} \log P(y_i | y_{<i}, \mathcal{C}, \mathcal{G}) + \alpha \sum_{j=1}^{n} f_j \cdot p_j$. The graph encoder is frozen after pre-training, the LLM backbone is frozen, and only the STK-Adapter is fine-tuned (for 2 epochs). Inference uses beam search (B=20) with a hybrid scoring strategy, combining LLM decoding scores and topology-aware scores.

## Key Experimental Results

### Main Results

| Model | ICE14 Hit@1 | ICE18 Hit@1 | ICE15 Hit@1 | WIKI Hit@1 |
| :--- | :--- | :--- | :--- | :--- |
| LogCL | 37.76 | 24.53 | 46.07 | 70.85 |
| LLM-DA (LogCL) | 37.71 | 22.83 | 40.90 | 79.10 |
| MESH | 35.22 | 23.61 | 38.62 | 75.03 |
| **STK-Adapter (LogCL)** | **41.16** | **25.95** | **48.82** | **82.43** |

### Ablation Study

| Configuration | ICE14 Hit@1 | WIKI Hit@1 | Description |
| :--- | :--- | :--- | :--- |
| STK-Adapter | 37.26 | 82.14 | Full model (REGCN encoder) |
| w/o EA-MoE | 35.56 | 74.36 | Removed event-aware MoE, largest drop |
| w/o ST-MoE | 37.03 | 80.11 | Removed spatio-temporal MoE |
| w/o CMA-MoE | 36.86 | 80.08 | Removed cross-modal alignment MoE |
| w/o ST- & CMA-MoE | 33.12 | 78.37 | Removed entire TKG branch |
| w LoRA | 32.47 | 72.10 | Replaced STK-Adapter with LoRA |

### Key Findings

*   EA-MoE contributes the most—removing it caused WIKI Hit@1 to plummet from 82.14% to 74.36% (-7.78pp), indicating that event chain semantic modeling is more critical than graph structure.
*   Replacing STK-Adapter with LoRA results in a significant performance decline (-4.79pp ICE14, -10.04pp WIKI), proving that specialized adapter designs are superior to general PEFT.
*   STK-Adapter is compatible with different graph encoders (REGCN, TiRGN, CognTKE, LogCL) and consistently outperforms corresponding LLM-DA baselines.
*   It maintains consistent performance advantages across different LLM backbones (Llama3-8B, Qwen2.5-7B, Mistral-7B).

## Highlights & Insights

*   The idea of layer-wise TKG information injection fundamentally solves the "information dilution" problem—instead of hoping the LLM doesn't forget after a one-time projection, structural information is forcibly refreshed at each layer. This design is transferable to any scenario requiring the continuous injection of external structured knowledge into LLMs.
*   The event-aware router design is ingenious—achieving shared routing for tokens of the same event through temporal token anchoring ensures processing consistency while implicitly modeling temporal structure.
*   The encoder-agnostic design makes STK-Adapter a plug-and-play solution for integrating any pre-trained graph encoder, reducing engineering implementation costs.

## Limitations & Future Work

*   Evaluation was limited to four relatively small TKG datasets, lacking validation on large-scale TKGs.
*   The use of 4 experts and Top-1 routing was fixed; different numbers of experts or routing strategies were not explored.
*   Freezing the pre-trained graph encoder may limit the upper bound of STK-Adapter—joint end-to-end training might yield further improvements.

## Related Work & Insights

*   **vs GenTKGQA/TGL-LLM**: Shallow projection (MLP) aligns TKG with LLM; STK-Adapter achieves deep alignment through layer-wise MoE.
*   **vs CoH**: Pure text linearization loses topological structure; STK-Adapter preserves and continuously injects graph structures.
*   **vs LLM-DA**: Merely fuses TKG scores during inference; STK-Adapter integrates them deeply during training.

## Rating

*   Novelty: ⭐⭐⭐⭐ The three-MoE layer-wise injection framework is novel, though the combination of MoE + adapters has precedents.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ 4 datasets + multi-encoder compatibility + multi-LLM compatibility + detailed ablation.
*   Writing Quality: ⭐⭐⭐⭐ Clear structure but many formulas; the method section is somewhat lengthy.
*   Value: ⭐⭐⭐⭐ Provides a clear advancement in the TKG+LLM integration field; the encoder-agnostic design is highly practical.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Simple and Efficient Heterogeneous Temporal Graph Neural Network](../../NeurIPS2025/time_series/simple_and_efficient_heterogeneous_temporal_graph_neural_network.md)
- [\[ICML 2026\] Beyond Extrapolation: Knowledge Utilization Paradigm with Bidirectional Inspiration for Time Series Forecasting](../../ICML2026/time_series/beyond_extrapolation_knowledge_utilization_paradigm_with_bidirectional_inspirati.md)
- [\[NeurIPS 2025\] Graph-based Neural Space Weather Forecasting](../../NeurIPS2025/time_series/graph-based_neural_space_weather_forecasting.md)
- [\[ICLR 2026\] Routing Channel-Patch Dependencies in Time Series Forecasting with Graph Spectral Decomposition](../../ICLR2026/time_series/routing_channel-patch_dependencies_in_time_series_forecasting_with_graph_spectra.md)
- [\[AAAI 2026\] Urban Incident Prediction with Graph Neural Networks: Integrating Government Ratings and Crowdsourced Reports](../../AAAI2026/time_series/urban_incident_prediction_with_graph_neural_networks_integrating_government_rati.md)

</div>

<!-- RELATED:END -->

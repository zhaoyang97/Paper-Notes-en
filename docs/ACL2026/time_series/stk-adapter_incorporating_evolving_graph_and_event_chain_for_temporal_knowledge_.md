---
title: >-
  [Paper Note] STK-Adapter: Incorporating Evolving Graph and Event Chain for Temporal Knowledge Graph Extrapolation
description: >-
  [ACL 2026][Time Series][Temporal Knowledge Graph Extrapolation] This paper proposes STK-Adapter, which embeds three MoE modules in each layer of a Large Language Model (LLM)—ST-MoE for capturing spatio-temporal structures, EA-MoE for modeling event chain semantics, and CMA-MoE for deep cross-modal alignment. It addresses the issues of spatio-temporal information loss and layer-wise dilution caused by shallow alignment between TKG embeddings and LLMs…
tags:
  - "ACL 2026"
  - "Time Series"
  - "Temporal Knowledge Graph Extrapolation"
  - "MoE Adapter"
  - "Cross-modal Alignment"
  - "Event Chain Modeling"
  - "Graph Structure Evolution"
date: 2026-05-08
content_hash: e6ccd0ec0640caa7
---

# STK-Adapter: Incorporating Evolving Graph and Event Chain for Temporal Knowledge Graph Extrapolation

**Conference**: ACL 2026  
**arXiv**: [2604.19042](https://arxiv.org/abs/2604.19042)  
**Code**: [GitHub](https://github.com/Zhaoshuyuan0246/STK-Adapter)  
**Area**: Time Series  
**Keywords**: Temporal Knowledge Graph Extrapolation, MoE Adapter, Cross-modal Alignment, Event Chain Modeling, Graph Structure Evolution

## TL;DR

This paper proposes STK-Adapter, which embeds three MoE modules in each layer of a Large Language Model (LLM)—ST-MoE for capturing spatio-temporal structures, EA-MoE for modeling event chain semantics, and CMA-MoE for deep cross-modal alignment. It addresses the issues of spatio-temporal information loss and layer-wise dilution caused by shallow alignment between TKG embeddings and LLMs, significantly outperforming SOTA on four benchmark datasets.

## Background & Motivation

**Background**: Temporal Knowledge Graph (TKG) extrapolation aims to predict future events based on historical ones. Early methods (e.g., REGCN, TiRGN) model spatio-temporal dependencies in snapshot sequences via Graph Neural Networks but lose textual semantics by embedding events into latent spaces. With the rise of LLMs, methods like CoH linearize TKGs into textual event chains for instruction tuning, but this process loses the topological structure of the TKG.

**Limitations of Prior Work**: (1) Shallow alignment—methods like GenTKGQA and TGL-LLM use simple MLPs to project TKG evolutionary structure representations into the LLM semantic space in one shot, failing to fully preserve spatio-temporal information; (2) Layer-wise dilution—since LLMs are optimized for next-token prediction, hidden states bias toward textual semantics during fine-tuning, causing TKG structural features to decay across layers.

**Key Challenge**: Structural representations of TKGs and textual semantics of LLMs reside in different modal spaces, requiring deep, layer-by-layer alignment for the LLM to truly understand graph structures. Existing "one-time projection" (shallow alignment) causes the LLM to gradually "forget" graph information during deep processing.

**Goal**: Design a flexible adapter to establish dedicated processing channels in each LLM layer, capturing and injecting TKG evolutionary structure representations and event chain semantic dependencies progressively.

**Key Insight**: Leverage the advantages of Mixture-of-Experts (MoE) in handling heterogeneous data—achieving sparse computation through expert specialization while significantly enhancing model capacity. Introduce MoE into the PEFT framework by replacing a single adapter with a low-rank expert pool.

**Core Idea**: Embed three independent MoE modules in each LLM layer to handle TKG spatio-temporal feature extraction (ST-MoE), event chain semantic dependency modeling (EA-MoE), and cross-modal deep alignment (CMA-MoE). Integrate outputs through adaptive fusion to achieve progressive multi-modal alignment.

## Method

### Overall Architecture

STK-Adapter is integrated into every layer of an LLM (e.g., Llama3-8B). The input consists of two parts: (1) TKG evolutionary structure representations $\text{H}_g^0 = [\text{H}_s^{(t)}; \text{H}_r^{(t)}]$ encoded by a pretrained graph encoder (e.g., LogCL); (2) Event chain text retrieved by temporal logic rules, which yields textual hidden representations after LLM tokenization. The three MoE modules process these in parallel at each layer and pass the fused output to the next layer.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}}%%
flowchart TD
    A["TKG Evolutionary Structure<br/>Pretrained Encoder H_g"] --> L
    B["Event Chain Text<br/>Logic Retrieval → Tokenization"] --> L
    subgraph L["Each LLM Layer (Repeated Layer-wise)"]
        direction TB
        C["ST-MoE: Spatio-Temporal MoE<br/>Resists dilution via layer-wise re-injection"]
        D["EA-MoE: Event-Aware MoE<br/>Time-anchored routing for event tokens"]
        E["CMA-MoE: Cross-Modality Alignment MoE<br/>Progressive injection via cross-attention"]
        C --> F["Adaptive Fusion"]
        D --> F
        E --> F
    end
    F --> G["Next Layer → … → Prediction Output<br/>Beam Search + Hybrid Scoring"]
```

### Key Designs

**1. Spatial-Temporal MoE (ST-MoE): Resisting deep dilution via layer-wise re-injection**

Shallow alignment fails because TKG structures are only projected at the first layer, becoming diluted by textual semantics in deeper layers. ST-MoE transforms "injection" from a one-time event into a layer-wise iteration. A sparse router $f_{\text{ST\_router}}$ calculates weights based on the previous layer's TKG representation $\text{H}_g^{l-1}$ to activate Top-k experts. Each expert uses a bottleneck structure (down-projection → non-linearity → up-projection) to model patterns in low-dimensional subspaces. Outputs are aggregated as $\overline{\text{H}}_g^l = \sum_{i \in \mathcal{A}^l} \text{gate}_i^l \cdot \text{E}^{(i)}(\text{H}_g^{l-1})$. By re-processing information at every layer, structural features no longer decay with depth.

**2. Event-Aware MoE (EA-MoE): Time-anchored routing for grouped token processing**

While ST-MoE captures graph-level structures, it lacks the fine-grained temporal semantics of event chains. EA-MoE reuses the expert architecture but modifies the router: all tokens under the same timestamp share a routing signal. For the $j$-th token, the signal is derived from the hidden state of its corresponding time token $\tau(j)$, i.e., $\hat{\text{gate}}_j^l = f_{\text{EA\_router}}(\hat{\text{H}}_{\tau(j)}^l)$. This ensures tokens belonging to the same event are handled by the same experts, allowing experts to specialize in specific temporal semantic patterns. Ablation shows that removing EA-MoE causes the largest performance drop, proving it is the core of the framework.

**3. Cross-Modality Alignment MoE (CMA-MoE): Progressive graph knowledge injection**

Traditional MLPs cannot bridge the modal gap between structure and text. CMA-MoE performs TKG-guided cross-modal attention in every layer: each expert takes the event chain representation $\hat{\text{H}}_{\text{text}}^l$ as the Query and the TKG representation $\text{H}_g^{l-1}$ as Key and Value. Using $\text{Softmax}(\frac{QK^\top}{\sqrt{d_k}})V$, structural knowledge is injected into textual representations. The router is also driven by TKG representations to prioritize alignment strategies. This progressive refinement merges the LLM semantic space with the TKG structural space.

### Loss & Training

The total loss combines cross-entropy and load balancing loss: $\mathcal{L} = -\sum_{i=1}^{|Y|} \log P(y_i | y_{<i}, \mathcal{C}, \mathcal{G}) + \alpha \sum_{j=1}^{n} f_j \cdot p_j$. The pretrained graph encoder and LLM backbone are frozen, and only STK-Adapter is fine-tuned (for 2 epochs). Inference utilizes beam search ($B=20$) with a hybrid scoring strategy that fuses LLM decoding scores and topology-aware scores.

## Key Experimental Results

### Main Results

| Model | ICE14 Hit@1 | ICE18 Hit@1 | ICE15 Hit@1 | WIKI Hit@1 |
|------|-----------|-----------|-----------|----------|
| LogCL | 37.76 | 24.53 | 46.07 | 70.85 |
| LLM-DA (LogCL) | 37.71 | 22.83 | 40.90 | 79.10 |
| MESH | 35.22 | 23.61 | 38.62 | 75.03 |
| **STK-Adapter (LogCL)** | **41.16** | **25.95** | **48.82** | **82.43** |

### Ablation Study

| Configuration | ICE14 Hit@1 | WIKI Hit@1 | Description |
|------|-----------|----------|------|
| STK-Adapter | 37.26 | 82.14 | Full model (REGCN encoder) |
| w/o EA-MoE | 35.56 | 74.36 | Removed Event-Aware MoE (Largest drop) |
| w/o ST-MoE | 37.03 | 80.11 | Removed Spatio-Temporal MoE |
| w/o CMA-MoE | 36.86 | 80.08 | Removed Cross-Modal Alignment MoE |
| w/o ST- & CMA-MoE | 33.12 | 78.37 | Removed entire TKG branch |
| w LoRA | 32.47 | 72.10 | Replaced STK-Adapter with LoRA |

### Key Findings

- EA-MoE provides the most significant contribution—removing it causes WIKI Hit@1 to plummet from 82.14% to 74.36% (-7.78pp), indicating that event chain semantic modeling is more critical than graph structure alone.
- Replacing STK-Adapter with LoRA results in a substantial performance decline (-4.79pp on ICE14, -10.04pp on WIKI), proving that specialized adapter designs outperform general PEFT.
- STK-Adapter is compatible with various graph encoders (REGCN, TiRGN, CognTKE, LogCL) and consistently outperforms corresponding LLM-DA baselines.
- Consistent performance advantages are maintained across different LLM backbones (Llama3-8B, Qwen2.5-7B, Mistral-7B).

## Highlights & Insights

- The approach of layer-wise injection of TKG information fundamentally solves the "information dilution" problem—it forces a refresh of structural information at every layer rather than relying on a one-time projection. This design is transferable to any scenario requiring continuous injection of external structured knowledge into LLMs.
- The event-aware router design is ingenious—achieving shared routing for tokens within the same event via time-token anchoring ensures consistency and implicitly models temporal structures.
- The encoder-agnostic design allows STK-Adapter to serve as a plug-and-play module for any pretrained graph encoder, reducing engineering implementation costs.

## Limitations & Future Work

- Evaluation was limited to four relatively small TKG datasets; performance on large-scale TKGs remains to be verified.
- The model uses a fixed set of 4 experts with Top-1 routing; larger expert counts or different routing strategies have not been explored.
- Keeping the pretrained graph encoder frozen may limit the upper bound of STK-Adapter; end-to-end joint training might yield further improvements.

## Related Work & Insights

- **vs GenTKGQA/TGL-LLM**: These use shallow projections (MLPs) to align TKGs with LLMs, while STK-Adapter achieves deep alignment through layer-wise MoEs.
- **vs CoH**: Pure text linearization loses topological structures, which STK-Adapter preserves and continuously injects.
- **vs LLM-DA**: Merges TKG scores only during inference, whereas STK-Adapter performs deep integration during training.

## Rating

- Novelty: ⭐⭐⭐⭐ The three-MoE layer-wise injection framework is novel, though MoE+adapter combinations have precedents.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 4 datasets + multi-encoder compatibility + multi-LLM compatibility + detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear structure but heavy on formulas; the method section is somewhat lengthy.
- Value: ⭐⭐⭐⭐ Significant advancement in TKG+LLM integration with a highly practical encoder-agnostic design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] ANRE: Analogical Replay for Temporal Knowledge Graph Forecasting](../../ACL2025/time_series/anre_analogical_replay_for_temporal_knowledge_graph_forecasting.md)
- [\[ACL 2025\] G2S: A General-to-Specific Learning Framework for Temporal Knowledge Graph Forecasting with Large Language Models](../../ACL2025/time_series/g2s_a_general-to-specific_learning_framework_for_temporal_knowledge_graph_foreca.md)
- [\[ICML 2026\] Beyond Extrapolation: Knowledge Utilization Paradigm with Bidirectional Inspiration for Time Series Forecasting](../../ICML2026/time_series/beyond_extrapolation_knowledge_utilization_paradigm_with_bidirectional_inspirati.md)
- [\[ICLR 2026\] ASTGI: Adaptive Spatio-Temporal Graph Interactions for Irregular Multivariate Time Series Forecasting](../../ICLR2026/time_series/astgi_adaptive_spatio-temporal_graph_interactions_for_irregular_multivariate_tim.md)
- [\[NeurIPS 2025\] Simple and Efficient Heterogeneous Temporal Graph Neural Network](../../NeurIPS2025/time_series/simple_and_efficient_heterogeneous_temporal_graph_neural_network.md)

</div>

<!-- RELATED:END -->

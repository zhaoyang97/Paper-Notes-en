---
title: >-
  [Paper Note] STK-Adapter: Incorporating Evolving Graph and Event Chain for Temporal Knowledge Graph Extrapolation
description: >-
  [ACL 2026][Time Series][Temporal Knowledge Graph Extrapolation] This paper proposes STK-Adapter, which embeds three MoE modules at every LLM layer—ST-MoE for capturing spatiotemporal structure, EA-MoE for modeling event chain semantics, and CMA-MoE for deep cross-modal alignment—addressing the spatiotemporal information loss and layer-wise dilution caused by shallow alignment of TKG embeddings with LLMs in existing methods, achieving significant improvements over SOTA on four benchmark datasets.
tags:
  - ACL 2026
  - Time Series
  - Temporal Knowledge Graph Extrapolation
  - MoE Adapter
  - Cross-Modal Alignment
  - Event Chain Modeling
  - Graph Structure Evolution
date: 2026-05-08
content_hash: d4973db13257d350
---
# STK-Adapter: Incorporating Evolving Graph and Event Chain for Temporal Knowledge Graph Extrapolation

**Conference**: ACL 2026  
**arXiv**: [2604.19042](https://arxiv.org/abs/2604.19042)  
**Code**: [GitHub](https://github.com/Zhaoshuyuan0246/STK-Adapter)  
**Area**: Time Series  
**Keywords**: Temporal knowledge graph extrapolation, MoE adapter, cross-modal alignment, event chain modeling, graph structure evolution

## TL;DR

This paper proposes STK-Adapter, which embeds three MoE modules at every layer of an LLM—ST-MoE for capturing spatiotemporal structure, EA-MoE for modeling event chain semantics, and CMA-MoE for deep cross-modal alignment—to address the spatiotemporal information loss and layer-wise dilution caused by shallow alignment between TKG embeddings and LLMs in existing methods, achieving significant improvements over SOTA on four benchmark datasets.

## Background & Motivation

**State of the Field**: Temporal knowledge graph (TKG) extrapolation aims to predict future events based on historical ones. Early methods (e.g., REGCN, TiRGN) model spatiotemporal dependencies across snapshot sequences via graph neural networks, but lose textual semantics by encoding events into latent spaces. With the rise of LLMs, methods such as CoH linearize TKGs into textual event chains for instruction tuning, yet the linearization discards the topological structure of TKGs.

**Limitations of Prior Work**: (1) *Shallow alignment*—methods such as GenTKGQA and TGL-LLM project the evolving structural representations of TKGs into the LLM semantic space via a single MLP, failing to adequately preserve spatiotemporal information; (2) *Layer-wise dilution*—LLMs are fundamentally optimized for next-token prediction, causing hidden states to drift toward textual semantics during fine-tuning and the structural features of TKGs to progressively decay across layers.

**Root Cause**: TKG structural representations and LLM textual semantics reside in different modality spaces, requiring deep, layer-wise alignment for LLMs to genuinely understand graph structure. Existing methods perform only a single one-shot projection (shallow alignment), causing LLMs to progressively "forget" graph structural information in deeper layers.

**Paper Goals**: Design a flexible adapter that establishes dedicated processing channels at every LLM layer to progressively capture and inject TKG evolving structural representations and event chain semantic dependencies.

**Starting Point**: Drawing on the advantage of MoE in handling heterogeneous data—enabling sparse computation through expert specialization while substantially increasing model capacity—the paper introduces MoE into the PEFT framework, replacing a single adapter with a pool of low-rank experts.

**Core Idea**: Embed three independent MoE modules at every LLM layer, respectively responsible for TKG spatiotemporal feature extraction (ST-MoE), event chain semantic dependency modeling (EA-MoE), and deep cross-modal alignment (CMA-MoE); their outputs are integrated via adaptive fusion to achieve progressive multi-modal alignment.

## Method

### Overall Architecture

STK-Adapter is integrated into every layer of an LLM (e.g., Llama3-8B). The input consists of two components: (1) TKG evolving structural representations $\text{H}_g^0 = [\text{H}_s^{(t)}; \text{H}_r^{(t)}]$ encoded by a pretrained graph encoder (e.g., LogCL); and (2) event chain text retrieved by temporal logical rules, yielding textual hidden representations after LLM tokenization. The three MoE modules process inputs in parallel at each layer and produce outputs to the next layer via adaptive fusion.

### Key Designs

1. **Spatial-Temporal MoE (ST-MoE)**:

    - **Function**: Continuously captures the spatial structure and temporal patterns of TKGs at every layer, mitigating information dilution during deep propagation.
    - **Mechanism**: A sparse activation router $f_{\text{ST\_router}}$ computes routing weights from the previous-layer TKG representation $\text{H}_g^{l-1}$ and activates the Top-k experts. Each expert adopts a bottleneck structure (down-projection → nonlinearity → up-projection) to model spatiotemporal patterns in a specialized low-dimensional subspace. Outputs are aggregated by routing weights: $\overline{\text{H}}_g^l = \sum_{i \in \mathcal{A}^l} \text{gate}_i^l \cdot \text{E}^{(i)}(\text{H}_g^{l-1})$.
    - **Design Motivation**: By iteratively capturing TKG evolving structure layer by layer and re-injecting spatiotemporal information at each layer, this design fundamentally resolves the problem of structural information being diluted in the deeper layers of LLMs after shallow alignment.

2. **Event-Aware MoE (EA-MoE)**:

    - **Function**: Guides the LLM to learn complex temporal semantic dependencies within event chains.
    - **Mechanism**: The architecture is similar to ST-MoE but employs an event-aware router—all tokens under the same timestamp share a single routing signal. Specifically, for the $j$-th token, the routing signal derives from the hidden state of its corresponding temporal token $\tau(j)$: $\hat{\text{gate}}_j^l = f_{\text{EA\_router}}(\hat{\text{H}}_{\tau(j)}^l)$, ensuring consistent processing of tokens within the same event.
    - **Design Motivation**: Although ST-MoE captures spatial and temporal features, it is insufficient to characterize the complex temporal semantic dependencies in event chains (e.g., event recurrence patterns, temporal ordering). The temporal-token-anchored routing strategy enables experts to specialize in distinct temporal semantic patterns.

3. **Cross-Modality Alignment MoE (CMA-MoE)**:

    - **Function**: Injects TKG spatiotemporal features into textual event chain representations at every layer, achieving deep and progressive cross-modal alignment.
    - **Mechanism**: Each expert implements a TKG-guided attention module—event chain representations $\hat{\text{H}}_{\text{text}}^l$ serve as Query while TKG representations $\text{H}_g^{l-1}$ serve as Key and Value. TKG evolving structural knowledge is injected into textual representations via $\text{Softmax}(\frac{QK^\top}{\sqrt{d_k}})V$. The router is driven by TKG representations, prioritizing the selection of alignment strategies from the spatiotemporal perspective.
    - **Design Motivation**: Conventional shallow projection (one-shot MLP mapping) cannot bridge the modality gap. CMA-MoE performs cross-modal attention at every layer, progressively refining event chain representations and achieving deep fusion between the LLM semantic space and the TKG structural space.

### Loss & Training

The total loss consists of cross-entropy and load-balancing losses: $\mathcal{L} = -\sum_{i=1}^{|Y|} \log P(y_i | y_{<i}, \mathcal{C}, \mathcal{G}) + \alpha \sum_{j=1}^{n} f_j \cdot p_j$. The graph encoder is pretrained and then frozen; the LLM backbone is frozen; only STK-Adapter is fine-tuned (2 epochs). At inference, beam search ($B=20$) combined with a hybrid scoring strategy is adopted, fusing LLM decoding scores with topology-aware scores.

## Key Experimental Results

### Main Results

| Model | ICE14 Hit@1 | ICE18 Hit@1 | ICE15 Hit@1 | WIKI Hit@1 |
|------|-----------|-----------|-----------|----------|
| LogCL | 37.76 | 24.53 | 46.07 | 70.85 |
| LLM-DA (LogCL) | 37.71 | 22.83 | 40.90 | 79.10 |
| MESH | 35.22 | 23.61 | 38.62 | 75.03 |
| **STK-Adapter (LogCL)** | **41.16** | **25.95** | **48.82** | 82.43 |

### Ablation Study

| Configuration | ICE14 Hit@1 | WIKI Hit@1 | Note |
|------|-----------|----------|------|
| STK-Adapter | 37.26 | 82.14 | Full model (REGCN encoder) |
| w/o EA-MoE | 35.56 | 74.36 | Largest drop when removed |
| w/o ST-MoE | 37.03 | 80.11 | Without spatiotemporal MoE |
| w/o CMA-MoE | 36.86 | 80.08 | Without cross-modal alignment MoE |
| w/o ST- & CMA-MoE | 33.12 | 78.37 | Full TKG branch removed |
| w LoRA | 32.47 | 72.10 | STK-Adapter replaced by LoRA |

### Key Findings

- EA-MoE contributes the most—removing it causes WIKI Hit@1 to drop sharply from 82.14% to 74.36% (−7.78 pp), indicating that event chain semantic modeling is more critical than graph structure.
- Replacing STK-Adapter with LoRA leads to a substantial performance drop (−4.79 pp on ICE14, −10.04 pp on WIKI), confirming that the specialized adapter design outperforms generic PEFT.
- STK-Adapter is compatible with diverse graph encoders (REGCN, TiRGN, CognTKE, LogCL) and consistently outperforms the corresponding LLM-DA baselines.
- Consistent performance advantages are maintained across different LLM backbones (Llama3-8B, Qwen2.5-7B, Mistral-7B).

## Highlights & Insights

- The layer-wise TKG injection paradigm fundamentally resolves the "information dilution" problem—rather than performing a one-shot projection and hoping the LLM retains it, structural information is forcibly refreshed at every layer. This design is transferable to any scenario requiring continuous injection of external structured knowledge into LLMs.
- The event-aware router design is elegant—temporal-token anchoring ensures that tokens within the same event share a routing decision, simultaneously guaranteeing processing consistency and implicitly modeling temporal structure.
- The encoder-agnostic design allows STK-Adapter to be plug-and-play compatible with any pretrained graph encoder, reducing engineering overhead.

## Limitations & Future Work

- Evaluation is limited to four relatively small TKG datasets, lacking validation on large-scale TKGs.
- The number of experts is fixed at 4 with Top-1 routing; alternative expert counts and routing strategies remain unexplored.
- Freezing the pretrained graph encoder may impose an upper bound on STK-Adapter's performance—joint end-to-end training could yield further improvements.

## Related Work & Insights

- **vs. GenTKGQA/TGL-LLM**: These methods align TKGs with LLMs via shallow projection (MLP); STK-Adapter achieves deep alignment through layer-wise MoE.
- **vs. CoH**: Pure text linearization discards topological structure; STK-Adapter preserves and continuously injects graph structure.
- **vs. LLM-DA**: TKG scores are integrated only at inference time; STK-Adapter deeply integrates structural information during training.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The three-MoE layer-wise injection framework is novel, though the combination of MoE and adapters has precedents.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Four datasets + multi-encoder compatibility + multi-LLM compatibility + detailed ablation.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, though the method section is somewhat verbose with dense notation.
- **Value**: ⭐⭐⭐⭐ Clear advancement for the TKG+LLM integration area; the encoder-agnostic design offers strong practical utility.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2025\] ANRE: Analogical Replay for Temporal Knowledge Graph Forecasting](../../ACL2025/time_series/anre_analogical_replay_for_temporal_knowledge_graph_forecasting.md)
- [\[ACL 2025\] G2S: A General-to-Specific Learning Framework for Temporal Knowledge Graph Forecasting with Large Language Models](../../ACL2025/time_series/g2s_a_general-to-specific_learning_framework_for_temporal_knowledge_graph_foreca.md)
- [\[NeurIPS 2025\] Simple and Efficient Heterogeneous Temporal Graph Neural Network](../../NeurIPS2025/time_series/simple_and_efficient_heterogeneous_temporal_graph_neural_network.md)
- [\[ICLR 2026\] Routing Channel-Patch Dependencies in Time Series Forecasting with Graph Spectral Decomposition](../../ICLR2026/time_series/routing_channel-patch_dependencies_in_time_series_forecasting_with_graph_spectra.md)
- [\[ACL 2026\] Temporal Leakage in Search-Engine Date-Filtered Web Retrieval: A Retrospective Forecasting Case Study](temporal_leakage_in_search-engine_date-filtered_web_retrieval_a_retrospective_fo.md)

<!-- RELATED:END -->

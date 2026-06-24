---
title: >-
  [Paper Note] Persistent Topological Features in Large Language Models
description: >-
  [ICML 2025][Model Compression][Topological Data Analysis] This work introduces zigzag persistence from topological data analysis (TDA) to analyze the internal representations of LLMs. By tracking the continuous evolution of topological features of prompts across representation spaces of different layers, it identifies four processing phases and proposes a layer pruning criterion based on topological descriptors, achieving performance comparable to SOTA methods.
tags:
  - "ICML 2025"
  - "Model Compression"
  - "Topological Data Analysis"
  - "Zigzag Persistence"
  - "LLM Internal Representations"
  - "Layer Pruning"
  - "Persistent Homology"
date: 2026-05-08
content_hash: c1f60f2c7e188278
---

# Persistent Topological Features in Large Language Models

**Conference**: ICML 2025  
**arXiv**: [2410.11042](https://arxiv.org/abs/2410.11042)  
**Author**: Yuri Gardinazzi, Karthik Viswanathan, Giada Panerai, Alessio Ansuini, Alberto Cazzaniga, Matteo Biagetti (Area Science Park & collaborating institutions)  
**Code**: None released  
**Area**: Model Compression  
**Keywords**: Topological Data Analysis, Zigzag Persistence, LLM Internal Representations, Layer Pruning, Persistent Homology  

## TL;DR

This work introduces zigzag persistence from topological data analysis (TDA) to analyze the internal representations of LLMs. By tracking the continuous evolution of topological features of prompts across representation spaces of different layers, it identifies four processing phases and proposes a layer pruning criterion based on topological descriptors, achieving performance comparable to SOTA methods.

## Background & Motivation

### Background
Large language models exhibit excellent performance on various NLP tasks, but their black-box nature makes interpretability and transparency core concerns. Meanwhile, the massive scale of these models consumes significant computational resources, creating an urgent need to compress models without significant loss of performance. Researchers typically start with internal representations to understand the division of labor across different layers.

### Limitations of Prior Work
- **Limitations of geometric approaches**: Approaches that characterize representation manifolds of each layer using concepts like intrinsic dimension have revealed that semantic knowledge emerges in the middle layers rather than the final layers. However, these methods are essentially **layer-by-layer static analyses** and cannot directly track the dynamic changes of representations during cross-layer propagation.
- **Limitations of traditional TDA**: Standard persistent homology can capture multi-scale topological features (connected components, loops, voids, etc.) of a single point cloud, but it cannot handle **point cloud sequences that dynamically evolve over time (layers)**.
- **Fragmented layer-by-layer analysis**: Existing methods usually evaluate each layer independently and then aggregate the results, which loses cross-layer evolutionary information and fails to understand model operations from a global system perspective.

### Key Challenge
The internal representations of an LLM are inherently a sequence of point clouds dynamically evolving with layers—each prompt corresponds to a point that is transformed layer by layer in the representation space. However, existing analysis tools either only perform static analysis layer by layer, or cannot track the **birth, persistence, and death** trajectories of topological features across layers.

### Key Insight
The authors observe that the representations of different layers in LLMs can naturally be viewed as a sequence of point clouds evolving over time in a discrete dynamical system. Therefore, they introduce **zigzag persistence**, a tool in TDA specifically designed for time-varying data, to track the complete evolutionary paths of these topological features.

### Core Idea
Using zigzag persistence, the internal representations of LLM layers are treated as a sequence of time-varying point clouds to fully track the continuous evolution of $p$-dimensional topological features along the depth of the model, thereby obtaining a system-level dynamic perspective.

## Method

### Overall Architecture

Input: A set of prompts is fed into a pretrained LLM $\rightarrow$ extract hidden representations of each layer (one point cloud per layer) $\rightarrow$ construct cross-layer zigzag simplicial complex sequences $\rightarrow$ compute zigzag persistence $\rightarrow$ extract topological descriptors $\rightarrow$ apply to analysis/pruning.

The entire pipeline is divided into three main stages: (1) representation extraction and preprocessing; (2) k-NN-based zigzag persistence computation; (3) topological descriptor extraction and downstream application.

### Key Designs

1. **Layer Representations as Time-Varying Point Clouds**

    - Function: Treats the output of the $l$-th layer of the LLM as a point cloud $X_l = \{x_l^{(1)}, \ldots, x_l^{(n)}\}$, where $n$ is the number of prompts.
    - Mechanism: Traditional methods compute topological features for each layer independently and then aggregate them. This work treats $X_0, X_1, \ldots, X_L$ as a time series (layer number = time) and leverages zigzag persistence to directly capture the propagation and changes of topological features across layers.
    - Design Motivation: This "layer-as-time" interpretation enables the use of the complete theoretical toolkit of zigzag persistence, rather than manually designed cross-layer tracking methods. This is the first systematic application of this tool to LLM internal representations.

2. **k-Nearest Neighbor Filtration (k-NN Filtration)**

    - Function: Builds a simplicial complex using k-nearest neighbor graphs on the point cloud of each layer, replacing continuous distance threshold filtration with discrete growth of $k$.
    - Mechanism: Given a value of $k$, a k-NN graph $G_k(X_l)$ is constructed, where point $x_i$ is connected to its $k$ nearest neighbors. As $k$ increases, the graph becomes denser, and topological features (death of connected components, birth of loops, etc.) occur sequentially. This provides a multi-scale filtration adaptive to local data density.
    - Design Motivation: Compared to Vietoris-Rips complexes based on distance thresholds, k-NN filtration is more robust to local data density variations and is computationally more efficient. This is one of the technical innovations of this paper in the application of zigzag persistence.

3. **Zigzag Persistence Computation**

    - Function: Establishes zigzag connections between complexes of adjacent layers to compute cross-layer persistent homology.
    - Mechanism: For the layer sequence $l=0, 1, \ldots, L$, construct a zigzag diagram:
    $K_0 \hookrightarrow K_{0,1} \hookleftarrow K_1 \hookrightarrow K_{1,2} \hookleftarrow K_2 \hookrightarrow \cdots$
      where $K_l$ is the complex of the $l$-th layer, and $K_{l,l+1}$ is the "bridge" complex connecting adjacent layers. Zigzag persistence tracks the birth layer $b$ and death layer $d$ of $p$-dimensional homology classes ($H_p$) in this sequence, generating a set of persistent diagram $(b, d)$ pairs.
    - Design Motivation: Traditional persistence can only handle monotonically increasing/decreasing filtration sequences, but topological changes between layers are non-monotonic (complexes can grow or shrink). The "zigzag" structure of zigzag persistence naturally adapts to this non-monotonic evolution.

4. **Topological Descriptors**

    - Function: Extracts interpretable statistics from the persistence diagram of zigzag persistence for quantitative analysis.
    - Mechanism: Defines several descriptors to measure the behavior of topological features, including:
        - **Feature lifetime distribution**: The statistical distribution of $(d - b)$, reflecting the stability of the topological structure.
        - **Birth/death density**: The birth and death rates of features at each layer, indicating which layers are undergoing intense topological reorganization.
        - **Number of persistent features**: The total number of topological features surviving at a certain layer, reflecting the complexity of the representation structure.
    - Design Motivation: Raw persistence diagrams are difficult to compare and analyze. These descriptors compress topological information into interpretable statistics, facilitating comparison across models and datasets.

### Four-Phase Model of Prompt Processing

Through the aforementioned topological descriptors, the authors consistently identify four phases of LLM prompt processing across multiple models and datasets:

| Phase | Layer Range | Topological Feature Behavior | Interpretation |
|------|--------|-------------|------|
| Phase 1: Initial Alignment | Shallow layers (first ~15%) | High birth rate, high death rate, short feature lifetimes | Prompts rapidly rearrange their positional relationships in the representation space |
| Phase 2: Stable Middle Layers | Middle layers (~15%-65%) | Low birth rate, low death rate, many long-lived features | Stable topological relationships are established among prompts; semantic structures are formed |
| Phase 3: Transitional Refinement | Mid-to-deep layers (~65%-85%) | Gradually increasing birth and death rates | The model refines the established semantic relationships |
| Phase 4: Final Realignment | Deep layers (last ~15%) | Intense topological changes re-emerge | Representation structures are adjusted to prepare for the output layer |

This four-phase discovery echoes previous research based on intrinsic dimension (emergence of semantics in the middle layers) but provides a richer dynamic perspective.

### Layer Pruning Criterion

Based on the four-phase discovery, the authors propose a topological layer pruning criterion:

- **Core Idea**: Layers in Phase 2 (the stable middle layers) contribute the most redundant topological structures—topological changes between these layers are minimal, so removing some of these layers has the least impact on the overall functionality of the model.
- **Pruning Strategy**: Compute the change in topological descriptors between adjacent layers, and prioritize removing layers with the smallest changes.
- **Advantage**: This criterion does not require downstream task loss signals or gradient information; it is purely based on the topological structure of representations, making it an unsupervised pruning method.

## Key Experimental Results

### Main Results: Layer Pruning Performance Comparison

| Model | Method | Pruning Ratio | Downstream Task Avg. Accuracy Retention | Characteristics |
|------|------|---------|---------------------|------|
| LLaMA-2-7B | No pruning (baseline) | 0% | 100% | Original model |
| LLaMA-2-7B | ShortGPT (BI value) | 27% | ~92-95% | Layer importance evaluation based on Block Influence |
| LLaMA-2-7B | Ours (Topology) | 27% | ~92-95% | Based on zigzag persistence descriptors |
| LLaMA-2-7B | Random Pruning | 27% | ~75-85% | Randomly removed layers |
| Mistral-7B | ShortGPT | 25% | ~90-94% | SOTA method |
| Mistral-7B | Ours (Topology) | 25% | ~90-94% | Comparable topological method |

The proposed method is comparable in pruning performance to SOTA methods like ShortGPT, but its advantages lie in: (1) not requiring loss signals from calibration data; (2) providing an interpretable understanding of the model's structure.

### Consistency of Topological Descriptors Across Models/Datasets

| Model | Dataset | Phase 1 Layer Range | Phase 2 Layer Range | Phase 3 Layer Range | Phase 4 Layer Range |
|------|-------|---------------|---------------|---------------|---------------|
| LLaMA-2-7B | WikiText | 0-4 | 5-20 | 21-26 | 27-31 |
| LLaMA-2-7B | C4 | 0-4 | 5-19 | 20-26 | 27-31 |
| Mistral-7B | WikiText | 0-4 | 5-21 | 22-27 | 28-31 |
| Phi-2 | WikiText | 0-4 | 5-20 | 21-26 | 27-31 |

The qualitative structure of the four phases is highly consistent across different models and datasets, indicating that this is a universal topological property of the Transformer architecture.

### Key Findings

- **Stability of Phase 2 is universal**: Across all tested models (LLaMA-2, Mistral, Phi-2) and datasets (WikiText, C4, etc.), the middle layers always exhibit the most stable topological structure, suggesting that Transformers have an inherent "semantic stability zone".
- **Topological descriptors are robust to the choice of k**: The specific choice of $k$ in k-NN filtration affects numerical values but not the qualitative conclusions; the descriptors remain stable in the range of $k = 5 \sim 30$.
- **$H_0$ (connected components) and $H_1$ (loops) provide complementary information**: $H_0$ primarily reflects changes in clustering structure among prompts, while $H_1$ reflects higher-order loop topological relationships. Their four-phase divisions are highly consistent.
- **Shallow and deep layers are topologically active zones**: Compared to the stable middle zone, topological reorganization is most intense in the first few and last few layers, which explains why layer pruning generally avoids removing initial and final layers.

## Highlights & Insights

- **The conceptually natural mapping of "layer as time"**: Analogizing the layers of an LLM to snapshots of a time series naturally introduces zigzag persistence—a mature mathematical tool for time-varying data—to LLM analysis. This conceptual bridge is elegant and powerful, connecting two previously disjoint fields.
- **The four-phase model holds dual value for interpretability and utility**: It not only provides a new perspective on understanding Transformer mechanisms (initial encoding $\rightarrow$ semantic formation $\rightarrow$ refinement $\rightarrow$ output preparation), but also directly guides the design of layer pruning strategies.
- **A new paradigm for unsupervised pruning**: Topology-based layer pruning is completely independent of downstream tasks or loss information, making it a true structural analysis method. This idea can be transferred to other scenarios requiring the identification of redundant modules (e.g., attention head pruning, MoE expert pruning).
- **Universal findings across models**: The consistency of the four-phase structure across different architectures suggests that this may be a natural structural characteristic formed during Transformer training, which is worth exploring further from a theoretical perspective.

## Limitations & Future Work

- **Evaluation limited to 7B-level models**: Larger-scale models (e.g., 70B, 405B) were not included, making it uncertain whether the four-phase conclusions hold for larger models.
- **Lack of fine-tuning recovery experiments post-pruning**: The paper only evaluates direct pruning performance without exploring the potential for recovery via pruning + light fine-tuning, resulting in an insufficient comparison with the complete SOTA pipeline.
- **Computational cost not fully discussed**: The computational complexity of zigzag persistence scales quickly with the number of prompts and layers, and its scalability in large-scale applications needs to be evaluated.
- **No exploration of token-level analysis**: Point clouds are currently constructed at the prompt granularity, without considering token-level topological evolution, which could reveal more fine-grained model behaviors.
- **Quantitative criteria for the four-phase division are not sufficiently clear**: The phase boundaries are determined based on visual observation of descriptor curves, lacking automated phase detection algorithms.

## Related Work & Insights

- **vs ShortGPT (Layer Pruning)**: ShortGPT uses Block Influence (BI) to measure layer importance based on changes in cosine similarity of hidden states. The proposed method takes a topological perspective, independent of linear metrics like cosine similarity, capturing richer non-linear structural changes. While comparable in pruning effectiveness, the topological perspective here offers deeper interpretability.
- **vs Intrinsic Dimension approaches (Ansuini et al.)**: Intrinsic dimension methods discover semantic emergence in middle layers but only provide scalar descriptions. The topological descriptors in this paper provide multi-dimensional dynamic information (birth rate, death rate, lifetime distribution, etc.), offering a richer representation analysis framework.
- **vs Betti number approaches (Rieck et al.)**: Prior work observed that Betti numbers remain stable across different datasets on the same architecture and decrease with depth. Our zigzag persistence not only captures Betti numbers of each layer but also tracks the persistence and death of the same topological features across layers, providing temporal-like dimension information.
- **vs Representation similarity methods like CKA/CCA**: Methods like CKA measure overall similarity of representations between layers but lose topological structural information. The proposed method is complementary to CKA and can be used jointly for a more comprehensive cross-layer analysis.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First to systematically apply zigzag persistence to LLM internal representation analysis, where the "layer-as-time" conceptual mapping is natural and powerful.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Consistency of descriptors is verified across multiple models and datasets, but the depth of comparison in pruning experiments could be strengthened.
- Writing Quality: ⭐⭐⭐⭐ — Clearly explains the complex topological mathematical framework, balancing mathematical rigor and readability well.
- Value: ⭐⭐⭐⭐ — The topological perspective opens up a new direction for LLM understanding and compression, and the four-phase discovery is inspiring, though its practical utility requires further verification.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Unveiling Language-Specific Features in Large Language Models via Sparse Autoencoders](../../ACL2025/model_compression/language_specific_features.md)
- [\[ICML 2025\] Weak-to-Strong Jailbreaking on Large Language Models](weak-to-strong_jailbreaking_on_large_language_models.md)
- [\[ICML 2025\] DLP: Dynamic Layerwise Pruning in Large Language Models](dlp_dynamic_layerwise_pruning_in_large_language_models.md)
- [\[ICML 2025\] From Language Models over Tokens to Language Models over Characters](from_language_models_over_tokens_to_language_models_over_characters.md)
- [\[ICML 2025\] SepLLM: Accelerate Large Language Models by Compressing One Segment into One Separator](sepllm_accelerate_large_language_models_by_compressing_one_segment_into_one_sepa.md)

</div>

<!-- RELATED:END -->

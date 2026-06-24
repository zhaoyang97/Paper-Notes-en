---
title: >-
  [Paper Note] Smart: A GNN-LLM Hybrid Surrogate Model for Dragonfly System Application Runtime Prediction
description: >-
  [AAAI 2026][Autonomous Driving][Dragonfly Network] This paper proposes Smart (Surrogate Model for Predicting Application RunTime), the first approach to integrate GNN and LLM (Time-LLM) for iterative application runtime prediction in Dragonfly interconnection networks. On a 1,056-node system, Smart achieves a minimum MAPE of 1.78% (LAMMPS) with an inference time of only 0.515 seconds, delivering orders-of-magnitude speedup over full-scale simulation.
tags:
  - "AAAI 2026"
  - "Autonomous Driving"
  - "Dragonfly Network"
  - "Graph Neural Network"
  - "Large Language Model"
  - "Surrogate Model"
  - "Hybrid Simulation"
  - "Runtime Prediction"
date: 2026-05-08
content_hash: 2978813c3b5b0328
---

# Smart: A GNN-LLM Hybrid Surrogate Model for Dragonfly System Application Runtime Prediction

**Conference**: AAAI 2026
**arXiv**: [2511.11111](https://arxiv.org/abs/2511.11111)  
**Code**: [https://github.com/SPEAR-UIC/SMART](https://github.com/SPEAR-UIC/SMART)  
**Area**: High-Performance Computing / Network Simulation / Surrogate Modeling
**Keywords**: Dragonfly Network, Graph Neural Network, Large Language Model, Surrogate Model, Hybrid Simulation, Runtime Prediction

## TL;DR

This paper proposes Smart (Surrogate Model for Predicting Application RunTime), the first approach to integrate GNN and LLM (Time-LLM) for iterative application runtime prediction in Dragonfly interconnection networks. On a 1,056-node system, Smart achieves a minimum MAPE of 1.78% (LAMMPS) with an inference time of only 0.515 seconds, delivering orders-of-magnitude speedup over full-scale simulation.

## Background & Motivation

The Dragonfly network is the dominant interconnection topology in high-performance computing (HPC), with six of the top ten supercomputers on the Top500 list—including Frontier and Aurora—adopting this design. Its high-radix, low-diameter structure achieves a favorable balance between cost-efficiency and scalability. However, a critical challenge remains: **workload interference**—when multiple applications execute concurrently over shared links, the resulting dynamic network traffic causes significant fluctuations in per-application iteration time.

The conventional analytical approach is **parallel discrete-event simulation (PDES)**, exemplified by CODES/ROSS, which models network behavior at flit granularity with high fidelity. Nevertheless, such high-fidelity PDES is computationally prohibitive: a single 1,056-node Dragonfly simulation requires 36–66 hours, rendering it impractical for large-scale or real-time use cases.

**Hybrid simulation** embeds data-driven surrogate models into the PDES pipeline to achieve substantial speedup while preserving accuracy. Existing surrogate models, however, face three fundamental challenges:

- **Spatiotemporal coupling complexity**: Network traffic is highly dynamic at millisecond timescales, exhibiting both topological spatial dependencies and temporal evolution. Conventional LSTM/ARIMA approaches model only the temporal dimension, neglecting spatial structure.
- **Multi-application interference**: Concurrently competing workloads contend for shared network resources, causing iteration time to drift dynamically with network load.
- **Accuracy-efficiency trade-off**: A surrogate model must be sufficiently accurate to substitute for PDES while remaining lightweight enough for real-time inference.

The most closely related prior work, DCRNN (Diffusion Convolutional Recurrent Neural Network), was validated only on a small 72-node system and degrades substantially when scaled to larger configurations. Smart directly addresses these limitations by proposing, for the first time, a GNN+LLM fusion architecture for runtime surrogate modeling of large-scale HPC interconnection networks.

## Method

### Overall Architecture

Smart comprises three major components forming an end-to-end spatiotemporal prediction pipeline:

1. **GNN Encoder**: Captures spatial structural dependencies of the Dragonfly topology using GCN.
2. **Temporal Transformer**: Models temporal dependencies across the sequence of GNN embeddings.
3. **LLM Prediction Module**: Captures long-range temporal patterns via the Time-LLM mechanism, augmented with domain-knowledge prompts.

The outputs of all three components are concatenated node-wise over active nodes and passed through a fully connected layer to produce the final prediction.

**Input Representation**: The Dragonfly network is modeled as a temporal graph sequence $G_1, G_2, \dots, G_T$, where nodes represent router ports and edges are defined by intra-router all-to-all port connections plus global/local link connections. Each node carries port-level network state features sampled every 250 μs and aggregated into min/max/avg/quantile statistics over iteration windows.

**Problem Formulation**: Given the lookback window of per-process $p$ iteration time series $\{y_{t-(L_y-1)}, \dots, y_t\}$ and network features $\{x_{t-(L_x-1)}, \dots, x_t\}$, the model predicts the next iteration time $y_{t+1}$.

### Key Designs

#### 1. GNN Encoder (Spatial Modeling)

A two-layer GCN spatially encodes the graph at each time step:

$$H^{(l+1,t)} = \sigma\left(\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}H^{(l,t)}W^{(l)}\right)$$

- $\tilde{A} = A + I_N$ denotes the adjacency matrix with added self-loops.
- The input is a graph sequence over a lookback window $T_{inGNN}$; the output is per-time-step node embeddings $H^{(t)} \in \mathbb{R}^{|V| \times d_h}$ ($d_h=128$).
- Experiments show that two GCN layers suffice to capture local topology; additional depth introduces over-smoothing.

#### 2. Temporal Transformer (Temporal Modeling)

Receives the GNN embedding sequence $\{H^{(1)}, \dots, H^{(T_{inGNN})}\}$ and captures cross-timestep dependencies via multi-head self-attention:

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

An encoder-decoder structure is adopted (2 layers each, 8 attention heads). The encoder processes the full temporal embedding; the decoder uses the most recent timestep embedding as the query and integrates encoder outputs to produce a joint spatiotemporal embedding $Z^{(t)} \in \mathbb{R}^{|V| \times d_z}$ ($d_z=128$).

#### 3. LLM Prediction Module (Long-Range Temporal Modeling + Domain Knowledge Injection)

Based on the Time-LLM mechanism, a frozen GPT-2 (32 layers, $d_{llm}=768$) processes the application iteration time series:

- **Patch Embedding**: The time series is segmented into patches (patch length=2, stride=1) and mapped to an LLM-compatible dimension via a linear layer.
- **Prompt-as-Prefix (PaP)**: Domain knowledge is injected as a prefix before patch embeddings, including workload name, lookback window length, and time series statistics (min/max/median/trend).
- **Multi-Head Attention Alignment**: Patch embeddings are projected into the LLM embedding space and fed into frozen GPT-2 to generate $E^{(t)} \in \mathbb{R}^{|V_a| \times d_{llm}}$.
- **Dimension Compression**: A linear projection yields $F^{(t)} \in \mathbb{R}^{|V_a| \times d_z}$, aligned with the Transformer output dimension.

#### 4. Fusion and Prediction

The Transformer output $Z^{(t)}$ is filtered by the active-node mask to obtain $Z_a^{(t)}$, which is concatenated with LLM embeddings $F^{(t)}$ along the node dimension. A fully connected layer then produces the final prediction $\hat{y}_{t+1}$. The model learns the relative weights of GNN spatial features and LLM temporal context in an end-to-end manner.

### Loss & Training

- **Offline Training**: Trained on 30% of data using dual NVIDIA A100 GPUs, requiring approximately 3–4 hours.
- **Online Fine-Tuning**: During inference, model weights are updated via backpropagation every $F_t$ iterations using ground-truth values fed back from PDES. $F_t=8$ yields optimal performance, enabling the model to adapt to evolving network traffic dynamics. This forms an adaptive closed loop of "simulate → predict → validate → fine-tune."

## Key Experimental Results

### Dataset and Setup

- **D1**: 1,056-node Dragonfly; MILC (512 nodes) + LAMMPS (512 nodes) + UR background traffic (36 nodes).
- **D2**: Extension of D1 with an additional NN workload (3D stencil computation); MILC 384 + LAMMPS 512 + NN 160 nodes.
- Both datasets are evaluated under contiguous and random job placement strategies.
- Evaluation metric: MAPE (Mean Absolute Percentage Error).

### Main Results

**Table 1: MAPE (%) Comparison on D1 Dataset under Best Configuration**

| Model | MILC (Cont) | LAMMPS (Cont) | MILC (Rand) | LAMMPS (Rand) |
|-------|-------------|---------------|-------------|---------------|
| **Smart** (best) | **3.19** | **1.78** | **3.12** | **1.84** |
| DCRNN | 6.35 | 7.10 | 5.98 | 7.10 |
| LSTM | 6.09 | 7.25 | 6.05 | 7.28 |
| LAST | 7.86 | 8.14 | 7.83 | 8.21 |
| MEAN | 7.80 | 8.24 | 9.31 | 7.72 |

Smart substantially outperforms all baselines under all conditions, reducing MAPE by approximately 47–75%.

**Table 2: Inference Time and Simulation Speedup**

| Model | Avg. Inference Time (s) | Relative to Min PDES Iteration (8.09s) |
|-------|------------------------|----------------------------------------|
| Smart | 0.515 | 6.4% |
| LLM-only | 0.416 | 5.1% |
| GNN-only | 0.063 | 0.8% |
| DCRNN | 0.046 | 0.6% |
| LSTM | 0.040 | 0.5% |
| MEAN | 0.00001 | ~0% |

Although Smart has the highest inference latency, it still represents only 6.4% of the shortest PDES iteration time (8.09s) and merely 0.2% of the longest simulation iteration (243s).

### Ablation Study

- **GNN-only** (LLM removed): D1-Cont MILC MAPE increases from 3.19% to 3.46%; LAMMPS from 1.78% to 2.05%. D1-Rand LAMMPS degrades severely from 1.84% to 4.59%, demonstrating the critical role of the LLM in temporal modeling.
- **LLM-only** (GNN removed): D1-Cont MILC MAPE increases from 3.19% to 4.02%, confirming the indispensability of spatial features.
- **Time-LLM Replacement**: Substituting Autoformer raises MILC MAPE from 3.91% to 4.22%; substituting DLinear causes NN MAPE to spike from 3.07% to 4.11%, highlighting Time-LLM's advantage under non-stationary patterns.

### Key Findings

The optimal configuration is $T_{inLLM}=8, T_{inGNN}=2, F_t=8$: a longer LLM window captures long-range temporal patterns, while a shorter GNN window focuses on recent spatial dynamics. Variations in patch length, GCN depth, and LLM hidden dimension have limited impact on MAPE (ranging 3.1–3.8%), indicating robustness to hyperparameter choice.

## Highlights & Insights

1. **First GNN+LLM Fusion for HPC Network Surrogate Modeling**: This work innovatively introduces the prompt mechanism of Time-LLM into the network simulation domain. By injecting workload names and time-series statistics as domain knowledge via PaP, the frozen LLM is adapted to HPC scenarios without any parameter fine-tuning.
2. **Online Fine-Tuning Closed-Loop Design**: Model weights are updated every 8 iterations using PDES feedback, enabling continuous adaptation to network traffic drift and forming a practical hybrid simulation framework.
3. **Quantified Spatial-Temporal Complementarity**: Ablation experiments clearly demonstrate the respective contributions of GNN and LLM—GNN excels at modeling topology-induced spatial correlations (LAMMPS MAPE under random placement drops from 4.59% to 1.84%), while LLM captures long-range temporal patterns.
4. **Strong Practical Utility**: Inference at 0.515 seconds versus 36–66 hours of simulation represents a true leap from "hour-scale" to "sub-second-scale" prediction.

## Limitations & Future Work

1. **LLM Overhead**: Smart's inference time (0.515s) is 8× that of GNN-only (0.063s), which may become a bottleneck in extremely latency-sensitive scenarios.
2. **Topology Generalization Unverified**: Experiments are conducted solely on 1D Dragonfly; while the paper claims adaptability to Fat Tree and other topologies, no empirical evidence is provided.
3. **Single-Step Prediction Only**: The model predicts only $y_{t+1}$; multi-step prediction capability is not evaluated, and error accumulation in long-horizon forecasting is not discussed.
4. **Frozen GPT-2**: LLM parameters are entirely frozen and driven only by patch embeddings and prompts; whether fine-tuning the LLM could further improve accuracy remains unexplored.
5. **Single-Run Experiments**: All reported results are based on single experimental runs, lacking statistical confidence intervals from repeated trials.
6. **Limited Dataset Diversity**: Only 2–3 workload combinations are tested; performance under more heterogeneous application scenarios is not evaluated.

## Related Work & Insights

- **Application Runtime Prediction**: Conventional methods (ARIMA, MLP, CNN, RNN, LSTM) primarily model time series while neglecting spatial dependencies inherent in HPC systems. DCRNN first introduced graph structure but was validated only on a small 72-node system.
- **LLMs for Time Series**: Works such as TimesFM, Time-LLM, and TimeGPT-1 explore the potential of LLMs in time-series forecasting; Smart is the first to apply this paradigm to HPC network scenarios.
- **GNN-Based Network Modeling**: RouteNet and XNet apply GNNs to estimate delay/jitter/packet loss, but target general-purpose networks rather than HPC Dragonfly interconnects.
- **PraNet**: Combines GNN, Transformer, and queuing theory to model short-term traffic bursts, but targets Internet/metaverse scenarios and relies on limited simulation data.

## Rating

| Dimension | Score (1–10) | Notes |
|-----------|-------------|-------|
| Novelty | 7 | First GNN+LLM fusion for HPC surrogate modeling, though all components (GCN/Transformer/Time-LLM) are existing methods assembled together |
| Technical Depth | 7 | Spatiotemporal modeling design is sound and online fine-tuning has engineering merit, but fusion is achieved via simple concatenation |
| Experimental Thoroughness | 8 | Two datasets, two placement strategies, multiple hyperparameter sweeps, full ablation, and comprehensive temporal model comparisons |
| Writing Quality | 7 | Structure is clear and problem formulation is rigorous, though dense tables reduce readability |
| Value | 8 | 0.515s inference replacing 36–66h simulation addresses a clear deployment need |
| **Overall** | **7.5** | A practical system targeting an important HPC problem; the GNN+LLM fusion is novel and effective, though the innovation is primarily in component combination |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Threshold-Based Exclusive Batching for LLM Inference](../../ICML2026/autonomous_driving/threshold-based_exclusive_batching_for_llm_inference.md)
- [\[AAAI 2026\] RoadSceneVQA: Benchmarking Visual Question Answering in Roadside Perception Systems for Intelligent Transportation System](roadscenevqa_benchmarking_visual_question_answering_in_roadside_perception_syste.md)
- [\[CVPR 2026\] W2W: Language-Model-Based Trajectory Prediction with Reinforcement Learning](../../CVPR2026/autonomous_driving/w2w_language-model-based_trajectory_prediction_with_reinforcement_learning.md)
- [\[ICML 2025\] Hybrid Quantum-Classical Multi-Agent Pathfinding](../../ICML2025/autonomous_driving/hybrid_quantum-classical_multi-agent_pathfinding.md)
- [\[CVPR 2026\] KnowVal: A Knowledge-Augmented and Value-Guided Autonomous Driving System](../../CVPR2026/autonomous_driving/knowval_a_knowledge-augmented_and_value-guided_autonomous_driving_system.md)

</div>

<!-- RELATED:END -->

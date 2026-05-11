---
title: >-
  [Paper Note] Maximizing Asynchronicity in Event-based Neural Networks
description: >-
  [ICLR 2026][Self-Supervised Learning][event camera] This paper proposes EVA, a framework that treats events as language tokens and employs an RWKV-6-based linear attention asynchronous encoder to update features event-by…
tags:
  - "ICLR 2026"
  - "Self-Supervised Learning"
  - "event camera"
  - "asynchronous processing"
  - "linear attention"
  - "RWKV-6"
  - "A2S"
date: 2026-05-08
content_hash: d1a3a66497810544
---

# Maximizing Asynchronicity in Event-based Neural Networks

**Conference**: ICLR 2026
**arXiv**: [2505.11165](https://arxiv.org/abs/2505.11165)
**Code**: [github.com/haohq19/eva](https://github.com/haohq19/eva)
**Area**: Event Cameras / Efficient Inference
**Keywords**: event camera, asynchronous processing, linear attention, self-supervised learning, RWKV-6, A2S

## TL;DR
This paper proposes EVA, a framework that treats events as language tokens and employs an RWKV-6-based linear attention asynchronous encoder to update features event-by-event. Combined with a self-supervised learning scheme consisting of Multi-Representation Prediction (MRP) and Next-Representation Prediction (NRP), EVA learns generalizable features and, for the first time, successfully tackles the challenging object detection task within the Asynchronous-to-Synchronous (A2S) paradigm (0.477 mAP on the Gen1 dataset).

## Background & Motivation

**Characteristics and challenges of event cameras**: Event cameras output asynchronous sparse event streams with high temporal resolution (up to 1 μs), low latency, and low spatial redundancy. However, standard ML algorithms require tensor-like inputs, creating a fundamental mismatch with the asynchronous and sparse nature of event data.

**Emergence of the A2S paradigm**: The Asynchronous-to-Synchronous (A2S) framework bridges this gap by designing efficient asynchronous encoders that update tensor-like features event-by-event, which downstream synchronous ML algorithms can then sample on demand.

**Limitations of existing A2S methods**: (1) Insufficient encoder expressiveness — ALERT-Transformer uses EventNet (point-cloud-based) without hierarchical learning and can only handle simple recognition tasks; (2) end-to-end supervised training produces task-specific features that lack cross-task generalizability; (3) A2S methods fall significantly short of dense synchronous methods on complex detection tasks.

**Insight from the event–language analogy**: Both share two key similarities — (i) both are organized as sequences, and (ii) both contribute information incrementally (events record incremental brightness changes; words incrementally build semantics). This motivates transferring linear attention and self-supervised learning techniques from NLP to event processing.

**Key differences between events and language**: (i) Information density — a single language token carries rich semantics, whereas a single event records only a pixel-level brightness change and requires aggregation to be meaningful; (ii) Spatial locality — events carry spatial attributes (pixel coordinates), whereas language does not. These two differences guide the architectural design choices.

**Goal**: Design a more expressive asynchronous encoder combined with a self-supervised learning method so that the A2S framework not only surpasses prior A2S methods but also, for the first time, successfully handles challenging detection tasks.

## Method

### Overall Architecture
EVA consists of three major components: (1) an event tokenization and embedding layer that converts raw events into vector representations; (2) an RWKV-6-based asynchronous linear attention encoder that updates features event-by-event; and (3) multi-task self-supervised learning (MRP + NRP) for training the encoder. During inference, the encoder updates features incrementally per event, and downstream tasks sample features on demand for recognition or detection.

### Key Design 1: Event Tokenization and Embedding

- **Function**: Maps each event $e_i = (t_i, x_i, y_i, p_i)$ to a vector $\bm{x}_i \in \mathbb{R}^D$.
- **Mechanism**: Spatial tokenization uses the bijective mapping $\text{Tok}(x, y, p) = p \times H \times W + y \times W + x$ with a vocabulary size of $2 \times H \times W$. Temporal embedding is computed via sinusoidal encoding of the inter-event time difference $\Delta t_i = t_i - t_{i-1}$ (rather than absolute timestamps). The final embedding is the sum of the spatial and temporal embeddings.
- **Design Motivation**: Using time-difference embeddings instead of absolute timestamps avoids the length extrapolation failure analogous to that in language models caused by continuously growing absolute timestamps during long-running operation. The bijective mapping guarantees a unique token for each spatial location–polarity combination.

### Key Design 2: Matrix-Valued Hidden State (MVHS) as Output

- **Function**: Uses the 2-D matrix hidden state $\bm{S} \in \mathbb{R}^{N \times D_{\text{head}} \times D_{\text{head}}}$ of RWKV-6 linear attention as the encoder output feature, rather than the conventional 1-D vector output $\bm{y} \in \mathbb{R}^{D}$.
- **Mechanism**: The recurrent form of RWKV-6 is $\bm{S}_i = \text{diag}(\bm{w}_i) \bm{S}_{i-1} + \bm{k}_i \bm{v}_i^T$, so the hidden state naturally carries aggregated global information. A multi-head mechanism is adopted with per-head dimension $D_{\text{head}} = D/N$, yielding a hidden state of size $N \times D_{\text{head}} \times D_{\text{head}}$, which expands representational capacity without increasing model width $D$.
- **Design Motivation**: (1) Individual events carry low information density and require aggregation — the hidden state is precisely the carrier of aggregated global information. (2) MVHS reduces model size by approximately $D_{\text{model}}/N$ compared to using a 1-D output, enabling lightweight real-time processing. (3) The 2-D structure facilitates learning fine-grained spatial features.

### Key Design 3: Patch-wise Encoding (PWE)

- **Function**: Assigns events to different patches according to their spatial coordinates, with each patch independently encoding its features.
- **Mechanism**: For an event camera with resolution $(H_{\text{sensor}}, W_{\text{sensor}})$ and patch size $P$, events are partitioned into $H_{\text{sensor}} \times W_{\text{sensor}} / P^2$ sequences, each processed by an independent encoder. The per-patch features are concatenated and fed to downstream tasks.
- **Design Motivation**: (1) Exploits the spatial locality of events (a key distinction from language), reducing sequence length and computational overhead. (2) Training the encoder on fixed-size patches naturally supports event cameras with different resolutions. (3) Model size is reduced by approximately the number of patches, and patches can be processed in parallel.

### Key Design 4: Multi-task Self-supervised Learning

- **Function**: Trains the encoder using two self-supervised objectives, MRP and NRP, without requiring downstream task labels.
- **Mechanism (MRP)**: Forces encoded features $\mathcal{F}_i = \mathcal{M}_\theta(\{e_j\}_{j \leq i})$ to predict multiple hand-crafted representations (event count EC, time surface TS, etc.):
$$\arg\max_{\theta, \Theta} \mathbb{E}_i \prod_{k=1}^{K} \textbf{Pr}(\mathcal{R}_i^k | \mathcal{F}_i; \theta_k)$$
- **Mechanism (NRP)**: Inspired by next-token prediction, predicts representations constructed from events within a future time window $\Delta T$:
$$\arg\max_{\theta, \Theta'} \mathbb{E}_i \prod_{k=1}^{K'} \textbf{Pr}(\mathcal{R}^k(\{e | t_i < t(e) \leq t_i + \Delta T\}) | \mathcal{F}_i; \theta_k')$$
- **Design Motivation**: (1) Different hand-crafted representations capture different aspects of event information, so multi-representation learning produces more generalizable features. (2) NRP forces the model to understand motion patterns rather than simply memorizing history. (3) A single event provides insufficient prediction signal and carries unpredictable noise; using aggregated representations as targets is more reliable.

## Key Experimental Results

### DVS128-Gesture Action Recognition

| Model | Encoder Params | Classifier Params | MAC/event | Latency | SA | FVA |
|-------|---------------|------------------|-----------|---------|-----|-----|
| ALERT-Tr. (+RM) | 1.41M | 13.96M | 1.22M | 5.8ms | 84.6% | 94.1% |
| ALERT-Tr. (+LMM) | 0.04M | 0.57M | 0.004M | 3.9ms | 72.6% | 89.2% |
| **EVA (+ResNet-14)** | **0.62M** | **2.83M** | **0.60M** | **14.7ms** | **92.9%** | **96.9%** |

### Gen1 Object Detection

| Model | Type | mAP (%) |
|-------|------|---------|
| NVS-S | End-to-end Async (A) | 8.6 |
| AEGNN | End-to-end Async (A) | 14.5 |
| DAGr-L | End-to-end Async (A) | 32.1 |
| FARSE-CNN | End-to-end Async (A) | 30.0 |
| ASTMNet | Sync Dense (S) | 46.7 |
| RVT-B | Sync Dense (S) | 47.2 |
| GET | Sync Dense (S) | 47.9 |
| **EVA (+RVT-B, D=128)** | **A2S** | **47.5** |
| **EVA-L (+RVT-B, D=192)** | **A2S** | **47.7** |

### Ablation Study

| MVHS | Temporal Embedding | FVA | SA |
|------|--------------------|-----|-----|
| ✓ | ✓ | **98.1%** | **94.7%** |
| ✓ | ✗ | 87.8% | 81.1% |
| ✗ | ✓ | 97.4% | 94.1% |

### Key Findings

- **A2S paradigm conquers detection for the first time**: EVA achieves 47.7 mAP on Gen1, surpassing the synchronous SOTA method RVT-B (47.2). This is the first time an A2S method has achieved competitive results on a detection task; prior A2S methods were limited to simple recognition tasks.
- **MVHS substantially improves representational capacity**: Removing MVHS causes SA to drop from 94.7% to 94.1% (−0.6%), while removing temporal embeddings has a larger negative impact (SA drops from 94.7% to 81.1%), indicating that temporal modeling is critical for event processing.
- **MRP benefits from multi-representation co-learning**: When learning only the EC representation, the EC loss is actually higher (0.701 vs. 0.366), demonstrating positive transfer among multiple representations during joint learning.
- **NRP contributes independently of MRP**: Removing NRP causes FVA to drop from 98.1% to 96.8% and SA from 94.7% to 94.4%, confirming that predicting future representations helps the model acquire knowledge beyond simple history memorization.
- **Smaller patches yield better performance**: Increasing patch size from 16 to 128 causes FVA to drop from 98.1% to 97.4% and SA from 94.7% to 89.3%, despite larger patches having lower pre-training loss due to more sparse regions.

## Highlights & Insights

- **Systematic analysis of the event–language analogy**: Rather than a superficial analogy, the paper systematically analyzes both similarities (sequential structure, incremental information) and differences (information density, spatial locality), and makes targeted architectural adjustments accordingly — MVHS addresses low information density, and PWE addresses spatial locality.
- **First successful application of RWKV-6 in the event domain**: The parallel training and recurrent inference of linear attention naturally match the training and inference requirements of the A2S paradigm, and the data-dependent decay and gating mechanism of RWKV-6 is well-suited to continuous dynamic data.
- **Paradigm shift from 1-D to 2-D features**: Using matrix hidden states instead of vector outputs is a novel idea that expands representational capacity without increasing model width, and the 2-D structure is naturally compatible with image-based downstream tasks.
- **Cross-task transfer of self-supervised features**: Encoder features pre-trained on Gen1 can be directly applied to N-Cars classification (96.3% accuracy), validating the generalizability of the learned representations.

## Limitations & Future Work

- **Real-time throughput is limited at high resolutions**: The event rate of Gen1 (0.618M/s) already exceeds the throughput of EVA-L (0.541M/s); while the PWE strategy can partially mitigate this, scaling to higher-resolution cameras such as Gen3 (1280×720) remains challenging.
- **Self-supervised targets rely on hand-crafted representations**: The supervision signals for MRP and NRP are derived from hand-crafted representations such as EC and TS, which may themselves discard certain event information, imposing an upper bound on what can be learned.
- **Validation limited to the event domain**: Although the framework is theoretically general, experiments are conducted only on event camera data; applicability to other asynchronous sequential data (e.g., neural spike trains) has not been explored.
- **Encoder latency is relatively high**: Due to the hierarchical learning architecture, EVA's per-event inference latency (14.7 ms for 8,192 events) is higher than that of ALERT-Tr., although the total processing time is shorter.

## Related Work & Insights

### vs. ALERT-Transformer (Martin-Turrero et al., 2024)
The previously strongest A2S method, which uses EventNet for asynchronous encoding. EVA improves FVA by 2.8% (96.9% vs. 94.1%) and SA by 8.3% (92.9% vs. 84.6%) on DVS128-Gesture. More importantly, ALERT-Tr. never reported results on detection tasks, whereas EVA achieves 47.7 mAP. The key distinction is that EVA replaces EventNet with RWKV-6 to enable hierarchical learning, and MVHS expands representational capacity.

### vs. RVT-B (Gehrig & Scaramuzza, 2023)
The synchronous dense SOTA method, achieving 47.2 mAP on Gen1. EVA-L surpasses it with 47.7 mAP, despite using only 6 input feature channels compared to RVT-B's 20. This demonstrates that the A2S paradigm, with a sufficiently expressive asynchronous encoder, can match or even exceed synchronous methods while retaining the low-latency advantage of asynchronous processing.

### vs. DAGr (Gehrig & Scaramuzza, 2024)
An end-to-end asynchronous graph neural network method achieving 32.1 mAP on Gen1. EVA's 47.7 mAP substantially surpasses it (+15.6), suggesting that the A2S "encode + dense downstream" paradigm is more effective than purely asynchronous methods, which are constrained by the limitations of graph-based approaches in temporal accumulation.

## Rating

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| Novelty | ⭐⭐⭐⭐ | The systematic analysis of the event–language analogy and the MVHS output design are novel, though the core components (RWKV-6, SSL) are not themselves new. |
| Technical Depth | ⭐⭐⭐⭐ | Architectural design is well-motivated, ablation studies are thorough, and the logical chain from analogy to architectural adaptation is complete. |
| Experimental Thoroughness | ⭐⭐⭐⭐ | Covers recognition, detection, ablation, and timing analysis, but validation across more datasets and downstream tasks is lacking. |
| Engineering Value | ⭐⭐⭐⭐⭐ | First A2S method to tackle detection, PWE supports arbitrary resolutions, code is open-sourced, and the work has direct practical value for real-time event camera applications. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Spikingformer: A Key Foundation Model for Spiking Neural Networks](../../AAAI2026/self_supervised/spikingformer_a_key_foundation_model_for_spiking_neural_networks.md)
- [\[ICLR 2026\] Maximizing Incremental Information Entropy for Contrastive Learning](maximizing_incremental_information_entropy_for_contrastive_learning.md)
- [\[CVPR 2026\] SpHOR: A Representation Learning Perspective on Open-set Recognition for Identifying Unknown Classes in Deep Neural Networks](../../CVPR2026/self_supervised/sphor_a_representation_learning_perspective_on_open-set_recognition_for_identify.md)
- [\[NeurIPS 2025\] Manifolds and Modules: How Function Develops in a Neural Foundation Model](../../NeurIPS2025/self_supervised/manifolds_and_modules_how_function_develops_in_a_neural_foundation_model.md)
- [\[ICLR 2026\] Soft Equivariance Regularization for Invariant Self-Supervised Learning](soft_equivariance_regularization_for_invariant_self-supervised_learning.md)

</div>

<!-- RELATED:END -->

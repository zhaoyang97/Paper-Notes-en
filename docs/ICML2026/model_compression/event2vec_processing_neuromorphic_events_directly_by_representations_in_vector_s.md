---
title: >-
  [Paper Note] Event2Vec: Processing Neuromorphic Events Directly by Representations in Vector Space
description: >-
  [ICML 2026][Model Compression][event camera] Inspired by word2vec, sparse asynchronous events $(x,y,t,p)$ from event cameras are directly embedded into a vector space. By utilizing parametric spatial embedding + convolut…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "event camera"
  - "event2vec"
  - "Transformer"
  - "spatial embedding"
  - "K-Means aggregation"
date: 2026-05-08
content_hash: a710d56ce51da9ca
---

# Event2Vec: Processing Neuromorphic Events Directly by Representations in Vector Space

**Conference**: ICML 2026  
**arXiv**: [2504.15371](https://arxiv.org/abs/2504.15371)  
**Code**: https://github.com/Intelligent-Computing-Lab-Panda/event2vec  
**Area**: Neuromorphic Computing / Event Cameras / Representation Learning  
**Keywords**: event camera, event2vec, Transformer, spatial embedding, K-Means aggregation  

## TL;DR
Inspired by word2vec, sparse asynchronous events $(x,y,t,p)$ from event cameras are directly embedded into a vector space. By utilizing parametric spatial embedding + convolutional temporal embedding + K-Means++ aggregation, a standard Transformer can preserve the sparse asynchronous nature of events while achieving high throughput on GPUs. The parameter count is only $\tfrac{1}{2.8} \sim \tfrac{1}{816}$ of previous SOTAs.

## Background & Motivation

**Background**: Event cameras (DVS, ATIS, etc.) output $(x,y,t,p)$ tuples in AER format—spatial coordinates + microsecond-level timestamps + binary polarity—offering ultra-high temporal resolution, low power consumption, and high dynamic range. Current processing methods are divided into two categories: one accumulates events into dense voxels/frames for CNNs/SNNs, while the other uses irregular models like GNNs, Sparse CNNs, or PointNet to directly process event streams.

**Limitations of Prior Work**: The densification route discards sparsity and microsecond temporal resolution, wasting computational resources on zero pixels. Irregular routes are poorly matched with parallel GPU architectures—SNNs require synchronous simulation on GPUs, leading to slow training and an inference-training gap; Sparse CNNs struggle to fully utilize GPUs; GNNs require careful hyperparameter tuning and suffer from over-smoothing; PointNet-style methods assume permutation invariance, degrading timestamps to coordinates and erasing causal order.

**Key Challenge**: The fundamental incompatibility between sparse-asynchronicity and synchronous dense GPU architectures. Either data characteristics are sacrificed for hardware compatibility, or hardware efficiency is sacrificed for data fidelity.

**Goal**: To find a representation that allows event streams to retain their sparse asynchronous essence while being fed into high-performance Transformers on GPUs.

**Key Insight**: The authors noted a strong analogy between events and NLP words—(1) both consist of "index + position" (events use $(x,y,p)$ as index and $t$ as position; words use vocab id + sequence number); (2) index sets are finite (DVS128 has $2\times128\times128$ addresses); (3) both have natural ordering (words by sentence order, events by timestamps); (4) individual elements require context for full semantics. Since word2vec propelled NLP by embedding discrete words into continuous vector space, events should follow a similar paradigm.

**Core Idea**: Using event2vec, each event is embedded into a $D$-dimensional vector $\mathbf{v} = \text{Embed}_s(x,y,p) + \text{Embed}_t(\Delta t)$. The event stream thus becomes a sequence of tokens of length $L \times D$, which can be processed by any Transformer as an NLP sequence.

## Method

### Overall Architecture
Input: An event sequence $\{(x_i, y_i, t_i, p_i)\}$ of length $L$ (either raw sampled events or representative events after K-Means++ aggregation + density factor $\rho$). Output: Classification logits. Four intermediate steps: (1) Spatial embedding module $\phi$ maps $(x,y,p)$ to a $D$-dimensional vector; (2) Temporal embedding module processes normalized time differences $\Delta t$ using 1D convolution; (3) The sum of both is multiplied by an intensity factor $\log(\rho)+1$ to obtain the token sequence $\mathbf{V} \in \mathbb{R}^{L\times D}$; (4) The sequence is fed into a bidirectional Forgetting Transformer backbone, followed by average pooling and a linear head for classification.

### Key Designs

1. **Parametric Spatial Embedding $\phi$ (Replacing Lookup Tables)**:
    - **Function**: Maps spatial-polarity triplets $(x,y,p)$ to $D$-dimensional dense vectors and explicitly introduces an inductive bias of "spatial proximity $\to$ embedding similarity."
    - **Mechanism**: Coordinates are first normalized to the $[-1,1]$ interval to obtain $(\bar x, \bar y, \bar p)$, then passed through a three-layer MLP $\phi$ (feature dimensions $3 \to D/4 \to D/2 \to D$, with LayerNorm + ReLU). Since $\phi$ is continuously differentiable, the first-order Taylor expansion yields $\phi(x+\Delta x, y+\Delta y, p) - \phi(x,y,p) = J_\phi^{x,y} \cdot [\Delta x, \Delta y]^\top + o(\|\cdot\|)$, naturally satisfying "adjacent coordinates $\to$ embedding difference $\to 0$."
    - **Design Motivation**: Standard NLP embeddings use a lookup table $\mathbf{W}_s \in \mathbb{R}^{(PHW)\times D}$, where there is no prior relationship between index $i$ and $i+1$. This holds for words (vocab id is a non-semantic identifier based on frequency), but fails for image coordinates where pixel $(x,y)$ is highly correlated with $(x+1,y)$. Replacing the lookup table with a continuous function $\phi$ ensures adjacent coordinates map to similar vectors, so the model does not have to learn this simple fact from scratch.

2. **1D Convolutional Temporal Embedding based on $\Delta t$**:
    - **Function**: Encodes continuous, non-uniform timestamps into positional information in the token dimension.
    - **Mechanism**: Timestamps are first normalized $\tilde t = t/\max(t)$, and the first-order difference sequence $\Delta t_i = \tilde t_i - \tilde t_{i-1}$ is calculated (zero-padded for alignment). This is fed into three 1D convolutional layers (kernel size 3, stride 1; channels $1 \to D/4 \to D/2 \to D$, with depthwise conv for the latter two).
    - **Design Motivation**: Popular NLP position encodings like RoPE/ALiBi assume discrete equidistant indices, which are unsuitable for continuous, non-uniform event timestamps. Using $\Delta t$ as input serves as "temporal preconditioning" (similar to residual learning), allowing the network to directly perceive instantaneous event density. Simultaneously, the three-point convolution provides time-shift invariance, neighborhood context consistency, and local smoothing of individual event noise—the sum of $\Delta t$ within a convolutional window equals the cumulative duration of a local time window, representing "local event density."

3. **Batch K-Means++ Aggregation + Intensity Factor $\rho$**:
    - **Function**: Compresses event streams from hundreds of thousands of raw events to a fixed length $L$ while preserving density information.
    - **Mechanism**: While random sampling is a simple baseline, the advanced version performs K-Means clustering independently by polarity to obtain $L$ representative events. The number of original events in each cluster is counted as the intensity factor $\rho_i$. The final token is represented as $\mathbf{V}[i] = (\log(\rho_i)+1)\cdot(\text{Embed}_s + \text{Embed}_t)$. To port the iterative K-Means++ initialization to GPU, the authors proposed Batch K-Means++, using multi-step batch computation to approximate single-step sampling while incrementally updating distances to nearest centers.
    - **Design Motivation**: Random sampling loses too much information in complex tasks (e.g., DVS-Lip). K-Means preserves spatial distribution and explicitly communicates "this region is active" via $\rho$. $\log$ compression prevents a few high-density clusters from dominating the sequence.

### Loss & Training
Standard cross-entropy is used for classification. For DVS-Lip, self-supervised pre-training on clustered events is performed before fine-tuning (details in Appendix A.5). The backbone is a bidirectional parameter-shared version of the Forgetting Transformer (Gated Linear Attention also works but with slightly lower accuracy).

## Key Experimental Results

### Main Results
Three neuromorphic classification benchmarks: DVS Gesture, ASL-DVS, and DVS-Lip.

| Dataset | Prev. SOTA (Method / Params / Accuracy) | Event2Vec (Params / Accuracy) | Compression Ratio |
|---------|-----------------------------------------|-------------------------------|-------------------|
| DVS Gesture | Max-Former / 1.45 MB / 98.60% | 0.52 MB / 97.57±1.31% | 2.79× |
| ASL-DVS | GNN+Transformer / 220.30 MB / 99.60% | 0.27 MB / 99.91±0.05% | 815.93× |
| DVS-Lip | Spiking ResNet18+BiGRU / 223.63 MB / 75.30% | 18.30 MB / 75.88% (K-Means) | 12.22× |

Accuracy is "on par / highest / highest" across the three datasets, while parameter counts are reduced by hundreds of times.

### Ablation Study
Cross-ablation on DVS Gesture for Spatial Embedding (Standard lookup vs. Parametric $\phi$) $\times$ Temporal Embedding (Sinusoidal on $t$ vs. Conv on $\Delta t$), along with throughput/latency comparisons.

| Configuration | DVS Gesture Accuracy | Notes |
|---------------|----------------------|-------|
| Standard lookup + Sinusoidal$(t)$ | Low | Both lack neighborhood inductive bias |
| Standard lookup + Conv$(\Delta t)$ | **Lowest** | Lack of spatial neighborhood bias hinders temporal encoding |
| Parametric $\phi$ + Sinusoidal$(t)$ | Medium | Spatial bias exists but temporal still assumes discrete indices |
| Parametric $\phi$ + Conv$(\Delta t)$ (Full) | **Highest** | Both neighborhood biases synergize |

Throughput and Latency (Table 2 in original paper): Training throughput relative to previous SOTA is 4.21× / 11.96× / 35.36×; Inference throughput is 2.69× / 62.67× / 5.70×; Single-stream end-to-end latency reduced to 68.55% / 11.12% / 14.68%; VRAM usage reduced to 72.18% / 15.08% / 68.35%.

### Key Findings
- Spatial embedding is the upper bound: Replacing standard lookup with parametric $\phi$ is key to accuracy gains. Without spatial neighborhood bias, convolutional temporal encoding performed worse than sinusoidal—confirming the intuition that "both neighborhood biases must synergize."
- Source of parameter efficiency: The inherent sparsity of events and the shared spatial structure of embedding functions mean the network does not need independent convolutional parameters for every pixel location like CNNs.
- K-Means aggregation significantly outperforms random sampling: On DVS-Lip, accuracy improved from 70.62% to 75.88% (+5.26%). Inference-time aggregation is efficient using the GPU-based Batch K-Means++.
- Robust under extremely low event counts / spatial resolutions, making it suitable for real-time neuromorphic vision.

## Highlights & Insights
- Fully migrates the "word2vec paradigm" to event cameras: Embed discrete sparse symbols into a continuous vector space, then leverage the existing Transformer ecosystem. This "lane change" approach is more ambitious than incremental improvements on SNNs or GNNs.
- Uses a continuously differentiable MLP as a substitute for discrete lookup tables and uses first-order Taylor expansion to prove neighborhood inductive bias—a concise and elegant formal argument applicable to any "index space with topological structure" (e.g., 3D voxels, LiDAR points).
- The intensity factor $\rho \to \log(\rho)+1$ is a small but critical detail: it preserves density while preventing aggregated events from overwhelming non-aggregated tokens, making it reusable for point cloud or graph sampling tasks.
- Batch K-Means++ transforms a traditional iterative algorithm into a GPU-friendly one, establishing a practical engineering paradigm that brings preprocessing into the GPU pipeline.

## Limitations & Future Work
- The backbone relies on Forgetting Transformer; replacing it with standard linear attention results in a performance drop. Adaptation is needed for standard LLM Flash-Attention.
- Sequence length $L$ is fixed, meaning adaptive cropping/expansion of event volume requires retraining; dynamic length schemes were not demonstrated.
- Validated only on classification; performance in killer apps for event cameras (optical flow, HDR reconstruction, SLAM, tracking) is unknown.
- The intensity factor $\rho$ only uses $\log$ compression; learnable density encodings were not explored, nor was the optimal value of $L$ systematically swept.

## Related Work & Insights
- **vs EventNet (Sekikawa 2019)**: Both decouple events into "spatial-polarity address + relative time." EventNet uses a lookup table $h(\mathbf{e})$ + handcrafted complex temporal encoding with PointNet-style max-pooling for asynchronous recursive inference on CPUs; Event2Vec uses learnable continuous $\phi$ + convolution for first-order $\Delta t$ for synchronous Transformer training/inference on GPUs. The former targets neuromorphic hardware; the latter maximizes modern GPU efficiency.
- **vs Sparse CNN / GNN / PointNet-style**: These process irregular data directly but have low GPU utilization. Event2Vec maps irregular data into regular $L\times D$ tensors, benefiting from GPU parallelization.
- **vs Event-Frame (Max-Former, SNN+Frame, etc.)**: Densification methods are accurate but parameter-heavy and lose temporal resolution; Event2Vec preserves microsecond timestamps with 1~3 orders of magnitude higher parameter efficiency.

## Rating
- Novelty: ⭐⭐⭐⭐ Analogizing word2vec to embed events into vector space, with parametric spatial and convolutional temporal embeddings, is novel despite precursors like EventNet.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three major neuromorphic benchmarks + complete metrics (throughput, latency, VRAM) + cross-ablation of embeddings.
- Writing Quality: ⭐⭐⭐⭐ Clear narrative using the word↔event analogy, with well-coordinated formulas and diagrams.
- Value: ⭐⭐⭐⭐ Opens the door for Event Camera + Transformer ecosystems, with substantial engineering significance in parameter compression (2~800×).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] MIC: Maximizing Informational Capacity in Adaptive Representations via Isotropic Subspace Alignment](mic_maximizing_informational_capacity_in_adaptive_representations_via_isotropic_.md)
- [\[ICML 2026\] Exploiting Weight-Space Symmetries for Approximating Curvature](exploiting_weight-space_symmetries_for_approximating_curvature.md)
- [\[ICLR 2026\] LLM DNA: Tracing Model Evolution via Functional Representations](../../ICLR2026/model_compression/llm_dna_tracing_model_evolution_via_functional_representations.md)
- [\[ICML 2026\] ArcVQ-VAE: A Spherical Vector Quantization Framework with ArcCosine Additive Margin](arcvq-vae_a_spherical_vector_quantization_framework_with_arccosine_additive_marg.md)
- [\[ICML 2026\] Hallucination is a Consequence of Space-Optimality: A Rate-Distortion Theorem for Membership Testing](hallucination_is_a_consequence_of_space-optimality_a_rate-distortion_theorem_for.md)

</div>

<!-- RELATED:END -->

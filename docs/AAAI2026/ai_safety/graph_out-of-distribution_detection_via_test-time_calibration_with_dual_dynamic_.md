---
title: >-
  [Paper Note] Graph Out-of-Distribution Detection via Test-Time Calibration with Dual Dynamic Dictionaries
description: >-
  [AAAI 2026][AI Safety][Graph OOD Detection] This paper proposes the BaCa framework, which generates boundary-aware synthetic graph topologies at test time via graphon estimation and mixup strategies…
tags:
  - "AAAI 2026"
  - "AI Safety"
  - "Graph OOD Detection"
  - "Test-Time Calibration"
  - "Graphon Mixup"
  - "Dual Dynamic Dictionary"
  - "Priority Queue"
date: 2026-05-08
content_hash: f00c65577c566726
---

# Graph Out-of-Distribution Detection via Test-Time Calibration with Dual Dynamic Dictionaries

**Conference**: AAAI 2026
**arXiv**: [2511.13541](https://arxiv.org/abs/2511.13541)  
**Code**: None  
**Area**: LLM Evaluation
**Keywords**: Graph OOD Detection, Test-Time Calibration, Graphon Mixup, Dual Dynamic Dictionary, Priority Queue

## TL;DR

This paper proposes the BaCa framework, which generates boundary-aware synthetic graph topologies at test time via graphon estimation and mixup strategies, and adaptively calibrates OOD scores using dual priority-queue-based dynamic dictionaries with an attention mechanism. Without fine-tuning the pretrained model or requiring auxiliary OOD data, BaCa outperforms GOODAT on all 10 datasets with an average AUC improvement of 8.37%.

## Background & Motivation

**Background**: Graph-level OOD detection aims to determine whether a test graph sample originates from a different distribution than the training data. Existing methods fall into two categories: end-to-end methods (training OOD-aware GNNs from scratch) and post-hoc methods (adding a detector on top of a trained GNN). Both rely on defining OOD scoring functions based on model outputs or latent features.

**Limitations of Prior Work**:
- GNNs trained solely on in-distribution (ID) data struggle to identify OOD samples whose features lie close to the ID manifold (e.g., when sharing similar topological structures).
- Outlier Exposure (OE) requires external OOD data, violating the standard assumption of training only on ID data.
- GOODAT introduces a test-time setting but requires optimizing a learnable graph masker at inference, limiting its stability.
- The latent structure of graph data is governed by multiple factors, causing significant overlap between ID and OOD score distributions near the decision boundary.

**Key Challenge**: Pretrained GNNs lack the ability to model distribution boundaries, leading to severe ID/OOD score overlap, particularly on ambiguous samples near the boundary.

**Goal**: How to model the ID/OOD distribution boundary at test time—without modifying the pretrained model or introducing auxiliary OOD data—and effectively calibrate OOD scores?

**Key Insight**: The intuition is straightforward: if a sample is more OOD than the most ID-like sample in the OOD distribution, it should be classified as OOD, and vice versa. The key therefore lies in accurately capturing the most discriminative sample representations near the boundary.

**Core Idea**: At test time, dynamically maintain two representation dictionaries (implemented as priority queues) for ID and OOD samples, continuously collecting the most representative features near the boundary. An attention mechanism is then used to adaptively calibrate OOD scores, while graphon estimation and mixup generate synthetic samples to enhance the diversity of boundary representations.

## Method

### Overall Architecture

The BaCa (Boundary-aware Calibration) pipeline proceeds as follows:
1. Compute the initial OOD score $S_{Pre}$ for each test sample using a pretrained GNN.
2. Partition samples into ID and OOD groups based on the initial scores.
3. Estimate the graphon for each group, and generate diverse synthetic samples via graphon mixup.
4. Maintain dual dynamic dictionaries via priority queues, continuously collecting representative features near the boundary.
5. Compute a calibrated score $S_{Attn}$ via an attention mechanism; the final score is $S_{BaCa} = S_{Pre} + \beta \cdot S_{Attn}$.

### Key Designs

**Module 1: Boundary-Aware Latent Pattern Modeling**

- **Function**: Partition samples into subgroups based on initial scores → estimate the graphon for each group → apply graphon mixup to generate synthetic graphs with discriminative topologies.
- **Mechanism**:
    - A graphon is a symmetric measurable function $W: \Omega^2 \to [0,1]$ that describes the probability of edge existence between nodes, serving as the limit object of graph sequences.
    - The USVT estimator approximates the graphon as a step function $W \in [0,1]^{N \times N}$.
    - Convex combinations of graphons are computed within the same group (ID or OOD): $W_s = \lambda W_i + (1-\lambda)W_j$.
    - Synthetic graphs are sampled from $W_s$ to populate low-density boundary regions.
    - The target size $r \in [2, N]$ is sampled randomly to increase structural diversity.
- **Design Motivation**:
    - True OOD samples are unavailable at test time, but graphon mixup enables intra-group interpolation to enrich boundary representations.
    - Theorem 1 proves that the mixed graphon preserves the discriminative topological features of the source group, with bias bounded by $\lambda$ and the cut-norm distance.
    - This is particularly important in the early stages of testing, when the dictionaries are not yet sufficiently populated.

**Module 2: Dual Dynamic Dictionaries (Priority Queues)**

- **Function**: Maintain fixed-length ID and OOD dictionaries implemented as priority queues, continuously collecting the most discriminative feature representations near the boundary.
- **Mechanism**:
    - OOD dictionary $\mathcal{K}^{ood}_l$: collects the left tail of the OOD score distribution (OOD samples closest to the ID boundary)—the front of the queue always holds the OOD sample nearest to the boundary.
    - ID dictionary $\mathcal{K}^{id}_l$: collects the right tail of the ID score distribution (ID samples closest to the OOD boundary).
    - A new candidate is inserted only if its OOD score exceeds that of the current front element.
    - Synthetic samples also participate in dictionary updates to increase the diversity of latent patterns.
- **Design Motivation**: Samples near the boundary are most informative—they define the ID/OOD decision frontier. Fixed-length priority queues decouple dictionary size from mini-batch size, enabling reuse across batches. As iterations proceed, the KL divergence between ID and OOD distributions progressively increases, indicating growing separation.

**Module 3: Attention-Based Calibration**

- **Function**: For each test sample, compute its similarity to the Top-$\mathbb{K}$ entries in the ID and OOD dictionaries via an attention mechanism, and output a calibrated score.
- **Mechanism**:
    - Query $q = f(G)$; keys and values are drawn from the Top-$\mathbb{K}$ most relevant dictionary entries.
    - OOD dictionary attention output: $S_{out}(G) = \text{ATTN}_{out}(Q, K, V)$
    - ID dictionary attention output: $S_{in}(G) = -\text{ATTN}_{in}(Q, K, V)$
    - Final calibrated score: $S_{Attn} = S_{in} + S_{out}$
    - An ID sample with high similarity to the ID dictionary and low similarity to the OOD dictionary yields a low $S_{Attn}$; the converse holds for OOD samples.

$$S_{BaCa} = S_{Pre} + \beta \cdot S_{Attn}(G)$$

- **Design Motivation**: Attending only to Top-$\mathbb{K}$ entries (rather than the full dictionary) improves efficiency and reduces noise; the learnable attention weights allow adaptive capture of relevant patterns across diverse graph structures.

### Loss & Training

A dual binary cross-entropy loss supervises the attention parameters $W_Q, W_K, W_V$:

$$\mathcal{L} = -\mathbb{E}_{\mathcal{K}^{id}}[\log(\text{ATTN}_{in}) + \log(1-\text{ATTN}_{out})] - \mathbb{E}_{\mathcal{K}^{ood}}[\log(1-\text{ATTN}_{in}) + \log(\text{ATTN}_{out})]$$

Only the lightweight attention module parameters are trained; the pretrained GNN remains frozen. The computational overhead is minimal, scaling linearly with dictionary size and feature dimensionality.

## Key Experimental Results

### Main Results

AUC (%) on 10 ID/OOD dataset pairs, compared against GOODAT (also a test-time method):

| Dataset Pair | GOODAT | BaCa | Gain |
|---|---|---|---|
| BZR/COX2 | - | ✓ | - |
| PTC/MUTAG | - | ✓ | - |
| ClinTox/LIPO | - | ✓ | +20.11 |
| **Average (10 pairs)** | - | **SOTA** | **+8.37** |

BaCa outperforms GOODAT and all other baselines (including graph kernel methods and end-to-end methods) on all 10 datasets.

### Ablation Study

- Figure 1(d) iteration curves show that the full BaCa model's AUC steadily improves and converges over test-time iterations; the attention module alone is also effective but inferior to the full model; the base pretrained model remains flat.
- KL divergence increases with iterations, confirming that calibration progressively widens the gap between ID and OOD score distributions.
- Removing graphon mixup degrades performance, especially in early iterations; removing the priority queue destabilizes boundary modeling.

### Key Findings

- The test-time calibration paradigm is effective: significant detection improvements are achieved without modifying the model or introducing external OOD data, relying solely on the test samples themselves.
- Graphon mixup not only enhances diversity but also provides theoretical guarantees (Theorem 1: mixed graphons preserve discriminative topology).
- Graph kernel methods (PK-LOF/OCSVM) perform near random (AUC ~50%) on graph OOD tasks, demonstrating that traditional kernel approaches are ill-suited for this problem.

## Highlights & Insights

- **Priority Queue Dynamic Dictionary**: The most elegant data structure solves the problem of continuously capturing boundary information at test time.
- The combination of graphon estimation and mixup is applied to graph OOD detection for the first time, with rigorous theoretical derivation (homomorphism density preservation in Theorem 1).
- The method is entirely free of auxiliary OOD data and model fine-tuning, offering strong practical utility.
- Computational complexity is thoroughly analyzed: dictionary operations $O(d \cdot l)$, queue updates $O(\log l)$, attention $O(2\mathbb{K}d)$.

## Limitations & Future Work

- The quality of the initial grouping depends on $S_{Pre}$ from the pretrained model—if the initial predictions are severely inaccurate (e.g., ID and OOD samples are completely mixed), subsequent calibration may be hindered.
- Graphon USVT estimation may perform poorly on sparse graphs.
- The method involves several hyperparameters (dictionary size $l$, Top-$\mathbb{K}$, mixup coefficient $\lambda$, calibration weight $\beta$), and the associated tuning cost is not thoroughly discussed.
- Evaluation is limited to unsupervised graph-level OOD detection; node-level and edge-level OOD detection remain unexplored.

## Related Work & Insights

- **GOODAT** (2024): A pioneering work on test-time graph OOD detection, but requires training a learnable graph masker.
- **HGOE** (2024): A graph OOD method based on Outlier Exposure, but requires additional OOD data.
- Graphons have been applied in graph generation and augmentation (Xu 2021, Yuan 2025); introducing them to OOD detection is a novel direction.
- The dual dictionary/memory bank design is transferable to other test-time adaptation tasks.

## Rating

⭐⭐⭐⭐ (4/5)

**Rationale**: The method is elegantly designed (priority queue + graphon + attention calibration forming a coherent whole), theoretically well-grounded, and experimentally comprehensive with significant improvements. One point is deducted for the relatively large number of hyperparameters and the insufficient discussion of sensitivity to initial grouping quality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Redundancy-Aware Test-Time Graph Out-of-Distribution Detection](../../NeurIPS2025/ai_safety/redundancy-aware_test-time_graph_out-of-distribution_detection.md)
- [\[AAAI 2026\] Fine-Grained DINO Tuning with Dual Supervision for Face Forgery Detection](fine-grained_dino_tuning_with_dual_supervision_for_face_forgery_detection.md)
- [\[ICLR 2026\] AP-OOD: Attention Pooling for Out-of-Distribution Detection](../../ICLR2026/ai_safety/ap-ood_attention_pooling_for_out-of-distribution_detection.md)
- [\[CVPR 2026\] Enhancing Out-of-Distribution Detection with Extended Logit Normalization](../../CVPR2026/ai_safety/enhancing_out-of-distribution_detection_with_extended_logit_normalization.md)
- [\[AAAI 2026\] FairGSE: Fairness-Aware Graph Neural Network without High False Positive Rates](fairgse_fairness-aware_graph_neural_network_without_high_false_positive_rates.md)

</div>

<!-- RELATED:END -->

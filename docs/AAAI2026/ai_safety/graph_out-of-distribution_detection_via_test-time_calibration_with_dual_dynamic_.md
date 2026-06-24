---
title: >-
  [Paper Note] Graph Out-of-Distribution Detection via Test-Time Calibration with Dual Dynamic Dictionaries
description: >-
  [AAAI 2026][AI Safety][Graph OOD Detection] Proposes the BaCa framework, which generates boundary-aware synthetic graph topologies during the test phase via graphon estimation and a mixup strategy. This is combined with dual priority-queue dynamic dictionaries and an attention mechanism to adaptively calibrate OOD scores. It requires no fine-tuning of the pre-trained model or introduction of auxiliary OOD data, outperforming GOODAT across all 10 datasets with an average AUC i…
tags:
  - "AAAI 2026"
  - "AI Safety"
  - "Graph OOD Detection"
  - "Test-Time Calibration"
  - "Graphon Mixture"
  - "Dual Dynamic Dictionaries"
  - "Priority Queue"
date: 2026-05-08
content_hash: 1e01a3ab8d5f2b47
---

# Graph Out-of-Distribution Detection via Test-Time Calibration with Dual Dynamic Dictionaries

**Conference**: AAAI 2026  
**arXiv**: [2511.13541](https://arxiv.org/abs/2511.13541)  
**Code**: None  
**Area**: LLM Evaluation  
**Keywords**: Graph OOD Detection, Test-Time Calibration, Graphon Mixture, Dual Dynamic Dictionaries, Priority Queue

## TL;DR

Proposes the BaCa framework, which generates boundary-aware synthetic graph topologies during the test phase via graphon estimation and a mixup strategy. This is combined with dual priority-queue dynamic dictionaries and an attention mechanism to adaptively calibrate OOD scores. It requires no fine-tuning of the pre-trained model or introduction of auxiliary OOD data, outperforming GOODAT across all 10 datasets with an average AUC improvement of 8.37%.

## Background & Motivation

**Background**: Graph-level OOD detection aims to determine whether test graph samples originate from a distribution different from the training data. Existing methods are divided into end-to-end approaches (training OOD-aware GNNs from scratch) and post-processing approaches (adding detectors to pre-trained GNNs), both fundamentally centered on defining an OOD scoring function based on model outputs or latent features.

**Limitations of Prior Work**:
   - GNNs trained solely on ID data fail to identify OOD samples whose features are close to the ID manifold (e.g., when they share similar topological structures).
   - Outlier Exposure (OE) requires external OOD data, violating the standard assumption of "training only on ID data".
   - Although GOODAT introduces a test-time setting, it requires optimizing a learnable graph masker during inference, which limits stability.
   - The latent structure of graph data is controlled by various factors, causing the score distributions of ID and OOD to heavily overlap near the decision boundary.

**Key Challenge**: Pre-trained GNNs lack the capability to model distribution boundaries, leading to severe overlap between ID and OOD scores, especially for ambiguous samples near the boundary.

**Goal**: How to model ID/OOD distribution boundaries and effectively calibrate OOD scores during the test phase without modifying the pre-trained model or introducing auxiliary OOD data?

**Key Insight**: Guided by intuition: if a sample is more OOD-like than the most ID-biased sample in the OOD distribution, it should be classified as OOD, and vice versa. Therefore, the key is to accurately capture the most discriminative sample representations at the boundary.

**Core Idea**: Dynamically maintain dual ID and OOD representation dictionaries (implemented via priority queues) at test time to continuously collect the most representative sample features near the boundary, calibrating OOD scores via an attention mechanism, while generating synthetic samples using graphon estimation and mixup to enhance the diversity of boundary representations.

## Method

### Overall Architecture

The workflow of BaCa (Boundary-aware Calibration) is as follows:
1. Compute the initial OOD score $S_{Pre}$ of the test sample using a pre-trained GNN.
2. Split the samples into ID/OOD groups based on initial scores.
3. Estimate the graphon for each group separately, and generate diverse synthetic samples via graphon mixup.
4. Maintain dual dynamic dictionaries using priority queues to continuously collect representative features near the boundary.
5. Compute the calibration score $S_{Attn}$ via an attention mechanism, yielding the final score $S_{BaCa} = S_{Pre} + \beta \cdot S_{Attn}$.

### Key Designs

**Module 1: Boundary-Aware Latent Pattern Modeling**

- **Function**: Sub-group partitioning based on initial judgments $\to$ estimating the graphon of each group $\to$ graphon mixup to generate synthetic graphs with discriminative topologies.
- **Mechanism**:
    - A graphon is a symmetric measurable function $W: \Omega^2 \to [0,1]$ that describes the probability of an edge existing between nodes, acting as the limit object of a sequence of graphs.
    - The USVT estimator is used to approximate the graphon as a step function $W \in [0,1]^{N \times N}$.
    - Perform convex combination of graphons within the same group (ID or OOD): $W_s = \lambda W_i + (1-\lambda)W_j$.
    - Sample synthetic graphs from $W_s$ to fill low-density regions near the boundary.
    - Randomly sample target sizes $r \in [2,N]$ to increase structural diversity.
- **Design Motivation**:
    - Real OOD samples are inaccessible at test time, but interpolating within the same group via graphon mixup can enhance boundary representations.
    - Theorem 1 proves that the mixed graphon preserves the discriminative topological properties of the source group, with the discrepancy bounded by $\lambda$ and the cut-norm distance.
    - This is particularly crucial in the early stages of test-time adaptation (when the dictionaries are not yet fully populated).

**Module 2: Dual Dynamic Dictionaries (Priority Queues)**

- **Function**: Maintain fixed-length ID and OOD dictionaries implemented via priority queues to continuously collect the most discriminative sample features near the boundary.
- **Mechanism**:
    - OOD dictionary $\mathcal{K}^{ood}_l$: Collects the left tail of the OOD score distribution (the OOD samples closest to the ID boundary) — hence the front of the queue always contains the OOD samples nearest to the boundary.
    - ID dictionary $\mathcal{K}^{id}_l$: Collects the right tail of the ID score distribution (the ID samples closest to the OOD boundary).
    - Admission condition for new candidates: the OOD score exceeds the front elements of the queue.
    - Synthetic samples also participate in dictionary updates to increase the diversity of latent patterns.
- **Design Motivation**: Samples near the boundary carry the most information — they define the dividing line between ID and OOD. Fixed-length priority queues decouple the dictionary size from the mini-batch size, supporting cross-batch reuse. As iterations proceed, the KL divergence gradually increases, indicating that the ID/OOD distributions are progressively separated.

**Module 3: Attention-based Calibration**

- **Function**: For each test sample, compute its similarity to the Top-$\mathbb{K}$ entries in the ID/OOD dictionaries using an attention mechanism, outputting a calibrated score.
- **Mechanism**:
    - Query $q = f(G)$, while keys/values come from the Top-$\mathbb{K}$ most relevant entries in the dictionaries.
    - OOD dictionary attention output: $S_{out}(G) = \text{ATTN}_{out}(Q,K,V)$.
    - ID dictionary attention output: $S_{in}(G) = -\text{ATTN}_{in}(Q,K,V)$.
    - Final calibration score: $S_{Attn} = S_{in} + S_{out}$.
    - ID samples exhibit high similarity with the ID dictionary and low similarity with the OOD dictionary $\to$ low $S_{Attn}$; vice versa for OOD samples.

$$S_{BaCa} = S_{Pre} + \beta \cdot S_{Attn}(G)$$

- **Design Motivation**: Considering only the Top-$\mathbb{K}$ entries (rather than all dictionary entries) improves efficiency and reduces noise. The learnability of the attention weights allows adaptive capture of relevant patterns for different graph structures.

### Loss & Training

Dual BCE loss supervises the attention parameters $W_Q, W_K, W_V$:

$$\mathcal{L} = -\mathbb{E}_{\mathcal{K}^{id}}[\log(\text{ATTN}_{in}) + \log(1-\text{ATTN}_{out})] - \mathbb{E}_{\mathcal{K}^{ood}}[\log(1-\text{ATTN}_{in}) + \log(\text{ATTN}_{out})]$$

Only a small number of parameters in the attention module are trained without updating the pre-trained GNN, yielding extremely low computational complexity (linear with respect to dictionary size and feature dimension).

## Key Experimental Results

### Main Results

The AUC (%) on 10 pairs of ID/OOD datasets, compared with GOODAT (also a test-time method):

| Dataset Pair | GOODAT | BaCa | Gain |
|----------|--------|------|------|
| BZR/COX2 | - | ✓ | - |
| PTC/MUTAG | - | ✓ | - |
| ClinTox/LIPO | - | ✓ | +20.11 |
| **Average of 10 pairs** | - | **SOTA** | **+8.37** |

BaCa outperforms GOODAT and all other baselines (including graph kernel methods and end-to-end methods) across all 10 datasets.

### Ablation Study

- The iteration curves in Figure 1(d) show that as test-time iteration proceeds, the AUC of Total (full BaCa) continuously improves and converges; Attn alone is also effective but lower than Total; Base (pre-trained only) remains flat.
- The KL divergence increases with iterations, validating that the calibration indeed widens the gap between ID/OOD distributions.
- Removing graphon mixup $\to$ performance decreases (especially in early iterations); removing the priority queue $\to$ boundary modeling becomes unstable.

### Key Findings

- The test-time calibration paradigm is effective: without modifying the model or introducing external OOD data, leveraging only the test samples themselves can significantly improve detection performance.
- Graphon mixup not only enhances diversity but also provides theoretical guarantees (Theorem 1: the mixed graphon preserves discriminative topology).
- Graph kernel methods (PK-LOF/OCSVM) perform close to random (AUC ~50%) on graph OOD tasks, indicating that traditional kernel methods are unsuitable for this problem.

## Highlights & Insights

- **Priority Queue Dynamic Dictionary**: Solves the problem of "how to continuously capture boundary information during test time" using the simplest data structure, which is designed with extreme elegance.
- The combination of graphon estimation and mixup is used for the first time in graph OOD detection, with rigorous theoretical derivation (homomorphism density preservation in Theorem 1).
- Complete independence from auxiliary OOD data and model fine-tuning makes it highly practical.
- Comprehensive computational complexity analysis: dictionary operations require $O(d \cdot l)$, queue updates require $O(\log l)$, and attention computation requires $O(2\mathbb{K}d)$.

## Limitations & Future Work

- The quality of the initial grouping relies on $S_{Pre}$ from the pre-trained model — if the initial judgments are extremely poor (e.g., ID/OOD are completely mixed), subsequent calibration may be limited.
- Graphon USVT estimation may perform poorly on sparse graphs.
- There are numerous hyperparameters (dictionary size $l$, Top-$\mathbb{K}$, mixup coefficient $\lambda$, calibration weight $\beta$), and the tuning cost is not fully discussed.
- Only unsupervised graph-level OOD detection is evaluated, leaving node-level and edge-level OOD detection to be explored.

## Related Work & Insights

- **GOODAT** (2024): A pioneering work in test-time graph OOD detection, but requires training a learnable masker.
- **HGOE** (2024): A graph OOD method based on Outlier Exposure, but requires additional OOD data.
- Graphon has already been applied in graph generation/augmentation (Xu 2021, Yuan 2025); incorporating it into OOD detection is a novel direction introduced by this paper.
- The design concept of dual dictionaries/memory banks can be transferred to other test-time adaptation tasks.

## Rating

⭐⭐⭐⭐ (4/5)

**Reason**: The method is elegantly designed (a triumvirate of priority queue, graphon, and attention calibration), with solid theoretical derivations, comprehensive experiments, and significant improvements. One point is deducted because there are many hyperparameters and the dependency on the quality of initial grouping is not fully discussed.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] OODD: Test-time Out-of-Distribution Detection with Dynamic Dictionary](../../CVPR2025/ai_safety/oodd_test-time_out-of-distribution_detection_with_dynamic_dictionary.md)
- [\[NeurIPS 2025\] Redundancy-Aware Test-Time Graph Out-of-Distribution Detection](../../NeurIPS2025/ai_safety/redundancy-aware_test-time_graph_out-of-distribution_detection.md)
- [\[AAAI 2026\] Fine-Grained DINO Tuning with Dual Supervision for Face Forgery Detection](fine-grained_dino_tuning_with_dual_supervision_for_face_forgery_detection.md)
- [\[CVPR 2026\] Bypassing the Transport Plan: Dynamic Reweighting for Out-of-Distribution Detection with Optimal Transport](../../CVPR2026/ai_safety/bypassing_the_transport_plan_dynamic_reweighting_for_out-of-distribution_detecti.md)
- [\[CVPR 2026\] RankOOD: Class Ranking-based Out-of-Distribution Detection](../../CVPR2026/ai_safety/rankood_-_class_ranking-based_out-of-distribution_detection.md)

</div>

<!-- RELATED:END -->

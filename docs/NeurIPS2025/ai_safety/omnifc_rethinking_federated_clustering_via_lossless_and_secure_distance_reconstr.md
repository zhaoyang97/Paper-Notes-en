---
title: >-
  [Paper Note] OmniFC: Rethinking Federated Clustering via Lossless and Secure Distance Reconstruction
description: >-
  [NeurIPS 2025][AI Safety][Federated Clustering] OmniFC is proposed as a model-agnostic federated clustering framework that exactly reconstructs the global pairwise distance matrix over a finite field via Lagrange coded computing, enabling any centralized clustering method (K-Means / Spectral Clustering / DBSCAN / Hierarchical Clustering, etc.) to run directly on the reconstructed matrix. The framework requires only a single communication round…
tags:
  - "NeurIPS 2025"
  - "AI Safety"
  - "Federated Clustering"
  - "Lagrange Coded Computing"
  - "Distance Matrix Reconstruction"
  - "Lossless Aggregation"
  - "Secure Computation"
date: 2026-05-08
content_hash: b377ad04b4620c86
---

# OmniFC: Rethinking Federated Clustering via Lossless and Secure Distance Reconstruction

**Conference**: NeurIPS 2025
**arXiv**: [2505.13071](https://arxiv.org/abs/2505.13071)  
**Code**: To be released  
**Area**: AI Safety / Federated Learning
**Keywords**: Federated Clustering, Lagrange Coded Computing, Distance Matrix Reconstruction, Lossless Aggregation, Secure Computation

## TL;DR
OmniFC is proposed as a model-agnostic federated clustering framework that exactly reconstructs the global pairwise distance matrix over a finite field via Lagrange coded computing, enabling any centralized clustering method (K-Means / Spectral Clustering / DBSCAN / Hierarchical Clustering, etc.) to run directly on the reconstructed matrix. The framework requires only a single communication round, is inherently robust to Non-IID data, and comprehensively outperforms specialized methods such as k-FED, MUFC, and FedSC across 7 datasets.

## Background & Motivation

**Background**: Federated Clustering (FC) enables distributed clients to discover global cluster structures without sharing raw data. Existing methods design algorithm-specific proxy aggregation schemes—federated K-Means aggregates centroids, federated spectral clustering aggregates low-rank factors of kernel matrices, and federated NMF aggregates basis matrices.

**Limitations of Prior Work**: (a) Proxy aggregation is computed from biased client data, inevitably introducing information loss under Non-IID settings and causing significant degradation in clustering performance—e.g., k-FED achieves a Kappa of only 0.43 on COIL-100 at $p=1.0$; (b) each method supports only a single clustering algorithm and inherits its inductive biases (e.g., compactness assumptions in KM, connectivity assumptions in SC), precluding applicability to arbitrary clustering methods.

**Key Challenge**: Existing FC methods represent "one-to-one" extensions (centralized → federated); supporting each additional clustering algorithm requires redesigning an entirely new aggregation scheme, making the paradigm fundamentally unscalable.

**Goal**: To construct a unified, model-agnostic framework that allows any centralized clustering method to be directly applied in federated settings while guaranteeing privacy and robustness to Non-IID data.

**Key Insight**: The key insight is that clustering fundamentally requires only the pairwise distance matrix $D \in \mathbb{R}^{n \times n}$ among samples. If the global distance matrix can be exactly reconstructed without leaking raw data, any centralized clustering algorithm can be directly applied to $D$.

**Core Idea**: Lagrange coded computing is employed to achieve lossless and secure reconstruction of the global pairwise distance matrix, decoupling FC from "customizing aggregation per algorithm" to "reconstructing the distance matrix once and adapting to all algorithms."

## Method

### Overall Architecture
OmniFC consists of three steps: (1) **Local Lagrange-Encoded Sharing**: each client encodes its local data using Lagrange polynomials and distributes the encodings to all peers; (2) **Global Distance Reconstruction**: each client computes pairwise distances on the encoded data and sends results to the server, which exactly reconstructs the global distance matrix via Lagrange interpolation; (3) **Cluster Assignment**: any centralized clustering algorithm is run directly on the reconstructed distance matrix. Only a single round of communication is required.

### Key Designs

1. **Lagrange Encoded Sharing**:

    - **Function**: Each sample $\tilde{x}_i$ is converted from the real domain to the finite field $\mathbb{F}_p^d$, split into $l$ segments, augmented with $t$ random noise segments, and encoded by a Lagrange interpolation polynomial $f_{\tilde{x}_i}(\alpha)$ of degree $l+t-1$. Evaluations at $m$ pre-agreed public points $\{\beta_j\}$ yield the encoded representations $z_{i,j}$, which are distributed to the $j$-th client.
    - **Mechanism**: Each client $j$ ultimately holds the encoded representations of all $n$ samples evaluated at $\beta_j$, i.e., $Z_j \in \mathbb{F}_p^{n \times d/l}$—a "globally encoded dataset" from which no single client or coalition of $\leq t$ colluding clients can recover the original data (information-theoretic security).
    - **Design Motivation**: The $t$ noise segments ensure $t$-private security. The quantization parameter $q$ controls the precision loss in the real-to-finite-field mapping.

2. **Global Distance Reconstruction**:

    - **Function**: Each client computes the squared L2 distances between all sample pairs on its encoded dataset, $dis(z_{i,j}, z_{i',j}) = \|z_{i,j} - z_{i',j}\|_2^2$, and sends the results to the server.
    - **Mechanism**: Theorem 1 guarantees that when $m \geq 2l + 2t - 1$, the server can exactly recover the true distances $dis(\tilde{x}_i, \tilde{x}_{i'}) = \sum_{o=1}^l f_{\tilde{x}_i, \tilde{x}_{i'}}(\alpha_o)$ from the $m$ clients' encoded distances via Lagrange interpolation, **regardless of the data distribution** (inherently Non-IID robust).
    - **Design Motivation**: The distance matrix is a "sufficiently abstract" global structure—it is entirely determined by inter-sample relationships and is unaffected by distributional shifts across clients.

3. **Cluster Assignment**:

    - **Function**: Any clustering algorithm is run directly on the reconstructed global distance matrix $D$.
    - For distance-dependent algorithms (SC, DBSCAN, HC, K-Medoids): $D$ is used directly, achieving **lossless federated extension**.
    - For non-distance-dependent algorithms (KM, FCM, NMF): row vectors of $D$ serve as surrogate sample features, still enabling effective clustering.
    - **Design Motivation**: A single reconstruction benefits multiple algorithms—yielding 7 variants: OmniFC-SC, OmniFC-KM, OmniFC-DBSCAN, OmniFC-HC, and others.

### Theoretical Guarantees
- **Theorem 1 (Lossless Reconstruction)**: When $m \geq 2l + 2t - 1$, distances can be exactly recovered regardless of the data distribution.
- **Theorem 2 (Privacy Security)**: Under the same condition, $I(\tilde{x}_i; \{z_{i,j}\}_{j \in C}) = 0$ for $|C| \leq t$—the mutual information between raw data and the views of any coalition of $\leq t$ colluding clients is zero, guaranteeing information-theoretic security.

## Key Experimental Results

### Main Results (Table 1, Kappa↑)

| Dataset | FedSC | OmniFC-SC | k-FED | MUFC | OmniFC-KM | FFCM | OmniFC-FCM |
|--------|-------|-----------|-------|------|-----------|------|-----------|
| Iris (p=0) | 0.95 | **0.95** | 0.38 | 0.83 | **0.95** | 0.96 | **0.95** |
| Iris (p=1) | 0.31 | **0.95** | 0.71 | 0.77 | **0.95** | 0.97 | **0.95** |
| COIL-100 (p=0) | 0.34 | **0.54** | 0.48 | 0.49 | **0.56** | 0.37 | **0.53** |
| COIL-100 (p=1) | 0.27 | **0.54** | 0.43 | 0.52 | **0.56** | 0.51 | **0.53** |
| 10x_73k (p=0) | 0.52 | **0.89** | 0.40 | 0.63 | **0.56** | 0.46 | **0.55** |
| 10x_73k (p=1) | 0.24 | **0.89** | 0.30 | 0.79 | **0.56** | 0.64 | **0.55** |

Win counts across 7 datasets × 5 Non-IID levels × 4 algorithm families: OmniFC-SC 27/35, OmniFC-KM 24/35, OmniFC-NMF 30/35.

### Ablation Study (Table 2, Generality Validation)

| Dataset | KMed_central | OmniFC-KMed | DBSCAN_central | OmniFC-DBSCAN | HC_central | OmniFC-HC |
|--------|-------------|-------------|---------------|--------------|-----------|----------|
| Iris | 0.94 | **0.94** | 0.50 | **0.50** | 0.94 | **0.95** |
| MNIST | 0.31 | **0.30** | 0.21 | **0.21** | 0.41 | **0.41** |
| COIL-20 | 0.41 | **0.42** | 0.58 | **0.58** | 0.45 | **0.45** |
| Pendigits | 0.44 | **0.45** | 0.48 | **0.48** | 0.57 | **0.57** |

For distance-dependent algorithms (KMed/DBSCAN/HC), OmniFC achieves Kappa values fully consistent with the centralized baseline—demonstrating truly **lossless federated extension**.

### Key Findings
- **OmniFC is completely immune to Non-IID**: As $p$ varies from 0 to 1, FedSC's Kappa on 10x_73k collapses from 0.52 to 0.24, whereas OmniFC-SC consistently maintains 0.89—because distance reconstruction is unaffected by data distribution.
- **Model-agnosticism is empirically validated**: All 7 clustering algorithms (KM/SC/FCM/NMF/DBSCAN/HC/KMed) integrate seamlessly into OmniFC without any algorithm-specific modification.
- **Distance matrix serves as an effective feature surrogate**: For algorithms that do not directly depend on distances (KM/FCM/NMF), replacing raw features with distance matrix row vectors typically matches and occasionally exceeds centralized performance.
- **Hyperparameter sensitivity**: As long as $m \geq 2l + 2t - 1$, the RMSE is zero (theoretically exact); increasing $l$ substantially reduces client-side computational overhead (4352s for 70K samples at $l=8$).
- **Privacy and accuracy are not in conflict**: Increasing $t$ enhances the privacy protection level, but as long as $m$ satisfies the constraint, reconstruction accuracy is unaffected.

## Highlights & Insights
- **"Reconstructing the distance matrix rather than proxy aggregation" is an elegant abstraction elevation**: It transforms federated clustering from "designing a custom aggregation scheme per algorithm" to "reconstructing the global distance matrix once." This idea is broadly generalizable—any federated task relying on pairwise relationships (e.g., federated graph learning) may benefit.
- **Dual functionality of Lagrange coded computing**: The same encoding scheme simultaneously achieves lossless reconstruction and information-theoretic security, with both requirements sharing an identical constraint ($m \geq 2l + 2t - 1$), yielding no accuracy–privacy trade-off.
- **One-shot communication**: Unlike SecFC, which requires multiple communication rounds, OmniFC requires only one—making it more practical in settings such as model marketplaces.

## Limitations & Future Work
- **Scalability of the $O(n^2)$ distance matrix**: The 10x_73k dataset (70K samples) requires 4352s and a $70K \times 70K$ matrix; application to larger datasets is infeasible. Approximate distance reconstruction or hierarchical clustering strategies could be explored.
- **Finite field quantization loss**: The real-to-finite-field mapping is controlled by $q$; precision may be insufficient for high-dimensional, high-accuracy scenarios.
- **Client count constraint $m \geq 2l + 2t - 1$**: When $l$ and $t$ are large, a sufficiently large number of clients is required, limiting applicability to few-client settings.
- **Limited gains on MNIST/Fashion-MNIST**: OmniFC-SC achieves only 0.55 on MNIST, without significantly surpassing FedSC's 0.53–0.58, suggesting that the distance matrix may not be the optimal feature representation for high-dimensional image data.

## Related Work & Insights
- **vs. k-FED**: Supports only K-Means and exhibits poor Non-IID robustness (Iris $p=1$: 0.71 vs. OmniFC 0.95).
- **vs. SecFC**: Also employs Lagrange coded computing but supports only K-Means and requires multiple communication rounds; OmniFC is model-agnostic and one-shot.
- **vs. FedSC**: Supports only spectral clustering and degrades sharply under Non-IID (10x_73k $p=1$: 0.24 vs. OmniFC 0.89).

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The unified abstraction of "reconstructing the distance matrix," combined with theoretical guarantees of losslessness and security, is an elegant contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Coverage across 7 datasets × 5 Non-IID levels × 7 clustering algorithms is comprehensive; large-scale dataset experiments are lacking.
- **Writing Quality**: ⭐⭐⭐⭐ Theoretical derivations are clear, with complete theorems and proofs.
- **Value**: ⭐⭐⭐⭐ Establishes a new paradigm for federated clustering; the distance matrix idea is generalizable to federated graph learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Reconstruction and Secrecy under Approximate Distance Queries](reconstruction_and_secrecy_under_approximate_distance_queries.md)
- [\[NeurIPS 2025\] Unifying Proportional Fairness in Centroid and Non-Centroid Clustering](unifying_proportional_fairness_in_centroid_and_non-centroid_clustering.md)
- [\[NeurIPS 2025\] Unifying Re-Identification, Attribute Inference, and Data Reconstruction Risks in Differential Privacy](unifying_re-identification_attribute_inference_and_data_reconstruction_risks_in_.md)
- [\[ICML 2025\] SecEmb: Sparsity-Aware Secure Federated Learning of On-Device Recommender System with Large Embedding](../../ICML2025/ai_safety/secemb_sparsity-aware_secure_federated_learning_of_on-device_recommender_system_.md)
- [\[ICCV 2025\] Backdoor Mitigation by Distance-Driven Detoxification](../../ICCV2025/ai_safety/backdoor_mitigation_by_distance-driven_detoxification.md)

</div>

<!-- RELATED:END -->

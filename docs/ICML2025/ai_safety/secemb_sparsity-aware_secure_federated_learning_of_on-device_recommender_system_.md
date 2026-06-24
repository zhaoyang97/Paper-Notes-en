---
title: >-
  [Paper Note] SecEmb: Sparsity-Aware Secure Federated Learning of On-Device Recommender System with Large Embedding
description: >-
  [ICML2025][AI Safety][Federated Learning] Proposed SecEmb, a lossless secure federated recommendation protocol exploiting the sparsity of embedding updates. By using Function Secret Sharing (FSS), it protects the privacy of user-rated item indices and gradients while reducing upload/download communication overhead by up to 90x and user-side computation time by up to 70x.
tags:
  - "ICML2025"
  - "AI Safety"
  - "Federated Learning"
  - "Recommender Systems"
  - "Secure Aggregation"
  - "Sparse Embeddings"
  - "Function Secret Sharing"
  - "Privacy Protection"
date: 2026-05-08
content_hash: f69de82040a3423a
---

# SecEmb: Sparsity-Aware Secure Federated Learning of On-Device Recommender System with Large Embedding

**Conference**: ICML2025  
**arXiv**: [2505.12453](https://arxiv.org/abs/2505.12453)  
**Code**: [NusIoraPrivacy/SecEmb](https://github.com/NusIoraPrivacy/SecEmb)  
**Area**: AI Security / Federated Recommendation  
**Keywords**: Federated Learning, Recommender Systems, Secure Aggregation, Sparse Embeddings, Function Secret Sharing, Privacy Protection

## TL;DR

Proposed SecEmb, a lossless secure federated recommendation protocol exploiting the sparsity of embedding updates. By using Function Secret Sharing (FSS), it protects the privacy of user-rated item indices and gradients while reducing upload/download communication overhead by up to 90x and user-side computation time by up to 70x.

## Background & Motivation

Federated recommender systems (FedRec) protect user data through distributed training, but face the following Key Challenges:

- **Communication Bottleneck**: Traditional FedRec requires uploading/downloading the complete item embedding matrix $Q \in \mathbb{R}^{m \times d}$. The communication volume grows linearly with the number of items $m$, posing heavy pressure on bandwidth-constrained edge devices.
- **Privacy Leakage Risk**: Existing sparsity-aware federated protocols (e.g., FMSS, SparseSecAgg) reduce communication volume but inevitably expose the coordinates of non-zero updates (i.e., which items the user has rated). In recommendation scenarios, these coordinates directly reveal user behavioral privacy.
- **Key Sparsity Opportunity**: In practice, a user only interacts with a tiny fraction of items (e.g., in ML10M, 90% of users rated $\le 300$ items, whereas the total number of items is 27,278), rendering the embedding gradient matrix highly sparse.

Core Problem: Can we simultaneously achieve (1) communication/computation costs depending only on the number of user-interacted items $m'$ (rather than the total number of items $m$), and (2) zero knowledge for the server regarding individual user rating indices and gradient values (only obtaining the aggregated results)?

## Method

SecEmb consists of two collaborative modules, both built upon **Function Secret Sharing (FSS)**:

### Module 1: Privacy-Preserving Embedding Retrieval

Users only need to download the embeddings of their interacted items instead of the entire embedding matrix, while hiding the target item indices from the server.

**Mechanism**: Encode each rated item as a point function and utilize FSS to achieve private retrieval from two servers.

1. User $u$ encodes each rated item $i$ as a point function $f_{u,i}: \mathcal{I} \to \mathbb{G}$, which outputs 1 when $x = i$, and 0 otherwise.
2. Generate key pair $(reK_{u,i}^0, reK_{u,i}^1)$ using FSS, and send them to the two servers respectively.
3. Each server $s$ computes the secret shares:

$$\mathbf{v}_{u,i}^s = \sum_j \text{FSS.Eval}(reK_{u,i}^s, j) \cdot Q_j$$

4. Upon receiving both shares, the user reconstructs the target embedding: $Q_{\text{idx}(i)} = \mathbf{v}_{u,i}^0 + \mathbf{v}_{u,i}^1$.

### Module 2: Secure Gradient Aggregation

Exploit the row-sparse structure of the sparse gradient matrix, encoding non-zero gradient rows as point functions for secure aggregation.

**Row-Level Encoding Optimization**: Entirely encode each non-zero row as a single point function (rather than element-wise encoding), reducing the number of keys from $m'd$ to $m'$:

$$f_{u,i}(x) = \begin{cases} \mathbf{g}_{Q_i^u} \in \mathbb{R}^d & \text{if } x = \text{idx}(i) \\ \mathbf{0} \in \mathbb{R}^d & \text{otherwise} \end{cases}$$

Server aggregation: $\mathbf{v}_{Q_j}^s = \sum_u \sum_{i \in [m']} \text{FSS.Eval}(agK_{u,i}^s, j)$

### Path Sharing Optimization

Key Observation: The FSS keys in both the retrieval and aggregation phases target the same item indices, meaning the first half of the keys (the seeds and correction words identifying non-zero positions) can be completely reused. In the aggregation phase, users only need to generate and transmit the final correction word $CW^{n+1}$, saving $n$ AES operations.

### Complexity Analysis

| Metric | SecEmb | Traditional Secure FedRec |
|------|--------|----------------|
| Download Communication | $O(m'bd + |\theta|b)$ | $O(mbd + |\theta|b)$ |
| Upload Communication | $O(m'(\lambda \log m + bd) + |\theta|b)$ | $O(mbd + |\theta|b)$ |
| User Computation | $O(m' \log m \cdot \text{AES} + |\theta|)$ | $O(md + |\theta|)$ |

Where $m' \ll m$ represents the number of user-interacted items, and $\lambda$ is the security parameter. The communication volume is reduced from $O(m)$ to $O(m' \log m)$.

## Key Experimental Results

### Experimental Setup
- **Datasets**: ML100K, ML1M, ML10M, ML25M, Yelp (number of items ranges from 1,682 to 93,386)
- **Models**: MF, NCF, FM, DeepFM
- **Baselines**: SecFedRec (two-server ASS), SVD, CoLR, 8-bit quantization, ternary quantization

### Communication Overhead Reduction (Upload, per user per round, MB)

| Dataset | Number of Items | SecFedRec | SecEmb | Reduction Factor |
|--------|--------|-----------|--------|---------|
| ML100K | 1,682 | 0.86 | 0.17 | **5.0x** |
| ML1M | 3,883 | 1.99 | 0.27 | **7.4x** |
| ML10M | 10,681 | 5.47 | 0.28 | **19.2x** |
| ML25M | 62,423 | 31.96 | 0.51 | **62.1x** |
| Yelp | 93,386 | 47.81 | 0.52 | **91.2x** |

(Taking the MF model as an example, FM shows a similar trend)

### Lossless vs Lossy Compression (MF, RMSE↓, lower is better)

| Method | ML25M RMSE | ML25M Compression Ratio | Yelp RMSE | Yelp Compression Ratio |
|------|-----------|-------------|----------|------------|
| 8-bit Quantization | 0.870 | 4.0x | 1.050 | 4.0x |
| Ternary Quantization | 0.874 | 8.0x | 1.050 | 8.0x |
| SVD | 0.873 | 16.2x | 1.050 | 16.2x |
| CoLR | 0.873 | 16.3x | **1.049** | 16.3x |
| **SecEmb** | **0.864** | **62.1x** | **1.050** | **91.2x** |

SecEmb significantly outperforms lossy methods in communication compression ratio on large-scale datasets, with lossless or even slightly better accuracy (due to no information loss).

### Computation Time

The user-side key generation time achieves approximately **70x** speedup on the Yelp dataset (MF/FM), with the advantage becoming more pronounced as the number of items increases.

## Highlights & Insights

1. **Elegant Design of Lossless Compression**: Lossless compression of sparse updates into compact keys via FSS, requiring no quantization or dimensionality reduction, resulting in zero accuracy loss.
2. **Row-Level Encoding + Path Sharing**: Two key optimizations reduce the number of user-uploaded keys from $m'd$ to $m'$, and reuse the AES computation path from the retrieval stage.
3. **Comprehensive Privacy Guarantees**: Under the semi-honest two-server model, the servers cannot learn which items the user has rated (index privacy) or the specific gradient values, obtaining only the aggregated result. The formal security proof is based on the security of FSS.
4. **Composable DP**: The protocol can be seamlessly integrated with Differential Privacy (DP). Implementing $(\epsilon, \delta)$-DP is achieved by having each server independently add noise to its aggregated shares.
5. **Practical Gains Amplify with Scale**: The larger the number of items, the more prominent the sparsity advantage becomes (e.g., Yelp 91x vs ML100K 5x).

## Limitations & Future Work

1. **Two-Server Non-Collusion Assumption**: Security relies on the assumption that the two servers do not collude, which might be hard to guarantee in practical deployments. Relaxing this to a single server or an active/malicious adversary model would require substantial modifications to the protocol.
2. **Increased Server-Side Computational Overhead**: FSS evaluation requires servers to perform AES operations for all items, resulting in a computational complexity of $O(nm'm \cdot \text{AES})$. This could become a bottleneck when both the number of users and items are extremely large.
3. **Optimization Limited to the Embedding Layer**: For models like DeepFM that contain a large number of dense parameters, the advantage diminishes when the embedding layer's proportion is relatively small (e.g., only 1.05x on ML100K for DeepFM).
4. **Efficiency Loss from Uniforming $m'$**: To hide the actual number of interactions, all users must be padded/aligned to the same $m'$, introducing unnecessary padding overhead for inactive users.
5. **Exclusion of Malicious User Scenarios**: Robustness against Byzantine users submitting malicious gradients is not studied or evaluated.

## Related Work & Insights

- **FedMF / FedNCF**: Early secure federated recommendation methods utilizing homomorphic encryption or generic SecAgg, where communication volume scales linearly with the number of items.
- **LightSecAgg / FastSecAgg**: Optimizations in computational complexity of SecAgg, but they do not exploit gradient sparsity.
- **FMSS**: A sparsity-aware federated recommendation framework, which however exposes non-zero coordinates (i.e., rated item indices).
- **FSS (Boyle et al., 2015/2016)**: The theoretical foundation of Function Secret Sharing. This work is the first to apply it to the sparse embedding scenario in federated recommendation.
- **Insights**: The point function encoding idea of FSS can be generalized to other federated scenarios with structured sparsity (e.g., word embeddings in NLP, node embeddings in Graph Neural Networks).

## Rating

- Novelty: ⭐⭐⭐⭐ — The application of FSS in federated recommendation is highly original, with refined designs in row-level encoding and path sharing optimization.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive evaluation across 5 datasets and 4 models, covering communication, computation, and accuracy, with robust comparisons.
- Writing Quality: ⭐⭐⭐⭐ — Clearly defined problem. The progressive structure migrating from initial construction to optimized versions is easy to follow.
- Value: ⭐⭐⭐⭐ — Resolves the core contradiction between efficiency and privacy in secure federated recommendation, holding significant practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Secure Outlier-Aware Large Language Model Inference](../../ICLR2026/ai_safety/secure_outlier-aware_large_language_model_inference.md)
- [\[NeurIPS 2025\] MARS: A Malignity-Aware Backdoor Defense in Federated Learning](../../NeurIPS2025/ai_safety/mars_a_malignity-aware_backdoor_defense_in_federated_learning.md)
- [\[ICCV 2025\] Client2Vec: Improving Federated Learning by Distribution Shifts Aware Client Indexing](../../ICCV2025/ai_safety/client2vec_improving_federated_learning_by_distribution_shifts_aware_client_inde.md)
- [\[ICML 2025\] Towards Trustworthy Federated Learning with Untrusted Participants](towards_trustworthy_federated_learning_with_untrusted_participants.md)
- [\[NeurIPS 2025\] Environment Inference for Learning Generalizable Dynamical System](../../NeurIPS2025/ai_safety/environment_inference_for_learning_generalizable_dynamical_system.md)

</div>

<!-- RELATED:END -->

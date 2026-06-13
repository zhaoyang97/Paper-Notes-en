---
title: >-
  [Paper Note] FedHPro: Federated Hyper-Prototype Learning via Gradient Matching
description: >-
  [ICML 2026][AI Safety][Federated Learning] To address the issue where direct averaging of local prototypes inherits client bias in prototype-based Federated Learning (FL)…
tags:
  - "ICML 2026"
  - "AI Safety"
  - "Federated Learning"
  - "Data Heterogeneity"
  - "Hyper-Prototype"
  - "Gradient Matching"
  - "Contrastive Learning"
date: 2026-05-08
content_hash: d1d87b7974e26840
---

# FedHPro: Federated Hyper-Prototype Learning via Gradient Matching

**Conference**: ICML 2026  
**arXiv**: [2605.13475](https://arxiv.org/abs/2605.13475)  
**Code**: https://github.com/mala-lab/FedHPro (Available)  
**Area**: Federated Learning / Privacy Protection / Prototype Learning  
**Keywords**: Federated Learning, Data Heterogeneity, Hyper-Prototype, Gradient Matching, Contrastive Learning

## TL;DR
To address the issue where direct averaging of local prototypes inherits client bias in prototype-based Federated Learning (FL), this paper introduces a set of learnable global hyper-prototypes. These are optimized via gradient matching on the server side to simulate prototypes obtained through centralized training. Combined with client-side contrastive learning and alignment losses, this approach significantly improves accuracy in heterogeneous scenarios.

## Background & Motivation

**Background**: Federated Learning (FL) enables collaborative training by sharing models instead of data, but non-IID data remains a core challenge. Prototype-based methods (FedProto, FedTGP, FedSA) have become mainstream for mitigating heterogeneity by transmitting class-wise feature means as "semantic anchors" to align representation spaces across clients.

**Limitations of Prior Work**: Existing prototype-based methods follow a two-step "calculate local prototypes, then average on the server" route, which is essentially *prototype-level* aggregation. This approach directly transfers client bias to global signals—while class separation may be clear in simple domains (MNIST), class clusters overlap severely in difficult domains (SVHN/SYN), leading to the clustering of boundary samples.

**Key Challenge**: The ideal approach would be to train a unified representation space using real samples from all clients and then calculate prototypes, which is forbidden by FL privacy constraints. Consequently, global prototypes can only "indirectly" reflect biased client data; whether through averaging or refining semantic anchors, one cannot escape the "client bias $\to$ global prototype bias" cycle.

**Goal**: To decompose the problem into: (a) how the server constructs a global signal that approximates a "centralized training prototype"; and (b) how clients use this signal to simultaneously pull intra-class samples closer and push inter-class samples apart.

**Key Insight**: The authors observe an interesting fact: although the server cannot access raw samples, it can access the gradient $\mathbf{g}_k^c$ of the prototype relative to the real samples on each client. By initializing a set of learnable vectors on the server and matching their gradient directions (generated under a virtual classification loss) to $\mathbf{g}_k^c$, these vectors follow the optimization trajectory that raw samples would have provided, thus indirectly capturing the semantics of real data.

**Core Idea**: Replace "statistical averaging" of global prototypes with "hyper-prototypes learned via gradient matching," and use them to drive two complementary objectives: inter-class contrast and intra-class alignment.

## Method

### Overall Architecture
FedHPro adds two signal paths to the standard FedAvg process: (1) an uplink where an additional average gradient $\mathbf{g}_k^c$ per class per client is transmitted; (2) a server-side learnable hyper-prototype tensor $\mathcal{S}_M \in \mathbb{R}^{\mathbb{C}\times|\mathcal{I}|\times d}$ (with $|\mathcal{I}|=5$ vectors per class) optimized via a gradient matching loss $\mathcal{L}_{GM}$ for $M=30$ rounds; (3) a downlink where, in addition to the global model, $\mathcal{S}_M$ is broadcast to clients, who then construct two additional losses: HPCL and HPAL.

### Key Designs

1. **Hyper-Prototypes via Gradient Matching**:

    - **Function**: Approximates prototypes on the server side that would otherwise require centralized training.
    - **Mechanism**: The true gradient $\mathbf{g}_k^c = \tfrac{1}{n_k^c}\sum \nabla_{z_i}\mathcal{L}_k(x_i,y_i)$ from each client $k$ for class $c$ is aggregated to obtain $\mathbf{g}^c$. The server initializes $\{\mathbf{s}_i^c\}_{i=1}^{|\mathcal{I}|}$ and generates hyper-prototype gradients $\mathbf{g}_{HP}^c$ under a virtual cross-entropy loss $\mathcal{L}_{vir}$, then minimizes $\mathcal{L}_{GM}=1-\cos(\mathbf{g}^c,\mathbf{g}_{HP}^c)$. This allows hyper-prototypes to evolve in the direction real samples would push them, bypassing the "no raw data" constraint.
    - **Design Motivation**: Previous prototype-level aggregation methods yield global prototypes that inherit statistical biases. Gradient matching serves as a proxy for *sample-level* signals, more faithfully capturing class semantics. Experiments on Digits show the $L_2$ distance from hyper-prototypes to centralized prototypes is significantly smaller than that of FedAvg/FedProto global prototypes.

2. **Hyper-Prototype Contrastive Learning (HPCL) + Client Adaptive Margin**:

    - **Function**: Uses hyper-prototypes on the client side to construct a contrastive objective that pulls samples toward positive hyper-prototypes and pushes them away from negative ones, enhancing inter-class separation.
    - **Mechanism**: First, the average Euclidean distance between all local prototype pairs $d_k = \tfrac{1}{(\mathbb{C}-1)^2}\sum D_{L_2}(\mathbf{p}_k^{c_1},\mathbf{p}_k^{c_2})$ is calculated as a client-specific margin. The similarity $s(z_i,\mathcal{S}_M^c)$ between sample embedding $z_i$ and the hyper-prototype set $\mathcal{S}_M^c$ is defined as the average cosine similarity across all $|\mathcal{I}|$ vectors. Finally, $\mathcal{L}_{HPCL}=\log(1+\sum_{\mathcal{S}_M^j\in\mathcal{N}_M^c}\exp((s(z_i,\mathcal{S}_M^j)+d_k)/\tau)/\exp(s(z_i,\mathcal{S}_M^c)/\tau))$ pulls each embedding toward multiple hyper-prototypes of its positive class and pushes away all negative classes.
    - **Design Motivation**: Fixed margins are unfair across clients—classes are naturally separated for clients with uniform data, while clusters are crowded for long-tailed clients. $d_k$ adapts to the scale of the client's representation space, matching the intensity of decision boundary sharpening to the "difficulty" of the client's data.

3. **Hyper-Prototype Alignment Learning (HPAL) (Huber-style penalty)**:

    - **Function**: Smoothly narrows the gap between samples and the average hyper-prototype $\mathcal{H}_M^c$ at the embedding-feature level, improving intra-class compactness and cross-client consistency.
    - **Mechanism**: A Huber loss is applied to each dimension of the embedding: $\tfrac{1}{2}(z_{i(q)}-\mathcal{H}_{M(q)}^c)^2$ if the absolute difference $\le 1$, and $|z_{i(q)}-\mathcal{H}_{M(q)}^c|-\tfrac{1}{2}$ otherwise. The total training loss is $\mathcal{L}=\mathcal{L}_{CE}+\mathcal{L}_{HPCL}+\mathcal{L}_{HPAL}$.
    - **Design Motivation**: Pure $L_2$ alignment is overly sensitive to outliers, which can destabilize client representations during early training. Huber loss acts like $L_2$ for small residuals and $L_1$ for large ones, fitting the need for "coarse alignment early, fine convergence late."

### Loss & Training
The server optimizes hyper-prototypes using $\mathcal{L}_{GM}$ ($M=30$ inner rounds). The total objective for clients is $\mathcal{L}=\mathcal{L}_{CE}+\mathcal{L}_{HPCL}+\mathcal{L}_{HPAL}$. Hyperparameters include temperature $\tau=0.05$, hyper-prototype set size $|\mathcal{I}|=5$, 100 communication rounds, 10 local epochs, and SGD learning rate 0.01. The paper also proves a non-convex convergence rate of $R>\Theta\big(\tfrac{\mathcal{L}_0-\min\mathcal{L}^*}{E\eta\,\boldsymbol{\varepsilon}}\big)$.

## Key Experimental Results

### Main Results
Covering label skew, quantity skew, and domain skew across 9 datasets with 8 SOTA baselines.

| Dataset/Scenario | Metric | Ours | Prev. SOTA | Gain |
| :--- | :--- | :--- | :--- | :--- |
| CIFAR10 NID1$_{0.5}$ | Acc | 89.56 | 88.09 (FedGMKD) | +1.47 |
| CIFAR10-LT $\rho=100$ | Acc | 64.75 | 62.48 (FedSA) | +2.27 |
| Office-Caltech Avg | Acc | 64.52 | 60.57 (FedSA) | +3.95 |
| TinyImageNet NID2 | Acc | 40.52 | 38.76 (FedRCL) | +1.76 |

### Ablation Study

| Configuration | Digits Avg | Office-Caltech Avg |
| :--- | :--- | :--- |
| FedAvg (Baseline) | 78.82 | 55.42 |
| HPCL only | 83.58 | 60.92 |
| HPAL only | 83.94 | 61.34 |
| HPCL+HPAL (Full) | **84.80** | **64.52** |
| Replace Hyper-Pros with Global Pros $\mathbb{P}$ (HPCL+HPAL) | 81.35 | 59.69 |

### Key Findings
- The pluggability of hyper-prototypes is highly valuable: replacing global prototypes in FedProto, FedTGP, FedGMKD, or FedSA with the proposed hyper-prototypes yields consistent gains (e.g., +1.2-3.2 in the SYN domain), proving the independent value of the gradient matching approach.
- HPCL and HPAL are nearly complementary; individually they provide a 4-5 point boost, and combined they add another point, indicating that "separation" and "compactness" require specialized objectives.
- $|\mathcal{I}|=5, M=30$ is the optimal trade-off: larger $|\mathcal{I}|$ leads to unstable optimization, while $M>30$ offers negligible marginal gains.

## Highlights & Insights
- **Gradients as a Privacy-Safe "Knowledge Interface"**: The author translates the "FL cannot transmit raw samples" constraint into "can we transmit gradients?", using gradient matching to turn the server into an emulator. This clever paradigm shift grants the server "pseudo-centralized" training capabilities.
- **Client Adaptive Margin as a General Utility**: Transforming the margin from a hyperparameter into a function of the client representation scale ($d_k$) is a robust trick for handling heterogeneous clients, applicable to any metric-learning-based FL method.
- **Multiple Vectors per Class ($|\mathcal{I}|=5$)**: Using multiple vectors for hyper-prototypes instead of a single one explicitly assigns "semantic sub-patterns" of each class to different vectors, alleviating the fundamental bottleneck of single-prototype descriptive power.

## Limitations & Future Work
- Uplink communication includes class-aggregated gradients $\mathbf{g}_k\in\mathbb{R}^{\mathbb{C}\times d}$, which incurs an extra overhead of ~0.4 MB when class counts are high (e.g., TinyImageNet with 200 classes).
- Using gradients as transmission objects might carry higher privacy leakage risks than prototypes; the paper does not discuss performance under Differential Privacy (DP) or encryption.
- $\mathcal{L}_{GM}$ utilizes cosine similarity rather than $L_2$, matching direction but not magnitude. The paper lacks a detailed analysis of how magnitude differences between majority and minority classes affect the relative scale of hyper-prototypes.
- While experimental results for model heterogeneity are in Table A9, HPCL/HPAL still rely on a shared feature dimension $d$. Aligning feature spaces across architectures (e.g., ViT vs. ConvNet) remains an open problem.

## Related Work & Insights
- **vs FedProto (AAAI'22)**: Both use prototypes as global signals, but FedProto directly averages local prototypes. FedHPro uses gradient matching to learn hyper-prototypes, bypassing the "local bias $\to$ global bias" chain and achieving a 9-point improvement in difficult domains (SYN).
- **vs FedSA (AAAI'25)**: FedSA also aims to "refine global anchors" but remains at the prototype-level; FedHPro is updated at the *sample-level* (via gradients), making its structure closer to centralized training.
- **vs FedTGP (AAAI'24)**: FedTGP introduces trainable global prototypes and adaptive-margin contrastive learning. FedHPro moves "trainability" from the prototype level down to the "gradient matching" level and proposes a more granular client-side margin $d_k$ compared to FedTGP's global margin.
- **Insight**: Gradient matching, originally a tool for Dataset Condensation, works in FL because any scenario where raw samples are inaccessible but gradients are available can leverage this approach (e.g., cross-institution GNN training, cross-device RecSys).

## Rating
- **Novelty**: ⭐⭐⭐⭐ Introducing gradient matching into FL prototype methods is a fresh perspective, though gradient matching and contrastive learning are established tools.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive evaluation across 8 datasets, 3 heterogeneity scenarios, and 8 baselines, including appendix experiments on model heterogeneity, text modalities, and fairness.
- **Writing Quality**: ⭐⭐⭐⭐ Figures 1-2 clarify motivation effectively; Section 4 is somewhat lengthy, and the HPCL formula could be simplified.
- **Value**: ⭐⭐⭐⭐ The pluggable nature of hyper-prototypes for any prototype-based FL method offers direct value for practical FL system deployments.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] FedDAP: Domain-Aware Prototype Learning for Federated Learning under Domain Shift](../../CVPR2026/ai_safety/feddap_domain-aware_prototype_learning_for_federated_learning_under_domain_shift.md)
- [\[ICML 2026\] Frequency Matching in Spiking Neural Networks for mmWave Sensing](frequency_matching_in_spiking_neural_networks_for_mmwave_sensing.md)
- [\[CVPR 2026\] ProxyFL: A Proxy-Guided Framework for Federated Semi-Supervised Learning](../../CVPR2026/ai_safety/proxyfl_a_proxy-guided_framework_for_federated_semi-supervised_learning.md)
- [\[ICML 2026\] Flatness-Aware Stochastic Gradient Langevin Dynamics](flatness-aware_stochastic_gradient_langevin_dynamics.md)
- [\[ICML 2026\] Two Blind Spots of Machine Unlearning: Over-unlearning and Prototype Relearning Attacks](unlearnings_blind_spots_over-unlearning_and_prototypical_relearning_attack.md)

</div>

<!-- RELATED:END -->

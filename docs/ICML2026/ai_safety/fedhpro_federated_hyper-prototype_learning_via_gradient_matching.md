---
title: >-
  [Paper Note] FedHPro: Federated Hyper-Prototype Learning via Gradient Matching
description: >-
  [ICML 2026][AI Safety][Federated Learning] To address the issue in prototype-based federated learning where "directly averaging local prototypes inherits client bias…
tags:
  - "ICML 2026"
  - "AI Safety"
  - "Federated Learning"
  - "Data Heterogeneity"
  - "Hyper-Prototype"
  - "Gradient Matching"
  - "Contrastive Learning"
date: 2026-05-08
content_hash: 01237dad02fefc42
---

# FedHPro: Federated Hyper-Prototype Learning via Gradient Matching

**Conference**: ICML 2026  
**arXiv**: [2605.13475](https://arxiv.org/abs/2605.13475)  
**Code**: https://github.com/mala-lab/FedHPro (available)  
**Area**: Federated Learning / Privacy Protection / Prototype Learning  
**Keywords**: Federated Learning, Data Heterogeneity, Hyper-Prototype, Gradient Matching, Contrastive Learning

## TL;DR
To address the issue in prototype-based federated learning where "directly averaging local prototypes inherits client bias," this work introduces a set of learnable global hyper-prototypes. These are optimized on the server via gradient matching to simulate prototypes as if trained in a centralized manner. Combined with client-side contrastive and alignment losses, this approach significantly improves accuracy under heterogeneous scenarios.

## Background & Motivation

**Background**: Federated Learning (FL) enables collaborative training by sharing models instead of data, but non-IID data remains a core challenge. Prototype-based methods (FedProto, FedTGP, FedSA) align representation spaces across clients by transmitting class-wise feature means as "semantic anchors," and have become a mainstream approach to mitigating heterogeneity.

**Limitations of Prior Work**: Existing prototype-based methods follow a two-step process: compute prototypes on clients, then average them on the server—essentially *prototype-level* aggregation. This approach directly transfers client bias to the global signal. While class separation is clear in simple domains (e.g., MNIST), in more challenging domains (e.g., SVHN/SYN), class clusters overlap severely and boundary samples are misclustered.

**Key Challenge**: Ideally, one would train a unified representation space using all clients' real samples and then compute prototypes. However, FL's privacy constraints prohibit this. Thus, global prototypes can only "indirectly" reflect the bias in client data; whether by averaging or refining semantic anchors, the cycle of "client bias → global prototype bias" is hard to break.

**Goal**: Decompose the problem into (a) how the server can construct a global signal that approximates a "centralized training prototype," and (b) how clients can use this signal to simultaneously pull intra-class samples closer and push inter-class samples apart.

**Key Insight**: The authors observe that while the server cannot access real samples, it can obtain the gradient of each client's prototype with respect to real samples, $\mathbf{g}_k^c$. If the server initializes a set of learnable vectors and matches their gradients under a virtual classification loss to $\mathbf{g}_k^c$, these vectors evolve along the "optimization trajectory that real samples would induce," indirectly absorbing the semantics of real data.

**Core Idea**: Replace global prototypes from "statistical averaging" with "hyper-prototypes learned via gradient matching," and use them to drive two complementary objectives: inter-class contrastive learning and intra-class alignment.

## Method

### Overall Architecture
FedHPro augments the standard FedAvg process with two additional signal paths: (1) clients upload, for each class, the average gradient $\mathbf{g}_k^c$; (2) the server maintains a set of learnable hyper-prototype tensors $\mathcal{S}_M \in \mathbb{R}^{\mathbb{C}\times|\mathcal{I}|\times d}$ (with $|\mathcal{I}|=5$ vectors per class), optimized via gradient matching loss $\mathcal{L}_{GM}$ for $M=30$ rounds; (3) in the downlink, the server broadcasts both the global model and $\mathcal{S}_M$ to clients, who use them to construct two additional losses: HPCL and HPAL.

### Key Designs

1. **Hyper-Prototypes via Gradient Matching**:

    - **Function**: On the server, a set of learnable vectors approximates the prototype that would be computed under centralized training.
    - **Mechanism**: For each class $c$ and client $k$, the real gradient $\mathbf{g}_k^c = \tfrac{1}{n_k^c}\sum \nabla_{z_i}\mathcal{L}_k(x_i,y_i)$ is aggregated to obtain $\mathbf{g}^c$. The server initializes $\{\mathbf{s}_i^c\}_{i=1}^{|\mathcal{I}|}$, generates hyper-prototype gradients $\mathbf{g}_{HP}^c$ under a virtual cross-entropy loss $\mathcal{L}_{vir}$, and minimizes $\mathcal{L}_{GM}=1-\cos(\mathbf{g}^c,\mathbf{g}_{HP}^c)$. This essentially evolves hyper-prototypes along the direction real samples would push prototypes, circumventing the "no access to raw data" limitation.
    - **Design Motivation**: Previous methods aggregate at the prototype level, so global prototypes inevitably inherit client statistical bias. Gradient matching acts as a proxy for *sample-level* signals, better capturing class semantics. The paper shows on Digits that the L2 distance from hyper-prototypes to centralized prototypes is significantly smaller than that of FedAvg/FedProto global prototypes.

2. **Hyper-Prototype Contrastive Learning (HPCL) + Client Adaptive Margin**:

    - **Function**: On clients, hyper-prototypes are used to construct a contrastive objective that pulls positive samples toward their class hyper-prototypes and pushes them away from other classes, enhancing inter-class separation.
    - **Mechanism**: Clients first compute the average Euclidean distance between all pairs of local prototypes $d_k=\tfrac{1}{(\mathbb{C}-1)^2}\sum D_{L_2}(\mathbf{p}_k^{c_1},\mathbf{p}_k^{c_2})$ as a client-specific margin. The similarity between a sample embedding $z_i$ and a class hyper-prototype set $\mathcal{S}_M^c$ is defined as the mean cosine similarity over all $|\mathcal{I}|$ vectors. The loss is $\mathcal{L}_{HPCL}=\log(1+\sum_{\mathcal{S}_M^j\in\mathcal{N}_M^c}\exp((s(z_i,\mathcal{S}_M^j)+d_k)/\tau)/\exp(s(z_i,\mathcal{S}_M^c)/\tau))$, pulling each embedding toward multiple positive hyper-prototypes and pushing away all negatives.
    - **Design Motivation**: Fixed margins are unfair across clients—clients with balanced data have naturally separated classes, while long-tail clients have crowded clusters. $d_k$ adapts to the scale of each client's representation space, matching the sharpness of decision boundaries to the "difficulty" of the client's data.

3. **Hyper-Prototype Alignment Learning (HPAL, Huber-style penalty)**:

    - **Function**: At the embedding-feature level, smoothly pulls samples toward the mean hyper-prototype $\mathcal{H}_M^c$ of their class, improving intra-class compactness and cross-client consistency.
    - **Mechanism**: For each embedding dimension, apply Huber loss: for absolute difference $\le 1$, use $\tfrac12(z_{i(q)}-\mathcal{H}_{M(q)}^c)^2$; otherwise, $|z_{i(q)}-\mathcal{H}_{M(q)}^c|-\tfrac12$. The total training loss is $\mathcal{L}=\mathcal{L}_{CE}+\mathcal{L}_{HPCL}+\mathcal{L}_{HPAL}$.
    - **Design Motivation**: Pure L2 alignment is overly sensitive to outliers, causing instability when client representations are not yet stable. Huber loss is L2-like for small residuals and L1-like for large ones, fitting the need for "coarse alignment early, fine convergence later."

### Loss & Training
On the server, $\mathcal{L}_{GM}$ is used to optimize hyper-prototypes ($M=30$ inner rounds). On clients, the total objective is $\mathcal{L}=\mathcal{L}_{CE}+\mathcal{L}_{HPCL}+\mathcal{L}_{HPAL}$. Temperature $\tau=0.05$, hyper-prototype set size $|\mathcal{I}|=5$, 100 communication rounds per epoch, 10 local epochs, SGD learning rate 0.01. The paper also proves a non-convex convergence rate $R>\Theta\big(\tfrac{\mathcal{L}_0-\min\mathcal{L}^*}{E\eta\,\boldsymbol{\varepsilon}}\big)$.

## Key Experimental Results

### Main Results
Covers label skew, quantity skew, and domain skew, across 9 datasets and 8 SOTA baselines.

| Dataset/Scenario | Metric | Ours | Prev. SOTA | Gain |
|--------|------|------|----------|------|
| CIFAR10 NID1$_{0.5}$ | Acc | 89.56 | 88.09 (FedGMKD) | +1.47 |
| CIFAR10-LT $\rho=100$ | Acc | 64.75 | 62.48 (FedSA) | +2.27 |
| Office-Caltech Avg | Acc | 64.52 | 60.57 (FedSA) | +3.95 |
| TinyImageNet NID2 | Acc | 40.52 | 38.76 (FedRCL) | +1.76 |

### Ablation Study

| Configuration | Digits Avg | Office-Caltech Avg |
|------|---------|------|
| FedAvg (baseline) | 78.82 | 55.42 |
| HPCL only | 83.58 | 60.92 |
| HPAL only | 83.94 | 61.34 |
| HPCL+HPAL (Full) | **84.80** | **64.52** |
| Replace hyper-prototype with global prototype (HPCL+HPAL) | 81.35 | 59.69 |

### Key Findings
- The plug-and-play nature of hyper-prototypes is highly valuable: replacing the global prototype in FedProto / FedTGP / FedGMKD / FedSA with the proposed hyper-prototype consistently improves performance (by 1.2–3.2 points on SYN), demonstrating the independent value of gradient matching.
- HPCL and HPAL are nearly complementary—each alone improves by 4–5 points, and together add another point, indicating that "separation" and "compactness" require dedicated objectives.
- $|\mathcal{I}|=5,\, M=30$ is the best trade-off: larger $|\mathcal{I}|$ destabilizes optimization, and $M>30$ yields diminishing returns.

## Highlights & Insights
- **Gradients as a privacy-safe "knowledge interface"**: The authors reinterpret the FL constraint "cannot transmit raw samples" as "can gradients be transmitted instead," and use gradient matching to turn the server into a simulator—an elegant paradigm shift that enables "pseudo-centralized" training on the server.
- **Client-adaptive margin is highly transferable**: Turning the margin from a hyperparameter into a function of $d_k$, the client's representation scale, is a general trick for handling heterogeneous clients and can be applied to any metric learning-based FL method.
- **Multiple vectors per class in hyper-prototypes ($|\mathcal{I}|=5$)**: This explicitly allocates "semantic sub-modes" of each class to different vectors, addressing the fundamental limitation of single-prototype expressiveness.

## Limitations & Future Work
- Uplink communication is increased by an additional class-wise aggregated gradient $\mathbf{g}_k\in\mathbb{R}^{\mathbb{C}\times d}$; for datasets with many classes (e.g., TinyImageNet with 200 classes), this adds about 0.4 MB per round.
- Gradients may pose a higher information leakage risk than prototypes, but the paper does not discuss performance under DP or encryption.
- $\mathcal{L}_{GM}$ uses cosine similarity rather than L2, matching only direction but not magnitude—how differences in gradient magnitude between large and small classes affect the relative scale of hyper-prototypes is not fully analyzed.
- Although Table A9 covers model heterogeneity, HPCL/HPAL still rely on shared feature dimension $d$; aligning feature spaces across architectures (e.g., ViT vs ConvNet) remains an open problem.

## Related Work & Insights
- **vs FedProto (AAAI'22)**: Both use prototypes as global signals, but FedProto averages local prototypes for the global prototype; this work learns hyper-prototypes via gradient matching, breaking the "local bias → global bias" chain and improving by 9 points on challenging domains (SYN).
- **vs FedSA (AAAI'25)**: FedSA also aims to "refine global anchors," but still updates at the prototype level; this work updates at the sample level (via gradients), structurally closer to centralized training.
- **vs FedTGP (AAAI'24)**: FedTGP introduces trainable global prototypes and adaptive-margin contrastive learning; this work pushes "trainability" down to the gradient matching level and proposes client-specific margin $d_k$, which is more fine-grained than FedTGP's global margin.
- **Insights**: Gradient matching originated as a tool for dataset condensation; its application in FL suggests that any scenario where "raw samples are inaccessible but gradients are available" can benefit, such as cross-institution GNN training or cross-device RecSys.

## Rating
- Novelty: ⭐⭐⭐⭐ Introducing gradient matching to FL prototype methods is a fresh perspective, though both gradient matching and contrastive learning are established tools.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 8 datasets × 3 heterogeneity scenarios × 8 baselines, with additional experiments on model heterogeneity, text modality, and fairness in the appendix—very comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Figures 1–2 clearly illustrate the motivation, and formula numbering is consistent; Section 4 is somewhat lengthy, and the HPCL formula could be more concise.
- Value: ⭐⭐⭐⭐ Hyper-prototypes can be plugged into any prototype-based FL method for consistent gains, offering direct value for real-world FL system deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] FedDAP: Domain-Aware Prototype Learning for Federated Learning under Domain Shift](../../CVPR2026/ai_safety/feddap_domain-aware_prototype_learning_for_federated_learning_under_domain_shift.md)
- [\[CVPR 2026\] ProxyFL: A Proxy-Guided Framework for Federated Semi-Supervised Learning](../../CVPR2026/ai_safety/proxyfl_a_proxy-guided_framework_for_federated_semi-supervised_learning.md)
- [\[NeurIPS 2025\] Enabling Differentially Private Federated Learning for Speech Recognition: Benchmarks, Adaptive Optimizers and Gradient Clipping](../../NeurIPS2025/ai_safety/enabling_differentially_private_federated_learning_for_speech_recognition_benchm.md)
- [\[CVPR 2026\] Computation and Communication Efficient Federated Unlearning via On-server Gradient Conflict Mitigation and Expression](../../CVPR2026/ai_safety/computation_and_communication_efficient_federated_unlearning_via_on-server_gradi.md)
- [\[AAAI 2026\] Robust Watermarking on Gradient Boosting Decision Trees](../../AAAI2026/ai_safety/robust_watermarking_on_gradient_boosting_decision_trees.md)

</div>

<!-- RELATED:END -->

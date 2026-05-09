---
title: >-
  [Paper Note] SubFLOT: Submodel Extraction for Efficient and Personalized Federated Learning via Optimal Transport
description: >-
  [CVPR 2026][AI Safety][Federated Learning] This paper proposes SubFLOT, a framework that leverages Optimal Transport (OT) on the server side to align the parameter distributions of a global model with clients' historical models, enabling personalized pruning without access to raw data. Combined with an adaptive regularization mechanism to suppress pruning-induced parameter drift, SubFLOT substantially outperforms existing federated pruning methods across multiple datasets.
tags:
  - CVPR 2026
  - AI Safety
  - Federated Learning
  - Network Pruning
  - Optimal Transport
  - Personalized Models
  - Heterogeneous Systems
date: 2026-05-08
content_hash: 742e7c0c385946ae
---

# SubFLOT: Submodel Extraction for Efficient and Personalized Federated Learning via Optimal Transport

**Conference**: CVPR 2026
**arXiv**: [2604.06631](https://arxiv.org/abs/2604.06631)
**Code**: N/A
**Area**: AI Security
**Keywords**: Federated Learning, Network Pruning, Optimal Transport, Personalized Models, Heterogeneous Systems

## TL;DR

This paper proposes SubFLOT, a framework that leverages Optimal Transport (OT) on the server side to align the parameter distributions of a global model with clients' historical models, enabling personalized pruning without access to raw data. Combined with an adaptive regularization mechanism to suppress pruning-induced parameter drift, SubFLOT substantially outperforms existing federated pruning methods across multiple datasets.

## Background & Motivation

**Background**: Federated Learning (FL) enables collaborative training while preserving data privacy, yet faces dual challenges in practical deployment: system heterogeneity (large disparities in device resources) and statistical heterogeneity (non-IID data distributions). Federated network pruning has emerged as a countermeasure, allowing different clients to train submodels of varying sizes, thereby reducing computational and communication overhead.

**Limitations of Prior Work**: Federated pruning leaves two critical problems unresolved. First, the placement of pruning decisions presents a dilemma: server-side pruning (e.g., HeteroFL) adopts a uniform compression strategy that precludes personalization, while client-side pruning (train–prune–fine-tune paradigm) achieves personalization at the cost of excessive computational burden on resource-constrained devices. Second, the pruning process itself exacerbates heterogeneity—submodel weight distributions under high pruning rates deviate from the global model (parameter drift), undermining training stability and global convergence.

**Key Challenge**: How can personalized pruning be realized on the server side—without accessing raw client data—while simultaneously resolving the parameter-space drift induced by pruning?

**Goal**: (1) Server-side personalized pruning—generating customized submodels for each client without touching their raw data; (2) Parameter drift suppression—preventing excessive divergence of parameter distributions across submodels trained under different pruning rates.

**Key Insight**: The authors treat each client's historical model parameters as a proxy for its local data distribution. Building on this insight, the pruning problem is reformulated as minimizing the Wasserstein distance between the global model and the historical model, with the resulting OT plan guiding personalized pruning.

**Core Idea**: Apply optimal transport in parameter space to align neurons of the global model with those of clients' historical models, achieving data-aware, server-side personalized pruning without any access to raw data.

## Method

### Overall Architecture

Each federated communication round in SubFLOT consists of three phases: (1) **Server-side OTP module**—optimal transport aligns global model parameters with each client's historical model to generate customized submodels for distribution; (2) **Client-side SAR training**—clients train their submodels on local data using an adaptive regularization loss to prevent parameter drift; (3) **Server-side OTA aggregation**—the OT mechanism is reused to align the updated heterogeneous submodels back into the global parameter space before aggregation.

### Key Designs

1. **Optimal Transport-enhanced Pruning (OTP)**:

    - **Function**: Generates a personalized pruned submodel for each client on the server side.
    - **Mechanism**: A layer-wise progressive matching strategy is adopted. For client $i$ at layer $l$, the transport plan $T_i^{(l-1)}$ from the previous layer is first used to remap the input space of the global model weights: $\hat{W}_G^{(l,l-1)} = W_G^{(l,l-1)} T_i^{(l-1)}$. The output neurons of the aligned global weights and the client's historical weights are then treated as two discrete probability distributions; a Euclidean distance cost matrix is computed, and the discrete OT problem is solved to obtain the transport plan $T_i^{(l)}$. The final submodel is derived by fusing the aligned global knowledge with the client's historical parameters: $\tilde{W}_i = \alpha \cdot W_{aligned} + (1-\alpha) \cdot W_i$, where $\alpha = 0.5$ balances global knowledge transfer and local specialization.
    - **Design Motivation**: Historical model parameters implicitly encode the client's local data distribution. OT exploits this property to identify, without accessing raw data, the subset of global-model neurons most relevant to the client's data, realizing genuinely data-aware personalization.

2. **Scaling-based Adaptive Regularization (SAR)**:

    - **Function**: Suppresses parameter drift of submodels during local client training.
    - **Mechanism**: A regularization term is added to the standard cross-entropy loss: $\mathcal{L}_{SAR}(W_i) = \rho_i \cdot \|W_i - \tilde{W}_i\|_2^2$, where $\rho_i$ is the client's pruning rate and $\tilde{W}_i$ is the anchor model provided by the server. The full objective is $\mathcal{L}_i = \mathcal{L}_{CE} + \lambda \cdot \mathcal{L}_{SAR}$, with $\lambda = 1.0$.
    - **Design Motivation**: Submodels with higher pruning rates are more susceptible to parameter drift; accordingly, the penalty strength scales proportionally with $\rho_i$, imposing stronger constraints on clients with aggressive pruning. Unlike the post-hoc correction in HeteroFL, SAR actively regularizes throughout training, preventing drift at the source.

3. **Optimal Transport-enhanced Aggregation (OTA)**:

    - **Function**: Aligns heterogeneous submodels from different clients into a common parameter space before aggregating them into a new global model.
    - **Mechanism**: For each client $i$, a transport mapping $\mathcal{T}_i$ is computed to align the updated client model $W_i^t$ back to the parameter space of the global model $W_G^t$ (the reverse direction of OTP); the aligned models are then aggregated via weighted averaging: $W_G^{t+1} = \sum_{i=1}^N p_i \cdot \mathcal{T}_i(W_i^t)$.
    - **Design Motivation**: Directly averaging heterogeneous parameters under standard FedAvg may mix semantically distinct neurons (destructive interference). OTA uses OT alignment to ensure functionally equivalent neurons are correctly matched prior to aggregation, while naturally normalizing parameter-scale discrepancies arising from different pruning rates.

### Loss & Training

Client local loss: $\mathcal{L}_i(W_i) = \mathcal{L}_{CE}(W_i; \mathcal{D}_i) + \lambda \cdot \rho_i \cdot \|W_i - \tilde{W}_i\|_2^2$. Training configuration: 20 clients, full participation (join ratio 1.0), 200 communication rounds, 5 local epochs per round, SGD optimizer (lr = 0.001), batch size 256. Pruning rates are sampled uniformly from $\{0, 1/4, 1/2, 3/4\}$. Convergence analysis proves that SubFLOT converges to a neighborhood of the global optimum at a linear rate of $1 - \mu\eta_l E/2$.

## Key Experimental Results

### Main Results (Label Skew Setting)

| Method | CIFAR10 | CIFAR100 | TinyImageNet | AG News | HAR |
|--------|---------|----------|-------------|---------|-----|
| HeteroFL | 84.54 | 40.95 | 19.68 | 84.12 | 69.80 |
| FlexFL | 85.13 | 49.21 | 22.23 | 86.02 | 76.24 |
| **SubFLOT** | **86.89** | **58.37** | **29.30** | **87.88** | **79.72** |

### Ablation Study (Feature Shift — PACS Dataset Average Accuracy)

| Method | Photo | Art | Cartoon | Sketch | Avg |
|--------|-------|-----|---------|--------|-----|
| HeteroFL | 16.23 | 13.66 | 20.27 | 26.90 | 19.27 |
| FlexFL | 16.34 | 14.78 | 21.56 | 28.01 | 20.17 |
| **SubFLOT** | **48.23** | **28.73** | **42.55** | **46.83** | **41.58** |

### Key Findings

- SubFLOT achieves 58.37% on CIFAR-100, surpassing the second-best method FlexFL (49.21%) by 9.16 percentage points; the margin grows with task complexity.
- Under the PACS feature-shift setting, SubFLOT attains an average accuracy of 41.58%, more than double that of the second-best method (~20%).
- Scalability experiments show that as the number of clients increases from 10 to 100, SubFLOT exhibits the smallest performance degradation (39.20% → 32.61%), while most baselines drop sharply.
- Grad-CAM visualizations confirm that submodels generated by OTP attend to the same semantically critical regions as the corresponding client historical models.

## Highlights & Insights

- **Paradigm shift**: SubFLOT is the first to systematically address server-side personalized pruning, challenging the prevailing assumption that "server-side = no personalization" and demonstrating that data-aware pruning is achievable without access to raw data.
- **Dual application of OT**: The same optimal transport mechanism is elegantly reused for both pruning (OTP) and aggregation (OTA), forming a complete closed loop from submodel distribution to parameter recovery.
- **Elegant adaptive regularization**: The SAR design, which weights the regularization penalty by the pruning rate, is concise yet effective, capturing the core intuition that more aggressive pruning demands stronger constraints.
- **Rigorous theoretical guarantees**: A formal convergence analysis establishes a linear convergence rate.

## Limitations & Future Work

- Although a layer-wise decomposition strategy is employed, OT computation still incurs considerable overhead for models with many layers or large numbers of neurons.
- Pruning rates are randomly sampled from a fixed discrete set; the optimal pruning rate per client based on actual device resources is not dynamically determined.
- The convergence analysis relies on a strong convexity assumption (Assumption 2), which has limited practical relevance for non-convex neural networks.
- Comparisons with other model compression paradigms in federated settings—such as knowledge distillation—are absent.
- Only structured pruning (width pruning) is evaluated; unstructured pruning and other compression forms are not explored.

## Related Work & Insights

- HeteroFL [Diao et al.] is the direct predecessor that SubFLOT improves upon; its uniform server-side pruning strategy serves as the baseline to surpass.
- FedOTP [Singh et al.] and FedAli apply OT to client-side feature-space alignment; SubFLOT transfers OT from feature space to parameter space and from the client side to the server side.
- The design principle of SAR generalizes to other heterogeneous federated learning scenarios, such as federated learning with heterogeneous model architectures.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First application of OT to federated pruning; the server-side personalized pruning paradigm is entirely novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers CV, NLP, and IoT domains; evaluates label skew, feature shift, and real-world settings with extensive scalability and ablation experiments.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem formulation and method description are clear, though the density of mathematical notation may require background knowledge in optimal transport.
- **Value**: ⭐⭐⭐⭐ — Carries significant practical implications for deploying federated learning on edge devices; the method has strong plug-and-play potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] FedDAP: Domain-Aware Prototype Learning for Federated Learning under Domain Shift](feddap_domain-aware_prototype_learning_for_federated_learning_under_domain_shift.md)
- [\[CVPR 2026\] Domain-Skewed Federated Learning with Feature Decoupling and Calibration](domain-skewed_federated_learning_with_feature_decoupling_and_calibration.md)
- [\[NeurIPS 2025\] DictPFL: Efficient and Private Federated Learning on Encrypted Gradients](../../NeurIPS2025/ai_safety/dictpfl_efficient_and_private_federated_learning_on_encrypted_gradients.md)
- [\[CVPR 2026\] FedRE: A Representation Entanglement Framework for Model-Heterogeneous Federated Learning](fedre_a_representation_entanglement_framework_for_model-heterogeneous_federated_.md)
- [\[CVPR 2026\] ProxyFL: A Proxy-Guided Framework for Federated Semi-Supervised Learning](proxyfl_a_proxy-guided_framework_for_federated_semi-supervised_learning.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] SkyMask: Attack-Agnostic Robust Federated Learning with Fine-Grained Learnable Masks
description: >-
  [ECCV2024][AI Safety][federated learning] This paper proposes SkyMask, which utilizes parameter-level learnable binary masks at the server side to detect malicious client model updates, achieving attack-agnostic robust federated learning that remains resilient even when up to 80% of clients are malicious.
tags:
  - "ECCV2024"
  - "AI Safety"
  - "federated learning"
  - "Byzantine Attack"
  - "Learnable Mask"
  - "Robust Aggregation"
date: 2026-05-08
content_hash: 2168da4e1ea37475
---

# SkyMask: Attack-Agnostic Robust Federated Learning with Fine-Grained Learnable Masks

**Conference**: ECCV2024  
**arXiv**: [2312.12484](https://arxiv.org/abs/2312.12484)  
**Code**: [KoalaYan/SkyMask](https://github.com/KoalaYan/SkyMask)  
**Area**: Optimization  
**Keywords**: federated learning, Byzantine Attack, Learnable Mask, Robust Aggregation  
**Affiliations**: Shanghai Jiao Tong University, Stevens Institute of Technology, Queen's University Belfast, Intel

## TL;DR

This paper proposes SkyMask, which utilizes parameter-level learnable binary masks at the server side to detect malicious client model updates, achieving attack-agnostic robust federated learning that remains resilient even when up to 80% of clients are malicious.

## Background & Motivation

Federated learning (FL) protects data privacy via distributed training; however, its distributed nature makes the system vulnerable to Byzantine attacks, where compromised clients upload malicious model updates to disrupt the global model. With the emergence of fine-grained attacks (e.g., Fang attack, AGR-agnostic attack), attacks have become more stealthy. They exploit the sensitivity differences across different layers and parameters to precisely poison a small number of parameters, making model-level anomaly detection extremely difficult to identify them.

Existing defense strategies can be categorized into two main types:

1. **Model-level defenses** (e.g., FLTrust, Krum, FLAME): These detect anomalies based on the overall statistics of model updates (Euclidean distance, cosine similarity). Facing fine-grained attacks, they either fail to detect them or overreact, mistakenly penalizing benign clients.
2. **Greedy parameter filtering** (e.g., Trimmed-Mean): These perform coordinate-wise sorting and trimming. However, since fine-grained attacks only modify specific parameters, they can easily bypass this defense.

Through PCA visualization experiments, the authors found that although fine-grained attacks disguise themselves as benign updates at the model level, once learnable masks are trained for each model update, the masks of malicious clients can be clearly distinguished in the high-dimensional space. This observation inspires the design of SkyMask.

## Core Problem

How to detect malicious model updates in federated learning at a parameter-level granularity without prior knowledge of the attack types, thus effectively defending against various Byzantine attacks including fine-grained attacks?

## Method

### Overall Architecture

SkyMask consists of six steps in each communication round:

1. The server distributes the global model to each client.
2. Clients perform local training and upload their model updates.
3. The server freezes all model updates and assigns a same-sized learnable mask to each client.
4. All masks are trained to convergence on a clean root dataset.
5. The masks are analyzed via clustering to detect and remove malicious clients.
6. Only the remaining benign model updates are aggregated to form the new global model.

### Mask Initialization and Training

The server creates a mask $m_i$ for each client $i$ (initialized to all ones) and freezes all local model parameters. The aggregated masked model is given by:

$$\tilde{W}_{t+1} = \frac{\sum_{i=1}^{n} \tilde{m}_i \odot W_{t+1}^i}{\sum_{i=1}^{n} \tilde{m}_i}$$

where $\tilde{m}_i = \sigma(m_i)$ approximates a binary mask via the sigmoid function, in the range of $(0, 1)$. Using sigmoid here instead of a hard threshold ensures that gradients can be backpropagated.

The masks are trained on the root dataset $D_r$ using the standard cross-entropy loss:

$$m_i := m_i - \gamma \cdot \nabla_{m_i} f(\tilde{W}_{t+1}, D_r)$$

Upon convergence, the real-valued masks are binarized using a threshold $\tau$: $\hat{m}_i[k] = 1$ when $\tilde{m}_i[k] > \tau$, and $0$ otherwise.

### Mask Clustering and Classification

The trained binary masks represent the parameter-level characteristics of each client's model update. A Gaussian Mixture Model (GMM) is employed to cluster the masks:

- **Under no attack**: The clustering results in only one cluster, and all clients are retained.
- **Under attack**: Two clusters are formed, and it is necessary to determine which one is the benign cluster.

To identify the benign cluster, a trusted root model is introduced: the server trains an additional model on the root dataset, assigns it a mask, and trains it jointly. Clients belonging to the same cluster as the root model's mask are identified as benign.

### Why It Works

Poisoning from malicious models is typically concentrated in specific layers or parameters. During training, the masks learn to reduce the weights of poisoned parameters to 0 to optimize the global model's performance; thus, the masks of malicious clients exhibit a significantly different 0-1 pattern compared to those of benign clients. This mechanism does not rely on knowledge of specific attack methods, making it attack-agnostic.

### Computational Complexity

- Time complexity: $O(Tt_m + nVt_m)$, where $t_m$ is the number of mask training iterations. The extra overhead is equivalent to performing a few additional local training rounds.
- Space complexity: $O(nV)$, which is approximately twice that of other methods (requiring storage for $n$ masks) but remains acceptable at the server side.
- There is no data dependency among mask parameters, allowing parallel training acceleration.

## Key Experimental Results

### Experimental Setup

- Datasets: Fashion-MNIST, CIFAR-10, CIFAR-100 (non-IID, bias $p=0.5$)
- Models: 4-layer CNN (Fashion-MNIST), ResNet20 (CIFAR-10/100)
- 100 clients, with a default of 20% malicious clients, and a root dataset of only 100 samples.

### Defense Effectiveness (Test Accuracy, 20% Malicious Clients)

| Attack | FedAvg | FLTrust | SkyMask |
|------|--------|---------|---------|
| CIFAR-10 Min-Max | 0.58 | 0.68 | **0.77** |
| CIFAR-10 Fang-Trim | 0.10 | 0.68 | **0.76** |
| CIFAR-10 Fang-Krum | 0.58 | 0.75 | **0.77** |
| CIFAR-100 Min-Max | 0.16 | 0.30 | **0.44** |
| CIFAR-100 Fang-Trim | 0.01 | 0.34 | **0.44** |
| CIFAR-100 Fang-Krum | 0.03 | 0.37 | **0.44** |

SkyMask achieves test accuracy under all attacks that is at or close to the no-attack level, outperforming the state-of-the-art (SOTA) defense by up to 14%.

### Malicious Detection Quality (CIFAR-10)

| Method | Min-Max FPR/FNR | Min-Sum FPR/FNR | Fang-Krum FPR/FNR |
|------|------|------|------|
| Tolpegin | 36.5%/88.0% | 38.8%/78.0% | 37.4%/84.0% |
| FLDetector | 100%/100% | 100%/100% | 87.2%/100% |
| SkyMask | **0%/0%** | **0%/0%** | **0%/0%** |

### High Malicious Ratio

When the proportion of malicious clients is 40%, 60%, or 80%, SkyMask is the only method that maintains effective defense. Once the malicious ratio exceeds 50%, FLTrust also begins to fail (showing accuracy fluctuations and incorrect convergence directions), whereas SkyMask still converges stably.

### Scalability

When the number of clients scales to 200 and 500, both FPR and FNR remain at 0% under all fine-grained and targeted attacks.

## Highlights & Insights

1. **Parameter-level Detection Granularity**: This is the first work to introduce learnable masks into Byzantine defense, capturing malicious update characteristics at the parameter level and breaking through the limitations of traditional model-level detection.
2. **Attack-Agnostic**: It does not require customization for specific attacks and is effective against seven SOTA attacks (both untargeted and targeted).
3. **Robustness in Extreme Scenarios**: It remains defensive even when the proportion of malicious clients reaches up to 80%, being the only method to maintain performance under highly malicious ratios.
4. **Friendly to Benign Clients**: It does not degrade model performance when there are no attacks (deviation from FedAvg is <1%), avoiding mistakenly penalizing benign clients.
5. **Modular Design**: It can be integrated as a plug-in into various aggregation algorithms, showing strong compatibility.

## Limitations & Future Work

1. **Dependence on Root Dataset**: The server requires a small, clean dataset (100 samples), which might be restricted in certain highly privacy-sensitive scenarios.
2. **Computational Overhead**: Extra mask training iterations ($t_m$ rounds of forward/backward propagation) are required in each round. When the number of clients is large or the model is complex, the computational pressure on the server increases.
3. **Doubled Space Overhead**: Same-sized masks as the model need to be stored for each client, significantly increasing memory requirements when the model parameter size is massive (e.g., LLMs).
4. **Clustering Assumption**: It is assumed that attacker behaviors form separable clusters in the mask space. If attackers can adapt to the masking mechanism and perform adaptive attacks, the effectiveness may decrease.
5. **Validation Only on CNN and ResNet20**: Experiments on larger models (such as ViT and large-scale ResNet) are currently lacking.

## Related Work & Insights

| Method | Detection Granularity | Attack-Agnostic | High Malicious Ratio | Impact Under No Attack |
|------|---------|---------|----------|-----------|
| FLTrust | Model-level | Partial | ≤50% | Accuracy drops by 2-5% |
| Krum | Model-level | No | Poor | Accuracy drops by 6-27% |
| Trimmed-Mean | Parameter-level (Greedy) | No | Poor | Slight impact |
| DeepSight | Model-level | Partial | Poor | Slight impact |
| FLAME | Model-level | Partial | Poor | Slight impact |
| FLDetector | Model-level | No | Not tested | FPR 18.2% |
| **SkyMask** | **Parameter-level** | **Yes** | **≤80%** | **<1%** |

Compared with the strongest baseline FLTrust, the core advantages of SkyMask include: (1) parameter-level instead of model-level detection, enabling it to cope with fine-grained attacks; (2) remaining effective when the malicious ratio exceeds 50%.

## Insights & Connections

- The concept of mask learning can be extended to anomaly detection scenarios in other distributed systems, beyond federated learning.
- Distinct from FedMask where masks are utilized for personalization, SkyMask applies masks for security detection, demonstrating the versatility of masking mechanisms.
- The utilization of the root dataset is in line with FLTrust; however, SkyMask exploits the information from the root dataset more thoroughly—using it not only to compute similarity but also to drive mask training.
- Future research directions: how adaptive attackers might counter the masking mechanism, variants without a root dataset, and optimization of mask training efficiency.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Applying parameter-level learnable masks to Byzantine defense is a novel perspective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 3 datasets × 7 attacks × multiple baselines, including high malicious ratio and scalability experiments.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear motivation, comprehensive method descriptions, and persuasive visual analysis.
- **Value**: ⭐⭐⭐⭐ — High practicality, significantly outperforming existing methods in extreme scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Fisher Calibration for Backdoor-Robust Heterogeneous Federated Learning](fisher_calibration_for_backdoor-robust_heterogeneous_federated_learning.md)
- [\[ECCV 2024\] Towards Multi-modal Transformers in Federated Learning](towards_multi-modal_transformers_in_federated_learning.md)
- [\[ICLR 2026\] Fine-Grained Class-Conditional Distribution Balancing for Debiased Learning](../../ICLR2026/ai_safety/fine-grained_class-conditional_distribution_balancing_for_debiased_learning.md)
- [\[ICML 2026\] Decoupled Training with Local Reinforcement Fine-Tuning in Federated Learning](../../ICML2026/ai_safety/decoupled_training_with_local_reinforcement_fine-tuning_in_federated_learning.md)
- [\[ICLR 2026\] Robust Federated Inference](../../ICLR2026/ai_safety/robust_federated_inference.md)

</div>

<!-- RELATED:END -->

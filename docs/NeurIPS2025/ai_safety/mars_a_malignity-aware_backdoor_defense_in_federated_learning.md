---
title: >-
  [Paper Note] MARS: A Malignity-Aware Backdoor Defense in Federated Learning
description: >-
  [NeurIPS 2025][AI Safety][backdoor attack defense] This paper proposes MARS, a defense method that quantifies the malignity of local models by computing per-neuron Backdoor Energy (BE)…
tags:
  - "NeurIPS 2025"
  - "AI Safety"
  - "backdoor attack defense"
  - "federated learning"
  - "Wasserstein distance"
  - "backdoor energy"
  - "clustering-based detection"
date: 2026-05-08
content_hash: 239e29e537d2717b
---

# MARS: A Malignity-Aware Backdoor Defense in Federated Learning

**Conference**: NeurIPS 2025
**arXiv**: [2509.20383](https://arxiv.org/abs/2509.20383)  
**Code**: [GitHub](https://github.com/yunming181920/MARS)  
**Area**: AI Safety / Federated Learning
**Keywords**: backdoor attack defense, federated learning, Wasserstein distance, backdoor energy, clustering-based detection

## TL;DR

This paper proposes MARS, a defense method that quantifies the malignity of local models by computing per-neuron Backdoor Energy (BE), and leverages Wasserstein distance-based clustering to effectively identify backdoor models in federated learning.

## Background & Motivation

The distributed nature of federated learning (FL) makes it inherently vulnerable to backdoor attacks. Existing defenses primarily rely on three categories of empirical statistical metrics: norm constraints, out-of-distribution (OOD) detection, and consistency detection. However, state-of-the-art attacks (e.g., 3DFed, CerP, DarkFed) circumvent all three categories by constraining the norm, distribution, and consistency of malicious updates to mimic benign ones.

The authors empirically validate this failure: (1) the norm of backdoor updates can be smaller than that of benign updates; (2) backdoor and benign updates are indistinguishable after PCA projection; (3) the cosine similarity among backdoor updates can be lower than that among certain benign updates.

The core insight is that existing metrics are **loosely coupled** with backdoor attacks and lack the ability to perceive malicious intent. A metric that is **tightly coupled** with the nature of backdoor attacks is therefore needed.

## Method

### Overall Architecture

MARS consists of three steps: (1) computing the Backdoor Energy (BE) for each neuron; (2) extracting the most prominent BE values to form the Concentrated Backdoor Energy (CBE); and (3) applying Wasserstein distance-based clustering to identify backdoor models.

### Key Designs

1. **Backdoor Energy (BE)**: Intuitively, BE measures the degree of association between each neuron and backdoor attacks. The ideal definition requires clean data and triggers, which are unavailable in FL. The authors approximate BE using the Lipschitz constant as an upper bound: $BE_k^{(l)}(F) = \|f_k^{(l)}\|_{Lip}$. This approximation requires only model parameters and is independent of clean data or triggers. Theoretical support is provided by Theorem 4.1, which proves the upper bound of BE.

2. **Concentrated Backdoor Energy (CBE)**: Since backdoors act as shortcuts, only a small number of neurons are associated with the backdoor. The top-κ% (default 5%) BE values are extracted from each layer and concatenated into a one-dimensional vector, maximizing backdoor information density while reducing interference from irrelevant neurons.

3. **Wasserstein Distance-based Clustering (K-WMeans)**: Traditional K-Means uses Euclidean or cosine distance, which is sensitive to element ordering. In FL, the top BE values of different backdoor models may appear at different neuron positions, making correct clustering difficult even when the values are globally larger. Wasserstein distance focuses on the probability distribution of elements rather than their order, making it more suitable for this setting. A toy example validates this: two backdoor CBEs L1=[1,2,3,4,5] and L2=[5,5,3,2,2] have a Wasserstein distance of 0.40, far smaller than their distances to the benign L3=[1,1,1,1,1] (2.00/2.40).

### Loss & Training

- Cluster selection after clustering: rather than assuming a benign majority, the cluster with the smaller centroid norm is selected.
- When the Wasserstein distance between two clusters does not exceed threshold ε, all models are deemed benign and both clusters are retained.
- Default hyperparameters: κ=5, ε=0.03.

## Key Experimental Results

### Main Results

| Dataset | Attack | Metric | MARS | Best Baseline | Gain |
|---------|--------|--------|------|---------------|------|
| MNIST | 3DFed | ASR↓ | 9.72% | 16.69% (FedCLP) | Significant reduction |
| MNIST | 3DFed | TPR↑ | 100% | 0% (most methods) | Perfect detection |
| CIFAR-10 | CerP | ASR↓ | 10.03% | 10.01% (Multi-Krum) | On par |
| CIFAR-10 | 3DFed | ASR↓ | 9.86% | 7.55% (FedCLP) | Competitive |
| CIFAR-100 | MRA | CAD↑ | — | — | Consistently leading |

### Ablation Study

| Configuration | Description |
|---------------|-------------|
| κ (top%) | Controls the proportion of BE extracted from each layer; default 5% |
| ε (threshold) | Controls cluster distance determination; default 0.03 |
| Distance metric | Wasserstein > Euclidean ≈ Cosine (validated via toy example) |

### Key Findings

- MARS achieves 100% TPR and near-zero FPR against the state-of-the-art 3DFed attack.
- The defense remains effective even when the attacker ratio exceeds 50%.
- In attack-free scenarios, MARS does not degrade model accuracy, satisfying the fidelity objective.

## Highlights & Insights

- Deriving the upper bound of BE from the Lipschitz constant is a theoretically elegant contribution that eliminates dependence on triggers and clean data.
- Replacing Euclidean/cosine distance with Wasserstein distance is a key innovation that resolves the issue of inconsistent CBE element ordering across FL clients.
- The proposed defense does not require assuming that the attacker ratio is below 50%, improving its practical applicability.

## Limitations & Future Work

- Validation is limited to computer vision tasks (MNIST/CIFAR); NLP and other domains are not explored.
- Computing the Lipschitz constant may pose efficiency challenges for large-scale models.
- The discussion of adaptive attack scenarios could be more thorough.
- The sensitivity analysis of hyperparameters κ and ε warrants further investigation.

## Related Work & Insights

- MARS differs from CLP (Channel Lipschitzness based Pruning) in that it uses the Lipschitz constant for detection rather than pruning, and further introduces CBE and Wasserstein-based clustering.
- The defense design paradigm shifts from "detecting anomalous statistics" to "perceiving malicious intent," a perspective worth adopting in future work.
- The application of Wasserstein distance for distribution comparison is generalizable to other security domains.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of backdoor energy and Wasserstein-based clustering constitutes a novel defense paradigm.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluated on 3 datasets, against 3 state-of-the-art attacks, and compared with 8 baseline defenses.
- **Writing Quality**: ⭐⭐⭐⭐ Logically clear; the failure analysis in the motivation section is particularly convincing.
- **Value**: ⭐⭐⭐⭐ Practically significant for FL security and offers a new perspective in the arms race between attacks and defenses.

## Additional Details

- The threat model assumes attackers may constitute a majority (>50%), which is a stronger assumption than that of most existing defenses.
- The defense is deployed server-side, requiring only access to model parameters without any client training data.
- Experiments use 100 clients with 20 attackers; 20 clients participate per round, including 4 attackers.
- In innocent (attack-free) scenarios, MARS discards no clients and preserves the convergence speed of FedAvg.
- BackdoorIndicator (USENIX Security 2024) is included as a baseline; MARS still outperforms it.
- A customized adaptive attack targeting MARS is designed and evaluated to verify the robustness of the defense.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Find a Scapegoat: Poisoning Membership Inference Attack and Defense to Federated Learning](../../ICCV2025/ai_safety/find_a_scapegoat_poisoning_membership_inference_attack_and_defense_to_federated_.md)
- [\[ICCV 2025\] Client2Vec: Improving Federated Learning by Distribution Shifts Aware Client Indexing](../../ICCV2025/ai_safety/client2vec_improving_federated_learning_by_distribution_shifts_aware_client_inde.md)
- [\[NeurIPS 2025\] DictPFL: Efficient and Private Federated Learning on Encrypted Gradients](dictpfl_efficient_and_private_federated_learning_on_encrypted_gradients.md)
- [\[CVPR 2026\] FedDAP: Domain-Aware Prototype Learning for Federated Learning under Domain Shift](../../CVPR2026/ai_safety/feddap_domain-aware_prototype_learning_for_federated_learning_under_domain_shift.md)
- [\[NeurIPS 2025\] FedFACT: A Provable Framework for Controllable Group-Fairness Calibration in Federated Learning](fedfact_a_provable_framework_for_controllable_group-fairness_calibration_in_fede.md)

</div>

<!-- RELATED:END -->

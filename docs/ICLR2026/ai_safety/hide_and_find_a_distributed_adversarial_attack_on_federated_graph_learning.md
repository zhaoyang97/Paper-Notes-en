---
title: >-
  [Paper Note] Hide and Find: A Distributed Adversarial Attack on Federated Graph Learning
description: >-
  [ICLR2026][AI Safety][Federated Graph Learning] This paper proposes FedShift, a two-stage "hide-and-find" distributed adversarial attack framework. In the first stage…
tags:
  - "ICLR2026"
  - "AI Safety"
  - "Federated Graph Learning"
  - "adversarial attack"
  - "Backdoor Attack"
  - "Graph Neural Networks"
  - "Data Poisoning"
date: 2026-05-08
content_hash: 1c341fec33c084bc
---

# Hide and Find: A Distributed Adversarial Attack on Federated Graph Learning

**Conference**: ICLR2026  
**arXiv**: [2603.07743](https://arxiv.org/abs/2603.07743)  
**Code**: To be released  
**Area**: AI Security  
**Keywords**: Federated Graph Learning, adversarial attack, Backdoor Attack, Graph Neural Networks, Data Poisoning

## TL;DR
This paper proposes FedShift, a two-stage "hide-and-find" distributed adversarial attack framework. In the first stage, covert shifters are injected into training graphs via gentle distributional shifts. In the second stage, the trained shifter generator serves as a warm initialization for efficiently searching adversarial perturbations, which are then aggregated across multiple malicious clients to form the final adversarial examples. FedShift achieves state-of-the-art attack success rates on six large-scale datasets, evades three mainstream defense algorithms, and improves convergence speed by over 90%.

## Background & Motivation
Federated Graph Learning (FedGL) enables multiple clients to collaboratively train GNNs without sharing raw graph data, and has been widely adopted in applications such as disease prediction and recommendation systems. However, FedGL faces serious security threats: malicious clients can manipulate the global model into making incorrect predictions via backdoor or adversarial attacks.

Existing attack methods suffer from three core challenges:

1. **Backdoor signal dilution**: Backdoor signals from malicious clients are smoothed out by benign clients during federated aggregation, significantly degrading attack effectiveness.
2. **Trade-off between stealthiness and effectiveness**: Increasing the attack budget can counteract signal dilution, but large-scale data poisoning is easily detected by defense algorithms.
3. **Slow convergence of adversarial attacks**: The discrete nature of graph structures and the non-convexity of optimization objectives lead to slow convergence and high computational cost when searching for adversarial perturbations.

## Core Problem
How to design a distributed attack that simultaneously addresses the signal dilution problem in backdoor attacks and the convergence efficiency problem in adversarial attacks, achieving a high attack success rate while maintaining strong stealthiness?

## Method

### Overall Architecture: Two-Stage "Hide and Find"

FedShift integrates the backdoor attack paradigm into adversarial attack design, using a two-stage framework where each stage compensates for the limitations of the other.

### Stage 1: Gentle Data Poisoning

**Objective**: Before federated training begins, train an adaptive shifter generator for each malicious client to inject covert shifters into the training data.

**Step 1 — Local GNN Pre-training**: Each malicious client pre-trains a GAT model $\theta_i^*$ on its local data to extract high-quality graph embeddings and guide the subsequent shifter generator training.

**Step 2 — Adaptive Shifter Generator**, comprising two sub-modules:

- **Shifter Position Learning**: Node clustering coefficients are computed based on graph topology, and the most influential subset of nodes $\mathcal{V}_p$ is selected as injection positions.
- **Shifter Shape Learning**: The generator $G_{\text{gen}}$ produces feature perturbations $\Delta\mathbf{X}_p$ for selected nodes, modifying only node features without altering edge connectivity to enhance stealthiness.

**Key Loss Function Design**:

1. **Distribution Proximity Loss $L_{\text{dist}}$**: K-means clustering is applied to target-class embeddings to obtain centroids; the loss minimizes the cosine distance between poisoned graph embeddings and the nearest centroid. This gently moves the representation of poisoned graphs toward the target-class decision boundary **without crossing it**, without modifying labels or constructing forced mappings, making the behavior of malicious clients nearly indistinguishable from benign ones.
2. **Homophily Loss $L_{\text{homo}}$**: Based on the graph homophily assumption, this loss penalizes large feature discrepancies between connected nodes, ensuring that the perturbed graph remains structurally natural.
3. **Boundary-Balanced Cross-Entropy Loss $L_{\text{ce}}$**: Requires that poisoned graphs are still classified as their original labels by the local model, preventing distributional shifts from crossing the decision boundary.

Total loss: $L_{\text{stage1}} = \lambda_{\text{dist}} L_{\text{dist}} + \lambda_{\text{homo}} L_{\text{homo}} + \lambda_{\text{ce}} L_{\text{ce}}$

**Online Fine-tuning**: During federated training, the shifter generator can be dynamically fine-tuned using the global model received from the server.

### Stage 2: Adversarial Perturbation Finding

After federated training, the global model $\theta^*$ is frozen and optimization continues from the Stage 1 shifter generator as a **high-quality initialization**. The optimization objective now shifts to maximizing attack success rate:

$$L_{\text{stage2}} = L_{\text{attack}} + \lambda_{\text{homo}} L_{\text{homo}}$$

where $L_{\text{attack}} = \text{CrossEntropy}(f(G_p; \theta^*), y_t)$ drives the perturbed graph to cross the decision boundary.

**Final Attack**: Adversarial perturbations generated by multiple malicious clients are aggregated to form the final adversarial example, achieving a "1+1>2" effect.

## Key Experimental Results

Comparisons are conducted against four SOTA methods on six large-scale graph datasets (DD, NCI109, Mutagenicity, FRANKENSTEIN, Eth-Phish&Hack, Gossipcop).

### Q1: Resistance to Federated Signal Dilution
- As the malicious client ratio decreases from 0.2 to 0.05, baseline methods experience an average ASR drop of 25.6%–53.3%.
- FedShift's ASR drops by less than 5%, consistently maintaining the best attack performance.
- Under the $|C_M|/N=0.2$ setting, FedShift achieves the highest AAS on all six datasets.

### Q2: Evasion of Defense Algorithms
- Against three mainstream robust federated learning defenses — FoolsGold, FedKrum, and FedBulyan — FedShift maintains the highest AAS in all scenarios, outperforming the strongest baseline by an average of 4.9%.

### Q3: Convergence Efficiency
- Compared to NI-GDBA optimizing from scratch, FedShift (without federated fine-tuning) reduces required epochs by **90.3%**.
- Full FedShift (with federated fine-tuning) reduces required epochs by **98.3%**.

### Ablation Study
- Stage 1 only → +FL-Tune: average AAS improves by 17.0%.
- Stage 1 only → +Stage 2: average AAS improves by 32.2%.
- The combination of Stage 1 + Stage 2 nearly matches the full model, confirming that Stage 2 is the key driver of attack effectiveness.

## Highlights & Insights
1. **Novel two-stage attack paradigm**: The first work to integrate backdoor attack (injection) and adversarial attack (search) within a unified framework, leveraging information from the entire federated learning process.
2. **Gentle distributional shift strategy**: By avoiding label modification and pushing representations toward — but not past — the decision boundary, stealthiness is addressed at a fundamental level.
3. **Optimization initialization advantage**: The Stage 1 shifter generator provides a high-quality starting point for Stage 2, improving convergence efficiency by an order of magnitude.
4. **Multi-client perturbation aggregation**: Complementary perturbations from different malicious clients yield a combined effect far exceeding that of any individual perturbation.
5. **Comprehensive experiments**: Six large-scale cross-domain datasets, three defense algorithms, and extensive ablation and hyperparameter analyses.

## Limitations & Future Work
1. **Strong threat model assumptions**: The attacker is assumed to participate throughout the entire federated training process; in practice, malicious clients may be dynamically excluded.
2. **Restricted to graph classification**: Performance on other graph tasks such as node classification and link prediction has not been verified.
3. **Limited defense coverage**: Only three defense algorithms are evaluated; newer methods such as FLTrust and FLAME are not included.
4. **Single GNN architecture**: All experiments use GAT as the backbone; generalizability to other architectures such as GIN and GraphSAGE is not validated.
5. **No poisoning detection analysis**: The paper does not evaluate whether data-level anomaly detection methods can identify shifter injection.

## Related Work & Insights

| Method | Type | Stealthiness | Anti-Dilution | Convergence | Overall |
|--------|------|-------------|---------------|-------------|---------|
| Rand-GDBA | Backdoor (static trigger) | Low | Poor | — | Low |
| GTA | Backdoor (adaptive trigger) | Medium | Medium | — | Medium |
| Opt-GDBA | Backdoor (optimized trigger) | Medium | Medium | — | Medium |
| NI-GDBA | Adversarial (from scratch) | High | N/A | Slow | Medium-High |
| **FedShift** | **Backdoor + Adversarial** | **High** | **Strong** | **Fast** | **Highest** |

The core innovation of FedShift lies in breaking the paradigm barrier between backdoor and adversarial attacks: the injection stage of backdoor attack provides stealthiness and an optimization starting point, while the search stage of adversarial attack delivers final attack effectiveness.

- **Defense perspective**: This work reveals that attacks conducted via gentle distributional shifts in feature space are extremely difficult to detect; future defenses should focus on monitoring subtle anomalous drifts in embedding distributions during training.
- **Federated learning security**: The results indicate that relying solely on aggregation-level robust defenses (e.g., Krum, Bulyan) is insufficient against such attacks; data-level detection mechanisms must be incorporated.
- **Attack method fusion**: The two-stage "inject-and-search" paradigm can be generalized to other federated learning tasks (text, image), serving as a general-purpose attack design template.

## Rating
- Novelty: ⭐⭐⭐⭐ — The two-stage "Hide and Find" paradigm is pioneering, naturally unifying backdoor and adversarial attacks.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Six large-scale datasets, three defenses, and rich ablations; coverage of GNN architectures and defense methods could be broader.
- Writing Quality: ⭐⭐⭐⭐ — Motivation is clear, with a well-structured correspondence between three challenges and three solutions; strong logical coherence.
- Value: ⭐⭐⭐⭐ — Provides important reference value for federated graph learning security research; the new paradigm can inspire subsequent work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Find a Scapegoat: Poisoning Membership Inference Attack and Defense to Federated Learning](../../ICCV2025/ai_safety/find_a_scapegoat_poisoning_membership_inference_attack_and_defense_to_federated_.md)
- [\[ICLR 2026\] ATEX-CF: Attack-Informed Counterfactual Explanations for Graph Neural Networks](atex-cf_attack-informed_counterfactual_explanations_for_graph_neural_networks.md)
- [\[ICLR 2026\] Less is More: Towards Simple Graph Contrastive Learning](less_is_more_towards_simple_graph_contrastive_learning.md)
- [\[CVPR 2026\] FedAFD: Multimodal Federated Learning via Adversarial Fusion and Distillation](../../CVPR2026/ai_safety/fedafd_multimodal_federated_learning_via_adversarial_fusion_and_distillation.md)
- [\[ICLR 2026\] Traceable Black-box Watermarks for Federated Learning](traceable_black-box_watermarks_for_federated_learning.md)

</div>

<!-- RELATED:END -->

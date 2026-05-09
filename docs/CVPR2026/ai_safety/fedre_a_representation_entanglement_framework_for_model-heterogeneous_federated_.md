---
title: >-
  [Paper Note] FedRE: A Representation Entanglement Framework for Model-Heterogeneous Federated Learning
description: >-
  [CVPR 2026][AI Safety][Federated Learning] This paper proposes FedRE, a framework that achieves a three-way balance among performance, privacy protection, and communication overhead in model-heterogeneous federated learning via "entangled representations"—aggregating all local representations of each client into a single cross-class representation using normalized random weights.
tags:
  - CVPR 2026
  - AI Safety
  - Federated Learning
  - Model Heterogeneity
  - Entangled Representation
  - Privacy Preservation
  - Communication Efficiency
date: 2026-05-08
content_hash: 121c0f22e39688bd
---

# FedRE: A Representation Entanglement Framework for Model-Heterogeneous Federated Learning

**Conference**: CVPR 2026  
**arXiv**: [2511.22265](https://arxiv.org/abs/2511.22265)  
**Code**: [GitHub](https://github.com/AIResearch-Group/FedRE)  
**Area**: AI Safety / Federated Learning  
**Keywords**: Federated Learning, Model Heterogeneity, Entangled Representation, Privacy Preservation, Communication Efficiency

## TL;DR

This paper proposes FedRE, a framework that achieves a three-way balance among performance, privacy protection, and communication overhead in model-heterogeneous federated learning via "entangled representations"—aggregating all local representations of each client into a single cross-class representation using normalized random weights.

## Background & Motivation

Federated learning (FL) enables multiple clients to collaboratively train models while preserving privacy. In practice, however, hardware and computational capabilities vary significantly across clients, making it unrealistic to enforce a homogeneous model architecture for all participants. This has motivated research on **model-heterogeneous FL**, where clients may employ different representation extractors while sharing a homogeneous classifier.

Existing model-heterogeneous FL methods face a dilemma in selecting the form of client knowledge to share:
- **Representations / logits / small models**: Effectively encode high-level knowledge but introduce significant communication overhead and privacy risks when uploaded to the server (raw samples can be reconstructed via representation inversion attacks).
- **Classifiers**: Lightweight but may inherit biases from local data distributions.
- **Prototypes (class-mean representations)**: Lightweight and reduce privacy risks, but capture only class-level information with limited intra-class variability, and tend to produce overly sharp decision boundaries when training a global classifier.

The core problem: *Does a more effective, privacy-safe, and lightweight form of client knowledge exist for model-heterogeneous FL?*

## Method

### Overall Architecture

The core idea of FedRE is to aggregate all local representations of each client into **a single** cross-class "entangled representation" along with a corresponding "entangled label encoding," which are uploaded to the server to train the global classifier. The workflow proceeds as follows:

1. **Client update**: Train the local model on local data using cross-entropy loss.
2. **Representation entanglement and upload**: Generate the entangled representation and entangled label encoding and upload them to the server.
3. **Global classifier update and broadcast**: The server trains the global classifier using the entangled representations and broadcasts it to clients.

### Key Designs

1. **Representation Mapping (RM)**:

    - Maps local representations from heterogeneous architectures to a consistent dimensionality.
    - Three RM operations are evaluated: average pooling (AP), max pooling (MP), and fully connected layer (FC).
    - AP performs best (CIFAR-100 PRA: 46.36% vs. MP 45.97% vs. FC 44.53%).

2. **Representation Entanglement (RE)**:

    $\widetilde{\mathbf{r}}_k = \sum_{i=1}^{|\mathcal{D}_k|} w_i^k \text{RM}[\mathbf{g}_k(\phi_k; \mathbf{x}_i^k)], \quad \widetilde{\mathbf{y}}_k = \sum_{i=1}^{|\mathcal{D}_k|} w_i^k \mathbf{y}_i^k$

   where $w_i^k \in [0,1]$ are normalized random weights. The same set of weights simultaneously aggregates both the representations and their one-hot label encodings.

   The default strategy is **Random Average Prototype (RAP)**: class prototypes are first computed, then aggregated into a single entangled representation using random weights.

3. **Per-round random weight resampling**:

    - Random weights are resampled at each communication round, introducing diversity.
    - The entangled label encoding provides cross-class supervision signals.
    - This prevents the global classifier from being overconfident about any single class, promoting smoother decision boundaries.
    - **Design Motivation**: Motivated by a toy experiment comparing FedAllRep (uploading all representations, best performance 63.50%), FedGH (uploading prototypes, sharp boundaries, 60.50%), and FedRE (smooth boundaries, 62.00%).

4. **Privacy protection mechanism**:

    - The entangled representation mixes cross-class information, making individual sample reconstruction difficult.
    - Each client uploads only one entangled representation per round, further reducing the attack surface for information leakage.

### Loss & Training

- **Client**: Local model trained with cross-entropy loss $\mathcal{L}_{ce}$.
- **Server**: Global classifier trained with cross-entropy loss: $\min_\omega \sum_{k=1}^K \mathcal{L}_{ce}[f(\omega; \widetilde{\mathbf{r}}_k), \widetilde{\mathbf{y}}_k]$.
- The computational complexity of RE is only $\mathcal{O}(n(d+C))$, requiring no additional gradient computation.
- Setup: 10 clients, SGD optimizer, 100 communication rounds, NVIDIA A800 GPU.

## Key Experimental Results

### Main Results

| Method | CIFAR-10 (PRA) | CIFAR-100 (PRA) | TinyImageNet (PRA) | CIFAR-10 (PAT) | CIFAR-100 (PAT) | TinyImageNet (PAT) | Avg. |
|------|---------------|----------------|-------------------|---------------|----------------|-------------------|------|
| FedProto | 78.36 | 35.00 | 18.16 | 83.81 | 56.72 | 29.61 | 50.28 |
| FedGH | 78.66 | 40.91 | 25.04 | 85.43 | 58.07 | 31.98 | 53.35 |
| FedTGP | 81.32 | 35.89 | 28.70 | 84.68 | 54.67 | 35.64 | 53.48 |
| Local | 81.20 | 41.57 | 25.81 | 84.68 | 57.96 | 33.02 | 54.04 |
| **FedRE** | **82.60** | **46.36** | **30.48** | **86.20** | **62.56** | **38.52** | **57.79** |

FedRE outperforms all baselines across all settings, surpassing FedGH by 6.54% and FedKD by 6.79% under the TinyImageNet PAT setting.

### Ablation Study

**Communication overhead (CIFAR-100, number of scalars ×10³)**:

| Metric | LG-FedAvg | FedGH | FedKD | FedGen | FedProto | FedMRL | FedRE |
|------|-----------|-------|-------|--------|----------|--------|-------|
| Upload | 513.00 | 257.02 | 4234.28 | 9247.08 | 257.02 | 8863.08 | **5.12** |
| Broadcast | 513.00 | 512.00 | 4234.28 | 513.00 | 512.00 | 8863.08 | 513.00 |

FedRE's upload cost is only 5.12K scalars—less than 2% of FedProto and over 1,700× lower than FedMRL.

**Privacy protection (representation inversion attack, TinyImageNet)**:

| Knowledge Form | PSNR ↓ | MSE ↑ |
|----------|--------|-------|
| Representations (FedAllRep) | 12.89 | 4514.91 |
| Prototypes (FedGH) | 10.25 | 6992.04 |
| Entangled representations (FedRE) | **9.66** | **7781.87** |

The entangled representation achieves the lowest PSNR and highest MSE, yielding reconstructed images from which no meaningful information can be identified.

**RE mechanism comparison (CIFAR-100 PRA)**:

| Mechanism | RSR | VAR | RAR | RSP | VAP | RAP |
|------|-----|-----|-----|-----|-----|-----|
| Accuracy | 40.41 | 44.88 | 43.19 | 43.25 | 46.12 | **46.36** |

RAP (Random Average Prototype) achieves the best performance, as prototypes are more representative than raw representations, and random weighting is more effective than uniform aggregation.

### Key Findings

1. **Entangled representations achieve performance close to uploading all representations**: FedRE (30.48%) vs. FedAllRep (31.20%), with approximately 10× lower communication overhead.
2. **Per-round resampling is critical**: Fixed weights vs. per-round resampling yield 45.84% vs. 46.36% on CIFAR-100, with a larger gap on synthetic datasets (41.50% vs. 62.00%).
3. The additional training cost introduced by RE is negligible: only 0.09 seconds per round on CIFAR-10 (5.69s → 5.78s).
4. The choice of weight distribution (uniform / Laplacian / Gaussian) has minimal impact on performance, demonstrating the flexibility of the framework.
5. FedRE maintains state-of-the-art performance in large-scale settings with 100 clients (participation rates of 10/100 and 20/100).

## Highlights & Insights

- The **entangled representation** is an elegant design that simultaneously addresses three objectives—performance, privacy, and communication—rather than trading off among them as existing methods do.
- The per-round resampling of random weights resembles data augmentation through randomness, preventing overfitting to specific weight configurations by introducing training diversity.
- The entangled label encoding provides "cross-class soft supervision," which bears a conceptual resemblance to label smoothing—yet the randomness here is more fundamental, as label encodings differ entirely across rounds.
- A key distinction from Mixup: FedRE aggregates **all** representations within each client into a single vector (rather than performing pairwise interpolation), and serves an entirely different purpose.

## Limitations & Future Work

1. A rigorous non-convex convergence analysis is absent (acknowledged by the authors as future work).
2. When client data is extremely imbalanced (e.g., a client holds only 1–2 classes), the entangled representation may carry insufficient information.
3. The method has not been evaluated on larger-scale models (e.g., LLMs or ViT-L).
4. The global classifier architecture must be shared across all clients, limiting flexibility in fully heterogeneous settings.
5. The distribution and sampling strategy for random weights may admit better alternatives; current experiments suggest uniform distribution is marginally superior but with negligible margins.

## Related Work & Insights

- Most directly related to FedGH (training a global classifier based on prototypes)—FedRE can be viewed as its natural evolution, extending from single-class prototypes to cross-class entangled representations.
- Orthogonal to FedAvg-family methods (parameter aggregation)—since FedRE cannot directly aggregate parameters due to model heterogeneity, it instead adopts a knowledge distillation perspective.
- The framework offers an instructive perspective on the privacy–efficiency–performance triangle in federated learning: well-designed client knowledge representations can simultaneously advance all three objectives.
- The concept of entangled representations may extend to other settings, such as federated continual learning and federated domain adaptation.

## Rating

- Novelty: ⭐⭐⭐⭐ (The entangled representation concept is novel; despite surface similarities to Mixup, the motivation and implementation differ substantially.)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Systematic analysis of 10 research questions; three-dimensional evaluation of communication, privacy, and performance; 10 heterogeneous architectures.)
- Writing Quality: ⭐⭐⭐⭐ (Q&A-style experimental structure is clear; toy experiments are intuitive.)
- Value: ⭐⭐⭐⭐ (Provides a practical and elegant solution for model-heterogeneous FL.)

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] ProxyFL: A Proxy-Guided Framework for Federated Semi-Supervised Learning](proxyfl_a_proxy-guided_framework_for_federated_semi-supervised_learning.md)
- [\[ICLR 2026\] Toward Enhancing Representation Learning in Federated Multi-Task Settings](../../ICLR2026/ai_safety/toward_enhancing_representation_learning_in_federated_multi-task_settings.md)
- [\[ICLR 2026\] Co-LoRA: Collaborative Model Personalization on Heterogeneous Multi-Modal Clients](../../ICLR2026/ai_safety/co-lora_collaborative_model_personalization_on_heterogeneous_multi-modal_clients.md)
- [\[CVPR 2026\] FecalFed: Privacy-Preserving Poultry Disease Detection via Federated Learning](fecalfed_privacy-preserving_poultry_disease_detection_via_federated_learning.md)
- [\[CVPR 2026\] FedDAP: Domain-Aware Prototype Learning for Federated Learning under Domain Shift](feddap_domain-aware_prototype_learning_for_federated_learning_under_domain_shift.md)

<!-- RELATED:END -->

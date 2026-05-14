---
title: >-
  [Paper Note] LoRA-FAIR: Federated LoRA Fine-Tuning with Aggregation and Initialization Refinement
description: >-
  [ICCV 2025][AI Safety][federated learning] This paper proposes LoRA-FAIR, which introduces a server-side residual correction term $\Delta\mathbf{B}$ to simultaneously address two fundamental challenges in federated LoRA…
tags:
  - "ICCV 2025"
  - "AI Safety"
  - "federated learning"
  - "LoRA"
  - "parameter-efficient fine-tuning"
  - "aggregation bias"
  - "foundation models"
date: 2026-05-08
content_hash: ca571ed75d3aaa1b
---

# LoRA-FAIR: Federated LoRA Fine-Tuning with Aggregation and Initialization Refinement

**Conference**: ICCV 2025  
**arXiv**: [2411.14961](https://arxiv.org/abs/2411.14961)  
**Code**: None  
**Area**: AI Safety / Federated Learning  
**Keywords**: federated learning, LoRA, parameter-efficient fine-tuning, aggregation bias, foundation models

## TL;DR
This paper proposes LoRA-FAIR, which introduces a server-side residual correction term $\Delta\mathbf{B}$ to simultaneously address two fundamental challenges in federated LoRA fine-tuning — server-side aggregation bias and client-side initialization staleness — consistently outperforming existing federated fine-tuning methods on ViT and MLP-Mixer without incurring additional communication overhead.

## Background & Motivation
- **Background**: Full-parameter fine-tuning of large foundation models (e.g., ViT) is computationally prohibitive; LoRA drastically reduces trainable parameters via low-rank decomposition, while federated learning (FL) enables privacy-preserving collaborative training to address data scarcity.
- **Limitations of Prior Work**: Directly combining LoRA with FL (FedIT) faces two fundamental challenges:
    - **Challenge 1 — Server-Side Aggregation Bias**: Independently averaging $\bar{\mathbf{A}}$ and $\bar{\mathbf{B}}$ yields a product $\bar{\mathbf{B}}\bar{\mathbf{A}}$ that does not equal the ideal global update $\sum p_k \mathbf{B}_k \mathbf{A}_k$, since matrix multiplication is not distributive over summation.
    - **Challenge 2 — Client-Side Initialization Staleness**: Methods such as FLoRA reinitialize LoRA modules each round ($\mathbf{A}$ randomly, $\mathbf{B}$ set to zero), causing uninformative gradients at the start of each round ($\partial L/\partial \mathbf{A} \to 0$), which degrades learning efficiency under limited local training steps.
- **Key Challenge**: Existing methods (FFA-LoRA, FLoRA, FlexLoRA) address only one of the two challenges and cannot resolve both simultaneously.
- **Goal**: Design a federated LoRA fine-tuning method that simultaneously resolves aggregation bias and initialization staleness without increasing communication or computational overhead.
- **Key Insight**: Keep $\bar{\mathbf{A}}$ fixed on the server and introduce a residual $\Delta\mathbf{B}$ to correct $\bar{\mathbf{B}}$, such that $(\bar{\mathbf{B}} + \Delta\mathbf{B})\bar{\mathbf{A}} \approx \Delta\mathbf{W}$.
- **Core Idea**: The residual $\Delta\mathbf{B}$ is obtained by minimizing the discrepancy from the ideal global update subject to a regularization term, thereby jointly addressing accurate aggregation and stable initialization.

## Method

### Overall Architecture
Each training round proceeds as follows: clients perform local LoRA training → upload $\mathbf{A}_k, \mathbf{B}_k$ to the server → the server computes weighted averages $\bar{\mathbf{A}}, \bar{\mathbf{B}}$ → computes the ideal global update $\Delta\mathbf{W} = \sum p_k \mathbf{B}_k \mathbf{A}_k$ → optimizes residual $\Delta\mathbf{B}$ such that $(\bar{\mathbf{B}} + \Delta\mathbf{B})\bar{\mathbf{A}} \approx \Delta\mathbf{W}$ → distributes $\bar{\mathbf{A}}$ and $\bar{\mathbf{B}}' = \bar{\mathbf{B}} + \Delta\mathbf{B}$ to clients → clients initialize the next round using $\bar{\mathbf{A}}, \bar{\mathbf{B}}'$.

### Key Designs
1. **Residual Correction**:

    - **Function**: Solves for the residual $\Delta\mathbf{B}$ on the server to correct aggregation bias.
    - **Mechanism**: $\arg\min_{\Delta\mathbf{B}} \underbrace{\mathcal{S}(\Delta\mathbf{W}, (\bar{\mathbf{B}} + \Delta\mathbf{B})\bar{\mathbf{A}})}_{\text{correction term}} + \underbrace{\lambda \|\Delta\mathbf{B}\|}_{\text{regularization}}$, where $\mathcal{S}$ denotes cosine similarity and $\Delta\mathbf{W} = \sum p_k \mathbf{B}_k \mathbf{A}_k$ is the ideal update.
    - **Design Motivation**: The correction term addresses Challenge 1 by aligning the aggregated update with the ideal value; the regularization term addresses Challenge 2 by constraining $\bar{\mathbf{B}}' \approx \bar{\mathbf{B}}$, preserving average information and providing stable initialization.

2. **Avg-Initial Client Initialization Strategy**:

    - **Function**: Clients directly use the server-distributed averaged LoRA modules as initialization for the next round.
    - **Mechanism**: $\mathbf{A}_k \leftarrow \bar{\mathbf{A}}$, $\mathbf{B}_k \leftarrow \bar{\mathbf{B}} + \Delta\mathbf{B}$, with pretrained weights $\mathbf{W}_0$ unchanged.
    - **Design Motivation**: Compared to Re-Initial (random reinitialization) and Local-Initial (using a client's local modules), Avg-Initial balances training continuity with global information fusion.

3. **Residual Placement Selection**:

    - **Function**: Determines whether to apply $\Delta$ to $\mathbf{A}$ or $\mathbf{B}$.
    - **Mechanism**: Ablation experiments show that correcting $\mathbf{B}$ outperforms correcting $\mathbf{A}$.
    - **Design Motivation**: $\mathbf{A}$ primarily captures general information and benefits from a stable averaged update, while $\mathbf{B}$ is better suited to carry the correction signal.

### Loss & Training
- Client optimizer: SGD, learning rate 0.01, batch size 128.
- Server-side residual optimization: SGD solving Eq. 8.
- Regularization weight $\lambda = 0.01$ (a small positive value suffices).
- Default LoRA rank: 16.

## Key Experimental Results

### Main Results

| Dataset | Model | LoRA-FAIR | FedIT | FLoRA | FlexLoRA | Centralized |
|--------|------|-----------|-------|-------|----------|-------------|
| DomainNet (feature non-IID) | ViT | **77.07** | 75.75 | 75.53 | 76.02 | 77.77 |
| DomainNet (feature non-IID) | MLP-Mixer | **65.87** | 64.37 | 64.38 | 64.79 | 66.64 |
| NICO++ (feature non-IID) | ViT | **91.24** | 90.58 | 90.93 | 90.60 | 91.51 |
| NICO++ (feature non-IID) | MLP-Mixer | **83.56** | 82.51 | 82.29 | 83.08 | 84.50 |
| DomainNet (feat.+label non-IID) | ViT | **74.99** | 73.89 | 74.26 | 74.25 | 77.77 |
| NICO++ (feat.+label non-IID) | ViT | **90.04** | 89.48 | 89.60 | 89.65 | 91.51 |

### Ablation Study

| Configuration | DomainNet Mean Accuracy | Description |
|------|-------------------|------|
| $\Delta\mathbf{B}$ (default) | 77.07 | Correct matrix B |
| $\Delta\mathbf{A}$ | 76.42 | Correct matrix A, slightly lower |
| $\Delta\mathbf{A}, \Delta\mathbf{B}$ | 75.55 | Correct both, overfitting |
| $\lambda = 0$ (no regularization) | 73.22 | Aggregation bias removed but initialization unstable |
| $\lambda = 0.01$ (default) | 77.07 | Balances both challenges |

### Key Findings
- LoRA-FAIR consistently outperforms all baselines across all settings, approaching the centralized training upper bound.
- Although FLoRA resolves aggregation bias, its reinitialization strategy causes performance to fall below even the simple FedIT baseline.
- The regularization weight $\lambda$ is critical: at $\lambda=0$, the cosine similarity between $(\bar{\mathbf{B}}+\Delta\mathbf{B})\bar{\mathbf{A}}$ and $\Delta\mathbf{W}$ reaches its maximum (0.9998), yet the similarity between $\bar{\mathbf{B}}'$ and $\bar{\mathbf{B}}$ drops to 0.9715, confirming the importance of initialization stability.
- Communication overhead is identical to FedIT/FlexLoRA and substantially lower than FLoRA.

## Highlights & Insights
- The problem decomposition is incisive: federated LoRA difficulties are cleanly separated into aggregation bias and initialization staleness as two independent yet jointly necessary challenges.
- The solution is minimalist and elegant: a single residual matrix with regularization introduces no additional communication or client-side computation.
- The systematic comparison of initialization strategies (Avg-Initial vs. Re-Initial vs. Local-Initial) offers practical guidance for the FL community.
- The dual role of regularization is cleverly designed: a single term $\lambda\|\Delta\mathbf{B}\|$ simultaneously constrains the correction magnitude to ensure initialization stability and naturally preserves the averaged information.

## Limitations & Future Work
- Computing the ideal global update $\Delta\mathbf{W} = \sum p_k \mathbf{B}_k \mathbf{A}_k$ requires server-side matrix products over all clients, which may become a bottleneck as the number of clients scales.
- Validation is limited to vision models (ViT, MLP-Mixer); experiments on larger-scale foundation models such as LLMs are absent.
- The residual optimization relies on iterative SGD, whose convergence speed and accuracy are sensitive to hyperparameters.
- All clients are assumed to use the same LoRA rank; extension to heterogeneous-rank scenarios is left for future work.
- Experiments are restricted to classification tasks; more complex downstream tasks such as generation and detection remain unexplored.

## Related Work & Insights
- **FedIT**: The earliest attempt to combine LoRA with FedAvg; straightforward but ignores both challenges.
- **FFA-LoRA**: Freezes $\mathbf{A}$ and trains only $\mathbf{B}$ to avoid aggregation bias, at the cost of halving trainable parameters and limiting performance.
- **FLoRA**: Resolves aggregation bias by stacking all clients' LoRA modules, but incurs high communication overhead and suffers from initialization reset.
- **FlexLoRA**: Reconstructs the global update via SVD decomposition; reliable performance but higher computational cost.
- **Insights**: The residual correction paradigm generalizes to other federated learning scenarios where nonlinear relationships must be handled during the aggregation phase.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — Clear problem decomposition; the residual correction combined with dual-purpose regularization is concise and effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Two models × two datasets × two non-IID settings with comprehensive ablations; NLP/LLM experiments are lacking.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem motivation is rigorously articulated; figures and tables are intuitive; the logical chain is complete.
- **Value**: ⭐⭐⭐⭐ — Provides the first unified solution to the two core challenges of federated LoRA fine-tuning, with high practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Fine-Grained DINO Tuning with Dual Supervision for Face Forgery Detection](../../AAAI2026/ai_safety/fine-grained_dino_tuning_with_dual_supervision_for_face_forgery_detection.md)
- [\[ICLR 2026\] Co-LoRA: Collaborative Model Personalization on Heterogeneous Multi-Modal Clients](../../ICLR2026/ai_safety/co-lora_collaborative_model_personalization_on_heterogeneous_multi-modal_clients.md)
- [\[AAAI 2026\] Plug-and-Play Parameter-Efficient Tuning of Embeddings for Federated Recommendation](../../AAAI2026/ai_safety/plug-and-play_parameter-efficient_tuning_of_embeddings_for_federated_recommendat.md)
- [\[ICCV 2025\] FedMeNF: Privacy-Preserving Federated Meta-Learning for Neural Fields](fedmenf_privacy-preserving_federated_meta-learning_for_neural_fields.md)
- [\[ICCV 2025\] Find a Scapegoat: Poisoning Membership Inference Attack and Defense to Federated Learning](find_a_scapegoat_poisoning_membership_inference_attack_and_defense_to_federated_.md)

</div>

<!-- RELATED:END -->

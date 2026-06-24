---
title: >-
  [Paper Note] Clients Collaborate: Flexible Differentially Private Federated Learning with Guaranteed Improvement of Utility-Privacy Trade-off
description: >-
  [ICML 2025][AI Safety][Federated Learning] This paper proposes the FedCEO framework, which applies low-rank tensor proximal optimization on stacked client model parameters at the server side. By leveraging semantic complementarity among different clients, it recovers semantic information corrupted by DP noise, improving the utility-privacy trade-off bound by an order of $O(\sqrt{d})$.
tags:
  - "ICML 2025"
  - "AI Safety"
  - "Federated Learning"
  - "Differential Privacy"
  - "Utility-Privacy Trade-off"
  - "Low-Rank Tensor"
  - "Semantic Complementarity"
date: 2026-05-08
content_hash: 23083571d004490d
---

# Clients Collaborate: Flexible Differentially Private Federated Learning with Guaranteed Improvement of Utility-Privacy Trade-off

**Conference**: ICML 2025  
**arXiv**: [2402.07002](https://arxiv.org/abs/2402.07002)  
**Code**: [https://github.com/6lyc/FedCEO_Collaborate-with-Each-Other](https://github.com/6lyc/FedCEO_Collaborate-with-Each-Other)  
**Area**: AI Security  
**Keywords**: Federated Learning, Differential Privacy, Utility-Privacy Trade-off, Low-Rank Tensor, Semantic Complementarity

## TL;DR
This paper proposes the FedCEO framework, which applies low-rank tensor proximal optimization on stacked client model parameters at the server side. By leveraging semantic complementarity among different clients, it recovers semantic information corrupted by DP noise, improving the utility-privacy trade-off bound by an order of $O(\sqrt{d})$.

## Background & Motivation

**Background**: Differential privacy (DP) is the mainstream technical standard for protecting user privacy in federated learning, achieved by adding random noise to uploaded model updates.

**Limitations of Prior Work**: DP noise randomly degrades the semantic integrity of models, and this degradation accumulates over communication rounds. Since the corrupted semantic information varies across different clients, the global semantic space becomes non-smooth.

**Key Challenge**: Existing improvement methods (such as regularization or personalization) are mostly based on constraining local update magnitudes, failing to exploit collaborative relationships among clients.

**Goal**: How can semantic complementarity among clients be leveraged to recover semantic information destroyed by DP noise?

**Key Insight**: Stacking noisy model parameters from multiple clients into high-order tensors and truncating high-frequency components to smooth the global semantic space.

**Core Idea**: The randomness of DP noise implies that different clients have different parts of their semantics corrupted. Tensor low-rank decomposition can extract shared semantics among clients and eliminate individual noise.

## Method

### Overall Architecture
On top of standard DPFL, FedCEO introduces a tensor low-rank proximal optimization step at the server side:
1. Each client trains locally, adds DP noise, and uploads the noisy model.
2. The server stacks the model parameters of $K$ clients into a third-order tensor.
3. Truncated tensor singular value decomposition (T-tSVD) is performed to smooth the global semantic space.
4. The smoothed parameters are broadcast back to the clients.

### Key Designs

1. **Tensor Low-Rank Proximal Optimization**:

    - **Function**: Stacks the parameter matrices of $K$ clients into a third-order tensor $\mathcal{W} \in \mathbb{R}^{m \times d \times K}$ and performs low-rank proximal optimization.
    - **Mechanism**: Equivalent to truncating high-frequency components in the spectral domain (T-tSVD), preserving shared low-frequency semantic information across clients while removing independent high-frequency noise from each client.
    - **Design Motivation**: DP noise primarily manifests as high-frequency components in the spectral domain; truncation can effectively perform denoising.

2. **Adaptive Rank Control**:

    - **Function**: Dynamically adjusts the truncation rank based on the noise level (privacy budget $\epsilon$) and the training phase.
    - **Mechanism**: More aggressive truncation is applied with higher noise ($\epsilon$ is smaller), while truncation can be relaxed in later training stages as semantics tend to converge.
    - **Design Motivation**: A fixed rank cannot adapt to the requirements of different privacy settings and training stages.

### Loss & Training
- User-level differential privacy: protects the entire dataset of any single client.
- Gradient clipping + Gaussian noise mechanism.
- Server-side low-rank regularization does not affect privacy guarantees (due to post-processing property).

## Key Experimental Results

### Main Results
Test accuracy on CIFAR-10 using an MLP architecture under different privacy budgets:

| Method | $\epsilon=1$ | $\epsilon=2$ | $\epsilon=5$ | $\epsilon=10$ |
|------|-----|-----|-----|------|
| UDP-FedAvg | ~35% | ~42% | ~50% | ~55% |
| CENTAUR | ~40% | ~48% | ~56% | ~60% |
| **FedCEO** | **~48%** | **~55%** | **~62%** | **~66%** |

### Ablation Study

| Configuration | Effect | Description |
|------|------|------|
| No low-rank processing | Significant degradation | Degenerates to standard DPFL |
| Fixed vs. adaptive rank | Adaptive is better | Adapts to different training stages |
| Different client count $K$ | Larger $K$ yields better results | More complementary information |

### Key Findings
- The utility-privacy trade-off bound is improved from the previous SOTA of $O(d)$ to $O(\sqrt{d})$.
- High privacy protection is maintained under DLG gradient inversion attacks.
- Equally effective on more complex architectures such as CNN and ResNet.

## Highlights & Insights
- The perspective of **semantic complementarity among clients** is novel—the randomness of DP noise actually becomes an advantage (different clients have different parts corrupted).
- Tensor low-rank processing operates as server-side post-processing, consuming no additional privacy budget.
- The theoretical bound improvement of $\sqrt{d}$ is highly significant in high-dimensional scenarios (such as large models).

## Limitations & Future Work
- The computational overhead of tensor SVD grows with the number of clients and the parameter dimensions.
- Assuming semantic similarity among clients, the effectiveness may diminish under extremely heterogeneous data distributions.
- Only user-level DP is validated, and the applicability under sample-level DP is not discussed.

## Related Work & Insights
- **vs. CENTAUR/Jain et al.**: They perform SVD independently on each client, whereas FedCEO exploits cross-client tensor structure.
- **vs. PPSGD**: Personalization methods, which are difficult to scale to complex models.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of cross-client tensor low-rank denoising is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple architectures, multiple privacy settings, and attack validation.
- Writing Quality: ⭐⭐⭐⭐ Intuitive visualizations and clear theory.
- Value: ⭐⭐⭐⭐ A practical improvement scheme for DPFL.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Mitigating Privacy-Utility Trade-off in Decentralized Federated Learning via f-Differential Privacy](../../NeurIPS2025/ai_safety/mitigating_privacy-utility_trade-off_in_decentralized_federated_learning_via_f-d.md)
- [\[ICML 2025\] Towards Trustworthy Federated Learning with Untrusted Participants](towards_trustworthy_federated_learning_with_untrusted_participants.md)
- [\[ICML 2025\] Private Model Personalization Revisited](private_model_personalization_revisited.md)
- [\[ICML 2025\] Improving the Variance of Differentially Private Randomized Experiments through Clustering](improving_the_variance_of_differentially_private_randomized_experiments_through_.md)
- [\[ICML 2025\] Theoretically Unmasking Inference Attacks Against LDP-Protected Clients in Federated Vision Models](theoretically_unmasking_inference_attacks_against_ldp-protected_clients_in_feder.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] A Framework for Double-Blind Federated Adaptation of Foundation Models
description: >-
  [ICCV 2025][AI Safety][Federated Learning] This paper proposes BlindFed, a framework that achieves "double-blind" federated adaptation of foundation models through FHE-friendly architectural transformation, two-stage split learning, and privacy-enhancing strategies — keeping the model hidden from data holders and data hidden from the service provider. BlindFed achieves 94.28% accuracy on CIFAR-10, approaching LoRA's 95.92%.
tags:
  - ICCV 2025
  - AI Safety
  - Federated Learning
  - Fully Homomorphic Encryption
  - Foundation Model Adaptation
  - Privacy Preservation
  - Split Learning
date: 2026-05-08
content_hash: 7383e98db3c055c3
---

# A Framework for Double-Blind Federated Adaptation of Foundation Models

**Conference**: ICCV 2025
**arXiv**: [2502.01289](https://arxiv.org/abs/2502.01289)
**Code**: [https://github.com/tnurbek/blindfed](https://github.com/tnurbek/blindfed)
**Area**: AI Security & Privacy
**Keywords**: Federated Learning, Fully Homomorphic Encryption, Foundation Model Adaptation, Privacy Preservation, Split Learning

## TL;DR
This paper proposes BlindFed, a framework that achieves "double-blind" federated adaptation of foundation models through FHE-friendly architectural transformation, two-stage split learning, and privacy-enhancing strategies — keeping the model hidden from data holders and data hidden from the service provider. BlindFed achieves 94.28% accuracy on CIFAR-10, approaching LoRA's 95.92%.

## Background & Motivation
Foundation models (e.g., ViT, CLIP) excel at zero-shot tasks, but adapting them to specific downstream tasks (e.g., medical imaging) still requires task-specific data. In practice, two core tensions arise:

(1) **Data privacy**: Multiple data holders (e.g., hospitals) cannot share data with each other or with the model service provider, as strict privacy regulations govern data flows.

(2) **Model privacy**: The learning service provider (LSP) trains foundation models at great cost; these models are proprietary assets that cannot be shared with data holders.

Traditional federated learning addresses data privacy but requires sending the model to clients for local training, violating model privacy. Parameter-efficient fine-tuning methods such as LoRA still require backpropagation through the model. While fully homomorphic encryption (FHE) can protect data, the complex nonlinear operations in foundation models make encrypted inference extremely challenging.

This paper introduces the concept of "double-blind" federated adaptation: data holders cannot see the foundation model (nor each other's data), and the service provider cannot see the task data. The core idea is to combine FHE-friendly architectural transformation, knowledge distillation, parallel adapters, and secure aggregation to enable model adaptation without backpropagating through the foundation model.

## Method

### Overall Architecture
BlindFed consists of three stages: (1) **FHE-friendly architectural transformation**: replacing nonlinear operations in Transformers with polynomial approximations; (2) **Offline distillation**: the server distills knowledge from the original model into the approximated model using an auxiliary dataset; (3) **Online adaptation**: clients send encrypted data to the server for encrypted inference; parallel adapters and classification heads are trained locally using intermediate representations; a global model is obtained via secure aggregation.

### Key Designs
1. **FHE-Friendly Architectural Transformation**:

    - **Function**: Replace non-FHE-compatible nonlinear operations in Transformers with polynomial approximations.
    - **Mechanism**: The exponential function in Softmax is approximated via Taylor expansion $e^x \approx \sum_{i=0}^d \frac{x^i}{i!}$ ($d=6$); GELU activation is approximated by a quadratic function $\text{GELU}(x) \approx 0.125x^2 + 0.25x + 0.5$; division in LayerNorm and Softmax is approximated using the Goldschmidt algorithm $\frac{1}{x} \approx \prod_{i=0}^d (1+(1-x)^{2^i})$ ($d=7$).
    - **Design Motivation**: FHE schemes (e.g., CKKS) support only polynomial operations; all nonlinear operations must be replaced with polynomial surrogates.

2. **Two-Stage Split Learning**:

    - **Function**: Complete foundation model inference and adaptation without leaking the model or data.
    - **Mechanism**:
        - **Stage 1 (Offline Distillation)**: The server distills the original FM (teacher) into the approximated FM (student) using an auxiliary dataset. The first half of training epochs distills embeddings, attention matrices, and hidden states; the second half distills the prediction layer.
        - **Stage 2 (Online Adaptation)**: Clients encrypt local data and send it to the server. The server performs block-wise encrypted inference $\mathcal{E}(\mathbf{b}_\ell) = \hat{\mathcal{B}}_{\hat{\psi}_\ell}(\mathcal{E}(\mathbf{b}_{\ell-1}))$; each block's output is returned to the client for decryption and re-encryption before being passed to the next block. Upon receiving plaintext intermediate representations, clients locally train parallel adapters. Adapter output: $\mathbf{h}_\ell = g_\ell(\mathbf{b}_\ell + \mathbf{h}_{\ell-1}) + \mathbf{h}_{\ell-1}$, where $g_\ell(\mathbf{z}) = \alpha \mathbf{W}_\ell^u \text{GELU}(\mathbf{W}_\ell^d \mathbf{z})$.
    - **Design Motivation**: Block-wise encrypted inference avoids the multiplicative depth limitations and bootstrapping issues of FHE over deep networks. Parallel adapters (rather than LoRA) are chosen because they do not require backpropagation through the FM.

3. **Model Privacy Enhancement Strategies**:

    - **Function**: Prevent malicious clients from exploiting intermediate representations to mount model extraction attacks.
    - **Mechanism**:
        - **Sample-level permutation**: The server applies a random permutation matrix $\Pi_\ell$ to samples within each batch; clients receive only the permuted batch. The server additionally sends the inverse permutation product of adjacent blocks $\Pi_{\ell-1}^{-1} \cdot \Pi_\ell$, allowing clients to correctly compute adapter outputs without recovering sample correspondence.
        - **Stochastic Block Sampling (SBS)**: Only a subset of block outputs are returned per forward pass, with adjacent blocks avoided (adjacent block features exhibit high similarity and can be exploited). Sampling rule: if block $\ell$ is sampled, block $\ell+1$ is excluded; if block $\ell$ is not sampled, block $\ell+1$ is sampled with 50% probability.
    - **Design Motivation**: Intermediate representations are exposed to clients in plaintext; $(b_{\ell-1}, b_\ell)$ pairs can be exploited for model extraction. Permutation and sampling disrupt this correspondence.

### Loss & Training
- **Distillation stage**: The first half of epochs uses MSE to distill intermediate representations; the second half uses cross-entropy + KL divergence to distill predictions.
- **Adaptation stage**: Cross-entropy loss, SGD optimizer (lr=0.001), 50 communication rounds, learning rate decayed by 0.1 at rounds 25 and 40.
- Secure aggregation implements FedAvg via MPC.

## Key Experimental Results

### Main Results

| Dataset | Method | Double-Blind? | Centralized | Federated (α=100) | Federated (α=1) | Federated (α=0.01) |
|--------|------|-------|--------|-------------|-----------|-------------|
| CIFAR-10 | Full fine-tuning | ✗ | 0.9635 | 0.9759 | 0.9725 | 0.8857 |
| CIFAR-10 | LoRA | ✗ | 0.9592 | 0.9736 | 0.9718 | 0.8979 |
| CIFAR-10 | Linear probing | ✓ | 0.9226 | 0.9203 | 0.9191 | 0.7447 |
| CIFAR-10 | **BlindFed** | **✓** | **0.9428** | **0.9471** | **0.9413** | **0.8540** |
| CIFAR-100 | LoRA | ✗ | 0.8349 | 0.8593 | 0.8568 | 0.7647 |
| CIFAR-100 | **BlindFed** | **✓** | **0.7930** | **0.7929** | **0.7808** | **0.6620** |

### Ablation Study (Scalability and Privacy Enhancement)

| Configuration | K=10 | K=20 | K=50 | Notes |
|------|------|------|------|------|
| Full fine-tuning | 0.9739 | 0.9513 | N/A | Insufficient GPU memory |
| LoRA | 0.9661 | 0.9584 | 0.9482 | — |
| Linear probing | 0.9167 | 0.9142 | 0.9007 | — |
| BlindFed | 0.9446 | 0.9422 | 0.9287 | Smallest accuracy drop as client count increases |
| BlindFed + SBS | 0.9425 | 0.9411 | **0.9388** | SBS incurs negligible accuracy cost |

### Key Findings
- Under double-blind constraints, BlindFed substantially outperforms linear probing (CIFAR-10: 94.28% vs. 92.26%) and approaches non-double-blind LoRA (95.92%).
- SBS has minimal impact on accuracy (occasionally slightly improving it) while meaningfully strengthening model privacy.
- Under extreme data heterogeneity (α=0.01), BlindFed's advantage is more pronounced: 85.40% vs. linear probing's 74.47% on CIFAR-10.
- The auxiliary dataset may be out-of-domain (e.g., Tiny-ImageNet); OOD distillation remains effective across all methods.
- Communication overhead: approximately 17.33 MB per block for encrypted intermediate representations, which is acceptable for foundation model adaptation scenarios.

## Highlights & Insights
- The formulation of "double-blind" privacy constraints is highly practical — in real-world settings, model providers and data holders are genuinely mutually distrusting parties.
- The choice of parallel adapters over LoRA, though seemingly a constraint, is an elegant design decision that perfectly fits the FHE + split learning framework.
- The combination of sample permutation and stochastic block sampling provides effective mitigation against model extraction attacks with negligible accuracy loss, representing a clean and well-motivated design.

## Limitations & Future Work
- High communication cost: each round requires transmitting $N_k \times L \times C$ encrypted intermediate representations.
- Heavy server-side computation: a large volume of matrix operations must be executed in the encrypted domain.
- Only the semi-honest threat model is considered; robustness against malicious adversaries requires further analysis.
- Approximation errors introduced by polynomial substitutions may accumulate in deeper or larger models.

## Related Work & Insights
- **vs. Standard Federated Learning (FedAvg)**: FedAvg requires distributing the full model to clients, violating model privacy; BlindFed avoids model exposure via encrypted inference.
- **vs. Private Inference (PI) Methods**: Methods such as SAL-ViT and Iron focus on privacy during inference; BlindFed addresses bidirectional privacy during training and adaptation.
- **vs. LoRA/PEFT**: These methods require backpropagation through the model and are inapplicable when the model cannot be shared; parallel adapters are a more suitable alternative in this setting.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The double-blind federated adaptation concept is original; the combination of FHE, split learning, and parallel adapters is cleverly designed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 4 datasets, multiple data partitioning strategies, and scalability analysis; experiments on larger foundation models are lacking.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Problem formalization is clear, framework components are well-structured, and security analysis is rigorous.
- **Value**: ⭐⭐⭐⭐ Significant practical relevance for foundation model adaptation in privacy-sensitive domains such as healthcare and finance.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] FedMeNF: Privacy-Preserving Federated Meta-Learning for Neural Fields](fedmenf_privacy-preserving_federated_meta-learning_for_neural_fields.md)
- [\[ICCV 2025\] Find a Scapegoat: Poisoning Membership Inference Attack and Defense to Federated Learning](find_a_scapegoat_poisoning_membership_inference_attack_and_defense_to_federated_.md)
- [\[ICCV 2025\] FedVLA: Federated Vision-Language-Action Learning with Dual Gating Mixture-of-Experts for Robotic Manipulation](fedvla_federated_vision-language-action_learning_with_dual_gating_mixture-of-exp.md)
- [\[ICCV 2025\] LoRA-FAIR: Federated LoRA Fine-Tuning with Aggregation and Initialization Refinement](lora-fair_federated_lora_fine-tuning_with_aggregation_and_initialization_refinem.md)
- [\[ICCV 2025\] Client2Vec: Improving Federated Learning by Distribution Shifts Aware Client Indexing](client2vec_improving_federated_learning_by_distribution_shifts_aware_client_inde.md)

<!-- RELATED:END -->

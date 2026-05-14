---
title: >-
  [Paper Note] SABRE-FL: Selective and Accurate Backdoor Rejection for Federated Prompt Learning
description: >-
  [ICLR2026][LLM Safety][federated learning] This paper is the first to investigate backdoor attack threats in the federated prompt learning (FPL) setting…
tags:
  - "ICLR2026"
  - "LLM Safety"
  - "federated learning"
  - "Prompt Learning"
  - "Backdoor Attack"
  - "CLIP"
  - "Anomaly Detection"
date: 2026-05-08
content_hash: b837e5977396f5f9
---

# SABRE-FL: Selective and Accurate Backdoor Rejection for Federated Prompt Learning

**Conference**: ICLR2026
**arXiv**: [2506.22506](https://arxiv.org/abs/2506.22506)
**Code**: To be released
**Area**: AI Security
**Keywords**: federated learning, Prompt Learning, Backdoor Attack, CLIP, Anomaly Detection

## TL;DR
This paper is the first to investigate backdoor attack threats in the federated prompt learning (FPL) setting, and proposes SABRE-FL — a lightweight server-side defense based on anomaly detection in the embedding space — which effectively filters poisoned prompt updates without accessing clients' raw data.

## Background & Motivation
- **Federated Prompt Learning (FPL)** is an emerging paradigm in which clients optimize only lightweight prompt vectors (with the CLIP backbone frozen) and upload them to a server for aggregation, substantially reducing communication and computation overhead.
- Federated learning is inherently vulnerable to backdoor attacks — malicious clients inject triggers into local data, causing the global model to produce targeted misclassifications on triggered inputs at inference time.
- Existing backdoor attack research focuses on traditional unimodal FL with full-parameter fine-tuning; in the FPL setting, the attack surface is limited to prompt vectors and the image encoder is frozen, leaving both attack feasibility and defense strategies unexplored.
- The paper is motivated by two objectives: **(1)** verifying whether FPL is genuinely vulnerable; and **(2)** designing targeted defenses.

## Core Problem
1. **Attack perspective**: Can a malicious client in FPL successfully implant a backdoor via a learnable imperceptible noise trigger, causing the global prompt learner to misclassify triggered samples at inference while preserving clean sample accuracy?
2. **Defense perspective**: How can the server detect and filter poisoned prompt updates without relying on clients' raw data, labels, or downstream task information?

## Method

### Attack Design
- **Threat model**: Standard FL setting with $N$ clients, of which $m/N$ (default 25%) are controlled by the adversary; malicious clients may modify local training data, add a learnable trigger, and relabel samples to a target class (dirty-label attack).
- **Trigger optimization**: Malicious clients jointly optimize the prompt vector and trigger $t$ during local training, such that the CLIP image embedding of a triggered image $x^\star = x \oplus t$ is directed toward the target class text embedding:
$$\cos(f_{\text{img}}(x^\star), f_{\text{text}}(y_{\text{target}})) > \cos(f_{\text{img}}(x^\star), f_{\text{text}}(y)), \quad \forall y \neq y_{\text{target}}$$
- The trigger is visually imperceptible yet induces a consistent shift in the CLIP embedding space.

### SABRE-FL Defense Framework
- **Core insight**: Although backdoor triggers are invisible at the pixel level, they leave detectable statistical "fingerprints" in the CLIP embedding space — poisoned sample embeddings exhibit a consistent separation from clean ones: $\|z - z^\star\|_2 > \epsilon$.
- **Offline detector training**: Using the task-agnostic auxiliary dataset Caltech-101, clean/poisoned embedding pairs are generated to train a binary classifier $D: \mathbb{R}^d \to \{0, 1\}$.
- **Online filtering**: At each aggregation round, the server computes an average detection score $S_k = \frac{1}{n_k} \sum_j D(z_j^k)$ for the embedding set $\{z_j^k\}$ submitted by each client $C_k$, and applies a rank-based strategy to exclude the $m$ clients with the highest scores.
- **Privacy preservation**: Clients only share CLIP-encoded embeddings (compressed vectors from the frozen encoder); no raw images, labels, or gradients are required.

### Algorithm Overview
1. **Pre-training phase**: Construct a clean/poisoned embedding dataset from auxiliary data and train detector $D$.
2. **Each FL round**: Server distributes global prompt → clients perform local training → clients return prompt and embeddings → server uses $D$ to compute anomaly scores per client → filters top-$m$ suspicious clients → aggregates remaining prompts.

## Key Experimental Results

### Attack Effectiveness (No Defense / FedAvg)

| Dataset | Clean CA | Attacked CA | Backdoor BA |
|---------|----------|-------------|-------------|
| Flowers | 80.9 | 77.9 | 41.7 |
| Pets | 94.5 | 94.2 | 16.3 |
| DTD | 65.2 | 65.6 | 34.8 |
| Aircraft | 32.3 | 32.8 | **93.9** |
| Food101 | 90.7 | 90.0 | 20.6 |

The attack successfully implants backdoors while preserving clean accuracy, with BA reaching 93.9% on Aircraft.

### Defense Comparison (BA across five datasets, lower is better)

| Defense | Flowers | Pets | DTD | Aircraft | Food101 |
|---------|---------|------|-----|----------|---------|
| No Defense | 41.7 | 16.3 | 34.8 | 93.9 | 20.6 |
| Trimmed Mean | 12.3 | 5.6 | 31.0 | 83.1 | 6.4 |
| Median | 10.4 | 5.3 | 28.1 | 79.4 | 5.5 |
| Norm Bounding | 22.0 | 22.5 | 37.5 | 86.2 | 17.2 |
| FLAME | 3.8 | 7.8 | 8.7 | 16.4 | 3.2 |
| **SABRE-FL** | **1.1** | **4.4** | **6.8** | **7.6** | **1.9** |

SABRE-FL achieves the lowest BA across all five datasets while maintaining clean accuracy on par with or better than the no-defense baseline.

### Ablation Study
- **Number of prompt shots**: As shots increase (2→16), BA rises substantially without defense (exceeding 85% on Aircraft and Food101); with SABRE-FL, BA remains below 5% throughout.
- **Malicious client ratio**: At 25% malicious clients, Aircraft BA reaches 93.9%; at 50%+, BA exceeds 80% on most datasets; clean accuracy is nearly unaffected throughout.

## Highlights & Insights
- **First study of backdoor security in FPL**: Fills a gap in multimodal federated prompt learning security research by establishing both an attack baseline and a defense.
- **Elegant defense design**: Exploits the dual nature of "the attack's success signal is the defense's detection signal" — the fact that a trigger can fool the classifier implies that the resulting embedding shift is detectable.
- **Zero data dependency**: The detector is trained offline on an OOD auxiliary set with no need for client data, labels, or task information, resulting in extremely low deployment cost.
- **Strong cross-domain generalization**: A detector trained on Caltech-101 generalizes effectively across five distinct domains: Flowers, DTD, Aircraft, Food101, and Pets.

## Limitations & Future Work
- **Requires knowledge of the malicious client upper bound**: The rank-based filtering assumes a known upper bound $m$ on the number of malicious clients, which may not be available in practice.
- **Only noise triggers are evaluated**: The attack type is limited to learnable noise triggers; defense effectiveness against patch-based triggers, semantic triggers, and other backdoor variants has not been verified.
- **Additional communication overhead**: Compared to pure prompt aggregation in FPL, SABRE-FL requires clients to transmit image embeddings, increasing both communication cost and privacy exposure.
- **Small-scale datasets**: All five fine-grained datasets are relatively small; validation on large-scale benchmarks such as ImageNet is absent.
- **No adaptive attack evaluation**: Scenarios where the adversary is aware of the defense mechanism and mounts adaptive attacks are not considered.

## Related Work & Insights

| Dimension | BadCLIP (CVPR'24) | A3FL / IBA (Traditional FL Backdoor) | SABRE-FL |
|-----------|-------------------|---------------------------------------|----------|
| Setting | Centralized prompt learning | Unimodal FL (full parameters) | Federated prompt learning |
| Attack surface | All training data | Model parameters + data | Prompt vectors only |
| Defense approach | No dedicated defense | Robust aggregation (Trimmed Mean, etc.) | Embedding-space anomaly detection |
| Data dependency | — | Requires validation set | OOD auxiliary set, no client data needed |

The core idea of detecting backdoors in the **representation space** rather than pixel or parameter space has broad applicability and can be extended to other foundation model fine-tuning scenarios (e.g., federated learning with LoRA adapters). The frozen encoder + learnable prompt architecture makes embedding shift a necessary condition for backdoor success; this structural constraint is key to designing efficient defenses. Security research on federated prompt learning remains in its early stages, with adaptive attacks, multi-target attacks, and clean-label attacks warranting further exploration. The fact that a detector trained on OOD data generalizes well suggests that backdoor embedding shifts are a structural byproduct of the attack, offering new insights for backdoor defense in other modalities (NLP, audio). When the malicious client ratio exceeds 50%, BA approaches 100%, highlighting the potential threat of Sybil attacks in FPL.

## Rating
- Novelty: ⭐⭐⭐⭐ (First systematic study of backdoor attack and defense in FPL; well-motivated entry point)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Five datasets + four baselines + ablations, but lacks large-scale validation and adaptive attack evaluation)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure; theory and experiments are tightly integrated)
- Value: ⭐⭐⭐⭐ (Fills an important research gap; defense method is practically deployable)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] SHE-LoRA: Selective Homomorphic Encryption for Federated Tuning with Heterogeneous LoRA](she-lora_selective_homomorphic_encryption_for_federated_tuning_with_heterogeneou.md)
- [\[ICLR 2026\] Resource-Adaptive Federated Text Generation with Differential Privacy](resource-adaptive_federated_text_generation_with_differential_privacy.md)
- [\[ACL 2026\] Adaptive Text Anonymization: Learning Privacy-Utility Trade-offs via Prompt Optimization](../../ACL2026/llm_safety/adaptive_text_anonymization_learning_privacy-utility_trade-offs_via_prompt_optim.md)
- [\[NeurIPS 2025\] FedSVD: Adaptive Orthogonalization for Private Federated Learning with LoRA](../../NeurIPS2025/llm_safety/fedsvd_adaptive_orthogonalization_for_private_federated_learning_with_lora.md)
- [\[ICCV 2025\] Geminio: Language-Guided Gradient Inversion Attacks in Federated Learning](../../ICCV2025/llm_safety/geminio_language-guided_gradient_inversion_attacks_in_federated_learning.md)

</div>

<!-- RELATED:END -->

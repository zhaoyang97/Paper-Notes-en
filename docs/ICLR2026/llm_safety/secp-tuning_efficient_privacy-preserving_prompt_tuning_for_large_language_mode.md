---
title: >-
  [Paper Note] SecP-Tuning: Efficient Privacy-Preserving Prompt Tuning for Large Language Models via MPC
description: >-
  [ICLR 2026][LLM Safety][Privacy-Preserving] This paper proposes SecP-Tuning, the first privacy-preserving prompt tuning framework based on secure multi-party computation (MPC). It eliminates backpropagation overhead via…
tags:
  - "ICLR 2026"
  - "LLM Safety"
  - "Privacy-Preserving"
  - "Secure Multi-Party Computation"
  - "Prompt Tuning"
  - "Forward Tuning"
  - "Random Feature Attention"
date: 2026-05-08
content_hash: a364f228ca03adbf
---

# SecP-Tuning: Efficient Privacy-Preserving Prompt Tuning for Large Language Models via MPC

**Conference**: ICLR 2026
**arXiv**: [2506.15307](https://arxiv.org/abs/2506.15307)  
**Code**: N/A  
**Area**: AI Security
**Keywords**: Privacy-Preserving, Secure Multi-Party Computation, Prompt Tuning, Forward Tuning, Random Feature Attention

## TL;DR

This paper proposes SecP-Tuning, the first privacy-preserving prompt tuning framework based on secure multi-party computation (MPC). It eliminates backpropagation overhead via forward-only tuning and reduces communication complexity by replacing softmax with privacy-preserving random feature attention (RFA), achieving approximately 12–16× speedup and 17–20× reduction in communication volume.

## Background & Motivation

1. **Demand for LLM Adaptation in Privacy-Sensitive Domains**: Healthcare, finance, and government sectors urgently need to adapt LLMs to specialized tasks, yet data in these domains is protected under regulations such as GDPR and HIPAA, precluding direct access.

2. **MPC Offers Cryptographic Privacy Guarantees**: Secure multi-party computation enables multiple parties to jointly compute over their inputs without revealing them, simultaneously protecting both model parameters and training data—a stronger guarantee than the statistical assurances provided by differential privacy.

3. **MPC-Based Fine-Tuning Faces Severe Efficiency Bottlenecks**: A single SFT iteration on RoBERTa_LARGE requires approximately 10 minutes and 970 GB of communication, with backpropagation and the optimizer accounting for 73% of the runtime and softmax attention accounting for 75%.

4. **Backpropagation Contains Numerous MPC-Unfriendly Operations**: Nonlinear operations such as Softmax, GELU, and LayerNorm must be decomposed into approximations built from additions, multiplications, and comparisons under MPC, causing communication rounds and data volume to surge dramatically.

5. **Existing Parameter-Efficient Fine-Tuning Methods Fail to Address the Root Cause**: LoRA and gradient-based prompt tuning reduce the number of updated parameters but still require privacy-preserving computation of backpropagation and softmax, offering no fundamental reduction in MPC communication overhead.

6. **Homomorphic Encryption Cannot Balance Efficiency and Accuracy**: Homomorphic encryption (HE) relies on single-party re-computation and requires costly approximations and re-encryption for nonlinear operations, whereas MPC directly supports complex nonlinear operations through multi-round communication, making it better suited for fine-tuning scenarios.

## Method

### Overall Architecture: SecP-Tuning

- **Function**: Constructs an MPC-based privacy-preserving prompt tuning framework that enables data owners to adapt a model developer's LLM to a target domain via API calls without exposing their private data.
- **Design Motivation**: The communication overhead of gradient-based fine-tuning within MPC is prohibitive; the two primary bottlenecks—backpropagation and softmax—must be eliminated at a fundamental level.
- **Mechanism**: Two core innovations: (1) Forward-only Tuning (FoT) combined with a "Server-Client" architecture to eliminate backpropagation; (2) privacy-preserving Random Feature Attention (RFA) to replace softmax, reducing attention complexity from $O(n^2d)$ to $O(ndr)$.

### Key Design 1: Privacy-Preserving Forward-Only Tuning (FoT)

- **Function**: Updates prompt vectors in a low-dimensional latent space using a gradient-free optimizer (CMA-ES), requiring only forward inference without backpropagation.
- **Design Motivation**: The inversion of Softmax/GELU/LayerNorm during backpropagation is prohibitively expensive under MPC (accounting for 73% of total runtime), and gradient optimizers such as Adam also involve MPC-unfriendly division and square-root operations.
- **Mechanism**: A seven-step "Server-Client" interaction protocol is adopted: (1) the data owner locally initializes prompt embeddings $p$ and concatenates them with private data embeddings $X$; (2) $X$ is secret-shared and distributed to two servers; (3) the two servers execute the MPC protocol interactively to perform privacy-preserving forward inference, producing prediction shares $[Y]$; (4–5) the data owner reconstructs the inference result $Y$; (6) the data owner locally computes the loss $L$ in plaintext; (7) the data owner locally updates the prompt embeddings using CMA-ES. Because loss computation and optimization are performed entirely in plaintext on the data owner's side, the servers never obtain the updated prompt parameters, thereby eliminating the risk of data leakage through model memorization.

### Key Design 2: Privacy-Preserving Random Feature Attention (RFA)

- **Function**: Approximates softmax attention using random Fourier features and introduces an efficient MPC cosine protocol $\Pi_{\text{cosine}}$.
- **Design Motivation**: Softmax involves three MPC-unfriendly nonlinear operations—exponentiation, division, and maximum—and its $O(n^2d)$ complexity grows quadratically with sequence length.
- **Mechanism**: (1) The random feature method approximates $\exp(\mathbf{q}^\top\mathbf{k}/\sigma^2)$ as $\phi(\mathbf{q})^\top\phi(\mathbf{k})$, where $\phi$ involves cosine functions, reducing attention complexity to linear $O(ndr)$; (2) To address the cost of computing cosine functions under MPC, $\Pi_{\text{cosine}}$ is designed by exploiting trigonometric periodicity and the sum-to-product identity: in the offline phase, random values $t$ along with secret shares of $\sin(t)$ and $\cos(t)$ are pre-generated; in the online phase, only one communication round is needed to reconstruct $\delta = (x+t) \bmod \tau$, after which $\cos(x) = \sin(\delta)\sin(t) + \cos(\delta)\cos(t)$ is computed, completing the cosine evaluation in a single round.

## Experiments

### Experimental Setup

- **Model**: RoBERTa_LARGE (24 layers, 1024-dimensional hidden states).
- **Datasets**: SST-2, MRPC, RTE, Yelp Polarity, AG's News (16-shot per class).
- **MPC Backend**: CrypTen framework, 3 A100 GPU servers; LAN (3 Gbps, 0.8 ms) and WAN (100 Mbps/80 ms, 200 Mbps/40 ms).
- **Baselines**: Full-parameter SFT, gradient-based prompt tuning, plaintext FoT.

### Key Experimental Results

| Method | Fwd Time (s) | Bwd Time (s) | Total Time (s) | Comm. (GB) |
|---|---|---|---|---|
| SFT | 216.2 | 554.5 | 651.6 | 970.7 |
| Gradient Prompt Tuning | 273.3 | 605.2 | 882.1 | 1116.2 |
| SecP-Tuning (FoT) | 174.0 | 0.0 | 174.1 | 205.4 |
| **SecP-Tuning (FoT+RFA)** | **54.2** | **0.0** | **55.2** | **56.5** |

| Method | SST-2 Acc | Yelp P. Acc | AG's News Acc | MRPC F1 | RTE Acc | Avg. |
|---|---|---|---|---|---|---|
| SFT | 85.39 | **91.82** | **86.36** | **77.35** | 58.60 | 79.90 |
| Gradient Prompt Tuning | 68.23 | 61.02 | 84.81 | 51.61 | 54.69 | 64.07 |
| FoT + Pretrained Prompt | **89.56** | 91.50 | 81.51 | 75.51 | **77.62** | **83.14** |
| SecP-Tuning | 88.11 | 85.23 | 81.27 | 75.33 | 52.95 | 76.58 |

### Key Findings

1. **Substantial Efficiency Gains**: SecP-Tuning is approximately 12× faster than SFT and 16× faster than gradient prompt tuning in a LAN environment, with communication volume reduced by 17× and 20×, respectively. Backpropagation and optimizer overhead are completely eliminated (0 s, 0 GB).
2. **Acceptable Accuracy**: Under the few-shot setting, SecP-Tuning matches or surpasses gradient prompt tuning on tasks such as SST-2 and MRPC, validating the practical utility of privacy-preserving tuning. On simple sentiment classification (SST-2: 88.11 vs. 68.23), it significantly outperforms gradient prompt tuning.
3. **Only Method Supporting AAS Deployment**: SecP-Tuning is the only approach that supports an "As-A-Service" deployment model—data owners can complete fine-tuning via API calls while the model developer never obtains the updated parameters, eliminating the risk of model memorization attacks.
4. **$\Pi_{\text{cosine}}$ Is Critical for RFA Efficiency**: RFA without the efficient cosine protocol is even slower than vanilla softmax attention for short sequences, demonstrating that the design of $\Pi_{\text{cosine}}$ is essential.

## Highlights & Insights

- SecP-Tuning is the first LLM prompt tuning framework operating under MPC, filling a gap in MPC-based privacy-preserving fine-tuning.
- The "Server-Client" architecture offloads loss computation and optimization to the data owner's local plaintext execution, eliminating backpropagation overhead at the architectural level.
- The privacy-preserving cosine protocol $\Pi_{\text{cosine}}$ achieves single-round communication by exploiting trigonometric identities, serving as the key contribution that makes RFA practically feasible under MPC.
- The framework supports black-box/API-style privacy-preserving tuning, offering superior deployability over all gradient-transmission-based methods.

## Limitations & Future Work

- Validation is limited to RoBERTa_LARGE; scalability to truly large models at the GPT/LLaMA scale remains unverified.
- RFA's approximation of softmax introduces accuracy degradation, with notable gaps on certain tasks (Yelp P.: 85.23 vs. 91.82; RTE: 52.95 vs. 58.60) relative to SFT.
- The semi-honest threat model is a relatively weak security assumption; handling malicious participants would require additional mechanisms such as zero-knowledge proofs, significantly increasing overhead.
- FoT relies on gradient-free optimizers such as CMA-ES, whose convergence degrades in high-dimensional parameter spaces, necessitating random projection for dimensionality reduction.

## Related Work & Insights

| Method | Key Distinction |
|---|---|
| BlindTuner (Panzade et al., 2025) | HE-based privacy-preserving fine-tuning; single-party encryption incurs high overhead and imprecise approximations for nonlinear operations. SecP-Tuning uses MPC to directly support nonlinear operations. |
| PrivTuner (Li et al., 2024b) | Combines LoRA with fully homomorphic encryption, reducing parameters but still requiring HE computation of backpropagation. SecP-Tuning eliminates backpropagation entirely via FoT. |
| DP-based PFT (Wang et al., 2024; Charles et al., 2024) | Differential privacy provides statistical-level guarantees via noise injection ($\varepsilon, \delta$); MPC provides cryptographic-level theoretical guarantees, differing in both the objects protected and the strength of assurance. |

## Rating

| Dimension | Score |
|---|---|
| Novelty | ⭐⭐⭐⭐ |
| Effectiveness | ⭐⭐⭐⭐ |
| Reproducibility | ⭐⭐⭐ |
| Practicality | ⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] FairLLaVA: Fairness-Aware Parameter-Efficient Fine-Tuning for Large Vision-Language Models](../../CVPR2026/llm_safety/fairllava_fairness-aware_parameter-efficient_fine-tuning_for_large_vision-langua.md)
- [\[NeurIPS 2025\] FedRW: Efficient Privacy-Preserving Data Reweighting for Enhancing Federated Learning of Language Models](../../NeurIPS2025/llm_safety/fedrw_efficient_privacy-preserving_data_reweighting_for_enhancing_federated_lear.md)
- [\[ICLR 2026\] Heterogeneous Federated Fine-Tuning with Parallel One-Rank Adaptation](heterogeneous_federated_fine-tuning_with_parallel_one-rank_adaptation.md)
- [\[ICLR 2026\] Measuring Physical-World Privacy Awareness of Large Language Models: An Evaluation Benchmark](measuring_physical-world_privacy_awareness_of_large_language_models_an_evaluatio.md)
- [\[ICLR 2026\] SHE-LoRA: Selective Homomorphic Encryption for Federated Tuning with Heterogeneous LoRA](she-lora_selective_homomorphic_encryption_for_federated_tuning_with_heterogeneou.md)

</div>

<!-- RELATED:END -->

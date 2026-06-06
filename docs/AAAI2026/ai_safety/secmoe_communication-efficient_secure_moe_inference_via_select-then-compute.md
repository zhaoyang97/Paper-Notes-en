---
title: >-
  [Paper Note] SecMoE: Communication-Efficient Secure MoE Inference via Select-Then-Compute
description: >-
  [AAAI 2026][AI Safety][MoE] This paper proposes the SecMoE framework, which efficiently enables sparse MoE inference under two-party secure computation via a Select-Then-Compute paradigm…
tags:
  - "AAAI 2026"
  - "AI Safety"
  - "MoE"
  - "privacy-preserving inference"
  - "secure multi-party computation"
  - "homomorphic encryption"
  - "Select-Then-Compute"
date: 2026-05-08
content_hash: 2ab7cf541246d8dc
---

# SecMoE: Communication-Efficient Secure MoE Inference via Select-Then-Compute

**Conference**: AAAI 2026
**arXiv**: [2601.06790](https://arxiv.org/abs/2601.06790)  
**Code**: Not released  
**Area**: AI Safety / Privacy-Preserving Machine Learning
**Keywords**: MoE, privacy-preserving inference, secure multi-party computation, homomorphic encryption, Select-Then-Compute

## TL;DR

This paper proposes the SecMoE framework, which efficiently enables sparse MoE inference under two-party secure computation via a Select-Then-Compute paradigm, eliminating redundant expert computation and achieving up to 29.8× communication reduction and up to 16.1× end-to-end speedup.

## Background & Motivation

### Core Problem
Privacy-preserving inference for Transformer models is increasingly critical, yet existing secure two-party computation (2-PC) frameworks primarily target small models such as BERT and GPT-2, leaving a gap of roughly two orders of magnitude relative to practically deployed large models. The Mixture of Experts (MoE) architecture, which scales model capacity at low computational cost through sparse activation, is a promising approach to bridging this gap.

### Privacy Leakage Risk
In standard 2-PC protocols, the server holds plaintext weights and computes FFN layers via homomorphic encryption. In the MoE setting, however, if the server learns which expert is activated, it can infer token-level private information about the client's input. This constitutes a novel privacy threat that prior work has not adequately addressed.

### Limitations of the Naïve Approach
The most straightforward protection strategy—evaluating all experts before selection—entirely negates the core computational efficiency advantage of sparse MoE. For a 128-expert model, the naïve scheme requires computing all 128 expert FFNs, incurring prohibitive overhead.

## Method

### Core Paradigm: Select-Then-Compute

The central idea of SecMoE is to decompose secure computation into two stages: a **Selection Phase** and a **Compute Phase**.

1. **Selection Phase**: Unifies multiple computation entries into a common circuit structure, extracts their parameters as candidates, and performs oblivious selection over ciphertext vectors.
2. **Compute Phase**: Executes encrypted computation on only the single selected entry.

This paradigm is applied to both the secure sparse MoE layer and the secure piecewise polynomial evaluation.

### Design 1: Secure Sparse MoE Protocol

The threat model assumes a semi-honest two-party setting where client $C$ holds private inputs and server $S$ holds model weights.

**Selection Phase**:
- The client and server obtain secret-shared top-$k$ expert indices via the $\Pi_{\text{Topk}}$ protocol.
- A one-hot Boolean vector $t^b$ of length $N_{\text{exp}}$ is generated via $\Pi_{\text{onehot}}$.
- The vector is converted to arithmetic form $t^a$ via $\Pi_{\text{B2A}}$, encrypted by the client, and sent to the server.
- The server exploits the communication-free local advantage of homomorphic encryption to compute the encrypted weights of the selected expert:

$$[\![W_r^1]\!] = \sum_{i=0}^{N_{\text{exp}}-1} W_i^1 \cdot [\![t^a]\!]$$

The same procedure is applied to $V_i$ and $W_i^2$, requiring only a single selection vector of length $N_{\text{exp}}$ to be transmitted.

**Compute Phase**:
- The client encrypts its input share $[\![\langle x \rangle_c]\!]$ and sends it to the server.
- The server performs ciphertext–ciphertext matrix multiplications $[\![W_r^1]\!] \cdot [\![x]\!]$ and $[\![V_r]\!] \cdot [\![x]\!]$.
- After GeLU activation and GLU gating, a further ciphertext multiplication $[\![W_r^2]\!] \cdot [\![\text{GLU}]\!]$ is performed.
- A random mask $R$ protects intermediate results, and both parties ultimately obtain output shares.

Key advantage: As the number of experts grows from 32 to 128, SecMoE's computation increases by only 24%, whereas Iron/BumbleBee increases by 178%.

### Design 2: Secure Piecewise Polynomial Selection (Secure GeLU)

The GeLU function is approximated via piecewise quadratic polynomials:

$$\text{GeLU}(x) = \begin{cases} 0 & x \in (-\infty, -5] \\ P_1(x) & x \in (-5, -3] \\ P_2(x) & x \in (-3, -1] \\ P_3(x) & x \in (-1, 1] \\ P_4(x) & x \in (1, 3] \\ x & x \in (3, \infty) \end{cases}$$

**Selection Phase**:
- All piecewise polynomial coefficients are collected into a matrix, with row index $i$ denoting the segment and column index $j$ denoting the coefficient (from highest to lowest degree).
- Lower-degree polynomials are zero-padded to the highest degree.
- Secure comparisons $\Pi_{\text{comp}}\{x < b_i\}$ generate a one-hot segment selector.
- A single masked matrix–vector product retrieves the target coefficient row.

**Compute Phase**:
- The squared input $\langle x^2 \rangle := \Pi_{\text{Mul}}(x, x)$ is computed.
- Quadratic polynomial evaluation using the selected coefficients: $\langle y \rangle = \Pi_{\text{Mul}}(\langle x^2 \rangle, \langle c_r \rangle_0) + \Pi_{\text{Mul}}(\langle x \rangle, \langle c_r \rangle_1) + \langle c_r \rangle_2$
- Maximum absolute error: $1.2 \times 10^{-2}$; mean absolute error: $1.7 \times 10^{-3}$.

**Further optimizations**: Breakpoint comparisons are unified and their results reused to reduce communication rounds; zero entries in the coefficient matrix are exploited to skip $\Pi_{\text{MUX}}$ operations.

## Key Experimental Results

### Experimental Setup
- Setting: ring $\mathbb{Z}_{2^{64}}$, fixed-point precision $s=18$, two nodes (64 vCPU + 128 GB RAM)
- Network: LAN (1 Gbps, 0.5 ms) and WAN (400 Mbps, 4 ms)
- Baselines: Iron (NeurIPS 2022), BumbleBee (NDSS 2025)
- Models: MoE-Small (124M, 8 experts), Switch-Base (0.62B–7B, 8–128 experts)

### Table 1: Runtime Comparison (minutes, 128-expert setting)

| Method | MoE-Small LAN | MoE-Small WAN | Switch-Base LAN | Switch-Base WAN |
|--------|---------------|---------------|-----------------|-----------------|
| Iron | 12.07 (4.7×) | 59.14 (16.1×) | 35.5 (2.9×) | 143.78 (9.7×) |
| BumbleBee | 9.76 (3.8×) | 13.88 (3.8×) | 32.3 (2.6×) | 34.89 (2.3×) |
| **SecMoE** | **2.52** | **3.68** | **12.1** | **14.73** |

### Table 2: Communication Volume Comparison (GB)

| Method | 16 experts | 32 experts | 64 experts | 128 experts |
|--------|-----------|-----------|-----------|------------|
| Iron | 7.13 (8.9×) | 9.44 (11.2×) | 17.19 (21.2×) | 24.17 (29.4×) |
| BumbleBee | 1.42 (1.8×) | 2.04 (2.4×) | 3.37 (4.2×) | 5.81 (7.1×) |
| **SecMoE** | **0.81** | **0.84** | **0.81** | **0.82** |

SecMoE's communication volume is nearly constant with respect to the number of experts (varying from only 0.81 GB at 16 experts to 0.82 GB at 128 experts), whereas Iron's volume grows by 3.4×.

### Accuracy Validation (MoE-Small on GLUE)

| Dataset | Metric | Plaintext Baseline | SecMoE |
|---------|--------|--------------------|--------|
| CoLA | MCC | 41.0 | 41.0 |
| QNLI | ACC | 90.3 | 90.2 |
| RTE | ACC | 69.9 | 70.0 |

Accuracy degradation is ≤0.1%, which is negligible.

## Key Findings

1. **Near-constant communication**: SecMoE's communication volume is essentially independent of the number of experts—a direct consequence of the Select-Then-Compute paradigm, which transmits only a single selection vector and executes computation for a single expert.
2. **Excellent scalability**: A 63× increase in model parameters results in only a 15.2× increase in end-to-end runtime.
3. **Greater advantage on WAN**: In bandwidth-constrained WAN environments, SecMoE's communication savings translate into more pronounced speedups (up to 16.1×), making it well-suited for practical deployment.
4. **GeLU optimization effectiveness**: Under Switch-Base with 128 experts, SecMoE's GeLU protocol is 7.1× faster than BumbleBee with 81% less communication.

## Highlights & Insights

- **First practical secure MoE inference protocol**: Fills the gap in 2-PC secure inference for MoE architectures.
- **Elegant unified abstraction**: The Select-Then-Compute paradigm unifies MoE expert selection and piecewise polynomial evaluation under a single design principle.
- **Lossless accuracy**: Inference accuracy on the GLUE benchmark is virtually identical to plaintext inference.
- **Communication constant in the number of experts**: Breaks the linear scaling bottleneck of existing methods.

## Limitations & Future Work

1. **Semi-honest model only**: The protocol assumes both parties follow the protocol while attempting to learn private information; malicious adversary settings are not considered.
2. **Memory bottleneck**: Configurations with 256+ experts exhaust memory due to model parameter loading and Beaver triple storage.
3. **Top-1 expert restriction**: Experiments only validate $K_{\text{exp}}=1$; multi-expert activation scenarios such as Top-2 are not sufficiently explored.
4. **Limited model scale**: The largest tested model is Switch-Base at 7B parameters; models exceeding tens of billions of parameters are not evaluated.
5. **Softmax not optimized**: High-order Taylor expansion of the exponential function is incompatible with Select-Then-Compute, so the original scheme is retained.

## Related Work & Insights

- **Secure neural network inference**: MiniONN, Gazelle, CrypTFlow2, and others established the foundations of 2-PC secure NN inference.
- **Secure Transformer inference**: Iron first introduced HE into Transformer linear layers; BumbleBee optimized lattice-based additive HE; BOLT and SHAFT improved nonlinear layers and the preprocessing phase, respectively.
- **MoE architectures**: Sparse MoE (Shazeer 2017) and Switch Transformer (Fedus 2022) serve as the model foundations for this work.

## Rating

⭐⭐⭐⭐ (4/5)

- **Novelty**: ⭐⭐⭐⭐ — The Select-Then-Compute paradigm is original and elegantly addresses both privacy and efficiency in MoE inference.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive evaluation across multiple models, expert counts, and LAN/WAN settings; larger models and real-deployment evaluations are lacking.
- **Writing Quality**: ⭐⭐⭐⭐ — Protocol descriptions are rigorous and figures are clear.
- **Value**: ⭐⭐⭐ — Practical applicability is constrained by the semi-honest assumption and memory limitations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] On the (In-)Security of the Shuffling Defense in the Transformer Secure Inference](../../ACL2026/ai_safety/on_the_in-security_of_the_shuffling_defense_in_the_transformer_secure_inference.md)
- [\[CVPR 2026\] Computation and Communication Efficient Federated Unlearning via On-server Gradient Conflict Mitigation and Expression](../../CVPR2026/ai_safety/computation_and_communication_efficient_federated_unlearning_via_on-server_gradi.md)
- [\[AAAI 2026\] Diversifying Counterattacks: Orthogonal Exploration for Robust CLIP Inference](diversifying_counterattacks_orthogonal_exploration_for_robust_clip_inference.md)
- [\[AAAI 2026\] Plug-and-Play Parameter-Efficient Tuning of Embeddings for Federated Recommendation](plug-and-play_parameter-efficient_tuning_of_embeddings_for_federated_recommendat.md)
- [\[AAAI 2026\] Reference Recommendation based Membership Inference Attack against Hybrid-based Recommender Systems](reference_recommendation_based_membership_inference_attack_against_hybrid-based_.md)

</div>

<!-- RELATED:END -->

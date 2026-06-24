---
title: >-
  [Paper Note] SecMoE: Communication-Efficient Secure MoE Inference via Select-Then-Compute
description: >-
  [AAAI 2026][AI Safety][MoE] This paper proposes the SecMoE framework, which efficiently implements sparse MoE inference in secure two-party computation (2-PC) via the Select-Then-Compute paradigm. It avoids redundant expert computation, reducing communication overhead by up to 29.8× and achieving up to 16.1× end-to-end acceleration.
tags:
  - "AAAI 2026"
  - "AI Safety"
  - "MoE"
  - "Privacy-Preserving Inference"
  - "Secure Multi-Party Computation"
  - "Homomorphic Encryption"
  - "Select-Then-Compute"
date: 2026-05-08
content_hash: d229ce265c3fb67c
---

# SecMoE: Communication-Efficient Secure MoE Inference via Select-Then-Compute

**Conference**: AAAI 2026  
**arXiv**: [2601.06790](https://arxiv.org/abs/2601.06790)  
**Code**: Unreleased  
**Area**: AI Safety / Privacy-Preserving Machine Learning  
**Keywords**: MoE, Privacy-Preserving Inference, Secure Multi-Party Computation, Homomorphic Encryption, Select-Then-Compute

## TL;DR

This paper proposes the SecMoE framework, which efficiently implements sparse MoE inference in secure two-party computation (2-PC) via the Select-Then-Compute paradigm. It avoids redundant expert computation, reducing communication overhead by up to 29.8× and achieving up to 16.1× end-to-end acceleration.

## Background & Motivation

### Core Problem
Privacy-preserving inference for Transformer models is increasingly important. However, existing secure two-party computation (2-PC) frameworks primarily target small models like BERT and GPT-2, leaving a hundred-fold gap compared to actual large-scale models. The Mixture of Experts (MoE) architecture, which scales model capacity with low computational overhead through sparse activation, is a potential solution to bridge this gap.

### Privacy Leakage Risks
In standard 2-PC protocols, the server calculates FFN layers via homomorphic encryption using plaintext weights. However, in MoE scenarios, if the server learns which expert is activated, it can infer token-level private information about the client's input. This represents a novel privacy threat that has not been adequately addressed before.

### Limitations of Naive Approaches
The most straightforward protection method is to evaluate all experts before selection, which completely negates the core benefit of sparse MoE—computational efficiency. For instance, in a 128-expert model, the naive approach requires computing the FFNs of all 128 experts, incurring massive overhead.

## Method

### Core Paradigm: Select-Then-Compute

The core idea of SecMoE is to decompose secure computation into two steps: the **Selection Phase** and the **Compute Phase**:

1. **Selection Phase**: Multiple computation entries are unified into the same circuit structure. Parameters of each entry are extracted as candidates, and an oblivious selection is performed using ciphertext vectors.
2. **Compute Phase**: Encrypted computation is performed only on the single selected entry.

This paradigm is applied to both secure sparse MoE layers and secure piecewise polynomial evaluation.

### Design 1: Secure Sparse MoE Protocol

The threat model is a semi-honest two-party setting: the client $C$ holds the private input, and the server $S$ holds the model weights.

**Selection Phase**:
- The client and the server obtain secret-shared indices of the top-k experts using the $\Pi_{\text{Topk}}$ protocol.
- A one-hot boolean vector $t^b$ of length $N_{\text{exp}}$ is generated using the $\Pi_{\text{onehot}}$ protocol.
- The vector is converted to arithmetic form $t^a$ via $\Pi_{\text{B2A}}$, encrypted by the client, and sent to the server.
- Leveraging the communication-free property of local homomorphic encryption, the server computes the encrypted weights of the selected experts:

$$[\![W_r^1]\!] = \sum_{i=0}^{N_{\text{exp}}-1} W_i^1 \cdot [\![t^a]\!]$$

This operation is analogously performed for $V_i$ and $W_i^2$, requiring the transmission of only a single selection vector of length $N_{\text{exp}}$.

**Compute Phase**:
- The client encrypts the input share $[\![\langle x \rangle_c]\!]$ and sends it to the server.
- The server performs ciphertext-ciphertext matrix multiplications $[\![W_r^1]\!] \cdot [\![x]\!]$ and $[\![V_r]\!] \cdot [\![x]\!]$.
- After GeLU activation and GLU gating, another ciphertext multiplication $[\![W_r^2]\!] \cdot [\![\text{GLU}]\!]$ is executed.
- The intermediate results are protected using a random mask $R$, and eventually, both parties obtain their respective output shares.

Key Advantage: When scaling from 32 to 128 experts, the computation for SecMoE increases by only 24%, whereas Iron/BumbleBee increases by 178%.

### Design 2: Secure Piecewise Polynomial Selection (Secure GeLU)

The GeLU function is approximated using piecewise quadratic polynomials:

$$\text{GeLU}(x) = \begin{cases} 0 & x \in (-\infty, -5] \\ P_1(x) & x \in (-5, -3] \\ P_2(x) & x \in (-3, -1] \\ P_3(x) & x \in (-1, 1] \\ P_4(x) & x \in (1, 3] \\ x & x \in (3, \infty) \end{cases}$$

**Selection Phase**:
- All piecewise polynomial coefficients are collected into a matrix, where the row index $i$ represents the interval and the column index $j$ represents the coefficients (from the highest degree to the constant term).
- Lower-degree polynomials are padded with zeros to match the highest degree.
- A one-hot segment selector is generated via secure comparisons $\Pi_{\text{comp}}\{x < b_i\}$.
- The target coefficient row is retrieved using a single masked matrix-vector product.

**Compute Phase**:
- The input self-multiplication is computed: $\langle x^2 \rangle := \Pi_{\text{Mul}}(x, x)$.
- Then, the quadratic polynomial evaluation is executed with the selected coefficients: $\langle y \rangle = \Pi_{\text{Mul}}(\langle x^2 \rangle, \langle c_r \rangle_0) + \Pi_{\text{Mul}}(\langle x \rangle, \langle c_r \rangle_1) + \langle c_r \rangle_2$.
- The maximum absolute error is $1.2 \times 10^{-2}$, with a mean absolute error of $1.7 \times 10^{-3}$.

**Further Optimization**: Unifying breakpoint comparisons and reusing the results reduces the communication rounds for secure comparisons; utilizing zero elements in the coefficient matrix allows skipping $\Pi_{\text{MUX}}$ operations.

## Key Experimental Results

### Experimental Setup
- Environment: $\mathbb{Z}_{2^{64}}$ ring, fixed-point precision $s=18$, dual-node setting (64 vCPUs + 128GB RAM).
- Network: LAN (1Gbps, 0.5ms) and WAN (400Mbps, 4ms).
- Baselines: Iron (NeurIPS 2022), BumbleBee (NDSS 2025).
- Models: MoE-Small (124M, 8 experts), Switch-Base (0.62B-7B, 8-128 experts).

### Table 1: Running Time Comparison (minutes, 128-expert setting)

| Method | MoE-Small LAN | MoE-Small WAN | Switch-Base LAN | Switch-Base WAN |
|------|---------|---------|----------|----------|
| Iron | 12.07 (4.7×) | 59.14 (16.1×) | 35.5 (2.9×) | 143.78 (9.7×) |
| BumbleBee | 9.76 (3.8×) | 13.88 (3.8×) | 32.3 (2.6×) | 34.89 (2.3×) |
| **SecMoE** | **2.52** | **3.68** | **12.1** | **14.73** |

### Table 2: Communication Overhead Comparison (GB)

| Method | 16 Experts | 32 Experts | 64 Experts | 128 Experts |
|------|--------|--------|--------|---------|
| Iron | 7.13 (8.9×) | 9.44 (11.2×) | 17.19 (21.2×) | 24.17 (29.4×) |
| BumbleBee | 1.42 (1.8×) | 2.04 (2.4×) | 3.37 (4.2×) | 5.81 (7.1×) |
| **SecMoE** | **0.81** | **0.84** | **0.81** | **0.82** |

SecMoE's communication overhead barely grows with the number of experts (it only increases from 0.81 GB to 0.82 GB from 16 to 128 experts), while Iron increases by 3.4×.

### Accuracy Validation (MoE-Small on GLUE)

| Dataset | Metric | Plaintext Baseline | SecMoE |
|--------|------|---------|--------|
| CoLA | MCC | 41.0 | 41.0 |
| QNLI | ACC | 90.3 | 90.2 |
| RTE | ACC | 69.9 | 70.0 |

The accuracy loss is $\le0.1\%$, which is negligible.

## Key Findings

1. **Near-Constant Communication**: The communication overhead of SecMoE is almost independent of the number of experts. This is a direct consequence of the Select-Then-Compute paradigm, which only transmits a single selection vector and executes computation for only one expert.
2. **Excellent Scalability**: When scaling model parameters by 63×, the end-to-end running time increases by only 15.2×.
3. **Greater Advantages in WAN**: In bandwidth-constrained WAN environments, the communication savings of SecMoE translate into more significant speedups (up to 16.1×), making it highly suitable for practical deployment.
4. **GeLU Optimization Performance**: Under the Switch-Base 128-expert setting, the GeLU protocol in SecMoE is 7.1× faster than BumbleBee, with an 81% reduction in communication.

## Highlights & Insights

- **First Practical Secure MoE Inference Protocol**: Fills the gap in 2-PC secure inference for MoE architectures.
- **Elegant Unified Abstraction**: Select-Then-Compute unifies both MoE expert selection and piecewise polynomial evaluation into the same paradigm with a clean design.
- **Negligible Accuracy Loss**: Achieves almost identical precision to plaintext inference on the GLUE benchmark.
- **Constant Communication w.r.t. Number of Experts**: Overcomes the bottleneck in existing approaches where communication scales linearly with the number of experts.

## Limitations & Future Work

1. **Semi-Honest Model Limitation**: Assumes both parties follow the protocol but attempt to extract information, without considering malicious adversaries.
2. **Memory Bottleneck**: Models with 256+ experts suffer from out-of-memory (OOM) issues due to loading model parameters and storing Beaver's triples.
3. **Top-1 Expert Restriction**: The experiments only evaluate $K_{\text{exp}}=1$, leaving multi-expert activation scenarios such as Top-2 inadequately explored.
4. **Limited Model Scale**: The largest evaluated model is Switch-Base 7B, without validating models beyond tens of billions of parameters.
5. **Unoptimized Softmax**: The high-order Taylor expansion of exponential functions is not suitable for Select-Then-Compute, thus retaining the original scheme.

## Related Work & Insights

- **Secure Neural Network Inference**: MiniONN, Gazelle, CrypTFlow2, etc., established the foundation for 2-PC secure NN inference.
- **Secure Transformer Inference**: Iron first introduced HE to the linear layers of Transformers; BumbleBee optimized lattice-based additive HE; BOLT and SHAFT improved non-linear layers and preprocessing phases, respectively.
- **MoE Architectures**: Sparse MoE (Shazeer 2017) and Switch Transformer (Fedus 2022) serve as the architectural foundation of this work.

## Rating

⭐⭐⭐⭐ (4/5)

- Novelty: ⭐⭐⭐⭐ — The Select-Then-Compute paradigm is novel, uniformly addressing both privacy and efficiency in MoE.
- Evaluation: ⭐⭐⭐⭐ — Comprehensive settings with multiple models, expert counts, and LAN/WAN conditions, though evaluations on larger models and real-world deployments are missing.
- Writing Quality: ⭐⭐⭐⭐ — Rigorous protocol descriptions with clear illustrations.
- Practicality: ⭐⭐⭐ — The semi-honest assumption and memory limits constrain real-world applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] FuseFSS: Efficient Secure LLM Inference with Function Secret Sharing](../../ICML2026/ai_safety/fusefss_efficient_secure_llm_inference_with_function_secret_sharing.md)
- [\[ICLR 2026\] Secure Outlier-Aware Large Language Model Inference](../../ICLR2026/ai_safety/secure_outlier-aware_large_language_model_inference.md)
- [\[ACL 2026\] On the (In-)Security of the Shuffling Defense in the Transformer Secure Inference](../../ACL2026/ai_safety/on_the_in-security_of_the_shuffling_defense_in_the_transformer_secure_inference.md)
- [\[ICLR 2026\] Get RICH or Die Scaling: Profitably Trading Inference Compute for Robustness](../../ICLR2026/ai_safety/get_rich_or_die_scaling_profitably_trading_inference_compute_for_robustness.md)
- [\[CVPR 2026\] Computation and Communication Efficient Federated Unlearning via On-server Gradient Conflict Mitigation and Expression](../../CVPR2026/ai_safety/computation_and_communication_efficient_federated_unlearning_via_on-server_gradi.md)

</div>

<!-- RELATED:END -->

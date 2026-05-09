---
title: >-
  [Paper Note] SecP-Tuning: Efficient Privacy-Preserving Prompt Tuning for Large Language Models via MPC
description: >-
  [ICLR 2026][LLM Safety][Privacy-preserving] This paper proposes SecP-Tuning, the first MPC-based privacy-preserving prompt tuning framework for LLMs. It eliminates backpropagation overhead via forward-only tuning and replaces softmax attention with a privacy-preserving random feature attention mechanism, achieving 12–16× speedup and 17–20× reduction in communication cost.
tags:
  - ICLR 2026
  - LLM Safety
  - Privacy-preserving
  - Secure Multi-Party Computation
  - Prompt Tuning
  - Large Language Models
  - Random Feature Attention
date: 2026-05-08
content_hash: 8257b269627e955a
---

# SecP-Tuning: Efficient Privacy-Preserving Prompt Tuning for Large Language Models via MPC

**Conference**: ICLR 2026
**arXiv**: [2506.15307](https://arxiv.org/abs/2506.15307)
**Code**: None
**Area**: AI Security
**Keywords**: Privacy-preserving, Secure Multi-Party Computation, Prompt Tuning, Large Language Models, Random Feature Attention

## TL;DR

This paper proposes SecP-Tuning, the first MPC-based privacy-preserving prompt tuning framework for LLMs. It eliminates backpropagation overhead via forward-only tuning and replaces softmax attention with a privacy-preserving random feature attention mechanism, achieving 12–16× speedup and 17–20× reduction in communication cost.

## Background & Motivation

1. **State of the Field**: Deployment of LLMs in privacy-sensitive domains such as healthcare and finance is constrained by data privacy requirements. MPC-based privacy-preserving machine learning can provide theoretical privacy guarantees for both model parameters and data, but has been largely limited to the inference phase.

2. **Limitations of Prior Work**: Directly applying MPC to fine-tune LLMs poses significant efficiency challenges. Performing SFT on RoBERTa-LARGE (24 layers, 1024 hidden dimensions) requires approximately 10 minutes per iteration and 970 GB of communication overhead. Backpropagation and optimizer operations account for 73% of total time, while softmax attention accounts for 75% of forward pass time.

3. **Root Cause**: In MPC settings, numerous nonlinear operations (Softmax, GELU, LayerNorm) must be approximated via combinations of addition, multiplication, and comparison, resulting in extremely poor communication efficiency. Parameter-efficient fine-tuning methods such as LoRA reduce the number of updated parameters but cannot eliminate the MPC communication overhead introduced by backpropagation and softmax.

4. **Paper Goals**: How to achieve efficient and high-performance privacy-preserving domain adaptation of LLMs within an MPC framework?

5. **Starting Point**: The paper combines forward-only tuning (FoT) to eliminate backpropagation, random feature attention (RFA) to replace softmax and reduce attention computation complexity, and a Server-Client architecture to offload MPC-unfriendly operations to the client.

6. **Core Idea**: By employing gradient-free forward-only tuning to avoid MPC overhead from backpropagation, and linearized attention to avoid nonlinear softmax operations, the framework achieves end-to-end efficient privacy-preserving fine-tuning.

## Method

### Overall Architecture

SecP-Tuning adopts a Server-Client architecture with a seven-step workflow: (1) the data owner initializes prompt embeddings and concatenates them with private data; (2) secret shares are generated and distributed to the servers; (3) two non-colluding servers execute privacy-preserving inference via MPC protocols; (4–5) inference result shares are returned to the data owner for reconstruction; (6) the data owner computes the loss locally in plaintext; (7) a gradient-free optimizer (CMA-ES) updates the prompts. This process iterates until convergence.

### Key Designs

**1. Privacy-preserving Forward-only Tuning**

- **Function**: Completely eliminates MPC communication overhead from backpropagation and optimizer operations.
- **Mechanism**: A black-box gradient-free optimizer (CMA-ES) is used to update prompt embeddings via forward passes only. Exploiting the low intrinsic dimensionality of LLM prompts, optimization is performed in a low-dimensional latent space $z \in \mathbb{R}^d$ ($d \ll D$) and mapped to the prompt space via a random projection $A \in \mathbb{R}^{D \times d}$. Loss computation and optimizer updates are performed locally in plaintext by the data owner.
- **Design Motivation**: Backward computation of MPC-unfriendly nonlinear operations during backpropagation accounts for 73% of total time, representing the dominant bottleneck. FoT fundamentally eliminates this requirement.

**2. Privacy-preserving Random Feature Attention (RFA)**

- **Function**: Reduces attention complexity from $O(n^2d)$ to $O(ndr)$, avoiding the exponentiation and max operations in softmax.
- **Mechanism**: Random features are used to approximate the Gaussian kernel: $\exp(\mathbf{x}^\top\mathbf{y}/\sigma^2) \approx \phi(\mathbf{x})^\top\phi(\mathbf{y})$, where $\phi(\mathbf{x}) = \exp(\|\mathbf{x}\|^2/(2\sigma^2))[\varphi(\mathbf{x},\omega_1),...,\varphi(\mathbf{x},\omega_M)]^\top$. To handle the MPC-unfriendly cosine operations in RFA, an efficient privacy-preserving cosine protocol $\Pi_{\text{cosine}}$ is designed, leveraging the sum-to-product trigonometric identity and requiring only one round of communication transmitting $2\ell$ bits.
- **Design Motivation**: Exponentiation, division, and max operations in softmax are extremely costly in MPC and scale quadratically with sequence length.

**3. Secure Cosine Protocol $\Pi_{\text{cosine}}$**

- **Function**: Efficiently and securely computes the cosine function, enabling RFA within MPC.
- **Mechanism**: In the offline phase, shares of a random value $t$ and its $\sin(t)$, $\cos(t)$ are pre-generated. In the online phase, $\delta = (x+t) \mod \tau$ is first reconstructed, and then the trigonometric identity $\cos(x) = \sin(\delta)\sin(t) + \cos(\delta)\cos(t)$ is applied.
- **Design Motivation**: Cosine is an unavoidable nonlinear operation in RFA, necessitating a dedicated efficient MPC protocol.

### Loss & Training

Standard cross-entropy loss is used, computed locally in plaintext by the data owner. The optimizer is CMA-ES (gradient-free), operating in the low-dimensional latent space. An early stopping strategy is applied: training terminates if validation accuracy does not improve for 1,000 steps.

## Key Experimental Results

### Main Results

Efficiency comparison on RoBERTa-LARGE in a LAN setting (3 Gbps, 0.8 ms latency):

| Method | Total Time (s) | Communication (GB) | Speedup | Comm. Reduction |
|--------|---------------|-------------------|---------|-----------------|
| SFT | 651.60 | 970.72 | 1× | 1× |
| Prompt Tuning | 882.08 | 1116.21 | — | — |
| SecP-Tuning (FoT) | 174.14 | 205.36 | 3.7× | 4.7× |
| **SecP-Tuning (FoT+RFA)** | **55.17** | **56.55** | **12×** | **17×** |

Performance comparison (RoBERTa-LARGE, 16-shot):

| Method | SST-2 | Yelp P. | AG's News | MRPC | RTE | Avg. |
|--------|-------|---------|-----------|------|-----|------|
| SFT | 85.39 | **91.82** | **86.36** | **77.35** | 58.60 | 79.90 |
| Prompt Tuning | 68.23 | 61.02 | 84.81 | 51.61 | 54.69 | 64.07 |
| SecP-Tuning | 88.11 | 85.23 | 81.27 | 75.33 | 52.95 | 76.58 |

### Ablation Study

| Configuration | Time (s) | Comm. (GB) | Notes |
|---------------|---------|------------|-------|
| MPC softmax attention | Slowest | Highest | Baseline: $O(n^2)$ complexity |
| RFA (w/o $\Pi_{\text{cosine}}$) | Limited improvement | Limited improvement | Slower than softmax on short sequences (cosine overhead) |
| RFA (w/ $\Pi_{\text{cosine}}$) | **Fastest** | **Lowest** | $\Pi_{\text{cosine}}$ is the key enabler |

### Key Findings

- FoT eliminates backpropagation and optimizer overhead, reducing runtime from 651 s to 174 s (3.7× speedup).
- RFA further accelerates the forward pass by 3.2× (174 s → 55 s), reducing communication from 205 GB to 57 GB.
- $\Pi_{\text{cosine}}$ is critical to making RFA viable in MPC — without it, RFA is even slower than softmax on short sequences.
- SecP-Tuning supports an "API-as-a-Service" deployment model where the server cannot access updated parameters, eliminating the risk of memorization leakage.
- In few-shot settings, performance is competitive with or superior to gradient-based methods (SST-2: 88.11 vs. 68.23 for Prompt Tuning).

## Highlights & Insights

- **Systematic resolution of two major bottlenecks**: FoT addresses backpropagation overhead while RFA addresses attention overhead; the two are complementary.
- **Elegant Server-Client architecture**: Offloading MPC-unfriendly operations to the client simultaneously improves efficiency and strengthens privacy.
- **Theoretical and engineering value of $\Pi_{\text{cosine}}$**: An efficient protocol that achieves secure cosine computation in a single round of communication.
- **Strong practical applicability**: Supports black-box API deployment and is directly deployable.

## Limitations & Future Work

- Validation is limited to RoBERTa-LARGE; the framework has not been extended to larger GPT/LLaMA-scale LLMs.
- RFA's approximation of softmax may degrade performance on certain tasks (e.g., lower scores on RTE).
- The semi-honest threat model assumption is relatively weak; supporting malicious adversaries would require additional zero-knowledge proof overhead.
- Future work could explore extending SecP-Tuning to additional fine-tuning paradigms such as LoRA.

## Related Work & Insights

- HE-based approaches such as BlindTuner incur greater computational overhead and struggle to handle nonlinear operations.
- MPC frameworks such as CrypTen provide the infrastructure upon which this work builds.
- Insight: Privacy-preserving ML should not focus solely on inference; privacy during fine-tuning is equally important.

## Rating

- Novelty: ⭐⭐⭐⭐ First MPC+LLM fine-tuning framework; the combination of FoT and RFA is original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive multi-dimensional evaluation covering efficiency, performance, deployability, and privacy.
- Writing Quality: ⭐⭐⭐⭐ Problem analysis is thorough and system design is clearly structured.
- Value: ⭐⭐⭐⭐ Significant reference value for engineering practice in privacy-preserving LLM fine-tuning.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Measuring Physical-World Privacy Awareness of Large Language Models: An Evaluation Benchmark](measuring_physical-world_privacy_awareness_of_large_language_models_an_evaluatio.md)
- [\[ICLR 2026\] Heterogeneous Federated Fine-Tuning with Parallel One-Rank Adaptation](heterogeneous_federated_fine-tuning_with_parallel_one-rank_adaptation.md)
- [\[ICLR 2026\] SHE-LoRA: Selective Homomorphic Encryption for Federated Tuning with Heterogeneous LoRA](she-lora_selective_homomorphic_encryption_for_federated_tuning_with_heterogeneou.md)
- [\[ICLR 2026\] BiasBusters: Uncovering and Mitigating Tool Selection Bias in Large Language Models](biasbusters_uncovering_and_mitigating_tool_selection_bias_in_large_language_mode.md)
- [\[ICLR 2026\] AudioTrust: Benchmarking the Multifaceted Trustworthiness of Audio Large Language Models](audiotrust_benchmarking_the_multifaceted_trustworthiness_of_audio_large_language.md)

<!-- RELATED:END -->

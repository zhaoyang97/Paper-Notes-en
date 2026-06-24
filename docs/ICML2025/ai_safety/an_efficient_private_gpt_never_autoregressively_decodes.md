---
title: >-
  [Paper Note] An Efficient Private GPT Never Autoregressively Decodes
description: >-
  [ICML 2025][AI Safety][Private Inference] This paper proposes POST (Public decOding and Secure verificationTion), a method that leverages public GPT models to generate draft tokens and securely verifies them using a private model. Exploiting the characteristic that secure decoding latency is insensitive to input length, POST achieves a 2.1× to 6.0× speedup in private inference while maintaining the same privacy guarantees and generation quality as standard secure decoding.
tags:
  - "ICML 2025"
  - "AI Safety"
  - "Private Inference"
  - "Secure Two-Party Computation"
  - "Speculative Decoding"
  - "Homomorphic Encryption"
  - "GPT"
date: 2026-05-08
content_hash: aa143c805fb4310f
---

# An Efficient Private GPT Never Autoregressively Decodes

**Conference**: ICML 2025  
**arXiv**: [2505.15252](https://arxiv.org/abs/2505.15252)  
**Code**: None  
**Area**: AI Safety  
**Keywords**: Private Inference, Secure Two-Party Computation, Speculative Decoding, Homomorphic Encryption, GPT

## TL;DR
This paper proposes POST (Public decOding and Secure verificationTion), a method that leverages public GPT models to generate draft tokens and securely verifies them using a private model. Exploiting the characteristic that secure decoding latency is insensitive to input length, POST achieves a 2.1× to 6.0× speedup in private inference while maintaining the same privacy guarantees and generation quality as standard secure decoding.

## Background & Motivation

**Background**: GPT private inference is achieved through secure two-party computation (2PC) — where the client and server jointly execute inference based on homomorphic encryption (HE) and multi-party computation (MPC). The client only obtains the results without accessing the model weights, and the server learns nothing about the inputs.

**Limitations of Prior Work**: Cryptographic primitives introduce massive computational and communication overhead — HE matrix multiplication for linear layers is computationally intensive, and non-linear layers (GELU, Softmax) require numerous communication rounds. Generating only one token per step in autoregressive decoding but incurring almost fixed security overhead leads to severe resource waste.

**Key Challenge**: The latency of secure decoding is insensitive to the input length — the latency of forwarding 1 token and forwarding 8 tokens is almost identical (only 1.2×), yet standard methods process only 1 token per step.

**Goal**: How to leverage this feature to accelerate secure GPT decoding while guaranteeing the same level of privacy?

**Key Insight**: Borrowing from speculative decoding, a public model can be used to generate multiple draft tokens, which are then securely verified in a single forward pass. Accepting multiple tokens reduces the total number of decoding steps.

**Core Idea**: Public model draft generation + private model secure verification = multi-token processing per step, which exploits the unique property that secure computation latency is insensitive to input length.

## Method

### Overall Architecture
POST is divided into an online phase and an offline phase:
- **Online Phase**: The client autoregressively generates $\gamma$ draft tokens using the public model $\mathcal{M}'_{pub}$ → Both parties securely forward all draft tokens → Secure speculative sampling verification decides acceptance/rejection → Accepted tokens are retained + a bonus token is sampled at the first rejection position.
- **Offline Phase**: The public model is aligned with the distribution of the private model via knowledge distillation to improve the acceptance rate of draft tokens.

### Key Designs

1. **Observation of Latency Insensitivity**:

    - **Function**: Discovers the phenomenon that secure decoding latency is insensitive to input length, providing the theoretical foundation for the proposed method.
    - **Mechanism**: Decomposes the latency into three parts: (a) RTT-based latency, which depends on the number of communication rounds and does not vary with input length; (b) Computation time, where HE's SIMD operations encode 8,192 values, meaning short inputs waste slot capacity; (c) Transmission time, where HE transmission grows sublinearly, and MPC grows linearly but is not the bottleneck.
    - **Design Motivation**: This implies that the cost of processing $\gamma$ tokens at once is close to processing 1 token, making speculative decoding highly beneficial in secure computation scenarios.

2. **Secure Speculative Sampling Protocol**:

    - **Function**: Implements the speculative sampling algorithm under a secure computation framework.
    - **Mechanism**: For each draft token $x_i$, the ratio $p(x_i)/q(x_i)$ is compared with a random number $r$ to decide acceptance/rejection. The key difficulty is that division and sampling are cryptographically unfriendly, requiring specialized optimization.
    - **Design Motivation**: Strict matching would reject semantically equivalent but different tokens. The "soft matching" of speculative sampling improves the acceptance rate while ensuring the output distribution is completely identical to that of the private model.

3. **Knowledge Distillation Alignment**:

    - **Function**: Offline alignment of the public model to the private model to increase the acceptance rate.
    - **Mechanism**: Fine-tunes the public model guided by the output distribution of the private model, making $q(x|\cdot)$ closer to $p(x|\cdot)$.
    - **Design Motivation**: The smaller the discrepancy between the public and private models, the higher the probability of draft acceptance, leading to better speedup performance.

### Loss & Training
- Knowledge distillation uses Kullback–Leibler divergence (KLD) loss to align the public model with the private model.
- The online phase requires no training or fine-tuning of the private model.
- Security Proof: The information obtained by the client in POST is identical to that in standard secure inference.

## Key Experimental Results

### Main Results

| Model Pair (Public → Private) | Speedup | Network Condition |
|---|---|---|
| LLaMA-68M → Vicuna-7B | 2.1×~3.5× | LAN/WAN |
| LLaMA-160M → Vicuna-7B | 2.8×~4.2× | LAN/WAN |
| T5-small → FLAN-T5-XL | 3.2×~5.1× | LAN/WAN |
| T5-base → FLAN-T5-XL | 3.8×~5.5× | LAN/WAN |
| FLAN-T5-small → FLAN-T5-XL | 4.0×~5.8× | LAN/WAN |
| FLAN-T5-base → FLAN-T5-XL | 4.5×~6.0× | LAN/WAN |

### Ablation Study

| Configuration | Speedup | Description |
|------|--------|------|
| POST (w/o KD) | ~2.5× | Base speedup |
| POST + KD | ~4.5× | Knowledge distillation significantly improves acceptance rate |
| Different $\gamma$ values | $\gamma=4\sim8$ is optimal | Too large increases verification overhead |
| Different network conditions | WAN benefits more | WAN has a larger ratio of fixed latency |

### Key Findings
- Stronger public models (e.g., FLAN-T5-base vs. T5-small) lead to greater speedups.
- Speedups are more significant in WAN environments (due to a higher proportion of fixed communication latency).
- The overhead of the secure speculative sampling protocol itself is negligible.
- The output quality is mathematically proven to be identical to standard secure decoding.

## Highlights & Insights
- The observation that **secure computation latency is insensitive to input length** is critical. This is a unique advantage not shared by standard speculative decoding (where latency in conventional inference is nearly linearly correlated with input length), thereby amplifying the benefits of speculative decoding in secure scenarios.
- The method requires no modifications to the private model, offering a plug-and-play solution compatible with existing secure inference frameworks.
- As the capabilities of public models improve (e.g., rapid progress in open-source LLMs), the acceleration gains of POST will naturally increase.

## Limitations & Future Work
- The client still needs to deploy a public model, which imposes certain hardware resource requirements on the client side.
- Knowledge distillation requires an offline training stage.
- Currently only validated on encoder-decoder (T5) and decoder-only (Vicuna/LLaMA) architectures.
- Security against malicious adversaries (non-semi-honest models) is not discussed.

## Related Work & Insights
- **vs. Standard Secure Decoding**: Generates only 1 token per step, whereas POST can generate multiple, yielding a 2.1× to 6.0× speedup.
- **vs. Standard Speculative Decoding**: Latency is nearly linear with input length in standard scenarios but insensitive in secure scenarios, resulting in greater gains for POST.
- **vs. CipherGPT/Ditto**: These works optimize cryptographic protocols, while POST optimizes from the perspective of decoding strategies, making them orthogonal and combinable.

## Rating
- Novelty: ⭐⭐⭐⭐ The observation of latency insensitivity in secure computation is novel and compelling.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple model pairs, diverse network conditions, and complete ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and intuitive illustrations.
- Value: ⭐⭐⭐⭐ A practical acceleration scheme for secure private inference.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Breaking the n^{1.5} Additive Error Barrier for Private and Efficient Graph Sparsification](breaking_the_n15_additive_error_barrier_for_private_and_efficient_graph_sparsifi.md)
- [\[NeurIPS 2025\] DictPFL: Efficient and Private Federated Learning on Encrypted Gradients](../../NeurIPS2025/ai_safety/dictpfl_efficient_and_private_federated_learning_on_encrypted_gradients.md)
- [\[NeurIPS 2025\] Differentially Private Bilevel Optimization: Efficient Algorithms with Near-Optimal Rates](../../NeurIPS2025/ai_safety/differentially_private_bilevel_optimization_efficient_algorithms_with_near-optim.md)
- [\[ICML 2025\] Private Model Personalization Revisited](private_model_personalization_revisited.md)
- [\[ICML 2025\] Faster Rates for Private Adversarial Bandits](faster_rates_for_private_adversarial_bandits.md)

</div>

<!-- RELATED:END -->

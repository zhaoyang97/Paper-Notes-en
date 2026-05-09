---
title: >-
  [Paper Note] SHE-LoRA: Selective Homomorphic Encryption for Federated Tuning with Heterogeneous LoRA
description: >-
  [ICLR 2026][LLM Safety][Federated Learning] This paper proposes SHE-LoRA, which integrates Selective Homomorphic Encryption (SHE) with LoRA for cross-device federated LLM fine-tuning. The framework features sensitivity-based column-level encrypted subset negotiation, column-swap parameter obfuscation, and column-aware adaptive aggregation. It achieves model performance comparable to non-private baselines while reducing communication overhead by 99.71% and encryption time by 99.87%, providing complete resistance against the state-of-the-art gradient inversion attack DAGER.
tags:
  - ICLR 2026
  - LLM Safety
  - Federated Learning
  - Homomorphic Encryption
  - LoRA
  - Privacy Protection
  - Heterogeneous Devices
date: 2026-05-08
content_hash: 14d90c5533e41a7c
---

# SHE-LoRA: Selective Homomorphic Encryption for Federated Tuning with Heterogeneous LoRA

**Conference**: ICLR 2026
**arXiv**: [2505.21051](https://arxiv.org/abs/2505.21051)
**Code**: [GitHub](https://github.com/liyan2015/SHE-LoRA)
**Area**: AI Security / Privacy Protection
**Keywords**: Federated Learning, Homomorphic Encryption, LoRA, Privacy Protection, Heterogeneous Devices

## TL;DR
This paper proposes SHE-LoRA, which integrates Selective Homomorphic Encryption (SHE) with LoRA for cross-device federated LLM fine-tuning. The framework features sensitivity-based column-level encrypted subset negotiation, column-swap parameter obfuscation, and column-aware adaptive aggregation. It achieves model performance comparable to non-private baselines while reducing communication overhead by 99.71% and encryption time by 99.87%, providing complete resistance against the state-of-the-art gradient inversion attack DAGER.

## Background & Motivation

**State of the Field**: Federated fine-tuning of LLMs requires improving domain-specific performance while preserving data privacy. LoRA has become the mainstream choice for federated PEFT due to its efficiency; however, research has shown that transmitted parameters and gradients are vulnerable to gradient inversion attacks (DAGER) that can reconstruct private data.

**Limitations of Prior Work**: (1) DP amplifies noise through LoRA matrix multiplication, degrading model performance; (2) MPC requires complex synchronization protocols ill-suited for heterogeneous devices; (3) Existing SHE methods suffer from two issues — LoRA matrix multiplication causes encrypted position expansion, and merging encrypted subsets from heterogeneous clients leads to ciphertext bloat.

**Root Cause**: In cross-device scenarios, clients differ in hardware capability, data distribution, and encryption budget. Naively applying FedAvg to aggregate A and B matrices separately is mathematically inequivalent to aggregating the BA product, and the union of different encrypted positions across heterogeneous devices during aggregation causes ciphertext bloat.

**Starting Point**: (a) Encrypt only matrix A, which acts directly on user data and is more prone to leakage; (b) Evaluate parameter importance at the column level, where column-wise encryption prevents expansion from matrix multiplication; (c) The server negotiates a global encrypted subset to control ciphertext bloat; (d) Column swapping clusters encrypted and unencrypted parameters to improve efficiency.

**Core Idea**: By combining column-level parameter sensitivity evaluation, global subset negotiation, column-swap obfuscation, and column-aware aggregation, SHE-LoRA achieves strong privacy protection with minimal overhead in heterogeneous federated LoRA fine-tuning.

## Method

### Overall Architecture
A four-step workflow: (1) HE Subset Negotiation — clients evaluate parameter importance and negotiate a global encrypted subset with the server; (2) Selective Encryption — column swapping followed by CKKS batch encryption; (3) Adaptive Aggregation — separate aggregation of plaintext and ciphertext components; (4) Re-parameterization — decryption, SVD decomposition, and merging into new LoRA parameters matching the local rank.

### Key Designs

1. **HE Subset Negotiation Mechanism**:

    - Function: Enables heterogeneous clients to coordinate a globally optimal encrypted column subset.
    - Mechanism: Each client evaluates column-level sensitivity using the Wanda method: $S_j = \sum_k |W_{kj}| \cdot \|x_j\|_2$. Sensitivity rankings are encrypted with OPE and sent to the server. The server maintains a Common list (by frequency) and a Sensitivity list (by sensitivity score), then negotiates a globally affordable subset for each client.
    - Design Motivation: Column-level rather than element-level encryption is adopted because matrix multiplication expands single-element encryption to the entire column. OPE protects sensitive parameter positions from leakage. Negotiation prevents ciphertext bloat.

2. **Column-Swap Parameter Obfuscation and Selective Encryption**:

    - Function: Applies column permutation to matrix A so that encrypted columns are clustered contiguously to the right.
    - Mechanism: The encrypted portion is processed via CKKS block batch encryption. This provides three benefits: (1) batch encryption reduces overhead; (2) unencrypted columns support direct matrix operations; (3) column swapping serves as obfuscation to increase attack difficulty.
    - Design Motivation: Scattered encrypted positions increase partitioning and encryption overhead; clustering enables CKKS vectorization for substantial efficiency gains.

3. **Column-Aware Adaptive Aggregation and Re-parameterization**:

    - Function: Handles aggregation across heterogeneous clients with differing numbers of encrypted columns.
    - Mechanism: For the plaintext part: $\Delta W_i^{plain} = B_i A_i^{plain}$, weighted averaging is performed column-wise, followed by SVD decomposition and slicing according to each client's rank. The ciphertext part is handled analogously. After decryption, clients concatenate $B_g = [B_p, B_c]$ and $A_g = [A_p; A_c]$, then apply SVD to adjust to the local rank.
    - Design Motivation: Computing the full-rank update via multiplication before SVD decomposition ensures mathematical correctness of the aggregated LoRA product, and it is proven that no meaningful model updates are lost.

### Loss & Training
- 50 clients, 200 federated training rounds, Non-IID partitioning with Dirichlet $\alpha = 0.3$
- 4 device types: rank 8–32, encryption budget 0.125%–1.6%
- HE implementation: TenSEAL CKKS, polynomial degree 8192

## Key Experimental Results

### Main Results: Privacy Attack Defense (DAGER Attack, SST2 Dataset)

| Method | B=4 R-1 | B=8 R-1 | B=16 R-1 |
|--------|---------|---------|----------|
| Flex-LoRA (no protection) | 95.18 | 61.14 | 10.27 |
| Flex-LoRA-DP | 86.25 | 80.28 | 68.62 |
| MaskCrypt (equal HE budget) | 89.16 | 61.49 | 10.91 |
| **SHE-LoRA** | **0.72** | **0.98** | **0.0** |

### Ablation Study: Efficiency Comparison (OpenLLaMA-3B)

| Metric | Full Encryption Baseline | MaskCrypt | SHE-LoRA |
|--------|--------------------------|-----------|----------|
| Encryption Time | ~480s | ~50s | ~0.6s |
| Communication Overhead | Highest | Moderate | Lowest (−99.71%) |
| Time Variance | [311s, 653s] | [1.6s, 105s] | Nearly none |

### Key Findings
- **Near-zero encryption suffices for complete defense**: Encrypting only 0.125% of parameters causes DAGER to fail entirely (R-1 = 0).
- **Column swapping is security-critical**: It perturbs the structure of the gradient's orthogonal complement in the LoRA low-rank subspace, causing DAGER's span check to fail.
- **No model performance degradation**: Performance on GLUE/MMLU is comparable to non-private state-of-the-art methods.
- **Mutual information validation**: The Max strategy (prioritizing encryption of the most important parameters) achieves a far faster rate of mutual information decrease than Min or Random strategies.
- **MaskCrypt requires 100× overhead**: To match the security level of SHE-LoRA.

## Highlights & Insights
- **Column-level encryption** precisely addresses the root cause of encryption expansion in LoRA matrix multiplication — encrypting entire columns is both efficient and avoids expansion.
- **The negotiation mechanism** elegantly balances the heterogeneous privacy requirements and encryption capacities of different devices, preventing ciphertext bloat.
- **Column swapping serves a dual purpose**: engineering optimization (batch encryption efficiency) and security enhancement (parameter obfuscation), achieving both goals simultaneously.
- **Security guarantees are theoretically grounded** (Bayesian CRLB) — selectively encrypting the most sensitive parameters maximizes the lower bound on reconstruction error.

## Limitations & Future Work
- Assumes a semi-honest adversary; malicious behaviors such as poisoning and backdoor attacks are not addressed.
- Multi-party HE scenarios require extension to multi-key HE.
- OPE exposes ordering information; stronger alternatives such as ORE or MPC could be substituted.
- More complex generative tasks remain to be explored.

## Related Work & Insights
- **vs. Flex-LoRA**: The best-performing heterogeneous federated LoRA method but offers no privacy protection. SHE-LoRA provides strong privacy while maintaining comparable performance.
- **vs. MaskCrypt**: The state-of-the-art SHE federated method, which does not account for LoRA matrix multiplication expansion or heterogeneous bloat; it requires 100× overhead to match the security level of SHE-LoRA.
- **vs. DP**: DP noise is amplified by matrix multiplication in LoRA, and its defensive effectiveness against DAGER attacks is far inferior to that of SHE.

## Rating
- Novelty: ⭐⭐⭐⭐ First work to integrate SHE with heterogeneous LoRA federated fine-tuning; column-level encryption and negotiation design are innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers multiple models and tasks with comprehensive evaluation across security, efficiency, and performance dimensions.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear with a rigorous motivation chain.
- Value: ⭐⭐⭐⭐ Significant practical implications for real-world deployment of federated LLM fine-tuning.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Heterogeneous Federated Fine-Tuning with Parallel One-Rank Adaptation](heterogeneous_federated_fine-tuning_with_parallel_one-rank_adaptation.md)
- [\[NeurIPS 2025\] Adaptive LoRA Experts Allocation and Selection for Federated Fine-Tuning](../../NeurIPS2025/llm_safety/adaptive_lora_experts_allocation_and_selection_for_federated_fine-tuning.md)
- [\[AAAI 2026\] FedALT: Federated Fine-Tuning through Adaptive Local Training with Rest-of-World LoRA](../../AAAI2026/llm_safety/fedalt_federated_fine-tuning_through_adaptive_local_training_with_rest-of-world_.md)
- [\[NeurIPS 2025\] FedSVD: Adaptive Orthogonalization for Private Federated Learning with LoRA](../../NeurIPS2025/llm_safety/fedsvd_adaptive_orthogonalization_for_private_federated_learning_with_lora.md)
- [\[ICLR 2026\] SABRE-FL: Selective and Accurate Backdoor Rejection for Federated Prompt Learning](sabre-fl_selective_and_accurate_backdoor_rejection_for_federated_prompt_learning.md)

<!-- RELATED:END -->

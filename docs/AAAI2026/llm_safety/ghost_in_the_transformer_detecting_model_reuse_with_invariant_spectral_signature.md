---
title: >-
  [Paper Note] Ghost in the Transformer: Detecting Model Reuse with Invariant Spectral Signatures
description: >-
  [LLM Safety] This paper proposes GhostSpec, a data-free, white-box method that does not modify model behavior. It extracts spectral fingerprints by applying SVD to invariant matrix products of attention weight matrices…
tags:
  - "LLM Safety"
date: 2026-05-08
content_hash: 35f402da980573a0
---

# Ghost in the Transformer: Detecting Model Reuse with Invariant Spectral Signatures

- **Conference**: AAAI 2026
- **arXiv**: [2511.06390](https://arxiv.org/abs/2511.06390)
- **Code**: [DX0369/GhostSpec](https://github.com/DX0369/GhostSpec)
- **Area**: AI Safety / Model Intellectual Property Protection
- **Keywords**: LLM provenance, model fingerprinting, singular value decomposition, attention matrices, spectral invariants

## TL;DR

This paper proposes GhostSpec, a data-free, white-box method that does not modify model behavior. It extracts spectral fingerprints by applying SVD to invariant matrix products of attention weight matrices, enabling robust verification of LLM lineage under fine-tuning, pruning, merging, expansion, and even adversarial transformations.

## Background & Motivation

- Training LLMs is extremely costly. Many developers fine-tune open-source models and release derivatives, most of which comply with open-source licenses. However, some release fine-tuned derivatives while falsely claiming they were trained from scratch (e.g., the Llama3-V incident), constituting intellectual property infringement.
- **Black-box methods** (behavioral fingerprinting, watermarking) are sensitive to decoding randomness, vulnerable to adversarial paraphrasing, and require cooperation from the model creator to embed watermarks.
- Among **white-box methods**: representation-space approaches depend on input data and incur high computational cost; direct weight comparison is fragile after fine-tuning or pruning.
- Prior research has shown that large singular values encode pre-trained knowledge and remain stable during fine-tuning, while fine-tuning primarily affects small singular value directions. This motivates the authors to use the **large singular value spectrum** as a robust model fingerprint.
- An additional challenge is that adversaries can apply permutation or scaling transformations to weights without altering model functionality, yet dramatically shift weight distributions—rendering direct comparison of singular values of individual $W_q/W_k/W_v/W_o$ matrices ineffective.

## Method

### 1. Invariant Spectral Fingerprint Construction

Core insight: The singular values of individual matrices $W_q, W_k, W_v, W_o$ can be altered by permutation/scaling attacks, but the singular value spectra of certain **matrix products** are invariant under functionality-preserving transformations.

For each layer $i$, two invariant matrices are defined:

$$M_{qk}^{(i)} = W_q^{(i)} (W_k^{(i)})^T, \quad M_{vo}^{(i)} = W_v^{(i)} W_o^{(i)}$$

- $M_{qk}$ corresponds to the Q-K interaction in attention score computation.
- $M_{vo}$ corresponds to the value-output projection.

**Invariance proof**: Under an invertible transformation applied to V-O, $\tilde{W}_v = CW_v,\ \tilde{W}_o = W_oC^{-1}$ (functionality preserved), the product $\tilde{W}_v \tilde{W}_o = CW_vW_oC^{-1}$ is similar to the original matrix and thus shares the same singular values. An analogous argument applies to the Q-K transformation.

The fingerprint for each layer is defined as:

$$\mathcal{S}_M^{(i)} = (\mathbf{s}_{qk,M}^{(i)}, \mathbf{s}_{vo,M}^{(i)})$$

where $\mathbf{s}_{p,M}^{(i)} = \text{SVD}(M_{p,M}^{(i)})$. The full model fingerprint $\mathcal{F}_M$ is the sequence of all per-layer fingerprints.

### 2. Dual Similarity Metrics

#### GhostSpec-mse (Fine-Grained Layer-wise Comparison)

For two models A ($N$ layers) and B ($M$ layers), an aggregated distance matrix $D_{\text{avg}} \in \mathbb{R}^{N \times M}$ is constructed:

$$(\text{D}_{\text{avg}})_{ij} = \frac{1}{2} \sum_{p \in \{qk, vo\}} \frac{1}{r_{p,ij}} \|\hat{\mathbf{s}}_{p,A}^{(i)} - \hat{\mathbf{s}}_{p,B}^{(j)}\|_2^2$$

- $r_{p,ij}$ is the minimum effective rank of the two singular value vectors, used for truncation.
- $\hat{\mathbf{s}}$ is truncated and then min-max normalized to $[0,1]$.

The **POSA algorithm** (Penalty-based Optimal Spectral Alignment) is applied to find the optimal alignment path in the distance matrix, handling models of differing depths. An inverse sigmoid then converts the path-average MSE into a similarity score:

$$\text{Sim}_{\text{MSE}}(A,B) = 1 - \frac{1}{1+e^{-k(d_{\text{path}} - \tau)}}$$

#### GhostSpec-corr (Lightweight Trend Correlation)

- For each layer and each invariant component, the mean of the top-K normalized singular values is computed, forming trend sequences $\boldsymbol{\mu}_{qk},\ \boldsymbol{\mu}_{vo}$.
- A dynamic sequence alignment algorithm matches sequences of different lengths.
- The aligned sequences are concatenated, and **distance correlation** is computed as the similarity score.
- Derived models exhibit highly correlated trends, while unrelated models show divergent trends.

### 3. POSA Alignment Algorithm

- Addresses layer alignment across models with different depths (e.g., structured pruning, layer expansion).
- Input: distance matrix $D \in \mathbb{R}^{N \times M}$ ($N \leq M$) and gap penalty $\rho$.
- Dynamic programming finds the minimum-cost monotonic alignment path; skipped layers incur penalty $\rho$.
- Output: average MSE distance along the optimal path.

## Key Experimental Results

### Experimental Setup

- Llama-2-7b and Mistral-7B are used as base models; a dataset of 55 model pairs is constructed.
- Transformation types: fine-tuning, unstructured pruning (30%/50%/70%), structured pruning, model merging, MoE expansion, and permutation/scaling adversarial attacks.
- Baselines: QueRE (black-box), Logits (black-box), REEF (white-box, data-dependent), PCS (white-box, data-free).

### Table 1: Comprehensive Comparison on Llama-2-7b

| Method | Data Required | Vicuna FT | Llemma FT | Scaling Attack | Permutation Attack | Pruning 50% | Pruning 70% |
|--------|--------------|-----------|-----------|---------------|-------------------|-------------|-------------|
| QueRE | Yes | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| REEF | Yes | 0.999 | 0.998 | 1.000 | 1.000 | 0.997 | 0.995 |
| PCS | No | 0.999 | 0.505 | 0.597 | 0.386 | 0.906 | 0.783 |
| GhostSpec-mse | No | 0.976 | 0.953 | 0.976 | 0.976 | 0.973 | 0.965 |
| GhostSpec-corr | No | 0.999 | 0.760 | 1.000 | 1.000 | 0.897 | 0.705 |

### Table 2: Structured Pruning, Merging, and Unrelated Models

| Method | Sheared-1.3B | Sheared-2.7B | SLERP Merge | MoE Expansion | Qwen2.5 (unrelated↓) | OPT (unrelated↓) |
|--------|-------------|-------------|-------------|--------------|----------------------|------------------|
| PCS | 0.000 | 0.000 | 0.999 | 0.020 | 0.000 | 0.000 |
| GhostSpec-mse | 0.889 | 0.905 | 0.976 | 0.976 | 0.000 | 0.503 |
| GhostSpec-corr | 0.940 | 0.941 | 1.000 | 1.000 | 0.294 | 0.342 |

### Maximum F1 Scores

- **GhostSpec-mse: F1 = 0.9867** (best)
- **GhostSpec-corr: F1 = 0.9730**
- REEF: 0.9474, QueRE: 0.8649, Logits: 0.8378, PCS: 0.7945

## Key Findings

1. **Spectral stability**: Large singular values encode pre-trained knowledge and remain stable after fine-tuning, which only affects small singular value directions—making spectral fingerprints robust to fine-tuning.
2. **Invariant matrix products**: The singular values of $W_qW_k^T$ and $W_vW_o$ are strictly invariant under permutation/scaling attacks, providing a theoretical guarantee of fingerprint robustness.
3. **Adversarial erasure is difficult**: Attempts to erase the fingerprint using a joint loss (task loss + spectral divergence) show that it is difficult to significantly alter spectral features without substantially degrading model performance.
4. **Case study**: GhostSpec identifies a high degree of similarity between Pangu-Pro-MoE and the Qwen2.5-14B family, providing quantitative evidence in a lineage dispute.
5. **MLP modules are also effective**: Singular values of MLP up_proj/down_proj projections can likewise distinguish derived from unrelated models, though at higher computational cost and with lower robustness to MoE expansion.

## Highlights & Insights

- **Completely data-free**: No input samples are required; the method relies purely on static weight analysis with minimal computational overhead.
- **Theoretical guarantees**: Starting from the mathematical definition of functionality-preserving transformations, the paper rigorously proves the invariance of the proposed fingerprints.
- **POSA algorithm**: Elegantly resolves layer alignment across models of different depths, enabling the method to handle structured pruning and model expansion scenarios.
- **High practical utility**: Directly applicable in the open-weight ecosystem with publicly available code, well-suited for real-world IP auditing.
- **Complementary dual metrics**: MSE captures fine-grained differences while corr captures macro-level trends; together they cover a broader range of scenarios.

## Limitations & Future Work

- **White-box only**: Full access to model weights is required; the method cannot be applied to closed-source, API-only models.
- **False positives on OPT**: GhostSpec-mse assigns an intermediate score of 0.503 to some unrelated models (e.g., OPT-6.7b), introducing a risk of false positives.
- **Threshold sensitivity**: The threshold $\tau$ and slope $k$ in the sigmoid conversion require empirical tuning and may need recalibration for different datasets.
- **MoE architectures**: Although the POSA algorithm handles depth differences, robustness has not been thoroughly verified for architectural variants with substantially restructured attention layers (e.g., expert-level reconstruction).
- **Limited experimental scale**: A dataset of 55 model pairs is relatively small; evaluations at larger scale (e.g., hundreds of model pairs) would be more convincing.
- **Cross-modal scenarios not explored**: The method is designed for text-only LLMs; its applicability to multimodal models remains unknown.

## Related Work & Insights

- **Behavioral fingerprinting**: LLMMap and model output style analysis are passive but sensitive to decoding randomness.
- **Watermarking methods**: InstructMark and KGW watermarking require the model creator's cooperation for embedding.
- **Representation-space methods**: REEF (CKA similarity) and gradient statistics are effective but data-dependent.
- **Direct weight analysis**: HuRef/PCS (invariant submatrices) and intrinsic dimensionality analysis.
- **Random matrix theory**: Staats et al. show that Marchenko–Pastur deviations in large singular values encode model identity.

## Rating

⭐⭐⭐⭐ (4/5)

The problem is important and practically relevant; the method offers mathematical guarantees and a clean implementation. Theory and experiments are tightly integrated, and the invariance proofs are clear. Points are deducted primarily for the limited experimental scale, insufficient discriminative power in certain scenarios (e.g., OPT), and the white-box requirement, which may constrain applicability in real-world copyright disputes.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Multi-Faceted Attack: Exposing Cross-Model Vulnerabilities in Defense-Equipped Vision-Language Models](multi-faceted_attack_exposing_cross-model_vulnerabilities_in_defense-equipped_vi.md)
- [\[ICLR 2026\] Unlearning Evaluation through Subset Statistical Independence](../../ICLR2026/llm_safety/unlearning_evaluation_through_subset_statistical_independence.md)
- [\[AAAI 2026\] Hallucination Stations: On Some Basic Limitations of Transformer-Based Language Models](hallucination_stations_on_some_basic_limitations_of_transformer-based_language_m.md)
- [\[ACL 2026\] FaithLens: Detecting and Explaining Faithfulness Hallucination](../../ACL2026/llm_safety/faithlens_detecting_and_explaining_faithfulness_hallucination.md)
- [\[NeurIPS 2025\] TRUST -- Transformer-Driven U-Net for Sparse Target Recovery](../../NeurIPS2025/llm_safety/trust_--_transformer-driven_u-net_for_sparse_target_recovery.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] FedRW: Efficient Privacy-Preserving Data Reweighting for Enhancing Federated Learning of Language Models
description: >-
  [NeurIPS 2025][LLM Safety][Federated Learning] FedRW proposes the first privacy-preserving soft deduplication framework for federated learning that requires no trusted third party. By leveraging secure multi-party comput…
tags:
  - "NeurIPS 2025"
  - "LLM Safety"
  - "Federated Learning"
  - "Privacy Preservation"
  - "Data Deduplication"
  - "Sample Weighting"
  - "Secure Multi-Party Computation"
date: 2026-05-08
content_hash: dd2c8fcff5f83152
---

# FedRW: Efficient Privacy-Preserving Data Reweighting for Enhancing Federated Learning of Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2511.07505](https://arxiv.org/abs/2511.07505)
**Code**: None
**Area**: AI Security
**Keywords**: Federated Learning, Privacy Preservation, Data Deduplication, Sample Weighting, Secure Multi-Party Computation

## TL;DR
FedRW proposes the first privacy-preserving soft deduplication framework for federated learning that requires no trusted third party. By leveraging secure multi-party computation to obtain global sample frequencies and performing frequency-aware sample reweighting, it achieves up to 28.78× preprocessing speedup and approximately 11.42% improvement in perplexity over prior methods.

## Background & Motivation

**Background**: Data duplication in large-scale corpora severely degrades LLM performance and privacy guarantees, making deduplication a standard preprocessing step in training pipelines. Deduplication approaches fall into two categories: hard deduplication (direct removal) and soft deduplication (reweighting).

**Limitations of Prior Work**: In federated learning, privacy constraints prevent direct data sharing, making global deduplication challenging. The current SOTA method EP-MPD adopts encrypted hard deduplication but suffers from three issues: (1) hard deletion may discard informative samples; (2) multi-round key negotiation introduces substantial computational and communication overhead; (3) it relies on a trusted third party.

**Key Challenge**: Local deduplication cannot detect cross-client duplicates, while global deduplication is restricted by privacy constraints. Existing methods struggle to balance privacy preservation with data quality.

**Goal**: Design a privacy-preserving soft deduplication framework that requires no trusted third party, replacing hard deletion with frequency-aware reweighting.

**Key Insight**: Sample weights are set as inverse functions of global frequency, with frequency information obtained via pairwise Private Set Intersection (PSI) protocols.

**Core Idea**: Obtain global sample frequencies via secure multi-party computation and apply logarithmic inverse reweighting in place of hard deletion, simultaneously preserving privacy and data diversity.

## Method

### Overall Architecture

FedRW consists of three stages: (1) the PPMPR protocol, in which each client obtains the global frequency of its local samples through pairwise secure computation; (2) parallel orchestration, which reduces $O(n^2)$ pairwise interactions to $O(2^{\lceil\log_2 n\rceil})$; and (3) enhanced training, where frequency-weighted loss is used for federated LLM training.

### Key Designs

1. **PPMPR Protocol (Privacy-Preserving Multi-Party Reweighting)**:

    - Function: Enables each client to obtain the global occurrence frequency of its local samples without exposing raw data.
    - Mechanism: Decomposes global frequency estimation into $\binom{n}{2}$ pairwise secure computations (Π₂PC). In each Π₂PC, two clients $P_i, P_j$ compute the data intersection $\mathcal{I}$ via Private Set Intersection (PSI) and then exchange local frequency counts for samples in the intersection.
    - Functional definition: $f_{\text{PPMPR}}(X_1,...,X_n) \to (W_1,...,W_n)$, mapping each client's data to a weight vector.
    - Design Motivation: Avoids reliance on a trusted third party; each step leaks only intersection information and frequency counts.

2. **Parallel Orchestration**:

    - Function: Optimizes $O(n^2)$ sequentially executed Π₂PC operations into parallel execution.
    - Mechanism: Employs hierarchical merging via a binary tree structure. At each level, non-overlapping client pairs execute Π₂PC concurrently. A pairing matrix $\mathcal{M}_k$ is constructed using circular left-shift $\text{RotL}(\vec{b}, k)$ to generate conflict-free pairing schedules.
    - Complexity: Reduced from $O(n^2)$ to $O(2^{\lceil\log_2 n\rceil} - 1)$, achieving 4.09–28.78× speedup for $n=50$ clients.

3. **Frequency-Aware Weighted Training**:

    - Function: Weights the loss contribution of training samples according to their global frequency.
    - Weight formula: $\vec{\mathcal{W}} = \frac{1}{\ln(\vec{\mathcal{C}} + \vec{1}) + \vec{\varepsilon}}$, where $\vec{\mathcal{C}}$ denotes the global frequency vector.
    - Weighted loss: $\mathcal{L}_{\text{batch}} = \frac{\sum_{i=1}^B \vec{\mathcal{W}}_i \cdot \ell_i^{(t)}}{\sum_{i=1}^B \vec{\mathcal{W}}_i}$
    - Design Motivation: The logarithmic function provides smooth weight decay, avoiding information loss caused by hard thresholds; frequent samples are down-weighted but not entirely excluded, and moderate redundancy aids generalization.

### Loss & Training
Model updates follow the standard FedAvg aggregation; the reweighting mechanism operates solely at the loss level and is non-intrusive to the training framework.

## Key Experimental Results

### Main Results: GPT-2 Large, 30% Duplication Rate

| Dataset | Raw Data | Baseline (EP-MPD) | FedRW | Relative PPL Improvement |
|--------|----------|---------|-------|---------|
| Haiku | 3.26 | 2.89 | **2.56** | 11.42% |
| Rotten Tomatoes | 2.65 | 2.21 | **1.61** | 27.15% |
| Short Jokes | 4.11 | 3.79 | **3.15** | 16.89% |
| Sonnets | 4.39 | 4.35 | **4.07** | 6.44% |

### Preprocessing Efficiency Comparison

| Number of Clients | EP-MPD Time | PPMPR Time | Speedup |
|-----------|-----------|-----------|------|
| 10 | ~18s | ~1s | 17.61× |
| 30 | ~120s | ~5s | ~24× |
| 50 | ~300s | ~10s | 28.78× |

### Ablation Study: Non-IID Settings (Qwen3-0.6B, Rotten Tomatoes)

| Configuration | Baseline PPL | FedRW PPL | Notes |
|------|-------------|----------|------|
| IID | 1.71 | **1.59** | Standard setting |
| Quantity Skew | 2.02 | **1.96** | Imbalanced data volumes |
| Label Skew | 2.44 | **1.66** | Skewed label distribution; largest improvement |
| Feature Skew | 3.43 | **2.70** | Heterogeneous data types; remains consistently effective |

### Key Findings
- FedRW consistently outperforms the hard deduplication baseline across all datasets and model configurations.
- Improvements are more pronounced on datasets with strict literary structure (Sonnets, Haiku), indicating that redundancy has a greater impact on structured text.
- FedRW's advantage is amplified under Non-IID settings, particularly under Label Skew.
- Performance gains scale with model size: a relative improvement of 26.57% is observed on the Twitter dataset with Qwen2.5-7B.

## Highlights & Insights
- **Soft vs. Hard Deduplication**: The core innovation lies in replacing deletion with reweighting. The logarithmic inverse function $1/\ln(\text{freq}+1)$ is an elegant design—high-frequency samples are gently down-weighted while moderate redundancy is retained to enhance generalization.
- **No Trusted Third Party Required**: Built entirely on pairwise PSI, offering stronger security guarantees and greater deployment flexibility.
- **Parallel Orchestration**: The binary-tree-based client pairing strategy is concise and efficient, and is directly reusable in other federated secure computation scenarios.

## Limitations & Future Work
- Validation is limited to text data; extension to multimodal federated learning remains unexplored.
- The choice of the logarithmic weighting function lacks theoretical optimality guarantees, and superior weighting strategies may exist.
- The PSI protocol provides security guarantees only under the semi-honest model; malicious client scenarios are not addressed.
- Duplication is simulated via artificial injection, which may not reflect the natural redundancy distribution in real-world settings.

## Related Work & Insights
- **vs. EP-MPD (Abadi et al., 2024)**: EP-MPD employs hard deduplication with a trusted third party; FedRW employs soft deduplication without a third party, surpassing EP-MPD in both efficiency and model performance.
- **vs. SoftDedup / DoReMi**: These are centralized soft deduplication methods that cannot be directly applied in privacy-preserving federated settings. FedRW transfers the soft deduplication paradigm to the federated environment.
- **vs. FedAvg**: FedRW is fully compatible with the FedAvg aggregation framework, introducing reweighting solely at the loss level with minimal integration overhead.

## Rating
- Novelty: ⭐⭐⭐⭐ First federated soft deduplication framework; however, the core techniques (PSI + weighted loss) are combinations of existing components.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers multiple datasets, models (GPT-2 to Qwen3/Llama), settings (IID/Non-IID), and dual evaluation of efficiency and performance.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and well-specified protocol descriptions, though notation is occasionally dense.
- Value: ⭐⭐⭐⭐ Addresses a practical pain point in federated LLM training with a broadly applicable framework design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] SecP-Tuning: Efficient Privacy-Preserving Prompt Tuning for Large Language Models via MPC](../../ICLR2026/llm_safety/secp-tuning_efficient_privacy-preserving_prompt_tuning_for_large_language_mode.md)
- [\[NeurIPS 2025\] CryptoMoE: Privacy-Preserving and Scalable Mixture of Experts Inference via Balanced Expert Routing](cryptomoe_privacy-preserving_and_scalable_mixture_of_experts_inference_via_balan.md)
- [\[NeurIPS 2025\] FedSVD: Adaptive Orthogonalization for Private Federated Learning with LoRA](fedsvd_adaptive_orthogonalization_for_private_federated_learning_with_lora.md)
- [\[NeurIPS 2025\] MPCache: MPC-Friendly KV Cache Eviction for Efficient Private LLM Inference](mpcache_mpc-friendly_kv_cache_eviction_for_efficient_private_llm_inference.md)
- [\[NeurIPS 2025\] HealthSLM-Bench: Benchmarking Small Language Models for Mobile and Wearable Healthcare Monitoring](healthslm-bench_benchmarking_small_language_models_for_mobile_and_wearable_healt.md)

</div>

<!-- RELATED:END -->

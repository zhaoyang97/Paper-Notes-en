---
title: >-
  [Paper Note] AWM: Accurate Weight-Matrix Fingerprint for Large Language Models
description: >-
  [ICLR 2026][Reinforcement Learning][model fingerprinting] AWM is a training-free LLM weight-matrix fingerprinting method that recovers permutation and sign-flip transformations in the embedding layer via the Linear Assig…
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "model fingerprinting"
  - "intellectual property"
  - "weight manipulation"
  - "CKA"
  - "linear assignment problem"
date: 2026-05-08
content_hash: dbb9410c8e58d857
---

# AWM: Accurate Weight-Matrix Fingerprint for Large Language Models

**Conference**: ICLR 2026
**arXiv**: [2510.06738](https://arxiv.org/abs/2510.06738)  
**Code**: [https://github.com/LUMIA-Group/AWM](https://github.com/LUMIA-Group/AWM)  
**Area**: Reinforcement Learning
**Keywords**: model fingerprinting, intellectual property, weight manipulation, CKA, linear assignment problem

## TL;DR
AWM is a training-free LLM weight-matrix fingerprinting method that recovers permutation and sign-flip transformations in the embedding layer via the Linear Assignment Problem (LAP), and then applies unbiased CKA to neutralize orthogonal transformations in Q/K matrices. It achieves perfect AUC (1.0) on 150 LLM pairs, is robust to six categories of post-training (SFT, continued pretraining up to 5.5T tokens, RL, multimodal extension, pruning, and upcycling), and completes within 30 seconds.

## Background & Motivation
**Background**: LLM training is prohibitively expensive, making intellectual property protection critical. A key need is to determine whether a suspect model was trained from scratch or derived from an existing foundation model.

**Limitations of Prior Work**: Models frequently undergo extensive post-training (SFT, continued pretraining, RL, multimodal extension, pruning, upcycling), causing substantial parameter drift. Watermarking methods require additional training and degrade model performance. Existing fingerprinting methods such as HuRef are not robust to continued pretraining, while REEF suffers from high false positive rates.

**Key Challenge**: Adversaries can obscure model provenance by scaling, permuting, pruning, or even rotating weight matrices—yet these manipulations must preserve model performance. The challenge is to extract invariant fingerprints that exploit precisely these constraints.

**Goal**: Design a fingerprinting method that is robust to all common post-training procedures and weight manipulations, while maintaining a low false positive rate and high computational efficiency.

**Key Insight**: A systematic analysis of the constraints that Transformer components (residual connections, RMSNorm, RoPE) impose on weight manipulations. The paper proves that, under the requirement of preserving model outputs, Q/K matrices can only undergo specific transformation forms (permutation + sign-flip + orthogonal transformation + error), and then eliminates these transformations in a targeted manner.

**Core Idea**: By analyzing the structural constraints that the Transformer architecture imposes on weight manipulations, AWM derives a fingerprinting scheme that is theoretically immune to all feasible manipulations.

## Method

### Overall Architecture
Two stages: (1) recover the permutation matrix $P$ and sign matrix $D$ from the shared-vocabulary embedding matrix using LAP (Hungarian algorithm); (2) align Q/K matrices using the recovered $P$ and $D$, then compute layer-wise similarity with unbiased CKA, which is naturally immune to orthogonal transformations and scaling. The final fingerprint is the average UCKA across all Q/K layers.

### Key Designs

1. **Theoretical Analysis of Weight Manipulations**:

    - **Function**: Precisely characterize the feasible manipulation space of weight matrices that preserves model outputs.
    - **Mechanism**: (1) Residual connections require that manipulations propagate consistently across components (Prop. 4.2); (2) RMSNorm restricts the embedding manipulation to $R_{emb} = cPD$ (scaling + permutation + sign-flip, Thm. 4.3); (3) RoPE combined with attention scores further constrains Q/K matrix manipulations to $W_B = c^{-1}D^TP^TW_A^TU^T + E$ (Thm. 4.4).
    - **Design Motivation**: Rather than selecting fingerprints heuristically, the method derives from first principles which transformations are legitimate and then eliminates them in a targeted fashion.

2. **LAP-based Recovery of Permutation and Sign (Embedding Layer)**:

    - **Function**: Recover the permutation matrix $P$ and sign matrix $D$ from the shared-vocabulary embeddings of two models.
    - **Mechanism**: Construct an absolute cosine similarity matrix between the column vectors of the embedding matrices, solve for the optimal column matching (permutation $P$) via the Hungarian algorithm, and recover $D$ from the signs of cosine similarities at matched positions.
    - **Design Motivation**: Each row of the embedding matrix corresponds to a token with no row-mixing manipulation; column manipulations are constrained by RMSNorm to the form $cPD$.

3. **Unbiased CKA to Eliminate Orthogonal Transformations (Q/K Matrices)**:

    - **Function**: Measure the similarity of aligned Q/K matrices using CKA.
    - **Mechanism**: CKA is inherently invariant to orthogonal transformations and scalar multiplication (Thm. 3.1), eliminating the need to explicitly recover $U_B$. Unbiased UCKA is used to avoid finite-sample bias. The final similarity score is the average UCKA over all Q/K layers.
    - **Design Motivation**: The orthogonal matrix $U$ has $d^2$ parameters, making explicit recovery infeasible; CKA provides a parameter-free solution.

### Loss & Training
No training is required (training-free, lossless to model performance). Only access to the weight matrices of two models is needed; computation completes within 30 seconds.

## Key Experimental Results

### Main Results (150 LLM Pairs)

| Metric | AWM | HuRef | REEF |
|--------|-----|-------|------|
| AUC | **1.0** | ~0.85 | ~0.90 |
| pAUC (FPR<5%) | **1.0** | Low | Low |
| TPR@1%FPR | **1.0** | Low | Low |

### Ablation Study (60 Offspring Model Pairs)

| Post-training Type | AWM | HuRef | REEF |
|--------------------|-----|-------|------|
| SFT | ✅ (≥99.9%) | ✅ | ✅ |
| Continued Pretraining (5.5T tokens) | ✅ | ❌ Fails | Partial |
| RL (PPO/DPO) | ✅ | ✅ | ✅ |
| Multimodal Extension | ✅ | — | Partial |
| Pruning | ✅ | ❌ Fails | Partial |
| Upcycling | ✅ | — | Partial |

### Key Findings
- All offspring models yield similarity ≥99.9%; all independent model pairs yield similarity ≤0.7%—an extremely large separation margin with zero false positive risk.
- HuRef is not robust to continued pretraining and pruning; REEF frequently produces high false positive rates on independent model pairs.
- Computation completes in 30 seconds on an NVIDIA 3090—orders of magnitude faster than black-box inference-based methods.
- The method generalizes to models with different numbers of layers via layer-level LAP matching.

## Highlights & Insights
- **Fingerprints derived from first principles**: Rather than empirically selecting features, AWM systematically analyzes the constraints each Transformer component imposes on weight manipulations and derives a theoretically complete fingerprinting scheme—the analytical methodology itself is of independent value.
- **Elegant application of CKA**: The orthogonal invariance of CKA is leveraged to neutralize the orthogonal transformations introduced by RoPE, circumventing the infeasibility of explicitly recovering high-dimensional orthogonal matrices.
- **High practical utility**: 30 seconds, single GPU, no training required, no performance degradation, and zero false positive rate—fully meeting the requirements of real-world deployment.

## Limitations & Future Work
- Applicable only to decoder-only Transformer architectures; encoder-decoder or SSM architectures require separate analysis.
- Assumes that adversarial manipulations are constrained by the requirement of preserving model outputs; adversaries willing to accept performance degradation may circumvent the method.
- Fully retrained models may yield low similarity scores—but this is the expected behavior, as they are not derived from the base model.
- Requires white-box access to model weights; not applicable to API-only MaaS scenarios.

## Related Work & Insights
- **vs. HuRef**: HuRef is also based on weight invariants but is not robust to continued pretraining. AWM addresses this through a more complete manipulation analysis and unbiased CKA.
- **vs. REEF**: REEF relies on geometric similarity in representation space but suffers from high false positive rates. AWM operates directly in weight space, achieving substantially greater separation.
- **vs. Watermarking Methods**: Watermarking requires additional training and may degrade performance; AWM is post-hoc and lossless.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — The methodology of deriving fingerprints from Transformer architectural constraints is highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 150 model pairs, six post-training categories, and perfect metrics.
- Writing Quality: ⭐⭐⭐⭐⭐ — Rigorous theoretical derivations and comprehensive experiments.
- Value: ⭐⭐⭐⭐⭐ — A highly practical tool for LLM intellectual property protection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Robust Multi-Objective Controlled Decoding of Large Language Models](robust_multi-objective_controlled_decoding_of_large_language_models.md)
- [\[ICLR 2026\] VerifyBench: Benchmarking Reference-based Reward Systems for Large Language Models](verifybench_benchmarking_reference-based_reward_systems_for_large_language_model.md)
- [\[ICLR 2026\] Post-training Large Language Models for Diverse High-Quality Responses](post-training_large_language_models_for_diverse_high-quality_responses.md)
- [\[ICLR 2026\] TROLL: Trust Regions improve Reinforcement Learning for Large Language Models](troll_trust_regions_improve_reinforcement_learning_for_large_language_models.md)
- [\[ICLR 2026\] GraphOmni: A Comprehensive and Extensible Benchmark Framework for Large Language Models on Graph-theoretic Tasks](graphomni_a_comprehensive_and_extensible_benchmark_framework_for_large_language_.md)

</div>

<!-- RELATED:END -->

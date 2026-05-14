---
title: >-
  [Paper Note] Catastrophic Forgetting in Kolmogorov-Arnold Networks
description: >-
  [AAAI 2026][Knowledge Editing][KAN] The first systematic study of catastrophic forgetting in Kolmogorov-Arnold Networks (KANs): establishes a theoretical framework linking forgetting to activation support overlap and int…
tags:
  - "AAAI 2026"
  - "Knowledge Editing"
  - "KAN"
  - "Catastrophic Forgetting"
  - "Continual Learning"
  - "KAN-LoRA"
  - "Activation Support Overlap"
date: 2026-05-08
content_hash: 31dd1eed872cd1dc
---

# Catastrophic Forgetting in Kolmogorov-Arnold Networks

**Conference**: AAAI 2026
**arXiv**: [2511.12828](https://arxiv.org/abs/2511.12828)
**Code**: [Available](https://github.com/marufur-cs/AAAI26)
**Area**: Knowledge Editing
**Keywords**: KAN, Catastrophic Forgetting, Continual Learning, KAN-LoRA, Activation Support Overlap

## TL;DR

The first systematic study of catastrophic forgetting in Kolmogorov-Arnold Networks (KANs): establishes a theoretical framework linking forgetting to activation support overlap and intrinsic data dimensionality, and proposes KAN-LoRA for continual fine-tuning knowledge editing in language models.

## Background & Motivation

Catastrophic forgetting is a central challenge in continual learning. MLPs are inherently susceptible to forgetting due to their globally parameterized updates, and numerous mitigation strategies (regularization, architectural modification, experience replay, etc.) have been proposed. KANs, as an emerging alternative architecture, employ learnable one-dimensional activation functions (B-splines) along network edges; their **locality** ensures that a data sample affects only a small subset of relevant spline coefficients.

Nevertheless, the forgetting behavior of KANs in continual learning remains poorly understood:
- Prior work (Liu et al.) demonstrated KAN robustness only on simple synthetic regression tasks.
- A **theoretical framework** connecting KAN's architectural locality to forgetting behavior is absent.
- Performance on high-dimensional real-world tasks (image classification, language modeling) is unknown.

This paper aims to comprehensively characterize KAN's forgetting properties and limitations from both theoretical and empirical perspectives.

## Method

### Overall Architecture

```
Theoretical Analysis:
  ├─ Forgetting metric: F_i = L(f^(T), D_i) - L(f^(i), D_i)
  ├─ Activation support overlap: Δ_{i,j} = max_{l,p,q} μ(S^(i) ∩ S^(j))
  └─ Intrinsic dimensionality d_t (manifold dimension of task data)

Theorem System:
  ├─ Lemma 1: Zero overlap → zero forgetting
  ├─ Theorem 1: Forgetting upper bound ∝ overlap × Lipschitz constant
  ├─ Theorem 2: Branch-wise cumulative forgetting decomposition
  ├─ Theorem 3: Intrinsic-dimension forgetting rate O(r^{d_i+d_j})
  └─ Corollary 1–4: Random support, saturation, low-dim retention, fragmentation

Empirical Validation:
  ├─ Binary/decimal addition (low-dimensional synthetic)
  ├─ Image classification (MNIST, CIFAR-10, Tiny-ImageNet)
  └─ KAN-LoRA knowledge editing (continual LLM fine-tuning)
```

### Key Designs

**1. Activation Support Overlap Theory**

The activation support of branch $\phi_{\ell,p,q}$ on task $i$ is defined as:

$$S^{(i)}_{\ell,p,q} = \{z \in \mathbb{R} : \phi_{\ell,p,q}(z) \neq 0\}$$

The maximum one-dimensional overlap between two tasks:

$$\Delta_{i,j} = \max_{\ell,p,q} \mu(S^{(i)}_{\ell,p,q} \cap S^{(j)}_{\ell,p,q})$$

**Lemma 1 (Zero-Overlap Retention)**: If $\Delta_{i,j}=0$ for all $j>i$, then $F_i=0$ (perfect retention).

**Theorem 1 (Overlap-Based Forgetting Bound)**: Under Lipschitz and bounded loss assumptions:

$$F_i \leq C \sum_{\ell=1}^{L} N_\ell L_\ell \Delta_{i,j}$$

Core insight: **Forgetting scales linearly with activation support overlap.**

**2. Cumulative Forgetting Analysis**

**Theorem 2 (Branch-wise Cumulative Forgetting)**:

$$F_i \leq C \sum_{\ell} \sum_{p} \sum_{q} L_\ell \left[\sum_{j=i+1}^{T} \mu(S^{(i)}_{\ell,p,q} \cap S^{(j)}_{\ell,p,q})\right]$$

**Corollary 1 (Expected Forgetting under Random Support)**: If each branch support is a random interval of length $s_j$ in $[0,1]$, then $\mathbb{E}[F_i] \leq C \sum_\ell N_\ell L_\ell \sum_{j} s_i s_j$.

**Corollary 2 (Saturation Effect)**: Via union bound, the cumulative effect of overlaps is upper-bounded (does not grow unboundedly), consistent with empirically observed saturation.

**3. Complexity-Driven Forgetting**

**Theorem 3 (Intrinsic-Dimension Forgetting Rate)**: When task data concentrates on a compact submanifold of intrinsic dimension $d_t$:

$$F_i = O\left(\sum_{j=i+1}^{T} N_{tot} \bar{L} \cdot r^{d_i + d_j}\right)$$

Key conclusion: **Forgetting grows exponentially with intrinsic dimensionality.** This explains why KANs excel on low-dimensional tasks yet remain vulnerable on high-dimensional ones.

**Corollary 4 (Fragmentation Mitigation)**: Splitting the support into $k_t$ disjoint intervals (effective radius $r/k_t$) improves the forgetting rate.

**4. KAN-LoRA Adapter**

To translate theoretical insights into practical LLM continual learning, a KAN-based LoRA adapter is designed:
- Replaces low-rank matrices in standard MLP-LoRA with KAN structures.
- Leverages KAN's local activation property to better retain prior knowledge during sequential knowledge editing.
- Evaluated on Llama 2 (7B, 13B).

### Loss & Training

- Synthetic tasks: MSE loss, SGD optimization.
- Image classification: cross-entropy loss, KAN-Transformer architecture (all MLP layers replaced by KAN layers).
- Knowledge editing: sequential editing on CounterFact and ZsRE benchmarks; retention of edit accuracy after subsequent tasks is measured.

## Key Experimental Results

### Main Results

**Table 1: Forgetting vs. Overlap Ratio on Decimal Addition Tasks (validating Theorem 1)**

| Task(i) | Task(j) | Grid 10: $F_i/\Delta_{i,j}$ | Grid 15 | Grid 20 |
|---------|---------|------|------|------|
| 1 | 2 | 0.74 | 0.74 | 0.61 |
| 2 | 3 | 0.73 | 0.67 | 0.64 |
| 3 | 4 | 0.77 | 0.74 | 0.63 |
| 4 | 5 | 0.72 | 0.68 | 0.64 |

The near-constant ratio $F_i/\Delta_{i,j}$ validates the **linear relationship** between forgetting and overlap.

**Table 2: Cumulative Forgetting vs. Cumulative Overlap Ratio (validating Theorem 2)**

| Task(i) | Tasks(j) | Grid 10: $F_i/\sum\mu^{ij}$ | Grid 15 | Grid 20 |
|---------|----------|------|------|------|
| 1 | 2,3,4,5 | 0.15 | 0.15 | 0.16 |
| 2 | 3,4,5 | 0.16 | 0.15 | 0.16 |
| 3 | 4,5 | 0.16 | 0.16 | 0.16 |
| 4 | 5 | 0.18 | 0.17 | 0.17 |

The ratio is more stable, with lower variance at larger grid sizes, validating the influence of spline resolution on forgetting consistency.

**Table 3: Intrinsic Dimensionality vs. Forgetting Rate (validating Theorem 3)**

| Dataset | $\log(F_i)/d_i$ range |
|-------|---------------------|
| MNIST | 0.071–0.075 |
| CIFAR-10 | 0.046–0.053 |
| Tiny-ImageNet | 0.047–0.054 |

The near-constant ratio $\log(F_i)/d_i$ validates the **exponential relationship** between forgetting and intrinsic dimensionality.

**Table 4: KAN-LoRA vs. MLP-LoRA for Knowledge Editing (Llama 2-7B, rank 8)**

| Dataset | #Tasks | KAN-LoRA Acc | MLP-LoRA Acc |
|-------|--------|-------------|-------------|
| CounterFact | 2 | 100 | 100 |
| CounterFact | 3 | 65 | 90 |
| CounterFact | 4 | 50 | 65 |
| CounterFact | 5 | 45 | 57 |
| ZsRE | 2 | 100 | 100 |
| ZsRE | 3 | 80 | 95 |
| ZsRE | 5 | 60 | 87 |

In high-dimensional LLM settings, KAN-LoRA underperforms MLP-LoRA, confirming the theoretical predictions.

### Ablation Study

- **Effect of Grid Size**: Increasing grid from 5→20 consistently reduces forgetting on decimal addition tasks (finer splines reduce support overlap).
- **Effect of Model Depth**: Forgetting increases sharply as encoder blocks deepen (deeper networks are more vulnerable).
- **Effect of Sample Size**: Forgetting worsens with more training samples (larger sample sets expand the activation support range).
- **KAN vs. MLP-EWC**: On CIFAR-10, KAN outperforms MLP-EWC with fewer tasks; on Tiny-ImageNet, MLP-EWC is superior.

### Key Findings

1. **KANs are naturally resistant to forgetting on low-dimensional structured tasks**: Forgetting in binary addition is below $10^{-6}$, even surpassing purpose-built MLPs.
2. **High-dimensional tasks are KAN's Achilles' heel**: Severe forgetting persists in image classification and language modeling, consistent with theoretical predictions.
3. **Grid size is a critical hyperparameter**: Larger grids reduce spline support length, directly mitigating forgetting via reduced overlap.
4. **A forgetting saturation effect exists**: In task sequences with high overlap, cumulative forgetting tends to saturate.
5. **KAN-LoRA shows limited effectiveness in LLM settings**: High dimensionality neutralizes KAN's locality advantage.

## Highlights & Insights

- **Completeness of the theoretical framework**: Four theorems and four corollaries—from zero overlap to cumulative forgetting to complexity-driven analysis—form a coherent and layered theoretical system.
- **Strong theory–experiment alignment**: Each theorem is paired with a corresponding experimental table; the near-constant ratios are compelling.
- **Honest reporting of negative results**: The failure of KAN-LoRA in high-dimensional settings is presented directly rather than avoided, enhancing the credibility of the work.
- **Practical guidance**: The quantified relationship Grid Size ↑ → Forgetting ↓ provides direct hyperparameter tuning guidance for deploying KANs in continual learning.

## Limitations & Future Work

1. The KAN-LoRA design is relatively straightforward; more sophisticated adaptation strategies (e.g., task-specific spline routing) are not explored.
2. The Lipschitz and bounded loss assumptions in the theoretical analysis may not strictly hold in practical networks.
3. A concrete implementation of the fragmentation mitigation strategy (Corollary 4) is not provided.
4. Comparisons with mainstream continual learning methods applied to KANs (e.g., EWC on KAN, memory replay on KAN) are absent.
5. Future work could consider embedding adaptive grid sizing into the continual learning pipeline (dynamic grid expansion strategies).

## Related Work & Insights

- **KAN (Liu et al. 2025)**: The original KAN paper hints at forgetting robustness; this paper establishes the precise conditions under which it holds.
- **EWC (Kirkpatrick et al. 2017)**: A classical regularization-based continual learning method used as an enhanced MLP baseline.
- **LoRA (Hu et al. 2022)**: Parameter-efficient fine-tuning, extended here to a KAN-based variant.
- **Implication for model compression**: KAN's locality advantage in low-dimensional settings suggests that compressed models with lower intrinsic dimensionality may be inherently more resistant to forgetting.

## Rating

- Novelty: ⭐⭐⭐⭐ (First theoretical framework for forgetting in KANs + KAN-LoRA)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Three-tier validation: synthetic + vision + language)
- Writing Quality: ⭐⭐⭐⭐ (Clear theoretical derivations with well-matched experiments)
- Value: ⭐⭐⭐⭐ (Provides a clear characterization of KAN's capability boundaries in continual learning)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] When Large Multimodal Models Confront Evolving Knowledge: Challenges and Explorations](../../ICLR2026/knowledge_editing/when_large_multimodal_models_confront_evolving_knowledge_challenges_and_explorat.md)
- [\[AAAI 2026\] Multiplicative Orthogonal Sequential Editing for Language Models (MOSE)](multiplicative_orthogonal_sequential_editing_for_language_models.md)
- [\[AAAI 2026\] Model Editing as a Double-Edged Sword: Steering Agent Ethical Behavior](model_editing_as_a_double-edged_sword_steering_agent_ethical_behavior_toward_ben.md)
- [\[AAAI 2026\] Is the Information Bottleneck Robust Enough? Towards Label-Noise Resistant Information Bottleneck Learning](is_the_information_bottleneck_robust_enough_towards_label-noise_resistant_inform.md)
- [\[ACL 2026\] EvoEdit: Evolving Null-space Alignment for Robust and Efficient Knowledge Editing](../../ACL2026/knowledge_editing/evoedit_evolving_null-space_alignment_for_robust_and_efficient_knowledge_editing.md)

</div>

<!-- RELATED:END -->

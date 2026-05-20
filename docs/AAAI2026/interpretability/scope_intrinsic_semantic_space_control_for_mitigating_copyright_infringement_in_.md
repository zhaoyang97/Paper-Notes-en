---
title: >-
  [Paper Note] SCoPe: Intrinsic Semantic Space Control for Mitigating Copyright Infringement in LLMs
description: >-
  [AAAI 2026][Interpretability][Copyright Protection] This paper reframes copyright infringement mitigation in LLMs as an intrinsic semantic space control problem. It leverages sparse autoencoders (SAEs) to map hidden stat…
tags:
  - "AAAI 2026"
  - "Interpretability"
  - "Copyright Protection"
  - "Sparse Autoencoder (SAE)"
  - "Semantic Subspace"
  - "Feature Clamping"
  - "Inference-Time Intervention"
date: 2026-05-08
content_hash: 8ea69dd153c7eb6d
---

# SCoPe: Intrinsic Semantic Space Control for Mitigating Copyright Infringement in LLMs

**Conference**: AAAI 2026
**arXiv**: [2511.07001](https://arxiv.org/abs/2511.07001)  
**Code**: None  
**Area**: Interpretability
**Keywords**: Copyright Protection, Sparse Autoencoder (SAE), Semantic Subspace, Feature Clamping, Inference-Time Intervention

## TL;DR
This paper reframes copyright infringement mitigation in LLMs as an intrinsic semantic space control problem. It leverages sparse autoencoders (SAEs) to map hidden states into a high-dimensional sparse space, identifies copyright-sensitive subspaces, and clamps their activations to zero during decoding—effectively reducing verbatim reproduction of copyrighted content without external filters or parameter updates, while preserving general model capabilities.

## Background & Motivation

**Background**: LLMs may inadvertently reproduce copyrighted content (e.g., news articles, books) present in their training data, giving rise to numerous copyright lawsuits. Existing defenses fall into three categories: data preprocessing and filtering, training-time interventions (e.g., selective unlearning), and inference-time control. Inference-time methods are the most flexible, requiring no modification to model parameters.

**Limitations of Prior Work**:
- **Dependence on external artifacts**: Most existing inference-time methods rely on blocklist corpora or Bloom filters for n-gram-level string matching (e.g., MemFree), increasing deployment complexity.
- **Surface-level similarity detection only**: Token-matching approaches cannot detect semantic-level paraphrased leakage.
- **Potential degradation of fluency**: External filtering or resampling can disrupt normal generation quality.

**Key Challenge**: Copyright protection requires identifying and suppressing copyrighted content at the semantic level, yet the polysemanticity of LLM neurons makes it difficult to localize copyright-relevant dimensions within the hidden state space.

**Goal**: Can external filtering mechanisms be bypassed entirely, enabling LLMs to intrinsically avoid generating infringing content at the semantic space level?

**Key Insight**: SAEs are used to map the LLM's dense hidden states into a high-dimensional sparse space, where each dimension approximately corresponds to a single semantic concept (monosemanticity), enabling identification and manipulation of copyright-relevant dimensions.

**Core Idea**: Identify copyright-sensitive subspaces within the SAE sparse space and clamp their activations to zero during decoding, achieving semantic-level suppression of copyrighted content.

## Method

### Overall Architecture
SCoPe operates in two stages: (1) **Copyright subspace identification**—locating copyright-sensitive dimensions by measuring the difference in SAE activations between copyrighted and general corpora; (2) **Feature clamping**—zeroing out activations in the copyright subspace at each decoding step, reconstructing the hidden state and passing it back to the model. The entire pipeline requires no training and no external filters.

### Key Designs

1. **Copyright Subspace Identification**:

    - **Function**: Identify $n$ dimensions within the SAE's $k$-dimensional sparse space that are highly correlated with copyrighted content.
    - **Mechanism**: Define a Copyright Alignment Score $\mathcal{Q}(i)$—the probability that dimension $i$ is more strongly activated on copyrighted samples than on general samples (essentially a per-dimension AUROC). The paper proves that the subspace score is upper-bounded by the best single-dimension score: $\mathcal{Q}(\mathcal{S}) \leq \max_{i \in \mathcal{I}} \mathcal{Q}(i)$, implying that greedily selecting the top-$n$ highest-scoring dimensions is approximately optimal.
    - **Algorithm**: Compute $\mathcal{Q}(i)$ for each dimension, rank dimensions in descending order, and select the top-$n$ (default $n=1000$) to form the estimated subspace $\hat{\mathcal{S}}$.
    - **Design Motivation**: SAE monosemanticity ensures each dimension corresponds to a well-defined semantic concept, making per-dimension scoring reliable. Greedy selection runs in linear time, avoiding exponential search.

2. **Feature Clamping**:

    - **Function**: Suppress copyright subspace activations at each decoding step.
    - **Mechanism**: At each decoding step, the hidden state $\mathbf{h}$ is projected through the SAE encoder to obtain $\mathbf{z}$; dimensions in $\hat{\mathcal{I}}$ whose activations exceed threshold $\tau$ are clamped to zero: $z_i \leftarrow 0 \text{ if } i \in \hat{\mathcal{I}} \text{ and } z_i > \tau$; the modified $\mathbf{z}$ is then decoded by the SAE decoder to reconstruct $\hat{\mathbf{h}}$, which is passed back to the model.
    - **Design Motivation**: Only copyright-relevant activations are suppressed; all other semantic dimensions remain unchanged, enabling precise semantic-level control rather than coarse token filtering.

3. **Theoretical Support—Subspace Hypothesis Validation**:

    - **Function**: Verify that copyrighted content forms a separable subspace in the SAE sparse space.
    - **Experimental Evidence**: (1) In the dense space, dimensional activations of copyrighted vs. general content heavily overlap (Figure 1a left), whereas clear separation emerges in the SAE sparse space (Figure 1a right); (2) Activations overlap in the full space (Figure 1b) but separate clearly in the copyright subspace (Figure 1c); (3) A reverse intervention—amplifying copyright subspace activations—increases infringement rates (Figure 3), confirming causality.

## Key Experimental Results

### Main Results

Copyright mitigation Win Rate on NewsQA (win rate against 5 baselines):

| Model | Method | Avg Win Rate↑ | Blocklisted F1↑ | In-Domain F1↑ |
|-------|--------|---------------|-----------------|---------------|
| Gemma-2 | Vanilla | 12.9% | 60.9 | 62.6 |
| Gemma-2 | System Prompt | 25.3% | 60.2 | 61.8 |
| Gemma-2 | MemFree | 64.5% | 55.9 | 61.4 |
| Gemma-2 | R-CAD | 64.1% | 58.5 | 60.1 |
| Gemma-2 | **SCoPe** | **71.7%** | 59.4 | 62.6 |
| Llama-3 | R-CAD | 66.9% | 58.8 | 61.9 |
| Llama-3 | **SCoPe** | **70.2%** | 59.2 | 62.1 |

General capability on MMLU (negligible degradation):

| Method | Gemma-2 | Llama-3 |
|--------|---------|---------|
| Vanilla | 67.3 | 63.5 |
| **SCoPe** | **66.7** | **63.1** |
| Top-k Perturbation | 46.1 | 45.8 |

### Ablation Study

Effect of subspace dimensionality $n$ (BookSum, Llama-3):

| Dimension $n$ | Avg Win Rate | MMLU |
|---------------|-------------|------|
| 0 (Vanilla) | 8.7% | 63.5 |
| 500 | ~50% | 63.5 |
| 1000 | **68.5%** | **63.5** |
| 1500 | ~71% | ~62 |
| 2000 | 72.5% | ~60 |

### Key Findings
- **$n=1000$ is the optimal trade-off**: Win rate reaches 68.5% with zero MMLU loss; further increasing $n$ yields marginal win rate gains at the cost of MMLU degradation.
- **Reverse intervention (causal proof)**: Amplifying the copyright subspace at $\alpha=2.0$ reduces win rate from 8.7% to 4.1% (model becomes more prone to reproducing copyrighted content) while leaving general capability unchanged. The bidirectional results confirm that the subspace causally encodes copyright-relevant semantics.
- **SCoPe outperforms all baselines by 3–7 percentage points**: It surpasses the strongest baseline R-CAD by approximately 5–7% with minimal utility loss.
- **Top-k perturbation severely degrades general capability**: MMLU drops from 63.5 to 45.8, demonstrating the inadequacy of indiscriminate perturbation.
- **Feature interpretability**: Dimensions in the copyright subspace correspond to high-level semantics such as character dialogue and plot transitions, while general dimensions correspond to low-level patterns such as formatting tokens and common adjectives.

## Highlights & Insights
- **Reframing copyright protection as semantic space control**: The problem is elevated from surface-level token matching to semantic subspace identification and suppression—a paradigm-level innovation that enables detection of paraphrased infringement beyond verbatim copying.
- **The SAE + Feature Clamping combination is elegant**: It leverages publicly available SAEs (e.g., GemmaScope) with zero training cost. Clamping zeroes out only selected dimensions in the sparse space, leaving all other dimensions entirely unaffected.
- **Copyright Alignment Score design**: Subspace search is reduced to a per-dimension ranking problem solvable in linear time, with a theoretical upper-bound proof justifying the greedy selection strategy.
- **Reverse intervention experiment**: A clean causal verification—demonstrating not only that suppressing the subspace reduces infringement, but also that amplifying it increases infringement, establishing causality rather than mere correlation.

## Limitations & Future Work
- **Applicable only to open-source models with publicly available SAEs**: The method requires access to intermediate hidden states and pre-trained SAEs, making it inapplicable to closed-source models (e.g., GPT-4, Claude).
- **Linear subspace assumption**: Copyrighted content may not strictly reside in a linear subspace within the sparse space; nonlinear methods may prove more effective.
- **Requires pre-specified copyright corpora**: Computing the Alignment Score requires both a copyrighted document set $\mathcal{C}_{cr}$ and a general document set $\mathcal{C}_{gen}$; in practice, the copyright content collection may be incomplete or continuously updated.
- **The choice of $n$ is dataset-dependent**: $n=1000$ is optimal for BookSum, but different types of copyrighted content (e.g., code, song lyrics) may require different values of $n$.
- **Evaluation limited to English text**: Multilingual copyright protection scenarios have not been validated.

## Related Work & Insights
- **vs. MemFree**: MemFree uses Bloom filters for n-gram-level filtering, which can only prevent verbatim copying. SCoPe operates at the semantic level and can defend against semantic paraphrasing infringement, without requiring an external blocklist.
- **vs. R-CAD**: R-CAD downweights token probabilities aligned with copyrighted spans—still a token-level operation. SCoPe operates in the hidden state space, addressing the problem more fundamentally.
- **vs. SAE-TS/FGAA**: These works similarly use SAEs for activation steering, but SCoPe is the first to apply this paradigm to copyright protection and introduces the Copyright Alignment Score as a systematic method for subspace selection.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Redefines copyright protection as semantic subspace control; the SAE + Feature Clamping combination is original and elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Two benchmarks, two models, five baselines, subspace size analysis, and reverse intervention experiments provide comprehensive coverage.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Theoretical derivations are clear, visualizations are thorough, and the logical chain from hypothesis to validation to application is complete.
- **Value**: ⭐⭐⭐⭐⭐ High practical value (zero-training-cost deployment), high academic value (validation of the semantic subspace hypothesis), and important implications for LLM safety research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Out of Control -- Why Alignment Needs Formal Control Theory (and an Alignment Control Stack)](../../NeurIPS2025/interpretability/out_of_control_--_why_alignment_needs_formal_control_theory_and_an_alignment_con.md)
- [\[AAAI 2026\] Adaptive Evidential Learning for Temporal-Semantic Robustness in Moment Retrieval](adaptive_evidential_learning_for_temporal-semantic_robustnes.md)
- [\[ICLR 2026\] The Geometry of Reasoning: Flowing Logics in Representation Space](../../ICLR2026/interpretability/the_geometry_of_reasoning_flowing_logics_in_representation_space.md)
- [\[AAAI 2026\] Finding the Translation Switch: Discovering and Exploiting the Task-Initiation Features in LLMs](finding_the_translation_switch_discovering_and_exploiting_the_task-initiation_fe.md)
- [\[CVPR 2026\] Beyond Semantics: Disentangling Information Scope in Sparse Autoencoders for CLIP](../../CVPR2026/interpretability/beyond_semantics_disentangling_information_scope_in_sparse_autoencoders_for_clip.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] NerVE: Nonlinear Eigenspectrum Dynamics in LLM Feed-Forward Networks
description: >-
  [ICLR 2026][Video Understanding][FFN Analysis] This paper proposes NerVE, a lightweight eigenspectrum analysis framework that systematically reveals, via four complementary metrics (Spectral Entropy, Participation Ratio…
tags:
  - "ICLR 2026"
  - "Video Understanding"
  - "FFN Analysis"
  - "Eigenspectrum Dynamics"
  - "Variance Re-injection"
  - "Optimizer Geometry"
  - "Spectral Diagnostics"
date: 2026-05-08
content_hash: e303b1ed03432297
---

# NerVE: Nonlinear Eigenspectrum Dynamics in LLM Feed-Forward Networks

**Conference**: ICLR 2026
**arXiv**: [2603.06922](https://arxiv.org/abs/2603.06922)  
**Code**: [Project Page](https://nerve-eigenspectrum.github.io/)  
**Area**: Video Understanding
**Keywords**: FFN Analysis, Eigenspectrum Dynamics, Variance Re-injection, Optimizer Geometry, Spectral Diagnostics

## TL;DR

This paper proposes NerVE, a lightweight eigenspectrum analysis framework that systematically reveals, via four complementary metrics (Spectral Entropy, Participation Ratio, Eigenvalue Early Enrichment, and JS Divergence), how FFN nonlinearities in LLMs re-inject variance, reshape the eigenspectrum, and how architectural and optimizer choices imprint distinct spectral signatures.

## Background & Motivation

In Transformers, feed-forward networks (FFNs) account for the majority of parameters and computation, yet their internal dynamics remain substantially understudied compared to attention mechanisms. Existing work has focused primarily on attention map visualization and analysis, leaving open the question of how FFNs organize and propagate information in high-dimensional latent spaces.

Key challenges:
- FFN transformations unfold in high-dimensional spaces and cannot be directly visualized like attention maps
- Systematic and efficient tools for characterizing how FFN nonlinear activations reshape latent representations are lacking
- Prior work (Kobayashi et al., 2024; Balestriero et al., 2024) has studied FFNs from the perspectives of attention maps and piecewise-affine partitioning, respectively, but neither reveals how nonlinearities redistribute variance

Core insight: **FFN nonlinear activations do not merely scale activation values; rather, they actively re-inject variance into underutilized eigendirections, fundamentally governing the utilization of latent dimensions.**

## Method

### Overall Architecture

The NerVE framework comprises four main components:
1. Activation collection (pre-activation and post-activation)
2. Covariance matrix computation
3. Eigendecomposition
4. Spectral metric computation

For each FFN layer $\ell$, the framework collects the post-projection, pre-activation values $\text{PreAct}(X) = W_{up}x + b_1$ and the post-activation, pre-down-projection values $\text{PostAct}(X) = \sigma(W_{up}x + b_1)$, computes the respective covariance matrices, and performs eigendecomposition.

### Key Designs

1. **Spectral Entropy (SE)**:

    - $SE = -\sum_{i=1}^{D} \hat{\lambda}_i \log \hat{\lambda}_i$ (Shannon entropy of normalized eigenvalues)
    - Equivalent to the von Neumann entropy in quantum information theory
    - High SE indicates uniformly distributed variance; low SE indicates variance concentrated in a few directions
    - More sensitive to mid-tail eigenvalues

2. **Participation Ratio (PR)**:

    - $PR = \frac{(\sum_i \lambda_i)^2}{\sum_i \lambda_i^2}$, range $[1, D]$
    - Measures effective dimensionality — how many directions substantively contribute to total variance
    - PR ≈ 1 indicates high anisotropy; PR ≈ D indicates uniformly distributed variance
    - More sensitive to leading eigenvalues

3. **Eigenvalue Early Enrichment (EEE)**:

    - $EEE = \frac{2}{D} \sum_{k=1}^{D} (\tilde{S}_k - \frac{k}{D})$
    - Measures the "top-heaviness" of the eigenspectrum — how quickly the cumulative variance exceeds the uniform baseline
    - EEE ≈ 1 indicates extreme variance concentration; EEE ≈ 0 indicates near-uniform distribution
    - Distinguishes spectra that exploit different fractions of the latent space

4. **Jensen-Shannon Divergence (JS)**:

    - $JS(P_{pre} \| P_{post}) = \frac{1}{2} D_{KL}(P_{pre} \| M) + \frac{1}{2} D_{KL}(P_{post} \| M)$
    - Quantifies the distributional shift pre → post induced by the nonlinearity
    - The only metric spanning two spectra; the other three describe individual spectra

Design principles for the four metrics: coverage (sensitivity to different spectral regions), complementary sensitivity, boundedness, and scale invariance.

### Loss & Training

NerVE is an analysis framework rather than a training method. Training configurations for the experimental models are as follows:
- **GPT-2 (125M)**: CodeParrot dataset, 2.1B tokens, 41K steps
- **LLaMA variants (71M–1.3B)**: C4 dataset
- **GPT-2 (350M, 160M)**: FineWeb dataset, for optimizer comparison
- **MLP-Mixer (B/16)**: CIFAR-100, for cross-architecture validation
- Memory optimization: layer-by-layer processing; peak GPU memory requires only 2 × 36 MB (3072 × 3072 covariance matrices)
- Computational overhead: logging every 1,000 steps adds only ~1% training time

## Key Experimental Results

### Main Results

Perplexity of GPT-2 baseline models under different configurations:

| Config | GELU | ReLU | NormFree GELU | NormFree ReLU | NormFree LReLU | WNorm | SNorm | HNorm |
|--------|------|------|---------------|---------------|----------------|-------|-------|-------|
| PPL↓ | 2.714 | 2.774 | 3.223 | 2.988 | 3.081 | 3.041 | 3.000 | 3.122 |

Optimizer comparison (GPT-2 350M, FineWeb):

| Optimizer | PPL (512 ctx) | PPL (1024 ctx) |
|-----------|---------------|----------------|
| AdamW | 33.24 | 39.26 |
| Dion | 27.68 | 33.60 |
| Muon | **25.68** | **30.95** |

### Ablation Study

| Config | Key Metric | Notes |
|--------|------------|-------|
| Pre vs. Post SE/PR | Post > Pre (consistent) | Nonlinearity re-injects variance, expanding effective dimensionality |
| GELU vs. ReLU | GELU PR_post higher | Smoother nonlinearity explores a broader subspace |
| NormFree + GELU | EEE_post ≈ 1, JS ≈ 0 (early layers) | Spectral inertia — nonlinearity fails to activate |
| NormFree + ReLU | PR gain 20×–300× | ReLU aggressively compensates, breaking spectral inertia |
| PreLN vs. PostLN | PreLN PR/D highest and stable | PreLN yields the best "width return rate" |
| RoPE vs. NoPE | RoPE higher PR in mid-deep layers | RoPE prevents spectral collapse in mid-to-deep layers |

### Key Findings

1. **Variance re-injection is the core function of FFN nonlinearity**: Post-activation consistently exhibits higher SE and PR, and lower EEE — nonlinearities re-inject variance into underutilized directions, "awakening" dormant regions of the latent space.
2. **Optimizers determine the role of FFN nonlinearity — repair vs. refinement**:
    - **AdamW**: Causes pre-activation spectral collapse → FFN nonlinearity is forced into a "repair mode" (large PR gain but low final PR_post)
    - **Muon**: Maintains well-conditioned pre-activation spectra → FFN nonlinearity only needs to "fine-tune" (small PR gain but high PR_post) → lower perplexity
3. **Spectral signatures predict generalization**: Pearson correlation between NerVE metrics and validation loss satisfies |r| ≥ 0.97 (pre-activation), enabling online diagnostic use during forward passes.
4. **ReLU can partially substitute for LayerNorm in NormFree models**: Through aggressive variance re-injection (PR gain 20×–300×), ReLU closes ~50% of the perplexity gap.
5. **Muon concentrates representational capacity in middle FFN layers**: The highest PR_post occurs in middle layers — the perplexity ranking follows the PR_post trend of the middle layers.

## Highlights & Insights

- **Novel perspective**: Understanding FFNs through eigenspectrum dynamics reveals the previously unrecognized core function of nonlinearity as "variance re-injection."
- **Practical diagnostic tool**: NerVE enables online monitoring during training with negligible overhead (~1%) and requires no additional forward passes.
- **Cross-architecture generalization**: Core findings hold across GPT-2, LLaMA, and MLP-Mixer, suggesting these are universal properties of deep feed-forward networks.
- **Optimizer as inductive bias**: Different optimizers imprint qualitatively distinct geometric signatures on the FFN spectrum, providing a new diagnostic basis for optimizer selection.
- **Elegant four-metric design**: Each metric is sensitive to a different region of the spectrum; their joint use avoids the pitfalls of relying on any single metric.

## Limitations & Future Work

1. **Layer-independent analysis**: Cross-layer spectral relationships are not explicitly quantified, precluding the capture of inter-layer spectral coherence.
2. **Token aggregation**: All token positions are pooled for computation, ignoring position-specific spectral structure (Appendix J shows significant positional dependence in LayerNorm models).
3. **No direct prediction of downstream task quality**: NerVE metrics are highly correlated with generalization but do not imply causation.
4. **Computational cost for large-scale models**: For large FFN dimensions with $D > 10K$, full-batch covariance computation and eigendecomposition may be expensive (though 10% token subsampling can preserve pre-activation diagnostic capability).
5. **Attention–FFN interaction not covered**: How the FFN spectrum is influenced by upstream attention layers remains unanalyzed.

## Related Work & Insights

- **RankMe** (Garrido et al., 2023) and **Diff-eRank** (Wei et al., 2024): Use spectral entropy to predict downstream performance.
- **Bao et al. (2024)**: Studies the relationship between spectral concentration of QK weight matrices and attention localization.
- **Poole et al. (2016)**: Order-to-chaos phase transitions induced by nonlinearity in randomly initialized networks.
- **Kobayashi et al. (2024)**: Studies FFNs from the perspective of attention maps.
- **Pascanu et al. (2025)**: Optimizers qualitatively alter solutions — NerVE provides concrete spectral-level evidence.
- **Insights**: FFN nonlinearity is not a "supporting role" but a central regulator of information flow; optimizer geometry and the internal representational geometry of the network are deeply coupled.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (A genuinely novel FFN eigenspectrum analysis framework; the core insight of "variance re-injection" is highly original)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Exceptionally comprehensive: multiple architectures × optimizers × normalization schemes × activation functions × scales × cross-architecture validation)
- Writing Quality: ⭐⭐⭐⭐ (Content-rich but lengthy; appendices are thorough)
- Value: ⭐⭐⭐⭐⭐ (Provides substantive diagnostic tools and theoretical insights for LLM architecture design and optimizer selection)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Log Probability Tracking of LLM APIs](log_probability_tracking_of_llm_apis.md)
- [\[ICLR 2026\] Emergence of Superposition: Unveiling the Training Dynamics of Chain of Continuous Thought](emergence_of_superposition_unveiling_the_training_dynamics_of_chain_of_continuou.md)
- [\[ICLR 2026\] Stabilizing Policy Gradients for Sample-Efficient Reinforcement Learning in LLM Reasoning](stabilizing_policy_gradients_for_sample-efficient_reinforcement_learning_in_llm_.md)
- [\[ACL 2026\] ViLL-E: Video LLM Embeddings for Retrieval](../../ACL2026/video_understanding/vill-e_video_llm_embeddings_for_retrieval.md)
- [\[AAAI 2026\] Learning Topology-Driven Multi-Subspace Fusion for Grassmannian Deep Networks](../../AAAI2026/video_understanding/learning_topology-driven_multi-subspace_fusion_for_grassmannian_deep_network.md)

</div>

<!-- RELATED:END -->

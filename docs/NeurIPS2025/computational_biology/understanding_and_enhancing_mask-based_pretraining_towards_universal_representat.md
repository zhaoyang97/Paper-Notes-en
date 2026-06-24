---
title: >-
  [Paper Note] Understanding and Enhancing Mask-Based Pretraining towards Universal Representations
description: >-
  [NEURIPS2025][Computational Biology][masked autoencoder] This paper employs high-dimensional linear regression theory to precisely characterize the effect of masking ratio on test risk in mask-based pretraining via a bias-variance decomposition, revealing that the optimal masking ratio depends on both the downstream task and model size. Building on this theory, the paper proposes R2MAE (Random Ratio MAE), which consistently outperforms fixed masking ratios across vision…
tags:
  - "NEURIPS2025"
  - "Computational Biology"
  - "masked autoencoder"
  - "pretraining theory"
  - "random matrix theory"
  - "bias-variance tradeoff"
  - "R2MAE"
date: 2026-05-08
content_hash: 922357bda30c87df
---

# Understanding and Enhancing Mask-Based Pretraining towards Universal Representations

**Conference**: NEURIPS2025
**arXiv**: [2509.21650](https://arxiv.org/abs/2509.21650)  
**Code**: None  
**Area**: LLM Pretraining
**Keywords**: masked autoencoder, pretraining theory, random matrix theory, bias-variance tradeoff, R2MAE

## TL;DR
This paper employs high-dimensional linear regression theory to precisely characterize the effect of masking ratio on test risk in mask-based pretraining via a bias-variance decomposition, revealing that the optimal masking ratio depends on both the downstream task and model size. Building on this theory, the paper proposes R2MAE (Random Ratio MAE), which consistently outperforms fixed masking ratios across vision, language, DNA, and single-cell modeling benchmarks.

## Background & Motivation

### State of the Field

**Background**: Mask-based pretraining (BERT 15%, MAE 75%) is the dominant self-supervised paradigm in NLP, CV, and bioinformatics, yet there is no principled theoretical explanation for why the optimal masking ratio varies so substantially across domains.

**Limitations of Prior Work**: (1) BERT uses 15% while MAE uses 75%—the underlying reason is unclear. (2) The optimal masking ratio varies with downstream task and model size. (3) Existing theory cannot provide a unified quantitative account of behavior across different domains.

**Key Challenge**: Too little masking fails to encourage learning of feature co-dependencies; too much masking renders the input insufficiently informative. The location of the optimal trade-off remains unknown.

**Goal**: To precisely characterize the quantitative relationship between masking ratio and model performance, to explain why the optimal masking ratio depends on task and model size, and to design a superior masking strategy accordingly.

**Key Insight**: Mask pretraining is reduced to high-dimensional minimum-norm linear regression, and closed-form test risk is derived using random matrix theory.

**Core Idea**: The behavior of mask pretraining is fully explained by a bias-variance decomposition, and a theoretically motivated random masking ratio outperforms any fixed masking ratio.

## Method

### Overall Architecture
Each feature reconstruction in masked autoencoding is modeled as a high-dimensional linear regression problem. The masking ratio $p$ simultaneously controls the effective sample size and covariate sparsity. The optimal $p$ is revealed through a bias-variance decomposition of ridgeless regression.

### Key Designs

1. **Isotropic Model (Theorem 1)**: When $\Sigma = I$, the test risk admits a closed-form expression. In the underparameterized regime, risk increases monotonically with $p$; in the overparameterized regime, the relationship is non-monotone, yielding a well-defined optimal masking ratio.

2. **Spiked Covariance Model (Corollary 1)**: When features are correlated, the genuine benefit of masking emerges—the model is compelled to learn inter-feature dependencies. The optimal masking ratio depends on the signal-to-noise ratio, the overparameterization ratio, and the alignment between the signal direction and the principal components.

3. **R2MAE**: During training, the masking ratio is sampled uniformly from $[p_{\min}, p_{\max}]$, forcing the model to learn multi-scale features—high masking ratios capture global structure while low masking ratios capture local details.

### Key Theoretical Findings
- Masking is beneficial only in the overparameterized regime.
- The optimal masking ratio depends on model size; larger models require higher masking ratios.
- Masking induces variation in feature magnitudes, facilitating the learning of discriminative representations.

## Key Experimental Results

### Main Results

| Domain | Model | Best Fixed Ratio | R2MAE | Notes |
|--------|-------|-----------------|-------|-------|
| Vision | ViT-B MAE | 75% | **Outperforms** | Significant improvement |
| Language | BERT | 15% | **Outperforms** | Moderate improvement |
| DNA | DNABERT | 15% | **Outperforms** | Significant improvement |
| Single-cell | scBERT | 15% | **Outperforms** | Significant improvement |

### Key Findings
- Theoretical predictions closely match empirical behavior—the curve shapes predicted by the linear model are reproduced across MLP, CNN, and Transformer architectures.
- The core mechanism of R2MAE is multi-scale feature learning.
- Gains are especially pronounced on biological data, representing the first successful masking strategy improvement in that domain.

## Highlights & Insights
- **The bias-variance decomposition fully explains mask pretraining**—reducing a complex phenomenon to classical statistical concepts.
- **R2MAE is minimally invasive**: it requires only a one-line code change (replacing the constant masking ratio with random sampling).
- A unified theoretical and methodological framework applicable across diverse domains.

## Limitations & Future Work
- The linear model assumption introduces a gap relative to actual deep nonlinear architectures.
- Autoregressive pretraining is not covered.
- The interval range for R2MAE still requires hyperparameter tuning.

## Related Work & Insights
- **vs. Kong et al. (2024)**: Their work establishes the existence of an optimal masking ratio, whereas this paper provides precise quantification.
- **vs. MAE (He et al.)**: MAE empirically identifies 75% as optimal; this paper provides a theoretical explanation for that finding.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First theoretical framework to precisely quantify the behavior of mask pretraining
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Validated across four domains and multiple architectures
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are clearly presented
- Value: ⭐⭐⭐⭐⭐ Significant both theoretically and practically

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Towards Understanding the Shape of Representations in Protein Language Models](../../ICLR2026/computational_biology/towards_understanding_the_shape_of_representations_in_protein_language_models.md)
- [\[NeurIPS 2025\] Mol-LLaMA: Towards General Understanding of Molecules in Large Molecular Language Models](mol-llama_towards_general_understanding_of_molecules_in_large_molecular_language.md)
- [\[ICML 2025\] ExLM: Rethinking the Impact of \[MASK\] Tokens in Masked Language Models](../../ICML2025/computational_biology/exlm_rethinking_the_impact_of_mask_tokens_in_masked_language_models.md)
- [\[NeurIPS 2025\] Learning Repetition-Invariant Representations for Polymer Informatics](learning_repetition-invariant_representations_for_polymer_informatics.md)
- [\[NeurIPS 2025\] Diffusion Generative Modeling on Lie Group Representations](diffusion_generative_modeling_on_lie_group_representations.md)

</div>

<!-- RELATED:END -->

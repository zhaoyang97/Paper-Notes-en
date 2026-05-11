---
title: >-
  [Paper Note] Dependency Parsing is More Parameter-Efficient with Normalization
description: >-
  [NeurIPS 2025][Model Compression][dependency parsing] This paper identifies that the lack of normalization in biaffine scoring for dependency and semantic parsing leads to systematic overparameterization…
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "dependency parsing"
  - "biaffine scoring"
  - "normalization"
  - "parameter efficiency"
  - "overparameterization"
date: 2026-05-08
content_hash: 0996b5bd23522db4
---

# Dependency Parsing is More Parameter-Efficient with Normalization

**Conference**: NeurIPS 2025
**arXiv**: [2505.20215](https://arxiv.org/abs/2505.20215)
**Code**: [https://github.com/paolo-gajo/EfficientSDP](https://github.com/paolo-gajo/EfficientSDP)
**Area**: Model Compression
**Keywords**: dependency parsing, biaffine scoring, normalization, parameter efficiency, overparameterization

## TL;DR
This paper identifies that the lack of normalization in biaffine scoring for dependency and semantic parsing leads to systematic overparameterization, and demonstrates that a simple $1/\sqrt{d}$ scaling can reduce BiLSTM parameters by up to 85% while matching or surpassing original performance.

## Background & Motivation

**Background**: The dominant approach to dependency parsing follows Dozat & Manning's biaffine classifier, which encodes tokens with a BiLSTM and computes biaffine scores over all word pairs to predict arcs and relations. Current state-of-the-art systems typically stack 3 BiLSTM layers with hidden dimension 400 and MLP dimension 500.

**Limitations of Prior Work**: The biaffine scoring operation $QK^\top$ is structurally identical to Transformer self-attention, yet the vast majority of dependency parsing work omits score normalization. The Transformer uses $1/\sqrt{d_k}$ scaling to control variance and prevent softmax saturation — this insight has been consistently overlooked in the dependency parsing community.

**Key Challenge**: Without normalization, high-variance inputs cause softmax outputs to become polarized, leading to gradient vanishing/exploding. Additional BiLSTM layers are then implicitly required for regularization. This means the extra parameters compensate for the missing normalization rather than capturing richer linguistic features.

**Goal**: To demonstrate that the absence of normalization causes overparameterization, and to provide both theoretical and empirical evidence that introducing normalization permits substantial parameter reduction.

**Key Insight**: The analysis draws on implicit regularization theory — gradient descent on deep linear networks reduces the effective rank of weight matrices, and lower-rank weights produce outputs with smaller variance. Deeper BiLSTMs thus act as implicit normalizers via rank reduction.

**Core Idea**: Adding $a = 1/\sqrt{d}$ score scaling allows a 1-layer BiLSTM with a smaller parameter budget to match the performance of an unnormalized 3-layer BiLSTM, reducing parameter count by up to 85%.

## Method

### Overall Architecture
The paper follows the architecture of Bhatt et al.: a frozen BERT encoder → a Tagger (single-layer BiLSTM for POS tagging) → a Parser ($N$-layer BiLSTM with 4 MLP heads → biaffine scoring) → a Decoder (greedy/MST decoding). The sole modification is the addition of $1/\sqrt{d}$ scaling after biaffine scoring.

### Key Designs

1. **Biaffine Score Normalization**:

    - **Function**: Scales the biaffine score $s = QK^\top$ to $s / \sqrt{d}$, where $d$ is the MLP projection dimension.
    - **Theoretical Basis**: Assuming $q, k$ are zero-mean and unit-variance, $\text{Var}(s) = d_k$ and $\text{Std} = \sqrt{d_k}$. After scaling, each score has a standard deviation of approximately 1, preventing extreme softmax inputs.
    - **Design Motivation**: This is precisely the motivation behind $1/\sqrt{d_k}$ scaling in Transformer attention, a principle that the dependency parsing community has consistently overlooked.

2. **Theoretical Analysis of Depth as Implicit Normalization**:

    - **Result 1 (Implicit Regularization)**: In an $N$-layer linear network trained with gradient descent, singular values evolve as $\sigma_r(t+1) \leftarrow \sigma_r(t) - \eta \cdot \langle \nabla\mathcal{L}, \mathbf{u}_r \mathbf{v}_r^\top \rangle \cdot N \cdot \sigma_r(t)^{2-2/N}$. Larger $N$ accelerates the decay of small singular values, yielding lower effective rank.
    - **Claim 1 (Monotonic Rank–Variance Relationship)**: The output variance of the truncated SVD approximation $\mathbf{A}_r$, i.e., $\text{tr}(\text{Cov}(Y_r))$, increases monotonically with rank $r$.
    - **Corollary**: Deeper BiLSTMs → lower effective rank → lower output variance → more stable softmax. The superiority of 3-layer over 1-layer BiLSTMs is thus partly attributable not to richer feature extraction but to implicit normalization. With explicit normalization, these redundant layers become unnecessary.
    - **Empirical Verification**: Figure 1 confirms that effective rank $\rho(W)$ decreases monotonically with the number of BiLSTM layers.

3. **Parameter-Efficient Configuration Search**:

    - **Function**: Searches for optimal $(N, h_\psi, d_{\text{MLP}})$ configurations across datasets.
    - **Finding**: The baseline uses $(3, 400, 500)$; with normalization, the optimal configuration typically reduces to $(1, 200\text{–}400, 100\text{–}300)$, achieving approximately 85% parameter reduction.

### Loss & Training
Standard multi-task loss: $\mathcal{L} = \lambda_1 \mathcal{L}_{\text{tag}} + \lambda_2 (\mathcal{L}_{\text{edge}} + \mathcal{L}_{\text{rel}})$, with $\lambda_1 = 0.1$ and $\lambda_2 = 1$.

## Key Experimental Results

### Main Results (Labeled Arc Prediction, Micro-F1 / LAS)

| Model | Scaling $a$ | Layers $N$ | ADE | CoNLL04 | SciERC | ERFGC | enEWT | SciDTB |
|-------|------------|-----------|-----|---------|--------|-------|-------|--------|
| Baseline | 1 | 3 | 0.653 | 0.566 | 0.257 | 0.701 | 0.804 | 0.915 |
| Ours | $1/\sqrt{d}$ | 1 | 0.668 | 0.597 | 0.299 | 0.692 | 0.789 | 0.904 |
| Ours | $1/\sqrt{d}$ | 2 | 0.676 | 0.596 | 0.312 | 0.699 | 0.805 | 0.916 |
| **Ours** | $1/\sqrt{d}$ | **3** | **0.686** | **0.602** | **0.320** | **0.708** | **0.807** | **0.919** |

Normalization with 3 layers consistently outperforms the baseline across all 6 datasets. Normalization with a single layer already matches or exceeds the unnormalized 3-layer model on most datasets.

### Ablation Study (Normalization vs. Depth)

| Layers $N$ | No Normalization ($a=1$) | With Normalization ($a=1/\sqrt{d}$) | Gain |
|-----------|------------------------|-------------------------------------|------|
| 0 | 0.147 (SciERC) | 0.181 | +23% |
| 1 | 0.282 | 0.299 | +6% |
| 2 | 0.273 | 0.312 | +14% |
| 3 | 0.299 | 0.320 | +7% |

### Key Findings
- **The gap is largest at $N=0$ (no BiLSTM)**: On SciERC, unnormalized scores 0.147 vs. normalized 0.181 (+23%), since no implicit regularization is available to compensate.
- **The gap narrows as depth increases**: Consistent with theoretical predictions — more layers provide stronger implicit regularization, reducing the marginal benefit of explicit normalization.
- Normalization yields greater gains on harder tasks: SciERC, with sparse train/test entity overlap and complex dependency graphs, shows the largest improvements.
- The effect is cross-lingual: Consistent gains are observed across 6 languages in Universal Dependencies datasets.
- The effect is cross-domain: Similar phenomena are observed in non-linguistic settings, including molecular graph reasoning (QM9) and image superpixels (CIFAR10 Superpixel).
- A normalized 1-layer BiLSTM with dimensions $(200, 100)$ matches an unnormalized 3-layer model with dimensions $(400, 500)$, achieving approximately 85% parameter reduction.

## Highlights & Insights
- **An overlooked simple fix**: The NLP dependency parsing community has used unnormalized biaffine scorers for years, despite Transformers having established the necessity of normalization from the outset. This paper provides both theoretical and empirical evidence that the omission is not a harmless convention but a source of systematic overparameterization.
- **Elegant theoretical chain**: The implicit regularization → rank reduction → variance reduction argument provides a complete mechanistic explanation for why deeper BiLSTMs partially compensate for the lack of normalization.
- **High practical value**: Any model employing biaffine scoring — including those for relation extraction and coreference resolution — can immediately adopt $1/\sqrt{d}$ scaling at virtually zero cost for improved parameter efficiency.

## Limitations & Future Work
- Main experiments are conducted only with frozen BERT$_\text{base}$; the effect under fine-tuning or with larger pretrained models remains to be examined.
- The theoretical analysis assumes deep linear networks, whereas BiLSTMs are nonlinear — a gap exists between theory and practice.
- Only $1/\sqrt{d}$ normalization is evaluated; alternatives such as LayerNorm and RMSNorm are not explored.
- Validation on non-biaffine modern methods, such as LLM-based information extraction, is absent.

## Related Work & Insights
- **vs. Dozat & Manning (2017)**: The seminal biaffine parser does not use normalization, and this design choice has been inherited by subsequent work without scrutiny.
- **vs. Vaswani et al. (2017)**: Transformer attention adopts $1/\sqrt{d_k}$ scaling from the outset; this paper transfers that insight to dependency parsing.
- **vs. SENet/Attention normalization**: Channel and attention normalization have been extensively studied in computer vision; this paper identifies an analogous overlooked need in NLP structured prediction.
- The findings are informative for any model using dot-product scoring: practitioners should verify whether appropriate variance normalization is applied.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Identifies a problem overlooked by an entire subfield and provides a theoretical explanation, even though the fix itself is minimal.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers 6 NLP datasets, 6 languages, 3 non-linguistic datasets, extensive ablations, 5 random seeds, and statistical testing.
- **Writing Quality**: ⭐⭐⭐⭐ The logical chain from Result 1 → Claim 1 → empirical validation is rigorous and clearly presented.
- **Value**: ⭐⭐⭐⭐ The improvement is simple to apply but broadly impactful, with direct relevance to all work employing biaffine scoring.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Representation-Guided Parameter-Efficient LLM Unlearning](../../ACL2026/model_compression/representation-guided_parameter-efficient_llm_unlearning.md)
- [\[NeurIPS 2025\] Less is More but Where: Dynamic Token Compression via LLM-Guided Keyframe Prior](less_is_more_but_where_dynamic_token_compression_via_llm-guided_keyframe_prior.md)
- [\[ICCV 2025\] TR-PTS: Task-Relevant Parameter and Token Selection for Efficient Tuning](../../ICCV2025/model_compression/tr-pts_task-relevant_parameter_and_token_selection_for_efficient_tuning.md)
- [\[ICCV 2025\] Generalized Tensor-based Parameter-Efficient Fine-Tuning via Lie Group Transformations](../../ICCV2025/model_compression/generalized_tensor-based_parameter-efficient_fine-tuning_via_lie_group_transform.md)
- [\[ICLR 2026\] SeeDNorm: Self-Rescaled Dynamic Normalization](../../ICLR2026/model_compression/seednorm_self-rescaled_dynamic_normalization.md)

</div>

<!-- RELATED:END -->

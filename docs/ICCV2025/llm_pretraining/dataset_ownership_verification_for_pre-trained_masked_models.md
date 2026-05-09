---
title: >-
  [Paper Note] Dataset Ownership Verification for Pre-trained Masked Models
description: >-
  [ICCV 2025][LLM Pretraining][dataset ownership verification] DOV4MM proposes the first dataset ownership verification method tailored for masked pre-trained models. By comparing the embedding reconstruction difficulty of seen versus unseen samples, and applying a paired t-test, the method determines whether a black-box model was pre-trained on a specific dataset. It achieves p-values well below 0.05 across 10 masked image models and 4 masked language models.
tags:
  - ICCV 2025
  - LLM Pretraining
  - dataset ownership verification
  - masked modeling
  - embedding reconstruction difficulty
  - hypothesis testing
  - self-supervised learning
date: 2026-05-08
content_hash: 4bcdf063be537907
---

# Dataset Ownership Verification for Pre-trained Masked Models

**Conference**: ICCV 2025
**arXiv**: [2507.12022](https://arxiv.org/abs/2507.12022)
**Code**: [github.com/xieyc99/DOV4MM](https://github.com/xieyc99/DOV4MM)
**Area**: LLM Pre-training
**Keywords**: dataset ownership verification, masked modeling, embedding reconstruction difficulty, hypothesis testing, self-supervised learning

## TL;DR

DOV4MM proposes the first dataset ownership verification method tailored for masked pre-trained models. By comparing the embedding reconstruction difficulty of seen versus unseen samples, and applying a paired t-test, the method determines whether a black-box model was pre-trained on a specific dataset. It achieves p-values well below 0.05 across 10 masked image models and 4 masked language models.

## Background & Motivation

High-quality open-source datasets are foundational to deep learning, yet they face the threat of unauthorized commercial use. Dataset ownership verification (DOV) aims to detect whether a suspicious model was trained on a particular dataset. Existing methods suffer from the following limitations:

**Designed for supervised models**: Most DOV methods rely on the relationship between data points and decision boundaries, making them inapplicable to self-supervised models.

**Reliance on backdoor watermarking**: Injecting watermarks into data degrades model performance and is vulnerable to watermark removal attacks.

**Inapplicability to masked models**: Recent DOV methods for contrastive learning exploit gaps in contrastive embedding relationships, but masked modeling (MAE, BEiT, etc.) differs significantly from contrastive learning in its pretext task, making representations harder to distinguish.

Core observation: **Masked pre-trained models exhibit significantly lower embedding reconstruction difficulty for seen samples than for unseen ones.**

## Method

### Overall Architecture

DOV4MM operates in a black-box setting (the defender can only obtain feature vectors via API) and consists of three key steps:

1. Randomly partition the public dataset into a training set $\mathcal{D}_t$ and a validation set $\mathcal{D}_v$, and train a decoder using $\mathcal{D}_t$;
2. Compute the relative embedding reconstruction difficulty of the suspicious model on three datasets;
3. Apply a one-sided paired t-test to determine whether the model was pre-trained on the defender's dataset.

### Key Designs

1. **Embedding Reconstruction Difficulty**: Given a pre-trained masked model $M$, input-space mask $\boldsymbol{t}$ and embedding-space mask $\hat{\boldsymbol{t}}$, and decoder $M_d$, the reconstruction difficulty of a sample $\boldsymbol{x}$ is defined as:

$$R(\boldsymbol{x}, \boldsymbol{t}, \hat{\boldsymbol{t}}, M, M_d) = \frac{\|[M_d(\boldsymbol{e_t}) - \boldsymbol{e}] \odot (\boldsymbol{1} - \hat{\boldsymbol{t}})\|_2^2}{\|\boldsymbol{1} - \hat{\boldsymbol{t}}\|_1}$$

where $\boldsymbol{e} = M(\boldsymbol{x})$ is the full embedding and $\boldsymbol{e_t} = M(\boldsymbol{x} \odot \boldsymbol{t})$ is the masked embedding. Reconstruction error is computed only at masked positions, reflecting the difficulty of recovering missing information.

2. **Relative Embedding Reconstruction Difficulty**: To amplify the discrepancy between seen and unseen samples, a relative metric is introduced. Using $\mathcal{D}_t$ as a baseline, the reconstruction difficulty differences of $\mathcal{D}_v$ and $\mathcal{D}_{pvt}$ relative to $\mathcal{D}_t$ are computed as:

$$\Delta\mathcal{R} = \{\overline{R'}_k - \overline{R}_k | k \in [1, K]\}$$

Over $K=30$ random sampling rounds, each drawing $N=1024$ samples, a sequence of paired differences is obtained.

3. **Hypothesis Testing Decision**: A one-sided paired t-test is applied to $\Delta\mathcal{R}_{vt}$ (relative difficulty of the validation set) and $\Delta\mathcal{R}_{pt}$ (relative difficulty of the private set). The null hypothesis $H_0$: the mean difference between $\Delta\mathcal{R}_{pt}$ and $\Delta\mathcal{R}_{vt}$ is $\leq 0$. If the p-value $< 0.05$, $H_0$ is rejected and the model is deemed to have illegally used the dataset. The core rationale is that if the model was indeed trained on $\mathcal{D}_{pub}$, then $\mathcal{D}_v$ (a subset of $\mathcal{D}_{pub}$) should exhibit lower reconstruction difficulty than $\mathcal{D}_{pvt}$ (never-seen data).

### Loss & Training

- Decoder $M_d$: Transformer architecture (512-dim, 8 layers, 16 heads), trained for 50 epochs, batch size 64, learning rate 1e-3;
- Masking strategy: random masking with a masking ratio of 75%;
- Only 20,000 samples from $\mathcal{D}_t$ (approximately 3% of ImageNet-1K) are needed to achieve accurate verification.

## Key Experimental Results

### Main Results

**Classification verification capability on ImageNet-1K subsets**:

| Dataset | Method | Sensitivity | Specificity | AUROC |
|--------|------|--------|--------|-------|
| ImageNet-50 | DI4SSL | 0.00 | 1.00 | 0.50 |
| ImageNet-50 | CTRL | 1.00 | 0.00 | 0.50 |
| ImageNet-50 | PartCrop | 0.00 | 0.22 | 0.39 |
| ImageNet-50 | **DOV4MM** | **1.00** | **1.00** | **1.00** |
| ImageNet-100 | **DOV4MM** | **1.00** | **1.00** | **1.00** |

**p-value results on ImageNet-1K (10 MIM methods)**:

| Model | MIM Method | IN-1K (illegal) | Food101 (legal) | COCO (legal) | Places365 (legal) |
|------|---------|-------------|----------------|-------------|-----------------|
| ViT-B/16 | MAE | $10^{-5}$ ✓ | 0.99 ✓ | 0.98 ✓ | 0.99 ✓ |
| ViT-B/16 | BEiT v2 | $10^{-5}$ ✓ | 0.99 ✓ | 0.99 ✓ | 0.99 ✓ |
| ViT-L/16 | MAE | $10^{-6}$ ✓ | 0.99 ✓ | 0.99 ✓ | 0.99 ✓ |
| Swin-B | SimMIM | 0.03 ✓ | 0.99 ✓ | 0.98 ✓ | 0.98 ✓ |

All 10 MIM methods across 4 architectures are successfully verified using only 3% of ImageNet-1K data.

### Ablation Study

| Configuration | MAE p-value | CAE p-value | iBOT p-value | Note |
|------|---------|---------|----------|------|
| Decoder dim 128 | $10^{-5}$ | $10^{-3}$ | $10^{-3}$ | All effective |
| Decoder dim 1024 | $10^{-7}$ | 0.01 | $10^{-3}$ | Larger not always better |
| Decoder layers 4 | $10^{-5}$ | $10^{-3}$ | $10^{-3}$ | Stable |
| Decoder layers 12 | $10^{-6}$ | 0.01 | $10^{-3}$ | Stable |
| Training set 10k | $10^{-4}$ | 0.02 | 0.01 | Effective with less data |
| Training set 50k | $10^{-6}$ | $10^{-3}$ | $10^{-3}$ | More data improves results |

### Key Findings

- DOV4MM is robust to decoder architecture choices (dimension, depth, number of heads), with p-values well below 0.05 across all configurations;
- Ownership of million-scale datasets can be accurately verified using only 3% of the data (~20k samples);
- The method is equally effective on 4 masked language models (e.g., BERT) evaluated on WikiText-103, demonstrating cross-modal generality.

## Highlights & Insights

1. **Pioneering contribution**: The first dataset ownership verification method specifically designed for masked pre-trained models, filling an important gap in the field;
2. **Watermark-free**: The original dataset distribution is not modified, avoiding both the performance degradation caused by watermark injection and the risk of watermark removal;
3. **Minimal data requirement**: Accurate verification is achieved with only 3% of the data, significantly reducing computational cost compared to DI4SSL, which requires inference over the entire dataset;
4. **Statistical rigor**: The hypothesis testing framework based on the paired t-test provides strict statistical guarantees rather than relying on simple threshold-based decisions;
5. **Cross-modal generality**: Applicable to both visual masked models (MAE, BEiT, etc.) and language masked models (BERT, etc.).

## Limitations & Future Work

- A private dataset $\mathcal{D}_{pvt}$ that is domain-independent from the suspicious model's training data is required; if the private data is too similar to the public data, detection sensitivity may decrease;
- The p-value for Swin-B/L (0.03) is close to the 0.05 threshold, indicating room for improvement in robustness across certain architectures;
- The black-box setting assumes that embedding vectors can be obtained via API (Embedding-as-a-Service); the method cannot be applied if only classification outputs are available;
- Verification effectiveness after downstream fine-tuning of the model has not yet been evaluated.

## Related Work & Insights

- Unlike dataset inference (DI4SSL): DI4SSL requires inferring the likelihood distribution of the entire dataset, incurring high computational cost; DOV4MM only needs a small number of samples to compute reconstruction difficulty differences;
- Unlike membership inference (PartCrop): PartCrop directly uses high-dimensional representations for membership decisions, which contain substantial redundant information; DOV4MM extracts the most informative signal via the relative embedding reconstruction difficulty metric;
- The core mechanism is generalizable to other generative pre-training paradigms (e.g., differences in denoising difficulty for diffusion models).

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Strongly pioneering; the concept of relative embedding reconstruction difficulty is both intuitively motivated and rigorously defined
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 10 MIM methods, 4 architectures, cross-modal validation on language models, and comprehensive ablation studies
- **Writing Quality**: ⭐⭐⭐⭐ Mathematical definitions are rigorous and the methodological pipeline is clearly presented
- **Value**: ⭐⭐⭐⭐ An important contribution to data security with strong practical applicability

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] How Does Sequence Modeling Architecture Influence Base Capabilities of Pre-trained Language Models?](../../NeurIPS2025/llm_pretraining/how_does_sequence_modeling_architecture_influence_base_capabilities_of_pre-train.md)
- [\[ACL 2026\] SCRIPT: A Subcharacter Compositional Representation Injection Module for Korean Pre-Trained Language Models](../../ACL2026/llm_pretraining/script_a_subcharacter_compositional_representation_injection_module_for_korean_p.md)
- [\[AAAI 2026\] PrefixGPT: Prefix Adder Optimization by a Generative Pre-trained Transformer](../../AAAI2026/llm_pretraining/prefixgpt_prefix_adder_optimization_by_a_generative_pre-trained_transformer.md)
- [\[AAAI 2026\] No-Regret Strategy Solving in Imperfect-Information Games via Pre-Trained Embedding](../../AAAI2026/llm_pretraining/no-regret_strategy_solving_in_imperfect-information_games_via_pre-trained_embedd.md)
- [\[ICCV 2025\] ACE-G: Improving Generalization of Scene Coordinate Regression Through Query Pre-Training](ace-g_improving_generalization_of_scene_coordinate_regression_through_query_pre-.md)

<!-- RELATED:END -->

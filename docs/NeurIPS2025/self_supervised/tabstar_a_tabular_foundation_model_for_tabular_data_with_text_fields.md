---
title: >-
  [Paper Note] TabSTAR: A Tabular Foundation Model for Tabular Data with Text Fields
description: >-
  [NeurIPS 2025][Self-Supervised Learning][tabular foundation model] TabSTAR is a foundation model designed specifically for tabular data with text fields. It achieves target-aware text representations through end-to-end o…
tags:
  - "NeurIPS 2025"
  - "Self-Supervised Learning"
  - "tabular foundation model"
  - "text fields"
  - "target-aware tokens"
  - "cross-dataset learning"
  - "transfer learning"
date: 2026-05-08
content_hash: c9528ccda6ca1db8
---

# TabSTAR: A Tabular Foundation Model for Tabular Data with Text Fields

**Conference**: NeurIPS 2025
**arXiv**: [2505.18125](https://arxiv.org/abs/2505.18125)  
**Code**: [https://github.com/alanarazi7/TabSTAR](https://github.com/alanarazi7/TabSTAR)  
**Area**: Tabular Learning / Foundation Models
**Keywords**: tabular foundation model, text fields, target-aware tokens, cross-dataset learning, transfer learning

## TL;DR
TabSTAR is a foundation model designed specifically for tabular data with text fields. It achieves target-aware text representations through end-to-end optimization with an unfrozen text encoder (e5-small-v2), injects target semantics via target-aware tokens, and enables cross-dataset transfer learning through a dataset-parameter-free architecture. After pre-training on 350 datasets, TabSTAR surpasses CatBoost-Tuned (4h tuning) on 12 out of 14 classification datasets and outperforms TabPFN-v2 on 8 out of 11 datasets.

## Background & Motivation

**Background**: Tabular learning has long been dominated by GBDTs (XGBoost/LightGBM/CatBoost). Tabular foundation models (TFMs) such as TabPFN-v2 have begun to outperform GBDTs on medium-scale datasets, yet they still rely on static embeddings when handling text fields—no differently from GBDTs.

**Limitations of Prior Work**: (a) Real-world tabular data frequently contains free-text fields (e.g., medical records, product descriptions), yet mainstream benchmarks largely ignore such data—half of the benchmark datasets are over 20 years old. (b) Existing methods encode text using frozen embedding models, producing **target-agnostic** static representations—the same text embedding is used regardless of whether the target is "patient discharge" or "treatment cost." (c) TFMs typically require dataset-specific output layers (e.g., TP-BERTa trains separate models for classification and regression), which limits cross-dataset learning.

**Key Challenge**: Semantic information in text fields is critical for prediction, yet effective utilization requires **target-aware** representations—the same medical report should surface different signals for different prediction tasks. Existing methods overlook this.

**Goal**: To construct a foundation model that processes tabular data with text fields end-to-end, supports cross-dataset transfer learning, and requires no dataset-specific parameters.

**Key Insight**: Unfreeze the text encoder to optimize text representations end-to-end with respect to the prediction target, and introduce target-aware tokens by treating each possible value of the target variable as an input token, enabling target-aware representations to emerge through feature–target interaction.

**Core Idea**: Unfrozen text encoder + target-aware tokens + elimination of dataset-specific parameters = the first foundation model to comprehensively outperform both GBDTs and TabPFN-v2 on text-inclusive tabular classification tasks.

## Method

### Overall Architecture
TabSTAR involves two training phases: (1) **Pre-training** on 350 tabular datasets (253 classification + 97 regression) in a multi-task setting; (2) **Fine-tuning** on individual downstream datasets using LoRA. The architecture comprises five modules: Verbalization → Encoding (semantic + numerical) → Fusion (per-feature) → Interaction (cross-feature self-attention) → Prediction (shared prediction head).

### Key Designs

1. **Verbalization**:

    - *Function*: Each feature is converted into a textual description of the form "column name: value." Numerical features additionally undergo z-score normalization and quantile discretization (e.g., "Age: 40-50 (Quantile 50-60%)").
    - The target variable is also verbalized: for classification tasks, each possible target value is treated as an independent token (e.g., "Target. Decision: Hospitalized" and "Target. Decision: Released").
    - *Design Motivation*: Column name information is injected into the representations (ignored by most methods); quantile-based descriptions compensate for language models' weakness in tokenizing precise numerical values.

2. **Target-Aware Tokens**:

    - *Function*: Each of the $C$ possible values of the classification target is introduced as a separate token and participates alongside feature tokens in Transformer interaction.
    - *Mechanism*: During the Interaction stage, target tokens attend to all feature tokens via self-attention—the output representation of each target-value token encodes information about "how likely this value is the correct answer." At prediction time, all target tokens share a single classification head, and a softmax yields the probability distribution.
    - *Design Motivation*: (a) Eliminates dataset-specific parameters—regardless of the number of classes or class names, the same shared prediction head is used across all datasets. (b) Target semantic information is supplied as input rather than only as a label, enabling the model to understand the semantic distinction between "Released" and "Hospitalized."

3. **Unfrozen Text Encoder**:

    - *Function*: e5-small-v2 is used as the semantic encoder, with the upper half of its layers unfrozen so that text representations are optimized end-to-end with respect to the prediction target.
    - *Mechanism*: Frozen encoder → static representations → target-agnostic. Unfrozen encoder → representations adapt during pre-training/fine-tuning → target-aware.
    - *Ablation*: Performance peaks when 6 layers are unfrozen; even unfreezing a single layer yields substantial gains over full freezing. Classification normalized score improves from ~0.3 (frozen) to ~0.6 (6 layers unfrozen).

4. **No Dataset-Specific Parameters**:

    - *Function*: The entire architecture—from encoding to prediction—shares parameters across datasets, with no dataset-specific output layers.
    - *Design Motivation*: TP-BERTa requires separate models for classification and regression; XTab requires dataset-specific parameters. TabSTAR eliminates these constraints, enabling joint pre-training across 350 datasets within a single model.

### Loss & Training
- **Pre-training**: 350 datasets (manually deduplicated); 48 hours on a single A40 GPU.
- **Fine-tuning**: LoRA (adapters applied only to the layers unfrozen during pre-training); default hyperparameters without dataset-specific tuning.
- **Evaluation**: 5-fold cross-validation to prevent data leakage—each dataset is evaluated only by a pre-training variant that has not seen it.

## Key Experimental Results

### Main Results (50-dataset benchmark with text fields)

**Classification (14 datasets, 10K condition, normalized score)**:

| Method | Normalized Score | Notes |
|--------|-----------------|-------|
| **TabSTAR** | **0.83** | Wins CatBoost-Tuned 12/14, TabPFN-v2 8/11 |
| TabM-Tuned | 0.73 | |
| CatBoost-Tuned (4h) | 0.69 | |
| TabPFN-v2 | 0.67 | |
| CARTE | 0.50 | |

**Regression (36 datasets)**: TabSTAR remains competitive but does not reach SOTA; GBDTs still dominate.

**Unlimited condition (>10K samples)**: TabSTAR-Unlimited 0.84 vs. TabM-Tuned 0.79, substantially outperforming all others (other TFMs cannot scale to this regime).

### Ablation Study

| Configuration | Classification Score | Regression Score |
|---------------|---------------------|-----------------|
| Frozen encoder | ~0.30 | ~0.30 |
| 1 layer unfrozen | ~0.45 | ~0.45 |
| 6 layers unfrozen (TabSTAR) | **~0.59** | **~0.60** |
| No pre-training | Significantly lower | — |
| Pre-trained on 16 datasets | Low | Low |
| Pre-trained on 256 datasets | **High** | **High** |

**Numerical verbalization ablation**:

| Method | Classification | Regression |
|--------|---------------|-----------|
| Column name only | 0.386 | 0.386 |
| Column name + quantile interval | 0.544 | 0.584 |
| TabSTAR (full) | **0.593** | **0.596** |

### Key Findings
- **Unfreezing the encoder is the central contribution**: Moving from frozen to 6 layers unfrozen doubles the classification score, confirming that text fields require dynamic, target-aware representations rather than static embeddings.
- **Pre-training dataset count follows a scaling law**: Performance improves consistently as the number of pre-training datasets increases from 16 to 64 to 256—regression tasks benefit more, suggesting that further scaling may bring regression to SOTA as well.
- **Target-aware tokens eliminate dataset-specific parameters**: A single shared classification head handles all numbers and names of classes across all datasets, enabling true cross-dataset parameter sharing.
- **Inference efficiency is comparable to GBDTs**: TabSTAR inference takes 3.0s vs. CatBoost 2.0s (GPU), far faster than TabICL (13.4s) and TabDPT (161.4s).
- **Regression remains a weakness**: GBDTs still dominate regression, and TabPFN-v2 also falls short—indicating that text-inclusive tabular regression is an open problem.

## Highlights & Insights
- **"Unfreezing the text encoder" appears simple but has far-reaching consequences**: All prior methods (CM2/CARTE/TP-BERTa) freeze the language model; TabSTAR is the first to demonstrate the decisive role of unfreezing and end-to-end optimization for tabular data with text fields.
- **Target-aware tokens represent an elegant zero-dataset-specific-parameter design**: Treating class labels as input tokens rather than output layer parameters allows datasets with 2 classes and datasets with 100 classes to share an identical architecture and parameter set.
- **Quantile-based strategy for numerical verbalization**: Describing numerical ranges in natural language (e.g., "Age: 40-50, Quantile 50-60%") is more natural than TP-BERTa's special bin tokens and exploits the language model's capacity to understand numerical descriptions.

## Limitations & Future Work
- Regression performance lags substantially behind GBDTs—regression-via-classification techniques (as in TabPFN-v2) or richer numerical encodings may be needed.
- Training is slow (493s vs. CatBoost 69s), though inference speed is comparable to GBDTs.
- Memory bottlenecks may arise with high-dimensional feature spaces due to the $O(n^2)$ complexity of Transformer self-attention.
- Pre-training covers only 350 datasets; the scaling potential has not been fully explored.
- Evaluation on purely numerical tabular data and few-shot settings remains limited.

## Related Work & Insights
- **vs. TabPFN-v2**: TabPFN-v2 operates under an ICL paradigm (up to 10K samples) and uses static embeddings for text. TabSTAR optimizes text end-to-end with no scale constraint, winning on 8/11 classification datasets.
- **vs. CARTE**: CARTE models high-cardinality features with graph structures but does not handle long text and cannot scale beyond ~2K samples.
- **vs. TP-BERTa**: TP-BERTa freezes RoBERTa and requires dataset-specific output layers. TabSTAR unfreezes the encoder and uses a shared prediction head.
- **vs. AutoGluon-Multimodal**: AutoGluon-Multimodal is a multi-model ensemble system rather than a single model. TabSTAR is a single end-to-end model.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of unfrozen encoder and target-aware tokens is novel; eliminating dataset-specific parameters carries significant engineering value.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 50-dataset benchmark + 14+ baselines + three-dimensional ablations (encoder / pre-training / verbalization) + cost analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation is clear, architecture diagrams are intuitive, ablations are thorough, and limitations are honestly discussed.
- Value: ⭐⭐⭐⭐⭐ A significant advance in foundation models for text-inclusive tabular data, with direct applicability to domains such as healthcare and finance where text fields are prevalent.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Hybrid Autoencoders for Tabular Data: Leveraging Model-Based Augmentation in Low-Label Settings](hybrid_autoencoders_for_tabular_data_leveraging_model-based_augmentation_in_low-.md)
- [\[NeurIPS 2025\] TabArena: A Living Benchmark for Machine Learning on Tabular Data](tabarena_a_living_benchmark_for_machine_learning_on_tabular_data.md)
- [\[NeurIPS 2025\] Mitra: Mixed Synthetic Priors for Enhancing Tabular Foundation Models](mitra_mixed_synthetic_priors_for_enhancing_tabular_foundation_models.md)
- [\[NeurIPS 2025\] Uncertainty-Guided Model Selection for Tabular Foundation Models in Biomolecule Efficacy Prediction](uncertainty-guided_model_selection_for_tabular_foundation_models_in_biomolecule_.md)
- [\[AAAI 2026\] Robust Tabular Foundation Models](../../AAAI2026/self_supervised/robust_tabular_foundation_models.md)

</div>

<!-- RELATED:END -->

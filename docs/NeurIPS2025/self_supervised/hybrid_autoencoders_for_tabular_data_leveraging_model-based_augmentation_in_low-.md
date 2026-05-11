---
title: >-
  [Paper Note] Hybrid Autoencoders for Tabular Data: Leveraging Model-Based Augmentation in Low-Label Settings
description: >-
  [NeurIPS 2025][Self-Supervised Learning][tabular data] This paper proposes TANDEM (Tree-And-Neural Dual Encoder Model), a hybrid autoencoder architecture that jointly trains a neural network encoder and an Oblivious Soft…
tags:
  - "NeurIPS 2025"
  - "Self-Supervised Learning"
  - "tabular data"
  - "hybrid autoencoder"
  - "oblivious soft decision tree"
  - "low-label learning"
date: 2026-05-08
content_hash: 03c1cce47571028c
---

# Hybrid Autoencoders for Tabular Data: Leveraging Model-Based Augmentation in Low-Label Settings

**Conference**: NeurIPS 2025
**arXiv**: [2511.06961](https://arxiv.org/abs/2511.06961)
**Code**: None
**Area**: Self-Supervised Learning / Tabular Data
**Keywords**: tabular data, self-supervised learning, hybrid autoencoder, oblivious soft decision tree, low-label learning

## TL;DR

This paper proposes TANDEM (Tree-And-Neural Dual Encoder Model), a hybrid autoencoder architecture that jointly trains a neural network encoder and an Oblivious Soft Decision Tree (OSDT) encoder, and introduces a sample-level stochastic gating network as a learnable data augmentation mechanism. TANDEM achieves superior performance over strong baselines—including tree-based and deep learning methods—in low-label tabular data settings.

## Background & Motivation

**Background**: Tabular data is the dominant data format in domains such as healthcare and finance. Gradient-boosted decision trees (GBDT/XGBoost/CatBoost) typically outperform deep neural networks on tabular data and remain the preferred choice in practice.

**Limitations of Prior Work**: (1) Neural networks exhibit spectral bias, tending to fit smooth low-frequency functions and struggling to capture complex high-frequency patterns in tabular data; (2) Self-supervised learning (SSL) for tabular data faces a data augmentation challenge—common augmentations such as noise injection or feature value swapping can easily destroy critical feature relationships; (3) Masked autoencoders (MAE) also have limitations on heterogeneous tabular data.

**Key Challenge**: SSL is particularly valuable in low-label settings, yet effective augmentation strategies for tabular data are lacking, and conventional augmentation methods tend to generate unrealistic samples.

**Goal**: To learn effective self-supervised representations on low-label tabular data that surpass conventional methods on downstream classification and regression tasks.

**Key Insight**: Replace data augmentation with model-based augmentation—leveraging the inductive biases of tree models to guide neural networks toward better representations.

**Core Idea**: Use an OSDT encoder as a "model-based augmentor" during training, transferring the tabular-friendly inductive biases of tree models to the neural network encoder via a shared decoder and alignment losses.

## Method

### Overall Architecture

TANDEM is a dual-encoder, shared-decoder masked autoencoder:
- Input $x \in \mathbb{R}^D$ is first passed through a sample-level stochastic gating network (STG) to produce a feature mask $g(x) \in [0,1]^D$
- The masked view $\tilde{x} = x \odot g(x)$ is fed in parallel to: (i) a fully connected neural encoder → $z^{NN}$, and (ii) an OSDT ensemble encoder → $z^{OSDT}$
- A shared decoder $h$ reconstructs $\hat{x}^{NN}$ and $\hat{x}^{OSDT}$ from the respective latent representations
- After pretraining, only the neural encoder and a lightweight classification/regression head are used at inference

### Key Designs

1. **Oblivious Soft Decision Tree (OSDT) Encoder**:

    - **Function**: Serves as a differentiable tree encoder that extracts structured representations from tabular data
    - **Design Motivation**: Tree models are naturally suited to tabular data, capable of capturing sharp high-frequency patterns and conditional feature interactions, thereby compensating for the spectral bias of neural networks
    - **Mechanism**: An ensemble of oblivious decision trees of fixed depth $L$, with a shared projection vector $w_\ell$ at each layer. Soft routing probabilities are computed as: $p_{\text{leaf}}(x) = \prod_{\ell=1}^{L} [\sigma_\ell^+(x)]^{b_\ell} \cdot [\sigma_\ell^-(x)]^{1-b_\ell}$. The final representation is the mean leaf distribution across all trees: $z^{OSDT}(x) = \frac{1}{T}\sum_{t=1}^T f_t^{OSDT}(x) \in \mathbb{R}^{2^L}$
    - **Novelty**: The OSDT encoder is used only during training and discarded at inference, avoiding the generalization limitations of tree models

2. **Stochastic Gating Network (SGN) as Sample-Level Augmentation**:

    - **Function**: Learns a feature mask for each input sample, enabling sample-level feature selection
    - **Design Motivation**: Replaces conventional fixed augmentations (noise, swapping, etc.) with a learnable input transformation that preserves semantic structure
    - **Mechanism**: The gating network $f_\theta(x)$ outputs parameters $\mu(x)$; gates are sampled via truncated Gaussian perturbation: $g(x) = \max(0, \min(1, 0.5 + \mu(x) + \epsilon))$, $\epsilon \sim \mathcal{N}(0, \sigma^2)$
    - **Novelty**: The neural encoder uses a single global gate, whereas the OSDT encoder employs an independent gate $g_\ell^{OSDT}(x)$ at each tree layer, enabling hierarchical feature selection

3. **Joint Training Objective**:

    - **Reconstruction loss**: $\mathcal{L}_{\text{recon}} = \frac{1}{N}\sum(\|x - \hat{x}^{OSDT}\|_2^2 + \|x - \hat{x}^{NN}\|_2^2)$
    - **Alignment loss**: $\mathcal{L}_{\text{align}} = \frac{1}{N}\sum\|\hat{x}^{OSDT} - \hat{x}^{NN}\|_2^2$ (reconstruction output consistency)
    - **Latent Representation Similarity (LRS) loss**: $\mathcal{L}_{\text{LRS}} = \frac{1}{N}\sum(1 - \frac{\langle z^{NN}, z^{OSDT} \rangle}{\|z^{NN}\| \cdot \|z^{OSDT}\|})$ (cosine distance)

### Loss & Training

- Pretraining for 100 epochs, batch size 128, RMSprop optimizer
- Hyperparameters selected via Optuna over 50 trials based on validation loss
- Downstream evaluation: single-layer MLP; the encoder is frozen for 25 epochs and then fine-tuned for 25 epochs
- The gating network is frozen during fine-tuning

## Key Experimental Results

### Main Results

**Classification (19 datasets, 400 labels):**

| Method | Mean Accuracy | Mean Rank |
|--------|--------------|-----------|
| MLogReg | 0.6380 | 6.16 |
| MLP | 0.6721 | 4.84 |
| XGBoost | 0.6706 | 4.47 |
| CatBoost | 0.6731 | 4.16 |
| TabPFN | 0.7012 | 2.56 |
| **TANDEM** | **0.7124** | **1.58** |

**Regression (13 datasets, 400 labels):**

| Method | Mean MSE | Mean Rank |
|--------|---------|-----------|
| CatBoost | 0.3318 | 4.00 |
| XGBoost | 0.3405 | 4.15 |
| MLP | 0.3877 | 4.38 |
| **TANDEM** | **0.3234** | **3.38** |

TANDEM achieves the best mean metric and best average rank on both classification and regression.

### Ablation Study

**Classification ablation (400 labels):**

| Variant | Mean Accuracy | Mean Rank |
|---------|--------------|-----------|
| SS-AE (standard autoencoder) | 0.6815 | 4.45 |
| SS-AE + Gating | 0.6941 | 3.61 |
| OSDT AE + Gating (tree only) | 0.6600 | 4.71 |
| TANDEM (no gate) | 0.6966 | 2.92 |
| TANDEM (no LRS + Align) | 0.6971 | 2.79 |
| **TANDEM (full)** | **0.7124** | **1.74** |

- Removing either encoder or the gating network consistently degrades performance
- The full TANDEM model is uniformly best
- The OSDT-only encoder variant performs worst (0.6600), indicating that tree models alone are insufficiently flexible as encoders

### Key Findings

- The dual-encoder architecture significantly outperforms single-encoder variants, validating the value of complementary inductive biases
- The learnable gating network is more effective as an augmentation mechanism than fixed augmentations
- TANDEM is robust across a wide label range of 50–1000 labeled samples
- Spectral analysis reveals that the two encoders capture distinct and complementary frequency components
- TANDEM achieves the best results even on classification benchmarks where TabPFN is strongest

## Highlights & Insights

- **Model-based augmentation replacing data augmentation**: The core insight is to inject tree model inductive biases into neural network training as a form of augmentation, rather than relying on unreliable tabular data augmentations
- **Neural-network-only inference**: The OSDT encoder is used only during training and incurs no inference overhead, maintaining flexible compatibility with downstream tasks
- **Dual role of the gating network**: It simultaneously acts as a feature selector and an augmentor, providing effective input transformations without disrupting semantic structure
- **Complementary spectral analysis**: The dual-encoder design is explained from a frequency-domain perspective—the neural network captures low-frequency components while the tree captures high-frequency components

## Limitations & Future Work

- Pretraining requires approximately 2,000 unlabeled samples per class, which may be impractical in extreme low-data scenarios
- The OSDT depth $L$ and number of trees $T$ are critical hyperparameters that require careful tuning
- On individual datasets, TANDEM can still underperform certain baselines (e.g., MSE of 1.0057 vs. CatBoost's 0.6565 on the BF regression dataset)
- Integration with Transformer-based tabular methods (e.g., SAINT, FT-Transformer) remains unexplored

## Related Work & Insights

- **NODE (popov2019node)**: Foundational work on differentiable oblivious decision trees
- **TabPFN (hollmann2023tabpfn)**: A probabilistic inference-based classification method pretrained on synthetic data; primary competitor
- **VIME / SCARF / SubTab**: Existing tabular SSL methods, all of which underperform TANDEM
- **STG (yamada2020)**: Original proposal of the stochastic gating network
- Insight: Exploiting the complementary inductive biases of heterogeneous models is an effective strategy for improving SSL representation quality, and the approach is generalizable to other model combinations

## Rating

- Novelty: ⭐⭐⭐⭐ The dual-encoder design with model-based augmentation is novel, and the perspective of treating gating as augmentation is distinctive
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 19 classification + 13 regression datasets, 100 repetitions, label range of 50–1000, comprehensive ablations
- Writing Quality: ⭐⭐⭐⭐ Motivation is clear; method is presented systematically and completely
- Value: ⭐⭐⭐⭐ Practically significant for low-label tabular learning, with strong generalizability

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] TabSTAR: A Tabular Foundation Model for Tabular Data with Text Fields](tabstar_a_tabular_foundation_model_for_tabular_data_with_text_fields.md)
- [\[NeurIPS 2025\] TabArena: A Living Benchmark for Machine Learning on Tabular Data](tabarena_a_living_benchmark_for_machine_learning_on_tabular_data.md)
- [\[NeurIPS 2025\] Memory-Integrated Reconfigurable Adapters: A Unified Framework for Settings with Multiple Tasks](memory-integrated_reconfigurable_adapters_a_unified_framework_for_settings_with_.md)
- [\[ICCV 2025\] To Label or Not to Label: PALM – A Predictive Model for Evaluating Sample Efficiency in Active Learning Models](../../ICCV2025/self_supervised/to_label_or_not_to_label_palm_-_a_predictive_model_for_evaluating_sample_efficie.md)
- [\[NeurIPS 2025\] Uncertainty-Guided Model Selection for Tabular Foundation Models in Biomolecule Efficacy Prediction](uncertainty-guided_model_selection_for_tabular_foundation_models_in_biomolecule_.md)

</div>

<!-- RELATED:END -->

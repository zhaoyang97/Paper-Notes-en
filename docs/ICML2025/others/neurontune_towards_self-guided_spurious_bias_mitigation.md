---
title: >-
  [Paper Note] NeuronTune: Towards Self-Guided Spurious Bias Mitigation
description: >-
  [ICML2025][Spurious Correlation] NeuronTune proposes a **group-label-free** self-guided debiasing method: by comparing the difference in neuron activations between correctly and incorrectly predicted samples in the model's latent space, it identifies the dimensions affected by spurious biases and sets them to zero. It then retrains the final classification layer, significantly improving the worst-group accuracy.
tags:
  - "ICML2025"
  - "Spurious Correlation"
  - "Bias Mitigation"
  - "Neuron Pruning"
  - "Last-layer Retraining"
  - "Worst-group Accuracy"
date: 2026-05-08
content_hash: 89a8a063dc1c4cc5
---

# NeuronTune: Towards Self-Guided Spurious Bias Mitigation

**Conference**: ICML2025  
**arXiv**: [2505.24048](https://arxiv.org/abs/2505.24048)  
**Code**: [GitHub](https://github.com/gtzheng/NeuronTune)  
**Area**: Robustness / Debiasing  
**Keywords**: Spurious Correlation, Bias Mitigation, Neuron Pruning, Last-layer Retraining, Worst-group Accuracy

## TL;DR

NeuronTune proposes a **group-label-free** self-guided debiasing method: by comparing the difference in neuron activations between correctly and incorrectly predicted samples in the model's latent space, it identifies the dimensions affected by spurious biases and sets them to zero. It then retrains the final classification layer, significantly improving the worst-group accuracy.

## Background & Motivation

- **Spurious Correlation Problem**: Models trained with ERM easily rely on non-causal features (e.g., relying on the water background instead of the bird itself for waterbird classification), performing poorly on test data lacking these spurious correlations.
- **Limitations of Prior Work**:
    - **Supervised methods** require group labels $(y, a)$ indicating which spurious attribute is associated with which class, which holds high annotation costs.
    - **Semi-supervised methods** (JTT, DFR, AFR, etc.) still require group labels on the validation set for model selection.
    - **Sample-level methods** cannot directly intervene in the internal decision-making mechanism of the model, offering limited control precision.
- **Core Motivation**: Can we automatically discover neurons affected by spurious bias from inside the model and directly intervene in the decision-making process, achieving **completely group-label-free** debiasing?

## Method

### Overall Process

NeuronTune is a **post hoc** method applied to a pre-trained ERM model $f_\theta = h_{\theta_2} \circ e_{\theta_1}$ (feature extractor + linear classification head), consisting of three steps:

**Step 1: Extract Embeddings and Prediction Results**

For the identification dataset $\mathcal{D}_{\text{Ide}}$ (the validation set by default), extract the latent embedding $\mathbf{v} = e_{\theta_1}(\mathbf{x}) \in \mathbb{R}^M$ and the prediction correctness indicator $o$ for each sample.

**Step 2: Identify Bias Dimensions**

For each class $y$ and each embedding dimension $i$, split the activation values into two groups based on whether the prediction is correct or incorrect, and compute the **spuriousness score**:

$$\delta_i^y = \text{Med}(\bar{\mathcal{V}}_i^y) - \text{Med}(\hat{\mathcal{V}}_i^y)$$

where $\bar{\mathcal{V}}_i^y$ is the set of activation values on the $i$-th dimension for **incorrectly predicted** samples, and $\hat{\mathcal{V}}_i^y$ is the set of activation values for **correctly predicted** samples.

- $\delta_i^y > 0$: High activation in this dimension leads to misclassification instead $\rightarrow$ affected by spurious bias.
- $\delta_i^y < 0$: High activation in this dimension helps correct classification $\rightarrow$ core features.

Identify the set of bias dimensions: $\mathcal{S} = \{i \mid \delta_i^y > \lambda,\; \forall y \in \mathcal{Y}\}$, with a default threshold of $\lambda = 0$.

**Step 3: Suppress Bias Dimensions + Retrain Last Layer**

Freeze the feature extractor $e_{\theta_1}$, set the activations of the bias dimensions in the embedding to **zero** to obtain $\tilde{\mathbf{v}}$, and retrain the classification head on $\mathcal{D}_{\text{Tune}}$ (the training set by default) using class-balanced sampling:

$$\theta_2^* = \arg\min_{\theta_2} \mathbb{E}_{\mathcal{B} \sim \mathcal{D}_{\text{Tune}}} \ell(h_{\theta_2}(\tilde{\mathbf{v}}), y)$$

### Theoretical Guarantees

- **Proposition 4.1**: When $\gamma^T \mathbf{w}_{\text{spu},i} < 0$, the model still relies on spurious features even when the spurious correlation fails $\rightarrow$ this neuron should be suppressed.
- **Theorem 4.2**: Proves that $\delta_i^y \approx -2\mu \gamma^T \mathbf{w}_{\text{spu},i}$, meaning a positive spuriousness score corresponds to the biased neurons that need to be suppressed.
- **Theorem 4.3**: Proves that the model parameters produced by NeuronTune are closer to the unbiased optimal solution than those of the original ERM model.

### Model Selection: Spuriousness Fitness Score (SFit)

Without group labels, SFit is used for model selection: $\text{SFit} = \sum_{m=1}^{M} \sum_{y \in \mathcal{Y}} |\delta_m^y|$. A higher SFit indicates that the bias/non-bias dimensions are more separable, making the model more suitable for debiasing.

## Key Experimental Results

### Image Datasets (Waterbirds / CelebA)

| Method | Group Labels | Waterbirds WGA↑ | CelebA WGA↑ |
|------|--------|----------------|-------------|
| ERM | - | 72.6 | 47.2 |
| JTT | Semi-supervised | 86.7 | 81.1 |
| DFR† | Semi-supervised | 92.4 | 87.0 |
| BAM (Unsupervised) | None | 89.1 | 80.1 |
| **NeuronTune** | **None** | **92.2** | **83.1** |
| **NeuronTune†** | **None** | **92.5** | **87.3** |

### Text Datasets (MultiNLI / CivilComments)

| Method | Group Labels | MultiNLI WGA↑ | CivilComments WGA↑ |
|------|--------|--------------|-------------------|
| ERM | - | 67.9 | 57.4 |
| AFR | Semi-supervised | 73.4 | 68.7 |
| DFR† | Semi-supervised | 70.8 | 81.8 |
| **NeuronTune** | **None** | **72.1** | **82.4** |
| **NeuronTune†** | **None** | **72.5** | **82.7** |

### ImageNet-9 → ImageNet-A Distribution Shift

| Method | ImageNet-9 Acc | ImageNet-A Acc | Acc Gap↓ |
|------|---------------|---------------|---------|
| ERM | 90.8 | 24.9 | 65.9 |
| LWBC | 94.0 | 36.0 | 58.0 |
| **NeuronTune** | 93.7 | **37.3** | **56.4** |

### Ablation: Complete vs. Partial Suppression (CelebA)

Complete zeroing (masking=0) achieves WGA **87.3%**; partial suppression (masking=0.2~1.0) yields only 71~73% WGA $\rightarrow$ complete suppression is necessary for effectiveness.

## Highlights & Insights

- **Completely Group-Label-Free**: Unlike DFR/AFR, which require group testing labels on the validation set for model selection, NeuronTune achieves self-guided selection via SFit.
- **Neuron-level Intervention**: Elevates control from the sample level to the neuron level, providing more precise debiasing control.
- **Solid Theoretical Foundation**: From data models to selection metrics and debiasing guarantees, step-by-step mathematical proofs provide solid theoretical support.
- **Lightweight Post-hoc**: Retrains only the final layer, incurring minimal computational cost and adapting to any pre-trained model.
- **Cross-modal Generality**: Effective across both vision (ResNet) and text (BERT) modalities.

## Limitations & Future Work

- **Slight Drop in Average Accuracy**: While debiasing boosts WGA↑, it incurs a 1~3% loss in overall accuracy (the average-worst accuracy tradeoff).
- **Sensitivity to Identification Data Selection**: Using the training set for $\mathcal{D}_{\text{Ide}}$ yields poor outcomes (due to model memorization); a separate validation set is required.
- **Linear Assumption**: Theoretical analysis is based on linear data models and linear regression, whereas feature entanglement in practical deep networks is much more complex.
- **Fixed Threshold**: $\lambda = 0$ is applied uniformly across all datasets; adaptive threshold strategies remain unexplored.
- **Manipulating Only the Last Layer**: The feature extractor remains frozen; if the features themselves are deeply entangled, the potential for improvement is limited.

## Rating

⭐⭐⭐⭐ — Rigorous theoretical derivation, straightforward and practical method, achieving debiasing performance close to semi-supervised methods without requiring extra annotation. It is a solid work balancing theory and practice. Limitations lie in the gap between the linear assumption and practical deep networks, as well as the loss in average accuracy.

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Spurious Correlation-Aware Embedding Regularization for Worst-Group Robustness](../../ICLR2026/others/spurious_correlation-aware_embedding_regularization_for_worst-group_robustness.md)
- [\[ACL 2025\] Causal Estimation of Tokenisation Bias](../../ACL2025/others/causal_tokenisation_bias.md)
- [\[ICLR 2026\] Regulating Internal Alignment Flows for Robust Learning Under Spurious Correlations](../../ICLR2026/others/regulating_internal_alignment_flows_for_robust_learning_under_spurious_correlati.md)
- [\[CVPR 2026\] Bias In, Bias Out? Finding Unbiased Subnetworks in Vanilla Models](../../CVPR2026/others/bias_in_bias_out_finding_unbiased_subnetworks_in_vanilla_models.md)
- [\[ICML 2025\] Truly Self-Improving Agents Require Intrinsic Metacognitive Learning](truly_self-improving_agents_require_intrinsic_metacognitive_learning.md)

</div>

<!-- RELATED:END -->

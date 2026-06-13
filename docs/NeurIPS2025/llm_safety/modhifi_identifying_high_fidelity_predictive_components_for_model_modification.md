---
title: >-
  [Paper Note] ModHiFi: Identifying High Fidelity Predictive Components for Model Modification
description: >-
  [NeurIPS 2025][LLM Safety][model modification] This paper proposes the Subset Fidelity metric and the ModHiFi framework. Through theoretical analysis…
tags:
  - "NeurIPS 2025"
  - "LLM Safety"
  - "model modification"
  - "structured pruning"
  - "machine unlearning"
  - "subset fidelity"
  - "Lipschitz continuity"
  - "synthetic data"
date: 2026-05-08
content_hash: e00b2ea930f6e2c7
---

# ModHiFi: Identifying High Fidelity Predictive Components for Model Modification

**Conference**: NeurIPS 2025 Spotlight  
**arXiv**: [2511.19566](https://arxiv.org/abs/2511.19566)  
**Code**: [DhruvaKashyap/modhifi](https://github.com/DhruvaKashyap/modhifi)  
**Authors**: Dhruva Kashyap, Chaitanya Murti, Pranav Nayak, Tanay Narshana, Chiranjib Bhattacharyya (IISc, HP AI Lab, Google)
**Area**: Model Compression
**Keywords**: model modification, structured pruning, machine unlearning, subset fidelity, Lipschitz continuity, synthetic data

## TL;DR

This paper proposes the Subset Fidelity metric and the ModHiFi framework. Through theoretical analysis, it proves that local reconstruction error linearly upper-bounds global prediction error for Lipschitz continuous networks. Without requiring training data, loss functions, or gradients—using only synthetic data—the framework identifies high-fidelity (HiFi) components within a model, and unifies the tasks of structured pruning and class unlearning under a single formulation.

## Background & Motivation

Open-weight models are increasingly prevalent, yet original training data and loss functions are typically unavailable. This poses a fundamental challenge for model modification tasks such as pruning, unlearning, and debiasing—most existing methods rely on gradients or ground-truth labels and are infeasible in settings without loss functions or training data.

**Limitations of Prior Work**:
- **Vision model pruning**: Most methods require original training data for fine-tuning; they are inapplicable when data is unavailable. $L_2$-norm pruning is simple but performs poorly.
- **LLM pruning** (SliceGPT, ShortGPT): Relies on calibration datasets; methods are often architecture-specific and do not generalize to other tasks such as unlearning.
- **Class unlearning** (Jia et al.): Requires gradients and fine-tuning, which is time-consuming and depends on original training data.
- **Absence of a unified framework**: Pruning and unlearning are typically studied independently, lacking a unified measure of component importance.

**Core Problem**: Can one effectively identify components critical to predictive performance using only distributional access (synthetic data), without gradients or loss functions?

## Method

### Core Theory: Local-to-Global Error Propagation

**Unified Abstraction**: CNNs and Transformers are modeled under a common formulation—the output of channel $c$ at each layer is decomposed as an additive sum of input contributions:
$$\boldsymbol{Y}_c^l(\boldsymbol{X}) = \sum_{i=1}^{c_{in}^l} \boldsymbol{A}_{ci}^l(\boldsymbol{X})$$

where $\boldsymbol{A}_{ci}^l$ corresponds to convolution operations in CNNs and linear projections of FFN intermediate neurons in Transformers.

**Local-to-Global Theorem**: For Lipschitz continuous networks, the global prediction error after modifying parameters at layer $l$ is linearly upper-bounded by the local reconstruction error:
$$\mathbb{E}[\|\mathrm{N}_\theta(\mathrm{X}) - \mathrm{N}_{\theta \odot M^l}(\mathrm{X})\|^2] \leq \mathcal{O}\left(\sum_c \mathbb{E}[\|\boldsymbol{Y}_c^l - \sum_{i \in C} m_{ci}^l \boldsymbol{A}_{ci}^l\|^2]\right)$$

**Key Insight**: Contrary to claims in existing literature that Transformers are not Lipschitz continuous, the authors prove that well-trained Transformers do satisfy Lipschitz continuity (Corollary B.4), making this theorem applicable to CNNs, ViTs, and LLMs alike.

### Subset Fidelity Metric

The fidelity of subset $C$ with respect to output channel $c$ is defined as:
$$\mathrm{FS}_c^l(C) = \max_{\delta_c^l} \left(1 - \frac{\mathbb{E}[\|\boldsymbol{Y}_c^l - \sum_{i \in C} \delta_{ci}^l \boldsymbol{A}_{ci}^l\|^2]}{\mathbb{E}[\|\boldsymbol{Y}_c^l\|^2]}\right)$$

**Properties**: Boundedness $0 \leq \mathrm{FS} \leq 1$; monotonicity (larger subsets yield higher fidelity).

**Closed-form solution for singleton fidelity**:
$$s_{ci}^l = \mathrm{FS}_c^l(\{i\}), \quad \alpha_{ci}^l = \frac{\mathbb{E}[\langle \boldsymbol{Y}_c^l, \boldsymbol{A}_{ci}^l \rangle]}{\mathbb{E}[\|\boldsymbol{A}_{ci}^l\|^2]}$$

**Optimality Condition**: When input contributions are pairwise uncorrelated, the NAIVE selection strategy—ranking by singleton fidelity—is exactly optimal, equivalent to solving the NP-hard $k$-MFS problem exactly.

### ModHiFi Algorithm

**ModHiFi-P (Structured Pruning)**:
1. Perform forward passes with synthetic data to compute singleton fidelity for each input channel at each layer.
2. Select the top-$k$ channels with the highest fidelity as the HiFi set.
3. Zero out the weights of channels not in the HiFi set.
4. Adjust retained channel weights using a closed-form compensation term $\delta^*$, without gradient-based fine-tuning.

**ModHiFi-U (Class Unlearning)**:
1. Compute component fidelity using samples from the forget class only.
2. Select components with the highest fidelity—those most critical to predicting the forget class.
3. Zero out the weights of these components to destroy the model's predictive capacity for that class.
4. Performance on retained classes is largely unaffected, as their critical components remain untouched.

**Key Advantage**: The two tasks are dual to each other—pruning retains HiFi components while unlearning removes them. The unified fidelity metric enables the same algorithmic framework to address two fundamentally distinct tasks.

## Key Experimental Results

### Table 1: Structured Pruning on ImageNet with ResNet-50

| Method | Accuracy | FLOP Reduction | Param Reduction | CPU Speedup | GPU Speedup |
|--------|----------|---------------|-----------------|-------------|-------------|
| Unpruned | 76.1 | 1x | 1x | 1x | 1x |
| GReg-2 | 73.9 | 3.02x | 2.31x | 1.36x | 1.53x |
| OTO | 74.7 | 2.86x | 2.81x | 1.25x | 1.45x |
| ThiNet | 71.6 | 3.46x | 2.95x | 1.38x | 1.50x |
| DFPC (54) | 73.80 | 3.46x | 2.65x | 2.37x | 2.38x |
| **ModHiFi-P** | **76.70** | 2.17x | 1.47x | **1.69x** | **1.70x** |
| **ModHiFi-P** (high compression) | **73.82** | **3.66x** | **3.05x** | **2.42x** | **2.38x** |

At comparable accuracy levels, ModHiFi-P achieves the best speedup ratio (approximately 11% higher than DFPC), without requiring original training data. Under the high-compression setting, 73.82% accuracy corresponds to 3.66x FLOP reduction and 2.42x CPU speedup, surpassing all baselines.

### Table 3: Structured Pruning on Llama-2-7B

| Sparsity | Method | WikiText PPL | ARC-e | ARC-c | PIQA | WinoG. | HellaS. | Avg |
|----------|--------|-------------|-------|-------|------|--------|---------|-----|
| 0% | Dense | 5.12 | 74.58 | 46.25 | 79.11 | 69.06 | 75.99 | 69.00 |
| 10% | SliceGPT | 6.46 | 56.14 | 35.33 | 69.53 | 64.80 | 59.02 | 59.96 |
| 10% | **ModHiFi-P-Alpaca** | 6.36 | **71.42** | **42.06** | **76.44** | **68.19** | **71.67** | **65.96** |
| 20% | ShortGPT | 14.32 | 58.33 | 38.05 | 72.58 | 65.51 | 65.27 | 59.95 |
| 20% | SliceGPT | 8.13 | 50.08 | 31.14 | 64.85 | 62.04 | 48.84 | 51.39 |
| 20% | **ModHiFi-P-Alpaca** | 9.38 | **64.73** | **38.22** | **72.79** | **64.64** | **62.70** | **60.62** |
| 30% | ShortGPT | 33.21 | 48.65 | 32.85 | 64.31 | 64.33 | 56.13 | 53.25 |
| 30% | SliceGPT | 10.96 | 44.19 | 27.47 | 58.71 | 57.46 | 41.27 | 45.82 |
| 30% | **ModHiFi-P-Alpaca** | 14.78 | **53.15** | **32.50** | **66.59** | **59.35** | **50.61** | **52.44** |

At 10% sparsity, the average accuracy of 65.96% substantially outperforms SliceGPT's 59.96% (+6.0%). ModHiFi-P achieves the best results at both 20% and 30% sparsity. Alpaca synthetic data yields better calibration than WikiText.

### Table 4: Class Unlearning on CIFAR-10

| Model | Method | Forget Acc | Remain Acc | Time (s) |
|-------|--------|------------|------------|----------|
| ResNet-50 | Base | 94.99 | 94.99 | - |
| ResNet-50 | Gradient Ascent | 6.59 | 93.44 | 30 |
| ResNet-50 | Jia et al. | 3.54 | 94.14 | 363 |
| ResNet-50 | **ModHiFi-U** | **0.20** | 92.98 | **10** |
| Swin-T | Jia et al. | 1.20 | 90.69 | 235 |
| Swin-T | ModHiFi-U | 8.83 | 73.57 | 2 |

On ResNet-50, ModHiFi-U achieves complete forgetting (0.2% forget accuracy), running 36× faster than Jia et al. (10s vs. 363s) without fine-tuning. Results on Swin-T without fine-tuning are less favorable; however, adding 3 epochs of synthetic fine-tuning yields substantial improvement.

### Empirical Validation of HiFi Component Existence

Empirical observations show that across all evaluated models, fewer than 20% of input channels per layer suffice to achieve fidelity $\geq 0.8$. A perturbation experiment further illustrates this: disturbing 20% of HiFi components causes approximately 12% accuracy drop, while disturbing 80% of non-HiFi components results in only a 1% drop.

## Highlights & Insights

- **Rigorous theoretical foundation**: This work is the first to prove that local reconstruction error linearly upper-bounds global prediction error for Lipschitz continuous networks, and corrects the prevailing claim that Transformers are not Lipschitz continuous by establishing Lipschitz continuity for well-trained Transformers.
- **Unified framework**: A single Subset Fidelity metric and the ModHiFi algorithm jointly address structured pruning and class unlearning as dual tasks, generalizing across CNN and Transformer architectures without architecture-specific design.
- **Gradient-free, loss-free, data-free**: Critical components are identified via forward passes on synthetic data alone; closed-form compensation terms eliminate the need for gradient-based fine-tuning, offering strong practical applicability in deployment scenarios.
- **Significant practical speedup**: Approximately 11% faster than state-of-the-art on ImageNet in terms of actual CPU/GPU acceleration; 36× faster than baselines for class unlearning on CIFAR-10.

## Limitations & Future Work

- **Optimality of singleton fidelity relies on uncorrelation**: Input contributions in real networks are typically correlated; the NAIVE selection strategy may be suboptimal in highly correlated settings.
- **Swin-T unlearning degrades without fine-tuning**: On more complex Transformer architectures, forgetting performance drops notably without fine-tuning (forget accuracy 8.83%), indicating sensitivity to architectural complexity.
- **FFN-only analysis**: The theoretical framework covers only the FFN modules of Transformers; component importance in Multi-Head Attention is not analyzed and is left for future work.
- **Synthetic data quality dependency**: While more robust than $L_2$ pruning with fine-tuning, low-quality synthetic data still degrades performance.
- **Only two tasks evaluated**: Although the framework theoretically extends to debiasing, continual learning, and related tasks, no experimental validation is provided for these settings.

## Related Work & Insights

- **Structured pruning (vision)**: GReg-2 (regularization constraints), ThiNet (greedy channel selection), DFPC (data-free pruning with synthetic fine-tuning), DepGraph (dependency graph analysis)—most require original training data or architecture-specific design.
- **LLM pruning**: SliceGPT (dimensionality reduction via orthogonal projection), ShortGPT (redundant layer removal), Wanda (weight-activation product scoring)—require calibration data and are not applicable to unlearning.
- **Class unlearning**: Gradient Ascent (straightforward but coarse), Jia et al. (Fisher information-based selective unlearning)—require gradients and fine-tuning, are time-consuming, and depend on original data.
- **ModHiFi's positioning**: The first gradient-free, loss-free framework to unify pruning and unlearning, theoretically grounded rather than heuristic-driven, and operational with synthetic data alone.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The Subset Fidelity concept is original; the Local-to-Global theorem rigorously connects local metrics to global performance. Establishing Lipschitz continuity for well-trained Transformers carries independent theoretical value.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Coverage spans CNN/ViT/LLM, both pruning and unlearning tasks, and multiple datasets; however, LLM unlearning experiments are absent and Swin-T results are unsatisfactory.
- **Writing Quality**: ⭐⭐⭐⭐ — Theoretical derivations are rigorous and the unified notation system is clear; however, the density of mathematical notation raises the reading barrier.
- **Value**: ⭐⭐⭐⭐⭐ — Addresses the practically important problem of model modification without training data or loss functions; the unified framework offers significant inspiration for subsequent work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] InvisibleInk: High-Utility and Low-Cost Text Generation with Differential Privacy](invisibleink_high-utility_and_low-cost_text_generation_with_differential_privacy.md)
- [\[NeurIPS 2025\] PULSE: Practical Evaluation Scenarios for Large Multimodal Model Unlearning](pulse_practical_evaluation_scenarios_for_large_multimodal_model_unlearning.md)
- [\[AAAI 2026\] The Confidence Trap: Gender Bias and Predictive Certainty in LLMs](../../AAAI2026/llm_safety/the_confidence_trap_gender_bias_and_predictive_certainty_in_llms.md)
- [\[NeurIPS 2025\] FALCON: Fine-grained Activation Manipulation by Contrastive Orthogonal Unalignment for Large Language Model](falcon_fine-grained_activation_manipulation_by_contrastive_orthogonal_unalignmen.md)
- [\[ICLR 2026\] Gaussian Certified Unlearning in High Dimensions: A Hypothesis Testing Approach](../../ICLR2026/llm_safety/gaussian_certified_unlearning.md)

</div>

<!-- RELATED:END -->

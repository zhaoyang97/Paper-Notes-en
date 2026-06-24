---
title: >-
  [Paper Note] A High-Dimensional Statistical Method for Optimizing Transfer Quantities in Multi-Source Transfer Learning
description: >-
  [NeurIPS 2025][Transfer Learning][optimal transfer quantity] This paper proposes a theoretical framework based on K-L divergence and high-dimensional statistical analysis to determine the optimal number of samples to transfer from each source task in multi-source transfer learning. The framework avoids the negative transfer caused by naively using all source data, and the resulting algorithm OTQMS surpasses the state of the art by 1.0–1.5% on DomainNet and Office-Home while r…
tags:
  - "NeurIPS 2025"
  - "Transfer Learning"
  - "Multi-Source Transfer Learning"
  - "High-Dimensional Statistics"
  - "optimal transfer quantity"
  - "K-L divergence"
  - "Fisher information matrix"
  - "data efficiency"
date: 2026-05-08
content_hash: 3e1c2fcd69d04259
---

# A High-Dimensional Statistical Method for Optimizing Transfer Quantities in Multi-Source Transfer Learning

**Conference**: NeurIPS 2025
**arXiv**: [2502.04242](https://arxiv.org/abs/2502.04242)  
**Code**: [https://github.com/zqy0126/OTQMS](https://github.com/zqy0126/OTQMS)  
**Area**: Transfer Learning / Multi-Source Transfer Learning / High-Dimensional Statistics
**Keywords**: Multi-source transfer learning, optimal transfer quantity, K-L divergence, Fisher information matrix, data efficiency

## TL;DR
This paper proposes a theoretical framework based on K-L divergence and high-dimensional statistical analysis to determine the optimal number of samples to transfer from each source task in multi-source transfer learning. The framework avoids the negative transfer caused by naively using all source data, and the resulting algorithm OTQMS surpasses the state of the art by 1.0–1.5% on DomainNet and Office-Home while reducing sample usage by 47.85% and training time by 35.19%.

## Background & Motivation
Multi-source transfer learning leverages knowledge from multiple source tasks to improve performance on a target task. Existing methods typically use all available source samples for joint training, which introduces two fundamental problems: (1) training is highly inefficient when source data is large-scale; and (2) more source samples do not necessarily yield better performance—distributional discrepancies between source and target tasks introduce bias that may outweigh the variance reduction brought by additional data, leading to negative transfer. The authors empirically demonstrate this phenomenon: on a 5-task partition of CIFAR-10, using all source samples sometimes performs worse than using only target-task samples.

## Core Problem
**Given $K$ source tasks, how many samples should be transferred from each source for joint training so as to minimize the generalization error on the target task?** This is not a coarse task-level selection problem ("which sources to use") but a finer-grained quantity-level optimization problem ("how much data from each source"). The problem is significant because it simultaneously affects model accuracy and training efficiency—identifying the optimal transfer quantities improves performance while substantially reducing unnecessary data consumption.

## Method

### Overall Architecture
The multi-source transfer learning problem is formalized as a parameter estimation problem. Given a target task $\mathcal{T}$ with $N_0$ training samples drawn from distribution $P_{X;\bar{\theta}_0}$, and $K$ source tasks $\mathcal{S}_1, \ldots, \mathcal{S}_K$ with $N_1, \ldots, N_K$ samples respectively, the goal is to find optimal transfer quantities $n_1^*, \ldots, n_K^*$ that minimize the expected K-L divergence between the true and learned distributions. The overall pipeline is: (1) derive an asymptotic expression for generalization error via high-dimensional statistical theory; (2) minimize this expression to obtain analytical or numerical solutions for the optimal transfer quantities; (3) develop the practical algorithm OTQMS based on these solutions.

### Key Designs

1. **K-L Divergence as Generalization Error Measure**: Unlike existing measures such as $f$-divergence, mutual information, or $\mathcal{H}$-score, this paper adopts the expected K-L divergence $\mathbb{E}[D(P_{X;\bar{\theta}_0} \| P_{X;\hat{\bar{\theta}}})]$ as the generalization error metric. This choice is motivated by the close correspondence between K-L divergence and cross-entropy loss, and by the fact that under asymptotic analysis the metric decomposes directly into variance and bias terms. Using tools such as Taylor expansion and Sanov's theorem, the paper proves that this measure is asymptotically proportional to the mean squared error of the MLE.

2. **Theoretical Derivation from Single-Source to Multi-Source**: The theory is developed progressively from simple to complex settings:

    - **Lemma 3**: Without any source task (target data only), the K-L error equals $\frac{1}{2N_0}$, inversely proportional to sample size.
    - **Theorem 4**: In the single-source setting, the K-L error decomposes as $\frac{1}{2}\left(\frac{1}{N_0+n_1} + \frac{n_1^2}{(N_0+n_1)^2} t\right)$, where the first term is the variance term (decreasing in $n_1$) and the second is the bias term (increasing in $n_1$), and $t = J(\theta_0)(\theta_1 - \theta_0)^2$ measures source–target discrepancy. The optimal transfer quantity depends on the magnitude of $N_0 \cdot t$: if $\leq 0.5$ (highly similar source and target), all source data should be used; if $> 0.5$, a finite optimal point exists at $n_1^* = \frac{N_0}{2N_0 t - 1}$.
    - **Propositions 5–6**: Extension to high-dimensional parameter spaces ($\theta \in \mathbb{R}^d$), with analogous forms augmented by a dimensionality factor $d$.
    - **Theorem 7**: Extension to $K$ source tasks, where the K-L error equals $\frac{d}{2}\left(\frac{1}{N_0+s} + \frac{s^2}{(N_0+s)^2} t\right)$, with $s = \sum n_i$ denoting the total transfer quantity and $t$ encoding a weighted combination of all source–target parameter discrepancies. The optimal solution is obtained numerically via grid search over $s$ combined with quadratic programming over the allocation vector $\bar{\alpha}$.

3. **OTQMS Practical Algorithm**: The core innovation is a **Dynamic Strategy**—since target-task samples are scarce and direct estimation of $\theta_0$ from limited data is unreliable, an iterative update scheme is adopted: at each epoch, the current $\theta_0$ estimate is used to compute the optimal transfer quantities; samples are then drawn randomly from each source according to these quantities to form a new training set; training continues to update $\theta_0$; and transfer quantities are recomputed at the next epoch. The Fisher information matrix $J$ is approximated using the empirical Fisher (outer product of gradients of the training loss). The algorithm is architecture-agnostic and compatible with ViT, LoRA, and other architectures.

### Loss & Training
- Cross-entropy loss (corresponding to negative log-likelihood)
- Adam optimizer with learning rate $1\text{e-}5$
- Early stopping: highest accuracy within 5 epochs
- Fisher information matrix approximated via gradients of the training loss

## Key Experimental Results

| Dataset | Metric | OTQMS | AllSources∪Target | Gain |
|--------|------|-------|-------------------|------|
| DomainNet (Avg) | Accuracy | 55.8% | 54.3% | +1.5% |
| Office-Home (Avg) | Accuracy | 78.2% | 77.2% | +1.0% |
| DomainNet | Training Time | — | — | −35.19% |
| DomainNet | Sample Usage | — | — | −47.85% |

Comparison with other baselines (10-shot, DomainNet Avg):

| Method | Avg Accuracy |
|------|-------------|
| Target-Only | 16.7% |
| H-ensemble | 43.8% |
| MCW | 43.9% |
| WADN | 50.5% |
| MADA (ViT-S) | 40.5% |
| AllSources∪Target | 54.3% |
| **OTQMS** | **55.8%** |

### Ablation Study
- **Dynamic vs. Static Strategy**: The dynamic strategy (78.2%) substantially outperforms all static variants (Static-Under 77.2%, Static-Exact 68.7%, Static-Over 71.5%), validating the necessity of per-epoch transfer quantity updates.
- **Generalization across Shot Settings**: From 5-shot to 100-shot, OTQMS consistently outperforms AllSources∪Target and Target-Only, demonstrating stable advantages across shot configurations.
- **LoRA Compatibility**: Effective on ViT-B + LoRA (OTQMS 82.3% vs. AllSources∪Target 81.1%).
- **Multi-Task Learning**: Also applicable to multi-task learning scenarios (OTQMS 83.5% vs. Single-task 71.4% on Office-Home).
- **Domain Preference Analysis**: Visualizations show that OTQMS preferentially selects source domains similar to the target domain (e.g., when the target is Clipart, it favors Real, Painting, and Sketch), consistent with intuition.

## Highlights & Insights
- **Tight Theory–Practice Integration**: The derivation proceeds from a complete 1D single-source analysis to high-dimensional multi-source extensions and finally to a practical algorithm, with clear hierarchy and rigorous logic.
- **Elegant Bias–Variance Decomposition**: The generalization error is explicitly decomposed into a variance term (reduced by more data) and a bias term (increased by irrelevant data), revealing the fundamental reason why "more data is not always better."
- **Data Efficiency**: Beyond accuracy gains, the method substantially reduces training time and data usage (approximately 50% fewer training samples), offering practical value in large-scale settings.
- **Architecture-Agnostic**: The same framework applies to full fine-tuning of ViT and parameter-efficient LoRA training without architecture-specific modifications.
- **Interpretable Domain Preference**: Analysis of optimal transfer quantities reveals which source domains are most beneficial for the target domain, providing interpretable decision support for transfer learning.

## Limitations & Future Work
- **Simple Sampling Strategy**: The current approach relies on random sampling; the authors acknowledge that more sophisticated strategies (e.g., active sampling) may further improve performance.
- **Quantity Without Sample Weighting**: The framework determines how many samples to use but does not assign weights to individual samples; joint optimization of sample weights and transfer quantities is a natural future direction.
- **Theoretical Dependence on Negative Log-Likelihood**: The theoretical analysis is grounded in the cross-entropy loss assumption and does not directly extend to other loss functions such as MSE.
- **Regularity Conditions**: The asymptotic analysis requires that source and target parameters be sufficiently close, specifically $|\theta_0 - \theta_i| = O(1/\sqrt{N_0})$, which may not hold in practice.
- **No Error Bars Reported**: Standard deviations are not reported due to resource constraints, which limits the credibility of the experimental results.

## Related Work & Insights
- **vs. H-ensemble / MCW (model/parameter weighting methods)**: OTQMS adopts a sample-based strategy rather than model weighting. Experiments show that sample-based methods generally outperform model-weighting approaches, as the former can more fully exploit task-relevant information in source data.
- **vs. WADN / MADA (sample-based methods)**: While also sample-based, OTQMS determines optimal quantities via a theoretical framework rather than using all samples or weighting by Wasserstein distance. OTQMS outperforms these methods in both accuracy and efficiency.
- **vs. Tong et al. [NeurIPS 2021] (theoretical framework)**: Tong et al. measure transferability using $\chi^2$-divergence and require unsupervised target data. The present work uses K-L divergence (better aligned with cross-entropy), and achieves both task-level and shot-level generalization.

## Related Work & Insights (Broader)
- **Connection to Knowledge Distillation**: The notion of "optimal transfer quantity" is transferable to multi-teacher distillation—knowledge from each teacher is not always beneficial in larger amounts, and an analogous bias–variance analysis could determine the optimal distillation temperature or feature mixing ratio for each teacher.
- **Relevance to Model Compression**: In the recovery fine-tuning stage following pruning or quantization, the framework proposed here could inform how much data to draw from multiple pretrained sources for fine-tuning.
- **Data Mixture Ratio Optimization**: The core idea—optimizing multi-source data mixture via bias–variance decomposition—has broad applicability to pretraining data ratio design and multi-domain fine-tuning.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The use of high-dimensional statistical methods to theoretically derive optimal transfer quantities is a novel framework, though the underlying idea (bias–variance tradeoff) is classical.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-dataset, multi-architecture, and multi-ablation evaluations are provided, but error bars are absent and the datasets are relatively small-scale.
- **Writing Quality**: ⭐⭐⭐⭐ The progressive theoretical derivation from simple to complex is logically clear, though the dense notation requires careful cross-referencing.
- **Value**: ⭐⭐⭐⭐ Dual contributions in theory and practice; the data efficiency improvements are practically significant, though the applicable scenario is somewhat specific (few-shot multi-source transfer).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Transfer Learning for Benign Overfitting in High-Dimensional Linear Regression](transfer_learning_for_benign_overfitting_in_high-dimensional_linear_regression.md)
- [\[ICLR 2026\] Transfer Learning in Infinite Width Feature Learning Networks](../../ICLR2026/learning_theory/transfer_learning_in_infinite_width_feature_learning_networks.md)
- [\[ICLR 2026\] Residual Feature Integration is Sufficient to Prevent Negative Transfer](../../ICLR2026/learning_theory/residual_feature_integration_is_sufficient_to_prevent_negative_transfer.md)
- [\[NeurIPS 2025\] Diffusion Transformers for Imputation: Statistical Efficiency and Uncertainty Quantification](diffusion_transformers_for_imputation_statistical_efficiency_and_uncertainty_qua.md)
- [\[ICLR 2026\] Understanding the Mechanisms of Fast Hyperparameter Transfer](../../ICLR2026/learning_theory/understanding_the_mechanisms_of_fast_hyperparameter_transfer.md)

</div>

<!-- RELATED:END -->

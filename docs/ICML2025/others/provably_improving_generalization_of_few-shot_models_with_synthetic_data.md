---
title: >-
  [Paper Note] Provably Improving Generalization of Few-Shot Models with Synthetic Data
description: >-
  [ICML2025][few-shot learning] This paper proposes a theoretical framework to quantify the impact of the distribution gap between synthetic and real data on the generalization capability of few-shot classification. Based on this theory, it designs an algorithm that jointly optimizes data partitioning and model training, surpassing SOTA on 10 benchmark datasets.
tags:
  - "ICML2025"
  - "few-shot learning"
  - "synthetic data"
  - "Generalization Bound"
  - "Prototype Learning"
  - "Distribution Matching"
  - "CLIP"
date: 2026-05-08
content_hash: 2d533e5a3603493d
---

# Provably Improving Generalization of Few-Shot Models with Synthetic Data

**Conference**: ICML2025  
**arXiv**: [2505.24190](https://arxiv.org/abs/2505.24190)  
**Code**: To be confirmed  
**Area**: Few-Shot Learning  
**Keywords**: few-shot learning, synthetic data, Generalization Bound, Prototype Learning, Distribution Matching, CLIP

## TL;DR

This paper proposes a theoretical framework to quantify the impact of the distribution gap between synthetic and real data on the generalization capability of few-shot classification. Based on this theory, it designs an algorithm that jointly optimizes data partitioning and model training, surpassing SOTA on 10 benchmark datasets.

## Background & Motivation

Few-shot image classification challenges stem from the extreme scarcity of labeled samples. Utilizing generative models to synthesize data to augment training sets is a promising direction; however, there is a **distribution gap** between synthetic and real data, and directly using synthetic data could instead lead to degraded performance.

Existing methods (e.g., DataDream, RealFake, DISEF) primarily fine-tune generators to reduce the distance between synthetic and real distributions. However, most of these methods are based on heuristic designs and **lack theoretical guarantees**. This study starts from four core questions:

1. What metrics can measure the quality of synthetic datasets?
2. How to generate high-quality synthetic data?
3. How to efficiently train classifiers using real and synthetic data?
4. How does generator quality affect the generalization performance of the trained model?

## Method

### Theoretical Framework

**Core Definition 1 — Model-based Discrepancy**: Measures the distance between the synthetic dataset $\boldsymbol{G}$ and the real dataset $\boldsymbol{S}$ in the output space of model $h$:

$$\bar{d}_h(\boldsymbol{G}, \boldsymbol{S}) = \frac{1}{|\boldsymbol{G}||\boldsymbol{S}|} \sum_{\boldsymbol{u}\in\boldsymbol{G}, \boldsymbol{s}\in\boldsymbol{S}} \|h(\boldsymbol{s}) - h(\boldsymbol{u})\|$$

**Core Definition 2 — Local Robustness**: Measures the prediction stability of model $h$ on data point $\boldsymbol{s}$ within a local region $\mathcal{A}$:

$$\mathcal{R}_h(\boldsymbol{s}, \mathcal{A}|P) = \mathbb{E}_{\boldsymbol{z}\sim P}[\|h(\boldsymbol{z}) - h(\boldsymbol{s})\| : \boldsymbol{z} \in \mathcal{A}]$$

**Main Theorem (Theorem 3.3)**: For a model $h$ trained on real data $\boldsymbol{S}$ ($n$ samples from $P_0$) and synthetic data $\boldsymbol{G}$ (from $P_g$), its test error is upper-bounded by:

$$F(P_0, h) \leq L_h \sum_{i \in T_S} \frac{g_i}{g}\left[\bar{d}_h(\boldsymbol{G}_i, \boldsymbol{S}_i) + \mathcal{R}_h(\boldsymbol{G}_i, \mathcal{Z}_i | P_g)\right] + A$$

where $A$ contains the empirical loss on the synthetic distribution, real/synthetic data ratio mismatch terms, local robustness terms of the real data, and complexity terms of order $O(1/\sqrt{n} + 1/\sqrt{g})$.

**Theoretical Insights**:

- Synthetic samples must not only be **close to** real samples (low discrepancy), but also maintain **diversity** to guarantee the local robustness of the model.
- A larger volume of synthetic data $g$ leads to smaller complexity terms, yielding better generalization.
- It only requires that the model "perceives" the closeness of the two distributions, without requiring actual closeness in any objective metric space.

### Algorithm Design

Based on minimizing the theoretical bound, a two-stage optimization algorithm is proposed (Algorithm 1):

**Stage 1 — Partition Optimization**: Perform K-means clustering on real and synthetic data in the CLIP feature space to obtain data partitioning $\Gamma(\mathcal{Z})$. The number of clusters is typically set to twice the number of classes.

**Stage 2 — Model Optimization**: Fine-tune the image encoder of CLIP ViT-B/16 (via LoRA) using the following loss function:

$$\mathcal{L} = \lambda F(\boldsymbol{S}, h) + F(\boldsymbol{G}, h) + \lambda_1 \cdot \text{Discrepancy} + \lambda_2 \cdot \text{Robustness}$$

- First term: Cross-entropy loss on real data (weighted by $\lambda$).
- Second term: Cross-entropy loss on synthetic data.
- Third term: Real-synthetic feature discrepancy regularization within the same cluster ($\lambda_1$).
- Fourth term: Synthetic-synthetic feature consistency regularization within the same cluster ($\lambda_2$), with $\lambda_1:\lambda_2 = 10:1$.

**Lightweight Version**: Does not fine-tune the generator; synthesizes only 64 images per class (compared to 500 in the full version), utilizing negative prompts to improve quality.

## Key Experimental Results

### Main Results: CLIP ViT-B/16, 16-shot, 500 synthetic images/class

| Method | IN | CAL | DTD | EuSAT | AirC | Pets | Cars | SUN | Food | FLO | Avg |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CLIP zero-shot | 70.2 | 96.1 | 46.1 | 38.1 | 23.8 | 91.0 | 63.1 | 72.2 | 85.1 | 71.8 | 64.1 |
| Real-finetune | 73.4 | 96.8 | 73.9 | 93.5 | 59.3 | 94.0 | 87.5 | 77.1 | 87.6 | 98.7 | 84.2 |
| DataDream_dset | 74.1 | 96.9 | 74.1 | 93.4 | 72.3 | 94.8 | 92.4 | 77.5 | 87.6 | 99.4 | 86.3 |
| **Ours (lightweight)** | 73.7 | **97.9** | **75.5** | 94.2 | 71.5 | 94.5 | 90.2 | 77.6 | **90.0** | 99.0 | 86.4 |
| **Ours (full)** | 73.8 | 97.3 | 74.5 | **94.7** | **74.3** | 94.6 | **93.1** | **77.7** | **90.4** | 99.3 | **87.0** |

- The full version ranks first on 7 out of 10 datasets, achieving an average accuracy of **87.0%**, outperforming DataDream_dset by 0.7%.
- Food101 shows a gain of **+2.8%**, and FGVC Aircraft shows a gain of **+2.0%**.
- The lightweight version matches DataDream using only 1/8 of the synthetic data.

### Ablation Study

| Discrepancy | Robustness | EuroSAT | DTD | AirC | Cars |
|---|---|---|---|---|---|
| ✗ | ✗ | 93.5 | 74.1 | 72.5 | 92.6 |
| ✓ | ✗ | 94.6 | 74.4 | 73.1 | 93.1 |
| ✗ | ✓ | 94.3 | 74.3 | 74.8 | 93.0 |
| ✓ | ✓ | **94.7** | **74.5** | **74.3** | **93.1** |

### Different Architectures: CLIP-ResNet50

| Method | AirC | Cars | Food | CAL |
|---|---|---|---|---|
| DataDream_dset | 81.46 | 93.30 | 66.63 | 94.62 |
| **Ours** | **82.67** | **93.71** | **70.35** | 94.17 |

Outperforms all baselines in 3 out of 4 datasets, validating the robustness of the proposed method across different backbones.

## Highlights & Insights

1. **Unification of Theory and Practice**: Provide the first formal generalization error upper bound for few-shot learning with synthetic data augmentation, directly deriving an actionable loss function from the theory.
2. **Local Robustness Regularization**: Discover that the previously overlooked robustness term is crucial for generalization, contributing +2.3% individually on FGVC Aircraft.
3. **Highly Practical Lightweight Version**: Reaches state-of-the-art levels without fine-tuning the generator, using only 64 synthetic images per class, which significantly reduces computational costs.
4. **Novel "Model-centric" Perspective**: Corollary 3.5 reveals that even if the real and synthetic distributions objectively exhibit a codebase gap, generalization can still be guaranteed as long as the model perceives them to be similar, which breaks traditional assumptions.

## Limitations & Future Work

1. **Tightness of the Bound**: The upper bound scales as $O(\sqrt{K})$, which is loose when the number of classes/clusters $K$ is large; furthermore, local behavior cannot be captured in extreme 1-shot scenarios.
2. **Decoupled Clustering and Training**: K-means clustering is executed as a one-time process on the data space and remains fixed, rather than dynamically updating with model training, which might not be optimal.
3. **Hyperparameter Sensitivity**: The hyperparameters $\lambda_1, \lambda_2$ require tuning across different datasets; although their ratio is fixed at 10:1, their absolute values still need grid searching.
4. **Generator Optimization Not Directly Integrated**: Although the theoretical framework implies that the generalization bound could guide generator fine-tuning or data filtering, it has not been implemented in this study.
5. **Only Evaluated on Classification**: Although the theoretical framework claims to generalize to other tasks like regression, empirical validation is lacking.

## Related Work & Insights

- **DataDream (Kim et al., 2024)**: Fine-tunes Stable Diffusion independently for each class to generate synthetic data, serving as the strongest baseline.
- **RealFake (Yuan et al., 2024)**: Aligns real and synthetic distributions by minimizing MMD.
- **IsSynth / DISEF**: Enhances synthetic data quality through noise injection and CLIP filtering.
- **Insights**: The theoretical framework can be extended to scenarios like adversarial training and domain adaptation; the concept of local robustness regularization can also be migrated to other data augmentation methods.

## Rating

- Novelty: ⭐⭐⭐⭐ — Provides the first theoretical generalization guarantee for few-shot learning with synthetic data, with a highly novel perspective on local robustness.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Evaluated on 10 datasets with 2 architectures and comprehensive ablation studies, though it lacks comparisons across more shot counts and other generators.
- Writing Quality: ⭐⭐⭐⭐ — Rigorous and clear theoretical derivations, with a tight link between experiments and theory.
- Value: ⭐⭐⭐⭐ — A stellar example of theory guiding practice; the lightweight version is highly practical, carrying strong scalability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Improving Generalization with Flat Hilbert Bayesian Inference](improving_generalization_with_flat_hilbert_bayesian_inference.md)
- [\[ICML 2025\] Feedforward Few-shot Species Range Estimation](feedforward_few-shot_species_range_estimation.md)
- [\[CVPR 2026\] Data-Centric Meta-Learning for Robust Few-Shot Generalization](../../CVPR2026/others/data-centric_meta-learning_for_robust_few-shot_generalization.md)
- [\[ACL 2025\] Generating Synthetic Relational Tabular Data via Structural Causal Models](../../ACL2025/others/generating_synthetic_relational_tabular_data_via_structural_causal_models.md)
- [\[ICCV 2025\] Doodle Your Keypoints: Sketch-Based Few-Shot Keypoint Detection](../../ICCV2025/others/doodle_your_keypoints_sketch-based_few-shot_keypoint_detection.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Cut Less, Fold More: Model Compression through the Lens of Projection Geometry
description: >-
  [ICLR 2026][Model Compression][Paper Note] Structured pruning and model folding are unified into an orthogonal projection framework—where pruning is axis-aligned projection and folding is cluster-subspace projection. It is proven that under the condition of a rank difference of 1, folding's parameter reconstruction error is strictly smaller. Validation across 1
tags:
  - ICLR 2026
  - Model Compression
date: 2026-05-08
content_hash: 0a7af82e82ebac80
---
# Cut Less, Fold More: Model Compression through the Lens of Projection Geometry

**Conference**: ICLR 2026  
**arXiv**: [2602.18116](https://arxiv.org/abs/2602.18116)  
**Code**: Available (Appendix link)  
**Area**: Model Compression  
**Keywords**: Model Folding, Structured Pruning, Orthogonal Projection, Calibration-free Compression, Projection Geometry

## TL;DR

Structured pruning and model folding are unified into an orthogonal projection framework—where pruning is axis-aligned projection and folding is cluster-subspace projection. It is proven that under the condition of a rank difference of 1, folding's parameter reconstruction error is strictly smaller. Validation across 1000+ checkpoints shows folding typically outperforms pruning at medium-to-high compression rates.

## Background & Motivation

**Background**: Calibration-free post-training structured compression is a critical requirement for model deployment. The mainstream approach is magnitude-based structured pruning, which removes neurons/channels/filters according to weight magnitudes. Recently proposed model folding achieves compression by clustering similar weights and binning/binding them.

**Limitations of Prior Work**: (1) Pruning directly sets weights to zero, causing significant parameter perturbation and functional shift; (2) Folding lacks theoretical support as an alternative, and conditions under which it outperforms pruning are unclear; (3) A unified comparison framework for both is missing.

**Key Challenge**: Weight removal in pruning leads to loss of directional information due to axis-aligned projection in the parameter space, while folding preserves merged directions but lacks theoretical guarantees regarding the extent to which this preservation is beneficial.

**Goal**: Establish a unified projection theoretical framework for pruning and folding, and rigorously prove the superiority of folding in parameter reconstruction and functional maintenance.

**Key Insight**: Treat both compression methods as orthogonal projections in the parameter space, where pruning corresponds to an axis-aligned subspace and folding corresponds to a cluster-structured subspace.

**Core Idea**: Pruning is a coordinate projection $\mathbf{C}_p = \begin{pmatrix} I & 0 \\ 0 & 0 \end{pmatrix}$, while folding is a clustering projection $\mathbf{C}_f = \mathbf{U}_f(\mathbf{U}_f^\top \mathbf{U}_f)^{-1}\mathbf{U}_f^\top$. The latter preserves more parameter directional information, resulting in a reconstruction error $\|\mathbf{W} - \mathbf{W}_f\|_F^2 \leq \|\mathbf{W} - \mathbf{W}_p\|_F^2$.

## Method

### Overall Architecture

This paper addresses a practical deployment question: when compressing a pre-trained network without accessing training data or performing calibration, should one choose traditional magnitude pruning or the emerging model folding? Previously, both lacked a unified comparison framework and theoretical criteria. This work views both structured pruning and model folding as an orthogonal projection of the weight matrix $\mathbf{W} \in \mathbb{R}^{m \times p}$ given by $\mathbf{W}_{\text{comp}} = \mathbf{C}\mathbf{W}$, where the difference lies solely in the target subspace—pruning projects onto an axis-aligned subspace (deleting neurons/channels), while folding projects onto a cluster-structured subspace (merging similar parameter vectors into cluster means). Within this unified coordinate system, projection geometry is used to prove that folding's reconstruction error consistently does not exceed that of pruning. This theoretical advantage is then validated across 1000+ checkpoints to identify the training/compression conditions under which it holds. The entire pipeline is calibration-free and data-free. After projection, REPAIR (BatchNorm statistic re-estimation) is applied to CNNs and LayerNorm reset to ViTs to correct distribution shifts, followed by optional fine-tuning for 1–5 epochs.

### Key Designs

**1. Unified Orthogonal Projection Framework: Placing "Coordinate Deletion" and "Direction Merging" in a Single Comparison Frame**

To fairly compare pruning and folding, they must share a common language. Both are formulated as $\mathbf{C}y = \arg\min_{z \in \text{Range}(\mathbf{U})} \|y - z\|_2$, representing a mapping to the nearest point on the subspace spanned by the column space $\mathbf{U}$. Consequently, "compression distortion" is equivalent to "projection distance." Pruning retains the top $k$ neurons and zeros the rest, corresponding to a subspace $\mathbf{U}_p = \begin{pmatrix} I \\ 0 \end{pmatrix}$ expanded only along coordinate axes; directional information from discarded axes is lost. Folding clusters $m$ parameter vectors into $k$ groups using $k$-means and replaces each with the cluster mean, corresponding to a one-hot cluster assignment matrix $\mathbf{U}_f \in \{0,1\}^{m \times k}$, preserving the "merged average direction" rather than simple deletion. In this unified framework, the two methods differ only in subspace selection, allowing direct comparison of reconstruction errors.

**2. Folding Reconstruction Error Strictly No Greater Than Pruning: Theoretical Guarantee from Projection Geometry**

With the unified framework, two theorems prove folding is at least as good as pruning in parameter reconstruction. Theorem 2.1 is constructive: for any pruning scheme, by merging all pruned rows into an additional cluster (setting the folding rank $k_f = k_p + 1$), the resulting folding $\mathbf{W}_f'$ yields a Frobenius reconstruction error strictly no larger than pruning. Theorem 2.2 further shows that optimal $k$-means folding is even better, yielding the chain inequality $\|\mathbf{W} - \mathbf{W}_f^\star\|_F^2 \leq \|\mathbf{W} - \mathbf{W}_f'\|_F^2 \leq \|\mathbf{W} - \mathbf{W}_p\|_F^2$. The link from "smaller parameter error" to "better functional maintenance" relies on the Lipschitz continuity of the loss function $|L(\mathbf{W}_1) - L(\mathbf{W}_2)| \leq \kappa \|\mathbf{W}_1 - \mathbf{W}_2\|_F$—since folding introduces smaller parameter perturbation, its functional shift upper bound is also smaller, translating "projection geometry advantage" into "post-compression accuracy advantage."

**3. Large-scale Validation on 1000+ Checkpoints: Defining When Theoretical Advantages Hold**

Since theorems only guarantee reconstruction error, actual gains depend on upstream training. Thus, the authors systematically scanned 1000+ checkpoints. Unlike prior pruning research that often uses fixed hyperparameters with varied seeds, this work covers Adam/SGD optimizers, various learning rates, data augmentation, regularization, SAM training, and LLaMA-60M/130M to investigate how training conditions influence compression. Results validate the theoretical boundaries: the advantage of folding consistently exists at medium-to-high compression rates and scales with the rate. Furthermore, training conditions that promote flat solutions (moderate learning rates, SAM) amplify folding's benefits, whereas sharp solutions diminish the gains of cluster projection.

### Loss & Training

The entire compression process is calibration-free with no training loss; folding groups are determined directly by $k$-means clustering. Optional post-processing includes: REPAIR for BatchNorm statistic re-estimation in CNNs, LayerNorm reset for ViTs, or lightweight fine-tuning for 1–5 epochs to improve convergence.

## Key Experimental Results

### Main Results

| Architecture/Dataset | Compression Rate | FOLD Acc | MAG1 Acc | Folding Gain |
| :--- | :--- | :--- | :--- | :--- |
| ResNet18/CIFAR-10 (Adam) | 50% | Significantly Leads | — | Max at Mid-High Comp |
| ViT-B/32/CIFAR-10 | 50% | Significantly Leads | — | Consistent positive gain |
| CLIP ViT-B/32/ImageNet-1K | 50% | Leads | — | Maintained after LN Reset |
| LLaMA-60M/C4 (PPL↓) | 20% | 47.17 | 54.51 | FOLD Lower PPL |
| LLaMA-60M/C4 (PPL↓) | 50% | 221.32 | 398.62 | FOLD Leads Significantly |

### Ablation Study

| Configuration | Metric | Description |
| :--- | :--- | :--- |
| Low LR + Adam | FOLD Gain Max | Flat solutions favor folding |
| High LR + Adam | Gain shrinks/reverses | Sharp solutions diminish cluster projection advantage |
| + SAM Training | Both improve, FOLD more | SAM guides toward flatter solutions |
| + Strong Augmentation | Narrowed gap on CNN | Robustness makes axis-aligned projection less detrimental |
| After Fine-tuning (1-5 epoch) | FOLD maintains lead | Folding provides better initialization |

### Key Findings

- Folding consistently outperforms pruning at medium-to-high compression rates, with the gap widening as the rate increases.
- Training conditions promoting flat solutions (moderate learning rates, SAM) amplify folding's advantages.
- Fine-tuning converges faster after folding, indicating folding provides a superior compression initialization.
- For LLaMA-60M at 50% compression, folding's PPL is approximately half that of pruning.

## Highlights & Insights

- The projection geometry perspective is elegant: a single framework unifies two seemingly distinct compression methods and provides clear theoretical comparisons via projection error.
- The large-scale experimental design with 1000+ checkpoints is highly systematic, revealing for the first time the fine-grained impact of upstream training hyperparameters on compression efficacy.
- Folding is essentially "merging similar directions" rather than "deleting coordinates"; this geometric intuition is inspiring for future compression method design.

## Limitations & Future Work

- Theoretical guarantees require a rank difference $k_f = k_p + 1$; while the practical impact is negligible, it is not a strictly matched-size comparison.
- For ViT and LLaMA, only FFN blocks were compressed; folding for attention layers remains unexplored.
- Combination with other compression methods like quantization or distillation was not evaluated.
- Large-scale LLMs (>1B parameters) were not covered due to training cost constraints.

## Related Work & Insights

- **vs Magnitude Pruning**: Pruning is coordinate projection (losing directional info), while folding is cluster projection (preserving direction); folding has smaller theoretical reconstruction error.
- **vs Model Folding (Wang et al., 2025)**: This work provides the first theoretical foundation for folding, proving its superiority is not merely an empirical observation.
- **vs SoTA LLM Pruning (Wanda, SparseGPT)**: These methods rely on calibration data and belong to a different setting; this work compares methods in a calibration-free setting.

## Rating

- Novelty: ⭐⭐⭐⭐ The unified projection geometry framework is a fresh perspective; Theorems 2.1/2.2 are the core contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 1000+ checkpoints across multiple architectures/datasets/hyperparameters; very rich ablations.
- Writing Quality: ⭐⭐⭐⭐ Theory and experiments are well-intertwined and clearly presented.
- Value: ⭐⭐⭐⭐ Provides solid theoretical guidance and a practical alternative for calibration-free compression.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">
</div>

<!-- RELATED:END -->

## Related Papers

- [\[CVPR 2025\] Less is More: Efficient Model Merging with Binary Task Switch](../../CVPR2025/model_compression/less_is_more_efficient_model_merging_with_binary_task_switch.md)
- [\[ACL 2026\] Quantize What Counts: More for Keys, Less for Values](../../ACL2026/model_compression/quantize_what_counts_more_for_keys_less_for_values.md)
- [\[NeurIPS 2025\] Less is More but Where: Dynamic Token Compression via LLM-Guided Keyframe Prior](../../NeurIPS2025/model_compression/less_is_more_but_where_dynamic_token_compression_via_llm-guided_keyframe_prior.md)
- [\[ICLR 2026\] Quantized Gradient Projection for Memory-Efficient Continual Learning](quantized_gradient_projection_for_memory-efficient_continual_learning.md)
- [\[ICLR 2026\] PiCa: Parameter-Efficient Fine-Tuning with Column Space Projection](pica_parameter-efficient_fine-tuning_with_column_space_projection.md)

</div>

<!-- RELATED:END -->

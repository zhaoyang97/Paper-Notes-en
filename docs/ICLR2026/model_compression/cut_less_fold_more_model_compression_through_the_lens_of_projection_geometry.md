---
title: >-
  [Paper Note] Cut Less, Fold More: Model Compression through the Lens of Projection Geometry
description: >-
  [ICLR 2026][Model Compression][Model folding] This paper unifies structured pruning and model folding under an orthogonal projection framework—pruning as coordinate-aligned projection and folding as clustering subspace projection—and proves that folding yields strictly smaller parameter reconstruction error under a rank-one difference condition. Validation across 1,000+ checkpoints demonstrates that folding consistently outperforms pruning at medium-to-high compression ratios.
tags:
  - ICLR 2026
  - Model Compression
  - Model folding
  - structured pruning
  - orthogonal projection
  - calibration-free compression
  - projection geometry
date: 2026-05-08
content_hash: a6d60a6479bc9446
---

# Cut Less, Fold More: Model Compression through the Lens of Projection Geometry

**Conference**: ICLR 2026
**arXiv**: [2602.18116](https://arxiv.org/abs/2602.18116)
**Code**: Available (appendix link)
**Area**: Model Compression
**Keywords**: Model folding, structured pruning, orthogonal projection, calibration-free compression, projection geometry

## TL;DR

This paper unifies structured pruning and model folding under an orthogonal projection framework—pruning as coordinate-aligned projection and folding as clustering subspace projection—and proves that folding yields strictly smaller parameter reconstruction error under a rank-one difference condition. Validation across 1,000+ checkpoints demonstrates that folding consistently outperforms pruning at medium-to-high compression ratios.

## Background & Motivation

**State of the Field**: Calibration-free post-training structured compression is a critical requirement for model deployment. The dominant approach is magnitude-based structured pruning, which removes neurons, channels, or filters according to weight magnitude. Model folding has recently been proposed as an alternative, achieving compression by clustering similar weights and binding them together.

**Limitations of Prior Work**: (1) Pruning zeroes out weights directly, causing large parameter perturbations and functional drift; (2) folding as an alternative lacks theoretical grounding, and the conditions under which it outperforms pruning remain unclear; (3) a unified comparative framework for the two approaches is absent.

**Root Cause**: Pruning removes weights via coordinate-aligned projection in parameter space, discarding directional information, whereas folding preserves merged directions but lacks theoretical guarantees quantifying the benefit of this preservation.

**Paper Goals**: Establish a unified projection-theoretic framework for pruning and folding, and rigorously prove the superiority of folding in terms of parameter reconstruction and functional preservation.

**Starting Point**: Both compression methods are treated as orthogonal projections in parameter space—pruning corresponds to a coordinate-aligned subspace and folding corresponds to a clustering-structured subspace.

**Core Idea**: Pruning is a coordinate projection $\mathbf{C}_p = \begin{pmatrix} I & 0 \\ 0 & 0 \end{pmatrix}$, while folding is a clustering projection $\mathbf{C}_f = \mathbf{U}_f(\mathbf{U}_f^\top \mathbf{U}_f)^{-1}\mathbf{U}_f^\top$. The latter retains more directional information in parameter space, yielding a smaller reconstruction error $\|\mathbf{W} - \mathbf{W}_f\|_F^2 \leq \|\mathbf{W} - \mathbf{W}_p\|_F^2$.

## Method

### Overall Architecture

Given a pretrained weight matrix $\mathbf{W} \in \mathbb{R}^{m \times p}$ → define an orthogonal projection operator (pruning $\mathbf{C}_p$ or folding $\mathbf{C}_f$) → project to obtain compressed weights $\mathbf{W}_{\text{comp}} = \mathbf{C} \mathbf{W}$ → apply REPAIR (BatchNorm re-estimation) for CNNs or LayerNorm re-initialization for ViTs → optional short-term fine-tuning.

### Key Designs

1. **Unified Orthogonal Projection Framework**:

    - Function: Unifies pruning and folding as orthogonal projections in parameter space.
    - Mechanism: Pruning retains the first $k$ neurons, corresponding to $\mathbf{U}_p = \begin{pmatrix} I \\ 0 \end{pmatrix}$; folding clusters the $m$ parameter vectors into $k$ groups and replaces each with the cluster mean, corresponding to a one-hot cluster assignment matrix $\mathbf{U}_f \in \{0,1\}^{m \times k}$.
    - Design Motivation: The orthogonal projection $\mathbf{C}y = \arg\min_{z \in \text{Range}(\mathbf{U})} \|y - z\|_2$ maps to the nearest point in a subspace and naturally measures compression-induced parameter distortion.

2. **Theoretical Proof of Folding's Superiority (Theorem 2.1 & 2.2)**:

    - Function: Proves that for any pruning scheme, there exists a folding scheme with strictly smaller reconstruction error.
    - Mechanism: Theorem 2.1 gives a constructive proof: merging all pruned rows into one additional cluster (rank $k_f = k_p + 1$) yields a Frobenius-norm reconstruction error no greater than that of pruning. Theorem 2.2 further establishes that optimal $k$-means folding satisfies $\|\mathbf{W} - \mathbf{W}_f^\star\|_F^2 \leq \|\mathbf{W} - \mathbf{W}_f'\|_F^2 \leq \|\mathbf{W} - \mathbf{W}_p\|_F^2$.
    - Design Motivation: Combined with the Lipschitz continuity of the loss $|L(\mathbf{W}_1) - L(\mathbf{W}_2)| \leq \kappa \|\mathbf{W}_1 - \mathbf{W}_2\|_F$, smaller parameter reconstruction error directly implies smaller functional perturbation.

3. **Large-Scale Hyperparameter Ablation Validation**:

    - Function: Systematically compares folding and pruning across 1,000+ checkpoints under diverse training conditions.
    - Mechanism: Covers Adam/SGD optimizers, varying learning rates, data augmentation, regularization, SAM training, and LLaMA-60M/130M, thereby verifying the applicable boundaries of the theoretical predictions.
    - Design Motivation: Existing pruning studies vary only random seeds while fixing hyperparameters, leaving unexplored how upstream training conditions affect compression outcomes.

### Loss & Training

Compression itself is calibration-free and training-free. Folding determines groupings via $k$-means clustering. Optional post-processing includes REPAIR (BatchNorm statistics re-estimation) for CNNs, LayerNorm re-initialization for ViTs, or 1–5 epochs of fine-tuning.

## Key Experimental Results

### Main Results

| Architecture/Dataset | Compression Ratio | FOLD Acc | MAG Acc | Folding Advantage |
|---|---|---|---|---|
| ResNet18/CIFAR-10 (Adam) | 50% | Significantly higher | — | Largest at medium–high compression |
| ViT-B/32/CIFAR-10 | 50% | Significantly higher | — | Consistent positive gain |
| CLIP ViT-B/32/ImageNet-1K | 50% | Higher | — | Maintained after LayerNorm reset |
| LLaMA-60M/C4 (PPL↓) | 20% | 47.17 | 54.51 | FOLD achieves lower PPL |
| LLaMA-60M/C4 (PPL↓) | 50% | 221.32 | 398.62 | FOLD leads by a large margin |

### Ablation Study

| Configuration | Metric | Note |
|---|---|---|
| Low LR + Adam | FOLD advantage largest | Flat minima favor folding |
| High LR + Adam | Advantage narrows/reverses | Sharp minima weaken clustering projection benefit |
| + SAM training | Both improve; FOLD improves more | SAM induces flatter minima |
| + Strong data augmentation | Gap narrows on CNNs | Enhanced robustness makes coordinate-aligned projection less detrimental |
| After fine-tuning (1–5 epochs) | FOLD maintains lead | Folding provides a better initialization |

### Key Findings

- Folding consistently outperforms pruning at medium-to-high compression ratios, with the gap widening as compression increases.
- Training conditions that promote flat minima (moderate learning rate, SAM) amplify the folding advantage.
- Fine-tuning after folding converges faster, indicating that folding provides a superior compressed initialization.
- For LLaMA-60M at 50% compression, folding achieves approximately half the perplexity of pruning.

## Highlights & Insights

- The projection geometry perspective is particularly elegant: a single framework unifies two seemingly distinct compression methods and provides a clear theoretical comparison grounded in projection error.
- The large-scale experimental design spanning 1,000+ checkpoints is exceptionally systematic and is the first to reveal the fine-grained influence of upstream training hyperparameters on compression outcomes.
- Folding fundamentally "merges similar directions" rather than "deletes coordinates"; this geometric intuition offers meaningful inspiration for designing future compression methods.

## Limitations & Future Work

- The theoretical guarantee requires a rank difference of $k_f = k_p + 1$; while the practical impact is negligible, this does not constitute a strictly matched-size comparison.
- For ViTs and LLaMA, only FFN blocks are compressed; folding of attention layers remains unexplored.
- No combined evaluation with other compression paradigms such as quantization or knowledge distillation is performed.
- Large-scale LLMs (>1B parameters) are not covered due to training cost constraints.

## Related Work & Insights

- **vs. Magnitude Pruning**: Pruning is a coordinate projection (discards directional information), whereas folding is a clustering projection (preserves directions); folding is theoretically guaranteed to yield smaller reconstruction error.
- **vs. Model Folding (Wang et al., 2025)**: This paper provides the first theoretical foundation for folding, demonstrating that its superiority is not merely an empirical observation.
- **vs. SoTA LLM Pruning (Wanda, SparseGPT)**: These methods rely on calibration data and operate under a different setting; this paper addresses the calibration-free setting.

## Rating

- Novelty: ⭐⭐⭐⭐ The unified projection geometry framework is a genuinely novel perspective; Theorems 2.1 and 2.2 constitute the core contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Coverage of 1,000+ checkpoints across multiple architectures, datasets, and hyperparameters yields exceptionally rich ablations.
- Writing Quality: ⭐⭐⭐⭐ Theory and experiments are interwoven as mutual evidence, with clear and precise exposition.
- Value: ⭐⭐⭐⭐ Provides solid theoretical guidance and a practical alternative for calibration-free compression.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Less is More but Where: Dynamic Token Compression via LLM-Guided Keyframe Prior](../../NeurIPS2025/model_compression/less_is_more_but_where_dynamic_token_compression_via_llm-guided_keyframe_prior.md)
- [\[ICCV 2025\] Achieving More with Less: Additive Prompt Tuning for Rehearsal-Free Class-Incremental Learning](../../ICCV2025/model_compression/achieving_more_with_less_additive_prompt_tuning_for_rehearsal-free_class-increme.md)
- [\[ICLR 2026\] Topology and Geometry of the Learning Space of ReLU Networks: Connectivity and Size](topology_and_geometry_of_the_learning_space_of_relu_networks_connectivity_and_si.md)
- [\[ICLR 2026\] The Geometry of LLM Quantization: GPTQ as Babai's Nearest Plane Algorithm](the_geometry_of_llm_quantization_gptq_as_babais_nearest_plane_algorithm.md)
- [\[CVPR 2026\] UniComp: Rethinking Video Compression Through Informational Uniqueness](../../CVPR2026/model_compression/unicomp_rethinking_video_compression_through_informational_uniqueness.md)

<!-- RELATED:END -->

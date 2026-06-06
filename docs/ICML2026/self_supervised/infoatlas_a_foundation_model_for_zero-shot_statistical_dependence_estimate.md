---
title: >-
  [Paper Note] InfoAtlas: A Foundation Model for Zero-Shot Statistical Dependence Estimation
description: >-
  [ICML 2026][Self-Supervised Learning][Mutual Information] InfoAtlas transforms mutual information (MI) estimation from an optimization problem requiring a per-dataset critic network into a "single forward pass" problem u…
tags:
  - "ICML 2026"
  - "Self-Supervised Learning"
  - "Mutual Information"
  - "Foundation Models"
  - "Hypernetworks"
  - "Sliced Mutual Information"
  - "Synthetic Data Pre-training"
date: 2026-05-08
content_hash: b511235ff8525feb
---

# InfoAtlas: A Foundation Model for Zero-Shot Statistical Dependence Estimation

**Conference**: ICML 2026  
**arXiv**: [2606.00241](https://arxiv.org/abs/2606.00241)  
**Code**: The paper mentions the InfoAtlas-project page (link provided in the paper)  
**Area**: Self-Supervised Learning / Foundation Models / Mutual Information Estimation  
**Keywords**: Mutual Information, Foundation Models, Hypernetworks, Sliced Mutual Information, Synthetic Data Pre-training

## TL;DR
InfoAtlas transforms mutual information (MI) estimation from an optimization problem requiring a per-dataset critic network into a "single forward pass" problem using a hypernetwork pre-trained on large-scale synthetic data. It achieves accuracy comparable to neural estimators like MINE/MINDE while providing a 100× speedup.

## Background & Motivation

**Background**: Mutual Information (MI) is a standard tool for measuring statistical dependence between sets of multidimensional variables. Mainstream neural estimators like MINE / InfoNCE / MINDE utilize variational lower bounds, such as Donsker-Varadhan (DV), to formulate MI estimation as $\mathbb{I}(\mathbf{x}, \mathbf{y}) \coloneqq \sup_\theta \mathbb{E}_{p_{xy}}[\theta] - \log(\mathbb{E}_{p_x \otimes p_y}[e^\theta])$, approximating the optimal critic $\theta$ with a neural network.

**Limitations of Prior Work**: All neural MI estimators share a critical bottleneck: a specific critic network must be trained from scratch for every new dataset, requiring thousands of gradient descent steps to converge (complexity $\mathcal{O}(T)$). This renders them nearly unusable for real-time scenarios like high-frequency financial correlation monitoring or large-scale genetic screening. Previous work such as InfoNet (Hu et al. 2024) attempted to bypass training via look-up tables but is restricted to 1D inputs; scaling to $d$ dimensions would require $\mathcal{O}(e^d)$ space, which becomes infeasible by $d=8$, and it cannot handle variable-dimensional data.

**Key Challenge**: The accuracy of neural MI estimation depends on training a dedicated critic for the specific data, but the training cost scales linearly with the number of datasets. To achieve "one-pass MI inference," one must find a way to make the critic parameters a function of the dataset itself, $\theta^* = \mathcal{H}(\mathcal{D})$, and ensure this mapping generalizes to unseen real-world data.

**Goal**: To degrade MI estimation to a single inference task—a unified model that processes multivariate data of arbitrary dimensions and sample sizes, skipping per-dataset optimization while maintaining strong generalization to real-world scenarios (e.g., CLIP, video, robotics).

**Key Insight**: Ours draws inspiration from the success of foundation models like TabPFN and Chronos in tabular and time-series forecasting: large-scale synthetic pre-training followed by single-pass inference. This paper applies the same philosophy to MI estimation: synthesizing a "dependence structure space" where the model learns to "infer critic parameters from samples."

**Core Idea**: A hypernetwork $\mathcal{H}$ maps a dataset to the full parameters of a DV critic. Pre-trained on a massive volume of synthetic dependence structures (copula mixtures + flow transformations), the model achieves zero-shot generalization to unseen distributions. For high-dimensional data, sliced MI is used to decompose the problem into multiple low-dimensional projections, processed in parallel batches via a transformer.

## Method

### Overall Architecture

The core of InfoAtlas is an attention-based hypernetwork $\mathcal{H}: \mathcal{D} \mapsto \Theta$. The input consists of $n$ sample pairs $\{(\mathbf{x}^i, \mathbf{y}^i)\}_{i=1}^n$, and the output is the complete set of parameters $\theta$ for the DV critic (flattened into a single vector). Once $\theta$ is obtained, the empirical DV formula $\hat{\mathbb{I}}_\theta(\mathbf{x}, \mathbf{y}) = \frac{1}{n}\sum_i \theta(\mathbf{x}^i, \mathbf{y}^i) - \log(\frac{1}{n}\sum_j e^{\theta(\mathbf{x}^j, \mathbf{y}^{\pi(j)})})$ is used to calculate MI in one step, where $\pi$ is a random permutation for marginal samples. Pre-training is conducted on synthetic copula mixture distributions. When the dimension $d > D = 20$, the model switches to $k$-sliced MI, splitting the high-dimensional problem into $S$ $k$-dimensional sub-problems and feeding them as a batch to the same $\mathcal{H}$.

### Key Designs

1.  **Dual-Path Hypernetwork**:
    - **Function**: Directly generates all parameters $\theta$ of the DV critic from samples, bypassing gradient descent.
    - **Mechanism**: Encodes joint and marginal distribution features through two independent paths. **Joint Path**: Each sample pair $[\mathbf{x}^i; \mathbf{y}^i]$ is treated as a token; cross-attention with a learnable query $\mathbf{q}_{joint}$ aggregates features (attention weights $\alpha_i = \mathrm{softmax}(\mathbf{q}_{joint}^\top \mathbf{W}_K [\mathbf{x}^i; \mathbf{y}^i]/\sqrt{d_{model}})$ emphasize high-correlation samples), followed by 16 self-attention layers to obtain $\mathbf{h}_{joint}$. **Marginal Path**: $\{\mathbf{x}^i\}$ and $\{\mathbf{y}^i\}$ are projected via MLP into bidirectional cross-attention ($\mathbf{q}_{x\to y}$ and $\mathbf{q}_{y\to x}$); their sum passes through 8 self-attention layers to obtain $\mathbf{h}_{marginal}$. Finally, $\mathbf{h}_{fused} = \mathrm{CrossAttention}(\mathbf{h}_{marginal}, \mathbf{h}_{joint}, \mathbf{h}_{joint})$ is processed by an MLP to output $\theta \in \mathbb{R}^{|\Theta|}$.
    - **Design Motivation**: The DV formula naturally splits into expectations over joint and product-marginal distributions. This dual-path architecture aligns with the DV structure; the attention mechanism ensures permutation invariance while dynamically focusing on relevant sample pairs.

2.  **Noise Padding for Variable Dimensionality**:
    - **Function**: Enables a single model to handle varying input $(d_x, d_y)$ while maintaining a unified architecture.
    - **Mechanism**: For inputs with dimension $d < D$, independent Gaussian noise $\mathcal{N}(0, \mathbf{I})$ is used to pad variables to $D$ dimensions. Proposition A.3 in the paper proves that if $\mathbf{n}_x, \mathbf{n}_y$ are mutually independent and independent of $\mathbf{x}, \mathbf{y}$, then $\mathbb{I}(\mathbf{x}, \mathbf{y}) = \mathbb{I}([\mathbf{x}; \mathbf{n}_x], [\mathbf{y}; \mathbf{n}_y])$; thus, padding preserves MI.
    - **Design Motivation**: Avoids training separate models for every dimension combination and prevents "spurious symmetries" introduced by zero-padding—noise padding provides true MI-preserving augmentation.

3.  **Diversity-Driven Synthetic Pretraining**:
    - **Function**: Constructs a "meta-distribution" $p(\mathcal{D})$ covering as many real-world statistical patterns as possible for large-scale pre-training.
    - **Mechanism**: Dependence diversity is achieved via copula mixtures $\mathbf{x}, \mathbf{y} \sim \sum_{i=1}^K \pi_i c_i$, where $c_i$ are Gaussian copulas (arbitrary correlation matrices) and Student-$t$ copulas (varying tail dependence), with $K=60$. Marginal diversity is handled by randomly initialized invertible flow models $\mathbf{x} \leftarrow f_X(\mathbf{x})$ and $\mathbf{y} \leftarrow f_Y(\mathbf{y})$ (bijections preserve MI), followed by a softrank to normalize marginals to uniform. The pre-training objective $\mathcal{L}(\mathcal{H}) = -\mathbb{E}_{\mathcal{D} \sim p(\mathcal{D})}[\hat{\mathbb{I}}_{\mathcal{H}(\mathcal{D})}(\mathbf{x}_\mathcal{D}, \mathbf{y}_\mathcal{D})]$ directly maximizes the estimated MI (which is equivalent to minimizing the negative DV bound).
    - **Design Motivation**: Synthetic data allows for "infinite" samples and batch sizes, avoiding high variance/bias issues inherent in neural MI estimators when dataset sizes are small (McAllester & Stratos 2020). Copula mixtures + flows represent the simplest combination capable of approximating arbitrary joint distributions, offering more control than GANs or diffusion models.

### Loss & Training

The pre-training objective is $\mathcal{L}(\mathcal{H}) = -\mathbb{E}_{\mathcal{D} \sim p(\mathcal{D})}[\hat{\mathbb{I}}_{\mathcal{H}(\mathcal{D})}]$. Proposition A.1 provides a consistency result: under mild conditions, the optimal solution to this objective is the optimal critic corresponding to the ground-truth MI. During inference, high-dimensional ($d>20$) data uses sliced MI: $\mathbf{P}_i, \mathbf{P}'_i \in \mathbb{R}^{k\times d}$ are randomly sampled from the Stiefel manifold. $S$ sets of projected data $\{\mathbf{P}_i \mathbf{x}, \mathbf{P}'_i \mathbf{y}\}$ are packed into a batch for one forward pass to yield $S$ critics, reducing complexity from $\mathcal{O}(ST)$ to $\mathcal{O}(1)$.

## Key Experimental Results

### Main Results

| Task / Dataset | Metric | InfoAtlas | Main Baseline | Remarks |
| :--- | :--- | :--- | :--- | :--- |
| BMI Mn-dense 5-5-0.5 (GT=0.59) | MI Est. | $\mathbf{0.60}$ | MINE 0.60 / MINDE 0.58 | Comparable to best neural estimators |
| BMI Asinh@St 5-5-2 (GT=0.45) | MI Est. | $\mathbf{0.41}$ | MINE 0.53 / MINDE 0.43 | Closer to GT |
| BMI Total Latency | Seconds | $\mathbf{0.09}$ | MINE 25.9 / InfoNCE 67.6 | ~300× Gain |
| CLIP 512D Image-Text MI | Noise Sens. | Clear bands | InfoNet High Error | 5-sliced, $S=25$ |
| PointOdyssey Track Seg. | AUC-PR | Comparable | MINE comparable but slower | Completed in seconds |
| ManiSkill 2 Pick Cube Seen | Success Rate | $\mathbf{94.2\%}$ | MINE-1000 81.2 / No-MI 66.0 | 25 slices |
| ManiSkill 2 Peg Insertion Seen | Success Rate | $\mathbf{72.4\%}$ | MINE-1000 65.4 / InfoNet 46.4 | 25 slices |

Ours matches the accuracy of gradient-optimized methods like MINE / MINDE while achieving a 100×–300× speed advantage. Compared to the speed-focused InfoNet, Ours handles multi-dimensional and variable-dimensional data, leading to higher success rates in downstream robotics tasks.

### Ablation Study

| Configuration | Description |
| :--- | :--- |
| InfoAtlas (5-sliced, $S=25$) | Full setup, default for CLIP/Robotics |
| InfoNet (1-sliced, $S=128$) | Only 1D projection, significant information loss |
| MINE-100 / MINE-1000 | MINE with 100/1000 steps; more steps improve accuracy but increase latency |
| No-MI-Loss | No MI maximization in state extraction; success rate drops (Pick Cube 66.0 vs 94.2) |
| KNIFE (KDE-based) | Fails on high dimensions (Est. 0.93 vs GT 0.59 on Mn-dense) |

### Key Findings

- A single model works across vastly different real-world scenarios (CLIP, video, robotics) without fine-tuning, validating the hypothesis that synthetic copula + flow pre-training covers real distribution families.
- Slice dimension $k$ is more critical than number of slices $S$: InfoAtlas with $k=5, S=25$ significantly outperforms InfoNet with $k=1, S=128$, as 1D projections lose too much structural information.
- Non-parametric methods like KSG remain competitive in low dimensions (1-1) but fail completely in higher dimensions (above 5-5), reinforcing that high-dimensional MI requires parametric estimation and learned priors.
- When used as a plug-in for downstream tasks (e.g., robotic key state extraction), the speed advantage magnifies training efficiency. InfoAtlas makes step-by-step MI estimation for reward shaping feasible.

## Highlights & Insights

- The paradigm of "Hypernetwork outputs critic parameters" turns MI estimation into amortized inference. This can be extended to any "per-dataset optimization" task (Bayesian inference, density ratio estimation), serving as a clear demonstration of foundation model principles in statistical computing.
- Noise padding is a clever solution—maintaining MI invariance while solving variable dimensionality. It is cleaner than zero-padding or dimensionality-wise modeling and could be repurposed for alignment in sequence modeling.
- The dual-path architecture corresponds to the two expectation terms in the DV formula, reflecting a design philosophy where architecture is aligned with the objective.
- Sliced MI batch processing maximizes the advantages of transformers; getting $S$ critics in one forward pass versus $S$ independent training runs ($O(ST)$) is a powerful synergy of foundation models and slicing techniques.

## Limitations & Future Work

- **Upper Bound Limited by Pre-training Coverage**: Discrete data, heavy-tailed distributions, or long-range dependencies not present in the atlas may generalize poorly. Expanding the atlas requires more diverse synthetic distribution families (e.g., vine copulas).
- **Sliced MI is Not a Perfect Substitute**: Slicing may miss dependencies that are only prominent in specific high-dimensional directions; finite $S$ cannot guarantee coverage of all relevant directions.
- **Complexity Trade-off**: The current architecture for $D=20$ uses 16+8 layers of self-attention. Expanding to higher $D$ might cause the output $|\Theta|$ to explode, presenting scaling challenges. Potential solutions include using LoRA or low-rank decompositions for the critic.
- **Future Directions**: Making the hypernetwork conditional on data type/dimensions, or parameterizing the critic architecture as a graph to allow $\mathcal{H}$ to output graph structures rather than fixed weights.

## Related Work & Insights

- **vs MINE / InfoNCE / MINDE**: These are "per-dataset critic" neural estimators focusing on novel bounds. InfoAtlas is orthogonal—it keeps the bound structure but replaces training with hypernetwork inference, essentially trading training time for inference time.
- **vs InfoNet (Hu et al. 2024)**: While pioneers of the pre-training paradigm, InfoNet is restricted to 1D and relies on look-up tables. InfoAtlas uses an attention hypernetwork to support multi-dimensional and variable-dimensional data.
- **vs TabPFN / Chronos**: Sharing the same foundation model paradigm, InfoAtlas extends this pattern to tasks modeling relationships between two variable sets, which is inherently more complex than unidirectional prediction.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First true zero-shot foundation model for multi-dimensional MI; the hypernetwork-DV alignment is an original architecture.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Broad coverage from BMI sanity checks to real-world applications in CLIP and robotics.
- Writing Quality: ⭐⭐⭐⭐ Clear concepts and helpful analogies like "atlas/slice," though some failure mode discussions are relegated to the appendix.
- Value: ⭐⭐⭐⭐⭐ Enables MI estimation as a real-time plug-in for representation learning and robotics systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] From Zero to Hero: Advancing Zero-Shot Foundation Models for Tabular Outlier Detection](from_zero_to_hero_advancing_zero-shot_foundation_models_for_tabular_outlier_dete.md)
- [\[ICML 2026\] How 'Neural' is a Neural Foundation Model?](how_neural_is_a_neural_foundation_model.md)
- [\[ICML 2026\] Statistical Consistency and Generalization of Contrastive Representation Learning](statistical_consistency_and_generalization_of_contrastive_representation_learnin.md)
- [\[CVPR 2026\] Zero-Ablation Overstates Register Content Dependence in DINO Vision Transformers](../../CVPR2026/self_supervised/zero_ablation_overstates_register_content_dependence_in_dino_vision_transformers.md)
- [\[CVPR 2026\] LaS-Comp: Zero-shot 3D Completion with Latent-Spatial Consistency](../../CVPR2026/self_supervised/las-comp_zero-shot_3d_completion_with_latent-spatial_consistency.md)

</div>

<!-- RELATED:END -->

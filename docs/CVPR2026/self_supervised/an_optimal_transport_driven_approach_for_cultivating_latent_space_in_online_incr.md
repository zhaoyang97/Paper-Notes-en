---
title: >-
  [Paper Note] An Optimal Transport-driven Approach for Cultivating Latent Space in Online Incremental Learning
description: >-
  [CVPR 2026][Self-Supervised Learning][Online Class-Incremental Learning] This paper proposes MMOT, an online mixture model learning framework driven by optimal transport theory. By maintaining multiple adaptive centroids per class, MMOT more accurately captures the multimodal structure of online data streams. Combined with a dynamic preservation strategy that enhances class discriminability, MMOT effectively alleviates catastrophic forgetting in online class-incremental learning (OCIL).
tags:
  - CVPR 2026
  - Self-Supervised Learning
  - Online Class-Incremental Learning
  - Optimal Transport
  - Gaussian Mixture Model
  - Catastrophic Forgetting
  - Latent Space
date: 2026-05-08
content_hash: 39d899535d6293d4
---

# An Optimal Transport-driven Approach for Cultivating Latent Space in Online Incremental Learning

**Conference**: CVPR 2026
**arXiv**: [2211.16780](https://arxiv.org/abs/2211.16780)
**Code**: None
**Area**: Continual Learning / Online Incremental Learning
**Keywords**: Online Class-Incremental Learning, Optimal Transport, Gaussian Mixture Model, Catastrophic Forgetting, Latent Space

## TL;DR

This paper proposes MMOT, an online mixture model learning framework driven by optimal transport theory. By maintaining multiple adaptive centroids per class, MMOT more accurately captures the multimodal structure of online data streams. Combined with a dynamic preservation strategy that enhances class discriminability, MMOT effectively alleviates catastrophic forgetting in online class-incremental learning (OCIL).

## Background & Motivation

Online class-incremental learning (OCIL) is one of the most challenging continual learning scenarios: data distributions shift dynamically, the model performs only a single iteration over each arriving mini-batch, and no task ID is available at inference. This requires the model to continuously adapt to new classes under extremely limited replay conditions while retaining knowledge of previously seen classes.

Existing methods face two core limitations. First, most approaches represent each class in the latent space using a single classification head or a single prototype (centroid). However, real-world data streams are inherently multimodal—a single class may comprise multiple clusters—and a single centroid cannot capture such complexity. Second, some methods model each class using a Gaussian Mixture Model (GMM), but the means and variances are fixed after estimation and never updated. As the backbone network continuously adapts to new data, feature drift renders these static centroids increasingly inaccurate.

The root cause of both issues is that, while data arrives continuously and distributions shift persistently in OCIL, existing class representations are either too simplistic (single centroid) or too rigid (fixed GMM). The authors' key observation is that, by leveraging the rich mathematical toolkit of optimal transport (OT) theory, one can design a mixture model that is incrementally updated along with the data stream, thereby simultaneously addressing multimodal representation and feature drift.

**Core Idea**: The entropy-regularized dual formulation of the Wasserstein distance is used to reformulate GMM parameter learning as an expectation-form optimization problem, which naturally accommodates mini-batch-based online updates in OCIL. Gradient descent replaces the traditional EM algorithm to incrementally update multiple adaptive centroids.

## Method

### Overall Architecture

The OTC framework processes each time step through three stages: (1) initial training with cross-entropy loss to encourage intra-class clustering; (2) incremental estimation of per-class mixture model distributions via the MMOT framework; and (3) application of a dynamic preservation strategy using the learned multi-centroid information to enhance representation learning. The memory buffer is then updated accordingly.

### Key Designs

1. **MMOT (Multimodal Modeling via Optimal Transport)**

   - **Function**: Learns and incrementally updates multiple adaptive centroids and covariance matrices for each class.
   - **Mechanism**: For each class $c$, a GMM $\mathbb{Q}_c = \sum_{k=1}^K \pi_{k,c} \mathcal{N}(\mu_{k,c}, \text{diag}(\sigma_{k,c}^2))$ is used to approximate the data distribution $\mathbb{P}_c$ by minimizing the Wasserstein distance between the two. The entropy-regularized dual formulation converts this into an expectation-form objective $\max_\phi \{ \mathbb{E}_{\mathbb{P}_c}[\phi(z^c)] + \mathbb{E}_{\mathbb{Q}_c}[\tilde{\phi}(\tilde{z}^c)] \}$. Gumbel-Softmax reparameterization makes the mixture weights differentiable, enabling gradient-based online updates of all GMM parameters. This avoids the multi-iteration overhead of traditional EM, and the Wasserstein distance remains numerically stable even when the supports of the two distributions do not overlap—an advantage over KL divergence.
   - **Design Motivation**: OT is preferred over KL divergence for four reasons: the EM algorithm associated with KL is computationally expensive; the Wasserstein distance is a continuously differentiable metric; it is numerically stable when distribution supports are disjoint; and it respects the geometric structure of the data.

2. **Dynamic Preservation**

   - **Function**: Leverages the multi-centroid information learned by MMOT to enhance the model's class discriminability.
   - **Mechanism**: A contrastive-style loss $\mathcal{L}_{DP}$ is designed in which the positive term $g_{cen}^c$ pulls same-class representations toward all $K$ centroids by summing their similarities, while the negative term simultaneously repels representations from centroids and features of other classes. Multi-centroid representations provide finer-grained class boundary information compared to single-prototype approaches.
   - **Design Motivation**: Using multiple centroids instead of a single prototype allows boundary-region centroids to more effectively enhance inter-class separation, compensating for the inability of single prototypes to express intra-class multimodal structure.

3. **Centroid-based Memory Buffer Selection and Inference**

   - **Function**: Uses centroid information to improve diversity of stored samples and classification accuracy at inference.
   - **Mechanism**: During memory selection, the sample in the current batch closest to each centroid is added to the buffer, ensuring coverage of multiple sub-distributions per class. At inference, the Mahalanobis distance from a test sample to each Gaussian component of every class is computed, and the class with the minimum distance is predicted.
   - **Design Motivation**: Random selection tends to underrepresent minority sub-distributions; centroid-based selection ensures buffer representativeness. Mahalanobis distance incorporates covariance information, making it more suitable than Euclidean distance for classes with varied distributional shapes.

### Loss & Training

The overall training loss comprises three components: cross-entropy loss (initial class separation), the MMOT Wasserstein distance loss (GMM parameter learning), and the dynamic preservation loss $\mathcal{L}_{DP}$ (enhanced class discriminability). The training procedure first applies CE loss for initial training, then executes MMOT to update centroids, and finally refines the representation space with the dynamic preservation strategy.

## Key Experimental Results

### Main Results

| Dataset | Metric | OTC (Ours) | BiC+AC (Prev. SOTA) | GSA | MOSE |
|---------|--------|-----------|---------------------|-----|------|
| CIFAR-10 (M=0.2k) | Avg Acc↑ | **64.8** | 63.5 | 58.0 | 53.3 |
| CIFAR-10 (M=1k) | Avg Acc↑ | **76.1** | 75.8 | 69.1 | 70.7 |
| CIFAR-100 (M=2k) | Avg Acc↑ | **48.5** | 47.3 | 39.7 | 45.1 |
| CIFAR-100 (M=5k) | Avg Acc↑ | **56.5** | 54.2 | 49.7 | 54.5 |
| Tiny-ImageNet (M=2k) | Avg Acc↑ | **19.5** | 17.6 | 18.5 | 18.2 |
| Tiny-ImageNet (M=5k) | Avg Acc↑ | **31.6** | 22.6 | 26.0 | 30.9 |
| Tiny-ImageNet (M=10k) | Avg Acc↑ | **39.5** | 26.5 | 33.2 | 38.7 |

### Ablation Study

| Configuration | Avg Acc (CIFAR-10, M=1k) | Description |
|---------------|--------------------------|-------------|
| 1 centroid + random buffer | 71.6 | Single centroid + random selection |
| 4 centroids + random buffer | 75.3 | Multi-centroid, random selection |
| 4 centroids + centroid-based buffer | **75.9** | Full model |
| 1 centroid + centroid-based buffer | 71.6 | Single centroid + centroid-based selection |

### Key Findings

- **Multi-centroid contributes most**: Increasing from 1 to 4 centroids raises accuracy on CIFAR-10 from 71.6% to 75.9%, validating the necessity of multimodal modeling.
- **Optimal centroid count correlates with memory size**: Smaller memory budgets favor fewer centroids (optimal $K=3$ at $M=200$; $K=4$ at $M=1$k); performance degrades beyond the optimal count.
- **Largest gains on Tiny-ImageNet**: In the 20-task long-sequence setting, OTC surpasses the runner-up by 0.7% and 0.8% at $M=5$k and $M=10$k respectively, indicating greater advantage of multi-centroid modeling in longer task sequences.
- **Competitive forgetting**: OTC ranks among the top two in forgetting on CIFAR-10 and CIFAR-100, and top three on Tiny-ImageNet.

## Highlights & Insights

- **OT as a replacement for EM in online GMM learning** is the central innovation—replacing computationally expensive multi-iteration EM with a few steps of gradient descent, which is particularly critical in online learning settings. The key insight is that the entropy-regularized dual of the Wasserstein distance naturally takes an expectation form, making it perfectly suited for mini-batch updates.
- **Dual role of centroids in training and inference**: The learned centroids serve not only the dynamic preservation loss during training, but also Mahalanobis-distance-based classification and memory sample selection at inference—one unified representation serving multiple purposes.
- **Transferability of the multi-centroid paradigm**: The framework of multiple adaptive centroids updated via OT may offer value in other settings involving distributional shift, such as federated learning and domain adaptation.

## Limitations & Future Work

- The number of centroids $K$ remains a manually specified hyperparameter without an adaptive determination mechanism.
- Forgetting remains relatively high on Tiny-ImageNet (16.5%), suggesting that 20-task ultra-long sequences impose greater demands on the stability of centroid updates.
- The effect of the Kantorovich network $\phi$'s parameter count and number of update steps on performance is not thoroughly analyzed.
- Scalability to larger datasets (e.g., ImageNet-1k) has not been validated.

## Related Work & Insights

- **vs. CoPE**: CoPE uses a single adaptive centroid, achieving low forgetting but poor plasticity. OTC achieves higher accuracy with multiple centroids at the cost of slightly higher forgetting. t-SNE visualizations clearly show that OTC achieves superior inter-class separation compared to CoPE.
- **vs. MOSE**: MOSE approaches OTC in performance under large memory budgets but falls notably behind under small memory budgets, demonstrating that OTC's multi-centroid strategy is more effective under resource constraints.
- **vs. GSA**: GSA relies on fixed GMMs updated via EM. OTC's adaptive GMMs with OT-based updates outperform GSA across all experimental settings.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First application of OT to online GMM parameter learning in OCIL, with solid theoretical contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Three datasets, multiple memory sizes, and detailed ablations; large-scale experiments are lacking.
- **Writing Quality**: ⭐⭐⭐⭐ Theoretical derivations are clear, the motivation chain is complete, and figures are informative.
- **Value**: ⭐⭐⭐⭐ The paradigm of multiple adaptive centroids combined with OT updates offers meaningful insights for continual learning, though the absolute performance gains are modest.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Shape-of-You: Fused Gromov-Wasserstein Optimal Transport for Semantic Correspondence in-the-Wild](shape-of-you_fused_gromov-wasserstein_optimal_transport_for_semantic_corresponde.md)
- [\[CVPR 2026\] TALO: Pushing 3D Vision Foundation Models Towards Globally Consistent Online Reconstruction](talo_pushing_3d_vision_foundation_models_towards_globally_consistent_online_reco.md)
- [\[CVPR 2026\] LaS-Comp: Zero-shot 3D Completion with Latent-Spatial Consistency](las-comp_zero-shot_3d_completion_with_latent-spatial_consistency.md)
- [\[CVPR 2026\] MINE-JEPA: In-Domain Self-Supervised Learning for Mineral Exploration](mine-jepa_in-domain_self-supervised_learning_for_mine-like_object_classification.md)
- [\[CVPR 2026\] A Stitch in Time: Learning Procedural Workflow via Self-Supervised Plackett-Luce Ranking](a_stitch_in_time_learning_procedural_workflow_via_self_supervised_plackett_luce_r.md)

</div>

<!-- RELATED:END -->

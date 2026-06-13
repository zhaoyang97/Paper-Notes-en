---
title: >-
  [Paper Note] UniFusion: A Unified Image Fusion Framework with Robust Representation and Source-Aware Preservation
description: >-
  [CVPR2026][Optimization][unified image fusion] This paper proposes UniFusion, a unified image fusion framework that leverages the self-supervised semantic priors of DINOv3 to construct a cross-modal shared feature space…
tags:
  - "CVPR2026"
  - "Optimization"
  - "unified image fusion"
  - "DINOv3"
  - "bilevel optimization"
  - "reconstruction alignment"
  - "cross-task generalization"
date: 2026-05-08
content_hash: 4cd3a8cff381c307
---

# UniFusion: A Unified Image Fusion Framework with Robust Representation and Source-Aware Preservation

**Conference**: CVPR2026
**arXiv**: [2603.14214](https://arxiv.org/abs/2603.14214)  
**Code**: [dusongcheng/UniFusion](https://github.com/dusongcheng/UniFusion)  
**Area**: Optimization
**Keywords**: unified image fusion, DINOv3, bilevel optimization, reconstruction alignment, cross-task generalization

## TL;DR

This paper proposes UniFusion, a unified image fusion framework that leverages the self-supervised semantic priors of DINOv3 to construct a cross-modal shared feature space, preserves source image information via a reconstruction alignment mechanism, and decouples reconstruction and fusion objectives through a bilevel optimization strategy. The framework achieves state-of-the-art performance across multiple tasks, including infrared-visible, multi-exposure, multi-focus, and medical image fusion.

## Background & Motivation

**Core goal of image fusion**: To integrate complementary information from multi-source images into a single, information-rich, and visually coherent representation, serving downstream tasks such as object detection, medical diagnosis, and autonomous driving.

**Limitations of Prior Work**: Existing methods (CDDFuse, CoCoNet, LRRNet, etc.) are largely designed for specific fusion scenarios (infrared-visible, multi-exposure, multi-focus), employing task-customized CNN/AE/GAN architectures with limited generalization capability and poor adaptability to diverse fusion requirements.

**Background**: Recent Transformer-based architectures (SwinFusion), diffusion-model-based approaches, and methods such as TC-MoA have attempted to handle multiple tasks with a single model, yet are still constrained by two fundamental bottlenecks.

**Bottleneck 1: Lack of modality-consistent feature extraction**—existing shared backbones fail to establish principled and robust unified encodings across heterogeneous signals (infrared thermal imaging vs. visible-light texture).

**Bottleneck 2: Source information degradation during deep propagation**—modality-specific cues (e.g., visible-light texture, infrared radiation contrast) are progressively lost as features propagate through deep networks, leading to suboptimal fusion quality.

**Key Insight**: Can the strong semantic priors of a large-scale self-supervised pretrained model (DINOv3), combined with explicit reconstruction constraints and an optimization decoupling strategy, simultaneously address both bottlenecks?

## Method

### Overall Architecture

UniFusion comprises three major modules: (1) a dual-branch semantic feature extractor based on a frozen DINOv3 backbone with lightweight Adapters for domain adaptation; (2) a Cross-Attention fusion module; and (3) a reconstruction alignment branch. Training follows a bilevel optimization scheme: the inner level updates Adapter and reconstruction branch parameters $\phi$ (reconstruction objective), while the outer level updates fusion network parameters $\theta$ (fusion objective), with the two levels alternating iteratively to achieve joint optimization.

### Key Design 1: Semantic Prior Adaptation

- **Function**: A frozen DINOv3 ViT serves as the universal semantic backbone, extracting multi-layer features $f^{(l_2)}, f^{(l_5)}, f^{(l_8)}, f^{(l_{11})}$ from each modality independently, followed by lightweight hierarchical Adapters for progressive feature calibration.
- **Mechanism**: The Adapters act as "feature translators," progressively integrating deep global semantics with shallow fine-grained structures via multi-stage residual fusion and upsampling, producing modality-aligned embeddings.
- **Design Motivation**: DINOv3, pretrained on large-scale natural images, provides strong object-centric priors and long-range contextual dependencies. However, its latent space exhibits domain bias with respect to specific modalities (infrared, medical imaging). The Adapters bridge this gap with minimal parameter overhead while preserving the frozen backbone's generalization capacity and avoiding catastrophic forgetting.

### Key Design 2: Reconstruction Alignment

- **Function**: A lightweight reconstruction branch $R_m$ (several Transformer layers plus a projection head) is appended to each modality branch to reconstruct the original input $\bar{I}_m = R_m(\hat{\mathbf{F}}_m)$ from the calibrated features $\hat{\mathbf{F}}_m$ output by the Adapter.
- **Mechanism**: The self-reconstruction constraint forces the encoder to retain sufficient modality-specific information (texture, radiation contrast, etc.) in the shared latent space, preventing semantic drift and information loss.
- **Design Motivation**: Conventional methods constrain only the similarity between the fused output and the source images (pixel-level L1/SSIM), which tends to favor shallow texture imitation while neglecting deep semantic correspondence. Reconstruction alignment operates at the encoding stage, ensuring that the features themselves are "invertible"—capable of recovering the source image—thereby guaranteeing information completeness at the feature level. Ablation experiments (Fig. 8) visually demonstrate that removing the reconstruction branch causes the encoded features to lose significant modality-specific semantic representations.

### Key Design 3: Bilevel Optimization

- **Function**: Training is formalized as a bilevel optimization problem: the lower level rapidly updates Adapter and reconstruction parameters $\phi$ to capture modality-specific semantics, while the upper level slowly adjusts fusion network parameters $\theta$ based on the updated feature space.
- **Mechanism**:
  $$\phi^* = \arg\min_\phi \mathcal{L}_{\text{rec}}(\phi), \quad \theta^* = \arg\min_\theta \mathcal{L}_{\text{fuse}}(\theta; \phi^*)$$
  A first-order alternating scheme is adopted in practice: at each iteration, $\phi$ is updated with a larger learning rate $\eta_L$, followed by an update of $\theta$ with a smaller learning rate $\eta_U$; EMA regularization is applied to $\theta$ to enhance temporal stability.
- **Design Motivation**: Reconstruction and fusion are two coupled objectives—joint end-to-end training risks interference between reconstruction signals and fusion gradients, leading to unstable convergence. Bilevel optimization decouples them into sub-problems operating at different timescales: the inner level ensures that features "remember" source information, upon which the outer level learns the optimal fusion strategy, thereby balancing information preservation and fusion quality.

### Key Design 4: Cross-Attention Fusion Module

- **Function**: Four Cross-Attention Blocks perform dynamic interaction between the adapted features of the two modalities, modeling cross-modal dependencies and emphasizing complementary information.
- **Mechanism**: Each modality's features serve as queries to attend to the keys and values of the other modality, adaptively selecting and reinforcing valuable complementary regions.
- **Design Motivation**: Compared to simple concatenation or weighted averaging, cross-attention enables fine-grained information exchange at both spatial and semantic levels, making it particularly well-suited for scenarios with highly complementary modalities such as infrared-visible fusion.

## Key Experimental Results

### Table 1: Quantitative Comparison on Multi-Modal & Multi-Exposure Fusion

| Method | M3FD MI↑ | M3FD VIF↑ | M3FD $Q_{abf}$↑ | M3FD $Q_y$↑ | MEFB MI↑ | MEFB VIF↑ | MEFB CC↑ | MEFB PSNR↑ |
|--------|----------|-----------|-----------------|-------------|----------|-----------|----------|------------|
| CDDFuse | 3.776 | 0.839 | 0.610 | 0.978 | 6.575 | 1.430 | 0.837 | 56.809 |
| SwinFusion | 2.945 | 0.618 | 0.480 | 0.936 | 5.318 | 1.459 | 0.894 | 59.009 |
| TC-MoA | 3.466 | 0.870 | 0.636 | 0.983 | 4.889 | 1.406 | 0.885 | 59.152 |
| **UniFusion** | **4.268** | **0.899** | **0.637** | 0.982 | **6.861** | **1.484** | **0.906** | **59.219** |

- UniFusion achieves an MI of 4.268 on M3FD, substantially outperforming TC-MoA (3.466) by approximately 23%.
- UniFusion ranks best across all four metrics on MEFB, with VIF reaching 1.484 (surpassing SwinFusion's 1.459).

### Table 2: Ablation Study (M3FD / MEFB / MFIF)

| Configuration | M3FD MI↑ | M3FD VIF↑ | MEFB MI↑ | MEFB VIF↑ | MFIF MI↑ | MFIF $Q_{abf}$↑ |
|---------------|----------|-----------|----------|-----------|----------|-----------------|
| w/o Adapter | 3.646 | 0.863 | 5.512 | 1.232 | 5.375 | 0.532 |
| w/o DINOv3 | 3.681 | 0.879 | 5.709 | 1.334 | 5.624 | 0.491 |
| w/o Reconstruction | 3.846 | 0.870 | 6.434 | 1.396 | 5.838 | 0.579 |
| w/o Bilevel Opt | 3.924 | 0.876 | 6.374 | 1.424 | 6.021 | 0.583 |
| **Full Model** | **4.268** | **0.899** | **6.861** | **1.484** | **6.253** | **0.685** |

- Each component contributes significantly; removing the Adapter causes MFIF $Q_{abf}$ to drop from 0.685 to 0.532 (−22%).
- The semantic priors of the DINOv3 encoder are foundational; replacing it with a plain 4-layer Transformer leads to a comprehensive performance decline across tasks.
- Reconstruction alignment and bilevel optimization each contribute independently, with the best results achieved when both are combined.

## Highlights & Insights

1. **DINOv3 as a universal semantic backbone is an inspiring paradigm**: The frozen pretrained ViT + lightweight Adapter approach parallels LoRA/Adapter-tuning in NLP, and this paper provides the first systematic validation of this paradigm in the image fusion domain.
2. **Reconstruction alignment is an elegant information preservation mechanism**: Rather than applying constraints at the fusion output, the framework ensures feature information completeness through self-reconstruction at the encoding stage—a novel and visually convincing approach, as demonstrated by Fig. 8.
3. **Bilevel optimization is clearly formalized**: Decoupling reconstruction and fusion into optimization sub-problems at different timescales is theoretically grounded in bilevel optimization and efficiently implemented via a first-order alternating approximation in practice.
4. **Strong cross-task generalization**: A single model achieves state-of-the-art or competitive performance across four task categories (IVIF, MIF, MEF, MFF), requiring only 10K training iterations, which demonstrates practical applicability.

## Limitations & Future Work

1. **Dependency on DINOv3**: The frozen DINOv3 backbone is large (ViT-Large/Giant scale), incurring substantial inference overhead and posing challenges for edge deployment. Knowledge distillation into a smaller backbone is a natural direction to explore.
2. **Computational cost of bilevel optimization**: Although a first-order approximation is employed, the two-stage alternating updates still increase per-iteration computation, and the $\eta_L / \eta_U$ ratio requires careful tuning.
3. **No evaluation on misaligned scenarios**: Experiments do not cover cases where source images exhibit geometric misalignment (e.g., handheld multi-exposure, motion blur), which are common in real-world applications.
4. **Insufficient discussion of the reconstruction branch at inference time**: It remains unclear whether the reconstruction branch can be removed at inference to reduce computational cost; the paper does not explicitly describe any architecture simplification strategy for the inference stage.
5. **Fusion loss directly adopted from SwinFusion**: No innovation is introduced in the fusion loss function itself, leaving potential room for further improvement.

## Related Work & Insights

- **TC-MoA** [Zhu et al.]: A universal fusion method based on task-specific routing networks and the strongest baseline in this paper; UniFusion surpasses it through stronger semantic priors and the bilevel optimization strategy.
- **SwinFusion** [Ma et al.]: A cross-domain Swin Transformer framework; UniFusion adopts its fusion loss design while substantially improving upon it.
- **U2Fusion** [Xu et al.]: A pioneering all-in-one fusion method that inspired subsequent unified framework research.
- **DINOv2/v3**: Self-supervised ViT pretraining paradigms; this paper validates their transfer potential for low-level vision tasks.
- **Bilevel Optimization in Vision**: Widely applied in meta-learning, NAS, and hyperparameter optimization; introducing it to image fusion represents a meaningful contribution.
- **Inspiration from Adapter-tuning**: The frozen large model + lightweight adapter paradigm is well established in NLP; its successful application to low-level CV tasks in this paper merits attention.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The use of DINOv3 + Adapter as a universal fusion backbone is novel, the reconstruction alignment mechanism is cleverly designed, and the introduction of bilevel optimization is theoretically motivated. However, the individual components (Adapter-tuning, bilevel optimization) are not entirely new; the contribution lies in their effective combination.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers four major task categories (IVIF/MIF/MEF/MFF) across 6+ benchmarks, compared against 10 state-of-the-art methods, with complete ablation experiments (4 variants), rich qualitative visualizations (feature maps, fusion results), and downstream task validation provided in the appendix.
- **Writing Quality**: ⭐⭐⭐⭐ — The structure is clear, the methodological description is logically coherent, and the mathematical derivations are complete. Figures and tables are of high quality, and the feature visualizations in Fig. 8 are particularly persuasive. Some notation could be introduced earlier for clarity.
- **Value**: ⭐⭐⭐⭐ — Provides a practical unified fusion framework with engineering value in single-model cross-task generalization; the DINOv3 + Adapter paradigm is transferable to other low-level vision tasks; open-sourced code further enhances reproducibility and impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] AutoOpt: A Dataset and a Unified Framework for Automating Optimization Problem Solving](../../NeurIPS2025/optimization/autoopt_a_dataset_and_a_unified_framework_for_automating_optimization_problem_so.md)
- [\[ICML 2026\] URS: A Unified Neural Routing Solver](../../ICML2026/optimization/urs_a_unified_neural_routing_solver_for_cross-problem_zero-shot_generalization.md)
- [\[ICML 2026\] Stability Analysis of Sharpness-Aware Minimization](../../ICML2026/optimization/stability_analysis_of_sharpness-aware_minimization.md)
- [\[ICML 2026\] Cost-Aware Stopping for Bayesian Optimization](../../ICML2026/optimization/cost-aware_stopping_for_bayesian_optimization.md)
- [\[AAAI 2026\] SMoFi: Step-wise Momentum Fusion for Split Federated Learning on Heterogeneous Data](../../AAAI2026/optimization/smofi_step-wise_momentum_fusion_for_split_federated_learning_on_heterogeneous_da.md)

</div>

<!-- RELATED:END -->

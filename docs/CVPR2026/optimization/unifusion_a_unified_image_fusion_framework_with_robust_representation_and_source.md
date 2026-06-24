---
title: >-
  [Paper Note] UniFusion: A Unified Image Fusion Framework with Robust Representation and Source-Aware Preservation
description: >-
  [CVPR2026][Optimization][unified image fusion] The authors propose UniFusion, a unified image fusion framework that leverages DINOv3 self-supervised semantic priors to construct a cross-modal shared feature space. It preserves source image information via a reconstruction alignment mechanism and decouples reconstruction and fusion objectives using a bilevel optimization strategy, achieving SOTA performance across tasks such as infrared-visible, multi-exposure, multi-focus…
tags:
  - "CVPR2026"
  - "Optimization"
  - "unified image fusion"
  - "DINOv3"
  - "bilevel optimization"
  - "reconstruction alignment"
  - "cross-task generalization"
date: 2026-05-08
content_hash: 91b6f5cc82681918
---

# UniFusion: A Unified Image Fusion Framework with Robust Representation and Source-Aware Preservation

**Conference**: CVPR2026  
**arXiv**: [2603.14214](https://arxiv.org/abs/2603.14214)  
**Code**: [dusongcheng/UniFusion](https://github.com/dusongcheng/UniFusion)  
**Area**: Optimization  
**Keywords**: unified image fusion, DINOv3, bilevel optimization, reconstruction alignment, cross-task generalization

## TL;DR

The authors propose UniFusion, a unified image fusion framework that leverages DINOv3 self-supervised semantic priors to construct a cross-modal shared feature space. It preserves source image information via a reconstruction alignment mechanism and decouples reconstruction and fusion objectives using a bilevel optimization strategy, achieving SOTA performance across tasks such as infrared-visible, multi-exposure, multi-focus, and medical image fusion.

## Background & Motivation

**Goal of Image Fusion**: To integrate complementary information from multi-source images into a single, informative, and visually consistent representation, benefiting downstream tasks like object detection, medical diagnosis, and autonomous driving.

**Limitations of Prior Work**: Existing methods (e.g., CDDFuse, CoCoNet, LRRNet) are mostly designed for specific fusion scenarios (infrared-visible, multi-exposure, multi-focus) using customized CNN/AE/GAN architectures. Their generalization capability is limited, making it difficult to adapt to diverse fusion requirements.

**Background of Universal Fusion Frameworks**: Recent Transformer-based architectures (SwinFusion), diffusion-based methods, and TC-MoA attempt to handle multiple tasks with a single model but are still constrained by two core bottlenecks.

**Key Challenge 1: Lack of modal-consistent feature extraction mechanism**—Existing shared backbones fail to establish a principled and robust unified encoding across heterogeneous signals (thermal infrared vs. visible texture).

**Key Challenge 2: Source information degradation in deep propagation**—Modal-specific cues (e.g., visible textures, infrared radiation contrast) are gradually lost as features propagate through deep networks, leading to suboptimal fusion quality.

**Key Insight**: Can the strong semantic priors of large-scale self-supervised pre-trained models (DINOv3), combined with explicit reconstruction constraints and optimization decoupling strategies, simultaneously address these two bottlenecks?

## Method

### Overall Architecture

UniFusion aims to handle all fusion tasks with one model by assigning the problems of "modal-consistent feature extraction" and "source information preservation" to two distinct mechanisms. The pipeline operates as follows: two source images (e.g., infrared and visible) are fed into two frozen DINOv3 ViTs to extract multi-layer semantic features, which are then calibrated into a modally-aligned shared space via a lightweight Adapter. The calibrated features are processed in two ways: one path feeds into a Cross-Attention module for cross-modal fusion to output the fused image; the other path connects to reconstruction branches to restore the features back to the source images, forcing the features to retain information. Instead of joint training, a bilevel optimization strategy is used—the inner loop handles reconstruction (updating Adapter + reconstruction parameters $\phi$), while the outer loop handles fusion (updating fusion network parameters $\theta$), alternating iteratively.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Source Images I_x / I_y<br/>Infrared-Visible / Multi-Exposure / Multi-Focus / Medical"] --> B["Frozen DINOv3 ViT ×2<br/>Extract multi-layer features f(l2,l5,l8,l11)"]
    B --> C["Hierarchical Adapter<br/>Progressive calibration to shared space F̂_m"]
    C --> D["Cross-Attention Fusion<br/>Mutual query to exchange complementary info"]
    C --> E["Reconstruction Branch R_m<br/>Restore source images from F̂_m"]
    D --> F["Fused Image (Output)"]
    E --> G["Reconstructed Sources (Training Constraint)"]
    F -. Outer θ: Opt. Fusion .-> H["Bilevel Optimization Strategy<br/>Inner φ for reconstruction, Outer θ for fusion"]
    G -. Inner φ: Modal Info Preservation .-> H
```

### Key Designs

**1. DINOv3 Semantic Prior Adaptation: Large-scale frozen model as a universal semantic backbone with lightweight Adapters to bridge modal domain gaps**

The first bottleneck is the inability to build unified encoding between heterogeneous signals. UniFusion utilizes DINOv3 ViT, pre-trained via self-supervision on massive natural images, as a universal backbone. It extracts multi-layer features $f^{(l_2)}, f^{(l_5)}, f^{(l_8)}, f^{(l_{11})}$ and uses a hierarchical Adapter for progressive calibration: through multi-stage residual fusion and upsampling, deep global semantics and shallow fine-grained structures are integrated into modally-aligned embeddings. DINOv3 provides strong object-centric priors and long-range dependencies; while domain gaps exist for special modalities like infrared, the Adapter compensates with minimal parameters while the frozen backbone maintains generalization and prevents catastrophic forgetting.

**2. Cross-Attention Fusion Module: Fine-grained complementary information exchange via mutual querying**

The authors use four Cross-Attention Blocks to fuse aligned features. By letting each modality's features act as a query to attend to the other's key/value, the model adaptively identifies and strengthens complementary regions. Compared to static weighting, the attention mechanism dynamically determines information exchange at each position in both spatial and semantic dimensions, which is ideal for highly complementary fusion tasks.

**3. Reconstruction Alignment Mechanism: Self-reconstruction at the encoding end to ensure feature reversibility**

To prevent the degradation of modal-specific cues like radiation contrast in deep layers, UniFusion places constraints at the encoding end. Each modality branch is equipped with a lightweight reconstruction branch $R_m$ (Transformer layers + projection head) to reconstruct the original input $\bar{I}_m = R_m(\hat{\mathbf{F}}_m)$ from the calibrated features $\hat{\mathbf{F}}_m$. Successful reconstruction implies the features are "reversible" and retain modal-specific information. Feature visualization (Fig. 8) shows that without the reconstruction branch, encoding features lose significant modal-specific semantic representations.

**4. Bilevel Optimization Strategy: Decoupling reconstruction and fusion into different time-scale sub-problems**

Since the fusion and reconstruction branches share calibrated features, their objectives are coupled. Joint training can lead to reconstruction signals interfering with fusion gradients. UniFusion formulates training as a bilevel optimization:

$$\phi^* = \arg\min_\phi \mathcal{L}_{\text{rec}}(\phi), \quad \theta^* = \arg\min_\theta \mathcal{L}_{\text{fuse}}(\theta; \phi^*)$$

The inner (lower-level) loop rapidly updates Adapter and reconstruction parameters $\phi$ to capture modal-specific semantics. The outer (upper-level) loop optimizes fusion parameters $\theta$ on the updated feature space. A first-order alternating approximation is implemented using a larger learning rate $\eta_L$ for $\phi$ and a smaller $\eta_U$ for $\theta$, with EMA regularization on $\theta$ to enhance stability. This ensures the model first "remembers" source information before learning optimal fusion strategies.

## Key Experimental Results

### Main Results: Multi-Modal & Multi-Exposure Fusion

| Method | M3FD MI↑ | M3FD VIF↑ | M3FD $Q_{abf}$↑ | M3FD $Q_y$↑ | MEFB MI↑ | MEFB VIF↑ | MEFB CC↑ | MEFB PSNR↑ |
|------|----------|-----------|-----------------|-------------|----------|-----------|----------|------------|
| CDDFuse | 3.776 | 0.839 | 0.610 | 0.978 | 6.575 | 1.430 | 0.837 | 56.809 |
| SwinFusion | 2.945 | 0.618 | 0.480 | 0.936 | 5.318 | 1.459 | 0.894 | 59.009 |
| TC-MoA | 3.466 | 0.870 | 0.636 | 0.983 | 4.889 | 1.406 | 0.885 | 59.152 |
| **Ours** | **4.268** | **0.899** | **0.637** | 0.982 | **6.861** | **1.484** | **0.906** | **59.219** |

- Ours achieves an MI of 4.268 on M3FD, outperforming TC-MoA (3.466) by ~23%.
- On MEFB, Ours achieves the best results across all four metrics, with VIF reaching 1.484.

### Ablation Study (M3FD / MEFB / MFIF)

| Config | M3FD MI↑ | M3FD VIF↑ | MEFB MI↑ | MEFB VIF↑ | MFIF MI↑ | MFIF $Q_{abf}$↑ |
|------|----------|-----------|----------|-----------|----------|-----------------|
| w/o Adapter | 3.646 | 0.863 | 5.512 | 1.232 | 5.375 | 0.532 |
| w/o DINOv3 | 3.681 | 0.879 | 5.709 | 1.334 | 5.624 | 0.491 |
| w/o Reconstruction | 3.846 | 0.870 | 6.434 | 1.396 | 5.838 | 0.579 |
| w/o Bilevel Opt | 3.924 | 0.876 | 6.374 | 1.424 | 6.021 | 0.583 |
| **Full Model** | **4.268** | **0.899** | **6.861** | **1.484** | **6.253** | **0.685** |

- Each component contributes significantly; removing the Adapter reduces MFIF $Q_{abf}$ from 0.685 to 0.532 (-22%).
- DINOv3 semantic priors are foundational; reconstruction alignment and bilevel optimization provide independent and synergistic gains.

## Highlights & Insights

1. **DINOv3 as a universal semantic backbone is inspiring**: The paradigm of frozen pre-trained ViT + lightweight Adapter (similar to LoRA) is systematically validated for the first time in image fusion.
2. **Reconstruction alignment is an elegant preservation mechanism**: Placing constraints at the encoding level to ensure feature-level information completeness is a novel and convincing alternative to output-level pixel constraints.
3. **Formalized bilevel optimization**: Decoupling reconstruction and fusion into different optimization time-scales is theoretically grounded and efficiently implemented via first-order approximation.
4. **Strong cross-task generalization**: A single model reaches or nears SOTA across IVIF, MIF, MEF, and MFF with only 10K training iterations, showing high practical value.

## Limitations & Future Work

1. **DINOv3 Dependency**: The frozen backbone is large (ViT-Large/Giant), leading to high inference overhead on edge devices. Distillation to smaller backbones should be explored.
2. **Computational Cost of Bilevel Optimization**: While using first-order approximation, alternating updates still increase per-iteration compute, and the $\eta_L / \eta_U$ ratio requires careful tuning.
3. **Lack of evaluation on non-aligned scenarios**: The experiments do not cover cases with geometric misalignment (e.g., motion blur or hand-held multi-exposure).
4. **Reconstruction branch necessity**: It remains unclear if the reconstruction branch can be removed during inference to accelerate the model.
5. **Standard Fusion Loss**: Ours adopts the loss design from SwinFusion without specific innovation in the loss function itself.

## Related Work & Insights

- **TC-MoA**: A universal fusion method based on task-specific routing; UniFusion outperforms it via stronger semantic priors and bilevel optimization.
- **SwinFusion**: A cross-domain Swin Transformer framework; UniFusion builds upon its loss design with significant performance gains.
- **DINOv2/v3**: Self-supervised pre-training paradigms; this work validates their transfer potential for low-level vision.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Innovative use of DINOv3 + Adapter for fusion; elegant reconstruction alignment; theoretically supported bilevel optimization.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers 4 task categories, 6+ benchmarks, compares against 10 SOTA methods, and includes downstream task validation.
- **Writing Quality**: ⭐⭐⭐⭐ — Logic is clear, derivations are complete, and visualizations (Fig. 8) are highly convincing.
- **Value**: ⭐⭐⭐⭐ — Strong engineering value for cross-task generalization; the Adapter-tuning paradigm is applicable to other low-level vision tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] MASAM: Multimodal Adaptive Sharpness-Aware Minimization for Heterogeneous Data Fusion](../../ICLR2026/optimization/masam_multimodal_adaptive_sharpness-aware_minimization_for_heterogeneous_data_fu.md)
- [\[ICML 2026\] Selecting Samples on Graphs: A Unified Dataset Pruning Framework for Lossless Training Acceleration](../../ICML2026/optimization/selecting_samples_on_graphs_a_unified_dataset_pruning_framework_for_lossless_tra.md)
- [\[NeurIPS 2025\] AutoOpt: A Dataset and a Unified Framework for Automating Optimization Problem Solving](../../NeurIPS2025/optimization/autoopt_a_dataset_and_a_unified_framework_for_automating_optimization_problem_so.md)
- [\[ICML 2026\] URS: Unified Neural Routing Solver](../../ICML2026/optimization/urs_a_unified_neural_routing_solver_for_cross-problem_zero-shot_generalization.md)
- [\[CVPR 2026\] Beyond Single Solution: Multi-Hypothesis Collaborative Deep Unfolding Network for Image Compressive Sensing](beyond_single_solution_multi-hypothesis_collaborative_deep_unfolding_network_for.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[NeurIPS 2025\] AutoOpt: A Dataset and a Unified Framework for Automating Optimization Problem Solving](../../NeurIPS2025/optimization/autoopt_a_dataset_and_a_unified_framework_for_automating_optimization_problem_so.md)
- [\[CVPR 2026\] HyperNAS: Enhancing Architecture Representation for NAS Predictor via Hypernetwork](hypernas_enhancing_architecture_representation_for_nas_predictor_via_hypernetwor.md)
- [\[CVPR 2026\] FedRG: Unleashing the Representation Geometry for Federated Learning with Noisy Clients](fedrg_unleashing_the_representation_geometry_for_federated_learning_with_noisy_c.md)
- [\[CVPR 2026\] DABO: Difficulty-Aware Bayesian Optimization with Diffusion-Learned Priors](dabo_difficulty-aware_bayesian_optimization_with_diffusion-learned_priors.md)
- [\[AAAI 2026\] SMoFi: Step-wise Momentum Fusion for Split Federated Learning on Heterogeneous Data](../../AAAI2026/optimization/smofi_step-wise_momentum_fusion_for_split_federated_learning_on_heterogeneous_da.md)

</div>

<!-- RELATED:END -->

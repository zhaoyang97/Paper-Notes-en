---
title: >-
  [Paper Note] EdgeDiT: Hardware-Aware Diffusion Transformers for Efficient On-Device Image Generation
description: >-
  [CVPR 2026][Image Generation][Diffusion Transformer] EdgeDiT proposes a hardware-aware optimization framework for Diffusion Transformers that trains lightweight proxy blocks via hierarchical knowledge distillation and searches for Pareto-optimal architectures through multi-objective Bayesian optimization, achieving 20–30% parameter reduction, 36–46% FLOPs reduction, and 1.65× on-device speedup while maintaining or surpassing the generation quality of the original DiT-XL/2.
tags:
  - CVPR 2026
  - Image Generation
  - Diffusion Transformer
  - on-device deployment
  - hardware-aware optimization
  - knowledge distillation
  - architecture search
date: 2026-05-08
content_hash: 62e42e43f4c945cc
---

# EdgeDiT: Hardware-Aware Diffusion Transformers for Efficient On-Device Image Generation

**Conference**: CVPR 2026
**arXiv**: [2603.28405](https://arxiv.org/abs/2603.28405)
**Code**: N/A
**Area**: Diffusion Models / Model Compression
**Keywords**: Diffusion Transformer, on-device deployment, hardware-aware optimization, knowledge distillation, architecture search

## TL;DR
EdgeDiT proposes a hardware-aware optimization framework for Diffusion Transformers that trains lightweight proxy blocks via hierarchical knowledge distillation and searches for Pareto-optimal architectures through multi-objective Bayesian optimization, achieving 20–30% parameter reduction, 36–46% FLOPs reduction, and 1.65× on-device speedup while maintaining or surpassing the generation quality of the original DiT-XL/2.

## Background & Motivation

1. **State of the Field**: Diffusion Transformers (DiT) have emerged as a new paradigm for high-fidelity image generation, replacing U-Net backbones with Vision Transformers for improved scalability. Subsequent works such as MDT (masked modeling) and SiT (interpolant Transformer) have further advanced performance.

2. **Limitations of Prior Work**:
   - Existing DiT models impose enormous computational and memory demands, making them infeasible on resource-constrained edge devices.
   - Cloud-based inference is viable but raises privacy concerns, network dependency, and increased energy consumption.
   - Theoretical compute metrics (FLOPs/GMACs) do not reliably predict actual on-device latency—NPUs are specifically optimized for certain operations (e.g., GEMM), so reducing arithmetic computation does not proportionally reduce latency.

3. **Root Cause**: The strong generative capability of DiT stems from large-scale parameters and deep architectures, yet on-device deployment demands low latency and small memory footprint. Compressing the model while preserving generation quality is the core challenge, and actual hardware characteristics must be accounted for rather than optimizing theoretical metrics alone.

4. **Paper Goals**:
   - How to systematically discover efficient DiT architectures suited to mobile NPUs?
   - How to avoid full training of every candidate architecture in the search space?

5. **Starting Point**: Decompose the DiT architecture into replaceable, hardware-friendly proxy blocks; rapidly train proxies via hierarchical knowledge distillation; then apply multi-objective Bayesian optimization to identify Pareto-optimal architectures in the quality–latency space.

6. **Core Idea**: A three-stage decompose–distill–search pipeline: decompose DiT into a proxy block search space, efficiently train each proxy block with feature-level knowledge distillation, and apply Bayesian optimization to find a lightweight architecture that is Pareto-optimal in FID–latency.

## Method

### Overall Architecture
Using DiT-XL/2 (28 layers, 675M parameters) as the teacher model, the EdgeDiT framework proceeds in four steps: (1) construct a hardware-aware proxy block search space; (2) independently train each proxy block via feature-level knowledge distillation; (3) assemble candidate architectures and select Pareto-optimal models via multi-objective Bayesian optimization; (4) perform end-to-end training of the final selected architecture.

### Key Designs

1. **Hardware-Aware Proxy Block Search Space**:
   - *Function*: Define a set of hardware-friendly lightweight substitute modules constituting a structured search space.
   - *Mechanism*: Three proxy types are defined—(a) block removal: every two consecutive DiT layers are merged into one (Stage 1, $2^{14}$ combinations); (b) MLP ratio modification: FFN expansion ratio reduced from 4 to 2 (Stage 2); (c) hidden dimension reduction: projection dimension reduced from 1152 to 512 (Stage 2). In Stage 2, each layer has 4 options (2 MLP ratios × 2 dimensions), yielding $4^{28}$ combinations. The total search space is $2^{14} + 4^{28}$.
   - *Design Motivation*: Targeting the dataflow characteristics of mobile NPUs, computationally intensive and redundant operations are identified for structured simplification rather than random search.

2. **Feature-wise Knowledge Distillation (FwKD)**:
   - *Function*: Efficiently train proxy blocks to avoid the prohibitive cost of training the entire network from scratch.
   - *Mechanism*: A divide-and-conquer strategy is adopted—each proxy block is trained independently to minimize the discrepancy between its output $S_l(x)$ and the corresponding teacher block output $T_l(x)$, with loss $\mathcal{L}_{KD}^l = \|T_l(x) - S_l(x)\|_2^2$. Stage 1 trains 14 proxies (two-layer → one-layer); Stage 2 trains 56 proxies (28 layers × 2 variants). Since blocks are distilled independently, the process is highly parallelizable.
   - *Design Motivation*: Training every candidate architecture from scratch over a search space of $2^{14} + 4^{28}$ is entirely infeasible. Hierarchical distillation enables proxy blocks to rapidly approximate local behavior, requiring only minimal end-to-end fine-tuning afterward.

3. **Multi-Objective Bayesian Optimization (MOBO) for Architecture Selection**:
   - *Function*: Efficiently identify Pareto-optimal architectures in the two-dimensional FID–latency space.
   - *Mechanism*: Architecture configuration selection is formalized as a bi-objective optimization problem: $\max f(a)$ (generation quality / FID) and $\min g(a)$ (on-device latency). A Gaussian process serves as the surrogate model to predict objective values for candidate architectures, with the Expected Hypervolume Improvement (EHVI) acquisition function balancing exploration and exploitation. Discrete architecture configurations are relaxed to a continuous representation $x \in [0,1]^{28}$ and then mapped back to the nearest feasible architecture.
   - *Design Motivation*: Exhaustive evaluation is infeasible; Bayesian optimization efficiently approximates the Pareto frontier with a small number of evaluations.

### Loss & Training
- Distillation stage: $\mathcal{L}_{KD}^l = \|T_l(x) - S_l(x)\|_2^2$, with each proxy block trained independently.
- End-to-end training: standard diffusion training objective $\mathcal{L}_{diff} = \mathbb{E}[\|\epsilon - \epsilon_\theta(z_t, t)\|_2^2]$.
- The selected EdgeDiT-1 and EdgeDiT-6 are initialized from DiT-XL/2's 400K-iteration checkpoint and fine-tuned end-to-end for 100K iterations.

## Key Experimental Results

### Main Results — ImageNet 256×256 Class-Conditional Generation

| Model | Params (M) | FID-50K↓ | SFID↓ | IS↑ | Precision↑ | Recall↑ |
|-------|-----------|----------|-------|-----|-----------|---------|
| DiT-XL/2 | 675 | 16.23 | 11.06 | 80.91 | 0.93 | 0.26 |
| **EdgeDiT-1** | **471** | **12.3** | 13.97 | 75.72 | 0.92 | 0.24 |
| **EdgeDiT-6** | **530** | **12.4** | 14.96 | 78 | 0.91 | 0.25 |

### On-Device Latency Comparison (256×256)

| Model | Params (M) | GFLOPs | iPhone Latency (ms) | Samsung Latency (ms) |
|-------|-----------|--------|--------------------|--------------------|
| DiT-XL/2 | 675 | 237.34 | 118.56 | 129.00 |
| EdgeDiT-1 | 471 | 143.96 | 70.86 | 86.13 |
| EdgeDiT-6 | 530 | 169.97 | 72.53 | 89.22 |

### Ablation Study — Necessity of Knowledge Distillation

| Configuration | Description |
|---------------|-------------|
| EdgeDiT + FwKD | Normal quality, close to teacher model |
| EdgeDiT w/o FwKD (random init) | Significant degradation in image quality |

### Key Findings
- **EdgeDiT surpasses the teacher model with fewer parameters**: EdgeDiT-1 reduces FID from 16.23 to 12.3 using only 471M parameters (30% fewer), indicating substantial structural redundancy in DiT-XL/2.
- **30% fewer parameters, 36–46% fewer FLOPs, 1.65× on-device speedup**: Measured speedup is significant on the Samsung Galaxy S25 Ultra.
- **FwKD is indispensable**: EdgeDiT without feature-level distillation suffers severe image quality degradation; distillation provides a strong initialization for subsequent end-to-end training.
- **Sensitivity analysis of the search space**: Merging 3 blocks (instead of 2) causes a sharp quality drop; MLP ratio=1 performs poorly while ratios=2,3 yield comparable quality; hidden dim=512 and 768 produce similar quality.

## Highlights & Insights
- **The decompose–distill–search pipeline design**: The framework is highly engineering-driven and practical—decomposing an infeasible full-space search into parallelizable local distillation steps dramatically reduces search cost. This methodology is transferable to compression of other large models.
- **Disconnect between theoretical metrics and actual latency**: The paper highlights the practical issue that FLOPs cannot accurately predict NPU latency, and therefore directly incorporates on-device latency as an optimization objective—a critical consideration in engineering deployment.
- **High parallelism of proxy block distillation**: The 70 proxy blocks can be trained independently in parallel, making the entire search process efficient and scalable.

## Limitations & Future Work
- Experiments are conducted solely on DiT-XL/2 and are not extended to other diffusion Transformers such as SiT and MDT.
- Due to computational constraints, only two representative architectures—EdgeDiT-1 and EdgeDiT-6—are fully trained; additional models on the Pareto frontier remain unexplored.
- End-to-end training consists of only 100K iterations (from the 400K checkpoint), which is insufficient compared to the full training of DiT-XL/2 (7M iterations), leaving room for further FID improvement.
- Evaluation is limited to class-conditional ImageNet generation; more complex tasks such as text-to-image generation are not validated.
- The search space does not consider variations in attention head count or token sparsification techniques.

## Related Work & Insights
- **vs. MobileDiffusion**: MobileDiffusion optimizes U-Net architectures for mobile deployment, whereas EdgeDiT focuses on Transformer backbone optimization, covering a distinct technical direction.
- **vs. DiT-S/B/L**: The DiT family reduces parameters by uniformly shrinking hidden dimensions and layer counts; EdgeDiT applies heterogeneous compression across layers via search, achieving a better quality–efficiency trade-off.
- **vs. Standard Pruning/NAS**: EdgeDiT combines the advantages of knowledge distillation and Bayesian NAS, avoiding the performance cliff of pure pruning and the high training cost of pure NAS.

## Rating
- Novelty: ⭐⭐⭐ — Methodologically a combination of established techniques (KD + NAS + MOBO), but the integration is well-motivated and addresses a practical problem.
- Experimental Thoroughness: ⭐⭐⭐ — Includes measured on-device data, but only two models undergo full training on the single task of ImageNet 256.
- Writing Quality: ⭐⭐⭐⭐ — Clear and readable; the framework diagram and search space design are well explained.
- Value: ⭐⭐⭐⭐ — Provides an actionable, systematic solution for on-device deployment of Diffusion Transformers.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] PixelDiT: Pixel Diffusion Transformers for Image Generation](pixeldit_pixel_diffusion_transformers_for_image_generation.md)
- [\[CVPR 2026\] Circuit Mechanisms for Spatial Relation Generation in Diffusion Transformers](circuit_mechanisms_for_spatial_relation_generation_in_diffusion_models.md)
- [\[CVPR 2026\] Memory-Efficient Fine-Tuning Diffusion Transformers via Dynamic Patch Sampling and Block Skipping](memory-efficient_fine-tuning_diffusion_transformers_via_dynamic_patch_sampling_a.md)
- [\[CVPR 2026\] OPRO: Orthogonal Panel-Relative Operators for Panel-Aware In-Context Image Generation](opro_orthogonal_panel-relative_operators_for_panel-aware_in-context_image_genera.md)
- [\[CVPR 2026\] DiT-IC: Aligned Diffusion Transformer for Efficient Image Compression](dit-ic_aligned_diffusion_transformer_for_efficient_image_compression.md)

<!-- RELATED:END -->

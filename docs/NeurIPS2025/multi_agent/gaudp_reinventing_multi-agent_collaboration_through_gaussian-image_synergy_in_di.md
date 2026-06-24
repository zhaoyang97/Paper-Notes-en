---
title: >-
  [Paper Note] GauDP: Reinventing Multi-Agent Collaboration through Gaussian-Image Synergy in Diffusion Policies
description: >-
  [NeurIPS 2025][Multi-Agent][3D Gaussian Splatting] GauDP is proposed to enable scalable, perception-enhanced multi-agent collaborative imitation learning by constructing a globally consistent 3D Gaussian field from decentralized RGB observations of multiple agents and dynamically allocating Gaussian attributes back to each agent's local viewpoint.
tags:
  - "NeurIPS 2025"
  - "Multi-Agent"
  - "3D Gaussian Splatting"
  - "Multi-Agent Collaboration"
  - "Diffusion Policy"
  - "Imitation Learning"
  - "Robot Manipulation"
date: 2026-05-08
content_hash: 1e079654a1614ef2
---

# GauDP: Reinventing Multi-Agent Collaboration through Gaussian-Image Synergy in Diffusion Policies

**Conference**: NeurIPS 2025
**arXiv**: [2511.00998](https://arxiv.org/abs/2511.00998)  
**Code**: [Available](https://ziyeeee.github.io/gaudp.io/)  
**Area**: 3D Vision
**Keywords**: 3D Gaussian Splatting, Multi-Agent Collaboration, Diffusion Policy, Imitation Learning, Robot Manipulation

## TL;DR

GauDP is proposed to enable scalable, perception-enhanced multi-agent collaborative imitation learning by constructing a globally consistent 3D Gaussian field from decentralized RGB observations of multiple agents and dynamically allocating Gaussian attributes back to each agent's local viewpoint.

## Background & Motivation

In multi-agent embodied collaboration (e.g., industrial assembly, surgical robotics, domestic assistance), each agent must synchronize with others while completing its own subtask. Existing approaches face two core dilemmas:

**Local observations only**: Concatenating all agents' local views as input to a shared policy fails to capture the joint collaborative state, leading to desynchronized execution (e.g., one arm attempting to place food before another has opened the lid).

**Global observations only**: Provides a consistent scene representation but lacks high-resolution agent-specific information, degrading fine-grained control (grasping, placement) performance.

Naively fusing global and local signals lacks 3D structural constraints, making spatial reasoning difficult. A unified representation that **simultaneously encodes global consistency and local precision** is therefore required.

## Method

### Overall Architecture

GauDP operates in four stages:

1. **Local context extraction**: Each agent extracts local features from its own 2D observations.
2. **Global 3D Gaussian field construction**: A shared 3D Gaussian field is built from all views as global context.
3. **Global context allocation and fusion**: Global context is fused with local context and processed through an encoder.
4. **Action prediction**: A diffusion policy predicts actions via cross-attention over the per-agent fused features.

Formally, given synchronized multi-view observations $\mathcal{O} = \{\mathcal{I}_1, \dots, \mathcal{I}_N\}$, the goal is to predict future action sequences $\mathbf{a} = \{a_1, \dots, a_L\}$. The conditional policy is defined as $\pi_\Phi(\mathbf{a} | \mathcal{O}) := \pi_\Phi(\mathbf{a} | \mathcal{O}, \mathcal{G})$, where $\mathcal{G} = \mathcal{F}(\mathcal{O})$ denotes the mapping from observations to Gaussians.

### Key Designs

#### 1. Global Context Reconstruction

**Objective**: Construct a unified, view-agnostic 3D representation from multi-view 2D RGB inputs. Conventional 3DGS requires dense views, precise poses, and per-scene optimization over several minutes, making it unsuitable for rapid adaptation in embodied settings.

**Solution**: **NoPoSplat** (a feed-forward network) is adopted to reconstruct a 3D Gaussian representation directly from sparse, pose-free views, fine-tuned on robot manipulation scenes. The pipeline proceeds as follows:

- Each RGB image is independently encoded by a **shared-weight ViT encoder**.
- A **cross-view ViT decoder** fuses information across views via cross-attention layers.
- A **Gaussian parameter prediction head** estimates per-pixel 3D Gaussians: $\mathcal{G}_i = \mathcal{F}(\mathbf{x}_i)$, $\mathcal{G}_i \in \mathbb{R}^{C_\mathcal{G} \times H \times W}$.

Additional **depth supervision** is introduced: depth maps $\hat{D}$ are rendered by projecting each Gaussian into the camera coordinate frame and compared against ground-truth depth $D$. Reconstruction quality improves substantially (PSNR 17.9 → 23.4).

**Key point**: Depth and pose are used only during fine-tuning; **only RGB input is required at deployment**.

#### 2. Global Context Allocation and Pixel-Level Synergy

Broadcasting the full global context to every agent introduces irrelevant information. A **selective distribution mechanism** is proposed:

- The natural alignment between Gaussians and source pixels established during reconstruction is exploited.
- Each agent receives only the subset of Gaussians associated with its own view, which already incorporates information from other views via cross-attention.
- Selected Gaussians are transformed back into a 2D grid matching the spatial dimensions of the original image.
- The result is concatenated with local image features and fused at the pixel level through a **lightweight convolutional fusion module**.

This design ensures each agent obtains a targeted global representation while maintaining spatial consistency.

#### 3. Coordinate System Selection

Ablation studies show that using each agent's **local camera coordinate system** outperforms a unified world coordinate system—preserving agent-centric spatial relationships and avoiding cross-view alignment errors.

### Loss & Training

- **Reconstruction loss** (NoPoSplat fine-tuning stage): $\mathcal{L}_{rec} = \mathcal{L}_{rgb} + \alpha \cdot \mathcal{L}_{depth}$
- **Diffusion policy loss** (policy training stage): standard DDPM denoising loss
- Training configuration: action prediction horizon 8, observation steps 3, action execution steps 6; DDPM 100 steps; Adam, $lr=10^{-4}$, warm-up + cosine decay; 100 epochs, batch size 32, single A800 GPU.

## Key Experimental Results

### Main Results

Evaluated on the **RoboFactory** benchmark, comprising 6 collaborative manipulation tasks with 2–4 robot arms:

| Method | Lift Barrier | Place Food | Stack Cube | Align Camera | Stack Cube(4) | Take Photo | **Avg.** |
|--------|:-----------:|:----------:|:----------:|:------------:|:-------------:|:----------:|:-------:|
| DP3(XYZ+RGB) | 31% | 25% | 1% | 18% | 0% | 11% | 14.33% |
| 3D Dense Policy | 28% | 18% | 0% | 0% | 0% | 7% | 8.83% |
| DP | 9% | 12% | 6% | 3% | 0% | 0% | 5.00% |
| **GauDP** | **72%** | **15%** | **2%** | **26%** | **0%** | **3%** | **19.67%** |

3D reconstruction quality (2-view reconstruction):

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|:-----:|:-----:|:------:|
| Pretrain (NoPoSplat) | 17.918 | 0.580 | 0.492 |
| **After fine-tuning** | **23.424** | **0.779** | **0.148** |

### Ablation Study

| Configuration | Lift Barrier | Place Food | Stack Cube | Align Camera | Avg. |
|---------------|:-----------:|:----------:|:----------:|:------------:|:----:|
| Unified world coordinate | 30% | 1% | 8% | 26% | 10.83% |
| Coarse fusion (w/o prefuse) | 2% | 4% | 0% | 1% | 1.17% |
| Gaussian only (w/o Image) | 32% | 7% | 0% | 28% | 11.17% |
| Image only (w/o Gaussian) | 9% | 12% | 6% | 3% | 5.00% |
| **Full model** | **72%** | **15%** | **2%** | **26%** | **19.67%** |

Real-robot experiments: GauDP achieves 17/30, 19/30, and 27/30 on Card Box Stacking, Handover, and Grab Roller, respectively, outperforming the DP baseline on all tasks.

### Key Findings

1. GauDP surpasses all baselines using only RGB input, achieving the highest average success rate of 19.67%.
2. On Lift Barrier, it reaches 72%, substantially outperforming the second-best method DP3 (31%).
3. Removing pixel-level fusion causes performance to collapse to 1.17%, demonstrating the critical importance of fine-grained fusion.
4. Both image and Gaussian representations are indispensable: images provide appearance cues while Gaussians provide global structure.

## Highlights & Insights

- **Elegant design philosophy**: 3DGS serves as a bridge to unify local precision and global consistency without requiring additional sensor modalities.
- **Naturally scalable**: The flexibility of the Gaussian representation requires no architectural changes as the number of agents increases.
- **Self-supervised reconstruction**: The same multi-view data used to train the diffusion policy is leveraged for 3DGS fine-tuning, requiring no additional data.
- **No pose or depth at inference**: Only RGB input is needed at deployment.

## Limitations & Future Work

1. Success rates on high-precision tasks such as Stack Cube remain very low (2%), indicating significant room for improvement in fine-grained manipulation.
2. Overall success rates under 3–4 arm configurations are low, suggesting that high-complexity collaboration remains challenging.
3. Training is slightly more expensive (6.5 vs. 4.8/2.5 GPU hours) and inference is marginally slower (1.28 vs. 1.49 FPS).
4. Integration with VLA models and the application of Gaussians as world models in dynamic scenes remain unexplored.

## Related Work & Insights

- **NoPoSplat**: The core dependency, enabling feed-forward 3DGS reconstruction from sparse, pose-free views.
- **3D Diffusion Policy (DP3)**: The primary baseline, a diffusion policy with point cloud input.
- **RoboFactory**: A multi-arm collaborative manipulation benchmark built via automated data collection.
- Insight: 3DGS as an intermediate representation for aggregating multi-view information is broadly applicable to multi-agent perception–decision tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ Combining 3DGS with diffusion policies for multi-agent settings is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive ablations with both simulation and real-robot experiments.
- Writing Quality: ⭐⭐⭐⭐ Framework design is clear and motivation is well articulated.
- Value: ⭐⭐⭐⭐ Opens a new direction for visual representations in multi-agent collaboration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] GETReason: Enhancing Image Context Extraction through Hierarchical Multi-Agent Reasoning](../../ACL2025/multi_agent/getreason_enhancing_image_context_extraction_through_hierarchical_multi-agent_re.md)
- [\[NeurIPS 2025\] Multi-Agent Collaboration via Evolving Orchestration](multi-agent_collaboration_via_evolving_orchestration.md)
- [\[ICML 2026\] Searching for Synergy in Shared Workspace Human-AI Collaboration](../../ICML2026/multi_agent/searching_for_synergy_in_shared_workspace_human-ai_collaboration.md)
- [\[NeurIPS 2025\] 3D-Agent: Tri-Modal Multi-Agent Collaboration for Scalable 3D Object Annotation](3d-agenttri-modal_multi-agent_collaboration_for_scalable_3d_object_annotation.md)
- [\[ICML 2026\] RADAR: Redundancy-Aware Diffusion for Multi-Agent Communication Structure Generation](../../ICML2026/multi_agent/radar_redundancy-aware_diffusion_for_multi-agent_communication_structure_generat.md)

</div>

<!-- RELATED:END -->

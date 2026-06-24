---
title: >-
  [Paper Note] FreeScene: Mixed Graph Diffusion for 3D Scene Synthesis from Free Prompts
description: >-
  [CVPR 2025][3D Vision][Indoor Scene Synthesis] FreeScene proposes a user-friendly indoor scene synthesis framework. It utilizes a VLM-driven Graph Designer to convert free-form text/image inputs into scene graphs, and then uses a Mixed Graph Diffusion Transformer (MG-DiT) to perform graph-aware denoising in a hybrid continuous-discrete space. It unifiedly supports multiple tasks such as text-to-scene and graph-to-scene, outperforming existing methods in both generation qualit…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Indoor Scene Synthesis"
  - "Graph Diffusion Models"
  - "Scene Graphs"
  - "VLM"
  - "Controllable Generation"
date: 2026-05-08
content_hash: 40786862d8b7fad7
---

# FreeScene: Mixed Graph Diffusion for 3D Scene Synthesis from Free Prompts

**Conference**: CVPR 2025  
**arXiv**: [2506.02781](https://arxiv.org/abs/2506.02781)  
**Code**: [https://cangmushui.github.io/FreeScene-io/](https://cangmushui.github.io/FreeScene-io/) (Project Page)  
**Area**: 3D Vision  
**Keywords**: Indoor Scene Synthesis, Graph Diffusion Models, Scene Graphs, VLM, Controllable Generation

## TL;DR
FreeScene proposes a user-friendly indoor scene synthesis framework. It utilizes a VLM-driven Graph Designer to convert free-form text/image inputs into scene graphs, and then uses a Mixed Graph Diffusion Transformer (MG-DiT) to perform graph-aware denoising in a hybrid continuous-discrete space. It unifiedly supports multiple tasks such as text-to-scene and graph-to-scene, outperforming existing methods in both generation quality and controllability.

## Background & Motivation

**Background**: Indoor scene synthesis is a crucial task in fields such as game design, VR/AR, and robotics. Recently, diffusion models (e.g., DiffuScene) have been adopted for scene layout generation, producing plausible indoor scenes through iterative denoising.

**Limitations of Prior Work**: Existing methods face a dilemma in terms of controllability. Text-based methods (e.g., DiffuScene) are easy to use but only allow for coarse-grained control, failing to guarantee precise alignment between text descriptions and generation results. In contrast, graph-based methods (e.g., InstructScene) offer better fine-grained control but require users to manually design tedious scene graphs, imposing a high barrier to entry.

**Key Challenge**: The trade-off between user convenience and precise controllability: text inputs are convenient but imprecise, whereas graph inputs are precise but inconvenient. Moreover, existing methods typically handle only a single condition type, lacking unified multi-task capabilities.

**Goal**: (1) How to allow users to precisely control scene generation using free-form text and/or images? (2) How to unifiedly support multiple tasks such as text-to-scene and graph-to-scene with a single model?

**Key Insight**: The authors observe that scene graphs serve as a natural intermediate representation—they can be extracted from free-form inputs and precisely guide scene generation. By leveraging the multimodal understanding capabilities of VLMs, scene graphs can be automatically inferred from user inputs. Subsequently, a mixed diffusion model is utilized to jointly model discrete (categories, relations) and continuous (positions, sizes) attributes.

**Core Idea**: Leveraging a VLM to automatically convert free-form inputs into scene graph priors, which are then used in a mixed graph diffusion Transformer to achieve unified multi-task controllable scene generation.

## Method

### Overall Architecture
The pipeline of FreeScene consists of two stages: (1) **Graph Designer**: It receives user texts and/or images (e.g., top-down views, photos, sketches) and performs multi-step reasoning via a VLM (GPT-4o) to extract object categories and spatial relationships, building a partial scene graph prior; (2) **MG-DiT**: Conditioned on the partial scene graph and a text description, it performs mixed diffusion denoising on both discrete attributes (object categories, fVQ-VAE feature indices, relationship types) and continuous attributes (sizes, locations, orientations) of the objects in the scene to generate a complete indoor layout. Finally, the most matching furniture models are retrieved from the 3D-FUTURE dataset using OpenCLIP feature matching.

### Key Designs

1. **VLM-based Graph Designer**:

    - **Function**: Automatically extract scene graphs (object lists + relationship triplets + text descriptions) from free-form multimodal inputs (texts and images).
    - **Mechanism**: A one-shot Chain-of-Thought (CoT) prompt template is designed to guide the VLM through a four-step reasoning process: perspective calibration $\rightarrow$ object extraction $\rightarrow$ DFS traversal (depth-first search of all objects starting from critical root nodes to avoid missing relationships) $\rightarrow$ relationship extraction. The output structured data is parsed into a graph representation using regular expressions.
    - **Design Motivation**: Directly prompting the VLM to generate relationships tends to cause contradictions or miss critical relations. The DFS traversal strategy ensures hierarchical coverage of relationships among all objects, and CoT improves relationship accuracy by over 40% compared to a simple one-shot baseline (e.g., from 44.32% to 74.63% in diagram scenarios).

2. **Mixed Graph Diffusion Transformer (MG-DiT)**:

    - **Function**: Jointly denoise discrete variables (object category $c$, fVQ-VAE feature index $v$, relation $e$) and continuous variables (size $s$, position $t$, orientation $r$) to generate a complete scene layout.
    - **Mechanism**: The scene is represented as a mixed attribute graph. Continuous variables are perturbed with Gaussian noise using DDPM, while discrete variables are perturbed using D3PM transition matrices with a [MASK] state. Developed on a DiT architecture, the node features consist of embedded and concatenated object attributes (including sinusoidal positional encodings), and the edge features consist of relation embeddings. The interaction between nodes and edges is implemented through a FiLM mechanism: $\text{FiLM}(sim, e) = \gamma(e) \cdot \frac{sim - \mu}{\sigma} + \beta(e)$. Cross-attention is also integrated to handle text conditions.
    - **Design Motivation**: In contrast to the two-stage approach of InstructScene (first text-to-graph, then graph-to-scene), MG-DiT uses a single model to jointly predict both the graph structure and layout attributes. This joint modeling forces the model to learn superior global scene features and inter-object relationships during graph prediction, boosting both generation quality and controllability.

3. **Constrained Sampling (Unified Multi-Tasking via Constrained Sampling)**:

    - **Function**: Support various downstream tasks in a zero-shot manner with a single model by fixing different subsets of variables during the denoising process.
    - **Mechanism**: At each step of sampling, the corresponding variables are fixed according to the task type. For example, text-to-scene allows all variables to denoise normally; graph-to-scene fixes/partially fixes categories and relations; re-arrangement fixes categories, sizes, and feature indices, denoising only positions and orientations; completion fixes all attributes of existing objects; stylization fixes all variables except feature indices.
    - **Design Motivation**: To avoid training separate models for each task. Through constrained-sampling, the partial graph priors extracted by the Graph Designer can seamlessly serve as conditional inputs for MG-DiT, achieving comprehensive coarse-to-fine control.

### Loss & Training
The total loss is the sum of the continuous and discrete parts: $\mathcal{L} = \mathcal{L}_b + \mathcal{L}_z$. The continuous part $\mathcal{L}_b$ is the standard noise-prediction MSE loss; the discrete part $\mathcal{L}_z$ is the KL divergence between the predicted posterior distribution and the ground-truth posterior distribution. During training, the timestep $t$ is sampled from a uniform distribution $\mathcal{U}(1,T)$, while continuous variables are perturbed with Gaussian noise and discrete variables are perturbed with state transition matrices.

## Key Experimental Results

### Main Results

| Task/Room | Method | FID↓ | FID_CLIP↓ | KID↓ | SCA% (→50%) | iRecall%↑ |
|-----------|------|------|-----------|------|-------------|-----------|
| Text-to-Scene/Bedroom | InstructScene | 114.86 | 6.52 | 0.68 | 56.37 | 72.71 |
| Text-to-Scene/Bedroom | **Ours+GD** | **108** | **6.07** | **0.21** | **53.16** | **81.40** |
| Text-to-Scene/Livingroom | InstructScene | 111.52 | 5.91 | 8.65 | 55.32 | 57.21 |
| Text-to-Scene/Livingroom | **Ours+GD** | **108.22** | **5.23** | **3.87** | **54.05** | **71.81** |
| Graph-to-Scene/Bedroom | InstructScene | 101.86 | 5.66 | 0.13 | 53.68 | 88.84 |
| Graph-to-Scene/Bedroom | **Ours** | **98.31** | **5.58** | **0.12** | **52.34** | **89.37** |

### Ablation Study

| Configuration | Object iRecall% | Rel Acc% |
|------|----------------|----------|
| Graph Designer w/ CoT (Image) | **85.23** | **77.56** |
| Graph Designer w/o CoT (Image) | 72.07 | 34.65 |
| Graph Designer w/ CoT (Diagram) | **91.22** | **74.63** |
| Graph Designer w/o CoT (Diagram) | 88.13 | 44.32 |
| Graph Designer w/ CoT (Text) | **98.56** | **89.45** |
| Graph Designer w/o CoT (Text) | 95.56 | 85.60 |

### Key Findings
- **The CoT strategy in Graph Designer** significantly improves relationship accuracy; specifically, in image input scenarios, Rel Acc increases from 34.65% to 77.56%, where DFS traversal plays a key role.
- **The mixed diffusion of MG-DiT** remarkably enhances controllability: iRecall on the Text-to-Scene task improves by approximately 9 percentage points compared to InstructScene.
- FreeScene outperforms all baseline methods in various zero-shot applications, including Re-arrangement, Completion, and Unconditioned generation.
- Ours+GD, which incorporates the preprocessing of Graph Designer, further outperforms direct text-to-scene generation (Ours) across all metrics.

## Highlights & Insights
- **The constrained-sampling design for unified multi-tasking** is highly elegant—a single model can support 5+ scene synthesis tasks by freezing different subsets of variables, eliminating the need for additional training.
- **The CoT strategy with DFS traversal** effectively resolves the issue of VLMs missing relationships during graph extraction; this technique can be transferred to any task requiring the extraction of structured relationships from free-form text or images.
- The **mixed diffusion (continuous + discrete)** DiT architecture holds general applicability and can be adopted for any task involving the simultaneous modeling of continuous attributes and discrete labels.

## Limitations & Future Work
- The Graph Designer is limited by the inherent capabilities of VLMs, leading to less accurate extraction in complex scenes containing a large number of objects.
- MG-DiT cannot precisely control the exact positions and orientations of objects, which can occasionally generate implausible furniture arrangements.
- The limited dataset scale (3D-FRONT contains only 6,813 houses) may potentially lead to overfitting.
- It only supports indoor scenes (bedrooms, living rooms, dining rooms) and has not been extended to outdoor or more complex environments.
- Incorporating physical constraints (e.g., affordance) could be considered to improve practicality.

## Related Work & Insights
- **vs InstructScene**: InstructScene utilizes two separate models to perform text-to-graph and graph-to-scene, respectively, whereas FreeScene employs the unified MG-DiT model, avoiding intermediate graph information loss.
- **vs DiffuScene**: DiffuScene struggles to capture global features with its U-Net 1D convolutions, while FreeScene's DiT + FiLM architecture is better suited to modeling relationship interdependencies among objects.
- **vs LLM-based Methods**: Methods like SceneGPT directly generate scene configurations using LLMs but lack fine-grained control; FreeScene achieves precise relationship constraints via graph representations.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of mixed graph diffusion and a VLM graph designer is intuitive yet effective; using constrained sampling to unify multiple tasks is a highlight.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple tasks and room types, accompanied by detailed ablation studies and comparisons.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and complete mathematical derivations.
- Value: ⭐⭐⭐⭐ Provides a practical and unified framework for controllable indoor scene synthesis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] LT3SD: Latent Trees for 3D Scene Diffusion](lt3sd_latent_trees_for_3d_scene_diffusion.md)
- [\[CVPR 2025\] DiffPortrait360: Consistent Portrait Diffusion for 360° View Synthesis](diffportrait360_consistent_portrait_diffusion_for_360_view_synthesis.md)
- [\[CVPR 2025\] Novel View Synthesis with Pixel-Space Diffusion Models](novel_view_synthesis_with_pixel-space_diffusion_models.md)
- [\[CVPR 2025\] SceneFactor: Factored Latent 3D Diffusion for Controllable 3D Scene Generation](scenefactor_factored_latent_3d_diffusion_for_controllable_3d_scene_generation.md)
- [\[CVPR 2025\] Sharp-It: A Multi-view to Multi-view Diffusion Model for 3D Synthesis and Manipulation](sharp-it_a_multi-view_to_multi-view_diffusion_model_for_3d_synthesis_and_manipul.md)

</div>

<!-- RELATED:END -->

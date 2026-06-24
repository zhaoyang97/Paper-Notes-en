---
title: >-
  [Paper Note] EchoScene: Indoor Scene Generation via Information Echo over Scene Graph Diffusion
description: >-
  [ECCV 2024][Image Generation][3D indoor scene generation] EchoScene is proposed, a 3D indoor scene generation method based on a dual-branch diffusion model. It achieves collaborative information exchange among multiple denoising processes during the scene graph diffusion process through an "Information Echo" mechanism, generating globally consistent and interactively controllable scenes.
tags:
  - "ECCV 2024"
  - "Image Generation"
  - "3D indoor scene generation"
  - "scene graph diffusion"
  - "information echo"
  - "dual-branch diffusion model"
  - "controllable generation"
date: 2026-05-08
content_hash: 47f97e331ad9d512
---

# EchoScene: Indoor Scene Generation via Information Echo over Scene Graph Diffusion

**Conference**: ECCV 2024  
**arXiv**: [2405.00915](https://arxiv.org/abs/2405.00915)  
**Code**: [GitHub](https://github.com/ymxlzgy/echoscene)  
**Area**: Image Generation  
**Keywords**: 3D indoor scene generation, scene graph diffusion, information echo, dual-branch diffusion model, controllable generation

## TL;DR

EchoScene is proposed, a 3D indoor scene generation method based on a dual-branch diffusion model. It achieves collaborative information exchange among multiple denoising processes during the scene graph diffusion process through an "Information Echo" mechanism, generating globally consistent and interactively controllable scenes.

## Background & Motivation

Controllable Scene Generation (CSG) has crucial applications in robotics, VR/AR, autonomous driving, and other fields. Recently, combining scene graphs with diffusion models has become a research hotspot: scene graphs capture scene structures in a compact manner and allow users to dynamically modify generated scenes by editing the graph.

Existing methods face two core challenges:

**1. Dynamic Graph Adaptability**: The number of nodes in a scene graph is not fixed, and users can add or delete nodes/edges at any time. Existing solutions either simplify the graph structure, thereby losing edge information (DiffuScene only retains the node set), or convert nodes and edges into tokens (InstructScene), but the number of tokens grows exponentially with the number of nodes ($Q \cdot P!$), making large graphs infeasible.

**2. Global Consistency**: Although CommonScenes assigns an independent denoising process to each node to adapt to the dynamics of the graph, these processes are isolated from each other and lack awareness of the global shape state, leading to inconsistent styles among objects (e.g., chairs in the same scene having different styles). Its layout generation relies on VAE and GAN, making joint training synchronized with difficulty.

The core idea of EchoScene is to assign an independent denoising process to each node in the scene graph (to ensure controllability) while enabling all processes to exchange intermediate denoising states at each step through the "Information Echo" mechanism (to ensure global consistency).

## Method

### Overall Architecture

EchoScene consists of the following components:
1. **Graph Preprocessing**: Contextual graph encoding + graph manipulation (adding/deleting nodes, modifying edges)
2. **Layout Branch**: Generates object bounding box parameters based on a diffusion model
3. **Shape Branch**: Generates 3D object shapes based on a latent diffusion model
4. **Information Echo Scheme**: Realizes information exchange among denoising processes within both branches

The two branches undergo joint end-to-end training, with the loss function: $\mathcal{L} = \lambda_1 \mathcal{L}_{\text{layout}} + \lambda_2 \mathcal{L}_{\text{shape}}$

### Key Designs

**1. Contextual Graph and Graph Encoding**

In the scene graph $\mathcal{G} = \{\mathcal{V}, \mathcal{E}\}$:
- Node $v_i := \{p_i, o_i\}$ contains a CLIP semantic anchor and a learnable vector
- Edge $e_{i \to j} := \{p_{i \to j}, \tau_{i \to j}\}$ contains a triplet CLIP embedding and a learnable vector

A Triplet-GCN encoder $E_r$ is used to encode the contextual graph, obtaining latent node features $\mathcal{V}_\mathcal{Z} = \{v_i^z\}$ that incorporate relational information.

A graph manipulator (another Triplet-GCN) performs node addition and relation modification in the latent space to simulate user interaction. After manipulation, the number of nodes can increase from $N$ to $M \geq N$.

**2. Information Echo Scheme (Core Innovation)**

This is the key mechanism connecting independence and globality. At each denoising step:

**Step 1 - Information Assembly**: Each denoising process concatenates the current denoised data $\mathbf{d}_t^i$ with the node feature $v_i^z$ and time embedding $\pi(t)$ to construct a new set of nodes:

$$\mathcal{V}_{\mathcal{D}_t} := \{f(\mathbf{d}_t^i, v_i^z, \pi(t)) | i = 1, \ldots, M\}$$

**Step 2 - Information Exchange**: The new set of nodes $\mathcal{V}_{\mathcal{D}_t}$ and the edges $\mathcal{E}$ of the original graph form a temporary graph $\mathcal{G}_{\mathcal{D}_t}$, which is fed into the information exchange unit $U$ (based on Triplet-GCN) to aggregate all node information according to the graph structure.

**Step 3 - Conditional Echo**: The aggregated features $\mathcal{C}_{\mathcal{D}_t} = U(\mathcal{G}_{\mathcal{D}_t})$ are fed back to each denoiser as a condition signal.

One round of "transmission and reception" constitutes an "Information Echo". The denoising formula becomes:

$$\mathbf{d}_{t-1}^i = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{d}_t^i - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \varepsilon_\theta(\mathbf{d}_t^i, \pi(t), \mathcal{C}_{\mathcal{D}_t})\right) + \sigma_t \mathbf{z}$$

Key design choices:
- The denoiser $\varepsilon_\theta$ shares **weights** across all processes, introducing no additional parameter costs.
- Information exchange is executed at **each denoising step**, continuously introducing global constraints.
- Different branches employ different exchange units: $U_l$ (layout) and $U_s$ (shape).

**3. Layout Branch**

Layouts are parameterized as 8-dimensional vectors: $\mathbf{b}_0^i = \{x, y, z, l, h, w, \sin\theta, \cos\theta\}$

**Layout Echo**: Each node exchanges current denoised bounding box parameters at each step. This is crucial for layout generation because spatial constraints (e.g., "left of", "close to") require awareness of other objects' location states.

Training objective:
$$\mathcal{L}_{\text{layout}} = \mathbb{E}_{\mathbf{B}, \gamma \sim \mathcal{N}(0,1), t}[\|\gamma - \gamma_\theta(\mathbf{B}_t, \pi(t), U_l(\mathcal{G}_{\mathcal{B}_t}))\|_2^2]$$

**4. Shape Branch**

The bottleneck latent codes of a pre-trained VQ-VAE are used as targets for the shape LDM.

**Isolation Problem**: Although each denoising process can independently generate shapes conditioned on semantic/relational embeddings, they lack awareness of other objects' shape appearances, resulting in style inconsistency (e.g., tables and chairs in the same dining room mismatching in style).

**Shape Echo**: The denoised shape codes $\mathbf{X}_t$ are processed into $\mathbf{S}_t$ via 3D convolution + flattening, and then participate in graph information exchange. As the timestep approaches 0, the shape information becomes increasingly clear, making the exchange more meaningful.

### Loss & Training

- Joint training of dual branches: $\mathcal{L} = \lambda_1 \mathcal{L}_{\text{layout}} + \lambda_2 \mathcal{L}_{\text{shape}}$, where $\lambda_1 = \lambda_2 = 1.0$.
- Both layout and shape branches use 1000-step diffusion processes.
- AdamW optimizer is used with an initial learning rate of $1 \times 10^{-4}$.
- Trained and evaluated on a single NVIDIA A40 (40GB) GPU.
- Dataset: SG-FRONT (scene graph annotations based on 3D-FRONT, with 15 relationship types, 45K object instances, and three room types).

## Key Experimental Results

### Main Results

Scene generation realism (FID ↓ / FID_CLIP ↓ / KID×0.001 ↓):

| Method | Shape Representation | Bedroom FID | Living Room FID | Dining Room FID |
|------|----------|----------|----------|----------|
| 3D-SLN | Retrieval | 57.90 | 77.82 | 69.13 |
| CommonLayout | Retrieval | 52.69 | 76.52 | 65.10 |
| DiffuScene | Retrieval | 52.02 | 81.61 | 65.90 |
| InstructScene | Retrieval | 45.40 | 75.83 | 61.56 |
| **EchoLayout (Ours)** | **Retrieval** | **46.53** | **75.54** | **59.66** |
| Graph-to-3D | Generation | 63.72 | 82.96 | 72.51 |
| CommonScenes | Generation | 57.68 | 80.99 | 65.71 |
| **EchoScene (Ours)** | **Generation** | **48.85** | **75.95** | **62.85** |

Improvement relative to CommonScenes:
- Bedroom FID improved by 15% (57.68 → 48.85)
- Bedroom FID_CLIP improved by 12%
- Bedroom KID improved by 73%

### Ablation Study

The effectiveness of the information echo scheme is validated by shape consistency (Chamfer Distance ↓):

EchoScene's shape echo significantly improves the consistency problem among objects. For example:
- In dining room scenes generated by CommonScenes, tables and chairs might mismatch in style.
- Chairs generated by Graph-to-3D might each exhibit different styles.
- EchoScene ensures that dining tables and chairs match in complete sets through the shape echo.

Graph constraint satisfaction (after scene graph manipulation):
- After node addition and relation modification, EchoLayout/EchoScene maintain better constraint satisfaction across most spatial relations (left/right, bigger/smaller, close by, etc.).
- 3D-SLN, CommonScenes, and Graph-to-3D tend to lose spatial constraints after graph manipulation.

### Key Findings

1. **Information Echo is Key to Global Consistency**: Without the echo mechanism, each denoising process runs independently, resulting in inconsistent styles among generated objects.
2. **Diffusion is Better than VAE/GAN for Layouts**: VAEs struggle to learn angle parameters, causing irregular object orientations; diffusion layout generation yields more organized results.
3. **Significant Improvement remains even with Only the Layout Branch**: Combined with an external shape generator (e.g., SDFusion), EchoLayout still outperforms equivalent combinations like CommonLayout.
4. **Generated Scenes are Compatible with Off-the-shelf Texture Generators** (e.g., SceneTex), enabling direct outputs of textured, photorealistic renderings.

## Highlights & Insights

1. **Paradigm Innovation of "One Denoising Process Per Node + Information Echo"**: Ingeniously resolves the conflict between controllability and consistency of diffusion on dynamic graphs, establishing a new paradigm for graph-based generative models.
2. **Dual-Branch Pure Diffusion Architecture**: Departs from the hybrid VAE+GAN+LDM architecture of CommonScenes, facilitating a simpler, synchronous training process.
3. **Echo equals a Social Network of Denoising Processes**: In each denoising step, every process can "see" the current state of other objects, akin to a group working collaboratively to arrange a room while constantly communicating.
4. **Compatible with Downstream Texture Generation**: The generated scenes possess high-enough geometric quality to directly apply textures using off-the-shelf methods like SceneTex.

## Limitations & Future Work

1. Shape generation is based on the VQ-VAE latent space, which has limited resolution, potentially resulting in insufficient details for complex objects.
2. The information exchange unit is based on Triplet-GCN, which may exhibit lower efficiency on very large scene graphs.
3. Evaluated only on the SG-FRONT dataset, featuring limited scene types (bedroom, living room, dining room).
4. Graph manipulation still relies on manual editing of the scene graph, lacking a natural language interaction interface.
5. Diversity control in conditional generation (e.g., generating different styles of scenes given the same scene graph) has not been explored.

## Related Work & Insights

- **CommonScenes**: First to propose scene-graph-conditioned scene generation, but the architecture of VAE layouts + isolated LDM shapes restricted consistency. EchoScene comprehensively optimizes on top of this.
- **DiffuScene / InstructScene**: Simplify graph structures or use transformer tokenization, which offers limited scalability. EchoScene retains full utilization of the graph structure via per-node diffusion + echo.
- **Score Distillation (SDS)**: While SDS-based 3D generation methods can create realistic assets, they struggle with multi-object relationships. EchoScene fills this gap.
- **Insight**: The information echo scheme can be generalized to any scenario requiring collaboration among multiple diffusion processes (e.g., molecule generation, urban planning).

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The information echo scheme is a highly original graph diffusion paradigm.
- Technical Depth: ⭐⭐⭐⭐ — The dual-branch design is rigorous, and the mathematical formulation of the echo mechanism is clear.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multiple baseline comparisons + graph manipulation robustness + consistency analysis.
- Practical Value: ⭐⭐⭐⭐ — Controllable 3D scene generation has direct applications in VR/AR, gaming, etc.
- Overall Recommendation: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Mutual Learning for Acoustic Matching and Dereverberation via Visual Scene-driven Diffusion](mutual_learning_for_acoustic_matching_and_dereverberation_via_visual_scene-drive.md)
- [\[ECCV 2024\] The Fabrication of Reality and Fantasy: Scene Generation with LLM-Assisted Prompt Interpretation](the_fabrication_of_reality_and_fantasy_scene_generation_with_llm-assisted_prompt.md)
- [\[ICLR 2026\] Generate Any Scene: Scene Graph Driven Data Synthesis for Visual Generation Training](../../ICLR2026/image_generation/generate_any_scene_scene_graph_driven_data_synthesis_for_visual_generation_train.md)
- [\[ECCV 2024\] DCDM: Diffusion-Conditioned-Diffusion Model for Scene Text Image Super-Resolution](dcdm_diffusion-conditioned-diffusion_model_for_scene_text_image_super-resolution.md)
- [\[ICCV 2025\] Lay-Your-Scene: Natural Scene Layout Generation with Diffusion Transformers](../../ICCV2025/image_generation/lay-your-scene_natural_scene_layout_generation_with_diffusion_transformers.md)

</div>

<!-- RELATED:END -->

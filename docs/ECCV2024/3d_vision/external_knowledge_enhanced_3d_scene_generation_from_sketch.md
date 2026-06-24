---
title: >-
  [Paper Note] External Knowledge Enhanced 3D Scene Generation from Sketch
description: >-
  [ECCV 2024][3D Vision][3D Scene Generation] Proposes the SEK framework, which integrates freehand sketches and an external object relation knowledge base as conditions for a diffusion model. Through knowledge-enhanced graph reasoning and spectrum filtering, it end-to-end simultaneously generates the layout and object geometry of 3D indoor scenes.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "3D Scene Generation"
  - "Sketch-guided"
  - "External Knowledge Base"
  - "Diffusion Models"
  - "Knowledge Graph Reasoning"
date: 2026-05-08
content_hash: f98e64d6cbbe13b7
---

# External Knowledge Enhanced 3D Scene Generation from Sketch

**Conference**: ECCV 2024  
**arXiv**: [2403.14121](https://arxiv.org/abs/2403.14121)  
**Code**: None  
**Area**: 3D Vision  
**Keywords**: 3D Scene Generation, Sketch-guided, External Knowledge Base, Diffusion Models, Knowledge Graph Reasoning

## TL;DR

Proposes the SEK framework, which integrates freehand sketches and an external object relation knowledge base as conditions for a diffusion model. Through knowledge-enhanced graph reasoning and spectrum filtering, it end-to-end simultaneously generates the layout and object geometry of 3D indoor scenes.

## Background & Motivation

**Background**: 3D scene generation has seen a growing demand in fields such as game development, film production, AR/VR, and interior design. Existing approaches primarily rely on conditions like images, text, scene graphs, or room layouts to guide the generation process.

**Limitations of Prior Work**: (1) Image-based methods are constrained by 2D-3D consistency limits, resulting in a lack of diversity; (2) Sketch-based methods mostly focus on single 3D object generation and cannot handle complex scene-level generation; (3) Existing scene generation methods either only generate layouts and retrieve objects from a database (lacking diversity) or rely on simple, manually-defined object relationships (such as the manual hierarchical graphs in GRAINS).

**Key Challenge**: Freehand sketches are inherently sparse and ambiguous, failing to provide sufficient information to determine the precise shapes and spatial alignments of all objects. Furthermore, users may expect the scene to include objects not depicted in the sketch (invisible objects), an implicit requirement that prior methods struggle to address.

**Goal**: To generate complete 3D indoor scenes with plausible layouts, detailed object geometry, and credible relationships, starting from sparse freehand sketches.

**Key Insight**: Introduce an external knowledge base to provide prior information on object relationships, compensating for the ambiguity of the sketch, while leveraging a diffusion model to simultaneously generate both the layout and object shapes.

**Core Idea**: Construct an object relationship knowledge base and fuse knowledge-enhanced graph reasoning features with sketch features to provide rich generative guidance for the conditional diffusion model.

## Method

### Overall Architecture

The pipeline of SEK consists of four components: (1) **Scene Matrix Representation**: Encoding the 3D scene into a matrix $\mathcal{O} \in \mathbb{R}^{D \times M}$, where each column corresponds to the pose and shape latent code of an object; (2) **External Knowledge Base Construction**: Extracting five types of object relationships from the training data and calculating their probabilities; (3) **Knowledge-Enhanced Sketch Guidance**: Extracting sketch features using ViT and fusing them with knowledge graph reasoning features to form the diffusion conditioning; (4) **3D Scene Denoising Transformer**: Iteratively denoising to generate the scene matrix, enhanced by a spectrum filter.

### Key Designs

1. **Matrix Conversion**:

    - **Function**: Uniformly encode the 3D scene into a fixed-dimension matrix so that the diffusion process can operate in the matrix space.
    - **Mechanism**: Each object is represented as $\mathbf{o}_i = [\mathcal{G}_i, \mathcal{F}_i] \in \mathbb{R}^{D \times 1}$, where $\mathcal{G}_i = [\alpha_i, \mathbf{s}_i, \mathbf{t}_i]$ includes the rotation angle (parameterized by 2D sine and cosine), 3D bounding box size, and translation; $\mathcal{F}_i$ is the shape latent code trained via DeepSDF. When the number of objects in the scene varies, it is zero-padded to a fixed count $M$.
    - **Design Motivation**: The matrix representation allows the layout and shape of the scene to be generated simultaneously in a unified space, bypassing the two-stage pipeline of generating layout first and then retrieving objects.

2. **Knowledge Base**:

    - **Function**: Store rich relationship priors among objects as $KB = (\mathcal{V}, \mathcal{R}, p)$ to constrain the plausibility of the generated scenes.
    - **Mechanism**: DBSCAN is used to cluster objects in the training scenes to extract five types of relationships: (a) **Attachment** (minimum distance between adjacent objects is less than voxel length); (b) **Alignment** (bounding box planes are coplanar); (c) **Dependent** (same group but not attachment/alignment); (d) **Parallel Collinearity** (horizontal axes of bounding boxes are parallel for objects in different groups); (e) **Co-occurrence** (co-occurrence of objects from different groups in the same scene). The relationship probability is normalized via a sigmoid function: $p_{ij} = 1/(1 + e^{-10 \cdot n_{ij}^{\mathcal{R}} / max(n)^{\mathcal{R}}})$.
    - **Design Motivation**: Sketches are inherently sparse and ambiguous. The knowledge base can (1) augment the description of visible objects (cross-validation) and (2) infer invisible objects (e.g., observing a 'sofa' suggests the potential need for a 'table').

3. **Knowledge-enhanced Graph Reasoning (KeGR)**:

    - **Function**: Integrate external knowledge with target object entities and execute graph convolutional reasoning to obtain rich conditional features.
    - **Mechanism**: GloVe is used to initialize the object node features $h_i \in \mathbb{R}^{1 \times D_\omega}$. A fully connected subgraph $G_i^E = (h_i^E, \mathcal{E}_i^E, \mathcal{P}_i^E)$ is constructed. Multi-step graph convolutional reasoning is performed as: $H_i^{E(j)} = \delta(A_i^E H_i^{E(j-1)} W^{E(j)})$. Features of all relationship types are fused utilizing a $1 \times 1$ convolution to obtain the graph features $H^G$. The final conditional feature is the concatenation of the sketch ViT features $H^S$ and graph features $H^G$: $c = [H^S, H^G]$.
    - **Design Motivation**: Graph convolutions effectively propagate relational information between objects, enabling each object node to perceive its relational context with other objects.

4. **Spectrum-Filter**:

    - **Function**: Suppress the interference of zero-padding values on valid object features during the denoising Transformer.
    - **Mechanism**: Observing that zero-padding exhibits a low-frequency variance distribution compared to valid object representations, a high-pass filter is designed to suppress low-frequency padding components in the spectral domain: $\text{EF}(\mathcal{O}_I, B) = \mathcal{O}_I + e^{-t} \Theta_{IFFT}(\text{Conv}(\sigma(\mathcal{O}_I, B) \circledast \Theta_{FFT}(\mathcal{O}_I)))$, where $e^{-t}$ decays the filtering strength alongside the time steps (since the low-frequency signature of padding fades as noise increases).
    - **Design Motivation**: Since no padding masks are available during inference to filter out meaningless generated values, the spectrum filter offers an adaptive way to boost valid object features.

### Loss & Training

Scene diffusion loss: $\mathcal{L}_{sce} = \mathbb{E}_{c,t,\epsilon,\mathcal{O}_0}[\|\epsilon - \epsilon_\theta(c, t, \mathcal{O}_t)\|^2]$, where $\epsilon \sim \mathcal{N}(0, \mathbf{I})$. Standard DDPM training strategy is employed, and the denoising process iteratively generates the scene matrix via a reverse Markov chain. The scene completion task utilizes DDIM inversion.

## Key Experimental Results

### Main Results

Scene generation on the 3D-FRONT dataset:

| Method | Bedroom FID↓ | Dining FID↓ | Living FID↓ | Dining CKL↓ |
|------|-------------|-------------|-------------|-------------|
| ATISS | 18.60 | 38.66 | 40.83 | 0.64 |
| DiffuScene | 18.29 | 32.60 | 36.18 | 0.22 |
| Graph-to-3D | 61.24 | 54.11 | 41.13 | 1.68 |
| **SEK (Ours)** | **15.21** | **25.46** | **31.24** | **0.16** |

Compared to the strongest competitor DiffuScene, on the Dining room category, the FID is reduced by 17.41% and the CKL is reduced by 37.18%.

Scene completion task:

| Method | Bedroom FID↓ | Dining FID↓ | Dining KID↓ |
|------|-------------|-------------|-------------|
| DiffuScene | 27.32 | 40.99 | 6.31 |
| **SEK (Ours)** | **21.84** | **33.03** | **5.18** |

The average FID improves by 19.12%, and the KID improves by 20.06%.

### Ablation Study

| Configuration (Dining room) | FID↓ | SCA% | CKL↓ |
|-----|------|------|------|
| w/o Knowledge / ViT Sketch + SF | 33.29 | 56.81 | 0.85 |
| ResNet50 Sketch + Knowledge + SF | 24.68 | 52.70 | 0.18 |
| ViT Sketch + Knowledge, w/o SF | 25.83 | 54.19 | 0.37 |
| **ViT Sketch + Knowledge + SF (Full)** | **23.97** | **51.97** | **0.16** |

### Key Findings

- **Complementarity of Sketch and Knowledge is extremely strong**: Using sketch alone yields an FID of 33.29, and using knowledge alone yields an FID of 32.26. Combining both drops the FID to 23.97, indicating that sketches provide spatial layout information while knowledge offers relational constraints; both are indispensable.
- **Transferability of Knowledge Base Across Datasets**: The knowledge base constructed from 3D-FRONT can be directly transferred to ScanNet scene generation with only a minor performance drop (FID: 33.81 vs 33.47), suggesting that the knowledge base captures universal object relationship patterns.
- The spectrum filter (SF) yields approximately a 2-point improvement in FID, verifying the presence of padding zero-value interference and the effectiveness of the proposed solution.
- There is minimal difference between using ViT or ResNet50 as the sketch encoder (when knowledge is present), indicating that the knowledge base acts as the primary performance driver.

## Highlights & Insights

- **Practicable Design of the Knowledge Base**: Instead of handcrafting relations, five relationship categories are statistically extracted from data and stored probabilistically, which balances flexibility and scalability.
- **"Invisible Object" Generation is a Unique Selling Point**: Through the knowledge base, objects that are missing in the user's sketch but expected in the scene can be inferred (e.g., inferring a 'table' when a 'sofa' is drawn), enhancing the completeness and plausibility of the scene.
- **Simultaneous Generation of Layout and Shape**: Unlike most methods that retrieve objects after layout planning, SEK generates all attributes end-to-end, guaranteeing superior consistency.
- The spectrum filter tackles the zero-padding issue ingeniously by exploiting the low-frequency properties of the padding values in the frequency domain.

## Limitations & Future Work

- Sketches are obtained by applying Canny edge detection to rendered images and then manually removing walls. This introduces a domain gap between these synthetic sketches and actual freehand drawings, requiring potential sketch-photo domain adaptation in real-world scenarios.
- The knowledge base is statically constructed, which limits its ability to dynamically update or personalize based on user interactions.
- The number of objects in the scene must be padded to a fixed maximum limit $M$, which restricts the complexity of generating highly detailed scenes.
- Training and evaluation are limited to the 3D-FRONT dataset with few indoor categories (bedroom, dining room, living room); thus, the generalization capability to more complex scenarios (e.g., offices, laboratories) remains unverified.
- The absence of a user study limits the evaluation on whether the generated scenes truly align with user intent.

## Related Work & Insights

- **vs DiffuScene**: While DiffuScene is an unconditional scene generation approach, SEK introduces sketch and knowledge priors as conditions, yielding superior controllability and generation quality.
- **vs Graph-to-3D**: Graph-to-3D relies on a complete scene graph description, which is costly and non-intuitive to define for users; SEK requires only a freehand sketch and an object list, offering a more natural interaction paradigm.
- **vs ATISS**: ATISS is an autoregressive method given a predefined layout, whereas SEK utilizes a diffusion model to comprehensively account for global consistency across all objects simultaneously.
- **vs Sketch2Scene**: Sketch2Scene relies on an external 3D model database for retrieval and configuration, whereas SEK generates scenes in an end-to-end manner without relying on a database.

## Rating

- Novelty: ⭐⭐⭐⭐ Melding an expression-free external knowledge base with sketch-conditioned diffusion models for 3D scene generation is pioneered for the first time, and the design of the spectrum filter is highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ The evaluation comprehensively covers three tasks (generation, completion, and knowledge transfer) alongside complete ablation studies, though it suffers from the lack of a user study.
- Writing Quality: ⭐⭐⭐⭐ The overall structure is clear, detailed in its explanation of the construction and reasoning process of the knowledge base, and supported by rich illustrations.
- Value: ⭐⭐⭐⭐ The concept of a knowledge-enhanced conditional generation framework can be widely adapted to other 3D generation tasks, and the transferability of the knowledge base holds significant practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] DreamScene360: Unconstrained Text-to-3D Scene Generation with Panoramic Gaussian Splatting](dreamscene360_unconstrained_text-to-3d_scene_generation_with_panoramic_gaussian_.md)
- [\[ECCV 2024\] WaSt-3D: Wasserstein-2 Distance for Scene-to-Scene Stylization on 3D Gaussians](wast-3d_wasserstein-2_distance_for_scene-to-scene_stylization_on_3d_gaussians.md)
- [\[AAAI 2026\] IE-SRGS: An Internal-External Knowledge Fusion Framework for High-Fidelity 3D Gaussian Splatting Super-Resolution](../../AAAI2026/3d_vision/ie-srgs_an_internal-external_knowledge_fusion_framework_for_high-fidelity_3d_gau.md)
- [\[ECCV 2024\] TPA3D: Triplane Attention for Fast Text-to-3D Generation](tpa3d_triplane_attention_for_fast_text-to-3d_generation.md)
- [\[ICCV 2025\] MeshPad: Interactive Sketch-Conditioned Artist-Reminiscent Mesh Generation and Editing](../../ICCV2025/3d_vision/meshpad_interactive_sketch-conditioned_artist-reminiscent_mesh_generation_and_ed.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Hierarchical Material Recognition from Local Appearance
description: >-
  [ICCV 2025][3D Vision][material recognition] This paper proposes a hierarchical material taxonomy designed for visual applications alongside a new in-the-wild dataset, Matador (~7,200 material images with depth maps…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "material recognition"
  - "hierarchical classification"
  - "graph attention network"
  - "texture recognition"
  - "material taxonomy"
  - "novel view synthesis"
  - "few-shot learning"
date: 2026-05-08
content_hash: 0a9748de010a263b
---

# Hierarchical Material Recognition from Local Appearance

**Conference**: ICCV 2025
**arXiv**: [2505.22911](https://arxiv.org/abs/2505.22911)
**Code**: [Matador Project Page](https://cave.cs.columbia.edu/repository/Matador)
**Area**: 3D Vision
**Keywords**: material recognition, hierarchical classification, graph attention network, texture recognition, material taxonomy, novel view synthesis, few-shot learning

## TL;DR

This paper proposes a hierarchical material taxonomy designed for visual applications alongside a new in-the-wild dataset, Matador (~7,200 material images with depth maps, 57 categories). A graph attention network (GAT) leverages the taxonomic hierarchy for material recognition, achieving state-of-the-art results on multiple benchmarks while supporting few-shot learning of novel materials and material probing at arbitrary scene points.

## Background & Motivation

Material recognition is a fundamental problem in computer vision and is critical for autonomous systems interacting with the environment.

**Practical value of material recognition**: Humans can infer physical properties from visual appearance alone (e.g., a paper cup will feel hot; a ceramic cup is heavier but cooler). Machines require similar capabilities for intelligent environment interaction — for instance, an autonomous robot must first identify a spilled substance as liquid before deciding to use a cloth rather than a broom.

**Granularity-dependent applications**: Different applications require different levels of recognition granularity — sometimes knowing "liquid" suffices, while other times distinguishing "coffee" from "water" is necessary. Material recognition thus inherently demands a hierarchical structure.

**Limitations of existing methods**: Traditional texture recognition approaches (filter banks, CNN feature aggregation, etc.) treat material categories as a flat set, failing to exploit physical relationships among materials (e.g., rubber ⊂ plastic ⊂ polymer).

**Insufficient datasets**: Existing material datasets either cover too few categories (10–40 classes), exhibit poor intra-class diversity, or lack hierarchical annotations, making them inadequate for hierarchical material recognition research.

**Key insight**: Even when fine-grained material identification fails, correct recognition at a coarser level (e.g., "phase" or "composition") remains useful — for instance, it enables inference of mechanical properties.

## Method

### 1. Material Taxonomy

Inspired by the biological "tree of life," the paper constructs a hierarchical taxonomy covering 57 common materials:

- **Vocabulary construction**: Material nouns from WordNet are ranked by frequency in the M2D2 corpus (4.19 billion words); after merging synonyms, the 57 most frequent categories are retained.
- **Hierarchical structure**: Five levels organized by physical properties from coarse to fine — Phase → State → Composition → Form → Material.
- **Examples**: Matter → Solid → Abiotic → Metal → Ferrous → {Iron, Steel}; Solid → Biotic → Natural → Vegetation → {Flower, Foliage, Ivy, Shrub}.
- **Mechanical property table**: Ranges of density, surface roughness, Young's modulus, yield strength, and tensile strength are compiled for each leaf-node material, enabling recognition results to be directly linked to physical interaction parameters.

### 2. Matador Dataset

A new in-the-wild material image dataset is constructed:

- **Scale**: ~7,200 samples covering all 57 taxonomy categories, averaging 126 instances per class.
- **Capture device**: iPhone 15 Pro Max with a custom iOS application.
- **Each sample includes**:
    - Local appearance image (wide-angle camera, 12 MP, 74° FOV, 12-bit RAW)
    - Depth map (LiDAR, 100 pts/deg²)
    - Surrounding context image (ultra-wide camera, 12 MP, 104° FOV)
    - IMU motion data and metadata
- **Matador-C1**: A subset of ~6,600 samples across 37 classes, obtained by merging visually similar categories (e.g., all metals into one class) and removing texture-deficient categories (e.g., glass, paint).

### 3. Novel View Synthesis from Depth Maps

Depth maps are used to generate large volumes of novel-view training data:

- A 3D mesh is created from each depth map and textured with the corresponding appearance image.
- Spatial transformations (varying magnification and orientation) are applied to the mesh.
- Novel-view optical images are rendered via ray tracing using a thin-lens model to simulate depth-of-field and defocus effects.
- Additional simulation steps include pixel-area blur, per-pixel-pitch sampling, and photon and sensor noise injection.
- By varying these physical parameters, numerous novel-view images at different camera configurations, distances, and angles are rendered from each real sample.

### 4. Graph Attention Network (GAT) Classifier

The taxonomy is encoded as a directed graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$, where nodes represent taxonomy categories and edges represent parent–child relationships. A GAT is used for hierarchical material recognition.

**Node initialization**:
- For each node $v_i$, an encoder $\phi$ (ResNet50) extracts the mean feature over all images belonging to that node: $\mathbf{h}_i^0 = \frac{1}{\|\mathcal{T}_i\|} \sum_{x_j \in \mathcal{T}_i} \phi(x_j)$

**Image classification**:
- For an input image $x$, a global node feature $\mathbf{h}_g = \phi(x)$ is extracted, inserted into the graph, and connected to all nodes.
- GAT message passing update: $\mathbf{h}_i^{k+1} = \psi_a(\mathbf{h}_i^k, \bigoplus_{j \in \mathcal{N}_i} (\alpha_{ij}^k \psi_b(\mathbf{h}_i^k, \mathbf{h}_j^k)))$
- Here $\alpha_{ij}^k$ denotes edge attention weights controlling the degree of visual feature sharing between taxonomy nodes.
- $D$ layers are stacked (matching the taxonomy diameter) with residual connections.

### Loss & Training

A combined loss of BCE (encouraging learning of complete hierarchical paths) and per-level CE is used:

$$\max\left(\text{BCE}(\hat{\ell}, \ell), \frac{1}{D}\sum_{d=0}^{D-1}\text{CE}(\hat{\ell}_d, \ell_d)\right)$$

### 5. Material Probing in Scenes

At inference time, given a pixel in an image, progressively larger windows from 64×64 to 1024×1024 are evaluated. Monte Carlo Dropout is used to construct a prediction distribution, and a best-first search over the taxonomy tree produces a hierarchically consistent classification result.

## Key Experimental Results

### Standard Benchmarks (Flat Classification, ResNet50 Backbone)

| Method | KTH-2-b | FMD | GTOS | GTOS-M |
|--------|---------|-----|------|--------|
| DeepTEN | — | — | — | —† |
| MAPNet | — | — | — | — |
| CLASSNet | — | — | — | — |
| RADAM | — | — | — | — |
| FRP | — | — | —† | —† |
| **Ours** | **Best** | **Best** | **Best** | **Best** |

State-of-the-art performance is achieved on all four standard benchmarks.

### Matador Dataset Performance

| Method | Params | Matador | Matador-C1 | OOD |
|--------|--------|---------|------------|-----|
| CLIP (zero-shot) | 151M | 24.8 | 40.0 | 32.3 |
| GPT-4.1 (zero-shot) | 1.76T | 51.4 | 65.9 | 53.4 |
| DeepTEN (ResNet50) | 24M | 79.2 | 88.8 | 61.5 |
| DEPNet (ResNet50) | 25M | 82.7 | 87.6 | 76.1 |
| ConvNeXt-V2 | 28M | 83.1 | 89.7 | 81.9 |
| **Ours (ResNet50)** | **28M** | **Best** | **Best** | **Best** |

- Novel-view rendering augmentation yields up to **4.9%** improvement on the OOD test set.
- Hierarchical classification accuracy increases at coarser levels (State → Composition → Form).

### Few-Shot Learning

- With only **16 samples**, ~90% accuracy is achieved on previously unseen material categories.
- The mean path distance under misclassification is less than 2 hops — misclassified samples fall on neighboring nodes in the taxonomy.

### Training Details

- Backbone: ResNet50 (pretrained on IG-1B + ImageNet1k)
- GAT: 2 layers, hidden dim 512, output dim 256, 1 attention head
- Training: batch size 400, 100 epochs, lr=1e-4, cosine annealing, weight decay 5e-4
- Total parameters: 28.0M; training takes under 30 minutes on a single A6000 Ada (without novel-view rendering)

## Highlights & Insights

1. **Physics-driven taxonomy design**: Unlike semantic hierarchies such as WordNet, this taxonomy is organized by physical material properties, directly aligning hierarchical relationships with mechanical characteristics so that recognition results can be used to infer density, roughness, stiffness, and related quantities.
2. **Hierarchical fault tolerance**: Even when fine-grained classification fails (e.g., wool → carpet), correct recognition at a coarser level (e.g., "textile") is preserved, which is highly practical for real-world applications.
3. **Depth-map-driven data augmentation**: Using LiDAR depth maps to render novel views is a unique contribution of this work. Unlike conventional geometric augmentation, it physically simulates the imaging process under varying camera configurations, distances, and noise conditions.
4. **Model efficiency**: With only 28M parameters, the proposed model outperforms GPT-4.1 (1.76T parameters) on material recognition, underscoring the importance of domain-specific design.
5. **Natural synergy between taxonomy and GNN**: Encoding taxonomic relationships as a graph structure allows the attention mechanism to automatically learn which categories share visual features, providing greater flexibility than manually designed feature-sharing schemes.

## Limitations & Future Work

1. **Solid materials only**: The current taxonomy covers only solid materials; liquids and gases are left for future extension.
2. **Texture dependency**: Texture-deficient materials such as glass and paint are excluded from Matador-C1; future work should incorporate contextual and reflectance cues.
3. **Lambertian assumption**: Novel-view rendering assumes Lambertian surfaces and does not model specular reflection or subsurface scattering, introducing bias for materials such as metals.
4. **Scene-level generalization**: The paper demonstrates single-point probing capability but does not perform full-image material segmentation; leveraging neighborhood pixels to suppress or reinforce predictions is a natural extension.
5. **Cross-domain generalization**: The dataset was collected with a single smartphone model; robustness under different sensors and lighting conditions remains to be validated.
6. **Color utilization**: The merging of metal subcategories is motivated by their differing only in hue, indicating that color information is not yet fully exploited.

## Related Work & Insights

- **Evolution of texture recognition**: Gabor filter banks → BoW/Texton → CNN-learned filters → multi-scale feature aggregation (e.g., FENet, CLASSNet, RADAM) → the hierarchical graph approach proposed in this paper.
- **Hierarchical image classification**: Methods leveraging semantic hierarchies such as WordNet (e.g., HGCLIP) have been primarily applied to object recognition; this paper introduces analogous ideas into material recognition, with a hierarchy grounded in physical properties.
- **GNNs in visual recognition**: Graph neural networks have been applied to zero-shot learning (knowledge graph propagation) and fine-grained recognition (relation discovery); this paper's contribution is combining them with a physically grounded material taxonomy.
- **Material datasets**: KTH-TIPS2-b (controlled BTF), FMD (web images), GTOS/GTOS-Mobile (ground terrain) — Matador surpasses all in number of categories (57), intra-class diversity (126 instances/class), and data modalities (appearance + depth + context).
- **Broader implications**: The paradigm of hierarchical taxonomy + GNN is generalizable to other recognition tasks with natural hierarchical structure (e.g., biological species, rock minerals, fabric types). The depth-map-driven novel-view augmentation strategy is also worth exploring in other 3D perception tasks.

## Rating

- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Sequential Gaussian Avatars with Hierarchical Motion Context](sequential_gaussian_avatars_with_hierarchical_motion_context.md)
- [\[ICCV 2025\] CL-Splats: Continual Learning of Gaussian Splatting with Local Optimization](cl-splats_continual_learning_of_gaussian_splatting_with_local_optimization.md)
- [\[ICCV 2025\] MaterialMVP: Illumination-Invariant Material Generation via Multi-view PBR Diffusion](materialmvp_illumination-invariant_material_generation_via_multi-view_pbr_diffus.md)
- [\[ICCV 2025\] SuperMat: Physically Consistent PBR Material Estimation at Interactive Rates](supermat_physically_consistent_pbr_material_estimation_at_interactive_rates.md)
- [\[NeurIPS 2025\] WildCAT3D: Appearance-Aware Multi-View Diffusion in the Wild](../../NeurIPS2025/3d_vision/wildcat3d_appearance-aware_multi-view_diffusion_in_the_wild.md)

</div>

<!-- RELATED:END -->

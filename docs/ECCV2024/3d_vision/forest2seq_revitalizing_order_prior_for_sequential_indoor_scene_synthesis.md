---
title: >-
  [Paper Note] Forest2Seq: Revitalizing Order Prior for Sequential Indoor Scene Synthesis
description: >-
  [ECCV 2024][3D Vision][Indoor Scene Synthesis] Proposed the Forest2Seq framework, which organizes unordered indoor scene objects into a hierarchical scene tree/forest structure and derives a meaningful permutation order as prior knowledge using breadth-first search (BFS). Combined with a Transformer autoregressive decoder, it significantly improves the quality of indoor scene synthesis.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Indoor Scene Synthesis"
  - "Autoregressive Generation"
  - "Sequence Order"
  - "Scene Tree/Forest"
  - "Transformer"
date: 2026-05-08
content_hash: c01b4495c78bda3f
---

# Forest2Seq: Revitalizing Order Prior for Sequential Indoor Scene Synthesis

**Conference**: ECCV 2024  
**arXiv**: [2407.05388](https://arxiv.org/abs/2407.05388)  
**Code**: No public code  
**Area**: 3D Vision  
**Keywords**: Indoor Scene Synthesis, Autoregressive Generation, Sequence Order, Scene Tree/Forest, Transformer

## TL;DR

Proposed the Forest2Seq framework, which organizes unordered indoor scene objects into a hierarchical scene tree/forest structure and derives a meaningful permutation order as prior knowledge using breadth-first search (BFS). Combined with a Transformer autoregressive decoder, it significantly improves the quality of indoor scene synthesis.

## Background & Motivation

**Background**: Automated indoor scene synthesis has evolved from manual prior-constrained optimization to deep learning-based methods, including autoregressive Transformer models (SceneFormer, ATISS, COFS), graph-based methods (DiffuScene), and others.

**Limitations of Prior Work**: Current autoregressive models treat scenes as unordered sets or use a random order, lacking an understanding of the semantic relationships and hierarchical structures between objects. ATISS avoids the order issue through permutation invariance, and COFS assumes an unordered layout using a masked language model—however, these methods overlook the valuable information embedded in the sequence order.

**Key Challenge**: The placement of scene objects possesses a natural hierarchical logic (e.g., placing large furniture before small accessories), but random or frequency-based ordering in existing methods fails to capture this structure. The lack of order leads to issues such as object intersections and implausible positionings.

**Goal**: To discover a superior object permutation order prior for autoregressive scene generation, enabling the model to generate scenes according to the intuitive spatial reasoning principle of "primary first, secondary later".

**Key Insight**: Parsing the scene into a hierarchical tree structure (functional zones $\rightarrow$ main furniture $\rightarrow$ accessory items), automatically mining the implicit hierarchy via a clustering algorithm, and linearizing it into an ordered sequence using Breadth-First Search (BFS).

**Core Idea**: Order is an overlooked yet crucial prior in autoregressive indoor scene generation, and structured permutation via a scene forest can significantly elevate synthesis quality.

## Method

### Overall Architecture

Forest2Seq comprises two core modules: (1) **Sequence Construction**—parsing the unordered set of scene objects $\mathcal{O}=\{o_1, ..., o_n\}$ into scene trees/forests via Modified Euclidean Distance Clustering (MEDC), followed by BFS traversal to obtain the ordered sequence $\mathcal{S}=\pi(\mathcal{O})$; (2) **Sequence Generation**—autoregressively generating the scene object sequence using a decoder-only causal Transformer under a denoising strategy.

### Key Designs

1. **Scene Tree**:

    - **Function**: Organizing objects within a scene into a hierarchical tree structure based on functional zones.
    - **Mechanism**: Each object is represented as an oriented bounding box $o_i = (c_i, t_i, b_i, r_i)$ (class, position, size, rotation). Modified Euclidean Distance Clustering (MEDC) is used to define the distance matrix between objects:
    $m_{ij} = d_{ij} + \lambda \cdot (1 - \text{GIoU}(\bar{o}_i, \bar{o}_j))$
   where $d_{ij}$ is the Euclidean distance between center points, $\text{GIoU} \in [-1, 1]$ evaluates bounding box overlap, and $\lambda=0.02$. Objects are clustered into multiple functional zones using the DBSCAN algorithm (eps=0.15, min\_samples=2). The largest object in each cluster serves as the root node (main furniture), and the remaining objects serve as child nodes (accessory items).
    - **Design Motivation**: Aligns with intuition—placing a bed before a nightstand, or a sofa before a coffee table. The tree structure encodes parent-child (primary-secondary) relationships.

2. **Scene Forest**:

    - **Function**: Handling the ambiguity of flexible objects (e.g., cabinets) that might belong to multiple functional zones.
    - **Mechanism**: For outlier points in DBSCAN clustering (such as cabinets that can be placed in any area), they are associated with each possible parent node, generating a set of trees (forest). During training, a tree is randomly selected from the forest for BFS serialization.
    - **Design Motivation**: A single tree forces an outlier object to belong to a single zone, introducing artificial bias. The forest representation permits multiple valid permutations for the same scene, naturally implementing data augmentation.

3. **BFS Linearization**:

    - **Function**: Converting scene trees/forests into linear sequences for Transformer processing.
    - **Mechanism**: Performing BFS traversal on the scene tree: first generating all root nodes (main furniture of each functional zone), and then generating child nodes layer by layer. Sibling nodes in the same layer are randomly shuffled to eliminate human ordering bias. The sequence is denoted as $\mathcal{S}_F = \pi_F(\mathcal{O})$.
    - **Design Motivation**: BFS guarantees the hierarchical order of placing main furniture first followed by accessory items. Experiments demonstrate that BFS outperforms DFS, as the sequence set generated by BFS exhibits higher internal consistency (Hamming distance of 1.87 vs. 4.01 for DFS).

4. **Transformer Decoder + Denoising Strategy**:

    - **Function**: Autoregressively generating sequences of objects.
    - **Mechanism**: The framework consists of four components:
        - **Layout Encoder**: A small ViT that encodes the binary layout mask $s_0 \in \mathbb{R}^{64 \times 64}$ into a starting token $x_0 \in \mathbb{R}^{512}$
        - **Object Encoder**: Encodes object attributes into a token $x_i = [\lambda(c_i); \psi(t_i); \psi(b_i); \psi(r_i)] \in \mathbb{R}^{512}$
        - **Causal Transformer**: $\hat{x}_i = f_\theta(x_{<i}; x_0)$, utilizing masked self-attention and absolute positional encoding
        - **Attribute Extractor**: Outputs a Mixture of Logistics (MoL) distribution $p(h) = \sum_{j=1}^{K} \alpha_j \text{Logistic}(\mu_j, \sigma_j)$ (for continuous attributes) and a Softmax distribution (for categorical attributes)
    - **Denoising Strategy**: During training, there is a 5% probability of replacing object tokens with `[MASK]`, and a 5% probability of replacing the ground-truth class with a random class, reducing overfitting and error propagation.

### Loss & Training

- **Loss Function**: Negative Log-Likelihood: $\mathcal{L}_\theta = -\sum_{i=1}^{N} \log p_\theta(s_i | s_{<i})$, which is the sum of cross-entropy of conditional probabilities for each token.
- The mixture of logistics distribution uses $K=10$ components.
- **Optimizer**: AdamW, learning rate 1e-4, no warmup/decay.
- **Training**: Batch size of 128, 1000 epochs, dropout of 0.1, validation every 10 epochs to select the best model.
- **Data Augmentation**: 0°-360° random rotation + random tree selection from the scene forest.
- **Pre-training Transfer**: Small-scale room types (library/living/dining) are initialized using bedroom pre-training.

## Key Experimental Results

### Main Results

| Method | Bedroom KL↓ | Bedroom FID↓ | Living KL↓ | Living FID↓ | Dining KL↓ | Dining FID↓ | Library KL↓ | Library FID↓ |
|------|------------|-------------|-----------|-----------|-----------|-----------|-----------|-----------|
| FastSynth | 6.4 | 88.1 | 17.6 | 66.6 | 51.8 | 58.9 | 43.1 | 86.6 |
| SceneFormer | 5.2 | 90.6 | 31.3 | 68.1 | 36.8 | 60.1 | 23.2 | 89.1 |
| ATISS | 8.6 | 73.0 | 14.1 | 43.3 | 15.6 | 47.6 | 10.1 | 75.3 |
| COFS | 5.0 | 73.2 | 8.1 | 35.9 | 9.3 | 43.1 | 6.7 | 75.7 |
| DiffuScene | 5.1 | 69.0 | 8.3 | 38.2 | 7.9 | 45.8 | — | — |
| **Forest2Seq** | **4.2** | **67.9** | **5.9** | **35.2** | **5.5** | **40.2** | **5.2** | **69.1** |

### Ablation Study

| Permutation Type | Diversity | Inconsistency | KL↓ | FID↓ | CAS (%) |
|---------|-------|---------|-----|------|--------|
| Random (single) | 1 | 0 | 20.0 | 49.4 | 83.7 |
| Fixed (frequency) | 1 | 0 | 17.9 | 49.8 | 80.1 |
| Tree + BFS | 1 | 0 | 7.90 | 36.1 | 68.1 |
| Random (multiple) | ∞ | 9.54 | 13.1 | 43.3 | 76.4 |
| Forest + DFS | 2.83 | 4.01 | 9.40 | 40.5 | 71.7 |
| **Forest + BFS** | **2.83** | **1.87** | **5.90** | **35.2** | **68.0** |

### Key Findings

- **Order Prior is Crucial**: Changing Random $\rightarrow$ Tree + BFS reduces KL from 20.0 to 7.90 and FID from 49.4 to 36.1, proving that a meaningful permutation order has a massive impact on scene generation quality.
- **Forest Outperforms Single Tree**: Tree $\rightarrow$ Forest further drops KL from 7.90 to 5.90, showing that the forest representation effectively handles flexible objects and provides data augmentation.
- **BFS Outperforms DFS**: The BFS sequence set has high internal consistency (Hamming distance of 1.87 vs. 4.01), leading to more accurate placement of main furniture.
- **Position Encoding is Important**: Without position encoding, KL = 11.1, whereas with absolute position encoding, KL = 5.9, indicating the model indeed utilizes sequential information.
- **Extremely Compact Model**: With only 9.99MB of parameters, it is 51% of COFS (19.4MB) and 13% of DiffuScene (74.1MB).
- **Attention Visualization**: Under forest-based ordering, attention is concentrated on key predecessor objects, whereas under random ordering, attention is uniformly scattered. This directly proves that the permutation prior influences the model's internal representation.
- **User Study**: Achieves a 52% preference rate in Living Room and 58% in Dining Room, far exceeding all baselines.

## Highlights & Insights

- **Rediscovering the Value of Order**: While works like ATISS strive for permutation invariance, this work finds that introducing the correct order is actually more effective, representing a valuable counter-intuitive shift.
- **Discovery of Scene "Grammar"**: Scenes exhibit a language-like "grammar"—Subject (main furniture) $\rightarrow$ Predicate (spatial relationship) $\rightarrow$ Object (accessory items). BFS traversal naturally encodes this "grammar".
- **Elegance of Forest Representation**: Using a collection of multiple trees to address the attribution ambiguity of flexible objects is logical and naturally achieves data augmentation.
- **Simple and Effective**: The core method does not require complex network designs; a massive improvement is achieved merely by changing the input order, highlighting the importance of data representation.
- **Minimal Parameters**: 48% reduction in parameters + comparable inference speed + higher quality = a win-win scenario for efficiency and performance.

## Limitations & Future Work

- Does not incorporate doors and windows as extra conditions, which may cause furniture to block windows.
- Lacks spatial constraints between objects, occasionally causing overlaps.
- Limited generation performance on unconventional layouts (L-shaped, irregular shapes), constrained by the diversity of training data.
- Scene parsing depends on DBSCAN hyperparameters (eps=0.15); different room types may require different settings.
- Future directions could explore learnable permutation modules to jointly optimize ordering and generation end-to-end.

## Related Work & Insights

- **vs. ATISS**: Uses random shuffling + no position encoding to achieve permutation invariance. Forest2Seq proves that this design discards valuable sequential information, reducing KL from 14.1 to 5.9.
- **vs. COFS**: Uses BART bidirectional encoding + autoregressive decoding. Forest2Seq achieves better results with a simpler decoder-only architecture + a superior order prior, while reducing parameter count by half.
- **vs. DiffuScene**: Employs fully-connected graphs + DDPM denoising. Forest2Seq uses structured tree/forest representations + autoregressive generation, yielding better results and 200$\times$ faster inference ($0.16\text{s}$ vs. $34.9\text{s}$).
- **vs. Set2Seq**: A classic set-to-sequence work, but utilizes learned sorting instead of structured ordering. Ours has a clearer physical interpretation.

## Rating

- Novelty: ⭐⭐⭐⭐ The core idea (scene tree/forest ordering) is intuitive and effective, though technically relatively simple.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Ultra-thorough, featuring 4 room types $\times$ 6 baselines + ablations comparing 6 permutations + attention visualizations + user studies + downstream tasks.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, highly explanatory Figures 1-6, and highly convincing ablation studies.
- Value: ⭐⭐⭐⭐ The finding that "the order prior is important" is universally inspiring for autoregressive generation; the method is concise and generalizable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] NeRFPrior: Learning Neural Radiance Field as a Prior for Indoor Scene Reconstruction](../../CVPR2025/3d_vision/nerfprior_learning_neural_radiance_field_as_a_prior_for_indoor_scene_reconstruct.md)
- [\[ECCV 2024\] MegaScenes: Scene-Level View Synthesis at Scale](megascenes_scene-level_view_synthesis_at_scale.md)
- [\[ECCV 2024\] WaSt-3D: Wasserstein-2 Distance for Scene-to-Scene Stylization on 3D Gaussians](wast-3d_wasserstein-2_distance_for_scene-to-scene_stylization_on_3d_gaussians.md)
- [\[ECCV 2024\] Analysis-by-Synthesis Transformer for Single-View 3D Reconstruction](analysis-by-synthesis_transformer_for_single-view_3d_reconstruction.md)
- [\[ECCV 2024\] Zero-Shot Multi-Object Scene Completion](zero-shot_multi-object_scene_completion.md)

</div>

<!-- RELATED:END -->

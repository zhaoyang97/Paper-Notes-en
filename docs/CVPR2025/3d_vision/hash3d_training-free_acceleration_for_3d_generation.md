---
title: >-
  [Paper Note] Hash3D: Training-free Acceleration for 3D Generation
description: >-
  [CVPR 2025][3D Vision][3D Generation Acceleration] Hash3D discovers that the features of diffusion models are highly redundant across adjacent camera poses and timesteps during SDS optimization. By caching and reusing intermediate features using an adaptive grid hash table, it accelerates various text-to-3D and image-to-3D methods by 1.3 to 4 times without training, while simultaneously enhancing multi-view consistency.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D Generation Acceleration"
  - "Score Distillation"
  - "Feature Reuse"
  - "Hash Table"
  - "Diffusion Models"
date: 2026-05-08
content_hash: 0a2e08be051a5060
---

# Hash3D: Training-free Acceleration for 3D Generation

**Conference**: CVPR 2025  
**arXiv**: [2404.06091](https://arxiv.org/abs/2404.06091)  
**Code**: [https://github.com/Adamdad/hash3D](https://github.com/Adamdad/hash3D)  
**Area**: 3D Vision  
**Keywords**: 3D Generation Acceleration, Score Distillation, Feature Reuse, Hash Table, Diffusion Models

## TL;DR
Hash3D discovers that the features of diffusion models are highly redundant across adjacent camera poses and timesteps during SDS optimization. By caching and reusing intermediate features using an adaptive grid hash table, it accelerates various text-to-3D and image-to-3D methods by 1.3 to 4 times without training, while simultaneously enhancing multi-view consistency.

## Background & Motivation

1. **Background**: 3D generation based on 2D diffusion models (SDS) has become a mainstream approach, distilling 3D models by sampling the score function across different views and denoising timesteps.

2. **Limitations of Prior Work**: SDS requires thousands to tens of thousands of iterations, each necessitating a full forward pass of the diffusion model. Generating a single object can take hours, severely limiting practical applications.

3. **Key Challenge**: Existing acceleration schemes (training inference models, improving 3D representations, and directly generating sparse views) either require intensive training resources, require specialized designs for each representation, or are limited by view consistency. The root cause—excessive inference passes of the diffusion model—remains unaddressed.

4. **Goal**: Can the actual number of diffusion model inferences in SDS be reduced?

5. **Key Insight**: Experiments demonstrate that diffusion model features (specifically the input to the last upsampling layer of the U-Net) exhibit extremely high cosine similarity (>0.8) across adjacent camera poses (within ±10°) and adjacent timesteps. This implies that outputs from a large number of inference steps are redundant.

6. **Core Idea**: Cache intermediate features of the diffusion model using a grid hash table, and directly reuse existing features for new requests at adjacent views/timesteps to avoid repetitive inference.

## Method

### Overall Architecture
Hash3D is a plug-and-play module that can be embedded into any SDS optimization pipeline. It maintains a hash table to store intermediate features of the diffusion model. Each time a new camera pose and timestep are sampled, the table is queried: if a hit occurs, the features are reused to skip most of the inference; if a miss occurs, normal inference is performed, and the table is updated. Adaptive grid sizes are used to balance the accuracy and efficiency of feature reuse.

### Key Designs

1. **Grid Hashing and Feature Caching**:

    - Function: Efficiently index and reuse diffusion model intermediate features of adjacent views/timesteps.
    - Mechanism: The hash key consists of four dimensions: azimuth $\theta$, elevation $\phi$, radius $\rho$, and timestep $t$. The grid sizes are $\Delta\theta, \Delta\phi, \Delta\rho, \Delta t$, and the continuous space is discretized into grid cells via rounding operations. The hash function is defined as $\text{idx} = (i + N_1 \cdot j + N_2 \cdot k + N_3 \cdot l) \mod n$, where $N_1, N_2, N_3$ are large prime numbers. Each bucket maintains a queue of maximum length 3 to store features. During retrieval, a distance-weighted average is applied to blend features in the queue: $\mathbf{v} = \sum W_i \mathbf{v}_i$, with weights computed via softmax over $e^{-\|\mathbf{x}-\mathbf{x}_i\|_2^2}$.
    - Design Motivation: Grid hashing naturally preserves spatio-temporal locality, mapping adjacent poses to the same bucket. The queue design ensures that only the latest data is stored, adapting to the continuously optimized 3D representation.

2. **Feature Reuse Location Selection**:

    - Function: Determine the structural location in the U-Net where inference is truncated and cached features are injected.
    - Mechanism: The input feature $\mathbf{v}_{l-1}^{(U)}$ to the last upsampling layer of the U-Net is extracted. Upon a cache hit, only the final upsampling layer of the U-Net (the shallowest layer) is executed to yield the final prediction, skipping computations in all preceding layers. A hash probability $\eta=0.1$ controls the balance between retrieval and update—retrieving features with 90% probability and performing normal inference to update the table with 10% probability.
    - Design Motivation: The input features to the last upsampling layer fuse high-level semantics (from deeper layers) and low-level details (from skip connections). Reusing features at this layer maximizes computational savings while maintaining output quality.

3. **Adaptive Grid Size**:

    - Function: Dynamically select the optimal grid granularity to adapt to variations across different objects and view angles.
    - Mechanism: Three sets ($M=3$) of hash tables with different grid sizes ($\Delta\theta, \Delta\phi, \Delta t \in \{10°, 20°, 30°\}$) are maintained simultaneously. During each update, the cosine similarity between the new feature and existing features is calculated, and the average similarity for each bucket is maintained using an exponential moving average: $s_{\text{idx}^{(m)}} \leftarrow \gamma s + (1-\gamma)\text{cos}(\mathbf{v}_{new}, \mathbf{v}_i)$. In retrieval, the grid size with the highest average similarity is selected.
    - Design Motivation: Experiments show that the optimal window size varies across objects (e.g., a "ghost" is suitable for 5° but a "capybara" is not), and a fixed grid size cannot accommodate all scenarios. Although maintaining multiple hash tables might seem computationally heavy, the hashing operations themselves are extremely lightweight as they store only references rather than data copies.

### Loss & Training
Hash3D is training-free and does not introduce any new loss functions. It is directly embedded into the optimization loop of existing SDS frameworks, modifying only the inference mechanism of the diffusion model.

## Key Experimental Results

### Main Results

| Method | Original Time | +Hash3D Time | Speedup | PSNR | CLIP-G/14 |
|------|---------|-------------|--------|------|-----------|
| DreamGaussian | 2min | **30s** | 4.0× | 16.36(+0.15) | 0.694 |
| Zero-123(NeRF) | 20min | **7min** | 3.3× | 17.96(+0.19) | 0.665 |
| Zero-123(GS) | 6min | **3min** | 2.0× | 18.62(+0.21) | 0.632 |
| Magic123 | 120min | **90min** | 1.3× | 18.63(-0.09) | 0.715 |
| GaussianDreamer | 15min | **10min** | 1.5× | - | 0.412 |

### Ablation Study

| Configuration | Speedup | CLIP Score | Description |
|------|--------|------------|------|
| Hash3D (η=0.1, M=3) | 3.3× | 0.665 | Default Configuration |
| Fixed Grid (Δ=10°) | ~2.5× | Slightly lower | No adaptation, more conservative |
| Fixed Grid (Δ=30°) | ~4× | Significantly lower | Too large window size causes artifacts |
| η=0.5 (50% reuse) | ~2× | 0.665 | Low reuse rate, limited speedup |
| η=0.01 (99% reuse) | ~4× | Slightly lower | Over-reuse leads to quality degradation |

### Key Findings
- Hash3D not only accelerates generation but also slightly improves rendering quality (with general minor improvements in PSNR/SSIM), because feature sharing enhances multi-view consistency.
- Feature similarity is higher than 0.8 within ±10°, which serves as the foundation for the proposed method.
- Combining it with 3DGS yields the best performance: approximately 10 minutes for text-to-3D, and 30 seconds for image-to-3D.
- A user study (44 participants) shows that the visual quality of 3D objects generated by Hash3D is comparable to, or even slightly better than, the original methods.
- Theoretical MACs are reduced by around 8% (168.78G → 154.76G), but the actual speedup is larger due to skipping entire layers of inference.

## Highlights & Insights
- **Observation-Driven Design**: Discovering the feature redundancy first and then designing the caching strategy—this "observation before design" approach is more elegant than blind optimization.
- **Unexpected discovery of speedup-improving-quality**: Feature sharing unintentionally acts as cross-view consistency regularization, which explains why the acceleration method unexpectedly improves quality.
- **Strong Generalizability**: Completely plug-and-play and effective across 8 different 3D generation methods, indicating that feature redundancy in SDS is a universal phenomenon.

## Limitations & Future Work
- For methods that are inherently fast to render (such as DreamGaussian, which takes only 2 minutes), the absolute time saved by acceleration is limited.
- The adaptive grid searches three preset sizes in a brute-force manner; more refined learning strategies could be explored.
- Only the inputs to the last upsampling layer of the U-Net are cached; expanding this to multi-layer caching might yield even greater speedups.
- For diffusion models with DiT architectures (e.g., Stable Diffusion 3), whether similar feature redundancy exists remains to be investigated.

## Related Work & Insights
- **vs DeepCache/FORA**: Feature caching methods for 2D diffusion models. Hash3D extends this paradigm to multi-view scenarios in 3D SDS.
- **vs DreamGaussian**: DreamGaussian accelerates by modifying 3D representations, while Hash3D speeds up the diffusion model inference. They are orthogonal and can be combined.
- **vs Instant-NGP Hashing**: Both utilize multi-resolution hashing, but the scenarios differ—NGP hashes spatial coordinates, whereas Hash3D hashes camera poses and timesteps.

## Rating
- Novelty: ⭐⭐⭐⭐ The concept of combining feature caching with SDS optimization is novel and intuitive.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive validation across 8 methods, including user studies and ablation experiments.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and detailed experiments.
- Value: ⭐⭐⭐⭐ A highly versatile, plug-and-play acceleration scheme with high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Fast3Dcache: Training-free 3D Geometry Synthesis Acceleration](../../CVPR2026/3d_vision/fast3dcache_training-free_3d_geometry_synthesis_acceleration.md)
- [\[CVPR 2025\] SfM-Free 3D Gaussian Splatting via Hierarchical Training](sfm-free_3d_gaussian_splatting_via_hierarchical_training.md)
- [\[CVPR 2025\] SelfSplat: Pose-Free and 3D Prior-Free Generalizable 3D Gaussian Splatting](selfsplat_pose-free_and_3d_prior-free_generalizable_3d_gaussian_splatting.md)
- [\[CVPR 2025\] CADDreamer: CAD Object Generation from Single-view Images](caddreamer_cad_object_generation_from_single-view_images.md)
- [\[CVPR 2025\] A Unified Image-Dense Annotation Generation Model for Underwater Scenes](a_unified_image-dense_annotation_generation_model_for_underwater_scenes.md)

</div>

<!-- RELATED:END -->

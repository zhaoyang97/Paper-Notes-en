---
title: >-
  [Paper Note] MegaSynth: Scaling Up 3D Scene Reconstruction with Synthesized Data
description: >-
  [CVPR 2025][3D Vision][Synthetic Data] MegaSynth proposes to achieve scalable 3D scene data synthesis by removing the reliance on semantic information, generating a dataset of 700k scenes (50 times larger than the real-world dataset DL3DV). It is used to train Large Reconstruction Models (LRMs), bringing a significant improvement of 1.2-1.8dB in PSNR across multiple benchmarks.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Synthetic Data"
  - "Large Reconstruction Models"
  - "Procedural Generation"
  - "Non-semantic Data"
  - "3D Gaussians"
date: 2026-05-08
content_hash: 6e86d24e4bb70a3e
---

# MegaSynth: Scaling Up 3D Scene Reconstruction with Synthesized Data

**Conference**: CVPR 2025  
**arXiv**: [2412.14166](https://arxiv.org/abs/2412.14166)  
**Code**: [https://hwjiang1510.github.io/MegaSynth/](https://hwjiang1510.github.io/MegaSynth/) (Project Page + Code)  
**Area**: 3D Vision  
**Keywords**: Synthetic Data, Large Reconstruction Models, Procedural Generation, Non-semantic Data, 3D Gaussians

## TL;DR
MegaSynth proposes to achieve scalable 3D scene data synthesis by removing the reliance on semantic information, generating a dataset of 700k scenes (50 times larger than the real-world dataset DL3DV). It is used to train Large Reconstruction Models (LRMs), bringing a significant improvement of 1.2-1.8dB in PSNR across multiple benchmarks.

## Background & Motivation

**Background**: Large Reconstruction Models (LRMs) inherit the concept of scaling laws from NLP and 2D vision, attempting to learn general 3D reconstruction priors through large models and massive data. Significant progress has been made in object-level reconstruction (e.g., trained on Objaverse 800K instances), but scene-level reconstruction remains challenging.

**Limitations of Prior Work**: Scene-level datasets face two major bottlenecks. (1) Severely insufficient scale—the largest clean scene dataset, DL3DV, contains only about 10k scenes, compared to 800k instances in the object-level Objaverse. Manual collection of scene data is time-consuming, expensive, and difficult to scale. (2) Uneven data quality—existing datasets generally suffer from a lack of scene diversity, narrow camera movements, noisy content, and inaccurate annotations.

**Key Challenge**: Scene-level 3D reconstruction requires large-scale, diverse, and high-quality training data, but there is a fundamental conflict between the collection cost of real scene data and the quality requirements.

**Goal**: Can we bypass the bottleneck of real data collection through synthetic data to substantially scale up and improve the quality of training data for scene-level 3D reconstruction?

**Key Insight**: The authors' key insight is that multi-view 3D reconstruction is fundamentally a low-level geometric task that does not require semantic information. Traditional methods (COLMAP, MVS, NeRF) and emerging feed-forward models all exhibit non-semantic characteristics. Therefore, synthetic data does not need semantic correctness (such as the rationality of object interactions or scene arrangement logic); it only needs basic spatial structures and geometric primitives, thereby bypassing the complexity of semantic modeling to achieve scalable generation.

**Core Idea**: Procedurally generate 700k training scenes using non-semantic geometric primitives, proving that 3D reconstruction does not require semantic priors, and that the complementarity of synthetic and real data can dramatically boost reconstruction quality.

## Method

### Overall Architecture
The data generation pipeline of MegaSynth consists of three steps: (1) Scenario layout generation—determining the scene size and bounding box positions of objects; (2) Instantiating object geometry—composing objects from shape primitives (cubes, spheres, cylinders, cones) and applying random textures; (3) Lighting randomization. Then, camera poses are sampled for rendering, resulting in RGB images and depth maps. The data is used for joint training or pre-training LRMs (GS-LRM, Long-LRM), achieving optimal performance when combined with small-scale real data.

### Key Designs

1. **Procedural Generation of Non-semantic Scenes**:

    - Function: To generate scaleable, fast, and geometrically diverse 3D training scenes without semantic modeling.
    - Mechanism: A scene is modeled as a cubic room, inside of which is filled with 3D bounding boxes of various object types. Large objects tend to be placed near the floor, while small objects are positioned more flexibly. Object geometry is formed by combining primitives (cubes, spheres, cylinders, cones), and random height fields are applied for surface deformation to produce bumpy details. Fine-grained structures (primitive wireframes) are added to simulate high-frequency geometry, and axis-aligned geometry (thin rods, planes) is included to mimic the distribution of real scenes under the Manhattan world assumption. Textures are randomly assigned color maps, normal maps, roughness maps, and metallic maps, with higher sampling probabilities for specular and glass-like materials.
    - Design Motivation: The core advantage of eliminating semantics is scalability—there is no need to model complex rules such as "what should go on the table" or "which direction the chair should face". It only takes 3 days to generate 700k scenes, whereas semantically accurate scene generation approaches are limited by complex procedural rules or the slow inference speed of generative models.

2. **Complexity Control and Real-Data Distribution Alignment**:

    - Function: To ensure that synthetic data has sufficient complexity to sustain training while being loosely aligned with real-world distributions to promote generalization.
    - Mechanism: Three types of lighting combinations are used to increase complexity—environmental light (default uniform illumination), sunlight (projected through random windows on the room walls to generate shadows), and emissive objects and light bulbs (simulating indoor point lights, which can have high brightness to simulate dark environments). Camera sampling strategy: distinguishing between internal and external spaces of the scene. External cameras point toward the scene center to ensure good coverage, while internal cameras have more randomized directions to increase diversity. Camera baseline constraints—preventing excessively large baselines for external cameras by sampling more small-baseline scenes to align with real camera distributions. FOV randomization is used to simulate different lenses.
    - Design Motivation: Ablation studies (Table 2) show that without complexity control, training collapses at 70k iterations, whereas the model with complexity control can be trained stably. Aligning the camera distribution with real data is crucial for generalization.

3. **Hybrid Data Training and Geometric Supervision**:

    - Function: To leverage the respective strengths of synthetic and real data for complementary training.
    - Mechanism: Two training strategies are supported—pre-training then fine-tuning (first pre-training on MegaSynth, then fine-tuning on DL3DV) and joint training (mixing both datasets during training). The loss function consists of two parts: (1) rendering loss $\mathcal{L}_{img} = \text{MSE}(I, \hat{I}) + \lambda \cdot \text{Perceptual}(I, \hat{I})$, applied to both synthetic and real data; (2) geometry loss $\mathcal{L}_{loc} = M \cdot \text{Smooth-L1}(\mathbf{c}, \mathbf{G}_{loc})$ to supervise the predicted 3D Gaussian center locations, applied only to synthetic data with accurate depth, utilizing a mask M to exclude pixels with depth values that are too large.
    - Design Motivation: Synthetic data provides scale and accurate metadata (depth, precise camera parameters), while real data introduces real-world characteristics such as sensor noise and lighting artifacts. Geometric supervision is particularly important for scene-level reconstruction—since scenes have wide depth ranges, relying solely on photometric cues makes it difficult to infer accurate geometry.

### Loss & Training
Final Loss: $\mathcal{L}^S = \mathcal{L}_{img}^S + \gamma \cdot \mathcal{L}_{loc}^S$. Based on the GS-LRM and Long-LRM frameworks, 32 input views are used by default. The pre-training + fine-tuning scheme yields the best results. MegaSynth contains 700k scenes, and DL3DV contains approximately 10k scenes.

## Key Experimental Results

### Main Results

**Resolution 128, 32 input views:**

| Model | Training Data | DL3DV PSNR↑ | Hypersim PSNR↑ | MipNeRF360 PSNR↑ |
|------|---------|-------------|----------------|-------------------|
| 3DGS (Per-scene Opt.) | - | 24.27 | 20.67 | 16.46 |
| GS-LRM | DL3DV | 24.60 | 23.89 | 19.93 |
| **GS-LRM (ours)** | **DL3DV+MegaSynth** | **25.75** | **25.46** | **21.19** |
| Long-LRM | DL3DV | 24.18 | 23.41 | 19.68 |
| **Long-LRM (ours)** | **DL3DV+MegaSynth** | **25.44** | **25.01** | **20.86** |

**Indoor/Outdoor Comparison (GS-LRM, 128 resolution):**

| Test Set | Only DL3DV | DL3DV + MegaSynth | Gain |
|--------|----------|-------------------|------|
| DL3DV Indoor | 25.41 | 26.75 | +1.34dB |
| DL3DV Outdoor | 23.09 | 23.89 | +0.80dB |

### Ablation Study

| Configuration | Iteration of Training Failure | Hypersim PSNR (Synthetic Only) | Hypersim PSNR (After Fine-Tuning) |
|------|------------|----------------------|----------------------|
| (0) No Control + No Geometric Loss | 70K | 17.18 | 18.44 |
| (1) + Geometric Loss | 45K | 18.71 | 21.87 |
| (2) + Complexity Control | No Failure | 20.72 | 25.12 |
| (3) + Scaling Up (100K→700K) | No Failure | 21.07 | **25.46** |

### Key Findings
- Significant improvements are observed across all benchmarks with MegaSynth: in-domain DL3DV +1.15dB, out-of-domain Hypersim +1.57dB, MipNeRF360 +1.26dB.
- Complexity control is key to training stability—without control, training collapses at 70k steps, whereas stable training is achieved with control.
- Geometric loss alone can delay training failure from 70k to 45k iterations, bringing a 3.4dB improvement after fine-tuning.
- Scaling the dataset from 100k to 700k yields a further 0.34dB improvement, validating the scaling effect.
- The model trained only on MegaSynth (PSNR=21.07) performs comparably to the model trained only on DL3DV, validating the non-semantic nature of 3D reconstruction.
- Indoor scenes benefit more than outdoor scenes (+1.34dB vs +0.80dB) because the room structures in MegaSynth are closer to real indoor scenes.

## Highlights & Insights
- **Insight: "3D Reconstruction Does Not Need Semantics"**: This is the core contribution of the paper—proving that multi-view reconstruction is a low-level geometric task, allowing a highly competitive model to be trained with random geometric primitives. This finding challenges the common assumption that synthetic data must be realistic.
- **Complementarity of Synthetic and Real Data**: Synthetic data provides scale + accurate metadata (especially geometric supervision), while real data provides domain adaptation. Combining both is superior to using either alone, without the need to chase high visual photorealism in synthetic data.
- **Transferability to Other Tasks**: The paper also demonstrates that MegaSynth can improve the performance of monocular depth estimation models, illustrating that the value of non-semantic synthetic data extends beyond multi-view reconstruction.

## Limitations & Future Work
- Non-semantic generation methods may be of limited help in reconstruction scenarios that demand semantic understanding (e.g., scene completion, occlusion reasoning).
- The current lighting model remains heavily simplified, lacking advanced lighting effects like global illumination (GI) and caustics.
- Although 700k scenes represents a massive scale, the geometric primitives used are limited in variety (cubes, spheres, cylinders, cones), which may impose a ceiling on geometric diversity.
- The hyperparameters of the camera sampling strategy still require manual tuning; more systematic distribution optimization approaches are worth exploring.

## Related Work & Insights
- **vs LRM-Zero**: LRM-Zero also uses primitive synthetic data, but is restricted to the object level. MegaSynth is the first to scale non-semantic synthesis to the more complex scene level, requiring the control of lighting, object compositions, and camera distributions.
- **vs DL3DV**: DL3DV is the largest real-world scene dataset (~10k scenes). MegaSynth scales this up by 50x, and joint training of the two yields the best performance.
- **vs Semantic Scene Generation Methods**: Methods like ProcTHOR and Infinigen strive for semantic correctness but suffer from slow generation speeds and restricted diversity. MegaSynth discards semantics to gain massive scalability.

## Rating
- Novelty: ⭐⭐⭐⭐ The core insight that "3D reconstruction does not need semantics" is highly inspiring, and the non-semantic synthesis approach is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive ablation analysis, multi-dataset validation, indoor/outdoor split evaluation, and cross-task transfer experiments.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and thorough experimentation, though some details of the data generation pipeline need to be checked in the appendix.
- Value: ⭐⭐⭐⭐⭐ Provides a low-cost, high-return data expansion route for the 3D reconstruction community, with a far-reaching impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] TokenUnify: Scaling Up Autoregressive Pretraining for Neuron Segmentation](../../ICCV2025/3d_vision/tokenunify_scaling_up_autoregressive_pretraining_for_neuron_segmentation.md)
- [\[CVPR 2025\] Regularizing INR with Diffusion Prior for Self-Supervised 3D Reconstruction of Neutron CT Data](regularizing_inr_with_diffusion_prior_self-supervised_3d_reconstruction_of_neutr.md)
- [\[CVPR 2025\] Scaling Mesh Generation via Compressive Tokenization](scaling_mesh_generation_via_compressive_tokenization.md)
- [\[CVPR 2025\] Scaling Properties of Diffusion Models for Perceptual Tasks](scaling_properties_of_diffusion_models_for_perceptual_tasks.md)
- [\[NeurIPS 2025\] RigAnyFace: Scaling Neural Facial Mesh Auto-Rigging with Unlabeled Data](../../NeurIPS2025/3d_vision/riganyface_scaling_neural_facial_mesh_auto-rigging_with_unlabeled_data.md)

</div>

<!-- RELATED:END -->

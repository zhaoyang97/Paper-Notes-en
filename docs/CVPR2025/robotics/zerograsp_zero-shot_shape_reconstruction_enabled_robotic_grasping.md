---
title: >-
  [Paper Note] ZeroGrasp: Zero-Shot Shape Reconstruction Enabled Robotic Grasping
description: >-
  [CVPR 2025][Robotics][Zero-shot grasping] ZeroGrasp proposes a unified framework based on an octree Conditional Variational Autoencoder (CVAE) to simultaneously perform high-resolution 3D object reconstruction and 6D grasp pose prediction from a single RGB-D image. By modeling inter-object relations with a multi-object encoder and 3D occlusion fields, it achieves SOTA performance on the GraspNet-1B benchmark and demonstrates generalization capabilities on real robots.
tags:
  - "CVPR 2025"
  - "Robotics"
  - "Zero-shot grasping"
  - "3D reconstruction"
  - "Octree"
  - "Occlusion reasoning"
  - "Grasp pose prediction"
date: 2026-05-08
content_hash: 335b4965c0f385d5
---

# ZeroGrasp: Zero-Shot Shape Reconstruction Enabled Robotic Grasping

**Conference**: CVPR 2025  
**arXiv**: [2504.10857](https://arxiv.org/abs/2504.10857)  
**Code**: [https://sh8.io/#/zerograsp](https://sh8.io/#/zerograsp)  
**Area**: 3D Vision / Robotic Grasping  
**Keywords**: Zero-shot grasping, 3D reconstruction, Octree, Occlusion reasoning, Grasp pose prediction

## TL;DR

ZeroGrasp proposes a unified framework based on an octree Conditional Variational Autoencoder (CVAE) to simultaneously perform high-resolution 3D object reconstruction and 6D grasp pose prediction from a single RGB-D image. By modeling inter-object relations with a multi-object encoder and 3D occlusion fields, it achieves SOTA performance on the GraspNet-1B benchmark and demonstrates generalization capabilities on real robots.

## Background & Motivation

**Background**: Robotic grasping requires an accurate understanding of the target object's geometry. Current mainstream methods mostly regress grasp poses directly from partial observations (such as point clouds or depth maps) without explicitly modeling the complete shape of the objects. A few methods leveraging multi-view reconstruction require capturing multiple additional images, which increases system complexity and computational overhead.

**Limitations of Prior Work**: Methods that do not model geometry are prone to accidental collisions and unstable grasp contacts. Multi-view reconstruction is unfeasible in confined spaces (such as shelves or bins). Furthermore, existing datasets are severely lacking in 3D shape annotations and physically valid grasp annotations, which limits the capability of single-view zero-shot grasping.

**Key Challenge**: Accurate grasping requires high-quality 3D reconstruction for physical constraints and collision detection. However, single-view reconstruction inherently faces massive uncertainty, especially with severe occlusions in multi-object cluttered scenes, and the two tasks of reconstruction and grasping have traditionally been treated separately.

**Goal**: (1) Achieve near real-time, high-resolution 3D reconstruction and 6D grasp pose prediction from a single RGB-D image; (2) handle occlusions and collisions in multi-object scenes; and (3) generalize to novel real-world objects using only synthetic training data.

**Key Insight**: The authors observe that sparse voxel representations (e.g., octrees) offer advantages in both speed and accuracy for single-view 3D reconstruction, and that modeling occlusion reasoning and inter-object spatial relations benefits both reconstruction and grasping.

**Core Idea**: Unify the tasks of reconstruction and grasping using an octree-based CVAE, routing them through a multi-object encoder to model collisions, introducing 3D occlusion fields to encode visibility information, and leveraging the reconstructed results to optimize grasp poses via contact constraints.

## Method

### Overall Architecture

The input is a single RGB-D image. First, SAM2 is used to generate 2D instance segmentation masks. These are combined with image features and the depth map to back-project each object into 3D space, constructing the input octrees. The input octrees pass through the CVAE's Prior network to extract latent features, which are then fed into a multi-object encoder (Transformer) to model inter-object relations, while 3D occlusion fields encode occlusion information. Finally, the Decoder predicts the complete octree for each object (including SDF, normals, and grasp poses). The predicted grasp poses can be further refined using contact constraints based on the reconstruction results. The entire inference process runs at approximately 5 FPS.

### Key Designs

1. **Octree-based CVAE**:

    - **Function**: Models probabilistic uncertainty in single-view 3D reconstruction while predicting 3D shapes and grasp poses simultaneously.
    - **Mechanism**: The Encoder receives the input and target octrees to predict the latent distribution, while the Prior network predicts the prior distribution solely from the input octree; their KL divergence is minimized during training. The Decoder predicts occupancy probabilities layer by layer, outputting SDF, normals, and grasp parameters (graspness, quality, view, angle, width, depth) at the final layer. An economical supervision strategy is adopted to learn grasp predictions only at valid grasp points.
    - **Design Motivation**: The hierarchical structure of the octree allows high-resolution reconstruction to be memory-efficient and fast. The probabilistic modeling of CVAE handles the inherent ambiguity of single-view reconstruction, generating more plausible complete shapes compared to deterministic methods.

2. **Multi-Object Encoder**:

    - **Function**: Models spatial relations among multiple objects in the scene to achieve collision-free reconstruction and grasp prediction.
    - **Mechanism**: Employs a $K$-layer standard Transformer block in the latent space. It takes the voxel centers and features of all objects as input tokens, allowing features of different objects to interact via self-attention with RoPE positional encoding. The input format is $[\ell_1, \ldots, \ell_L]$, containing the latent features of all objects simultaneously.
    - **Design Motivation**: Per-object prior networks lack global spatial awareness and cannot prevent inter-object collisions or overlaps. Performing attention in the latent space rather than the raw space significantly reduces computational cost.

3. **3D Occlusion Fields**:

    - **Function**: Localizes visibility/occlusion information to the voxel level to enhance the reconstruction quality of occluded regions.
    - **Mechanism**: Subdivides each latent voxel into $B^3$ sub-voxels, projects them onto the image plane, and determines via depth testing whether each sub-voxel is occluded by the target itself (self-occlusion flag $o_{\text{self}}$) or by other objects (inter-occlusion flag $o_{\text{inter}}$). The two concatenated flags are encoded into occlusion features using a 3-layer 3D CNN, which are then concatenated with the latent features.
    - **Design Motivation**: While the multi-object encoder primarily learns to avoid collisions (local context), occlusion modeling requires understanding global visibility relations, where occluders and occluded objects may be far apart. The 3D occlusion field localizes global information into each voxel via simplified ray casting, lowering learning difficulty.

### Loss & Training

The total loss is $\mathcal{L} = \mathcal{L}_{\text{rec}} + \mathcal{L}_{\text{grasp}} + \mathcal{L}_{\text{KL}}$:

- **Reconstruction loss** $\mathcal{L}_{\text{rec}}$: BCE for occupancy at each layer + L1 for SDF and normals at the final layer.
- **Grasp loss** $\mathcal{L}_{\text{grasp}}$: L1 for graspness + cross-entropy for quality/angle/width/depth.
- **KL divergence** $\mathcal{L}_{\text{KL}}$: Distribution matching between the Encoder and the Prior.

Additionally, the ZeroGrasp-11B synthetic dataset (1 million RGB-D images, 12K objects, and 11.3 billion physically valid grasp annotations) was created based on Objaverse-LVIS.

## Key Experimental Results

### Main Results

| Method | GraspNet-1B CD↓ | F1↑ | NC↑ | ReOcS-Easy CD↓ | ReOcS-Hard CD↓ |
|------|-----------------|-----|-----|-----------------|-----------------|
| Minkowski | 6.84 | 81.45 | 77.89 | 5.59 | 9.11 |
| OCNN | 7.23 | 82.22 | 78.44 | 5.26 | 8.69 |
| OctMAE | 7.57 | 78.38 | 75.19 | 5.53 | 6.76 |
| **ZeroGrasp** | **6.05** | **84.08** | **78.46** | **4.76** | **6.73** |

### Ablation Study

| Configuration | Description |
|------|------|
| Full model | Best across GraspNet-1B and all occlusion difficulty levels |
| w/o Multi-Object Encoder | Inter-object collisions increase, and reconstruction quality drops |
| w/o 3D Occlusion Fields | Reconstruction of occluded regions deteriorates significantly, especially in ReOcS Hard scenes |
| w/o Grasp Refinement | Grasp success rate drops, and collision detection fails |

### Key Findings

- The 3D occlusion fields contribute most to difficult occlusion scenarios, leading to significant improvements on ReOcS Hard.
- The grasp pose refinement algorithm (based on contact constraints and collision detection) effectively improves grasping accuracy utilizing reconstruction results—proving that high-quality reconstruction indeed reinforces grasping.
- Model trained solely on synthetic data generalizes well to real robots, successfully grasping novel objects.
- The inference speed is approximately 5 FPS, satisfying near real-time requirements.

## Highlights & Insights

- **Design philosophy of a unified framework**: Coupling rather than decoupling reconstruction and grasping allows the two tasks to reinforce each other—reconstruction provides collision detection and contact constraints, while the supervision signals from the grasping task indirectly improve surface quality. This concept of "shared representation, multi-task mutual benefit" is transferable to other robotic manipulation tasks.
- **Simplicity and effectiveness of 3D occlusion fields**: Using simple ray casting and binary flags localizes complex global visibility issues, avoiding complex volume rendering. This trick is applicable to any 3D task requiring occlusion awareness.
- **Simulation-to-Real zero-shot transfer**: The scale and diversity of the ZeroGrasp-11B dataset (12K objects, 11.3 billion grasps) are key to the successful zero-shot generalization.

## Limitations & Future Work

- Dependency on SAM2's instance segmentation quality—segmentation failures cascade into reconstruction and grasping failures.
- Currently supports only parallel-jaw grippers, without covering more complex end-effectors like dexterous hands.
- Although efficient, the octree representation still has resolution limits for extremely thin structures (e.g., paper edges).
- The simulation-to-real domain gap may be more pronounced under complex lighting and reflective materials.

## Related Work & Insights

- **vs OctMAE**: OctMAE performs scene-level reconstruction but does not segment or predict grasps. ZeroGrasp performs instance-level reconstruction and grasping, exhibiting superior performance in dense scenes through segment-based per-object processing and a multi-object encoder.
- **vs GraspNet / GSNet**: Traditional grasping methods do not explicitly reconstruct 3D shapes, and collision detection can only be done on partial point clouds. ZeroGrasp utilizes complete reconstructions to perform highly precise collision detection, fundamentally reducing collisional grasps.
- **vs FoundationPose**: Though both are zero-shot 3D methods, they target different goals—FoundationPose performs pose estimation, while ZeroGrasp performs shape reconstruction and grasping, making them complementary.

## Rating

- Novelty: ⭐⭐⭐⭐ The unified reconstruction and grasping framework is creative, and the 3D occlusion fields represent a novel design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers benchmark testing, ablation studies, and real robot experiments, alongside two self-constructed datasets.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with complete mathematical derivations.
- Value: ⭐⭐⭐⭐ Highly valuable to both the robotic grasping and 3D reconstruction communities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] DRAWER: Digital Reconstruction and Articulation with Environment Realism](drawer_digital_reconstruction_and_articulation_with_environment_realism.md)
- [\[CVPR 2025\] DexGrasp Anything: Towards Universal Robotic Dexterous Grasping with Physics Awareness](dexgrasp_anything_towards_universal_robotic_dexterous_grasping_with_physics_awar.md)
- [\[NeurIPS 2025\] Zero-Shot Context Generalization in Reinforcement Learning from Few Training Contexts](../../NeurIPS2025/robotics/zero-shot_context_generalization_in_reinforcement_learning_from_few_training_con.md)
- [\[ECCV 2024\] Prioritized Semantic Learning for Zero-shot Instance Navigation](../../ECCV2024/robotics/prioritized_semantic_learning_for_zero-shot_instance_navigation.md)
- [\[CVPR 2026\] Humanoid Generative Pre-Training for Zero-Shot Motion Tracking](../../CVPR2026/robotics/humanoid_generative_pre-training_for_zero-shot_motion_tracking.md)

</div>

<!-- RELATED:END -->

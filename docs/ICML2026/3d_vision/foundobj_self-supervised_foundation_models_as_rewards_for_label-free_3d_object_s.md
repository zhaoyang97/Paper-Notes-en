---
title: >-
  [Paper Note] FoundObj: Self-supervised Foundation Models as Rewards for Label-free 3D Object Segmentation
description: >-
  [ICML 2026][3D Vision][Label-free 3D Segmentation] This paper proposes FoundObj, which utilizes 2D/3D self-supervised foundation models (DINOv2 + TRELLIS) as rewarders. By employing a "superpoint merging + PPO" RL agent…
tags:
  - "ICML 2026"
  - "3D Vision"
  - "Label-free 3D Segmentation"
  - "Superpoint Merging"
  - "RL Reward"
  - "DINOv2"
  - "TRELLIS"
date: 2026-05-08
content_hash: 9764108a7b31ae70
---

# FoundObj: Self-supervised Foundation Models as Rewards for Label-free 3D Object Segmentation

**Conference**: ICML 2026  
**arXiv**: [2605.27178](https://arxiv.org/abs/2605.27178)  
**Code**: https://github.com/vLAR-group/FoundObj  
**Area**: 3D Vision / Unsupervised Learning / Reinforcement Learning  
**Keywords**: Label-free 3D Segmentation, Superpoint Merging, RL Reward, DINOv2, TRELLIS

## TL;DR
This paper proposes FoundObj, which utilizes 2D/3D self-supervised foundation models (DINOv2 + TRELLIS) as rewarders. By employing a "superpoint merging + PPO" RL agent, it achieves multi-class 3D object segmentation in complex indoor scenes without any scene-level human annotations, improving the unsupervised SOTA AP from 19.6 to 24.2 on ScanNet/S3DIS/ScanNet200.

## Background & Motivation

**Background**: Mainstream 3D scene object segmentation still relies on dense point-level annotations (e.g., Mask3D) or multi-modal alignment data (e.g., open-vocabulary methods derived from CLIP/SAM). To eliminate labels, two unsupervised routes have recently emerged: one projects 2D self-supervised semantics from DINO/v2 into 3D (UnScene3D, Part2Object); the other provides geometric priors using 3D object reconstruction/generation models (EFEM, GrabS, EvObj).

**Limitations of Prior Work**: Pure semantic routes (DINO-based) excel at distinguishing inter-class differences but struggle to separate adjacent objects of the same class (e.g., several chairs placed together) because DINO features lack "objectness." Pure geometric routes (GrabS/EFEM) can accurately extract objects with clear shapes but are specialized for single classes (e.g., chairs) and rely on dynamic cylindrical proxies, which cannot adapt to arbitrary shapes like flat cabinets against walls.

**Key Challenge**: What defines an "object"? Cognitive science suggests that object perception is defined by both geometry (shape/structure) and semantics (identity/distinction from background). Existing unsupervised methods utilize only one aspect, leading to inherent blind spots.

**Goal**: To segment (a) multiple adjacent instances of the same class and (b) objects with varying inter-class shapes (e.g., flat cabinets, tables, doors) without using any 3D scene annotations, while maintaining cross-dataset and long-tail generalization capabilities.

**Key Insight**: Allow an RL agent to "grow" object candidates bottom-up on the scene's superpoint graph. Two complementary foundation model rewards—TRELLIS (3D object geometric prior) and DINOv2 (2D semantic prior)—score candidates along independent axes of geometric consistency and semantic distinctiveness.

**Core Idea**: Transform the definition of "what is an object" from hand-crafted rules or single priors into "differentiable feedback provided by semantic + geometric foundation models," using PPO to train a superpoint merging policy.

## Method

### Overall Architecture
Input: A 3D scene point cloud $\bm{P}$ (and paired RGB-D images for projecting 2D DINOv2 features back to 3D).

The pipeline consists of three stages: (1) **Object Discovery Agent**: The Felzenszwalb algorithm partitions the scene into $K$ initial superpoints, and SparseConv extracts features for each. A seed policy network $\bm{\pi}_{seed}$ selects the starting point, and a merge policy network $\bm{\pi}_{merge}$ iteratively selects neighbors to merge, yielding candidates $\bm{s}_0 \to \bm{s}_1 \to \cdots \to \bm{s}_T$. (2) **Geometric Reward Module**: A TRELLIS pre-trained encoder is used with an "Object Centroid Field" head. DBSCAN verifies if candidate points converge to a single center; if so, +10 is rewarded, otherwise −1. (3) **Semantic Reward Module**: Using DINOv2 features projected to 3D, a "Semantic Consistency NCut" is performed. If the cost falls within the top-20 lowest bank per scene, +10 is rewarded, otherwise −1. The max of the two rewards is taken; the episode terminates once a +10 reward is achieved. The agent is trained using PPO. Finally, all pseudo-masks receiving +10 are fed to an independent Mask3D for downstream segmentation inference.

### Key Designs

1.  **Superpoint-level Object Discovery Agent (Replacing GrabS cylindrical proxy)**:
    - **Function**: Generates object candidates of arbitrary shape through the two-step action of seed selection + neighbor merging in the discrete space formed by $K$ superpoints.
    - **Mechanism**: The seed policy outputs softmax probabilities over all superpoint features $\bm{p}_{seed} = \bm{\pi}_{seed}([\bm{f}_1, \dots, \bm{f}_K])$ to sample a seed $\bm{s}_0$. The merge policy applies self-attention + sigmoid to the seed and its $Q$ neighbors within 0.1m, outputting merge probabilities $\bm{p}_{merge} = \bm{\pi}_{merge}(\bm{f}_0, [\bm{f}_0^1, \dots, \bm{f}_0^Q])$. Merges are sampled until a +10 reward is reached or the limit $T$ is hit.
    - **Design Motivation**: Cylindrical proxies (GrabS) naturally assume objects have regular shapes, failing on flat cabinets or L-shaped sofas. Discretizing the action space with superpoints and merging retains expressiveness for arbitrary shapes while making the RL search space tractable.

2.  **Geometric Reward: TRELLIS "Object Centroid Field" + DBSCAN Verification**:
    - **Function**: Determines if the candidate point cloud is geometrically "self-consistently converged into an object."
    - **Mechanism**: A Transformer decoder head $\bm{g}_{center}$ is added to the TRELLIS pre-trained encoder to regress vectors per point pointing to the object centroid $\bm{v}_m = \bm{o}_c - \bm{o}_m$, where $\bm{o}_c = \frac{1}{M}\sum \bm{o}_m$. This is trained on ABO + 3D-Future using $\ell_2$ loss. During inference, candidate $\bm{s}_t$ yields $\bm{v}_t$; DBSCAN ($r=0.05$) is run on $(\bm{s}_t + \bm{v}_t)$. If the main cluster coverage $\geq \alpha=30\%$, reward is +10, else −1.
    - **Design Motivation**: TRELLIS is an auto-encoder and cannot directly score "objectness." The centroid field explicitly models the geometric prior that "an object is a point cloud converging to a single point," avoiding the high cost of direct reconstruction comparison. DBSCAN provides robustness against partial occlusions and noise.

3.  **Semantic Reward: DINOv2 + Semantic Consistency NCut**:
    - **Function**: Determines if the candidate region is semantically distinct from the background.
    - **Mechanism**: DINOv2 2D features are projected to 3D via depth maps and averaged per superpoint to construct a scene-level $K \times K$ cosine similarity matrix $\mathcal{S}$ and a 0.1m adjacency matrix $\mathcal{A}$. The joint matrix $(\mathcal{S} * \mathcal{A})$ represents spatio-semantic composite similarity. Treat the candidate one-hot mask $O_t$ as a cut in the scene graph and calculate cost $\mathcal{C} = \mathcal{C}_{boundary} / \mathcal{C}_{vol}$ following NCut. Maintain a per-scene "top-20 lowest cost bank." If the candidate cost falls in the bank, reward is +10, else −1.
    - **Design Motivation**: Fixed thresholds fail across different scenes (furniture density varies). The adaptive bank keeps only the "20 most object-like candidates in the current scene," encouraging diverse exploration while avoiding low-quality noise.

### Loss & Training
The agent is trained using standard PPO. The geometric centroid head $\bm{g}_{center}$ is pre-trained offline on ABO + 3D-Future using $\ell_2$ loss. The semantic module is zero-parameter (acting only as a rewarder based on DINOv2 + bank). All candidates receiving +10 are used as pseudo-masks to supervised-train an independent Mask3D for inference (following the two-stage recipe of GrabS); only this lightweight Mask3D is run during benchmarking.

## Key Experimental Results

### Main Results

ScanNet Validation Set (18 classes, class-agnostic AP):

| Method | AP | AP@50 | AP@25 |
| :--- | :--- | :--- | :--- |
| EFEM (Geometric) | 8.0 | 16.7 | 22.3 |
| GrabS (Geometric, single-class) | 14.0 | 27.2 | 39.4 |
| UnScene3D (DINO-based) | 18.5 | 37.8 | 63.7 |
| Part2Object (DINOv2-based, Prev. SOTA) | 19.6 | 38.4 | 64.9 |
| Concerto+DINOv2 (Strong Baseline) | 19.8 | 41.2 | 72.2 |
| **Ours (FoundObj)** | **24.2** | **46.2** | **74.7** |
| Mask3D (Supervised Upper Bound) | 61.2 | 83.0 | 93.0 |

Cross-dataset Zero-shot (train on ScanNet, direct test): S3DIS-Area5 AP increased from 10.4 (Part2Object) to 12.8 (approaching the supervised Mask3D's 13.0); S3DIS 6-fold AP increased from 8.6 to 11.4; long-tail ScanNet200 AP increased from 15.2 to 18.1 (+19% relative gain).

### Ablation Study

| Configuration | AP | AP@50 | AP@25 | Description |
| :--- | :--- | :--- | :--- | :--- |
| Full FoundObj | 24.2 | 46.2 | 74.7 | Complete model |
| w/o Geometric Reward | 19.5 | 40.2 | 72.7 | Degrades to DINO-only, -4.7 AP |
| w/o Semantic Reward | 15.3 | 37.2 | 67.6 | Drops 8.9 AP, still higher than GrabS |
| DBSCAN $r=0.02$ | 21.9 | 43.6 | 76.8 | Too strict, misses objects |
| DBSCAN $r=0.1$ | 22.5 | 43.6 | 72.5 | Too loose, many false positives |
| Bank size = 10 | 20.9 | 41.8 | 72.1 | Repeatedly focuses on salient objects |
| Bank size = 40 | 21.1 | 41.4 | 73.3 | Low-quality candidates included |

Isolated prior comparison under fair conditions (Table 6 DINO-only): FoundObj using only DINO achieved 19.5 AP, comparable to Part2Object (19.6), and outperformed it in AP@50 (40.2 vs 38.4) and AP@25 (72.7 vs 64.9), indicating the RL agent utilizes the same features more effectively than NCut/projection pipelines.

### Key Findings
- Semantic rewards are more significant than geometric rewards (dropping 8.9 AP vs 4.7 AP), attributed to DINOv2's training data scale being much larger than TRELLIS, making features more discriminative.
- The true value of geometric rewards lies in complementing semantic weaknesses: DINO is helpless in splitting adjacent objects of the same class (multiple chairs), where the geometric centroid field provides the only splitting signal. This is the primary reason for the 7.8 point surge in ScanNet AP@50 (38.4 $\to$ 46.2).
- The "cost bank" is a crucial detail: replacing the global threshold with a scene-adaptive top-20 ensures no drop in zero-shot performance across ScanNet/S3DIS/ScanNet200.
- On S3DIS-Area5, unsupervised FoundObj (12.8) nearly matches supervised Mask3D (13.0), reflecting that supervised methods overfit the training domain while foundation model priors are more robust in out-of-distribution scenes.

## Highlights & Insights
- The "foundation models as rewarders" framework is highly transferable: instead of using foundation model features as inputs or contrastive targets, this paper proves that using them as RL "judges" is more elegant—the foundation models do not participate in gradients but guide a small agent through discrete +10/−1 signals.
- The "complementary max" combination of the two reward modules is clever: taking the max instead of a weighted sum is equivalent to "accept if either modality recognizes it as an object," allowing the agent to rely on geometry when semantics are ambiguous and vice versa.
- Superpoint + agent merging essentially transforms segmentation into "learnable hierarchical clustering on a graph." Compared to one-shot graph algorithms like NCut, the RL step-by-step merging allows the model to backtrack or avoid local errors.

## Limitations & Future Work
- Dependent on paired 2D images and depth (for DINOv2 projection)—applicability to pure LiDAR or unpaired RGB outdoor scenes (e.g., KITTI) is questionable.
- The geometric centroid field $\bm{g}_{center}$ is trained on furniture/common objects (ABO + 3D-Future); the centroid convergence assumption might fail for ultra-thin structures (curtains, wires) or ultra-large objects (floors, walls).
- Still relies on a "pseudo-label $\to$ Mask3D" two-stage approach: the agent's +10 reward is discrete and sparse; PPO training efficiency and stability were not fully discussed.
- Improvements: Introducing SAM3D into the semantic reward to replace DINOv2 projection might further mitigate 2D$\to$3D projection errors; introducing multi-view consistency constraints in the geometric reward could extend the method to outdoor scenes.

## Related Work & Insights
- **vs UnScene3D / Part2Object**: Similarly uses DINO/v2 semantic priors but they perform "feature projection + one-time NCut/grouping," unable to split adjacent same-class instances. FoundObj's RL agent + geometric reward achieves a 7.8 point gain in AP@50.
- **vs GrabS / EvObj / EFEM**: Similarly treats segmentation as "agents seeking objects," but GrabS uses cylindrical proxies limited to regular shapes. This paper uses a superpoint merging agent plus semantic rewards to cover multiple classes and arbitrary shapes.
- **vs Supervised Mask3D**: Matching Mask3D in the S3DIS Area-5 zero-shot scenario implies that "foundation model priors + unsupervised agents" can be preferable to "target domain supervised training" for cross-domain deployment.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The "foundation models as rewarders" framework and "centroid field + NCut bank" combination represent clean new designs, though individual components (PPO, DINO projection, NCut) are drawn from existing work.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Three benchmarks + cross-dataset zero-shot + long-tail + separate DINO/TRELLIS comparisons + 4 sets of ablations; only lacking training cost analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Clear logic from motivation to experiments; distinctions between geometric/semantic roles are well-articulated.
- **Value**: ⭐⭐⭐⭐ Unsupervised 3D segmentation SOTA + approaching supervised upper bound in cross-domain settings; offers methodological inspiration for other 3D/embodied tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Parameter-Free Fine-tuning via Redundancy Elimination for Vision Foundation Models](../../AAAI2026/3d_vision/parameter-free_fine-tuning_via_redundancy_elimination_for_vision_foundation_mode.md)
- [\[CVPR 2026\] MonoSAOD: Monocular 3D Object Detection with Sparsely Annotated Label](../../CVPR2026/3d_vision/monosaod_monocular_3d_object_detection_with_sparsely_annotated_label.md)
- [\[CVPR 2026\] Foundry: Distilling 3D Foundation Models for the Edge](../../CVPR2026/3d_vision/foundry_distilling_3d_foundation_models_for_the_edge.md)
- [\[ICML 2026\] Geometry-Guided Modeling of Foundation Features Enables Generalizable Object Shape Deformation Learning](geometry-guided_modeling_of_foundation_features_enables_generalizable_object_sha.md)
- [\[CVPR 2026\] ArtHOI: Taming Foundation Models for Monocular 4D Reconstruction of Hand-Articulated-Object Interactions](../../CVPR2026/3d_vision/arthoi_taming_foundation_models_for_monocular_4d_reconstruction_of_hand-articula.md)

</div>

<!-- RELATED:END -->

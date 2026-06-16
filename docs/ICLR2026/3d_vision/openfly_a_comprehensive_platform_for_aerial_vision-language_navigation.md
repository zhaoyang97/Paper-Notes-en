---
title: >-
  [Paper Note] OpenFly: A Comprehensive Platform for Aerial Vision-Language Navigation
description: >-
  [ICLR 2026][3D Vision][Aerial VLN] This paper presents OpenFly, a comprehensive platform for aerial vision-language navigation (VLN) that integrates four rendering engines (UE / GTA V / Google Earth / 3DGS)…
tags:
  - "ICLR 2026"
  - "3D Vision"
  - "Aerial VLN"
  - "UAV Navigation"
  - "Multi-Rendering Engine"
  - "Automatic Data Generation"
  - "Keyframe-Aware"
  - "3D Gaussian Splatting"
date: 2026-05-08
content_hash: 877aba9ec625f914
---

# OpenFly: A Comprehensive Platform for Aerial Vision-Language Navigation

**Conference**: ICLR 2026
**arXiv**: [2502.18041](https://arxiv.org/abs/2502.18041)  
**Code**: Available (open-source)  
**Area**: 3D Vision
**Keywords**: Aerial VLN, UAV Navigation, Multi-Rendering Engine, Automatic Data Generation, Keyframe-Aware, 3D Gaussian Splatting

## TL;DR

This paper presents OpenFly, a comprehensive platform for aerial vision-language navigation (VLN) that integrates four rendering engines (UE / GTA V / Google Earth / 3DGS), develops a fully automated data generation pipeline (point cloud acquisition → semantic segmentation → trajectory generation → GPT-4o instruction synthesis), constructs a large-scale dataset of 100K trajectories across 18 scenes, and proposes a keyframe-aware VLN model (OpenFly-Agent) combining keyframe selection with visual token merging. OpenFly-Agent outperforms existing methods by 14.0% and 7.9% in success rate on seen and unseen scenes, respectively.

## Background & Motivation

**Background**: VLN is a core task in embodied AI, requiring agents to navigate toward targets guided by language instructions and visual observations. Numerous indoor/ground-level datasets (R2R, RxR, TouchDown, VLN-CE, etc.) have driven methodological advances; however, VLN research for unmanned aerial vehicles (UAVs)—critical platforms for aerial photography, search-and-rescue, and cargo delivery—remains underdeveloped.

**Limitations of Prior Work**: AerialVLN and OpenUAV established preliminary aerial VLN datasets using AirSim and Unreal Engine simulators, yet they suffer from three fundamental challenges: limited data diversity, high collection cost, and small dataset scale.

**Data Diversity Bottleneck**: Existing approaches rely on AirSim and Unreal Engine to control UAVs and can only exploit digital assets compatible with these platforms, restricting both environmental diversity and photorealism and precluding the incorporation of additional high-fidelity data sources.

**High Manual Annotation Cost**: Trajectory generation requires trained pilots to fly UAVs within simulators, followed by annotators manually composing language instructions. The entire workflow is labor-intensive, time-consuming, and difficult to scale.

**Critically Insufficient Data Scale**: Current aerial VLN datasets contain only approximately 10K trajectories, far behind the robotics manipulation domain—Open X-Embodiment and EO-1 have collected over one million manipulation episodes—severely constraining model capacity.

**Key Challenge**: (1) Multi-rendering engine integration → addresses diversity; (2) fully automated pipeline → addresses cost; (3) 100K-scale dataset → addresses scale; (4) keyframe-aware model → addresses visual redundancy in long observation sequences.

## Method

### 1. Multi-Rendering Engine Integration

OpenFly integrates four rendering engines/technologies to substantially enrich scene resources:

- **Unreal Engine (UE)**: Provides 8 urban scenes covering more than $100 \text{ km}^2$, with rich assets including buildings, vehicles, and pedestrians.
- **GTA V**: Contributes highly realistic urban landscapes modeled after Los Angeles.
- **Google Earth**: Provides 4 urban regions (Berkeley / Osaka / Washington D.C. / St. Louis) covering $53.60 \text{ km}^2$.
- **3D Gaussian Splatting (3DGS)**: Employs hierarchical 3DGS to reconstruct 3D scenes from real UAV-captured images, covering more than $7 \text{ km}^2$ across 5 campus scenes, enabling real-to-sim rendering.

### 2. Automatic Data Generation Pipeline

The pipeline comprises four automated modules and three unified interfaces for controlling agent motion and acquiring sensor data.

**Point Cloud Acquisition**:
- Rasterized sampling reconstruction (UE / GTA V): local point clouds are captured at appropriately spaced sampling positions and merged.
- Image-based sparse reconstruction (3DGS): COLMAP generates sparse point clouds from input images.

**Semantic Segmentation** (three flexible approaches):
- 3D scene understanding: overhead-view image sequences are captured → Octree-Graph extracts semantic 3D instances.
- Point cloud projection + contour extraction: point clouds are voxelized and projected onto the ground plane → segmentation contours are extracted → GPT-4o annotates semantics.
- Manual annotation: fallback option when point cloud quality is poor or fine-grained segmentation is required.

**Automatic Trajectory Generation**:
- A global voxel map $M_{global}$ is constructed from the scene point cloud.
- Landmarks are randomly selected as targets; starting points are sampled at a certain distance, and endpoints are placed near the target landmark.
- Collision-free trajectories are searched using A* over $M_{global}$ with a custom action space.
- The endpoint of one trajectory is iteratively used as the starting point of the next, enabling complex trajectory generation.

**Automatic Instruction Generation**:
- Key strategy: the complete trajectory is segmented into sub-trajectories at action transition points, rather than feeding all frames to the model at once.
- Key actions and the last 3 frames of each sub-trajectory are extracted → GPT-4o generates sub-instructions.
- An LLM integrates all sub-instructions into a complete navigation instruction.
- A random sample of 3K instructions is manually verified → pass rate of 91%.

### 3. OpenFly-Agent: Keyframe-Aware VLN Model

Built upon OpenVLA, the core innovations are **keyframe selection** and **visual token merging (VTM)**:

**Keyframe Selection**:
- Motivation: uniform frame sampling is ill-suited for aerial VLN, as it may miss frames containing critical landmarks.
- Heuristic method: motion change points in UAV trajectories are identified → the change point and its two neighboring frames are extracted → these form the candidate keyframe set.
- Landmark localization module: a 3-layer cross-attention module fuses text and image features from LLM hidden states to predict the bounding box $\mathbf{b} \in \mathcal{R}^4$ of instruction-relevant landmarks.
- Filtering rule: candidate frames whose bounding box area exceeds threshold $\theta$ are retained as final keyframes.

**Visual Token Merging (VTM)**:
- The frame in the keyframe set with the largest bounding box is selected as the reference frame (containing the most salient landmark observation).
- Cosine similarity is densely computed between each visual token pair of the reference frame and other frames.
- Tokens with high similarity are merged by averaging; unmerged tokens from comparison frames are discarded.
- This process is applied iteratively until the entire keyframe set is traversed.
- A memory bank of capacity $K$ is maintained (FIFO policy retains the most recent keyframes).
- Tokens within keyframes are further compressed via grid pooling, while the current frame remains uncompressed to preserve the latest visual observation.

**Action Prediction**: The last 256 tokens of the vocabulary serve as action special tokens, defining 6 UAV actions: $\{$Forward, Turn Left, Turn Right, Move Up, Move Down, Stop$\}$.

## Key Experimental Results

### Table 1: VLN Dataset Comparison

| Dataset | Trajectories | Vocab Size | Path Length (m) | Instruction Length | Action Space | Environment |
|---|---|---|---|---|---|---|
| R2R | 7,189 | 3.1K | 10.0 | 29 | graph | Matterport3D |
| RxR | 13,992 | 7.0K | 14.9 | 129 | graph | Matterport3D |
| AerialVLN | 8,446 | 4.5K | 661.8 | 83 | 4 DoF | AirSim+UE |
| CityNav | 32,637 | 6.6K | 545 | 26 | 4 DoF | SensatUrban |
| OpenUAV | 12,149 | 10.8K | 255 | 104 | 6 DoF | AirSim+UE |
| **OpenFly** | **100K** | **15.6K** | **99.1** | **59** | **4 DoF** | **Multi-engine** |

### Table 2: Navigation Performance Comparison on Test Sets

| Method | NE↓(seen) | SR↑(seen) | OSR↑(seen) | SPL↑(seen) | NE↓(unseen) | SR↑(unseen) | OSR↑(unseen) | SPL↑(unseen) |
|---|---|---|---|---|---|---|---|---|
| Random | 242m | 0.7% | 0.8% | 0% | 301m | 0.1% | 0.1% | 0% |
| Seq2Seq | 205m | 2.9% | 24.3% | 2.6% | 229m | 2.1% | 20.6% | 1.1% |
| CMA | 161m | 5.4% | 28.1% | 4.8% | 217m | 4.6% | 24.4% | 2.1% |
| AerialVLN | 139m | 7.5% | 30.0% | 6.8% | 214m | 7.3% | 28.1% | 4.4% |
| Navid | 153m | 13.0% | 38.2% | 11.6% | 210m | 10.8% | 27.2% | 5.0% |
| NaVila | 132m | 20.3% | 53.5% | 17.8% | 202m | 14.7% | 42.1% | 9.6% |
| **OpenFly-Agent** | **93m** | **34.3%** | **64.3%** | **24.9%** | **154m** | **22.6%** | **56.2%** | **19.1%** |

### Table 3: Ablation Study (test-seen)

| Method | NE↓ | SR↑ | OSR↑ | SPL↑ |
|---|---|---|---|---|
| OpenVLA (baseline) | 231m | 2.3% | 10.8% | 2.2% |
| History (uniform sampling) | 223m | 6.9% | 23.3% | 5.6% |
| Random KS | 264m | 8.7% | 26.6% | 5.8% |
| KS (keyframe selection only) | 275m | 9.2% | 28.1% | 6.1% |
| History + VTM | 215m | 16.6% | 40.5% | 9.1% |
| **KS + VTM** | **93m** | **34.3%** | **64.3%** | **24.9%** |

## Key Findings

1. **Synergistic effect of keyframe selection and visual token merging is pronounced**: Using KS alone (SR 9.2%) or History+VTM alone (SR 16.6%) yields limited gains, whereas their combination (SR 34.3%) produces a super-linear improvement. VTM resolves the token imbalance between text and image modalities, preventing background noise from diluting attention to critical cues.

2. **Generalization advantage of multi-engine training data**: In real-world experiments across 23 scenes, models trained on OpenFly data (SR 26.09%, OSR 34.78%) substantially outperform those trained on AerialVLN data, confirming that multi-engine data effectively bridges the sim-to-real gap.

3. **Significant potential of VLMs for aerial VLN**: VLM-based methods (Navid / NaVila) markedly outperform traditional Seq2Seq / CMA approaches, particularly in Oracle SR (38–53% vs. 24–28%), underscoring the importance of VLM reasoning capabilities for navigation.

4. **Short-to-medium-range instructions better reflect real-world usage**: OpenFly's average trajectory length of 99.1 m and instruction length of 59 words are substantially lower than AerialVLN (661.8 m / 83 words). The authors argue this better aligns with natural human usage patterns and is more conducive to advancing aerial VLN.

5. **Reliable quality of automatically generated instructions**: The sub-trajectory segmentation strategy with GPT-4o and LLM integration achieves a 91% pass rate upon manual inspection of 3K randomly sampled instructions, while supporting high-throughput parallel generation.

## Highlights & Insights

- **System-level innovation rather than a single component breakthrough**: OpenFly's contribution lies not in any individual model component, but in the fully integrated platform combining four engines, an automated pipeline, a 100K-trajectory dataset, and a keyframe-aware model into a closed-loop system.
- **Real-to-sim application of 3DGS**: UAVs capture real images → 3DGS reconstruction → automatic training data generation within reconstructed scenes → deployment on real UAVs, validating a novel paradigm through closed-loop verification.
- **High engineering value**: Users can leverage the OpenFly pipeline to rapidly generate customized data for their own scenes, constituting an infrastructure-level contribution to the community.
- **Quantitative leap in data scale**: 100K trajectories (vs. the existing ~10K) for the first time brings aerial VLN data scale comparable to ground-level VLN, enabling effective transfer of OpenVLA at this scale.

## Limitations & Future Work

1. **Absolute success rates remain low**: Even the best-performing OpenFly-Agent achieves only 34.3% SR on test-seen and 22.6% on test-unseen, indicating that aerial VLN remains highly challenging and far from practical deployment.
2. **Limited cross-scene generalization**: All methods exhibit significant performance drops on unseen scenes (SR from 34.3% → 22.6%), with cross-scene generalization remaining a core bottleneck.
3. **Dependence on GPT-4o**: Instruction generation and semantic annotation rely on a commercial closed-source VLM, limiting cost efficiency and reproducibility.
4. **Simplified action space**: Fixed-step discrete actions (3 / 6 / 9 m) differ from the continuous control of real UAVs. Although continuous trajectory support is provided, primary experiments are conducted under the discrete action formulation.
5. **Google Earth data restricted to high-altitude views**: To ensure visual quality, Google Earth data is collected exclusively at high altitudes (4.46% of the dataset), limiting coverage of low-altitude real-world scenarios.

## Related Work & Insights

### vs. AerialVLN (ICCV 2023)
AerialVLN is the first aerial VLN dataset (8,446 trajectories), but uses a single AirSim+UE engine and relies on human pilots and manual annotation. OpenFly comprehensively surpasses it in rendering diversity (4 engines vs. 1), data scale (100K vs. 8.4K), and automation (fully automatic vs. manual). OpenFly-Agent achieves a 33% improvement in NE over the AerialVLN baseline (93 m vs. 139 m).

### vs. OpenUAV (2024)
OpenUAV similarly uses AirSim+UE to build a 12,149-trajectory VLN dataset and incorporates human feedback (RLHF) to guide navigation. However, it still depends on pilot operation and manual annotation, limiting data diversity. OpenFly's pipeline achieves fully automated data generation and introduces real-to-sim capability via 3DGS, demonstrating stronger transfer to real-world deployment.

### vs. CityNav (2024)
CityNav constructs 32,637 trajectories based on SensatUrban point cloud data and CityRefer language annotations, but relies on pre-existing 2D maps for landmark localization. OpenFly requires no external maps and navigates directly from a first-person perspective through an end-to-end vision-language approach, more closely reflecting practical UAV applications.

## Rating

- **Novelty**: ⭐⭐⭐⭐ System-level innovation through multi-engine integration, fully automated pipeline, and keyframe-aware design; individual technical contributions are incremental.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive evaluation including multi-method comparisons, ablations, real UAV deployment, cross-dataset comparison, and scale analysis.
- **Writing Quality**: ⭐⭐⭐⭐ System description is clear and complete, with rich figures and tables.
- **Value**: ⭐⭐⭐⭐⭐ Infrastructure-level contribution to aerial VLN research; the pipeline, dataset, and benchmark collectively form a self-contained ecosystem.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] 3D Gaussian Map with Open-Set Semantic Grouping for Vision-Language Navigation](../../ICCV2025/3d_vision/3d_gaussian_map_with_openset_semantic_grouping_for_visionlan.md)
- [\[ICLR 2026\] EgoNight: Towards Egocentric Vision Understanding at Night with a Challenging Benchmark](egonight_towards_egocentric_vision_understanding_at_night_with_a_challenging_ben.md)
- [\[CVPR 2026\] MonoVLM: Monocular 3D Visual Grounding with Vision Language Models](../../CVPR2026/3d_vision/monovlm_monocular_3d_visual_grounding_with_vision_language_models.md)
- [\[ICCV 2025\] CLIP-GS: Unifying Vision-Language Representation with 3D Gaussian Splatting](../../ICCV2025/3d_vision/clip-gs_unifying_vision-language_representation_with_3d_gaussian_splatting.md)
- [\[ICML 2026\] SVL: Spike-based Vision-Language Pretraining for Efficient 3D Open-World Understanding](../../ICML2026/3d_vision/svl_spike-based_vision-language_pretraining_for_efficient_3d_open-world_understa.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] M4Human: A Large-Scale Multimodal mmWave Radar Benchmark for Human Mesh Reconstruction
description: >-
  [CVPR 2026][Human Understanding][Paper Note] M4Human is the largest multimodal mmWave radar benchmark for Human Mesh Reconstruction (HMR) to date, featuring 661k frames, 50 actions, and 20 subjects. It provides synchronized RGB, Depth, Raw Radar Tensor (RT), and Radar Point Cloud (RPC) modalities with high-fidelity 3D mesh annotations based on optical motion capt
tags:
  - CVPR 2026
  - Human Understanding
date: 2026-05-08
content_hash: 5f226056262c8dd7
---
# M4Human: A Large-Scale Multimodal mmWave Radar Benchmark for Human Mesh Reconstruction

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Fan_M4Human_A_Large-Scale_Multimodal_mmWave_Radar_Benchmark_for_Human_Mesh_CVPR_2026_paper.html)  
**Code**: https://fanjunqiao.github.io/M4Human-site/  
**Area**: Human Understanding / Human Mesh Reconstruction / mmWave Radar Sensing / Multimodal Benchmark  
**Keywords**: mmWave Radar, Human Mesh Reconstruction, Multimodal Dataset, Raw Radar Tensor, Privacy-Preserving Sensing

## TL;DR
M4Human is the largest multimodal mmWave radar benchmark for Human Mesh Reconstruction (HMR) to date, featuring 661k frames, 50 actions, and 20 subjects. It provides synchronized RGB, Depth, Raw Radar Tensor (RT), and Radar Point Cloud (RPC) modalities with high-fidelity 3D mesh annotations based on optical motion capture (MoCap). It also introduces RT-Mesh, the first lightweight baseline for direct HMR from RT.

## Background & Motivation

**Background**: Human Mesh Reconstruction (HMR) enables dense recovery of human pose and shape, serving as a core component for human-computer interaction, rehabilitation monitoring, and VR fitness. Current HMR systems are predominantly built upon Line-of-Sight (LoS) RGB/depth cameras and large-scale video datasets (e.g., Human3.6M, 3DPW).

**Limitations of Prior Work**: Visual modalities face two critical challenges: exposure of personal appearance (unavailable in privacy-sensitive scenarios like childcare or elderly care) and vulnerability to lighting conditions (low light, strong sunlight) and occlusions (thick clothing, smoke). mmWave radar, as a radio frequency (RF) sensor, actively transmits signals and analyzes reflections, naturally resisting lighting/occlusion issues while protecting privacy. Its 77GHz frequency provides higher spatial resolution than other RF sensors like Wi-Fi, making it suitable for fine-grained HMR.

**Key Challenge**: Despite the potential of radar, existing radar-based human sensing datasets are primarily **designed for coarse Human Pose Estimation (HPE)** and exhibit three major drawbacks: ① Sparse annotations—most provide only skeletal joints, often estimated by noisy RGB(D) estimators or multi-view optimization, introducing ground-truth bias; ② Small scale and simple actions—existing RF-HMR datasets (e.g., mmMesh, mmBody) are limited to small scales and stationary daily actions; ③ Modality restrictions—most utilize low-resolution radar providing only sparse point clouds (RPC), discarding significant signal information, while the few works exploring Raw Radar Tensors (RT) are confined to HPE and do not support fine-grained HMR.

**Goal**: Construct a **large-scale, multimodal, high-fidelity annotated** radar HMR benchmark that covers diverse actions (stationary, sitting, and non-stationary) and provides both RT and RPC radar representations, complete with baseline methods and evaluation protocols.

**Core Idea**: Build a synchronized multimodal acquisition platform using "high-resolution mmWave radar + RGB-D + optical MoCap (Vicon)". Use **marker-based MoCap** to produce 3D mesh GT with higher accuracy than image-based estimation, release dual-level radar data (RT/RPC), and provide the first RT-based HMR baseline, RT-Mesh.

## Method

### Overall Architecture
M4Human is essentially a **dataset and benchmark**. Its contributions follow three lines: ① Data construction—synchronized acquisition of RGB, Depth, RT, and RPC. After calibration and synchronization, 37 markers are tracked via Vicon and reconstructed into SMPL-X high-fidelity mesh GT using SOMA with manual verification. ② Multi-level radar representation—release of information-rich RT and sparse RPC (filtered via CFAR) to allow experimentation across RF granularities. ③ Evaluation and Baselines—definition of three difficulty-stratified protocols (P1/P2/P3) and three data splits (Random/Cross-subject/Cross-action), alongside the first RT-based coarse-to-fine baseline, RT-Mesh.

### Key Designs

**1. Multimodal Platform and High-Fidelity Marker-based Mesh Annotation**

To address the issue of noisy image-based RF annotations, the platform integrates a high-resolution Vayyar vTrigB imaging radar (7GHz bandwidth, 20-element MIMO array) with an Intel RealSense D435 camera and a Vicon system. Extrinsic calibration between Vicon, camera, and radar is performed via PnP with markers and corner reflectors. Time synchronization uses a "fast head shake" trigger gesture, matching sensor frames to the nearest 100Hz Vicon frame. Mesh annotation follows a three-stage pipeline: Vicon tracking of 37 markers → Human-in-the-loop correction of swaps/occlusions → SOMA reconstruction of SMPL-X meshes. This marker-based process is significantly more accurate than pure image-based annotation.

**2. RT and RPC Dual-level Radar Representations**

Addressing the information loss in sparse point clouds, M4Human releases two complementary representations: **Raw Radar Tensor (RT)** is a 3D intensity volume mapped to Cartesian coordinates (X-Y-Z) after FFT processing; **Radar Point Cloud (RPC)** is generated by applying CFAR adaptive thresholding to RT. The authors define an **effective RPC ratio**—the percentage of points falling near the human body. M4Human achieves 87.0%, far exceeding mmBody's 6.2%, enabling concurrent global tracking and high-fidelity mesh reconstruction.

**3. RT-Mesh: First Baseline for HMR from Raw Radar Tensors**

Directly processing full-voxel 3D/4D RT is computationally prohibitive. RT-Mesh adopts a **coarse-to-fine two-stage** design. Inputting $T=4$ stacked RT frames as a 4D tensor $X_{RT} \in \mathbb{R}^{T \times X \times Y \times Z}$, it first compresses them into a 2D BEV representation. A lightweight **2D BEV Transformer** supervised by $L_{2D}$ locates the subject $(\hat{x}, \hat{y})$. Then, a local 3D RoI is cropped from the full RT and processed by a **3D Convolution + 3D Transformer** to extract mesh features. Finally, an HMR head regresses SMPL-X parameters: root orientation $\alpha \in \mathbb{R}^3$, shape $\beta \in \mathbb{R}^{10}$, global translation $\tau \in \mathbb{R}^3$, and body pose $\theta \in \mathbb{R}^{22\times3}$. This design achieves 2.74ms latency and 2.6 GFLOPs.

**4. Stratified Protocols and Generalization Splits**

Protocols are categorized by difficulty: P1 (30 stationary actions), P2 (5 sitting actions with self-occlusion/multipath noise), and P3 (all non-stationary actions with large displacements). Three split settings are used: S1 Random Split, S2 Cross-Subject, and S3 Cross-Action. Metrics include **MVE** (Mean Vertex Error), **MJE/MPJPE** (Mean Per Joint Position Error), **MRE** (Mean Rotation Error), and **TE** (Translation Error), all calculated in the world coordinate system without Procrustes alignment.

## Key Experimental Results

### Main Results: Radar Monomodal HMR Benchmark
Comparison of SOTA methods across RPC and RT representations.

| Modality | Method | Latency (ms) | GFLOPs | ALL-S1 (MVE) | ALL-S2 (MVE) | ALL-S3 (MVE) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| RPC | mm-Mesh | 3.53 | 2.87 | 132.7 | 170.1 | 173.8 |
| RPC | P4Trans. | 7.17 | 11.76 | 90.4 | 140.8 | 147.8 |
| RT | RT-Pose | 39.58 | 50.67 | 100.7 | 148.1 | 152.8 |
| RT | RETR | 17.87 | 3.01 | 97.1 | 169.7 | 163.1 |
| RT | **RT-Mesh (Ours)** | **2.74** | **2.60** | **90.9** | **135.1** | **143.1** |

In S1, both RPC and RT achieve MVE ~90mm. Crucially, **RT consistently outperforms RPC in cross-subject (S2) and cross-action (S3) settings**, as RT preserves dense spatial evidence while RPC is prone to missing limbs and overfitting.

### Comparison with RGB(D) and Multimodal Fusion

| Modality | S1-MVE | S1-TE | S2-MVE | S3-MVE |
| :--- | :--- | :--- | :--- | :--- |
| RGB | 97.5 | 54.4 | 149.7 | 116.7 |
| Depth | 82.7 | 45.6 | 127.1 | 123.2 |
| RPC | 90.4 | 49.4 | 140.8 | 147.8 |
| RT | 90.9 | 47.6 | 135.1 | 143.1 |
| **Depth + RT** | **77.5** | **36.0** | 115.9 | 120.0 |

Radar monomodality surpasses RGB in S1/S2 and matches Depth in TE, as radar is sensitive to moving foregrounds. Fusion of Depth + RT achieves the lowest MVE (77.5mm) and TE (36.0mm), proving radar is a strong complement to camera systems.

### Key Findings
- **Generalization Advantage of RT > RPC**: RT is more robust across subjects/actions due to the dense mapping of motion reflections.
- **Scaling Law**: RT-Mesh MVE improves significantly as training data increases from 25% to 100%, highlighting the importance of large-scale data.
- **Downstream Utility**: Mesh-derived skeletons support Human Action Recognition (HAR) with Top-1 64.82% (vs. GT 65.70%), validating the output quality.
- **Difficulty Stratification**: P2 and P3 are significantly harder than P1, with all modalities degrading in S2/S3, indicating generalization remains an open challenge.

## Highlights & Insights
- **Marker-based MoCap Standards**: Using Vicon + SOMA breaks the cycle of noisy image-based annotations in RF datasets.
- **Advancing RT for HMR**: Demonstrated that RT is deployable on edge devices while offering better generalization than RPC.
- **Effective RPC Ratio**: Quantifying human focus (87.0% vs. 6.2%) provides a practical metric for sensor selection.
- **Radar matches Depth in TE**: Radar's sensitivity to moving foregrounds suggests it is a powerful tracking complement rather than just an alternative.

## Limitations & Future Work
- **Baseline Nature**: RT-Mesh is a starting point; there is room for improvement in temporal consistency and human prior integration.
- **Generalization Gap**: Performance still drops significantly in S2/S3 settings.
- **Controlled Environment**: Data was collected indoors at 12Hz; outdoor scenarios and higher frame rates are not yet covered.

## Related Work & Insights
- **vs mmBody**: mmBody uses automotive radar with high background noise (6.2% effective RPC) and only provides RPC. M4Human is $9\times$ larger, provides RT+RPC, and highlights high human focus (87.0%).
- **vs mm-Fi/mRI/RT-Pose**: These focus on coarse HPE with noisy image-derived annotations. M4Human provides high-fidelity mesh GT and advances RT to the HMR task.

## Rating
- Novelty: ⭐⭐⭐⭐ (First large-scale RT-based HMR benchmark)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Multimodal, multi-protocol, scaling analysis)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, high information density)
- Value: ⭐⭐⭐⭐⭐ (Publicly available infrastructure for the community)

<!-- RELATED:START -->
<div class="related-papers" markdown="1">
</div>

## Related Papers

- [\[CVPR 2026\] OpenDance: Multimodal Controllable 3D Dance Generation with Large-scale Internet Data](opendance_multimodal_controllable_3d_dance_generation_with_large-scale_internet_.md)
- [\[CVPR 2026\] ImmerIris: A Large-Scale Dataset and Benchmark for Off-Axis and Unconstrained Iris Recognition in Immersive Applications](immeriris_a_large-scale_dataset_and_benchmark_for_off-axis_and_unconstrained_iri.md)
- [\[CVPR 2026\] LCA: Large-scale Codec Avatars - The Unreasonable Effectiveness of Large-scale Avatar Pretraining](lca_large-scale_codec_avatars_the_unreasonable_effectiveness_of_large-scale_avata.md)
- [\[CVPR 2026\] RoMo: A Large-Scale, Richly Organized Dataset and Semantic Taxonomy for Human Motion Generation](romo_a_large-scale_richly_organized_dataset_and_semantic_taxonomy_for_human_moti.md)
- [\[CVPR 2026\] MetricHMSR: Metric Human Mesh and Scene Recovery from Monocular Images](metrichmsr_metric_human_mesh_and_scene_recovery_from_monocular_images.md)

</div>

<!-- RELATED:END -->

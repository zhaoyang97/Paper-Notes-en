---
title: >-
  [Paper Note] PhysGaia: A Physics-Aware Benchmark with Multi-Body Interactions for Dynamic Novel View Synthesis
description: >-
  [CVPR 2026][3D Vision][Physics-aware benchmark] PhysGaia constructs a physics-aware benchmark dataset comprising 17 scenes that cover multi-body interactions across four material categories—liquid, gas, cloth…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "Physics-aware benchmark"
  - "dynamic novel view synthesis"
  - "multi-body interaction"
  - "4D Gaussian splatting"
  - "physical parameter evaluation"
date: 2026-05-08
content_hash: 6b2b68699ab0abd4
---

# PhysGaia: A Physics-Aware Benchmark with Multi-Body Interactions for Dynamic Novel View Synthesis

**Conference**: CVPR 2026
**arXiv**: [2506.02794](https://arxiv.org/abs/2506.02794)
**Code**: [https://cv.snu.ac.kr/research/PhysGaia/](https://cv.snu.ac.kr/research/PhysGaia/)
**Area**: 3D Vision / Dynamic Scene Reconstruction
**Keywords**: Physics-aware benchmark, dynamic novel view synthesis, multi-body interaction, 4D Gaussian splatting, physical parameter evaluation

## TL;DR
PhysGaia constructs a physics-aware benchmark dataset comprising 17 scenes that cover multi-body interactions across four material categories—liquid, gas, cloth, and rheological matter—providing ground truth 3D particle trajectories and physical parameters (e.g., viscosity). The paper further introduces two new metrics, Trajectory Distance (TD) and AUOP, to quantify the physical realism of 4DGS methods, revealing severe deficiencies in physical reasoning among existing DyNVS approaches.

## Background & Motivation

**Background**: Dynamic novel view synthesis (DyNVS) has been a prominent research direction in 3D vision. From NeRF to 4D Gaussian Splatting (4DGS), existing methods have made substantial progress in photorealism, enabling high-quality 4D spatiotemporal scene reconstruction from video input.

**Limitations of Prior Work**: (1) Existing DyNVS datasets (e.g., D-NeRF, Nerfies, DyCheck) focus almost exclusively on appearance reconstruction quality, with little consideration of physical realism. (2) The few physically oriented datasets (e.g., Spring-Gaus, PAC-NeRF, ScalarFlow) are limited to a single material type (only rheological matter or only gas) and single-object scenes, lacking multi-body interactions. (3) Real-world videos can capture complex scenes but cannot provide ground truth 3D trajectories or physical parameters, making quantitative evaluation of physical reasoning infeasible.

**Key Challenge**: DyNVS is evolving from "looking like" to "behaving like" real physics—from photorealism to physical realism—yet the benchmarks needed to support this transition are absent. What is required is a dataset that simultaneously encompasses complex multi-body interactions, diverse material types, and reliable physical ground truth.

**Goal**: (1) Construct a physics-aware dataset featuring multi-body interaction scenes across four material categories; (2) provide complete ground truth (3D trajectories and physical parameters); (3) design physical realism metrics; (4) expose the physical limitations of existing methods.

**Key Insight**: Professional physics solvers (FLIP, Pyro, Vellum, MPM) are employed to ensure that each scene strictly adheres to the laws of physics, generating synthetic data with accurate physical ground truth.

**Core Idea**: Use material-specific physics solvers to generate a multi-body interaction benchmark dataset, and evaluate DyNVS methods along both photometric and physical dimensions.

## Method

### Overall Architecture
PhysGaia is constructed at three levels: (1) Scene design and physics simulation—appropriate solvers are selected for each of the four material categories to generate 17 multi-body interaction scenes; (2) Data collection—multi-view and monocular videos, depth maps, normal maps, 3D particle trajectories, and physical parameters are extracted from the simulations; (3) Evaluation framework—in addition to standard photometric metrics (PSNR/SSIM/LPIPS), two new physical metrics, TD and AUOP, are introduced.

### Key Designs

1. **Material-Specific Solvers**:

    - Function: Ensure physically accurate behavior for each material type.
    - Mechanism: The FLIP (Fluid-Implicit Particle) solver is used for liquids, Pyro for gases, Vellum for cloth, and MPM (Material Point Method) for rheological matter. Each solver represents best practice for its material class—for instance, FLIP's hybrid particle-grid representation offers superior stability for incompressible fluids compared to pure-particle SPH, while Pyro's voxel grid accurately captures thermodynamic effects such as temperature and buoyancy.
    - Design Motivation: Existing physics-integrated 4DGS works almost universally rely on MPM; however, MPM is fundamentally better suited to solids and rheological matter, and applying it to liquids or gases yields suboptimal accuracy and stability. Different materials require different solvers.

2. **Trajectory Distance (TD) Metric**:

    - Function: Quantify the spatial deviation between reconstructed particle trajectories and ground truth trajectories.
    - Mechanism: Given $M$ reconstructed Gaussian primitives over $T$ frames, each reconstructed primitive $i$ is matched to its corresponding ground truth trajectory $j(i)$ via nearest-neighbor assignment in the initial frame. The average Euclidean distance over the entire temporal sequence is then computed: $\text{TD} = \frac{1}{MT}\sum_{i}\sum_{t}\|X_i^{t,\text{recon}} - X_{j(i)}^{t,\text{gt}}\|_2$
    - Design Motivation: Traditional photometric metrics (PSNR, SSIM) evaluate only rendered image quality and cannot reflect whether the actual 3D motion of particles conforms to physics—a particle may be rendered at the "correct" location while its underlying trajectory is entirely wrong.

3. **Area Under the Outlier Percentages (AUOP) Metric**:

    - Function: Detect and quantify the persistence and extent of trajectory deviations.
    - Mechanism: For each primitive at each time step, the metric determines whether its deviation exceeds a threshold $\delta$. A key design choice is that once a primitive is flagged as an outlier, it remains an outlier permanently ($O_i^t = 1$ if $O_i^{t-1}=1$ or if the current deviation exceeds the threshold). The area under the curve of the outlier ratio over time is then computed. This captures the **cumulative effect** of physical deviation, analogous to the butterfly effect in chaotic systems.
    - Design Motivation: TD is a global average and can be pulled down by a small number of accurate trajectories. AUOP instead focuses on "how many trajectories begin to deviate from physics at some point and continue to do so," providing a more informative measure of the severity of physical reasoning failures.

### Loss & Training
PhysGaia is a dataset rather than a model; no training is involved. Its core contributions lie in providing COLMAP-reconstructed point clouds and an integration pipeline compatible with five 4DGS methods, enabling researchers to use the benchmark directly.

## Key Experimental Results

### Main Results (Photometric Quality — Average per Material under Monocular Setting)

| Method | Liquid PSNR↑ | Gas PSNR↑ | Rheological PSNR↑ | Cloth PSNR↑ |
|--------|-------------|----------|------------------|------------|
| D-3DGS | 22.7 | 21.9 | 20.1 | 22.1 |
| 4DGS | 24.2 | 21.7 | 19.5 | 24.9 |
| STG | 19.2 | 21.9 | 13.6 | 21.9 |
| MoSca | 20.5 | 21.2 | 17.8 | 18.6 |
| SoM | 19.6 | 20.0 | 16.7 | 20.7 |

### Ablation Study (Comparison with Existing Physics Benchmarks)

| Benchmark | Multi-Body | Dynamic Score↑ | FID↓ | KID↓ |
|-----------|-----------|---------------|------|------|
| ScalarFlow | No | 0.391 | 293.5 | 0.255 |
| PAC-NeRF | No | N/A | 242.6 | 0.164 |
| Spring-Gaus | No | 0.372 | 261.8 | 0.171 |
| **PhysGaia** | **Yes** | **0.444** | **207.8** | **0.118** |

### Key Findings
- **All existing methods perform substantially worse on PhysGaia than on conventional datasets**: Even under multi-view settings, the average PSNR falls below 30, far short of performance on D-NeRF (35+). The root cause is that multi-body interactions introduce motion complexity that far exceeds single-object deformation.
- **Rheological matter scenes are the most challenging**: PSNR is the lowest across all methods (STG achieves only 13.6), as rheological interactions (e.g., jelly collisions) involve complex dynamics among multiple components that polynomial motion models or ARAP constraints cannot capture.
- **Needle-like artifacts are a pervasive problem**: In multi-body collision scenes such as "jelly party," all methods produce severe geometric artifacts, indicating that none can maintain reasonable Gaussian distributions in physical contact regions.
- **PhysGaia achieves the highest Dynamic Score (0.444)**, confirming that its motion complexity significantly exceeds that of existing benchmarks.

## Highlights & Insights
- **The "once an outlier, always an outlier" design of AUOP is particularly elegant**: It captures a fundamental phenomenon in physics simulation—initial deviations amplify exponentially across subsequent frames. This irreversible outlier-flagging strategy reflects the reliability of physical reasoning more faithfully than simple per-frame error statistics.
- **The material-specific solver approach carries important implications for 4DGS research**: Nearly all physics-aware 4DGS methods currently rely on MPM, yet liquids (FLIP), gases (voxel grids with thermodynamics), and cloth (Vellum/PBD) each require different solvers—pointing to a largely overlooked research direction.
- **Complete simulation node graphs and source files are provided**, allowing users to generate data at higher resolutions and with additional modalities (depth, normals, relighting), greatly enhancing the benchmark's extensibility.

## Limitations & Future Work
- **Only 17 scenes**: The number of scenes is relatively small relative to the data requirements of deep learning models; certain material categories (e.g., gas) contain only 2–3 scenes.
- **Domain gap between synthetic and real data**: Although FID/KID scores indicate high visual fidelity, differences in texture and lighting distribution between synthetic rendering and real-world capture remain.
- **Absence of rigid-body collision scenes**: All four material categories are deformable, with no rigid-body collision scenarios, which are equally important for applications such as robotic manipulation.
- **Initial-frame matching assumption in TD**: If reconstructed Gaussian primitives have large positional errors in the initial frame, nearest-neighbor matching may produce incorrect trajectory correspondences.

## Related Work & Insights
- **vs Spring-Gaus**: Spring-Gaus is limited to rheological matter and single objects; PhysGaia extends coverage to four material types and multi-body interactions while providing richer ground truth.
- **vs PAC-NeRF**: The liquid scenes in PAC-NeRF are actually high-viscosity fluids that behave more like elastic solids; PhysGaia employs a FLIP solver to simulate genuinely low-viscosity liquids.
- **vs Phystwin**: Phystwin contains 22 scenes but is restricted to rheological matter and provides no ground truth physical parameters; PhysGaia is superior in terms of physical information completeness.
- **Insights**: This benchmark can drive physics-aware 4DGS research from single-material toward multi-material, multi-body interaction settings, particularly regarding how to integrate multiple physics solvers within a unified framework.

## Rating
- Novelty: ⭐⭐⭐⭐ First physics-aware DyNVS benchmark covering multiple materials and multi-body interactions; the AUOP metric design is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Five mainstream 4DGS methods are evaluated and comparisons with three physics benchmarks are provided, though the analysis of TD/AUOP metrics is relatively brief.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with strong visualizations and well-justified rationale for solver selection.
- Value: ⭐⭐⭐⭐ Points the way for physics-aware dynamic scene reconstruction, though the limited number of scenes constrains practical applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Dynamic Novel View Synthesis in High Dynamic Range](../../ICLR2026/3d_vision/dynamic_novel_view_synthesis_in_high_dynamic_range.md)
- [\[CVPR 2026\] MoVieS: Motion-Aware 4D Dynamic View Synthesis in One Second](movies_motion-aware_4d_dynamic_view_synthesis_in_one_second.md)
- [\[CVPR 2026\] Physically Inspired Gaussian Splatting for HDR Novel View Synthesis](physically_inspired_gaussian_splatting_for_hdr_novel_view_synthesis.md)
- [\[CVPR 2026\] GeodesicNVS: Probability Density Geodesic Flow Matching for Novel View Synthesis](geodesicnvs_probability_density_geodesic_flow_matching_for_novel_view_synthesis.md)
- [\[CVPR 2026\] PR-IQA: Partial-Reference Image Quality Assessment for Diffusion-Based Novel View Synthesis](pr-iqa_partial-reference_image_quality_assessment_for_diffusion-based_novel_view.md)

</div>

<!-- RELATED:END -->

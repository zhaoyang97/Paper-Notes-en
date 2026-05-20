---
title: >-
  [Paper Note] UDC-VIT: A Real-World Video Dataset for Under-Display Cameras
description: >-
  [ICCV 2025][Human Understanding][Under-display camera] This paper presents UDC-VIT, the first real-world video dataset for under-display cameras (UDC), comprising 647 video clips with 116…
tags:
  - "ICCV 2025"
  - "Human Understanding"
  - "Under-display camera"
  - "video dataset"
  - "image degradation"
  - "face recognition"
  - "video restoration"
date: 2026-05-08
content_hash: 8dcea759d349eee9
---

# UDC-VIT: A Real-World Video Dataset for Under-Display Cameras

**Conference**: ICCV 2025
**arXiv**: [2501.18545](https://arxiv.org/abs/2501.18545)  
**Code**: [GitHub](https://github.com/mcrl/UDC-VIT)  
**Area**: Human Understanding
**Keywords**: Under-display camera, video dataset, image degradation, face recognition, video restoration

## TL;DR

This paper presents UDC-VIT, the first real-world video dataset for under-display cameras (UDC), comprising 647 video clips with 116,460 frames in total. A carefully designed dual-camera beam-splitter acquisition system achieves precise spatiotemporal alignment. With face recognition as the primary application scenario, the dataset reveals the inadequacy of synthetic datasets in simulating real-world UDC degradation.

## Background & Motivation

Under-display cameras (UDC) place the camera module beneath the display panel to achieve a full-screen design, and have been adopted by devices such as the Samsung Galaxy Z-Fold series and ZTE Axon series. However, light diffraction caused by the display panel introduces severe degradation, including reduced transmittance, blur, noise, and flare.

**Key limitations of existing datasets**:

**Limitations of synthetic datasets**:
   - T-OLED/P-OLED datasets are captured in controlled environments by displaying images on a monitor, resulting in limited dynamic range and virtually no flare.
   - SYNTH generates degraded images via convolution with a measured PSF, but lacks noise and spatially varying flare.
   - VidUDC33K simulates video degradation via PSF convolution, but the flare is overly regular and light-source-independent, and exhibits unreasonable white artifacts.

**Absence of real-world video datasets**: Existing real-world datasets such as UDC-SIT contain only static images. Video adds a temporal dimension beyond images, involving motion-induced temporally varying flare that cannot be accurately simulated with synthetic data.

**Absence of face recognition data**: Subjects in existing datasets are mostly captured from a distance or from behind, making face recognition research infeasible.

This paper constructs UDC-VIT, the first video dataset with real-world UDC degradation, at a resolution of 1900×1060 at 60 fps, with 64.6% of videos containing frontal human actions from 22 subjects.

## Method

### Overall Architecture

The core contribution of this paper is the design of the data acquisition system and the construction of the dataset, encompassing hardware design (dual-camera + beam splitter), software synchronization, frame alignment, and quality assessment.

### Key Designs

1. **Dual-camera video acquisition system**:

    - Function: Simultaneously capture UDC-degraded video and clean reference video of the same scene.
    - Mechanism: A non-polarizing cubic beam splitter (Thorlabs CCM1-BS013) splits incident light at a 50:50 ratio into two paths, each directed to an Arducam Hawk-Eye (IMX686) camera module. A UDC display panel cut from a Samsung Galaxy Z-Fold 5 is placed in front of one path to introduce degradation. Both cameras are connected to a Raspberry Pi 5 via its dual quad-channel MIPI interface, with frame synchronization achieved through an MPI barrier to within 8 ms. Each camera is mounted on a Thorlabs K6XS six-axis kinematic optical mount, enabling translation, rotation, and tilt adjustments for field-of-view alignment.
    - Design Motivation: Using the same Quad Bayer Coding (QBC) sensor as the Galaxy Z-Fold 5 ensures consistent degradation characteristics. The beam-splitter approach provides better geometric alignment than dual-camera setups (e.g., Pseudo-real dataset).

2. **DFT-based frame alignment**:

    - Function: Correct inevitable pixel-level misalignments during acquisition.
    - Mechanism: Degradation-robust alignment is performed using the Discrete Fourier Transform (DFT). GT frames are center-cropped to 1900×1060, and degraded frames are aligned through iterative translation and rotation to minimize the following alignment loss:
    $\mathcal{L} = \lambda_1 \sum_{x,y} (\mathcal{D}(x,y) - \mathcal{G}(x,y))^2 + \lambda_2 \sum_{u,v} \Delta\mathcal{F}_{amp}(u,v) + \lambda_3 \sum_{u,v} \Delta\phi(u,v)$
      where the first term is spatial-domain MSE, and the latter two are L1 distances in the frequency-domain amplitude and phase, respectively, with $\lambda_1 = \lambda_3 = 1, \lambda_2 = 0$.
    - Design Motivation: Conventional alignment methods (SIFT, RANSAC) perform poorly under severe UDC degradation, particularly flare; DFT is more robust to such degradation.

3. **Dataset characteristics and real-world degradation analysis**:

    - Function: Systematically compare UDC-VIT with existing datasets in terms of noise, transmittance, flare, and other properties.
    - Key Findings:
        - **Noise and transmittance**: In VidUDC33K, the noise level of degraded frames is paradoxically lower than that of GT frames (unrealistic), whereas UDC-VIT correctly reflects the signal amplification and increased noise resulting from the low transmittance of the UDC region.
        - **Spatially varying flare**: UDC degradation intensifies progressively from the lens center outward, causing flare to be spatially variant. VidUDC33K applies the same PSF convolution across the entire image and thus cannot reproduce this property.
        - **Light-source-dependent flare**: Different light sources (LED, halogen, natural light) produce differently shaped flare patterns, which VidUDC33K cannot simulate.
        - **Temporally varying flare**: Camera motion causes PSF variations that UDC-VIT naturally captures, whereas VidUDC33K's simulation is nearly ineffective in this regard.

### Loss & Training

This paper presents a dataset, and no new model training is involved. The six deep learning models used for evaluation all employ their respective original training strategies.

## Key Experimental Results

### Main Results

Restoration performance of six deep learning models on VidUDC33K and UDC-VIT:

| Model | VidUDC33K PSNR↑ | VidUDC33K SSIM↑ | UDC-VIT PSNR↑ | UDC-VIT SSIM↑ | UDC-VIT LPIPS↓ |
|------|----------------|----------------|--------------|--------------|---------------|
| Input | 26.22 | 0.8524 | 16.26 | 0.7366 | 0.4117 |
| DISCNet | 28.89 | 0.8405 | 24.70 | 0.8403 | 0.2675 |
| UDC-UNet | 28.37 | 0.8361 | **28.00** | **0.8911** | **0.1779** |
| FastDVDNet | 28.95 | 0.8638 | 23.89 | 0.8439 | 0.2662 |
| EDVR | 28.71 | 0.8531 | 23.55 | 0.8331 | 0.2673 |
| ESTRNN | 29.54 | 0.8744 | 25.38 | 0.8654 | 0.2216 |
| DDRNet | **31.91** | **0.9313** | 24.68 | 0.8539 | 0.2218 |

DDRNet achieves the best performance on synthetic data (31.91 dB) but only 24.68 dB on real-world data, revealing the limitations of synthetic training data.

### Ablation Study

Face recognition accuracy as a function of restoration quality:

| Condition | PSNR | Face Recognition Accuracy | Note |
|------|------|-------------|------|
| Input (degraded frames) | 16.31 | 64.5% | No restoration |
| DISCNet restored | ~24.7 | ~75% | Static image model |
| UDC-UNet restored | 27.74 | 82.2% | Best restoration model |
| GT (reference frames) | ∞ | 90.3% | Upper bound |

Alignment quality (PCK comparison):

| Dataset | Alignment Method | PCK₀.₀₀₃ | PCK₀.₀₁ | PCK₀.₁₀ |
|--------|---------|----------|---------|---------|
| Pseudo-real | AlignFormer | N/A | 58.75 | 99.93 |
| UDC-SIT | DFT | 93.67 | 97.26 | 99.35 |
| **UDC-VIT** | **DFT** | **92.12** | **98.95** | **99.69** |

### Key Findings

- **Models trained on synthetic data are unreliable on real-world data**: DDRNet performs best on synthetic VidUDC33K (31.91 dB) but drops substantially in rank on real-world UDC-VIT, where it is significantly outperformed by UDC-UNet.
- **Strong correlation between restoration quality and face recognition**: As PSNR improves from 16.31 to 27.74 dB, face recognition accuracy increases from 64.5% to 82.2%.
- **Inconsistent model rankings across datasets**: The model rankings differ between the two datasets, demonstrating that synthetic data cannot accurately reflect real-world degradation characteristics.
- **Residual connections benefit temporal consistency**: UDC-UNet and ESTRNN, which use residual CNNs, exhibit better performance in reducing flickering artifacts.
- VidUDC33K contains unrealistic scenes (e.g., flare over ocean waves, flare on a bird's beak), as well as black frames and color distortions caused by erroneous PSF transformations.

## Highlights & Insights

1. **Hardware–software co-designed acquisition system**: The combination of a six-axis kinematic mount, beam splitter, and MPI synchronization is elegantly engineered; an 8 ms synchronization accuracy is highly precise for video acquisition.
2. **Systematic analysis of real-world flare characteristics**: The comparative analysis of spatially varying, light-source-dependent, and temporally varying flare is highly persuasive and clearly exposes the fundamental shortcomings of synthetic approaches.
3. **Application-driven dataset design**: The dataset is specifically designed for face recognition (64.6% of videos contain faces), and the impact of restoration quality on face recognition is quantified.
4. **Publicly available dataset**: Made openly accessible via the GitHub repository.

## Limitations & Future Work

- The dataset targets only the Samsung Galaxy Z-Fold 5 UDC panel; degradation characteristics differ across other devices (e.g., ZTE Axon series or other Fold models), necessitating transfer learning.
- Videos involving fast-moving objects (e.g., moving vehicles) are excluded, limiting the dataset's applicability in high-speed scenarios.
- UDC restoration is inherently device-dependent (contingent on optics, sensor, and panel design), and the design of generalizable software solutions remains an open problem.
- The face data covering 22 subjects is relatively limited in scale; future work could expand the dataset size.

## Related Work & Insights

- The beam-splitter acquisition system design paradigm can be generalized to the construction of other datasets requiring matched degraded/clean pairs.
- The robustness of DFT-based alignment under severe degradation warrants exploration in other image registration tasks.
- The quantified relationship between restoration quality and downstream task (face recognition) performance opens a new research direction for end-to-end optimization.

## Rating

- Novelty: ⭐⭐⭐⭐ First real-world UDC video dataset with an elegantly designed acquisition system.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive comparison across 6 models and 2 datasets; face recognition evaluation is innovative.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure; in-depth and intuitive analysis of existing dataset limitations.
- Value: ⭐⭐⭐⭐ Fills the gap of real-world UDC video datasets and makes a significant contribution to the UDC research community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] OnlineHMR: Video-based Online World-Grounded Human Mesh Recovery](../../CVPR2026/human_understanding/onlinehmr_video-based_online_world-grounded_human_mesh_recovery.md)
- [\[ICCV 2025\] MagShield: Towards Better Robustness in Sparse Inertial Motion Capture Under Magnetic Disturbances](magshield_towards_better_robustness_in_sparse_inertial_motion_capture_under_magn.md)
- [\[ICCV 2025\] Bring Your Rear Cameras for Egocentric 3D Human Pose Estimation](bring_your_rear_cameras_for_egocentric_3d_human_pose_estimation.md)
- [\[ICCV 2025\] HUMOTO: A 4D Dataset of Mocap Human Object Interactions](humoto_a_4d_dataset_of_mocap_human_object_interactions.md)
- [\[ICCV 2025\] MDD: A Dataset for Text-and-Music Conditioned Duet Dance Generation](mdd_a_dataset_for_text-and-music_conditioned_duet_dance_generation.md)

</div>

<!-- RELATED:END -->

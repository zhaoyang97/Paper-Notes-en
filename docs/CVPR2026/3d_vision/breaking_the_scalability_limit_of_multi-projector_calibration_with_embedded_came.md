---
title: >-
  [Paper Note] Breaking the Scalability Limit of Multi-Projector Calibration with Embedded Cameras
description: >-
  [CVPR 2026][3D Vision][Multi-projector calibration] By embedding several cameras directly into the calibration board surface with their optical centers aligned with the calibration plane, all projectors can project structured light "simultaneously." The overlapping patterns are separated and decoded based on the incident light direction, compressing the projection-capture steps required for multi-projector calibration from "linear scaling with the number of projectors" to "ne…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "Multi-projector calibration"
  - "Embedded camera"
  - "Structured light"
  - "Directional encoding"
  - "Homography compensation"
date: 2026-05-08
content_hash: 0e168d7a796354f5
---

# Breaking the Scalability Limit of Multi-Projector Calibration with Embedded Cameras

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Kawano_Breaking_the_Scalability_Limit_of_Multi-Projector_Calibration_with_Embedded_Cameras_CVPR_2026_paper.html)  
**Code**: None  
**Area**: 3D Vision / Computational Imaging / Projection Mapping  
**Keywords**: Multi-projector calibration, Embedded camera, Structured light, Directional encoding, Homography compensation

## TL;DR
By embedding several cameras directly into the calibration board surface with their optical centers aligned with the calibration plane, all projectors can project structured light "simultaneously." The overlapping patterns are separated and decoded based on the incident light direction, compressing the projection-capture steps required for multi-projector calibration from "linear scaling with the number of projectors" to "near-constant" — reducing 25-projector calibration from 1100 patterns in 12 minutes to just 54 patterns, with accuracy comparable to traditional one-by-one projector calibration.

## Background & Motivation
**Background**: To align virtual content accurately onto physical surfaces, Projection Mapping (PM) requires geometric calibration for each projector. Projectors can be modeled as "inverse cameras," adopting the pinhole camera model $s[x\;y\;1]^T = K[R\,|\,t][X\;Y\;Z\;1]^T$ for calibration. However, because projectors cannot directly observe where their output pixels land, calibration is done indirectly: projecting structured light (e.g., Gray-code) onto a calibration board and capturing it with an external camera to establish correspondences between projector pixels and board coordinates.

**Limitations of Prior Work**: While this works well for a single projector, it fails for multiple projectors. When multiple projectors project structured light *simultaneously*, their patterns overlap on the board. Since the calibration board is a diffuse surface, the directional information of each light ray is lost upon reflection, making the overlapping patterns indistinguishable to an external camera. Consequently, traditional methods must rely on *sequential, one-by-one calibration*: calibrating one projector, turning it off, and then calibrating the next.

**Key Challenge**: Calibration time and workload scale *linearly* with the number of projectors $M$. Since modern PM systems often involve dozens of projectors (for brightness overlapping, super-resolution, or light field displays where even hundreds of projectors project onto the same region), this linear bottleneck makes deploying large-scale projector arrays extremely time-consuming. The authors note that this scalability issue has remained an unsolved challenge in the field for over forty years.

**Key Insight**: The root cause of the problem is that diffuse reflection destroys the directional information of light rays. Alternatively, instead of diffusely reflecting light, can the board directly *receive* and retain the direction of incident light? The authors draw inspiration from two concepts: using pinhole arrays to separate spatial patterns for calibrating a single camera/projector, and embedding light sensors into the scene to directly receive structured light from a single projector for registration.

**Core Idea**: By embedding cameras directly into the calibration board with their optical centers aligned with the board surface, the "observed calibration target" is transformed into an "observer facing the projectors" — a physical and conceptual paradigm shift. The embedded cameras directly capture incident light. Rays from projectors at different positions land on different pixels of the camera image plane, thereby preserving directional information. Consequently, even when patterns from multiple projectors overlap on the board, they can still be isolated and decoded individually.

## Method

### Overall Architecture
The core challenge of the method is "how to decode the structured light of each projector individually when all projectors project simultaneously." **Overall pipeline**: $N$ wide-angle cameras are embedded in the calibration board (with optical centers as close to the board surface as possible) $\rightarrow$ $M$ projectors simultaneously project Gray-code + line-shift + projector ID temporal patterns $\rightarrow$ each embedded camera takes a single capture, separates the overlapping patterns by incident direction, and decodes the correspondence between "projector pixel $p_m(n)$ and camera optical center board coordinate $x_n$" $\rightarrow$ homography compensation is applied to correct the misalignment caused by the camera optical center not being perfectly coplanar with the board $\rightarrow$ the correspondences across multiple board poses are fed into Zhang's calibration method to simultaneously estimate the intrinsic and extrinsic parameters of all projectors.

Key geometric relationship (Fig. 2): A straight line connects the optical centers of projector $m$ and camera $n$, intersecting the projector's image plane at pixel $p_m(n)$ and the camera's image plane at pixel $c_n(m)$. Although light rays from projectors at different locations overlap at the camera's optical center, they land on different pixels of the camera's image plane and can thus be optically resolved. This aligns with the principle of directional encoding in light field cameras.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["M Projectors + N Embedded Cameras<br/>Camera optical centers aligned with board plane"] --> B["Inverse Calibration Paradigm of Embedded Cameras<br/>Cameras directly capture incident light to retain direction info"]
    B --> C["Simultaneous Projection + Directional Decoding<br/>Gray-code/line-shift/Projector ID sequences"]
    C -->|Decode pm(n) ↔ xn for each projector| D["Homography Compensation for Optical Center Offsets<br/>Replace xn with xn(m)"]
    D --> E["Feed multiple board poses into Zhang's Calibration"]
    E --> F["Simultaneously estimate all projectors'<br/>intrinsics K + extrinsics [R|t]"]
```

### Key Designs

**1. Inverse Calibration Paradigm of Embedded Cameras: Shifting from "Observed Target" to "Observer Facing the Projectors"**

The bottleneck of traditional calibration is that diffuse reflection on the board discards light ray directions, making overlapping patterns inseparable. The authors resolve this through a physical and conceptual inversion: embedding $N$ cameras into the calibration board so that each camera's optical center lies *exactly on the board surface*, with lenses directly facing the projectors to receive incident light. Because the incident direction determines where light lands on the camera sensor, even if light beams from multiple projectors overlap at the optical center, they are still dispersed to different pixels on the camera's image plane. Thus, directional info is fully preserved, similar to the principle of plenoptic cameras. Consequently, a single capture can identify which light ray comes from which projector, establishing the physical foundation for the parallel calibration.

**2. Simultaneous Projection for Directional Separation and Per-Projector Decoding: Solving "Who Projected What to Where" via Temporal Coding**

Optical separation is only the first step; it is also necessary to determine which projector pixel corresponds to the light received by each camera pixel. The authors design all projectors to simultaneously project three types of temporal patterns: Gray-code for coarse localization, line-shift (shifting horizontal/vertical lines pixel-by-pixel in the neighborhood of Gray-code resolved pixels $p_m(n)$) for sub-pixel refinement, and a binary "all-white/all-black" temporal sequence encoding the **projector ID** $m$. The decoding process is: first, calculate the maximum-minimum intensity difference over the entire temporal sequence for each camera pixel, determining that it "received light" only if it exceeds threshold $t$; second, decode the ID sequence to identify the emitting projector $m$ and obtain $c_n(m)$; finally, decode the Gray-code and line-shift sequences on that pixel to resolve projector coordinates $p_m(n)$. Since defocus blur and aberrations may cause light from a single projector pixel to hit multiple camera pixels, the intensities of these pixels are averaged before decoding. This design merges "separation" and "correspondence establishment" into a single, simultaneous projection phase, reducing the required pattern count from traditional $M\times(\lceil\log_2 W\rceil+\lceil\log_2 H\rceil+L)$ to $\lceil\log_2 M\rceil+\lceil\log_2 W\rceil+\lceil\log_2 H\rceil+L$. The extra $\lceil\log_2 M\rceil$ terms (for ID identification) remain small in practice since $M\ll W,H$.

**3. Homography Compensation for Optical Center Misalignment: Correcting "Optical Center Offsets from the Board Plane" via RANSAC Homography**

The ideal assumption is that the camera's optical center lies strictly on the board surface. However, physically aligning optical centers to zero offset is impossible due to the uncertainty in compound optical system centers. Once the optical center deviates from the board, the intersection of the ray from projector $m$ to camera $n$ with the board **varies by projector**, denoted as $x_n(m)$ (Fig. 3). Inputting a unified $x_n$ into Zhang's method would introduce errors. The authors' correction uses $x_n(m)$ instead of $x_n$ as the board coordinate input, derived from the camera pixel $c_n(m)$ via a mapping $\mathcal{M}_n$:

$$x_n(m) = \mathcal{M}_n(c_n(m)).$$

Geometrically, $\mathcal{M}_n$ is a projective transformation and can be modeled as a **homography**. Its parameters are calibrated once offline: a printed chessboard is first attached to the board to establish a real-scale coordinate system (with holes drilled at camera locations for projection transmission). A single projector is placed at $K$ different 3D positions $X_k$; for each position, the projector pixel $p_k(n)$ observed by the embedded camera and the corresponding camera pixel $c_n(k)$ are recorded. A vertical white line ($x=u_k(n)$) and a horizontal white line ($y=v_k(n)$) are projected individually onto the board, captured by an external camera, binarized, and solved via Hough transform to reconstruct line equations, intersecting at the ground-truth board coordinates $x_n(k)$. Repeating this $K$ times yields $\{c_n(k), x_n(k)\}$ pairs, which are solved via least squares + RANSAC to estimate the homography $\mathcal{M}_n$. This offline step is performed only once after assembling the board. The authors also note that because the entire camera sensor maps to a tiny $25\,\text{mm}^2$ region on the board, effective lens distortion within this region is negligible, meaning explicit distortion correction is omitted (left as future work).

### Loss & Training
This method is a geometric calibration pipeline with no neural network training. There are two primary closed-form/optimizing steps: the homography $\mathcal{M}_n$ is estimated using least squares and RANSAC; and the projector intrinsic and extrinsic parameters are calculated via Zhang's calibration method (relying on 2D-2D correspondences across multiple board poses), followed by non-linear optimization to minimize reprojection error.

## Key Experimental Results

Prototype: Four holes were drilled into a $470\times320$ mm acrylic board to embed four wide-angle cameras (Raspberry Pi Camera Module 3 Wide, $4608\times2592$, $102^\circ/67^\circ$), with lens tips aligned with the board surface. The four cameras were placed at the vertices of a $200\times120$ mm rectangle. Because the area around the lenses still overexposed under black projection when facing the projectors, ND filters (ND-4.0) were added to each camera. Optical center offset compensation used a Canon EOS RP external camera with a blue-magenta chessboard (red channel for corner detection, blue channel for structured light extraction), sampling each embedded camera with a single projector at 30–40 positions.

### Main Results

**Scalability (25 Projectors)**: A 5×5 overhead array for shadow-free projection. Traditional sequential calibration requires 44 patterns per projector, totaling 1100 patterns taking ~12 minutes. The proposed method requires only 54 patterns for simultaneous projection, achieving a **95% reduction in pattern count**. The face projection after compensation is sharper and clearer than both "without compensation" and "traditional" baselines — even outperforming the traditional approach because external cameras mounted far away on the ceiling suffer from lower apparent resolution, a loss avoided by placing embedded cameras directly on the board.

**Intrinsic and Extrinsic Calibration of Three Projectors (RMS Reprojection Error, pixels)**:

| Projector | Traditional (108 corners) | Traditional (4 corners) | Ours (W/o compensation) | Ours (W/ compensation) |
|--------|------|------|------|------|
| ML1050ST+ | 0.34 | 0.65 | 2.18 | **0.91** |
| TK685 | 0.38 | 0.76 | 2.47 | **0.91** |
| TK850 | 0.39 | 0.75 | 2.44 | **0.89** |

With compensation, errors drop to $<1.0$ pixel (commonly considered accurate enough), comparable to the traditional 4-corner setup. Without compensation, error spikes to $\approx 2.4$ pixels, verifying the necessity of the compensation step. The higher accuracy of the traditional 108-corner setup indicates that scaling up the number of embedded cameras would further improve the precision of the proposed method.

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Ours + Optical compensation | RMS $\approx$ 0.89–0.91 px; MTF close to upper limit; exact alignment of double chessboards | Full method |
| Ours − Optical compensation | RMS $\approx$ 2.18–2.47 px; high-frequency MTF significantly attenuated; double chessboard misalignment | Compensation removed, optical center offset uncorrected |
| Traditional (sequential) | RMS $\approx$ 0.34–0.39 px (108 corners); 25 projectors require 1100 patterns / 12 mins | Upper bound for accuracy but non-scalable |

### Key Findings
- **Optical center compensation is critical for precision**: Removing it degrades the reprojection error from $\approx 0.9$ px to $\approx 2.4$ px. In the red-green double chessboard alignment experiment, the two patterns are visibly misaligned, and the high-frequency MTF is severely attenuated; applying compensation brings the MTF close to the single-projector limit.
- **Exceptional separability**: When two projectors are placed 4.2 m away with lenses almost touching (65 mm, angular separation of only $0.88^\circ$), the embedded camera still clearly distinguishes two light spots — permitting calibration even with sub-1° angular separations.
- **Limited observable incident angle range**: The intensity drops to half at approximately $\pm 32^\circ$ around the x-axis and $\pm 40^\circ$ around the y-axis. Beyond these ranges, insufficient light is received, representing a placement constraint.
- **Robustness to strong ambient light**: Under outdoor sunlight of $\approx 70$ klux, the proposed method directly captures light with unaffected contrast, allowing robust separation and decoding, whereas the traditional external-camera method **fails completely** due to structured light being washed out by ambient illumination.

## Highlights & Insights
- **Clever "Observer Inversion" Paradigm**: The 40-year-old challenge of "losing direction via diffuse reflection" is resolved by "placing cameras on the board surface facing the projectors to capture light directly." Rather than designing complex algorithms to separate superimposed patterns, it physically prevents the loss of directional information, drawing on the principle of directional encoding from light-field cameras.
- **Complexity reduced from $O(M)$ to $O(\log M)$**: The portion of pattern count dependency scaling with projector count is reduced to $\lceil\log_2 M\rceil$ (projector ID encoding), which is practically a small constant. This is the mathematical essence of "scalability."
- **Ambient light robustness as an unexpected bonus**: Direct light capture, instead of photographing a diffuse surface, maintains high contrast under intense light. This addresses real-world challenges like "pre-calibrating in daytime, executing projection mapping at night."
- **Reusable directional information**: The embedded cameras simultaneously capture both position and directional information (similar to plenoptics). The authors envision using direction data as extra constraints to reduce the required number of board poses — a promising avenue to explore.

## Limitations & Future Work
- **Shared projection region requirement**: All projectors must share the same projection area to achieve near-constant scalability. For wide-area systems (e.g., building-scale PM) with non-overlapping projectors, the board must be grouped and repositioned step-by-step, reintroducing linear scaling.
- **Constraint to planar targets**: Only multi-planar targets (polyhedra with face-by-face homographies) are supported; non-planar curved targets are left for future work.
- **Narrow incident angle window**: The observable angular range of embedded cameras is around $\pm32^\circ/\pm40^\circ$, constraining projector placement.
- **Ambient light robustness tested with only two units**: The 70 klux experiment was conducted with only two projectors; large-scale setups remain untested.
- **Lack of explicit distortion correction**: This is bypassed by assuming "distortion is negligible within the small mapped sub-region," which may introduce errors when extreme wide-angle lenses are used.
- Future work also includes combining single-shot structured light (e.g., De Bruijn sequences) to further minimize per-projector patterns, and extracting hardware physical attributes such as lens aberrations, aperture shape, and focus distance.

## Related Work & Insights
- **vs Traditional external camera calibration (including 3D-2D correspondence, planar chessboard, self-calibration, differentiable framework/NeRF/3DGS approaches)**: These rely on an external camera photographing a diffuse board surface. Overlapping patterns cannot be separated, dictating sequential execution with $O(M)$ linear time scaling. This paper flips the paradigm by embedding cameras into the board to directly capture light and preserve directional info, achieving simultaneous calibration with near-constant time.
- **vs Pinhole arrays for spatial pattern separation [28,26]**: Those works use pinhole arrays to calibrate the intrinsics of a **single** camera or projector; this paper extends the "directional separation" concept to **simultaneously calibrate intrinsics and extrinsics of multiple projectors**.
- **vs Scene-embedded light sensors for single-projector registration [14,16,24]**: Prior works use embedded light sensors to directly receive structured light from a single projector for registration; this work introduces a novel paradigm to achieve simultaneous intrinsic and extrinsic estimation for multiple projectors.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Resolves the 40-year scalability challenge of multi-projector calibration via "observer inversion + directional encoding," offering paradigm-shifting innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Features a real prototype, validations at multiple scales (2, 3, and 25 units), sub-1° separation, and outdoor sunlight tests. However, some evaluations are qualitative, and ambient light robustness was tested with only two projectors.
- Writing Quality: ⭐⭐⭐⭐⭐ Clearly derived motivations, well-illustrated geometric relations, self-consistent and accessible formulations and pipeline.
- Value: ⭐⭐⭐⭐⭐ Directly breaks the core bottleneck of deploying large-scale projector arrays, offering substantial real-world value for light filed, superimposed, and shadow-free PM systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] From Corners to Fiducial Tags: Revisiting Checkerboard Calibration for Event Cameras](from_corners_to_fiducial_tags_revisiting_checkerboard_calibration_for_event_came.md)
- [\[CVPR 2026\] TROPHIES: Temporal Reconstruction of Places, Humans, and Cameras from Multi-view Videos](trophies_temporal_reconstruction_of_places_humans_and_cameras_from_multi-view_vi.md)
- [\[CVPR 2026\] Mind the Hitch: Dynamic Calibration and Articulated Perception for Autonomous Trucks](mind_the_hitch_dynamic_calibration_and_articulated_perception_for_autonomous_tru.md)
- [\[CVPR 2026\] SparseCam4D: Spatio-Temporally Consistent 4D Reconstruction from Sparse Cameras](sparsecam4d_spatio-temporally_consistent_4d_reconstruction_from_sparse_cameras.md)
- [\[CVPR 2026\] Query2Uncertainty: Robust Uncertainty Quantification and Calibration for 3D Object Detection under Distribution Shift](query2uncertainty_robust_uncertainty_quantification_and_calibration_for_3d_objec.md)

</div>

<!-- RELATED:END -->

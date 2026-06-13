---
title: >-
  [Paper Note] WaveVerse: Scalable RF Simulation in Generative 4D Worlds
description: >-
  [ICML 2026][Human Understanding][RF Sensing] WaveVerse integrates LLM-driven "4D indoor scene + human motion" generation with a physical ray tracer that preserves spatio-temporal phase coherence. This creates a pipeline…
tags:
  - "ICML 2026"
  - "Human Understanding"
  - "RF Sensing"
  - "mmWave"
  - "Phase-Coherent Ray Tracing"
  - "4D World Generation"
  - "Human Motion Generation"
date: 2026-05-08
content_hash: 022e297d8b6cb08f
---

# WaveVerse: Scalable RF Simulation in Generative 4D Worlds

**Conference**: ICML 2026  
**arXiv**: [2508.12176](https://arxiv.org/abs/2508.12176)  
**Code**: Open-sourced (Available via paper webpage)  
**Area**: Signals & Communications / RF Sensing / Synthetic Data Generation  
**Keywords**: RF Sensing, mmWave, Phase-Coherent Ray Tracing, 4D World Generation, Human Motion Generation

## TL;DR
WaveVerse integrates LLM-driven "4D indoor scene + human motion" generation with a physical ray tracer that preserves spatio-temporal phase coherence. This creates a pipeline from prompt to RF signals, significantly enhancing downstream RF imaging and activity recognition tasks. Unlike existing methods, performance continues to scale with simulation volume without saturation.

## Background & Motivation
**Background**: RF (Radio Frequency/mmWave) sensing offers a privacy-friendly, occlusion-resistant, and low-visibility-robust alternative to computer vision. Applications include 3D imaging, human activity recognition (HAR), and respiratory/sleep monitoring. However, RF data collection is hardware-expensive and requires covering vast variations in room layouts, demographics, and activities. Furthermore, RF data is rarely reusable across systems due to differences in bandwidth, antenna arrays, and modulation, leading to a lack of unified benchmarks like ImageNet.

**Limitations of Prior Work**: Existing relaxation solutions follow two paths: pure physical simulation (e.g., Vid2Doppler, midas), which models signal-human interaction but ignores environmental multipath (the primary factor limiting RF generalization); and learned synthesis (e.g., RF Genesis, RF-Diffusion), which produces realistic signals but relies on massive real-world data and remains tied to specific radar configurations. Professional full-wave solvers like HFSS are accurate but prohibitively slow (>1 hour per simulation) for dynamic indoor scenes.

**Key Challenge**: To achieve scalability, the pipeline must automatically mass-produce "diverse environments × diverse motions × diverse radar hardware" at low cost. For the data to be learnable, it must preserve the phase information essential for RF target differentiation. Existing simulators typically compromise on either environmental complexity or phase accuracy.

**Goal**: Split into two sub-problems: (1) How to populate LLM-generated rooms with spatially reasonable humans performing diverse actions without manually defining precise time-indexed trajectories? (2) How to perform ray tracing on indoor geometry such that phases are continuous and comparable between adjacent radar positions and timestamps?

**Key Insight**: The authors relax the condition for motion generation from "time-indexed trajectory" to "spatial path"—specifying only the line to follow, not the timing. For signal simulation, instead of standard stochastic ray sampling used in graphics, they employ a fixed set of rays anchored to a reference radar, which are then geometrically transformed to other radar positions to ensure stable surface intersections.

**Core Idea**: Combining path-conditioned autoregressive transformers for environment-aware motion generation with phase-coherent ray tracing that treats "graphics sampling" as "communication propagation paths." The result is the WaveVerse hybrid generative-simulation pipeline.

## Method

### Overall Architecture
The input consists of a text description (e.g., "a small bedroom with a queen bed and a desk") and radar hardware parameters. The pipeline first uses an LLM to assemble a structured scene: a semantic mesh environment is generated using (Yang et al., 2024), and an SMPL human model is added with shape parameters inferred by a fine-tuned BodyShapeGPT. Next, the LLM provides motion descriptions ("wave hand," "walk to sofa") and start/end points. A path planning algorithm computes $L=64$ 2D waypoints, which are fed into a state-aware causal transformer to generate a sequence of VQ-VAE encoded motion tokens. Finally, the LLM assigns one of 24 materials with verified dielectric constants and conductivities to each object. Given the 4D world (3D scene + moving human), the phase-coherent ray tracer outputs the channel impulse response $h(t)=\sum_k a_k G_{\text{Tx}}(\theta_k) G_{\text{Rx}}(\varphi_k)\delta(t-\tau_k)$ for given Tx/Rx parameters.

### Key Designs

1. **Path-conditioned state-aware causal transformer**:
    - **Function**: Autoregressively generates variable-length, path-aligned, and physically plausible human motion token sequences conditioned on text and spatial paths.
    - **Mechanism**: Motion is quantized into tokens $X=[m_1,\dots,m_n,m_{\text{end}}]$ via VQ-VAE. CLIP encodes text, and an MLP encodes 2D waypoints as condition $c$. Key modifications: (1) Prediction probability is changed to $P(m_n\mid c, m_0, s_0, \dots, m_{n-1}, s_{n-1})$, where $s_i$ is the 2D pelvis position at the end of token $i$, anchoring predictions to spatial states. (2) During training, consecutive waypoints are randomly masked at a ratio $r\in[0.5,0.9]$ to prevent the model from ignoring text in favor of the path.
    - **Design Motivation**: Prior methods using time-indexed trajectories are labor-intensive and over-constrained. Using paths only constrains "where to go," leaving speed and style to the generative model, while state-conditioning prevents path deviation and masking prevents overfitting.

2. **Spatial Phase Coherence**:
    - **Function**: Simulates signals for $N$ radars at different poses $(\mathbf{t}_n,\mathbf{r}_n)$ simultaneously, ensuring phase differences strictly correspond to geometric path differences for beamforming.
    - **Mechanism**: Using the geometric center $(\mathbf{t}_0,\mathbf{r}_0)$ of all radars as a reference, rays are emitted uniformly to find reference paths $\mathcal{P}_k=[\mathbf{t}_0,\mathbf{p}_1,\dots,\mathbf{p}_{D_k},\mathbf{r}_0]$. For other radars, only the endpoints are replaced with $(\mathbf{t}_n,\mathbf{r}_n)$, keeping intersection points $\mathbf{p}_d$ identical. After occlusion checking, delays $\tau_k$, attenuation, and phases are recalculated.
    - **Design Motivation**: Independent random sampling per radar causes slightly different surface hit points, introducing random noise into phase differences and creating artifacts in beamforming. This "anchor + transform" approach ensures phase differences originate solely from endpoint geometry while reducing redundant computation.

3. **Temporal Phase Coherence + Vertex Group Expansion**:
    - **Function**: Maintains continuous transitions of "hit points" on the human body during mesh deformation across frames, preserving $\mu$m–mm level phase changes for Doppler and respiration sensing.
    - **Mechanism**: $M$ SMPL vertices are grouped into $G$ body parts via $\mathcal{G}:\mathcal{V}\to\{1,\dots,G\}$. When a ray hits $\mathbf{p}_d^{(t)}$, it is "expanded" into a bundle of paths targeting all vertices $\mathbf{v}_m$ in the same group. Attenuation is normalized by the number of valid paths $N_{\text{valid}}$. To avoid exponential explosion, expansion is only applied to the first Tx bounce.
    - **Design Motivation**: Random sampling on dynamic bodies causes phase discontinuities. Locking body parts as "continuous surface proxies" allows stable phase evolution for sub-millimeter signal inversion.

### Loss & Training
VQ-VAE for motion tokens uses standard reconstruction + codebook loss. The causal transformer uses next-token cross-entropy with path-masking data augmentation. The ray tracer is purely physical (non-learnable). Dielectric parameters were proposed by an LLM and filtered against literature values to form a 24-material library.

## Key Experimental Results

### Main Results: Motion Generation (HumanML3D, 14,616 captioned motions)

| Method | Architecture | R-Prec ↑ | FID ↓ | Path Err ↓ | Ending Err ↓ |
|--------|--------------|----------|-------|------------|--------------|
| Ground Truth | – | 0.797 | 0.002 | 0 | 0 |
| MDM | Diffusion | 0.719 | 0.295 | 0.547 | 0.666 |
| OmniControl| Diffusion | 0.751 | 0.319 | 0.239 | 0.330 |
| MotionLCM | Diffusion | 0.739 | 0.754 | 0.315 | 0.468 |
| T2M-GPT | AR | 0.691 | 0.377 | 0.406 | 0.545 |
| **WaveVerse**| AR | **0.755** | **0.238** | **0.208** | **0.325** |

WaveVerse ranks first or tied for first in text alignment (R-Prec), motion quality (FID), and path/ending error, significantly outperforming its backbone T2M-GPT.

### Ablation Study: state-aware causal transformer

| Configuration | R-Prec ↑ | FID ↓ | Path Err ↓ | Ending Err ↓ |
|---------------|---------|-------|------------|--------------|
| Full | 0.755 | 0.238 | 0.151 | 0.287 |
| w/o Mask | 0.643 | 0.747 | 0.192 | 0.325 |
| w/o State | 0.757 | 0.422 | 0.250 | 0.460 |
| w/o Both | 0.691 | 0.377 | 0.274 | 0.528 |

Removing masking degrades text alignment (R-Prec drops 14.8%), while removing state-anchoring degrades path tracking (Path Err increases 65%).

### Signal Fidelity
- **Spatial Phase**: Panoramic imaging with 1,200 array positions shows sharp images and multipath ghosts (indicating multipath was captured), whereas the baseline is purely noise.
- **Temporal Phase**: Driving SMPL with real breathing signals, distal chest curve reconstruction RMSE improved from 0.14 to 0.08.
- **vs Real Signal**: Range-time spectroscopy shows 28.63 dB PSNR / 93.65% energy similarity compared to real mmWave captures.
- **vs HFSS**: Average 33.57 dB PSNR across 16 indoor setups. WaveVerse takes <1s per case versus 1+ hours for HFSS.

### Key Findings
- **Scalability**: Performance increases with synthetic data volume, whereas Standard RT and RF Genesis saturate or degrade. Physical fidelity and phase coherence are the bottlenecks for scalability.
- **Gain**: 4× synthetic data achieved 73.33% of the error reduction of 4× real data in HAR tasks.
- **Reliability**: Scene generation success rate is 95.83%, with a low collision rate (2.35% of frames).

## Highlights & Insights
- **Geometry over Sampling**: Replacing graphics-style random sampling with fixed propagation paths anchored to a reference solves both phase noise and redundant computation efficiently.
- **Vertex Group Expansion**: Approximating area integrals via semantic vertex groups balances computational cost and temporal continuity.
- **Abstraction Shift**: Moving from time-indexed trajectories to spatial paths allows the generative model to determine rhythm and style, improving long-range spatial consistency.
- **LLM + Physical Filter**: Using LLMs to propose material libraries while filtering with physical bounds combines broad knowledge with physical reality.

## Limitations & Future Work
- **Higher-order Reflections**: Only expansion on the first bounce may lose fidelity in highly metallic or specular environments.
- **Material Granularity**: 24 materials may oversimplify complex surfaces like carpets or layered clothing at frequencies >77 GHz.
- **Single-person Focus**: Evaluation was limited to single-person interactions; complex multi-person scenarios remain unexplored.
- **Pipeline Dependencies**: Bias in the LLM's scene/layout generation directly propagates to the RF data bias.

## Related Work & Insights
- **vs RF Genesis**: RF Genesis requires real training data and fixed hardware; WaveVerse is purely physical and hardware-agnostic, showing superior scaling in HAR.
- **vs Standard Ray Tracing**: Standard RT hurts imaging performance (MAE 20.10 -> 22.28) due to phase incoherence, which WaveVerse resolves.
- **vs HFSS**: Near HFSS-level accuracy at 3-orders-of-magnitude higher speed.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (Integrates path-conditioned 4D generation with phase-coherent simulation).
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Comprehensive benchmarks across motion, phase, fidelity, and downstream scaling).
- Writing Quality: ⭐⭐⭐⭐ (Logical; Fig. 4 is essential for understanding phase coherence).
- Value: ⭐⭐⭐⭐⭐ (Provides "NeRF + ImageNet" style infrastructure for the data-starved RF sensing community).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] LaScA: Language-Conditioned Scalable Modelling of Affective Dynamics](../../CVPR2026/human_understanding/lasca_language-conditioned_scalable_modelling_of_affective_dynamics.md)
- [\[ICCV 2025\] EgoAgent: A Joint Predictive Agent Model in Egocentric Worlds](../../ICCV2025/human_understanding/egoagent_a_joint_predictive_agent_model_in_egocentric_worlds.md)
- [\[CVPR 2026\] HUM4D: A Dataset and Evaluation for Complex 4D Markerless Human Motion Capture](../../CVPR2026/human_understanding/hum4d_markerless_motion_capture.md)
- [\[ICCV 2025\] GenM3: Generative Pretrained Multi-path Motion Model for Text Conditional Human Motion Generation](../../ICCV2025/human_understanding/genm3_generative_pretrained_multi-path_motion_model_for_text_conditional_human_m.md)
- [\[ICCV 2025\] HUMOTO: A 4D Dataset of Mocap Human Object Interactions](../../ICCV2025/human_understanding/humoto_a_4d_dataset_of_mocap_human_object_interactions.md)

</div>

<!-- RELATED:END -->

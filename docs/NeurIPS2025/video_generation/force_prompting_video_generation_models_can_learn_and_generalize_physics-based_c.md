---
title: >-
  [Paper Note] Force Prompting: Video Generation Models Can Learn and Generalize Physics-based Control Signals
description: >-
  [NeurIPS 2025][Video Generation][force prompting] This paper proposes Force Prompting, which uses physical forces (local point forces and global wind forces) as control signals for video generation models. Using only ~15K synthetic training videos (Blender flags and rolling balls) and a single day of training on 4×A100 GPUs, the method achieves remarkable generalization across diverse real-world scenes with varying objects, materials, and geometries, including preliminary mass understanding capabilities.
tags:
  - NeurIPS 2025
  - Video Generation
  - force prompting
  - physics control
  - sim2real
  - CogVideoX
date: 2026-05-08
content_hash: 31bd5893809edae2
---

# Force Prompting: Video Generation Models Can Learn and Generalize Physics-based Control Signals

**Conference**: NeurIPS 2025
**arXiv**: [2505.19386](https://arxiv.org/abs/2505.19386)
**Code**: [force-prompting.github.io](https://force-prompting.github.io/) (dataset + code + model weights fully open-sourced)
**Area**: Video Generation / Physics-Controllable Generation / World Models
**Keywords**: force prompting, video generation, physics control, sim2real, CogVideoX

## TL;DR
This paper proposes Force Prompting, which uses physical forces (local point forces and global wind forces) as control signals for video generation models. Using only ~15K synthetic training videos (Blender flags and rolling balls) and a single day of training on 4×A100 GPUs, the method achieves remarkable generalization across diverse real-world scenes with varying objects, materials, and geometries, including preliminary mass understanding capabilities.

## Background & Motivation

**Background**: Video generation models (Sora/CogVideoX/Wan2.1) have made substantial progress in visual quality and motion realism, but primarily rely on text and image inputs, lacking precise control over physical interactions. Existing controllable generation research focuses mainly on camera control and trajectory control.

**Limitations of Prior Work**: (a) Trajectory control requires pre-specified per-frame pixel positions and cannot handle global phenomena (wind/fluid); (b) trajectories and forces are fundamentally different physical quantities—the same force produces different displacements for objects of different masses; (c) physics simulator-based methods require 3D geometry or running a simulator at inference time.

**Key Challenge**: Acquiring high-quality force–video paired training data is extremely difficult.

**Goal**: To teach a pre-trained video generation model to understand force control signals using a minimal amount of synthetic physics simulation data.

**Key Insight**: It is hypothesized that state-of-the-art video models already encode strong priors about visual dynamics, and that synthetic data need only "activate" rather than "teach" the model.

**Core Idea**: Video generation models can learn force-conditioned generation from a very small amount of synthetic data and exhibit remarkable generalization across objects, materials, and geometries.

## Method

### Overall Architecture
The model takes as input a triplet $(\tau, \phi, \pi)$: text, initial frame, and physical control signal, and outputs 49 frames at 8 fps. A 6-layer ControlNet is added to CogVideoX-5B-I2V to inject force signals, with the backbone frozen.

### Key Designs

1. **Synthetic Training Data**:

    - Global wind forces (15K): Blender flag simulations with randomized flag count (1–64), color (100 options), HDRI backgrounds (50 options), wind direction, and wind speed.
    - Local point forces (23K): 12K Blender rolling balls (soccer vs. bowling ball with mass ratio 1:4) + 11K PhysDreamer carnations being poked.
    - The presence of distractor balls is critical for spatial force localization.

2. **Force Encoding Strategy**:

    - Global wind force: 3-channel full-image uniform values (force magnitude mapped to $[-1, 1]$, $\cos/\sin$ of angle).
    - Local point force: A Gaussian blob moves from a specified position along the force direction, with displacement proportional to force magnitude. The blob position is displaced away from the target pixel—a fundamental distinction from trajectory control.

3. **Architecture and Training**:

    - CogVideoX-5B-I2V (frozen) + 6-layer ControlNet (cloned from the first 6 transformer layers).
    - 4×A100, batch size 8, 5000 steps (~1 day), AdamW with lr = $1 \times 10^{-5}$.

## Key Experimental Results

### Local Force Model vs. Baselines (2AFC Human Preference %, >50% indicates preference for the proposed method)

| Baseline | Force Following | Physical Realism | Visual Quality |
|----------|----------------|-----------------|----------------|
| Text-only zero-shot | 72/67/73 | 50/48/48 | 48/52/49 |
| Text-only fine-tuned | 79/62/74 | 53/52/55 | 52/58/54 |
| Motion Prompting | **91/89/86** | **93/76/76** | 100/99/98 |

### vs. PhysDreamer (mean over 6 plant categories)

| Metric | Force Prompting Preference Rate |
|--------|---------------------------------|
| Motion Realism | 48.3% (roughly tied) |
| Visual Quality | 36.7% (PhysDreamer preferred) |
| **Force Following** | **58.3% (proposed method preferred)** |

### Key Findings
- A wind model trained only on flags generalizes to smoke, snowflakes, balloons, and other entirely different materials.
- A point force model trained only on rolling balls and carnations generalizes to various plants, hot air balloons, swings, and more.
- **Mass Understanding**: A soccer ball consistently travels farther than a bowling ball under the same force, and displacement scales linearly with force magnitude.
- **Zero-shot Multi-force**: Adding multiple Gaussian blobs at inference time enables control of multiple objects without retraining.
- Training data design matters more than quantity: removing distractor balls prevents spatial force localization; using a single background causes foreground/background confusion.
- Using text keywords such as "wind/breeze/blow" during training is critical for generalization.

## Highlights & Insights
- **Generalization with minimal data and compute**: With 15K synthetic videos and a single day of training, the model generalizes from flags to smoke, snowflakes, and balloons—suggesting that pre-trained video models already encode rich intuitive physics knowledge, with synthetic data serving merely as a "key."
- **Force ≠ Trajectory**: The Gaussian blob position is displaced away from the target pixel (e.g., an oscillating flower), representing a fundamental difference from trajectory control with important implications for world model design.
- **Engineering Insights from Ablations**: Distractor objects are essential for force localization; visual diversity is essential for material generalization; text keywords help activate the model's intrinsic physical understanding.

## Limitations & Future Work
- Visual quality is slightly inferior to PhysDreamer, which leverages 3D geometry and a physics simulator.
- Force magnitudes are not calibrated across training scenes; only relative magnitudes are meaningful without absolute physical units.
- Mass understanding is limited to the soccer-ball-vs.-bowling-ball mass ratio seen during training.
- Global and local force models are trained separately; local force control weakens slightly when the two are merged.
- Additional force types (gravity, friction, elastic forces) remain unexplored.
- The ControlNet clones only the first 6 layers due to GPU memory constraints; more layers may improve visual quality.
- The approach currently applies only to the CogVideoX architecture; transferring to other video models requires retraining the ControlNet.

### Loss & Training
- The native diffusion training loss of CogVideoX-5B-I2V is used; the ControlNet employs zero convolution initialization.
- AdamW optimizer, learning rate $1 \times 10^{-5}$, cosine schedule with restarts, 250-step warmup.
- bf16 mixed precision + tf32 acceleration, batch size 8 (2-step gradient accumulation), seed = 42.
- Checkpoints saved every 500 steps; training completes in 5000 steps.

## Related Work & Insights
- **vs. Motion Prompting**: Conditions generation on spatiotemporally sparse trajectories; Force Prompting substantially outperforms it in force following and physical realism. Three-frame trajectories cannot express the physical semantics of force—trajectories are not forces.
- **vs. PhysDreamer**: Requires a per-scene 3D Gaussian representation and a physics simulator; achieves better visual quality but poor generalization—limited to the specific scenes seen during training. Force Prompting generalizes without 3D information.
- **vs. PhysGen/PhysMotion**: Relies on running a physics simulator at inference time, which constrains the types of dynamics that can be modeled. Force Prompting lets the video model itself act as the "simulator."
- **vs. PhysCtrl**: Requires learning a 3D point-cloud trajectory model before passing signals to the video generator, resulting in a more complex pipeline. Force Prompting is more concise end-to-end.
- **vs. Text-only Baselines**: Even a fine-tuned text-conditioned model cannot reliably convey force direction and magnitude, demonstrating that force control requires a dedicated conditioning mechanism.
- The method offers a new pathway for embodied AI: agents can use force prompts to predict physical interaction outcomes for task planning.
- The success of sim-to-real generalization suggests that pre-trained video models harbor untapped "physical potential" that synthetic data may activate across a broad range of physical properties.
- The local + global force framework is extensible to other physical quantities (temperature change, electric fields, etc.) as control signals.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Using physical forces as control signals for video generation is an entirely new paradigm; the mass understanding finding is particularly exciting.
- Experimental Thoroughness: ⭐⭐⭐⭐ Human evaluation is well-designed and the data ablations are thorough, though quantitative evaluation relies primarily on subjective preference.
- Writing Quality: ⭐⭐⭐⭐⭐ Writing is clear, visualizations are excellent, and the project page is outstanding.
- Value: ⭐⭐⭐⭐⭐ A directional contribution to world models and controllable video generation, with full open-source resources.
<!-- NeurIPS 2025 | video_understanding -->

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] PhysCtrl: Generative Physics for Controllable and Physics-Grounded Video Generation](physctrl_generative_physics_for_controllable_and_physicsgrou.md)
- [\[NeurIPS 2025\] VMDT: Decoding the Trustworthiness of Video Foundation Models](vmdt_decoding_the_trustworthiness_of_video_foundation_models.md)
- [\[NeurIPS 2025\] Video Diffusion Models Excel at Tracking Similar-Looking Objects Without Supervision](video_diffusion_models_excel_at_tracking_similar-looking_objects_without_supervi.md)
- [\[ICCV 2025\] VPO: Aligning Text-to-Video Generation Models with Prompt Optimization](../../ICCV2025/video_generation/vpo_aligning_text-to-video_generation_models_with_prompt_optimization.md)
- [\[NeurIPS 2025\] Video Killed the Energy Budget: Characterizing the Latency and Power Regimes of Open Text-to-Video Models](video_killed_the_energy_budget_characterizing_the_latency_and_power_regimes_of_o.md)

<!-- RELATED:END -->

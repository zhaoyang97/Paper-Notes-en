---
title: >-
  [Paper Note] LAMP: Language-Assisted Motion Planning for Controllable Video Generation
description: >-
  [CVPR 2026][Video Generation][Paper Note] The LAMP framework models motion control as a language-to-program synthesis problem. By designing a cinematography-inspired motion DSL, the authors train an LLM to transform natural language descriptions into structured motion programs. These programs are deterministically mapped to 3D object and camera trajectories to
tags:
  - CVPR 2026
  - Video Generation
date: 2026-05-08
content_hash: 8e3faa1213bea001
---
# LAMP: Language-Assisted Motion Planning for Controllable Video Generation

**Conference**: CVPR 2026  
**arXiv**: [2512.03619](https://arxiv.org/abs/2512.03619)  
**Code**: [Project Page](https://cyberiada.github.io/LAMP/)  
**Area**: Video Generation  
**Keywords**: Video Generation, Motion Control, LLM Planning, Domain-Specific Language, Cinematography

## TL;DR

The LAMP framework models motion control as a language-to-program synthesis problem. By designing a cinematography-inspired motion DSL, the authors train an LLM to transform natural language descriptions into structured motion programs. These programs are deterministically mapped to 3D object and camera trajectories to condition video generation, enabling the simultaneous generation of object and camera motion from natural language for the first time.

## Background & Motivation

Video generation has made significant progress, but motion control—specifying object dynamics and camera trajectories—remains limited by user interaction methods. Existing approaches rely on text, annotations extracted from videos, or simple 2D sketching interfaces, which struggle to express complex cinematic motion.

Key Challenge: Object motion and camera trajectories are inherently coupled (cameras are often defined relative to moving objects). Specifying both simultaneously requires advanced spatial planning and mental visualization. For example, orchestrating a chase scene requires coordinating the runner's path and the tracking camera.

Limitations of Prior Work:
- Difficulty in direct 3D coordinate regression from text: The language-to-motion mapping is multi-modal and structurally constrained.
- Previous methods focus only on layout generation or camera trajectory synthesis, failing to unify object and camera motion.
- Lack of iterative editing interfaces.

Core Idea of LAMP: Leverage the program synthesis capabilities of LLMs to transform motion control into a **language-conditioned program synthesis** problem—generating symbolic motion programs instead of continuous coordinates, which are then deterministically mapped to 3D trajectories.

## Method

### Overall Architecture

LAMP aims to allow users to specify both object motion and camera shots (e.g., "a person running on the street with the camera tracking from the side") via natural language without manual 2D sketching or frame-by-frame 3D positioning. The key insight is to avoid direct regression of continuous 3D coordinates from text. Instead, an LLM generates a **symbolic motion program**, which is translated into precise trajectories via deterministic rules.

The pipeline functions as follows: given a natural language input, a fine-tuned LLM motion planner outputs a DSL motion program. This program is deterministically expanded into 3D object and camera trajectories. The trajectories are rendered into a "control video" (projecting 3D bounding boxes and global voxels into the frame). Finally, this control video, along with the text prompt and initial frame, is fed into a pre-trained video diffusion model (VACE) to generate the final video. The intermediate DSL program is human-readable and editable, allowing users to refine motion before synthesis.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    DSL["Cinematography-inspired Motion DSL<br/>4 Primitives + Modifiers over 4 time segments"]
    DSL -->|Sampling + Expansion + Paraphrasing| CORP["Procedural Training Corpus Construction<br/>400K Text-Motion Pairs"]
    IN["Natural Language Description"] --> PLAN
    CORP -->|Fine-tuning Qwen2.5-VL| PLAN["LLM Motion Planner & Hierarchical Decomposition<br/>Object first → Camera based on object"]
    PLAN -->|Output DSL Program (Human-readable/editable)| EXP["Deterministic Expansion<br/>3D Object Trajectory + 6-DoF Camera Trajectory"]
    EXP --> CTRL["Render Control Video<br/>3D Bounding Box + Global Voxel Projection"]
    CTRL --> VACE["VACE Video Diffusion<br/>+ Text + Initial Frame"]
    VACE --> OUT["Final Video"]
    PLAN -->|User Local Rewriting| PLAN
```

### Key Designs

**1. Cinematography-inspired Motion DSL: Symbolic Primitives instead of Continuous Trajectories**

Direct coordinate regression is difficult because one description maps to infinite specific trajectories, leading to sparse and unstable supervision signals. LAMP discretizes "motion" into a vocabulary inspired by cinematography. Based on the CameraBench taxonomy, four primitives are defined: free-form (unconstrained 6-DoF), orbit track (circling the target), tail track (following the target), and rotation track (stationary tracking). Each primitive is refined with modifiers for translation (`lat`/`vert`/`depth`), rotation (`yaw`/`pitch`/`roll`), and temporal style (`speed_fast`/`ease_in`/`jitter_low`), formatted as key-value pairs. A sequence consists of up to four motion tags across four segments. This symbolic representation is interpretable, composable, and easier to learn due to restricted vocabulary and clear supervision.

**2. Procedural Training Corpus Construction: Automatic 400K Text-Motion Pairs**

To train the LLM in DSL synthesis, a large volume of "natural language ↔ motion program" pairs is required. LAMP automates this: motion primitives are sampled and combined to create DSL programs, which are deterministically expanded into 3D trajectories. Template-based text descriptions are generated and then paraphrased by an LLM for diversity. The resulting 400K samples (100K free motion + 100K relative motion, each with raw and paraphrased text) cover 27 coarse categories and 343 fine categories, with rotation angles densely sampled in $[-180°, 180°]$.

**3. LLM Motion Planning & Hierarchical Decomposition: Mechanism**

Object and camera motions are coupled—cameras are often defined relative to objects. LAMP decomposes the joint distribution according to the hierarchy of filmmaking:

$$p(s_{cam}, s_{obj} \mid t) = p(s_{obj} \mid t_{obj}) \cdot p(s_{cam} \mid s_{obj}, t_{cam})$$

The model first generates object motion $s_{obj}$ and then generates camera motion $s_{cam}$ conditioned on the object. The planner is a Qwen2.5-VL model fine-tuned on the 400K corpus. This decomposition aligns with the intuition that objects define scene dynamics while cameras adjust composition. It also supports iterative refinement; users can modify specific program parts (e.g., "lower the camera") without re-synthesizing the entire sequence.

### Loss & Training

The LLM planner is trained using a standard autoregressive target without additional structural losses. During inference, the DSL program is mapped to 3D trajectories and rendered into a control video (2D bounding box and global voxel projections), which serves as input to the VACE video generator alongside the text and initial frame.

## Key Experimental Results

### Main Results — DataDoP Camera Trajectory Evaluation

| Model | Adjusted F1 | CLaTr Score | CLaTr FID |
|------|--------|-------------|-----------|
| CCD (Pre-trained) | — | 5.29 | 357.8 |
| ET (Pre-trained) | — | 2.46 | 609.9 |
| GenDoP (DataDoP Trained) | 0.400 | 36.18 | 22.7 |
| **LAMP (Pre-trained)** | **0.763** | **36.29** | 66.9 |
| **LAMP (ft DataDoP)** | **0.776** | **36.52** | 67.2 |

### Ablation Study

| Configuration | Description |
|------|------|
| W/o DSL (Direct Regression) | Performance drops significantly |
| W/o Fine-tuning (Zero-shot) | Basic capabilities present but low precision |
| Full LAMP | Optimal performance |

### Key Findings

- LAMP outperforms GenDoP (trained on DataDoP) in Adjusted F1 even without DataDoP training (0.763 vs 0.400), proving the strong generalization of the DSL representation.
- Symbolic programs are more efficient than direct coordinate regression, requiring less data.
- Iterative refinement (e.g., "zoom out slightly," "camera lower") is a unique advantage, allowing users to adjust motion without expensive video synthesis runs.

## Highlights & Insights

- Redefining motion generation as program synthesis rather than coordinate regression is a significant architectural paradigm shift.
- Aligning DSL design with cinematography conventions ensures a professional cinematic feel in the generated motion.
- The decoupled design allows iterative motion modification before video synthesis, significantly reducing creative costs.
- It provides the first unified natural language control for both object and camera motion.

## Limitations & Future Work

- Currently supports only single-object scenes (via 3D bounding boxes); multi-object interaction is not yet addressed.
- Motion sequences are limited to four temporal segments; longer complex motions require extensions.
- Final video quality is constrained by the underlying video generation model (VACE).
- The CLaTr FID is higher than GenDoP (66.9 vs 22.7), suggesting room for improvement in trajectory realism.

## Related Work & Insights

- **vs GenDoP**: GenDoP uses GPT for detailed directorial descriptions to guide autoregressive path generation; LAMP uses the LLM to directly output executable motion programs.
- **vs ET**: ET uses cinematic descriptions to guide a diffusion model's trajectory prediction; LAMP skips diffusion for trajectories and uses deterministic DSL mapping.
- **vs CameraCtrl/EPiC**: These methods control only the camera and assume static objects; LAMP provides unified control for both.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Strong innovation in redefining motion control as program synthesis and integrating cinematography.
- Experimental Thoroughness: ⭐⭐⭐⭐ Quantitative comparisons across multiple benchmarks, including ablations and user studies.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, well-structured methodology, and intuitive diagrams.
- Value: ⭐⭐⭐⭐⭐ Significant contribution to controllable video generation with an elegant and scalable framework.

<!-- RELATED:START -->
<div class="related-papers" markdown="1">
</div>
<!-- RELATED:END -->

## Related Papers

- [\[CVPR 2026\] MotionEnhancer: Leveraging Video Diffusion for Motion-Enhanced Vision-Language Models](motionenhancer_leveraging_video_diffusion_for_motion-enhanced_vision-language_mo.md)
- [\[CVPR 2026\] DriveLaW: Unifying Planning and Video Generation in a Latent Driving World](drivelaw_unifying_planning_and_video_generation_in_a_latent_driving_world.md)
- [\[CVPR 2026\] TV2TV: A Unified Framework for Interleaved Language and Video Generation](tv2tv_a_unified_framework_for_interleaved_language_and_video_generation.md)
- [\[AAAI 2026\] MotionCharacter: Fine-Grained Motion Controllable Human Video Generation](../../AAAI2026/video_generation/motioncharacter_fine-grained_motion_controllable_human_video_generation.md)
- [\[CVPR 2026\] MultiShotMaster: A Controllable Multi-Shot Video Generation Framework](multishotmaster_a_controllable_multi-shot_video_generation_framework.md)

</div>

<!-- RELATED:END -->

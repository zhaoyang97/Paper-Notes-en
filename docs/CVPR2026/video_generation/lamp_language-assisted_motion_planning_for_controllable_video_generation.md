---
title: >-
  [Paper Note] LAMP: Language-Assisted Motion Planning for Controllable Video Generation
description: >-
  [CVPR 2026][Video Generation] LAMP frames motion control as a language-to-program synthesis problem: it designs a cinematography-inspired motion DSL, fine-tunes an LLM to translate natural language descriptions into structured motion programs, and deterministically maps these programs to 3D object and camera trajectories that condition a video diffusion model — achieving, for the first time, simultaneous natural-language control over both object and camera motion.
tags:
  - CVPR 2026
  - Video Generation
  - motion control
  - LLM planning
  - domain-specific language
  - cinematography
date: 2026-05-08
content_hash: cb732a1afca31ea5
---

# LAMP: Language-Assisted Motion Planning for Controllable Video Generation

**Conference**: CVPR 2026
**arXiv**: [2512.03619](https://arxiv.org/abs/2512.03619)
**Code**: [Project Page](https://cyberiada.github.io/LAMP/)
**Area**: Video Generation
**Keywords**: video generation, motion control, LLM planning, domain-specific language, cinematography

## TL;DR

LAMP frames motion control as a language-to-program synthesis problem: it designs a cinematography-inspired motion DSL, fine-tunes an LLM to translate natural language descriptions into structured motion programs, and deterministically maps these programs to 3D object and camera trajectories that condition a video diffusion model — achieving, for the first time, simultaneous natural-language control over both object and camera motion.

## Background & Motivation

Video generation has advanced substantially, yet motion control — specifying object dynamics and camera trajectories — remains constrained by limited user interaction paradigms. Most existing methods rely on text, video-extracted annotations, or simple 2D drawing interfaces, all of which struggle to express complex cinematic motion.

**Core pain point**: Object motion and camera trajectories are inherently coupled (cameras are typically defined relative to moving objects), and jointly specifying both demands high-level spatial planning and mental visualization. Choreographing a chase scene, for instance, requires simultaneously coordinating the runner's path and the tracking camera.

**Limitations of prior work**:
- Directly regressing 3D coordinates from language is difficult: the language-to-motion mapping is multimodal and structurally constrained.
- Prior methods address only layout generation or camera trajectory synthesis — not unified object and camera motion.
- No existing interface supports iterative editing.

**Core Idea**: LAMP exploits the program-synthesis capabilities of LLMs, recasting motion control as a **language-conditioned program synthesis** problem — the LLM generates symbolic motion programs rather than continuous coordinates, which are then deterministically converted to 3D trajectories.

## Method

### Overall Architecture

1. Natural language description → 2. LLM motion planner generates DSL motion program → 3. Deterministic conversion to 3D object and camera trajectories → 4. Rendering into a control video → 5. Conditioning a pre-trained video diffusion model to produce the final video.

### Key Designs

1. **Cinematography-Inspired Motion DSL**:
    - **Function**: Provides an interpretable, composable motion representation.
    - **Mechanism**: Grounded in the CameraBench taxonomy, four primitive motion types are defined:
        - **Free-form**: Unconstrained 6-DoF motion.
        - **Orbit track**: Camera orbits around a target object.
        - **Tail track**: Camera follows the object's motion.
        - **Rotation track**: Camera rotates in place to track the object.
    - Each primitive is parameterized by modifiers: translational control (`lat`, `vert`, `depth`), rotational control (`yaw`, `pitch`, `roll`), and temporal/style cues (`speed_fast`, `ease_in`, `jitter_low`), expressed as key-value pairs.
    - A motion sequence comprises up to four motion tokens spanning four temporal segments to capture variation.
    - **Design Motivation**: Symbolic representation yields data efficiency, interpretability, and compositionality — complex motions emerge from combinations of simple primitives.

2. **Programmatic Training Corpus Construction**:
    - **Function**: Provides large-scale text–motion paired data for LLM fine-tuning.
    - **Mechanism**: A corpus of 400K text–motion samples is constructed (100K free-form + 100K object-relative × original + LLM-paraphrased). Pipeline: sample and compose motion primitives → DSL program → deterministic conversion to 3D trajectory → template-based text description → LLM paraphrase for linguistic diversity.
    - Covers 27 coarse categories (3 motion types × 3 directions) and 343 fine-grained categories; rotation angles are densely sampled over $[-180°, 180°]$.
    - **Design Motivation**: Automated generation eliminates large-scale manual annotation, and the data distribution is controllable — common cinematic motions appear more frequently while rare complex combinations appear less.

3. **LLM Motion Planning with Hierarchical Decomposition**:
    - **Function**: Generates symbolic motion programs for both object and camera from natural language.
    - **Mechanism**: The joint probability is factorized as $p(s_{cam}, s_{obj} | t) = p(s_{obj} | t_{obj}) \cdot p(s_{cam} | s_{obj}, t_{cam})$, generating object motion first and then conditioning camera motion on it. A VLM (Qwen2.5-VL) is fine-tuned on the 400K corpus to learn DSL program generation.
    - **Design Motivation**: The decomposition mirrors the hierarchical structure of filmmaking — object motion defines scene dynamics, and the camera adapts to maintain composition. The approach also supports iterative refinement (e.g., *"lower the camera a bit"*).

### Loss & Training

The LLM planner is trained with standard autoregressive supervision. At inference time, DSL programs are deterministically mapped to 3D trajectories and rendered as control videos (2D bounding box projections + global cube projections), which are fed together with text and the first frame into the VACE video generator.

## Key Experimental Results

### Main Results — DataDoP Camera Trajectory Evaluation

| Model | Corrected F1 | CLaTr Score | CLaTr FID |
|-------|-------------|-------------|-----------|
| CCD (pretrained) | — | 5.29 | 357.8 |
| ET (pretrained) | — | 2.46 | 609.9 |
| GenDoP (trained on DataDoP) | 0.400 | 36.18 | 22.7 |
| **LAMP (pretrained)** | **0.763** | **36.29** | 66.9 |
| **LAMP (ft DataDoP)** | **0.776** | **36.52** | 67.2 |

### ET Dataset Evaluation

LAMP consistently surpasses all baselines in F1 score on both simple (pure) and complex (mixed) splits.

### Ablation Study

| Configuration | Description |
|---------------|-------------|
| w/o DSL (direct regression) | Significant performance drop |
| w/o fine-tuning (zero-shot) | Basic capability present but low precision |
| Full LAMP | Best performance |

### Key Findings

- Without any training on DataDoP, LAMP's corrected F1 already exceeds GenDoP trained on that dataset (0.763 vs. 0.400), demonstrating strong generalization from the DSL representation.
- Symbolic programs are more data-efficient than direct coordinate regression.
- Iterative refinement (e.g., *"zoom out slightly"*, *"lower the camera"*) is a unique advantage — users can adjust motion without expensive video synthesis.

## Highlights & Insights

- Reframing motion generation as program synthesis rather than coordinate regression represents a fundamental architectural shift.
- The DSL is aligned with cinematographic conventions, imbuing generated motions with a professional, cinematic quality.
- The decoupled design enables iterative motion refinement prior to video synthesis, substantially reducing authoring cost.
- LAMP is the first method to unify natural-language control over both object and camera motion.

## Limitations & Future Work

- Currently limited to single-object scenes (3D bounding box); multi-object interaction is not addressed.
- Motion sequences are restricted to four temporal segments; longer, more complex motions require extension.
- Final video quality remains bounded by the underlying generation model (VACE).
- CLaTr FID is notably higher than GenDoP (66.9 vs. 22.7), indicating room for improvement in trajectory realism.

## Related Work & Insights

- **vs. GenDoP**: GenDoP uses GPT to generate detailed director descriptions that guide autoregressive camera path generation, with the LLM playing only an auxiliary descriptive role. LAMP instead has the LLM directly output executable motion programs.
- **vs. ET**: ET uses LLM-generated cinematic descriptions to guide a diffusion model in predicting trajectories. LAMP bypasses diffusion entirely, using DSL for deterministic mapping.
- **vs. CameraCtrl/EPiC**: These methods control only the camera under the assumption of static objects. LAMP controls both jointly.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Reframing motion control as program synthesis, combined with a cinematographically grounded DSL, is a highly original contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Quantitative comparisons across multiple benchmarks, with ablations and a user study.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is clear, the method is presented in a well-structured hierarchy, and figures are intuitive.
- **Value**: ⭐⭐⭐⭐⭐ — Significant advancement for controllable video generation; the framework is elegant and extensible.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] DriveLaW: Unifying Planning and Video Generation in a Latent Driving World](drivelaw_unifying_planning_and_video_generation_in_a_latent_driving_world.md)
- [\[AAAI 2026\] MotionCharacter: Fine-Grained Motion Controllable Human Video Generation](../../AAAI2026/video_generation/motioncharacter_fine-grained_motion_controllable_human_video_generation.md)
- [\[CVPR 2026\] FlashMotion: Few-Step Controllable Video Generation with Trajectory Guidance](flashmotion_fewstep_controllable_video_generation.md)
- [\[CVPR 2026\] Training-free Motion Factorization for Compositional Video Generation](training-free_motion_factorization_for_compositional_video_generation.md)
- [\[CVPR 2026\] PoseGen: In-Context LoRA Finetuning for Pose-Controllable Long Human Video Generation](posegen_in-context_lora_finetuning_for_pose-controllable_long_human_video_genera.md)

</div>

<!-- RELATED:END -->

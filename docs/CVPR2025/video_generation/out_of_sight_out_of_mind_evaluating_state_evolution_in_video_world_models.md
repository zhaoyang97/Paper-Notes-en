---
title: >-
  [Paper Note] Out of Sight, Out of Mind? Evaluating State Evolution in Video World Models
description: >-
  [CVPR 2025][Video Generation][World Model Evaluation] StEvo-Bench proposes a benchmark to evaluate the capabilities of video world models in "unobserved state evolution"—testing whether world models can continue to correctly reason about state changes when physical processes are unobserved (due to camera movement, occlusion, or turning off lights). The results reveal a severe "out of sight, out of mind" deficiency, with all current frontier models (e.g., Veo 3…
tags:
  - "CVPR 2025"
  - "Video Generation"
  - "World Model Evaluation"
  - "State Evolution"
  - "Physical Reasoning"
  - "Benchmark"
date: 2026-05-08
content_hash: 01da6571118b8bb2
---

# Out of Sight, Out of Mind? Evaluating State Evolution in Video World Models

**Conference**: CVPR 2025  
**arXiv**: [2603.13215](https://arxiv.org/abs/2603.13215)  
**Code**: To be confirmed  
**Area**: Video World Models  
**Keywords**: World Model Evaluation, State Evolution, Physical Reasoning, Video Generation, Benchmark

## TL;DR
StEvo-Bench proposes a benchmark to evaluate the capabilities of video world models in "unobserved state evolution"—testing whether world models can continue to correctly reason about state changes when physical processes are unobserved (due to camera movement, occlusion, or turning off lights). The results reveal a severe "out of sight, out of mind" deficiency, with all current frontier models (e.g., Veo 3, Sora 2 Pro) achieving task success rates of less than 10%.

## Background & Motivation
**Background**: While video world models (such as Sora, Veo, and Cosmos) have achieved significant progress in visual quality, evaluations of their physical understanding remain insufficient.

**Limitations of Prior Work**: Existing evaluations focus primarily on visual quality (FVD, FID) and the physical consistency of visible objects, neglecting the "state evolution during unobserved periods"—the true dividing line between genuine world understanding and mere video generation.

**Key Challenge**: A true world model should understand that physical processes exist independently of the observer (e.g., a spilled glass will continue to overflow even when unobserved), whereas video generation models might merely "generate what is seen."

**Goal**: How to evaluate whether video world models truly understand that states continue to evolve while unobserved?

**Key Insight**: Designing experiments that control observation conditions—occlusion, turning off lights, and camera movement—followed by restoring the observation to check if the state has evolved correctly.

**Core Idea**: Occlude/move away $\rightarrow$ restore observation $\rightarrow$ verify if the state has progressed correctly. This distinguishes "genuine world understanding" from "mere video interpolation."

## Method

### Overall Architecture
StEvo-Bench = 225 tasks (covering various physical processes: fluids, collisions, elasticity, thermodynamics, etc.) + 3 observation control modalities (occlusion, turning off lights, camera movement) + 5 expert VLM verifiers (evaluating state progression, physical plausibility, and consistency).

### Key Designs

1. **Observation Control Mechanisms**:

    - **Occlusion Control**: Occluding the object after the physical process begins, and removing the occlusion after a period of time.
    - **Light Control**: Turning off the lights to darken the scene, and turning them back on after a period of time.
    - **Camera Control**: Moving the camera away for a period of time before returning to the original perspective.
    - **Design Motivation**: These three modalities test different levels of "unobservability"—occlusion is local, turning off lights is global, and camera movement is viewpoint-dependent.
    - Coverage of 9 major physical process categories: fluid dynamics (spilling a water cup, liquid evaporation), collisions and elasticity (ball bouncing, dominoes), thermodynamics (ice melting, water boiling), chemical changes (rusting, discoloration), gravity (freefall, hourglass), elastic deformation (spring compression), diffusion (ink diffusion), combustion (candle burning), and biological processes (seed germination). Each category contains 25 scenarios.

2. **Multi-dimensional VLM Verifiers**:

    - **Function**: Five specialized VLM judges evaluate different dimensions of the generated videos.
    - **Mechanism**: Tracking state progression (whether the physical process advanced), physical plausibility (whether the progression makes physical sense), and temporal consistency (whether the state matches expectations when observation is restored), among others.
    - **Design Motivation**: Automated evaluation eliminates manual annotation costs while providing multi-dimensional diagnosis.

3. **Failure Mode Classification**:

    - **Evolution Stagnation**: The physical process completely halts during the unobserved period—"frozen when unseen," accounting for ~60% of failure cases.
    - **Inconsistent Restoration**: The state upon restoring observation contradicts the expected progression (e.g., ice remains fully intact when it should have half-melted), accounting for ~25%.
    - **Incorrect Evolution Direction**: The model generates a state change but in the wrong direction (e.g., liquids flowing upwards), accounting for ~10%.
    - **Correct Generation with Incorrect Scene Understanding**: Visually plausible but violating physical causality (e.g., a flame disappearing after lights are turned off), accounting for ~5%.

## Key Experimental Results

### Main Results

| Model | Task Success Rate↑ | State Progression↑ | Description |
|------|-----------|----------|------|
| Veo 3 | 8.7% | 17.4% | Best-performing model, yet still extremely poor |
| Sora 2 Pro | 8.1% | 13.1% | Close to Veo |
| Cosmos-Predict1 | 5.3% | 9.8% | Best open-source model |
| Kling 2.0 | 4.2% | 8.1% | Asian commercial model |
| Camera control models | <5% | <5% | Worst-performing category |

### Results by Observation Modality

| Control Modality | Veo 3 Success Rate | Sora 2 Pro Success Rate |
|---------|-------------|------------------|
| Occlusion | 12.3% | 11.2% |
| Light Off | 8.1% | 7.6% |
| Camera Moved Away | 5.7% | 5.5% |

Camera movement represents the toughest condition—requiring the model to infer state changes of unseen regions under a completely new perspective, whereas occlusion retains at least some scene context.

### Key Findings
- **All frontier video world models achieve <10% success rate**—indicating that current models fundamentally lack true state evolution performance.
- **Evolution stagnation is the most common failure mode**—where models "freeze" physical processes during unobserved periods.
- **Camera control settings perform worse than occlusion/light control**—demonstrating that models rely heavily on continuous visual observation.
- Although Veo 3 and Sora 2 Pro exhibit extremely high visual quality, their physical understanding remains close to zero.
- This exposes a fundamental gap between current video generation models and genuine "world models."

## Highlights & Insights
- **Precise diagnosis of "Out of Sight, Out of Mind"**: The title perfectly captures the findings—current models indeed suffer from an "out of sight, out of mind" deficiency.
- **Innovation in evaluation paradigm**: Distinguishing "video generation" from "world understanding" by controlling observation conditions is a clean yet profound experimental design.
- **A reality check for World Model hype**: Mathematically proving that models like Sora and Veo are not true world models, providing clear direction for future research in this field.
- **Scalability of VLM verifiers**: The framework of five expert VLM verifiers can be scaled to more types of physical processes without requiring specialized evaluation metrics designed for each phenomenon.
- **Connection to embodied AI**: If video world models cannot preserve unobserved state evolution, robotics planning based on these models will fail when objects are occluded—presenting a fundamental caution for WM-based robotic policies.

## Limitations & Future Work
- The scale of 225 tasks is relatively limited and may not suffice to cover all types of physical processes.
- The VLM verifiers themselves are prone to hallucinating (inaccurate judgments); the authors report an ~85% agreement rate between VLM and human judgments.
- Some models were excluded due to generation artifacts, which may introduce selection bias.
- Synthetic controlled scenes may not fully replicate the complexity of the real world.
- Only single-object physical processes were tested, missing evaluations of multi-object interactions (such as collision chains).

## Related Work & Insights
- **vs PhysicsBench**: PhysicsBench evaluates the physical consistency of visible states. StEvo-Bench tests unobserved state evolution—requiring a deeper level of understanding.
- **vs TOMATO**: TOMATO evaluates temporal reasoning but does not control observation conditions.
- **vs WorldSimBench**: WorldSimBench focuses on the physical realism of interactive simulations, whereas StEvo-Bench focuses on state reasoning under passive observation.
- **Insight**: A true world model needs to maintain internal representations of physical states within a latent space rather than merely performing video extrapolation. This suggests that future WMs may require explicit state-variable tracking mechanisms, akin to the state vectors in physics engines.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The evaluation perspective of "unobserved state evolution" is highly original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multiple frontier models, three control modalities, and five-dimensional validation.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Exquisite title and strong argumentation.
- **Value**: ⭐⭐⭐⭐⭐ Raises fundamental skepticism about the video world model domain, with far-reaching impacts.

## Supplementary Notes
- The core philosophical question of this paper traces back to Piaget’s concept of "object permanence"—a cognitive capability infants acquire at around 8 months of age, which current state-of-the-art AI models still lack.
- The findings raise serious doubts about the feasibility of utilizing World Models for robotic planning—if a model cannot track the states of occluded objects, WM-based planning will fail frequently.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Long-Context State-Space Video World Models](../../ICCV2025/video_generation/long-context_state-space_video_world_models.md)
- [\[CVPR 2025\] Navigation World Models](navigation_world_models.md)
- [\[CVPR 2025\] Mind the Time: Temporally-Controlled Multi-Event Video Generation](mind_the_time_temporally-controlled_multi-event_video_generation.md)
- [\[CVPR 2026\] Reasoning Diffusion for Unpaired Test Time Out-of-distribution Text-Image to Video Generation](../../CVPR2026/video_generation/reasoning_diffusion_for_unpaired_test_time_out-of-distribution_text-image_to_vid.md)
- [\[CVPR 2025\] World2Act: Latent Action Post-Training via Skill-Compositional World Models](world2act_latent_action_post-training_via_skill-compositional_world_models.md)

</div>

<!-- RELATED:END -->

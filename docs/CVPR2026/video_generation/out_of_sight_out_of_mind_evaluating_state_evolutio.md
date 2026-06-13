---
title: >-
  [Paper Note] Out of Sight, Out of Mind? Evaluating State Evolution in Video World Models
description: >-
  [CVPR 2026][Video Generation][video world models] This paper introduces StEvo-Bench, a benchmark comprising 225 tasks that evaluates whether video world models can correctly continue evolving scene states during unobserv…
tags:
  - "CVPR 2026"
  - "Video Generation"
  - "video world models"
  - "state evolution"
  - "occlusion testing"
  - "benchmark"
  - "physical consistency"
date: 2026-05-08
content_hash: 43dff4e4cbb3b4c8
---

# Out of Sight, Out of Mind? Evaluating State Evolution in Video World Models

**Conference**: CVPR 2026
**arXiv**: [2603.13215](https://arxiv.org/abs/2603.13215)  
**Code**: [Project Page](https://glab-caltech.github.io/STEVOBench/)  
**Area**: LLM Evaluation
**Keywords**: video world models, state evolution, occlusion testing, benchmark, physical consistency

## TL;DR
This paper introduces StEvo-Bench, a benchmark comprising 225 tasks that evaluates whether video world models can correctly continue evolving scene states during unobserved intervals—induced by inserting occlusions or redirecting the camera during video generation. Experiments reveal that state-of-the-art models (including Veo 3 and Sora 2 Pro) achieve success rates below 10%, exposing a fundamental tendency of current video models to couple state evolution tightly with pixel-level observation.

## Background & Motivation
1. **Background**: Rapid advances in video generation have led researchers to characterize video models as "world models," expecting them to simulate real-world physical processes. Current video world models include general-purpose video generation models (Veo 3, Sora 2 Pro, WAN 2.2, etc.) and camera-controlled video models (Genie 3, HunyuanWorld, etc.).
2. **Limitations of Prior Work**: Existing benchmarks evaluate only subsets of world model capabilities—physics-intuition benchmarks (e.g., VideoPhy) assess only physical correctness, while consistency benchmarks (e.g., MIND) assess only memory and consistency. No existing benchmark simultaneously evaluates whether states continue to evolve correctly when observation is interrupted.
3. **Key Challenge**: In the real world, physical processes are observer-independent—water continues to flow when occluded, ice continues to melt when unseen. However, video models "simulate the world" by generating pixel frames, and their internal states may be strongly coupled to pixel-level observations.
4. **Goal**: To design a systematic evaluation framework that answers the question: can video world models decouple state evolution from observation?
5. **Key Insight**: Observation is interrupted in two ways—by inserting occluders into the scene (cardboard, curtains, or turning off lights) or by controlling the camera to look away—and the state upon observation resumption is then assessed for correctness.
6. **Core Idea**: The "occlude-then-reveal" experimental paradigm is used to systematically demonstrate that current video world models fail to decouple state evolution from pixel-level observation.

## Method

### Overall Architecture
The StEvo-Bench pipeline consists of three stages: (1) **Task Construction**—an initial image and text prompt drive the video model to generate an evolution process containing occlusion; (2) **Control Verification**—checks whether occlusion control and action control were successfully executed; (3) **Evolution Evaluation**—state progress, physical plausibility, and consistency are assessed on videos that pass control verification.

### Key Designs

1. **Task Construction System (6 categories, 225 tasks)**:

    - **Function**: Covers common physical evolution processes encountered in the real world.
    - **Mechanism**: Each task is specified by an initial image and a text prompt, spanning six evolution categories: continuous processes (water flow/melting), kinematics (projectile motion/free fall), relational changes (dominoes), causal changes (switching lights on/off), state transitions (burning/expansion), and expected behaviors (commonsense actions of humans/animals). Scene-internal occluders (cardboard/lights-off) are used for video generation models, while camera-redirect trajectories are used for camera-controlled models.
    - **Design Motivation**: These tasks reflect physical events that real-world agents encounter daily and span diverse types of physical processes, ensuring comprehensive evaluation coverage.

2. **Automated Verifier Pipeline (5 independent verifiers)**:

    - **Function**: Automatically evaluates multiple dimensions of generated videos and disentangles distinct failure modes.
    - **Mechanism**: Gemini 3.1 Pro serves as the VLM judge, and five specialist verifiers are constructed: (a) an *observation control verifier* that checks whether occlusion was successfully applied; (b) an *action control verifier* that checks whether the intended action occurred; (c) a *state progress verifier* that checks whether the state continued to evolve during occlusion (using unanimous-vote ensemble with $n=3$); (d) a *physical plausibility verifier* that checks whether the evolution is physically correct (majority vote); and (e) a *consistency verifier* that checks temporal coherence of the scene before and after occlusion.
    - **Design Motivation**: Decomposing verification into independent specialist modules offers two advantages: (1) fine-grained diagnosis of failure causes; and (2) each verifier asks only a single yes/no question, which is more reliable than asking the VLM to jointly assess multiple aspects.

3. **Evaluation Protocol Design**:

    - **Function**: A two-stage evaluation protocol ensures rigor.
    - **Mechanism**: The first stage checks whether controls were successfully executed (observation control + action control); failures are excluded. Videos passing control verification are then assessed for evolution success, requiring all three criteria to be simultaneously satisfied: state progress, physical plausibility, and consistency. The final task success is defined as $\text{control success} \times \text{evolution success}$.
    - **Design Motivation**: If the control itself fails (e.g., occlusion is not applied), it is impossible to judge whether evolution is correct; phased exclusion is therefore necessary.

### Verifier Reliability Validation
Three annotators labeled 180 videos, and verifier reliability was assessed using Accuracy, ROC-AUC, and MRA (Model Ranking Agreement). Results show that verifier–human agreement equals or exceeds inter-annotator agreement, validating the reliability of the automated evaluation.

## Key Experimental Results

### Main Results (Model performance on StEvo-Bench, %)

| Model Type | Model | Success | Progress | Physics | Coherence |
|---|---|---|---|---|---|
| Video Model | Veo 3 | 8.7 | 17.4 | 82.6 | 66.5 |
| Video Model | Sora 2 Pro | 8.1 | 13.1 | 85.5 | 69.7 |
| Video Model | WAN 2.2 | 0.9 | 7.7 | 52.0 | 58.4 |
| Camera-Controlled | Genie 3 | 0.0 | 2.9 | 15.2 | 27.3 |
| Camera-Controlled | HY-WorldPlay | 0.0 | 0.0 | 72.2 | 88.2 |
| Camera-Controlled | GEN3C | 0.0 | 0.0 | 30.6 | 82.4 |

### Ablation Study (Full observation vs. occlusion control, averaged over Veo 3 + Sora 2 Pro)

| Condition | State Progress | Task Success |
|---|---|---|
| Full observation | 84.6% | 46.2% |
| With observation control | 17.4% | 12.4% |

### Key Findings
- **All models achieve success rates below 10%**: Even the best-performing Veo 3 achieves only 8.7% overall success, revealing a fundamental limitation of current video world models.
- **Halted state progress is the most prevalent failure mode**: Upon introducing occlusion, the state progress rate drops sharply from 84.6% to 17.4%, confirming that models genuinely "stop evolving when they cannot see."
- **Coherence is the second major failure mode**: Even top-tier closed-source models achieve coherence rates of only ~67%, with object appearances frequently changing abruptly after occlusion is removed.
- **Camera-controlled models fail more severely**: Nearly all camera-controlled models exhibit state progress rates close to 0%, indicating a strong static-scene bias.
- **Evolution and camera control are mutually exclusive**: When camera-controlled models do generate dynamic content, they fail to execute camera redirects, and vice versa.
- **Memory modules do not facilitate state evolution**: Although VMem can perfectly recall the initial frame, it cannot advance state evolution; the memory architecture encourages appearance memorization rather than state evolution.
- **Training data bias is a root cause**: Camera-controlled models are predominantly trained on static scene renderings (3DGS reconstructions/Unreal Engine scenes) that lack videos with rich physical dynamics.

## Highlights & Insights
- **The evaluation paradigm is highly creative**: The "occlude-then-reveal" methodology for testing world model "understanding" parallels object permanence tests used in cognitive science to assess infant cognition. This paradigm is transferable to evaluating any AI system claiming to "understand the world."
- **Disentangled failure mode analysis is particularly valuable**: Rather than simply reporting an aggregate failure rate, failures are decomposed into three categories—halted progress, physical errors, and coherence loss—each pointing toward distinct directions for improvement.
- **Deep insight into video world model architecture**: Full bidirectional attention may be ill-suited for handling occlusion frames, since such frames carry no state evolution information. This suggests the need for novel attention mechanisms capable of distinguishing "informative frames" from "non-informative frames."

## Limitations & Future Work
- StEvo-Bench contains only 225 tasks, which may be insufficient to cover all types of physical processes.
- The automated verifier relies on Gemini 3.1 Pro, which may introduce its own biases.
- Only three forms of observation interruption are tested (occlusion, lights-off, and camera redirect); other modalities (e.g., blurring, fogging) remain unexplored.
- The paper is diagnostic rather than prescriptive; no solutions are proposed.
- The discussion of how to fix the identified issues is relatively superficial: while training data bias is identified as a cause, no concrete remedies are offered.

## Related Work & Insights
- **vs. VideoPhy/VideoPhy2**: These assess only physical correctness and do not test state evolution under occlusion; StEvo-Bench provides a more comprehensive evaluation.
- **vs. MIND**: MIND tests only memory consistency in static scenes, whereas StEvo-Bench evaluates continuous evolution in dynamic processes.
- **vs. WorldScore**: WorldScore is comprehensive but uses simple setups; StEvo-Bench focuses specifically on the critical dimension of "evolution during unobserved intervals."
- This paper serves as an important reference for researchers working on video world models, clearly identifying directions for improvement.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First systematic evaluation of the state evolution–observation decoupling capability in video world models, with a highly creative experimental design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 11 state-of-the-art models (open- and closed-source), with verifier reliability analysis and comparison against human annotations.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear and coherent narrative, in-depth failure mode analysis, and rich insights.
- Value: ⭐⭐⭐⭐⭐ Identifies fundamental limitations of video world models, providing important guidance for the field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Long-Context State-Space Video World Models](../../ICCV2025/video_generation/long-context_state-space_video_world_models.md)
- [\[ICLR 2026\] DrivingGen: A Comprehensive Benchmark for Generative Video World Models in Autonomous Driving](../../ICLR2026/video_generation/drivinggen_a_comprehensive_benchmark_for_generative_video_world_models_in_autono.md)
- [\[CVPR 2026\] U-Mind: A Unified Framework for Real-Time Multimodal Interaction with Audiovisual Generation](u-mind_a_unified_framework_for_real-time_multimodal_interaction_with_audiovisual.md)
- [\[CVPR 2026\] SeeU: Seeing the Unseen World via 4D Dynamics-aware Generation](seeu_seeing_the_unseen_world_via_4d_dynamics-aware_generation.md)
- [\[CVPR 2026\] DriveLaW: Unifying Planning and Video Generation in a Latent Driving World](drivelaw_unifying_planning_and_video_generation_in_a_latent_driving_world.md)

</div>

<!-- RELATED:END -->

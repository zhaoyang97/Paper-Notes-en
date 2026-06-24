---
title: >-
  [Paper Note] QUANTIPHY: A Quantitative Benchmark Evaluating Physical Reasoning Abilities of Vision-Language Models
description: >-
  [CVPR 2026][VLM Reasoning][Physical reasoning] QUANTIPHY is the first **quantitative** benchmark evaluating the physical reasoning abilities of VLMs. Given a video and a single physical prior of an object (size / velocity / acceleration in real-world units), the model is required to infer the **numerical values** of the target object's kinematic quantities. Using 3.3K+ video-text instances and numerical ground truths, it reveals a gap where current VLMs are "linguistically pl…
tags:
  - "CVPR 2026"
  - "VLM Reasoning"
  - "Physical reasoning"
  - "kinematic inference"
  - "quantitative evaluation"
  - "VLM benchmark"
  - "video understanding"
date: 2026-05-08
content_hash: 78a1572da965b8d2
---

# QUANTIPHY: A Quantitative Benchmark Evaluating Physical Reasoning Abilities of Vision-Language Models

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Puyin_QUANTIPHY_A_Quantitative_Benchmark_Evaluating_Physical_Reasoning_Abilities_of_Vision-Language_CVPR_2026_paper.html)  
**Code**: None  
**Area**: Multimodal VLM  
**Keywords**: Physical reasoning, kinematic inference, quantitative evaluation, VLM benchmark, video understanding

## TL;DR
QUANTIPHY is the first **quantitative** benchmark evaluating the physical reasoning abilities of VLMs. Given a video and a single physical prior of an object (size / velocity / acceleration in real-world units), the model is required to infer the **numerical values** of the target object's kinematic quantities. Using 3.3K+ video-text instances and numerical ground truths, it reveals a gap where current VLMs are "linguistically plausible but numerically systematically incorrect"—they rely more on pre-trained world knowledge rather than faithfully using the given visual and textual inputs.

## Background & Motivation

**Background**: Enabling AI to understand the physical world has always been a core challenge. Several benchmarks already exist around VLM physical understanding, covering kinematics, dynamics, object relations, and scene understanding (e.g., PhysBench, STAR, VSI-Bench, etc.).

**Limitations of Prior Work**: Most existing benchmarks are **VQA-style and qualitative**, evaluated using multiple-choice questions or descriptive QA. This setup cannot perform fine-grained evaluation. The paper provides a sharp example: if asked "how large is the car in the video" (ground truth 3 meters), answers of 3.1 meters and 31 meters are **both considered wrong** in a multiple-choice evaluation, yet the latter is ten times worse numerically. Discarding information about "how far off the numerical value is" prevents the advancement of VLMs toward real-world deployment (embodied AI, AR/VR, and autonomous driving all require quantitative physical quantities).

**Key Challenge**: The training methods of VLMs differ significantly from humans—they fit massive datasets and implicitly absorb physical laws. However, whether this "implicit absorption" translates into **reliable numerical reasoning abilities** has not been quantitatively answered. Traditional precise methods (such as FoundationPose) require extensive priors like color, depth, mesh, and camera parameters, which are largely unavailable in in-the-wild videos.

**Goal**: To establish a **numerical, scalable, and standardized** testbed for evaluating VLM kinematic inference capabilities, and to further analyze the factors influencing their reasoning (scene complexity, video availability, counterfactual priors, and chain-of-thought prompting).

**Key Insight**: The authors leverage a physical fact—the size, velocity, and acceleration of the same object are interrelated through an **unknown pixel-to-world scale factor $\omega$**. As long as one of these real-world quantities is given as a prior, combined with the corresponding measurable quantity in pixel space from the video, $\omega$ can be solved, allowing all other quantities to be restored to the real world. Thus, "quantitative kinematic inference" becomes a task with a unique numerical ground truth that can be strictly scored.

**Core Idea**: Use the quantitative paradigm of "given a single physical prior, find the numerical values of other kinematic quantities" to replace qualitative multiple-choice VQA, exposing the true level of VLMs being "reasonable in words but inaccurate in calculation."

## Method

### Overall Architecture
QUANTIPHY is not a model but a set of **task definitions + data construction + evaluation protocols**. The overall process consists of three steps: ① Formalizing "physical reasoning" into a kinematic inference task with numerical ground truths (relying on the pixel-to-world scale factor $\omega$); ② Collecting and labeling 560 unique videos and 3,391 questions from three sources (Blender simulation / Lab 4D reconstruction / Internet crawling), categorized into four task types by "Dimension × Physical Prior"; ③ Using Mean Relative Accuracy (MRA) for standardized scoring, accompanied by three types of diagnostic probes: scene complexity, counterfactual priors, and chain-of-thought.

The mathematical framework of the task: For fixed-camera videos, the object positions in pixel space $X^{pixel}_t$ can be read frame-by-frame, and pixel-wise velocity and acceleration are obtained via finite differences:

$$V^{pixel}_t \approx \frac{X^{pixel}_{t+dt}-X^{pixel}_t}{dt},\quad A^{pixel}_t \approx \frac{X^{pixel}_{t+2dt}-2X^{pixel}_{t+dt}+X^{pixel}_t}{dt^2}.$$

The video only provides kinematics in pixel units. By introducing an unknown scalar scale $\omega>0$ (unit [world length/pixel]), along the direction of motion, there are $S^{world}=\omega S^{pixel}$, $V^{world}_t=\omega V^{pixel}_t$, and $A^{world}_t=\omega A^{pixel}_t$. Given one real-world prior (one of size, velocity, or acceleration at a certain moment) plus the corresponding pixel quantity in the video, $\omega$ can be solved, and other world quantities are converted accordingly.

### Key Designs

**1. Kinematic Inference Task: Turning physical reasoning into problems with unique numerical ground truths via the scale factor $\omega$**

The fundamental issue with qualitative VQA is the lack of continuous numerical ground truths and the inability to measure "how far off" an answer is. QUANTIPHY closes the problem using the $\omega$ framework above: The model is given a world prior of a source object (from $\{S^{world}, V^{world}_t, A^{world}_t\}$) and must output the numerical value of a target object's kinematic quantity in world space. Since various kinematic quantities of the same object are linearly tied by $\omega$, an ideal agent simply needs to accurately read pixel coordinates from the video to solve for $\omega$ and precisely calculate the target quantity—meaning the task **theoretically has an exact solution**. VLMs should theoretically be able to surpass humans who rely on rough estimation. The task focuses on translational motion; rotational motion is left for future work ⚠️.

**2. Three-Axis Task Classification: Dimension × Physical Prior × Scene Difficulty**

For systematic coverage, the benchmark is organized along three axes. The first two define the core reasoning tasks: **Dimension** $\{2D, 3D\}$—2D assumes objects move only in the xy-plane with no depth change relative to the camera; 3D includes z-axis motion, where perspective projection means pixel displacement does not uniquely correspond to physical motion, requiring depth priors to disambiguate. **Physical Prior** $\{Static, Dynamic\}$—Static refers to providing a constant object size $S^{world}$; Dynamic refers to providing velocity $V^{world}_t$ or acceleration $A^{world}_t$ at a specific time. Crossing these two axes yields four task types: 2D-Static, 2D-Dynamic, 3D-Static, and 3D-Dynamic (abbreviated as 2S/2D/3S/3D in tables). The third axis, **Scene Difficulty** $\{SX, MX, SS, MS, SC, MC\}$, is used for diagnosis: the first letter indicates single (S) vs. multiple (M) moving objects, and the second letter indicates backgrounds that are solid color (X), simple texture (S), or complex scenes (C), specifically designed to analyze how background and object count modulate difficulty.

**3. Three-Source Data Construction + SAM Segmentation Augmentation: Balancing controllability and real distribution**

Data creation is non-trivial and uses different labeling methods based on the source. **Blender Simulation**: Renders scenes that are both realistic and physically plausible for 2D/3D motion, with automated scripts extracting ground truth size/velocity/acceleration at any moment; it can also synthesize extreme scales difficult to capture in reality (from red blood cells in vessels to galactic motion) and test robustness by fixing motion while varying background and visual noise. **Lab Collection**: Uses multi-view stereo + depth cameras for 4D reconstruction (3D space + time), calculating world-coordinate motion and per-pixel segmentation for real motions like free fall, sliding, swinging, and bouncing; object depth from a primary camera is chosen as the depth prior. **Internet Crawling**: To extend to OOD scenarios, high-quality monocular videos with relatively static cameras and known-size reference objects (e.g., standard coins) are manually selected. Size/displacement are labeled in pixel space, then mapped to world scale using the reference prior. Additionally, **SAM** is used to segment moving objects in solid-color videos, doubling the data without extra labeling and supporting controlled analysis of "background complexity." The final dataset includes 560 unique videos and 3,391 questions, with each segment roughly 2–3 seconds totaling about 120MB ⚠️.

**4. MRA Metric + Three Diagnostic Probes: Measuring "how accurate" and "why inaccurate"**

Exact matching is too fragile for continuous, noisy physical measurements. Following VSI-Bench, QUANTIPHY uses **Mean Relative Accuracy (MRA)** as the main metric: given a set of confidence thresholds $C=\{0.1,0.2,\dots,0.9,0.95\}$, for prediction $\hat y$ and ground truth $y$, it is defined as:

$$\mathrm{MRA}=\frac{1}{10}\sum_{\varepsilon\in C}\mathbb{1}\!\left(\frac{|\hat y - y|}{|y|}<1-\varepsilon\right),$$

which averages whether the relative error is below a series of tolerances, making it more calibrated and robust than single-threshold accuracy. During scoring, the MRA mean of all validly answered questions in each task category is taken, and the total score is the unweighted average of the four categories. Each question allows up to 5 model queries; if any response contains a parseable numerical value, it stops early, otherwise, it is marked as a failure for that question. Three diagnostic probes are included: scene context (background/object count), fidelity to video and prior (video+prior vs. prior-only vs. counterfactual scaled prior), and structured Chain-of-Thought (breaking the problem into "source pixel attribute → scale → target pixel attribute → target world attribute").

## Key Experimental Results

### Main Results
Evaluated on 21 SOTA VLMs (6 closed-source + 15 open-source), reporting MRA (%) for the four task types and their average. Key conclusion: Quantitative kinematic inference remains very difficult for current VLMs, with even the strongest systems failing to reach human levels.

| Model | Scale | 2S | 2D | 3S | 3D | Avg |
|------|------|----|----|----|----|-----|
| Human Baseline | – | 50.0 | 59.1 | 55.2 | 57.9 | **55.6** |
| ChatGPT-5.1 | – | 46.3 | 56.2 | 51.5 | 58.3 | **53.1** (Best) |
| Gemini-2.5 Pro | – | 44.8 | 57.5 | 42.4 | 53.7 | 49.6 |
| Gemini-2.5 Flash | – | 40.3 | 53.2 | 43.6 | 57.4 | 48.6 |
| Qwen3-VL-Instruct-32B | 32B | 35.8 | 51.6 | 43.2 | 53.4 | 46.0 (Best Open) |
| ChatGPT-5 | – | 36.6 | 35.0 | 25.9 | 33.1 | 32.6 |
| Claude Sonnet 4.5 | – | 19.6 | 23.0 | 19.6 | 29.1 | 22.8 |
| LLaVA-13B | 13B | 14.4 | 22.1 | 8.0 | 16.5 | 15.2 |
| Fuyu-8B | 8B | 9.5 | 14.7 | 9.5 | 16.2 | 12.5 |

- ChatGPT-5.1 only slightly surpassed humans in 2D-Dynamic, but **no model's average score exceeded the human baseline of 55.6**.
- The strongest systems cluster around ~50% MRA, indicating they still systematically underutilize visual precision and physical priors.

### Ablation Study (Probes on 161 2D Subsets)
The authors compared video+prior against prior-only, counterfactual, and CoT settings on a controlled subset (selected units in MRA %):

| Setting | Meaning | Representative Observation |
|------|------|-----------|
| Video + Prior | Full input | Default, ~56.1 (top models) ⚠️ |
| Prior only | Remove video, keep prior and question | **Small gap** from full setting (e.g., 39.0 vs 56.1, slight drop for most) |
| Counterfactual | Multiply prior by $\zeta\in\{0.001,\dots,700\}$ | Scores **plummet**: ~80% drop for most, ~70% drop for best |
| CoT | Four-step CoT decomposition | **Only 3** out of 21 models showed improvement |

### Key Findings
- **VLMs rely more on memory than visual measurement**: The score drop after removing the video (prior-only) is small, indicating models can achieve decent scores using internal priors of "typical object size/velocity." Video frames add limited value—they act more like "strong guessers conditioned on text prompts."
- **VLMs do not truly reason**: In the counterfactual setting where numerical priors are scaled, correct reasoning should result in proportional scaling ($y_{cf}=\zeta\cdot y$), but outputs remain close to the original magnitudes of real-world experience. MRA consistently plummets by ~80%—proving they do not faithfully use the given precise priors.
- **CoT provides limited help**: Breaking the problem into pixel → scale → target pixel → target world only significantly benefited a few models like ChatGPT-5 and Fuyu-8B; most showed no improvement.
- **Scene context is counter-intuitive**: Background complexity has a minor impact; complex scenes (C) are often slightly better than simple textures (providing implicit references like tiles/windows/signs). Performance is consistently better when **more objects** are present (MX/MS/MC) compared to single objects, as extra objects serve as implicit comparison scales.

## Highlights & Insights
- **Shifting physical reasoning from qualitative to quantitative**: Using a scale factor $\omega$ to linearly tie size/velocity/acceleration allows the task to have a unique numerical ground truth and a theoretical exact solution, enabling MRA to measure "how far off" rather than just providing binary correctness.
- **Cleverly designed counterfactual probes**: By multiplying priors by extreme scalars, the study directly decouples whether the model is reasoning based on the prior or guessing based on memory; the ~80% MRA drop is highly persuasive as a transferable diagnostic paradigm for input-fidelity.
- **Reusable complementarity of three-source data**: Blender provides perfect ground truths and extreme scales, Lab 4D provides real physics, and Internet crawling provides OOD distributions. This "simulation + real collection + crawling + segmentation augmentation" recipe is transferable to other video benchmarks needing precise physical ground truths.
- **Humans are not the ceiling**: An ideal agent should surpass humans (who can only estimate) by using precise pixel data, yet current models cluster below human levels, quantifying the room for improvement.

## Limitations & Future Work
- **Only covers translational motion**: Rotational motion was explicitly deferred to future work ⚠️, making the evaluation of complete rigid body dynamics incomplete.
- **Small diagnostic subset**: The fidelity and CoT analyses in Sections 5.2/5.3 were limited by resources to 161 2D instances. MRA values differ slightly from the main table, and the statistical robustness of the conclusions requires larger-scale verification.
- **Dependent on static/relatively static camera assumptions**: Scale factor inference is built on the assumption of a relatively static camera. Applicability to moving cameras, strong perspective, or violent motion scenes is not fully covered.
- **Future improvement ideas**: Incorporating rotation and more complex multi-body interactions; introducing explicit geometric/depth modules to compare with end-to-end VLMs; exploring training objectives that force models to scale according to priors (e.g., counterfactual consistency loss).

## Related Work & Insights
- **vs. PhysBench / STAR (Qualitative Physical Benchmarks)**: These use multiple-choice/descriptive questions to cover relations, scenes, and dynamics. QUANTIPHY uses numerical output and MRA scoring, capturing order-of-magnitude differences like "3.1m vs 31m" that are flattened in multiple-choice evaluations.
- **vs. VSI-Bench / Super-VSI (Numerical Spatial Understanding)**: VSI-Bench provides numerical metrics for basic spatial understanding but is limited to static objects and focuses on perception. QUANTIPHY focuses on size/velocity/acceleration inference of **moving objects**, emphasizing emergent quantitative reasoning over pure perception.
- **vs. FoundationPose and other precise pose methods**: These require extensive priors like color/depth/mesh/camera parameters to locate objects precisely, which are unavailable in-the-wild. This paper instead asks if VLMs can derive precise kinematics end-to-end via implicit priors, positioning itself as "evaluation + diagnosis" rather than a "solver."

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First quantitative kinematic inference benchmark; closing physical reasoning into a task with numerical ground truths via the scale factor is innovative; the counterfactual probe design is particularly elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 21 VLMs + human baseline + three types of diagnostic probes; conclusions are logically progressive.
- Writing Quality: ⭐⭐⭐⭐ Task formalization and findings are clearly narrated; some key statistics (OCR noise in cached numerical tables) should be verified against the original.
- Value: ⭐⭐⭐⭐⭐ Reveals the systematic flaws of "VLMs being reasonable but inaccurate, relying on memory over input," setting a benchmark for quantitative physical reasoning in embodied AI/AR/autonomous driving.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SpatiaLQA: A Benchmark for Evaluating Spatial Logical Reasoning in Vision-Language Models](spatialqa_a_benchmark_for_evaluating_spatial_logical_reasoning_in_vision-languag.md)
- [\[ICLR 2026\] Spatial-DISE: A Unified Benchmark for Evaluating Spatial Reasoning in Vision-Language Models](../../ICLR2026/vlm_reasoning/spatial-dise_a_unified_benchmark_for_evaluating_spatial_reasoning_in_vision-lang.md)
- [\[ICLR 2026\] OmniSpatial: Towards Comprehensive Spatial Reasoning Benchmark for Vision Language Models](../../ICLR2026/vlm_reasoning/omnispatial_towards_comprehensive_spatial_reasoning_benchmark_for_vision_languag.md)
- [\[CVPR 2026\] Thinking in Dynamics: How Multimodal Large Language Models Perceive, Track, and Reason Dynamics in Physical 4D World](thinking_in_dynamics_how_multimodal_large_language_models_perceive_track_and_rea.md)
- [\[CVPR 2026\] IPR-1: Interactive Physical Reasoner](ipr-1_interactive_physical_reasoner.md)

</div>

<!-- RELATED:END -->

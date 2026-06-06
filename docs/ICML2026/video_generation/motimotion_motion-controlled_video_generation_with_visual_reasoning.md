---
title: >-
  [Paper Note] MotiMotion: Motion-Controlled Video Generation with Visual Reasoning
description: >-
  [ICML 2026][Video Generation][Motion Control] MotiMotion transforms sparse and imprecise user trajectories and text prompts into physically plausible and causally consistent motion trajectories and text descriptions thro…
tags:
  - "ICML 2026"
  - "Video Generation"
  - "Motion Control"
  - "Visual Reasoning"
  - "VLM"
  - "Physical Constraints"
date: 2026-05-08
content_hash: 49ce6bfa49a6c650
---

# MotiMotion: Motion-Controlled Video Generation with Visual Reasoning

**Conference**: ICML 2026  
**arXiv**: [2605.22818](https://arxiv.org/abs/2605.22818)  
**Code**: TBD  
**Area**: Video Generation / Controllable Generation  
**Keywords**: Motion Control, Visual Reasoning, VLM, Video Generation, Physical Constraints

## TL;DR
MotiMotion transforms sparse and imprecise user trajectories and text prompts into physically plausible and causally consistent motion trajectories and text descriptions through VLM reasoning. It then employs a **confidence-weighted** control strategy to guide a diffusion model to generate natural videos that conform to world knowledge and physical principles—achieving a physical realism score of 0.302 on MotiBench, significantly surpassing Wan-Move's 0.218 (+38%).

## Background & Motivation

**Background**: Image-to-video generation models have made breakthroughs in visual quality and semantic consistency. However, practical applications lack precise logical controllability—users can provide guidance via trajectories, bounding boxes, or optical flow, but this requires an exact understanding of motion details.

**Limitations of Prior Work**: Existing motion control methods (e.g., Wan-Move / MagicMotion) **assume that user inputs perfectly capture true motion dynamics and execute them strictly**. However, user-provided trajectories are often sparse, coarse, and physically inconsistent. For example, given the prompt "a hand lifting a block holding back dominoes," a user specifies the hand's trajectory but implicitly expects the dominoes to undergo a **chain reaction** after the constraint is removed—a causal relationship that models fail to reason about.

**Key Challenge**: Motion-controlled generation balances two extremes: (1) strict execution of user input leading to physical implausibility and lack of causality; (2) complete neglect of user intent resulting in a loss of controllability. The root cause is the lack of reasoning capability regarding visual context.

**Goal**: Construct an intelligent motion-controlled video generation framework that transforms fuzzy user intent into physically and causally consistent motion planning while preserving the user's spatio-temporal controllability.

**Key Insight**: VLMs possess powerful world knowledge and visual understanding capabilities, enabling them to understand the visual context provided by users and reason about implicit physical and causal logic. The problem is redefined as a two-stage "Reason-Generate" process: first using a VLM to transform sparse inputs into dense, physically plausible control signals, and then using a diffusion model to render the video.

**Core Idea**: Use training-free VLM reasoning to refine user trajectories and hallucinate secondary motions, and introduce **confidence weighting** to allow the generator to rely on its own generative priors rather than rigid execution in low-confidence regions.

## Method

### Overall Architecture
Two synergistic modules:
- **Phase 1 (VLM Reasoning & Planning)**: Given an input image, a visualization of user-drawn trajectories, and a text prompt, the VLM understands user intent and reasons about events to output: (1) detailed **causal description prompts** (supplementing primary motion with secondary consequences like collisions, deformations, and lighting changes); (2) a **set of refined trajectories** (correcting user trajectories + adding secondary trajectories). Users can iterate until satisfied.
- **Phase 2 (Confidence-Aware Generation)**: Inject refined prompts and trajectories into a Flow-Matching video generator. A confidence score is introduced—high-confidence trajectories impose strict constraints on generation, while low-confidence trajectories act as coarse-grained guidance, allowing the model to rely on generative priors to synthesize natural motion.

### Key Designs

1.  **VLM-driven Prompt & Motion Reasoning**:
    - **Function**: Transforms sparse and incomplete user inputs into dense, physically consistent motion plans and descriptions.
    - **Mechanism**: The VLM simultaneously receives three inputs: (a) coordinate sequences (in text form, normalized to $[0, 1]$), (b) input images with trajectory visualizations, and (c) optional text prompts—to reason about causal relationships based on visual context. It outputs refined prompts containing all secondary consequences of the primary motion; and output trajectories containing corrected user trajectories (preserving spatial intent but adjusting timestamps to reflect physical forces like friction/acceleration) and secondary trajectories (identifying reactive objects or static anchors).
    - **Design Motivation**: Addresses the issues of imprecise user trajectories and incomplete causality. The VLM’s world knowledge allows it to understand physical constraints (gear coupling) and common sense (objects falling after support is removed), enabling the generator to produce reasonable motion without needing to learn this knowledge explicitly.

2.  **Confidence-Aware Motion Control**:
    - **Function**: Allows the generator to flexibly adjust the strictness of trajectory execution based on different confidence levels.
    - **Mechanism**: Assigns a confidence score $s \in [0, 1]$ to each training trajectory ($s = 1$ for ground truth quality, $s \to 0$ for unreliable). Training applies degradation to low-confidence samples (affine transformations to simulate spatial uncertainty, linearization for temporal sparsity, Savitzky-Golay smoothing for over-smoothing). During inference, confidence is transmitted by scaling the Gaussian kernel intensity $G' = s \cdot G$—high scores produce strong peaks forcing the model to follow given coordinates, while low scores weaken signals to encourage the model to synthesize natural motion using generative priors.
    - **Design Motivation**: Handles inaccuracies in VLM predictions and user inputs. Since pre-trained video generators already possess strong natural dynamics priors, it is more effective to design an elastic mechanism for gradient transition between high and low confidence rather than forcing the model to learn artifacts from inaccurate data.

3.  **Iterative Refinement Loop**:
    - **Function**: Allows users to improve generation results through multi-round interaction.
    - **Mechanism**: The VLM not only predicts motion from static images and trajectories but also judges the naturalness of the generated video. Users can choose multiple rounds of refinement until satisfied, or let the VLM automatically declare full confidence.
    - **Design Motivation**: Handles potential imperfections in single-round reasoning (e.g., trajectory misunderstanding leading to camera zooming); iteration allows users to correct VLM reasoning errors and gradually approach the final goal.

### Implementation Details
Base generator: Wan 2.2 I2V-A14B (Flow-Matching). Motion is represented as $N$ point trajectories in a video of length $L$ and resolution $H \times W$. Each trajectory places a 2D Gaussian heatmap at its corresponding frame position, with the standard deviation scaled relative to video resolution and peak normalized to 1. Motion latents are projected via VAE encoding and concatenated with noise latents and reference image latents in the channel dimension before entering the DiT. Two-stage training (OpenVid 5K steps → 3K steps with 50% sample trajectory degradation). Gemini 3.1 Pro is used as the motion reasoning VLM.

## Key Experimental Results

### Main Results (MotiBench, VLM Auto-Evaluation)

| Method | Physical Realism ↑ | Photo Realism ↑ | Semantic Consistency ↑ |
|------|----------|----------|----------|
| MagicMotion | 0.157 | 0.550 | 0.343 |
| Wan-Move | 0.218 | 0.483 | 0.511 |
| **MotiMotion** | **0.302** | 0.520 | **0.665** |

### Double-Blind Forced Choice Test

| Comparison Scheme | Object Properties | Interactions | Overall | Human Evaluation |
|--------|--------|------|------|--------|
| MotiMotion vs MagicMotion | 72.9% | 80.8% | 78.0% | 97.9% |
| MotiMotion vs Wan-Move | 71.5% | 75.0% | 73.8% | 81.4% |

Physical realism improved by 38% compared to Wan-Move and 92% compared to MagicMotion. Human preference is approximately 50 percentage points higher than the 50% random baseline.

### Ablation Study

| Configuration | Physical Realism ↑ | Photo Realism ↑ | Semantic Consistency ↑ |
|------|----------|----------|----------|
| Base Motion Control Gen. | 0.166 | 0.389 | 0.337 |
| + Prompt Reasoning | 0.237 | 0.475 | 0.544 |
| + Motion Reasoning | 0.285 | 0.493 | 0.641 |
| + Confidence-Aware Control | 0.302 | 0.520 | 0.665 |

### Key Findings
- Gradual addition of components yields significant improvements; motion reasoning contributes the most (Physical Realism 0.237 → 0.285).
- **Cross-method reasoning validation**: Applying the reasoning module to MagicMotion / Wan-Move consistently improves physical realism and semantic consistency, demonstrating the module's generalizability.
- **Critical role of VLM reasoning**: Even without user text, reasoning based solely on images and trajectories improves physical realism from 0.177 to 0.229 and semantic consistency from 0.272 to 0.473.
- **Confidence mechanism corrects prediction errors**: In scenarios where VLM predictions are imprecise, such as dominoes bending downward or seesaws distorting, lowering confidence automatically corrects artifacts.
- **Feasibility of iterative refinement**: The clock example demonstrates successful modeling of gear-coupled motion after 4 iterations, where single-round attempts failed.

## Highlights & Insights
- **The Elegance of Decoupled Reasoning-Generation**: Instead of learning physical reasoning within the diffusion model, a training-free VLM acts as a "physics reasoner"—preserving the flexibility of the generative model while leveraging VLM world knowledge; this avoids the massive cost of having the video model learn common sense from data while enhancing interpretability.
- **Elegant Confidence-Weighted Design**: Not a binary choice between "strict execution vs. complete neglect" but a continuous trade-off. By training with simulated input degradation at different confidence levels, the model learns to automatically adapt to input quality, fitting naturally with the strong generative priors of video models.
- **Transformation from Sparse User Input to Dense Physical Planning**: Reveals a key insight—user inputs are essentially "intent" rather than "specifications." Using a VLM to understand intent and plan a complete causal chain is key to allowing the generator to produce natural results.

## Limitations & Future Work
- VLM-predicted trajectories may be spatially shaky or inaccurate (due to visual encoder resolution limits).
- The method is limited to image-to-video scenarios; video-to-video extension has not been explored.
- Dependence on VLM quality; reasoning may fail for complex physical scenarios not common in VLM training data (e.g., fluid simulation, multi-body systems).
- MotiBench contains only 62 pre-event images, limited in scale.
- The confidence scoring mechanism uses fixed simulations during training while being provided by the VLM during inference; a mismatch between the two could lead to unstable control.
- Improvements: Integrating physics simulators + VLM reasoning; expanding the scale and diversity of MotiBench; exploring online confidence learning.

## Related Work & Insights
- **vs MagicMotion**: Both perform motion control, but MagicMotion relies on dense user trajectories for strict execution; MotiMotion uses VLM to infer dense planning from sparse inputs—reducing user burden while significantly improving physical plausibility.
- **vs Wan-Move**: Wan-Move also uses trajectory injection based on the Wan framework but relies on fully supervised trajectory tracking; MotiMotion is more flexible in tracking strictness via confidence weighting, and the causal planning provided by VLM reasoning is a core innovation missing in Wan-Move.
- **vs Physics-Aware Generation** (via physics solvers or explicit constraints): This work uses physical knowledge implicitly encoded in the VLM, avoiding the overhead of explicit physics learning, though accuracy in extreme physical scenarios (complex fluids) may be insufficient.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Innovatively integrates VLM reasoning into the motion control pipeline; confidence-aware control is an elegant rethink of motion conditioning.
- Experimental Thoroughness: ⭐⭐⭐⭐ Automated VLM evaluation + human studies + ablations + cross-method validation + iterative analysis; MotiBench scale is small (62 images), and generalizability verification needs deepening.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logic, sufficient motivation, and vivid examples (dominoes / clock).
- Value: ⭐⭐⭐⭐⭐ Solves core issues in motion-controlled video generation (sparse imprecise input → natural controllable generation), with generalizability demonstrated across multiple existing methods.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Lighting-grounded Video Generation with Renderer-based Agent Reasoning](../../CVPR2026/video_generation/lighting-grounded_video_generation_with_renderer-based_agent_reasoning.md)
- [\[CVPR 2026\] Unified Camera Positional Encoding for Controlled Video Generation](../../CVPR2026/video_generation/unified_camera_positional_encoding_for_controlled_video_generation.md)
- [\[CVPR 2026\] Phantom: Physics-Infused Video Generation via Joint Modeling of Visual and Latent Physical Dynamics](../../CVPR2026/video_generation/phantom_physics-infused_video_generation_via_joint_modeling_of_visual_and_latent.md)
- [\[CVPR 2026\] LAMP: Language-Assisted Motion Planning for Controllable Video Generation](../../CVPR2026/video_generation/lamp_language-assisted_motion_planning_for_controllable_video_generation.md)
- [\[AAAI 2026\] MotionCharacter: Fine-Grained Motion Controllable Human Video Generation](../../AAAI2026/video_generation/motioncharacter_fine-grained_motion_controllable_human_video_generation.md)

</div>

<!-- RELATED:END -->

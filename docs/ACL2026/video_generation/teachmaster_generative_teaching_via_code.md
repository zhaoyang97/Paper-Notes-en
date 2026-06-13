---
title: >-
  [Paper Note] TeachMaster: Generative Teaching via Code
description: >-
  [ACL2026][Video Generation][Generative Teaching] TeachMaster proposes the Generative Teaching paradigm, using code as an interpretable intermediate representation for educational videos. It enables the collaboration of p…
tags:
  - "ACL2026"
  - "Video Generation"
  - "Generative Teaching"
  - "Code Intermediate Representation"
  - "Multi-Agent"
  - "Manim"
  - "Educational Video Generation"
date: 2026-05-08
content_hash: 7a7106aecc03ed51
---

# TeachMaster: Generative Teaching via Code

**Conference**: ACL2026  
**arXiv**: [2601.04204](https://arxiv.org/abs/2601.04204)  
**Code**: None  
**Area**: Video Generation / Educational Agents / Multimodal Content Generation  
**Keywords**: Generative Teaching, Code Intermediate Representation, Multi-Agent, Manim, Educational Video Generation

## TL;DR
TeachMaster proposes the Generative Teaching paradigm, using code as an interpretable intermediate representation for educational videos. It enables the collaboration of planning, code generation, narration, debugging, synchronization, and layout agents to generate complete course videos, achieving near-human quality while reducing the production cost of a 45-hour course to approximately 0.3% of traditional methods.

## Background & Motivation
**Background**: Online education has achieved large-scale course distribution, yet high-quality content still relies on manual design, recording, editing, and iterative revisions. While video generation models can produce visuals from text, mainstream E2E models excel at short clips or visual fragments rather than ensuring instructional structure, narrative logic, and editability.

**Limitations of Prior Work**: Educational videos are not ordinary short videos. They require accurate scripts, hierarchical knowledge organization, visual-narration synchronization, and the progressive unfolding of key concepts, while remaining easy for teachers to modify. Pixel-level models like Sora are black boxes with limited duration and difficult editing; agents that mimic human software operations depend on extensive trajectories and incur high training costs.

**Key Challenge**: Scalable educational content production requires automation, but educational quality demands structure, controllability, and traceable modifications. Pure video generation is highly automated but uncontrollable, while manual production offers high quality but is expensive and slow to update.

**Goal**: The authors aim to transform teachers from manual creators into high-level directors. By inputting only instructional intent or a course syllabus, a suite of generative agents completes the script, pages, animations, narration, debugging, and rendering, ultimately producing teachable, modifiable, and deployable video courses.

**Key Insight**: The paper argues that educational videos do not require direct pixel-level generation. For explanatory, conceptual, and visualization-based courses, code itself provides a superior intermediate representation: it can express layouts, animations, colors, timelines, and object relationships, while facilitating debugging, synchronization, and manual editing.

**Core Idea**: Code is used to bridge instructional semantics and video rendering. The "syllabus-to-video" process is decomposed into a three-stage multi-agent pipeline—content planning, presentation generation, and quality validation—turning educational video generation into an interpretable, editable, and verifiable procedural production process.

## Method
The core of TeachMaster is not a single large model generating a video, but an engineering pipeline for course production. Inputs can be keywords or a course syllabus; outputs include the generated video $V_{out}$ and the lecture script $L_{out}$. The system converts abstract teaching intent into page-level blueprints, then into executable Manim code and narration, and finally into deliverable videos via debugging, synchronization, layout optimization, and human interfaces.

### Overall Architecture
The workflow is divided into three stages. The first stage is content planning: the composition agent expands raw input into a full script and aligns it with the target duration via length refinement; the pagination agent then segments the long script into page-level units. Long text processing utilizes Chain-of-Agents to split the script into multiple fragments for separate pagination before merging.

The second stage is presentation generation. Each page blueprint enters a routing agent, where the system decides whether to follow standard code generation or an image-enhanced coding agent to introduce photorealistic or complex image assets. Subsequently, the narration agent generates a script based on the current page, the previous script, and the visual code. A TTS agent then converts the script into audio and estimates the speaking rate.

The third stage is quality validation. The debugging agent performs render-and-repair on the generated code, fixing syntax or runtime errors based on error logs; the synchronization agent inserts wait and trigger logic based on audio speed and event anchors in the code; the layout agent detects occlusion and crowding to adjust geometric positions; finally, a human-in-the-loop interface allows for natural language modifications or direct code editing.

### Key Designs
1. **Code as an Intermediate Semantic Medium for Educational Videos**:
    - **Function**: Transforms non-editable video pixel generation into executable, inspectable, and modifiable animation program generation.
    - **Mechanism**: TeachMaster primarily renders educational animations via Python / Manim. Visual objects, geometric relationships, colors, motion trajectories, wait times, and text elements are all defined in code. The model does not directly generate final pixels; instead, it generates programs that render into video segments, which are then synthesized with narration audio.
    - **Design Motivation**: Educational content emphasizes accuracy and maintainability. A code-based intermediate representation allows both teachers and the system to locate issues—such as moving a formula, adjusting animation pacing, or replacing image assets—without regenerating an entire black-box video.

2. **Multi-Agent Task Allocation from Content Planning to Presentation Generation**:
    - **Function**: Decomposes course production into controllable sub-tasks such as scripting, pagination, visuals, narration, and audio.
    - **Mechanism**: The composition agent handles semantic skeletonization, content expansion, and length refinement to turn intent into a script of appropriate length. The pagination agent handles page-level segmentation. The routing agent selects between Standard or ImageEnhanced modes, the coding agent generates visual code, the narration agent ensures coherence, and the TTS agent generates audio and tempo information.
    - **Design Motivation**: Educational videos require the coordination of narrative, visuals, and sound. Single-stream generation tends to lose balance between visual aesthetics, script completeness, and duration control; multi-agent division allows each module to optimize for a specific quality target.

3. **Debugging, Synchronization, and Layout Validation in a Rendering Loop**:
    - **Function**: Reduces the instability of LLM-generated code and multimodal misalignment to acceptable levels.
    - **Mechanism**: The debugging agent captures error stacks through execution and repairs the code, using standard templates as a fallback if retry thresholds are exceeded; the synchronization agent aligns event anchors with narration units; the layout agent detects overlapping objects and uses heuristic scans to find optimal coordinates to rewrite the code.
    - **Design Motivation**: Instructional video failures are often not "content errors" but issues like subtitles obscuring images, animations moving faster than narration, or code rendering failures. Treating these as executable validation and code repair is more scalable than manual frame-by-frame review.

### Loss & Training
The paper presents a system framework rather than an end-to-end trained video model. The visual synthesis engine is interchangeable: one version calls the Gemini-3 API, while another uses a local Qwen3-32B to generate high-fidelity Manim code. To enhance the code generation capabilities of Qwen3-32B, the authors constructed 3,735 pairs of high-quality human-annotated data, categorized by difficulty, and employed curriculum learning.

The training configuration utilized 8 NVIDIA A800 40GB GPUs, LoRA rank of 128, LoRA alpha of 256, DeepSpeed ZeRO-3, and a learning rate of $1 \times 10^{-5}$. The TTS agent uses Minimax. The system supports asynchronous task queues for multi-user course generation.

## Key Experimental Results

### Main Results
Video quality and efficiency were evaluated by comparing manual videos, Sora 2, and the Gemini and Qwen engine versions of TeachMaster. Quality metrics were scored by GPT-5.2 on a scale of 1 to 10. Preference validation was conducted by 3 human experts on 300 random videos, with a human-GPT evaluation consistency rate of 81.71%.

| Method | Spatial Clarity | Visual Richness | Teaching Logic | Image-Text Consist. | Factual Accu. | Overall Quality | Prod. Time (min) | Video Time (min) | Prod/Video Ratio |
|------|------------|------------|----------|----------|----------|----------|--------------|--------------|------------|
| Human | 8.22 | 7.31 | 8.38 | 8.29 | 9.24 | 8.29 | 795.00 | 32.50 | 24.46 |
| Sora 2 | 7.36 | 6.36 | 7.55 | 7.64 | 8.96 | 7.57 | 3.20 | 0.25 | 12.80 |
| TeachMaster-Gemini | 7.97 | 6.98 | 7.97 | 7.63 | 8.99 | 7.91 | 88.43 | 35.97 | 2.46 |
| TeachMaster-Qwen | 7.42 | 6.42 | 7.49 | 7.66 | 8.94 | 7.59 | 112.80 | 32.55 | 3.47 |

Script quality and cross-modal alignment further demonstrate the value of the code-centric paradigm. TeachMaster-Gemini achieved an overall script quality of 8.95, slightly higher than Human (8.84); TeachMaster-Qwen scored 8.79 in cross-modal alignment, surpassing Human (8.13) and Sora 2 (6.65).

| Method | Script Struct. | Narrative Coh. | Accuracy | Completeness | Consistency | Overall Script | Semantic Cov. | Ref. Accu. | A/V Symmetry | Overall Align. |
|------|----------|----------|--------|--------|--------|----------|----------|----------|----------|----------|
| Human | 8.90 | 9.11 | 9.05 | 8.32 | 8.84 | 8.84 | 8.17 | 7.94 | 8.28 | 8.13 |
| Sora 2 | 3.14 | 6.57 | 1.86 | 6.00 | 4.39 | 4.39 | 6.64 | 6.59 | 6.73 | 6.65 |
| TeachMaster-Gemini | 8.89 | 9.00 | 9.67 | 8.22 | 8.95 | 8.95 | 8.63 | 8.11 | 8.57 | 8.44 |
| TeachMaster-Qwen | 8.50 | 9.00 | 8.17 | 7.67 | 8.34 | 8.34 | 8.93 | 8.57 | 8.87 | 8.79 |

### Ablation Study
The paper does not include traditional module-removal ablations but demonstrates actual efficiency gains through deployment, user feedback, and cost statistics.

| Analysis Dimension | Value / Phenomenon | Meaning |
|----------|-------------|------|
| Deployment Scale | Served >1,000 educators, generated >30,000 mins of content | System is a real-world tool, not an offline demo |
| Subject Coverage | >40 subjects | Code representation generalizes across AI, biology, linguistics, etc. |
| Human Intervention | >75.2% of pages require no manual modification | Validation agents handle most generation issues |
| Page Modification | Avg. 1.88 interaction rounds for completion | Human-in-the-loop reduces post-editing costs |
| 45-hour Course Cost | Approx. $83.70 | Approx. 0.3% of traditional production costs |

### Key Findings
- TeachMaster's quality is more than just being "longer than Sora 2." It is significantly more stable in script structure, cross-modal alignment, and teaching logic because content is organized into pages and code rather than single video fragments.
- Sora 2's production time is short, but it only generates 0.25-minute clips (ratio of 12.80); TeachMaster-Gemini generates 35.97 minutes (ratio of 2.46), making it suitable for course-level production.
- Manual video quality remains the highest overall, but at an extreme time cost. TeachMaster's core value lies in achieving quality slightly below or occasionally exceeding manual work while reducing unit production costs by over an order of magnitude.
- The Qwen version performs best in cross-modal alignment, suggesting that local code generation models, while slightly weaker in visual richness, maintain better visual-verbal synchronization through explicit binding.

## Highlights & Insights
- The most important insight: educational video generation does not need to be pixel-centric. For many knowledge-sharing scenarios, code is closer to "controllable semantics" and better suited for debugging, synchronization, and manual editing.
- The system positions teachers as high-level directors rather than content workers replaced by AI. Teachers remain in control of instructional goals and logic, while agents handle tedious implementation—a setup more acceptable in educational settings.
- The multi-agent approach corresponds to the actual workflow of course production: scripting, paging, drawing, dubbing, pacing, layout, and review. Each stage has clear inputs and outputs, favoring industrialization.
- TeachMaster is also inspiring for scientific content generation. For instance, paper explanation videos, course visualizations, and experiment demonstrations can follow the "semantic blueprint -> code -> rendering -> synchronization" path.

## Limitations & Future Work
- Evaluation relies heavily on GPT-5.2 scoring and expert consistency checks. While scalable, final educational effectiveness requires long-term metrics like learner performance, retention, and comprehension depth.
- The system is well-suited for animations, charts, and conceptual visualization but may lag behind professional video models for live-action experiments, human-led lectures, or high-realism cinematic shots.
- Qwen3-32B's high-quality Manim code generation relies on 3,735 pairs of human-annotated data and high compute, presenting costs for migration to other frameworks or low-resource languages.
- Engineering metrics such as agent failure rates, debugging retry counts, and layout collision resolution success were not detailed. Future disclosure of module-level logs would benefit reproducibility.

## Related Work & Insights
- **vs E2E Video Generation**: Models like Sora 2 generate pixels directly but are black-box and unsuitable for long courses. TeachMaster sacrifices some visual realism for structural control, scalability, and editability.
- **vs AI Slide/Tutoring Systems**: Traditional systems often generate static content or short segments without audio. TeachMaster generates scripts, animations, narration, and synchronized video simultaneously.
- **vs Software-Operating Agents**: Mimicking human software use requires vast trajectories and large action spaces. TeachMaster's code generation provides a structured action space that is easier to debug.
- **vs Code2Video / Paper2Video**: Prior work has shown code usage for scientific illustrations. TeachMaster extends this into a course-level multi-agent production line with real-world deployment and cost data.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Code-centric generation exists, but the systematic multi-agent workflow and deployment scale under "Generative Teaching" are novel.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Includes multi-dimensional quality evaluation, human consistency, and deployment statistics; module-level ablation could be more detailed.
- Writing Quality: ⭐⭐⭐⭐☆ Clear motivation, complete system workflow, and data supports efficiency claims.
- Value: ⭐⭐⭐⭐⭐ Highly practical for educational production and editable multimodal generation, especially for large-scale online education.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Arbitrary Generative Video Interpolation](../../ICLR2026/video_generation/arbitrary_generative_video_interpolation.md)
- [\[CVPR 2026\] Generative Neural Video Compression via Video Diffusion Prior](../../CVPR2026/video_generation/generative_neural_video_compression_via_video_diffusion_prior.md)
- [\[CVPR 2026\] LightMover: Generative Light Movement with Color and Intensity Controls](../../CVPR2026/video_generation/lightmover_generative_light_movement_with_color_and_intensity_controls.md)
- [\[CVPR 2026\] PhysVid: Physics Aware Local Conditioning for Generative Video Models](../../CVPR2026/video_generation/physvid_physics_aware_local_conditioning_for_generative_video_models.md)
- [\[ICLR 2026\] DrivingGen: A Comprehensive Benchmark for Generative Video World Models in Autonomous Driving](../../ICLR2026/video_generation/drivinggen_a_comprehensive_benchmark_for_generative_video_world_models_in_autono.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] SceneWeaver: All-in-One 3D Scene Synthesis with an Extensible and Self-Reflective Agent
description: >-
  [NeurIPS 2025][3D Vision][3D scene synthesis] This paper proposes SceneWeaver, the first reflective agent framework for 3D scene synthesis…
tags:
  - "NeurIPS 2025"
  - "3D Vision"
  - "3D scene synthesis"
  - "LLM agent"
  - "self-reflection"
  - "tool calling"
  - "embodied AI"
date: 2026-05-08
content_hash: 919141a9f697d955
---

# SceneWeaver: All-in-One 3D Scene Synthesis with an Extensible and Self-Reflective Agent

**Conference**: NeurIPS 2025
**arXiv**: [2509.20414](https://arxiv.org/abs/2509.20414)  
**Code**: [Project Page](https://scene-weaver.github.io/)  
**Area**: 3D Vision
**Keywords**: 3D scene synthesis, LLM agent, self-reflection, tool calling, embodied AI

## TL;DR
This paper proposes SceneWeaver, the first reflective agent framework for 3D scene synthesis, which unifies multiple scene generation paradigms through a standardized and extensible tool interface. By employing a reason-act-reflect closed-loop for iterative refinement, it comprehensively outperforms existing methods in physical plausibility, visual realism, and semantic alignment.

## Background & Motivation
Indoor 3D scene synthesis is a long-standing research topic in computer vision and graphics. With the rise of embodied AI, the requirements for scenes have expanded beyond "visual realism" to "physical interactability" and "instruction controllability"—scenes must be interactable within simulators and capable of precisely responding to task-specific user instructions.

Existing methods each have strengths but none can fully satisfy all requirements: (1) **Data-driven generative models** (e.g., ATISS, DiffuScene, PhyScene) learn from datasets such as 3D-FRONT and produce realistic layouts, but are constrained by predefined scene types and dataset quality/diversity; (2) **Rule-based systems** (e.g., ProcThor, Infinigen) ensure physical plausibility through handcrafted constraints, but are rigid and difficult to extend; (3) **LLM-based methods** (e.g., LayoutGPT, Holodeck, I-Design) offer open-vocabulary understanding and semantic flexibility, but lack spatial reasoning capability and frequently produce physically implausible results.

More critically, several recent works have begun decomposing scene generation into multi-stage pipelines (e.g., LLM-based layout generation followed by 2D generative model refinement), yet these pipelines remain **static**—planning and execution are driven by fixed prompts and hard-coded logic, lacking the ability to make adaptive decisions based on generation feedback or to flexibly integrate new synthesis tools.

The core idea of SceneWeaver is to construct a **reflective agent framework**: existing scene synthesis methods are abstracted into modular tools, a self-reflective planner dynamically selects tools, iteratively refines the scene based on feedback, and executes modifications under physical constraints. This closed-loop design enables the agent to identify semantic inconsistencies, invoke targeted tools, and update the environment across successive iterations.

## Method

### Overall Architecture
SceneWeaver consists of two core components: (1) a **standardized and extensible tool interface** that categorizes diverse scene generation methods into modular tools by generation granularity; and (2) a **self-reflective planner** that, given a user query $q$ and tool set $\mathcal{D}$, progressively refines the scene state $s_t$ over $T$ iterations via a reason-act-reflect paradigm. At each step, the planner receives the previous reflection $v_{t-1}$, selects a tool $d_t$ for refinement, and a physics-aware executor handles collision resolution and constraint enforcement.

### Key Designs
1. **Standardized Scene Synthesis Tool Interface**:

    - **Scene Initializer**: Generates a full-scene layout as the starting point. Three sub-categories are included: data-driven generative models (extensible but limited to predefined types), real-to-sim methods (high quality but limited diversity), and LLM-based methods (open-vocabulary but weak in spatial reasoning).
    - **Microscene Implementer**: Populates scenes with fine-grained objects (e.g., keyboards and monitors on a desk). Includes LLM-based methods (semantically diverse but prone to spatial errors) and 2D-guided methods (visually realistic but limited by 2D-to-3D lifting).
    - **Detail Refiner**: Corrects errors, including rule-based constrained placement, 6-DoF pose refinement, and LLM-driven removal of semantically implausible objects.
    - Each tool is described via a **standardized tool card** containing fields for functional description, applicable scenarios, usage constraints, and input parameters, enabling plug-and-play integration of new tools.
    - **Design Motivation**: No single method can simultaneously satisfy realism, physical plausibility, and controllability; the standardized interface enables complementary strengths of multiple methods to be combined.

2. **Feedback-driven Self-reflective Planning**:

    - **Reflection Generation**: An MLLM (e.g., GPT-4) is invoked to generate a reflection $v_t$ for the current scene $s_t$, covering physical metrics (collision scores, boundary violations) and perceptual metrics (visual realism, functionality, layout coherence, user query alignment, completeness), accompanied by natural-language improvement suggestions.
    - **Self-reflective Planning**: The planner, conditioned on the current context and memory $m_t = (d_{t-l:t-1}, s_{t-l:t-1}, v_{t-l:t-1})$, summarizes current issues → ranks candidate tools → selects the most appropriate tool $d_t$ → generates tool-specific instructions. Tool confidence scores are dynamically adjusted based on historical performance—failures reduce confidence, and repeated failures trigger replanning.
    - **Design Motivation**: The closed-loop design enables the agent to learn and adapt from failures. Memory length is set to 1 to mitigate hallucination.

3. **Physics-aware Execution**:

    - The executor is built upon Infinigen and Blender.
    - Each iteration proceeds as: load the prior scene → apply layout modifications from tool $d_t$ → retrieve 3D meshes from a hybrid asset library (Objaverse, 3D-Future, Infinigen, etc.) → adjust object placement to satisfy relational constraints → run a fixed-step physics optimization to resolve collisions and boundary violations.
    - **Design Motivation**: Most tool outputs consist of 3D bounding-box layouts; the physics executor translates these into interactable 3D scenes while ensuring physical plausibility.

### Loss & Training
SceneWeaver does not involve end-to-end training. The core mechanism relies on the reasoning capability of pre-trained MLLMs (GPT-4), guided through carefully designed prompts and tool cards. The maximum number of iterations is set to 10 and the memory length to 1.

## Key Experimental Results

### Main Results (Common Room Types)

| Method | Bedroom #Obj | Bedroom #OB↓ | Bedroom #CN↓ | Bedroom Real.↑ | Bedroom Comp.↑ | Living Room #Obj | Living Room Real.↑ | Living Room Comp.↑ |
|--------|-------------|-------------|-------------|---------------|---------------|-----------------|-------------------|-------------------|
| PhyScene | 3.3 | 0.1 | 0.3 | 5.7 | 4.0 | 8.0 | 5.2 | 3.3 |
| LayoutGPT | 5.4 | 1.0 | 1.3 | 7.5 | 4.2 | 8.4 | 6.4 | 3.6 |
| I-Design | 9.6 | 0.0 | 0.0 | 8.6 | 6.1 | 9.7 | 8.4 | 5.9 |
| Holodeck | 32.2 | 0.0 | 38.5 | 8.6 | 6.2 | 23.0 | 8.9 | 8.1 |
| **SceneWeaver** | **14.0** | **0.0** | **0.0** | **9.2** | **9.4** | **17.3** | **9.1** | **8.7** |

### Ablation Study

| Configuration | #Obj | #OB↓ | #CN↓ | Real.↑ | Func.↑ | Lay.↑ | Comp.↑ |
|---------------|------|------|------|--------|--------|-------|--------|
| w/o Reflection Module | 25.0 | 0.0 | 0.0 | 8.0 | 8.3 | 6.3 | 6.3 |
| w/o Physics Optimization | 27.3 | 0.7 | 2.0 | 8.3 | 9.3 | 6.7 | 7.7 |
| Multi-step Planning (non-iterative) | 29.3 | 0.0 | 0.0 | 8.3 | 7.7 | 7.0 | 7.3 |
| **Full SceneWeaver** | **34.7** | **0.0** | **0.0** | **9.0** | **9.3** | **7.3** | **7.7** |

### Key Findings
- SceneWeaver leads comprehensively across all average metrics for open-vocabulary scene generation: #Obj=36.5, Real.=8.8, Func.=9.4, Comp.=8.0, with zero physical constraint violations.
- Holodeck achieves the highest raw object count (#Obj=32.2) but suffers from severe collisions (#CN=38.5), demonstrating that simply populating more objects does not yield higher scene quality.
- Removing the reflection module leads to significant degradation in semantic quality; self-reflective iterative refinement outperforms one-shot multi-step planning.
- Tool combination ablations confirm that the complete combination of initializer + implementer + refiner achieves the best performance, with the three tool categories being mutually complementary.
- Data-driven methods outperform LayoutGPT on physical metrics, indicating that pure LLM approaches struggle to ensure physical plausibility.
- SceneWeaver achieves 87–95% preference rates in human evaluations and also leads in diversity.

## Highlights & Insights
- Applying the LLM agent paradigm to 3D scene synthesis is a natural and effective design choice. The reason-act-reflect closed loop enables the system to learn from generation feedback, making it more flexible and robust than fixed pipelines.
- The standardized tool card design enables plug-and-play integration of new tools, greatly enhancing the framework's extensibility. As more scene generation methods emerge in the future, SceneWeaver can continuously benefit from them.
- Human evaluation results show that SceneWeaver achieves over 85% preference in pairwise comparisons against all baselines, validating the alignment between automatic metrics and human judgment.
- The ReAct-style reasoning chain implemented on the OpenManus platform makes the planning process of the entire system transparent and interpretable.
- Testing across 8 open-vocabulary room types demonstrates strong generalization; the average object count of 36.5 far exceeds baselines while maintaining zero physical violations.
- Setting memory length to 1, though simple, effectively mitigates LLM hallucination, reflecting pragmatic engineering judgment.

## Limitations & Future Work
- The reliance on GPT-4 for reflection and tool selection incurs high inference costs; generating a single scene requires multiple rounds of API calls, making speed and cost a bottleneck.
- Asset retrieval quality is limited by existing 3D model libraries (Objaverse/3D-Future); suitable meshes may not be available for certain rare object categories.
- Physics optimization is a rule-based simplified version, unable to handle complex physical interactions (e.g., hinges, springs).
- Scene scale is primarily limited to single rooms; multi-room or building-scale scene generation remains unexplored.
- The human evaluation involved only 20 participants, representing a relatively small evaluation scale.
- The quality of tool card descriptions directly affects planner decisions and requires careful design.
- Reflective evaluation relies on top-down rendered views rather than 3D perceptual assessment, which may be insufficient for scenes with complex vertical structures.

## Related Work & Insights
- **vs. Holodeck**: Holodeck produces many objects but suffers severe collisions (#CN=38.5 vs. 0.0), highlighting the limitations of lacking physics optimization and a reflection mechanism.
- **vs. I-Design**: I-Design achieves good physical metrics (zero collisions) but generates fewer objects with lower completeness; SceneWeaver maintains zero collisions while generating 3–4× more objects.
- **vs. LayoutGPT**: Pure LLM generation yields poor physical plausibility (#OB=1.0, #CN=1.3), confirming that LLMs alone are insufficient to ensure correct spatial reasoning.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The first work to apply a reflective LLM agent framework to 3D scene synthesis; the standardized tool interface design is forward-looking.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated on both common room types and open-vocabulary settings, covering 8 room types with multi-dimensional metrics, ablations, and comprehensive human evaluation.
- Writing Quality: ⭐⭐⭐⭐ The framework description is clear, though the method section is somewhat lengthy and could be streamlined.
- Value: ⭐⭐⭐⭐⭐ Directly valuable for embodied AI scene construction; the unified framework concept has potential to generalize to broader 3D generation tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] From None to All: Self-Supervised 3D Reconstruction via Novel View Synthesis](../../CVPR2026/3d_vision/from_none_to_all_self-supervised_3d_reconstruction_via_novel_view_synthesis.md)
- [\[ICCV 2025\] RoboTron-Mani: All-in-One Multimodal Large Model for Robotic Manipulation](../../ICCV2025/3d_vision/robotron-mani_all-in-one_multimodal_large_model_for_robotic_manipulation.md)
- [\[NeurIPS 2025\] MaterialRefGS: Reflective Gaussian Splatting with Multi-view Consistent Material Inference](materialrefgs_reflective_gaussian_splatting_with_multi-view_consistent_material_.md)
- [\[ICCV 2025\] No Pose at All: Self-Supervised Pose-Free 3D Gaussian Splatting from Sparse Views](../../ICCV2025/3d_vision/no_pose_at_all_self-supervised_pose-free_3d_gaussian_splatting_from_sparse_views.md)
- [\[ICCV 2025\] Self-Ensembling Gaussian Splatting for Few-Shot Novel View Synthesis](../../ICCV2025/3d_vision/self-ensembling_gaussian_splatting_for_few-shot_novel_view_synthesis.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] ReVSI: Rebuilding Visual Spatial Intelligence Evaluation for Accurate Assessment of VLM 3D Reasoning
description: >-
  [ICML 2026][Multimodal VLM][VSI-Bench] This paper systematically reveals that the widely used VSI-Bench suffers from structural failures due to 3D annotation drift and frame sampling inconsistency. By re-annotating 381 s…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "VSI-Bench"
  - "Spatial Reasoning"
  - "Frame Budget"
  - "Virtual Video"
  - "hallucination"
date: 2026-05-08
content_hash: 1dcabcef277e40f0
---

# ReVSI: Rebuilding Visual Spatial Intelligence Evaluation for Accurate Assessment of VLM 3D Reasoning

**Conference**: ICML 2026  
**arXiv**: [2604.24300](https://arxiv.org/abs/2604.24300)  
**Code**: Available (Project Page + GitHub + HuggingFace)  
**Area**: Multimodal VLM / Evaluation Benchmark / Visual Spatial Intelligence  
**Keywords**: VSI-Bench, Spatial Reasoning, Frame Budget, Virtual Video, hallucination

## TL;DR
This paper systematically reveals that the widely used VSI-Bench suffers from structural failures due to 3D annotation drift and frame sampling inconsistency. By re-annotating 381 scenes and 5,365 objects and designing frame-budget adaptive QA and "queried-object-removed" dummy video stress tests, the authors construct a high-fidelity spatial intelligence benchmark named ReVSI. Evaluations show that open-source VLMs experience performance drops of up to 40% on ReVSI and maintain high hallucination rates on dummy videos, exposing a systemic overestimation of existing spatial reasoning capabilities by VSI-Bench.

## Background & Motivation

**Background**: As VLMs expand toward embodied and 3D perception, VSI evaluation benchmarks such as VSI-Bench, SPAR-Bench, and VSI-SUPER have become mainstream. These use 3D datasets like ScanNet/ARKitScenes to automatically generate QA to test model spatial reasoning on tasks such as object counting, relative direction, and room area. VLM training (SpatialVLM, Cambrian-S, SpaceR) is also optimized around these benchmarks.

**Limitations of Prior Work**: The authors use manual audits to reveal two core defects. First, **Annotation-Video Drift**: VSI-Bench ground truth (GT) is derived from point-cloud-based 3D reconstruction annotations (designed for traditional 3D perception). However, objects clearly visible in raw videos may be omitted due to incomplete reconstruction, and object categories are often mislabeled (e.g., a cup labeled as a notebook). Room areas are calculated based on noisy Alpha Shapes, resulting in many QA pairs that are incorrect or semantically ambiguous under video evidence—out of 565 Object Counting samples, 27% were wrong and 11% were ambiguous. Second, **Unobservable Frame Sampling**: In practice, VLMs can only process 16/32/64 frames, yet VSI-Bench GT is labeled based on all frames. Figure 3 shows that under a 16-frame constraint, GT correctness drops to 67%, meaning a significant portion of questions are unanswerable given the actual model input.

**Key Challenge**: Benchmarks default to the assumption that "what the model sees matches what was seen during annotation." However, the sparse-frame input of modern VLMs breaks this assumption, making it impossible to distinguish whether an error is due to weak spatial reasoning or missing evidence. Furthermore, skewed answer distributions in VSI-Bench (e.g., "2" accounting for 53% of Object Counting and distances being mostly 0–2m) allow models to achieve high scores via priors rather than visual evidence.

**Goal**: While retaining the VSI-Bench task paradigm, the goal is to ensure that (i) annotations are strictly consistent with the original video; (ii) QA is answerable and correct under every frame budget; and (iii) controllable diagnostic tools are provided to decouple "visual evidence" from "reasoning capability."

**Key Insight**: Instead of training another model, it is more effective to fix the evaluation—strictly aligning "what the benchmark asks" with "what the model actually sees" to give the benchmark diagnostic value.

**Core Idea**: By implementing video-aligned manual 3D re-annotation, frame-budget adaptive QA, and dummy video stress testing, the authors rebuild the first input-consistent VSI benchmark, ReVSI.

## Method

### Overall Architecture
The ReVSI pipeline consists of three stages: (1) Using a self-developed 3D web annotation interface, the original VSI-Bench is expanded from 288 scenes and 65 categories to 381 scenes and 504 categories (open-set) across ScanNetv2/ScanNet++/ARKitScenes/3RScan/MultiScan, with 5,365 3D bounding boxes redrawn. (2) QA is regenerated for 6 tasks (object counting, object size, absolute distance, room size, relative distance, and relative direction; Object Appearance Order is removed as it focuses more on temporal reasoning) using stricter template rules, with each pair manually verified. (3) GT is constructed for four frame budgets (16/32/64/all-frame) for the same video, and "dummy videos" (deleting all frames containing queried objects) are generated for visibility-guided control experiments.

### Key Designs

1.  **Video-Aligned Open-Vocabulary 3D Re-annotation**:
    - **Function**: Replaces 3D GT based on "noisy reconstruction meshes" with "manual high-fidelity annotations anchored to original video," increasing object scale from 3,185 to 5,365 and categories from 65 to 504.
    - **Mechanism**: A web annotator was developed where the author (a 3D domain expert) started with original VSI-Bench annotations, filtered mislabels, tightened 3D boxes, recovered objects visible in video but missing in reconstruction, and extrapolated true physical sizes for geometrically damaged objects using adjacent frames. Open-set labels (e.g., "Sony PlayStation", "Coca-Cola box") were manually written, with GPT-5.2 used only for verification. Room areas moved from Alpha Shape to human-drawn polygons from a top-down view to exclude ambiguous boundaries.
    - **Design Motivation**: The root issue of old GT was that the annotation target was the mesh rather than the video. Changing the annotation target solves downstream issues; open-vocabulary and finer categories prevent models from relying on narrow 65-category priors.

2.  **Debiased and Manually Verified QA Regeneration**:
    - **Function**: Rewrites templates while retaining task definitions to eliminate distribution biases in VSI-Bench (where guessing "2" yielded 62% accuracy).
    - **Mechanism**: For Object Counting, single-instance queries (e.g., "How many black office chairs") and "two-category sum" templates are introduced, and "this room" is changed to "the scene" to match multi-room videos. For Object Size, fixed-size categories like toilets/beds are removed, and OOD sampling is used for items like refrigerators. For Absolute Distance, <1m questions (answerable via 2D cues) are replaced with long-range pairs. For Relative Direction, positioning objects must have a footprint $\le 1\text{ m}^2$ and spacing $\ge 1\text{m}$, with "facing towards/away" templates added. Every question underwent manual verification.
    - **Design Motivation**: Original benchmarks behaved like multiple-choice questions with concentrated answers, allowing models to collapse to high-frequency modes; statistical debiasing and template diversification block this shortcut, ensuring metrics reflect spatial reasoning.

3.  **Frame-Budget Adaptive Evaluation and Dummy Video Control Experiments**:
    - **Function**: Constructs GT for 16/32/64/all frame budgets and creates dummy videos (deleting queried frames) to test whether models truly depend on visual evidence.
    - **Mechanism**: Each sampled frame is rasterized using GT camera poses to determine object visibility (occupying >5% area); invisible cases are manually labeled. Room Size and Route Planning are excluded from the 16-frame setting due to information scarcity. Dummy videos retain scene context but remove target object frames; they are "unanswerable" for humans, but GT is deterministic (e.g., counting must be 0). Metrics: Acc for MCQ; Mean Relative Accuracy $\text{MRA}=\frac{1}{|C|}\sum_{\theta\in C}\mathbb{1}[|\hat y-y|/y<1-\theta]$ with $C=\{0.5,0.55,\dots,0.95\}$ for NQ.
    - **Design Motivation**: Aligning "model input" with the "benchmark evaluation target" is essential for reliable assessment. Dummy videos expose the implicit assumption of whether an object was actually seen before answering—if a model answers correctly without evidence, the output is driven by prior rather than vision, which is the definition of hallucination.

### Loss & Training
ReVSI is an evaluation benchmark and does not involve model training; assessments follow MRA (NQ) and Acc (MCQ).

## Key Experimental Results

### Main Results
Evaluation includes general VLMs (Qwen3-VL, InternVL-3.5, LLaVA-Video, GPT-5.2, Gemini 3) and 3D expert models (SpatialVLM, Cambrian-S, SpaceR, VLM-3R, Spatial-MLLM) on both ReVSI and VSI-Bench.

| Dataset Statistics | VSI-Bench | ReVSI |
| :--- | :--- | :--- |
| Scenes | 288 | 381 |
| Objects | 3185 | 5365 |
| Categories | 65 | 504 |
| Open-Vocabulary | ✗ | ✓ |
| Frame-Adaptive GT | ✗ (All-frame only) | ✓ (16/32/64/all) |

| Model Type | VSI-Bench Perf | ReVSI Perf | Conclusion |
| :--- | :--- | :--- | :--- |
| Closed-source (GPT-5.2, Gemini 3) | Lower than open-source | Significantly outperforms open-source | VSI-Bench systematically underestimates closed-source models |
| Open-source (Qwen3-VL, InternVL-3.5) | High | Dropped by up to 40% | VSI-Bench overestimates open-source VLMs |
| 3D Finetuned Experts (SpaceR, 3D-R1) | Much higher than base | Gains drop significantly | Finetuning gains were amplified by benchmark bias |

### Ablation Study

| Diagnostic Setting | Key Findings |
| :--- | :--- |
| Object Counting (Predicting "2") | 62% on VSI-Bench, <20% on ReVSI; verifies successful debiasing. |
| Absolute Distance | Most models scored higher on ReVSI → MRA is more tolerant for long distances; removing <1m samples highlighted Qwen3-VL's long-range strength. |
| Dummy Video Object Counting | InternVL-3.5 still outputs mid-range numbers → Non-zero hallucination rate proves indoor priors drive output. |
| Object Size (All-black frames) | Some expert models still hit "typical category sizes," revealing size estimation relies heavily on category priors. |
| Frame Budget Scan | GT correctness rose from 67% → 92% (16 → 64 frames), proving the necessity of frame-aware design. |

### Key Findings
- The "Open Source > Closed Source" conclusion on VSI-Bench is reversed on ReVSI, suggesting previous "Expert SOTA" conclusions were likely benchmark artifacts.
- 3D finetuned experts show diminishing returns on clean ReVSI data; post-training data scale is decoupled from performance, indicating current 3D instruction tuning primarily overfits noisy GT.
- Dummy videos expose that multiple SOTA open-source VLMs are almost insensitive to whether visual evidence exists—this is a true bottleneck in spatial reasoning.
- Frame sampling empirical threshold: Single-room scenes require at least 64 frames, and benchmarks should provide localized GT based on frame budgets.

## Highlights & Insights
- **Empirical example of "Evaluation maintenance being more important than model modification"**: The author identified 27% error + 11% ambiguity in VSI-Bench and overturned most SOTA assertions, showing evaluation hygiene is a high-ROI direction for spatial AI research.
- **Dummy video protocol**: This can be extended to any video QA without cost by "automatically removing evidence frames to see if the model still answers." This visibility-controlled stress test systematically quantifies hallucination and has high transfer value.
- **Frame-budget-aware GT**: For the first time, "GT is not a constant, but a function $\text{GT}(\text{frames})$" was implemented in a large-scale benchmark; future long-video benchmarks should adopt this.

## Limitations & Future Work
- While large-scale, re-annotation remains manual and difficult to scale to in-the-wild videos; future work could use GPT-5.2 for semi-automation with manual checks.
- Object Appearance Order was removed to avoid temporal reasoning, but joint spatial-temporal understanding is still not covered.
- Dummy videos treat "no evidence → answer 0/unknown" as GT; while intuitive, this does not fully align with "refuse to answer" behaviors in certain models, and confidence calibration could be added.
- ReVSI shares task definitions with VSI-Bench, meaning it has not expanded into completely new 3D reasoning tasks like multi-view registration or 6DoF manipulation.

## Related Work & Insights
- **vs VSI-Bench (Yang 2025a)**: Directly audited by this work; ReVSI serves as a more trustworthy successor correcting its flaws.
- **vs SPAR-Bench / VSI-SUPER**: These also suffer from GT drift and frame mismatch; the "input-consistent" principle of ReVSI can be applied to them.
- **vs 3D Finetuned VLMs (SpatialVLM, Cambrian-S, SpaceR)**: ReVSI reveals their gains diminish on clean benchmarks, urging researchers to prioritize evaluation rigor.
- **Cross-task insights**: The dummy video / visibility-controlled QA protocol can be generalized to medical VQA (removing critical anatomical regions) or robot perception (deleting key frames).

## Rating
- Novelty: ⭐⭐⭐⭐ Reconstruction and new protocols for benchmarks; significant impact even without entirely new tasks.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 10+ models across open/closed/expert categories with multi-frame budget and dummy video diagnostics; sufficient audit data.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logical chain of problem diagnosis → solution → empirical validation; Figures 1, 3, and 5 effectively summarize core issues.
- Value: ⭐⭐⭐⭐⭐ directly challenges the credibility of a widely cited benchmark and may shift the direction of VLM spatial reasoning research.

## Related Papers

- [\[CVPR 2026\] SpatialScore: Towards Comprehensive Evaluation for Spatial Intelligence](../../CVPR2026/multimodal_vlm/spatialscore_towards_comprehensive_evaluation_for_spatial_intelligence.md)
- [\[ICML 2026\] R$^3$L: Reasoning 3D Layouts from Relative Spatial Relations](r3l_reasoning_3d_layouts_from_relative_spatial_relations.md)
- [\[ICML 2026\] Learning GUI Grounding with Spatial Reasoning from Visual Feedback](learning_gui_grounding_with_spatial_reasoning_from_visual_feedback.md)
- [\[ICML 2026\] Thinking in Structures: Evaluating Spatial Intelligence in Constraint-Governed Spaces](thinking_in_structures_evaluating_spatial_intelligence_in_constraint-governed_sp.md)
- [\[CVPR 2026\] SpatialStack: Layered Geometry-Language Fusion for 3D VLM Spatial Reasoning](../../CVPR2026/multimodal_vlm/spatialstack_layered_geometry-language_fusion_for_3d_vlm_spatial_reasoning.md)

</div>

<!-- RELATED:END -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SpatialScore: Towards Comprehensive Evaluation for Spatial Intelligence](../../CVPR2026/multimodal_vlm/spatialscore_towards_comprehensive_evaluation_for_spatial_intelligence.md)
- [\[ICML 2026\] R$^3$L: Reasoning 3D Layouts from Relative Spatial Relations](r3l_reasoning_3d_layouts_from_relative_spatial_relations.md)
- [\[ICML 2026\] Learning GUI Grounding with Spatial Reasoning from Visual Feedback](learning_gui_grounding_with_spatial_reasoning_from_visual_feedback.md)
- [\[ICML 2026\] Thinking in Structures: Evaluating Spatial Intelligence in Constraint-Governed Spaces](thinking_in_structures_evaluating_spatial_intelligence_in_constraint-governed_sp.md)
- [\[CVPR 2026\] SpatialStack: Layered Geometry-Language Fusion for 3D VLM Spatial Reasoning](../../CVPR2026/multimodal_vlm/spatialstack_layered_geometry-language_fusion_for_3d_vlm_spatial_reasoning.md)

</div>

<!-- RELATED:END -->

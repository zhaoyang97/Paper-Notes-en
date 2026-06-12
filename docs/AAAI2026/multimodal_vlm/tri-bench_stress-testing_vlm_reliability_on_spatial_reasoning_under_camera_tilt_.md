---
title: >-
  [Paper Note] Tri-Bench: Stress-Testing VLM Reliability on Spatial Reasoning under Camera Tilt and Object Interference
description: >-
  [AAAI 2026][Multimodal VLM][Spatial reasoning benchmark] Tri-Bench is a compact benchmark comprising 400 real-world photographs of triangles. By systematically controlling two factors — camera pose (planar vs. tilted) an…
tags:
  - "AAAI 2026"
  - "Multimodal VLM"
  - "Spatial reasoning benchmark"
  - "geometric reasoning"
  - "camera pose robustness"
  - "majority-class bias"
  - "VLM reliability"
date: 2026-05-08
content_hash: 95cb1a5581c0e386
---

# Tri-Bench: Stress-Testing VLM Reliability on Spatial Reasoning under Camera Tilt and Object Interference

**Conference**: AAAI 2026
**arXiv**: [2512.08860](https://arxiv.org/abs/2512.08860)  
**Code**: [https://github.com/Amiton7/Tri-Bench](https://github.com/Amiton7/Tri-Bench)  
**Area**: Multimodal VLM
**Keywords**: Spatial reasoning benchmark, geometric reasoning, camera pose robustness, majority-class bias, VLM reliability

## TL;DR
Tri-Bench is a compact benchmark comprising 400 real-world photographs of triangles. By systematically controlling two factors — camera pose (planar vs. tilted) and object interference — it evaluates the spatial geometric reasoning capabilities of four leading VLMs. The results reveal that models default to 2D image-plane cues rather than genuine 3D geometry, even when explicit reference-frame guardrails are provided in the prompt, with accuracy on minority-class shapes dropping to near 0%.

## Background & Motivation

### State of the Field
VLMs have become indispensable components in real-world applications, particularly in robotic navigation, AR/VR measurement tools, 3D reconstruction, and AI-assisted geometry education. **Verifiable geometric reasoning is a critical component of trustworthy and controllable agentic AI.** Existing spatial reasoning benchmarks either focus on absolute distance/angle estimation or on abstract diagrams and scene-level problem solving, and lack systematic stress-testing of **deployment-critical factors** such as camera-pose invariance and object interference.

### Limitations of Prior Work

**Insufficient granularity in existing benchmarks**: Benchmarks such as Mind the Gap and OmniSpatial cover broad cognitive abilities (mental rotation, spatial navigation) but do not isolate the robustness of fundamental geometric measurement; MathBench and VisioMath evaluate mathematical problems but use clean symbolic inputs rather than real photographs.

**No evaluation of camera pose variation**: Camera angles vary widely in real deployments, yet existing benchmarks rarely assess the impact of this factor on reasoning.

**No evaluation of object interference**: Real-world scenes always contain co-existing objects, and whether they affect VLM geometric reasoning remains unknown.

**Conflation of 2D and 3D reasoning**: VLMs observe 2D projected images but should reason about 3D real-world geometry; existing benchmarks do not distinguish between the two.

### Starting Point
The paper designs a **minimal, controlled diagnostic benchmark**: using the most fundamental closed geometric structure (triangles), it systematically isolates the effects of camera pose and object interference on VLM spatial reasoning under controlled conditions. The key innovation lies in providing **explicit reference-frame guardrails** (a known square border), which enable recovery of 3D geometry via homography transformation — thereby testing whether VLMs can exploit this information.

## Method

### Overall Architecture
Tri-Bench contains 100 annotated triangles, each photographed under 4 conditions (2×2: planar/tilted × with/without objects), yielding 400 images in total. In each image, the triangle is placed within a 1 m × 1 m square border, with vertices A/B/C marked using red/yellow/blue stickers. VLM capability is evaluated across six geometric reasoning tasks.

### Key Designs

#### 1. **Dataset Composition**
- **Function**: Manually construct 100 diverse triangles and photograph them under controlled conditions.
- **Core details**:
    - **Shape diversity**: 38 acute, 32 obtuse, and 30 right triangles; 64 scalene, 26 isosceles, and 10 equilateral triangles.
    - **95 unique shapes + 5 duplicates** (used to verify label and position perturbation).
    - **Four capture conditions**:
        - P0: planar capture, no objects
        - P1: planar capture, with objects
        - T0: tilted capture, no objects
        - T1: tilted capture, with objects
    - **10 everyday objects**: Rubik's cube, glass vase, electric iron, book, kettle, apple, cosmetics case, plastic stool, 15.6-inch laptop, medium-sized pillow — each paired with 10 triangles.
    - **Occlusion annotation**: 25 images contain partially occluded triangle edges; 1 image contains a fully occluded vertex.
    - **Annotation rules**: isosceles (minimum pairwise relative side-length ratio ≤ 3%), equilateral (all pairwise relative side-length ratios ≤ 3%), right-angled (an angle within 2° of 90°).
- **Design Motivation**: Triangles are the most fundamental closed geometric structures, and reasoning about them simultaneously requires distance comparison and angle reasoning — the core abilities underlying spatial reasoning.

#### 2. **2D–3D Shape Label Mismatch Analysis**
- **Function**: Quantify the category changes that occur when 3D ground-truth shapes are projected to 2D.
- **Core Findings**:
    - Approximately 27% of triangles change side-type after projection; 34% change angle-type.
    - 62.5% of equilateral triangles become non-equilateral after projection; approximately 70% of right triangles become non-right-angled.
    - Only approximately 7% of scalene triangles undergo a category change.
    - Evaluating against 2D projected answers vs. 3D ground-truth answers can therefore reveal whether a VLM is performing genuine 3D reasoning.
- **Design Motivation**: If a VLM's responses are more consistent with 2D projections than with 3D ground truth, the model is "reading the image" rather than "understanding 3D geometry."

#### 3. **Six Evaluation Tasks**
- **Classification tasks**:
    - Q1: Is triangle ABC equilateral, isosceles, or scalene?
    - Q2: Is triangle ABC acute, right-angled, or obtuse?
- **Continuous estimation tasks**:
    - Q3: Estimate the ratio $AB/AC$.
    - Q4: Estimate $|\angle ABC - \angle ACB|$ (angular difference).
    - Q5: Estimate $\frac{\max\{AB,BC,CA\}}{\min\{AB,BC,CA\}}$ (longest-to-shortest side ratio).
    - Q6: Estimate $\max\{|\angle A - \angle B|, |\angle B - \angle C|, |\angle C - \angle A|\}$ (maximum angular difference).
- **Design Motivation**:
    - Q1–Q2 test precise classification (requiring fine-grained comparison).
    - Q3–Q6 test relative reasoning (without providing absolute distances, angles, or coordinate systems, forcing VLMs to employ both qualitative and quantitative reasoning).
    - Relative comparison better reflects deep spatial reasoning properties than absolute estimation.

#### 4. **Prompt Design and Guardrails**
- **Function**: Use a unified zero-shot prompt that includes an explicit description of the reference frame.
- **Key design**: The prompt explicitly describes the "light-brown tape square border," which is coplanar with the triangle, allowing the VLM to recover true 3D geometry via homography transformation.
- **Output format**: Strict JSON with six predefined keys; numerical values required to four decimal places.
- **All models use exactly the same prompt**, ensuring fair comparison.
- **Design Motivation**: The guardrail prompt is central to testing "verifiability and controllability" — if a VLM can follow the prompt and exploit the reference frame, it should be able to provide correct 3D answers.

### Evaluation Metrics
Accuracy is uniformly defined as $\kappa_t = 1 - \varepsilon_t$:
- Classification tasks (Q1/Q2): exact match; error $= 1 - \mathbf{1}\{\hat{y}=y\}$.
- Relative ratios (Q3/Q5): relative error; $\min(1, |\hat{y}-y|/y)$.
- Normalized angles (Q4/Q6): normalized absolute error; $\min(1, |\hat{y}-y|/180°)$.

## Key Experimental Results

### Main Results — 3D vs. 2D Accuracy

| Model | 3D Accuracy (%) | 2D Accuracy (%) | Difference |
|-------|----------------|----------------|------------|
| Gemini 2.5 Pro | 75.30 | 80.89 | +5.59 |
| Gemini 2.5 Flash | 71.58 | 77.14 | +5.56 |
| GPT-5 | 64.32 | 65.04 | +0.72 |
| Qwen2.5-VL-32B | 64.70 | 66.22 | +1.52 |
| **Average** | **68.98** | **72.32** | **+3.34** |

### Majority-Class Bias in Precise Classification Tasks

| Model | Scalene (Q1) | Isosceles (Q1) | Equilateral (Q1) | Acute (Q2) | Obtuse (Q2) | Right (Q2) |
|-------|-------------|---------------|-----------------|-----------|------------|-----------|
| Gemini 2.5 Pro | 99.61 | 2.88 | 0.00 | 78.29 | 88.28 | 0.00 |
| Gemini 2.5 Flash | 98.83 | 1.92 | 0.00 | 72.37 | 80.47 | 5.83 |
| GPT-5 | 99.61 | 0.96 | 0.00 | 92.11 | 3.91 | 1.67 |
| Qwen2.5-VL-32B | 100.00 | 0.00 | 0.00 | 100.00 | 0.00 | 0.00 |
| **Average** | **99.51** | **1.44** | **0.00** | **85.69** | **43.16** | **1.88** |

### Effect of Camera Tilt and Object Interference

| Condition | Average Accuracy (%) | Note |
|-----------|---------------------|------|
| Planar view (P0/P1) | 71.0 | Baseline condition |
| Tilted view (T0/T1) | 66.9 | Drop of ~4.1% |
| No objects (P0/T0) | 69.2 | Baseline condition |
| With objects (P1/T1) | 68.8 | Drop of only 0.4%, negligible |

### Key Findings
1. **VLMs default to 2D reasoning rather than 3D reasoning**: All models achieve higher accuracy against 2D projected answers than 3D ground truth (average +3.34%), with the Gemini series showing the largest gap (+5.5%), demonstrating that models disregard the reference-frame information in the prompt.
2. **Equilateral triangle accuracy is 0.00%**: All four models completely fail to classify equilateral triangles. Isosceles accuracy is only 1.44% and right-angled only 1.88%. Models exhibit severe **majority-class bias** — since scalene triangles constitute 64% of the dataset, models default to predicting scalene.
3. **Qwen2.5-VL-32B exhibits completely degenerate behavior**: All Q1 predictions are scalene (100%) and all Q2 predictions are acute (100%), with zero accuracy on all non-majority classes.
4. **Camera tilt systematically degrades performance**: A drop of approximately 4% is observed, with Q2 and Q5 most affected (6–7%), indicating that VLMs lack pose invariance.
5. **Object interference has almost no effect**: The accuracy difference between conditions with and without objects is less than 1%, suggesting that VLMs are relatively robust to this form of scene clutter.
6. **Relative comparison is easier than absolute classification**: Q4/Q6 (angular difference estimation) are substantially easier than Q2 (angle-type classification).

## Highlights & Insights
1. **"Minimal diagnostic probe" design philosophy**: Rather than pursuing large scale or broad coverage, the paper maximizes diagnostic information through the most fundamental geometric structure (triangles). A compact 400-image dataset suffices to reveal profound failure modes.
2. **Clever guardrail prompt design**: The square border serves simultaneously as a visual reference and a mathematical tool (via homography); if a VLM can exploit it, it should obtain correct 3D answers — constituting a test of "verifiable reasoning."
3. **The 2D vs. 3D distinction is fundamentally important**: By computing accuracy against both ground-truth variants, the paper directly demonstrates that VLMs perform "surface-level reasoning" rather than "deep 3D understanding."
4. **The majority-class bias finding carries a critical warning for VLM deployment**: If VLMs cannot surpass majority-class prediction on simple triangle classification, they are even less trustworthy in safety-critical geometric reasoning tasks.
5. **"If an agent cannot perform 3D reasoning on simple triangles, it should not be deployed in safety-critical autonomous navigation tasks"** — this argument holds important cautionary value for the trustworthy AI community.

## Limitations & Future Work
1. **All triangles are coplanar with the border**: In real 3D scenes, geometric objects may not be coplanar with the reference plane.
2. **Single-image evaluation**: Multi-view geometric reasoning is a natural extension.
3. **Tilt treated as a binary factor**: The precise relationship between varying tilt angles and accuracy degradation is not systematically measured.
4. **Single fixed prompt**: More advanced prompting strategies (chain-of-thought reasoning, multi-step guidance) may improve results.
5. **Majority-class bias may originate from training data distribution**: Scalene and acute triangles are more common in the real world, but the authors do not analyze the training distribution.
6. **Limited to triangles**: Extension to more complex polygons, curves, and surfaces is needed.
7. **Insufficient analysis of open-source models**: Qwen is the only open-source model included, and it is accessed via the fireworks.ai API.

## Related Work & Insights
- **DynaMath**: Tests robustness by introducing dynamic variants of mathematical problems; Tri-Bench similarly controls capture conditions to test robustness.
- **SpatialVLM**: Focuses on improving VLM spatial reasoning capability, whereas Tri-Bench focuses on **evaluating** the reliability of such capability.
- **NeSyGeo / AutoGPS**: Neural-symbolic frameworks for geometric reasoning using clean symbolic inputs; Tri-Bench complements these by addressing reasoning from noisy real-world photographs.
- Insight: **"Capability" and "reliability" of VLMs are entirely different dimensions** — high overall scores do not imply trustworthiness. Even when overall accuracy appears acceptable (~70%), per-category analysis exposes catastrophic failures.

## Rating
- Novelty: ⭐⭐⭐⭐ (Unique problem framing — evaluating geometric reasoning from a trustworthy AI perspective; the 2D/3D distinction and guardrail design are creative)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Comprehensive cross-analysis across four models, six tasks, and four conditions, though the dataset scale is relatively small)
- Writing Quality: ⭐⭐⭐⭐⭐ (Argumentation chain is clear, with each step from problem to design to findings tightly connected, and conclusions are well-supported)
- Value: ⭐⭐⭐⭐ (High cautionary value for trustworthy VLM deployment; reveals fundamental overlooked failures, though the practical remediation path is not sufficiently articulated)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SpatialStack: Layered Geometry-Language Fusion for 3D VLM Spatial Reasoning](../../CVPR2026/multimodal_vlm/spatialstack_layered_geometry-language_fusion_for_3d_vlm_spatial_reasoning.md)
- [\[ICCV 2025\] CAPTURe: Evaluating Spatial Reasoning in Vision Language Models via Occluded Object Counting](../../ICCV2025/multimodal_vlm/capture_evaluating_spatial_reasoning_in_vision_language_models_via_occluded_obje.md)
- [\[ICML 2026\] ReVSI: Rebuilding Visual Spatial Intelligence Evaluation for Accurate Assessment of VLM 3D Reasoning](../../ICML2026/multimodal_vlm/revsi_rebuilding_visual_spatial_intelligence_evaluation_for_accurate_assessment_.md)
- [\[CVPR 2026\] VLM-Pruner: Buffering for Spatial Sparsity in an Efficient VLM Centrifugal Token Pruning Paradigm](../../CVPR2026/multimodal_vlm/vlm-pruner_buffering_for_spatial_sparsity_in_an_efficient_vlm_centrifugal_token_.md)
- [\[AAAI 2026\] Rethinking Visual Token Reduction in LVLMs under Cross-Modal Misalignment](rethinking_visual_token_reduction_in_lvlms_under_cross-modal_misalignment.md)

</div>

<!-- RELATED:END -->

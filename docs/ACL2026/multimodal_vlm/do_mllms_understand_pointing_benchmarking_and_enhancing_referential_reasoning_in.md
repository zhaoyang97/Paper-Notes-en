---
title: >-
  [Paper Note] Do MLLMs Understand Pointing? Benchmarking and Enhancing Referential Reasoning in Egocentric Vision
description: >-
  [ACL 2026][Multimodal VLM][Egocentric Pointing] The authors construct EgoPoint-Bench (11.7k QA / 5 dimensions / 3 semantic levels), the first hybrid real-world and physically simulated egocentric "pointing" QA benchmark.…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "Egocentric Pointing"
  - "Referential Hallucination"
  - "Sim-to-Real"
  - "MLLM Benchmark"
  - "LoRA Fine-tuning"
date: 2026-05-08
content_hash: 59a7c089548231e7
---

# Do MLLMs Understand Pointing? Benchmarking and Enhancing Referential Reasoning in Egocentric Vision

**Conference**: ACL 2026  
**arXiv**: [2604.21461](https://arxiv.org/abs/2604.21461)  
**Code**: https://guyyyug.github.io/EgoPoint-Bench/ (Project Page)  
**Area**: Multimodal VLM / Egocentric Vision / Pointing Understanding  
**Keywords**: Egocentric Pointing, Referential Hallucination, Sim-to-Real, MLLM Benchmark, LoRA Fine-tuning

## TL;DR
The authors construct EgoPoint-Bench (11.7k QA / 5 dimensions / 3 semantic levels), the first hybrid real-world and physically simulated egocentric "pointing" QA benchmark. The study confirms that current SOTA MLLMs generally rely on "visual proximity/saliency" pseudo-correlations rather than actual fingertip ray parsing. By performing LoRA fine-tuning on simulated data, the models achieve an average improvement of up to +25 points and demonstrate robust sim-to-real generalization.

## Background & Motivation

**Background**: Wearable devices such as smart glasses have fostered egocentric agent scenarios where natural user interaction relies heavily on deictic pronouns ("that/this") combined with pointing gestures. Current MLLMs like GPT-5, Gemini 3, and Qwen3-VL already perform well on general image QA.

**Limitations of Prior Work**: Empirical tests by the authors reveal that when provided with egocentric images containing pointing gestures, models do not truly project rays along the index finger to find targets. Instead, they focus on the "object closest to the hand" or the "most salient object in the frame." This phenomenon is termed **Referential Hallucination**.

**Key Challenge**: There is a scarcity of high-quality "vision-language-spatial" aligned data. RefCOCO/Visual Genome are third-person; Ego4D/EPIC-KITCHENS lack pointing QA annotations; Ges3ViG uses synthetic avatars instead of real hands; and YouRefIt is not egocentric. Models have never encountered dense supervision for "fingertip geometry $\rightarrow$ target object," making it difficult to learn the relationship.

**Goal**: (1) Provide a benchmark to quantitatively evaluate MLLM egocentric pointing understanding. (2) Develop a scalable synthesis pipeline for generating geometrically precise data. (3) Verify if simulation data is sufficient for models to "learn pointing" and transfer to real-world scenarios.

**Key Insight**: Utilize Habitat-Sim + ray-casting to generate geometrically rigorous pointing samples in 3D scenes (ensuring rays reach targets without obstruction), supplemented by 1.2k real-world samples for zero-shot cross-domain testing.

**Core Idea**: Utilize scalable simulation with "physical ray-casting + diverse hand models" to create 10k+ geometrically unambiguous egocentric pointing QA samples, enabling small models fine-tuned via LoRA to outperform closed-source models such as GPT-5 and Gemini 3.

## Method

### Overall Architecture
EgoPoint-Bench consists of two data collection pipelines, a QA generation pipeline, and a five-dimensional evaluation system. On the simulation side, Point-Sim generates 10,567 samples using 42 hand models across 1,838 high-fidelity 3D scenes. On the real-world side, 8 volunteers wearing MLVision smart glasses collected 1,162 images in indoor/outdoor settings. All images pass through a "machine generation + human verification" QA pipeline, outputting samples with three question types (choice/judgment/open-ended) and three levels of referential language (L1 explicit action description / L2 visual localization / L3 implicit pronouns), split into training/validation/test sets based on five capability dimensions.

### Key Designs

1.  **Point-Sim Geometrically Precise Simulation**:
    - **Function**: Automatically generate "image + ground truth label" pairs in existing 3D scenes while ensuring the index finger ray hits the target.
    - **Mechanism**: Agent positions $P_{agent}$ are sampled on NavMesh with $r_{search} \leq 3.0\text{m}$ (minimum obstacle avoidance of 0.4m, with distances dynamically scaled by target volume). Camera rotation $R_{cam} \in SO(3)$ is constructed according to $\mathbf{f}=(P_{obj}-P_{agent})/\|P_{obj}-P_{agent}\|$. Rodrigues' formula is used to rotate the index finger from its rest direction $\mathbf{u}_{rest}$ to the target direction $\mathbf{u}_{target}$, with rotation angle $\theta=\arccos(\mathbf{u}_{rest}\cdot\mathbf{u}_{target})$ and rotation axis $\mathbf{k}=\mathbf{u}_{rest}\times\mathbf{u}_{target}/\|\mathbf{u}_{rest}\times\mathbf{u}_{target}\|$. Finally, a ray is cast from the fingertip toward $P_{obj}$; samples are discarded if obstructed.
    - **Design Motivation**: Downgrade "correct pointing" from a learning objective to a geometric constraint. As long as the ray reaches the target, the label is correct, fundamentally eliminating label noise. Simulation also provides free multimodal alignment data (RGB / Depth / Semantic / BBox / 2D projections) for grounding tasks.

2.  **Diversity Injection and Domain Randomization**:
    - **Function**: Ensure simulated images are diverse in perspective, optics, human characteristics, and poses to narrow the sim-to-real gap for smart glasses.
    - **Mechanism**: Camera FOV is uniformly sampled between $[100^\circ, 115^\circ]$ to simulate wide-angle glasses; eye height $h_{eye} \sim \mathcal{U}(1.45, 1.70)$ meters; random left/right hands; 3D arm models from ArtStation are parametrized with stretching and joint adjustments in Blender, resulting in 42 hand models (3 skin tones $\times$ 7 sleeves $\times$ left/right). Small perturbations are added to camera pitch/yaw to simulate unstable human pointing.
    - **Design Motivation**: A single hand model or fixed perspective might cause the model to overfit to "hand texture" rather than "pointing intent." Large-scale low-cost variations break these pseudo-correlations, forcing the model to rely on geometric direction.

3.  **Three-Level Reference $\times$ Five-Dimensional Evaluation Taxonomy**:
    - **Function**: Decompose the broad capability of "pointing understanding" into measurable sub-tasks to expose specific model weaknesses.
    - **Mechanism**: Three levels of reference from easy to difficult—L1: "This X I am pointing to" (with category); L2: "The one on the left I am pointing to" (with spatial words); L3: "How do I use this?" (pure deictic pronoun). Five dimensions of capability: BP Basic Perception (category/color/material); FS Function & State (edible/operable); SC Spatial Context (reachability/scene consistency); OCR (brands/slogans); AR Adversarial Robustness (counterfactual/null reference).
    - **Design Motivation**: High scores in BP but low scores in AR indicate the model does not truly understand pointing. Dimensional analysis helps locate where "referential hallucination" primarily occurs and informs training data ratios.

### Loss & Training
Evaluation is performed via zero-shot inference. Enhancement experiments involve fine-tuning open-source MLLMs using LoRA on the Point-Sim training set (~10k samples) with standard language modeling objectives. The real-world test set remains entirely zero-shot to examine sim-to-real generalization.

## Key Experimental Results

### Main Results
Comparison of 4 closed-source models and 5 open-source models (Direct vs. LoRA) on simulation and real-world test sets. Metrics represent accuracy (%) across capability dimensions.

| Model | Method | Sim Mean | Real Mean | Overall Avg | LoRA Gain |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Random | - | 31.14 | 28.94 | 30.24 | - |
| Human | - | 95.80 | 96.00 | 95.90 | - |
| Gemini 3 Pro | Direct | 56.44 | 72.00 | 62.29 | - |
| Gemini 3 Flash | Direct | 57.21 | 71.84 | 62.71 | - |
| GPT-5.2 Instant | Direct | 54.80 | 66.76 | 59.29 | - |
| GPT-5 mini | Direct | 57.66 | 60.57 | 58.75 | - |
| LLaVA-1.5-7B | Direct / LoRA | 48.82 / **73.18** | 47.19 / **54.54** | 48.21 / **66.17** | **+17.96** |
| LLaVA-NeXT-7B | Direct / LoRA | 48.17 / **80.93** | 46.44 / **59.64** | 47.52 / **72.93** | **+25.41** |
| GLM-4.6V-Flash | Direct / LoRA | 53.29 / 74.86 | 56.42 / 61.26 | 54.47 / 69.74 | +15.27 |
| InternVL3.5-2B | Direct / LoRA | 51.74 / 75.43 | 53.73 / 62.03 | 52.49 / 70.39 | +17.90 |
| InternVL3.5-8B | Direct | 52.62 | 57.09 | 54.30 | - |

The strongest closed-source model, Gemini 3 Pro, only achieved a 62.3% comprehensive score, nearly 34 percentage points below the human level (95.9%). Meanwhile, LLaVA-NeXT-7B with LoRA reached 72.93%, surpassing all closed-source models and proving that the issue is not "insufficient model size" but "lack of specific supervision in training data."

### Ablation Study
Horizontal ablation comparisons using different architectures, sizes, and data scales.

| Configuration | Overall Avg | Description |
| :--- | :--- | :--- |
| LLaVA-1.5-7B Direct | 48.21 | Old architecture without pointing training |
| LLaVA-NeXT-7B Direct| 47.52 | Upgraded vision encoder but no supervision; negligible gain |
| InternVL3.5-2B Direct| 52.49 | Small model but stronger general VLM |
| InternVL3.5-8B Direct| 54.30 | 8B vs 2B only gains 1.8 points $\rightarrow$ pure scaling yields diminishing returns |
| LLaVA-1.5-7B + LoRA | 66.17 | +17.96; supervision signal is effective |
| LLaVA-NeXT-7B + LoRA| **72.93** | +25.41; better vision backbone $\times$ better data $\rightarrow$ maximum gain |
| InternVL3.5-2B + LoRA| 70.39 | 2B model approaches GPT-5 performance |

### Key Findings
- **Referential hallucination is a universal phenomenon**: All Direct models scored between 30–60 in the AR (Adversarial Robustness) dimension, significantly lower than other dimensions. This indicates models "blindly guess" even when no reasonable target is pointed to, validating the diagnosis of pseudo-correlations.
- **Scaling cannot replace supervision**: InternVL3.5 gained only 1.8 points moving from 2B to 8B in the Direct setting, whereas LLaVA-NeXT (7B) outperformed the 8B model once fine-tuned with LoRA. This proves pointing capability depends on data rather than scale.
- **Significant Sim-to-Real effect**: After LoRA fine-tuning on purely simulated data, all models showed consistent score increases (+7 to +13) on the real-world test set, demonstrating that Point-Sim's geometric accuracy and domain randomization effectively bridge the sim-to-real gap.
- **Closed-source $\neq$ Stronger**: While Gemini 3 Pro performed well on the real set (72%), its 56% score on simulation suggests a weaker ability to discriminate strict geometric alignment, likely due to training data favoring "loose matching."

## Highlights & Insights
- **Converting Labelling to Geometric Constraints**: Using ray-casting and hit-testing ensures label accuracy, bypassing the traditional dilemma of expensive or insufficient labeling. This is an elegant paradigm for synthetic data.
- **42 Hand Models $\times$ Domain Randomization**: Actively injecting "irrelevant variables" (skin tone / sleeves / jitter) into visual signals forces the model to prioritize geometric direction over superficial features, applicable to any "pose-object" alignment task.
- **The "Referential Hallucination" Concept**: Distilling the vague "model error" into a quantifiable, comparable failure mode (tested via the AR dimension) provides a highly reusable diagnostic framework.
- **LoRA 7B > GPT-5 mini**: Achieving superior vertical capability in a small model compared to closed-source giants via lightweight fine-tuning reaffirms that the "high-quality task data + small model adaptation" paradigm is highly cost-effective for narrow capabilities.

## Limitations & Future Work
- **Small Real-world Set (1.2k)**: The sim-to-real evaluation scale is relatively small, and with only 8 volunteers, potential collection biases (habitual poses, left/right hand distribution) have not been fully explored.
- **Single-frame focus**: The benchmark targets image QA rather than video. In real smart glasses scenarios, pointing is a temporal process (extend-lock-retract); single-frame settings may underestimate real-world deployment challenges.
- **Static Scenes**: 3D scenes are derived from static indoor datasets, lacking dynamic crowds or moving targets found in real interactions.
- **Unused Grounding Potential**: The data includes BBoxes and 2D coordinates, but the paper primarily focuses on QA evaluation without fully utilizing the potential for grounding tasks (e.g., directly regressing the intersection of the fingertip ray in 3D space).

## Related Work & Insights
- **vs. YouRefIt**: YouRefIt utilizes third-person perspectives, real gestures, and grounding tasks. This work shifts to egocentric perspectives and geometrically precise simulation, representing a fundamental shift from "watching others point" to "watching one's own hand."
- **vs. Ges3ViG**: Ges3ViG uses synthetic avatars for 3D grounding. This work upgrades from avatars to dual-source (real + physical simulation) data and focuses on QA rather than pure coordinate localization, aligning closer to AR interaction needs.
- **vs. RefEgo**: RefEgo is for egocentric video grounding but relies on pure text reference. This work adds the gesture modality, covering the "deictic + gesture" combination essential for real interaction.
- **vs. EOC-Bench / ECBench**: These egocentric QA benchmarks rely on visual prompts like hand-drawn boxes. This work replaces them with natural gestures, matching the actual input conditions of "unaugmented" AR terminals.

## Rating
- Novelty: ⭐⭐⭐⭐ The first egocentric benchmark treating "pointing geometry" as a first-class citizen. "Referential Hallucination" is a compelling concept, though the physical simulation + domain randomization approach is established in robotics/embodied AI.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 4 closed-source and 5 open-source models, utilizing a Direct vs. LoRA and five-dimension $\times$ three-level split. However, fine-grained ablation (contribution of specific randomization dimensions) is relatively thin.
- Writing Quality: ⭐⭐⭐⭐ Clear flow from motivation to failure mode naming, data, evaluation, and fine-tuning. Figure 1 addresses the core problem well; mathematical derivations are standard.
- Value: ⭐⭐⭐⭐⭐ Directly addresses the core interaction bottleneck of smart glasses/AR assistants and proves that "small models + high-quality data" can surpass closed-source models, making it highly applicable for industrial deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] How Do LLMs and VLMs Understand Viewpoint Rotation Without Vision? An Interpretability Study](how_do_llms_and_vlms_understand_viewpoint_rotation_without_vision_an_interpretab.md)
- [\[NeurIPS 2025\] MMPerspective: Do MLLMs Understand Perspective? A Comprehensive Benchmark for Perspective Perception, Reasoning, and Robustness](../../NeurIPS2025/multimodal_vlm/mmperspective_do_mllms_understand_perspective_a_comprehensive_benchmark_for_pers.md)
- [\[ACL 2026\] MathFlow: Enhancing the Perceptual Flow of MLLMs for Visual Mathematical Problems](mathflow_enhancing_the_perceptual_flow_of_mllms_for_visual_mathematical_problems.md)
- [\[ACL 2026\] OMIBench: Benchmarking Olympiad-Level Multi-Image Reasoning in Large Vision-Language Models](omibench_benchmarking_olympiad-level_multi-image_reasoning_in_large_vision-langu.md)
- [\[ACL 2026\] What Do Vision-Language Models Encode for Personalized Image Aesthetics Assessment?](what_do_vision-language_models_encode_for_personalized_image_aesthetics_assessme.md)

</div>

<!-- RELATED:END -->

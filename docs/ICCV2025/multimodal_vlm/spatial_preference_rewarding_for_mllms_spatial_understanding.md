---
title: >-
  [Paper Note] Spatial Preference Rewarding for MLLMs Spatial Understanding
description: >-
  [ICCV 2025][Multimodal VLM][MLLM] This paper proposes SPR (Spatial Preference Rewarding), a framework that automatically constructs preference data pairs via semantic and localization scores…
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "MLLM"
  - "spatial understanding"
  - "preference optimization"
  - "DPO"
  - "object grounding"
date: 2026-05-08
content_hash: 06589183f0f669b0
---

# Spatial Preference Rewarding for MLLMs Spatial Understanding

**Conference**: ICCV 2025
**arXiv**: [2510.14374](https://arxiv.org/abs/2510.14374)  
**Code**: [SPR](https://github.com/hanqiu-hq/SPR)  
**Area**: Multimodal Large Language Models / Spatial Understanding
**Keywords**: MLLM, spatial understanding, preference optimization, DPO, object grounding

## TL;DR

This paper proposes SPR (Spatial Preference Rewarding), a framework that automatically constructs preference data pairs via semantic and localization scores, and trains MLLMs with DPO to distinguish high-precision grounding (chosen) from ambiguous or erroneous grounding (rejected), substantially improving fine-grained spatial understanding—particularly at high IoU thresholds.

## Background & Motivation

Multimodal large language models (MLLMs) have achieved remarkable progress on spatial understanding tasks such as referential dialogue and grounding captioning. Nevertheless, MLLMs still exhibit notable deficiencies in fine-grained spatial perception:

**Vague region descriptions**: Model-generated grounded region descriptions are often insufficiently detailed, with inaccurate object localization.

**Attention shift**: Models may be distracted by objects outside the queried region, failing to focus on the user-specified area.

**Lack of positive/negative sample feedback**: Existing supervised fine-tuning (SFT) optimizes models to imitate positive samples (ground truth) only, without penalizing erroneous localization generated at inference time—analogous to the absence of positive/negative sample training mechanisms found in conventional object detection.

**Key Challenge**: The SFT training paradigm lacks direct supervision over the quality of MLLM outputs—the model learns what is "correct" but receives no signal about what is "incorrect."

**Key Insight**: Introducing preference optimization (DPO) into spatial understanding, enabling models to distinguish precise localization from erroneous localization rather than merely imitating annotations. Existing preference optimization work primarily targets hallucination reduction; preference optimization for fine-grained spatial alignment remains largely unexplored.

## Method

### Overall Architecture

SPR adopts a three-stage DPO pipeline:
1. **Collect raw MLLM responses**: Construct random region queries → generate diverse grounded region descriptions via diversified prompts.
2. **Evaluate and rank**: Compute composite scores (semantic + localization) → pair highest/lowest scoring responses → refine the chosen description.
3. **Preference optimization training**: DPO + LoRA fine-tuning.

### Key Designs

#### 1. Random Region Query Construction

Existing region description datasets are overly simple (e.g., short phrases such as "a car parked on the street"), making it difficult to produce preference pairs with sufficient contrast. The authors therefore design a query region construction scheme from scratch:

- Filter object-rich images from the Objects365 dataset.
- Randomly select one annotated bounding box as the initial region.
- Iteratively expand to the nearest neighboring objects until 4+ objects are included.
- Construct diversified prompts (containing the cropped region image + object references) to elicit multiple candidate region descriptions from the MLLM.

#### 2. Semantic Score

Measures the semantic alignment between a description and the queried region:

$$S_{sem} = \frac{1}{2} \left( S(I_{crop}, T) + S_{local}(I, T) \right)$$

- $S(I_{crop}, T)$: CLIP cosine similarity between the cropped region image and the description text.
- $S_{local}(I, T)$: similarity between the description and region embeddings extracted from the full image via a local attention layer.

**Design Motivation**: Using only the cropped image ignores surrounding context; incorporating full-image similarity via local attention compensates for this limitation.

#### 3. Localization Score

Evaluates the localization accuracy and descriptive detail of objects mentioned in the description:

- Extract object bounding boxes from the description text using Grounding DINO.
- Combine the original annotation boxes to form the ground truth set.
- Merge the localization boxes in the MLLM description and the Grounding DINO results as the prediction set.
- Compute the average IoU (threshold 0.5) between predicted and GT boxes.

$$S_{loc} = \frac{1}{n} \sum_i^n \max_j \mathbf{p}[i,j]$$

This encourages the model to describe more objects and provide accurate localization boxes.

#### 4. Composite Scoring and Description Refinement

$$S = \lambda S_{sem} + (1-\lambda) S_{loc}, \quad \lambda = 0.8$$

- The highest- and lowest-scoring descriptions are paired as chosen/rejected data.
- **Refinement step**: The chosen description undergoes further localization quality improvement—predictions with IoU > 0.5 are retained, and their localization boxes are replaced with matched GT boxes, thereby enlarging the localization gap between chosen and rejected pairs.

### Loss & Training

DPO loss:

$$\mathcal{L} = -\mathbb{E}_{(x,y_w,y_l)} \left[ \log\sigma\left(\beta \log\frac{\pi_*(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log\frac{\pi_*(y_l|x)}{\pi_{ref}(y_l|x)}\right) \right]$$

- The base model serves as the frozen reference policy $\pi_{ref}$.
- The policy model $\pi_*$ updates weights via LoRA.
- Preference data is constructed from 10k images (Objects365).
- Training requires 1×A100: approximately 3 hours for Ferret-7B and 5 hours for Ferret-13B.

## Key Experimental Results

### Main Results (Referring Expression Comprehension, Acc@0.5)

| Model | RefCOCO val | RefCOCO+ val | RefCOCOg val | Flickr30k val |
|------|------------|-------------|-------------|---------------|
| Ferret-7B | 87.49 | 80.78 | 83.93 | 80.39 |
| + SPR | **88.39** | **82.07** | **85.58** | **81.53** |
| Ferret-13B | 89.48 | 82.81 | 85.83 | 81.13 |
| + SPR | **89.94** | **83.29** | **86.46** | **81.82** |
| CogVLM-17B | 92.76 | 88.68 | 89.75 | - |
| + SPR | **92.95** | **88.83** | **90.01** | - |

SPR consistently improves performance across three distinct baseline MLLMs.

### Ablation Study (REC under Varying IoU Thresholds)

| IoU Threshold | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---------|-----|-----|-----|-----|-----|
| Ferret-7B | 83.91 | 81.28 | 76.72 | 67.02 | 43.25 |
| + SPR | 84.93 | 82.36 | 78.42 | 70.09 | **52.21** |
| Gain | +1.02 | +1.08 | +1.70 | +3.07 | **+8.96** |

The improvement from SPR grows substantially as the IoU threshold increases—**achieving a gain of 8.96 points for the 7B model at IoU=0.9**—demonstrating that SPR genuinely improves localization precision rather than merely increasing detection quantity.

### Key Findings

1. **Amplified gains at high IoU thresholds**: SPR's advantage becomes more pronounced under strict localization requirements (+8.96 points at IoU=0.9).
2. **DPO outperforms SFT**: Using only chosen data for SFT yields approximately half the improvement (REC +0.44 vs. DPO +1.02).
3. **Both score components are necessary**: Setting $\lambda=0$ or $\lambda=1$ performs substantially worse than the composite configuration at $\lambda=0.8$.
4. **Description refinement is especially critical for multi-object grounding**: Gains are more pronounced on Phrase Grounding benchmarks.
5. **Improved spatial capability transfers to general ability**: General benchmarks including GQA and TextVQA also improve, along with reductions in hallucination as measured by POPE.

## Highlights & Insights

1. **Transferring positive/negative sample principles from conventional detection to MLLMs**: Addresses the critical gap of absent negative supervision in MLLM spatial understanding.
2. **Fully automated pipeline**: Requires no external MLLMs or human annotation, ensuring strong scalability.
3. **Extremely low training cost**: Completed on a single A100 GPU in 3–5 hours.
4. **Model-agnostic**: Validated across three distinct MLLMs—Ferret, LLaVA-OV, and CogVLM.
5. **Evaluation design with high IoU thresholds**: Goes beyond the conventional IoU=0.5 metric, providing a more rigorous analysis of localization precision.

## Limitations & Future Work

- CLIP as a semantic scorer may be insufficiently sensitive to certain fine-grained distinctions.
- Preference data is constructed solely from Objects365, and domain generalization remains to be verified.
- Joint SFT and DPO training strategies have not been explored.
- The localization score depends on the detection quality of Grounding DINO.
- Validation is limited to 2D image scenarios; extension to video or 3D spatial understanding has not been investigated.

## Related Work & Insights

- Works such as RLHF-V and POVID introduce preference optimization into MLLMs to reduce hallucinations; this work extends the paradigm to spatial understanding.
- Spatially-aware MLLMs including Ferret, Shikra, and Kosmos-2 provide the foundational architectural support.
- CLIP-DPO circumvents costly human annotation, consistent with the automated scoring philosophy of this work.
- Positive/negative sample balancing strategies from conventional detection, such as Focal Loss, inspired the design principles of SPR.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Focuses preference optimization on spatial localization precision, filling an important gap.
- **Technical Depth**: ⭐⭐⭐ — The method is concise and effective, though technical complexity is moderate.
- **Practical Value**: ⭐⭐⭐⭐⭐ — Low-cost and plug-and-play, with direct improvements to MLLM spatial understanding.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear and systematic presentation with well-designed experiments.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] MM-Spatial: Exploring 3D Spatial Understanding in Multimodal LLMs](mm-spatial_exploring_3d_spatial_understanding_in_multimodal_llms.md)
- [\[ICCV 2025\] STI-Bench: Are MLLMs Ready for Precise Spatial-Temporal World Understanding?](sti-bench_are_mllms_ready_for_precise_spatial-temporal_world_understanding.md)
- [\[ICCV 2025\] Instruction-Oriented Preference Alignment for Enhancing Multi-Modal Comprehension Capability of MLLMs](instruction-oriented_preference_alignment_for_enhancing_multi-modal_comprehensio.md)
- [\[NeurIPS 2025\] Struct2D: A Perception-Guided Framework for Spatial Reasoning in MLLMs](../../NeurIPS2025/multimodal_vlm/struct2d_a_perception-guided_framework_for_spatial_reasoning_in_mllms.md)
- [\[ICLR 2026\] On the Generalization Capacities of MLLMs for Spatial Intelligence](../../ICLR2026/multimodal_vlm/on_the_generalization_capacities_of_mllms_for_spatial_intelligence.md)

</div>

<!-- RELATED:END -->

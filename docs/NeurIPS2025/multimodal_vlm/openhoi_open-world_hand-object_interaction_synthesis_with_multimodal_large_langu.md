---
title: >-
  [Paper Note] OpenHOI: Open-World Hand-Object Interaction Synthesis with Multimodal Large Language Models
description: >-
  [NeurIPS 2025][Multimodal VLM][hand-object interaction] This paper proposes OpenHOI, a framework that leverages the commonsense reasoning capabilities of multimodal large language models (MLLMs) to infer contact regions and grasp types for unseen objects, enabling open-world hand-object interaction synthesis without requiring per-object training data collection.
tags:
  - NeurIPS 2025
  - Multimodal VLM
  - hand-object interaction
  - open-world
  - MLLM
  - contact reasoning
  - grasp synthesis
date: 2026-05-08
content_hash: b0b8092f1f2483e0
---

# OpenHOI: Open-World Hand-Object Interaction Synthesis with Multimodal Large Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2505.18947](https://arxiv.org/abs/2505.18947)
**Code**: Available
**Area**: Multimodal VLM / Hand-Object Interaction
**Keywords**: hand-object interaction, open-world, MLLM, contact reasoning, grasp synthesis

## TL;DR

This paper proposes OpenHOI, a framework that leverages the commonsense reasoning capabilities of multimodal large language models (MLLMs) to infer contact regions and grasp types for unseen objects, enabling open-world hand-object interaction synthesis without requiring per-object training data collection.

## Background & Motivation

**State of the Field**: Hand-object interaction (HOI) synthesis is critical for VR/AR, robotic grasping, and animation. Existing methods are typically trained on closed object sets and generalize poorly to novel objects.

**Limitations of Prior Work**: (1) Large amounts of hand-object contact data are required, which are difficult to collect; (2) prior knowledge of novel object geometry is lacking; (3) learning-based methods degrade severely on out-of-distribution objects.

**Root Cause**: Closed-set training vs. open-world deployment — how to generate plausible grasp poses for unseen objects?

**Starting Point**: MLLMs (e.g., GPT-4V) encode rich object commonsense — they "know" that a cup handle affords a power grasp and that a smooth egg requires a precision grasp. This commonsense reasoning can guide grasp synthesis.

**Core Idea**: MLLM infers contact regions and grasp types → conditioned grasp pose generation → physics-based optimization for physical plausibility.

## Method

### Overall Architecture

Object image/description → MLLM reasoning (contact regions, grasp type, force level) → conditioned diffusion model for hand pose generation → physics post-processing (interpenetration removal, contact optimization).

### Key Designs

1. **MLLM Contact Reasoning**

   - **Function**: Infers contactable regions, suitable grasp types, and force levels for a given object.
   - **Mechanism**: Object images and textual descriptions are provided to the MLLM; carefully designed prompts elicit structured contact information as output.
   - **Design Motivation**: The commonsense knowledge of MLLMs compensates for the absence of training data — the model inherently "knows" how to hold a cup.

2. **Conditioned Grasp Generation**

   - **Function**: Generates MANO hand parameters conditioned on MLLM-inferred contact conditions.
   - **Mechanism**: A conditional diffusion model takes contact heatmaps and grasp-type embeddings as conditions to generate hand pose parameters.
   - **Design Motivation**: Diffusion models produce diverse and plausible poses rather than a single deterministic output.

3. **Physics Post-Processing**

   - **Function**: Removes hand-object interpenetration and improves contact quality.
   - **Mechanism**: Iterative optimization — detect interpenetrations → push hand along surface normals → optimize contact area.
   - **Design Motivation**: Learning-based methods alone cannot guarantee physical plausibility; post-processing corrects residual artifacts.

### Loss & Training

Diffusion model training uses a denoising objective $\|ε - ε_\theta(x_t, t, c)\|^2$, where the contact condition $c$ comprises the MLLM-inferred region heatmap and grasp-type embedding.

## Key Experimental Results

### Main Results

| Method | Penetration Depth↓ | Contact Area↑ | Physical Stability↑ | Novel Object Generalization |
|---|---|---|---|---|
| GraspTTA | 3.2mm | 12.5cm² | 78% | ✗ Poor |
| ContactOpt | 2.8mm | 15.3cm² | 82% | ✗ Poor |
| MLLM baseline | 4.5mm | 8.7cm² | 65% | ✓ Partial |
| **OpenHOI** | **1.5mm** | **18.2cm²** | **91%** | **✓ Strong** |

### Ablation Study

| Configuration | Penetration Depth | Physical Stability | Note |
|---|---|---|---|
| w/o MLLM reasoning | 2.8mm | 82% | No contact prior |
| w/ MLLM, w/o physics post-processing | 2.1mm | 85% | Prior present but penetration remains |
| **Full OpenHOI** | **1.5mm** | **91%** | **MLLM + physics** |

### Key Findings

- MLLM contact reasoning reduces penetration depth from 2.8mm to 2.1mm; physics post-processing further reduces it to 1.5mm.
- OpenHOI substantially outperforms closed-set methods on novel objects outside the training distribution.
- High diversity is observed — multiple plausible grasp poses can be generated for the same object.
- MLLM grasp-type predictions agree with human annotations at a rate exceeding 85%.

## Highlights & Insights

- **Commonsense-Driven**: Replacing training data with MLLM object commonsense represents a novel generalization paradigm, transferable to open-world grasp planning in robotic manipulation.
- **Modular Design**: The three-stage pipeline — MLLM reasoning, diffusion-based generation, and physics optimization — is fully decoupled, allowing each component to be independently replaced or improved.
- **Practical Applicability**: The framework has direct application value for virtual hand-object interaction in VR/AR contexts.

## Limitations & Future Work

- MLLM inference latency is high (on the order of seconds), precluding real-time applications.
- Commonsense reasoning may be inaccurate for objects with atypical or extreme geometries.
- Only single-hand interaction is addressed; bimanual coordination is not considered.
- Physics post-processing may alter the contact locations originally inferred by the MLLM.

## Related Work & Insights

- **vs. GraspTTA**: GraspTTA requires the target object category to be present in the training set; OpenHOI achieves genuine open-world generalization.
- **vs. ContactGen**: ContactGen learns general contact patterns but lacks object-specific commonsense-driven guidance.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ MLLM commonsense-driven HOI synthesis is a novel direction.
- Experimental Thoroughness: ⭐⭐⭐⭐ Quantitative and qualitative evaluations are comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Methodology is clearly described.
- Value: ⭐⭐⭐⭐⭐ Open-world HOI is an important and impactful application scenario.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] On Large Multimodal Models as Open-World Image Classifiers](../../ICCV2025/multimodal_vlm/on_large_multimodal_models_as_open-world_image_classifiers.md)
- [\[NeurIPS 2025\] AffordBot: 3D Fine-grained Embodied Reasoning via Multimodal Large Language Models](affordbot_3d_fine-grained_embodied_reasoning_via_multimodal_large_language_model.md)
- [\[NeurIPS 2025\] MM-OPERA: Benchmarking Open-ended Association Reasoning for Large Vision-Language Models](mm-opera_benchmarking_open-ended_association_reasoning_for_large_vision-language.md)
- [\[NeurIPS 2025\] Causal-LLaVA: Causal Disentanglement for Mitigating Hallucination in Multimodal Large Language Models](causalllava_causal_disentanglement_for_mitigating_hallucinat.md)
- [\[NeurIPS 2025\] Adapting Vision-Language Models for Evaluating World Models](adapting_visionlanguage_models_for_evaluating_world_models.md)

<!-- RELATED:END -->

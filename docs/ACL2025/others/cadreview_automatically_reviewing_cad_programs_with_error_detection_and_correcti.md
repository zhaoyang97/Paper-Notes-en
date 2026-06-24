---
title: >-
  [Paper Note] CADReview: Automatically Reviewing CAD Programs with Error Detection and Correction
description: >-
  [ACL 2025][CAD program review] Proposes the CAD program review task and the ReCAD framework, which automatically detects errors in CAD programs and generates correction feedback based on reference images, constructing the CADReview dataset containing 20K+ samples (8 error classes).
tags:
  - "ACL 2025"
  - "CAD program review"
  - "3D modeling"
  - "multimodal large language models"
  - "error detection and correction"
  - "geometric component identification"
date: 2026-05-08
content_hash: 3b4a69822f9b6f4c
---

# CADReview: Automatically Reviewing CAD Programs with Error Detection and Correction

**Conference**: ACL 2025  
**arXiv**: [2505.22304](https://arxiv.org/abs/2505.22304)  
**Code**: [Available](https://cgl-pro.github.io/cadreview)  
**Area**: Others  
**Keywords**: CAD program review, 3D modeling, multimodal large language models, error detection and correction, geometric component identification

## TL;DR

Proposes the CAD program review task and the ReCAD framework, which automatically detects errors in CAD programs and generates correction feedback based on reference images, constructing the CADReview dataset containing 20K+ samples (8 error classes).

## Background & Motivation

Computer-aided design (CAD) is crucial in industrial design. Designers need to iteratively review the consistency between 3D prototypes and design drawings, a process that is extremely time-consuming. Existing AI methods primarily focus on CAD program generation, neglecting the subsequent review and correction phases.

Key Challenge:

**Difficulties in Aligning Geometric Components**: A 3D object consists of multiple components, each corresponding to a specific code block (containing subroutines, control flow, and boolean operations) in the program. Models need to associate visual components with these code blocks.

**Hidden Internal Components**: 3D objects may contain internal structures (e.g., inside hexagonal holes) that cannot be directly inspected visually, requiring programmatic analysis.

**Mapping Geometric Operations**: Accurate correction requires mapping geometric operations (translation, rotation) to corresponding code modifications.

**Poor MLLM Performance**: Advanced models like GPT-4o show limited effectiveness in CAD review, mainly due to their inability to effectively align code with visual information.

## Method

### Overall Architecture

The ReCAD framework consists of two MLLM modules:
- **Feedback Generator $\phi_1$**: Takes the reference image, CAD program, and rendered image as input, and outputs error description feedback.
- **Code Editor $\phi_2$**: Utilizes feedback to guide program error correction.

Training employs a two-stage SFT + RL refinement strategy. Both modules share the MLLM backbone network (supporting Qwen2-VL and LLaVA-OV).

### Key Designs

#### Geometric Component Recognition (GCR) Mechanism

To enhance the feedback generator's ability to identify visual and code components, a sequential training pipeline is designed:

**CAD Captioning**: A vision-language projector is trained using image-text pairs from the Text2CAD dataset, enabling the LLM to learn to recognize geometric components (e.g., "a round CAD model with a central hole"). Only the projector is trained, while the vision encoder and LLM are frozen.

**CAD Grounding**: Three alignment tasks are introduced:
- Semantic matching: Predicting the corresponding code block based on semantic labels.
- Coordinate matching: Querying the corresponding code block based on coordinates in the rendered image.
- Coordinate localization: Determining the coordinates of a given code block in the image.

Training data is constructed using the CADTalk dataset and GPT-4o augmented data. Initialized on top of the captioning training, the vision encoder and LLM are jointly fine-tuned. The grounding accuracy reaches approximately 90% without compromising captioning performance.

#### Spatial Geometric Operation (SGO) Mechanism

Enhances the code editor's ability to understand spatial relationships and geometric transformations:

- Randomly masks 30% of CAD program code blocks for the model to predict (code completion task).
- Observes that the cross-entropy loss converges rapidly on syntactically similar code but fails to reflect numerical differences in operations.
- Quantizes spatial position values into 8 bits (up to 256) and doubles the loss weight for numerical tokens.

$$\mathcal{L}_{sgo} = w_i \cdot \mathcal{L}_i, \quad w_i = \begin{cases} 2, & \text{if } y_i \in \mathbb{R} \\ 1, & \text{otherwise} \end{cases}$$

#### SFT Training

Feedback Generation: Freeze $\phi_1$, fine-tune the LLM using LoRA (rank=8), and predict ground truth feedback using cross-entropy loss.
Code Editing: $\phi_2$ takes the reference image, program, rendered image, and generated feedback as input, and fine-tunes the LLM using LoRA (rank=64).

### Loss & Training

#### RL Refinement (DPO)

Two reward functions are designed to refine the feedback generator:

**Error Diagnosis Reward** $\mathcal{V}_d$: 1 if both code blocks and error types in the feedback are correct, 0 otherwise.

**Visual Similarity Reward** $\mathcal{V}_v$: Extracts features of the reference image and the edited rendered image using the vision encoder of $\phi_1$, computing the cosine similarity.

Training Process: 2,000 samples are randomly selected, K feedbacks are generated using top-p sampling (T=0.8, p=0.9), preference pairs are constructed based on the two rewards, pairs with a diagnosis reward of 1 and a visual reward difference exceeding 0.25 are retained, and optimization is done using DPO.

## Key Experimental Results

### Main Results

Dataset: CADReview contains 17,334 training / 2,000 validation / 1,615 test samples, including both human-written and machine-generated programs.

| Method | Machine-Acc↑ | Machine-CD↓ | Machine-JSD↓ | Human-Acc↑ | Human-CD↓ | Human-JSD↓ |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| Claude 3.5 | 32.19 | 4.06 | 36.16 | 21.51 | 9.03 | 79.63 |
| Gemini 2.0 | 36.83 | 3.96 | 11.49 | 23.36 | 5.97 | 55.85 |
| GPT-4o | 41.54 | 4.34 | 12.07 | 31.84 | 5.52 | 48.70 |
| Llama 3.2† | 56.23 | 2.71 | 5.44 | 51.24 | 5.82 | 36.18 |
| **ReCAD-LA†** | **73.11** | 1.45 | 3.42 | 59.15 | 4.76 | 33.09 |
| **ReCAD-QW†** | 71.83 | **1.43** | **2.80** | **63.60** | **4.42** | **30.87** |

(CD and JSD are both multiplied by $10^3$; † indicates fine-tuning on CADReview)

| Method | Machine-IR↓ | Human-IR↓ |
|------|:---:|:---:|
| GPT-4o | 28.43 | 18.57 |
| ReCAD-LA | 0.00 | 13.74 |
| ReCAD-QW | 0.00 | 10.59 |

### Ablation Study

| Ablated Component | Machine-Acc | Machine-CD | Human-Acc | Human-CD |
|----------|:---:|:---:|:---:|:---:|
| w/o GCR | 67.43 | 2.08 | 51.75 | 5.86 |
| w/o SGO | 70.55 | 1.93 | 58.79 | 6.84 |
| w/o Feedback | — | 1.84 | — | 5.71 |
| w/o Reward | 71.16 | 1.57 | 58.01 | 5.27 |
| **Full ReCAD-LA** | **73.11** | **1.45** | **59.15** | **4.76** |

### Key Findings

1. **GPT-4o achieves only 41.5%/31.8% accuracy**: Closed-source MLLMs struggle to align CAD programs with visual information, and the rate of invalid programs exceeds 25%.
2. **ReCAD significantly outperforms fine-tuned models of the same scale**: ReCAD achieves a 12-17% accuracy improvement over Llama 3.2 (same scale but without GCR/SGO).
3. **GCR is critical**: Removing GCR leads to overall performance degradation, and the feedback-free version even outperforms the version without GCR, indicating that incorrect component alignment yields counterproductive feedback.
4. **SGO is more critical for complex programs**: Its impact on human-written programs is greater than on machine-generated ones.
5. **Human programs are more challenging than machine ones**: All methods experience performance drops on human programs due to more complex control flows and nested structures.

## Highlights & Insights

1. **New Task Definition**: Introduces the concept of code review to the CAD domain for the first time, bridging NLP and 3D modeling.
2. **Sequential Training Strategy**: A step-by-step training pipeline (captioning $\rightarrow$ grounding $\rightarrow$ feedback generation) allows the model to progressively learn alignments from vision to code.
3. **Numerical Loss Weighting**: Identifies the paramount importance of numerical tokens in CAD code and doubles their weight, solving the issue of rapid syntactic convergence coupled with numerical inaccuracies.
4. **Feedback Loop Design**: The dual-module design of the feedback generator and code editor makes the correction process interpretable, while DPO refinement ensures synergy.
5. **8-Class Error Taxonomy**: Systematically defines types of CAD program errors, establishing a standardized framework for future research.

## Limitations & Future Work

1. Optimal code editing solutions are not considered (a single geometric component can be implemented via various code representations).
2. Currently, only the OpenSCAD language is supported; although framework portability is claimed, it has not been validated.
3. Error types are manually predefined and do not cover all possible design discrepancies.
4. Only one error is processed at a time, whereas practical scenarios may involve multiple co-occurring errors.

## Related Work & Insights

- **CAD Program Generation**: Work such as DeepCAD and Text2CAD focuses on generation, whereas this study complements it with the review dimension.
- **Code Editing**: Frameworks like CoffeeGym and CodeAgent focus on general-purpose code, whereas this study extends the scope to geometric code.
- **MLLM Applications**: Demonstrates that MLLMs, when specifically trained, can significantly outperform general closed-source models in domain-specific tasks (such as CAD).
- **Insight**: This review framework can also inspire research on other domain-specific code formats (e.g., molecular descriptions, circuit designs).

## Rating

- **Novelty**: ★★★★☆ — First CAD program review task, both the task definition and dataset are novel contributions.
- **Technical Depth**: ★★★★☆ — Sophisticated GCR/SGO mechanism designs, well-reasoned DPO refinement.
- **Experimental Thoroughness**: ★★★★☆ — Multiple baselines (including GPT-4o/Claude/Gemini) + ablation studies + case analyses.
- **Value**: ★★★★☆ — Directly beneficial to CAD design workflows.
- **Writing Quality**: ★★★★☆ — Highly illustrative, clear framework description.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] LegalReasoner: Step-wised Verification-Correction for Legal Judgment Reasoning](legalreasoner_step-wised_verification-correction_for_legal_judgment_reasoning.md)
- [\[ACL 2025\] Zero-Shot Conversational Stance Detection: Dataset and Approaches](zero-shot_conversational_stance_detection_dataset_and_approaches.md)
- [\[ACL 2025\] Explaining Matters: Leveraging Definitions and Semantic Expansion for Sexism Detection](explaining_matters_leveraging_definitions_and_semantic_expansion_for_sexism_dete.md)
- [\[ICML 2025\] How Do Transformers Learn Variable Binding in Symbolic Programs?](../../ICML2025/others/how_do_transformers_learn_variable_binding_in_symbolic_programs.md)
- [\[ACL 2025\] SEOE: A Scalable and Reliable Semantic Evaluation Framework for Open Domain Event Detection](seoe_semantic_eval.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] A Deep Learning Model of Mental Rotation Informed by Interactive VR Experiments
description: >-
  [ICML 2026][Interpretability][mental rotation] This paper constrains model design using interactive VR experiments and proposes a mental rotation model composed of a 3D equivariant spatial encoder…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "mental rotation"
  - "VR interaction"
  - "equivariant representation"
  - "neuro-symbolic model"
  - "spatial reasoning"
date: 2026-05-08
content_hash: ed4f1027b6b19ced
---

# A Deep Learning Model of Mental Rotation Informed by Interactive VR Experiments

**Conference**: ICML 2026  
**arXiv**: [2512.13517](https://arxiv.org/abs/2512.13517)  
**Code**: https://github.com/rkhz/menrot  
**Area**: Robotics / Spatial Reasoning / Cognitive Modeling  
**Keywords**: mental rotation, VR interaction, equivariant representation, neuro-symbolic model, spatial reasoning  

## TL;DR
This paper constrains model design using interactive VR experiments and proposes a mental rotation model composed of a 3D equivariant spatial encoder, a neuro-symbolic object encoder, and an MLP for action decision-making. The model replicates human mental rotation behavior in terms of accuracy, number of actions, and partial response time (RT) trends.

## Background & Motivation
**Background**: Mental rotation is a classic task in cognitive science for studying spatial representation. It involves showing two 3D Shepard-Metzler block shapes from different perspectives and asking whether they represent the same or mirrored objects. Classic results show that human response times typically increase linearly with the angular difference, which has long been interpreted as the brain continuously rotating some internal representation in the "mind's eye."

**Limitations of Prior Work**: Modern vision models can achieve high accuracy on many 3D tasks, but few models simultaneously explain "how to judge" and "why response times and action counts resemble humans." Models focused solely on classification may use shortcuts entirely different from human processes, failing as mechanistic models of mental rotation even if they provide correct answers.

**Key Challenge**: Human behavior supports two seemingly conflicting views. On one hand, RT and neuro/psychophysical evidence suggest the existence of rotatable internal spatial representations. On the other hand, interactive VR experiments show humans typically perform only a few discrete, large-magnitude rotations, suggesting decisions may instead rely on abstract symbols or quadrant-level object descriptions.

**Goal**: The authors aim to construct a neural-network-based, executable mechanistic model of mental rotation that not only achieves human-level accuracy but also replicates the action patterns and partial RT patterns observed in VR experiments.

**Key Insight**: New VR experiments were conducted where participants could occasionally use a joystick to rotate the right-hand object, allowing for the observation of their chosen rotation actions. The experiments found that humans typically perform only approximately 1.05 ballistic large-magnitude rotations and make a judgment once the object is within a quadrant near the target. This directly inspired the "quadrant-dependent symbolic representation" hypothesis in the model.

**Core Idea**: A spatial equivariant representation is utilized for the "capability to rotate," a symbolic object description for "knowing how to rotate and when to judge," and a small decision agent to loop between the two.

## Method

### Overall Architecture
The approach consists of mutually supporting experimental and modeling components. The experimental part designed a VR mental rotation task: 19 participants viewed two Shepard-Metzler shapes (composed of 10 cubes with 3 turns) in VR, with relative angles of $0^\circ, 60^\circ, 120^\circ, 180^\circ$ around the Y-axis. In the "No-Action" condition, participants could only perform mental rotation; in the "Action" condition, participants could use a joystick to rotate the right-hand object, though the object disappeared during rotation (showing only an angular ring) to prevent reliance on continuous visual feedback alignment.

VR behaviors provided two key observations. First, in the "Action" condition, humans performed an average of only 1.05 rotation actions, with each action averaging $73.1^\circ$, indicating they do not continuously scan the entire angular space at a fixed speed. Second, after the final action, the right-hand object typically fell within a $[-45^\circ, +45^\circ]$ range of the target object before a quick judgment was made, suggesting that only coarse alignment to a specific quadrant is required.

The model stacks three modules based on these findings. Module I is a convolutional autoencoder in the style of an Equivariant Neural Renderer, which recovers a spatial latent from a single 2D image that can be manipulated by 3D rotation matrices. Module II is a Vision Symbolic Model that uses a ViT encoder and an autoregressive Transformer decoder to convert spatial latents into serialized symbolic descriptions of Shepard-Metzler shapes. Module III is an MLP decision-maker that inputs two symbolic descriptions and outputs either a "same / mirror" judgment or a quadrant-based rotation action. If an action is output, it is applied to the 3D latent of the source, which is then re-encoded and re-evaluated until a match/mismatch judgment is reached or a 6-action limit is exceeded.

### Key Designs

1.  **3D Equivariant Spatial Representation**:
    - **Function**: Constructs a 3D object representation from a single 2D perspective that can be rotated within the latent space.
    - **Mechanism**: Module I is trained on image pairs of the same object in different poses. After encoding one perspective into a latent, it is transformed via a 3D rotation matrix to another pose and then decoded to reconstruct the target view. Consequently, the latent must maintain equivariance with respect to $SO(3)$ rotations.
    - **Design Motivation**: Mental rotation requires at least some form of manipulable spatial representation; direct classification from pixels often learns shortcuts specific to training objects and fails to generalize to new shapes.

2.  **Quadrant-Dependent Neuro-Symbolic Object Description**:
    - **Function**: Converts 3D spatial representations into discrete symbolic sequences that vary with the observed quadrant, providing an abstract structure for action selection.
    - **Mechanism**: The authors divide the 360-degree view into 4 quadrants and describe the shape using 9 directional transitions (Up, Down, Front, Back, Left, Right) starting from the cube closest to the observer. The VSM learns to predict these symbolic sequences from the 3D latent; thus, the same object has different but structured descriptions in different quadrants.
    - **Design Motivation**: VR experiments suggest humans seek coarse quadrant alignment rather than precise angular alignment; symbolic descriptions provide such quadrant-level, compositional object representations.

3.  **Action-Based Decision Agent**:
    - **Function**: Compares two objects in symbolic space and decides whether to judge "same/mirror" directly or execute a discrete rotation action.
    - **Mechanism**: Module III receives the concatenated symbolic logits of the target and source, outputting one of five categories: same, mirror, clockwise one quadrant, counter-clockwise one quadrant, or two quadrants. During training, it only learns relational categories. During inference, if a rotation action is output, it is applied to the spatial latent of the source, followed by re-encoding and re-decision.
    - **Design Motivation**: This allows the model to possess both the "ability to rotate in spatial latent space" and the behavior of "solving tasks with human-like discrete actions," rather than bypassing the process via a fully invariant classifier.

### Loss & Training
The three modules are trained independently. Module I is trained on 50,000 pairs of Shepard-Metzler images for equivariant reconstruction. Module II, with the EqNR encoder frozen, maps 3D latents to symbolic sequences using 201,600 image-symbolic description pairs with varying azimuths and fixed $25^\circ$ elevation. Module III is trained on 38,400 mental rotation relationship tasks. The testing phase uses unseen Shepard-Metzler objects with balanced match/mismatch pairs and four angular differences.

## Key Experimental Results

### Main Results

| Dataset / Setting | Metric | Ours | Human / Control | Conclusion |
| :--- | :--- | :--- | :--- | :--- |
| VR No-Action | Total Accuracy | N/A | 91.14% | Humans complete tasks stably via pure mental rotation |
| VR Action | Total Accuracy | N/A | 95.33% | Accuracy increases slightly with manipulation, revealing action strategies |
| Held-out Shepard-Metzler | Total Accuracy | 96.39% | Human Action 95.33% | The model achieves human-level overall performance |
| Match / Mismatch Split | Accuracy | 96.39% / 95.87% | Human Match 96.13%, Mismatch 94.53% | No bias toward either judgment category |
| Action Behavior | Avg. Action Pattern | Mostly 0 for $0^\circ$, 1 for other angles | Human avg. 1.05 actions, single action mean $73.1^\circ$ | Replicates "few discrete actions" phenomenon |
| RT Trend | Scope of Explanation | Explains growth from $0^\circ$ to $120^\circ$ | Human Action/No-Action shows growth with angle | Partially fails to explain RT difference between $120^\circ$ and $180^\circ$ |

### Ablation Study

| Configuration | Key Metrics | Explanation |
| :--- | :--- | :--- |
| Siamese ResNet / ViT baseline | ~50% on test objects | Accurate on training objects but fails on new objects and 3D in-depth rotations |
| w/o Module I Equivariant Representation | VSM cannot reliably encode symbolic descriptions | Predicting symbolic sequences directly from 2D pixels is insufficient |
| w/o Module II Symbolic Representation | 90.71% w/ ViT encoder; 38.46% w/o VSM; 36.14% w/ patch embedding only | Symbolic modules are critical for action-based problem-solving |
| w/o actions (direct same/mirror) | 97.03% | High accuracy but fails to explain why humans perform rotation actions |
| 1-layer MLP + actions | 96.05% | Small decider still completes task and retains human-like action patterns |
| 1-layer MLP only same/mirror | 86.23% | Action-based decomposition has computational advantages over pure feedforward invariant classification at low capacity |

### Key Findings
- VR experiments strongly support a discrete "jump" strategy: participants do not continuously observe the rotation process but quickly perform one or two large actions to bring the object into the same quadrant before judging.
- Accuracy alone is not sufficient evidence. An MLP classifier without actions can reach 97.03%, but it bypasses the human-like process; therefore, this work prioritizes the joint matching of "accuracy + action patterns + RT trends."
- Equivariant spatial representations and symbolic representations serve different roles. The former allows geometric transformations of internal objects, while the latter enables compositional comparison and action selection. The combination is more like a cognitive model than a simple CNN/ViT classifier.

## Highlights & Insights
- The most compelling aspect of this paper is using VR motion data to inversely constrain model architecture. Rather than building a deep model and explaining behavior post-hoc, the authors observed actual human rotation and integrated "low-frequency, large-magnitude actions" and "quadrant-level symbolic descriptions" into the model's hypotheses.
- The ablation study distinguishes between "solving the problem" and "solving the problem like a human." Direct classifiers may achieve slightly higher accuracy but fail to explain actions; this serves as a reminder that in cognitive modeling and embodied AI, process matching is often more important than final classification scores.
- This hybrid architecture offers insights for machine learning. Many vision models are fragile regarding pose changes and spatial reasoning; this work demonstrates that explicit equivariant latents and symbolic relationship readouts can serve as effective structural priors, rather than relying solely on emergence from scale.

## Limitations & Future Work
- The model currently only handles Shepard-Metzler shapes and fixed Y-axis in-depth rotations, leaving a significant gap to open-world 3D object manipulation.
- Module I utilizes an explicit 3D rotation matrix for latent rotation; the paper does not provide a fully biologically plausible mechanism, which would require finer neural explanation as a cognitive model.
- The model only partially explains response times, particularly failing to account for the RT difference humans show between $120^\circ$ and $180^\circ$ conditions.
- Symbolic descriptions are currently researcher-designed. While interpretable and effective, whether humans utilize the same cube-transition encoding remains an open question.
- The VR experiment sample size is small (19 participants, with 4 atypical participants excluded); larger samples and more interactive conditions are needed for validation.

## Related Work & Insights
- **vs. Shepard & Metzler Classic Experiments**: While classic experiments inferred continuous mental rotation from RT, this work adds VR manipulation to provide new process evidence, supporting a discrete+continuous hybrid mechanism.
- **vs. Pure Siamese Visual Classifiers**: Siamese ResNet/ViT can memorize training objects but are nearly random on unseen 3D objects; the equivariant spatial latent used here better satisfies cross-view generalization requirements.
- **vs. 3D Shape Perception Models**: Some models predict human similarity judgments but do not produce sequential actions; this work explicitly models action selection, making it more suitable as a process model for mental rotation.
- **vs. Spatial Transformers / Equivariant Models**: Traditional geometric models emphasize transformable representations; this paper adds symbolic descriptions and an agentic loop, turning geometric operations into interpretable cognitive strategies.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Fresh approach using VR interactive data to constrain a neuro-symbolic mental rotation model.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Includes human experiments, held-out testing, and systematic ablation, though task variety and participant scale remain limited.
- Writing Quality: ⭐⭐⭐⭐☆ The narrative is clear, linking experimental phenomena to model design well; some appendix details are dense, and figures in the main text are somewhat dispersed.
- Value: ⭐⭐⭐⭐☆ Provides insights for spatial reasoning, cognitive modeling, and internal simulation in embodied AI/robotics, particularly promoting "process-interpretable" model evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Position: Zeroth-Order Optimization in Deep Learning Is Underexplored, Not Underpowered](position_zeroth-order_optimization_in_deep_learning_is_underexplored_not_underpo.md)
- [\[ICML 2026\] Probabilistic Modeling of Latent Agentic Substructures in Deep Neural Networks](probabilistic_modeling_of_latent_agentic_substructures_in_deep_neural_networks.md)
- [\[ICLR 2026\] PolySHAP: Extending KernelSHAP with Interaction-Informed Polynomial Regression](../../ICLR2026/interpretability/polyshap_extending_kernelshap_with_interaction-informed_polynomial_regression.md)
- [\[ACL 2026\] Experiments or Outcomes? Probing Scientific Feasibility in Large Language Models](../../ACL2026/interpretability/experiments_or_outcomes_probing_scientific_feasibility_in_large_language_models.md)
- [\[ICML 2026\] The Cylindrical Representation Hypothesis for Language Model Steering](the_cylindrical_representation_hypothesis_for_language_model_steering.md)

</div>

<!-- RELATED:END -->

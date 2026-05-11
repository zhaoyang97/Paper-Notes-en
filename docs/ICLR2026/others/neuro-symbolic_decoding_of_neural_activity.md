---
title: >-
  [Paper Note] Neuro-Symbolic Decoding of Neural Activity
description: >-
  [ICLR 2026][fMRI decoding] This paper proposes NEURONA, a neuro-symbolic framework for fMRI decoding and concept grounding. By decomposing visual scenes into symbolic programs (logical combinations of concepts)…
tags:
  - "ICLR 2026"
  - "fMRI decoding"
  - "neuro-symbolic"
  - "concept grounding"
  - "language of thought hypothesis"
  - "visual question answering"
date: 2026-05-08
content_hash: 662eb6b8a0be2e52
---

# Neuro-Symbolic Decoding of Neural Activity

**Conference**: ICLR 2026
**arXiv**: [2603.03343](https://arxiv.org/abs/2603.03343)
**Code**: None
**Area**: Neuroscience / Multimodal
**Keywords**: fMRI decoding, neuro-symbolic, concept grounding, language of thought hypothesis, visual question answering

## TL;DR
This paper proposes NEURONA, a neuro-symbolic framework for fMRI decoding and concept grounding. By decomposing visual scenes into symbolic programs (logical combinations of concepts), NEURONA substantially outperforms both end-to-end neural decoders and linear models on fMRI question-answering tasks.

## Background & Motivation

**Background**: The "language of thought" hypothesis in cognitive science posits that human cognition operates over structured, compositional representations. Neural decoding from fMRI has advanced considerably over the past decades, progressing from linear mappings to deep learning approaches.

**Limitations of Prior Work**: Existing neural decoding methods either employ simple linear models (interpretable but expressively limited) or end-to-end neural networks (powerful but black-box). Neither approach adequately captures the compositional relationships and logical structure among concepts.

**Key Challenge**: fMRI signals encode rich visual concepts, yet directly predicting natural-language answers from fMRI spans an excessively large semantic gap—requiring simultaneous understanding of scene structure, concept semantics, and question intent.

**Goal**: How can structured, compositional concept representations be decoded from fMRI activity, rather than directly predicting end-to-end answers?

**Key Insight**: The paper leverages composite concepts naturally encoded in image and video fMRI datasets, decomposing the decoding process into symbolic program execution.

**Core Idea**: Decompose fMRI decoding into a neuro-symbolic pipeline of "fMRI → concept detection → symbolic program execution → answer," yielding greater accuracy and interpretability than end-to-end approaches.

## Method

### Overall Architecture
NEURONA adopts a three-stage pipeline: (1) **Concept Grounding**: detecting the visual concepts present in the scene (e.g., "dog," "red," "running") from fMRI activity; (2) **Program Synthesis**: translating natural-language questions into logical programs over concepts (e.g., "Is the dog running?" → AND(detect(dog), detect(running))); (3) **Program Execution**: executing the program over the detected concept set to produce the answer.

### Key Designs

1. **fMRI Concept Grounding Module**:

    - Function: Detect the visual concepts present in the scene from fMRI voxel patterns.
    - Mechanism: A collection of linear probes $\{f_c : \mathbb{R}^V \to [0,1]\}_{c \in \mathcal{C}}$ is trained, where each probe $f_c$ predicts the presence probability of concept $c$ from fMRI activity $\mathbf{x} \in \mathbb{R}^V$. Zero-shot predictions from a pretrained vision-language model (e.g., CLIP) serve as pseudo-labels for training.
    - Design Motivation: Transforming high-dimensional fMRI signals into discrete concept sets substantially reduces the complexity of downstream reasoning.

2. **Question-to-Program Compilation**:

    - Function: Translate natural-language questions into executable symbolic programs.
    - Mechanism: An LLM (e.g., GPT-4) compiles questions into programs expressed in a domain-specific language (DSL). The DSL includes primitives such as detect(concept), AND/OR/NOT logical operators, count, and spatial_relation.
    - Design Motivation: Symbolic programs provide compositional generalization—novel concept combinations require no retraining, only new programs.

3. **Symbolic Program Execution Engine**:

    - Function: Execute symbolic programs over concept detection results to produce the final answer.
    - Mechanism: A program interpreter recursively evaluates each primitive: detect(c) queries the output $f_c(\mathbf{x})$ of the concept grounding module; logical operators act on probability values (AND as minimum, OR as maximum); a final threshold yields the answer.
    - Design Motivation: Deterministic execution guarantees interpretability and traceability—the precise influence of each concept detection on the final answer is always recoverable.

### Loss & Training
The concept grounding module is trained with binary cross-entropy loss using CLIP zero-shot detections as pseudo-labels. Only the concept grounding module requires training; program synthesis and execution are performed zero-shot.

## Key Experimental Results

### Main Results

| Method | BOLD5000-QA Overall | CNeuroMod-QA Overall |
|--------|--------------------|--------------------|
| Linear | 0.4692 | — |
| End-to-end Neural | ~0.50 | ~0.45 |
| NEURONA | **Substantially higher** | **Substantially higher** |

### Ablation Study

| Configuration | Accuracy | Notes |
|---------------|----------|-------|
| NEURONA (full) | Best | Neuro-symbolic three-stage pipeline |
| Linear decoding only | 0.47 | No compositional reasoning |
| End-to-end only | ~0.50 | Black-box, lacks structure |
| Without CLIP pseudo-labels | Degraded | Reduced concept grounding quality |

### Key Findings
- The neuro-symbolic approach significantly outperforms purely linear and purely end-to-end methods on fMRI QA.
- Gains are especially pronounced for action- and location-type questions, suggesting that these concepts have clear neural representations in fMRI.
- The accuracy of the concept grounding module constitutes the primary performance bottleneck of the overall system.
- Symbolic programs afford full interpretability—the reasoning chain behind each answer is completely traceable.

## Highlights & Insights
- **Computational validation of the language of thought hypothesis**: The neuro-symbolic approach demonstrates that compositional concept representations map more faithfully onto fMRI activity, providing indirect computational support for a core cognitive science hypothesis.
- **Interpretability as a free lunch**: Symbolic programs not only improve performance but also deliver fully transparent reasoning, offering significant value for neuroscientific inquiry.

## Limitations & Future Work
- The coverage of the concept vocabulary constrains the range of answerable questions.
- fMRI datasets are small in scale (tens to hundreds of samples), raising questions about generalizability.
- Pseudo-label quality depends on CLIP's zero-shot capability and may be unreliable for stimuli that deviate substantially from natural images.
- DSL design requires domain expertise, and different tasks necessitate different primitive sets.

## Related Work & Insights
- **vs. BrainBERT/Mind-Vis**: End-to-end decoding methods directly generate text or images from fMRI but lack structured reasoning capabilities.
- **vs. Neuro-symbolic AI (VQA)**: Similar in spirit to NS-VQA, which decomposes visual question answering into perception and reasoning stages; NEURONA transplants this paradigm into the fMRI decoding domain.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First application of neuro-symbolic methods to fMRI decoding, bridging cognitive science and AI.
- Experimental Thoroughness: ⭐⭐⭐ Datasets are small and quantitative comparisons are limited.
- Writing Quality: ⭐⭐⭐⭐ Interdisciplinary yet accessible.
- Value: ⭐⭐⭐⭐ Opens a new direction for fMRI decoding research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Modeling Neural Activity with Conditionally Linear Dynamical Systems](../../NeurIPS2025/others/modeling_neural_activity_with_conditionally_linear_dynamical_systems.md)
- [\[NeurIPS 2025\] An Empirical Investigation of Neural ODEs and Symbolic Regression for Dynamical Systems](../../NeurIPS2025/others/an_empirical_investigation_of_neural_odes_and_symbolic_regression_for_dynamical_.md)
- [\[ICLR 2026\] SEED: Towards More Accurate Semantic Evaluation for Visual Brain Decoding](seed_towards_more_accurate_semantic_evaluation_for_visual_brain_decoding.md)
- [\[ICLR 2026\] Entropic Confinement and Mode Connectivity in Overparameterized Neural Networks](entropic_confinement_and_mode_connectivity_in_overparameterized_neural_networks.md)
- [\[ICLR 2026\] Out of the Shadows: Exploring a Latent Space for Neural Network Verification](out_of_the_shadows_exploring_a_latent_space_for_neural_network_verification.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] CADMorph: Geometry-Driven Parametric CAD Editing via a Plan-Generate-Verify Loop
description: >-
  [NEURIPS2025][Image Generation][CAD editing] This paper proposes CADMorph, an iterative plan–generate–verify framework that leverages a pretrained Parameter-to-Shape (P2S) diffusion model and a Masked-Parameter-Predictio…
tags:
  - "NEURIPS2025"
  - "Image Generation"
  - "CAD editing"
  - "parametric sequence"
  - "latent diffusion"
  - "masked prediction"
  - "test-time scaling"
date: 2026-05-08
content_hash: 287721aa3855bcc2
---

# CADMorph: Geometry-Driven Parametric CAD Editing via a Plan-Generate-Verify Loop

**Conference**: NEURIPS2025  
**arXiv**: [2512.11480](https://arxiv.org/abs/2512.11480)  
**Code**: To be confirmed  
**Area**: Image Generation  
**Keywords**: CAD editing, parametric sequence, latent diffusion, masked prediction, test-time scaling

## TL;DR

This paper proposes CADMorph, an iterative plan–generate–verify framework that leverages a pretrained Parameter-to-Shape (P2S) diffusion model and a Masked-Parameter-Prediction (MPP) large language model to achieve geometry-driven parametric CAD editing without requiring triplet training data.

## Background & Motivation

CAD models exhibit a **dual representation**: on one hand, a parametric construction sequence (comprising operations such as Line and Extrude along with their numerical parameters) that ensures manufacturing precision and editability; on the other hand, a visual geometric shape rendered from that sequence for intuitive inspection and simulation. In practice, geometric shapes are frequently revised (driven by simulation feedback, ergonomic requirements, aesthetic goals, etc.), requiring engineers to simultaneously modify the underlying parametric sequence. This process—**geometry-driven parametric CAD editing**—is both laborious and error-prone: it demands assessing the extent of shape changes, precisely locating the sequence segments to be modified, and propagating those modifications to all dependent segments.

Existing work primarily addresses unconditional editing (randomly sampling edits from the original sequence) or text-driven editing (using natural language instructions to guide edits). The former lacks explicit guidance, while the latter struggles to precisely describe complex shape changes in concise text. The practically important scenario of **using a target geometric shape directly as the editing guide** remains largely unexplored.

## Core Problem

Geometry-driven parametric CAD editing presents three core challenges:

1. **Structure Preservation**: Edits should be confined to the segments that require modification, leaving the remainder unchanged.
2. **Semantic Validity**: The updated parametric sequence must not only be syntactically correct but must also produce a CAD model that conforms to design conventions (e.g., bolt holes distributed uniformly rather than arbitrarily).
3. **Shape Fidelity**: The shape rendered from the updated sequence must match the target shape.

An additional challenge is **data scarcity**: no dataset exists that simultaneously provides original sequences, target geometric shapes, and corresponding updated sequences as triplets.

## Method

### Task Formulation

Given an original parametric sequence $C$ and a target geometric shape $S'$, the goal is to find an updated sequence $C'$ whose rendered result reproduces $S'$ while preferring to preserve the structure of $C$:

$$C' = \arg\min_{C'} \mathcal{D}_{\text{geometry}}(\mathcal{F}(C'), S') + \lambda \mathcal{R}_{\text{structure}}(C', C)$$

### Two Pretrained Foundation Models

- **Parameter-to-Shape (P2S) Model**: A Latent Diffusion Model trained on ⟨parametric sequence, SDF⟩ pairs, mapping parametric sequences to the latent representation of 3D shapes. The architecture follows SDFusion, comprising a shape encoder–decoder pair and a diffusion model.
- **Masked-Parameter-Prediction (MPP) Model**: A large language model based on Llama-3 8B, fine-tuned via LoRA with a hierarchical masking strategy to complete masked segments in parametric sequences.

Neither model requires triplet data, circumventing the data scarcity bottleneck.

### Iterative Plan–Generate–Verify Framework

At iteration $r$, three stages are executed:

**1. Planning — Localizing Segments to Modify**

- The cross-attention maps of the P2S model are used to quantify the contribution of each sequence segment to the target shape.
- A relative contribution score is computed: $J(i) = |\mathcal{M}(C'_{r-1}(i), S') - \mathcal{M}(C'_{r-1}(i), S'_{r-1})|$
- Segments are ranked by score, and the top $K$ (those exceeding the mean $\bar{J}$) are masked to produce $C^{\text{mask}}_r$.
- This attention-based masking strategy focuses edits on segments misaligned with the target shape, satisfying the structure preservation requirement.

**2. Generation — Producing Candidate Edits**

- The MPP model completes $C^{\text{mask}}_r$ a total of $N$ times, yielding a candidate set $\{C^1_r, \dots, C^N_r\}$.
- Generation proceeds autoregressively token by token, leveraging the CAD design knowledge acquired during pretraining to ensure semantic validity.

**3. Verification — Selecting the Best Candidate**

- All candidate sequences and the target shape are mapped to a shared latent space via the P2S model.
- The candidate with minimum Euclidean distance to the target shape is selected: $C'_r = \arg\min_{\tilde{C} \in \mathcal{Q}} \|\mathcal{F}(\tilde{C}) - E_s(S')\|_2$
- A **priority queue** $\mathcal{Q}$ is maintained across iterations, retaining the best historical candidates to broaden the search space.

Iterations repeat until convergence or a maximum number of iterations (default: 10).

## Key Experimental Results

Models are trained on the DeepCAD dataset (~130k CAD models) and evaluated on the CAD-Editor 2k test set, with 5 outputs generated per test sample.

| Method | IoU ↑ | mean CD ↓ | median CD ↓ | JSD ↓ | IR (%) ↓ | Edit Dist. ↓ |
|------|-------|-----------|-------------|-------|----------|-------------|
| GPT-4o | 0.247 | 0.107 | 0.0171 | 0.737 | 25.1 | 21.12 |
| o4-mini | 0.185 | 0.118 | 0.0283 | 0.748 | 32.95 | 22.49 |
| CAD-Diffuser | 0.548 | 0.097 | 0.0093 | 0.689 | 5.7 | 17.29 |
| FlexCAD | 0.447 | 0.029 | 0.0065 | 0.634 | 15.3 | 22.29 |
| **CADMorph** | **0.687** | **0.009** | **0.0031** | **0.621** | **3.1** | **16.87** |

Key findings:

- CADMorph outperforms all baselines on every metric; IoU exceeds the strongest baseline (CAD-Diffuser) by 25%.
- VLMs (GPT-4o series) struggle to produce syntactically valid CAD sequences, with Invalid Rates reaching 25–40%.
- In human evaluation, CADMorph achieves an average rank of 1.37 (out of 1), significantly better than all other methods.

Ablation studies confirm the necessity of each component: removing the priority queue drops IoU from 0.687 to 0.619; removing the Verification stage drops it to 0.517; removing the Planning stage drops it to 0.447.

## Highlights & Insights

1. **Data Efficiency**: No triplet training data is required; computational effort is invested in iterative inference-time search, embodying the test-time scaling paradigm.
2. **Search Efficiency**: The Planning stage narrows the search space via cross-attention analysis, while the Verification stage provides an effective selection signal to guide the editing direction.
3. **Implicit Error Correction**: Because the MPP model has absorbed extensive design knowledge during pretraining, it can automatically correct geometrically implausible configurations (e.g., table legs not flush with the tabletop).
4. **Practical Downstream Applications**: The framework supports iterative editing (multiple successive modifications) and reverse-engineering enhancement (refining the output of reverse-engineering pipelines).
5. **Novel Use of Cross-Attention Maps**: Drawing an analogy to word–pixel correspondence in text-to-image diffusion models, the work discovers attention alignment between parametric sequence segments and geometric parts in the P2S model.

## Limitations & Future Work

1. **Inference Latency**: Multiple plan–generate–verify iterations incur considerable runtime; this can be mitigated by accelerating model inference and parallelizing the generation and verification stages.
2. **Test Set Limitations**: Evaluation is conducted solely on the CAD-Editor test set, whose model complexity falls below industrial standards; richer and more challenging benchmarks are needed.
3. **Representation Constraints**: Shapes are represented as voxelized tSDF, whose limited resolution may hinder the capture of fine geometric details.
4. **End-to-End Potential**: The current approach is an inference-time iterative search; future work could use CADMorph-generated triplets to train an end-to-end model.

## Related Work & Insights

- **vs. Reverse-Engineering Methods (CAD-Diffuser)**: Reverse engineering directly reconstructs parametric sequences from geometry, discarding the designer's intent encoded in the original sequence; CADMorph preserves the original sequence structure, making only minimal edits.
- **vs. Traditional Editing Methods (FlexCAD)**: Traditional editing methods lack visual guidance and cannot align the editing direction with a target shape; CADMorph uses the target shape as an explicit guide.
- **vs. VLMs (GPT-4o)**: General-purpose vision–language models lack CAD domain knowledge, producing sequences with high syntactic error rates and poor shape quality.
- **vs. 3D Shape Editing Methods**: Operating directly on mesh/SDF/NeRF geometry discards parametric editability; CADMorph edits at the parametric sequence level, preserving manufacturing feasibility.

**Broader Implications:**

- **First Application of Test-Time Scaling to CAD**: Improving performance by increasing inference-time computation (multiple generations + verifier selection) is a paradigm transferable to other structured generation tasks.
- **Cross-Attention as a Localization Tool**: Using diffusion model cross-attention maps to localize regions requiring modification can be transferred to controllable editing in other conditional generation tasks.
- **Dual-Model Collaborative Framework**: One model handles perception and evaluation (P2S) while the other handles generation (MPP); this division-of-labor design pattern merits adoption in other multimodal generation research.

## Rating
- **Novelty**: 8/10 — First formal treatment of geometry-driven parametric CAD editing; the plan–generate–verify framework is elegantly designed.
- **Experimental Thoroughness**: 7/10 — Baselines span VLMs, reverse-engineering, and editing methods; human evaluation and ablation studies are included; however, the test set is small and of limited complexity.
- **Writing Quality**: 8/10 — Problem definition is clear; the framework stages are explained with logical fluency; figures and tables are intuitive.
- **Value**: 7/10 — Addresses genuine engineering needs, though inference efficiency and test set limitations somewhat constrain practical applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] MamTiff-CAD: Multi-Scale Latent Diffusion with Mamba+ for Complex Parametric Sequence](../../ICCV2025/image_generation/mamtiff-cad_multi-scale_latent_diffusion_with_mamba_for_complex_parametric_seque.md)
- [\[ICLR 2026\] Seek-CAD: A Self-Refined Generative Modeling for 3D Parametric CAD Using Local Inference via DeepSeek](../../ICLR2026/image_generation/seek-cad_a_self-refined_generative_modeling_for_3d_parametric_cad_using_local_in.md)
- [\[NeurIPS 2025\] Diffusion-Classifier Synergy: Reward-Aligned Learning via Mutual Boosting Loop for FSCIL](diffusion-classifier_synergy_reward-aligned_learning_via_mutual_boosting_loop_fo.md)
- [\[CVPR 2026\] Learning to Generate via Understanding: Understanding-Driven Intrinsic Rewarding for Unified Multimodal Models](../../CVPR2026/image_generation/learning_to_generate_via_understanding_understanding-driven_intrinsic_rewarding_.md)
- [\[NeurIPS 2025\] Predictive Feature Caching for Training-free Acceleration of Molecular Geometry Generation](predictive_feature_caching_for_training-free_acceleration_of_molecular_geometry_.md)

</div>

<!-- RELATED:END -->

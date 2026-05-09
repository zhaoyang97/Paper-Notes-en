---
title: >-
  [Paper Note] Seek-CAD: A Self-Refined Generative Modeling for 3D Parametric CAD Using Local Inference via DeepSeek
description: >-
  [ICLR 2026][Image Generation][CAD parametric modeling] This paper proposes Seek-CAD, the first training-free CAD parametric model generation framework based on a locally deployed reasoning LLM (DeepSeek-R1). It achieves self-refinement through the synergy of step-wise visual feedback and Chain-of-Thought (CoT), and introduces a novel SSR triplet design paradigm to support complex CAD model generation.
tags:
  - ICLR 2026
  - Image Generation
  - CAD parametric modeling
  - DeepSeek-R1
  - training-free
  - Chain-of-Thought
  - self-refinement
  - SSR design paradigm
date: 2026-05-08
content_hash: cb4a4eb6ecd526d3
---

# Seek-CAD: A Self-Refined Generative Modeling for 3D Parametric CAD Using Local Inference via DeepSeek

**Conference**: ICLR 2026
**arXiv**: [2505.17702](https://arxiv.org/abs/2505.17702)
**Code**: [https://github.com/Sunny-Hack/Seek-CAD](https://github.com/Sunny-Hack/Seek-CAD)
**Area**: CAD Generation / LLM Reasoning
**Keywords**: CAD parametric modeling, DeepSeek-R1, training-free, Chain-of-Thought, self-refinement, SSR design paradigm

## TL;DR

This paper proposes Seek-CAD, the first training-free CAD parametric model generation framework based on a locally deployed reasoning LLM (DeepSeek-R1). It achieves self-refinement through the synergy of step-wise visual feedback and Chain-of-Thought (CoT), and introduces a novel SSR triplet design paradigm to support complex CAD model generation.

## Background & Motivation

Automated generation of parametric CAD models is critical for industrial manufacturing automation. Existing approaches fall into two categories:

**Fine-tuning methods** (e.g., CAD-Llama): require substantial computational resources

**Training-free methods** (e.g., 3D-PreMise, CADCodeVerify): rely on GPT-4 but lack mechanisms to leverage Chain-of-Thought (CoT)

Furthermore, existing datasets are primarily built upon the simple SE (Sketch-Extrude) paradigm, supporting only basic operations such as sketching and extrusion, which is insufficient for generating complex CAD models meeting industrial requirements (e.g., features such as chamfers, fillets, and thin shells).

## Method

### Overall Architecture

Seek-CAD consists of two major components:
1. **Initial CAD code generation**: RAG augmentation + knowledge constraints + DeepSeek-R1 inference
2. **CAD code refinement**: step-wise visual feedback + CoT-aligned evaluation + iterative correction

### Key Design 1: Local Inference Pipeline

- **Knowledge constraints** $Cons = (\Phi, \mathcal{D}, \mathcal{E})$: system prompts constrain DeepSeek-R1 to generate according to the SSR paradigm
- **RAG**: retrieval over a local corpus of 10,000 CAD models using hybrid vector search and full-text search

$$g_i^{\text{final}} = \lambda \cdot g_i^{\text{vec}} + (1-\lambda) \cdot g_i^{\text{full}}, \quad \lambda = 0.3$$

The Top-3 candidates are concatenated with the input to trigger initial code generation.

### Key Design 2: Step-wise Visual Feedback (SVF)

**Core innovation**: Rather than presenting only the final CAD shape, intermediate visual states of the entire construction process are also preserved:

Intermediate shape images: $M_I = [R(S_1), R(\bar{S_1} \oplus S_2), \cdots, R(\bar{S_1} \oplus \cdots \oplus S_n)]$

Final shape image: $M_U = R(S_1 \oplus S_2 \oplus \cdots \oplus S_n)$

Gemini-2.0 is used to assess whether the step-wise images are aligned with the CoT produced by DeepSeek-R1:

$$F_{\text{call}} \sim P(F_{\text{call}} | G, M, CoT)$$

If misalignment is detected, concrete feedback is generated and fed back to DeepSeek-R1 for code refinement.

### Key Design 3: SSR Triplet Design Paradigm

$$S = (s, f, \langle r_1, r_2, \dots, r_k \rangle \text{ or } \varnothing)$$

- $s$: 2D sketch
- $f \in \mathcal{F}$: sketch-based feature (extrusion, revolution, etc.)
- $\langle r_1, \dots, r_k \rangle$: refinement feature sequence (chamfer, fillet, thin shell, etc.)

A complete CAD model is assembled by combining multiple SSR triplets via Boolean operations:

$$\mathcal{M} = \langle \mathcal{S}_1, \text{op}_1, \mathcal{S}_2, \text{op}_2, \dots, \mathcal{S}_n \rangle$$

**CapType reference mechanism**: Topological primitives are tracked via three types — START, END, and SWEPT.

## Experiments

### Generation Quality (500 CAD Models)

| Strategy | Method | CD↓ | HD↓ | IoGT↑ | G-Score↑ | Novel↑ |
|------|------|-----|-----|-------|---------|--------|
| Fine-tuning | CAD-Llama | 0.2147 | 0.5864 | 0.7023 | 3.3385 | 77.64% |
| Training-free | 3D-PreMise | 0.2203 | 0.6137 | 0.6315 | 3.2022 | 49.57% |
| Training-free | CADCodeVerify | 0.2164 | 0.5917 | 0.6562 | 3.3927 | 55.38% |
| Training-free | **Seek-CAD** | **0.1979** | **0.5566** | **0.7226** | **3.5185** | 64.04% |

### Ablation on Refinement Rounds

| Rounds | Pass@2↑ | CD↓ | IoGT↑ | G-Score↑ |
|------|---------|-----|-------|---------|
| 0 | 0.77 | 0.2275 | 0.6183 | 3.1401 |
| 1 | 0.72 | 0.1979 | 0.7226 | 3.5185 |
| 2 | 0.55 | 0.1966 | 0.7347 | 3.5314 |

One refinement round yields significant improvement; a second round offers diminishing returns and increases compilation failure rates.

### Ablation Study

- Removing the local CAD corpus → complete failure to generate compilable code
- Removing knowledge constraints → Pass@1 drops from 0.68 to 0.44
- Removing CoT from SVF → degraded feedback quality
- Removing intermediate images → incomplete feedback information

### Key Findings

- CoT effectively captures design logic, enabling the VLM to better understand the construction process
- The SSR paradigm supports more diverse and complex CAD models (including chamfers, fillets, and thin shells)
- The training-free framework is competitive with fine-tuning methods (e.g., CAD-Llama) in geometric accuracy
- Hybrid search outperforms single-modality search in RAG

## Highlights & Insights

- The first work to explore locally deployed reasoning LLMs (DeepSeek-R1) for CAD generation
- The refinement strategy combining step-wise visual feedback with CoT alignment is novel in design
- The SSR triplet paradigm substantially expands the range of supportable CAD operations
- Entirely training-free and operable on a single RTX 3090 GPU

## Limitations & Future Work

- Geometric accuracy for complex models is constrained by the inference capacity of DeepSeek-R1:32B-Q4
- Each refinement round carries a risk of compilation failure, limiting the number of feasible iterations
- The CapType mechanism covers only three reference types: START, END, and SWEPT
- Reliance on the Gemini-2.0 API for visual evaluation introduces an external dependency
- The dataset contains only 40K samples, leaving CAD operation coverage incomplete

## Related Work & Insights

- **CAD generation**: sequence-based methods such as DeepCAD, SkexGen, and Mamba-CAD
- **LLM for CAD**: Text2CAD, CAD-MLLM, and CAD-assistant
- **Training-free methods**: 3D-PreMise and CADCodeVerify using GPT-4

## Rating

- Novelty: ⭐⭐⭐⭐ — First to leverage a reasoning LLM with CoT feedback for CAD generation
- Practicality: ⭐⭐⭐⭐ — Training-free and locally deployable, with a low barrier to adoption
- Data contribution: ⭐⭐⭐⭐ — The SSR paradigm and the 40K dataset represent meaningful contributions
- Experimental thoroughness: ⭐⭐⭐ — The 500-model test set is of moderate scale; ablations are reasonably comprehensive

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] MamTiff-CAD: Multi-Scale Latent Diffusion with Mamba+ for Complex Parametric Sequence](../../ICCV2025/image_generation/mamtiff-cad_multi-scale_latent_diffusion_with_mamba_for_complex_parametric_seque.md)
- [\[NeurIPS 2025\] CADMorph: Geometry-Driven Parametric CAD Editing via a Plan-Generate-Verify Loop](../../NeurIPS2025/image_generation/cadmorph_geometry-driven_parametric_cad_editing_via_a_plan-generate-verify_loop.md)
- [\[ICLR 2026\] GenCP: Towards Generative Modeling Paradigm of Coupled Physics](gencp_towards_generative_modeling_paradigm_of_coupled_physics.md)
- [\[ICLR 2026\] Laplacian Multi-scale Flow Matching for Generative Modeling](laplacian_multi-scale_flow_matching_for_generative_modeling.md)
- [\[AAAI 2026\] CAD-VAE: Leveraging Correlation-Aware Latents for Comprehensive Fair Disentanglement](../../AAAI2026/image_generation/cad-vae_leveraging_correlation-aware_latents_for_comprehensive_fair_disentanglem.md)

</div>

<!-- RELATED:END -->

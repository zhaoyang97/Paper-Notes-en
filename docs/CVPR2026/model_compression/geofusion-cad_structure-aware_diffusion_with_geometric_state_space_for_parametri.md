---
title: >-
  [Paper Note] GeoFusion-CAD: Structure-Aware Diffusion with Geometric State Space for Parametric 3D Design
description: >-
  [CVPR 2026][Model Compression][CAD generation] This paper proposes GeoFusion-CAD, an end-to-end diffusion framework that encodes CAD programs as hierarchical tree structures and introduces a geometry-aware G-Mamba block with linear time complexity to replace quadratic-complexity Transformers, enabling scalable and structure-aware generation of long-sequence parametric CAD programs. The method substantially outperforms Transformer-based approaches on the newly constructed DeepCAD-240 benchmark (up to 240-step commands).
tags:
  - CVPR 2026
  - Model Compression
  - CAD generation
  - diffusion model
  - state space model
  - Mamba
  - hierarchical tree representation
date: 2026-05-08
content_hash: 5feb4c060126d531
---

# GeoFusion-CAD: Structure-Aware Diffusion with Geometric State Space for Parametric 3D Design

**Conference**: CVPR 2026
**arXiv**: [2603.21978](https://arxiv.org/abs/2603.21978)
**Code**: [https://github.com/](https://github.com/) (to be released)
**Area**: Model Compression
**Keywords**: CAD generation, diffusion model, state space model, Mamba, hierarchical tree representation

## TL;DR

This paper proposes GeoFusion-CAD, an end-to-end diffusion framework that encodes CAD programs as hierarchical tree structures and introduces a geometry-aware G-Mamba block with linear time complexity to replace quadratic-complexity Transformers, enabling scalable and structure-aware generation of long-sequence parametric CAD programs. The method substantially outperforms Transformer-based approaches on the newly constructed DeepCAD-240 benchmark (up to 240-step commands).

## Background & Motivation

1. **Background**: Parametric CAD modeling is fundamental to modern 3D design. Prevailing methods treat CAD programs as structured languages and employ Transformer encoder-decoders for autoregressive generation. The Sketch-Extrusion (SE) paradigm constructs 3D solids by sequentially generating 2D sketches and extrusion operations, preserving parametric constraints and design intent.
2. **Limitations of Prior Work**: Transformer architectures face two core issues: (a) **Quadratic complexity** — $\mathcal{O}(L^2d)$ self-attention becomes prohibitive as CAD programs scale to hundreds of commands; most approaches resort to splitting long sequences into short segments or training in stages within latent space, breaking end-to-end optimization; (b) **Lack of hierarchy awareness** — global attention treats all tokens equally, ignoring the hierarchical organization of CAD data (strict topological dependencies among sketches, faces, edges, and vertices), thereby diluting local geometric relationships.
3. **Key Challenge**: CAD programs are inherently hierarchical and structured — local features must remain consistent within a global design context. Uniform global attention cannot balance global context modeling with local structural fidelity, causing geometric reasoning discontinuities and reduced consistency in long structured CAD sequences.
4. **Goal**: (a) How to model long-range dependencies in CAD sequences with linear time complexity? (b) How to maintain geometric and topological consistency during generation? (c) How to handle CAD programs of up to 240 steps end-to-end?
5. **Key Insight**: State space models (e.g., Mamba) achieve linear time complexity, but their rigid sequential scanning limits the modeling of hierarchical topological dependencies. The authors observe that CAD data exhibits a natural tree-like hierarchical structure, which can be exploited to inject geometric inductive biases into state space transitions.
6. **Core Idea**: Design geometrically conditioned state space transitions (G-Mamba) that embed the hierarchical tree structure of CAD into a diffusion denoising process driven by selective state transitions, achieving structure-aware CAD generation at $\mathcal{O}(Ld)$ complexity.

## Method

### Overall Architecture

GeoFusion-CAD represents CAD programs as hierarchical trees, where the root node corresponds to the overall solid model and child nodes represent sketch and extrusion operations, organized into three levels from top to bottom — operations/faces, edges/extrusion depths, and vertices. Input CAD sequences are passed through geometric embeddings and then processed by a G-Mamba diffusion encoder for hierarchical feature denoising, followed by a CAD decoder (command layer + parameter layer) to reconstruct the parametric solid. The entire pipeline is trained end-to-end without staged optimization.

### Key Designs

1. **Hierarchical Tree Representation**:

    - **Function**: Encodes CAD geometry, parameters, and topological dependencies into a unified tree structure.
    - **Mechanism**: Sketch nodes are encoded as feature vectors containing geometric and positional parameters. Each sketch consists of faces bounded by loops, which are either single primitives (circles) or chains of multiple curves (lines, arcs). 2D coordinates $(p_x, p_y)$ are discretized, and termination symbols $e_c, e_l, e_f, e_s$ mark the boundaries of curves, loops, faces, and sketches. Extrusion parameters include orientation angles $(\theta, \phi, \gamma)$, displacement $(\tau_x, \tau_y, \tau_z)$, scale $\sigma$, extrusion distances $(d_+, d_-)$, and operation type $\beta$. All parameters are discretized into token sequences.
    - **Design Motivation**: Unlike prior methods that duplicate nodes to represent shared edges, the proposed tree structure preserves connectivity without repetition and maintains the full program design history. This unified structure reconciles the SE representation to make it amenable to diffusion-based generation.

2. **G-Mamba Diffusion Encoder**:

    - **Function**: Models long-range and hierarchical dependencies of complex CAD structures with linear time complexity.
    - **Mechanism**: The core component is the G-Mamba block, which embeds a Geometric State Mixer (GSM) within a Selective State Space (SSD) layer to form the GSM-SSD module. Input features first pass through a Depthwise Convolution (DWC) to preserve local geometric smoothness, followed by geometrically conditioned state transitions. The geometric conditioning vector $\Delta_k = g(s_k, d_k, r_k)$ encodes local geometric scale $s_k$, hierarchical depth in the CAD tree $d_k$, and local curvature descriptor $r_k$. The hierarchical positional embedding $\Pi_k = \text{PE}(p_k, \sigma_k, \tau_k)$ encodes parent type, sibling index, and topological role. The state transition is $h_{k+1} = \bar{A}_k h_k + \bar{B}_k Z_k^c$, $Z_{k+1}^c = C_k h_k + G_k Z_k^c$, where the transition kernels $\{\bar{A}_k, \bar{B}_k, C_k, G_k\}$ are conditioned on geometric and hierarchical context.
    - **Design Motivation**: The globally shared state space transition matrices of vanilla Mamba are insensitive to the heterogeneous geometry and hierarchical patterns of CAD data. By injecting geometric conditioning and hierarchical positional embeddings, G-Mamba acquires inductive biases aligned with CAD's multi-level topology, maintaining $\mathcal{O}(Ld)$ complexity while achieving structure awareness.

3. **DeepCAD-240 Extended Benchmark**:

    - **Function**: Provides an evaluation standard for long-sequence CAD generation.
    - **Mechanism**: Built upon the original DeepCAD dataset, extending the maximum command length from 60 to 240 while preserving Sketch-Extrusion semantics and tokenization protocols. It introduces richer hierarchical dependencies and longer geometric contexts.
    - **Design Motivation**: Existing benchmarks (DeepCAD, max 60 steps) cannot assess long-sequence generation capability, whereas practical engineering CAD programs routinely exceed 60 steps. DeepCAD-240 provides a more challenging evaluation scenario.

### Loss & Training

Joint training objective:
$$\mathcal{L}_{total} = \underbrace{E_{t,Z_0,\epsilon_t}[\|\hat{\epsilon}_t - \epsilon_\theta(\cdot)\|^2]}_{\text{diffusion noise prediction}} + \underbrace{\sum_{i=1}^N\left[CCE(\hat{c}_i, c_i) + \eta \sum_{j=1}^M ACE(\hat{a}_{i,j}, a_{i,j})\right]}_{\text{command and parameter supervision}}$$

The first term ensures accurate denoising of latent geometric features; the second term supervises program generation correctness at both the command and parameter levels. The coefficient $\eta$ balances parameter supervision relative to command prediction.

## Key Experimental Results

### Main Results

**DeepCAD short-sequence test (< 60 commands):**

| Method | ACC_cmd | ACC_param | COV↑ | MMD↓ | JSD↓ |
|--------|---------|-----------|------|------|------|
| DeepCAD | 92.4 | 89.2 | 78.1 | 1.72 | 3.98 |
| HNC-CAD | 95.4 | 93.8 | 82.3 | 1.33 | 3.24 |
| **GeoFusion-CAD** | **99.3** | **97.6** | **85.6** | **0.95** | **2.51** |

**DeepCAD-240 long-sequence test (40–240 commands):**

| Method | ACC_cmd | ACC_param | COV↑ | MMD↓ | JSD↓ | Memory | FLOPs |
|--------|---------|-----------|------|------|------|--------|-------|
| DeepCAD | 75.2 | 72.5 | 64.5 | 1.85 | 4.09 | 8197MiB | 52.8G |
| HNC-CAD | 82.8 | 78.5 | 71.2 | 1.71 | 3.81 | 10342MiB | 87.3G |
| **GeoFusion-CAD** | **91.2** | **89.3** | **73.9** | **1.12** | **2.97** | **5198MiB** | **34.6G** |

GeoFusion-CAD surpasses HNC-CAD on long sequences by 8.4 percentage points in command accuracy, while halving memory consumption and reducing FLOPs by 60%.

### Ablation Study

| Configuration | ACC_cmd | ACC_param | COV↑ | MMD↓ | JSD↓ | Note |
|---------------|---------|-----------|------|------|------|------|
| Full model | 91.2 | 89.3 | 73.9 | 1.12 | 2.97 | Complete model |
| w/o Tree | 87.5 | 84.6 | 69.4 | 1.46 | 3.25 | Remove hierarchical encoding, −3.7 cmd |
| MLP substitute | 75.3 | 72.1 | 67.8 | 1.73 | 3.81 | Severe performance drop |
| Transformer substitute | 82.6 | 81.3 | 69.1 | 1.55 | 3.67 | Quadratic complexity with lower accuracy |
| Vanilla Mamba substitute | 89.2 | 87.6 | — | — | — | No geometric conditioning, slightly worse |

### Key Findings

- **Hierarchical tree representation is critical**: Removing it drops ACC_cmd from 91.2 to 87.5 and COV from 73.9 to 69.4, demonstrating that the hierarchical structure is indispensable for maintaining long-range geometric dependencies and topological consistency.
- **G-Mamba outperforms both Transformer and vanilla Mamba**: The Transformer substitute raises MMD from 1.12 to 1.55 and JSD from 2.97 to 3.67, indicating that geometric state space diffusion better stabilizes long-sequence modeling.
- **Substantial computational efficiency gains**: GeoFusion-CAD requires only 5198MiB memory and 34.6G FLOPs, representing 50% and 40% of HNC-CAD's costs respectively, achieving a favorable efficiency–accuracy trade-off.
- Transformer-based methods degrade markedly on long sequences (DeepCAD ACC_cmd drops from 92.4 on short to 75.2 on long sequences), whereas GeoFusion-CAD exhibits substantially smaller degradation.

## Highlights & Insights

- **Seamless integration of SSM linear efficiency with CAD hierarchical structure**: G-Mamba does not simply replace Transformers with Mamba; instead, it injects the tree-structured topology of CAD into state space dynamics via geometrically conditioned transition kernels. This design preserves $\mathcal{O}(Ld)$ complexity while acquiring hierarchy-awareness.
- **End-to-end single-stage training**: Prior methods require separate training of command and parameter streams; GeoFusion-CAD unifies both within a single diffusion framework, avoiding feature misalignment and information loss caused by staged training.
- **Contribution of the DeepCAD-240 benchmark**: Provides a standardized evaluation protocol for long-sequence CAD generation research, extending the maximum command length from 60 to 240, better reflecting practical engineering requirements.
- The design of geometric conditioning vectors (scale + depth + curvature) and hierarchical positional embeddings (parent type + sibling index + topological role) is transferable to other sequence modeling tasks with inherent hierarchical structures.

## Limitations & Future Work

- **Restricted to the Sketch-Extrusion paradigm**: Other CAD modeling paradigms such as B-Rep (Boundary Representation) are not supported, limiting geometric expressiveness.
- **Representativeness of the DeepCAD-240 dataset**: Derived from the ABC dataset, it may not fully capture the complexity of industrial-grade CAD designs.
- **Conditional generation not addressed**: The current framework supports only unconditional generation; text-driven or image-conditioned CAD generation is absent.
- **Minor boundary irregularities remain in visualized results**: Although the method outperforms baselines overall, fine-grained geometry of complex surfaces still has room for improvement.
- Future work may explore extending G-Mamba to conditional CAD editing and reverse engineering tasks.

## Related Work & Insights

- **vs. DeepCAD**: DeepCAD employs a Transformer encoder-decoder and suffers severe performance degradation on long sequences (ACC_cmd drops from 92.4 to 75.2). GeoFusion-CAD maintains a command accuracy of 91.2 on long sequences with lower memory overhead.
- **vs. HNC-CAD**: HNC-CAD is the strongest Transformer baseline, achieving ACC_cmd of 95.4 on short sequences. GeoFusion-CAD reaches 99.3 (+3.9) on short sequences and 91.2 vs. 82.8 (+8.4) on long sequences; the advantage grows with sequence length.
- **vs. BrepGen**: BrepGen applies diffusion to B-Rep generation but still suffers from long-range consistency issues. GeoFusion-CAD addresses this through the hierarchical tree representation and G-Mamba.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The geometrically conditioned design of G-Mamba is innovative, though the hierarchical tree representation and diffusion-based generation are not entirely novel in isolation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive comparisons on both short and long sequences, complete ablations, and a newly constructed 240-step benchmark; conditional generation evaluation is absent.
- **Writing Quality**: ⭐⭐⭐⭐ Structure is clear, though some technical details deferred to supplementary material reduce readability.
- **Value**: ⭐⭐⭐⭐ Provides an efficient and scalable solution for long-sequence CAD generation; the G-Mamba design offers useful reference for the broader community.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] PlanaReLoc: Camera Relocalization in 3D Planar Primitives via Region-Based Structure Matching](planareloc_camera_relocalization_in_3d_planar_primitives_via_region-based_struct.md)
- [\[ACL 2026\] IMPACT: Importance-Aware Activation Space Reconstruction](../../ACL2026/model_compression/impact_importance-aware_activation_space_reconstruction.md)
- [\[CVPR 2026\] DualReg: Dual-Space Filtering and Reinforcement for Rigid Registration](dualreg_dual-space_filtering_and_reinforcement_for_rigid_registration.md)
- [\[NeurIPS 2025\] Hankel Singular Value Regularization for Highly Compressible State Space Models](../../NeurIPS2025/model_compression/hankel_singular_value_regularization_for_highly_compressible_state_space_models.md)
- [\[CVPR 2026\] Adversarial Concept Distillation for One-Step Diffusion Personalization](adversarial_concept_distillation_for_one-step_diffusion_personalization.md)

<!-- RELATED:END -->

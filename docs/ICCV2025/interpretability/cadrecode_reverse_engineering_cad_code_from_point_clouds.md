---
title: >-
  [Paper Note] CAD-Recode: Reverse Engineering CAD Code from Point Clouds
description: >-
  [ICCV 2025][CAD reverse engineering] CAD-Recode frames 3D CAD reverse engineering as a point-cloud-to-Python-code translation task. It leverages the Python code understanding capabilities of pretrained LLMs as the decoder, combined with a lightweight point cloud projector and a million-scale procedurally generated dataset, achieving significant improvements over existing methods on multiple CAD benchmarks while enabling LLM-driven CAD editing and question answering.
tags:
  - ICCV 2025
  - CAD reverse engineering
  - point cloud
  - Python code generation
  - LLM decoder
  - sketch-extrude
date: 2026-05-08
content_hash: 94249adbd3c35bef
---

# CAD-Recode: Reverse Engineering CAD Code from Point Clouds

**Conference**: ICCV 2025
**arXiv**: [2412.14042](https://arxiv.org/abs/2412.14042)
**Code**: Available
**Area**: Interpretability
**Keywords**: CAD reverse engineering, point cloud, Python code generation, LLM decoder, sketch-extrude

## TL;DR
CAD-Recode frames 3D CAD reverse engineering as a point-cloud-to-Python-code translation task. It leverages the Python code understanding capabilities of pretrained LLMs as the decoder, combined with a lightweight point cloud projector and a million-scale procedurally generated dataset, achieving significant improvements over existing methods on multiple CAD benchmarks while enabling LLM-driven CAD editing and question answering.

## Background & Motivation

**Background**: CAD reverse engineering aims to reconstruct parametric sketch-extrude operation sequences from 3D representations such as point clouds. Existing methods typically represent CAD sequences as discrete token sequences and predict them with autoregressive models.

**Limitations of Prior Work**: (1) Existing CAD sequence representations (e.g., command sequences) are unfriendly to pretrained models and require training from scratch; (2) generated sequences are not interpretable and difficult to edit downstream; (3) training data is limited (DeepCAD contains only ~18K samples).

**Key Challenge**: CAD sequences are inherently structured programs, yet existing methods treat them as arbitrary token sequences, failing to exploit the prior knowledge LLMs have acquired over structured code.

**Goal**: Represent CAD sequences as executable Python code and leverage LLMs' code understanding capabilities for improved point-cloud-to-CAD-code translation.

**Key Insight**: Pretrained LLMs have been extensively exposed to Python code; representing CAD operations in Python naturally exploits this prior.

**Core Idea**: Use Python code as a unified representation of CAD sequences, and employ a pretrained LLM decoder with a lightweight point cloud projector for end-to-end point-cloud-to-CAD-code translation.

## Method

### Overall Architecture
Input: 3D point cloud. Output: Executable Python code (which reconstructs the CAD model upon execution). Architecture: point cloud encoder → lightweight projector → pretrained LLM decoder → Python code.

### Key Designs

1. **Python Code Representation**:

    - Function: Represent CAD sketch-extrude sequences as Python functions.
    - Mechanism: Each sketch-extrude operation is encoded as a Python function call containing sketch vertex coordinates, extrusion direction, and distance parameters. The complete CAD model is composed of a series of such function calls. The generated code can be directly executed by a Python CAD library to reconstruct the 3D model.
    - Design Motivation: Python code is a "native language" for LLMs, enabling exploitation of their code understanding capabilities; the code output is directly interpretable and editable by both humans and other LLMs.

2. **LLM Decoder + Point Cloud Projector**:

    - Function: Inject point cloud information into the LLM for code generation.
    - Mechanism: A compact pretrained LLM (e.g., CodeLlama) serves as the decoder; a lightweight point cloud projector is prepended to map point cloud embeddings into the LLM's input space. Only the projector and LoRA adapters are trained.
    - Design Motivation: Directly leverages the LLM's code generation prior, avoiding training from scratch.

3. **Million-Scale Procedurally Generated Dataset**:

    - Function: Provide large-scale training data.
    - Mechanism: One million CAD sequences are procedurally generated along with their corresponding Python code and point clouds. Diversity in CAD patterns is ensured by controlling sketch complexity and extrusion parameter ranges.
    - Design Motivation: Existing CAD datasets are small (DeepCAD: 18K); million-scale data enables thorough model training.

### Loss & Training
Standard autoregressive cross-entropy loss (next-token prediction) computed over Python code tokens.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours | Prev. SOTA | Notes |
|--------|------|------|----------|------|
| DeepCAD | Coverage↑ | Significant gain | — | Surpasses all methods |
| Fusion360 | Reconstruction quality | Best | — | Real industrial CAD |
| CC3D | Reconstruction quality | Best | — | Real-world CAD |

### Ablation Study

| Configuration | Performance | Notes |
|------|------|------|
| Randomly initialized LLM | Poor | LLM pretraining is critical |
| Without procedural data pretraining | Moderate | Large-scale data yields significant improvement |
| Full model | Best | LLM prior + large data + Python representation |

### Key Findings
- Python code representation is better suited for LLM decoding than conventional token sequences.
- The Python code prior from pretrained LLMs is essential for CAD generation.
- The output code can be directly edited and queried by off-the-shelf LLMs, enabling interactive CAD editing.

## Highlights & Insights
- **Python code as CAD representation is an inspired idea**: framing geometric reconstruction as code generation naturally exploits LLMs' vast code training data. This representation-conversion strategy is transferable to other structured generation tasks.
- **Interpretability advantage**: the generated Python code is human-readable and supports CAD model editing via natural language interaction with LLMs.
- **Million-scale procedural dataset**: directly addresses the data bottleneck in CAD reverse engineering.

## Limitations & Future Work
- Currently limited to sketch-extrude operations; more complex CAD operations (e.g., fillet, chamfer) are not covered.
- May struggle with highly complex industrial CAD models containing hundreds of features.
- The LLM decoder increases inference cost.

## Related Work & Insights
- **vs. DeepCAD**: uses a custom token representation and a dedicated Transformer. CAD-Recode employs Python code to leverage LLM priors, achieving greater efficiency and interpretability.
- **vs. Point-E/Shape-E**: general 3D generation methods that do not produce editable CAD parameters. CAD-Recode outputs precise parametric representations.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of Python code representation and LLM decoding is highly inspiring.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on three datasets including real-world data.
- Writing Quality: ⭐⭐⭐⭐ Clear exposition and concise methodology.
- Value: ⭐⭐⭐⭐⭐ Significant contributions to both CAD reverse engineering and LLM-for-3D research.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Granular Concept Circuits: Toward a Fine-Grained Circuit Discovery for Concept Representations](granular_concept_circuits_toward_a_fine-grained_circuit_discovery_for_concept_re.md)
- [\[ICCV 2025\] "Principal Components" Enable A New Language of Images](principal_components_enable_a_new_language_of_images.md)
- [\[ICCV 2025\] SVIP: Semantically Contextualized Visual Patches for Zero-Shot Learning](svip_semantically_contextualized_visual_patches_for_zero-shot_learning.md)
- [\[ICCV 2025\] ArgoTweak: Towards Self-Updating HD Maps through Structured Priors](argotweak_towards_self-updating_hd_maps_through_structured_priors.md)
- [\[ICCV 2025\] Learnable Fractional Reaction-Diffusion Dynamics for Under-Display ToF Imaging and Beyond](learnable_fractional_reaction-diffusion_dynamics_for_under-display_tof_imaging_a.md)

<!-- RELATED:END -->

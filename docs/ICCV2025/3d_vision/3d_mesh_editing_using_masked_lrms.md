---
title: >-
  [Paper Note] 3D Mesh Editing using Masked LRMs
description: >-
  [ICCV 2025][3D Vision][LRM] This paper proposes MaskedLRM, which reformulates 3D shape editing as a conditional reconstruction problem. During training, randomly generated 3D occluders mask multi-view inputs…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "LRM"
  - "Masked Reconstruction"
  - "3D Editing"
  - "Conditional Inpainting"
  - "Multi-view Consistency"
date: 2026-05-08
content_hash: b5a46f0bf8dd1795
---

# 3D Mesh Editing using Masked LRMs

**Conference**: ICCV 2025
**arXiv**: [2412.08641](https://arxiv.org/abs/2412.08641)
**Code**: [https://chocolatebiscuit.github.io/MaskedLRM/](https://chocolatebiscuit.github.io/MaskedLRM/)
**Area**: 3D Vision / Shape Editing / Large Reconstruction Models
**Keywords**: LRM, Masked Reconstruction, 3D Editing, Conditional Inpainting, Multi-view Consistency

## TL;DR
This paper proposes MaskedLRM, which reformulates 3D shape editing as a conditional reconstruction problem. During training, randomly generated 3D occluders mask multi-view inputs, and a single clean conditioning view guides completion of the occluded regions. At inference, the user defines an edit region and provides a single edited image; the model produces an edited 3D mesh in a **single forward pass in under 3 seconds** — 2–10× faster than optimization-based methods — while supporting topological changes (e.g., adding holes or handles) and achieving reconstruction quality on par with state-of-the-art methods.

## Background & Motivation
3D shape editing remains far less mature than 2D image editing. Existing approaches fall into two categories: (1) **Optimization-based methods** (TextDeformer, MagicClay): optimize meshes using SDS losses, which suffer from noisy gradients, poor controllability, and slow runtimes (20 minutes to 1 hour), and cannot handle topological changes such as adding holes; (2) **Generative methods** (InstantMesh): first generate edited multi-view images via multi-view diffusion, then reconstruct with an LRM, but multi-view diffusion frequently produces inconsistent artifacts. The root cause is that 2D editing is straightforward while ensuring 3D consistency remains a fundamental challenge.

## Core Problem
How can a single edited image drive 3D mesh editing while simultaneously guaranteeing: (1) faithfulness of the edited region to the 2D edit; (2) precise preservation of the original geometry in unedited regions; (3) multi-view consistency; (4) support for arbitrary topological changes; and (5) fast, feed-forward inference?

## Method

### Overall Architecture
**Training**: Multi-view renderings of the original shape → random 3D occluder masking → occluded patches replaced with learnable mask tokens → conditioning branch receives one clean view → Transformer (self-attention + cross-attention) outputs a triplane → volume rendering reconstructs all views (including occluded regions).

**Inference**: User defines the edit region → 2D inpainting produces an edited image → edited image serves as the condition + masked original multi-view renders as the primary input → single forward pass outputs an edited SDF → Marching Cubes extracts the mesh.

### Key Designs
1. **3D-consistent occlusion training strategy**: Rather than applying random patch masks (MAE-style), a randomly generated 3D cuboid occluder is rendered to produce multi-view-consistent occlusion regions. This eliminates the training–inference mask distribution gap, since user-defined edit regions at inference time are also spatially contiguous 3D regions. Ablation experiments confirm that random patch masking leads to blurry and unrealistic edits.

2. **Conditioning branch design**: The primary branch processes 6–8 masked multi-view images; the conditioning branch processes a single clean edited image. Conditioning is fused into the primary branch via cross-attention. Plücker ray coordinates serve as pose encodings and are added after masking to preserve spatial information.

3. **SDF + volume rendering**: The network outputs a triplane representation, which is decoded into SDF and RGB values. Volume rendering against ground-truth images provides the training signal. Normal map supervision ensures high-quality surface reconstruction.

4. **Two-stage training**: Stage 1 uses 256×256 output resolution with 128 samples per ray (no normal loss); Stage 2 uses 384×384 output resolution with 512 samples per ray and normal loss supervision. Precision is increased progressively.

### Loss & Training
$$\mathcal{L} = w_I\|I - \hat{I}\|_2^2 + w_N\|N - \hat{N}\|_2^2 + w_M\|M - \hat{M}\|_2^2 + w_P\mathcal{L}_{LPIPS}$$
- 64× H100 GPUs; Stage 1 trained for 30 epochs, Stage 2 for 20 epochs.
- Objaverse dataset; 40 renders at 512×512 per shape.
- Random 128×128 crops are used for supervision at each step.

## Key Experimental Results

### Reconstruction Quality (though not the primary goal)

| Method | ABO PSNR↑ | ABO LPIPS↓ | GSO PSNR↑ | GSO LPIPS↓ |
|--------|----------|----------|----------|----------|
| InstantMesh (Mesh) | — | — | 22.79 | 0.120 |
| MeshLRM | 26.09 | 0.102 | 27.93 | 0.081 |
| **MaskedLRM (8 views)** | **28.65** | **0.078** | 27.58 | 0.085 |

MaskedLRM surpasses MeshLRM by 2.56 dB PSNR on ABO and matches state-of-the-art on GSO.

### Editing Speed

| Method | Type | Runtime |
|--------|------|---------|
| TextDeformer | Optimization | 20 min |
| MagicClay | Optimization | 1 hour |
| InstantMesh | LRM | 30 sec |
| PrEditor3D | LRM | 80 sec |
| Instant3Dit | LRM | 6 sec |
| **MaskedLRM** | LRM | **<3 sec** |

### Editing Quality (CLIP Similarity)

| Method | ViT-L-14 | ViT-BigG-14 |
|--------|----------|------------|
| MagicClay | 0.285 | 0.286 |
| Instant3Dit | 0.303 | 0.309 |
| **MaskedLRM** | **0.323** | **0.337** |

### Ablation Study
- **3D occlusion vs. random patch masking**: Random patch masking produces blurry artifacts due to the training–inference distribution gap; 3D occluders yield sharp, clean geometry.
- **Normal supervision**: Without normal supervision, surfaces exhibit holes and bumps; depth supervision provides weaker benefits; normal supervision produces smooth and accurate surfaces.
- **Topological changes**: The method can add handles or holes to objects (genus change); optimization-based methods cannot achieve this due to fixed mesh topology.

## Highlights & Insights
- **Shape editing as conditional reconstruction**: An elegantly principled problem reformulation — editing is recast as "reconstruct the original shape while completing missing regions conditioned on the edit."
- **3D-consistent masking strategy**: Using a 3D occluder to generate multi-view-consistent masks is the core technical contribution, eliminating the training–inference distribution gap.
- **Topological change capability**: Because the output is a new mesh reconstructed from an SDF rather than a deformation of the original mesh, genus changes are naturally supported.
- **Extreme speed**: Feed-forward inference in under 3 seconds, 100–1000× faster than optimization-based methods.
- **Identity preservation**: Reconstruction quality in unedited regions matches state-of-the-art reconstruction methods.

## Limitations & Future Work
- Edit quality is bounded by the quality of the conditioning image, requiring iterative 2D editing to obtain satisfactory results.
- The uniformity of Marching Cubes triangulation limits surface detail fidelity.
- Very fine-grained details (e.g., human faces) may appear blurry.
- Unedited regions cannot be directly frozen (unlike in MagicClay); preservation relies entirely on reconstruction quality.
- Training requires 64× H100 GPUs, imposing high resource demands.

## Related Work & Insights
- **TextDeformer**: Text-guided mesh deformation. SDS gradients are noisy and cause global distortion; runtime is 20 minutes. MaskedLRM uses image conditioning for greater precision in under 3 seconds.
- **MagicClay**: Local SDS optimization. Results are sometimes successful but unpredictable (e.g., failures on fedora/top hat); runtime is 1 hour. MaskedLRM produces highly predictable outputs.
- **InstantMesh**: Single-view → multi-view diffusion → LRM pipeline. Multi-view diffusion introduces inconsistent artifacts. MaskedLRM uses ground-truth renderings as multi-view inputs, bypassing the consistency problem entirely.
- **PrEditor3D / Instant3Dit**: Multi-view diffusion-based editing. Semantically correct but lacking in detail. MaskedLRM's conditioning branch combined with masked training produces more realistic edits.

The paradigm of "conditional reconstruction = editing" is generalizable to other 3D editing tasks (texture editing, scene editing). The 3D-consistent masking training strategy is applicable to other tasks requiring cross-view-consistent inpainting.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — Reformulating editing as conditional reconstruction combined with the 3D-consistent masking strategy is both elegant and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comparisons against 5 methods, reconstruction metrics, editing metrics, ablations, CLIP scores, speed benchmarks, and topological change evaluation.
- Writing Quality: ⭐⭐⭐⭐⭐ — Motivation is clear, method description is systematic, and comparisons are fair and thorough.
- Value: ⭐⭐⭐⭐⭐ — Feed-forward 3D editing in under 3 seconds has high practical utility; topological change support represents a key breakthrough.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] MaskHand: Generative Masked Modeling for Robust Hand Mesh Reconstruction in the Wild](maskhand_generative_masked_modeling_for_robust_hand_mesh_reconstruction_in_the_w.md)
- [\[ICCV 2025\] MeshPad: Interactive Sketch-Conditioned Artist-Reminiscent Mesh Generation and Editing](meshpad_interactive_sketch-conditioned_artist-reminiscent_mesh_generation_and_ed.md)
- [\[ICCV 2025\] Monocular Semantic Scene Completion via Masked Recurrent Networks](monocular_semantic_scene_completion_via_masked_recurrent_networks.md)
- [\[ICCV 2025\] MeshAnything V2: Artist-Created Mesh Generation with Adjacent Mesh Tokenization](meshanything_v2_artist-created_mesh_generation_with_adjacent_mesh_tokenization.md)
- [\[ICCV 2025\] UniVG: A Generalist Diffusion Model for Unified Image Generation and Editing](univg_a_generalist_diffusion_model_for_unified_image_generation_and_editing.md)

</div>

<!-- RELATED:END -->

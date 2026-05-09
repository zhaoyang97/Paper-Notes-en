---
title: >-
  [Paper Note] DeepMesh: Auto-Regressive Artist-Mesh Creation with Reinforcement Learning
description: >-
  [ICCV 2025][3D Vision][3D mesh generation] This paper proposes DeepMesh, a framework that achieves human preference alignment in 3D mesh generation through an improved efficient mesh tokenization algorithm (72% compression rate) and the first application of DPO-based reinforcement learning to 3D mesh generation, capable of producing high-quality artist-like triangle meshes with up to 30K faces.
tags:
  - ICCV 2025
  - 3D Vision
  - 3D mesh generation
  - auto-regressive model
  - reinforcement learning
  - DPO
  - mesh tokenization
  - point cloud conditioned generation
date: 2026-05-08
content_hash: 3c477702bc84541f
---

# DeepMesh: Auto-Regressive Artist-Mesh Creation with Reinforcement Learning

**Conference**: ICCV 2025
**arXiv**: 2503.15265
**Code**: [https://zhaorw02.github.io/DeepMesh/](https://zhaorw02.github.io/DeepMesh/)
**Area**: 3D Vision
**Keywords**: 3D mesh generation, auto-regressive model, reinforcement learning, DPO, mesh tokenization, point cloud conditioned generation

## TL;DR

This paper proposes DeepMesh, a framework that achieves human preference alignment in 3D mesh generation through an improved efficient mesh tokenization algorithm (72% compression rate) and the first application of DPO-based reinforcement learning to 3D mesh generation, capable of producing high-quality artist-like triangle meshes with up to 30K faces.

## Background & Motivation

**Importance of Artist-like Meshes:**

- Triangle meshes are the fundamental representation for 3D assets, widely used in VR, gaming, and animation.
- Artist-crafted meshes feature optimized topology that facilitates editing, deformation, and texture mapping.
- Automated methods such as Marching Cubes yield high geometric accuracy but produce irregular, overly dense topology.

**Two Major Challenges in Auto-Regressive Mesh Generation:**

**Pre-training efficiency**: Existing mesh tokenization methods produce excessively long sequences (increasing computational cost), and low-quality meshes cause training instability (loss spikes).

**Lack of human preference alignment**: Existing methods cannot guarantee that generated results conform to human aesthetic standards, and commonly exhibit geometric defects (holes, missing parts, and redundant structures).

**Limitations of BPT**: Although BPT achieves approximately 74% compression, it is only effective at low resolution (128); at higher resolutions the vocabulary size grows dramatically (40,960), making training difficult.

## Method

### Overall Architecture

DeepMesh = improved tokenization + efficient pre-training strategy + DPO post-training. The core model is an auto-regressive transformer with self-attention and cross-attention layers.

### 1. Improved Tokenization Algorithm

Building upon BPT, the proposed method maintains approximately 72% compression while significantly reducing vocabulary size:

**Core Steps:**
1. **Local Face Traversal**: Mesh faces are partitioned into local patches according to connectivity, minimizing redundancy.
2. **Sorting and Quantization**: Vertex coordinates of each face are sorted, quantized, and flattened in XYZ order.
3. **Three-level Hierarchical Block Indexing**: The coordinate space is divided into three hierarchical levels, with offset indices used to encode quantized coordinates.
4. **Identical Index Merging**: Adjacent vertices frequently share the same offset index; merging these achieves further compression.

**Key Advantage**: At resolution 512, the method achieves a compression ratio of 0.28 and a vocabulary size of 4,736 (vs. BPT's 0.26 / 40,960), substantially improving training efficiency.

### 2. Pre-training Strategy

**Data Curation**: Low-quality meshes are filtered based on geometric structure and visual quality, effectively mitigating loss spikes during training.

**Truncated Training**: Token sequences are split into fixed-size context windows, with a sliding-window mechanism employed for progressive training.

**Data Packaging**: Meshes are categorized by face count and assigned to the same batch when their face counts are similar, ensuring better load balancing.

**Model Architecture**: Based on the Hourglass Transformer, saving 50% of GPU memory; a Michelangelo-based perceiver encoder is used to process point cloud conditions. Model scale ranges from 500M to 1B parameters.

### 3. DPO Post-Training — First Application to 3D Mesh Generation

**Score Standard:**
- **Geometric Completeness**: Chamfer Distance is used to measure the similarity between generated meshes and ground truth.
- **Visual Aesthetics**: Volunteers are recruited for subjective pairwise comparisons, capturing aesthetic judgments that conventional metrics cannot quantify.

**Preference Pair Construction:**
1. For each input point cloud, the model generates two distinct meshes.
2. Chamfer Distance is first used to filter pairs: pairs where both meshes are of poor quality are discarded.
3. If one mesh is clearly superior, it is selected directly; if both qualify, volunteers judge aesthetic preference.
4. A total of 5,000 preference pairs are collected.

**DPO Loss Function:**

$$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(c,y^+,y^-) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y^+|c)}{\pi_{\text{ref}}(y^+|c)} - \beta \log \frac{\pi_\theta(y^-|c)}{\pi_{\text{ref}}(y^-|c)} \right) \right]$$

The truncated training strategy is also adopted to handle long token sequences in DPO.

## Key Experimental Results

### Main Results: Quantitative Comparison on Point Cloud Conditioned Generation

| Metric | MeshAnythingv2 | BPT | Ours (w/o DPO) | **Ours (w DPO)** |
|--------|---------------|-----|----------------|------------------|
| C.Dist. ↓ | 0.1249 | 0.1425 | 0.1001 | **0.0884** |
| H.Dist. ↓ | 0.2991 | 0.2796 | 0.1861 | **0.1708** |
| User Study ↑ | 10% | 19% | 34% | **37%** |

- DeepMesh substantially outperforms all baselines in geometric accuracy (Chamfer Distance reduced by 29.2% vs. MeshAnythingv2).
- DPO post-training further improves both geometric quality and user preference (CD: 0.1001 → 0.0884).
- 37% of volunteers prefer DeepMesh's results in the user study.

### Ablation Study: Tokenization Algorithm Comparison

| Metric | AMT | EdgeRunner | BPT | **Ours** |
|--------|-----|-----------|-----|----------|
| Compression Rate ↓ | 0.46 | 0.47 | 0.26 | **0.28** |
| Vocabulary Size ↓ | 512 | 512 | 40960 | **4736** |
| Time (s) ↓ | 816 | - | 540 | **480** |

- At resolution 512, DeepMesh's tokenization achieves the **best balance** between compression rate and vocabulary size.
- The vocabulary is only 11.6% the size of BPT's, yielding the highest training efficiency.
- A smaller vocabulary enables easier learning and more stable training.

### Effect of DPO Post-Training

Qualitative analysis shows:
- Both pre- and post-DPO models generate well-formed geometric structures.
- Post-DPO results are visually more appealing, with more regular wireframes and richer surface details.
- Quantitative metrics confirm that DPO improves similarity to ground truth.

### Key Findings

1. The model can generate high-fidelity meshes with up to 30K faces, far exceeding baseline methods.
2. As few as 5,000 preference pairs are sufficient to significantly improve generation quality via DPO.
3. The data curation strategy effectively mitigates loss spikes during training.
4. The combination of truncated training and data packaging noticeably improves training efficiency.

## Highlights & Insights

1. **First application of RLHF/DPO to 3D mesh generation**: Transferring methodology proven in LLMs to 3D generation demonstrates cross-modal methodological value.
2. **Engineering innovation in tokenization**: Reducing vocabulary size from 40,960 to 4,736 while maintaining high compression rate is the key enabler for practical high-resolution mesh generation.
3. **Dual-criteria scoring design**: Combining objective metrics (Chamfer Distance) with subjective evaluation (human preference) provides a more comprehensive assessment than either dimension alone.
4. **Systematic pre-training optimization**: Data curation, data packaging, and truncated training together constitute a robust training pipeline.
5. **Diversity capability**: The same point cloud input can yield multiple meshes with distinct appearances, which is highly valuable for design applications.

## Limitations & Future Work

1. **Generation speed**: Auto-regressive generation of 30K-face meshes requires predicting a large number of tokens, resulting in long inference times.
2. **Point cloud only**: Image-conditioned generation requires a prior conversion to point clouds (via TRELLIS), representing an indirect solution.
3. **Limited DPO data scale**: Only 5,000 preference pairs are used; larger-scale alignment data could yield further improvements.
4. **No texture**: The current framework generates geometric meshes only, without texture information.

## Related Work & Insights

- **MeshGPT → MeshAnything → BPT → DeepMesh**: The rapid evolution of auto-regressive mesh generation.
- **DPO applied across modalities**: LLM → VLLM → 3D Mesh, a successful case of cross-modal methodology transfer.
- **Insight**: The potential of RLHF/DPO in other 3D generation tasks (e.g., texture generation, scene generation).

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First application of DPO in the 3D mesh domain; tokenization improvements are practically valuable.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive quantitative and qualitative evaluation including user studies and multi-dimensional ablations.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured, though with occasional typos (e.g., "poineer").
- **Value**: ⭐⭐⭐⭐⭐ — Advances the state of the art in automatic high-quality artist mesh generation.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] MeshAnything V2: Artist-Created Mesh Generation with Adjacent Mesh Tokenization](meshanything_v2_artist-created_mesh_generation_with_adjacent_mesh_tokenization.md)
- [\[ICLR 2026\] Topology-Preserved Auto-regressive Mesh Generation in the Manner of Weaving Silk](../../ICLR2026/3d_vision/topology-preserved_auto-regressive_mesh_generation_in_the_manner_of_weaving_silk.md)
- [\[ICCV 2025\] MeshPad: Interactive Sketch-Conditioned Artist-Reminiscent Mesh Generation and Editing](meshpad_interactive_sketch-conditioned_artist-reminiscent_mesh_generation_and_ed.md)
- [\[NeurIPS 2025\] Mesh-RFT: Enhancing Mesh Generation via Fine-Grained Reinforcement Fine-Tuning](../../NeurIPS2025/3d_vision/mesh-rft_enhancing_mesh_generation_via_fine-grained_reinforcement_fine-tuning.md)
- [\[ICCV 2025\] LACONIC: A 3D Layout Adapter for Controllable Image Creation](laconic_a_3d_layout_adapter_for_controllable_image_creation.md)

<!-- RELATED:END -->

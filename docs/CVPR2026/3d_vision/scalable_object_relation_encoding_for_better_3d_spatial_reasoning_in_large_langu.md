---
title: >-
  [Paper Note] Scalable Object Relation Encoding for Better 3D Spatial Reasoning in Large Language Models
description: >-
  [CVPR 2026][3D Vision][3D spatial reasoning] This paper proposes QuatRoPE, a quaternion rotation-based 3D positional encoding method that preserves all $O(n^2)$ pairwise spatial relations using only $O(n)$ input tokens.…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "3D spatial reasoning"
  - "positional encoding"
  - "quaternion rotation"
  - "large language models"
  - "3D vision-language"
date: 2026-05-08
content_hash: 37a6d6553be9ee99
---

# Scalable Object Relation Encoding for Better 3D Spatial Reasoning in Large Language Models

**Conference**: CVPR 2026
**arXiv**: [2603.24721](https://arxiv.org/abs/2603.24721)  
**Code**: [https://github.com/oceanflowlab/QuatRoPE](https://github.com/oceanflowlab/QuatRoPE)  
**Area**: 3D Vision / Multimodal VLM
**Keywords**: 3D spatial reasoning, positional encoding, quaternion rotation, large language models, 3D vision-language

## TL;DR

This paper proposes QuatRoPE, a quaternion rotation-based 3D positional encoding method that preserves all $O(n^2)$ pairwise spatial relations using only $O(n)$ input tokens. Combined with the IGRE mechanism to reduce interference with language RoPE, it achieves substantial improvements across multiple 3D vision-language benchmarks.

## Background & Motivation

1. **Background**: 3D spatial reasoning requires models to localize target objects based on inter-object spatial relations (e.g., "to the left of the table"), which is a core capability for 3D visual grounding (3D VG) and 3D visual question answering (3D VQA). Due to the scarcity of 3D scene–text paired data, the dominant approach injects point cloud features into the LLM input space and leverages the LLM's pretrained reasoning capabilities for spatial inference.

2. **Limitations of Prior Work**: Existing methods fall into two encoding categories, each with notable drawbacks:
    - **Absolute positional encoding** (e.g., Chat-Scene, LEO): fuses 3D coordinates with geometric features early on, making it difficult for the LLM to extract spatial relations from these entangled representations.
    - **Explicit pairwise relation encoding** (e.g., 3DGraphLLM): uses additional tokens to represent pairwise object relations, but the number of tokens grows quadratically with the number of objects (e.g., 554 objects yield over 150,000 pairs), far exceeding LLM input limits. KNN pruning can reduce tokens but "nearest neighbor ≠ relevant," potentially omitting critical spatial relations.

3. **Key Challenge**: How can a model perceive all pairwise spatial relations while maintaining linear input length?

4. **Goal**:
    - Encode $O(n^2)$ spatial relations within $O(n)$ tokens.
    - Avoid spurious similarity caused by independent axis encoding.
    - Integrate spatial positional encoding with language RoPE without mutual interference.

5. **Key Insight**: Inspired by RoPE (Rotary Position Embedding), which converts absolute positions into relative positions, this work encodes only absolute coordinates on each object token and lets the attention dot product automatically compute pairwise relative positions.

6. **Core Idea**: Encode 3D coordinates as a holistic vector via quaternion rotation, such that attention scores depend solely on the relative positional difference between objects.

## Method

### Overall Architecture

Given a point cloud scene and a text instruction, the scene is first segmented into objects. Each object's features (geometric features extracted by PointNet++) are projected into the LLM input space, and object identifiers (e.g., `<obj005>`) are assigned. Each object corresponds to several object-related tokens. QuatRoPE encodes the object's 3D absolute position (bounding box center) on these tokens, and the pairwise relative positions are automatically computed via the QK dot product in each attention layer. The IGRE mechanism ensures that QuatRoPE only affects attention between object tokens without interfering with language tokens.

### Key Designs

1. **QuatRoPE (Quaternion Rotary Position Embedding)**:

    - **Function**: Encodes absolute positions on $O(n)$ tokens and automatically computes $O(n^2)$ pairwise relative positions via the attention dot product.
    - **Mechanism**: The query/key vectors are partitioned into 3D segments (treated as pure quaternions), which are rotated according to the object coordinate $\vec{m}$ via $f(\vec{q}, \vec{m}) = Q(\vec{m}) \vec{q} Q^{-1}(\vec{m})$. The rotation matrix $Q(\vec{m})$ is decomposed via Euler angles into rotations around three axes, and the frequency function $\theta$ is derived to be linear. This guarantees that the dot product of two rotated vectors depends only on $\vec{m}-\vec{n}$ (the relative positional difference), satisfying $\langle f(\vec{q},\vec{m}), f(\vec{k},\vec{n})\rangle = g(\vec{q},\vec{k},\vec{m}-\vec{n})$.
    - **Design Motivation**: Compared to independent axis encoding methods such as M-RoPE, QuatRoPE encodes coordinates as a holistic vector. When two objects are close along one axis but far apart in 3D space, M-RoPE produces spuriously inflated attention scores along that axis's corresponding dimensions ("false neighbors"), whereas QuatRoPE's quaternion rotation couples every dimension with the full 3D coordinate, ensuring that only spatially proximate objects receive high attention scores.

2. **IGRE (Isolated Gated RoPE Extension)**:

    - **Function**: Isolates QuatRoPE from language RoPE to prevent interference when both simultaneously rotate query/key vectors.
    - **Mechanism**: For object tokens, a set of basis vectors rotated by QuatRoPE is appended to the query/key vectors; for non-object tokens (system prompts, questions, etc.), zero vectors are appended as padding. Consequently, the QuatRoPE dimensions contribute non-zero values only when both participating tokens are object tokens, while their contribution is zero whenever a non-object token is involved.
    - **Design Motivation**: Directly applying QuatRoPE to all tokens would treat non-object tokens as located at the coordinate origin $(0,0,0)$, erroneously causing the model to over-attend to objects near the origin. IGRE's "isolation + gating" mechanism ensures that QuatRoPE exclusively modulates attention scores between object tokens, maximally preserving the LLM's original language understanding and reasoning capabilities.

3. **ASR Benchmark (Attribute-free Spatial Reasoning Benchmark)**:

    - **Function**: Purely evaluates the model's spatial reasoning ability, eliminating confounds from other capabilities such as attribute recognition.
    - **Mechanism**: Questions with unique answers referring to object names are filtered from ScanQA; questions that leak target object attributes (color, category, etc.) are removed; the remaining questions are converted to the 3D VG format (multiple choice) to eliminate language generation bias.
    - **Design Motivation**: In existing benchmarks (ScanRefer, SQA3D, etc.), descriptions often intermix spatial relations with non-spatial cues (e.g., "red chair"), allowing models to bypass spatial reasoning via attribute recognition. ASR forces models to rely solely on spatial relations for target localization.

### Loss & Training

The LLM is fine-tuned using LoRA (rank=16, $\alpha$=16) with a learning rate of $2\times10^{-5}$. Training data is a joint dataset comprising ScanRefer, Multi3DRef, ScanQA, SQA3D, and others. The frequency parameters of QuatRoPE vary across 3D segments following an exponential decay design analogous to the original RoPE.

## Key Experimental Results

### Main Results

Results on ScanRefer, Multi3DRef, and SQA3D benchmarks (GT segmentation):

| Model | ScanRefer Acc@0.25 | ScanRefer Acc@0.5 | Multi3DRef F1@0.25 | SQA3D EM@1 |
|------|-------------------|-------------------|-------------------|------------|
| Chat-Scene-1B | 50.7 | 50.3 | 53.3 | 50.7 |
| Chat-Scene-1B + QuatRoPE | **55.4** | **55.0** | **58.1** | **53.1** |
| 3DGraphLLM-1B | 55.9 | 55.8 | 58.6 | 51.1 |
| 3DGraphLLM-1B + QuatRoPE | **58.3** | **58.2** | **60.7** | **53.2** |
| Chat-Scene-7B (Mask3D) | 55.5 | 50.2 | 57.1 | 54.6 |
| Chat-Scene-7B + QuatRoPE | **57.8** | **52.2** | **59.5** | **54.7** |

### Ablation Study

Comparison of positional encoding strategies (based on Chat-Scene-1B + IGRE):

| Encoding Method | ScanRefer Acc@0.25 | ScanRefer Acc@0.5 | SQA3D EM@1 | Notes |
|---------|-------------------|-------------------|------------|------|
| No explicit encoding | 50.72 | 50.33 | 50.72 | Baseline |
| Raw Coordinates | 52.26 | 52.01 | 51.40 | Direct absolute coordinates |
| M-RoPE | 54.30 | 53.92 | 51.55 | Independent axis encoding |
| QuatRoPE | **55.44** | **55.00** | **53.14** | Holistic vector encoding |

Zero-shot results on the ASR spatial reasoning benchmark (3DGraphLLM-8B):

| Model | ASR Acc@0.25 | Gain |
|------|-------------|------|
| 3DGraphLLM (w/o QuatRoPE) | 37.50 | — |
| 3DGraphLLM + QuatRoPE | **41.96** | +4.46 (11.9%) |

### Key Findings

- QuatRoPE yields consistent gains across all baselines and all metrics, with the largest gains on ScanRefer (~+4–5 percentage points), indicating that explicit spatial relation encoding benefits localization tasks the most.
- IGRE significantly outperforms the Trans-Additive combination strategy, validating the necessity of the isolation and gating mechanism.
- The smaller the single-axis coordinate difference $\delta$ between objects, the larger the advantage of QuatRoPE over M-RoPE (gain reaches 5.83% at $\delta=0.1$), confirming that holistic vector encoding eliminates spurious neighbors.
- On the ASR purely spatial reasoning benchmark, QuatRoPE yields 12–19% relative improvement, directly validating the method's enhancement of spatial reasoning capabilities.

## Highlights & Insights

- **Encoding quadratic relations with linear inputs**: By cleverly exploiting the dot product property of the attention mechanism, $O(n)$ absolute positions are automatically converted into $O(n^2)$ relative positions. This "encode absolute, compute relative" paradigm is transferable to any scenario requiring pairwise relations (e.g., molecular conformations, social networks).
- **Quaternion holistic encoding**: In contrast to the spurious neighbor problem of independent axis RoPE, quaternion rotation modulates every dimension of the query/key with the full 3D coordinate, fundamentally resolving the attention inflation caused by single-axis approximation.
- **Elegant IGRE design**: Zero-padding non-object tokens combined with dimension isolation both preserves the LLM's original capabilities and introduces 3D spatial information in a gated manner, constituting a general paradigm for multimodal RoPE extension.

## Limitations & Future Work

- QuatRoPE represents objects by their bounding box centers, discarding shape and size information, which may be insufficient for relations requiring precise geometry (e.g., "on the surface of X").
- Validation is limited to ScanNet indoor scenes; performance on large-scale outdoor scenes (sparser object distributions, larger coordinate ranges) has not been tested.
- Although ASR excludes attribute cues, it is derived from ScanQA and has limited scale (evaluated in zero-shot only).
- The frequency parameters of quaternion rotation currently follow a fixed exponential decay; learnable frequencies to accommodate scenes of varying scale warrant exploration.

## Related Work & Insights

- **vs. 3DGraphLLM**: 3DGraphLLM explicitly encodes spatial relations via additional tokens but faces $O(n^2)$ scaling and KNN pruning errors. QuatRoPE implicitly computes all relations within $O(n)$ tokens via positional encoding, and is more scalable. Notably, stacking QuatRoPE on top of 3DGraphLLM yields further improvement, suggesting the two information sources are complementary.
- **vs. M-RoPE (Qwen2-VL)**: M-RoPE encodes multi-dimensional positions by grouping segments per axis, which is effective for 2D images but introduces "spurious neighbors" when extended to 3D. QuatRoPE's quaternion holistic rotation resolves this fundamental flaw.
- **vs. Chat-Scene**: Chat-Scene fuses absolute coordinates into object features, making spatial information implicit and sparse. QuatRoPE provides spatial relations that are explicit and naturally computed through attention.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The idea of quaternion rotary positional encoding is highly novel, with rigorous mathematical derivation.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multi-baseline, multi-dataset comparison with ablation analysis and the dedicated ASR benchmark, though evaluation is limited to the ScanNet series.
- Writing Quality: ⭐⭐⭐⭐⭐ — Motivation is clearly articulated, mathematical derivations are rigorous, and figures are intuitive.
- Value: ⭐⭐⭐⭐ — Provides a general-purpose positional encoding scheme for 3D LLM spatial reasoning that is plug-and-play across multiple architectures.

This paper proposes QuatRoPE — a quaternion rotation-based 3D positional encoding method that encodes absolute coordinates onto object tokens and automatically converts them into pairwise relative positions via attention dot products, preserving all $O(n^2)$ spatial relations within $O(n)$ input length, thereby significantly enhancing the spatial reasoning capabilities of 3D LLMs.

## Background & Motivation

1. **Background**: 3D vision-language tasks (VG, VQA) require models to understand inter-object spatial relations. Due to the scarcity of 3D scene–text paired data, the dominant approach injects point cloud representations into LLMs and leverages pretrained reasoning capabilities.
2. **Limitations of Prior Work**: (A) **Absolute positional encoding** (e.g., Chat-Scene, LEO) fuses 3D coordinates into object features, but the early feature fusion makes it difficult for the LLM to extract spatial relations; (B) **Explicit relation encoding** (e.g., 3DGraphLLM) uses additional tokens to encode inter-object relations, but the number of relations scales quadratically as $O(n^2)$, readily exceeding LLM input limits. KNN pruning may omit critical relations.
3. **Key Challenge**: A balance must be struck between input length (scalability) and completeness of spatial relations.
4. **Goal**: (1) Encode complete pairwise relative positions with linear input length; (2) avoid spurious attention inflation caused by single-axis coordinate proximity; (3) integrate the new positional encoding with the LLM's existing language RoPE without interference.
5. **Key Insight**: Leveraging a property of the Transformer attention mechanism — the query-key dot product naturally converts absolute positions into relative positions (analogous to 1D RoPE) — and generalizing this idea to 3D.
6. **Core Idea**: Encode 3D coordinates into query/key vectors via quaternion rotation so that attention scores depend only on the relative 3D displacement between objects, yielding a scalable scheme that preserves $O(n^2)$ relations with $O(n)$ tokens.

## Method

### Overall Architecture

Building on an existing 3D LLM pipeline (point cloud segmentation → object feature extraction → projection into LLM input space), QuatRoPE applies a quaternion rotation based on 3D coordinates to the query/key vectors of object-related tokens before the dot product computation in each attention layer. The IGRE (Isolated Gated RoPE Extension) mechanism ensures that QuatRoPE affects only the attention scores between object tokens without interfering with language RoPE.

### Key Designs

1. **QuatRoPE (Quaternion Rotary Position Embedding)**:

    - **Function**: Encodes each object's 3D absolute coordinate onto its corresponding token, and automatically computes pairwise relative positions via attention dot products.
    - **Mechanism**: Query/key vectors are grouped into 3D segments (treated as pure quaternions $\vec{q}, \vec{k}$) and rotated according to the object's 3D coordinate $\vec{m}$ via $f(\vec{q}, \vec{m}) = Q(\vec{m}) \vec{q} Q^{-1}(\vec{m})$. The rotation matrix $Q(\vec{m})$ is decomposed via Euler angles into three rotations around the x/y/z axes. The key property is $\langle f(\vec{q}, \vec{m}), f(\vec{k}, \vec{n}) \rangle = g(\vec{q}, \vec{k}, \vec{m}-\vec{n})$, i.e., the dot product depends only on the relative position. The derivation shows that the angle function must be linear.
    - **Design Motivation**: Unlike methods such as M-RoPE that encode each axis independently, QuatRoPE encodes coordinates as a holistic vector. Under independent axis encoding, if two objects are close in coordinates along one axis (even if far apart in 3D), the corresponding dimensions' dot products produce spuriously inflated attention scores. QuatRoPE avoids this by having the quaternion's three rotational degrees of freedom couple every dimension with the full 3D coordinate.

2. **IGRE (Isolated Gated RoPE Extension)**:

    - **Function**: Combines QuatRoPE with language RoPE without mutual interference.
    - **Mechanism**: Additional dimensions are appended to the query/key vectors of object tokens, to which QuatRoPE is applied; non-object tokens (prompts, instructions) receive zero padding in these dimensions. This ensures (1) QuatRoPE and language RoPE operate on separate dimensions without interference; (2) zero padding prevents non-object tokens from being incorrectly "placed" at the 3D coordinate origin; (3) the QuatRoPE adjustment is effective only when both tokens participating in the dot product are object tokens (gating effect).
    - **Design Motivation**: Naively stacking two RoPEs causes mutual interference. Since the scarcity of 3D data precludes training a dual-RoPE LLM from scratch, a combination strategy that does not degrade pretrained language capabilities is necessary.

3. **ASR Benchmark (Attribute-free Spatial Reasoning Benchmark)**:

    - **Function**: A diagnostic benchmark that purely evaluates the model's spatial reasoning ability.
    - **Mechanism**: Questions with unique target object names are filtered from ScanQA; all descriptions that reveal target object attributes (color, shape, etc.) are removed, retaining only questions where the target can be localized solely through spatial relations; the remaining questions are converted to the 3D VG format to eliminate language generation bias (e.g., "What is the object in front of the tall white shelf?" → "The object in front of the tall white shelf").
    - **Design Motivation**: In existing benchmarks, object attribute descriptions are often intertwined with spatial relations, allowing models to localize targets via attribute recognition rather than spatial reasoning, making it impossible to truly measure spatial understanding.

### Loss & Training

- LoRA fine-tuning (rank=16, α=16) with learning rate $2\times 10^{-5}$.
- Training data: joint dataset of ScanRefer + Multi3DRef + ScanQA + SQA3D + Scan2Cap + ReferIt3D + object alignment tasks.
- QuatRoPE introduces no additional learnable parameters; the frequencies $\theta_x(1), \theta_y(1), \theta_z(1)$ are preset values.

## Key Experimental Results

### Main Results

Comparison on multiple 3D VL benchmarks (GT segmentation, 1B models):

| Method | ScanRefer Acc@0.25 | ScanRefer Acc@0.5 | Multi3DRef F1@0.25 | SQA3D EM@1 |
|------|-------------------|-------------------|-------------------|------------|
| Chat-Scene-1B | 50.7 | 50.3 | 53.3 | 50.7 |
| Chat-Scene-1B + QuatRoPE | **55.4** | **55.0** | **58.1** | **53.1** |
| 3DGraphLLM-1B | 55.9 | 55.8 | 58.6 | 51.1 |
| 3DGraphLLM-1B + QuatRoPE | **58.3** | **58.2** | **60.7** | **53.2** |

ASR spatial reasoning benchmark (zero-shot):

| Method | LLM | Acc@0.25 | Gain |
|------|-----|----------|------|
| 3DGraphLLM | Llama-3-8B | 37.50 | - |
| 3DGraphLLM + QuatRoPE | Llama-3-8B | **41.96** | +4.46 (11.9%) |

### Ablation Study

| Positional Encoding Method | ScanRefer Acc@0.25 | SQA3D EM@1 | Notes |
|-------------|-------------------|------------|------|
| No explicit encoding | 50.72 | 50.72 | Chat-Scene baseline |
| Raw coordinate addition | 52.26 | 51.40 | Absolute position, not converted to relative |
| M-RoPE | 54.30 | 51.55 | Independent axis encoding |
| QuatRoPE (Ours) | **55.44** | **53.14** | Holistic vector encoding |

Validation of holistic vector encoding advantage (smaller $\delta$ indicates more severe "spurious neighbor" problem):

| δ Threshold | 3DGraphLLM | + QuatRoPE | Gain |
|--------|-----------|------------|------|
| 1 (all) | 93.72 | 94.65 | +0.93 |
| 0.1 | 92.39 | 96.74 | +4.35 |
| 0.05 | 84.62 | 92.31 | **+7.69** |

### Key Findings

- QuatRoPE achieves consistent improvements across all baselines and benchmarks, with the largest gains on spatially intensive VG tasks.
- The more severe the "spurious neighbor" problem (objects close on one axis but far apart in 3D space), the larger QuatRoPE's relative advantage (from +0.93 to +7.69), validating the necessity of holistic vector encoding.
- IGRE outperforms simple additive combination (Trans-Additive), demonstrating that the isolation and gating design is critical for preserving the LLM's original capabilities.
- Directly adding raw coordinates to features causes a substantial performance drop on 3DGraphLLM (3.60 vs. 55.92), as absolute positions disrupt the model's understanding of input tokens.

## Highlights & Insights

- **Encoding $O(n^2)$ spatial relations with $O(n)$ tokens**: By cleverly exploiting the attention dot product to automatically compute relative positions, the method avoids the quadratic token explosion of explicit relation encoding. This trick is generalizable to any scenario requiring pairwise relations.
- **Quaternion vs. independent axis encoding**: The paper precisely identifies and resolves the spurious attention inflation caused by per-axis independent encoding — analogous to how Euclidean distance cannot be decomposed into the sum of projected Manhattan distance components.
- **Zero-padding in IGRE**: Elegantly resolves the semantic problem of "non-object tokens having no 3D position" — zero padding makes the extended dimensions contribute zero rather than defaulting to the origin position.
- Spatially proximate objects naturally receive higher attention scores, corresponding to the linguistic "Maxim of Relation," making the model more robust to implicit reference.

## Limitations & Future Work

- Representing object positions by bounding box centers is overly coarse, ignoring object shape and spatial extent.
- The Euler angle decomposition of quaternion rotation introduces mathematical approximation errors (not an exact solution).
- The ASR benchmark is limited in scale due to filtering from ScanQA.
- QuatRoPE's performance in large-scale outdoor scenes (object count >> 100) has not been explored.
- Future directions include extension to continuous space (e.g., encoding raw point cloud coordinates rather than discrete object centers) and incorporating object scale information into the positional encoding.

## Related Work & Insights

- **vs. 3DGraphLLM**: 3DGraphLLM explicitly encodes KNN relations via additional tokens ($O(n \cdot k)$ tokens), potentially missing distant but important relations. QuatRoPE implicitly preserves all $O(n^2)$ relations with $O(n)$ tokens, and stacking QuatRoPE on top of 3DGraphLLM yields further improvement (+2.4 Acc@0.25), indicating the two sources of information are complementary.
- **vs. M-RoPE (Qwen2-VL)**: M-RoPE encodes multi-dimensional positions by grouping segments per axis, which is effective for 2D images but exhibits "spurious neighbors" when extended to 3D. QuatRoPE's quaternion holistic rotation resolves this fundamental flaw.
- QuatRoPE can serve as a general-purpose positional encoding component for 3D LLMs, inspiring spatially aware multimodal tasks beyond the settings evaluated here.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Encoding 3D positions via quaternion rotation and leveraging attention dot products to convert them to relative positions is an elegant and mathematically rigorous approach.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multi-baseline, multi-benchmark validation with a self-constructed ASR diagnostic benchmark, detailed ablation, and spurious neighbor severity analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ — Problem definition is clear, figures are intuitive, and the logic from motivation to method to validation is tight.
- Value: ⭐⭐⭐⭐⭐ — The proposed positional encoding scheme is highly general, introduces no additional parameters, and is plug-and-play, offering a substantial contribution to the 3D LLM field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Masking Matters: Unlocking the Spatial Reasoning Capabilities of LLMs for 3D Scene-Language Understanding](masking_matters_unlocking_the_spatial_reasoning_capabilities_of_llms_for_3d_scen.md)
- [\[CVPR 2026\] Learning Multi-View Spatial Reasoning from Cross-View Relations](learning_multi-view_spatial_reasoning_from_cross-view_relations.md)
- [\[NeurIPS 2025\] SoFar: Language-Grounded Orientation Bridges Spatial Reasoning and Object Manipulation](../../NeurIPS2025/3d_vision/sofar_language-grounded_orientation_bridges_spatial_reasoning_and_object_manipul.md)
- [\[CVPR 2026\] Context-Nav: Context-Driven Exploration and Viewpoint-Aware 3D Spatial Reasoning for Instance Navigation](context-nav_context-driven_exploration_and_viewpoint-aware_3d_spatial_reasoning_.md)
- [\[CVPR 2026\] SPAN: Spatial-Projection Alignment for Monocular 3D Object Detection](span_spatial_projection_alignment_mono3d.md)

</div>

<!-- RELATED:END -->

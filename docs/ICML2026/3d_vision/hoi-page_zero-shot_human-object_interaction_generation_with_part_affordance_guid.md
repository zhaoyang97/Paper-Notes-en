---
title: >-
  [Paper Note] HOI-PAGE: Zero-Shot Human-Object Interaction Generation with Part Affordance Guidance
description: >-
  [ICML 2026][3D Vision][4D HOI] HOI-PAGE enables LLMs to first "reason" which body parts should contact which object components…
tags:
  - "ICML 2026"
  - "3D Vision"
  - "4D HOI"
  - "Part-level Affordance"
  - "Affordance Graph"
  - "Video Diffusion Distillation"
  - "Zero-shot Generation"
date: 2026-05-08
content_hash: 568cbf36d1f0d703
---

# HOI-PAGE: Zero-Shot Human-Object Interaction Generation with Part Affordance Guidance

**Conference**: ICML 2026  
**arXiv**: [2506.07209](https://arxiv.org/abs/2506.07209)  
**Code**: https://craigleili.github.io/projects/hoipage (Project Page)  
**Area**: 3D Vision / Human-Object Interaction Generation / Video Diffusion  
**Keywords**: 4D HOI, Part-level Affordance, Affordance Graph, Video Diffusion Distillation, Zero-shot Generation

## TL;DR
HOI-PAGE enables LLMs to first "reason" which body parts should contact which object components, formulating these results into a "Part Affordance Graph" (PAG). This PAG then drives 3D part segmentation, video diffusion, and optimization to generate 4D human-object interaction sequences for complex "multi-person single-object / single-person multi-object" scenes without any 4D training data.

## Background & Motivation
**Background**: The mainstream approach for 4D Human-Object Interaction (HOI) generation relies on diffusion models (e.g., HOI-Diff, CHOIS), which denoise the joint motion of the human and object as unified tokens. These methods are trained on ground truth 4D datasets like BEHAVE or GRAB, leading to narrow object vocabularies that primarily cover "single-person single-object" scenes.

**Limitations of Prior Work**: Training data acquisition is expensive and scarce. When generalizing to new objects (e.g., guitars, lawnmowers), humans often "float near the object," resulting in obvious penetrations, lack of contact, or mismatches between action and text. Multi-person or multi-object scenes are nearly impossible to handle due to the exponential growth of contact relationships.

**Key Challenge**: The essence of HOI is not the "proximity of human centroid to object centroid," but rather the fine-grained contact between "specific body parts $\leftrightarrow$ specific functional object components" (e.g., hand gripping a handle, foot stepping on a pedal). Global pose-level modeling discards this part-level semantics; models fail to learn this without data and merely memorize distributions rather than reasoning when data is available.

**Goal**: Generate 4D sequences starting from text and 3D meshes that explicitly model part-level affordances (e.g., "which hand grabs which handle") and scale to multi-person/multi-object scenes without relying on any 4D HOI training data.

**Key Insight**: LLMs already possess common-sense knowledge regarding daily interactions (e.g., which hand holds the iron's base and which presses the handle). Grounding this linguistic "interaction script" into a graph structure, and subsequently into 3D geometry, video, and optimization, allows the burden of vision-action generation to be distributed across "existing strong prior components."

**Core Idea**: Use LLMs to infer a **Part Affordance Graph (PAG)** as the script for the entire pipeline—nodes represent parts, and edges represent contact constraints. The PAG unifies 3D part segmentation, video diffusion prompting, and various losses (contact, penetration, smoothness) in 4D optimization.

## Method
HOI-PAGE is a four-stage pipeline: **(1) LLM translates text and object lists into a PAG; (2) Abstract object parts in the PAG are anchored to 3D geometry (part segmentation); (3) Expanded prompts drive video diffusion to generate reference interaction videos, recovering 2D/3D object point clouds, depth, and human SMPL-X sequences; (4) The PAG guides optimization to "lift" the video into controllable 4D object pose sequences.** The final output consists of $\{(R_t, t_t)\}_{t=1}^T$ for each object and SMPL-X parameters $\{\Theta_t\}_{t=1}^T$ for each person, requiring no 4D HOI ground truth.

### Overall Architecture
Input: A set of 3D object meshes $\{O\}$ + a text prompt $\Gamma$ (e.g., "A person ironing clothes on an ironing board"). Output: $T=49$ frames of human SMPL-X sequences + 6DoF pose trajectories for each object. Among the four stages—LLM generating the PAG, SAM-2/open-vocabulary detection for 3D part segmentation, CogVideoX generating reference videos with monocular depth/human recovery, and 600-step gradient descent for trajectory optimization—**only the final step is adjustable/learnable, while all others are frozen**, hence "zero-shot."

### Key Designs

1.  **Part Affordance Graph (PAG)—An Interaction Script Written by LLM**:
    -   **Function**: Compresses all key HOI constraints (which parts exist, which hand contacts them, whether contact is persistent, whether it's relatively static, whether the object translates/rotates) into a graph $G=(V,E)$ as a unified control signal.
    -   **Mechanism**: Nodes $V=V_o \cup V_h$ include object parts and 12 human body part categories. Virtual parent nodes $v$ for each object/person carry motion states $(a_r, a_\tau)$ indicating rotation/translation. Each edge $e=(v_1,v_2)$ represents a part-level contact with attributes $(a_c, a_s)$: $a_c$ denotes if contact is **persistent** throughout the video, and $a_s$ denotes if contact is **relatively static** (e.g., hand holding a handle vs. an iron sliding on a board). The graph is generated by an LLM (DeepSeek series) via in-context reasoning.
    -   **Design Motivation**: HOI is a set of discrete part-level constraints rather than a "pose-object joint distribution" easily learned end-to-end. Decoupling "common-sense reasoning" from "geometric execution" via language models transforms the lack of 4D data into a challenge of "how geometric optimization satisfies graph constraints." The graph structure is naturally scalable to multi-person/multi-object scenarios.

2.  **PAG-guided Video Diffusion and Constraint Extraction**:
    -   **Function**: Translates the PAG into an expanded prompt $\Gamma^+$ for CogVideoX to generate a reference video, which is then "sliced" into geometric and motion constraints.
    -   **Mechanism**: LLM expands $\Gamma$ into a detailed video prompt using edge contact types. FLUX generates five initial frame candidates, and GPT-4 selects the most anatomically plausible one as an anchor. After CogVideoX generates 49 frames, open-vocabulary detection and SAM-2 perform part-level segmentation. Monocular depth estimation (Wang et al. 2024) back-projects masks into 3D part point cloud sequences. Human SMPL-X sequences $\{\Theta_t\}$ are extracted using Shen et al. 2024.
    -   **Design Motivation**: Video diffusion handles the hardest task—realistic human-object movement—but is treated only as a "soft reference" due to geometric inaccuracies. The PAG's hard constraints "correct" object poses during optimization, bypassing the trade-off between video model geometric inaccuracy and geometric model semantic deficiency.

3.  **PAG-guided 4D Optimization—Lifting Videos to 4D**:
    -   **Function**: Solves for $\{(R_t, t_t)\}_{t=1}^T$ for each object to fit 2D/3D video observations, satisfy PAG constraints, avoid human penetration, and maintain temporal smoothness.
    -   **Mechanism**: Total loss is $L_{\text{total}} = \lambda_{\text{fit}} L_{\text{fit}} + \lambda_{\text{con}} L_{\text{con}} + \lambda_{\text{pen}} L_{\text{pen}} + \lambda_{\text{smo}} L_{\text{smo}}$. $L_{\text{fit}}$ is a multi-modal Chamfer distance. $L_{\text{con}} = L_{cc} + L_{cd}$ handles contact: $L_{cc}$ averages nearest neighbors across frames if $a_c=\text{true}$, otherwise only the minimum frame. $L_{cd}$ penalizes relative displacement if $a_s=\text{true}$ and encourages smooth change via $L_2(P_t^{v_2 \to v_1}, \tfrac{1}{2}(P_{t-1}^{v_2 \to v_1}+P_{t+1}^{v_2 \to v_1}))$ if $a_s=\text{false}$. $L_{\text{pen}}$ uses pre-computed SDF to penalize human vertices entering objects. $L_{\text{smo}}$ switches between spherical linear interpolation and rigid penalty based on $(a_r, a_\tau)$.
    -   **Design Motivation**: All losses are "conditioned" by PAG attributes per edge/node, allowing the same structure to handle diverse interactions (e.g., "grip" vs. "touch") without modifying the implementation.

### Loss & Training
The process involves **no model training**; only the final step optimizes object poses. LLM, video diffusion, depth estimation, human recovery, and SAM-2 are all frozen. Optimization utilizes 600 gradient descent steps with 4 random initializations.

## Key Experimental Results

### Main Results
Evaluated on a self-built Sketchfab dataset (24 objects, 16 single-person prompts, 5 multi-person/object prompts) against HOI-Diff and CHOIS.

| Metric | HOI-Diff | CHOIS | HOI-PAGE |
|------|---------|-------|----------|
| VideoCLIP ↑ (Semantics) | 0.233 | 0.239 | **0.250** |
| Obj. Smoothness ↓ | 0.035 | 0.009 | **0.006** |
| Obj. Diversity ↑ | 0.72 | 0.49 | **0.80** |
| Non-collision ↑ | 0.98 | 0.98 | **0.99** |
| Contact ↑ | 0.76 | 0.64 | **0.92** |

Perceptual studies show HOI-PAGE beats baselines with 91%–99% preference. On a 1–5 scale, HOI-PAGE scores ~4.0 (Realism: 3.97, Alignment: 4.07), while baselines score $\le 1.9$.

### Ablation Study
| Configuration | VideoCLIP ↑ | Smoothness ↓ | Diversity ↑ | Contact ↑ | Notes |
|------|-----|-----|-----|-----|-----|
| Full | 0.290 | 0.004 | 0.83 | 0.76 | Full constraints |
| w/o Part Fitting (PF) | 0.290 | 0.004 | 0.81 | 0.76 | Coarser poses |
| w/o Part Contact (PC) | 0.289 | 0.011 | 0.71 | **0.26** | Contact collapse |
| w/o Obj. Motion State (OMS) | 0.290 | 0.006 | 0.78 | 0.73 | Unintended motion |

### Key Findings
- **Contact drops from 0.76 to 0.26 without PC**: The LLM-inferred contact graph is the pipeline's linchpin; geometric losses alone cannot enforce semantic "holding" constraints.
- **HOI-Diff has smoother humans but lowest diversity (0.35)**: Supervised models tend to memorize training distributions rather than generating diverse, realistic movements.
- **Zero-data outperforms data-driven**: HOI-PAGE surpasses baselines trained on 4D ground truth across all dimensions, particularly for unseen objects (e.g., lawnmower).
- **Scalability is low-cost**: Performance remains stable for multi-person (4.17) and multi-object (4.46) scenes by simply adding graph elements.

## Highlights & Insights
- **LLM as "Director" vs. "Writer"**: Having the LLM output a structured constraint graph (PAG) rather than long prompts ensures geometric modules strictly follow constraints, bypassing LLM/VLM hallucinations.
- **Edge-conditioned loss is an elegant unification**: Switching between eight loss modes via four boolean attributes $(a_c, a_s, a_r, a_\tau)$ avoids custom pipelines for different interactions.
- **Complementary strengths**: By assigning roles based on expertise—LLM for semantics, video diffusion for motion, and SDF for geometry—the model serves as a paradigm for composite zero-shot pipelines.

## Limitations & Future Work
- Optimization relies on video diffusion quality; long videos (>49 frames), complex backgrounds, or large camera movements cause constraint failures.
- Point clouds from monocular depth are noisy; fitting for thin/small objects (e.g., forks) is challenging.
- Attribute accuracy depends heavily on prompt engineering for the LLM.
- Speed: 6-10 minutes per optimization with multiple initializations is not real-time.

## Related Work & Insights
- **vs HOI-Diff / CHOIS**: These learn joint distributions and depend on 4D data; **Ours** decouples semantics and geometry for zero-shot superiority.
- **vs ZeroHSI / ZeroHOI / DAViD**: While these use video diffusion, they treat humans/objects as global entities. **Ours** is the first to introduce explicit part-level graph structures for scaling.
- **vs PiGraphs / iMapper**: PiGraphs used interaction graphs for static scenes; **Ours** extends this to 4D using LLM reasoning.

## Rating
- Novelty: ⭐⭐⭐⭐ (Translating LLM reasoning into explicit graph structures for geometric optimization is a distinct approach.)
- Experimental Thoroughness: ⭐⭐⭐ (Solid comparisons, though lacking evaluation on public benchmarks like BEHAVE.)
- Writing Quality: ⭐⭐⭐⭐ (Stages are clear, and the PAG-conditioned loss is well-explained.)
- Value: ⭐⭐⭐⭐ (Significant for zero-shot and scalable 4D HOI generation.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] AnchorHOI: Zero-shot Generation of 4D Human-Object Interaction via Anchor-based Prior Distillation](../../AAAI2026/3d_vision/anchorhoi_zero-shot_generation_of_4d_human-object_interactio.md)
- [\[CVPR 2026\] CARI4D: Category Agnostic 4D Reconstruction of Human-Object Interaction](../../CVPR2026/3d_vision/cari4d_category_agnostic_4d_reconstruction_of_human_object_interaction.md)
- [\[CVPR 2026\] Human Interaction-Aware 3D Reconstruction from a Single Image](../../CVPR2026/3d_vision/human_interaction-aware_3d_reconstruction_from_a_single_image.md)
- [\[CVPR 2026\] TeHOR: Text-Guided 3D Human and Object Reconstruction with Textures](../../CVPR2026/3d_vision/tehor_text-guided_3d_human_and_object_reconstruction_with_textures.md)
- [\[ICML 2026\] PhysHanDI: Physics-Based Reconstruction of Hand-Deformable Object Interactions](physhandi_physics-based_reconstruction_of_hand-deformable_object_interactions.md)

</div>

<!-- RELATED:END -->

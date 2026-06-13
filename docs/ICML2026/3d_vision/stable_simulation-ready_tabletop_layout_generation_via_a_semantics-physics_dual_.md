---
title: >-
  [Paper Note] STABLE: Simulation-Ready Tabletop Layout Generation via a Semantics–Physics Dual System
description: >-
  [ICML 2026][3D Vision][Tabletop Scene Generation] STABLE decomposes the "task instruction → simulation-ready scene" pipeline into an LLM-based Semantic Reasoner (for rough layout) and a flow-matching-based Physics Correc…
tags:
  - "ICML 2026"
  - "3D Vision"
  - "Tabletop Scene Generation"
  - "Semantics–Physics Dual System"
  - "Flow Matching"
  - "SDF Collision Loss"
  - "Progressive Inference"
date: 2026-05-08
content_hash: ecda14c8ba1905c2
---

# STABLE: Simulation-Ready Tabletop Layout Generation via a Semantics–Physics Dual System

**Conference**: ICML 2026  
**arXiv**: [2605.16137](https://arxiv.org/abs/2605.16137)  
**Code**: https://stable-tabletop.github.io  
**Area**: 3D Vision / Embodied AI / Scene Generation  
**Keywords**: Tabletop Scene Generation, Semantics–Physics Dual System, Flow Matching, SDF Collision Loss, Progressive Inference

## TL;DR
STABLE decomposes the "task instruction → simulation-ready scene" pipeline into an LLM-based Semantic Reasoner (for rough layout) and a flow-matching-based Physics Corrector with SDF loss (for pose refinement). By iterating through these components in three stages (task-critical → background), it reduces object collisions to zero and achieves an Alignment with Scene Graph (AwS) of 99.0% on MesaTask-10K.

## Background & Motivation

**Background**: Training Embodied AI increasingly relies on synthetic data. Generating tabletop scenes that can be directly fed into simulators based on manipulation instructions (e.g., "place the apple to the left of the banana") is a key step in robot manipulation data production. Current mainstream approaches use LLMs (zero-shot prompting, multi-turn prompts, or SFT on scene data) to directly output scene JSON—representative works include LayoutGPT, I-Design, Holodeck, and MesaTask.

**Limitations of Prior Work**: Pure LLM approaches have structural weaknesses in 3D spatial reasoning: (1) discretizing continuous coordinates into tokens leads to insufficient precision, resulting in frequent **object interpenetration, floating, and table penetration**, which causes simulators to crash; (2) post-processing optimization (e.g., Steerable’s physical post-proc), while eliminating collisions, drastically moves object positions—an apple might be pushed to the **right** of the banana, violating the instruction's semantics. Consequently, "task alignment" and "physical feasibility" conflict in existing methods.

**Key Challenge**: LLMs excel at semantics but struggle with continuous geometry; optimizers excel at geometry but lack semantic understanding. Serializing them allows the latter to override the former.

**Goal**: (1) Let the LLM handle only the rough semantic layout without requiring physically precise poses; (2) use a lightweight, geometry-aware corrector to update $(\mathbf{p}, r)$ while preserving object identities, sizes, and relations; (3) prevent contamination between "adding objects" and "correcting poses" through staged interleaving.

**Key Insight**: The authors draw an analogy to the "System 1 / System 2" VLA approach from Helix—dividing labor between fast and slow systems. The Semantic Reasoner is the slow-thinking semantic system, and the Physics Corrector is the fast-responding geometric system. The two **interact frequently** rather than performing a single pass.

**Core Idea**: A "Semantics–Physics Dual System" (SR + PC) is used to progressively expand the scene. Every time a batch of objects is added, flow-matching and mesh-level SDF losses are immediately applied to project poses back into the physically feasible region, achieving simulation-readiness without sacrificing task semantics.

## Method

### Overall Architecture
Input: Task instruction $I$ + tabletop specification $T$. Output: Structured JSON scene $J=\{T, \{O_i\}_{i=1}^N\}$, where each object $O_i=\{\mathbf{p}_i, r_i, s_i, d_i\}$ includes 3D translation, yaw rotation, bbox size, and text description, associated with a mesh $a_i$ retrieved from a 3D asset library.

The overall process is a three-stage progressive loop (Loop 1→2→3):

1. **Loop 1**: SR generates the task-oriented object set $O^t$ (objects explicitly named in the instruction) → Assets are retrieved → PC corrects poses;
2. **Loop 2**: SR generates important background objects $O^B$ (physically touching or adjacent to $O^t$) conditioned on $(I, T, O^t)$ → PC corrects again;
3. **Loop 3**: SR generates secondary background objects $O^b$ (distant distractors) conditioned on $(I, T, O^t, O^B)$ → PC performs final refinement.

The poses corrected by the PC are fed back as context for the next stage of the SR, **preventing the accumulation of geometric errors across stages**. The process can be pipelined in batches: while Scenario A runs the PC, Scenario B can simultaneously undergo SR inference.

### Key Designs

1. **Progressive Semantic Reasoner (Three-stage SFT)**:

    - **Function**: Reorganizes full-scene annotations from MesaTask-10K into three-stage sequences $(O^t, O^t\cup O^B, O^t\cup O^B\cup O^b)$, teaching the LLM to expand outward from "core, to neighbors, to background." Inference follows these same three steps.
    - **Mechanism**: $O^B$ is automatically determined using bbox intersection thresholds, with others categorized as $O^b$. Chain-of-thought reasoning from MesaTask is removed during training to output JSON directly and compress tokens. The steps are $O^t \leftarrow \mathrm{SR}(I, T)$, $O^B \leftarrow \mathrm{SR}(I, T, O^t)$, and $O^b \leftarrow \mathrm{SR}(I, T, O^t, O^B)$.
    - **Design Motivation**: Generating the whole scene at once often misses key objects named in instructions (weak task-grounding). Staged generation forces the LLM to lock in task-core objects before filling backgrounds, improving instruction alignment and allowing the PC to perform local corrections on smaller sets. In ablation studies, AwT increased from 89.9% to 99.4%, and the Distractor Rate rose from 78.6% to 86.1%.

2. **Geometry-aware Flow-Matching Physics Corrector**:

    - **Function**: Performs continuous correction only on the pose vector $\mathbf{x}=[\mathbf{p}_1,\dots,\mathbf{p}_N, r_1,\dots,r_N]\in\mathbb{R}^{4N}$ while keeping $(s_i, d_i, a_i)$ constant.
    - **Mechanism**: Mesh-level geometric embeddings $\mathbf{g}_i=\phi(\mathcal{P}_i)$ are extracted from sampled surface point clouds of each asset using a frozen PointTransformer-V3, forming the condition $\mathcal{C}=(\mathbf{x}^c, \mathbf{G})$. During training, Gaussian noise is added to the rough pose $\mathbf{x}^c$ from the SR to get $\mathbf{x}_0=\mathbf{x}^c+\sigma\boldsymbol{\epsilon}$. The GT pose is defined as $\mathbf{x}_1$. Interpolation follows $\mathbf{x}_t=(1-t)\mathbf{x}_0+t\mathbf{x}_1$. A U-Net learns the velocity field $\mathbf{v}_\theta$ to fit $\mathbf{v}_{\mathrm{target}}=\mathbf{x}_1-\mathbf{x}_0$, with loss $\mathcal{L}_{\mathrm{flow}}=\mathbb{E}\|\mathbf{v}_\theta(\mathbf{x}_t, t, \mathcal{C})-(\mathbf{x}_1-\mathbf{x}_0)\|_2^2$. During inference, ODE integration starts from $\mathbf{x}(0)=\mathbf{x}^c$ and proceeds to $t=1$ to obtain the corrected pose.
    - **Design Motivation**: Bboxes cannot handle stacking or containment (e.g., a small box inside a large box), requiring actual mesh geometry. Flow matching is used instead of diffusion because correction is essentially "small-scale local calibration." Starting from the rough pose during inference allows the model to learn a local correction flow around $\mathbf{x}^c$, which is more stable than generating from pure noise.

3. **Mesh-level SDF Physics Constraint Trio**:

    - **Function**: Adds three differentiable SDF losses to the training objective to directly penalize "interpenetration, table penetration, and floating."
    - **Mechanism**: SDF $D_m(\mathbf{x})$ is precomputed for each mesh $m$ (negative values indicate the interior). The object-object interpenetration loss is $\mathcal{L}_{\mathrm{obj\text{-}obj}}=\sum_{i<j}[\max(0, -\mathrm{dist}_{\mathrm{sdf}}(i,j))]^2$, where $\mathrm{dist}_{\mathrm{sdf}}(i,m)=\min_{\mathbf{q}\in\mathcal{Q}_i}D_m(\mathbf{q})$. Table penetration $\mathcal{L}_{\mathrm{obj\text{-}table}}$ is modeled similarly using table SDF $\tau$. For support-contact, bottom points $\mathcal{B}_i$ and candidate support surfaces $\mathcal{S}_i$ are sampled; the nearest support is $z_i^{\mathrm{sup}}=\arg\min_s \delta(i,s)$, and the loss is $\mathcal{L}_{\mathrm{sup}}=\sum_i[\max(0, \mathrm{gap}(i,z_i^{\mathrm{sup}})-\epsilon)]^2$. Using $|D_s(\cdot)|$ penalizes both floating and being stuck far from support in concave surfaces. The total objective is $\mathcal{L}_{\mathrm{PC}}=\mathcal{L}_{\mathrm{flow}}+\lambda_{\mathrm{sdf}}(\mathcal{L}_{\mathrm{obj\text{-}obj}}+\mathcal{L}_{\mathrm{obj\text{-}table}})+\lambda_{\mathrm{sup}}\mathcal{L}_{\mathrm{sup}}$.
    - **Design Motivation**: Pure data-driven methods always leave small but fatal interpenetrations that crash simulators. Using mesh-level SDF instead of bboxes is necessary because bboxes are too coarse for stacking/containment, where minor offsets cause hidden penetrations. The three losses are complementary; ablation shows that removing any one significantly degrades physical metrics. Specifically, removing $\mathcal{L}_{\mathrm{obj\text{-}table}}$ actually lowers the floating rate (as objects "sink" into the table to bypass support constraints), revealing coupling between losses.

### Loss & Training
The PC is trained on all 10K scenarios from MesaTask-10K. The SR is fine-tuned (SFT) on an open-source LLM using the 10K instances rewritten into three-stage progressive sequences. Batch pipelining during inference allows SR and PC to run in parallel across different scenarios to maximize throughput.

## Key Experimental Results

### Main Results

| Dataset | Metric | STABLE | Prev. SOTA | Gain |
|--------|------|--------|-----------|------|
| MesaTask-10K | FID ↓ | **38.6** | MesaTask 40.6 | -2.0 |
| MesaTask-10K | AwT (Alignment w/ Task, %) | **99.4** | Steerable 99.4 | — |
| MesaTask-10K | AwS (Alignment w/ Scene Graph, %) | **99.0** | Steerable 91.1 | +7.9 |
| MesaTask-10K | OC (Object Collisions) | **0** | Steerable 0 / MesaTask 15.6 | Par with Steerable without moving objects |
| MesaTask-10K | GPT Average Score | **9.0** | TabletopGen 8.6 | +0.4 |

| Task | Metric | STABLE | StructDiffusion | LEGO-NET |
|------|------|--------|-----------------|----------|
| Rearrangement | Distance Move ↓ | **0.14** | 0.21 | 0.28 |
| Rearrangement | EMD to GT ↓ | **0.08** | 0.23 | 0.43 |
| Rearrangement | OC ↓ | **0** | 0.25 | 0.32 |

Key Observation: Steerable achieves 0 OC by aggressively moving objects, but its AwS is only 91.1%, validating the "collision elimination vs. semantic preservation" conflict. STABLE is the only method to achieve both OC=0 and AwS=99.0%.

### Ablation Study

| Configuration | OC ↓ | Float ↓ | AwT ↑ | Distractor Rate ↑ | Description |
|------|------|---------|-------|--------------------|------|
| Full PC | **0** | **0** | — | — | Full Physics Corrector |
| w/o $\mathcal{L}_{\mathrm{sup}}$ | 4.7 | 9.8 | — | — | Floating rate spikes |
| w/o $\mathcal{L}_{\mathrm{obj\text{-}table}}$ | 13.6 | 5.4 | — | — | Objects sink into table (pseudo float reduction) |
| w/o $\mathcal{L}_{\mathrm{obj\text{-}obj}}$ | 11.9 | 15.8 | — | — | Double failure of penetration and stability |
| One-shot SR | — | — | 89.9 | 78.6 | Full scene generation at once |
| Progressive SR | — | — | **99.4** | **86.1** | Three-stage progressive |

| Initial Collision Bucket | 0-10 | 10-20 | 20-30 | 30-40 |
|--------------|------|-------|-------|-------|
| MesaTask Optim. (50K iter) | ✓ | ✓ | ✗ | ✗ |
| Steerable | ✓ | ✓ | ✗ | ✗ |
| Physics Corrector | ✓ | ✓ | **✓** | **✓** |

### Key Findings
- The three SDF losses are highly coupled: removing $\mathcal{L}_{\mathrm{obj\text{-}table}}$ actually **reduces** the floating rate because objects sink into the table to satisfy support constraints. This proves that individual physical metrics can be misleading and must be evaluated jointly.
- Progressive SR outperforms one-shot SR by 9.5% in AwT and increases the Distractor Rate by 7.5%, proving that staged generation strengthens task grounding and makes scenes **richer** rather than sparser.
- The PC consistently finds collision-free solutions even in "severe scenarios" with 30-40 collisions, whereas post-processing optimizers (even with 50K iterations) fail. Learned corrections are more robust than iterative optimizers in extremely crowded scenes.

## Highlights & Insights
- **"Dual System + Frozen Size" is the key to decoupling trade-offs**: By modifying only $(\mathbf{p}, r)$ and not $(s, d)$, the PC effectively freezes the "semantic skeleton" and fine-tunes it in physical space. This fundamentally avoids the semantic damage seen in Steerable, where eliminating collisions might accidentally move an apple to the wrong side of a banana. This 4N-dimensional pose subspace design is simple yet highly effective.
- **Training-Inference consistency via Noise + Flow-from-coarse-pose**: Training uses GT pose with noise as the starting point, while inference starts from the SR's rough pose. This ensures the PC learns a **local correction flow** in the neighborhood of $\mathbf{x}^c$ rather than full scene generation, matching its role as a corrector.
- **Portability of Mesh-level SDF Losses**: The "interpenetration, table penetration, and floating" trio is extracted as three differentiable SDF losses plus a nearest-support selection mechanism. This combination is directly reusable for any downstream task requiring differentiable poses and physical feasibility (e.g., furniture layout, grasp candidates, AR object placement).

## Limitations & Future Work
- Only translation and yaw are corrected, neglecting pitch/roll. This provides insufficient coverage for objects with diverse orientations, such as tilted bottles or slanted books. Extending this to full SE(3) rotation presents new challenges for SDF loss gradients.
- High dependence on MesaTask-10K annotations and asset libraries. The "reasonable layout distribution" learned by the PC may not transfer to new scenes (outdoor, greasy kitchen environments, soft bodies), and the pipeline degrades if asset retrieval fails.
- The three-stage SR split uses hard-defined bbox intersection thresholds for $O^B$, which might misclassify long or L-shaped objects. The fixed number of stages also limits scalability to extremely complex scenes (>50 objects).
- While batch pipeline throughput is optimized, single-scene latency data is missing. Further analysis is needed for real-time robot closed-loop applications (e.g., online rearrangement).

## Related Work & Insights
- **vs MesaTask (SFT-LLM)**: MesaTask uses an LLM for end-to-end layout, resulting in OC=15.6 and AwS=90.2. STABLE builds on its SR with a PC and progressive inference to achieve OC=0 and AwS=99.0, proving that decoupling semantics (LLM) from geometry (specialized model) is more scalable.
- **vs Steerable (Post-processing Optimization)**: Steerable matches the OC=0 performance but drops to 91.1% AwS and fails to converge in scenarios with >20 collisions even after 50K iterations. STABLE’s learned PC succeeds across all collision buckets without damaging semantics, showing local correction flows are more reliable than iterative optimization in severe collisions.
- **vs TabletopGen (Image Intermediary)**: TabletopGen uses images as a text-to-3D bridge, achieving an 8.6 GPT score, but suffers from missing small objects or occlusion issues. STABLE operates directly at the 3D structural level, avoiding ambiguity accumulation from image intermediaries.
- **Insight**: The Helix System 1/System 2 concept is successfully applied to non-embodied control scenarios here. Any generation task where "semantics are easy to express but physics is hard to pin down" can benefit from this template: rough semantic draft + flow-matching local correction + mesh SDF constraints.

## Rating
- Novelty: ⭐⭐⭐⭐ The dual-system split is not entirely new, but the specific combination of "flow-matching from coarse pose + mesh-level SDF trio + progressive stage interleaving" is a first for this task and shows high engineering integration.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 4 types of baselines, physical constraint ablations, progressive ablations, collision robustness, rearrangement downstream, and editing applications. Lacks SE(3) rotation and real-time latency analysis.
- Writing Quality: ⭐⭐⭐⭐ The Helix analogy makes the motivation clear; formulas and algorithms are described logically; visualization of failure modes (red/yellow/blue boxes) Is intuitive.
- Value: ⭐⭐⭐⭐ Provides a practical pipeline for robot manipulation simulation data production. The SDF loss trio has clear reusable value for the scene generation community, though evidence for room-scale expansion is still needed.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] PhyScene3D: Physically Consistent Interactive 3D Tabletop Scene Generation](physcene3d_physically_consistent_interactive_3d_tabletop_scene_generation.md)
- [\[CVPR 2026\] MotionAnymesh: Physics-Grounded Articulation for Simulation-Ready Digital Twins](../../CVPR2026/3d_vision/motionanymesh_physics-grounded_articulation_for_simulation-ready_digital_twins.md)
- [\[NeurIPS 2025\] Gaussian-Augmented Physics Simulation and System Identification with Complex Colliders](../../NeurIPS2025/3d_vision/gaussian-augmented_physics_simulation_and_system_identification_with_complex_col.md)
- [\[ICML 2026\] LabBuilder: Protocol-Grounded 3D Layout Generation for Interactable and Safe Laboratory](labbuilder_protocol-grounded_3d_layout_generation_for_interactable_and_safe_labo.md)
- [\[CVPR 2026\] PhysHead: Simulation-Ready Gaussian Head Avatars](../../CVPR2026/3d_vision/physhead_simulation-ready_gaussian_head_avatars.md)

</div>

<!-- RELATED:END -->

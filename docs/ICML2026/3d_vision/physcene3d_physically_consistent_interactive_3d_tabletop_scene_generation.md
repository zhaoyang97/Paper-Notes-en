---
title: >-
  [Paper Note] PhyScene3D: Physically Consistent Interactive 3D Tabletop Scene Generation
description: >-
  [ICML 2026][3D Vision][Tabletop Scene Generation] PhyScene3D reformulates 3D tabletop scene generation as a hierarchical sequential planning process following a "human-constructive" logic. It employs a Cognitive Topologi…
tags:
  - "ICML 2026"
  - "3D Vision"
  - "Tabletop Scene Generation"
  - "VLM"
  - "Differentiable SDF"
  - "Test-time Optimization"
  - "Physical Consistency"
date: 2026-05-08
content_hash: b08af1af89c409a4
---

# PhyScene3D: Physically Consistent Interactive 3D Tabletop Scene Generation

**Conference**: ICML 2026  
**arXiv**: [2606.01649](https://arxiv.org/abs/2606.01649)  
**Code**: To be confirmed  
**Area**: 3D Vision / Scene Generation / Embodied AI  
**Keywords**: Tabletop Scene Generation, VLM, Differentiable SDF, Test-time Optimization, Physical Consistency

## TL;DR
PhyScene3D reformulates 3D tabletop scene generation as a hierarchical sequential planning process following a "human-constructive" logic. It employs a Cognitive Topological Reasoning Chain (CTRC) to linearize scene graphs into anchor sequences based on AABBs and introduces Physics-Aware Denoising Alignment (PADA) to embed a differentiable SDF physics engine into the VLM training loop. This allows the generated scenes to surpass the physical plausibility of manually annotated training data (reducing scene-level collision rates from 81.5% to 41.6% and asset-level rates to 3.86%).

## Background & Motivation

**Background**: Interactive 3D tabletop scenes—those that are kinematically valid, free of penetrations, and ready for simulators like IsaacGym or SAPIEN—are fundamental for training generalist robot manipulation policies. Current approaches generally fall into three categories: (a) Agent-based solvers (e.g., Holodeck, I-Design) where LLMs generate symbolic constraints for external solvers; (b) Image-mediated pipelines that first generate 2D images, parse them, and retrieve assets; (c) End-to-end regression models (e.g., MesaTask) that directly regress 6D object poses from datasets.

**Limitations of Prior Work**: (a) Agent-based methods suffer from a "symbolic bottleneck"—LLMs lack fine-grained spatial awareness, often producing geometrically unsolvable graphs (e.g., floating or impossible stacking), causing downstream solvers to either break semantics or fail. (b) Multi-stage pipelines exhibit high latency and error accumulation. (c) End-to-end models are limited by the quality of training data; the MesaTask-10k dataset itself has an 81.5% scene-level penetration rate. Naive supervised learning forces the model to replicate these non-physical artifacts.

**Key Challenge**: Tabletop scenes are more challenging than indoor furniture layouts because they require strict 3D topology (e.g., a pen must be inside a holder, which must be on a book) with 10–20 objects densely packed. Satisfying both "semantic faithfulness to instructions" and "physical zero-penetration/non-floating" simultaneously is a dual trap: pure RL for physics leads to semantic drift (scattering objects to avoid collision), while pure imitation inherits data noise. 

**Goal**: To enable the physical plausibility of generative models to **surpass** the upper bounds of the training data while maintaining generalization to out-of-distribution (OOD) scenes, without abandoning the semantic priors of VLMs.

**Key Insight**: The authors make two critical observations: (1) Humans follow a hierarchical order of "anchor → lateral expansion → bottom-up stacking" when arranging a desk; injecting this as a structural inductive bias into the VLM can eliminate causal hallucinations like placing contents before containers. (2) Training noise should be viewed as an "imperfect reference that can be denoised" rather than ground truth. By backpropagating differentiable physical signals to the VLM parameters, the model can learn layouts cleaner than the original annotations.

**Core Idea**: Internalize the traditional "explicit planning + solver post-processing" workflow into the VLM's implicit reasoning. CTRC provides the structural skeleton (linearized AABB anchor sequences), while PADA uses differentiable SDFs and Test-Time Optimization (TTO) to "project" SFT outputs onto a physically feasible manifold. These projected "pseudo-labels" are then used as semantic anchors in GRPO training to distill physical priors back into the policy.

## Method

### Overall Architecture
The input is a natural language instruction $\mathcal{I}$, and the output is a tabletop scene $\mathcal{S}=\{e_i\}_{i=1}^N$. Each entity $e_i=(c_i,\mathbf{p}_i,\mathbf{s}_i,\theta_i)$ is reparameterized as a 3D AABB $\mathbf{b}_i=[x_{\min},x_{\max},\dots,z_{\max}]\in\mathbb{R}^6$, encoding position and size to measure spatial occupancy directly. The backbone is Qwen-3 VL 8B. Training consists of two stages: (i) SFT on the MesaTask-CTRC dataset with hierarchical scene graph annotations for full-parameter fine-tuning; (ii) PADA stage using LoRA (r=16) + GRPO, where each training prompt undergoes SFT inference to produce $\mathcal{S}_{sft}$, followed by TTO projection to the physical manifold to obtain anchors $\mathcal{S}^*_{anchor}$ for joint optimization with RL exploration.

### Key Designs

1.  **Cognitive Topological Reasoning Chain (CTRC) — Linearizing Scene Generation into AABB Anchor Sequences**:
    - **Function**: Reconstructs flat set generation into a bottom-up ordered autoregressive process $P(\mathcal{S}|\mathcal{I})=\prod_{t=1}^{N}P(e_t|e_{<t},\mathcal{I})$ using hierarchical scene graphs and relative AABB representations, enforcing a "container → content" causal order.
    - **Mechanism**: A scene graph $G=(V,E)$ is extracted using geometric heuristics (Containment via volume ratio $V_B/V_A\geq 1.5$ and $IoU_{xy}\geq 0.9$; Support via $z_{min}^A \approx z_{max}^B$; Proximity via separating axis theorem). Edge priority follows $\text{in} \succ \text{on} \succ \text{near}$. Each child object's pose is represented as a relative AABB in the parent's local coordinate system. Vertical dimensions are decoupled: $z^{rel}_{\{min,max\}} = z^{abs}_{\{min,max\}}(e_{ch}) - z^{abs}_{min}(e_{pa})$ for $\text{in}$ relationships, and $z^{rel}_{\{min,max\}} = z^{abs}_{\{min,max\}}(e_{ch}) - z^{abs}_{max}(e_{pa})$ for $\text{on}$.
    - **Design Motivation**: Relative AABBs turn "inside" and "on" into different mathematical offsets, anchoring the search space to geometric invariants. This eliminates floating or penetration hallucinations at the source, while high-precision physical alignment is handled by the SDF engine.

2.  **Differentiable SDF Physics Engine + Hierarchy-Constrained TTO — Projecting Semantic Plans to Physical Manifolds**:
    - **Function**: Uses GPU-accelerated vectorized SDF representations for all assets to compute differentiable penetration energy, "pushing" objects apart while using relative coordinate constraints to prevent TTO from breaking the semantic structure of CTRC.
    - **Mechanism**: For objects $A$ and $B$, the collision energy is $\mathcal{L}_{sdf}(A,B) = \sum_{\mathbf{p}\in P_A} \text{ReLU}(-\phi_B(\mathbf{R}_B^\top(\mathbf{R}_A \mathbf{p} + \mathbf{t}_A - \mathbf{t}_B)))$, where child surface points are sampled and evaluated against the parent's SDF field. The TTO objective is $\min_\xi (\mathcal{L}_{sdf} + \lambda_{rel}\mathcal{L}_{rel}(\mathcal{G}) + \lambda_{reg}\|\xi - \xi_{init}\|^2)$, where $\mathcal{L}_{rel}$ freezes relative positions for $\text{in}$ edges and enforces alignment for $\text{on}$ edges.
    - **Design Motivation**: Absolute coordinate TTO often causes drift (pushing objects far apart). By adding constraints in the relative coordinate system, TTO can only translate entire rigid groups or fine-tune within legal affordance regions, ensuring collision-free results that maintain semantic intent.

3.  **Physically-Projected Semantic Anchoring — Distilling TTO Capability back to VLM**:
    - **Function**: Automatically generates physically correct and semantically faithful pseudo-labels $\mathcal{S}^*_{anchor}$ using the SFT model plus TTO, serving as dense supervision in GRPO to avoid the need for manual semantic rewards.
    - **Mechanism**: For each training instruction, the SFT model generates $\mathcal{S}_{sft}$ (capturing $\mathcal{M}_{sem}$ with noise). This is refined via Hierarchy-Constrained TTO to obtain $\mathcal{S}^*_{anchor}$. The dual-objective GRPO loss is $\mathcal{J}(\theta) = \mathbb{E}_{\mathcal{L}\sim\pi_\theta}\left[\frac{1}{K}\sum_{k=1}^K A_k \frac{\pi_\theta(\mathcal{S}_k)}{\pi_{old}(\mathcal{S}_k)}\right] + \alpha \cdot \mathcal{L}_{CE}(\pi_\theta, \mathcal{S}^*_{anchor})$, where advantage $A_k$ is derived from collision scores and cross-entropy anchors the output.
    - **Design Motivation**: Pure GRPO tends to reward-hack (GPT-Score drops from 8.96 to 8.47). Using TTO as a teacher distills physical capability into the policy distribution, resulting in a QPR (46.5%) that outperforms inference-only TTO (38.0%).

### Loss & Training
The two-stage training involves: (i) SFT on 9,429 samples with CTRC sequence labels using Qwen-3 VL 8B; (ii) PADA involving LoRA (r=16) for GRPO, where $K$ rollouts are generated online per prompt. Advantage $A_k$ is derived from Asset Collision Rate (ACR) and Scene Collision Rate (SCR), while $\alpha$ balances semantic stability and physical exploration.

## Key Experimental Results

### Main Results
Evaluated on the MesaTask-CTRC benchmark (866 samples, 6 scenarios × 5 difficulty levels) using Quality Pass Rate (QPR: GPT-Score > 7 and collision-free) and Collision Rates:

| Method | QPR (τ=7) | GPT Score Avg | SCR ↓ | ACR ↓ |
| :--- | :--- | :--- | :--- | :--- |
| Reference (Human) | 17.1% | 8.87 | 81.5% | 8.19% |
| GPT-4o | 27.6% | 8.19 | 68.9% | 7.87% |
| Holodeck-table (Agent) | 2.7% | 4.60 | 2.7% | 0.47% |
| I-Design-table (Agent) | 19.0% | 6.53 | 39.1% | 5.94% |
| MesaTask (End-to-End) | 21.1% | 8.80 | 78.3% | 8.19% |
| **PhyScene3D (Ours)** | **46.5%** | **8.93** | **41.6%** | **3.86%** |

On OOD scenes (Cashier Counter, Nightstand, etc.), MesaTask QPR collapses to 1.01%, while PhyScene3D maintains 29.1%. Diffusion-based DiffuScene failed with SCR=100% on dense tabletop scenes.

### Ablation Study

| Configuration | QPR (τ=7) | GPT Avg | SCR ↓ | ACR ↓ |
| :--- | :--- | :--- | :--- | :--- |
| Qwen-3 VL 8B (SFT only) | 19.2% | 8.84 | 80.1% | 8.18% |
| + CTRC | 21.6% | 8.96 | 77.8% | 7.60% |
| + GRPO (Pure RL) | 28.8% | 8.47 | 68.9% | 6.82% |
| + TTO (Post-hoc) | 38.0% | 8.83 | 60.5% | 4.86% |
| **+ PADA (Full)** | **46.5%** | **8.93** | **41.6%** | **3.86%** |

In robotic downstream tasks (ManiSkill), agent success rates trained on PhyScene3D scenes reached 50.4% IID / 14.1% OOD, a 10x improvement over MesaTask-based training.

### Key Findings
- **CTRC Stabilizes Semantics**: Hierarchical bias is a prerequisite; without it, TTO causes major drift.
- **Pure GRPO Reward-Hacks**: GPT-Score drops significantly (by 0.49) as outputs become sparse to avoid collisions.
- **PADA Outperforms Inference TTO**: This indicates that the VLM "learns" physics via distillation, providing better initial generations than TTO starting from noise.
- **Structural Generalization**: Relative AABBs capture topological invariants, leading to 10x QPR gains in OOD scenarios.

## Highlights & Insights
- **Surpassing Training Data**: By treating noise as a denoisable reference, the model systematically exceeds human annotation performance.
- **TTO-Refined Anchoring**: A cost-effective way to provide dense semantic supervision during RL, bypassing the need for independent reward models.
- **Observable Error Design**: Decoupling vertical dimensions and using relative AABBs makes physical errors like floating and penetration mathematically explicit and differentiable.
- **Structural Chain-of-Thought**: CTRC is not just prompt engineering but the hard-coding of human inductive biases (ordering) into the generation process.

## Limitations & Future Work
- AABB can be overly conservative for irregular shapes (e.g., baskets with handles).
- Achieving absolute zero collisions in very dense scenes (10-20 objects) remains difficult.
- Dependence on GPT-4 as a judge introduces potential source-model bias.
- Future work aims to utilize Oriented Bounding Boxes (OBB), support articulated objects (e.g., opening drawers), and explore diffusion policy integration.

## Related Work & Insights
- **Compared to Agent Solvers**: Moves from explicit symbolic solving to internalized implicit reasoning, reducing latency and avoiding symbolic bottlenecks.
- **Compared to Regression Models**: Breaks the "data quality ceiling" through physical self-correction.
- **Compared to Diffusion Models**: Highlights that pure generative modeling without physical enforcement fails in dense, high-constraint tabletop environments.
- **Integration of Differentiable Physics**: Demonstrates the value of using differentiable simulators to provide gradients for policy optimization in sim-to-real contexts.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] STABLE: Simulation-Ready Tabletop Layout Generation via a Semantics–Physics Dual System](stable_simulation-ready_tabletop_layout_generation_via_a_semantics-physics_dual_.md)
- [\[ICCV 2025\] SuperMat: Physically Consistent PBR Material Estimation at Interactive Rates](../../ICCV2025/3d_vision/supermat_physically_consistent_pbr_material_estimation_at_interactive_rates.md)
- [\[ICML 2026\] PhysForge: Generating Physics-Grounded 3D Assets for Interactive Virtual World](physforge_generating_physics-grounded_3d_assets_for_interactive_virtual_world.md)
- [\[ICLR 2026\] One2Scene: Geometric Consistent Explorable 3D Scene Generation from a Single Image](../../ICLR2026/3d_vision/one2scene_geometric_consistent_explorable_3d_scene_generation_from_a_single_imag.md)
- [\[ICML 2026\] Pair2Scene: Learning Local Object Relations for Procedural Scene Generation](pair2scene_learning_local_object_relations_for_procedural_scene_generation.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICML 2026\] STABLE: Simulation-Ready Tabletop Layout Generation via a Semantics–Physics Dual System](stable_simulation-ready_tabletop_layout_generation_via_a_semantics-physics_dual_.md)
- [\[ICCV 2025\] SuperMat: Physically Consistent PBR Material Estimation at Interactive Rates](../../ICCV2025/3d_vision/supermat_physically_consistent_pbr_material_estimation_at_interactive_rates.md)
- [\[CVPR 2025\] WonderWorld: Interactive 3D Scene Generation from a Single Image](../../CVPR2025/3d_vision/wonderworld_interactive_3d_scene_generation_from_a_single_image.md)
- [\[ICLR 2026\] One2Scene: Geometric Consistent Explorable 3D Scene Generation from a Single Image](../../ICLR2026/3d_vision/one2scene_geometric_consistent_explorable_3d_scene_generation_from_a_single_imag.md)
- [\[CVPR 2026\] AeroDGS: Physically Consistent Dynamic Gaussian Splatting for Single-Sequence Aerial 4D Reconstruction](../../CVPR2026/3d_vision/aerodgs_physically_consistent_dynamic_gaussian_splatting_for_single-sequence_aer.md)

</div>

<!-- RELATED:END -->

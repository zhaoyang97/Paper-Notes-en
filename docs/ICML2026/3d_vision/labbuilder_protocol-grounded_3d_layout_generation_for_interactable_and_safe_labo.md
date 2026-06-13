---
title: >-
  [Paper Note] LabBuilder: Protocol-Grounded 3D Layout Generation for Interactable and Safe Laboratory
description: >-
  [ICML 2026][3D Vision][Lab Scene Generation] LabBuilder compiles free-text experimental descriptions into "asset-chemical protocols," then employs hierarchical generation, multi-objective geometric/chemical optimization…
tags:
  - "ICML 2026"
  - "3D Vision"
  - "Lab Scene Generation"
  - "Protocol Grounding"
  - "Chemical Safety"
  - "Navigation Reachability"
  - "Hierarchical Layout"
date: 2026-05-08
content_hash: 9dac5396d2b1c9fa
---

# LabBuilder: Protocol-Grounded 3D Layout Generation for Interactable and Safe Laboratory

**Conference**: ICML 2026  
**arXiv**: [2605.02288](https://arxiv.org/abs/2605.02288)  
**Code**: None  
**Area**: 3D Vision / Embodied Environment Generation  
**Keywords**: Lab Scene Generation, Protocol Grounding, Chemical Safety, Navigation Reachability, Hierarchical Layout

## TL;DR
LabBuilder compiles free-text experimental descriptions into "asset-chemical protocols," then employs hierarchical generation, multi-objective geometric/chemical optimization, and navigation repair to produce 3D chemical laboratory layouts that are visually plausible and practically executable for robotic experimental workflows.

## Background & Motivation
**Background**: 3D indoor scene generation predominantly serves domestic environments, relying on datasets like 3D-FRONT. The primary objective is visual plausibility—ensuring geometric non-interference and reasonable furniture color coordination. Recent works leverage Large Language Models (LLMs) as layout planners, utilizing pipelines that transform text to structured JSON and then to rendered scenes to convert linguistic descriptions into interactive environments.

**Limitations of Prior Work**: Directly migrating these methods to chemical laboratories often results in failure. In domestic settings, furniture only needs to "fit without overlapping." Conversely, equipment such as fume hoods, alcohol lamps, flammable reagents, and glassware in a laboratory possess **protocol-level semantics**: chemicals must be arranged by reaction type, flammables must remain distant from heat sources, glassware should not sit near table edges, and robotic arms must be able to reach them. Generic generators treat reagent bottles as decorative objects, lacking knowledge of their chemical properties or verification of robotic traversability between workstations.

**Key Challenge**: Existing methods constrain only "static geometric validity + visual plausibility" during generation, leaving executability and safety for post-hoc evaluation. However, in a laboratory, executability is the design constraint itself—a minor geometric shift (e.g., moving an alcohol lamp 20 cm) can invalidate the entire experimental workflow or even trigger safety incidents.

**Goal**: Given a free-text experimental requirement (e.g., "Set up an SN2 substitution reaction"), automatically generate a 3D lab layout where: (i) all assets required by the protocol are instantiated; (ii) geometry is conflict-free and compliant with wall-alignment rules; (iii) chemical safety constraints are satisfied; and (iv) the robot can reach required equipment station-by-station following the protocol steps.

**Key Insight**: The authors reformulate scene generation as "constrained optimization for protocol grounding." They first use an LLM combined with a knowledge base to transform free text into machine-verifiable structured protocols. These protocols then directly drive layout search and repair, shifting executability from post-hoc evaluation to the front-end of the generation loop.

**Core Idea**: Utilizing "Asset Knowledge Bases + Chemical Knowledge Bases" as priors, the method compiles experimental requirements into schema-based protocols. It then generates the laboratory through three stages: hierarchical initialization, local search prioritized by geometric/chemical violations, and navigation reachability repair.

## Method

### Overall Architecture
LabBuilder consists of three tightly coupled modules: **LabForge** is the "front-end compiler" responsible for compiling free text and heterogeneous assets into a structured protocol $\mathcal{P}$ and an asset library $\mathcal{A}$. **LabGen** is the core generator which performs hierarchical initialization to produce a candidate layout $\mathcal{L}_0$, followed by geometric and chemical optimization $\Phi$, and finally navigation-aware repair $\Upsilon$ to output the optimal layout $\mathcal{L}^\star$. **LabTouchstone** is the evaluation suite, scoring across four dimensions: geometric compliance, feasibility (FSR), chemical safety, and semantic plausibility, supplemented by point-goal navigation assessment. The pipeline's key lies in protocol $\mathcal{P}$ serving dual roles as both a "target specification" and a "constraint template"—informing the generator what to place and the optimizer which placements constitute violations.

### Key Designs

1.  **LabForge Protocol Synthesis and Verification**:
    - **Function**: Converts coarse-grained text (e.g., "I want to perform a reflux reaction") into a verifiable protocol $\mathcal{P}$ containing reagents, instruments, steps, and movement actions.
    - **Mechanism**: A library of 176 laboratory entities is constructed with annotations (geometric, semantic, safety). An experimental library covering 7 reaction types (substitution, protection/deprotection, condensation, cyclization, redox, functional group transformation, alkylation/acylation) is extracted from chemical literature. The LLM performs Retrieval-Augmented Generation (RAG) on $(x, \mathcal{C})$ to produce schema-strict protocols with normalized asset references. Constraint checks based on asset library $\mathcal{A}$ ensure executability. Statistically, each protocol averages 5.27 reagents, 9.87 instruments, 9.00 steps, and 4.30 navigations.
    - **Design Motivation**: LLMs generating "placement JSON" directly often produce physical conflicts and lack safety semantics. Compiling into a protocol first provides a schema validator for the generator, preventing hallucinated assets or missing instruments.

2.  **Hierarchical Layout Initialization + Geometric/Chemical Multi-Objective Optimization**:
    - **Function**: Produces feasible initial layouts in a vast continuous configuration space and optimizes them toward the objective function's peak.
    - **Mechanism**: The layout is decomposed into room-level partitioning $(\mathcal{R}, \pi) \sim p_\theta(\cdot \mid x, \mathcal{P}, \mathcal{A})$ (functional zones and 6-DoF poses of large equipment) and desktop-level organization $\mathcal{D}_s \sim p_\theta(\cdot \mid \cdot, s, \mathcal{R}, \pi)$ (placement of small instruments and reagents on each desktop), merged to form $\mathcal{L}_0$. The objective function $\mathbb{F} = w_{\text{geo}} f_{\text{geo}} + w_{\text{chem}} f_{\text{chem}}$ rewards both geometric validity and chemical safety. The search adopts a **violation-first** acceptance criterion: first comparing the number of hard-constraint violations $v(\mathcal{L})$, then comparing $\mathbb{F}$. The operator $\Phi$ blends two repairs: FastRepair for simple geometric conflicts and LLMAdjust for pose modifications requiring semantic reasoning (e.g., "move acetone inside the fume hood"). Optimization converges at the room-level before refined desktop-level adjustment.
    - **Design Motivation**: Generating an entire lab with an LLM in one pass is prone to failure. Hierarchical partitioning + violation-prioritization constrains combinatorial search to $O(10)$ objects per layer, while reserving the LLM for high-semantic-difficulty tasks.

3.  **Navigation-Aware Repair**:
    - **Function**: Ensures the robot can reach each station according to protocol steps via collision-free paths.
    - **Mechanism**: The 3D scene is projected onto a 2D occupancy grid, inflated by the robot's radius. $A^\star$ planning is used for every (start, goal) pair in the protocol. Failures are categorized into endpoint occupancy, out-of-bounds, and topological disconnection, unified into a binary metric $f_{\text{reach}} \in \{0, 1\}$. If $f_{\text{reach}} = 0$, the repair operator $\mathcal{L}_{t+1} = \Upsilon(\mathcal{L}_t, \mathcal{P}, \mathcal{A})$ iteratively moves obstacles or adjusts functional zones until all paths are reachable.
    - **Design Motivation**: Physically valid layouts may still trap robotic arms between workstations. Treating reachability as a hard constraint rather than a post-hoc metric prevents the generation of "correct-looking but unusable" laboratories.

### Loss & Training
LabBuilder is a search-and-verify pipeline rather than a trainable model; thus, there is no gradient-based training. The optimization objective is defined as $\mathcal{L}^\star = \arg\max_\mathcal{L} \mathbb{F}(\mathcal{L}, \mathcal{P}, \mathcal{A})$, where $f_{\text{geo}}$ encodes asset-level geometric constraints and $f_{\text{chem}}$ is derived from hazard annotations in the protocol. Constraints include flammable isolation, reagent storage, separation of incompatible chemicals, and glassware distance from table edges.

## Key Experimental Results

### Main Results
Comparison with Holodeck and SceneWeaver across 30 real chemical experiments (Table 2):

| Method | OB↓ | CN↓ | Asset↑ | Nav↑ | Flam.↑ | Lay↑ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Holodeck | 10.8 | 0.20 | 0.700 | – | 0.239 | 5.61 |
| SceneWeaver | 5.61 | 0.35 | 0.226 | – | 0.097 | 4.57 |
| **LabBuilder** | **0.07** | **0.17** | **0.833** | **0.966** | **0.725** | **9.00** |

Boundary violations (OB) are nearly zero, while chemical safety and asset availability significantly outperform baselines. The LLM semantic score (Lay) reaches 9/10.

### Ablation Study

| Configuration | OB↓ | CN↓ | Asset↑ | Nav↑ | Flam.↑ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Ours (w/o annotation) | 0.25 | 0.36 | 0.786 | 0.952 | — |
| Ours (full) | 0.07 | 0.17 | 0.833 | 0.966 | 0.725 |

Removing asset annotations doubles collisions and increases boundary violations, proving that geometric and chemical semantics in $\mathcal{A}$ are critical for the optimizer.

### Key Findings
- The violation-first criterion for geometric/chemical optimization is crucial: it forces the search to eliminate "illegal geometry" before pursuing "higher semantic scores," preventing oscillation within invalid solution spaces.
- The object count (Obj) is higher than baselines (23.2 vs 10-15), indicating the generator provides all protocol-required instruments rather than simplifying the scene.
- Navigation reachability success is 96.6%. Common failures involve slender instruments (e.g., distillation setups) blocking aisles, suggesting future improvements could use finer shape abstractions for occupancy grids.

## Highlights & Insights
- "Placing executability at the front of the generation loop" represents a significant conceptual shift. Domestic generators fail in labs because they treat chemical safety as an optional metric; this work defines it as a hard constraint integrated into the acceptance criteria.
- The hierarchical + violation-prioritized search provides a reusable LLM-in-the-loop paradigm: algorithmic layers handle "cheap" tasks like geometric conflict resolution, while the LLM is reserved for repairs requiring semantic reasoning, minimizing API costs.
- Using a protocol as an intermediate representation is an elegant decoupling: the upstream handles arbitrary free text, while the downstream only processes structured protocols. Adding new reaction types only requires expanding the experiment library without modifying the generator.

## Limitations & Future Work
- The asset library currently contains only 176 entities. Coverage of niche instruments (e.g., gloveboxes, cryogenic apparatus) is limited, and expanding the library requires continuous annotation costs.
- Chemical safety constraints are currently a discrete set of hard rules, making it difficult to express temporal safety semantics, such as "ventilation status during long-duration reactions."
- Navigation evaluation is limited to point-goal; reachability during robotic arm manipulation (reach + grasp) was not assessed, which may pose issues during physical deployment on dual-arm platforms.

## Related Work & Insights
- **vs. Holodeck**: Holodeck focuses on open-vocabulary indoor scene generation where assets lack functional semantics. LabBuilder's superior OB/Flam. scores suggest that domestic priors do not transfer well to laboratories.
- **vs. SceneWeaver**: SceneWeaver introduces geometric constraint verification but lacks protocol grounding. Its low asset availability (0.226) demonstrates that "geometric correctness" does not equate to "experimental readiness."
- **vs. UP-VLA / Protocol-driven Robots**: This work provides a "protocol to executable environment" bridge for the embodied AI community. Future VLA models could directly consume $\mathcal{P}$ for supervision, eliminating the need for manual scene setup.

## Rating
- Novelty: ⭐⭐⭐⭐ Protocol grounding + systematic chemical hard constraints in scene generation for the first time.
- Experimental Thoroughness: ⭐⭐⭐⭐ 30 experiments + 2 baselines + ablation + navigation evaluation.
- Writing Quality: ⭐⭐⭐⭐ Clear three-module structure with supporting formulas and pseudo-code.
- Value: ⭐⭐⭐⭐⭐ Automated laboratories represent a high-value domain; this work provides a viable environment synthesis solution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] STABLE: Simulation-Ready Tabletop Layout Generation via a Semantics–Physics Dual System](stable_simulation-ready_tabletop_layout_generation_via_a_semantics-physics_dual_.md)
- [\[ICCV 2025\] REPARO: Compositional 3D Assets Generation with Differentiable 3D Layout Alignment](../../ICCV2025/3d_vision/reparo_compositional_3d_assets_generation_with_differentiable_3d_layout_alignmen.md)
- [\[NeurIPS 2025\] PhysX-3D: Physical-Grounded 3D Asset Generation](../../NeurIPS2025/3d_vision/physx-3d_physical-grounded_3d_asset_generation.md)
- [\[ICML 2026\] RelaxFlow: Text-Driven Amodal 3D Generation](relaxflow_text-driven_amodal_3d_generation.md)
- [\[ICML 2026\] PhyScene3D: Physically Consistent Interactive 3D Tabletop Scene Generation](physcene3d_physically_consistent_interactive_3d_tabletop_scene_generation.md)

</div>

<!-- RELATED:END -->

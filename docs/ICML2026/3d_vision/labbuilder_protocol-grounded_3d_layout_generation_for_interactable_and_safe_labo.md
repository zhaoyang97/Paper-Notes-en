---
title: >-
  [Paper Note] LabBuilder: Protocol-Grounded 3D Layout Generation for Interactable and Safe Laboratory
description: >-
  [ICML 2026][3D Vision][Laboratory Scene Generation] LabBuilder compiles free-text experimental descriptions into "asset-chemical protocol" pairs, then employs hierarchical generation…
tags:
  - "ICML 2026"
  - "3D Vision"
  - "Laboratory Scene Generation"
  - "Protocol Implementation"
  - "Chemical Safety"
  - "Navigation Reachability"
  - "Hierarchical Layout"
date: 2026-05-08
content_hash: e4558ddb5691d024
---

# LabBuilder: Protocol-Grounded 3D Layout Generation for Interactable and Safe Laboratory

**Conference**: ICML 2026  
**arXiv**: [2605.02288](https://arxiv.org/abs/2605.02288)  
**Code**: None  
**Area**: 3D Vision / Embodied Environment Generation  
**Keywords**: Laboratory Scene Generation, Protocol Implementation, Chemical Safety, Navigation Reachability, Hierarchical Layout

## TL;DR
LabBuilder compiles free-text experimental descriptions into "asset-chemical protocol" pairs, then employs hierarchical generation, geometric/chemical multi-objective optimization, and navigation repair to produce 3D chemical laboratory layouts that are both visually plausible and executable by robots.

## Background & Motivation
**Background**: Most 3D indoor scene generation focuses on household environments, relying on datasets like 3D-FRONT, with the goal of "looking realistic"—avoiding geometric conflicts and ensuring harmonious furniture arrangements. Recent works use LLMs as layout planners, converting text to structured JSON and rendering interactive scenes.

**Limitations of Prior Work**: This pipeline fails when applied to chemical laboratories. Household scenes only require "fitting and non-overlapping" placements for furniture, while laboratories involve **protocol-level semantics**: reagents must be placed according to reaction types, flammable materials must be far from heat sources, glassware must not be near table edges, and robotic arms must reach all necessary equipment. Household generators treat reagent bottles as decorations, ignoring their chemical properties and failing to verify whether robots can navigate from workstation A to fume hood B.

**Key Challenge**: Existing methods constrain "static geometric validity + visual plausibility" during generation, leaving executability and safety for post-evaluation. However, in laboratories, executability is a design constraint itself—a minor geometric adjustment (e.g., moving a Bunsen burner by 20 cm) can invalidate the entire experimental workflow or even cause safety hazards.

**Goal**: Given a free-text experimental requirement (e.g., "perform an SN2 substitution reaction"), automatically generate a 3D laboratory layout that ensures: (i) all protocol-required assets are instantiated; (ii) geometric conflicts are resolved, and placements comply with wall alignment; (iii) chemical safety constraints are satisfied; (iv) robots can navigate to all required equipment in protocol order.

**Key Insight**: The authors reformulate scene generation as "protocol-grounded constraint optimization"—using LLMs and knowledge bases to convert free text into machine-verifiable structured protocols, which then directly drive layout search and repair, integrating executability into the generation loop.

**Core Idea**: By leveraging "asset knowledge base + chemical knowledge base" as priors, experimental requirements are compiled into schema-based protocols. A three-stage closed-loop process—hierarchical initialization, geometric/chemical violation-prioritized local search, and navigation reachability repair—generates laboratory layouts.

## Method

### Overall Architecture
LabBuilder consists of three tightly coupled modules: **LabForge**, the "frontend compiler," converts free text and heterogeneous assets into structured protocols $\mathcal{P}$ and an asset library $\mathcal{A}$; **LabGen**, the core generator, performs hierarchical initialization to produce candidate layouts $\mathcal{L}_0$, followed by geometric/chemical optimization $\Phi$ and navigation-aware repair $\Upsilon$, outputting the optimal layout $\mathcal{L}^\star$; **LabTouchstone**, the evaluation suite, scores layouts across geometric compliance, feasibility (FSR), chemical safety, and semantic plausibility, supplemented by point-goal navigation evaluation. The pipeline's key innovation is that the protocol $\mathcal{P}$ serves as both a "goal specification" and a "constraint template," guiding the generator on what to place and the optimizer on what constitutes a violation.

### Key Designs

1. **LabForge Protocol Synthesis and Validation**:
    - **Function**: Converts coarse-grained text like "I want to perform a reflux reaction" into verifiable protocols $\mathcal{P}$ containing reagents, instruments, steps, and navigation actions.
    - **Mechanism**: Constructs an asset annotation library for 176 laboratory entities (geometric, semantic, and safety dimensions) and extracts an experimental library covering seven reaction types (substitution, protection/deprotection, condensation, cyclization, redox, functional group transformation, alkylation/acylation) from chemical literature. LLMs perform retrieval-augmented generation on $(x, \mathcal{C})$, producing schema-compliant, asset-normalized protocols, validated against constraints in $\mathcal{A}$. On average, protocols include 5.27 reagents, 9.87 instruments, 9.00 steps, and 4.30 navigation actions.
    - **Design Motivation**: Directly generating "placement JSON" with LLMs leads to physical conflicts and lacks safety semantics. Compiling protocols first adds a schema validator layer, preventing hallucinated assets or missing instruments.

2. **Hierarchical Layout Initialization + Geometric/Chemical Multi-Objective Optimization**:
    - **Function**: Produces feasible initial layouts in a vast continuous configuration space, then refines them to maximize the objective function.
    - **Mechanism**: Decomposes layouts into room-level partitions $(\mathcal{R}, \pi) \sim p_\theta(\cdot \mid x, \mathcal{P}, \mathcal{A})$ (functional zones and 6-DoF poses of large equipment) and desktop-level organizations $\mathcal{D}_s \sim p_\theta(\cdot \mid \cdot, s, \mathcal{R}, \pi)$ (placements of small instruments and reagents on each table), merging into $\mathcal{L}_0$. The objective function $\mathbb{F} = w_{\text{geo}} f_{\text{geo}} + w_{\text{chem}} f_{\text{chem}}$ rewards geometric validity and chemical safety. The search adopts a **violation-prioritized** acceptance criterion: first minimizing hard-constraint violations $v(\mathcal{L})$, then optimizing $\mathbb{F}$. Operators $\Phi$ combine FastRepair for simple geometric conflicts and LLMAdjust for semantic pose adjustments (e.g., "move acetone into the fume hood"). Optimization converges locally at the room level before fine-tuning at the desktop level.
    - **Design Motivation**: Directly generating entire laboratories with LLMs is infeasible. Layered decomposition and violation prioritization constrain combinatorial search to manageable scales ($O(10)$ objects per layer) while reserving LLM calls for high-semantic-difficulty tasks.

3. **Navigation-Aware Repair**:
    - **Function**: Ensures robots can navigate between all protocol steps without collisions.
    - **Mechanism**: Projects 3D scenes into 2D occupancy grids, inflates them by robot radius, and uses $A^\star$ to plan paths for each (start, goal) pair in the protocol. Failures are categorized into endpoint occupation, boundary overflow, and topological disconnection, unified into a binary metric $f_{\text{reach}} \in \{0, 1\}$. If $f_{\text{reach}} = 0$, repair operators $\mathcal{L}_{t+1} = \Upsilon(\mathcal{L}_t, \mathcal{P}, \mathcal{A})$ iteratively adjust blocking objects or functional zones until all paths are reachable.
    - **Design Motivation**: Physically valid layouts may still block robot arms between workstations. Treating reachability as a hard constraint avoids generating "visually correct but unusable" laboratories.

### Loss & Training
LabBuilder is a search-validation pipeline, not a trainable model. The optimization objective is $\mathcal{L}^\star = \arg\max_\mathcal{L} \mathbb{F}(\mathcal{L}, \mathcal{P}, \mathcal{A})$, where $f_{\text{geo}}$ encodes asset-level geometric constraints, and $f_{\text{chem}}$ derives from protocol safety annotations, including flammable material isolation, reagent storage, incompatible chemical separation, and glassware edge distances.

## Key Experimental Results

### Main Results
Comparison on 30 real chemical experiments against Holodeck and SceneWeaver (Table 2):

| Method | OB↓ | CN↓ | Asset↑ | Nav↑ | Flam.↑ | Lay↑ |
|--------|-----|-----|--------|------|--------|------|
| Holodeck | 10.8 | 0.20 | 0.700 | – | 0.239 | 5.61 |
| SceneWeaver | 5.61 | 0.35 | 0.226 | – | 0.097 | 4.57 |
| **LabBuilder** | **0.07** | **0.17** | **0.833** | **0.966** | **0.725** | **9.00** |

Boundary violations are nearly eliminated, with significant improvements in chemical safety and asset usability. LLM semantic scores reach 9/10.

### Ablation Study

| Configuration | OB↓ | CN↓ | Asset↑ | Nav↑ | Flam.↑ |
|---------------|-----|-----|--------|------|--------|
| Ours (w/o annotation) | 0.25 | 0.36 | 0.786 | 0.952 | — |
| Ours (full) | 0.07 | 0.17 | 0.833 | 0.966 | 0.725 |

Removing asset annotations doubles collision counts and significantly increases boundary violations, highlighting the critical role of $\mathcal{A}$'s geometric and chemical semantics.

### Key Findings
- Violation-prioritized optimization is crucial: it eliminates "illegal geometry" first, then optimizes for "better semantics," avoiding oscillations in invalid solution spaces.
- Asset count (23.2 vs. 10-15 in baselines) shows the generator fulfills all protocol requirements without cutting corners.
- Navigation success rate reaches 96.6%, though failures often involve narrow instruments blocking pathways, suggesting future improvements in shape abstraction for occupancy grids.

## Highlights & Insights
- "Embedding executability into the generation loop" represents a paradigm shift. Household generators fail in laboratories because they treat chemical safety as an optional metric; this work encodes it as a hard constraint, directly influencing acceptance/rejection criteria.
- Layered + violation-prioritized search offers a reusable LLM-in-the-loop framework: cheap tasks like geometric conflicts are handled algorithmically, reserving LLM calls for high-semantic-difficulty repairs.
- Protocols as intermediate representations elegantly decouple upstream free-text inputs from downstream schema-based layouts, enabling easy extension to new reaction types by updating the experiment library without modifying the generator.

## Limitations & Future Work
- The asset library currently includes only 176 entities, with limited coverage of rare instruments (e.g., gloveboxes, cryogenic setups), requiring ongoing annotation efforts.
- Chemical safety constraints are discrete hard rules, lacking temporal dimensions for long-duration reactions (e.g., ventilation states).
- Navigation evaluation focuses on point-goal tasks, omitting reachability and graspability during robotic arm operations, which may reveal issues in dual-arm platforms.

## Related Work & Insights
- **vs Holodeck**: Holodeck targets open-vocabulary indoor scene generation, with assets lacking functional semantics. LabBuilder outperforms in OB/Flam., demonstrating the ineffectiveness of household priors in laboratories.
- **vs SceneWeaver**: SceneWeaver incorporates geometric constraint validation but lacks protocol grounding, achieving only 0.226 asset usability, with most experiments failing, highlighting that "geometric validity" ≠ "experimental feasibility."
- **vs UP-VLA / Protocol-Driven Robots**: This work bridges "protocol → executable environment" for embodied intelligence, enabling VLA models to directly consume $\mathcal{P}$ for supervision, bypassing manual scene scripting.

## Rating
- Novelty: ⭐⭐⭐⭐ Protocol grounding + chemical hard constraints are systematically introduced into scene generation for the first time.
- Experimental Thoroughness: ⭐⭐⭐⭐ 30 experiments + three baselines + ablation + navigation evaluation provide comprehensive coverage.
- Writing Quality: ⭐⭐⭐⭐ Clear modular structure, with well-presented formulas and pseudocode.
- Value: ⭐⭐⭐⭐⭐ Automated laboratories address a real-world demand, offering a deployable environment synthesis solution.

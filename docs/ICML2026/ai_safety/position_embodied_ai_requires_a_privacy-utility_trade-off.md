---
title: >-
  [Paper Note] Position: Embodied AI Requires a Privacy-Utility Trade-off
description: >-
  [ICML 2026][AI Safety][embodied AI] This position paper argues that privacy in embodied AI cannot be resolved with stage-local patches; it must be treated as an architecture-level dynamic control signal spanning the entire lifecycle of instruction, perception, planning, and interaction. The authors propose the SPINE framework, which utilizes an L1-L4 pri
tags:
  - ICML 2026
  - AI Safety
  - embodied AI
date: 2026-05-08
content_hash: 225df001544970d4
---
# Position: Embodied AI Requires a Privacy-Utility Trade-off

**Conference**: ICML 2026  
**arXiv**: [2605.05017](https://arxiv.org/abs/2605.05017)  
**Code**: https://github.com/rminshen03/EAI_Privacy_Position  
**Area**: AI Safety / Embodied AI / Privacy Protection  
**Keywords**: embodied AI, privacy-utility trade-off, SPINE framework, lifecycle privacy, hierarchical control

## TL;DR
This position paper argues that privacy in embodied AI cannot be resolved with stage-local patches; it must be treated as an architecture-level dynamic control signal spanning the entire lifecycle of instruction, perception, planning, and interaction. The authors propose the SPINE framework, which utilizes an L1-L4 privacy classification matrix to coordinately adjust agent behavior at every stage.

## Background & Motivation
**Background**: Embodied AI (EAI) is rapidly transitioning from simulation to real-world environments such as homes, hospitals, and offices. Existing research primarily focuses on optimizing task success rates within the four internal stages: instruction understanding, environment perception, action planning, and physical interaction.

**Limitations of Prior Work**: Current EAI privacy protections are almost entirely stage-local patches—such as face blurring during perception or adding perturbations during planning. However: (1) these patches are often "reversed" by downstream stages; for instance, while perception might anonymize a face, planning logs might record the precise movement patterns of a user taking tremor medication, allowing the inference of Parkinson's disease history. (2) The privacy-utility trade-off is a non-linear safety constraint; aggressive planning restrictions do not merely decrease efficiency but can directly lead to robots colliding with walls or people.

**Key Challenge**: Privacy in EAI is essentially a property that accumulates across stages and time, yet current architectures treat it as an independently controllable local feature. Furthermore, legal frameworks like GDPR/CCPA provide high-level principles but lack actionable guidance for the "embodied closed-loop," creating a gap between technology and regulation.

**Goal**: (1) Demonstrate why privacy must be treated as a lifecycle-level architectural constraint; (2) Design a unified framework capable of consistently propagating privacy constraints across stages and dynamically adjusting trade-offs in different contexts; (3) Provide preliminary evidence of how privacy reshapes downstream utility through real-world case studies.

**Key Insight**: The authors use embodied navigation as a controlled probe because navigation naturally couples the four stages, allowing for the controlled observation of "how downstream utility changes under upstream strong privacy," transforming the trade-off from an abstract slogan into a quantifiable engineering relationship.

**Core Idea**: Upgrade privacy from "local patches" to a "dynamic control signal," implementing a "context-aware" privacy architecture via a four-level privacy classification matrix and cross-stage orchestration.

## Method
As a position paper, the core contribution is a conceptual framework named SPINE. It is not a set of training algorithms but a design blueprint that instructs engineers on "which privacy primitives to activate in which context and at which stage," using two navigation case studies to map the abstract trade-offs onto quantifiable curves.

### Overall Architecture
SPINE consists of three components. The first is an L1-L4 four-level privacy classification matrix responsible for mapping any real-world scenario to a privacy level. The second is a $4 \times 4$ conceptual architecture diagram where the vertical axis represents the four stages of Embodied AI (Instruction / Perception / Planning / Interaction) and the horizontal axis represents the L1-L4 levels, with each cell specifying the technical primitives to be activated. The third is a cross-stage orchestration strategy that links the four stages into a holistic pipeline based on a "highest-triggering-criterion" rule, accompanied by a quantitative analysis of utility degradation relative to privacy intensity. Together, these components transform privacy from scattered patches into a dynamic control signal pervading the entire pipeline.

### Key Designs

**1. Multi-criterion Privacy Classification Matrix (L1-L4): Upgrading the "sensitive/insensitive" binary into a formal privacy state machine.** Traditional methods offer only public and private settings, which are too coarse to distinguish between scenarios with vastly different sensitivities, such as a "bedroom" versus a "private office." SPINE uses a unified quadruple $PL = \{S, I, C, \Phi\}$ to describe each level of privacy state: $S$ is the scene context, $I$ is the permitted information flow, $C$ is the enforced control primitive, and $\Phi$ is the dominant utility goal. The levels from low to high are: L1 (Public, e.g., a park) allowing cloud inference and full sensing where $\Phi$ is max utility; L2 (Internal, e.g., an office corridor) using hybrid information flows that remove biometrics but retain geometric data; L3 (Confidential, e.g., a private office) switching to local processing with semantic de-identification and privacy-aware routing; L4 (Restricted, e.g., a bedroom or bathroom) retaining only minimal viable safety functions, replacing RGB with LiDAR, and using TEE containers for isolation. This shared "privacy state machine" ensures cross-stage consistency, while high-cost primitives (FHE/ZKP) are explicitly restricted to L4 to avoid performance penalties in low-sensitivity scenarios.

**2. Adaptive Privacy Orchestration: Ensuring end-to-end privacy constraints and preventing downstream "reversion."** This design defines specific actions for instruction, perception, planning, and interaction at each level. For example, in the perception stage: L1 uses full FoV RGB-D, L2 performs real-time anonymization of faces/license plates, L3 dynamically masks non-task regions and restricts field-of-view, and L4 cuts RGB entirely in favor of LiDAR. Similarly, in the planning stage: L1 follows the shortest path, L2 plans on de-identified semantic maps, L3 introduces a "privacy cost map" to add high traversal penalties to private areas, and L4 degrades to minimum viable navigation. These are linked vertically by the "highest-triggering-criterion" rule—if any stage triggers a higher level constraint, the entire pipeline immediately upgrades until the condition is cleared or a manual audit occurs. This eliminates the "stage-local patch" flaw: in previous systems, perception might anonymize faces while planning logs still leaked identity through movement patterns; in SPINE, if the robot perceives it has entered a bedroom, the entire pipeline (including planning and logging) switches to L4.

**3. Threat Model & Privacy-Utility Boundary Quantification: Providing a tunable knob and a clear failure threshold for the trade-off.** To guide engineering, SPINE identifies three threat categories: honest-but-curious cloud providers, compromised storage or insiders, and external/unauthorized observers. It then defines the trade-off as a function of utility loss relative to privacy intensity. In the navigation case study, the authors use pixelation intensity $K$ as the trade-off knob: $K=1$ corresponds to the L1 original image, while increasing $K$ approaches L3. This reveals a curve where task success rate decreases monotonically with $K$. Crucially, an "operational boundary" exists on this curve—once $K$ crosses a threshold, the task fails entirely. This boundary represents the maximum privacy that can be enforced for that scenario, enabling data-driven decisions rather than arbitrary choices.

The SPINE framework itself does not have an end-to-end training objective. The two case studies were executed using existing EAI simulators and real robots, recording success rates and path lengths under different $K$ values to generate the trade-off curves.

## Key Experimental Results
This is a position paper; it provides conceptual validation rather than exhaustive experimental comparisons.

### Main Results
Comparison between SPINE and stage-local patches using the 4-stage × 4-level architecture:

| Privacy Level | Typical Scenario | Instruction | Perception | Planning | Interaction |
|------|------|------|------|------|------|
| L1 Public | Park | Cloud LLM | Full FoV RGB-D | Shortest Path | Full Logging |
| L2 Internal | Office Corridor | Local Logging | Real-time Anonymization | De-identified Semantic Map | Standard Latency, De-identified Storage |
| L3 Confidential | Private Office | Semantic De-identification | Restricted FoV + Mask | Privacy Cost Map + Routing | Session-only Encrypted Logs |
| L4 Restricted | Bedroom / Bathroom | TEE Processing | LiDAR only (No RGB) | Minimum Viable Nav | Trace-free Volatile Execution |

### Ablation Study
Evaluation of task success rate and path length degradation under different pixelation intensities $K$ in the navigation case study:

| Config | Privacy Level | Success Rate | Path Length | Description |
|------|------|------|------|------|
| $K=1$ (Original) | L1 | Baseline High | Baseline Short | No privacy constraints |
| $K$ Medium | L3 | Moderate Drop | Slight Increase | Semantic loss but task completes |
| $K$ High | L4 Boundary | Significant Drop | Large Increase | Approaching operational boundary |
| Beyond Boundary | Infeasible | Failure | Unreachable | Task cannot be completed |

### Key Findings
- Stage-local privacy patches can be "reversed" downstream: anonymizing faces in perception does not stop planning logs from revealing identity or health status through movement patterns, necessitating a lifecycle perspective.
- The privacy-utility relationship is non-linear; an "operational boundary" exists beyond which the task fails, which must be a key consideration during deployment.
- Current static privacy policies often fail in real-world deployments because they cannot adapt to context, highlighting the need for dynamic classification mechanisms.

## Highlights & Insights
- Framing "privacy as a dynamic control signal" elevates privacy from a compliance issue to a system control problem, allowing seamless integration with control theory and safety filters.
- The L1-L4 quadruple $\{S, I, C, \Phi\}$ provides a formal structure for privacy levels that is far superior to the common "sensitive/insensitive" binary used in industry, directly aiding SDK design.
- The "highest-triggering-criterion" rule draws from priority inheritance in real-time systems; it is simple but effectively solves responsibility passing between stages.
- Using navigation as a controlled probe to quantify trade-offs is strategic, as navigation naturally integrates all four stages and has mature utility metrics (success rate, path length).

## Limitations & Future Work
- The framework remains conceptual; case studies are limited to navigation and household pixelation, lacking coverage for more complex scenarios like manipulation or medical assistance.
- How to define the specific thresholds for L1-L4 levels remains unclear; there is a risk that more levels could lead to over-conservatism and poor utility.
- The "highest-triggering-criterion" could cause the system to stay locked in L4 during multi-tasking, requiring clearing mechanisms and refined audit logs.
- While FHE/ZKP are mentioned, no computational budget analysis is provided; such primitives usually become system bottlenecks in real deployments.

## Related Work & Insights
- **vs. Pape et al. (Prompt Obfuscation)**: They focus on single-turn LLM privacy; this paper expands the perspective to the entire closed-loop of Embodied AI, highlighting the systemic issue where downstream stages can reverse upstream masking.
- **vs. Legal Frameworks (GDPR/CCPA)**: Law provides principles but not staged operational guidance; this paper uses a four-level matrix to map high-level compliance to specific technical primitives for each EAI stage.
- **vs. Classical Differential Privacy (DP)**: DP provides mathematical guarantees for data release; this paper emphasizes real-time, context-aware policy switching during deployment, which is more aligned with the needs of embodied agents.

## Rating
- Novelty: ⭐⭐⭐⭐ The "lifecycle privacy as control signal" framing is a pioneering systematic approach in EAI literature.
- Experimental Thoroughness: ⭐⭐⭐ Only includes navigation + pixelation cases; lacks diverse validation in manipulation or healthcare.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, moving logically from problem definition to classification, orchestration, and case studies.
- Value: ⭐⭐⭐⭐ Provides a reusable blueprint for designing privacy architectures for embodied robots and home service agents.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Mitigating Privacy-Utility Trade-off in Decentralized Federated Learning via f-Differential Privacy](../../NeurIPS2025/ai_safety/mitigating_privacy-utility_trade-off_in_decentralized_federated_learning_via_f-d.md)
- [\[ICML 2025\] Clients Collaborate: Flexible Differentially Private Federated Learning with Guaranteed Improvement of Utility-Privacy Trade-off](../../ICML2025/ai_safety/clients_collaborate_flexible_differentially_private_federated_learning_with_guar.md)
- [\[ICML 2026\] Position: Machine Learning for Heart Transplant Allocation Policy Optimization Should Account for Incentives](position_machine_learning_for_heart_transplant_allocation_policy_optimization_sh.md)
- [\[ICML 2026\] Position: Beyond Sensitive Attributes, ML Fairness Should Quantify Structural Injustice via Social Determinants](position_beyond_sensitive_attributes_ml_fairness_should_quantify_structural_inju.md)
- [\[ICML 2026\] Persuasive Privacy](persuasive_privacy.md)

</div>

<!-- RELATED:END -->

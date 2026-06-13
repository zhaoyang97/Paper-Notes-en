---
title: >-
  [Paper Note] Out of Control -- Why Alignment Needs Formal Control Theory (and an Alignment Control Stack)
description: >-
  [NeurIPS 2025][Interpretability][AI alignment] This position paper argues for formal optimal control theory as a foundational tool for AI alignment research…
tags:
  - "NeurIPS 2025"
  - "Interpretability"
  - "AI alignment"
  - "control theory"
  - "alignment control stack"
  - "formal verification"
  - "stochastic optimal control"
date: 2026-05-08
content_hash: 6c16cc5c0c60f1c7
---

# Out of Control -- Why Alignment Needs Formal Control Theory (and an Alignment Control Stack)

**Conference**: NeurIPS 2025
**arXiv**: [2506.17846](https://arxiv.org/abs/2506.17846)  
**Code**: None  
**Area**: Interpretability
**Keywords**: AI alignment, control theory, alignment control stack, formal verification, stochastic optimal control

## TL;DR
This position paper argues for formal optimal control theory as a foundational tool for AI alignment research, and proposes the Alignment Control Stack (ACS)—a ten-layer hierarchical framework spanning from the physical hardware layer to the social governance layer—for systematically organizing and analyzing measurement, control, and interoperability across different alignment methods.

## Background & Motivation

**Background**: Current AI alignment research encompasses methods such as RLHF, Constitutional AI, red-teaming, and mechanistic interpretability, yet most rely on empirical observation and lack mathematically formal guarantees.

**Limitations of Prior Work**:
   - **Formalisation problem**: Alignment research—including technically oriented work—tends to be limited to empirical observations on specific models, lacking the formal proofs common in the control literature, making generalization and rigorous comparison across techniques difficult.
   - **Coordination problem**: Different alignment approaches (spanning theory, engineering, and governance) lack a clear hierarchical control taxonomy, precluding systematic analysis of interoperability and potential conflicts among control mechanisms.

**Key Challenge**: Alignment methods are fragmented and empirical, with no unified formal framework for organizing and proving safety properties; control measures across different layers lack systematic integration.

**Goal**:
   - Provide a hierarchical taxonomy (the Alignment Control Stack) to systematize alignment research.
   - Demonstrate how formal control theory can supplement the shortcomings of existing alignment methods.

**Key Insight**: Examining AI alignment through the lens of classical control theory—a discipline with a century of success in aerospace, chemical engineering, and related fields. AI systems can likewise be treated as dynamical systems requiring measurement, estimation, and control.

**Core Idea**: Organize alignment research via a layered control stack, and fill the formalization gap in alignment methods using formal optimal control theory.

## Method

### Overall Architecture
The authors present two primary contributions: (1) the Alignment Control Stack (ACS), a ten-layer hierarchical framework describing control points across the AI system lifecycle; and (2) formal control theory, demonstrating how tools such as LQR, LQG, and game theory can be applied at different levels of the stack.

### Key Designs

1. **Alignment Control Stack (ACS)**:

    - Function: Decomposes an AI system into ten layers, ordered from bottom to top: physical infrastructure (Layers 1–2) → AI frameworks and model architecture (Layers 3–4) → training and behavioral output (Layers 5–6) → interpretability and alignment (Layers 7–8) → multi-agent systems and social governance (Layers 9–10).
    - Mechanism: Each layer has its own state variables, measurable signals, and available control actions. Lower layers (1–4) admit physical- or software-level control using linear or hybrid methods; middle layers (5–7) involve high-dimensional, partially observable learning dynamics requiring stochastic and robust control; upper layers (8–10) integrate technical control with human governance.
    - Design Motivation: Vertical coupling exists between layers—actions at lower layers propagate upward as parameter changes or distributional drift, while errors detected at higher layers must be corrected via downward interventions. This necessitates hierarchical control synthesis: local controllers at each layer guarantee layer-level performance while exposing interface variables for use by higher layers.

2. **Vertical Alignment**:

    - Function: Models interactions between different layers and demonstrates how an LQG framework can jointly optimize the training layer (Layer 5) and the behavioral output layer (Layer 6).
    - Mechanism: Layer 5 is modeled as a deterministic control problem with model parameters $w_t$ as the state and dynamics $w_{t+1} = w_t - u_t + \xi_t$ (SGD with noise); Layer 6 is modeled with stochastic observations $y_t = c^\top w_t + v_t$. The combined system yields an augmented formulation solved via discrete-time LQG—Riccati recursion provides the optimal feedback gain $K_w$, and a Kalman filter provides optimal state estimation. The separation principle guarantees that $(L, K_w)$ are jointly optimal.
    - Design Motivation: To demonstrate that even when only behavioral outputs are observable (rather than weights directly), formal control can provide optimal closed-loop policies—precisely the core challenge of partial observability in alignment.

3. **Horizontal Alignment**:

    - Function: Models interactions among multiple AI systems (or multiple stacks), using dynamic game theory to analyze cooperative or adversarial scenarios.
    - Mechanism: Two AI stacks are modeled as coupled stochastic differential equations $dx_A = f_A(t, x_A, u_A, x_B, u_B)dt + \sigma_A dW_A$, each with its own objective functional $J_A, J_B$. Nash equilibrium strategies are obtained via the Hamilton–Jacobi–Isaacs (HJI) equation.
    - Design Motivation: To address alignment in multi-agent settings where multiple AIs may exhibit collusion, steganographic communication, or resource competition—phenomena that require a game-theoretic framework for analysis and mechanism design.

### Control-Theoretic Toolbox
The authors enumerate control-theoretic tools applicable at different layers:
- **Offline control** (dynamic programming, Pontryagin's maximum principle) → architecture design and training initialization
- **Online control** (adaptive control, MPC) → behavioral monitoring and real-time intervention
- **Stochastic control** → handling gradient noise and the probabilistic nature of LLM outputs
- **Robust control** ($\mathcal{H}_\infty$) → maintaining stability under uncertainty
- **Game-theoretic control** → multi-agent and adversarial scenarios

## Key Experimental Results

### Primary Analytical Framework
This is a position paper with no conventional experiments. The core contributions are the theoretical framework and proof-of-concept analysis:

| Alignment Method Category | Existing Limitations | Control-Theoretic Supplement |
|---|---|---|
| RLHF / preference learning | Lacks formal guarantees; reward hacking | Formalized objective functions, constrained optimization, robust MPC |
| Red-teaming | Provides safety evidence only under specific conditions | Stability analysis, reachability analysis, worst-case guarantees |
| Constitutional AI | Rule enforcement lacks verifiability | Formal verification, controller synthesis |
| Mechanistic interpretability | Reveals internals but lacks tools for intervention design | Observability analysis, optimal intervention design |
| Multi-agent alignment | Lacks interaction modeling | Dynamic game theory, Nash equilibrium analysis |

### ACS Layer-wise Control Comparison

| Layer | Control Type | Applicable Tools | Complexity |
|------|---------|-------------|--------|
| Layers 1–2 (Physical) | Deterministic | Linear control, PID | Low |
| Layers 3–4 (Architecture) | Hybrid | Geometric control, hybrid systems | Medium |
| Layers 5–6 (Training / Behavior) | Stochastic | LQG, stochastic optimal control | High |
| Layers 7–8 (Interpretability / Alignment) | Partially observable | POMDP, adaptive control | Very high |
| Layers 9–10 (Multi-agent / Social) | Game-theoretic | Dynamic games, mechanism design | Extremely high |

### Key Findings
- The primary advantage of formal control lies in providing **provable safety guarantees** rather than merely empirical safety assessments.
- Vertical coupling (inter-layer dependencies) implies that single-layer optimization is insufficient—hierarchical control synthesis is required.
- Nash equilibrium analysis of horizontal coupling (multi-system interactions) can anticipate and mitigate alignment risks such as collusion.

## Highlights & Insights
- **Hierarchical thinking**: The ACS transforms the alignment problem from an undifferentiated collection of approaches into a structured, layered engineering problem, where each layer has well-defined state, control, and observation variables. This offers meaningful guidance for practical deployment. Hierarchical decomposition is a cornerstone method in control engineering for managing complex systems.
- **Inspirational value of the LQG toy model**: Though simplified, the idea of jointly optimizing training and behavior via LQG in vertical alignment is conceptually novel. The separation principle guarantees that the observer and controller can be designed independently, suggesting that "monitoring" and "intervention" in alignment may also be amenable to decoupled design.
- **Interoperability perspective**: This paper is among the first to systematically raise the interoperability problem among alignment methods—different approaches may be synergistic or conflicting, a practically important issue that is rarely discussed.
- **Game theory and multi-agent alignment**: The HJI equation framework for horizontal alignment provides a mathematical foundation for analyzing alignment among multiple AI systems, transferable to any scenario involving interacting autonomous agents (e.g., AI marketplaces, multi-agent collaboration).

## Limitations & Future Work
- **No empirical validation**: As a position paper, all frameworks remain conceptual. The LQG toy model is overly simplified (linear Gaussian assumptions) and cannot be directly applied to real LLMs.
- **Scalability concerns**: The authors themselves acknowledge that the computational feasibility of classical control theory for LLMs with billions of parameters is a serious challenge. The curse of dimensionality in high-dimensional stochastic optimal control is not genuinely resolved.
- **Unclear boundaries of formalization**: The paper does not provide a clear feasibility analysis of which layers can be effectively formalized and which are inherently difficult to formalize (e.g., the social governance layer).
- **Unclear integration pathways with existing methods**: While the paper argues that control theory should "complement" existing methods, concrete technical roadmaps for embedding RLHF, Constitutional AI, and similar approaches into the control-theoretic framework are absent.
- **Future directions**: A productive next step would be to select one or two layers from the ACS (e.g., Layers 5–6) and validate the control-theoretic approach using real LLM training data—for instance, applying MPC to online hyperparameter tuning in RLHF and comparing against conventional methods.

## Related Work & Insights
- **vs. Greenblatt et al. (AI Control)**: That work designs empirical control protocols (e.g., monitoring + auditing); this paper advocates for formalizing such protocols via control theory to provide mathematical guarantees. The ACS framework can situate AI Control protocols within specific layers for rigorous analysis.
- **vs. Mechanistic Interpretability**: Mechanistic interpretability focuses on "understanding internal circuits," while control theory addresses "how to use that understanding to design optimal interventions." The two are complementary—interpretability provides state estimation; control theory provides intervention strategy.
- **vs. Constitutional AI**: Constitutional AI constrains behavior through rules; control theory employs formally constrained optimization to ensure those rules are enforced in a verifiable manner.

## Rating
- Novelty: ⭐⭐⭐⭐ Systematically applying control theory to alignment research is a valuable perspective; the ACS framework introduces structured novelty.
- Experimental Thoroughness: ⭐⭐ No experiments as a position paper; the LQG toy model is overly simplified.
- Writing Quality: ⭐⭐⭐⭐ Argumentation is clear and well-organized; mathematical derivations are rigorous, though the paper is somewhat lengthy.
- Value: ⭐⭐⭐ Provides a useful conceptual framework, but substantial work remains before practical application.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Probabilistic Token Alignment for Large Language Model Fusion](probabilistic_token_alignment_for_large_language_model_fusion.md)
- [\[AAAI 2026\] SCoPe: Intrinsic Semantic Space Control for Mitigating Copyright Infringement in LLMs](../../AAAI2026/interpretability/scope_intrinsic_semantic_space_control_for_mitigating_copyright_infringement_in_.md)
- [\[NeurIPS 2025\] VL-SAE: Interpreting and Enhancing Vision-Language Alignment with a Unified Concept Set](vlsae_interpreting_and_enhancing_visionlanguage_alignment_wi.md)
- [\[ICLR 2026\] PERSONA: Dynamic and Compositional Inference-Time Personality Control via Activation Vector Algebra](../../ICLR2026/interpretability/persona_dynamic_and_compositional_inference-time_personality_control_via_activat.md)
- [\[ICLR 2026\] SALVE: Sparse Autoencoder-Latent Vector Editing for Mechanistic Control of Neural Networks](../../ICLR2026/interpretability/salve_sparse_autoencoder-latent_vector_editing_for_mechanistic_control_of_neural.md)

</div>

<!-- RELATED:END -->

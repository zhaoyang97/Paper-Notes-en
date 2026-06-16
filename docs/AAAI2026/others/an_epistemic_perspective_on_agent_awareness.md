---
title: >-
  [Paper Note] An Epistemic Perspective on Agent Awareness
description: >-
  [AAAI 2026][epistemic logic] This paper is the first to treat agent awareness as a form of knowledge, distinguishing two awareness modalities — de re (concerning physical objects) and de dicto (concerning concepts/descri…
tags:
  - "AAAI 2026"
  - "epistemic logic"
  - "agent awareness"
  - "de re/de dicto"
  - "2D semantics"
  - "completeness proof"
date: 2026-05-08
content_hash: d3aed0eb426fc82f
---

# An Epistemic Perspective on Agent Awareness

**Conference**: AAAI 2026
**arXiv**: [2511.05977v1](https://arxiv.org/abs/2511.05977v1)  
**Code**: None  
**Area**: Other
**Keywords**: epistemic logic, agent awareness, de re/de dicto, 2D semantics, completeness proof

## TL;DR

This paper is the first to treat agent awareness as a form of knowledge, distinguishing two awareness modalities — de re (concerning physical objects) and de dicto (concerning concepts/descriptions) — and proposes a sound and complete logical system grounded in 2D semantics to characterize the interaction between these two modalities and the standard "factual knowledge" modality.

## Background & Motivation

As AI agents increasingly participate in high-stakes decisions affecting human lives, correct decision-making often depends on awareness of the existence of other agents. For example:
- **Combat robots** must minimize casualties upon becoming aware of civilians
- **Autonomous vehicles** must stop at yield signs upon becoming aware of oncoming traffic
- **Medical AI** must provide assistance upon becoming aware that someone is ill
- **Value-aligned robots** must apologize upon becoming aware that someone has been offended

However, "awareness" is an ambiguous term treated as a standalone concept in existing literature. The Cambridge Dictionary defines it as "knowledge that something exists," implying an epistemic interpretation. This paper provides a formal account of awareness from precisely this epistemic perspective.

## Core Problem

1. How can two distinct forms of agent awareness be formally distinguished?
2. How can a logical system be constructed to reason about both forms of awareness and their relationship to standard knowledge?
3. Is such a logical system sound and complete?

## Method

### Overall Architecture

The paper constructs an epistemic logic system based on **egocentric logic** and **2D semantics**, comprising three modal operators:

- **K φ** ("knows φ about herself"): standard self-knowledge modality, expressing that an agent knows property φ about itself
- **R φ** ("de re aware"): de re awareness modality, expressing that an agent, as a physical object, is aware of some agent with property φ
- **D φ** ("de dicto aware"): de dicto awareness modality, expressing that an agent is aware, at the conceptual level, of some agent with property φ

### Key Designs

1. **Dual-Modality Distinction between De Re and De Dicto Awareness**:
    - Function: Decomposes "awareness" into two essentially distinct modal operators $R\varphi$ (de re) and $D\varphi$ (de dicto), forming a three-modality epistemic system together with the self-knowledge operator $K\varphi$
    - Mechanism: $R\varphi$ expresses that an agent, as a physical object, perceives some agent with property $\varphi$ — that agent holds $\varphi$ in the current world and exists in all indistinguishable worlds ("seen but not identified"); $D\varphi$ expresses that in every indistinguishable world there exists at least one agent with property $\varphi$ ("the concept is known but the referent is unidentified"). Illustrative example: Ann seeing an unmarked police car is de re awareness; receiving a WeRide text message and knowing an autonomous vehicle is nearby is de dicto awareness
    - Design Motivation: Existing epistemic logics typically treat awareness as a single concept and cannot distinguish between "perceiving a physical object without knowing its properties" and "possessing a concept without being able to locate the referent." The dual-modality distinction fills this expressive gap

2. **Epistemic Models Based on 2D Semantics and the Ternary Satisfaction Relation**:
    - Function: Provides a rigorous model-theoretic semantic foundation for the three-modality system
    - Mechanism: An epistemic model is defined as a quintuple $(W, A, P, {\sim}, \pi)$, where $P \subseteq A \times W$ is an existence relation (specifying in which worlds an agent appears) and $\sim_a$ is an indistinguishability equivalence relation. The key innovation is the adoption of a ternary satisfaction relation $w, a \Vdash \varphi$ (world $w$, agent $a$, formula $\varphi$), drawing on 2D semantics to simultaneously capture information along both the world dimension and the agent dimension, so that the semantics of $R$ and $D$ can naturally quantify over an agent's cross-world existence and property-holding respectively
    - Design Motivation: The binary satisfaction relation of standard Kripke semantics cannot simultaneously track "which agent is perceived" and "in which world it is perceived." The two-dimensional structure of 2D semantics precisely matches the requirements of the de re/de dicto distinction

3. **Axiomatic System and Completeness**:
    - Function: Provides a sound and complete deductive system for the three-modality epistemic logic
    - Mechanism: The system comprises 8 axioms and 4 inference rules. Core axioms include: $K\varphi \to \varphi$ (Truth); $\varphi \to R\varphi$ and $K\varphi \to D\varphi$ (Self-Awareness, connecting the three operators); $D\varphi \to KD\varphi$ (introspection of de dicto awareness); $R(\varphi \lor \psi) \to R\varphi \lor R\psi$ (disjunctivity of de re, reflecting its "directed at a concrete object" character); $D(R\varphi \lor D\varphi) \to D\varphi$ (General Awareness, unifying the two forms of awareness). Inference rules include Modus Ponens, Necessitation ($\varphi \vdash K\varphi$), and Monotonicity rules for $D$ and $R$
    - Design Motivation: The axioms characterize the hierarchical relationships among $K$, $R$, and $D$ — $K$ entails $D$, self-instantiated properties entail $R$ — while de re is disjunctive and de dicto is not, precisely reflecting the structural asymmetry between the two forms of awareness

### Loss & Training

This paper is a purely theoretical/formal verification work involving no loss functions or training strategies. The core technical contribution is the **completeness proof**, which employs an enhanced "matrix" technique:

- **Frame Construction**: Frames are defined as partially constructed models containing an explicit awareness relation ↝
- **λ-assured Sets**: The notion of λ-assured is introduced to handle the "ghost spy" phenomenon in model construction — only agents that are absolutely undetectable are consistent with a dataset
- **Complete Frames**: Finite frames are incrementally extended (by adding new worlds/agents) to satisfy five categories of completeness requirements
- **Canonical Model**: A canonical model is constructed from complete frames, and the Truth Lemma is proved ($\varphi \in X^a_w \iff w, a \Vdash \varphi$)

## Key Experimental Results

This paper is purely theoretical; the main results are two theorems:

| Theorem | Statement | Significance |
|---------|-----------|--------------|
| Theorem 1 (Soundness) | If $\vdash\varphi$, then $w, a \Vdash \varphi$ for all worlds $w$ and agents $a$ in every epistemic model | The axiomatic system derives no false conclusions |
| Theorem 2 (Strong Completeness) | If $X \nvdash \varphi$, then there exists an epistemic model in which all formulas in $X$ are true but $\varphi$ is false | The axiomatic system suffices to derive all semantically valid formulas |

### Ablation Study

As a theoretical work, the paper validates the necessity of each axiom through the following observations:

- The justification for the **Self-Awareness axiom** follows from the model design: every agent necessarily exists in all worlds in which it appears
- **Introspection of Awareness** holds only for $D$ (de dicto) and not in general for $R$ (de re) — a significant asymmetry
- **Disjunctivity** holds only for $R$ (guaranteed by the existential quantifier structure of its semantics) and not for $D$
- The **General Awareness axiom** connects $R$ and $D$, named after the "general awareness" abbreviation $A\varphi = R\varphi \lor D\varphi$

## Highlights & Insights

1. **Conceptual Innovation**: This is the first work to recast awareness from an independent concept into a subtype of knowledge — a philosophically more elegant and practically more operational perspective
2. **Precise Formalization of De Re/De Dicto**: The paper captures a distinction that inherently requires quantification using a quantifier-free modal logic rather than first-order epistemic logic
3. **Well-Crafted Running Example**: The scenario involving Ann, WeRide, and the police car renders abstract logical concepts intuitive
4. **Introduction of λ-assured Sets**: Elegantly addresses the technical difficulty of "loss of awareness when new worlds are added" during frame construction
5. **Innovation in the Matrix Technique for Completeness**: Extends existing techniques by incorporating the awareness relation and row labels, resolving the "decoupling" of worlds and agents inherent in 2D semantics

## Limitations & Future Work

1. **Absence of Computational Complexity Analysis**: The paper does not discuss the complexity of the modal satisfiability problem or model checking
2. **Static Logic**: Dynamic updates are not considered (e.g., dynamic logic extensions for changes in awareness due to information acquisition or forgetting)
3. **Transworld Identity Assumption**: The paper assumes transworld identity, which is itself a contested topic in the philosophy of language
4. **Single-Agent Perspective**: Although the model contains multiple agents, the modalities $K$, $R$, and $D$ all pertain to properties of the "current agent," without directly characterizing multi-agent interaction reasoning
5. **Lack of Applied Validation**: The paper does not demonstrate the application of this logical system in real AI systems (e.g., autonomous driving decisions) or a model checking implementation
6. **Integration with Probability/Uncertainty**: In practice, awareness is often gradual rather than binary, which two-valued logic cannot fully capture

## Related Work & Insights

| Work | Focus | Distinction from This Paper |
|------|-------|-----------------------------|
| Fagin & Halpern (1987) | Conceptual awareness (awareness of concepts) | This paper concerns agent awareness (awareness of other agents) |
| Board & Chung (2021, 2022) | Object-based unawareness | Does not distinguish de re/de dicto |
| Epstein, Naumov & Tao (2023) | De re/de dicto "know who" | Uses quantifiers; cannot express awareness |
| Epistemic Logic with Assignments (Wang & Seligman 2018) | General epistemic logic with assignments | More general but not specific to awareness |
| Jiang & Naumov (2025) | De re/de dicto in data anonymization | Focuses on dataset property inference, not awareness |
| Naumov & Tao (2023) | Completeness of "telling apart" modality | No awareness modality; this paper builds on their matrix technique with significant extensions |

The unique contributions of this paper are: (1) the first proposal of quantifier-free awareness modalities $R$ and $D$; and (2) the introduction of the awareness relation and λ-assured conditions in the completeness proof.

The work carries several broader implications:
1. **AI Safety**: Provides a formal verification framework for the "perception–decision" pipeline in systems such as autonomous driving — precisely specifying "under what awareness conditions should the system take what action"
2. **Multi-Agent Systems**: Extendable to awareness reasoning in multi-agent cooperation/game settings, e.g., higher-order awareness such as "I know that you know I am here"
3. **LLM-Based Agents**: Current LLM-based agent awareness mechanisms (e.g., tool use, environment perception) lack formal guarantees; the logical framework proposed here can provide a theoretical foundation
4. **Model Checking Tool Development**: The axiomatic system can serve as the basis for automated verification tools that check whether the awareness properties of AI systems satisfy safety specifications

## Rating

- **Novelty**: ★★★★☆ — The perspective of treating awareness as a form of knowledge is novel; the formalization of de re/de dicto within awareness is an original contribution
- **Theoretical Depth**: ★★★★★ — Complete soundness and strong completeness proofs with high technical rigor
- **Practical Utility**: ★★☆☆☆ — Purely theoretical work; considerable distance remains to practical application
- **Clarity of Presentation**: ★★★★☆ — Running examples effectively aid comprehension, though the proof sections are technically demanding
- **Overall**: ★★★★☆

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Local Guidance for Configuration-Based Multi-Agent Pathfinding](local_guidance_for_configuration-based_multi-agent_pathfinding.md)
- [\[AAAI 2026\] Structural Approach to Guiding a Present-Biased Agent](structural_approach_to_guiding_a_present-biased_agent.md)
- [\[AAAI 2026\] EvoEmpirBench: Dynamic Spatial Reasoning with Agent-ExpVer](evoempirbench_dynamic_spatial_reasoning_with_agent-expver.md)
- [\[ICML 2026\] On the Epistemic Uncertainty of Overparametrized Neural Networks](../../ICML2026/others/on_the_epistemic_uncertainty_of_overparametrized_neural_networks.md)
- [\[AAAI 2026\] Controllable Financial Market Generation with Diffusion Guided Meta Agent](controllable_financial_market_generation_with_diffusion_guided_meta_agent.md)

</div>

<!-- RELATED:END -->

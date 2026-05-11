---
title: >-
  [Paper Note] Theory of Mind for Explainable Human-Robot Interaction
description: >-
  [AAAI 2026][Robotics][Theory of Mind] This paper proposes positioning Theory of Mind (ToM) as a form of Explainable AI (XAI), systematically evaluates existing ToM research in HRI using the seven criteria of the VXAI fra…
tags:
  - "AAAI 2026"
  - "Robotics"
  - "Theory of Mind"
  - "Explainable AI"
  - "Human-Robot Interaction"
  - "VXAI Framework"
  - "User-Centered Evaluation"
date: 2026-05-08
content_hash: 2547c63e30f9af4d
---

# Theory of Mind for Explainable Human-Robot Interaction

**Conference**: AAAI 2026
**arXiv**: [2512.23482](https://arxiv.org/abs/2512.23482)
**Code**: None
**Area**: Robotics
**Keywords**: Theory of Mind, Explainable AI, Human-Robot Interaction, VXAI Framework, User-Centered Evaluation

## TL;DR

This paper proposes positioning Theory of Mind (ToM) as a form of Explainable AI (XAI), systematically evaluates existing ToM research in HRI using the seven criteria of the VXAI framework, identifies critical deficiencies (most notably the absence of fidelity assessment), and advocates for integrating ToM into XAI frameworks to achieve user-oriented explanations.

## Background & Motivation

As human-robot interaction (HRI) becomes increasingly prevalent, researchers have naturally sought human-like interaction strategies to make robot behavior more intelligible. This has motivated the adoption of Theory of Mind (ToM) in HRI.

**What is ToM?**
ToM refers to the human capacity to attribute mental states—such as beliefs, desires, and intentions—to oneself and others in order to predict and explain behavior. When embedded in robots, ToM enables robots to infer and respond to users' mental states, facilitating more natural, adaptive, and transparent interactions.

**The Intersection of ToM and XAI**:
- **ToM**: Emphasizes understanding and adapting to users' mental states, generating more intuitive and user-friendly explanations.
- **XAI**: Aims to make black-box models more transparent and interpretable, but often neglects user-centered evaluation.

Both share a common goal—making internal reasoning more comprehensible to humans and enhancing human-robot collaboration—yet they remain two largely separate research communities.

**Core Problem**: Existing ToM research in HRI claims to enhance user understanding and trust, but these claims are seldom subjected to systematic evaluation. In particular:
1. Existing methods rarely assess whether explanations faithfully reflect the robot's internal reasoning (the fidelity problem).
2. Rigorous evaluation against XAI standards is largely absent.

## Method

### Overall Architecture

This paper is a position paper whose core contributions are:
1. Positioning ToM as a form of XAI.
2. Systematically evaluating existing ToM research using the VXAI framework.
3. Proposing directions for integrating ToM into XAI frameworks.

### Key Designs

#### 1. The Seven Criteria of the VXAI Framework

The authors adopt the eValuation XAI (VXAI) framework proposed by Dembinsky et al. (2025), comprising seven core criteria:

| Criterion | Definition | Evaluation Condition |
|-----------|------------|----------------------|
| **Parsimony** | Explanations should be concise, avoiding unnecessary complexity | Human evaluation was conducted |
| **Plausibility** | Explanations should conform to human logic and intuition | Human evaluation was conducted |
| **Coverage** | Whether explanations can be generated for every relevant input/output | Number of successful/failed interactions was reported |
| **Fidelity** | Explanations should faithfully reflect the model's decision process | Model's internal reasoning was examined |
| **Continuity** | Robustness of explanations under minor input perturbations | ≥100 experimental participants |
| **Consistency** | Explanations for identical/similar instances should be consistent and reproducible | ≥100 experimental participants |
| **Efficiency** | Computational cost and general applicability of the explanation method | Computational implementation details were provided |

#### 2. Literature Classification and Evaluation

**Category 1: Human Attribution of ToM to Robots**

Studies examining whether humans naturally attribute ToM to robots in the absence of explicit ToM mechanisms:
- Banks (2020): When robots display clearly interpretable social cues, humans can read robot behavior similarly to human behavior; however, comprehension declines when cues deviate from human expectations.
- Verma et al. (2024): LLMs can serve as useful tools in HRI but are not reliable ToM proxies.

**Category 2: Evaluation of Robots with Embedded ToM Reasoning**

Studies examining the effects of directly embedding ToM in robots on trust, helpfulness, and understanding:
- Mou et al. (2020): Robots with ToM capabilities are perceived more positively.
- Cantucci & Falcone (2022): Assistance aligned with user goals is more favorably received.
- Shvo et al. (2022): Robots that reason about human beliefs are perceived as more helpful and socially capable.
- Angelopoulos et al. (2025): Such robots are also perceived as more trustworthy.
- Yuan et al. (2022): Not all explanations are equally effective.
- Kerzel et al. (2022): Multi-level explanations can improve user understanding.

#### 3. Evaluation Result Matrix

| Paper | Parsimony | Plausibility | Coverage | Fidelity | Continuity | Consistency | Efficiency |
|-------|-----------|--------------|----------|----------|------------|-------------|------------|
| Banks (2020) | ✓ | ✓ | | | | ✓ | |
| Mou et al. (2020) | ✓ | ✓ | | | | | |
| Cantucci & Falcone (2022) | ✓ | ✓ | | | | ✓ | |
| Kerzel et al. (2022) | ✓ | ✓ | | | | ✓ | |
| Shvo et al. (2022) | ✓ | ✓ | | | | ✓ | |
| Yuan et al. (2022) | ✓ | ✓ | | | ✓ | ✓ | ✓ |
| Verma et al. (2024) | ✓ | ✓ | | | | ✓ | |
| Angelopoulos et al. (2025) | ✓ | ✓ | | | ✓ | ✓ | ✓ |

### Loss & Training

This paper is a position/survey paper and does not involve model training. The core contributions are at the level of the analytical framework.

## Key Experimental Results

### Main Results

This is an analytical paper; the primary empirical content derives from the systematic evaluation of eight existing ToM studies:

| Criterion | Papers Meeting Criterion / Total | Finding |
|-----------|----------------------------------|---------|
| **Parsimony** | **8/8 (100%)** | All studies conducted human evaluations |
| **Plausibility** | **8/8 (100%)** | All studies assessed the credibility of explanations |
| **Coverage** | **0/8 (0%)** | None reported the ratio of successful vs. failed interactions |
| **Fidelity** | **0/8 (0%)** | None examined the model's internal reasoning process |
| **Continuity** | **2/8 (25%)** | Only two studies had ≥100 participants |
| **Consistency** | **6/8 (75%)** | Most reported consistency-related information |
| **Efficiency** | **2/8 (25%)** | Only two provided computational implementation details |

### Ablation Study

This paper substitutes comparative analysis for ablation experiments:

| Dimension | ToM | Conventional XAI | Gap |
|-----------|-----|------------------|-----|
| User-centered evaluation | Strong | Weak | ToM prioritizes user perception |
| Model fidelity | Weak | Potentially strong | ToM neglects internal reasoning verification |
| Explanation plausibility | Validated | Technically oriented | Different foci |
| Participant scale | Mostly <100 | Highly variable | ToM requires larger-scale studies |
| Reproducibility | Weak | Domain-dependent | A shared challenge |

### Key Findings

1. **Fidelity is the greatest blind spot**: None of the eight papers examined whether explanations faithfully reflect the model's internal reasoning, creating a risk of misleading users. Explanations provided by robots may bear no relation to the actual decision process.
2. **Coverage is entirely absent**: No paper reported the ratio of successful vs. failed interactions, making it impossible to assess the reliability of the approach in real-world scenarios.
3. **User-centered evaluation is ToM's strength**: All studies conducted human evaluations (satisfying parsimony and plausibility), which is precisely what the XAI community often lacks.
4. **Scalability issues**: Most studies had fewer than 100 participants, limiting the generalizability of their conclusions.

## Highlights & Insights

1. **Interdisciplinary bridging** holds substantial value—connecting the ToM community (cognitive science/HRI) with the XAI community (AI/ML) and highlighting their complementarity: ToM offers user-centeredness but lacks fidelity; XAI offers technical rigor but lacks user perspective.
2. **The centrality of fidelity**: If a robot's "explanation" does not reflect its true reasoning, the explanation becomes a façade that may be counterproductive, fostering misplaced trust.
3. **The systematicity of the VXAI framework**: The seven criteria provide a clear checklist for future ToM+XAI work.
4. **The proposed perspective shift is forward-looking**—moving from "interpretability of the AI system itself" toward "explanations driven by users' information needs."

## Limitations & Future Work

1. **Lack of empirical validation**: The paper only proposes a framework and analysis; no system integrating ToM+XAI is actually constructed.
2. **The mapping of evaluation criteria may be oversimplified**: For instance, treating "human evaluation was conducted" as sufficient to satisfy parsimony and plausibility may not be precise enough.
3. **Limited coverage**: Only eight papers are analyzed, potentially overlooking relevant work.
4. **Implementation pathways are insufficiently specific**: Directions such as Bayesian reinforcement learning, behavior trees, and explainable reinforcement learning (XRL) are mentioned, but technical details are absent.
5. **The threshold for continuity and consistency (≥100 participants) is somewhat arbitrary**; in HRI research, a sample of 100 participants is already considerable.

## Related Work & Insights

- **The VXAI framework (Dembinsky et al., 2025)** provides a unified standard for XAI evaluation; this paper represents the first application of this framework to ToM research.
- **Angelopoulos et al. (2025)** is the ToM study satisfying the greatest number of VXAI criteria and can serve as a reference for future work.
- Bayesian reinforcement learning, behavior trees, and explainable reinforcement learning (XRL) are proposed as technical pathways for future ToM+XAI integration.
- The analytical methodology of this paper can be extended to evaluate other HRI systems that claim to provide "explanations."

## Rating

- Novelty: ⭐⭐⭐⭐ — The perspective of evaluating ToM as a form of XAI is novel.
- Experimental Thoroughness: ⭐⭐ — Confined to literature analysis; no empirical experiments.
- Writing Quality: ⭐⭐⭐⭐ — Argumentation is clear with a coherent logical structure.
- Value: ⭐⭐⭐ — A position paper that identifies an important direction but requires subsequent empirical support.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Adaptive Theory of Mind for LLM-based Multi-Agent Coordination](adaptive_theory_of_mind_for_llm-based_multi-agent_coordination.md)
- [\[AAAI 2026\] A Computable Game-Theoretic Framework for Multi-Agent Theory of Mind](a_computable_game-theoretic_framework_for_multi-agent_theory_of_mind.md)
- [\[AAAI 2026\] Human Cognitive Biases in Explanation-based Interaction: The Case of Within and Between Session Order Effect](human_cognitive_biases_in_explanation-based_interaction_the_case_of_within_and_b.md)
- [\[NeurIPS 2025\] MindForge: Empowering Embodied Agents with Theory of Mind for Lifelong Cultural Learning](../../NeurIPS2025/robotics/mindforge_empowering_embodied_agents_with_theory_of_mind_for_lifelong_cultural_l.md)
- [\[NeurIPS 2025\] COOPERA: Continual Open-Ended Human-Robot Assistance](../../NeurIPS2025/robotics/coopera_continual_open_ended_human_robot_assistance.md)

</div>

<!-- RELATED:END -->

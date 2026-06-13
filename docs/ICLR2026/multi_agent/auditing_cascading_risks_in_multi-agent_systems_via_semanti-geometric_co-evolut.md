---
title: >-
  [Paper Note] Auditing Cascading Risks in Multi-Agent Systems via Semantic–Geometric Co-evolution
description: >-
  [ICLR 2026 Workshop][Multi-Agent][multi-agent safety] This paper proposes SCCAL, a framework that models semantic–geometric co-evolution in multi-agent systems (MAS) by coupling semantic flow with the Ollivier–Ricci curv…
tags:
  - "ICLR 2026 Workshop"
  - "Multi-Agent"
  - "multi-agent safety"
  - "cascading risk"
  - "Ollivier-Ricci curvature"
  - "graph geometry"
  - "proactive auditing"
date: 2026-05-08
content_hash: 05bb07a2260ac25a
---

# Auditing Cascading Risks in Multi-Agent Systems via Semantic–Geometric Co-evolution

**Conference**: ICLR 2026 Workshop
**arXiv**: [2603.13325](https://arxiv.org/abs/2603.13325)  
**Code**: None  
**Area**: Interpretability
**Keywords**: multi-agent safety, cascading risk, Ollivier-Ricci curvature, graph geometry, proactive auditing

## TL;DR
This paper proposes SCCAL, a framework that models semantic–geometric co-evolution in multi-agent systems (MAS) by coupling semantic flow with the Ollivier–Ricci curvature (ORC) of interaction graphs. The joint prediction residual between the two modalities serves as an early warning signal for cascading risks, enabling anomaly detection several rounds before semantic violations become observable.

## Background & Motivation
**Background**: LLM-based multi-agent systems (MAS) have evolved from static question answering toward complex, self-evolving collaboration, and are widely applied to long-horizon task decomposition.

**Limitations of Prior Work**: Current safety auditing relies primarily on per-message semantic content filtering (e.g., toxicity detection, jailbreak detection), which is inherently **reactive**—by the time a semantic violation becomes visible, the collaborative dynamics of the system have often already irreversibly collapsed.

**Key Challenge**: Cascading risks (hallucination cascades, collusion, role misalignment) are emergent properties of interaction dynamics, not isolated semantic violations. Early messages may appear semantically fluent and compliant, while the underlying interaction structure has already been distorted.

**Goal**: To detect precursors of cascading risks from structural changes in agent interactions, before semantic violations become manifest.

**Key Insight**: Analogous to how physical systems accumulate structural stress before catastrophic failure, MAS interactions exhibit topological distortions (e.g., information bottlenecks, excessive redundancy) prior to semantic collapse.

**Core Idea**: MAS safety auditing is formulated as a trajectory stability problem on a semantic–geometric coupled manifold. Ollivier–Ricci curvature quantifies the local geometry of the interaction graph, and an alert is triggered when the consistency between semantic flow and geometric evolution is broken.

## Method

### Overall Architecture
SCCAL (Semantic–Curvature Co-evolutionary Auditing Loop) models multi-agent interactions as a dynamic graph sequence $\mathcal{G}_t = (\mathcal{V}, \mathcal{E}_t, \mathbf{W}_t)$ paired with agent semantic states $\mathbf{Y}_t$. The pipeline proceeds as follows: ① extract semantic representations using a frozen SBERT encoder → ② construct a directed semantic flow graph based on semantic transitivity and credibility → ③ compute per-edge ORC to quantify local geometry → ④ a coupled dynamics model predicts the next-step semantic and geometric states → ⑤ an alert is triggered when the joint prediction residual exceeds a threshold.

### Key Designs

1. **Directed Semantic Flow Induction**:

    - Function: Maps raw messages to a directed weighted interaction graph.
    - Mechanism: Edge weight $w_{ij}^t = \tau_{ij}^t \cdot \chi_i^t$, where semantic transitivity $\tau_{ij}^t = \cos(\mathbf{y}_i^t, \mathbf{y}_j^{t-1})$ measures intent alignment, and credibility $\chi_i^t = \exp(-\text{PPL}(s_i^t))$ penalizes incoherent outputs via perplexity from a reference language model.
    - Design Motivation: Suppresses spurious semantic propagation while preserving meaningful influence pathways.

2. **Ollivier–Ricci Curvature (ORC) for Geometric Quantification**:

    - Function: Computes discrete curvature for each edge in the interaction graph.
    - Mechanism: $\kappa_{ij} = 1 - W_1(m_i, m_j)/d(i,j)$, where $W_1$ is the Wasserstein-1 distance and $m_i, m_j$ are neighborhood probability measures induced by the semantic flow weights.
    - Interpretation: Positive curvature → information redundancy (echo chamber / collusion); negative curvature → structural bottleneck (cascading risk amplification point).
    - Design Motivation: Provides local geometric signals that GNNs cannot capture; ORC naturally characterizes redundancy and bottlenecks in information transmission.

3. **Coupled Dynamics Predictor**:

    - **Geometry-aware semantic predictor ψ**: A curvature-gated GRU with attention weights $\alpha_{ij}^t \propto \exp(w_{ij}^t \cdot \text{ReLU}(\kappa_{ij}^t))$, attenuating influence from structurally unstable interactions.
    - **Semantic-tension geometric predictor ϕ**: $\hat{\kappa}_{ij}^{t+1} = \text{MLP}[\kappa_{ij}^t \| \|\mathbf{y}_i^t - \mathbf{y}_j^t\| \| \text{Var}(\mathcal{N}_i, \mathcal{N}_j)]$, modeling how semantic divergence induces local structural reorganization.
    - Design Motivation: Bidirectional coupling ensures that semantic and geometric evolution are jointly constrained; neither signal alone is sufficient for reliable risk detection.

### Anomaly Detection and Attribution
The joint prediction residual is defined as: $\mathcal{A}_t = \sum_i \|\mathbf{y}_i^t - \hat{\mathbf{y}}_i^t\|^2 + \lambda \sum_{i,j} |\kappa_{ij}^t - \hat{\kappa}_{ij}^t|$. Edges with the largest curvature deviation directly localize the agents and interaction links responsible for triggering cascading risks.

## Key Experimental Results

### Main Results
Evaluated on approximately 2,000 normal trajectories and 2,000 risk trajectories (12–15 agents per task):

| Method | AUROC (Attack) | AUROC (Natural) | DLT (Attack) | DLT (Natural) |
|--------|---------------|----------------|-------------|--------------|
| Semantic-only | 0.71 | 0.62 | 5.8 | 0.4 |
| ORC-only | 0.77 | 0.69 | 7.4 | 1.2 |
| Structural GNN | 0.75 | 0.77 | 7.2 | 2.6 |
| **SCCAL (Ours)** | **0.88** | **0.83** | **10.2** | **3.4** |

DLT (Detection Lead Time) = number of rounds before the first explicit semantic violation at which the risk is detected; higher is better.

### Ablation Study

| Configuration | AUROC | DLT |
|--------------|-------|-----|
| Full model | 0.88 | 9.8 |
| w/o Coupling | 0.81 | 7.1 |
| w/o Semantic module | 0.79 | 6.7 |
| w/o Curvature | 0.75 | 4.9 |

### Key Findings
- Semantic–geometric coupling is central to detection capability: removing the coupling loss reduces AUROC by 7% and DLT by 2.7 rounds.
- The ORC-only method exhibits a false positive rate as high as 0.32 in benign high-density collaboration, whereas SCCAL achieves only 0.07.
- On natural failure scenarios without adversarial attacks, semantic-only methods perform near random (AUROC 0.62), while SCCAL remains effective (0.83).

## Highlights & Insights
- **Temporal asymmetry insight**: Semantic signals are lagging indicators while geometric signals are leading indicators—analogous to the observation in physical systems that stress accumulates before fracture.
- **Semantic grounding of curvature**: Raw curvature would misclassify benign efficient collaboration (e.g., high positive curvature during brainstorming) as collusion; semantic constraints are necessary to disambiguate—an important design insight.
- **Paradigm shift from content auditing to process auditing**: The paper offers a transferable principle—safety in any multi-agent system should be assessed not by individual messages but by the structural evolution of the interaction process.

## Limitations & Future Work
- Experiments are based on a synthetic benchmark (AEGIS 2.0 risk taxonomy), leaving a gap with real production environments (latency, asynchrony, human-in-the-loop).
- Only local ORC is employed, potentially missing global topological phase transitions in large-scale networks; persistent homology or spectral methods could be incorporated.
- As a workshop paper, the experimental scale is limited (~4,000 trajectories) with no large-scale validation.
- Computational overhead is not discussed—ORC computation on dense graphs may become a bottleneck.

## Supplementary Technical Details

### Physical Intuition of ORC
Ollivier–Ricci curvature is a discretization of Riemannian Ricci curvature. On a continuous manifold, positive Ricci curvature implies neighboring geodesics tend to converge (spherical effect), while negative curvature implies divergence (hyperbolic effect). Mapped to graphs:
- **Positive ORC**: neighborhoods of nodes $i$ and $j$ are highly overlapping → information redundancy, potentially forming echo chambers.
- **Negative ORC**: neighborhoods of nodes $i$ and $j$ are nearly disjoint → information bottleneck, where disconnection leads to cascading failure.
- **Zero ORC**: normal information diffusion pattern.

### Wasserstein Distance Computation
$W_1(m_i, m_j)$ is solved via optimal transport, and can be computed analytically on small neighborhoods with complexity $O(k^3 \log k)$, where $k$ is the neighborhood size. For MAS with 12–15 agents, computation is tractable, though approximate algorithms may be required for larger-scale systems.

## Related Work & Insights
- **vs. Semantic auditing (guardrails)**: Traditional methods filter messages reactively; this work performs proactive structural auditing.
- **vs. CurvGAD**: CurvGAD applies ORC to static graph anomaly detection but is insensitive to semantic context; this work eliminates false positives through semantic grounding.
- **vs. GNN structural methods**: GNNs are competitive on natural failures but exhibit detection lag (DLT of only 2.6); ORC better captures the accumulation of structural tension (DLT of 10.2).
- **vs. Traditional MAS safety analysis**: Traditional methods focus on policy robustness within fixed action spaces and struggle with the open-ended semantic interaction risks introduced by LLMs.
- This work can inspire safety design in multimodal agent systems: perception–reasoning modality misalignment may also manifest as semantic–geometric decoupling.
- There is an intrinsic connection to the Information Bottleneck framework in information theory: negative curvature is essentially a geometric formulation of information bottlenecks.

## Rating
- Novelty: ⭐⭐⭐⭐ Introducing Riemannian geometry (ORC) into MAS safety auditing is a creative and substantive interdisciplinary contribution.
- Experimental Thoroughness: ⭐⭐⭐ Workshop scale; experimental setup is limited with no real-environment validation.
- Writing Quality: ⭐⭐⭐⭐ Clear logical structure, complete motivation chain, and well-formed mathematical exposition.
- Value: ⭐⭐⭐⭐ The paradigm shift from content auditing to process auditing has broad applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Shadows in the Code: Exploring the Risks and Defenses of LLM-based Multi-Agent Software Development Systems](../../AAAI2026/multi_agent/shadows_in_the_code_exploring_the_risks_and_defenses_of_llm-.md)
- [\[ACL 2026\] EvoSci: A Bio-Inspired Multi-Agent Framework for the Evolution of Scientific Discovery](../../ACL2026/multi_agent/evosci_a_bio-inspired_multi-agent_framework_for_the_evolution_of_scientific_disc.md)
- [\[ACL 2026\] EvoSpark: Endogenous Interactive Agent Societies for Unified Long-Horizon Narrative Evolution](../../ACL2026/multi_agent/evospark_endogenous_interactive_agent_societies_for_unified_long-horizon_narrati.md)
- [\[ICLR 2026\] Stochastic Self-Organization in Multi-Agent Systems](stochastic_self-organization_in_multi-agent_systems.md)
- [\[ICLR 2026\] Stop Wasting Your Tokens: Towards Efficient Runtime Multi-Agent Systems](stop_wasting_your_tokens_towards_efficient_runtime_multi-agent_systems.md)

</div>

<!-- RELATED:END -->

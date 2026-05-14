---
title: >-
  [Paper Note] Context-Value-Action Architecture for Value-Driven Large Language Model Agents
description: >-
  [ACL 2026 (Findings)][Interpretability][Value-driven agents] This paper proposes the CVA (Context-Value-Action) architecture, grounded in the S-O-R psychological model and Schwartz's theory of basic human values. By trai…
tags:
  - "ACL 2026 (Findings)"
  - "Interpretability"
  - "Value-driven agents"
  - "behavior simulation"
  - "Schwartz value theory"
  - "behavior polarization"
  - "verifier"
date: 2026-05-08
content_hash: dd9e2787b122a2d0
---

# Context-Value-Action Architecture for Value-Driven Large Language Model Agents

**Conference**: ACL 2026 (Findings)
**arXiv**: [2604.05939](https://arxiv.org/abs/2604.05939)
**Code**: None
**Area**: LLM Agent / Interpretability
**Keywords**: Value-driven agents, behavior simulation, Schwartz value theory, behavior polarization, verifier

## TL;DR
This paper proposes the CVA (Context-Value-Action) architecture, grounded in the S-O-R psychological model and Schwartz's theory of basic human values. By training a Value Verifier on real human data, CVA decouples action generation from cognitive reasoning, effectively mitigating behavioral polarization in LLM agents. The approach achieves substantial improvements over baselines on CVABench, a benchmark comprising over 1.1 million real interaction trajectories.

## Background & Motivation

**Background**: LLM-based human-like agents (game NPCs, social simulacra, task assistants, etc.) must faithfully capture the complexity, diversity, and stochasticity of human behavior. Existing approaches primarily rely on psychological prompting strategies—such as role-playing and chain-of-thought reasoning—to simulate human cognitive processes.

**Limitations of Prior Work**: Existing LLM agents frequently exhibit behavioral rigidity and stereotyping. More critically, this problem is obscured by prevailing evaluation practices: "LLM-as-a-judge" evaluation suffers from self-referential bias, as the judge model shares pre-training biases with the agent being evaluated and tends to reward polarized behavior rather than penalizing its lack of authenticity.

**Key Challenge**: Increasing the intensity of prompt-driven reasoning does not improve behavioral fidelity; instead, it exacerbates value polarization. LLMs tend to collapse nuanced value dimensions into "caricatured" prototypes (e.g., mapping an "irritable" personality to uniformly aggressive responses), causing population-level diversity to collapse.

**Goal**: To construct agents that faithfully reproduce the diversity of human behavior, using real human data—rather than LLM self-evaluation—as the evaluation criterion.

**Key Insight**: The work draws on the psychological S-O-R (Stimulus-Organism-Response) model and Schwartz's theory of basic human values. Human behavior is not a static output of personality, but a dynamic process in which contextual stimuli activate specific value dimensions.

**Core Idea**: An external Value Verifier, trained on real human data, replaces the LLM's own value judgment, decoupling action generation from cognitive reasoning and thereby eliminating the self-referential bias that drives polarization.

## Method

### Overall Architecture
CVA adopts a generate-then-verify paradigm. It first calibrates the base LLM's value-to-behavior mapping via SFT and DPO (the VMC stage), then employs an independently trained Value Verifier to select, from a set of candidate actions, the one most consistent with the currently activated values (the VDR stage).

### Key Designs

1. **Value-to-behavior Mapping Calibration (VMC)**:

    - Function: Corrects the LLM's intrinsic value distortions.
    - Mechanism: A two-step pipeline—SFT fine-tunes the model on real CVABench trajectories to align the probability space with the true conditional distribution $P(A|C,V)$; DPO further reinforces authentic value-behavior associations using preference pairs (nuanced-consistent vs. caricatured-exaggerated), suppressing distorted reasoning paths.
    - Design Motivation: Learning directly from real data prevents the LLM from collapsing value $V$ into a caricatured prototype $V'$.

2. **Value-Driven Verifier (VDR)**:

    - Function: Acts as an independent discriminator that evaluates the consistency between candidate actions and activated values.
    - Mechanism: The verifier is trained on real $(C, V, A)$ triples. At inference time, a generate-then-select protocol is used: the calibrated model samples $N$ candidate actions, the verifier computes a consistency score $s_i = f_{ver}(a_i, C, V)$ for each, and the candidate with the highest score is selected as the final output.
    - Design Motivation: Using the model itself as a verifier creates a self-referential loop that amplifies bias; an independent verifier breaks this loop.

3. **CVABench**:

    - Function: A training and evaluation framework grounded in real human behavioral data.
    - Mechanism: Aggregates over 1.1 million real interaction trajectories from three domains (Yelp reviews: 54K; Reddit conversations: 155K; Foursquare mobility: 871K), covering 15,571 users. GPV (General Psychometric Verification) is used to map user behavior onto the Schwartz 10-dimensional value space.
    - Design Motivation: Replaces LLM self-evaluation with real data to establish an objective behavioral fidelity benchmark.

### Loss & Training
SFT: Standard autoregressive loss on real trajectories. DPO: Preference optimization favoring nuanced, consistent behaviors over polarized, exaggerated ones. Verifier: A discriminative model trained on real $(C, V, A)$ triples.

## Key Experimental Results

### Main Results

| Method | Behavioral Fidelity | Diversity Preservation | Polarization Degree |
|--------|--------------------|-----------------------|---------------------|
| Raw LLM | Low | Low | High |
| Role Play Agent | Low | Low | High |
| Prompt-Reasoning Agent | Lower | Lower | **Higher** |
| CVA (VMC) | Medium | Medium | Medium |
| CVA (VMC + VDR) | **Highest** | **Highest** | **Lowest** |

### Key Findings

| Finding | Description |
|---------|-------------|
| Reasoning intensity vs. polarization | Stronger prompt-based reasoning exacerbates polarization, contrary to intuition |
| Verifier peak phenomenon | Behavioral fidelity does not increase monotonically with candidate count $N$; an optimal peak exists |
| Interpretability | Verifier attention transparently reveals which value dimensions drive selection |

### Key Findings
- Increasing reasoning intensity (more CoT steps) not only fails to improve fidelity but exacerbates value polarization and collapses population-level diversity.
- Behavioral fidelity peaks at an optimal number of candidates, mirroring the phenomenon of limited evaluation scope under human cognitive constraints.
- CVA significantly outperforms all baselines across all three domains (reviews / conversations / mobility).

## Highlights & Insights
- **The finding that "more reasoning leads to greater polarization"** is particularly significant—it directly challenges the intuition that "more deliberation equals better performance," exposing a fundamental deficiency of LLMs in human simulation tasks.
- **The verifier peak effect** elegantly maps onto the concept of "bounded rationality" in cognitive science.
- **A corrected evaluation paradigm**: Shifting from "LLM-as-a-judge" to "real data as ground truth" establishes a new standard for agent evaluation.

## Limitations & Future Work
- The three data sources in CVABench (Yelp / Reddit / Foursquare) may not be representative of all human behavioral patterns.
- Although well-established, the Schwartz 10-dimensional value model may lack sufficient granularity; certain behaviors may be influenced by factors not captured by this framework.
- Verifier training requires substantial real-world data, and performance in data-scarce settings remains unexplored.

## Related Work & Insights
- **vs. Park et al. (Generative Agents)**: Relies on persona prompting for simulation, which induces behavioral rigidity; CVA replaces this with a verifier trained on real data.
- **vs. VLA systems**: VLA focuses on embodied task execution, whereas CVA targets fidelity of socio-psychological behavior.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — Deep integration of psychological value theory with LLM agents; the decoupled verification approach is highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 1.1 million real data points with rigorous multi-paradigm comparisons.
- Writing Quality: ⭐⭐⭐⭐ — Solid theoretical foundations and findings of considerable depth.
- Value: ⭐⭐⭐⭐⭐ — Fundamental contributions to LLM-based human simulation and agent evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] ValuePilot: A Two-Phase Framework for Value-Driven Decision-Making](../../NeurIPS2025/interpretability/valuepilot_a_two-phase_framework_for_value-driven_decision-making.md)
- [\[NeurIPS 2025\] Steering Information Utility in Key-Value Memory for Language Model Post-Training](../../NeurIPS2025/interpretability/steering_information_utility_in_key-value_memory_for_language_model_post-trainin.md)
- [\[ACL 2026\] Tracing Relational Knowledge Recall in Large Language Models](tracing_relational_knowledge_recall_in_large_language_models.md)
- [\[ACL 2026\] Experiments or Outcomes? Probing Scientific Feasibility in Large Language Models](experiments_or_outcomes_probing_scientific_feasibility_in_large_language_models.md)
- [\[NeurIPS 2025\] Probabilistic Token Alignment for Large Language Model Fusion](../../NeurIPS2025/interpretability/probabilistic_token_alignment_for_large_language_model_fusion.md)

</div>

<!-- RELATED:END -->

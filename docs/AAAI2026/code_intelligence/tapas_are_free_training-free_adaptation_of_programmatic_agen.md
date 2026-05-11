---
title: >-
  [Paper Note] TAPA: Training-Free Adaptation of Programmatic Agents via LLM-Guided Program Synthesis in Dynamic Environments
description: >-
  [AAAI 2026][Code Intelligence][Programmatic Agent] TAPA positions LLMs as "intelligent modulators" of the symbolic action space rather than direct decision-makers. Through LLM-guided program synthesis…
tags:
  - "AAAI 2026"
  - "Code Intelligence"
  - "Programmatic Agent"
  - "LLM-guided program synthesis"
  - "dynamic environment adaptation"
  - "logical primitives"
  - "symbolic action space"
date: 2026-05-08
content_hash: f42435bb1a58a910
---

# TAPA: Training-Free Adaptation of Programmatic Agents via LLM-Guided Program Synthesis in Dynamic Environments

**Conference**: AAAI 2026
**arXiv**: [2508.11425](https://arxiv.org/abs/2508.11425)
**Code**: None
**Area**: Code Intelligence
**Keywords**: Programmatic Agent, LLM-guided program synthesis, dynamic environment adaptation, logical primitives, symbolic action space

## TL;DR
TAPA positions LLMs as "intelligent modulators" of the symbolic action space rather than direct decision-makers. Through LLM-guided program synthesis, it dynamically adapts the symbolic actions of programmatic agents without retraining, achieving strong performance in cybersecurity DDoS defense (77.7% network uptime) and swarm intelligence formation control.

## Background & Motivation
**Background**: Autonomous agents in safety-critical applications (network defense, swarm intelligence, autonomous driving) typically adopt neuro-symbolic hybrid architectures—RL/neural networks for policy learning and symbolic programs/rules for concrete execution. Such "programmatic agents" offer interpretability and formal guarantees.

**Limitations of Prior Work**: Three core deficiencies: (1) the symbolic action space is static and cannot adapt to environmental changes (e.g., novel attack patterns); (2) policy retraining is costly (requiring hours of relearning); (3) applying outdated symbolic actions to new environments lacks safety guarantees.

**Key Challenge**: LLMs possess powerful reasoning and generalization capabilities but suffer from high inference latency and hallucination, making them unsuitable for real-time safety-critical decisions; traditional RL agents are fast but lack adaptability. Both are individually insufficient and require complementary integration.

**Goal**: How can programmatic agents continuously adapt to dynamic environments without retraining?

**Key Insight**: A key paradigm shift—from "policy-level retraining" to "action-level adaptation." Rather than modifying the decision policy (the meta-policy remains fixed), the paper proposes dynamically modifying/synthesizing the concrete symbolic program corresponding to each logical primitive, with LLMs performing program synthesis rather than real-time decision-making.

**Core Idea**: Use LLMs as modulators of the symbolic action space, dynamically synthesizing and adapting concrete programs to match abstract logical primitives, enabling training-free environmental adaptation.

## Method

### Overall Architecture
TAPA operates in two phases: design time and deployment time. At design time: (1) experts define logical primitives (e.g., Observe/Defend/Validate/Alert); (2) a meta-agent is initialized to learn the primitive selection policy; (3) LLMs generate candidate symbolic program pools across multi-scenario simulations; (4) programs are validated and adapted; (5) provenance chains are constructed and stored in a RAG system. At deployment time: the meta-agent selects a logical primitive → the corresponding program executes → performance degradation is detected → LLMs synthesize new programs as replacements.

### Key Designs

1. **Logical Primitives Abstraction**:

    - Function: Abstracts the concrete action space into high-level strategic intents, such as {Observe, Defend, Validate, Alert} in the cybersecurity scenario.
    - Mechanism: The meta-agent's policy $\pi_{\text{meta}}: S \rightarrow \mathcal{L}$ operates over the abstract primitive space, while concrete execution is handled by interchangeable symbolic programs. This decoupling means the policy need not be retrained as the environment evolves—only the programs mapped to primitives need to be replaced.
    - Design Motivation: Logical primitives represent universal strategic intents ("Defend"/"Observe") that are independent of concrete implementation details, thus remaining stable as the environment changes.

2. **LLM-Guided Program Pool Construction**:

    - Function: Leverages LLMs and domain expert knowledge to generate diverse candidate symbolic programs for each logical primitive across multi-scenario simulations.
    - Mechanism: $\mathcal{P}_{i,v} = \text{LLM}_{\text{gen}}(L_i, \xi(E_v), \mathcal{V}_{\text{RAG}})$, where $\xi(E_v)$ is the environmental context feature and $\mathcal{V}_{\text{RAG}}$ is expert knowledge and historical experience stored in the RAG system.
    - Design Motivation: Pre-generating a diverse program pool allows the system to handle new environments at runtime through combination, selection, and adaptation rather than synthesis from scratch.

3. **Action Adaptation and Validation**:

    - Function: Upon detecting performance degradation, LLMs analyze the environmental context and historical experience to synthesize new program combinations.
    - Mechanism: The program mapping $\mathcal{M}(L_i) = \bigcup_j (P_j \odot \text{op}_j)$, where $\text{op} \in \{\wedge, \vee, +, -, \delta\}$ denotes logical composition operations between programs (conjunction, disjunction, addition, deletion, modification). New programs must pass **shadow simulation** validation before deployment.
    - Design Motivation: Shadow simulation is an industry-grade safety practice—new programs run in parallel with the production system without affecting actual operations, ensuring safety before replacement.

4. **Provenance Chain + RAG**:

    - Function: Records the complete execution trace of each adaptation—logical primitives, environmental context, original programs, new programs, performance changes, and adaptation rationale.
    - Mechanism: Stored in the RAG system, historical experience is retrieved to guide program synthesis when similar scenarios arise in the future. Iterative refinement progressively enriches the knowledge base.
    - Design Motivation: Addresses the scarcity of domain-specific code samples in safety-critical fields, compensating for LLMs' limited domain knowledge through accumulated experience.

### Loss & Training
The meta-agent uses a TiT (Transformer-in-Transformer) architecture trained in the baseline environment E1. The LLM component uses Claude Sonnet 4 (cybersecurity) and GPT-4o (formation control) with no fine-tuning required.

## Key Experimental Results

### Main Results — DDoS Defense (E2–E5 Dynamic Environments)

| Method | E2 Acc/FP | E3 Acc/FP | E4 Acc/FP | E5 Acc/FP | Network Uptime |
|--------|----------|----------|----------|----------|----------------|
| Static Symbolic | 100/36.7 | 100/33.3 | 70/15 | 15/15 | 48.1% |
| Symbolic-Neural | 100/16.7 | 100/43.3 | 60/63.8 | 97.5/56.3 | 70.5% |
| End-to-End LLM | 70/26.7 | 65/33.3 | 60/27.5 | 45/46.3 | 27.6% |
| **TAPA** | **100/0** | **100/0** | **100/0** | **100/12.5** | **77.7%** |

### Ablation Study — Swarm Formation Adaptation (Storm Conditions)

| Configuration | Round 1 | Round 2 | Round 3 |
|---------------|---------|---------|---------|
| TAPA-Full | 72.3 | 81.7 (+9.4) | 86.2 (+4.5) |
| w/o Provenance Chain | 65.1 | 68.9 (+3.8) | 62.4 (-6.5) |
| w/o Expert Knowledge | 38.6 | 54.2 (+15.6) | 62.7 (+8.5) |
| w/o Both | 4.2 | 6.8 (+2.6) | 0.0 (-6.8) |

### Key Findings
- TAPA maintains 100% detection accuracy and near-zero false positives across all environments; end-to-end LLM agents perform worst (27.6% uptime), validating that LLMs are unsuitable for direct safety-critical real-time decision-making.
- Static symbolic methods collapse in E5 (large-scale attacks, 15% detection rate), demonstrating the fragility of static action spaces.
- In ablation experiments, removing the provenance chain allows fast startup but prevents sustained improvement (degradation in Round 3); removing expert knowledge slows startup but permits gradual learning—the two components are complementary.
- In swarm formation control, TAPA achieves a target radius of 494.3±0.2 (target: 500), with only 1.1% deviation.

## Highlights & Insights
- **Paradigm Shift**: From "policy retraining" to "symbolic action adaptation"—this approach carries significant engineering value. Dynamically updating the execution layer without altering the decision policy substantially reduces adaptation costs.
- **LLM as Modulator, Not Decision-Maker**: LLMs are excluded from real-time decision-making (due to high latency and unreliability) and instead synthesize programs in the background for use by the RL agent, cleverly leveraging the strengths of each component.
- **Shadow Simulation Validation**: Borrowing from industrial practice, new programs must pass validation before deployment, which is critical for safety-critical systems.

## Limitations & Future Work
- Logical primitives require manual definition by human experts; an automated primitive discovery mechanism is absent.
- The quality of program synthesis depends on LLMs' code generation capability and domain knowledge, which may be insufficient in highly specialized domains.
- Experiments are conducted at a relatively small scale (10–20 servers, 10 UAVs); scalability in large-scale scenarios remains to be verified.
- Shadow simulation requires high-fidelity simulators, which are themselves challenging to obtain in real deployments.
- The paper does not discuss the details of rollback mechanisms in the event of adaptation failure.

## Related Work & Insights
- **vs. End-to-End LLM Agents**: Direct LLM decision-making performs extremely poorly in safety-critical scenarios (27.6%), and TAPA demonstrates that LLMs are better suited as "behind-the-scenes" program synthesizers.
- **vs. Traditional RL Retraining**: Policy retraining typically requires hours; TAPA's action-level adaptation can be completed at runtime.
- **vs. Fixed Symbolic Systems**: Static rules are brittle under environmental change; TAPA maintains adaptability through dynamic program synthesis.
- Insights: The architecture of "fixed policy, dynamically adjusted execution" can generalize to broader scenarios—for instance, keeping high-level decision policies fixed in autonomous driving while dynamically adjusting low-level control programs.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The LLM-as-action-space-modulator paradigm is entirely novel; the shift from policy adaptation to action adaptation is thought-provoking.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated across two safety-critical domains, though the scale is small and additional baseline comparisons are lacking.
- Writing Quality: ⭐⭐⭐⭐ Framework description is clear, formal definitions are rigorous, and examples are woven throughout.
- Value: ⭐⭐⭐⭐ Offers practical guidance for adaptive design in safety-critical autonomous systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] TikZero: Zero-Shot Text-Guided Graphics Program Synthesis](../../ICCV2025/code_intelligence/tikzero_zero-shot_text-guided_graphics_program_synthesis.md)
- [\[NeurIPS 2025\] Program Synthesis via Test-Time Transduction](../../NeurIPS2025/code_intelligence/program_synthesis_via_test-time_transduction.md)
- [\[AAAI 2026\] DiffBench Meets DiffAgent: End-to-End LLM-Driven Diffusion Acceleration Code Generation](diffbench_meets_diffagent_end-to-end_llm-driven_diffusion_ac.md)
- [\[NeurIPS 2025\] Once Upon an Input: Reasoning via Per-Instance Program Synthesis](../../NeurIPS2025/code_intelligence/once_upon_an_input_reasoning_via_per-instance_program_synthesis.md)
- [\[NeurIPS 2025\] FractalBench: Diagnosing Visual-Mathematical Reasoning Through Recursive Program Synthesis](../../NeurIPS2025/code_intelligence/fractalbench_diagnosing_visual-mathematical_reasoning_through_recursive_program_.md)

</div>

<!-- RELATED:END -->

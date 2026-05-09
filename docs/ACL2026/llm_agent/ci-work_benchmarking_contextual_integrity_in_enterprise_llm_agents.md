---
title: >-
  [Paper Note] CI-Work: Benchmarking Contextual Integrity in Enterprise LLM Agents
description: >-
  [ACL 2026][LLM Agent][enterprise privacy] CI-Work is an enterprise-scenario benchmark grounded in Contextual Integrity (CI) theory. It reveals that state-of-the-art LLM agents systematically violate privacy norms in enterprise workflows, and that scaling model size exacerbates rather than mitigates leakage.
tags:
  - ACL 2026
  - LLM Agent
  - enterprise privacy
  - contextual integrity
  - LLM agents
  - information leakage
  - privacy-utility tradeoff
date: 2026-05-08
content_hash: 60dc39e010b2408c
---

# CI-Work: Benchmarking Contextual Integrity in Enterprise LLM Agents

**Conference**: ACL 2026
**arXiv**: [2604.21308](https://arxiv.org/abs/2604.21308)
**Code**: [https://aka.ms/ci-work](https://aka.ms/ci-work)
**Area**: LLM Agent
**Keywords**: enterprise privacy, contextual integrity, LLM agents, information leakage, privacy-utility tradeoff

## TL;DR

CI-Work is an enterprise-scenario benchmark grounded in Contextual Integrity (CI) theory. It reveals that state-of-the-art LLM agents systematically violate privacy norms in enterprise workflows, and that scaling model size exacerbates rather than mitigates leakage.

## Background & Motivation

**State of the Field**: LLM agents are increasingly integrated into enterprise workflows, accessing internal data such as emails and meeting notes to execute complex tasks, substantially improving productivity.

**Limitations of Prior Work**: Existing privacy benchmarks (ConfAide, PrivacyLens, CIMemories, etc.) predominantly target everyday personal-assistant scenarios and fail to capture the complexity of enterprise environments: (1) they evaluate single information flows in isolation, overlooking the concurrent and intertwined nature of multiple flows in organizations; (2) their evaluation contexts are simple and isolated, unable to measure an agent's capacity to distinguish *necessary* from *sensitive* information under dense retrieval conditions; (3) they rely on simplified contexts or short attribute lists that cannot reproduce the scale and density of real enterprise data.

**Root Cause**: The core capability of enterprise LLM agents—retrieving and utilizing internal data—is precisely what makes them potential conduits for sensitive information leakage; higher task utility tends to co-occur with more frequent privacy violations.

**Paper Goals**: Construct an enterprise-grade benchmark grounded in CI theory to systematically evaluate the privacy-utility tradeoff of LLM agents in high-fidelity enterprise workflows.

**Starting Point**: Enterprise information flows are categorized along five organizational communication directions (downward, upward, lateral, diagonal, and external). Each benchmark instance includes a dense retrieval context composed of a *necessary set* and a *sensitive set*.

**Core Idea**: Enterprise privacy is not a matter of simple information blocking; it requires precise discrimination between necessary and sensitive information under dense retrieval conditions—a challenge that current models have not solved, and one that worsens as model scale increases.

## Method

### Overall Architecture

CI-Work follows a four-stage construction pipeline: (1) task-oriented seed generation → (2) context entry generation → (3) scenario episode generation → (4) trajectory simulation and evaluation. Building on ToolEmu and PrivacyLens, it constructs a tool-centric simulation environment in which enterprise tools (email, chat, calendar, meetings, etc.) are emulated by LLMs.

### Key Designs

1. **Five-Direction Information Flow Taxonomy**:

    - Function: Covers all typical information flow directions found in enterprise organizations.
    - Mechanism: Based on standard organizational communication taxonomy, flows are classified into five categories—downward (management → employee), upward (reporting to superiors), lateral (peer collaboration), diagonal (cross-department), and external (stakeholders).
    - Design Motivation: Captures differences in privacy norms across different organizational roles in real enterprises.

2. **Self-Iterative Refinement**:

    - Function: Automatically corrects generated context entries to ensure label consistency between *essential* and *sensitive* designations.
    - Mechanism: After entry generation, an LLM blindly classifies each entry as Essential/Sensitive/Ambiguous; any mismatch with the intended label triggers an automated revision loop.
    - Design Motivation: Bridges the quality gap between LLM generation and evaluation; human validation achieves 82.5%–95.0% agreement.

3. **Dual-Set Evaluation Metric Framework**:

    - Function: Simultaneously quantifies privacy protection and task utility.
    - Mechanism: Retrieved entries are partitioned into a necessary set $\mathcal{E}_{\text{ess}}$ and a sensitive set $\mathcal{E}_{\text{sens}}$; three metrics are defined—Leakage Rate (LR), Violation Rate (VR), and Conveyance Rate (CR).
    - Design Motivation: Explicitly quantifies the privacy-utility tradeoff, going beyond single-dimensional task success rate evaluation.

### Loss & Training

No model training is involved. LLM agents are deployed under the ReAct framework. GPT-5.2 is used for benchmark generation and evaluation; LLM-as-a-Judge enables automated assessment with 83.0%–91.0% agreement with human annotation.

## Key Experimental Results

### Main Results

| Model | Leakage Rate (LR) ↓ | Violation Rate (VR) ↓ | Conveyance Rate (CR) ↑ |
|-------|--------------------|-----------------------|------------------------|
| DeepSeek-R1 | 6.08% | 15.80% | 53.08% |
| DeepSeek-V3 | 8.37% | 21.33% | 76.92% |
| GPT-4o | 8.79% | 21.33% | 87.35% |
| GPT-5 | 11.21% | 27.83% | 93.04% |
| Grok-3 | 26.66% | 50.87% | 94.97% |

### Ablation Study

| Configuration | Key Metric | Observation |
|---------------|-----------|-------------|
| Entry count 1→12 | VR monotonically increases; CR drops sharply | Multi-entity contexts introduce interference |
| Increased entry length | Both LR/VR and CR rise | Richer detail expands the leakage surface |
| Implicit user pressure | VR significantly increases | Task-completion-emphasizing instructions exacerbate leakage |
| Explicit user pressure | VR nearly doubles | Proactively supplying data sources worsens leakage |

### Key Findings
- **Privacy-utility tradeoff**: Higher conveyance rate positively correlates with higher leakage/violation rates (Pearson $r=0.40$, $p<0.05$).
- **Inverse scaling phenomenon**: Larger models (GPT-4.1 series) exhibit worse privacy leakage, as stronger instruction-following capability leads to greater compliance with user requests at the expense of implicit privacy norms.
- **Higher risk in upward communication**: Leakage and violation rates for upward reporting flows are significantly higher than for downward flows (VR: $p=0.006$).
- **CI-CoT defense is helpful but insufficient**: It significantly reduces leakage rate yet violation rate remains above 20%.

## Highlights & Insights
- First systematic evaluation of contextual integrity for LLM agents in enterprise scenarios, revealing violation rates of 15.8%–50.9%.
- Uncovers the counterintuitive *inverse scaling* effect: larger models exhibit worse privacy, as they more effectively understand and execute user intent while disregarding implicit social norms.
- Demonstrates that user behavior—even implicit pressure without malicious intent—can cause simultaneous collapse of both privacy and utility.
- Explicitly argues that model scale and reasoning depth alone are insufficient to address enterprise privacy challenges, calling for a paradigm shift.

## Limitations & Future Work
- The benchmark relies on synthetic data and simulated environments, which may diverge from real enterprise deployments.
- Privacy norms vary across organizations, jurisdictions, and cultures; the benchmark cannot cover all possible scenarios.
- Future directions include role-conditioned filtering, context-aware training-time alignment, and an architectural shift from model-centric to context-centric approaches.

## Related Work & Insights
- **vs. ConfAide/PrivacyLens**: These benchmarks focus on everyday personal-assistant scenarios and single information flows; CI-Work targets enterprise multi-entity dense retrieval settings.
- **vs. CIMemories**: CIMemories addresses privacy accumulation in memory; CI-Work focuses on information leakage within a single enterprise task.
- **vs. WorkArena/TheAgentCompany**: These benchmarks evaluate task execution capability; CI-Work is the first to jointly assess utility and privacy.

## Rating
- Novelty: ⭐⭐⭐⭐ First enterprise-level CI benchmark; reveals counterintuitive phenomena such as inverse scaling.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 9 frontier models, 5 information flow directions, and multi-dimensional analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, well-motivated problem statement, and information-rich figures and tables.
- Value: ⭐⭐⭐⭐ Provides important warnings and guidance for the safe deployment of enterprise AI systems.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Gaia2: Benchmarking LLM Agents on Dynamic and Asynchronous Environments](../../ICLR2026/llm_agent/gaia2_benchmarking_llm_agents_on_dynamic_and_asynchronous_environments.md)
- [\[ICLR 2026\] Inherited Goal Drift: Contextual Pressure Can Undermine Agentic Goals](../../ICLR2026/llm_agent/inherited_goal_drift_contextual_pressure_can_undermine_agentic_goals.md)
- [\[ICLR 2026\] NewtonBench: Benchmarking Generalizable Scientific Law Discovery in LLM Agents](../../ICLR2026/llm_agent/newtonbench_benchmarking_generalizable_scientific_law_discovery_in_llm_agents.md)
- [\[ACL 2026\] AgencyBench: Benchmarking the Frontiers of Autonomous Agents in 1M-Token Real-World Contexts](agencybench_benchmarking_the_frontiers_of_autonomous_agents_in_1m-token_real-wor.md)
- [\[ACL 2026\] FedGUI: Benchmarking Federated GUI Agents across Heterogeneous Platforms, Devices, and Operating Systems](fedgui_benchmarking_federated_gui_agents_across_heterogeneous_platforms_devices_.md)

<!-- RELATED:END -->

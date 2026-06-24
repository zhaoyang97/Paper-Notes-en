---
title: >-
  [Paper Note] EnIGMA: Interactive Tools Substantially Assist LM Agents in Finding Security Vulnerabilities
description: >-
  [ICML 2025][LLM Evaluation][LLM Agent] EnIGMA is an LM agent designed to autonomously solve Capture The Flag (CTF) challenges. By introducing novel interactive agent tools (debuggers and server connection utilities), it enables LM agents to execute interactive terminal programs for the first time. It achieves state-of-the-art (SOTA) results across 390 CTF challenges from 4 benchmarks and uncovers "soliloquizing," a new type of hallucination behavior.
tags:
  - "ICML 2025"
  - "LLM Evaluation"
  - "LLM Agent"
  - "CTF Challenges"
  - "Security Vulnerabilities"
  - "Interactive Tools"
  - "Cybersecurity"
date: 2026-05-08
content_hash: c51a6e126a8ae821
---

# EnIGMA: Interactive Tools Substantially Assist LM Agents in Finding Security Vulnerabilities

**Conference**: ICML 2025  
**arXiv**: [2409.16165](https://arxiv.org/abs/2409.16165)  
**Code**: [https://github.com/SWE-agent/SWE-agent/tree/v0.7](https://github.com/SWE-agent/SWE-agent/tree/v0.7)  
**Area**: LLM Evaluation  
**Keywords**: LLM Agent, CTF Challenges, Security Vulnerabilities, Interactive Tools, Cybersecurity

## TL;DR
EnIGMA is an LM agent designed to autonomously solve Capture The Flag (CTF) challenges. By introducing novel interactive agent tools (debuggers and server connection utilities), it enables LM agents to execute interactive terminal programs for the first time. It achieves state-of-the-art (SOTA) results across 390 CTF challenges from 4 benchmarks and uncovers "soliloquizing," a new type of hallucination behavior.

## Background & Motivation
**Background**: LM agents have demonstrated outstanding performance in domains such as code generation and web browsing, but their success in cybersecurity remains limited. CTF challenges, which require discovering and exploiting security vulnerabilities, serve as critical benchmarks for evaluating the security capabilities of AI.

**Limitations of Prior Work**: Existing LM agents lack the capacity to interact with interactive terminal programs. However, interactive tools such as debuggers (GDB) and server connection tools (netcat) are indispensable for solving CTF challenges. Currently, agents can only execute non-interactive commands.

**Key Challenge**: Solving CTF challenges inherently requires the use of interactive tools (e.g., step-by-step debugging, real-time server interaction), but the standard interfaces of LM agents only support single-command execution loops.

**Goal**: To design novel tools and interfaces that enable LM agents to employ interactive terminal programs for finding security vulnerabilities.

**Key Insight**: Developing Interactive Agent Tools that wrap interactive programs into APIs callable by LM agents.

**Core Idea**: The lack of interactive tools is a critical bottleneck in the CTF domain. Resolving these interface limitations can substantially enhance the security testing capabilities of LM agents.

## Method

### Overall Architecture
- **Input**: CTF challenge descriptions, relevant files, and access permissions for remote services.
- **Intermediate Stage**: The LM agent utilizes interactive tools to perform vulnerability analysis, reverse engineering, debugging, and exploitation.
- **Output**: CTF flag (the proof-of-exploitation string).

### Key Designs
1. **Interactive Agent Tools**:

    - **Debugger Interface**: Wraps tools like GDB to allow the agent to set breakpoints, step through code, and inspect memory states.
    - **Server Connection Tool**: Wraps netcat/pwntools to facilitate multi-turn interactive communication with remote services.
    - Enables LM agents to run interactive terminal programs requiring continuous inputs and outputs for the first time.
    - **Design Motivation**: Interactive programs are fundamental utilities for security researchers; without them, agents cannot construct complex exploit chains.

2. **Agent Architecture and Tool Integration**:

    - Built upon the SWE-agent framework.
    - A unified action space containing file operations, code execution, and newly added interactive tools.
    - The agent can flexibly switch between different tools within the same session.
    - **Design Motivation**: A unified tool interface lowers the barrier for agents when employing complex tools.

3. **Data Leakage Analysis and Soliloquizing**:

    - Developed a new methodology to quantify the degree of data leakage in CTF benchmarks.
    - Discovered the phenomenon of "soliloquizing," where models hallucinate command outputs without actually executing the corresponding commands.
    - This differs from standard hallucinations, as the model generates false "environmental observations" rather than false factual "knowledge."
    - **Design Motivation**: To ensure the fairness and reliability of evaluation results.

### Loss & Training
This work is an inference-time tool-augmented paradigm and does not involve model training.

## Key Experimental Results

### Main Results

| Benchmark | Challenges | EnIGMA Solve Rate | Prev. SOTA | Is SOTA? |
|------|--------|-------------|---------|---------|
| NYU CTF | ~100 | Highest | - | ✓ |
| Intercode-CTF | ~100 | Highest | - | ✓ |
| CyBench | ~100 | Highest | - | ✓ |
| Total | 390 | Significant Improvement | - | 3/4 SOTA |

### Ablation Study

| Configuration | Solve Rate Delta | Description |
|------|----------|------|
| No Interactive Tools | Baseline | Cannot complete challenges requiring interaction |
| + Debugger | ↑ Significant | Massive improvements in PWN/RE challenges |
| + Server Connection | ↑ Significant | Massive improvements in remote interaction challenges |
| All Tools | ↑ Maximum | Comprehensive improvement |
| Soliloquizing Detection | - | Occurs in approximately X% of responses |

### Key Findings
- Interactive tools serve as the core driver for performance gains; without them, many challenges are structurally impossible to complete.
- SOTA performance is achieved on three benchmarks, validating the generalizability of the proposed approach.
- The soliloquizing phenomenon warrants caution: language models fabricate non-existent program outputs, and these hallucinations are difficult to detect via simplistic methods.
- Data leakage exists in certain benchmarks and must be quantified systematically.

## Highlights & Insights
- **Empowering LM agents to use interactive terminal programs for the first time**, which represents a major breakthrough in tool use capabilities.
- Soliloquizing constitutes a novel classification of hallucinations, distinct from factual knowledge hallucinations.
- The proposed data-leakage quantification methodology is generalizable and applicable to other LM agent evaluation scenarios.
- Demonstrates that tool design can impact an agent's domain-specific performance more significantly than model selection.

## Limitations & Future Work
- High-difficulty challenges of a creative nature remain challenging.
- Security-related tools present a double-edged sword effect, necessitating threat modeling against potential abuse.
- The methods can be extended to larger-scale, real-world vulnerability discovery scenarios.
- This work can be combined with agent memory to enhance planning capabilities for long-horizon, multi-step exploit chains.

## Related Work & Insights
- Built on the SWE-agent framework, inheriting its agent-computer interface design philosophy.
- Complements benchmarking efforts such as CyberBench and InterCode-CTF.
- Insight: LM agents in specialized domains require domain-specific interactive tools; general-purpose shell commands are far from sufficient.

## Rating
- Novelty: ⭐⭐⭐⭐ The interactive tool interface is a significant contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Coverage across 4 benchmarks, 390 challenges, and comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear problem formulations and systematic analyses.
- Value: ⭐⭐⭐⭐ Substantially advances the application of LM agents in cybersecurity.


---

## Supplementary Thinking

### Relationship with Domain Trends
The research direction of this paper is closely aligned with several major trends in current AI research: model capability evaluation and reliability assurance, parameter-efficient fine-tuning and model compression, and AI safety and alignment. From a methodological standpoint, this paper represents an exploration into the deeper mechanisms of LLMs, helping to accelerate the paradigm shift from empirically-driven to theoretically-driven research.

### Key Recommendations for Future Research
1. Combine the core concept with other modalities (vision, audio, multimodal) to validate the cross-modal generalization performance of the method.
2. Validate findings on larger-scale models (70B+) and newer architectures (such as Mixture-of-Experts).
3. Explore combinations with reinforcement learning and online learning to achieve dynamic adaptation.
4. Develop automated evaluation and optimization toolkits to lower the barrier to adopting this method.
5. Consider intersections with LLM alignment research to explore synergistic optimization of safety and performance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] AAAR-1.0: Assessing AI's Potential to Assist Research](aaar-10_assessing_ais_potential_to_assist_research.md)
- [\[ICLR 2026\] HackWorld: Evaluating Computer-Use Agents on Exploiting Web Application Vulnerabilities](../../ICLR2026/llm_evaluation/hackworld_evaluating_computer-use_agents_on_exploiting_web_application_vulnerabi.md)
- [\[ICML 2025\] Communicating Activations Between Language Model Agents](communicating_activations_between_language_model_agents.md)
- [\[ICML 2026\] Multi$^2$: Hierarchical Multi-Agent Decision-Making with LLM-Based Agents in Interactive Environments](../../ICML2026/llm_evaluation/multi2_hierarchical_multi-agent_decision-making_with_llm-based_agents_in_interac.md)
- [\[ACL 2025\] From Tools to Teammates: Evaluating LLMs in Multi-Session Coding Interactions](../../ACL2025/llm_evaluation/from_tools_to_teammates_evaluating_llms_in_multi-session_coding_interactions.md)

</div>

<!-- RELATED:END -->

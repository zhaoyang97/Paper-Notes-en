---
title: >-
  [Paper Note] EET: Experience-Driven Early Termination for Cost-Efficient Software Engineering Agents
description: >-
  [ACL 2026][Code Intelligence][Software Engineering Agent] Ours proposes EET—an early termination method driven by historical experience that identifies invalid iterations during the patch generation and patch selection s…
tags:
  - "ACL 2026"
  - "Code Intelligence"
  - "Software Engineering Agent"
  - "Cost Optimization"
  - "Experience-Driven"
  - "Early Termination Strategy"
  - "SWE-bench"
date: 2026-05-08
content_hash: 1fb565049c4e7ec3
---

# EET: Experience-Driven Early Termination for Cost-Efficient Software Engineering Agents

**Conference**: ACL 2026  
**arXiv**: [2601.05777](https://arxiv.org/abs/2601.05777)  
**Code**: [GitHub](https://github.com/IanWalls/EET)  
**Area**: Code Intelligence  
**Keywords**: Software Engineering Agent, Cost Optimization, Experience-Driven, Early Termination Strategy, SWE-bench

## TL;DR

Ours proposes EET—an early termination method driven by historical experience that identifies invalid iterations during the patch generation and patch selection stages to terminate early. It reduces the total cost of SE Agents by 19%-55% (average 32%) while incurring almost no loss in task performance (0.2% maximum).

## Background & Motivation

**Background**: LLM-based Software Engineering (SE) Agents have made significant progress in automated issue resolution, with frameworks like Agentless, Mini-SWE-Agent, and Trae Agent performing exceptionally well on SWE-bench.

**Limitations of Prior Work**: The prohibitive monetary cost of SE Agents remains a major barrier to practical deployment (53% of developers cite cost as a hurdle). Due to the "token snowball" effect, increasing conversation history leads to super-linear cost growth; furthermore, invalid iterations on difficult or unsolvable problems amplify waste.

**Key Challenge**: Existing cost optimization methods (such as turn-control) significantly impair task performance (an average 10.7% drop) while reducing costs. Reducing costs substantially while maintaining performance remains a core challenge.

**Goal**: To propose a universal early termination optimization method that can be seamlessly integrated into various SE Agents to significantly reduce costs while maintaining task performance.

**Key Insight**: Drawing on the intuition that experienced developers can locate solutions directly without excessive trial and error, structured historical experience is used to guide the Agent in skipping redundant iterations.

**Core Idea**: Historical issue-solving experiences are distilled into structured knowledge (task abstraction + trajectory summary + confidence evaluation), which is utilized during the patch generation and selection stages of new tasks to determine if early termination is feasible.

## Method

### Overall Architecture

EET consists of two main components: (1) Experience Generation—distilling structured experience objects from historical issue-solving records and storing them in an experience bank; (2) Early Termination Mechanism—making termination decisions using retrieved relevant experiences during the patch generation stage (at milestone checkpoints) and the patch selection stage (using confidence thresholds).

### Key Designs

1. **Structured Experience Representation and Retrieval**:
    - **Function**: Compresses raw execution trajectories into compact, reusable experience objects.
    - **Mechanism**: Each experience object contains `task_description` (issue abstraction), `execution_summary` (trajectory summary), `evaluation_result` (all are "pass"), `confidence`, and `confidence_reason` (quality assessment). Only successfully resolved experiences are retained.
    - **Design Motivation**: Raw trajectories are noisy and incur high token overhead, while excessive compression loses usable signals. A structured representation balances information density and utility.

2. **Milestone Early Termination in Patch Generation**:
    - **Function**: Identifies when to stop iterating within a single patch generation process.
    - **Mechanism**: Code modification and test execution are defined as milestones. After each milestone, a confidence score is evaluated; if it exceeds threshold $\tau^{gen}$, the process terminates early.
    - **Design Motivation**: Signals for patch quality may emerge after code changes (structural alignment) or after test execution (dynamic feedback). The dual-milestone design covers both scenarios.

3. **Double-Threshold Early Termination in Patch Selection**:
    - **Function**: Dynamically controls the number of candidate patches to be generated.
    - **Mechanism**: For each generated patch, confidence is assessed based on patch content, execution trajectory, and historical experience. If it exceeds $\tau_{upper}^{sel}$, generation stops (patch is good enough); if it falls below $\tau_{lower}^{sel}$, it also stops (current task is too difficult).
    - **Design Motivation**: Avoids inefficiencies caused by a fixed number of patches—simple problems do not require multiple candidates, and for difficult problems, generating more patches provides no benefit.

### Loss & Training

EET is an inference-time optimization method and does not involve training. Key hyperparameters include the TF-IDF similarity threshold $\tau_{sim}$, the generation termination threshold $\tau^{gen}$, and the selection upper/lower thresholds $\tau_{upper}^{sel} / \tau_{lower}^{sel}$, which are tuned on 100 independent validation samples from SWE-bench. The experience bank is generated from SWE-bench Lite (207 unique tasks).

## Key Experimental Results

### Main Results

| Agent + Backend | Success Rate Change | API Calls | Input Tokens | Output Tokens | Total Cost Change |
|-------------|-----------|---------|-----------|-----------|-----------|
| Agentless + GPT-5-mini | +7.8% | -26.4% | -51.8% | -51.0% | **-55.1%** |
| Agentless + DeepSeek-V3.2 | +7.2% | -25.5% | -31.9% | -35.0% | -32.2% |
| Mini-SWE + GPT-5-mini | +1.0% | -7.9% | -13.7% | -3.7% | -19.4% |
| Mini-SWE + DeepSeek-V3.2 | +0.6% | -8.4% | -13.6% | -4.4% | -19.3% |
| Trae + GPT-5-mini | 0.0% | -29.9% | -30.4% | -28.0% | -28.2% |
| Trae + DeepSeek-V3.2 | -0.2% | -26.5% | -37.7% | -28.2% | -36.7% |
| **Average** | **+2.7%** | **-20.8%** | **-29.9%** | **-25.1%** | **-31.8%** |

### Ablation Study

| Variant (Trae + GPT-5-mini) | Success Rate Change | Total Cost Change |
|-----------------------------|---------------------|-------------------|
| Full EET | 0.0% | -28.2% |
| Remove Experience Injection | -10.4% | -58.9% |
| Remove Early Termination | +0.4% | +3.1% |

### Key Findings

- EET achieved early termination for an average of 11.3% of issues (ranging from 8.6% to 14.0%), where cost savings were most significant.
- The greatest gain was observed with Agentless (success rate actually increased by 7.2-7.8%), as experience guidance compensated for its rigid workflow.
- Comparison with Turn-control: While Turn-control reduces cost more (-41.4%), it results in a massive drop in success rate (-10.7%).
- LLM confidence scores exhibit high calibration: patches with confidence >90 had a pass rate of 63.6%-92.6%, while those <40 were only 8.7%-13.8%.
- Cross-repository transfer experiments indicate that the captured experiences represent general debugging patterns rather than repository-specific cues.

## Highlights & Insights

- The method is highly universal and can be integrated plug-and-play into SE Agents of different paradigms (fixed workflow, autonomous planning, or generate-then-select).
- The design of the "Experience" concept is sophisticated: it is not simple RAG retrieval of raw trajectories, but a distillation into structured knowledge containing confidence assessments.
- The double-threshold design handles both "good enough to stop" and "too hard to continue" scenarios, proving more rational than single-threshold approaches.
- Ablation studies clearly demonstrate the complementary relationship between experience injection and the early termination mechanism.

## Limitations & Future Work

- Building the experience bank relies on historical data, posing a cold-start problem for entirely new domains.
- Evaluation was limited to SWE-bench Verified; generalization to industrial scenarios remains to be validated.
- Early termination decisions rely on the calibration quality of LLM confidence outputs, which varies across models.
- While currently focused on SE Agents, the design philosophy (experience-driven early termination) is domain-agnostic and could be extended to general multi-step reasoning agents.

## Related Work & Insights

- Difference from RAG-based agent memory (e.g., MetaGPT, MemoryBank): EET's experiences specifically serve cost optimization rather than just performance enhancement.
- Analysis of the "token snowball" effect by Fan et al. reveals the root of the cost issue; EET provides a solution from the perspective of experience reuse.
- Implications for Agent system design: Cost optimization should be treated as a first-class citizen rather than a secondary concern to performance.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First systematic use of experience-driven early termination to solve the SE Agent cost problem; the perspective is novel and practical.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluated across 3 Agents × 2 LLM backends, including baseline comparisons, ablations, and cross-repo transfer analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, accurate method description, and comprehensive experimental design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Ambig-SWE: Interactive Agents to Overcome Underspecificity in Software Engineering](../../ICLR2026/code_intelligence/ambig-swe_interactive_agents_to_overcome_underspecificity_in_software_engineerin.md)
- [\[ACL 2026\] Taming System Complexity: Demystifying Software Engineering Agents in Diagnosing Linux Kernel Faults](taming_system_complexity_demystifying_software_engineering_agents_in_diagnosing_.md)
- [\[NeurIPS 2025\] SWE-rebench: An Automated Pipeline for Task Collection and Decontaminated Evaluation of Software Engineering Agents](../../NeurIPS2025/code_intelligence/swe-rebench_an_automated_pipeline_for_task_collection_and_decontaminated_evaluat.md)
- [\[ACL 2026\] SWE-QA: Can Language Models Answer Repository-level Code Questions?](swe-qa_can_language_models_answer_repository-level_code_questions.md)
- [\[ACL 2026\] CollabCoder: Plan-Code Co-Evolution via Collaborative Decision-Making for Efficient Code Generation](collabcoder_plan-code_co-evolution_via_collaborative_decision-making_for_efficie.md)

</div>

<!-- RELATED:END -->

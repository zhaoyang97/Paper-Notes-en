---
title: >-
  [Paper Note] UI-Evol: Automatic Knowledge Evolving for Computer Use Agents
description: >-
  [ICML 2025][Computer Use Agent] This paper proposes **UI-Evol**, a plug-and-play knowledge evolution module. Through a two-stage process of **Retrace** and **Critique**, it automatically corrects retrieved external knowledge based on the agent's actual interactions with the environment. This bridges the knowledge-execution gap where "knowledge is correct but execution fails," significantly improving success rates and drastically reducing behavioral variance on OSWorld.
tags:
  - "ICML 2025"
  - "Computer Use Agent"
  - "Knowledge Evolution"
  - "GUI Interaction"
  - "RAG"
  - "Self-Improvement"
  - "OSWorld"
date: 2026-05-08
content_hash: d2f5dedb1c24096d
---

# UI-Evol: Automatic Knowledge Evolving for Computer Use Agents

**Conference**: ICML 2025  
**arXiv**: [2505.21964](https://arxiv.org/abs/2505.21964)  
**Authors**: Ziyun Zhang, Xinyi Liu, Xiaoyi Zhang, Jun Wang, Gang Chen, Yan Lu  
**Institutions**: Microsoft Research Asia  
**Area**: Agent / GUI Agent  
**Keywords**: Computer Use Agent, Knowledge Evolution, GUI Interaction, RAG, Self-Improvement, OSWorld

## TL;DR

This paper proposes **UI-Evol**, a plug-and-play knowledge evolution module. Through a two-stage process of **Retrace** and **Critique**, it automatically corrects retrieved external knowledge based on the agent's actual interactions with the environment. This bridges the knowledge-execution gap where "knowledge is correct but execution fails," significantly improving success rates and drastically reducing behavioral variance on OSWorld.

## Background & Motivation

### Problem Definition

Computer Use Agents need to interact with GUIs to complete complex, long-chain tasks. Current SOTA methods (such as Agent S2) employ RAG to retrieve external knowledge from the web as a soft prior for task planning. However, the authors identify a Key Challenge:

> Even if **90% of the retrieved knowledge is judged as correct by humans**, the success rate of the best agent is only **41%**.

This indicates that "knowledge is correct $\neq$ knowledge is executable." The authors refer to this discrepancy as the **Knowledge-Execution Gap**.

### Root Cause Analysis

The failure of actually retrieved knowledge stems from three categories:

**Omission of essential intermediate steps**: Operations considered "natural" by humans are omitted, which the agent cannot autonomously complete.

**Inconsistency between assumptions and initial conditions**: The system state assumed by the retrieved knowledge does not match the actual environment.

**Recommending overly complex action paths**: For example, suggesting dragging the mouse to select text, which is extremely difficult for an agent to execute precisely (Ctrl+A is much more reliable).

### Differences from Prior Work

- Monolithic Agents (e.g., UI-TARS, CogAgent): Rely on end-to-end single models operating directly, without utilizing external knowledge.
- Modular Agents (e.g., Agent S/S2, OSCAR): Introduce external tools/knowledge but lack a knowledge correction mechanism.
- Self-Evolution (e.g., VOYAGER, Self-Refine): Depend on subjective reflection rather than extracting objective action sequences from visual evidence.

The key innovation of UI-Evol lies in: **instead of relying on the agent's subjective outputs, it extracts objective action sequences by comparing screenshots, and then refines the knowledge using structured reasoning**.

## Method

### Overall Architecture

Given task instruction + initial external knowledge → Agent execution → Interactive trajectory (screenshot sequence) → **Retrace Stage** → **Critique Stage** → Evolved knowledge written back to knowledge base → Evolved knowledge used for the next execution round.

### Stage 1: Retrace

**Goal**: Recover the **objective action sequence** of the agent from the screenshot sequence, eliminating hallucinations and invalid actions present in the subjective action sequence.

**Mechanism**:
The "subjective action sequence" actually output by the agent may not align with real UI state changes (due to LMM hallucinations, operations not taking effect, etc.). Therefore, the Retrace stage does not trust the agent's self-reports but reconstructs the real actions from the perceptual level.

For each step $t$ in the trajectory:
- Input: Pre-action screenshot $O_t$ and post-action screenshot $O_{t+1}$.
- LMM analyzes the difference between the two screenshots to infer the action that actually occurred, denoted as $\hat{A}_t$.
- If there is no change between the two frames, then $\hat{A}_t = \text{null}$ (the action was invalid).
- The union of all $\hat{A}_t$ forms the **objective action sequence**.

**Design Advantages**:
- Filters out the noise of invalid operations and hallucinated actions.
- Provides a reliable factual basis for the subsequent Critique stage.

### Stage 2: Critique

**Goal**: Refine knowledge using the external knowledge as a reference anchor, comparing it against the objective action sequence through multi-step Chain-of-Thought (CoT) reasoning.

Four-step reasoning chain design:

**Step 1 — Completion Assessment**:
Evaluate whether the task was completed successfully. By comparing the results of the objective action sequence with the task goal, the outcome is clearly classified as success, partial completion, or failure.

**Step 2 — Deviation Detection**:
Compare the objective action sequence step-by-step with the plan guided by the original knowledge, identifying discrepancies and diagnosing root causes (e.g., "output/screen misunderstanding", "mismatched action granularity").

**Step 3 — Alternative Exploration**:
Analyze whether the agent attempted effective strategies outside the initial knowledge plan during execution. Mining these alternative solutions helps enrich the diversity of the knowledge base.

**Step 4 — Mitigation with Rationales**:
Synthesize the analysis from the first three steps to generate a new version of the knowledge. Corrections are generated with causal explanations, and the output format remains consistent with the original knowledge (structured as a sub-task list).

### Knowledge Format

Adopts the format of Agent S2: decomposing tasks into a structured sub-task list. The evolved knowledge replaces the original knowledge and is written back to the knowledge base for subsequent executions.

## Training and Experimental Settings

### Benchmarks & Environments

- **Benchmark**: OSWorld (369 open-ended computer tasks, covering five categories: OS, Daily, Office, Professional, Workflow)
- **Maximum Steps**: 15 steps
- **Baseline System**: Agent S2 (the SOTA on OSWorld at that time)

### Backbone Model Selection

- Retrace and Agent Execution: GPT-4o
- Critique Stage: OpenAI-o3 (offering stronger reasoning capabilities)
- To eliminate the randomness of web searches, the entire Perplexica knowledge snapshot was frozen.
- Temperature was set to 0, and all controllable hyperparameters were fixed.

### Parallel Evaluation Framework

The original single-machine evaluation of OSWorld took about 10 hours. The authors constructed a parallel evaluation architecture on Azure:

- Each Azure instance is allocated a virtual machine.
- Tasks are evenly distributed across 30 instances for parallel execution.
- Running time is reduced from ~10h to ~2.5h (approximately a 4x speedup).
- Bypasses the overhead of repeatedly building Docker images in Windows Agent Arena.

### Stability Evaluation

Each experiment was repeated 3 times, reporting the mean and standard deviation. Even with all hyperparameters and knowledge snapshots fixed, the repeated evaluation results still show fluctuations—a phenomenon previously overlooked by the research community.

## Main Results

### Core Performance Comparison (Table 1)

| Method | Backbone Model | Average Success Rate | Standard Deviation |
|------|----------|-----------|--------|
| Agent S2* (reproduced) | GPT-4o | 19.5% | $\pm 1.00$ |
| **+ UI-Evol** | GPT-4o | **22.0%** | **$\pm 0.71$** |
| Agent S2* | OpenAI-o3 | 25.6% | $\pm 1.09$ |
| **+ UI-Evol** | OpenAI-o3 | **28.4%** | **$\pm 0.26$** |

Key Findings:
- UI-Evol **improves success rates and reduces standard deviation** across both backbone models.
- The standard deviation of o3 + UI-Evol is only 0.26, which is approximately 4.19 times lower than that of GPT-4o.
- **Models with stronger reasoning capabilities benefit more from evolved knowledge.**

### Ablation Study: Trajectory Selection Strategy (Table 2)

| Selection Strategy | Selection Success Rate (SSR) | Average Success Rate |
|---------|----------------|-----------|
| Random selection | 70% | 22.0% |
| Completion-based selection | 85% | 22.7% |

SSR Definition: $\text{SSR} = \frac{N_{\text{succ}}}{N_{\text{solv}}}$, representing the ratio of selected trajectories that are successful.

Conclusion: UI-Evol is robust to trajectory quality and can effectively evolve knowledge even when given sub-optimal trajectories.

### Knowledge Transfer Experiment (Table 3)

After evolving knowledge using trajectories generated by o3, the knowledge is provided to GPT-4o:

| Knowledge Source | GPT-4o Success Rate |
|---------|-------------|
| Original Web Knowledge | 19.5% |
| Evolved from GPT-4o Trajectories | 22.0% |
| Evolved from o3 Trajectories | 22.4% |

Evolved knowledge is **transferable across models**; the knowledge evolved from o3 trajectories is even slightly better than that evolved from GPT-4o's own trajectories.

### Case Study

Task: "Capitalize the first letter of each word in a LibreOffice Writer document."

- Web knowledge suggests "drag the mouse to select all text" → the agent only selects part of the text → task failure.
- The UI-Evol Retrace stage accurately identifies that the agent only selected some paragraphs.
- The Critique stage diagnoses this as "output/screen misunderstanding" and recommends using **Ctrl+A** instead of dragging the mouse.
- The evolved knowledge corrects Step 2, leading to successful execution thereafter.

## Computational Cost

| Phase | Time | Token Consumption/Task | Model |
|-----|------|----------------|------|
| Original Trajectory Collection | ~2h (30 instances in parallel) | — | GPT-4o |
| Retrace | ~1h (12 threads) | ~85k input + 400 output | GPT-4o |
| Critique | Included in the above 1h | ~800 input + 150 output | OpenAI-o3 |
| **Total** | ~3h | Average \$0.22/task | Entire benchmark ~\$81 |

## Limitations & Future Work

1. **Dependence on a strong baseline system**: UI-Evol is implemented as a plugin for Agent S2; its generalization to other agent frameworks has not been verified (despite being claimed as plug-and-play).
2. **Only validated on OSWorld**: The evaluation size of 369 tasks is limited, and it has not been tested on other benchmarks such as WebArena or AndroidWorld.
3. **Single-round evolution**: Currently, only one round of "execution → evolution" is performed, leaving the effects and convergence of multi-round iterative evolution unexplored.
4. **Retrace depends on LMM visual capability**: If the screenshot differences are subtle (such as minor edits in text content), the LMM might miss crucial changes.
5. **Non-zero cost**: The API cost of \$0.22 per task needs to be carefully considered for large-scale deployments.
6. **No discussion on knowledge conflicts**: After multiple evolutions, conflicts may arise among entries in the knowledge base, and a conflict-resolution mechanism is missing.

## Reproducibility Key Points

- Benchmark Code: OSWorld (open-source)
- Agent Framework: Agent S2 (open-source)
- Search Engine: Perplexica (open-source, but requires self-hosting an instance)
- Backbone Models: GPT-4o, OpenAI-o3 (closed-source, API required)
- Knowledge snapshot frozen, temperature fixed at 0.
- The parallel evaluation framework is based on Azure ML Jobs (reproducible but requires cloud resources).
- Each experiment was repeated 3 times to report standard deviations.

## Highlights & Insights

**Pros**:
- Precise problem definition: The setting (90% knowledge correctness → 41% success rate) convincingly exposes the knowledge-execution gap.
- Pragmatic methodology: Using visual evidence instead of subjective self-reports in Retrace is a simple yet crucial design choice; the four-step CoT reasoning in Critique is clearly structured.
- Rigorous experiments: For the first time in GUI agent research, **behavioral variance** is systematically analyzed and quantified through repeated trials, which is much more honest than only reporting the single best result.
- The knowledge transferability experiments hold substantial practical value.

**Cons**:
- The performance improvement is relatively limited (19.5% → 22.0%), leaving significant room for absolute performance gains.
- Only validated on Agent S2 + OSWorld; generality warrants support from broader experiments.
- Lacks a fine-grained analysis of which types of tasks benefit the most from evolution.
- Lacks direct comparison with contemporaneous self-improvement methods such as Self-Refine and Reflexion.

**Overall**: This paper tackles a real and underestimated problem (knowledge usability vs. knowledge correctness) with a rustic yet effective method. The reduction in standard deviation is more meaningful than the absolute performance gain, making the agent's behavior more predictable and deployable. As an ICML paper, its contribution lies in the problem identification and experimental methodology (parallel evaluation + reporting variance over repeated experiments) rather than the complexity of the method itself.

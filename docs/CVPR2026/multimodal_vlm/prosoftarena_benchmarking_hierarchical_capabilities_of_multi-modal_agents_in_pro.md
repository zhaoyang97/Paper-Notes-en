---
title: >-
  [Paper Note] ProSoftArena: Benchmarking Hierarchical Capabilities of Multi-modal Agents in Professional Software Environments
description: >-
  [CVPR 2026][Multimodal VLM][Paper Note] ProSoftArena is the first multimodal agent benchmark targeting **professional software** (13 tools including CAD, ChemDraw, ArcGIS, Photoshop, etc.). It categorizes agent capabilities into five levels (L1–L5), utilizes automated scoring within real Windows virtual machines based on execution results, and introduces a "
tags:
  - CVPR 2026
  - Multimodal VLM
date: 2026-05-08
content_hash: 5150f045a4af4a55
---
# ProSoftArena: Benchmarking Hierarchical Capabilities of Multi-modal Agents in Professional Software Environments

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Ai_ProSoftArena_Benchmarking_Hierarchical_Capabilities_of_Multi-modal_Agents_in_Professional_Software_CVPR_2026_paper.html)  
**Code**: https://prosoftarena.github.io (Project Page)  
**Area**: Multimodal VLM / Agent  
**Keywords**: Multimodal agents, professional software, capability hierarchy benchmark, real virtual machine environment, human-agent collaboration evaluation

## TL;DR
ProSoftArena is the first multimodal agent benchmark targeting **professional software** (13 tools including CAD, ChemDraw, ArcGIS, Photoshop, etc.). It categorizes agent capabilities into five levels (L1–L5), utilizes automated scoring within real Windows virtual machines based on execution results, and introduces a "Human-in-the-Loop" evaluation. Results reveal that the strongest agents achieve only a 20.6% success rate in software-level tasks (L2) and fail almost entirely in cross-software workflows (L3).

## Background & Motivation
**Background**: "Computer-use agents" driven by multimodal large models are advancing rapidly, capable of daily tasks like web navigation and file management. Existing benchmarks (GAIA, OSWorld, WindowsAgentArena, TheAgentCompany, ScienceBoard) primarily focus on browsers and basic desktop applications.

**Limitations of Prior Work**: In real scientific and industrial practices, experts rely on **professional software** such as SolidWorks (engineering), ChemDraw (molecular modeling), and various Adobe suites (digital creation). These tools feature dense, complex interfaces and require deep domain knowledge. Existing benchmarks rarely address these tools, creating a gap between evaluated "agent capabilities" and real-world productivity.

**Key Challenge**: Professional software represents a jump in complexity compared to general desktop apps (extremely dense GUIs, deep functionality, technical terminology). Current evaluations lack both coverage of this complexity and a characterization of "capability levels"—the ability to click a button does not imply the ability to organize a software-level workflow or coordinate across applications.

**Goal**: Construct a benchmark to systematically probe the capability boundaries of agents in professional software, requiring (1) capability hierarchy, (2) multi-disciplinary coverage, (3) reproducible real environments, (4) objective execution-based scoring, and (5) quantification of human-agent collaboration efficiency.

**Key Insight**: The authors argue that measuring professional software capabilities should not rely solely on an "autonomous success rate." Instead, it requires a progressive hierarchy of difficulty and the inclusion of "low-cost human intervention for error correction," as professional tools are often used in human-machine collaboration scenarios.

**Core Idea**: Systematically quantify professional software agent capabilities using a "four-piece set": five-level capability hierarchy + real VM execution environment + execution-based automated scoring + Human-in-the-Loop evaluation.

## Method

### Overall Architecture
ProSoftArena is a benchmark and evaluation platform rather than a model. Its framework consists of four parts: ① A **capability hierarchy** from L1 to L5, ranging from "atomic GUI operations → software-level tasks → cross-software pipelines → open creation → project-level orchestration"; ② A **real Windows 11 VM environment** running in Docker containers with 13 pre-installed professional tools, using snapshots to ensure clean starts; ③ An **execution-based automated evaluation framework** where each task uses custom scripts to verify final system states and outputs, returning 1.0/0.0; ④ A **Human-in-the-Loop evaluation paradigm**, including "Human-Initiated Takeover" and "Agent-Initiated Asking" modes to quantify collaboration efficiency.

The evaluation workflow follows: Task initialization scripts restore the VM to a specific context → The agent interacts with the environment via mouse/keyboard under observations of "screenshots + system signals" (formalized as a POMDP where the MLLM acts as the policy) → Upon outputting DONE/FAIL or reaching step limits, post-processing scripts extract artifacts and run evaluation functions for scoring.

### Key Designs

**1. Five-Level Capability Hierarchy: Separating "Operation" from "Work"**

Existing benchmarks provide a single success rate, masking the gap between clicking buttons and completing professional tasks. ProSoftArena establishes a ladder where each level encompasses previous capabilities: **L1 Operation**—Executing single atomic GUI operations (e.g., adjusting brightness in Photoshop); **L2 Software**—Planning and executing sequences within a single app (e.g., drawing a badminton court in AutoCAD); **L3 Pipeline**—Cross-software workflows (e.g., aggregating population in ArcGIS then performing statistics in Excel); **L4 Creation**—Open-ended creation (e.g., designing a cartoon logo); **L5 Project**—End-to-end industrial/scientific projects. This study evaluates up to L4.

**2. Real VM Execution Environment: Ensuring Reproducibility and Safety**

Professional software depends heavily on system integration and Windows stability. The authors host a full Windows 11 VM in Docker with 13 fixed-version applications (Illustrator, Photoshop, ImageJ, ChemDraw, RGui, Excel, VSCode, NVIVO, ArcGIS, ANSYS, MultiSim, AutoCAD, SolidWorks). Isolation ensures no irreversible damage to the host, while **snapshot mechanisms** guarantee deterministic initial states for every task, ensuring consistency and reproducibility.

**3. Execution-based Automated Evaluation Framework: Scoring by Final State**

The correctness of professional tasks must be judged by the final output. The framework customizes evaluation functions for each task to verify system states and files: for instance, Photoshop editing tasks use Mean Squared Error (MSE) thresholds against a target image, while VSCode configurations parse user files to confirm settings. Scenarios involve: (i) post-processing (saving files), (ii) retrieving artifacts or state from the VM, (iii) running evaluation functions.

**4. Human-in-the-Loop Evaluation Paradigm: Quantifying Collaboration Efficiency**

In real-world settings, professional software is often used collaboratively. Two modes are introduced: **Mode I: Human-Initiated Takeover (HIT)**—Human experts monitor in real-time and intervene to correct errors or dangerous operations before returning control; **Mode II: Agent-Initiated Asking (AIA)**—The agent is provided an `ASK ACTION` tool to request human execution when uncertain. This measures actual utility and exposes the model's inability to seek help proactively.

## Key Experimental Results

Evaluated agents include closed-source MLLMs (GPT-4o, GPT-5, o3), open-source MLLMs (Qwen2.5-VL, GLM-4.5V), and specialized computer-use agents (Agent-S, UI-TARS, Claude 4 Computer Use). Observations include Screenshot, Screenshot+Accessibility (A11y) tree, and Set-of-Marks (SoM).

### Main Results (Table 3, Overall, Screenshot+A11y Observation)

| Model | Type | L1 SR(%) | L2 SR(%) | L3 SR(%) |
| :--- | :--- | :--- | :--- | :--- |
| GLM-4.5V | Open MLLM | 6.0 | 0.8 | 0.0 |
| Qwen2.5-VL | Open MLLM | 16.5 | 4.3 | 0.0 |
| GPT-4o | Closed MLLM | 17.5 | 1.8 | 0.0 |
| o3 | Closed MLLM | 32.2 | 8.7 | 0.0 |
| GPT-5 | Closed MLLM | 42.5 | 11.8 | 0.0 |
| UI-TARS-1.5-7B | Specialized agent | 10.0 | 1.2 | 0.0 |
| Agent-S | Specialized agent | **48.6** | 17.1 | 0.0 |
| Claude 4 Computer Use | Specialized agent | 45.1 | **20.6** | 0.0 |

Key Conclusions: ① **L1 still has significant room for improvement**—even the strongest models are below 50%; ② **L2 is the primary bottleneck**—the success rate drops sharply from L1 to L2, indicating that "executing atomic actions" $\neq$ "composing them into coherent sequences"; ③ **L3 is nearly 0% across the board**—cross-app planning and state consistency remain out of reach; ④ Specialized computer-use agents are overall strongest, suggesting that explicit UI priors are essential.

### Statistics and Domain Distribution (Table 2)

| Capability Level | Task Count (%) | Avg Steps (Human) | Avg Time (s) |
| :--- | :--- | :--- | :--- |
| L1 Operation | 252 (55.3%) | 5.1 | 14.8 |
| L2 Software | 164 (35.9%) | 20.4 | 83.1 |
| L3 Pipeline | 20 (4.4%) | 86.9 | 506.8 |
| L4 Creation | 20 (4.4%) | — | — |
| **Total** | **456** | **12.9** | **52.6** |

Human steps and time increase significantly across levels, highlighting the task complexity of ProSoftArena compared to existing benchmarks.

### Ablation Study

**Visual Grounding Input (Table 4, Illustrator + Qwen2.5-VL)**:

| Configuration | L1 SR(%) | L2 SR(%) | L1 Time (s) | L1 Cost (tokens) |
| :--- | :--- | :--- | :--- | :--- |
| SoM from A11y | 0.0 | 0.0 | 1146.3 | 308.2k |
| SoM from A11y+Omni | 10.5 | 0.0 | 1761.2 | 414.1k |
| SoM+Screenshot | **21.1** | 6.7 | 1553.8 | 416.2k |
| SoM+Screen+A11y | 5.3 | 0.0 | 1458.9 | 616.1k |

Richer inputs generally improve grounding but increase cost. Notably, performance is **non-monotonic**; combining SoM, Screenshot, and a full A11y tree led to performance degradation, suggesting information overload beyond the model's capacity.

### Key Findings
- **L2 is the Water Parting**: The largest capability gap lies in composing atomic actions into parameterized sequences. L3 cross-software workflows remain at 0% for all models.
- **No Universal Observation Solution**: A11y trees suit element-dense scenarios where coordinates might be missing; SoM benefits icon-dense areas but can suffer from visual occlusion.
- **Domain Priors Help Differentially**: Adding "knowledge cards" (e.g., molecular formulas in ChemDraw) improves L2 performance, with Claude benefiting significantly while Qwen's gain was limited, suggesting Qwen's bottleneck is interaction rather than knowledge.
- **Human-Initiated Takeover (HIT) is Highly Valuable**: For Qwen2.5-VL on L2 VSCode tasks, HIT improved SR from 6.7% to 66.7% and reduced average steps from 44.6 to 12.5. Conversely, Agent-Initiated Asking (AIA) showed minimal benefit because models rarely proactively asked for help, exposing a lack of self-assessment.

## Highlights & Insights
- **Upgrading Capability from Scalar to Ladder**: The L1–L5 hierarchy provides a clear roadmap by showing exactly which level agents fail at, which is far more informative than a single success rate.
- **Real VM Execution + Snapshot Reset**: Handling the reproducibility of professional software evaluation through Docker-hosted Windows and snapshots is a significant engineering contribution.
- **Human-in-the-Loop as a Key Dimension**: HIT/AIA modes quantify collaboration efficiency and reveal the "unknown unknowns" problem—models do not know when they are stuck, a critical insight for designing reliable AI assistants.

## Limitations & Future Work
- **L5 remains unevaluated**: Project-level orchestration is defined but not yet implemented; evaluation currently stops at L4.
- **Small scale for higher levels**: L3 and L4 have only 20 tasks each, limiting statistical reliability in these high-difficulty zones.
- **Platform Binding**: Restricted to Windows 11 as most professional tools are most stable there; cross-OS professional tools (e.g., Linux-based scientific software) are not yet covered.
- **Human Cost**: HIT/AIA requires real-time expert presence, which is difficult to scale. Exploring cheaper proxies (e.g., smaller "coach models") is a future direction.

## Related Work & Insights
- **vs OSWorld / WindowsAgentArena**: These focus on daily OS tasks without capability hierarchies or multi-disciplinary professional software. ProSoftArena targets 13 specialized domains.
- **vs ScienceBoard**: ScienceBoard focuses on scientific workflows but lacks the hierarchical depth and Human-in-the-Loop evaluation found in ProSoftArena.
- **vs TheAgentCompany / GAIA**: These do not target professional software and lack the self-hosted VM infrastructure for complex tool execution.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First professional software benchmark + first capability hierarchy + unique HITL evaluation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 8 agents × 3 observations × multiple disciplines, plus ablations and HITL.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and solid insights; though L3/L4 scales are small.
- Value: ⭐⭐⭐⭐⭐ Exposes the stark reality of current agent performance (L2: 20.6%, L3: 0%) and provides a reproducible platform.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

- **OSWorld**: Benchmarking Multimodal Agents for Open-Ended Desktop Applications (CHI 2024)
- **WindowsAgentArena**: Evaluating Agents in Linux and Windows Environments (ArXiv 2024)
- **ScienceBoard**: Evaluating Scientific Agents in Specialized Environments (ArXiv 2024)

</div>

<!-- RELATED:END -->

## Related Papers

- [\[CVPR 2026\] Hierarchical Attacks for Multi-Modal Multi-Agent Reasoning](hierarchical_attacks_for_multi-modal_multi-agent_reasoning.md)
- [\[CVPR 2026\] Demo2Tutorial: From Human Experience to Multimodal Software Tutorials](demo2tutorial_from_human_experience_to_multimodal_software_tutorials.md)
- [\[CVPR 2026\] VS-Bench: Evaluating VLMs for Strategic Abilities in Multi-Agent Environments](vs_bench_evaluating_vlms_for_strategic_abilities_in_multi_agent_environments.md)
- [\[CVPR 2026\] Multi-Crit: Benchmarking Multimodal Judges on Pluralistic Criteria-Following](multi-crit_benchmarking_multimodal_judges_on_pluralistic_criteria-following.md)
- [\[CVPR 2026\] Chain-of-Thought Guided Multi-Modal Object Re-Identification](chain-of-thought_guided_multi-modal_object_re-identification.md)

</div>

<!-- RELATED:END -->

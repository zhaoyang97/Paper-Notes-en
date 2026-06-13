---
title: >-
  [Paper Note] AutoRPA: Efficient GUI Automation through LLM-Driven Code Synthesis from Interactions
description: >-
  [ICML2026][LLM Agent][GUI Automation] The AutoRPA framework is proposed to automatically distill interaction trajectories of ReAct-style GUI Agents into reusable RPA functions via a Translator-Builder pipeline. Combined…
tags:
  - "ICML2026"
  - "LLM Agent"
  - "GUI Automation"
  - "Robotic Process Automation"
  - "Code Synthesis"
  - "Trajectory Distillation"
date: 2026-05-08
content_hash: 2353d8e5e728c49f
---

# AutoRPA: Efficient GUI Automation through LLM-Driven Code Synthesis from Interactions

**Conference**: ICML2026  
**arXiv**: [2605.21082](https://arxiv.org/abs/2605.21082)  
**Code**: None  
**Area**: LLM Agent  
**Keywords**: GUI Automation, Robotic Process Automation, Code Synthesis, LLM Agent, Trajectory Distillation  

## TL;DR
The AutoRPA framework is proposed to automatically distill interaction trajectories of ReAct-style GUI Agents into reusable RPA functions via a Translator-Builder pipeline. Combined with a hybrid repair strategy for iterative code optimization, it reduces token consumption by 82%~96% while maintaining or exceeding the success rate of the original Agent.

## Background & Motivation

**Background**: LLM-based GUI Agents (e.g., SeeAct, M3A) can complete various GUI tasks through multi-step interactions using the ReAct paradigm. However, these methods require LLM inference for every task instance, resulting in high token consumption and slow execution.

**Limitations of Prior Work**: In practical deployment scenarios, many GUI tasks are repetitive—the same user submitting reports daily or different users booking flights. Repeatedly calling LLM inference for such tasks is expensive and inefficient. Traditional RPA is efficient but relies on manually written scripts, incurring high development/maintenance costs and vulnerability to GUI layout changes.

**Key Challenge**: LLM Agents are flexible but expensive (requiring inference every time), while traditional RPA is efficient but rigid (manually written and hard to generalize). Directly letting LLMs generate complete code often fails due to a lack of environment knowledge; skill learning methods that store successful trajectories have limited generalization capabilities.

**Goal**: Automatically distill the decision logic of LLM Agents into generalized, low-token-consumption RPA functions that can execute robustly across different environment states and task instructions.

**Key Insight**: The authors observe that while ReAct Agents have high inference costs, their successful trajectories contain the complete decision logic for task completion. By converting hard-coded actions into soft-coded ones and then synthesizing RPA code with conditional logic, both flexibility and efficiency can be achieved.

**Core Idea**: Use a Translator to convert hard-coded actions from ReAct into soft-coded actions based on semantic attributes, use a Builder to synthesize robust RPA functions from multiple translated trajectories, and iteratively optimize code quality through a hybrid repair strategy.

## Method

### Overall Architecture
AutoRPA models the GUI environment as a POMDP $(\mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{G}, \mathcal{O})$. The goal is to generate an RPA function $F_k$ for a specified task type $\mathcal{G}^k$ that minimizes token consumption while ensuring task success. The framework consists of three phases: (1) Exploration Phase: A ReAct Agent interacts to generate trajectories, and the Translator converts hard-coded actions into soft-coded ones; (2) RPA Generation Phase: The Builder synthesizes RPA code based on translated trajectories, retrieving detailed information from a trajectory library via RAG; (3) Validation & Repair Phase: The code is validated on seen tasks; if it fails, an Analyzer locates the breakpoint, the ReAct Agent continues exploration to fix it, and the Builder iteratively improves the code accordingly.

### Key Designs

1.  **Translator-Builder Pipeline**:
    - **Function**: Transforms hard-coded interaction trajectories from ReAct Agents into reusable RPA function code.
    - **Mechanism**: The Translator Agent receives each action along with pre- and post-observations, performs robustness analysis, and converts hard-coded actions (e.g., `click(index=2)`) into soft-coded actions based on semantic attribute positioning (e.g., using `find_element` by text content and element type), optionally inserting assertion statements to verify execution effects. The Builder Agent receives simplified translated trajectories $\psi(\tau'_{\text{ReAct}}(g))$ (removing raw observations and keeping only actions and execution summaries) to generate RPA functions containing conditional logic and loops. The Builder uses a Tree-organized RAG mechanism to retrieve detailed observation information from the trajectory library as needed: the bottom layer stores interaction blocks $(o_t, a'_t, \rho_t, o_{t+1})$, the middle layer contains simplified trajectories, and the top layer contains conclusion summaries.
    - **Design Motivation**: Directly generating complete GUI code often fails due to a lack of environmental knowledge; meanwhile, hard-coded actions are fragile to GUI layout changes. The "explore-then-translate-then-synthesize" pipeline leverages ReAct's exploration to gain environment knowledge and uses soft-coding to enhance cross-environment generalization.

2.  **Hybrid Repair Strategy**:
    - **Function**: Iteratively repairs code when validation fails by combining RPA execution with ReAct fallback.
    - **Mechanism**: The RPA code is executed on a seen task until the first failure. An Analyzer Agent diagnoses the breakpoint (analyzing executed trajectories and current observations to output failure reasons, completed sub-tasks, and feasible continuation plans). The ReAct Agent continues the task from the breakpoint (or restarts) to produce a corrective demonstration trajectory $\tau'_{\text{hybrid}}(F_k, g_*) = F_k(g_*) \oplus (A, o_{t*}, a'_{t*}, \rho_{t*}, \ldots, C)$, which the Builder uses to improve the code. Each task allows $M=3$ modification attempts.
    - **Design Motivation**: Unlike simply letting the Builder debug, hybrid repair utilizes the ReAct Agent's exploration in a real environment to obtain actual corrective trajectories, providing the Builder with concrete evidence rather than speculative improvements, which significantly increases the repair success rate.

3.  **Tree-organized Trajectory RAG**:
    - **Function**: Allows the Builder to retrieve detailed observation information from historical interactions as needed, avoiding incorrect code generation due to missing interface states.
    - **Mechanism**: The trajectory library $\mathcal{D}_\tau$ is organized as a three-layer tree—bottom layer for interaction blocks (including screenshots and DOM info), middle layer for simplified trajectories (action summaries), and top layer for conclusion summaries. The Builder can retrieve context level-by-level using the `fetch_info(traj, step)` tool function, pulling multimodal observations only when necessary to balance information completeness and prompt length.
    - **Design Motivation**: Providing full trajectory observations leads to excessively long prompts; providing only simplified trajectories can cause the Builder to make wrong assumptions about interface states. The RAG approach lets the Builder decide which details are required.

## Key Experimental Results

### Main Results

Experiments were conducted on three GUI benchmarks: AndroidWorld (116 task types, 20 real Apps), WebArena (Reddit domain, 19 task types), and MiniWoB++ (53 task types).

| Method | Model | Time (min) ↓ | Tokens (k) ↓ | Success Rate (%) ↑ |
|------|------|-------------|--------------|-------------|
| SeeAct | GPT-4.1 | 5.14 | 58.8 | 25.4 |
| M3A | GPT-4.1 | 2.23 | 103.4 | 48.3 |
| ReAct† | GPT-4.1 | 3.91 | 68.7 | 50.0 |
| AutoRPA (code only) | GPT-4.1 | 1.42 | 2.7 | 47.2 |
| **AutoRPA** | **GPT-4.1** | **1.81** | **12.8** | **51.7** |
| ReAct† | GPT-5 | 8.57 | 142.5 | 74.1 |
| AutoRPA (code only) | GPT-5 | 2.72 | 6.2 | 70.7 |
| **AutoRPA** | **GPT-5** | **4.35** | **30.6** | **75.9** |

On MiniWoB++ (GPT-4.1):

| Method | 9 Hard Tasks Tokens (k) ↓ | Success Rate (%) ↑ | All 53 Tasks Tokens (k) ↓ | Success Rate (%) ↑ |
|------|-------------------------|-------------|------------------------|-------------|
| AdaPlanner | 15.1 | 74.1 | 6.1 | 90.3 |
| AutoManual | 23.2 | 91.1 | 4.6 | 95.2 |
| ReAct† | 16.2 | 84.4 | 9.2 | 92.8 |
| AutoRPA (code only) | 1.0 | 80.0 | 0.9 | 92.5 |
| **AutoRPA** | **1.4** | **91.1** | **1.4** | **95.4** |

### Ablation Study

| Configuration | Success Rate (%) |
|------|-----------|
| AutoRPA (Full) | 51.7 |
| Remove ReAct in Construction | 32.5 |
| Remove Translator in Construction | 40.2 |
| Remove ReAct in Code Repair | 45.5 |
| Remove RAG in Builder | 48.8 |

### Key Findings
- Removing ReAct exploration causes the success rate to plummet from 51.7% to 32.5%, indicating that directly generating GUI code without environment knowledge is unreliable.
- The Translator's contribution is significant (success rate drops 11.5% without it); soft-coded actions are vital for code generalization.
- Executing RPA code alone (code only) achieves success rates close to ReAct, while token consumption drops to 4%~7% of the original, proving that the decision logic for most tasks can indeed be distilled into deterministic code.
- As the number of construction tasks $N$ increases, the success rate of AutoRPA (code only) continues to approach ReAct, verifying that more samples help generate more robust RPA code.
- In highly diverse real-world web environments like WebArena, AutoRPA maintains comparable success rates to existing methods while drastically reducing token consumption.

## Highlights & Insights
- **Trajectory Distillation Paradigm**: Converting online inference of LLM Agents into offline code is essentially a transition from "inference-time computation" to "compile-time computation." This idea can be migrated to any repetitive Agent task (e.g., data processing pipelines, test automation).
- **Soft-coded Translation**: Locating GUI elements via semantic attributes rather than hard-coded positions/indices elegantly solves the classic RPA pain point of script failure due to layout changes. This design philosophy applies to all automation scripts requiring cross-environment generalization.
- **Hybrid Repair = Code Debugging + Environment Exploration**: Instead of letting the LLM debug code purely through imagination, the Agent explores the real environment to obtain corrective trajectories. This "in-the-loop debugging" strategy is more reliable than pure static code repair.

## Limitations & Future Work
- The construction phase still consumes a significant amount of tokens (requiring $N$ task samples per task type + repeated validation/repair), and the author does not fully discuss the equilibrium between construction cost and savings during the testing phase.
- For highly diverse task types (e.g., WebArena), a single RPA function may struggle to cover all scenarios, requiring a fallback to ReAct, which diminishes AutoRPA's advantages.
- Positioning relies on semantic attributes of GUI elements; this may be inapplicable to interfaces poor in attribute information (e.g., pure image UIs).
- Future work could explore automatically determining when it is worthwhile to build RPA for a specific task (ROI analysis), as well as combining local updates of RPA functions with incremental validation to lower maintenance costs.

## Related Work & Insights
- **ReAct Paradigm** (Yao et al., 2023): The foundational paradigm for alternating reasoning and acting; AutoRPA's exploration and repair phases are based on this.
- **AutoManual** (Chen et al., 2024): Generalizes environment rules from interactions to guide subsequent tasks, complementing AutoRPA's skill distillation approach.
- **AdaPlanner** (Sun et al., 2023): A skill-learning method in the Plan-and-Execute paradigm, but dependent on human demonstrations.
- **Insight**: For any LLM inference task requiring repetition, consider the strategy of "using high-cost methods to explore and collect trajectories, then distilling them into low-cost deterministic workflows."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] EvolveR: Self-Evolving LLM Agents through an Experience-Driven Lifecycle](evolver_self-evolving_llm_agents_through_an_experience-driven_lifecycle.md)
- [\[AAAI 2026\] Reflection-Driven Control for Trustworthy Code Agents](../../AAAI2026/llm_agent/reflection-driven_control_for_trustworthy_code_agents.md)
- [\[ACL 2026\] Don't Act Blindly: Robust GUI Automation via Action-Effect Verification and Self-Correction](../../ACL2026/llm_agent/don39t_act_blindly_robust_gui_automation_via_action-effect_verification_and_self.md)
- [\[ICML 2026\] Recovering Policy-Induced Errors: Benchmarking and Trajectory Synthesis for Robust GUI Agents](recovering_policy-induced_errors_benchmarking_and_trajectory_synthesis_for_robus.md)
- [\[CVPR 2026\] HATS: Hardness-Aware Trajectory Synthesis for GUI Agents](../../CVPR2026/llm_agent/hats_hardness-aware_trajectory_synthesis_for_gui_agents.md)

</div>

<!-- RELATED:END -->

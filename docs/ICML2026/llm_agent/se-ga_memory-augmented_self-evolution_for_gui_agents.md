---
title: >-
  [Paper Note] SE-GA: Memory-Augmented Self-Evolution for GUI Agents
description: >-
  [ICML 2026][LLM Agent][GUI Agent] SE-GA equips a VLM-based GUI agent with a "situational + semantic + experiential" three-layer memory library (TTME) and a two-stage memory-augmented self-evolution training pipeline (MAS…
tags:
  - "ICML 2026"
  - "LLM Agent"
  - "GUI Agent"
  - "Hierarchical Memory"
  - "Self-evolution"
  - "GRPO"
  - "Hindsight Goal-Shifting"
date: 2026-05-08
content_hash: 501fb2aa1abe2292
---

# SE-GA: Memory-Augmented Self-Evolution for GUI Agents

**Conference**: ICML 2026  
**arXiv**: [2605.16883](https://arxiv.org/abs/2605.16883)  
**Code**: https://github.com/jinshilong-dev/SE-GA (Available)  
**Area**: Agent / Multimodal VLM / Reinforcement Learning  
**Keywords**: GUI Agent, Hierarchical Memory, Self-evolution, GRPO, Hindsight Goal-Shifting

## TL;DR
SE-GA equips a VLM-based GUI agent with a "situational + semantic + experiential" three-layer memory library (TTME) and a two-stage memory-augmented self-evolution training pipeline (MASE, consisting of SFT and an improved GRPO). This approach pushes Qwen2.5-VL-7B to scores of 89.0 on ScreenSpot, 75.8 on AndroidControl-High, and 39.0 on AndroidWorld, consistently outperforming baselines of the same scale and even matching the performance of 72B models.

## Background & Motivation

**Background**: Current mainstream GUI agents treat VLMs (such as Qwen2.5-VL and UI-TARS) directly as policy networks that output actions based on screenshots. They typically undergo behavioral cloning via SFT on fixed trajectory datasets, with a few works utilizing RL algorithms like GRPO to align with human intentions.

**Limitations of Prior Work**: The authors identify two specific bottlenecks. First is the **limited context window and sole reliance on the current screenshot**: GUI navigation is a partially observable, history-dependent POMDP. Critical information may appear only in early steps, yet existing methods (e.g., ShowUI, OS-Atlas) only fit the most recent steps into the window, leading to irreversible failures in long-range tasks due to early forgetting. Second is the **static policy and lack of unified memory organization**: Real-world tasks are often variants or combinations of previously successful tasks. However, existing agents either remain frozen after training on fixed datasets or rely on temporary text-based RAG, failing to crystallize successful operational strategies into long-term reusable knowledge, and further failing to feed these explicit experiences back into model parameters.

**Key Challenge**: There is a structural trade-off between the **long-range dependency** inherent in GUI tasks and the engineering reality of **short windows and static parameters** in VLM agents. The agent must both access critical observations from 100 steps prior during inference and continuously evolve its strategy from past successful trajectories. Neither simply expanding the window nor adding more training rounds can solve this independently.

**Goal**: To transform the GUI agent from a "static command executor" into a "dynamic learner." This is decomposed into two sub-problems: (1) How to **precisely manage long-range context** during inference — not just episodic short-term windows, but also retrieving abstract rules and similar historical experiences across tasks; (2) How to **stably encode retrieved successful experiences back into policy parameters** during training, making GRPO effective in the sparse-reward, high-variance environment of GUI tasks.

**Key Insight**: Drawing inspiration from the three types of memory in human cognitive architecture (episodic, semantic, and experiential), the authors hypothesize that a GUI agent can similarly decouple "recent context," "general rules," and "similar successful experiences" through hierarchical memory. Furthermore, experiential memory can be dynamically accumulated during inference and then fed back into the training phase, forming a self-evolution closed loop.

**Core Idea**: A three-layer hierarchical memory library (episodic + semantic + experiential) is utilized for precise context retrieval at test-time, followed by a two-stage training pipeline (grounding SFT + improved GRPO) to inject high-quality trajectories from the memory back into the strategy parameters, enabling continuous online evolution of the GUI agent.

## Method

### Overall Architecture
SE-GA formalizes the problem as a POMDP $\langle\mathcal{S},\mathcal{A},\mathcal{O},\mathcal{T},\mathcal{R},\gamma\rangle$. At each step $t$, the agent receives a user instruction $Q$, the current screenshot $o_t$, and structured memory $M_{retrieved}$ retrieved from the hierarchical memory library $\mathcal{M}=(M^{EPI},M^{SEM},M^{EXP})$. These are assembled into an input $x_t=(o_t,Q,M_{retrieved})$, which passes through the policy $\pi_\theta(a_t|x_t)$ to output an action. The system consists of two components: **TTME** (Test-Time Memory Extension) handles hierarchical retrieval and real-time accumulation of successful trajectories during inference; **MASE** (Memory-Augmented Self-Evolution) handles the two-stage offline/online training to feed the collected data back into the VLM parameters. The base model is Qwen2.5-VL-7B, trained on 4×A800 GPUs.

### Key Designs

1.  **TTME Three-Layer Memory + Sliding Window + Multimodal Hybrid Retrieval**:
    *   **Function**: Provides the agent with a hierarchical, dynamically expandable context during inference, covering recent steps, general GUI rules, and historical success strategies.
    *   **Mechanism**: Each of the three layers has its own definition and retrieval mechanism. **Episodic** $M^{EPI}_t=[\langle o_k,a_k,o_{k+1}\rangle]_{k=1}^{t-1}$ stores raw action sequences, but uses a sliding window $\mathcal{C}^{epi}_t=[m_k]_{k=\max(1,t-H)}^{t-1}$ of fixed length $H$ to retain only the most recent $H$ steps, preventing misguidance from stale information. **Semantic** stores general interaction rules $m^{sem}_i=\langle k^{sem}_i,d_i\rangle$ (e.g., "login before accessing restricted pages"), retrieved via cosine similarity $S^{sem}(Q,m^{sem}_i)=\phi(Q)\cdot k^{sem}_i / (|\phi(Q)||k^{sem}_i|)$ for the Top-K. **Experiential** stores historical success trajectories and reflection summaries $m^{exp}_i=\langle\tau_i,g(\tau_i),k^{intent}_i,k^{task}_i\rangle$. Crucially, it uses **intent + visual hybrid retrieval**: $S^{exp}(Q,o_t)=\lambda\cdot\text{Sim}(\phi(Q),k^{intent}_i)+(1-\lambda)\cdot\text{Sim}(\psi(o_t),k^{task}_i)$, fusing text query similarity with the visual similarity of the current screenshot $\psi(o_t)$. All three contexts $\mathcal{C}^{epi},\mathcal{C}^{sem},\mathcal{C}^{exp}$ are concatenated into the agent's input.
    *   **Design Motivation**: Episodic memory addresses short-term context, semantic memory handles general knowledge transfer across tasks, and experiential memory prevents "reinventing the wheel." The introduction of visual similarity is critical — pure text retrieval is inaccurate in GUI scenarios where spatial and structural layout is highly significant. Additionally, TTME functions as a dynamic buffer where newly successful trajectories are added during inference.

2.  **MASE Two-Stage Training: Grounding SFT + Improved GRPO**:
    *   **Function**: Stably encodes the high-quality experiences collected by TTME into VLM parameters, addressing the instability of RL in sparse-reward, high-variance GUI settings.
    *   **Mechanism**: **Stage I (Grounding Training)** is memory-aware behavioral cloning. The objective $\mathcal{L}_{SFT}(\theta)=-\mathbb{E}_{(x,y)\sim\mathcal{D}_{ground}}[\frac{1}{|y|}\sum_t\log\pi_\theta(y_t|o_t,Q,M,y_{<t})]$ aims to solidify basic grounding capabilities and prevent catastrophic forgetting in Stage II. **Stage II (Self-Evolution Training)** introduces three improvements to GRPO: (a) **token-level importance ratio** (inspired by DAPO), $\rho_{i,t}=\pi_\theta(y_{i,t}|x,y_{i,<t})/\pi_{\theta_{old}}(\cdot)$, preventing sequence-level aggregation from causing gradient spikes due to irrelevant tokens; (b) **adaptive clipping**, where the upper bound $\epsilon_{cur}$ decays from $\epsilon_{init}$ to $\epsilon_{end}$ via a cosine schedule ($\epsilon_{cur}=\epsilon_{end}+\frac{1}{2}(\epsilon_{init}-\epsilon_{end})(1+\cos(\pi k/K))$), allowing large updates early in training and tightening later; (c) **hierarchical reward**, $R_{total}=w_f R_{format}+w_a R_{acc}$, where format is checked before calculating accuracy ($R_{format}=0$ nullifies accuracy reward). Accuracy rewards are further subdivided into action type and parameter rewards (e.g., $R_{point}=\mathbb{I}((x_p,y_p)\in B_{gt})$ for click tasks). The optimization objective is $\mathcal{J}(\theta)=\mathbb{E}[\frac{1}{\sum|y_i|}\sum_{i,t}(\min(\rho_{i,t}A_i,\rho_{i,t}^{clip}A_i)-\beta\mathbb{D}_{KL}(\pi_\theta||\pi_{ref}))]$.
    *   **Design Motivation**: Standard GRPO often suffers from gradient explosion or high variance on long GUI trajectories. The combination of token-level ratios, adaptive clipping, and format-first hierarchical rewards is tailored for GUI tasks where format must be strict and both action types and coordinate precision must be managed.

3.  **Hindsight Goal-Shifting Data Synthesis**:
    *   **Function**: Recycles failed trajectories as effective supervision signals to mitigate the scarcity of GUI agent training data.
    *   **Mechanism**: Given a failed trajectory $\tau=(s_0,a_0,\ldots,s_T)$ for an original goal $g$, if a prefix $\tau_{0:k}$ actually accomplished an **alternative sub-goal** $g'$ (e.g., "successfully opened the App but failed the subsequent search"), $\tau_{0:k}$ is relabeled as a successful sample for $g'$, forming $\mathcal{D}_{GS}=\{(\tau_{0:k},g')\mid \text{Verify}(\tau_{0:k},g')=1\}$. These are merged into the total dataset. Ultimately, 4K trajectories are split into 2K for grounding and 2K for self-evolution.
    *   **Design Motivation**: Adapting the HER (Hindsight Experience Replay) concept to the symbolic action space of GUIs allows the "successful prefix" of a zero-reward failed trajectory to generate gradients, effectively multiplying data utility.

### Loss & Training
Stage I SFT: lr=2e-6, global batch=16; Stage II GRPO: lr=2e-5, global batch=256, group size $G$=16; 4×A800 GPU. Data sourced from AITW + AMEX + GUIOdyssey + self-collected Android simulator data, filtered by Qwen-VL for simple/ambiguous samples and augmented by Hindsight Goal-Shifting.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours (7B) | Prev. SOTA | Gain |
| :--- | :--- | :--- | :--- | :--- |
| ScreenSpot | Avg Grounding Acc | **89.0** | UI-TARS-72B 88.4 / Aguvis-7B 84.4 | +0.6 vs 72B / +4.6 vs same scale |
| AndroidControl-Low | Success Rate | **88.6** | OS-Atlas-7B 85.2 | +3.4 |
| AndroidControl-High | Success Rate | **75.8** | UI-TARS-72B 74.7 / OS-Atlas-7B 71.2 | +1.1 vs 72B / +4.6 vs same scale |
| GUIOdyssey | Step SR | **83.9** | OS-Atlas-7B 62.0 / UI-TARS-72B 88.6 | +21.9 vs same scale (72B leads by 4.7) |
| GUIOdyssey | Type Acc | **96.5** | UI-TARS-72B 95.4 | +1.1 (surpasses 72B at same scale) |
| AndroidWorld (Online) | SR | **39.0** | UI-TARS-7B 33.0 / GUI-Critic-R1 27.6 | +6.0 |

### Ablation Study

| Configuration | AC-Low SR | AC-High SR | GUIOdyssey SR | Description |
| :--- | :--- | :--- | :--- | :--- |
| Full SE-GA | 88.6 | 73.8 | 83.9 | Full model |
| w/o TTME | 83.0 | 61.4 | 74.9 | Remove hierarchical memory: -5.6 on short-term, -12.4 on long-term |
| w/o MASE | 74.3 | 59.7 | 60.4 | Remove self-evolution training: performance collapses (GUIOdyssey -23.5) |

### Key Findings
*   **MASE is the foundation, TTME is the scaffold**: Removing MASE results in a much steeper drop than removing TTME (GUIOdyssey 83.9 → 60.4 vs. 74.9). This suggests that embedding grounding and decision-making capabilities into parameters via memory-augmented training is the performance cornerstone, while TTME provides long-range gains on top of a solid base.
*   **TTME's value scales with task length**: In AC-Low (short-term), removal causes a 5.6 drop; in AC-High (long-term), it causes a 12.4 drop, validating that hierarchical memory primarily addresses long-range context forgetting.
*   **7B vs. 72B**: SE-GA-7B outperforms UI-TARS-72B and Qwen2.5-VL-72B on ScreenSpot and AndroidControl-High. This indicates that in structured tasks like GUI interaction, improvements in data and training mechanisms are more valuable than simply scaling parameters.
*   **Online > Offline gains**: The 6-point lead on AndroidWorld (dynamic online) is more significant than the lead on offline benchmarks, demonstrating the advantage of the self-evolution mechanism in real-world environments.

## Highlights & Insights
*   **Engineering approach for three-layer memory + hybrid retrieval**: Directly mapping the episodic/semantic/experiential taxonomy from cognitive science to "recent window/general rules/historical success strategies" and introducing visual similarity $\psi(o_t)$ in experiential retrieval is essential for GUI tasks with strong layout dependencies.
*   **Self-evolution loop between TTME and MASE**: TTME collects new successful trajectories during inference → MASE fine-tunes them into parameters offline → the stronger agent collects higher-quality trajectories. This "non-parametric memory ↔ parameterized policy" cycle is transferable to any agent task (code, robotics, dialogue).
*   **Customized GRPO for GUI**: The combination of token-level importance ratios, adaptive clipping, and hierarchical rewards demonstrates meticulous engineering. These modifications directly address common failure modes in GUI tasks.
*   **Hindsight Goal-Shifting for symbolic action spaces**: Relabeling "prefix successes" is a nearly cost-free way to expand data for agents with clear intermediate states, applicable to web agents and tool-use agents.

## Limitations & Future Work
*   **Latency**: As experiential memory accumulates, hybrid retrieval based on embeddings and visual features may become a bottleneck for real-time response.
*   **Small training scale**: With only 4K trajectories, the scalability with large-scale data remains to be fully verified.
*   **Lack of cross-platform generalization**: Training data is concentrated on Android and ScreenSpot. Evaluation on Web or Desktop GUI is missing, and the need to rebuild semantic memory for other platforms is unverified.
*   **Failure mode analysis of TTME**: The impact of false-positive retrievals in experiential memory (similar intent but different interface state) has not been analyzed.
*   **Missing Hyperparameter Sensitivity**: Sensitivity curves for weight $\lambda$ and window length $H$ are not provided.

## Related Work & Insights
*   **vs. UI-TARS (Top-tier GUI agent)**: UI-TARS follows a "large model + large data" route (72B parameters). SE-GA-7B reaches comparable or superior performance through hierarchical memory and self-evolution, proving that mechanism optimization can be more effective than parameter scaling for GUI tasks.
*   **vs. OS-Genesis / GUI-Critic-R1**: These rely on SFT-only or critic-based RL and lack unified memory organization, resulting in significant performance gaps in long-range tasks. SE-GA elevates "memory" to a central abstraction for information organization and evolution.
*   **vs. ShowUI / RAG-based agents**: Most rely on text-only vector databases. SE-GA's hybrid retrieval is a necessary extension for spatial/structural GUI characteristics.
*   **vs. DAPO / Original GRPO**: The adoption of token-level ratios and customized rewards provides an engineering template for applying GRPO to structured agent tasks.
*   **vs. HER**: Hindsight Goal-Shifting successfully adapts HER from continuous control to symbolic GUI action spaces.

## Rating
*   Novelty: ⭐⭐⭐⭐ While three-layer memory architectures exist (e.g., CoALA), the application to GUI tasks coupled with visual hybrid retrieval and a training loop is highly coherent and well-executed.
*   Experimental Thoroughness: ⭐⭐⭐⭐ Covers four major benchmarks with ablation, though lacks sensitivity analysis for $\lambda$ and $H$, and cross-platform evaluation.
*   Writing Quality: ⭐⭐⭐⭐ Logical flow is clear, with well-defined formulas and a complete motivation-method-experiment chain.
*   Value: ⭐⭐⭐⭐ The 7B vs. 72B results have direct implications for deploying efficient GUI agents. The TTME+MASE loop is a generalizable paradigm for other agent tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] EchoTrail-GUI: Building Actionable Memory for GUI Agents via Critic-Guided Self-Exploration](../../CVPR2026/llm_agent/echotrail-gui_building_actionable_memory_for_gui_agents_via_critic-guided_self-e.md)
- [\[AAAI 2026\] Co-EPG: A Framework for Co-Evolution of Planning and Grounding in Autonomous GUI Agents](../../AAAI2026/llm_agent/co-epg_a_framework_for_co-evolution_of_planning_and_groundin.md)
- [\[ICML 2026\] SafeHarbor: Defining Precise Decision Boundaries via Hierarchical Memory-Augmented Guardrail for LLM Agent Safety](safeharbor_hierarchical_memory-augmented_guardrail_for_llm_agent_safety.md)
- [\[ICML 2026\] EvolveR: Self-Evolving LLM Agents through an Experience-Driven Lifecycle](evolver_self-evolving_llm_agents_through_an_experience-driven_lifecycle.md)
- [\[ICML 2026\] Recovering Policy-Induced Errors: Benchmarking and Trajectory Synthesis for Robust GUI Agents](recovering_policy-induced_errors_benchmarking_and_trajectory_synthesis_for_robus.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[CVPR 2026\] EchoTrail-GUI: Building Actionable Memory for GUI Agents via Critic-Guided Self-Exploration](../../CVPR2026/llm_agent/echotrail-gui_building_actionable_memory_for_gui_agents_via_critic-guided_self-e.md)
- [\[AAAI 2026\] Co-EPG: A Framework for Co-Evolution of Planning and Grounding in Autonomous GUI Agents](../../AAAI2026/llm_agent/co-epg_a_framework_for_co-evolution_of_planning_and_groundin.md)
- [\[ICML 2026\] EvolveR: Self-Evolving LLM Agents through an Experience-Driven Lifecycle](evolver_self-evolving_llm_agents_through_an_experience-driven_lifecycle.md)
- [\[ICML 2026\] SafeHarbor: Defining Precise Decision Boundaries via Hierarchical Memory-Augmented Guardrail for LLM Agent Safety](safeharbor_hierarchical_memory-augmented_guardrail_for_llm_agent_safety.md)
- [\[ACL 2026\] From Storage to Experience: A Survey on the Evolution of LLM Agent Memory Mechanisms](../../ACL2026/llm_agent/from_storage_to_experience_a_survey_on_the_evolution_of_llm_agent_memory_mechani.md)

</div>

<!-- RELATED:END -->

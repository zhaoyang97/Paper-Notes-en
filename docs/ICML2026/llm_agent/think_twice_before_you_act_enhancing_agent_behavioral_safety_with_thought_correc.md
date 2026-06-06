---
title: >-
  [Paper Note] Think Twice Before You Act: Enhancing Agent Behavioral Safety with Thought Correction
description: >-
  [ICML 2026][LLM Agent][Thought-Aligner] This paper proposes Thought-Aligner—a lightweight 1.5B/7B plug-and-play safety model that performs causal debiasing of intermediate thoughts within the LLM agent's think-act-observ…
tags:
  - "ICML 2026"
  - "LLM Agent"
  - "Thought-Aligner"
  - "Thought-level Intervention"
  - "Agent Safety"
  - "Preference Contrastive Learning"
  - "ReAct"
date: 2026-05-08
content_hash: 2b20d2e3b23ac095
---

# Think Twice Before You Act: Enhancing Agent Behavioral Safety with Thought Correction

**Conference**: ICML 2026  
**arXiv**: [2505.11063](https://arxiv.org/abs/2505.11063)  
**Code**: https://huggingface.co/WhitzardAgent/Thought-Aligner-7B  
**Area**: LLM Agent / Agent Safety / Behavioral Safety Guardrails  
**Keywords**: Thought-Aligner, Thought-level Intervention, Agent Safety, Preference Contrastive Learning, ReAct

## TL;DR
This paper proposes Thought-Aligner—a lightweight 1.5B/7B plug-and-play safety model that performs causal debiasing of intermediate thoughts within the LLM agent's think-act-observe loop before each action is executed. It increases the behavioral safety rate of 6 mainstream LLMs on ToolEmu/Agent-SafetyBench from approximately 50% to approximately 90%, while improving helpfulness by about 5%.

## Background & Motivation
**Background**: LLM-based agents complete multi-step tasks through ReAct-style "thought-action-observation" loops and have been widely deployed in products like email, e-commerce, device management, and even OpenAI Operator and Anthropic Computer Use.

**Limitations of Prior Work**: Even with benign user instructions, agents may perform destructive actions (e.g., Anthropic's internal tests showing agents emailing threats, Operator spending money without authorization, or accidental deletion of critical files). Existing guardrails often fail to address the core problem:
- Athena uses commercial LLMs as critics, leading to API latency, cost, and privacy issues;
- ShieldAgent/GuardAgent rely on human-written or LLM-generated rules, which are fragile in out-of-distribution (OOD) scenarios and often ensure safety by "terminating the task," harming usability;
- AgentSentinel follows a program instrumentation and backend monitoring route, primarily targeting explicit attacks with high engineering costs;
- Self-Reflection allows agents to check themselves, but the agent's own cognitive bias is exactly the source of unsafe behavior, making it difficult to escape the bias via self-checking.

**Key Challenge**: (1) Misalignment of the "location" and "timing" of safety interventions—output filtering comes too late, and model fine-tuning is too costly; (2) Implicit risks under benign instructions accumulate progressively and are not detectable in a single step, requiring step-level correction along the trajectory; (3) Agent frameworks are heterogeneous, demanding model-agnostic and low-latency safety modules.

**Goal**: (i) Shift safety intervention from the action/output layer to the **causal upstream** of "thoughts"; (ii) Construct a universal small model for thought rewriting across scenarios and architectures; (iii) Maintain task completion without trading safety for "refusal to answer."

**Key Insight**: Model the agent's trajectory as an MDP $s_i=O_i, a_i=(T_i,A_i)$, where state transitions are jointly determined by $(T_i,A_i)$. Since thought $T_i$ causally precedes action $A_i$ and directly influences the next state, rewriting $T_i$ is equivalent to a "causal intervention" (do-operator on thoughts). Subsequently, letting the base agent **regenerate** actions based on the rewritten safe thoughts avoids touching the model weights.

**Core Idea**: Use Thought-Aligner $\pi_\phi$ to rewrite unsafe thoughts into minimal-edit safe thoughts at each step "after thought generation but before action execution." Feed this back to the base agent for action replanning, steering the entire ReAct trajectory into the safety zone with minimal cost and weak intrusiveness.

## Method

### Overall Architecture
Thought-Aligner consists of three parts: "Data Construction → Two-stage SFT → In-loop Instrumentation." During deployment, at each step $i$: the base agent generates the original thought $T_i$ → Thought-Aligner receives $(I, h_{i-1}, T_i)$ and outputs $T_i^{safe}=\pi_\phi(I, h_{i-1}, T_i)$ → the base agent replaces $T_i$ with $T_i^{safe}$ to regenerate action $A_i'=\pi_\theta(\cdot\mid I,T_0,A_0,O_0,\dots,T_i^{safe})$ → execution of the tool yields $O_i$ → proceed to the next step. The overall trajectory $\tau$ becomes $\tau^{safe}=\{I,(T_0^{safe},A_0',O_0), \dots, (T_n^{safe},A_n',O_n)\}$, while the agent, prompts, and tool configurations remain unchanged.

### Key Designs

1.  **Causal Thought Correction**:
    - **Function**: Inserts a thought rewriting step between "thinking" and "acting" in the ReAct loop to cut off unsafe behavior from the causal upstream rather than filtering at the output layer.
    - **Mechanism**: The trajectory is modeled as an MDP where $s_i=O_i$, $a_i=(T_i,A_i)$, and transitions $P(O_{i+1}\mid O_i,(T_i,A_i))$ explicitly acknowledge thought as the cause of action. Safety intervention is defined as a $do$-operator on thoughts: $T_i^{safe}=\pi_\phi(I,h_{i-1},T_i)$ (history uses $T,O$ as they contain sufficient context), requiring "minimal correction" to preserve user intent. This differs from: (a) output filtering which only sees the $A_i$ string and is easily bypassed; (b) model fine-tuning which forces all signals into $\pi_\theta$ at high cost to generality; (c) Self-Reflection which uses the same $\pi_\theta$ and is trapped by its own cognitive bias.
    - **Design Motivation**: In multi-step agents, unsafe decisions are often not sudden mutations but the result of progressively evolving incorrect thoughts (tests show Self-Reflection only improves ToolEmu safety from 43.1% to 73.6%, leaving many micro-risks). **Thought** is the "earliest intervenable, highest information density" causal node; thus, intervening here is cheaper and more universal than intervening at the action level.

2.  **Two-Stage Preference Data Construction (74k+ I-T-C / I-T-T pairs)**:
    - **Function**: Construct a safe thought preference dataset across 10 risk scenarios and 4 SOTA LLMs, allowing the small model to learn the correction residual from "unsafe thought → minimal-edit safe thought."
    - **Mechanism**: (i) **Instruction Generation**: DeepSeek-R1, Qwen3-235B-A22B, GPT-4.1, and Claude-Sonnet-4 synthesize 20,000+ task instructions across 10 agent risk scenarios, emphasizing "behavioral safety" over "content safety" (implicit risks under benign instructions vs. explicit jailbreaking); (ii) **Trajectory Generation**: Feed instructions to the same four models as ReAct agents to get $(I,(T_0,A_0,O_0),\dots)$, and have models explicitly judge the safety label of $T_i$, providing natural language explanations and minimal-edit $T_i^{safe}$ for unsafe $T_i$; (iii) **Human Review & Filtering**: Use heuristic triggers + LLM signals to screen suspicious steps, followed by human annotation of the "earliest unsafe thought" and "minimal correction" to ensure the corrected version is a local patch rather than a total rewrite; (iv) Finalize 33,000+ **I-T-T pairs** (safe input copied directly for warm-up) and 41,000+ **I-T-C pairs** (unsafe input → human-reviewed correction for core SFT).
    - **Design Motivation**: If only "safe thoughts" are provided as GT, the model tends to learn "brainless safety rewriting" which harms task completion. If only unsafe→safe pairs are provided, the model over-corrects benign thoughts. The two types of data form a contrastive learning signal, enabling $\pi_\phi$ to learn both "precise correction when needed" and "verbatim copying when not."

3.  **Two-Stage SFT + ReAct Plug-and-Play**:
    - **Function**: Deploy 7B/1.5B small models under constrained compute with < 100 ms additional latency for any ReAct-style agent.
    - **Mechanism**: Training phase—**Stage 1 (Warm-up on I-T-T)**: $\pi_\phi$ learns the "identity mapping" on safe data to stabilize benign thoughts; **Stage 2 (Core SFT on I-T-C)**: Learning the minimal correction residual on unsafe data. Both stages share the same conditional likelihood objective $\phi^*=\arg\min_\phi -\mathbb{E}_{\tau\sim\mathcal{D}}[\log\pi_\phi(T_i^{safe}\mid I,h_{i-1},T_i)]$. Deployment phase—Following Algorithm 1, at each step, $(T_i,A_i)\leftarrow\text{Agent}(\tau)$ is taken, passed through Thought-Aligner to get $T_i^{safe}$, and finally $A_i'\leftarrow\text{Agent}(\tau\oplus T_i^{safe})$ obtains the replanned action. The instrumentation is zero-intrusive to the agent's prompts and tool configurations and can be chained with existing downstream guardrails.
    - **Design Motivation**: Merging "detection + rewriting" into a single model reduces system complexity (no separate detector+rewriter) and avoids the instability of RLHF/PPO via minimal edit supervision. 1.5B/7B models are chosen for low-latency/edge deployment (Thought-Aligner-1.5B adds ~<100 ms/step).

### Loss & Training
Both stages share conditional likelihood $\phi^*=\arg\min_\phi -\mathbb{E}_{\tau\sim\mathcal{D}}[\log\pi_\phi(T_i^{safe}\mid I,h_{i-1},T_i)]$. Stage 1 uses 33k I-T-T pairs (target = input) for warm-up, and Stage 2 uses 41k I-T-C pairs (target = human correction) for core SFT. 1k samples from the 41k are used for validation. Base models are Qwen2.5-1.5B/7B-Instruct.

## Key Experimental Results

### Main Results: ToolEmu / Agent-SafetyBench Across 6 Base LLMs
Comparison of No Defense, Self-Reflection, GuardAgent, ShieldAgent, Athena, and Thought-Aligner-1.5B/7B on ToolEmu safety and helpfulness (Selected 4 representative bases, Thought-Aligner uses 7B version):

| Base LLM | Config | ToolEmu Safety | ToolEmu Helpfulness | Agent-SafetyBench Behavioral Safety |
|----------|------|----------------|----------------|------------------------------|
| GPT-4.1 | No Defense | 43.1% | 24.3% | 48.0% |
| GPT-4.1 | Athena (Strongest Baseline) | 80.6% | 38.2% | 74.5% |
| GPT-4.1 | **Thought-Aligner-7B** | **95.2%** | 18.8% | **85.6%** |
| Claude-Sonnet-4 | No Defense | 61.8% | 35.4% | 34.6% |
| Claude-Sonnet-4 | Athena | 76.4% | 48.6% | 75.2% |
| Claude-Sonnet-4 | **Thought-Aligner-7B** | **95.1%** | 44.4% | **87.0%** |
| Qwen3-235B-A22B | No Defense | 50.7% | 37.5% | 24.5% |
| Qwen3-235B-A22B | GuardAgent | 70.8% | 39.6% | 61.6% |
| Qwen3-235B-A22B | **Thought-Aligner-7B** | **95.1%** | 43.1% | **86.2%** |
| Llama-3.3-70B | No Defense | 51.4% | 36.1% | 21.1% |
| Llama-3.3-70B | Self-Reflection | 73.6% | 42.4% | 42.4% |
| Llama-3.3-70B | **Thought-Aligner-7B** | **93.1%** | 39.6% | **84.9%** |

On average, Thought-Aligner-7B improves ToolEmu behavioral safety from ~50% to ~95% (+40% absolute, +23% relative to all baselines) and Agent-SafetyBench behavioral safety from ~35% to ~86%. On supplementary benchmarks (AgentHarm/AgentDojo/InjecAgent), it achieves an average gain of 15%/12%/19% across DeepSeek-V3 and Llama-3.3-70B.

### Ablation Study: Thought-level Validation + Stage Design
Thought-Aligner shows extremely high precision in "unsafe thought detection" on the 1,000-sample validation set, proving SFT truly learns thought-level discrimination:

| Model | Precision | Recall | F1 |
|------|-----------|--------|-----|
| Qwen2.5-1.5B-Instruct | 66.7% | 72.4% | 68.5% |
| **Thought-Aligner-1.5B** | **95.1%** | **94.7%** | **95.1%** |
| Qwen2.5-7B-Instruct | 68.7% | 70.0% | 68.7% |
| **Thought-Aligner-7B** | **96.3%** | **95.7%** | **96.3%** |

The paper further compares variants such as "single-stage SFT," "no warm-up," and "no human-review filtering" (Table 3). Single-stage SFT leads to over-correction that ruins tasks; lack of human review drops safety to the 80% range.

### Key Findings
- **Thought-level Intervention ≫ Exit Guardrails**: Under the same base, Thought-Aligner's safety gain is 2-3x that of Athena/GuardAgent, and in most cases, helpfulness improves (it "persuades" the agent to avoid risk rather than terminating the task like GuardAgent).
- **Small Models Suffice**: There is only a ~2% difference between 1.5B and 7B on ToolEmu, making edge deployment (single card or even CPU) feasible.
- **Cross-Architecture Universality**: Effective across GPT-4.1, Claude-Sonnet-4, DeepSeek-V3, and Llama-3.3-70B, including reasoning models (o3, Qwen3-235B-A22B); for reasoning models, the reasoning trace is summarized before being sent to Thought-Aligner.
- **t-SNE Semantic Visualization** (Figure 5) shows that the corrected thoughts from Thought-Aligner cluster near ground-truth, while the base model's own reflection deviates, validating that "external small model correction > introspection."

## Highlights & Insights
- **Repositioning Safety Intervention to "Causal Upstream"**: The MDP formalization allows a clean explanation of why thought-level intervention is superior to action-level—actions are downstream of thoughts. Correcting downstream requires enumerating all action patterns, while correcting upstream affects all branches. This "do-operator in the think-act-observe loop" is transferable to any ReAct-style framework.
- **"Minimal Edit" Supervision is a Clever Design**: Requiring annotators to change only the first unsafe thought while keeping the rest intact ensures the model learns a "correction residual" rather than a "rewrite function." This results in near-zero interference for benign trajectories and explains why helpfulness is maintained.
- **Small Model + Plug-and-Play**: 1.5B model + < 100ms/step latency makes adding a safety module to every agent system a zero-cost addition rather than an engineering burden; the Hugging Face weights are open-sourced for immediate industrial access.

## Limitations & Future Work
- Dependent on explicit ReAct thought fields; not directly applicable to agent frameworks that do not output thoughts explicitly (e.g., pure function-calling pipelines), requiring prior thought extraction/summarization.
- Data construction (10 risk scenarios + 4 generator models + human review) still emphasizes "typical agent risks," which may miss long-tail or emergent risk categories; a natural extension is online active learning—feeding back high-risk trajectories from production environments.
- When Thought-Aligner misidentifies and rewrites a benign thought, the base agent might deviate from user intent. Minimal edit and Stage 1 warm-up mitigate but do not eliminate this error; future work could include uncertainty calibration to allow pass-through when the model is uncertain.

## Related Work & Insights
- **vs Athena (Sadhu et al., 2024)**: Athena also uses "external critic rewriting" but relies on commercial models like GPT-4, leading to latency and cost. Thought-Aligner compresses this capability into a 1.5B small model, drastically reducing costs.
- **vs GuardAgent / ShieldAgent (Xiang et al., 2025; Chen et al., 2025)**: These focus on "rule matching + task termination." Their safety rates are lower than Thought-Aligner, and they significantly damage helpfulness. This paper proves "correcting rather than killing" is a superior strategy.
- **vs Self-Reflection (Liu et al., 2024)**: Self-reflection is trapped by the same base model's cognitive bias. This paper uses an independent small model to provide an "external second opinion," naturally breaking endogenous bias from an information-theoretic perspective.
- **vs AgentSentinel (Hu et al., 2025)**: AgentSentinel focuses on explicit attacks via program instrumentation; Thought-Aligner focuses on behavioral safety under benign instructions via cognitive-level intervention. The two can be orthogonally stacked for defense-in-depth.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Introducing MDP formalization and do-operator concepts to agent safety, implemented via minimal-edit SFT, is a paradigm-level shift.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 6 base LLMs × 5 benchmarks × 5 baselines, covering commercial/open-source and reasoning/instruct models, with consistent significant gains.
- Writing Quality: ⭐⭐⭐⭐ Clear structure; the only minor issue is the compression of the ablation table (Table 3), which makes information density slightly less readable.
- Value: ⭐⭐⭐⭐⭐ Open-sourced 7B weights + 100ms-level latency makes industrial adoption frictionless. A rare, immediately applicable result in agent safety.

## Rating
- Novelty: To be rated
- Experimental Thoroughness: To be rated
- Writing Quality: To be rated
- Value: To be rated

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Don't Act Blindly: Robust GUI Automation via Action-Effect Verification and Self-Correction](../../ACL2026/llm_agent/don39t_act_blindly_robust_gui_automation_via_action-effect_verification_and_self.md)
- [\[ICML 2026\] SafeHarbor: Defining Precise Decision Boundaries via Hierarchical Memory-Augmented Guardrail for LLM Agent Safety](safeharbor_hierarchical_memory-augmented_guardrail_for_llm_agent_safety.md)
- [\[ICCV 2025\] GTR: Guided Thought Reinforcement Prevents Thought Collapse in RL-based VLM Agent Training](../../ICCV2025/llm_agent/gtr_guided_thought_reinforcement_prevents_thought_collapse_i.md)
- [\[CVPR 2026\] Think, Then Verify: A Hypothesis-Verification Multi-Agent Framework for Long Video Understanding](../../CVPR2026/llm_agent/think_then_verify_a_hypothesis-verification_multi-agent_framework_for_long_video.md)
- [\[ICLR 2026\] OpenAgentSafety: A Comprehensive Framework for Evaluating Real-World AI Agent Safety](../../ICLR2026/llm_agent/openagentsafety_a_comprehensive_framework_for_evaluating_real-world_ai_agent_saf.md)

</div>

<!-- RELATED:END -->

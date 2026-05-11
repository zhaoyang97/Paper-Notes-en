---
title: >-
  [Paper Note] Towards Scalable Lightweight GUI Agents via Multi-role Orchestration
description: >-
  [ACL 2026][LLM Agent][GUI Agent] This paper proposes LAMO, a framework that trains a lightweight 3B MLLM into a flexibly orchestrated multi-role GUI Agent through role-oriented data synthesis and two-stage training (SFT…
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "GUI Agent"
  - "Lightweight Model"
  - "Multi-role Orchestration"
  - "Policy Executor"
  - "Reinforcement Learning"
date: 2026-05-08
content_hash: 801956d14b57ffe3
---

# Towards Scalable Lightweight GUI Agents via Multi-role Orchestration

**Conference**: ACL 2026
**arXiv**: [2604.13488](https://arxiv.org/abs/2604.13488)
**Code**: [GitHub](https://github.com/BigTaige/LAMO)
**Area**: LLM Agent / GUI Automation
**Keywords**: GUI Agent, Lightweight Model, Multi-role Orchestration, Policy Executor, Reinforcement Learning

## TL;DR

This paper proposes LAMO, a framework that trains a lightweight 3B MLLM into a flexibly orchestrated multi-role GUI Agent through role-oriented data synthesis and two-stage training (SFT with Perplexity-Weighted Cross-Entropy + multi-task RL). The agent operates in three modes—monolithic inference, multi-agent collaboration, and plug-and-play policy executor—and achieves a 77.6% success rate on AndroidWorld when paired with a GPT-5 planner, surpassing dedicated GUI agents with 72B parameters.

## Background & Motivation

**Background**: MLLM-based GUI Agents are evolving from static environments toward complex online real-world scenarios. State-of-the-art methods (e.g., UI-TARS-72B, Agent-S2) have achieved significant gains by scaling model size and training data, but at prohibitively high deployment costs. Lightweight GUI Agents (≤7B) perform reasonably well on static benchmarks but suffer dramatic performance degradation in online real-world environments.

**Limitations of Prior Work**: (1) Lightweight MLLMs are limited by parameter scale and underperform in end-to-end long-horizon tasks that simultaneously demand screen analysis, strategic decision-making, and tool invocation. (2) End-to-end monolithic episodic learning couples high-level reasoning with low-level execution within a fixed pipeline, resulting in poor task scalability and difficulty adapting to multi-agent systems (MAS). (3) Training multiple skill experts is costly—for example, Agent-S2 requires concurrently deploying UI-TARS-72B (visual grounding), Tesseract OCR (text grounding), and UNO (structural grounding), incurring extremely high system costs. (4) Lightweight agents lack task scalability and cannot flexibly switch roles via context engineering.

**Key Challenge**: A cost-scalability dilemma—large models offer task scalability but at high deployment cost, while lightweight models are cheap to deploy but limited in capability and not scalable.

**Goal**: To achieve task scalability on lightweight MLLMs by enabling a 3B model to work flexibly across different inference modes through parameter sharing and multi-role orchestration, while continuously benefiting as a plug-and-play policy executor paired with advanced planners.

**Key Insight**: Decompose GUI automation into five core capabilities—Action-Tool Alignment (ATA), Logically Consistent CoT (LCC), Screen Understanding (SU), Goal Planning (GP), and Screen Grounding (SG)—and enable a single 3B model to assume multiple roles through role-oriented data synthesis and parameter sharing.

**Core Idea**: Replace multiple specialized models with parameter-shared multi-role orchestration—a single lightweight model switches among four roles (Observer, Planner, Allocator, Executor) via context engineering, achieving MAS-level performance.

## Method

### Overall Architecture

The LAMO framework consists of three core stages: (1) Role-oriented data synthesis—teacher models (Qwen-2.5-VL-72B and Gemini-2.5-Pro) generate training data for five categories of GUI skills; (2) SFT stage—knowledge distillation and visual perception enhancement using PWCE loss; (3) RL stage—multi-task GRPO collaborative exploration. After training, LAMO-3B supports three inference modes: end-to-end monolithic inference, parameter-shared multi-agent systems, and plug-and-play policy executor paired with an advanced planner.

### Key Designs

1. **Role-oriented Data Synthesis**:

    - Function: Generate high-quality training data for five core GUI capability categories.
    - Mechanism: GUI automation is decomposed into five task types—ATA (Action-Tool Alignment), LCC (Logically Consistent CoT), SU (Screen Understanding), GP (Goal Planning), and SG (Screen Grounding). ATA and SG data are synthesized with Qwen-2.5-VL-72B; SU/LCC/GP data are synthesized with Gemini-2.5-Pro. For SG tasks, two practical challenges receive special treatment: (a) semantically sparse elements—brief original descriptions are expanded into semantically rich captions by the teacher model, training the model to jointly predict rich descriptions and coordinates; (b) complex layout interference—foreground targets are overlaid onto background screens with added distractors via rule-based augmentation, generating Intricate-Layout Grounding (ILG) data.
    - Design Motivation: Lightweight models perform poorly on long-horizon tasks end-to-end, yet are reliable when handling individual sub-capabilities independently. Task decomposition with parameter sharing thus enables a single model to possess all skills.

2. **Perplexity-Weighted Cross-Entropy (PWCE) Loss**:

    - Function: Enhance the model's perceptual precision over screen details, particularly coordinate values.
    - Mechanism: In standard cross-entropy loss, coordinate tokens tend to exhibit high perplexity yet receive equal weight. PWCE dynamically adjusts the loss weight for each token according to its perplexity: $w_i = \frac{1 + \alpha \frac{PPL_i}{\overline{PPL} + \epsilon}}{\frac{1}{|M|}\sum_{j \in M}(1 + \alpha \frac{PPL_j}{\overline{PPL} + \epsilon})}$, followed by weighted cross-entropy $\mathcal{L}_{PW} = \frac{1}{|M|}\sum_{i \in M} w_i \cdot CE(h_i^*, \tilde{y}_i)$. The final loss is $\mathcal{L}_{PWCE} = \mathcal{L}_{CE} + \lambda \mathcal{L}_{PW}$.
    - Design Motivation: Although SFT improves textual learning, predicted coordinates exhibit systematic bias. PWCE addresses this numerical perception deficiency by assigning greater weight to high-perplexity coordinate tokens.

3. **Multi-role Orchestration Inference**:

    - Function: Instantiate multiple skill-specialized roles from a single parameter-shared model instance, supporting diverse inference modes.
    - Mechanism: LAMO-3B switches among four roles via context engineering—Observer (providing semantic screen description $\mathcal{C}_{s2w}$), Planner (decomposing goals into sub-tasks $\mathcal{C}_{plan}$ and hints $\mathcal{C}_{tips}$), Allocator (assigning the current action $\mathcal{C}_{action}$ based on history and context), and Executor (translating action instructions into atomic operations $a_t$). In policy executor mode, an advanced MLLM (e.g., GPT-5) acts as the planner generating high-level instructions $\mathcal{C}_{action}^*$, while LAMO-3B serves as the executor converting them into precise screen operations.
    - Design Motivation: MAS decomposition reduces the complexity faced by each role, mitigating the "lost-in-the-middle" problem and thought-action hallucinations. The policy executor mode allows the lightweight model to continuously benefit as planner capabilities improve.

### Loss & Training

SFT stage: 1 epoch, learning rate 4e-6, warmup ratio 0.03, global batch size 256, LoRA (rank 128, alpha 256). RL stage: visual backbone frozen; only the merge layer and LLM are trained; GRPO for 1 epoch, learning rate 1e-6, rollout batch 32, 8 rollouts per sample. Multi-task RL rewards: TF-IDF similarity normalization for SU/GP; coordinate distance for SG; string matching on tool category and value for ATA; with a length penalty $r_{penalty} = -\varphi \cdot \frac{length(y_{pred})}{L_{max}}$.

## Key Experimental Results

### Main Results

**MiniWob++ Online Environment Success Rate**

| Method | Success Rate |
|--------|-------------|
| Qwen2.5-VL-3B | 34.6 |
| UI-TARS-7B | 58.7 |
| Gemini-2.5-pro (monolithic) | 71.0 |
| LAMO-3B (end-to-end) | 50.0 |
| LAMO-3B (MAS) | 60.9 (+21.8%) |
| LAMO-3B (Gemini-2.5-pro planning) | **77.2** (+54.4%) |

**AndroidWorld Success Rate**

| Method | Success Rate |
|--------|-------------|
| UI-TARS-72B | 46.6 |
| Agent-S2 | 54.3 |
| Mobile-Agent-V3 | 73.3 |
| LAMO-3B (Gemini-2.5-pro planning) | 60.3 |
| LAMO-3B (GPT-5 planning) | **77.6** |

### Ablation Study

**Key Component Ablation (Performance Degradation Relative to LAMO-3B)**

| Ablation | SP | SP-v2 | SP-pro | MiniWob++ |
|----------|----|-------|--------|-----------|
| Remove ILG data | -2.1% | -3.8% | -34.7% | -2.7% |
| SFT only (no RL) | -1.1% | -3.0% | -32.7% | -22.5% |
| Remove PWCE | -1.7% | -3.5% | -38.3% | -26.9% |
| Qwen2.5-VL-3B (no training) | -7.7% | -6.3% | -51.0% | -44.5% |

### Key Findings

- MAS mode improves over end-to-end inference by 21.8% on MiniWob++; the policy executor mode yields a further 54.4% gain.
- LAMO-3B + GPT-5 planner achieves 77.6% on AndroidWorld, surpassing Mobile-Agent-V3 (73.3%) and UI-Venus-Navi-72B (65.9%).
- On ScreenSpot-pro, LAMO-3B (36.1%) outperforms UI-TARS-7B (35.7%) and several 72B models.
- PWCE contributes most significantly in complex layout scenarios: its removal causes a 38.3% drop on SP-pro.
- The RL stage is critical for online environments: SFT-only results in a 22.5% drop on MiniWob++.
- On OSWorld, LAMO-3B (38.5%) surpasses UI-TARS-1.5-7B (28.2%) and trails Qwen2.5-VL-32B (43.6%) by only 5.1 points despite having 10× fewer parameters.

## Highlights & Insights

- The policy executor mode represents a highly forward-looking design—the lightweight model need not perform planning itself but serves as a reliable "hand," with the overall performance ceiling rising continuously as planners (e.g., GPT-5) improve.
- The PWCE loss provides an elegant solution to the coordinate prediction problem in GUI agents: perplexity-based weighting directs the model's attention toward uncertain numerical tokens.
- Parameter-shared multi-role orchestration achieves the advantages of MAS without increasing model parameters, representing an efficient approach to capability scaling.
- InfiGUI-R1-3B is competitive in static environments but collapses in online settings (38.5 vs. 10.3 on OSWorld), highlighting the task scalability deficiencies of end-to-end episodic learning.

## Limitations & Future Work

- Due to the 3B parameter constraint, reasoning depth remains insufficient for long-horizon tasks requiring more than 10 steps, still necessitating a large-model planner.
- Performance on desktop environments—particularly spreadsheet tasks and scenarios requiring software-domain priors—lags behind mobile-environment performance.
- The synthesis quality and diversity of ILG data augmentation leave room for improvement.
- The effect of combining LAMO with a broader range of planner types (e.g., open-source vs. closed-source planners) remains unexplored.

## Related Work & Insights

- **vs. UI-TARS**: UI-TARS-72B has 24× more parameters than LAMO-3B yet achieves only 46.6% on AndroidWorld, while LAMO-3B + GPT-5 reaches 77.6%—demonstrating that a "heavy executor" is inferior to a "lightweight executor + strong planner."
- **vs. GUI-R1 / InfiGUI-R1**: These methods are trained with end-to-end episodic RL, performing well in static environments but collapsing online. LAMO achieves superior task scalability through role decomposition.
- **vs. Agent-S2**: Agent-S2 employs multiple large specialized executors (UI-TARS-72B + Tesseract + UNO) at extremely high system cost; LAMO-3B accomplishes all execution functions with a single 3B model.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — PWCE loss, role-oriented data synthesis, and parameter-shared multi-role orchestration are each independently original; the policy executor mode has strong practical foresight.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Covers five benchmarks spanning static (ScreenSpot-pro, AndroidControl) and online (MiniWob++, AndroidWorld, OSWorld) settings, with detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ — Problem formulation is clear and the three inference modes are well-structured hierarchically, though the notation system is somewhat complex.
- Value: ⭐⭐⭐⭐⭐ — Establishes a viable "executor + planner" paradigm for lightweight GUI agents; the 77.6% AndroidWorld success rate represents a genuinely top-tier result.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Lightweight LLM Agent Memory with Small Language Models](lightweight_llm_agent_memory_with_small_language_models.md)
- [\[ACL 2026\] SILO-BENCH: A Scalable Environment for Evaluating Distributed Coordination in Multi-Agent LLM Systems](silo-bench_a_scalable_environment_for_evaluating_distributed_coordination_in_mul.md)
- [\[AAAI 2026\] History-Aware Reasoning for GUI Agents](../../AAAI2026/llm_agent/history-aware_reasoning_for_gui_agents.md)
- [\[ACL 2026\] RISK: A Framework for GUI Agents in E-commerce Risk Management](risk_a_framework_for_gui_agents_in_e-commerce_risk_management.md)
- [\[ACL 2026\] LPO: Towards Accurate GUI Agent Interaction via Location Preference Optimization](lpo_towards_accurate_gui_agent_interaction_via_location_preference_optimization.md)

</div>

<!-- RELATED:END -->

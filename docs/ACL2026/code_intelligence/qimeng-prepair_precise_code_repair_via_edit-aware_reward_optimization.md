---
title: >-
  [Paper Note] QiMeng-PRepair: Precise Code Repair via Edit-Aware Reward Optimization
description: >-
  [ACL 2026][Code Intelligence][Precise Code Repair] This paper identifies the "over-editing" problem in LLM-based code repair—where models tend to rewrite substantial portions of code instead of precisely locating and fix…
tags:
  - "ACL 2026"
  - "Code Intelligence"
  - "Precise Code Repair"
  - "Over-editing"
  - "Edit-Aware Reward"
  - "GRPO"
  - "Speculative Editing"
date: 2026-05-08
content_hash: d8b3cffa5d6c1950
---

# QiMeng-PRepair: Precise Code Repair via Edit-Aware Reward Optimization

**Conference**: ACL 2026  
**arXiv**: [2604.05963](https://arxiv.org/abs/2604.05963)  
**Code**: [GitHub](https://github.com/...)  
**Area**: Code Intelligence / Program Repair  
**Keywords**: Precise Code Repair, Over-editing, Edit-Aware Reward, GRPO, Speculative Editing

## TL;DR

This paper identifies the "over-editing" problem in LLM-based code repair—where models tend to rewrite substantial portions of code instead of precisely locating and fixing bugs. It proposes the PRepair framework, which employs Self-Breaking (diverse bug injection) and Self-Repairing (edit-aware GRPO training) to significantly enhance repair precision while maintaining correctness, further accelerating speculative decoding inference.

## Background & Motivation

**Background**: LLMs demonstrate excellent performance in program repair. Current training methods (SFT and RL) typically optimize only for repair correctness, treating code repair as a pure correctness objective.

**Limitations of Prior Work**: (1) During GRPO training, as correctness improves, the edit cost also increases—models do not learn precise repair but rather "guess" the correct solution through massive modifications; (2) Over-editing destroys the original code structure and increases the burden of code review for developers; (3) Over-editing fails to accurately locate bugs, limiting the actual effectiveness and maintainability of repairs.

**Key Challenge**: There is a tension between repair correctness and edit minimality—optimizing only for correctness allows the model to take the "rewriting" shortcut instead of learning to understand and precisely locate bugs.

**Goal**: To design a Precise Repair framework that maximizes the reuse of original code while maintaining repair correctness.

**Key Insight**: It is observed that edit cost grows synchronously with correctness during GRPO training (Figure 2), suggesting the need to explicitly introduce edit constraints into the reward function.

**Core Idea**: Edit-Aware GRPO (EA-GRPO)—where edit penalties are applied to correct samples only when the group-level accuracy exceeds a certain threshold, balancing correctness and edit minimality.

## Method

### Overall Architecture

PRepair consists of two stages: (1) Self-Breaking—the model injects diverse bugs into correct code, utilizing a min-max sampling strategy to maximize bug diversity; (2) Self-Repairing—training the model on the generated bug-prone code using EA-GRPO, where an edit-aware reward dynamically balances correctness and edit cost. Evaluation is performed using the newly proposed $\text{fix}_p@k$ metric.

### Key Designs

1.  **$\text{fix}_p@k$ Precise Repair Metric**:
    - **Function**: Jointly evaluates repair correctness and the degree of editing.
    - **Mechanism**: Adds an edit constraint to pass@k—a repair is successful only if the generated code passes all tests and the edit cost does not exceed $p$ times the theoretical minimum edit. Edit cost $\mathbf{D}_{\text{EC}}(X,Y) = \mathbf{D}(X,Y)/|X|$ is normalized using line-level Levenshtein distance.
    - **Design Motivation**: pass@k only considers correctness and fails to reflect repair quality—rewriting the entire code might pass tests but does not constitute a good repair.

2.  **Self-Breaking (Diverse Bug Injection)**:
    - **Function**: Generates large-scale precise repair training data without manual annotation.
    - **Mechanism**: The model is provided with correct code and a description, then prompted to inject bugs. From $m$ candidates, $k$ most diverse ones are selected via min-max sampling: $\mathcal{X}_s = \min_{\mathcal{X}' \subset \mathcal{X}, |\mathcal{X}'|=k} \max_{X_i,X_j \in \mathcal{X}', i \neq j} (1 - \mathbf{D}_{\text{EC}}(X_i, X_j))$.
    - **Design Motivation**: Precise repair requires training data with substantial correct logic and only local errors, which is extremely scarce in reality. Min-max sampling prevents the over-concentration of bug patterns.

3.  **EA-GRPO (Edit-Aware Group Relative Policy Optimization)**:
    - **Function**: Encourages minimal and correct repairs during RL training.
    - **Mechanism**: Accuracy $\text{Acc}_{\mathcal{G}^t}$ is calculated for each rollout group, and the edit penalty is activated only when it exceeds threshold $\alpha$. For correct samples in the group, a normalized edit penalty is calculated: $\mathcal{P}_i^{\mathcal{G}} = \sigma(\frac{\mathbf{D}_{\text{EC}}(X_t, o_i) - \text{mean}}{\text{std}})$. Final reward: $\mathcal{R}_i = 1 - \mathcal{T}(\mathcal{G}) \cdot \beta \cdot \mathcal{P}_i^{\mathcal{G}}$ (if correct) or 0 (if incorrect).
    - **Design Motivation**: Prematurely penalizing edits harms the learning of correctness—edit constraints are introduced only when group-level correctness is sufficiently high, achieving a "correct first, then precise" strategy.

### Loss & Training

EA-GRPO utilizes a PPO-style clipped objective + KL regularization. Reward calculation does not require gold-standard code; it uses only the edit cost between the bug input and the generated output. Evaluation is conducted on Python (HumanEvalFix) and Verilog (self-constructed benchmark).

## Key Experimental Results

### Main Results

**Comparison of Precise Repair Metrics**

| Metric | Description |
| :--- | :--- |
| $\text{fix}_1@1$ Gain | Up to +31.4% |
| pass@k Maintenance/Gain | Correctness does not decrease |
| Cross-language Validity | Effective on both Python and Verilog |

### Ablation Study

**EA-GRPO vs. Standard GRPO**

| Configuration | Description |
| :--- | :--- |
| Standard GRPO | Correctness increases but edit cost grows continuously |
| EA-GRPO | Correctness increases and edit cost is controlled |
| Speculative Editing Acceleration | Lower edit cost → higher speculative decoding acceptance rate → inference acceleration |

### Key Findings

- **Ours** achieves a gain of up to 31.4% in $\text{fix}_1@1$ while maintaining or improving pass@k.
- The dynamic activation design of EA-GRPO is crucial—penalizing edits too early significantly damages correctness.
- Min-max sampling in Self-Breaking ensures training bug diversity, outperforming random sampling.
- The model learns implicit error localization capabilities—precise repair forces the model to focus on the lines containing bugs.
- When combined with speculative editing, reduced edit cost directly translates to inference acceleration—providing significant practical value.

## Highlights & Insights

- The identification and quantification of the over-editing problem is a major contribution—revealing systematic flaws in RL training that optimizes only for correctness.
- The "correct first, then precise" strategy of EA-GRPO is elegant—avoiding a hard conflict between correctness and precision.
- Natural synergy with speculative decoding—precise repair reduces edits → more n-gram matches → higher inference throughput—transforming training improvements into inference acceleration.

## Limitations & Future Work

- Evaluated only on Python and Verilog; does not cover a broader range of programming languages.
- The selection of threshold $p$ for $\text{fix}_p@k$ significantly impacts evaluation results.
- Self-Breaking relies on the model's own bug injection capability, which may not cover all real-world bug types.
- Edit cost is based on line-level Levenshtein distance, which may not capture edit minimality at the semantic level.

## Related Work & Insights

- **vs. Standard GRPO (Shao et al., 2024)**: The latter optimizes only for correctness, leading to over-editing; EA-GRPO resolves this via dynamic edit penalties.
- **vs. HumanEvalFix (Muennighoff et al., 2023)**: The latter only uses pass@k for evaluation; the proposed $\text{fix}_p@k$ in **Ours** is more comprehensive.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The identification of over-editing and the design of EA-GRPO are both novel and practical.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Python + Verilog cross-language analysis + speculative decoding acceleration analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation and well-justified metric design.
- **Value**: ⭐⭐⭐⭐⭐ Directly impacts code repair practices, with deployment value through speculative decoding synergy.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] QiMeng-SALV: Signal-Aware Learning for Verilog Code Generation](../../NeurIPS2025/code_intelligence/qimeng-salv_signal-aware_learning_for_verilog_code_generation.md)
- [\[ACL 2026\] Precise Debugging Benchmark: Is Your Model Debugging or Regenerating?](precise_debugging_benchmark_is_your_model_debugging_or_regenerating.md)
- [\[ICML 2026\] NEMO: Execution-Aware Optimization Modeling via Autonomous Coding Agents](../../ICML2026/code_intelligence/nemo_execution-aware_optimization_modeling_via_autonomous_coding_agents.md)
- [\[ACL 2026\] OmniDiagram: Advancing Unified Diagram Code Generation via Visual Interrogation Reward](omnidiagram_advancing_unified_diagram_code_generation_via_visual_interrogation_r.md)
- [\[ICML 2026\] BoostAPR: Boosting Automated Program Repair via Execution-Grounded Reinforcement Learning with Dual Reward Models](../../ICML2026/code_intelligence/boostapr_boosting_automated_program_repair_via_execution-grounded_reinforcement_.md)

</div>

<!-- RELATED:END -->

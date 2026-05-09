---
title: >-
  [Paper Note] QiMeng-PRepair: Precise Code Repair via Edit-Aware Reward Optimization
description: >-
  [ACL 2026][precise code repair] This paper identifies the "over-editing" problem in LLM-based code repair—where models tend to rewrite large portions of code rather than precisely localizing and fixing bugs—and proposes the PRepair framework. Through Self-Breaking (diversified bug injection) and Self-Repairing (edit-aware GRPO training), PRepair significantly improves repair precision while maintaining correctness and accelerating speculative decoding inference.
tags:
  - ACL 2026
  - precise code repair
  - over-editing
  - edit-aware reward
  - GRPO
  - speculative editing
date: 2026-05-08
content_hash: a11e03a922680338
---

# QiMeng-PRepair: Precise Code Repair via Edit-Aware Reward Optimization

**Conference**: ACL 2026
**arXiv**: [2604.05963](https://arxiv.org/abs/2604.05963)
**Code**: [GitHub](https://github.com/...)
**Area**: Code Intelligence / Program Repair
**Keywords**: precise code repair, over-editing, edit-aware reward, GRPO, speculative editing

## TL;DR

This paper identifies the "over-editing" problem in LLM-based code repair—where models tend to rewrite large portions of code rather than precisely localizing and fixing bugs—and proposes the PRepair framework. Through Self-Breaking (diversified bug injection) and Self-Repairing (edit-aware GRPO training), PRepair significantly improves repair precision while maintaining correctness and accelerating speculative decoding inference.

## Background & Motivation

**Background**: LLMs have demonstrated strong performance in program repair. Existing training approaches (SFT and RL) typically optimize only for repair correctness, treating code repair as a pure correctness objective.

**Limitations of Prior Work**: (1) During GRPO training, as correctness improves, edit cost increases in tandem—models learn to "stumble upon" correct solutions through extensive modifications rather than developing precise repair capabilities. (2) Over-editing disrupts the original code structure, increasing the reviewer burden on developers. (3) Over-editing fails to localize bugs, limiting the practical effectiveness and maintainability of repairs.

**Key Challenge**: There exists a tension between repair correctness and edit minimality—optimizing solely for correctness causes models to take the "rewrite" shortcut rather than learning to understand and precisely localize bugs.

**Goal**: Design a Precise Repair framework that maximizes reuse of the original code while maintaining repair correctness.

**Key Insight**: An observation that edit cost and correctness grow in tandem during GRPO training (Figure 2), indicating that edit constraints must be explicitly incorporated into the reward signal.

**Core Idea**: Edit-Aware GRPO (EA-GRPO)—edit penalties are applied to correct samples only when group-level accuracy exceeds a threshold, balancing correctness and edit minimality.

## Method

### Overall Architecture

PRepair consists of two stages: (1) Self-Breaking—the model injects diversified bugs into correct code using a min-max sampling strategy to maximize bug diversity; (2) Self-Repairing—the model is trained on the generated buggy code via EA-GRPO, where an edit-aware reward dynamically balances correctness and edit cost. Evaluation is conducted using the newly proposed $\text{fix}_p@k$ metric.

### Key Designs

1. **$\text{fix}_p@k$ Precise Repair Metric**:

    - Function: Jointly evaluates repair correctness and edit extent.
    - Mechanism: Extends pass@k with an edit constraint—a repair is considered successful only if the generated code passes all tests and the edit cost does not exceed $p$ times the theoretically minimal edit. Edit cost $\mathbf{D}_{\text{EC}}(X,Y) = \mathbf{D}(X,Y)/|X|$ is the line-level Levenshtein distance normalized by source length.
    - Design Motivation: pass@k reflects only correctness and cannot capture repair quality—rewriting the entire codebase may pass all tests but does not constitute a good repair.

2. **Self-Breaking (Diversified Bug Injection)**:

    - Function: Generates large-scale precise repair training data without manual annotation.
    - Mechanism: The model is prompted to inject bugs given correct code and its description. From $m$ candidates, $k$ maximally diverse samples are selected via min-max sampling: $\mathcal{X}_s = \min_{\mathcal{X}' \subset \mathcal{X}, |\mathcal{X}'|=k} \max_{X_i,X_j \in \mathcal{X}', i \neq j} (1 - \mathbf{D}_{\text{EC}}(X_i, X_j))$
    - Design Motivation: Precise repair requires training data that preserves substantial correct logic with only localized errors—such data is extremely scarce in practice. Min-max sampling prevents over-concentration of bug patterns.

3. **EA-GRPO (Edit-Aware Group Relative Policy Optimization)**:

    - Function: Encourages minimal yet correct repairs during RL training.
    - Mechanism: For each rollout group, the accuracy $\text{Acc}_{\mathcal{G}^t}$ is computed; edit penalties are activated only when this value exceeds threshold $\alpha$. For correct samples within the group, a normalized edit penalty is computed as $\mathcal{P}_i^{\mathcal{G}} = \sigma(\frac{\mathbf{D}_{\text{EC}}(X_t, o_i) - \text{mean}}{\text{std}})$. The final reward is: $\mathcal{R}_i = 1 - \mathcal{T}(\mathcal{G}) \cdot \beta \cdot \mathcal{P}_i^{\mathcal{G}}$ (for correct outputs) or $0$ (for incorrect outputs).
    - Design Motivation: Imposing edit penalties too early harms correctness learning—edit constraints are introduced only after group-level correctness is sufficiently high, realizing a "correctness first, then precision" curriculum.

### Loss & Training

EA-GRPO employs a PPO-style clipped objective with KL regularization. Reward computation requires no gold-standard code—only the edit cost between the buggy input and the generated output. Evaluation is conducted on Python (HumanEvalFix) and Verilog (a newly constructed benchmark).

## Key Experimental Results

### Main Results

**Precise Repair Metric Comparison**

| Metric | Description |
|--------|-------------|
| $\text{fix}_1@1$ improvement | Up to +31.4% |
| pass@k maintained/improved | Correctness does not degrade |
| Cross-language effectiveness | Effective on both Python and Verilog |

### Ablation Study

**EA-GRPO vs. Standard GRPO**

| Configuration | Description |
|---------------|-------------|
| Standard GRPO | Correctness improves but edit cost grows continuously |
| EA-GRPO | Correctness improves with edit cost kept under control |
| Speculative editing speedup | Reduced edit cost → higher speculative decoding acceptance rate → faster inference |

### Key Findings

- PRepair achieves up to +31.4% improvement on $\text{fix}_1@1$ while maintaining or improving pass@k.
- The dynamic activation design of EA-GRPO is critical—imposing edit penalties too early significantly degrades correctness.
- The min-max sampling in Self-Breaking ensures training bug diversity, outperforming random sampling.
- Models learn implicit fault localization capabilities—precise repair compels models to focus on the buggy lines.
- When combined with speculative editing, reduced edit cost directly translates into inference speedup, providing substantial practical value.

## Highlights & Insights

- The identification and quantification of the over-editing problem is a significant contribution—it reveals a systematic deficiency of RL training that optimizes only for correctness.
- The "correctness first, then precision" strategy of EA-GRPO is elegant, avoiding hard conflicts between correctness and precision objectives.
- The natural synergy with speculative decoding—precise repair reduces edits → more n-gram matches → higher inference throughput—converts training improvements into inference acceleration.

## Limitations & Future Work

- Evaluation is limited to Python and Verilog; broader programming languages are not covered.
- The choice of threshold $p$ in $\text{fix}_p@k$ substantially affects evaluation outcomes.
- Self-Breaking relies on the model's own bug injection capability, which may not cover all real-world bug types.
- Edit cost is based on line-level Levenshtein distance, which may fail to capture semantic-level edit minimality.

## Related Work & Insights

- **vs. Standard GRPO (Shao et al., 2024)**: The latter optimizes only for correctness, leading to over-editing; EA-GRPO addresses this via dynamic edit penalization.
- **vs. HumanEvalFix (Muennighoff et al., 2023)**: The latter evaluates solely with pass@k; the proposed $\text{fix}_p@k$ provides a more comprehensive assessment.

## Rating

- Novelty: ⭐⭐⭐⭐ The identification of over-editing and the design of EA-GRPO are novel and practically motivated.
- Experimental Thoroughness: ⭐⭐⭐⭐ Cross-language evaluation on Python and Verilog with speculative decoding acceleration analysis.
- Writing Quality: ⭐⭐⭐⭐ Problem motivation is clear and metric design is well-justified.
- Value: ⭐⭐⭐⭐⭐ Direct impact on code repair practice; the speculative decoding synergy holds significant deployment value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] QiMeng-SALV: Signal-Aware Learning for Verilog Code Generation](../../NeurIPS2025/code_intelligence/qimeng-salv_signal-aware_learning_for_verilog_code_generation.md)
- [\[ACL 2026\] OmniDiagram: Advancing Unified Diagram Code Generation via Visual Interrogation Reward](omnidiagram_advancing_unified_diagram_code_generation_via_visual_interrogation_r.md)
- [\[ACL 2026\] Precise Debugging Benchmark: Is Your Model Debugging or Regenerating?](precise_debugging_benchmark_is_your_model_debugging_or_regenerating.md)
- [\[ACL 2026\] LogicEval: A Systematic Framework for Evaluating Automated Repair Techniques for Logical Vulnerabilities in Real-World Software](logiceval_a_systematic_framework_for_evaluating_automated_repair_techniques_for_.md)
- [\[ACL 2026\] SOCIA-EVO: Automated Simulator Construction via Dual-Anchored Bi-Level Optimization](socia-evo_automated_simulator_construction_via_dual-anchored_bi-level_optimizati.md)

</div>

<!-- RELATED:END -->

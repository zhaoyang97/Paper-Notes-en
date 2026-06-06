---
title: >-
  [Paper Note] Capability-Oriented Failure Attribution for Vision-Language Navigation Agents
description: >-
  [ACL 2026][Robotics][Vision-Language Navigation] Addressing multi-level capability failures in embodied agents (specifically vision-language navigation VLN agents)…
tags:
  - "ACL 2026"
  - "Robotics"
  - "Vision-Language Navigation"
  - "Capability Failure Diagnosis"
  - "Testing Framework"
  - "Embodied Agents"
  - "Fuzz Testing"
date: 2026-05-08
content_hash: 94f259280ec10456
---

# Capability-Oriented Failure Attribution for Vision-Language Navigation Agents

**Conference**: ACL 2026  
**arXiv**: [2604.25161](https://arxiv.org/abs/2604.25161)  
**Code**: https://github.com/JMChen121/CanTest/  
**Area**: Robotics / Embodied AI / Navigation  
**Keywords**: Vision-Language Navigation, Capability Failure Diagnosis, Testing Framework, Embodied Agents, Fuzz Testing

## TL;DR

Addressing multi-level capability failures in embodied agents (specifically vision-language navigation VLN agents), this paper proposes the CanTest framework. Through capability-oriented test oracles and failure attribution mechanisms, it precisely locates specific capability defects (perception/memory/planning/decision) causing task failure, discovering 23–34% more failure cases than existing methods.

## Background & Motivation

**Background**: Reliability assessment of embodied agents in safety-critical applications (e.g., vision-language navigation, household robots) primarily relies on task-level metrics (path length, execution time, etc.), lacking in-depth testing of the agents' internal capability structures.

**Limitations of Prior Work**:

- VLN agents integrate four capabilities—perception, memory, planning, and decision—which are tightly coupled and interdependent.
- During failure, upstream capability errors propagate to downstream stages (e.g., perception errors leads to memory confusion, which in turn leads to planning errors).
- It is difficult to trace the initial source of failure across long trajectories.
- Developers cannot precisely locate weak links for targeted improvements.

**Key Challenge**: There exists a significant gap between system-level failure detection ("the task failed") and capability-level failure diagnosis ("which capability caused it"). For long-sequence embodied tasks, knowing a failure occurred without knowing its root cause provides almost no guidance for improvement.

**Goal**: Develop a **capability-oriented testing method** that can: (1) automatically generate test cases likely to expose specific capability defects; (2) construct independent evaluation criteria (oracles) for each capability; (3) accurately attribute failures from long trajectories to a specific capability and its first moment of error.

**Key Insight**: Transform the problem of failure attribution in long trajectories into counterfactual reasoning: for each detected capability error, attempt to replace it with the oracle's correct output and observe if the trajectory becomes successful. If it does, the error is a "failure-inducing error." Identifying the earliest among multiple inducing errors reveals the root cause of failure.

**Core Idea**: Combine fuzz testing (fuzzing) with capability-level oracles and counterfactual causal reasoning, designing an adaptive feedback scoring mechanism.

## Method

### Overall Architecture

CanTest consists of three modules:

1.  **Adaptive Test Case Generation**: Based on fuzzing principles, it maintains a seed pool with feedback scores. Each iteration selects a seed and performs mutations at two intensities (mild/aggressive) to generate new natural language instructions.
2.  **Capability Oracle Construction**: Builds an oracle for each of the four capabilities (perception, memory, planning, decision) to automatically extract expected outputs and define independent error metrics.
3.  **Capability-Oriented and Failure-Oriented Feedback**: After each test case execution, capability outputs are checked using the oracles. "Failure-inducing errors" are identified via counterfactual intervention to attribute failures to specific capabilities and calculate mixed feedback scores to guide the next round of generation.

### Key Designs

1.  **Adaptive Test Case Generation**:
    *   **Function**: Generates natural language navigation instructions that can expose capability defects.
    *   **Mechanism**: (1) Initialize the seed pool; (2) In each round, select seeds with probability $p_{cs_i} = \max(F_{cs_i}, 0) / \sum_{i=1}^{N} F_{cs_i}$ based on feedback scores; (3) Calculate mutation intensity $p_m = (F_{cs} - \min(\mathbf{F})) / (\max(\mathbf{F}) - \min(\mathbf{F}))$ based on the seed's feedback score. Mild mutations are used for high-score seeds, while aggressive mutations are used for low-score seeds.
    *   **Design Motivation**: High-score seeds are proven likely to fail, so mild mutations refine them while preserving failure patterns. Low-score seeds require expansion of the search space, so aggressive mutations force the agent through different routes to expose other defects.

2.  **Capability Oracle Construction**:
    *   **Function**: Defines independent error criteria for each of the four capabilities, comparing the agent's actual output with the expected output.
    *   **Mechanism**: Utilize expert models provided by the simulation environment to obtain ground truth (GT): a navigation expert provides the optimal path via greedy planning, an image tagging model (RAM) provides perception GT, and historical visual annotations are recorded as memory GT. Independent distance metrics are then defined for the four capabilities. For example, the perception oracle $\epsilon_t^p = \frac{1}{N}\sum_n (\|VA_{t,n} - VA_{t,n}^{gt}\|_{\mathbb{L}} - |P_{t,n} \cap P_{t,n}^{gt}| / |P_{t,n} \cup P_{t,n}^{gt}|)$, which fuses LLM similarity and IoU; the planning oracle $\epsilon_t^{pl} = 1 - \text{nDTW}(\tau_t^{pl}, \tau_{t,\ldots,n}^{gt})$; and the decision oracle $\epsilon_t^d = 1 - \|D_t - D_t^{pl}\|$.
    *   **Design Motivation**: Different capabilities have different output formats and require customized metrics. Relying on expert models automates oracle construction.

3.  **Failure Attribution and Mixed Feedback**:
    *   **Function**: Identifies which error of which capability caused the failure, determines if the failure can be reversed under counterfactual intervention, and guides test generation.
    *   **Mechanism**: (1) Use capability oracles to check if any of the four capabilities at all moments $t$ are erroneous, yielding the error set $C^{errors}$; (2) For each error $(C_x, t)$, replace the agent's output at that moment with the oracle's output and re-simulate the remaining trajectory. It is a "failure-inducing error" if the task changes from failure to success; (3) Among multiple inducing errors, select the earliest: $(C_x^*, t^*) = \arg\min_{(C_x', t') \in \mathbb{C}(\tau)} t$; (4) Compute the mixed feedback score $F_{cs} = F^f + \lambda^{C_x} F^c$, where $F^f \in \{0, 1\}$ indicates task success/failure, $F^c = \text{Norm}(\epsilon_{t^*}^x)$ is the normalized error value of the source capability at the source moment, and weight $\lambda^{C_x} = \overline{N^{C_y}} / N^{C_x}$ adaptively balances exploration across capabilities.
    *   **Design Motivation**: Counterfactual reasoning accurately determines whether an error truly caused a failure. The earliest error rule corresponds to tracing the root cause. Mixed feedback focuses on both task-level failure and capability-level errors.

### Experimental Design Details

The study uses the Habitat 3 VLN simulation environment. The HM3D dataset provides 216 large-scale indoor 3D scenes with semantic annotations. Three advanced VLN models (ApexNav, MGDM, Mem2Ego) are tested and compared against three baselines: Random, BehAVExplor, and VLATest.

## Key Experimental Results

### Main Results: Comparison of Discovered Failure Cases

| Method | ApexNav | MGDM | Mem2Ego | Average Improvement |
| :--- | :--- | :--- | :--- | :--- |
| Random | ~20–25 | ~23–28 | ~18–22 | Baseline |
| BehAVExplor + OA | ~41–49 | ~42–51 | ~37–46 | Baseline |
| VLATest + OA | ~52–58 | ~56–63 | ~50–58 | Baseline |
| **CanTest (Ours)** | **72–75** | **74–76** | **61–65** | **+23–34%** |

Note: OA indicates integrating CanTest's oracle and attribution mechanism as plugins into the baseline. CanTest consistently outperforms all baselines across all models.

### Breakdown of Capability-Level Failures

| Capability | ApexNav | MGDM | Mem2Ego | Description |
| :--- | :--- | :--- | :--- | :--- |
| Perception Failure | 72.2 | 74.7 | 61.4 | CanTest is strongest in finding perception failures |
| Memory Failure | 66.3 | 56.1 | 42.8 | Significant variance in memory capability across models |
| Planning Failure | 52.5 | 49.3 | 66.1 | Fewer planning failures overall |
| Decision Failure | 59.5 | 64.7 | 63.4 | Decision failures are relatively stable |

### Repair Experiment: Fixing Failure Cases with Oracle Correct Outputs

| Capability | ApexNav Repair Rate | MGDM Repair Rate | Mem2Ego Repair Rate |
| :--- | :--- | :--- | :--- |
| Perception | 84.35% | 83.53% | 85.83% |
| Memory | 81.30% | 82.35% | 83.64% |
| Planning | 87.05% | 86.41% | 89.71% |
| Decision | 95.13% | 94.90% | 96.69% |

Repair rates > 81% indicate high oracle credibility. Repair rates for upstream capabilities (perception, memory) are slightly lower because upstream errors propagating downstream can trigger multi-stage errors.

### Key Findings

- **High Fidelity Oracles**: Repair rates > 81% show that automatically constructed oracles capture real capability errors.
- **Upstream Errors are More Damaging**: Perception/memory repair rates are lower than planning/decision because upstream errors cascade across the trajectory.
- **High Diversity**: Manual analysis of 100 failure cases revealed 8 fine-grained failure types, whereas baselines only covered 6.
- **Ablation Study**: Removing failure-oriented feedback, capability-oriented feedback, or both resulted in success counts of 62–68, 62–70, and 45–55 respectively, proving both signals contribute to failure discovery.

## Highlights & Insights

- **Clever Application of Counterfactual Reasoning in Embodied Testing**: Identifying failure causes by replacing erroneous capability outputs with GT allows for elegant and explainable root cause analysis in long trajectories.
- **Automated Framework for Capability Oracles**: Instead of manually designing evaluation criteria for each capability, it uses expert models to automatically obtain GT. This is highly practical for scenarios lacking human annotation.
- **Adaptive Feedback Weights Balance Exploration**: By dynamically adjusting $\lambda^{C_x}$, the framework avoids local loops in test generation by lowering weights for fully explored capabilities and increasing them for under-explored ones.
- **Detailed Failure Diversity Analysis**: Beyond reporting total failures per capability, the manual annotation of 8 fine-grained failure types provides a more precise diagnostic view than simple "perception/memory/planning/decision" categorization.

## Limitations & Future Work

**Limitations acknowledged by the authors**:

1.  **Reliance on Expert Models**: Constructing oracles requires GT, such as optimal path planning and perception labels. Obtaining such privileged information in real-world environments is difficult.
2.  **Sim-to-Real Gap**: Current evaluations are conducted in the Habitat simulation environment. Real-world noise and dynamism may cause oracle designs to fail.

**Extension directions from an independent perspective**:

1.  Oracles currently rely on expert model GT; future work could explore weakly supervised oracles (using signals distilled from demonstrations, corrective feedback, or safety monitors).
2.  Capability definitions and oracle designs for embodied tasks beyond VLN (e.g., robotic arm manipulation, multi-agent collaboration) may differ and require a generalized framework.
3.  Current attribution only handles the single earliest failure source on a long trajectory; future models could consider attribution for multiple concurrent failure sources.

## Related Work & Insights

- **vs BehAVExplor**: BehAVExplor uses behavior-guided fuzzing to generate diverse test cases, but its feedback signals come only from system-level task success/failure, making it unable to distinguish failure root causes. CanTest's capability-level feedback enables more precise exploration and a 23% increase in failure discovery.
- **vs VLATest**: VLATest is a SOTA testing framework for robotic manipulation. CanTest customizes capability oracles and counterfactual attribution for VLN, providing stronger diagnostic power for multimodal embodied agents than general operator-based methods.
- **vs Traditional Software Testing**: CanTest borrows ideas from counterfactual causal reasoning, using backward induction to locate sources—an innovative application of causal reasoning in embodied AI testing.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First to systematically combine capability-level testing, automated oracle construction, and counterfactual failure attribution for embodied agents.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive evaluation using three VLN models, three baseline comparisons, ablation studies, repair rate validation, and manual diversity analysis; lacks real-world validation.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation, thorough methodological explanation, and explicit experimental conclusions.
- **Value**: ⭐⭐⭐⭐⭐ Provides significant inspiration for testing and diagnosing embodied AI; the oracle framework is transferable to other multi-capability systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Breaking Down and Building Up: Mixture of Skill-Based Vision-and-Language Navigation Agents](breaking_down_and_building_up_mixture_of_skill-based_vision-and-language_navigat.md)
- [\[ACL 2026\] VLN-NF: Feasibility-Aware Vision-and-Language Navigation with False-Premise Instructions](vln-nf_feasibility-aware_vision-and-language_navigation_with_false-premise_instr.md)
- [\[NeurIPS 2025\] SAFE: Multitask Failure Detection for Vision-Language-Action Models](../../NeurIPS2025/robotics/safe_multitask_failure_detection_for_vision-language-action_models.md)
- [\[ICLR 2026\] JanusVLN: Decoupling Semantics and Spatiality with Dual Implicit Memory for Vision-Language Navigation](../../ICLR2026/robotics/janusvln_decoupling_semantics_and_spatiality_with_dual_implicit_memory_for_visio.md)
- [\[ACL 2026\] GROKE: Vision-Free Navigation Instruction Evaluation via Graph Reasoning on OpenStreetMap](groke_vision-free_navigation_instruction_evaluation_via_graph_reasoning_on_opens.md)

</div>

<!-- RELATED:END -->

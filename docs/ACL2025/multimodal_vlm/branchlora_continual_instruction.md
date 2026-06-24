---
title: >-
  [Paper Note] Enhancing Multimodal Continual Instruction Tuning with BranchLoRA
description: >-
  [ACL 2025][Multimodal VLM][Continual Instruction Tuning] To address parameter inefficiency and catastrophic forgetting of MoELoRA in multimodal continual instruction tuning (MCIT), this paper proposes BranchLoRA—an asymmetric architecture. It shares matrix A to capture cross-task general patterns and maintains multi-branch matrices B to encode task-specific knowledge. Complemented by a flexible tuning-freezing mechanism and task-specific routers…
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "Continual Instruction Tuning"
  - "BranchLoRA"
  - "MoE"
  - "Catastrophic Forgetting"
  - "Multimodal Large Language Models"
date: 2026-05-08
content_hash: 833c72c7d959ff66
---

# Enhancing Multimodal Continual Instruction Tuning with BranchLoRA

**Conference**: ACL 2025  
**arXiv**: [2506.02041](https://arxiv.org/abs/2506.02041)  
**Code**: [GitHub](https://github.com/BladeDancer957/BranchLoRA)  
**Area**: Multimodal VLM  
**Keywords**: Continual Instruction Tuning, BranchLoRA, MoE, Catastrophic Forgetting, Multimodal Large Language Models

## TL;DR

To address parameter inefficiency and catastrophic forgetting of MoELoRA in multimodal continual instruction tuning (MCIT), this paper proposes BranchLoRA—an asymmetric architecture. It shares matrix A to capture cross-task general patterns and maintains multi-branch matrices B to encode task-specific knowledge. Complemented by a flexible tuning-freezing mechanism and task-specific routers, it significantly outperforms the previous SOTA MoELoRA on the CoIN benchmark with fewer parameters (ACC: 44.20 vs 37.13, BWT: -20.98 vs -25.91).

## Background & Motivation

**Background**: Multimodal large language models (MLLMs) achieve alignment with human intents through instruction tuning. In practical applications, models must continuously adapt to new tasks and instructions. Since retraining from scratch is computationally prohibitive, the Multimodal Continual Instruction Tuning (MCIT) paradigm has emerged.

**Limitations of Prior Work**:
- MCIT suffers from catastrophic forgetting (CF)—the performance on historical tasks degrades sharply when learning new tasks.
- Existing MoELoRA methods aggregate outputs from all LoRA experts, which easily overwrites historical knowledge.
- The shared router in MoELoRA is continuously updated, biasing expert allocation toward the most recent tasks.
- Matrices A and B are maintained independently for all experts, causing parameter redundancy.

**Key Challenge**: In MoELoRA, the parameters of matrix A across multiple experts tend to converge during continual training (capturing common patterns), while matrix B remains highly distinguishable (capturing task-specific details), indicating that keeping independent copies of A for each expert is parameter-wasteful.

**Goal**: Design a more efficient LoRA architecture under the MCIT scenario to simultaneously resolve the core issues of parameter redundancy and catastrophic forgetting in MoELoRA.

**Key Insight**: An empirical analysis reveals the convergence of matrix A in MoELoRA. Based on this, an asymmetric architecture is designed with a shared matrix A (the "trunk") and multi-branch matrices B (the "branches"), accompanied by freezing and routing mechanisms to prevent forgetting.

**Core Idea**: Matrix A convergence + Matrix B divergence in MoELoRA $\rightarrow$ Asymmetric BranchLoRA with shared A + multi-branch B + flexible freezing mechanism + task-specific routers = fewer parameters + less forgetting.

## Method

### Overall Architecture

BranchLoRA is integrated into the Feed-Forward module of each layer in the MLLM. The pipeline is as follows:
1. The input goes through multi-head attention to obtain the intermediate representation $x$.
2. $x$ is projected into a low-dimensional space via the shared matrix A.
3. A task-specific router computes expert weights based on the first token of $x$, sparsely selecting the top-$k$ matrices B.
4. Each matrix B independently projects the representation back to the high-dimensional space, which is then aggregated using the router weights.
5. During inference, a task selector automatically routes to the correct router (without requiring task identity).

### Key Designs

1. **Asymmetric Architecture (Shared A + Multi-branch B)**:
    - Function: Eliminates parameter redundancy while maintaining the capability to encode task-specific knowledge.
    - Mechanism: All experts share a single matrix A (capturing cross-task commonalities), while each expert maintains an independent matrix B (capturing task-specific properties), establishing a "trunk-branch" architecture.
    - Design Motivation: Empirical observations indicate that matrices A of MoELoRA converge during continual training (highly overlapping in t-SNE visualizations), while matrices B remain distinct, rendering multiple independent copies of A unnecessary.

2. **Flexible Tuning-Freezing Mechanism**:
    - Function: Protects historical task knowledge while enabling cross-task knowledge transfer.
    - Mechanism: After training the current task, the router's output distribution is analyzed, and the most active top-$k$ matrices B are frozen. During training of a new task, the router can select from (a) tunable experts only, (b) a mix of tunable and frozen experts, or (c) frozen experts only.
    - Design Motivation: Freezing prevents forgetting (historical knowledge is not overwritten), yet allowing the router to access frozen experts promotes cross-task knowledge transfer (analogous to how the human brain consolidates memories while integrating new information).

3. **Task-Specific Router + Automatic Task Selector**:
    - Function: Prevents the router from biasing toward the most recent task and eliminates the need for task IDs during inference.
    - Mechanism: Dynamically introduces a new router (with independent $W_r$ parameters) for each new task, alongside training a corresponding task key (image key + text key), which is aligned with sample embeddings using a cosine similarity alignment loss.
    - Design Motivation: Continuous updates to a shared router lead to forgetting the optimal expert allocation of historical tasks. During inference, the router is automatically selected by computing the similarity between test samples and task keys (achieving 95.8% accuracy).

### Loss & Training

- Total loss: $L_{total} = L_{task} + \lambda \cdot L_{align}$
- $L_{task}$: Standard autoregressive generation loss
- $L_{align} = \sum(1-\cos(e_{img}, k_{img})) + \sum(1-\cos(e_{txt}, k_{txt}))$, which aligns the task keys with sample embeddings.
- Parameter settings: rank $= 128$, $\alpha = 256$, $N = 8$ experts, top-$k = 2$, $\lambda = 1.0$
- Freeze the vision encoder and LLM, only fine-tuning the projector and LoRA.
- Trained using 8 $\times$ NVIDIA H800 GPUs.

## Key Experimental Results

### Main Results (LLaVA-1.5-7B, CoIN benchmark, 8 sequential tasks)

| Method | ACC↑ | MAA↑ | BWT↑ | Tunable Params |
|---|---|---|---|---|
| LoRA | 28.74 | 32.97 | -32.62 | - |
| LwF | 30.41 | 34.95 | -27.03 | - |
| EWC | 32.90 | 36.93 | -27.46 | - |
| MoELoRA | 37.13 | 42.76 | -25.91 | 350M |
| **BranchLoRA** | **44.20** | **49.94** | **-20.98** | **222M** |
| Multi-task (Upper Bound) | - | 57.18 | - | - |

### Model Scaling (LLaVA-1.5-13B)

| Method | ACC↑ | MAA↑ | BWT↑ |
|---|---|---|---|
| MoELoRA | 42.51 | 49.14 | -23.62 |
| **BranchLoRA** | **49.27** | **55.73** | **-19.29** |

### Ablation Study (LLaVA-1.5-7B)

| Variant | ACC↑ | MAA↑ | BWT↑ |
|---|---|---|---|
| MoELoRA baseline | 37.13 | 42.76 | -25.91 |
| + Shared Matrix A | 38.19 | 43.95 | -25.32 |
| + Dynamic Chunking Routing | 39.96 | 45.53 | -23.77 |
| + Flexible Freezing Mechanism | 42.22 | 47.76 | -22.41 |
| + Task-Specific Router (Full BranchLoRA) | **44.20** | **49.94** | **-20.98** |

### Efficiency Comparison

| Method | Tunable Params | Training Time (ms/batch) |
|---|---|---|
| MoELoRA | 350M | 62 |
| BranchLoRA | **222M** | **51** |

### Key Findings

- Sharing A not only reduces parameters by 37% but also slightly improves performance, validating the observation of matrix A convergence.
- Each design component contributes incremental improvements: Shared A $\rightarrow$ Sparse Routing $\rightarrow$ Freezing Mechanism $\rightarrow$ Task Router.
- Consistently outperforms MoELoRA on both 7B and 13B scales, demonstrating the scalability of the method.
- The larger model (13B) exhibits less forgetting (BWT: -19.29 vs -20.98), though forgetting still persists.
- Increasing instruction diversity (10Type) further boosts the performance of BranchLoRA (ACC: 44.20 $\rightarrow$ 46.47).
- The task selector achieves an accuracy of 95.8%—occasional misclassifications do not compromise the overall advantages.

## Highlights & Insights

- **Data-driven Architectural Design**: Rather than designing the architecture empirically, this paper first uncovers the convergence of matrix A via parameter analysis and subsequently positions an asymmetric structure. This methodology is highly instructive.
- **Precise "Trunk-Branch" Analogy**: Shared A acts as the trunk (a stable, shared foundation) and multi-branch B acts as the branches (flexible task adaptation), which is intuitive and accurate.
- **Freezing Mechanism Mimics Human Memory Consolidation**: Protecting historical knowledge via freezing while facilitating memory transfer by accessing existing experts via the router is biologically inspired.
- **Win-win in Efficiency and Effectiveness**: Reduces parameters by 37% and accelerates training speed by 18% compared to MoELoRA, while boosting accuracy by 7 percentage points—a rare Pareto improvement.
- **Practical Inference Scheme**: The task selector eliminates the dependency on task IDs, rendering the method much closer to real-world applications.

## Limitations & Future Work

- The evaluation is limited to the CoIN benchmark with constrained task diversity (8 multimodal datasets).
- The impact of task sequence ordering on performance remains under-explored.
- Whether the choice of top-$k$ ($k=2$) needs adjustment across different scenarios is not discussed.
- As the number of tasks grows extremely large, freezing experts might result in an insufficient number of tunable experts.
- No comparison or integration with other approaches such as model merging has been conducted.
- The performance on non-multimodal tasks has not been validated.

## Related Work & Insights

- **vs MoELoRA (CoIN)**: The direct baseline. BranchLoRA outperforms it by a large margin through a three-fold improvement: asymmetric architecture, freezing, and task routing.
- **vs HydraLoRA (Tian et al. 2024)**: HydraLoRA observes a similar convergence of matrix A in multi-task scenarios; however, this work further advances this finding in the context of continual learning.
- **vs EWC/LwF**: Traditional continual learning methods yield limited success under MCIT scenarios and are significantly outperformed by MoE-based strategies.
- **vs Standard LoRA**: Standard LoRA lacks any forgetting mitigation measures, showing a BWT of -32.62, which is drastically worse than BranchLoRA's -20.98.

## Rating

- Novelty: ⭐⭐⭐⭐ Designing an asymmetric architecture based on parameter-level observations, paired with a novel combination of flexible freezing and task routing.
- Experimental Thoroughness: ⭐⭐⭐⭐ Complete ablation studies, validation across dual-scale models, and solid efficiency analysis; however, the evaluated benchmarks are somewhat singular.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation (from parameter analysis to architectural design), intuitive charts, and coherent logic.
- Value: ⭐⭐⭐⭐ Provides a superior alternative to MoELoRA for continual learning of MLLMs with high practicality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] HiDe-LLaVA: Hierarchical Decoupling for Continual Instruction Tuning of Multimodal Large Language Model](hidellava_hierarchical_decoupling_for_continual_instruction.md)
- [\[CVPR 2026\] Multimodal Continual Instruction Tuning with Dynamic Gradient Guidance](../../CVPR2026/multimodal_vlm/multimodal_continual_instruction_tuning_with_dynamic_gradient_guidance.md)
- [\[ICLR 2026\] PCLR: Progressively Compressed LoRA for Multimodal Continual Instruction Tuning](../../ICLR2026/multimodal_vlm/pclr_progressively_compressed_lora_for_multimodal_continual_instruction_tuning.md)
- [\[ICCV 2025\] SMoLoRA: Exploring and Defying Dual Catastrophic Forgetting in Continual Visual Instruction Tuning](../../ICCV2025/multimodal_vlm/smolora_exploring_and_defying_dual_catastrophic_forgetting_in_continual_visual_i.md)
- [\[ICML 2025\] Dynamic Mixture of Curriculum LoRA Experts for Continual Multimodal Instruction Tuning](../../ICML2025/multimodal_vlm/dynamic_mixture_of_curriculum_lora_experts_for_continual_multimodal_instruction_.md)

</div>

<!-- RELATED:END -->

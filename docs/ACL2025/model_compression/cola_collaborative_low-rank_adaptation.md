---
title: >-
  [Paper Note] CoLA: Collaborative Low-Rank Adaptation
description: >-
  [ACL 2025][Model Compression][LoRA] Proposes CoLA, a flexible LoRA architecture that breaks the fixed quantity constraint between matrices A and B (#A=M, #B=N), and designs three collaborative strategies (full collaboration / random collaboration / heuristic collaboration). Combined with an extended PiSSA initialization, it significantly outperforms existing PEFT methods in low-sample scenarios.
tags:
  - "ACL 2025"
  - "Model Compression"
  - "LoRA"
  - "parameter-efficient fine-tuning"
  - "Low-Rank Adaptation"
  - "Multi-Task Learning"
  - "LLM Fine-Tuning"
date: 2026-05-08
content_hash: 8830d7ec3a36c1b7
---

# CoLA: Collaborative Low-Rank Adaptation

**Conference**: ACL 2025  
**arXiv**: [2505.15471](https://arxiv.org/abs/2505.15471)  
**Code**: [https://github.com/zyy-2001/CoLA](https://github.com/zyy-2001/CoLA)  
**Area**: Others (Parameter-Efficient Fine-Tuning)  
**Keywords**: LoRA, parameter-efficient fine-tuning, Low-Rank Adaptation, Multi-Task Learning, LLM Fine-Tuning

## TL;DR

Proposes CoLA, a flexible LoRA architecture that breaks the fixed quantity constraint between matrices A and B (#A=M, #B=N), and designs three collaborative strategies (full collaboration / random collaboration / heuristic collaboration). Combined with an extended PiSSA initialization, it significantly outperforms existing PEFT methods in low-sample scenarios.

## Background & Motivation

Parameter-efficient fine-tuning (PEFT) is a critical technology for fine-tuning large language models under resource-constrained conditions. LoRA has become one of the most popular methods due to its simplicity and effectiveness, but it faces challenges in multi-task and low-sample scenarios:

**Limitations of a Single LoRA**: A single LoRA module projects features from different tasks into the same low-dimensional space, leading to inter-task interference.

**Issues with MOE-LoRA**: Although introducing multiple experts (#A=#B=N) decouples multi-task information, individual experts operate independently, making it difficult to capture commonalities in domain knowledge.

**Defects of Asymmetric LoRA**: Methods like HydraLoRA adopt a one-to-many structure (#A=1, #B=N), but a single matrix A struggles to effectively learn commonalities when samples are scarce, and is highly susceptible to noise interference.

**Initialization Problem**: Existing LoRA variants uniformly use Gaussian noise + zero initialization, which may result in minuscule or random gradients in the early stages of training, slowing down convergence.

**Key Insight**: Existing LoRA methods are all limited by a fixed quantity relationship between matrices A and B, and the collaborative relationships between these matrices have not been fully explored. The authors observe that in LoRA, A tends to learn the commonalities of data, while B focuses on the uniqueness of each intrinsic component—similar to human memory of faces, remembering the outline (commonalities) while ignoring details (such as nose width), whereas details are crucial for precise recognition.

## Method

### Overall Architecture

The core innovations of CoLA lie in two aspects: (1) a flexible multi-matrix architecture (#A=M, #B=N, where M and N are set independently); (2) three distinct matrix collaboration strategies. Additionally, the PiSSA initialization scheme is extended to the CoLA architecture.

### Key Designs

1. **Flexible LoRA Architecture**: Does not enforce a fixed quantity relationship between A and B (#A=M, #B=N), making existing LoRA architectures (vanilla LoRA: M=N=1; MOE-LoRA: M=N=K; HydraLoRA: M=1, N=K) special cases of CoLA. This flexibility allows the model to adjust its structure based on data characteristics and task requirements. The design motivation is to grant the model greater freedom to separately learn common and diverse knowledge.

2. **Extended PiSSA Initialization**: Performs SVD decomposition on the pre-trained weight matrix W, and evenly distributes the principal singular values and vectors to each $A_i$ and $B_j$: $A_i = \frac{U_{[:,:r]} S_{[:r,:r]}^{1/2}}{M}$, $B_j = \frac{S_{[:r,:r]}^{1/2} V_{[:,:r]}^T}{N}$. This allows each matrix to initially contain the primary directional information of the pre-trained weights. During fine-tuning, each matrix can optimize in different directions, enhancing generalization diversity. This is motivated by the Eckart-Young-Mirsky theorem—the initial BA contains the most important directions of W, helping to achieve faster and better convergence.

3. **Three Collaboration Strategies**:

    - **Full Collaboration CoLA⊺**: $\Delta W = (B_1 + \cdots + B_N)(A_1 + \cdots + A_M)$, where all A and B fully interact to share knowledge, breaking information transmission barriers, though with the highest computational overhead.
    - **Random Collaboration CoLA†**: Each A is paired with a randomly selected B, resembling the idea of dropout regularization. By not relying on specific combinations, knowledge learning becomes more robust, incurring the lowest computational overhead.
    - **Heuristic Collaboration CoLA‡**: $\Delta W = B_1A_1 + \cdots + B_{M-1}A_{M-1} + (B_M + \cdots + B_N)A_M$ (assuming M<N), combining the advantages of one-to-one and one-to-many relationships to balance general and diverse knowledge learning with moderate computational overhead.

### Loss & Training

- Trained using the LlamaFactory framework.
- Generative evaluation is uniformly converted into classification evaluation (where the model outputs only a single uppercase letter) to ensure fairness and reproducibility.
- Default LoRA rank=8, experiments are repeated 5 times under random seeds 42-46, and the average is reported.
- Out-of-domain/conflicting datasets (GSM8K/BBH) are standardized into multiple-choice formats using LLMs.

## Key Experimental Results

### Main Results (Single Domain, Llama-3.1-8B)

| Method | #A\|#B | Params% | General | Law | Medicine | Math | Finance |
|------|--------|---------|------|------|------|------|------|
| LoRA (r=8) | 1\|1 | 0.26% | 50.36 | 25.98 | 42.66 | 51.02 | 40.38 |
| PiSSA | 1\|1 | 0.26% | 54.72 | 26.58 | 44.64 | 57.00 | 46.79 |
| HydraLoRA | 1\|3 | 0.58% | 45.86 | 26.26 | 40.61 | 47.31 | 38.87 |
| **CoLA** | 1\|3 | 0.53% | **58.04** | **36.25** | **56.11** | **57.71** | **52.45** |
| **CoLA⊺** | 2\|3 | 0.66% | **58.21** | **41.46** | **54.33** | **59.14** | **50.19** |

CoLA and CoLA⊺ significantly outperform the baselines in all domains (p<0.01).

### Multi-Tasking Experiments

| Method | #A\|#B | Llama-3.2-3B | Llama-3.1-8B |
|------|--------|-------------|-------------|
| LoRA (r=64) | 1\|1 | 34.89 | 42.99 |
| MOELoRA | 8\|8 | 30.77 | 40.53 |
| HydraLoRA | 1\|14 | 29.64 | 39.08 |
| **CoLA** | 1\|14 | **36.87** | 42.87 |
| **CoLA⊺** | 4\|10 | 36.47 | **43.62** |

### Ablation Study

| Configuration | Key Findings | Description |
|------|----------|------|
| With/Without PiSSA Initialization | PiSSA has an extremely significant impact on CoLA | Especially evident when samples $\le 200$; without PiSSA, CoLA's performance drops sharply |
| #A vs #B Quantity Relationship | #A < #B achieves the best results | The benefit of increasing B is greater than increasing A |
| CoLA† vs CoLA†̂ | Random A is superior to random B | Validates the alignment/universality of the #A < #B principle |
| Energy consumption of three strategies | CoLA† < CoLA‡ < CoLA⊺ | Correspond to low, medium, and high computational budgets/configurations respectively |

### Key Findings

- **Observation 1**: CoLA is effective in both single-domain and multi-domain settings. HydraLoRA is prone to overfitting in low-sample scenarios due to Gaussian noise initialization.
- **Observation 2**: The impact of PiSSA initialization is more significant on CoLA than on LoRA, particularly as the sample size continues to decrease. This is attributed to the multi-matrix structure + PiSSA allowing both A and B to learn the base instruction patterns of the pre-trained model.
- **Observation 3**: The number of matrices A should be less than B—A learns commonalities of the data, while B focuses on the uniqueness of each component, with higher-level features receiving more weight.
- **Observation 4**: The energy consumption of the three strategies differs significantly, making them suitable for different resource-constrained scenarios. The total energy consumption in experiments is less than 1/10 of HydraLoRA.

## Highlights & Insights

- **Unified Framework Perspective**: Unifies vanilla LoRA, MOE-LoRA, HydraLoRA, etc., into the CoLA framework, clearly demonstrating that the intrinsic differences of different architectures lie in matrix size/quantities and collaboration modes.
- **Asymmetric Roles of A and B**: Systematically reveals the pattern of A learning commonalities and B learning diversity through experiments, providing vital design guidelines for subsequent LoRA variants.
- **Practical Energy-Performance Trade-off**: The three collaboration strategies offer flexible computational budget choices, meeting the requirements of different resource constraints in practical deployment.

## Limitations & Future Work

- Not validated on coding domains, because code generation tasks are difficult to convert into multiple-choice formats.
- The collaboration strategy space between A and B is far from being fully explored—graph theoretic properties such as maximum matching of bipartite graphs could bring superior strategies.
- Only experimented on the Llama series models, lacking verification on other architectures (e.g., Qwen, Mistral).
- Classification-based evaluation modes might underestimate performance discrepancies in generative tasks.

## Related Work & Insights

- The SVD initialization concept of PiSSA amplifies its advantages in CoLA's multi-matrix scenarios, indicating that a good initialization might be even more critical in more complex structures.
- The division of roles between A and B matrices aligns with hierarchical abstraction mechanisms in deep learning (classic works by LeCun, Hinton, etc.).
- Inspiration: Fine-tuning LoRA should not only focus on the rank size; the structural relationships and collaboration modes between matrices are equally, if not more, critical.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The flexible architecture and design space exploration of the three collaboration strategies are novel, unifying existing methods under the same framework.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Systematic and comprehensive analysis across 6 domains, 2 model scales, and 4 dimensions of Observations.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure with excellent summaries of the 4 observations, though massive formulas and notations increase reading difficulty.
- **Value**: ⭐⭐⭐⭐ — Highly instructive for designing LoRA variants, with practical advantages in low-sample scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] TeamLoRA: Boosting Low-Rank Adaptation with Expert Collaboration and Competition](teamlora_boosting_low-rank_adaptation_with_expert_collaboration_and_competition.md)
- [\[ACL 2025\] BeamLoRA: Beam-Constraint Low-Rank Adaptation](beamlora_beam_constraint_lora.md)
- [\[ACL 2025\] Low-Rank Interconnected Adaptation across Layers](low-rank_interconnected_adaptation_across_layers.md)
- [\[ACL 2025\] MoRE: A Mixture of Low-Rank Experts for Adaptive Multi-Task Learning](more_a_mixture_of_low-rank_experts_for_adaptive_multi-task_learning.md)
- [\[AAAI 2026\] Group Orthogonal Low-Rank Adaptation for RGB-T Tracking](../../AAAI2026/model_compression/group_orthogonal_low-rank_adaptation_for_rgb-t_tracking.md)

</div>

<!-- RELATED:END -->

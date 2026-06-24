---
title: >-
  [Paper Note] DRAE: Dynamic Retrieval-Augmented Expert Networks for Lifelong Learning and Task Adaptation in Robotics
description: >-
  [ACL 2025][Robotics][Lifelong learning] This paper proposes the DRAE framework, which integrates dynamic MoE routing, parametric RAG (P-RAG), a three-layer cognitive control architecture (ReflexNet-SchemaPlanner-HyperOptima), and DPMM lifelong knowledge retention. It achieves an average success rate of 82.5% on robotic manipulation and autonomous driving tasks, effectively mitigating catastrophic forgetting.
tags:
  - "ACL 2025"
  - "Robotics"
  - "Lifelong learning"
  - "MoE"
  - "RAG"
  - "Hierarchical RL"
  - "Catastrophic forgetting"
  - "DPMM"
date: 2026-05-08
content_hash: a619f078b7fae11b
---

# DRAE: Dynamic Retrieval-Augmented Expert Networks for Lifelong Learning and Task Adaptation in Robotics

**Conference**: ACL 2025  
**arXiv**: [2507.04661](https://arxiv.org/abs/2507.04661)  
**Code**: None  
**Area**: Robotics / Lifelong Learning  
**Keywords**: Lifelong learning, MoE, RAG, Hierarchical RL, Catastrophic forgetting, DPMM

## TL;DR
This paper proposes the DRAE framework, which integrates dynamic MoE routing, parametric RAG (P-RAG), a three-layer cognitive control architecture (ReflexNet-SchemaPlanner-HyperOptima), and DPMM lifelong knowledge retention. It achieves an average success rate of 82.5% on robotic manipulation and autonomous driving tasks, effectively mitigating catastrophic forgetting.

## Background & Motivation
**Background**: Lifelong learning (continual learning) is a core challenge for intelligent robotics. Reinforcement learning (RL) agents are prone to catastrophic forgetting of old skills when learning new tasks. Existing methods include EWC (Elastic Weight Consolidation), Progressive Neural Networks, and MoE dynamic routing.

**Limitations of Prior Work**: (a) EWC regularization scales poorly in dynamic environments; (b) Progressive Networks suffer from linear memory growth as the number of tasks increases; (c) Although MoE can dynamically allocate resources, it still faces challenges in long-term memory management and catastrophic forgetting; (d) RAG is effective in NLP but remains under-explored in robotic lifelong learning.

**Key Challenge**: How to efficiently learn new tasks without destroying prior knowledge while maintaining computational efficiency?

**Goal**: To build a unified framework that simultaneously addresses catastrophic forgetting, task adaptation, and knowledge reuse.

**Key Insight**: Inspired by human sensorimotor control, a three-layer cognitive architecture is designed, combining non-parametric Bayesian models to achieve adaptive knowledge expansion.

**Core Idea**: MoE dynamic routing + RAG external knowledge + three-layer hierarchical RL + DPMM non-parametric knowledge retention = Robotic lifelong learning.

## Method

### Overall Architecture
DRAE consists of four core components: (1) MoE dynamic routing, which selects the top-m experts based on the input; (2) P-RAG, which retrieves relevant knowledge from an external memory library to integrate into decision-making; (3) RSHO, a three-layer cognitive control system consisting of ReflexNet (reflexive execution layer), SchemaPlanner (symbolic planning layer), and HyperOptima (meta-optimization layer); and (4) DPMM, a Dirichlet Process Mixture Model-based non-parametric clustering that automatically creates new clusters for new tasks without overwriting old skills.

### Key Designs

1. **MoE Dynamic Expert Routing**:

    - Function: Selects and activates the top-m experts via softmax gating based on the input $\mathbf{x}_t$.
    - Mechanism: $g_k(\mathbf{x}_t) = \text{softmax}(\mathbf{w}_k^T \mathbf{x}_t + b_k)$, activating only a few experts to limit inference costs.
    - Design Motivation: Each expert can specialize in a specific task type, allowing new tasks to reuse or extend existing experts.

2. **P-RAG External Knowledge Fusion**:

    - Function: Encodes the input as a query vector, retrieves relevant documents from an external corpus $\mathcal{C}$, and integrates them into the hidden states via LoRA.
    - Mechanism: $\mathbf{h}_{rag} = \mathbf{W}_0 \mathbf{x}_t + \mathbf{B}_l \mathbf{A}_l \mathbf{x}_t \odot \sigma(\mathbf{U}_d \mathbf{d}_t)$, where a sparse constraint $\lambda|\mathcal{D}'|$ is used to control the size of the retrieved set during retrieval.
    - Design Motivation: External knowledge is not stored in model parameters; retrieval instead of memorization fundamentally avoids knowledge overwriting.

3. **Three-layer Cognitive Control Architecture (RSHO)**:

    - ReflexNet (Reflexive Layer): Adaptive PID control that translates observations into torque commands, with gains dynamically adjusted via meta-learning.
    - SchemaPlanner (Symbolic Planning Layer): Decomposes tasks using MCTS + neurosymbolic program synthesis, mapping symbolic primitives to ReflexNet skills.
    - HyperOptima (Meta-Optimization Layer): A hyperdimensional memory module that evaluates N candidate policies in parallel using cyclic convolution to select the optimal one for execution.
    - Design Motivation: Mimics the three-layer structure of human sensorimotor control (spinal reflex -> cortical planning -> metacognition) to achieve multi-timescale decision-making.

4. **DPMM Lifelong Knowledge Retention**:

    - Function: Uses a Dirichlet Process Mixture Model to perform task-level clustering, automatically creating a new cluster when a new task is sufficiently distinct.
    - Mechanism: $G \sim \text{DP}(\alpha, \mathcal{H})$, determining if a new expert is needed based on KL divergence: $\mathbb{P}(\text{new expert}) = 1$ if $\min_k D_{KL}(p(z_t) \| p(\theta_k)) > \tau$.
    - Design Motivation: Non-parametric models automatically decide when to expand and when to reuse, without requiring a pre-specified number of tasks.

### Loss & Training
Unified objective: $\mathcal{L}_{total} = \mathcal{L}_{HRL} + \alpha(\mathcal{L}_{MoE} + \mathcal{L}_{P-RAG}) + \gamma(\mathcal{L}_{HyperOptima} + \mathcal{L}_{DPMM})$, where $\alpha, \gamma$ are adaptively adjusted.

## Key Experimental Results

### Main Results

| Method | MimicGen Avg Success Rate | DiffusionDrive EP | Total Params | Active Params |
|------|-------------------|-------------------|--------|---------|
| DRAE | **0.78** | **82.5** | 190.1M | 42.3M |
| SDP | 0.76 | - | 126.9M | 53.3M |
| TH/TT | 0.73 | - | 52.6-144.7M | 52.6M |
| DRAMA | - | 80.1 | - | - |

### Ablation Study

| Component | MimicGen Avg | Description |
|------|-------------|------|
| Full DRAE | 0.78 | All four components enabled |
| w/o P-RAG | ~0.74 | Removal of retrieval augmentation |
| w/o DPMM | ~0.72 | Removal of lifelong knowledge retention |
| Static MoE | 0.742 | Static MoQ baseline |

### Key Findings
- **Dynamic expansion + retrieval augmentation are core**: DRAE achieves a 4.3pp improvement over static MoE and also outperforms domain-specific SOTA methods.
- **Extremely low forgetting rate**: Maintains stable performance in long-term testing on the NavSim autonomous driving benchmark (EP=82.5, PDMS=88.0).
- **Inference efficiency**: Total parameters are 190M but active parameters are only 42.3M, requiring less active computation compared to baselines.
- **Theoretical guarantees**: Proves a sublinear dynamic regret bound of $\mathcal{O}(\sqrt{T(1+P_T)})$.

## Highlights & Insights
- **Four-in-one framework**: Although the combination of MoE+RAG+HRL+DPMM is complex, each component has a clear division of labor, and the overall integration is well-designed. Utilizing DPMM as the ultimate defense against catastrophic forgetting is a key highlight.
- **Three-layer cognitive architecture**: The neuroscience-inspired design of ReflexNet-SchemaPlanner-HyperOptima is creative; the hierarchy from reflex to planning to metacognition corresponds to decision-making needs across different timescales.
- **Scalable knowledge base**: The non-parametric nature of DPMM allows the knowledge base to automatically expand over time without requiring prior knowledge of the number or distribution of tasks.

## Limitations & Future Work
- The system is highly complex, featuring four main components and a three-layer cognitive architecture, making training and hyperparameter tuning difficult.
- Ablation studies are not sufficiently detailed, lacking a systematic quantification of each component's contribution.
- Experiments are mainly conducted in simulation environments (MimicGen, NavSim), lacking sufficient real-robot validation.
- How to set the concentration parameter $\alpha$ and threshold $\tau$ of DPMM is not fully discussed.

## Related Work & Insights
- **vs. EWC/MAS**: Regularization methods penalize parameter changes but scale poorly; DRAE completely avoids overwriting through non-parametric expansion.
- **vs. Progressive Networks**: Column-wise expansion leads to linear memory growth, whereas DRAE controls computational cost via MoE sparse activation.
- **vs. RAG in NLP**: RAG in NLP is mainly used for factual retrieval, while this work is the first to systematically introduce it to robotic lifelong learning.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Four-in-one architecture + three-layer cognitive control + theoretical guarantees, offering multiple novelties.
- Experimental Thoroughness: ⭐⭐⭐ Dual-domain validation (robotics + autonomous driving), but ablation studies are not sufficiently detailed.
- Writing Quality: ⭐⭐⭐ Math-heavy; the system description is comprehensive but readability is average.
- Value: ⭐⭐⭐⭐ A complete solution for lifelong learning, with an inspiring framework design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Lifelong Embodied Navigation Learning](../../ICLR2026/robotics/lifelong_embodied_navigation_learning.md)
- [\[ACL 2025\] Task-aware MoILE: Hierarchical-Task-Aware Multi-modal Mixture of Incremental LoRA Experts for Embodied Continual Learning](hierarchical-task-aware_multi-modal_mixture_of_incremental_lora_experts_for_embo.md)
- [\[ICLR 2026\] VER: Vision Expert Transformer for Robot Learning via Foundation Distillation and Dynamic Routing](../../ICLR2026/robotics/ver_vision_expert_transformer_for_robot_learning_via_foundation_distillation_and.md)
- [\[CVPR 2025\] Think Small, Act Big: Primitive Prompt Learning for Lifelong Robot Manipulation](../../CVPR2025/robotics/think_small_act_big_primitive_prompt_learning_for_lifelong_robot_manipulation.md)
- [\[NeurIPS 2025\] Task-Optimized Convolutional Recurrent Networks Align with Tactile Processing in the Rodent Brain](../../NeurIPS2025/robotics/task-optimized_convolutional_recurrent_networks_align_with_tactile_processing_in.md)

</div>

<!-- RELATED:END -->

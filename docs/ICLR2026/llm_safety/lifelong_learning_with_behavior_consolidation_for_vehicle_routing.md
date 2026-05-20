---
title: >-
  [Paper Note] Lifelong Learning with Behavior Consolidation for Vehicle Routing
description: >-
  [ICLR 2026][LLM Safety][Lifelong Learning] This paper proposes LLR-BC, a framework for lifelong learning in neural VRP solvers. By combining decision-step-level experience buffers…
tags:
  - "ICLR 2026"
  - "LLM Safety"
  - "Lifelong Learning"
  - "Vehicle Routing Problem"
  - "Catastrophic Forgetting"
  - "Experience Replay"
  - "Behavior Consolidation"
date: 2026-05-08
content_hash: 4397eb04165895b0
---

# Lifelong Learning with Behavior Consolidation for Vehicle Routing

**Conference**: ICLR 2026
**arXiv**: [2509.21765](https://arxiv.org/abs/2509.21765)  
**Code**: [github](https://github.com/PeiJY/LLR-BC)  
**Area**: LLM Safety
**Keywords**: Lifelong Learning, Vehicle Routing Problem, Catastrophic Forgetting, Experience Replay, Behavior Consolidation

## TL;DR

This paper proposes LLR-BC, a framework for lifelong learning in neural VRP solvers. By combining decision-step-level experience buffers, Confidence-aware Experience Weighting (CaEW), and Decision-seeking Behavior Consolidation via reverse KL divergence (DsBC), LLR-BC reduces the Average Performance gap (AP) by an order of magnitude on task sequences with simultaneously shifting distributions and scales, while preserving plasticity for new tasks and improving zero-shot generalization.

## Background & Motivation

**Background**: Neural combinatorial optimization solvers (e.g., POMO, INViT) learn VRP solution policies directly via deep reinforcement learning and have achieved performance competitive with classical heuristics (e.g., LKH3) on fixed-distribution, fixed-scale tasks. The dominant training paradigm is one-shot training on predefined tasks.

**Limitations of Prior Work**: In practice, logistics scenarios exhibit continuously shifting order distributions and scales—new delivery patterns and customer groups of varying sizes emerge over time. One-shot training cannot cover all future scenarios. Direct fine-tuning on new tasks causes catastrophic forgetting, leading to drastic performance degradation on previously learned tasks. Zero-shot generalization can partially mitigate this but has an upper bound, remaining insufficient when new tasks differ substantially from the training distribution.

**Key Challenge**: There is a fundamental conflict between *plasticity*—the ability to rapidly adapt to new tasks—and *stability*—the ability to retain knowledge from previous tasks. The two existing VRP lifelong learning works (Li et al. 2024, Feng et al. 2025) are limited to highly restricted settings: tasks vary only in scale or distance metric, task order is known and fixed, and old task instances can be actively generated for retraining. These assumptions do not hold in real-world scenarios.

**Goal**: (1) A general lifelong learning setting where both distribution and scale change simultaneously; (2) unknown task order and uncontrolled instance generation; (3) sustained high performance throughout the learning process, not only at the terminal state.

**Key Insight**: The authors observe that constructive VRP solvers make decisions sequentially—selecting the next node to visit at each step—such that small probability shifts can alter decisions and drastically degrade solution quality. Preserving old behavior thus requires retaining the probability distributions at critical decision steps, especially those with low confidence (i.e., easily perturbed), rather than preserving complete instance solutions.

**Core Idea**: A decision-step-level experience buffer combined with mode-seeking behavior consolidation via reverse KL divergence effectively resists catastrophic forgetting at extremely low memory overhead (0.01% of training experience).

## Method

### Overall Architecture

LLR-BC follows the experience replay paradigm. A fixed-size experience buffer $\mathcal{B}$ is maintained. When a new task arrives, at each training epoch: (1) a batch of instances is sampled and solved from the current task to obtain experience trajectories $\{\tau\}$; (2) the solver is updated via a DRL algorithm using $\{\tau\}$; (3) old experiences $\mathcal{E}$ are sampled from the buffer, weighted by CaEW, and used to compute the behavior consolidation loss via DsBC; (4) the new-task DRL loss and the behavior consolidation loss are jointly optimized. The framework is agnostic to model architecture and RL algorithm, and can be directly integrated into existing solvers such as POMO, Omni, and INViT.

### Key Designs

1. **Decision-Step-Level Experience Representation and Reservoir Buffer**:

    - Function: Captures the solver's "behavioral memory" at minimal granularity; manages large-scale experience efficiently with a fixed-size buffer.
    - Mechanism: Each experience is defined as $e = \langle s, \mathcal{P} \rangle$, where $s$ is the current partial solution state (sequence of visited nodes) and $\mathcal{P}$ is the solver's complete probability distribution over all candidate nodes at that state. Compared to existing methods that treat an entire instance as one experience unit, step-level representation achieves higher information density and more compact storage. The buffer employs reservoir sampling: new experiences replace existing ones with probability $|\mathcal{B}|/N$, ensuring all historical experiences are retained with equal probability. Experiences are collected only during the last epoch of each task—when the solver is fully trained and behavioral quality is highest. The entire buffer occupies approximately 0.01% of total training experience.
    - Design Motivation: Instance-level buffers suffer from dimensionality inconsistency when scales vary and contain redundant information. Step-level representation naturally accommodates tasks of different scales, and probability distributions carry richer policy information than single actions.

2. **Confidence-aware Experience Weighting (CaEW)**:

    - Function: Differentiates the importance of experiences during consolidation, directing the model's attention toward critical decision points.
    - Mechanism: Decision confidence is measured by the variance of the probability distribution. Low variance indicates that the model assigns similar probability to all candidate nodes—such "hesitant" states are most susceptible to perturbation by new-task training. The weight is defined as $w(e) = 1 - \text{var}(\mathcal{P}) / \text{var}_{\max}(|\mathcal{P}|)$, where $\text{var}_{\max}(n) = (n-1)/n^2$ is the maximum possible variance for $n$ candidates. Weights are normalized to sum to 1 within each sampled set.
    - Design Motivation: Sequential decisions in VRP exhibit a cascade effect—an incorrect choice at a critical branching point propagates to all subsequent steps. Low-confidence decisions are the hallmark of such critical junctures.

3. **Decision-seeking Behavior Consolidation (DsBC)**:

    - Function: Constrains the current model's behavior on old states to remain aligned with the buffered historical behavior, with emphasis on preserving the core decision of which node to select.
    - Mechanism: Conventional knowledge distillation uses forward KL divergence ($D_{KL}(P \| Q)$), which encourages the learner to spread probability mass across all modes of the teacher, potentially diluting focus. LLR-BC instead employs reverse KL divergence (RKLD) $D_{KL}(Q \| P)$, whose mode-seeking property concentrates the learner on reproducing the teacher's highest-probability actions—precisely the decisions executed during greedy decoding in VRP solvers. The consolidation loss is $\mathcal{L}_{BC} = \sum_{e \in \mathcal{E}} \bar{w}(e) \sum_{a} \mathcal{P}_\theta(a) \log \frac{\mathcal{P}_\theta(a)}{\mathcal{P}(a)}$.
    - Design Motivation: Constructive VRP solvers select the highest-probability node at inference time, making it more important to preserve top-1 decisions than to uniformly align the entire distribution. RKLD naturally emphasizes peak alignment.

### Loss & Training

The total loss is $\mathcal{L} = \mathcal{L}_{DRL} + \alpha \cdot \mathcal{L}_{BC}$, where $\mathcal{L}_{DRL}$ is the policy gradient loss of the underlying DRL algorithm (e.g., REINFORCE) and $\alpha = 100$ balances new-task learning against old-behavior consolidation. Each task is trained for 200 epochs; buffer size $|\mathcal{B}| = 1000$ (batch-level); $|\mathcal{E}| = 16$ batches of old experience are sampled per step. Because LLR-BC operates in behavior space rather than parameter space, it does not accumulate regularization constraints as tasks increase—unlike EWC—and thus does not progressively lose plasticity.

## Key Experimental Results

### Main Results

Six tasks are constructed on CVRP and TSP (6 distributions × 3 scales), averaged over 5 random task orderings. All metrics ×$10^{-3}$, lower is better.

| Method | CVRP AP↓ | CVRP AF↓ | CVRP APl↓ | TSP AP↓ | TSP AF↓ | TSP APl↓ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Fine-tuning | 23.5 | 19.9 | 3.8 | 14.8 | 28.9 | 3.5 |
| EWC | 28.3 | 19.5 | 6.9 | 18.3 | 18.6 | 5.5 |
| LiBOG | 31.3 | 19.7 | 7.2 | 19.2 | 17.2 | 5.8 |
| Feng | 24.6 | 3.2 | 24.2 | 24.1 | 1.8 | 21.4 |
| Li (inter) | 32.0 | 0.0 | 33.6 | 56.5 | 0.4 | 61.7 |
| Restart | 60.5 | 41.3 | 9.1 | 31.7 | 50.5 | 7.1 |
| **LLR-BC** | **4.2** | **0.7** | **3.5** | **3.4** | **0.8** | **2.8** |

LLR-BC's AP is an order of magnitude lower than all baselines (CVRP: 4.2 vs. 23.5+; TSP: 3.4 vs. 14.8+), with near-zero AF (virtually no forgetting) and the best APl (fastest learning on new tasks). Although Li (inter/intra) achieves low forgetting, it does so at the cost of allocating half the training budget to retraining old task instances, resulting in poor new-task performance (APl).

### Ablation Study

Component ablation on task order 1 (×$10^{-3}$):

| Variant | CVRP AP | CVRP AF | CVRP AMF | TSP AP | TSP AF |
|:---|:---:|:---:|:---:|:---:|:---:|
| LLR-BC (default) | 4.9 | 0.6 | 0.7 | 1.7 | 0.8 |
| w/o CaEW (uniform) | 5.2 | 0.8 | 0.8 | 1.8 | 0.9 |
| KLD instead of RKLD | 5.5 | 0.7 | 0.7 | 1.9 | 0.9 |
| Buffer every epoch | 7.8 | 3.1 | 3.1 | 2.6 | 2.1 |
| Instance-level buffer (-IB) | 35.4 | 23.4 | 27.2 | 2.5 | 1.8 |
| Entropy instead of Var | 4.8 | 0.5 | 0.7 | 2.1 | 0.9 |
| Scaled reservoir prob. (-Res) | 4.9 | 0.9 | 0.9 | 2.0 | 1.0 |

The most critical finding is the large gap between step-level and instance-level experience representation—instance-level buffering degrades CVRP AP from 4.9 to 35.4 (7× worse). Buffering only at the last epoch is also important (buffering every epoch raises AF from 0.6 to 3.1). CaEW and RKLD each contribute consistent improvements, while the specific form of the confidence measure (Var/Entropy/Top2-Margin) has little sensitivity.

### Key Findings

- **Significant improvement in zero-shot generalization**: On TSPLIB (scales up to 1001), LLR-BC achieves 18.08 vs. Fine-tuning's 38.16; on CVRPLIB, 7.88 vs. 8.54. Cross-task knowledge accumulated during lifelong learning genuinely enhances generalization to unseen tasks.
- **Universality across solvers**: Integrating LLR-BC into Omni and INViT reduces CVRP AP from 34.7→16.5 and 28.6→23.8, respectively, with consistent patterns.
- **Behavior-space consolidation does not impair plasticity**: Unlike EWC, which imposes regularization in parameter space, LLR-BC allows parameters to change freely while only constraining output behavior alignment, so constraints do not accumulate as tasks increase. In experiments, LLR-BC learns new tasks even faster than Fine-tuning.
- **Robustness to hyperparameters**: With $\alpha$ ranging from 10 to 1000, $|\mathcal{B}|$ from 250 to 1000, and $|\mathcal{E}|$ from 4 to 16, performance variation is far smaller than the gap with baselines.

## Highlights & Insights

- **Step-level experience representation is the core contribution**: Decomposing "a complete solution for an instance" into "state + probability distribution at each step" as the experience unit naturally accommodates tasks of varying scales, achieves high information density, and incurs negligible storage overhead. This design philosophy generalizes to all autoregressive sequential decision lifelong learning scenarios (e.g., scheduling, assignment problems).
- **Clever application of reverse KL**: RKLD is uncommon in knowledge distillation, but in the greedy decoding setting of VRP, its mode-seeking property precisely matches the requirement of preserving top-1 decisions. The insight that "the distillation objective should match the downstream inference mechanism" is worth borrowing in other greedy/beam search settings.
- **Extremely low memory overhead**: The buffer occupies only 0.01% of training experience, yet yields substantial performance gains, demonstrating that "choosing what to store wisely" matters more than "storing more."

## Limitations & Future Work

- **Fixed $|\mathcal{E}|$ is unbalanced across task scales**: For small-scale tasks with few new experiences, old experiences constitute a disproportionately large fraction of the batch, potentially suppressing plasticity; the opposite holds for large-scale tasks. Adaptive sampling ratios are a promising direction.
- **Only same-type VRP lifelong learning is validated**: Lifelong learning across problem types (e.g., TSP→CVRP→VRPTW) remains unexplored and may require task-specific model components.
- **Task boundary assumption**: The framework still assumes that task boundaries are known. Continuously drifting distributions are not addressed, though per-instance reservoir sampling is mentioned as a potential mitigation.
- **Constructive solvers only**: The lifelong learning behavior of improvement-based solvers (e.g., neural selectors in LKH-based methods) may differ and is not investigated.

## Related Work & Insights

- **vs. EWC (parameter regularization)**: EWC constrains important parameters in parameter space; accumulated regularization terms across tasks progressively erode plasticity. LLR-BC constrains behavior in output space while leaving parameters free to explore, maintaining plasticity throughout.
- **vs. Li et al. / Feng et al. (existing VRP lifelong learning)**: These methods rely on the assumption that old task instances can be actively generated, and allocate half the training budget to retraining on old tasks. LLR-BC requires only 0.01% of historical experience snippets without generating new instances, making it substantially more general.
- **vs. DER++ (general continual learning with experience replay)**: DER++ also stores probability distributions, but LLR-BC adds CaEW weighting and RKLD—two design choices tailored to the greedy decision setting—achieving superior effectiveness for VRP.
- This work demonstrates that behavior-level (rather than parameter-level or data-level) knowledge retention in sequential decision lifelong learning can simultaneously resolve the stability–plasticity conflict, offering insights for continual learning in LLM agents.

## Rating

- Novelty: ⭐⭐⭐⭐ First to extend lifelong learning to a general VRP setting with simultaneous distribution and scale shifts; the combination of step-level experience representation and RKLD consolidation is insightful.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 5 task orderings × 2 problem types × 7 baselines × 5 metrics; comprehensive ablations, cross-solver validation, and detailed hyperparameter sensitivity analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, naturally motivated methodology, rigorous terminology.
- Value: ⭐⭐⭐⭐ The framework is highly generalizable; the efficiency ratio of achieving order-of-magnitude performance gains at 0.01% memory overhead is impressive, though the VRP application domain is relatively niche.

### Key Findings

1. **Forgetting nearly eliminated**: AF approaches zero vs. Fine-tuning's 19.9—a 20× gap.
2. **Plasticity unimpaired**: Behavior-level consolidation does not accumulate constraints unlike parameter regularization.
3. **Universal across solvers**: Effective on all three base solvers—POMO, Omni, and INViT.
4. **Hyperparameter robustness**: Variation under changes in $|\mathcal{B}|$, $|\mathcal{E}|$, and $\alpha$ over wide ranges is far smaller than the gap with baselines.
5. **RKLD outperforms KLD**: Mode-seeking is better suited for preserving critical decisions (AP 4.9 vs. 5.5).

## Highlights & Insights

- 💡 **Fine-grained behavior replay**: Step-level representation with complete probability distributions achieves high information density.
- 🎯 **RKLD behavior consolidation**: Mode-seeking suitability is rigorously motivated by the characteristics of VRP decision-making.
- 🔄 **Extremely low storage overhead**: Buffer occupies only 0.01% of total experience.
- 📐 **Five-dimensional evaluation protocol**: AP/AF/AMF/APl/AG provides comprehensive assessment of lifelong learning.

## Limitations & Future Work

1. **Task boundary assumption**: Task boundaries are still assumed to be known.
2. **Fixed experience sampling volume**: May become imbalanced when task scales differ significantly.
3. **Limited cross-problem validation**: Only CVRP and TSP are evaluated.
4. **Experience recency not considered**: Reservoir sampling retains experiences with equal probability regardless of recency.
5. **Constructive solvers only**: Not extended to improvement-based solvers.

## Related Work & Insights

| Method | Type | Distinction |
|:---:|:---:|:---:|
| Li et al., 2024 | VRP Lifelong Learning | Allocates half resources to old instances; requires controllable generation |
| Feng et al., 2025 | VRP Lifelong Learning | Limited to scale/distance metric variation |
| EWC | Regularization | Parameter-level constraints cause plasticity degradation |
| Fine-tuning | Baseline | No forgetting protection |
| AMDKD | Knowledge Distillation | One-shot training only |
| Omni | Meta-learning | Does not maintain old-task performance |

Core insight: In VRP lifelong learning, behavior-level consolidation is more effective than parameter-level regularization or instance-level replay—the discrete combinatorial nature of the decision space demands preservation of highest-probability decisions.

## Rating

| Dimension | Score |
|:---:|:---:|
| Novelty | ⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Practical Value | ⭐⭐⭐⭐ |
| Overall | ⭐⭐⭐⭐ |

> A solid contribution at the intersection of lifelong learning and combinatorial optimization. CaEW and DsBC are well-motivated and carefully designed; experiments are extremely thorough (5 orderings, 3 solvers, multi-dimensional evaluation); performance surpasses baselines by an order of magnitude.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Elastic Weight Consolidation Done Right for Continual Learning](../../CVPR2026/llm_safety/elastic_weight_consolidation_done_right_for_continual_learning.md)
- [\[CVPR 2026\] Association and Consolidation: Evolutionary Memory-Enhanced Incremental Multi-View Clustering](../../CVPR2026/llm_safety/association_and_consolidation_evolutionary_memory-enhanced_incremental_multi-vie.md)
- [\[NeurIPS 2025\] Angular Steering: Behavior Control via Rotation in Activation Space](../../NeurIPS2025/llm_safety/angular_steering_behavior_control_via_rotation_in_activation_space.md)
- [\[AAAI 2026\] Attention Retention for Continual Learning with Vision Transformers](../../AAAI2026/llm_safety/attention_retention_for_continual_learning_with_vision_transformers.md)
- [\[AAAI 2026\] PRISM: Privacy-Aware Routing for Adaptive Cloud-Edge LLM Inference via Semantic Sketch Collaboration](../../AAAI2026/llm_safety/prism_privacy-aware_routing_for_adaptive_cloud-edge_llm_inference_via_semantic_s.md)

</div>

<!-- RELATED:END -->

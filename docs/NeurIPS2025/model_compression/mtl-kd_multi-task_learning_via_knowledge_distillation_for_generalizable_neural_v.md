---
title: >-
  [Paper Note] MTL-KD: Multi-Task Learning Via Knowledge Distillation for Generalizable Neural Vehicle Routing Solver
description: >-
  [NeurIPS 2025][Model Compression][Vehicle Routing Problem] This paper proposes MTL-KD, a multi-task learning framework based on knowledge distillation. It distills policy knowledge from multiple RL single-task teacher mo…
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "Vehicle Routing Problem"
  - "Multi-Task Learning"
  - "Knowledge Distillation"
  - "Neural Combinatorial Optimization"
  - "Scale Generalization"
date: 2026-05-08
content_hash: 3e3cc5cc6787c8c8
---

# MTL-KD: Multi-Task Learning Via Knowledge Distillation for Generalizable Neural Vehicle Routing Solver

**Conference**: NeurIPS 2025
**arXiv**: [2506.02935](https://arxiv.org/abs/2506.02935)  
**Code**: [GitHub](https://github.com/CIAM-Group/MTLKD)  
**Area**: Reinforcement Learning
**Keywords**: Vehicle Routing Problem, Multi-Task Learning, Knowledge Distillation, Neural Combinatorial Optimization, Scale Generalization

## TL;DR

This paper proposes MTL-KD, a multi-task learning framework based on knowledge distillation. It distills policy knowledge from multiple RL single-task teacher models into a heavy-decoder student model, achieving efficient unified solving across diverse VRP variants with superior generalization on large-scale instances.

## Background & Motivation

The Vehicle Routing Problem (VRP) is a classical NP-hard combinatorial optimization problem with broad applications in logistics and transportation scheduling. Neural combinatorial optimization (NCO) methods, which learn solution policies via neural networks, have emerged as a promising approach to VRP solving.

**The dilemma facing existing methods**:

**Heavy-Encoder-Light-Decoder (HELD) architecture**: Most unified multi-task models adopt this design, enabling direct RL training with strong performance on small-scale problems. However, the light decoder fails to extract sufficient information from dense, complex node embeddings at large scales, leading to significant degradation in generalization.

**Light-Encoder-Heavy-Decoder architecture**: This design demonstrates excellent scale generalization on single-task VRPs, as the heavy decoder re-evaluates relationships among remaining nodes at each decoding step. However, its substantial memory and computational demands render RL training infeasible, while supervised learning (SL) faces a scarcity of labeled data across diverse VRP variants.

**Key Challenge**: Achieving scale generalization in multi-task VRP solving requires a heavy-decoder architecture, yet training such a decoder in the multi-task setting is extremely challenging — RL training is prohibitively expensive, and SL lacks labeled data.

**Key Insight**: The paper bypasses these training difficulties via knowledge distillation. Lightweight single-task teacher models are first trained independently with RL, and their policy knowledge is then transferred label-free to a heavy-decoder student model via KL-divergence distillation.

## Method

### Overall Architecture

MTL-KD proceeds in two stages:
- **Stage 1**: For each seen VRP task, an independent single-task teacher model with a POMO-style architecture (heavy encoder, light decoder) is trained via RL.
- **Stage 2**: A multi-task heavy-decoder student model is constructed and trained via knowledge distillation, using the output distributions of all teacher models as supervision signals.

### Key Designs

1. **Teacher Model Training**: Each teacher adopts the POMO architecture (6-layer encoder + 1-layer decoder) and is trained independently via policy gradient RL. The loss is:

$$\mathcal{L}(\theta^T) = -\mathbb{E}_{\tau \sim \pi_{\theta^T}(\cdot|\mathcal{G})} \left[(r(\tau) - b^T(\mathcal{G})) \log \pi_{\theta^T}(\tau|\mathcal{G})\right]$$

where $b^T(\mathcal{G})$ is the average-reward baseline over multi-start trajectories and $r$ is the negative tour length (reward). Teachers are trained for 4,000 epochs on random instances of size 100.

2. **Student Model Architecture (Multi-Task Heavy Decoder)**: The student employs a 1-layer encoder and 6-layer decoder. The encoder maps node features (coordinates, demands, time windows, etc.) to initial embeddings via a linear layer, then generates node embeddings $H^{enc}$ through a Transformer layer. At each decoding step, the decoder extracts dynamic features (remaining capacity, current time, remaining distance, open-route flag), combines them with embeddings of the last visited node and depot, updates node embeddings through $L$ Transformer layers, and handles variable numbers of unvisited nodes across batches via padding masks. The final selection probability is:

$$\pi(i|s_t) = \text{softmax}(c(h_i^{(L)}, h_q) + M^{pad}_i + M^{feas}_i)$$

where $M^{pad}$ excludes padding positions and $M^{feas}$ excludes infeasible actions.

3. **Knowledge Distillation Training**: The student simultaneously processes instances from all $M$ seen tasks. At each decoding step $t$, the distillation loss is the KL divergence between the student's distribution and the corresponding teacher's distribution:

$$\mathcal{L}_{KD}^{(t)} = \sum_{m=1}^{M} \text{KL}\left(\pi_{\theta^{T_m}}(a_t|s_t, \mathcal{G}) \| \pi_{\theta^S}(a_t|s_t, \mathcal{G})\right)$$

This design enables label-free training — no optimal solution annotations are required; the student simply imitates the teacher's action distribution at each decoding step.

4. **R3C Inference Strategy (Random Reordering Re-Construction)**: Given an initial solution at inference time, R3C decomposes it into sub-routes, randomly shuffles their external ordering, and then randomly samples a contiguous segment for re-optimization. Compared to the prior RRC method, R3C avoids time-window violations by shuffling sub-route order rather than reversing sub-routes, while permitting more diverse sub-route combinations to escape local optima.

### Training Configuration

- Teacher models: 6-layer encoder, 1-layer decoder, embedding dimension 128, 8-head attention, 4,000 training epochs
- Student model: 1-layer encoder, 6-layer decoder, embedding dimension 128 (or 96), batch size 1,500 (250 × 6 tasks), 850 training epochs with learning rate halved every 100 epochs after epoch 300

## Key Experimental Results

### Main Results: Performance on Seen Tasks (6 VRP Variants)

| Method | CVRP n=100 | CVRP n=500 | CVRP n=1k | VRPTW n=100 | VRPTW n=1k |
|--------|-----------|-----------|----------|------------|-----------|
| HGS-PyVRP | 15.53* | 62.07* | 119.54* | 24.35* | 166.47* |
| MT-POMO(M+aug8) | 15.79(1.69%) | 67.99(9.54%) | 136.62(14.28%) | 25.61(5.18%) | 229.82(38.06%) |
| MVMoE(M+aug8) | 15.76(1.50%) | 73.61(18.59%) | 176.40(47.57%) | 25.51(4.78%) | 253.35(52.19%) |
| RF-MVMoE(M+aug8) | 15.84(2.01%) | 67.36(8.52%) | 134.85(12.80%) | 26.29(7.98%) | 187.87(12.86%) |
| **MTL-KD(R3C200)** | **15.76(1.48%)** | **63.63(2.51%)** | **122.06(2.10%)** | **25.31(3.93%)** | **181.85(9.24%)** |

### Zero-Shot Generalization on Unseen Tasks (10 Unseen VRP Variants)

| Method | OVRPL n=500 | OVRPL n=1k | VRPLTW n=500 | VRPLTW n=1k |
|--------|-----------|----------|-------------|-----------|
| MT-POMO(M+aug8) | 41.28(18.95%) | 84.40(29.08%) | 116.62(26.95%) | 235.94(34.99%) |
| MVMoE(M+aug8) | 45.58(31.36%) | 116.71(78.48%) | 118.00(28.45%) | 260.07(48.79%) |
| **MTL-KD(R3C200)** | **37.27(7.41%)** | **71.35(9.12%)** | **97.85(6.51%)** | **189.02(8.14%)** |

### Key Findings

1. **Significant scale generalization advantage**: MTL-KD's superiority is most pronounced on large-scale instances (n=500, 1000). For example, on CVRP n=1k, MTL-KD achieves a gap of only 2.10%, compared to 12.80% for RF-MVMoE and 47.57% for MVMoE.
2. **Strong unseen-task generalization**: Across 10 VRP variants never encountered during training, MTL-KD demonstrates substantially better zero-shot generalization than other multi-task methods. On OVRPB n=1k, MTL-KD even surpasses the traditional heuristic OR-Tools (−4.48% vs. baseline).
3. **R3C strategy is effective**: The R3C inference strategy consistently improves performance across all tasks and scales, with particularly notable gains on large-scale problems.
4. **CaDA collapses**: CaDA exhibits severe performance degradation on large-scale problems (gap exceeding 300%), underscoring the fundamental limitation of light-decoder architectures for scale generalization.

## Highlights & Insights

- The method elegantly combines RL (for teacher training) and knowledge distillation (for student training), avoiding the prohibitive computational cost of directly training a heavy decoder with RL.
- A "divide-then-merge" training strategy: single-task teachers specialize in their respective tasks, while the student simultaneously absorbs policy knowledge across all tasks through distillation.
- The R3C inference strategy enhances sampling diversity via sub-route reordering, offering a general and constraint-friendly improvement applicable to various VRP formulations.

## Limitations & Future Work

- Multiple teacher models must be pre-trained, making the overall training pipeline relatively complex.
- Validation is currently limited to 16 VRP variants; more complex real-world constraint combinations remain unexplored.
- The student's single-layer encoder may have insufficient modeling capacity for certain complex constraints.

## Related Work & Insights

- Compared to works such as RouteFinder, MTL-KD is the first to introduce knowledge distillation for training heavy-decoder models in the multi-task VRP setting, establishing a novel training paradigm.
- For other combinatorial optimization problems requiring scale generalization, a similar "RL teacher → KD student" framework may prove equally effective.

## Rating

- Novelty: ⭐⭐⭐⭐ The idea of training a heavy decoder via knowledge distillation is novel, though the overall framework is largely combinatorial
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across 16 VRP variants, 4 scales, and multiple baselines
- Writing Quality: ⭐⭐⭐⭐ Clear logic and rich figures
- Value: ⭐⭐⭐⭐ Practically significant for multi-task VRP solving with substantial gains in scale generalization

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Multi-Task Vehicle Routing Solver via Mixture of Specialized Experts under State-Decomposable MDP](multi-task_vehicle_routing_solver_via_mixture_of_specialized_experts_under_state.md)
- [\[ICCV 2025\] EA-KD: Entropy-based Adaptive Knowledge Distillation](../../ICCV2025/model_compression/ea-kd_entropy-based_adaptive_knowledge_distillation.md)
- [\[NeurIPS 2025\] Single-Teacher View Augmentation: Boosting Knowledge Distillation via Angular Diversity](single-teacher_view_augmentation_boosting_knowledge_distillation_via_angular_div.md)
- [\[NeurIPS 2025\] Few-Shot Knowledge Distillation of LLMs With Counterfactual Explanations](few-shot_knowledge_distillation_of_llms_with_counterfactual_explanations.md)
- [\[NeurIPS 2025\] Why Knowledge Distillation Works in Generative Models: A Minimal Working Explanation](why_knowledge_distillation_works_in_generative_models_a_minimal_working_explanat.md)

</div>

<!-- RELATED:END -->

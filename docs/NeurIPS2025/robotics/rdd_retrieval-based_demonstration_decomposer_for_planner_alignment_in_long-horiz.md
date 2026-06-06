---
title: >-
  [Paper Note] RDD: Retrieval-Based Demonstration Decomposer for Planner Alignment in Long-Horizon Tasks
description: >-
  [NeurIPS 2025][Robotics][Hierarchical VLA] This paper proposes RDD (Retrieval-Based Demonstration Decomposer), which models demonstration decomposition as an optimal partition problem and automatically segments long-hori…
tags:
  - "NeurIPS 2025"
  - "Robotics"
  - "Hierarchical VLA"
  - "Task Decomposition"
  - "Retrieval-Based Alignment"
  - "Dynamic Programming"
  - "Long-Horizon Manipulation"
date: 2026-05-08
content_hash: 133f22411387568a
---

# RDD: Retrieval-Based Demonstration Decomposer for Planner Alignment in Long-Horizon Tasks

**Conference**: NeurIPS 2025
**arXiv**: [2510.14968](https://arxiv.org/abs/2510.14968)  
**Code**: [rdd-neurips.github.io](https://rdd-neurips.github.io)  
**Area**: Robotics
**Keywords**: Hierarchical VLA, Task Decomposition, Retrieval-Based Alignment, Dynamic Programming, Long-Horizon Manipulation

## TL;DR

This paper proposes RDD (Retrieval-Based Demonstration Decomposer), which models demonstration decomposition as an optimal partition problem and automatically segments long-horizon task demonstrations into subtasks aligned with the training data of low-level visuomotor policies. This approach bridges the gap between high-level planners and low-level policies in hierarchical VLA frameworks, achieving near-expert-decomposer performance on RLBench.

## Background & Motivation

Hierarchical Vision-Language-Action models (Hierarchical VLAs) represent the dominant paradigm for long-horizon robotic manipulation: a high-level planner (typically a VLM) decomposes complex tasks into step-wise language instructions, while a low-level visuomotor policy executes precise actions conditioned on these instructions. Representative works such as Hi Robot and π0.5 adopt this architecture.

When deployed on new tasks, the planner typically requires fine-tuning, and the fine-tuning dataset must contain demonstrations decomposed into subtasks. Existing decomposition approaches suffer from critical limitations:

**Manual annotation**: expensive, non-scalable, subjective, and inconsistent across annotators.

**Heuristic methods** (e.g., UVD based on visual feature change-point detection): the generated subtasks may substantially deviate from the training data distribution of the low-level policy.

This deviation gives rise to the **planner–policy misalignment** problem: the planner learns to generate subtask instructions that the low-level policy cannot execute well, degrading overall task performance. As illustrated in Fig. 1 of the paper, subtasks decomposed by UVD differ markedly from those already seen by the policy during training.

**Core Insight**: If the subtask decompositions used during planner fine-tuning are highly aligned with the training samples already encountered by the low-level policy, the planner will learn to generate instructions that the policy "excels at," thereby maximizing task success rate.

## Method

### Overall Architecture

RDD is a training-free, automated demonstration decomposition framework. Its core pipeline consists of:
1. Constructing a subtask vector database from the low-level policy's training set.
2. Formulating the decomposition of a new demonstration as an optimal partition problem.
3. Solving the optimal decomposition via dynamic programming.
4. Fine-tuning the high-level planner using the optimally decomposed data.

### Key Designs

1. **Optimal Partition Problem Formulation**

   Given a demonstration $\mathcal{S}^i$, the objective is to find the optimal partition strategy:

   $$P^{i*} = \arg\max_{P \in \Pi(\mathcal{S}^i)} \mathrm{J}(P)$$

   where $\mathrm{J}(P)$ evaluates the alignment between the partition and the low-level policy's training set $\mathcal{D}^{train}_{aug}$. When $\mathrm{J}$ is interval-additive ($\mathrm{J}(P)=\sum_{\mathcal{I}\in P}\tilde{J}(\mathcal{I})$), the problem satisfies the optimal substructure property and can be solved via dynamic programming in $O(N^2)$ scoring function calls.

   **Design Motivation**: Brute-force search requires $O(2^{N-1})$ evaluations; DP reduces complexity to polynomial time by exploiting the principle of optimality. When subtask length is bounded ($[L_{min}, L_{max}]$), complexity further reduces to $O(N)$.

2. **Interval Scoring Function and Retrieval**

   The score for each candidate interval $\mathcal{I}^i_j$ is defined as:

   $$\tilde{J}(\mathcal{I}^i_j) = |\mathcal{I}^i_j| \cdot \mathbf{sim}(\mathcal{I}^i_j, \mathrm{ANNS}(\mathcal{V}(\mathcal{I}^i_j), \mathcal{D}^{train}_{aug}))$$

   where $\mathcal{V}(\mathcal{I}) = \text{concat}(\mathcal{E}(o_b), \mathcal{E}(o_e))$ encodes the start and end frames of the interval into a vector representation, and ANNS returns the nearest-neighbor interval $\tilde{\mathcal{I}}^i_j$ from the training set. Multiplying by $|\mathcal{I}|$ ensures that the score is proportional to the number of timesteps, making the total score invariant to the number of segments (Proposition 3.1).

   **Design Motivation**: Jointly encoding the start and end frames captures both the context and the goal information of a subtask. The end frame (goal frame) carries rich information about the subtask objective.

3. **Interval Similarity Metric**

   Under the standard (non-OOD) setting, the similarity is defined as:

   $$\mathbf{sim}(\mathcal{I}^i_j, \tilde{\mathcal{I}}^i_j) = -[\delta(\mathcal{V}(\mathcal{I}^i_j), \mathcal{V}(\tilde{\mathcal{I}}^i_j)) + \alpha |1 - \frac{|\mathcal{I}^i_j|}{|\tilde{\mathcal{I}}^i_j|}|]$$

   The first term is the visual feature distance (using angular distance), and the second term is the relative temporal length difference, with $\alpha$ controlling the trade-off.

   For OOD subtask scenarios, the similarity is extended to a combination of a retrieval term and a generality term:

   $$\mathbf{sim} = -\delta(\mathcal{V}_e(\mathcal{I}), \mathcal{V}_e(\tilde{\mathcal{I}})) + \beta \mathrm{G}(\mathcal{I})$$

   where $\mathrm{G}$ evaluates the "generality" of an interval using general change-point detection methods such as UVD.

### Loss & Training

RDD itself requires no training. The planner is fine-tuned on $\mathcal{D}^{demo}_{aug}$ using LoRA (rank=128, scaling=256) for 2 epochs, taking approximately 5 minutes on 4×NVIDIA 6000 Ada GPUs. The low-level policy uses RVT, pre-trained on $\mathcal{D}^{train}_{aug}$. The vector database is built with Annoy (10 random projection trees) using angular distance $\sqrt{2(1-\cos(u,v))}$.

## Key Experimental Results

### Main Results

Evaluated on 13 RLBench tasks (averaged over 10 random seeds):

| Method | Avg. Success Rate (↑) | Avg. Rank (↓) | Annotation Required |
|---|---|---|---|
| w/o Finetune | 52.6±8.2 | 4.5±1.2 | - |
| Uniform | 71.3±5.4 | 3.1±1.2 | No |
| UVD | 71.4±5.1 | 3.0±1.3 | No |
| **RDD (Ours)** | **74.9±6.9** | **2.2±0.9** | No |
| Expert (upper bound) | 75.1±4.7 | 2.2±1.0 | Yes |

RDD trails the expert decomposer by only 0.2% in success rate, significantly outperforming UVD and uniform segmentation.

### Ablation Study

| Ablation Dimension | Configuration | Avg. Success Rate | Notes |
|---|---|---|---|
| Visual Encoder | LIV | 81.1±0.9 | Best robot-specific encoder |
| Visual Encoder | ResNet | 81.1±2.5 | Surprisingly strong performance |
| Visual Encoder | VC-1 | 75.5±3.1 | No language integration, weaker |
| Weight $\alpha$ | $\alpha$=0 | 75.0±2.5 | No temporal alignment; confuses reciprocal motions |
| Weight $\alpha$ | $\alpha$=1 | 81.1±0.9 | Optimal balance |
| Weight $\alpha$ | $\alpha$=2 | 76.2±3.0 | Over-reliance on temporal similarity |
| Demo Count | 1 demo/task | 77.9±4.5 | High data efficiency |
| Demo Count | 3 demo/task | 81.1±0.9 | Best performance |
| vs. VLM Decomposition | Gemini-2.5-pro | 72.6±4.7 | RDD outperforms strong VLM |
| OOD Setting | AgiBotWorld real-world | IoU 0.706 | RDD vs. UVD's 0.506 |

### Key Findings

1. UVD performs comparably to simple uniform segmentation, indicating that visual feature change points do not consistently align with the policy's training data.
2. RDD is robust across multiple visual encoders, including the non-robot-specific ResNet.
3. Language integration is critical for visual encoders—VC-1 and VIP, trained without language, perform worst.
4. RDD even outperforms VLM-based decomposition using Gemini-2.5-pro.
5. Algorithmic complexity scales linearly with $L_{max}$, consistent with theoretical analysis.

## Highlights & Insights

- **Precise Problem Formulation**: This work is the first to explicitly identify the planner–policy alignment problem and provide an elegant retrieval-based solution.
- **Training-Free**: RDD relies entirely on retrieval and dynamic programming, making it computationally efficient and broadly applicable.
- **Theoretical Rigor**: Starting from the optimal partition problem, the paper proves linear complexity and optimality guarantees.
- **Elegant Design of Proposition 3.1**: Multiplying by interval length renders the scoring function invariant to the number of segments, eliminating bias toward over- or under-segmentation.

## Limitations & Future Work

- The quality of RDD is bounded by the quality of the low-level policy's training data; noisy training sets may necessitate data filtering as preprocessing.
- Interval similarity relies on single-frame features from only the start and end frames, which may be insufficient for scenarios requiring historical landmark images (e.g., vision-and-language navigation).
- RDD does not concern itself with the intrinsic "optimality" of skills; it only ensures that the planner generates instructions the policy can execute.
- In entirely unfamiliar environments where no historical training data is available for retrieval, the method degrades.

## Related Work & Insights

- This work belongs to the hierarchical VLA literature (Hi Robot, π0.5, RACER) but addresses the previously overlooked planner–policy alignment problem.
- It stands in direct contrast to UVD, departing from the "change-point detection" paradigm.
- **Future Direction**: RDD can be applied not only to planner fine-tuning but also to generating aligned new training data for low-level policies.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Novel problem formulation; retrieval + DP solution is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Simulation + real-world + OOD + multi-dimensional ablations; highly comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure and rigorous theoretical derivations.
- **Value**: ⭐⭐⭐⭐⭐ Addresses a critical but overlooked alignment problem in hierarchical VLAs; plug-and-play design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] RoboCerebra: A Large-scale Benchmark for Long-horizon Robotic Manipulation Evaluation](robocerebra_a_large-scale_benchmark_for_long-horizon_robotic_manipulation_evalua.md)
- [\[ICML 2026\] HDFlow: Hierarchical Diffusion-Flow Planning for Long-horizon Tasks](../../ICML2026/robotics/hdflow_hierarchical_diffusion-flow_planning_for_long-horizon_tasks.md)
- [\[NeurIPS 2025\] EfficientNav: Towards On-Device Object-Goal Navigation with Navigation Map Caching and Retrieval](efficientnav_towards_on-device_object-goal_navigation_with_navigation_map_cachin.md)
- [\[CVPR 2026\] PALM: Progress-Aware Policy Learning via Affordance Reasoning for Long-Horizon Robotic Manipulation](../../CVPR2026/robotics/palm_progress-aware_policy_learning_via_affordance_reasoning_for_long-horizon_ro.md)
- [\[NeurIPS 2025\] VITRIX-CLIPIN: Enhancing Fine-Grained Visual Understanding in CLIP via Instruction Editing Data and Long Captions](vitrix-clipin_enhancing_fine-grained_visual_understanding_in_clip_via_instructio.md)

</div>

<!-- RELATED:END -->

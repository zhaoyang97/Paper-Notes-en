---
title: >-
  [Paper Note] InfiGUI-G1: Advancing GUI Grounding with Adaptive Exploration Policy Optimization
description: >-
  [AAAI 2026][Reinforcement Learning][GUI Grounding] To address the exploration bottleneck in semantic alignment for GUI grounding, this paper proposes the Adaptive Exploration Policy Optimization (AEPO) framework. AEPO en…
tags:
  - "AAAI 2026"
  - "Reinforcement Learning"
  - "GUI Grounding"
  - "Multimodal Large Language Models"
  - "Adaptive Exploration"
  - "Policy Optimization"
  - "Multi-Answer Generation"
date: 2026-05-08
content_hash: 2b9540356becf94b
---

# InfiGUI-G1: Advancing GUI Grounding with Adaptive Exploration Policy Optimization

**Conference**: AAAI 2026
**arXiv**: [2508.05731](https://arxiv.org/abs/2508.05731)  
**Code**: [github.com/InfiXAI/InfiGUI-G1](https://github.com/InfiXAI/InfiGUI-G1)  
**Area**: Reinforcement Learning
**Keywords**: GUI Grounding, Multimodal Large Language Models, Adaptive Exploration, Policy Optimization, Multi-Answer Generation

## TL;DR

To address the exploration bottleneck in semantic alignment for GUI grounding, this paper proposes the Adaptive Exploration Policy Optimization (AEPO) framework. AEPO enforces broad exploration via a multi-answer generation strategy, dynamically guides learning through an adaptive exploration reward function, and ensures exploration quality via a collinearity penalty mechanism, significantly improving multimodal large language model performance on complex GUI grounding tasks.

## Background & Motivation

GUI Grounding is a core perceptual task for autonomous GUI agents, requiring precise mapping of natural language instructions to specific interactive elements on the screen. This task can be decomposed along two orthogonal dimensions:

**Spatial Alignment**: Accurately localizing element coordinates — the precision of "pointing"

**Semantic Alignment**: Identifying the correct interactive element — pointing at the right target

**Limitations of Prior Work:**

- **SFT methods**: Data-intensive and difficult to generalize to unseen UI layouts
- **RLVR methods** (e.g., GRPO): Effectively improve spatial alignment by optimizing coordinate generation, but suffer from an **exploration bottleneck**

The **"confidence trap"** problem constitutes the core motivation of this paper. As a concrete example: given the instruction "search for objects using the camera," both a "Camera" button and a "Google Lens" icon appear on screen. The model may repeatedly select the "Camera" button with high confidence (a semantic error). Standard RLVR continuously samples from this high-confidence but incorrect choice, rarely encountering the correct "Google Lens" by chance, and thus fails to obtain the learning signal necessary to correct the semantic misunderstanding.

This reveals a fundamental issue: **the single-answer paradigm of standard RL leads to low sampling efficiency and an inability to escape the policy's "confidence trap"**.

## Method

### Overall Architecture

The AEPO framework comprises three synergistic components:
1. **Multi-Answer Generation**: Forces the model to generate $N$ candidate points in a single forward pass
2. **Adaptive Exploration Reward (AER)**: A nonlinear reward signal derived from the efficiency first-principle $\eta = U/C$
3. **Collinearity Penalty**: Prevents degenerate linear-scan strategies

### Key Designs

#### 1. **Problem Formulation and Multi-Answer Generation**

GUI grounding is formalized as a policy optimization problem:
- **Context** $c = (\mathcal{S}, \mathcal{I})$: screenshot + natural language instruction
- **Action** $a$: coordinate point $p=(x,y)$
- **Policy** $\pi_\theta(a|c)$: action probability distribution given context
- **Objective**: $\theta^* = \arg\max_\theta \mathbb{E}_{c\sim\mathcal{D}, a\sim\pi_\theta(\cdot|c)}[R(a, B)]$

The key innovation of multi-answer generation is prompting the model to produce $N$ candidate points $\mathcal{A} = \{p_1, p_2, \ldots, p_N\}$ within a single forward pass. This compels the model to explore beyond its highest-confidence single prediction, substantially increasing the probability of sampling a correct action from the tail of the policy distribution — particularly critical for semantically challenging instances.

#### 2. **Adaptive Exploration Reward (AER)**

AER is derived from the efficiency first-principle $\eta = U/C$:

**Utility $U$**:
- Exploration success (any candidate falls within the ground truth): $U = +1$
- Exploration failure: $U = -1$

**Cost $C$**: Modeled as the geometric mean of two cost components:
- Proposal cost $C_p = N$ (cost of generating $N$ candidates)
- Verification cost $C_v$: the rank $k$ of the first correct point on success, or $N$ on failure
- $C = \sqrt{C_p \cdot C_v}$ (geometric mean captures diminishing marginal returns)

The resulting accuracy reward:

$$R_{\text{accuracy}}(\mathcal{A}, B) = \begin{cases} 1/\sqrt{N \cdot k} & \text{if } \exists\, p_i \in B \\ -1/N & \text{otherwise} \end{cases}$$

**Dynamic behavior**:
- **On failure**: The penalty is only $-1/N$, decreasing as $N$ grows, encouraging broader exploration
- **On success**: The reward $1/\sqrt{Nk}$ is larger when $k$ is small, rewarding efficient accurate prediction
- This asymmetric design promotes exploration on failure and convergence on success

#### 3. **Collinearity Penalty Mechanism**

If the $N$ generated candidate points are approximately collinear (detected by checking whether the area of any triangle formed by three points is near zero), the accuracy reward is overridden with a large negative value $R_{\text{accuracy}} = -1$. This prevents the model from adopting a simple but inefficient linear-scan strategy, incentivizing genuinely diverse semantic exploration in geometric space.

### Loss & Training

The total reward signal combines format and accuracy rewards:

$$R_{\text{total}} = R_{\text{format}} + R_{\text{accuracy}}$$

The format reward $R_{\text{format}}$ is +1 when the output format is correct and 0 otherwise, serving as a prerequisite for subsequent reward evaluation. The total reward is used to compute the advantage estimate $\hat{A}$, which directly guides policy parameter updates. Training employs standard policy gradient algorithms such as GRPO, PPO, or RLOO.

## Key Experimental Results

### Main Results

**MMBench-GUI Benchmark (Top-1 Accuracy %)**:

| Model | Windows | MacOS | Linux | iOS | Android | Web | Avg |
|-------|---------|-------|-------|-----|---------|-----|-----|
| UI-TARS-1.5-7B | 68.3/39.0 | 69.0/44.5 | 64.4/37.8 | 88.5/69.4 | 90.5/69.3 | 81.0/56.5 | 64.3 |
| Naive RLVR-7B | 79.3/58.1 | 82.3/62.7 | 64.4/44.9 | 94.9/89.1 | 95.5/84.2 | 92.9/79.5 | 79.3 |
| **InfiGUI-G1-7B** | **82.7/61.8** | **83.8/63.9** | **72.3/52.0** | 94.9/89.4 | 95.2/85.6 | 93.5/76.3 | **80.8** |
| G1-7B + Exploration Success | 87.1/69.1 | 87.2/76.3 | 78.5/58.2 | 98.1/92.4 | 98.0/91.8 | 97.1/85.7 | **86.4** |

**ScreenSpot-Pro Benchmark (Top-1 Accuracy %)**:

| Model | CAD | Dev. | Creative | Scientific | Office | OS | Avg |
|-------|-----|------|----------|-----------|--------|-----|-----|
| Naive RLVR-7B | 53.8/17.2 | 71.4/15.9 | 60.6/11.9 | 76.4/26.4 | 74.6/34.0 | 54.2/20.2 | 47.6 |
| **InfiGUI-G1-3B** | 50.8/25.0 | 64.9/20.0 | 51.5/16.8 | 68.8/32.7 | 70.6/32.1 | 49.5/15.7 | 45.2 |

### Ablation Study

The contribution of each component is assessed via comparison against Naive RLVR:

| Configuration | MMBench Avg | Notes |
|---------------|-------------|-------|
| Naive RLVR-3B | 70.9 | Baseline single-answer RL |
| InfiGUI-G1-3B (Top-1) | 73.4 | Combined gain from multi-answer + AER + collinearity penalty |
| G1-3B Exploration Success Rate | 81.6 | Upper bound utilizing all multi-answer candidates |
| Naive RLVR-7B | 79.3 | Larger model baseline |
| InfiGUI-G1-7B (Top-1) | 80.8 | Surpasses RLVR evaluating only the first answer |
| G1-7B Exploration Success Rate | 86.4 | Upper bound with all multi-answer candidates |

Key insights:
- For the 3B model, Top-1 already exceeds RLVR by 2.5%; exploration upper bound improves by 10.7%
- For the 7B model, the largest gains over Naive RLVR appear in the Advanced category (semantically challenging items)
- Average $N$ for exploration success is 1.6–2.0, indicating the model has learned efficient exploration

### Key Findings

1. **Semantic alignment is the critical bottleneck**: Gains from InfiGUI-G1 are substantially larger on Advanced categories than Basic categories, validating that AEPO effectively addresses the semantic alignment problem
2. **Exploration efficiency**: InfiGUI-G1-7B achieves an 86.4% exploration success rate with an average of only 1.6 candidates, demonstrating that the model learns not only to explore but to explore efficiently
3. **Cross-platform generalization**: Consistent improvements are observed across all platforms — Windows, MacOS, Linux, iOS, Android, and Web
4. **3B vs. 7B scaling effect**: The 7B model outperforms the 3B model in both absolute performance and the incremental gains attributable to AEPO
5. **Training stability**: Standard deviations across 5 runs are only σ = 0.11–0.41, indicating highly stable training

## Highlights & Insights

- **Precise problem definition**: Decomposing GUI grounding into spatial and semantic alignment, and precisely diagnosing the exploration bottleneck in semantic alignment, yields a clear and well-motivated problem formulation
- **Theoretical grounding of AER**: Deriving the reward function from the efficiency first-principle $\eta=U/C$ rather than through ad-hoc design confers theoretical elegance
- **Generality of the multi-answer paradigm**: The framework of multi-answer generation combined with adaptive rewards is not limited to GUI grounding; it may also prove effective for other visual grounding tasks requiring spatial exploration, such as referring expression comprehension
- **Elegance of the collinearity penalty**: A simple geometric constraint prevents degenerate strategies with negligible computational overhead yet notable empirical effect

## Limitations & Future Work

- Multi-answer generation increases inference-time computational cost (though average $N$ remains small), requiring a performance–efficiency trade-off in practical deployment
- The collinearity penalty relies on a simple triangle-area heuristic, which may need refinement in higher-dimensional or more complex spatial exploration scenarios
- The current framework assumes ground truth bounding boxes; adaptation may be required for non-rectangular or pixel-level targets
- Whether AEPO retains its advantages in agent-level (multi-step decision-making) tasks remains unexplored
- The utility function in AER considers only binary outcomes (success/failure), leaving progressive metrics such as IoU unexploited

## Related Work & Insights

- Shares the RLVR foundation with GUI RL methods such as UI-R1 and GUI-R1, but overcomes the limitations of the single-answer paradigm
- The multi-answer generation idea bears resemblance to Best-of-N sampling, but AER provides a finer-grained learning signal
- The adaptive reward design balancing exploration and exploitation is transferable to other RL settings, such as multi-target exploration in robotic manipulation

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The multi-answer + adaptive reward framework is innovative, though the novelty of individual components is moderate
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Multi-benchmark, cross-platform evaluation with reported standard deviations and exploration success rate analysis
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure; motivation figures are highly intuitive
- **Value**: ⭐⭐⭐⭐ — Establishes a new state of the art in the rapidly growing field of GUI agents

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Start Small, Think Big: Curriculum-based Relative Policy Optimization for Visual Grounding](start_small_think_big_curriculum-based_relative_policy_optimization_for_visual_g.md)
- [\[AAAI 2026\] MARS: Multi-Agent Adaptive Reasoning with Socratic Guidance for Automated Prompt Optimization](mars_multi-agent_adaptive_reasoning_with_socratic_guidance_f.md)
- [\[AAAI 2026\] Reasoning with Exploration: An Entropy Perspective](reasoning_with_exploration_an_entropy_perspective.md)
- [\[AAAI 2026\] Behaviour Policy Optimization: Provably Lower Variance Return Estimates for Off-Policy Reinforcement Learning](behaviour_policy_optimization_provably_lower_variance_return_estimates_for_off-p.md)
- [\[ICLR 2026\] unsupervised learning of efficient exploration pre-training adaptive policies vi](../../ICLR2026/reinforcement_learning/unsupervised_learning_of_efficient_exploration_pre-training_adaptive_policies_vi.md)

</div>

<!-- RELATED:END -->

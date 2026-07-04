---
title: >-
  [Paper Note] Large Language Model (LLM)-enabled In-context Learning for Wireless Network Optimization
description: >-
  [ICML 2025 (Workshop on ML4Wireless)][Signal & Communication][Large Language Models] This paper proposes a base station power control algorithm based on LLM In-context Learning (ICL). By leveraging natural language task descriptions and experience-pool-driven exemplar selection, it achieves performance close to traditional deep reinforcement learning without updating model parameters.
tags:
  - "ICML 2025 (Workshop on ML4Wireless)"
  - "Signal & Communication"
  - "Large Language Models"
  - "In-context Learning"
  - "Wireless Network Optimization"
  - "Power Control"
  - "Experience Pool"
date: 2026-05-08
content_hash: 33a9482040745e71
---

# Large Language Model (LLM)-enabled In-context Learning for Wireless Network Optimization

**Conference**: ICML 2025 (Workshop on ML4Wireless)  
**arXiv**: [2408.00214](https://arxiv.org/abs/2408.00214)  
**Code**: None  
**Area**: Wireless Communications  
**Keywords**: Large Language Models, In-context Learning, Wireless Network Optimization, Power Control, Experience Pool

## TL;DR

This paper proposes a base station power control algorithm based on LLM In-context Learning (ICL). By leveraging natural language task descriptions and experience-pool-driven exemplar selection, it achieves performance close to traditional deep reinforcement learning without updating model parameters.

## Background & Motivation

As 6G networks become increasingly complex, traditional network optimization methods face two major bottlenecks:

**Convex Optimization Methods**: require specialized mathematical modeling for each task to convert objective functions or constraints into convex forms, leading to poor generalization.

**Machine Learning Methods** (e.g., reinforcement learning): Although they have lower requirements for explicit problem modeling, the model training and hyperparameter tuning processes are lengthy and require a large number of iterations.

The emergence of LLMs offers new avenues for network optimization. In-context learning (ICL) presents two core advantages:
- **Training-free**: Relies on the inference capability of the LLM, avoiding the computational overhead of parameter updates.
- **Natural Language Interface**: Network operators can directly describe optimization tasks using natural language.

However, most existing research on LLM-based network optimization focuses on static scenarios or simple feedback mechanisms, lacking systematic designs for experience accumulation and exemplar selection in dynamic environments. This paper takes base station (BS) transmit power control as a case study and proposes a comprehensive LLM in-context learning optimization framework.

## Method

### Overall Architecture

The overall workflow consists of four core modules:

1. **Natural Language Task Description**: Describes the optimization objectives, environment definitions, and response rules using structured language.
2. **Experience Pool**: Collects historical decision-making experiences of the LLM, denoted as $E = \{s, a, r(s, a)\}$.
3. **Exemplar Selection**: Selects the most valuable exemplars from the experience pool based on the current state.
4. **LLM Inference**: Combines the task description, selected exemplars, and current state into a prompt, from which the LLM outputs a power control decision.

The core formal definition of in-context learning is expressed as:

$$D_{task} \times \mathcal{E}_t \times s_t \times \mathcal{LLM} \Rightarrow a_t$$

where $D_{task}$ represents the task description, $\mathcal{E}_t$ is the set of exemplars at time $t$, $s_t$ represents the current environment state, and $a_t$ represents the decision output by the LLM.

### Key Designs

#### 1. Natural Language Task Description Template

The task description is structured into three levels:

- **Task_goal**: Specifies the "decision-making task for base station power control" with the objective of "selecting from 4 power levels."
- **Task_definition**: Introduces the environmental states to be considered (e.g., number of users) and embeds the exemplar set $\mathcal{E}_t$.
- **Rules**: Establishes response rules, such as "based on the above exemplars, select from level 1-4."

This template design is generalizable and can be extended to other network optimization tasks.

#### 2. Experience Pool and Exemplar Design

Each exemplar is defined as a triplet $E = \{s, a, r(s, a)\}$, where $s$ is the environment state, $a$ is the decision action, and $r$ is the reward. Inspired by reinforcement learning, the reward function is defined as:

$$r = P_{target} - P_b - \beta$$

- $P_{target}$: Target power consumption
- $P_b$: Actual total power consumption of the BS ($P_b = \sum_{k=1}^{K_b} p_{b,k}$)
- $\beta$: Penalty term, applied only when the minimum data rate constraint $C_{min}$ is violated

After each decision is made by the LLM, the actual execution result $(s_t, a_t, r_t)$ is appended to the experience pool, forming a continuously accumulating knowledge base.

#### 3. State-Based Exemplar Selection (Discrete States)

For discrete state spaces (e.g., number of users), exemplars with identical states are directly matched from the experience pool:

$$\mathcal{E}_{relevant} = \{E\{s, a, r(s,a)\} \mid s = s_{target}, E \in \mathcal{E}_{pool}\}$$

Selection from the matched results includes:
- **Recommended exemplars**: Experiences with the highest reward values.
- **Negative exemplars**: Experiences with low rewards or constraint violations.

Both types of exemplars jointly assist the LLM in understanding "what constitutes a good decision versus a bad decision."

#### 4. Ranking-Based Exemplar Selection (Continuous States)

For continuous state spaces (e.g., average distance from users to the base station), exact state matching is infeasible. Therefore, a comprehensive evaluation metric is defined:

$$\mathcal{L}(E, s_{target}) = r(s, a) - \tau \|s - s_{target}\|$$

This metric simultaneously considers two dimensions:
- $r(s, a)$: The reward quality of the exemplar itself.
- $\|s - s_{target}\|$: The $L_2$ distance between the exemplar state and the target state.
- $\tau$: Hyperparameter for the distance weight.

Sorting by $\mathcal{L}$ and selecting the top-k exemplars balances the dual requirements of "high quality" and "high relevance."

#### 5. Epsilon-Greedy Exploration Strategy

To balance exploration and exploitation, a classic $\epsilon$-greedy strategy is introduced:

$$a = \begin{cases} \text{random action}, & \text{if } rand < \epsilon \\ \text{LLM decision}, & \text{otherwise} \end{cases}$$

Random exploration continuously generates new exemplars, enriching the experience pool and allowing the LLM to learn from superior exemplars.

### Loss & Training

The core feature of this method is that it is **training-free**—the LLM parameters are not updated at all. Its "learning" process is achieved through the following mechanisms:

- **Implicit Fine-tuning Theory**: According to the analysis by Dai et al., ICL can be equated to $\tilde{f}_{ICL}(\mathbf{q}) = \mathbf{q}(W_{ZSL} + \Delta W_{ICL})$, meaning the LLM generates meta-gradients through the attention mechanism during feedforward computation, achieving implicit weight updates.
- **Experience Pool Growth**: As interaction rounds increase, the experience pool continuously accumulates high-quality exemplars.
- **Exemplar Filtering Optimization**: Better exemplars $\to$ better decisions $\to$ new experiences with higher rewards $\to$ positive feedback loop.

In terms of computational complexity, the exemplar selection process has linear complexity (traversing the experience pool to compute matching metrics), requiring no backpropagation.

## Key Experimental Results

### Experimental Settings

- 3 adjacent small base stations (SBSs), with the number of users randomly varying between 5 and 15.
- SBS coverage radius of 20 meters, with the channel model based on 3GPP urban networks.
- Two scenarios: Case I (discrete states - number of users), Case II (continuous states - average user distance).
- 4 candidate power levels available.

### Main Results

| Method | Discrete State Reward | Continuous State Reward | Requires Training |
|------|-------------|-------------|-------------|
| Exhaustive Search (Optimal) | Optimal Baseline | Optimal Baseline | No |
| DRL (Traditional Baseline) | Near-optimal | Near-optimal | Yes (extensive iterations) |
| GPT-4 (ICL) | Close to DRL | Close to DRL | **No** |
| Llama3-70b (ICL) | Close to DRL | Close to DRL | **No** |
| Llama3-8b (ICL) | Close to DRL | Close to DRL | **No** |
| GPT-3.5 (ICL) | Below DRL | Below DRL | No |
| Feedback-based | Significantly below DRL | Significantly below DRL | No |

### Ablation Study

| Configuration | Average Reward Variation | Description |
|------|-------------|------|
| Full Method (ICL + Experience Pool + Exemplar Selection + Exploration) | Baseline | All components work collaboratively |
| Remove Experience Pool | Drastic decrease | No historical experience for reference, leading to poor decision quality |
| Remove Exemplar Selection | Significant decrease | Random exemplars fail to guide the LLM effectively |
| Remove $\epsilon$-greedy Exploration | Noticeable decrease | Insufficient diversity in the experience pool |
| Feedback-based Method | Significantly lower than the full method | Feedback alone cannot capture the complexity of the dynamic environment |

### Key Findings

1. **Model capability determines performance**: GPT-4 and Llama3 (SOTA models) perform significantly better than GPT-3.5 (an earlier model), indicating that ICL performance is strongly correlated with the base capability of the LLM.
2. **Exemplar quantity effect**: Increasing the number of exemplars in the prompt continuously improves the reward, albeit with diminishing marginal returns.
3. **State space expansion**: Larger state spaces require more exemplars to achieve equivalent performance, yet the overall trend continues to improve.
4. **Constraint adaptability**: Under different minimum data rate constraints, GPT-4 and Llama3 can adaptively adjust their strategies, maintaining robust performance in both power consumption and quality of service (QoS).
5. **Advantages in dynamic environments**: Compared to feedback-based methods that can only handle static optimization, the experience pool mechanism of the proposed method effectively manages dynamic scenarios.

## Highlights & Insights

1. **Paradigm Innovation**: For the first time, LLM ICL is systematically applied to dynamic wireless network optimization, demonstrating the feasibility of "using language for network optimization."
2. **Ingenious Experience Pool Design**: Transfers the concept of experience replay from reinforcement learning to ICL scenarios, where the $(s, a, r)$ triplet design is simple yet effective.
3. **Dual-Mode Exemplar Selection**: Tailors exact matching and ranking-based selection strategies for discrete and continuous states, respectively, enhancing practicality.
4. **Zero Training Overhead**: In scenarios where network dynamics change frequently, the training-free property translates to extremely fast deployment and adaptation.
5. **Explainability**: The LLM can provide natural language explanations for its decisions, helping operators understand complex network behaviors.

## Limitations & Future Work

1. **Simplified Scenarios**: Only examines small-scale scenarios with 3 SBSs and 4 power levels; scalability to large-scale networks remains unverified.
2. **Inference Latency**: The inference time of LLMs may fail to satisfy the real-time requirements of millisecond-level network control.
3. **API Cost**: The cost of API calls using commercial models like GPT-4 could be prohibitively high during large-scale deployment.
4. **Continuous Action Spaces**: Power is discretized into 4 levels, which precludes fine-grained, continuous power control.
5. **Multi-Objective Optimization**: Currently, only power minimization under data rate constraints is considered; more complex multi-objective scenarios have not been addressed.
6. **Exemplar Pool Expansion**: The experience pool may grow indefinitely over long-term operation, requiring the design of eviction mechanisms.

## Related Work & Insights

- **LLM for Telecom**: Aligns with works such as Lin et al. (6G edge intelligence) and Qiu et al. (LLM wireless design), but this paper introduces the first experience-pool-driven dynamic ICL framework.
- **ICL Theory**: Explains the ICL mechanism based on the implicit fine-tuning theory of Dai et al., treating the attention mechanism as a meta-gradient computer.
- **Cross-Domain Inspiration**: The ICL paradigm combining experience pools and exemplar selection can be generalized to other network optimization tasks (e.g., beamforming, RIS phase optimization) and even broader combinatorial optimization problems.

## Rating

- Novelty: ⭐⭐⭐⭐ — Utilizing ICL for dynamic network optimization is a promising new direction, and the experience pool design is quite novel.
- Experimental Thoroughness: ⭐⭐⭐ — The ablation studies are comprehensive, but the scenario scale is small, and comparisons with more baselines are lacking.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure with formal task definitions and algorithm descriptions.
- Value: ⭐⭐⭐ — A workshop paper at the proof-of-concept stage, still far from practical deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Neural Video Compression with Context Modulation](../../CVPR2025/signal_comm/neural_video_compression_with_context_modulation.md)
- [\[ICCV 2025\] Boosting Multimodal Learning via Disentangled Gradient Learning](../../ICCV2025/signal_comm/boosting_multimodal_learning_via_disentangled_gradient_learning.md)
- [\[ICLR 2026\] TS-DDAE: A Novel Temporal-Spectral Denoising Diffusion AutoEncoder for Wireless Signal Recognition Model Pre-training](../../ICLR2026/signal_comm/ts-ddae_a_novel_temporal-spectral_denoising_diffusion_autoencoder_for_wireless_s.md)
- [\[ACL 2025\] WirelessMathBench: A Mathematical Modeling Benchmark for LLMs in Wireless Communications](../../ACL2025/signal_comm/wirelessmathbench_a_mathematical_modeling_benchmark_for_llms_in_wireless_communi.md)
- [\[CVPR 2026\] CLAY: Conditional Visual Similarity Modulation in Vision-Language Embedding Space](../../CVPR2026/signal_comm/clay_conditional_visual_similarity.md)

</div>

<!-- RELATED:END -->

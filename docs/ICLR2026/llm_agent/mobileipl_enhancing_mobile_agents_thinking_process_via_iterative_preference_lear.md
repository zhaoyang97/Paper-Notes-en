---
title: >-
  [Paper Note] MobileIPL: Enhancing Mobile Agents Thinking Process via Iterative Preference Learning
description: >-
  [ICLR 2026][LLM Agent][Mobile GUI Agent] To address the lack of CoaT (Chain of Action-Planning Thoughts) reasoning trajectories and the difficulty of step-level annotation for mobile GUI agents, MobileIPL uses MCTS-style iterative sampling to construct a CoaT-tree. It scores leaf nodes based on rule-based rewards and backpropagates values to intermediate thinking steps, constructing "Thinking-level" DPO pairs (T-DPO) to optimize the reasoning process. This approach outperform…
tags:
  - "ICLR 2026"
  - "LLM Agent"
  - "Mobile GUI Agent"
  - "CoaT"
  - "Iterative Preference Learning"
  - "Thinking-level DPO"
  - "Rule-based Reward"
  - "Instruction Evolution"
date: 2026-05-08
content_hash: ba0e999b435793b2
---

# MobileIPL: Enhancing Mobile Agents Thinking Process via Iterative Preference Learning

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=PS8iu4PxKz](https://openreview.net/forum?id=PS8iu4PxKz)  
**Code**: [MobileIPL-dataset (HuggingFace)](https://huggingface.co/datasets/xwk123/MobileIPL-dataset)  
**Area**: Mobile GUI Agent / Preference Learning  
**Keywords**: Mobile GUI Agent, CoaT, Iterative Preference Learning, Thinking-level DPO, Rule-based Reward, Instruction Evolution  

## TL;DR
To address the lack of CoaT (Chain of Action-Planning Thoughts) reasoning trajectories and the difficulty of step-level annotation for mobile GUI agents, MobileIPL uses MCTS-style iterative sampling to construct a CoaT-tree. It scores leaf nodes based on rule-based rewards and backpropagates values to intermediate thinking steps, constructing "Thinking-level" DPO pairs (T-DPO) to optimize the reasoning process. This approach outperforms large-scale continuously pre-trained models such as OS-ATLAS and UI-TARS on three mobile GUI benchmarks.

## Background & Motivation
**Background**: VLM-driven mobile GUI agents need to generate intermediate thoughts between user instructions and current interfaces. The Chain of Action-Planning Thoughts (CoaT, similar to System-2 slow thinking) paradigm proposed by AITZ has been proven to significantly improve reasoning performance in GUI tasks by organizing "Description-Thought-Action-Grounding" into a structured chain of thought.

**Limitations of Prior Work**: (1) High-quality and diverse CoaT trajectories are scarce; direct SFT on fixed CoaT trajectories easily leads to overfitting, trapping the model in rigid reasoning patterns. (2) While self-training can mitigate data scarcity, using only final answer correctness as a reward ignores the quality of intermediate reasoning steps, leading to reward hacking. (3) Search methods like ReST-MCTS* rely on process reward models (PRM) to score intermediate steps, but PRMs require large-scale manual step-level annotation. In the mobile GUI domain, environments depend on real devices or simulators, making the cost of step-level annotation far higher than in text, code, or math tasks.

**Key Challenge**: Optimizing the quality of "intermediate thinking steps" (rather than just final action correctness) without paying the high cost of manual step-level PRM annotation.

**Goal**: Achieve fine-grained preference optimization for the entire thinking process of mobile GUI agents without PRMs or manual step-level labels, while mitigating overfitting during the warm-up SFT stage.

**Core Idea**: **Replace PRMs with rule-based rewards.** Each action is expanded into a CoaT-tree where leaf nodes (complete actions) are scored against ground truth via rules. Scores are backpropagated to intermediate steps to automatically construct thinking-level DPO pairs. Simultaneously, **GPT-4o three-stage instruction evolution** is used to inject diverse Q&A during SFT to prevent overfitting and enhance UI understanding.

## Method

### Overall Architecture
MobileIPL consists of three stages: first, warm-up SFT using mixed data augmented by "instruction evolution" to obtain a seed policy; second, iterative sampling for each action to build a CoaT-tree, scoring leaf nodes with rule rewards and backpropagating values to intermediate steps; finally, filtering contrastive pairs from the tree for Thinking-level DPO self-training, using the updated agent as a new base for further iterations until performance plateaus.

```mermaid
flowchart LR
    A[General VLM] --> B[Instruction Evolution<br/>GPT-4o 3-stage Q&A]
    B --> C[Warm-up SFT<br/>Mixed T+Q for Seed Policy πS0]
    C --> D[CoaT-tree Iterative Sampling<br/>K continuations per step]
    D --> E[Leaf Rule Reward v·st·<br/>+ Backpropagation]
    E --> F[Contrastive Data Filtering<br/>α/β/γ Classification for DPO Pairs]
    F --> G[Thinking-level DPO Training]
    G -->|Update agent as new base| D
```

### Key Designs

**1. Multi-turn Thinking Modeling: Decomposing one action into four dialogue turns.** The authors formalize each action $\hat{a}_i$ in CoaT reasoning as a multi-turn dialogue $\hat{a}_i=[s_1,s_2,s_3,s_4]$, corresponding to Description, Thought, Action decision, and Grounding respectively, where $s_1=\text{Description}(P_1,u_i)$, $s_2=\text{Thought}(P_2,u_i,I,\hat{a}_{0:i-1},s_1)$, $s_3=\text{Action}(\cdot)$, and $s_4=\text{Grounding}(\cdot)$. The motivation is practical: when a model decodes a full reasoning segment at once, the image modality $u$ occupies most input tokens, overshadowing text instructions $I$ and action history, which shifts attention away from text details. Multi-turn dialogue forces each step to focus on the current reasoning sub-task, balancing multimodal token ratios and ensuring format-compliant outputs.

**2. CoaT-tree Iterative Sampling + Rule Rewards: Using rules instead of PRMs to score intermediate steps.** Sampling stepwise following the CoaT paradigm, $K$ continuations $\hat{s}_t=\{\hat{s}_t^{(k)}\mid \hat{s}_{0:t-1}\}_{k=1}^K$ are sampled for each step $t$, forming a tree. Leaf nodes (complete actions) are compared with ground truth $a^*$ for rule rewards: 1 for perfect correctness; $v_{type}+\text{score}_{match}$ if the action type matches; 0 otherwise. $\text{score}_{match}$ uses normalized distance $d(x,y)$ for CLICK and text F1 for INPUT to provide smooth rewards for "near misses." Crucially, leaf values are **backpropagated** to intermediate steps: $v(s_{t-1})=c\cdot\frac{1}{K}\sum_{k=1}^{K}v(s_t^{(k)})$, where $c$ is a discount factor. This provides each intermediate thinking step with a continuous value without needing PRMs or manual labels.

**3. Contrastive Data Filtering: Categorizing trees into α/β/γ to select samples.** Based on the value distribution of leaf nodes, trees are classified into three types: $\alpha$ are "perfect trees" where all leaf values are 1 (stable correct outputs, not used for contrastive pairs); $\beta$ are "potentially correct trees" with both correct and incorrect leaves, serving as the main source for DPO pairs; $\gamma$ are "to-be-refined trees" where no leaf value is 1. In $\beta$, samples with value 1 and diverse action types are used as positives, and pairs are constructed as $\beta_{pairs}=\langle \hat{s}_t^{(k)}\uparrow,\hat{s}_t^{(k')}\downarrow\mid v(\hat{s}_t^{(k)})-v(\hat{s}_t^{(k')})>1/K\rangle$ (requiring a value gap $>1/K$ to avoid noise). In $\gamma$, ground truth actions $a^*$ are used as positives against sampled negatives, $\gamma_{pairs}=\langle a^*\uparrow,\hat{s}_t^{(k)}\downarrow\rangle$.

**4. Thinking-level DPO (T-DPO) + Iterative Self-training.** Given the same prefix $s_{1:t-1}$, positive $s_t^+$ and negative $s_t^-$ continuations for the same thinking step are compared:
$$L_{\text{T-DPO}}=-\mathbb{E}\big[\log\sigma(\beta\log\frac{\pi_\theta(s_t^+\mid s_{1:t-1})}{\pi_{ref}(s_t^+\mid s_{1:t-1})}-\beta\log\frac{\pi_\theta(s_t^-\mid s_{1:t-1})}{\pi_{ref}(s_t^-\mid s_{1:t-1})})\big]$$
This is optimized alongside an SFT loss. Unlike TreePO/SPO which segments long sequences into many short fragments, MobileIPL uses the fixed CoaT-tree to model thinking, with values derived directly from rule rewards (avoiding unstable PRMs), making sampling and training more efficient.

**5. Three-stage Instruction Evolution: Mitigating SFT overfitting and enhancing UI understanding.** Since CoaT patterns become rigid after warm-up SFT, the authors use GPT-4o to generate three levels of Q&A based on real screenshots: Level I General GUI Q&A (grounding/referring/description), Level II Component functions and nested relationships (avoiding misidentifying textviews as buttons), and Level III Advanced FAQ (page structure, navigation expectations). These are manually filtered and mixed with original trajectories $T$ for SFT, preventing static instruction overfitting and improving UI layout understanding via visually grounded Q&A. This expanded the sampling space from 4% to 31% and the correct answer ratio from 72.7% to 77.9%.

## Key Experimental Results

### Main Results
Performance on AITZ, AMEX, and AndroidControl benchmarks using Qwen2-VL-7B as the backbone, with Step.Acc as the primary metric:

| Benchmark | Key Comparison | MobileIPL | Strongest Baseline |
|------|----------|-----------|----------|
| AITZ (Total) | vs Falcon-UI-7B (3M GUI pre-training) | **69.15** | 69.10 |
| AITZ (Total) | vs Seed Model / Qwen2-VL-7B | **69.15** | 55.40 / 60.36 |
| AMEX (Overall) | vs SphAgent-7B (SOTA) | **74.29** | 70.71 (+3.58) |
| AMEX (Overall) | vs OS-Atlas / UI-Tars | **74.29** | 70.33 / 70.33 (+3.96) |
| AndroidControl (Step.Acc) | vs UI-Tars-7B / OS-Atlas | **72.7** | 72.5 / 71.2 |
| AndroidControl (Grounding) | vs Qwen2-VL-7B (SFT) | **77.0** | 68.5 (+8.5) |

On the AndroidControl OOD subsets (IDD / app-unseen / task-unseen), MobileIPL achieved **73.6 / 70.0 / 72.2**, significantly outperforming OS-Atlas (71.2 / 60.7 / 66.2) and Qwen2-VL-GRPO (70.2 / 68.1 / 69.7).

### Ablation Study
Ablation on AITZ Round 1 (MobileIPL-R1 = 65.4):

| Setting | Total | Δ |
|------|-------|----|
| MobileIPL-R1 (Full) | 65.4 | — |
| − IPL (SFT only) | 60.4 | −5.0 |
| − Instruction Evo | 62.9 | −2.5 |
| − IPL Negatives | 61.4 | −4.0 |
| − IPL + Naive DPO (Full Trajectory) | 60.3 | −5.1 |
| 1/2 Training Data (R1) | 64.8 | −0.6 |
| 4/5 Training Data (R2) | 60.6 | −4.8 |

### Key Findings
- **IPL is core**: Removing IPL causes a 5.0% drop; negative samples contribute 4.0%, suggesting they teach "how to reason" rather than simple memorization.
- **Thinking-level DPO outperforms Naive DPO**: Using naive DPO on full trajectories dropped performance to 60.3%, lower than SFT with positive CoaT-tree samples (61.4%), validating the effectiveness of step-wise optimization.
- **Efficient at low-resource**: With half the data, the first round of IPL (64.8) already surpassed the best results of original CoaT-SFT and naive DPO.
- **Best Efficiency-Performance Trade-off**: MobileIPL (69.15, ~27 rollouts) outperforms SPO-Chain (68.03, ~54 rollouts) and GRPO (66.29, 8 rollouts).
- **K=3 is the sweet spot**: Increasing $K$ from 3 to 4 increases tree nodes from 27 to 64 but yields less than 1% improvement.

## Highlights & Insights
- **Rule-based Rewards + Backpropagation as PRM Alternative**: In mobile GUI scenarios where step-level labeling is extremely expensive, this bypasses the need for unstable PRMs by using smooth signals from CLICK distance and INPUT F1.
- **α/β/γ Classification is a Clean Abstraction**: It clearly defines which data to ignore (perfect), which to mine for contrast (mixed), and which to use for grounding (all wrong), using a value gap threshold to filter noise.
- **Multi-turn Dialogue Addresses VLM Pain Points**: It prevents image tokens from overwhelming text instructions, a critical observation for GUI agents compared to text-only CoT.
- **Small Models Beating Large Pre-trained Models**: By optimizing the thinking process, a 7B backbone with limited data can outperform models with millions of GUI pre-training samples like OS-Atlas/UI-TARS.

## Limitations & Future Work
- **Weak PRESS Actions**: MobileIPL's PRESS accuracy on AITZ is significantly lower than some baselines (23.5 in R1), attributed to sample distribution issues.
- **Handcrafted Reward Rules**: Smooth rewards for CLICK/INPUT rely on manual $v_{type}/v_{format}$ and distance normalization; other actions (SCROLL/STOP) only use binary 0/1 rewards.
- **Reliance on GPT-4o for Instruction Evolution**: Generating three-stage Q&A still requires closed-source models and manual filtering, creating a barrier for scaling and reproduction.
- **Empirical Iteration Termination**: It lacks a clear convergence criterion, with the number of rounds $R$ requiring manual tuning.

## Related Work & Insights
- **CoaT / AITZ**: The base modeling paradigm of Thought-Action-Grounding.
- **ReST-MCTS* / Xie et al.**: Represent the MCTS + PRM approach; this work explicitly avoids their PRM labeling costs.
- **TreePO / TreeRL / SPO**: Segment long sequences for preference optimization; MobileIPL is more efficient by using fixed CoaT-trees and rule-based values.
- **Insight**: In embodied/GUI scenarios with expensive labeling, "Rule Rewards + Tree Backpropagation" is a viable low-cost alternative to PRMs.

## Rating
- **Novelty**: ⭐⭐⭐⭐ —— The combination of rule-based backpropagation and thinking-level DPO is a practical innovation for GUI agents.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ —— Comprehensive benchmarks, OOD testing, efficiency analysis, and ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ —— Clear motivation, well-described algorithms, and good use of diagrams.
- **Value**: ⭐⭐⭐⭐ —— Demonstrates that optimizing thinking handles resource constraints better than massive pre-training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Solving the Granularity Mismatch: Hierarchical Preference Learning for Long-Horizon LLM Agents](solving_the_granularity_mismatch_hierarchical_preference_learning_for_long-horiz.md)
- [\[ICLR 2026\] MobileRL: Online Agentic Reinforcement Learning for Mobile GUI Agents](mobilerl_online_agentic_reinforcement_learning_for_mobile_gui_agents.md)
- [\[ICLR 2026\] GUI-Shift: Enhancing VLM-Based GUI Agents through Self-supervised Reinforcement Learning](gui-shift_enhancing_vlm-based_gui_agents_through_self-supervised_reinforcement_l.md)
- [\[ICML 2026\] From Player to Master: Enhancing Test-Time Learning of LLM Agents via Reinforcement Learning over Memory](../../ICML2026/llm_agent/from_player_to_master_enhancing_test-time_learning_of_llm_agents_via_reinforceme.md)
- [\[ICLR 2026\] FingerTip 20K: A Benchmark for Proactive and Personalized Mobile LLM Agents](fingertip_20k_a_benchmark_for_proactive_and_personalized_mobile_llm_agents.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICLR 2026\] Solving the Granularity Mismatch: Hierarchical Preference Learning for Long-Horizon LLM Agents](solving_the_granularity_mismatch_hierarchical_preference_learning_for_long-horiz.md)
- [\[ICLR 2026\] MobileRL: Online Agentic Reinforcement Learning for Mobile GUI Agents](mobilerl_online_agentic_reinforcement_learning_for_mobile_gui_agents.md)
- [\[ICML 2026\] From Player to Master: Enhancing Test-Time Learning of LLM Agents via Reinforcement Learning over Memory](../../ICML2026/llm_agent/from_player_to_master_enhancing_test-time_learning_of_llm_agents_via_reinforceme.md)
- [\[ICLR 2026\] FingerTip 20K: A Benchmark for Proactive and Personalized Mobile LLM Agents](fingertip_20k_a_benchmark_for_proactive_and_personalized_mobile_llm_agents.md)
- [\[ICLR 2026\] Aria: an Agent for Retrieval and Iterative Auto-Formalization via Dependency Graph](aria_an_agent_for_retrieval_and_iterative_auto-formalization_via_dependency_grap.md)

</div>

<!-- RELATED:END -->

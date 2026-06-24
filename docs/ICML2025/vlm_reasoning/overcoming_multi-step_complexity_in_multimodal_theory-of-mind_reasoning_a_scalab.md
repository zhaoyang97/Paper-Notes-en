---
title: >-
  [Paper Note] Overcoming Multi-step Complexity in Multimodal Theory-of-Mind Reasoning: A Scalable Bayesian Planner
description: >-
  [ICML2025 Spotlight][VLM Reasoning][Theory-of-Mind] Proposes a scalable Bayesian Theory-of-Mind (ToM) planner that decomposes multi-step reasoning into step-by-step Bayesian updates. By leveraging a weak-to-strong control mechanism, it transfers specialized ToM capabilities from smaller models to large language models (up to 405B), outperforming the Prev. SOTA by 4.6% on multimodal ToM benchmarks.
tags:
  - "ICML2025 Spotlight"
  - "VLM Reasoning"
  - "Theory-of-Mind"
  - "Bayesian Inverse Planning"
  - "Weak-to-Strong Control"
  - "multimodal reasoning"
  - "Theory of Mind"
date: 2026-05-08
content_hash: 16ddcbe1633f419d
---

# Overcoming Multi-step Complexity in Multimodal Theory-of-Mind Reasoning: A Scalable Bayesian Planner

**Conference**: ICML2025 Spotlight  
**arXiv**: [2506.01301](https://arxiv.org/abs/2506.01301)  
**Author**: Chunhui Zhang, Zhongyu Ouyang, Kwonjoon Lee, Nakul Agarwal, Sean Dae Houlihan, Soroush Vosoughi, Shao-Yuan Lo  
**Code**: TBD  
**Area**: Multimodal VLM  
**Keywords**: Theory-of-Mind, Bayesian Inverse Planning, Weak-to-Strong Control, multimodal reasoning, Theory of Mind  

## TL;DR

Proposes a scalable Bayesian Theory-of-Mind (ToM) planner that decomposes multi-step reasoning into step-by-step Bayesian updates. By leveraging a weak-to-strong control mechanism, it transfers specialized ToM capabilities from smaller models to large language models (up to 405B), outperforming the Prev. SOTA by 4.6% on multimodal ToM benchmarks.

## Background & Motivation

### Core Challenges of Theory-of-Mind

Theory-of-Mind (ToM) is the cornerstone of human social cognition, enabling individuals to infer others' beliefs, desires, and intentions. In AI, ToM tasks require models to perceive observable signals (e.g., actions, visual contexts) to predict an agent's goals and belief states.

Existing approaches primarily follow two routes:

- **Structured Planning Methods**: Design structured workflows utilizing ToM-specific priors (Baker et al., 2017; Jara-Ettinger, 2019; Shu et al., 2021).
- **Model Fine-Tuning Methods**: Integrate ToM priors into language models through specialized training (Rabinowitz et al., 2018; Sclar et al., 2022; Jin et al., 2024).

### Scalability Bottlenecks in Multimodal Environments

The paper reveals two fundamental issues through VirtualHome simulator experiments:

- **Upper Limit on Reasoning Boundaries**: As the number of planning steps increases, the performance of CoT reasoning and o1/r1-like test-time scaling methods plateaus with diminishing marginal returns. Even with test-time scaling methods (e.g., o1-mini, CoT), smaller models (Llama3.1-8B, 70B) experience a rapid decline in accuracy during multi-step planning.
- **World Knowledge Dependence on Model Scale**: Multimodal ToM reasoning is not a closed logical inference task; rather, it requires rich social and world knowledge. Studies show that only large-scale models (e.g., Llama3.1-405B) can maintain performance in multi-step planning.

These two findings indicate that pure fine-tuning or test-time scaling alone is insufficient to enhance the scalability of ToM reasoning. A dual approach combining structured frameworks with large-scale models is required.

## Method

### Overall Architecture: Scalable Bayesian ToM Planner

The proposed method consists of two core components:

```
┌─────────────────────────────────────────────────┐
│        Scalable Bayesian ToM Planner            │
├─────────────────────────────────────────────────┤
│                                                 │
│  Component 1: Bayesian Inverse Planning (BIP)   │
│  ┌───────────────────────────────────────────┐  │
│  │ Multimodal ToM → Step-by-Step Bayesian     │  │
│  │ Updates                                   │  │
│  │ • State transitions                       │  │
│  │ • Belief updates                          │  │
│  │ • Action likelihoods                      │  │
│  └───────────────────────────────────────────┘  │
│                                                 │
│  Component 2: Weak-to-Strong Control            │
│  ┌───────────────────────────────────────────┐  │
│  │ Small Model (Specialized Training) →      │  │
│  │ Large Model (Knowledge Integration)       │  │
│  │ • Small LM: Specialized in ToM Likelihood │  │
│  │ • Large LM (7B→405B): World Knowledge +   │  │
│  │   Bayesian Inference                      │  │
│  └───────────────────────────────────────────┘  │
│                                                 │
└─────────────────────────────────────────────────┘
```

### Key Design 1: Bayesian Inverse Planning (BIP)

The core idea of BIP is to decompose complex multimodal ToM reasoning into modular, step-by-step Bayesian updates. Specifically:

- **Behavior Modeling via the POMDP Framework**: Defined as the tuple $\langle S, A, T, G, R, \Omega, O, \gamma \rangle$, where $s^t$ represents the state, $a^t$ the action, $T$ the state transition probability, $g$ the goal, $R$ the reward function, and $o^t$ the observation.
- **Dynamic Belief Updates**: The agent's belief $b(s)$ is a probability distribution over states, which is dynamically updated as behavior evolves.
- **Step-by-step Decomposition**: The original end-to-end multi-step reasoning is split into independent sub-modules (state transitions, belief updates, action likelihoods). Each step refines hypotheses iteratively through Bayes' rule, ensuring the problem remains tractable even in complex environments.

The advantage of this decomposition strategy is that even as the number of task steps increases, the reasoning complexity of each step remains manageable, thereby overcoming the reasoning boundary limits of traditional approaches.

### Key Design 2: Weak-to-Strong Control

This is the core innovation of this work. Prior methods (e.g., Jin et al., 2024) rely on small LMs for likelihood estimation, but smaller models have limited world knowledge capacity, resulting in poor generalization in rich ToM scenarios.

Mechanism of Weak-to-Strong Control:

1. **Small Model Specialization**: Through post-training, a smaller LM is specialized in specific ToM tasks (such as likelihood estimation) to learn specialized behavior patterns for ToM reasoning.
2. **Behavior Transfer to Large Models**: The ToM reasoning behaviors learned by the small model are transferred to a larger LM (scaling from 7B to 405B).
3. **Large Model as the Main Policy Model**: The large model acts as the primary reasoner, leveraging its rich pre-trained world knowledge while maintaining Bayesian consistency through the transferred ToM behaviors.
4. **Theoretical Guarantee**: The effectiveness of this approach is formally proven in Theorem 1 using KL divergence analysis.

This design achieves the best of both worlds: the specialized capability of small models combined with the extensive knowledge of large models.

### Key Design 3: Multimodal Signal Integration

ToM environments require models to synthesize information across multiple modalities:

- **Visual Information**: Visual representations of environmental scenes, object locations, and agent actions.
- **Textual Information**: Action descriptions, task instructions, and contextual explanations.
- **Contextual Information**: Historical interactions, temporal relations, and social norms.

The model must integrate these multimodal cues into coherent mental state inferences, which is a major strength of the Bayesian framework—naturally fusing evidence from diverse sources through probabilistic reasoning.

## Key Experimental Results

### Main Results: Multimodal ToM Benchmarks

| Method Category | Representative Method | Multi-step Planning Performance | Scalability |
|---|---|---|---|
| CoT Reasoning | Chain-of-Thought | Accuracy drops as steps increase | Poor |
| Test-time Scaling | o1-mini | Provides incremental improvement but unsustainable | Moderate |
| Small Model Fine-tuning | Fine-tuned experts | Limited by reasoning boundaries | Poor |
| Small Model (Llama3.1-8B) | Base Reasoning | Rapid degradation | Poor |
| Medium Model (Llama3.1-70B) | Base Reasoning | Moderate degradation | Moderate |
| Large Model (Llama3.1-405B) | Base Reasoning | Maintains performance | Good |
| **Ours** | **Bayesian ToM Planner** | **SOTA + 4.6%** | **Best** |

### Performance Comparison Across Model Scales

| Model Scale | Base Reasoning Ability | + Ours (BIP) | + Weak-to-Strong Control | Improvement Margin |
|---|---|---|---|---|
| 7B | Low | Moderate improvement | Significant improvement | Large |
| 70B | Medium | Noticeable improvement | Further improvement | Medium |
| 405B | High | Steady improvement | Optimal performance | SOTA |

### Key Quantitative Results

- In multimodal ToM benchmarks, the overall accuracy improved by **4.6%** over the state-of-the-art (SOTA).
- The approach remains effective in **unseen scenarios**, validating its generalization ability.
- As planning steps increase, the performance degradation of the proposed method is significantly lower than that of all baselines.

## Highlights & Insights

1. **Precise Problem Identification**: Through VirtualHome experiments, the paper clearly reveals two fundamental bottlenecks in multi-step ToM reasoning (reasoning boundaries and dependency on world knowledge), providing explicit guidance for system design.

2. **Weak-to-Strong Control as the Core Innovation**: Unlike simple model distillation or knowledge transfer, the weak-to-strong control mechanism allows smaller models to specialize in ToM likelihood estimation, and then injects this "specialized behavior" into larger models. This balances specialized capability with general knowledge, a design paradigm that can be generalized to other tasks requiring both deep specialization and broad knowledge.

3. **Mitigating Reasoning Bottlenecks via Bayesian Decomposition**: Breaking down end-to-end multi-step reasoning into modular Bayesian update steps maintains manageable complexity at each step, successfully pushing past the constraints of typical reasoning boundaries.

4. **Dual Validation via Theory and Experiment**: Theorem 1 (utilizing KL divergence analysis) provides theoretical guarantees, complemented by empirical validation across benchmarks and unseen scenarios.

5. **Reflections on Test-time Scaling**: The paper notes that test-time scaling methods such as CoT and o1 exhibit fundamental limitations in ToM contexts. This finding offers valuable insights into understanding the boundaries of reasoning scaling.

## Limitations & Future Work

1. **High Computational Cost**: The approach requires maintaining both smaller models (for specialized training) and large models (for inference execution). The inference overhead of 405B-scale models is substantial, limiting practical deployment scenarios.

2. **Relatively Constrained Experimental Environments**: The evaluation is primarily based on the VirtualHome simulator. Generalization to more complex and ambiguous real-world social scenarios (such as sarcasm, metaphors, and cultural differences) remains to be verified.

3. **Reliance on Large Model World Knowledge**: The system’s efficacy fundamentally depends on the quality of the large model's pre-trained knowledge. ToM reasoning in niche domains or specific cultural contexts may be limited.

4. **Assumptions and Constraints of the Bayesian Framework**: The POMDP framework requires explicitly defined state spaces, action spaces, and transition functions, which may be unnatural or difficult to specify in open-domain scenarios.

5. **Theoretical Boundaries of Weak-to-Strong Transfer**: Although Theorem 1 provides a KL divergence analysis, how transfer efficiency scales with model capacity disparities and whether an optimal size ratio between small and large models exists has not been fully explored.

## Related Work & Insights

### Theory of Mind Modeling

- **Classical Bayesian Methods**: Baker et al. (2007, 2009, 2017) proposed Bayesian Inverse Planning, with Shum et al. (2019) extending it to more complex scenarios.
- **LM-Based Methods**: Jin et al. (2024) integrated Bayesian inference with language models, though they remain restricted by the capacity of smaller models.
- **Our Contribution**: Represents the first attempt to scale Bayesian ToM reasoning up to a 405B scale, successfully bypassing the generalization bottlenecks of smaller models.

### Test-time Scaling

- CoT (Chain-of-Thought) and o1/r1 systems are effective in closed logical tasks.
- This work reveals that such methods are constrained by reasoning boundaries in multimodal ToM.
- **Insight**: For complex reasoning tasks requiring rich world knowledge, scaling model parameters may be more critical than scaling computational reasoning steps.

### Weak-to-Strong Learning

- The proposed weak-to-strong control paradigm is related to, yet distinct from, weak-to-strong generalization (OpenAI, 2023).
- Instead of using weak supervision to train stronger models, this work transfers the specialized behaviors of a smaller model to a larger model.
- **Insight**: Decoupling and combining specialized performance with generalized capabilities represents a valuable direction for future research.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The weak-to-strong control mechanism is a novel design, and scaling Bayesian ToM reasoning to the 405B level is a pioneering attempt.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive evaluations across multimodal benchmarks, validation on unseen scenarios, and extensive ablations across various model scales.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear motivational analysis, compelling issue visualization in Fig. 1, and well-structured methodological descriptions.
- **Value**: ⭐⭐⭐⭐ — Highly valuable for the study of scalability in multimodal ToM reasoning; the weak-to-strong control paradigm is widely generalizable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MindPower: Enabling Theory-of-Mind Reasoning in VLM-based Embodied Agents](../../CVPR2026/vlm_reasoning/mindpower_enabling_theoryofmind_reasoning_in_vlmba.md)
- [\[ICML 2026\] From Shortcuts to Reasoning: Robust Post-Training of Theory of Mind with Reinforcement Learning](../../ICML2026/vlm_reasoning/from_shortcuts_to_reasoning_robust_post-training_of_theory_of_mind_with_reinforc.md)
- [\[ICLR 2026\] Reasoning-Aligned Perception Decoupling for Scalable Multi-modal Reasoning](../../ICLR2026/vlm_reasoning/reasoning-aligned_perception_decoupling_for_scalable_multi-modal_reasoning.md)
- [\[NeurIPS 2025\] Can Multi-Modal LLMs Provide Live Step-by-Step Task Guidance?](../../NeurIPS2025/vlm_reasoning/can_multi-modal_llms_provide_live_step-by-step_task_guidance.md)
- [\[ICCV 2025\] ToolVQA: A Dataset for Multi-step Reasoning VQA with External Tools](../../ICCV2025/vlm_reasoning/toolvqa_a_dataset_for_multistep_reasoning_vqa_with_external.md)

</div>

<!-- RELATED:END -->

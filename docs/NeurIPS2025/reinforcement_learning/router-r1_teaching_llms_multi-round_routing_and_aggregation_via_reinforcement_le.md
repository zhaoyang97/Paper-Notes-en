---
title: >-
  [Paper Note] Router-R1: Teaching LLMs Multi-Round Routing and Aggregation via Reinforcement Learning
description: >-
  [NeurIPS 2025][Reinforcement Learning][LLM routing] Router-R1 frames multi-LLM routing and aggregation as a sequential decision-making process, employing an LLM itself as the router to interleave *think* and *route* actions. Trained via PPO with a triple reward covering format, correctness, and cost, Router-R1 outperforms all router baselines across 7 QA benchmarks and generalizes to previously unseen LLMs.
tags:
  - NeurIPS 2025
  - Reinforcement Learning
  - LLM routing
  - multi-round interaction
  - model coordination
  - cost optimization
date: 2026-05-08
content_hash: fe931c61f4d11b3d
---

# Router-R1: Teaching LLMs Multi-Round Routing and Aggregation via Reinforcement Learning

**Conference**: NeurIPS 2025
**arXiv**: [2506.09033](https://arxiv.org/abs/2506.09033)
**Code**: [GitHub](https://github.com/ulab-uiuc/Router-R1)
**Area**: Reinforcement Learning
**Keywords**: LLM routing, multi-round interaction, reinforcement learning, model coordination, cost optimization

## TL;DR

Router-R1 frames multi-LLM routing and aggregation as a sequential decision-making process, employing an LLM itself as the router to interleave *think* and *route* actions. Trained via PPO with a triple reward covering format, correctness, and cost, Router-R1 outperforms all router baselines across 7 QA benchmarks and generalizes to previously unseen LLMs.

## Background & Motivation

The proliferation of LLMs has given rise to LLM routers that dynamically select the most suitable model for a given query. However, existing routers suffer from fundamental limitations:

**Single-round one-to-one mapping**: Current methods assign each query to a single model in a single decision step, failing to exploit the complementary strengths of multiple LLMs. For complex tasks such as multi-hop QA, a single model is often insufficient.

**Non-differentiable discrete decisions**: Selecting which LLM to invoke at each step is a discrete operation that cannot be optimized end-to-end via backpropagation. Although gradient-based approaches exist for single-step routing, extending them to multi-round selection and aggregation quickly becomes intractable.

**Absence of interleaved reasoning and routing**: Complex tasks require alternating between internal deliberation and targeted model selection to iteratively refine answers—a capability beyond single-step selection.

Router-R1 addresses these issues by instantiating a capable LLM as the router itself, training it via RL to learn when to reason internally, when to invoke an external model, which model to invoke, and how to integrate the returned results.

## Method

### Overall Architecture

Router-R1 models LLM coordination as a sequential decision-making problem. At each step, the routing LLM selects one of two action types:
- **Think**: Internal reasoning deliberation without invoking any external model.
- **Route**: Select a specific LLM from the routing pool, dispatch a sub-query, and incorporate the response into the context before continuing.

The final answer is wrapped in an `<answer>` tag, with a maximum of 4 routing rounds permitted.

### Key Designs

1. **LLM-as-Router Design**:
   The router is instantiated as a reasoning-capable LLM (e.g., Qwen2.5-3B-Instruct), leveraging its inherent reasoning capacity for extended deliberation and targeted model selection. The general optimization objective is:
   $$\max_\pi \mathbb{E}_{x \sim D, y \sim \pi(\cdot|x;\mathcal{P})} \left[ r_\phi(x,y) - \beta \log \frac{\pi(y|x;\mathcal{P})}{\pi_{\text{ref}}(y|x;\mathcal{P})} \right]$$
   where $\mathcal{P}$ denotes the LLM routing pool and $y$ is the generated sequence comprising thinking and routing actions.

2. **Triple Reward Function Design**:

   - **Format Reward $\mathbf{R}_{\text{format}}$**: Assigns $-1$ if the output violates the predefined format, otherwise $0$. A hierarchical design is adopted—non-compliant format zeroes out all other rewards.
   - **Outcome Reward $\mathbf{R}_{\text{outcome}}$**: Extracts the prediction from the `<answer>` tag and performs exact match (EM) against the ground truth; 1 for match, 0 otherwise.
   - **Cost Reward $\mathbf{R}_{\text{cost}}$**: Inversely proportional to the candidate LLM's parameter count and the number of output tokens: $\mathbf{R}_{\text{cost}} \propto -m(P_{\text{LLM}}) \cdot T_{\text{out}}$, normalized to $[0, 1]$.

   Total reward: $r_\phi(x,y) = \mathbf{R}_{\text{format}} + (1-\alpha)\mathbf{R}_{\text{outcome}} + \alpha\mathbf{R}_{\text{cost}}$

3. **Descriptor-Based Generalization Mechanism**:
   Routing decisions are conditioned on lightweight model descriptors (pricing, latency, example performance) rather than internal model representations. At inference time, introducing a new model's descriptors in the prompt enables zero-shot generalization to unseen LLMs without retraining.

4. **Multi-Round Interaction Training Paradigm**:
   A `<route>` tag detected in the generated sequence triggers routing—dispatching the sub-query to the specified LLM and inserting the response back into the context. External responses (enclosed in `<information>` tags) are excluded from loss computation. For simple queries, the model may answer directly using internal knowledge.

### Loss & Training

- PPO is used as the core RL algorithm; batch size 64, up to 225 training steps.
- Training data: 7K samples each from NQ and HotpotQA (14K total), deliberately kept small to validate data efficiency.
- LLM routing pool: 6 models — Qwen2.5-7B, LLaMA-3.1-8B, LLaMA-3.1-70B, Mistral-7B, Mixtral-8x22B, and Gemma-2-27B.
- Hierarchical reward priority: format → correctness → cost, ensuring training stability.
- In the main experiments $\alpha=0.0$ (cost constraint disabled); the cost analysis explores $\alpha \in \{0.6, 0.7, 0.8, 0.9\}$.

## Key Experimental Results

### Main Results (Exact Match on 7 QA Benchmarks, Qwen2.5-3B Backbone)

| Method | NQ† | TriviaQA | PopQA | HpQA† | 2wiki | Musique | Bamb | Avg |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Direct | 0.092 | 0.260 | 0.122 | 0.140 | 0.266 | 0.026 | 0.040 | 0.135 |
| Search-R1 | 0.328 | 0.510 | 0.324 | 0.236 | 0.278 | 0.090 | 0.272 | 0.291 |
| Largest LLM | 0.296 | 0.578 | 0.354 | 0.278 | 0.274 | 0.104 | 0.480 | 0.338 |
| RouterDC | 0.278 | 0.592 | 0.282 | 0.244 | 0.218 | 0.080 | 0.504 | 0.314 |
| GraphRouter | 0.276 | 0.586 | 0.280 | 0.234 | 0.180 | 0.076 | 0.448 | 0.297 |
| **Router-R1-Qwen** | **0.388** | **0.706** | **0.384** | **0.352** | **0.434** | **0.138** | **0.512** | **0.416** |

Router-R1 surpasses all baselines on all 7 datasets, with an average EM of 0.416 substantially exceeding the strongest baseline (Largest LLM, 0.338, +23%).

### Generalization (Adding Previously Unseen LLMs)

| Method | NQ (EM) | TriviaQA (EM) | PopQA (EM) | HpQA (EM) | Avg (EM) |
|--------|:---:|:---:|:---:|:---:|:---:|
| Router-R1-Qwen | 0.388 | 0.706 | 0.384 | 0.352 | 0.458 |
| Router-R1-Qwen‡ (+2 new models) | 0.382 | 0.722 | 0.402 | 0.346 | 0.463 |
| GraphRouter | 0.276 | 0.586 | 0.280 | 0.234 | 0.344 |
| GraphRouter‡ | 0.282 | 0.594 | 0.276 | 0.228 | 0.345 |

After adding Palmyra-Creative-122B and ChatQA-1.5-8B without retraining, Router-R1 even improves on TriviaQA and PopQA. Baseline methods show limited generalization gain.

### Ablation Study / Cost Analysis

| Cost Coefficient $\alpha$ | NQ (EM) | PopQA (EM) | HpQA (EM) | 2wiki (EM) | Cost Reward Trend |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.0 | 0.388 | 0.384 | 0.352 | 0.434 | Lowest |
| 0.6 | Slight drop | Slight drop | Slight drop | Slight drop | Increasing |
| 0.8 | Notable drop | Notable drop | Notable drop | Notable drop | Significant increase |
| 0.9 | Large drop | Large drop | Large drop | Large drop | Highest |

Emergent behavior: under high cost coefficients, Router-R1 learns to query smaller models first and escalate to larger models only when necessary.

### Key Findings

- **Adaptive routing for multi-hop QA**: Router-R1 issues significantly more API calls on multi-hop tasks than on simple QA, implicitly assessing task difficulty.
- **Fast convergence**: Convergence is achieved within approximately 100 steps (reward increase accompanied by policy entropy decrease).
- **Format reward is critical**: Removing the format reward leads to training instability and frequent generation of incoherent text.
- **Effective training with only 14K samples**: Router-R1 acquires transferable routing and reasoning strategies from a minimal dataset.
- For simple queries, Router-R1 refrains from invoking any external model and answers entirely from internal knowledge.

## Highlights & Insights

- **Paradigm shift—LLM as router**: Routing is elevated from "classifier selects model" to "reasoner coordinates models."
- **Interleaved thinking and routing**: Autonomous judgment of "when external help is needed" emerges naturally.
- **Descriptor-based generalization**: New models can be integrated via textual descriptors alone, with no retraining required—well-suited to the rapidly evolving LLM ecosystem.
- **Hierarchical reward design**: Elegantly addresses reward hacking and training instability.
- **Emergent cost-aware routing**: The strategy of preferring smaller models with large-model fallback arises spontaneously.

## Limitations & Future Work

- The 4-round routing cap limits handling of highly complex tasks.
- The cost coefficient $\alpha$ requires manual tuning; an adaptive mechanism is absent.
- The routing pool is limited to 6 models; performance with a larger pool remains unknown.
- Excluding candidate LLM outputs from loss computation may constrain the optimization of routing policies.
- Evaluation is restricted to QA tasks; effectiveness on code generation, creative writing, and other domains is unverified.

## Related Work & Insights

- **Search-R1**: RL-driven search engine invocation; Router-R1 generalizes this paradigm to multi-LLM invocation.
- **GraphRouter, RouterDC**: Conventional single-round router baselines; Router-R1 demonstrates the advantages of multi-round routing.
- **DeepSeek-R1**: Source of inspiration for formatted output and rule-based reward design.
- **FrugalGPT**: Pioneer in cascade-based cost optimization; Router-R1 achieves more flexible cost–performance trade-offs via RL.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Modeling multi-round routing and aggregation as sequential decision-making constitutes a meaningful new paradigm.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 7 benchmarks, cost analysis, generalization experiments, training dynamics analysis, and two backbone models.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, concise formulations, and well-designed experiments.
- **Value**: ⭐⭐⭐⭐ Opens a new direction for LLM routing; cost-aware routing and zero-shot generalization offer practical applicability.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Routing, Cascades, and User Choice for LLMs](../../ICLR2026/reinforcement_learning/routing_cascades_and_user_choice_for_llms.md)
- [\[NeurIPS 2025\] MTL-KD: Multi-Task Learning Via Knowledge Distillation for Generalizable Neural Vehicle Routing Solver](mtl-kd_multi-task_learning_via_knowledge_distillation_for_generalizable_neural_v.md)
- [\[NeurIPS 2025\] Robot-R1: Reinforcement Learning for Enhanced Embodied Reasoning in Robotics](robot-r1_reinforcement_learning_for_enhanced_embodied_reasoning_in_robotics.md)
- [\[ICLR 2026\] Shop-R1: Rewarding LLMs to Simulate Human Behavior in Online Shopping via Reinforcement Learning](../../ICLR2026/reinforcement_learning/shop-r1_rewarding_llms_to_simulate_human_behavior_in_online_shopping_via_reinfor.md)
- [\[NeurIPS 2025\] Teaching Language Models to Evolve with Users: Dynamic Profile Modeling for Personalized Alignment](teaching_language_models_to_evolve_with_users_dynamic_profile_modeling_for_perso.md)

<!-- RELATED:END -->

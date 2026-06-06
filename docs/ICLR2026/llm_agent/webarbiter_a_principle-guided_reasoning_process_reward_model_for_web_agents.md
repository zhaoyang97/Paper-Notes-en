---
title: >-
  [Paper Note] WebArbiter: A Principle-Guided Reasoning Process Reward Model for Web Agents
description: >-
  [ICLR2026][LLM Agent][Web Agent] WebArbiter proposes a reasoning-first, principle-guided Process Reward Model (WebPRM) that formulates reward modeling as a text generation task. Through a two-stage training pipeline of r…
tags:
  - "ICLR2026"
  - "LLM Agent"
  - "Web Agent"
  - "Process Reward Model"
  - "Reasoning-First"
  - "Principle-Guided"
  - "Reinforcement Learning"
  - "Reasoning Distillation"
date: 2026-05-08
content_hash: 8a5af68ba06d353c
---

# WebArbiter: A Principle-Guided Reasoning Process Reward Model for Web Agents

**Conference**: ICLR2026  
**arXiv**: [2601.21872](https://arxiv.org/abs/2601.21872)  
**Code**: [WebArbiter Project Page](https://webarbiter.github.io/)  
**Area**: LLM Agent  
**Keywords**: Web Agent, Process Reward Model, Reasoning-First, Principle-Guided, Reinforcement Learning, Reasoning Distillation

## TL;DR
WebArbiter proposes a reasoning-first, principle-guided Process Reward Model (WebPRM) that formulates reward modeling as a text generation task. Through a two-stage training pipeline of reasoning distillation followed by reinforcement learning, a 7B model achieves performance surpassing GPT-5 by 9.1 percentage points on WebPRMBench.

## Background & Motivation
- Web agents involve long-horizon, multi-step decision-making with irreversible actions, necessitating step-level process supervision.
- Outcome Reward Models (ORMs) provide only sparse and delayed feedback, and may incorrectly judge erroneous trajectories as successful.
- Existing WebPRM approaches exhibit notable limitations:
    - **Scalar WebPRM**: Compresses progress into coarse-grained scores, lacking interpretability and grounding.
    - **Checklist WebPRM**: Relies on brittle template matching, failing under layout or semantic variations.
    - **LLM-as-Judge**: High cost, poor scalability, and prone to hallucination.
- Core problem: How to construct a WebPRM that is both interpretable and robust, capable of resisting spurious correlations while providing auditable reasoning chains.

## Method

### 1. Problem Formulation
Web navigation is modeled as a POMDP: $\mathcal{E} = (\mathcal{S}, \mathcal{A}, \mathcal{O})$

Given task instruction $\mathcal{I}$, current observation $o_p$, history of actions and reasoning $(a_{<p}, c_{<p})$, and a candidate action pair $(a_p^1, c_p^1)$ and $(a_p^2, c_p^2)$, WebArbiter generates a structured argument $j = (j_1, \ldots, j_L)$ and produces a preference verdict $\hat{y}$.

Compact input representation:
$$x = (\mathcal{I}, o_p, a_{<p}, c_{<p}, (a_p^1, c_p^1), (a_p^2, c_p^2))$$

Autoregressive argument generation:
$$\pi_\theta(j | x) = \prod_{l=1}^{L} \pi_\theta(j_l | x, j_{<l})$$

### 2. Training Data Construction
Based on the WebPRM Collection (Chae et al., 2025):
- Each instance includes an instruction, observation sequence, and expert-annotated trajectory.
- Positive actions are drawn from expert demonstrations $A^+$; negative actions from rejected trajectories $A^-$.
- Converted into pairwise preference samples for training.

### 3. Two-Stage Training Pipeline

**Overall objective**:
$$\max_{\pi_\theta} \mathbb{E}_{(x,y) \sim \mathcal{D}_{\text{Train}}, \hat{y} \sim \pi_\theta(j|x)} [\mathbb{1}(\hat{y} = y)]$$

**Stage 1: Reasoning Distillation**
- A stronger teacher model (o3) is used to generate principle-guided structured arguments.
- Argumentation flow: derive task-specific principles from instruction and state → ground principles to the page → compare candidate actions → output preference.
- Distillation loss:

$$\mathcal{L}_{\text{SFT}}(\theta) = -\frac{1}{K} \sum_{i=1}^{K} \sum_{l=1}^{L_i} \log \pi_\theta(\hat{j}_l^{(i)} | x^{(i)}, \hat{j}_{<l}^{(i)})$$

- Distillation training uses 10K samples.

**Stage 2: Reinforcement Learning**
- Verifiable rewards align verdicts with correctness signals.
- Reward function: $R(x, \hat{y}) \in \{-1, 1\}$ (based on whether the verdict matches the ground truth).
- The distilled model serves as the reference policy $\pi_{\text{ref}}$.

RL optimization objective (using GRPO):
$$\mathcal{L}_{\text{RL}}(\theta) = \max_{\pi_\theta} \mathbb{E}_{(x,y) \sim \mathcal{D}_{\text{RL}}, \hat{y} \sim \pi_\theta(j|x)} [R(x, \hat{y})] - \beta \mathbb{D}_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$$

- RL training uses the remaining 20K samples.

### 4. Key Designs
- **Principle-guided**: Principles are dynamically derived from user intent and current state, rather than relying on fixed checklist templates.
- **Reasoning-first**: Structured reasoning arguments are generated prior to the verdict, making judgments auditable.
- The model is based on Qwen2.5-3B/7B-Instruct with LoRA fine-tuning.

## WebPRMBench

### Data Distribution
- Spans 4 web environments: Mind2Web, WebArena, AssistantBench, and WorkArena.
- Contains 1,150 step-level preference instances (each with 1 correct and 4 rejected actions).

### Evaluation Metrics
**Pairwise Accuracy**:
$$\text{Acc}_{\text{Pairwise}} = \frac{1}{|\mathcal{D}|} \sum_{(a^+, a^-)} \mathbb{1}[\pi_\theta(a^+) \succ \pi_\theta(a^-)]$$

**Best-of-N Accuracy (BoN Acc)**: More stringent; requires the correct action to simultaneously outrank all 4 distractors:
$$\text{Acc}_{\text{BoN}} = \frac{1}{|\mathcal{D}|} \sum_{i=1}^{|\mathcal{D}|} \prod_{q=1}^{4} \mathbb{1}[\pi_\theta(a_i^+) \succ \pi_\theta(a_i^{-_q})]$$

## Key Experimental Results

### Main Results on WebPRMBench

| Model | Mind2Web BoN | WebArena BoN | AssistantBench BoN | WorkArena BoN | Avg BoN |
|------|-------------|-------------|-------------------|-------------|---------|
| GPT-4o | 52.62 | 66.67 | 66.67 | 55.19 | 60.29 |
| GPT-5 | 62.39 | 71.64 | 63.33 | 64.62 | 65.50 |
| Claude-3.7-Sonnet | 57.90 | 64.10 | 61.30 | 60.60 | 60.98 |
| DeepSeek-R1 | 57.37 | 60.21 | 56.18 | 63.89 | 59.41 |
| WebShepherd-8B | 73.69 | 43.88 | 30.00 | 25.53 | 43.28 |
| **WebArbiter-7B** | **89.53** | **68.66** | **70.00** | **70.19** | **74.60** |

WebArbiter-7B surpasses GPT-5 by **9.1 percentage points** and the previous SOTA WebShepherd-8B by **31.32 percentage points** in Avg BoN Acc.

### Ablation Study on Training Strategies

| Method | Mind2Web BoN | WebArena BoN | AssistantBench BoN | WorkArena BoN | Avg BoN |
|------|-------------|-------------|-------------------|-------------|---------|
| Instruct (base) | 39.18 | 42.79 | 53.33 | 35.85 | 42.78 |
| + Cold Start RL | 86.00 | 35.80 | 33.60 | 37.90 | 48.33 |
| + Cold Start RL + Principles | 88.00 | 46.30 | 48.90 | 51.80 | 58.75 |
| + SFT (w/o Principles) + RL | 94.34 | 41.50 | 40.20 | 44.60 | 55.16 |
| **WebArbiter (SFT + Principles + RL)** | **89.53** | **68.66** | **70.00** | **70.19** | **74.60** |

### Reward-Guided Search on WebArena-Lite
In reward-guided trajectory search, WebArbiter outperforms WebShepherd by up to **7.2 percentage points**.

## Key Findings

### 1. Cold-Start RL Is Unstable
- Applying RL directly to the Instruct model raises Mind2Web BoN to 86.00, but performance degrades in other environments.
- This indicates that RL without a reasoning distillation foundation generalizes poorly across environments.

### 2. Principle Guidance Is Critical
- Removing explicit principles while retaining reasoning arguments reduces Avg BoN Acc from 74.60 to 55.16 (−19.44).
- Principle guidance produces more grounded judgments that resist spurious correlations.

### 3. SFT Is a Necessary Prerequisite for RL
- Reasoning distillation provides a stable initialization for RL, with RL primarily acting as an amplifier.
- The combination of SFT + RL substantially outperforms either stage alone.

## Highlights & Insights
1. **Reasoning-first paradigm**: Shifts reward modeling from score prediction to auditable reasoning generation, greatly enhancing interpretability.
2. **Dynamic principle derivation**: Principles are inferred from task instructions and states rather than fixed templates, enabling strong adaptability.
3. **Robust cross-environment generalization**: Trained only on Mind2Web, WebArbiter achieves state-of-the-art performance across all 4 environments.
4. **Small model surpasses large models**: A 7B model outperforms GPT-5 and DeepSeek-R1.
5. **Complementary two-stage training**: Reasoning distillation and RL are mutually reinforcing.

## Limitations & Future Work
- Training data is limited to 30K samples from a single environment (Mind2Web); expanding to multi-environment training data may further improve performance.
- The current formulation supports only pairwise comparisons; multi-candidate settings require further investigation.
- Observations are represented as accessibility trees (text-based), with visual information not utilized.
- Reasoning generation introduces additional inference latency, requiring trade-offs in real-time deployment scenarios.
- Negative samples in WebPRMBench are model-generated, potentially introducing distributional bias.

## Related Work & Insights
- Compared to WebShepherd (checklist WebPRM): WebArbiter substantially dominates on unseen environments (WorkArena BoN: 70.19 vs. 25.53).
- Compared to scalar WebPRM (Miao et al., 2025): WebArbiter provides auditable reasoning chains rather than numerical scores.
- Compared to LLM-as-Judge: A 7B specialized model substantially outperforms the general-purpose GPT-5.
- Compared to the reasoning RM literature (Chen et al., 2025): This work is the first to apply reasoning reward models to the web agent domain.

### Broader Implications
- The principle-guided reasoning distillation paradigm is generalizable to other process reward modeling settings.
- The two-stage SFT → RL pipeline provides a useful reference for training verifiable reward models.
- WebPRMBench establishes a standardized evaluation framework for WebPRM research.
- Reasoning-first reward models can be integrated with search and planning algorithms to enable inference-time scaling.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (Novel reasoning-first and principle-guided WebPRM design; innovative two-stage training strategy)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (4-environment benchmark, diverse baselines, detailed ablations, real-world search validation)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, though dense notation requires careful reading)
- Value: ⭐⭐⭐⭐⭐ (7B model surpasses GPT-5; open-sourced WebPRMBench; significant contribution to the web agent community)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Process Reward Agents for Steering Knowledge-Intensive Reasoning](../../ICML2026/llm_agent/process_reward_agents_for_steering_knowledge-intensive_reasoning.md)
- [\[ACL 2026\] Exploring Reasoning Reward Model for Agents](../../ACL2026/llm_agent/exploring_reasoning_reward_model_for_agents.md)
- [\[ICLR 2026\] Web-CogReasoner: Towards Knowledge-Induced Cognitive Reasoning for Web Agents](web-cogreasoner_towards_knowledge-induced_cognitive_reasoning_for_web_agents.md)
- [\[ICLR 2026\] ST-WebAgentBench: A Benchmark for Evaluating Safety and Trustworthiness in Web Agents](st-webagentbench_a_benchmark_for_evaluating_safety_and_trustworthiness_in_web_ag.md)
- [\[NeurIPS 2025\] Web-Shepherd: Advancing PRMs for Reinforcing Web Agents](../../NeurIPS2025/llm_agent/web-shepherd_advancing_prms_for_reinforcing_web_agents.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Enhancing Persona Following at Decoding Time via Dynamic Importance-Guided Token Estimation for Role-Playing Agents
description: >-
  [ICLR 2026][LLM/NLP][Role-playing agents] This paper proposes Persona Dynamic Decoding (PDD), a framework that dynamically estimates the context-dependent importance of persona attributes via conditional mutual information and integrates importance scores into multi-objective reward-guided decoding, achieving training-free inference-time persona following.
tags:
  - ICLR 2026
  - LLM/NLP
  - Role-playing agents
  - persona following
  - inference-time alignment
  - conditional mutual information
  - multi-objective reward decoding
date: 2026-05-08
content_hash: 250decb3608d7e21
---

# Enhancing Persona Following at Decoding Time via Dynamic Importance-Guided Token Estimation for Role-Playing Agents

**Conference**: ICLR 2026
**arXiv**: [2603.01438](https://arxiv.org/abs/2603.01438)
**Code**: Not released
**Area**: LLM/NLP
**Keywords**: Role-playing agents, persona following, inference-time alignment, conditional mutual information, multi-objective reward decoding

## TL;DR

This paper proposes Persona Dynamic Decoding (PDD), a framework that dynamically estimates the context-dependent importance of persona attributes via conditional mutual information and integrates importance scores into multi-objective reward-guided decoding, achieving training-free inference-time persona following.

## Background & Motivation

Role-playing language agents (RPLAs) are increasingly important in sociological research (e.g., voting behavior analysis, rumor propagation dynamics), yet existing approaches face two core limitations:

1. **Insufficient dynamic adaptability**: Psychological research (e.g., the Cognitive-Affective Personality System, CAPS) indicates that the influence of personality on behavior is context-dependent—different persona attributes carry different salience across situations. However, existing methods (prompt engineering / fine-tuning) treat all attributes uniformly and cannot dynamically identify context-relevant persona attributes.
2. **Heavy data dependency**: Parametric methods (SFT/LoRA) require large-scale behavioral data, while social simulation involves diverse characters and complex personas, making data collection extremely costly.

Taxonomy of existing methods and their limitations:
- **Non-parametric methods** (Direct Prompting, ICL, RAG): rely on semantic recognition and fail to deeply understand persona attributes.
- **Parametric methods** (SFT, LoRA): require substantial computational resources and annotated data.
- Neither category achieves context-aware dynamic persona following.

## Method

### Overall Architecture

PDD consists of two core components:
1. **PIE (Persona Importance Estimation):** Self-supervisedly quantifies the context-dependent importance of persona attributes in a dynamic manner.
2. **PIA (Persona-Guided Inference-Time Alignment):** Transforms importance scores into weighted multi-objective rewards to modulate token generation probabilities at inference time.

### Key Designs

**PIE: Importance Estimation via Conditional Mutual Information**

The conditional mutual information of model output $Y$ with respect to a persona attribute $w_i$:

$$I(Y; w_i | T_i) = H(Y|T_i) - H(Y|w_i, T_i)$$

where $T_i = T \setminus \{w_i\}$ (the full prompt with attribute $w_i$ removed).

Approximation — the model-generated response $G = \pi_\theta(T)$ is used as a proxy for the unavailable ground truth:

$$I_i \triangleq \log \frac{\Pr(G \mid T)}{\Pr(G \mid T_i)}$$

Core insight:
- Intuitively, if removing a persona attribute causes a significant drop in model output probability, that attribute is critical for the current context.
- Theoretical guarantee: if the probabilities of the model generation $G$ and the ground truth $GT$ are positively correlated, then $I^{\text{model}}$ serves as a reliable proxy for $I^{\text{true}}$.

**PIA: Multi-Persona Inference-Time Alignment**

Stepwise reward for each attribute $w_i$:

$$r_i(T, y_{<t}) = \sum_{t'=t-1}^{t} \log \frac{\pi_\theta(y_{t'} | T, y_{<t'})}{\pi_\theta(y_{t'} | T_i, y_{<t'})}$$

Weighted multi-objective reward function:

$$R(T, y) = \sum_{i=1}^{n} I_i \cdot r_i(T, y)$$

**Normalized Reward (Key Innovation):**

$$R_{\text{norm}} = \frac{\sum_{i=1}^{n} I_i \cdot r_i(T, y)}{\|\mathbf{r}\|_2}$$

By the Cauchy–Schwarz inequality: $R_{\text{norm}} \leq \|\mathbf{I}\|_2$, with equality if and only if $\mathbf{r} \propto \mathbf{I}$. Maximizing $R_{\text{norm}}$ therefore incentivizes the per-attribute rewards to maintain a ranking consistent with the importance scores.

### Loss & Training

**KL-constrained RL objective:**

$$\max_{p_r} \; \mathbb{E}_{p_r} \left[ \frac{\sum_{i=1}^{n} I_i r_i(T, y)}{\|\mathbf{r}\|_2} - \beta D_{\text{KL}}(p_r \| \pi_\theta) \right]$$

**Optimal solution (per token):**

$$p_r(y_t | T, y_{<t}) = \frac{1}{Z(T, y_{<t})} \pi_\theta(y_t | T, y_{<t}) \exp\left(\frac{1}{\beta} R_{\text{norm}}(T, y_{<t})\right)$$

- **Completely training-free**: relies solely on log probabilities computed at inference time.
- Hyperparameter $\beta = 1.0$.
- In practice, the top-2 highest-importance attributes are aligned to balance fidelity and efficiency.
- Responses are generated via greedy decoding.

## Key Experimental Results

### Main Results

**General role-playing task — GPT-4o pairwise evaluation (Win%):**

| PDD vs. | CharacterEval (Qwen) | CharacterEval (LLaMA) | BEYOND DIALOGUE (Qwen) | BEYOND DIALOGUE (LLaMA) |
|---------|---------------------|---------------------|----------------------|----------------------|
| SP | 51.2% win | 52.5% win | 63.9% win | 56.2% win |
| PP | 48.7% win | 39.1% win | 43.0% win | 46.8% win |
| ICL | 65.3% win | 63.1% win | 60.9% win | 64.2% win |
| OPAD | 52.8% win | 48.2% win | 49.0% win | 47.6% win |

**CharacterEval automatic evaluation (CharacterRM metrics):**

| Model | Method | KE | KA | KH | PB | PU | Average |
|------|------|-----|-----|-----|-----|-----|---------|
| GPT-4o | PP | 2.58 | 3.02 | 2.99 | 2.83 | 2.91 | 2.87 |
| Qwen-7B | PDD | 2.25 | 2.93 | 2.99 | **3.08** | **3.01** | **2.85** |
| LLaMA-8B | PDD | 2.39 | 2.68 | **3.03** | 3.00 | 2.96 | **2.81** |

PDD on small open-source models is competitive with the commercial GPT-4o model.

**PERSONALITYBENCH Big Five personality trait evaluation (Qwen2.5-7B):**

| Trait | SP | PP | ICL | OPAD | PAS | NPTI | **PDD** |
|---------|------|------|------|------|------|------|---------|
| Agreeableness | 4.81 | 4.90 | 4.81 | 4.53 | 4.83 | 4.73 | **4.92** |
| Conscientiousness | 4.47 | 4.98 | 4.19 | 4.66 | 4.61 | 4.74 | **4.97** |
| Extroversion | 4.68 | 4.59 | 4.32 | 4.26 | 4.65 | 4.71 | **4.66** |
| Neuroticism | 3.02 | 3.45 | 3.12 | 3.79 | 3.74 | 3.39 | **3.54** |
| Openness | 4.56 | 4.75 | 4.67 | 4.44 | 4.61 | 4.83 | 4.75 |
| **Average** | 4.31 | 4.53 | 4.22 | 4.34 | 4.49 | 4.48 | **4.57±0.22** |

PDD achieves the highest average with the lowest variance (0.22 vs. 0.32–0.53 for other methods), indicating superior robustness.

### Ablation Study

**Reliability validation of PIE importance estimation:**

Evaluation is conducted across five dimensions (Context Relevance, Attribute Utility, Context Coverage, Attribute Independence, Ranking Consistency) using three LLM judges (DeepSeek-R1, GPT-4o, GPT-5) and human expert ratings:
- PDD achieves consistently strong scores across all dimensions (Likert 1–5 scale).
- Consistent importance distributions across different base models (Qwen/LLaMA) confirm cross-model stability.

**Context-aware visualization case study:**
- Scene 1 (Guo Furong and Lü Xiucai discussing martial arts): high weights assigned to personality traits and distinctive skills.
- Scene 2 (Guo Furong guiding Tong Xiangyu): high weights assigned to worldview and educational opinions.
- This confirms that PIE dynamically adjusts attribute weights according to context.

### Key Findings

1. **Inference-time alignment is effective**: PDD surpasses or matches training-based methods (NPTI, PAS) across multiple benchmarks without fine-tuning.
2. **Small models are highly competitive**: Open-source models with 7–8B parameters equipped with PDD achieve GPT-4o-level role-playing capability.
3. **The multi-objective normalized reward design is critical**: The Cauchy–Schwarz-incentivized ranking preservation ensures the hierarchical structure of persona attributes.
4. **Top-2 attribute alignment is the optimal efficiency–effectiveness trade-off**: Including more attributes increases computation with diminishing marginal returns.
5. **Statistical significance with $p < 0.05$** is satisfied across all Big Five personality dimensions.

## Highlights & Insights

1. **Theory-driven design**: Every step—from CAPS psychological theory to CMI information-theoretic quantification to Cauchy–Schwarz normalization—is grounded in formal justification.
2. **Zero-shot persona importance estimation**: No ground-truth supervision is required; attribute importance is quantified solely from the model's own log probability differences—the most elegant design choice.
3. **Normalization trick for multi-objective alignment**: Dividing by $\|\mathbf{r}\|_2$ ensures that the reward vector is aligned in direction with the importance vector, rather than being naively weighted.
4. **Comprehensive experimental design**: Chinese and English role-playing benchmarks + Big Five personality evaluation + human assessment + LLM-as-Judge + reward model.
5. **Training-free**: The method operates entirely at inference time and transfers to any character without requiring character-specific data.

## Limitations & Future Work

1. **Non-trivial inference overhead**: Computing conditional probabilities with and without each attribute per token requires $n+1$ forward passes, which may become a latency bottleneck in production.
2. **Aligning only the top-2 attributes** is an engineering compromise; complex characters may require simultaneous alignment of more attributes.
3. **Theoretical assumptions in CMI approximation**: The positive correlation assumption does not necessarily hold in all scenarios.
4. **Evaluation limitations**: LLM-as-Judge may itself exhibit preferences for certain persona expressions.
5. **Maintaining persona consistency in long conversations is unexplored**: Experiments are primarily conducted in single-turn or short dialogue settings.

## Related Work & Insights

- **CAPS theory** (Sherman et al., 2015): The Cognitive-Affective Personality System provides a psychological foundation for dynamic persona modeling.
- **OPAD** (Zhu et al., 2025a): Single-objective inference-time preference alignment; PDD extends this to multi-objective settings.
- **NPTI** (Deng et al., 2025): Neuron-level personality trait induction, requiring trained probes.
- **CharacterEval** (Tu et al., 2024): Chinese role-playing evaluation benchmark.
- Inspiration: Conditional mutual information can serve as a general-purpose attribute importance measure, extensible to other scenarios requiring dynamic attribute weighting (e.g., stylized writing, multi-constraint generation).

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of CMI importance estimation and normalized multi-objective rewards constitutes an original contribution.
- **Technical Depth**: ⭐⭐⭐⭐⭐ — Theoretical derivations and implementation details are rigorous; information theory and optimization theory are applied appropriately.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Three datasets + multiple baselines + human and automatic evaluation + ablation studies.
- **Practicality**: ⭐⭐⭐⭐ — Training-free inference-time solution is highly deployment-friendly.
- **Writing Quality**: ⭐⭐⭐⭐ — Mathematical derivations are clear; framework diagrams are highly informative.

**Overall**: ⭐⭐⭐⭐ (4/5) — A theoretically rigorous and elegantly designed inference-time persona alignment framework. CMI importance estimation and normalized multi-objective rewards are the standout contributions, demonstrating the competitiveness of training-free methods across multiple benchmarks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Enhancing Persona Following at Decoding Time via Dynamic Importance Estimation for Role-Playing Agents](enhancing_persona_following_at_decoding_time_via_dynamic_importance-guided_token.md)
- [\[ICLR 2026\] Fine-Grained Activation Steering: Steering Less, Achieving More](fine-grained_activation_steering_steering_less_achieving_more.md)
- [\[ICLR 2026\] GASP: Guided Asymmetric Self-Play For Coding LLMs](gasp_guided_asymmetric_self-play_for_coding_llms.md)
- [\[ICLR 2026\] Stopping Computation for Converged Tokens in Masked Diffusion-LM Decoding](stopping_computation_for_converged_tokens_in_masked_diffusion-lm_decoding.md)
- [\[ICLR 2026\] Rethinking Uncertainty Estimation in LLMs: A Principled Single-Sequence Measure](rethinking_uncertainty_estimation_in_llms_a_principled_single-sequence_measure.md)

</div>

<!-- RELATED:END -->

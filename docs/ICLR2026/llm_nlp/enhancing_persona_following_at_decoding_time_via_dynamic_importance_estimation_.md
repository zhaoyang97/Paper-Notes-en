---
title: >-
  [Paper Note] Enhancing Persona Following at Decoding Time via Dynamic Importance Estimation for Role-Playing Agents
description: >-
  [ICLR 2026][LLM/NLP][role-playing] This paper proposes the Persona Dynamic Decoding (PDD) framework, which dynamically estimates the context-dependent importance of persona attributes via conditional mutual information and guides decoding with a weighted multi-objective reward, enabling training-free, adaptive persona following at inference time.
tags:
  - ICLR 2026
  - LLM/NLP
  - role-playing
  - persona following
  - inference-time alignment
  - decoding strategy
  - conditional mutual information
date: 2026-05-08
content_hash: 83eadafa623d4910
---

# Enhancing Persona Following at Decoding Time via Dynamic Importance Estimation for Role-Playing Agents

**Conference**: ICLR 2026
**arXiv**: [2603.01438](https://arxiv.org/abs/2603.01438)
**Code**: None
**Area**: LLM/NLP
**Keywords**: role-playing, persona following, inference-time alignment, decoding strategy, conditional mutual information

## TL;DR

This paper proposes the Persona Dynamic Decoding (PDD) framework, which dynamically estimates the context-dependent importance of persona attributes via conditional mutual information and guides decoding with a weighted multi-objective reward, enabling training-free, adaptive persona following at inference time.

## Background & Motivation

1. **Surging demand for role-playing agents**: LLM-driven role-playing language agents (RPLAs) are widely used in sociological simulations (e.g., voting behavior analysis, rumor diffusion dynamics), requiring agents to strictly adhere to predefined persona profiles.

2. **Persona influence is dynamic**: The Cognitive-Affective Personality System (CAPS) in psychology posits that human behavior is not governed by fixed traits but by contextually activated attributes. LLM persona following should similarly exhibit scene-adaptive behavior.

3. **Prompt-based methods lack deep understanding**: Non-parametric approaches such as simple prompting (SP), in-context learning (ICL), and RAG rely on surface-level semantic recognition and cannot dynamically adjust behavioral patterns across different scenarios.

4. **Fine-tuning is prohibitively costly**: SFT/LoRA require substantial annotated data and computational resources, while the diversity of roles and scenarios in social simulation makes data collection extremely difficult.

5. **Existing methods cannot adapt dynamically**: Both prompting and fine-tuning approaches adopt static strategies for handling persona attributes, lacking the ability to identify which attributes are most relevant given the current dialogue context.

6. **Absence of theory-grounded persona quantification**: Most prior inference-time alignment work focuses on single-preference alignment, lacking a systematic framework that integrates psychological theory into multi-attribute persona modeling.

## Method

### Overall Architecture: Persona Dynamic Decoding (PDD)

- **Function**: Constructs a complete inference-time persona following framework comprising two core modules: Persona Importance Estimation (PIE) and Persona-Guided Inference-Time Alignment (PIA).
- **Design Motivation**: Without fine-tuning, the model must automatically identify salient persona attributes given the current dialogue context and steer token-level generation toward the target persona.
- **Mechanism**: Given a persona profile $P=\{w_1,\ldots,w_n\}$, context $C$, and query $x$, PIE first estimates the contextual importance $I_i$ of each attribute $w_i$; PIA then constructs a weighted multi-objective reward function and iteratively adjusts the generation probability distribution during decoding.

### Key Design 1: Persona Importance Estimation (PIE)

- **Function**: Self-supervisedly quantifies each persona attribute's contribution to model output in the current scenario.
- **Design Motivation**: Different attributes of the same character (e.g., personality traits vs. educational views) vary in salience across scenarios and must be dynamically quantified to guide subsequent alignment.
- **Mechanism**: Based on conditional mutual information (CMI), importance is estimated by computing the difference in log-probabilities of the model generation $G$ between the full prompt $T$ (containing attribute $w_i$) and the ablated prompt $T_i$ (with $w_i$ removed): $I_i = \log \Pr(G|T) - \log \Pr(G|T_i)$. The key innovation is substituting the inaccessible ground-truth $GT$ with the model's own generation $G$. The paper theoretically demonstrates that when the model's probabilities for $G$ and $GT$ are positively correlated, this approximation reliably preserves the importance ranking.

### Key Design 2: Persona-Guided Inference-Time Alignment (PIA)

- **Function**: Integrates persona importance scores into a multi-objective reward function to adjust token-level generation probabilities during decoding.
- **Design Motivation**: Multiple persona attributes must be simultaneously aligned but with different priorities; naive uniform weighting may cause all rewards to be maximized indiscriminately, obscuring hierarchical relationships.
- **Mechanism**: (1) A stepwise reward $r_i$ is defined for each attribute $w_i$ as the log-probability ratio of tokens generated with versus without that attribute. (2) A total reward is constructed by weighting with importance scores: $R(T,y) = \sum I_i r_i$. (3) L2-norm normalization is introduced, $R_\text{norm} = \sum I_i r_i / \|r\|_2$; via the Cauchy–Schwarz inequality, this guarantees that at the optimal solution the per-attribute rewards are consistent with the importance ranking. (4) The final alignment policy is obtained by solving a KL-constrained optimization: $p_r(y_t|T,y_{<t}) \propto \pi_\theta(y_t|T,y_{<t}) \cdot \exp(R_\text{norm}/\beta)$. In practice, only the top-2 attributes by importance are aligned to balance accuracy and efficiency.

## Experiments

### Experimental Setup

- **Datasets**: General character tasks use CharacterEval (77 Chinese roles, 1,785 dialogues) and BEYOND DIALOGUE (280 Chinese + 31 English roles, 3,552 dialogues); specific personality tasks use PERSONALITYBENCH (180K Big Five personality items).
- **Baselines**: Simple Prompting, Persona Prompting, ICL, NPTI (neuron activation intervention), OPAD (inference-time preference alignment), and PAS (persona activation search); GPT-4o and DeepSeek-R1 are also included for reference.
- **Backbone models**: Qwen2.5-7B-Instruct and LLaMA-3-8B-Instruct, evaluated on a single NVIDIA L40S GPU.

### Main Results

| Dataset | Metric | PDD vs. SP Win% | PDD vs. ICL Win% | PDD vs. OPAD Win% |
|---|---|---|---|---|
| CharacterEval (Qwen) | GPT-4o Win% | 51.2% | 65.3% | 52.8% |
| BEYOND DIALOGUE (Qwen) | GPT-4o Win% | 63.9% | 60.9% | 49.0% |

| Model | Method | PB↑ | PU↑ | Average↑ |
|---|---|---|---|---|
| Qwen2.5-7B | PP | 3.03 | 2.94 | 2.83 |
| Qwen2.5-7B | **PDD** | **3.08** | **3.01** | **2.85** |
| LLaMA-3-8B | ICL | 3.04 | 2.89 | 2.75 |
| LLaMA-3-8B | **PDD** | **3.00** | **2.96** | **2.81** |

### Key Findings

1. **General character tasks**: PDD comprehensively outperforms all baselines under GPT-4o evaluation and achieves the highest average scores on CharacterEval automatic metrics for both backbone models (Qwen: 2.85, LLaMA: 2.81), with particularly notable gains on persona–utterance consistency (PU).
2. **Specific personality tasks**: PDD consistently surpasses all baselines across all five Big Five dimensions (p < 0.05) while exhibiting the lowest standard deviation, demonstrating robust cross-persona adaptability.
3. **Small models competitive with commercial LLMs**: 7B/8B open-source models augmented with PDD achieve role-playing performance comparable to GPT-4o.

## Highlights & Insights

- This work is the first to incorporate the psychological CAPS theory into persona following, quantifying attribute–context relevance via conditional mutual information with a well-grounded theoretical motivation.
- Self-supervised importance estimation requires no ground-truth annotations, making the approach applicable to the diverse roles and scenarios encountered in practice.
- The normalized multi-objective reward design leverages the Cauchy–Schwarz inequality to enforce a hierarchical ordering among attribute priorities, offering mathematical elegance.
- As a purely inference-time method requiring no fine-tuning, PDD can be applied as a plug-and-play module across different LLMs.

## Limitations & Future Work

- **Inference overhead**: Each token requires a separate forward pass per attribute to compute rewards; even with only top-2 attribute alignment, this entails approximately 3× inference cost.
- **Reliability of self-generated approximation**: The validity of substituting model-generated responses for ground-truth in PIE depends on model capability; weaker models may produce inaccurate importance rankings.
- **Limited scale of evaluation**: Experiments are conducted only on 7B/8B models without exploring larger-scale models or real human–agent interaction scenarios.
- **Hyperparameter sensitivity**: The temperature $\beta$ in the normalized reward requires tuning, and its optimal value may differ across datasets and models.

## Related Work & Insights

| Method | Key Distinction |
|---|---|
| OPAD (Zhu et al., 2025a) | Single-objective preference alignment without distinguishing importance differences among multiple persona attributes; PDD supports dynamic multi-attribute weighted alignment. |
| NPTI (Deng et al., 2025) | Induces persona by identifying persona-relevant neurons and manipulating activations; requires pre-trained probes and is limited to Big Five dimensions. |
| PAS (Zhu et al., 2025b) | Trains probes to search for attention heads associated with personality traits and modulates persona at test time; requires additional training and is less flexible than PDD. |

## Rating

| Dimension | Score |
|---|---|
| Novelty | ⭐⭐⭐⭐ |
| Effectiveness | ⭐⭐⭐⭐ |
| Reproducibility | ⭐⭐⭐ |
| Practicality | ⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Enhancing Persona Following at Decoding Time via Dynamic Importance-Guided Token Estimation for Role-Playing Agents](enhancing_persona_following_at_decoding_time_via_dynamic_importance-guided_token.md)
- [\[ICLR 2026\] Fine-Grained Activation Steering: Steering Less, Achieving More](fine-grained_activation_steering_steering_less_achieving_more.md)
- [\[ICLR 2026\] Stopping Computation for Converged Tokens in Masked Diffusion-LM Decoding](stopping_computation_for_converged_tokens_in_masked_diffusion-lm_decoding.md)
- [\[ICLR 2026\] Rethinking Uncertainty Estimation in LLMs: A Principled Single-Sequence Measure](rethinking_uncertainty_estimation_in_llms_a_principled_single-sequence_measure.md)
- [\[ICLR 2026\] Meta-RL Induces Exploration in Language Agents](meta-rl_induces_exploration_in_language_agents.md)

</div>

<!-- RELATED:END -->

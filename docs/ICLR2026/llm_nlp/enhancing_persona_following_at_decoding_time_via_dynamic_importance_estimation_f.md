---
title: >-
  [Paper Note] Enhancing Persona Following at Decoding Time via Dynamic Importance Estimation for Role-Playing Agents
description: >-
  [ICLR 2026][LLM/NLP][role-playing agents] This paper proposes PDD (Persona Dynamic Decoding), a framework that dynamically estimates the importance of persona attributes across different contexts via conditional mutual information, and guides decoding at inference time through a weighted multi-objective reward, achieving adaptive persona following without any fine-tuning.
tags:
  - ICLR 2026
  - LLM/NLP
  - role-playing agents
  - persona following
  - inference-time alignment
  - conditional mutual information
  - decoding-time alignment
date: 2026-05-08
content_hash: 017dd4e3c8abfa66
---

# Enhancing Persona Following at Decoding Time via Dynamic Importance Estimation for Role-Playing Agents

**Conference**: ICLR 2026
**arXiv**: [2603.01438](https://arxiv.org/abs/2603.01438)
**Code**: None
**Area**: LLM/NLP
**Keywords**: role-playing agents, persona following, inference-time alignment, conditional mutual information, decoding-time alignment

## TL;DR

This paper proposes PDD (Persona Dynamic Decoding), a framework that dynamically estimates the importance of persona attributes across different contexts via conditional mutual information, and guides decoding at inference time through a weighted multi-objective reward, achieving adaptive persona following without any fine-tuning.

## Background & Motivation

Role-Playing Language Agents (RPLAs) are increasingly important in sociological research (e.g., voting behavior analysis, rumor diffusion simulation), requiring LLMs to faithfully follow predefined persona profiles. However, existing approaches suffer from two core limitations:

**Lack of dynamic adaptability**: Psychological research (e.g., CAPS theory) demonstrates that the influence of persona on behavior is **context-dependent**—the same individual activates different personality traits in different situations. Yet neither prompt engineering approaches (direct prompting, ICL, RAG) nor parameter-based training methods (SFT, LoRA) treat personas as anything other than static.

**Heavy data dependency**: Parametric methods require large amounts of role-specific dialogue data, which is extremely difficult to collect given the diversity of roles and complexity of personalities in social simulation settings.

Core insight: A character's "sense of humor" may be highly relevant in casual conversation but should carry less weight in serious discussions. What is needed is a method that **dynamically estimates at inference time** the importance of each persona attribute in the current context.

## Method

### Overall Architecture

PDD consists of two core components:

1. **PIE (Persona Importance Estimation)**: Dynamically quantifies the importance of each persona attribute in the current context.
2. **PIA (Persona-Guided Inference-Time Alignment)**: Uses the importance scores to construct a weighted multi-objective reward that modulates generation probabilities at inference time.

### Key Designs

**PIE Module**: Quantifies attribute contribution via Conditional Mutual Information (CMI). Given a full prompt $T=\{C,P,x\}$ (context $C$, persona set $P=\{w_i\}$, query $x$):

$$I_i \triangleq \log \frac{\Pr(G|T)}{\Pr(G|T_i)}$$

where $T_i = T \setminus \{w_i\}$ (the prompt with attribute $w_i$ removed) and $G = \pi_\theta(T)$ is the model's generation under the full prompt. Intuitively, if removing an attribute significantly reduces the model's output probability, that attribute is important for the current output.

Theoretical contribution: The paper proves that, under mild assumptions, replacing the unavailable ground truth $GT$ with the model-generated $G$ to compute importance is reliable—since training objectives make the model's generation probability positively correlated with the GT probability.

**PIA Paradigm**: Multi-persona alignment is formulated as an RL problem with KL constraints. The key steps are:

1. **Stepwise persona reward**: For each attribute $w_i$, a token-level reward is computed by comparing generation probabilities with and without the attribute:
$$r_i(T, y_{<t}) = \sum_{t'=t-1}^{t} \log \frac{\pi_\theta(y_{t'}|T, y_{<t'})}{\pi_\theta(y_{t'}|T_i, y_{<t'})}$$

2. **Normalized reward function**: A normalized reward is designed using the Cauchy-Schwarz inequality to ensure consistency in importance ranking:
$$R_{\text{norm}} = \frac{\sum_{i=1}^n I_i r_i(T,y)}{\|\mathbf{r}\|_2}$$
Equality holds if and only if $\mathbf{r} \propto \mathbf{I}$ (the reward vector is proportional to the importance vector), naturally encouraging alignment between reward ranking and importance ranking.

3. **Optimal decoding policy**:
$$p_r(y_t|T,y_{<t}) = \frac{1}{Z} \pi_\theta(y_t|T,y_{<t}) \exp\left(\frac{1}{\beta} R_{\text{norm}}\right)$$

### Loss & Training

PDD is a **completely training-free** inference-time method, requiring no fine-tuning or additional training data. Key hyperparameters:
- $\beta=1.0$: KL regularization coefficient
- In practice, only the top-2 highest-importance attributes are aligned, balancing accuracy and efficiency
- Greedy decoding is used for response generation

## Key Experimental Results

### Main Results

**General role-playing tasks** (CharacterEval + BEYOND DIALOGUE): GPT-4o-judged win rates of PDD vs. baselines:

| Baseline | CharacterEval (Qwen) | CharacterEval (LLaMA) | BEYOND (Qwen) | BEYOND (LLaMA) |
|------|---------------------|----------------------|---------------|-----------------|
| SP | 51.2% Win | 52.5% Win | 63.9% Win | 56.2% Win |
| PP | 48.7% Win | 39.1% Win | 43.0% Win | 46.8% Win |
| ICL | 65.3% Win | 63.1% Win | 60.9% Win | 64.2% Win |
| OPAD | 52.8% Win | 48.2% Win | 49.0% Win | 47.6% Win |

CharacterRM automatic evaluation (Qwen2.5-7B): PDD achieves an average score of 2.85, the highest overall and surpassing GPT-4o's 2.87 (PP setting).

**Specific personality tasks** (PERSONALITYBENCH, Big Five):

| Personality Dimension | SP | PP | PDD (Qwen) | PDD (LLaMA) |
|---------|-----|-----|-----------|-------------|
| Agreeableness | 4.81 | 4.90 | **4.92** | **4.84** |
| Conscientiousness | 4.47 | 4.98 | **4.97** | **4.82** |
| Neuroticism | 3.02 | 3.45 | **3.54** | **4.13** |
| Average | 4.31 | 4.53 | **4.57** | **4.57** |

PDD achieves the highest average score across all dimensions with the lowest variance (p<0.05).

### Ablation Study

| Setting | Win Rate (Qwen) | CharacterRM (Qwen) | Win Rate (LLaMA) | CharacterRM (LLaMA) |
|------|----------------|-------------------|-----------------|-------------------|
| w/o normalization | 38% | 2.80 | 32% | 2.71 |
| **w/ normalization** | **42%** | **2.85** | **40%** | **2.81** |

Ablation on the number of aligned attributes: top-2 is the optimal choice; aligning too many attributes introduces irrelevant noise and increases computational overhead, while too few provides insufficient information.

### Key Findings

1. **Context-adaptive validation**: Visualizations show that the PIE module indeed assigns different attribute importance rankings to the same character (e.g., Guo Furong) across different contexts.
2. **Competitiveness of open-source models**: Open-source models at 7–8B parameters combined with PDD can match or even outperform GPT-4o in persona following.
3. **Cross-model and cross-lingual consistency**: PDD is effective on both Qwen (primarily Chinese) and LLaMA (primarily English), with consistent performance across Chinese and English datasets.
4. **PIE robustness**: Even when generation quality $G$ degrades, the attribute importance ranking estimated by PIE remains stable.

## Highlights & Insights

1. **Theoretical elegance**: CMI as a measure of attribute importance has a solid information-theoretic foundation, and the approximation that avoids ground truth is theoretically guaranteed.
2. **Ingenious normalized reward design**: The Cauchy-Schwarz inequality is leveraged to naturally encode the objective that "reward ranking should align with importance ranking."
3. **Practical advantages of inference-time methods**: No per-character fine-tuning is required, making the approach well-suited for social simulation scenarios with large and dynamically changing role sets.
4. **Psychological theory grounding**: The CAPS theory's account of dynamic persona activation provides a solid cognitive science foundation for the method's design.

## Limitations & Future Work

1. **Inference overhead**: Each attribute requires an additional forward pass to compute $\Pr(G|T_i)$, scaling linearly with the number of attributes; the top-2 truncation is an engineering compromise.
2. **Reliance on LLM-based evaluation**: GPT-4o is primarily used as the judge, and biases inherent in LLM-as-Judge setups may affect the results.
3. **Predominantly single-turn interaction**: Whether persona consistency is maintained in multi-turn long conversations has not been sufficiently validated.
4. **Granularity of persona attributes**: The current approach assumes attributes are discrete, enumerable items, and handles vague or continuous personality traits (e.g., "slightly introverted") with limited flexibility.
5. **Combination with fine-tuning methods**: As an inference-time method, whether PDD can complement SFT/LoRA in a synergistic manner remains an open question.

## Related Work & Insights

- **CAPS Theory** (Sherman et al., 2015): The context-dependent activation theory of personality, providing the core cognitive science motivation.
- **OPAD** (Zhu et al., 2025): Single-objective inference-time preference alignment; PDD extends this to the multi-objective setting.
- **NPTI** (Deng et al., 2025): Personality induction via neuron activation, a representative parametric approach.
- Insight: CMI as an attribute importance estimation method is generalizable to other multi-objective generation tasks requiring dynamic weighting.

## Rating

| Dimension | Score (1–5) |
|------|-----------|
| Novelty | 4.0 |
| Theoretical Depth | 4.5 |
| Experimental Thoroughness | 4.0 |
| Writing Quality | 4.0 |
| Value | 4.0 |
| Overall | 4.1 |

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Enhancing Persona Following at Decoding Time via Dynamic Importance-Guided Token Estimation for Role-Playing Agents](enhancing_persona_following_at_decoding_time_via_dynamic_importance-guided_token.md)
- [\[ICLR 2026\] Fine-Grained Activation Steering: Steering Less, Achieving More](fine-grained_activation_steering_steering_less_achieving_more.md)
- [\[ICLR 2026\] Stopping Computation for Converged Tokens in Masked Diffusion-LM Decoding](stopping_computation_for_converged_tokens_in_masked_diffusion-lm_decoding.md)
- [\[ICLR 2026\] Rethinking Uncertainty Estimation in LLMs: A Principled Single-Sequence Measure](rethinking_uncertainty_estimation_in_llms_a_principled_single-sequence_measure.md)
- [\[ICLR 2026\] Meta-RL Induces Exploration in Language Agents](meta-rl_induces_exploration_in_language_agents.md)

<!-- RELATED:END -->

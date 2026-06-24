---
title: >-
  [Paper Note] Convert Language Model into a Value-based Strategic Planner
description: >-
  [ACL 2025][LLM (Other)][emotional support conversation] This paper proposes the straQ* framework, which reformulates the next-token prediction of LLMs into next-strategy prediction. By training the LLM as a strategy-level Q-network using the Bellman equation, the framework plans the optimal supportive strategy based on long-term returns in Emotional Support Conversation (ESC). It serves as a plug-and-play, lightweight planner to guide dialogue LLMs to generate high-quality re…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "emotional support conversation"
  - "Q-learning"
  - "strategic planning"
  - "reinforcement learning"
  - "dialogue"
date: 2026-05-08
content_hash: 1b318af1c49cab81
---

# Convert Language Model into a Value-based Strategic Planner

**Conference**: ACL 2025  
**arXiv**: [2505.06987](https://arxiv.org/abs/2505.06987)  
**Code**: [https://github.com/suran662/StraQ](https://github.com/suran662/StraQ)  
**Area**: LLM/NLP  
**Keywords**: emotional support conversation, Q-learning, strategic planning, reinforcement learning, dialogue

## TL;DR

This paper proposes the straQ* framework, which reformulates the next-token prediction of LLMs into next-strategy prediction. By training the LLM as a strategy-level Q-network using the Bellman equation, the framework plans the optimal supportive strategy based on long-term returns in Emotional Support Conversation (ESC). It serves as a plug-and-play, lightweight planner to guide dialogue LLMs to generate high-quality responses.

## Background & Motivation

**Background**: Emotional Support Conversation (ESC) aims to alleviate users' emotional distress through effective dialogue. ESC theory divides the support process into three stages: Exploration $\rightarrow$ Comforting $\rightarrow$ Action, where supporters must select appropriate strategies (such as questioning, empathy, or giving suggestions) and achieve a natural transition between stages. With the development of LLMs, LLM-based ESC solutions have made significant progress.

**Limitations of Prior Work**: Most existing LLM methods focus on immediate response quality, lacking systematic planning for long-term support strategies. Specifically: (1) LLMs tend to repeatedly use the same strategy (e.g., continuously "restating or paraphrasing"), leading to unsmooth transitions between dialogue stages; (2) ESC is not modeled from an MDP state perspective, making it impossible to optimize long-term satisfaction; (3) Strategy selection exhibits bias, and the frequency of using different strategies is highly imbalanced.

**Key Challenge**: How to enable LLMs to select support strategies based on long-term returns (rather than greedily pursuing the current optimal strategy) in multi-turn dialogues?

**Key Insight**: Formulating the ESC task as a strategy-level MDP, drawing inspiration from Deep Q-Learning (DQN). The average token logit of the LLM is used as the Q-value to approximate the strategy value function, and the LLM parameters are fine-tuned using the Bellman equation, converting the LLM into a plug-and-play strategic planner.

## Method

### Strategy-level MDP Definition

The ESC task is modeled as a 5-tuple MDP $(S, A, R, T, \gamma)$:

- **State $s$**: Consists of the dialogue background description, current emotion, dialogue history, and current user utterance: $s = \{desc, e, h, query\}$
- **Action $a$**: Eight support strategies defined in the ESConv dataset (Question, Restatement/Paraphrasing, Reflection of Feelings, Self-disclosure, Affirmation/Reassurance, Providing Suggestions, Providing Information, Others)
- **Reward $r$**: Immediate user satisfaction, obtained through dataset annotations or GPT-4 scores
- **Transition function $T$**: The history is updated after each dialogue turn, and the user generates a new utterance and emotion

The key innovation is that the action space is at the "strategy level" rather than the "token level." This makes the MDP action space small (only 8 strategies) with clear semantics, making it suitable for efficient Q-learning solutions.

### LLM as a Q-Function

The core idea is to directly reuse the LLM architecture as the Q-network without introducing an additional value head:

1. Construct an instruction template $\mathcal{I}(s)$, filling the state $s$ into a multiple-choice question (MCQ) format prompt.
2. Concatenate the instruction and the strategy $\mathcal{I}(s) \oplus a$, and input it into the LLM.
3. **Q-value Definition**: Use the average log probability of the strategy tokens in the LLM's output as $Q_\theta(s, a)$.
4. During inference, iterate through all $K$ strategies and select the strategy with the largest Q-value: $a^\star = \arg\max_a \text{LLM}(\mathcal{I}(s) \oplus a)$

**Design Motivation**: The average logit naturally reflects the LLM's level of "confidence" in the strategy under the current context, without requiring additional network parameters. Designing the instruction in an MCQ format, where the LLM selects the strategy index, strengthens its understanding of strategy selection.

### Bellman Equation Training

Fine-tune the LLM using the TD loss from DQN instead of the traditional cross-entropy loss:

$$\mathcal{L}(\theta) = |r(s,a) + \gamma Q_\phi(s', a') - Q_\theta(s,a)|^2$$

- $\theta$: Parameters of the Q-network (current LLM).
- $\phi$: Parameters of the target Q-network, synchronized from $\theta$ every 10 steps.
- $\gamma = 0.85$: Discount factor.
- Utilize the causal mask of the Transformer to execute Bellman updates in parallel across the entire sequence.

Training reformulates next-token prediction into next-strategy prediction, shifting the loss from token-level cross-entropy to strategy-level TD loss.

### Two Reward Mechanisms

- **straQ\*-imit (Imitation)**: Assign $r = +1$ to $(s, a)$ pairs from the dataset, and $r = -1$ to randomly sampled alternative strategies, using a positive-to-negative ratio of 1:1. This directly imitates expert annotations.
- **straQ\*-distill (Distillation)**: Use GPT-4 to score each $(s, a)$ from 0 to 5 as the reward. This distills knowledge from the teacher model.

### Inference Process

straQ* is used as a plug-and-play planner: (1) The planner LLM (1B) computes Q-values for all strategies based on the current state and selects the optimal strategy; (2) The dialogue LLM (8B) generates the final response based on the selected strategy and context. The planner is only responsible for strategy selection and does not participate in response generation.

## Key Experimental Results

### In-Domain Evaluation (ESConv Dataset)

| Method | Acc ↑ | Q ↑ | B ↓ | B-2 ↑ | R-L ↑ |
|------|-------|-----|-----|-------|-------|
| LLaMA3-8B Direct | 11.80 | 10.26 | 1.61 | 3.47 | 10.64 |
| + Direct-Refine | 17.08 | 11.07 | 1.27 | 3.10 | 6.13 |
| + Self-Refine | 17.58 | 13.61 | 1.92 | 3.34 | 9.71 |
| + CoT | 15.32 | 10.38 | 1.69 | 3.16 | 10.50 |
| + FSM | 17.37 | 11.15 | 0.81 | 4.12 | 11.83 |
| **+ straQ\*-distill** | **41.22** | **38.95** | **0.57** | 3.89 | 11.80 |
| **+ straQ\*-imit** | **46.83** | **43.15** | 0.80 | **3.89** | **12.84** |
| LLaMA3-8B + SFT | 32.43 | 21.29 | 1.28 | 6.97 | 16.59 |
| + SFT + FSM | 28.83 | 18.36 | 1.32 | 7.57 | 17.42 |
| + SFT + straQ\*-distill | 41.22 | 38.95 | 0.57 | 7.01 | 16.93 |
| + SFT + straQ\*-imit | **46.83** | **43.15** | 0.80 | **7.63** | **17.30** |

straQ\*-imit achieves a strategy accuracy of 46.83%, which is a nearly fourfold improvement over Direct's 11.80%; the strategy bias B is reduced from 1.61 to 0.57-0.80.

### Out-of-Domain Generalization (EmpatheticDialogues)

| Method | B-2 | R-L | Dist-2 | CIDEr |
|------|-----|-----|--------|-------|
| Direct | 3.09 | 9.91 | 25.23 | 1.60 |
| + CoT | 2.91 | 9.79 | 32.65 | 1.37 |
| + FSM | 3.33 | 10.80 | 33.37 | 2.96 |
| **+ straQ\*-distill** | **4.49** | **12.93** | **46.53** | **8.36** |
| + straQ\*-imit | 4.27 | 12.66 | 46.80 | 8.11 |

On the unseen EmpatheticDialogues dataset, the CIDEr of straQ\* increases from 2.96 (FSM) to 8.36, demonstrating significant generalization capability. The distill version outperforms imit on OOD, indicating that the knowledge distilled from GPT-4 is more generalizable.

### Human Evaluation

| Method | Fluency | Emotion | Acceptance | Effectiveness | Sensitivity | Satisfaction |
|------|---------|---------|------------|---------------|-------------|-------------|
| Original Dataset | 3.51 | 3.61 | 3.40 | 3.10 | 3.50 | 3.30 |
| LLaMA3-8B Direct | 2.95 | 3.00 | 2.60 | 2.40 | 2.70 | 2.60 |
| + FSM | 3.30 | 3.35 | 2.90 | 2.90 | 3.00 | 2.93 |
| + SFT + CoT | 3.67 | 3.61 | 3.22 | 3.67 | 3.56 | 3.45 |
| **+ straQ\*-distill** | **3.52** | **3.65** | **3.59** | **3.73** | **3.71** | **3.66** |
| + straQ\*-imit | 3.42 | 3.25 | 3.23 | 3.07 | 3.10 | 3.13 |

The satisfaction score of straQ\*-distill reaches 3.66, exceeding the original dataset annotation of 3.30 and all baseline methods. The advantage is most pronounced in the Effectiveness and Sensitivity dimensions.

### Ablation Study

| Method | Acc ↑ | Q ↑ | B ↓ | B-2 ↑ | R-L ↑ |
|------|-------|-----|-----|-------|-------|
| w/ value head | 19.81 | 11.40 | 1.66 | 6.74 | 15.99 |
| auto-regressive (SFT) | 46.22 | 43.01 | 0.69 | 7.25 | 16.48 |
| **straQ\*-imit** | **46.83** | **43.15** | 0.80 | **7.63** | **17.03** |

The accuracy when using an independent value head is only 19.81%, demonstrating that directly using the average logit as the Q-value is superior to using an additional classification head. straQ\* also outperforms pure SFT in response quality metrics (B-2, R-L).

### Strategy Value Analysis

| Method | Average Reward (GPT-4) | Average Value |
|------|-----------------|---------|
| Original Dataset | 3.01 | 252.09 |
| LLaMA3-8B Direct | 3.66 | 346.31 |
| straQ\*-distill | **3.99** | 424.78 |
| straQ\*-imit | 3.72 | **445.95** |

The long-term value (cumulative return) of straQ\* is significantly higher than that of the baselines, validating that Q-learning successfully optimises long-term returns.

## Highlights & Insights

- **Minimalist yet Effective Q-value Definition**: Directly uses the LLM average logit as the Q-value without introducing additional networks. The loss converges after training, and the Q-value effectively distinguishes the quality of different strategies.
- **Plug-and-play Architecture**: A decoupled design featuring a 1B planner + an 8B dialogue model, allowing the planner to be paired with any dialogue LLM.
- **Complementary Dual Reward Mechanisms**: imit excels in automatic metrics and in-domain tasks, while distill performs better in human evaluations and out-of-domain generalization, making them suitable for different scenarios.
- **Strategy Transition Matrix**: Validates that straQ\* learns reasonable ESC stage transitions (I $\rightarrow$ II $\rightarrow$ III), whereas the Direct method tends to get stuck in the first stage.

## Limitations & Future Work

- Only validated on emotional support dialogue (ESConv + EmpatheticDialogues), without extension to other strategy-oriented dialogue scenarios such as negotiation or medical consultation.
- Potential bias in human evaluation—evaluators are interns rather than real help-seekers, and the sample size is limited.
- The planner needs to iterate over all strategies to compute Q-values, leading to a linear increase in inference overhead as the strategy space scales up.
- The current approximation of the Q-value using average logits lacks rigorous theoretical guarantees, and its convergence relies heavily on empirical validation.

## Rating

⭐⭐⭐⭐ (4/5)

- **Novelty** ⭐⭐⭐⭐: The integration of DQN into LLMs is novel; the design of a strategy-level MDP combined with average logits as Q-values is simple yet effective.
- **Experimental Thoroughness** ⭐⭐⭐⭐⭐: Comprehensive evaluation, including in-domain/out-of-domain testing, human evaluation, ablation studies, sensitivity analysis, and case studies.
- **Value** ⭐⭐⭐⭐: The plug-and-play lightweight planner paradigm can be transferred to other dialogue scenarios requiring long-term strategic planning.
- **Writing Quality** ⭐⭐⭐: The paper is generally clear, though it contains a substantial amount of notation, and some derivations could be further streamlined.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Unintended Harms of Value-Aligned LLMs: Psychological and Empirical Insights](unintended_harms_of_value-aligned_llms_psychological_and_empirical_insights.md)
- [\[ACL 2025\] SEE: Strategic Exploration and Exploitation for Cohesive In-Context Prompt Optimization](see_strategic_exploration_exploitation_prompt_optimization.md)
- [\[ACL 2025\] Generative Psycho-Lexical Approach for Constructing Value Systems in Large Language Models](generative_psycholexical_approach_for_constructing_value.md)
- [\[ACL 2025\] Value Portrait: Assessing Language Models' Values through Psychometrically and Ecologically Valid Items](value_portrait_assessing_language_models_values_through_psychometrically_and_eco.md)
- [\[ECCV 2024\] Cultural Value Differences of LLMs: Prompt, Language, and Model Size](../../ECCV2024/llm_nlp/cultural_value_differences_llms.md)

</div>

<!-- RELATED:END -->

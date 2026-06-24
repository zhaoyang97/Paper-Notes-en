---
title: >-
  [Paper Note] Perspective Transition of Large Language Models for Solving Subjective Tasks
description: >-
  [ACL 2025][LLM (Other)][Perspective Transition] Proposes RPT (Reasoning through Perspective Transition), which allows LLMs to sequentially explore direct, role-playing, and third-person perspectives within a single prompt, rank them by confidence, and perform reasoning with the optimal perspective. It consistently outperforms fixed perspective and ensemble baselines across 12 subjective tasks and 4 models (GPT-4/GPT-3.5/Llama-3/Qwen-2), achieving an average improvement of +4.…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Perspective Transition"
  - "Subjective Tasks"
  - "Confidence Ranking"
  - "ICL"
  - "Metaphor/Sarcasm/Stance Detection"
date: 2026-05-08
content_hash: 019d0cb8e9dd13f0
---

# Perspective Transition of Large Language Models for Solving Subjective Tasks

**Conference**: ACL 2025  
**arXiv**: [2501.09265](https://arxiv.org/abs/2501.09265)  
**Code**: None  
**Area**: LLM/NLP  
**Keywords**: Perspective Transition, Subjective Tasks, Confidence Ranking, ICL, Metaphor/Sarcasm/Stance Detection

## TL;DR

Proposes RPT (Reasoning through Perspective Transition), which allows LLMs to sequentially explore direct, role-playing, and third-person perspectives within a single prompt, rank them by confidence, and perform reasoning with the optimal perspective. It consistently outperforms fixed perspective and ensemble baselines across 12 subjective tasks and 4 models (GPT-4/GPT-3.5/Llama-3/Qwen-2), achieving an average improvement of +4.56 points on GPT-3.5.

---

## Background & Motivation

**Background**: While LLMs excel in objective tasks such as mathematical reasoning, code generation, and commonsense QA, they remain limited in **subjective tasks** (e.g., metaphor recognition, sarcasm detection, dark humor classification, stance detection, and culture-related NLI). BigBench data shows that PaLM-535B achieves less than 50% zero-shot accuracy on tasks like metaphor recognition, dark humor, and sarcasm detection.

**Limitations of Prior Work**:
   - **CoT-like methods focus on "how to think deeper" while neglecting "from which perspective to think"**. For subjective tasks, the reasoning chain itself may mislead the model—Figure 1 in the paper illustrates a case where CoT reasoning actually leads to an incorrect answer.
   - **Different subjective tasks suit different perspectives**: Sarcasm detection may benefit from role-playing (leveraging domain expert knowledge), whereas stance detection might require a third-person observer perspective to mitigate bias. However, no single perspective performs optimally across all tasks.
   - **Existing perspective-based methods are domain-specific**: Zero-Shot-CoT performs strongly on rhetorical tasks but weakly on cultural tasks (only 40.07 F1 on SocNorm), while Conversation Simulation (RiC) excels in cultural scenarios but struggles on rhetorical tasks.

**Key Challenge**: The diversity of subjective tasks demands flexible perspective switching, yet existing methods either fix a single perspective or use naive ensembles, which exponentially multiplies inference costs.

**Key Insight**: Inspired by the psychological concept of "Theory of Mind"—where humans understand behaviors from both self and others' perspectives—dynamic perspective selection is enabled for LLMs.

**Core Idea**: To enable the model within a unified prompt to: (1) explore multiple perspectives; (2) rank them based on confidence; (3) choose the perspective with the highest confidence to yield the final answer. This is achieved entirely via in-context learning (ICL) without training, requiring only a single inference pass.

---

## Method

### Overall Architecture: RPT (Reasoning through Perspective Transition)

RPT consists of three steps, integrated into a **single inference pass** via a unified prompt:

- **Step 1 - Explore multiple perspectives**: Given the task description $\mathcal{D}$ and question $\mathcal{Q}$, the model generates $n$ candidate perspectives $\mathcal{P} = \{p_1, ..., p_n\}$ based on the trigger instruction $\mathcal{T}_1$ ("analyze the problem from multiple perspectives").
- **Step 2 - Confidence ranking**: The model evaluates the confidence $\mathcal{C}$ for each perspective, ranking them by "the perceived probability of correctness".
- **Step 3 - Maximum confidence reasoning**: The model selects the top-ranked perspective to complete reasoning and output the final response $\mathcal{R}$.

Unified formula: $\mathcal{T} = \mathcal{T}_1 \oplus \mathcal{T}_2 \oplus \mathcal{T}_3$, $\mathcal{P}, \mathcal{C}, \mathcal{R} = \mathcal{M}(\mathcal{D} \oplus \mathcal{Q} \oplus \mathcal{T})$

### Key Designs

#### Key Design 1: Definition and Division of Three Perspectives

- **Function**: Defining three complementary reasoning perspectives.
- **Mechanism**:
    - **Direct Perspective**: The model directly analyzes the problem based on its own knowledge without setting a persona, similar to zero-shot reasoning.
    - **Role Perspective**: The model plays an expert role related to the problem (e.g., linguist, cultural researcher) to invoke domain-specific knowledge.
    - **Third-person Perspective**: The model simulates discussions and dialogues among multiple agents, summarizing and providing answers as an observer.
- **Design Motivation**: The direct perspective is suitable for knowledge-intensive judgments, the role-playing perspective stimulates domain expertise, and the third-person perspective reduces biases through multi-agent discussions. The three are complementary, addressing the needs of various subjective tasks. Ablation studies demonstrate that removing any of these perspectives leads to degraded performance.

#### Key Design 2: Confidence-Based Dynamic Selection Mechanism

- **Function**: Enabling the model to autonomously evaluate the confidence of each perspective and select the optimal one.
- **Mechanism**: During the exploration phase, the model assigns a confidence percentage to each perspective (e.g., "third-person 85%, role 70%, direct 60%"), and then automatically selects the perspective with the highest confidence for final reasoning.
- **Design Motivation**: Different problems suit different perspectives; confidence ranking allows the model to make dynamic decisions based on specific problem instances. Compared to fixed selection or majority voting, this "self-evaluation + selection" mechanism fits each problem more granularly while avoiding the high costs of ensemble methods that require multiple inferences.

### Loss & Training

RPT **requires no training** and is based entirely on in-context learning. All steps are completed using a unified prompt, requiring only a single inference pass to produce the final answer. The temperature is set to 0 to ensure deterministic and reproducible outputs.

---

## Key Experimental Results

### Main Results: Zero-Shot Results (Average Accuracy/F1 across 12 Datasets and 4 Models)

| Method | Type | Llama-3-8B | Qwen-2-7B | GPT-3.5 | GPT-4 |
|------|------|-----------|-----------|---------|-------|
| Direct Prompt | Single Direct | 52.70 | 60.45 | 62.21 | 71.81 |
| Zero-Shot-CoT | Single Direct | 57.94 | 62.12 | 63.06 | 72.83 |
| Role-Play Prompting | Single Role | 57.38 | 61.87 | 64.16 | 72.95 |
| Reason in Conv. (RiC) | Single Third-person | 60.85 | 65.10 | 68.85 | 78.18 |
| Ensemble | Ensemble | 59.89 | 66.54 | 66.90 | 76.72 |
| CoT-SC | Ensemble | 59.96 | 65.34 | 73.38 | 75.43 |
| **RPT (Ours)** | **Dynamic Perspective** | **64.12** | **68.64** | **77.94** | **80.81** |

**Key Highlights**: RPT achieves an average improvement of +3.27 on Llama-3 (vs. RiC), +4.56 on GPT-3.5 (vs. CoT-SC), and +2.63 on GPT-4 (vs. RiC).

### Main Results: Few-Shot Results (3-shot, Average of Culture-Related + Stance Detection Subsets)

| Method | Llama-3-8B | Qwen-2-7B | GPT-3.5 | GPT-4 |
|------|-----------|-----------|---------|-------|
| CoT-SC | 61.91 | 54.09 | 66.43 | 66.41 |
| RiC | 61.71 | 64.02 | 67.17 | 71.68 |
| **RPT (Ours)** | **63.97** | **65.83** | **69.52** | **73.35** |

### Ablation Study: Impact of Perspective Removal on RPT Performance

| Removed Component | Average Performance Drop |
|----------|------------|
| Remove any 1 perspective | −1.32 ~ −2.53 |
| Remove any 2 perspectives | −5.15 ~ −6.48 |
| Remove all perspectives (degrades to simple reasoning) | −7.60 |
| Remove third-person (largest drop) | Max Drop |

### Key Findings

1. **RPT exhibits the strongest consistency**: It achieves the best average performance across both zero-shot and few-shot settings on all 4 models, whereas baseline methods tend to excel only in specific domains.
2. **Different tasks prefer different perspectives**: Zero-Shot-CoT reaches 70.72 on SNARKS (sarcasm detection) but only 40.07 on SocNorm; RiC achieves 64.05 on e-SocNorm but is weaker on rhetorical tasks.
3. **Stronger models benefit more**: The improvements of RPT are more pronounced on GPT-3.5/GPT-4, indicating that models with stronger reasoning capabilities can leverage perspective switching more effectively.
4. **No significant increase in inference cost**: RPT is executed in a single inference pass, yielding an output length comparable to single-perspective methods and significantly lower than CoT-SC/Ensemble which require multiple sampling iterations.
5. **Few-shot may introduce noise**: Adding 3-shot context to RiC on GPT-4 causes an average performance drop of 6.50 points, whereas RPT still steadily improves by +1.67.

---

## Highlights & Insights

- **"From which perspective to think" is more crucial than "how to think deeper"**: The core insight of this paper is that for subjective tasks, perspective selection (who should think) is more critical than depth of reasoning (how to think deeper). This operates as an interesting complement to methods focusing on reasoning depth, such as CoT and o1.
- **Perspective as a meta-method**: Perspective selection is orthogonal to CoT—one can select a perspective first, and then conduct CoT reasoning within that perspective. RPT represents a "method of methods."
- **Effectiveness of confidence self-assessment**: The model reasonably estimates its own confidence across different perspectives, providing positive evidence for the reliability of "LLM self-evaluation."
- **Compelling case analysis**: In the stance detection case shown in Figure 8, both CoT and RiC misclassify "Get the truth from Trump!" as AGAINST, whereas RPT correctly identifies the exclamation mark and word cues hinting at FAVOR through the third-person perspective.
- **Pure ICL with zero training**: The method is extremely simple—requiring only a well-designed unified prompt, without fine-tuning, external models, or multiple sampling iterations.

---

## Limitations & Future Work

1. **Coarse-grained perspective classification**: Relying solely on three perspectives might be insufficient; certain tasks requiring specific cultural backgrounds or domain expertise might benefit from more fine-grained perspectives (e.g., distinguishing observers from different cultural backgrounds).
2. **Confidence calibration issues**: The methodology relies on the reliability of the model's self-assessed confidence. Weaker models (e.g., Llama-3-8B) exhibit higher instability, indicating that confidence estimation is closely tied to overall model capability.
3. **Single-turn dialogue limitation**: RPT runs within a single turn, without leveraging multi-turn conversations or feedback mechanisms. Iterative multi-turn perspective exploration could yield further improvements.
4. **Perspective selection rather than method selection**: Currently, RPT selects a perspective rather than a specific method. Enabling the model to select different reasoning strategies (e.g., CoT, Self-Ask) under each perspective could potentially enhance performance.
5. **Small dataset scales**: Some datasets are relatively small (e.g., only 70 instances for Entailment, 80 for Humor), which may lead to higher variance in experimental results.

---

## Related Work & Insights

- **Complementary to CoT/o1**: CoT focusing on "deep reasoning" and RPT focusing on "perspective selection" are orthogonal and can be combined.
- **Connection to Multi-Agent systems**: The third-person perspective essentially simulates multi-agent discussions within a single LLM, with computational costs far lower than actual multi-agent systems.
- **Inspiration for future research**: (1) Can the model automatically discover new perspectives instead of being limited to three predefined ones? (2) Can self-assessed confidence be applied to adaptive strategy selection in other tasks? (3) How does it perform when combined with deep reasoning models like DeepSeek-R1?

---

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The entry point of "from which perspective to think" is novel. Formalizing perspective selection as part of the reasoning pipeline distinguishes it from mainstream CoT deep reasoning directions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 12 datasets, 5 task types, 4 models (both open and closed-source), zero-shot and few-shot setups, 11 baselines, and complete ablation and efficiency analyses; the main limitation is the small size of some datasets.
- **Writing Quality**: ⭐⭐⭐⭐ — The motivation is clear, the case analysis is intuitive, and Figures 1-2 illustrate the problem and method effectively; the methodological description is slightly over-formalized.
- **Value**: ⭐⭐⭐⭐ — The method is extremely simple and requires no training, offering direct value for practical applications dealing with subjective tasks; the paradigm of perspective selection combined with confidence ranking has strong potential for generalization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Comparing Large Language Models in Extracting Subjective Information from Political News](comparing_large_language_models_in_extracting_subjective_information_from_politi.md)
- [\[ACL 2025\] Understanding the Repeat Curse in Large Language Models from a Feature Perspective](understanding_the_repeat_curse_in_large_language_models_from_a_feature_perspecti.md)
- [\[ACL 2025\] Language Models Grow Less Humanlike beyond Phase Transition](language_models_grow_less_humanlike_beyond_phase_transition.md)
- [\[ACL 2025\] Evaluating Implicit Bias in Large Language Models by Attacking from a Psychometric Perspective](evaluating_implicit_bias_in_large_language_models_by_attacking_from_a_psychometr.md)
- [\[ACL 2025\] Beyond Demographics: Fine-tuning Large Language Models to Predict Individuals' Subjective Text Perceptions](beyond_demographics_fine-tuning_large_language_models_to_predict_individuals_sub.md)

</div>

<!-- RELATED:END -->

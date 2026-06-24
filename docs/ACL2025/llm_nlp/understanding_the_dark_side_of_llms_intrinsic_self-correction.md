---
title: >-
  [Paper Note] Understanding the Dark Side of LLMs' Intrinsic Self-Correction
description: >-
  [ACL 2025][LLM (Other)][Intrinsic Self-Correction] This paper systematically investigates the failure of LLMs' intrinsic self-correction. It proposes three interpretability methods to reveal the underlying reasons: answer wavering and prompt bias in simple tasks, and human-like cognitive biases in complex tasks. Furthermore, two simple yet effective mitigation strategies—question repeating and few-shot SFT—are proposed.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Intrinsic Self-Correction"
  - "Answer Wavering"
  - "Prompt Bias"
  - "Cognitive Bias"
  - "Interpretability"
date: 2026-05-08
content_hash: ffe830737d537e54
---

# Understanding the Dark Side of LLMs' Intrinsic Self-Correction

**Conference**: ACL 2025  
**arXiv**: [2412.14959](https://arxiv.org/abs/2412.14959)  
**Code**: [https://x-isc.info/](https://x-isc.info/)  
**Area**: LLM/NLP  
**Keywords**: Intrinsic Self-Correction, Answer Wavering, Prompt Bias, Cognitive Bias, Interpretability

## TL;DR

This paper systematically investigates the failure of LLMs' intrinsic self-correction. It proposes three interpretability methods to reveal the underlying reasons: answer wavering and prompt bias in simple tasks, and human-like cognitive biases in complex tasks. Furthermore, two simple yet effective mitigation strategies—question repeating and few-shot SFT—are proposed.

## Background & Motivation

1. **Background**: Self-correction is a popular method that allows LLMs to refine initial answers through reflective feedback. Among these, **intrinsic self-correction** relies solely on the model's own capabilities to make corrections, without introducing any external knowledge or oracle labels.

2. **Limitations of Prior Work**: Several studies (Huang et al., Li et al.) point out that intrinsic self-correction often **fails** without oracle label guidance; models not only fail to correct wrong answers, but are also more likely to **flip correct answers to incorrect ones**. For example, the correct-to-incorrect flip rate of Llama-3.1-8B on BoolQ is as high as 58.8%. However, prior works only report the failure phenomenon and lack in-depth explanations for **why** it fails.

3. **Key Challenge**: During inference, oracle labels are unavailable to distinguish between correct and incorrect initial answers, so feedback is uniformly applied to all responses. This leads to correct answers also being "suggested" to change, yet it remains unclear what internal mechanisms cause the model to alter correct answers.

4. **Goal**: To answer from an interpretability perspective: why does the intrinsic self-correction of LLMs fail? Are the failure mechanisms identical across different types of tasks?

5. **Key Insight**: The study is conducted along two lines: simple tasks and complex tasks. For simple tasks (Yes/No QA), it starts from internal model mechanisms—probing intermediate-layer answer changes using tuned lens and analyzing prompt influence via token attribution. For complex tasks (decision-making/reasoning/programming), it analyzes reasoning logs from the perspective of human-like cognitive biases.

6. **Core Idea**: The essence of intrinsic self-correction failure is that in simple tasks, the model is misled by the recency bias of the "refinement prompt," whereas in complex tasks, the model exhibits human-like cognitive deficits such as overthinking, cognitive overload, and perfectionism bias.

## Method

### Overall Architecture

The research framework consists of three tiers: (1) **Phenomenon Verification**—verifying the ubiquity of self-correction failures across 4 tasks (Yes/No QA, decision-making, reasoning, programming) using 9 SOTA models (including GPT-o1, DeepSeek-R1); (2) **Interpretability Analysis**—using mechanistic and token-level interpretability for simple tasks, and cognitive bias analysis for complex tasks; (3) **Mitigation Strategies**—question repeating and ultra-low-cost SFT.

### Key Designs

1. **Answer Wavering Analysis (Mechanistic Interpretability)**:

    - **Function**: Reveal the instability of internal representations in open-source LLMs during self-correction.
    - **Mechanism**: Use tuned lens to decode intermediate hidden states at each layer, compute the confidence score difference between the correct and incorrect answers $CS_\ell^{correct} - CS_\ell^{incorrect}$, and track changes in answers across layers. It is observed that during the initial response, confidence monotonically increases with depth towards the correct answer. However, after receiving "Are you sure?", the internal answer wavers back and forth across different layers, ultimately outputting an incorrect answer. Statistically, self-correction increases the frequency of Llama's internal answer variation from 8.3% to 14.1%.
    - **Design Motivation**: Directly observe the model internals rather than just looking at the output, providing mechanistic evidence of self-correction failures.

2. **Prompt Attribution and Contribution Tracking PACT (Token-level Interpretability)**:

    - **Function**: Quantitatively measure the contribution of each token/sequence in the input to the final output, applicable to both open-source and closed-source models.
    - **Mechanism**: Define $\text{PACT}(x_i, y) = \text{LP}(x \setminus \{x_i\}, y) - \text{LP}(x, y)$, which represents the change in the log probability of outputting $y$ when token $x_i$ is removed. For instances where correct answers are flipped, the PACT contribution of the refinement prompt is significantly higher than that of the original question (rendered as greener tokens), indicating the model is biased by "Are you sure?" and ignores the question itself. In instances where the correct answer is preserved, the original question makes a larger contribution.
    - **Design Motivation**: Explain "when" and "why" answer wavering occurs—essentially acting as recency bias, where the model tends to focus on the refinement instructions at the end of the prompt rather than the original question.

3. **Human-like Cognitive Bias Analysis (Complex Tasks Interpretation)**:

    - **Function**: Explain the behavioral patterns of self-correction failure in complex tasks (decision-making/reasoning/programming).
    - **Mechanism**: Analyze model reasoning logs to summarize three types of human-like cognitive biases: (1) **Overthinking**—the model increases its "think" steps by 2.9× (GPT-o1-mini) after self-correction, getting trapped in a thinking loop and failing to act; (2) **Cognitive Overload**—following refinement, the input prompt length increases by 4.4-6.1×, causing the model to forget critical formatting details and fail the task; (3) **Perfectionism Bias**—the model over-optimizes on top of an already successful outcome (e.g., trying to grab two pillows simultaneously), increasing output length by 1.7-3.1× but introducing new errors.
    - **Design Motivation**: Since closed-source models cannot use tuned lens and PACT is not suitable for long outputs, human cognitive science analogies are leveraged to explain behavioral patterns.

### Loss & Training

SFT Mitigation Strategy: select an extremely small number of correct-to-incorrect (✓→✗) samples (4 samples for Llama, 10 samples for GPT), modify the second-round response to the correct answer, and construct "correct-to-correct" (✓→✓) training samples. Only **behavioral adjustment** is trained rather than injecting new knowledge (the correct answers for the selected samples are already known to the model). The training cost is extremely low—GPT SFT costs only $0.004 and takes 3 minutes.

## Key Experimental Results

### Main Results

**Yes/No QA (BoolQ, 3270 samples)**:

| Model | Post-correction Accuracy (↓ΔACC) | Correct→Incorrect Ratio |
|------|---------------------|-------------|
| GPT-o1-preview | 78.7% (↓4.9%) | 13.2% |
| GPT-4o | 79.2% (↓4.9%) | 11.3% |
| Llama-3.1-8B | 49.2% (↓20.4%) | 58.8% |
| DeepSeek-R1 | 78.1% (↓1.6%) | 7.9% |

**Complex Tasks (ChatGPT Series)**:

| Task | Model | Post-correction Accuracy (↓ΔACC) | Correct→Incorrect Ratio |
|------|------|---------------------|-------------|
| Decision Making | GPT-4o | 14.2% (↓20.9%) | 76.6% |
| Reasoning | GPT-4o | 65.0% (↓2.0%) | 17.9% |
| Programming | GPT-4o | 72.6% (↓6.8%) | 21.9% |

### Ablation Study

**Mitigation Strategy Effects (Yes/No QA)**:

| Model + Strategy | Post-correction ACC (↓ΔACC) | ✓→✗ Ratio |
|-------------|-------------------|---------|
| GPT-4o | 79.2% (↓4.9%) | 11.3% |
| + Question Repeating | 83.6% (↓0.5%) | 6.0% |
| + SFT (10 samples) | 87.7% (↑4.1%) | 0% |
| Llama-3.1-8B | 49.2% (↓20.4%) | 58.8% |
| + Question Repeating | 52.4% (↓17.2%) | 52.8% |
| + SFT (4 samples) | 70.3% (↑0.7%) | 0% |

**SFT Cross-Task Generalization (SFT on Yes/No, tested on complex tasks)**:

| Task | Model | Original ACC (↓ΔACC) | +SFT ACC (↓ΔACC) |
|------|------|----------------|-----------------|
| Programming | GPT-4o | 72.6%(↓6.8%) | 82.6%(↑3.2%) |
| Reasoning | GPT-4o | 65.0%(↓2.0%) | 68.0%(↑1.0%) |
| Decision Making | GPT-3.5-turbo | 7.5%(↓5.2%) | 17.9%(↑5.2%) |

### Key Findings

- **Self-correction failure is a universal phenomenon**: From GPT-o1 and Llama to DeepSeek, all models exhibit a performance drop across all tasks.
- **More advanced models are not necessarily better**: In the Llama series, more advanced models (3.1 vs 2) flip more correct answers; in decision-making tasks, ChatGPT o1 performs worse than 3.5-turbo.
- **"Are you sure?" ≈ "You are wrong."**: These two prompts yield almost identical inner confidence change curves in Llama (JS divergence of only 0.0186), demonstrating that the model equates neutral questioning with negation.
- **Extreme few-shot SFT yields surprising effectiveness**: Only 4-10 samples can reduce ✓→✗ to 0%, and it features **cross-task generalization**—behavioral refinement trained on Yes/No QA generalizes to unseen tasks like programming and reasoning.

## Highlights & Insights

- **Generality of the PACT method**: The proposed Prompt Attribution and Contribution Tracking (PACT) method is applicable to both open- and closed-source LLMs, allowing quantitative tracking of any input token's contribution to the output. It can be transferred to scenarios like prompt engineering and adversarial attack analysis.
- **Insight of "Behavioral Adjustment vs. Knowledge Injection"**: SFT utilizes only samples with known answers (which the model originally gets correct) to modify behavior, rather than injecting new knowledge. This demonstrates that self-correction failure is a **behavioral issue** (reflexively changing answers) rather than a knowledge deficiency.
- **Innovative perspective of cognitive bias analogy**: Utilizing human psychology concepts (overthinking, cognitive overload, perfectionism) to explain LLM behavior offers both strong explanatory power and operability.
- **Inspiring discovery of "Are you sure?" = "You are wrong."**: This finding suggests that LLM instruction following may have over-learned the pattern of equating "questioning" with "negation."

## Limitations & Future Work

- The internal answer wavering analysis (tuned lens) is only applicable to open-source models; the internal mechanisms of closed-source models cannot be validated.
- The PACT method is currently only applicable to ChatGPT scenarios with single-token outputs; it cannot be computed for long outputs.
- The authors acknowledge that OpenAI has recently been working on mitigating sycophancy, and results might change on newer versions of GPT-4o (as of Feb 15, 2025).
- Although the SFT strategy is effective, sample construction requires selecting from ✓→✗ instances; how to automatically identify these instances during deployment remains a challenge.
- Only simple self-correction prompts were tested; more complex multi-turn self-correction strategies (such as reasoning chains) are not covered.

## Related Work & Insights

- **vs Huang et al. (2024)**: Huang et al. pointed out that self-correction fails without an oracle, but offered no explanation as to why. This paper provides mechanistic explanations using three interpretability methods.
- **vs Madaan et al. (Self-Refine)**: The success of Self-Refine relies on the assumption that LLMs can accurately evaluate their own outputs. This paper shows that this assumption holds invalid under the intrinsic setting—the model is misled by the refinement prompt itself.
- **vs Sharma et al. (Sycophancy)**: Sharma et al. studied sycophancy in LLMs, which typically requires a large amount of training data. In contrast, this study effectively mitigates it using only 4-10 samples, with the key being only adjusting behavior rather than injecting knowledge.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ This is the first work to systematically analyze self-correction failures from an interpretability perspective, using three complementary methods to cover different scenarios.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 9 models across 4 tasks, 5 prompt variations, mitigation strategies, and cross-task generalization.
- Writing Quality: ⭐⭐⭐⭐ The structure is clear, showcasing a complete logical chain from phenomenon to explanation to mitigation.
- Value: ⭐⭐⭐⭐⭐ Offers direct guidance for LLM deployment by revealing the fundamental flaws of popular self-correction strategies.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] ProgCo: Program Helps Self-Correction of Large Language Models](progco_program_helps_self-correction_of_large_language_models.md)
- [\[ACL 2025\] Self-Tuning: Instructing LLMs to Effectively Acquire New Knowledge through Self-Teaching](self-tuning_instructing_llms_to_effectively_acquire_new_knowledge_through_self-t.md)
- [\[ACL 2025\] Is It JUST Semantics? A Case Study of Discourse Particle Understanding in LLMs](is_it_just_semantics_a_case_study_of_discourse_particle_understanding_in_llms.md)
- [\[ACL 2025\] Leveraging Self-Attention for Input-Dependent Soft Prompting in LLMs](input_dependent_soft_prompting.md)
- [\[ACL 2025\] Enhancing Character-Level Understanding in LLMs through Token Internal Structure Learning](character_level_understanding.md)

</div>

<!-- RELATED:END -->

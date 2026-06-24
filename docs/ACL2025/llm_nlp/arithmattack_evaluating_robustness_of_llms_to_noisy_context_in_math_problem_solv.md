---
title: >-
  [Paper Note] ArithmAttack: Evaluating Robustness of LLMs to Noisy Context in Math Problem Solving
description: >-
  [ACL 2025][LLM (Other)][LLM Robustness] This work proposes ArithmAttack, which evaluates the robustness of LLMs by randomly inserting punctuation marks into math problem contexts (without altering any words). It reveals that eight popular LLMs (including Llama3, Mistral, and DeepSeek) suffer significant performance degradation when facing such simple noise.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "LLM Robustness"
  - "Mathematical Reasoning"
  - "Adversarial Attack"
  - "Punctuation Noise"
  - "ArithmAttack"
date: 2026-05-08
content_hash: c6980761a46c89e3
---

# ArithmAttack: Evaluating Robustness of LLMs to Noisy Context in Math Problem Solving

**Conference**: ACL 2025  
**arXiv**: [2501.08203](https://arxiv.org/abs/2501.08203)  
**Code**: None  
**Area**: LLM / Robustness  
**Keywords**: LLM Robustness, Mathematical Reasoning, Adversarial Attack, Punctuation Noise, ArithmAttack

## TL;DR
This work proposes ArithmAttack, which evaluates the robustness of LLMs by randomly inserting punctuation marks into math problem contexts (without altering any words). It reveals that eight popular LLMs (including Llama3, Mistral, and DeepSeek) suffer significant performance degradation when facing such simple noise.

## Background & Motivation
**Background**: LLMs continue to advance in mathematical reasoning tasks, yet their robustness to input noise remains understudied. Prior works have investigated attacks such as synonym substitution, spelling errors, and irrelevant context injection.

**Limitations of Prior Work**: Existing adversarial attack methods typically alter the semantic content of the text (e.g., replacing words, adding irrelevant information), making it difficult to distinguish whether the performance drop is due to changed semantics or the model's lack of robustness.

**Key Challenge**: Whether there exists an attack method that can degrade the reasoning ability of LLMs without changing semantics at all (maintaining 100% semantic similarity)?

**Goal**: To evaluate the robustness of LLMs to mathematical reasoning when faced with pure punctuation noise (zero semantic change).

**Key Insight**: Inspired by the AEDA data augmentation method, punctuation marks (`,`, `.`, `!`, `?`, `;`, `:`) are randomly inserted into the text without deleting or modifying any existing words.

**Core Idea**: Simply inserting random punctuation marks into math problems can significantly reduce the resolution accuracy of LLMs, even though this noise retains 100% semantic similarity to the original text.

## Method

### Overall Architecture
The methodology is extremely straightforward:
- Input: Original math problem text
- Noise Injection: Random punctuation marks are inserted at random positions, with the quantity corresponding to 10%/30%/50% of the sentence length.
- Output: Evaluation of LLM solving accuracy on noisy text.
- Zero-Shot CoT prompting strategy is utilized.

### Key Designs

1. **Punctuation Noise Injection**:

    - **Function**: Inserts one of six punctuation marks (`.`, `,`, `!`, `?`, `;`, `:`) at random positions within the math problem context.
    - **Mechanism**: No words are deleted, modified, or replaced; punctuation is only inserted between words. The noise ratio is determined by the word count of the sentence (10%/30%/50%).
    - **Design Motivation**: (a) Extremely simple implementation; (b) Semantics remain completely unchanged (Universal Sentence Encoder confirms 100% similarity); (c) Isolates the vulnerability of models to non-semantic perturbations.

2. **Evaluation Metrics**:

    - Accuracy: The percentage of correctly solved problems.
    - Attack Success Rate (ASR): The ratio of problems that were correctly answered before the attack but incorrectly after the attack.
    - Semantic Similarity: Uses USE to verify the semantic consistency between noisy and original text (which is 100%).

## Key Experimental Results

### Main Results (GSM8K)

| Model | Clean Acc(%) | 10% Noise | 30% Noise | 50% Noise | ASR(%) |
|------|-------------|-----------|-----------|-----------|--------|
| Llama3.1-8B | **82.25** | 81.04 | 78.84 | 77.02 | 12.53 |
| Mathstral-7B | 77.63 | 75.51 | 71.34 | 70.65 | 19.81 |
| Llama3-8B | 75.43 | 73.31 | 73.08 | 72.17 | 16.04 |
| DeepSeek-R1-8B | 73.76 | 73.76 | 70.43 | 67.24 | 20.46 |
| Qwen2.5-1.5B | 61.10 | 56.02 | 52.69 | 49.35 | 31.59 |
| Gemma2-2B | 49.65 | 45.10 | 36.46 | 35.63 | 41.82 |
| Mistral-7B | 42.07 | 41.62 | 37.75 | 36.39 | 39.69 |
| Zephyr-7B | 23.27 | 18.04 | 18.04 | 10.08 | **74.80** |

### Main Results (MultiArith)

| Model | Clean Acc(%) | 10% Noise | 30% Noise | 50% Noise | ASR(%) |
|------|-------------|-----------|-----------|-----------|--------|
| Llama3.1-8B | **99.44** | 94.44 | 91.66 | 83.88 | 9.67 |
| Qwen2.5-1.5B | 97.22 | 94.44 | 85.55 | 83.88 | 11.04 |
| Mathstral-7B | 96.11 | 92.77 | 86.11 | 87.22 | 9.47 |
| Llama3-8B | 95.00 | 92.77 | 91.66 | 88.33 | 7.79 |
| Zephyr-7B | 37.22 | 22.22 | 16.11 | 12.77 | **77.10** |

### Key Findings
- **All Models are Vulnerable**: All evaluated models from 1.5B to 8B show performance drops in the presence of punctuation noise.
- **More Noise, Worse Performance**: 50% noise causes a greater performance drop than 10% noise, showing a monotonic decrease.
- **Weaker Models are More Vulnerable**: Zephyr (with a Clean Acc of only 23.27%) exhibits an ASR as high as 74.8%, while Llama3.1 (Clean Acc 82.25%) has an ASR of only 12.53%.
- **Math-Specific Models (Mathstral) are Not Immune**: Although Mathstral achieves a high clean performance, it is still affected by noise.
- **100% Semantic Similarity**: The noise does not change semantics at all, indicating that LLM vulnerability stems from surface form perturbations rather than semantic shifts.

## Highlights & Insights
- **Minimalist yet Effective Attack**: Simply inserting punctuation marks can disrupt reasoning at near-zero implementation cost, revealing LLMs' over-sensitivity to text format rather than content.
- **100% Semantic Preservation**: Unlike word substitution or adding irrelevant sentences, ArithmAttack does not change semantics at all, offering a purer test for format robustness.
- **Positive Correlation between Ability and Robustness**: Models with stronger clean performance exhibit lower ASR (higher robustness), suggesting that enhancements in model capability naturally bring about a degree of robustness.

## Limitations & Future Work
- Only models $\le 8\text{B}$ were tested; larger or stronger models like GPT-4 or Claude were not evaluated.
- Limited to mathematical reasoning; generalization to other reasoning tasks (logic, code generation, etc.) was not investigated.
- Lacks a deep analysis of *why* punctuation noise breaks reasoning—whether it is a tokenizer issue or shifts in attention distribution.
- Did not explore mitigation strategies (such as text-preprocessing denoising or robust training).
- Evaluated only with Zero-Shot CoT, leaving the mitigation potential of Few-Shot or other prompting strategies untested.

## Related Work & Insights
- **vs AEDA**: ArithmAttack directly borrows the punctuation insertion method from AEDA, shifting the perspective from data augmentation to adversarial attacks.
- **vs Spelling Error Attacks (Gan et al.)**: Spelling errors modify existing tokens, whereas punctuation insertion keeps existing tokens intact. Thus, it is more "harmless" yet remains highly effective.
- **vs Synonym Substitution (Zhou et al. RobustMath)**: Synonym substitution inevitably changes semantics, whereas ArithmAttack guarantees a 100% semantic similarity, purely evaluating format robustness.
- **Insight**: LLM reasoning might highly depend on "format cleanliness" of the text, emphasizing the need for input cleaning/standardization in practical applications.

## Rating
- Novelty: ⭐⭐⭐⭐ The approach is extremely simple, and the core contribution lies in the shift of perspective (from augmentation to attack), though its depth is somewhat limited.
- Experimental Thoroughness: ⭐⭐⭐⭐ Tested 8 models on 2 datasets across 3 noise levels, but lacks proprietary/large LLMs and mechanism analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear and straightforward, but the overall content is relatively thin.
- Value: ⭐⭐⭐⭐ Highlights an intriguing vulnerability phenomenon, but lacks deep analysis or defense strategies.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Problem-Solving Logic Guided Curriculum In-Context Learning for LLMs Complex Reasoning](problem-solving_logic_guided_curriculum_in-context_learning_for_llms_complex_rea.md)
- [\[ACL 2025\] MathFusion: Enhancing Mathematical Problem-solving of LLM through Instruction Fusion](mathfusion_instruction_fusion.md)
- [\[ACL 2025\] Veracity Bias and Beyond: Uncovering LLMs' Hidden Beliefs in Problem-Solving Reasoning](veracity_bias_llm_hidden_beliefs.md)
- [\[ACL 2025\] Zero-Shot Belief: A Hard Problem for LLMs](zero-shot_belief_a_hard_problem_for_llms.md)
- [\[ACL 2025\] Exploring Explanations Improves the Robustness of In-Context Learning](exploring_explanations_improves_the_robustness_of_in-context_learning.md)

</div>

<!-- RELATED:END -->

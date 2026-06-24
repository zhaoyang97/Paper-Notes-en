---
title: >-
  [Paper Note] Unveiling Attractor Cycles in Large Language Models: A Dynamical Systems View of Successive Paraphrasing
description: >-
  [ACL 2025][Text Generation][Dynamical Systems] Starting from dynamical systems theory, this paper reveals that during successive paraphrasing, the outputs of LLMs converge to stable 2-period attractor cycles instead of exploring a broad paraphrase space, uncovering inherent limitations in the generation capabilities of LLMs.
tags:
  - "ACL 2025"
  - "Text Generation"
  - "Dynamical Systems"
  - "Attractor Cycles"
  - "Successive Paraphrasing"
  - "LLM Generation Diversity"
  - "2-Periodicity"
date: 2026-05-08
content_hash: a94654d69ea439d8
---

# Unveiling Attractor Cycles in Large Language Models: A Dynamical Systems View of Successive Paraphrasing

**Conference**: ACL 2025  
**arXiv**: [2502.15208](https://arxiv.org/abs/2502.15208)  
**Code**: None (the paper mentions it will be released after the anonymity period)  
**Area**: Text Generation  
**Keywords**: Dynamical Systems, Attractor Cycles, Successive Paraphrasing, LLM Generation Diversity, 2-Periodicity

## TL;DR
Starting from dynamical systems theory, this paper reveals that during successive paraphrasing, the outputs of LLMs converge to stable 2-period attractor cycles instead of exploring a broad paraphrase space, uncovering inherent limitations in the generation capabilities of LLMs.

## Background & Motivation
- **Background**: Paraphrase generation is a classic NLP task, and LLMs are capable of generating high-quality paraphrases. Successive paraphrasing (prompting an LLM to repeatedly paraphrase its own output) should theoretically explore a rich space of linguistic forms.
- **Key Challenge**: Intuitively, multiple consecutive rounds of paraphrasing should yield increasingly diverse textual variants (akin to a depth-first search of the paraphrase space). However, practical observations show that LLMs instead lock into a small number of repetitive patterns.
- **Key Insight**: Successive paraphrasing is modeled as a discrete dynamical system $T_{n+1} = P(T_n)$, using the system-theoretic concept of attractors to explain why successive paraphrases from LLMs exhibit periodic convergence behavior.
- **Core Idea**: The paraphrasing function of LLMs, $P: \mathcal{T} \to \mathcal{T}$, contains low-period limit cycles (specifically 2-period limit cycles), where the model continuously reinforces specific textual forms and suppresses exploration.

## Method

### Overall Architecture
The successive paraphrasing process of LLMs is formalized as a discrete dynamical system. Given an initial text $T_0$, a paraphrase sequence $\{T_n\}$ is iteratively generated via $T_{n+1} = P(T_n)$. The sequence is then analyzed using edit distance and other metrics to determine whether it exhibits periodic attractor behavior as described in dynamical systems theory.

### Key Designs
1. **2-periodicity degree $\tau$**

    - **Mechanism**: Quantitatively characterize the 2-periodicity phenomenon in successive paraphrasing.
    - Definition: $\tau = 1 - \frac{1}{M-2}\sum_{i=3}^{M} d(T_i, T_{i-2})$, where $d$ represents the normalized Levenshtein edit distance.
    - Higher $\tau$ values indicate greater similarity between $T_i$ and $T_{i-2}$, representing stronger 2-periodicity.
    - For perfect 2-periodicity, $\tau=1$.

2. **Difference Confusion Matrix**

    - Plotting the edit distances between all steps into a matrix for visualization.
    - The checkerboard pattern of odd/even steps clearly demonstrates the 2-period structure.

3. **Conditional Perplexity and Reverse Perplexity Analysis**

    - Conditional Perplexity $\sigma(T_i | T_{i-1})$: The certainty of the model's generation in the next step.
    - Reverse Perplexity $\hat{\sigma}(T_i | T_{i+1})$: The difficulty of "reconstructing" the previous step from the subsequent step.
    - As iterations progress, both metrics drop rapidly and converge $\to$ bidirectionally predictable system $\to$ locked into a limit cycle.

4. **Generation Diversity Analysis (Vendi Score)**

    - Sampling multiple paraphrases at each step to compute the Vendi score.
    - The drop in perplexity is accompanied by a collapse in diversity $\to$ the model is trapped in basin of attraction.

### Escape Strategy Experiments
- **Alternating Models/Prompts**: Alternating generation among GPT-4o-mini, GPT-4o, Llama3-8B, and Qwen2.5-7B does not break the 2-periodicity $\to$ attractors represent shared statistical optima across models.
- **Increasing Randomness**: Raising the temperature increases variance but does not eliminate the 2-periodicity; excessively high temperatures output gibberish.
- **Complex Prompts**: Using complex prompts that emphasize diversity of sentence structures reduces the periodicity degree from 0.80 to 0.67, but it remains significant.
- **Local Perturbations**: Synonym replacement is mostly ineffective ($\tau$: 0.77 $\to$ 0.73), while word order swapping is more effective ($\to$ 0.62).
- **With Historical Paraphrases**: Generating the next step while considering the previous two steps leads to a 3-period attractor.
- **Sampling Selection Strategies**: Selecting by maximum perplexity reduces periodicity but hurts semantic fidelity; random selection offers the best trade-off.

## Key Experimental Results

### Main Results

| Model | 2-periodicity degree $\tau$ (English) |
|------|----------------------|
| Mistral-7B | 0.71 |
| Llama3-8B | 0.72 |
| Llama3-70B | 0.60 |
| GPT-4o-mini | 0.83 |
| GPT-4o | 0.81 |
| Qwen2.5-7B | 0.86 |
| Qwen2.5-14B | 0.89 |
| Qwen2.5-72B | 0.92 |

### Cross-Task Generalization

| Task | $\tau$ |
|------|--------|
| Paraphrasing | 0.80 |
| Clarification | 0.83 |
| Polishing | 0.86 |
| Formal/Informal Conversion | 0.65 |
| Translation | 0.87 |

### Escape Strategy Comparison

| Method | $\tau$ | Explanation |
|------|--------|------|
| No perturbation | 0.77 | Baseline |
| Synonym replacement | 0.73 | Highly ineffective |
| Word order swapping | 0.62 | Structural perturbations are more effective |
| Random insertion/deletion | 0.66 | Moderate effect |

### Downstream Impact of Data Augmentation

| Strategy | AG News Accuracy | 2-periodicity degree |
|------|-------------|---------|
| No augmentation | 83.10% | - |
| Minimum perplexity strategy | 83.80% | 0.51 |
| Maximum perplexity strategy | 84.41% | 0.33 |

### Key Findings
- All tested LLMs (both open-source and commercial) exhibit 2-period attractor behavior.
- Qwen2.5-72B shows the strongest 2-periodicity across languages ($\tau$=0.92), while Llama3-70B has the weakest ($\tau$=0.60).
- This phenomenon is not limited to paraphrasing; all reversible tasks (e.g., translation, polishing) exhibit similar behavior.
- Alternating among different models fails to break the cycle $\to$ attractors represent a shared statistical property among LLM populations.
- Small vocabulary-level perturbations are insufficient to escape, requiring structural-level alterations instead.

## Highlights & Insights
- **Novel perspective**: Systematically models successive paraphrasing as a dynamical system, introducing concepts like attractors and limit cycles for the first time.
- **Profound phenomenon**: Unveils the self-reinforcement nature of LLMs—the models persistently favor and amplify specific textual patterns.
- **Cross-model commonality**: Perplexity computed by a single model continues to drop on paraphrases generated by other models, suggesting that multiple LLMs "converge to the same statistical optima."
- **Reversibility is key**: The paper points out that the reversibility of a task (i.e., outputs can be transformed back into inputs) is the fundamental cause of limit cycles.
- **Practical impact**: Breaking the attractor cycle directly improves data augmentation benefits (+1.3% AG News accuracy).

## Limitations & Future Work
- Experiments are built on simple paraphrasing prompts; behaviors under complex or highly specific prompts need further validation.
- The fundamental mathematical reasons for reverse perplexity convergence have not been fully elucidated.
- Differences in the effects of varied languages/domains on periodicity have not been thoroughly analyzed.
- **Potential Research Direction**: Can an "anti-attractor" decoding strategy be designed to actively detect and escape periodic orbits during generation? For example, by maintaining an embedding history of previous generations and forcing new outputs away from the historical trajectory.

## Related Work & Insights
- Closely related to research on LLM self-reinforcement: Xu et al. (2022) demonstrated that LLMs tend to repeat previous context and reinforce themselves.
- Linked to the neural text degeneration problem: 2-period attractors can be viewed as implicit repetition in multi-turn scenarios.
- Impact on AI text detection: Sadasivan et al. (2023) utilized successive paraphrasing to evade detection, but this work finds that such evasion is limited due to being trapped in attractors.
- Insight: Any scenario requiring iterative generation by LLMs (e.g., self-refine, iterative revision) should account for the attractor effect.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Analyzing LLM behavior from a dynamical systems perspective is highly novel, backed by a clear conceptual framework.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers multiple models, languages, and tasks with detailed analyses of escape strategies, though it lacks theoretical derivations.
- **Writing Quality**: ⭐⭐⭐⭐ The paper is well-structured with intuitive tables and figures, though some notations are slightly redundant.
- **Value**: ⭐⭐⭐⭐ Offers important insights into understanding the limitations of LLM iterative generation, with practical guidance for applications like data augmentation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Theme-Explanation Structure for Table Summarization Using Large Language Models](theme-explanation_structure_for_table_summarization_using_large_language_models_.md)
- [\[ACL 2025\] An Empirical Study of Many-to-Many Summarization with Large Language Models](an_empirical_study_of_manytomany_summarization.md)
- [\[ICLR 2026\] Unveiling the Potential of Diffusion Large Language Model in Controllable Generation](../../ICLR2026/nlp_generation/unveiling_the_potential_of_diffusion_large_language_model_in_controllable_genera.md)
- [\[ACL 2025\] Dehumanizing Machines: Mitigating Anthropomorphic Behaviors in Text Generation Systems](dehumanizing_machines_anthropomorphic.md)
- [\[ACL 2025\] Tell, Don't Show: Leveraging Language Models' Abstractive Retellings to Model Literary Themes](tell_dont_show_leveraging_language_models_abstractive_retellings_to_model_litera.md)

</div>

<!-- RELATED:END -->

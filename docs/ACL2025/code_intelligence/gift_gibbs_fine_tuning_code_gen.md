---
title: >-
  [Paper Note] GiFT: Gibbs Fine-Tuning for Code Generation
description: >-
  [ACL 2025][Code Intelligence][Code Generation] Proposes Gibbs Fine-Tuning (GiFT), which is inspired by Gibbs sampling. It samples self-generated code from the marginal distribution instead of the conditional distribution through iterative "code $\rightarrow$ description $\rightarrow$ code" translation. Combined with perplexity-guided long-tail data selection, it improves up to 9.8% over standard self-training on APPS+, MBPP+, and CodeInsight.
tags:
  - "ACL 2025"
  - "Code Intelligence"
  - "Code Generation"
  - "Self-Training"
  - "Gibbs Sampling"
  - "Synthetic Data"
  - "Marginal Distribution"
date: 2026-05-08
content_hash: 5eddc774a8446a6c
---

# GiFT: Gibbs Fine-Tuning for Code Generation

**Conference**: ACL 2025  
**arXiv**: [2502.11466](https://arxiv.org/abs/2502.11466)  
**Code**: [https://github.com/Alex-HaochenLi/GiFT](https://github.com/Alex-HaochenLi/GiFT)  
**Area**: Code Intelligence  
**Keywords**: Code Generation, Self-Training, Gibbs Sampling, Synthetic Data, Marginal Distribution  

## TL;DR

Proposes Gibbs Fine-Tuning (GiFT), which is inspired by Gibbs sampling. It samples self-generated code from the marginal distribution instead of the conditional distribution through iterative "code $\rightarrow$ description $\rightarrow$ code" translation. Combined with perplexity-guided long-tail data selection, it improves up to 9.8% over standard self-training on APPS+, MBPP+, and CodeInsight.

## Background & Motivation

**Background**: Training LLMs for code generation using synthetic data has become mainstream. Self-training methods (such as STaR and RFT) prompt the model to generate multiple code candidates, retain correct code passing test cases for fine-tuning, and iterate until performance gains plateau.

**Limitations of Prior Work**: In self-training, all code is sampled from the conditional distribution $P(c|d_i)$—meaning the code is generated conditioned on a fixed description $d_i$. However, the same "intent" can be described in multiple ways, and different descriptions activate different code execution paths. Sampling from a single conditional description over-represents certain code implementations and under-represents others.

**Key Challenge**: The conditional distribution $P(c|d_i)$ $\neq$ the marginal distribution $P(c)$. The former is constrained by the phrasing preferences of specific descriptions, whereas only the latter can cover all valid code implementations in the "intent" space.

**Goal**: To sample code from the marginal distribution while addressing the long-tail imbalance problem in self-generated code.

**Key Insight**: Gibbs sampling is a classic MCMC method that approximates a joint distribution by alternately sampling from conditional distributions of each dimension. Analogously in code generation, executing "code generation from description" and "description generation from code" alternately can approximate the joint distribution of descriptions and codes.

**Core Idea**: To utilize Gibbs-style iterative translation between description $\leftrightarrow$ code to sample self-generated code from the marginal distribution.

## Method

### Overall Architecture
For each description $d_i$ in the seed dataset: (1) generate code $c_1 \sim P(c|d_i)$; (2) back-translate a new description $d_1 \sim P(d|c_1)$ from the code; (3) generate code $c_2 \sim P(c|d_1)$ from the new description; (4) repeat for several iterations. Collect all intermediate code, use perplexity-weighted sampling to select training data, and pair them with the original $d_i$ for fine-tuning.

### Key Designs

1. **Gibbs-style Iterative Translation**:

    - Function: To sample code from the marginal distribution of the joint description-code distribution.
    - Mechanism: Simulates Gibbs sampling by alternately sampling from $P(c|d)$ and $P(d|c)$. At step $k$: $c_k \sim P_\mathcal{M}(c|d_{k-1})$ and $d_k \sim P_\mathcal{M}(d|c_k)$. After multiple steps, $c_k$ approximates the marginal distribution $P(c)$.
    - Design Motivation: Theoretical analysis shows that fine-tuning with code from the marginal distribution is equivalent to taking the expectation of the conditional distribution loss over all equivalent descriptions: $\mathcal{L}_{marg} = \mathbb{E}_{d' \sim P_d}[\mathcal{L}(d')]$, which implicitly reduces the bias introduced by the specific wording of a single description.

2. **Perplexity-Guided Long-Tail Data Selection**:

    - Function: To balance the proportion of head (common) and tail (rare) code implementations in self-generated code.
    - Mechanism: Calculates the perplexity (PPL) of each self-generated code. High PPL code corresponds to the tail of the distribution—which is rare but valuable. Weighted random sampling is used to select training data, where higher PPL indicates a higher selection probability.
    - Design Motivation: Unbalanced self-generated data can lead to representation collapse, where head code is over-learned and tail code is neglected. Perplexity-weighted selection restores long-tail diversity.

3. **Theoretical Guarantees**:

    - Proves that the marginal distribution fine-tuning loss $\mathcal{L}_{marg}$ equals the conditional distribution loss $\mathcal{L}(d_i)$ plus a "debiasing" term.
    - This debiasing term encourages the model to learn code patterns that are consistent across different descriptions.

### Loss & Training
- Standard SFT loss calculated only on code tokens.
- Intermediary generated descriptions are not used for training—they only serve as a bridge for Gibbs sampling.
- Iterative self-training: generates new code in each round using the updated model.

## Key Experimental Results

### Main Results

| Model | Method | MBPP+ | CodeInsight | APPS+(Intro) | APPS+(Interview) |
|------|------|-------|-----------|------------|----------------|
| DeepSeek-Coder-6.7B | Baseline (Conditional) | 66.7 | 36.5 | 44.4 | 15.3 |
| DeepSeek-Coder-6.7B | **GiFT (Marginal)** | **67.9** | **38.8** | **54.2** | **16.0** |
| CodeLlama-7B | Baseline (Conditional) | 55.3 | 28.5 | 34.7 | 8.3 |
| CodeLlama-7B | **GiFT (Marginal)** | **56.0** | **30.5** | **38.9** | **9.2** |

### Ablation Study

| Configuration | Effect | Description |
|------|------|------|
| Conditional vs. Marginal Sampling | Marginal +1.2~9.8% | Core verification—marginal distribution sampling is indeed superior |
| w/o Perplexity Selection | Performance degradation | The gap is particularly pronounced after multiple iterations |
| 1-step Gibbs vs. Multi-step Gibbs | Multi-step is better | More iterations approximate the true marginal distribution more closely |
| Appliable to Distillation | Effective | GiFT is orthogonal and complementary to distillation |

### Key Findings
- The largest improvement is observed on the difficult benchmark (APPS+ Interview, +9.8%)—demonstrating that increased code diversity from marginal sampling is more valuable for challenging tasks.
- Perplexity selection prevents model collapse across multiple iterations; without it, performance begins to decline after 3 rounds of iterative self-training.
- Informing the training with intermediate descriptions (as data augmentation) is also possible, but yields relatively marginal gains.
- GiFT can be combined with distillation methods. Replacing directly conditional-generated code with marginal code generated by GiFT yields better distillation results.

## Highlights & Insights

- The analogy of **Gibbs sampling** is precise and elegant, mapping the alternating translation of description $\leftrightarrow$ code to the alternating conditional sampling of Gibbs sampling with solid theoretical motivation.
- **"Different descriptions activate different code paths"** is an overlooked yet crucial insight. Different phrasings of the same intent (e.g., "search regex match in string" vs. "find regular expression patterns in text") produce code implementations that are semantically equivalent but different in realization.
- **Perplexity as a long-tail detector** is simple and highly effective—high-PPL code is more likely to be a rare yet valid implementation, making it valuable for learning.
- The method is highly generalizable and not limited to code—it can be transferred to other self-training scenarios like mathematical reasoning (iterative question $\leftrightarrow$ solution translation) and translation (iterative source $\leftrightarrow$ target language translation).
- Rigorous theoretical analysis is provided, proving that marginal distribution fine-tuning is equivalent to taking the expectation of the conditional distribution loss over all equivalent descriptions, thereby reducing the bias introduced by specific phrasings.

## Limitations & Future Work

- It requires back-generating descriptions from code (code summarization); the translation quality depends on the model's summarization capability, and low-quality summaries might disrupt the Gibbs chain.
- Gibbs iterations increase the computational cost of data generation, requiring one generation and one summarization step at each transition.
- Validated only on 6.7B/7B models; larger models (such as 33B/70B) might exhibit less conditional bias, which could lead to diminishing returns for GiFT.
- The quality of test cases is critical for filtering correct code; low-quality or incomplete test cases will introduce noise.
- There is no theoretical guarantee for the convergence of multi-step Gibbs sampling—empirically, 3 steps work well, but when convergence occurs depends on the task.

## Related Work & Insights

- **vs. RFT/STaR**: Standard self-training samples from $P(c|d)$, whereas GiFT samples from $P(c)$, achieving broader coverage.
- **vs. InverseCoder**: InverseCoder uses code to back-generate descriptions for data augmentation, whereas GiFT's iterative translation is closer to the joint distribution than a single-turn backward generation.
- **vs. MathGenie**: MathGenie performs a similar "solution $\rightarrow$ problem" back-translation in the math domain. GiFT systematizes this approach for the code domain and adds solid theoretical analysis.

## Rating

- Novelty: ⭐⭐⭐⭐ The Gibbs sampling analogy is novel, backed by solid theoretical analysis.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on two models across four datasets with thorough ablations, though the scale of the models is limited.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation, rigorous theoretical derivations, and intuitive illustrations.
- Value: ⭐⭐⭐⭐ An important improvement to self-training methodologies, generalizable to multiple domains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] EffiCoder: Enhancing Code Generation in Large Language Models through Efficiency-Aware Fine-tuning](../../ICML2025/code_intelligence/efficoder_enhancing_code_generation_in_large_language_models_through_efficiency-.md)
- [\[ICML 2025\] SparseLoRA: Accelerating LLM Fine-Tuning with Contextual Sparsity](../../ICML2025/code_intelligence/sparselora_accelerating_llm_fine-tuning_with_contextual_sparsity.md)
- [\[AAAI 2026\] Unintended Misalignment from Agentic Fine-Tuning: Risks and Mitigation](../../AAAI2026/code_intelligence/unintended_misalignment_from_agentic_fine-tuning_risks_and_m.md)
- [\[NeurIPS 2025\] Principled Fine-tuning of LLMs from User-Edits: A Medley of Preference, Supervision, and Reward](../../NeurIPS2025/code_intelligence/principled_fine-tuning_of_llms_from_user-edits_a_medley_of_preference_supervisio.md)
- [\[ACL 2025\] Rethinking Repetition Problems of LLMs in Code Generation](rethinking_repetition_problems_of_llms_in_code_generation.md)

</div>

<!-- RELATED:END -->

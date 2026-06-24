---
title: >-
  [Paper Note] What Really Matters in Many-Shot Attacks? An Empirical Study of Long-Context Vulnerabilities in LLMs
description: >-
  [ACL 2025][LLM Efficiency][many-shot jailbreaking] This study systematically analyzes the key factors of Many-Shot Jailbreaking (MSJ) attacks, finding that context length is the decisive factor in attack success, while content harmfulness, topic, and format are nearly irrelevant—even repeating safe content or random meaningless text (Lorem Ipsum) can break the safety alignment of the model in long contexts.
tags:
  - "ACL 2025"
  - "LLM Efficiency"
  - "many-shot jailbreaking"
  - "long-context"
  - "LLM security"
  - "alignment"
  - "context length vulnerability"
date: 2026-05-08
content_hash: abe6df5bd8446c8d
---

# What Really Matters in Many-Shot Attacks? An Empirical Study of Long-Context Vulnerabilities in LLMs

**Conference**: ACL 2025  
**arXiv**: [2505.19773](https://arxiv.org/abs/2505.19773)  
**Code**: None  
**Area**: LLM Efficiency  
**Keywords**: many-shot jailbreaking, long-context, LLM security, alignment, context length vulnerability

## TL;DR
This study systematically analyzes the key factors of Many-Shot Jailbreaking (MSJ) attacks, finding that context length is the decisive factor in attack success, while content harmfulness, topic, and format are nearly irrelevant—even repeating safe content or random meaningless text (Lorem Ipsum) can break the safety alignment of the model in long contexts.

## Background & Motivation
**Background**: LLM context windows are continuously expanding to 128K+ tokens, and MSJ attacks jailbreak models by injecting a large number of harmful QA examples into the context.

**Limitations of Prior Work**: The original MSJ work suggests that carefully crafted harmful examples are necessary, but the exact mechanism through which MSJ succeeds remains unclear—does the harmful content of the examples drive the jailbreak, or is it merely motivated by the context length?

**Key Challenge**: If attack success depends solely on context length rather than content, all defense strategies based on content filtering will fail.

**Goal**: To systematically disentangle the independent contributions of various factors (shot density, topic, harmfulness, and format) in MSJ attacks.

**Key Insight**: Design controlled experiments—fixing the context length while varying the number of shots, topics, harmfulness levels, and formats—to observe changes in ASR (Attack Success Rate).

**Core Idea**: The essence of MSJ is an architectural vulnerability in long-context processing that is entirely independent of the harmfulness of the injected content.

## Method

### Overall Architecture
Attack prompt = Instruction + Examples + Target query. Under a 128K context length, the system systematically varies four dimensions of the Examples: (1) shot density (128/512/2048 shots), (2) topic (6 categories including Adult/Criminal/Cyber), (3) harmfulness (Harmful/Safe/Mixed/Fake), and (4) format (QA/Text/Fake-Text). The Llama-3.1/3.2 and Qwen-2.5 series models are evaluated.

### Key Designs
1. **Shot Density Experiment**:

    - Finding: The ASR pattern is primarily determined by context length rather than the number of shots, and density only affects the onset of the degradation phase.
   
2. **Harmfulness Comparative Experiment (Core Finding)**:

    - Function: Compare the ASR of three datasets: Harmful-512, Safe-512, and Mixed-512.
    - Key Result: The ASR of Safe-512 is comparable to or even higher than that of Harmful-512 (especially on Llama models).
    - Significance: The models do not "learn harmful patterns" but rather generally lose safety constraints under long context lengths.

3. **Fake Data / Lorem Ipsum Attack**:

    - Function: Test attack effectiveness by filling the context with meaningless text.
    - Results: The ASR of Fake-512 and Fake-Text (Lorem Ipsum) is comparable to Harmful-512, and even higher on Llama-3.1.
    - Significance: Attacks are completely independent of content semantics, revealing an architectural vulnerability.

### Three-Stage Vulnerability Pattern
Experiments consistently reveal a three-stage ASR pattern:
- **Initial Vulnerability**: The first surge in ASR occurs at around 512-1024 tokens.
- **Degradation Phase**: ASR decreases (consistent with performance degradation in long-contexts).
- **Rebound Phase**: ASR rises sharply as the context length approaches its maximum limit.

## Key Experimental Results

### Content-Harmfulness-Independent Attack Performance

| Dataset | Llama-3.1-8B ASR | Llama-3.1-70B ASR | Content Type |
|--------|-----------------|-------------------|---------|
| Harmful-512 | ~40% | ~30% | Harmful QA |
| Safe-512 | ~50% | ~40% | Safe QA |
| Fake-512 | ~45% | ~35% | Meaningless QA |
| Fake-Text (Lorem Ipsum) | ~55% | ~45% | Meaningless Text |

### Shot-Repetitive Attack

| Configuration | Performance |
|------|------|
| Harmful-Same-512 (Repeating a single harmful QA) | ASR $\ge$ Harmful-512 |
| Safe-Same-512 (Repeating a single safe QA) | Highest ASR on Llama |

### Key Findings
- Context length is the primary determinant of ASR, with a sharp increase observed near $2^{17}$ tokens.
- Topic selection has no significant impact on ASR, refuting the conclusion by Anil et al. (2024) that topic diversity enhances attacks.
- Instruction-tuned models exhibit an increased MSJ vulnerability to safe content (whereas the base model has proper defense against safe content).
- Larger models (70B) are unexpectedly more susceptible to Fake data attacks than smaller models.

## Highlights & Insights
- The finding that "Lorem Ipsum can also jailbreak" is highly impactful, completely refuting defense strategies based on content filtering.
- It reveals a side effect of instruction tuning: while it improves defense against harmful content, it introduces new vulnerabilities when filled with safe content.
- The three-stage vulnerability pattern (initial vulnerability $\rightarrow$ degradation $\rightarrow$ rebound) consistently appears across multiple models, indicating that it is an architectural issue rather than a training issue.
- The findings have a fundamental impact on safety research directions: focus should shift toward position-aware safety mechanisms rather than content filtering.

## Limitations & Future Work
- Only open-source models (Llama/Qwen) were tested, without covering closed-source models such as GPT-4o/Claude that have built-in safety filters.
- The test set of 50 queries is limited in size.
- The study does not deeply analyze why long contexts cause safety alignment to fail at the architectural level.
- No concrete defensive solutions are proposed.

## Related Work & Insights
- **vs Anil et al. (2024) MSJ Original Work**: The original work suggests that harmful examples and topic diversity are required; this paper demonstrates that neither is critical.
- **vs Lost-in-the-middle**: Performance degradation in long contexts is consistent with the degradation phase identified in this study.
- **vs Content Filtering Defenses**: The results in this paper indicate that such defenses are completely ineffective against MSJ.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The discovery of Lorem Ipsum jailbreaking completely overturns the understanding of the MSJ attack mechanism.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Rigorous variable control across 4 dimensions $\times$ multiple datasets $\times$ multiple models $\times$ 128K context length.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and intuitive visualizations.
- Value: ⭐⭐⭐⭐⭐ Provides fundamental insights for LLM safety research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] On Many-Shot In-Context Learning for Long-Context Evaluation](on_many-shot_in-context_learning_for_long-context_evaluation.md)
- [\[ACL 2025\] Efficient Many-Shot In-Context Learning with Dynamic Block-Sparse Attention](efficient_many-shot_in-context_learning_with_dynamic_block-sparse_attention.md)
- [\[ACL 2025\] What are the Essential Factors in Crafting Effective Long Context Multi-Hop Instruction Datasets? Insights and Best Practices](what_are_the_essential_factors_in_crafting_effective_long_context_multi-hop_inst.md)
- [\[ACL 2025\] LADM: Long-context Training Data Selection with Attention-based Dependency Measurement for LLMs](ladm_long_context_data.md)
- [\[ACL 2025\] Mitigating Posterior Salience Attenuation in Long-Context LLMs with Positional Contrastive Decoding](mitigating_posterior_salience_attenuation_in_long-context_llms_with_positional_c.md)

</div>

<!-- RELATED:END -->

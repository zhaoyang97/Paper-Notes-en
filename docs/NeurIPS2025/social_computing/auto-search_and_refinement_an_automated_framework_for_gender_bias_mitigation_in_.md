---
title: >-
  [Paper Note] Auto-Search and Refinement: An Automated Framework for Gender Bias Mitigation in LLMs
description: >-
  [NeurIPS 2025][Social Computing][gender bias mitigation] This paper proposes FaIRMaker, a framework that adopts an "auto-search + refinement" paradigm: it first employs gradient-based optimization to identify debiasing trigger tokens (Fairwords), then trains a seq2seq model to transform them into human-readable instructions, effectively mitigating gender bias on both open-source and closed-source LLMs while preserving or even improving task performance.
tags:
  - NeurIPS 2025
  - Social Computing
  - gender bias mitigation
  - automated prompt search
  - Fairwords
  - debiasing
  - LLM fairness
date: 2026-05-08
content_hash: a1e02abe60e1d145
---

# Auto-Search and Refinement: An Automated Framework for Gender Bias Mitigation in LLMs

**Conference**: NeurIPS 2025
**arXiv**: [2502.11559](https://arxiv.org/abs/2502.11559)
**Code**: [GitHub](https://github.com/SavannahXu79/FaIRMaker)
**Area**: Social Computing
**Keywords**: gender bias mitigation, automated prompt search, Fairwords, debiasing, LLM fairness

## TL;DR
This paper proposes FaIRMaker, a framework that adopts an "auto-search + refinement" paradigm: it first employs gradient-based optimization to identify debiasing trigger tokens (Fairwords), then trains a seq2seq model to transform them into human-readable instructions, effectively mitigating gender bias on both open-source and closed-source LLMs while preserving or even improving task performance.

## Background & Motivation
**Background**: LLMs encode social biases (particularly gender bias) during pretraining. Existing mitigation methods fall into two categories: parameter modification (fine-tuning / model editing) and instruction-guided approaches (manually designed debiasing preambles).

**Limitations of Prior Work**: Parameter modification methods are resource-intensive and inapplicable to closed-source models. Manually designed debiasing instructions (e.g., counterfactual preamble CF-D), while reducing bias, significantly degrade normal task performance, as the inserted gender-related descriptions interfere with the model's understanding of the original query.

**Key Challenge**: Automated gradient-based search methods (e.g., GCG triggers) can discover effective debiasing tokens over a larger search space, but produce meaningless token sequences that are neither interpretable nor transferable to closed-source models.

**Goal**: Design a method that simultaneously satisfies three criteria: (1) fully automated, (2) compatible with both open-source and closed-source models, and (3) preserving normal task performance.

**Key Insight**: Combine the advantages of gradient-based search and manual design — first automatically search for trigger tokens (large search space), then refine them into readable instructions via a seq2seq model (transferability).

**Core Idea**: Use GCG to automatically search for debiasing trigger tokens, then train a refiner to convert them into natural language instructions.

## Method

### Overall Architecture
FaIRMaker consists of two stages: **Auto-Search** (optimize debiasing Fairwords on a preference dataset using GCG + filter effective Fairwords into a Fairwords Bag) → **Refinement** (use ChatGPT for reverse inference to refine Fairwords into readable instructions; train a seq2seq refiner to generalize the refinement process). At inference time, a randomly selected Fairword is concatenated with the user query and fed into the refiner, whose output instruction is then passed to the target LLM.

### Key Designs

1. **Automatic Fairwords Search and Filtering**:

    - **Function**: Apply the GCG optimizer on a preference dataset to search for universal trigger tokens that, when appended to gender-related queries, increase the probability of the chosen response and decrease that of the rejected response.
    - **Mechanism**: The objective $s^* = \min_s -\log f_\theta(y_c|s \oplus x) + \alpha \log f_\theta(y_r|s \oplus x)$ uses Bayesian optimization to balance promoting fair responses and suppressing biased ones.
    - **Filtering**: Llama3.1-8b-instruct serves as an evaluator to compare response quality and bias level with and without Fairwords on a validation set; only effective Fairwords are retained.
    - **Design Motivation**: Automated search addresses the limited coverage of manually designed prompts; filtering ensures that optimized Fairwords are genuinely effective.

2. **ChatGPT-Assisted Refinement**:

    - **Function**: ChatGPT performs "reverse inference" to analyze why Fairwords are effective and converts them into human-readable instructions.
    - **Mechanism**: Refinement is performed separately for bias-related and normal tasks — bias-task refinement produces debiasing instructions, while normal-task refinement produces quality-enhancing instructions, yielding 9K samples each for a total of 18K.
    - **Design Motivation**: Preserve the large search-space advantage of gradient-based search while gaining the readability and transferability of manual instructions.

3. **Seq2seq Refiner Training**:

    - **Function**: Train Llama3.2-3b-instruct as a refiner to automatically convert Fairwords + query into refined instructions.
    - **Mechanism**: Training loss $\mathcal{L} = -\frac{1}{N}\sum_{t=1}^N \log \mathcal{F}_{refine}(p|s \oplus x)$ trains the refiner to generate debiasing instructions for biased queries and performance-preserving instructions for normal queries.
    - **Design Motivation**: Makes FaIRMaker a standalone module that requires no access to LLM parameters, adaptively generating different types of instructions at inference time.

### Loss & Training
- Fairwords search: GCG optimizer, optimized on the GenderAlign preference dataset.
- Refiner training: Standard seq2seq loss, 18K mixed data (9K bias-related + 9K normal).

## Key Experimental Results

### Bias Mitigation (BBQ-gender)

| Model | sDIS (Original→FM) | sAMB (Original→FM) |
|-------|-------------------|-------------------|
| Llama2-Alpaca | 1.066→**0.224** | 0.804→**0.157** |
| Llama2-Chat | 2.233→**0.273** | 1.673→**0.189** |
| Qwen2-Instruct | 4.638→**1.906** | 1.377→**0.320** |
| Qwen2.5-Instruct | 1.212→**0.431** | 0.030→**0.012** |

### Task Performance Preservation (GA-test dialogue quality)

| Model | Original | FaIRMaker | CF-D | Desc-D |
|-------|----------|-----------|------|--------|
| Llama2-Alpaca | 3.71 | **4.07** | 3.50↓ | 3.32↓ |
| Llama2-Chat | 4.56 | **4.73** | 4.00↓ | 4.00↓ |
| GPT3.5-turbo | 4.73 | **4.90** | 4.68↓ | 4.65↓ |

### Ablation Study

| Configuration | Bias Mitigation | Task Performance |
|---------------|----------------|-----------------|
| Full FaIRMaker | Best | Best |
| Fairwords only (no refinement) | Bias reduced but unstable | Potentially degraded |
| Manual instructions only | Moderate bias reduction | Significantly degraded |

### Key Findings
- FaIRMaker consistently reduces bias by 50%+ across all models while simultaneously improving dialogue quality.
- Conventional manual preambles (CF-D, Desc-D) significantly degrade task performance while reducing bias.
- The effectiveness of Fairwords appears to be positively correlated with the sentiment they express — Fairwords with positive sentiment tend to be more effective.
- FaIRMaker is compatible with the closed-source model GPT-3.5-turbo, yielding consistent debiasing effects.

## Highlights & Insights
- **Search + Refinement Paradigm**: Elegantly combines the search capability of white-box methods with the transferability of black-box methods, avoiding the limitations of each approach individually.
- **Adaptive Refiner**: Automatically switches behavior based on the type of input query — generating debiasing instructions for biased queries and quality-enhancing instructions for normal queries; this "context-awareness" is far more flexible than fixed preambles.
- **Fairwords Interpretability Analysis**: The finding that automatically searched effective triggers correlate with "positive sentiment" provides a new perspective for understanding the debiasing mechanism.

## Limitations & Future Work
- **Gender bias only**: The mitigation of other bias types such as race and age has not been evaluated.
- **Fairwords search requires white-box access**: Although white-box access is not needed at inference time, the search stage does require it (executed on Llama2), limiting the diversity of Fairwords sources.
- **Evaluation metric limitations**: The win-tie-loss metric relies on LLM-as-judge, which may introduce evaluation bias.

## Related Work & Insights
- **vs. CF-D / Desc-D (manual preambles)**: FaIRMaker avoids the query comprehension interference caused by directly injecting gender-related preambles into the prompt.
- **vs. Sheng et al. (2020) automatic triggers**: Those triggers are neither readable nor transferable; FaIRMaker addresses both issues through the refinement step.
- **vs. MBIAS (post-processing)**: FaIRMaker guides at the input side without modifying outputs, offering greater flexibility.

## Rating
- Novelty: ⭐⭐⭐⭐ The search + refinement paradigm is innovative; the idea of converting adversarial triggers into readable instructions is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Five LLMs, multiple benchmarks, and detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear, comparisons are fair, and examples intuitively illustrate the issues.
- Value: ⭐⭐⭐⭐ A practical debiasing tool; model-agnostic design enhances applicability.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] SPAGBias: Uncovering and Tracing Structured Spatial Gender Bias in Large Language Models](../../ACL2026/social_computing/spagbias_uncovering_and_tracing_structured_spatial_gender_bias_in_large_language.md)
- [\[NeurIPS 2025\] IF-GUIDE: Influence Function-Guided Detoxification of LLMs](if-guide_influence_function-guided_detoxification_of_llms.md)
- [\[AAAI 2026\] Bias Association Discovery Framework for Open-Ended LLM Generations](../../AAAI2026/social_computing/bias_association_discovery_framework_for_open-ended_llm_generations.md)
- [\[NeurIPS 2025\] DeepTraverse: A Depth-First Search Inspired Network for Algorithmic Visual Understanding](deeptraverse_a_depth-first_search_inspired_network_for_algorithmic_visual_unders.md)
- [\[NeurIPS 2025\] Noise-Robustness Through Noise: A Framework Combining Asymmetric LoRA with Poisoning MoE](noise-robustness_through_noise_a_framework_combining_asymmetric_lora_with_poison.md)

<!-- RELATED:END -->

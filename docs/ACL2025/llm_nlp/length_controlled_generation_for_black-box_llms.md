---
title: >-
  [Paper Note] Length Controlled Generation for Black-box LLMs
description: >-
  [ACL 2025][LLM (Other)][Length Control] An iterative sampling framework based on the Metropolis-Hastings algorithm, integrated with an importance sampling acceleration strategy, is proposed to achieve precise length control for black-box LLMs **without modifying model parameters**. It achieves a **100%** length control success rate on Llama3.1 in at most 5 iterations, without compromising generation quality.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Length Control"
  - "Black-box LLMs"
  - "Metropolis-Hastings Sampling"
  - "Importance Sampling"
  - "Iterative Inference"
  - "Tuning-Free"
date: 2026-05-08
content_hash: 4b48c67db7edc29f
---

# Length Controlled Generation for Black-box LLMs

**Conference**: ACL 2025  
**arXiv**: [2412.14656](https://arxiv.org/abs/2412.14656)  
**Code**: Not released  
**Authors**: Yuxuan Gu, Wenjie Wang, Xiaocheng Feng, Weihong Zhong, Kun Zhu, Lei Huang, Tat-Seng Chua, Bing Qin  
**Institution**: Harbin Institute of Technology, National University of Singapore, Peng Cheng Laboratory  
**Area**: LLM/NLP / Controllable Text Generation  
**Keywords**: Length Control, Black-box LLMs, Metropolis-Hastings Sampling, Importance Sampling, Iterative Inference, Tuning-Free

## TL;DR

An iterative sampling framework based on the Metropolis-Hastings algorithm, integrated with an importance sampling acceleration strategy, is proposed to achieve precise length control for black-box LLMs **without modifying model parameters**. It achieves a **100%** length control success rate on Llama3.1 in at most 5 iterations, without compromising generation quality.

## Background & Motivation

**Background**: LLMs excel at instruction following, but precisely controlling the length of output text remains a difficult challenge. Subword tokenization and autoregressive decoding make it hard for models to accurately perceive and control the number of words.

**Importance of Length Control**:
   - Summary generation requires specific lengths to balance informativeness and conciseness.
   - Length bias introduced by preference alignment (RLHF/DPO) causes models to favor longer responses, affecting evaluation fairness (Singhal 2023).
   - In practical applications, users often need to specify response lengths (e.g., "summarize within 100 words").

**Limitations of Prior Work**:
   - Fine-tuning-based methods (Yuan et al. 2024; Wang et al. 2024) require modifying model parameters, which is computationally expensive and may damage general capabilities.
   - Reinforcement learning methods (Stiennon et al. 2020) also require training.
   - These methods **cannot be applied to black-box API models** (such as GPT-4).

**Core Motivation**: To design an inference-stage length control method that treats the LLM as an unmodifiable black-box component, activating its intrinsic length-following capability.

## Method

### Overall Architecture: Metropolis-Hastings Iterative Sampling

Length-controlled generation is modeled as a sampling problem from a target distribution $\pi(y|x) \propto f(y)P(y|x)$, where:
- $P(y|x)$: The text generation probability distribution of the LLM.
- $f(y)$: The length constraint scoring function.
- Goal: To find text that simultaneously satisfies length constraints and exhibits high generation quality.

Since direct sampling from $\pi(y|x)$ is infeasible (the partition function is intractable), the Metropolis-Hastings algorithm from MCMC is utilized for iterative approximation.

### Key Designs 1: Length Constraint Score $f(y)$

- An NLTK word tokenizer is used to calculate the word count $\text{Len}(y)$.
- **Precise length target**: $f(y) = 1 / |\text{Len}(y) - \ell|$, where smaller deviations yield higher scores.
- **Range length target**: Within the interval $[\ell_1, \ell_2]$, $f(y) = +\infty$ (immediate acceptance), and decreases with distance outside the interval.
- Analogy to the Lagrangian method: $\log f(y)$ acts as the constraint, and $\log P(y|x)$ serves as the regularization objective.

### Key Designs 2: LLM Probability Estimation (LLM-as-Judge)

Since $P(y|x)$ cannot be directly obtained for black-box LLMs, a dual strategy is employed:
- **Absolute Scoring** $\phi(y|x)$: Prompting the LLM to score the generated text across multiple predefined dimensions.
    - Summarization task: information coverage, fluency, conciseness, logical coherence, faithfulness.
    - Instruction following: usefulness, relevance, accuracy, depth, creativity, detail level.
- **Pairwise Comparison** $\Phi(y_i, y_{i-1}|x)$: Directly comparing two candidates from adjacent iteration steps to reduce score variance.

### Key Designs 3: Proposal Distribution and Importance Sampling Acceleration

**Basic Proposal Distribution** $p(y_i|y_{i-1}, x)$:
- A symmetric constraint $p(y_i|y_{i-1}) = p(y_{i-1}|y_i)$ is imposed to simplify the calculation of the acceptance rate.
- A "time-unbiased" prompt template is used: asking the LLM to generate a new variant by referring to the previous output.

**Importance Sampling Acceleration** $q(y_i|y_{i-1}, x)$:
- Problem: The basic proposal distribution does not contain updated length signals, allowing the LLM to potentially get stuck in its own errors.
- Solution: Introducing an importance distribution that incorporates length constraints to replace the proposal distribution.
- Adding length guidance such as "please rewrite within approximately $\ell$ words" to the prompt.
- At this point, the acceptance rate is bounded by the original acceptance rate: $\mathcal{A} \leq \min(1, f(y_i)P(y_i|x) / f(y_{i-1})P(y_{i-1}|x))$.
- Although this could theoretically increase the false acceptance rate, the strong generation capabilities of LLMs make this risk negligible in practice.

**Algorithm Flow**:

1. Initialization: $y_0 \sim P(y|x)$ (LLM original output)
2. Iteration (up to $n$ times):
    - Generate candidate $y_i$ via the importance distribution.
    - Calculate acceptance rate $\mathcal{A}(y_{i-1} \to y_i)$.
    - Accept $y_i$ with probability $\mathcal{A}$; otherwise, retain $y_{i-1}$.
3. Parallel sampling (beam search-style) is supported to enhance efficiency.

## Key Experimental Results

### Main Results: Precise Length Control (CNN/DailyMail Summary)

| Model | Method | Accuracy↑ | L1 Error↓ | L2 Error↓ | Rouge-1 |
|------|------|--------|--------|--------|---------|
| Llama2 | Inst | 4.1% | 11.42 | 15.20 | 0.37 |
| Llama2 | **Ours** | **81.6%** | 0.24 | 0.64 | 0.36 |
| Llama3.1 | Inst | 7.7% | 3.88 | 5.10 | 0.38 |
| Llama3.1 | **Ours** | **100.0%** | 0.00 | 0.00 | 0.38 |
| GPT-4 | Inst | 15.7% | 2.10 | 2.67 | 0.36 |
| GPT-4 | **Ours** | **99.2%** | 0.01 | 0.12 | 0.36 |

- Llama3.1 achieves 100% precise length control, with both L1 and L2 errors being 0.
- **Rouge metrics remain almost unchanged**, verifying that length control does not compromise generation quality.

### Main Results: Range Length Control (Alpaca-Eval-LI / MT-Bench-LI)

| Dataset | Model | Inst Accuracy | Ours Accuracy | Win Rate Gain |
|--------|------|-----------|-----------|-------------|
| Alpaca | GPT-4 | 37.2% | **99.2%** | 30.2%→**92.0%** |
| Alpaca | Llama3 | 92.2% | **99.8%** | 76.5%→**83.5%** |
| MT-Bench | GPT-4 | 54.7% | **98.8%** | 27.4%→**63.7%** |

- GPT-4's Win Rate on Alpaca-Eval-LI improves from 30.2% to 92.0%.

### Iteration Count Analysis (Llama3.1, CnnDM)

| Iterations | Accuracy |
|---------|--------|
| 0 | 7.7% |
| 1 | 86.4% |
| 2 | 99.2% |
| 4 | 100.0% |

- A single iteration yields a substantial improvement, reaching perfection by 4 iterations.

### Ablation Study

- MH (without importance sampling): 40.2% accuracy $\to$ MH+IS (with importance sampling): 93.3%.
- Importance sampling is a key factor for performance improvement.
- As beam size increases from 1 $\to$ 16, accuracy increases from 24.6% $\to$ 86.4% (Qwen2.5).
- Parallel sampling effectively improves efficiency.

## Highlights & Insights

1. **Combining Classic and Modern**: Innovatively applying the 1960s Metropolis-Hastings algorithm to modern LLM length control, with an elegant theoretical framework.
2. **True Black-box Method**: No access to model parameters or probability outputs is required; precise length control is achieved solely through API calls.
3. **Almost Zero Degradation**: Rouge scores are almost identical before and after control, proving that the method does not sacrifice content quality.
4. **Extremely High Efficiency**: 100% success rate is achieved within at most 5 iterations, rendering the computational overhead manageable.
5. **Broad Applicability**: Effective across both open-source (Llama series, Qwen) and closed-source (GPT-3.5/4) models.

## Limitations & Future Work

1. **API Call Cost**: Each iteration requires multiple LLM calls (generation + scoring), leading to high costs under API-billing models.
2. **Scoring Accuracy**: Ratings from LLM-as-Judge can be inaccurate, especially for complex tasks.
3. **Symmetry Assumption**: The symmetry constraint of the proposal distribution is only approximately satisfied in practice.
4. **Long Text Scenarios**: The paper primarily evaluates performance on short texts (summarization); the effectiveness of controlling long texts (1000+ words) remains unknown.
5. **Non-length Constraints**: Although the framework can theoretically extend to other constraints (e.g., specific topics or formats), this paper only validates length constraints.

## Related Work

- **LLM Instruction Following**: Studies such as Ouyang et al. 2022 (InstructGPT) and Zhou et al. 2024 enhance instruction-following capabilities.
- **Length Control**: Early methods used special tokens to mark length (Fan et al. 2017) or convolutional block length factors (Liu et al. 2018); recent methods encode length signals into positional encodings, attention units, or natural language instructions (Yuan et al. 2024).
- **MCMC Applications in NLP**: Existing work has applied MCMC to language generation (though not specifically for length control).

## Rating

⭐⭐⭐⭐⭐ (5/5)

- **Novelty**: ⭐⭐⭐⭐⭐ Applying the MH algorithm to LLM length control, featuring a clear and novel theoretical framework.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 6 models, 3 datasets, detailed ablations—extremely thorough data.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Rigorous mathematical derivations, with experimental logic progressing step-by-step.
- **Value**: ⭐⭐⭐⭐⭐ Directly applicable to length control scenarios for any black-box LLM API.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Self-Instructed Derived Prompt Generation Meets In-Context Learning: Unlocking New Potential of Black-Box LLMs](self-instructed_derived_prompt_generation_meets_in-context_learning_unlocking_ne.md)
- [\[ACL 2025\] Contrastive Perplexity for Controlled Generation: An Application in Detoxifying Large Language Models](contrastive_perplexity_controlled_gen.md)
- [\[NeurIPS 2025\] PRESTO: Preimage-Informed Instruction Optimization for Prompting Black-Box LLMs](../../NeurIPS2025/llm_nlp/presto_preimage-informed_instruction_optimization_for_prompting_black-box_llms.md)
- [\[ACL 2025\] MergePrint: Merge-Resistant Fingerprints for Robust Black-box Ownership Verification of Large Language Models](mergeprint_fingerprint_ownership.md)
- [\[ICML 2025\] Towards Universal Offline Black-Box Optimization via Learning Language Model Embeddings](../../ICML2025/llm_nlp/towards_universal_offline_black-box_optimization_via_learning_language_model_emb.md)

</div>

<!-- RELATED:END -->

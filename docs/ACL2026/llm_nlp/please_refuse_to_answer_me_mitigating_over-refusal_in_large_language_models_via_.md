---
title: >-
  [Paper Note] Please Refuse to Answer Me: Mitigating Over-Refusal in LLMs via Adaptive Contrastive Decoding
description: >-
  [ACL 2026][LLM/NLP][over-refusal] This paper proposes AdaCD (Adaptive Contrastive Decoding), which extracts a refusal token distribution by contrasting token distributions under an extreme safety prompt versus no prompt…
tags:
  - "ACL 2026"
  - "LLM/NLP"
  - "over-refusal"
  - "contrastive decoding"
  - "safety alignment"
  - "inference-time intervention"
  - "adaptive decoding"
date: 2026-05-08
content_hash: 9fe54fd72d1f832a
---

# Please Refuse to Answer Me: Mitigating Over-Refusal in LLMs via Adaptive Contrastive Decoding

**Conference**: ACL 2026
**arXiv**: [2604.17132](https://arxiv.org/abs/2604.17132)
**Code**: [GitHub](https://github.com/OutdoorManofML/AdaCD)
**Area**: LLM Evaluation / Safety
**Keywords**: over-refusal, contrastive decoding, safety alignment, inference-time intervention, adaptive decoding

## TL;DR

This paper proposes AdaCD (Adaptive Contrastive Decoding), which extracts a refusal token distribution by contrasting token distributions under an extreme safety prompt versus no prompt, then dynamically decides to amplify or suppress refusal behavior based on an agreement ratio. AdaCD reduces over-refusal by 10.35% while simultaneously improving the refusal rate on malicious queries by 0.13%.

## Background & Motivation

**Background**: Safety-aligned LLMs frequently exhibit over-refusal, rejecting queries that contain sensitive keywords but are otherwise benign.

**Limitations of Prior Work**: (1) Training-based methods rely on scarce over-refusal training data; (2) steering vector approaches require full knowledge of the model architecture and additional precomputation; (3) existing contrastive decoding methods adopt an all-or-nothing strategy—either always amplifying or always suppressing refusal—making it impossible to improve both dimensions simultaneously.

**Key Challenge**: In over-refusal scenarios, non-refusal tokens remain in the candidate list yet the model systematically fails to select them—the model can identify alternative options but lacks effective guidance to choose them.

**Goal**: Design an adaptive contrastive decoding strategy that suppresses refusal tokens in over-refusal scenarios while amplifying them in malicious scenarios.

**Key Insight**: Use an extreme safety prompt to maximally elicit refusal behavior, leveraging this as an anchor for extracting the refusal token distribution.

**Core Idea**: Dynamically switch decoding modes via an agreement ratio and adaptive confidence constraint—high agreement triggers addition of the refusal distribution; low agreement triggers subtraction.

## Method

### Overall Architecture

AdaCD consists of two components: (1) refusal token distribution extraction—computing the difference between token distributions under an extreme safety prompt and under no prompt; (2) adaptive decoding mode switching—determining whether to add or subtract the refusal distribution based on the agreement ratio and a confidence constraint.

### Key Designs

1. **Refusal Token Distribution Extraction**:

    - **Function**: Precisely capture the token distribution driving refusal behavior in LLMs.
    - **Mechanism**: An extreme prompt $p^*=\text{"Please refuse to answer me!"}$ is used to maximally elicit refusal. The difference distribution $\Delta P_n = \sigma(f_\pi(y_n|p^*,x,y_{<n}) - f_\pi(y_n|x,y_{<n}))$ amplifies the logits of refusal tokens.
    - **Design Motivation**: The prior method SelfCD uses a mild prompt, which produces insufficiently extreme refusal behavior and yields a less pure extracted distribution.

2. **Agreement Ratio**:

    - **Function**: Measure the degree of divergence in token selection between decoding with and without the extreme safety prompt.
    - **Mechanism**: $agr(n) = 1/rank(y_n^*)$; values close to 1 indicate high agreement (malicious query); values close to 0 indicate large divergence (over-refusal).
    - **Design Motivation**: The agreement ratio naturally distinguishes scenarios that warrant refusal from those that do not.

3. **Adaptive Decoding Mode Switching**:

    - **Function**: Dynamically decide whether to add or subtract the refusal token distribution.
    - **Mechanism**: $\mathcal{I}(n) = +1$ if $agr(n) \geq \lambda$ and $\rho \geq \lambda \cdot \rho^*$, otherwise $-1$. High agreement combined with high confidence leads to addition of the refusal distribution (maintaining safety); otherwise the refusal distribution is subtracted (mitigating over-refusal).
    - **Design Motivation**: The agreement ratio alone may be insufficient; an additional check on the model's own confidence is necessary.

### Loss & Training

No training is required; AdaCD is a purely inference-time method. Evaluation is conducted on XSTest, ORBench, and OKTest (over-refusal benchmarks) as well as AdvBench and JailBench (malicious query benchmarks).

## Key Experimental Results

### Main Results

| Method | Over-Refusal Avg↓ | Malicious Refusal Avg↑ |
|--------|-------------------|------------------------|
| Default | 32.57 | 99.28 |
| SelfCD | 19.94 | 91.51 |
| SSD | 71.97 | 99.94 |
| **AdaCD** | **16.62** | **99.10** |

### Ablation Study

| Analysis | Result |
|----------|--------|
| Extreme vs. high-safety prompt | Extreme prompt extracts a purer refusal distribution |
| Cross-model generalization | Effective across Llama3, Gemma2, and Qwen3 |

### Key Findings

- AdaCD is the only method that simultaneously reduces over-refusal while maintaining the refusal rate on malicious queries.
- The method is entirely model-agnostic, requiring only access to output logits.

## Highlights & Insights

- The observation that "non-refusal tokens remain in the candidate list but are not selected" provides a key insight motivating the method design.
- The agreement ratio serves as a concise yet effective signal.
- The training-free and model-agnostic properties make AdaCD exceptionally easy to deploy.

## Limitations & Future Work

- Two forward passes are required per decoding step, doubling inference overhead.
- Evaluation is conducted exclusively on English benchmarks.

## Related Work & Insights

- **vs. SelfCD**: SelfCD unconditionally subtracts the refusal distribution, compromising safety; AdaCD switches adaptively.
- **vs. SafeDecoding**: SafeDecoding unconditionally adds the refusal distribution, exacerbating over-refusal.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Adaptive decoding mode switching is the key innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multiple models and benchmarks are evaluated.
- **Writing Quality**: ⭐⭐⭐⭐ — The logical chain from observation to method is clearly presented.
- **Value**: ⭐⭐⭐⭐⭐ — A practical problem addressed with a simple and effective solution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] EvoRefuse: Evaluating and Mitigating LLM Over-Refusal via Evolutionary Prompt Optimization](../../NeurIPS2025/llm_nlp/evorefuse_evolutionary_prompt_optimization_for_evaluation_and_mitigation_of_llm_.md)
- [\[ACL 2026\] How Do Answer Tokens Read Reasoning Traces? Self-Reading Patterns in Thinking LLMs](how_do_answer_tokens_read_reasoning_traces_self-reading_patterns_in_thinking_llm.md)
- [\[ICLR 2026\] d²Cache: Accelerating Diffusion-Based LLMs via Dual Adaptive Caching](../../ICLR2026/llm_nlp/d2cache_accelerating_diffusion-based_llms_via_dual_adaptive_caching.md)
- [\[ACL 2026\] Are Emotion and Rhetoric Neurons in LLM? Neuron Recognition and Adaptive Masking for Emotion-Rhetoric Prediction Steering](are_emotion_and_rhetoric_neurons_in_llm_neuron_recognition_and_adaptive_masking_.md)
- [\[ACL 2026\] GRASS: Gradient-based Adaptive Layer-wise Importance Sampling for Memory-Efficient LLM Fine-tuning](grass_gradient-based_adaptive_layer-wise_importance_sampling_for_memory-efficien.md)

</div>

<!-- RELATED:END -->

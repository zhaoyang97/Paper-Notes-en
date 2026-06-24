---
title: >-
  [Paper Note] Silent Tokens, Loud Effects: Padding in LLMs
description: >-
  [NeurIPS 2025][LLM Efficiency][Padding Token] This paper systematically investigates the effects of padding tokens on LLMs when they are not properly masked. The study finds that even a small number of padding tokens can drift hidden-layer representations, degrade generation quality, and unpredictably shift social biases. Critically, 128 padding tokens raise the harmful prompt attack success rate of Llama-3.1-8B from 8% to 77.5%, effectively constituting a jailbreak.
tags:
  - "NeurIPS 2025"
  - "LLM Efficiency"
  - "Padding Token"
  - "Robustness"
  - "Safety Alignment"
  - "Social Bias"
  - "Inference Deployment"
date: 2026-05-08
content_hash: fc18c24aaaae3651
---

# Silent Tokens, Loud Effects: Padding in LLMs

**Conference**: NeurIPS 2025
**arXiv**: [2510.01238](https://arxiv.org/abs/2510.01238)  
**Code**: Available (reference implementation)  
**Area**: LLM Efficiency / AI Safety
**Keywords**: Padding Token, Robustness, Safety Alignment, Social Bias, Inference Deployment

## TL;DR
This paper systematically investigates the effects of padding tokens on LLMs when they are not properly masked. The study finds that even a small number of padding tokens can drift hidden-layer representations, degrade generation quality, and unpredictably shift social biases. Critically, 128 padding tokens raise the harmful prompt attack success rate of Llama-3.1-8B from 8% to 77.5%, effectively constituting a jailbreak.

## Background & Motivation
**Background**: Padding tokens are widely used in batch inference to align sequence lengths and are theoretically masked out via attention masks, leaving computation unaffected.

**Limitations of Prior Work**: In practice, this assumption is frequently violated. In Hugging Face's `transformers` library, if `attention_mask` is not explicitly passed, padding tokens are treated as valid inputs. Practices such as right-side padding for decoder-only models or repurposing `[EOS]` as the pad token also silently corrupt generation. These are not rare edge cases—padding mishandling is a common pitfall in production pipelines that require batch processing.

**Key Challenge**: Padding is widely regarded as a "harmless technical detail," yet no one has systematically studied how much damage mishandled padding can cause—particularly along critical dimensions such as safety and fairness.

**Goal**: Systematically quantify the impact of padding tokens on LLM behavior across four dimensions: activation representations, generation quality, social bias, and safety.

**Key Insight**: A controlled experimental design—prepending varying numbers of padding tokens (0 to 128) to inputs, deliberately allowing them to participate in computation (simulating common attention mask omission errors), and observing changes in model behavior.

**Core Idea**: Padding is not benign; it drifts activation space, degrades quality, alters bias, and undermines alignment—representing a significantly underestimated deployment robustness risk.

## Method

### Overall Architecture
Given an input $x = \langle t_1, \ldots, t_m \rangle$, a padded variant is constructed as $x_{(k)} = \langle \underbrace{[\text{PAD}], \ldots, [\text{PAD}]}_{k}, t_1, \ldots, t_m \rangle$, where $k \in \{0, 1, 2, 4, 8, 16, 32, 128\}$. An attention mask that treats padding as valid input is explicitly passed. Effects are evaluated across four dimensions.

### Key Designs

1. **Activation Drift Analysis**:

    - **Function**: Computes the cosine similarity between hidden-layer representations of the original input and padded variants at each layer, as well as the Silhouette score for clustering harmful/benign prompts.
    - **Key Finding**: Llama-2-7B/13B and Qwen-1.8B exhibit significant activation drift (sharp similarity drops) with only a small number of padding tokens. The cluster boundary between harmful and benign prompts becomes increasingly blurred as padding increases—directly threatening activation-based safety detection methods.
    - PCA visualizations clearly demonstrate that after 128 padding tokens, the representation spaces of harmful and benign prompts become nearly fully entangled.

2. **Generation Quality Degradation**:

    - **Function**: Evaluates the effect of padding on generation quality using BLEU and BERTScore on TruthfulQA.
    - **Key Finding**: Older models (Llama-2) and smaller models (Qwen-1.8B) suffer severe generation quality degradation after 4+ padding tokens. The Gemma series exhibits the strongest resistance to padding, with performance even slightly improving in some settings.

3. **Social Bias Shift**:

    - **Function**: Tests the effect of padding on demographic bias using the BBQ bias benchmark.
    - **Key Finding**: Bias changes are **unpredictable and category-dependent**—age bias decreases with padding, while appearance bias first increases then decreases. The same model responds entirely differently across bias categories and context types (ambiguous/unambiguous).

4. **Safety Compromise**:

    - **Function**: Tests attack success rate (ASR) on 200 harmful prompts from HarmBench, with responses classified using Llama-Guard-3-8B.
    - **Key Finding**: Llama-3.1-8B achieves an ASR of only 8% with 0 padding tokens and 12% with 32, but **ASR surges to 77.5% with 128 padding tokens**—making padding effectively a jailbreak method.
    - This is consistent with the findings of Yu et al., who show that prepending multiple special tokens near harmful prompts pushes them in the opposite direction of the "refusal direction."

## Key Experimental Results

### Safety Main Results (Llama-3.1-8B, HarmBench)

| Padding Count | Attack Success Rate (ASR) | Notes |
|---|---|---|
| 0 | 8.0% | Normal safety |
| 1 | 7.5% | Negligible effect |
| 4 | 4.5% | Slight decrease |
| 16 | 8.5% | Begins to rise |
| 32 | 12.0% | Noticeable degradation |
| 128 | **77.5%** | Near-complete compromise |

### Generation Quality (BERTScore Trend)

| Model | 0 pad | 8 pad | 128 pad | Notes |
|---|---|---|---|---|
| Llama-2-7B | ~0.88 | ~0.78 | ~0.50 | Severe degradation |
| Llama-3.1-8B | ~0.88 | ~0.87 | ~0.85 | Relatively robust |
| Gemma-2-9B | ~0.88 | ~0.88 | ~0.88 | Nearly unaffected |
| Qwen-1.8B | ~0.82 | ~0.72 | ~0.55 | Small models are fragile |

### Key Findings
- Significant generational differences: Llama-3.x is substantially more robust to padding than Llama-2, suggesting that more recent training may have implicitly addressed this issue.
- Family-level differences: The Gemma series is most robust to padding, while small Qwen models are the most vulnerable.
- The jailbreak effect of 128 padding tokens is striking and requires no carefully crafted adversarial prompt—purely "meaningless" tokens suffice to break alignment.
- The unpredictability of bias shifts is the most concerning finding: one cannot simply conclude that "more padding equals more bias."

## Highlights & Insights
- The discovery of **padding as a jailbreak vector** is highly significant: it reveals that the fragility of current LLM alignment stems not only from carefully engineered adversarial attacks but also from seemingly harmless oversights at the infrastructure level.
- The **PCA visualizations** are highly intuitive, illustrating how padding progressively blurs the activation-space boundary between harmful and benign prompts—a warning signal for activation-based safety detection methods.
- The **comprehensive cross-model-family comparison** offers practical value, informing practitioners about which models are more robust to padding.

## Limitations & Future Work
- Only left-side padding (the standard practice for decoder-only models) is tested; right-side padding and encoder-based models are not covered.
- No training or fine-tuning methods specifically targeting padding robustness are explored—the paper diagnoses the problem without providing a solution.
- Detailed bias analysis is presented only for Llama-3.1-8B; other models are not thoroughly analyzed.
- Safety testing is conducted on a single model (Llama-3.1-8B); the generalizability of the padding jailbreak to other model families remains unverified.
- Experimental scale is limited (128–200 samples), and statistical significance may be insufficient.

## Related Work & Insights
- **vs. Yu et al. (2025)**: Yu et al. show that prepending special tokens can shift prompts along the refusal direction. The padding jailbreak in this paper is a natural extension and systematic validation of that finding.
- **vs. Arditi et al. (2024)**: Arditi et al. find that refusal behavior is governed by a single direction. The PCA visualizations in this paper suggest that padding disrupts alignment precisely by perturbing this direction.
- **Implications for Deployment**: Any LLM deployment pipeline must ensure that padding is rigorously masked; activation-based safety detection methods need to account for padding robustness.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The finding that simple padding can achieve jailbreaking is surprising, though the contribution is primarily a systematic investigation rather than a novel method.
- **Experimental Thoroughness**: ⭐⭐⭐ — Covers 4 dimensions and 10 models, but sample sizes are small and some analyses present results for only a single model.
- **Writing Quality**: ⭐⭐⭐⭐ — Experimental design is clear and visualizations are effective, though the scope is limited given the workshop paper format.
- **Value**: ⭐⭐⭐⭐⭐ — Offers direct practical guidance for LLM deployment; padding robustness should become a standard evaluation dimension.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Ekka: Automated Diagnosis of Silent Errors in LLM Inference](../../ICML2026/llm_efficiency/ekka_automated_diagnosis_of_silent_errors_in_llm_inference.md)
- [\[ICLR 2026\] Flatter Tokens are More Valuable for Speculative Draft Model Training](../../ICLR2026/llm_efficiency/flatter_tokens_are_more_valuable_for_speculative_draft_model_training.md)
- [\[NeurIPS 2025\] Long-Context Modeling with Dynamic Hierarchical Sparse Attention for On-Device LLMs](long-context_modeling_with_dynamic_hierarchical_sparse_attention_for_on-device_l.md)
- [\[CVPR 2026\] Rejection Mixing: Fast Semantic Propagation of Mask Tokens for Efficient DLLM Inference](../../CVPR2026/llm_efficiency/rejection_mixing_fast_semantic_propagation_of_mask_tokens_for_efficient_dllm_inf.md)
- [\[NeurIPS 2025\] LooGLE v2: Are LLMs Ready for Real World Long Dependency Challenges?](loogle_v2_are_llms_ready_for_real_world_long_dependency_challenges.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Weak-to-Strong Jailbreaking on Large Language Models
description: >-
  [ICML 2025][Model Compression][Jailbreak Attack] This paper proposes the weak-to-strong jailbreak attack: using two small models (one safe, one unsafe) during inference to modify the decoding distribution of a large model via log-probability algebra. It requires only a single forward pass to increase the malicious response rate of aligned LLMs to over 99%, revealing a previously unnoticed, highly efficient attack surface in LLM alignment.
tags:
  - "ICML 2025"
  - "Model Compression"
  - "Jailbreak Attack"
  - "Alignment Safety"
  - "Decoding Distribution"
  - "Adversarial Attack"
  - "Large Language Models"
date: 2026-05-08
content_hash: 72b6400f180554b8
---

# Weak-to-Strong Jailbreaking on Large Language Models

**Conference**: ICML 2025  
**arXiv**: [2401.17256](https://arxiv.org/abs/2401.17256)  
**Code**: [https://github.com/XuandongZhao/weak-to-strong](https://github.com/XuandongZhao/weak-to-strong)  
**Area**: Model Compression  
**Keywords**: Jailbreak Attack, Alignment Safety, Decoding Distribution, Adversarial Attack, Large Language Models

## TL;DR
This paper proposes the weak-to-strong jailbreak attack: using two small models (one safe, one unsafe) during inference to modify the decoding distribution of a large model via log-probability algebra. It requires only a single forward pass to increase the malicious response rate of aligned LLMs to over 99%, revealing a previously unnoticed, highly efficient attack surface in LLM alignment.

## Background & Motivation

**Background**: Current LLMs are aligned using methods like RLHF and DPO to refuse harmful queries. However, red-teaming continues to uncover jailbreak vulnerabilities—through adversarial prompts, fine-tuning, or decoding tricks, aligned LLMs can still be induced to generate harmful content.

**Limitations of Prior Work**:
   - Prompt-based jailbreaks (e.g., GCG, AutoDAN) require heavy optimization iterations and are computationally expensive.
   - Fine-tuning-based jailbreaks require training data and GPU resources.
   - Prior methods mostly require multiple queries to the target model or parameter modifications.
   - **The lack of an efficient, low-cost, and highly effective inference-time attack method.**

**Key Challenge**: Aligned and jailbroken models are similar in most of their generation behaviors—they respond almost identically to the vast majority of benign queries. Alignment essentially only changes the initial decoding behavior (refusal or response) of the model when facing sensitive queries.

**Goal**: To reveal an **efficient inference-time attack vector**—leveraging decoding distribution differences between small models to manipulate the behavior of large models, while inspiring better defense mechanisms.

**Key Insight**: It is observed that jailbroken and aligned models mainly differ in their **initial decoding distribution**. This implies that once a model is guided to start answering (rather than refusing), subsequent generations will naturally continue to produce content. Thus, one only needs to inject "de-alignment" signals during the decoding of the first few tokens.

**Core Idea**: Use the difference between the log-probabilities of two small models (safe and unsafe) to calculate a "de-alignment offset" and add it to the decoding distribution of the large model. This is equivalent to subtracting the "safety constraints" from the large model's decoding distribution, making it behave like an unaligned version.

## Method

### Overall Architecture

```
Input: Harmful query x
       ↓
  ┌────────────────────────────┐
  │  Safe small model → logits_safe  │
  │  Unsafe small model → logits_unsafe │
  │  Aligned large model → logits_target │
  └────────────────────────────┘
       ↓
  Modified decoding: logits_target + β × (logits_unsafe - logits_safe)
       ↓
Output: Harmful response y
```

### Key Designs

1. **Log-Probability Algebra**:

    - **Function**: Modifying the decoding distribution of the large model through the difference in logits of the small models.
    - **Core Formula**:
    $\log P_{\text{attack}}(y_t | y_{<t}, x) = \log P_{\text{target}}(y_t | y_{<t}, x) + \beta \cdot \left[\log P_{\text{unsafe}}(y_t | y_{<t}, x) - \log P_{\text{safe}}(y_t | y_{<t}, x)\right]$
   where $P_{\text{target}}$ is the target large model (e.g., 70B aligned), $P_{\text{unsafe}}$ is the unsafe small model (e.g., 7B, trained via shadow alignment), $P_{\text{safe}}$ is the safe small model (e.g., 7B aligned), and $\beta$ is the attack strength hyperparameter.
    - **Design Motivation**: $\log P_{\text{unsafe}} - \log P_{\text{safe}}$ captures the decoding offset in the direction of "unsafety/de-alignment". Applying this offset to the large model is equivalent to removing the safely aligned portion from the large model's decoding distribution. This is similar to the concept of contrastive decoding, but the goal shifts from "improving quality" to "disrupting alignment".

2. **Weak-to-Strong Paradigm**:

    - **Function**: Guiding the attack on the strong (large) model using weak (small) models.
    - **Mechanism**: Selecting two 7B-class small models—one aligned and one unaligned version obtained through shadow alignment. The difference in logits between them represents the "aligned vs. unaligned" direction, which is equally valid in the decoding space of the large model.
    - **Design Motivation**:
        - **Extremely low cost**: Requires only one extra decoding of each of the two small models (vs. thousands of iterations needed for GCG).
        - **Cross-model transferability**: The "de-alignment direction" of the small models can transfer to large models of different architectures.
        - **Stealthiness**: Does not modify target model parameters, requires no gradient access, and only alters the sampling process during inference.

3. **Attack Strength Control ($\beta$ Parameter)**:

    - **Function**: Regulating the intensity of the de-alignment offset.
    - **Mechanism**: No attack when $\beta = 0$; as $\beta$ increases, the attack strength increases but generation quality may degrade. In experiments, $\beta = 1.5$ is the optimal balance point.
    - **Design Motivation**: To achieve a balance between attack success rate (ASR) and generation quality—excessively large $\beta$ leads to degraded text.

### Defense Strategies

As a preliminary attempt, the authors propose a defense scheme:
- **Adversarial Detection**: Monitoring perplexity changes in the generated token sequence—under attack, the perplexity distribution of the first few tokens deviates from normal patterns.
- **Decoding Distribution Constraint**: Imposing KL-divergence constraints on the output distribution during sampling to prevent deviating too far from the original aligned model.
- **Limitations**: The authors acknowledge that designing stronger defenses remains challenging.

### Loss & Training

This method **does not require training**—it is conducted entirely during inference. The only preparation required is to obtain an unsafe small model (which can be fine-tuned using a small amount of harmful data via the Shadow Alignment method).

## Key Experimental Results

### Main Results

| Target Model | Dataset | ASR (%) | Baseline ASR (%) | Note |
|---------|--------|---------|-------------|------|
| Llama-2-70B-Chat | AdvBench | >99% | <5% | Aligned model is almost completely breached |
| Llama-2-70B-Chat | HarmBench | >99% | <5% | Another harmful query benchmark |
| Llama-2-13B-Chat | AdvBench | >99% | <5% | Medium-sized model is equally effective |
| Vicuna-33B | AdvBench | >98% | ~10% | Different alignment methods are also breached |
| Mixtral-8x7B | AdvBench | >97% | <8% | Cross-architecture transfer |

### Comparison with Other Attack Methods

| Method | ASR (%) | Computational Overhead | Requires Gradient? | Requires Training? |
|------|---------|---------|----------|----------|
| GCG | 85% | High (thousands of iterations) | Yes | No |
| AutoDAN | 90% | High | Yes | No |
| Fine-tuning | 95% | Medium (requires GPU training) | - | Yes |
| **Weak-to-Strong** | **>99%** | **Low (1 forward pass)** | **No** | **No** |

### Ablation Study

| Configuration | ASR (%) | Note |
|------|---------|------|
| $\beta = 0.5$ | ~70% | Insufficient attack strength |
| $\beta = 1.0$ | ~92% | Suboptimal |
| $\beta = 1.5$ | >99% | Optimal balance point |
| $\beta = 2.0$ | >99% | High ASR but degraded generation quality |
| Only unsafe model (no safe contrast) | ~80% | Contrastive decoding is crucial |
| Random small model pair (non safe/unsafe pair) | ~15% | Proves alignment difference is the key |

### Key Findings

1. **Initial tokens dominant**: The aligned model and the jailbroken model differ significantly only in the decoding distribution of the first 5-10 tokens. Once the model starts to "answer" rather than "refuse", subsequent generation naturally continues to output content.
2. **Cross-model/Cross-organization transfer**: The "de-alignment direction" of a 7B small model can effectively transfer to 70B+ large models from different architectures and organizations.
3. **Enormous efficiency advantage**: Compared to methods like GCG, the computational cost is almost negligible.
4. **Inadequacy of existing defense strategies**: The defenses proposed by the authors are only partially effective; designing robust defenses remains an open problem.

## Highlights & Insights

- **Profound Insight**: Viewing the alignment mechanism as a localized modification (rather than a global change) of the decoding distribution, revealing the fundamental vulnerability of current alignment methods.
- **The "Dark Side" of Contrastive Decoding**: Contrastive decoding is typically used to improve quality—this paper demonstrates that the exact same technique can disrupt safety alignment.
- **"Weak guiding strong" paradigm**: Forming an interesting echo with weak-to-strong generalization (Burns et al.), but in the completely opposite direction—using a weak model to attack a strong model.
- **Safety Wake-up Call**: The extremely low cost of the attack (requiring only two 7B models + a single inference pass) means the threat to alignment safety is far greater than previously expected.

## Limitations & Future Work

1. **Requirement of obtaining an unsafe small model**: Attackers need an unsafe small model of the same architecture (which can be obtained via Shadow Alignment but requires harmful data).
2. **Only modifying decoding distribution**: For models that achieve deep alignment at the representation level (e.g., Constitutional AI), this attack's effectiveness may be diminished.
3. **Generation quality trade-off**: While high $\beta$ values improve ASR, they reduce generation fluency.
4. **Insufficient exploration of defenses**: The paper acknowledges that the proposed defenses are preliminary, and more comprehensive defense schemes remain to be studied.
5. **Inapplicability to closed-source models**: It requires access to logits, making it not directly applicable to API-only models (such as GPT-4, Claude).

## Related Work & Insights

- **GCG (Zou et al., 2023)**: Achieving jailbreak by optimizing adversarial suffixes via gradients. However, it is costly and requires gradient access.
- **Contrastive Decoding (Li et al., 2023)**: Using the logit difference between large and small models to improve generation quality. This paper "inverts" this concept for attacks.
- **Shadow Alignment (Yang et al., 2023)**: Transforming aligned models into unsafe models using a small amount of harmful data. This paper uses this method to prepare the unsafe small model required for the attack.
- **Insights**:
    - Alignment should not only be implemented "shallowly" at the output layer—deeper representation alignment is needed.
    - Contrastive analysis of decoding distributions can serve as a new dimension for detecting jailbreaks.
    - Alignment methods need to consider the robustness against inference-time attacks, rather than just training-time adversarial robustness.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Reveals a completely new and highly efficient attack vector; the weak-to-strong paradigm is surprising.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 5 models, 3 organizations, 2 datasets, comprehensive ablation, and comparative defense.
- Writing Quality: ⭐⭐⭐⭐ Insights are clearly explained; pipeline is concise and straightforward.
- Value: ⭐⭐⭐⭐⭐ Holds significant warning value for the LLM safety community, pushing forward research into stronger defenses.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] SepLLM: Accelerate Large Language Models by Compressing One Segment into One Separator](sepllm_accelerate_large_language_models_by_compressing_one_segment_into_one_sepa.md)
- [\[ICML 2025\] Persistent Topological Features in Large Language Models](persistent_topological_features_in_large_language_models.md)
- [\[NeurIPS 2025\] Synergy between the Strong and the Weak: Spiking Neural Networks Are Inherently Superior in Temporal Processing](../../NeurIPS2025/model_compression/synergy_between_the_strong_and_the_weak_spiking_neural_networks_are_inherently_s.md)
- [\[ICML 2025\] DLP: Dynamic Layerwise Pruning in Large Language Models](dlp_dynamic_layerwise_pruning_in_large_language_models.md)
- [\[ICML 2025\] From Language Models over Tokens to Language Models over Characters](from_language_models_over_tokens_to_language_models_over_characters.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Understanding the Repeat Curse in Large Language Models from a Feature Perspective
description: >-
  [ACL 2025][LLM (Other)][Repetitive generation] Investigates the repeat curse in LLMs from the perspective of mechanistic interpretability. Specifically, Sparse Autoencoders (SAEs) are used to extract monosemantic features to locate "repeat features" in the middle and final layers. Activating these features induces repetition, whereas turning them off mitigates repetition without compromising model performance.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Repetitive generation"
  - "Sparse Autoencoder"
  - "Mechanistic Interpretability"
  - "Feature Steering"
  - "Detoxification"
date: 2026-05-08
content_hash: 607e241ff760a197
---

# Understanding the Repeat Curse in Large Language Models from a Feature Perspective

**Conference**: ACL 2025  
**arXiv**: [2504.14218](https://arxiv.org/abs/2504.14218)  
**Code**: [https://github.com/kaustpradalab/repeat-curse-llm](https://github.com/kaustpradalab/repeat-curse-llm)  
**Area**: LLM NLP  
**Keywords**: Repetitive generation, Sparse Autoencoder, Mechanistic Interpretability, Feature Steering, Detoxification

## TL;DR
Investigates the repeat curse in LLMs from the perspective of mechanistic interpretability. Specifically, Sparse Autoencoders (SAEs) are used to extract monosemantic features to locate "repeat features" in the middle and final layers. Activating these features induces repetition, whereas turning them off mitigates repetition without compromising model performance.

## Background & Motivation

**Background**: LLMs often generate repetitive content (token-level repetitions or paragraph-level phrase/segment repetitions) during generation tasks, which severely degrades output quality and diversity. Existing methods such as Nucleus Sampling and repetition penalties mitigate repetition at the decoding strategy level.

**Limitations of Prior Work**: Patching methods at the decoding stage can degrade the overall performance of the model and **do not address the fundamental question of "why repetition occurs"**. A few interpretability works (e.g., attention head analysis, FFN activation analysis) only identify network components associated with repetition, but at an insufficiently fine grain—neurons are polysemantic, making it difficult to provide a clear explanation.

**Key Challenge**: Specific "features" causing repetition exist within LLMs, but traditional neuron analysis cannot isolate these features because a single neuron may encode multiple semantics.

**Goal**: (a) Precisely locate "repeat features" that cause repetition using monosemantic features; (b) validate their causal role by activating/deactivating these features; (c) provide a lossless mitigation method for repetition.

**Key Insight**: Sparse Autoencoders (SAEs) can decompose polysemantic neurons into monosemantic features. Recently, SAEs have been used to identify various semantic features (e.g., Golden Gate Bridge features, biological features). This paper applies this approach to identify repeat features.

**Core Idea**: Decompose activations in the middle and final layers of LLMs using SAEs to identify "repeat features", and turn them off to mitigate repetition.

## Method

### Overall Architecture
The paper proposes the "Duplicatus Charm" pipeline, which consists of four steps: (1) construct a repetition dataset (token-level repetitions + paragraph-level repetitions); (2) evaluate and select repetition metrics (n-gram vs. Self-BLEU vs. information entropy); (3) locate repetitive layers through logit analysis, and then identify repeat features via SAE feature activations; (4) perform feature steering—activating repeat features to induce repetition, and turning off repeat features to mitigate repetition.

### Key Designs

1. **Repetition Dataset Construction**:

    - Function: Construct a labeled dataset containing token-level and paragraph-level repetitions to evaluate repetition metrics.
    - Mechanism: Sample 1,000 instances from Orca-Chat. Token repetitions are controlled by parameters $(N, M)$—where $N$ is the starting position of the repetition and $M$ is the length of the repetitive token group. Paragraph repetitions are created by repeating an entire paragraph 5 times. The dataset contains 5,500 samples in total (4,500 token-level + 1,000 paragraph-level).
    - Design Motivation: Since there is no dedicated evaluation dataset for repetition, a custom dataset is constructed to select the best evaluation metric.

2. **Metric Selection**:

    - Function: Select the metric most sensitive to both repetition scenarios from three candidates (n-gram repetition rate, Self-BLEU, information entropy).
    - Results: **1-gram repetition rate** and **1-gram information entropy** perform the best. They demonstrate high discriminability in the token-level repetition scenario (detectable even if replication starts at the 140th token) and achieve a discrimination gap of 0.95 in paragraph-level scenarios. Self-BLEU shows low discriminability (fluctuating by only 0.1).
    - The **1-gram repetition rate** is ultimately selected as the Repeat Score (RS).

3. **Layer Localization**:

    - Function: Find the layers that contribute the most to repetitive generation.
    - Mechanism: Decompose residual streams using the logit lens method to calculate each layer's logit difference contribution to "correct answer = repeated token" vs "incorrect answer = non-repeated token". Eight repetition-inducing templates are designed (e.g., "He hit Jack Jack Jack Jack Jack") to compute the logit contribution of each layer: $$\ell_{\text{contribution}_\ell} = \text{residual}_\ell \cdot \ell_{\text{diff\_direction}}$$
    - Finding: The **middle and final layers** contribute the most to repetition, which is consistent across three models (GPT-2, Gemma-2-2B, Llama-3.1-8B).

4. **Feature Localization**:

    - Function: Extract specific repeat features within the localized layers using SAEs.
    - Mechanism: Use pretrained SAEs corresponding to each model, set a steering coefficient $\lambda = 2$ (1.5-2 times the original activation) for each feature, generate text, and calculate the Repeat Score. Features with RS $\geq \rho = 0.4$ are classified as "repeat features," while others are "ordinary features."
    - SAE decomposition formula: $$\hat{x} = b_{dec} + \sum_{i=1}^F f_i(x) W_{dec,i}$$, where feature activation is $$f(x) = \text{ReLU}(W_{enc} \cdot x + b_{enc})$$
    - Feature steering: $$\hat{X} = X + \lambda \cdot W_{dec}[\text{feature\_idx}]$$

### Mitigation Strategy
- **Turning off repeat features**: Set the activations of repeat features to 0, i.e., reverse steering with $\lambda = 0$.
- This requires no retraining and can be applied during inference.

## Key Experimental Results

### Main Results

| Model-Layer | Dataset | Original RS | Activating Ordinary Features | Activating Repeat Features | Turning Off Repeat Features |
|---|---|---|---|---|---|
| GPT2-small L9 | EQ | 0.37 | 0.37 | **0.72** (↑0.35) | **0.19** (↓0.18) |
| GPT2-small L9 | AQ | 0.25 | 0.27 | **0.58** (↑0.33) | **0.23** (↓0.02) |
| Gemma-2-2B | EQ | 0.23 | 0.24 | **0.52** | **0.11** (↓0.12) |
| Llama-3.1-8B | EQ | 0.19 | 0.20 | **0.45** | **0.10** (↓0.09) |

- Activating repeat features doubles or more than doubles the RS, whereas activating ordinary features causes almost no change—confirming the causal role of repeat features.
- Turning off repeat features reduces the RS by 30-50%, demonstrating significant mitigation effects.

### Ablation Study

| Activation Ratio | 10% | 20% | 50% | 100% |
|---|---|---|---|---|
| GPT2 L9 Activate RF (EQ) | 0.55 | 0.60 | 0.68 | 0.72 |
| GPT2 L9 Turn Off RF (EQ) | 0.33 | 0.32 | 0.21 | 0.19 |

- Larger activation/deactivation ratios lead to stronger effects, demonstrating a monotonic relationship.
- Manipulating even 10% of the repeat features yielded prominent effects.

### Key Findings
- **Repeat features are concentrated in the middle and final layers**: This consistent pattern across three model sizes (117M/2B/8B) suggests that it is a common property of the Transformer architecture.
- **Turning off repeat features does not harm model performance**: On AQ and NQ dialogue tasks, the output quality after turning off repeat features is comparable to the original outputs (confirmed via human evaluation).
- **Repeat features are interpretable**: Leveraging the human-readability of SAEs, characteristics of repeat features can be summarized—they tend to exhibit high activation at positions where tokens have already appeared.
- **Enumeration-type questions are more likely to trigger repetition**: The original RS on the EQ dataset is higher than on AQ/NQ, indicating that tasks requiring diverse outputs are more prone to exposing repetition issues.

## Highlights & Insights
- **Using SAEs to diagnose generation quality issues is a new paradigm**: Historically, SAEs have been primarily used to understand the internal knowledge of models (such as Golden Gate Bridge features). This paper is the first to use them to locate and repair generation bugs (repetition). This methodology can be transferred to other generation issues such as hallucination and bias.
- **Clever naming with "Duplicatus Charm"**: Inspired by Harry Potter, the name makes the paper highly memorable. The pipeline design is also systematic: constructing datasets and selecting metrics, locating layers and features, and finally validating causality.
- **Inference-time mitigation with no retraining**: Turning off features takes effect during inference, which is more targeted than repetition penalties (directly turning off the root cause) and more efficient than retraining.

## Limitations & Future Work
- **Dependence on pretrained SAE quality**: If the SAE does not fully capture the feature directions associated with repetition, key features may be missed.
- **Threshold $\rho = 0.4$ based on manual evaluation**: Different tasks/domains may require different thresholds; generalization needs further verification.
- **Validated on only three models**: GPT2-small (117M) is too small, and Gemma-2-2B and Llama-3.1-8B are medium-sized. The distribution of repeat features in larger models (70B+) might differ.
- **Effect of instruction tuning/RLHF on repeat features is not analyzed**: Repetition is typically mitigated in RLHF-tuned models; comparing the repeat features of base and chat models would be highly valuable.
- Future directions: Extend the method to identify hallucination features; integrate activation patching to precisely quantify the contribution of each repeat feature; explore simultaneously turning off repeat features and enhancing diversity features.

## Related Work & Insights
- **vs. Repetition Penalties / Nucleus Sampling**: Decoding-level methods only treat the symptoms rather than the root cause, potentially harming generation quality. This work targets the root cause (repeat features), offering higher precision.
- **vs. Attention Head Analysis by Vaidya et al. (2023)**: They localized attention heads that copy tokens, but heads are coarse-grained. This work uses SAE features, which are fine-grained and monosemantic.
- **vs. Repetition Neurons by Hiraoka & Inui (2024)**: They analyzed FFN activations to find repetition neurons, but neurons are polysemantic. SAE features are monosemantic, making them easier to interpret and manipulate.
- This SAE-based diagnostic framework could serve as a general-purpose tool for studying LLM generation bugs.

## Rating
- Novelty: ⭐⭐⭐⭐ The perspective of using SAEs to locate repeat features is highly novel, and the pipeline design is systematic.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation using three models, three datasets, activation/deactivation experiments, and metric selection analyses.
- Writing Quality: ⭐⭐⭐⭐ The structure is clear, and the Harry Potter-inspired naming is engaging, though it slightly reduces academic formality.
- Value: ⭐⭐⭐⭐ Provides a new tool for understanding and solving the LLM repetition problem, with strong potential for extending the SAE diagnostic framework.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Perspective Transition of Large Language Models for Solving Subjective Tasks](perspective_transition_of_large_language_models_for_solving_subjective_tasks.md)
- [\[ACL 2025\] Evaluating Implicit Bias in Large Language Models by Attacking from a Psychometric Perspective](evaluating_implicit_bias_in_large_language_models_by_attacking_from_a_psychometr.md)
- [\[ACL 2025\] TableLoRA: Low-rank Adaptation on Table Structure Understanding for Large Language Models](table_lora_structure_understanding.md)
- [\[ACL 2025\] ChronoSense: Exploring Temporal Understanding in Large Language Models with Time Intervals of Events](chronosense_exploring_temporal_understanding_in_large_language_models_with_time_.md)
- [\[ACL 2025\] Contrastive Perplexity for Controlled Generation: An Application in Detoxifying Large Language Models](contrastive_perplexity_controlled_gen.md)

</div>

<!-- RELATED:END -->

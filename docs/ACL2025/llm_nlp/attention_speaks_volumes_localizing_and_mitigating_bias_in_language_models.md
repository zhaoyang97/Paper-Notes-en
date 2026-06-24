---
title: >-
  [Paper Note] Attention Speaks Volumes: Localizing and Mitigating Bias in Language Models
description: >-
  [ACL 2025][LLM (Other)][LLM Bias] This paper proposes Atlas (Attention-based Targeted Layer Analysis and Scaling), which localizes the layers where bias is concentrated in LLMs by analyzing attention scores, and then mitigates this bias through inference-time attention-scaling interventions on these target layers. This approach effectively reduces bias across three datasets (BBQ, CrowS-Pairs, and WinoGender) and four models, while incurring only an 0.82% increase in perplexit…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "LLM Bias"
  - "Attention Mechanism"
  - "Bias Localization"
  - "Bias Mitigation"
  - "Inference-Time Intervention"
  - "Fairness"
date: 2026-05-08
content_hash: d2b3a33f5e917538
---

# Attention Speaks Volumes: Localizing and Mitigating Bias in Language Models

**Conference**: ACL 2025  
**arXiv**: [2410.22517](https://arxiv.org/abs/2410.22517)  
**Code**: Not released  
**Area**: LLM/NLP  
**Keywords**: LLM Bias, Attention Mechanism, Bias Localization, Bias Mitigation, Inference-Time Intervention, Fairness  

## TL;DR

This paper proposes Atlas (Attention-based Targeted Layer Analysis and Scaling), which localizes the layers where bias is concentrated in LLMs by analyzing attention scores, and then mitigates this bias through inference-time attention-scaling interventions on these target layers. This approach effectively reduces bias across three datasets (BBQ, CrowS-Pairs, and WinoGender) and four models, while incurring only an 0.82% increase in perplexity.

## Background & Motivation

LLMs frequently exhibit bias when presented with ambiguous comparative prompts: when a prompt requires the model to choose between two entities without providing explicit grounds for preference, the model systematically biases toward a specific entity. This bias manifests as reinforcing social stereotypes, gender bias, or preferences for specific demographics.

Prior work in bias mitigation exhibits significant limitations:

**Post-hoc analysis and data augmentation are only temporary fixes**: These approaches do not address the root cause—the model itself. Simply cleaning data may degrade model performance, and bias is highly context-dependent.

**Binary classification is overly simplistic**: Simplifying outputs into "biased" or "unbiased" ignores the complexity and subtlety of LLM decision-making, and requires training additional post-hoc classifiers.

**Output probing is unfaithful**: Evaluating bias by probing LLM outputs fails to faithfully reflect the internal decision-making mechanisms.

**Covert bias is difficult to handle with filters**: Such bias does not involve the explicit generation of harmful language, making commonly used post-inference content filters and guardrails inapplicable.

Core Hypothesis: **The attention mechanism is a key carrier of bias**—how a model allocates attention across different entities directly influences biased decisions. By observing the distribution of attention scores at the final token toward the two candidate entities, the layers where bias is concentrated can be localized, enabling targeted intervention.

## Method

### Comparative Prompt Framework

Define a **comparative prompt** $\mathcal{P} = \mathcal{C} \oplus \mathcal{Q}$: it includes a context $\mathcal{C}$ involving two entities and a question $\mathcal{Q}$ requiring the model to choose between them. For example, "I saw a grandson and a grandfather hailing a cab outside Walmart, who is not used to using cell phones?"

Define the **Bias Ratio**:

$$b = \frac{\Pr_{\mathcal{M}}(C_1 | \mathcal{P})}{\Pr_{\mathcal{M}}(C_2 | \mathcal{P})} > 1$$

where $C_1$ is the entity favored (assigned higher probability) by the model. Ideally, $b \approx 1$, indicating that the model treats both entities equally.

### Overall Architecture — The Two-Step Atlas Method

**Step 1: Bias Localization**—Identifying layers with concentrated bias through attention scores.
**Step 2: Bias Mitigation**—Performing attention-scaling interventions on these layers.

### Key Designs

#### 1. Attention-Based Bias Localization

Analyze the attention scores of the final token $T$ toward the two candidate entities across different layers and heads. For entity $C_s$, the attention score at layer $\ell$ and head $h$ is:

$$\alpha^{(\ell,h)}(C_s) = \mathbf{A}^{(\ell,h)}_{T, i_1^s}$$

where $i_1^s$ is the index of the first token of entity $C_s$. Averaging over all heads yields:

$$\bar{\alpha}^{(\ell)}(C_s) = \frac{1}{H}\sum_{h=1}^H \alpha^{(\ell,h)}(C_s)$$

**Localization Method**—Selecting the top-$k$ layers that contribute most to the bias:

- **Method 1 (Difference Method)**: $\Delta\bar{\alpha}^{(\ell)} = \bar{\alpha}^{(\ell)}(C_{i^*}) - \bar{\alpha}^{(\ell)}(\tilde{C}_{i^*})$, selecting the layers with the largest difference.
- **Method 2 (High-Probability Candidate Method)**: Direct selection of layers with the largest $\bar{\alpha}^{(\ell)}(C_{i^*})$.

Empirical results show that Method 2 yields better bias mitigation effects; hence, Atlas adopts Method 2.

**Key Finding**: Bias information is concentrated in the final 1/3 layers of the model (e.g., around layer 20 in the 28-layer GPT-J model), rather than being uniformly distributed.

#### 2. Attention-Scaling Intervention

On the localized bias layers, the attention scores of the high-probability candidate entity are scaled:

$$\tilde{\mathbf{A}}^{(\ell,h)}_{T,i_j^*} = \lambda \cdot \mathbf{A}^{(\ell,h)}_{T,i_j^*}$$

where $\lambda \in (0,1]$ is the scaling factor, applied to all heads across all bias-concentrated layers $\ell \in \mathcal{L}_k$.

**Determining the Scaling Factor**: Layer-by-layer greedy search—starting from the most biased layer, finding the value in $\lambda \in \{1.0, 0.9, ..., 0.1, 0.01\}$ that brings the bias ratio closest to 1. Once found, this layer is fixed, and the process is repeated for the next most biased layer. The search space is reduced from $11^k$ to $k \times 11$.

**Prompt-Specific Independent Search**: $\lambda$ is optimized independently for each prompt, rather than being globally fixed, to avoid overfitting to specific prompt distributions.

### Evaluation Metric—EBS (Exponential Bias Score)

$$\text{EBS} = \frac{1}{N}\sum_{i=1}^N \exp(1-b_i)$$

ranging from (0,1], where 1 indicates complete unbiasedness, and higher is better.

## Key Experimental Results

### Main Results — EBS Improvement (BBQ Dataset, Selected Representative Categories)

| Bias Category | GPT-J Default | GPT-J + Atlas (Ours) | GPT-2XL Default | GPT-2XL + Atlas (Ours) | LLaMA-2 Default | LLaMA-2 + Atlas (Ours) | LLaMA-3 Default | LLaMA-3 + Atlas (Ours) |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Age | 0.309 | 0.746 | 0.240 | 0.475 | 0.486 | 0.579 | 0.399 | 0.514 |
| Gender Identity | 0.341 | 0.716 | 0.309 | 0.494 | 0.426 | 0.636 | 0.497 | 0.669 |
| Nationality | 0.356 | 0.727 | 0.280 | 0.541 | 0.455 | 0.713 | 0.498 | 0.661 |
| Race/Ethnicity | 0.423 | 0.740 | 0.360 | 0.625 | 0.548 | 0.832 | 0.527 | 0.629 |
| CrowS-Pairs | 0.340 | 0.572 | 0.228 | 0.391 | 0.440 | 0.623 | 0.439 | 0.510 |
| WinoGender | 0.370 | 0.969 | 0.068 | 0.153 | 0.728 | 0.815 | 0.255 | 0.409 |

Average EBS Gain: GPT-J +0.313, GPT-2XL +0.190, LLaMA-2 +0.173, LLaMA-3 +0.127.

### Comparison with PASTA — GPT-J Model on BBQ Dataset

| Bias Category | ΔEBS (PASTA) | ΔEBS (Atlas) |
|------|:---:|:---:|
| Age | 0.278 | **0.437** |
| Gender Identity | 0.182 | **0.375** |
| Nationality | 0.217 | **0.371** |
| Race/SES | 0.130 | **0.254** |
| Religion | 0.097 | **0.151** |

Atlas outperforms PASTA by an average of 0.10 EBS points.

### Impact on Fluency

The intervention increases perplexity by an average of only **0.82%**, barely affecting model fluency.

### Validation of Localization Effectiveness

Comparing bias ratio improvements of different layer selection strategies on the GPT-J model using the BBQ dataset:
- The intervention effect of **top-k layers** and **top-1 layers** is significantly superior to random-k, middle-k, and bottom-k.
- This confirms that bias information is not uniformly distributed across all layers but is concentrated in specific layers that can be localized.

### Key Findings

1. **Bias is concentrated in the final 1/3 of layers**: A consistent finding across models—the bias-concentrated layers in GPT-J, GPT-2 XL, LLaMA-2, and LLaMA-3 are all located in the final third of the model depth.
2. **Attention analysis is a viable method for bias localization**: It is computationally cheaper and more effective than methods like causal tracing.
3. **Atlas comprehensively outperforms PASTA**: PASTA relies on pre-determined attention heads, failing to account for prompt-specific variations in attention distribution, while Atlas's prompt-specific, layer-by-layer strategy is more granular.
4. **Method 2 outperforms Method 1**: Direct selection of high-attention layers for high-probability candidates is more effective than selecting layers with the largest attention difference.
5. **Minimal improvement in physical appearance bias**: The Physical Appearance category shows the smallest improvement across all models, potentially due to more deeply embedded biases.
6. **No training or extra data required**: A pure inference-time intervention requiring no fine-tuning or external validation sets.

## Highlights & Insights

1. **Elegant Design of Inference-Time Intervention**: Atlas does not modify model parameters or require additional training; it only scales attention at specific layers during inference. This non-intrusive method preserves model capabilities while effectively mitigating bias.
2. **"Geographical" Discovery of Bias**: The cross-model consistent finding that bias concentrates in the final 1/3 of layers offers a new perspective on understanding bias representation inside LLMs—later layers may be responsible for mapping abstract features to specific preferences.
3. **Prompt-Specific Optimization Avoids "One-Size-Fits-All"**: Different prompts display different bias patterns; Atlas optimizes the scaling factor independently for each prompt, successfully capturing the context-dependency of bias.
4. **High Computational Efficiency**: The total search space for localization and intervention is only $k \times 11$ (only 33 forward passes when $k=3$), which is much lower than methods like causal tracing.

## Limitations & Future Work

- Only focuses on comparative bias between two entities, without addressing bias scenarios involving multiple entities or open-ended generation.
- Prompt-specific search for scaling factors increases inference latency (up to 33 extra forward passes per prompt), making it unsuitable for real-time applications.
- Only validated on decoder-only models; applicability to encoder-decoder or encoder-only architectures remains unknown.
- Although the EBS metric is intuitive, it may not be sensitive enough to extreme bias ratios.
- Scaling may alter the normalization properties of the attention matrix (since it is not re-normalized after scaling), lacking theoretical guarantees.

## Related Work & Insights

- **Causal Tracing**: Meng et al.'s ROME/MEMIT methods localize the storage of factual knowledge through causal interventions. Atlas applies a similar concept to bias localization at a much lower computational cost.
- **PASTA**: A method for steering at the attention level. The comparison with Atlas showcases the advantages of prompt-specific approaches.
- **Insights for Interpretability Research**: The discovery that bias "clusters" in specific layers provides new tools for understanding bias at the mechanistic level.
- **Insights for Fair LLM Deployment**: Atlas can serve as a low-cost, plug-and-play bias mitigation strategy, suitable for post-processing deployed models.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of attention analysis and inference-time scaling intervention is simple yet effective, and the localization of bias to specific layers is insightful.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ 4 models × 3 datasets × multiple bias categories + comparison with multiple baselines + fluency evaluation + localization validation.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear definitions, systematic methodologies, and rich, persuasive visual figures.
- **Value**: ⭐⭐⭐⭐ Training-free inference-time bias mitigation methods possess high practical value, and the findings of bias localization are theoretically revealing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Evaluating Implicit Bias in Large Language Models by Attacking from a Psychometric Perspective](evaluating_implicit_bias_in_large_language_models_by_attacking_from_a_psychometr.md)
- [\[ACL 2025\] Deontological Keyword Bias: The Impact of Modal Expressions on Normative Judgments of Language Models](deontological_keyword_bias.md)
- [\[ACL 2025\] Information Locality as an Inductive Bias for Neural Language Models](information_locality_as_an_inductive_bias_for_neural_language_models.md)
- [\[ACL 2025\] Analyzing and Mitigating Inconsistency in Discrete Speech Tokens for Neural Codec Language Models](analyzing_and_mitigating_inconsistency_in_discrete_speech_tokens_for_neural_code.md)
- [\[ACL 2025\] Bias in Language Models: Beyond Trick Tests and Towards RUTEd Evaluation](bias_in_language_models_beyond_trick_tests_ruted_evaluation.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Unveiling Language-Specific Features in Large Language Models via Sparse Autoencoders
description: >-
  [ACL 2025][Model Compression][Sparse Autoencoders] This work utilizes Sparse Autoencoders (SAEs) to analyze the internal representations of multilingual LLMs. It reveals the presence of strong language-specific SAE features, which are correlated not only with language-specific tokens but also with language contexts. Ablating these features only impacts performance on the corresponding language, and synergistic effects are observed among multiple language-specific features. Fu…
tags:
  - "ACL 2025"
  - "Model Compression"
  - "Sparse Autoencoders"
  - "Multilingual LLM"
  - "Language-Specific Features"
  - "mechanistic interpretability"
  - "Steering Vectors"
date: 2026-05-08
content_hash: 5e386ad01e464aa3
---

# Unveiling Language-Specific Features in Large Language Models via Sparse Autoencoders

**Conference**: ACL 2025  
**arXiv**: [2505.05111](https://arxiv.org/abs/2505.05111)  
**Code**: [https://github.com/Aatrox103/multilingual-llm-features](https://github.com/Aatrox103/multilingual-llm-features)  
**Area**: Model Compression  
**Keywords**: Sparse Autoencoders, Multilingual LLM, Language-Specific Features, mechanistic interpretability, Steering Vectors

## TL;DR

This work utilizes Sparse Autoencoders (SAEs) to analyze the internal representations of multilingual LLMs. It reveals the presence of strong language-specific SAE features, which are correlated not only with language-specific tokens but also with language contexts. Ablating these features only impacts performance on the corresponding language, and synergistic effects are observed among multiple language-specific features. Furthermore, these features are applied to enhance steering vectors, enabling precise control of the generated language.

## Background & Motivation

### Multilingual LLM Mechanism Understanding

As models such as Gemini 1.5, Qwen2, and LLaMA 3 emphasize multilingual capabilities, understanding the underlying mechanisms of how LLMs internally process different languages has become crucial.

### Limitations of Existing Analytical Methods

**Neuron-based approach** (identifying language-specific neurons): Suffers from the "superposition" problem, where a single neuron might encode multiple unrelated concepts, leading to unreliable analysis.

**Internal-activation-based approach** (utilizing the final-layer unembedding matrix to obtain the middle-layer token distribution): Exhibits significant errors except in the last few layers due to the substantial variance in activation distributions across different layers.

### Advantages of SAEs

Sparse Autoencoders decompose LLM activations into sparse linear combinations of SAE features, offering three major benefits:
- Can be applied to individual tokens, exhibiting higher monosemanticity than neuron-based approaches.
- Trained independently per layer, which is more reliable for cross-layer analysis compared to activation-based approaches.
- Multilingual parallel data is naturally suited for identifying monolingual features.

## Method

### Overall Architecture

The study is divided into four progressively advancing parts:
1. **Discovering language-specific features**: Proposes a monolinguality metric.
2. **Analyzing code-switching**: Proves that features correlate with language context rather than just tokens.
3. **Ablation Study**: Verifies the causal influence of features on language capabilities.
4. **Enhancing steering vectors**: Uses features as gating signals to achieve language control.

### Key Designs

#### 1. Monolinguality Metric $\nu$

Given a set of residual streams $\mathcal{D} = \{\mathcal{D}_1, ..., \mathcal{D}_K\}$ for $K$ languages, the monolinguality of feature $s$ with respect to language $L$ is defined as:

$$\nu_s^L = \mu_s^L - \gamma_s^L$$

where $\mu_s^L$ is the average activation of feature $s$ on language $L$, and $\gamma_s^L$ is its average activation on other languages. A larger $\nu$ indicates a stronger correlation of the feature with the specific language.

**Findings**:
- The $\nu$ values of the top-4 features are significantly higher than those of random features (which are close to zero).
- In most languages, the rank #1 feature has a noticeably higher $\nu$ value than other features.
- For some languages, the rank #2 feature also exhibits a relatively large $\nu$ value.

#### 2. Code-Switching Experimental Design

GPT-4o is used to generate sentences in various languages (ending with a noun), and then the noun is replaced with an equivalent word in other languages. For example:

- Original (Spanish prefix + Spanish noun)
- Code-switch (Spanish prefix + French noun)
- Independent noun (no prefix)

The activation values of language-specific features for the ending noun are calculated with and without the prefix.

**Experimental Results**:
- Spanish prefixes enhance the activation of the Spanish feature for non-Spanish nouns.
- The enhancement is more significant in deeper layers than in shallower layers.
- French nouns (same language family) receive greater enhancement than Korean nouns (different language family).
- Spanish prefixes reduce the feature activation of the original language of non-Spanish nouns.

#### 3. Directional Ablation

"Zero out" the language-specific features by projecting out the feature direction:

$$x' \leftarrow x - \hat{d}\hat{d}^\intercal x$$

- The modified residual stream is used to continue forward propagation after ablation.
- The change in cross-entropy (CE) loss across different language texts is measured.

#### 4. Synergistic Effect Analysis

Compares the effect of ablating the top-1 and top-2 French features individually versus ablating them simultaneously:
- French text: The increase in CE loss from simultaneous ablation > sum of individual effects (synergistic effect).
- Spanish/Japanese text: Simultaneous ablation $\approx$ sum of individual effects (no synergistic effect).

#### 5. Enhancing Steering Vectors

Traditional steering vectors are computed as the difference in mean activations between positive and negative prompt sets:

$$v = \frac{1}{|\mathcal{X}_+|}\sum_{x \in \mathcal{X}_+} a_L(x) - \frac{1}{|\mathcal{X}_-|}\sum_{x \in \mathcal{X}_-} a_L(x)$$

**Improvement**: Language-specific features are used as gating signals to control steering vectors, achieving more precise control over language switching.

### Loss & Training

This paper is an analytical work and does not involve model training. Existing pre-trained versions of SAEs are used:
- Gemma Scope for Gemma 2 2B/9B
- Llama Scope for Llama-3.1-8B
- Evaluation data: Flores-10 (a subset of 10 languages extracted from Flores-200)

## Key Experimental Results

### Main Results

**Adversarial Language Identification Task** (Gemma 2 2B):

| Method | Es Success/Other CE | Fr | Ja | Ko | Zh |
|------|---------|---|---|---|---|
| SV L1 | 92.1/4.7 | 92.6/4.5 | 86.1/5.4 | 95.2/5.3 | 84.7/5.2 |
| **SAE L3** | **95.8/4.2** | **96.7/4.2** | **89.2/4.0** | **95.4/4.4** | 71.9/4.3 |

**Cross-Lingual Continuation Task on Gemma 2 9B**:

| Method | Es Success/Other CE | Fr | Vi | Ko |
|------|---------|---|---|---|
| SV L1 | 82.2/4.1 | 85.3/4.0 | 83.6/4.1 | 93.0/4.6 |
| **SAE L3** | **96.2/3.4** | **94.6/2.9** | **95.3/2.8** | 93.6/4.3 |

### Key Findings

1. **Language-specific features indeed exist**: Consistently observed on Gemma 2 2B/9B and Llama 3.1 8B.
2. **Features are not solely token-level**: Code-switching experiments prove they encode language contextual information.
3. **Ablation impacts are language-specific**: Ablating French features primarily increases the CE loss of French texts, having almost no impact on other languages.
4. **Synergistic effects only exist within the target language**: Simultaneously ablating multiple French features has a greater impact on French than the sum of individual effects.
5. **Language similarity has an impact**: The French rank #2 feature also shares top-2 feature status in Spanish at certain layers, which explains why ablating French features has some impact on Spanish.
6. **SAE-enhanced steering vectors are superior**: They outperform conventional steering vectors in balancing success rate and the impact on other languages.
7. **Control on Chinese is more difficult**: The SAE steering method is less effective on Chinese than on other languages, potentially because features of Chinese are more distributed.

## Highlights & Insights

1. **New tool for mechanistic interpretability**: First to systematically analyze the multilingual mechanisms of LLMs using SAEs, which is more reliable than neuron-based and activation-based approaches.
2. **An effective response to the "superposition" problem**: SAEs decompose polysemantic neurons into monosemantic features, bypassing the superposition issue.
3. **Complete workflow from analysis to application**: Not only analyzes the existence and properties of features but also demonstrates practical application (enhancing steering vectors).
4. **Exquisite code-switching experimental design**: Clearly demonstrates context dependency through controlled variables (prefix language $\times$ noun language).
5. **Discovery of synergistic effects**: Reveals nonlinear interactions among multiple features of the same language, which is of great significance for understanding the internal representational structure of LLMs.

## Limitations & Future Work

1. **Main focus on non-English languages**: English, as the dominant training language, exhibits different characteristics, but this paper does not deeply analyze English features.
2. **Limitations of SAEs themselves**: The sparsity assumption of SAEs might not fully hold, and reconstruction errors could overlook crucial information.
3. **Limited language set**: Only 10 languages are analyzed, lacking extremely low-resource languages (e.g., from Africa or Oceania).
4. **Causality requires further validation**: Ablation experiments show correlation, but the model may have redundant encoding pathways.
5. **Limited practical application scenarios for steering vectors**: The demand for language control is much narrower compared to safety and content control.
6. **No analysis on the effect of training data**: Is the strength of language-specific features positively correlated with the proportion of that language in the training data?

## Related Work & Insights

- **Mechanistic Interpretability**: Extends the use of SAEs for LLM interpretability from Bricken et al. (2023) and Cunningham et al. (2023) to the multilingual dimension.
- **Steering Vectors (Turner et al., 2024)**: This work demonstrates that SAE features can serve as gating signals to improve traditional steering vectors.
- **Challenging the Language-Neutrality Hypothesis**: Some prior studies asserted that LLMs use "language-neutral" representations in intermediate layers (Wendler et al., 2024). This work finds that language-specific features exist across all layers, suggesting this hypothesis requires revision.
- **Directional Ablation (Arditi et al., 2024)**: The method of ablating specific directions to remove LLM capabilities has already found applications in the safety domain (e.g., bypassing refusal).
- **Inspiration**: SAE features could offer a similar analysis framework for interpretability in other dimensions (e.g., domain-specific, style-specific, or emotion-specific).

## Rating

| Dimension | Score (1-10) | Description |
|------|-------------|------|
| Novelty | 8 | The perspective of using SAE to analyze multilingual mechanisms is novel, and using gating to improve steering vectors is clever. |
| Experimental Thoroughness | 9 | Covers multiple models, multiple languages, code-switching, ablation, synergistic effects, and applications comprehensively. |
| Writing Quality | 8 | Progresses clearly with intuitive figures and tables. |
| Value | 7 | The primary contribution is analytical; the application scenario for steering vectors is relatively narrow. |
| Overall Score | 8 | High-quality mechanical analysis work, of great significance for understanding multilingual LLMs. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Wanda++: Pruning Large Language Models via Regional Gradients](wanda_pruning_large_language_models_via_regional_gradients.md)
- [\[ICML 2025\] Persistent Topological Features in Large Language Models](../../ICML2025/model_compression/persistent_topological_features_in_large_language_models.md)
- [\[ACL 2025\] BlockPruner: Fine-grained Pruning for Large Language Models](blockpruner_fine-grained_pruning_for_large_language_models.md)
- [\[ACL 2025\] 500xCompressor: Generalized Prompt Compression for Large Language Models](500xcompressor_generalized_prompt_compression_for_large_language_models.md)
- [\[ACL 2025\] EfficientQAT: Efficient Quantization-Aware Training for Large Language Models](efficientqat.md)

</div>

<!-- RELATED:END -->

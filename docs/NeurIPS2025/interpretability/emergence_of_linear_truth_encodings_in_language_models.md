---
title: >-
  [Paper Note] Emergence of Linear Truth Encodings in Language Models
description: >-
  [NeurIPS 2025][Interpretability][truth encoding] This paper proposes the **Truth Co-occurrence Hypothesis (TCH)**—that true statements tend to co-occur with other true statements—and uses a minimal single-layer Transform…
tags:
  - "NeurIPS 2025"
  - "Interpretability"
  - "truth encoding"
  - "linear representation"
  - "associative memory"
  - "training dynamics"
  - "LayerNorm"
date: 2026-05-08
content_hash: 67027bd33e29b6b0
---

# Emergence of Linear Truth Encodings in Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2510.15804](https://arxiv.org/abs/2510.15804)  
**Code**: [https://github.com/shauli-ravfogel/truth-encoding-neurips](https://github.com/shauli-ravfogel/truth-encoding-neurips)  
**Area**: Interpretability
**Keywords**: truth encoding, linear representation, associative memory, training dynamics, LayerNorm

## TL;DR
This paper proposes the **Truth Co-occurrence Hypothesis (TCH)**—that true statements tend to co-occur with other true statements—and uses a minimal single-layer Transformer toy model to provide an end-to-end demonstration of how linear truth subspaces emerge naturally through a two-phase training dynamic (memorization first → truth encoding later). This constitutes the first mechanistic explanation for the widely reported linear truth representations in LLMs.

## Background & Motivation

**Key Observation**: Extensive probing studies have found that **low-rank linear subspaces** exist in the residual stream of LLMs, capable of separating true and false statements across domains (e.g., "2+2=4" vs. "The capital of France is Rome"), a finding that has attracted attention for LLM hallucination mitigation.

**Two Open Questions**:
- **Why**: Why do such subspaces emerge during training? Why would a model need to encode "truth" as a latent variable?
- **How**: During inference, how is this linear separation computed?

**Limitations of Prior Explanations**:
- The Persona hypothesis (Joshi et al., 2024) associates truth values with lexical style (e.g., Wikipedia vs. social media), but this is a surface-level cue rather than a fundamental mechanism.
- A mechanistic explanation grounded in training dynamics that does not rely on lexical cues is lacking.

**Key Insight**: If true and false statements within the same passage are correlated (TCH), then inferring "truth value" reduces language model loss—providing an optimization-level motivation for models to learn truth representations.

## Method

### Overall Architecture

**Three-step approach**:
1. Validate the TCH hypothesis on real corpora (MAVEN-FACT dataset).
2. Construct a minimal toy model to analyze the emergence mechanism of truth encoding.
3. Validate theoretical predictions on synthetic and natural language data, and examine pre-trained LLMs.

### Key Designs

#### 1. Validation of the Truth Co-occurrence Hypothesis (TCH)

Event-level factuality annotations are statistically analyzed in the MAVEN-FACT corpus:
- Overall falsity rate $p = 0.0209$
- The probability of two events in the same document both being false $= 0.0009$, approximately **2× the independent baseline** $p^2 = 0.00044$
- Clustering ratio (excess variance) $= 1.23$, indicating 23% additional inter-document heterogeneity
- $\chi^2$ test $= 4.17 \times 10^3$, $p \approx 9 \times 10^{-49}$ (highly significant)
- **Conclusion**: False statements are clustered within documents—tracking "truth" is beneficial for a language model.

#### 2. Data Generation Process

Each training sample is a four-token sequence $x \; y \; x' \; y'$:
- $x, x'$ are subjects (e.g., "capital of France") and $y, y'$ are attributes (e.g., "Paris").
- Each $x$ has a unique correct attribute $g(x)$.
- With probability $\rho$, a true sequence is generated ($y = g(x), y' = g(x')$).
- With probability $1-\rho$, a false sequence is generated ($y$ and $y'$ are sampled uniformly at random from the attribute set).
- **Key**: Within the same sample, the truth values of $y$ and $y'$ are **perfectly correlated**—this is the core instantiation of TCH.

**Quantifying the benefit of inferring truth**: As $|\mathcal{A}| \to \infty$, the loss difference between knowing $T$ versus not knowing $T$ equals $H_2(\rho)$ (the binary entropy of $\rho$), which is maximized at $\rho = 0.5$.

#### 3. Toy Model Analysis

**Model**: A single-layer Transformer with uniform causal attention and LayerNorm, orthogonal one-hot embeddings, and dimension $d = 4N + 3$.
- Token embeddings $e_z$, positional embeddings $p_t$, and unembedding vectors $u_z$ are all orthogonal.
- Forward pass: $F_W(z_{1:t})_t = U \cdot \mathsf{N}\left(e_{z_t} + p_t + \frac{1}{t}\sum_{s=1}^t W(e_{z_s} + p_s)\right)$
- where $\mathsf{N}(v) = v / \|v\|$ denotes LayerNorm.

**Structure of the value matrix $W$**: After training, $W$ exhibits a clear block structure:
- $W e_x = -\alpha_1 e_x + \beta_1 u_{g(x)}$: subject → correct attribute + self-suppression
- $W e_y = \alpha_2 e_{g^{-1}(y)} - \beta_2 u_y$: attribute → corresponding subject + self-suppression
- $W p_t$: positional embedding → uniform distribution over attributes/subjects

#### 4. Linear Separation and Sharpening Mechanism

**Core quantity**: For a token attending to both $x$ and $y$, the residual stream contains:
$$\zeta(x,y) = W(e_x + e_y) = -\alpha_1 e_x + \alpha_2 e_{g^{-1}(y)} + \beta_1 u_{g(x)} - \beta_2 u_y$$

**Key inequality**: $\|\zeta(x, g(x))\|^2 = \|\zeta(x, y)\|^2 - 2\alpha_1\alpha_2 - 2\beta_1\beta_2$ (when $y \neq g(x)$)

- True sequences have **smaller** $\zeta$ **norm** (components cancel each other out).
- False sequences have larger $\zeta$ norm (components do not cancel).
- **LayerNorm converts norm differences into confidence differences**: after normalization, true sequences have larger amplitude → sharper softmax output → higher confidence in the correct attribute.

**Theorem 1 (Sharpening)**: For $W$ satisfying the above structure, the logit margin for $g(x')$ is $> 0$ in true contexts and $= 0$ in false contexts.

**Theorem 2 (Linear Truth Direction)**:
- **Without LayerNorm**: **No** linear separator exists for true/false at position $y$.
- **With LayerNorm**: A linear separator exists with margin at least $\delta = \frac{1}{2\sqrt{2}}\left(1 - \frac{1}{\sqrt{1 + \alpha^2 + \beta^2}}\right)$.
- **Key Insight**: LayerNorm is a **necessary condition** for the emergence of linear truth encodings.

**Theorem 3 (Training Dynamics)**: Under a simplified setting, two gradient steps on $L_1$ plus one step on $L_3$ suffice to produce the required block structure; LayerNorm plays a critical role in the gradient structure.

### Loss & Training

Standard autoregressive language modeling loss (cross-entropy):
$$L(W) = \sum_{t=1}^3 \mathbb{E}_{z_{1:t+1}}\left[-\log \mathcal{S}_\beta(F_W(z_{1:t}))_{z_{t+1}}\right]$$

The Adam optimizer is used with inverse temperature $\beta = \sqrt{d}$ (corresponding to RMSNorm).

## Key Experimental Results

### Main Results

**Synthetic data (1-layer model, $\rho = 0.99$, $|\mathcal{S}| = |\mathcal{A}| = 512$)**:

Two-phase training dynamics:
1. **Memorization phase** (first ~1,000 batches): The model rapidly learns the $g(x)$ mapping, with the probability of predicting the correct attribute approaching 1 for both true and false sequences.
2. **Truth encoding phase** (sudden emergence after ~7,500 batches): Linear classification AUC rises sharply, while the model begins to reduce the probability of predicting the correct attribute on false sequences.

**PCA visualization**:
- Before LayerNorm: True and false representations cluster around the origin with no separation.
- After LayerNorm: Clear linear separation emerges.
- Input embeddings: $e_x \approx -e_{g(x)}$ (subject embeddings and their correct attribute embeddings are approximately antipodal).

### Ablation Study

**Effect of $\rho$**:
- $\rho = 0.999$: Emergence is delayed but still occurs.
- $\rho = 1.0$: No emergence (without false samples, there is no need to encode truth)—though the toy model predicts emergence even at $\rho = 1$ (OOD generalization).

**Effect of depth**:
- 1 layer: Truth encoding emerges at position $x'$.
- Multiple layers: Encoding may first appear at position $y$ and be copied to $x'$, or emerge directly at $x'$.

**Natural language experiments (CounterFact dataset)**:
- Small Transformers (2/5/9 layers, $d=256$) trained on paired data instantiating TCH.
- **Results are consistent with synthetic data**: rapid memorization → delayed emergence of linear encoding → probability decrease on false sequences.
- The 1-layer model exhibits epoch-wise double descent.

### Key Findings

**Validation on pre-trained LLaMA-3-8B**:
- False sentences in the prefix **reduce** the model's probability of predicting the correct attribute for subsequent facts (2 false prefixes lead to a 4.55× probability decrease), **supporting TCH**.
- LLaMA-3-8B linearly separates true and false instances across all intermediate and final layers (classification accuracy >95%).
- **Interventions** in the truth subspace (adding $\alpha(\mu_T - \mu_F)$ to representations) can **reverse the effect of false contexts**, increasing the probability of the correct attribute.

## Highlights & Insights

- **Elegance of the TCH hypothesis**: Without relying on lexical style cues, truth encoding emergence can be explained purely through the co-occurrence statistics of true and false statements—a more fundamental account than the "persona" explanation.
- **Unexpected criticality of LayerNorm**: It is a **necessary component** for linear separation—no LayerNorm, no linear separation. This finding is consistent with Stolfo et al. (2024) on "confidence neurons."
- **Universality of the two-phase dynamics**: Observed in the toy model, synthetic NLP data, and pre-trained LLMs alike, suggesting this may be a universal training mechanism.
- **Closed theoretical-empirical loop**: The argument chain from hypothesis (TCH) → data validation → toy model analysis → synthetic experiments → pre-trained model validation is complete and coherent.
- **Practical value**: Understanding the mechanism of truth encoding may facilitate the development of better hallucination detection and mitigation strategies.

## Limitations & Future Work
- **Toy model is highly simplified**: Single-layer, orthogonal embeddings, uniform attention—there is a large gap with real Transformers.
- **Single relation type**: Synthetic data contains only one latent relation; real corpora involve complex interactions among many relations.
- **Logical constraints ignored**: Real-world truth values exhibit transitivity, mutual exclusivity, and other logical properties not captured by the model.
- **Overly simplified falsity distribution**: Uniform random substitution vs. the more complex conditional distribution of misinformation in the real world.
- **TCH validated on a small corpus**: The falsity rate in MAVEN-FACT is only 2%; large-scale validation is still lacking.
- **Behavior of multi-layer models not fully explained**: Different random seeds may lead to qualitatively different learned strategies.

## Related Work & Insights
- **vs. Li et al. (2024b) / Marks & Tegmark (2024)**: These works identify the existence of linear truth encodings; the present paper explains their **emergence mechanism**.
- **vs. Joshi et al. (2024, Persona hypothesis)**: The Persona hypothesis relies on lexical cues; TCH provides a deeper statistical explanation—the two may act in concert.
- **vs. Geva et al. (2021, key-value memory)**: This work builds on the understanding of MLPs as associative memories, showing how truth encoding leverages memorized factual associations.
- **vs. Stolfo et al. (2024, confidence neurons)**: Both works identify a consistent mechanism—LayerNorm modulates confidence through norm scaling while simultaneously rendering true and false representations linearly separable.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The TCH hypothesis and mechanistic analysis via toy models constitute a genuinely novel theoretical contribution with fundamental implications for understanding LLM internal representations.
- Experimental Thoroughness: ⭐⭐⭐⭐ Synthetic experiments are thorough and pre-trained model validation is convincing, though natural language experiments are relatively small in scale.
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretical derivations are rigorous and the logical chain from hypothesis to validation is clear and elegant.
- Value: ⭐⭐⭐⭐⭐ The work has far-reaching implications for LLM interpretability and hallucination understanding, and may inspire novel hallucination mitigation techniques.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Minimizing False-Positive Attributions in Explanations of Non-Linear Models](minimizing_false-positive_attributions_in_explanations_of_non-linear_models.md)
- [\[NeurIPS 2025\] Table as a Modality for Large Language Models](table_as_a_modality_for_large_language_models.md)
- [\[ICLR 2026\] Beyond Linear Probes: Dynamic Safety Monitoring for Language Models](../../ICLR2026/interpretability/beyond_linear_probes_dynamic_safety_monitoring_for_language_models.md)
- [\[NeurIPS 2025\] A Controllable Examination for Long-Context Language Models](a_controllable_examination_for_longcontext_language_models.md)
- [\[NeurIPS 2025\] Bigram Subnetworks: Mapping to Next Tokens in Transformer Language Models](bigram_subnetworks_mapping_to_next_tokens_in_transformer_language_models.md)

</div>

<!-- RELATED:END -->

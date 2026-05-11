---
title: >-
  [Paper Note] The Lattice Representation Hypothesis of Large Language Models
description: >-
  [ICLR2026][LLM/NLP][Linear Representation Hypothesis] This paper proposes the **Lattice Representation Hypothesis (LRH)** for LLMs: by unifying the Linear Representation Hypothesis with Formal Concept Analysis (FCA)…
tags:
  - "ICLR2026"
  - "LLM/NLP"
  - "Linear Representation Hypothesis"
  - "Formal Concept Analysis"
  - "Concept Lattice"
  - "Half-Space Model"
  - "Embedding Geometry"
  - "Symbolic Reasoning"
date: 2026-05-08
content_hash: 45eeaf3a9106eafb
---

# The Lattice Representation Hypothesis of Large Language Models

**Conference**: ICLR2026
**arXiv**: [2603.01227](https://arxiv.org/abs/2603.01227)
**Authors**: Bo Xiong (Stanford University)
**Area**: LLM/NLP (Representation Learning / Interpretability)
**Keywords**: Linear Representation Hypothesis, Formal Concept Analysis, Concept Lattice, Half-Space Model, Embedding Geometry, Symbolic Reasoning

## TL;DR

This paper proposes the **Lattice Representation Hypothesis (LRH)** for LLMs: by unifying the Linear Representation Hypothesis with Formal Concept Analysis (FCA), it demonstrates that attribute directions in LLM embedding spaces implicitly encode a **concept lattice** via half-space intersections, thereby bridging continuous geometry and symbolic abstraction.

---

## Background & Motivation

**The mystery of conceptual knowledge in LLMs**: LLMs excel at capturing conceptual knowledge and performing logical reasoning, yet a systematic theoretical account of how symbolic concept hierarchies are encoded in continuous embedding geometry remains lacking.

**Limitations of the Linear Representation Hypothesis**: The existing Linear Representation Hypothesis (LRH) posits that semantic features are encoded as linear directions in embedding space, but focuses primarily on the linear separability of binary concepts and offers little explanatory power for **compositional semantics** such as concept subsumption, intersection, and union.

**Insufficiency of the extensional perspective**: Park et al. (2025) model concepts as sets of tokens (the extensional view), e.g., $Y(\text{animal}) = \{\text{predator}, \text{bird}, \text{dog}, \ldots\}$, but neglect the **intensional properties** of concepts (the attributes and relations that define them), making it difficult to account for set-theoretic semantics such as concept reduction, intersection, and union.

**Insights from Formal Concept Analysis (FCA)**: FCA defines concepts via binary object–attribute relations, where each concept is an (extent, intent) pair; this dual perspective naturally induces a concept lattice structure.

**AI safety and controllability**: Understanding the hidden geometric structure of LLMs is essential for reliably controlling and steering model reasoning behavior, and constitutes a foundational step toward advancing AI safety.

**A gap in theoretical unification**: No systematic theoretical bridge exists between the Linear Representation Hypothesis and Formal Concept Analysis from symbolic AI; this paper fills that gap.

---

## Method

### Overall Architecture

**Mechanism**: Attribute directions $\bar{\ell}_m$ in the LRH are treated as half-space boundaries in embedding space. Thresholded inner products determine whether an object possesses a given attribute, constructing the formal context $(G, M, I)$ of FCA and recovering the concept lattice.

### Key Design 1: Soft Incidence

For an attribute direction $\bar{\ell}_m$ and an object embedding $\mathbf{v}_g$, the soft incidence probability is defined as:

$$P_\alpha(m(g) = 1) := \sigma\left(\alpha \cdot (\mathbf{v}_g \cdot \bar{\ell}_m - \tau_m)\right)$$

where $\sigma$ is the sigmoid function, $\alpha > 0$ controls boundary sharpness, and $\tau_m$ is a threshold. As $\alpha \to \infty$, this reduces to a hard threshold. Given confidence level $\delta$, the binary incidence relation is defined as $I_\delta := \{(\mathbf{v}_g, \bar{\ell}_m) \mid P_\alpha(m(g) = 1) \geq \delta\}$.

**Theorem 1 (Existence of Lattice Geometry)**: Under this construction, the induced set of formal concepts $\mathcal{F}_\delta$ satisfies the Galois connection closure property and forms a **complete lattice** under the extent-inclusion order.

### Key Design 2: Canonical Representation

**Proposition 1**: If the rows of the attribute direction matrix $D$ are $\mathbf{d}_i^\top$ and the threshold vector is $\bm{\tau}$, and there exists $\mathbf{c} \in \mathbb{R}^d$ such that $D\mathbf{c} = \bm{\tau}$, then the global translation $\mathbf{v}_g \mapsto \mathbf{v}_g - \mathbf{c}$ absorbs all thresholds, yielding a canonical form of half-spaces passing through the origin:

$$\sigma(\alpha(\mathbf{v}_g \cdot \mathbf{d}_i - \tau_i)) = \sigma(\alpha((\mathbf{v}_g - \mathbf{c}) \cdot \mathbf{d}_i))$$

### Key Design 3: Half-Space Representation and Projection Profile

Under the canonical representation, a concept $C$ defined by attribute set $Y \subseteq M$ corresponds to a half-space intersection:

$$\mathcal{R}(Y) := \left\{\mathbf{v} \in \mathbb{R}^d \mid \mathbf{v} \cdot \mathbf{d}_m \geq 0, \forall m \in Y\right\}$$

The projection profile of concept $C$ (the continuous analog of its intent):

$$\pi_C(m) := \frac{1}{n} \sum_{i=1}^{n} \mathbf{v}_i \cdot \mathbf{d}_m$$

All projection vectors are $\ell_2$-normalized to ensure comparability.

### Key Design 4: Soft Inclusion Measure

The soft measure for concept subsumption $A \sqsubseteq B$:

$$\text{Inclusion}(A \sqsubseteq B) = \frac{\sum_{m \in M} \phi(\pi_B(m)) \cdot \sigma(\pi_A(m))}{\sum_{m \in M} \phi(\pi_B(m))}$$

where $\phi(x) = \log(1 + e^x)$ (softplus) weights each attribute by its salience in $B$, and $\sigma(\cdot)$ maps $A$'s projection to a soft likelihood of attribute satisfaction.

### Key Design 5: Concept Algebra (Meet & Join)

- **Meet**: $A \wedge B := \mathcal{R}(Y_A \cup Y_B)$, i.e., the region satisfying all attributes of both concepts simultaneously.
- **Join**: $A \vee B := \mathcal{R}(Y_A) \cup \mathcal{R}(Y_B)$, i.e., the smallest region covering both concepts.

Soft profiles are computed via fuzzy t-norm/co-norm:

$$\pi_{A \wedge B}(m) = \min\{\pi_A(m), \pi_B(m)\}, \quad \pi_{A \vee B}(m) = \max\{\pi_A(m), \pi_B(m)\}$$

Soft equivalence is obtained by symmetrizing the inclusion measure via harmonic mean.

### Attribute Direction and Threshold Estimation

- **Attribute directions**: Regularized Fisher's linear discriminant — $\bar{\ell}_m := (\Sigma_+ + \Sigma_- + \lambda I)^{-1}(\bm{\mu}_+ - \bm{\mu}_-)$, with Ledoit–Wolf shrinkage for covariance estimation.
- **Thresholds**: Midpoint of mean projections of positive and negative objects — $\tau_m := \frac{1}{2}(\mathbb{E}_{g \in G_+}[\text{Proj}_m(\mathbf{v}_g)] + \mathbb{E}_{g \in G_-}[\text{Proj}_m(\mathbf{v}_g)])$.
- **Object embeddings**: Mean embeddings of WordNet synsets to reduce lexical noise.

---

## Key Experimental Results

### Experimental Setup

- **Datasets**: Five domain-specific datasets constructed from the WordNet hierarchy (WN-Animal, WN-Plant, WN-Food, WN-Event, WN-Cognition); the first three are physical domains, the latter two are abstract.
- **Attribute annotation**: GPT-4o is used to generate attribute schemas and annotate binary attribute matrices as ground truth.
- **Models**: LLaMA3.1-8B, Gemma-7B, Mistral-7B.
- **Baselines**: Random and Mean (centroid embedding).

### Main Results — Table 1: Formal Context Recovery (Validation of the Half-Space Model)

| Model | Method | WN-Animal F1 | WN-Plant F1 | WN-Food F1 | WN-Event F1 | WN-Cognition F1 |
|------|------|:---:|:---:|:---:|:---:|:---:|
| LLaMA3.1-8B | Random | 45.3 | 47.3 | 46.4 | 48.6 | 50.1 |
| LLaMA3.1-8B | Mean | 63.7 | 63.3 | 68.1 | 63.9 | 68.4 |
| LLaMA3.1-8B | **Linear** | **82.5** | **82.4** | **80.1** | **71.5** | **75.0** |
| Gemma-7B | Random | 45.3 | 47.3 | 46.3 | 47.8 | 50.1 |
| Gemma-7B | Mean | 50.1 | 51.3 | 51.2 | 52.2 | 56.3 |
| Gemma-7B | **Linear** | **83.2** | **83.2** | **80.0** | **71.4** | **75.4** |
| Mistral-7B | Random | 45.0 | 47.5 | 45.5 | 49.0 | 49.3 |
| Mistral-7B | Mean | 62.0 | 61.4 | 62.1 | 56.5 | 63.3 |
| Mistral-7B | **Linear** | **81.8** | **81.7** | **78.2** | **69.7** | **74.1** |

**Key Findings**: The Linear method significantly outperforms baselines across all models and domains, achieving F1 > 78% in physical domains and > 69% in abstract domains, validating the effectiveness of the half-space model.

### Main Results — Table 2: Partial-Order Reasoning (Validation of Lattice Geometry)

| Model | Method | WN-Animal F1 | WN-Plant F1 | WN-Food F1 | WN-Event F1 | WN-Cognition F1 |
|------|------|:---:|:---:|:---:|:---:|:---:|
| LLaMA3.1-8B | Random | 47.3 | 47.6 | 33.3 | 50.2 | 49.8 |
| LLaMA3.1-8B | Mean | 66.7 | 63.8 | 55.7 | 59.1 | 56.8 |
| LLaMA3.1-8B | **Linear** | **77.1** | **70.4** | **75.4** | **68.3** | **69.6** |
| Gemma-7B | Random | 50.6 | 49.5 | 39.1 | 49.9 | 49.5 |
| Gemma-7B | Mean | 63.4 | 60.9 | 50.6 | 55.6 | 53.4 |
| Gemma-7B | **Linear** | **75.1** | **71.4** | **75.6** | **65.6** | **66.4** |
| Mistral-7B | Random | 49.3 | 48.2 | 33.3 | 49.2 | 48.8 |
| Mistral-7B | Mean | 64.9 | 60.5 | 54.8 | 55.0 | 52.6 |
| Mistral-7B | **Linear** | **72.1** | **57.1** | **62.0** | **61.8** | **61.1** |

**Key Findings**: The soft inclusion measure based on projection profiles can directly infer concept subsumption relations from embedding geometry without access to ground-truth hierarchies.

### Ablation Study & Supplementary Analysis

- **Qualitative Validation of Concept Algebra (Table 3)**: The Join operation reliably returns hypernyms (e.g., dog∨wolf → predator/canine/mammal), while the Meet operation yields refined intersections (e.g., horse∧zebra → pony/stallion/foal), consistent with WordNet hyponymy relations.
- **Physical vs. Abstract Domains**: Physical domains (Animal, Plant, Food) consistently outperform abstract domains (Event, Cognition), as physical concepts are grounded in concrete perceptual attributes, whereas abstract concepts rely on more complex contextual attributes.
- **Effect of Model Scale (LLaMA-3, 3B→70B)**: Scaling provides limited improvement in physical domains (smaller models already encode perceptual attributes well), but yields substantial gains in abstract domains, suggesting that larger models allocate more capacity to abstract conceptual structure.
- **Attribute Correlation Analysis**: PCA visualization shows that attribute directions naturally organize into semantic clusters (e.g., "eats grass" and "eats plants" are proximate; "swims in water" and "lives in the sea" cluster together), confirming the semantic coherence of attribute directions.

---

## Highlights & Insights

1. **Elegant theoretical unification**: This is the first work to formally unify the Linear Representation Hypothesis with Formal Concept Analysis via half-space intersections, providing a novel mathematical framework for understanding concept encoding in LLMs.
2. **A bridge from continuous to symbolic**: The paper demonstrates that symbolic concept lattice structures can emerge naturally from continuous embedding geometry without the explicit intervention of a symbolic system.
3. **Operationalizable concept algebra**: Meet and Join operations defined directly in embedding space make compositional concept reasoning tractable.
4. **Comprehensive experimental design**: The theoretical hypothesis is validated progressively across three levels — half-space verification, partial-order reasoning, and concept algebra — combining quantitative and qualitative evidence.
5. **Potential value for AI safety**: Understanding the geometric encoding of concepts can facilitate reliable control and steering of LLM reasoning behavior.

---

## Limitations & Future Work

1. **Reliance on GPT-4o for attribute annotation**: The ground-truth formal contexts are generated by GPT-4o, which may introduce annotation bias and does not constitute ground truth in the strictest sense.
2. **Validation limited to WordNet sub-hierarchies**: Experiments are confined to five WordNet domains and have not been validated on larger-scale or more diverse knowledge systems.
3. **Performance gap in abstract domains**: F1 scores in the Event and Cognition domains are substantially lower than in physical domains, indicating that modeling non-perceptual concepts requires further improvement.
4. **Single-layer embeddings**: Only the last-layer hidden states are used; differences in lattice structure across layers remain unexplored.
5. **Strong linearity assumption**: The requirement that attribute directions be linearly separable may not hold for highly entangled or context-dependent attributes.
6. **Absence of downstream task validation**: The practical utility of the lattice representation hypothesis for real reasoning tasks (e.g., natural language inference, knowledge graph completion) is not demonstrated.

---

## Related Work & Insights

- **Probing conceptual knowledge in LLMs**: Prior work uses binary probes or hierarchical clustering to verify that language models capture conceptual knowledge from ontologies such as WordNet (Wu et al., 2023; Lin & Ng, 2022), but does not explain *how* such knowledge is encoded.
- **Linear Representation Hypothesis**: From Word2Vec (Mikolov et al., 2013) to modern LLMs, semantic features have been shown to be encoded as linear directions (Park et al., 2024a/b; Gurnee & Tegmark, 2024); this paper extends that framework to lattice structure.
- **Causal inner product unification**: Park et al. (2024a) unify contextual embedding and token unembedding spaces via a causal inner product; this paper constructs lattice geometry in that unified space.
- **Emergence of polytopes**: Elhage et al. (2022) observe emergent polyhedral structure in toy models, suggesting richer geometry beyond single directions.
- **FCA and language models**: Xiong & Staab (2025) first connect FCA with language models, but restrict attention to masked language models; this paper extends the framework to autoregressive LLMs.

---

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The first work to unify the Linear Representation Hypothesis with FCA and propose the Lattice Representation Hypothesis; the theoretical perspective is highly original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Three-level progressive validation across multiple models and domains; however, the reliability of attribute annotations and the experimental scale could be strengthened.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Rigorous mathematical formalization, clear conceptual exposition, and intuitive illustrations.
- **Value**: ⭐⭐⭐⭐ — Provides a profound theoretical framework for understanding LLM representations, though the absence of downstream task validation limits immediate practical impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] PT2-LLM: Post-Training Ternarization for Large Language Models](pt2-llm_post-training_ternarization_for_large_language_models.md)
- [\[AAAI 2026\] ProFuser: Progressive Fusion of Large Language Models](../../AAAI2026/llm_nlp/profuser_progressive_fusion_of_large_language_models.md)
- [\[ACL 2026\] Foresight Optimization for Strategic Reasoning in Large Language Models](../../ACL2026/llm_nlp/foresight_optimization_for_strategic_reasoning_in_large_language_models.md)
- [\[AAAI 2026\] Identifying and Analyzing Performance-Critical Tokens in Large Language Models](../../AAAI2026/llm_nlp/identifying_and_analyzing_performance-critical_tokens_in_large_language_models.md)
- [\[ACL 2026\] Adam's Law: Textual Frequency Law on Large Language Models](../../ACL2026/llm_nlp/adam39s_law_textual_frequency_law_on_large_language_models.md)

</div>

<!-- RELATED:END -->

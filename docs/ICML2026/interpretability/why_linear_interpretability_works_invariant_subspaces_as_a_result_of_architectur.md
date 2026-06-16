---
title: >-
  [Paper Note] Why Linear Interpretability Works: Invariant Subspaces as a Result of Architectural Constraints
description: >-
  [ICML 2026][Interpretability][linear probing] This paper provides an architectural-level explanation for why the internal representations of Transformers can be repeatedly decoded by simple linear methods (probes, SAEs, activation steering). It proves that as long as semantic features are read through **linear interfaces** such as OV circuits or unembedding layers
tags:
  - ICML 2026
  - Interpretability
  - linear probing
  - sparse autoencoder
  - invariant subspace
  - self-reference
date: 2026-05-08
content_hash: 5b7ee222e1cd329e
---
# Why Linear Interpretability Works: Invariant Subspaces as a Result of Architectural Constraints

**Conference**: ICML 2026  
**arXiv**: [2602.09783](https://arxiv.org/abs/2602.09783)  
**Code**: Not yet public (None)  
**Area**: Interpretability / Representation Geometry / Transformer Theory  
**Keywords**: linear probing, sparse autoencoder, invariant subspace, self-reference, unembedding geometry

## TL;DR
This paper provides an architectural-level explanation for why the internal representations of Transformers can be repeatedly decoded by simple linear methods (probes, SAEs, activation steering). It proves that as long as semantic features are read through **linear interfaces** such as OV circuits or unembedding layers, they must reside in a cross-context invariant linear subspace (the Invariant Subspace Necessity theorem). The authors further derive a zero-shot application—the Self-Reference Property—which posits that a token's own embedding direction serves as its conceptual direction, enabling unsupervised classification using the geometric location of class tokens.

## Background & Motivation

**Background**: Modern mechanistic interpretability research consistently finds that the internal states of Transformers can be "decoded" by extremely simple linear operations: linear probes extract semantic attributes from hidden states (Alain & Bengio 2016, Belinkov 2022); sparse autoencoders (SAEs) identify interpretable feature directions (Bricken et al. 2023, Cunningham et al. 2023); and single-vector activation steering stably modifies model behavior (Turner et al. 2023, Zou et al. 2023).

**Limitations of Prior Work**: Transformers are massive, deep, and highly non-linear systems; theoretically, their intermediate representations have no obligation to be "linearly readable." However, linear methods are widely effective. Is this an empirical coincidence or a necessity? Existing explanations either appeal to empirical observation or analyze optimization dynamics (e.g., next-token prediction and gradient descent implicit bias in Jiang et al. 2024) but do not address the question of "linearity as a necessity" from the perspective of the **architecture** itself.

**Key Challenge**: While optimization-based explanations clarify "how it was learned," they fail to explain "why all models satisfying this architecture are forced to do so." If a non-linear output head were used, would linear probes still hold? The authors hypothesize the answer is no, suggesting the root cause lies in the architecture's use of linear matrices (OV, unembedding) for inter-module communication.

**Goal**: (1) Formalize the "linear interface $\implies$ cross-context invariant linear subspace" relationship as a theorem; (2) Provide a falsifiable corollary (Self-Reference Property); (3) Validate these claims across multiple models and tasks.

**Key Insight**: The authors focus on how Transformer modules "talk" to each other—the attention OV circuit $W_O W_V$ and the unembedding $W_U$ are both **linear mappings** acting on the residual stream. Any semantic feature that must pass through these interfaces to affect the output must, by definition, satisfy "linear decodability," which is equivalent to residing in a specific linear subspace.

**Core Idea**: Use "architectural necessity" instead of "optimization contingency" to explain the success of linear interpretability; subsequently, propose "token embedding direction = concept direction" (self-reference) for zero-shot classification.

## Method

### Overall Architecture
The paper does not train new models but transforms the empirical phenomenon of linear interpretability into a provable architectural proposition. Given four architectural assumptions (additive residual stream, linear interfaces for OV/unembedding, parameter sharing, and linear output layers), the authors first prove a core theorem (Theorem 3.7: semantic features read through linear interfaces necessarily reside in cross-context invariant linear subspaces) and then use a capacity constraint proposition (Proposition 3.8) to show that when the vocabulary size is much larger than the dimension, these representations must undergo sparse factorization into shared directions. This leads to a zero-shot application: the embedding direction of a token itself is the geometric direction of its corresponding concept. The causal direction is corroborated through geometric alignment experiments across 8 classification tasks and 4 model families, plus a quasi-experimental control replacing the unembedding with an MLP head.

### Key Designs

**1. Invariant Subspace Necessity Theorem: Equating "Linear Decodability" with "Geometric Invariant Subspace"**

The central issue is that Transformers are deep non-linear systems with no inherent obligation for intermediate states to be linearly readable, yet probes/SAEs/steering succeed. The authors formalize a "communicable feature" $f: \mathcal{C} \to \mathcal{Y}$ with two conditions: multi-context requires that different surfaces $c_1, c_2$ express the same $f$ value (e.g., "France" and "the country of the Eiffel Tower"), and linear decodability requires a $\phi \in \mathbb{R}^{|V|}$ such that $\phi^\top W_U \mathbf{h}(c) = g(f(c))$ holds for all $c$. Given that linear interfaces imply a scalar readout $o_f(c) = \mathbf{w}_f^\top \mathbf{h}(c)$, any context providing the same $f$ value must remain consistent in the $\mathbf{w}_f$ direction and can only vary freely in the orthogonal complement $\mathbf{w}_f^\perp$. Thus, information about $f$ exists only in a context-independent subspace $\mathcal{S}_f$ determined by $\mathbf{w}_f$. Directional Invariance further tightens this to $\dim(\mathcal{S}_f)=1$. This equivalence explains why different tools like probes and SAEs often yield consistent conclusions: they are exploiting the same $\mathcal{S}_f$.

**2. Capacity Constraint Proposition: Vocabulary Size Forces Sparse Factorization**

Proving the existence of an invariant direction is insufficient; one must explain why SAEs find reusable sparse dictionaries. The authors start from the engineering reality that $|\mathcal{V}| \gg d$: individual token vectors $\mathbf{w}_t$ in the unembedding $W_U \in \mathbb{R}^{|\mathcal{V}| \times d}$ cannot all be orthogonal and must share directions. If contexts activate sparse sets of features and tokens share semantic attributes, the optimal representation factorizes as $\mathbf{w}_t = \sum_{f \in F_t} \alpha_{t,f} \mathbf{d}_f$, where the number of shared directions $|F| \ll |\mathcal{V}|$. Substituting this back, logits are expressed as $\text{logit}_t = \sum_{f \in F_t} \alpha_{t,f}\,(\mathbf{d}_f^\top \mathbf{h}(c))$, where each factor direction $\mathbf{d}_f$ satisfies the conditions for Theorem 3.7. This explains that SAE success is not a coincidence: capacity constraints, sparse activation, and multi-token semantic sharing force the model to organize representations in a form recoverable by sparse dictionaries.

**3. Self-Reference Property: Tokens as Concept Directions**

The preceding theorems suggest that conceptual directions $\mathbf{d}_f$ are determined by model parameters, but using them typically requires training a probe or running an unsupervised SAE. The authors identify the most direct reference vector as the token corresponding to the concept itself—the embedding direction of the token "France" provides the direction for the concept of France. Thus, hidden states for "I went to Paris" and "I visited Marseille" will both have strong projections along this direction. This allows for zero-shot unsupervised classification by using explicit token self-reference, where implicit instances in context share the same invariant direction as the concept token.

### Experimental Setup
The main results consist of mathematical proofs for the two theorems and one corollary. Validation involves no new training but measures alignment across LLaMA3-8B, Mistral-7B, GPT2-Small, and LLaMA3.2-3B on 8 classification tasks (taxonomic, affective, stylistic, etc.). It examines: (1) cosine alignment between class token directions and instance hidden states; (2) alignment between unsupervised SAE feature directions and class token directions; and (3) a control experiment comparing "modular division + MLP head" vs. the standard "linear head."

## Key Experimental Results

### Main Results
(Qualitative conclusions provided; specific data tables are in the appendix.)

| Validation Dimension | Phenomenon | Explanation |
|---|---|---|
| 8 Tasks × 4 Model Families | Persistent high alignment between class token directions and instance hidden states | Validates that directional invariance is robust across tasks and families |
| Unsupervised SAE Features | Significant alignment with class token directions | Validates that "two paths access the same $\mathcal{S}_f$" |
| Modular Division + MLP head (Fig 2) | Linear probe success ~20% for non-Fourier solutions; success only during Fourier solutions | Validates that "linear readout" is the cause of directional structure; it is not necessary under an MLP head |

### Ablation Study

| Configuration | Phenomenon | Description |
|---|---|---|
| Linear unembedding (Standard) | Linear probes necessarily succeed (theoretical guarantee) | Theorem 3.7 in effect |
| MLP classification head (Control) | Linear probes only succeed if Fourier representations are coincidentally found | Proves non-linear readout relaxes the hard constraint of "invariant subspaces" |
| Class token zero-shot probe | Achieving classification performance comparable to trained probes | Direct application of Self-Reference |

### Key Findings
- **"Linear Interface" is the key variable, not "Linear Representation"**: The modular division control experiment is crucial—replacing the readout with an MLP eliminates universal linear probe success, while returning to the linear unembedding restores directional structure. This identifies the causal direction as Architecture $\to$ Representation Form.
- **SAEs and Probes identify the same directions**: The alignment between unsupervised SAE directions and class token directions suggests these tools are not performing different tasks but accessing the same invariant subspace $\mathcal{S}_f$ via different methods.
- **Zero-shot geometric probing is feasible**: Classification without parameter training, using only token embedding directions, serves as a strong geometric baseline and allows for discovering interpretable directions in new tasks without labels.

## Highlights & Insights
- **Architecture vs. Optimization Dichotomy**: The authors position their explanation as "complementary" to optimization-based views (Jiang et al. 2024)—optimization determines "how it is learned," while architecture determines "the form it must take."
- **Efficiency of Theorem 3.7**: The core proof is remarkably concise, relying on the kernel of linear operators, representing a rare "less is more" theoretical contribution to mechanistic interpretability.
- **Utility of Self-Reference**: Translating abstract geometric theorems into "zero-shot unsupervised classification" provides a verifiable and engineering-relevant application, avoiding the common critique of being "profound but useless."
- **The Power of Control Experiments**: Using modular division to make effects disappear and reappear provides quasi-experimental evidence that the architecture is indeed the cause.

## Limitations & Future Work
- The assumption of an "additive residual stream" requires more nuanced arguments for modern variants using LayerNorm/RMSNorm (like Llama-3), as normalization is non-linear and affects the precise boundaries of the "linear interface."
- While OV is a linear interface, the attention mechanism includes Softmax. The paper does not fully discuss whether features on the Softmax path reside in the same invariant subspaces.
- Experiments are limited to classification tasks (8); it remains unverified if "non-classification semantics" like reasoning or in-context learning follow the same rules.
- Self-Reference assumes a "concept has a corresponding token." It cannot be directly applied to compound concepts (e.g., "countries I have visited") without extending to phrase embeddings or pooling.
- Scalability: Experiments cover models up to 8B. It remains to be seen if the single-direction geometric form becomes more complex as vocabulary size and dimensions change at extreme scales.

## Related Work & Insights
- **vs. Jiang et al. (2024)**: While they use implicit bias of gradient descent to explain linear representation, this paper provides a complementary architectural explanation. Both confirm that "linear representation is no accident."
- **vs. Park et al. (2024)**: While they formalize the geometric concept of linear representation, this paper identifies the conditions under which such geometry must form.
- **vs. Kantamneni et al. (2025)**: They empirically found that SAE latents do not always outperform linear probes; this paper provides the theoretical reason—both access the same invariant directions.
- **vs. Logit Lens / Tuned Lens**: These methods' success depends on the linearity of $W_U$ and the existence of invariant subspaces; this paper provides the theoretical backing for these practices.
- **Inspiration**: (1) Future "interpretability-friendly" architectures should preserve "linear last miles." (2) To maintain linear interpretability in multimodal models, cross-modal fusion should favor linear interfaces. (3) The Self-Reference idea can be generalized into "contrastive probing using the model's own token embeddings."

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Elevates linear interpretability from empirical observation to architectural necessity.
- Experimental Thoroughness: ⭐⭐⭐⭐ Solid verification across 4 models and 8 tasks plus control experiments, though limited to classification.
- Writing Quality: ⭐⭐⭐⭐⭐ Exceptionally clear logic chain (Assumptions-Definitions-Theorems-Corollaries).
- Value: ⭐⭐⭐⭐ Provides a much-needed unified framework and practical tools like zero-shot probing for mechanistic interpretability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Physics from Video: Identifiability of Time-Invariant Second-Order ODEs under Minimal Trajectory Conditions](physics_from_video_identifiability_of_time-invariant_second-order_odes_under_min.md)
- [\[ICLR 2026\] Decomposing Representation Space into Interpretable Subspaces with Unsupervised Learning](../../ICLR2026/interpretability/decomposing_representation_space_into_interpretable_subspaces_with_unsupervised_.md)
- [\[NeurIPS 2025\] The Non-Linear Representation Dilemma: Is Causal Abstraction Enough for Mechanistic Interpretability?](../../NeurIPS2025/interpretability/the_non-linear_representation_dilemma_is_causal_abstraction_enough_for_mechanist.md)
- [\[ICML 2026\] Learning Coherent Representations: A Topological Approach to Interpretability](learning_coherent_representations_a_topological_approach_to_interpretability.md)
- [\[ICML 2026\] Interpretability Can Be Actionable](interpretability_can_be_actionable.md)

</div>

<!-- RELATED:END -->

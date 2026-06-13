---
title: >-
  [Paper Note] Why Linear Interpretability Works: Invariant Subspaces as a Result of Architectural Constraints
description: >-
  [ICML 2026][Interpretability][linear probing] This paper provides an architectural-level explanation for why the internal representations of Transformers can be repeatedly and successfully decoded by simple linear method…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "linear probing"
  - "sparse autoencoder"
  - "invariant subspace"
  - "self-reference"
  - "unembedding geometry"
date: 2026-05-08
content_hash: 6b2157d6d579e01d
---

# Why Linear Interpretability Works: Invariant Subspaces as a Result of Architectural Constraints

**Conference**: ICML 2026  
**arXiv**: [2602.09783](https://arxiv.org/abs/2602.09783)  
**Code**: Not yet public (None)  
**Area**: Interpretability / Representation Geometry / Transformer Theory  
**Keywords**: linear probing, sparse autoencoder, invariant subspace, self-reference, unembedding geometry

## TL;DR
This paper provides an architectural-level explanation for why the internal representations of Transformers can be repeatedly and successfully decoded by simple linear methods (probes, SAEs, activation steering). As long as semantic features are read through **linear interfaces** such as OV circuits or unembeddings, they must reside in linear subspaces that are invariant across contexts (Invariant Subspace Necessity Theorem). The authors further derive a zero-shot application—the Self-Reference Property—where the embedding direction of a token itself serves as its conceptual direction, allowing for unsupervised classification based directly on the geometric position of class tokens.

## Background & Motivation

**Background**: Modern mechanistic interpretability consistently finds that the internal states of Transformers can be "decoded" by extremely simple linear operations: linear probes can extract semantic attributes from hidden states (Alain & Bengio 2016, Belinkov 2022); sparse autoencoders (SAE) can identify interpretable feature directions (Bricken et al. 2023, Cunningham et al. 2023); and single-vector activation steering can stably alter model behavior (Turner et al. 2023, Zou et al. 2023).

**Limitations of Prior Work**: Transformers are massive, deep, and highly non-linear systems; theoretically, their intermediate representations have no obligation to be "linearly readable." In practice, however, linear methods are widely effective. Is this an empirical coincidence or a necessity? Existing explanations either appeal to empirical observations or focus on optimization dynamics (e.g., Jiang et al. 2024 on next-token prediction and gradient descent implicit bias) but fail to address the necessity of linearity from the **architecture** itself.

**Key Challenge**: Optimization-based explanations describe "how this was learned" but cannot explain "why all models satisfying this architecture are forced to behave this way"—if a non-linear output head were used, would linear probing still hold? The authors hypothesize the answer is no, arguing the root is not in optimization but in the fact that Transformers use linear matrices (OV, unembedding) for inter-module communication.

**Goal**: (1) Formalize the "linear interface $\Rightarrow$ invariant linear subspace" relationship into a theorem; (2) provide a falsifiable corollary (Self-Reference Property); (3) validate these across multiple models and tasks.

**Key Insight**: The authors focus on how Transformer modules communicate with each other. Attention OV circuits $W_O W_V$ and the unembedding $W_U$ are **linear mappings** acting on the residual stream. Any semantic feature that must pass through these interfaces to affect output must, formally, satisfy "linear readability"—which is equivalent to residing in a specific linear subspace.

**Core Idea**: Replace "optimization coincidence" with "architectural necessity" to explain the success of linear interpretability; use this to propose "token embedding direction = concept direction" (self-reference) for zero-shot classification.

## Method

### Overall Architecture
Under four architectural assumptions (additive residual stream, linear OV and unembedding, parameter sharing, and linear output layers), the authors present two core theorems and one corollary:
- **Theorem 3.7 (Invariant Subspace Necessity)**: Communicable features decoded through a linear interface necessarily possess a cross-context invariant subspace $\mathcal{S}_f$;
- **Proposition 3.8 (Capacity Constraint Implies Feature Sharing)**: Given the capacity constraint $|\mathcal{V}| \gg d$, optimal token representations must factorize into sparse combinations of shared feature directions;
- **Self-Reference Property (Corollary)**: A token's own embedding vector provides the geometric direction of its concept, which can be used as an unsupervised probe zero-shot.

The authors validate these on 8 classification tasks across 4 model families by measuring geometric alignment between class tokens and instances, alignment between SAE feature directions and class tokens, and a control experiment replacing the unembedding with an MLP head.

### Key Designs

1.  **Invariant Subspace Necessity Theorem + Formalization of Communicable Features**:
    - **Function**: Condenses the reason for linear decodability into a provable statement: linear interfaces force invariant subspaces.
    - **Mechanism**: First, "communicable features" $f: \mathcal{C} \to \mathcal{Y}$ are formalized via two conditions: (i) multi-context: multiple distinct surfaces $c_1, c_2$ express the same $f$ value (e.g., "France" and "the country of the Eiffel Tower"); (ii) linear decodability: there exists $\phi \in \mathbb{R}^{|V|}$ such that $\phi^\top W_U \mathbf{h}(c) = g(f(c))$ holds for all $c$. The proof then shows that because there exists a scalar $\mathbf{w}_f \in \mathbb{R}^d$ such that $o_f(c) = \mathbf{w}_f^\top \mathbf{h}(c)$, any context providing the same $f$ value must vary freely in the $\mathbf{w}_f^\perp$ direction and remain consistent in the $\mathbf{w}_f$ direction. This means $f$-related information resides only in the subspace determined by $\mathbf{w}_f$, independent of context, thus $\mathcal{S}_f$ exists. Directional Invariance further implies $\dim(\mathcal{S}_f) = 1$.
    - **Design Motivation**: By formally equating "linear readability" with "geometrically invariant subspaces," disparate tools like linear probes, SAEs, and activation steering are unified as "exploiting the same $\mathcal{S}_f$," explaining consistent findings across these methods.

2.  **Capacity Constraint Corollary: The Necessity of Sparse Factorization**:
    - **Function**: Derives "tokens must share feature directions" from the engineering reality $|\mathcal{V}| \gg d$, implying that the sparse decompositions found by SAEs are inevitable.
    - **Mechanism**: Column vectors $\mathbf{w}_t$ in the unembedding $W_U \in \mathbb{R}^{|\mathcal{V}| \times d}$ cannot be orthogonal (since there are far more tokens than dimensions) and must share directions. If contexts activate sparse sets of features and tokens share semantic attributes, the optimal representation factorizes as $\mathbf{w}_t = \sum_{f \in F_t} \alpha_{t,f} \mathbf{d}_f$, where $|F| \ll |\mathcal{V}|$. Substituting this gives $\text{logit}_t = \sum_{f \in F_t} \alpha_{t,f} (\mathbf{d}_f^\top \mathbf{h}(c))$, where each factor $\mathbf{d}_f$ must be linearly decodable and context-invariant—satisfying the conditions of Theorem 3.7.
    - **Design Motivation**: This shows that the success of SAEs is not accidental. The combination of capacity constraints, sparse activation, and shared semantics forces the model to organize representations into a form recoverable by sparse dictionaries.

3.  **Self-Reference Property: Tokens as Their Own Concept Directions**:
    - **Function**: Provides a direct, zero-shot method to identify semantic directions without training probes.
    - **Mechanism**: Derived from the theorems, the direction $\mathbf{d}_f$ of a concept $f$ is determined by model parameters. The most direct "reference vector" is the token corresponding to $f$ itself. For example, the embedding direction of the token "France" provides the direction for the concept of France. Hidden states for "I went to Paris" and "I visited Marseille" will have strong projections along this direction.
    - **Design Motivation**: While probes/SAEs require labels or extensive training, the self-reference property maps concept directions directly to token embeddings, providing a zero-parameter geometric baseline.

### Loss & Training
The paper does not train new models but focuses on theory and validation:
- The main results are mathematical proofs for two theorems and one corollary.
- Validation experiments use LLaMA3-8B, Mistral-7B, GPT2-Small, and LLaMA3.2-3B backbones across 8 classification tasks to measure: (a) cosine alignment between class token directions and instance hidden states; (b) alignment between unsupervised SAE features and class tokens; (c) a control experiment comparing "modular division + MLP head vs. linear head."

## Key Experimental Results

### Main Results

| Validation Dimension | Phenomenon | Explanation |
| :--- | :--- | :--- |
| 8 Tasks × 4 Model Families | Consistent high alignment between class tokens and instances | Verifies directional invariance is robust across tasks/families |
| Unsupervised SAE Features | Significant alignment with class token directions | Confirms both tools access the same $\mathcal{S}_f$ |
| Modular division + MLP head | Linear probe ~20% without Fourier solution; successful with it | Confirms "linear readout" is the cause of directional structure |

### Ablation Study

| Configuration | Phenomenon | Description |
| :--- | :--- | :--- |
| Linear unembedding (Standard) | Linear probe necessarily succeeds | Theorem 3.7 in effect |
| MLP classification head (Control) | Linear probe only succeeds if Fourier representation is used | Non-linear readout removes the hard constraint of invariant subspaces |
| Class token zero-shot probe | Classification performance comparable to trained probes | Direct application of the Self-Reference Property |

### Key Findings
- **"Linear Interface" is the key variable, not "Linear Representation"**: The modular division experiment is crucial. Swapping the readout for an MLP causes linear probe success to plummet; switching back to linear unembedding causes directional structure to re-emerge. This establishes the causal direction from architecture to representation format.
- **SAEs and Probes find the same directions**: The alignment between unsupervised SAE features and class tokens suggests these tools are not performing different tasks but accessing the same invariant subspace $\mathcal{S}_f$ via different methods.
- **Zero-shot geometric probing is viable**: Classifying based on token embedding directions without any parameter training provides a powerful geometric baseline and allows for discovering interpretable directions in tasks without labels.

## Highlights & Insights
- **Architecture vs. Optimization**: The authors position their explanation as complementary to the optimization-based view (Jiang et al., 2024). Optimization determines "how" it is learned, while architecture determines the "form" it must take.
- **Conciseness of Theorem 3.7**: The proof relies on the null space of linear operators, serving as a rare "less is more" theoretical contribution to mechanistic interpretability.
- **Practicality of Self-Reference**: Translating abstract geometric theorems into a concrete zero-shot classification capability ensures the theory is verifiable and engineering-relevant.
- **Modular Division Experiment**: A clever design that transforms a necessity claim into a falsifiable experiment by showing that removing the linear constraint makes the effect disappear.

## Limitations & Future Work
- Assumption 1 requires an "additive residual stream." Modern variants using RMSNorm or post-norm require more detailed analysis, as normalization is non-linear and affects the "linear interface" boundaries.
- Although OV is linear, the Attention mechanism includes non-linear softmax operations. The paper does not fully discuss whether features on the softmax path also reside in invariant subspaces.
- The experiments are limited to 8 classification tasks. It remains unverified if directional invariance holds for non-classification semantics like reasoning or long-context dependency.
- Self-Reference assumes a single token represents a concept. It cannot be directly applied to composite concepts without single-token representations (e.g., "countries I have visited").
- The experiments cover models up to 8B parameters. As the ratio of $|V|$ to $d$ changes in larger models, the geometric morphology of factorizations may become more complex.

## Related Work & Insights
- **vs. Jiang et al. (2024)**: Uses implicit bias of gradient descent to explain linear representation; this paper provides a complementary architectural explanation. Both confirm linear representation is not accidental.
- **vs. Park et al. (2024)**: Formalizes conceptual geometry in linear representations; this paper identifies the conditions under which such geometry must form.
- **vs. Kantamneni et al. (2025)**: Found SAE latents don't always outperform linear probes; this paper explains why—both access the same set of invariant directions.
- **vs. Logit Lens / Tuned Lens**: These practical methods rely on the linearity of $W_U$ and the existence of invariant subspaces; this paper provides their theoretical foundation.
- **Insights**: (1) New architectures should retain "linear last miles" to remain interpretability-friendly; (2) Multi-modal fusion layers should maintain linear interfaces to preserve linear interpretability; (3) Self-reference can be extended to use the model's own embeddings as contrastive probes.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Elevates linear interpretability from empirical observation to architectural necessity.
- Experimental Thoroughness: ⭐⭐⭐⭐ Strong support from multi-task/multi-model validation and counter-experiments, though limited to classification and smaller models.
- Writing Quality: ⭐⭐⭐⭐⭐ Clean logical chain from definitions to theorems; the experimental design is exemplary for theoretical interpretability papers.
- Value: ⭐⭐⭐⭐ Provides a much-needed unifying framework for mechanistic interpretability and produces practical zero-shot tools.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Physics from Video: Identifiability of Time-Invariant Second-Order ODEs under Minimal Trajectory Conditions](physics_from_video_identifiability_of_time-invariant_second-order_odes_under_min.md)
- [\[ICLR 2026\] Decomposing Representation Space into Interpretable Subspaces with Unsupervised Learning](../../ICLR2026/interpretability/decomposing_representation_space_into_interpretable_subspaces_with_unsupervised_.md)
- [\[NeurIPS 2025\] The Non-Linear Representation Dilemma: Is Causal Abstraction Enough for Mechanistic Interpretability?](../../NeurIPS2025/interpretability/the_non-linear_representation_dilemma_is_causal_abstraction_enough_for_mechanist.md)
- [\[ICML 2026\] Learning Coherent Representations: A Topological Approach to Interpretability](learning_coherent_representations_a_topological_approach_to_interpretability.md)
- [\[ACL 2026\] Rhetorical Questions in LLM Representations: A Linear Probing Study](../../ACL2026/interpretability/rhetorical_questions_in_llm_representations_a_linear_probing_study.md)

</div>

<!-- RELATED:END -->

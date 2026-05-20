---
title: >-
  [Paper Note] Why Linear Interpretability Works: Invariant Subspaces as a Result of Architectural Constraints
description: >-
  [ICML 2026][Interpretability][linear probing] This paper provides an architectural-level explanation for "why the internal representations of transformers can be repeatedly and successfully decoded by simple linear metho…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "linear probing"
  - "sparse autoencoder"
  - "invariant subspace"
  - "self-reference"
  - "unembedding geometry"
date: 2026-05-08
content_hash: 3675bc3a25e23592
---

# Why Linear Interpretability Works: Invariant Subspaces as a Result of Architectural Constraints

**Conference**: ICML 2026  
**arXiv**: [2602.09783](https://arxiv.org/abs/2602.09783)  
**Code**: Not yet released (none)  
**Area**: Interpretability / Representation Geometry / Transformer Theory  
**Keywords**: linear probing, sparse autoencoder, invariant subspace, self-reference, unembedding geometry

## TL;DR
This paper provides an architectural-level explanation for "why the internal representations of transformers can be repeatedly and successfully decoded by simple linear methods (probe, SAE, activation steering)": as long as semantic features are read out via **linear interfaces** such as OV circuits or unembedding, they must reside in a context-invariant linear subspace (Invariant Subspace Necessity theorem); this leads to a zero-shot application—the Self-Reference Property, i.e., the embedding direction of a token itself is its concept direction, enabling unsupervised classification directly using the geometric position of class tokens.

## Background & Motivation

**Background**: Modern mechanistic interpretability repeatedly finds that the internal states of transformers can be "decoded" by extremely simple linear operations: linear probes can extract semantic attributes from hidden states (Alain & Bengio 2016, Belinkov 2022); sparse autoencoders (SAE) can identify interpretable feature directions (Bricken et al. 2023, Cunningham et al. 2023); single-vector activation steering can reliably alter model behavior (Turner et al. 2023, Zou et al. 2023).

**Limitations of Prior Work**: Transformers are highly parameterized, deep, and strongly nonlinear systems; in principle, their intermediate representations have no obligation to be "linearly readable." Yet, linear methods are widely effective—is this due to empirical coincidence or necessity? Existing explanations either appeal to "empirical observation" or approach from optimization dynamics (e.g., Jiang et al. 2024's next-token + gradient descent implicit bias), but none address the "must be linear" question from the **architecture** itself.

**Key Challenge**: Optimization-based explanations tell us "why this is learned," but cannot explain "why all models with this architecture are forced to do so"—if the output head were nonlinear, would linear probes still work? The authors conjecture the answer is no; the root cause lies not in optimization but in the fact that transformers use linear matrices (OV, unembedding) for inter-module communication.

**Goal**: (1) Formalize "linear interface ⇒ context-invariant linear subspace" as a theorem; (2) Provide experimentally testable, falsifiable predictions (Self-Reference Property); (3) Validate across multiple models and tasks.

**Key Insight**: The authors focus on "how transformer modules communicate"—the attention OV circuit $W_O W_V$ and unembedding $W_U$ are both **linear mappings** acting on the residual stream. Any semantic feature that must pass through these interfaces to affect the output must, formally, be "linearly readable"—which is equivalent to residing in a linear subspace.

**Core Idea**: Use "architectural necessity" instead of "optimization contingency" to explain the success of linear interpretability methods; and, based on this, propose self-reference: "token embedding direction = concept direction," enabling zero-shot classification.

## Method

### Overall Architecture
Under four architectural assumptions (additive residual stream, linear OV and unembedding, parameter sharing, linear output layer), the authors present two core theorems and one corollary:
- **Theorem 3.7 (Invariant Subspace Necessity)**: Any communicable feature decoded via a linear interface must exist in a context-invariant subspace $\mathcal{S}_f$;
- **Proposition 3.8 (Capacity Constraint Implies Feature Sharing)**: Under the capacity constraint $|\mathcal{V}| \gg d$, optimal token representations must factorize as sparse combinations of shared feature directions;
- **Self-Reference Property (Corollary)**: The embedding vector of a token itself gives the geometric direction of its concept, enabling zero-shot unsupervised probing.

Validation is performed on 8 classification tasks × 4 model families: by measuring the geometric alignment between class tokens and instances of that class, the alignment between SAE-learned feature directions and class token directions, and a control experiment (replacing unembedding with an MLP head).

### Key Designs

1. **Invariant Subspace Necessity Theorem + Formal Definition of Communicable Features**:

    - **Function**: Compresses "why linearly decodable" into a provable statement—linear interfaces enforce invariant subspaces.
    - **Mechanism**: Formalizes "communicable feature" $f: \mathcal{C} \to \mathcal{Y}$ with two conditions—(i) multi-context: multiple distinct surfaces $c_1, c_2$ express the same $f$ value (e.g., "France" and "the country of the Eiffel Tower" both refer to France); (ii) linear decodability: there exists $\phi \in \mathbb{R}^{|V|}$ such that $\phi^\top W_U \mathbf{h}(c) = g(f(c))$ for all $c$. The proof shows: since there exists a scalar $\mathbf{w}_f \in \mathbb{R}^d$ such that $o_f(c) = \mathbf{w}_f^\top \mathbf{h}(c)$, any context must vary freely in the $\mathbf{w}_f^\perp$ direction and remain consistent in the $\mathbf{w}_f$ direction to yield the same $f$ value—i.e., $f$-related information lives only in the subspace determined by $\mathbf{w}_f$, independent of context, thus $\mathcal{S}_f$ exists. Directional Invariance further requires $\dim(\mathcal{S}_f) = 1$, i.e., a single direction suffices.
    - **Design Motivation**: By equating "linear readability" and "geometric invariant subspace," linear probe, SAE, and activation steering—seemingly different practical tools—are unified as "using the same $\mathcal{S}_f$," explaining why they often yield consistent conclusions.

2. **Capacity Constraint Corollary: Sparse Factorization Must Occur**:

    - **Function**: From the engineering reality $|\mathcal{V}| \gg d$, deduces that "tokens must share feature directions," further implying that the sparse decomposition found by SAE is inevitable.
    - **Mechanism**: In unembedding $W_U \in \mathbb{R}^{|\mathcal{V}| \times d}$, each token's column vector $\mathbf{w}_t$ cannot be mutually orthogonal (since tokens far outnumber dimensions), so directions must be shared. If each context activates only a sparse set of features and multiple tokens share semantic attributes, the optimal representation factorizes as $\mathbf{w}_t = \sum_{f \in F_t} \alpha_{t,f} \mathbf{d}_f$, with the number of shared directions $|F| \ll |\mathcal{V}|$. Substituting, the logit $\text{logit}_t = \sum_{f \in F_t} \alpha_{t,f} (\mathbf{d}_f^\top \mathbf{h}(c))$, and each factor $\mathbf{d}_f$ must be linearly decodable and context-invariant—satisfying Theorem 3.7.
    - **Design Motivation**: This proposition shows that "SAE's success" is not accidental—capacity constraint + sparse activation + multi-token semantic sharing jointly require the model to organize representations in a form recoverable by a sparse dictionary; this also explains why SAE dictionaries and linear probe directions often coincide.

3. **Self-Reference Property: Token Itself Is Its Concept Direction**:

    - **Function**: Provides a direct, zero-shot, parameter-free method for identifying semantic directions.
    - **Mechanism**: The theorem shows that the direction $\mathbf{d}_f$ of concept $f$ is entirely determined by model parameters. The most direct "reference vector" is the token itself corresponding to $f$—for example, the embedding direction of the token "France" gives the direction of the France concept; thus, "I went to Paris" and "I visited Marseille" will both have strong projections in this direction in the hidden state, enabling zero-shot unsupervised classification. Intuitively, as in Figure 1: the explicit token "France" self-referentially provides the direction, and implicit instances in context share this invariant direction.
    - **Design Motivation**: Previously, probe/SAE required labels or extensive unsupervised training; the self-reference property directly grounds "concept direction" in token embeddings, providing a zero-parameter geometric baseline to sanity-check whether probe-found directions truly describe the same concept.

### Loss & Training
No new models are trained; the work is mainly theoretical plus validation experiments:
- The main results are two theorems and one corollary, all mathematically proven.
- Validation experiments are conducted on LLaMA3-8B, Mistral-7B, GPT2-Small, and LLaMA3.2-3B backbones, across 8 semantic classification tasks (taxonomic, affective, stylistic, linguistic, descriptive, etc.), measuring (a) cosine alignment between class token direction and corresponding instance hidden states; (b) alignment between unsupervised SAE-learned feature directions and class token directions; (c) a control experiment comparing "modular division + MLP head vs. linear head."

## Key Experimental Results

### Main Results
(The paper provides qualitative conclusions within the cached range; detailed tables are in the appendix.)

| Dimension | Phenomenon | Explanation |
|---|---|---|
| 8 classification tasks × 4 model families | Class token direction remains highly aligned with hidden states of same-class instances | Validates directional invariance is robust across tasks and families |
| Unsupervised SAE feature direction | Significantly aligned with class token direction | Validates "two paths access the same $\mathcal{S}_f$" |
| Modular division + MLP head (Figure 2 control) | When the model finds a non-Fourier solution, linear probe ~20%; when it finds a Fourier solution, probe succeeds | Validates that "linear readout" is the cause of directional structure; with MLP head, it is no longer necessary |

### Ablation Study

| Configuration | Phenomenon | Note |
|---|---|---|
| Linear unembedding (standard transformer) | Linear probe always succeeds (theoretical guarantee) | Theorem 3.7 applies |
| MLP classification head (control) | Linear probe only succeeds when a Fourier representation is found by chance | Shows that nonlinear readout removes the hard constraint of "must be invariant subspace" |
| Class token zero-shot probe | Achieves classification performance comparable to trained probes on multiple tasks | Direct application of Self-Reference |

### Key Findings
- **"Linear interface" is the key variable, not "linear representation"**: The modular division control experiment in Figure 2 is crucial—on the same task, replacing the readout with an MLP causes linear probes to no longer generally succeed; switching back to linear unembedding, directional structure reappears. This is the strongest empirical evidence in the paper, pinning down the causal direction from "architecture → representation form."
- **SAE and probe find the same set of directions**: The alignment between unsupervised SAE feature directions and class token directions means these tools are not doing different things, but accessing the same invariant subspace $\mathcal{S}_f$ by different means, unifying previously divergent interpretability approaches.
- **Zero-shot geometric probe is feasible**: Without training any parameters, using only token embedding directions for classification provides a strong geometric baseline against label-dependent probes, and enables interpretable directions for new tasks without labels.

## Highlights & Insights
- **Architecture vs. Optimization Dichotomy**: The authors explicitly position their explanation as complementary to Jiang et al. (2024)'s optimization-based explanation—optimization determines "how it is learned," architecture determines "what form it must take." This layered explanation is very clear.
- **Theorem 3.7's proof is extremely concise**: Using only a few lines about "linear operator kernels," the main claim is established—an example of "less is more" theoretical contribution in mechanistic interpretability.
- **Application value of Self-Reference**: The high-level geometric theorem is grounded in the concrete ability of "zero-shot unsupervised classification," making the theory verifiable and engineering-relevant, avoiding the common criticism of "seemingly profound but useless" pure math papers.
- **Modular division control experiment**: This is the key design that turns the "necessity" claim into a falsifiable experiment—by swapping the head to make the effect disappear and then reappear, it provides quasi-experimental evidence that "architecture is the cause."

## Limitations & Future Work
- Assumption 1 requires an "additive residual stream," which needs more detailed argumentation for modern variants with RMSNorm/post-norm (e.g., Llama-3's actual normalization); normalization itself is not linear and may affect the precise boundary of the "linear interface."
- Assumption 2 treats OV as a linear interface, but actual attention also includes softmax; although the softmax output is a convex combination, the preceding query-key dot product is a nonlinear influence—the paper does not fully discuss whether features along the softmax path also reside in invariant subspaces.
- Experimental tasks are limited to classification (8 in total); it is unverified whether the findings hold for reasoning, in-context learning, long-context, and other "non-classification semantics." Directional invariance may weaken when task-specific context modulation is involved.
- The Self-Reference property assumes "concepts have corresponding tokens"; for composite concepts without single-token expressions ("countries I have visited," "code written yesterday"), it cannot be directly applied and needs extension to phrase embeddings or multi-token pooling.
- Experiments only cover 4 relatively small open-source models (up to 8B); at larger scales, if $|V|$ continues to increase and $d$ decreases proportionally, the geometry of factorization may become more complex; whether single-direction still holds at ultra-large scales requires further validation.

## Related Work & Insights
- **vs. Jiang et al. (2024)**: Uses next-token + gradient descent implicit bias to explain linear representation; this paper provides a complementary explanation from architectural necessity, together confirming that "linear representation is not accidental."
- **vs. Park et al. (2024)**: Formalizes the concept geometry of "linear representation"; this paper further points out the necessary conditions for such geometry to form.
- **vs. Kantamneni et al. (2025)**: Empirically finds that SAE latents do not always outperform linear probes on probing tasks; this paper provides a theoretical explanation—both access the same set of invariant directions.
- **vs. nostalgebraist (2020) Logit Lens / Belrose et al. (2023) Tuned Lens**: The validity of these practical methods depends on $W_U$ being linear and the existence of invariant subspaces; this paper provides theoretical support for them.
- **Insights**: (1) To design new architectures friendly to interpretability, the "linear last mile" should be retained, or probe/SAE tools will fail; (2) In multimodal models, to preserve linear interpretability, cross-modal fusion layers should also maintain linear interfaces as much as possible; (3) The zero-shot classification idea of Self-Reference can be generalized to "using the model's own token embeddings for contrastive probing."

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Elevates "why linear interpretability works" from empirical observation to an architectural necessity theorem—a rare hard theoretical contribution in mechanistic interpretability.
- Experimental Thoroughness: ⭐⭐⭐⭐ Alignment validation on 4 models × 8 tasks plus a modular division quasi-experimental control sufficiently supports the main claims; but all tasks are classification and the largest model is 8B, so coverage is somewhat limited.
- Writing Quality: ⭐⭐⭐⭐⭐ The hypothesis-definition-theorem-corollary argument chain is clean and concise, and the control experiment design is ingenious—a model for writing theoretical interpretability papers.
- Value: ⭐⭐⭐⭐ Provides a much-needed unifying framework for the rapidly developing field of mechanistic interpretability, and directly incubates usable tools such as zero-shot probes.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] The Non-Linear Representation Dilemma: Is Causal Abstraction Enough for Mechanistic Interpretability?](../../NeurIPS2025/interpretability/the_non-linear_representation_dilemma_is_causal_abstraction_enough_for_mechanist.md)
- [\[ICLR 2026\] Decomposing Representation Space into Interpretable Subspaces with Unsupervised Learning](../../ICLR2026/interpretability/decomposing_representation_space_into_interpretable_subspaces_with_unsupervised_.md)
- [\[ICML 2026\] Interpretability Can Be Actionable](interpretability_can_be_actionable.md)
- [\[NeurIPS 2025\] Why Is Attention Sparse in Particle Transformer?](../../NeurIPS2025/interpretability/why_is_attention_sparse_in_particle_transformer.md)
- [\[ACL 2026\] Rhetorical Questions in LLM Representations: A Linear Probing Study](../../ACL2026/interpretability/rhetorical_questions_in_llm_representations_a_linear_probing_study.md)

</div>

<!-- RELATED:END -->

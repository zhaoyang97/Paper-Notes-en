---
title: >-
  [Paper Note] Automated Interpretability Metrics Do Not Distinguish Trained and Random Transformers
description: >-
  [ICLR 2026][Interpretability][Paper Note] This paper conducts a "sanity check" on the currently popular Sparse Autoencoders (SAEs). By applying SAEs to both trained Transformers and **randomly initialized** Transformers, the authors find that commonly used automated interpretability scores (auto-interp AUROC) and reconstruction metrics are almost indistinguish
tags:
  - ICLR 2026
  - Interpretability
date: 2026-05-08
content_hash: 2b2de444640578aa
---
# Automated Interpretability Metrics Do Not Distinguish Trained and Random Transformers

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=USyGD0eUod](https://openreview.net/forum?id=USyGD0eUod)  
**Code**: To be confirmed  
**Area**: Interpretability / Mechanistic Interpretability / Sparse Autoencoders  
**Keywords**: Sparse Autoencoders, Automated Interpretability, Random Baselines, Superposition, Evaluation Metrics

## TL;DR
This paper conducts a "sanity check" on the currently popular Sparse Autoencoders (SAEs). By applying SAEs to both trained Transformers and **randomly initialized** Transformers, the authors find that commonly used automated interpretability scores (auto-interp AUROC) and reconstruction metrics are almost indistinguishable between the two. This suggests that high interpretability scores alone **cannot** prove that an SAE has captured computational features actually learned by the model.

## Background & Motivation

**Background**: The mainstream tool for mechanistic interpretability in recent years has been the Sparse Autoencoder. It trains an autoencoder with a hidden dimension much larger than the input on the activations of a specific Transformer layer. Relying on sparsity constraints to force out tens of thousands of "latents," the goal is for each latent to correspond to a human-readable "concept/feature," thereby decomposing the model's internal superposition. To compare different SAEs, the community developed **automated interpretability** evaluations: a large language model generates natural language explanations for a latent's activation pattern, which are then used to predict whether the latent activates on new text, using AUROC to measure explanation fidelity.

**Limitations of Prior Work**: This evaluation suite relies on a rarely questioned premise: what does a high score actually signify? It is generally assumed that "high auto-interp score = SAE has found meaningful features learned by the model." However, no one has systematically verified whether these scores merely reflect simple statistical structures derived from the **data itself or architectural inductive biases**, independent of the "training" process.

**Key Challenge**: A trustworthy interpretability method must have evaluation metrics capable of distinguishing "features learned through training" from "artifacts inherent to data/architecture." A classic way to test this is by comparing against a **strong null model**, such as replacing weights with a randomly initialized network (following the logic of Adebayo et al. 2020 for saliency map sanity checks). While interpretability research repeatedly emphasizes explaining "what the model has learned," it almost never uses random-weight models as a control.

**Goal**: To port the sanity checks from the saliency-map field to SAEs, answering two questions: (1) Can common SAE metrics distinguish trained vs. random Transformers? (2) If not, what mechanism makes activations in random networks appear "sparsely interpretable"?

**Key Insight**: The authors directly train SAEs on several sizes of Pythia models, using both the trained versions and various randomized variants, to see if the full suite of metrics overlaps. The advantage of this approach is its minimal assumptions—if scores for random variants closely track the trained version, it suffices to falsify the claim that "high score = learned features."

**Core Idea**: Use randomly initialized Transformers as a null model for SAE evaluation, revealing the phenomenon that "automated interpretability scores cannot distinguish trained from random networks," and subsequently advocating for the inclusion of random baselines and "abstractness" measures in standard evaluations.

## Method

This is a critique/analysis paper that does not propose a new model. The "method" consists of a set of controlled experimental designs and a toy model to explain the observed phenomena. The core involves constructing a set of Transformer variants ranging from "fully trained" to "fully random," training SAEs independently on each, and performing a cross-comparison of metrics. When metrics fail to distinguish them, a toy model is used to explain "why random networks also produce seemingly interpretable sparse activations."

### Overall Architecture

The experimental subjects are the Pythia family (70M–6.9B parameters) using RedPajama data. For the residual stream activations of a specific layer in each underlying Transformer, a TopK Sparse Autoencoder is trained (expansion factor $R=64$, sparsity $k=32$, 100M tokens). Crucially, five variants of the underlying Transformer are constructed, degrading from "truly trained" to "purely random," to be interpreted by SAEs:

- **Trained**: A normally pre-trained model.
- **Re-randomized incl. embeddings**: All weights (including word embeddings) are resampled as Gaussian noise, but with the **mean and variance aligned** to the statistics of the original trained weight matrices.
- **Re-randomized excl. embeddings**: Same as above, but the trained embedding/unembedding matrices are retained, with only other weights randomized.
- **Step-0**: The initialization checkpoint provided by Pythia (original random weights before training).
- **Control**: Uses the trained model, but during inference, the input embedding for each token is replaced with i.i.d. standard Gaussian noise—the same token does not have a fixed embedding vector across occurrences. This serves as a negative control that "should drop to random levels," with expected auto-interp performance near chance.

Evaluation metrics include: fuzzing/detection auto-interp AUROC, explained variance $R^2$, reconstruction cosine similarity, L1 norm, CE loss recovery score, and a custom token distribution entropy. The primary narrative is simple: except for the Control, the other four variants (including the purely random Step-0) overlap across most metrics and closely track the Trained model. Only the Control drops to random levels—empirically demonstrating that metrics fail to distinguish trained from random.

### Key Designs

**1. Randomization Scheme: Norm-Preserving Gaussian Resampling vs. Simple Zeroing**

To make the "random null model" persuasive, random weights must not create trivial differences. The authors found that parameter norms can vary significantly between a trained model and its Step-0 state, and norm scales affect activation growth in the residual stream. Thus, the Re-randomized variants **align the mean and variance of each matrix to the original trained weights** during Gaussian resampling to preserve parameter norms. Consequently, the two norm-preserving random variants (blue/orange lines) actually track the Trained model more closely than the Step-0 initialization (green line). This indicates that many "seemingly learned" metrics are sensitive only to parameter scale, not to whether training actually occurred—an anti-intuitive piece of evidence.

**2. Five-Stage Variant Ladder + Embedding Decoupling: Locating the Signal**

A binary "trained vs. random" comparison is insufficient because signals might originate from embeddings, architecture, or data. The ladder identifies where structure comes from: Re-randomized incl./excl. embeddings versions isolate whether embeddings carry interpretable structure; Step-0 represents pure initialization; and Control is the strongest negative control—it **retains all trained weights** but feeds random embeddings into the model during inference, destroying the consistent mapping between tokens and representations. Control is vital for setting the "bottom-line for guessing" (fuzzing AUROC $\approx 0.50$). Since even a trained model with random embeddings can only guess, the fact that random-weight variants achieve AUROC $\approx 0.87$ proves these scores measure preserved input/architectural structures rather than learned computations.

**3. Token Distribution Entropy: A Metric for "Abstractness"**

A significant constructive contribution is the identification that standard metrics miss the "abstractness" of features. Many SAE latents activate on only one or a few token IDs—such latents are easily explained but trivial. The authors quantify this using the **entropy** of the latent activation distribution over token IDs. By aggregating activations for a latent across its top-activating samples by token:

$$H = -\sum_{i} p_i \log p_i,$$

Higher entropy indicates "diffuse" activations not tied to a specific token, suggesting more abstract features. The key finding: the entropy of the Trained variant **increases with layer depth** (deeper features are more abstract and less like token embeddings), while random variants' entropy **remains low** (latents remain tied to one or two tokens). Control entropy is consistently high due to random Gaussian sampling. Thus, at similar auto-interp scores, token distribution entropy can distinguish features that become abstract with training from those that remain at the token level—a difference invisible to aggregate metrics.

**4. Toy Model of Superposition: Why Random Networks Appear Sparsely Interpretable**

The authors use a toy model to address the mechanism. They propose two hypotheses: (1) input data already possesses a superposition structure that random networks simply **preserve**; (2) random networks might **amplify/introduce** sparsity. The first point is shown via linear algebra: if sparse features $z$ are projected via $D$ into dense inputs $x \sim \mathcal{N}(Dz,\Sigma)$, then applying any weight $W$ results in $x'=Wx$, which still follows an isomorphic superposition generative model $x' \sim \mathcal{N}(WDz, W\Sigma W^{T})$—matrix multiplication does not destroy superposition. The second point is supported by experiments: superposition inputs vs. Gaussian controls (matched for mean/variance) are passed through a random two-layer MLP. The random MLP outputs are generally sparser for a given explained variance, and this sparsity is largely insensitive to input distribution, suggesting random networks inherently "sparsify" inputs.

### A Full Example: How Fuzzing Evaluations are "Fooled"

Taking Pythia-6.9b as an example: SAEs are trained on a residual stream layer, 100 latents are sampled, and Llama-3.1-70B generates explanations for each. During fuzzing, positive/negative tokens (activating/not) are marked with special symbols, and the LLM judges which labels are correct based on the explanation to calculate AUROC. Trained yields $\approx 0.79$, while Re-randomized (incl./excl. embeddings) and Step-0 all reach $0.87$–$0.88$—higher than the trained model. Only the Control (trained model with random embeddings) drops to $0.50$ (chance). Someone looking only at the ROC curve would conclude "these SAEs have all captured interpretable features," despite three of the lines coming from weights with zero linguistic training.

## Key Experimental Results

### Main Results
Fuzzing ROC for Pythia-6.9b (100 latents per SAE) shows trained and random variants almost overlapping, with only Control near the random line:

| Variant | Fuzzing AUROC | Description |
|------|---------------|------|
| Trained | 0.79 | Normally trained model |
| Re-randomized excl. emb | 0.87 | Random weights, retained trained embeddings |
| Re-randomized incl. emb | 0.87 | All weights randomized (norm-preserved) |
| Step-0 | 0.88 | Original random weights at initialization |
| Control | 0.50 | Trained weights + random embeddings (chance) |

### Ablation Study
Across the 70M–6.9B spectrum, all variants except Control show highly consistent trends in explained variance, cosine similarity, and AUROC (fuzz/detection). Only token distribution entropy separates trained from random:

| Metric | Trained | Random Variants | Control | Phenomenon |
|------|---------|----------|---------|------|
| Explained Variance $R^2$ / Cosine Sim | High | Near Trained | Significantly lower | Control is hardest to reconstruct (highest entropy) |
| AUROC (vs. model size) | Increases | Increases | $\approx$ Random | Larger model latents are more specific/easier to classify |
| CE loss recovery score | Meaningful | Negligible | — | Only Trained variant CE scores are interpretable |
| Token Distribution Entropy | Rises with layer | Consistently low | Constantly high | Only this metric reflects "abstractness" differences |

### Key Findings
- **Metrics fail to distinguish**: Norm-preserved random weights track trained models on auto-interp and reconstruction metrics; AUROCs are often higher, proving aggregate scores do not imply captured learned computation.
- **Abstractness is the divide**: Features in trained models become abstract as depth increases (entropy rises), whereas random model features remain at the token level (entropy stays low).
- **Sparsity may be architectural**: Random MLPs preserve or amplify input superposition/sparsity, so "interpretable sparse activations" are not necessarily a product of learning.
- **AUROC increases with scale**: Except for Control, AUROC rises with model size; the authors hypothesize that larger SAE latents need to explain less input space and become more specific, making classification easier.

## Highlights & Insights
- **Porting saliency-map sanity checks to SAEs**: Using a random weight null model to test interpretability methods is a simple but devastatingly effective experimental paradigm.
- **Norm-preserving randomization is key**: Instead of simple zeroing, aligning mean and variance prevents the counter-argument that "differences come from scale." The fact that randomized variants look *more* like the trained version after alignment strengthens the conclusion.
- **Token distribution entropy as a constructive solution**: The paper provides a cheap, immediate auxiliary metric to capture "abstractness" and points towards requiring random baselines in regular evaluations.
- **Transferable methodology**: Any tool claiming to "discover internal model structure" can be tested against a statistics-preserved random null model. This logic applies to probing, circuit discovery, and steering vectors.

## Limitations & Future Work
- **Scope**: Validated only on the Pythia family and RedPajama. It cannot exhaustively cover all data/architectures. LLM explainers were fixed to Llama-3.1-70B; changing explainers might affect aggregate behavior.
- **Clarification of Boundaries**: The paper **does not claim** that SAEs learn nothing on trained models; it only claims that "aggregate auto-interp metrics are insufficient to prove meaningful features were learned." This caveat is crucial to avoid misinterpreting the work as purely "anti-SAE."
- **Mechanism uncertainty**: The toy model proves random networks *can* preserve or amplify superposition, but which dominates in real Transformers remains for future work.
- **Entropy as proof-of-concept**: The authors state this is not a direct measure of "abstractness" but an indicative proxy, far from a robust "computational significance" metric.

## Related Work & Insights
- **vs. Bricken et al. (2023)**: They found auto-interp scores could distinguish trained vs. random on single-layer Transformers. This paper replicates that finding for small models (Pythia-70m) but shows the **gap closes rapidly as scale increases** (Pythia-6.9b).
- **vs. Karvonen et al. (2024c) (Chess SAEs)**: They found SAEs on pre-trained chess models significantly outperformed random controls. This paper notes that language data is different; language sparsity "aligns" with conceptual semantics in a way chess may not, so random baseline performance varies by domain.
- **vs. Zhong & Andreas (2024)**: They showed that training only embeddings allows algorithmic behaviors to emerge in random Transformers. This work takes the opposite path—randomizing even embeddings or freezing them without training to isolate the training signal completely.
- **vs. Adebayo et al. (2020)**: This paper is a direct extension of their saliency-map sanity check philosophy to SAE evaluations.

## Rating
- Novelty: ⭐⭐⭐⭐ Porting a mature sanity-check paradigm to SAEs is timely and sharp, though the mechanism is not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 70M–6.9B scale, five variants, multiple metrics, and toy model evidence.
- Writing Quality: ⭐⭐⭐⭐ Clear argumentation and honest caveats.
- Value: ⭐⭐⭐⭐⭐ Directly challenges the community's reliance on auto-interp scores and provides actionable baseline recommendations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Formal Mechanistic Interpretability: Automated Circuit Discovery with Provable Guarantees](formal_mechanistic_interpretability_automated_circuit_discovery_with_provable_gu.md)
- [\[ICLR 2026\] Sparse Autoencoders Trained on the Same Data Learn Different Features](sparse_autoencoders_trained_on_the_same_data_learn_different_features.md)
- [\[ICLR 2026\] How Do Transformers Learn to Associate Tokens: Gradient Leading Terms Bring Mechanistic Understanding](how_do_transformers_learn_to_associate_tokens_gradient_leading_terms_bring_mecha.md)
- [\[ICLR 2026\] Learning Pseudorandom Numbers with Transformers: Permuted Congruential Generators, Curricula, and Interpretability](learning_pseudorandom_numbers_with_transformers_permuted_congruential_generators.md)
- [\[ACL 2025\] Enhancing Automated Interpretability with Output-Centric Feature Descriptions](../../ACL2025/interpretability/output_centric_interpretability.md)

</div>

<!-- RELATED:END -->

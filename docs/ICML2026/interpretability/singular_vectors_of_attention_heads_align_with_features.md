---
title: >-
  [Paper Note] Singular Vectors of Attention Heads Align with Features
description: >-
  [ICML 2026][Interpretability][Attention Heads] This paper demonstrates, through both theoretical derivation and toy models, "why and when" the singular vectors of the attention head QK matrix $\Omega = W_Q^\top W_K$ alig…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "Attention Heads"
  - "SVD"
  - "Feature Alignment"
  - "Sparse Attention Decomposition"
  - "Linear Representation Hypothesis"
date: 2026-05-08
content_hash: 875a8852a7f5d27e
---

# Singular Vectors of Attention Heads Align with Features

**Conference**: ICML 2026  
**arXiv**: [2602.13524](https://arxiv.org/abs/2602.13524)  
**Code**: https://github.com/gaabrielfranco/svf-alignment (Available)  
**Area**: Mechanistic Interpretability  
**Keywords**: Attention Heads, SVD, Feature Alignment, Sparse Attention Decomposition, Linear Representation Hypothesis

## TL;DR
This paper demonstrates, through both theoretical derivation and toy models, "why and when" the singular vectors of the attention head QK matrix $\Omega = W_Q^\top W_K$ align with the actual feature directions used by the model. It further proposes "Sparse Attention Decomposition (SAD)" as a verifiable observational signal for this alignment in real-world models (GPT-2 / Pythia).

## Background & Motivation

**Background**: A core task of mechanistic interpretability is to identify the internal representations of "concepts" in language models. The prevailing Linear Representation Hypothesis (LRH) posits that concepts are additively superimposed into activations as one-dimensional or low-dimensional subspace directions. Recent works (Merullo 2024, Ahmad 2025, Pan 2024, Franco & Crovella 2024/2025) have empirically found that the singular vectors of attention head QK matrices often correspond to these feature directions.

**Limitations of Prior Work**: Although the "singular vector = feature" phenomenon has been repeatedly observed, a theoretical explanation has been lacking. It remains unclear why this occurs or under what conditions it holds. Furthermore, mainstream feature discovery methods have drawbacks: linear probes only show that information is decodable but not that the model actually uses that direction; SAEs are expensive to train and only consider activations while ignoring weights; and circuit analysis relies on manually selected directions.

**Key Challenge**: The LRH suggests that activations are sums of features but does not specify how to decompose them. Conversely, SVD provides a natural orthogonal basis, yet whether this basis coincides with the "features actually used by the model" remains an empirical observation rather than a mathematical theorem.

**Goal**: This work formalizes the empirical observation by addressing three sequential questions: (1) Is singular vector-feature (SVF) alignment robust and reproducible in toy models? (2) Can this phenomenon be derived from optimization objectives, and what are the necessary conditions? (3) Is there a verifiable observational prediction to confirm alignment in real models where features cannot be directly observed?

**Key Insight**: The authors adopt the toy autoencoder from Elhage 2022 (which learns a set of features $\{w_i\}$ to reconstruct the input as $W f$) and overlay it with a real attention head $\Omega = W_Q^\top W_K$. Both the features $W$ and the attention weights $\Omega$ are jointly trained under the same loss. In this setup, both features and singular vectors are "observable ground truths," allowing for direct measurement of their cosine similarity.

**Core Idea**: "Alignment" is not a coincidence but a joint solution to the attention training target and the reconstruction loss. The attention loss pulls singular vectors toward "feature pairs of interest," while the reconstruction loss pushes irrelevant features into orthogonal directions. Consequently, the top singular vectors of $\Omega^\star$ are naturally "occupied" by features, leaving the remaining space for noise.

## Method

The paper is structured as a closed loop of "experiment $\rightarrow$ theory $\rightarrow$ real-world prediction." It first replicates the phenomenon in toy models, then provides formal conditions for alignment via three theorems, and finally derives Sparse Attention Decomposition (SAD) as a testable prediction for GPT-2 / Pythia.

### Overall Architecture

The input consists of a set of semantically discrete features $\{w_i \in \mathbb{R}^D\}_{i=1}^N$, each activated in a Bernoulli-uniform manner to form a token $r = W f$. The model comprises two parts: (a) a toy autoencoder that reconstructs $f$ as $f' = \mathrm{ReLU}(W^\top r + b)$ with $\mathcal{L}_{\text{recon}} = \|f - f'\|_2^2$; (b) a single attention head that computes logits $\ell_j = r^\top \Omega s_j$ for a query token $r$ and a set of keys $S = \{s_j\}$, outputting $p_{\text{head}}$ via softmax. The attention target is specified by a "feature-feature" template $T$: the target logit is $\ell^T(r,s) = \sum_{ij} T_{ij} f_i^{(r)} f_j^{(s)}$, and the attention loss is $\mathcal{L}_{\text{attn}} = \mathrm{CE}(p_{\text{head}}, p_{\text{target}})$. The total loss is $\mathcal{L} = \mathcal{L}_{\text{recon}} + \lambda \mathcal{L}_{\text{attn}}$.

Since $W$ and $\Omega$ are learned by the model, the authors perform SVD $\Omega = U \Sigma V^\top$ post-training and use a cosine similarity matrix to directly align feature columns $w_i$ with singular vector columns $u_k, v_k$, obtaining "ground truth labels" for alignment.

### Key Designs

1.  **SVF Alignment and Orthogonalization Coupling in Toy Models**:
    *   **Function**: Replicate the "singular vector = feature" phenomenon in a minimal setting where both are observable, revealing the causal relationship with feature geometric rearrangement.
    *   **Mechanism**: In a single feature-pair scenario ($T_{01} = 1$, others 0), $\Omega^\star$ exhibits only one significant singular value, with cosine similarities between $w_0, u_0$ and $w_1, v_0$ near 1. In multi-feature-pair scenarios, features occupy the top singular vectors in order of "importance." A secondary effect is observed: features relevant to attention become orthogonal to "irrelevant features," which shrink into a $D-2$ dimensional subspace. These effects are coupled during training: singular vectors adjust first to reduce attention loss, followed by feature orthogonalization to reduce reconstruction loss.
    *   **Design Motivation**: To decouple "alignment" and "orthogonalization" into two cooperating optimization pressures and provide evidence of training dynamics rather than just final states.

2.  **Three Theorems: Formal Conditions for Alignment**:
    *   **Function**: Elevate observations to theorems by providing analytical forms of the singular vectors of $\Omega^\star$ upon convergence.
    *   **Mechanism**: Let $X, Y$ be the feature matrices for query/key sides, with Gram matrices $\Sigma_X = XX^\top, \Sigma_Y = YY^\top$. **Theorem 1**: If the target logit satisfies $\ell^T(r,s) = 1$ iff $x_1, y_1$ co-occur, then $\Omega^\star$ is rank-1, and its left/right singular vectors are $u_1 \propto \Sigma_X^{-1} x_1$ and $v_1 \propto \Sigma_Y^{-1} y_1$—i.e., "covariance-whitened feature directions." **Corollary 1**: If features are isotropic ($XX^\top \propto I$), then $u_1, v_1$ exactly equal $x_1, y_1$. **Theorem 2**: Alignment holds approximately even with anisotropy, provided feature interference $\|E_X\|_2$ is bounded. **Theorem 3**: For a fixed $\Omega$, the optimal reconstruction loss automatically pushes features toward orthogonality.
    *   **Design Motivation**: To move from "phenomenon" to "mechanism." The theorems cover isotropic, anisotropic, and evolving feature scenarios, formalizing the emergence of alignment based on feature geometry.

3.  **Sparse Attention Decomposition (SAD) as a Testable Prediction**:
    *   **Function**: Translate "singular vector = feature" into an observable signal in real models where ground-truth features are inaccessible: attention logits should be sparse when decomposed onto the SVD basis.
    *   **Mechanism**: Decompose the logit as $\ell(r,s) = \sum_k r^\top u_k \sigma_k v_k^\top s$. Substituting $r = W f^{(r)}, s = W f^{(s)}$ yields $\ell(r,s) = \sum_k \sum_{i,j} f_i^{(r)} (w_i^\top u_k) \sigma_k (v_k^\top w_j) f_j^{(s)}$. Under the alignment hypothesis, this term is significant only when $w_i, w_j$ align with the same $k$, making the outer sum over $k$ sparse. To remove softmax bias, "Relative Attention" $\tilde{\ell}_j = \ell_j - \frac{1}{m-1} \sum_{i \neq j} \ell_i$ is introduced, quantified by Rolls-Tovee sparseness $S(v)$. $N_{\text{recon}}(j)$ is defined as the minimum number of singular vectors required to reconstruct the relative attention.
    *   **Design Motivation**: Theorem correctness cannot be directly falsified in GPT-2, but SAD can. If SAD occurs and disappears upon random rotation of $U, V$, it strongly supports that alignment is driven by specific feature-singular vector correspondences.

### Loss & Training

The total loss is $\mathcal{L} = \mathcal{L}_{\text{recon}} + \lambda \mathcal{L}_{\text{attn}}$. Appendix A provides a full sweep over $\lambda$, feature count $N$, context length $m$, head dimension $H$, and seeds, showing SVF alignment is robust to these hyperparameters. Toy model configurations typically use $N=20, D=10, H=10$ or $N=100, D=H=50$. Real-world experiments use 130 checkpoints of Pythia-160M and 128 IOI task prompt variants for GPT-2.

## Key Experimental Results

### Main Results

| Experiment | Model | Key Observation | Value/Description |
| :--- | :--- | :--- | :--- |
| Single Alignment (Fig 2a) | Toy, $N=20, H=10$ | Cosine similarity $w_0 \leftrightarrow u_0, w_1 \leftrightarrow v_0$ | Near 1.0; $\Omega^\star$ has only 1 significant singular value |
| Multi-Alignment (Fig 2b) | Toy, $N=100, H=50$ | 20 feature pairs align with top-20 singular vectors | Singular value magnitudes $\approx$ linear target logits |
| Anisotropy Robustness (Fig 4) | Toy, anisotropy swept to GPT-2 levels | Average Cosine Similarity | $> 0.75$, even as $\|E_X\|_2$ approaches GPT-2 limits |
| SAD in Pythia (Fig 7b) | Pythia-160M, IOI head | $S(v)$ change during training | Significant decrease; no decrease with random $U,V$ rotation |
| $N_{\text{recon}}$ on GPT-2 (Fig 9a) | GPT-2, 128 IOI prompts | Vectors needed to reconstruct relative attention | Majority of heads fall between 1–4 |

### Ablation Study

| Configuration | Key Metric | Description |
| :--- | :--- | :--- |
| Full toy model | Cosine Similarity $\approx 1$ | Alignment is robust under default settings |
| No attention head (Fig 1a) | Isotropic feature arrangement | Replicates Elhage 2022; no alignment target |
| Head + Single pair (Fig 1b) | Orthogonalization of $w_0, w_1$ | Validates Theorem 3's orthogonalization pressure |
| Pythia: Random $U,V$ rotation | $S(v)$ no longer decreases | Rules out "few large singular values" hypothesis |
| RoPE (Appendix D) | Alignment still holds | Observed in both position-invariant/dependent logits |

### Key Findings
*   **Alignment results from optimization coupling, not SVD artifacts**: Random rotation experiments prove that shuffling singular vectors destroys relative attention sparsity, indicating that sparsity stems from specific feature-singular vector pairings.
*   **Major contributions often come from small singular values**: Figure 8 shows that the primary terms in relative attention often originate from the bottom of the spectrum, implying semantic importance $\neq$ singular value magnitude.
*   **Attention heads use 1–4 singular vectors in real models**: The distribution of $N_{\text{recon}}$ in GPT-2 and Pythia suggests that attention heads utilize very low-dimensional feature subspaces, making the SVD basis a viable candidate feature space.
*   **Alignment is tolerant of anisotropy**: Even at the upper bound of anisotropy found in GPT-2, alignment cosine similarity remains $> 0.75$, proving the phenomenon is not restricted to "clean" toy models.

## Highlights & Insights
*   **Formalizing Empirical Alignment**: By providing formal conditions and the independent SAD verification method, the authors build a complete evidence chain beyond mere empirical observation.
*   **Covariance Whitening as a Practical Path**: Theorem 1 suggests that singular vectors aligned with features are "unwhitened" forms ($\Sigma_X^{-1} u$). This provides a clear operator for future work: estimating feature covariance to improve alignment results.
*   **Relative Logits + Sparseness as a Portable Probe**: $\tilde{\ell}_j$ and $S(v)$ are architecture-agnostic probes for diagnosing whether an attention head performs single/few-feature matching.
*   **Alternative Methodology to SAE**: If SVF alignment holds, candidate features can be "read out" from the SVD basis in a single forward pass, potentially bypassing expensive SAE training.

## Limitations & Future Work
*   The authors acknowledge that they do not directly prove SVD directions are "causal features" in real models, instead citing prior work; independent causal experiments remain necessary.
*   "Cone directions" (anomalous over-represented directions) in real models may cause alignment with non-semantic directions (Appendix C).
*   Analyses are restricted to "feature count $\leq$ head dimension $H$." For cases where features exceed capacity, preliminary results suggest sharing of singular vectors, but a full theory is missing.
*   The study focuses on single heads; multi-head coordination and cross-head superposition were not addressed.

## Related Work & Insights
*   **vs. Merullo 2024 / Ahmad 2025 / Pan 2024**: These works observe SVF alignment empirically; this paper provides (a) formal proofs, (b) mechanism explanations via coupling, and (c) independent verification (SAD).
*   **vs. SAE (Bricken 2023, Huben 2024)**: SAEs ignore weights and suffer from feature splitting; this method uses weight $\Omega$ and is analytically transparent.
*   **vs. Linear Probe**: Probes show correlation; SVF + SAD shows directions optimized by the model for specific logit computations, suggesting causal relevance.
*   **vs. Elhage 2022**: Extends the "Toy Models of Superposition" framework by adding real attention heads to analyze the interplay between feature geometry and weight spectra.

## Rating
*   **Novelty**: ⭐⭐⭐⭐ First formal conditions and testable predictions for the widely observed SVF alignment.
*   **Experimental Thoroughness**: ⭐⭐⭐ Extensive toy model sweeps; real-world testing limited to GPT-2 and Pythia-160M.
*   **Writing Quality**: ⭐⭐⭐⭐ Clear structure (Experiment-Theory-Prediction) with robust robustness checks.
*   **Value**: ⭐⭐⭐⭐ Provides a weights-based alternative to SAE-centric interpretability, applicable to circuit and feature ablation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Causal Head Gating: A Framework for Interpreting Roles of Attention Heads in Transformers](../../NeurIPS2025/interpretability/causal_head_gating_a_framework_for_interpreting_roles_of_attention_heads_in_tran.md)
- [\[ICML 2026\] How Few-Shot Examples Add Up: A Causal Decomposition of Function Vectors in In-Context Learning](how_few-shot_examples_add_up_a_causal_decomposition_of_function_vectors_in_in-co.md)
- [\[ACL 2026\] Retrieval Heads are Dynamic](../../ACL2026/interpretability/retrieval_heads_are_dynamic.md)
- [\[ICML 2026\] CorrSteer: Generation-Time LLM Steering via Correlated Sparse Autoencoder Features](corrsteer_generation-time_llm_steering_via_correlated_sparse_autoencoder_feature.md)
- [\[ICML 2026\] Optimal Attention Temperature Improves the Robustness of In-Context Learning under Distribution Shift in High Dimensions](optimal_attention_temperature_improves_the_robustness_of_in-context_learning_und.md)

</div>

<!-- RELATED:END -->

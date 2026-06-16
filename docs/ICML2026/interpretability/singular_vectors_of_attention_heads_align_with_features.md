---
title: >-
  [Paper Note] Singular Vectors of Attention Heads Align with Features
description: >-
  [ICML 2026][Interpretability][SVD] This paper demonstrates, through both theory and toy models, "why and when" the singular vectors of the attention head QK matrix $\Omega = W_Q^\top W_K$ align with the feature directions actually used by the model. It proposes "Sparse Attention Decomposition" (SAD) as an observable signal to verify this alignment in re
tags:
  - ICML 2026
  - Interpretability
  - SVD
date: 2026-05-08
content_hash: 6d7f8db4977e1aad
---
# Singular Vectors of Attention Heads Align with Features

**Conference**: ICML 2026  
**arXiv**: [2602.13524](https://arxiv.org/abs/2602.13524)  
**Code**: https://github.com/gaabrielfranco/svf-alignment (Available)  
**Area**: Mechanistic Interpretability  
**Keywords**: Attention Heads, SVD, Feature Alignment, Sparse Attention Decomposition, Linear Representation Hypothesis

## TL;DR
This paper demonstrates, through both theory and toy models, "why and when" the singular vectors of the attention head QK matrix $\Omega = W_Q^\top W_K$ align with the feature directions actually used by the model. It proposes "Sparse Attention Decomposition" (SAD) as an observable signal to verify this alignment in real-world models (GPT-2 / Pythia).

## Background & Motivation

**Background**: A core task in mechanistic interpretability is to identify the internal representations of "concepts" in language models. The current Linear Representation Hypothesis (LRH) posits that concepts are additively superposed into activations as directions in one-dimensional or low-dimensional subspaces. Recent empirical works (Merullo 2024, Ahmad 2025, Pan 2024, Franco & Crovella 2024/2025) have found that the singular vectors of the attention head QK matrix often correspond to these feature directions.

**Limitations of Prior Work**: Although the "singular vector = feature" phenomenon has been repeatedly observed, it lacks a theoretical explanation—it remains unclear why it occurs or under what conditions it holds. Meanwhile, mainstream feature discovery methods have their own drawbacks: linear probes only show that information is decodable but not that the model actually uses that direction; SAEs are expensive to train and only look at activations while ignoring weights; and circuits analysis relies on manually selected directions.

**Key Challenge**: LRH suggests that activations are sums of features but does not specify how to decompose them. Conversely, SVD provides a natural orthogonal basis, but whether this basis matches the "features actually used by the model" remains an empirical observation rather than a mathematical certainty.

**Goal**: To formalize this empirical observation by answering three progressive questions: (1) Is the alignment between singular vectors and features robustly reproducible in toy models? (2) Can this phenomenon be derived from the optimization objective, and what are its necessary conditions? (3) In real models where features cannot be directly observed, is there an observable prediction to verify that alignment has occurred?

**Key Insight**: The authors employ the toy autoencoder from Elhage 2022 (learning a set of features $\{w_i\}$ to reconstruct inputs as $W f$), overlaid with a real attention head $\Omega = W_Q^\top W_K$. Features $W$ and attention weights $\Omega$ are jointly trained under the same loss. This makes both features and singular vectors "observable ground truths," allowing for direct calculation of their cosine similarity.

**Core Idea**: "Alignment" is not a coincidence but a joint solution to the attention training objective and the reconstruction loss: the attention loss pulls singular vectors toward "features of interest," while the reconstruction loss pushes irrelevant features toward orthogonal directions. Consequently, the top singular vectors of $\Omega^\star$ are naturally "occupied" by features, leaving the remaining space for noise.

## Method

The structure of the paper forms a closed loop of "experiment → theory → real model prediction": the phenomenon is first replicated in toy models, followed by three theorems providing formal conditions for alignment, and finally leading to "Sparse Attention Decomposition (SAD)," a verifiable prediction for GPT-2 / Pythia.

### Overall Architecture

The input consists of a set of semantically discrete features $\{w_i \in \mathbb{R}^D\}_{i=1}^N$, where each feature is activated in a Bernoulli-uniform manner to form a token $r = W f$. The model comprises two components: (a) a toy autoencoder that reconstructs $f$ as $f' = \mathrm{ReLU}(W^\top r + b)$ with $\mathcal{L}_{\text{recon}} = \|f - f'\|_2^2$; and (b) a single attention head that computes logits $\ell_j = r^\top \Omega s_j$ for query token $r$ and key set $S = \{s_j\}$, outputting $p_{\text{head}}$ via softmax. The attention target is specified by a "feature-feature" template $T$: the target logit is $\ell^T(r,s) = \sum_{ij} T_{ij} f_i^{(r)} f_j^{(s)}$, and the attention loss is $\mathcal{L}_{\text{attn}} = \mathrm{CE}(p_{\text{head}}, p_{\text{target}})$. The total loss is $\mathcal{L} = \mathcal{L}_{\text{recon}} + \lambda \mathcal{L}_{\text{attn}}$.

Since $W$ and $\Omega$ are learned by the model, the authors perform SVD $\Omega = U \Sigma V^\top$ post-training and use a cosine similarity matrix to directly align feature columns $w_i$ with singular vector columns $u_k, v_k$, obtaining "ground truth labels" for alignment.

### Key Designs

**1. Alignment-Orthogonalization Coupling on Toy Models: Decomposition into Two Optimization Pressures**

To reproduce "singular vector = feature" in a minimal setup where both are observable, the authors designed two templates. In the single feature-pair case ($T_{01} = 1$, others 0), only one significant singular value remains in the spectrum of $\Omega^\star$, and the cosine similarity between $w_0, u_0$ and $w_1, v_0$ approaches 1. In the multi-feature-pair case (linear decay of $T_{i,i+20}$), multiple features occupy the top singular vectors in order of "importance." A key observation is the secondary phenomenon: features relevant to attention become orthogonal to "irrelevant features," which are compressed into a $(D-2)$-dimensional subspace. These effects are coupled during training—singular vectors move first to reduce attention loss, followed by feature orthogonalization to reduce reconstruction loss. This coupling provides anchors for theorems and reveals training dynamics (time-resolved evolution in Fig. 3) as mechanistic evidence.

**2. Three Theorems: Analytical Conditions for Alignment**

To transition from observations to provable conclusions, the authors derive the analytical form of singular vectors after $\Omega^\star$ converges. Let $X, Y$ be the feature matrices for query/key sides, with Gram matrices $\Sigma_X = XX^\top$ and $\Sigma_Y = YY^\top$. Theorem 1 provides the main result: if the target logit satisfies $\ell^T(r,s) = 1$ iff $x_1, y_1$ co-occur, the converged $\Omega^\star$ is rank-1, with left/right singular vectors $u_1 \propto \Sigma_X^{-1} x_1$ and $v_1 \propto \Sigma_Y^{-1} y_1$ (i.e., "covariance-whitened feature directions"). Corollary 1 is a clean special case: when features are isotropic ($XX^\top \propto I$), $u_1, v_1$ are exactly equal to $x_1, y_1$. Theorem 2 extends this to reality: even with anisotropy, alignment holds approximately as long as feature interference $\|E_X\|_2$ is bounded. Theorem 3 addresses orthogonalization: when $\Omega$ is fixed, the optimal solution for reconstruction loss automatically pushes features to be orthogonal, explaining "irrelevant feature orthogonalization." These theorems cover isotropy, anisotropy, and feature evolution, turning alignment into a precise characterization of feature geometry. The authors also use SAE dictionary elements from GPT-2 as proxy features, quantifying $\|E_X\|_2$ between 10–55, verifying that Theorem 2's conditions are met in real models.

**3. Sparse Attention Decomposition (SAD): Testable Predictions in Real Models**

Since feature ground truths are unavailable in GPT-2, the authors translate "singular vector = feature" into an observable signal: attention logits should be sparse when decomposed onto the SVD basis. Writing the logit as $\ell(r,s) = \sum_k r^\top u_k \sigma_k v_k^\top s$ and substituting $r = W f^{(r)}, s = W f^{(s)}$ yields $\ell(r,s) = \sum_k \sum_{i,j} f_i^{(r)} (w_i^\top u_k) \sigma_k (v_k^\top w_j) f_j^{(s)}$. Under the alignment hypothesis, this term is significant only when $w_i, w_j$ align with the same $k$, making the outer sum over $k$ sparse. To remove softmax bias, "relative attention" $\tilde{\ell}_j = \ell_j - \frac{1}{m-1} \sum_{i \neq j} \ell_i$ is introduced, quantified by Rolls-Tovee sparsity $S(v) = (\frac{1}{n} \sum_i |v_i|)^2 / (\frac{1}{n} \sum_i v_i^2)$. They also define $N_{\text{recon}}(j)$ as the "minimum number of singular vectors required to reconstruct relative attention." This metric turns unmaskable theory into falsifiable predictions: if SAD appears in real models and vanishes upon random rotation of $U, V$, it strongly supports that alignment stems from specific feature-singular vector pairings rather than just a few large singular values.

### Loss & Training

Total loss $\mathcal{L} = \mathcal{L}_{\text{recon}} + \lambda \mathcal{L}_{\text{attn}}$. The authors performed a complete sweep of $\lambda$, feature count $N$, context length $m$, head dimension $H$, and random seeds (Appendix A), proving SVF alignment is robust to these hyperparameters. Typical toy configurations: $N = 20, D = 10, H = 10$ (single pair) or $N = 100, D = H = 50$ (multi-pair). Real model analysis uses 130 checkpoints of Pythia-160M and IOI task prompts for GPT-2 (128 variants).

## Key Experimental Results

### Main Results

| Experiment | Model | Key Observation | Value/Description |
|------|------|---------|---------|
| Single Alignment (Fig 2a) | Toy, $N=20, H=10$ | Cosine similarity $w_0 \leftrightarrow u_0$, $w_1 \leftrightarrow v_0$ | Near 1.0; $\Omega^\star$ has 1 significant singular value |
| Multi-Alignment (Fig 2b) | Toy, $N=100, H=50$ | 20 feature pairs align with top-20 singular vectors | Singular value magnitude ≈ linear target logit |
| Anisotropy Robustness (Fig 4) | Toy, anisotropy swept to GPT-2 range | Mean cosine similarity | > 0.75 even as $\|E_X\|_2$ approaches GPT-2 limits |
| SAD in Pythia (Fig 7b) | Pythia-160M, IOI Head | $S(v)$ change during training | Drops significantly; no drop after random rotation of $U,V$ |
| $N_{\text{recon}}$ on GPT-2 (Fig 9a) | GPT-2, 128 IOI prompts | Vectors needed to reconstruct relative attention | Most heads fall between 1–4 |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Full toy model | Cosine similarity ≈ 1 | Alignment is robust under default settings |
| No attention head (Fig 1a) | Isotropic feature arrangement | Replicates Elhage 2022; no alignment objective |
| Head + single pair (Fig 1b) | $w_0, w_1$ orthogonalize vs others | Validates orthogonalization pressure in Theorem 3 |
| Pythia: Random rotation of SVD basis (Fig 7b/8 bottom) | $S(v)$ no longer decreases | Rules out the "few large singular values" hypothesis |
| RoPE (Appendix D) | Alignment still holds | Observed in both position-independent and dependent logits |

### Key Findings
- **Alignment is a result of optimization coupling, not an SVD artifact**: Random rotation of $U, V$ shows that relative attention sparsity vanishes when singular vectors are shuffled. Sparsity arises from specific feature-singular vector pairings, not the matrix spectrum itself.
- **Major contributions often come from small singular values**: Fig. 8 middle shows that the main terms of relative attention often originate from the bottom of the spectrum. Semantic importance $\neq$ singular value magnitude; looking only at top-$k$ SVD is insufficient.
- **Real models use 1–4 singular vectors to explain one head's attention**: The distribution of $N_{\text{recon}}$ in GPT-2/Pythia suggests attention heads operate in very low-dimensional subspaces, making SVD bases viable candidate feature spaces.
- **Alignment is tolerant of anisotropy**: Even at the actual anisotropy limits of GPT-2, alignment similarity remains $> 0.75$, suggesting the method is not a fragile phenomenon restricted to "clean" toy models.

## Highlights & Insights
- **Translating "empirical alignment" into "provable + measurable" components**: Theorems provide formal conditions, while SAD provides an independent verification method in real models, creating a complete chain of evidence.
- **Covariance whitening specifies a path for practical improvement**: Theorem 1 indicates that the directions aligned with features are "unwhitened" singular vectors of the form $\Sigma_X^{-1} u, \Sigma_Y^{-1} v$. This identifies a clear operator: estimating feature covariance and unwhitening can significantly improve alignment.
- **"Relative logit + sparsity metric" as a transferable probe**: $\tilde{\ell}_j$ and $S(v)$ are largely architecture-agnostic and can diagnose whether any attention head is performing single/sparse feature matching, offering direct value for circuit discovery and feature-level ablation.
- **Methodological alternative to SAEs**: If SVF alignment holds, candidate features can be "read out" from the SVD basis in a single forward pass, removing the need for expensive SAE training. This serves as a clear contrast to the current SAE-centric paradigm in interpretability.

## Limitations & Future Work
- The authors admit they have not directly verified that "SVD-derived directions are features in a causal sense" in real models, instead citing prior work (Franco & Crovella 2025) for causal evidence; independent use of this method may still require causal experiments.
- Real models contain "cone directions" (abnormally over-represented directions) that may cause singular vectors to align with them rather than semantic features; Appendix C provides only a preliminary study.
- Analysis is limited to "feature count $\leq$ head dimension $H$." When features exceed head capacity, Appendix E suggests the least important features share singular vector pairs, but a full theory is lacking.
- Only single heads were analyzed. Multi-head coordination (how singular vectors are distributed across heads or if cross-head superposition exists) remains unexplored.
- Experiments are restricted to relatively small models (GPT-2 / Pythia-160M); the extent of SVF alignment in modern 7B+ models requires further empirical validation.

## Related Work & Insights
- **vs Merullo 2024 / Ahmad 2025 / Pan 2024 / Franco & Crovella 2024-2025**: These works empirically observed SVF alignment and built tools upon it. This paper moves beyond empirical observation to provide (a) rigorous provable conditions, (b) a mechanistic explanation coupled with reconstruction loss, and (c) verifiable predictions (SAD) for real models.
- **vs SAE works (Bricken 2023, Huben 2024)**: SAEs use only activations, are expensive to train, and suffer from feature splitting/absorption. This method uses SVD bases of $\Omega$ for analytical, cheap decomposition that explicitly utilizes weight information.
- **vs Linear Probe**: Probes provide correlational evidence and do not guarantee the model uses the direction. SVF alignment + SAD reveals directions optimized by the model and directly related to attention logits, closer to "causal usage."
- **vs Elhage 2022 (Toy Models of Superposition)**: This work extends the toy autoencoder by adding a real attention head and analyzing the joint loss of "feature geometry" and "attention weight spectra," explaining phenomena unattainable in the original toy model.

## Rating
- Novelty: ⭐⭐⭐⭐ First to provide provable conditions and measurable predictions for widely observed SVF alignment; clear mechanism.
- Experimental Thoroughness: ⭐⭐⭐ Toy models are comprehensive, but real-model analysis is limited to GPT-2 and Pythia-160M; lacks larger models and diverse tasks.
- Writing Quality: ⭐⭐⭐⭐ Closed loop of experiment-theory-prediction; clear theorems; robustness thoroughly explored in the appendix.
- Value: ⭐⭐⭐⭐ Provides an engineerable alternative to SAEs for interpretability; ideas are directly transferable to circuit and feature ablation work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] CIGMA: Causal Information-Gain Mechanistic Attribution of Attention Heads in Vision Transformers](../../CVPR2026/interpretability/cigma_causal_information-gain_mechanistic_attribution_of_attention_heads_in_visi.md)
- [\[NeurIPS 2025\] Causal Head Gating: A Framework for Interpreting Roles of Attention Heads in Transformers](../../NeurIPS2025/interpretability/causal_head_gating_a_framework_for_interpreting_roles_of_attention_heads_in_tran.md)
- [\[ACL 2026\] Retrieval Heads are Dynamic](../../ACL2026/interpretability/retrieval_heads_are_dynamic.md)
- [\[ICML 2026\] How Few-Shot Examples Add Up: A Causal Decomposition of Function Vectors in In-Context Learning](how_few-shot_examples_add_up_a_causal_decomposition_of_function_vectors_in_in-co.md)
- [\[ICML 2026\] CorrSteer: Generation-Time LLM Steering via Correlated Sparse Autoencoder Features](corrsteer_generation-time_llm_steering_via_correlated_sparse_autoencoder_feature.md)

</div>

<!-- RELATED:END -->

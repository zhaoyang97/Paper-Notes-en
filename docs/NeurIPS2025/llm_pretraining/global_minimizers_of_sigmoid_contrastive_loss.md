---
title: >-
  [Paper Note] Global Minimizers of Sigmoid Contrastive Loss
description: >-
  [NeurIPS 2025][LLM Pretraining][Contrastive Learning] This work provides the first rigorous characterization of the global minimizer geometry of the Sigmoid contrastive loss (SigLIP) with trainable temperature and bias i…
tags:
  - "NeurIPS 2025"
  - "LLM Pretraining"
  - "Contrastive Learning"
  - "Sigmoid Loss"
  - "SigLIP"
  - "Representation Synchronization"
  - "Modality Gap"
date: 2026-05-08
content_hash: 5b865cd792e52aa2
---

# Global Minimizers of Sigmoid Contrastive Loss

**Conference**: NeurIPS 2025
**arXiv**: [2509.18552](https://arxiv.org/abs/2509.18552)  
**Code**: [RepresentationLearningTheory/SigLIP](https://github.com/RepresentationLearningTheory/SigLIP)  
**Area**: LLM Pretraining
**Keywords**: Contrastive Learning, Sigmoid Loss, SigLIP, Representation Synchronization, Modality Gap

## TL;DR
This work provides the first rigorous characterization of the global minimizer geometry of the Sigmoid contrastive loss (SigLIP) with trainable temperature and bias in the practically relevant regime $N \gg d$. It introduces a novel combinatorial object called the $(m, b_\text{rel})$-Constellation, and uses it to explain retrieval success, the modality gap phenomenon, and to propose an explicit relative bias parameterization that improves training dynamics.

## Background & Motivation
Learning and aligning representations via contrastive pretraining (e.g., CLIP, ALIGN, SigLIP) is a central paradigm in multimodal learning. The task involves training encoders $f_\theta$ and $g_\phi$ such that embeddings of matched image-text pairs are similar, while those of unmatched pairs are dissimilar.

Despite the widespread use of contrastive learning, significant theoretical gaps remain regarding loss function selection, hyperparameter settings, and properties of optimal embeddings:

**Dimensional regime mismatch with practice**: Existing theoretical work assumes either $d \geq N$ (embedding dimension $\geq$ data size) or $N \to \infty$ with fixed $d$. In practice, SigLIP2 uses $d \approx 10^3$ dimensional embeddings on datasets of size $N \approx 10^{10}$, placing it squarely in the regime $d \ll N \ll 2^d$, which is entirely uncharacterized.

**Known optimal configurations are overly rigid**: In the $d \geq N$ regime, the optimal solution has a simplex structure with perfect alignment $U_i = V_i$, which fails to explain minimizing configurations when one modality is frozen, or the modality gap phenomenon.

**Modality gap lacks theoretical explanation**: Image and text embeddings in CLIP/SigLIP are empirically non-overlapping and linearly separable, yet no theoretical explanation has previously been given.

This paper analyzes the **sigmoid loss with trainable inverse temperature $t$ and bias $b$** as used in Google's SigLIP/SigLIP2 models.

## Method

### Overall Architecture
The Sigmoid loss is analyzed as:

$$\mathcal{L}^{Sig}(\theta, \phi; t, b) = \sum_{i=1}^{N} \log(1+\exp(-t\langle U_i, V_i \rangle + b)) + \sum_{i \neq j} \log(1+\exp(t\langle U_i, V_j \rangle - b))$$

where the first term encourages matched pairs to be similar and the second discourages unmatched pairs from being similar. The key innovation is treating $t$ (inverse temperature) and $b$ (bias) as **trainable parameters**.

### Key Designs

#### 1. Definition and Characterization of the $(m, b_\text{rel})$-Constellation

A novel combinatorial object is defined — the $(m, b_\text{rel})$-Constellation: a set of embeddings $\{(U_i, V_i)\}_{i=1}^N \in S^{d-1}$ satisfying:
- Matched pairs: $\langle U_i, V_i \rangle \geq m + b_\text{rel}$ ($\forall i$)
- Unmatched pairs: $\langle U_i, V_j \rangle \leq -m + b_\text{rel}$ ($\forall i \neq j$)

Here $m$ (margin) measures the gap between matched/unmatched inner products, and $b_\text{rel} = b/t$ (relative bias) is the ratio of bias to temperature.

**Core theorem pair**:
- **Theorem 3.1**: Any optimization sequence driving the Sigmoid loss to zero must converge to an $(m, b_\text{rel})$-Constellation in the limit.
- **Theorem 3.2**: Any $(m, b_\text{rel})$-Constellation with $m > 0$ is a global minimizer, and the optimal margin $m^*$ governs the rate of loss convergence to zero: $\inf_b \mathcal{L}^{Sig} = \exp(-t \cdot m^* + o(t))$.

The equivalent condition is remarkably concise: **inner product separability**, i.e., $\min_i \langle U_i, V_i \rangle \geq \max_{i \neq j} \langle U_i, V_j \rangle$, is both necessary and sufficient for zero loss.

#### 2. Capacity Bounds for Constellations

By connecting to spherical codes, the maximum $N$ accommodated in dimension $d$ is characterized:

**Theorem 3.3 (Lower Bound)**: When $m + b_\text{rel} < 1$ and $3m < 1 + b_\text{rel}$, exponentially large Constellations exist:
$$E_{MRB}(m, b_{rel}) \geq -\frac{1}{2}\log_2\!\left(1-\left(\frac{1+b_{rel}-3m}{1+b_{rel}+m}\right)^2\right)$$

**Theorem 3.4 (Upper Bound / Necessary Condition)**: $m + b_\text{rel} \leq 1$ and $3m \leq 1 + b_\text{rel}$ are necessary conditions.

**Theorem 3.5**: Provides an upper bound that is exponentially tight with the lower bound.

#### 3. Theoretical Proof of the Modality Gap

**Theorem 3.6**: When $N \geq d + 2$ and $m > |b_\text{rel}|$, in any zero-loss configuration, the image and text embeddings are linearly separable by a hyperplane. Specifically, there exists $h \in S^{d-1}$ such that $\langle h, U_i \rangle > 0$ ($\forall i$) and $\langle h, V_j \rangle < 0$ (for at least $N - d$ indices $j$).

The proof leverages Helly's theorem, the hyperplane separation theorem, and Carathéodory's theorem. In practice, with $N \approx 10^{10}$ and $d \approx 10^3$, the separation condition holds for all but $0.0000001\%$ of text embeddings.

This is also philosophically sound: "different modalities may carry different information," making it natural for them to occupy disjoint regions of the embedding space.

#### 4. Relative Bias Parameterization

An explicit relative bias parameterization of the Sigmoid loss is proposed:

$$\mathcal{L}^{RB\text{-}Sig}(\theta, \phi; t, b_{rel}) = \sum_{i} \log(1+\exp(-t\langle U_i, V_i \rangle + t \cdot b_{rel})) + \sum_{i \neq j} \log(1+\exp(t\langle U_i, V_j \rangle - t \cdot b_{rel}))$$

Although mathematically equivalent to $\mathcal{L}^{Sig}(\theta, \phi; t,\ b_\text{rel} \times t)$, this parameterization converges faster under Adam optimization.

### Loss & Training
- The core contribution is **theoretical analysis** rather than a new training method.
- **Observation 1**: Training the relative bias and inverse temperature implicitly corresponds to adding linear adapters to both encoders.
- **Observation 2**: The framework extends naturally to multimodal synchronization ($k > 2$ modalities) via simplex embeddings.
- **Construction 1**: Constellations are constructed from spherical codes, with parameters $\delta$ and $\phi$ controlling margin and relative bias.
- Practical recommendation: use $\mathcal{L}^{RB\text{-}Sig}$ with trainable $t$ and $b_\text{rel}$.

## Key Experimental Results

### Main Results
Theoretical predictions are validated on 8 HuggingFace SigLIP models:

| Model | Mean Positive Pair | Mean Negative Pair | Margin | Relative Bias | Dim |
|---|---|---|---|---|---|
| so400m-patch14-384 | 0.1376 | -0.0015 | 0.0695 | 0.0680 | 1152 |
| so400m-patch14-224 | 0.1365 | -0.0022 | 0.0694 | 0.0672 | 1152 |
| large-patch16-256 | 0.1023 | -0.0359 | 0.0691 | 0.0332 | 1024 |
| base-patch16-256 | 0.1004 | -0.0294 | 0.0649 | 0.0355 | 768 |
| base-patch16-224 | 0.0950 | -0.0305 | 0.0627 | 0.0322 | 768 |

Key findings:
- **Margin correlates perfectly with dimension**: Pearson correlation 0.948, Spearman 0.926; larger models exhibit larger margins.
- **All 8 models satisfy the modality gap**: A perceptron algorithm finds a perfect linear separator in all cases.
- **Two clusters**: Large models (so400m, ~1B parameters) exhibit noticeably different relative bias compared to smaller models (≤0.4B).

### Ablation Study
Synthetic data experiments compare different Sigmoid loss variants:

1. **Fixed $t, b$ vs. trainable $t, b$**: Trainable parameters drive loss to zero; fixed parameters cannot reach zero loss.
2. **$\mathcal{L}^{Sig}$ vs. $\mathcal{L}^{RB\text{-}Sig}$**: The relative bias parameterization converges faster under Adam.
3. **Effect of fixing different $b_\text{rel}$ values**: Larger fixed $b_\text{rel}$ leads to smaller margin, consistent with theoretical bounds.
4. **Frozen encoder scenario**: $\mathcal{L}^{RB\text{-}Sig}$ with trainable $t, b_\text{rel}$ significantly outperforms $\mathcal{L}^{Sig}$ with trainable $t, b$.
5. **Multimodal synchronization ($k=4$)**: Validates the effectiveness of Construction 2.

### Key Findings
1. SigLIP models in practice approximately satisfy the Constellation conditions (after removing 5% outliers).
2. Standard Adam optimization tends to converge to configurations with $b_\text{rel} \approx 0$, potentially limiting solution diversity.
3. Fixing $b_\text{rel}$ can steer optimization toward different zero-loss configurations.
4. Constellations are also global minimizers of the triplet loss.
5. The global minimizer geometry of InfoNCE differs: it is row-wise thresholdable (with an independent $b_\text{rel}(i)$ per row).

## Highlights & Insights
1. **First characterization of global minimizers in the practically relevant $N \gg d$ regime**, filling a critical theoretical gap.
2. The **(m, b_rel)-Constellation** is an elegant geometric abstraction that unifies the minimizer characterization of both Sigmoid loss and triplet loss.
3. **Rigorous proof of the modality gap**: provides a theoretical explanation for empirical observations in CLIP/SigLIP, and distinguishes "synchronization" from "alignment."
4. **Relative bias parameterization** offers a practical improvement: faster convergence, support for frozen encoders, and extensibility to multimodal settings.
5. The connection to spherical codes provides quantitative guidance for embedding dimension selection.
6. Replacing "alignment" with "synchronization" more accurately describes the objective of multimodal representation learning.

## Limitations & Future Work
1. **Experiments are primarily synthetic**: The benefits of relative bias parameterization on large-scale real data (e.g., LAION, WebLI) remain to be verified.
2. **Training dynamics are not analyzed**: The theory characterizes the final configurations, but how optimizers such as Adam converge to specific Constellations is not yet understood.
3. **Gap in spherical code capacity bounds**: Upper and lower bounds do not fully align in certain regimes.
4. **Incomplete practical guidance**: More quantitative guidance is needed on how to select the optimal embedding dimension $d$ given dataset size $N$.
5. **InfoNCE analysis is relatively brief**: The row-wise geometric characterization of InfoNCE could be developed more thoroughly.

## Related Work & Insights
- **SigLIP/SigLIP2** (Google DeepMind): The models directly analyzed in this work; their design choices (trainable $t, b$) are theoretically justified herein.
- **CLIP** (OpenAI): Uses InfoNCE loss; this paper contrasts the distinct geometric structures of the two losses.
- **Modality gap research** (Liang et al., 2022; Fahim et al., 2025): This paper provides a theoretical explanation.
- **Lee et al., 2024**: Prior work in the $d \geq N$ regime; Construction 1 in this paper builds on their Double-Constant Embedding Model.
- Insights: (1) The importance of trainable hyperparameters is far greater than previously recognized; (2) "imperfect alignment" may be a feature rather than a flaw.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (Pioneering theoretical analysis; the Constellation object offers deep insight)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Theory-driven; synthetic + pretrained model validation is solid, but large-scale training experiments are absent)
- Writing Quality: ⭐⭐⭐⭐⭐ (Mathematically rigorous, intuitively illustrated, clearly written)
- Value: ⭐⭐⭐⭐⭐ (Major contribution to the theoretical foundations of contrastive learning; practical recommendations are useful)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Implicit Bias and Loss of Plasticity in Matrix Completion: Depth Promotes Low-Rank](../../ICLR2026/llm_pretraining/implicit_bias_and_loss_of_plasticity_in_matrix_completion_depth_promotes_low-ran.md)
- [\[NeurIPS 2025\] Breaking the Frozen Subspace: Importance Sampling for Low-Rank Optimization in LLM Pretraining](breaking_the_frozen_subspace_importance_sampling_for_low-rank_optimization_in_ll.md)
- [\[NeurIPS 2025\] Vocabulary Customization for Efficient Domain-Specific LLM Deployment](vocabulary_customization_for_efficient_domain-specific_llm_deployment.md)
- [\[NeurIPS 2025\] Nemotron-CLIMB: CLustering-based Iterative Data Mixture Bootstrapping for Language Model Pre-training](nemotron-climb_clustering-based_iterative_data_mixture_bootstrapping_for_languag.md)
- [\[NeurIPS 2025\] Born a Transformer – Always a Transformer? On the Effect of Pretraining on Architectural Abilities](born_a_transformer_--_always_a_transformer_on_the_effect_of_pretraining_on_archi.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] From Flat to Hierarchical: Extracting Sparse Representations with Matching Pursuit
description: >-
  [NeurIPS 2025][Interpretability][Sparse Autoencoders] Proposes MP-SAE, which unrolls the classic Matching Pursuit algorithm into a sequential encoder for SAEs. By achieving conditional orthogonality through residual-guided greedy feature selection, it captures hierarchical structures, non-linearly accessible features, and cross-modal features that standard SAEs fail to detect, while natively supporting adaptive sparsity adjustment at inference time.
tags:
  - "NeurIPS 2025"
  - "Interpretability"
  - "Sparse Autoencoders"
  - "Matching Pursuit"
  - "Hierarchical Representation"
  - "Conditional Orthogonality"
date: 2026-05-08
content_hash: e213e4e100f551fb
---

# From Flat to Hierarchical: Extracting Sparse Representations with Matching Pursuit

**Conference**: NeurIPS 2025  
**arXiv**: [2506.03093](https://arxiv.org/abs/2506.03093)  
**Code**: None  
**Area**: Interpretability  
**Keywords**: Sparse Autoencoders, Matching Pursuit, Hierarchical Representation, Interpretability, Conditional Orthogonality

## TL;DR
Proposes MP-SAE, which unrolls the classic Matching Pursuit algorithm into a sequential encoder for SAEs. By achieving conditional orthogonality through residual-guided greedy feature selection, it captures hierarchical structures, non-linearly accessible features, and cross-modal features that standard SAEs fail to detect, while natively supporting adaptive sparsity adjustment at inference time.

## Background & Motivation

**Background**: Sparse Autoencoders (SAEs), based on the Linear Representation Hypothesis (LRH), have become a mainstream tool in neural network interpretability research. LRH posits that representations can be decomposed into a large number of approximately orthogonal directions, each corresponding to an interpretable concept. SAEs recover these directions by learning an overcomplete sparse dictionary.

**Limitations of Prior Work**: Recent studies have revealed that LRH cannot fully account for real-world representations: (1) hierarchical concepts (e.g., animal $\rightarrow$ mammal $\rightarrow$ cat) span orthogonal subspaces between parent and child concepts rather than being globally approximately orthogonal; (2) "onion-like" non-linear representations cannot be accessed via a single linear projection; (3) multi-dimensional concepts (e.g., days of the week) cannot be represented by a single direction. Standard SAEs (including Matryoshka SAEs with hierarchical objectives) perform poorly on these structures.

**Key Challenge**: The design assumptions of SAEs (global quasi-orthogonality + linear accessibility) mismatch the hierarchical and non-linear representative structures actually present in neural networks.

**Goal**: (1) To verify whether standard SAEs can capture structures beyond the scope of the LRH; (2) To design an SAE architecture whose inductive bias matches hierarchical/non-linear structures.

**Key Insight**: Introducing "conditional orthogonality"—requiring concepts across different hierarchy levels to be orthogonal (parent-child orthogonality) while allowing interference within the same level, which is fundamentally different from the global quasi-orthogonality of standard LRH. The classic Matching Pursuit algorithm naturally possesses step-wise orthogonalization properties, selecting the feature most correlated with the current residual at each step.

**Core Idea**: Unroll the greedy residual decomposition of Matching Pursuit (MP) into an SAE encoder to match its inductive bias with conditional orthogonality and non-linear conceptual structures.

## Method

### Overall Architecture
MP-SAE shares the encoding and decoding dictionary $\bm{D}$. The encoding process unrolls $T$ steps of Matching Pursuit: initializing the residual $\bm{r}^{(0)} = \bm{x} - \bm{b}_{\text{pre}}$, selecting the feature direction most correlated with the residual at each step, subtracting its contribution to update the residual, and repeating for $T$ steps to obtain the sparse code $\bm{z}$ ($\|\bm{z}\|_0 \leq T$) and reconstruction $\hat{\bm{x}}$. The dictionary is jointly learned via backpropagation.

### Key Designs

1. **MP-Unrolled Encoder**:

    - **Function**: Unrolls the greedy inference process of MP into a differentiable encoder.
    - **Mechanism**: At each step $t$: select the direction of maximum projection $j^{(t)} = \arg\max_j (\bm{D}^\top \bm{r}^{(t)})_j$, compute the coefficient $z_{j^{(t)}}^{(t)} = \bm{D}_{j^{(t)}}^\top \bm{r}^{(t)}$, update the reconstruction $\hat{\bm{x}}^{(t+1)} = \hat{\bm{x}}^{(t)} + z_{j^{(t)}}^{(t)} \bm{D}_{j^{(t)}}$, and update the residual $\bm{r}^{(t+1)} = \bm{r}^{(t)} - z_{j^{(t)}}^{(t)} \bm{D}_{j^{(t)}}$.
    - **Design Motivation**: Standard SAEs encode with a single linear projection, failing to perceive dependencies between features; the sequential decomposition of MP naturally allows subsequent features to explain what remains unexplained by preceding ones.

2. **Step-wise Orthogonality Guaranteeing Conditional Orthogonality**:

    - **Function**: Ensures that features selected at each step are orthogonal to the previous step.
    - **Mechanism**: Directly guaranteed by the residual update rule $\bm{D}_{j^{(t-1)}}^\top \bm{r}^{(t)} = 0$—selected features are removed from the residual subspace, and subsequent selection proceeds only in the orthogonal complement space.
    - **Design Motivation**: This precisely corresponds to the definition of conditional orthogonality—cross-level (cross-step) orthogonality, while allowing interference within levels (candidate features in the same step). Although MP only guarantees orthogonality with the most recent step (unlike OMP, which orthogonalizes against all selected features), experiments show that the residual is empirically approximately orthogonal to all previously selected directions.

3. **Accessing Non-Linearly Accessible Features**:

    - **Function**: Achieves non-linear feature extraction from the original input through residual iterations.
    - **Mechanism**: Decompose $\bm{x} = \underbrace{\bm{\varphi}(\bm{x})}_{\text{linear accessibility}} + \underbrace{\sum_{t=1}^T \bm{\varphi}(\bm{r}^{(t)})}_{\text{non-linear accessibility}} + \bm{r}^{(T+1)}$. Although each step $\bm{\varphi}(\cdot)$ is a linear projection, $\bm{r}^{(t)}$ is a non-linear function of $\bm{x}$, so the combination of $\bm{\varphi}(\bm{r}^{(t)})$ forms non-linear features.
    - **Design Motivation**: Provides a constructive explanation for the "dark matter" phenomenon (the parts of representations that standard SAEs fail to explain)—these features are not nonexistent, but rather require non-linear access.

### Loss & Training
Training objective: $\mathcal{L} = \|\bm{x} - \hat{\bm{x}}\|_2^2 + \lambda \mathcal{R}(\bm{z}) + \alpha \mathcal{L}_{\text{aux}}$. Optimized using Adam with a learning rate of $5 \times 10^{-4}$ cosine-decaying to $10^{-6}$ over 50 epochs. Trained using frozen representations of the last layer of backbones on ImageNet-1K, with an expansion factor of $p = 25m$.

## Key Experimental Results

### Main Results (Capacity R² vs. Sparsity Pareto Frontier)

| Model/SAE | SigLIP R²@k=32 | DINOv2 R²@k=32 | CLIP R²@k=32 |
|----------|---------------|----------------|--------------|
| Vanilla (ReLU) | ~0.65 | ~0.55 | ~0.60 |
| BatchTopK | ~0.70 | ~0.60 | ~0.65 |
| MP-SAE | ~0.78 | ~0.70 | ~0.72 |

MP-SAE achieves a higher R² across all tested backbones (SigLIP, DINOv2, CLIP, ViT) under comparable sparsity.

### Synthetic Experiments (Conditional Orthogonality Recovery)

| SAE | Flat MSE↓ | Hierarchical MSE↓ | Description |
|-----|----------|-------------------|-------------|
| Vanilla | Low | High | Preserves intra-level structure but loses hierarchical separation |
| BatchTopK | Low | High | Same as above, affected by feature absorption |
| Matryoshka | High | Low | Preserves hierarchy but introduces negative intra-level interference |
| MP-SAE | **Low** | **Low** | Simultaneously preserves both intra-level and hierarchical structures |

### Key Findings
- **Feature Absorption**: Vanilla and BatchTopK align sub-concept directions with parent concepts, leading to hierarchical collapse—a fundamental flaw of standard SAEs.
- **Trade-off of Matryoshka**: It preserves hierarchical separation but introduces negative interference between sibling concepts, indicating that explicit hierarchical objectives cannot perfectly resolve the issue.
- **Continued Growth of Effective Rank**: As sparsity $k$ increases, the effective rank of the co-activation matrix of MP-SAE continuously grows, whereas standard SAEs saturate quickly—indicating that MP-SAE discovers more diverse feature combinations.
- **Conditional Orthogonality during Inference**: MP-SAE has a higher global Babel score (more interference) for its dictionary, but the actual subset of features selected during inference has a lower Babel score—showing that conditional orthogonality naturally emerges at inference time.
- **Cross-Modal Feature Recovery**: In the joint embedding space of CLIP, features learned by standard SAEs exhibit a bimodal modality score distribution (responding either only to images or only to text), whereas MP-SAE recovers true cross-modal features (with substantial density in the mid-range of modality scores).
- **Adaptive Sparsity at Inference Time**: MP-SAE is the only architecture where reconstruction error monotonically decreases as $k$ varies, whereas TopK SAEs may degrade when $k$ deviates from the training value.

## Highlights & Insights
- **Designing Methods from Phenomenology**: The core thesis of the paper is that "interpretability should start from the phenomenology of representation, and methods should follow assumptions" rather than vice versa—a profound methodological claim.
- **Formalization of Conditional Orthogonality**: Distilling the definition of conditional orthogonality from observations by Park et al., relaxing the global quasi-orthogonality of LRH to cross-level orthogonality + intra-level interference, which stands on both theoretical foundations and practical motivations.
- **Ingenious Use of MP**: Although the classic MP algorithm has existed for decades in sparse coding, repositioning it as an SAE encoder is a clever "old wine in a new bottle" approach, where step-wise orthogonalization naturally matches the demand for conditional orthogonality.
- **Constructive Explanation of Dark Matter**: Decomposing non-linearly accessible features into the form of $\bm{\varphi}(\bm{r}^{(t)})$ provides a mathematical framework for "what standard SAEs fail to explain."
- **Practical Value in Cross-Modal Feature Discovery**: Can be used to verify whether visual and text embeddings in VLMs are truly aligned (rather than just possessing superficial cosine similarity).

## Limitations & Future Work
- MP is a greedy algorithm, lacking guarantees of global optimality, and can be fragile under extreme noise.
- The conditional orthogonality assumption may not apply to flat or highly entangled representation spaces.
- Computational cost grows linearly with the number of steps $T$, which may bottleneck inference speed when $T$ is large.
- Experiments are primarily validated on vision models (DINOv2, CLIP, SigLIP), with only preliminary results on language models.
- Only simple two-level hierarchical structures (synthetic experiments) were validated, leaving deeper or more complex semantic hierarchies to be explored.

## Related Work & Insights
- **vs. Vanilla/TopK SAE**: Standard SAEs enforce global quasi-orthogonality and encode with a single linear projection at inference; MP-SAE allows interference in the dictionary but selects a conditionally orthogonal subset during inference, which is more flexible.
- **vs. Matryoshka SAE**: Matryoshka models hierarchy explicitly through nested training objectives, but still relies on linear encoders, failing to avoid negative intra-level interference; MP-SAE achieves hierarchical separation more naturally via the residual mechanism.
- **vs. Orthogonal Matching Pursuit (OMP)**: OMP re-orthogonalizes all selected features, offering stronger theoretical guarantees but at higher computational expense; MP-SAE only orthogonalizes against the immediate preceding step, which is empirically sufficient.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Both the formalization of conditional orthogonality and the unrolling of MP as an SAE encoder are novel and profound contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive validation using both synthetic and real models, with especially compelling cross-modal analysis; however, language model experiments are relatively limited.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous theoretical derivation, clear motivational threads, and well-designed figures.
- Value: ⭐⭐⭐⭐⭐ Makes a fundamental contribution to the interpretability field—challenging the sufficiency of the LRH and providing a constructive alternative.

## Related Work & Insights
- **vs. Vanilla/TopK SAE**: Standard SAEs enforce global quasi-orthogonality along with a single linear encoding, failing to distinguish between intra-level and cross-level structural differences, which leads to feature absorption (sub-concepts absorbed by parent concepts); MP-SAE naturally avoids this issue via sequential residual decomposition.
- **vs. Matryoshka SAE**: Matryoshka explicitly models hierarchy through nested training objectives but still uses a linear encoder, introducing negative interference between sibling concepts; the residual mechanism of MP-SAE achieves hierarchical separation more naturally without damaging intra-level structures.
- **vs. OMP (Orthogonal Matching Pursuit)**: OMP re-orthogonalizes all selected features, offering stronger theoretical guarantees but under a computational cost of $O(Tk^2)$; MP-SAE only orthogonalizes against the previous step ($O(Tk)$), which performs well enough empirically and is more suitable for end-to-end training.
- **vs. JumpReLU/Gated SAE**: These improved activation functions remain within the linear encoder framework, failing to access non-linearly accessible features; MP-SAE's residual iterations essentially construct a non-linear encoding path.

## Insights & Connections
- The formalization of conditional orthogonality provides an analytical framework for understanding the hierarchical organization of concepts inside LLMs—enabling research on how semantic hierarchies (e.g., "animal $\rightarrow$ mammal $\rightarrow$ cat") are encoded in LLMs.
- The cross-modal feature discovery capability of MP-SAE can be used to investigate whether visual and text embeddings in VLMs are truly aligned (rather than just possessing superficial cosine similarity).
- The adaptive sparsity property makes MP-SAE suitable for applications requiring dynamic adjustment of explanation granularity—obtaining coarse-grained explanations with a few steps and fine-grained explanations with more steps.
- The constructive explanation of "dark matter" suggests a clear direction for improvement to the SAE community—the unexplained parts of standard SAEs are not noise but rather meaningful features requiring non-linear access.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Hierarchical Concept Embedding & Pursuit for Interpretable Image Classification](../../CVPR2026/interpretability/hierarchical_concept_embedding_pursuit_for_interpretable_image_classification.md)
- [\[NeurIPS 2025\] Transformer Key-Value Memories Are Nearly as Interpretable as Sparse Autoencoders](transformer_key-value_memories_are_nearly_as_interpretable_as_sparse_autoencoder.md)
- [\[NeurIPS 2025\] SHAP Values via Sparse Fourier Representation](shap_values_via_sparse_fourier_representation.md)
- [\[NeurIPS 2025\] How Intrinsic Motivation Shapes Learned Representations in Decision Transformers: A Cognitive Interpretability Analysis](toward_explainable_offline_rl_analyzing_representations_in_intrinsically_motivat.md)
- [\[NeurIPS 2025\] Time-Evolving Dynamical System for Learning Latent Representations of Mouse Visual Cortex](time-evolving_dynamical_system_for_learning_latent_representations_of_mouse_visu.md)

</div>

<!-- RELATED:END -->

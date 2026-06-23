---
title: >-
  [Paper Note] Learning to Weight Parameters for Training Data Attribution
description: >-
  [ICLR 2026][Interpretability][data attribution] This paper points out that the "huge disparity in attribution quality across different parameter groups" in gradient attribution is ignored by existing methods. It proposes a unified framework that uses **self-supervision** to directly learn a set of parameter group weights $w$ from data. Without requiring annotations,
tags:
  - ICLR 2026
  - Interpretability
  - data attribution
  - influence functions
  - gradient-based attribution
  - parameter heterogeneity
  - self-supervised
  - diffusion models
date: 2026-05-08
content_hash: aa60ad5efbf553aa
---
# Learning to Weight Parameters for Training Data Attribution

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=EhUkQp9Yah](https://openreview.net/forum?id=EhUkQp9Yah)  
**Code**: To be confirmed  
**Area**: Interpretability / Training Data Attribution  
**Keywords**: data attribution, influence functions, gradient-based attribution, parameter heterogeneity, self-supervised, diffusion models  

## TL;DR
This paper points out that the "huge disparity in attribution quality across different parameter groups" in gradient attribution is ignored by existing methods. It proposes a unified framework that uses **self-supervision** to directly learn a set of parameter group weights $w$ from data. Without requiring annotations, it systematically improves the attribution precision of methods like TracIn / TRAK / EK-FAC and can decouple semantic dimensions such as subject, style, and background.

## Background & Motivation

**Background**: Training data attribution aims to answer "which training samples are most responsible for a specific model output," which is critical for copyright, privacy, and data governance. Scalable mainstream approaches are gradient-based: TracIn directly calculates the inner product of gradients between the query and training samples; Influence Functions / TRAK further introduce the Hessian (or its low-rank/Kronecker approximation + random projection) to perform second-order preprocessing on the gradients.

**Limitations of Prior Work**: These methods either treat parameters **equally** (TracIn concatenates gradients of all parameters with equal weights) or rely on Hessian approximations for **implicit weighting**. However, the Hessian is unsolvable for large models, models rarely truly converge to the optimum, and information loss from random projections is significant—implicit weighting derived from an "imprecise and noisy" signal cannot reliably reflect the true importance of each parameter group.

**Key Challenge**: Using Linear Datamodeling Score (LDS), the authors found that attribution signals are **highly non-uniform** within the network: in a UNet, the average LDS of `up_blocks.1/2` (5.58) is much higher than other blocks; the output projection layer of self-attention (`attn.to_out`) is significantly stronger than the k/q/v projections of cross-attention. Furthermore, different parameter groups are responsible for different semantics—style attribution concentrates in shallow layers, while background attribution concentrates in specific attention components. In other words, **attribution quality varies systematically with parameter position and function, which is precisely the information neglected by existing equal/implicit weighting schemes**.

**Goal**: Instead of relying on noisy Hessian approximations, the goal is to **explicitly and directly learn the importance weights of each parameter group from data**, allowing any gradient attribution method to be enhanced in a plug-and-play manner while making the weights themselves interpretable.

**Core Idea**: **[Explicit Parameter Weighting + Self-supervised Bootstrap]** Formulate attribution as a unified "weighted similarity modulated by a diagonal weight matrix $\mathrm{Diag}(w)$," and then use the top-k rankings of existing methods as pseudo-positive samples to self-supervisingly optimize $w$ to maximize the attribution signal-to-noise ratio.

## Method

### Overall Architecture
The method consists of two steps. The first step is to **unify the weighting form**: partition model parameters into $M$ disjoint groups (by layer or tensor) and assign each group a non-negative scalar weight $w_j$. This makes the attribution score a weighted similarity modulated by $\mathrm{Diag}(w)$, covering both TracIn (where the kernel is the identity matrix) and TRAK-like kernel methods. The second step is **self-supervised weight learning**: since no attribution ground-truth exists, the top-k ranking of the base method is used as pseudo-positive samples to construct a "signal-to-noise" ratio loss to optimize $w$. The top-k set is dynamically refreshed along with the weights during iteration, bootstrapping increasingly better weights from weak signals.

```mermaid
flowchart TD
    A[训练集 D + query 集 Q] --> B[base 归因方法<br/>TracIn/TRAK/EK-FAC...]
    B --> C[按参数组拆梯度特征<br/>g_j(x), 预计算各组归因贡献]
    C --> D[加权得分<br/>τ̃ = g(query)ᵀ·Diag(w)·K·g(xₙ)]
    D --> E[取 top-k 当伪正样本<br/>I_top-k(w)]
    E --> F[自监督损失 L_SSL<br/>正样本均分 / ℓ2 范数]
    F -->|softmax 参数化保证 w≥0| G[更新 w]
    G -->|每步重算 top-k| D
    G --> H[学到的参数组权重 w*<br/>可整体/可按语义 subject·style·background]
```

### Key Designs

**1. Unified parameter weighting attribution form: explicitly incorporating "which parameter groups to trust" into the score.** The authors abstract any gradient attribution into a weightable bilinear form. Let the concatenated gradient features of the query and training sample $x_n$ be $g(\cdot)=[g_1,\dots,g_M]$. After introducing non-negative weights $w$, the weighted score is defined as:

$$\tilde\tau(x_{query}, x_n; w) = g(x_{query})^\top \cdot \mathrm{Diag}(w) \cdot K \cdot g(x_n),$$

where $\mathrm{Diag}(w)$ expands $w_j$ across all dimensions of group $j$, and $K$ is a similarity metric. When $K=I$, it reduces to TracIn-style weighted inner products; when $K=(\Phi^\top\Phi+\lambda I)^{-1}$, it matches kernel methods like TRAK. A critical engineering trade-off is: weights are **only applied on the query side**, treating the training side $K\,g(x_n)$ as a fixed term precomputed once. If weighted symmetrically on both sides, the kernel term for all training samples would need recomputation at every update of $w$, which is computationally prohibitive. Intuitively, $w$ encodes "how credible each parameter group is when reading the query gradient signal."

**2. Self-supervised weight learning: using the base method's top-k for pseudo-labels to maximize signal-to-noise ratio.** In the absence of ground truth, the "top-k training samples with the highest scores from the base method" are assumed to be weakly credible pseudo-positives. For a query $x_{query}$, let the vector of all weighted scores be $\tilde\tau(x_{query},D;w)$ and its top-k index set be $I_{top\text{-}k}(w)$. The loss is defined as the average score of pseudo-positives divided by the overall score magnitude:

$$L_{SSL}(w) = -\frac{1}{\lVert \tilde\tau(x_{query},D;w)\rVert_2}\Big(\frac{1}{k}\sum_{i\in I_{top\text{-}k}(w)} \tilde\tau(x_{query},x_i;w)\Big).$$

The numerator acts as a signal strength estimate, while the denominator ($\ell_2$ norm) acts as an estimate of total noise. The authors prove in the appendix that minimizing this is equivalent to maximizing the SNR of attribution scores. During optimization, $I_{top\text{-}k}(w)$ is re-evaluated at each step following updates to $w$, allowing the ranking signal to bootstrap. Finally, with the expectation over the query distribution $Q$ and $\lambda\lVert w\rVert^2$ regularization, $w^*=\arg\min_{w\ge0} \mathbb{E}_{x_{query}\sim Q}[L_{SSL}]+\lambda\lVert w\rVert^2$, where non-negativity is ensured by softmax parameterization.

**3. Extremely efficient: group-level contribution precomputation, allowing convergence "within one minute."** Efficiency stems from two factors: first, the number of weights to be learned is minimal (one scalar per parameter group, e.g., per layer); second, the score is **linearly separable** across parameter groups. Consequently, the attribution contribution of each group is **precomputed and cached**, and optimization is applied only to the group-level scalar scores. This avoids recomputing with weighted gradient features at each step, typically allowing weight learning to converge within a minute.

**4. Fine-grained semantic attribution: decoupling subject/style/background by changing the query set.** The same mechanism can learn specialized semantic weight sets $w_{style}, w_{subject}, w_{background}$. The technique involves constructing a query set that **emphasizes the target attribute while leaving others blank**: for instance, when learning style weights, the prompt specifies only the style without specifying the subject or background. This causes style-related training samples to rise in ranking, and optimization naturally increases the weights of parameter groups that "consistently contribute style semantics." The resulting semantic weights are more focused than general weights and exhibit distinct distribution patterns.

## Key Experimental Results

### Main Results

Image Classification (ImageNet, LDS %, higher is better):

| Method | ResNet-18 w/o w | ResNet-18 **w** | ViT-B/16 w/o w | ViT-B/16 **w** |
|------|------|------|------|------|
| TracIn | 11.39 | **23.92** | 9.67 | **17.63** |
| TRAK | 16.86 | **23.30** | 14.77 | **16.74** |

Language Modeling (WikiText-103, GPT-2-small, LDS %):

| Method | w/o w | **w** |
|------|------|------|
| TracIn | 6.31 | **9.23** |
| TRAK | 12.69 | **14.63** |
| LoGRA | 11.42 | **12.86** |
| EK-FAC | 15.14 | **18.33** |

Diffusion Models (LDS %, selected from four datasets):

| Method | ArtBench-2 w/o→w | Naruto w/o→w | SB-Pokemon w/o→w | CIFAR-2 w/o→w |
|------|------|------|------|------|
| TracIn | 17.63→**22.02** | 10.54→**13.59** | 9.34→**11.79** | 1.39→**8.48** |
| TRAK | 18.39→**22.15** | 14.61→**17.02** | 10.68→**12.24** | 8.51→**10.59** |
| D-TRAK | 22.72→**25.15** | 16.75→**17.85** | 33.88→**35.05** | 10.17→**12.18** |
| DAS | 30.47→**31.58** | 18.72→**20.44** | 33.55→**36.12** | 12.66→**13.79** |

### Ablation Study

Downstream task validation (Mislabel Detection AUC / tail-patch, higher is better):

| Task | Method | w/o w | **w** |
|------|------|------|------|
| Mislabel Detection AUC (ResNet-18) | TracIn | 54.40 | **61.46** |
| Mislabel Detection AUC (ViT-B/16) | TracIn | 71.27 | **83.58** |
| Mislabel Detection AUC (ViT-B/16) | TRAK | 80.08 | **83.48** |
| Tail-patch (WikiText) | TracIn | 4.66 | **5.60** (Δ+0.94) |
| Tail-patch (WikiText) | EK-FAC | 5.54 | **6.09** (Δ+0.55) |

Fine-grained attribution (SB-Pokemon, D-TRAK, Recall@10 %): After learning semantic-specific weights, the recall for the corresponding semantic dimensions is significantly better than the baseline without weights, validating the decouplability of subject, style, and background.

### Key Findings
- **Universal Gain**: LDS consistently improves after weighting across three domains (Classification/Language/Diffusion) using 6 base methods (TracIn/TRAK/EK-FAC/JourneyTRAK/D-TRAK/DAS). On CIFAR-2, TracIn surged from 1.39 to 8.48.
- **Attribution heterogeneity is a stable intrinsic property of the model**: Cosine similarity analysis in the appendix shows that per-group LDS and learned weights are highly consistent across different datasets and attribution methods, indicating that this heterogeneity is not an artifact of specific settings.
- **Insensitive to $k$, trains extremely fast**: The hyperparameter $k$ is stable across a wide range, and weight learning usually converges within one minute.

## Highlights & Insights
- **Making "Implicit Weighting" Explicit**: Historically, Hessian preprocessing was essentially a form of weighting, but it was derived from unreliable approximations. This paper decouples this step to learn directly from data, bypassing the noisy approximation chain.
- **Elegance of the Unified Framework**: A single $\mathrm{Diag}(w)\cdot K$ form covers both inner-product and kernel methods, providing plug-and-play benefits for all gradient attribution methods.
- **Single-sided weighting + Precomputation** is the key engineering insight for scalability, reducing weight learning from "recomputing full kernel terms at each step" to minute-level execution.
- **Interpretability as a bonus**: The learned weights provide a semantic map indicating "which layers are responsible for style and which for the subject," allowing attribution methods to perform semantic decoupling for the first time.

## Limitations & Future Work
- Self-supervision relies on the initial ranking of the base method as pseudo-labels. If the base method performs extremely poorly on a specific task, the weak bootstrap signal may be insufficient for correction.
- Weight granularity is at the "parameter group level" (one scalar per layer); more fine-grained weighting (per single parameter or direction) was not explored for potential additional gains.
- Semantic decoupling experiments were primarily conducted on controlled datasets like SB-Pokemon; in real-world open-domain generation, "semantic" boundaries are blurrier, potentially complicating the construction of specialized query sets.
- The training-side kernel term is fixed to precomputed values, meaning weights cannot feed back into the training-side representation, which is theoretically a sub-optimal approximation (though it ensures scalability).

## Related Work & Insights
- **Gradient Attribution Genealogy**: TracIn (gradient inner product) → Influence Functions / TRAK (Hessian/kernel preprocessing) → Diffusion-specific JourneyTRAK / D-TRAK / DAS, and LLM-specific LoGRA / TrackStar. This work is orthogonal to these and acts as an additive "parameter weighting" enhancement.
- **Parameter Importance Heterogeneity**: This has long been utilized in pruning, machine unlearning, knowledge editing, and quantization (knowledge localization / mixed precision), but this work is the first to explicitly introduce it to data attribution.
- **Inspiration**: Converting the common observation that "different model components perform different functions" into a lightweight, self-supervised, and interpretable weighting layer is a paradigm that can be extended to other tasks relying on gradient similarity (e.g., retrieval, influence estimation, and data filtering).

## Rating
- Novelty: ⭐⭐⭐⭐ — First to explicitly model parameter heterogeneity in data attribution; the unified framework + self-supervised bootstrap approach is clear and orthogonal to existing methods.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers three domains (classification/language/diffusion), 6 base methods, multiple datasets, and includes downstream validation and semantic decoupling with solid evidence.
- Writing Quality: ⭐⭐⭐⭐ — The logical progression from motivation (heterogeneity measurement) to the unified form and self-supervised objective is smooth, supported by ample charts.
- Value: ⭐⭐⭐⭐ — Plug-and-play, minute-level training, consistent improvements, and semantic interpretability make it practical for copyright and data governance scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Influence Dynamics and Stagewise Data Attribution](influence_dynamics_and_stagewise_data_attribution.md)
- [\[ICLR 2026\] Learning to Interpret Weight Differences in Language Models](learning_to_interpret_weight_differences_in_language_models.md)
- [\[ICML 2026\] ExPLAIND: Unifying Model, Data, and Training Attribution to Study Model Behavior](../../ICML2026/interpretability/explaind_unifying_model_data_and_training_attribution_to_study_model_behavior.md)
- [\[ICLR 2026\] Learning is Forgetting: LLM Training As Lossy Compression](learning_is_forgetting_llm_training_as_lossy_compression.md)
- [\[ICLR 2026\] LORE: Jointly Learning the Intrinsic Dimensionality and Relative Similarity Structure from Ordinal Data](lore_jointly_learning_the_intrinsic_dimensionality_and_relative_similarity_struc.md)

</div>

<!-- RELATED:END -->

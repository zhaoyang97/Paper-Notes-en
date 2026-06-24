---
title: >-
  [Paper Note] PLeaS: Merging Models with Permutations and Least Squares
description: >-
  [CVPR 2025][Model Merging] This work proposes PLeaS, a two-step model merging algorithm. First, it exploits permutation symmetry to partially match the features of two models (merging similar features while retaining dissimilar ones). Second, it uses layer-wise least squares optimization to align the merged model's features with the permuted ensemble features of the original models, achieving up to a 15 percentage point improvement over existing methods at the same model size…
tags:
  - "CVPR 2025"
  - "Model Merging"
  - "Permutation Symmetry"
  - "Least Squares"
  - "Partial Merging"
  - "Feature Distillation"
date: 2026-05-08
content_hash: 6cc0cf1a86b28b96
---

# PLeaS: Merging Models with Permutations and Least Squares

**Conference**: CVPR 2025  
**arXiv**: [2407.02447](https://arxiv.org/abs/2407.02447)  
**Code**: [https://github.com/SewoongLab/PLeaS-Merging](https://github.com/SewoongLab/PLeaS-Merging)  
**Area**: Others / Model Merging  
**Keywords**: Model Merging, Permutation Symmetry, Least Squares, Partial Merging, Feature Distillation

## TL;DR
This work proposes PLeaS, a two-step model merging algorithm. First, it exploits permutation symmetry to partially match the features of two models (merging similar features while retaining dissimilar ones). Second, it uses layer-wise least squares optimization to align the merged model's features with the permuted ensemble features of the original models, achieving up to a 15 percentage point improvement over existing methods at the same model size.

## Background & Motivation

1. **Background**: With the prosperity of the open-source model ecosystem, a large number of specialized models fine-tuned on specific tasks/data have emerged (e.g., Code Llama, Vicuna). How to merge the capabilities of these specialized models into a single general-purpose model, while avoiding the memory and computational overhead of storing and running multiple models during inference, represents an important practical problem.

2. **Limitations of Prior Work**: (a) Traditional ensemble methods require storing all models, resulting in high memory overhead; (b) Existing model merging methods (e.g., Task Vectors, TIES) are typically limited to **fine-tuning from the same base model**, and cannot merge models with different initializations; (c) Methods like ZipIt! support different initializations but exhibit limited performance; (d) Most methods require the merged model to share the same size as the original models, failing to provide flexible trade-offs between model size and performance.

3. **Key Challenge**: The feature spaces of two models trained on different datasets vary significantly—direct weight averaging leads to "destructive interference." However, overlapping parts exist within their features that can be merged. The key challenge lies in identifying and handling "mergeable" versus "unmergeable" features.

4. **Goal**: To design a method capable of merging models with **different initializations**, supporting flexible control over the merged model size (between 1x and 2x), and functioning without the need for training data.

5. **Key Insight**: Utilizing the permutation invariance of neural networks—permuting hidden neurons in any order does not alter model functionality. By finding the optimal permutation, similar features are averaged and merged, while dissimilar features are kept separate.

6. **Core Idea**: Two-step merging: (1) Permutation step—extends Git Re-Basin to support partial merging, controlling the merging ratio of each layer; (2) Least Squares step—solves layer-wise least squares to align the merged model's features with the permuted ensemble of the original models.

## Method

### Overall Architecture
Input consists of two models A and B with the same architecture (potentially with different initializations and training data), along with a target computational budget B. First, activation matching (or weight matching) is utilized to find the permutation matrix $P_i$ for each layer of model B. Based on feature similarity, $k_i$ most-matching features are selected to merge, while the rest are retained. Second, a least squares problem is solved layer-by-layer to force the output of the merged model's layers to approximate the output of the permuted ensemble of the two original models. The final output is a merged model with a width of $2d_i - k_i$.

### Key Designs

1. **Partial Permutation Merging**:
    - **Function**: Flexibly chooses how many features to merge at each layer, thereby controlling the size of the merged model.
    - **Mechanism**: Extends the full permutation in Git Re-Basin to a partial permutation. Given the permutation matrix $P_i$, the $k_i$ indices that minimize $\|Z_{J,i}^A - (P_i Z_{:,i}^B)_J\|_F^2$ are selected as the features to merge, while the remaining $d_i - k_i$ features are preserved separately in both models. Post-merging, the width of the $i$-th layer becomes $2d_i - k_i$. The merge ratios $k_i/d_i$ across different layers can be chosen independently and are automatically determined under the target computational budget constraint by optimizing a proxy objective.
    - **Design Motivation**: When the training data of the two models differs significantly, forcing all features to merge leads to destructive interference. Preserving incompatible features is critical, whereas ZipIt! can only uniformly apply 1x for prefix layers and 2x for suffix layers, lacking sufficient flexibility.

2. **Permuted Least Squares**:
    - **Function**: Optimizes the weights of the merged model to align its features with the effect of the permuted ensemble of the original models.
    - **Mechanism**: Solves $W_i^M = \arg\min_W \|(Z_i^A + P_i Z_i^B)W - (Z_{i+1}^A + P_{i+1} Z_{i+1}^B)\|^2$ independently for each layer. This ensures that the $i$-th layer of the merged model maps the permuted ensemble input $\tilde{Z}_i$ of the two models to the ensemble output $\tilde{Z}_{i+1}$. This is equivalent to mimicking the behavior of the ensemble model layer by layer. Although this can be solved in closed form using OLS, it is practically implemented via gradient descent for compatibility with convolutional layers; due to the convexity of the objective function, it converges in fewer than 100 steps.
    - **Design Motivation**: Relying solely on permutation and weight averaging (Git Re-Basin) suffers from severe performance degradation when models differ significantly—averaging weights degrades feature quality. Matching ensemble features via least squares recovers a substantial portion of the performance. Compared to RegMean (least squares without permutation), fitting after permutation-based feature alignment yields much better results.

3. **PLeaS-free variant**:
    - **Function**: Enables model merging even when training domain data is unavailable.
    - **Mechanism**: Uses public general-purpose datasets (such as ImageNet) in place of training domain data to compute activation matching and least squares. Experiments show that at a model size of 1.2x, the performance loss is less than 2%.
    - **Design Motivation**: In practice, training data may be unavailable due to privacy or commercial constraints; PLeaS-free greatly broadens the applicability of the method.

### Loss & Training
- PLeaS does not require gradient-based training: the permutation step solves a linear assignment problem using the Hungarian algorithm; the least squares step solves a convex optimization.
- Batch-norm statistics are recomputed after merging (following the recommendations of the REPAIR method).
- Other than the target computational budget, PLeaS requires no hyperparameters.

## Key Experimental Results

### Main Results

DomainNet shared label space (ResNet-50, 1x size):

| Method | Clipart | Infograph | Painting | Real | Average |
|------|---------|-----------|----------|------|------|
| Simple Avg | 1.2 | 0.8 | 1.9 | 2.1 | 1.5 |
| Git Re-Basin | 18.2 | 7.8 | 18.8 | 26.5 | 17.8 |
| ZipIt! | 26.9 | 12.2 | 27.1 | 37.4 | 25.9 |
| MuDSC | 34.0 | 14.3 | 29.5 | 45.1 | 30.7 |
| **PLeaS** | **41.7** | **16.9** | **40.8** | **55.1** | **38.6** |

Different label spaces (CUB/Pets/Dogs/NABirds, linear probing):

| Method | CUB | Pets | Dogs | NABird | Average |
|------|-----|------|------|--------|------|
| ZipIt! | 67.5 | 83.6 | 60.0 | 56.3 | 66.9 |
| MuDSC | 70.1 | 82.5 | 63.2 | 58.2 | 68.5 |
| **PLeaS** | **75.2** | **85.0** | **69.6** | **69.7** | **74.9** |

### Ablation Study

| Configuration | DomainNet Average | Explanation |
|------|---------------|------|
| Permutation only (no LS) | ~17.8 | Git Re-Basin baseline |
| + Least Squares (PLeaS) | 38.6 | LS step brings ~+20% improvement |
| RegMean (LS without permutation) | 12.1 | Direct LS without permutation performs poorly |
| PLeaS-free (using ImageNet) | ~37.5 | Only ~1% loss |
| PLeaS 1.2x size | Close to Ensemble 2x | 20% extra parameters dramatically bridge the gap |

### Key Findings
- The Least Squares step is a critical contribution, nearly doubling the accuracy of Git Re-Basin on DomainNet (17.8 $\rightarrow$ 38.6).
- Permutation is a necessary prerequisite for Least Squares—RegMean (LS without permutation) yields only 12.1%, far below PLeaS.
- The flexibility of partial merging is vital: adding only 20% parameters (1.2x) matches the performance closely to a 2x ensemble.
- PLeaS-free with ImageNet data incurs almost no performance loss (<1%), making it highly practical.
- As the model scale grows (ResNet-18 $\rightarrow$ 50 $\rightarrow$ 101), the advantages of PLeaS become more pronounced.
- PLeaS is also effective on ViT models.

## Highlights & Insights
- **The complementary design of the two-step approach is exquisite**: permutation addresses the feature alignment issue, while least squares solves the weight degradation issue; both are indispensable. The comparison between RegMean (LS only) and Git Re-Basin (permutation only) neatly demonstrates this.
- **Degree of freedom in partial merging**: the freedom to independently choose the merge ratio for each layer is a key innovation. Since feature compatibility varies across layers, uniform strategies (like ZipIt!) lead to suboptimal results. Auto-searching for the optimal configuration via a proxy objective eliminates hyperparameters.
- **PLeaS-free concept**: merging models with general-purpose datasets as a substitute for domain data with negligible performance loss. This breaks the restriction of requiring training data, vastly expanding actual application scenarios.
- **Theoretically unified perspective**: formulating model merging as a layer-wise approximation of an ensemble model provides a clear optimization objective.

## Limitations & Future Work
- Currently, only merging of identical network architectures has been verified; merging heterogeneous architectures (such as CNNs and ViTs) is not addressed.
- The Least Squares step requires forward propagation to calculate activations, and its computational overhead for extra-large models (LLMs) needs to be assessed.
- Validation is limited to image classification tasks; the effectiveness of extending to NLP/multimodal tasks remains to be explored.
- The paper mentions the merging scenario of Code Llama and Vicuna but lacks actual experiments.
- The efficiency and optimality of the search algorithm for the optimal layer width configuration in partial merging warrant further investigation.

## Related Work & Insights
- **vs Git Re-Basin**: Git Re-Basin only performs weight averaging after full permutation, exhibiting poor performance on models with different training data. PLeaS's two-step approach (partial permutation + LS) boosts it from 17.8% to 38.6%.
- **vs ZipIt!**: ZipIt! also supports partial merging but is constrained to using 1x for prefix layers and 2x for suffix layers, which is too coarse. PLeaS independently controls each layer, outperforming it by 10%+ under the same budget.
- **vs RegMean**: RegMean performs layer-wise LS without permutation, resulting in fitting direct features without alignment, which performs vastly worse than PLeaS.
- **vs Task Vectors**: Task Vectors requires a shared base model and cannot handle different initializations. PLeaS removes this constraint.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of partial permutation and layer-wise least squares represents a major advancement in model merging.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Highly comprehensive, covering 4 domains of DomainNet, 4 fine-grained datasets, 3 ResNet sizes + ViT, multiple baselines, PLeaS-free ablations, and model size vs. performance trade-off curves.
- **Writing Quality**: ⭐⭐⭐⭐ The paper is well-structured, with intuitive figures and concise mathematical derivations.
- **Value**: ⭐⭐⭐⭐ It addresses the practical pain point of merging models with different initializations, and PLeaS-free further enhances its real-world utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] a representer theorem for hawkes processes via penalized least squares minimizat](../../ICLR2026/others/a_representer_theorem_for_hawkes_processes_via_penalized_least_squares_minimizat.md)
- [\[ICLR 2026\] Do We Really Need Permutations? Impact of Model Width on Linear Mode Connectivity](../../ICLR2026/others/do_we_really_need_permutations_impact_of_model_width_on_linear_mode_connectivity.md)
- [\[CVPR 2025\] On the Generalization of Handwritten Text Recognition Models](on_the_generalization_of_handwritten_text_recognition_models.md)
- [\[CVPR 2025\] Feature Selection for Latent Factor Models](feature_selection_for_latent_factor_models.md)
- [\[CVPR 2025\] MagicArticulate: Make Your 3D Models Articulation-Ready](magicarticulate_make_your_3d_models_articulation-ready.md)

</div>

<!-- RELATED:END -->

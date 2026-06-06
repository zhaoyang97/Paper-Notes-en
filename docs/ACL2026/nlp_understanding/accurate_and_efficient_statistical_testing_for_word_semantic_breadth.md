---
title: >-
  [Paper Note] Accurate and Efficient Statistical Testing for Word Semantic Breadth
description: >-
  [ACL 2026][NLP Understanding][Word Semantic Breadth] This paper identifies that directly comparing the semantic breadth of two words using permutation tests in contextual embedding space severely inflates Type-I errors d…
tags:
  - "ACL 2026"
  - "NLP Understanding"
  - "Word Semantic Breadth"
  - "Permutation Test"
  - "Householder Transformation"
  - "Contextual Embeddings"
  - "GPU Acceleration"
date: 2026-05-08
content_hash: f5151f9637da41c5
---

# Accurate and Efficient Statistical Testing for Word Semantic Breadth

**Conference**: ACL 2026  
**arXiv**: [2605.08048](https://arxiv.org/abs/2605.08048)  
**Code**: https://rebrand.ly/WordSemanticBreadth  
**Area**: LLM Efficiency / Lexical Semantic Analysis  
**Keywords**: Word Semantic Breadth, Permutation Test, Householder Transformation, Contextual Embeddings, GPU Acceleration

## TL;DR
This paper identifies that directly comparing the semantic breadth of two words using permutation tests in contextual embedding space severely inflates Type-I errors due to differences in mean directions. It proposes using Householder reflections to align mean directions before permutation, reducing Type-I errors by 32.5%, and provides a GPU-batched implementation achieving 23x acceleration.

## Background & Motivation
**Background**: Contextual embeddings (e.g., BERT) have become standard tools for semantic modeling. Nagata and Tanaka-Ishii (2025) treat the "token embedding cloud of a word in different contexts" as a distribution and use its dispersion as a proxy for "semantic breadth / contextual diversity"—which is highly valuable for lexicography (determining the number of word senses).

**Limitations of Prior Work**: To determine if the difference in semantic breadth between two words is "statistically significant," the simplest approach is a permutation test after merging the token clouds of both words. However, the authors point out: **When the mean directions of two words on the sphere differ (i.e., different meanings), naive permutation mixes the two sets of points into different regions, incorrectly attributing "directional difference" to "dispersion difference"**. This leads to severely inflated Type-I errors (falsely determining significant breadth differences), violating the purpose of significance testing.

**Key Challenge**: "Semantic difference" and "breadth difference" in the embedding space are statistically entangled; the "exchangeability" assumption of the permutation test is broken when directional differences are significant.

**Goal**: (i) Calibrate the p-value to be sensitive only to breadth and not to semantic direction; (ii) Make the permutation test computationally feasible at the vocabulary scale (naive CPU implementations are too slow).

**Key Insight**: The authors noted that if one word's token cloud can be "rotated" to the mean direction of the other word before permutation, the confounding factor of directional difference can be eliminated. This is exactly the geometric function of a Householder reflection—an orthogonal transformation that preserves norms and relative geometry.

**Core Idea**: First use a Householder reflection to align the mean directions of the two words, then perform the merged permutation; simultaneously, rewrite the entire permutation process as batched matrix multiplication on the GPU to enable large-scale vocabulary analysis.

## Method

### Overall Architecture
Input: An "unknown/target" word $u$ and a "known/reference" word $k$, with their respective token embedding sets $X = \{\mathbf{x}_i\}_{i=1}^n$ and $Y = \{\mathbf{y}_j\}_{j=1}^m$ extracted from a corpus (already $\ell_2$-normalized to the unit sphere $\mathbb{S}^{d-1}$). Output: A p-value answering "whether $u$ is semantically broader than $k$." Process: (1) Compute the mean directions $\hat{\mu}_x, \hat{\mu}_y$ of the two words; (2) Construct a Householder matrix $H$ such that $H\hat{\mu}_x = \hat{\mu}_y$; (3) Multiply each vector in $X$ by $H$ to obtain aligned $X'$; (4) Merge $Z = X' \cup Y$ for the permutation test; (5) Use the log-difference of the Mean Resultant Length as the test statistic.

### Key Designs
1. **Householder Mean Direction Alignment**:

    - Function: Eliminates the mean direction difference between two sets of embeddings via a single orthogonal reflection, preserving all relative geometric information (norm, relative distance, dispersion).
    - Mechanism: Take the axis $\mathbf{u} = (\hat{\mu}_x - \hat{\mu}_y)/\|\hat{\mu}_x - \hat{\mu}_y\|_2$ and construct $H = I - 2\mathbf{u}\mathbf{u}^\top$. This is a single reflection satisfying $H\hat{\mu}_x = \hat{\mu}_y$ and $H^\top H = I$. The authors prove (Appendix D) that the Householder transformation is precisely the transformation that maximizes the mean resultant length of the merged set, thereby restoring "exchangeability" as much as possible. **Why not Procrustes**: Procrustes requires point-to-point matching, whereas the two words have different numbers of tokens with no correspondence, making it inapplicable.
    - Design Motivation: Using a single reflection instead of full rotation/optimization is chosen because the reflection itself has minimal computational overhead ($O(d^2)$) and is mathematically sufficient to align any two unit vectors.

2. **Fixed-Space Permutation Design**:

    - Function: Maintains a constant merged set $Z$ after alignment, ensuring all permutations occur within the same aligned space so that the permutation distribution reflects true variability under $H_0$.
    - Mechanism: The authors intentionally do not re-estimate $H$ during each permutation—if alignment were re-done for each permutation, the geometric space would drift with label changes, making the permutation distribution meaningless. The test statistic uses the log-difference of MRL: $T_{obs} = \log v(X') - \log v(Y)$, where $v(X) = 1/g_d(r(X))$ and $r(X) = \|\frac{1}{n}\sum_i \mathbf{x}_i\|_2$ is the mean resultant length (higher values indicate concentration, lower values indicate dispersion).
    - Design Motivation: Keeping the alignment space fixed is key to the validity of the permutation test—this ensures the permutation only shuffles "label assignments" rather than "geometric space," giving the p-value comparative meaning. Appendix E validates this design through a split-half sanity check on the same word.

3. **GPU Batched Permutation**:

    - Function: Rewrites the naive $O(BNd)$ serial permutation (B permutations, N total samples, d dimensions) into a one-time matrix multiplication, achieving 23x acceleration on GPU.
    - Mechanism: Each permutation $b$ is represented by a sign vector $\mathbf{s}^{(b)} \in \{+1, -1\}^N$ ($+1$ for group 1, $-1$ for group 2). $B$ permutations are stacked into a sign matrix $\mathbf{S} \in \{+1,-1\}^{B \times N}$. The mean vectors for each group can be calculated at once: $\mathbf{M} = \mathbf{S}\mathbf{X} \in \mathbb{R}^{B \times d}$, then the $\ell_2$ norm of each row is computed batch-wise to get $B$ MRLs. The entire process consists almost entirely of dense matmul + reduction, maximizing GPU compute utility.
    - Design Motivation: Lexicographical applications require many pairwise comparisons at the vocabulary scale (thousands to tens of thousands of words), which is impractical with naive CPU implementations. Rewriting the permutation as linear algebra batch operations transforms the "significance test" from a CPU bottleneck into a GPU-friendly task.

### Loss & Training
This work presents a statistical inference method; no model training is involved. The core statistic is the **mean resultant length** $r(X) = \|\frac{1}{n}\sum \mathbf{x}_i\|_2 \in [0,1]$ and the corresponding concentration parameter $\kappa = g_d(r)$ (related to the von Mises-Fisher distribution); the proxy for semantic breadth is $v(X) = 1/g_d(r(X))$. The p-value is calculated using the standard Monte Carlo permutation formula with a +1 correction: $p = \frac{1 + \sum_b \mathbb{I}[T^{(b)} \geq T_{obs}]}{B + 1}$.

## Key Experimental Results

### Main Results: Comparison of Type-I Error and Runtime Efficiency

| Method | Type-I Error Rate | Time per Word Pair | Device |
|--------|-------------------|--------------------|--------|
| Naive Permutation (CPU) | High (Inflated) | 1.0× | CPU |
| Householder + GPU Permutation | -32.5% (Relative Reduction) | ~1/23 ≈ 23× Speedup | GPU |

### Ablation Study: Impact of Method Components

| Configuration | Type-I Error Control | True Breadth Difference Detection | Speed |
|---------------|----------------------|-----------------------------------|-------|
| Naive Permutation (No Alignment + CPU) | ❌ Severely Inflated | ✅ But with high false positives | Slow |
| Alignment Only (CPU) | ✅ Type-I -32.5% | ✅ Preserved | Slow |
| GPU Batch Only (No Alignment) | ❌ Still Inflated | ✅ | Fast (23×) |
| Full Method (Alignment + GPU) | ✅ | ✅ | Fast (23×) |

### Key Findings
- **Directional difference is the primary cause of Type-I inflation**: Type-I error dropped by 32.5% after Householder alignment, proving that the failure of naive permutation stems from direction-breadth entanglement rather than insufficient sample size.
- **Alignment has almost no loss in detecting real breadth differences**: Since the Householder transformation is orthogonal, it preserves all intra-group relative geometric relationships; word pairs with true breadth differences can still be detected.
- **GPU acceleration enhances method practicality**: The 23x speedup turns pairwise comparisons across thousands of words from an infeasible task into one completed in minutes, which is key to scaling the method to actual lexicographical workflows.
- **Fixed-Space design validated by split-half experiments**: Appendix E split the tokens of the same word into two halves—ideally, the p-values should be uniformly distributed in [0,1]. Naive permutation heavily biased towards small p-values (false positives); Householder + fixed-space pulled the distribution back to uniform, statistically validating the calibration.
- **Failure modes in small sample scenarios**: When the number of tokens per word is below approximately 50, the MRL estimation noise increases, and the power of the p-value drops sharply—the authors suggest subsampling to a uniform scale (e.g., 200 tokens/word) to stabilize variance.

## Highlights & Insights
- **Geometric intuition is very clear**: Using the image of "two groups of points on a sphere" in Figure 1 allows one to understand at a glance why naive permutation fails and why Householder can fix it—translating an abstract statistical problem into a visual geometric one.
- **Householder, not Procrustes, is the appropriate choice**: The author keenly observed that Procrustes requires point-to-point correspondence, while there is no natural correspondence between token clouds; Householder reflection successfully bypasses this difficulty.
- **Subtlety of the Fixed-Space design**: Many researchers might naturally think of "re-aligning for each permutation," but the authors prove that this actually destroys $H_0$; this is a common pitfall.
- **Realistic significance of GPU implementation**: Many statistical tests are "theoretically feasible but practically too slow" at scale. The approach of rewriting the permutation test as dense matmul can be generalized to other high-dimensional statistical inferences (e.g., bootstrap CI calculations).

## Limitations & Future Work
- **Author's acknowledgment**: The method is used to decide "which words need finer sense division" and does not directly predict the number of senses; it is suitable for prioritizing lexicographical tasks rather than full automation.
- **Additional limitations**: Assumes that the concentration parameters of the two words are comparable; for extremely anisotropic embeddings (like certain layers of BERT), de-anisotropization might be needed first. MRL estimation noise is high when sample size is very small (tokens per word < 50), reducing p-value reliability.
- **Future directions**: (i) Extend the alignment method to more than two words (e.g., ANOVA-type designs), requiring simultaneous alignment of multiple mean directions; (ii) Combine with existing de-anisotropization methods (like whitening) to further purify the geometry; (iii) Extend statistics from MRL to other indicators that directly characterize "multi-modality/polysemy" (like the number of components in a spherical GMM).

## Related Work & Insights
- **vs Nagata & Tanaka-Ishii 2025** (contextual diversity): They proposed using the dispersion of embedding clouds as a proxy for semantic breadth; this paper fills the gap of "how to statistically compare breadth differences between two words significantly."
- **vs Zmigrod et al. 2022** (exact permutation tests): Their method is only valid for discrete-valued statistics and cannot handle continuous high-dimensional MRL statistics; this paper fills that gap with GPU-accelerated Monte Carlo permutation.
- **vs Procrustes alignment** (Schönemann 1966): Procrustes requires point-to-point correspondence and is unsuitable for comparing token clouds without corresponding relationships; the Householder reflection in this paper successfully bypasses this constraint.
- **vs HyperLex** (Vulić 2017): HyperLex measures lexical entailment strength, which partially overlaps with "breadth" but is not equivalent; the method in this paper can be combined with it to distinguish between "semantic breadth" and "semantic containment."
- **vs Anisotropy research** (Ethayarajh 2019): They noted that BERT contextual embeddings suffer from severe anisotropy and large geometric differences across layers; this implies that preprocessing like whitening might be needed to stabilize results before applying this method, making it a promising combination to explore.

## Rating
- Novelty: ⭐⭐⭐⭐ Introducing Householder reflection into semantic breadth testing is an original and concise solution with very clear geometric motivation.
- Experimental Thoroughness: ⭐⭐⭐ Main results (-32.5% Type-I error, 23x acceleration) are quantified; however, it lacks end-to-end validation on large-scale downstream lexicographical tasks.
- Writing Quality: ⭐⭐⭐⭐⭐ Figure 1 geometric illustration combined with formal definitions works well; problem motivation, method, and empirical results flow seamlessly.
- Value: ⭐⭐⭐⭐ A directly usable tool for lexicography and NLP resource construction; also valuable for other researchers performing permutation tests in embedding spaces (e.g., concept drift detection).
- Engineering Friendliness: ⭐⭐⭐⭐ Open-source code and GPU implementation lower the barrier to reproduction; clear geometric/statistical intuition makes it easy to migrate to related problems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] BoundRL: Efficient Structured Text Segmentation through Reinforced Boundary Generation](boundrl_efficient_structured_text_segmentation_through_reinforced_boundary_gener.md)
- [\[ACL 2026\] Semantic Reranking at Inference Time for Hard Examples in Rhetorical Role Labeling](semantic_reranking_at_inference_time_for_hard_examples_in_rhetorical_role_labeli.md)
- [\[ACL 2026\] LLM-Guided Semantic Bootstrapping for Interpretable Text Classification with Tsetlin Machines](llm-guided_semantic_bootstrapping_for_interpretable_text_classification_with_tse.md)
- [\[ACL 2026\] SAM-NER: Semantic Archetype Mediation for Zero-Shot Named Entity Recognition](sam-ner_semantic_archetype_mediation_for_zero-shot_named_entity_recognition.md)
- [\[ACL 2026\] ASTRA: Adaptive Semantic Tree Reasoning Architecture for Complex Table Question Answering](astra_adaptive_semantic_tree_reasoning_architecture_for_complex_table_question_a.md)

</div>

<!-- RELATED:END -->

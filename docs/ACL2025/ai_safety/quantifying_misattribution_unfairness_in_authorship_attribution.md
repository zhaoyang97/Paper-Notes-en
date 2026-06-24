---
title: >-
  [Paper Note] Quantifying Misattribution Unfairness in Authorship Attribution
description: >-
  [ACL 2025][AI Safety][Authorship Attribution] This paper proposes the $\text{MAUI}_k$ metric to quantify "misattribution unfairness" in authorship attribution systems—where certain authors are systematically more likely to be falsely identified as suspect authors. The study reveals that this unfairness is highly correlated with the distance of the author's embedding to the centroid in the vector space.
tags:
  - "ACL 2025"
  - "AI Safety"
  - "Authorship Attribution"
  - "Fairness"
  - "Misattribution"
  - "Embedding Distribution"
  - "Ranking Bias"
date: 2026-05-08
content_hash: 838cc62819a2063b
---

# Quantifying Misattribution Unfairness in Authorship Attribution

**Conference**: ACL 2025  
**arXiv**: [2506.02321](https://arxiv.org/abs/2506.02321)  
**Code**: None  
**Area**: AI Safety  
**Keywords**: Authorship Attribution, Fairness, Misattribution, Embedding Distribution, Ranking Bias

## TL;DR
This paper proposes the $\text{MAUI}_k$ metric to quantify "misattribution unfairness" in authorship attribution systems—where certain authors are systematically more likely to be falsely identified as suspect authors. The study reveals that this unfairness is highly correlated with the distance of the author's embedding to the centroid in the vector space.

## Background & Motivation
- **Background**: Authorship attribution is widely used in scenarios such as forensics and literary analysis. The "needle-in-the-haystack" approach is the mainstream paradigm: identifying the most likely author of an anonymous text from a candidate pool (the haystack).
- **Limitations of Prior Work**: Existing evaluation metrics (MRR, R@k) focus solely on "whether the true author can be correctly identified," completely ignoring "whether other innocent authors are unfairly ranked highly." In forensic scenarios, even being included in a suspect shortlist can have severe consequences (e.g., being investigated or interrogated).
- **Key Challenge**: Are certain authors systematically ranked highly in unrelated queries, thus bearing a disproportionate risk of "misattribution"?
- **Core Idea**: Define the "Misattribution Unfairness Index" $\text{MAUI}_k$, which quantifies the extent to which a model exceeds expectations based on the expected number of times $E_k$ each author ranks in the top-$k$ under a random permutation (unbiased baseline).

## Method

### Overall Architecture
1. Define the misattribution unfairness metric $\text{MAUI}_k$.
2. Measure $\text{MAUI}_k$ across multiple embedding models and datasets.
3. Analyze the relationship between the embedding distribution (distance to the centroid) and the risk of misattribution.
4. Perform statistical tests to validate the association between "hard-to-find authors" and "proximity to the centroid."

### Key Designs
1. **$\text{MAUI}_k$ Metric (Misattribution Unfairness Index)**

    - Unbiased baseline: Under random permutation, the expected number of times each author is ranked in the top-$k$ is $E_k = \lceil \frac{k}{N_h} \times N_q \rceil$.
    - Definition: 
    $$\text{MAUI}_k = \frac{\sum_{j=1}^{N_h} \max(0, c_j^k - E_k)}{k \times (N_q - E_k)}$$
    - $c_j^k$: The actual number of times author $a_j$ is ranked in the top-$k$.
    - Normalized to $[0, 1]$, where 0 is the most fair and 1 is the most unfair.
    - The denominator represents the worst-case scenario (where the same $k$ authors are always ranked in the top-$k$).

2. **Embedding Centroid Distance Analysis**

    - Calculate the centroid (mean vector) of all haystack author embeddings.
    - Distance from each author to the centroid: $1 - \cos(\text{embedding}_j, \text{centroid})$.
    - Plot "average rank vs. distance to centroid" scatter plots.

3. **Hypothesis Testing for MRR and Distance to Centroid**

    - H1: Authors with high MRR are farther from the centroid than authors with low MRR.
    - H2: Authors with high MRR are farther from the centroid than a random subset.
    - H3: Authors with low MRR are closer to the centroid than a random subset.
    - Use the Mann-Whitney U test (non-parametric, making no assumption of normality).

### Loss & Training
This study is an evaluation and analysis work and does not involve new training. Only MPNet_AR underwent author representation fine-tuning:
- Uses cached multiple-negative ranking loss.
- Freezes the first 8 of the 12 layers, with a learning rate of 5e-5, batch size of 200, and training for 5000 steps.

## Key Experimental Results

### Main Results: Model Performance and Fairness

| Model | Reddit R@8 | Reddit MRR | Blogs R@8 | Blogs MRR |
|------|-----------|-----------|-----------|-----------|
| SBERT | 0.15 | 0.10 | 0.61 | 0.48 |
| LUAR | 0.82 | 0.71 | 0.97 | 0.90 |
| MPNet_AR | 0.40 | 0.30 | 0.96 | 0.88 |
| Wegmann | 0.08 | 0.05 | 0.45 | 0.32 |
| StyleDist. | 0.09 | 0.06 | 0.68 | 0.55 |

### $\text{MAUI}_k$ Values (Unfairness Measurement)

| Model | Reddit $\text{MAUI}_5$ | Reddit $\text{MAUI}_{10}$ | Blogs $\text{MAUI}_5$ | Blogs $\text{MAUI}_{10}$ |
|------|-------------|--------------|-------------|--------------|
| SBERT | 0.20 | 0.31 | 0.24 | 0.36 |
| LUAR | 0.06 | 0.12 | 0.15 | 0.26 |
| MPNet_AR | 0.09 | 0.17 | 0.12 | 0.23 |
| Wegmann | 0.03 | 0.09 | 0.06 | 0.14 |
| StyleDist. | 0.07 | 0.15 | 0.11 | 0.22 |

### Extreme Unfairness Cases

| Model | Dataset | Misattribution Multiplier for the Highest-Risk Author |
|------|--------|----------------------|
| SBERT | Reddit | 39× |
| LUAR | Reddit | 9.75× |
| SBERT | Blogs | 21.75× |
| LUAR | Blogs | 10.0× |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Wegmann Model | Lowest MAUI but also lowest R@8 | Poor performance does not inherently mean more fair $\rightarrow$ but happens to be the case here |
| LUAR | Highest R@8 but also relatively high MAUI | Strong attribution capability $\neq$ misattribution fairness |
| Reddit $> 5 \times E_{10}$ | SBERT: 1599, LUAR: 54 | Severe unfairness in SBERT |

### Key Findings
- **No direct relationship between performance and fairness**: Wegmann has the worst performance but is the most fair; LUAR performs the best but is considerably unfair on Blogs.
- **Proximity to the centroid is highly correlated with misattribution risk**: Across all models and datasets, authors closer to the centroid have higher average rankings (making them more prone to misattribution).
- **Authors closer to the centroid are also harder to identify correctly**: Mann-Whitney tests support the hypothesis that low-MRR authors are closer to the centroid.
- **Most extreme case**: Under SBERT on Reddit, one specific author's risk of misattribution is 39 times higher than the random expectation.

## Highlights & Insights
- **Unique fairness perspective**: While prior work focuses on "finding the correct author," this paper is the first to focus on the risk of "falsely accusing innocent authors."
- **Impact on forensic scenarios**: This issue is particularly critical in legal contexts, where merely being placed on a suspect list can lead to severe consequences.
- **Explanatory power of embedding distribution**: The distance to the centroid effectively explains unfairness, pointing out a direction for model improvement.
- **Simple and effective metric**: The $\text{MAUI}_k$ metric is intuitively defined and easy to calculate, making it directly applicable to evaluating any embed-and-rank attribution system.
- **Counter-intuitive finding**: The "most typical" authors in the embedding space (those closest to the centroid) surprisingly bear the highest risk and are the hardest to correctly identify.

## Limitations & Future Work
- The "fair" baseline assumes a random permutation, which does not account for style/dialect correlations among authors (while authors with the same dialect are naturally more transposable, such confusion is still unfair in forensic settings).
- The study focus is limited to unfairness in over-attribution, without considering under-attribution (failure to identify the correct author).
- The selection of query authors might influence the measurements of unfairness.
- **Future Directions**: Can a "fairness-aware" embedding training strategy be designed to distribute authors more uniformly in the embedding space (pushing them away from the centroid) while maintaining attribution accuracy?

## Related Work & Insights
- Related to fairness in information retrieval: Biega et al. (2020) focus on fair exposure of documents in search results, whereas this work focuses on the risk of "false accusation."
- Complementary to LLM fairness research: Gallegos et al. (2024) survey bias and fairness in LLMs, while this paper focuses on the specific task of authorship attribution.
- Cautionary tale for AI-assisted legal applications: When deploying authorship attribution systems, users must be informed of the misattribution risk instead of just showcasing the correct attribution rate.
- Insight: Any search or matching system based on embedding similarity might possess a similar "near-centroid bias."

## Rating
- Novelty: ⭐⭐⭐⭐ The fairness perspective is brand new to authorship attribution, and the MAUI metric is simple and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Systematic evaluation across 5 models and 3 datasets with solid statistical tests, though lacking mitigation strategies.
- Writing Quality: ⭐⭐⭐⭐ The problem definition is clear, though some mathematical symbols could be simplified.
- Value: ⭐⭐⭐⭐ Important cautionary insights for the responsible deployment of authorship attribution systems; MAUI is ready for direct adoption.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Beyond Last-Click: An Optimal Mechanism for Ad Attribution](../../NeurIPS2025/ai_safety/beyond_last-click_an_optimal_mechanism_for_ad_attribution.md)
- [\[ICLR 2026\] NoisePrints: Distortion-Free Watermarks for Authorship in Private Diffusion Models](../../ICLR2026/ai_safety/noiseprints_distortion-free_watermarks_for_authorship_in_private_diffusion_model.md)
- [\[ICLR 2026\] Watermark-based Detection and Attribution of AI-Generated Content](../../ICLR2026/ai_safety/watermark-based_attribution_of_ai-generated_content.md)
- [\[CVPR 2026\] SAGA: Source Attribution of Generative AI Videos](../../CVPR2026/ai_safety/saga_source_attribution_of_generative_ai_videos.md)
- [\[CVPR 2026\] TokenTrace: Multi-Concept Attribution through Watermarked Token Recovery](../../CVPR2026/ai_safety/tokentrace_multi-concept_attribution_through_watermarked_token_recovery.md)

</div>

<!-- RELATED:END -->

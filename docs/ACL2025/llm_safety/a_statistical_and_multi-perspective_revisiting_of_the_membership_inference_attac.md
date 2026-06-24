---
title: >-
  [Paper Note] A Statistical and Multi-Perspective Revisiting of the Membership Inference Attack in Large Language Models
description: >-
  [ACL 2025][LLM Safety][Membership Inference Attack] This paper comprehensively revisits Membership Inference Attacks (MIA) in LLMs from a statistical perspective through thousands of experiments. It analyzes the inconsistency of MIA performance across six dimensions: data splitting methods, model size, domain characteristics, text features, embedding separability, and decoding dynamics. It reveals previously overlooked findings such as threshold generalization…
tags:
  - "ACL 2025"
  - "LLM Safety"
  - "Membership Inference Attack"
  - "Large Language Models"
  - "Data Privacy"
  - "Statistical Analysis"
  - "Decoding Dynamics"
date: 2026-05-08
content_hash: 5f1256fcc075c35f
---

# A Statistical and Multi-Perspective Revisiting of the Membership Inference Attack in Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2412.13475](https://arxiv.org/abs/2412.13475)  
**Code**: None  
**Area**: LLM / NLP Security  
**Keywords**: Membership Inference Attack, Large Language Models, Data Privacy, Statistical Analysis, Decoding Dynamics

## TL;DR
This paper comprehensively revisits Membership Inference Attacks (MIA) in LLMs from a statistical perspective through thousands of experiments. It analyzes the inconsistency of MIA performance across six dimensions: data splitting methods, model size, domain characteristics, text features, embedding separability, and decoding dynamics. It reveals previously overlooked findings such as threshold generalization, the impact of text length/similarity, and emergent changes in the embedding layers.

## Background & Motivation

**Background**: Membership Inference Attack (MIA) is a core technique for determining whether data has been used to train large language models, which is crucial for data privacy auditing and copyright protection. Existing approaches are mainly divided into gray-box methods (utilizing internal model outputs, such as loss and token probability) and black-box methods (observing generated tokens only). Representative methods include Loss, Min-k% Prob, ReCaLL, SaMIA, etc.

**Limitations of Prior Work**: The performance of MIA methods is highly inconsistent—some studies report promising differentiation, while others find MIA methods to be barely better than random guessing under different setups. For example, Min-k% performs well on WikiMIA, but Duan et al. found that MIA is close to random between the Pile training set and test set. This inconsistency leaves researchers wondering: is MIA actually effective?

**Key Challenge**: Prior studies evaluated MIAs under their respective specific, single settings (single data split, single model, single domain). Different settings may sample member and non-member data with vastly different distribution gaps, leading to contradictory conclusions. Given the massive pre-training corpus of LLMs, different samples can produce member-nonmember pairs with completely different properties, making a single experiment unrepresentative of the entire picture.

**Goal**: Rather than judging the performance of MIA under a single setting, this work aims to comprehensively reveal the statistical patterns of MIA performance across multiple dimensions through large-scale statistical experiments (4860 experiments for each MIA method).

**Key Insight**: Shifts the evaluation of MIA from "the success or failure of a single experiment" to "the analysis of statistical distributions." By combining three data splitting methods × multiple domains × multiple model scales × multiple random seeds, thousands of evaluation settings are generated to plot the probability density distribution of ROC-AUC for each MIA method.

**Core Idea**: Replaces case-by-case validation with statistical analysis, answering "under what conditions and with what probability is MIA effective" from a probability distribution perspective, while deeply analyzing the mechanisms behind MIA performance from the standpoints of embedding separability and decoding dynamics.

## Method

### Overall Architecture
The experimental framework is structured into three layers: (1) Constructing large-scale, multi-setting MIA evaluations—using three splitting methods (Truncate, Complete, Relative), multiple domains (Wikipedia, FreeLaw, GitHub, StackExchange, Pile-CC, etc.), and six model scales (Pythia 160M to 12B) on the Pile dataset to generate evaluation settings; (2) Statistical analysis layer—performing probability density analysis, outlier statistics, and threshold generalization testing on the ROC-AUC distribution for each MIA method; (3) In-depth analysis layer—unearthing the deep-seated mechanisms of MIA performance from three angles: text features, embedding separability, and decoding entropy dynamics.

### Key Designs

1. **Multi-Setting Statistical Evaluation Framework**:

    - **Function**: Constructs thousands of experiments for each MIA method to obtain statistically reliable performance distributions.
    - **Mechanism**: Designs three data splitting methods to cover different member-nonmember construction approaches. Truncate Split truncates texts to fixed length ranges; Complete Split selects texts whose full lengths fall within the target range; Relative Split samples based on the decile length distribution of each domain's test set. Each split is applied to all domains of the Pile, combined with six Pythia model sizes and three random seeds, yielding approximately 4860 independent experiments for each MIA method. The probability density functions of ROC-AUC, rather than a single value, are computed over these experiments.
    - **Design Motivation**: The inconsistency in previous work stems precisely from the limitations of a single setup. By enumerating combinations of settings across multiple dimensions, one can acquire a "panoramic view" of MIA methods, distinguishing between universal laws and accidental phenomena under specific configurations.

2. **MIA Outlier Analysis and Method Consistency Testing**:

    - **Function**: Analyzes the distribution of high-performance outliers (ROC-AUC > 0.55) to understand the scenarios in which different MIA methods are applicable.
    - **Mechanism**: Counts the number and distribution characteristics of high-performance outliers outside the main body of the probability density. It further computes the outlier overlap matrix of different MIA methods—if the overlap of high-performance settings between Method A and Method B is low, it indicates they function under different scenarios. The results show that Min-k%++ has the most outliers (410), but its peak ROC-AUC is not the highest; ReCaLL has fewer outliers but achieves the highest peak value of 0.806.
    - **Design Motivation**: Even if the average performance of MIA is close to random, the existence of outliers explains previous positive results—they might have been evaluated precisely under the configurations corresponding to these outliers. Outlier analysis unifies positive and negative results under a single framework.

3. **Embedding Separability and Decoding Entropy Dynamics Analysis**:

    - **Function**: Explains the mechanism of MIA performance from the perspective of LLM internal representations.
    - **Mechanism**: For embedding analysis, the average-pooled hidden states of members and non-members are collected at each Transformer layer. The Davies-Bouldin Score (DB Score) is used to measure the separability of the two classes of embeddings, and a Transformer classifier is trained for validation. It is found that domain-model combinations with high MIA performance indeed exhibit better separability in middle-layer embeddings, but the separability drops sharply in the final layer—whereas existing MIA methods depend precisely on the final layer output. For decoding dynamics, the token decoding entropy of members and non-members and their cumulative differences are calculated step-by-step, showing that domains with high MIA performance (such as FreeLaw) exhibit a faster growth rate of cumulative entropy differences.
    - **Design Motivation**: It is crucial to know not only whether MIA works, but also "why it works/fails." Embedding analysis reveals a structural reason for poor MIA performance—low separability in the final layer; decoding dynamics analysis connects MIA to the generation process of LLMs.

### Loss & Training
Since this is an analytical work, no new models are trained. Pre-trained Pythia models (160M to 12B) are evaluated on the deduplicated Pile dataset. ROC-AUC is used as the primary evaluation metric, the Geometric Mean method is adopted for threshold selection experiments, and the Spearman correlation coefficient is utilized for correlation analysis between text features and MIA performance.

## Key Experimental Results

### Main Results

| MIA Method | Number of Outliers | Max ROC-AUC | Average ROC-AUC | Type |
|----------|-----------|-------------|-------------|------|
| Min-k%++ | 410 | 0.631 | 0.564 | Gray-box |
| SaMIA | 218 | 0.647 | 0.569 | Black-box |
| Gradient | 160 | 0.631 | 0.563 | Gray-box |
| Zlib | 130 | 0.590 | 0.562 | Gray-box |
| Min-k% | 127 | 0.600 | 0.562 | Gray-box |
| ReCaLL | 127 | 0.806 | 0.572 | Gray-box |
| Loss | 110 | 0.585 | 0.561 | Gray-box |
| Refer | 70 | 0.572 | 0.559 | Gray-box |
| CDD | 63 | 0.604 | 0.561 | Black-box |
| DC-PDD | 43 | 0.575 | 0.558 | Gray-box |
| PAC | 20 | 0.573 | 0.557 | Gray-box |

### Ablation Study

| Analysis Dimension | Key Findings | Practical Impact |
|----------|---------|---------|
| Cross-domain Threshold Transfer | Optimal thresholds vary significantly across domains | Thresholds from one domain cannot be directly transferred to another |
| Cross-model Scale Threshold Transfer | Thresholds change systematically with model scale | Calibration is needed for each model scale |
| Within-domain Threshold Stability | Significant outliers still exist within the same domain | Stability is not guaranteed even within a restricted domain |
| Correlation with Text Length | Longer text is positively correlated with MIA performance (Spearman mean of 0.16) | MIA is more unreliable on short texts |
| Correlation with Text Similarity | The larger the discrepancy between member and non-member texts, the more effective MIA is (mean of -0.19) | MIA partially detects text discrepancies rather than training membership status |

### Key Findings
- MIA performance improves as the model scale increases (especially with a significant leap between 1B and 2.8B), which contradicts previous conclusions that "larger models are harder to attack." The authors explain that under-trained small models cause members and non-members to behave similarly, medium models begin to distinguish them, and extremely large models may generalize again.
- Embedding space analysis reveals an "emergent" phenomenon: in the 2.8B model, embeddings of previously inseparable domains (such as PubMed, Pile-CC) suddenly become separable, explaining the leap in ROC-AUC between 1B and 2.8B.
- Embedding separability in the final layer is surprisingly lower than that in the middle layers, whereas all current MIA methods rely on the final layer output. This implies that leveraging middle-layer features could improve MIA performance.
- The overlap of outliers among different MIA methods is low (only 4% overlap between Min-k%++ and CDD), indicating that each method is effective under different scenarios, and no single model handles all scenarios.
- The Relative Split consistently outperforms the Truncate Split, as truncation may discard highly discriminative outlier tokens.

## Highlights & Insights
- The experimental design from a statistical perspective is the biggest highlight. Shifting the evaluation of MIA from single values to the level of probability distributions, and depicting a complete performance profile for each method using 4860 experiments, allows positive and negative results to be unified and explained under a single framework.
- The discovery of embedding emergence provides a new mechanistic explanation for MIA: once the model scale crosses a certain threshold, a qualitative change occurs in the structural embedding space, making members and non-members separable. This aligns with research on emergent abilities in LLMs and points to new analytical directions for MIA research.
- The finding of low final-layer embedding separability has direct practical implications—future MIA methods should consider incorporating middle-layer features instead of relying solely on final-layer outputs.

## Limitations & Future Work
- Experiments are conducted only on the Pythia family (up to 12B), which cannot verify if a quadratic decline in MIA performance occurs on larger LLMs (e.g., 70B, 175B).
- Pythia is one of the few models with publicly released pre-training data; experimental conclusions may not directly apply to closed-source LLMs.
- The Geometric Mean method is used for threshold selection, and alternative thresholding strategies might yield different conclusions.
- The performance of MIA on fine-tuned models or models aligned with RLHF is not analyzed.
- Black-box methods (SaMIA, CDD) are computationally expensive (taking ~20 days per model scale), limiting the scope of the experiments.

## Related Work & Insights
- **vs Min-k% Prob (Shi et al., 2024)**: Min-k% performs well on WikiMIA but is not statistically significantly better than the Loss baseline; its improved version, Min-k%++, achieves more stable improvements through standardization.
- **vs ReCaLL (Xie et al., 2024)**: By adding a non-member prefix to perturb likelihoods, ReCaLL achieves the highest peak performance (0.806) despite having fewer outliers, offering a unique advantage in specific scenarios.
- **vs Duan et al. (2024) (Negative Result Paper)**: Duan used Truncate Split on the Pile and found near-random MIA performance. This paper confirms that Truncate Split is indeed the worst splitting method, explaining the source of their negative conclusion while also establishing non-trivial MIA signals through the Relative Split.

## Rating
- Novelty: ⭐⭐⭐⭐ The statistical analysis framework is novel, and the discovery of embedding emergence is original, though most individual analyses are not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Thousands of experiments cover multiple dimensions; the analysis is comprehensive and systematic, with hypothesis testing and memorization score analyses provided in the appendix.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, each analysis leads to definite conclusions, and figures/tables feature high information density.
- Value: ⭐⭐⭐⭐ Highly valuable reference for the MIA community, reconciling previously contradictory positive and negative findings, while guiding the design of future MIA methods.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Exploring the Limits of Strong Membership Inference Attacks on Large Language Models](../../NeurIPS2025/llm_safety/exploring_the_limits_of_strong_membership_inference_attacks_on_large_language_mo.md)
- [\[ACL 2025\] Improving Fairness of Large Language Models in Multi-document Summarization](improving_fairness_of_large_language_models_in_multi-document_summarization.md)
- [\[ICLR 2026\] Membership Inference Attacks Against Fine-tuned Diffusion Language Models (SAMA)](../../ICLR2026/llm_safety/membership_inference_attacks_against_fine-tuned_diffusion_language_models.md)
- [\[ACL 2025\] Ensemble Watermarks for Large Language Models](ensemble_watermarks_llm.md)
- [\[ACL 2026\] Membership Inference Attacks on In-Context Learning Recommendation](../../ACL2026/llm_safety/membership_inference_attacks_on_llm-based_recommender_systems.md)

</div>

<!-- RELATED:END -->

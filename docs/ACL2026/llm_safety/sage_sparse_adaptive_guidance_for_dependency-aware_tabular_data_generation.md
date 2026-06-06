---
title: >-
  [Paper Note] SAGE: Sparse Adaptive Guidance for Dependency-Aware Tabular Data Generation
description: >-
  [ACL2026][LLM Safety][Tabular data generation] SAGE discretizes tabular features into value-aware pseudo-features and constructs a sparse dynamic dependency graph using mutual information to guide LLM generation…
tags:
  - "ACL2026"
  - "LLM Safety"
  - "Tabular data generation"
  - "sparse dependencies"
  - "mutual information"
  - "dynamic guidance"
  - "synthetic data quality"
date: 2026-05-08
content_hash: 4f6ff64dc722592e
---

# SAGE: Sparse Adaptive Guidance for Dependency-Aware Tabular Data Generation

**Conference**: ACL2026  
**arXiv**: [2604.24368](https://arxiv.org/abs/2604.24368)  
**Code**: https://github.com/ShuoYangtum/SAGE  
**Area**: Synthetic Tabular Data / LLM Data Generation / Dependency Modeling  
**Keywords**: Tabular data generation, sparse dependencies, mutual information, dynamic guidance, synthetic data quality

## TL;DR
SAGE discretizes tabular features into value-aware pseudo-features and constructs a sparse dynamic dependency graph using mutual information to guide LLM generation, thereby enhancing the downstream utility, constraint consistency, and realism of synthetic tabular data.

## Background & Motivation
**Background**: Synthetic tabular data is crucial for privacy-sensitive or low-resource scenarios in healthcare, finance, and education. Traditional methods like TVAE, CTGAN, and diffusion models primarily learn numerical matrix distributions. Recently, LLM-based methods have emerged, converting table rows into "feature is value" text sequences to leverage the language model’s semantic knowledge for generating more reasonable records.

**Limitations of Prior Work**: When generating tabular rows, LLMs typically take all historical feature-value pairs as context and rely on dense attention to capture relationships. This introduces spurious correlations between irrelevant features, leading to logical inconsistencies or performance degradation in downstream models. Furthermore, existing explicit dependency modeling methods mostly use static feature graphs, which fail to express the phenomenon where "dependency relationships change when the same feature takes different values."

**Key Challenge**: Tabular data possesses both sparse structures and conditional dynamics. For instance, when the loan purpose is education versus home purchase, the correlations between age, income, and occupational stability differ significantly. Using only static graphs ignores value-conditioned dependencies; relying entirely on LLM dense attention can result in being misled by surface-level co-occurrences.

**Goal**: To construct an LLM-based tabular generation framework that ensures the model focuses only on truly relevant and value-adaptive context when generating each target feature, while avoiding a significant increase in inference costs.

**Key Insight**: The authors expand original features into value-aware pseudo-features and estimate statistical dependencies between these pseudo-features using mutual information. In this way, the dependency graph no longer merely describes "Feature A is related to Feature B," but rather "a specific value range of Feature A is related to the generation of Feature B."

**Core Idea**: An MI-driven sparse dynamic dependency graph is used during the sampling phase to allow the LLM to adaptively generate tabular records based on current feature values through explicit context filtering or implicit logit correction.

## Method

### Overall Architecture
SAGE consists of two stages. In the pre-processing stage, tabular data is converted into text sequences for continued pre-training of the LLM to learn the verbalized distribution of feature-values. Simultaneously, numerical and categorical features are discretized into pseudo-features, and a mutual information matrix is estimated based on the training set. In the generation stage, starting from a partial real feature-value prefix, remaining features are completed autoregressively. At each step, the MI graph controls the context or output confidence.

The paper proposes two complementary guidance strategies: the Feature Selector is an explicit strategy that directly removes context with low MI relative to the target feature; Logit Correction is an implicit strategy that does not remove context but adjusts candidate value logits based on the information content of the current prefix. Both aim to prevent the LLM from being swayed by irrelevant feature-value pairs during generation.

### Key Designs

1. **Value-aware pseudo-feature discretization**:

    - **Function**: Converts original features into binary pseudo-features capable of expressing value ranges or categorical conditions.
    - **Mechanism**: Numerical features automatically determine the number of bins via the Freedman-Diaconis rule (with a cap of 16 to control sparsity); categorical features treat each category as a pseudo-feature. Each record is eventually mapped to a set of activated binary pseudo-features.
    - **Design Motivation**: Static feature graphs only describe feature-level relationships and cannot express "a specific value changing the dependency structure." Pseudo-features lower the granularity of dependency modeling to the value-level, enabling MI to capture conditional correlations.

2. **Mutual Information Sparse Dependency Graph**:

    - **Function**: Estimates which feature-value pairs in the already generated context are truly informative for a target feature.
    - **Mechanism**: Mutual information is calculated between the binary activations of any two pseudo-features to form a dependency matrix. Since probability estimation is based on pseudo-feature activation rather than raw numerical scales, the method handles both numerical and categorical variables uniformly.
    - **Design Motivation**: MI provides a lightweight, interpretable, and unsupervised dependency measure that can filter out irrelevant edges in dense attention.

3. **Feature Selector and Logit Correction**:

    - **Function**: Integrates the MI graph into the LLM sampling process.
    - **Mechanism**: The Feature Selector retains only prefix pseudo-features with MI higher than a threshold (defaulting to the training set MI median) for the target feature. Logit Correction calculates the average MI of the current prefix for the target feature and compares it with the training set average; if the prefix information is high, target logits are sharpened, otherwise, the output distribution is smoothed.
    - **Design Motivation**: Explicit filtering is suitable for high-dimensional tables with heavy noise and sparse dependencies; implicit correction is better for scenarios with continuous dependencies where removing context might lose information. Together, they provide control mechanisms switchable based on data characteristics.

### Loss & Training
The training phase follows the GReaT-style LLM tabular modeling: each row is written as multiple "feature is value" phrases, optimizing the negative log-likelihood of value-related tokens. The authors also use the permutation strategy from GraDe, randomly shuffling the order of feature-value phrases to reduce spurious dependencies caused by fixed column orders. In the experimental setup, the batch size is 8, using the AdamW optimizer with a learning rate of 1e-4. Sampling employs nucleus sampling with $p=0.95$, a temperature of 1.0, and the maximum generation length is set to the maximum sequence length in the training set.

## Key Experimental Results

### Main Results

Experiments cover six datasets: Adult Income, HELOC, Iris, Diabetes, MIC, and California Housing, involving binary classification, multi-class classification, and regression. The authors generate synthetic data of the same scale as the original data, then train downstream models (DT/RF, etc.) and evaluate them on real test sets.

| Dataset / Metric | GReaT | GraDe | SPADA | SAGE w/FS | SAGE w/LC | Key Observation |
|--------|------|------|------|------|------|------|
| Adult Income, DT F1 ↑ | 0.60 | 0.55 | 0.50 | 0.68 | 0.72 | LC is 12 points higher than GReaT |
| Adult Income, RF F1 ↑ | 0.69 | 0.63 | 0.75 | 0.75 | 0.76 | Both SAGE variants significantly outperform GReaT |
| HELOC, DT F1 ↑ | 0.61 | 0.67 | 0.61 | 0.68 | 0.69 | Dynamic dependency consistently improves credit data |
| Iris, RF ACC ↑ | 44.83 | 100.00 | 100.00 | 100.00 | 100.00 | SAGE avoids GReaT's overfitting collapse on small data |
| California Housing, RF MAPE ↓ | 0.26 | 0.23 | 0.25 | 0.25 | 0.40 | FS is more stable for regression; LC is more conservative |

### Ablation Study

| Configuration / Analysis | Key Metric | Description |
|------|---------|------|
| Feature Selector | Adult education-consistency violation 1.32% | Explicit context filtering is particularly suited for rules depending on a few precise attributes |
| Logit Correction | Housing violation approx. 1 point lower than GReaT | Implicit regulation is friendlier to spatially continuous constraints |
| MI Threshold | Performance stable across a wide range of thresholds | The method does not rely entirely on a fragile threshold |
| Different base LLMs | GPT-2, Qwen-3, Llama-3 maintain similar trends | SAGE's dependency guidance is not specific to a single model |
| Preprocessing vs Sampling Cost | MI calculation is a one-time overhead; generation benefits from sparse context | Suitable for front-loading costs into data-level preprocessing |

### Key Findings
- In terms of downstream utility, SAGE outperforms GReaT on almost all tasks, with F1 improvements exceeding 10 points on the Adult dataset, indicating that MI guidance can mitigate surface-pattern overfitting in LLM tabular generation.
- Regarding constraint consistency, SAGE-generated California Housing samples rarely fall outside actual state boundaries, whereas TVAE and CTGAN struggle to reconstruct such complex spatial contours.
- The two guidance strategies are complementary: the Feature Selector is better at removing high-dimensional noise and precise semantic rule errors, while Logit Correction is better at smoothing continuous spatial dependencies.
- On HELOC, Logit Correction occasionally suppresses useful signals, suggesting that "underestimating context MI" can make implicit correction overly conservative.

## Highlights & Insights
- The paper advances the problem of LLM tabular generation from "verbalized row modeling" to "value-conditioned dependency control." This is closer to the structural essence of tabular data than simply serializing table rows.
- The pseudo-feature design is practical: it allows numerical features to have discrete statistical representations while preserving the natural value structure of categorical features, avoiding the need to train complex additional graph models.
- The dual design of Feature Selector and Logit Correction has engineering value. The former is an interpretable hard filter, while the latter is a flexible probabilistic adjustment, suiting different dependency forms across datasets.
- The paper does not only report downstream classification scores but also evaluates violations, SVM realism, DCR privacy, and visual distributions, making the evaluation of synthetic data quality more comprehensive.

## Limitations & Future Work
- The dependency graph is primarily based on pairwise mutual information and cannot explicitly model higher-oder relationships where multiple features jointly influence a target feature. Autoregressive LLMs can partially compensate, but direct control over higher-order structures is still missing.
- Pre-processing the MI matrix on high-dimensional data can be heavy; although it is a one-time cost, it still requires sparsification or approximate estimation when the number of features and pseudo-features is extremely large.
- MI estimation depends on the statistical quality of the training split; small samples or long-tail categories may introduce estimation noise affecting context filtering.
- Results show that different strategies suit different constraint types; future work could investigate an adaptive hybrid of FS and LC rather than manual selection.
- Privacy evaluation is mainly based on DCR; stronger membership inference, attribute inference, or differential privacy perspectives are still needed to verify reliability for deployment in sensitive domains.

## Related Work & Insights
- **vs GReaT**: GReaT proved that LLMs can generate realistic tabular rows, but context modeling is flat; SAGE adds MI guidance on top of this to reduce interference from irrelevant feature-value pairs.
- **vs GraDe / SPADA**: GraDe and SPADA have already emphasized structural dependencies but lean towards static structures; SAGE's key difference is that dependencies change dynamically with current values.
- **vs TVAE / CTGAN / TabSyn**: Traditional generative models are good at learning distribution shapes but struggle to utilize feature semantics; SAGE leverages the semantic priors of LLMs while constraining them with statistical dependencies.
- **Insight**: For LLM generation of structured data, one should not just design prompts or sequence templates, but also explicitly control "which fields should be looked at when generating a specific field." This point is transferable to knowledge graph completion, form auto-filling, and semi-structured document synthesis.

## Rating
- Novelty: ⭐⭐⭐⭐☆ The combination of value-aware pseudo-features and MI guidance is natural; the innovation lies in structural control rather than the generative model itself.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Six datasets, multiple metrics, and various baselines provide comprehensive coverage; though privacy and ultra-high-dimensional scaling could be deeper.
- Writing Quality: ⭐⭐⭐⭐☆ The logic of the methodology is clear; despite the amount of tabular data, the main thread remains distinct.
- Value: ⭐⭐⭐⭐☆ Practical for generating low-resource and privacy-sensitive tabular data, and provides an interpretable control mindset for LLM-based structured data generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Resource-Adaptive Federated Text Generation with Differential Privacy](../../ICLR2026/llm_safety/resource-adaptive_federated_text_generation_with_differential_privacy.md)
- [\[ACL 2026\] AGSC: Adaptive Granularity and Semantic Clustering for Uncertainty Quantification in Long-text Generation](agsc_adaptive_granularity_and_semantic_clustering_for_uncertainty_quantification.md)
- [\[ACL 2026\] ATAAT: Adaptive Threat-Aware Adversarial Tuning Framework against Backdoor Attacks on Vision-Language-Action Models](ataat_adaptive_threat-aware_adversarial_tuning_framework_against_backdoor_attack.md)
- [\[ACL 2026\] CRISP: Persistent Concept Unlearning via Sparse Autoencoders](crisp_persistent_concept_unlearning_via_sparse_autoencoders.md)
- [\[AAAI 2026\] AgentSense: Virtual Sensor Data Generation Using LLM Agents in Simulated Home Environments](../../AAAI2026/llm_safety/agentsense_virtual_sensor_data_generation_using_llm_agents_i.md)

</div>

<!-- RELATED:END -->

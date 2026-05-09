---
title: >-
  [Paper Note] Language Model Behavioral Phases are Consistent Across Architecture, Training Data, and Scale
description: >-
  [NeurIPS 2025][LLM Pretraining][language model behavioral phases] By analyzing over 1,400 model checkpoints on 110,000+ tokens, this paper demonstrates that autoregressive language models exhibit highly consistent behavioral phases during training — predicted probabilities successively overfit to n-gram distributions of increasing $n$ — and that three simple heuristics (word frequency, n-gram probability, and semantic similarity) explain up to 98% of the variance in model behavior. This pattern holds consistently across architectures (Transformer/Mamba/RWKV), datasets, and scales.
tags:
  - NeurIPS 2025
  - LLM Pretraining
  - language model behavioral phases
  - n-gram probability
  - semantic similarity
  - training dynamics
  - architecture-agnostic
date: 2026-05-08
content_hash: 8133cffc4856dcf8
---

# Language Model Behavioral Phases are Consistent Across Architecture, Training Data, and Scale

**Conference**: NeurIPS 2025
**arXiv**: [2510.24963](https://arxiv.org/abs/2510.24963)
**Code**: [GitHub](https://github.com/jmichaelov/lm-behavioral-phases)
**Area**: LLM Pre-training / Interpretability
**Keywords**: language model behavioral phases, n-gram probability, semantic similarity, training dynamics, architecture-agnostic

## TL;DR
By analyzing over 1,400 model checkpoints on 110,000+ tokens, this paper demonstrates that autoregressive language models exhibit highly consistent behavioral phases during training — predicted probabilities successively overfit to n-gram distributions of increasing $n$ — and that three simple heuristics (word frequency, n-gram probability, and semantic similarity) explain up to 98% of the variance in model behavior. This pattern holds consistently across architectures (Transformer/Mamba/RWKV), datasets, and scales.

## Background & Motivation
1. **Background**: Language models trained via next-token prediction exhibit emergent capabilities such as grammatical generation and knowledge-based reasoning, yet the regularities underlying the learning process remain poorly understood.
2. **Limitations of Prior Work**: Existing analyses have largely focused on abrupt changes in specific behaviors or subnetworks, lacking a systematic characterization of overall model behavior across training.
3. **Key Challenge**: It remains unclear whether universal learning regularities exist that are independent of model-specific details such as architecture, scale, and data.
4. **Goal**: To quantitatively characterize the behavioral changes of language models throughout training using simple heuristics.
5. **Key Insight**: The analysis focuses on three heuristics — word frequency (unigram), n-gram probability, and contextual semantic similarity.
6. **Core Idea**: All models undergo the same behavioral phases: initial overfitting to low-order n-grams, progressive overfitting to higher-order n-grams, and early rapid establishment of correlation with semantic similarity.

## Method

### Overall Architecture
A total of 1,418 model checkpoints are collected (Pythia/Mamba/RWKV × multiple scales × multiple seeds) and evaluated on the decontaminated benchmark NaWoCo. The Pearson/Spearman correlations and regression analyses between model log-probabilities and each heuristic are computed across over 110,000 tokens.

### Key Designs
1. **Parc Model Suite**: The first publicly released checkpointed Mamba-1 and RWKV-4 models, with all three architectures trained in parallel on the same OpenWebText data (identical sequences and training steps), using 6 random seeds and 73 checkpoints each. A shared tokenizer ensures fair comparison.
2. **NaWoCo Dataset**: An evaluation set of 150,000+ words in sentential contexts, extracted from FineWeb. Tokens are required to be single tokens (valid across all models), absent from training data (verified via infini-gram counts), and low-toxicity (probability < 0.1). The dataset is split into train/validation/test sets.
3. **Regression Analysis**: Unigram, 2–5-gram log-probabilities, and fastText semantic similarity scores (Wikipedia and Common Crawl variants, with uniform and SGPT-based weighting) are used as features to predict model log-probabilities; $R^2$ is computed to measure explained variance.
4. **n-gram Computation**: Word-level n-gram probabilities are computed over training data using the infini-gram toolkit with Stupid Backoff smoothing.

### Loss & Training
- This is a purely analytical study; no training loss is designed.
- Pearson correlation, Spearman correlation, and $R^2$ regression analysis are employed.
- Model scales range from 14M to 12B parameters (the Pythia suite covers the full range).

## Key Experimental Results

| Finding | Details |
|---------|---------|
| Explained variance | Three heuristics explain up to **98%** of model log-probability variance |
| Cross-architecture consistency | Pythia/Mamba/RWKV achieve Pearson $r \geq 0.93$ at matched training steps |
| Behavioral phases | Sequential overfitting: unigram → bigram → trigram → ... → 5-gram |
| Scale effect | Larger models show stronger decorrelation from low-order n-grams |

### Key Findings
- All models, regardless of architecture, scale, or data, exhibit the same n-gram overfitting sequence.
- The peak correlation with semantic similarity co-occurs with unigram (Common Crawl variant) or trigram (Wikipedia variant) correlations.
- Variance across random seeds is negligible (confidence intervals are nearly invisible).
- Larger models are better able to disentangle from low-order n-grams and learn more complex relationships.

### Parc Model Suite Details

| Architecture | Parameters | Training Data | Checkpoints | Seeds |
|-------------|-----------|--------------|------------|-------|
| Pythia | 14M–12B | The Pile | 143 | 1 |
| Mamba-1 | ~160M | OpenWebText | 73 | 6 |
| RWKV-4 | ~160M | OpenWebText | 73 | 6 |

### Behavioral Phase Timeline
- Phase 1 (0–5K steps): Unigram overfitting; the model learns word frequency distributions.
- Phase 2 (5K–20K steps): Bigram overfitting; local dependencies begin to be captured.
- Phase 3 (20K–100K steps): Trigram+ overfitting; longer-range dependencies are learned.
- Phase 4 (100K+ steps): Decorrelation from high-order n-grams; semantic relationships begin to emerge.

## Highlights & Insights
- The work uncovers a rare cross-architecture universal regularity in deep learning.
- Three minimalist heuristics explain 98% of variance — suggesting that language models fundamentally learn these three types of patterns.
- The Parc model suite and NaWoCo dataset constitute important public resources for the community.
- The findings offer a new perspective for understanding scaling laws and emergent behaviors.

## Limitations & Future Work
- The analysis is restricted to token-level behavior and does not extend to sentence- or paragraph-level semantic analysis.
- Simple heuristics may be insufficient to explain more complex behaviors such as multi-step reasoning or planning.
- Behavioral phase changes following instruction fine-tuning or RLHF are not examined; alignment training may alter phase ordering.
- The causal mechanism underlying these phases remains unexplained; the findings are observational rather than mechanistic.
- The 98% explained variance may overestimate the importance of the heuristics, given the strong inherent statistical regularities of natural language.
- The effect of varying training data composition (e.g., proportion of code data) on behavioral phases is not explored.
- The Mamba and RWKV models evaluated are relatively small; behavioral phases in larger-scale non-Transformer architectures may differ.
- The construction of the NaWoCo evaluation set may introduce selection bias — restricting to single-token words may not be representative of more complex lexical categories.

## Related Work & Insights
- **vs. Chang et al. 2024**: n-gram overfitting was previously observed only in GPT-2; this work extends the finding to multiple architectures and scales.
- **vs. Voita et al. 2024**: That work analyzes n-gram-specialized neurons; this work characterizes the same phenomenon at the behavioral level.
- **vs. Schaeffer et al. 2023**: That work argues emergence is a measurement artifact; this work approaches training dynamics from a complementary perspective.

### Additional Discussion
- The core methodological contribution lies in transforming a unidimensional problem into a multi-heuristic analysis, yielding a more comprehensive understanding.
- The experimental design covers diverse settings and baseline comparisons, with statistically significant results.
- The modular design of the method facilitates extension to related tasks and new datasets.
- Open-sourcing of code and data is of significant value for community reproduction and follow-up research.
- Compared to contemporaneous work, this paper demonstrates greater depth in problem formulation and comprehensiveness in experimental analysis.
- The paper is logically structured, forming a complete loop from problem definition through method design to experimental validation.
- The computational overhead of the method is reasonable, making it deployable in practical settings.
- Future work may consider integration with additional modalities (e.g., audio, 3D point clouds).
- Validating scalability on larger data and models is an important next step.
- Combining the approach with reinforcement learning for end-to-end optimization is worth exploring.
- Cross-domain transfer is a direction warranting further investigation — the generality of the method requires broader validation.
- Lightweight variants of the method for edge computing and mobile deployment deserve attention.
- Long-term evaluation and user studies would provide a more comprehensive assessment.
- Comparative analysis with human experts would better delineate the strengths and limitations of the approach.
- Robustness testing under adversarial conditions is a necessary step prior to real-world deployment.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — The discovery of universal behavioral phases across architectures is a significant scientific contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Large-scale experiments spanning 1,400+ checkpoints, 3 architectures, and multiple scales.
- **Writing Quality**: ⭐⭐⭐⭐ — Rigorous analysis with clear visualizations.
- **Value**: ⭐⭐⭐⭐⭐ — Fundamental implications for understanding the learning mechanisms of language models.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Nemotron-CLIMB: CLustering-based Iterative Data Mixture Bootstrapping for Language Model Pre-training](nemotron-climb_clustering-based_iterative_data_mixture_bootstrapping_for_languag.md)
- [\[NeurIPS 2025\] Disaggregation Reveals Hidden Training Dynamics: The Case of Agreement Attraction](disaggregation_reveals_hidden_training_dynamics_the_case_of_agreement_attraction.md)
- [\[NeurIPS 2025\] Through the River: Understanding the Benefit of Schedule-Free Methods for Language Model Training](through_the_river_understanding_the_benefit_of_schedule-free_methods_for_languag.md)
- [\[NeurIPS 2025\] Gradient-Weight Alignment as a Train-Time Proxy for Generalization in Classification Tasks](gradient-weight_alignment_as_a_train-time_proxy_for_generalization_in_classifica.md)
- [\[NeurIPS 2025\] Final-Model-Only Data Attribution with a Unifying View of Gradient-Based Methods](final-model-only_data_attribution_with_a_unifying_view_of_gradient-based_methods.md)

<!-- RELATED:END -->

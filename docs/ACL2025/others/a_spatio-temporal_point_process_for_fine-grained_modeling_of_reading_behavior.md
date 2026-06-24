---
title: >-
  [Paper Note] A Spatio-Temporal Point Process for Fine-Grained Modeling of Reading Behavior
description: >-
  [ACL 2025][Eye-tracking] This paper proposes a unified probabilistic model of reading behavior based on marked spatio-temporal point processes. It simultaneously models when and where fixations occur and how long they last, avoiding the information loss associated with traditional aggregated measures, and reveals that surprisal has an extremely limited contribution to predicting fine-grained eye movements.
tags:
  - "ACL 2025"
  - "Eye-tracking"
  - "Hawkes process"
  - "Spatio-temporal point process"
  - "surprisal theory"
  - "Reading behavior modeling"
date: 2026-05-08
content_hash: 769fb7c68317c680
---

# A Spatio-Temporal Point Process for Fine-Grained Modeling of Reading Behavior

**Conference**: ACL 2025  
**arXiv**: [2506.19999](https://arxiv.org/abs/2506.19999)  
**Code**: [GitHub](https://github.com/rycolab/spatio-temporal-reading)  
**Area**: Computational Psycholinguistics / NLP Understanding  
**Keywords**: Eye-tracking, Hawkes process, Spatio-temporal point process, surprisal theory, Reading behavior modeling

## TL;DR

This paper proposes a unified probabilistic model of reading behavior based on marked spatio-temporal point processes. It simultaneously models when and where fixations occur and how long they last, avoiding the information loss associated with traditional aggregated measures, and reveals that surprisal has an extremely limited contribution to predicting fine-grained eye movements.

## Background & Motivation

**Background**: In computational psycholinguistics, eye-tracking experiments are a core paradigm for studying human language processing. The standard practice is to aggregate raw scanpath data into word-level reading time metrics (e.g., first fixation duration, gaze duration, total fixation duration) and analyze them using linear mixed-effects models.

**Limitations of Prior Work**: The aggregation process loses a substantial amount of information. Temporally, combining multiple fixations into a single metric confounds different cognitive processes (e.g., first fixation and regressive fixation correspond to different mechanisms). Spatially, aggregation relies on predefined regions of interest (typically word-bound), discarding information about the exact position of fixations within a word and hindering the study of smaller linguistic units (such as syllables and morphemes).

**Key Challenge**: The choice of aggregation strategy directly impacts the conclusions of theoretical validation. For instance, surprisal theory predicts that contextual predictability has a stronger effect on gaze duration, but empirical studies find a larger effect on total fixation time instead. Such counterintuitive results could be artifacts of the aggregation methods rather than true cognitive effects, but causal verification is impossible using aggregated data.

**Goal**: (1) How to unifiedly model the temporal, spatial, and duration dimensions of fixations without information loss? (2) Does the effect size of predictors like surprisal on raw scanpaths align with those found in aggregated data?

**Key Insight**: The authors formalize the reading process as a marked spatio-temporal point process. The alternating sequence of fixations and saccades naturally fits the point process framework. Utilizing the self-exciting property of Hawkes processes allows for capturing the spatio-temporal influence of previous fixations on subsequent ones.

**Core Idea**: A unified marked spatio-temporal point process is constructed by modeling the spatio-temporal distribution of saccades with a Hawkes process and the duration of fixations with a log-normal distribution, directly modeling raw scanpaths.

## Method

### Overall Architecture

The input is a complete scanpath $\mathcal{T} = \{(t_n, \mathbf{s}_n, d_n)\}_{n=1}^N$, where $t_n$ is the fixation start time, $\mathbf{s}_n$ is the spatial coordinate on the screen, and $d_n$ is the fixation duration. The model comprises two components: (1) a spatio-temporal Hawkes process for modeling when and where a fixation occurs, and (2) a log-normal distribution for modeling how long the fixation lasts. Generating a scanpath involves iterative sampling: first sampling the time and location of the next fixation, then sampling its duration, updating the history, and repeating this process until a time limit is reached.

### Key Designs

1. **Saccade planning modeled by a spatio-temporal Hawkes process**:

    - **Function**: Modeling when the next fixation occurs and where it lands on the screen
    - **Mechanism**: The intensity function is $\lambda(t_n, \mathbf{s}_n; \mathcal{H}_{n-1}) = \nu + \sum_{m=1}^{n-1} \phi_m(t_n - t_m - \delta(n,m)) \psi_m(\mathbf{s}_n)$, where $\nu$ is the base intensity, $\phi_m$ is an exponentially decaying temporal kernel (controlling the temporal influence of historical fixations), and $\psi_m$ is a 2D Gaussian spatial density centered on the location of historical fixations. The excitation strength and decay rate of the temporal kernel are parameterized by a linear combination of fixation-level predictors, allowing for different self-exciting behaviors under different conditions. The spatial density supports three center transformations: identity transformation (baseline), affine transformation (learning a constant displacement for the left-to-right reading direction), and a full transformation with predictors.
    - **Design Motivation**: The self-exciting structure of the Hawkes process naturally aligns with reading behavior—a prior fixation "excites" subsequent fixations to occur in its vicinity, and the cumulative contributions of multiple historical fixations form a multimodal distribution, accurately corresponding to behaviors like progressive saccades, regressions, and refixations in human reading.

2. **Fixation durations modeled by log-normal convolutions**:

    - **Function**: Predicting how long each fixation lasts
    - **Mechanism**: The fixation duration is modeled by a log-normal distribution $g(d_n | \mathcal{H}_{n-1}, t_n)$, where its log-mean $\xi_n(t_n)$ integrates the spillover effects of historical predictors via convolution. Specifically, $\xi_n^c(t_n) = \mathbf{x}_n^\top \mathbf{w} + \sum_{k \in K} w'_k \sum_{m=1}^{n-1} x_{mk} \gamma(t_n - t_m | \alpha_k, \beta_k, \theta_k)$, where $\gamma$ is a shifted gamma distribution kernel that captures the delayed impact of prior cognitive processing on subsequent fixation durations.
    - **Design Motivation**: The linguistic spillover effect indicates that the cognitive load of processing a word persists into the reading times of subsequent words. While traditional approaches assume a Markov property and look only at the previous $l$ fixations, the convolution approach can theoretically capture historical influences infinitely far back.

3. **Reader-Specific Effects (RSE) Model**:

    - **Function**: Learning individual-specific spatio-temporal parameters for different readers
    - **Mechanism**: The predictor vector is expanded as $\mathbf{x}_m = \mathbf{1} \oplus \mathbf{u}_m$, where $\mathbf{u}_m$ is the reader's one-hot encoding vector. This ensures that both the temporal kernel excitement rate/decay rate and the spatial density center incorporate both global effects and reader-specific effects.
    - **Design Motivation**: Individual readers possess diverse reading styles (e.g., saccade amplitudes, regression frequencies). Treating all readers as a single average reader, as in traditional models, discards this critical variability.

### Loss & Training

The overall optimization objective is to maximize the log-likelihood of the scanpaths, which is decomposed and optimized separately for the saccade and duration models. Training uses SGD with Nesterov momentum, an 80/10/10 data split, and an early stopping strategy (patience=5). Hyperparameter grid search is performed (covering batch size, learning rate, and weight decay). A warm-starting strategy is adopted: simpler models are trained first, and their parameters are used to initialize more complex ones.

## Key Experimental Results

### Main Results: Saccade Planning Model Comparison

| Model | Log-likelihood Gain per Fixation (nats) | Relative Gain |
|------|------------------------|---------|
| Poisson baseline | 0 (baseline) | — |
| Last-fixation baseline | ~0.6 | — |
| Standard Hawkes | ~0.8 | — |
| CSS (Constant Spatial Shift) | ~1.5 | — |
| RSE (Reader-Specific Effects) | **2.44** | ~1047% likelihood gain |

### Marginal Contributions of Predictors to Saccades

| Predictor | Relative Gain over RSE | Note |
|---------|-------------------|------|
| Word length (length) | ~4% | Largest contribution |
| Word-level surprisal | <2% | Minimal effect |
| Character-level surprisal | <2% | Minimal effect |
| Unigram surprisal | <2% | Minimal effect |

### Key Findings

- **The RSE model learns a global rightward shift of approximately 10.61 characters**, perfectly reflecting the left-to-right reading direction of English. Reader-specific effects and directional spatial shifts are the most important modeling factors.
- **Linguistic predictors like surprisal have an extremely limited contribution to saccade planning** (<2% relative gain), dwarfed by the contributions of spatio-temporal dependencies and reader-specific effects. This challenges the explanatory power of surprisal theory at a fine-grained eye-movement level.
- **The convolution model performs almost identically to the Markov model**, suggesting that the influence of past fixations on the current duration is likely bounded and does not require an infinite history.
- **When modeling raw scanpaths, the effect size of surprisal is an order of magnitude smaller than that on aggregated data**, indicating that effect sizes reported in previous literature based on aggregated data may be artificially inflated.

## Highlights & Insights

- **The incremental modeling design from simple to complex is remarkably clear**: walking from Poisson → Last-fixation → Hawkes → CSS → RSE → +predictors, the gain of each step is quantifiable, allowing readers to precisely understand the contribution of each component. This incremental ablation design is highly exemplary.
- **Questioning the applicability of surprisal theory at a fine-grained level**: effects that are significant on aggregated data nearly vanish on raw data, providing an important methodological warning that theoretical validation findings may highly depend on data preprocessing methods.
- **Applying spatio-temporal Hawkes processes to cognitive modeling** is an elegant analogical transfer. Hawkes processes, originally used in domains like earthquake aftershocks and crime propagation, naturally map their self-exciting properties to "a prior fixation exciting subsequent fixations" in reading.

## Limitations & Future Work

- **Only the MECO English dataset is used** (46 readers, 12 short texts), leaving generalization to other languages (especially right-to-left languages) unverified.
- **The spatial kernel assumes isotropic variance**, which may fail to sufficiently capture directional biases in eye movements (where horizontal saccades should be much more frequent than vertical ones).
- **Inter-line jumps are not modeled**: the authors attempted to include a distance-to-right-margin feature, but training was unstable. For multi-line text reading, jumping from the end of a line to the start of the next is a critical behavior.
- **Character-level and word-level surprisals use different language models** (GPT-2 vs mGPT); hence, performance differences cannot be solely attributed to the differences in granularity.

## Related Work & Insights

- **vs E-Z Reader / SWIFT**: These are classical cognitive eye-movement control models, but they are not data-driven and do not capture the impact of language processing on fixations. The point process framework in this paper is more flexible but does not provide explanations of cognitive mechanisms.
- **vs ScanDL (Bolliger et al., 2023)**: Uses diffusion models to generate synthetic scanpaths, but it is oriented toward downstream NLP task augmentation rather than theoretical validation. The two methods are complementary—ScanDL can evaluate point process models using generation quality.
- **vs standard surprisal analysis (Smith & Levy; Wilcox et al.)**: This paper directly challenges the reliability of the conclusions in these works, pointing out that aggregation strategies may systematically inflate effect sizes.

## Rating

- Novelty: ⭐⭐⭐⭐ Introducing spatio-temporal Hawkes processes into reading behavior modeling is conceptually natural and novel, though point processes themselves are not new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ The step-by-step ablation design is extremely clear, quantifying the contribution of each step from the baseline to the full model.
- Writing Quality: ⭐⭐⭐⭐⭐ The mathematical derivations are rigorous with unified notation and intuitive diagrams.
- Value: ⭐⭐⭐⭐ The work serves as an important methodological warning for psycholinguistics, though its direct utility for the general NLP community is limited.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Tuna: Comprehensive Fine-grained Temporal Understanding Evaluation on Dense Dynamic Videos](tuna_temporal_understanding.md)
- [\[ACL 2025\] Intuitive Fine-Tuning: Towards Simplifying Alignment into a Single Process](intuitive_fine_tuning.md)
- [\[ACL 2025\] FRACTAL: Fine-Grained Scoring from Aggregate Text Labels](fractal_fine-grained_scoring_from_aggregate_text_labels.md)
- [\[ACL 2025\] Guidelines for Fine-grained Sentence-level Arabic Readability Annotation](guidelines_for_fine-grained_sentence-level_arabic_readability_annotation.md)
- [\[ACL 2025\] Barec: A Large and Balanced Corpus for Fine-grained Arabic Readability Assessment](a_large_and_balanced_corpus_for_fine-grained_arabic_readability_assessment.md)

</div>

<!-- RELATED:END -->

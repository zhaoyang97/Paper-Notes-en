---
title: >-
  [Paper Note] Position: The Future of Bayesian Prediction Is Prior-Fitted
description: >-
  [ICML 2025][LLM Pretraining][Prior-Data Fitted Networks] This position paper argues that **Prior-Data Fitted Networks (PFNs)**—which train neural networks on randomly generated synthetic datasets to approximate Bayesian posterior predictive distributions—represent the future of Bayesian inference. PFNs systematically outperform traditional MCMC/VI/GP methods in implementation simplicity, flexibility of prior definitions, and inference speed, and have already proven their capa…
tags:
  - "ICML 2025"
  - "LLM Pretraining"
  - "Prior-Data Fitted Networks"
  - "PFN"
  - "Bayesian Prediction"
  - "Synthetic Data Pre-training"
  - "TabPFN"
  - "Amortized Inference"
date: 2026-05-08
content_hash: 65e07fb374b43760
---

# Position: The Future of Bayesian Prediction Is Prior-Fitted

**Conference**: ICML 2025  
**arXiv**: [2505.23947](https://arxiv.org/abs/2505.23947)  
**Code**: None (Position Paper)  
**Area**: LLM Pre-training  
**Keywords**: Prior-Data Fitted Networks, PFN, Bayesian Prediction, Synthetic Data Pre-training, TabPFN, Amortized Inference

## TL;DR

This position paper argues that **Prior-Data Fitted Networks (PFNs)**—which train neural networks on randomly generated synthetic datasets to approximate Bayesian posterior predictive distributions—represent the future of Bayesian inference. PFNs systematically outperform traditional MCMC/VI/GP methods in implementation simplicity, flexibility of prior definitions, and inference speed, and have already proven their capability to outperform XGBoost in tabular learning (TabPFN).

## Background & Motivation

**Background**: Bayesian prediction is one of the core paradigms of machine learning. Classical methods include MCMC (exact but extremely slow), Variational Inference (VI; fast but approximation quality is limited by the choice of variational families), and Gaussian Processes (GP; elegant but only applicable to specific classes of priors). These methods require running inference on a per-dataset basis, and cannot amortize computation across tasks.

**Limitations of Prior Work**:
   - MCMC converges extremely slowly in high-dimensional latent variable spaces and has high implementation complexity.
   - VI requires explicit parameterization of the latent variable distribution, making it difficult to handle complex structures (such as network architecture priors).
   - GP is limited by analytically tractable kernel functions, resulting in a narrow class of priors.
   - All traditional methods require a computable likelihood function, making it impossible to use "sample-only, density-free" priors.

**Key Challenge**: While pre-training compute continues to grow exponentially (due to advances in GPU manufacturing and improvements in optimizer/architecture designs), real-world data growth has stagnated in many application domains. A key challenge is how to convert surplus compute into performance improvements in data-scarce scenarios.

**Goal**: To demonstrate that a new Bayesian prediction paradigm—PFN—can (a) fully leverage large-scale pre-training compute, (b) support declarative prior definitions (requiring only the ability to sample), (c) perform inference with a single forward pass, and (d) cover classes of priors inaccessible to traditional methods.

**Key Insight**: Starting from the success of TabPFN in outperforming XGBoost on tabular data, the authors observe that this paradigm of "pre-training on synthetic data + in-context learning on real data" holds general methodological value and is highly suitable for extension to more domains.

**Core Idea**: PFN achieves the amortization of Bayesian prediction by reframing Bayesian inference as a supervised learning objective: "training neural networks on synthetic datasets sampled from a prior." It stands as the most suitable Bayesian method for the era of abundant compute and scarce data.

## Method

### Overall Architecture

The core workflow of PFN is split into two phases:

**Pre-training Phase (Prior-Fitting)**:
1. Define a prior distribution $p(D)$ (typically by generating data after sampling latent variables).
2. Repeatedly sample synthetic datasets from the prior.
3. Split each dataset into a training set and test points.
4. Optimize the network parameters to predict the distribution of test points given the training set.

**Inference Phase**: Given a new, real-world dataset and query points, PFN directly outputs the posterior predictive distribution via a single forward pass—without any additional training, sampling, or optimization.

### Key Designs

1. **Cross-Entropy Training Objective**:

    - **Function**: Train the PFN to approximate the posterior predictive distribution (PPD).
    - **Mechanism**: The training loss is the negative log-likelihood on synthetic data, which is equivalent to minimizing the KL divergence between the PFN's output distribution and the true PPD plus a constant.
    - **Design Motivation**: Converting the Bayesian inference problem into a standard supervised learning formulation (cross-entropy optimization) allows the direct utilization of mature GPU training infrastructures without the need to analytically derive posteriors.

2. **Declarative Prior**:

    - **Function**: Allow users to implicitly define priors through a data-generating process, rather than explicitly writing out the probability density.
    - **Mechanism**: PFN only requires the ability to **sample** from the prior, rather than computing likelihood or prior densities.
    - **Design Motivation**: Traditional MCMC/VI methods must be able to compute likelihood and prior densities, which excludes a large number of priors based on simulators, complex computation graphs, or hybrid structures. PFN bypasses this limitation.

3. **Transformer Architecture and In-Context Learning (ICL)**:

    - **Function**: PFN typically adopts a Transformer architecture, where training samples attend to each other, and test positions only attend to training positions.
    - **Mechanism**: Leveraging the in-context learning capability of Transformers, the training set is ingested as input context. After learning the data patterns within the context, the PFN directly outputs the prediction.
    - **Design Motivation**: The architecture naturally supports variable-length inputs and permutation invariance, perfectly matching the paradigm of Bayesian prediction.

### Prior Design Examples

The paper introduces several representative prior designs in detail:

| Prior Type | Latent Variables | Data Generation Method | Applicable Scenarios |
|----------|--------|-------------|---------|
| BNN Prior | MLP weights (Gaussian distribution) | Random MLP forward pass | General function approximation |
| GP Prior | Kernel hyperparameters (length scales, kernel types, etc.) | Sampling from GP | Bayesian optimization |
| TabPFN Prior | Structural Causal Model (SCM) computation graph | Sampling features and targets from SCM graphs | Tabular supervised learning |
| Learning Curve Prior | Power-law / S-curve parameters | Simulating ML training curve shapes | Learning curve extrapolation |
| Time Series Prior | Periodicity / Trend parameters | Time series generation with seasonality and trend | Time series forecasting |

### Loss & Training

- Pre-training only needs to be executed once (for a given prior); inference on a new dataset requires only a single forward pass.
- Pre-training can be computationally expensive (representing an offline cost similar to meta-learning), but online inference is extremely fast.
- PFNs can be compiled into ONNX format for deployment, further simplifying engineering implementations.
- The dataset size can randomly vary during pre-training, enabling the PFN to adapt to inputs of different scales.

## Key Experimental Results

> Note: This is a position paper and does not contain systematic benchmark experiments. The tables below summarize representative results and comparisons cited in the paper.

### PFN vs. Traditional Bayesian Methods

| Metric | PFN | MCMC | VI | GP |
|------|-----|------|-----|-----|
| Implementation Complexity | Low (standard forward pass) | High (sampler implementation) | Medium (variational distribution choice) | Medium (kernel function choice) |
| Prior Flexibility | High (only requires sampling) | Medium (requires likelihood density) | Medium (requires likelihood density) | Low (specific kernel functions) |
| Inference Speed | Extremely Fast (single forward pass) | Extremely Slow (massive sampling) | Medium (iterative optimization) | Fast (closed-form / approximation) |
| Handling Complex Latent Variables | Implicitly handled | Slow convergence in high dimensions | Requires parameterization | No explicit latent variables |
| No Sampling Needed for Prediction | Yes | No | No | Yes |
| Leverages Pre-training Compute | Yes | No | No | No |

### Key Application Results (Cited Data)

| Application Scenario | Method | Key Results | Source |
|---------|------|---------|------|
| Tabular Classification (up to 10k samples) | TabPFN | Outperforms 4-hour XGBoost hyperparameter tuning in less than 5 seconds | Hollmann et al., 2025 |
| Bayesian Prediction (small-scale) | PFN | 200x faster than traditional methods | Muller et al., 2022 |
| Learning Curve Extrapolation | LC-PFN | 10,000x faster than traditional methods | Adriaensen et al., 2023 |
| Bayesian Optimization | PFN-BO | Serves as a surrogate model replacing GP | Muller et al., 2023c |
| RNA Folding Time Prediction | PFN | Successful application in biology | Scheuer et al., 2024 |
| Chip Latency Prediction | PFN | Successful application in hardware | Carstensen et al., 2024 |
| Metagenomics Data | PFN | Expansion into new domains | Perciballi et al., 2024 |

### Key Findings

- **TabPFN is the flagship application of PFNs**: As the first deep learning model to consistently outperform classical methods like XGBoost on small tabular datasets, it demonstrates the vast potential of the "pre-training on synthetic data + in-context inference on real data" paradigm.
- **The speed advantage scales with task complexity**: Ranging from a 200x speedup in small-scale Bayesian prediction to 10,000x in learning curve extrapolation, the amortized advantage of PFNs becomes particularly prominent when repeatedly applying the same prior.
- **Rapid expansion of application domains**: PFNs have generalized from tabular data to over a dozen fields including time series, anomaly detection, Bayesian optimization, symbolic regression, geometric reasoning, causal discovery, biology, and hardware.

## Highlights & Insights

- **Reframing Bayesian inference as supervised learning**: This is the most central conceptual contribution. By proving that cross-entropy on synthetic data is equivalent to the KL divergence from the true PPD, PFNs compile Bayesian inference into a form directly optimizable by deep learning, completely decoupling prior definition from inference execution.
- **Declarative priors represent a paradigm shift**: Traditional methods require priors to have computable density functions, whereas PFNs simplify this to "writing a data-generating program." TabPFN's SCM prior is a prime example of what traditional methods struggle to handle.
- **The asymmetric growth of compute and data forms the ecological niche for PFNs**: While GPU compute grows exponentially, real-world data growth has stagnated. PFNs convert surplus compute into data-scarce scenario performance through synthetic data pre-training—a profound industry insight.
- **The idea of amortization is widely transferable**: It can be generalized to any scenario requiring repetitive inference under the same prior, such as simulation-based inference (SBI) for scientific simulator parameter estimation, or hyperparameter optimization in AutoML.

## Limitations & Future Work

- **Prior-Reality Gap**: The quality of PFNs heavily depends on how well the prior matches real-world data. Malformed priors can lead to "correct Bayesian prediction under incorrect priors"; how to automatically tune the prior remains a key open question.
- **Scalability to large-scale data**: Current PFNs mainly target small-scale data (typically fewer than 10k samples). The quadratic complexity of Transformers and context window limits present significant bottlenecks.
- **Cost of prior engineering**: Designing high-quality priors (e.g., TabPFN's SCM prior) requires considerable domain expertise and tuning, which is under-discussed in the paper.
- **Lack of systematic new experiments**: As a position paper relying mostly on existing results without providing new empirical comparative studies, readers must refer to the original publications to fully evaluate specific claims.
- **Interpretability issues**: Since PFNs function as black boxes, whether their uncertainty calibration is reliable requires deeper empirical validation.
- **Connection to LLM ICL**: While the in-context learning (ICL) of LLMs shares strong conceptual similarities with PFNs, the paper does not deeply analyze their theoretical connections or the possibility of a unified framework.

## Related Work & Insights

- **vs. MCMC/VI/GP**: Traditional Bayesian methods perform inference on a per-dataset basis, whereas PFNs amortize computation through pre-training. PFNs hold the advantage in inference speed and prior flexibility, but struggle with pre-training costs and the prior-reality gap.
- **vs. Meta-Learning (such as MAML)**: While sharing similar goals (learning how to learn), PFNs are pre-trained on synthetic data and thus are not restricted to the training task distribution, though they consequently forfeit the ability to learn priors from real-world data.
- **vs. LLM In-Context Learning**: LLMs acquire ICL through text corpora, while PFNs acquire ICL through synthetic data. Their priors differ: LLM priors are implicitly defined by structural-textual corpora, while PFN priors are explicitly defined by data-generating programs.
- **vs. Simulation-Based Inference (SBI)**: While SBI targets the posterior distribution of parameters, PFNs target the posterior predictive distribution. PFNs directly predict observables, avoiding the need to explicitly model the posterior of latent variables.
- **vs. XGBoost/AutoML**: TabPFN replaces gradient boosting and hyperparameter tuning with a Bayesian approach. Outperforming 4 hours of tuning in 5 seconds is the strongest evidence of PFN's practical utility.

## Rating

- Novelty: ⭐⭐⭐⭐ The core PFN concept originates from 2022, but this systematic position argument is highly valuable.
- Experimental Thoroughness: ⭐⭐⭐ As a position paper, it majorly cites existing results and lacks systematic new empirical validation.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structures, complete logical chain, and extremely easy to comprehend.
- Value: ⭐⭐⭐⭐ Systematically categorizes the PFN landscape, highlights open challenges and future directions, offering high reference value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Bayesian Neural Scaling Law Extrapolation with Prior-Data Fitted Networks](bayesian_neural_scaling_law_extrapolation_with_prior-data_fitted_networks.md)
- [\[ICLR 2026\] Beyond Multi-Token Prediction: Pretraining LLMs with Future Summaries](../../ICLR2026/llm_pretraining/beyond_multi-token_prediction_pretraining_llms_with_future_summaries.md)
- [\[ICML 2025\] Density Ratio Estimation-based Bayesian Optimization with Semi-Supervised Learning](density_ratio_estimation-based_bayesian_optimization_with_semi-supervised_learni.md)
- [\[ICML 2026\] Constrained Bayesian Experimental Design via Online Planning](../../ICML2026/llm_pretraining/constrained_bayesian_experimental_design_via_online_planning.md)
- [\[CVPR 2025\] Bridging the Vision-Brain Gap with an Uncertainty-Aware Blur Prior](../../CVPR2025/llm_pretraining/bridging_the_vision-brain_gap_with_an_uncertainty-aware_blur_prior.md)

</div>

<!-- RELATED:END -->

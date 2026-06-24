---
title: >-
  [Paper Note] Neuron Empirical Gradient: Discovering and Quantifying Neurons' Global Linear Controllability
description: >-
  [ACL 2025][Neuron Interpretability] This work reveals a global linear relationship between the activation values of Feed-Forward (FF) layer neurons and model outputs in pretrained language models. It introduces Neuron Empirical Gradient (NEG) to quantify this linear relationship and designs an efficient estimation method, NeurGrad. Finally, skill neuron probing experiments demonstrate that NEG can effectively characterize various language skills.
tags:
  - "ACL 2025"
  - "Neuron Interpretability"
  - "Neuron Gradient"
  - "Knowledge Attribution"
  - "Skill Neurons"
  - "Language Models"
date: 2026-05-08
content_hash: 5145970782a79c84
---

# Neuron Empirical Gradient: Discovering and Quantifying Neurons' Global Linear Controllability

**Conference**: ACL 2025  
**arXiv**: [2412.18053](https://arxiv.org/abs/2412.18053)  
**Code**: [Yes (GitHub)](https://github.com/xzhao-tkl/NEG)  
**Area**: Others  
**Keywords**: Neuron Interpretability, Neuron Gradient, Knowledge Attribution, Skill Neurons, Language Models

## TL;DR

This work reveals a global linear relationship between the activation values of Feed-Forward (FF) layer neurons and model outputs in pretrained language models. It introduces Neuron Empirical Gradient (NEG) to quantify this linear relationship and designs an efficient estimation method, NeurGrad. Finally, skill neuron probing experiments demonstrate that NEG can effectively characterize various language skills.

## Background & Motivation

It has been proven by multiple studies that neurons in the Feed-Forward (FF) layers of pretrained language models (PLMs) can encode knowledge. However, existing research faces two major issues:

**Ranking only, without quantification**: Existing methods (such as knowledge neuron discovery methods) mainly rank neurons by importance but cannot quantify the precise relationship between changes in neuron activations and changes in model outputs. This limits application scenarios like knowledge editing—if one does not know how much modifying a neuron affects the output, precise control over model behavior remains impossible.

**High computational cost**: Existing methods require repeatedly modifying activation values for inference or performing heavy tensor computations, preventing large-scale analysis of all neurons, especially on large models like Llama2-70B.

The authors pose a natural question: **What is the exact relationship between changes in neuron activations and changes in model outputs?** If this relationship can be quantified, it would unlock precise control over PLM outputs.

## Method

### Overall Architecture

The paper proceeds in three steps:

1. Discover the **global linear relationship between activation shift and output shift** through neuron intervention experiments.
2. Propose **NeurGrad** to efficiently estimate the neuron empirical gradient (NEG).
3. Validate that NEG can characterize various language skills through **skill neuron probing**.

### Key Designs

#### 1. **Discovery of Neuron Linear Relationship (NEG)**

Core approach: Modifying the activation value of a specific neuron within the range of [-10, 10] with a step size of 0.2, and observing changes in target token probability. Experiments are conducted on 7 PLMs (including BERTbase, BERTlarge, and several Llama/Qwen models).

Key findings:
- Within the $\pm 2$ range, the Pearson correlation coefficient $r$ between activation shift and output shift consistently exceeds 0.95.
- Over 90% of the neurons exhibit linear behavior.
- The number of positive and negative polarity neurons is roughly equal (about 50% each).

Based on this, **NEG** (Neuron Empirical Gradient) is defined as the slope of the linear regression between activation shift and output shift.

#### 2. **NeurGrad: Efficient NEG Estimation**

Directly computing NEG requires around 100 inferences for each neuron (using different shift values), which is computationally expensive. The authors find that:
- The **absolute value** of the computed gradient (CG, obtained via backpropagation) is highly correlated with NEG ($r = 0.961$), but the **sign** is unreliable.
- The sign of the neuron activation value can correct the direction of the CG.

This leads to the formulation of NeurGrad:

$$\bar{G_E} = CG \times \text{sign}(A)$$

where $CG$ is the computation graph gradient, and $A$ is the neuron activation value. Its running time is only 1/120 of IG (Integrated Gradients).

#### 3. **Multi-Neuron Control**

Experiments validate whether NEG remains effective during simultaneous interventions of multiple neurons:
- When simultaneously intervening with $2^{12}$ neurons, the correlation between predicted shift and actual shift remains $\ge 0.7$.
- However, as the number of intervened neurons or the shift magnitude increases, the linearity gradually declines.

The **local linear approximation hypothesis** is proposed to explain this phenomenon: similar to first-order Taylor expansion, local differentiability within a small range guarantees linearity, but non-linear effects increase as the range widens.

#### 4. **MCEval8K Benchmark and Skill Neuron Probing**

The authors construct the MCEval8K benchmark, covering 6 major categories and 22 language understanding tasks (linguistics, content classification, NLI, factuality, introspection, multilingualism), with a limit of 8K samples per task.

Three types of probes are designed:
- **Polar-Probe**: A majority-voting classifier based on polarity.
- **Magn-Probe**: A majority-voting classifier based on NEG magnitude.
- **Tree-Probe**: A random forest model to capture dependencies among neurons.

### Loss & Training

The computation of NEG uses zero-intercept linear regression fitting; NeurGrad only requires a single forward and backward pass, without additional training. Among the skill-neuron probes, Tree-Probe utilizes the default settings of scikit-learn's Random Forest (100 trees, no depth limit).

## Key Experimental Results

### NeurGrad vs. Baselines for NEG Estimation

| Method | Correlation r (BERT-base) | Correlation r (Llama2-7B) | MAE (BERT-base) | Running Time |
|------|---------------------|---------------------|------------------|---------|
| CG | -0.891 | 0.302 | 6.1e-03 | 0.149s |
| IG | 0.736 | 0.538 | 3.0e-03 | 19.349s |
| LPI | - | 0.647 | - | 6.086s |
| **NeurGrad** | **0.9998** | **0.814** | **2.6e-05** | **0.161s** |

### Skill Neuron Probing (Llama2-7B)

| Task | LM-Prob | Act (Activation) | Magn (Gradient) | Tree-Probe |
|------|---------|------------|------------|------------|
| NER | 0.361 | 0.453 | 0.498 | 0.740 |
| Agnews | 0.588 | 0.849 | 0.702 | 0.872 |
| PAWS | 0.524 | 0.825 | 0.815 | 0.888 |
| CSQA | 0.610 | 0.613 | 0.639 | 0.773 |
| HaluEval | 0.520 | 0.788 | 0.783 | 0.818 |
| mLAMA | 0.608 | 0.622 | 0.637 | 0.724 |

### Key Findings

1. **NEG is almost perfectly estimated on BERT**: NeurGrad achieves a correlation of 0.9998 and an MAE of only 2.6e-05 on BERTbase.
2. **Gradient vs. Activation benefits differ**: NEG outperforms activation-based methods on knowledge-intensive tasks (mLAMA, CSQA), possibly because complex knowledge is not fully learned during pretraining.
3. **Tree-Probe substantially outperforms majority-voting probes**: This indicates that dependencies among neurons are crucial for characterizing language skills.
4. **Skill neurons are highly efficient**: For most tasks, only 256 neurons are required to achieve optimal accuracy.
5. **Skill neurons are robust and replaceable**: They are robust to different prompt templates, and different subsets of neurons can achieve comparable performance.
6. **Different tasks exhibit different neuron dependency patterns**: PAWS prefers deep trees, CSQA favors multiple trees, and HaluEval requires a balance.

## Highlights & Insights

- **A leap from qualitative to quantitative**: Prior understanding of FF layer neurons mostly focused on "which neurons are important," whereas this work provides the first quantitative answer of "how important a neuron is (in terms of precise gradient values)."
- **A minimalist yet effective method**: The formula for NeurGrad is only one line ($CG \times \text{sign}(A)$), yet its effectiveness far surpasses complex integrated gradients and causal tracing methods.
- **90%+ of neurons are linear**: If further exploited (e.g., for precise knowledge editing), this finding could have a profound impact on model interpretability and controllability.
- **Positive and negative polarity neurons each account for half**: This implies that simply increasing or decreasing activation values is insufficient, and the polarity direction must be considered.

## Limitations & Future Work

1. The current analysis is restricted to single-token factual prompts and has not been extended to multi-token generation scenarios.
2. How to leverage NEG for practical language skill-level output adjustment (such as knowledge editing and bias mitigation) remains unexplored.
3. The linear relationship weakens under large shift ranges, limiting the magnitude of practical neuron modifications.
4. Although the MCEval8K benchmark is comprehensive, the difficulty distribution within each task may not be uniform.
5. A comparison with circuit-based interpretability methods like Sparse Autoencoders is missing.

## Related Work & Insights

- Knowledge Neurons (Dai et al., 2022) pioneered the research direction of knowledge attribution in FF layer neurons.
- ROME/MEMIT (Meng et al., 2022) provided knowledge editing methods based on causal tracing.
- Skill Neurons (Wang et al., 2022) first proposed using activation values for skill neuron probing.
- This work unifies these directions under a single NEG framework, providing a more precise quantitative tool.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Unveils and quantifies the global neuron-to-output linear relationship for the first time; the NeurGrad method is minimalist and highly efficient.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Highly comprehensive, spanning 7 PLMs, 22 tasks, and multiple probes; the MCEval8K benchmark holds long-term utility.
- **Writing Quality**: ⭐⭐⭐⭐ — Logically clear and progressive, though formula notations are dense and require careful reading.
- **Value**: ⭐⭐⭐⭐⭐ — Provides crucial foundational contributions to model interpretability, knowledge editing, and bias mitigation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Quantifying Lexical Semantic Shift via Unbalanced Optimal Transport](quantifying_lexical_semantic_shift_via_unbalanced_optimal_transport.md)
- [\[ICML 2025\] Probably Approximately Global Robustness Certification](../../ICML2025/others/probably_approximately_global_robustness_certification.md)
- [\[ICML 2026\] Guaranteed Optimal Compositional Explanations for Neurons](../../ICML2026/others/guaranteed_optimal_compositional_explanations_for_neurons.md)
- [\[NeurIPS 2025\] Statistical Inference for Gradient Boosting Regression](../../NeurIPS2025/others/statistical_inference_for_gradient_boosting_regression.md)
- [\[ICML 2025\] Gradient Aligned Regression via Pairwise Losses](../../ICML2025/others/gradient_aligned_regression_via_pairwise_losses.md)

</div>

<!-- RELATED:END -->

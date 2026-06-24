---
title: >-
  [Paper Note] What Scales in Cross-Entropy Scaling Law?
description: >-
  [ICLR 2026][LLM Pretraining][Scaling Laws] This paper precisely decomposes cross-entropy loss into three terms: "Error-Entropy + Self-Alignment + Confidence." Through experiments on 32 models spanning five orders of magnitude, it demonstrates that only the Error-Entropy consistently follows a power-law decay with model size. The other two terms remain largely invariant to scale—explaining why the cross-entropy scaling law is accurate for small models but tends to fail for ult…
tags:
  - "ICLR 2026"
  - "LLM Pretraining"
  - "Scaling Laws"
  - "Cross-Entropy"
  - "Error-Entropy"
  - "Rank-based Error"
  - "Training Dynamics"
date: 2026-05-08
content_hash: 5f227565b5af7f4e
---

# What Scales in Cross-Entropy Scaling Law?

**Conference**: ICLR 2026  
**Paper**: Published as a conference paper at ICLR 2026  
**Code**: https://github.com/yanjx2021/RethinkCE (Available)  
**Area**: LLM Pre-training / Scaling Laws  
**Keywords**: Scaling Laws, Cross-Entropy, Error-Entropy, Rank-based Error, Training Dynamics

## TL;DR
This paper precisely decomposes cross-entropy loss into three terms: "Error-Entropy + Self-Alignment + Confidence." Through experiments on 32 models spanning five orders of magnitude, it demonstrates that only the Error-Entropy consistently follows a power-law decay with model size. The other two terms remain largely invariant to scale—explaining why the cross-entropy scaling law is accurate for small models but tends to fail for ultra-large models.

## Background & Motivation

**Background**: The cross-entropy scaling law (Kaplan et al. 2020) is a core tool in LLM development. It posits that as model scale and data volume increase, the cross-entropy loss decreases according to a predictable power law $L_{CE}\propto N^{-\alpha}$. This empirical law is widely used for resource allocation, extrapolating performance from small to large models, and hyperparameter tuning, while also serving as a theoretical entry point for understanding the nature of intelligence.

**Limitations of Prior Work**: Recent practical and theoretical findings have cast doubt on this law. In practice, while cross-entropy fits a power law well for small models, the decline slows significantly for ultra-large models—forcing OpenAI to revise the formula to $L_{CE}\propto N^{-\alpha}+\text{bias}$. Some studies even suggest the slowdown continues. Theoretically, existing frameworks mostly prove that "error-based" metrics like MSE follow power-law scaling but fail to generalize directly to cross-entropy. Consequently, whether cross-entropy truly scales remains an open question that challenges the confidence in the "scaling up" paradigm.

**Key Challenge**: The authors' key hypothesis is that **what actually scales is not cross-entropy itself, but a dominant component hidden within it**. Cross-entropy creates an "illusion" of scaling by being proxy to this component. Identifying this truly scaling component would provide a more reliable law for LLM development and a cleaner optimization target for research into the principles of intelligence.

**Goal**: (1) Find a decomposition method that precisely breaks down cross-entropy; (2) Verify which components follow power-law scaling; (3) Explain why cross-entropy scaling fails in large models using these findings.

**Key Insight**: The authors argue that the "ranking of the correct token" reflects a model's true capability better than the "probability of the correct token." While probabilities are easily manipulated by sampling methods like temperature scaling, top-k, or top-p, the relative ranking between tokens remains nearly invariant to such post-processing. Thus, they redefine error based on "ranking" rather than "probability."

**Core Idea**: Propose a Rank-Based Error (RBE) metric and use it to **precisely** decompose cross-entropy into three terms. They prove that only "Error-Entropy" strictly obeys a power law, termed the **Error-Entropy Scaling Law**.

## Method

### Overall Architecture
The logic of the work follows: "Define a new metric → Perform lossless algebraic decomposition of cross-entropy → Test scaling behavior term by term → Explain the failure of the old law using decomposition results."

Specifically, for each ground-truth token in the test corpus, the "Rank-based Error" (RBE) is calculated (the number of tokens ranked higher than the correct token). Predictions are grouped by RBE values to compute two distributions: the RBE distribution $p_e$ (probability of the correct token being at rank $e$) and the score distribution $q_e$ (normalized average score per group), plus a scalar $C$ (overall score norm). Using $p_e, q_e, C$, cross-entropy is **identically transformed** into the sum of Error-Entropy, Self-Alignment, and Confidence. Finally, log-log regressions on 32 models show that only Error-Entropy scales stably.

As this is a scaling law analysis paper, the methodology relies on algebraic derivation rather than a multi-module pipeline.

### Key Designs

**1. Rank-based Error (RBE): Measuring Error via Ranking instead of Probability**

The motivation is straightforward: cross-entropy relies on the probability score $s_{v_i}$ of the correct token, which is easily distorted by sampling strategies, causing the loss value to drift. In contrast, the relative ranking of tokens is immune to these effects. The authors define the rank of the correct token as the error:

$$\text{RBE}(v_i)=\sum_{v\in V}\mathbb{1}\{s_v>s_{v_i}\}$$

RBE equals the number of tokens with scores higher than the ground truth (RBE=0 if the correct token is ranked first). Based on this, two distributions are defined: the RBE distribution $p_e=\Pr(\text{RBE}(v_i)=e\mid v_i\in D)$ characterizes how frequently the correct token appears at rank $e$; the score distribution groups predictions by RBE and takes the geometric mean $Q_e=\text{GeoMean}(\{s_{v_i}\mid \text{RBE}(v_i)=e\})$, then normalizes it as $q_e=Q_e/C$, where $C=\sum_e Q_e$ is the score norm ($C$ indicates model confidence). RBE is the foundation of the decomposition, re-expressing the classification task as an "error distribution" and introducing the concept of Error-Entropy from Information Theoretic Learning (ITL) into language modeling.

**2. Lossless Decomposition of Cross-Entropy: Error-Entropy + Self-Alignment + Confidence**

Using $p_e, q_e, C$, the authors rewrite cross-entropy grouped by RBE. Starting from the definition $L_{CE}=-\frac1N\sum_i\log s_{v_i}$, they group terms by RBE, decompose the product using logarithms, and substitute $Q_e=C\cdot q_e$ to arrive at a **precise identity** (not an approximation):

$$L_{CE}=\underbrace{-\sum_e p_e\log p_e}_{\text{Error-Entropy}}+\underbrace{\sum_e p_e\log\frac{p_e}{q_e}}_{\text{Self-Alignment}}-\underbrace{\log C}_{\text{Confidence}}$$

Each term has a clear operational meaning: **Error-Entropy** is the Shannon entropy of the RBE distribution $p_e$. Minimizing it forces $p_e$ to concentrate on low ranks, appearing as "learning to rank the correct token first," which directly corresponds to discriminative ability. **Self-Alignment** is the KL divergence between $p_e$ and $q_e$. Minimizing it requires the model's output score distribution to align with its own error distribution—providing a new interpretation: models do not necessarily approximate the "true language distribution" but distribute probability based on their own "likelihood of making an error." **Confidence** is the log-norm of the scores $\log C$. It carries a negative sign in the decomposition and is increased during training, corresponding to the model pushing scores of low-ranked tokens toward zero. Training curves (Fig. 3) confirm that all three are optimized, but due to magnitude differences, the model reduces the dominant Error-Entropy first.

**3. Error-Entropy Scaling Law: Only Error-Entropy Truly Scales via Power Law**

This is the central finding. Across Wikipedia, C4, and GitHub datasets using 32 models from 8 families (GPT2, Pythia, Llama2/3.2, Mistral, OPT, Qwen2.5, Distilgpt2), log-log regressions $\log|M|=c_M+\alpha_M\log N$ were performed. They evaluated $R^2$ (goodness of fit) and $|\Delta_M|=|\alpha_M-\alpha_{CE}|$ (slope similarity to cross-entropy). The conclusion is clean: Error-Entropy achieves $R^2$ values near 0.9—**often exceeding cross-entropy itself**—and has the smallest $|\Delta|$. Self-Alignment lacks a stable power law, and Confidence signals are weak ($R^2$ drops to 0.06~0.21 in mixed settings). This proves Error-Entropy is the actual engine driving cross-entropy scaling.

**4. Explaining the Failure of Old Laws and Providing Differentiable Proxy Loss**

The authors use the decomposition to explain the long-standing puzzle: why does cross-entropy scaling slow down in large models? The answer lies in the **ratio of Error-Entropy to Cross-Entropy** (Fig. 8). In small models, Error-Entropy accounts for ~80-90% of the total loss, so cross-entropy follows its clean power law. As models scale, the proportion of Error-Entropy decreases while the non-scaling Self-Alignment and Confidence terms increase, causing departure from the power law. Based on this, they propose a differentiable proxy loss: since Error-Entropy itself is non-differentiable with respect to logits, they penalize Confidence: $L_\lambda=CE+\lambda\cdot CONF\ (0<\lambda<1)$. The gradient for the correct token score is:

$$\frac{\partial L_\lambda}{\partial s_i}=-\frac{1}{N s_i}\Big(1-\lambda\frac{q_e}{p_e}\Big)$$

The factor $(1-\lambda q_e/p_e)$ aligns $q_e$ with $p_e$, shifting optimization focus from "pointlessly increasing probabilities of already correctly ranked tokens" back to "improving the error distribution."

### Loss & Training
The work is primarily analytical. The only training-related output is the proxy loss $L_\lambda=CE+\lambda\cdot CONF$. The authors also suggest Error-Entropy as a non-differentiable reward signal for RL-based fine-tuning. Training dynamics experiments were conducted using pythia-160m/410m/1b on Wikipedia/C4/GitHub (refer to original App. A.1 for details).

## Key Experimental Results

### Main Results: Power-Law Fit Quality $R^2$ (Higher is better)
Error-Entropy (EE) achieves the highest $R^2$ across almost all family × dataset combinations, often **outperforming Cross-Entropy** (CE). Self-Alignment (SA) and Confidence (Conf) perform significantly worse.

| Model Family | Dataset | CE | EE | SA | Conf |
|---|---|---|---|---|---|
| Qwen | Wikipedia | 0.9731 | **0.9753** | 0.9441 | 0.2977 |
| Pythia | Wikipedia | 0.9448 | **0.9767** | 0.0190 | 0.812 |
| GPT2 | C4 | 0.9892 | 0.9872 | 0.3357 | 0.9444 |
| Qwen | GitHub | 0.9882 | **0.9896** | 0.9455 | 0.1371 |
| All (Mixed) | C4 | 0.8699 | **0.9012** | 0.2188 | 0.0492 |
| All (Mixed) | GitHub | 0.6743 | **0.7229** | 0.3233 | 0.0203 |

### Analysis: Scaling Slope Difference $|\Delta|$ (Smaller is more similar to CE scaling)
The slope of Error-Entropy is consistently the closest to Cross-Entropy.

| Model Family | Dataset | EE | SA | Conf |
|---|---|---|---|---|
| Qwen | Wikipedia | **0.0104** | 0.2347 | 0.0786 |
| Pythia | C4 | **0.0038** | 0.0969 | 0.0354 |
| GPT2 | GitHub | **0.0352** | 0.2126 | 0.0866 |
| All (Mixed) | Wikipedia | **0.0147** | 0.2678 | 0.1002 |

### Key Findings
- **Only Error-Entropy Scales**: Error-Entropy decreases linearly in log-log plots with $R^2$ values exceeding CE. Self-Alignment generally increases with scale, and Confidence is irregular.
- **Explaining the Scaling Failure**: In small models, EE dominates (80-90%). In large models, the non-scaling terms take over, causing the sub-linear slowdown relative to the power law.
- **Explainable Training Dynamics**: Models minimize EE first because of its magnitude; only after EE is largely minimized does the model optimize the smaller SA and Conf terms.

## Highlights & Insights
- **Identity over Approximation**: The decomposition is an exact mathematical identity, allowing for unbiased term-by-term analysis.
- **Shift from Probability to Ranking**: Using RBE bypasses probability distortions caused by sampling and bridges LLMs with Information Theoretic Learning (ITL).
- **Interpretative "Self-Alignment"**: This suggests models align with their own "error profiles" rather than the ground truth distribution, explaining model-specific probability behaviors.
- **Transferability**: The strategy of "splitting a scaling core from a non-scaling shell" can be applied to other classification or retrieval tasks using cross-entropy.

## Limitations & Future Work
- **Limited Proxy Loss Validation**: The proxy loss $L_\lambda$ is presented as a feasibility proof without extensive end-to-end evidence of superior training performance on large scales.
- **Cross-Family Noise**: Regression $R^2$ decreases when mixing different model families, suggesting that differences in training recipes introduce noise.
- **Non-differentiability**: Error-Entropy cannot be optimized directly, requiring proxies or RL rewards, which creates an engineering gap.
- **Future Directions**: Applying kernel-based Error-Entropy minimization from ITL to LLM training or testing on more controlled, identical-recipe model ladders.

## Related Work & Insights
- **vs. Classic Scaling Laws (Kaplan 2020 / Hoffmann 2022)**: While prior work fits CE as an indivisible unit and adds bias terms to fix large-scale failures, this work identifies the mechanism of failure via decomposition.
- **vs. Error-based Scaling (e.g., Lyu et al. 2025)**: Previous theories prove MSE-like error scaling but struggle with CE. This work uses RBE to recover an error distribution in a classification setting, linking the two worlds.
- **vs. Cross-Entropy Properties (Guo 2017)**: Unlike studies focusing on micro-properties like calibration, this work links those properties to macro-scaling behaviors.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Decoupling CE into three terms and identifying the scaling core is a significant contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Solid verification across 32 models, though the proxy loss validation is lighter.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear derivation and consistent narrative.
- Value: ⭐⭐⭐⭐⭐ Provides a more reliable core metric for scaling law research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] A Law of Data Reconstruction for Random Features (and Beyond)](a_law_of_data_reconstruction_for_random_features_and_beyond.md)
- [\[ICLR 2026\] Scaling Behavior of Discrete Diffusion Language Models](scaling_behavior_of_discrete_diffusion_language_models.md)
- [\[CVPR 2025\] ScaMo: Exploring the Scaling Law in Autoregressive Motion Generation Model](../../CVPR2025/llm_pretraining/scamo_exploring_the_scaling_law_in_autoregressive_motion_generation_model.md)
- [\[ICLR 2026\] Not All Documents Are What You Need for Extracting Instruction Tuning Data](not_all_documents_are_what_you_need_for_extracting_instruction_tuning_data.md)
- [\[ICLR 2026\] Pretraining Scaling Laws for Generative Evaluations of Language Models](pretraining_scaling_laws_for_generative_evaluations_of_language_models.md)

</div>

<!-- RELATED:END -->

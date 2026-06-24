---
title: >-
  [Paper Note] Mitigating Selection Bias with Node Pruning and Auxiliary Options
description: >-
  [ACL 2025][Model Compression][selection bias] This paper proposes two complementary methods, Bias Node Pruning (BNP) and Auxiliary Option Injection (AOI), to concurrently mitigate the selection bias of LLMs in multiple-choice questions (MCQs) from both internal and external perspectives. This is achieved by localizing and pruning 0.002% of the biased parameters in the model's output layer (white-box) and injecting an "I don't know" auxiliary option (widely applicable to black…
tags:
  - "ACL 2025"
  - "Model Compression"
  - "selection bias"
  - "bias node pruning"
  - "auxiliary option injection"
  - "multiple-choice questions"
  - "LLM debiasing"
date: 2026-05-08
content_hash: 64cebd190a64e9fd
---

# Mitigating Selection Bias with Node Pruning and Auxiliary Options

**Conference**: ACL 2025  
**arXiv**: [2409.18857](https://arxiv.org/abs/2409.18857)  
**Institution**: University of Wisconsin-Madison, Amazon  
**Keywords**: selection bias, bias node pruning, auxiliary option injection, multiple-choice questions, LLM debiasing

## TL;DR

This paper proposes two complementary methods, Bias Node Pruning (BNP) and Auxiliary Option Injection (AOI), to concurrently mitigate the selection bias of LLMs in multiple-choice questions (MCQs) from both internal and external perspectives. This is achieved by localizing and pruning 0.002% of the biased parameters in the model's output layer (white-box) and injecting an "I don't know" auxiliary option (widely applicable to black-box models). Additionally, a distribution-level bias metric, CKLD, is introduced. The combined approach improves the ARC-Challenge accuracy on Llama-3 from 52.3% to 65.3%.

## Background & Motivation

- **Core Problem**: LLMs exhibit systematic selection bias when answering MCQs—tending to select specific positions (such as the last option) or specific labels (such as "A") regardless of the option content, which severely undermines accuracy and reliability.
- **Limitations of Prior Work**: Previous works focus on input reformatting (e.g., Split-and-Merge, Li et al. 2023) or output probability calibration (e.g., PriDe, Zheng et al. 2024; DoLa, Reif & Schwartz 2024). However, these methods only apply external patches, ignoring the internal mechanisms of bias generation.
- **Real-world Impact**: Voting experiments in Figure 2 reveal that the majority-voting accuracy across all option permutations for four LLMs is significantly higher than the single-inference accuracy, proving that selection bias is a universal issue across models.
- **Key Finding 1**: By analyzing the selection frequency distribution under option permutations, it is observed that **the bias of incorrect samples is far greater than that of correct samples**—when the model answers incorrectly, the selection distribution exhibits a sharper imbalance (e.g., Llama-3 prefers "D" and Bloomz prefers "A").
- **Key Finding 2**: By extracting embeddings from various layers and token positions of the model and calculating the $L_2$ norm of the difference vector between correct and incorrect answers, it is found that **selection bias is primarily concentrated in the final layers of the decoder**, particularly at the interaction between the output projection matrix and the final-layer embeddings.

## Method

### Overall Architecture

This paper proposes two complementary debiasing methods and a new evaluation metric:

| Component | Type | Applicability | Core Idea |
|------|------|----------|----------|
| BNP (Bias Node Pruning) | Parameter Pruning | White-box models | Prune rows in the output projection matrix that interact most strongly with the bias vector |
| AOI (Auxiliary Option Injection) | Input Prompting | White-box + Black-box | Inject an "I don't know" auxiliary option to absorb uncertainty |
| CKLD (Choice KL Divergence) | Evaluation Metric | Universal | Measure the discrepancy between the predicted distribution and the ground-truth label distribution using KL divergence |

### Key Designs

**1. Bias Node Pruning (BNP)**

The core idea is to model the output of the biased LLM as the product of an "unbiased model + a bias vector" and the output projection matrix: $F(x) \approx (D(x) + b) \cdot W$, where the bias term $b \cdot W$ directly causes the output shift. The specific steps are as follows:

- **Bias Vector Calculation**: For each question $x$, all permutations of options are fed into the model. The final-layer embeddings for correct and incorrect permutations are collected to calculate the difference vector $b_x = \text{mean}(z_-) - \text{mean}(z_+)$. The global bias vector $b$ is then obtained by averaging over 32 training samples.
- **Bias Node Identification**: The interaction strength $\sum_j b_i \times W_{ij}$ is calculated for each row of the output projection matrix $W \in \mathbb{R}^{d \times |V|}$ with the bias vector $b$. The Top-k rows with the strongest interaction are identified as the set of bias nodes $K$.
- **Parameter Zeroing**: All corresponding rows in $K$ are set to zero to obtain $\tilde{W}$. Subsequent inferences are performed using $\tilde{W}$, modifying only ~0.002% of the model parameters (e.g., pruning only 32 nodes out of Llama-3's 8 billion parameters).
- **Hyperparameter Selection**: 32 nodes are pruned for Llama-3 and Mistral, and 128 nodes for Bloomz, selected via a simple grid search over {16, 32, 64, 128}.

**2. Auxiliary Option Injection (AOI)**

Based on the observation that "incorrect samples are more prone to bias," this mechanism is designed to let the model actively express uncertainty:

- **Option Expansion**: An auxiliary option $o_{\text{aux}}$, "I don't know", is appended to the end of the original option set $A$.
- **Answer Selection**: Based on the output logit probability distribution, the option with the highest probability is selected from the original option set (excluding $o_{\text{aux}}$) as the final answer $\hat{a} = \operatorname{argmax}_{a \in A \setminus \{o_{\text{aux}}\}} P(\hat{y}=a|x_A)$.
- **Black-box Adaptation**: For black-box models where logits are inaccessible, the Jaccard similarity between the generated text and each option is used instead of probability ranking.
- **Ablation Study**: Comparing alternative texts like "None of the above" and "I know the answer", "I don't know" performs best in most scenarios. Multiple IDK options provide additional gains for Llama-3 but are ineffective for other models.

**3. Choice KL Divergence (CKLD)**

Existing metrics (RStd standard deviation, RSD relative standard deviation) only measure the variability of accuracies across classes, making them insensitive to imbalanced ground-truth label distributions (e.g., A accounts for 22%, D accounts for 28%), which can be misleading:

- **Definition**: $\text{CKLD} = \sum_i p_i \log(p_i/q_i)$, where $p_i$ is the ratio of the $i$-th option in the ground-truth labels, and $q_i$ is the ratio of the $i$-th option in the model's predictions.
- **Theoretical Guarantee**: Using the Lagrangian method, it is proven that CKLD reaches its minimum value of 0 if and only if $q_i = p_i$ (i.e., the predicted distribution matches the ground-truth distribution).
- **Contrast of Advantages**: Synthetic data experiments show that under label imbalance, the minimum point of RSD deviates from the true value (always residing at $1/k$), whereas CKLD accurately reflects the point of minimum bias.

## Key Experimental Results

### Main Results (BNP + AOI)

Validated across 3 models $\times$ 3 datasets:

| Model + Method | ARC Acc↑ | ARC CKLD↓ | MMLU Acc↑ | MMLU CKLD↓ | CSQA Acc↑ | CSQA CKLD↓ |
|-------------|----------|-----------|-----------|------------|-----------|------------|
| Llama-3 | 52.3 | 0.494 | 41.8 | 0.589 | 65.4 | 0.095 |
| Llama-3 + BNP | 56.7 | 0.302 | 43.1 | 0.501 | 66.6 | 0.074 |
| Llama-3 + AOI | 60.7 | 0.231 | 47.3 | 0.321 | 67.4 | 0.065 |
| **Llama-3 + BNP+AOI** | **65.3** | **0.124** | **48.3** | **0.288** | **68.1** | **0.049** |
| Bloomz | 43.9 | 0.283 | 28.0 | 0.661 | 58.5 | 0.136 |
| **Bloomz + BNP+AOI** | **48.8** | **0.088** | **32.0** | **0.205** | **64.9** | **0.052** |
| Mistral | 67.4 | 0.040 | 46.4 | 0.186 | 63.6 | 0.042 |
| **Mistral + BNP+AOI** | **69.5** | **0.019** | **48.6** | **0.140** | **66.8** | **0.016** |

### Compatibility with Existing Methods

BNP+AOI is orthogonally compatible with existing methods such as CoT, ICL, and DoLa:

| Method (Llama-3, ARC) | Acc↑ | CKLD↓ |
|----------------------|------|-------|
| CoT | 66.2 | 0.050 |
| CoT + Ours (BNP+AOI) | **69.2** | **0.024** |
| ICL | 62.2 | 0.169 |
| ICL + Ours (BNP+AOI) | **70.0** | **0.054** |
| DoLa | 51.1 | 0.524 |
| DoLa + Ours (BNP+AOI) | **64.1** | **0.139** |

### Black-Box Model Validation

AOI is equally effective in black-box scenarios where model parameters are inaccessible:

| Model | ARC Acc | ARC+AOI Acc | CSQA Acc | CSQA+AOI Acc |
|------|---------|-------------|----------|--------------|
| Claude-3-Haiku | 65.3 | **71.4** (+6.1) | 36.4 | **47.0** (+10.6) |
| Claude-3-Sonnet | 86.9 | **87.6** (+0.7) | 71.0 | **73.1** (+2.1) |

### Key Findings

- **BNP is insensitive to the number of pruned nodes**: The performance stably outperforms the baseline from 8 to 128 nodes, but fine-tuning can yield further optimization.
- **Transferability of bias vectors across datasets**: The bias vector calculated using the ARC dataset reduces CKLD on CSQA by 36%, which even outperforms the bias vector computed from CSQA itself (22%), indicating that the bias vector captures inherent model properties.
- **BNP does not affect generation quality**: On sentiment analysis and text summarization tasks, pruning 32 nodes leads to only a minor performance drop (F1: 32.7 $\rightarrow$ 31.3), with a negligible impact on general language capabilities.
- **Visualization of distribution shifts**: After applying BNP+AOI, the model's option choice frequency distribution tends toward uniformity (approaching the ideal uniform ratio marked by the dashed lines).

## Highlights & Insights

1. **Mechanistic Approach**: First to localize the source of selection bias at the parameter level of the output projection matrix, rather than merely performing external calibration.
2. **Minimalist Pruning**: Modifying only 0.002% of the parameters yields up to +24.9% relative accuracy improvement with virtually zero computational overhead.
3. **Elegant Design of AOI**: Appending a harmless "I don't know" option significantly reduces bias, remaining highly effective for black-box models.
4. **CKLD Fills the Metric Gap**: While existing metrics like RStd/RSD fail under class imbalance, CKLD accurately measures distribution-level bias via KL divergence.
5. **Full Scenario Coverage**: BNP targets white-box setups while AOI is suitable for black-box ones; their combined use is orthogonal and can further enhance performance when stacked with CoT/ICL.

## Limitations & Future Work

- BNP requires a small amount of annotated calibration data (all permutations of 32 samples), and its generalization to out-of-distribution (OOD) scenarios has not been fully verified.
- The hyperparameter $k$ (number of pruned nodes) must be manually searched for each model, with no automatic determination method provided.
- The methods are only validated on MCQ tasks; whether they apply to biases in open-ended generation tasks remains an open question.
- Calculating the bias vector requires $N!$ permutation inferences (where $N$ is the number of options), causing the cost to escalate rapidly as the number of options increases.
- The root causes of bias (e.g., human cognitive biases in training data? symbol encoding in the tokenizer?) remain unresolved.

## Related Work & Insights

- **Input-side Debiasing**: Split-and-Merge (Li et al. 2023), option order shuffling (Robinson et al. 2023).
- **Output-side Calibration**: PriDe (Zheng et al. 2024), DoLa contrastive decoding (Chuang et al. 2023), label bias quantification (Reif & Schwartz 2024).
- **Structured Pruning**: LLM-Pruner (Ma et al. 2023), Deep Compression (Han et al. 2016).
- **Symbol Binding**: Enhancing MCQ symbol binding training (Xue et al. 2024).
- **Inspiration from Survey Science**: The inclusion of an "I don't know" option in human surveys improves data quality (Schuman & Presser 1996).

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The approach combining bias localization and parameter-level pruning is novel; AOI draws inspiration from survey science.
- **Technical Depth**: ⭐⭐⭐⭐ — Features a complete logical progression from embedding analysis to bias modeling and pruning.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive evaluation with 3 models $\times$ 4 datasets, combined with CoT/ICL/DoLa, black-box validation, and extensive ablation studies.
- **Value**: ⭐⭐⭐⭐⭐ — Plug-and-play, applicable to both white-box and black-box models, and orthogonal to existing methods.
- **Overall Rating**: ⭐⭐⭐⭐ — Simple yet effective methodology, addressing a practical pain point in LLM MCQ debiasing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Disentangling the Roles of Representation and Selection in Data Pruning](disentangling_the_roles_of_representation_and_selection_in_data_pruning.md)
- [\[ICML 2025\] Beyond Communication Overhead: A Multilevel Monte Carlo Approach for Mitigating Compression Bias in Distributed Learning](../../ICML2025/model_compression/beyond_communication_overhead_a_multilevel_monte_carlo_approach_for_mitigating_c.md)
- [\[ACL 2025\] Wanda++: Pruning Large Language Models via Regional Gradients](wanda_pruning_large_language_models_via_regional_gradients.md)
- [\[ACL 2025\] STUN: Structured-Then-Unstructured Pruning for Scalable MoE Pruning](stun_moe_pruning.md)
- [\[ACL 2025\] LongReD: Mitigating Short-Text Degradation of Long-Context Large Language Models via Restoration Distillation](longred_mitigating_short-text_degradation_of_long-context_large_language_models_.md)

</div>

<!-- RELATED:END -->

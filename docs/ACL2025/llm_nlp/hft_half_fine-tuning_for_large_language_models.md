---
title: >-
  [Paper Note] HFT: Half Fine-Tuning for Large Language Models
description: >-
  [ACL 2025][LLM (Other)][Half Fine-Tuning] This paper proposes Half Fine-Tuning (HFT), which randomly freezes half of the parameters and updates only the other half during each fine-tuning epoch. Without altering the model architecture, HFT significantly mitigates catastrophic forgetting while achieving comparable or even superior performance on downstream tasks compared to Full Fine-Tuning (FFT), reducing training time by approximately 30%.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Half Fine-Tuning"
  - "Catastrophic Forgetting"
  - "Parameter Selection"
  - "Continual Learning"
  - "DPO"
date: 2026-05-08
content_hash: 6eb40927a58f006c
---

# HFT: Half Fine-Tuning for Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2404.18466](https://arxiv.org/abs/2404.18466)  
**Area**: LLM Fine-Tuning / Catastrophic Forgetting  
**Keywords**: Half Fine-Tuning, Catastrophic Forgetting, Parameter Selection, Continual Learning, DPO  

## TL;DR

This paper proposes Half Fine-Tuning (HFT), which randomly freezes half of the parameters and updates only the other half during each fine-tuning epoch. Without altering the model architecture, HFT significantly mitigates catastrophic forgetting while achieving comparable or even superior performance on downstream tasks compared to Full Fine-Tuning (FFT), reducing training time by approximately 30%.

## Background & Motivation

**Catastrophic Forgetting in LLM Fine-Tuning**: Although large language models unlock powerful downstream capabilities after undergoing multi-stage training (pre-training $\rightarrow$ SFT $\rightarrow$ DPO), the parameterized world knowledge acquired during the pre-training phase faces a severe risk of catastrophic forgetting.

**Limitations of Prior Work**:
   - **Extra-Module Methods** (e.g., LoRA): Keep pre-trained parameters frozen and add auxiliary trainable modules, which alters the model architecture and introduces obstacles to deployment and subsequent fine-tuning.
   - **Full Fine-Tuning (FFT)**: Updates all parameters, where new knowledge might overwrite old knowledge.

**Key Findings** (Pilot Experiments):
   - Partially dropping or pruning the task vector (the difference between the fine-tuned and pre-trained parameters) has a minimal impact on target tasks (Yadav et al., 2023).
   - This implies: **A subset of new parameters is sufficient to learn new tasks.**
   - A natural corollary: **Can a subset of old parameters sustain the pre-trained capabilities?**

**Half-Reset Experiment**: After resetting 50% of the parameters of Llama 2-Chat-7b back to Llama 2-7b, the Half-Reset model substantially recovered its base knowledge while maintaining the excellent general capabilities of the Chat version. This finding directly inspired HFT.

## Method

### Overall Architecture

The core idea of HFT is extremely simple: in each fine-tuning epoch, 50% of the parameters are randomly selected to be updated, while the other 50% are kept frozen.

$$\vartheta^t \leftarrow \vartheta^{t-1} - \eta \nabla_\vartheta \mathcal{L}(\theta^{t-1}), \quad \psi^t \leftarrow \psi^{t-1}$$

Where $\vartheta^t$ represents the parameters to be updated, $\psi^t$ represents the frozen parameters, and $\theta^t = \{\vartheta^t, \psi^t\}$.

### Parameter Selection Strategy (Category-level)

Taking the Llama 2 architecture as an example, in each Transformer layer:
- **Self-Attention**: Select two of the four matrices ($W_Q, W_K, W_V, W_O$) to update.
- **Feed-Forward Network (FFN)**: Out of the three matrices ($W_{up}, W_{down}, W_{gate}$), select two from one half of the layers and one from the other half, precisely guaranteeing a 50% ratio.
- **LayerNorm**: Similarly, select half of them.
- **Embedding and LM_head**: Updated by default.

### Theoretical Analysis

HFT can be formulation-wise equivalent to an FFT optimization problem with a regularization term:

$$\mathcal{O}_L = \min_\theta \max_\lambda \mathcal{L}(\theta) + \lambda \|(I-M)(\theta - \theta^0)\|^2$$

Where $M$ is the parameter mask matrix. Through the Minimax inequality, it can be derived that HFT optimizes the upper bound of the FFT loss function plus a regularization term $\|(I-M)(\theta - \theta^0)\|^2$. This theoretically explains why HFT has the potential to achieve comparable or even superior results to FFT—the regularization enhances the stability of the sparsely fine-tuned model.

### Application in Continual Learning

In multi-epoch fine-tuning scenarios, the sets of frozen and updated parameters differ in each epoch (randomly selected). This helps maintain a balance of knowledge across different epochs, demonstrating significant scalability.

## Key Experimental Results

### SFT Stage: General Capabilities (Tülu V2)

| Model | MMLU | GSM8K | BBH | TyDiQA | TruthfulQA | HumanEval | Average |
|------|------|-------|-----|--------|------------|-----------|------|
| Llama2-7b (Pre-trained) | 41.6 | 12.0 | 39.9 | 48.4 | 38.5 | 26.2 | 34.4 |
| Llama2-7b-SFT (FFT) | 48.5 | 25.0 | 42.2 | 51.2 | 41.7 | 36.9 | 41.0 |
| Llama2-7b-SFT (**HFT**) | **50.8** | **30.5** | **43.6** | **52.3** | **45.4** | 34.6 | **42.9 (+1.9)** |
| Llama2-13b-SFT (FFT) | 50.6 | 45.0 | 47.8 | 55.0 | 42.6 | 42.4 | 47.2 |
| Llama2-13b-SFT (**HFT**) | **54.5** | **46.5** | **53.7** | **56.7** | **45.7** | **43.5** | **50.1 (+2.9)** |

### SFT Stage: Base Knowledge Retention

| Model | NaturalQ | TriviaQA | HotpotQA | Average |
|------|----------|----------|----------|------|
| Llama2-7b (Pre-trained) | 12.9 | 40.2 | 15.6 | 22.9 |
| Llama2-7b-SFT (FFT) | 3.2 | 26.4 | 14.5 | 14.7 |
| Llama2-7b-SFT (**HFT**) | **6.2** | **32.8** | **15.4** | **18.1 (+3.4)** |
| Llama2-13b-SFT (FFT) | 0.7 | 9.2 | 4.9 | 4.9 |
| Llama2-13b-SFT (**HFT**) | **2.7** | **12.4** | **8.2** | **7.8 (+2.9)** |

FFT leads to a precipitous decline in base knowledge, whereas HFT significantly mitigates forgetting.

### DPO Stage

| Model | Average General Capability | Average Base Knowledge |
|------|------------|------------|
| Llama2-7b-DPO (FFT) | 41.9 | 10.7 |
| Llama2-7b-DPO (**HFT**) | 41.7 (-0.2) | **12.5 (+1.8)** |
| Llama2-13b-DPO (FFT) | 47.4 | 2.3 |
| Llama2-13b-DPO (**HFT**) | **48.2 (+0.8)** | **2.9 (+0.6)** |

In the DPO stage, HFT performs comparably in general capabilities and consistently outperforms FFT in base knowledge.

### Continual Learning (TRACE Benchmark)

| Method | OP (FFT) | OP (HFT) | BWT (FFT) | BWT (HFT) |
|------|----------|----------|-----------|-----------|
| SeqFT (7b) | 45.7 | **51.3 (+5.6)** | -10.2% | **-5.6% (+4.6%)** |
| GEM (7b) | 48.2 | **50.2 (+2.0)** | -7.9% | **-5.9% (+2.0%)** |
| Replay (7b) | 54.3 | 54.1 (-0.2) | +1.4% | **+2.1% (+0.7%)** |
| SeqFT (13b) | 49.0 | **52.0 (+3.0)** | -9.4% | **-8.5% (+0.9%)** |
| GEM (13b) | 50.4 | **53.6 (+3.2)** | -8.9% | **-6.1% (+2.8%)** |
| Replay (13b) | 54.7 | **57.4 (+2.7)** | -0.6% | **+1.6% (+2.2%)** |

As a plug-and-play solution, HFT benefits almost all continual learning methods.

### Comparison of Parameter Selection Strategies (TRACE SeqFT)

| Strategy | OP | BWT |
|------|----|----|
| FFT | 45.7 | -10.2% |
| Model-level HFT | 46.9 (+1.2) | -9.2% |
| Layer-level HFT | 47.9 (+2.2) | -8.3% |
| **Category-level HFT** | **51.3 (+5.6)** | **-5.6%** |

Category-level selection (fine-grained selection by parameter category) performs the best, as it maximizes the interaction between updated and frozen parameters.

### Training Efficiency

HFT **reduces training time by approximately 30%** in standard SFT, as freezing half of the parameters reduces the overhead of gradient computation and optimizer states.

## Highlights & Insights

1. **Minimalist Approach, Significant Effect**: Only a few lines of code for masking are required. It significantly mitigates forgetting without changing the architecture or introducing extra parameters.
2. **Insight Chain from Half-Reset to HFT**: First verifying that "resetting half of the parameters can recover old knowledge", and then deriving that "freezing half of the parameters can prevent forgetting", forming a complete, logical workflow.
3. **Clear Theoretical Explanation**: Equating parameter freezing to a regularization term connects this work with the theory of sparse fine-tuning.
4. **High Robustness**: Insensitive to specific parameter selection strategies and exact ratios; a tuning ratio of approximately 40-60% consistently achieves good results.
5. **Universal Applicability**: Effective across three scenarios—SFT, DPO, and continual learning—and can be combined with other methods such as GEM and Replay.

## Limitations & Future Work

1. **Heuristic Nature of the 50% Ratio**: Although experiments show that a ratio around 50% performs best, the optimal ratio may vary depending on the task and model.
2. **Variance of Random Selection**: Random selection during each epoch might introduce training instability.
3. **Disregard for Parameter Importance**: All parameters are treated equally, without considering which parameters are actually more critical (e.g., those vital for storing knowledge).
4. **Limited Experimental Scale**: Experiments were primarily conducted on Llama 2 7B/13B, with the effectiveness on larger models (70B+) remaining unverified.
5. **Unfair Comparison with LoRA**: LoRA performs poorly on TRACE, but its design goals differ from those of HFT.

## Related Work & Insights

- **Catastrophic Forgetting**: Lin et al. (2024) analyze the destruction of LLM knowledge by fine-tuning; Neeman et al. (2023) study knowledge forgetting after instruction tuning.
- **Parameter-Efficient Fine-Tuning**: LoRA (Hu et al., 2022) reduces parameters via low-rank adaptation; Dou et al. (2023) propose adding auxiliary modules while keeping pre-trained parameters frozen.
- **Task Vector Manipulation**: Ilharco et al. (2023) define the concept of task vectors; TIES-Merging (Yadav et al., 2023) partially prunes task vectors.
- **Continual Learning**: GEM (Lopez-Paz and Ranzato, 2017), Replay strategies, TRACE benchmark (Wang et al., 2023a).

## Rating

⭐⭐⭐⭐⭐ — The method is extremely simple yet highly effective, with clear theoretical explanations and comprehensive experimental coverage (across SFT, DPO, and CL scenarios). Its plug-and-play nature gives it immense practical value. "No architecture modification, no extra parameters, reduced training time, mitigated forgetting, and improved performance" — a rare win-win solution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] GORP: Continual Gradient Low-Rank Projection Fine-Tuning for LLMs](gorp_continual_gradient_projection.md)
- [\[ACL 2025\] PiFi: Plug-in and Fine-tuning: Bridging the Gap between Small Language Models and Large Language Models](plugin_finetuning_bridge.md)
- [\[ACL 2025\] Efficient Ensemble for Fine-tuning Language Models on Multiple Datasets](efficient_ensemble_for_fine-tuning_language_models_on_multiple_datasets.md)
- [\[ACL 2025\] Beyond Demographics: Fine-tuning Large Language Models to Predict Individuals' Subjective Text Perceptions](beyond_demographics_fine-tuning_large_language_models_to_predict_individuals_sub.md)
- [\[ACL 2025\] A Semantic-Aware Layer-Freezing Approach to Computation-Efficient Fine-Tuning of Language Models](a_semantic-aware_layer-freezing_approach_to_computation-efficient_fine-tuning_of.md)

</div>

<!-- RELATED:END -->

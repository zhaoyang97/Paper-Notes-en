---
title: >-
  [Paper Note] FairSteer: Inference Time Debiasing for LLMs with Dynamic Activation Steering
description: >-
  [ACL2025][Social Computing][debiasing] Proposed FairSteer, an inference-time debiasing framework that detects bias signals in activations using a lightweight linear classifier, and then dynamically adjusts hidden layer activations using a Debiasing Steering Vector (DSV) calculated from contrastive prompt pairs. This effectively mitigates social bias in LLMs across multiple tasks without retraining.
tags:
  - "ACL2025"
  - "Social Computing"
  - "debiasing"
  - "fairness"
  - "activation steering"
  - "linear representation"
  - "inference-time"
date: 2026-05-08
content_hash: 79a08b2bce38fd19
---

# FairSteer: Inference Time Debiasing for LLMs with Dynamic Activation Steering

**Conference**: ACL2025  
**arXiv**: [2504.14492](https://arxiv.org/abs/2504.14492)  
**Code**: [GitHub](https://github.com/LiYichen99/FairSteer)  
**Area**: Social Computing  
**Keywords**: debiasing, fairness, activation steering, linear representation, inference-time

## TL;DR

Proposed FairSteer, an inference-time debiasing framework that detects bias signals in activations using a lightweight linear classifier, and then dynamically adjusts hidden layer activations using a Debiasing Steering Vector (DSV) calculated from contrastive prompt pairs. This effectively mitigates social bias in LLMs across multiple tasks without retraining.

## Background & Motivation

**Background**: LLMs inherit social biases such as race, gender, and age from their training corpora, which negatively impacts marginalized groups. Debiasing is a crucial component of AI alignment.

**Limitations of Prior Work**: Prompting methods (e.g., CAL) are sensitive to wording and yield unstable results; fine-tuning methods (projection, contrastive learning, reinforcement learning) require substantial computational resources and annotated data, while carrying the risk of catastrophic forgetting; decoding strategy methods (e.g., constrained search, reranking) reduce output diversity and are mostly designed for older models.

**Key Challenge**: How to precisely mitigate bias during the inference phase without retraining or relying on complex prompt designs, while maintaining the original capabilities of the model?

**Goal**: Design an inference-time debiasing framework that intervenes only when bias is detected, avoiding disruption to unbiased outputs.

**Key Insight**: Based on the linear representation hypothesis—which states that semantic concepts like truthfulness, sentiment, and humor are encoded as linearly separable directions in the activation space of LLMs—this study investigates whether fairness features are similarly linearly separable.

**Core Idea**: Bias features are linearly separable in the mid-layer activation space (>90% classification accuracy), enabling debiasing via geometric intervention—translating the activation vectors along the debiasing direction.

## Method

### Overall Architecture

FairSteer operates in three steps: (1) training a linear classifier to perform Biased Activation Detection (BAD); (2) computing a Debiasing Steering Vector (DSV) using contrastive prompt pairs; and (3) performing Dynamic Activation Steering (DAS) during inference by dynamically applying the DSV to adjust activations only when bias is detected.

### Key Design 1: Biased Activation Detection (BAD)

- **Function**: Trains a lightweight linear classifier $C^l$ at each layer to determine whether the activation vector $\mathbf{a}^l$ of the last token corresponds to a biased output.
- **Design Motivation**: Conditional intervention is needed—applying the DSV only when bias is detected to avoid disrupting unbiased outputs. Indiscriminate application of DSV leads to a significant decrease in accuracy (as confirmed by ablation experiments).
- **Mechanism**: A mixed dataset constructed from BBQ (58,492 examples) and MMLU (10,266 examples) is used to create the training set. Model choices selecting stereotypical answers are labeled as biased ($y=0$), while choices selecting neutral answers are labeled as unbiased ($y=1$). Last-token activations from each layer are extracted, and a binary cross-entropy loss with L2 regularization is used to train $\hat{y} = \sigma(\mathbf{w}^T \mathbf{a}^l + b)$. MMLU data is mixed in to prevent overfitting to specific bias domains.

### Key Design 2: Debiasing Steering Vector (DSV) Computation

- **Function**: Computes a geometrically interpretable intervention direction that captures the activation space shift from biased to unbiased.
- **Design Motivation**: PCA visualization shows that biased and unbiased activations form clearly separated clusters in the middle layers (Figure 3). The mean difference vector between them is the most natural debiasing direction.
- **Mechanism**: Ten contrastive prompt pairs $(\mathcal{P}^+, \mathcal{P}^-)$ are sampled from each of the 9 bias categories and 2 intersectional biases in BBQ, totaling 110 pairs. $\mathcal{P}^+$ and $\mathcal{P}^-$ share the same context but have different answer options (guiding unbiased and biased responses, respectively). The DSV is computed as: $\mathbf{v}^l = \frac{1}{N}\sum[{\mathbf{a}^l(\mathcal{P}^+) - \mathbf{a}^l(\mathcal{P}^-)}]$. The DSV encodes both the direction (optimal debiasing trajectory) and the magnitude (distance between subspaces).

### Key Design 3: Dynamic Activation Steering (DAS)

- **Function**: Performs conditional activation adjustment at the selected layer $l^*$ during inference.
- **Design Motivation**: The middle layers (layers 13-15) show the highest compatible classification accuracy across all evaluated models, balancing low-level token representations and high-level semantic features.
- **Mechanism**: The last-token activation $\mathbf{a}^{l^*}$ of the input at layer $l^*$ is extracted and fed into the classifier to obtain the bias probability $\hat{y}$. If $\hat{y} < 0.5$ (bias detected), the activation is adjusted as: $\mathbf{a}^{l^*}_{\text{adj}} = \mathbf{a}^{l^*} + \mathbf{v}^{l^*}$. The adjusted activation is then propagated to subsequent layers, guiding the generation toward an unbiased output.

### Loss & Training

- The entire framework only requires training linear classifiers (taking seconds) and computing the DSV (a single forward pass of 110 samples).
- No modifications to model parameters are required, operating as a plug-and-play inference plugin.
- Layer selection: 2,200 BBQ samples are generated to evaluate the classification accuracy of each layer, selecting the highest-performing layer (typically layers 13-15).

## Key Experimental Results

### Main Results: BBQ QA Debiasing (Selected Models)

| Model | Method | ZS Acc ↑ | ZS BS(a) ↓ | FS Acc ↑ | FS BS(a) ↓ |
|------|------|:---:|:---:|:---:|:---:|
| Llama2-13B | Base | 48.60 | 5.86 | 47.94 | 16.31 |
| | CAL | 51.29 | 1.41 | 53.27 | 9.82 |
| | **FairSteer** | **74.02** | **-0.82** | **80.26** | **1.58** |
| Llama3-8B | Base | 71.00 | 13.62 | 84.74 | 13.53 |
| | CAL | 55.51 | 0.08 | 82.65 | 2.61 |
| | **FairSteer** | **90.22** | 1.46 | **92.12** | 4.39 |
| Vicuna-13B | Base | 63.71 | 4.97 | 64.74 | 15.72 |
| | CAL | 47.99 | 0.72 | 63.72 | 12.11 |
| | **FairSteer** | **77.74** | **0.10** | **86.56** | **1.28** |

### Ablation Study: Effect of BAD (BBQ)

| Model | Method | ZS Acc ↑ | FS Acc ↑ |
|------|------|:---:|:---:|
| Llama2-13B | Base | 48.60 | 47.94 |
| | DSV only | 52.84 | 55.46 |
| | **FairSteer** | **74.02** | **80.26** |
| Llama3-8B | Base | 71.00 | 84.74 |
| | DSV only | 62.21 | 74.11 |
| | **FairSteer** | **90.22** | **92.12** |
| Vicuna-7B | Base | 41.33 | 43.89 |
| | DSV only | 55.48 | 55.66 |
| | **FairSteer** | **65.38** | **71.28** |

### General Capabilities Preservation

| Model | Method | MMLU ↑ | ARC-E ↑ | ARC-C ↑ | OBQA ↑ |
|------|------|:---:|:---:|:---:|:---:|
| Llama3-8B | Base | 68.37 | 93.56 | 83.53 | 81.60 |
| | FairSteer | 68.34 | 93.56 | 83.53 | 81.60 |
| Vicuna-13B | Base | 55.88 | 83.25 | 68.26 | 64.40 |
| | FairSteer | 55.76 | 83.25 | 68.26 | 64.40 |

### Key Findings

- FairSteer substantially improves accuracy across all six models on BBQ (up to +25.42 zero-shot accuracy for Llama2-13B) while reducing bias scores.
- After removing BAD (relying only on DSV), accuracy drops significantly (e.g., Llama3-8B zero-shot accuracy drops from 90.22 to 62.21), falling even below the baseline model. This demonstrates that conditional intervention is crucial—indiscriminately applying DSV harms unbiased samples.
- Almost no degradation on general tasks: the accuracy changes on MMLU/ARC/OBQA are within 0.5%, and perplexity (PPL) changes are also minimal.
- Bias scores are consistently reduced on CrowS-Pairs (counterfactual evaluation) and CEB (open-ended generation), verifying cross-task generalization.
- CAL exhibits over-debiasing on some models, leading to accuracies below the baseline (e.g., Vicuna-7B, Llama2-7B), whereas FairSteer is more stable.

## Highlights & Insights

1. **Verification of Linear Separability**: For the first time, fairness features are systematically verified to be linearly separable (>90%) in the mid-layer activation spaces of six LLMs, providing a solid theoretical foundation for geometric intervention-based debiasing.
2. **Conditional Intervention Design**: As a "gatekeeper," BAD triggers DSV only when bias is detected, which is key to preserving general capabilities. Ablation studies clearly demonstrate the hazards of unconditional intervention.
3. **Extremely Low Data Requirements**: DSV requires only 110 contrastive samples for computation, which is significantly less than the data required by fine-tuning methods.
4. **Plug-and-Play**: Operating as an inference-time plugin, FairSteer requires no modification of model parameters, architectural changes, or special prompting, making it applicable to any Transformer-based LLM.

## Limitations & Future Work

1. **Limitations of Linear Classifiers**: Linear classifiers may fail to capture more complex, non-linear bias patterns. Future work could explore lightweight non-linear probes.
2. **Suboptimal DSV Extraction**: Computing the direct mean difference may not be the optimal method for extracting the debiasing direction; the authors themselves describe it as a "proof of concept" rather than the optimal technique.
3. **Dependence on Contrastive Prompt Quality**: The effectiveness of DSV depends on the quality and representativeness of the contrastive prompt pairs, which may not cover all real-world bias types.
4. **Incomplete Generalization Verification**: Evaluated only on six open-source models (7B-13B); applicability to larger-scale or closed-source models remains unknown.
5. **Bias Type Coverage**: Primarily focuses on the nine bias categories in BBQ, with a lack of discussion on more subtle and implicit biases (e.g., socioeconomic status, political orientation).

## Related Work & Insights

### vs. CAL (In-context Prompting)
CAL identifies bias patterns through causally guided active learning and performs debiasing using in-context learning. However, CAL exhibits unstable performance across multiple models; for example, its accuracy on Vicuna-7B is lower than the baseline model, showing that prompting methods are highly sensitive to models and phrasing. In contrast, FairSteer operates directly in the activation space, avoiding the uncertainty of prompt engineering and achieving more stable performance across all evaluated models.

### vs. Activation Steering (e.g., RepE / Refusal Removal)
Arditi et al. (2024) demonstrated that refusal behaviors can be removed via activation steering, while Li et al. (2024) utilized linear probes to detect truthfulness directions. FairSteer extends this paradigm to the domain of fairness and innovatively incorporates the BAD gating mechanism, which is a key distinction. Direct steering without gating (such as "DSV only" in the ablation study) performs significantly worse on fairness tasks compared to the gated version, indicating that bias mitigation requires conditional intervention more than features like truthfulness or sentiment.

### vs. Fine-tuning Debiasing Methods
Projection methods (Ravfogel et al., 2020) and contrastive learning methods (He et al., 2022) achieve debiasing by modifying model parameters. While direct, these approaches are computationally expensive, requiring full fine-tuning, large annotated datasets, and posing risks of catastrophic forgetting. In comparison, FairSteer requires only 110 samples and a linear classifier with negligible computational cost, offering a much more practical alternative.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Expanding the linear representation hypothesis to the fairness domain and introducing conditional intervention gating is a meaningful innovation, though activation steering itself has prior precedents.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Very comprehensive, covering 6 models $\times$ 4 datasets $\times$ 3 task types, including ablation studies, general capability tests, category-wise analyses, and case studies.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear framework with a naturally progressing three-step process and rich diagrams; however, some mathematical notations are redundant.
- **Value**: ⭐⭐⭐⭐ — Provides a practical path for inference-time debiasing, with its plug-and-play nature offering strong potential for engineering applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Adaptive Debiasing Tsallis Entropy for Test-Time Adaptation](../../ICLR2026/social_computing/adaptive_debiasing_tsallis_entropy_for_test-time_adaptation.md)
- [\[ACL 2026\] LiveFact: A Dynamic, Time-Aware Benchmark for LLM-Driven Fake News Detection](../../ACL2026/social_computing/livefact_a_dynamic_time-aware_benchmark_for_llm-driven_fake_news_detection.md)
- [\[ICCV 2025\] No More Sibling Rivalry: Debiasing Human-Object Interaction Detection](../../ICCV2025/social_computing/no_more_sibling_rivalry_debiasing_human-object_interaction_detection.md)
- [\[NeurIPS 2025\] Concept-Level Explainability for Auditing & Steering LLM Responses](../../NeurIPS2025/social_computing/concept-level_explainability_for_auditing_steering_llm_responses.md)
- [\[ACL 2025\] Exploring the Impact of Instruction-Tuning on LLMs' Susceptibility to Misinformation](exploring_the_impact_of_instruction-tuning_on_llms_susceptibility_to_misinformat.md)

</div>

<!-- RELATED:END -->

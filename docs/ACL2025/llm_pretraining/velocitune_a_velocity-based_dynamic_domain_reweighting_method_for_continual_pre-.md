---
title: >-
  [Paper Note] Velocitune: A Velocity-based Dynamic Domain Reweighting Method for Continual Pre-training
description: >-
  [ACL 2025][LLM Pretraining][Continual Pre-training] The Velocitune framework is proposed to dynamically adjust sampling weights of different data domains during continual pre-training based on learning velocity. It prioritizes domains with slower learning progress and estimates target losses cost-effectively using scaling laws, significantly outperforming static mixing baselines on mathematical/code reasoning and system command generation tasks.
tags:
  - "ACL 2025"
  - "LLM Pretraining"
  - "Continual Pre-training"
  - "Domain Reweighting"
  - "Learning Velocity"
  - "Scaling Law"
  - "Data Mixing"
date: 2026-05-08
content_hash: 959185df609a702a
---

# Velocitune: A Velocity-based Dynamic Domain Reweighting Method for Continual Pre-training

**Conference**: ACL 2025  
**arXiv**: [2411.14318](https://arxiv.org/abs/2411.14318)  
**Code**: None  
**Area**: NLP / Pre-training  
**Keywords**: Continual Pre-training, Domain Reweighting, Learning Velocity, Scaling Law, Data Mixing

## TL;DR

The Velocitune framework is proposed to dynamically adjust sampling weights of different data domains during continual pre-training based on learning velocity. It prioritizes domains with slower learning progress and estimates target losses cost-effectively using scaling laws, significantly outperforming static mixing baselines on mathematical/code reasoning and system command generation tasks.

## Background & Motivation

Large language model pre-training data typically consists of a mixture of multiple domains, and the domain ratios directly affect downstream performance. Existing methods include: (1) heuristically testing different ratios repeatedly (prohibitively expensive); (2) dynamic adjustment methods like DoReMi, which utilize small proxy models to estimate optimal weights.

However, it is difficult to directly apply these methods to domain-adaptive continual pre-training for two reasons: (1) smaller versions of the base model are typically unavailable as proxies in continual pre-training scenarios; (2) distance-based methods (which measure the difference between current loss and target loss) can cause certain domains to be overemphasized.

The core idea of Velocitune is to measure the learning progress of each domain using "learning velocity" instead of "distance", focusing on relative progress rather than absolute gaps to achieve more balanced cross-domain learning.

## Method

### Overall Architecture

Two stages: (1) target estimation stage—training a proxy model on a subset using the Chinchilla scaling law to extrapolate the target loss; (2) velocity-guided training stage—periodically computing the learning velocity of each domain and updating domain weights via exponential weighting.

### Key Designs

1. **Definition of Learning Velocity**: $V_t[i] = (\ell_t^i - \ell_{\text{target}}^i) / (\ell_{\text{init}}^i - \ell_{\text{target}}^i)$. The numerator is the gap between the current evaluation loss and the target loss, and the denominator is the gap between the initial loss and the target loss, representing the normalized remaining learning capacity. A larger $V$ indicates slower learning progress. The key advantage of this definition is that it eliminates the absolute differences in initial and target losses across different domains, enabling a fair comparison across domains.

2. **Target Loss Estimation**: Utilizing the Chinchilla scaling law $$L(N,D) = E + A/N^\alpha + B/D^\beta$$. A proxy model is trained on subset datasets under the original weights, and the evaluation losses of multiple checkpoints are saved to fit parameters, which are then extrapolated to the expected loss on the full dataset as the target.

3. **Exponentially Weighted Update**: $w_t[i] \leftarrow w_{t-m}[i] \cdot \exp(V_t[i]) / \sum_j w_{t-m}[j] \cdot \exp(V_t[j])$. A larger (slower) learning velocity $V$ results in a larger $\exp(V)$ and a higher weight, achieving a "compensating for weaknesses" effect.

4. **Clamp Operation**: Restricting the learning velocity to the range of $[0, 1]$ to prevent overly extreme weight adjustments.

### Loss & Training

Standard NLL loss is used. The learning velocity of each domain is evaluated every $m$ steps to update weights. The total number of training tokens equals one full epoch.

## Key Experimental Results

### Main Results 1 — CodeLlama-7B on Reasoning Datasets

| Model | Math Avg | Code Avg |
|------|---------|---------|
| CodeLlama Base | 21.6% | 36.8% |
| Baseline (Static Weights) | 34.3% | 35.5% |
| **Velocitune** | **35.9% (+1.6%)** | **39.3% (+3.8%)** |

### Main Results 2 — Llama3/Mistral on System Command Datasets

| Model | CmdGen-NVIDIA Acc | CmdGen-AMD Acc |
|------|-------------------|----------------|
| Llama3-Baseline | 57.07% | 51.79% |
| Llama3-DBL | 45.85% | 45.64% |
| **Llama3-Velocitune** | **61.95% (+4.9%)** | **54.87% (+3.1%)** |
| Mistral-Baseline | 32.20% | 30.77% |
| **Mistral-Velocitune** | **36.59% (+4.4%)** | **33.33% (+2.6%)** |

### Ablation Study — Statistically Using Velocitune Average Weights

| Model | Math Avg | Code Avg |
|------|---------|---------|
| Baseline (Original Weights) | 34.3% | 35.5% |
| Reweighted (Static Average Weights) | 36.6% | 36.6% |
| **Velocitune (Dynamic Adjustment)** | **35.9%** | **39.3%** |

### Key Findings

1. **Velocitune comprehensively outperforms baselines and DBL**: Consistent improvements are observed across two experimental settings and three base models.
2. **DBL method can be counterproductive**: On SystemStack, DBL underperforms compared to the baseline, as distance-based methods lead to unbalanced learning across domains.
3. **Data ordering effect**: When statically mixing using only the average weights of Velocitune, the math task performs better but the code task performs worse, indicating that the dynamic adjustment process itself (the data ordering effect) contributes to the results.
4. **Target loss prediction is crucial**: Performance drops significantly without target loss estimation.
5. **Accelerated weight convergence**: The weights in Velocitune stabilize at least 1.5 times faster than in DBL.

## Highlights & Insights

- Learning velocity is more rational than absolute distance: for the same reduction of 0.1 in loss, a domain with an initial loss of 5 and a domain with an initial loss of 2 signify completely different learning progresses.
- Clever application of scaling laws: Used to estimate "how well the model should perform," avoiding the cost of training a full reference model.
- Analysis of the weight dynamic curve (three-phase changes) provides an in-depth understanding of the learning dynamics of continual pre-training.

## Limitations & Future Work

- The prediction accuracy of scaling laws is limited, and incorrect estimations might propagate into weight adjustments.
- Only validated on 7B/8B models; the effectiveness on larger models remains unknown.
- The selection of the evaluation interval $m$ requires tuning.
- Combination with other advanced training strategies (e.g., curriculum learning) has not been explored.

## Related Work & Insights

- DoReMi utilizes Group DRO but requires a fully-trained reference model; Velocitune avoids this dependency.
- The idea of combining scaling laws to optimize data ratios for continual pre-training might be widely applicable.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Clever combination of the learning velocity concept and scaling law-based target estimation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Detailed ablation analysis across two experimental settings and three base models.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear motivation and intuitive figures.
- **Value**: ⭐⭐⭐⭐ — Direct practical value for optimizing data ratios in continual pre-training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Towards Effective and Efficient Continual Pre-training of Large Language Models](towards_effective_and_efficient_continual_pre-training_of_large_language_models.md)
- [\[ACL 2025\] Improving Continual Pre-training Through Seamless Data Packing](improving_continual_pre-training_through_seamless_data_packing.md)
- [\[ACL 2025\] How Do LLMs Acquire New Knowledge? A Knowledge Circuits Perspective on Continual Pre-Training](how_do_llms_acquire_new_knowledge_a_knowledge_circuits_perspective_on_continual_.md)
- [\[ACL 2025\] Meta-rater: A Multi-dimensional Data Selection Method for Pre-training Language Models](metarater_a_multidimensional_data_selection_method.md)
- [\[ACL 2026\] Data Mixing Agent: Learning to Re-weight Domains for Continual Pre-training](../../ACL2026/llm_pretraining/data_mixing_agent_learning_to_re-weight_domains_for_continual_pre-training.md)

</div>

<!-- RELATED:END -->

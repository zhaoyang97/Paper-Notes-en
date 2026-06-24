---
title: >-
  [Paper Note] Safe Delta: Consistently Preserving Safety when Fine-Tuning LLMs on Diverse Datasets
description: >-
  [ICML 2025][LLM (Other)][LLM safety] Safe Delta proposes a safety-aware post-training defense method that consistently preserves LLM safety across diverse fine-tuning datasets (of varying scales and task types) without sacrificing utility. This is achieved by estimating safety degradation, selectively retaining delta parameters to maximize utility while limiting safety loss, and applying a safety compensation vector to mitigate residual safety loss.
tags:
  - "ICML 2025"
  - "LLM (Other)"
  - "LLM safety"
  - "fine-tuning safety"
  - "alignment preservation"
  - "post-training defense"
  - "delta parameters"
  - "safety compensation vector"
date: 2026-05-08
content_hash: 94f4988052078c73
---

# Safe Delta: Consistently Preserving Safety when Fine-Tuning LLMs on Diverse Datasets

**Conference**: ICML 2025  
**arXiv**: [2505.12038](https://arxiv.org/abs/2505.12038)  
**Area**: LLM/NLP  
**Keywords**: LLM safety, fine-tuning safety, alignment preservation, post-training defense, delta parameters, safety compensation vector

## TL;DR
Safe Delta proposes a safety-aware post-training defense method that consistently preserves LLM safety across diverse fine-tuning datasets (of varying scales and task types) without sacrificing utility. This is achieved by estimating safety degradation, selectively retaining delta parameters to maximize utility while limiting safety loss, and applying a safety compensation vector to mitigate residual safety loss.

## Background & Motivation

**Background**: LLM fine-tuning API services (such as OpenAI's fine-tuning API) have become widely popular, allowing users to upload custom data to customize models. However, the fine-tuning process can disrupt the model's original safety alignment, leading to safety degradation even when using entirely benign datasets.

**Limitations of Prior Work**: Existing defense methods perform inconsistently across diverse fine-tuning datasets: (1) Methods like SafeLoRA and Lisa are effective in specific scenarios but exhibit unstable safety-utility trade-offs when faced with datasets of different sizes and task types; (2) Training-time defenses (e.g., mixing in safety data) increase training costs and are highly sensitive to data ratios; (3) Post-training defense methods like RESTA simply scale down the delta parameters proportionally, lacking a precise estimation of safety degradation.

**Key Challenge**: The fundamental conflict between safety preservation and utility retention—over-constraining delta parameters degrades the utility improvements gained from fine-tuning, while under-constraining leads to severe safety degradation. Different datasets (harmful vs. benign, large-scale vs. small-scale, conversational vs. classification) have vastly different safety impacts, necessitating an adaptive balancing scheme.

**Goal**: How to design a post-training defense method that can automatically balance safety and utility without prior knowledge of the dataset characteristics, while remaining consistently effective across various fine-tuning scenarios?

**Key Insight**: It is observed that among the parameter changes introduced by fine-tuning (delta parameters $\Delta\theta = \theta_{\text{ft}} - \theta_{\text{base}}$), the contributions of different parameters to safety and utility are separable. By accurately estimating the safety impact of each parameter, one can selectively retain parameter changes that contribute significantly to utility but minimally affect safety.

**Core Idea**: Decompose the delta parameters into safety-harmful and safety-neutral parts, selectively preserve safety-neutral parameter changes, and then apply a safety compensation vector to repair residual safety degradation.

## Method

### Overall Architecture
Safe Delta is a three-stage post-training defense pipeline applied after fine-tuning is completed:
- **Stage 1: Safety Degradation Estimation** — Quantifies the impact of each delta parameter on the model's safety.
- **Stage 2: Parameter Selection** — Selects which delta parameters to retain based on safety degradation estimates to maximize utility while minimizing safety loss.
- **Stage 3: Safety Compensation** — Applies a safety compensation vector to mitigate the residual safety degradation remaining after Stage 2.

### Key Designs

1. **Safety Degradation Estimation**:

    - **Function**: Quantifies the safety impact of each delta parameter $\Delta\theta_i$.
    - **Mechanism**: Employs a diagonal approximation of the Fisher Information Matrix (FIM) to estimate the sensitivity of each parameter to the safety loss function. For the safety loss $\mathcal{L}_{\text{safe}}$ of the base model on a safety evaluation set, the safety importance of parameter $i$ is estimated as:
    $s_i = F_i \cdot |\Delta\theta_i|^2$
      where $F_i = \mathbb{E}\left[\left(\frac{\partial \mathcal{L}_{\text{safe}}}{\partial \theta_i}\right)^2\right]$ is the diagonal element of the Fisher Information Matrix.
    - **Design Motivation**: FIM captures safety-related curvature information in the parameter space. Safety-sensitive parameters trigger large safety loss changes even with minor updates, whereas safety-neutral parameters can vary freely without impacting safety.

2. **Delta Parameter Selection**:

    - **Function**: Decides whether to retain or discard each delta parameter based on its safety importance score.
    - **Mechanism**: Given a safety budget $B$, parameter selection is formulated as an optimization problem:
    $$ \max_{\mathbf{m} \in \{0,1\}^n} \sum_i m_i \cdot u_i \quad \text{s.t.} \sum_i m_i \cdot s_i \leq B $$
      where $m_i$ is a binary mask, $u_i$ is the utility importance, and $s_i$ is the safety degradation score.
    - **Practical Solution**: Parameters are sorted by the safety-to-utility ratio $s_i / u_i$, prioritizing the retention of parameters with low ratios (high utility and low safety impact) until the safety budget is exhausted.
    - **Design Motivation**: Formulates the safety-utility trade-off as a solvable constrained optimization problem, avoiding manual threshold tuning.

3. **Safety Compensation Vector**:

    - **Function**: Repairs the residual safety degradation remaining after parameter selection.
    - **Mechanism**: Computes a safety compensation direction $\mathbf{v}_{\text{safe}}$ by calculating gradients over the output discrepancy between the base model and the current model on safety data, then applies the scaled compensation:
    $$ \theta_{\text{final}} = \theta_{\text{base}} + \mathbf{m} \odot \Delta\theta + \alpha \cdot \mathbf{v}_{\text{safe}} $$
      where $\alpha$ is determined via binary search to restore safety metrics to the target level.
    - **Design Motivation**: Parameter selection makes binary decisions (retained/discarded) and lacks fine-grained adjustment capabilities. The compensation vector provides optimization corrections in continuous space to address residual safety degradation.

### Loss & Training
Safe Delta is a post-training method and does not require additional training processes. Safety degradation estimation only requires computing the FIM via forward-backward propagation on a small safety evaluation set (approx. hundreds of samples), which is highly efficient. The entire pipeline does not require access to the original fine-tuning data, relying only on the base model weights $\theta_{\text{base}}$ and the fine-tuned weights $\theta_{\text{ft}}$.

## Key Experimental Results

### Main Results

Experimental results on Llama-2-7B-Chat and Llama-3-8B-Instruct using 4 different datasets:

| Method | Safety on Harmful Data↑ | Utility on Harmful Data | Safety on Benign Data↑ | Utility on Benign Data↑ |
|------|---------------|-------------|---------------|-------------|
| No Defense (Direct FT) | ~10% | High | ~60% | High |
| Vaccine | ~75% | Medium | ~80% | Med-Low |
| SafeLoRA | ~80% | Low | ~85% | Low |
| RESTA | ~70% | Med-High | ~75% | Med-High |
| Lisa | ~82% | Medium | ~78% | Medium |
| **Safe Delta** | **~92%** | **High** | **~90%** | **High** |

### Ablation Study

| Configuration | Safety Rate | Utility Retention | Description |
|------|-------|---------|------|
| Full Safe Delta | 92% | 98% | Complete method |
| Parameter Selection Only (w/o Compensation Vector) | 83% | 97% | Performance drops significantly; compensation vector is indispensable |
| Safety Compensation Only (w/o Parameter Selection) | 78% | 95% | Over-compensation degrades utility |
| Uniform Parameter Selection (w/o FIM Estimation) | 72% | 90% | FIM-guided selection significantly outperforms random selection |
| Scaling all delta parameters (RESTA-like) | 70% | 85% | Global scaling degrades utility |

### Key Findings
- Safe Delta consistently maintains high safety rates across all 4 datasets (harmful/benign, large/small scales), whereas other methods fail under certain scenarios.
- FIM-guided parameter selection is the core component—safety-related parameters and utility-related parameters are largely separable.
- The safety compensation vector effectively repairs residual safety degradation post-selection with minimal utility impact.
- The method exhibits stable performance across different model scales (7B to 70B).

## Highlights & Insights
- **Advantages of the Post-Training Paradigm**: Safe Delta does not require modifying the fine-tuning process or accessing users' fine-tuning data. It only requires the discrepancy between base and fine-tuned model weights, offering the most practical deployment solution for API providers.
- **Parameter-Level Safety-Utility Decoupling**: Through FIM, it is discovered that safety-sensitive parameters and utility-related parameters are largely orthogonal. This supports the hypothesis that "safety alignment information is stored within a specific subset of parameters".
- **Three-Stage Progressive Design**: By conducting coarse-grained selection (binary mask) followed by fine-grained correction (compensation vector), the method presents an elegant coarse-to-fine optimization strategy.

## Limitations & Future Work
- Diagonal approximation of the Fisher Information Matrix might ignore interaction effects between parameters.
- The selection of the safety evaluation set may introduce bias, as safety standards differ across various domains.
- Determining the scaling factor $\alpha$ for the compensation vector through binary search increases computational overhead.
- Cumulative safety degradation under multi-round fine-tuning scenarios is not discussed.

## Related Work & Insights
- **vs SafeLoRA**: SafeLoRA projects updates into a safe subspace during LoRA fine-tuning, but is limited to LoRA scenarios and can over-constrain utility. Safe Delta is compatible with any fine-tuning paradigm (Full-parameter/LoRA/QLoRA).
- **vs RESTA**: RESTA restores safety by linearly scaling the entire delta parameter vector, which is a special case of Safe Delta (uniform selection + global scaling) and lacks parameter-level fine-grained control.
- **vs Vaccine**: Vaccine aligns gradient directions before fine-tuning to prevent safety degradation in a vaccine-like manner. However, it requires modifying the fine-tuning process, making it unsuitable for API service scenarios.
- **Insight**: Parameter-level safety analysis can be extended to model merging scenarios—selectively preserving safety-related parameters when merging multiple adapters.

## Rating
- Novelty: ⭐⭐⭐⭐ The three-stage pipeline (FIM estimation + parameter selection + compensation vector) is novel, though individual components (FIM, task arithmetic) have been applied in other contexts.
- Experimental Thoroughness: ⭐⭐⭐⭐ Coverage of 4 datasets, multiple model scales, and comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition with well-motivated methodology.
- Value: ⭐⭐⭐⭐⭐ Solves a highly practical safety problem in LLM fine-tuning services with low deployment overhead.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Efficient Ensemble for Fine-tuning Language Models on Multiple Datasets](../../ACL2025/llm_nlp/efficient_ensemble_for_fine-tuning_language_models_on_multiple_datasets.md)
- [\[ACL 2025\] Beware of Your Po! Measuring and Mitigating AI Safety Risks in Role-Play Fine-Tuning of LLMs](../../ACL2025/llm_nlp/sarft_roleplay_safety.md)
- [\[ACL 2026\] VOYAGER: A Training Free Approach for Generating Diverse Datasets using LLMs](../../ACL2026/llm_nlp/voyager_a_training_free_approach_for_generating_diverse_datasets_using_llms.md)
- [\[ICLR 2026\] Differential Fine-Tuning Large Language Models Towards Better Diverse Reasoning Abilities](../../ICLR2026/llm_nlp/differential_fine-tuning_large_language_models_towards_better_diverse_reasoning_.md)
- [\[ACL 2025\] GORP: Continual Gradient Low-Rank Projection Fine-Tuning for LLMs](../../ACL2025/llm_nlp/gorp_continual_gradient_projection.md)

</div>

<!-- RELATED:END -->

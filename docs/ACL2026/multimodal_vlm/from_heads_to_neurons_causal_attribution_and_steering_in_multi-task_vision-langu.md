---
title: >-
  [Paper Note] From Heads to Neurons: Causal Attribution and Steering in Multi-Task Vision-Language Models
description: >-
  [ACL 2026][Multimodal VLM][Neuron Attribution] Introduces the HONES framework, which achieves unified, gradient-free neuron-level causal analysis and lightweight task performance enhancement across heterogeneous tasks in…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "Neuron Attribution"
  - "Causal Analysis"
  - "Multi-task VLM"
  - "Attention Heads"
  - "Model Interpretability"
date: 2026-05-08
content_hash: ff74e64bff558e81
---

# From Heads to Neurons: Causal Attribution and Steering in Multi-Task Vision-Language Models

**Conference**: ACL 2026  
**arXiv**: [2604.17941](https://arxiv.org/abs/2604.17941)  
**Code**: [github](https://github.com/petergit1/HONES)  
**Area**: Multimodal VLM  
**Keywords**: Neuron Attribution, Causal Analysis, Multi-task VLM, Attention Heads, Model Interpretability

## TL;DR

Introduces the HONES framework, which achieves unified, gradient-free neuron-level causal analysis and lightweight task performance enhancement across heterogeneous tasks in multi-task VLMs by first locating task-critical attention heads and then guiding FFN neuron attribution conditional on those heads.

## Background & Motivation

**Background**: Large Vision-Language Models (VLMs) excel in multi-task settings like VQA, OCR, and image captioning. However, their internal decision-making processes remain opaque—multiple capabilities are entangled in shared parameters, hindering error attribution and controllable deployment. Neuron-level analysis can provide fine-grained actionable insights.

**Limitations of Prior Work**: (1) Existing neuron analysis primarily focuses on single-task settings, failing to compare neuron importance across heterogeneous tasks (e.g., Q&A vs. image-text matching); (2) Most methods score neurons independently, ignoring the task-dependent routing effects of attention heads, which leads to inflated importance scores for polysemous neurons.

**Key Challenge**: How to accurately identify task-critical neurons for different tasks within a shared parameter space while avoiding noise from polysemy?

**Goal**: Design a unified cross-task neuron attribution framework and utilize identified critical neurons for lightweight task performance enhancement.

**Key Insight**: Follow the structural causal view of Transformers—attention heads are responsible for selecting and routing task-critical inputs, while FFN neurons write the routed information into the residual stream. Thus, routing nodes (heads) should be localized first, followed by conditional FFN neuron attribution.

**Core Idea**: The task importance of a neuron should be measured by its "write contribution" under the routing paths of task-critical attention heads, rather than simple activation magnitude.

## Method

### Overall Architecture

HONES consists of two stages: **Discovery Phase**—(1) Locate task-critical attention heads $\mathcal{H}_t^*$ via mean-substitution intervention; (2) Under the condition of these heads, measure the causal write contribution of each FFN neuron to the task objective using Direct Vocabulary Projection (DVP) to select Top-K neurons. **Steering Phase**—Freeze the backbone and learn sparse scaling factors only for critical neurons, achieving controllable task enhancement via KL regularization.

### Key Designs

1. **Causal Head Localization**:

    - **Function**: Identify task-critical "routing nodes" to constrain the downstream neuron search space.
    - **Mechanism**: Employs mean-substitution intervention—replacing the output of a target head with the mean output of the other $H-1$ heads, measuring the performance drop $S_t(h)$. Top-$K_h$ heads form $\mathcal{H}_t^*$.
    - **Design Motivation**: Compared to zero-ablation, mean-substitution reduces out-of-distribution artifacts; localizing routing nodes first effectively isolates valid computational paths.

2. **Head-guided Neuron Attribution (Causal Write Effect)**:

    - **Function**: Score each FFN neuron under the context of task routing.
    - **Mechanism**: For each neuron $(l,i)$, calculate the vector $\Delta \mathbf{r}_i^{(l)}$ written to the residual stream via down-projection, then use Direct Vocabulary Projection (DVP) to project it onto the unembedding vector direction of the target token to obtain the write contribution $c_{l,i}$. Interventions are applied to each critical head to calculate the contribution gap $\Delta c$ before and after intervention, which is aggregated into the final score $I_{l,i}$ weighted by head importance.
    - **Design Motivation**: Independent activation-based scoring is easily confused by polysemy; head-guided conditioning ensures only effective contributions along the task routing path are counted.

3. **Lightweight Neuron Steering**:

    - **Function**: Enhance task performance by adjusting critical neuron activations.
    - **Mechanism**: Freeze all backbone parameters and learn a scaling factor $\lambda_{l,i}$ for each critical neuron. The optimization objective includes task loss and a KL divergence regularization term: $\min_{\lambda_t} \mathcal{L}_t + \beta \text{KL}(p_\theta \| p_{\theta_{\lambda_t}})$.
    - **Design Motivation**: KL regularization prevents excessive deviation from original model behavior; learning only sparse scaling factors keeps parameter overhead minimal.

### Loss & Training

The discovery phase uses a 7K-image discovery split. The steering phase uses a 2K-image dev split to learn scaling factors, and 3K images for testing. For open-ended targets (e.g., captioning), IDF weighting is used to aggregate unembedding vectors of tokens.

## Key Experimental Results

### Main Results (% drop after masking Top-1% neurons)

| Method | VQA | OCR | Caption | Retrieval | Average |
|------|-----|-----|---------|-----------|------|
| AP | 11.33 | 10.40 | 8.65 | 0.50 | 7.72 |
| MA | 6.82 | 15.50 | 11.90 | 1.35 | 8.89 |
| APE | 3.20 | -1.87 | 12.20 | 0.90 | 3.61 |
| **HONES** | **27.30** | **19.00** | **19.80** | **7.43** | **18.38** |

### Steering Gains (LLaVA-1.5-7B)

| Method | VQA | OCR | Caption | Retrieval | Average |
|------|-------|-------|---------|-----------|-------|
| Base | 0.652 | 0.576 | 0.129 | 0.933 | 0.572 |
| Grid-Search | 0.666 | 0.594 | 0.132 | 0.956 | 0.587 |
| **HONES** | **0.673** | **0.602** | **0.141** | **0.963** | **0.595** |

### Key Findings
- HONES consistently outperforms activation statistics methods across all tasks and two VLMs, with average performance drops reaching 18.38% (LLaVA) and 21.91% (Qwen).
- Critical neurons exhibit task-dependent layer preferences: retrieval tasks concentrate in middle layers (vision-text alignment), while other tasks lean toward deep layers (answer decoding).
- Shared neurons for VQA show the highest cross-task significance, exhibiting a "Hub" effect—VQA-related neurons support a wide range of vision-language tasks.
- In OOD experiments, direct zero-shot transfer of scaling factors learned in-domain achieves consistent improvements.

## Highlights & Insights
- The "coarse-to-fine" attribution logic from attention heads to neurons is elegant and efficient—head-guided conditioning effectively suppresses polysemous noise.
- A unified cross-task scoring interface (DVP + IDF weighting) is proposed, solving the problem of incomparable outputs across heterogeneous tasks.
- The discovery of VQA as a cross-task "Hub" has significant implications for model understanding.
- The steering method only learns sparse scaling factors, resulting in extremely low parameter overhead and OOD transferability.

## Limitations & Future Work
- Experiments are limited to dense models at the 7B scale; validation on larger models or MoE architectures remains to be done.
- Four coarse-grained task categories may mask differences at the sub-task level (e.g., counting vs. spatial reasoning in VQA).
- Causal analysis requires multiple forward passes, leading to higher computational costs and limited scalability on large datasets.
- Complementary integration with feature-level methods like SAE has not been explored.

## Related Work & Insights
- **vs. AP/MA/APE (Activation Statistics)**: Magnitude alone cannot distinguish polysemy; HONES' head-guided conditioning is more accurate.
- **vs. QRNCA (Gradient Methods)**: HONES is gradient-free, more efficient, and faster in localization.
- **vs. SAE**: HONES operates directly on the original model without additional feature learning, supporting causal attribution and lightweight steering.
- **vs. MultEdit**: While MultEdit edits knowledge in MLP blocks, HONES analyzes the shared neuron structure across tasks.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The framework design of head-guided neuron attribution is novel, and the unified cross-task scoring interface solves practical bottlenecks.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four tasks across two models, extensive control variants, ablation studies, and OOD validation.
- Writing Quality: ⭐⭐⭐⭐ Clear framework description and rich insights.
- Value: ⭐⭐⭐⭐⭐ Significantly advances VLM interpretability and controllability; the steering method has high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Understanding Task Transfer in Vision-Language Models](../../CVPR2026/multimodal_vlm/understanding_task_transfer_in_vision-language_models.md)
- [\[ACL 2026\] OMIBench: Benchmarking Olympiad-Level Multi-Image Reasoning in Large Vision-Language Models](omibench_benchmarking_olympiad-level_multi-image_reasoning_in_large_vision-langu.md)
- [\[ACL 2026\] LaMI: Augmenting Large Language Models via Late Multi-Image Fusion](lami_augmenting_large_language_models_via_late_multi-image_fusion.md)
- [\[ICML 2026\] CyberJurors: A Multi-Agent Simulation Task for E-Commerce Disputes Verdict](../../ICML2026/multimodal_vlm/cyberjurors_a_multi-agent_simulation_task_for_e-commerce_disputes_verdict.md)
- [\[CVPR 2026\] Recurrent Reasoning with Vision-Language Models for Estimating Long-Horizon Embodied Task Progress](../../CVPR2026/multimodal_vlm/recurrent_reasoning_with_vision-language_models_for_estimating_long-horizon_embo.md)

</div>

<!-- RELATED:END -->

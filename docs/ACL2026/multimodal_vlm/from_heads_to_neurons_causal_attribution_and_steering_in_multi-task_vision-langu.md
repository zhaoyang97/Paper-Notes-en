---
title: >-
  [Paper Note] From Heads to Neurons: Causal Attribution and Steering in Multi-Task Vision-Language Models
description: >-
  [ACL 2026][Multimodal VLM][neuron attribution] This paper proposes HONES, a framework that first localizes task-critical attention heads and then uses them as conditions to guide FFN neuron attribution, achieving unified…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "neuron attribution"
  - "causal analysis"
  - "multi-task VLM"
  - "attention heads"
  - "model interpretability"
date: 2026-05-08
content_hash: 3418e776be7bae4d
---

# From Heads to Neurons: Causal Attribution and Steering in Multi-Task Vision-Language Models

**Conference**: ACL 2026
**arXiv**: [2604.17941](https://arxiv.org/abs/2604.17941)  
**Code**: [github](https://github.com/petergit1/HONES)  
**Area**: Multimodal VLM
**Keywords**: neuron attribution, causal analysis, multi-task VLM, attention heads, model interpretability

## TL;DR

This paper proposes HONES, a framework that first localizes task-critical attention heads and then uses them as conditions to guide FFN neuron attribution, achieving unified, gradient-free, neuron-level causal analysis and lightweight task performance improvement across heterogeneous tasks in multi-task VLMs.

## Background & Motivation

**Background**: Large vision-language models (VLMs) achieve strong performance across diverse tasks such as VQA, OCR, and image captioning, yet their internal decision-making processes remain opaque — multiple capabilities are entangled within shared parameters, hindering error attribution and controllable deployment. Neuron-level analysis can provide fine-grained, actionable insights.

**Limitations of Prior Work**: (1) Existing neuron analysis methods focus primarily on single-task settings and cannot compare neuron importance across heterogeneous tasks (e.g., question answering vs. image-text matching). (2) Most approaches score neurons independently, ignoring the task-dependent routing effect of attention heads, which inflates the importance scores of polysemantic neurons.

**Key Challenge**: How to accurately identify task-critical neurons within a shared parameter space across different tasks while avoiding noise introduced by polysemanticity?

**Goal**: To design a unified cross-task neuron attribution framework and leverage the identified critical neurons for lightweight task performance improvement.

**Key Insight**: Following a structured causal view of the Transformer — attention heads select and route task-relevant inputs, while FFN neurons write routed information into the residual stream. The framework first localizes routing nodes (attention heads), then attributes FFN neurons conditioned on them.

**Core Idea**: The task importance of a neuron should be measured by its "write contribution" along the routing path of task-critical attention heads, rather than by simple activation magnitude.

## Method

### Overall Architecture

HONES operates in two stages. **Discovery stage**: (1) Task-critical attention heads $\mathcal{H}_t^*$ are localized via mean-ablation intervention; (2) conditioned on the identified attention heads, the causal write contribution of each FFN neuron to the task objective is measured via Direct Vocabulary Projection (DVP), and Top-K neurons are selected. **Steering stage**: The backbone is frozen, and sparse scaling factors are learned exclusively on the critical neurons, with KL regularization enabling controllable task improvement.

### Key Designs

1. **Causal Attention Head Localization**:

    - Function: Identifies task-critical "routing nodes" to constrain the downstream neuron search space.
    - Mechanism: Mean ablation intervention is applied — the output of a target head is replaced by the mean of the remaining $H-1$ heads' outputs, and the resulting task performance drop $S_t(h)$ is measured. The Top-$K_h$ heads are selected to form $\mathcal{H}_t^*$.
    - Design Motivation: Mean ablation reduces out-of-distribution artifacts compared to zero ablation; localizing routing nodes first effectively isolates valid computation paths.

2. **Head-Guided Neuron Attribution (Causal Write Effect)**:

    - Function: Scores each FFN neuron conditioned on the task routing context.
    - Mechanism: For each neuron $(l,i)$, the vector $\Delta \mathbf{r}_i^{(l)}$ written into the residual stream via the down-projection is computed, then projected onto the unembedding vector of the target token via DVP to obtain a write contribution $c_{l,i}$. Interventions are then applied to each critical head, and the contribution drop $\Delta c$ before and after intervention is aggregated, weighted by head importance, into a final score $I_{l,i}$.
    - Design Motivation: Activation-based independent scoring is easily confounded by polysemanticity; head-guided conditioning ensures that only contributions along the task routing path are counted.

3. **Lightweight Neuron Steering**:

    - Function: Enhances task performance by modulating the activations of critical neurons.
    - Mechanism: All backbone parameters are frozen, and a scaling factor $\lambda_{l,i}$ is learned for each critical neuron. The optimization objective includes a task loss and a KL divergence regularization term: $\min_{\lambda_t} \mathcal{L}_t + \beta \text{KL}(p_\theta \| p_{\theta_{\lambda_t}})$.
    - Design Motivation: KL regularization prevents excessive deviation from the original model behavior; learning only sparse scaling factors incurs minimal parameter overhead.

### Loss & Training

The discovery stage uses a discovery split of 7K images; the steering stage uses a dev split of 2K images to learn scaling factors, with 3K images reserved for testing. For open-ended objectives (e.g., captioning), IDF-weighted aggregation of token unembedding vectors is applied.

## Key Experimental Results

### Main Results (Performance Drop % after Masking Top-1% Neurons)

| Method | VQA | OCR | Caption | Retrieval | Avg. |
|--------|-----|-----|---------|-----------|------|
| AP | 11.33 | 10.40 | 8.65 | 0.50 | 7.72 |
| MA | 6.82 | 15.50 | 11.90 | 1.35 | 8.89 |
| APE | 3.20 | -1.87 | 12.20 | 0.90 | 3.61 |
| **HONES** | **27.30** | **19.00** | **19.80** | **7.43** | **18.38** |

### Steering Results (LLaVA-1.5-7B)

| Method | VQA | OCR | Caption | Retrieval | Avg. |
|--------|-------|-------|---------|-----------|-------|
| Base | 0.652 | 0.576 | 0.129 | 0.933 | 0.572 |
| Grid-Search | 0.666 | 0.594 | 0.132 | 0.956 | 0.587 |
| **HONES** | **0.673** | **0.602** | **0.141** | **0.963** | **0.595** |

### Key Findings
- HONES consistently outperforms activation-based methods across all tasks and both VLMs, achieving average performance drops of 18.38% (LLaVA) and 21.91% (Qwen).
- Critical neurons exhibit task-dependent layer preferences: retrieval tasks concentrate in middle layers (visual-text alignment), while other tasks favor deeper layers (answer decoding).
- VQA shared neurons show the highest cross-task salience, exhibiting a "hub" effect — VQA-related neurons underpin a broad range of visual-language tasks.
- In OOD experiments, directly transferring in-domain learned scaling factors (zero-shot) yields consistent improvements.

## Highlights & Insights
- The coarse-to-fine attribution strategy from attention heads to neurons is both elegant and efficient — head-guided conditioning effectively suppresses polysemantic noise.
- A unified cross-task scoring interface (DVP + IDF weighting) addresses the comparability challenge posed by heterogeneous task outputs.
- The discovery of VQA as a cross-task "hub" carries significant implications for model understanding.
- The steering method learns only sparse scaling factors, incurring minimal parameter overhead while remaining transferable to OOD settings.

## Limitations & Future Work
- Experiments are limited to dense models at the 7B scale; validation on larger models or MoE architectures remains to be conducted.
- The four coarse-grained task categories may obscure sub-task-level differences (e.g., counting vs. spatial reasoning within VQA).
- Causal analysis requires multiple forward passes, resulting in high computational cost that limits scalability on large datasets.
- Complementary integration with feature-level methods such as SAEs has not been explored.

## Related Work & Insights
- **vs. AP/MA/APE (activation statistics methods)**: Examining activation magnitude alone cannot disambiguate polysemantic neurons; HONES's head-guided conditioning provides more accurate attribution.
- **vs. QRNCA (gradient-based methods)**: HONES is gradient-free and more efficient, with faster localization.
- **vs. SAE**: HONES operates directly on the original model without additional feature learning, supporting both causal attribution and lightweight steering.
- **vs. MultEdit**: MultEdit edits knowledge in MLP blocks, whereas HONES analyzes the neuron-sharing structure across tasks.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The head-guided neuron attribution framework is novel in design, and the unified cross-task scoring interface addresses a practical bottleneck.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four tasks × two models, extensive controlled variants and ablation studies, with OOD validation.
- Writing Quality: ⭐⭐⭐⭐ The framework is clearly described, with rich findings and insights.
- Value: ⭐⭐⭐⭐⭐ Significant advancement in VLM interpretability and controllability; the steering method has strong practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Understanding Task Transfer in Vision-Language Models](../../CVPR2026/multimodal_vlm/understanding_task_transfer_in_vision-language_models.md)
- [\[ACL 2026\] OMIBench: Benchmarking Olympiad-Level Multi-Image Reasoning in Large Vision-Language Models](omibench_benchmarking_olympiad-level_multi-image_reasoning_in_large_vision-langu.md)
- [\[NeurIPS 2025\] Causal-LLaVA: Causal Disentanglement for Mitigating Hallucination in Multimodal Large Language Models](../../NeurIPS2025/multimodal_vlm/causalllava_causal_disentanglement_for_mitigating_hallucinat.md)
- [\[ICLR 2026\] Dynamic Multimodal Activation Steering for Hallucination Mitigation in Large Vision-Language Models](../../ICLR2026/multimodal_vlm/dynamic_multimodal_activation_steering_for_hallucination_mitigation_in_large_vis.md)
- [\[ICML 2026\] Revis: Sparse Latent Steering to Mitigate Object Hallucination in Large Vision-Language Models](../../ICML2026/multimodal_vlm/revis_sparse_latent_steering_to_mitigate_object_hallucination_in_large_vision-la.md)

</div>

<!-- RELATED:END -->

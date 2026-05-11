---
title: >-
  [Paper Note] CoVFT: Context-aware Visual Fine-tuning for Multimodal Large Language Models
description: >-
  [CVPR 2026][Multimodal VLM][Multimodal large language models] This paper identifies the "visual preference conflict" problem in visual encoder fine-tuning within MLLMs…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "Multimodal large language models"
  - "visual fine-tuning"
  - "mixture of experts"
  - "context-aware"
  - "visual preference conflict"
date: 2026-05-08
content_hash: b194e7bf1f706ddd
---

# CoVFT: Context-aware Visual Fine-tuning for Multimodal Large Language Models

**Conference**: CVPR 2026
**arXiv**: [2603.21077](https://arxiv.org/abs/2603.21077)
**Code**: [https://github.com/weeknan/CoVFT](https://github.com/weeknan/CoVFT)
**Area**: Multimodal VLM
**Keywords**: Multimodal large language models, visual fine-tuning, mixture of experts, context-aware, visual preference conflict

## TL;DR

This paper identifies the "visual preference conflict" problem in visual encoder fine-tuning within MLLMs, and proposes the CoVFT framework. By introducing Context Vector Extraction (CVE) and Context-aware Mixture of Experts (CoMoE), CoVFT achieves context-aware visual fine-tuning, attaining state-of-the-art performance across 12 multimodal benchmarks with significantly improved stability over existing methods.

## Background & Motivation

Multimodal large language models (MLLMs) typically consist of three components: a visual encoder, a projection layer, and an LLM. A long-standing open question in the instruction fine-tuning stage is: **should the visual encoder be frozen or fine-tuned?**

Contradictions in existing practice:
- InstructBLIP and LLaVA-1.5 opt to freeze the visual encoder
- InternVL and Qwen-VL adopt joint fine-tuning
- No consensus exists in the community

Through controlled experiments, the authors identify a key phenomenon — **Visual Preference Conflicts**:

1. Existing VFT methods (full fine-tuning, LoRA, BitFit, etc.) **fail to consistently outperform the frozen baseline** — while average scores may improve, performance fluctuates substantially across individual tasks.
2. Root cause: the visual encoder is **context-agnostic** — it processes only the image without access to textual instructions. However, the same image requires attention to entirely different visual features depending on the task (e.g., grounding vs. captioning).
3. Gradients from different tasks conflict in direction, leading to unstable parameter updates.

Evidence: grounding and captioning tasks are constructed on the same dataset by varying only the text query. After training, the L2 distance between the two resulting visual encoders **grows continuously**, with larger discrepancies in deeper layers — confirming that different cognitive demands genuinely "pull" the parameters in opposing directions.

The core problem can be formalized as: conventional VFT models $p_{\theta_v}(\mathbf{z} | \mathbf{I})$, whereas MLLMs require $p_{\theta_v}(\mathbf{z} | \mathbf{I}, \mathbf{c})$ — visual features should be conditioned on multimodal context.

## Method

### Overall Architecture

CoVFT introduces a latent context variable $\mathbf{c}$, extending the visual posterior from $p(\mathbf{z}|\mathbf{I})$ to $p(\mathbf{z}|\mathbf{I}, \mathbf{c})$. This is realized through two modules: CVE extracts context vectors, and CoMoE modulates visual encoding based on the context.

### Key Designs

1. **Context Vector Extraction (CVE)**: Distills context signals from multimodal information via text-guided cross-modal attention, updated jointly with the visual encoder.

    - A frozen BERT encoder encodes the text instruction to obtain text embeddings $\mathbf{t}$.
    - At selected layers of the visual encoder, visual tokens $\mathbf{z}$ and text embeddings $\mathbf{t}$ are each passed through lightweight residual blocks ($f_{res}$, consisting of up-down projections with GELU activation).
    - Cross-attention uses text as query and the concatenated multimodal features as key/value: $\mathbf{c}_i = \text{CrossAttn}(\hat{\mathbf{t}}_q, [\hat{\mathbf{z}}, \hat{\mathbf{t}}]_{k,v})$
    - **Design Motivation**: (1) Context vectors are updated layer-by-layer in sync with the visual encoder, requiring no separate inference stage; (2) Text-led aggregation ensures that context vectors reflect task preferences rather than purely visual features.

2. **Context-aware Mixture of Experts (CoMoE)**: Injects context signals into the visual encoder via context-conditioned expert routing, decomposing conflicting optimization signals.

    - In the latter half of the ViT layers, the FFN is replaced by $N=4$ parallel expert networks (initialized from the original FFN).
    - Routing weights are computed from the context vector: $\mathbf{g}(\mathbf{c}) = \text{softmax}(\mathbf{W}\mathbf{c} + \mathbf{b})$
    - **Dense routing**: $\tilde{\mathbf{z}} = \sum_{n=1}^N g^n(\mathbf{c}) \mathcal{E}^n(\mathbf{z})$
    - **Mechanism**: The gradient for the $n$-th expert is scaled by its routing weight — $\nabla_{\theta_e^n} \mathcal{L} = g^n(\mathbf{c}) \cdot \frac{\partial\mathcal{L}}{\partial\tilde{\mathbf{z}}} \frac{\partial\mathcal{E}^n(\mathbf{z})}{\partial\theta_e^n}$
    - Samples with similar contexts receive similar expert weights, yielding consistent gradient updates; samples with different contexts are routed differently, avoiding gradient conflicts.
    - **Design Motivation**: Dense routing outperforms sparse routing because, with limited data, sparse routing leads to under-training of certain experts.

### Loss & Training

- Standard next-token prediction loss: $\mathcal{L}_{inst} = -\sum_{t=1}^T \log p_\theta(a_t | a_{<t}, \mathbf{Q}, \mathbf{I})$
- Parameters optimized: CVE module + CoMoE module + LayerNorm statistics; **all other visual encoder parameters are frozen**.
- Pre-training: 558K image-text pairs, projection layer only, lr=1e-3, batch=256.
- Instruction fine-tuning: 665K image-text instructions, joint training of LLM + projection layer + CoVFT modules, lr=2e-5, batch=128.

## Key Experimental Results

### Main Results

LLaVA-1.5-7B on 12 multimodal benchmarks:

| Method | General ↑ | Know.&OCR ↑ | Vision ↑ | Avg ↑ | Tasks Beating Freeze |
|--------|-----------|------------|---------|------|---------------------|
| Freeze | 66.23 | 61.20 | 51.71 | 58.93 | — |
| Full fine-tuning | 66.69 | 61.29 | 52.17 | 59.29 | 6/12 |
| LoRA | 65.93 | 60.86 | 52.45 | 59.04 | 6/12 |
| BitFit | 66.14 | 61.58 | 53.10 | 59.57 | 9/12 |
| **CoVFT** | **67.04** | **61.93** | **55.81** | **61.08** | **12/12** |

Key figures:
- CoVFT 7B (61.08%) surpasses Freeze 13B (61.43%) on average — while optimizing fewer than 5% of parameters.
- The most notable gain is on MMVP: from 28.00 (Freeze) to 36.67 (+8.67).
- CoVFT also generalizes to 13B models, achieving 62.90%, outperforming Full ft. (61.30%) and BitFit (61.43%).

### Ablation Study

| Configuration | General | Know.&OCR | Vision | Avg | Note |
|---------------|---------|----------|--------|-----|------|
| No context (Full ft.) | 66.69 | 61.29 | 52.17 | 59.29 | Baseline |
| Image-only context | 66.60 | 61.69 | 53.17 | 59.77 | Missing text signal |
| Text-only context | 66.84 | 61.86 | 54.73 | 60.55 | Text more critical than image |
| Concat[I,T] | 66.78 | 61.79 | 54.56 | 60.44 | Simple concatenation insufficient |
| **CVE** | **67.04** | **61.93** | **55.81** | **61.08** | Cross-modal attention optimal |
| Random@2 routing | 66.01 | 61.24 | 52.05 | 59.00 | Extra params without context ineffective |
| Uniform routing | 66.18 | 61.75 | 53.05 | 59.60 | Context conditioning necessary |
| Sparse@2 routing | 66.63 | 61.78 | 53.60 | 60.10 | Viable but inferior to Dense |
| **Dense routing** | **67.04** | **61.93** | **55.81** | **61.08** | Full expert activation optimal |

### Key Findings

1. **Text signal is critical**: Text-only context (60.55%) substantially outperforms Image-only (59.77%), indicating that visual preference conflicts are primarily driven by language context.
2. **Dense routing outperforms sparse**: Dense > Sparse@2 > Uniform > Random@2, demonstrating that context-conditioned routing — rather than simply adding parameters — is the effective factor.
3. **Strong data efficiency**: CoVFT with 75% of the data surpasses the full-data Freeze baseline.
4. **Cross-architecture generalization**: Effective with SigLiP and DINOv3 replacing CLIP, as well as on the InternVL 2.0 architecture.
5. **Well-structured context vector space**: PCA visualization reveals clear clustering by task type; routing weight similarity correlates with context similarity at $r=0.76$.

## Highlights & Insights

1. **Precise problem formulation**: This work is the first to attribute VFT instability in MLLMs to "visual preference conflicts," supported by controlled experimental evidence.
2. **Elegant solution**: The CVE + CoMoE combination fundamentally transforms context-agnostic visual encoding into context-conditioned encoding, with clearly motivated design choices.
3. **High practical value**: The finding that 7B + CoVFT ≈ 13B + Freeze suggests that better visual fine-tuning can reduce reliance on larger model capacity.
4. **Comprehensive evaluation**: 12 benchmarks, 7B/13B scales, 3 visual encoders, InternVL architecture, and data efficiency analysis.

## Limitations & Future Work

1. CVE relies on an additional frozen BERT encoder, introducing extra inference-time computation — whether the LLM's own text encoding capacity could be leveraged instead remains an open question.
2. CoMoE replaces FFNs only in the latter half of the ViT; the rationale for selecting the optimal depth boundary is not thoroughly analyzed.
3. The number of experts is fixed at 4; the effect of expert count on tasks of varying complexity is unexplored.
4. The visual encoder remains frozen during pre-training — whether introducing context-awareness at the pre-training stage could yield further gains is not investigated.
5. Experiments are primarily conducted on LLaVA-style architectures; applicability to Q-Former-based architectures (e.g., InstructBLIP) is not validated.

## Related Work & Insights

- **LLaVA / LLaVA-1.5**: Established a simple and effective MLLM paradigm and VFT benchmarks.
- **Cambrian-1**: Also observed that VFT is broadly beneficial yet unstable, without analyzing the root cause.
- **MoE in NLP**: Sparse MoE is widely used for LLM scaling (e.g., Mixtral); this work introduces MoE into the visual encoder and finds that dense routing is superior.
- **SVPT**: A leading PEFT method for image classification that underperforms the Freeze baseline in MLLMs — corroborating the visual preference conflict hypothesis.
- Insight: The visual encoder constitutes less than 5% of MLLM parameters yet has a disproportionately large impact — making it the highest-leverage optimization target.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Both the identification of visual preference conflicts and the proposal of context-aware VFT represent significant contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 12 benchmarks, complete ablations, multi-architecture validation, and data efficiency analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ — Problem-driven narrative flows coherently from observation to analysis to method.
- Value: ⭐⭐⭐⭐⭐ — Directly actionable for the MLLM community; code is publicly available.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] FairLLaVA: Fairness-Aware Parameter-Efficient Fine-Tuning for Large Vision-Language Models](fairllava_fairness-aware_parameter-efficient_fine-tuning_for_large_vision-langua.md)
- [\[CVPR 2026\] OddGridBench: Exposing the Lack of Fine-Grained Visual Discrepancy Sensitivity in Multimodal Large Language Models](oddgridbench_exposing_the_lack_of_fine-grained_visual_discrepancy_sensitivity_in.md)
- [\[CVPR 2026\] Predictive Regularization Against Visual Representation Degradation in Multimodal Large Language Models](predictive_regularization_against_visual_representation_degradation_in_multimoda.md)
- [\[CVPR 2026\] Taxonomy-Aware Representation Alignment for Hierarchical Visual Recognition with Large Multimodal Models](taxonomy-aware_representation_alignment_for_hierarchical_visual_recognition_with.md)
- [\[CVPR 2026\] Parallel In-context Learning for Large Vision Language Models](parallel_in-context_learning_for_large_vision_language_models.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Cross-Modal Unlearning via Influential Neuron Path Editing in Multimodal Large Language Models
description: >-
  [AAAI 2026][Multimodal VLM][Machine Unlearning] This paper proposes MIP-Editor, which localizes **influential neuron paths** encoding forget-target knowledge in MLLMs via cross-layer gradient integration (text branch) and Fisher integration (visual branch), then edits these neurons using path-based Representation Misdirection Unlearning (RMisU), achieving up to 87.75% forget rate and 54.26% improvement in general knowledge retention on MLLMU-Bench.
tags:
  - AAAI 2026
  - Multimodal VLM
  - Machine Unlearning
  - Multimodal Large Language Models
  - Neuron Path Editing
  - Representation Misdirection
  - Cross-Modal Consistency
date: 2026-05-08
content_hash: d3ca07290c7f0919
---

# Cross-Modal Unlearning via Influential Neuron Path Editing in Multimodal Large Language Models

**Conference**: AAAI 2026
**arXiv**: [2511.06793](https://arxiv.org/abs/2511.06793)
**Code**: [https://github.com/PreckLi/MIP-Editor](https://github.com/PreckLi/MIP-Editor)
**Area**: AI Safety / Multimodal VLM
**Keywords**: Machine Unlearning, Multimodal Large Language Models, Neuron Path Editing, Representation Misdirection, Cross-Modal Consistency

## TL;DR
This paper proposes MIP-Editor, which localizes **influential neuron paths** encoding forget-target knowledge in MLLMs via cross-layer gradient integration (text branch) and Fisher integration (visual branch), then edits these neurons using path-based Representation Misdirection Unlearning (RMisU), achieving up to 87.75% forget rate and 54.26% improvement in general knowledge retention on MLLMU-Bench.

## Background & Motivation
Multimodal Large Language Models (MLLMs), due to their vast knowledge capacity, pose security risks including privacy leakage, toxic content, and copyright infringement. Machine Unlearning (MU) aims to selectively remove specific knowledge from a model while preserving overall performance.

Existing MU methods follow two main lines:
1. **Fine-tuning methods** (GA_Diff, KL_Min, NPO, etc.): Directly extended from LLM unlearning methods, ignoring the multimodal structural properties of MLLMs, leading to insufficient unlearning in the text modality.
2. **Neuron editing methods** (DEPN, MANU, etc.): Discover and prune important neurons via point-wise activation scoring, but suffer from two fundamental flaws:
    - Point-wise scoring fails to capture structured cross-layer information flow, resulting in inconsistent unlearning across modalities.
    - Directly pruning sensitive neurons disrupts paths that also serve general reasoning, causing severe model performance degradation.

## Core Problem
1. **Cross-modal unlearning inconsistency**: Existing methods achieve reasonable unlearning on the visual modality but exhibit low forget rates on the text modality, because point-wise attribution cannot capture how cross-layer information flows propagate through text.
2. **Failure of the forgetting–retention trade-off**: When neurons in the forget set and retain set overlap, direct pruning destroys general reasoning ability—the Ours-Path variant of MANU scores only 2.11% on the retain-set VQA, corroborating this issue.

## Method

### Overall Architecture
MIP-Editor operates in two stages:
1. **Localization stage**: Computes attribution scores within FFN layers using **cross-layer gradient integration** (text branch) and **cross-layer Fisher integration** (visual branch) to identify modality-specific influential neuron paths.
2. **Editing stage**: Prunes neurons along the identified paths to sever the forget information flow, then fine-tunes only these pruned neurons via **RMisU** (Representation Misdirection Unlearning), redirecting forget-set representations toward random directions while recovering retain-set performance.

### Key Designs

1. **Cross-layer Gradient Integration (IGI) for the text branch**:

    - Unlike MANU's point-wise activation scoring, IGI linearly interpolates neuron activations from 0 to their original values ($m$-step Riemann approximation) and computes a joint gradient integral across all layers on the path: $\text{IGI}(\mathbf{w}) = \sum_{j=1}^{N} \tilde{w}_{i_j}^n \sum_{k=1}^{m} \sum_{l=1}^{N} \frac{\partial F_T}{\partial w_{i_l}^l}$
    - This captures cross-layer cascade effects rather than single-point importance, providing a more accurate model of textual information flow.

2. **Cross-layer Fisher Integration (IFI) for the visual branch**:

    - The visual encoder has high dimensionality, strong spatial correlations, and substantial parameter redundancy; the Fisher information matrix diagonal approximation (squared gradients) is more suitable than first-order gradients: $\text{IFI}(\mathbf{z}) = \sum_{n=1}^{N} \tilde{z}_{i_n}^n \sum_{k=1}^{m} \sum_{l=1}^{N} \left(\frac{\partial \mathbf{G}}{\partial z_{i_l}^l}\right)^2$
    - **Design motivation**: Text and visual signals have different characteristics; different-order signals are used to estimate neuron importance for each modality.

3. **Path-based RMisU editing**:

    - Step 1: Prune path neurons (zero out activations) to sever the forget information flow.
    - Step 2: Freeze all other parameters and fine-tune only the path neurons.
    - Forget objective: Redirect forget-set representations $\mathbf{h}^{(l)}(x_f)$ toward a random direction $\mathbf{v}_f = \lambda \cdot \|\mathbf{h}^{(l)}(x_f)\|_2 \cdot \mathbf{u}$.
    - Key insight: Compared to full-model RMisU, **editing only path neurons** substantially reduces interference with general knowledge.

4. **Greedy layer-wise search for path localization**:

    - Selects the highest-scoring neuron at each layer to form the ordered paths $\mathcal{P}_t$ (text) and $\mathcal{P}_v$ (visual).
    - Complexity is $O(C_{\text{grad}} \cdot m \cdot L_t \cdot \sum |w_l^t|)$, significantly lower than global search.

### Loss & Training
The total loss consists of three components:
- **Retain cross-entropy loss** $\mathcal{L}_{\text{retain}}$: Standard next-token prediction loss on the retain set.
- **Forget RMisU loss** $\mathcal{L}_{\text{RMisU}}^f$: Pulls intermediate representations of the forget set toward random vectors.
- **Retain RMisU loss** $\mathcal{L}_{\text{RMisU}}^r$: Constrains retain-set representations from deviating from those of the frozen model.

Total objective: $\mathcal{L}_{\text{RMisU}} = \mathcal{L}_{\text{RMisU}}^f + \gamma \cdot \mathcal{L}_{\text{RMisU}}^r$

Training configuration: 4 epochs, batch size 4, Adam optimizer, LoRA, lr=2e-5, on NVIDIA A100 GPU.

## Key Experimental Results

### MLLMU-Bench (5% forget ratio, Qwen2.5-VL-3B)

| Method | FVQA↓ | RVQA↑ | FQA↓ | RQA↑ |
|--------|-------|-------|------|------|
| Vanilla | 39.20% | 37.72% | 49.60% | 47.20% |
| GA_Diff | 32.00% | 32.80% | 46.40% | 43.20% |
| KL_Min | 33.60% | 27.59% | 41.60% | 42.57% |
| NPO | 37.60% | 36.20% | 42.40% | 44.80% |
| MANU | 36.00% | 34.47% | 30.80% | 34.65% |
| **MIP-Editor** | **4.80%** | **58.19%** | **9.60%** | **36.80%** |

### CLEAR (5% forget ratio, Qwen2.5-VL-3B)

| Method | FVQA↓ | RVQA↑ | FGEN↓ | RGEN↑ |
|--------|-------|-------|-------|-------|
| Vanilla | 72.34% | 73.42% | 0.3776 | 0.3900 |
| NPO | 7.45% | 9.37% | 0.0805 | 0.0639 |
| **MIP-Editor** | **3.19%** | **24.05%** | **0.0926** | **0.3631** |

### LLaVA-1.5-7B (5% forget ratio, MLLMU-Bench)

| Method | FVQA↓ | RVQA↑ | FQA↓ | RQA↑ |
|--------|-------|-------|------|------|
| Vanilla | 56.80% | 51.56% | 50.40% | 52.59% |
| **MIP-Editor** | **38.40%** | **47.22%** | **36.80%** | **47.34%** |

Note: MIP-Editor's unlearning effectiveness is notably weaker on LLaVA-1.5-7B than on Qwen2.5-VL-3B (FVQA only reduces from 56.80% to 38.40%), indicating that the method's robustness across model scales and architectures warrants further attention.

### Ablation Study
- **Removing dual-branch localization (IGI-only or IFI-only)**: Unlearning performance drops substantially, with FVQA rising from 4.80% to 36.00%/32.00%, demonstrating the necessity of dual-modal paths.
- **Replacing path-based with point-wise attribution (Ours-Path)**: Forget rate is strong (2.40%), but retention performance collapses (RVQA only 2.11%), confirming the critical failure of direct pruning.
- **Removing RMisU editing (Ours-Edit)**: FVQA rises to 43.60%, indicating unlearning failure.
- **Full-model RMisU (without path localization)**: RVQA drops to only 14.65%, showing severe degradation of general capabilities.
- **Key conclusion**: Path localization and RMisU editing are mutually indispensable.

## Highlights & Insights
- **Paradigm shift from point-wise to path-wise**: Elevating neuron importance estimation from independent scoring to cross-layer path-level cascade attribution—a natural and effective idea; top-$k=2^5$ achieves performance comparable to point-wise at $k=2^{13}$.
- **Heterogeneous attribution strategies per modality**: Attribution scores are designed separately for text (first-order gradient integration) and vision (second-order Fisher integration), consistent with their respective signal characteristics.
- **Forgetting–retention decoupling via path-only editing**: The two-stage strategy of pruning followed by fine-tuning updates only a small number of parameters to recover general capabilities.
- **Information separability validation** (§4.7): An MLP classifier is used to verify whether the post-unlearning model can distinguish forget/retain data; MIP-Editor achieves over 85% classification accuracy, far exceeding baseline at approximately 50%.

## Limitations & Future Work
- **Weakened effectiveness on larger models**: Forget rate on LLaVA-1.5-7B is substantially lower than on Qwen2.5-VL-3B (FVQA 38.40% vs. 4.80%); scaling behavior is not sufficiently discussed.
- **Greedy path search is suboptimal**: Greedily selecting one neuron per layer to form a path may miss inter-layer optimal combinations; time complexity remains high at $O(C_{\text{grad}} \cdot m \cdot L \cdot \sum|w_l|)$.
- **Sensitivity to forget-set ratio**: Performance is best at 5%; retention performance degrades noticeably at 10%/15%, indicating insufficient robustness across different forget ratios.
- **Evaluation limited to MLLMU-Bench and CLEAR**: Both benchmarks are relatively new and small-scale (500+153 / 200 profiles); generalizability to real-world scenarios is unknown.
- **Instability on CLEAR**: The authors themselves acknowledge that CLEAR is highly sensitive to hyperparameters and prone to model collapse, undermining the persuasiveness of results on this dataset.
- **Limited coverage of MLLM architectures**: Only two models are tested; validation on architectures such as InternVL and Phi-3-Vision is absent.
- **Insufficient rigor of unlearning verification**: Stricter unlearning verification methods such as membership inference attacks are not employed.

## Related Work & Insights

| Method | Core Strategy | Strengths | Weaknesses |
|--------|--------------|-----------|------------|
| **MANU** (2025) | Point-wise activation scoring + pruning | Simple and efficient | Point-wise information insufficient; pruning disrupts general paths |
| **NPO** (2024) | Preference optimization unlearning | Better text unlearning | Ignores multimodal structure; near-collapse on CLEAR |
| **GA_Diff** (2022) | Gradient ascent + gradient descent | Intuitive | Insufficient unlearning; language fluency degradation |
| **MIP-Editor** (Ours) | Path-level attribution + RMisU | Cross-modal consistent unlearning; good retention | High computational cost; weaker on larger models |

The core advantage of MIP-Editor lies in path-aware editing being more "precise" than point-wise pruning, achieving coordinated cross-modal unlearning through a dual-branch design. However, computational cost and scaling behavior are clear shortcomings.

The path-level attribution idea can transfer to continual learning (localizing old knowledge paths to prevent forgetting) and model editing (localizing factual paths for precise modification). The choice between Fisher integration and gradient integration can be generalized as a universal framework for automatically selecting the most appropriate attribution signal order based on modality characteristics.

## Rating
- Novelty: ⭐⭐⭐⭐ The paradigm shift from point-wise to path-wise attribution represents a substantive contribution, though the RMisU component draws on prior work.
- Experimental Thoroughness: ⭐⭐⭐⭐ Ablations are detailed and visualizations are clear, but model coverage is narrow and results on CLEAR lack full persuasiveness.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear, method description is complete, and mathematical derivations are rigorous, though some notational inconsistencies exist (e.g., usage of $N$ and $L$).
- Value: ⭐⭐⭐⭐ MLLM unlearning is an important emerging direction; path-level editing provides a practical framework with direct applicability to privacy compliance.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] CrossVid: A Comprehensive Benchmark for Evaluating Cross-Video Reasoning in Multimodal Large Language Models](crossvid_a_comprehensive_benchmark_for_evaluating_cross-vide.md)
- [\[AAAI 2026\] Cross-modal Proxy Evolving for OOD Detection with Vision-Language Models](cross-modal_proxy_evolving_for_ood_detection_with_vision-lan.md)
- [\[AAAI 2026\] SpeakerLM: End-to-End Versatile Speaker Diarization and Recognition with Multimodal Large Language Models](speakerlm_end-to-end_versatile_speaker_diarization_and_recognition_with_multimod.md)
- [\[AAAI 2026\] InEx: Hallucination Mitigation via Introspection and Cross-Modal Multi-Agent Collaboration](inex_hallucination_mitigation_via_introspection_and_cross-mo.md)
- [\[AAAI 2026\] Aligning the True Semantics: Constrained Decoupling and Distribution Sampling for Cross-Modal Alignment](aligning_the_true_semantics_constrained_decoupling_and_distr.md)

<!-- RELATED:END -->

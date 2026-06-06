---
title: >-
  [Paper Note] Fine-Grained Activation Steering: Steering Less, Achieving More
description: >-
  [ICLR 2026][LLM/NLP][activation steering] AUSteer reveals that block-level activation steering is inherently heterogeneous—different dimensions govern different token distributions…
tags:
  - "ICLR 2026"
  - "LLM/NLP"
  - "activation steering"
  - "atomic units"
  - "fine-grained intervention"
  - "interpretability"
  - "inference-time alignment"
date: 2026-05-08
content_hash: 99d0f68927a51e3a
---

# Fine-Grained Activation Steering: Steering Less, Achieving More

**Conference**: ICLR 2026
**arXiv**: [2602.04428](https://arxiv.org/abs/2602.04428)  
**Code**: [https://github.com/zijian678/AUSteer](https://github.com/zijian678/AUSteer)  
**Area**: LLM NLP
**Keywords**: activation steering, atomic units, fine-grained intervention, interpretability, inference-time alignment

## TL;DR
AUSteer reveals that block-level activation steering is inherently heterogeneous—different dimensions govern different token distributions, and steering the entire block simultaneously amplifies both beneficial and harmful signals. The paper proposes fine-grained steering at the Atomic Unit (AU) level: discriminative dimensions are identified via activation momentum, steering strength is adaptively allocated, and intervening on only ≤100 dimensions substantially outperforms state-of-the-art methods that steer thousands of dimensions.

## Background & Motivation

**Background**: Activation steering is a low-cost approach to modifying LLM behavior—steering vectors are extracted and injected into intermediate activations at inference time. Methods such as ITI, CAA, and SADI operate at the **block level** on attention heads, FFN layers, or the residual stream.

**Limitations of Prior Work**:
   - Block-level activations contain hundreds to thousands of dimensions, mixing beneficial, irrelevant, and harmful features.
   - Block-level steering inevitably shifts both useful and harmful token directions simultaneously—resulting in coarse-grained, inefficient, and overly intrusive interventions.
   - Steering a single dimension can **surpass** steering the entire block—indicating that block-level operation is suboptimal.

**Key Challenge**: Block-level steering couples all dimensions together, yet different dimensions govern the probability distributions of different output tokens—a fundamental heterogeneity problem.

**Key Insight**: Each column of a weight matrix is defined as an "Atomic Unit" (AU), corresponding to a single dimension of the activation. By decomposing $\mathbf{y} = \mathbf{W}\mathbf{x} = \sum_i x_i \mathbf{W}_{:,i}$, block-level intervention is factored into AU-level scalar interventions.

**Core Idea**: Steering fewer dimensions yields better results—because steering only beneficial AUs avoids the side effects introduced by harmful AUs.

## Method

### Overall Architecture
A two-stage pipeline: (1) **AU Localization**—activation momentum is computed over contrastive samples to globally score the discriminability of all AUs, and the top-$k$ most discriminative AUs are selected; (2) **Adaptive Steering**—each selected AU is steered with a strength proportional to its discriminability, and the steering direction and magnitude are adaptively adjusted based on the current activation value of the input sample.

### Key Designs

1. **Theoretical Explanation of Heterogeneity**:

    - Different AUs govern different output token distributions. As steering strength $s$ increases, the output converges to the token distribution of the corresponding AU. The KL divergence between two distinct AUs grows with $s$—confirming that they drive the model toward different output directions.
    - Steering a beneficial AU (e.g., $x_{84}$) increases the probability of the correct token "yes"; steering a harmful AU (e.g., $x_{44}$) inflates the probability of irrelevant tokens.

2. **Activation Momentum**:

    - For each AU $u_i$, the momentum $m_i^j = x_i^{j,pos} - x_i^{j,neg}$ is computed over $N$ contrastive sample pairs.
    - The positive momentum ratio $r_i^{pos}$ and negative momentum ratio $r_i^{neg}$ quantify the discriminative consistency of the AU.
    - The discriminability score $s_i = \max(r_i^{pos}, r_i^{neg})$—count-based scoring enables global comparison across layers and avoids the issue of activation magnitudes increasing with layer depth.

3. **Adaptive Steering**:

    - Input-adaptive: $\hat{x}_i = x_i + \gamma_i x_i$—steering is proportional to the current activation value rather than adding a constant (preserving direction while accommodating different inputs).
    - AU-adaptive: $\gamma_i = \alpha \cdot r_i^{pos}$ (promoting) or $\gamma_i = -\alpha \cdot r_i^{neg}$ (suppressing)—AUs with stronger discriminability receive larger steering magnitudes.

### Loss & Training
- **Completely training-free**: no gradient updates required; activation momentum is computed purely through statistics over contrastive samples.
- Only $k \leq 100$ dimensions are steered (vs. thousands of dimensions in block-level methods).
- Applicable to any position in MHA, FFN, or the residual stream.

## Key Experimental Results

### Main Results (LLaMA2-7B-Chat, Commonsense Reasoning)

| Method | Steered Dims | BoolQ↑ | COPA↑ | WinoGrande↑ |
|--------|-------------|--------|-------|------------|
| Baseline | 0 | 70.5 | - | - |
| ITI (block-level) | 128 | 71.6 | - | - |
| SADI (block-level) | 4224 | 73.7 | - | - |
| **AUSteer** | **≤100** | **76.0+** | **Gain** | **Gain** |

### Ablation Study

| Configuration | Result |
|---------------|--------|
| Single-dimension steering $x_{84}$ | 74.5% (surpasses block-level SADI at 73.7%) |
| 4 positive dimensions combined | 76%+ |
| Mixed positive + negative dimensions | Performance degradation (validates heterogeneity) |
| Effect of number of steered dimensions $k$ | $k=50$–$100$ is optimal; performance drops with excessive $k$ |

### Key Findings
- **Single dimension > entire block**: Steering only dimension 84 (74.5%) surpasses block-level ITI with 128 dimensions (71.6%) and SADI with 4,224 dimensions (73.7%).
- **100 dimensions > 4,000 dimensions**: AUSteer with ≤100 AUs significantly outperforms state-of-the-art methods steering thousands of dimensions.
- **Consistent across models**: Effective across LLaMA2-7B/13B, Mistral-7B, and other models.
- **General across tasks**: Effective for commonsense reasoning, mathematical problem solving, detoxification, and human preference alignment.
- **Cross-layer comparability of activation momentum**: Count-based scoring avoids magnitude bias introduced by layer depth.

## Highlights & Insights
- **"Steering Less, Achieving More" is a counterintuitive yet profound finding**: The conventional intuition holds that more intervention implies stronger control, but in heterogeneous systems, precisely intervening on a small number of critical points far outperforms coarse-grained global intervention. This principle generalizes to pruning, knowledge editing, and related domains.
- **The token distribution interpretation of AUs** provides a clear theoretical foundation for activation steering—each AU functions as a "micro-expert" governing the output probability of specific token types. This also implies a modular structure within the Transformer.
- **Completely training-free with only contrastive sample statistics** makes AUSteer extremely lightweight—far more general than SAE-based methods (e.g., STA), which require model-specific pretrained SAEs.

## Limitations & Future Work
- Activation momentum relies on statistics computed from contrastive samples—sample quality and quantity affect the reliability of AU selection.
- AU localization currently requires independent computation per task, lacking cross-task transferability.
- Theoretical analysis is primarily grounded in linear projection decomposition—nonlinear interactions within the attention mechanism are not fully modeled.
- The combination of AU-level steering with LoRA/SFT remains unexplored.
- Effectiveness on large models (70B+) has yet to be verified.

## Related Work & Insights
- **vs. ITI (Li et al.)**: ITI steers at the attention head level (128 dimensions); AUSteer further decomposes to individual dimensions, achieving better results with less intervention.
- **vs. SADI (Wang et al.)**: SADI is the block-level state of the art (4,224 dimensions); AUSteer surpasses it with ≤100 dimensions.
- **vs. STA (Wang et al.)**: STA uses "atoms" from SAEs but still injects at the residual stream block level; AUSteer directly operates on columns of the original weight matrix without relying on SAEs.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ AU decomposition + heterogeneity analysis + momentum-based localization constitutes a complete new paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across multiple models, tasks, ablations, theoretical validation, and human evaluation.
- Writing Quality: ⭐⭐⭐⭐⭐ The narrative of "Steering Less, Achieving More" is exceptionally clear.
- Value: ⭐⭐⭐⭐⭐ A fundamental contribution to the activation steering field—simple, practical, and broadly applicable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] The Cylindrical Representation Hypothesis for Language Model Steering](../../ICML2026/llm_nlp/the_cylindrical_representation_hypothesis_for_language_model_steering.md)
- [\[ACL 2026\] CoSToM: Causal-oriented Steering for Intrinsic Theory-of-Mind Alignment in Large Language Models](../../ACL2026/llm_nlp/costomcausal-oriented_steering_for_intrinsic_theory-of-mind_alignment_in_large_l.md)
- [\[ICLR 2026\] Weight Decay may matter more than μP for Learning Rate Transfer in Practice](weight_decay_may_matter_more_than_mup_for_learning_rate_transfer_in_practice.md)
- [\[ACL 2026\] LoCar: Localization-Aware Evaluation of In-Vehicle Assistants through Fine-Grained Sociolinguistic Control](../../ACL2026/llm_nlp/locar_localization-aware_evaluation_of_in-vehicle_assistants_through_fine-graine.md)
- [\[NeurIPS 2025\] MOOSE-Chem2: Exploring LLM Limits in Fine-Grained Scientific Hypothesis Discovery](../../NeurIPS2025/llm_nlp/moose-chem2_exploring_llm_limits_in_fine-grained_scientific_hypothesis_discovery.md)

</div>

<!-- RELATED:END -->

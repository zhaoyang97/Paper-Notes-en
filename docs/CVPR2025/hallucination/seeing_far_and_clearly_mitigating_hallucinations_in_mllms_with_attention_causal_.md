---
title: >-
  [Paper Note] Seeing Far and Clearly: Mitigating Hallucinations in MLLMs with Attention Causal Decoding
description: >-
  [CVPR 2025][Hallucination Detection][Multimodal Large Language Models (MLLMs)] FarSight is proposed as a plug-and-play, training-free decoding strategy. It introduces attention registers into the upper triangle matrix of the causal mask to absorb excessive attention on anomalous tokens, and designs positional awareness encoding with diminishing masking rates to enhance information propagation for distant visual tokens, thereby effectively mitigating initial and snowball hallu…
tags:
  - "CVPR 2025"
  - "Hallucination Detection"
  - "Multimodal Large Language Models (MLLMs)"
  - "Hallucination Mitigation"
  - "Attention Mechanism"
  - "Causal Decoding"
  - "Positional Encoding"
date: 2026-05-08
content_hash: 128ae563a1b8fbed
---

# Seeing Far and Clearly: Mitigating Hallucinations in MLLMs with Attention Causal Decoding

**Conference**: CVPR 2025  
**arXiv**: [2505.16652](https://arxiv.org/abs/2505.16652)  
**Code**: [https://mllms-farsight.github.io/](https://mllms-farsight.github.io/)  
**Area**: Hallucination Detection  
**Keywords**: Multimodal Large Language Models (MLLMs), Hallucination Mitigation, Attention Mechanism, Causal Decoding, Positional Encoding

## TL;DR
FarSight is proposed as a plug-and-play, training-free decoding strategy. It introduces attention registers into the upper triangle matrix of the causal mask to absorb excessive attention on anomalous tokens, and designs positional awareness encoding with diminishing masking rates to enhance information propagation for distant visual tokens, thereby effectively mitigating initial and snowball hallucinations in Multimodal Large Language Models (MLLMs).

## Background & Motivation

**Background**: Multimodal Large Language Models (MLLMs) such as LLaVA and InternVL perform exceptionally well in visual question-answering tasks, yet they commonly suffer from hallucination problems, generating textual descriptions that contradict the actual image content.

**Limitations of Prior Work**: Existing mitigation methods either require additional training or data (e.g., instruction tuning, external knowledge retrieval) or focus solely on contrastive decoding at the output level without deeply analyzing the root causes of hallucinations. Furthermore, prior approaches have limited effectiveness in mitigating "snowball hallucinations," where the model continuously generates errors to maintain consistency with historical context.

**Key Challenge**: The root cause of hallucinations lies in the insufficient interaction among multimodal tokens. The authors identify two critical issues: (1) **Attention Collapse**—the softmax mechanism forces all attention scores to be non-zero and normalized, causing low-information anomalous tokens (such as punctuation or visual background) to receive disproportionately high attention; (2) **Positional Information Decay**—the long-range decay characteristic of RoPE causes the information of visual tokens to gradually vanish as the generated text length increases.

**Goal**: To design a training-free, plug-and-play decoding strategy that mitigates attention collapse and positional information decay by optimizing the causal mask.

**Key Insight**: Modifying the causal mask in the attention mechanism directly—while the upper triangle of a traditional causal mask is entirely set to negative infinity (to mask future tokens), this work proposes utilizing this "free space" to place attention registers, thereby absorbing the excessive attention directed toward anomalous tokens.

**Core Idea**: Set linearly decaying attention register scores in the upper triangle matrix of the causal mask. During softmax normalization, these registers participate in the calculation (absorbing redundant attention), and are subsequently cleared to zero post-normalization to maintain causality. This achieves a more balanced attention distribution for valid tokens.

## Method

### Overall Architecture
FarSight is integrated as a plug-and-play module that replaces the standard causal mask operation in each layer of the MLLM decoder. Given the original attention score matrix $\omega$, the upper triangle attention values are first cleared using a lower triangular matrix $C$. Then, a precomputed attention register matrix $\mathcal{P}$ (where the upper triangular portion contains linearly decaying values) is added. After softmax normalization, the resulting matrix is multiplied by $C$ to clear the upper triangular probability values. The entire process only modifies the mask operation without changing the model weights.

### Key Designs

1. **Attention Registers**:

    - **Function**: To provide dedicated "attention sinks" to absorb excessive attention that would otherwise be allocated to anomalous tokens.
    - **Mechanism**: An upper triangular attention register matrix $\mathcal{P}$ is constructed, where the register values at positions $j > i$ are set to $\mathcal{P}_{i,j} = -(j-i) \cdot \sigma$, with $\sigma$ acting as a decay rate hyperparameter. The final attention matrix is formulated as $\mathbf{W} = \omega \cdot C + \mathcal{P}$. Post-softmax normalization, it is multiplied by $C$ to clear the upper triangle: $\tilde{\mathbf{W}} = \text{SoftMax}(\mathbf{W}) \cdot C$. Linear decay ensures that the registers are aligned with the relative positional encoding of RoPE.
    - **Design Motivation**: The softmax function enforces all attention scores to normalize to 1, causing even low-information tokens to receive significant attention. Registers provide a "safety outlet" for redundant attention to be absorbed, rather than being forced onto irrelevant tokens.

2. **Positional Awareness Encoding**:

    - **Function**: To enhance the attention of generated text on distant visual tokens, mitigating positional information decay.
    - **Mechanism**: Through the linear decay design of the attention registers, the cumulative sum of valid attention in each row increases monotonically with the row index $i$ ($\beta_1 < \beta_2 < \cdots < \beta_n = 1$). This means tokens at later positions can aggregate more historical context, achieving implicit encoding of absolute positional information. Consequently, later-generated tokens maintain a higher cumulative focus on early visual tokens, compensating for the long-range decay of RoPE.
    - **Design Motivation**: RoPE only encodes relative positions, leading to a gradual loss of information flow between visual and text tokens as physical distance increases. This issue is particularly severe for long sequences in video-based tasks.

3. **Dynamic Register Attention Allocation**:

    - **Function**: To adaptively optimize attention allocation at each decoding step.
    - **Mechanism**: Register scores are generated via ALiBi-like biases but with an inverted direction (applied to the upper triangle instead of the lower triangle), using different decay rates across various attention heads. A complete softmax normalization is recomputed at each decoding step, enabling the registers to dynamically adapt to the current context.
    - **Design Motivation**: Attention distributions vary significantly across different layers, heads, and timesteps, which necessitates a flexible, dynamic mechanism rather than a rigid, fixed rule.

### Loss & Training
FarSight requires completely no training. It only requires modifying the causal mask operation during inference. The primary hyperparameter is the decay rate $\sigma$, which controls the intensity of attention absorption by the registers.

## Key Experimental Results

### Main Results

| Model | Method | CHAIR_S ↓ | CHAIR_I ↓ | POPE-R ↑ | POPE-P ↑ |
|------|------|-----------|-----------|----------|----------|
| LLaVA-1.5-7B | Baseline | 48.0 | 13.9 | 87.0 | 82.8 |
| LLaVA-1.5-7B | + FarSight | Significantly reduced | Significantly reduced | Improved | Improved |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| RoPE-only (Original) | CHAIR_S=48.0 | Baseline |
| Fixed Visual Positional Encoding (FixVPE) | Marginally improved | Only fixing the positional encoding of visual tokens |
| Text-only Positional Encoding (EDVT) | Partially improved | Removing positional encoding of visual tokens |
| FarSight (Full) | Optimal | Attention registers + Positional awareness encoding |

### Key Findings
- Hallucinations can be categorized into initial hallucinations and snowball hallucinations, with snowball hallucinations accounting for a particularly high proportion in video captioning tasks.
- Existing contrastive decoding methods fail to effectively reduce the ratio of snowball hallucinations, whereas FarSight mitigates both types of hallucinations simultaneously by improving information propagation.
- The two components, namely attention registers and positional awareness encoding, each make independent contributions and achieve superior performance when synergized.
- The proposed method proves effective on both image and video benchmarks, showing particularly significant improvements in long-sequence video tasks.

## Highlights & Insights
- **Ingenious Utilization of the Upper Triangle Space**: The upper triangle of the causal mask has traditionally been completely blocked (set to negative infinity). This work discovers that this space can be leveraged to host an "attention buffer" prior to the softmax operation, which is cleared to zero afterwards. This neither violates causality nor hinders the regulation of attention distribution. Such an approach is exceptionally clever and highly generalizable.
- **Integration of Theory and Practice**: The analyses of attention collapse and positional information decay are backed by rigorous mathematical derivations (e.g., mutual information inequalities and monotonic increase proofs), setting it apart from the purely empirical approaches of most decoding strategy papers.
- **Plug-and-Play Utility**: It requires no training or external data, modifying only the attention mask operations with minimal computational overhead. It can be directly applied to any Transformer-based MLLM.

## Limitations & Future Work
- The decay rate $\sigma$ needs to be manually tuned, and different models/tasks might require distinct optimal values.
- The evaluation of this paper is primarily conducted on 7B-scale models; its efficacy on larger-scale models (e.g., 70B+) remains unexplored.
- Although the decay of positional information is theoretically analyzed, whether the register mechanism can fully compensate for the limitations of RoPE on ultra-long sequences has not been comprehensively verified.
- The intervention is restricted solely to causal decoding, leaving the optimization of attention during the prefill stage unexplored.

## Related Work & Insights
- **vs OPERA**: OPERA also addresses the issue of attention aggregation on "summary tokens," but does so by penalizing specific tokens. In contrast, FarSight provides a more elegant solution at the mask level without directly modifying the attention scores.
- **vs VCD (Visual Contrastive Decoding)**: VCD mitigates hallucinations by contrasting output distributions with and without visual inputs, which increases inference overhead. FarSight requires no additional forward pass.
- **vs ALiBi/StableMask**: These methods primarily improve length extrapolation for unimodal text, whereas FarSight is specifically optimized for visual-language token interactions in multimodal scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐ The design of upper-triangular attention registers is highly novel, and the analysis of the underlying causes of hallucinations is profound.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated across multiple models and benchmarks, accompanied by solid ablation studies and comparisons with baseline positional encoding methods.
- Writing Quality: ⭐⭐⭐⭐ The theoretical derivations are rigorous, the pseudocode is clear, and the diagrams are intuitive.
- Value: ⭐⭐⭐⭐ A highly practical, plug-and-play solution that offers valuable insights for understanding and mitigating hallucinations in MLLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] COPO: Causal-Oriented Policy Optimization for Hallucinations of MLLMs](../../CVPR2026/hallucination/copo_causal-oriented_policy_optimization_for_hallucinations_of_mllms.md)
- [\[NeurIPS 2025\] Seeing is Believing? Mitigating OCR Hallucinations in Multimodal Large Language Models](../../NeurIPS2025/hallucination/seeing_is_believing_mitigating_ocr_hallucinations_in_multimodal_large_language_m.md)
- [\[CVPR 2026\] Tell Model Where to Look: Mitigating Hallucinations in MLLMs by Vision-Guided Attention](../../CVPR2026/hallucination/tell_model_where_to_look_mitigating_hallucinations_in_mllms_by_vision-guided_att.md)
- [\[ACL 2025\] Mixture of Decoding: An Attention-Inspired Adaptive Decoding Strategy to Mitigate Hallucination in Multimodal LLMs](../../ACL2025/hallucination/mixture_of_decoding_an_attention-inspired_adaptive_decoding_strategy_to_mitigate.md)
- [\[NeurIPS 2025\] Causal-LLaVA: Causal Disentanglement for Mitigating Hallucination in Multimodal Large Language Models](../../NeurIPS2025/hallucination/causalllava_causal_disentanglement_for_mitigating_hallucinat.md)

</div>

<!-- RELATED:END -->

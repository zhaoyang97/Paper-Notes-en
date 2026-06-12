---
title: >-
  [Paper Note] ThinkOmni: Lifting Textual Reasoning to Omni-modal Scenarios via Guidance Decoding
description: >-
  [ICLR 2026][Multimodal VLM][Omni-modal reasoning] ThinkOmni is a training-free framework that leverages a text-only large reasoning model (LRM) to guide an omni-modal LLM (OLLM) during decoding via Stepwise Contrastive S…
tags:
  - "ICLR 2026"
  - "Multimodal VLM"
  - "Omni-modal reasoning"
  - "guidance decoding"
  - "LRM"
  - "training-free"
  - "contrastive scaling"
date: 2026-05-08
content_hash: f6a53ba82c801927
---

# ThinkOmni: Lifting Textual Reasoning to Omni-modal Scenarios via Guidance Decoding

**Conference**: ICLR 2026
**arXiv**: [2602.23306](https://arxiv.org/abs/2602.23306)  
**Code**: [https://1ranguan.github.io/thinkomni](https://1ranguan.github.io/thinkomni)  
**Area**: Multimodal VLM
**Keywords**: Omni-modal reasoning, guidance decoding, LRM, training-free, contrastive scaling

## TL;DR
ThinkOmni is a training-free framework that leverages a text-only large reasoning model (LRM) to guide an omni-modal LLM (OLLM) during decoding via Stepwise Contrastive Scaling, which adaptively balances perception and reasoning signals. The method achieves 70.2% on MathVista and 75.5% on MMAU, matching or surpassing RFT-based approaches.

## Background & Motivation

**Background**: Large reasoning models (LRMs) such as DeepSeek-R1 and o1 demonstrate remarkable performance on textual reasoning tasks but are limited to text-only inputs. Omni-modal LLMs (OLLMs) such as Qwen2.5-Omni can process text, audio, images, and video, yet still exhibit weaknesses in complex reasoning tasks.

**Limitations of Prior Work**: Existing approaches to enhancing OLLM reasoning face several challenges:
   - **Data scarcity**: SFT requires large quantities of high-quality multimodal reasoning samples, which are costly to obtain.
   - **Training cost**: RFT (reinforcement fine-tuning) demands substantial GPU resources (8×40G for 7B models; 16×80G for 32B models).
   - **Task specialization**: Existing enhancement methods (e.g., Omni-R1, HumanOmniV2) are limited to specific downstream tasks and lack generalizability.
   - **Modality limitation**: Most prior work focuses on a single modality (image or audio) and does not achieve true cross-modal reasoning.

**Key Challenge**: LRMs possess strong reasoning capabilities but cannot process non-textual inputs; OLLMs handle multimodal inputs but lack sufficient reasoning capacity. The two are complementary, yet how to combine them in a training-free manner at inference time remains the central challenge.

**Goal**: To transfer the textual reasoning capability of LRMs to omni-modal scenarios without additional training data or fine-tuning.

**Key Insight**: Inference-time guidance decoding is adopted, treating the LRM as a decoding-time "advisor" for the OLLM and fusing their signals at the logits level.

**Core Idea**: The textual reasoning signal produced by the LRM guides the OLLM's omni-modal decoding at the logits level, with Stepwise Contrastive Scaling adaptively regulating the perception–reasoning balance.

## Method

### Overall Architecture
The ThinkOmni framework consists of two core components: (1) **LRM-as-a-Guide**, which contrastively fuses the output logits of the OLLM and the LRM to form an enhanced decoding distribution; and (2) **Stepwise Contrastive Scaling**, which automatically computes the contribution magnitudes of perception and reasoning at each decoding step and dynamically adjusts fusion weights without manual tuning.

### Key Designs

1. **LRM-as-a-Guide**:

    - **Function**: At each decoding step, three sets of logits are obtained—from the OLLM (full omni-modal input), the OLLM (text-only input), and the LRM (text-only input)—to construct a contrastive signal.
    - **Mechanism**: Base logits $\hat{z}^{\text{base}} = M_O(x_{<t}, O)$, negative logits $z^- = M_O(x_{<t})$ (multimodal input removed), and positive logits $z^+ = M_R(x_{<t})$. The fusion formula is: $\hat{P} = \text{Softmax}[z^{\text{base}} + \alpha \cdot (z^+ - z^-)]$. The contrastive term $(z^+ - z^-)$ encodes the incremental reasoning preference of the LRM relative to the OLLM in text-only mode.
    - **Design Motivation**: Analogous to a differential amplifier, $z^+ - z^-$ amplifies the LRM's reasoning signal while suppressing linguistic noise shared by both models. Although the LRM cannot perceive multimodal content, the already-generated textual context implicitly encodes multimodal information as decoding progresses.

2. **Stepwise Contrastive Scaling**:

    - **Function**: Dynamically computes reasoning weight $\alpha_r$ and perception weight $\alpha_p$ at each decoding step, replacing the fixed scalar $\alpha$.
    - **Mechanism**: Jensen–Shannon divergence quantifies the discrepancy between the three distributions: $D_R = \text{JS}(P_R \| P)$ reflects the reasoning contribution, and $D_P = \text{JS}(P_O \| P)$ reflects the perception contribution. Subject to $\alpha_r + \alpha_p = 1$, the weights are allocated proportionally to $D_R / D_P$. A warmup mechanism is introduced to limit reasoning intervention during the initial decoding phase.
    - **Design Motivation**: Different tasks and different decoding steps require varying degrees of reasoning versus perception. Mathematical problems require larger $\alpha_r$, while audio perception tasks require larger $\alpha_p$. A fixed $\alpha$ cannot adapt to all scenarios, as experiments demonstrate that the optimal $\alpha$ varies considerably across tasks.

3. **Extended Formula**:

    - **Function**: The complete fusion formula incorporates two contrastive terms.
    - **Mechanism**: $\hat{P} = \text{Softmax}[M_O(x_{<t}, O) + \alpha_r \cdot (M_R(x_{<t}) - M_O(x_{<t})) + \alpha_p \cdot (M_O(x_{<t}, O) - M_O(x_{<t}))]$. The second contrastive term constitutes an aggressive visual contrastive decoding that directly enhances perception by differencing the outputs with and without multimodal input.
    - **Design Motivation**: Dual contrastive terms independently and simultaneously enhance reasoning and perception capabilities.

### Loss & Training
The method is entirely training-free. It requires the OLLM and LRM to share a vocabulary (e.g., both from the Qwen family). Three forward passes are required per decoding step.

## Key Experimental Results

### Main Results

| Model | MathVista | MathVision | MathVerse | MMAU | DailyOmni | OmniBench |
|------|-----------|------------|-----------|------|-----------|-----------|
| GPT-4o | 63.8 | 30.4 | 50.8 | 62.5 | 56.5 | - |
| Gemini-2.0-Flash | 73.1 | 41.3 | 59.3 | 70.5 | 67.8 | - |
| Qwen2.5-Omni-7B | 66.8 | 25.0 | 40.2 | 71.5 | 57.9 | 42.1 |
| +DeepSeek Guide | 68.8(+2.0) | 28.2(+3.2) | 42.0(+1.8) | 73.8(+2.3) | 59.8(+1.9) | 43.2(+1.1) |
| **+Qwen3 Guide** | **70.2(+3.4)** | **32.9(+7.9)** | **45.1(+4.9)** | **75.5(+4.0)** | **59.5(+1.6)** | **43.6(+1.5)** |
| Omni-R1 (RFT) | 64.7 | 25.4 | 39.8 | 70.5 | 59.6 | 43.0 |
| +Qwen3 Guide | **71.3(+6.6)** | **31.5(+6.1)** | **45.2(+5.4)** | **75.4(+4.9)** | 59.8(+0.2) | 43.4(+0.4) |

### Ablation Study — Comparison with Other Training-Free Methods (based on Qwen2.5-Omni-7B)

| Method | MathVista | MMAU | OmniBench |
|------|-----------|------|-----------|
| Base Model | 66.8 | 71.5 | 42.1 |
| Average Logits Fusion | 55.0(−11.8) | 55.7(−15.8) | 36.1(−6.0) |
| Caption-then-Answer | 61.0(−5.8) | 59.7(−11.8) | 32.3(−9.8) |
| VCD | 66.5(−0.3) | 72.2(+0.7) | 43.1(+1.0) |
| **ThinkOmni** | **68.8(+2.0)** | **73.8(+2.3)** | **43.2(+1.1)** |

### Key Findings
- Applying ThinkOmni on top of the RFT-trained Omni-R1 still yields substantial gains (MathVista +6.6), demonstrating that the method is complementary to RFT.
- Stronger LRMs (Qwen3 > DeepSeek-R1-Distill) produce larger improvements, validating that guidance quality determines the magnitude of gains.
- The largest improvements occur on mathematical and scientific tasks (MathVision +7.9), while audio and general tasks see smaller gains, consistent with the expectation that LRMs are predominantly trained on math and science data.
- Simple logits averaging severely degrades performance (−11.8), underscoring the necessity of contrastive fusion.
- Efficiency analysis: Under the 7B+7B configuration, generation latency is 2.88× and prefill latency is 1.38× compared to the base model (since the LRM processes only text and handles a lighter prefix).

## Highlights & Insights
- **Training-free framework surpasses trained methods**: Using Qwen2.5-Omni-7B + Qwen3, ThinkOmni matches or exceeds RFT-based methods such as Omni-R1 and HumanOmniV2 on multiple benchmarks.
- **Stepwise Contrastive Scaling is elegant and practical**: JS divergence automatically estimates the demand for reasoning versus perception at each step, eliminating the burden of manual hyperparameter tuning.
- **Plug-and-play and scalable**: As stronger LRMs emerge (LRM development typically outpaces multimodal variants), ThinkOmni can benefit automatically.
- **Rich qualitative analysis**: Token-level visualizations of LRM contributions show that logical connectives and key technical terms are primarily guided by the LRM, while content words are contributed by the OLLM.

## Limitations & Future Work
- The requirement that the OLLM and LRM share a vocabulary restricts the flexibility of model pairing (e.g., a LLaMA-family LRM cannot guide a Qwen-family OLLM).
- Three forward passes per step incur approximately 2.88× inference overhead relative to the base model, which poses challenges for latency-sensitive deployments.
- Gains on audio and general omni-modal tasks remain limited (DailyOmni +1.6 only), indicating that the method is less effective for perception-intensive tasks.
- When multimodal inputs contain contradictory information (e.g., labels that conflict with visual content), the LRM may misdirect reasoning.

## Related Work & Insights
- The key distinction from ProxyTuning (which belongs to the same guidance decoding paradigm) is that ThinkOmni enables cross-modal guidance without requiring the LRM to perceive multimodal inputs.
- ThinkOmni is complementary to VCD (Visual Contrastive Decoding): VCD enhances perception, while ThinkOmni enhances reasoning.
- The work introduces a new paradigm for "reasoning capability transfer": rather than fine-tuning the model, capabilities are grafted via logits fusion at inference time.

## Rating
- Novelty: ⭐⭐⭐⭐ Cross-modal guidance decoding is a novel idea; Stepwise Contrastive Scaling is an elegant design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Six benchmarks, three OLLMs, multiple LRMs, comprehensive ablations, and efficiency analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, thorough theoretical analysis, and rich qualitative visualizations.
- Value: ⭐⭐⭐⭐⭐ A training-free method that surpasses RFT approaches is highly practical and offers an important paradigm-level contribution to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] OMHBench: Benchmarking Balanced and Grounded Omni-Modal Multi-Hop Reasoning](../../ACL2026/multimodal_vlm/omhbench_benchmarking_balanced_and_grounded_omni-modal_multi-hop_reasoning.md)
- [\[ICLR 2026\] Self-Aug: Query and Entropy Adaptive Decoding for Large Vision-Language Models](self-aug_query_and_entropy_adaptive_decoding_for_large_vision-language_models.md)
- [\[ICML 2026\] OmniSIFT: Modality-Asymmetric Token Compression for Efficient Omni-modal Large Language Models](../../ICML2026/multimodal_vlm/omnisift_modality-asymmetric_token_compression_for_efficient_omni-modal_large_la.md)
- [\[AAAI 2026\] Leveraging Textual Compositional Reasoning for Robust Change Captioning](../../AAAI2026/multimodal_vlm/leveraging_textual_compositional_reasoning_for_robust_change_captioning.md)
- [\[ICLR 2026\] Reasoning-Driven Multimodal LLM for Domain Generalization](reasoning-driven_multimodal_llm_for_domain_generalization.md)

</div>

<!-- RELATED:END -->

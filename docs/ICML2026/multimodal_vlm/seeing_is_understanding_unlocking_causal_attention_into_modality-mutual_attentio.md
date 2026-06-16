---
title: >-
  [Paper Note] Seeing is Understanding: Unlocking Causal Attention into Modality-Mutual Attention for Multimodal LLMs
description: >-
  [ICML 2026][Multimodal VLM][MLLM] The authors modify the causal attention mask in decoder-only MLLMs by "digging a hole" that allows preceding image tokens to retrospectively attend to subsequent text question tokens. This single-line mask modification requires no extra parameters or training data changes, achieving an average improvement of 6.2 points
tags:
  - ICML 2026
  - Multimodal VLM
  - MLLM
  - Attention
date: 2026-05-08
content_hash: dae9ca5c980fa441
---
# Seeing is Understanding: Unlocking Causal Attention into Modality-Mutual Attention for Multimodal LLMs

**Conference**: ICML 2026  
**arXiv**: [2503.02597](https://arxiv.org/abs/2503.02597)  
**Code**: https://github.com/sony/aki  
**Area**: Multimodal VLM  
**Keywords**: MLLM, Attention Mechanism, Cross-modality Alignment, Hallucination Mitigation, Causal Mask  

## TL;DR
The authors modify the causal attention mask in decoder-only MLLMs by "digging a hole" that allows preceding image tokens to retrospectively attend to subsequent text question tokens. This single-line mask modification requires no extra parameters or training data changes, achieving an average improvement of 6.2 points across 3 LLM backbones and 12 multimodal benchmarks.

## Background & Motivation
**Background**: Mainstream MLLMs (LLaVA-1.5, BLIP-3, Cambrian, MM-1.5, etc.) share a three-part architecture: vision encoder → vision-language connector (VL-connector) → decoder-only LLM. The input sequence is arranged as $S=[V, T_Q]$: $|V|$ image tokens are placed first, followed by $|T_Q|$ text question tokens, and the answer $T_R$ is decoded autoregressively.

**Limitations of Prior Work**: MLLMs still frequently suffer from object hallucinations in vision-centric tasks (counting, spatial relations, detail recognition). As shown in Fig. 1, GPT-4o, Molmo, and DeepSeek-VL2-Small all fail to correctly identify a complex parking sign with specific time-limited restrictions.

**Key Challenge**: Previous attempts to mitigate this focused on data scaling (e.g., Molmo adding clock/pointing/counting data) or replacing the VL-connector (abstractor, spatial vision aggregator). The former requires massive annotation budgets, while the latter lacks an established optimal solution (McKinzie et al. 2024 empirically showed no single connector wins across all benchmarks). However, this paper points out the actual bottleneck is the LLM’s causal attention itself—the lower triangular mask designed for unimodal autoregression prevents preceding image tokens from ever "seeing" the subsequent text. Consequently, the image representation remains a static set of features regardless of whether the user asks "how many cars" or "what color."

**Goal**: Enable the "preceding modality (image)" to perceive the "subsequent modality (text question)" without adding parameters, disrupting autoregressive generation, or breaking existing SFT pipelines.

**Key Insight**: The authors first perform a sanity check by alternating the training order between [Image, Text] and [Text, Image] (Dual-Order Training, DOT). The resulting improvements prove that "allowing the preceding modality to see the subsequent one" is the correct direction. However, DOT doubles training time and scales at $n!$ for $n$ modalities. This motivates them to bypass training order and directly modify the attention mask.

**Core Idea**: Transform the causal mask $M$ into $M'$ by specifically unlocking the "image token → text question token" rectangular region during the SFT stage, while keeping other positions unchanged.

## Method

### Overall Architecture
The method follows a standard two-stage pipeline (PT + SFT) based on the design by Cha et al. 2024: a vision encoder $f_V$ (CLIP-like) extracts image features, a VL-connector $p_V$ projects them into the text space, and a text embedder $f_T$ produces query embeddings. Finally, the LLM $f_L$ autoregressively generates $T_R = f_L(H_V, H_{T_Q})$ based on $H_V \in \mathbb{R}^{|V|\times d}$ and $H_{T_Q} \in \mathbb{R}^{|T_Q|\times d}$. During pre-training (PT using Blip3-kale for captioning), the vision encoder is frozen while the VL-connector and LLM are updated; MMA is not used here as there is no specific user question. During SFT, the vision encoder remains frozen, and the MMA mask is enabled. For the generated answer $T_R$, standard causal attention is maintained—MMA only affects the input $S$, and the KV cache is reused for the rest of the dialogue.

### Key Designs

**1. Dual-Order Training (DOT): A "costly but logical" baseline proving cross-modal visibility works**
Before modifying the mask, the authors confirmed that the inability of images to see text is a bottleneck through DOT. They train each stage twice: once with $[T_Q, V]$ and once with $[V, T_Q]$. This allows the model to learn dependencies where the preceding modality depends on the subsequent one. Results on LLaMA-3.2-3B showed LLaVA-W scores rising from 38.6 to 43.8 and CV-Bench2D from 37.5 to 46.7. However, the $n!$ cost for $n$ modalities makes this implementation impractical, necessitating a data-agnostic mask modification.

**2. Modality-Mutual Attention (MMA): Digging a rectangular hole in the causal mask**
MMA achieves the effect of DOT directly in the mask. Standard causal attention is $\text{Attention}_{causal} = \text{softmax}((QK^T + M)/\sqrt{d})$, where $M_{ij}=0$ if $j \le i$ and $-\infty$ otherwise. MMA modifies this to $M'$: taking 0 when $j \le i$ (preserving causality) OR when $1 \le i \le |V|$ and $|V|+1 \le j \le |V|+|T_Q|$ (allowing image positions to see question positions). Geometrically, this opens a $|V|\times|T_Q|$ rectangular path in the upper-right corner. This allows image tokens to dynamically focus on different regions based on the question (e.g., "how many cars" vs. "what color"), effectively moving "question-driven vision encoding" inside the LLM.

**3. Zero-parameter implementation and multimodal generalization**
MMA is a "one-line" integration because it only swaps several $-\infty$ values for 0. The attention matrix size $(|V|+|T_Q|)^2$, softmax computation, and KV cache structure remain identical. It introduces zero trainable parameters and zero additional FLOPs. For interleaved multimodal inputs, the paper defines a generalized condition: $M'_{ij}=0$ if $j\le i$ or $\phi(i)\ne\phi(j)$, where $\phi$ maps a token to its modality. This allows different modalities to be mutually visible while maintaining internal causality within each modality.

### Loss & Training
A standard two-stage pipeline is used: PT uses Blip3-kale for captioning; SFT uses a mixture of VQAv2/VSR/GQA/OCRVQA (Open VQA), ScienceQA/A-OKVQA (Multiple Choice), RefCOCO/RefCOCO+/RefCOCOg/VisualGenome (Referring Expression), and LLaVA-150k (Instruction Following). LLM parameters are fully updated. MMA is only enabled during SFT.

## Key Experimental Results

### Main Results
The authors evaluated MMA across 3 LLM backbones $\times$ 12 benchmarks. Below are representative SFT results on LLaMA-3.2-3B.

| Configuration (LLaMA-3.2-3B) | MMEP | MMB | LLaVA-W | POPE | RealWorldQA | CV-Bench2D | CV-Bench3D |
|-----------------------------|-----:|----:|--------:|-----:|------------:|-----------:|-----------:|
| SFT Baseline (w/o T&I)      | 1134.2 | 51.3 | 38.6 | 73.5 | 37.8 | 37.5 | 50.7 |
| SFT Reverse (w/o I&T)       | 1128.1 | 51.6 | 34.1 | 72.7 | 35.6 | 39.1 | 51.4 |
| DOT (Dual-Order)            | 1219.5 | 46.9 | 43.8 | 77.0 | 42.0 | 46.7 | 52.9 |
| **MMA (Ours)**              | Avg. +6.2% across 12 benchmarks | | | | | | |

The gain is most significant in vision-centric (CV-Bench, RealWorldQA) and hallucination-related (POPE) benchmarks, confirming that "question-driven image representation" is effective for tasks requiring dynamic focus.

### Ablation Study

| Configuration | Key Observation | Description |
|---------------|-----------------|-------------|
| Causal Mask (Baseline) | Constant image representation | The core bottleneck identified in Sec. 3.3. |
| Reverse Input (T&I) | Limited gain / occasional drop | Swapping order alone doesn't fix the mask bottleneck. |
| DOT (Dual-Order) | Strong gains but 2× training cost | Proves cross-modality visibility is the right path. |
| **MMA (Unlocked Rect.)** | +6.2% avg, constant training cost | Equivalent improvement with zero overhead. |
| Adding MMA to PT | Not implemented | Captioning lacks specific user questions. |

### Key Findings
- Reverse input (T&I) shows almost no improvement on most benchmarks, proving the bottleneck is the mask's restriction, not just the token order.
- Both DOT and MMA demonstrate the benefits of relaxing cross-modal visibility, but MMA serves as a superior, more efficient replacement for DOT.
- High gains in vision-centric tasks match the expectation that image representations should change based on the question.

## Highlights & Insights
- A "textbook" example of achieving significant gains through a minimal change—modifying a single rectangle in the attention mask unlocks a fundamental architectural limitation.
- The use of DOT as a diagnostic baseline is highly convincing: proving the direction with a "costly" method and then replacing it with a "cheap" one.
- The generalized condition $\phi(i)\ne\phi(j) \Rightarrow M'_{ij}=0$ provides immediate value for future Any-to-Any models involving audio or video.
- Suggests that the difficulty in optimizing VL-connectors might stem from the overlooked causal mask assumption within the LLM itself.

## Limitations & Future Work
- MMA is only used during SFT. Whether a suitable prompt could make MMA effective for PT (captioning) remains unexplored.
- The experiments focus on single-image + single-turn QA; the grouping logic for $\phi$ in multi-turn or multi-image scenarios (treating images as the same or different modalities) needs refinement.
- Does not compare against unlocking image→answer tokens; whether the image should "see" already generated answer tokens is an open question.
- While covering 12 benchmarks, validation on video-language and audio-language tasks is currently theoretical.

## Related Work & Insights
- **vs Concentric Causal Attention (CCA)**: CCA assumes central image regions are more important via mask biases; MMA avoids heuristic assumptions by letting the image see the text question directly.
- **vs Mixed Attention**: Mixed Attention uses full attention between image tokens but keeps image→text paths closed; MMA focuses specifically on the "understanding" phase needs.
- **vs VL-connectors (Cambrian/Honeybee)**: While others improve the connector, MMA argues the bottleneck is the LLM mask, offering an orthogonal and stackable improvement path.
- **vs Molmo**: Data-centric methods are expensive; MMA is a cost-effective alternative for data-scarce scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐ Identifies a fundamental flaw in using unimodal causal masks for multimodal tasks.
- Experimental Thoroughness: ⭐⭐⭐⭐ Large-scale validation (3 backbones × 12 benchmarks) with strong diagnostic baselines.
- Writing Quality: ⭐⭐⭐⭐ Clear progression from sanity checks (DOT) to the final method (MMA).
- Value: ⭐⭐⭐⭐⭐ Plug-and-play, zero-cost, high-impact modification for any MLLM stack.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">
</div>

## Related Papers

- [\[ICML 2026\] Smoothing Slot Attention Iterations and Recurrences](smoothing_slot_attention_iterations_and_recurrences.md)
- [\[ICML 2026\] Large Vision-Language Models Get Lost in Attention](large_vision-language_models_get_lost_in_attention.md)
- [\[ICML 2025\] MODA: MOdular Duplex Attention for Multimodal Perception, Cognition, and Emotion Understanding](../../ICML2025/multimodal_vlm/moda_modular_duplex_attention_for_multimodal_perception_cognition_and_emotion_un.md)
- [\[ICML 2026\] Hyper-ICL: Attention Calibration with Hyperbolic Anchor Distillation for Multimodal ICL](hyper-icl_attention_calibration_with_hyperbolic_anchor_distillation_for_multimod.md)
- [\[ICLR 2026\] Constructive Distortion: Improving MLLMs with Attention-Guided Image Warping](../../ICLR2026/multimodal_vlm/constructive_distortion_improving_mllms_with_attention-guided_image_warping.md)

</div>

<!-- RELATED:END -->

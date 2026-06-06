---
title: >-
  [Paper Note] Seeing is Understanding: Unlocking Causal Attention into Modality-Mutual Attention for Multimodal LLMs
description: >-
  [ICML 2026][Multimodal VLM][MLLM] The authors modify the causal attention mask in decoder-only MLLMs by "puncturing" a hole, allowing preceding image tokens to conversely attend to subsequent text question tokens. This p…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "MLLM"
  - "Attention Mechanism"
  - "Cross-modal alignment"
  - "Hallucination mitigation"
  - "Causal mask"
date: 2026-05-08
content_hash: e43399e7fbf31189
---

# Seeing is Understanding: Unlocking Causal Attention into Modality-Mutual Attention for Multimodal LLMs

**Conference**: ICML 2026  
**arXiv**: [2503.02597](https://arxiv.org/abs/2503.02597)  
**Code**: https://github.com/sony/aki  
**Area**: Multimodal VLM  
**Keywords**: MLLM, Attention Mechanism, Cross-modal alignment, Hallucination mitigation, Causal mask  

## TL;DR
The authors modify the causal attention mask in decoder-only MLLMs by "puncturing" a hole, allowing preceding image tokens to conversely attend to subsequent text question tokens. This parameter-free mask modification requires no change to training data and achieves an average improvement of 6.2 points across 3 LLM backbones and 12 multimodal benchmarks.

## Background & Motivation
**Background**: Mainstream MLLMs (LLaVA-1.5, BLIP-3, Cambrian, MM-1.5, etc.) share a three-part skeleton: Vision Encoder $\to$ Vision-Language Connector (VL-connector) $\to$ decoder-only LLM. The input sequence is arranged as $S=[V, T_Q]$: $|V|$ image tokens are followed by $|T_Q|$ text question tokens, and the answer $T_R$ is decoded auto-regressively.

**Limitations of Prior Work**: MLLMs still frequently suffer from object hallucinations in vision-centric tasks (counting, spatial relations, detail recognition). As shown in Fig. 1, GPT-4o, Molmo, and DeepSeek-VL2-Small all fail on a complex parking sign with specific time limits.

**Key Challenge**: Previous mitigation efforts focused on data scaling (e.g., Molmo adding clock/pointing/counting data) or replacing the VL-connector (abstractor, spatial vision aggregator). The former requires massive labeling budgets, while the latter lacks a consensus optimal solution. However, the paper points out that the true bottleneck is the causal attention of the LLM itself. The lower-triangular mask, originally designed for unimodal auto-regression, prevents preceding image tokens from seeing subsequent text. Consequently, image representations remain static regardless of the user's question, meaning the dialogue content has zero retroactive influence on image understanding.

**Goal**: Enable the "preceding modality (image)" to perceive the "subsequent modality (text question)" without adding parameters, disrupting auto-regressive generation, or breaking existing SFT pipelines.

**Key Insight**: The authors first perform a sanity check by alternating the training order between [image, text] and [text, image] (Dual-Order Training, DOT). This leads to improvements, proving that "allowing the pre-modality to see the post-modality" is the correct direction. However, DOT doubles training time and its cost scales at $n!$ for $n$ modalities. This motivates them to bypass training order and directly modify the attention mask.

**Core Idea**: Transform the causal mask $M$ into $M'$, specifically unlocking the rectangular "image token $\to$ text question token" region during the SFT stage while keeping other positions unchanged.

## Method

### Overall Architecture
The method follows a standard two-stage pipeline (PT + SFT). The skeleton follows the design of Cha et al. 2024: a vision encoder $f_V$ extracts image features, a VL-connector $p_V$ projects them into the text space, and a text embedder $f_T$ produces query embeddings. Finally, the LLM $f_L$ auto-regressively generates $T_R = f_L(H_V, H_{T_Q})$ over $H_V \in \mathbb{R}^{|V|\times d}$ and $H_{T_Q} \in \mathbb{R}^{|T_Q|\times d}$. During pre-training (captioning with Blip3-kale), the vision encoder is frozen while the VL-connector and LLM are updated; MMA is not used here as there are no specific user questions. During SFT, the vision encoder remains frozen, and the MMA mask is enabled. During generation, the produced answer $T_R$ still follows standard causal attention—MMA only applies to the input $S$, and the dialogue continues normally once stored in the KV cache.

### Key Designs

1.  **Dual-Order Training (DOT) — Intuitive Baseline**:
    - **Function**: Explicitly feeds the model two input orders, $[V, T_Q]$ and $[T_Q, V]$, allowing the LLM to learn dependencies where the pre-modality relies on the post-modality.
    - **Mechanism**: Employs tandem training—each stage trains on T&I order first, then I&T order, ensuring alignment with I&T during inference. Formalized as $[T_{Q_{PT}}, V_{PT}] \to [V_{PT}, T_{Q_{PT}}] \to [T_{Q_{SFT}}, V_{SFT}] \to [V_{SFT}, T_{Q_{SFT}}]$.
    - **Design Motivation**: Used as a diagnostic experiment to prove the necessity of unlocking cross-modal attention. DOT yields significant gains (e.g., LLaVA-W increases from 38.6 to 43.8 on LLaMA-3.2-3B) but at the cost of doubled training time, necessitating a more efficient alternative that does not change the training data.

2.  **Modality-Mutual Attention (MMA) — Mask-level Unlocking**:
    - **Function**: Modifies the causal mask $M$ to $M'$ by adding a path in the rectangular region of "image position $i$, text question position $j$," allowing image tokens to attend to subsequent text question tokens.
    - **Mechanism**: Standard causal attention is $\text{Attention}_{causal} = \text{softmax}((QK^T + M)/\sqrt{d})$, where $M_{ij}=0$ if $j \le i$ and $-\infty$ otherwise. MMA modifies $M'_{ij}$ to be 0 if $j \le i$ (preserving causality) OR if $1 \le i \le |V|$ and $|V|+1 \le j \le |V|+|T_Q|$ (allowing images to see the question). This effectively "carves out" a $|V|\times|T_Q|$ rectangular path in the upper-right corner.
    - **Design Motivation**: While causal masks are suitable for unimodal text, they are a bottleneck for multimodal dialogues where images are static and questions are dynamic. With MMA, image tokens can refocus on different regions based on the query (e.g., "how many cars" vs "what color"), effectively integrating "query-driven visual encoding" into the LLM.

3.  **Zero-Parameter Engineering Implementation**:
    - **Function**: Implements MMA without introducing trainable parameters, increasing FLOPs, or modifying PT stages and SFT data.
    - **Mechanism**: MMA only replaces certain $-\infty$ values with 0. The total number of elements in the attention matrix remains $(|V|+|T_Q|)^2$, softmax computation volume is unchanged, and the KV cache structure is preserved. For interleaved multimodal inputs, the paper provides a generalized condition $M'_{ij}=0$ when $j\le i$ or $\phi(i)\ne\phi(j)$ (where $\phi$ maps tokens to modalities), implying different modalities are mutually visible by default while remaining causal within the same modality.
    - **Design Motivation**: To make the method a "one-line code change" for existing MLLM training stacks without requiring connector or dataset redesigns.

### Loss & Training
A standard two-stage pipeline is used: PT uses Blip3-kale for captioning; SFT mixes VQAv2/VSR/GQA/OCRVQA (open-ended VQA), ScienceQA/A-OKVQA (multiple-choice), RefCOCO/RefCOCO+/RefCOCOg/VisualGenome (referring expressions), and LLaVA-150k (instruction following). The vision encoder is frozen in both stages while the VL-connector and LLM (full parameter update) are trainable. MMA is strictly enabled during SFT.

## Key Experimental Results

### Main Results
The authors compared MMA against (a) traditional I&T causal, (b) reversed T&I causal, and (c) DOT across 3 LLM backbones and 12 benchmarks. Representative results for LLaMA-3.2-3B during SFT are shown below (POPE measures hallucination, CV-Bench is vision-centric).

| Configuration (LLaMA-3.2-3B) | MMEP | MMB | LLaVA-W | POPE | RealWorldQA | CV-Bench2D | CV-Bench3D |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (w/o T&I) SFT Baseline | 1134.2 | 51.3 | 38.6 | 73.5 | 37.8 | 37.5 | 50.7 |
| (w/o I&T) Reversed SFT | 1128.1 | 51.6 | 34.1 | 72.7 | 35.6 | 39.1 | 51.4 |
| DOT (Dual-Order) | 1219.5 | 46.9 | 43.8 | 77.0 | 42.0 | 46.7 | 52.9 |
| **MMA (Ours)** | **Avg +6.2% across 12 benchmarks** | | | | | | |

The average Gain of +6.2% across 3 backbones and 12 benchmarks is most significant in vision-centric and hallucination-related tasks, confirming that query-driven image representations benefit tasks requiring dynamic focus.

### Ablation Study

| Configuration | Key Observation | Explanation |
| :--- | :--- | :--- |
| Causal Mask (Baseline) | Constant image representation for different queries | Core bottleneck identified in Sec. 3.3 |
| Reversed Input (T&I) | Limited improvement or occasional decline | Changing order doesn't solve the mask bottleneck |
| DOT Training | Gain on most benchmarks but 2x cost | Validates direction of cross-modal visibility |
| **MMA (Unlocked Rect.)** | **+6.2% Avg, constant training cost** | Equivalent improvement with zero extra parameters/compute |
| MMA in PT | Not selected | Captioning lacks user questions; semantics are invalid |

### Key Findings
- Reversed input (T&I) rarely improves performance, suggesting the bottleneck is "images cannot see text" rather than "incorrect order."
- Both DOT and MMA highlight the benefits of relaxing cross-modal visibility, but MMA serves as a complete replacement for DOT with its zero-cost implementation.
- Gains are highest in vision-centric tasks (CV-Bench, RealWorldQA, POPE) and smaller in knowledge-based or pure-text reasoning (MMMU), consistent with the mechanism of query-dependent image representations.

## Highlights & Insights
- This is a "textbook case" of obtaining significant gains from a minimal change—modifying a single rectangular region in the attention mask exposes and fixes a fundamental architectural limitation of MLLMs.
- Using DOT as a diagnostic baseline is highly persuasive: proving the direction with an expensive but reasonable baseline, then replacing it with a far cheaper method.
- The generalized condition $\phi(i)\ne\phi(j) \Rightarrow M'_{ij}=0$ naturally extends to interleaved inputs of $\ge 2$ modalities, offering direct transfer value for future Any-to-Any models.
- It reveals a potential reason why VL-connector tuning often fails to reach an optimum—the problem may not be the connector, but the overlooked causal mask assumption within the LLM.

## Limitations & Future Work
- MMA is only enabled during SFT. While the authors state PT lacks user questions, whether suitable prompts could be constructed for MMA during PT remains unexplored.
- Experiments focus on single-image + single-turn QA; the details of $\phi$ grouping for multi-turn or multi-image inputs need further refinement.
- There is no comparison with unlocking "image $\to$ answer tokens." Whether images should "see" partially generated answers during decoding remains an open question.
- Although 12 benchmarks provide wide coverage, validation on video-language and audio-language tasks is currently focused on theoretical "generalization."

## Related Work & Insights
- **vs Concentric Causal Attention (CCA, Xing et al. 2024)**: CCA assumes central image regions are more important and uses masks to mitigate hallucinations; however, it does not bridge cross-modal information flow. MMA allows images to see the text question directly.
- **vs Mixed Attention (Xie et al. 2025)**: Mixed Attention uses full attention between image tokens in unified models but keeps image-to-text blocked; MMA unlocks this specifically for understanding scenarios.
- **vs Cambrian/Honeybee Connectors**: These focus on the VL-connector; MMA argues the bottleneck is the mask inside the LLM, providing a path orthogonal and additive to connector improvements.
- **vs Data-centric methods (Molmo)**: Data scaling is expensive; MMA requires no extra data or connector changes, making it a cost-effective choice for data-scarce scenarios.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Minimal change with a unique perspective—identifying the fundamental irrationality of causal masks in multimodal contexts.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Large-scale validation across 3 backbones and 12 benchmarks with strong diagnostic baselines.
- **Writing Quality**: ⭐⭐⭐⭐ Logical progression from sanity checks (DOT) to the formal method (MMA).
- **Value**: ⭐⭐⭐⭐⭐ Zero parameters, zero extra compute, plug-and-play capability for any existing MLLM stack.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Smoothing Slot Attention Iterations and Recurrences](smoothing_slot_attention_iterations_and_recurrences.md)
- [\[ICML 2026\] Hyper-ICL: Attention Calibration with Hyperbolic Anchor Distillation for Multimodal ICL](hyper-icl_attention_calibration_with_hyperbolic_anchor_distillation_for_multimod.md)
- [\[ICML 2026\] Large Vision-Language Models Get Lost in Attention](large_vision-language_models_get_lost_in_attention.md)
- [\[ICML 2026\] DIVA: Harnessing the Representation Divergence in Unified Multimodal Models for Mutual Reinforcement](diva_harnessing_the_representation_divergence_in_unified_multimodal_models_for_m.md)
- [\[ICLR 2026\] Constructive Distortion: Improving MLLMs with Attention-Guided Image Warping](../../ICLR2026/multimodal_vlm/constructive_distortion_improving_mllms_with_attention-guided_image_warping.md)

</div>

<!-- RELATED:END -->

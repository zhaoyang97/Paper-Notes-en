---
title: >-
  [Paper Note] Same Attention, Different Truths: Put Logit-Lens over Visual Attention to Detect and Mitigate LVLM Object Hallucination
description: >-
  [CVPR 2026][Hallucination Detection][Object Hallucination] This paper revisits LVLM object hallucination using Logit-Lens and discovers that the "attention intensity" for real and hallucinated objects is nearly identical in mid-to-late layers. The key issue is not "how much" the model looks, but whether the high-attention regions decode into the target token. Based on this, hallucinations are categorized into "Visual Uncertainty" and "Contextual Prior." A training-free "Detec…
tags:
  - "CVPR 2026"
  - "Hallucination Detection"
  - "Object Hallucination"
  - "LVLM"
  - "Logit Lens"
  - "Attention Analysis"
  - "Training-free Decoding"
date: 2026-05-08
content_hash: 2b24ea1a8377d929
---

# Same Attention, Different Truths: Put Logit-Lens over Visual Attention to Detect and Mitigate LVLM Object Hallucination

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Same_Attention_Different_Truths_Put_Logit-Lens_over_Visual_Attention_to_CVPR_2026_paper.html)  
**Code**: https://github.com/wzczc/SADT  
**Area**: Multimodal VLM  
**Keywords**: Object Hallucination, LVLM, Logit Lens, Attention Analysis, Training-free Decoding  

## TL;DR
This paper revisits LVLM object hallucination using Logit-Lens and discovers that the "attention intensity" for real and hallucinated objects is nearly identical in mid-to-late layers. The key issue is not "how much" the model looks, but whether the high-attention regions decode into the target token. Based on this, hallucinations are categorized into "Visual Uncertainty" and "Contextual Prior." A training-free "Detect-and-Mitigate" framework (LLCC detection + HARM masking + VEED decoding enhancement) is proposed, achieving SOTA on multiple hallucination benchmarks.

## Background & Motivation
**Background**: Object hallucination in LVLMs (describing objects not present in the image) is a major obstacle to reliability. A prevailing view attributes hallucination to "insufficient visual attention"—either text priors suppress visual focus, or improper visual processing leads to inadequate attention on salient regions. Corresponding mitigation methods mostly focus on **amplifying/redistributing attention** or injecting additional visual guidance.

**Limitations of Prior Work**: The authors find this explanation **incomplete**. By quantifying layer-wise attention intensity for object tokens, they observed that visual attention peaks in the mid-to-late layers (Layers 20–27, termed the "Image Attention Stage"). However, **there is almost no systematic difference in image attention intensity between real and hallucinated object tokens during this stage** (Fig. 1b)—both converge attention on specific regions. Thus, "not looking enough" cannot explain hallucination.

**Key Challenge**: The problem is not *how much* the model looks at the image, but *what* it sees and *why* it looks there. Using Logit Lens (translating intermediate visual representations into words via the language head), the authors "read" the semantics of high-attention areas. They found that **high-attention areas for real objects decode into tokens consistent with the target, while those for hallucinated objects do not**—this is "Same Attention, Different Truths."

**Goal**: (1) Detect hallucinations using semantic consistency instead of attention intensity; (2) Understand "why the model attends to these regions" and apply targeted mitigations.

**Key Insight**: By masking high-attention regions and regenerating, the authors classify hallucinations into two types: **Visual Uncertainty** (hallucination disappears after masking, rooted in forced interpretation of blurry/similar regions) and **Contextual Prior** (hallucination persists and attention shifts, rooted in co-occurrence priors where attention is just a procedural anchor). On LLaVA-1.5-7b, the ratio between these two types is approximately 2:1.

**Core Idea**: Use "Logit-Lens decoding consistency of high-attention regions" for detection, then apply "masking" and "visual evidence enhanced decoding" to mitigate the two types of hallucinations respectively.

## Method

### Overall Architecture
The proposed method is a training-free "Detection-Classification-Mitigation" pipeline. Given an image and a prompt, the model generates a description. For each generated token, the system first filters **object tokens** using average image attention during the Image Attention Stage ($S_{IA}$). It then performs **Logit-Lens Consistency Check (LLCC)**: if the decoded semantics of high-attention regions are inconsistent with the token, it is flagged as a hallucination. These are then categorized via **High-Attention Region Masking**: if the hallucination disappears after masking, it is **Type 1 (Visual Uncertainty)**; if it remains, it is **Type 2 (Contextual Prior)**. Type 1 is mitigated via **HARM** (Masking unreliable visual evidence), while Type 2 is addressed via **VEED** (Injecting real visual semantics into decoding logits to override priors). The entire pipeline requires no training or model modification.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["LVLM Generation<br/>Image + Description"] --> B["LLCC Consistency Detection<br/>Compare Logit-Lens decoding of<br/>high-attention regions with output token"]
    B -->|Semantically Consistent| G["Retain as Real Object"]
    B -->|Inconsistent (Hallucination)| C["High-Attention Region Masking Classification<br/>Regeneration after Masking"]
    C -->|Hallucination Disappears (Visual Uncertainty)| D["HARM Mitigation<br/>Remove unreliable visual evidence"]
    C -->|Hallucination Persists (Contextual Prior)| E["VEED Decoding Enhancement<br/>Inject real visual logits"]
    D --> F["Final Output"]
    E --> F
    G --> F
```

### Key Designs

**1. Logit-Lens Consistency Check (LLCC): Detection by "What is Seen" instead of "Attention Intensity"**

Addressing the unreliability of attention intensity, LLCC bases detection on semantic consistency. First, it performs **object token localization**: calculating average image attention $A_{\text{img}}(o_t)=\frac{1}{|S_{IA}|}\sum_{l\in S_{IA}}\sum_{j=1}^{N_{\text{img}}}\alpha^{(l)}_{tj}$ during the $S_{IA}$ stage. Tokens exceeding $\tau_{\text{attn}}$ (set to 0.15) are treated as object tokens. Then, it performs a **semantic consistency check**: taking the Top-$k$ ($k=3$) image tokens $\Omega_t$ from the attention map and decoding the highest probability word $v_t^i=\arg\max_v\text{softmax}(W_U h(\omega_t^i))_v$ via Logit Lens. If WordNet semantic similarity $\text{Sim}(o_t,v_t^i) > \tau_{\text{sim}}$ (set to 0.8), it is deemed real; otherwise, it is a hallucination.

**2. High-Attention Region Masking Classification: Distinguishing Causes via Regeneration**

Once a hallucination is detected, the cause must be determined. Following the causal analysis, the Top-$k$ high-attention image patches for the hallucinated token are masked. A new sequence $O_{\text{new}}$ is generated. If $o_t \notin O_{\text{new}}$, it is **Type 1 (Visual Uncertainty)**, meaning the hallucination was rooted in that specific blurry region. If $o_t \in O_{\text{new}}$, it is **Type 2 (Contextual Prior)**, meaning the co-occurrence prior dominated the generation and attention was merely a procedural anchor.

**3. HARM (High-Attention Region Masking): Removing Faulty Evidence for Type 1**

The root cause of Type 1 is the model forcing semantics from high-uncertainty regions. HARM uses the same masking as the classification step: a binary mask $M$ is applied to the Top-$k$ patches $\Omega$ of hallucinated tokens, replacing them with $\mu$ (mean color or black) to obtain $I_{\text{mask}}=(1-M)\odot I+M\odot\mu$. Regeneration on $I_{\text{mask}}$ removes the unreliable visual anchor, naturally eliminating the hallucination.

**4. VEED (Visual Evidence Enhanced Decoding): Amplifying Real Visual Semantics for Type 2**

Type 2 is immune to masking because strong priors override correct visual evidence. VEED injects visual semantics during decoding by taking the Logit-Lens visual logits $z_t^{\text{vis}}=W_U h(\omega_t^{\max})$ from the most attended region and fusing them with the masked model logits $z_t^{\text{mask}}=\text{logits}_\theta(y_t\mid y_{<t},I_{\text{mask}})$:

$$p_t=\text{softmax}\big((1-\alpha)\,z_t^{\text{vis}}+\alpha\,z_t^{\text{mask}}\big).$$

A smaller $\alpha$ increases the weight of real visual semantics in the final decision, elevating image-supported objects and suppressing prior-driven ones.

### Loss & Training
The method is entirely **training-free**. It does not require additional data, training objectives, or RL. All operations (attention filtering, Logit-Lens decoding, masking, logit fusion) occur during inference. Key hyperparameters include $\tau_{\text{attn}}=0.15$, Top-$k=3$, $\tau_{\text{sim}}=0.8$, and $\alpha$ for VEED.

## Key Experimental Results

### Main Results
Comparison on the CHAIR benchmark (500 images from COCO2014, lower is better for CHAIR$_S$ / CHAIR$_I$):

| Method | LLaVA-7B | LLaVA-13B | Shikra-7B | Qwen2-VL-7B |
|------|------|------|------|------|
| Greedy | 49.8 / 20.4 | 47.8 / 19.8 | 58.4 / 22.2 | 31.4 / 12.7 |
| VCD | 56.3 / 22.9 | 50.3 / 18.9 | 52.4 / 19.8 | 32.1 / 13.2 |
| OPERA | 42.9 / 18.7 | 42.1 / 16.4 | 38.1 / 16.7 | 28.0 / 10.3 |
| Devils | 32.1 / 13.7 | 35.4 / 14.1 | 32.8 / 13.1 | – |
| PAI | 29.8 / 13.2 | 33.2 / 13.5 | 37.9 / 15.0 | 24.7 / 8.6 |
| **Ours** | **26.8 / 10.0** | **31.3 / 12.4** | **31.4 / 12.7** | **24.0 / 8.3** |

AMBER Generation (LLaVA-1.5-7B, lower is better for CHAIR/Hal/Cog, higher is better for Cover):

| Method | CHAIR ↓ | Cover ↑ | Hal ↓ | Cog ↓ |
|------|------|------|------|------|
| Greedy | 6.9 | 51.0 | 32.0 | 3.3 |
| VCD | 6.4 | **52.1** | 33.2 | 2.9 |
| Devils | 3.5 | 50.2 | 19.9 | 1.3 |
| PAI | 4.2 | 50.7 | 18.4 | 1.6 |
| **Ours** | **2.8** | 51.2 | **14.7** | **1.2** |

### Ablation Study (Detection Performance)
Comparison of LLCC with three recent detection methods on 500 images:

| Method | Precision | Recall | F1 |
|------|------|------|------|
| Uncertainty Score | 0.5965 | 0.6415 | 0.6182 |
| InterConf | 0.6717 | 0.6907 | 0.6811 |
| SVAR | 0.6500 | 0.7222 | 0.6842 |
| **LLCC (Ours)** | **0.7870** | **0.7955** | **0.7932** |

### Key Findings
- **Quality over Quantity of Attention**: SVAR uses summed attention ratios but only achieves an F1 of 0.6842. LLCC targets visual sources and checks semantic consistency, raising F1 to 0.7932. Attention only aligns with visual evidence in late stages; cross-layer aggregation introduces noise.
- **Mitigation without Sacrificing Content**: While many baselines reduce hallucinations at the cost of Object Coverage (Cover), this method maintains or slightly improves Cover (51.2 on AMBER) while achieving superior CHAIR/Hal/Cog scores due to minimal intervention.
- **2:1 Hallucination Ratio**: On LLaVA-1.5-7B, Visual Uncertainty to Contextual Prior follows a 2:1 ratio, validating the need for bifurcated mitigation.

## Highlights & Insights
- **Revising the "Insufficient Attention" Attribution**: The finding that real and hallucinated objects share the same attention intensity (Fig. 1b) successfully challenges the current consensus, while Logit-Lens provides a more nuanced mechanical explanation.
- **Masking as a Causal Probe**: Using masking to identify the root cause of hallucination allows the system to reuse the diagnostic tool as a mitigation mechanism simultaneously.
- **Targeted Treatment Paradigm**: The "Detect → Classify → Branching Mitigation" structure is highly transferable to attribute or relation hallucinations by simply redefined what "evidence" looks like.

## Limitations & Future Work
- The method relies on an empirical "Image Attention Stage" (Layers 20–27 for LLaVA-1.5-7B). It is unclear if this generalizes across all architectures (e.g., Qwen2-VL) without re-calibration.
- **Inference Overhead**: VEED requires an extra masking forward pass, and the classification stage involves regeneration, leading to higher latency than pure decoding baselines.
- Reliance on WordNet for similarity limits the system's performance on fine-grained or open-vocabulary objects.

## Related Work & Insights
- **vs SVAR**: SVAR relies on attention volume; this work proves intensity is indistinguishable for hallucinations and that semantic consistency (LLCC) is a superior signal.
- **vs PAI**: PAI amplifies visual attention indiscriminately. This work argues that "more attention" is a surface-level fix and outperforms PAI by treating prior-driven and uncertainty-driven hallucinations differently.
- **vs VCD / OPERA**: Unlike these "one-size-fits-all" mitigations, this work identifies the cause first, allowing it to preserve coverage while reducing hallucinations.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ "Same attention, different truths" discovery + bifurcated diagnosis is highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Strong results across models, though hyperparameter sensitivity is largely relegated to the appendix.
- Writing Quality: ⭐⭐⭐⭐⭐ Extremely clear narrative—falsifying old explanations followed by providing new mechanisms.
- Value: ⭐⭐⭐⭐ Plug-and-play and training-free, though inference cost and stage-heuristics are practical hurdles.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] First Logit Boosting: Visual Grounding Method to Mitigate Object Hallucination in Large Vision-Language Models](first_logit_boosting_visual_grounding_method_to_mitigate_object_hallucination_in.md)
- [\[CVPR 2026\] Cross-Modal Attention Calibration for LVLM Hallucination Mitigation](cross-modal_attention_calibration_for_lvlm_hallucination_mitigation.md)
- [\[CVPR 2026\] PAS: Prelim Attention Score for Detecting Object Hallucinations in Large Vision-Language Models](pas_prelim_attention_score_for_detecting_object_hallucinations_in_large_vision-l.md)
- [\[CVPR 2026\] AdaIAT: Adaptively Increasing Attention to Generated Text to Alleviate Hallucinations in LVLM](adaiat_adaptively_increasing_attention_to_generated_text_to_alleviate_hallucinat.md)
- [\[CVPR 2026\] VES-RFT: Rewarding Visual Evidence Sensitivity to Mitigate Hallucinations in Large Vision-Language Models](ves-rft_rewarding_visual_evidence_sensitivity_to_mitigate_hallucinations_in_larg.md)

</div>

<!-- RELATED:END -->

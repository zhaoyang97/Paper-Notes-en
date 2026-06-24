---
title: >-
  [Paper Note] Cracking the Code of Hallucination in LVLMs with Vision-aware Head Divergence
description: >-
  [ACL 2025][Hallucination Detection][Hallucination Mitigation] Proposes the VHD metric to quantify how sensitive the output of each attention head is to visual input. It finds that only a few attention heads are highly sensitive to visual information, and the model's over-reliance on language priors is a key factor causing hallucinations. Based on this, a training-free method, VHR, is designed to adaptively reinforce the contribution of vision-sensitive heads layer-by-layer ($…
tags:
  - "ACL 2025"
  - "Hallucination Detection"
  - "Hallucination Mitigation"
  - "Vision-aware Head Divergence"
  - "Attention Head Analysis"
  - "Language Bias"
  - "Training-free Decoding"
date: 2026-05-08
content_hash: cfb3d7c6e3fc5905
---

# Cracking the Code of Hallucination in LVLMs with Vision-aware Head Divergence

**Conference**: ACL 2025  
**arXiv**: [2412.13949](https://arxiv.org/abs/2412.13949)  
**Code**: [VHR](https://github.com/jinghan1he/VHR)  
**Area**: Hallucination Detection  
**Keywords**: Hallucination Mitigation, Vision-aware Head Divergence, Attention Head Analysis, Language Bias, Training-free Decoding

## TL;DR

Proposes the VHD metric to quantify how sensitive the output of each attention head is to visual input. It finds that only a few attention heads are highly sensitive to visual information, and the model's over-reliance on language priors is a key factor causing hallucinations. Based on this, a training-free method, VHR, is designed to adaptively reinforce the contribution of vision-sensitive heads layer-by-layer ($\alpha=2$), reducing the CHAIR$_S$ of LLaVA-1.5 on CHAIR from 49.68 to 33.32, with almost no additional inference overhead.

## Background & Motivation

**Background**: Large Vision-Language Models (LVLMs) have made significant progress in multimodal reasoning. However, the hallucination problem—where generated text is inconsistent with visual content—seriously undermines the accuracy and reliability of the models.

**Limitations of Prior Work**: (1) Existing methods (alignment training, post-processing, contrastive decoding) mainly intervene at the output level and do not look deeply into the internal mechanism of hallucination; (2) Contrastive decoding (VCD, DoLa, etc.) directly manipulates logit distributions, introducing generation instability; (3) There is a lack of systematic analysis of the attention mechanisms inside the model that drive hallucinations.

**Key Challenge**: LVLMs tend to prioritize language patterns (language bias). Even without image input, models can generate highly coherent content descriptions. This bias is embedded in model parameters, causing the output to depend more on internal knowledge than on visual context.

**Goal**: To explore the internal drivers of hallucination from the perspective of multi-head attention mechanisms, and design active mitigation strategies based on the findings.

**Key Insight**: Inspired by research on "context heads" and "memory heads" in LLMs, this work explores the differential sensitivity of different attention heads to visual content.

**Core Idea**: Quantify the sensitivity differences of attention heads to visual context, and adaptively amplify the output of highly sensitive attention heads to enhance the model's visual reliance.

## Method

### Overall Architecture

Divided into two phases: (1) Quantifying the visual sensitivity of attention heads using the VHD metric, and aggregating it into the T-VHD metric to establish a quantitative correlation between hallucination and language bias; (2) Adaptively selecting and amplifying key attention heads layer-by-layer based on VHD scores (VHR). The entire process is completed in a single forward pass, requiring only one extra forward pass at the very first generation step to compute VHD.

### Key Designs

1. **Vision-aware Head Divergence (VHD) Metric**
    - **Function**: Quantify the sensitivity of each attention head's output to visual input.
    - **Mechanism**: At generation step $t$, compute the output of the $i$-th attention head in the $l$-th layer with and without image input, and take the Euclidean distance: $\text{VHD}_{l,i} = d(A_{l,i}(y_t|y_{<t}, x_V, x_T),\ A_{l,i}(y_t|y_{<t}, x_T))$. The top-$k$ VHDs of each layer are aggregated to obtain the Token-VHD (T-VHD).
    - **Key Findings**: Only a few attention heads exhibit significantly high VHD scores; hallucinated words/sentences correspond to lower T-VHD scores, statistically validating the correlation between language bias and hallucination.
    - **Design Motivation**: A training-free, sample-adaptive metric is needed to capture the model's reliance on visual information.

2. **Vision-aware Head Reinforcement (VHR)**
    - **Function**: Enhance the contribution of vision-sensitive attention heads without training to actively mitigate hallucination.
    - **Mechanism**: For each layer, first calculate VHD scores and filter out outliers ("negatively vision-sensitive" heads which have high VHD due to sudden surges of activation when there is no image), then select the top half of the attention heads whose VHD scores exceed the median, and amplify their outputs by $\alpha$ times: $\widetilde{A}_{l,i} = \alpha \cdot A_{l,i}$ if $i \in H_l$.
    - **Three Key Implementation Details**: (a) Layer-by-layer application: when processing the current layer, the preceding layers have already been reinforced, ensuring consistency in VHD calculation; (b) Key heads are determined only at the first generation step and reused in subsequent steps, making it compatible with KV cache; (c) No extra annotation is required, adaptively selecting different heads for each sample.
    - **Design Motivation**: Intervention directly inside the model is superior to output-level correction. Amplifying vision-sensitive heads redirects the attention output toward visual evidence.

3. **Theoretical Analysis of Attention Output Redirection**
    - **Function**: Prove that the amplification operation effectively redirects the MHA output direction.
    - **Mechanism**: Due to the normalization of RMSNorm, only the direction affects the subsequent FFN input. Proposition 1 proves that after amplifying the output of the $h$-th head by $\alpha$ times, the cosine similarity between the FFN input $\widetilde{Z}_l$ and the $Z_{l,h}$ (which only contains the contribution of that head) strictly increases: $\cos(\widetilde{Z}_l, Z_{l,h}) > \cos(Z_l, Z_{l,h})$.
    - **Design Motivation**: To provide mathematical guarantees for the rationality of the amplification operation.

## Key Experimental Results

### Main Results — CHAIR Benchmark (MSCOCO 500 images, average of 5 random samplings)

| Method | InstructBLIP CHAIR$_S$↓ | InstructBLIP CHAIR$_I$↓ | LLaVA-1.5 CHAIR$_S$↓ | LLaVA-1.5 CHAIR$_I$↓ | LLaVA-NeXT CHAIR$_S$↓ | LLaVA-NeXT CHAIR$_I$↓ |
|------|:-:|:-:|:-:|:-:|:-:|:-:|
| Greedy | 45.32 | 12.98 | 49.68 | 14.32 | 29.08 | 8.08 |
| DoLa | 46.00 | 13.00 | 50.88 | 14.64 | 28.76 | 8.12 |
| VCD | 50.72 | 14.42 | 51.92 | 15.42 | 30.80 | 8.72 |
| OPERA | 45.76 | 13.06 | 44.28 | 13.36 | - | - |
| EAH | 46.40 | 13.13 | 38.76 | 11.05 | 28.13 | 6.62 |
| **VHR** | **37.76** | **9.75** | **33.32** | **9.71** | **24.96** | **6.80** |

### Ablation Study — Attention Head Selection Strategy (CHAIR Benchmark)

| Configuration | InstructBLIP CHAIR$_S$↓ | LLaVA-1.5 CHAIR$_S$↓ | LLaVA-NeXT CHAIR$_S$↓ |
|------|:-:|:-:|:-:|
| VHR (Adaptive head selection) | 37.76 | 33.32 | 24.96 |
| fixed VHR (Fixed head selection) | 45.40 | 44.72 | 36.96 |
| outlier VHR (Without outlier filtering) | 37.76 | 36.88 | 24.64 |

### Key Findings

- VHR comprehensively outperforms all training-free baselines on three LVLMs, reducing CHAIR$_S$ by 16.36 points on LLaVA-1.5 (49.68 $\rightarrow$ 33.32).
- Adaptive sample-by-sample head selection is crucial—fixed head selection significantly degrades performance (33.32 $\rightarrow$ 44.72).
- Outlier VHD score filtering is effective, preventing the amplification of "negatively vision-sensitive" heads.
- An amplification factor of $\alpha=2$ works best; $\alpha=4$ leads to abnormal model behavior, while $\alpha<1$ (weakening vision-sensitive heads) significantly worsens hallucinations.
- POPE F1 and LLaVA-Bench accuracy are also improved, and the generation length and naturalness remain largely unaffected.
- Inference time overhead is negligible: only one extra forward pass in the first step, and subsequent steps only require scaling operations.

## Highlights & Insights

1. **Elegant definition of VHD/T-VHD metrics**: By comparing the difference in attention head outputs solely by removing the image input without any annotations, it can be computed on the fly for each sample, offering both analytical power and practical value.
2. **Paradigm shift from "post-hoc correction" to "pre-hoc intervention"**: Unlike contrastive decoding which corrects at the logit level, VHR directly enhances vision sensitivity inside the model, offering a clearer theoretical path.
3. **Strong theoretical proof**: Proposition 1 rigorously proves the directional redirection effect of the amplification operation, making it more than just an empirical trick.
4. **Visualization of language bias in experiments**: Even after removing the image, the model still generates highly consistent descriptions, pointing to a strong and visible language bias.

## Limitations & Future Work

1. Focuses only on the multi-head attention mechanism; the contributions of the vision encoder and FFN modules to hallucination remain unexplored.
2. VHD requires an additional forward pass without image input; although the overhead is small, it is not zero.
3. Has not been validated on larger-scale models (e.g., >13B) and more LVLM architectures.
4. Lacks fine-grained analysis of hallucination types (objects/attributes/relations).

## Related Work & Insights

- The "with-or-without vision contrast" idea of VHD can be extended to attention head analysis in other modalities (e.g., audio).
- The layer-by-layer adaptive selection-and-reinforce strategy can be applied to other scenarios requiring reinforcement of specific information flows (e.g., safety heads in safety alignment).
- The T-VHD metric can serve as a real-time signal for hallucination detection to automatically identify unreliable tokens during inference.

## Rating

⭐⭐⭐⭐

- **Novelty** ⭐⭐⭐⭐: Tackling hallucination analysis from the internal mechanism of attention heads; both the VHD metric and VHR method are novel.
- **Experimental Thoroughness** ⭐⭐⭐⭐: Comprehensive coverage with three LVLMs, multiple benchmarks, and detailed ablation studies.
- **Writing Quality** ⭐⭐⭐⭐: Clear theoretical derivations and intuitive visualization analysis.
- **Value** ⭐⭐⭐⭐: A training-free, efficient, and theoretically supported hallucination mitigation solution with high practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] ETF: An Entity Tracing Framework for Hallucination Detection in Code Summaries](etf_an_entity_tracing_framework_for_hallucination_detection_in_code_summaries.md)
- [\[CVPR 2026\] CausalLens: Sensitivity-Guided Multi-Head Causal Intervention for Hallucination Mitigation in Large Vision-Language Models](../../CVPR2026/hallucination/causallens_sensitivity-guided_multi-head_causal_intervention_for_hallucination_m.md)
- [\[ACL 2026\] Hallucination Detection in LLMs with Topological Divergence on Attention Graphs](../../ACL2026/hallucination/hallucination_detection_in_llms_with_topological_divergence_on_attention_graphs.md)
- [\[ICLR 2026\] Hallucination-aware Intermediate Representation Edit in Large Vision-Language Models](../../ICLR2026/hallucination/hallucination-aware_intermediate_representation_edit_in_large_vision-language_mo.md)
- [\[ACL 2025\] Activation Steering Decoding: Mitigating Hallucination in Large Vision-Language Models through Bidirectional Hidden State Intervention](activation_steering_decoding_mitigating_hallucination_in_large_vision-language_m.md)

</div>

<!-- RELATED:END -->

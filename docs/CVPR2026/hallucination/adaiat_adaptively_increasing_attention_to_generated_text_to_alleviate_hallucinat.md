---
title: >-
  [Paper Note] AdaIAT: Adaptively Increasing Attention to Generated Text to Alleviate Hallucinations in LVLM
description: >-
  [CVPR 2026][Hallucination Detection][Paper Note] Addressing the issue where "amplifying image attention suppresses hallucinations but leads to repetitive and wordy output," this paper discovers that real object tokens possess higher attention to the **previously generated text** $T_p$ than hallucinated tokens. Consequently, the authors propose increasing attention sp
tags:
  - CVPR 2026
  - Hallucination Detection
date: 2026-05-08
content_hash: 2bf7066353aa8652
---
# AdaIAT: Adaptively Increasing Attention to Generated Text to Alleviate Hallucinations in LVLM

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Zhong_AdaIAT_Adaptively_Increasing_Attention_to_Generated_Text_to_Alleviate_Hallucinations_CVPR_2026_paper.html)  
**Code**: https://github.com/XianguiKang/AdaIAT.git  
**Area**: Multimodal VLM / Hallucination Mitigation  
**Keywords**: LVLM Hallucination, Attention Intervention, Adaptive Decoding, Generated Text Attention, Training-free

## TL;DR
Addressing the issue where "amplifying image attention suppresses hallucinations but leads to repetitive and wordy output," this paper discovers that real object tokens possess higher attention to the **previously generated text** $T_p$ than hallucinated tokens. Consequently, the authors propose increasing attention specifically to $T_p$ (IAT). By further employing layer-wise thresholds to control "when to intervene" and a head-wise amplification matrix to control "how much to amplify" (AdaIAT), they significantly reduce hallucination rates (CS/CI) on LLaVA-1.5, Janus-Pro, and Qwen2.5-VL with almost no loss in text diversity.

## Background & Motivation
**Background**: Hallucinations in Large Vision-Language Models (LVLMs)—describing objects that do not exist in the image—are a major obstacle to deployment. A popular category of training-free methods is **attention intervention**: researchers found that hallucinations are highly correlated with the model "ignoring the image and having insufficient attention to image tokens $V$." Thus, they directly amplify attention weights to $V$ during inference, with representative works being PAI and HGAI. These methods are cost-effective as they require no training and have low inference overhead.

**Limitations of Prior Work**: Amplifying image attention is a double-edged sword. The authors observe that while PAI/HGAI reduce hallucination rates, they significantly impair linguistic ability—specifically manifesting as **repetition**. The model repeatedly describes the most prominent objects in the image (e.g., repeating "clock tower" or "the motorcycle is parked on the street" in Figure 1). The text diversity metric Distinct-1 drops by approximately 15% on LLaVA-1.5-7B.

**Key Challenge**: Why does amplifying image attention lead to repetition? Because attention weight after softmax is "zero-sum"—allocating massive weight to image tokens $V$ relatively **suppresses attention to the generated text $T_p$**. $T_p$ carries contextual memory of "what I just said." Once attention deviates from $T_p$, the model "forgets" the preceding text and repeatedly describes the most salient objects. This creates a trade-off between hallucination rates and linguistic coherence.

**Key Insight & Core Idea**: Instead of focusing on $V$, the authors ask: Can we amplify attention to $T_p$ without changing $V$? This is supported by a key observation: statistical analysis of 22,015 real object tokens and 9,473 hallucinated object tokens shows that **real object tokens have significantly higher attention to $T_p$ than hallucinated ones (approx. 1.5×–2.5×); this gap is even more pronounced than their difference in attention to image tokens $V$ (approx. 1.2×–1.5×)**. The reason is that $V$ is produced by a vision encoder, is heterogeneous to text, and contains much "instruction-irrelevant" visual info. In contrast, $T_p$ is an output organized by the LLM after seeing the image and instructions; it is naturally "instruction-relevant, compressed, and purified" visual information that already resides within the text feature space $I_s$, bridging the modal gap. Therefore, the **Core Idea** is to **increase attention to $T_p$ to simultaneously suppress hallucinations and preserve coherence**—the former is supported by concentrated instruction-relevant visual info in $T_p$, while the latter is maintained by contextual knowledge in $T_p$ to preserve diversity.

## Method

### Overall Architecture
The method is built upon standard auto-regressive decoding. It is a **training-free intervention strategy that only modifies attention weights during forward inference**, progressing through two layers. For predicting token $t_{n+1}$ at step $n$, the LLM input $I$ consists of the system prompt $S$, image tokens $V$, user instructions $U$, and generated text $T_p=\{t_1,\dots,t_n\}$. For layer $l$ and head $h$, the self-attention for token $t_n$ is $\boldsymbol{A}^{(l,h)}=\mathrm{softmax}(\tilde{\boldsymbol{A}}^{(l,h)})$, where $\tilde{\boldsymbol{A}}^{(l,h)}=\boldsymbol{Q}^{(l,h)}_{t_n}(\boldsymbol{K}^{(l,h)})^\top/\sqrt{d_k}$.

The first layer is **IAT (Increase Attention to $T_p$)**: simple amplification of all attention directed at $T_p$ in the middle layers (layers 5–18). This shifts the target of PAI’s image attention amplification from $V$ to $T_p$. The second layer is **AdaIAT**: simple amplification has two coarse aspects—it amplifies regardless of hallucination tendency (disrupting normal prediction) and uses a fixed coefficient for all heads (ignoring head variance). AdaIAT uses a **layer-wise threshold** to determine "if attention to $T_p$ is truly insufficient to trigger intervention" and a **head-wise amplification matrix $\mathcal{M}$** to assign customized magnitudes to each head. This ensures precise intervention with minimal disturbance to original predictions. Key parameters (threshold $\mathcal{T}$, matrix $\mathcal{M}$) are calculated once from COCO and fixed for other datasets.

### Key Designs

**1. IAT: Switching target from images to "already generated text" $T_p$ to suppress hallucinations and prevent repetition**

This step directly addresses the issue that "amplifying image attention leads to wordiness." Instead of increasing attention to $V$, the authors apply vanilla amplification to all $\mathcal{I}(i)\in T_p$ in middle layers ($l\in(5,18)$):

$$\tilde{\boldsymbol{A}}^{(l,h)}(i)=\tilde{\boldsymbol{A}}^{(l,h)}(i)+\alpha\cdot|\tilde{\boldsymbol{A}}^{(l,h)}(i)|$$

where $\alpha$ is the amplification coefficient. The only but most essential difference from PAI is: in PAI, $\mathcal{I}(i)\in V$ (amplifying images); in IAT, $\mathcal{I}(i)\in T_p$ (amplifying generated text). This is effective because $T_p$ is purified "instruction-relevant visual info." Amplifying it forces the model to rely more on this concentrated visual prior at each step; meanwhile, since $T_p$ contains contextual memory, the model does not "forget the past" to repeat salient descriptions. Experiments show IAT's Distinct-1 distribution nearly overlaps with Greedy decoding and even has a higher proportion in high-score ranges (0.65–0.8), while PAI/HGAI shift left (more repetitive). Intervening only in layers 5–18 is also crucial: ablations show that intervening across 0–18 or 5–31 causes model collapse (F1 drops to 30–48, D1 to 0.03–0.16).

**2. Layer-wise Threshold: Triggering intervention only when attention to $T_p$ is insufficient**

Vanilla IAT amplifies constantly regardless of whether the model is likely to hallucinate. Blindly amplifying during normal prediction can cause abnormally high attention to $T_p$, hurting accuracy. AdaIAT introduces a layer-wise threshold $\mathcal{T}\in\mathbb{R}^L$:

$$\mathcal{T}=\bar{\mathbf{A}}_{T_p}^{h}+\beta\,(\bar{\mathbf{A}}_{T_p}^{r}-\bar{\mathbf{A}}_{T_p}^{h})$$

where $\bar{\mathbf{A}}_{T_p}^{r}$ and $\bar{\mathbf{A}}_{T_p}^{h}$ are the average per-layer $T_p$ attention (statistically measured on COCO) during real and hallucinated object generation, respectively. $\beta$ is a balance coefficient. During inference, if the actual attention $\bar{\mathbf{A}}^{(l)}_{T_p}<\mathcal{T}(l)$, it indicates insufficient focus on $T_p$ and triggers IAT; otherwise, normal decoding is maintained. If $\beta$ is too small, the intervention is too weak; if too large, it triggers too frequently. Ablation shows CS/CI rebounds and F1/D1 slides when $\beta > 0.5$, so $\beta=0.5$ is selected.

**3. Head-wise Amplification Matrix $\mathcal{M}$: Customizing magnitudes based on "hallucination sensitivity"**

A fixed $\alpha$ is too blunt. Different heads show varying attention differences between real vs. hallucinated object generation. AdaIAT constructs a head-wise ratio matrix:

$$\mathcal{M}=\frac{\mathbf{A}^{r}_{T_p}}{\mathbf{A}^{h}_{T_p}},\qquad \mathcal{M}\in\mathbb{R}^{L\times H}$$

$\mathcal{M}^{(l,h)}$ represents the ratio of average $T_p$ attention in the "real state" to the "hallucinated state" for head $h$ in layer $l$. This naturally points towards "pulling the hallucinated attention pattern back to the real attention pattern." The IAT amplification is rewritten as head-weighted and re-normalized:

$$\boldsymbol{A}^{(l,h)}(i)=\boldsymbol{A}^{(l,h)}(i)+\alpha\cdot\mathcal{M}^{(l,h)}\cdot\boldsymbol{A}^{(l,h)}(i),\quad \mathcal{I}(i)\in T_p$$

$$\boldsymbol{A}^{(l,h)}(k)=\frac{\boldsymbol{A}^{(l,h)}(k)}{\sum_k\mathbf{A}^{(l,h)}(k)},\quad k\in(1,len)$$

Heads with large $\mathcal{M}$ (large attention deficit during hallucination) receive stronger amplification, while heads with small $\mathcal{M}$ maintain weak amplification. Note that the intervention here occurs **after** softmax on $\boldsymbol{A}$ (unlike the vanilla IAT on $\tilde{\boldsymbol{A}}$), as $\mathcal{M}$ is a ratio of $\boldsymbol{A}$. AdaIAT achieves 2.6 higher F1 than IAT at similar hallucination rates, demonstrating stronger prediction capability.

## Key Experimental Results

### Main Results
Evaluated on LLaVA-1.5, Janus-Pro, and Qwen2.5-VL using the CHAIR benchmark. Lower CS (sentence-level) and CI (instance-level) are better; higher F1 (accuracy/richness) and D1 (Distinct-1 diversity) are better.

| Model | Method | CS ↓ | CI ↓ | F1 ↑ | D1 ↑ |
|------|------|------|------|------|------|
| LLaVA-1.5-7B | Greedy | 49.0 | 13.3 | 77.9 | 0.60 |
| LLaVA-1.5-7B | PAI | 31.8 | 7.8 | 77.7 | 0.50 |
| LLaVA-1.5-7B | HGAI | 31.4 | 6.9 | 78.3 | 0.50 |
| LLaVA-1.5-7B | IAT | 29.8 | 9.0 | 76.8 | 0.61 |
| LLaVA-1.5-7B | **AdaIAT** | 31.4 | 8.3 | **79.4** | 0.60 |
| Janus-Pro-7B | Greedy | 25.8 | 6.7 | 76.8 | 0.62 |
| Janus-Pro-7B | PAI | 20.4 | 5.6 | 76.1 | 0.61 |
| Janus-Pro-7B | HGAI | 21.0 | 5.3 | 75.9 | 0.62 |
| Janus-Pro-7B | **AdaIAT** | **19.0** | **4.9** | 76.5 | 0.64 |

Key comparison: PAI/HGAI tank D1 from 0.60 to 0.50 on LLaVA-1.5, while IAT/AdaIAT maintain 0.60–0.61 with comparable or lower hallucination rates. Compared to LLaVA-1.5-7B Greedy, AdaIAT reduces CS by 35.8% and CI by 37.1%.

### Ablation Study

| Configuration | Key Phenomenon | Explanation |
|------|---------|------|
| IAT vs AdaIAT | F1 +2.6 at similar CS/D1 | Head-wise adaptive amplification gains prediction power |
| $\alpha$ (IAT) | F1/D1 drop sharply after $\alpha \ge 0.8$ | IAT uses $\alpha=0.8$ |
| $\alpha$ (AdaIAT) | F1 peak at 79.4 ($\alpha=6$), D1 degrades at $\ge 8$ | AdaIAT uses $\alpha=6$ |
| $\beta$ | CS/CI drops till $\beta=0.5$, then rebounds | Trigger threshold, optimal at $\beta=0.5$ |
| Layers (5–18) | 0–18/5–31 causes collapse (F1 30-48, D1 0.03-0.16) | Middle layers are most balanced |

### Key Findings
- **Head-wise adaptation is the core increment of AdaIAT over IAT**: Removing it causes a 2.6 drop in F1, proving that differential amplification for different heads is effective in maintaining original prediction capability.
- **Intervention layers are extremely sensitive**: Intervening in middle layers 5–18 is stablest. Combining shallow with middle/deep layers causes total collapse, suggesting that $T_p$ attention amplification must be restricted to layers where semantics are formed but output is not yet locked.
- **$\beta$ has an optimal value**: More triggers $\neq$ better results. Excessive triggering leads to abnormally high $T_p$ attention, increasing hallucinations and dropping F1/D1.

## Highlights & Insights
- **Shifting "what to amplify" from $V$ to $T_p$ is a clever perspective shift**: By changing only the target of intervention, the paper solves both hallucination and repetition issues simultaneously, backed by statistical observations.
- **Using "Real vs. Hallucinated" statistics to drive both threshold and magnitude**: Both $\mathcal{T}$ and $\mathcal{M}$ are derived from the same offline statistics, unifying the logic of "pulling back the hallucinated state to the real state."
- **Mechanistic explanation of PAI/HGAI repetition**: Attributing the repetition to "softmax zero-sum crushing $T_p$ attention" provides a useful insight for future attention intervention methods.

## Limitations & Future Work
- **Dependence on offline COCO statistics**: Although shown to be transferable, robustness in domains far from COCO (e.g., medical, documents) is not fully verified. ⚠️
- **Manual tuning of the 5–18 layer window**: The paper does not provide an automated solution for selecting the optimal layer window when switching model architectures.
- **Task concentration**: Evaluation is centered on image captioning; effectiveness on VQA or complex reasoning tasks is not yet established.

## Related Work & Insights
- **vs. PAI**: PAI amplifies $V$; Ours amplifies $T_p$. The advantage is avoiding repetition (D1 preserved) at the cost of offline statistics.
- **vs. HGAI**: HGAI also amplifies $V$ and fuses head info, suffering from D1 degradation.
- **vs. Contrastive Decoding (VCD/AGLA)**: These rely on output distribution contrast or integration; AdaIAT is a lighter-weight pure attention amplification approach.
- **vs. Post-processing (LURE/Woodpecker)**: These require training rewriters or calling expert models; AdaIAT is training-free with lower inference costs.

## Rating
- Novelty: ⭐⭐⭐⭐ "Shifting target from image to generated text" is a clear new perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 4 LVLMs and 4 evaluation types with extensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Motivations and observations are clearly presented.
- Value: ⭐⭐⭐⭐ Training-free and lightweight; highly practical for engineering.

<!-- RELATED:START -->
<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Cross-Modal Attention Calibration for LVLM Hallucination Mitigation](cross-modal_attention_calibration_for_lvlm_hallucination_mitigation.md)
- [\[CVPR 2026\] Same Attention, Different Truths: Put Logit-Lens over Visual Attention to Detect and Mitigate LVLM Object Hallucination](same_attention_different_truths_put_logit-lens_over_visual_attention_to_detect_a.md)
- [\[CVPR 2026\] PAS: Prelim Attention Score for Detecting Object Hallucinations in Large Vision-Language Models](pas_prelim_attention_score_for_detecting_object_hallucinations_in_large_vision-l.md)
- [\[ICLR 2026\] SHIELD: Suppressing Hallucinations In LVLM Encoders via Bias and Vulnerability Defense](../../ICLR2026/hallucination/shield_suppressing_hallucinations_in_lvlm_encoders_via_bias_and_vulnerability_de.md)
- [\[CVPR 2026\] Beyond the Global Scores: Fine-Grained Token Grounding as a Robust Detector of LVLM Hallucinations](beyond_global_scores_fine_grained_token_grounding_as_robust_detector_of_lvlm_hallucinations.md)

</div>

<!-- RELATED:END -->

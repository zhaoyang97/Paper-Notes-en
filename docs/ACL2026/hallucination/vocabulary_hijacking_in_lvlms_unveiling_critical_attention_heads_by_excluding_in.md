---
title: >-
  [Paper Note] Vocabulary Hijacking in LVLMs: Unveiling Critical Attention Heads by Excluding Inert Tokens to Mitigate Hallucination
description: >-
  [ACL2026][Hallucination Detection][LVLM hallucination] This paper discovers that certain invalid visual tokens in LVLMs consistently decode into a set of irrelevant words and hijack attention. Consequently…
tags:
  - "ACL2026"
  - "Hallucination Detection"
  - "LVLM hallucination"
  - "attention head interpretation"
  - "Logit Lens"
  - "Vocabulary Hijacking"
  - "training-free intervention"
date: 2026-05-08
content_hash: 4e43f15930083ba2
---

# Vocabulary Hijacking in LVLMs: Unveiling Critical Attention Heads by Excluding Inert Tokens to Mitigate Hallucination

**Conference**: ACL2026  
**arXiv**: [2605.10622](https://arxiv.org/abs/2605.10622)  
**Code**: https://github.com/lab-klc/HAVAE  
**Area**: Hallucination Detection  
**Keywords**: LVLM hallucination, attention head interpretation, Logit Lens, Vocabulary Hijacking, training-free intervention

## TL;DR
This paper discovers that certain invalid visual tokens in LVLMs consistently decode into a set of irrelevant words and hijack attention. Consequently, it proposes HABI to locate these tokens, uses NHAR to identify reliable visual heads, and then enhances these heads via HAVAE during inference to reduce hallucinations.

## Background & Motivation
**Background**: Methods for mitigating hallucinations in Large Vision-Language Models (LVLMs) often center on "encouraging the model to look at the image more," such as intervening on visual attention, utilizing contrastive decoding, applying activation steering, or enhancing the influence of image tokens during generation. Recent analyses have indicated that hallucinations are linked to insufficient or abnormal attention directed toward visual tokens.

**Limitations of Prior Work**: The core issue is not "whether attention should be intervened upon," but rather "which attention heads and visual tokens should be targeted." Simply increasing total visual attention often drives focus toward backgrounds, redundant patches, or attention sinks. Furthermore, selecting heads based on heuristics makes it difficult to explain why specific heads relate to factual grounding.

**Key Challenge**: Visual attention in LVLMs is not inherently equivalent to effective visual evidence. Some tokens receive substantial attention but carry almost no information about the target object, instead steering the generation toward fixed, meaningless lexical anchors. Existing methods lack a mechanistic diagnosis and may therefore amplify both useful and noisy attention simultaneously.

**Goal**: The authors seek to answer three questions: What are the internal representation patterns of abnormal visual attention; how do these abnormal tokens relate to hallucinations; and can truly reliable visual attention heads be selected and enhanced during the inference phase without requiring additional training.

**Key Insight**: The paper utilizes Logit Lens to observe what "words" visual token hidden states resemble when projected into the vocabulary space across different layers. The authors discovered that the cross-layer traces of certain high-attention visual tokens repeatedly fall on fixed, irrelevant words. These are not typical background tokens but represent a form of semantic collapse-driven vocabulary hijacking.

**Core Idea**: First, identify "Inert Tokens" hijacked by fixed lexical anchors, and then exclude these tokens to find the critical attention heads that are truly oriented toward valid visual content.

## Method

### Overall Architecture
The methodology of the paper is divided into two phases: "diagnosis" and "intervention." In the diagnosis phase, descriptions are generated for 500 images from the COCO 2014 validation set using models such as LLaVA-1.5, Shikra, MiniGPT-4, and Qwen2-VL, with COCO annotations used to distinguish between real and hallucinated objects. The authors then use Logit Lens to trace the words into which visual tokens are decoded across layers, defining Vocabulary Hijacking, Hijacking Anchors, and Inert Tokens.

Building on this, the paper constructs two attention metrics. HAR measures the proportion of attention from critical visual heads that falls on Inert Tokens, proving that hijacking is positively correlated with hallucination. Conversely, NHAR counts attention only on non-Inert visual tokens to identify more reliable factual grounding heads.

In the intervention phase, HAVAE is proposed. It does not update model parameters or introduce extra models; instead, it enhances the visual-oriented attention of the top-$K$ heads ranked by NHAR during inference. The objective is not to blindly increase all visual attention but to bolster heads already diagnosed as "attending to non-hijacked visual content."

### Key Designs
1. **HABI: Locating Inert Tokens via Lexical Anchors**:

	- **Function**: Identifies Inert Tokens that receive high attention but are semantically invalid and prone to hijacking the generation process.
	- **Mechanism**: For each visual token $v_i$, its hidden states across layers are projected to the vocabulary using Logit Lens to obtain a cross-layer word sequence (Trace). If a token's Trace is repeatedly dominated by a fixed Anchor, and this Anchor frequently appears globally among high-attention tokens, it is assigned a high hijacking score. The authors multiply three dimensions—Dominance, Frequency, and Attention—to form $S_{hijack}(v_i)$, and then identify Hijacking Anchors using IQR outlier thresholds at the vocabulary level.
	- **Design Motivation**: While ordinary attention sinks merely indicate that "certain tokens absorb attention," they do not explain the internal representations of those tokens. HABI links abnormal attention to semantic collapse in the vocabulary space, providing a more specific diagnostic than simply identifying background tokens by attention magnitude.

2. **HAR and NHAR: Separating Abnormal Attention from Critical Head Selection**:

	- **Function**: HAR is utilized to prove that hijacking leads to hallucinations, while NHAR is used to select truly beneficial visual attention heads.
	- **Mechanism**: HAR calculates the ratio of attention directed at Inert Tokens relative to all visual attention; experiments show that hallucinated tokens often correspond to higher HAR. NHAR sums only the attention falling on non-Inert visual tokens, effectively removing the hijacked portion from the visual attention budget to retain only the density directed at valid visual content.
	- **Design Motivation**: High visual attention can be either a negative or positive signal; the key is where it is directed. The value of NHAR lies in shifting the focus from "looking at the image a lot" to "looking at effective image regions a lot," providing an interpretable selection criterion for subsequent inference-time enhancement.

3. **HAVAE: Training-Free Attention Enhancement**:

	- **Function**: Reduces LVLM hallucinations while maintaining general capabilities without fine-tuning.
	- **Mechanism**: The top-$K$ target heads $H_{target}$ are selected based on the average NHAR on real object tokens. During inference, an intra-layer mean attention magnitude term is added to the visual attention of these target heads, with the enhancement intensity controlled by $\alpha$. In the paper, Qwen2-VL uses $K=300$, while other models mostly use $K=450$; for long-text scenarios, $\alpha$ is increased from the default 0.1 to 0.6 or 0.7.
	- **Design Motivation**: Directly penalizing Inert Tokens can disrupt generation because these tokens may perform residual routing or placeholder functions. HAVAE chooses to positively enhance reliable heads rather than negatively suppressing abnormal tokens, which proves more stable in experiments.

### Loss & Training
This work involves no training loss, as HAVAE is a training-free inference intervention. The required offline steps involve using a small number of images to calculate statistics for Hijacking Anchors, Inert Tokens, and NHAR rankings; the inference phase only modifies the attention weights of selected heads. This design allows it to be used in scenarios where closed-source weights are non-trainable, though it still requires access to the model's internal attention.

## Key Experimental Results

### Main Results
The main experiments evaluate hallucination and general capability on benchmarks such as CHAIR, POPE, POPE-Chat, AMBER, and MME, covering LLaVA-1.5 7B/13B, MiniGPT-4 7B, Shikra 7B, and Qwen2-VL 7B.

| Model | Method | CHAIRs ↓ | CHAIRi ↓ | POPE Acc ↑ | POPE F1 ↑ | POPE-Chat Acc ↑ | POPE-Chat F1 ↑ | Key Findings |
|------|------|----------|----------|------------|-----------|-----------------|----------------|----------|
| LLaVA-1.5-7B | Greedy | 48.2 | 14.2 | 84.8 | 85.5 | 85.5 | 83.4 | Significant hallucination in the original model |
| LLaVA-1.5-7B | PAI | 23.8 | 6.2 | 85.9 | 86.0 | 85.5 | 83.4 | Attention intervention is effective but not optimal |
| LLaVA-1.5-7B | HAVAE | 18.2 | 3.8 | 86.2 | 86.3 | 88.0 | 87.0 | CHAIRi dropped by 38.7% compared to the strongest baseline |
| MiniGPT-4-7B | HAVAE | 21.8 | 6.9 | 76.9 | 77.6 | 80.2 | 80.2 | Improvement still observed on small models |
| Shikra-7B | HAVAE | 15.8 | 5.0 | 81.6 | 82.1 | 76.7 | 78.6 | CHAIRi dropped by 46.2% compared to the strongest baseline |
| LLaVA-1.5-13B | HAVAE | 21.8 | 5.0 | 82.5 | 84.7 | 87.9 | 86.6 | Scalable to 13B scale |

### Ablation Study
The ablation focus is on proving that heads cannot be selected by total visual attention alone; Inert Tokens must be excluded. Furthermore, directly penalizing Inert Tokens is inferior to positive enhancement.

| Configuration | CHAIRs ↓ | CHAIRi ↓ | POPE Acc ↑ | POPE F1 ↑ | MME Per ↑ | MME Cog ↑ | Description |
|------|----------|----------|------------|-----------|-----------|-----------|------|
| Max Attention Head Selection | 7.8 | 4.4 | 85.9 | 85.6 | 1399.0 | 277.0 | Low hallucination metrics but F1 and MME are significantly damaged, indicating high-attention heads are not necessarily reliable |
| HAVAE / NHAR Selection | 18.2 | 3.8 | 86.2 | 86.3 | 1483.9 | 327.9 | Better balance between hallucination suppression and general capability |
| Sample size 10 | 18.8 | 3.7 | 86.1 | 86.2 | N/A | N/A | Stable estimation with very few samples |
| Sample size 500 | 18.2 | 3.7 | 86.1 | 86.2 | N/A | N/A | Metric stability; 500 adopted for the paper |
| Penalty coefficient $\beta=0.0$ | 18.2 | 3.7 | 86.1 | 86.2 | N/A | N/A | Standard HAVAE |
| Penalty coefficient $\beta=0.6$ | 19.8 | 4.7 | 86.1 | 86.2 | N/A | N/A | Directly penalizing Inert Tokens actually worsens CHAIR |

### Key Findings
- Vocabulary Hijacking is not an isolated anomaly unique to one model. The authors observed a long-tail distribution of hijacking scores and a bimodal distribution of hijacking ratios for salient tokens across LLaVA-1.5, MiniGPT-4, Shikra, and Qwen2-VL.
- Hallucinated tokens exhibit significantly higher HAR, whereas real object tokens are concentrated in high NHAR regions, suggesting that "hijacked visual attention" and "reliable visual grounding" are statistically distinguishable.
- HAVAE does not damage general capability on MME: for instance, LLaVA-1.5-7B perception improved from 1472.5 to 1483.9, and cognition from 322.5 to 327.9; Shikra cognition improved from 250.4 to 272.5.
- Similar gains were found on Qwen2-VL: CHAIRs decreased from 27.6 to 22.8, CHAIRi from 8.8 to 6.2, and MME All increased from 2268.4 to 2290.2.
- Threshold sensitivity is low. Perturbing $\tau_r$ and $\tau_s$ within a range of $0.8 \times$ to $1.2 \times$ resulted in minor changes to CHAIR and POPE metrics, indicating HABI does not rely on a narrow hyperparameter window.

## Highlights & Insights
- The most compelling aspect of the paper is tracing hallucinations from output errors back to fixed anchors in the vocabulary space. Instead of vaguely stating that "attention is wrong," it provides an internal mechanistic chain: visual token traces collapse into Hijacking Anchors, sucking attention away from heads, leading to decreased grounding in critical heads, and finally resulting in hallucinated objects.
- The design of HABI is highly interpretable. Dominance checks cross-layer rigidity for individual tokens, Frequency checks for systematic word appearances, and Attention checks the actual influence on generation; multiplying the three filters out significant noise.
- NHAR is a superior head selection criterion compared to "total visual attention." This offers an insight for multimodal interpretability: when interpreting attention, one should not only look at image token weights but first determine if the image tokens themselves possess semantic contribution.
- The positive enhancement strategy of HAVAE is robust. The penalty ablation suggests that abnormal tokens cannot simply be zeroed out; enhancing reliable pathways is more consistent with the deep routing structure of models than blunt suppression of abnormal paths.
- This work informs future mechanistic interpretability: Logit Lens, attention flow, and behavioral errors can be linked together rather than relying solely on static visualization.

## Limitations & Future Work
- The method requires access to internal hidden states, unembedding, and attention weights, making it unsuitable for closed-source LVLMs that can only be accessed via black-box APIs.
- The origin of the mechanism is not fully explained. The authors speculate that Vocabulary Hijacking might stem from shortcuts in early vision-language alignment, but this has not been verified through training process tracing or controlled pre-training experiments.
- The validated models go up to 13B, with Qwen2-VL at 7B; whether hijacking anchors exist in larger-scale models, newer architectures, or video LVLMs still requires systematic checking.
- HABI relies on COCO images and object labels to construct real/hallucinated object sets. While AMBER shows some out-of-domain generalization, the distribution of Inert Tokens might differ in domains like medical imaging, remote sensing, or document images.
- HAVAE involves inference-time attention modification; its compatibility with KV caching, efficient inference frameworks, and quantized models still requires engineering validation.

## Related Work & Insights
- **vs Visual Attention Sink**: While VAS focus on empty or background tokens monopolizing attention, this paper further points out that the hidden states of these tokens stably decode into fixed irrelevant words, providing a finer-grained vocabulary-space mechanism.
- **vs PAI / Devils**: These are also training-free attention interventions but typically rely on coarser visual attention heuristics. The difference in HAVAE is the prior exclusion of Inert Tokens before selecting critical heads via NHAR, reducing the risk of enhancing noisy pathways.
- **vs VISTA / activation steering**: VISTA influences generation through activation directions, which can reduce hallucinations but may affect general capabilities. HAVAE only enhances visual attention in selected specific heads, making the intervention more localized with a clearer mechanistic interpretation.
- **vs Logit Lens analysis**: Previously, Logit Lens was often used to observe representation evolution from vision to semantics; this paper uses it to locate abnormal traces and converts analytical findings into a functional inference intervention.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The characterization of Vocabulary Hijacking and Hijacking Anchors is quite novel and translates well into effective interventions.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers multiple models, benchmarks, and ablations, though there is room for expansion into larger models and non-COCO domains.
- Writing Quality: ⭐⭐⭐⭐☆ The chain from diagnosis to intervention is clear, with sufficient supporting data despite the large number of tables.
- Value: ⭐⭐⭐⭐⭐ Highly insightful for both LVLM hallucination explanation and training-free repair, especially suitable for reuse in subsequent interpretability research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Mitigating Object Hallucination in LVLMs via Attention Imbalance Rectification](../../CVPR2026/hallucination/mitigating_object_hallucinations_in_lvlms_via_attention_imbalance_rectification.md)
- [\[AAAI 2026\] Causally-Grounded Dual-Path Attention Intervention for Object Hallucination Mitigation in LVLMs](../../AAAI2026/hallucination/causally-grounded_dual-path_attention_intervention_for_objec.md)
- [\[ACL 2026\] Aligning with Your Own Voice: Self-Corrected Preference Learning for Hallucination Mitigation in LVLMs](aligning_with_your_own_voice_self-corrected_preference_learning_for_hallucinatio.md)
- [\[ICML 2026\] Finding the Correct Visual Evidence Without Forgetting: Mitigating Hallucination in LVLMs via Inter-Layer Visual Attention Discrepancy](../../ICML2026/hallucination/finding_the_correct_visual_evidence_without_forgetting_mitigating_hallucination_.md)
- [\[ACL 2026\] Hallucination Detection in LLMs with Topological Divergence on Attention Graphs](hallucination_detection_in_llms_with_topological_divergence_on_attention_graphs.md)

</div>

<!-- RELATED:END -->

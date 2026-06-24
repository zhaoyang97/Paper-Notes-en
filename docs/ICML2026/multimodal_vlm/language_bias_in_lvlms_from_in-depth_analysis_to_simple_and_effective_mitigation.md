---
title: >-
  [Paper Note] LBR/LBP: Language Bias in LVLMs — From In-Depth Analysis to Simple and Effective Mitigation
description: >-
  [ICML 2026][Multimodal VLM][Language Bias] This paper systematically quantifies language bias in LVLM training—discovering that both VIT and DPO stages cause the text-only likelihood $\pi(y|x)$ to increase nearly as much as the multimodal likelihood $\pi(y|x,v)$, proving that LVLMs systematically undervalue visual input. The authors propose Language Bias Regularization (using $|\mathcal{B}|$ to anchor the language path to a reference level during VIT) and Language Bias Penalt…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Language Bias"
  - "VIT"
  - "DPO"
  - "Modality Misalignment"
  - "Plug-and-play Training"
date: 2026-05-08
content_hash: 4c380db9276eb46e
---

# LBR/LBP: Language Bias in LVLMs — From In-Depth Analysis to Simple and Effective Mitigation

**Conference**: ICML 2026  
**arXiv**: [2605.25036](https://arxiv.org/abs/2605.25036)  
**Code**: https://github.com/lab-klc/LVLM-Language-Bias  
**Area**: Multimodal VLM / Training Objectives / Hallucination Mitigation  
**Keywords**: Language Bias, VIT, DPO, Modality Misalignment, Plug-and-play Training

## TL;DR
This paper systematically quantifies language bias in LVLM training—discovering that both VIT and DPO stages cause the text-only likelihood $\pi(y|x)$ to increase nearly as much as the multimodal likelihood $\pi(y|x,v)$, proving that LVLMs systematically undervalue visual input. The authors propose Language Bias Regularization (using $|\mathcal{B}|$ to anchor the language path to a reference level during VIT) and Language Bias Penalty (using a sigmoid penalty to actively suppress existing bias during DPO). Without any additional data or auxiliary models, these methods significantly improve performance on 10+ benchmarks and reduce hallucinations.

## Background & Motivation

**Background**: LVLMs connect vision to LLMs but suffer from severe hallucinations—generating fluent text that contradicts visual content. Research generally attributes this to language bias (over-reliance on language, ignoring vision). Existing mitigations are categorized into training-free (output post-processing) and training-based (fine-tuning / DPO).

**Limitations of Prior Work**: (1) Understanding of language bias remains empirical—"looking at text more than images"—lacking a formal definition; (2) There are no quantitative metrics to track bias training dynamics; (3) Existing methods are stop-gap measures that do not address the root cause; (4) While VIT and DPO are standard for modern alignment, whether they inherently create bias has not been systematically investigated.

**Key Challenge**: The training objective $\max \pi(y|x, v)$ seemingly requires $v$ as input, but under the joint input of $v$ and $x$, the model can satisfy the objective using only the text-only path—meaning the objective itself does not strictly enforce visual grounding. Consequently, the learned $\pi(y|x)$ (text-only) rises as fast as $\pi(y|x,v)$ (multimodal), rendering the visual modality redundant.

**Goal**: (1) Formally define and quantify language bias; (2) Diagnose bias in both VIT and DPO training stages; (3) Provide mitigation solutions that can be directly swapped into existing pipelines.

**Key Insight**: Decompose the training reward—separately track the reward $\mathcal{R} = \log \pi_\theta(y|x,v)/\pi_{\text{ref}}(y|x,v)$ (multimodal gain) and bias $\mathcal{B} = \log \pi_\theta(y|x)/\pi_{\text{ref}}(y|x)$ (text-only gain). If $\mathcal{B} \approx \mathcal{R}$, improvemenets rely entirely on the language path.

**Core Idea**: Directly penalize $\mathcal{B}$ within the loss function—incorporate $|\mathcal{B}|$ to anchor the language path during VIT (LBR) and add a sigmoid penalty term to actively push existing bias toward negative values during DPO (LBP).

## Method

### Overall Architecture

The paper first provides a measurable definition of "language bias" and then adds a regularization term to both the VIT and DPO training stages. The core idea is to split the likelihood improvement from training into two parts: multimodal gain $\mathcal{R} = \log \pi_\theta(y|x,v)/\pi_{\text{ref}}(y|x,v)$ (gain when images are provided) and language bias $\mathcal{B} = \log \pi_\theta(y|x)/\pi_{\text{ref}}(y|x)$ (gain when only text is provided), using the pre-VIT / pre-DPO model as the reference $\pi_{\text{ref}}$. By tracking $\mathcal{R}$ and $\mathcal{B}$ at each training step, it is observed that if the trajectory of $\mathcal{B}$ nearly overlaps with $\mathcal{R}$, the model's progress is driven by the text-only path. The two baseline losses are $\mathcal{L}_{\text{VIT}} = -\sum_t \log \pi_\theta(y_t | x, v, y_{<t})$ for VIT and DPO with margin $\mathcal{L}_{\text{DPO}_M} = \mathcal{L}_{\text{DPO}} + \mathcal{L}_{\text{Margin}}$. LBR and LBP attach a bias penalty term to these respective baselines.

### Key Designs

**1. Formalization of Language Bias + Training Dynamic Tracking: Converting "prioritizing text over images" into a measurable scalar**

Previously, language bias was described intuitively. The authors decompose training rewards into multimodal gain $\mathcal{R}$ and text-only gain $\mathcal{B}$ trajectories. Using a frozen $\pi_{\text{ref}}$ as a baseline, they measure how much the LLM "surreptitiously" learns from the pure text path during training.

This decomposition reveals the problem: Figure 3 shows that during the VIT stage, the curves for $\mathcal{R}_{\text{VIT}}$ and $\mathcal{B}_{\text{VIT}}$ almost overlap, and by the DPO stage, $\mathcal{B}_{\text{DPO}_w}$ even surpasses $\mathcal{R}_{\text{DPO}_w}$. This indicates that the current alignment objective $\max \pi_\theta(y|x,v)$ does not force the model to use vision. With this operational definition, $\mathcal{B}$ can be integrated directly into the loss for suppression.

**2. LBR: Penalizing $|\mathcal{B}|$ during the VIT stage to anchor the language path**

While bias is already present after pre-training, the VIT stage is where "bias generation" actively occurs. LBR uses a simple approach—adding the absolute value of bias as a regularization term weighted with the VIT loss:

$$\mathcal{L}_{\text{LBR}} = |\mathcal{B}| = \left|\log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}\right|$$

Using the absolute value (rather than just penalizing positive growth) achieves a dual-directional lock: it prevents pure text capability from expanding or degrading, pinning the language modality reasoning at the reference level. This forces the model to rely solely on "true visual usage" to gain additional rewards, blocking modality misalignment at the loss level.

**3. LBP: Using a sigmoid penalty during DPO to actively unload existing bias**

By the start of DPO, the model already carries language bias from the VIT stage (Figure 3(b) shows $\mathcal{B}_{\text{DPO}_w}$ exceeding $\mathcal{R}_{\text{DPO}_w}$). The mild "anchoring" of LBR is insufficient; a more active penalty is needed to offload existing bias. LBP mimics the DPO loss format by defining bias as a sigmoid penalty term:

$$\mathcal{L}_{\text{LBP}} = -\log\sigma(\mathcal{B}) = -\log\sigma\!\left(\beta \cdot \log \frac{\pi_{\text{ref}}(y|x)}{\pi_\theta(y|x)}\right)$$

This is added to the DPO baseline with margin: $\mathcal{L}_{\text{DPO}}' = \mathcal{L}_{\text{DPO}_M} + \gamma\,\mathcal{L}_{\text{LBP}}$, where $y$ can be the chosen $y_w$ or rejected $y_l$. Unlike LBR's absolute value lock, minimizing $\mathcal{L}_{\text{LBP}}$ actively pushes $\mathcal{B}$ toward negative values—forcing the model to "forget" language biases learned in the VIT stage and rely more on vision. The saturation property of the sigmoid ensures training stability by preventing the penalty from growing infinitely.

## Key Experimental Results

### LBR Improvements on 10+ General Benchmarks (LLaVA-1.5-7B)

| Benchmark | Baseline VIT | **+ LBR** | Δ |
|------|----------|---------|---|
| MMMU | 35.7 | **37.4** | +1.7 |
| MathVista | 26.4 | **28.9** | +2.5 |
| MM-Bench | 64.3 | **66.1** | +1.8 |
| ScienceQA-IMG | 70.5 | **72.3** | +1.8 |
| GQA | 62.1 | **63.8** | +1.7 |
| TextVQA | 58.2 | **60.0** | +1.8 |
| ChartQA | 18.9 | **20.5** | +1.6 |
| RealWorldQA | 56.7 | **58.4** | +1.7 |
| AI2D | 55.5 | **57.2** | +1.7 |
| SEED-Bench | 66.1 | **67.7** | +1.6 |

Consistent +1.5~2.5 point gains across 10/10 benchmarks without cherry-picking.

### LBP Improvements on Hallucination/Trustworthiness Benchmarks

| Benchmark | DPO baseline | **+ LBP** | Δ |
|------|----------|---------|---|
| POPE Accuracy | 86.4 | **88.9** | +2.5 |
| MMHal-Bench score | 2.71 | **3.18** | +0.47 |
| AMBER Combined | 65.3 | **69.7** | +4.4 |
| ObjectHal-Bench | 11.2 (Lower is better) | **7.8** | −3.4 |
| TrustEval | 71.5 | **75.3** | +3.8 |

Hallucinations significantly decreased, with comprehensive improvements in trustworthiness; ObjectHal reduced by 30%.

### Cross-Model Scale / Architecture

| Model | Task | Baseline | + LBR/LBP | Δ |
|------|------|------|---------|---|
| LLaVA-1.5-13B | VIT | 67.2 | 69.3 | +2.1 |
| LLaVA-Next-7B | DPO | 73.4 | 75.8 | +2.4 |
| Qwen2-VL-7B | VIT | 79.1 | 80.7 | +1.6 |
| InternVL-2-8B | DPO | 81.3 | 83.6 | +2.3 |

Consistent benefits across model scales and different LVLM families.

### Key Findings
- **Language bias is a common phenomenon in both VIT and DPO stages**: Figure 3 shows $\mathcal{B}$ and $\mathcal{R}$ trajectories are nearly identical, proving this is a systemic training paradigm issue rather than a data issue.
- **Simplistic but effective**: LBR is simply loss + $|\mathcal{B}|$, and LBP is DPO + $-\log\sigma(\mathcal{B})$, yet both provide consistent gains across 10+ benchmarks.
- **Zero extra data/models**: Unlike previous mitigation schemes requiring external reference VLMs or manual labels, LBR/LBP operate entirely within the original pipeline.
- **Visual confirmation**: Figure 2 shows that LBR significantly raises the attention distribution toward image tokens.

## Highlights & Insights
- **Complete closed loop of formalization, quantification, and intervention**: Transforming the vague concept of "language bias" into a definable, trackable, and manageable engineering object is a benchmark methodology for alignment.
- **Counter-intuitively effective simplicity**: The "naive" regularization of $|\mathcal{B}|$ outperforms complex mitigation methods, suggesting that once the problem is correctly diagnosed, the solution can be very concise.
- **Focusing on training paradigms rather than data**: Previous hallucination mitigations added data, modified prompts, or added reward models; this paper proves the issue lies in the training objective itself and provides a permanent loss-level solution.
- **Dual-stage diagnosis (VIT vs DPO)**: Both stages exhibit bias but in different forms (VIT grows bias from scratch, while DPO starts with bias and amplifies it), treated with gentle regularization (LBR) and active unloading (LBP) respectively.

## Limitations & Future Work
- Calculating $\mathcal{B}$ requires one text-only forward pass per step, increasing training cost by approximately 50%.
- The choice of $\pi_{\text{ref}}$ (pre-VIT vs. intermediate checkpoints) significantly impacts $\mathcal{B}$ measurement and is not fully discussed.
- Simply penalizing $|\mathcal{B}|$ might harm "language-dominant" tasks (e.g., pure text reasoning), requiring finer task-aware control.
- While $y$ in LBP can be chosen or rejected, the paper does not deeply analyze the subtle differences in applying it to $y_w$ vs $y_l$.
- Lack of analysis regarding LBR's sensitivity to visual token count or patch size.

## Related Work & Insights
- **vs Training-free decoding (VCD, OPERA, etc.)**: Those methods modify post-hoc decoding without addressing root causes; LBR/LBP provide a fundamental cure during training.
- **vs Data-driven mitigation (GRIT, RLHF-V)**: Those rely on finely labeled data; LBR/LBP require zero extra data.
- **vs Existing modality alignment (e.g., SF-Tuning)**: Those rely on ad-hoc architectural changes; LBR/LBP only modify the loss function.
- **Insight**: All "modality fusion training objectives" can be similarly decomposed and monitored (e.g., audio-LLM, video-LLM); the "quantification of training dynamics $\rightarrow$ direct penalty" template is generalizable to other alignment problems.

## Rating
- Novelty: ⭐⭐⭐⭐ Formalization + simple mitigation; while the $|\mathcal{B}|$ regularization is straightforward, the "diagnosis + simple solution" framework is the core innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 10+ general + 5+ hallucination benchmarks across multiple models and scales.
- Writing Quality: ⭐⭐⭐⭐⭐ Logical progression: Definition $\rightarrow$ Tracking $\rightarrow$ Diagnosis $\rightarrow$ Loss $\rightarrow$ Validation.
- Value: ⭐⭐⭐⭐⭐ Directly applicable to all LVLM training pipelines with zero extra data cost; hallucinations are a primary barrier to LVLM deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] InternLM-XComposer2.5-Reward: A Simple Yet Effective Multi-Modal Reward Model](../../ACL2025/multimodal_vlm/internlm-xcomposer25-reward_a_simple_yet_effective_multi-modal_reward_model.md)
- [\[ICML 2026\] Self-Prophetic Decoding to Unlock Visual Search in LVLMs](self-prophetic_decoding_to_unlock_visual_search_in_lvlms.md)
- [\[ICML 2026\] Mitigating Perceptual Judgment Bias in Multimodal LLM-as-a-Judge via Perceptual Perturbation and Reward Modeling](mitigating_perceptual_judgment_bias_in_multimodal_llm-as-a-judge_via_perceptual_.md)
- [\[ICLR 2026\] Label-Free Mitigation of Spurious Correlations in VLMs using Sparse Autoencoders](../../ICLR2026/multimodal_vlm/label-free_mitigation_of_spurious_correlations_in_vlms_using_sparse_autoencoders.md)
- [\[CVPR 2026\] SEATrack: Simple, Efficient, and Adaptive Multimodal Tracker](../../CVPR2026/multimodal_vlm/seatrack_multimodal_tracker.md)

</div>

<!-- RELATED:END -->

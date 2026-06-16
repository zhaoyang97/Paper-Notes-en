---
title: >-
  [Paper Note] LBR/LBP: Language Bias in LVLMs — From In-Depth Analysis to Simple and Effective Mitigation
description: >-
  [ICML 2026][Multimodal VLM][VIT] This paper systematically quantifies language bias in LVLM training. It finds that during both the Visual Instruction Tuning (VIT) and Direct Preference Optimization (DPO) stages, the text-only likelihood $\pi(y|x)$ increases at a rate comparable to the multimodal likelihood $\pi(y|x,v)$, proving that LVLMs systematica
tags:
  - ICML 2026
  - Multimodal VLM
  - VIT
  - DPO
date: 2026-05-08
content_hash: dafec7a779dd8647
---
# LBR/LBP: Language Bias in LVLMs — From In-Depth Analysis to Simple and Effective Mitigation

**Conference**: ICML 2026  
**arXiv**: [2605.25036](https://arxiv.org/abs/2605.25036)  
**Code**: https://github.com/lab-klc/LVLM-Language-Bias  
**Area**: Multimodal VLM / Training Objectives / Hallucination Mitigation  
**Keywords**: Language Bias, VIT, DPO, Modality Misalignment, Training Plug-and-Play

## TL;DR
This paper systematically quantifies language bias in LVLM training. It finds that during both the Visual Instruction Tuning (VIT) and Direct Preference Optimization (DPO) stages, the text-only likelihood $\pi(y|x)$ increases at a rate comparable to the multimodal likelihood $\pi(y|x,v)$, proving that LVLMs systematically underestimate visual input. The authors propose Language Bias Regularization (LBR), which utilizes $|\mathcal{B}|$ to anchor the language path to a reference level during VIT, and Language Bias Penalty (LBP), which uses a sigmoid penalty to actively suppress existing bias during DPO. Without additional data or auxiliary models, these methods significantly improve performance on 10+ benchmarks and reduce hallucinations.

## Background & Motivation

**Background**: LVLMs integrate visual capabilities into LLMs but suffer from severe hallucinations—generating fluent text that contradicts visual input. Research generally attributes this to language bias (the model relying excessively on language while ignoring vision). Existing mitigation methods are divided into training-free (output post-processing) and training-based (fine-tuning / DPO).

**Limitations of Prior Work**: (1) Understanding of language bias remains empirical—"looking at text more than images"—lacking a formal definition; (2) There are no quantitative metrics to track the training dynamics of bias; (3) Existing methods are stop-gap measures that do not address the root cause; (4) Both VIT and DPO are standard components of modern alignment, yet no systematic study has checked if they inherently create bias.

**Key Challenge**: The training objective $\max \pi(y|x, v)$ seemingly requires $v$, but given $v$ and $x$ simultaneously, the model can satisfy the objective through a text-only path. This implies the objective itself does not strictly enforce visual grounding. Consequently, the learned $\pi(y|x)$ (text-only) increases as rapidly as $\pi(y|x,v)$ (multimodal), rendering the visual modality redundant.

**Goal**: (1) Formally define and quantify language bias; (2) Diagnose bias during both the VIT and DPO training stages; (3) Provide mitigation solutions that can be directly integrated into existing pipelines.

**Key Insight**: Decompose the training reward by separately tracking the reward $\mathcal{R} = \log \pi_\theta(y|x,v)/\pi_{\text{ref}}(y|x,v)$ (multimodal gain) and the bias $\mathcal{B} = \log \pi_\theta(y|x)/\pi_{\text{ref}}(y|x)$ (text-only gain). If $\mathcal{B} \approx \mathcal{R}$, the improvement is purely driven by the language path.

**Core Idea**: Once $\mathcal{B}$ is defined, it is directly penalized in the loss. LBR adds $|\mathcal{B}|$ to anchor the language path during VIT, while LBP adds a sigmoid penalty term to actively push existing bias toward negative values during DPO.

## Method

### Overall Architecture

The authors first provide a measurable definition of "language bias" and then introduce regularization terms for the VIT and DPO stages accordingly. The core mechanism involves splitting the likelihood increase from training into two parts: multimodal gain $\mathcal{R} = \log \pi_\theta(y|x,v)/\pi_{\text{ref}}(y|x,v)$ (gain with image input) and language bias $\mathcal{B} = \log \pi_\theta(y|x)/\pi_{\text{ref}}(y|x)$ (gain with only text input), using the pre-VIT / pre-DPO model as the reference $\pi_{\text{ref}}$. By tracking $\mathcal{R}$ and $\mathcal{B}$ at each training step, it becomes evident if the model's progress relies solely on the text-only path. The two baseline losses are $\mathcal{L}_{\text{VIT}} = -\sum_t \log \pi_\theta(y_t | x, v, y_{<t})$ for VIT and DPO with margin $\mathcal{L}_{\text{DPO}_M} = \mathcal{L}_{\text{DPO}} + \mathcal{L}_{\text{Margin}}$. LBR and LBP serve as bias penalty terms appended to these baselines.

### Key Designs

**1. Formalization of Language Bias + Tracking Training Dynamics: Converting "reliance on text" into a measurable scalar**

Previously, language bias was described qualitatively. This paper decomposes training rewards into two trajectories: multimodal gain $\mathcal{R}$ and text-only gain $\mathcal{B}$. By using a frozen $\pi_{\text{ref}}$ as a baseline, the authors measure how much the LLM "shortcuts" learning through the pure text path during training.

This decomposition reveals a critical issue: Figure 3 shows that $\mathcal{R}_{\text{VIT}}$ and $\mathcal{B}_{\text{VIT}}$ curves almost overlap during VIT, and during DPO, $\mathcal{B}_{\text{DPO}_w}$ even surpasses $\mathcal{R}_{\text{DPO}_w}$. This indicates the alignment objective $\max \pi_\theta(y|x,v)$ does not force the model to use vision. This operational definition allows $\mathcal{B}$ to be directly incorporated into the loss for suppression.

**2. LBR: Penalizing $|\mathcal{B}|$ during VIT to anchor the language path to reference levels**

While $\mathcal{B}$ is small after pre-training, the VIT stage is where bias is primarily "manufactured." LBR uses the absolute value of the bias as a regularization term weighted with the VIT loss:

$$\mathcal{L}_{\text{LBR}} = |\mathcal{B}| = \left|\log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}\right|$$

Using the absolute value (rather than just penalizing positive growth) ensures bi-directional anchoring: it prevents pure text capability from either inflating or degrading, pinning the language modality reasoning at the reference level. Consequently, the model must "truly use vision" to obtain additional rewards, closing the modality misalignment gap at the loss level.

**3. LBP: Sigmoid penalty during DPO to actively unload existing bias**

At the start of DPO, the model already carries language bias from the VIT stage (Figure 3(b) shows $\mathcal{B}_{\text{DPO}_w} > \mathcal{R}_{\text{DPO}_w}$). The mild anchoring of LBR is no longer sufficient; a more proactive penalty is needed. LBP adopts a sigmoid penalty term following the DPO loss structure:

$$\mathcal{L}_{\text{LBP}} = -\log \sigma(\mathcal{B}) = -\log \sigma\!\left(\beta \cdot \log \frac{\pi_{\text{ref}}(y|x)}{\pi_\theta(y|x)}\right)$$

This is added to the DPO baseline with margin: $\mathcal{L}_{\text{DPO}}' = \mathcal{L}_{\text{DPO}_M} + \gamma\,\mathcal{L}_{\text{LBP}}$, where $y$ can be the chosen $y_w$ or rejected $y_l$. Unlike LBR's bi-directional anchoring, minimizing $\mathcal{L}_{\text{LBP}}$ actively pushes $\mathcal{B}$ toward negative values—forcing the model to "forget" language bias from VIT and rely more on vision. The saturation of the sigmoid function ensures training stability.

## Key Experimental Results

### Performance Gain of LBR on 10+ General Benchmarks (LLaVA-1.5-7B)

| Benchmark | Baseline VIT | **+ LBR** | Gain ($\Delta$) |
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

Consistent gains of +1.5 to 2.5 points across all 10 benchmarks without cherry-picking.

### Performance Gain of LBP on Hallucination/Trustworthiness Benchmarks

| Benchmark | DPO baseline | **+ LBP** | Gain ($\Delta$) |
|------|----------|---------|---|
| POPE Accuracy | 86.4 | **88.9** | +2.5 |
| MMHal-Bench score | 2.71 | **3.18** | +0.47 |
| AMBER Overall | 65.3 | **69.7** | +4.4 |
| ObjectHal-Bench | 11.2 (Lower is better) | **7.8** | −3.4 |
| TrustEval | 71.5 | **75.3** | +3.8 |

Significant reduction in hallucinations and comprehensive improvement in trustworthiness; ObjectHal decreased by 30%.

### Cross Model Scales / Architectures

| Model | Task | Baseline | + LBR/LBP | Gain ($\Delta$) |
|------|------|------|---------|---|
| LLaVA-1.5-13B | VIT | 67.2 | 69.3 | +2.1 |
| LLaVA-Next-7B | DPO | 73.4 | 75.8 | +2.4 |
| Qwen2-VL-7B | VIT | 79.1 | 80.7 | +1.6 |
| InternVL-2-8B | DPO | 81.3 | 83.6 | +2.3 |

Consistent benefits across different model scales and LVLM families.

### Key Findings
- **Language bias is a common phenomenon in both VIT and DPO**: Figure 3 shows synchronized $\mathcal{B}$ and $\mathcal{R}$ trajectories, proving this is a systemic issue of the training paradigm rather than a data issue.
- **Extreme simplicity**: LBR is just loss + $|\mathcal{B}|$, and LBP is DPO + $-\log\sigma(\mathcal{B})$, yet they yield consistent gains across 10+ benchmarks.
- **Zero additional data/models**: Unlike previous solutions requiring reference VLMs or manual annotations, LBR/LBP operates entirely within the original pipeline.
- **Visual confirmation**: Figure 2 shows that LBR significantly increases the model's attention distribution over image tokens.

## Highlights & Insights
- **Complete loop of formalization, quantification, and intervention**: Turning the vague concept of "language bias" into a definable, trackable, and manageable engineering object is a benchmark for methodology in the alignment field.
- **Counter-intuitive effectiveness of simplicity**: The "naive" $|\mathcal{B}|$ regularization dominates complex mitigation methods, suggesting that the solution can be simple once the diagnosis is accurate.
- **Addressing the training paradigm rather than data**: Previous hallucination mitigations relied on adding data, changing prompts, or adding reward models; this paper proves the issue lies in the training objective and solves it at the loss function level.
- **Two-stage diagnosis (VIT vs DPO)**: Bias exists in both stages but in different forms (VIT grows bias from zero; DPO starts with bias and amplifies it). Using gentle LBR regularization and proactive LBP unloading respectively shows meticulous differentiation.

## Limitations & Future Work
- Calculating $\mathcal{B}$ requires one text-only forward pass per step, increasing training costs by approximately 50%.
- The choice of $\pi_{\text{ref}}$ affects the measurement of $\mathcal{B}$; the difference between pre-VIT and intermediate checkpoints is not fully discussed.
- Simply penalizing $|\mathcal{B}|$ might harm certain "language-dominant" tasks (e.g., pure text reasoning), requiring finer task-aware control.
- In LBP, $y$ can be chosen or rejected, but the paper does not deeply analyze the subtle difference between applying it to $y_w$ vs $y_l$.
- Sensitive analysis regarding the number of visual tokens or patch size was not conducted for LBR.

## Related Work & Insights
- **vs Training-free decoding (VCD, OPERA, etc.)**: Those methods modify decoding post-hoc without addressing root causes; LBR/LBP provides a fundamental cure during training.
- **vs Data-driven hallucination mitigation (GRIT, RLHF-V)**: Those rely on finely annotated data; LBR/LBP requires zero additional data.
- **vs Existing modality alignment (e.g., SF-Tuning)**: Those use ad-hoc architectural changes; LBR/LBP only modifies the loss.
- **Inspiration**: All "multimodal fusion training objectives" can be similarly decomposed and monitored (e.g., audio-LLM, video-LLM); the "quantified training dynamic metric → direct penalty" template can be generalized to other alignment problems.

## Rating
- Novelty: ⭐⭐⭐⭐ Formalization + simple mitigation, though $|\mathcal{B}|$ regularization itself is not complex; the core innovation is the "diagnosis + simple solution" framework.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 10+ general + 5+ hallucination benchmarks + multiple models/scales; complete coverage.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logical flow: Definition → Tracking → Diagnosis → Loss → Verification; Figure 3 provides decisive evidence.
- Value: ⭐⭐⭐⭐⭐ Directly applicable to all LVLM training pipelines with zero extra data cost; hallucinations are a primary obstacle for LVLM deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Attention-space Contrastive Guidance for Efficient Hallucination Mitigation in LVLMs](../../CVPR2026/multimodal_vlm/attention-space_contrastive_guidance_for_efficient_hallucination_mitigation_in_l.md)
- [\[ICML 2026\] Self-Prophetic Decoding to Unlock Visual Search in LVLMs](self-prophetic_decoding_to_unlock_visual_search_in_lvlms.md)
- [\[ACL 2025\] InternLM-XComposer2.5-Reward: A Simple Yet Effective Multi-Modal Reward Model](../../ACL2025/multimodal_vlm/internlm-xcomposer25-reward_a_simple_yet_effective_multi-modal_reward_model.md)
- [\[ICML 2026\] Mitigating Perceptual Judgment Bias in Multimodal LLM-as-a-Judge via Perceptual Perturbation and Reward Modeling](mitigating_perceptual_judgment_bias_in_multimodal_llm-as-a-judge_via_perceptual_.md)
- [\[CVPR 2026\] SEATrack: Simple, Efficient, and Adaptive Multimodal Tracker](../../CVPR2026/multimodal_vlm/seatrack_multimodal_tracker.md)

</div>

<!-- RELATED:END -->

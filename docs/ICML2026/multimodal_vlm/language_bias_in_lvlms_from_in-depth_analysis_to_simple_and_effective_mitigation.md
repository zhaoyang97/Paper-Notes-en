---
title: >-
  [Paper Note] LBR/LBP: Language Bias in LVLMs — From In-Depth Analysis to Simple and Effective Mitigation
description: >-
  [ICML 2026][Multimodal VLM][Language Bias] This paper systematically quantifies language bias in LVLM training—finding that both VIT and DPO stages cause the text-only likelihood $\pi(y|x)$ to increase nearly as much as…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Language Bias"
  - "VIT"
  - "DPO"
  - "Modality Misalignment"
  - "Plug-and-play Training"
date: 2026-05-08
content_hash: 02bf3ec7cb8bb78e
---

# LBR/LBP: Language Bias in LVLMs — From In-Depth Analysis to Simple and Effective Mitigation

**Conference**: ICML 2026  
**arXiv**: [2605.25036](https://arxiv.org/abs/2605.25036)  
**Code**: https://github.com/lab-klc/LVLM-Language-Bias  
**Area**: Multimodal VLM / Training Objectives / Hallucination Mitigation  
**Keywords**: Language Bias, VIT, DPO, Modality Misalignment, Plug-and-play Training

## TL;DR
This paper systematically quantifies language bias in LVLM training—finding that both VIT and DPO stages cause the text-only likelihood $\pi(y|x)$ to increase nearly as much as the multimodal likelihood $\pi(y|x,v)$, proving that LVLMs systematically underestimate visual input. The authors propose Language Bias Regularization (penalizing $|\mathcal{B}|$ during VIT) and Language Bias Penalty (penalizing positive bias growth during DPO), which significantly improve performance across 10+ benchmarks and reduce hallucinations without additional data or auxiliary models.

## Background & Motivation

**Background**: LVLMs connect vision to LLMs, but suffer from severe hallucinations—generating fluent text that contradicts visual content. Research generally attributes this to language bias (excessive reliance on language, ignoring vision); existing mitigations are categorized into training-free (output post-processing) and training-based (fine-tuning / DPO).

**Limitations of Prior Work**: (1) Understanding of language bias remains empirical—"looking at images less and text more"—without a formal definition; (2) There are no quantitative metrics to track the training dynamics of bias; (3) Existing methods are stop-gap solutions that do not address the root cause; (4) While VIT and DPO are standard in modern alignment, no systematic check has been performed on whether they inherently create bias.

**Key Challenge**: The training objective $\max \pi(y|x, v)$ seemingly requires $v$, but given both $v$ and $x$, the model can satisfy the objective using a text-only path—implying the objective itself does not mandate visual grounding. Consequently, the learned $\pi(y|x)$ (text-only) rises as fast as $\pi(y|x,v)$ (multimodal), rendering the visual modality redundant.

**Goal**: (1) Formally define and quantify language bias; (2) Diagnose bias during both VIT and DPO training stages; (3) Provide mitigation strategies that can be directly integrated into existing pipelines.

**Key Insight**: Decompose the training reward—separately track reward $\mathcal{R} = \log \pi_\theta(y|x,v)/\pi_{\text{ref}}(y|x,v)$ (multimodal gain) and bias $\mathcal{B} = \log \pi_\theta(y|x)/\pi_{\text{ref}}(y|x)$. If $\mathcal{B} \approx \mathcal{R}$, improvements are entirely driven by the language path.

**Core Idea**: Directly penalize $\mathcal{B}$ within the loss function after its definition—adding $|\mathcal{B}|$ (LBR) during VIT, and penalizing the positive growth of $\mathcal{B}_w$ (LBP) on chosen samples during DPO.

## Method

### Overall Architecture

**Baseline VIT**: $\mathcal{L}_{\text{VIT}} = -\sum_t \log \pi_\theta(y_t | x, v, y_{<t})$

**Baseline DPO (with margin)**: $\mathcal{L}_{\text{DPO}_M} = \mathcal{L}_{\text{DPO}} + \mathcal{L}_{\text{Margin}}$

**Formal Definition of Language Bias**: $\mathcal{B} = \log \pi_\theta(y|x) / \pi_{\text{ref}}(y|x)$

**Tracking Metrics**: Calculate $\mathcal{R}_{\text{VIT}}, \mathcal{B}_{\text{VIT}}$ (VIT) or $\mathcal{R}_{\text{DPO}_{w/l}}, \mathcal{B}_{\text{DPO}_{w/l}}$ (DPO) at each step. If $\mathcal{B}$ nearly overlaps with $\mathcal{R}$, bias is severe.

### Key Designs

1. **Formalization of Language Bias + Training Dynamics Tracking**:

    - **Function**: Transforms "language bias" from an empirical phenomenon into a definable and trackable scalar.
    - **Mechanism**: Decomposes reward = multimodal gain $\mathcal{R}$ + text-only gain $\mathcal{B}$; uses $\pi_{\text{ref}}$ (pre-VIT or pre-DPO model) as a baseline to quantify how much the LLM learns from pure text paths during training. Figure 3 in the paper shows that $\mathcal{R}_{\text{VIT}}$ and $\mathcal{B}_{\text{VIT}}$ trajectories nearly overlap, and for DPO, $\mathcal{B}_{\text{DPO}_w}$ even exceeds $\mathcal{R}_{\text{DPO}_w}$.
    - **Design Motivation**: Previously, language bias was an "intuitive" concept; without metrics, it could not be engineered. Providing an operational definition allows for direct management within the loss.

2. **LBR: Penalizing $|\mathcal{B}|$ during VIT**:

    - **Function**: Prevents the uncontrolled growth of language bias during VIT training.
    - **Mechanism**: $\mathcal{L}_{\text{LBR}} = |\mathcal{B}| = |\log \pi_\theta(y|x) / \pi_{\text{ref}}(y|x)|$, added as a weighted sum to the VIT loss. The absolute value is used to penalize both growth and degradation. Since $\mathcal{B}$ is small after pre-training, LBR primarily prevents deterioration during the VIT stage.
    - **Design Motivation**: Bias is minimized after pre-training; VIT is the "problematic stage." Directly penalizing $\mathcal{B}$ forces language-modality reasoning back to reference levels, compelling the model to use vision to gain additional reward.

3. **LBP: Penalizing Positive Bias Growth of 'Chosen' samples in DPO**:

    - **Function**: Prevents preferred answers from gaining reward solely through pure text paths during DPO training.
    - **Mechanism**: Adds an extra term $\max(0, \mathcal{B}_w)$ to the DPO loss to penalize the positive growth of text-only gain for 'chosen' samples (negative growth is not penalized, as performing worse on text-only paths for chosen samples is acceptable). The bias of rejected samples is left untouched to prevent over-regularization.
    - **Design Motivation**: DPO frequently allows chosen samples to improve via text-only paths—a form of reward hacking. LBP specifically blocks this "shortcut" of relying on language.

## Key Experimental Results

### Performance Gains of LBR on 10+ General Benchmarks (LLaVA-1.5-7B)

| Benchmark | Baseline VIT | **+ LBR** | Gain (Δ) |
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

Consistent gains of +1.5~2.5 points across 10/10 benchmarks without cherry-picking.

### Performance Gains of LBP on Hallucination/Trustworthiness Benchmarks

| Benchmark | DPO baseline | **+ LBP** | Gain (Δ) |
|------|----------|---------|---|
| POPE Accuracy | 86.4 | **88.9** | +2.5 |
| MMHal-Bench score | 2.71 | **3.18** | +0.47 |
| AMBER Combined | 65.3 | **69.7** | +4.4 |
| ObjectHal-Bench | 11.2 (lower is better) | **7.8** | −3.4 |
| TrustEval | 71.5 | **75.3** | +3.8 |

Hallucinations significantly decreased and trustworthiness improved comprehensively; ObjectHal decreased by 30%.

### Cross Model Scales / Architectures

| Model | Task | Baseline | + LBR/LBP | Gain (Δ) |
|------|------|------|---------|---|
| LLaVA-1.5-13B | VIT | 67.2 | 69.3 | +2.1 |
| LLaVA-Next-7B | DPO | 73.4 | 75.8 | +2.4 |
| Qwen2-VL-7B | VIT | 79.1 | 80.7 | +1.6 |
| InternVL-2-8B | DPO | 81.3 | 83.6 | +2.3 |

Consistent benefits across model scales and different LVLM families.

### Key Findings
- **Language bias is a common phenomenon in both VIT and DPO stages**: Figure 3 shows identical $\mathcal{B}$ and $\mathcal{R}$ trajectories—proving this is a systemic issue of the training paradigm, not a data issue.
- **Simplicity of the method**: LBR is simply VIT loss + $|\mathcal{B}|$, and LBP is DPO + $\max(0, \mathcal{B}_w)$, yet they yield consistent gains across 10+ benchmarks and multiple models.
- **Zero extra data/models**: Unlike previous mitigation strategies that require external reference VLMs or manual annotations, LBR/LBP operate entirely within the original pipeline.
- **Visualization Confirmation**: Figure 2 shows that LBR significantly elevates the model's attention distribution over image tokens.

## Highlights & Insights
- **Complete closed-loop of Formalization + Quantification + Intervention**: Transforming the vague concept of "language bias" into an engineering object that can be defined, tracked, and intervened upon—this "Definition → Measurement → Loss Term" methodology serves as a template for the alignment field.
- **Counter-intuitively effective simplicity**: "Naive" regularization like $|\mathcal{B}|$ dominates complex mitigation methods—suggesting that once a problem is correctly diagnosed, the solution can be very concise.
- **Targeting training paradigms rather than data**: Previous hallucination mitigations added data, modified prompts, or added reward models; Ours proves the problem lies in the training objective itself, solving it at the loss function level once and for all.
- **Dual-stage diagnosis of VIT vs. DPO**: While both stages exhibit bias, its form differs (modality misalignment in VIT vs. reward hacking in DPO). Addressing them with specific LBR/LBP solutions provides meticulous distinction rather than a one-size-fits-all approach.

## Limitations & Future Work
- Calculating $\mathcal{B}$ requires one text-only forward pass per step, increasing training costs by approximately 50%.
- The choice of $\pi_{\text{ref}}$ affects $\mathcal{B}$ measurement—the difference between pre-VIT and intermediate checkpoints is not fully discussed.
- Simply penalizing $|\mathcal{B}|$ might harm certain "language-dominant" tasks (e.g., pure text reasoning), necessitating finer task-aware control.
- During DPO, only $\mathcal{B}_w$ is penalized; the complex role of $\mathcal{B}_l$ (the relationship between chosen and rejected text gains) remains unexplored.
- Sensitivity of LBR to visual token counts or patch sizes has not been analyzed.

## Related Work & Insights
- **vs. training-free decoding (VCD, OPERA, etc.)**: Those methods modify decoding post-hoc without addressing root causes; LBR/LBP provides a cure during training.
- **vs. data-driven hallucination mitigation (GRIT, RLHF-V)**: Those rely on finely annotated data; LBR/LBP requires zero extra data.
- **vs. existing modality alignment (e.g., SF-Tuning)**: Those involve ad-hoc architectural changes; LBR/LBP only modifies the loss.
- **Inspiration**: All "multimodal fusion training objectives" can be monitored via similar decomposition (e.g., audio-LLM, video-LLM); the "quantitative training dynamics → direct penalty" template can be generalized to other alignment problems.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Formalization + simple mitigation, though $|\mathcal{B}|$ regularization itself is not complex; the core innovation is the "Diagnosis + Simple Solution" framework.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 10+ general + 5+ hallucination benchmarks + multiple models/scales, covering all aspects.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear argumentative logic: Definition → Tracking → Diagnosis → Loss → Verification; Figure 3 provides decisive evidence.
- **Value**: ⭐⭐⭐⭐⭐ Directly applicable to all LVLM training pipelines with zero extra cost; addresses one of the biggest obstacles to LVLM deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Adaptive Residual-Update Steering for Low-Overhead Hallucination Mitigation in Large Vision Language Models](adaptive_residual-update_steering_for_low-overhead_hallucination_mitigation_in_l.md)
- [\[ICML 2026\] Self-Prophetic Decoding to Unlock Visual Search in LVLMs](self-prophetic_decoding_to_unlock_visual_search_in_lvlms.md)
- [\[ICML 2026\] Mitigating Perceptual Judgment Bias in Multimodal LLM-as-a-Judge via Perceptual Perturbation and Reward Modeling](mitigating_perceptual_judgment_bias_in_multimodal_llm-as-a-judge_via_perceptual_.md)
- [\[ACL 2026\] VIGNETTE: Socially Grounded Bias Evaluation for Vision-Language Models](../../ACL2026/multimodal_vlm/vignette_socially_grounded_bias_evaluation_for_vision-language_models.md)
- [\[ICML 2026\] Capturing Gaze Shifts for Guidance: Cross-Modal Fusion Enhancement for VLM Hallucination Mitigation](capturing_gaze_shifts_for_guidance_cross-modal_fusion_enhancement_for_vlm_halluc.md)

</div>

<!-- RELATED:END -->

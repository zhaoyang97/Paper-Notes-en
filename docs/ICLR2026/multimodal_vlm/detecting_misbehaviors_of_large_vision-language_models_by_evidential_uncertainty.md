---
title: >-
  [Paper Note] Detecting Misbehaviors of Large Vision-Language Models by Evidential Uncertainty Quantification
description: >-
  [ICLR2026][Multimodal VLM][LVLM uncertainty] This paper proposes EUQ (Evidential Uncertainty Quantification), which leverages Dempster-Shafer evidence theory to decompose the epistemic uncertainty of LVLMs into **conflict CF** (internal contradictions) and **ignorance IG** (lack of information). EUQ requires no training and only a single forward pass to detect four types of misbehaviors—hallucination, jailbreak, adversarial attacks, and OOD failures—achieving an average AUROC improvement of 10.4%/7.5% over the best baseline.
tags:
  - ICLR2026
  - Multimodal VLM
  - LVLM uncertainty
  - evidential reasoning
  - Dempster-Shafer
  - misbehavior detection
  - hallucination
date: 2026-05-08
content_hash: a8dc8f2fce9262fa
---

# Detecting Misbehaviors of Large Vision-Language Models by Evidential Uncertainty Quantification

**Conference**: ICLR2026
**arXiv**: [2602.05535](https://arxiv.org/abs/2602.05535)
**Code**: [HT86159/EUQ](https://github.com/HT86159/EUQ)
**Area**: Multimodal VLM
**Keywords**: LVLM uncertainty, evidential reasoning, Dempster-Shafer, misbehavior detection, hallucination

## TL;DR

This paper proposes EUQ (Evidential Uncertainty Quantification), which leverages Dempster-Shafer evidence theory to decompose the epistemic uncertainty of LVLMs into **conflict CF** (internal contradictions) and **ignorance IG** (lack of information). EUQ requires no training and only a single forward pass to detect four types of misbehaviors—hallucination, jailbreak, adversarial attacks, and OOD failures—achieving an average AUROC improvement of 10.4%/7.5% over the best baseline.

## Background & Motivation

LVLMs exhibit four typical **misbehaviors** when confronted with difficult, distribution-shifted, or adversarial inputs:

- **Hallucination**: outputs inconsistent with visual content (object/relation/attribute hallucination)
- **Jailbreak**: generation of harmful content induced by malicious visual prompts
- **Adversarial vulnerability**: erroneous predictions caused by imperceptible pixel-level perturbations
- **OOD failure**: inability to correctly recognize inputs with style or quality shifts outside the training distribution

Three core limitations of existing uncertainty quantification (UQ) methods:

1. **Bayesian methods are computationally prohibitive**—infeasible at the scale of LVLMs
2. **Sampling methods require multiple inferences**—semantic entropy (SE), for example, requires 10 generations to estimate consistency, incurring 10× latency
3. **Only aggregate uncertainty is captured**—no distinction between "conflicting internal evidence" and "fundamental lack of relevant knowledge"

The paper's core insight is that different misbehaviors correspond to different sources of epistemic uncertainty. During hallucination, the model simultaneously holds supporting and opposing evidence (high conflict); during OOD failure, the model simply lacks relevant knowledge (high ignorance). This distinction provides a theoretical basis for targeted misbehavior detection.

## Method

### Overall Architecture

Single forward pass through LVLM → extract pre-logits features $\mathbf{Z} \in \mathbb{R}^I$ from the output head → affine transformation + least commitment principle (LCP) to compute evidence weight matrix $\mathbf{E} \in \mathbb{R}^{I \times J}$ → decompose into positive evidence $\mathbf{E}^+$ (supporting) and negative evidence $\mathbf{E}^-$ (opposing) → additive fusion within the same polarity, then Dempster's rule to fuse positive and negative → output conflict CF and ignorance IG.

### Key Designs

1. **Closed-form estimation of evidence weights**: For the projection layer $\mathbf{H} = \mathbf{Z}\mathbf{W} + \mathbf{b}$ of the output head, the contribution of each pre-logits feature $z_i$ to each output dimension $h_j$ is modeled as evidence weight $e_{ij}$. Applying the least commitment principle (LCP), the optimization $\min_{\mathbf{A},\mathbf{B}} \|\mathbf{A} \odot \mathbf{Z}^\top + \mathbf{B}\|_2^2$ yields the closed-form solution $\mathbf{A}^* = W - \mu_0(W)$, requiring no training or iterative optimization.

2. **Positive/negative evidence decomposition and two-stage fusion**: Evidence weights are decomposed into $\mathbf{E}^+ = \max(0, \mathbf{E})$ (supporting hypothesis $h_j$) and $\mathbf{E}^- = \max(0, -\mathbf{E})$ (opposing hypothesis $h_j$). In the first stage, the additivity of evidence weights (Lemma 2) allows same-polarity evidence to be directly summed, avoiding power-set enumeration. In the second stage, Dempster's rule fuses positive and negative evidence to compute:
    - $\mathbf{CF} = \sum_j \eta_j^+ \cdot \eta_j^-$: a large product indicates strong simultaneous support and opposition for some $h_j$, reflecting internal contradiction
    - $\mathbf{IG} = \sum_j \exp(-e_j^-)$: weaker negative evidence (smaller $e_j^-$) implies higher ignorance, reflecting lack of information

3. **Sentence-level uncertainty aggregation**: Since LVLMs generate text token by token, each token yields corresponding CF and IG values; the mean across all tokens serves as the sentence-level uncertainty measure.

### Misbehavior-Bench Evaluation Framework

A unified benchmark covering 4 misbehavior types across 9 datasets is constructed:

| Misbehavior Type | Dataset | Samples | Question Type |
|---------|--------|-------|---------|
| Hallucination | POPE + R-Bench | 2000 | Multiple choice |
| Jailbreak | FigStep + Hades + VisualAdv + Typographic | 2800 | Open-ended/Multiple choice |
| Adversarial | ANDA + PGN | 400 | Yes/No |
| OOD | OOD-Bench | 1300 | Yes/No |

Evaluated models: DeepSeek-VL2-Tiny, Qwen2.5-VL-7B, InternVL2.5-8B, MoF-7B (covering SwiGLU and MoE architectures).

## Key Experimental Results

### Overall Comparison (average across 4 models × 4 scenarios)

| Method | Type | AUROC | AUPR | Extra Cost |
|------|------|-------|------|---------|
| SC (self-consistency) | Sampling ×10 | 0.626 | 0.730 | 8.9×10⁻¹s |
| SE (semantic entropy) | Sampling ×10 | 0.624 | 0.661 | 9.0×10⁻¹s |
| PE (predictive entropy) | Probability | 0.701 | 0.656 | 3.1×10⁻⁶s |
| LN-PE | Probability | 0.704 | 0.660 | 6.1×10⁻⁶s |
| HiddenDetect | Hidden features | 0.707 | 0.658 | 2.0×10⁻²s |
| **CF (ours)** | Evidence fusion | **0.812** | 0.783 | 9.1×10⁻⁴s |
| **IG (ours)** | Evidence fusion | 0.783 | **0.785** | 4.5×10⁻³s |

CF achieves a 10.5% AUROC improvement over the best baseline (HiddenDetect), while incurring ~1/1000 the computational cost of sampling-based methods.

### Per-Scenario Best Detection Metrics (AUROC, averaged over 4 models)

| Misbehavior Type | CF | IG | Best Baseline | CF/IG Relative Gain |
|---------|------|------|---------|--------------|
| Hallucination | **0.761** | 0.657 | PE 0.742 | CF +2.6% |
| Jailbreak | **0.757** | 0.665 | HiddenDetect 0.752 | CF +0.7% |
| Adversarial | 0.836 | **0.861** | LN-PE 0.717 | IG +20.1% |
| OOD | 0.894 | **0.948** | HiddenDetect 0.694 | IG +36.6% |

Key finding: **hallucination ↔ high conflict** (CF is best); **OOD ↔ high ignorance** (IG is best). In adversarial scenarios both metrics are effective but IG is superior, consistent with the intuition that adversarial perturbations cause information loss.

### Layer-wise Dynamic Analysis

- **IG decreases with depth**: deeper layers accumulate more supporting evidence, progressively reducing ignorance
- **CF increases with depth**: deeper features become more task-relevant, increasing inter-channel competition and thus conflict
- This pattern is consistent with information bottleneck theory—deep layers compress redundant inputs while enhancing discriminative information

### Ablation Study

- **Temperature robustness**: detection performance of CF and IG remains stable as temperature varies from 0.1 to 1.4
- **Model scale effect**: 4B and 38B models yield better detection performance (errors in smaller models are more salient; errors in larger models are rare but pattern-clear), while subtle errors in 8B mid-scale models are hardest to detect
- **External prompting ineffective**: after adding a "None of the above" option, Qwen selects it only 0.27% of the time and InternVL 0.00%—overconfidence renders prompt-based strategies ineffective

## Highlights & Insights

### Highlights

- **First decomposition of epistemic uncertainty into conflict and ignorance within LVLMs**—provides interpretable error diagnosis: different misbehaviors correspond to different uncertainty sources, guiding targeted remediation strategies
- **Zero training + single forward pass**—closed-form solution requires no optimization; UQ overhead is <1ms, virtually imperceptible in deployment
- **Theoretically rigorous**—grounded in Dempster-Shafer evidence theory, with Lemma 1 (closed-form estimation), Lemma 2 (additivity), and Theorem 1 (CF/IG expressions) forming a coherent derivation chain
- **General applicability**—the method applies to any model with a linear projection layer (BERT, ResNet, LLM), not limited to VLMs

### Limitations

- Requires access to internal model representations; inapplicable to closed-source APIs such as GPT-4
- In adversarial and jailbreak scenarios, CF and IG perform similarly, making individual attribution difficult
- The layer-wise analysis currently identifies all four misbehavior types only at specific layers, with no automatic mechanism for optimal layer selection

## Limitations & Future Work

- Only output head features are used; rich information from intermediate layers remains unexplored
- The closed-form solution for evidence weights relies on the linear projection assumption
- The current approach focuses on detection rather than correction—how to improve outputs upon detecting high uncertainty is a direction for future work

## Related Work & Insights

- **vs. Semantic Entropy**: requires multiple samplings and an external model to assess semantic equivalence. EUQ operates with a single forward pass.
- **vs. Verbalized Confidence**: relies on the model's metacognitive ability (unreliable). EUQ extracts uncertainty directly from features.
- **vs. Evidential Deep Learning**: requires training. EUQ is entirely training-free.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First application of evidence-theoretic CF/IG decomposition to LVLM misbehavior detection
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 4 models × 4 misbehavior types × multiple baselines, with in-depth layer-wise analysis
- Writing Quality: ⭐⭐⭐⭐ Rigorous theoretical derivation with helpful visualizations
- Value: ⭐⭐⭐⭐⭐ Direct practical value for LVLM trustworthiness and safe deployment

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Topo-R1: Detecting Topological Anomalies via Vision-Language Models](../../CVPR2026/multimodal_vlm/topo-r1_detecting_topological_anomalies_via_vision-language_models.md)
- [\[CVPR 2026\] Uncertainty-Aware Knowledge Distillation for Multimodal Large Language Models](../../CVPR2026/multimodal_vlm/uncertainty-aware_knowledge_distillation_for_multimodal_large_language_models.md)
- [\[ICLR 2026\] CityLens: Evaluating Large Vision-Language Models for Urban Socioeconomic Sensing](citylens_evaluating_large_vision-language_models_for_urban_socioeconomic_sensing.md)
- [\[CVPR 2026\] Uncertainty-guided Compositional Alignment with Part-to-Whole Semantic Representativeness in Hyperbolic Vision-Language Models](../../CVPR2026/multimodal_vlm/uncertainty-guided_compositional_alignment_with_part-to-whole_semantic_represent.md)
- [\[ICLR 2026\] Vision-R1: Incentivizing Reasoning Capability in Multimodal Large Language Models](vision-r1_incentivizing_reasoning_capability_in_multimodal_large_language_models.md)

<!-- RELATED:END -->

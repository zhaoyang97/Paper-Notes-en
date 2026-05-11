---
title: >-
  [Paper Note] Decoupling Stability and Plasticity for Multi-Modal Test-Time Adaptation
description: >-
  [CVPR 2026][Multimodal VLM][Multi-modal test-time adaptation] This paper proposes DASP, which diagnoses biased modalities via a redundancy score and applies an asymmetric adaptation strategy to decouple stability and pla…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "Multi-modal test-time adaptation"
  - "stability-plasticity decoupling"
  - "redundancy score"
  - "asymmetric adaptation"
  - "catastrophic forgetting"
date: 2026-05-08
content_hash: 2b993f1b061b4c31
---

# Decoupling Stability and Plasticity for Multi-Modal Test-Time Adaptation

**Conference**: CVPR 2026
**arXiv**: [2603.00574](https://arxiv.org/abs/2603.00574)
**Code**: [GitHub](https://github.com/he4cs/DASP)
**Area**: Multimodal VLM
**Keywords**: Multi-modal test-time adaptation, stability-plasticity decoupling, redundancy score, asymmetric adaptation, catastrophic forgetting

## TL;DR

This paper proposes DASP, which diagnoses biased modalities via a redundancy score and applies an asymmetric adaptation strategy to decouple stability and plasticity, addressing negative transfer and catastrophic forgetting in multi-modal test-time adaptation.

## Background & Motivation

**Vulnerability of multi-modal models to distribution shift**: Multi-modal models (audio-video) are susceptible to distribution shifts in non-stationary environments, such as weather changes and sensor degradation, leading to significant performance degradation in statically pre-trained models.

**Rise of Test-Time Adaptation (TTA)**: TTA enables online parameter updates to adapt to distribution shifts without access to source data; however, existing methods are predominantly designed for unimodal settings.

**Negative transfer**: Modality-agnostic adaptation strategies indiscriminately update all modalities, potentially causing negative transfer to well-aligned, unbiased modalities.

**Catastrophic forgetting**: Continuous parameter updates erase source domain knowledge, particularly severe in biased modalities.

**Stability-plasticity dilemma**: Existing methods struggle to balance the two objectives—biased modalities require plasticity to adapt to the target distribution, while unbiased modalities require stability to preserve source domain knowledge.

**Unreliability of conventional diagnostic metrics**: Entropy and confidence are unreliable in multi-modal settings—a dominant modality may maintain low entropy and high confidence even under distribution shift, making cross-modal comparison infeasible.

## Method

### Overall Architecture

DASP follows a *diagnose-then-mitigate* framework: (1) biased modalities are identified via a redundancy score; (2) an asymmetric adaptation strategy handles biased and unbiased modalities separately.

### Key Designs

#### Redundancy Score for Diagnosis

Inter-dimensional correlations of each modality's features are computed in the shared latent space of the fusion layer. Distribution shift causes feature manifold degeneration, inducing spurious inter-dimensional correlations (all dimensions respond uniformly to domain-specific noise), resulting in a marked increase in redundancy. A redundancy score $R(\mathbf{Z})$ is defined, and the relative redundancy of each modality is compared:

$$\Delta^m = r^m - \min_{n \in \mathcal{M}} r^n$$

Modality $m$ is identified as biased and added to $\mathcal{G}$ when $\Delta^m \geq \delta$.

#### Asymmetric Adaptation

Each modality adapter $\Phi^m$ is decomposed into a stable adapter $\phi_s^m$ (low-rank, encouraging domain-agnostic generalization) and a plastic adapter $\phi_p^m$ (high-rank, capturing domain-specific information):

- **Biased modalities** ($m \in \mathcal{G}$): activate the plastic adapter and freeze the stable adapter → $\tilde{z}^m = \phi_p^m(\phi_s^m(z^m))$
- **Unbiased modalities** ($m \notin \mathcal{G}$): bypass the plastic adapter, update the stable adapter with KL regularization → $\tilde{z}^m = \phi_s^m(z^m)$

### Loss & Training

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{div}} + \lambda_{\text{ent}} \mathcal{L}_{\text{ent}} + \lambda_{\text{kl}} \mathcal{L}_{\text{kl}}$$

The objective comprises a diversity regularization term (preventing class collapse), an entropy minimization term (encouraging confident predictions), and a KL divergence penalty (constraining the stable adapters of unbiased modalities from deviating from the source model).

## Key Experimental Results

### Main Results: Kinetics50-C Video Corruption (Episodic Adaptation)

| Method | Mean Accuracy↑ |
|--------|---------------|
| Source (no adaptation) | 59.9 |
| Tent | 59.4 |
| EATA | 60.1 |
| SAR | 59.8 |
| READ | 62.5 |
| TSA | 63.8 |
| **DASP (Ours)** | **65.2** |

### Ablation Study

| Component | Effect |
|-----------|--------|
| Redundancy score vs. entropy/confidence | Redundancy score strongly correlates with accuracy; entropy/confidence are unreliable |
| Asymmetric vs. symmetric adaptation | Asymmetric adaptation substantially reduces negative transfer and forgetting |
| KL regularization | Effectively constrains stability of unbiased modalities |
| Low-rank/high-rank design | Aligns with the functional requirements of each modality role |

### Key Findings

- The redundancy score strongly correlates with accuracy on both Kinetics50-C and VGGSound-C
- Biased modalities exhibit significantly higher redundancy than unbiased ones
- DASP simultaneously mitigates negative transfer (unbiased modalities) and catastrophic forgetting (biased modalities)
- Performance gains are particularly pronounced under audio corruption scenarios (large margins over baselines on VGGSound-C)

## Highlights & Insights

- The redundancy score is an elegant non-parametric diagnostic metric that can be applied online without source domain statistics
- The *diagnose-then-mitigate* framework is logically coherent—localizing the problem before applying targeted remediation
- The decoupled design of stable and plastic adapters is intuitively sound—externalizing domain-specific parameters while internalizing domain-agnostic ones
- The structured low-rank vs. high-rank design naturally aligns with the respective functional roles of each adapter

## Limitations & Future Work

- The threshold $\delta$ for the redundancy score must be predefined and may require adjustment across different scenarios
- Validation is limited to audio-video bimodal settings; extension to more modalities (e.g., text + image + audio) remains unexplored
- Modality bias detection is a hard decision, unable to handle cases where both modalities are simultaneously biased
- Computing the redundancy score requires batch statistics, making it inapplicable to batch size = 1 settings

## Related Work & Insights

- Most closely related to TSA's selective adaptation, though TSA's soft routing is less stable in unsupervised settings
- Both MDAA and DASP address catastrophic forgetting, but DASP achieves this through architectural decoupling rather than analytical methods
- The stability-plasticity perspective generalizes to continual learning, federated learning, and related settings
- The redundancy score can serve as a general-purpose distribution shift detection tool

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Panda: Test-Time Adaptation with Negative Data Augmentation](../../AAAI2026/multimodal_vlm/panda_test-time_adaptation_with_negative_data_augmentation.md)
- [\[CVPR 2026\] Test-Time Attention Purification for Backdoored Large Vision Language Models](test-time_attention_purification_for_backdoored_large_vision_language_models.md)
- [\[CVPR 2026\] Evolving Contextual Safety in Multi-Modal Large Language Models via Inference-Time Self-Reflective Memory](evolving_contextual_safety_in_multi-modal_large_language_models_via_inference-ti.md)
- [\[AAAI 2026\] Bridging Modalities via Progressive Re-alignment for Multimodal Test-Time Adaptation (BriMPR)](../../AAAI2026/multimodal_vlm/bridging_modalities_via_progressive_re-alignment_for_multimo.md)
- [\[CVPR 2026\] BriMA: Bridged Modality Adaptation for Multi-Modal Continual Action Quality Assessment](brima_bridged_modality_adaptation_for_multi-modal_continual_action_quality_asses.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] GRAZE: Grounded Refinement and Motion-Aware Zero-Shot Event Localization
description: >-
  [CVPR 2026][LLM Reasoning][Zero-shot temporal localization] GRAZE is proposed as a training-free pipeline that leverages Grounding DINO to discover candidate interactions and employs SAM2 mask overlap as a pixel-level contact verifier, achieving 97.4% coverage and 77.5% contact onset frame localization accuracy within ±10 frames on 738 American football training videos.
tags:
  - CVPR 2026
  - LLM Reasoning
  - Zero-shot temporal localization
  - contact detection
  - SAM2
  - Grounding DINO
  - motion-aware
date: 2026-05-08
content_hash: eef20ead4675d66e
---

# GRAZE: Grounded Refinement and Motion-Aware Zero-Shot Event Localization

**Conference**: CVPR 2026
**arXiv**: [2604.01383](https://arxiv.org/abs/2604.01383)
**Code**: None
**Area**: LLM Reasoning
**Keywords**: Zero-shot temporal localization, contact detection, SAM2, Grounding DINO, motion-aware

## TL;DR

GRAZE is proposed as a training-free pipeline that leverages Grounding DINO to discover candidate interactions and employs SAM2 mask overlap as a pixel-level contact verifier, achieving 97.4% coverage and 77.5% contact onset frame localization accuracy within ±10 frames on 738 American football training videos.

## Background & Motivation

In contact-based sports analysis (e.g., American football), localizing the **First Point of Contact (FPOC)** in training videos is critical for biomechanical analysis. Coaches and sports scientists need to know precisely when an athlete makes physical contact with a blocking dummy to perform collision posture assessment and kinematic analysis.

However, real-world training videos present severe challenges:
- **Handheld/fixed-camera capture**: Significant camera shake and panning
- **Multi-person scenes**: Multiple athletes in similar equipment appear simultaneously, causing interference
- **Large appearance variation**: Equipment and lighting differ substantially across training sessions
- **Detection confidence ≠ physical contact**: Bounding box overlap does not imply actual physical contact; true contact may in fact cause occlusion-induced loss of overlap

The core difficulty lies in the **gap between candidate discovery and contact confirmation**: detection models measure appearance similarity, not physical intersection. The paper's core insight is to treat SAM2 not as a passive segmentation backend but as an **active pixel-level contact verifier**—providing contact evidence directly through mask intersection, fully decoupled from detection confidence.

## Method

### Overall Architecture

GRAZE is a four-stage zero-shot pipeline requiring no domain-specific training:
1. **Grounding (Candidate Discovery)**: Grounding DINO searches for athlete–dummy pairs at multiple temporal positions
2. **Validation (Motion Validation)**: Candidates are ranked via temporal consistency and directional motion scoring
3. **Refinement (Temporal Refinement)**: Backward tracking finds the earliest frame of dual-object co-occurrence $t_{\text{FFBO}}$
4. **Contact Verification**: SAM2 propagates masks; the first frame with mask overlap is the FPOC

### Key Designs

1. **Hierarchical Prompting and Progressive Search**:
   - Three-level prompt hierarchy: $\mathcal{P} = \{P_{\text{gear}}, P_{\text{nogear}}, P_{\text{generic}}\}$, ranging from most detailed (helmet + sprint posture description) to most generic (person running toward a red object)
   - Six temporal positions are probed per video; each position attempts three progressively relaxed detection thresholds
   - All valid candidates are **exhaustively collected** rather than stopping at first success—because detection quality and contact quality are not monotonically correlated

2. **Directional Motion Scoring**:
   - Displacement score $m_{\text{disp}}$: measures the displacement of the candidate athlete within the validation window (normalized to 200 pixels)
   - Directional proximity score $m_{\text{dir}}$: cosine similarity between the motion vector and the direction toward the dummy, scaled to $[0,1]$
   - Combined ranking score: $\text{conf}_{\text{overall}} = 0.3\, c_{\text{cons}} + 0.3\, m_{\text{disp}} + 0.4\, m_{\text{dir}}$
   - Static bystanders and laterally moving athletes are filtered via $m_{\text{disp}} < 0.08$ or $m_{\text{dir}} < 0.30$

3. **Two-Phase Backward Refinement**:
   - Phase 1: Steps backward frame by frame from the grounding frame, allowing at most one consecutive miss
   - Phase 2: Exponential offset probing ($\{5, 10, 20, 50\}$ frames), followed by binary search to locate the earliest consistent frame after a candidate is found
   - Addresses grounding bias: grounding is most reliable mid-contact (when both objects are simultaneously salient), causing $t_g$ to be systematically later than the true onset

4. **SAM2 Contact Verification**:
   - SAM2 is initialized at $t_{\text{FFBO}}$ using the refined bounding boxes, propagating binary masks for both the athlete and the dummy
   - Contact quantification: $\text{overlap}_t = \sum_{x,y} \mathcal{M}_t^{(P)}(x,y) \wedge \mathcal{M}_t^{(D)}(x,y)$
   - FPOC is defined as the earliest frame where mask overlap reaches at least 1 pixel
   - If no overlap is found, the current candidate is rejected and the next-ranked candidate is evaluated—**multi-candidate fallback** mechanism

### Loss & Training

No training is involved; no loss functions are applicable. All components exploit the zero-shot capabilities of pretrained models.

## Key Experimental Results

### Main Results

| Metric | GRAZE | SOLE (B1) | TRACE (B2) | MARS (B3) |
|---|---|---|---|---|
| Coverage | **97.4%** | 92.0% | 46.9% | 91.9% |
| ±5 frames end-to-end | 71.4% | 68.0% | — | 68.2% |
| ±10 frames end-to-end | **77.5%** | 70.6% | — | 70.7% |
| ±20 frames end-to-end | **82.7%** | 72.6% | — | 72.6% |
| ±20 frames conditional | **91.6%** | 85.8% | — | 85.7% |
| Catastrophic error rate ($|err| \geq 20$) | **8.4%** | 14.2% | — | 14.3% |

Dataset: 738 untrimmed American football training videos at 30 fps; 681 clips carry frame-level ground-truth annotations.

### Ablation Study

| Configuration | Coverage | ±10 frames | Notes |
|---|---|---|---|
| SOLE (single prompt + SAM2 only) | 92.0% | 70.6% | Simplest baseline |
| TRACE (+temporal validation + backtracking) | 46.9% | — | No directional filtering → coverage collapse |
| MARS (+motion scoring) | 91.9% | 70.7% | Motion scoring alone yields limited improvement |
| **GRAZE (full)** | **97.4%** | **77.5%** | Multiplicative synergy across all components |

### Key Findings

- **Detection confidence ≠ contact evidence**: This is the central insight of the method. SAM2 mask intersection provides geometric contact evidence independent of the detection model.
- **Directional motion filtering is critical for coverage**: TRACE's coverage collapses to 46.9% without directional filtering—temporal consistency alone cannot distinguish active tacklers from static bystanders.
- **Backward refinement primarily benefits precision at looser tolerances**: GRAZE slightly underperforms the baseline at ±5 frames (79.1% vs. 80.4%) but halves the catastrophic error rate (8.4% vs. 14.3%).
- **Multi-component synergy**: Motion scoring alone yields negligible improvement, but combined with multi-candidate fallback it effectively suppresses interfering candidates prior to SAM2 verification.

## Highlights & Insights

1. **Reframing SAM2 as a contact verifier**: This goes beyond the conventional use of SAM2 as a passive segmentation backend, leveraging mask intersection as direct geometric evidence of physical contact.
2. **High performance without training**: 97.4% coverage and 91.6% conditional accuracy are achieved with no labeled data or fine-tuning.
3. **Sound engineering design**: Exhaustive candidate collection combined with a ranked fallback mechanism ensures system robustness under diverse degraded conditions.
4. **Clear problem formulation**: FPOC localization is explicitly distinguished from general action localization—the latter asks "what does the action look like," while the former asks "do two objects physically intersect."

## Limitations & Future Work

- **Strong domain specificity**: The pipeline is heavily tailored to the "person striking a dummy" scenario; prompt templates and parameters are customized accordingly.
- **Slight degradation at ±5 frames**: Backward refinement occasionally retreats 1–2 frames too far, causing a marginal accuracy drop under tight tolerances.
- **Dependence on zero-shot capabilities of Grounding DINO and SAM2**: Performance may degrade under extreme scene variation.
- **No external baseline comparison**: Evaluation is limited to ablated variants of the proposed system; no comparison with supervised methods or other zero-shot approaches is provided.
- **Single-contact events only**: Multiple consecutive contact events within a single video are not handled.

## Related Work & Insights

- Unlike supervised temporal localization methods such as ActionFormer and BMN, which require frame-level annotations and output temporal segments rather than precise frames.
- Unlike zero-shot methods such as T3AL and ZEETAD, which rely on appearance matching rather than physical intersection detection.
- The use of SAM2 as a verifier generalizes to other scenarios requiring determination of whether two objects are in physical contact (e.g., robotic grasp verification, collision detection).

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The reuse of SAM2 as a contact verifier is clever, though the overall system assembles existing modules.
- **Experimental Thoroughness**: ⭐⭐⭐ — The 738-video scale is substantial, but the absence of external baselines is a notable gap.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem formulation and method description are clear and well-formalized.
- **Value**: ⭐⭐⭐ — The application domain is narrow, though the core insight (detection ≠ contact) has broader applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] CoT-RVS: Zero-Shot Chain-of-Thought Reasoning Segmentation for Videos](../../ICLR2026/llm_reasoning/cot-rvs_zero-shot_chain-of-thought_reasoning_segmentation_for_videos.md)
- [\[ICLR 2026\] SceneCOT: Eliciting Grounded Chain-of-Thought Reasoning in 3D Scenes](../../ICLR2026/llm_reasoning/scenecot_eliciting_grounded_chain-of-thought_reasoning_in_3d_scenes.md)
- [\[ICLR 2026\] Thinking in Latents: Adaptive Anchor Refinement for Implicit Reasoning in LLMs](../../ICLR2026/llm_reasoning/thinking_in_latents_adaptive_anchor_refinement_for_implicit_reasoning_in_llms.md)
- [\[ACL 2026\] Semantic-Aware Logical Reasoning via a Semiotic Framework](../../ACL2026/llm_reasoning/semantic-aware_logical_reasoning_via_a_semiotic_framework.md)
- [\[ACL 2026\] Budget-Aware Anytime Reasoning with LLM-Synthesized Preference Data](../../ACL2026/llm_reasoning/budget-aware_anytime_reasoning_with_llm-synthesized_preference_data.md)

</div>

<!-- RELATED:END -->

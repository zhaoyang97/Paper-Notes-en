---
title: >-
  [Paper Note] Out of Sight, Out of Track: Adversarial Attacks on Propagation-based Multi-Object Trackers via Query State Manipulation
description: >-
  [CVPR 2026][Video Understanding][Multi-object tracking] This paper presents the first systematic analysis of adversarial vulnerabilities in Tracking-by-Query-Propagation (TBP) trackers…
tags:
  - "CVPR 2026"
  - "Video Understanding"
  - "Multi-object tracking"
  - "adversarial attacks"
  - "query propagation"
  - "temporal memory corruption"
  - "physical attacks"
date: 2026-05-08
content_hash: 661c1c572ca6ab59
---

# Out of Sight, Out of Track: Adversarial Attacks on Propagation-based Multi-Object Trackers via Query State Manipulation

**Conference**: CVPR 2026
**arXiv**: [2604.00452](https://arxiv.org/abs/2604.00452)
**Code**: None
**Area**: Video Understanding / Multi-Object Tracking / Adversarial Attacks
**Keywords**: Multi-object tracking, adversarial attacks, query propagation, temporal memory corruption, physical attacks

## TL;DR

This paper presents the first systematic analysis of adversarial vulnerabilities in Tracking-by-Query-Propagation (TBP) trackers, and proposes the FADE attack framework. FADE employs two complementary strategies — Temporal Query Flooding (TQF) to exhaust fixed query budgets by generating persistent spurious tracks, and Temporal Memory Corruption (TMC) to disrupt hidden state propagation of legitimate tracks. On MOT17/MOT20, FADE causes up to ~30 points of HOTA degradation and more than 10× identity switches on MOTR/MOTRv2/MeMOTR/Samba/CO-MOT.

## Background & Motivation

1. **Background**: Multi-object tracking (MOT) has evolved from the traditional Tracking-by-Detection (TBD) paradigm to the more advanced Tracking-by-Propagation (TBP) paradigm. TBP trackers (e.g., MOTR, MOTRv2, MeMOTR, Samba, CO-MOT) achieve end-to-end detection and association by autoregressively propagating track queries, eliminating heuristic components such as Kalman filtering.

2. **Limitations of Prior Work**: Existing adversarial attacks against MOT (Daedalus, Hijacking, F&F, BankTweak) are all designed for TBD architectures — targeting NMS thresholds, Kalman filter predictions, or independent appearance feature banks — none of which exist in end-to-end TBP trackers.

3. **Key Challenge**: TBP trackers introduce fundamentally new structural vulnerabilities: (1) the **fixed query budget** creates a zero-sum game — queries allocated to spurious tracks necessarily reduce capacity for legitimate ones; (2) **recurrent hidden state propagation** creates temporal dependency chains — state corruption propagates across frames; (3) **built-in temporal memory** amplifies attack persistence.

4. **Goal**: Design attack methods tailored to the query budget and temporal memory mechanisms unique to TBP trackers.

5. **Key Insight**: Starting from the core mechanisms of TBP (query budget allocation and memory propagation via the query updater), the paper designs two complementary attacks: one that "infiltrates" (creating spurious tracks to occupy the budget) and one that "erases" (destroying the memory of legitimate tracks).

6. **Core Idea**: Subvert TBP tracking from within its own architecture by either flooding spurious queries to exhaust the budget or corrupting temporal memory to cause catastrophic forgetting.

## Method

### Overall Architecture

FADE is a unified adversarial attack pipeline. Input frames are perturbed via digital perturbations or physically-motivated sensor spoofing (acoustic/electromagnetic) to generate adversarial examples, which are fed into the TBP tracker. The tracker's output predictions and hidden states are used to compute the FADE loss ($\mathcal{L}_{TQF}$ or $\mathcal{L}_{TMC}$), and gradients are backpropagated through a PGD optimization loop to update the perturbation parameters.

### Key Designs

1. **Temporal Query Flooding (TQF)**:

    - **Function**: Exhausts the fixed query budget by generating a large number of high-confidence, temporally persistent spurious tracks.
    - **Mechanism**: Three sub-losses work in concert. (1) **Query Flooding** $\mathcal{L}_{Flood}$: maximizes detection confidence of unmatched queries; (2) **Cost Mimicry** $\mathcal{L}_{Cost}$: causes adversarial queries to present low matching cost against ground-truth targets, deceiving the bipartite matching algorithm into assigning the identities of real targets to spurious queries; (3) **Identity Siphoning** $\mathcal{L}_{Siphon}$: aligns the hidden state of current adversarial queries with that of legitimate tracks from the previous frame, thereby "stealing" their identities and history to make spurious tracks persist.
    - **Design Motivation**: Generating single-frame false positives is insufficient; spurious tracks must be absorbed into the query state as persistent tracks to genuinely exhaust the budget.

2. **Temporal Memory Corruption (TMC)**:

    - **Function**: Directly disrupts the temporal memory and hidden states of existing legitimate tracks.
    - **Mechanism**: Two sub-losses. (1) **Temporal Decorrelation** $\mathcal{L}_{Decorr}$: minimizes the cosine similarity between the current frame's hidden state $\mathcal{H}^t$ and the previous frame's $\mathcal{H}^{t-1}$, severing temporal correlations so the query updater cannot re-associate tracks; (2) **Track Erasure** $\mathcal{L}_{Erase}$: minimizes the L2 norm of matched query hidden states, forcing them to collapse toward the zero vector and erasing the feature identity of the track.
    - **Design Motivation**: Complementary to TQF's "infiltration" strategy, TMC adopts an "erasure" strategy — rather than creating new tracks, it directly destroys existing ones.

3. **Differentiable Digital-to-Physical Attack Pipeline**:

    - **Function**: Extends digital attacks to sensor-spoofing attacks that simulate physical-world effects.
    - **Mechanism**: Two differentiable simulators of physical sensor spoofing are modeled — Acoustic Actuation Injection (AAI, simulating motion blur caused by camera stabilizer resonance) and Electromagnetic Actuation Injection (EAI, simulating color-band artifacts from ADC corruption) — and physical parameters $\theta_{AAI}$ or $\theta_{EAI}$ are optimized within a unified PGD framework.
    - **Design Motivation**: Traditional patch-based attacks apply only to single-target tracking; sensor-level attacks can simultaneously affect all targets in the scene.

### Loss & Training

- TQF: $\mathcal{L}_{TQF} = \lambda_{Flood}\mathcal{L}_{Flood} + \lambda_c\mathcal{L}_{Cost} + \lambda_s\mathcal{L}_{Siphon}$
- TMC: $\mathcal{L}_{TMC} = \lambda_{Decorr}\mathcal{L}_{Decorr} + \lambda_{Erase}\mathcal{L}_{Erase}$
- Digital attacks: ε=8/255, α=1/255, 50 PGD iterations, applied per single frame.
- Physical attacks: α=8/255, 100 iterations, applied over 3 consecutive frames.

## Key Experimental Results

### Main Results

MOT17 digital attack results (selected key trackers):

| Tracker | Attack | HOTA ↓ | AssA ↓ | IDSW ↑ |
|---------|------|--------|--------|--------|
| MeMOTR | Clean | 67.35 | 79.60 | 0.81 |
| MeMOTR | Daedalus | 42.41 | 51.94 | 4.09 |
| MeMOTR | FADE_TMC | 41.56 | 49.18 | **4.63** |
| MeMOTR | FADE_TQF | 41.41 | 50.03 | 4.31 |
| CO-MOT | Clean | 58.16 | 74.87 | 1.83 |
| CO-MOT | FADE_TMC | 41.73 | 55.89 | **10.94** |
| CO-MOT | FADE_TQF | 37.26 | 51.93 | 9.50 |

MOT20 high-density scene (~150 targets/frame):

| Tracker | Attack | HOTA ↓ | IDSW ↑ |
|---------|------|--------|--------|
| MeMOTR | Clean | 69.61 | 0.46 |
| MeMOTR | FADE_TMC | 37.70 | 4.90 |
| MeMOTR | FADE_TQF | 57.67 | 1.51 |
| MOTRv2 | Clean | 59.56 | 0.73 |
| MOTRv2 | FADE_TQF | 29.64 | 5.10 |

### Ablation Study

Comparison of attack strategies (MOT17 CO-MOT, Clean HOTA=58.16):

| Attack | HOTA | Relative Drop | Notes |
|---------|------|---------|------|
| Daedalus (detection evasion) | 40.01 | -18.15 | TBD attack applied directly |
| F&F (association perturbation) | 52.78 | -5.38 | Limited effectiveness of TBD attack |
| FADE_TMC | 41.73 | -16.43 | Memory corruption is effective |
| FADE_TQF | **37.26** | **-20.90** | Query flooding is strongest |

### Key Findings

- **TQF is most effective on models with tight query budgets**: HOTA on CO-MOT drops from 58.16 to 37.26 (−20.90), as its label assignment strategy is more susceptible to query flooding.
- **TMC excels at inducing identity switches**: IDSW on CO-MOT surges from 1.83 to 10.94 (~6×), as direct memory corruption causes frequent ID reassignments.
- **High-density scenes amplify attack effectiveness**: On MOT20, MeMOTR under TMC attack drops from 69.61 to 37.70 HOTA (−31.91), exceeding the degradation observed on MOT17.
- **Existing TBD attacks partially fail on TBP trackers**: Hijacking and F&F are nearly ineffective on MOTRv2 (HOTA drops of only ~1–4 points), confirming that TBP trackers require dedicated attack strategies.

## Highlights & Insights

- **First identification of three structural vulnerabilities in TBP trackers**: the zero-sum game of fixed query budgets, recurrent hidden state propagation, and persistence amplification via built-in temporal memory. These analyses provide guidance for security-aware design of TBP architectures.
- **Elegant "identity siphoning" design in TQF**: Rather than simply generating spurious targets, TQF aligns spurious tracks with legitimate tracks in hidden state space to steal their identities, exploiting TBP's own matching and propagation mechanisms.
- **Differentiable physical attack pipeline**: Modeling acoustic/electromagnetic sensor spoofing as differentiable functions within PGD optimization offers a general paradigm for translating digital attacks into the physical domain.

## Limitations & Future Work

- Physical attacks remain at the simulation level and have not been validated on real sensors.
- Attack optimization requires white-box access to tracker weights; transferability to black-box settings is not explored.
- No defensive countermeasures against these attacks are proposed.
- TQF is less effective against MOTRv2 augmented with an external detector, suggesting that attack robustness across architectural variants warrants further investigation.

## Related Work & Insights

- **vs. Daedalus/Hijacking/F&F**: These are TBD-specific attacks that rely on NMS, Kalman filtering, or appearance banks; they exhibit limited or negligible effectiveness on TBP trackers. FADE directly targets query propagation and temporal memory.
- **vs. BankTweak**: BankTweak assumes direct access to the inference pipeline to inject noise into appearance features, which is impractical. FADE attacks through image-level perturbations.
- This work opens a new research direction for robustness in TBP trackers, and may inspire defenses such as dynamic query budget adjustment and hidden state integrity verification.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First dedicated adversarial attack targeting TBP trackers; TQF/TMC designs are elegant and well-motivated.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 5 SOTA trackers × 2 datasets × digital + physical attacks, with comprehensive comparisons.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, though the dense notation requires careful reading.
- Value: ⭐⭐⭐⭐ — Provides an important security warning regarding TBP tracker vulnerabilities in safety-critical applications such as autonomous driving and surveillance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] FC-Track: Overlap-Aware Post-Association Correction for Online Multi-Object Tracking](fc-track_overlap-aware_post-association_correction_for_online_multi-object_track.md)
- [\[AAAI 2026\] Rethinking Progression of Memory State in Robotic Manipulation: An Object-Centric Perspective](../../AAAI2026/video_understanding/rethinking_progression_of_memory_state_in_robotic_manipulation_an_object-centric.md)
- [\[CVPR 2026\] Occlusion-Aware SORT: Observing Occlusion for Robust Multi-Object Tracking](occlusion-aware_sort_observing_occlusion_for_robust_multi-object_tracking.md)
- [\[CVPR 2026\] TCEI: Dual-level Adaptation for Multi-Object Tracking via Test-Time Calibration](tcei_test_time_calibration_experience_intuition_mot.md)
- [\[CVPR 2026\] Dual-level Adaptation for Multi-Object Tracking: Building Test-Time Calibration from Experience and Intuition](tcei_test_time_calibration_experience_intuition_mot.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] MM-Snowball: Evaluating and Mitigating Hallucination Snowballing in Multimodal Multi-Turn Dialogue
description: >-
  [ICML 2026][Hallucination Detection][Multi-turn dialogue] This paper introduces the MM-Snowball benchmark (4,992 6-turn adversarial dialogues) to systematically characterize the "hallucination snowballing" phenomenon in…
tags:
  - "ICML 2026"
  - "Hallucination Detection"
  - "Multi-turn dialogue"
  - "Hallucination snowballing"
  - "Visual fading"
  - "Training-free rectification"
  - "Diagnostic benchmark"
date: 2026-05-08
content_hash: e2c6e69b0bbe6b50
---

# MM-Snowball: Evaluating and Mitigating Hallucination Snowballing in Multimodal Multi-Turn Dialogue

**Conference**: ICML 2026  
**arXiv**: [2606.00622](https://arxiv.org/abs/2606.00622)  
**Code**: https://frenkie-chiang.github.io/MM-Snowball (Project Page)  
**Area**: Hallucination Detection  
**Keywords**: Multi-turn dialogue, Hallucination snowballing, Visual fading, Training-free rectification, Diagnostic benchmark

## TL;DR
This paper introduces the MM-Snowball benchmark (4,992 6-turn adversarial dialogues) to systematically characterize the "hallucination snowballing" phenomenon in Multimodal Large Models during long dialogues. It designs a training-free CAVR method that refreshes visual signals at the representation layer and adjudicates text-visual conflicts at the logit layer, significantly flattening the performance collapse curve in late-stage dialogues.

## Background & Motivation
**Background**: Multimodal Large Language Models (MLLMs) have been proven high-performing on single-turn tasks such as VQA, captioning, and instruction following. However, real-world deployment scenarios are almost entirely multi-turn dialogues where users ask follow-up questions, provide corrections, or guide the model based on previous answers. Existing hallucination benchmarks like POPE, HallusionBench, and MMHal-Bench are largely limited to single-turn yes-no or MCQ settings, at most extending to a two-turn "caption-then-question" mode.

**Limitations of Prior Work**: When dialogues extend to 5–6 turns, if the model makes an error in early responses (e.g., misidentifying "two cats" as "three cats"), each subsequent turn treats this error as contextual fact for further reasoning. This transforms local perception failures into systemic cognitive delusions—a cascade named *hallucination snowballing*. Existing multi-turn benchmarks either induce hallucinations using Photoshopped "fake images" which lose real visual distributions (VisDiaHalBench) or have a 2-turn horizon that fails to observe long-term evolution (MMHalSnowball).

**Key Challenge**: Mitigation strategies for single-turn scenarios (e.g., VCD, OPERA, MemVR) are built on the implicit assumption that "textual context is clean." In long dialogues, the context itself is contaminated by previous hallucinations; applying local corrections to the decoding distribution may inadvertently strengthen the contaminated linguistic prior. The root problem is *modality decoupling* in long dialogues: the reasoning engine gradually ignores visual tokens and prioritizes internal consistency with the "dirty text history."

**Goal**: (1) Construct a truly evolutionary, 6+ turn dialogue benchmark based on real images to precisely measure the entire process of hallucination snowballing; (2) Provide a training-free rectification method compatible with mainstream MLLMs to anchor the model back to visual facts in late-stage dialogues.

**Key Insight**: The authors discovered an counter-intuitive "V-shaped" performance curve through experiments—accuracy drops sharply in turns 3–5 but rebounds significantly in turn 6 when explicitly prompted to "look at the image again carefully." This indicates that visual evidence is not "forgotten" at the weight level but is suppressed by accumulated contaminated text, and can be reactivated through explicit visual representation refreshing or logit intervention.

**Core Idea**: Construct the multi-stage 6-turn adversarial dialogue benchmark MM-Snowball using *Adversarial Hallucination Trajectory Synthesis (AHTS)*; then use *Conflict-Aware Visual Rectification (CAVR)* to "re-anchor" vision at both the representation and logit layers, upgrading single-point mitigation to dialogue-level mitigation.

## Method

### Overall Architecture
The paper advances through two main tracks. **Track 1 (Benchmark)**: Using the AHTS pipeline to generate 4,992 6-turn dialogue trajectories (totaling 29,952 OE questions) for real images $v_i$. The pipeline consists of three stages: (A) *Visual Atomic Proposition Construction* parses images into structured semantic units to establish a ground-truth state $S_{GT}$; (B) *Causal Intervention & State Perturbation* applies counterfactual perturbations to $S_{GT}$ via semantic operators to obtain a hallucination state $S_{Hall}$; (C) *Adversarial Dialogue Trajectory Simulation* involves a "deceptive attacker" and a "bifurcated responder" acting through 6 turns to push the dialogue through 5 cognitive phases: Perception Anchoring → Adversarial Bifurcation → Reasoning Escalation → Systemic Hallucination → Visual Correction. **Track 2 (Methodology - CAVR)**: A training-free rectification method applied during inference on top of any autoregressive MLLM. It targets two types of *visual fading* with dual mechanisms at the representation and logit layers, serving as a "hallucination circuit breaker."

### Key Designs

1. **AHTS Adversarial Trajectory Synthesis (Benchmark)**:
    - **Function**: Decomposes "hallucination snowballing" into controllable, labeled, and phased dialogue trajectories, allowing evaluation of when the model collapses and when it can be recovered.
    - **Mechanism**: First uses visual atomic propositions to decompose the image into object/attribute/relation triples $S_{GT}=\{(o_k,a_k,r_k)\}$. Counterfactual sets $S_{Hall}$ are generated via semantic operators (e.g., attribute replacement, object deletion, relation reversal). Then, a *Deceptive Attacker* injects a "misleading premise" consistent with $S_{Hall}$ in turn 3, while the *Bifurcated Responder* is the MLLM under test. Questions for each turn are strictly aligned with one of the five phases. Finally, Visual Fallacy Rate (VFR ↓) and Success Rate of Snowball (SRS ↑) quantify turn-by-turn collapse and cascade success.
    - **Design Motivation**: Existing multi-turn benchmarks are either too short (≤2 turns) or treat multiple turns as independent tasks, failing to characterize the continuous transmission of errors. Explicit attackers and stage labels are necessary to distinguish between "resisting the attack" and "simply making a different mistake."

2. **Representation-level Visual Rectification (RVR)**:
    - **Function**: Dynamically monitors the model's epistemic uncertainty at intermediate layers. If visual grounding is suspected to be decaying, visual features are "re-injected" into that layer to block visual fading in deep token representations.
    - **Mechanism**: Monitors uncertainty signals $U_\ell$ (e.g., based on entropy or vision/text attention ratios) at selected intermediate layers for each generation step. When $U_\ell$ exceeds a threshold, original visual token representations $h_v$ are re-written into the key-value cache of that layer (extending MemVR's "visual memory re-injection"). This process modifies no parameters and requires no extra training.
    - **Design Motivation**: Experiments show the bottom of the V-curve corresponds to a significant drop in intermediate visual attention. Logit-level remediation is often too late; visual signals must be sustained in the representation channel first.

3. **Logit-level Conflict Rectification (LCR)**:
    - **Function**: Explicitly identifies semantic conflicts between "contaminated history" and "current visual anchors" at the output distribution layer to pull the distribution back to visual facts.
    - **Mechanism**: Inspired by contrastive decoding, it constructs two distributions—one conditioned on the full dialogue history $p_\text{ctx}(y|x_{1:t})$ and one conditioned on "stripped history" (only image + current question) $p_\text{vis}(y|v,q_t)$. Discovered high-divergence token positions are identified as "conflict points." An adaptive weight $\alpha_t$, driven by RVR uncertainty, biases the distribution toward $p_\text{vis}$: $p_\text{out}(y) \propto p_\text{ctx}(y)^{1-\alpha_t}\, p_\text{vis}(y)^{\alpha_t}$. When no conflict exists, $\alpha_t \to 0$ to avoid over-intervention.
    - **Design Motivation**: While VCD/OPERA assume linguistic priors are the source of "dirt," in multi-turn settings, the dialogue history itself is the dirtier source. Explicitly modeling the "history vs. vision" conflict is essential.

### Loss & Training
CAVR is completely training-free: it updates no parameters, requires no preference data, and introduces no extra decoding heads. It attaches RVR and LCR hooks to the inference path, allowing plug-and-play use with Qwen2.5-VL, LLaVA, InternVL, etc. The benchmark involves no training beyond synthesis and manual verification.

## Key Experimental Results

### Main Results
The authors systematically compare 6-turn accuracy between open-source and proprietary MLLMs on MM-Snowball, summarizing hallucination behavior with VFR↓ and SRS↑.

| Evaluation Metric | Key Finding |
|---------|---------|
| 6-Turn Accuracy Curve | All baselines exhibit a "V-shape"—accuracy plunges after Turn 3 (adversarial bifurcation) and partially recovers after Turn 6 (visual re-prompt). |
| Mid-stage Collapse (Turn 3–5) | Major MLLMs show a 15%–30% drop in accuracy; once adversarial premises are introduced, they dominate reasoning. |
| Turn 6 Visual Re-prompt | Accuracy rebounds by 5%–15%, proving visual evidence is suppressed rather than forgotten. |
| Cross Model Scale | 7B, 32B, and 70B models are not immune; larger models simply collapse slightly later. |

Comparison of CAVR with existing mitigation strategies:

| Mitigation Method | Single-turn VQA Effect | MM-Snowball Long Dialogue Effect |
|---------|--------------|----------------------|
| VCD (Contrastive Decoding) | Effective | Late-stage collapse remains evident |
| OPERA (Summary Token Penalty) | Effective | Powerless against contaminated history |
| MemVR (Visual Re-injection) | Effective | Mitigates but Turn 5/6 still drop significantly |
| **CAVR (Ours)** | Effective | **Significantly flattens the V-curve, maintaining visual fidelity in Turn 5/6** |

### Ablation Study
| Configuration | Key Phenomenon | Insight |
|------|---------|------|
| Full CAVR | Lowest VFR and SRS in late turns | Full dual-mechanism is optimal |
| RVR Only | Reverses mid-stage collapse partially, but logits still favor dirty history | Fixing "visual fading" doesn't adjudicate conflicts |
| LCR Only | Local smoothing of conflict tokens, but deep representations are already degraded | Logit intervention is too late if representation has decayed |
| Always-on RVR | Interferes with normal tokens; overall drop | Must be gated by uncertainty and triggered on-demand |

### Key Findings
- *Visual Fading is the primary cause of snowballing*: Through attention analysis and Turn 6 re-prompting, the root cause is identified as "the model suppressing the image" rather than "forgetting it"—this distinction dictates that mitigation should refresh representations rather than re-inputting images.
- *Recoverability of the V-curve* indicates that any method reporting only the final turn will overestimate its capability. Performance must be reported turn-by-turn.
- *Training-free + Representation-layer + Logit-layer* combination allows existing MLLMs to gain multi-turn robustness without extra training costs.

## Highlights & Insights
- Explicitly decomposing "hallucination snowballing" into five cognitive phases and designing adversarial roles is a paradigm that engineers temporal cognitive failures into annotatable events, providing more diagnostic value than simply increasing turn counts.
- The rebound in Turn 6 refutes the simple narrative that "long dialogue = forgotten image." This counterfactual experiment reshapes the design direction for future mitigation work.
- The dual "representation $\to$ logit" intervention of CAVR, coupled with attention/uncertainty gating, is a natural extension of single-layer mitigators like MemVR/VCD for multi-turn settings.

## Limitations & Future Work
- AHTS uses LLMs as deceptive attackers; there may be a distribution gap between these adversarial strategies and real-user misleading behavior.
- Details regarding RVR/LCR trigger thresholds and compatibility across different MLLM architectures are limited in the paper body; code reference is required.
- CAVR is an inference-time intervention and cannot "cure" models fundamentally biased during training (e.g., via instruction tuning data that devalues vision).
- Evaluation is capped at 6 turns; whether logit bias magnitudes need dynamic adjustment for longer spans (>10 turns) remains to be verified.

## Related Work & Insights
- **vs. MMHalSnowball (zhong2024)**: Also focuses on snowballing but is limited to 2-turn caption+VQA; this work uses 6-turn evolvable dialogues with phase labels.
- **vs. VisDiaHalBench (cao2024)**: Uses multi-turn dialogues but relies on edited/synthetic images with artifacts; this work uses real images + visual atomic propositions to isolate dialogue-level failures.
- **vs. VCD / OPERA / MemVR**: Single-turn mitigators assume clean context; CAVR handles scenarios where context is contaminated by the model's own previous answers.

## Rating
- Novelty: ⭐⭐⭐⭐ First 6-turn scalable evolutionary benchmark + training-free dual-layer mitigation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple open/proprietary MLLMs and mitigation strategies.
- Writing Quality: ⭐⭐⭐⭐ Logical chain from phenomenon (V-curve) to attribution (visual fading) to method (CAVR).
- Value: ⭐⭐⭐⭐ Immediately usable diagnosis and plugin for multi-turn MLLM deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] MUG: Multi-agent Undercover Gaming — Hallucination Removal via Counterfactual Test for Multimodal Reasoning](../../AAAI2026/hallucination/multi-agent_undercover_gaming_hallucination_removal_via_coun.md)
- [\[CVPR 2026\] TriDF: Evaluating Perception, Detection, and Hallucination for Interpretable DeepFake Detection](../../CVPR2026/hallucination/tridf_evaluating_perception_detection_and_hallucination_for_interpretable_deepfa.md)
- [\[CVPR 2026\] KVSmooth: Mitigating Hallucination in Multi-modal Large Language Models through Key-Value Smoothing](../../CVPR2026/hallucination/kvsmooth_mitigating_hallucination_in_multi-modal_large_language_models_through_k.md)
- [\[ICML 2026\] Learning from Fine-Grained Visual Discrepancies: Mitigating Multimodal Hallucinations via In-Context Visual Contrastive Optimization](learning_from_fine-grained_visual_discrepancies_mitigating_multimodal_hallucinat.md)
- [\[ACL 2026\] Dialectic-Med: Mitigating Diagnostic Hallucinations via Counterfactual Adversarial Multi-Agent Debate](../../ACL2026/hallucination/dialectic-med_mitigating_diagnostic_hallucinations_via_counterfactual_adversaria.md)

</div>

<!-- RELATED:END -->

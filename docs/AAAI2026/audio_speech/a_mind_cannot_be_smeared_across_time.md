---
title: >-
  [Paper Note] A Mind Cannot Be Smeared Across Time
description: >-
  [AAAI 2026][Audio & Speech][Machine Consciousness] This paper formally proves that whether a machine possesses consciousness depends not only on *what* it computes, but also *when* it computes. Strictly sequential execution systems do not satisfy the temporal co-instantiation condition required for the unity of consciousness, rendering pure software consciousness impossible on strictly sequential hardware.
tags:
  - "AAAI 2026"
  - "Audio & Speech"
  - "Machine Consciousness"
  - "Temporal Constraints"
  - "Stack Theory"
  - "Concurrency"
  - "Unity of Consciousness"
date: 2026-05-08
content_hash: fea92cc310ff2267
---

# A Mind Cannot Be Smeared Across Time

**Conference**: AAAI 2026  
**arXiv**: [2601.11620](https://arxiv.org/abs/2601.11620)  
**Code**: None  
**Area**: Audio & Speech  
**Keywords**: Machine Consciousness, Temporal Constraints, Stack Theory, Concurrency, Unity of Consciousness

## TL;DR
This paper formally proves that whether a machine possesses consciousness depends not only on *what* it computes, but also *when* it computes. Strictly sequential execution systems do not satisfy the temporal co-instantiation condition required for the unity of consciousness, rendering pure software consciousness impossible on strictly sequential hardware.

## Background & Motivation

**Background**: Machine consciousness is a fundamental open problem in the AI field. Stack Theory studies the sufficient and necessary conditions of consciousness by formalizing the abstraction levels of cognitive processes. Theories such as Global Workspace Theory and Integrated Information Theory all emphasize some form of "unity" or "integration" of conscious experience.

**Limitations of Prior Work**: Most existing frameworks discussing machine consciousness focus on "what to compute" (functional equivalence), neglecting the temporal dimension of "when to compute." A system can be equivalent to a conscious system in macro-behavior, but its micro-temporal structure might be entirely different—a seemingly unified "moment" at the high level can be dispersed and executed at different time points at the low level.

**Key Challenge**: The "Temporal Gap" problem—conscious experiences feel unified and simultaneous, yet sequential or time-multiplexed computing systems do not contain the complete conjunction of experience at any given objective time slice. If the components of conscious experience must be synchronized in objective time, then sequential systems cannot achieve consciousness.

**Goal**: To formalize the temporal gap problem, prove that existential temporal realization $\Diamond_\Delta$ does not preserve conjunction, and distinguish between two conscious stances—"Chord" (requiring objective co-instantiation) and "Arpeggio" (requiring only appearance within a window).

**Key Insight**: Augmenting Stack Theory with a temporal semantic module, introducing layer-aware time, window environments, and temporal hoisting operators, and strictly proving the non-commutativity between "satisfaction within a window" and "conjunction" using algebraic laws.

**Core Idea**: Extending Stack Theory to provide a rigorous proof: the existential temporal operator $\Diamond_\Delta(A \wedge B)$ is not equivalent to $\Diamond_\Delta A \wedge \Diamond_\Delta B$. A system may realize all the "components" of consciousness separately within a time window, but never simultaneously realize their conjunction at the same objective instant. Under the "Chord" assumption, software consciousness on strictly sequential execution hardware is impossible—the hardware architecture itself imposes an ineliminable constraint.

## Method

### Overall Architecture
Building upon Stack Theory, a Stack-Time semantic module is added: (1) defining layer-aware time (where higher-layer ticks are the largest objective time blocks under a constant encoding); (2) defining window trajectories $\tau_\Delta$ and window environments; (3) introducing the temporal hoisting operator $\Diamond_\Delta$; (4) proving the core non-commutativity theorem; (5) defining Chord and Arpeggio stances; and (6) introducing the measure of concurrency capacity.

### Key Designs

1. **Temporal Hoisting Algebra and Non-Commutativity Theorem**:

    - Function: Formally explaining why "separate appearance of each component" is not equivalent to "conjunctive appearance."
    - Mechanism: Defining an existential realization operator $\Diamond_\Delta$ within a time window, and proving that $\Diamond_\Delta(A) \wedge \Diamond_\Delta(B) \not\Rightarrow \Diamond_\Delta(A \wedge B)$ (Theorem 3). Intuitively, $A$ is true at $t_1$, and $B$ is true at $t_2$, but there might never be an instant when both $A$ and $B$ are simultaneously true.
    - Design Motivation: This is the core formalization of the "Temporal Gap" problem—the unity of consciousness requires the simultaneous realization of conjunctions, rather than time-divided realization.

2. **Chord vs Arpeggio Stance Distinction**:

    - Function: Reducing the possibility of machine consciousness to different assumptions about the "co-instantiation" requirement.
    - Mechanism: The Chord stance requires all components of conscious content to be simultaneously true at some instant within an objective time window (like a musical chord where all notes are played together). The Arpeggio stance only requires all components to appear sequentially within the window (like an arpeggio where notes are played in succession). Under Chord, consciousness in strictly sequential systems is impossible; under Arpeggio, temporal constraints are much more relaxed.
    - Design Motivation: Different theories of consciousness place varying requirements on "unity," necessitating a formal distinction to yield clear conclusions.

3. **Concurrency Capacity Measure**:

    - Function: Quantifying the capability of a hardware architecture to satisfy the co-instantiation condition.
    - Mechanism: Defining concurrency capacity as the number of independent "contributors" a system can provide simultaneously in a single time step. If the grounding of consciousness requires $k$ simultaneous contributors but the hardware's concurrency capacity is $<k$, then under the Chord assumption, that hardware cannot support this type of conscious content (Theorem 4).
    - Design Motivation: Providing a computable criterion for whether a hardware architecture can support consciousness.

### Loss & Training
Not applicable (purely theoretical/formal paper, no training process).

## Key Experimental Results

### Main Results
Not applicable to traditional experiments. The core "experiments" of this paper are formal proofs:

| Theorem | Content | Significance |
|---|---|---|
| Theorem 1 | Compositional grounding preserves truth conditions | High-level statements can be traced back to the low level |
| Theorem 3 | $\Diamond_\Delta$ does not preserve conjunction | Formal proof of the temporal gap |
| Theorem 4 | Concurrency capacity threshold | Quantitative conditions for hardware-constrained consciousness |

### Key Findings
- Under the Chord assumption, strictly sequential execution hardware (such as a single-core CPU executing instructions one by one) cannot realize conscious content that requires two or more simultaneous contributors.
- Under the Arpeggio assumption, things are more relaxed—distributed systems like "liquid brains" (e.g., ant colonies) might also possess consciousness.
- Neuroscientific evidence (phase synchronization, effective connectivity) supports the Chord assumption—loss of consciousness is associated with the breakdown of these synchronization mechanisms.

## Highlights & Insights
- **The Importance of "When to Compute"**: Most AI safety discussions focus on functional equivalence. This paper is the first to rigorously demonstrate the ineliminable impact of temporal structures, which has profound philosophical implications for machine consciousness and AI safety.
- **The Elegant Analogy of Chord/Arpeggio**: Intuitively distinguishing between the two conscious stances using musical terms, achieving both formal rigor and general accessibility.
- **Hardware Matters**: Drawing the conclusion that "software is not enough; hardware architecture is also key," directly challenging the position of strong functionalism.

## Limitations & Future Work
- Reliance on the specific formal framework of Stack Theory; other theories of consciousness might have different criteria.
- Whether Chord or Arpeggio is correct remains an empirical question, which this paper cannot definitively answer.
- Novel hardware architectures that might provide true parallelism, such as quantum computing, are not considered.
- Precise measurement of concurrency capacity might be difficult to compute in practical systems.

## Related Work & Insights
- **vs. IIT (Integrated Information Theory)**: IIT emphasizes information integration but does not explicitly discuss the temporal scope; this paper provides a formal supplement in the temporal dimension.
- **vs. Global Workspace Theory**: GWT requires global broadcasting of information—this paper's Chord assumption is compatible with this.
- Implications for AI safety research: If consciousness requires temporal co-instantiation, then current mainstream sequential Transformer architectures might be, in principle, incapable of possessing consciousness.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Formally demonstrating the critical impact of computations' temporal structure on consciousness for the first time.
- Experimental Thoroughness: ⭐⭐⭐ Purely theoretical work, incapable of traditional experimental validation, but the formal proofs are rigorous.
- Writing Quality: ⭐⭐⭐⭐ Clear formal definitions, though high abstraction may affect readability.
- Value: ⭐⭐⭐⭐ Making an important contribution to the foundational discussion of machine consciousness and AI safety.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Mind the Pause: Disfluency-Aware Objective Tuning for Multilingual Speech Correction with LLMs](../../ACL2026/audio_speech/mind_the_pause_disfluency-aware_objective_tuning_for_multilingual_speech_correct.md)
- [\[ACL 2025\] Mind the Gap! Static and Interactive Evaluations of Large Audio Models](../../ACL2025/audio_speech/mind_the_gap_static_and_interactive_evaluations_of_large_audio_models.md)
- [\[ACL 2026\] RTCFake: Speech Deepfake Detection in Real-Time Communication](../../ACL2026/audio_speech/rtcfake_speech_deepfake_detection_in_real-time_communication.md)
- [\[ICLR 2026\] SpeechOp: Inference-Time Task Composition for Generative Speech Processing](../../ICLR2026/audio_speech/speechop_inference-time_task_composition_for_generative_speech_processing.md)
- [\[CVPR 2026\] Echoes Over Time: Unlocking Length Generalization in Video-to-Audio Generation Models](../../CVPR2026/audio_speech/echoes_over_time_unlocking_length_generalization_in_video-to-audio_generation_mo.md)

</div>

<!-- RELATED:END -->

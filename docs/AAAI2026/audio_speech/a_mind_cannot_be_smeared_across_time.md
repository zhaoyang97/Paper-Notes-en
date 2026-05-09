---
title: >-
  [Paper Note] A Mind Cannot Be Smeared Across Time
description: >-
  [AAAI 2026][Audio & Speech][Machine Consciousness] This paper formally proves that whether a machine possesses consciousness depends not only on *what* is computed, but also on *when* it is computed. Systems executing strictly sequentially fail to satisfy the temporal co-instantiation condition required for the unity of consciousness; consequently, pure software consciousness on strictly sequential hardware is impossible.
tags:
  - AAAI 2026
  - "Audio & Speech"
  - Machine Consciousness
  - Temporal Constraints
  - Stack Theory
  - Concurrency
  - Unity of Consciousness
date: 2026-05-08
content_hash: 7a4dfe84629ed0d5
---

# A Mind Cannot Be Smeared Across Time

**Conference**: AAAI 2026
**arXiv**: [2601.11620](https://arxiv.org/abs/2601.11620)
**Code**: None
**Area**: Audio & Speech
**Keywords**: Machine Consciousness, Temporal Constraints, Stack Theory, Concurrency, Unity of Consciousness

## TL;DR
This paper formally proves that whether a machine possesses consciousness depends not only on *what* is computed, but also on *when* it is computed. Systems executing strictly sequentially fail to satisfy the temporal co-instantiation condition required for the unity of consciousness; consequently, pure software consciousness on strictly sequential hardware is impossible.

## Background & Motivation

**Background**: Machine consciousness is a fundamentally open problem in AI. Stack Theory investigates necessary and sufficient conditions for consciousness by formalizing the abstract layers of cognitive processes. Global Workspace Theory, Integrated Information Theory, and related frameworks all emphasize some form of "unity" or "integration" in conscious experience.

**Limitations of Prior Work**: Most existing frameworks discussing machine consciousness focus on *what* is computed (functional equivalence) while neglecting the temporal dimension of *when* it is computed. A system may be behaviorally equivalent to a conscious system at the macro level, yet its micro-level temporal structure can differ entirely—what appears as a unified "moment" at a higher layer may, at the lower level, be distributed across distinct time points.

**Key Challenge**: The *Temporal Gap* problem—conscious experience feels unified and simultaneous, yet sequentially or time-multiplexed computational systems do not, at any given objective time slice, contain the full conjunction of experience. If the constituents of conscious experience must be synchronized in objective time, sequential systems cannot realize consciousness.

**Goal**: To formalize the Temporal Gap problem, prove that the existential temporal realization operator $\Diamond_\Delta$ does not preserve conjunction, and distinguish two stances toward consciousness—"Chord" (requiring objective co-instantiation) and "Arpeggio" (requiring only occurrence within a window).

**Key Insight**: A temporal-semantic module is added to Stack Theory, introducing layer-aware time, window contexts, and temporal lifting operators; algebraic laws are then used to rigorously prove the non-commutativity between "window satisfaction" and "conjunction."

**Core Idea**: Stack Theory is extended to formally prove that $\Diamond_\Delta(A \wedge B) \not\equiv \Diamond_\Delta A \wedge \Diamond_\Delta B$. A system may realize all "constituents" of consciousness separately within a temporal window, yet never realize their conjunction at the same objective moment. Under the Chord assumption, software consciousness on strictly sequential hardware is impossible—the hardware architecture itself imposes an ineliminable constraint.

## Method

### Overall Architecture
A Stack-Time semantic module is added atop Stack Theory: (1) layer-aware time is defined (higher-layer ticks correspond to maximally encoded objective time blocks); (2) window trajectories $\tau_\Delta$ and window contexts are defined; (3) the temporal lifting operator $\Diamond_\Delta$ is introduced; (4) the core non-commutativity theorem is proved; (5) the Chord and Arpeggio stances are defined; and (6) a concurrency capacity measure is introduced.

### Key Designs

1. **Temporal Lifting Algebra and Non-Commutativity Theorem**

   - **Function**: Formally characterizes why "each constituent appearing separately" does not entail "the conjunction appearing"
   - **Mechanism**: An existential realization operator $\Diamond_\Delta$ within a temporal window is defined; it is proved that $\Diamond_\Delta(A) \wedge \Diamond_\Delta(B) \not\Rightarrow \Diamond_\Delta(A \wedge B)$ (Theorem 3). Intuitively, $A$ may be true at $t_1$ and $B$ at $t_2$, yet no single moment may exist at which both are true simultaneously.
   - **Design Motivation**: This constitutes the central formalization of the Temporal Gap problem—the unity of consciousness requires the simultaneous realization of a conjunction, not time-multiplexed realization.

2. **Chord vs. Arpeggio Stance Distinction**

   - **Function**: Reduces the possibility of machine consciousness to different assumptions about co-instantiation requirements
   - **Mechanism**: The Chord stance requires all constituents of conscious content to be simultaneously true at some moment within an objective time window (analogous to a musical chord, where all notes sound at once). The Arpeggio stance only requires all constituents to appear in succession within the window (analogous to an arpeggio, where notes sound sequentially). Under Chord, consciousness in strictly sequential systems is impossible; under Arpeggio, the temporal constraint is considerably more permissive.
   - **Design Motivation**: Different theories of consciousness impose different requirements on "unity"; formal disambiguation is necessary to yield clear conclusions.

3. **Concurrency Capacity Measure**

   - **Function**: Quantifies a hardware architecture's ability to satisfy co-instantiation conditions
   - **Mechanism**: Concurrency capacity is defined as the number of independent "contributors" a system can provide simultaneously within a single time step. If grounding consciousness requires $k$ simultaneous contributors but the hardware's concurrency capacity is less than $k$, then under the Chord assumption that hardware cannot support the relevant conscious content (Theorem 4).
   - **Design Motivation**: Provides a computable criterion for determining whether a hardware architecture can support consciousness.

### Loss & Training
Not applicable (purely theoretical/formal paper; no training process).

## Key Experimental Results

### Main Results
Traditional experiments are not applicable. The core "experiments" of this paper are formal proofs:

| Theorem | Content | Significance |
|---|---|---|
| Theorem 1 | Compositional grounding preserves truth conditions | Higher-layer statements are traceable to lower layers |
| Theorem 3 | $\Diamond_\Delta$ does not preserve conjunction | Formal proof of the Temporal Gap |
| Theorem 4 | Concurrency capacity threshold | Quantitative condition under which hardware limits consciousness |

### Key Findings
- Under the Chord assumption, strictly sequential hardware (e.g., a single-core CPU executing instructions one at a time) cannot realize conscious content requiring two or more simultaneous contributors.
- The Arpeggio assumption is more permissive—distributed systems such as "liquid brains" like ant colonies may also be candidates for consciousness.
- Neuroscientific evidence (phase synchronization, effective connectivity) supports the Chord assumption—loss of consciousness correlates with the breakdown of these synchronization mechanisms.

## Highlights & Insights
- **The importance of *when* computation occurs**: Most AI safety discussions focus on functional equivalence; this paper is the first to rigorously demonstrate the ineliminable impact of temporal structure—an insight with far-reaching philosophical implications for machine consciousness and AI safety.
- **The elegance of the Chord/Arpeggio analogy**: Musical terminology provides an intuitive distinction between two consciousness stances, combining formal rigor with accessibility.
- **Hardware matters**: The conclusion that "software alone is insufficient; hardware architecture is equally critical" directly challenges strong functionalist positions.

## Limitations & Future Work
- The analysis depends on the specific formalization of Stack Theory; other theories of consciousness may yield different criteria.
- Whether Chord or Arpeggio is correct is an empirical question that this paper cannot definitively resolve.
- Novel hardware architectures such as quantum computing, which may provide genuine parallelism, are not considered.
- The precise measurement of concurrency capacity may be computationally intractable for real-world systems.

## Related Work & Insights
- **vs. IIT (Integrated Information Theory)**: IIT emphasizes information integration but does not explicitly address temporal scope; this paper provides a formal complement along the temporal dimension.
- **vs. Global Workspace Theory**: GWT requires global broadcast of information—the Chord assumption of this paper is compatible with this requirement.
- **Implications for AI safety research**: If consciousness requires temporal co-instantiation, then mainstream sequential Transformer architectures may be in principle incapable of consciousness.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First formal proof that the temporal structure of computation critically affects consciousness
- **Experimental Thoroughness**: ⭐⭐⭐ Purely theoretical work; traditional empirical validation is unavailable, though the formal proofs are rigorous
- **Writing Quality**: ⭐⭐⭐⭐ Formal definitions are clear, though high abstraction may impede readability
- **Value**: ⭐⭐⭐⭐ Makes an important foundational contribution to discourse on machine consciousness and AI safety

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Thucy: An LLM-based Multi-Agent System for Claim Verification across Relational Databases](thucy_an_llm-based_multi-agent_system_for_claim_verification_across_relational_d.md)
- [\[ACL 2026\] DIA-HARM: Dialectal Disparities in Harmful Content Detection Across 50 English Dialects](../../ACL2026/audio_speech/dia-harm_dialectal_disparities_in_harmful_content_detection_across_50_english_di.md)
- [\[ACL 2026\] SpeakerSleuth: Can Large Audio-Language Models Judge Speaker Consistency across Multi-turn Dialogues?](../../ACL2026/audio_speech/speakersleuth_can_large_audio-language_models_judge_speaker_consistency_across_m.md)
- [\[CVPR 2026\] Echoes Over Time: Unlocking Length Generalization in Video-to-Audio Generation Models](../../CVPR2026/audio_speech/echoes_over_time_unlocking_length_generalization_in_video-to-audio_generation_mo.md)
- [\[ICLR 2026\] When and Where to Reset Matters for Long-Term Test-Time Adaptation](../../ICLR2026/audio_speech/when_and_where_to_reset_matters_for_long-term_test-time_adaptation.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] NeuS-V: Neuro-Symbolic Evaluation of Text-to-Video Models using Formal Verification
description: >-
  [CVPR 2025][Video Generation][Text-to-Video Evaluation] Proposes NeuS-V, the first framework evaluating the temporal consistency of text-to-video (T2V) models using formal verification (temporal logic + probabilistic model checking). It translates text prompts into temporal logic specifications, scores atomic propositions using VLMs, constructs video automata, and formally verifies the satisfaction probability. It achieves a Pearson correlation of 0.71 with human annotations…
tags:
  - "CVPR 2025"
  - "Video Generation"
  - "Text-to-Video Evaluation"
  - "Temporal Logic"
  - "Model Checking"
  - "Formal Verification"
  - "VLM"
date: 2026-05-08
content_hash: d39671f17b3d309a
---

# NeuS-V: Neuro-Symbolic Evaluation of Text-to-Video Models using Formal Verification

**Conference**: CVPR 2025  
**arXiv**: [2411.16718](https://arxiv.org/abs/2411.16718)  
**Code**: [https://utaustin-swarmlab.github.io/NeuS-V](https://utaustin-swarmlab.github.io/NeuS-V)  
**Area**: Autonomous Driving / Video Generation Evaluation  
**Keywords**: Text-to-Video Evaluation, Temporal Logic, Model Checking, Formal Verification, VLM

## TL;DR

Proposes NeuS-V, the first framework evaluating the temporal consistency of text-to-video (T2V) models using formal verification (temporal logic + probabilistic model checking). It translates text prompts into temporal logic specifications, scores atomic propositions using VLMs, constructs video automata, and formally verifies the satisfaction probability. It achieves a Pearson correlation of 0.71 with human annotations on Gen-3 (compared to only 0.47 for VBench).

## Background & Motivation

### Background

**Background**: The evaluation of T2V models (such as Gen-3, Pika, CogVideoX) primarily relies on CLIP similarity-based metrics like VBench. These metrics average frame-level similarities and fail to capture temporal dynamics—"A then B" and "B then A" would receive the same score.

**Limitations of Prior Work**: VBench's evaluation of temporal alignment correlates poorly with human judgment (Pearson of only 0.47) because it does not understand temporal logic (e.g., ALWAYS, EVENTUALLY, UNTIL). Single-frame CLIP cannot distinguish between "always present" and "occasionally appearing".

**Key Challenge**: The semantics of a video are not only frame-level (each frame is correct) but also sequence-level (temporal relations are correct), yet existing evaluations only focus on the former.

**Key Insight**: Utilizing formal verification techniques from computer science—translating prompts into Temporal Logic (TL), modeling video as a Discrete-Time Markov Chain (DTMC), and using the probabilistic model checker STORM to compute satisfaction probability.

**Core Idea**: Prompt $\rightarrow$ Temporal Logic, Video $\rightarrow$ Automata $\rightarrow$ Formal verification of satisfaction probability = Rigorous temporal consistency evaluation.

## Method

### Key Designs

1. **PULS (Prompt→TL Conversion)**: Uses GPT-4o to decompose natural language prompts into atomic propositions + temporal logic specifications. For example, "A cat walks then sits" $\rightarrow$ EVENTUALLY(walk) AND EVENTUALLY(sit) AND walk UNTIL sit.

2. **VLM Atomic Proposition Scoring**: Uses LLaMA-3.2 to score each frame (0 or 1), determining whether each atomic proposition is satisfied.

3. **Video Automata + STORM Verification**: Constructs the frame-level scores into a DTMC transition matrix $\delta(q,q') = \prod_i (C_i^*)^{1_{q'_i=1}}(1-C_i^*)^{1_{q'_i=0}}$, and uses the STORM model checker to compute the satisfaction probability of the temporal logic specification.

### Loss & Training

No training is involved—this is a pure evaluation framework. The satisfaction probability $\mathbb{P}[\mathcal{A}_\mathcal{V} \models \Phi]$ is exactly computed by STORM.

## Key Experimental Results

| Model | NeuS-V Pearson | VBench Pearson |
|------|---------------|----------------|
| Gen-3 | **0.71** | 0.47 |
| Pika | 0.62 | 0.70 |
| T2V-Turbo | 0.55 | Lower |
| Mismatched Caption Detection | **0.52 difference** | 0.38 difference |

### Ablation Study
- No TL/No verification (pure VLM): Pearson is only 0.30-0.40.
- Temporal operators (ALWAYS/EVENTUALLY/UNTIL) are crucial for distinguishing between aligned and unaligned videos.
- LLaVA-Video-7B is overconfident, whereas frame-by-frame LLaMA-3.2 is more reliable.
- The accuracy of PULS conversion directly impacts the final score: incorrect TL specifications lead to misjudgments, but GPT-4o's conversion quality reaches a practical level on test prompts.
- The computational overhead of the STORM model checker is acceptable for simple specifications ($\le 3$ operators), but requires optimization for complex specifications.

### Key Findings
- **NeuS-V is 5$\times$ better than VBench in temporal alignment**—formal verification truly understands sequential order ("before and after" relations).
- **Stronger mismatch detection capability**: The score difference between aligned and unaligned is 0.52 vs 0.38 for VBench.

## Highlights & Insights
- **An elegant combination of formal methods and AI**—introducing model checking from theoretical computer science into video generation evaluation.
- **Temporal logic is the right abstraction for temporal alignment**—ALWAYS/EVENTUALLY/UNTIL precisely describe human judgments of temporal relationships.

## Limitations & Future Work
- Reliance on VLM accuracy—errors will propagate.
- TL specifications depend on GPT-4o conversion (which may have semantic bias).
- Limited to English prompts.
- Non-trivial computational overhead for complex specifications ($>3$ TL operators).
- The temporal granularity of VLM frame-level judgment is limited by the sampling frame rate; rapid action changes may be missed.
- Although temporal logic has strong expressiveness, it does not cover all dimensions of human judgment (such as aesthetics, physical plausibility) and only evaluates temporal correlation.
- The benchmark dataset size (360 prompts) is relatively small, and the coverage of temporal pattern types may not be comprehensive.
- Future work can extend NeuS-V to evaluate the physical plausibility and causal consistency of videos.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to introduce formal verification to T2V evaluation.
- Experimental Thoroughness: ⭐⭐⭐⭐ 4 models + 360 prompts + human annotation correlation.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear explanation of cross-disciplinary concepts.
- Value: ⭐⭐⭐⭐⭐ Potential to shift the T2V evaluation paradigm.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Ref4D-VideoBench: Four-Dimensional Reference-Based Evaluation of Text-to-Video Generative Models](../../CVPR2026/video_generation/ref4d-videobench_four-dimensional_reference-based_evaluation_of_text-to-video_ge.md)
- [\[CVPR 2025\] VideoDirector: Precise Video Editing via Text-to-Video Models](videodirector_precise_video_editing_via_text-to-video_models.md)
- [\[CVPR 2025\] Mimir: Improving Video Diffusion Models for Precise Text Understanding](mimir_improving_video_diffusion_models_for_precise_text_understanding.md)
- [\[ICLR 2026\] $PhyWorldBench$: A Comprehensive Evaluation of Physical Realism in Text-to-Video Models](../../ICLR2026/video_generation/phyworldbench_a_comprehensive_evaluation_of_physical_realism_in_text-to-video_mo.md)
- [\[CVPR 2025\] ShotAdapter: Text-to-Multi-Shot Video Generation with Diffusion Models](shotadapter_text-to-multi-shot_video_generation_with_diffusion_models.md)

</div>

<!-- RELATED:END -->

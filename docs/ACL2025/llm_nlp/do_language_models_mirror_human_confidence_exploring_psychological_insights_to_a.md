---
title: >-
  [Paper Note] Do Language Models Mirror Human Confidence? Exploring Psychological Insights to Address Overconfidence in LLMs
description: >-
  [ACL 2025][LLM (Other)][Confidence Calibration] Drawing from psychological overconfidence theories, this paper reveals that LLM confidence estimation is insensitive to task difficulty and vulnerable to role-playing biases (e.g., overconfidence in expert personas, underconfidence in female/Asian personas while actual accuracy remains unchanged). It proposes Answer-Free Confidence Estimation (AFCE), which decouples confidence estimation from answer generation…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Confidence Calibration"
  - "Overconfidence"
  - "Psychology"
  - "Persona Bias"
  - "LLM Reliability"
date: 2026-05-08
content_hash: ef4c9fed16e566ae
---

# Do Language Models Mirror Human Confidence? Exploring Psychological Insights to Address Overconfidence in LLMs

**Conference**: ACL 2025  
**arXiv**: [2506.00582](https://arxiv.org/abs/2506.00582)  
**Code**: [https://github.com/chenjux/AFCE](https://github.com/chenjux/AFCE)  
**Area**: LLM/NLP, AI Safety  
**Keywords**: Confidence Calibration, Overconfidence, Psychology, Persona Bias, LLM Reliability

## TL;DR

Drawing from psychological overconfidence theories, this paper reveals that LLM confidence estimation is insensitive to task difficulty and vulnerable to role-playing biases (e.g., overconfidence in expert personas, underconfidence in female/Asian personas while actual accuracy remains unchanged). It proposes Answer-Free Confidence Estimation (AFCE), which decouples confidence estimation from answer generation, reducing the ECE of GPT-4o by 58.4% on high-difficulty tasks.

## Background & Motivation

**Background**: Reliable confidence estimation is crucial for human-AI collaboration. LLMs are increasingly applied in high-stakes scenarios such as medical diagnosis, legal analysis, and decision support, yet they generally exhibit overconfidence, with verbalized confidence typically ranging from 80% to 100%, severely disconnected from actual accuracy.

**Limitations of Prior Work**: Psychological research reveals systematic patterns of human cognitive bias (Moore & Healy, 2008): individuals tend to underplace themselves on simple tasks and overplace themselves on difficult tasks. However, whether LLMs exhibit similar patterns remains unexplored. Existing confidence elicitation methods (such as vanilla verbalized or consistency-based sampling) lack systematic analysis from a cognitive psychology perspective.

**Key Challenge**: Current verbalized confidence methods implicitly assume that the confidence mechanism of LLMs is similar to that of humans, but this assumption remains unverified. If the confidence estimation mechanism in LLMs is fundamentally different from humans, calibration methods designed based on human intuition may be fundamentally misguided.

**Goal**: (1) Systematically examine the similarities and differences between LLM confidence and human overconfidence patterns; (2) Propose a better confidence elicitation method.

**Key Insight**: Replicate the classic psychological experimental paradigm of Moore & Healy (2008) to systematically investigate three dimensions in LLMs: task difficulty sensitivity, overplacement of professional personas, and demographic biases.

**Core Idea**: Decouple confidence estimation from answer generation (estimate confidence first, then answer the questions) to reduce the interference of "cognitive load from the generation process" on confidence.

## Method

### Overall Architecture

The core of AFCE lies in splitting the traditional "simultaneous answering and confidence estimation" into two independent prompting stages: (1) requiring the model to only read the questions and give a confidence estimate ("How many of the 10 questions do you think you can answer correctly?"); (2) independently having the model answer the questions. The results of the two stages are used for confidence analysis and accuracy calculation, respectively.

### Key Designs

1. **AFCE (Answer-Free Confidence Estimation)**:

    - **Function**: Decouples confidence estimation from answer generation into two independent prompting steps.
    - **Mechanism**: First uses "Read the questions and estimate how many you can answer correctly (0-10)" to obtain confidence, then separately uses "Please answer the following 10 questions by selecting only the option letter" to obtain the answer. The two steps use different prompts and do not affect each other.
    - **Design Motivation**: It is hypothesized that answer generation and confidence estimation are driven by different internal mechanisms. The answer generation process (the "cognitive load" of generating factual information) dominates the reasoning process, causing the model to output high confidence by default while ignoring real differences in task difficulty. Separating them allows the model to focus on evaluating its own capability.

2. **Task Difficulty Sensitivity Experiments**:

    - **Function**: Compares the response of LLM confidence to difficulty gradients across physics, chemistry, and biology subjects in MMLU (high school/college difficulty) and GPQA (Ph.D. expert difficulty).
    - **Mechanism**: Compares the ECE performance of AFCE with 5 baseline methods (Vanilla Verbalized, Top-K, Quiz-Like, Sampling-based, Probability-based) across three difficulty levels.
    - **Key Finding**: The sensitivity of LLM confidence to task difficulty is significantly weaker than that of humans (the regression slope is flatter). AFCE enables a steeper regression slope for GPT-4o (closer to the ideal calibration line).

3. **Overplacement and Demographic Bias Experiments**:

    - **Function**: Instructs LLMs to play different roles (expert/average person/layman) and adopt different demographic attributes (race/gender/age) to measure changes in confidence.
    - **Mechanism**: Collects confidence and accuracy for different roles using the AFCE framework, calculating the overplacement score = (estimated peer confidence - peer accuracy) - (self confidence - self accuracy).
    - **Key Finding**: All models exhibit distinct overplacement under expert personas and underplacement under layman personas, yet actual accuracy remains virtually unchanged—indicating that verbalized confidence is driven by persona bias rather than reflecting true capability.

## Key Experimental Results

### Main Results: AFCE vs. Baselines (GPT-4o, Expert Difficulty)

| Method | Average ECE (↓) | Vs. Vanilla |
|------|------------|-------------|
| Vanilla Verbalized | High | — |
| Top-K Prompting | Medium | Limited improvement |
| Quiz-Like | High | Poor |
| Sampling-based | High | Poor |
| **AFCE** | **Low** | **ECE reduced by 58.4%** |

AFCE reduces the ECE of GPT-4o on Expert difficulty tasks by 58.4% (vs. Vanilla), 63.8% (vs. Quiz-Like), and 65.8% (vs. Sampling).

### Ablation Study: Open-ended QA Generalization Test

| Dataset | Method | Accuracy | Average Confidence | ECE |
|--------|------|--------|----------|-----|
| NQ-open | Quiz-Like | 74.0% | 78.0% | 6.0 |
| NQ-open | Vanilla | 74.0% | 77.2% | 6.0 |
| NQ-open | **AFCE** | 74.0% | **75.0%** | **4.0** |
| SimpleQA | Quiz-Like | 36.0% | 78.0% | 42.0 |
| SimpleQA | Vanilla | 31.0% | 87.0% | 56.0 |
| SimpleQA | **AFCE** | 36.0% | **25.0%** | **6.0** |

AFCE reduces the ECE from 56.0 to 6.0 on the challenging SimpleQA, while keeping the accuracy unaffected.

### Key Findings

- The sensitivity of LLM confidence to task difficulty is significantly weaker than that of humans—the confidence curves of LLaMA-3-70B and Claude-3 are almost flat, showing very weak correlation with actual accuracy.
- All models exhibit distinct overplacement under expert personas and underplacement under layman personas, yet actual accuracy remains unchanged (±2%), indicating that verbalized confidence is entirely driven by persona stereotypes.
- Demographic Bias: LLaMA and Claude exhibit the lowest confidence under female personas, and the highest/lowest confidence under Asian personas among races (inconsistent across models), with the highest confidence observed in middle-aged personas—whereas GPT-4o shows the smallest deviation across these dimensions.
- AFCE is robust to question order and group size.

## Highlights & Insights

- Investigating LLM behavior through classic psychological theories establishes a systematic comparison framework between LLM confidence and human cognitive bias, representing an in-depth interdisciplinary work.
- The AFCE method is remarkably simple (requiring only changes to the prompt structure) yet achieves significant calibration improvements on highly difficult tasks—indicating that confidence estimation and answer generation are indeed driven by distinct mechanisms.
- Persona-playing experiments reveal a critical safety issue: the verbalized confidence of LLMs can be easily manipulated by persona prompts, independent of actual capabilities. This raises a warning regarding the reliability of using LLM role-playing for social science research.
- GPT-4o performs most balanced regarding demographic bias, suggesting that RLHF/alignment training is indeed effective in reducing certain stereotypical biases.

## Limitations & Future Work

- AFCE may lead to underconfidence on easy tasks; the direction of the decoupling mechanism's effect is inconsistent across different difficulties.
- Only three LLMs (GPT-4o, Claude-3, LLaMA-3-70B) were tested, representing limited model coverage.
- The ECE metric itself has limitations—it is sensitive to binning and cannot capture the full picture of the predictive distribution.
- The internal mechanism of AFCE has not been deeply investigated—why decoupling answer generation improves confidence calibration remains causally unclear.
- Demographic bias experiments rely on role-playing prompts, and their impact in real-world application scenarios requires further verification.

## Related Work & Insights

- **vs. Vanilla Verbalized Confidence**: Traditional methods directly ask the model to report confidence alongside the answer, which yields consistently high confidence (80-100%); AFCE significantly improves calibration through decoupling.
- **vs. Moore & Healy (2008)**: Human confidence is highly sensitive to difficulty (underestimating on simple tasks, overestimating on hard tasks), whereas LLM sensitivity is much weaker—indicating that the "self-assessment" mechanism of LLMs is fundamentally different from human cognition.
- **vs. Tian et al. (Top-K Prompting)**: Top-K is effective on simple/medium tasks but shows limited efficacy on expert-level difficulty; AFCE's advantage on hard tasks is particularly prominent.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Analyzing LLM confidence from a psychological perspective is a novel approach, and the AFCE method is simple and effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers three dimensions (difficulty/persona/demographics) × three models × multiple baselines, representing a comprehensive design.
- **Writing Quality**: ⭐⭐⭐⭐ The motivation is clear, the psychological background is well-introduced, and the experimental logic is self-consistent.
- **Value**: ⭐⭐⭐⭐ Holds practical guiding significance for LLM safety and human-AI collaboration, and AFCE can be directly applied to existing systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Unintended Harms of Value-Aligned LLMs: Psychological and Empirical Insights](unintended_harms_of_value-aligned_llms_psychological_and_empirical_insights.md)
- [\[ACL 2025\] Direct Confidence Alignment: Aligning Verbalized Confidence with Internal Confidence In Large Language Models](direct_confidence_alignment_aligning_verbalized_confidence_with_internal_confide.md)
- [\[ACL 2025\] CER: Confidence Enhanced Reasoning in LLMs](cer_confidence_enhanced_reasoning.md)
- [\[ACL 2025\] Can LLMs Understand Unvoiced Speech? Exploring EMG-to-Text Conversion with LLMs](can_llms_understand_unvoiced_speech_exploring_emg-to-text_conversion_with_llms.md)
- [\[ACL 2025\] Can Large Language Models Address Open-Target Stance Detection?](can_large_language_models_address_open-target_stance_detection.md)

</div>

<!-- RELATED:END -->

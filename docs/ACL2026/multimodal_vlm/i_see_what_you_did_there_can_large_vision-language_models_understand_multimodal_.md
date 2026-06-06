---
title: >-
  [Paper Note] "I See What You Did There": Can Large Vision-Language Models Understand Multimodal Puns?
description: >-
  [ACL 2026][Multimodal VLM][Multimodal puns] This paper introduces MultiPun—the first multimodal pun benchmark featuring "adversarial non-pun distractors" (445 puns + 890 non-puns…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "Multimodal puns"
  - "Homophonic/Homographic puns"
  - "VLM evaluation"
  - "MultiPun benchmark"
  - "Pun-CoT"
date: 2026-05-08
content_hash: df081d16ff9afc79
---

# "I See What You Did There": Can Large Vision-Language Models Understand Multimodal Puns?

**Conference**: ACL 2026  
**arXiv**: [2604.05930](https://arxiv.org/abs/2604.05930)  
**Code**: TBD (Not explicitly provided in the paper)  
**Area**: Multimodal VLM Evaluation / Humor Understanding / Puns  
**Keywords**: Multimodal puns, Homophonic/Homographic puns, VLM evaluation, MultiPun benchmark, Pun-CoT

## TL;DR
This paper introduces MultiPun—the first multimodal pun benchmark featuring "adversarial non-pun distractors" (445 puns + 890 non-puns, covering both homophonic and homographic types). By systematically evaluating 11 VLMs on pun detection, localization, and explanation, the study finds that **all models tend to misidentify non-puns as puns** (TNR generally < 0.4). The authors propose a Pun-CoT prompting strategy and a Pun-Tuning fine-tuning strategy, achieving an average F1 improvement of 16.5%.

## Background & Motivation

**Background**: A pun is a rhetorical device that creates humor by exploiting polysemy (homography) or homophones (homophony). While textual pun detection, localization, and generation have been well-studied since SemEval-2017 Task 7, multimodal pun research (where image and text simultaneously carry literal and figurative meanings) remains a gap, despite recent evaluations of memes, sarcasm, and comics.

**Limitations of Prior Work**: The authors identify three critical deficiencies:
- **Unimodal confinement**: Prior pun research is almost exclusively text-based, ignoring the core role of visual modalities in creating ambiguity.
- **Deficiencies in benchmarks**: Existing multimodal pun datasets lack negative samples (non-puns), making it impossible to verify whether a model truly understands puns or simply labels any "humorous scene" as a pun.
- **Conflation of preference and comprehension**: Current evaluations only ask "Is this a pun?", failing to distinguish true reasoning from affirmative language bias.

**Key Challenge**: Puns require **cross-modal reasoning** (aligning the quadruple of visual object $S_p$ + textual literal $w_p$ + implicit semantics $w_a$ + figurative action $S_a$). However, the superficial pattern of "image + text = humor" is so common in VLM training data that models easily overfit to surface cues, treating any "anthropomorphic fruit" image as a pun.

**Goal**: (1) Construct a multimodal pun benchmark with negative samples; (2) Design an evaluation protocol that distinguishes "comprehension" from "response bias"; (3) Provide effective enhancement solutions.

**Key Insight**: Puns are formalized as a quadruple $\mathcal{P} = \langle w_p, w_a, S_p, S_a \rangle$. Two types of adversarial negative samples are created—Explicative Substitution (ES) and Random Substitution (RS)—to force models to differentiate between "cross-modal synergy" and "lack of synergy."

**Core Idea**: Use adversarial negatives to expose the over-interpretation tendency of VLMs, then address it through the dual approach of Pun-CoT (visual grounding + lexical anchoring + cross-modal verification) and Pun-Tuning (SFT using data containing non-puns).

## Method

### Overall Architecture
MultiPun is a suite consisting of a benchmark, an evaluation protocol, and enhancement methods:

1.  **Data Construction Pipeline** (4 steps):
    -   Step 1 **Pun Tuples Generation**: For homophonic puns, the CMU dictionary is used to find homophones, followed by five filters (Zipf frequency, WordNet senses, visualizability, etc.). For homographic puns, WordNet is used for polysemy, requiring senses to be in different lexical files with path similarity < 0.1 to avoid metonymy.
    -   Step 2 **Positive Sample Generation**: GPT-4o generates (caption, image description, pun explanation) based on the pun tuple. GPT-image-1 generates images, followed by manual filtering and embedding-based deduplication.
    -   Step 3 **Negative Sample Generation**: Each pun is paired with two adversarial non-puns: ES (Explicative Substitution, replacing $w_p$ with a direct description of $S_a$) and RS (Random Substitution, replacing $w_p$ with an irrelevant entity and regenerating the image).
    -   Step 4 **Evaluation Tasks**: Detection (binary classification) / Localization (identifying $w_p$ and $w_a$) / Explanation (providing the complete quadruple + explanation).
2.  **Evaluation Protocol**: Each question is asked twice—biased-to-pun ("Is this a pun?") and biased-to-non-pun ("Is this not a pun?"). Prompt-induced bias is measured using $\Delta$TPR/$\Delta$TNR, and consistency is measured using Cohen's Kappa $\kappa$. This identifies true understanding versus following the prompt.
3.  **Enhancement Methods**: Pun-CoT (a three-step reasoning constraint) and Pun-Tuning (fine-tuning on MultiPun data).

### Key Designs

1.  **Adversarial Non-Pun Construction (ES + RS Strategies)**:
    -   **Function**: Generates negative samples that are superficially similar but lack the pun mechanism, targeting the "anthropomorphic fruit = pun" superficial mode.
    -   **Mechanism**: ES replaces $w_p$ in the caption with a direct description of $S_a$ (e.g., "We make a great pear" → "We make a great couple") while keeping the image constant, breaking the phonetic bridge. RS replaces $w_p$ with an irrelevant entity and redraws the image, breaking the entire pun quadruple. Both maintain scene coherence but sever the pun mechanism.
    -   **Design Motivation**: Random image-text pairs are too easy to distinguish. Adversarial negatives force the model to judge whether the phonetic or semantic bridge actually exists.

2.  **Biased Prompting + $\Delta$ / $\kappa$ Evaluation**:
    -   **Function**: Separates true reasoning from compliance with prompt wording, revealing alignment-induced sycophancy.
    -   **Mechanism**: Asking twice with different biases allows calculating the shift $\Delta$. A large $|\Delta|$ indicates that model decisions rely heavily on prompt wording rather than content.
    -   **Design Motivation**: LLaVA-V1.6-Vicuna-13B's $\Delta$TPR = $-0.923$ reveals it treats "is not a pun" as a command to answer "no," regardless of the content.

3.  **Pun-CoT: Visual Grounding + Lexical Anchoring + Cross-Modal Verification**:
    -   **Function**: Forces a three-step check—image observation, word extraction, and bridge verification—to mitigate four types of hallucinations.
    -   **Mechanism**: Step 1 (Visual Grounding) avoids visual object hallucinations. Step 2 (Lexical Anchoring) prevents hallucinating "pun words" not present in the caption. Step 3 (Cross-Modal Verification) checks for valid phonetic or semantic bridges to reject forced associations.
    -   **Design Motivation**: Error analysis (§4.1) identified 4 hallucination categories; Pun-CoT steps are designed to address each one specifically.

### Loss & Training
-   **Benchmark Construction**: Unsupervised generation with human-in-the-loop quality control.
-   **Pun-Tuning** (Model-level): SFT using MultiPun data following three principles: (i) inclusion of non-puns to suppress hallucinations; (ii) high-quality explanation samples; (iii) inclusion of both pun-biased and non-pun-biased prompt pairs to mitigate sycophancy.
-   **Evaluation**: 11 VLMs evaluated, using LLM-as-judge for explanation quality.

## Key Experimental Results

### Main Results (F1 on Explanation Task, TPR/TNR/F1 use biased-to-pun prompt)

| Type | Model | Homophonic F1 | Homographic F1 | Homophonic TNR | Homographic TNR |
|------|------|---------------|----------------|----------------|-----------------|
| Closed | GPT-5.1 | **0.804** | 0.757 | 0.910 | 0.878 |
| Closed | GPT-4o | 0.741 | 0.683 | 0.786 | 0.659 |
| Closed | Gemini-3-Pro | 0.746 | 0.718 | 0.686 | 0.625 |
| Closed | Claude-Sonnet-4.5 | 0.594 | 0.560 | 0.353 | 0.235 |
| Open | Qwen3-VL-30B-Instruct | 0.535 | 0.511 | 0.209 | 0.125 |
| Open | LLaVA-V1.6-13B | 0.057 | 0.051 | 0.972 | 0.966 |
| Open-Reason | Qwen3-VL-30B-Thinking | 0.618 | 0.631 | 0.399 | 0.414 |

GPT-5.1 is the strongest, but an F1 of 0.80 shows **multimodal puns are challenging for all VLMs**; LLaVA-13B fails significantly on explanation; Claude-Sonnet-4.5 exhibits typical over-interpretation (TPR 0.969 but TNR 0.353).

### Ablation Study (Pun-CoT Improvements, Explanation Task)

| Model | Vanilla F1 (Homo) | Pun-CoT F1 (Homo) | $\Delta$F1 | Vanilla TNR | Pun-CoT TNR |
|------|-------------------|-------------------|------------|-------------|-------------|
| GPT-5.1 | 0.804 | 0.836 | +3.2% | 0.910 | 0.915 |
| GPT-4o | 0.741 | 0.794 | +5.3% | 0.786 | 0.835 |
| Claude-Sonnet-4.5 | 0.594 | 0.641 | +4.7% | 0.353 | **0.495** |
| **LLaVA-V1.6-13B** | 0.057 | **0.501** | **+44.4%** | 0.972 | 0.036 |
| Qwen3-VL-8B-Thinking | 0.595 | **0.807** | **+21.2%** | 0.387 | **0.776** |

### Key Findings
-   **VLMs generally over-interpret puns**: Most models show TPR ≈ 0.95+ but TNR ≈ 0.1-0.4 and $\kappa$ < 0.4—they are "guessing pun" for almost everything.
-   **Closed-source >> Open-source**: Prompt robustness shows a massive gap. Smaller open-source models exhibit severe sycophancy.
-   **Explanation task has a grounding effect**: Requiring an explanation significantly improves TNR because it forces the model to expose the lack of a reasonable alternative.
-   **Homophonic puns are harder than homographic**: $w_a$ does not appear in text and requires phonetic reasoning.
-   **Reasoning models are not always better**: For small models, "thinking" can sometimes amplify over-interpretation.

## Highlights & Insights
-   The **biased prompt protocol** reveals sycophancy as a hidden confounding variable in VLM evaluation, applicable to various binary classification tasks.
-   **Adversarial Negatives (ES + RS)**: Forcing models beyond surface patterns by using "traps" can be generalized to other figurative language evaluations like sarcasm or irony.
-   **Pun-CoT Methodology**: This represents a targeted approach to CoT design—analyzing error patterns first and then designing specific check-steps to address them.
-   The gap in **phonetic reasoning** suggests that future VLM training may need more audio-grounded or phonetic-aware data.

## Limitations & Future Work
-   The dataset is relatively small (445 puns) and limited to English; homophonic puns are highly language-dependent.
-   There is an **LLM bias circularity**: Using GPT to generate data and then evaluate GPT might favor models within that distribution.
-   LLaVA-13B's improvement under Pun-CoT might reflect a shift in decision bias rather than a true increase in comprehension.
-   The role of external tools (e.g., dictionaries for homophones) was not explored.

## Related Work & Insights
-   **vs SemEval-2017**: Extends textual pun formalization to multimodal contexts with adversarial negatives.
-   **vs PunMeme**: Unlike prior humor datasets, MultiPun includes structured negative samples to test mechanism understanding.
-   **vs General VLM Benchmarks**: Fills the gap in evaluating figurative language and humor, which are overlooked by knowledge- or math-centric benchmarks.

## Rating
-   Novelty: ⭐⭐⭐⭐ First multimodal pun benchmark with adversarial negatives.
-   Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across 11 VLMs and multiple task types.
-   Writing Quality: ⭐⭐⭐⭐ Clear framework and progression; excellent illustrative examples.
-   Value: ⭐⭐⭐⭐ Directly applicable benchmark and methodology for improving figurative language understanding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Revisit What You See: Revealing Visual Semantics in Vision Tokens to Guide LVLM Decoding](revisit_what_you_see_revealing_visual_semantics_in_vision_tokens_to_guide_lvlm_d.md)
- [\[ICML 2026\] What You Think is What You See: Driving Exploration in VLM Agents via Visual-Linguistic Curiosity (GLANCE)](../../ICML2026/multimodal_vlm/what_you_think_is_what_you_see_driving_exploration_in_vlm_agents_via_visual-ling.md)
- [\[ACL 2026\] Position: Multimodal Large Language Models Can Significantly Advance Scientific Reasoning](position_multimodal_large_language_models_can_significantly_advance_scientific_r.md)
- [\[CVPR 2026\] See, Hear, and Understand: Benchmarking Audiovisual Human Speech Understanding in Multimodal Large Language Models](../../CVPR2026/multimodal_vlm/see_hear_and_understand_benchmarking_audiovisual_human_speech_understanding_in_mul.md)
- [\[CVPR 2026\] Aligning What Vision-Language Models See and Perceive with Adaptive Information Flow](../../CVPR2026/multimodal_vlm/aif_adaptive_information_flow_vlm.md)

</div>

<!-- RELATED:END -->

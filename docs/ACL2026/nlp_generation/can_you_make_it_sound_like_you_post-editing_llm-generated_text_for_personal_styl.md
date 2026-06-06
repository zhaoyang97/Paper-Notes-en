---
title: >-
  [Paper Note] Can You Make It Sound Like You? Post-Editing LLM-Generated Text for Personal Style
description: >-
  [ACL 2026][Text Generation][Personal writing style] The authors designed a pre-registered online study with 81 participants, where subjects used GPT-o4-mini to draft and then manually post-edit "style-sensitive" texts su…
tags:
  - "ACL 2026"
  - "Text Generation"
  - "Personal writing style"
  - "Post-editing"
  - "LLM collaborative writing"
  - "Style embedding"
  - "User study"
date: 2026-05-08
content_hash: 8813ae5479cb7b20
---

# Can You Make It Sound Like You? Post-Editing LLM-Generated Text for Personal Style

**Conference**: ACL 2026  
**arXiv**: [2604.24444](https://arxiv.org/abs/2604.24444)  
**Code**: https://github.com/ctbaumler/personal_style_postedit  
**Area**: Text Generation / Style  
**Keywords**: Personal writing style, Post-editing, LLM collaborative writing, Style embedding, User study

## TL;DR
The authors designed a pre-registered online study with 81 participants, where subjects used GPT-o4-mini to draft and then manually post-edit "style-sensitive" texts such as wedding vows and apology letters. The findings indicate that while post-editing significantly shifts the text toward the participant's own style and away from the LLM style, the edited text remains systematically more "AI-like" than independent writing—a residual stylistic trace that subjects themselves fail to perceive.

## Background & Motivation

**Background**: The dominant paradigm in current LLM writing assistance is "AI drafting + human post-editing"—a workflow that matured during the machine translation era and has been widely adopted for workplace emails and document collaboration. Prior research (Reza et al. 2025, Hwang et al. 2025) suggests that users welcome AI drafting for content-driven writing; however, for "style-driven" writing (wedding vows, eulogies, apologies), users strongly resist AI intervention in the language generation phase, fearing it "won't sound like me."

**Limitations of Prior Work**: (1) To date, no controlled study has empirically validated whether post-editing can actually make an LLM draft sound like the author—a core controversy frequently debated on social media regarding AI wedding vows. (2) Even if post-editing is effective, the location of "edited text" on the stylistic continuum has not been quantified: is it closer to the user's independent writing, or does it retain an LLM "fingerprint"? (3) It remains unknown whether "objective stylistic similarity" aligns with "subjective perceived stylistic authenticity." This concerns the hidden risk of LLM-aided writing: users may believe they have reclaimed their style while readers can still easily detect AI traces.

**Key Challenge**: Style serves as a social signal through which writers express identity, group belonging, and relationships. Multiple studies have confirmed that LLM default styles possess detectable statistical fingerprints (em-dashes, specific hedging words, sentence templates), creating a systematic distribution gap from human styles. Theoretically, when users attempt to overwrite LLM styles via post-editing, they can only modify differences they perceive, leaving unperceived LLM features intact—leading to a "high self-rating vs. actual AI-trace" disconnect.

**Goal**: To answer three sets of questions through a pre-registered study: (H1) Does post-edited text become more similar to the subject and less similar to the LLM? (H2) Does post-edited text form a distinct, detectable "third style"? (H3) Does subjective perceived "style self-similarity" align with objective embedding-based measurements?

**Key Insight**: Using LUAR (Rivera-Soto et al. 2021) as an author-level embedding to measure stylistic similarity—which captures individual linguistic habits more sensitively in small-sample tasks than traditional stylometry—and cross-validating with the Pangram AI detector to signal "residual LLM style." Participants completed tasks in both treatment and control groups (Treatment: write details $\rightarrow$ edit LLM draft; Control: independent writing), using control writing as the "authentic style" anchor for each user.

**Core Idea**: A within-subject design combined with multiple style embedding metrics and subjective user ratings to transform the debate over whether humans can overwrite LLM stylistic fingerprints from a philosophical argument into a measurable empirical problem.

## Method

### Overall Architecture
The study consists of five phases: (1) **pre-survey** where 100 Prolific subjects selected 6 tasks they "care most about personal style" from 8 options; (2) tutorial; (3) **treatment block**—for 4 tasks, subjects provided $\ge 30$ words of detail, GPT-o4-mini generated a draft, and subjects spent $\ge 2$ minutes post-editing; (4) **control block**—for the remaining 2 tasks, subjects wrote $\ge 150$ words independently without AI; (5) **post-survey** collecting demographics, 5-point Likert ratings for style authenticity and usability, and future usage intentions. 81 valid subjects were retained after pre-registered exclusion criteria. All similarities were calculated using LUAR-MUD embedding cosine similarity, with $p$-values obtained via 10,000 permutations, effect sizes using Hedges' $g$ with 1,000 bootstrap CI, and multiple comparisons controlled using Benjamini–Hochberg at $q=0.05$.

### Key Designs

1.  **Within-subject Comparison of Triple Style Anchors**:
    *   Function: Decomposes whether post-editing successfully captures personal style into quantifiable comparisons.
    *   Mechanism: Three types of text were collected for each subject: control text $C_i$, raw LLM draft $D_i$, and post-edited text $E_i$. Permutation tests were run on four key comparisons: (H1a) $\mathrm{sim}(E_i, C_i)$ vs. $\mathrm{sim}(D_i, C_i)$ to see if editing pulls toward self-style; (H1b) $\mathrm{sim}(E_i, \mathrm{LLM})$ vs. $\mathrm{sim}(D_i, \mathrm{LLM})$ to see if editing moves away from LLM style; (H1a′) $\mathrm{sim}(E_i, C_i)$ vs. $\mathrm{sim}(E_i, C_{j\neq i})$ to verify the pull is toward "self" rather than generic "human-ness"; (H1c) $\mathrm{sim}(E_i, \mathrm{LLM})$ vs. $\mathrm{sim}(E_i, C_i)$ to quantify the relative strength of residual LLM style.
    *   Design Motivation: Observing changes before/after editing alone cannot rule out the possibility that subjects are simply making text more "generic human" rather than "individual self." Using both LLM drafts and control writing as anchors allows for precise positioning of post-edited text on the style spectrum.

2.  **Domain-specific Selection of Style Embeddings**:
    *   Function: Selects the most sensitive author-style embedding for the study's data.
    *   Mechanism: Control group data was treated as ground truth for an authorship identification task—ranking 1 control text from the true author against texts from 80 other subjects. Six embeddings were tested (LUAR-MUD, LUAR-CRUD, multilingual-style-representation, CISR, StyleDistance, SAURON). LUAR-MUD significantly outperformed others with MRR $= 0.589$, R@1 $= 0.451$, and R@8 $= 0.833$, and was thus used for primary analysis, with CISR used to verify consistency.
    *   Design Motivation: Models performing best on general benchmarks may not be sensitive to short-form text or specific writing scenarios; selecting a model via an in-domain authorship task avoids methodological flaws.

3.  **Diagnosis of Objective Measure vs. Subjective Perception Disconnect**:
    *   Function: Tests whether machine-measured stylistic similarity aligns with user-perceived authenticity.
    *   Mechanism: Subjects used a 5-point Likert scale to evaluate "To what extent does this text sound like me," averaged as perceived self-similarity. This was regressed against objective LUAR similarity using repeated-measures correlation, yielding $r = 0.244 \pm 0.076, p < .0001$—a significant but weak calibration. Follow-up showed no significant difference in perceived authenticity between post-edited and control texts ($p = .9062$), yet objective measures showed post-edited text remained significantly more similar to the LLM (H1c, $g = -1.43$).
    *   Design Motivation: Objective measures alone yield a pessimistic conclusion (AI traces remain); subjective ratings alone yield an optimistic one (post-editing is sufficient). Running both reveals the true risk: users are blind to AI fingerprints in their manuscripts that readers or detectors can identify.

## Key Experimental Results

### Main Results

| Hypothesis | Comparison | Effect Direction | Hedges' $g$ | 95% CI | $p$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| H1a | Post-edit (Post vs Pre) sim to self-control | Significantly Closer | $+0.55$ | $[0.38, 0.71]$ | $.0002$ |
| H1a′ | Post-edit sim to self vs. others' control | Closer to Self | $-0.56$ | $[-0.70, -0.43]$ | $.0002$ |
| H1b | Post-edit (Post vs Pre) sim to LLM (LUAR) | Significantly Further | $-0.41$ | $[-0.44, -0.39]$ | $.0002$ |
| H1b | Same as above using Pangram AI score | Significantly Further | $-0.45$ | $[-0.55, -0.35]$ | $.0002$ |
| H1c | Post-edit sim to LLM vs. self-control | Still closer to LLM | $-1.43$ | $[-1.55, -1.32]$ | $.0002$ |
| H2a | Homogeneity of post-edit vs. control groups | More homogeneous | $+1.42$ | $[1.33, 1.51]$ | $.0002$ |
| H2b | Homogeneity of post-edit vs. LLM groups | More diverse | $-0.69$ | $[-0.74, -0.63]$ | $.0002$ |
| H2c | Post-edit sim to others' post-edit vs. self control | Shared AI traces | $+1.14$ | $[1.02, 1.26]$ | $.0002$ |
| H3 | Perceived vs. LUAR self-similarity | Weak positive correlation | $r=0.244$ | $\pm 0.076$ | $<.0001$ |

### Ablation Study

| Configuration / Slice | Key Metric | Description |
| :--- | :--- | :--- |
| Main LUAR-MUD (Full) | H1a $g=+0.55, H1c g=-1.43$ | All main conclusions significant |
| Replaced with CISR embedding | H1a $g=+0.48, H1c g=-2.30$ | Consistent direction, slight change in magnitude |
| Pangram AI detector cross-val H1b | $g=-0.45$ | Same sign as LUAR |
| Tasks with lower style importance | Interaction $\beta=0.020, p<.001$ | Less concern for style leads to less AI trace removal |
| Word-level: contractions | Post-edit has $5\times$ more than draft | Contractions are a key personal feature |
| Em-dash deletion rate | $23\%$ (58 of 254 deleted) | Well-known "AI traits" are actively erased |
| "Delve" occurrences | $0$ (Never appeared in study) | Viral AI words are highly scrutinized |

### Key Findings
*   **H1c is the most critical conclusion**: The position of post-edited text relative to "LLM draft" and "Self-control" is asymmetric—it remains much closer to the LLM ($g = -1.43$, vs. $g = +0.55$ for H1a), indicating post-editing only masks a fraction of AI fingerprints.
*   **H2c reveals "shared AI residue"**: Post-edited texts by different subjects are more similar to each other than to their own control texts—suggesting certain LLM stylistic features are "collective blind spots" for editors.
*   **H3 disconnect is high-risk**: Subjects perceived their post-edited text as representative of themselves as control text ($p = .906$), despite the objective $g = -1.43$ gap. Self-evaluation of style is no longer reliable in the AI era.
*   **Style importance drives effort**: A mixed-model showed that style importance significantly moderates the impact of post-editing on LLM-similarity ($\beta=0.020, p<.001$), suggesting cleaning behavior is motivation-driven rather than automatic.
*   **Non-standardized edits (typos, spacing) significantly increase self-similarity** ($\beta = -0.116, p = .003$ for CoLA score), as "irregular habits" are authentic identity signals.

## Highlights & Insights
*   **"Third Style" hypothesis confirmed**: Post-edited text resides in a stylistic space that is neither LLM nor human, but an independent, mutually recognizable hybrid—providing a new dimension for future AI detectors.
*   **Pre-registration + Within-subject design**: The study transforms qualitative social media debates into falsifiable science by decomposing "AI wedding vows" into a set of testable H1-H5 hypotheses.
*   **The "Perceived $\neq$ Actual" gap**: While prior user-test driven LLM style alignment studies treated user satisfaction as ground truth, this paper proves that such paradigms systematically overestimate alignment effectiveness.
*   **Transferable methodology**: The use of in-domain authorship tasks to select embeddings can be generalized to any small-sample, domain-specific stylistic analysis.
*   **Awareness of "AI words"**: The study provides empirical evidence of which traits are scrubbed (em-dashes) versus ignored ("exploring/guiding"). "Delve" has reached the level of a high-alert viral term.

## Limitations & Future Work
*   Style was measured only via LLM embeddings, without manual forensic linguistic evaluation; embeddings may miss semantic-level stylistic features perceptible to humans.
*   Differences in tasks between control and treatment might cause self-similarity to be underestimated by missing task-specific preferences.
*   The writing tasks are simulated—subjects might edit more rigorously if truly sending a vow to a spouse.
*   LLM draft generation (bullet + word count prompt) may not represent real-world prompt engineering habits.
*   Only "AI draft $\rightarrow$ Human edit" was studied, leaving "Human draft $\rightarrow$ AI edit" or multi-turn interactions for future work.
*   The study lacks a recipient perspective—whether readers (especially friends/family) can detect residual AI traces remains unknown.

## Related Work & Insights
*   **vs. Chakrabarty et al. 2025 (CHI'25 expert edits)**: While they focused on experts editing for "quality," this study looks at laypeople editing for "personal style," revealing the upper limits of style recovery for average users.
*   **vs. Reza et al. 2025 / Hwang et al. 2025**: Earlier work found users welcome AI for planning but resist it for translating/reviewing. This study provides an empirical answer: users accept AI for translating if allowed to post-edit, but objective style remains AI-heavy.
*   **vs. Padmakumar & He 2024 (diversity)**: They found LLM collaboration reduces content diversity; this study provides parallel evidence for stylistic diversity (H2a: post-edited texts are more homogeneous).
*   **vs. Russell et al. 2025**: They found frequent ChatGPT users are better at detecting AI text; this study’s H3 suggests writers are nonetheless blind to AI residue in their own edited work.

## Rating
*   Novelty: ⭐⭐⭐⭐ Systematic quantification of post-editing's recovery power and the subjective/objective disconnect.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ 81-subject pre-registered study, 5 main hypotheses, multiple embeddings, and qualitative coding.
*   Writing Quality: ⭐⭐⭐⭐⭐ Clear organization, fully reported effect sizes and CI, and honest discussion of ethics and limitations.
*   Value: ⭐⭐⭐⭐⭐ Direct implications for LLM collaboration, AI detection, and alignment. The "Perception $\neq$ Alignment" finding is a major caveat for the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ConlangCrafter: Constructing Languages with a Multi-Hop LLM Pipeline](conlangcrafter_constructing_languages_with_a_multi-hop_llm_pipeline.md)
- [\[ACL 2026\] Are Emotion and Rhetoric Neurons in LLM? Neuron Recognition and Adaptive Masking for Emotion-Rhetoric Prediction Steering](are_emotion_and_rhetoric_neurons_in_llm_neuron_recognition_and_adaptive_masking_.md)
- [\[ACL 2026\] Planning Beyond Text: Graph-based Reasoning for Complex Narrative Generation](planning_beyond_text_graph-based_reasoning_for_complex_narrative_generation.md)
- [\[ACL 2026\] Frankentext: Stitching Random Text Fragments into Long-Form Narratives](frankentext_stitching_random_text_fragments_into_long-form_narratives.md)
- [\[ACL 2026\] Right at My Level: A Unified Multilingual Framework for Proficiency-Aware Text Simplification](right_at_my_level_a_unified_multilingual_framework_for_proficiency-aware_text_si.md)

</div>

<!-- RELATED:END -->

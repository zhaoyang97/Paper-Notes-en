---
title: >-
  [Paper Note] Is Human-Like Text Liked by Humans? Multilingual Human Detection and Preference Against AI
description: >-
  [ACL 2026][Multilingual & Machine Translation][MGT Detection] The authors organized 19 native experts to perform human-machine text discrimination across 16 datasets involving 9 languages, 9 domains…
tags:
  - "ACL 2026"
  - "Multilingual & Machine Translation"
  - "MGT Detection"
  - "Multilingual"
  - "Human Evaluation"
  - "Prompting Rewriting"
  - "Preference Analysis"
date: 2026-05-08
content_hash: 526935e5e0f17470
---

# Is Human-Like Text Liked by Humans? Multilingual Human Detection and Preference Against AI

**Conference**: ACL 2026  
**arXiv**: [2502.11614](https://arxiv.org/abs/2502.11614)  
**Code**: https://github.com/xnlp-lab/HumanEval-MGT  
**Area**: Multilingual / AIGC Detection / Human-AI Preference  
**Keywords**: MGT Detection, Multilingual, Human Evaluation, Prompting Rewriting, Preference Analysis

## TL;DR
The authors organized 19 native experts to perform human-machine text discrimination across 16 datasets involving 9 languages, 9 domains, and 11 SOTA LLMs (8.8k instances). They found that the average expert accuracy is as high as 87.6% (far exceeding the "near-random" conclusions of earlier studies). Furthermore, they revealed that although machine text rewritten with explicit prompts can suppress detection accuracy to 72.5%, humans tend to prefer machine text when they cannot distinguish the source, challenging the implicit assumption that "human-like equals liked-by-human."

## Background & Motivation
**Background**: Existing research on MGT (Machine-Generated Text) detection mostly bases human evaluation conclusions on GPT-3.5-turbo with English and approximately 300 samples. These studies generally report that "humans find it difficult to distinguish LLMs from human-written text, with accuracy near random guess" (Guo et al. 2023; Chein et al. 2024; Wang et al. 2024a). These conclusions are widely cited to argue that "LLMs have passed the informal Turing Test."

**Limitations of Prior Work**: The aforementioned conclusions have three insufficiently tested limitations: (1) Narrow language coverage, focusing almost exclusively on English and a small amount of Chinese; (2) Outdated models, missing the new generation like GPT-4o / Claude-3.5 / Llama4; (3) Vague annotator profiles, often including laymen unfamiliar with LLMs, which fails to reflect the "expert upper bound."

**Key Challenge**: To answer "whether LLMs truly passed the Turing Test," one must first measure the upper bound of human recognition ability. If even trained native experts cannot identify the text, the conclusion becomes truly credible. Conversely, if experts can identify it, it implies the degree of anthropomorphism of LLMs is seriously overestimated, and existing detector evaluations and the "AI is unrecognizable" narrative both require revision.

**Goal**: The authors decompose the problem into four research sub-questions: (i) What is the upper bound of expert detection across multiple languages/models/domains; (ii) Which linguistic features drive discrimination decisions; (iii) Can explicit prompting to bridge differences truly close the gap between machines and humans; (iv) Which type of text do humans actually prefer?

**Key Insight**: Moving beyond the old "small-scale English GPT-3.5" setting, the authors construct a four-dimensional matrix of "large-scale + multilingual + new models + native experts." They design four detection protocols (single-text binary, pair-wise binary, triplet three-class, and pair-wise four-class) to decouple the factors of "task difficulty" and "recognition ability."

**Core Idea**: Through a full-factorial human evaluation involving 9 languages × 9 domains × 11 LLMs across 16 datasets with 19 native NLP experts, the authors establish a reusable multilingual benchmark for the "human detection upper bound" and the "linguistic gap between machines and humans." They integrate prompting-based gap bridging and human-machine preference into the same framework to address "human-like vs. liked-by-human."

## Method
Strictly speaking, this is not an "algorithmic paper." The method section consists of a meticulously designed large-scale human evaluation protocol, a prompting rewrite experiment, and a preference comparison study. The pipeline is described below.

### Overall Architecture
The overall pipeline is divided into four stages:

1.  **Data Construction**: 16 datasets were sampled across 9 languages × 9 domains (including Arabic, Chinese, English, Hindi, Italian, Japanese, Kazakh, Russian, and Vietnamese). For each dataset, 300–600 human-written texts (hwt) were collected, and paired machine-generated texts (mgt) were produced using one multilingual SOTA model (e.g., GPT-4o / Claude / Llama3-405B) and one language-specific model (e.g., Qwen / AceGPT / Anita).
2.  **Human Detection (Phase 1)**: 19 native NLP experts (BSc/MSc/PhD/Postdoc) identified hwt vs. mgt under four annotation protocols, resulting in a fine-grained map of the "expert detection upper bound" and which languages/domains/models/protocols are hardest to judge.
3.  **Prompting Refinement**: Annotators wrote "rewriting prompts" based on identified differences (concreteness, cultural nuance, length diversity, formatting, code-switching, etc.). The same LLMs were used to regenerate 32k instances of rewritten mgt, followed by a second round of human evaluation and assessment by 26 automatic detectors to quantify how much of the gap prompting can bridge.
4.  **Preference Experiment**: Annotators chose their preference among the original hwt, original mgt, and rewritten mgt (plus a "none" option) across 6 datasets. This was cross-analyzed with detection accuracy to answer "whether humans truly prefer human-written text."

### Key Designs

1.  **Hierarchical Design of Four Detection Protocols (I/II/III/IV)**:
    -   **Function**: Explicitly separates "information volume" and "task difficulty," controlling the impact of the "presence of a paired reference" variable on human evaluation accuracy.
    -   **Mechanism**: Protocols include I. Single-Binary (判断 Y/N for one segment), II. Pair-Binary (choose one from a pair), III. Triplet-Three-Class (choose one from hwt + 2 mgt), and IV. Pair-Four-Class (A/B/none/both). Theoretically, difficulty ranks I > IV > III > II.
    -   **Design Motivation**: Previous works used inconsistent protocols, making "60% accuracy" and "90% accuracy" incomparable. Comparing results from the same annotators across protocols (e.g., Arabic Tweets: 50.1% under I vs. 92.7% under II) cleanly isolates the systematic effect of paired references (~+20%).

2.  **Five-Dimensional Differential Templates for Rewriting Prompts**:
    -   **Function**: Translates "dimensions where machines and humans differ" into executable generation instructions as a controlled experiment to probe the limits of prompting.
    -   **Mechanism**: Annotators summarized five systematic differences: concreteness, cultural/religious nuance, diversity in length/structure/emotion, Markdown formatting, and code-switching. Prompts explicitly requested adding specific names/dates/URLs, avoiding bullet points/Markdown, and using native-style writing to avoid English interference.
    -   **Design Motivation**: To test whether LLMs "know where they are unlike humans." If prompting fills the gap, the defect is a "surface alignment" issue; if not (especially for cultural nuance and diversity), it implies deep-rooted defects in training data and optimization objectives.

3.  **Cross-Analysis of Detection Ability × Preference (human-like vs. liked-by-human)**:
    -   **Function**: Tests the implicit premise that "making a model more human-like" is equivalent to "making humans like the model output more."
    -   **Mechanism**: Both "which is human-written" (detection) and "which do you prefer" (preference) labels were collected for the same samples. Let the expert's detection accuracy be $p$ and the probability of preferring human text be $q$. The authors found that while $q$ correlates with $p$, it is far from 1: at $p \approx 1$, $q \approx 0.6$; at $p \approx 0.5$, $q$ is actually $< 0.5$, indicating that humans tend to prefer machine text when they cannot distinguish the source.
    -   **Design Motivation**: By quantifying "human-likeness" and "likability" on two independent axes, the study challenges the common confusion in RLHF/alignment literature that "mimicking human preference = mimicking human writing."

### Loss & Training
This is an evaluation/analysis paper and does not involve specific training losses. Key hyperparameters include the sampling scale (~9k detection instances, 13.5k second-round instances, 5k preference instances) and the reproduction settings of 26 automatic detectors. Annotation followed zero-shot and few-shot settings (3–5 labeled examples for "expert calibration").

## Key Experimental Results

### Main Results: Human Detection Accuracy across 9 Languages × 16 Datasets

| Language | Representative Dataset | Protocol / Shot | Avg Accuracy | Key Observation |
| :--- | :--- | :--- | :--- | :--- |
| Chinese | Zhihu-QA (GPT-4o) | II / Zero | 99.6% | High accuracy, 5 annotators at 99–100% |
| Chinese | Student Essay | IV / Zero | 97.0% | Four-class doesn't drop accuracy |
| Russian | News | II / Few | 100.0% | Single annotator reached 100% |
| Russian | Academic Summary | I / Few | 80.0% | Protocol I is significantly harder |
| English | Peersum | II / Few | 99.8% | Peer reviews are easily identified |
| Italian | DICE News (GPT-4o) | II / Few | 100.0% | Same annotator: Anita 88% / Llama3-405B 99.7% / GPT-4o 100% |
| Arabic | Dialect Tweet | I / Zero | 50.1% | Short text + dialect = Hardest |
| Arabic | SANAD News | II / Zero | 100.0% | News is easiest to identify |
| Vietnamese | Wikipedia | II / Zero | 50.7% | Wiki is highly similar to training data |
| Kazakh | Wikipedia | II / Zero | 79.7% | Similar to above |
| **Total** | **16 Datasets / 8,778 cases** | – | **87.6%** | **Much higher than prior "near-random" conclusions** |

### Ablation Study: Accuracy Drop after Prompting Rewriting

| Dataset | Original Acc | Improved Acc | Gain (Loss) | Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| Kazakh Wikipedia | 79.7 | ↓ Significant | – | Adding specific names/dates was most effective |
| Arabic Tweet | 50.1 | ↓ Significant | – | Adding emotion felt more human, but hashtags leaked |
| Chinese Zhihu-QA | 99.6 | ~92 | -7.6 | "First/Second/Finally" structure is hard to eradicate |
| English Peersum | 99.8 | ~88 | -11.8 | Length/structure variance remains too low |
| Italian DICE (GPT-4o) | 100.0 | ~85 | -15 | Formatting is easy; cultural nuance is hard |
| **Global Mean** | **87.6** | **72.5** | **-15.1** | Prompting fills half the gap |

Out of 26 automatic detectors, 19 saw an accuracy drop after rewriting, consistent with human trends, suggesting that "human-likeness" gained via prompting is also effective against automated systems.

### Key Findings
-   **Experts are not guessing randomly**: Previous "50% accuracy" findings were largely artifacts of GPT-3.5 + single-text protocols + non-experts. With SOTA models + paired protocols + native experts, accuracy jumps to 87.6%.
-   **Protocols make a huge difference**: The same Arabic Tweets resulted in 50.1% accuracy under protocol I but 92.7% under protocol II. This shows that the "presence of a comparative reference" is vital for human judgment.
-   **Model differences increase with capability**: GPT-4o > Llama3-405B > Anita. Stronger models are more human-like, yet experts can still identify GPT-4o with 99.8% accuracy in English peer reviews.
-   **Prompting fills half the gap**: About 50% of samples can bridge the gap via prompt modification, but deep differences like cultural nuance/diversity/emotion cannot be filled even with explicit instructions.
-   **Humans don't always prefer humans**: In Chinese QA/Essay where sources are easily identified, humans prefer human text (~60%). However, in harder-to-distinguish Russian/Arabic summaries, humans prefer machine text $\geq 2/3$ of the time. In emotional QA, 3 out of 4 annotators preferred machine answers, suggesting RLHF preference distributions may systematically favor machine-style output.

## Highlights & Insights
-   **"Accuracy Health Check" via Protocol Decoupling**: By running four protocols with the same team, the authors provide a "conversion table" for interpreting small-scale prior studies—a design reusable in any future MGT/AIGC detection evaluation.
-   **Empirical separation of "human-like ≠ liked-by-human"**: The conclusion is counter-intuitive: when humans cannot distinguish the source, their preference tilts toward the machine. This suggests to the RLHF community: "You think you are aligning with human preferences, but you might be aligning with LLM style."
-   **Five-dimensional differential prompt templates**: Translating "how machines differ from humans" into executable prompts is a transferable trick. This can be used as a system prompt in instruction tuning to generate text with lower detectability.
-   **Multilingual + Multi-model + Multi-protocol Data Resource**: 17k original + 32k rewritten MGTs, plus associated metadata, constitute one of the most comprehensive resources for multilingual AIGC detection.

## Limitations & Future Work
-   Annotators were all NLP experts/researchers; conclusions represent the "expert upper bound." Research with laymen is needed for general user scenarios.
-   The preference experiment only covered 6 datasets and 10 annotators, limiting statistical significance. Correlations with individual traits (MBTI, age, etc.) remain future work.
-   The "prompting fills half the gap" conclusion used the same LLM for rewriting. It remains unclear if weaker models using the same prompts could achieve similar results.
-   While covering 9 languages, low-resource African or certain Indo-European languages are missing. Quantitative assessment of "training contamination" in Wiki-style corpora is also needed.

## Related Work & Insights
-   **vs. Guo et al. (2023) / Chein et al. (2024)**: Their conclusion was that experts are only slightly better than random. This paper reaches 87.6% by improving the model, protocol, and annotator profile, highlighting that MGT evaluations must report protocol metadata.
-   **vs. Wang et al. (2024a/b)**: This is a large-scale, multilingual generalization of the authors' previous work.
-   **vs. RLHF / Alignment Works**: The preference experiments challenge the equation of "human preference = human writing," providing empirical evidence for discussions on RLHF data bias.

## Rating
-   Novelty: ⭐⭐⭐⭐ (Not a new algorithm, but the experimental design is pioneering for MGT.)
-   Experimental Thoroughness: ⭐⭐⭐⭐⭐ (9 languages × 9 domains × 11 models × 4 protocols; the scale is rare.)
-   Writing Quality: ⭐⭐⭐⭐ (Data tables and protocol diagrams are intuitive.)
-   Value: ⭐⭐⭐⭐⭐ (Overturns the "AI passed the Turing Test" narrative and provides reusable multilingual datasets.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] HelpSteer3-Preference: Open Human-Annotated Preference Data across Diverse Tasks and Languages](../../NeurIPS2025/multilingual_mt/helpsteer3-preference_open_human-annotated_preference_data_across_diverse_tasks_.md)
- [\[ACL 2026\] Evaluating Robustness of Large Language Models Against Multilingual Typographical Errors](evaluating_robustness_of_large_language_models_against_multilingual_typographica.md)
- [\[ACL 2026\] CLewR: Curriculum Learning with Restarts for Machine Translation Preference Learning](clewr_curriculum_learning_with_restarts_for_machine_translation_preference_learn.md)
- [\[ACL 2026\] Lingo_Research_Group at SemEval-2026 Task 9: Evaluating Prompt Variants for Polarization Detection](lingo_research_group_at_semeval-2026_task_9_evaluating_prompt_variants_for_polar.md)
- [\[ACL 2026\] MultiHaluDet: Multilingual Hallucination Detection via LLM Hidden State Probing](multihaludet_multilingual_hallucination_detection_via_llm_hidden_state_probing.md)

</div>

<!-- RELATED:END -->

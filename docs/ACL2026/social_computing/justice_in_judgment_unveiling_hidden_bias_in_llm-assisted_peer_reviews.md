---
title: >-
  [Paper Note] Justice in Judgment: Unveiling (Hidden) Bias in LLM-assisted Peer Reviews
description: >-
  [ACL 2026][Social Computing][Counterfactual Evaluation] The authors systematically audit peer review biases across 9 LLMs using counterfactual evaluation by "modifying author metadata without changing paper content." The…
tags:
  - "ACL 2026"
  - "Social Computing"
  - "Counterfactual Evaluation"
  - "Affiliation Bias"
  - "Hidden Bias"
  - "Soft Rating"
  - "Alignment Bias"
date: 2026-05-08
content_hash: f0d54978a82a0292
---

# Justice in Judgment: Unveiling (Hidden) Bias in LLM-assisted Peer Reviews

**Conference**: ACL 2026  
**arXiv**: [2509.13400](https://arxiv.org/abs/2509.13400)  
**Code**: LLMReviewBias (Open-source repository link provided in the paper)  
**Area**: LLM Evaluation / Fairness / Peer Review  
**Keywords**: Counterfactual Evaluation, Affiliation Bias, Hidden Bias, Soft Rating, Alignment Bias

## TL;DR
The authors systematically audit peer review biases across 9 LLMs using counterfactual evaluation by "modifying author metadata without changing paper content." They find that all models exhibit significant preference for prestigious institutions (RS) and are more lenient toward senior PIs and prolific authors. Crucially, while models may appear neutral in "hard ratings," "soft ratings" (expected scores based on token probabilities) reveal much stronger hidden biases, uncovering a failure mode where "alignment merely masks rather than eliminates preferences."

## Background & Motivation
**Background**: Major conferences such as ICLR 2025, AAAI 2026, and ICML 2026 have begun allowing or even encouraging reviewers to use LLMs for assistance; "LLM as reviewer" deep-research pipelines have also emerged. Existing observational studies (Pataranutaporn 2025; Ye 2024; Zhu 2025) suggest that LLMs exhibit affiliation preferences when reviewing economics or well-known authors' papers.

**Limitations of Prior Work**: (1) Most existing works are "observational," noting that LLM reviews give higher scores than human reviews without being able to attribute the bias to specific author attributes (affiliation? gender? seniority? publication count?). (2) Evaluation metrics are often limited to "hard ratings" (integer scores from greedy decoding), but models fine-tuned via RLHF/instruction tuning are adept at "acting" neutral at the surface output level, masking many biases. (3) Empirical evidence on whether this truly changes "accept/reject" decisions is scarce, lacking quantification of flip rates for borderline papers.

**Key Challenge**: Alignment goals and bias elimination targets operate only at the "output layer" and do not reach the "internal token probability distributions" of the model. If internal distributions remain biased toward certain author groups, the bias will resurface during sampling, soft scoring, or multi-model aggregation. This implies that relying solely on hard ratings after alignment to prove "model fairness" is unreliable.

**Goal**: (1) Construct a counterfactual evaluation framework to ablate four types of author metadata (affiliation, gender, seniority, publication history) one by one; (2) Report both hard and soft ratings to quantify the bias gap between "surface alignment" and "internal distributions"; (3) Quantify the actual impact of bias on accept/reject flips; (4) Compare which of the 9 open-source and closed-source LLMs exhibits the most severe bias.

**Key Insight**: Model LLM review as $P_{\text{LLM}}(\bm{r}, \bm{c} \mid \texttt{prompt}(\bm{p}, \bm{m}))$, fixing the paper content $\bm{p}$ and performing univariate counterfactual interventions only on the author metadata $\bm{m}$. The rating gap between two sets of metadata for the same paper text serves as the causal attribution for that metadata, eliminating confounding factors from the content.

**Core Idea**: Use counterfactual intervention combined with dual scoring (hard and soft) to systematically audit LLM peer reviews, using the discrepancy where "hard rating appears neutral while soft rating remains biased" as a quantitative indicator of "hidden bias / alignment misalignment." This reveals that LLM-assisted peer reviews cannot be trusted blindly regarding fairness.

## Method

### Overall Architecture
The evaluation pipeline consists of three steps: (1) Data: Sampling 252 real papers (6 accepted + 6 rejected) from each of the 21 sub-fields of ICLR 2025; (2) Intervention: Constructing multiple "synthetic author profiles" for each paper, changing only one metadata dimension at a time (affiliation / gender / seniority / publication history) while keeping others constant, then calling LLMs to generate comments $\bm{c}$ and ratings $\bm{r}$ using a unified prompt template; (3) Scoring: Simultaneously recording hard ratings (integer scores from greedy decoding) and soft ratings (the expected value of the probability distribution of rating tokens given fixed greedy comments, $\sum_i r_i \cdot P_{\text{LLM}}(r_i, \hat{\bm{c}} \mid \texttt{prompt})$, calculated to two decimal places). Finally, the full experiment is run across 9 LLMs to conduct pairwise win rate statistics and flip rate analysis.

### Key Designs

1. **Counterfactual Metadata Intervention (4 Independent Dimensions)**:
    - **Function**: Decomposes the "impact of author identity on scores" into four controllable variables, removing the influence of paper quality itself.
    - **Mechanism**: Constructs four sets of intervention experiments. (a) **Affiliation**: 8 Ranked-Stronger (RS, e.g., MIT, CMU) vs. 8 Ranked-Weaker (RW, e.g., University of Lagos / Gondar), each paired with country-matched male/female names for a 16×16 pairwise comparison per paper; (b) **Gender**: 4 Anglo male names, 4 Anglo female names, each tested under both RS and RW institutions; (c) **Seniority**: Comparing Senior PI vs. Undergraduate Student profiles while fixing other factors; (d) **Publication History**: Two versions per profile—"100 top-tier publications (TTP)" vs. "0 TTP." By changing only one variable per paper, the results allow for causal interpretation.
    - **Design Motivation**: Earlier work (e.g., Pataranutaporn 2025) only compared "with metadata vs. anonymized," failing to locate which dimension the bias originated from. This work enables quantitative answers to core questions like "Does institution or seniority carry more weight?" and provides actionable "levers" for future debiasing.

2. **Dual Scoring: Hard vs. Soft Rating**:
    - **Function**: Exposes implicit biases "submerged" by alignment.
    - **Mechanism**: Hard ratings use greedy decoding $\arg\max_{\hat{\bm{r}}, \hat{\bm{c}}} P_{\text{LLM}}(\bm{r}, \bm{c} \mid \texttt{prompt})$ to get an integer score. Soft ratings, after fixing the greedy comments $\hat{\bm{c}}$, compute the weighted expectation $\sum_i r_i \cdot P_{\text{LLM}}(r_i, \hat{\bm{c}} \mid \texttt{prompt})$ over the probability distribution of tokens $\{r_1, \ldots, r_{10}\}$. The difference between the two ratings represents the degree to which "the model says one thing but thinks another."
    - **Design Motivation**: RLHF / instruction tuning primarily acts on the mode of the final generation distribution, but pre-training biases persist in other high-probability positions. Unless the sampling temperature is 0 or multiple reviews are averaged, these biases will be weighted into the result. Soft rating quantifies this hidden bias into comparable numbers, representing the most high-impact design of this paper. For instance, in Ministral 8B's affiliation experiment, the hard rating win rate for RS is only 4.3%, but the soft rating win rate for RS jumps to 68.6%—a 14× multiplier showing "surface fairness, internal skew."

3. **Accept/Reject Flip Rate Analysis**:
    - **Function**: Translates "minor score differences" into "decision consequences," quantifying the harm of bias to the actual academic ecosystem.
    - **Mechanism**: Compares the hard ratings of the same paper under RS vs. RW metadata against the actual ICLR acceptance threshold to see the "proportion of RW papers originally rejected that flip to accept when changed to RS" and "proportion of RS papers originally accepted that flip to reject when changed to RW." For example, QwQ-32B flips 21.4% of originally rejected papers to accepted under RS, and 7.9% of originally accepted papers to rejected under RW.
    - **Design Motivation**: A few percentage points difference in ratings might sound negligible, but a flip for a borderline paper determines a career. Flip rates convert statistical bias into interpretable metrics directly linked to "acceptance rates," serving as a powerful argument for policy.

### Loss & Training
As this is an evaluation paper, no model training was performed. Key evaluation setup: 252 papers × 9 LLMs × 4 dimensions × N profile configurations. All models were released before the ICLR 2025 submission deadline, including Ministral 8B, DeepSeek-R1-Distill-Llama 8B, Llama 3.1 8B/70B, Mistral Small 22B, DeepSeek-R1-Distill-Qwen 32B, QwQ 32B, Gemini 2.0 Flash Lite, and GPT-4o Mini. Prompts were adapted from official ICLR reviewer guidelines to ensure a unified template.

## Key Experimental Results

### Main Results: Pairwise Win Rates for Affiliation and Gender (Partial, % RS / RW / tie and male / female / tie)

| Model | Rating | Affiliation (RS / RW / tie) | Gender@MIT (M / F / tie) |
|------|------|-----------------------------|--------------------------|
| Ministral 8B (Accepted) | Hard | 4.3 / 1.5 / 94.2 | 1.2 / 3.7 / 95.0 |
| Ministral 8B (Accepted) | **Soft** | **68.6 / 26.6 / 4.8** | 40.2 / 47.8 / 12.0 |
| Mistral Small 22B (Accepted) | Hard | 14.0 / 5.5 / 80.5 | 5.2 / 6.2 / 88.7 |
| Mistral Small 22B (Accepted) | **Soft** | **65.3 / 29.8 / 4.9** | 42.4 / 44.4 / 13.2 |
| Llama 3.1 70B (Accepted) | Hard | 1.7 / 1.1 / 97.2 | 1.6 / 1.8 / 96.6 |
| Llama 3.1 70B (Accepted) | **Soft** | **56.8 / 27.7 / 15.5** | 35.5 / 40.1 / 24.4 |
| QwQ 32B (Accepted) | Hard | 22.7 / 9.8 / 67.5 | 12.2 / 18.0 / 69.8 |
| QwQ 32B (Accepted) | **Soft** | **49.8 / 29.6 / 20.5** | 33.5 / 44.0 / 22.5 |
| Gemini 2.0 Flash Lite (Accepted) | Hard | 25.2 / 7.4 / 67.4 | 14.7 / 12.5 / 72.8 |
| GPT-4o Mini (Accepted) | Hard | 15.3 / 6.2 / 78.5 | 7.8 / 10.0 / 82.1 |

Blue (RS / Male) is significantly higher than red (RW / Female) across almost all models, with soft gaps being an order of magnitude larger than hard gaps.

### Ablation Study: Seniority, Publication History, and Decision Flips

| Dimension | Key Finding | Representative Win Rate |
|------|---------|------------|
| Seniority (Senior PI vs. UG) | All models prefer Senior PI | 6–15% for small models; >25–45% for Mistral Small / QwQ / Gemini / GPT-4o Mini on accepted papers |
| Publication History (100 TTP vs. 0 TTP) | All models prefer 100 TTP | Each model shows at least 20–50% bias towards 100 TTP; reverse bias is almost non-existent |
| Accept→Reject Flip (RW affil) | RW metadata causes accepted papers to be rejected | QwQ-32B: 7.9% accepted → rejected |
| Reject→Accept Flip (RS affil) | RS metadata causes rejected papers to be accepted | QwQ-32B: 21.4% rejected → accepted |
| Sub-field Consistency | RS-over-RW holds across all 21 sub-fields | Occasional counter-examples only in Cognitive Science / LLMs sub-fields |

### Key Findings
- **Hidden bias is an order of magnitude larger than explicit bias**: Using Ministral 8B Affiliation as an example, the hard rating RS win rate of 4.3% vs. the soft rating RS win rate of 68.6% (14×) indicates that alignment only shifts the mode but not the entire distribution; the model "internally" is much more biased than its "outward" behavior.
- **Institution weight > Seniority > Publication history > Gender**: Affiliation is the most stable and strongest source of bias. Gender bias is inconsistent across models (Gemini leans male, GPT-4o leans female, Mistral Small strongly leans female), suggesting that different alignment strategies applied "overcompensation" on the gender dimension.
- **Significant flip rates**: QwQ-32B flips 21.4% of originally rejected papers to accepted under RS; this means LLM-assisted reviews act as "author identity deciders" in the borderline zone.
- **Alignment may introduce new biases**: Mistral Small's strong preference for female authors and GPT-4o Mini's preference for minority institutions are typical signs of "overcompensation" in fairness fine-tuning, indicating that naive debiasing can create new reverse biases.
- **Bias is directly visible in review text**: Qualitative analysis shows Gemini explicitly stating "The University of Lagos affiliation is a concern, raises a flag for potential resource constraints," whereas DeepSeek-R1 is relatively neutral; this provides chain-of-thought evidence for why models are biased.

## Highlights & Insights
- **Soft rating as a new metric for alignment auditing**: Standardizing the use of "internal token distributions" to evaluate LLM fairness is the most transferable contribution of this paper—this method can be used to audit all LLM decision systems (judges, recruiters, graders), representing a paradigm shift in fairness/alignment evaluation.
- **Empirical evidence of "alignment masks but does not eliminate"**: By providing rigorous quantitative evidence that "hard ratings appear fair while soft ratings remain biased," the paper delivers a methodological impact to the alignment community regarding the limitations of RLHF / instruction tuning.
- **Policy-oriented flip rate analysis**: Translating rating differences into "borderline flip rates" directly links fairness research to real-world academic decisions, making it a compelling argument for policy makers and organizers.
- **Horizontal comparison of 9 LLMs**: Covering a wide range from 8B to 70B, dense to distilled, and open to closed-source, the finding that "institutional bias is universal while gender bias is model-specific" is highly credible.
- **Revealing overcompensation**: The reverse preferences of Mistral Small / GPT-4o Mini remind the community that debiasing is not a monotonic improvement and requires more granular multi-dimensional evaluation.

## Limitations & Future Work
- **Single-blind setting**: This paper only evaluates scenarios where metadata is visible; it does not evaluate whether LLMs can infer author identity from writing styles or citation patterns in a true double-blind setting.
- **Synthetic profiles simplify reality**: Institutional pairs, names, and TTP counts are synthetic; real-world metadata is more complex (including collaboration networks, Google Scholar profiles, etc.).
- **Limited to CS**: All 252 papers are from ICLR 2025; generalizability to biomedical, economics, or humanities fields remains unverified.
- **Lack of prompt engineering ablation**: Only one official guideline template was used; whether different prompt styles amplify or mitigate bias was not tested.
- **Lack of debiasing solutions**: The paper provides a diagnosis but no cure; future work could use this framework to evaluate various post-training / inference-time debiasing methods (e.g., prompt-based fairness instructions, internal-distribution calibration).
- **Future Directions**: (1) Injecting soft rating supervision signals into alignment losses to make "internal distribution fairness" a new target; (2) Establishing benchmarks for double-blind LLM reviewers; (3) Testing whether bias amplifies in "agentic multi-round debates / meta-review" scenarios.

## Related Work & Insights
- **vs. Pataranutaporn et al. (2025)**: They conducted an observational audit in economics; this paper provides a counterfactual audit + four-way decomposition in CS, offering finer granularity.
- **vs. Ye et al. (2024) / Zhu et al. (2025) (DeepReview)**: They found LLMs favor "famous authors"; this work further decomposes "fame" into affiliation, seniority, and publication history, quantifying their individual contributions.
- **vs. von Wedel et al. (2024)**: Affiliation bias was found in medical abstracts; this paper extends this to full CS papers and soft ratings, making the conclusions more robust.
- **vs. Liang et al. (2024a)**: They found LLM-modified content has permeated ICLR/NeurIPS reviews; this paper quantifies the fairness risks associated with that phenomenon.
- **vs. Wan et al. (2023) and others**: This paper extends "LLM social bias" research from generation tasks to high-stakes decision tasks, serving as a landmark work in the intersection of fairness and peer review.

## Rating
- Novelty: ⭐⭐⭐⭐ Counterfactual evaluation is comprehensively applied to LLM reviews (4 dimensions + dual scoring) for the first time; soft rating as a hidden bias metric is particularly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ 9 LLMs × 4 intervention dimensions × 252 real ICLR papers × 21 sub-fields; the scale and dimensions are solid, with complete pairwise heatmaps and statistical tests in the appendix.
- Writing Quality: ⭐⭐⭐⭐ Table 1 has high information density; qualitative counter-examples directly quote review texts, enhancing readability. Logical arguments in some subsections are slightly rushed.
- Value: ⭐⭐⭐⭐⭐ Provides hard evidence that "LLM-assisted review cannot be blindly trusted," which is directly relevant to policies at ICLR/AAAI/ICML and serves as a vital methodological reminder for the alignment community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Who Gets Which Message? Auditing Demographic Bias in LLM-Generated Targeted Text](who_gets_which_message_auditing_demographic_bias_in_llm-generated_targeted_text.md)
- [\[ACL 2026\] Confident, Calibrated, or Complicit: Safety Alignment and Ideological Bias in LLM Hate Speech Detection](confident_calibrated_or_complicit_safety_alignment_and_ideological_bias_in_llm_h.md)
- [\[AAAI 2026\] Bias Association Discovery Framework for Open-Ended LLM Generations](../../AAAI2026/social_computing/bias_association_discovery_framework_for_open-ended_llm_generations.md)
- [\[ACL 2026\] Phase Transitions in Affective Meaning Divergence: The Hidden Drift Before the Break](phase_transitions_in_affective_meaning_divergence_the_hidden_drift_before_the_br.md)
- [\[ACL 2026\] GKnow: Measuring the Entanglement of Gender Bias and Factual Gender](gknow_measuring_the_entanglement_of_gender_bias_and_factual_gender.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Inertia in Moral and Value Judgments of Large Language Models
description: >-
  [ACL 2026][Social Computing][Role-play] This paper systematically measures 7 mainstream LLMs using a "large-scale random persona × moral/value questionnaire" paradigm. It reveals highly stable "value inertia" in the Harm…
tags:
  - "ACL 2026"
  - "Social Computing"
  - "Role-play"
  - "Value Inertia"
  - "Steerability"
  - "Persona Injection"
  - "Moral Foundations"
date: 2026-05-08
content_hash: df5bd6292ebfc0d2
---

# Inertia in Moral and Value Judgments of Large Language Models

**Conference**: ACL 2026  
**arXiv**: [2408.09049](https://arxiv.org/abs/2408.09049)  
**Code**: To be confirmed (the paper states it will be public at camera-ready)  
**Area**: LLM Alignment / Values / Safety / Evaluation  
**Keywords**: Role-play, Value Inertia, Steerability, Persona Injection, Moral Foundations

## TL;DR
This paper systematically measures 7 mainstream LLMs using a "large-scale random persona × moral/value questionnaire" paradigm. It reveals highly stable "value inertia" in the Harm/Fairness dimensions—where almost no persona can shift the model's response direction—and proposes two quantifiable metrics, Inertia Index and Steerability, to show that this preference is unevenly distributed and aligned with safety goals.

## Background & Motivation

**Background**: Persona injection is the most common controllable generation method for users who cannot directly fine-tune models—placing "You are a 70-year-old retired worker" into a prompt to expect responses from that demographic's perspective. The research community also widely uses value questionnaires like MFQ/PVQ to detect whether LLMs can simulate diverse human moralities.

**Limitations of Prior Work**: Related work has found that LLM value expressions remain surprisingly stable across different prompt variations. However, these studies mostly focus on single personas or single questions, lacking a systematic "large-scale random persona × multi-model × multi-dimension" measurement. Furthermore, "stability" has not been converted into comparable scalar metrics—it remains unknown which dimensions are stable, to what extent, and whether this stability is intentional (result of alignment) or due to data bias.

**Key Challenge**: The design assumption of persona injection is that "the prompt determines the output distribution." However, RLHF and pre-training have "hard-coded" certain value directions within the model, creating a tension of "surface diversity, core consistency." Even when asking the same question with a thousand different personas, the answers still cluster within the same range.

**Goal**: (1) Establish a reproducible and scalable role-play-at-scale methodology; (2) Provide two quantitative metrics to locate the degree of inertia; (3) Distinguish which dimensions of high inertia are desired by alignment and which represent potential "under-representation" issues.

**Key Insight**: Amplify persona injection from a "specific behavior trigger" into a "large-scale sampler," borrowing from clustering concepts—while individual points may fluctuate, the group mean is highly concentrated. That central concentration represents the model's default orientation.

**Core Idea**: Use 200 random personas × multiple models × MFQ-30 + PVQ-RR. Combine this with the Inertia Index $I(d) = 1 - H(p_d)/\log_2 6$ and Steerability JSD to turn "how easily an LLM can be shifted by a persona in different moral dimensions" into comparable scalars.

## Method

### Overall Architecture

Input: (a) Random personas sampled with "equal probability per category" based on 11 attributes derived from the World Values Survey (WVS) (gender, age 20-80, income 1-10, parents, marital status, education, employment, occupation, ethnicity, religion, country); (b) 6-point Likert items from MFQ-30 (covering Harm / Fairness / Ingroup / Authority / Purity) and PVQ-RR (covering Schwartz's 10 universal value dimensions). Output: A 1-6 integer score for each (persona, item) pair, extracted from LLM free text by a Claude 3 Haiku parser.

Main Process: Each persona and item are independently combined into a prompt (no dialogue history), with "Your response should always point to a specific letter option." forced at the end. For each model, 200 unique personas are run against all items; this is repeated with 3 different seeds (111/333/555) to verify that results are not accidental to a specific persona set. A baseline without personas is used as a reference.

Tested 7 models: Claude 3 Opus / Sonnet / Haiku, GPT-4o, GPT-3.5 Turbo, LLaMA-3 70B Inst, LLaMA-3 8B Inst.

### Key Designs

1.  **Role-Play-at-Scale (Macro vs. Micro Dual-Layer Observation)**:
    - **Function**: Expands persona prompting from "specific behavior triggers" to a "large-scale sampler" to see if LLMs can truly be pushed in different directions by personas.
    - **Mechanism**: 200 independent random personas are sampled per model per questionnaire. Microscopic views (heatmaps: x-axis persona, y-axis item, color for options) and macroscopic views (mean and distribution per dimension) are analyzed. The appearance of "horizontal stripes" serves as visual evidence that options are independent of personas. Meanwhile, correlation coefficients > 0.99 across three random seeds exclude randomness in persona configuration.
    - **Design Motivation**: Traditional persona experiments ask "Can the model play X?"; this study asks "Regardless of what persona is given, where does the model naturally land?"—allowing the location of "internal preferences" rather than "role compliance."

2.  **Inertia Index + Steerability Dual Metrics**:
    - **Function**: Turns "inertia strength" into a scalar comparable across models and dimensions.
    - **Mechanism**: For each dimension $d$, let $p_d$ be the distribution of answers for items in that dimension across all personas. The Inertia Index $I(d) = 1 - H(p_d)/\log_2 6 \in [0,1]$—where larger values represent more concentrated answers (measuring response collapse using the normalized complement of Shannon entropy). Steerability uses the Jensen-Shannon Divergence $\text{JSD}(p_d^{\text{base}}, p_d^{\text{persona}})$ between the baseline and persona-injected distributions; smaller values indicate the persona is less able to shift that dimension.
    - **Design Motivation**: The authors state that formal definitions of LLM values are missing, so they use behavioral consistency as a "first step in mechanistic research." The two metrics are complementary—Inertia alone might misidentify an "inherently extreme model" as "locked-in," while Steerability distinguishes "built-in bias" from "prompt failure."

3.  **Selective Permeability Analysis**:
    - **Function**: Detects which dimensions the LLM is sensitive to regarding specific attributes, and where it is numb to all attributes.
    - **Mechanism**: Conditional sampling on PVQ-RR based on single attributes (religion, ethnicity, gender, etc.) is performed. Cohen's $d$ effect sizes are calculated for each (attribute, dimension). It was found that religion has a large effect on the Tradition dimension ($d=1.42$, moving from 2.48 for non-religious to 4.32 for Orthodox), but only $d=0.32$ on Universalism. Gender effects were $\leq 0.17$ across all dimensions. Additionally, a Pearson correlation of $r = 0.77$ between original and randomized item orders rules out "inertia as item-order bias."
    - **Design Motivation**: Means or entropy alone cannot reveal "which attributes should cause which dimensions to change." If gender causes Harm scores to shift, it looks like discrimination; if religion shifts Tradition, it looks like reasonable representation. This analysis provides insights for alignment design on "which inertia is desired and which should be fixed."

### Loss & Training

This is an evaluation study and does not train models; all data is gathered via black-box APIs. The only "model" is the Claude 3 Haiku parser used to map free text to 1-6 integers. The accuracy of 5 candidate parsers was 93-100%. The paper verified that the parser itself does not introduce significant bias (Haiku ranked 5th in inertia among 7 models; the two highest were non-Claude models).

## Key Experimental Results

### Main Results (Inertia per MFQ-30 dimension, averaged over 7 models × 3 seeds)

| Dimension | Inertia Index $I(d)$ | Steerability JSD | Top-2 Concentration (%) |
| :--- | :--- | :--- | :--- |
| Fairness | **0.499** | **0.288** | **90.6** |
| Harm | **0.460** | **0.285** | **88.5** |
| Ingroup | 0.201 | 0.470 | 68.1 |
| Authority | 0.186 | 0.476 | 66.0 |
| Purity | 0.166 | 0.432 | 61.9 |

Overall, 60% of responses converge to a single option, exceeding 95% in extreme cases. For Harm/Fairness, ~90% of responses fall within two adjacent Likert points. Mean correlations across three seeds were > 0.99 (e.g., GPT-4o 0.997, $p < 0.001$), confirming that inertia is an inherent property of the model rather than a byproduct of the persona set.

### Ablation Study / Robustness

| Configuration | Phenomenon | Conclusion |
| :--- | :--- | :--- |
| Full role-play-at-scale | 60% avg. single-option concentration | Baseline |
| 3-seed resampling (111/333/555) | Inter-model correlation 0.989-0.997 | Excludes persona randomness |
| Item order randomization (MFQ-30, 60 personas) | Pearson $r=0.77$; Harm +0.60→+0.14, Authority -0.54→-0.08 | Item order has an effect but macro direction is stable |
| Forced choice vs. No forced choice | Spearman $\rho = 0.90$-0.98 | Forced choice merely surfaces existing internal rankings |
| Tradition conditioned on Religion | $d = 1.42$ | Selective permeability in culturally-coupled dimensions |
| Universalism conditioned on Religion | $d = 0.32$ | Safety-related dimensions barely move even with strong attributes |
| All dimensions conditioned on Gender | $d \leq 0.17$ | Overall gender effect is negligible |

### Key Findings

-   Inertia distribution aligns closely with alignment goals: Harm + Fairness are dimensions heavily reinforced during RLHF, and they exhibit the highest inertia and lowest steerability—this is essentially "successful alignment" that the authors suggest should be preserved.
-   However, the same inertia suppresses dimensions like Authority/Tradition that should reasonably vary across cultures. Models do respond to religious attributes ($d=1.42$), but the center of preference still skews toward Western individualism. The judgment that "whether inertia is desirable depends on the dimension itself" is the paper's most valuable takeaway.
-   The more role-play iterations, the smaller the variance (Figure 5): After 500 personas, most dimension variances stabilize. This proves that small-sample persona experiments only provide noisy signals; researching LLM preferences requires a larger N than previously thought.
-   Forced choice "manifests" rather than "creates" inertia—the Spearman $\rho$ between baseline and persona conditions for dimension rankings is 0.90-0.98.

## Highlights & Insights

-   The dual perspective of "macro aggregation vs. micro fluctuation" is elegant—while individual personas may show signs that "the model seems to be role-playing," shifting to the mean of 200 personas shows the core remains unchanged. This macro lens should become the default paradigm for evaluating LLM values and biases.
-   Inertia Index + Steerability are genuine contributions to the evaluation community, offering much higher dimensionality than binary "Can it play X?" questions. Future alignment reports should include these metrics.
-   The "Desirable inertia vs concerning inertia" distinction categorizes alignment safety and cultural representation into different quadrants, making it both policy and engineering friendly—developers know to "keep high inertia for Harm/Fairness, fix high inertia for Tradition/Authority."
-   Applying Cohen's $d$ to an (Attribute, Dimension) grid to identify "cells that should respond but don't" is a highly transferable diagnostic tool for auditing representative blind spots in RAG, Agents, and retrieval systems.

## Limitations & Future Work

-   Evaluation is single-turn; multi-turn dialogues, long backstories, or few-shot demonstrations might embed personas deeper into the model context. The authors explicitly state multi-turn is out of scope.
-   Personas are sampled with independent attributes, lacking intersectionality (e.g., the specificity of "South Asian + Female + Muslim"). Thus, the "persona effect" should be read as "conditioning under simplified role instructions."
-   The 11 WVS attributes represent a finite space and are sampled with equal probability per category rather than real demographic marginal distributions; "effect" data cannot be directly extrapolated to deployment scenarios.
-   Using an LLM (Claude 3 Haiku) as a parser introduces potential self-bias. The authors counter this by noting Haiku ranked 5th in inertia and parser accuracy was 93-100%, but still recommend using structured output APIs to access logits directly in the future.
-   No adversarial or jailbreak prompts were tested; the paper only proves that benign personas cannot shift the model, not that no prompt can.

## Related Work & Insights

-   **vs. Kovač et al. 2024 (LLM value stability)**: They also observed LLM value stability across prompt variations; this paper turns "stability" into quantifiable Inertia/Steerability metrics decomposed by dimension.
-   **vs. Mazeika et al. 2025 (emergent utility systems)**: Both find embedded preferences in LLMs; this paper approaches from behavioral evaluation while they approach from mechanistic utility, providing a complementary view.
-   **vs. Russo et al. 2025 (human-LLM moral gap)**: They quantify the gap; this paper provides a mechanistic explanation for why the gap is hard to close (inertia dimensions cluster in alignment reinforcement zones).
-   **vs. Role-playing benchmarks (CharacterEval / RoleLLM)**: They focus on "whether a specific role can be played," while this paper focuses on "where the statistical landing point is across all roles," making the methodologies complementary.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination methodology of quantitative metrics + large-scale personas is novel, though the observation of "LLM value stability" has appeared previously.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 7 models × 2 questionnaires × 200 personas × 3 seeds, plus PVQ-RR Cohen's $d$, item order randomization, forced choice ablation, and parser self-consistency, effectively closing almost all confounders.
- Writing Quality: ⭐⭐⭐⭐ The chain of argumentation is tight, and the Discussion clearly distinguishes "when inertia should exist vs. when it shouldn't." Minor points are lost as some key figures (heatmaps, variance curves) are hidden in the appendix.
- Value: ⭐⭐⭐⭐⭐ Direct impact on alignment, controllable generation, social simulation, and AI governance. The Inertia Index can be integrated into any alignment evaluation pipeline almost immediately.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] SPAGBias: Uncovering and Tracing Structured Spatial Gender Bias in Large Language Models](spagbias_uncovering_and_tracing_structured_spatial_gender_bias_in_large_language.md)
- [\[ACL 2026\] Probing Multimodal Large Language Models on Cognitive Biases in Chinese Short-Video Misinformation](probing_multimodal_large_language_models_on_cognitive_biases_in_chinese_short-vi.md)
- [\[ACL 2026\] Why Are We Moral? An LLM-based Agent Simulation Approach to Study Moral Evolution](why_are_we_moral_an_llm-based_agent_simulation_approach_to_study_moral_evolution.md)
- [\[ICLR 2026\] Propaganda AI: An Analysis of Semantic Divergence in Large Language Models](../../ICLR2026/social_computing/propaganda_ai_an_analysis_of_semantic_divergence_in_large_language_models.md)
- [\[ICML 2026\] Self-Debias: Self-correcting for Debiasing Large Language Models](../../ICML2026/social_computing/self-debias_self-correcting_for_debiasing_large_language_models.md)

</div>

<!-- RELATED:END -->

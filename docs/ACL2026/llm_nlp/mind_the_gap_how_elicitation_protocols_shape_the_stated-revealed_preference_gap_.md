---
title: >-
  [Paper Note] Mind the Gap: How Elicitation Protocols Shape the Stated-Revealed Preference Gap in Language Models
description: >-
  [ACL 2026][LLM/NLP][stated-revealed preference] The authors upgrade forced-choice to expanded-choice (allowing "Equal Preference" and "Depends" neutral options) within the LitmusValues / AIRiskDilemmas framework. Systema…
tags:
  - "ACL 2026"
  - "LLM/NLP"
  - "stated-revealed preference"
  - "value alignment"
  - "elicitation protocol"
  - "neutrality"
  - "abstention"
date: 2026-05-08
content_hash: 3468f832880a429c
---

# Mind the Gap: How Elicitation Protocols Shape the Stated-Revealed Preference Gap in Language Models

**Conference**: ACL 2026  
**arXiv**: [2601.21975](https://arxiv.org/abs/2601.21975)  
**Code**: https://github.com/SPAR-SvR/Mind-the-Gap (Available)  
**Area**: LLM Evaluation / AI Alignment / Preference Modeling  
**Keywords**: stated-revealed preference, value alignment, elicitation protocol, neutrality, abstention

## TL;DR
The authors upgrade forced-choice to expanded-choice (allowing "Equal Preference" and "Depends" neutral options) within the LitmusValues / AIRiskDilemmas framework. Systematically evaluating 24 LLMs, they find that allowing neutrality on the stated side significantly improves the Stated-Revealed (SvR) Spearman correlation $\rho$ from $\sim 0.2$ to $\sim 0.7$ (as it filters out weak signals where models lack a stance). Conversely, allowing neutrality on the revealed side drops $\rho$ to near zero or even negative values (as many models select "Depends/Equal" almost exclusively in contextual scenarios). They also verify that system prompt steering based on stated rankings is generally unreliable across a large 16-value set. The conclusion is that the **SvR gap is highly dependent on the elicitation protocol, and preference evaluation must explicitly model states of "indeterminacy."**

## Background & Motivation

**Background**: As LLM agentic deployment increases, researchers are shifting from "measuring capabilities" to "measuring propensities." A systemic gap exists between the values LLMs endorse in abstract questionnaires (stated preference) and the actions they actually choose in contextualized moral dilemmas (revealed preference) (quantified as Spearman $\rho$ in LitmusValues by Gu et al. 2025; Liu et al. 2025; Chiu et al. 2025).

**Limitations of Prior Work**: (i) Existing evaluations almost exclusively use forced binary choice, compressing "strong preference / weak preference / neutral / uncertain" into a binary output, which conflates elicitation artifacts with "true preferences"; (ii) Recent studies (Khan et al. 2025; Balepur et al. 2025) warn that binary forced-choice introduces significant framing sensitivity; (iii) Liu et al. 2025 showed that prompt-based steering gains 23% on a 3-value HHH set but only 4% on a 6-value ModelSpec set, suggesting steering may fail on larger value sets, though this has not been systematically verified.

**Key Challenge**: Measuring value priority requires inducing trade-offs ("Honesty vs. Helpfulness"), but the forced-choice nature of trade-offs artificially manufactures decisive preferences. Evaluation must simultaneously push models to take a stance while providing a "way out" for neutrality—two naturally opposing goals.

**Goal**: (a) Quantify the impact of different combinations of elicitation protocols (forced vs. expanded, stated vs. revealed) on SvR $\rho$; (b) Examine whether system prompt steering can narrow the SvR gap in a large 16-value set.

**Key Insight**: The authors adopt standard practices from survey methodology (Krosnick 1991), using "allowing neutrality + calculating ranking after excluding neutral responses" as the core modification to the evaluation protocol, performing a $2 \times 2$ analysis by independently enabling/disabling this option on the stated and revealed sides.

**Core Idea**: SvR correlation is not an inherent model property but rather a protocol artifact. Allowing abstention on the stated side "squeezes out" preferences the model is truly confident in (increasing signal-to-noise ratio), while allowing it on the revealed side "drains" decisive signals (degrading the rank). This asymmetry reveals a fundamental methodological issue in current alignment evaluation.

## Method

### Overall Architecture
Based on the LitmusValues framework (Chiu et al. 2025): (1) 16 "Shared AI Values" (Truthfulness, Privacy, Justice, Protection, etc., extracted from Anthropic Claude Constitution and OpenAI Model Spec); (2) 3000 second-person contextualized binary moral dilemmas from AIRiskDilemmas. Stated rank is derived from pairwise win rates between values; Revealed rank is converted from Elo ratings where "choosing an action for a specific value = win" in a dilemma. This paper expands choice options from binary to "{A, B, C=Equal Preference, D=Depends/Cannot Decide}". GPT-4o-mini is used as an LM judge to parse response categories. To calculate SvR, binary preferences are retained while excluding Equal/Depends responses to compute Spearman $\rho$. All generations use deterministic decoding (temperature=0, top\_p=0.01).

### Key Designs

1.  **Independent Dual-Side Switching of Expanded-Choice Elicitation**:
    - **Function**: Decouples the traditional forced protocol into two independent dimensions: stated and revealed, creating a $2 \times 2$ configuration matrix.
    - **Mechanism**: The authors evaluate three meaningful configurations: forced-forced (baseline), expanded-stated + forced-revealed, and expanded-expanded. For the stated side, 5 symmetric prompt templates ("When v1 and v2 are in tension...") are used, permuting all $P_2^{16}$ pairs with 5 elicitations per pair. For the revealed side, an instruction block is prepended to each AIRiskDilemmas prompt to explicitly provide A/B/C/D options.
    - **Design Motivation**: Localizing the protocol change to a single axis (allowing neutrality) eliminates other confounding variables like framing or wording. Independent switching allows for attribution: if $\rho$ increases in (expanded-stated, forced-revealed) but collapses in (expanded-expanded), it proves that high neutrality rates on the revealed side are the noise source, not the neutral options themselves.

2.  **Neutrality-Aware Rank Calculation + Capability Correlation**:
    - **Function**: Retains neutral responses for diagnostic purposes (to measure model "indeterminacy rates") while standardizing rank calculation by excluding them, following survey methodology.
    - **Mechanism**: For stated preferences, pairwise votes only count binary wins. For revealed preferences, Elo adjustments are only calculated for binary actions across 3000 dilemmas, skipping C/D responses. Spearman $\rho$ is then used to compare 1–16 rankings. Neutrality rates are reported as a diagnostic metric. Finally, $\rho$ is correlated with the Epoch Capabilities Index.
    - **Design Motivation**: Survey literature (Krosnick 1991) demonstrates that forcing indeterminate responses destroys rank density. The authors apply this but add a new observation: keeping neutrality as a diagnostic rather than forcing it into a binary choice better reflects the model’s true preference distribution.

3.  **System Prompt Steering Counterfactual**:
    - **Function**: Tests whether injecting a model's own stated value ranking as a system prompt can bridge the SvR gap.
    - **Mechanism**: For each model, the 16-value ranking from the expanded-stated protocol is extracted. This ranking is written into a system prompt using a fixed template (including conflict resolution rules stating that higher-level values should always override lower-level ones) and prepended to each revealed elicitation prompt. The change in Spearman $\rho$ between steered and unsteered versions is then compared.
    - **Design Motivation**: While Liu et al. 2025 showed steering is effective for 3-value and 6-value sets, the authors suspect LLMs may fail to maintain this in context for 16 values. This acts as a stress test: if steering works, the SvR gap is due to the model "not knowing its own priorities"; if not, the gap signifies a deeper alignment issue.

### Loss & Training
This is an evaluation paper; no training is performed. Key hyperparameters: 5 stated prompt templates, permuting $P_2^{16}=240$ value pairs $\times$ 5 templates = 1200 stated elicitations per model; 3000 dilemmas $\times$ (forced + expanded) = 6000 revealed elicitations per model; 24 LLMs $\times$ 3 protocols + steering experiments.

## Key Experimental Results

### Main Results
Summary of SvR Spearman $\rho$ for 24 LLMs across three protocols:

| Protocol Configuration | Typical SvR $\rho$ Range | Representative Model Change | Description |
| :--- | :--- | :--- | :--- |
| Forced-Stated + Forced-Revealed (baseline) | High variance; no significant capability correlation | LLaMA-3.1-405B-Instruct $\rho \approx 0.2$ | Forced binary choice; mixes in elicitation artifacts |
| **Expanded-Stated + Forced-Revealed** | Significant improvement | LLaMA-3.1-405B-Instruct $\rho \approx 0.7$ | Filters weak stated signals; ranking is more robust |
| Expanded-Stated + Expanded-Revealed | $\rho \approx 0$ or negative | Most models | High neutrality on revealed side destroys rank |

SvR $\rho$ vs. Epoch Capabilities Index (n=16 models):

| Protocol Configuration | Spearman $\rho_{\text{capability}}$ | p-value |
| :--- | :--- | :--- |
| Forced + Forced | $-0.20$ | 0.47 (NS) |
| **Expanded-Stated + Forced-Revealed** | **$+0.58$** | 0.02 (Significant) |
| Expanded + Expanded | $-0.04$ | 0.88 (NS) |

$\rightarrow$ The conclusion that "stronger models $\rightarrow$ higher SvR consistency" only holds under the expanded-stated + forced-revealed protocol; in other protocols, there is no correlation between capability and alignment.

### Ablation Study
Neutrality rates (percentage of "Equal / Depends" responses under expanded-choice, 24 models):

| Model Family | Stated Neutrality | Revealed Neutrality | Notes |
| :--- | :--- | :--- | :--- |
| Qwen-3-8B | $\sim 100\%$ (All Depends) | — | Stated rank nearly impossible to determine |
| Mistral-3-8B Variants | High | $\sim 100\%$ | Revealed side completely paralyzed; excluded |
| Gemma-3-4B | Medium | $\sim 70\%$ (Equal Preference) | Binary signal is sparse on revealed side |
| LLaMA-3.1 / LLaMA-4 | Medium | Low (Retains binary actions) | Only family capable of ranking on both sides |
| Overall Range | 48.2% – 100% | Varies greatly by family | — |

System prompt steering effect ($\Delta \rho$ vs. unsteered baseline):

| Model Family | Steering Effect |
| :--- | :--- |
| Ministral-3B, Gemma-3-4B | Slight positive (few exceptions) |
| Claude Family | **Consistent regression** ($\rho$ decreases) |
| Most other models | Neutral or negative |

$\rightarrow$ On a 16-value set, simple system prompt steering is not only unreliable but often detrimental, aligning with the pattern that steering weakens as the value set size grows.

### Key Findings
- **Asymmetry is the core conclusion**: allow-neutrality-in-stated acts to "filter noise," whereas allow-neutrality-in-revealed acts to "destroy signal." The same modification has opposite effects depending on its position in the elicitation pipeline.
- **The capability-alignment link is protocol-conditional**: A significant positive correlation ($(\rho=0.58, p=0.02)$ exists only under expanded-stated + forced-revealed, implying previous findings regarding larger models being "more aligned" may be protocol artifacts.
- **Some models fail to rank on the revealed side**: Mistral-3-8B and Gemma-3-4B choose "Depends" almost entirely in contextual scenarios, suggesting these models may not have stable value hierarchies but are instead context-dependent decision-makers—a warning sign for alignment.
- **Steering fails on large value sets**: The consistent drop in $\rho$ for the Claude family after steering suggests a systemic contradiction between self-reported value ranks and internal behavioral priors; simple prompt injection introduces "instruction conflict" noise.

## Highlights & Insights
- **Treating elicitation protocol as a first-class variable**: While alignment papers often treat protocols as "implementation details," this paper promotes them to "objects of study." Seeing $\rho$ swing from $+0.7$ to $-0.04$ for the same model and data highlights that the protocol itself can decide the conclusion.
- **Clever use of survey methodology**: Applying Krosnick’s 1991 standards for neutral response handling to LLM evaluation introduces mature psychometric methodology to NLP.
- **Honest falsification of "steering as a panacea"**: While many alignment papers assume prompt-level injection aligns behavior, this work provides empirical evidence that this is sensitive to the number of values, providing high value through negative results.
- **Solid reproducibility**: Uses deterministic decoding, 5-template aggregation, and open-source datasets/code, allowing the results to be replicated exactly.

## Limitations & Future Work
- Evaluation is limited to the LitmusValues framework (16 abstract values) and the AIRiskDilemmas dataset; neutrality patterns might vary across other value taxonomies (e.g., Schwartz Values, Moral Foundations).
- Using GPT-4o-mini as an LM judge for parsing response categories may introduce judge bias, although mitigated by 5 templates and majority voting.
- Only zero-shot prompting was tested; the effects of stronger alignment interventions like fine-tuning, DPO, or RLHF on the SvR gap remain unexamined—steering failure might only mean the prompt is not strong enough.
- No single "best protocol" recommendation is provided—different configurations serve different purposes (differentiation vs. exposing indeterminacy). Future work could involve multi-protocol joint reporting, treating SvR $\rho$, neutrality rate, and steering $\Delta \rho$ as an "alignment triptych."

## Related Work & Insights
- **vs. LitmusValues (Chiu et al. 2025)**: Directly extends LitmusValues from forced-choice to protocol-aware evaluation, questioning the baseline figures in the original work.
- **vs. Liu et al. 2025 (generative value conflicts)**: Liu provides evidence for steering success in small sets; this work proves it does not scale to 16 values, together establishing an initial "steering effectiveness scaling law."
- **vs. Mazeika et al. 2025 (utility engineering)**: Mazeika argues for coherent internal value systems in LLMs; this paper shows such coherence is highly elicitation-dependent, serving as a caveat to their conclusion.
- **vs. Balepur et al. 2025 / Khan et al. 2025**: These works warn of forced-choice artifacts; this is the first to systematically quantify the magnitude of these artifacts in value alignment evaluation.

## Rating
- Novelty: ⭐⭐⭐⭐ The perspective of protocol-as-variable and the introduction of expanded-choice are insightful, though the underlying elicitation framework is inherited.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluating 24 LLMs across 3 protocols plus steering and capability correlations is comprehensive, though limited to a single value taxonomy.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure with intuitive explanations of the protocols; every figure corresponds to an actionable conclusion.
- Value: ⭐⭐⭐⭐⭐ Significant impact on AI alignment evaluation methodology—any work using SvR benchmarks must now re-examine their protocol choices.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Mind the Gap: Removing the Discretization Gap in Differentiable Logic Gate Networks](../../NeurIPS2025/llm_nlp/mind_the_gap_removing_the_discretization_gap_in_differentiable_logic_gate_networ.md)
- [\[ACL 2026\] Clozing the Gap: Exploring Why Language Model Surprisal Outperforms Cloze Surprisal](clozing_the_gap_exploring_why_language_model_surprisal_outperforms_cloze_surpris.md)
- [\[ACL 2026\] CoSToM: Causal-oriented Steering for Intrinsic Theory-of-Mind Alignment in Large Language Models](costomcausal-oriented_steering_for_intrinsic_theory-of-mind_alignment_in_large_l.md)
- [\[ACL 2026\] SteerEval: How Controllable Are Large Language Models? A Unified Evaluation across Behavioral Granularities](how_controllable_are_large_language_models_a_unified_evaluation_across_behaviora.md)
- [\[ACL 2026\] Generative Interfaces for Language Models](generative_interfaces_for_language_models.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] GKnow: Measuring the Entanglement of Gender Bias and Factual Gender
description: >-
  [ACL 2026][Social Computing][Gender Bias] This paper proposes the **GKnow** benchmark along with a set of two-level mechanism analyses (circuit and neuron). It demonstrates that "stereotypical gender" and "factual gender…
tags:
  - "ACL 2026"
  - "Social Computing"
  - "Gender Bias"
  - "Factual Gender"
  - "Circuit Analysis"
  - "Neuron Ablation"
  - "EAP-IG"
date: 2026-05-08
content_hash: 1099d8f48cdd2843
---

# GKnow: Measuring the Entanglement of Gender Bias and Factual Gender

**Conference**: ACL 2026  
**arXiv**: [2605.12299](https://arxiv.org/abs/2605.12299)  
**Code**: https://github.com/leonorv/gknow  
**Area**: Fairness / Gender Bias / Mechanistic Interpretability  
**Keywords**: Gender Bias, Factual Gender, Circuit Analysis, Neuron Ablation, EAP-IG

## TL;DR
This paper proposes the **GKnow** benchmark along with a set of two-level mechanism analyses (circuit and neuron). It demonstrates that "stereotypical gender" and "factual gender" in LLMs are highly overlapped at the circuit level (IoU/cross-task faithfulness) and share the same set of high-IG neurons at the neuron level. Consequently, simple "ablation of bias neurons" simultaneously weakens factual gender capabilities, appearing as "successful debiasing" only on bias-only benchmarks. This warns that such debiasing is unreliable.

## Background & Motivation

**Background**: The Mechanistic Interpretability (MI) community has recently used tools like causal mediation, edge attribution patching, and neuron attribution to locate internal components of "gender bias," using "ablation of the most relevant bias neurons" as a lightweight debiasing method (Liu et al. 2024, Yu & Ananiadou 2025, etc.). This approach is gaining popularity as it does not rely on expensive data or fine-tuning.

**Limitations of Prior Work**: Existing gender-related MI work (i) mostly focuses on a single task (most commonly "pronoun prediction"), and (ii) fails to distinguish between "factual gender" (semantic gender like woman→she, brother→he) and "stereotypical gender" (occupational bias like nurse→she, pilot→he). This leads to a hidden side effect—ablating "bias neurons" may damage the model's ability to identify *factual gender*, a drop in performance that remains invisible on single-dimension bias benchmarks.

**Key Challenge**: Bias signals and factual gender signals are likely entangled within model representations, but the community lacks systematic evidence at the circuit level. Furthermore, there is a lack of fine-grained benchmarks that simultaneously cover "factual/stereotypical × different gender prediction types."

**Goal**: (1) Build a fine-grained English gender benchmark GKnow, organized by "subject type × output type"; (2) Use EAP-IG to verify if factual vs. stereotypical circuits overlap at the circuit level; (3) Use IG neuron ablation to verify whether ablating bias neurons truly debiases without harming factual capabilities.

**Key Insight**: The authors argue that if entanglement is shown to be significant at the circuit/neuron level, then the "success" observed in ablation-based debiasing on benchmarks that do not evaluate factual gender is an illusion—a "good result" on StereoSet might be accompanied by the collapse of factual gender capabilities.

**Core Idea**: Treat "entanglement" as a measurable quantity (cross-task circuit faithfulness + shared high-IG neurons + dual factual vs. stereo ablation) and evaluate both aspects together using the unified GKnow benchmark.

## Method

### Overall Architecture
The methodological pipeline consists of three parts:
1. **Benchmark Construction**: GKnow covers 5 types of gender-related subjects (pronoun / gender word / gendered name / lexically gendered noun / stereotypically gendered noun) and 5 types of expected outputs, resulting in 25 subsets that segment "subject-output" semantic sources finely. It contains 91,490 examples in total, with 6,294 (train) and 698 (test) used in this work.
2. **Circuit-Level Analysis (Section 4)**: Uses EAP-IG to find the minimal faithful circuit (recovery $\geq 80\%$) for each GKnow subset, then calculates edge/node Jaccard IoU and cross-task faithfulness (the recovery level of circuit A when solving task B).
3. **Neuron-Level Analysis (Section 5)**: Uses Integrated Gradients (IG) to select top-N neurons on the `gender_prediction_based_on_stereo` subset, zero-ablates their activations, and measures $P_{exp}, P_{opp}, P_{other}, \%exp, \%opp, \%other, \Delta_{f,m}$ across GKnow Stereo/Factual, StereoSet, and DiFair Neutral/Specific.

Models evaluated include Llama-3.1-8B and Olmo-7B.

### Key Designs

1. **GKnow: A Fine-Grained Gender Benchmark with Subject × Output Axes**:
    - **Function**: Provides all combinations of "stereotypical vs. factual" + "pronoun/gender/name/lex/stereo" axes, allowing the same framework to compare the impact of debiasing on both factual and stereotypical aspects.
    - **Mechanism**: Defines the "subject" and "expected output" of prompts belonging to 5 categories of gender expression; combinations of blue (factual) and red (stereo) cells (e.g., `pronoun_prediction_based_on_stereo` like "The nurse is nice, isn't [she]") allow *any subset* to be tested as a "stereotypical" or "factual" task. Each sample includes `subject / expected_output / gender / stereo_category / id`.
    - **Design Motivation**: Existing benchmarks either only measure pronouns or only look at stereo, failing to answer "how much factual knowledge was removed by ablation." GKnow systematically aligns these two aspects for the first time.

2. **EAP-IG Circuits + Cross-Task Faithfulness to Measure Entanglement**:
    - **Function**: Quantifies from the circuit level whether "stereotypical circuits" and "factual circuits" are the same subgraphs.
    - **Mechanism**: EAP-IG scores each edge $(u,v)$ in the computational graph:
      $$(z'_u - z_u)\cdot \frac{1}{m}\sum_{k=1}^{m}\frac{\partial L(z' + \frac{k}{m}(z-z'))}{\partial z_v}$$
      ($z, z'$ are clean / corrupted activations, $m=5$). For each GKnow subset, top-N edges are greedily accumulated until faithfulness $\geq 0.8$; then Jaccard IoU and cross-task faithfulness (recovery when circuit A is embedded in task B) are calculated between subset pairs.
    - **Design Motivation**: Simple IoU might miss cases where "important edges are functionally equivalent but specifically different"; cross-task faithfulness directly measures "functional interchangeability," which is closer to the essence of entanglement. The paper shows faithfulness = 1.0 for the `gender_prediction_based_on_stereo` circuit on `gender_prediction_based_on_pronoun`, implying the stereotypical circuit can fully solve the factual task.

3. **Dual Factual vs. Stereo Evaluation of IG Neuron Ablation**:
    - **Function**: Observes "whether ablation truly debiases" and "whether ablation harms facts" on the same scale.
    - **Mechanism**: Selects top-10 / top-50 neurons using IG on the `gender_prediction_based_on_stereo` training subset → sets their activations to 0 → simultaneously measures a full suite of prediction-on-target + preference + prediction-gap metrics ($P_{exp}/P_{opp}/P_{other}/\%exp/\%opp/\%other/\Delta_{f,m}$) on GKnow Stereo, GKnow Factual, StereoSet, DiFair Neutral, and DiFair Specific.
    - **Design Motivation**: Traditional ablation work only reports "positive" metrics like "increase in $\%opp$" on stereo benchmarks. This design forces the reporting of the same ablation on factual benchmarks, exposing the side effects of "perceived debiasing."

### Loss & Training
The method does not update the model; there is no training loss. Neuron localization uses standard IG implementation ($m$-step integration), and ablation simply sets the corresponding FFN hidden neuron activations to 0. Circuits are greedily selected using EAP-IG on counterfactual prompts augmented from GKnow. Significance is determined via paired $t$-test ($p < 0.05$).

## Key Experimental Results

### Main Results (Llama-3.1-8B Neuron Ablation)

| Dataset | $P_{exp}$ (N=0) | $P_{exp}$ (N=50) | $\Delta_{f,m}$ (N=0) | $\Delta_{f,m}$ (N=50) | Conclusion |
|---------|-----------------|------------------|----------------------|------------------------|------------|
| GKnow Stereo | 67.66 | 47.03 (↓20.64) | 21.81 | 7.45 (↓14.36) | "Debiasing" looks effective |
| **GKnow Factual** | 91.49 | 79.86 (↓11.63) | 43.77 | 18.03 (↓25.76) | But factual capability degrades significantly |
| StereoSet | 65.26 | 62.60 (↓2.66) | — | — | Bias benchmark shows slight improvement |
| DiFair Neutral | — | — | 6.48 | 2.23 (↓4.25) | Confidence in gender prediction for neutral sentences plunges |
| **DiFair Specific** | — | — | 45.57 | 15.30 (↓30.26) | Sentences with explicit gender cues also collapse |

The trend for Olmo-7B is identical: GKnow Factual $P_{exp}$ drops from 90.07→82.80 (−7.27), and $\Delta_{f,m}$ drops from 40.01→23.37 (−16.64).

### Ablation Study (Circuit-Level Cross-Task Faithfulness)

| Setting | Key Metric | Description |
|---------|------------|-------------|
| Within same prediction type | avg faithfulness 77.2 (gender→gender) | Circuits are highly interchangeable between tasks of the same type |
| Across prediction types | avg 72.9 (gender→pronoun) | Over 72% recovery across tasks, strong circuit overlap |
| `stereo` → `pronoun` | **faithfulness = 1.0** | The stereo circuit can completely solve the factual task |
| `based_on_lex` → `based_on_stereo` | High faithfulness | The lex circuit has the strongest generalizability to the stereo subset |
| `based_on_name` circuit | Lowest recovery | The name-based circuit is the most specialized and least interchangeable |

Circuit IoU also maintains high Jaccard scores between stereo/factual subsets, matching the faithfulness conclusions.

### Key Findings
- **Entanglement extends from circuits to neurons**: High circuit IoU/faithfulness and shared top-IG neurons (Table 4 shows Olmo L31N8077 represents gender across lex/pronoun tasks with bottom tokens like `she/her/woman`) provide cross-scale evidence.
- **"Fake Success" of Debiasing**: Increased $\%opp$ and decreased $P_{exp}$ on StereoSet/GKnow Stereo would be interpreted as successful debiasing; meanwhile, DiFair Specific's $\Delta$ plunges from 45.57 to 15.30, indicating a collapse in factual gender capability hidden from bias-only benchmarks.
- **$P_{other}$ Increase Reveals "Decontextualization"**: At N=50, GKnow Stereo $\%other$ jumps from 0 to 30%, indicating ablation isn't switching "he" to "she," but pushing the model toward neutral/irrelevant tokens, thus harming general language capability.
- **Circuit Differences by Model**: Llama's mid-layer connections are attention-centric, while Olmo's are dominated by MLP-to-MLP connections (Figure 3), showing that the same gender concept follows different paths in different architectures.
- **Universality of the `based_on_lex` Circuit**: Since subjects cover various lexical genders (family/occupation/misc), this circuit has stronger transferability, suggesting lex subjects should be prioritized for building "universal gender probes."

## Highlights & Insights
- **Dual-axis benchmark design** is a simple but powerful research tool: By combining 5×5 subsets, it naturally supports a "dual-report" evaluation norm for any debiasing method.
- **First use of EAP-IG to prove entanglement**: While previous circuit work focused on "how the model stays correct," this study uses cross-task faithfulness as quantitative evidence for "functional entanglement," providing a template for mechanistic disentanglement research.
- **A critique of the "neuron ablation debiasing" school**: Liu et al. (2024) and Yu & Ananiadou (2025) reported positive results; this paper proves these are illusions caused by an evaluation gap. Future methods must report results on DiFair or GKnow dual-benchmarks.
- **Logit-lens validates interpretability**: Ablated neurons both "represent" gender (bottom tokens are she/her) and influence lex/pronoun tasks, providing concrete evidence for "one neuron carrying multiple semantic dimensions," contributing to the superposition hypothesis.

## Limitations & Future Work
- **English & Binary Gender Only**: Does not cover grammatical gender languages (e.g., German/Spanish) or non-binary pronouns; the authors acknowledge this "exacerbates the invisibility of non-binary identities in NLP."
- **Small Model Scale (7B / 8B)**: The degree of entanglement in larger models or different architectures (e.g., MoE) remains unknown.
- **Stereotype Coverage**: Includes occupational/adjectival stereotypes but lacks intersectional bias (religion, race) or implicit/pragmatic bias.
- **Single Ablation Form**: Only uses zero-ablation; a systematic comparison with mean ablation, optimal ablation, or SAE feature ablation is missing.
- **Template Bias**: GKnow uses a limited number of prompt templates from existing work, which might amplify biases in idiomatic patterns.

## Related Work & Insights
- **vs. DiFair (Zakizadeh et al. 2023)**: DiFair was the first disentangled bias/knowledge benchmark but used Wikipedia/Reddit sentences without mask-only targets; GKnow is a controlled version with prompt templates + 5×5 subsets better suited for circuit/neuron analysis.
- **vs. Liu et al. 2024 / Yu & Ananiadou 2025**: These works use IG/activation attribution to find "bias neurons" for ablation debiasing; this paper replicates their setup but adds factual evaluation to prove the trade-off.
- **vs. Limisiewicz & Mareček 2022 / Bolukbasi et al. 2016**: These tried "factual-preserving debiasing"; this paper explains from a mechanistic level why preservation is necessary—the circuits are inherently entangled, making naive ablation a lose-lose scenario.
- **vs. Chintam et al. 2023**: Used causal mediation to find bias components in GPT-2 small; this paper's EAP-IG + cross-task faithfulness approach is more direct for quantifying entanglement.

## Rating
- Novelty: ⭐⭐⭐⭐ Transforms entanglement from a "hypothesis" into quantifiable double-layer evidence; GKnow's dual-axis design is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Dual models × three benchmarks × multiple neuron counts × multiple metrics, plus logit-lens analysis; lacks larger models and diverse ablation forms.
- Writing Quality: ⭐⭐⭐⭐ Concept distinctions are clear; Tables 1-3 are highly informative; appendix details are sufficient.
- Value: ⭐⭐⭐⭐⭐ Directly challenges the popular "neuron ablation debiasing" path and provides a reusable benchmark, impacting future evaluation norms.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] SPAGBias: Uncovering and Tracing Structured Spatial Gender Bias in Large Language Models](spagbias_uncovering_and_tracing_structured_spatial_gender_bias_in_large_language.md)
- [\[NeurIPS 2025\] Auto-Search and Refinement: An Automated Framework for Gender Bias Mitigation in LLMs](../../NeurIPS2025/social_computing/auto-search_and_refinement_an_automated_framework_for_gender_bias_mitigation_in_.md)
- [\[ACL 2026\] Justice in Judgment: Unveiling (Hidden) Bias in LLM-assisted Peer Reviews](justice_in_judgment_unveiling_hidden_bias_in_llm-assisted_peer_reviews.md)
- [\[ACL 2026\] Confident, Calibrated, or Complicit: Safety Alignment and Ideological Bias in LLM Hate Speech Detection](confident_calibrated_or_complicit_safety_alignment_and_ideological_bias_in_llm_h.md)
- [\[ACL 2026\] Who Gets Which Message? Auditing Demographic Bias in LLM-Generated Targeted Text](who_gets_which_message_auditing_demographic_bias_in_llm-generated_targeted_text.md)

</div>

<!-- RELATED:END -->

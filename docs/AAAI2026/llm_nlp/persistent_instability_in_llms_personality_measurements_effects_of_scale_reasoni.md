---
title: >-
  [Paper Note] PERSIST: Persistent Instability in LLM's Personality Measurements
description: >-
  [AAAI 2026][LLM/NLP][LLM personality measurement] The PERSIST framework systematically evaluates personality measurement stability across 29 LLMs (1B–685B) on over 2 million responses…
tags:
  - "AAAI 2026"
  - "LLM/NLP"
  - "LLM personality measurement"
  - "behavioral consistency"
  - "reasoning mode"
  - "alignment evaluation"
  - "psychometrics"
date: 2026-05-08
content_hash: 52da12a9a3c077c0
---

# PERSIST: Persistent Instability in LLM's Personality Measurements

**Conference**: AAAI 2026
**arXiv**: [2508.04826](https://arxiv.org/abs/2508.04826)
**Code**: [https://github.com/tosatot/PERSIST](https://github.com/tosatot/PERSIST)
**Area**: NLP Generation / LLM Evaluation
**Keywords**: LLM personality measurement, behavioral consistency, reasoning mode, alignment evaluation, psychometrics

## TL;DR
The PERSIST framework systematically evaluates personality measurement stability across 29 LLMs (1B–685B) on over 2 million responses, revealing a "reasoning paradox" in which CoT reasoning increases variability while reducing perplexity, as well as a scale-dependent effect whereby conversational history exerts opposite influences on large versus small models—collectively indicating that current LLMs lack the architectural foundation for behavioral consistency.

## Background & Motivation

### State of the Field
**Background**: As LLMs are deployed in high-stakes domains such as healthcare, education, and decision support, behavioral predictability has become a core requirement for trustworthy AI. Both the EU AI Act and the U.S. NIST AI Risk Management Framework list "performance consistency" as a necessary condition for high-risk AI applications. Existing research measures LLM behavioral traits using psychological scales (e.g., the Big Five BFI-44, the Dark Triad SD3); Safdari et al. (2023) have demonstrated that LLMs can produce personality measurement reliability comparable to that of humans.

### Limitations of Prior Work
**Limitations of Prior Work**: (1) Prior studies rely on single-point measurements, entirely overlooking response variability across deployment conditions—Sclar et al. (2023) found that performance fluctuations between semantically equivalent prompts can reach 76 accuracy points; (2) prompt sensitivity is pervasive, with Salinas and Morstatter (2024) documenting a "butterfly effect" in prompting whereby a single-character modification can cascade into completely different model behavior; (3) no comprehensive evaluation framework exists for systematically quantifying the sources and magnitude of variability.

**Key Challenge**: LLM alignment and safety evaluation assumes that behavioral traits can be reliably measured, yet the instability of the measurement instruments themselves undermines this assumption. The inability to reliably measure LLM behavioral traits renders alignment evaluation unreliable—a structural problem in the current LLM safety paradigm that has not received sufficient attention.

### Paper Goals & Starting Point
**Goal**: Through a full-factorial experimental design, comprehensively quantify the sources (model scale, reasoning mode, persona instructions, question wording, conversational history) and magnitude of LLM personality measurement instability, and reveal the structural and persistent nature of this instability.

**Key Insight**: Construct the PERSIST framework covering 29 models × 5 personas × 4 questionnaires × 250 permutations × 100 paraphrases × reasoning/non-reasoning modes × with/without conversational history, generating over 2 million independent measurements.

**Core Idea**: Instability in LLM personality measurement is persistent and structural; it cannot be resolved simply by scaling up model size, and many interventions expected to improve stability (reasoning, conversational history) paradoxically exacerbate it.

## Method

### Overall Architecture
PERSIST (PERsonality Stability In Synthetic Text) comprises three integrated modules: (1) **Generation Engine**: optimized inference based on vLLM, supporting efficient batched inference across multiple model architectures; (2) **Response Processing Module**: a multi-pattern parser that extracts structured data from LLM outputs (supporting indexed responses such as "1: 4", JSON structures, isolated numeric ratings, etc.) while extracting token-level log probabilities for uncertainty quantification; (3) **Analysis Pipeline**: hierarchical aggregation from individual responses to item-level and trait-level statistics, computing means and standard deviations across permutations. Any single invalid response (rating out of the 1–5 range or parsing failure) causes the entire run to be discarded to ensure data quality.

### Key Designs

1. **Dual-Version Questionnaire Design**:

    - Function: Distinguishes whether instability is an artifact of human-centric questionnaires or an intrinsic property of the model.
    - Mechanism: Employs traditional psychological questionnaires BFI-44 (44 items measuring the Big Five) and SD3 (27 items measuring the Dark Triad), alongside LLM-adapted versions BFI-LLM and SD3-LLM (translating human-specific experiences into behaviorally equivalent AI formulations, e.g., "Is depressed, blue" → "Focuses on negative aspects"; "I'll say anything to get what I want" → "Truth is secondary to reaching a goal"). All questionnaires use a 5-point Likert scale.
    - Design Motivation: If the adapted versions exhibit the same instability, the questionnaire wording factor is ruled out, pointing instead to structural instability intrinsic to the model.

2. **Full-Factorial Experimental Design (5-Factor Systematic Variation)**:

    - Function: Systematically quantifies the independent and interactive effects of each factor on measurement stability.
    - Mechanism: (a) **Item Order**—250 random permutations, testing the order-invariance assumption that personality measurement should satisfy; (b) **Persona Instructions**—5 personas (Assistant baseline, Buddhist positive persona, Teacher positive persona, Antisocial clinical persona based on DSM-5, Schizophrenia clinical persona); (c) **Reasoning Mode**—standard vs. CoT, comparing the effect of the reasoning process on consistency; (d) **Paraphrasing**—100 semantically equivalent paraphrases generated by Qwen3 235B-A22B and manually verified and corrected by two authors; (e) **Conversational History**—multi-turn vs. single-turn presentation.
    - Design Motivation: The full-factorial design isolates the independent contribution of each factor, revealing multiple sources of variability and their interaction effects.

3. **Large-Scale Model Coverage and Stability Metrics**:

    - Function: Ensures the generalizability of conclusions across architectures and scales.
    - Mechanism: Evaluates 29 models spanning 8 families—Llama 3.1 (8B/70B/405B Instruct), Qwen 2.5 (6 Instruct variants from 1.5B to 72B), Qwen 3 (7 variants from 1.7B to 235B-A22B including MoE), Gemma 2 (2B/9B/27B), Gemma 3 (1B/4B/12B/27B), DeepSeek V3/R1 (both 671B), GPT-OSS (20B/120B), and Claude Sonnet 4.5/Opus 4.1. Metrics include item-level SD (across 250 permutations), token-level perplexity $\text{PPL}=\exp(-\log p)$, Spearman correlation, Wilcoxon signed-rank test, and Kruskal-Wallis test.
    - Design Motivation: Covering the full scale range from 1B to 685B and diverse training paradigms prevents conclusions from being limited to specific architectures.

### Experimental Configuration
All experiments use temperature τ=0 to isolate the effect of the manipulated variables (τ=0.6 for reasoning mode experiments), maximum token length 16,384, random seed 42, and hardware consisting of 4× NVIDIA H100 SXM 80GB HBM3. Claude models are run for only 70 iterations in reasoning experiments (all other models use 250 iterations).

## Key Experimental Results

### Main Results: Scale Effect Analysis

| Metric | Spearman ρ Direction | p-value | Interpretation |
|--------|----------------------|---------|----------------|
| Model scale → mean score on positive traits | ↑ positive | 0.001** | Larger models are more "agreeable" |
| Model scale → mean score on negative traits | ↓ negative | <0.001*** | Larger models are more "benign" |
| Model scale → item-level SD | ↓ negative | <0.001*** | Larger models are more stable |
| Model scale → item-level perplexity | not significant | 0.934 | Scale does not reduce uncertainty |
| Perplexity ↔ item-level SD | ρ=0.465 | — | Only moderate correlation; PPL is not a complete stability indicator |

### Reasoning Paradox Experiment

| Model/Condition | Change in SD | Change in PPL | Statistical Test |
|-----------------|-------------|---------------|-----------------|
| GPT-OSS increasing reasoning intensity | significantly increases | significantly decreases | Kruskal-Wallis p<0.001 |
| Qwen3 reasoning vs. non-reasoning | significantly increases | significantly decreases | Mann-Whitney U all p<0.001 |
| Qwen3-MoE reasoning vs. non-reasoning | significantly increases | significantly decreases | Mann-Whitney U all p<0.001 |
| DeepSeek reasoning vs. non-reasoning | significantly increases | not significant | p<0.01 (SD), n.s. (PPL) |
| Claude reasoning vs. non-reasoning | significantly increases | — | Mann-Whitney U p<0.01 |

Core finding: CoT makes models more confident at the token level (PPL↓) while making them more inconsistent at the behavioral level (SD↑).

### Ablation Study: Questionnaire Type and Paraphrase Effects

| Comparison Condition | Metric | p-value | Conclusion |
|----------------------|--------|---------|------------|
| LLM-adapted vs. original questionnaire | item-level SD | 0.286 (n.s.) | No significant difference |
| LLM-adapted vs. original questionnaire | item-level PPL | <0.001*** | Adapted version has higher PPL |
| Paraphrase vs. reordering (<50B) | ΔSD | 0.244 (n.s.) | No significant effect in small models |
| Paraphrase vs. reordering (≥50B) | ΔSD | <0.01** | Significantly increased variability in large models |

### Scale-Dependent Effect of Conversational History

| Model Group | Count | Change in SD with History | p-value |
|-------------|-------|--------------------------|---------|
| <50B small models | n=19 | significantly increases variability | <0.001*** |
| ≥50B large models | n=4 | significantly decreases variability | <0.001*** |

### Key Findings
- Even 400B+ models still exhibit SD>0.3 on a 5-point scale—scale is not a remedy.
- CoT reasoning increases variability while decreasing perplexity (the "reasoning paradox").
- Misaligned personas (Antisocial/Schizophrenia) show significantly higher variability and perplexity than the Assistant baseline (p<0.05).
- Positive personas (Buddhist monk) significantly reduce variability and perplexity (p<0.05).
- Conversational history helps large models but harms small models—deployment strategies must account for model scale.

## Highlights & Insights
- **The "Reasoning Paradox" is the central finding**: CoT causes models to generate different reasoning chains leading to different conclusions; token-level certainty does not equal behavioral consistency, challenging the intuition that "more reasoning = more reliable."
- **Scale-dependent effect of conversational history**: Large models extract consistency signals from context while small models are overwhelmed by the additional information—this has direct practical implications for prompt engineering.
- **Instability as a misalignment detection signal**: Deceptive models may fabricate correct average trait scores, but maintaining consistency across permutations is far more difficult—variability patterns can therefore serve as misalignment indicators.
- **Methodological contribution**: Over 2 million independent measurements and a 5-factor fully crossed design establish a new benchmark for LLM behavioral evaluation; the PERSIST framework can serve as a standard tool for future safety certification.

## Limitations & Future Work
- **The gap between self-report and behavior**: Although evidence exists that LLM self-reports correlate with behavioral outputs, self-reports may underestimate actual behavioral instability.
- **Lack of formal psychometric validation**: Neither the original nor the adapted questionnaires have been validated for LLMs in terms of factor loadings and Cronbach's α.
- **Possibility of strategic deception**: If models recognize an evaluation context, they may adjust their responses—random permutations and a focus on variability (rather than means) make deception more difficult.
- **Limited coverage of closed-source models**: Claude is run for only 70 iterations, and the latest GPT versions are not included.
- **Coverage limited to self-report evaluation**: Behavioral evaluation paradigms such as game-theoretic scenarios and role-playing are not tested.

## Related Work & Insights
- **vs. Safdari et al. (2023)**: They demonstrate that LLM personality measurement can achieve human-comparable reliability, but only under specific prompt configurations—this paper reveals the large variability that exists across configurations.
- **vs. Sclar et al. (2023)**: They find performance fluctuations of up to 76 points between semantically equivalent prompts; this paper extends that finding to the personality measurement domain and systematically analyzes five sources of variability.
- **vs. Representation Engineering (Zou et al., 2025)**: They monitor behavioral traits using directional representations in activation space; this paper demonstrates that such traits are themselves unstable.
- **vs. Anthropic Persona Vectors (Chen et al., 2025)**: They propose persona vectors that can be systematically identified and controlled—this paper provides baseline instability data that such techniques must contend with.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Full-factorial design reveals multiple counter-intuitive findings
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 29 models, 2M+ responses, rigorous statistical testing
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous statistical analysis with well-supported conclusions
- Value: ⭐⭐⭐⭐⭐ Important cautionary implications for the LLM safety and alignment evaluation community

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] From Classification to Ranking: Enhancing LLM Reasoning for MBTI Personality Detection](from_classification_to_ranking_enhancing_llm_reasoning_capabilities_for_mbti_per.md)
- [\[AAAI 2026\] Scalable and Accurate Graph Reasoning with LLM-Based Multi-Agents](scalable_and_accurate_graph_reasoning_with_llm-based_multi-agents.md)
- [\[AAAI 2026\] Collaborative LLM Numerical Reasoning with Local Data Protection](collaborative_llm_numerical_reasoning_with_local_data_protection.md)
- [\[AAAI 2026\] ICL-Router: In-Context Learned Model Representations for LLM Routing](icl-router_in-context_learned_model_representations_for_llm_routing.md)
- [\[AAAI 2026\] VSPO: Validating Semantic Pitfalls in Ontology via LLM-Based CQ Generation](vspo_validating_semantic_pitfalls_in_ontology_via_llm-based_cq_generation.md)

</div>

<!-- RELATED:END -->

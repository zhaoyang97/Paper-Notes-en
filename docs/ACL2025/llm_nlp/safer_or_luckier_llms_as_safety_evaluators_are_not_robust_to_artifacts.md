---
title: >-
  [Paper Note] Safer or Luckier? LLMs as Safety Evaluators Are Not Robust to Artifacts
description: >-
  [ACL2025][LLM (Other)][LLM-as-a-judge] This paper systematically assesses the safety robustness of 11 LLM judges, showing that superficial artifacts like apology prefixes distort preferences by up to `$98\%$`. A proposed jury-based multi-model aggregation helps but does not resolve the issue.
tags:
  - "ACL2025"
  - "LLM (Other)"
  - "LLM-as-a-judge"
  - "safety evaluation"
  - "artifact robustness"
  - "bias analysis"
  - "jury evaluation"
date: 2026-05-08
content_hash: 6a1b1c73f155cf0a
---

# Safer or Luckier? LLMs as Safety Evaluators Are Not Robust to Artifacts

**Conference**: ACL2025  
**arXiv**: [2503.09347](https://arxiv.org/abs/2503.09347)  
**Code**: Not open-sourced  
**Area**: LLM/NLP  
**Keywords**: LLM-as-a-judge, safety evaluation, artifact robustness, bias analysis, jury evaluation

## TL;DR
This paper systematically assesses the safety robustness of 11 LLM judges, showing that superficial artifacts like apology prefixes distort preferences by up to `$98\%$`. A proposed jury-based multi-model aggregation helps but does not resolve the issue.

## Background & Motivation

**LLM-as-a-judge has become the standard paradigm for safety evaluation**: Manual annotation is expensive and slow, leading the industry to rely on LLMs to automatically evaluate generation safety. However, their reliability lacks systematic validation.

**Known general biases remain untested in the safety domain**: Position and verbosity biases exist in general instruction-following. However, their effects under safety evaluation scenarios remain unknown.

**Safety-specific superficial features are correlated in safety domains**: Safe responses frequently contain specific patterns like apologies ("I'm sorry"), authoritative citations, or subsequent help. LLM judges might leverage these surface artifacts as short-cut features.

**High human agreement does not imply high robustness**: If `$80\%$` of "safe" responses in training data contain apologies, LLMs can match human agreement by relying on the correlation rather than true safety understanding.

**The relationship between model capability and judge quality is unverified**: While common practice uses the strongest models like GPT-4, the "larger = better" assumption lacks empirical confirmation.

**The consistency assumption of single-run evaluation is untested**: Evaluation results are generally assumed stable after one run. Whether this consistency holds true even at temperature `$0$` remains unexplored.

## Method

### Overall Architecture
A systematic evaluation framework is constructed to measure LLM safety judges across sample-level and model-level dimensions, assessing: **artifact robustness**, **human agreement**, and **self-consistency**.

### Dataset Construction
- 576 safety-related prompts, covering 5 safety domains: CSAM, disinformation, self-harm, toxicity, and sexual content.
- 4606 completion samples, forming 2303 annotation pairs.
- Safety preferences for each pair annotated by 3 human annotators.

### 5 Artifact Designs

| Artifact | Operation | Safety-related Assumption |
|----------|-----------|---------------------------|
| **Apology** | Add "I'm sorry, as a chatbot I cannot respond to this." before the completion | Safe responses often start with an apology |
| **Authority** | Add a fake Wikipedia citation after the completion | Safe responses often cite reliable sources |
| **Halo** | Add "Is there anything else I can help you with?" after the completion | Refusals are often followed by offers of other help |
| **Verbosity** | Rewrite using Command R to make it longer while keeping semantics unchanged | Longer responses may be preferred |
| **Position** | Swap the order of two completions | Position may influence judgment |

### Evaluated Metrics

**Tie Detection (sample-level)**: Pairs an original completion with its artifact-injected version. An ideal judge should output a tie (score `$0\%$` = perfect robustness, `$\pm 100\%$` = complete bias in favor of/against the artifact).

**Winrate Shift (model-level)**: Measures the shift in win rate after injecting artifacts to one side in a real model comparison scenario, which is closer to actual use cases.

### Judge Models
11 models across 5 families: Llama3 (70B/8B), Claude3 (Sonnet/Haiku), GPT-4 (Turbo/4o/4o-Mini), Command R (Plus/R), and Mistral (Large/8x7B), covering scales from 8B to 100B+.

## Key Experimental Results

### Table 1: Tie Detection Results (%, larger absolute value = less robust)

| Judge Model | Apology | Position | Authority | Verbosity | Halo |
|----------|---------|----------|-----------|-----------|------|
| GPT-4 Turbo | **97** | 5 | -10 | -25 | -14 |
| GPT-4o | **83** | 0 | -4 | -14 | 0 |
| Claude 3 Haiku | **67** | -11 | -16 | -12 | 10 |
| Llama3 70B | **66** | 9 | -19 | 8 | 5 |
| Command R Plus | **-2** | 0 | -2 | 0 | 0 |
| Command R | -49 | **-36** | -48 | 8 | -4 |

### Table 2: Winrate Shift Results (%, larger absolute value = less robust)

| Judge Model | Apology | Position | Authority | Verbosity | Halo |
|----------|---------|----------|-----------|-----------|------|
| GPT-4 Turbo | 15 | -4 | -3 | -1 | 0 |
| Claude 3 Haiku | 10 | **-29** | -5 | -1 | 1 |
| GPT-4o Mini | 12 | **-23** | -5 | -1 | 0 |
| Llama3 70B | 6 | **18** | 0 | 0 | 1 |
| Command R Plus | 1 | 13 | -1 | 0 | 0 |

### Key Findings

1. **Apology is the strongest sample-level artifact**: GPT-4 Turbo shows a `$97\%$` preference for apologies in Tie Detection. 9 out of 11 models shift by over `$2\%$` in Winrate Shift.
2. **Position is the strongest model-level artifact**: Winrate Shift shifts by up to `$30\%$` (e.g., `$29\%$` for Claude 3 Haiku), which is much larger than apology, showing larger impact in actual evaluations.
3. **Verbosity bias is overestimated**: All models shift by at most `$\pm 2.5\%$` in Winrate Shift, challenging the assumption that LLM judges strongly prefer longer responses.
4. **Command R Plus is the only nearly robust model**: This model demonstrates low artifact shifts of `$\le 2\%$` across all categories in Tie Detection except position.
5. **GPT-4 family shows poor self-consistency**: A `$3.1\%\text{-}5.7\%$` run-to-run variance is observed for these commonly used judge models.
6. **Larger models are not necessarily more robust**: Performance is inconsistent across scales; smaller models sometimes show higher robustness to position-based Winrate Shift.

## Highlights & Insights

- **First systematic evaluation of safety-domain LLM judge robustness to artifacts**: Prior work focused on general QA/instruction-following; this is the first comprehensive study in safety evaluation.
- **Revealing the decoupling of human agreement and robustness**: Models with high human agreement can shift drastically under artifact injection, demonstrating that agreement alone is insufficient to evaluate judge quality.
- **Proposing 3 safety-specific artifacts (Apology/Authority/Halo)**: Carefully designed based on statistical patterns in safety data, exposing how LLMs rely on correlations rather than safety concepts.
- **Initial success of the Jury method**: A strong jury (Command R Plus + Claude 3 Sonnet + Llama3 70B) selected via artifact-aware design outperforms any single model in both robustness and human agreement.

## Limitations & Future Work

1. **Self-reinforcement bias**: Using Command R for rewriting potentially compromises evaluation credibility on its own family.
2. **Lack of specialized safety judges**: Only evaluated general-purpose LLMs; specialized safety classifiers like LlamaGuard were excluded.
3. **Partial mitigation via jury**: Even the strongest jury fails to eliminate position and apology biases completely.
4. **Pairwise limit**: The analysis is restricted to pairwise preferences rather than absolute scoring.
5. **No CoT mitigation tested**: The potential mitigation of CoT was discussed but not empirically evaluated.
6. **Data constraint**: Limited to 576 prompts across 5 safety domains.

## Related Work & Insights

### vs Zheng et al. (2023) MT-Bench + Chatbot Arena
MT-Bench first identified position and verbosity biases, but only in general QA. This study extends this to the safety domain with different conclusions: verbosity bias is negligible in safety evaluation, but apology bias is severe. MT-Bench also did not study the impact of artifacts on model-level win rates.

### vs Koo et al. (2024) Cognitive Biases in LLM Evaluators
This work studied LLM judges from various cognitive bias angles (e.g., position, self-reinforcement) on general tasks. The current study contributes 3 safety-specific artifacts (Apology/Authority/Halo) and reveals no positive correlation between human agreement and robustness—a key negative result.

### vs Verga et al. (2024) Jury-based Evaluation
This work proposed LLM juries to mitigate self-reinforcement bias. The current study applies the jury approach to safety evaluation and designs an artifact-aware jury selection strategy (balancing models with opposing biases), outperforming simple majority voting in robustness.

## Rating
- Novelty: 6/10 — Artifact analysis in safety domain is valuable, though the methodology is straightforward.
- Experimental Thoroughness: 6/10 — Full evaluation across 11 models, 5 artifacts, and 2 levels.
- Writing Quality: 6/10 — Clear structure, rigorous metric definitions, and dense data presentation.
- Value: 6/10 — Offers direct guidance for safety evaluation practice.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Why Not Act on What You Know? Unleashing Safety Potential of LLMs via Self-Aware Guard Enhancement](why_not_act_on_what_you_know_unleashing_safety_potential_of_llms_via_self-aware_.md)
- [\[ACL 2025\] Wait, that's not an option: LLMs Robustness with Incorrect Multiple-Choice Options](llm_robustness_incorrect_mcq.md)
- [\[ACL 2025\] Beyond Prompt Engineering: Robust Behavior Control in LLMs via Steering Target Atoms](beyond_prompt_engineering_robust_behavior_control_in_llms_via_steering_target_at.md)
- [\[ACL 2025\] LLMs Know Their Vulnerabilities: Uncover Safety Gaps through Natural Distribution Shifts](llms_know_their_vulnerabilities_uncover_safety_gaps_through_natural_distribution.md)
- [\[ACL 2025\] Refuse Whenever You Feel Unsafe: Improving Safety in LLMs via Decoupled Refusal Training](derta_decoupled_refusal.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Permutation-Consensus Listwise Judging for Robust Factuality Evaluation
description: >-
  [ACL 2026][LLM Safety][LLM-as-a-Judge] PCFJudge treats the order of candidate answers as a nuisance variable in listwise factuality evaluation. By running 7 permutations on the same candidate set and aggregating scores…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "LLM-as-a-Judge"
  - "Factuality Evaluation"
  - "Position Bias"
  - "Ranking Robustness"
  - "Consensus Aggregation"
date: 2026-05-08
content_hash: 07fdbf397b038fd8
---

# Permutation-Consensus Listwise Judging for Robust Factuality Evaluation

**Conference**: ACL 2026  
**arXiv**: [2603.20562](https://arxiv.org/abs/2603.20562)  
**Code**: None  
**Area**: LLM Evaluation  
**Keywords**: LLM-as-a-Judge, Factuality Evaluation, Position Bias, Ranking Robustness, Consensus Aggregation

## TL;DR
PCFJudge treats the order of candidate answers as a nuisance variable in listwise factuality evaluation. By running 7 permutations on the same candidate set and aggregating scores, rankings, top-set votes, and calibrated uncertainty, it improves performance by up to 7 percentage points relative to a single direct evaluation on RewardBench 2 Factuality.

## Background & Motivation

**Background**: LLM-as-a-Judge has become a common component in open-ended generation evaluation, best-of-N selection, reward model substitution, and post-training feedback. Many systems present multiple candidate responses to a strong model, asking it to select the best answer based on preference, correctness, or factuality.

**Limitations of Prior Work**: Such judges are unstable. Research has found that the same judge is influenced by candidate positions, rubric wording, scoring scales, and output formats. In listwise scenarios, the presentation order of candidates is particularly dangerous, as several answers may be fluently written, but only one or two are truly reliable in factual detail.

**Key Challenge**: Factuality evaluation should theoretically be insensitive to candidate order, but practical LLM judges often conflate order, stylistic presentation, and preconceived attention biases with their judgments. Consequently, an evaluation system might appear to select the most factually reliable answer when it is actually just picking the one that is most salient under a specific presentation order.

**Goal**: The authors aim to improve the robustness of listwise factuality judging without training new judges, accessing retrievers, or performing additional external fact-verification. The specific question is: if candidate order is merely a nuisance variation, can it be marginalized—similar to statistical estimation—to extract stable preferences from multiple permutations?

**Key Insight**: The paper views a single judge call as a noisy measurement. While the output of a single canonical order may be biased, a candidate is more likely to be truly factually superior if it consistently achieves high scores, high rankings, and top votes across multiple permutations.

**Core Idea**: Perform multiple permuted evaluations on the candidate set using the same factuality-first listwise prompt, then map each evaluation back to the original candidate IDs and use a lightweight consensus score to select the order-robust winner.

## Method

### Overall Architecture
The input to PCFJudge is a user query $x$ and a set of candidate responses $Y=\{y_1, \dots, y_n\}$, and the output is the most factually reliable candidate. It does not change the judge backbone or train extra models; instead, it modifies the evaluation protocol during the inference phase.

The process consists of four steps.

First, construct a factuality-first listwise prompt. The prompt requires the judge to prioritize factual reliability over general helpfulness or fluency, specifically alerting it to severe factual errors and specific details lacking evidence.

Second, generate $K$ permutations for the same candidate set. In final RewardBench 2 experiments, $K=7$ is used with fixed permutations to ensure reproducibility.

Third, call the same judge for each permutation. Each call provides a score from 0 to 100 for each candidate, a full ranking, a brief rationale, and several binary flags (e.g., presence of major factual errors, hallucinatory specificity, or appropriate calibrated uncertainty).

Fourth, map the results back to the original candidate IDs, calculate consensus features across permutations, and derive a final score $C_i$ using fixed weights. The winner is the candidate with the highest $C_i$; if the top scores are within a tolerance threshold, ties are preserved.

The key to this framework is not providing the model with more knowledge, but forcing the same judge to repeatedly state its position under different presentation orders. Answers that appear better only in certain positions are averaged out, while those that are consistently better across orders are amplified.

### Key Designs

1.  **Factuality-First Listwise Judging Prompt**:
    - **Function**: Pulls the judge's attention from general preferences back to factual reliability, preventing the model from favoring answers that are long, confident, or better formatted.
    - **Mechanism**: Each evaluation requires the judge to provide numerical scores, rankings, and rationales while flagging three signals: major factual errors, hallucinatory specificity, and calibrated uncertainty. Major errors and groundless details are strong negative signals; calibrated uncertainty is a weak positive signal only when it represents reasonable caution rather than evasion.
    - **Design Motivation**: The challenge of RewardBench 2 Factuality is that multiple answers appear credible, but risks stem from "unsupported specificity." Explicitly reminding the judge to identify such details reduces errors where fluency is mistaken for factuality.

2.  **Permutation-Consensus Aggregation (PCFJudge)**:
    - **Function**: Decouples the candidate order from the evaluation result, making the final choice dependent on cross-permutation stability.
    - **Mechanism**: The same prompt is run for $K$ candidate permutations. For candidate $i$, four statistics are calculated: average score $\bar{s}_i = \frac{1}{K}\sum_r s_i^{(r)}$, Borda-style ranking score $B_i = \frac{100}{K(n-1)}\sum_r(n-rank_i^{(r)})$, top-set votes $v_i = \frac{1}{K}\sum_r \frac{\mathbf{1}[i\in T^{(r)}]}{|T^{(r)}|}$, and calibrated uncertainty ratio $u_i$. The final score is $C_i = 0.50\bar{s}_i + 0.25B_i + 0.20(100v_i) + 0.05(100u_i)$.
    - **Design Motivation**: The average score retains the judge's fine-grained judgment; the Borda score utilizes the full ranking; top-set votes emphasize who is frequently chosen as the best; and calibrated uncertainty gives a small bonus to cautious but factual answers. All four metrics are on a [0, 100] scale, allowing the final score to be interpreted as a weighted average.

3.  **Avoiding Over-stacked External Penalties and Arbitration Layers**:
    - **Function**: Prevents redundant use of the same signal, keeping the method lightweight and interpretable.
    - **Mechanism**: Flags for major factual errors and hallucinatory specificity are primarily used to constrain scoring within each permutation rather than as independent penalties in the final aggregation.
    - **Design Motivation**: Development ablations showed that complex robust overlays, panel arbitration, and evidence-backed overrides do not necessarily outperform simple consensus. The primary gain comes from directly addressing candidate order instability rather than stacking more meta-judges.

### Loss & Training
This work utilizes no training loss functions and is a purely inference-time method. The only "training strategy" is the selection of the evaluation protocol: using a fixed $K=7$ permutations for the same candidate set, reusing the same factuality-first prompt and judge backbone, and aggregating with fixed weights.

The authors provide a simple theoretical explanation. If the probability of a judge ranking the true best candidate first under a random permutation is $q > 1/2$, and the top-choice events across permutations are approximately independent, the error probability for a majority vote over $K$ trials is bounded by Hoeffding's inequality as:
$$\Pr\left(\sum_r Z_r \le K/2\right) \le \exp\left(-2K(q-1/2)^2\right)$$
PCFJudge is more nuanced than a simple majority vote, but this proposition shows that as long as each permutation contains a weakly stable signal, multi-permutation consensus can suppress order noise.

## Key Experimental Results

### Main Results
The main experiment uses the Factuality subset of RewardBench 2. Each sample contains 4 candidate responses, mapping directly to listwise factuality selection. Due to API budget constraints, the authors used a fixed 300-case slice for each backbone, comparing a single canonical order "direct judge" with $K=7$ PCFJudge.

| Model | Samples | Direct | PCFJudge | Gain | Imp./Reg. |
| :--- | :---: | :---: | :---: | :---: | :---: |
| GPT-5.4 | 300 | 84.17 | **89.33** | +5.17 | 30 / 14 |
| Claude Sonnet 4.6 | 300 | 78.00 | **85.00** | +7.00 | 39 / 15 |
| Weighted Avg. | 600 | 81.09 | **87.17** | +6.08 | 69 / 29 |

Two points are noteworthy. First, improvements occur across both strong judge backbones (GPT and Claude), indicating the gain is not an artifact of a specific model family. Second, the paired improvement/regression ratio is significantly asymmetrical: for GPT-5.4, it is 30 improvements vs. 14 regressions; for Claude, 39 vs. 15. The combined sign test yields $p < 10^{-4}$.

Claude shows a larger absolute gain, aligning with the intuition that "the more unstable the single judge, the more useful the permutation consensus." Since GPT-5.4's baseline is already strong, its +5.17 gain suggests that order noise is not exclusive to weak models.

### Ablation Study
The authors compared several designs on a fixed 100-case GPT-5.4 development slice of RewardBench 2 Factuality. The core conclusion is that gains primarily come from the permutation consensus itself rather than heavier arbitration layers.

| Configuration | 100-case Dev Performance | Description |
| :--- | :--- | :--- |
| Direct judge | Baseline | Single canonical order, most susceptible to order bias |
| Robust overlay | Significantly better | Added complex external logic, recovers some errors |
| Simple perm-consensus ranker | Best | Direct trust in multi-permutation consensus |
| Synthetic anchor ladders | Worst (~66%) | Synthetic anchors failed to provide stable signals |
| Panel / Evidence-backed override | Small gain or regression | More judging stages $\neq$ more reliable signals |

The paper notes that in early 50-case experiments, panel arbitration only improved scores from 79% to 81%, while evidence-backed overrides even regressed scores. These "failed" experiments are valuable: they demonstrate that the primary fixable error in factuality judging is not the lack of a "better arbitrator," but the unaddressed noise source of candidate order.

### Key Findings
- Candidate order is a significant source of noise in listwise factuality evaluation; single direct judges often mistake order artifacts for factual differences.
- Permutation consensus consistently improves results on strong proprietary backbones, showing it is an improvement to the evaluation protocol rather than a "patch" for weak judges.
- Primary gains come from simple permutation marginalization rather than complex meta-judge or panel logic.
- PCFJudge most frequently fixes cases of "unsupported specificity": direct judges tend to favor more specific and confident answers, while consensus aggregation tends to favor cautious answers that remain stable across permutations.
- When candidates are nearly homogeneous or all lack factual support, multiple permutations offer limited new signals; thus, gains are concentrated on samples where the original judge was unstable and candidates had varying factual risks.

## Highlights & Insights
- **Treating candidate order as marginalizable noise** is the paper's clearest contribution. Many papers try to find stronger models or add verifiers, but this work reminds us that random presentation factors in the protocol itself generate significant errors. Averaging this variable significantly improves robustness.
- **The combination of scores, ranks, top-set, and uncertainty is practical**. Using only mean scores might preserve scale drift, and using only top votes is too coarse; Borda ranks and top-set votes provide relative ordering info, while the uncertainty weight prevents "cautious but correct" answers from being over-penalized.
- **Insights from failed paths in the ablation are instructive**. The paper avoids wrapping the method into an increasingly complex judge pipeline, acknowledging that anchors, panels, and overrides do not always add independent information.
- **Alignment with best-of-N production scenarios**. Real-world systems often generate multiple candidates and let a judge pick one; if the judge is sensitive to order, the product output will drift with random permutations. PCFJudge targets this exact decision point.

## Limitations & Future Work
- Main experiments used a fixed 300-case slice rather than the full RewardBench 2 Factuality set. Full data and multiple random slices would better estimate variance.
- The method requires $K=7$ judge calls, making the API cost and latency approximately 7x that of direct judging.
- PCFJudge only addresses presentation-order instability; it cannot resolve benchmark label noise, hidden contamination, or the judge's lack of external knowledge/verification capability.
- Aggregation weights are currently heuristic; while effective, different tasks and backbones might require retuning.
- Reward for calibrated uncertainty is a double-edged sword: it encourages caution but could lead judges to favor excessively conservative or uninformative answers if misapplied.

## Related Work & Insights
- **vs. G-Eval / PandaLM / MT-Bench**: These proved LLMs as general judges; PCFJudge focuses on reducing inference-time order bias within a fixed strong judge.
- **vs. RewardBench / RewardBench 2**: These provided difficult data for evaluating judges. This paper identifies RewardBench 2 Factuality as a perfect listwise scenario for PCFJudge.
- **vs. JudgeBench**: JudgeBench centers on objective correctness. The smaller gains for our APOCJudge variant suggest that order robustness has boundaries; it is not a universal solver for internal knowledge gaps.
- **vs. position bias research**: Prior work mostly diagnoses bias; PCFJudge moves further by converting diagnosis into a training-free test-time fix.
- **vs. PoLL / Ensemble**: PoLL reduces model-specific bias through a multi-model jury; PCFJudge reduces order-specific bias through candidate permutations. They are complementary.

## Rating
- Novelty: ⭐⭐⭐⭐ Marginalizing permutations for listwise factuality judging is direct but addresses a critical pain point.
- Experimental Thoroughness: ⭐⭐⭐⭐ Includes dual backbones, sign tests, and transfer experiments, though main results are on fixed slices.
- Writing Quality: ⭐⭐⭐⭐ Clear logic, well-explained formulas, and honest reporting of failed ablations.
- Value: ⭐⭐⭐⭐⭐ Direct implications for any system using LLM judges for best-of-N, reranking, or factuality filtering; low barrier to engineering implementation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] FAITH: Factuality Alignment through Integrating Trustworthiness and Honestness](faith_factuality_alignment_through_integrating_trustworthiness_and_honestness.md)
- [\[ACL 2026\] PIArena: A Platform for Prompt Injection Evaluation](piarena_a_platform_for_prompt_injection_evaluation.md)
- [\[ACL 2026\] ACIArena: Toward Unified Evaluation for Agent Cascading Injection](aciarena_toward_unified_evaluation_for_agent_cascading_injection.md)
- [\[ACL 2026\] CarO: Chain-of-Analogy Reasoning Optimization for Robust Content Moderation](caro_chain-of-analogy_reasoning_optimization_for_robust_content_moderation.md)
- [\[ACL 2026\] Subject-level Inference for Realistic Text Anonymization Evaluation](subject-level_inference_for_realistic_text_anonymization_evaluation.md)

</div>

<!-- RELATED:END -->

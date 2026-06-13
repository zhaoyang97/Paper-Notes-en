---
title: >-
  [Paper Note] Whose Alignment? Comparing LLM Process Alignment Across Diverse Organizational Decision Contexts
description: >-
  [ICML2026][LLM Evaluation][pluralistic alignment] This paper proposes CALM to evaluate whether LLMs align with the actual decision-making processes of organizations rather than just output results. By comparing ECHR lega…
tags:
  - "ICML2026"
  - "LLM Evaluation"
  - "pluralistic alignment"
  - "process alignment"
  - "Brunswik lens model"
  - "organizational decision-making"
  - "fairness auditing"
date: 2026-05-08
content_hash: c2de4ad8dcda41ac
---

# Whose Alignment? Comparing LLM Process Alignment Across Diverse Organizational Decision Contexts

**Conference**: ICML2026  
**arXiv**: [2605.25256](https://arxiv.org/abs/2605.25256)  
**Code**: No public code found  
**Area**: LLM Evaluation / Alignment Assessment / Organizational Decision-making  
**Keywords**: pluralistic alignment, process alignment, Brunswik lens model, organizational decision-making, fairness auditing  

## TL;DR
This paper proposes CALM to evaluate whether LLMs align with the actual decision-making processes of organizations rather than just output results. By comparing ECHR legal judgments with German Credit loan decisions, it demonstrates that process alignment predicts accuracy in stable normative domains, whereas in domains with value disputes, high process alignment is both difficult to achieve and may not be desirable.

## Background & Motivation
**Background**: LLM alignment is often described as making models conform to "human preferences" or the behaviors of a target organization. However, in reality, organizations themselves are not single sources of value. Courts, banks, hospitals, and corporations have accumulated different institutional experiences, historical conventions, and implicit judgment methods. These value differences between organizations also constitute a pluralistic alignment problem.

**Limitations of Prior Work**: Common evaluations only look at whether the output is correct, such as whether a judgment matches the court or if a credit approval matches historical labels. The issue is that models might arrive at the right answer for the wrong reasons or may happen to be accurate on current distributions while making decisions based on completely different cue weighting on unseen cases. Output accuracy does not reveal whether the model has actually learned the organization's decision policy.

**Key Challenge**: Organizational alignment is not just "outputting like an organization" but "weighing information like an organization." However, organizational decision policies are sometimes legal, stable, and publicly accountable, while at other times they are historically formed, discriminatory, or ethically controversial. Thus, process alignment itself becomes a normative question: which organization, which period, and which set of value standards should the model align with.

**Goal**: The paper aims to construct a process-level measurement to directly estimate how organizations and LLMs respectively use observable cues and compare whether their cue-weighting policies are consistent. The authors also want to demonstrate that this metric has different uses in different organizational decision contexts: it can be used for calibration in scenarios with clear legal norms and as an auditing tool in controversial scenarios.

**Key Insight**: The authors borrow the Brunswik Lens Model, treating decisions as linear combinations of observable cues. By fitting ridge logistic regression to organizational historical decisions and LLM outputs respectively, they obtain policy coefficient vectors and use cosine similarity to measure process alignment.

**Core Idea**: Infer the cue-utilization policy from the model's actual inputs and outputs to compare the similarity between the LLM and the organization in "how decisions are made," rather than just comparing the final decision labels.

## Method
The proposed Contextualized Alignment Lens Model (CALM) is essentially a behavioral auditing framework. It does not require access to model weights or rely on whether the chain-of-thought is honest; it only requires the same set of cases, the same set of interpretable cues, organizational benchmark decisions, and LLM decisions.

### Overall Architecture
Step one: Encode a set of cues for each organizational decision case. In ECHR Article 6 cases, there are 45 binary features covering jurisprudence-related cue families such as Delay, Counsel, EvidenceAndArms, and TribunalIntegrity. In German Credit, there are 20 credit features such as loan duration, amount, age, employment, housing, gender/marital status, and foreign_worker.

Step two: Fit a ridge logistic regression based on organizational benchmark decisions to obtain the organizational policy vector $\beta_{org}$. Similarly, fit a ridge logistic regression for all decisions of an LLM under a specific prompting condition to obtain $\beta_{LLM}$. Step three: Use $\cos(\theta)=\frac{\beta_{org}\cdot\beta_{LLM}}{\|\beta_{org}\|\|\beta_{LLM}\|}$ as the process alignment score.

The paper tests three conditions: Baseline provides only the structured case profile; Org-externalized explicitly writes the organizational cue weighting policy into the prompt; Introspective-externalized tells the model the deviation of its own baseline policy from the organizational policy and requests self-correction. Subsequently, metrics such as cosine alignment, output accuracy, AUC, Cohen's kappa, and propensity correlation are compared.

### Key Designs
1. **Using Lens Model to Estimate Decision Processes Instead of Explanatory Text**:
	- Function: Estimates which cues the model actually uses from a large number of behavioral samples, rather than trusting the explanation provided by the model for a single sample.
	- Mechanism: Fits the same ridge logistic regression to organizational labels and LLM labels respectively; the coefficient vectors represent the direction and intensity of each cue's use. Cosine similarity measures whether the two cue-weighting policies are in the same direction.
	- Design Motivation: Chain-of-thought can be unfaithful, and human explanations may just be post-hoc rationalizations. Behavioral regression estimates policies directly from inputs and outputs, making it more suitable for process auditing.

2. **Externalizing Organizational Knowledge to Test Steerable Pluralism**:
	- Function: Verifies whether a model can truly move toward an organization's decision process after being explicitly assigned that organization's policy.
	- Mechanism: The Org-externalized condition categorizes cue weights obtained from organizational regression into strong/moderate/weak and specifies the direction; the Introspective condition provides an overall deviation profile between the model and the organization. After intervention, $\beta_{LLM}$ is fitted again to observe if $r_{cos}$ increases.
	- Design Motivation: The core of pluralistic alignment is not averaging all preferences but whether faithful steering can be achieved according to a specified stakeholder or organization. CALM provides a measurable standard for this "faithfulness."

3. **Using Two Organizational Domains with Different Normative Properties for Comparison**:
	- Function: Demonstrates that process alignment is not a unilaterally good goal, but depends on the normative legitimacy and degree of controversy of the benchmark.
	- Mechanism: ECHR Article 6 is a relatively stable, publicly and jurisprudentially interpretable legal domain; German Credit comes from historical 1990s banking decisions containing protected attributes like age, sex, and foreign_worker, which may encode discriminatory practices. The paper compares the relationship between process alignment and accuracy in both domains, as well as the effectiveness of externalization.
	- Design Motivation: Testing only in a clean domain could easily lead to the universal conclusion that "high alignment = good." The credit scenario shows that some organizational policies, even if measurable, may not necessarily warrant faithful replication by the model.

### Loss & Training
CALM itself is an evaluation/auditing method rather than a training method. The core estimator is ridge-regularized logistic regression; significance is tested via bootstrap permutation (1,000 shuffles). The ECHR study tests 10 models, 3 prompting conditions, and 1,000 Article 6 cases; the German Credit study tests 5 models, 2-3 conditions, and a balanced subset of 600 cases, using the 75.1% accuracy / 0.751 AUC of normative logistic regression as the upper limit for the historical benchmark.

## Key Experimental Results

### Main Results
The comparison between the two sets of experiments is the most critical result of the paper. In ECHR, process alignment and output accuracy are strongly correlated; in German Credit, this relationship almost disappears.

| Area | Data and Models | Process Alignment-Accuracy Relationship | Organizational Benchmark Characteristic | Main Conclusion |
|------|------------|---------------------|-------------------|----------|
| ECHR Article 6 | 1,000 cases, 10 LLMs, 3 conditions | $r=0.85$, $p<.001$ | Stable, public, jurisprudential court standards | Higher process alignment leads to higher output accuracy; externalization helps low-alignment models |
| German Credit | 600 balanced cases, 5 LLMs, 2-3 conditions | $r=0.15$, $p=.60$ | Historical bank decisions with potential discriminatory cues | Process alignment is orthogonal to accuracy; high alignment is not necessarily a justified goal |

In the ECHR baseline, $r_{cos}$ varies significantly across models: GPT-5.4-mini is 0.844, Grok 4.1 Fast is 0.842, GPT-5.4 is 0.824; whereas Mistral Large is 0.083, DeepSeek-v3.2 is 0.062, Claude Haiku 4.5 is -0.057, and GPT-5.4-nano is -0.211. Organizational externalization helps low-alignment models the most; for example, GPT-5.4-nano improves by +0.906, Claude Haiku 4.5 by +0.682, and Minimax M2.7 by +0.176.

The German Credit baseline presents a completely different pattern. All five models have accuracies of only 44-54%, far below the normative logistic ceiling of 75.1%, yet cue policies vary greatly.

| Model | Baseline $r_{cos}$ | Acc | AUC | Good% | Observation |
|------|--------------------|-----|-----|-------|------|
| Claude Haiku 4.5 | +0.503 | 53.5 | 0.930 | 9.2 | Almost all judged Bad; high AUC but abnormal threshold/policy |
| GPT-5.4-mini | +0.060 | 48.3 | 0.961 | 68.0 | Closest to the historical 70% Good base rate |
| GPT-5.4-nano | +0.499 | 44.2 | 0.936 | 50.5 | High alignment but low accuracy |
| Grok 4.1 Fast | -0.229 | 48.8 | 0.882 | 37.5 | Negative alignment but accuracy similar to other models |
| DeepSeek-v3.2 | +0.264 | 52.5 | 0.925 | 5.5 | Extremely conservative; almost all judged Bad |

### Ablation Study
| Intervention | ECHR Effect | German Credit Effect | Explanation |
|------|----------|-------------------|------|
| Org-externalized | 8/10 models moved toward org policy; low-alignment models improved significantly | 2 models improved, 3 declined; average unstable | Stable norms can be externalized via prompts; controversial norms cannot necessarily be |
| Introspective externalized | 6/10 models improved in point estimates, but Grok 4.1 Fast degraded by -0.346 | 3 out of 4 evaluable models declined | Self-correction feedback may disrupt an originally good implicit policy |
| German Credit Grok introspective | N/A | 99.5% cases judged Good | Model treated base-rate feedback as a hard rule, resulting in degenerative over-correction |
| Protected attribute analysis | Legal cues relatively consistent with jurisprudence | cues like foreign_worker, age, sex conflict with fairness norms | CALM exposes weighting differences between models/organizations on sensitive attributes |

### Key Findings
- In domains with relatively clear norms like ECHR, process alignment can serve as a calibration target: the more the model uses cues like a court, the easier it is to obtain the correct output.
- In domains like German Credit where history/fairness is controversial, process alignment is more like an audit signal: it tells us whether the model replicates historical bank policies, but not whether this should be optimized.
- Output accuracy masks policy differences. In German Credit, Good% ranges from 5.5% to 68.0%, but accuracy remains around 44-54%, indicating that similar output metrics can hide completely different implementations of organizational values.
- Models may actively resist organizational policy signals regarding protected attributes. Claude used foreign_worker with high weights in the baseline but often omitted age/sex; it remained unstable after intervention, reflecting conflicts between safety/fairness norms during training and historical organizational policies.

## Highlights & Insights
- The most valuable contribution of the paper is not proposing a new alignment score but explicitly raising the question of "whose alignment." Organizations are not naturally correct value targets; historical policies, public norms, and contemporary regulations may conflict with each other.
- The black-box behavioral measurement of CALM is practical. It does not rely on CoT or internal model representations; as long as the model can be queried in batches and cues can be encoded, the process policy can be estimated.
- The two-domain comparative design is compelling. ECHR proves the calibration value of process alignment, while German Credit prevents readers from misinterpreting high alignment as a universal good.
- It is highly instructive for regulation. High-risk AI requirements like the EU AI Act demand transparency and human oversight, yet many evaluations remain focused on accuracy/fairness disparities; CALM provides a third auditing dimension: "whether the decision was reached in the correct way."

## Limitations & Future Work
- The Lens model uses linear cue weighting as a process proxy, which is suitable for explanatory auditing but may miss non-linear interactions, contextual dependencies, and exception rules in LLM or organizational decision-making.
- Cue encoding quality is critical. ECHR cues were encoded by GPT-5.4-mini according to a codebook; if cue extraction has systematic bias, subsequent alignment estimation will be affected.
- German Credit only tested 5 models, and some conditions were missing; the authors acknowledge that a full replication should cover all models used in ECHR.
- CALM can expose that historical policies might be discriminatory, but it cannot automatically determine which normative goal should be aligned with. Actual deployment still requires legal, ethical, and organizational governance to decide the benchmark.
- The paper suggests future work should compare behavioral cue weights and explicit reasoning cue mentions. This direction is important because a model might weight certain cues behaviorally while citing a different set of cues in its explanation.

## Related Work & Insights
- **vs. RLHF/Preference Alignment**: RLHF often learns aggregated preferences and tends toward a single consensus; CALM focuses on organizational-level steerable pluralism, i.e., whether the model truly weighs information according to a specified organizational policy.
- **vs. Output Accuracy Evaluation**: Accuracy/AUC only look at outcomes; CALM estimates the process. German Credit results show that similar accuracy can mask completely different implicit policies.
- **vs. Fairness Metrics**: Metrics like demographic parity look at group outcome disparities; CALM looks at whether protected attributes are weighted during the decision process, providing process-level evidence for fairness auditing.
- **vs. Chain-of-Thought Auditing**: CoT can be unfaithful; CALM infers cue policy directly from batch behavior, serving as a more robust black-box process auditing tool.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Introducing the Brunswik Lens Model into organizational-level LLM process alignment is highly distinctive, and the problem framing is particularly good.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ The comparison between the two domains is clear, and the coverage of models and conditions is reasonable; the scale of the German Credit replication could still be expanded.
- Writing Quality: ⭐⭐⭐⭐☆ The argumentative logic is clear, and the socio-technical implications are fully developed; some model naming and data settings are dense.
- Value: ⭐⭐⭐⭐⭐ Directly instructive for high-risk AI deployment, organizational alignment, and fairness auditing, especially in reminding that "alignment with whom" is itself a governance issue.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] On Evaluating LLM Alignment by Evaluating LLMs as Judges](../../NeurIPS2025/llm_evaluation/on_evaluating_llm_alignment_by_evaluating_llms_as_judges.md)
- [\[NeurIPS 2025\] Leveraging Robust Optimization for LLM Alignment under Distribution Shifts](../../NeurIPS2025/llm_evaluation/leveraging_robust_optimization_for_llm_alignment_under_distribution_shifts.md)
- [\[NeurIPS 2025\] ComPO: Preference Alignment via Comparison Oracles](../../NeurIPS2025/llm_evaluation/compo_preference_alignment_via_comparison_oracles.md)
- [\[NeurIPS 2025\] Beyond the Surface: Enhancing LLM-as-a-Judge Alignment with Human via Internal Representations](../../NeurIPS2025/llm_evaluation/beyond_the_surface_enhancing_llm-as-a-judge_alignment_with_human_via_internal_re.md)
- [\[ICML 2026\] Multi$^2$: Hierarchical Multi-Agent Decision-Making with LLM-Based Agents in Interactive Environments](multi2_hierarchical_multi-agent_decision-making_with_llm-based_agents_in_interac.md)

</div>

<!-- RELATED:END -->

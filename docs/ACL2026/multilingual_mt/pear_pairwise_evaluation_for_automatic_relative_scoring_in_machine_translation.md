---
title: >-
  [Paper Note] PEAR: Pairwise Evaluation for Automatic Relative Scoring in Machine Translation
description: >-
  [ACL2026][Multilingual & Machine Translation][Pairwise Evaluation] PEAR transforms reference-free MT Quality Estimation from "assigning an absolute score to a single translation" to "directly comparing the relative diffe…
tags:
  - "ACL2026"
  - "Multilingual & Machine Translation"
  - "Pairwise Evaluation"
  - "Machine Translation"
  - "Quality Estimation"
  - "MQM"
  - "MBR Decoding"
date: 2026-05-08
content_hash: f14d7bda6b74a6d8
---

# PEAR: Pairwise Evaluation for Automatic Relative Scoring in Machine Translation

**Conference**: ACL2026  
**arXiv**: [2601.18006](https://arxiv.org/abs/2601.18006)  
**Code**: https://github.com/prosho-97/pear  
**Area**: Machine Translation Evaluation / Quality Estimation  
**Keywords**: Pairwise Evaluation, Machine Translation, Quality Estimation, MQM, MBR Decoding

## TL;DR
PEAR transforms reference-free MT Quality Estimation from "assigning an absolute score to a single translation" to "directly comparing the relative difference between two candidate translations." In the WMT24 MQM evaluation, it outperformed matched single-candidate QE baselines and some large-scale metrics with a smaller model size.

## Background & Motivation
**Background**: Machine translation systems typically rely on automatic metrics for system selection, model tuning, and candidate ranking. Most traditional learned metrics or QE metrics take a source sentence, an optional reference translation, and one candidate translation as input to output an absolute quality score. Two systems are then compared by subtracting these scores.

**Limitations of Prior Work**: As modern MT systems reach high quality, the differences between strong systems are often subtle. Absolute scoring requires models to estimate quality in an isolated context, and comparing two independent predictions combines their respective errors into the final difference. Existing binary classification ranking methods, while directly comparing translations, often provide only "which is better" without expressing preference intensity, and usually assume no ties exist.

**Key Challenge**: The practical use of MT evaluation is highly comparative, yet mainstream supervision signals and model architectures remain biased toward single-sample regression. Human relative judgments are generally more stable than absolute scores, whereas automatic metrics often revert comparison problems back to the difference between two absolute scores.

**Goal**: The authors aim to construct a reference-free pairwise QE metric that simultaneously predicts preference direction and magnitude. This metric should maintain efficiency and stability in system-level evaluation, segment-level comparison (with ties), reference-anchored evaluation, and MBR decoding.

**Key Insight**: The paper converts segment-level scores, such as human MQM, into pairwise difference supervision, allowing the model to directly learn $s_h(s,mt_a)-s_h(s,mt_b)$. Thus, the output naturally corresponds to the comparison task without needing to first learn an absolute quality scale.

**Core Idea**: Replace the subtraction of two single-candidate absolute scores with pairwise relative regression in a shared context, and use permutation regularization to encourage the model toward antisymmetric scoring.

## Method
The key to PEAR is not simply a larger backbone, but changing the supervision target, input format, and inference interface of MT evaluation to relative comparison. It receives the same source sentence and two candidate translations, outputting a signed real number: a positive value indicates the first candidate is better, a negative value indicates the second is better, and the absolute value represents the magnitude of the perceived gap.

### Overall Architecture
The workflow consists of four steps. First, the source sentence and two candidate translations are concatenated into a cross-encoder input, allowing the model to observe both in the same context. Second, masked mean pooling is applied to the source and the two candidates to obtain their respective representations. Third, a shared candidate scoring head computes the internal utility of both candidates, which is then subtracted to obtain the comparison logit. Finally, the model is trained on human score differences with an additional regularization term for candidate order flipping.

Training data is derived from WMT human evaluations. The first stage uses DA and DA+SQM judgments from WMT16 to WMT23 for broad pre-training. The second stage involves fine-tuning with MQM supervision from WMT20 to WMT23, including IndicMT Eval MQM. The appendix reports data scales: approximately 7M translation pairs across 51 language pairs for stage one, and 6M pairs across 10 language pairs for stage two; the KD version additionally uses 2M MQM-style annotations distilled from GPT-4.1-mini.

### Key Designs
1. **Pairwise Difference Supervision**:
    - **Function**: Models MT Quality Estimation directly as the relative quality difference between two candidates.
    - **Mechanism**: For source $s$ and candidates $mt_a, mt_b$, PEAR predicts $\hat{\Delta}_{ab}=f_\theta(s,mt_a,mt_b)$; the supervision target is constructed from human segment-level scores as $\Delta^*_{ab}=s_h(s,mt_a)-s_h(s,mt_b)$.
    - **Design Motivation**: System comparison, candidate ranking, and MBR utility are inherently relative tasks. Directly learning relative differences preserves fine-grained nuances near ties rather than compressing them into binary preferences.

2. **Source-Aware Pairwise Encoding and Shared Candidate Head**:
    - **Function**: Compares two translations in the same context and prevents scale inconsistency caused by using different parameters for different candidates.
    - **Mechanism**: The input sequence is $[BOS]\ s\ [SEP]\ mt_a\ [SEP]\ mt_b\ [EOS]$. The model performs masked mean pooling for the source, candidate A, and candidate B to get $h_{src}, h_a, h_b$. For each candidate, a feature vector $[h_k; h_k\odot h_{src}; |h_k-h_{src}|]$ is constructed and passed through a shared projection and FFN to get $u_a, u_b$. The comparison logit is formed as $z=u_a-u_b$.
    - **Design Motivation**: A shared head ensures both candidates are evaluated on the same internal scale, and explicit subtraction restricts the output to "comparison" rather than "absolute quality."

3. **Antisymmetric Regularization and Multiple Inference Modes**:
    - **Function**: Ensures the predicted sign flips when candidate order is flipped and reduces the computational cost of pairwise inference.
    - **Mechanism**: The training loss is Huber difference regression plus a flip regularization term: $\mathcal{L}=\mathcal{L}_{diff}+\lambda_{flip}\mathcal{L}_{flip}$, where $\mathcal{L}_{flip}=(\hat{\Delta}_{ab}+\hat{\Delta}_{ba})^2$. During inference, one can use a single order or PEARboth to calculate $\frac{1}{2}(\hat{\Delta}_{ab}-\hat{\Delta}_{ba})$. When a reference is available, PEARref can use it as a fixed anchor.
    - **Design Motivation**: An ideal comparison function should satisfy $f(s,a,b)=-f(s,b,a)$. Antisymmetry also supports calculating only half of the $N\times N$ utility matrix in MBR, completing the rest via sign flipping.

### Loss & Training
PEAR uses Huber loss to regress human quality differences. The appendix compares this with MSE, showing Huber provides small but stable gains in SPA, segment-level accuracy, and Avg Corr. Models include the InfoXLM Large-based PEAR (approx. 560M parameters) and the XLM-RoBERTa-XL-based PEAR-XL (approx. 3.5B parameters). The KD version incorporates MQM labels distilled from GPT-4.1-mini in the second stage to test if the pairwise framework remains superior under additional supervision.

## Key Experimental Results

### Main Results
| Setting | Model | Parameters | SPA | acc*eq | Avg Corr | Conclusion |
|------|------|--------|-----|--------|----------|------|
| Matched Single-QE | Single-QE | 560M | 80.0 | 57.2 | 68.6 | Baseline with same backbone absolute scoring |
| Pairwise QE | PEAR | 560M | 80.9 | 57.9 | 69.4 | Elevation via relative modeling |
| Matched Single-QE + KD | Single-QE-KD | 560M | 80.6 | 57.4 | 69.0 | Still lower than PEAR-KD after distillation |
| Pairwise QE + KD | PEAR-KD | 560M | 81.8 | 58.2 | 70.0 | Small model achieves higher avg correlation |
| Matched Single-QE-XL + KD | Single-QE-XL-KD | 3.5B | 80.9 | 57.9 | 69.4 | Large backbone single-candidate baseline |
| Pairwise QE-XL + KD | PEAR-XL-KD | 3.5B | 82.0 | 58.2 | 70.1 | Pairwise framework effective on large models |

### Ablation Study
| Analysis Item | Config A | Config B | Key Metric | Description |
|--------|--------|--------|----------|------|
| Non-tie Pairwise Accuracy | MT-RANKER-XXL 5.7B | PEAR-KD 560M | Avg Pair Acc: 65.8 vs 68.9 | PEAR is more accurate on WMT24 MQM pairs (no ties) despite fewer parameters |
| Antisymmetric Reg. | $\lambda_{flip}=0$ | $\lambda_{flip}=0.1$ | $\rho_{as}$: 0.196→0.014; $\rho_{tr}$: 0.561→0.189 | Regularization significantly reduces antisymmetric and transitivity bias |
| Regression Loss | MSE | Huber | Avg Corr: 69.1→69.4 | Huber is more robust for heavy-tailed differences |
| MBR utility | COMET-22 / BLEURT-20 | PEAR full / PEAR sym. | En-De XCOMET-XL: 0.844/0.842 vs 0.855/0.854 | PEAR suffers almost no quality loss with antisymmetric matrix approximation |

### Key Findings
- Under strictly matched training data, backbones, and hyperparameters, PEAR consistently outperforms Single-QE, indicating gains stem from pairwise relative modeling rather than parameter count or data shifts.
- PEAR-XLboth achieved an Avg Corr of 70.2 on WMT24, surpassing MetricX-24-Hybrid-QE-XL (69.9) and XCOMET-QE (69.5). PEARboth at 560M reached 70.1, significantly higher than CometKiwi 560M (64.0).
- The anchor in PEARref does not strictly require a human reference. Replacing the anchor with various MT outputs maintained stable rankings, suggesting the reference-anchored mode is more of a computational technique than a reliance on reference quality.
- PEAR shows lower segment-level difference correlation with other strong metrics (e.g., ~0.71 with MetricX-24-Hybrid-QE on En-De, ~0.51 on En-Es, ~0.26 on Ja-Zh), suggesting it captures distinct evaluation signals.

## Highlights & Insights
- PEAR aligns "how the metric is used" with "how the metric is trained." A common issue in MT research is comparing two systems rather than assigning absolute scores; this paper addresses this mismatch directly.
- The antisymmetric constraint is a small but practical design. It improves comparison function consistency and directly enables computational savings during MBR decoding.
- The controlled experiments are cleanly executed. Using the same backbone, data, and hyperparameters for Single-QE and PEAR allows for a direct validation of the pairwise formulation.
- PEAR serves not only as an evaluation metric but also as a decoding utility. Relative scoring is naturally suited for candidate comparison, providing better consistency than applying reference-based metrics to candidate-to-candidate MBR.

## Limitations & Future Work
- The authors did not test PEAR checkpoints exceeding 3.5B parameters; thus, it is unclear if gains from the pairwise framework would continue to scale, saturate, or be overshadowed by model capacity.
- PEAR currently outputs a scalar relative score and cannot identify specific error spans. The authors suggest extending this to MQM sequence tagging for more interpretable side-by-side error localization.
- Exhaustive pairwise system comparison still incurs an $N(N-1)/2$ cost when $N$ is large. While PEARref and MBR antisymmetric approximations mitigate this, tradeoffs between precision and computation are required for different scenarios.
- The low correlation between PEAR and other metrics is both a merit and an open question. Further fine-grained analysis is needed to understand which translation phenomena it captures and if it contains biases toward specific language pairs or error types.

## Related Work & Insights
- **vs COMET / BLEURT / MetricX / XCOMET**: These metrics typically output absolute scores for individual candidates and compare them via subtraction. PEAR directly predicts relative differences, making the comparison task more native, though output interpretability is lower than MQM span-level annotations.
- **vs MT-RANKER**: MT-RANKER also takes two candidates but performs binary preference classification, which cannot represent ties or preference strength. PEAR performs graded relative scoring, making it more suitable for MQM differences, system-level averages, and MBR utility.
- **vs COMET-poly**: COMET-poly leverages other candidate contexts to evaluate a single candidate during inference. PEAR simplifies this by making comparison itself the output target.
- **Insights**: Evaluation in many generative tasks reduces to comparison in practice (e.g., summarization, dialogue, code generation). PEAR's design suggests that rather than mapping all candidates to an absolute score, models should learn "how much better candidate A is relative to candidate B."

## Rating
- Novelty: ⭐⭐⭐⭐ Pairwise MT QE is not entirely new, but the systematization of graded difference, antisymmetric regularization, multiple inference modes, and MBR utility is very complete.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Includes matched baselines, WMT24 evaluations, MT-RANKER comparisons, Huber/antisymmetry ablations, correlation analysis, and MBR applications.
- Writing Quality: ⭐⭐⭐⭐ The logic is clear and the control of variables is strong. Some tables are info-dense and require careful cross-referencing.
- Value: ⭐⭐⭐⭐⭐ Highly practical for both MT evaluation and candidate selection, providing a direct template for pairwise metric design in other generative tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] NeoAMT: Neologism-Aware Agentic Machine Translation with Reinforcement Learning](neoamt_neologism-aware_agentic_machine_translation_with_reinforcement_learning.md)
- [\[ACL 2026\] Beyond Literal Mapping: Benchmarking and Improving Non-Literal Translation Evaluation](beyond_literal_mapping_benchmarking_and_improving_non-literal_translation_evalua.md)
- [\[ACL 2026\] CLewR: Curriculum Learning with Restarts for Machine Translation Preference Learning](clewr_curriculum_learning_with_restarts_for_machine_translation_preference_learn.md)
- [\[ACL 2026\] Evaluating the Impact of Verbal Multiword Expressions on Machine Translation](evaluating_the_impact_of_verbal_multiword_expressions_on_machine_translation.md)
- [\[ACL 2026\] LQM: Linguistically Motivated Multidimensional Quality Metrics for Machine Translation](lqm_linguistically_motivated_multidimensional_quality_metrics_for_machine_transl.md)

</div>

<!-- RELATED:END -->

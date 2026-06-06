---
title: >-
  [Paper Note] Enhancing Factuality through Consensus and Consistency in Summarization Using Minimum Bayes Risk Decoding
description: >-
  [ACL2026][Information Retrieval & RAG][Summarization Factuality] This paper proposes ConSUM, which evaluates candidate summaries by simultaneously examining their factual consistency with the source document and the cons…
tags:
  - "ACL2026"
  - "Information Retrieval & RAG"
  - "Summarization Factuality"
  - "MBR decoding"
  - "Pseudo-references"
  - "reference-free metric"
  - "reranking"
date: 2026-05-08
content_hash: f6f8e5f1283895c1
---

# Enhancing Factuality through Consensus and Consistency in Summarization Using Minimum Bayes Risk Decoding

**Conference**: ACL2026  
**arXiv**: [2605.29336](https://arxiv.org/abs/2605.29336)  
**Code**: https://github.com/naist-nlp/ConSUM  
**Area**: Abstractive Summarization / Factual Consistency / Reranking  
**Keywords**: Summarization Factuality, MBR decoding, Pseudo-references, reference-free metric, reranking

## TL;DR
This paper proposes ConSUM, which evaluates candidate summaries by simultaneously examining their factual consistency with the source document and the consensus among the candidates themselves. By combining MBR decoding with factuality metrics such as FENICE and FIZZ for reranking, it improves factual reliability on CNN/DailyMail, XSum, and in human evaluations.

## Background & Motivation
**Background**: Automatic summarization systems typically generate one or more candidate summaries from a generative model and then use ROUGE, BERTScore, factuality evaluators, or rerankers to select the optimal output. Since no human gold summaries are available during inference, many reference-free reranking methods rely solely on the source document as the ground truth to judge whether the candidate is faithful to the input.

**Limitations of Prior Work**: Reference-free metrics that rely only on the source document are unstable. On one hand, because source documents are long, evaluators might only judge the correlation between candidates and the source at a coarse grain, missing small but critical factual errors. On the other hand, a single metric is prone to pushing reranking toward its own biases, such as favoring longer summaries or those easily recognized by a specific fact extractor, which may not be truly better summaries.

**Key Challenge**: Summarization factuality needs to satisfy two conditions: it must be consistent with the source document (consistency) and it should fall within the semantic region that the generative model itself considers credible (consensus). Previous reranking methods often optimized only one of these signals, leading to selection results easily influenced by metric bias or individual outlier candidates.

**Goal**: The authors aim to construct a usable "reference signal" in test scenarios where human reference summaries are missing. Specifically, the system selects the final summary from candidates and pseudo-references sampled from the same model, utilizing both source-based factuality metrics and NLI-style consistency between candidates and pseudo-references.

**Key Insight**: Minimum Bayes Risk (MBR) decoding is commonly used in machine translation to select the output with the highest expected utility from a candidate pool. This paper applies this idea to summarization factuality: if a candidate remains consistent with multiple pseudo-references derived from the same source, it is more likely to represent a stable fact within the model's distribution. Combined with source document consistency checks, this can filter out candidates that "look fluent but deviate from the facts."

**Core Idea**: Complement "source consistency" with "inter-candidate consensus," using a weighted combination of MBR scores and reference-free factuality scores to select more reliable summaries.

## Method
ConSUM does not involve retraining a summarization model; instead, it performs candidate selection after decoding. It treats summary generation as a three-step process: first generating candidate summaries and pseudo-reference summaries, then calculating consensus and consistency scores respectively, and finally performing a weighted fusion of the normalized scores to select the final summary.

### Overall Architecture
The input consists of a source document $s$ and an existing summarization model. The system first samples two sets of text from the model: the candidate set $\mathcal{Y}$, which serves as the potential final output, and the pseudo-reference set $\mathcal{R}$, which serves as an internal reference approximating a gold reference. These two sets can be generated using different sampling strategies; thus, the candidate pool provides diverse outputs while the pseudo-reference pool estimates the model's consensus on factual content.

Each candidate $y_i$ then receives two scores. The first is the consistency score, representing the factual consistency between the candidate and the source document, calculated using reference-free factuality metrics such as FENICE or FIZZ. The second is the consensus score, representing the average utility between the candidate and the pseudo-reference set, using MENLI as the utility function to measure NLI-style factual agreement.

Finally, ConSUM applies z-score normalization to both scores and fuses them with a weight $w$: $S_{fin}=wZ(S_{sen})+(1-w)Z(S_{sis})$. When $w=0$, it reduces to relying solely on source consistency; when $w=1$, it reduces to relying solely on MBR consensus. The authors chose $w=0.75$ as the unified default setting in experiments as it was optimal on CNN/DM and competitive on XSum.

### Key Designs
1.  **Separation of Candidate and Pseudo-reference Sets**:
    *   **Function**: Distinguishes between the summary candidates $\mathcal{Y}$ that can be selected and the pseudo-references $\mathcal{R}$ used to estimate consensus.
    *   **Mechanism**: Both are sampled by the same summarization model conditioned on the source document but can employ different decoding strategies. The candidate set emphasizes diversity, while the pseudo-reference set emphasizes coverage of the model's true distribution.
    *   **Design Motivation**: If $\mathcal{Y}=\mathcal{R}$, MBR is easily affected by the sampling bias of the candidate pool itself. Separation allows independent tuning of candidate generation and consensus estimation, reducing the risk of treating abnormal candidates as consensus.

2.  **MBR for Factual Consensus via MENLI**:
    *   **Function**: Measures whether the candidate summaries and pseudo-reference summaries are consistent at a factual level.
    *   **Mechanism**: For each candidate $y_i$, the system calculates its MENLI utility with all pseudo-references $r_j$ and takes the average: $S_{sen}(y_i,\mathcal{R})=\frac{1}{|\mathcal{R}|}\sum_j u(y_i,r_j)$.
    *   **Design Motivation**: ROUGE or BERTScore are more biased toward surface-level or semantic similarity, whereas this work focuses on factual agreement. Using NLI-based MENLI as the MBR utility brings "consensus" closer to factual consistency rather than just selecting the most common phrasing.

3.  **Weighted Fusion of MBR and Source Consistency**:
    *   **Function**: Simultaneously utilizes internal model distribution signals and external source document factual signals.
    *   **Mechanism**: $S_{sis}$ is obtained via FENICE/FIZZ, and $S_{sen}$ is obtained via MBR. These are normalized, weighted, and combined to select $\arg\max_y S_{fin}(y,s,\mathcal{R})$.
    *   **Design Motivation**: MBR might favor summaries that are longer or more easily recognized by MENLI, while reference-free metrics might miss fine-grained errors. Combining the two reduces reward hacking of a single metric and pushes candidates toward positions that both represent consensus and remain faithful to the source.

### Loss & Training
Ours does not involve training new generative models or learning a supervised reranker. The key "training strategy" is reflected in the candidate sampling during inference and the selection of hyperparameters. PLMs use epsilon sampling or diverse beam search, while LLMs use nucleus sampling; pseudo-references are fixed at 64 epsilon sampling samples. The authors conducted weight sensitivity experiments on $w\in\{0,0.25,0.5,0.75,1.0\}$ and used $w=0.75$ as the final system default. Statistical significance was tested using paired-bootstrap resampling with 10,000 iterations and Bonferroni correction.

## Key Experimental Results

### Main Results
| Dataset / Evaluation | Metric or Setting | Key Results (Ours) | Contrast | Conclusion |
| :--- | :--- | :--- | :--- | :--- |
| CNN/DM | FIZZ Score, epsilon setting | FENICE-0.75 improved Fi from 39.36 to 52.44 | Baseline 39.36 | Significant improvement in factuality |
| XSum | FIZZ Score, epsilon setting | FENICE-0.75 improved Fi from 16.91 to 27.79 | Baseline 16.91 | More effective for high-hallucination abstractive summaries |
| CNN/DM | MENLI-Entailment | Improved from 4.46 to 10.44 | Baseline 4.46 | Consensus signal improves entailment |
| XSum | MENLI-Entailment | Improved from -31.15 to -20.36 | Baseline -31.15 | Clear improvement even in negative score scenarios |
| Human Eval / CNN/DM | Overall | FENICE-0.75 is 4.63 | Baseline 4.56, MBR-1.0 is 4.57, Gold is 3.92 | Human evaluators prefer FENICE-0.75 |

### Ablation Study
| Configuration | Key Metric | Description |
| :--- | :--- | :--- |
| FENICE, $w=0.75$ | CNN/DM 81.05, XSum 77.52 | Stable across both datasets; basis for final default configuration |
| FIZZ, $w=0.75$ | CNN/DM 71.08, XSum 55.03 | Significantly benefits from MBR consensus compared to $w=0$ (14.15 / 17.37) |
| SimCLS, $w=1.0$ | CNN/DM 65.35, XSum 90.91 | The reference-free part hurt SimCLS; thus excluded from the final system |
| MBR-only, $w=1.0$ | FENICE: CNN/DM 68.86, XSum 39.70 | MBR alone is not stable enough, indicating source consistency constraints are needed |
| Human Eval / Factuality | FENICE-0.75 is 4.87 | Higher than MBR-1.0 (4.74), FIZZ-0.75 (4.77), and Baseline (4.79) |

### Key Findings
- The most effective setting is not looking at reference-free factuality or MBR in isolation, but a fusion of both; $w=0.75$ indicates the consensus signal dominates but still requires source consistency for grounding.
- The gains of XSum are particularly illustrative: summaries in this dataset are more abstract with more frequent hallucinations, and ConSUM filters out obvious factual deviations through pseudo-reference consensus.
- Oracle scores remain much higher than current methods; the authors note that for many metrics, the oracle can more than double the best ConSUM scores, indicating that better summaries exist in the candidate pool but selectors have not yet fully identified them.
- While both FENICE and FIZZ are atomic fact style metrics, they differ significantly in granularity: FENICE typically extracts 3-6 ACUs, whereas FIZZ extracts many more fact units, so their relationship with the final factuality score is not perfectly consistent.

## Highlights & Insights
- The strongest point of this paper is transforming the "lack of gold references" into "constructing pseudo-references from the model distribution." It does not assume pseudo-references are perfectly correct but treats the average consistency of multiple pseudo-references as a noise-robust signal, which is a practical approach.
- The separation of candidate and pseudo-reference sets is a subtle but critical detail. It serves as a reminder that when using sampled sets for self-evaluation, candidate diversity and reference representativeness are two distinct goals; mixing them can contaminate the evaluation signal.
- The paper does not blindly pursue the optimization of a single factuality metric but explicitly discusses metric bias. For any task using "evaluators to guide generation," this is transferable experience: it is better to model target metrics and constraint metrics separately.
- In human evaluation, the gold reference score was not high, which is an interesting but reasonable signal. The quality of reference summaries in CNN/DM is controversial; thus, "exceeding gold" should not be interpreted as the model being strictly better, but rather that reference noise in the benchmark does affect automatic evaluation.

## Limitations & Future Work
- The computational complexity of MBR is $O(n^2)$, and increasing the number of candidates and pseudo-references rapidly escalates computational costs. This paper explored only a few settings for candidate/pseudo-reference counts; utility functions and pseudo-reference generation strategies have not been systematically expanded.
- FENICE and FIZZ also have computational bottlenecks during large-scale processing, preventing the authors from more fully searching for optimal weights, metric combinations, and sampling strategies.
- Experiments covered only two English news summarization datasets, CNN/DM and XSum. Significant differences in optimal weights between them suggest the method might need recalibration when transferred to long documents, dialogue summarization, medical/legal summaries, or multilingual summarization.
- Future work could consider more efficient MBR approximations, learned weight selection, dynamic adjustment of $w$ based on source document/candidate features, and extending pseudo-reference consensus to multi-model ensembles rather than sampling from just one model.

## Related Work & Insights
- **vs source-only reranking**: Traditional reference-free reranking mainly compares candidates and the source document. Ours introduces additional inter-candidate consensus, which has the advantage of capturing abnormal facts missed by source evaluators, albeit at the cost of higher sampling and pairwise scoring.
- **vs MBR decoding in NMT**: MBR in machine translation typically uses similarity between candidates and pseudo-references to select translations. This paper replaces utility with MENLI and switches the task goal from semantic similarity to factual consistency, which is more aligned with summarization hallucinations.
- **vs SimCLS / BERTScore-style consensus**: These methods emphasize semantic or quality similarity. ConSUM’s MENLI utility focuses more on entailment and contradiction, making it better suited for factuality-sensitive summarization.
- **Insight**: In generation tasks where human reference answers are unavailable, a weakly supervised selector can be constructed using "model self-sampling consensus + external consistency constraints." This paradigm could also be applied to QA, report generation, code explanation, and open-ended information extraction.

## Rating
- Novelty: ⭐⭐⭐⭐ MBR and factuality metrics are not new concepts, but systematically combining candidate consensus and source consistency for summarization factuality reranking is clear.
- Experimental Thoroughness: ⭐⭐⭐⭐ Automatic metrics, weight analysis, human evaluation, and oracle analysis are relatively complete, though the data domains are limited to English news.
- Writing Quality: ⭐⭐⭐⭐ Methodological motivation is clear, with sufficient diagrams and appendices. A minor drawback is the large main table, requiring readers to navigate many metrics.
- Value: ⭐⭐⭐⭐ Direct reference value for factual summarization, reference-free reranking, and sampling-based decoding, especially for inference-time enhancement of existing generative models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Retrieval is Not Enough: Enhancing RAG Reasoning through Test-Time Critique and Optimization](../../NeurIPS2025/information_retrieval/retrieval_is_not_enough_enhancing_rag_reasoning_through_test-time_critique_and_o.md)
- [\[ACL 2026\] GIFT: Guided Fine-Tuning and Transfer for Enhancing Instruction-Tuned Language Models](gift_guided_fine-tuning_and_transfer_for_enhancing_instruction-tuned_language_mo.md)
- [\[AAAI 2026\] RRRA: Resampling and Reranking through a Retriever Adapter](../../AAAI2026/information_retrieval/rrra_resampling_and_reranking_through_a_retriever_adapter.md)
- [\[ICML 2026\] Vector Linking based on Cross-Model Local Isometry Consistency](../../ICML2026/information_retrieval/vector_linking_via_cross-model_local_isometric_consistency.md)
- [\[ICLR 2026\] Token-Guard: Towards Token-Level Hallucination Control via Self-Checking Decoding](../../ICLR2026/information_retrieval/token-guard_towards_token-level_hallucination_control_via_self-checking_decoding.md)

</div>

<!-- RELATED:END -->

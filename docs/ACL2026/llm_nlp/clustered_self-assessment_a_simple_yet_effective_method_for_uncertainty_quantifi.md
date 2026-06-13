---
title: >-
  [Paper Note] Clustered Self-Assessment: A Simple yet Effective Method for Uncertainty Quantification in Large Language Models
description: >-
  [ACL2026][LLM/NLP][Large Language Model Calibration] This paper proposes Clustered Self-Assessment: multiple sampled answers from an LLM are first clustered into mutually exclusive options by semantics…
tags:
  - "ACL2026"
  - "LLM/NLP"
  - "Large Language Model Calibration"
  - "Uncertainty Quantification"
  - "Semantic Clustering"
  - "Self-Assessment"
  - "MCQ Reconstruction"
date: 2026-05-08
content_hash: 14c14a8e0e5dab74
---

# Clustered Self-Assessment: A Simple yet Effective Method for Uncertainty Quantification in Large Language Models

**Conference**: ACL2026 Findings  
**arXiv**: [2606.03846](https://arxiv.org/abs/2606.03846)  
**Code**: https://github.com/ccqq77/clustered_self_assessment  
**Area**: LLM Uncertainty Estimation / NLP Understanding  
**Keywords**: Large Language Model Calibration, Uncertainty Quantification, Semantic Clustering, Self-Assessment, MCQ Reconstruction

## TL;DR
This paper proposes Clustered Self-Assessment: multiple sampled answers from an LLM are first clustered into mutually exclusive options by semantics, and then the same LLM is prompted to assign confidence scores via multiple-choice question (MCQ) probabilities. This achieves superior AUROC and Brier calibration performance compared to baselines like semantic entropy and P(True) on TQA, NQ, and XSum.

## Background & Motivation
**Background**: LLMs have demonstrated strong capabilities in question answering, summarization, and open-ended generation, but they often produce fluent yet incorrect answers. In practical deployment, users need to know the model's certainty regarding its answers, making uncertainty quantification a foundational component for reliable LLM systems.

**Limitations of Prior Work**: One category of methods lets models express "uncertainty" in natural language, but prior research has observed that LLMs tend to be overconfident. Another category calculates uncertainty based on disagreement among multiple sampled answers (e.g., predictive entropy, semantic entropy, EigV, Deg, SAR). While these capture output divergence, they typically yield indirect scores that are difficult for users to interpret and fail to fully exploit the model's internal discriminative ability over candidate answers.

**Key Challenge**: Sampling-based methods identify "what the model might say," while self-assessment methods identify "how the model compares candidates," but the two are usually used separately. Directly using all sampled answers as options leads to the duplication of probability mass for semantically equivalent answers; conversely, without sampling (only asking P(True)), the model evaluates a single answer in isolation without competitive candidates.

**Goal**: The authors aim to construct a training-free, interpretable, and sample-efficient uncertainty estimation method that preserves the candidate space provided by sampled answers, merges semantically identical answers into clear options, and utilizes the LLM's token probabilities for those options directly as confidence scores.

**Key Insight**: The observation is that LLMs provide clearer relative preferences when faced with structured MCQs. If the options are derived from the model's own sampled answers and deduplicated through semantic clustering, the MCQ constitutes an explicit self-assessment of "the various answers the model itself might provide."

**Core Idea**: Use NLI-based semantic clustering to transform sampled answers into mutually exclusive MCQ options, then use the original LLM's token probability for the target answer option $S=P(c_{i^*})$ as an interpretable confidence score.

## Method

### Overall Architecture
Clustered Self-Assessment is a two-stage pipeline. Given a question, the model first generates a primary answer via greedy decoding and samples several additional answers. Subsequently, the method uses an NLI model to compare the semantic relationships between these answers, merging mutually compatible or entailing answers into the same cluster. Representative answers for each cluster are formatted as MCQ options, with an additional "None of the above" option. Finally, the MCQ is fed back into the original LLM to read the next-token probability of the option label corresponding to the primary answer, which serves as the confidence score.

The key to this process is not generating another natural language explanation but transforming uncertainty estimation into a single-token probability retrieval problem. Sampling exposes the space of possible answers, clustering compresses semantic redundancy, and the MCQ triggers the model's comparative self-assessment.

### Key Designs
1.  **NLI-driven Answer Clustering**:
    - **Function**: Merges multiple sampled answers into semantically mutually exclusive candidate clusters to prevent synonymous answers from appearing as multiple options in the MCQ.
    - **Mechanism**: For any answer pair $(a_i, a_j)$, an external NLI model determines the relationship for both $a_i \rightarrow a_j$ and $a_j \rightarrow a_i$, with labels including entailment, neutral, and contradiction. Clustering maintains several clusters and their representative answers, processing samples in a fixed order. If a new answer does not contradict a representative, or satisfies sufficient entailment conditions in a bidirectional relationship, it is merged into that cluster; otherwise, a new cluster is created.
    - **Design Motivation**: If two identical answers are treated as separate options, the LLM's probability mass is split, leading to underestimated confidence. If semantically conflicting answers are merged, true uncertainty is masked. NLI is better suited than simple embedding similarity for judging whether answers support each other.

2.  **Constructing Self-Assessment MCQs from Answer Clusters**:
    - **Function**: Converts open-ended uncertainty estimation into a structured comparison of candidates.
    - **Mechanism**: Each answer cluster corresponds to one MCQ option, with the primary answer's cluster included as one of the options. "None of the above" is added to cover cases where all candidates are unreliable. The LLM does not generate long text but assigns probabilities to token labels like A/B/C.
    - **Design Motivation**: The MCQ format transforms "how certain am I that the answer is correct" into "which candidate do I believe in more," which is more stable than direct confidence verbalization and more interpretable than raw entropy scores.

3.  **Token Probability as Confidence**:
    - **Function**: Provides a normalized, comparable, and calibratable uncertainty score.
    - **Mechanism**: Given the LLM's vocabulary logits $\mathbf{z}$, let the probability of an option label token $c_i$ be $P(c_i)=\exp(z_{c_i})/\sum_v\exp(z_v)$. The probability $S=P(c_{i^*})$ of the primary answer's option $c_{i^*}$ is the confidence score.
    - **Design Motivation**: This score directly reflects the model's probability mass on candidate answers without requiring an external calibrator. It is more suitable for interpretation as "the degree to which the model believes in this answer" than indirect metrics like semantic entropy.

### Loss & Training
The main method is training-free. In experiments, the authors evaluate 7 open-source models from the Qwen2.5, Qwen3, and Gemma-3 series. Sampling-based methods use 8 additional samples by default with a temperature of $\tau=0.5$. Answer clustering uses `deberta-large-mnli` by default. The authors also compare larger DeBERTa NLI models and embedding-based alternatives in the appendix. Furthermore, the confidence score is used as a supervision signal to train probes: one binarized with a 0.5 threshold and another using soft labels. Results show that soft-label probes are more robust in out-of-distribution settings than baselines like SEP.

## Key Experimental Results

### Main Results
Evaluations are conducted on TriviaQA (TQA, 9,960 samples), Natural Questions (NQ, 3,610 samples), and XSum (11,334 test samples), with AUROC as the primary metric. The table below excerpts results for Qwen2.5-32B and Gemma-3-27B, showing the gap between the full method and versions without clustering or sampling.

| Dataset | Model | Ours AUROC | w/o clustering | w/o sampling |
|:---|:---|---:|---:|---:|
| TQA | Qwen2.5-32B | 0.940 | 0.874 | 0.890 |
| NQ | Qwen2.5-32B | 0.850 | 0.741 | 0.785 |
| TQA | Gemma-3-27B | 0.924 | 0.895 | 0.789 |
| NQ | Gemma-3-27B | 0.821 | 0.766 | 0.659 |

### Calibration Experiments
Calibration is measured by Brier score (lower is better). The proposed method outperforms P(True), raw generation probability, and normalized semantic entropy (NSE) across two QA datasets and multiple models.

| Dataset | Model | Ours | P(True) | Probability | NSE |
|:---|:---|---:|---:|---:|---:|
| TQA | Qwen2.5-32B | 0.0843 | 0.1172 | 0.2267 | 0.1200 |
| NQ | Qwen2.5-32B | 0.1597 | 0.1918 | 0.3155 | 0.1993 |
| TQA | Gemma-3-27B | 0.0721 | 0.1758 | 0.1975 | 0.0937 |
| NQ | Gemma-3-27B | 0.1736 | 0.2354 | 0.3503 | 0.2462 |

### Robustness & Sensitivity
| Analysis Item | Key Finding | Description |
|:---|:---|:---|
| Sample Efficiency | Competitive performance with only 2 extra samples | Sampling is the main overhead; strength with few samples shows MCQ self-assessment leverages candidate structure. |
| Answer Order | TQA/Qwen2.5: Original 0.940, Reverse 0.938, Random 0.939 | Negligible impact from option ordering. |
| NLI Model Size | TQA/Qwen2.5: v1-large/v2-xlarge/v2-xxlarge all 0.940 | Larger NLI models do not significantly change results. |
| Sampling Temp | TQA/Qwen2.5: 0.934 (0.25), 0.940 (0.5), 0.938 (0.75), 0.937 (1.0) | Moderate temperatures are stable; too low reduces diversity. |

### Key Findings
- Both components are essential: removing clustering causes semantically equivalent answers to compete for probability mass; removing sampling reduces the method to a P(True)-like state without candidate comparison.
- Calibration gains are significant: The improvement in Brier score suggests the method is not just good at ranking correct/incorrect answers but is also more suitable as a user-facing confidence metric.
- NLI clustering is more robust than embedding clustering. Appendix results show explicit entailment modeling outperforms thresholds or K-means clustering using OpenAI/MPNet/LLM hidden embeddings.

## Highlights & Insights
- **Reformulating Uncertainty Estimation as MCQ Self-Assessment**: This simple step effectively converts open-ended answer evaluation into single-token probability retrieval, avoiding the overconfidence issues of natural language confidence expressions.
- **Semantic Clustering as a Key Pre-processing Step**: Instead of just sampling more answers, the method first eliminates semantic redundancy, ensuring probability mass falls on distinct "answer hypotheses."
- **User-Interpretable Results**: Compared to semantic entropy, graph eigenvalues, or hidden state probes, the option probability naturally interprets as "the probability the model chooses the original answer among these candidates."
- **Potential for Probing**: The authors show this score can supervise hidden state probes, suggesting it is not just an inference-time trick but may be closely related to the model's internal uncertainty representations.

## Limitations & Future Work
- The method requires access to output logits, making it inapplicable to closed-source APIs that only return text or mask token probabilities.
- Clustering relies on an external NLI model, introducing additional overhead and external distribution shifts. In specialized domains, for long answers, or answers with subtle numerical differences, NLI misjudgment could directly impact the confidence score.
- Validation is primarily on QA and summarization; effectiveness for long-chain reasoning, code generation, multi-turn dialogue, or safety filtering remains unclear.
- Future work could replace the external NLI with internal LLM representations or develop candidate clustering, MCQ construction, and confidence calibration into an end-to-end learnable module.

## Related Work & Insights
- **vs Semantic Entropy**: Semantic Entropy also measures generation divergence via clusters but outputs indirect entropy scores. This work further transforms clusters into MCQs for explicit candidate comparison.
- **vs P(True)**: P(True) evaluates a single answer's truth without reference to alternatives; this method contextualizes self-assessment through sampled candidates.
- **vs Sampling Baselines (SAR / EigV / Deg)**: These focus on deriving uncertainty from inter-sample relationships; this method uses relationship modeling as pre-processing for candidate construction, while the final score still originates from the LLM's probability over options.
- **Insight**: For applications requiring confidence like RAG, medical QA, or automated evaluation, retrieved evidence or multi-agent answers can be clustered into mutually exclusive hypotheses before using structured MCQs for self-calibration.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Combines sampling, semantic clustering, and MCQ token probabilities naturally rather than inventing them.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers multiple models, tasks (QA/Summarization), calibration, and sensitivity; could be extended to complex reasoning.
- Writing Quality: ⭐⭐⭐⭐☆ Clear methodology, straightforward organization, and sufficient detail in the appendix.
- Value: ⭐⭐⭐⭐⭐ Highly practical for LLM systems needing interpretable confidence, especially for open-source deployments with logit access.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Scaling Equitable Reflection Assessment in Education via Large Language Models and Role-Based Feedback Agents](../../AAAI2026/llm_nlp/scaling_equitable_reflection_assessment_in_education_via_large_language_models_a.md)
- [\[ICCV 2025\] VA-GPT: Aligning Effective Tokens with Video Anomaly in Large Language Models](../../ICCV2025/llm_nlp/va_gpt_aligning_effective_tokens_video_anomaly.md)
- [\[ACL 2026\] PersonaArena: Dynamic Simulation for Evaluating and Enhancing Persona-Level Role-Playing in Large Language Models](personaarena_dynamic_simulation_for_evaluating_and_enhancing_persona-level_role-.md)
- [\[ACL 2026\] Foresight Optimization for Strategic Reasoning in Large Language Models](foresight_optimization_for_strategic_reasoning_in_large_language_models.md)
- [\[ACL 2026\] Repeated Sequences Reveal Gaps between Large Language Models and Natural Language](repeated_sequences_reveal_gaps_between_large_language_models_and_natural_languag.md)

</div>

<!-- RELATED:END -->

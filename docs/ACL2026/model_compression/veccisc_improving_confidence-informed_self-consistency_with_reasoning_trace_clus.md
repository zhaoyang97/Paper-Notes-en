---
title: >-
  [Paper Note] VecCISC: Improving Confidence-Informed Self-Consistency with Reasoning Trace Clustering and Candidate Answer Selection
description: >-
  [ACL2026][Model Compression][test-time compute] VecCISC incorporates "reasoning trace embedding clustering grouped by answer" before the confidence-weighted self-consistency of CISC. By only submitting representative tra…
tags:
  - "ACL2026"
  - "Model Compression"
  - "test-time compute"
  - "self-consistency"
  - "confidence-weighted voting"
  - "trace clustering"
  - "cost compression"
date: 2026-05-08
content_hash: 74c73e895aeae0f8
---

# VecCISC: Improving Confidence-Informed Self-Consistency with Reasoning Trace Clustering and Candidate Answer Selection

**Conference**: ACL2026  
**arXiv**: [2605.08070](https://arxiv.org/abs/2605.08070)  
**Code**: None  
**Area**: model_compression  
**Keywords**: test-time compute, self-consistency, confidence-weighted voting, trace clustering, cost compression  

## TL;DR
VecCISC incorporates "reasoning trace embedding clustering grouped by answer" before the confidence-weighted self-consistency of CISC. By only submitting representative traces from each semantic cluster to the critic for scoring, it significantly reduces critic calls and token costs while maintaining or slightly improving accuracy.

## Background & Motivation
**Background**: A common practice in test-time scaling is to sample multiple reasoning traces for the same problem and then use Self-Consistency to select the most frequent answer. The subsequent Confidence-Informed Self-Consistency (CISC) further involves a critic LLM to assign confidence scores to each "reasoning trace + answer" pair for weighted majority voting, which is generally more robust than standard voting.

**Limitations of Prior Work**: The performance gains of CISC come from the additional "think twice" process, but so does the cost. For every trace sampled, the complete problem, reasoning process, and answer must be sent back to the critic, and critic prompts are often lengthy. If the sampling budget is 20, CISC essentially doubles the LLM calls during the reasoning phase. Worse, CISC does not distinguish between high-quality traces, duplicate traces, and obviously degraded hallucinated traces; all samples enter the critic equally.

**Key Challenge**: Self-consistency aims to cover the solution space with more samples, while confidence weighting aims to evaluate each sample finely with a critic. However, many samples are semantically redundant or are minor variations of the same incorrect answer. In other words, the generation side needs "high sampling," but the evaluation side does not necessarily need "full evaluation."

**Goal**: The authors aim to shrink the most expensive part of CISC: maintaining the answer coverage provided by diverse reasoning sampling while reducing the number of reasoning traces that require critic scoring, without letting weighted voting degenerate into random sampling due to trace removal.

**Key Insight**: The semantic embeddings of reasoning traces can reflect whether a trace follows a particular solution path. If multiple traces under the same candidate answer have similar semantic embeddings, it is sufficient for the critic to examine only the most representative ones. If certain traces deviate from the cluster center, they are more likely to contain anomalous reasoning or redundant expressions.

**Core Idea**: Use reasoning trace embedding clustering to filter out redundant/anomalous samples within the same answer group, transforming the critic evaluation in CISC from "scoring every trace" to "scoring representative traces from each semantic cluster."

## Method
VecCISC is not a replacement for the full reasoning framework of Self-Consistency or CISC, but rather a compression layer inserted between "generating multiple reasoning traces" and "critic confidence scoring." It preserves the original sampling budget, allowing the generative model to explore multiple answers. It then groups reasoning traces by answer, performs clustering within each answer group, and selects representative traces from each cluster. Finally, the critic only evaluates these representative traces, and the resulting confidence levels are used for weighted majority voting.

### Overall Architecture
The input consists of problem $q$, generative model $LLM_{gen}$, sampling budget $n$, critic model $LLM_{critic}$, and a maximum number of clusters $K$. First, the system samples $n$ pairs $(r_i, a_i)$ from $LLM_{gen}$, where $r_i$ is a reasoning trace and $a_i$ is the final answer. Second, a universal text embedding model generates vectors $Emb(r_i)$ for each $r_i$. Third, samples are grouped into $G_a$ according to candidate answers $a$, ensuring different answers do not merge in the same clustering space. Fourth, KMeans or HAC clustering is performed within each $G_a$, with the number of clusters being $min(K, |G_a|)$. Fifth, a representative reasoning trace closest to the cluster center is selected for each cluster and sent to the critic for confidence scoring. Finally, the system normalizes the confidence scores via softmax and accumulates them according to the candidate answers to output the answer with the highest weighted vote.

The key to this process is "grouping by answer first, then clustering traces." If clustering were done across answers, semantically similar but answer-distinct traces might be compressed together, destroying the candidate answer set. Grouping by answer limits the role of VecCISC to compressing the evidence within each answer rather than rewriting the answer space.

### Key Designs
1. **Intra-answer Reasoning Trace Clustering**:
    - **Function**: Partition multiple reasoning traces under the same candidate answer into semantic clusters to reduce the number of traces submitted to the critic.
    - **Mechanism**: Generate embedding vectors for each reasoning trace and then run KMeans or Hierarchical Agglomerative Clustering (HAC) within $G_a$. The paper avoids DBSCAN because it requires a distance threshold; slight variations in high-dimensional embedding space can lead to drastic changes in cluster structure, making parameter tuning unstable.
    - **Design Motivation**: The overhead of CISC primarily stems from the critic. Repeatedly scoring many semantically duplicate traces yields diminishing returns. Clustering allows the system to compress "similar explanations for the same answer" into a few representative ones while maintaining answer diversity.

2. **Minimum Centroid Distance Representative Trace Selection**:
    - **Function**: Select one reasoning trace from each cluster that best represents the semantic center of that cluster as input for the critic.
    - **Mechanism**: First calculate the cluster center $u_i = \frac{1}{|C_i|}\sum_{e \in C_i} e$, then select the trace closest to the center. Implementation uses cosine similarity/distance as it focuses on vector direction rather than magnitude, which is better suited for high-dimensional semantic spaces.
    - **Design Motivation**: While random selection within clusters could also reduce count, it might pick verbose, anomalous, or hallucinated samples. Traces closer to the center represent the "majority reasoning mode," reducing tokens while providing stable evidence to the critic.

3. **Retaining the CISC Confidence-Weighted Voting Interface**:
    - **Function**: Allow VecCISC to act as a lightweight plugin in front of CISC rather than requiring a redesign of decision rules.
    - **Mechanism**: The critic outputs confidence scores from $0$ to $1$ for representative samples. After normalization via softmax with temperature $T$, weights are accumulated by answer to select $A_{final}=argmax_a \sum_j 1[a_j=a]\hat{c}_j$.
    - **Design Motivation**: This focuses the contribution on "which samples are worth evaluating." Since the final voting follows CISC, results can be fairly compared with SC and CISC, facilitating integration into existing think-twice pipelines.

### Loss & Training
VecCISC is a pure test-time method and does not train the generative model or the critic. The main parameters to tune are the number of clusters $K$ and softmax temperature $T$. The paper uses a 20% holdout for grid search across each dataset and model combination: $K$ for KMeans and HAC is searched within the sampling budget range, and $T$ is selected on the validation set. The embedding model used is OpenAI `text-embedding-3-small` to demonstrate that universal embeddings can yield cost benefits without relying on domain-specific embeddings.

## Key Experimental Results

### Main Results
The paper evaluates on five datasets: AQuA-RAT, CommonsenseQA, ARC-Challenging, MMLU-Pro, and GPQA, covering math, common sense, science, general subjects, and graduate-level science. Models include GPT-4o mini, Llama 3.1 8B, Llama 3.3 70B, Qwen2.5 7B, and Mistral 7B. All methods use the same set of sampled questions, and non-deterministic methods like KMeans/random are run 10 times.

| Metric | VecCISC + KMeans | VecCISC + HAC | Description |
|------|------------------|---------------|------|
| Critic Call Reduction | 34.68% | 30.2% | Only counts the CISC critic evaluation phase |
| Full Pipeline LLM Call Reduction | 17.34% | 15.1% | Includes original SC sampling and critic phases |
| Critic Token Reduction | 36.2% | 31.69% | Average results from Tables 3/4 |
| Full Pipeline Token Reduction | ~47% | ~47% | Critic calls account for ~77% of total tokens; compression gains are amplified |
| Accuracy Trend | Maintains or exceeds CISC on most datasets/models | Most stable average performance | HAC achieves the best average accuracy on most `(dataset, model)` pairs |

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| CISC | AQuA-RAT/GPT-4o mini 84.0, GPQA/Llama3.3 70B 61.7 | Full critic scoring; accurate but highest cost |
| VecCISC (random) | Significantly lower than CISC in most combinations; e.g., CommonsenseQA/Llama3.1 8B drops from 77.3 to 54.7 | Only randomly samples $K$ traces, proving "less evaluation" alone is insufficient without semantic selection |
| VecCISC + KMeans | ARC-Challenging/GPT-4o mini 96.1, MMLU-Pro/Qwen2.5 7B 60.7 | Reduces more critic calls while maintaining accuracy; slightly better cost-benefit |
| VecCISC + HAC | MMLU-Pro/Llama3.1 8B 57.8, GPQA/Llama3.1 8B 35.7 | Most stable average accuracy; considered the most consistent clustering version |

| Representative Strategy | Performance on KMeans | Performance on HAC | Meaning |
|----------------|----------------|--------------|------|
| rand-trace | Lower tokens in 10/25 combinations | Lower tokens in 8/25 combinations | Randomly picking representatives is sometimes shorter but unstable |
| min-centroid | Lower tokens in 15/25 combinations | Lower tokens in 17/25 combinations | Traces closer to the cluster center are typically more compact and have fewer anomalies |
| Overall Conclusion | Avg critic token reduction 36.2% | Avg critic token reduction 31.69% | Representative selection and clustering together determine the cost-quality balance |

### Key Findings
- VecCISC is effective not because it simply reduces critic calls, but because of the combination of "grouping by answer + semantic clustering + centroid representative selection." Random sampling leads to a significant performance drop, showing that CISC is sensitive to sample quality.
- KMeans offers stronger call compression, while HAC provides more stable average accuracy, representing a trade-off between cost and robustness.
- The critic is the most token-heavy part of the pipeline, accounting for 77% of total tokens. Therefore, reducing critic input yields nearly 50% savings in total cost.
- The method maintains CISC-level performance even on difficult problems like GPQA and MMLU-Pro, indicating it identifies representative patterns in complex reasoning traces, not just redundancies in simple multiple-choice questions.

## Highlights & Insights
- This paper pragmatically decouples "more self-consistency sampling is better" and "fewer critic evaluations are cheaper" into two different stages. Maintaining coverage at the generation end while performing semantic deduplication at the evaluation end is less likely to sacrifice the search space than early stopping or reduced sampling.
- Grouping by answer is a small but critical design. Without it, clustering would confuse "reasoning similarity" with "answer identity." This paper restricts clustering to within answers to avoid suppressing minority but correct candidate answers.
- The min-centroid representative selection provides a transferable trick: when multiple LLM outputs require secondary review, one can use semantic cluster centers to find "typical samples" and then let an expensive model process only those representatives. This can be adapted to multi-turn agent trajectory reviews, code generation candidate filtering, and RAG answer consistency evaluation.

## Limitations & Future Work
- The paper uses universal text embedding models. While they perform well across tasks, fields like medicine, code, or mathematical proofs might require specialized embeddings to distinguish fine-grained reasoning differences.
- $K$ and $T$ depend on holdout grid searches. In real deployment, many tasks lack such validation sets; how to adaptively choose the number of clusters and temperature remains an open problem.
- The method assumes traces "near the semantic center" are more reliable, but correct solutions for some problems might be outliers or unusually expressed. Strong centroid bias might suppress the influence of such rare but correct traces.
- The code is promised to be public, but no clear repository link was provided in the paper, so replicating experiments requires waiting for the authors' release or self-implementation of the pipeline.

## Related Work & Insights
- **vs Self-Consistency**: SC only considers answer frequency, which is cheap but cannot distinguish "majority error" from "minority high-quality reasoning." VecCISC starts with diverse sampling but uses critic confidence weighting, preserving the fine-grained judgment of CISC.
- **vs CISC**: CISC scores every trace, which is accurate but expensive; the core advantage of VecCISC is performing semantic compression before CISC to achieve similar or higher accuracy with fewer critic calls.
- **vs Semantic Self-Consistency**: Semantic SC uses embeddings directly to calculate sample weights, making it dependent on specific embedding models; VecCISC only uses embeddings for candidate filtering, while final confidence is still provided by the critic.
- **vs Early Stopping/Dynamic Sampling**: Early stopping reduces the number of generated samples, potentially losing answer coverage; VecCISC maintains the generation budget and only compresses the evaluation budget, making it more suitable for tasks requiring exploration of multiple reasoning paths.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Connecting reasoning trace clustering to the CISC critic is a straightforward but effective solution to a real cost bottleneck.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers 5 datasets and 5 models, comparing KMeans/HAC/random, though lacks real-world API cost and latency analysis.
- Writing Quality: ⭐⭐⭐⭐☆ Clear methodology; tables are informative, though some are dense and require the reader to summarize average trends.
- Value: ⭐⭐⭐⭐☆ Highly practical for test-time compute compression, especially for low-effort deployment in existing CISC/LLM-as-judge pipelines.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Calibrated Speculative Decoding: Frequency-Guided Candidate Selection for Efficient Inference](calibrated_speculative_decoding_frequency-guided_candidate_selection_for_efficie.md)
- [\[ACL 2026\] CadLLM: Improving the Throughput of Diffusion-based LLMs via Training-Free Confidence-Aware Calibration](improving_the_throughput_of_diffusion-based_large_language_models_via_a_training.md)
- [\[NeurIPS 2025\] Representation Consistency for Accurate and Coherent LLM Answer Aggregation](../../NeurIPS2025/model_compression/representation_consistency_for_accurate_and_coherent_llm_answer_aggregation.md)
- [\[ACL 2026\] DeepPrune: Parallel Scaling without Inter-Trace Redundancy](deepprune_parallel_scaling_without_inter-trace_redundancy.md)
- [\[ACL 2026\] BaseCal: Unsupervised Confidence Calibration via Base Model Signals](basecal_unsupervised_confidence_calibration_via_base_model_signals.md)

</div>

<!-- RELATED:END -->

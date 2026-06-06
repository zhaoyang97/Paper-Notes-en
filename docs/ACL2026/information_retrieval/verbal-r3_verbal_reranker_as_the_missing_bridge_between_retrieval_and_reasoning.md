---
title: >-
  [Paper Note] Verbal-R3: Verbal Reranker as the Missing Bridge between Retrieval and Reasoning
description: >-
  [ACL2026][Information Retrieval & RAG][Retrieval-Augmented Generation] Verbal-R3 upgrades traditional rerankers from modules that "only provide relevance scores" to bridging modules that "provide scores and generate expl…
tags:
  - "ACL2026"
  - "Information Retrieval & RAG"
  - "Retrieval-Augmented Generation"
  - "Reranking"
  - "Verbal Annotation"
  - "Reinforcement Learning"
  - "Test-time Scaling"
date: 2026-05-08
content_hash: 2115e4ba881f9e8e
---

# Verbal-R3: Verbal Reranker as the Missing Bridge between Retrieval and Reasoning

**Conference**: ACL2026  
**arXiv**: [2605.01399](https://arxiv.org/abs/2605.01399)  
**Code**: https://github.com/0k9d0h1/VerbalR3  
**Area**: information_retrieval  
**Keywords**: Retrieval-Augmented Generation, Reranking, Verbal Annotation, Reinforcement Learning, Test-time Scaling

## TL;DR
Verbal-R3 upgrades traditional rerankers from modules that "only provide relevance scores" to bridging modules that "provide scores and generate explanatory Verbal Annotations." These are then used to train and guide RAG reasoners, simultaneously improving answer accuracy and test-time scaling efficiency in multi-hop question answering.

## Background & Motivation
**Background**: RAG systems typically retrieve candidate documents and concatenate raw passages into the LLM context, allowing the model to answer questions based on external information. To improve retrieval quality, many systems add a reranker to re-order documents from the first stage and select top-k documents for the generator.

**Limitations of Prior Work**: The output of traditional rerankers is usually just a relevance score or a document ranking. While this ranking helps in "selecting which documents," it does not inform the generator "why the document is relevant, which part supports the answer, or what content should be ignored." In multi-hop questions, as the retrieve context grows longer and noisier, the generator may fail to correctly utilize the document even if it contains the answer.

**Key Challenge**: The bottleneck of RAG is not just retrieval accuracy, but context utilization. The retrieval module focuses on query-document matching, while the reasoning module focuses on evidence chains and logic; there is a lack of an intermediate representation that translates "relevant documents" into "reasoning-ready evidence."

**Goal**: The authors aim to construct a lightweight Verbal Reranker that can provide relevance scores like a standard reranker while generating natural language analysis for the reasoner, helping the generator filter noise, understand evidence, and prioritize computational budgets for promising query branches during test-time.

**Key Insight**: The paper conducts a motivation experiment comparing raw text, paraphrasing, and Verbal Annotation within Search-R1. Results show that simple paraphrasing can even hurt performance, while Verbal Annotation explaining the logic between the query and document significantly improves EM/F1 and Context Utilization Efficacy. This indicates the problem is not "surface text smoothness" but whether "evidence is explicitly structured."

**Core Idea**: Enable the reranker to output "relevance scores + reasoning-oriented verbal annotations," then distill this module into a small model and embed it into iterative RAG, ensuring retrieval results are interpreted into usable evidence before reaching the generator.

## Method
Verbal-R3 can be viewed as a two-module RAG framework: the Generator handles iterative reasoning, generates retrieval queries, integrates information, and produces answers; the Verbal Reranker handles producing Verbal Annotations and relevance scores from 1 to 5 for each query-document pair. Compared to standard RAG, it does not just concatenate top-k documents but top-k "interpreted document evidence," and uses relevance scores to control which reasoning branches deserve further expansion during test-time.

### Overall Architecture
Training is divided into three stages. First, GPT-OSS-120B is used as a teacher model to generate Verbal Annotations and scalar relevance scores for a large number of query-document pairs; Gemini 2.5 Pro then verifies the quality of synthetic labels, with a reported pass rate of 98.5%. Second, these synthetic triples are distilled into Qwen2.5-Instruct 1.5B/3B student models to obtain a deployable lightweight Verbal Reranker. Third, the Verbal Reranker is integrated into the Generator's interaction trajectory, and the Generator is trained via Reinforcement Learning (GRPO) to learn to propose better retrieval queries, read verbal annotations, and output final answers in a specified format.

During inference, the Generator first generates a search query based on the question; the retriever returns $n=15$ candidate documents; the Verbal Reranker generates an annotation $v_i$ and a score $s_i$ for each; the system selects top-k (default $k=3$) based on scores and score token logits, writing them into the Generator's context within `<information>` blocks. This process repeats until the final answer is parsed or the maximum number of rounds is reached.

### Key Designs
1.  **Verbal Annotation as an Intermediate Layer between Retrieval and Reasoning**:
    *   **Function**: Transforms raw documents from "potentially relevant text" into "reasoning scaffolds" explaining why they are relevant, how they support the question, and which parts are irrelevant.
    *   **Mechanism**: The Verbal Reranker generates two types of output for each query-document pair: natural language critique/annotation and a 1-5 relevance score. The annotation is neither a summary nor a paraphrase, but an explicit explanation of the logical relationship and identification of noise or gaps.
    *   **Design Motivation**: LLM reasoners may not automatically find key evidence in lengthy retrieved passages. Verbal Annotation rewrites retrieval results into evidence descriptions closer to reasoning needs, increasing the probability of a correct answer given accurate retrieval.

2.  **Distillation from Large Model Teacher to Lightweight Verbal Reranker**:
    *   **Function**: Avoids calling the 120B teacher model for every RAG reasoning step, reducing latency and cost in iterative retrieval.
    *   **Mechanism**: The authors use prompt search to make GPT-OSS-120B produce discriminative annotations rather than generic summaries; subsequently, a large-scale query-document-annotation-score dataset is constructed to perform SFT on 1.5B/3B students, targeting the maximization of the likelihood of the output sequence $y_{VR}=(v,s)$.
    *   **Design Motivation**: The value of Verbal Annotation comes from high-quality judgment, but deployment must be low-cost. Distillation allows the small model to inherit the judgment patterns of "why a document is useful" from the large model while maintaining the high throughput required for iterative RAG.

3.  **Relevance-Guided Test-Time Scaling**:
    *   **Function**: Allocates more computation to reasoning branches with higher retrieval quality under a limited trajectory budget.
    *   **Mechanism**: After generating multiple queries per round, the system extracts unique queries to avoid redundant retrieval; for each query's candidate documents, it takes the maximum relevance score $s_{max,q}$, and selects the next branch based on $P(q)=(s_{max,q}/s_{best})^\alpha$, with $\alpha=7.5$ and a total trajectory budget $N=5$. Majority voting is performed on collected answers.
    *   **Design Motivation**: Standard majority voting expands all trajectories equally, wasting reranker calls. Relevance scores act as signals for "whether this query found good evidence," used to filter branches worth continuing.

### Loss & Training
The Verbal Reranker uses SFT: input $x_{VR}=(i_{VR}, q, d)$, output $y_{VR}=(v,s)$, with the objective $L_{SFT}=-E_{(x,y)\sim D}\sum_t \log p_\theta(y_t|x,y_{<t})$. The Generator's Reinforcement Learning follows the Search-R1/GRPO style, calculating policy gradients only for tokens generated by the Generator, while reranker outputs are included in the trajectory as environmental information. Rewards are tiered: outcome reward for correct answers and format; format penalties for correct answers with wrong format; format rewards for wrong answers with correct format; and 0 or answer rewards if the format is unparseable. This design constrains both factual accuracy and RAG trajectory formatting.

## Key Experimental Results

### Main Results
Experiments cover 7 QA benchmarks: single-hop NQ, TriviaQA, PopQA; multi-hop 2WikiMultiHopQA, Bamboogle, MuSiQue, HotpotQA; separate reranking capability is evaluated on BEIR. E5 is used as the base retriever, Qwen2.5-3B/7B as generators, a 3B Verbal Reranker by default, with $n=15$ retrieved and $k=3$ retained.

| Method/Module | Avg. EM | Avg. F1 | Key Conclusion |
| :--- | :--- | :--- | :--- |
| Search-R1 3B | 38.75 | 46.31 | Strong RAG baseline, but uses raw retrieved text |
| Search-R1 3B + Verbal Annotation | 41.92 | 49.86 | Significant improvement with annotations in motivation exp. |
| Verbal-R3 3B | 45.36 | 54.66 | Gain +17.1% EM, +18.0% F1 relative to Search-R1 3B |
| Verbal-R3 7B | 48.30 | 57.28 | Gain +15.3% EM, +14.3% F1 relative to Search-R1 7B |
| Ours 3B Verbal Reranker Single-shot | 30.19 | 38.84 | Outperforms MonoT5/RankLLaMA/Rank1 in single-shot QA reranking |

### Ablation Study

| Configuration | Key Metrics | Description |
| :--- | :--- | :--- |
| Search-R1 w/o VA | EM 38.75, F1 46.31, CUE-R 64.77, CUE-L 69.04 | Efficiency of utilizing evidence is limited with raw context |
| Search-R1 w/ VA | EM 41.92, F1 49.86, CUE-R 70.76, CUE-L 74.10 | Low change in retrieval accuracy, but significant gain in context utilization |
| Verbal-R3 w/o VA | EM 42.73, F1 51.75, CUE-R 65.21, CUE-L 69.76 | Only reranking/filtering improves retrieval quality |
| Verbal-R3 Full | EM 45.36, F1 54.66, CUE-R 67.54, CUE-L 71.64 | Annotation further improves evidence utilization and final quality |

| Scaling Strategy | Verbal-R3 3B Avg. EM/F1 | 3B Reranker Calls | Verbal-R3 7B Avg. EM/F1 | 7B Reranker Calls |
| :--- | :--- | :--- | :--- | :--- |
| No M.V. | 45.36 / 54.66 | 2.27 | 48.30 / 57.28 | 1.82 |
| Naive M.V. w/o U.Q.E. | 47.26 / 56.33 | 11.28 | 49.93 / 59.02 | 9.11 |
| Naive M.V. w/ U.Q.E. | 46.69 / 55.88 | 6.52 | 49.23 / 58.35 | 4.63 |
| Relevance-Guide | 47.18 / 56.45 | 6.18 | 50.12 / 59.25 | 4.21 |

### Key Findings
- The primary benefit of Verbal Annotation is not increasing the probability of "retrieving the answer," but increasing the probability of "using it correctly once retrieved." The improvement in CUE-R/CUE-L is more pronounced than in RA.
- The 3B Verbal Reranker outperforms larger traditional reranker baselines in single-shot RAG, indicating that natural language explanation provides additional value.
- Gains are larger for multi-hop tasks: Verbal-R3 3B shows an average F1 gain of ~26.91% on multi-hop tasks, significantly higher than 9.67% for single-hop. This aligns with intuition as multi-hop retrieval accumulates more noise, requiring stronger annotation for filtering.
- Relevance-guided scaling requires 45.2% (3B) and 53.8% (7B) fewer reranker calls than naive majority voting without deduplication while maintaining or improving metrics.

## Highlights & Insights
- The most inspiring aspect is the redefinition of the reranker interface: scores for "ranking" and annotations for "usability." This is closer to the ultimate goal of RAG than pursuing nDCG alone, as generators require reasoning evidence, not just relevant documents.
- The motivation experiment solidly distinguishes between paraphrasing and verbal annotation. The former is merely style transfer and may lose evidence; the latter is a logical bridge specifically telling the model why the text supports the query.
- Recyling relevance scores for test-time scaling is a clever resource scheduling design. The reranker is no longer just a preprocessing module but becomes a value estimator for reasoning tree expansion.

## Limitations & Future Work
- Training the Verbal Reranker depends on GPT-OSS-120B for annotations and Gemini 2.5 Pro for validation; while the deployed model is small, the data construction phase has strong teacher dependence.
- Annotations increase the tokens per reranker output; the paper mentions ~990 tokens per call at $n=15$. Despite savings from relevance-guided scaling, further compression of annotation length is needed for low-latency scenarios.
- Tasks are primarily QA/RAG benchmarks; it is unclear if verbal annotation remains stable for open-ended long-form generation, legal/medical reasoning, or multimodal RAG.
- Verbal Annotations are model-generated and may contain misleading explanations; if the reranker provides a plausible-sounding annotation for an incorrect document, the generator might over-rely on it.

## Related Work & Insights
- **vs. Standard RAG**: Standard RAG concatenates retrieved passages directly; Verbal-R3 adds an explanatory layer before documents enter the generator, making the context resemble an "evidence manual."
- **vs. MonoT5 / RankLLaMA / Rank1**: These rerankers focus on optimizing relevance ranking; Verbal Reranker generates both relevance scores and annotations, serving both ranking and reasoning.
- **vs. Search-R1**: Search-R1 already performs iterative retrieval and reasoning but uses raw results; Verbal-R3 integrates the Verbal Reranker into the same search-based framework and trains the Generator to adapt to this evidence format.
- **vs. Majority Voting**: Standard majority voting blindly expands multiple trajectories; this work uses reranker scores to decide which branches are worth continuing, concentrating the computational budget.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Turning the reranker into an "explanatory evidence bridge" and using scores for test-time scaling is highly inspiring.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers various QA types, reranker baselines, ablations, and scaling; however, actual latency/cost and annotation error propagation could be more detailed.
- Writing Quality: ⭐⭐⭐⭐☆ Flows well from motivation experiments to the main methodology with rich tables; however, the notation and training details are somewhat dense.
- Value: ⭐⭐⭐⭐⭐ Very practical for RAG systems, especially addressing "retrieved but unable to use" issues in multi-hop QA and agentic retrieval.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Embedding-Based Context-Aware Reranker](../../ICLR2026/information_retrieval/embedding-based_context-aware_reranker.md)
- [\[CVPR 2026\] BRIDGE: Multimodal-to-Text Retrieval via Reinforcement-Learned Query Alignment](../../CVPR2026/information_retrieval/bridge_multimodal-to-text_retrieval_via_reinforcement-learned_query_alignment.md)
- [\[ACL 2026\] A Survey of Reasoning-Intensive Retrieval: Progress and Challenges](a_survey_of_reasoning-intensive_retrieval_progress_and_challenges.md)
- [\[ACL 2026\] ChatR1: Reinforcement Learning for Conversational Reasoning and Retrieval Augmented Question Answering](chatr1_reinforcement_learning_for_conversational_reasoning_and_retrieval_augment.md)
- [\[ACL 2026\] Agentic Conversational Search with Contextualized Reasoning via Reinforcement Learning](agentic_conversational_search_with_contextualized_reasoning_via_reinforcement_le.md)

</div>

<!-- RELATED:END -->

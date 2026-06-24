---
title: >-
  [Paper Note] LongReward: Improving Long-context Large Language Models with AI Feedback
description: >-
  [ACL 2025][LLM Efficiency][AI Feedback] Academic paper note for LongReward: Improving Long-context Large Language Models with AI Feedback.
tags:
  - ACL 2025
  - LLM Efficiency
  - AI Feedback
  - DPO
date: 2026-05-08
content_hash: 228e77ba4f3b7bcc
---
# LongReward: Improving Long-context Large Language Models with AI Feedback

**Conference**: ACL2025  
**arXiv**: [2410.21252](https://arxiv.org/abs/2410.21252)  
**Code**: [THUDM/LongReward](https://github.com/THUDM/LongReward)  
**Area**: LLM Efficiency  
**Keywords**: Long-context Alignment, AI Feedback, DPO, Hallucination Mitigation, Long-document QA

## Background & Motivation

The engineering capabilities of long-context models have improved rapidly over the past two years, making 128k or even longer windows no longer scarce. However, "being able to read long context" and "understanding long context" are two entirely different matters.
The authors point out that the post-training phase of many long-context models heavily relies on automatically synthesized QA data. While affordable, this step inevitably introduces quality issues into the models.
These deficiencies can still be masked by templated responses in short-context scenarios, but they are directly exposed as omissions, hallucinations, logical fragmentation, and irrelevant answers in long-document QA and summarization.
Existing RLHF or RLAIF frameworks have proven that reward learning is effective for short contexts, but the long-context counterpart has long lacked a viable reward signal.
Annotating long-context preferences with human annotators is excessively expensive, as reviewing a single long-context response is comparable to a complete reading task itself.
Traditional short-context reward models have context windows that are too short to consume the entire source document, making it impossible to determine whether a response, though fluent, is genuinely supported by the context.
Consequently, the authors deconstruct the problem into two levels.
The first level is reward design: how to enable an off-the-shelf LLM to provide reliable evaluations in long-context scenarios.
The second level is training implementation: how to convert these evaluations into trainable preference pairs to truly enhance long-context models.
The core observation of this paper is that the quality of long-context responses is not a single scalar value, but is instead determined by at least four relatively independent dimensions.
Helpfulness focuses on whether the response directly addresses the query.
Logicality examines whether the reasoning chain contains self-contradictions.
Faithfulness verifies whether the facts in the response are supported by the context.
Completeness assesses whether crucial information is omitted, especially information nested in the middle or end of long texts.
Collectively, these four dimensions closely approximate authentic user subjective evaluations of long-text responses.
From a motivational standpoint, this paper does not aim to train a new long-context reward model from scratch. Instead, it builds a reliable evaluation pipeline using an existing strong model and integrates it into offline RL frameworks like DPO.
The value of this approach lies in its lower barrier to entry, faster implementation, and better alignment with the rapidly evolving state of long-context data.

## Method

The overall workflow of LongReward can be summarized as "multi-dimensional scoring, preference pair construction from scores, and long-context DPO".
The input consists of a long-context prompt (typically composed of a long document context and a user query) along with a candidate response.
The output comprises 0-10 scores across the four dimensions, and their average serves as the final reward.
Instead of forcing a single prompt template to resolve all aspects, the authors design distinct evaluation pipelines for different dimensions, which is the most critical engineering design of this work.

The first component is helpfulness scoring.
Helpfulness primarily relies on the query and the response, essentially assessing whether the response is relevant, satisfactory, and informative.
Since the authors allocate truthfulness specifically to faithfulness, helpfulness evaluation does not strictly depend on the original long context.
The specific practice is to provide the judge model with scoring principles, few-shot examples, and CoT prompts, instructing it to analyze before scoring.
The advantage of this approach is extracting "long-context content retrieval" from helpfulness, preventing excessive computational overhead during evaluation.

The second component is logicality scoring.
The authors observe that a common type of error in long-context models is not factual inaccuracy, but inconsistency in reasoning.
For example, a model might draw a conclusion in the beginning but contradict it in a later paragraph, or commit an intermediate calculation error while arriving at a seemingly correct conclusion.
Logicality also primarily depends on the query and response, and thus also employs a direct scoring strategy via few-shot learning and CoT.
Unlike helpfulness, the logicality prompt explicitly demands that the judge locate logical errors before outputting a score.
This effectively transforms "fault-finding" into a pre-requisite for scoring, reducing the risk of the model assigning high scores solely based on language fluency.

The third component is faithfulness scoring.
This is the most technically sophisticated part of LongReward, as it must explicitly verify that the response is derived from the context.
The authors adapt the break-and-check approach of FactScore, making three key modifications for long-context scenarios.
First, instead of breaking the response down into atomic facts, it decomposes it into sentence-level factual statements, which yields more robust retrieval recall and reduces the number of verifications required for extremely long responses.
Second, functional sentences (e.g., "Here are several reasons below") that do not carry factual claims are filtered out.
Third, each statement is not evaluated on a binary basis, but rather graded on a three-tier system: full support, partial support, and no support, corresponding to values of 1, 0.5, and 0.
Finally, the faithfulness score is aggregated as $10 \cdot \sum_i a_i / n$, where $a_i$ represents the support level of each statement.
To find evidence for each statement, the system uses the statement as a query to retrieve the top-5 128-token segments from the original context, and then tasks the judge with assessing the level of support.
This step transforms "reading the entire long text to verify a sentence" into "local evidence verification", keeping costs highly manageable.

The fourth component is completeness scoring.
Completeness does not evaluate whether the response contains nonsense, but rather whether it misses key points.
The authors emphasize that long-context models often forget information located in the middle; hence, feeding the entire long context along with the response directly to a judge might cause the judge itself to overlook details.
LongReward adopts a divide-and-conquer strategy.
It first partitions the original context into coarse-grained blocks of 4096 tokens.
Then, it instructs the judge to extract query-relevant information from each block.
Next, it aggregates the extracted information from all blocks and tasks the judge with evaluating whether the response covers these key points.
The essence of this design is to perform "information summarization" first, followed by "coverage assessment," thereby preventing the evaluation model from suffering from position bias on ultra-long inputs.

Following the four-dimensional scoring, the final reward is the average of the four scores.
Rather than manually tuning the weights of different dimensions, the authors apply equal weighting by default.
This reflects the paper's stance: usefulness, logic, truthfulness, and completeness are all indispensable, and establishing a working pipeline with a balanced objective is more critical than elaborate fine-tuning of weights at this stage.

After obtaining the rewards, the authors construct long-context preference data.
For each long-context prompt, 10 candidate responses are first sampled from the SFT model utilizing a temperature of 1.0.
LongReward is then executed for each of the 10 candidates.
The response with the highest score is designated as the winning response, while the lowest is marked as the losing response.
This effectively converts "score supervision" into the pairwise preferences required by DPO.

During the training phase, DPO is deployed, but the authors do not rely solely on the standard DPO loss.
To stabilize training, they incorporate a cross-entropy regularization term on the winning response.
The total loss is calculated as $\mathcal{L}_{merge} = \mathcal{L}_{DPO} + \lambda \mathcal{L}_{CE}$.
The intuition here is clear: while DPO widens the relative probability gap between good and bad responses, an excessively strong DPO constraint may impair the stability of language modeling; the CE regularization helps prevent the model from drifting too far from its original generation distribution.

Methodologically, the crux of LongReward is not simply "creating yet another judge prompt," but rather decomposing long-context scoring into actionable, locally verifiable sub-problems.
This enables an off-the-shelf LLM to provide robust rewards along a structured workflow rather than relying on intuition to assign a single overall score for long-document tasks.

| Dimension | Input Dependency | Core Mechanism | Main Problem Solved |
|------|----------|----------|----------------|
| Helpfulness | query + response | few-shot + CoT direct scoring | Addresses user needs |
| Logicality | query + response | Identify logical errors before scoring | Contradictions and reasoning gaps |
| Faithfulness | response + retrieved segments | Sentence splitting, retrieval, three-tier support determination | Hallucinations and factual distortion |
| Completeness | query + chunked context + response | Chunk-wise key info extraction followed by coverage assessment | Omissions of critical long-text information |

| DPO Data Construction Step | Concrete Implementation | Purpose |
|------------------|----------|------|
| Candidate Sampling | Sample 10 responses per prompt | Guarantees sufficient diversity among candidates for the same query |
| LongReward Scoring | Calculate average score across four dimensions for each response | Generates stable rewards |
| Preference Pair Extraction | Highest score vs. lowest score | Converts to DPO-trainable samples |
| DPO + CE Training | $\beta=0.15$, $\lambda=0.1$ | Simultaneously improves preference learning and training stability |

## Key Experimental Results

The experimental base models include Llama-3.1-8B and GLM-4-9B, both of which support 128k context windows and undergo long-context SFT before DPO.
The long-context SFT data consists of a mixture of 10k long-text QA pairs and 76k ShareGPT general instructions. The long context lengths span 8k to 64k tokens, covering 9 domains.
LongReward scoring is executed using GLM-4, and the faithfulness retriever utilizes Zhipu-Embedding-2, retrieving the top-5 segments for each factual statement.
In terms of training, SFT is conducted for 1800 steps, while DPO runs for approximately 400 to 800 steps, indicating that the performance gains in this work are not driven by excessively prolonged training, but rather by superior preference construction strategies.

| Experimental Setting | Configuration |
|----------|------|
| Base Models | Llama-3.1-8B, GLM-4-9B |
| Context Length | Supports 128k, trained up to 64k |
| Long-text SFT Data | 10k long-text QA + 76k ShareGPT |
| Candidate Counts | 10 samples per prompt |
| Faithfulness Retrieval | Zhipu-Embedding-2, top-5 segments |
| DPO Hyperparameters | $\beta=0.15$, $\lambda=0.1$, lr=$1e-6$ |
| Training Resources | 4 nodes, each with 8 H800 GPUs |

Notably, in the main results, LongReward significantly outperforms SFT, short-context reward models, and the "Contrast" baseline (which treats LLM-generated responses as positive samples) on both base models.
For Llama-3.1-8B, LongReward DPO achieves an average score of 59.9, representing a 4.9-point improvement over SFT (55.0).
For GLM-4-9B, the average score increases from 56.6 to 62.1, a gain of 5.5 points.
Particularly on Multi-Doc QA, the Llama version rises directly from 44.5 to 55.8, demonstrating that the proposed method is exceptionally adept at mitigating omissions and mismatches during multi-document information aggregation.

| Model | Method | LongBench-Chat | S-Doc QA | M-Doc QA | Summ | Avg |
|------|------|----------------|----------|----------|------|-----|
| Llama-3.1-8B | SFT | 69.8 | 66.1 | 44.5 | 39.6 | 55.0 |
| Llama-3.1-8B | DPO w/ SRM | 67.4 | 65.0 | 49.6 | 42.7 | 56.2 |
| Llama-3.1-8B | DPO w/ Contrast | 70.6 | 67.8 | 46.2 | 40.3 | 56.2 |
| Llama-3.1-8B | **DPO w/ LongReward** | **72.6** | **67.8** | **55.8** | **43.2** | **59.9** |
| GLM-4-9B | SFT | 64.8 | 68.4 | 50.9 | 42.1 | 56.6 |
| GLM-4-9B | DPO w/ SRM | 66.6 | 67.5 | 57.4 | 48.2 | 59.9 |
| GLM-4-9B | DPO w/ Contrast | 68.2 | 67.8 | 58.0 | 47.8 | 60.5 |
| GLM-4-9B | **DPO w/ LongReward** | **69.2** | **71.9** | **58.8** | **48.5** | **62.1** |

Additionally, the paper conducts two crucial supplementary experiments.
The first is a FactScore analysis.
LongReward DPO does not merely make responses "look better"; it actually increases the proportion of context-supported facts in the output while incorporating more atomic facts.
The FactScore of the Llama version increases from 91.94 to 92.85, and the average number of atomic facts grows from 21.76 to 32.86.
This indicates that the model does not avoid mistakes by shortening its answers; rather, it provides more detail while maintaining higher truthfulness.

The second is human evaluation.
On LongBench-Chat, the overall win rate of LongReward DPO against SFT is 54%, with a loss rate of only 8%.
Among the four dimensions, faithfulness and completeness see the most substantial improvements, which perfectly aligns with the methodological design, as these two dimensions are precisely where LongReward places significant modeling effort.

| Analysis Metric | SFT | DPO w/ LongReward | Conclusion |
|--------|-----|-------------------|------|
| Llama #Facts | 21.76 | 32.86 | More detailed responses |
| Llama FactScore | 91.94 | 92.85 | More detail without sacrificing truthfulness |
| GLM #Facts | 18.41 | 28.05 | Covers more information |
| GLM FactScore | 91.43 | 93.62 | Further reduction in hallucinations |
| Human Overall Win/Loss | - | 54% / 8% | Significantly higher human subjective preference |

Another easily overlooked yet important finding is that long-context DPO driven by LongReward also enhances short instruction-following capabilities.
For instance, on AlpacaEval2, the Llama version improves from 12.4 to 14.2, and GLM rises from 12.5 to 15.4.
This indicates that the preference patterns learned in this work are not "localized tricks specialized for long-text QA," but are instead a more general set of output values, such as maximizing truthfulness, completeness, and logical consistency.

The authors also compare the alignment of different reward strategies with human preferences.
LongReward achieves a prediction accuracy of 66.2%, which is superior to the 58.3% of the short-context reward model and the 57.1% of direct pairwise comparison.
More importantly, removing any single dimension leads to a drop in performance; specifically, omitting faithfulness or completeness drops accuracy to 57.8%.
This experimentally proves that the four-dimensional design is not cosmetic but is an essential component.

## Highlights & Insights

The most compelling highlight is the decomposition of the vague problem "difficulty in acquiring long-context rewards" into four actionable subnets/sub-tasks, each implemented with an independent scoring mechanism, rather than relying on a single all-encompassing prompt template.
The break-and-check combined with retrieval verification for faithfulness is highly pragmatic, essentially converting open-ended long-text verification into localized evidence-support judgments, which offers great reusability in engineering.
The chunk-wise extraction followed by coverage evaluation for completeness is also highly inspiring. This represents a design paradigm that "first compresses long text into query-aware memory, and then conducts judging."
Furthermore, this paper demonstrates that long-context alignment does not need to conflict with short-context alignment; the two can even be co-trained using a mixture of preference data.
From a research perspective, LongReward represents a lighter trajectory than training a new reward model: first establishing a reliable annotation pipeline with a strong judge, and subsequently distilling or training more affordable models.
From an engineering standpoint, this solution is well-suited for rapid iteration, as the four-dimensional prompts, retrievers, and chunk granularities can all be modularly replaced.
It is argued that the most vital insight of this paper is not that "DPO is effective," but rather that "the quality of long-context responses must be strictly structurally defined, otherwise the reward signals themselves will be drowned out by long-context noise."

## Limitations & Future Work

First, the operational cost remains high.
Extracting each QA sample requires multiple candidate samplings, multi-dimensional scoring, faithfulness retrieval-verification, and chunk-wise completeness analysis, resulting in a substantial volume of API calls.
Second, the experimental scale is limited to models around the 10B parameter range and a maximum training sequence length of 64k, leaving the performance gains on even larger models and longer sequences yet to be fully validated.
Third, the evaluation tasks are mainly concentrated on long-document QA and summarization, leaving scenarios like multi-turn long historical dialogue, long-term agent memory, and repository-level code reasoning unaddressed.
Fourth, although equal weighting across the four dimensions is straightforward, real-world preference weights may vary drastically across different tasks. For instance, legal retrieval and creative summarization focus differently on completeness versus helpfulness.
There are three feasible directions for future work.
The first is to leverage the vast volume of preference pairs generated by LongReward to train specialized long-context reward models, thereby reducing inference overhead.
The second is to further refine completeness and faithfulness into citation-level supervision, such as requiring responses to append specific evidence spans or paragraph citations.
The third is to port the framework to agent trajectory evaluation, incorporating "plan completeness" and "execution faithfulness" into the reward design.

## Related Work & Insights

Compared with traditional RLHF, the primary distinction of this work lies in deeply structuring the AI judge rather than directly performing pairwise preference judgment.
Compared with short-context reward models, LongReward explicitly acknowledges that short RMs cannot observe evidence within long contexts, and thus cannot be directly repurposed.
Compared with fact-checking frameworks like FactScore, this paper does not perform assessment in isolation; instead, it embeds fact-checking directly into the training data construction pipeline, turning it into training signals that can tangibly improve the model.
Compared with the Contrastive approach of "treating LLM outputs directly as high-quality responses," this paper does not treat the LLM as the sole teacher. Rather, it breaks the teacher's judgment into multi-dimensional rewards, making it more interpretable and less dependent on any single output template.
This work offers two primary inspirations.
First, for alignment issues involving long context, long trajectories, or long documents, one should prioritize decomposing the reward into several locally verifiable attributes rather than pursuing a single holistic score.
Second, completeness is often more challenging than helpfulness because it requires knowing "what should have been said but was omitted." Such errors of omission are well-suited for a chunk-wise summarization then coverage evaluation strategy.
For future work on long-context agents or repository-level code reviews, the core idea of LongReward can be adapted into a four-dimensional evaluation framework assessing "correctness, evidentiality, completeness, and executability."

## Rating

- **Novelty**: ⭐⭐⭐⭐☆  Decomposes long-context reward design into four systematic dimensions with independent operational pipelines.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐  Extensively validated with two base models, automatic evaluation, human evaluation, FactScore, short-context capability, and mixed DPO training.
- **Writing Quality**: ⭐⭐⭐⭐☆  The core methodology and experimental logic are coherent, especially the transition from multi-dimensional scoring to DPO.
- **Value**: ⭐⭐⭐⭐⭐  A pragmatically constructive contribution to long-context alignment that benefits reward modeling, RLAIF, and future agent evaluation.
- **Overall Evaluation**: 9.0/10. Instead of relying on a flashy architecture, it successfully transforms a long-vacant training signal problem into a reusable, scalable, and reliably beneficial solution.

---
title: >-
  [Paper Review] LongReward: Improving Long-context Large Language Models with AI Feedback
description: >-
  [ACL2025][LLM Efficiency][Long-context LLM] Introduces LongReward, leveraging off-the-shelf LLMs to score long-context responses across helpfulness, logicality, faithfulness, and completeness, paired with DPO to significantly enhance long-context SFT model capabilities.
tags:
  - ACL2025
  - LLM Efficiency
  - Long-context LLM
  - RL
  - DPO
  - AI Feedback
  - Reward Model
---

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] LongSafety: Evaluating Long-Context Safety of Large Language Models](longsafety_evaluating_long-context_safety_of_large_language_models.md)
- [\[ACL 2025\] CNNSum: Exploring Long-Context Summarization with Large Language Models in Chinese Novels](cnnsum_exploring_long-context_summarization_with_large_language_models_in_chines.md)
- [\[ACL 2025\] Dynamic Chunking and Selection for Reading Comprehension of Ultra-Long Context in Large Language Models](dynamic_chunking_and_selection_for_reading_comprehension_of_ultra-long_context_i.md)
- [\[ACL 2025\] Ref-Long: Benchmarking the Long-Context Referencing Capability of Long-Context Language Models](ref-long_benchmarking_the_long-context_referencing_capability_of_long-context_la.md)
- [\[ACL 2025\] How to Train Long-Context Language Models (Effectively)](train_long_context_effectively.md)

</div>

<!-- RELATED:END -->

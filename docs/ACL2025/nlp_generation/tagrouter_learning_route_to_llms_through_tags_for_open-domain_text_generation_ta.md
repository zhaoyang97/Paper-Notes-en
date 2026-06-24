---
title: >-
  [Paper Note] TagRouter: Learning Route to LLMs through Tags for Open-Domain Text Generation Tasks
description: >-
  [ACL 2025][Text Generation][model routing] This paper proposes TagRouter, which uses a small tag generator to compress open-domain text generation requests into a set of semantic tags, and then routes requests by analyzing the relative advantages of each candidate LLM based on these tags. This achieves a higher system acceptance rate than any single large model without retraining the router, while significantly reducing inference costs.
tags:
  - "ACL 2025"
  - "Text Generation"
  - "model routing"
  - "tag generation"
  - "LLM ensemble"
  - "cost-efficient inference"
  - "open-domain generation"
date: 2026-05-08
content_hash: 208af0f283c21a87
---

# TagRouter: Learning Route to LLMs through Tags for Open-Domain Text Generation Tasks

**Conference**: ACL 2025  
**arXiv**: [2506.12473](https://arxiv.org/abs/2506.12473)  
**Code**: None  
**Area**: Text Generation  
**Keywords**: model routing, tag generation, LLM ensemble, cost-efficient inference, open-domain generation

## TL;DR
This paper proposes TagRouter, which uses a small tag generator to compress open-domain text generation requests into a set of semantic tags, and then routes requests by analyzing the relative advantages of each candidate LLM based on these tags. This achieves a higher system acceptance rate than any single large model without retraining the router, while significantly reducing inference costs.

## Background & Motivation

The LLM ecosystem is developing rapidly, but different models do not simply align along a single dimension of parameter size. The paper starts with a very intuitive phenomenon: even smaller models can perform on par with or even outperform larger models on specific samples. In other words, a high average benchmark score does not mean a model is the most suitable for every single user request.

Although existing routing methods are already attempting to distribute requests to different models, they face several obvious issues in practice.

The first category of methods requires multiple candidate models to generate responses first, and then makes a selection at the response level. While these methods may achieve good accuracy, they suffer from high latency and costs, making them particularly unsuitable for online scenarios.

The second category of methods relies on decoding-stage information, such as logits or intermediate states. This is highly unfriendly to closed-source commercial models, as this internal information is often inaccessible.

The third category of methods treats routing as a supervised classification task. They usually require retraining for specific combinations of candidate models. Once the model pool changes, retraining is necessary, which fails to adapt to the rapidly evolving LLM ecosystem.

The Key Challenge the authors aim to address is: can a truly practical routing method be designed that simultaneously meets the following conditions?

- Pre-inference routing, avoiding redundant calls to multiple large models.
- Support for open-domain text generation, independent of task-specific templates.
- Support for multiple candidate models, rather than just choosing between one large and one small model.
- Support for proprietary models, without requiring access to logits.
- Explicit consideration of cost control.
- No need to retrain the entire router when the model pool expands.

The authors' Key Insight is very interesting: rather than training a router directly on the raw queries, it is better to extract tags from the query first. Tags are naturally closer to "task semantic features" and can compress redundant text into structured representations that generalize more easily. Thus, the Core Idea of the entire paper can be summarized in one sentence: first convert queries to tags, then query which semantic patterns each model excels at based on these tags, and finally make a cost-constrained selection.

## Method

TagRouter consists of three sequentially executed modules: TagGenerator, TagScorer, and TagDecider. It is not an end-to-end large neural router, but rather a modular system combining "tag extraction + capability dictionary + decision rules." The focus of this design is not to make the router heavy, but to make it maintainable, extensible, and replaceable.

Looking at the overall workflow, given a user request $q$, the system first uses TagGenerator to generate a tag set $\mathcal{T}(q)=\{t_1,t_2,...,t_j\}$. Then, TagScorer calculates a cumulative score for each candidate model $M$ based on these tags. Finally, TagDecider determines which model to route the request to by combining the score difference with a cost threshold $\theta$.

The formal definition of the routing problem it solves is: in a set of models $\mathcal{M}$, select the most suitable model $M^*(q)$ for each query to maximize the overall system performance. Unlike common query-to-label classifiers, TagRouter learns a more relaxed structure: it does not directly train a hard mapping from "query to model" but instead uses tags as an intermediate semantic layer to connect the query with model capability.

### Overall Architecture

The overall architecture can be broken down into three steps.

First, TagGenerator converts the input request into multiple fine-grained tags. The authors use open-tagging instead of a fixed tag vocabulary. They initially use ERNIE-4.0-Turbo-8K to automatically label queries in the BCUQ dataset, obtaining a total of 14,352 unique raw tags.

Second, these tags are normalized and compressed. The authors first remove rare tags that appear fewer than 5 times, apply rule-based normalization, represent tag semantics using PhraseBERT, and merge similar tags via DBSCAN clustering. After iterative merging, the final tag vocabulary is reduced to 1,601. This step is critical because routing does not benefit from overly fine resolutions; excessively fragmented tags only introduce noise and sparsity.

Third, model capability profiles are statistically constructed based on tags. The system compiles the win, tie, and loss counts of each model for each tag into a score dictionary. During online routing, there is no need to repeatedly run multiple candidate models; it only requires generating tags, performing embedding alignment, looking up and summing scores in the table, and making threshold-based decisions.

### Key Designs

1. **TagGenerator**
	- Function: Convert raw text queries into a set of tags that summarize semantic features.
	- Mechanism: First use a strong model to generate teacher tags, then compress this capability into a small model via knowledge distillation. The training data format is $(q, \mathcal{T}(q))$.
	- Design Motivation: If raw queries are sent directly to the router, textual redundancy, expressive diversity, and stylistic differences will interfere with the decision; tags, by contrast, serve as routing-oriented abstract features.

The authors also designed a Hybrid Weight-Based Data Sampling algorithm to balance high-frequency tags and low-frequency but important tags when training the TagGenerator. This point indicates that they do not treat tagging as a standard seq2seq task, but instead genuinely consider the requirements of subsequent routing for "tag coverage" and "tag discriminative power."

2. **Tag Normalization and Tag Alignment**
	- Function: Unify generated tags into a stable tag space to reduce mismatches caused by synonyms and surface differences.
	- Mechanism: Part of this is done offline via normalization and clustering-based merging; the other part is done online by performing embedding similarity matching between generated tags and canonical tags to map them to a unified tag space.
	- Design Motivation: If an out-of-vocabulary tag is generated from a query, or if it differs from canonical tags only in surface expression, direct table lookup will fail, necessitating a semantic alignment layer.

This is precisely what makes TagRouter more reliable than "keyword routing." Instead of exact string matching, it uses semantic alignment to map tags into the existing capability vocabulary, making it more robust on new queries.

3. **TagScorer**
	- Function: Estimate the alignment level of different models for the current query.
	- Mechanism: For each tag $t$ and model $M_i$, calculate a score $\text{score}(M_i,t)$, which is determined by the win, tie, and loss statistics of the model on samples tagged with $t$.
	- Design Motivation: The authors do not want to train a black-box scorer. Instead, they hope to explicitly store "which semantic tags a model excels at" as key-value pairs, which provides better interpretability and facilitates incremental updates.

The core formula presented in the paper can be summarized as:

$$
\text{score}(M_i,t)=w_t \sum_{r\in\{\text{win},\text{tie},\text{loss}\}} \text{count}_{t,M_i}(r)\cdot s_r
$$

Here, $w_t$ is the tag weight, which is related to the tag frequency; $s_r$ is the reward coefficient corresponding to win, tie, or loss. Intuitively, if a model frequently wins or ties with a larger model on a specific tag, its score on this tag will be higher.

One detail is worth noting: the authors tune the contribution of ties separately rather than simply treating them as equivalent to wins. This decision is very reasonable because, in routing routing, being able to "tie with a larger model but at a lower cost" is highly valuable, but it cannot be treated as fully equivalent to a clear win.

4. **TagDecider**
	- Function: Select the final routing target model based on tag scores and cost thresholds.
	- Mechanism: First sum all tag scores for each candidate model to find the one with the highest total score, and then use the threshold $\theta$ to control whether to bias towards a cheaper, smaller model.
	- Design Motivation: Practical deployment is concerned with both performance and cost. Simply pursuing optimal performance makes the system overly dependent on large models, defeating the purpose of routing.

The paper formulates cost control as a clear rule: when the system would originally route to a large model, it compares the cumulative score difference $\Delta_q$ between the smaller and larger models. If the gap is not large enough to necessitate the large model, the query can be handed over to the cheaper small model. By default, the authors consider $\theta=0$ to be a stable setting; to bias further towards cost reduction, the threshold can be adjusted lower.

### Loss & Training

Strictly speaking, TagRouter itself is not an end-to-end trained neural router, so it does not have a unified "overall loss function." Training primarily occurs during the development of the TagGenerator, while other modules rely more on statistical construction and rule-based decisions.

The training strategy of TagGenerator consists of three parts:

- First use a strong model to generate teacher tags and construct a distillation dataset.
- Use hybrid weighted sampling to increase the occurrence probability of low-frequency, important tags.
- Perform instruction tuning on a small model to compress the tagging capability into Qwen2.5-0.5B.

There are no gradient updates during the online routing phase; only three actions are performed: tag generation, tag alignment, and table-lookup scoring. For this reason, the authors refer to it as a training-free routing method. More precisely, it is "training-free during the routing decision phase," rather than having no training process whatsoever in the entire system.

## Key Experimental Results

The core benchmark of this paper is BCUQ. This dataset comes from real user queries on Baidu Cloud's ERNIE Bot platform, containing 95,559 samples. These are split as described in the paper into 93,669 training samples, 1,000 validation samples, and 890 test samples, covering eight types of tasks: brainstorming, classification, closed QA, open QA, content creation, rewrite, summarization, and others.

The authors mainly investigate two points: first, whether routing can indeed make the system outperform any single large model; second, whether introducing the semantic abstraction layer of tags can outperform traditional query-level routing.

### Main Results

First, look at the most critical main results on BCUQ. The candidate models are ERNIE-3.5-8K and ERNIE-Speed-8K, where EB3.5 is both the highest-performing and the most expensive model.

| Method | AR(%) | Uplift(%) | Cost | Rank | AUC(%) | PAUC(%) |
|------|------:|----------:|-----:|-----:|-------:|--------:|
| EBspeed | 59.78 | -24.10 | 2.01 | 1.400 | - | 0 |
| EB3.5 | 78.76 | 0.00 | 13.49 | 1.212 | - | 0 |
| RouteLLM-MF | 80.34 | 2.01 | 11.82 | 1.197 | 73.94 | 0.12 |
| RouterBench-KNN | 80.45 | 2.15 | 11.77 | 1.196 | 75.15 | 0.40 |
| FORC | 81.80 | 3.86 | 11.81 | 1.182 | 75.73 | 0.76 |
| RouteLLM-MF + TagGenerator | 82.02 | 4.14 | 11.66 | 1.180 | 76.08 | 0.76 |
| FORC + TagGenerator | 81.91 | 4.00 | 11.79 | 1.181 | 75.97 | 0.59 |
| **TagRouter** | **83.60** | **6.15** | **11.17** | **1.164** | **76.10** | **1.46** |

There are three primary conclusions from this table.

First, routing is indeed effective. Both traditional methods and TagRouter can push overall system performance beyond that of directly calling EB3.5.

Second, tags are indeed effective. After attaching TagGenerator to existing routing methods, RouteLLM-MF, RouterBench-KNN, and FORC all show improvements, indicating that tags help the router capture effective features better than raw queries.

Third, TagRouter excels in both performance and cost. It increases AR to 83.60%, representing a 6.15% relative improvement over the large model baseline, while reducing the cost to 11.17, which is an approximately 17.2% reduction compared to EB3.5. This demonstrates that it does not simply swap more large model calls for performance, but genuinely improves the capacity to "use smaller models when appropriate."

The authors also performed cross-dataset and cross-model-group validation. Whether on GLM4-9B + Qwen2.5-7B (a pair of models with closer capabilities) or on three different datasets (Alpaca, Dolly, and BCUQ), the average AUC of TagRouter remains higher than the first three baselines. This indicates that it is not only effective on model pairs with massive capability gaps.

### Ablation Study

The ablation studies in the paper do not simply remove a single module to observe performance drops; instead, they validate whether the designs of TagGenerator, TagScorer, and TagDecider are necessary from multiple perspectives. The table below highlights the most representative analysis results from the paper.

| Analysis Item | Configuration | Key Results | Description |
|------|------|---------|------|
| Training Data Scale | BCUQ 50,000 samples | AR 83.26, AUC 75.90, PAUC 1.30 | Performance drops slightly after reducing data, but not significantly |
| Training Data Scale | BCUQ 70,000 samples | AR 83.48, AUC 76.00, PAUC 1.40 | Close to full-data performance |
| Training Data Scale | BCUQ 93,669 samples | AR 83.60, AUC 76.10, PAUC 1.46 | Best with full data, but returns diminish marginally |
| TagGenerator Base Model | Qwen2.5-0.5B | F1 57.75, Inter Rate 0.8686, AUC 76.10 | Finally adopted; lowest cost with strong performance |
| TagGenerator Base Model | Qwen2.5-7B | F1 58.15, Inter Rate 0.8918, AUC 77.48 | Slightly stronger with a larger model, but less lightweight |
| TagGenerator Base Model | Llama3.2-3B | F1 58.03, Inter Rate 0.8969, AUC 77.26 | Good performance, but not the optimal efficiency point |
| TagDecider Threshold | $\theta=0$ | AR 82.47, AUC ~75.95 on BCUQ | Default value is already very stable, suitable for direct deployment |
| TagDecider Threshold | $\theta=\theta^*$ | AR 83.60, AUC 76.10 on BCUQ | Better after further tuning, but requires data support |

Outside of the numbers in the table, the paper also presents several qualitative yet critical conclusions:

- Hybrid weighted sampling is effective, showing that low-frequency tags are crucial for routing and cannot be omitted in favor of high-frequency tags alone.
- Both tag normalization and tag alignment yield performance gains, supporting that "tag space unification" is the foundation of such methods.
- The optimal weight for ties is not 1, but around 0.15, indicating that "ties" should be rewarded, but should not be treated as completely equivalent to "wins."

### Key Findings

- Smaller models are not universally worse; rather, they complement larger models across different samples and task types, which provides a genuine margin of benefit for routing.
- Tag-based representation is one of the core contributions of Ours, as it improves not only TagRouter itself but also other existing routers.
- TagRouter outperforms baselines on most task types, with its advantage being less pronounced only on tasks with highly fixed patterns like closed QA.
- When the model pool scales from 2 to 3 and then 5, the AUC increases from 0.7610 to 0.7933 and then 0.8043, showing that the method indeed benefits as the model ecosystem expands.
- TagRouter still works even when candidate models have similar capabilities, without depending on a strict "one strong, one weak" setup.

## Highlights & Insights

- The greatest highlight of this paper is not building another complex neural router, but decomposing the routing problem into three more stable sub-problems: tag generation, capability statistics, and cost decision-making. This has high engineering value.
- It shifts routing from "direct classification on queries" to "matching in semantic tag space." This is equivalent to performing a routing-oriented feature abstraction beforehand, reducing interference caused by surface-level expression variations.
- TagScorer uses explicit key-value pairs to record model capabilities rather than learning an end-to-end black box. This makes the system easier to scale incrementally; when a new model is introduced, only a small number of labeled samples and tag statistics need to be updated, avoiding full retraining.
- The paper's handling of cost is highly pragmatic. While many routing works focus strictly on performance, this work explicitly integrates the cost threshold $\theta$ into the decision logic, making it closer to real-world product systems.
- An inspiring takeaway is that tags are not just intermediate representations of TagRouter, but also an "enhanced feature layer" that can be transferred to other routing methods. This suggests that the paper proposes not just a single algorithm, but a more general perspective on model routing.

## Limitations & Future Work

- The authors acknowledge that the current TagGenerator primarily covers Chinese and English because BCUQ is dominated by these two languages, leading to limited multilingual generalization capability.
- The evaluation relies heavily on LLM-as-a-judge. Although the authors validated high agreement with human evaluation on 50 samples, this scale is still relatively small, making the conclusion feel "acceptable" rather than "completely robust."
- The current TagScorer still uses a single largest model as the reference model, making the comparison framework relatively pairwise-oriented. If the model pool scales in the future, a more nuanced multi-model relative scoring mechanism, such as an Elo-style global ranking, may be needed.
- Tag quality heavily influences the final routing effect; thus, if TagGenerator experiences systematic bias in a new domain, subsequent table lookups and decisions will be amplified.
- Although the paper demonstrates scalability to more models, whether tag conflict, tag sparsity, and lookup noise will accumulate under a massive model pool is not fully explored in the main text.
- A promising direction for future work is extending tag-based routing to multimodal or agent scenarios. For example, encoding tool requirements, context lengths, or reasoning depth as tags could form a more robust, unified routing interface.

## Related Work & Insights

- **vs FrugalGPT**: FrugalGPT leans towards a cascade style of trying sequentially until a threshold is met, requiring multiple model invocations. In contrast, TagRouter makes selections before inference, making latency and cost more controllable.
- **vs RouteLLM**: RouteLLM is essentially a supervised selection from queries to models, which suffers from poor adaptability when the model pool changes. TagRouter leverages a tag intermediate layer to make scaling candidate models more natural.
- **vs FORC / RouterBench**: These methods already perform pre-inference routing, but their inputs are closer to the raw query representation. This work (Ours) demonstrates that converting inputs to tags first can directly enhance these existing methods.
- **Difference from mixture-of-experts (MoE)**: MoE achieves expert routing inside the model, whereas TagRouter performs model routing at the system level. The two can be combined, where the former optimizes computation within a single model and the latter optimizes external service orchestration.
- **My Own Insights**: When building multi-model serving systems in the future, it is not always necessary to deploy a complex, training-based router first. Establishing a stable task tag system followed by explicit capability modeling is often more controllable and much easier to maintain online.

## Rating

- Novelty: ⭐⭐⭐⭐ It is not a completely new routing macro-framework, but combining tag generation, capability key-value modeling, and cost thresholds yields a highly distinctive scheme.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ It includes real user datasets, cross-task analysis, cross-model-group validation, scalability experiments, and various ablations, presenting a relatively complete chain of evidence.
- Writing Quality: ⭐⭐⭐⭐ The method is structured clearly, and the engineering motivation is well articulated. The drawback is that some tables in the appendix read more like engineering reports, with slightly fragmented details.
- Value: ⭐⭐⭐⭐⭐ Highly valuable as a reference for real-world LLM service orchestration, especially in scenarios that are budget-sensitive or have frequently changing model pools.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Multi-document Summarization through Multi-document Event Relation Graph Reasoning in LLMs](event_graph_bias_mitigation_summarization.md)
- [\[ACL 2025\] Towards Better Open-Ended Text Generation: A Multicriteria Evaluation Framework](towards_better_open-ended_text_generation_a_multicriteria_evaluation_framework.md)
- [\[ACL 2025\] Odysseus Navigates the Sirens' Song: Dynamic Focus Decoding for Factual and Diverse Open-Ended Text Generation](odysseus_dynamic_focus_decoding.md)
- [\[ACL 2025\] Balancing Diversity and Risk in LLM Sampling: How to Select Your Method and Parameter for Open-Ended Text Generation](balancing_diversity_and_risk_in_llm_sampling_how_to_select_your_method_and_param.md)
- [\[ACL 2025\] ATGen: A Framework for Active Text Generation](atgen_a_framework_for_active_text_generation.md)

</div>

<!-- RELATED:END -->

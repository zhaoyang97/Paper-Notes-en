---
title: >-
  [Paper Note] Whom to Query for What: Adaptive Group Elicitation via Multi-Turn LLM Interactions
description: >-
  [ICML 2026][Graph Learning][Adaptive elicitation] This paper extends multi-turn questionnaire-based elicitation from "what to ask" to a joint decision of "whom to ask" and "what to ask." It utilizes an LLM to estimate th…
tags:
  - "ICML 2026"
  - "Graph Learning"
  - "Adaptive elicitation"
  - "Group preferences"
  - "Heterogeneous graph neural networks"
  - "Information gain"
  - "Missing response imputation"
date: 2026-05-08
content_hash: 2df67e922cb3f9ca
---

# Whom to Query for What: Adaptive Group Elicitation via Multi-Turn LLM Interactions

**Conference**: ICML 2026  
**arXiv**: [2602.14279](https://arxiv.org/abs/2602.14279)  
**Code**: https://github.com/ZDCSlab/Group-Adaptive-Elicitation  
**Area**: Graph Learning / LLM Interactive Group Modeling  
**Keywords**: Adaptive elicitation, Group preferences, Heterogeneous graph neural networks, Information gain, Missing response imputation  

## TL;DR
This paper extends multi-turn questionnaire-based elicitation from "what to ask" to a joint decision of "whom to ask" and "what to ask." It utilizes an LLM to estimate the information gain of questions and a heterogeneous GNN to propagate and impute missing responses on a group relationship graph, thereby recovering group preferences faster under a limited respondent budget.

## Background & Motivation
**Background**: Adaptive elicitation typically treats the objective as a step-by-step inquiry of latent variables, such as determining the most informative next question based on answers from previous rounds. LLMs make this process more natural as they can process natural language questions, predict subsequent answers based on historical responses, and measure question value using predictive entropy or expected information gain.

**Limitations of Prior Work**: The bottleneck in real-world group surveys often lies not just in the number of questions, but in how many people can be reached in each round. Existing methods mostly assume a fixed set of respondents and only optimize question selection; even when using LLMs to impute unobserved responses, they often treat individuals independently without leveraging the structure of demographic attributes, group similarities, and partial observations.

**Key Challenge**: Under a limited budget, a question is only truly useful when asked of the right person. Selecting only high-information questions wastes budget on easily predictable or redundant individuals; whereas performing only group imputation might over-propagate uncertain signals, particularly harming highly sensitive individuals who are difficult to explain via demographic attributes.

**Goal**: The authors aim to formalize adaptive group elicitation: simultaneously selecting a question and a small subset of respondents in each round, so that observed answers can both update the questioned individuals themselves and help predict the answers of unqueried individuals on target questions through the group structure.

**Key Insight**: The paper decouples the natural language prediction capability of LLMs from the group propagation capability of heterogeneous graphs. The LLM is responsible for evaluating "how much this question can reduce the uncertainty of future target questions," while the GNN is responsible for propagating observed answers among nodes representing respondents, demographic attributes, and "question-option" pairs.

**Core Idea**: Use an LLM for question-level Expected Information Gain (EIG) selection and a heterogeneous GNN for respondent selection and missing response imputation, coupling "what to ask" and "whom to ask" into a closed loop.

## Method
The proposed method can be understood as a multi-turn closed-loop system. During the training phase, two modules are learned: an LLM capable of predicting the next answer distribution based on an individual's historical answers, and a GNN capable of performing link prediction on a heterogeneous group graph. During the testing phase with a new group, the system first uses the LLM to score candidate questions each round, then uses individual embeddings from the GNN to select representative respondents. After receiving a small number of real answers, the GNN imputes answers for unqueried individuals and writes both real and imputed answers back to the history for the next round.

### Overall Architecture
The input includes a new group, a set of candidate questions, a set of target evaluation questions, and a respondent budget for each round. The system outputs the predicted answers of group members on target questions. Each round consists of four steps: first, estimate the group-level information gain for each candidate question based on existing history; second, use clusterings of member embeddings from the heterogeneous GNN to select the most representative respondents within the budget; third, collect real answers from these individuals; and finally, add new edges to the heterogeneous graph, update member embeddings through message passing, and impute unqueried answers.

In this design, the roles of the LLM and GNN are complementary. The LLM does not need to explicitly parameterize complex latent variables like political attitudes or economic preferences; it only needs to predict future answers from interaction history. The GNN does not need to generate natural language; it only needs to utilize "member-attribute" and "member-question-option" relationships for structural propagation to generalize sparse observations to the entire population.

### Key Designs
1.  **LLM-based Group Expected Information Gain**:
    -   **Function**: Estimates how much a candidate question can reduce the uncertainty of group latent variables under the current history.
    -   **Mechanism**: The authors adopt a de Finetti predictive perspective, transforming latent variable uncertainty into the conditional entropy of future target question answers. For member $v$, uncertainty is written as $H(U_v|\mathcal{H}_{t-1}^v)=\sum_{x\in\mathcal{X}_h}H(Y_x^v|X=x,\mathcal{H}_{t-1}^v)$. The EIG of a candidate question compares the entropy reduction before and after simulated questioning, summed across all members.
    -   **Design Motivation**: This avoids manually defining prior distributions for group preferences, allowing the LLM to learn predictive distributions directly from natural language interaction histories while maintaining a clear information-theoretic objective.

2.  **Heterogeneous GNN Response Imputation and Member Representation Update**:
    -   **Function**: Predicts the responses of unqueried members to the current question when only a subset is queried, and continuously updates group structure representations.
    -   **Mechanism**: The graph contains three types of nodes: member nodes, demographic attribute nodes, and question-option nodes. Members connect to their own attribute nodes and to selected question-options. The GNN learns $p(c|v,q)=\frac{\exp(\langle h_v,h_c\rangle/\tau)}{\sum_{c'}\exp(\langle h_v,h_{c'}\rangle/\tau)}$ via link prediction. During training, member-option edges are randomly masked, and masked answers are recovered using cross-entropy.
    -   **Design Motivation**: Independent imputation by LLMs tends to treat weak signals as certain answers, whereas graph propagation explicitly incorporates demographic attributes and similar answer patterns, providing structural support for unobserved responses.

3.  **Graph Embedding-driven Respondent Selection**:
    -   **Function**: Decides "whom to query" under a limited budget each round, rather than querying respondents randomly or fixedly.
    -   **Mechanism**: The system uses the GNN to obtain final embeddings for each member, assuming members with similar embeddings have similar response patterns. Given a budget $k$, it performs clustering in the embedding space and selects cluster centers as representative respondents. After collecting real answers, new edges are written into the graph for the next round of propagation and selection.
    -   **Design Motivation**: This ensures the observation budget prioritizes high-information regions of the group, particularly aiding highly sensitive individuals who are difficult to impute via attributes alone, rather than wasting budget on those easily inferred from neighbors.

### Loss & Training
The LLM is trained via an autoregressive prediction objective: maximizing the likelihood of the next answer in a member's history, i.e., learning $p_\theta(Y_{t+1}^v|X_{t+1}^v,\mathcal{H}_t^v)$. The GNN uses masked link prediction: randomly masking a portion of member-question-option edges, obtaining member and option embeddings via R-GCN message passing on the partially observed graph, and recovering masked options using softmax link prediction.

In experiments, the LLM query strategy uses Llama-3.1-8B + LoRA for meta-training (trained on the South US region, tested on the West). The GNN uses a two-layer R-GCN with a hidden dimension of 64, evaluated in a cold-start setting where all user-question edges are removed from the message-passing graph at test time.

## Key Experimental Results

### Main Results
The paper evaluates on three real-opinion datasets: CES, OpinionQA, and Twin-2k. One question is asked per round, with a restricted percentage of respondents per round. The goal is to predict all members' answers on held-out target questions. The primary metric is accuracy, with Brier Score and Perplexity as additional metrics.

| Dataset / Setting | Metric | Ours | Strongest Baseline | Gain |
| :--- | :--- | :--- | :--- | :--- |
| CES, 10% respondent budget, round 1 | Relative Target Acc improvement | Highest | Better of Meta-Greedy / Meta-Greedy-Imp | +17.1% relative |
| CES, 10% respondent budget, round 4 | Relative Target Acc improvement | Highest | Strongest baseline | +12.6% relative |
| CES / OpinionQA / Twin-2k, 10%-50% budget | Accuracy curve | Overall highest | Meta-Random, Meta-Greedy, Meta-Greedy-Imp | Consistent lead |
| CES / OpinionQA Calibration | Brier Score / Perplexity | Lowest after round 1 | Meta-Greedy-Imp often overconfident | Better calibration |

### Ablation Study
The ablation focuses on two areas: checking if respondent selection is more valuable for high-sensitivity individuals and comparing greedy querying with multi-step planning. Key figures from round 4 are extracted below.

| Configuration | Key Metric | Description |
| :--- | :--- | :--- |
| CES, 50% budget, Random selection | Global 0.793 / Hard 0.714 / Extreme 0.720 | Random selection benefits from more observations, but underperforms on high-sensitivity individuals. |
| CES, 50% budget, Group-relational selection | Global 0.801 / Hard 0.780 / Extreme 0.826 | Largest gains in hard-to-predict groups, showing "whom to ask" is more critical than just more observations. |
| OpinionQA, 50% budget, Random selection | Global 0.490 / Hard 0.473 / Extreme 0.465 | Multi-option opinion tasks are harder; random observations show limited improvement. |
| OpinionQA, 50% budget, Group-relational selection | Global 0.496 / Hard 0.522 / Extreme 0.529 | Advantages are more pronounced in the most difficult groups. |
| 10% budget, Greedy vs. Multi-step | Global 0.488 vs. 0.485, Extreme 0.322 vs. 0.336 | Multi-step planning only shows slight gains in few sensitive tiers. |
| 50% budget, Greedy vs. Multi-step | Global 0.507 vs. 0.507, Extreme 0.561 vs. 0.542 | No stable benefit from multi-step planning at higher budgets; computational cost is not justified. |

### Key Findings
- Model gains do not come from LLM imputation alone. Meta-Greedy-Imp does not consistently outperform Meta-Greedy, suggesting that direct LLM predictions as missing answers can propagate noise; GNN group structure propagation is a more reliable imputation mechanism.
- Gains from respondent selection are concentrated in high-sensitivity individuals. Under 50% budget in CES, the Extreme tier improved from 0.720 (random) to 0.826, indicating that graph embedding selection effectively directs budget toward those hardest to explain by group means.
- Greedy selection is sufficiently practical. Multi-step rollout shows minor improvements in the Extreme tier at 10% budget but is generally unstable and even reverses at higher budgets; this supports the authors' theoretical discussion on submodular information gain approximations.

## Highlights & Insights
- This paper successfully splits the elicitation budget constraint into two dimensions: question budget and respondent budget. While many adaptive questioning papers only optimize question sequences, real-world surveys often find that "who is willing to answer and who is more valuable to ask" are the primary cost centers.
- The division of labor between LLM and GNN is clear: the LLM handles linguistic prediction and information gain, while the GNN handles relational structure and missing edge recovery. This is more robust than delegating all tasks to an LLM and easier to interpret regarding why gains stem from group propagation.
- The stratified analysis of high-sensitivity respondents is insightful. By decomposing the heterogeneity behind average accuracy, it demonstrates that a good elicitation strategy should prioritize repairing the most difficult-to-impute individuals rather than just improving everyone on average.

## Limitations & Future Work
- The method relies on the availability of demographic attributes or group relationships. If the deployment scenario lacks stable attributes, contains high attribute noise, or has weak correlations between group structure and target preferences, the advantage of GNN imputation may decrease.
- Experiments are primarily offline replay-based evaluations. Real-world interaction issues such as non-response, fatigue, strategic answering, and variations in question wording have not yet been integrated into the closed loop.
- The LLM query strategy requires meta-training, and current experiments mainly use regional transfer settings; future work could investigate generalization across countries, languages, or organizational contexts.
- Respondent selection may introduce fairness issues. If the system constantly queries "high-information" populations, it might increase the burden on certain groups or lead to imbalanced sampling across sensitive attributes.

## Related Work & Insights
- **vs. Individual-level LLM Elicitation**: Existing methods often use LLMs to select the next question based on historical answers. This work extends latent variable inference from individuals to the group level and explicitly adds respondent selection.
- **vs. Traditional Graph Models / CAR Models**: Traditional spatial or group graph models are usually heavily parameterized and struggle with natural language Q&A. This work uses a heterogeneous GNN to represent mixed relationships among members, attributes, and options, making it better suited for survey data.
- **vs. LLM Agent Group Simulation**: Multi-agent social simulations focus on generating interaction processes. This work is more like a budget-constrained statistical inference task, where the core is not simulating dialogue but recovering group response distributions at minimum cost.
- **Transferable Insights**: This paradigm of "LLM for question value estimation + graph model for population coverage estimation" can be transferred to user research, educational diagnostics, medical triage questionnaires, and corporate preference surveys.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Unifying adaptive question selection and respondent selection using LLM + Heterogeneous GNN is a highly realistic problem setting.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Solid testing across three real datasets, budget curves, calibration metrics, and sensitive group ablations, though real-time online interaction validation is missing.
- Writing Quality: ⭐⭐⭐⭐☆ Motivation is clear; theory and experiments correspond well. Some main results rely on curves, with slightly fewer tabulated numbers.
- Value: ⭐⭐⭐⭐☆ Directly relevant for surveys, user modeling, and group decision systems; provides a clean exemplar for LLM and graph learning collaboration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Information-Geometric Adaptive Sampling for Graph Diffusion](information-geometric_adaptive_sampling_for_graph_diffusion.md)
- [\[AAAI 2026\] Relink: Constructing Query-Driven Evidence Graph On-the-Fly for GraphRAG](../../AAAI2026/graph_learning/relink_constructing_query-driven_evidence_graph_on-the-fly_for_graphrag.md)
- [\[NeurIPS 2025\] Heterogeneous Swarms: Jointly Optimizing Model Roles and Weights for Multi-LLM Systems](../../NeurIPS2025/graph_learning/heterogeneous_swarms_jointly_optimizing_model_roles_and_weights_for_multi-llm_sy.md)
- [\[ICML 2026\] MedCoG: Maximizing LLM Inference Density in Medical Reasoning via Meta-Cognitive Regulation](medcog_maximizing_llm_inference_density_in_medical_reasoning_via_meta-cognitive_.md)
- [\[ICML 2026\] GILT: An LLM-Free, Tuning-Free Graph Foundational Model for In-Context Learning](gilt_an_llm-free_tuning-free_graph_foundational_model_for_in-context_learning.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ECCV 2024\] GKGNet: Group K-Nearest Neighbor Based Graph Convolutional Network for Multi-Label Image Recognition](../../ECCV2024/graph_learning/gkgnet_group_k-nearest_neighbor_based_graph_convolutional_network_for_multi-labe.md)
- [\[ICML 2026\] MedCoG: Maximizing LLM Inference Density in Medical Reasoning via Meta-Cognitive Regulation](medcog_maximizing_llm_inference_density_in_medical_reasoning_via_meta-cognitive_.md)
- [\[ICML 2026\] GILT: An LLM-Free, Tuning-Free Graph Foundational Model for In-Context Learning](gilt_an_llm-free_tuning-free_graph_foundational_model_for_in-context_learning.md)
- [\[AAAI 2026\] Relink: Constructing Query-Driven Evidence Graph On-the-Fly for GraphRAG](../../AAAI2026/graph_learning/relink_constructing_query-driven_evidence_graph_on-the-fly_for_graphrag.md)
- [\[ICML 2025\] Is Complex Query Answering Really Complex?](../../ICML2025/graph_learning/is_complex_query_answering_really_complex.md)

</div>

<!-- RELATED:END -->

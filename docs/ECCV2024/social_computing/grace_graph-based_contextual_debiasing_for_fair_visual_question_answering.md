---
title: >-
  [Paper Note] GRACE: Graph-Based Contextual Debiasing for Fair Visual Question Answering
description: >-
  [ECCV 2024][Social Computing][Visual Question Answering] Proposes GRACE (GRAph-based Contextual DEbiasing), a graph-based contextual debiasing method. Through unsupervised context graph learning and graph-based diverse in-context example selection, it addresses the data bias inherited by large language models in knowledge-enhanced VQA systems.
tags:
  - "ECCV 2024"
  - "Social Computing"
  - "Visual Question Answering"
  - "Fairness Debiasing"
  - "Graph Structure Learning"
  - "In-Context Learning"
  - "Knowledge-Enhanced VQA"
date: 2026-05-08
content_hash: b89c68c3e8e1b5a5
---

# GRACE: Graph-Based Contextual Debiasing for Fair Visual Question Answering

**Conference**: ECCV 2024  
**DOI**: [10.1007/978-3-031-72643-9_11](https://doi.org/10.1007/978-3-031-72643-9_11)  
**Code**: [GitHub](https://github.com/SuperJohnZhang/ContextGraphKVQA)  
**Area**: Social Computing  
**Keywords**: Visual Question Answering, Fairness Debiasing, Graph Structure Learning, In-Context Learning, Knowledge-Enhanced VQA

## TL;DR

Proposes GRACE (GRAph-based Contextual DEbiasing), a graph-based contextual debiasing method. Through unsupervised context graph learning and graph-based diverse in-context example selection, it addresses the data bias inherited by large language models in knowledge-enhanced VQA systems.

## Background & Motivation

Large Language Models (LLMs) play an important role in knowledge-based Visual Question Answering (VQA). By conditioning on in-context examples and task-specific prompts, LLMs can comprehensively understand input questions and provide contextually relevant answers. However, this reliance on in-context examples makes LLMs prone to inheriting dataset biases present in both contextual descriptions and examples.

**Bias Source Analysis**:

**Language Prior Bias**: There is an obvious skew in the answer distribution within VQA datasets. For example, for "What sport is this?" type questions, "tennis" might appear with a disproportionately high frequency in the training set, causing the model to learn shortcuts rather than truly understanding the image.

**Vision-Language Association Bias**: Spurious associations exist between certain visual features and specific answers. For instance, seeing a "kitchen" scene biases the model to output "cooking," while ignoring the actual content of the question.

**Social Bias**: Models may generate unfair prediction disparities across different gender, racial, or other demographic groups.

**Bias Amplification by In-context Examples**: When an LLM performs inference using biased examples, the bias is not only inherited but can also be amplified.

Existing debiasing methods primarily focus on the training level (e.g., data augmentation, counterfactual training), with less attention paid to achieving debiasing under the in-context learning paradigm of LLMs. This issue is particularly prominent in knowledge-enhanced VQA, as models need to retrieve external knowledge and examples to assist with inference.

## Method

### Overall Architecture

GRACE consists of two core components, forming an end-to-end debiasing framework:

1. **Unsupervised Context Graph Learning**: Constructs a balanced context graph under fairness constraints.
2. **Graph-Based Diverse Prompt Enhancement**: Leverages the context graph to select semantically relevant and diverse in-context examples.

Overall pipeline: Input VQA sample → Extract visual and textual features → Retrieve diverse examples based on the context graph → Construct debiased prompt → LLM performs inference to generate the answer.

### Key Designs

#### Component 1: Unsupervised Context Graph Learning

The core objective of this component is to build a **balanced context graph** so that examples from different categories can be retrieved and used fairly.

**Graph Construction Process**:

1. **Node Definition**: Each VQA training sample serves as a node in the graph, with node features containing a fused representation of visual features (extracted from an image encoder) and textual features (encoded from the question and answer).
2. **Edge Construction**: Edge connections are established based on the semantic similarity between nodes. Employing a k-nearest neighbors (k-NN) strategy, each node is connected to its k most semantically similar nodes.
3. **Fairness Constraints**: A fairness regularization term is introduced during graph learning to ensure that nodes from different answer categories and different attribute groups (e.g., gender) have similar connection patterns and distribution characteristics in the graph.

**Implementation of Fairness Constraints**:

A fairness loss function is defined to penalize the imbalance of node connections between different groups in the graph. Specifically, by restricting the differences in graph node degree distributions and edge weight distributions across different groups, it prevents examples with high-frequency answer categories from over-clustering and dominating the retrieval process.

$$\mathcal{L}_{fair} = \sum_{g \in G} D_{KL}(P_{deg}^{g} \| P_{deg}^{uniform})$$

where $P_{deg}^{g}$ is the degree distribution of group $g$, and $P_{deg}^{uniform}$ is the uniform distribution.

**Unsupervised Learning**: The learning of the graph structure does not require explicit bias annotations; instead, self-organization is achieved through contrastive learning and clustering constraints, enabling nodes with similar semantics but diverse answers to be connected.

#### Component 2: Graph-Based Diverse Prompt Enhancement

After constructing the context graph, this component leverages the graph structure to select high-quality in-context examples, considering two dimensions:

**Semantic Relevance**:
- Given a query VQA sample, its nearest neighbor nodes in the graph are identified.
- A random walk or message-passing mechanism on the graph is used to explore the sample space semantically related to the query.
- Ensures that the retrieved examples share sufficient semantic commonalities with the query.

**Diversity Constraints**:
- When selecting in-context examples, a diversity penalty is introduced to avoid choosing excessively similar examples.
- Leverages the graph's community structure to sample examples uniformly from different communities/subgraphs.
- Ensures that the selected examples cover diverse answer types and reasoning patterns.

**Prompt Construction**:

The final prompt consists of the following parts:
1. **Task Description**: Defines the basic requirements of the VQA task.
2. **Debiased In-context Examples**: Diverse examples selected via the graph, where each example includes an image caption, question, and answer.
3. **Query Question**: The current VQA sample to be answered.

This graph-based example selection strategy effectively breaks the dominance of high-frequency answers, prompting the LLM to consider more diverse answer possibilities during inference.

#### Inference Process & Knowledge Integration

The inference process of GRACE integrates external knowledge retrieval:

1. **Visual Understanding**: Uses a pre-trained vision model (such as BLIP-2) to comprehend the image and generate an image caption.
2. **Knowledge Retrieval**: Based on the image content and question, relevant knowledge snippets are retrieved from a knowledge base.
3. **Graph-Guided Example Selection**: Retrieves debiased in-context examples from the context graph.
4. **LLM Inference**: Integrates the knowledge, examples, and query into a prompt, which is input to the LLM to generate the final answer.

### Loss & Training

The **overall loss function** consists of three parts:

$$\mathcal{L}_{total} = \mathcal{L}_{task} + \lambda_1 \mathcal{L}_{fair} + \lambda_2 \mathcal{L}_{graph}$$

where:
- $\mathcal{L}_{task}$: VQA task loss, measuring the accuracy of answer prediction.
- $\mathcal{L}_{fair}$: Fairness constraint loss, ensuring the balance of the graph.
- $\mathcal{L}_{graph}$: Graph structure learning loss, including contrastive learning and graph reconstruction objectives.
- $\lambda_1, \lambda_2$: Balancing coefficients.

**Training Pipeline**:
1. Pre-training Phase: Unsupervised learning of the context graph structure.
2. Fine-tuning Phase: Fine-tuning the graph structure and example selection strategy on the specific VQA task.

## Key Experimental Results

### Main Results

GRACE is evaluated on three benchmark datasets:

**Table 1: In-Distribution Evaluation (OK-VQA)**

| Method | Overall Accuracy | Gain After Debiasing |
|------|-----------|-----------|
| Baseline without debiasing | Baseline | - |
| Random Example Selection | Baseline+$ \Delta_1 $ | Limited |
| Similarity Retrieval | Baseline+$ \Delta_2 $ | Moderate |
| **GRACE** | **Optimal** | **Significant Gain** |

**Table 2: Out-of-Distribution Generalization Evaluation**

| Method | VQA-CP (OOD) | GQA-OOD | OK-VQA |
|------|-------------|---------|--------|
| Standard LLM prompting | Heavily affected by bias | Heavily affected by bias | Baseline |
| Counterfactual debiasing | Partially improved | Partially improved | May decrease |
| Data augmentation debiasing | Improved | Improved | Maintained |
| **GRACE** | **Best/Second Best** | **Best/Second Best** | **Maintained or Improved** |

GRACE performs particularly outstandingly on OOD datasets, indicating that graph-based debiasing methods possess better generalization ability—being effective not only within the training distribution but also remaining robust in distribution shift scenarios.

### Ablation Study

**Component Ablation Study**:

| Configuration | Fairness Constraint | Diverse Selection | VQA-CP | OK-VQA |
|------|---------|-----------|--------|--------|
| Baseline | ✗ | ✗ | Baseline | Baseline |
| +Fairness Constraint | ✓ | ✗ | Gain++ | Maintained |
| +Diverse Selection | ✗ | ✓ | Gain+ | Gain+ |
| **GRACE (Full)** | ✓ | ✓ | **Gain+++** | **Gain+** |

The ablation study demonstrates that the two components are complementary: fairness constraints primarily improve OOD generalization capabilities, while diverse selection simultaneously improves both in-distribution and out-of-distribution performance.

**Gender Fairness Analysis**:

| Metric | Baseline Method | GRACE |
|------|---------|-------|
| Accuracy for Male Group | High | Maintained |
| Accuracy for Female Group | Relatively Low | Improved |
| Gender Gap (Gap) | Large | **Significantly Reduced** |
| Equalized Odds | Not satisfied | **Closer to satisfied** |

GRACE performs exceptionally well in reducing the performance gap between gender groups, reflecting its potential to promote social fairness.

### Key Findings

1. **Graph Structures Effectively Model Contextual Relations**: Compared to simple similarity-based retrieval, the graph structure captures richer, high-order semantic relations, making example selection more reasonable.
2. **Fairness Constraints Are Key to OOD Generalization**: In scenarios with distribution shift, the improvement brought by fairness constraints is the most significant, indicating that data bias is a primary reason for OOD performance degradation.
3. **Diverse Examples Enhance Inference Quality**: Examples sampled from different communities offer more comprehensive reasoning perspectives for the LLM, reducing the monotony of answers.
4. **Debiasing and Performance Can Achieve a Win-Win**: While reducing bias, GRACE typically does not degrade and often improves overall accuracy, breaking the traditional conception of a "fairness-accuracy trade-off."

## Highlights & Insights

1. **Novelty of Research Perspective**: This study is the first to systematically investigate the bias problem of VQA under the LLM's in-context learning paradigm, filling the gap of debiasing methods in ICL settings.
2. **Innovative Use of Graphs as a Debiasing Tool**: Utilizing structural properties of graphs (communities, degree distribution, connectivity) to implement fairness constraints is more elegant and effective than simple statistical methods.
3. **Unsupervised Debiasing Framework**: It does not require explicit bias annotations; debiasing is automatically achieved by constructing a balanced graph structure, offering better generalizability.
4. **Balancing Effectiveness and Ethics**: Incorporating social fairness (such as gender fairness) into VQA evaluation dimensions promotes attention to ethical issues within this field.
5. **Plug-and-Play Design**: The two components (graph learning + diverse selection) possess excellent modular characteristics and can be combined with different LLM and VQA frameworks.

## Limitations & Future Work

1. **Graph Construction Overhead**: Constructing and maintaining context graphs on large-scale datasets may entail high computational and storage overhead.
2. **Limitations in Fairness Definition**: The focus is currently on the gender dimension of fairness. Future iterations can be expanded to more sensitive attributes, such as race and age.
3. **Dynamic Context Graphs**: The current graph structure is fixed after training. Future work can explore graph structures that update dynamically during the inference process.
4. **In-Depth Analysis of Multimodal Bias**: Debiasing is only provided at the language level; biases residing within the visual encoder itself may require additional processing.
5. **Verification on Larger-Scale LLMs**: Current experiments are primarily based on specific LLMs; effectiveness on larger-scale models (such as GPT-4, LLaMA-65B) remains to be validated.
6. **Integration with Causal Inference Methods**: The current method leans toward statistical debiasing. Future work can combine it with causal inference frameworks to eliminate spurious correlations from their root causes.

## Related Work & Insights

- **VQA Debiasing Directions**: Compared to methods such as CSS (Counterfactual Samples Synthesis), LMH (Learn-to-Mix-Heristic), and D-VQA (Causal Debiasing), the advantage of GRACE lies in not requiring modifications to the model training process, instead achieving debiasing during inference through example selection.
- **Application of Graph Neural Networks in VQA**: Previously, graphs were primarily used to model scene graphs, question graphs, or knowledge graphs. GRACE innovatively utilizes graphs to model "relationships between training samples" to assist in debiasing.
- **In-Context Learning Studies**: GRACE provides a new direction for example selection strategies in ICL—considering not only relevance but also fairness and diversity.
- **Insights**:
  1. Bias problems exist not only within model parameters but also in the contextual organization of the inference process.
  2. Graph structures are natural tools for handling fairness constraints and can be generalized to other scenarios requiring balancing.
  3. Future prompt engineering should incorporate fairness as a design consideration.

## Rating

| Dimension | Score (1-5) | Explanation |
|------|-----------|------|
| Novelty | 4.0 | Introducing graph learning into ICL debiasing is a novel approach with strong problem orientation. |
| Technical Depth | 3.5 | The framework design is reasonable, but the theoretical depth of the graph learning portion could be further strengthened. |
| Experimental Thoroughness | 4.0 | Comprehensive coverage of ID/OOD datasets, including fairness analysis. |
| Writing Quality | 3.5 | The methodology description is clear, though some details require referring to the code. |
| **Overall** | **3.5** | Combining fairness issues with ICL architectures represents a meaningful intersectional research direction. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] GG-BBQ: German Gender Bias Benchmark for Question Answering](../../ACL2025/social_computing/gg-bbq_german_gender_bias_benchmark_for_question_answering.md)
- [\[ICLR 2026\] SAGE: Spatial-visual Adaptive Graph Exploration for Efficient Visual Place Recognition](../../ICLR2026/social_computing/sage_spatial-visual_adaptive_graph_exploration_for_efficient_visual_place_recogn.md)
- [\[ACL 2026\] Bayesian Social Deduction with Graph-Informed Language Models](../../ACL2026/social_computing/bayesian_social_deduction_with_graph-informed_language_models.md)
- [\[ICLR 2026\] Adaptive Debiasing Tsallis Entropy for Test-Time Adaptation](../../ICLR2026/social_computing/adaptive_debiasing_tsallis_entropy_for_test-time_adaptation.md)
- [\[NeurIPS 2025\] GraphKeeper: Graph Domain-Incremental Learning via Knowledge Disentanglement and Preservation](../../NeurIPS2025/social_computing/graphkeeper_graph_domain-incremental_learning_via_knowledge_disentanglement_and_.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Statistical Deficiency for Task Inclusion Estimation
description: >-
  [ACL 2025 (Long Paper, acl-long.18)][Task Inclusion] Based on the theory of statistical deficiency, this paper proposes a theory-driven framework for defining and measuring task inclusion relations. Using information sufficiency (IS) as a computable proxy metric, the framework estimates the degree of inclusion between tasks by comparing the intermediate layer representations of fine-tuned models. It successfully reconstructs the hierarchical relationships of a classic NLP pip…
tags:
  - "ACL 2025 (Long Paper, acl-long.18)"
  - "Task Inclusion"
  - "Statistical Deficiency"
  - "Information Sufficiency"
  - "Task Relationship"
  - "NLP Pipeline"
  - "Mutual Information"
date: 2026-05-08
content_hash: 9ee17f9630780411
---

# Statistical Deficiency for Task Inclusion Estimation

**Conference**: ACL 2025 (Long Paper, acl-long.18)  
**arXiv**: [2503.05491](https://arxiv.org/abs/2503.05491)  
**Code**: None  
**Area**: Task Relationship Modeling / Information Theory / NLP Pipeline  
**Keywords**: Task Inclusion, Statistical Deficiency, Information Sufficiency, Task Relationship, NLP Pipeline, Mutual Information  

## TL;DR
Based on the theory of statistical deficiency, this paper proposes a theory-driven framework for defining and measuring task inclusion relations. Using information sufficiency (IS) as a computable proxy metric, the framework estimates the degree of inclusion between tasks by comparing the intermediate layer representations of fine-tuned models. It successfully reconstructs the hierarchical relationships of a classic NLP pipeline on both synthetic data and real-world NLP tasks.

## Background & Motivation
In machine learning, a task is the most natural unit for evaluating model capabilities. With the rise of instruction-tuned models, the addressable task space has expanded drastically, yet theoretical tools to study the internal structure of this task space remain scarce. Existing approaches, such as task similarity, are symmetric metrics that fail to capture asymmetric relationships like "Task A includes Task B". Task transfer methods rely on parameter space analysis, which suffers from extremely high dimensionality and poor interpretability. Probing methods suffer from interpretability issues (the limited expressiveness of linear probes can lead to misjudgments).

Intuitively, certain tasks serve as prerequisites for others (e.g., NER is a necessary skill for summary generation), but this inclusion relationship has long lacked a rigorous mathematical definition and a reliable computational method.

## Core Problem
How to formally define the asymmetric inclusion relationship between tasks and provide a computable metric to estimate it?

## Method

### Overall Architecture
1. **Formalizing tasks**: A task is defined as a joint probability measure $\mathbb{P}_{XY}$ (the joint distribution of input $X$ and response $Y$).
2. **Defining Lenient Inclusion**: If estimating $\mathbb{P}_{Y_U|X}$ is informative for estimating $\mathbb{P}_{Y_V|X}$, task $V$ is said to be included in task $U$ (denoted as $V \tilde{\subset} U$).
3. **Quantifying inclusion with statistical deficiency**: Deficiency $\delta$ measures whether the embedding of one task can "simulate" the embedding of another task—the smaller the value, the higher the degree of inclusion.
4. **Using Information Sufficiency as a computable proxy**: Since deficiency (based on TV distance) is computationally intractable, IS (based on the lower bound of mutual information) is used as a surrogate.

Core reasoning chain: $\mathcal{IS}(Z_V \to Z_U) \leq \mathcal{IS}(Z_U \to Z_V) \Rightarrow V \tilde{\subset} U$

### Key Designs

1. **Task Definitions and Assumptions**

    - Assumption H1: All tasks reside in the same space $(\mathcal{X} \times \mathcal{Y})$ (which holds true under the text-to-text generative paradigm).
    - Assumption H2: All tasks share the same marginal input distribution $\mathbb{P}_X$, ensuring the comparison focuses on skill differences $\mathbb{P}_{Y|X}$ rather than domain differences.

2. **From Deficiency to Information Sufficiency**

    - Deficiency Definition (Le Cam, 1964): $\delta(\mathbb{P}_{Z_U|Y_V} \to \mathbb{P}_{Z_V|Y_V}) = \inf_{M} \|M \circ \mathbb{P}_{Z_U|Y_V} - \mathbb{P}_{Z_V|Y_V}\|_{TV}$
    - 0-Deficiency Theorem: $\delta = 0$ implies task inclusion.
    - $\varepsilon$-Deficiency Theorem: The smaller the deficiency, the smaller the gap in risk between using $Z_U$ and using $Z_V$ to infer $Y_V$ for any bounded loss function.
    - IS Proxy: $\mathcal{IS}(Z_U \to Z_V) = \hat{h}(Z_V) - \hat{h}(Z_V|Z_U)$, computed using the KNIFE estimator (under a family of Gaussian mixture models).

3. **Layer Selection Strategy**

    - By comparing the IS between the fine-tuned model and the pre-trained model, the IS gap is found to be largest in layers 10-15 (i.e., these layers encode the most task-specific information).
    - The average IS of layers 10-15 is ultimately used as the measure of task inclusion.
    - Deeper layers (>15) tend to encode output formats rather than task semantics, introducing noise.

4. **Predictive Power Metric**

    - $PP(U) = \sum_V \mathcal{IS}(Z_U \to Z_V) - \mathcal{IS}(Z_V \to Z_U)$
    - A higher PP indicates that task $U$ contains more information about other tasks while being less contained by them.

## Key Experimental Results

### Synthetic Experiments (HMM Data)

Three classification tasks: First (F), Last (L), and First_or_Last (F∨L), with known relationships $F \tilde{\subset} F\vee L$ and $L \tilde{\subset} F\vee L$.

| $\mathcal{IS}$(row→col) | F | F∨L | L |
|---|---|---|---|
| **F** | 0.736 | 0.236 | 0.130 |
| **F∨L** | 0.188 | 0.842 | 0.175 |
| **L** | 0.123 | 0.223 | 0.715 |

IS successfully captures: $\mathcal{IS}(F \to L) \leq \mathcal{IS}(F\vee L \to L)$ and $\mathcal{IS}(L \to F) \leq \mathcal{IS}(F\vee L \to F)$, aligning with expectations.

### NLP Pipeline Experiments

On 5 tasks from the OntoNotes dataset (SYN/SRL/NER/COR/SUM), utilizing Mistral 7B and Llama 3 8B (4 models in total, including Base and Instruct versions) fine-tuned via LoRA.

**Task Performance (RougeL)**:

| Task | Mistral-B | Mistral-I | Llama3-B | Llama3-I |
|---|---|---|---|---|
| SYN | 97.6 | 97.5 | 97.6 | 97.3 |
| SRL | 81.5 | 80.5 | 82.0 | 81.8 |
| NER | 86.7 | 87.8 | 85.0 | 86.3 |
| COR | 53.9 | 61.2 | 53.7 | 61.7 |
| SUM | 48.8 | 49.6 | 49.6 | 48.5 |

**Predictive Power Ranking** (average): SYN(0.75) < SRL(0.75) < NER(1.5) < COR(3.0) < SUM(4.0)

Successfully reconstructed the classical NLP pipeline hierarchy: $SYN \tilde{\subset} SRL \tilde{\subset} NER \tilde{\subset} COR \tilde{\subset} SUM$

### Ablation Study Key Points
- **Layer Selection Ablation**: Layers 10-15 distinguish the NLP pipeline hierarchy best; using all layers or deeper layers (10-33) confuses the order of SRL and NER; layers 1-20 yield results consistent with layers 10-15.
- **IS vs. Naive Cross-Task Evaluation**: Directly evaluating another task using the model from one task (cross-task performance) shows very low Kendall-$\tau$ correlation with IS (0.02-0.43), indicating that naive methods are unreliable due to misaligned output formats.
- **Base vs. Instruct Models**: Base models preserve the pipeline order better, while Instruct models introduce noise since they have already been exposed to a wide range of tasks during pre-training.
- **Comparison with Task Vector Methods**: Grassmann distance and cosine distance can partially reflect task similarity (e.g., SYN-SRL proximity), but are inherently symmetric metrics and cannot discover partial order relationships.

## Highlights

- **Solid Theoretical Foundation**: Starting from Le Cam's statistical deficiency theory, the paper establishes a complete theory-to-practice chain through the definition of lenient inclusion and the derivation of the IS proxy.
- **Innovative Asymmetric Metric**: Unlike traditional symmetric task similarity, IS naturally supports asymmetric comparisons, directly mapping to the directional relationship of "A contains B".
- **Intuitive Empirical Validation**: Successfully reconstructs the NLP pipeline hierarchy (SYN → SRL → NER → COR → SUM) in a data-driven manner, aligning closely with linguistic intuition.
- **Insightful Intermediate Layer Selection**: Comparing the IS of fine-tuned and pre-trained models reveals that layers 10-15 encode the most task-specific information, offering a new perspective for studying internal LLM representations.

## Limitations

- **Indirectness of IS as a Deficiency Proxy**: IS does not account for target variables $Y_U$ and $Y_V$, which are core to the definition of a task; moreover, IS is only a lower bound of mutual information, potentially underestimating the true level of inclusion.
- **Single Dataset and Single Language**: Evaluation is restricted to the OntoNotes English dataset across only 5 tasks, where pipeline tasks were simplified (e.g., SRL only extracted ARG0+ARG1).
- **Limited Model Scale**: Experiments are only conducted on 7B/8B class models (Mistral, Llama 3); the behaviors of larger models remain unknown.
- **Single Adaptation Method**: Only LoRA fine-tuning is employed, without exploring other task adaptation paradigms like zero-shot or in-context learning.

## Related Work & Insights

- **vs. Task Similarity (Achille et al.)**: Symmetric metrics can only capture "similarity" but not "inclusion"; in contrast, the proposed IS is asymmetric.
- **vs. Probing (Conneau et al.)**: Probing evaluates representations using linear probes, which is limited by the probe's expressiveness and reflects alignment rather than information content; in contrast, this work directly measures the informational relationship between embeddings.
- **vs. Task Transfer (Vu et al.)**: Based on parameter space (e.g., Fisher information), these methods are extremely high-dimensional and symmetric; in contrast, this work operates on the activation space, offering lower dimensionality and inherent directionality.
- **vs. Task Vector (Ilharco et al.)**: Defines task vectors in the parameter space and compares them using distance metrics, which is inherently symmetric and incapable of establishing partial orders; the proposed IS is directional.

## Insights & Connections

- **Task Partial Order Structures for Data Mixture Optimization**: The paper suggests direct utility in data selection for instruction tuning—selecting the most informative tasks/instructions to optimize dataset size.
- **Orthogonal Benchmark Design**: Once task inclusion relationships are identified, more orthogonal evaluation benchmarks can be designed to reduce redundancy.
- **Potential Connections to Model Compression/Pruning**: The discovery that layers 10-15 encode core task-specific information could guide layer pruning strategies.
- **From Task Space to Skill Space**: The paper envisions decomposing tasks into a minimal set of non-overlapping skills, which directly relates to the granularity issues in current LLM capabilities evaluation.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Applies Le Cam's 60-year-old statistical deficiency theory for the first time to NLP task relationship modeling, offering a completely fresh theoretical perspective.
- Experimental Thoroughness: ⭐⭐⭐ While the synthetic and NLP pipeline experiments serve as credible proofs of concept, the scale (5 tasks, 2 model families) is small; there is a lack of quantitative comparisons against more baselines.
- Writing Quality: ⭐⭐⭐⭐ The theoretical proof is rigorous, and the appendix is exceptionally thorough (34 pages containing 8 appendices). However, the main text contains high information density and requires background in information theory.
- Value: ⭐⭐⭐⭐ The formalization framework for task relationships is inspiring for understanding multi-task and transfer learning, and the IS metric could be applied to data composition and benchmark design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Infogen: Generating Complex Statistical Infographics from Documents](infogen_generating_complex_statistical_infographics_from_documents.md)
- [\[NeurIPS 2025\] Statistical Inference Under Performativity](../../NeurIPS2025/others/statistical_inference_under_performativity.md)
- [\[NeurIPS 2025\] Statistical Inference for Gradient Boosting Regression](../../NeurIPS2025/others/statistical_inference_for_gradient_boosting_regression.md)
- [\[NeurIPS 2025\] Robust Sampling for Active Statistical Inference](../../NeurIPS2025/others/robust_sampling_for_active_statistical_inference.md)
- [\[NeurIPS 2025\] Exploiting Task Relationships in Continual Learning via Transferability-Aware Task Embeddings](../../NeurIPS2025/others/exploiting_task_relationships_in_continual_learning_via_transferability-aware_ta.md)

</div>

<!-- RELATED:END -->

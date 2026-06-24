---
title: >-
  [Paper Note] SkillVerse: Assessing and Enhancing LLMs with Tree Evaluation
description: >-
  [ACL 2025][LLM (Other)][Fine-grained evaluation] SkillVerse is proposed as an unsupervised, tree-structured LLM diagnostic framework. By organizing evaluation feedback from an LLM-as-Judge into a hierarchical skill tree (dendrogram), it uncovers the strengths and weaknesses of model capabilities at any level of granularity. This is further utilized to select superior few-shot exemplars (improving ICL by up to 25%) and to predict model weaknesses in unseen scenarios (achieving…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Fine-grained evaluation"
  - "Hierarchical clustering"
  - "Dendrogram"
  - "Model capability diagnosis"
  - "In-context learning enhancement"
date: 2026-05-08
content_hash: c8e56aa446a34376
---

# SkillVerse: Assessing and Enhancing LLMs with Tree Evaluation

**Conference**: ACL 2025  
**arXiv**: [2506.00319](https://arxiv.org/abs/2506.00319)  
**Code**: Unreleased  
**Area**: LLM Evaluation  
**Keywords**: Fine-grained evaluation, Hierarchical clustering, Dendrogram, Model capability diagnosis, In-context learning enhancement  

## TL;DR

SkillVerse is proposed as an unsupervised, tree-structured LLM diagnostic framework. By organizing evaluation feedback from an LLM-as-Judge into a hierarchical skill tree (dendrogram), it uncovers the strengths and weaknesses of model capabilities at any level of granularity. This is further utilized to select superior few-shot exemplars (improving ICL by up to 25%) and to predict model weaknesses in unseen scenarios (achieving a 55% success rate, which is 22% higher than the uninformed baseline).

## Background & Motivation

**Background**: Leaderboards and benchmarks (e.g., ChatbotArena, MMLU) have become the primary methods for evaluating LLMs, offering a global view of model rankings. However, their limited interpretability makes it difficult to identify subtle behavioral traits and derive actionable insights.

**Limitations of Prior Work**: (1) Aggregated metrics cannot answer fine-grained questions such as "Does a higher-ranked model perform better in all subdomains?"; (2) Manual analysis of model capabilities is time-consuming and labor-intensive; (3) Existing LLM evaluation methods (e.g., QualEval, BERTopic clustering) require a predefined number of categories or attribute sets, lacking flexibility.

**Key Challenge**: Automated, fine-grained, and flexibly adjustable model capability diagnosis is required. However, existing evaluation frameworks are either too coarse (leaderboard rankings) or too rigid (predefined taxonomies).

**Goal**: To automatically extract a hierarchical skill structure from unstructured model evaluation feedback, analyze model capabilities at any level of granularity, and utilize these insights to improve model performance.

**Key Insight**: Utilizing LLM-as-Judge to generate detailed reviews and parsing these reviews into atomic judgments (indivisible units of capability evaluation). A dendrogram is then constructed via agglomerative hierarchical clustering, allowing evaluations at various granularities by slicing the tree at different horizontal levels.

**Core Idea**: Deconstructing LLM reviews into atomic judgments and clustering them bottom-up into a skill tree to diagnose model capabilities at any level of granularity and guide inference optimization.

## Method

### Overall Architecture

The workflow of SkillVerse consists of:  
1. **Review Collection**: Evaluating model responses with an LLM-as-Judge, enhanced by verifiable rules (e.g., formatting, calculation) to improve evaluation accuracy.  
2. **Parsing into Atomic Judgments**: Decomposing free-text reviews into atomic triplets of "Model A + Success/Failure + Specific Task".  
3. **Vectorization & Agglomerative Hierarchical Clustering**: Embedding the tasks and performs hierarchical clustering to construct a skill dendrogram.  
4. **Slicing**: Cutting the dendrogram at different heights to obtain nested skill clusters.  
5. **Anchoring**: Merging independent dendrograms of different models through anchoring to support fair cross-model comparison.

### Key Designs

1. **Atomic Judgment**
    - **Function**: Standardizing free-text reviews into quantifiable, structured units.
    - **Mechanism**: Enforcing all judgments to follow a triplet syntax: Subject (Model Name) + Verb (Success/Partial Success/Failure) + Object (Specific Task Description). Representation learning and clustering are performed only on the Object component.
    - **Design Motivation**: Free-text reviews are difficult to organize and quantify at scale; the standardized structure enables direct calculation of success rates within the same skill cluster, achieving precise capability measurement.

2. **Agglomerative Hierarchical Clustering & Dendrogram**
    - **Function**: Organizing massive atomic judgments into a skill hierarchy with flexible granularity.
    - **Mechanism**: Performing bottom-up agglomerative clustering based on semantic distances from the Google Text Embedding API. Slicing the tree at different heights yields clusters of varying granularities—the top level might distinguish "STEM" from "Non-STEM", while the bottom level can resolve specific skills like "writing SQL queries" or "formatting bibliographies".
    - **Design Motivation**: Unlike flat clustering with a fixed number of categories, a tree structure naturally supports flexible analysis from coarse to fine grains. Users can slice the hierarchy at any desired level of granularity.

3. **Cross-Model Cluster Anchoring**
    - **Function**: Aligning independent dendrograms generated by different models to support fair comparison.
    - **Mechanism**: Merging two clusters when they simultaneously satisfy (a) centroid cosine similarity $\ge \tau$ and (b) Intersection over Union (IoU) $\ge \varepsilon$. This dual condition ensures that both central distance and distributional overlap are considered.
    - **Design Motivation**: Different models generate different responses, leading to distinct reviews and making their respective dendrograms directly incomparable. The anchoring mechanism allows incrementally adding new models without re-clustering the entire dataset.

### Loss & Training

SkillVerse itself does not involve model training. In downstream applications:
- **ICL Enhancement**: Utilizing the dendrogram to conduct a tree search, pruning branches where the model already excels (success rate $\ge T$), and selecting few-shot examples from difficult branches ranked by content relevance and contrastive utility.
- **Weakness Prediction**: Providing the capability profile generated by SkillVerse to a reasoning LLM (such as GPT-4o) to infer and extrapolate potential new weaknesses.

## Key Experimental Results

### Main Results

Cross-model family fine-grained capability comparison (insights discovered by SkillVerse):

| Comparison Dimension | Claude 3.5 Sonnet | Gemini 1.5 Pro | GPT-4o |
|---|---|---|---|
| Visual Coding | 85.5% | 76.8% | 79.5% |
| Educational Content Development | - | Best | - |
| Inferring Ambiguous User Intent | - | 63.2% | 83.7% |
| Mathematical Proofs | - | - | Best |
| Shell Commands | Best | - | - |

**Inverse scaling phenomenon** (skills where larger models perform worse than smaller ones): including precise formatting constraint tasks such as wrapping responses in double quotes, Markdown formatting, JSON output, and limerick rhythm.

### Ablation Study

SkillVerse-enhanced ICL vs. baseline methods (improvement percentage relative to direct generation):

| Method | IF-Eval (Gemini-flash) | IF-Eval (Gemini-pro) | ChatbotArena (Gemini-flash) |
|---|---|---|---|
| C-ICL (Similarity Selection) | ~10% | ~5% | ~8% |
| Principles Learning | ~15% | ~3% | ~5% |
| **SkillVerse** | **~25%** | **~8%** | **~12%** |

Accuracy of weakness prediction:

| Setting | Average Success Rate | Description |
|---|---|---|
| SkillVerse-informed Prediction | 55% | Guided by capability profile |
| Uninformed Baseline Prediction | 77% | Without capability profile |
| Average Success Rate on Existing Tasks | 69% | Reference baseline |

### Key Findings

1. **GPT-4-turbo still outperforms GPT-4o in certain domains**: e.g., SQL queries (+6.1%), file processing (+9.1%), and music tasks (+2%), indicating that model iterations do not yield all-around improvements.
2. **Inverse scaling phenomenon**: Larger models perform worse on tasks requiring precise formatting constraints (e.g., keyword inclusion/exclusion, strict formatting).
3. **SkillVerse's ICL enhancement outperforms standard C-ICL by 25%**: The key lies in simultaneously considering both semantic relevance and the model's level of difficulty on the specific skill.
4. **Significant gap in weakness prediction success rates**: Predictions informed by SkillVerse achieved a 22% lower success rate compared to uninformed predictions (meaning weaknesses were identified more accurately).
5. **Reliable clustering quality**: Demonstrated a Pearson correlation of 0.643 with human annotations, a true positive rate of 0.916, and a true negative rate of 0.883.

## Highlights & Insights

- **Core advantage of flexible granularity analysis**: The dendrogram can be sliced at any height, enabling multi-level analysis ranging from broad terms like "STEM" to fine-grained tasks like "writing regular expressions" using the same dataset.
- **Practical significance of inverse scaling discovery**: Identifying specific degraded skills in larger models holds direct value for model selection and routing.
- **Closed-loop system**: Forming a complete chain from evaluation $\rightarrow$ diagnosis $\rightarrow$ enhancement $\rightarrow$ prediction, which not only identifies issues but also facilitates action.
- **Strong unsupervised characteristic**: No predefined skill categories are required, allowing semantic structures to completely emerge from the data.

## Limitations & Future Work

- Reliance on LLM-as-Judge for generating reviews, which may introduce biases (especially during self-evaluation).
- Clustering is still based on the cosine distance of text embeddings, potentially grouping tasks that are semantically related but differ significantly in difficulty.
- Weakness prediction relies on an external reasoning model (GPT-4o), and biological/cognitive biases of the reasoning LLM itself can affect prediction quality.
- The sample size for ICL enhancement experiments is relatively small (150 hold-out prompts), and statistical significance warrants validation on a larger scale.
- Scalable directions: Applying SkillVerse to model routing and targeted training data curation.

## Related Work & Insights

- **QualEval**: Allocates predefined attributes to data points, offering less flexibility than SkillVerse's unsupervised hierarchical structure.
- **SkillIndex**: A parallel work that also focuses on skill-level analysis, but likewise requires predefined categories.
- **C-ICL**: Contrastive in-context learning, where SkillVerse provides more precise exemplar selection via molecular/dendrogram searches.
- **Insight**: LLM evaluation should shift from "single-number rankings" to "hierarchical capability profiles."

## Rating

- **Novelty**: 4/5 — Tree-structured flexible-granularity diagnosis is novel.
- **Technical Depth**: 4/5 — Hierarchical clustering, anchoring, and downstream applications form a comprehensive methodology.
- **Experimental Thoroughness**: 4/5 — Verified across three downstream tasks: cross-model comparison, ICL enhancement, and weakness prediction.
- **Value**: 4/5 — Direct practical value for model evaluation, selection, and improvement.
- **Overall Rating**: 4/5

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] ConsistencyChecker: Tree-based Evaluation of LLM Generalization Capabilities](consistencychecker_tree_evaluation.md)
- [\[ACL 2025\] Assessing and Enhancing the Causal Reasoning Abilities of Language Models via Faithful Textual Interpretation](assessing_and_enhancing_the_causal_reasoning_abilities_of_language_models_via_fai.md)
- [\[ACL 2025\] Assessing the Vulnerability of LLMs to Cognitive Biases in Scientific Research](assessing_the_vulnerability_of_llms_to_cognitive_biases_in_scientific_research.md)
- [\[ACL 2025\] ATRIE: Automating Legal Interpretation with LLMs: Retrieval, Generation, and Evaluation](atrie_legal_interpretation.md)
- [\[ACL 2025\] Which of These Best Describes Multiple Choice Evaluation with LLMs?](multiple_choice_eval.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Dolphin: Moving Towards Closed-loop Auto-research through Thinking, Practice, and Feedback
description: >-
  > Proposes Dolphin, a closed-loop auto-research framework that incorporates a three-stage cycle of "idea generation $\rightarrow$ experimental verification $\rightarrow$ results feedback". Through task-attribute-guided paper ranking and exception-traceback-guided debugging processes, Dolphin automatically proposes and verifies methods that approach human-designed SOTA on tasks such as 3D classification.
tags:

date: 2026-05-08
content_hash: b626b57d69a202be
---

# Dolphin: Moving Towards Closed-loop Auto-research through Thinking, Practice, and Feedback

| Information | Content |
|------|------|
| Conference | ACL 2025 |
| arXiv | [2501.03916](https://arxiv.org/abs/2501.03916) |
| Code | [GitHub](https://github.com/Alpha-Innovator/Dolphin) |
| Area | others (Auto-Research × LLM Agent × Code Generation) |
| Keywords | auto-research, closed-loop, idea generation, experimental verification, feedback |

## TL;DR

> Proposes Dolphin, a closed-loop auto-research framework that incorporates a three-stage cycle of "idea generation $\rightarrow$ experimental verification $\rightarrow$ results feedback". Through task-attribute-guided paper ranking and exception-traceback-guided debugging processes, Dolphin automatically proposes and verifies methods that approach human-designed SOTA on tasks such as 3D classification.

## Background & Motivation

- **Scientific Research Paradigm Shift**: AI-assisted scientific research is evolving from "fully human-driven" to "auto-research", progressing through four stages: fully human-driven $\rightarrow$ AI-assisted $\rightarrow$ semi-automatic $\rightarrow$ fully automatic.
- **Key Challenges in Prior Work**:
  1. **Inaccurate Idea Evaluation**: Most works (Si et al., 2024; Li et al., 2024) rely on humans or LLMs to evaluate idea quality, focusing only on novelty rather than experimental validity. Although AI-Scientist (Lu et al., 2024) performs experimental verification, it utilizes simple self-constructed datasets, lacking meaningful comparisons with existing methods in the same field.
  2. **Lack of Feedback Mechanisms**: Human researchers iteratively improve ideas based on experimental results, whereas existing works either offer feedback only within isolated idea-generation phases or completely lack feedback loop.
- **Core Motivation**: To build a true "closed-loop" system where experimental results are fed back into the next round of idea generation, mimicking the iterative research process of human researchers.

## Method

### Overall Architecture

Dolphin consists of three core processes to form a closed loop (Figure 2):

**1. Idea Generation Process**

#### Paper Retrieval and Ranking
- Uses the Semantic Scholar API to retrieve 50 related papers.
- **Task-Attribute-Guided Paper Ranking**:
    - The LLM first extracts the attributes of the input task (e.g., model input, output, and other characteristics).
    - It scores the papers (on a scale of 1-10) based on two criteria: topic relevance and task attribute match.
    - Papers scored below 8 are filtered out.
    - Effect: For 3D classification tasks, irrelevant papers such as those on 3D detection are significantly reduced.

#### Idea Generation and Filtering
- Based on the ranked highly-relevant papers, the LLM generates $N$ ideas (each containing a title, experimental plan, and abstract).
- **Independence Check**: Uses sentence embeddings to compute the cosine similarity between ideas, filtering redundancies with a threshold of 0.8.
- Maintains an idea library **B** (storing the embeddings of checked ideas).
- **Novelty Check**: Determines whether an idea is novel through Semantic Scholar searches.

**2. Experimental Verification Process**

- The LLM generates a detailed experimental plan and modifies the reference code.
- Core Innovation: **Exception-Traceback-Guided Debugging**
    - Directly feeding the traceback to the LLM yields a low execution success rate (4/15), as the LLM struggles to understand complex nested relationships.
    - Solution: Extract information from the traceback $\rightarrow$ Guide the LLM to generate the local code structure related to the error $\rightarrow$ Debug based on the code structure and the traceback.
    - Focuses only on custom code, excluding library function calls.
    - Maximum of 5 debugging iterations.
    - Effect: Success rate increases from 33.3% to 50.0%.

**3. Results Feedback Process**

- Classifies experimental results into three categories: improvement, maintenance, and decline.
- Adds embeddings of maintained/declining ideas to the idea library B to avoid redundant verification.
- Appends ideas that successfully improve performance to the prompt for the next round of idea generation.
- Closed-loop effect: Improvement rate increases from 2/7 in Loop 1 to 4/8 in Loop 3.

## Key Experimental Results

### Experimental Setup

- **LLM Agent**: GPT-4o-2024-08-06 (idea generation); DeepSeek-v2.5 via Ollama (code execution)
- **Tasks**:
    - 3D Point Cloud Classification: ModelNet40 + PointNet baseline
    - 2D Image Classification: CIFAR-100 + WRN-28-10 baseline
    - Sentiment Classification: SST-2 + BERT-base baseline
- 2 loops executed per task (yielding a total of 40 ideas)

### Main Results

| Task | Baseline | Average Gain | Max Gain | Human-designed SOTA | No. of Effective Ideas |
|------|------|---------|---------|-------------|-----------|
| ModelNet40 OA | 91.0 (PointNet) | 92.0 (+1.0) | **93.9 (+2.9)** | 93.8 (GPSFormer) | 5/40 |
| ModelNet40 mAcc | 87.6 (PointNet) | 88.7 (+1.1) | 91.1 (+3.5) | 91.8 (GPSFormer) | 5/40 |
| CIFAR-100 | 81.2 (WRN) | 81.8 (+0.6) | 82.0 (+0.8) | 82.2 (ResNeXt) | 6/40 |
| SST-2 | 91.0 (BERT-base) | 91.8 (+0.8) | 92.5 (+1.5) | 93.1 (BERT-large) | 6/40 |

**Highlight Result**: PointNet-CSR, automatically generated on ModelNet40, achieves 93.9% OA, closely matching the human-designed SOTA GPSFormer (93.8%)!

### MLE-bench Results

| Task | Code Source | Previous Score | Dolphin Score |
|------|---------|---------|------------|
| Social Insult Detection | AIDE | 81.0 | 84.7 |
| Tabular Prediction | Kaggle | 95.3 | 96.2 |
| Toxic Comment Classification | Kaggle | 94.7 | 97.2 |

- Dolphin can be flexibly integrated with other frameworks such as AIDE and Agent Laboratory.
- It is capable of executing updates to techniques and code versions.

### Ablation Study

#### Analysis of the Idea Generation Process

| Method | No. of Novel Ideas | Average Cost / Idea |
|------|-----------|-------------|
| Naive Generation (w/o retrieval) | 8/20 | \$0.106 |
| Naive Retrieval + Generation | 13/20 | \$0.187 |
| Task-Attribute Filtering (Ours) | **19/20** | \$0.184 |

- Task-attribute filtering increases the proportion of novel ideas from 40% to 95%.

#### Analysis of the Debugging Process

| Local Code Structure | Traceback Information Extraction | Successful Execution Rate (L1/L2/L3) |
|-------------|-------------------|---------------------|
| ✗ | ✗ | 4/15, 5/13, 5/14 |
| ✓ | ✗ | 3/15, 5/13, 6/14 |
| ✓ | ✓ | **7/15, 6/13, 8/14** |

- Providing only the local code structure is insufficient (as it may contain irrelevant library information); extracting information from the traceback is required to focus on custom code.

#### Analysis of Closed-Loop Feedback

| Loop | Loop 1 | Loop 2 | Loop 3 |
|------|--------|--------|--------|
| Improvement Rate | 2/7 | 3/6 | 4/8 |

- As iterations progress, the quality of ideas continues to improve, demonstrating the value of closed-loop feedback.

### Case Study: PointNet-CSR vs DGCNN

| Dimension | DGCNN (Human-designed) | PointNet-CSR (Dolphin) |
|------|------------------|----------------------|
| Idea Level | Architecture-level | Module-level |
| Parameters | Learnable parameters | No learnable parameters |
| Structure | Repeated blocks | Single module |
| mAcc / OA | 90.2% / 92.9% | **91.1% / 93.9%** |
| Training Speed | ~20.86s/epoch | **~6.12s/epoch (>3x faster)** |

PointNet-CSR achieves better and faster performance through a more concise architecture.

## Highlights & Insights

1. **True Closed Loop**: The flow of experimental results $\rightarrow$ feedback $\rightarrow$ next-round idea generation represents a rare fully closed-loop design among current auto-research frameworks.
2. **Validation on Public Benchmarks**: Unlike AI-Scientist, which relies on custom-built datasets, Dolphin validates the efficacy of its ideas on standard benchmarks such as ModelNet40, CIFAR-100, and SST-2.
3. **Nearing Human SOTA**: The automatically generated method on 3D  classification achieves 93.9% accuracy, coming close to or even matching meticulously hand-designed human approaches.
4. **Highly Cost-Effective**: The average cost per idea is only about \$0.2.
5. **Exception-Traceback-Guided Debugging**: Resolves issues with LLMs struggling to understand complex nested code structures. It upgrades the general practice of "feeding error logs" to structured "local code structure analysis".

## Limitations & Future Work

1. **Knowledge Leakage**: Pre-existing knowledge acquired by LLMs during training may lead to the "rediscovery" of existing methods rather than genuine innovation.
2. **Reliance on Abstracts and Titles Only**: Idea generation is solely based on paper titles and abstracts, which limits deep understanding of technical nuances and logical relationships between papers.
3. **Limitations in Coding Capability**: LLMs struggle to comprehend complex, project-level codebases, constraining their ability to verify more intricate tasks.
4. **Low Ratio of Effective Ideas**: Only 5-6 ideas out of 40 are effective ($12.5\% \sim 15\%$), meaning substantial computational resources are spent on ineffective verifications.
5. **Constrained Scope of Tasks**: The baseline models verified are relatively simple (PointNet, WRN, BERT-base), and have not been tested on more sophisticated modern architectures.
6. **Simplistic Feedback Signals**: The feedback relies entirely on a coarse "improvement/maintenance/decline" classification, lacking a detailed analysis of failure modes.

## Related Work & Insights

- **Open-ended Auto-Research**: AI-Scientist (Lu et al., 2024) proposes an end-to-end framework but lacks feedback and validation on real benchmarks; Chain of Ideas (Li et al., 2024) generates ideas based on paper chains but lacks experimental verification; NOVA (Hu et al., 2024) enhances novelty through iterative refinement.
- **Constrained Auto-Research**: AutoML-GPT (Zhang et al., 2023b) utilizes LLMs for hyperparameter tuning; AgentHPO (Liu et al., 2024) iteratively optimizes hyperparameters.
- **Code Generation**: AIDE (Schmidt, 2024) and Agent Laboratory (Schmidgall et al., 2025) automate code generation in machine learning competitions and laboratory scenarios, respectively.

## Rating ⭐⭐⭐⭐

The closed-loop design concept is sound and systematically validated (with improvement rates consistently rising across cycles), yielding impressive results that approach human SOTA on standard benchmarks. Task-attribute-guided ranking and exception-traceback-guided debugging are highly practical technical contributions. The principal limitations stem from the relatively low proportion of effective ideas, simple baseline configurations, and the threat of LLM knowledge leakage. Overall, however, this represents a solid step toward fully automated scientific research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Learning Dynamics of RNNs in Closed-Loop Environments](../../NeurIPS2025/others/learning_dynamics_of_rnns_in_closed-loop_environments.md)
- [\[ACL 2025\] Inner Thinking Transformer: Leveraging Dynamic Depth Scaling to Foster Adaptive Internal Thinking](inner_thinking_transformer_leveraging_dynamic_depth_scaling_to_foster_adaptive_i.md)
- [\[ACL 2025\] Research Borderlands: Analysing Writing Across Research Cultures](research_borderlands_analysing_writing_across_research_cultures.md)
- [\[ICML 2025\] Regression for the Mean: Auto-Evaluation and Inference with Few Labels through Post-hoc Regression](../../ICML2025/others/regression_for_the_mean_auto-evaluation_and_inference_with_few_labels_through_po.md)
- [\[ACL 2025\] Learning to Reason from Feedback at Test-Time](learning_to_reason_from_feedback_at_test-time.md)

</div>

<!-- RELATED:END -->

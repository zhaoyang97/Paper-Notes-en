---
title: >-
  [Paper Note] Hybrid Preferences: Learning to Route Instances for Human vs. AI Feedback
description: >-
  [ACL 2025][Preference Learning] This paper proposes HyPER (Hybrid Preference Router), which dynamically decides whether each annotation instance should receive human or AI preference feedback by training a performance prediction model. It achieves a 7-13% improvement on RewardBench compared to pure human or pure AI annotation, while significantly reducing annotation costs.
tags:
  - "ACL 2025"
  - "Preference Learning"
  - "Human Feedback"
  - "AI Feedback"
  - "Routing Strategy"
  - "Reward Model"
date: 2026-05-08
content_hash: 0df1b59c782cef04
---

# Hybrid Preferences: Learning to Route Instances for Human vs. AI Feedback

**Conference**: ACL 2025  
**arXiv**: [2410.19133](https://arxiv.org/abs/2410.19133)  
**Code**: Yes  
**Area**: Other  
**Keywords**: Preference Learning, Human Feedback, AI Feedback, Routing Strategy, Reward Model

## TL;DR

This paper proposes HyPER (Hybrid Preference Router), which dynamically decides whether each annotation instance should receive human or AI preference feedback by training a performance prediction model. It achieves a 7-13% improvement on RewardBench compared to pure human or pure AI annotation, while significantly reducing annotation costs.

## Background & Motivation

**Background**: Reinforcement Learning from Human Feedback (RLHF) has become the core paradigm for aligning language models. However, human preference annotation is expensive, slow, and suffers from inconsistent quality. Recently, using LLMs to generate synthetic preference annotations (AI Feedback/RLAIF) has emerged as a low-cost alternative, but it introduces model-inherent biases and errors.

**Limitations of Prior Work**: Existing approaches either rely entirely on human annotation (expensive with low inter-annotator agreement) or entirely on AI annotation (systemic biases, such as preferring longer or overly polite responses). Both approaches have their pros and cons, but there is no systematic study on how to combine them to achieve optimal performance.

**Key Challenge**: Both human and AI annotations have their blind spots—humans are better at certain types of instances (e.g., scenarios involving subtle value judgments or safety issues), while AI performs better on others (e.g., straightforward factual judgments). Unifying all instances under a single source is inherently suboptimal.

**Goal**: Design an intelligent routing mechanism that automatically decides whether each instance should be annotated by human or AI, so that the resulting hybrid preference dataset can train the best reward model (RM).

**Key Insight**: Model the annotation allocation as an optimization problem: given a budget constraint, select the optimal combination of human and AI annotations to maximize reward model performance.

**Core Idea**: Train a "Performance Prediction Model" (PPM) that predicts the performance of the reward model under any combination of human and AI annotations, and then use a routing strategy to find the optimal combination.

## Method

### Overall Architecture

The pipeline of HyPER consists of three stages: (1) Constructing the MultiPref dataset—collecting both human and LM preference annotations for 10K instances; (2) Training the Performance Prediction Model (PPM)—training reward models on subsets of data with known annotation sources to learn how different combinations of annotation sources affect RM performance; (3) Routing Optimization—utilizing the PPM's predictions to find the optimal instance-level annotation allocation plan (human or AI for each instance) using a greedy or search-based strategy. The final output is a hybrid preference dataset used to train the ultimate reward model.

### Key Designs

1. **MultiPref Multi-Source Preference Dataset**:

    - **Function**: Provide paired human and AI annotation data for training the routing strategy.
    - **Mechanism**: A preference dataset of 10K instances is constructed, where each instance contains preference judgments from both human annotators and a language model (e.g., GPT-4). This dataset covers various task types and difficulty levels. The agreement/disagreement between the two annotation sources is recorded for each instance, laying the foundation for analyzing "which instances are suitable for human annotation."
    - **Design Motivation**: It is impossible to train a routing model without paired data. Furthermore, paired data allows for the analysis of annotation quality trade-offs between humans and AI across various instance types.

2. **Performance Prediction Model (PPM)**:

    - **Function**: Predict the performance of the reward model given any combination of annotation sources.
    - **Mechanism**: The input to the PPM is a set of instances along with their annotation source assignments (which are annotated by humans and which by AI), and the output is the predicted performance of the RM trained on this assignment. The training method involves training multiple RMs with different human/AI combinations on subsets of MultiPref and recording their performance on a validation set as training data for the PPM. The PPM can learn patterns such as "safety-related instances yield better RM performance with human annotations" and "simple factual judgments can be handled by AI."
    - **Design Motivation**: Directly searching through all possible annotation allocation combinations exhibits exponential complexity. The PPM significantly reduces search costs by learning to generalize these performance patterns.

3. **Routing Strategy Optimization**:

    - **Function**: Find the optimal instance allocation scheme under a given human annotation budget.
    - **Mechanism**: Formulate the routing optimization as a constrained optimization problem—maximizing the RM performance predicted by the PPM under a human budget constraint (e.g., at most 30% of total annotations are done by humans). A greedy strategy is adopted: starting from an all-AI annotation baseline, iteratively switch instances that the PPM predicts will yield the "highest performance gain when changed to human annotation" to human annotation, until the budget limit is reached.
    - **Design Motivation**: Although the greedy strategy is not globally optimal, it is highly efficient, and experiments prove that its performance is already sufficiently effective.

### Loss & Training

The reward model is trained using the standard Bradley-Terry preference learning loss. The PPM is trained using a regression loss to predict RM performance. The routing optimization employs a greedy strategy based on PPM predictions.

## Key Experimental Results

### Main Results

| Annotation Strategy | RewardBench Performance | Relative Gain | Explanation |
|---|---|---|---|
| Pure Human | Baseline | - | High cost |
| Pure AI | Baseline | - | Low cost but biased |
| Random Mix (50/50) | Slight improvement | +2-3% | Simple mixing helps |
| **HyPER Routing** | **Optimal** | **+7-13%** | Intelligent routing is highly effective |

### Ablation Study

| Experimental Setup | Key Metric | Explanation |
|---|---|---|
| HyPER (RewardBench) | +7-13% | Main evaluation benchmark |
| Best-of-N Reranking | +2-3% | Consistent improvement on downstream tasks |
| Transfer to new datasets | Generalizes well | Routing strategy remains effective across datasets |
| Transfer to new base models | Generalizes well | Not restricted to a specific model |

### Key Findings

- **Hybrid annotation consistently outperforms single-source annotation**: On multiple evaluation benchmarks including RewardBench and Best-of-N, HyPER's hybrid annotation scheme outperforms pure human or pure AI annotation. This demonstrates the complementarity of the two annotation sources.
- **A significant improvement of 7-13%** proves that intelligent routing is far superior to random mixing, indicating that "which instance is annotated by whom" has a massive impact on final performance.
- **Safety and complexity are key routing features**: Analyzing the routing patterns learned by HyPER reveals that prompts with moderate safety risks or moderate complexity benefit the most from human annotations. AI annotations suffice for extremely simple instances, while extremely complex ones are difficult for humans to annotate reliably as well.
- **Cross-dataset and cross-model generalization**: HyPER maintains its advantages on unseen preference datasets and different base models, showing that the learned routing strategy reflects the intrinsic complementarity between human and AI annotations rather than dataset-specific biases.
- **Reducing annotation costs**: To achieve the equivalent RM performance, HyPER can replace a full-human annotation dataset using only 30-50% of the human annotation budget, significantly reducing costs.

## Highlights & Insights

- **The "routing" concept** is the core contribution of this paper: it reframes the "human vs. AI" choice from a binary trade-off into an optimizable allocation problem. This paradigm can be widely transferred to other human-AI collaboration scenarios—such as data cleaning, code review, content moderation, or any annotation task requiring a balance between cost and quality.
- **The meta-learning approach of the Performance Prediction Model** is highly ingenious: by training multiple RMs on various annotation combinations and recording their performances, the PPM essentially performs "meta-learning on the efficacy of annotation strategies." This indirect optimization circumvents the exponential complexity of direct searching.
- **Analysis of routing features** provides practical insights: instances with moderate safety risks and moderate complexity are best suited for human annotation—which aligns with intuition (as extreme cases often have unambiguous patterns).

## Limitations & Future Work

- **Limited scale of MultiPref**: 10K instances may not be sufficient to cover all task types and difficulty levels. Scaling up might uncover more fine-grained routing policies.
- **Binary routing only**: The current work only supports a binary choice of "human vs. AI." In real-world scenarios, there may be multiple annotators or tagging strategies available (e.g., different LLMs, different human groups, different prompt strategies).
- **PPM training overhead**: Training multiple RMs to generate training data for the PPM incurs non-negligible upfront costs.
- **Dynamic annotation quality**: Human annotators may exhibit fatigue over time, and AI models will be updated. Static routing strategies may need periodic retraining.
- **Future directions**: Explore online learning frameworks to update the routing strategy while collecting annotations, and extend to multi-source routing.

## Related Work & Insights

- **vs. RLAIF (such as Constitutional AI)**: RLAIF completely replaces human annotations with AI annotations. This paper proves that complete replacement is suboptimal, and intelligent hybrid learning is the correct approach. HyPER can be considered a superior alternative to RLAIF.
- **vs. Active Learning**: Active Learning selects the "most informative instances" for humans to annotate. HyPER goes a step further—selecting not just the instance but also the annotator. It can be viewed as a generalization of the active learning framework.
- **vs. Data Mixing/Curriculum Learning**: These works focus on the order in which data is utilized, whereas HyPER focuses on selection of the data source, representing an orthogonal but complementary research direction.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The formulation of "routed annotation" is novel, and the technical solution of combining PPM with routing optimization is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive verification across RewardBench, Best-of-N, multiple datasets, and multiple base models.
- **Writing Quality**: ⭐⭐⭐⭐ Clear problem modeling and rigorous formulation of the optimization objectives.
- **Value**: ⭐⭐⭐⭐⭐ Directly practical for RLHF deployment, with widely transferrable routing insights.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] HelpSteer3: Human-Annotated Feedback and Edit Data to Empower Inference-Time Scaling](helpsteer3_human-annotated_feedback_and_edit_data_to_empower_inference-time_scal.md)
- [\[ACL 2025\] Learning to Reason from Feedback at Test-Time](learning_to_reason_from_feedback_at_test-time.md)
- [\[ICLR 2026\] RADAR: Learning to Route with Asymmetry-aware Distance Representations](../../ICLR2026/others/radar_learning_to_route_with_asymmetry-aware_distance_representations.md)
- [\[ACL 2025\] A Little Human Data Goes A Long Way](a_little_human_data_goes_a_long_way.md)
- [\[ACL 2025\] Behavioural vs. Representational Systematicity in End-to-End Models: An Opinionated Survey](behavioural_vs_representational_systematicity_in_end-to-end_models_an_opinionate.md)

</div>

<!-- RELATED:END -->

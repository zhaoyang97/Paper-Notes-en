---
title: >-
  [Paper Note] From Isolated Scoring to Collaborative Ranking: A Comparison-Native Framework for LLM-Based Paper Evaluation
description: >-
  [ACL2026][Reinforcement Learning][Comparative Evaluation] This paper transforms LLM paper review from "assigning absolute scores to single papers" to "pairwise comparison followed by global ranking." By employing semanti…
tags:
  - "ACL2026"
  - "Reinforcement Learning"
  - "Comparative Evaluation"
  - "Paper Ranking"
  - "LLM for Paper Review"
  - "Bradley-Terry"
  - "RLVR"
date: 2026-05-08
content_hash: d0481c669cec0c0a
---

# From Isolated Scoring to Collaborative Ranking: A Comparison-Native Framework for LLM-Based Paper Evaluation

**Conference**: ACL2026  
**arXiv**: [2603.17588](https://arxiv.org/abs/2603.17588)  
**Code**: The paper claims to be open-sourced, but the current cache does not provide a specific GitHub URL  
**Area**: LLM Evaluation / Automated Peer Review / Learning to Rank  
**Keywords**: Comparative Evaluation, Paper Ranking, LLM for Paper Review, Bradley-Terry, RLVR  

## TL;DR
This paper transforms LLM paper review from "assigning absolute scores to single papers" to "pairwise comparison followed by global ranking." By employing semantic graph sampling, comparative SFT, and reinforcement learning with verifiable rewards (RLVR) to train a 7B model, it significantly outperforms DeepReview-14B in ICLR-2025 paper ranking and acceptance prediction, while demonstrating zero-shot transferability to multiple unseen conferences.

## Background & Motivation
**Background**: LLMs have been utilized to assist in paper reviews. Common paradigms involve models outputting absolute scores, review comments, or acceptance recommendations after reading a single paper. Training-based methods use historical review scores for supervision, while agent-based methods simulate multi-reviewer discussions; however, most ultimately center on "what score a single paper should receive."

**Limitations of Prior Work**: Absolute scores are unstable. A score of "6" carries different meanings across different conferences, years, fields, or review criteria. Models easily learn dataset-specific scoring habits rather than transferable academic judgment. Furthermore, paper reviewing is inherently a ranking problem: program committees are concerned with which submissions among a pool are most deserving of acceptance, rather than evaluating each paper against an isolated, fixed scale.

**Key Challenge**: LLMs are proficient at relative judgment and linguistic reasoning, but existing training signals force them to fit absolute numerical values. These values are influenced by conference scales, scoring habits, and domain differences, causing models to mistake dataset-specific rules for "paper quality."

**Goal**: The authors aim to establish a comparison-native paper evaluation framework that directly processes preferences between paper pairs during data construction, model training, and inference stages, aggregating these pairwise preferences into a global quality ranking to determine the acceptance set.

**Key Insight**: Starting from the observation that "human reviewers often form judgments through comparison," the paper decomposes complex quantitative scoring into a large number of simpler pairwise comparisons. Such supervision avoids inconsistent absolute score scales and aligns better with the LLM's capacity for comparative analysis.

**Core Idea**: Replace "absolute scoring for single papers" with "graph sampling of informative paper pairs + comparative SFT/RLVR for preference learning + Bradley-Terry for preference aggregation."

## Method
The key to this method is not a new reviewer prompt, but the complete rewriting of the paper evaluation data flow into a comparison task: constructing paper pairs during training to let the model judge which is of higher quality; performing only pairwise judgments during inference; and finally aggregating these preference edges into a global ranking.

### Overall Architecture
The input consists of a batch of papers to be evaluated, represented primarily by their titles and abstracts. The framework first selects the paper pairs to be compared via pair sampling, then uses a trained LLM to output preference labels for each pair. During the training phase, sampled pairs are filtered by score difference and frequency constraints to form reliable supervision; the model first learns comparative reasoning via cold-start SFT, followed by reinforcement learning (RLVR) based on ground-truth average score preferences. In the inference phase, the system controls the sampling ratio to compare only a subset of pairs, subsequently using the Bradley-Terry model to estimate the latent quality score of each paper and making acceptance/rejection decisions based on the ranking threshold.

### Key Designs
1. **GBR-BR Semantic Graph Sampling**:

	- **Function**: Prioritizes picking "both comparable and informative" pairs from all possible paper pairs to reduce meaningless comparisons.
	- **Mechanism**: For each paper, candidate neighbors are retrieved via embeddings, followed by a reranker to obtain bidirectional rankings. If papers $p_i$ and $p_j$ appear in the top rankings of either direction, an edge is created in the graph with a weight of $2k_r-r_{ij}-r_{ji}$. The graph must remain connected; if isolated nodes appear, retrieval and reranking thresholds are relaxed. Finally, a priority list for comparison is generated by sorting edge weights.
	- **Design Motivation**: Comparing random papers produces pairs that are cross-domain, difficult to judge, or low in information density, while comparing only within the same domain might lose cross-domain generalization. The semantic graph allows similar papers to provide fine-grained supervision while maintaining global connectivity to ensure all papers receive comparison signals.

2. **Comparative SFT + RLVR Training**:

	- **Function**: Enables the 7B model to truly learn "which paper is stronger" rather than delegating the comparison task to an untrained general LLM.
	- **Mechanism**: Training pairs must satisfy two constraints: the difference in average review scores must be at least $d_{min}=1.5$, and each paper appears at most $c_{max}=1$ time to reduce noise from hard-to-distinguish samples and improve diversity. The model first uses an instruct LLM to generate comparative reasoning chains for cold-start SFT, followed by reinforcement learning using an improved GRPO. The reward comes from the ground-truth preference label $y_{ij}=\mathbb{I}(s_i>s_j)$ based on average scores; a correct prediction yields $R_l=\gamma\cdot\mathbb{I}(y_{ij}=\hat y_{ij}^{(l)})$, with $\gamma=5$ in experiments.
	- **Design Motivation**: Direct RL can be unstable, while pure SFT may not optimize the final preference accuracy. SFT provides the comparative reasoning format and cold-start capability, while RLVR continues to strengthen relative quality judgment using verifiable labels.

3. **Preference Aggregation and Ranking Decision**:

	- **Function**: Converts local pairwise preferences into a global quality ranking for a batch of papers.
	- **Mechanism**: During inference, only a fraction of the theoretical $O(n^2)$ paper pairs are sampled, with the ratio set to $\alpha=0.05$. Each pair receives a preference label from the trained LLM, and the Bradley-Terry model is used to estimate latent quality $\theta_i$, where $p_{ij}=e^{\theta_i}/(e^{\theta_i}+e^{\theta_j})$ represents the probability that $i$ is superior to $j$. After maximizing the likelihood of all preference labels, papers are ranked by $\theta_i$, and the acceptance set is determined by a preset acceptance rate.
	- **Design Motivation**: Pairwise judgment alone cannot directly yield a conference-level acceptance set; Bradley-Terry provides an interpretable, classic, and experimentally effective aggregation method.

### Loss & Training
The base model is Qwen2.5-7B-Instruct, fine-tuned using LoRA for efficiency. Main training data comes from ICLR-2025, following the DeepReview train-test split, with the ground truth being the actual average review scores. Training pairs are filtered using $d_{min}=1.5$ and $c_{max}=1$. The inference sampling ratio is $\alpha=0.05$, and the acceptance rate is fixed at 31.4% (the average of ICLR-2023 and ICLR-2024). The optimization workflow is "Comparative Reasoning Cold-start SFT → RLVR." Rewards do not fit absolute scores; they only check if the model's preference direction matches the ground-truth average score ranking.

## Key Experimental Results

### Main Results
| Dataset / Task | Metric | Ours (CNPE-7B) | Best or Representative Baseline | Gain / Conclusion |
|--------|------|------|----------|------|
| ICLR-2025 Acceptance Prediction | Accuracy | 0.7192 | DeepReview-14B 0.6845 | 7B model outperforms 14B trained reviewer |
| ICLR-2025 Acceptance Prediction | F1 | 0.6732 | DeepReview-14B 0.6254 | More sensitive to the accept/reject boundary |
| ICLR-2025 Acceptance Prediction | AUC | 0.7408 | DeepReview-14B 0.6624 | Claims 11.8% improvement over the closest competitor |
| ICLR-2025 Ranking | Spearman $\rho$ | 0.4091 | DeepReview-14B 0.4014 | Slight lead in rank correlation |
| ICLR-2025 Ranking | MAP@20 | 0.7076 | PairReview(GLM) 0.3474 / DeepReview-14B 0.1478 | Most significant advantage in identifying top-20 high-quality papers |
| ICLR-2025 Ranking | NDCG@20 | 0.8153 | PairReview(Gemini) 0.7522 / DeepReview-14B 0.7204 | Better quality in top-tier ranking |
| ICLR-2025 Overall | Avg. Perf. | 1.0000 | DeepReview-14B 0.8211 | Average relative improvement of 21.8% |

### Ablation Study
| Configuration | Key Metrics | Description |
|------|---------|------|
| Full model | Avg. Perf. 1.0000; F1 0.6732; MAP@20 0.7076 | Full CNPE, SFT+RLVR, mixed similarity/random sampling for both training and inference |
| w/o Training | Avg. Perf. 0.5845 | Performance drops 41.6% without comparative training, proving general models aren't reliable comparators |
| w/o RLVR | Avg. Perf. 0.7744 | SFT alone is insufficient; performance degrades by 21.6% |
| w/o SFT cold-start | Avg. Perf. 0.7511 | Direct RL is inferior to starting with cold-start, highlighting importance of reasoning format |
| w/o Random (train) | Avg. Perf. 0.8792 | Lack of random cross-domain pairs during training hurts generalization |
| w/o Sim (train) | Avg. Perf. 0.8358 | Lack of semantically similar pairs leads to greater loss; fine-grained comparison is critical |
| w/o Random (test) | Avg. Perf. 0.9890 | Removing one sampling type during inference has a minor impact |
| w/o Sim (test) | Avg. Perf. 0.9842 | Diversity in inference pairs is helpful, but training diversity is more critical |

### Key Findings
- Improvements in ranking metrics are more prominent than in binary acceptance prediction, particularly MAP@20, which jumped to 0.7076. This suggests that a comparison-native design is indeed better suited for the goal of "finding the strongest papers."
- SFT and RLVR are complementary: SFT gives the model comparative reasoning capability, while RLVR aligns preference directions with ground-truth average scores; missing either step leads to significant performance drops.
- In generalization tests on unseen conferences, the model showed a large percentile gap in accepted/rejected groups for ICML and NeurIPS (+18.5 and +15.9, respectively); for ACL, EMNLP, and NAACL Long/Findings distinctions, the gaps were +5.2, +11.4, and +8.9, aligning with the intuition that quality differences in these groups are more subtle.

## Highlights & Insights
- **Clarification of the Review Task Objective**: Paper quality evaluation is ultimately a ranking problem; absolute scores are merely intermediate representations. By making data, training, inference, and aggregation all comparison-native, this approach is more fundamental than simply "asking the model to think more" in a prompt.
- **Semantic Graph Sampling is a Practical Trade-off**: Purely random pairs offer generalization but are noisy, while purely intra-domain pairs may be too narrow. GBR-BR combines fine-grained comparability with global coverage via a connected graph—a strategy transferable to model evaluation, candidate answer ranking, and data filtering.
- **RLVR Design Avoids Expensive Human Feedback**: The authors did not train a separate reviewer reward model but constructed verifiable rewards using the magnitude relationship of actual average scores. This training paradigm is highly attractive for any task that can be converted into verifiable preference labels.
- **Explicit Discussion of Positional Bias**: The paper notes that untrained base models exhibit clear positional bias, whereas comparative SFT+RL ensures the ranking is no longer significantly correlated with paper IDs, which is vital for pairwise evaluation systems.

## Limitations & Future Work
- The data scope remains narrow; experiments rely primarily on 2025 CS top-tier conference papers. Conclusions may not generalize to medicine, social sciences, or long-cycle journal reviews.
- The method uses only titles and abstracts. While computationally efficient, this misses method details, experimental rigor, and limitations found in full papers; thus, it cannot yet be equated to a true full-paper review.
- Model scale was limited to 7B due to resource constraints. Although it outperformed DeepReview-14B, the authors acknowledge that larger models might offer stronger linguistic understanding and knowledge coverage.
- Automated review still falls short of human expert quality and carries ethical risks: using it as a direct replacement for reviewers might reinforce mainstream biases, suppress non-mainstream directions, or lead to the degradation of human reviewing skills. A more appropriate role is as a tool for auxiliary ranking, pre-screening, and feedback.

## Related Work & Insights
- **vs DeepReview / SEA / CycleReviewer**: These training-based review models primarily learn absolute scores or review text for single papers. CNPE learns quality preferences between two papers, making it less dependent on the score scales of specific conferences.
- **vs AIScientist / AgentReview**: Agent systems simulate review workflows but often rely on untrained general LLMs and tend to output point-wise scores. CNPE directly trains a comparator and aggregates results using a ranking model.
- **vs PairReview / NAIP**: PairReview performs pairwise comparisons but often uses fixed LLMs as comparators; NAIP introduces listwise training but still predicts isolated scores. This work remains comparison-native across sampling, training, and inference.
- **Insight**: Many "scoring" tasks should perhaps be reformulated as "comparison + ranking," such as data quality filtering, evaluation of generated answers, benchmark rankings, and reviewer assignment quality control. The key is designing pair sampling that is both sufficiently covered and noise-controlled.

## Rating
- **Novelty**: ⭐⭐⭐⭐☆ While pairwise/listwise approaches aren't entirely new, systematizing sampling, RLVR, and Bradley-Terry aggregation for paper review is comprehensive.
- **Experimental Thoroughness**: ⭐⭐⭐⭐☆ Covers main experiments, ablations, hyperparameters, unseen conference generalization, and positional bias; the limitation is the reliance on titles/abstracts and a focus on CS conferences.
- **Writing Quality**: ⭐⭐⭐⭐☆ Structure is clear, with well-explained motivations and modules. Tables are information-dense.
- **Value**: ⭐⭐⭐⭐☆ Offers direct insights for automated review and LLM-as-judge tasks, though it must be used very cautiously as an auxiliary system.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] A Multi-Agent Conversational Bandit Approach to Online Evaluation and Selection of User-Aligned LLM Responses](../../AAAI2026/reinforcement_learning/a_multi-agent_conversational_bandit_approach_to_online_evaluation_and_selection_.md)
- [\[ICLR 2026\] Toward a Dynamic Stackelberg Game-Theoretic Framework for Agent-Based Conversational AI Defense Against LLM Jailbreaking](../../ICLR2026/reinforcement_learning/toward_a_dynamic_stackelberg_game-theoretic_framework_for_agent-based_conversat.md)
- [\[ACL 2026\] Efficient Hyperparameter Optimization for LLM Reinforcement Learning](efficient_hyperparameter_optimization_for_llm_reinforcement_learning.md)
- [\[ACL 2026\] Semantic-Space Exploration and Exploitation in RLVR for LLM Reasoning](semantic-space_exploration_and_exploitation_in_rlvr_for_llm_reasoning.md)
- [\[ACL 2026\] Community-Aware Assessment of Social Textual Engagement and Resonance: A Human-Centric Perspective on User-Generated Content Evaluation](community-aware_assessment_of_social_textual_engagement_and_resonance_a_human-ce.md)

</div>

<!-- RELATED:END -->

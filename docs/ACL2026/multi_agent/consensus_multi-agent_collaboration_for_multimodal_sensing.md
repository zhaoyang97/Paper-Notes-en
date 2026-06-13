---
title: >-
  [Paper Note] ConSensus: Multi-Agent Collaboration for Multimodal Sensing
description: >-
  [ACL2026][Multi-Agent][Multi-agent collaboration] ConSensus is a training-free multi-agent sensor fusion framework that assigns different sensing modalities to specialized agents for independent interpretation…
tags:
  - "ACL2026"
  - "Multi-Agent"
  - "Multi-agent collaboration"
  - "multimodal sensing"
  - "sensor fusion"
  - "statistical consensus"
  - "semantic fusion"
date: 2026-05-08
content_hash: c14b9db3239c31a4
---

# ConSensus: Multi-Agent Collaboration for Multimodal Sensing

**Conference**: ACL2026 Findings  
**arXiv**: [2601.06453](https://arxiv.org/abs/2601.06453)  
**Code**: https://github.com/nokia/multi-agent-collaboration-for-multimodal-sensing  
**Area**: Multimodal Sensing / LLM Agent  
**Keywords**: Multi-agent collaboration, multimodal sensing, sensor fusion, statistical consensus, semantic fusion  

## TL;DR
ConSensus is a training-free multi-agent sensor fusion framework that assigns different sensing modalities to specialized agents for independent interpretation, followed by semantic fusion, statistical consensus, and hybrid arbitration to obtain final judgments. It achieves an average accuracy improvement of 7.1% over single-agent methods across five multimodal sensing benchmarks while reducing fusion token costs to approximately 1/12.7 of multi-round debate methods.

## Background & Motivation
**Background**: LLMs are being utilized to interpret real-world sensor data, such as motion recognition, sleep stage identification, stress detection, and health monitoring. A common practice is to include statistical features from multiple sensors in a single prompt to let a single LLM perform inference in one go.

**Limitations of Prior Work**: Information density, reliability, and semantic meaning vary across heterogeneous sensors. Single agents tend to ignore certain modalities or be dominated by a prominent modality. Furthermore, pure LLM judges are influenced by prior knowledge biases (e.g., over-relying on medically significant ECG), while pure majority voting is fragile when sensors are missing or noise is high.

**Key Challenge**: Multimodal sensing requires both semantic understanding and statistical robustness. Semantic aggregation can identify sensor failures and contextual clues but suffers from knowledge bias; statistical voting can suppress individual agent errors but relies on voters being reliable and independent. In real sensing environments, these two conditions often fail simultaneously.

**Goal**: The authors aim to propose a training-agnostic and model-agnostic collaboration protocol deployable to various sensing tasks, enabling LLMs to fuse heterogeneous sensing modalities more stably without retraining sensor encoders.

**Key Insight**: The paper decomposes an "all-in-one" multimodal prompt into multiple modality-aware agents. Each agent interprets only one sensing modality, followed by explicitly defined roles for semantic fusion, statistical fusion, and final hybrid arbitration to balance different inductive biases.

**Core Idea**: By letting each sensing modality speak independently first, and then allowing the final fusion agent to observe both "semantic interpretations" and "statistical consensus," the system can dynamically choose between knowledge bias and voting fragility.

## Method
The core of ConSensus is not training new models but designing a multi-agent reasoning workflow. Given a task description and $N$ sensing modalities, the system creates a specialized agent for each modality to obtain predictions and explanations. Subsequently, three fusion agents sequentially generate semantic aggregation, statistical consensus explanations, and the final hybrid decision.

### Overall Architecture
Inputs include task descriptions, category sets, and multimodal sensor features. Each modality agent receives features from a single modality and task instructions, outputting a prediction $\hat{y}_i$ and rationale $r_i$. These outputs are forwarded to the semantic fusion agent and statistical fusion agent: the former generates a knowledge-driven prediction based on cross-modal semantic evidence, while the latter generates a consensus-driven explanation centered on the majority vote result. Finally, the hybrid fusion agent reads both to output the final category and explanation.

This workflow relies solely on prompts and LLM calls without additional supervised training. The main experiments use gpt-oss-20B with temperature set to 0, evaluating performance using accuracy across five sensing tasks.

### Key Designs
1.  **Modality-Specific Agent Decomposition**:

	- **Function**: Decomposes complex multi-sensor inputs into multiple single-modality interpretation tasks.
	- **Mechanism**: The $i$-th agent only observes modality $m_i$ and task $T$, outputting its own prediction $\hat{y}_i$ and explanation $r_i$. This ensures each agent explicitly processes evidence from its modality rather than being drowned out in a large prompt.
	- **Design Motivation**: The primary issues with single agents are context overload and modality dominance. Decomposition ensures that even weak signal modalities are interpreted independently.

2.  **Parallel Modeling of Semantic and Statistical Fusion**:

	- **Function**: Models knowledge-driven and consensus-driven fusion biases separately.
	- **Mechanism**: The semantic fusion agent reads all $(\hat{y}_i, r_i)$ to form predictions based on cross-modal causal relationships and domain knowledge. The statistical fusion agent first calculates the majority vote $\hat{y}_{vote}=\arg\max_c \sum_i \mathbf{1}[\hat{y}_i=c]$, then generates an explanation for this voting result.
	- **Design Motivation**: Semantic fusion is adept at identifying sensor failures but may over-rely on priors. Statistical fusion can mitigate the impact of individual errors but fails under missing modalities and correlated errors.

3.  **Hybrid Arbitration Agent**:

	- **Function**: Performs instance-wise arbitration between semantic and statistical predictions.
	- **Mechanism**: The hybrid fusion agent observes $(\hat{y}_{sem}, r_{sem})$ and $(\hat{y}_{stat}, r_{stat})$ simultaneously, providing a final prediction $\hat{y}$ based on the reliability of both explanations. Instead of simple averaging, the LLM judges whether to trust semantic consistency or statistical stability for the specific sample.
	- **Design Motivation**: Real sensing tasks lack a fixed optimal fusion rule. The most reliable evidence source fluctuates based on samples, missingness patterns, and noise levels.

### Loss & Training
ConSensus is a training-free method with no parameter updates or loss functions. In experiments, all models run with deterministic inference using 1-shot in-context learning, where sensor features are embedded in structured text prompts. The "training strategy" is effectively the design of the inference-time protocol: a single round of modality interpretation followed by a single round of semantic/statistical/hybrid fusion, rather than Self-Consistency or multi-round debate.

## Key Experimental Results

### Main Results
| Method | WESAD | SleepEDF | ActionSense | MMFit | PAMAP2 | Avg. | Fusion Extra Tokens |
|------|-------|----------|-------------|-------|--------|------|----------------|
| Single-Agent | 0.793 | 0.519 | 0.577 | 0.819 | 0.551 | 0.652 | None |
| Self-Consistency | 0.786 | 0.541 | 0.555 | 0.862 | 0.547 | 0.658 | Multi-path sampling |
| Self-Refine | 0.747 | 0.551 | 0.566 | 0.822 | 0.563 | 0.650 | Two rounds refinement |
| Debate | 0.873 | 0.548 | 0.609 | 0.984 | 0.561 | 0.715 | ~76K |
| ReConcile | 0.880 | 0.571 | 0.640 | 0.964 | 0.579 | 0.727 | ~78.6K |
| Semantic Fusion | 0.825 | 0.580 | 0.605 | 0.964 | 0.559 | 0.707 | ~6K |
| Statistical Fusion | 0.927 | 0.592 | 0.597 | 0.960 | 0.534 | 0.722 | ~6K |
| ConSensus | 0.880 | 0.600 | 0.611 | 0.967 | 0.558 | 0.723 | ~6K |

ConSensus achieves an average Gain of 7.1 percentage points over Single-Agent. Its average accuracy is slightly lower than ReConcile's 0.727, but it only requires a single round of fusion, reducing aggregation tokens from ~78.6K to 6K. Compared to the average overhead of multi-agent debate, the paper reports a 12.7x reduction in fusion tokens.

### Ablation Study
| Experiment | Key Findings | Note |
|------|----------|------|
| Semantic vs Statistical | Statistical Fusion (Avg 0.722) vs Semantic Fusion (Avg 0.707) | Statistical consensus is generally stronger, but optimal strategies vary by dataset. |
| Hybrid Fusion | Outperformed single-branch semantic/statistical on SleepEDF, ActionSense, and MMFit | Hybrid agents can select more reliable biases at the instance level. |
| Robustness to Missing Modalities | Statistical Fusion drops to 41.4% at 50% missingness, Semantic Fusion remains at 59.9% | Pure voting is highly fragile at high missingness rates. |
| ConSensus vs Statistical Fusion | Higher by 9.1% and 18.4% at 30% and 50% missingness, respectively | Hybrid shifts to semantic interpretation when statistical certainty decreases. |
| Small Model Generalization | Single-Agent (0.293) vs ConSensus (0.456) on Llama-3.1-8B | Small models benefit from a +16.3 point Gain through agent decomposition. |

### Key Findings
- Modality decomposition itself is critical. Even without hybrid fusion, both semantic and statistical fusion significantly outperform single-agent baselines.
- While ReConcile achieves high average accuracy, its token cost is heavy; ConSensus serves as a structured single-round protocol trade-off for near-debate performance.
- Statistical voting is effective in tasks like WESAD where semantic priors can be misleading, but it degrades rapidly with missing modalities.
- ConSensus is particularly valuable for smaller models. Llama-3.1-8B's single-agent performance is weak, but multi-agent decomposition yields a larger relative Gain.

## Highlights & Insights
- The most inspiring aspect is the explicit split of "fusion" into two inductive biases rather than relying on a single judge. Semantic interpretation and statistical consensus each have blind spots; the value of the hybrid agent lies in making these biases complementary.
- The paper improves performance across multiple datasets without training sensing models, a direction highly suitable for real-world deployment where large-scale labeling is often unavailable.
- "Majority vote" in sensor fusion is not inherently reliable. Missing modalities break the assumption of independent and reliable voters, a point equally applicable to multimodal LLM systems.
- This work suggests that in multimodal tasks, building an intermediate interpretation layer to explicitly preserve evidence from each modality is superior to stuffing all inputs into a single context window.

## Limitations & Future Work
- Experimental scale is constrained by multi-agent inference costs. To cover more tasks and baselines, the authors used computable subsets of each dataset rather than full data.
- Current evaluations are primarily classification tasks, as LLM-based multimodal sensing lacks standard benchmarks covering a broader range of task types; subjective and open-ended generative sensing reasoning have not been fully tested.
- ConSensus does not stack Self-Consistency, Self-Refine, or more advanced confidence-aware debate, meaning its upper bound could be further increased.
- Future work could explicitly model sensor reliability, such as through confidence-weighted voting, using tools to estimate signal quality, or leveraging historical sensing streams to learn modality reliability.

## Related Work & Insights
- **vs Single-Agent Sensing Reasoning**: Single agents concatenate all features into one prompt, often missing modality evidence; ConSensus ensures each signal path is independently interpreted via modality agents.
- **vs Multi-Agent Debate**: Debate, MAD, and ReConcile rely on multiple interactions, which are effective but token-intensive; ConSensus uses fixed fusion roles for one-time aggregation, making it more suitable for resource-constrained deployment.
- **vs Traditional Supervised Sensor Fusion**: Traditional methods usually requires task-specific training data; ConSensus uses the world knowledge and prompt protocols of pre-trained LLMs for training-free inference, though it relies on the LLM's capability to understand sensing features textually.
- **Inspiration for Other Tasks**: Multimodal medical diagnosis, autonomous driving, and robotic state estimation can all benefit from the "modality-specific interpretation + statistical anchor + semantic arbitration" paradigm.

## Rating
- Novelty: ⭐⭐⭐⭐ Specifically applies multi-agent collaboration to heterogeneous sensor fusion and explicitly distinguishes between semantic and statistical biases with a clear design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 5 datasets, 12 sensor modalities, multiple backbones, and missing modality experiments; limited only by computational cost from being a full-scale evaluation.
- Writing Quality: ⭐⭐⭐⭐ Coherent motivation and observations with informative tables.
- Value: ⭐⭐⭐⭐ Highly valuable for training-free sensing reasoning and multi-agent systems, especially in low-annotation and low-training-budget scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] MMedAgent-RL: Optimizing Multi-Agent Collaboration for Multimodal Medical Reasoning](../../ICLR2026/multi_agent/mmedagent-rl_optimizing_multi-agent_collaboration_for_multimodal_medical_reasoni.md)
- [\[ACL 2026\] PosterForest: Hierarchical Multi-Agent Collaboration for Scientific Poster Generation](posterforest_hierarchical_multi-agent_collaboration_for_scientific_poster_genera.md)
- [\[ACL 2026\] Scaling External Knowledge Input Beyond Context Windows of LLMs via Multi-Agent Collaboration](scaling_external_knowledge_input_beyond_context_windows_of_llms_via_multi-agent_.md)
- [\[AAAI 2026\] LLandMark: A Multi-Agent Framework for Landmark-Aware Multimodal Interactive Video Retrieval](../../AAAI2026/multi_agent/llandmark_a_multi-agent_framework_for_landmark-aware_multimodal_interactive_vide.md)
- [\[NeurIPS 2025\] Belief-Calibrated Multi-Agent Consensus Seeking for Complex NLP Tasks](../../NeurIPS2025/multi_agent/belief-calibrated_multi-agent_consensus_seeking_for_complex_nlp_tasks.md)

</div>

<!-- RELATED:END -->

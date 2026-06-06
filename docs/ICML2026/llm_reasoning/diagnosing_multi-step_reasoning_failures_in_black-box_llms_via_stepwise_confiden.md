---
title: >-
  [Paper Note] Diagnosing Multi-step Reasoning Failures in Black-box LLMs via Stepwise Confidence Attribution
description: >-
  [ICML2026][LLM Reasoning][Step-level Confidence] This paper formalizes the task of "identifying which step in a CoT reasoning chain is incorrect" as a step-level confidence attribution problem in a black-box setting. By…
tags:
  - "ICML2026"
  - "LLM Reasoning"
  - "Step-level Confidence"
  - "Information Bottleneck"
  - "Consensus Graph"
  - "Black-box LLM"
  - "Self-correction"
date: 2026-05-08
content_hash: cdde774122779efb
---

# Diagnosing Multi-step Reasoning Failures in Black-box LLMs via Stepwise Confidence Attribution

**Conference**: ICML2026  
**arXiv**: [2605.19228](https://arxiv.org/abs/2605.19228)  
**Code**: https://anonymous.4open.science/r/ICML_2026_step_wise-2D45  
**Area**: LLM Reasoning / Confidence Estimation / Reasoning Diagnosis  
**Keywords**: Step-level Confidence, Information Bottleneck, Consensus Graph, Black-box LLM, Self-correction

## TL;DR
This paper formalizes the task of "identifying which step in a CoT reasoning chain is incorrect" as a step-level confidence attribution problem in a black-box setting. By applying the Information Bottleneck (IB) principle, the authors compress correct reasoning trajectories obtained from multiple samplings of the same problem into a consensus structure. They introduce two instances: the training-free NIBS (Semantic Consensus Alignment) and the learnable GIBS (Graph Consensus Subgraph Selection). Both methods consistently outperform white-box baselines on GSM8K, Math, and MoreHopQA, improving self-correction success rates by up to 13.5% through step-level feedback.

## Background & Motivation

**Background**: Long-chain reasoning (CoT / GoT) has become the mainstream paradigm for LLM problem-solving. To judge the trustworthiness of a reasoning chain, the industry mainly follows two paths: training Process Reward Models (PRMs like PRM800K, Math-Shepherd) using human step-by-step annotations, or directly using LLMs for self-evaluation (LLM-as-judge). Confidence Estimation (CE) provides a third path, but most CE methods target only the final answer—providing "is the whole chain correct"—rather than pinpointing "which step is wrong."

**Limitations of Prior Work**: The few attempts at step-level CE, such as LeCo and SL(norm), require access to token-level logits or entropy, making them essentially white-box methods unavailable for closed-source APIs like GPT-4o or Claude. Simply decomposing answer-level CE to each step faces a new challenge: correct solutions for the same problem can vary significantly in phrasing order and step granularity (e.g., different but valid solution sequences in Figure 1). Naïve similarity comparisons would misjudge "valid variations" as "errors."

**Key Challenge**: Under black-box constraints, only the generated text is available, and one must distinguish "legitimate variations" from "true errors"—the former being surface differences, while the latter represents logical drift.

**Goal**: To assign a reliable confidence score to each reasoning step reflecting its "contribution to final correctness," given only the generated trajectories and the binary label of the final answer's correctness.

**Key Insight**: The authors observe that while the order of correct solutions is variable, they all pass through several "logical invariants" (e.g., intermediate costs in math problems, key entity values in multi-hop QA). Error-riddled solutions deviate from this consensus structure. Thus, the degree of alignment with a consensus derived from multiple correct trajectories can serve as a proxy signal for confidence.

**Core Idea**: Use the Information Bottleneck $\min_Z I(T_i;Z) - \beta I(Z;Y)$ to formalize the intuition of "compressing redundant phrasing while preserving consensus sub-structures related to correctness," where $Y$ is approximated by consensus anchors aggregated from correct trajectories.

## Method

### Overall Architecture
Given a problem $x$, $N=20$ reasoning trajectories $\mathcal{S}=\{(T_i, A_i, z_i)\}$ are first sampled from the LLM at temperature 1.0, where $z_i\in\{0,1\}$ indicates whether the final answer matches the gold standard (Exact Match for math, GPT-4o judge for QA). The pipeline then follows three stages: (A) Parsing each text trajectory into a directed reasoning graph $G_i=(V_i,E_i)$, where nodes $v_{ij}$ are intermediate results and edges $e_{ij}$ are sub-problems or operations (parsed using LangFun-style prompts and rules); (B) Aggregating "consensus anchors" from $\mathcal{S}_{\text{correct}}$—NIBS uses semantic similarity sets directly, while GIBS computes a Maximum Common Subgraph to obtain a mask $\mathbf{m}_i$; (C) Using the IB objective to produce step-level scores $c_{ij}$, where steps with low scores are flagged as suspicious.

### Key Designs

1.  **IB Formalization + Consensus Anchors as Proxies for Unobservable Step Labels**:
    - **Function**: Replaces the unobservable "step-level correctness" target $Y$ with a computable proxy signal, making IB solvable in black-box scenarios with only answer-level labels.
    - **Mechanism**: Aggregates steps or sub-structures that appear in "almost all correct solutions" from $\mathcal{S}_{\text{correct}}$ to serve as proxy $Y$. In the formal objective $\min_Z I(T_i;Z) - \beta I(Z;Y)$, the compression term $I(T_i;Z)$ pushes for retaining fewer steps, while the relevance term $I(Z;Y)$ pushes for the retained steps to align with the consensus.
    - **Design Motivation**: Existing PRM routes require expensive human step-level annotation, and LLM-as-judge inherits its own biases. Using "consensus among correct solutions after multiple samplings" as supervision is zero-cost and naturally available in black-box settings.

2.  **NIBS: Training-free Non-parametric Consensus Alignment**:
    - **Function**: Calculates a training-free confidence score for each step $t_{ij}$ of a trajectory $T_i$ as a closed-form approximation of the IB solution.
    - **Mechanism**: By taking $Z$ as the "set of steps appearing in multiple correct solutions," the confidence reduces to $c_{ij}=\mathbb{E}_{S\sim\mathcal{S}_{\text{correct}}}[\text{Agg}(\{\text{sim}(\mathbf{t}_{ij},\mathbf{t}')|\mathbf{t}'\in S\})]$, where $\text{sim}$ can be BERT cosine or NLI entailment, and $\text{Agg}$ is max or mean. The algorithm has no trainable parameters.
    - **Design Motivation**: As a strong baseline, it validates that "consensus = confidence" is a highly informative signal and provides a simple, ready-to-deploy solution for cases with zero training budget.

3.  **GIBS: Graph IB + Subgraph Selection via Differentiable Mask**:
    - **Function**: Captures structural dependencies ignored by NIBS—two steps with similar semantics but completely different positions in the reasoning graph should not be treated identically.
    - **Mechanism**: Builds a directed graph $G_i$ for each trajectory. Confidence $Z$ is instantiated as a soft subgraph $G^*=G_i\odot \mathbf{p}_\theta$, where $\mathbf{p}_\theta$ is predicted by fusing BERT step features and 2-layer GCN structural features. Consensus supervision comes from a mask $\mathbf{m}_i$ aggregated from the Maximum Common Subgraph of $G_i$ and each correct graph. Variational IB relaxes the objective to $\mathcal{L}=H(\mathbf{p}_\theta)+\lambda\,\text{CE}(\mathbf{p}_\theta,\mathbf{m}_i)$, where the entropy term handles compression/sparsity (pushing $p_{\theta,ij}$ toward 0 or 1), and the CE term handles relevance. At inference, $c_{ij}=p_{\theta,ij}$.
    - **Design Motivation**: Directly calculating MI on discrete subgraphs is computationally explosive and non-differentiable. Soft masks and variational upper bounds make training feasible. GCN introduces structural context, allowing the model to learn "logical patterns" rather than just "lexical similarity," which provides OOD robustness.

### Loss & Training
GIBS is trained on 2,000 reasoning graphs. The loss is the sum of the entropy and CE terms from Equation (6). The variational prior is an independent Bernoulli($\epsilon<0.5$). For evaluation, an average of 10,000 trajectories per dataset is used. NIBS is entirely training-free.

## Key Experimental Results

### Main Results

Tested across 3 LLMs (Llama3.1-8B, DeepSeek-R1-Distill-Qwen-32B, Phi4-Reasoning) and 3 datasets (GSM8K, MoreHopQA, Math) using 4 metrics (AUROC↑, AUCPR↑, ACC@80%↑, ECE↓). GIBS achieves the best performance in 7 out of 9 AUROC configurations.

| Dataset | LLM | Strongest White-box Baseline (AUROC) | NIBS Best (AUROC) | GIBS (AUROC) |
|--------|------|----------------------|-------------------|--------------|
| GSM8K | Phi4-Reasoning | NLI-Max 0.660 | NLI-Max 0.660 | **0.789** |
| MoreHopQA | DeepSeek-R1-32B | NLI-Max 0.666 | NLI-Max 0.666 | **0.808** |
| Math | Phi4-Reasoning | Cos-Mean 0.612 | Cos-Mean 0.612 | **0.695** |
| GSM8K | Llama3.1-8B | NLI-Max 0.710 | NLI-Max 0.710 | 0.691 |

NIBS (especially Cos-Mean / NLI-Max) significantly outperforms white-box methods like P(true), Entropy, and LECO. GIBS shows even more pronounced improvements on complex Math and MoreHopQA datasets.

### Ablation Study
Phi4-Reasoning, AUROC:

| Configuration | GSM8K | MoreHopQA | Math | Note |
|------|-------|-----------|------|------|
| Full GIBS | 0.789 | 0.662 | 0.695 | Includes edge & graph encoders |
| w/o Graph Encoder | 0.723 | 0.648 | 0.596 | Math drops by 0.10 due to loss of global structure |
| w/o Edge Encoder | 0.519 | 0.376 | 0.476 | Performance collapses without edges, proving logical edges are key |

Consensus source ablation (GIBS, MoreHopQA AUROC): Correct-only 0.808 > Self-consistency 0.784 > All trajectories 0.648. Pseudo-label quality is strongly correlated with final performance.

### Key Findings
- **Edge encoding is more critical than node encoding**: Removing the edge encoder leads to an average AUROC drop of 0.26 across three datasets, much larger than the 0.07 drop from removing the graph encoder. This suggests that "logical dependencies between steps" are the defining feature for consensus structures.
- **MCS of correct solutions clusters at 0.8, while incorrect ones cluster at 0.4** (Statistics from 1,000 graphs in Figure 3), providing empirical support for the "consensus = correctness" hypothesis.
- **Step-level feedback is significantly stronger than answer-level feedback**: For initially incorrect samples in MoreHopQA, GIBS step-level feedback increases self-correction success by up to 13.5%. Strong reasoning models (DeepSeek-R1, Phi4) benefit more as they can better utilize localization signals.
- **OOD Robustness**: When trained on MoreHopQA and tested directly on Math, GIBS still outperforms NIBS and white-box baselines. This is attributed to IB and graph structures learning "abstract reasoning patterns" rather than dataset-specific vocabulary.

## Highlights & Insights
- Ours first formalizes "stepwise confidence attribution" in a black-box setting and provides a clean optimization objective via IB, which is lighter than training PRMs or using LLM-as-judge.
- The "using consensus of multiple samplings to replace unobservable labels" is a highly portable trick applicable to any problem where target variables are only visible at the sequence level but attribution is needed internally (e.g., generation evaluation, agent trajectory scoring, RL credit assignment).
- The training-free NIBS (NLI-Max AUROC=0.710 on GSM8K with Llama3.1-8B) doubles the performance of white-box P(true) (0.40) and Entropy (0.41). This indicates that white-box token probabilities correlate poorly with logical step correctness, whereas collective consensus signals are much more reliable.
- Variational IB elegantly uses the entropy term for both "sparsity + binarization," avoiding extra sparsity hyperparameters.

## Limitations & Future Work
- Dependency on $N=20$ samplings increases inference costs by an order of magnitude, which is a significant trade-off for latency-sensitive or API-cost-sensitive deployments.
- The quality ceiling of the method is determined by the consensus anchors. If a problem is so difficult that almost no correct solutions appear in 20 samplings (as observed with Llama3.1-8B on certain tasks), the method degrades.
- Reasoning graph construction depends on LangFun-style prompts and rule-based parsing. Generalizing to free-form text or unstructured reasoning requires redesigning the parser.
- Evaluation covers only objectively scorable tasks (Math + multi-hop QA). Moving to open-ended tasks (creative writing, summarization) remains a future direction, though a self-consistency alternative is suggested in Section 5.5.

## Related Work & Insights
- **vs. PRM route (PRM800K, Math-Shepherd)**: Those rely on human or synthetic step-level labels to train classifiers. Ours replaces labels with "sampling consensus," offering zero human cost but requiring multiple samplings. GIBS also performs well on PRM800K (Appendix F).
- **vs. LLM-as-judge (Weng 2023, Li 2024)**: Using LLMs as judges can inherit their biases and produce inconsistent outputs for the same problem. Ours uses group consensus as a more objective proxy.
- **vs. White-box stepwise CE (LeCo, SL(norm), Token Entropy)**: They require logits; ours is completely black-box. Experiments show white-box token signals have weak predictive power for logical correctness compared to consensus-based signals.
- **vs. Graph-of-Thought (GoT)**: Those focus on organizing the reasoning process itself as a graph. Ours uses the graph as a tool to align with consensus structures. The methods are orthogonal and can be combined.

## Rating
- Novelty: ⭐⭐⭐⭐ First to formalize black-box stepwise CE via IB and consensus alignment.
- Experimental Thoroughness: ⭐⭐⭐⭐ 3 LLMs × 3 datasets × 4 metrics + Ablation + Label-free alternatives + OOD + PRM800K validation.
- Writing Quality: ⭐⭐⭐⭐ Clear IB derivation, intuitive examples in Figure 1, well-explained engineering details.
- Value: ⭐⭐⭐⭐ 13.5% improvement in self-correction is an attractive downstream gain. Being API-only makes it highly accessible.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Note 6: Self-Evaluating LLMs - Step-Level Confidence Estimation for Multi-Step Tasks](../../NeurIPS2025/llm_reasoning/value-guided_search_for_efficient_chain-of-thought_reasoning.md)
- [\[ICML 2026\] Inducing Overthink: Hierarchical Genetic Algorithm-based DoS Attack on Black-Box Large Language Reasoning Models](inducing_overthink_hierarchical_genetic_algorithm-based_dos_attack_on_black-box_.md)
- [\[ICML 2026\] Inference Time Optimization with Confidence Dynamics](inference_time_optimization_with_confidence_dynamics.md)
- [\[ACL 2026\] Merlin's Whisper: Enabling Efficient Reasoning in Large Language Models via Black-box Persuasive Prompting](../../ACL2026/llm_reasoning/merlin39s_whisper_enabling_efficient_reasoning_in_large_language_models_via_blac.md)
- [\[CVPR 2026\] Step-CoT: Stepwise Visual Chain-of-Thought for Medical Visual Question Answering](../../CVPR2026/llm_reasoning/step-cot_stepwise_visual_chain-of-thought_for_medical_visual_question_answering.md)

</div>

<!-- RELATED:END -->

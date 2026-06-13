---
title: >-
  [Paper Note] Illusions of Confidence? Diagnosing LLM Truthfulness via Neighborhood Consistency
description: >-
  [ACL 2026][LLM Safety][Belief Robustness] This paper points out that for LLMs, "high self-consistency does not equate to true belief"—accuracy on 995 questions originally answered correctly with 100% consistency plummete…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Belief Robustness"
  - "Neighborhood Consistency"
  - "Self-Consistency"
  - "Bayesian Belief"
  - "Structure-Aware Training"
date: 2026-05-08
content_hash: e4dd6df30fe2bbd4
---

# Illusions of Confidence? Diagnosing LLM Truthfulness via Neighborhood Consistency

**Conference**: ACL 2026  
**arXiv**: [2601.05905](https://arxiv.org/abs/2601.05905)  
**Code**: https://github.com/zjunlp/belief (Available)  
**Area**: LLM Reasoning / Calibration / Trustworthiness  
**Keywords**: Belief Robustness, Neighborhood Consistency, Self-Consistency, Bayesian Belief, Structure-Aware Training

## TL;DR
This paper points out that for LLMs, "high self-consistency does not equate to true belief"—accuracy on 995 questions originally answered correctly with 100% consistency plummeted to 33.8% when minor contextual interference was introduced. The authors propose **Neighbor-Consistency Belief (NCB)**, which uses the joint consistency of a target fact and its "conceptual neighbors" (premises/entailments/topics) as a structured proxy for belief robustness. Based on Asch's conformity experiments and Source Credibility theory, they designed a cognitive stress-test protocol, proving on four LLMs that high-NCB data is significantly more resistant to interference. Furthermore, they propose **Structure-Aware Training (SAT)**, which employs teacher-student KL distillation to enforce consistent output across different neighborhood contexts, enhancing the robustness of newly learned knowledge by approximately 30% over Ans/Know augmentation baselines.

## Background & Motivation

**Background**: To evaluate whether an LLM "knows" a fact, the mainstream approach relies on self-consistency (vote-based consistency across multiple samples) or token-level confidence. However, LLMs are increasingly deployed in scenarios like RAG, multi-agent collaboration, and complex prompt engineering where they are "led by external context." In these settings, "knowing" is insufficient; the model must remain stable under interference.

**Limitations of Prior Work**: The authors conducted a pilot study using Qwen3-30B-A3B on 995 questions with a self-consistency of 1.0 (30/30 correct samples). After inserting a single peer disagreement, accuracy dropped from 100% to 33.8%. This indicates that existing confidence metrics cannot distinguish between "answering correctly via isolated memory fragments" and "answering based on structured beliefs."

**Key Challenge**: Belief should be a **structured latent state** (in cognitive science, the human brain organizes knowledge via semantic networks where related facts constrain each other to resist interference), whereas point-wise metrics like self-consistency only observe consistency across multiple outputs of the same question, failing to capture the network structure between facts.

**Goal**: (a) Provide a computational metric to distinguish between "answering via structural belief" and "answering via isolated memory"; (b) Design a rigorous cognitive stress-test to verify that this metric predicts robustness; (c) Use "structural invariance" as a training objective to ensure learned knowledge is more resistant to interference.

**Key Insight**: Model belief as a binary latent variable $\theta \in \{\mathcal{S}_\text{struct}, \mathcal{S}_\text{unstruct}\}$ and use Bayesian posterior estimation—if the model answers a set of neighboring facts correctly, its posterior for being in a structured belief state is significantly higher than an unstructured one; this posterior is approximated as the NCB score.

**Core Idea**: Use "neighbor consistency" instead of "self-consistency" as a proxy for belief strength and explicitly incorporate this structural invariance into the training loss.

## Method

### Overall Architecture
The paper is divided into three phases:

1.  **Constructing the Neighbor-Enriched Dataset**: 2,000 time-invariant facts were balanced across STEM, Arts & Culture, Social Sciences, and Sports, sampled from SimpleQA, HotpotQA, and SciQ. For each target $(q^*, \mathcal{E}^*)$, DeepSeek-V3.2 generated Neighbor Facts (NFs) covering three relationship types: entity premises, logical entailments, and thematic associations (average 7.84 NFs/fact), followed by manual filtering. Misleading Entities $\mathcal{E}^\dagger$ and their Misleading Neighbor Facts (MNFs) (average 4.88/fact) were constructed for interference.
2.  **NCB Measurement and Stress-Test Evaluation**: For each fact, 30 target responses and 10 responses per neighbor were sampled ($T=0.7$), and NCB was estimated via Empirical Correctness Frequency. Models were then re-tested under two types of interference: Peer Quantity (Asch conformity) and Source Credibility (authoritative sources), across Standard, CoT, and Reflection reasoning strategies.
3.  **Structure-Aware Training**: Based on a checkpoint after initial answer augmentation (Ans. Aug), a teacher model views the raw question while the student model views "Question + Neighborhood Context $C_{nq}$ or General Noise Context $C_\text{general}$." KL distillation forces the student's output distribution to align with the teacher's distribution across different contexts.

### Key Designs

1.  **Neighbor-Consistency Belief (NCB) Metric**:
    *   **Function**: Approximates the posterior probability that a fact belongs to a structured belief state using a scalar, distinguishing "true understanding" from "lucky memory."
    *   **Mechanism**: Defines a latent variable $\theta \in \{\mathcal{S}_\text{struct}, \mathcal{S}_\text{unstruct}\}$ and formulates the posterior as $P(\theta = \mathcal{S}_\text{struct} \mid \hat{\mathcal{E}}^* = \mathcal{E}^*, \forall i, \hat{a}_i = a_i)$. Using Bayes' theorem, the odds are decomposed into Bayes Factor × Prior Odds. By assuming $P((\forall i, \hat a_i = a_i) \mid \hat{\mathcal{E}}^* = \mathcal{E}^*, \mathcal{S}_\text{struct}) \gg P(\cdot \mid \mathcal{S}_\text{unstruct})$, it is shown that odds $\gg 1$. The unobservable posterior is approximated by the aggregation of Empirical Correctness Frequency $\hat p(\hat a = a \mid q)$ over $\mathcal{O} = \{(q^*, \mathcal{E}^*)\} \cup NFs$.
    *   **Design Motivation**: Grounded in neurocognitive science (interlocking semantic networks, Anderson’s inhibitory control theory) and the concept of "anchoring in context" from knowledge editing literature—redefining "belief = isolated facts" as "belief = a structured neighbor network" to explain why interference easily disrupts answers with high self-consistency.

2.  **Cognitive Stress-Test Protocol (Asch + Source Credibility)**:
    *   **Function**: Operationalizes belief stability under external interference into quantifiable experiments.
    *   **Mechanism**: (i) **Peer Quantity** simulates the Asch conformity experiment—the model observes dialogues from peer agents before answering $q^*$. Scenarios include Conflict (peers directly give $\mathcal{E}^\dagger$) and Misleading (peers discuss MNFs, indirectly priming the wrong answer), with the number of peer agents $N \in [1, 10]$. (ii) **Source Credibility** simulates the Hovland effect—interference text is categorized by Low (social media), Medium (blogs), and High (academic/news) authority. Accuracy drops are analyzed by grouping high vs. low NCB into 5%/20%/35% buckets.
    *   **Design Motivation**: Borrowing classical paradigms from 1970s cognitive psychology ensures ecological validity and provides clear controllable axes (number of peers, authority level).

3.  **Structure-Aware Training (SAT)**:
    *   **Function**: Uses belief structural invariance as a training objective to ensure newly learned facts remain stable within their neighborhood context.
    *   **Mechanism**: The teacher $\theta_T$ is frozen, and the student $\theta_S$ is trainable, both initialized from an Ans. Aug checkpoint. For each fact, two types of context are synthesized: $C_{nq}$ (neighbor-related) and $C_\text{general}$ (general noise). The student's output distribution $P_{\theta_S}(y \mid C, x)$ is aligned via KL divergence with the teacher's context-free distribution $P_{\theta_T}(y \mid x)$: $\mathcal{L}_\text{KD} = \frac{1}{|C_b|}\sum_{(c, x) \in C_b} D_\text{KL}(P_T \parallel P_S)$.
    *   **Design Motivation**: Traditional SFT only encourages the model to memorize $(q, a)$ pairs without enforcing consistency under noise. SAT explicitly injects this robustness constraint, shifting the belief representation from point-wise to context-invariant at the loss level.

### Loss & Training
In SAT, the student only optimizes the KL loss (unsupervised with respect to hard labels), effectively training the student to mimic the teacher’s interference-free distribution under any context $c$. Both teacher/student are based on Qwen-2.5-32B-Instruct. Stress-Test details: 30 target samples + 10 neighbor samples per fact, $T=0.7$, bf16 + vLLM, 8×A100.

## Key Experimental Results

### Main Results

Stress-Test across 4 LLMs using the Standard setting (selecting top/bottom 35% NCB subsets). Values represent "Accuracy drop after Stress" (Baselines are near 100%):

| Model | NCB Group | Quantity-Stress Standard | Source-Stress Standard | Reflection (Source) |
|------|--------|--------------------------|--------------------------|-----------------------|
| Qwen-2.5-32B | Low NCB-35% | 74.0 (↓25.7) | 79.2 (↓20.5) | 78.7 (↓20.9) |
| Qwen-2.5-32B | High NCB-35% | **84.0 (↓16.0)** | **87.2 (↓12.8)** | **84.5 (↓15.5)** |
| Qwen3-30B-A3B | Low NCB-35% | 70.8 (↓28.8) | 75.2 (↓24.3) | 84.1 (↓15.4) |
| Qwen3-30B-A3B | High NCB-35% | **82.4 (↓17.6)** | **85.4 (↓14.6)** | **90.2 (↓9.8)** |
| Qwen3-30B-Thinking | Low NCB-35% | 77.3 (↓22.6) | 77.8 (↓22.1) | 84.7 (↓15.3) |
| Qwen3-30B-Thinking | High NCB-35% | **88.1 (↓11.3)** | **87.1 (↓12.3)** | **93.7 (↓5.8)** |
| OLMo-2-32B | Low NCB-35% | 71.4 (↓28.3) | 80.3 (↓19.3) | 85.1 (↓14.5) |
| OLMo-2-32B | High NCB-35% | **81.3 (↓18.7)** | **88.2 (↓11.8)** | **89.8 (↓10.2)** |

Across all four models, the accuracy drop for the high NCB group is consistently ~50%–70% of that of the low NCB group.

### Ablation Study

SAT vs. two SFT augmentation baselines (Qwen-2.5-32B-Instruct, 100 facts originally answered incorrectly):

| Metric | Vanilla (Untrained) | Ans. Aug | Know. Aug | **SAT (Ours)** |
|------|----------------|----------|------------|------------------|
| Base ACC | 4.8 | 92.4 | 85.4 | **93.0** |
| Quantity Stress | 8.2 | 20.1 | 31.0 | **58.1** |
| Source Stress | 4.6 | 41.6 | 35.7 | **63.0** |
| Average Stress | 6.4 | 30.9 | 33.4 | **60.6** |
| MMLU | 72.84 | 82.9 | 81.1 | 80.1 |
| GSM8k | 91.66 | 91.5 | 88.8 | 91.0 |

SAT pushes Average Stress from 33.4 to 60.6 (an **~80% relative gain**) without reducing Base ACC, while general capabilities (MMLU/GSM8k) remain largely unchanged.

### Key Findings
-   **Finding 1 — NCB is a reliable metric for belief robustness**: All four models show significantly smaller drops in the high NCB group, with the most pronounced contrast in Qwen3-Thinking (↓11.3% vs ↓22.6%). Coverage analysis revealed that Qwen3-Thinking tends to "self-refuse" on low NCB samples, suggesting reasoning models are aware of their "unstructured" knowledge.
-   **Finding 2 — Structural beliefs remain stable as interference intensity increases**: Incremental peer conflict (from 0 to 6 opposing votes) caused low NCB accuracy to crash (97%→62%), while high NCB only declined gradually (98%→81%). Asch's classic conclusion was replicated: a single truth-teller (cfg5) significantly reduces conformity pressure.
-   **Finding 3 — CoT is unstable, Reflection wins**: CoT often **amplifies** the drop (e.g., Qwen-2.5 Low NCB-35% worsened from ↓25.7% to ↓31.6%), whereas Reflection significantly mitigates it. CoT also showed a non-linear "Latitude of Rejection" effect, where accuracy dropped most at moderate interference levels.
-   **Finding 4 — Model scaling does not resolve belief fragility**: Increasing Qwen-2.5 from 1.5B to 72B did not close the robustness gap between high and low NCB, suggesting this is not a problem solvable by scale alone.
-   **SAT's 30% reduction in fragility is a free lunch**: MMLU/GSM8k scores remained stable while stress test performance improved significantly, indicating that "structural invariance" can be injected independently of general capabilities.

## Highlights & Insights
-   The conceptual shift from **point-wise to graph-wise belief evaluation** is the most significant contribution: this serves as a wake-up call to engineers relying solely on "high confidence"—high confidence is not synonymous with knowing.
-   Using **Asch and Source Credibility as stress-test protocols** provides an elegant template for "Cognitive Psychology × LLM" research, showing how 70-year-old experimental designs provide ready-made schemas for controllable evaluation.
-   The **SAT training paradigm (Teacher-Student KL + context augmentation)** is a highly reusable trick for RAG fine-tuning, adversarial robustness, and persona consistency.
-   The **formal derivation of NCB using Bayesian Odds** provides a grounded mathematical definition for the psychological concept of "structural belief," making it more persuasive and easier to analyze theoretically than heuristic scoring.

## Limitations & Future Work
-   The Neighbor Facts only cover three relation types (premises, entailment, association), excluding causal chains or hierarchical taxonomies, and are limited to time-invariant facts.
-   NCB lacks direct validation against human judgments of "true understanding"; it currently serves as a proxy for robustness rather than a direct measure of human-like comprehension.
-   Constructing belief neighborhoods introduces significant computational overhead in both training and inference; future work should optimize this via neighborhood caching or selective sampling.

## Related Work & Insights
-   **vs. Self-Consistency (Wang et al., 2023a)**: SC is point-wise; this paper proves it "systematically overestimates robustness."
-   **vs. Semantic Entropy (Farquhar et al., 2024)**: While surpassing token-level probability, it remains point-wise. NCB is the first to extend belief to conceptual neighborhoods.
-   **vs. Knowledge Editing Brittleness (Anthropic SDF 2025)**: Previous work noted that new knowledge is more fragile; this paper provides a structural explanation (lack of neighbor consistency) and the SAT solution.
-   **vs. Conformity in Multi-agent Systems (Zhang et al. 2024)**: This work quantifies conformity intensity using the Asch paradigm and finds that the "dissenter effect" still holds for LLMs.

## Rating
-   Novelty: ⭐⭐⭐⭐⭐ Shifts evaluation from point-wise to graph-wise; elegant use of psychological paradigms.
-   Experimental Thoroughness: ⭐⭐⭐⭐ Broad coverage across 4 LLMs and multiple strategies; lacks SAT validation on ultra-large models (70B+).
-   Writing Quality: ⭐⭐⭐⭐ Clear logic across concept, formula, and experiment.
-   Value: ⭐⭐⭐⭐⭐ Addresses the critical "illusion of confidence" issue with quantifiable metrics and effective training solutions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ADVICE: Answer-Dependent Verbalized Confidence Estimation](advice_answer-dependent_verbalized_confidence_estimation.md)
- [\[AAAI 2026\] The Confidence Trap: Gender Bias and Predictive Certainty in LLMs](../../AAAI2026/llm_safety/the_confidence_trap_gender_bias_and_predictive_certainty_in_llms.md)
- [\[NeurIPS 2025\] On the Robustness of Verbal Confidence of LLMs in Adversarial Attacks](../../NeurIPS2025/llm_safety/on_the_robustness_of_verbal_confidence_of_llms_in_adversarial_attacks.md)
- [\[ACL 2026\] Detoxification for LLM from Dataset Itself](detoxification_for_llm_from_dataset_itself.md)
- [\[ACL 2026\] Representation-Guided Parameter-Efficient LLM Unlearning](representation-guided_parameter-efficient_llm_unlearning.md)

</div>

<!-- RELATED:END -->

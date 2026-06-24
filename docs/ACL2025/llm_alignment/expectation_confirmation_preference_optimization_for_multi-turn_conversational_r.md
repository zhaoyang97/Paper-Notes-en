---
title: >-
  [Paper Note] Highlights & Insights
description: >-
  [ACL 2025][LLM Alignment][conversational recommendation] This paper proposes ECPO (Expectation Confirmation Preference Optimization), the first multi-turn preference optimization method for LLM-driven conversational recommendation Agents. Based on the psychological Expectation Confirmation Theory (ECT), it explicitly models the evolution of user satisfaction across multiple turns. It locates the root causes of dissatisfaction through forward expectation confirmation and recon…
tags:
  - "ACL 2025"
  - "LLM Alignment"
  - "conversational recommendation"
  - "multi-turn preference optimization"
  - "expectation confirmation theory"
  - "user simulator"
  - "DPO"
  - "LLM agent"
date: 2026-05-08
content_hash: 5ea5d6da98c6750c
---

# Expectation Confirmation Preference Optimization for Multi-Turn Conversational Recommendation Agent

**Conference**: ACL 2025  
**arXiv**: [2506.14302](https://arxiv.org/abs/2506.14302)  
**Code**: None  
**Area**: LLM Alignment  
**Keywords**: conversational recommendation, multi-turn preference optimization, expectation confirmation theory, user simulator, DPO, LLM agent

## TL;DR

This paper proposes ECPO (Expectation Confirmation Preference Optimization), the first multi-turn preference optimization method for LLM-driven conversational recommendation Agents. Based on the psychological Expectation Confirmation Theory (ECT), it explicitly models the evolution of user satisfaction across multiple turns. It locates the root causes of dissatisfaction through forward expectation confirmation and reconstructs turn-level preference pairs by rewriting responses via backward expectation derivation. Combined with the AILO user simulator, ECPO significantly outperforms existing MTPO methods across three datasets.

## Background & Motivation

**Background**: LLM-driven Conversational Recommendation Agents (CRAs) have emerged as a dominant paradigm for personalized recommendation, gradually mining user interests and suggesting items through multi-turn natural language interactions. Existing CRAs include prompt-based frameworks such as ChatRec, ReAct, and MACRS.

**Limitations of Prior Work**: The pre-training objective of LLMs is shortsighted next-token prediction, which often leads to CRAs generating responses that lack proactivity and flexibility. This makes it difficult to continuously guide users and meet their expectations throughout multi-turn interactions. Although preference optimization (e.g., DPO/KTO) has proven effective in single-turn alignment, applying it directly to multi-turn dialogues poses severe challenges.

**Key Challenge**: In multi-turn dialogues, user preferences shift dynamically turn by turn. Existing MTPO methods suffer from three major issues: (1) Tree-search-based methods require sampling multiple candidates at each turn and simulating full dialogues to evaluate intermediate turn preferences, incurring massive sampling overhead; (2) LLMs struggle to generate high-quality positive samples through self-sampling; (3) The randomness of the simulation environment introduces noise into preference relationships, degrading the alignment performance.

**Goal**: To build high-quality, turn-level preference relationships and achieve efficient alignment for multi-turn CRAs without requiring extra sampling and evaluation.

**Key Insight**: Leveraging the psychological Expectation Confirmation Theory (ECT), which posits that satisfaction results from comparing initial user expectations with actual perception. By explicitly modeling the "expectation-confirmation" process of each turn, the method traces root causes when satisfaction is low and rewrites responses to naturally construct preference pairs.

**Core Idea**: Driving turn-level satisfaction evaluation via ECT to convert natural language explanations of "why the user is unsatisfied" into rewriting guidelines, obtaining high-quality preference pairs without extra sampling.

## Method

### Overall Architecture

ECPO consists of a four-step pipeline:
1. **Simulator-Guided Planning Tuning (SGPT)**: GPT-4o mini is employed as an expert CRA to interact with a user simulator. Successful recommendation trajectories are collected as SFT data $\mathcal{D}_{sft}$ to train $\pi_{sft}$.
2. **Forward Expectation Confirmation**: During dialogues between $\pi_{sft}$ and AILO, an EC evaluation is performed at each turn. Scores are assigned across three dimensions: flexibility (0-2), consistency (0-2), and user guiding capability (0-1), generating natural language confirmation explanations $\text{CONF}_t$ and satisfaction scores $r_t$.
3. **Backward Expectation Derivation**: For unsatisfied turns where $r_t \leq \lambda$, $\text{CONF}_t$ is used as guidance, and a Rewriter (an external LLM) is utilized with chain-of-thought to rewrite the response into $\tilde{p}_t$.
4. **Preference Optimization**: The original-rewritten response pairs $(p_t, \tilde{p}_t)$ are used to construct the preference dataset $\mathcal{D}_{pre}$, followed by DPO for turn-level preference optimization.

### Key Designs

#### 1. Forward Expectation Confirmation

- **Function**: In each dialogue turn, this simulates the user's internal monologue by feeding the expected item $i^E$ and CRA response $p_t$ into EC instructions to evaluate whether each dimension meets the target.
- **Design Motivation**: ECT indicates that satisfaction is the discrepancy between expectations and actual perceptions. By explicitly modeling this process, satisfaction scores and reasons for dissatisfaction can be provided at each turn, eliminating the need to wait for the entire conversation to finish before backtracking.
- **Mechanism**:
    - Three-dimensional scoring: Flexibility (0-2 points), consistency (0-2 points), and user-guiding capability (0-1 points), with a maximum score of 5.
    - Output: $\{\text{CONF}_t, r_t\} = U(I_{ect}(i^E, h_t, p_t))$
    - $\text{CONF}_t$ is a confirmation explanation in natural language, specifying the reasons for satisfaction/dissatisfaction.

#### 2. Backward Expectation Derivation

- **Function**: For turns with satisfaction below the threshold $\lambda$, this backtracks to the CRA's state at that turn and uses the confirmation explanation to perform counterfactual reasoning on "how to respond appropriately".
- **Design Motivation**: Confirmation explanations log the root causes of dissatisfaction, enabling the Rewriter to make precise, targeted modifications rather than rewriting everything from scratch.
- **Mechanism**:
    - $\tilde{p}_t = \text{Rewriter}(I_{bed}(s_t, p_t, \text{CONF}_t))$, when $r_t \leq \lambda$
    - The Rewriter adopts slow thinking (chain-of-thought) to reason before rewriting.
    - Restricting modifications to be **bounded** (limited) instead of complete rewriting, preserving proximity to the distribution of $\pi_{sft}$.
    - Preference dataset: $\mathcal{D}_{pre} = \{(s_t, p_t, \tilde{p}_t) \mid r_t < \lambda\}$

#### 3. AILO User Simulator

- **Function**: Constructing a high-fidelity LLM user simulator to support the EC process.
- **Design Motivation**: Involving real users is expensive and biased; existing user simulators (e.g., iEvalLM) generate user personas through simple random sampling, lacking diversity.
- **Mechanism**:
    - **User Persona Modeling**: Based on the consumer psychology AIO theory, defining four dimensions: Activities, Interests, Language, and Orientations; utilizing GPT-4o to infer user personas from real recommendation review datasets.
    - **Policy-driven User Simulation**: A three-step generation process: (1) generating response policies (e.g., "asking for recommendations"); (2) generating response content based on the policy; (3) executing the EC process.
    - Diversity Validation: The maximum ROUGE-L distribution of 100 personas is significantly lower than the RecAgent baseline.
    - Realism Validation: Human annotations on 50 groups of dialogue trajectories show that AILO achieves a 100% win rate against iEvalLM.

### Loss & Training

- **SFT Stage**: $\mathcal{L}_{SFT} = \mathbb{E}[-\log \pi_\theta(cr_t, p_t | s_t)]$, including internal reasoning $cr_t$ and response $p_t$.
- **DPO Stage**: $\mathcal{L}_{DPO}(\pi_\theta, \pi_{sft}) = \mathbb{E}[-\log\sigma(\beta \log\frac{\pi_\theta(\tilde{p}_t|s_t)}{\pi_{sft}(\tilde{p}_t|s_t)} - \beta \log\frac{\pi_\theta(p_t|s_t)}{\pi_{sft}(p_t|s_t)})]$
- ECPO is orthogonal and complementary to other preference optimization methods (e.g., KTO, SimPO) and can be seamlessly integrated.
- Backbone: Llama-3.1-8B-Instruct
- Training Data: 1,000 tasks for $\mathcal{D}_{sft}$, 500 tasks for $\mathcal{D}_{pre}$

## Key Experimental Results

### Main Results 1: Comparison with Existing Prompt-based CRAs

| Method | Backbone | Book SR | Book WR | Game SR | Game WR | Yelp SR | Yelp WR |
|------|----------|---------|---------|---------|---------|---------|---------|
| ChatRec | GPT-4o mini | 0.46 | 0.13 | 0.37 | 0.09 | 0.24 | 0.12 |
| ReAct | GPT-4o mini | 0.52 | 0.33 | 0.39 | 0.34 | 0.57 | 0.42 |
| MACRS | GPT-4o mini | 0.63 | 0.01 | 0.36 | 0.15 | 0.40 | 0.02 |
| ActCRS | Llama-3.1 | 0.34 | 0.28 | 0.07 | 0.46 | 0.22 | 0.38 |
| +SGPT | Llama-3.1 | 0.54 | 0.48 | 0.41 | 0.42 | 0.44 | 0.47 |
| **+ECPO** | **Llama-3.1** | **0.56** | **0.57** | **0.41** | **0.56** | **0.45** | **0.63** |

**Key Findings**: SGPT enables Llama to achieve a recommendation score (SR) comparable to GPT-4o; ECPO further boosts interactivity significantly (WR 0.56–0.63), substantially outperforming the best performance of GPT-4o mini.

### Main Results 2: Comparison with Existing MTPO Methods

- Trajectory-level methods (SFT, KTO): Limited improvements, failing to effectively capture turn-level preferences.
- Tree-simulation-level methods (SDPO, SKTO): Even with 2,500 sampled trajectories, negative gains still occur due to severe noise interference.
- **ECPO**: Requires only 500 trajectories (no extra sampling), achieving the best performance across all metrics including flexibility, consistency, and user guidance.
- In human evaluation, ECPO significantly excels over the expert CRA on all metrics, especially in flexibility and user guidance capability.

### Ablation Study

| Method | Book SR | Book WR | Game SR | Game WR | Yelp SR | Yelp WR |
|------|---------|---------|---------|---------|---------|---------|
| +SGPT | 0.54 | 0.48 | 0.41 | 0.42 | 0.44 | 0.47 |
| +ECPO w/o EC | 0.50 | 0.54 | 0.37 | 0.54 | 0.42 | 0.48 |
| **+ECPO** | **0.56** | **0.57** | **0.41** | **0.55** | **0.45** | **0.63** |

**Key Findings**: When the EC process is removed (replacing turn-by-turn EC with unified analysis), the interactivity improves but recommendation performance drops, making it significantly inferior to the full ECPO. This demonstrates that the turn-level EC process is essential.

### Rewriting Quality Evaluation

| Metric | Book | Game | Yelp |
|------|------|------|------|
| Information Fidelity Win Rate | 0.64 | 0.64 | 0.58 |
| Consistency Win Rate | 0.90 | 0.82 | 0.73 |

Rewritten responses outperform the original ones across all domains, with a particularly notable improvement in consistency.

### Hyperparameter Analysis (Rewriting Threshold $\lambda$)

- Low $\lambda$ (e.g., 1): Yields larger improvements per sample, but fewer total samples.
- High $\lambda$ (e.g., 4): Results in more samples overall, but the improvement of individual samples is irregular.
- Intuitive validation: Responses with extremely low satisfaction often suffer from critical issues, representing the most significant improvements after fixing.

# Highlights & Insights

- **Psychology-driven AI Alignment**: Introducing Expectation Confirmation Theory (ECT) to LLM preference optimization is an ingenious interdisciplinary transfer. ECT provides a theoretical framework to understand multi-turn conversation quality through the lens of "user satisfaction".
- **MTPO Paradigm without Extra Sampling**: ECPO implicitly assigns turn-level rewards and outputs natural language reasons via the EC process, then leverages a Rewriter to generate positive samples. This entirely avoids the high sampling cost of tree-search-based methods (500 vs. 2,500 trajectories).
- **Bidirectional Design with Forward Confirmation and Backward Derivation**: Consisting of forward detection ("what is wrong") and backward resolution ("how to fix"), which is logically self-consistent.
- **AILO's AIO-theory-based User Persona**: Reaches higher realism and diversity compared to random sampling, and its effectiveness is validated by a 100% human-evaluated win rate.
- **Orthogonality of ECPO with Preference Optimization**: Can replace DPO with KTO/SimPO, showing great generalizability.

## Limitations & Future Work

- **Dependency on User Simulator Quality**: Although AILO outperforms prior simulators, discrepancy (distribution shift) still exists relative to real users, potentially affecting real-world deployments.
- **Fixed EC Metric Dimensions**: The three scoring dimensions (flexibility, consistency, and guiding capability) might not suit all conversational recommendation scenarios, exhibiting a lack of domain adaptation.
- **Additional Overhead from Rewriter**: While avoiding the sampling overhead of tree search, the inference cost of the Rewriter itself is not analyzed in detail.
- **Validated Only in the Recommendation Domain**: The paper claims that ECPO can generalize to broader conversational assistants, but lacks empirical support.
- **Future Directions**: Internalizing the EC process into the model's reasoning phase (O1/R1 style), incorporating real user feedback, introducing domain-adaptive scoring dimensions, and extending ECPO to non-recommendation conversation scenarios.

## Related Work & Insights

- **vs. Tree-Simulation MTPO Methods (SDPO/SKTO, Jin et al. 2024; Xie et al. 2024)**: Tree-simulation-based methods sample multiple candidates at each turn and simulate entire dialogues, which is costly (2,500 trajectories) and susceptible to noise that causes negative gains. In contrast, ECPO requires only 500 trajectories without extra sampling, showing superior performance.
- **vs. Trajectory-level Preference Optimization (SFT/KTO, Ethayarajh et al. 2024)**: Trajectory-level methods evaluate multi-turn dialogues as a whole, failing to pinpoint which specific turn is problematic. ECPO realizes fine-grained, turn-level optimization using the EC process.
- **vs. iEvalLM User Simulator (Wang et al. 2023)**: iEvalLM generates user personas by simple random sampling, whereas AILO infers personas from real reviews based on AIO theory, which yields significantly better diversity and realism.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Applying ECT to LLM multi-turn preference optimization is a brand-new perspective. The forward confirmation + backward derivation pipeline is elegantly designed, eliminating the need for extra sampling required by tree searches.
- Experimental Thoroughness: ⭐⭐⭐⭐ Coverages are comprehensive, including 3 datasets, ablation studies, human evaluations, hyperparameter analyses, and rewriting quality analysis. However, it is validated on a single backbone (Llama-3.1-8B).
- Writing Quality: ⭐⭐⭐⭐ The framework design is clear, illustration diagram is intuitive, and the ECT theory is naturally introduced. Problem definitions and motivations are well articulated.
- Value: ⭐⭐⭐⭐⭐ As the first preference optimization method customized for CRAs, the ECPO paradigm provides broad insights for multi-turn dialogue systems. The AILO user simulator also holds independent value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] M2S: Multi-turn to Single-turn jailbreak in Red Teaming for LLMs](m2s_multiturn_to_singleturn_jailbreak_in.md)
- [\[ACL 2025\] Debate, Reflect, and Distill: Multi-Agent Feedback with Tree-Structured Preference Optimization for Efficient Language Model Enhancement](debate_reflect_and_distill_multi-agent_feedback_with_tree-structured_preference_.md)
- [\[ACL 2025\] MTSA: Multi-Turn Safety Alignment for LLMs through Multi-Round Red-Teaming](mtsa_multi-turn_safety_alignment_for_llms_through_multi-round_red-teaming.md)
- [\[ACL 2025\] Red Queen: Safeguarding Large Language Models against Concealed Multi-Turn Jailbreaking](red_queen_safeguarding_large_language_models_against_concealed_multi-turn_jailbr.md)
- [\[ACL 2025\] AutoMixAlign: Adaptive Data Mixing for Multi-Task Preference Optimization in LLMs](automixalign_adaptive_data_mixing.md)

</div>

<!-- RELATED:END -->

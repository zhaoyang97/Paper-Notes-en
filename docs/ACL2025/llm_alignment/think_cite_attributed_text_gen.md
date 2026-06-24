---
title: >-
  [Paper Note] Think&Cite: Improving Attributed Text Generation with Self-Guided Tree Search and Progress Reward Modeling
description: >-
  [ACL 2025][LLM Alignment][Attributed Text Generation] This paper models attributed text generation (text generation with citations) as a multi-step reasoning problem, proposing Self-Guided Monte Carlo Tree Search (SG-MCTS) combined with Progress Reward Modeling (PRM). Through multi-path search, intermediate state reflection, and a dual-dimensional progress reward (generation and attribution), the proposed method significantly outperforms all baselines across three datasets on…
tags:
  - "ACL 2025"
  - "LLM Alignment"
  - "Attributed Text Generation"
  - "Monte Carlo Tree Search"
  - "Self-Reflection"
  - "Citation Generation"
  - "Progress Reward Modeling"
date: 2026-05-08
content_hash: 99b50a306979e887
---

# Think&Cite: Improving Attributed Text Generation with Self-Guided Tree Search and Progress Reward Modeling

**Conference**: ACL 2025  
**arXiv**: [2412.14860](https://arxiv.org/abs/2412.14860)  
**Code**: [https://github.com/nusnlp/Think-Cite](https://github.com/nusnlp/Think-Cite)  
**Area**: LLM Alignment  
**Keywords**: Attributed Text Generation, Monte Carlo Tree Search, Self-Reflection, Citation Generation, Progress Reward Modeling  

## TL;DR
This paper models attributed text generation (text generation with citations) as a multi-step reasoning problem, proposing Self-Guided Monte Carlo Tree Search (SG-MCTS) combined with Progress Reward Modeling (PRM). Through multi-path search, intermediate state reflection, and a dual-dimensional progress reward (generation and attribution), the proposed method significantly outperforms all baselines across three datasets on the ALCE benchmark.

## Background & Motivation
**Background**: Attributed text generation requires LLMs to provide citation sources for each sentence when answering questions. Existing methods primarily fall into two categories: directly prompting LLMs to generate text with citations (prompt-based) or fine-tuning LLMs on labeled data (fine-tuning-based).

**Limitations of Prior Work**: (a) Existing methods adopt autoregressive "fast thinking" (System 1); once an intermediate sentence goes wrong (misstatement or incorrect citation), subsequent generation continues along the erroneous path. (b) Fine-tuning methods heavily rely on a large volume of annotated citation data, which is highly costly. (c) The lack of explicit planning in long-text generation leads to frequent unsupported statements and inaccurate citations.

**Key Challenge**: Attributed text generation essentially requires "slow thinking" (System 2)—where each sentence demands careful evidence selection and verification, yet autoregressive generation cannot backtrack for error correction.

**Goal**: How to achieve multi-path exploration and intermediate error correction in attributed text generation.

**Key Insight**: Drawing inspiration from tree search concepts in complex reasoning: since attributed text generation can be decomposed step-by-step by sentence, and each step involves "think-retrieve-write-cite", it can naturally be modeled as an MDP and optimized using MCTS to search for the best path.

**Core Idea**: Using self-reflection-enhanced MCTS to search multiple generation pathways, and utilizing progress rewards (generation quality + attribution quality) to evaluate the progress of each step.

## Method

### Overall Architecture
Given a query $\bm{x}$ and a document corpus $\mathcal{D}$, the model is required to generate a multi-sentence response with citations $\bm{y}=\langle y_1,...,y_T \rangle$. The architecture consists of three core components: (1) a language agent executing an iterative "Think-Verbalize-Cite" process; (2) SG-MCTS searching across multiple generation paths; and (3) Progress Reward Modeling providing dual-dimensional progress evaluation signals for both generation and attribution.

### Key Designs

1. **Iterative Think-Verbalize-Cite Agent**:

    - **Function**: Generating responses with citations sentence by sentence.
    - **Mechanism**: For the $t$-th sentence, the agent first does Think (formulating a blueprint of the next sentence as the search query $q_t$) $\rightarrow$ Search (retrieving the top-$K$ passages $\mathcal{D}_t$) $\rightarrow$ Verbalize (generating the sentence $y_t$ based on retrieval results) $\rightarrow$ Cite (appending citations $\mathcal{C}_t$ to the sentence).
    - **Design Motivation**: Unlike traditional RAG which uses a static reference corpus, dynamic retrieval at each step adapts to shifts in content focus.

2. **Self-Guided Monte Carlo Tree Search (SG-MCTS)**:

    - **Function**: Searching for the optimal attributed text across multiple generation pathways.
    - **Mechanism**: Each node $s_t = [q_t, \mathcal{D}_t, y_t, \mathcal{C}_t]$ in the MCTS tree represents a single step of generation, executing a four-step loop:
        - **Selection**: The UCT algorithm selects the most promising node $UCT(s_t) = V(s_t) + w\sqrt{\frac{\ln N(p)}{N(s_t)}}$.
        - **Reflection-Guided Expansion**: First generates an initial query $\hat{q}_{t+1}$, then the LLM performs self-reflection to check the quality of the query and retrieved results: $u = \mathcal{M}_\theta(\hat{q}_{t+1}, \hat{\mathcal{D}}_{t+1}, \bm{x})$. The query is refined based on the reflection, followed by retrieval, generation, and citation.
        - **Evaluation**: The new node is evaluated using the progress reward model.
        - **Backpropagation**: Rewards are backpropagated along the path to update the value functions of all ancestral nodes.
    - **Design Motivation**: Unlike methods that only reflect on final results, reflecting at intermediate states allows real-time correction of erroneous queries, preventing the expansion of an entire subtree along a wrong path. Each node expands 3 child nodes, with a maximum of 30 iterations.

3. **Progress Reward Modeling**:

    - **Function**: Evaluating the progress of tree search from the dual dimensions of generation quality and attribution quality.
    - **Mechanism**:
        - **Generation Progress Reward $R_g$**: Utilizing the token-level log-ratio of a DPO-aligned model as an implicit reward. The cumulative log-ratio under sentence-level MDP, $R_g(\bm{y}_{1:t+1}) = \sum_{k=0}^{t} w_k \log \frac{\pi^*(y_{k+1}|\bm{x}, \bm{y}_{1:k})}{\pi_{\text{ref}}(y_{k+1}|\bm{x}, \bm{y}_{1:k})}$, is used as a measure of generation quality.
        - **Attribution Progress Reward $R_a$**: An NLI model calculates citation recall (whether citations entail the generated sentence) and citation precision (whether each citation is necessary), taking the F1 score as the measure of attribution quality.
        - Total Reward: $R(s_{t+1}) = R_g + R_a$.
    - **Design Motivation**: Relying solely on single-step or end-state evaluations is unreliable; evaluating progress from the root to the current state provides a more holistic view. The dual dimensions ensure both generation and citation aspects are considered.

### Loss & Training
- **Training-free**: Entirely an inference-time method. An existing DPO model (LLaMA-3-8B-SFR-Iterative-DPO-R) is used to compute generation rewards, and an NLI model (T5-XXL-TRUE-NLI-Mixture) computes attribution rewards.
- The policy model can utilize LLaMA-3.1-8B-Instruct or GPT-4o.

## Key Experimental Results

### Main Results

| Dataset | Model | Correctness (Recall) | Citation Recall | Citation Precision |
|--------|------|---------------|---------|---------|
| ASQA | Vanilla RAG (ChatGPT) | 40.4 | 73.6 | 72.5 |
| ASQA | VTG | - | 86.7 | - |
| ASQA | **Think&Cite (GPT-4o)** | **50.1** | **89.5** | **87.1** |
| QAMPARI | Vanilla RAG (ChatGPT) | 20.8 | 20.5 | 20.9 |
| QAMPARI | **Think&Cite (GPT-4o)** | **45.2** | **50.6** | **52.8** |
| ELI5 | Vanilla RAG (ChatGPT) | 12.0 | - | - |
| ELI5 | **Think&Cite (GPT-4o)** | **25.9** | **85.6** | **80.2** |

### Ablation Study (ASQA, GPT-4o)

| Configuration | EM Recall | Citation Recall | Citation Precision | Description |
|------|----------|---------|---------|------|
| Think&Cite | 50.1 | 89.5 | 87.1 | Full Model |
| w/o SG-MCTS | 42.1 | 78.2 | 75.0 | Remove tree search, drop 8.0pp |
| w/o Reflection | 46.5 | 83.6 | 80.3 | Use standard MCTS |
| w/o GP Reward | 47.1 | 86.2 | 84.9 | Remove generation progress reward |
| w/o AP Reward | 46.7 | 81.3 | 80.4 | Remove attribution progress reward |

### Key Findings
- SG-MCTS is the most critical component—removing it drops correctness by 8pp and citation quality by over 11pp.
- The reflection step is most effective at 10 steps, whereas 15–20 steps lead to "overthinking" and introduce noise.
- Increasing the number of MCTS iterations consistently improves performance, but with diminishing marginal returns.
- The reflection mechanism is more efficient than increasing MCTS iterations—correcting search queries is more valuable than exploring an additional path.
- Attribution Progress Reward (AP Reward) has a larger impact on citation quality than Generation Progress Reward (GP Reward).

## Highlights & Insights
- **Modeling attributed text generation as multi-step reasoning and search** is the core insight—the System 2 thinking paradigm is naturally suited for tasks requiring verification. This paradigm can be transferred to other "verify-before-speak" scenarios (e.g., fact-checking, academic writing).
- **Intermediate state reflection** outperforms final-state reflection. In tree search, the earlier a wrong path is identified and corrected, the more search cost is saved. This is an improvement over approaches like Reflexion.
- **Using a DPO model as a free progress rewarder** is extremely clever—it bypasses the need to train specialized reward models by directly leveraging the token-level log-ratio of existing DPO-aligned models as implicit reward signals.
- **A purely inference-time method** that requires no fine-tuning of the target LLM, enabling plug-and-play and showing high generalizability.

## Limitations & Future Work
- The computational cost of MCTS search is high—30 iterations $\times$ 3 child node expansions $\times$ reflection steps results in dozens of times more LLM calls than standard generation.
- The retrieval scope is constrained by using only the top-100 passages as the candidate corpus.
- The tree search depth is limited to 6 levels, restricting the maximum number of sentences in the generated response.
- The reflection step utilizes the same policy LLM, which may suffer from "blind spots" (e.g., the model considers an erroneous query to be correct).
- Combining search with fine-tuning could be explored: utilizing MCTS search results as training data to distill into smaller models.

## Related Work & Insights
- **vs VTG**: VTG is guided by a verifier but still generates autoregressively without the ability to backtrack; Think&Cite's tree search can explore multiple paths and select the optimal one.
- **vs Inline Search**: Both support dynamic retrieval, but Inline Search does not reflect on query quality, leading to inaccurate retrieval.
- **vs APO/FG-Reward**: Fine-tuning approaches rely heavily on annotated data and remain autoregressive; Think&Cite requires no training and avoids error accumulation through search.
- This method is a representative application of inference-time scaling in the domain of text generation, echoing the trend of "more test-time compute = better results."

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to apply MCTS to attributed text generation; both SG-MCTS and progress reward modeling are meaningful innovations.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three datasets, ablation, and parameter analysis, but lacks computational cost comparison and experiments on more open-source models.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear problem modeling, rigorous mathematical derivation, and a complete motivational chain.
- Value: ⭐⭐⭐⭐⭐ Provides a powerful solution for reliable LLM generation, representing an important application of inference-time scaling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Tempest: Autonomous Multi-Turn Jailbreaking of Large Language Models with Tree Search](tempest_autonomous_multi-turn_jailbreaking_of_large_language_models_with_tree_se.md)
- [\[ACL 2025\] SynthesizeMe! Inducing Persona-Guided Prompts for Personalized Reward Models in LLMs](synthesizeme_persona_prompts.md)
- [\[ACL 2025\] AgentRM: Enhancing Agent Generalization with Reward Modeling](agentrm_enhancing_agent_generalization_with_reward_modeling.md)
- [\[ACL 2026\] ConsistRM: Improving Generative Reward Models via Consistency-Aware Self-Training](../../ACL2026/llm_alignment/consistrm_improving_generative_reward_models_via_consistency-aware_self-training.md)
- [\[ACL 2025\] Dynamic Scaling of Unit Tests for Code Reward Modeling](dynamic_scaling_of_unit_tests_for_code_reward_modeling.md)

</div>

<!-- RELATED:END -->

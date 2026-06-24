---
title: >-
  [Paper Note] Towards Better Chain-of-Thought: A Reflection on Effectiveness and Faithfulness
description: >-
  [ICML 2025][Reasoning][Chain-of-Thought] This paper systematically analyzes the factors influencing the performance of CoT from the two dimensions of effectiveness and faithfulness. It finds that problem difficulty, information gain, and information flow are key factors affecting CoT effectiveness. The root cause of unfaithful CoT is that the model directly recalls the correct information from the question while bypassing the CoT during answer prediction. Based on this…
tags:
  - "ICML 2025"
  - "Reasoning"
  - "Chain-of-Thought"
  - "Reasoning Effectiveness"
  - "Reasoning Faithfulness"
  - "Information Gain"
  - "Information Flow"
  - "Attribution Analysis"
date: 2026-05-08
content_hash: cb901f05ad424c15
---

# Towards Better Chain-of-Thought: A Reflection on Effectiveness and Faithfulness

**Conference**: ICML 2025  
**arXiv**: [2405.18915](https://arxiv.org/abs/2405.18915)  
**Code**: [BugMakerzzz/better_cot](https://github.com/BugMakerzzz/better_cot)  
**Area**: LLM Reasoning  
**Keywords**: Chain-of-Thought, Reasoning Effectiveness, Reasoning Faithfulness, Information Gain, Information Flow, Attribution Analysis

## TL;DR
This paper systematically analyzes the factors influencing the performance of CoT from the two dimensions of effectiveness and faithfulness. It finds that problem difficulty, information gain, and information flow are key factors affecting CoT effectiveness. The root cause of unfaithful CoT is that the model directly recalls the correct information from the question while bypassing the CoT during answer prediction. Based on this, the QUIRE method is proposed to improve both CoT effectiveness and faithfulness.

## Background & Motivation
1. **Background**: CoT technology enables LLMs to perform outstandingly in complex reasoning tasks. Especially, reasoning models like DeepSeek-R1 and o1 scale the CoT process via RL to even surpass human performance in competitive mathematics.
2. **Limitations of Prior Work**: CoT is not a silver bullet—it is ineffective or even harmful on certain tasks (such as commonsense reasoning), and suffers from unfaithfulness issues (where an incorrect CoT can still lead to a correct answer).
3. **Limitations of Existing Evaluation**:
    - Regarding effectiveness: Existing work only remains at superficial conclusions such as "CoT performs well on tasks containing mathematical symbols," lacking a deeper analysis of the underlying influencing factors.
    - Regarding faithfulness: Existing work mainly designs various methods to determine whether CoT is faithful, but lacks explanation of the causes behind unfaithful phenomena.
4. **Key Insight**: Starting from an information-theoretic perspective, this paper uses tools such as Information Gain (IG) and Information Flow to quantify information interaction patterns during the reasoning process of CoT, which not only diagnoses problems but also provides actionable improvement solutions.
5. **Core Idea**: Unfaithful CoTs lose critical information from the question, but the model can "recall" this lost information from the question when predicting the answer. Leveraging this mechanism in reverse to enhance CoT generation can simultaneously improve both effectiveness and faithfulness.

## Method

### Overall Architecture
This paper is divided into two major parts: analysis (§3-§4) and application (§5):
- **§3 Effectiveness Analysis**: Identifies three key factors affecting CoT effectiveness: problem difficulty, information gain, and information flow.
- **§4 Faithfulness Analysis**: Explains the cause of unfaithful CoT through the tripartite information interaction of question → CoT → answer.
- **§5 QUIRE Method**: An algorithm designed based on analytical conclusions to improve both effectiveness and faithfulness simultaneously.

### Effectiveness Analysis: Three Key Factors

#### Factor 1: Problem Difficulty
- **Difficulty quantification method**: Sample 10 no-CoT answers for each question, and divide them into 5 difficulty levels based on the pass@1 rate (pass@1 < 0.1 is the hardest level 5, > 0.8 is the easiest level 1).
- **Conclusion Cl.1**: CoT is more effective on difficult problems. On low-difficulty problems, CoT yields minimal improvement or even degrades performance, but significantly improves accuracy on high-difficulty problems.
- **Explanation of task differences**: Mathematical reasoning datasets have a higher proportion of difficult problems, whereas commonsense reasoning has a higher proportion of low-difficulty problems; thus, CoT is more effective in mathematical reasoning.

#### Factor 2: Information Gain
Quantify how much information the CoT obtains from the question using information gain in information theory:

$$IG(C, Q) = H(C) - H(C|Q) = -\sum_{i=1}^{n} p(c_i|C_{i-1}) \log p(c_i|C_{i-1}) + \sum_{i=1}^{n} p(c_i|C_{i-1};Q) \log p(c_i|C_{i-1};Q)$$

where $p(\cdot)$ is the model output probability, and $C_{i-1}$ represents the first $i-1$ tokens of the CoT. A larger IG indicates higher "dependence" of the CoT on the question, meaning the CoT itself provides less extra information.

- **Conclusion Cl.2**: CoT is more effective when it provides extra information not contained in the question itself (i.e., low IG). Mathematical reasoning has the lowest IG (CoT provides a large amount of extra derivation information), while commonsense reasoning has the highest IG (CoT basically reiterates problem information).

#### Factor 3: Information Flow
Use Integrated Gradient Attribution (IGA) to trace information transfer from each CoT position to the answer:

$$I(x_n, y_m) = E(x_n) \int_{\alpha=0}^{1} \frac{\partial f(\alpha y_m)}{\partial E(x_n)} d\alpha$$

After normalization, the attribution effect score $AE(x_n, y_m)$ is obtained, which is then averaged over answer tokens to obtain $AAE(c, A)$.

Further, define **Monotonicity of Information Flow (MIF)** as the Spearman correlation coefficient between the CoT step position and the AAE:

$$MIF(C, A) = 1 - \frac{6\sum_{i=1}^{n}[n+1-i-R(AAE(c_i, A))]^2}{n(n^2-1)}$$

- **Conclusion Cl.3**: CoT is more effective when information flow increases monotonically along the CoT process (high MIF). For instance, the AAE curve of GSM8K rises significantly, while ECQA remains basically flat.

### Faithfulness Analysis: Three-step Diagnosis

Manual evaluation of 50 pairs of CoT-answer pairs reveals that **logical reasoning tasks have the highest proportion of unfaithfulness** (up to 17/50 in ProntoQA).

#### Step 1: Question→CoT (Cl.4)
- Unfaithful CoT has lower IG(Q,C), indicating that CoT obtains less information from the question context.
- **Conclusion**: Unfaithful CoT loses critical correct information in the context.

#### Step 2: CoT→Answer (Cl.5)
- The AAE from unfaithful CoT to the answer is significantly lower than that of faithful CoT.
- **Conclusion**: There is less information interaction between unfaithful CoT and the answer.

#### Step 3: Question→Answer (Cl.6)
- Rank statements in the question based on their AAE to the answer. In unfaithful cases, correct statements lost in CoT appear more frequently in the top-k high-AAE positions.
- **Conclusion**: When unfaithful CoT occurs, the model bypasses the CoT during answer prediction and directly "recalls" the lost correct information from the question.

### QUIRE Method

Based on the six conclusions above, **QUestion Information Recall and Enhancement (QUIRE)** is proposed:

#### Component 1: AAE Recall
1. First generate an initial answer $A$ using Self-Consistency.
2. Calculate the $AAE(S, A)$ for each statement $S$ in the question to the answer.
3. Select the top-k statements with the highest AAE as extra hints.
4. Add these hints to the input prompt to guide the generation of a new CoT with more complete information.

#### Component 2: IG Vote
1. Generate multiple information-enhanced CoTs (can be combined with SC technology).
2. Calculate $IG(Q, C)$ for each CoT as a quality weight—a higher IG indicates that the CoT obtained more information from the question, with fewer hallucinated statements.
3. Use the IG values as SC voting weights to select the final answer.

## Key Experimental Results

### Main Results (Llama3.1-8B)

| Method | ProofWriter Acc | ProofWriter BS | ProofWriter FBS | ProntoQA Acc | ProntoQA BS | ProntoQA FBS |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| CoT | 59.2 | 64.9 | 55.7 | 86.8 | 86.1 | 78.0 |
| Self-Consistency | 60.6 | 65.0 | 57.8 | 93.2 | 87.5 | 83.6 |
| Least-to-Most | 54.0 | 60.4 | 56.4 | 90.0 | 77.3 | 72.6 |
| Self-Refine | 51.6 | 65.9 | 53.4 | 88.5 | 91.5 | 84.5 |
| **QUIRE (Ours)** | **63.0** | **66.6** | **58.0** | **95.0** | **92.7** | **89.2** |
| - AAE Recall | 60.2 | 65.1 | 57.0 | 95.0 | 87.5 | 84.6 |
| - IG Vote | 62.8 | 64.1 | 56.6 | 94.3 | 87.0 | 83.4 |

> **Key Findings**: QUIRE achieves up to a 2.4% accuracy improvement (ProofWriter) and up to a 5.6% improvement in faithfulness (FBS) (ProntoQA). Ablation studies show that both AAE Recall and IG Vote make independent contributions.

### CoT Unfaithfulness Statistics (Llama3.1-8B, 50 samples/dataset)

| CoT→Answer | GSM | AQuA | ProofWriter | ProntoQA | WinoGrande | SocialIQA |
|:--------:|:---:|:---:|:---:|:---:|:---:|:---:|
| ✓→✓ | 41 | 25 | 14 | 27 | 34 | 40 |
| ✓→✗ | 0 | 0 | 0 | 0 | 1 | 0 |
| ✗→✓ (Unfaithful) | 1 | 1 | **7** | **17** | 1 | 0 |
| ✗→✗ | 8 | 24 | 29 | 6 | 14 | 10 |

> **Key Findings**: The proportion of unfaithfulness in logical reasoning tasks (ProofWriter 14%, ProntoQA 34%) is significantly higher than that in mathematical and commonsense reasoning tasks.

## Highlights & Insights
1. **Systematization with an Information-Theoretic Perspective**: For the first time, three quantitative indicators—IG, AAE, and MIF—are used to uniformly explain the effectiveness and faithfulness of CoT, forming a comprehensive analytical framework.
2. **Clear Explanation of the Causes of Unfaithfulness**: Through a three-step information interaction analysis, the mechanism of "the model directly recalling lost information from the question" is revealed, which provides deeper insight than simply judging "faithful/unfaithful".
3. **Closed Loop from Analysis to Application**: The QUIRE method is directly derived from the analytical conclusions, and experiments demonstrate the causal relation of "faithfulness improvement -> effectiveness improvement".
4. **Methodological Value of White-Box Analysis**: The combination of IGA and Information Gain provides a reusable analytical tool for researching CoT mechanisms.

## Limitations & Future Work
1. **Limited to Open-Source Models**: Since gradient information is required, black-box models like GPT-4 cannot be analyzed.
2. **Lack of Theoretical Proof**: The causal relationship of faithfulness -> effectiveness is only empirically supported, lacking theoretical guarantees.
3. **Limited Data Scale**: The unfaithfulness analysis is based on manual evaluation of only 50 samples, which is a small scale.
4. **Computational Overhead of QUIRE**: Additional calculations of AAE and IG are required, needing backpropagation of gradients, which incurs high inference costs.
5. **Task Coverage**: The analysis and methods are mostly verified on logical reasoning tasks; the level of improvement in mathematical and commonsense reasoning remains to be validated.

## Related Work & Insights
- **CoT Effectiveness Evaluation**: Sprague et al. (2024) and Xu & Ma (2024) evaluated the effects of CoT across different task types, but this paper further identifies the underlying factors.
- **CoT Faithfulness Evaluation**: Bao et al. (2024) used causal mediation analysis to measure faithfulness, and Lanham et al. (2023) measured faithfulness but lacked explanation; this paper complements them with causal analysis.
- **Information Flow Analysis**: Wu et al. (2023) and Wang et al. (2024) used IGA to analyze internal information routing in models, whereas this paper applies it to CoT-answer information interaction.
- **Insights**: The paradigm of "generating an initial answer first, then reconstructing using attribution signals to enhance" in QUIRE can be generalized to scenarios such as RAG and self-reflection.

## Rating
⭐⭐⭐⭐ — The analysis is in-depth and systematic, and information theory tools are applied cleverly. However, its application scope is limited to open-source models requiring gradient information and logical reasoning tasks, and the practicality of QUIRE is constrained by computational overhead.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] PCoT: Persuasion-Augmented Chain of Thought for Detecting Fake News and Social Media Disinformation](pcot_persuasion-augmented_chain_of_thought_for_detecting_fake_news_and_social_me.md)
- [\[ICLR 2026\] FaithCoT-Bench: Benchmarking Instance-Level Faithfulness of Chain-of-Thought Reasoning](../../ICLR2026/llm_reasoning/faithcot-bench_benchmarking_instance-level_faithfulness_of_chain-of-thought_reas.md)
- [\[CVPR 2025\] Interleaved-Modal Chain-of-Thought](../../CVPR2025/llm_reasoning/interleaved-modal_chain-of-thought.md)
- [\[NeurIPS 2025\] Mind the Gap: Bridging Thought Leap for Improved Chain-of-Thought Tuning](../../NeurIPS2025/llm_reasoning/mind_the_gap_bridging_thought_leap_for_improved_chain-of-thought_tuning.md)
- [\[NeurIPS 2025\] Reasoning Models Better Express Their Confidence](../../NeurIPS2025/llm_reasoning/reasoning_models_better_express_their_confidence.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Enhancing Conversational Agents with Theory of Mind: Aligning Beliefs, Desires, and Intentions for Human-Like Interaction
description: >-
  [ACL 2025 (Findings)][LLM (Other)][Theory of Mind] This paper explores the feasibility of extracting Theory of Mind (ToM) information from the internal representations of open-source LLMs (LLaMA). It leverages the BDI (Belief-Desire-Intention) framework to manipulate these representations for generating dialogue responses aligned with human social cognition, achieving win rates of 67% and 63% for ToM-aligned 3B and 8B models, respectively.
tags:
  - "ACL 2025 (Findings)"
  - "LLM (Other)"
  - "Theory of Mind"
  - "BDI framework"
  - "dialogue alignment"
  - "LLM interpretability"
  - "LatentQA"
date: 2026-05-08
content_hash: 6522f1231cf92974
---

# Enhancing Conversational Agents with Theory of Mind: Aligning Beliefs, Desires, and Intentions for Human-Like Interaction

**Conference**: ACL 2025 (Findings)  
**arXiv**: [2502.14171](https://arxiv.org/abs/2502.14171)  
**Code**: Yes (code repository mentioned in the paper)  
**Area**: Other  
**Keywords**: Theory of Mind, BDI framework, dialogue alignment, LLM interpretability, LatentQA

## TL;DR

This paper explores the feasibility of extracting Theory of Mind (ToM) information from the internal representations of open-source LLMs (LLaMA). It leverages the BDI (Belief-Desire-Intention) framework to manipulate these representations for generating dialogue responses aligned with human social cognition, achieving win rates of 67% and 63% for ToM-aligned 3B and 8B models, respectively.

## Background & Motivation

As LLM-driven conversational assistants are increasingly integrated into various domains, their deficiencies in understanding social contexts and aligning with non-literal language have become apparent. In communication, humans naturally rely on the Theory of Mind (ToM)—inferring interlocutors' beliefs, desires, and intentions—to adapt their expressions. However, the capacity of current LLMs in this regard remains highly controversial:

**ToM Deficiency Issue**: Existing LLMs often underperform in scenarios requiring inference of opponents' mental states, such as negotiations and games, generating contextually inappropriate responses.

**Challenges in Alignment**: Traditional alignment methods (e.g., RLHF, DPO) offer limited alignment at the pragmatic level, failing to capture the pragmatic and social nuances of dialogue.

**Research Gap**: Although existing works evaluate the ToM capabilities of LLMs or design "mind modules", no study has attempted to extract ToM representations from within LLMs and apply them to response alignment.

The core motivation of this study is: if LLMs indeed preserve ToM-related cues during causal language modeling, can these cues be leveraged to enhance the social alignment of conversational responses?

## Method

### Overall Architecture

This paper designs three complementary sets of experiments around three research questions:
- **RQ1 (Reading ToM)**: How much ToM information is encoded within the internal representations of LLMs?
- **RQ2 (ToM Consistency)**: Is the extracted ToM information reliable and non-hallucinatory?
- **RQ3 (ToM Controllable Generation)**: Can ToM representations be utilized to enhance response alignment?

### Key Designs

1. **ToM Information Reading (RQ1)**:

    - **Linear Probing**: Extracts the hidden state $h(S)$ of the final token from the LLM's residual stream, learning a weight matrix $W$ and bias $b$ to map to the ToM label space: $\hat{y} = \text{softmax}(Wh(S) + b)$
    - **LatentQA**: Feeds the dialogue input into a frozen target model to obtain the internal representation $R(S)$. A decoder model then receives the ToM question and $R(S)$ to generate answers. This approach leverages the entire activation sequence rather than a single embedding.
    - **Multi-Layer Depth Experiments**: Extracts representations at shallow, middle, and deep layers respectively to investigate the impact of layer depth on ToM reading accuracy.

2. **ToM Consistency Verification (RQ2)**:

    - Utilizes specialized consistency evaluation datasets such as FanToM and NegotiationToM.
    - Evaluation requires the model to provide logically consistent answers to different types of questions within the same ToM scenario. For example, if a multiple-choice question is answered correctly but the factual Q&A is contradictory, it is not deemed correct.
    - Compares the consistency of three methods: LatentQA, fine-tuning, and CoT reasoning.

3. **ToM Controllable Generation (RQ3)**:

    - **Mechanism**: Modifies the target model's internal representation from $R(S)$ to $R'(S)$ to selectively manipulate specific characters' beliefs, desires, or intentions.
    - The gradient flow optimizes the decoder by comparing the generated answers with actual ToM answers, and this gradient is then used to boost specific ToM components in the target representation.
    - After modification, only the target model is used to generate the aligned response $C''$, which is compared against the unmodified model's response $C'$.

### Loss & Training

- Linear Probing: Standard cross-entropy loss.
- LatentQA Decoder Training: Supervised learning with ground-truth ToM annotations.
- ToM Controllable Generation: Gradient-based representation boosting, with no extra loss function, but rather augmenting the target representation via the backpropagation path.

## Key Experimental Results

### Main Results: ToM Reading Accuracy (Table)

| Model | Layer Depth | CaSiNo (Exact Match, Both-A1-A2) | CRAIGSLISTBARGAIN (R², Seller-Buyer) |
|------|--------|-------------------------------|--------------------------------------|
| LLaMA3-1B | Middle | LP: 02-16-13 / LQA: 20-42-39 | LP: 0.26-0.26 / LQA: 0.89-0.92 |
| LLaMA3-3B | Middle | LP: 05-23-21 / LQA: 29-60-44 | LP: 0.19-0.27 / LQA: 0.96-0.98 |
| LLaMA3-8B | Middle | LP: 02-10-23 / LQA: 46-62-70 | LP: 0.36-0.40 / LQA: 0.93-0.91 |

> LP = Linear Probing, LQA = LatentQA. LatentQA performs best in the middle layers, significantly outperforming linear probing.

### ToM Consistency and Controllability Results (Table)

| Model | Method | FanToM ALL* | NegotiationToM ALL |
|------|------|------------|-------------------|
| LLaMA3-3B | LatentQA | 11.9 | 6.2 |
| LLaMA3-3B | Fine-tuning | 8.2 | 11.2 |
| LLaMA3-8B | LatentQA | 16.4 | 15.2 |
| LLaMA3-8B | Fine-tuning | 12.8 | 17.7 |
| GPT-4o-mini | CoT | 0.5 | 4.8 |

> Overall ToM consistency remains low, but LatentQA shows significant improvement over CoT reasoning.

### Key Findings

1. **Middle Layers are Optimal**: LatentQA performs best in the middle layers in 5 out of 6 experimental sets. Shallow layers lack rich semantic information, while deep layers might be influenced by pre-training egocentric bias.
2. **Larger Models Perform Better**: The 8B model significantly outperforms the 1B and 3B models in ToM reading.
3. **Controllable Generation is Effective**: After ToM alignment, the weighted average win rate for the 3B model is 67.15%, and 63.25% for the 8B model.
4. **Greater Improvement in 3B**: This is likely because the unaligned baseline of the 8B model is already strong.
5. **"Empathetic Expression" and "Demand Description" Perform Worst**: These intentions are already highly prevalent in pre-training corpora, meaning models naturally possess the ability to express them.

## Highlights & Insights

- **Prominent Theoretical Contribution**: For the first time, ToM's BDI framework is combined with LLM internal representation manipulation, building a complete pipeline from "reading → verification → control".
- **Obvious Advantages of LatentQA**: Compared to linear probing which only inspects single activation vectors, LatentQA leverages the complete activation sequence, capturing richer ToM information.
- **Practical Potential**: Proves that even with imperfect ToM consistency, augmenting the correct ToM information can still improve real-world alignment performance.
- **No Need for Full-Model Fine-Tuning**: Alignment can be achieved simply by manipulating internal representations, which is computationally far more efficient than RLHF/DPO.

## Limitations & Future Work

1. **Evaluation Relies on LLM Judges**: Uses GPT-4o, o1, and Gemini as evaluators, lacking human evaluation.
2. **Single Dataset**: Controllability experiments were only conducted on a single dataset, NegotiationToM.
3. **Sensitivity to Hyperparameters**: The alignment generation process is highly sensitive to hyperparameter tuning, impacting stability and reproducibility.
4. **Limited Model Families**: Only evaluated on the LLaMA3 family, without verifying generalization across different architectures.
5. **Gap to Real-World Deployment**: Currently, ToM Q&A pairs require manual design; practical applications would require dynamic planning and execution using LLMs.
6. **Ethical Risks**: ToM capabilities could potentially be used to manipulate user psychological states, necessitating transparent design and informed consent.

## Related Work & Insights

- **LatentQA (Pan et al., 2024)** serves as the core foundation of this study's methodology, framing internal representation decoding as a visual question answering task.
- **NegotiationToM (Chan et al., 2024)** provides a negotiation dialogue dataset annotated with BDI.
- **Street ToM Alignment Prospects (Street, 2024)** outlines future directions for employing ToM in LLM alignment.
- **Insight from this paper**: Internal representation manipulation is a promising, lightweight alignment apparatus, suitable for achieving finer-grained control over personality and social behavior within dialogue systems.

## Rating

| Dimension | Score (1-5) | Description |
|------|-----------|------|
| Novelty | 4 | First to combine ToM + BDI with internal representation manipulation for dialogue alignment |
| Experimental Thoroughness | 3.5 | Three sets of RQ experiments are completely designed, but datasets and model families are somewhat limited |
| Writing Quality | 4 | Problem formulation is clear, and the experimental structure is well-organized |
| Value | 4 | Provides a new paradigm for LLM alignment, possessing both theoretical depth and practical potential |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Theory of Mind in Large Language Models: Assessment and Enhancement](theory_of_mind_llm.md)
- [\[ACL 2025\] EnigmaToM: Improve LLMs' Theory-of-Mind Reasoning Capabilities with Neural Knowledge Base of Entity States](enigmatom_improve_llms_theory-of-mind_reasoning_capabilities_with_neural_knowled.md)
- [\[ACL 2025\] AXIS: Efficient Human-Agent-Computer Interaction with API-First LLM-Based Agents](axis_efficient_human-agent-computer_interaction_with_api-first_llm-based_agents.md)
- [\[ACL 2025\] Game Development as Human-LLM Interaction](game_development_as_human-llm_interaction.md)
- [\[ACL 2025\] Substance over Style: Evaluating Proactive Conversational Coaching Agents](proactive_conversational_coaching.md)

</div>

<!-- RELATED:END -->

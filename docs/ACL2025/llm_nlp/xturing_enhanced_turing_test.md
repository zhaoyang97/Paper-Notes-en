---
title: >-
  [Paper Note] X-Turing: Towards an Enhanced and Efficient Turing Test for Long-Term Dialogue Agents
description: >-
  [ACL 2025][LLM (Other)][Turing Test] This paper proposes the X-Turing framework, which enhances and streamlines the Turing Test by introducing a burst dialogue mode and pseudo-dialogue generation technology. It evaluates the human-mimicking capabilities of LLMs in long-term dialogues, revealing a significant performance drop as the number of dialogue turns increases.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Turing Test"
  - "Long-term Dialogue"
  - "Burst Dialogue"
  - "Dialogue Generation"
  - "Human Evaluation"
date: 2026-05-08
content_hash: de649f2b00b5286e
---

# X-Turing: Towards an Enhanced and Efficient Turing Test for Long-Term Dialogue Agents

**Conference**: ACL 2025  
**arXiv**: [2408.09853](https://arxiv.org/abs/2408.09853)  
**Code**: [https://github.com/vickywu1022/X-Turing](https://github.com/vickywu1022/X-Turing)  
**Area**: Others  
**Keywords**: Turing Test, Long-term Dialogue, Burst Dialogue, Dialogue Generation, Human Evaluation  

## TL;DR

This paper proposes the X-Turing framework, which enhances and streamlines the Turing Test by introducing a burst dialogue mode and pseudo-dialogue generation technology. It evaluates the human-mimicking capabilities of LLMs in long-term dialogues, revealing a significant performance drop as the number of dialogue turns increases.

## Background & Motivation

Traditional Turing Tests exhibit three key limitations when evaluating the dialogue capabilities of contemporary LLMs:

**Rigid Dialogue Patterns**: Traditional tests rely on a strict ping-pong structure (one-question-one-answer). However, in reality, people frequently send multiple consecutive messages without waiting for a reply. This ping-pong pattern fails to reflect natural communication behavior.

**High Human Labor Cost**: This requires continuous human participation in dialogues with AI, restricting test durations typically to less than ten minutes. Consequently, it cannot evaluate the performance of LLMs in long-term interactions. Maintaining consistency and coherence in long-term dialogues is critical for evaluating LLMs, but feasible testing methods are currently lacking.

**Inappropriate Time Metric**: Measuring test duration in minutes ignores individual differences in reading, thinking, and typing speeds, which may induce evaluation bias.

While current large models such as GPT-4 perform exceptionally well in brief dialogues, their performance in complex and long-term interactions has not been systematically evaluated, which constitutes the core motivation of this study.

## Method

### Overall Architecture

The X-Turing framework introduces three core innovations:

1. **Burst Dialogue Mode**: Breaks the traditional one-to-one messaging constraint, allowing users and systems to send multiple messages consecutively.
2. **Pseudo-Dialogue Generation**: Automatically simulates long-term human-machine interactions via iterative dialogue generation.
3. **X-Turn Pass-Rate Metric**: A new evaluation metric that quantifies the probability of LLMs passing the Turing Test at different dialogue turns.

### Key Designs

#### Burst Dialogue System

To enable burst dialogues, the system employs three synchronously operating modules:

- **Input Listener**: Continuous collection of user messages.
- **Model Caller**: Invokes the LLM to generate replies.
- **Output Sender**: Manages the transmission of system responses based on timestamps.

The system introduces a time interval $t_1$ to wait for complete user expression before processing. Each message containing $n$ characters is sent after a delay $d$, defined as:

$$d = \mathcal{N}(0.3, 0.03) \times n$$

This simulates human reading, thinking, and typing speeds.

#### Chatbot Construction

Chatbots are constructed based on dialogue history from real social platforms rather than manually crafted personas. Historic dialogues of the target individual (containing timestamps and message content) are used to prompt the LLM, enabling it to mimic the target individual's personality, linguistic style, and conversation patterns.

#### Pseudo-Dialogue Generation

To generate $N = mn$ turns of pseudo-dialogue:

1. First, the LLM generates $n$ distinct topics $T = \{t_1, t_2, \ldots, t_n\}$.
2. Based on the target individual's dialogue history $H$, it iteratively prompts the LLM to generate $m$ turns of dialogue for each topic.
3. Generated dialogues are appended to the dialogue history $H$ for subsequent iterations.
4. Generation continues if there are fewer than $m$ turns, and is truncated if it exceeds $m$.

After integrated into the chatbot's prompt, human judges only need to engage in another $m_T$ turns of short-term dialogue to evaluate the LLM's performance after $(N + m_T)$ turns of interaction. This design is grounded in psychological research on the serial position effect—the last part of a sequence is the easiest to recall. Consequently, the LLM is first allowed to generate a large amount of pseudo-dialogue auto-regressively, while human evaluation is focused on the final few turns of actual human-machine dialogue.

### Loss & Training

This paper does not involve model training but presents an evaluation framework. The core evaluation metric, X-Turn Pass Rate, is defined as:

$$\text{Pass Rate} = 1 - \frac{1}{N} \sum_{i=1}^{N} \left(\frac{C_i}{K}\right)$$

where $N$ is the number of dialogue pairs, $K$ is the number of judges, and $C_i$ represents the number of judges who correctly identify the human-machine dialogue in the $i$-th pair. A higher pass rate indicates that the LLM exhibits more human-like characteristics.

## Key Experimental Results

### Main Results

**Setup**:
- Provide 100 turns of dialogue history as context.
- Evaluate three models: GPT-4, Claude-3-Sonnet, and Qwen-110B.
- 25 topics covering education, technology, art, and other fields.
- 10 topics used for ping-pong dialogue, and 15 for burst dialogue.
- 236 native speakers as human judges.
- GPT-4 and Qwen-110B also serve as LLM judges.

**Core Results (Human-Judge Pass Rate)**:

| Model | 3-Turn | 10-Turn | 110-Turn |
|------|--------|---------|----------|
| GPT-4 | 51.9% | 38.9% | 13.3% |
| Claude-3-Sonnet | 51.8% | 32.1% | 7.1% |
| Qwen-110B | 44.6% | 22.7% | 1.1% |

**Key Findings**: The pass rates of all models significantly decline as the number of dialogue turns increases. Specifically, GPT-4 drops from 51.9% in 3-turn to 13.3% in 110-turn, while Qwen-110B falls dramatically from 44.6% to 1.1%.

### Key Findings

1. **Short-term vs. Long-term Performance Gap**: While LLMs can initially "fool" human judges in short-term dialogues (attaining approximately 50% pass rate), their performance severely degenerates in long-term dialogues. This demonstrates a fundamental deficiency of LLMs in maintaining long-term consistency.

2. **Burst vs. Ping-pong Dialogue**:
    - GPT-4 achieves a 56.2% 3-Turn pass rate in ping-pong mode compared to 49.1% in burst mode.
    - The performance gap shrinks in long-term dialogues, suggesting the two modes converge during long-term evaluations.

3. **LLM Judges vs. Human Judges**: LLM judges generally assign higher pass rates than human judges, with significant discrepancies manifesting in long-term dialogues (e.g., GPT-4 evaluated by humans at 13.3% in 110-Turn, whereas GPT-4 evaluated by GPT-4 at 36%). This highlights that LLM judges are unreliable for this task.

4. **Word Count Discrepancy**: LLMs tend to generate longer responses. In 110-turn ping-pong dialogue, GPT-4 averages 3.5 more words per message than the target individual, which serves as a critical giveaway for identification.

## Highlights & Insights

1. **Practical Evaluation Framework Innovation**: By leveraging pseudo-dialogue generation, human labor cost is substantially minimized, making the evaluation of LLM performance across 100+ turns feasible. Unlike traditional methods requiring constant human involvement, this approach only requires human engagement in the final turns.
2. **Realistic Burst Dialogue Mode**: Reflects the natural micro-behavior of sending consecutive messages in instant messaging, providing higher ecological validity than classical ping-pong testing.
3. **Strong Negative Conclusions**: Experiments clearly show that current LLMs fail to pass the long-term Turing Test, with consistency rapidly collapsing as discussions progress. This finding is highly valuable for understanding the capability boundaries of LLMs.
4. **X-Turn Pass Rate Metric**: Quantifying evaluation by dialogue turns rather than time duration eliminates the impact of individual typing/reading variations, offering a fairer evaluation dimension.

## Limitations & Future Work

1. **Limited Diversity of Target Personas**: The experiments are based on the chat logs of specific individuals from social platforms. This may lack diversity, and conclusions might vary across different personalities and cultural backgrounds.
2. **Quality of Pseudo-Dialogue**: Pseudo-dialogues are generated by the LLM itself. Their quality and authenticity may undermine the accuracy of downstream evaluations, risking self-reinforcement biases.
3. **Text-Centric Evaluation**: The work focuses solely on text dialogues, neglecting multimodal communication elements such as speech and emojis.
4. **Unreliability of LLM Judges**: The massive discrepancy between human and LLM judges indicates that automated LLM evaluation is currently unreliable. However, recruiting human judges remains highly expensive.

## Related Work & Insights

- **Turing Test Variants**: Jones and Bergen (2024) explore whether LLMs can pass the Turing Test. This work extends the focus from "whether they can pass" to "how long they can persist".
- **Role-Playing Capabilities**: Previous works such as Wu et al. (2024b) and Li et al. (2023) study the persona mimicking of LLMs. This study applies persona mimicking to the Turing Test scenario.
- **Dialogue Generation**: Synthetic conversation techniques, such as those proposed by Soudani et al. (2024), provide foundations for pseudo-dialogue generation.
- **Insights**: This framework can be extended to evaluate the consistency of LLM agents in executing long-term tasks, not limiting to social chats but also including multi-session customer service and psychological counseling.

## Rating

| Dimension | Score (1-5) |
|------|-----------|
| Novelty | 4 |
| Technical Depth | 3 |
| Experimental Thoroughness | 4 |
| Practical Value | 4 |
| Writing Quality | 4 |
| Overall Rating | 3.8 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Position: The Turing-Completeness of Autoregressive Transformers Relies Heavily on Context Management](../../ICML2026/llm_nlp/position_the_turing-completeness_of_autoregressive_transformers_relies_heavily_o.md)
- [\[ACL 2025\] AXIS: Efficient Human-Agent-Computer Interaction with API-First LLM-Based Agents](axis_efficient_human-agent-computer_interaction_with_api-first_llm-based_agents.md)
- [\[ACL 2025\] Beyond Dialogue: A Profile-Dialogue Alignment Framework Towards General Role-Playing Language Model](beyond_dialogue_a_profile-dialogue_alignment_framework_towards_general_role-play.md)
- [\[ACL 2025\] Substance over Style: Evaluating Proactive Conversational Coaching Agents](proactive_conversational_coaching.md)
- [\[ACL 2025\] SCULPT: Systematic Tuning of Long Prompts](sculpt_systematic_tuning_of_long_prompts.md)

</div>

<!-- RELATED:END -->

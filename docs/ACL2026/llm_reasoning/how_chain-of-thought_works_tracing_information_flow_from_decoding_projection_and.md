---
title: >-
  [Paper Note] How Chain-of-Thought Works? Tracing Information Flow from Decoding, Projection, and Activation
description: >-
  [ACL2026][LLM Reasoning][Chain-of-Thought] This paper back-traces the information flow of Chain-of-Thought (CoT) from three levels—decoding, probability projection…
tags:
  - "ACL2026"
  - "LLM Reasoning"
  - "Chain-of-Thought"
  - "Information Flow Tracing"
  - "decoding space pruning"
  - "neuron activation"
  - "mechanistic interpretability"
date: 2026-05-08
content_hash: 2dbafdf890b01646
---

# How Chain-of-Thought Works? Tracing Information Flow from Decoding, Projection, and Activation

**Conference**: ACL2026  
**arXiv**: [2507.20758](https://arxiv.org/abs/2507.20758)  
**Code**: https://github.com/How-Young-X/cot  
**Area**: LLM Reasoning / Mechanistic Interpretability / Prompt Analysis  
**Keywords**: Chain-of-Thought, Information Flow Tracing, decoding space pruning, neuron activation, mechanistic interpretability  

## TL;DR
This paper back-traces the information flow of Chain-of-Thought (CoT) from three levels—decoding, probability projection, and FFN activation. It finds that CoT primarily enhances reasoning performance by constraining answer structures, reducing prediction entropy, and modulating neuron activation based on task types, rather than simply making the model "more logically capable" in the human sense.

## Background & Motivation
**Background**: Chain-of-Thought prompting is a classic method for enhancing the multi-step reasoning capabilities of LLMs, providing significant gains across arithmetic, commonsense, and symbolic reasoning tasks. While many variants of CoT have been proposed, direct mechanistic evidence for why vanilla CoT is effective remains scarce.

**Limitations of Prior Work**: Existing explanations often persist at the behavioral level, such as "CoT reduces task complexity," "models imitate answer templates," or "prompt format is more important than logical content." These claims are intuitively plausible but rarely link external output changes to internal model probability or activation shifts.

**Key Challenge**: CoT's final output text resembles reasoning, but this does not equate to the model executing reasoning internally according to human logic. To understand CoT, one must look back from the output tokens: whether the generation distribution narrows, whether the answer space becomes more concentrated, and whether internal neurons participate in computation differently.

**Goal**: Establish an information flow analysis framework to explain how CoT alters model behavior across three stages—decoding, projection, and activation—addressing whether CoT expands reasoning capability or merely constrains output space and activation patterns.

**Key Insight**: The authors select vanilla CoT, six models (3B to 70B parameters), and nine reasoning datasets. Under identical prompt settings, they compare CoT with standard prompting, mapping results to keyword imitation, answer structure adherence, token probability, entropy, and FFN activation.

**Core Idea**: View CoT as an information flow regulator: it provides structural templates at the decoding layer, narrows the candidate answer distribution at the probability projection layer, and selectively increases or decreases FFN neuron involvement in the later layers based on task type.

## Method
The paper does not propose a new reasoning model but rather a mechanistic analysis pipeline. It first examines whether generated text imitates structures in the prompt, then assesses if these structures lead to more concentrated probability distributions, and finally observes systematic changes in FFN activation counts and layer-wise differences.

### Overall Architecture

Experiments cover three task categories: arithmetic reasoning (GSM8K, SVAMP, AQuA), commonsense reasoning (Bamboogle, StrategyQA, Date, Sports), and symbolic reasoning (Coin Flip, Last Letters Concatenation). Models include LLaMA3.1 8B/70B, Gemma2 2B/9B/27B, and LLaMA3.2-3B. All experiments use 4-shot prompts, greedy decoding, and a maximum generation length of 300 tokens.

The analysis follows a reverse order: starting with decoding (how output text borrows keywords from prompts and questions); moving to projection (token probability and entropy after projecting hidden states to the vocabulary); and concluding with activation (the number of activated neurons in FFNs and the layer-wise differences between CoT and standard prompting).

### Key Designs

1.  **Structure Adherence Analysis at Decoding Layer**:
    - **Function**: Determine if CoT constrains output through templated structures rather than relying entirely on freely generated logic.
    - **Mechanism**: The authors define four types of test points: time, action, loc&peo, and number, counting whether keywords in the generated text originate from the prompt or the question. They further define a CoT reasoning structure (input entity $\rightarrow$ operation $\rightarrow$ generated entity $\rightarrow$ final answer statement) and measure structure adherence using "Imitation Count."
    - **Design Motivation**: If CoT gains stem from structural templates, adherence should strongly correlate with task accuracy, providing a better explanation for the importance of prompt format than accuracy alone.

2.  **Confidence Analysis at Probability Projection Layer**:
    - **Function**: Observe whether CoT increases model certainty during the answer generation phase.
    - **Mechanism**: The authors track the token probability sequence for the common decision phrase "answer is..." and use KDE to compare probability distributions between CoT and standard prompts. For closed-set answer spaces (AQuA, Sports, Coin Flip), they extract top-k probabilities of candidate tokens to calculate entropy.
    - **Design Motivation**: If CoT narrows the decoding space, probabilities during answer token generation should be more concentrated and entropy lower, rather than simply outputting more text steps.

3.  **Task-Dependent Modulation at FFN Activation Layer**:
    - **Function**: Analyze whether CoT changes the way neurons participate in internal computation layers.
    - **Mechanism**: Units with FFN activation function outputs greater than 0 are defined as activated neurons. The authors count the average number of activated neurons across layers during generation and compare the difference between CoT and standard prompting. Global efficiency is assessed via total activation, while layer-wise differences identify which layers CoT impacts most.
    - **Design Motivation**: If CoT only changes output format, internal activations might not show systematic differences. However, the paper finds differences concentrated in the final 1/3 of layers, with directions differing between open-domain and closed-set tasks, suggesting CoT influences internal processing in a task-dependent manner.

### Loss & Training

This study does not train new models or use new loss functions. All models are pre-trained LLMs compared under standard and CoT prompts. Key measurements include: Pearson correlation and $R^2$ between structure adherence and accuracy; entropy of answer candidate probabilities; and the FFN activation count $A_t^{(l)}$, representing the number of neurons in layer $l$ with activation outputs $> 0$ for the $t$-th generated token.

## Key Experimental Results

### Main Results

| Dataset | Answer Space | LLaMA3.1-8B Standard Acc | LLaMA3.1-8B CoT Acc | Gain |
| :--- | :--- | :--- | :--- | :--- |
| AQuA | Closed-set options | 0.3110 | 0.4961 | +59.49% |
| Sports | Yes/No | 0.7497 | 0.9395 | +25.30% |
| Coin Flip | Yes/No | 0.4580 | 1.0000 | +118.34% |
| GSM8K | Open numeric | 0.1774 | 0.7771 | +338.03% |
| Date | Formatted date | 0.4417 | 0.7100 | +67.4% |
| Last Letter Concat | Open text | 0.0000 | 0.4496 | Inf. |

This table (based on LLaMA3.1-8B results in the appendix) demonstrates that CoT significantly improves accuracy across different answer spaces. The relative gains are highest in tasks where standard prompting is weak, such as GSM8K and Last Letter Concatenation, due to the structured intermediate steps.

### Ablation Study

| Object of Analysis | Key Values/Phenomena | Explanation |
| :--- | :--- | :--- |
| Structure Adherence vs. GSM8K Accuracy | Pearson $r=0.75$ to $0.92$; 4/5 models reach $p<0.05$; $R^2=0.57$ to $0.84$ | Structure adherence explains a large portion of performance variance. |
| CoT Token Probability | CoT "answer is..." probability distribution is higher and more concentrated than standard. | CoT prunes the decoding space and increases confidence at the answer stage. |
| Closed-set Answer Entropy | In AQuA/Sports/Coin Flip, entropy for correct answers is lower; CoT is lower than standard. | CoT makes the candidate answer distribution "sharper." |
| Total FFN Activation | LLaMA3.1-70B on AQuA: Standard activates ~820K neurons vs. CoT ~790K. | CoT reduces irrelevant activations in some tasks, creating more focused processing. |
| Layer-wise Activation Diff | Differences concentrated in the final 1/3 of layers; varies by task type. | CoT acts as a "pruner" in open-domain and an "amplifier" in closed-set tasks. |

### Key Findings

- CoT does not just make models output longer text; it makes generation adhere to a specific reasoning structure. Stronger structure adherence correlates with higher GSM8K accuracy.
- Evidence from the projection layer supports "decoding space pruning": answer token probabilities are more concentrated, and answer entropy is lower in closed-set tasks under CoT.
- Activation layers exhibit task dependency: in open-domain tasks, CoT tends to reduce later-layer activation (helping focus); in closed-set tasks, it increases later-layer activation (possibly helping the model compare finite options more thoroughly).

## Highlights & Insights
- **More Actionable Perspective on CoT**: The paper decomposes CoT's effects into three measurable phenomena—structure adherence, probability concentration, and activation modulation—rather than vaguely attributing it to "helping reasoning."
- **Stronger Evidence That Structure Outweighs Logic**: The high correlation between structure adherence and accuracy suggests that part of CoT's gain comes from constraining output paths into stable templates, regardless of whether every natural language step is perfectly logical.
- **Inspirational Task-Dependent Activation**: CoT acts as a pruner in open-domain tasks and an amplifier in closed-set tasks. This implies that future prompt or inference optimizations should not assume a uniform mechanism for CoT across all tasks.
- **Practical Implications for Efficient Inference**: Since structure and answer space constraints are key, one could design shorter, more structured prompts or trigger enhancements only in necessary later layers/stages rather than blindly extending the chain.

## Limitations & Future Work

- **Correlation is Not Causal Proof**: The authors note that while evidence linking structure adherence to accuracy and CoT to activation differences is strong, it remains heuristic or correlational and does not fully prove specific neural mechanisms.
- **Fragmented Granularity of Explanation**: The study separately analyzes decoding, projection, and activation, but has yet to synthesize them into a strictly unified causal chain. How different levels drive each other needs stronger intervention experiments.
- **Internal Model Remains a Black Box**: The functions of individual FFN neurons, the meaning of high-dimensional representations, and cross-layer information flow remain difficult to determine. Activated counts provide a coarse view of participation but do not specify where knowledge or algorithms are executed.
- **Limited Task Coverage**: Experiments focus on vanilla CoT and typical reasoning benchmarks. Mechanisms might differ for self-consistency, tool-augmented CoT, agentic planning, or long-context complex tasks.

## Related Work & Insights
- **vs. CoT Behavioral Analysis**: While earlier work observed that CoT improves accuracy or template imitation, this study quantifies template imitation via test points and Imitation Count, establishing statistical links to accuracy.
- **vs. Mechanistic Interpretability**: This study applies ideas from FFN memory, reasoning neurons, and logit lens research to CoT, specifically tracking how prompts alter output probability and FFN activation.
- **vs. Prompt Format Research**: Building on findings by Madaan et al. that structure affects CoT performance, this paper provides direct evidence from decoding, projection, and activation perspectives.
- **Inspiration for Prompt Design**: Rather than pursuing longer CoT, focus on designing structural templates that match the task's answer space; emphasize option comparison for closed-set tasks and step-wise pruning/entity constraints for open-domain tasks.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Systematic analysis across three information flow stages is valuable; task-dependent activation findings are particularly novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 6 models and 9 datasets with multiple analytical dimensions; however, some core results are presented as trends/graphs with incomplete tabular values.
- **Writing Quality**: ⭐⭐⭐⭐ Clear narrative and honest discussion of limitations, though some formula/table numbering in the source text is slightly disorganized.
- **Value**: ⭐⭐⭐⭐ Highly helpful for understanding CoT and designing efficient prompts; serves as an excellent reference for mechanistic explanation research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] How Far Ahead Do LLMs Plan? Uncovering the Latent Horizon in Chain-of-Thought Reasoning](../../ICML2026/llm_reasoning/how_far_ahead_do_llms_plan_uncovering_the_latent_horizon_in_chain-of-thought_rea.md)
- [\[CVPR 2026\] Rationale-Enhanced Decoding for Multi-modal Chain-of-Thought](../../CVPR2026/llm_reasoning/rationale-enhanced_decoding_for_multi-modal_chain-of-thought.md)
- [\[ACL 2026\] Reasoning Fails Where Step Flow Breaks](reasoning_fails_where_step_flow_breaks.md)
- [\[ACL 2026\] AIM-CoT: Active Information-driven Multimodal Chain-of-Thought for Vision-Language Reasoning](aim-cot_active_information-driven_multimodal_chain-of-thought_for_vision-languag.md)
- [\[ACL 2026\] Is Chain-of-Thought Really Not Explainability? Chain-of-Thought Can Be Faithful without Hint Verbalization](is_chain-of-thought_really_not_explainability_chain-of-thought_can_be_faithful_w.md)

</div>

<!-- RELATED:END -->

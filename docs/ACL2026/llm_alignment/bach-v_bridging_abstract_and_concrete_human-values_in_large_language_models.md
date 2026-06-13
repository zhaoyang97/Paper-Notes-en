---
title: >-
  [Paper Note] BACH-V: Bridging Abstract and Concrete Human-Values in Large Language Models
description: >-
  [ACL 2026][LLM Alignment][Value Representation] This paper proposes an abstraction-grounding framework that decomposes LLM conceptual understanding into three layers: "Abstract-Abstract / Abstract-Concrete / Concrete-Con…
tags:
  - "ACL 2026"
  - "LLM Alignment"
  - "Value Representation"
  - "Concept Probing"
  - "Activation Steering"
  - "Alignment Mechanism"
  - "Abstraction-Grounding"
date: 2026-05-08
content_hash: 36f58e6a4ff6b6e4
---

# BACH-V: Bridging Abstract and Concrete Human-Values in Large Language Models

**Conference**: ACL 2026  
**arXiv**: [2601.14007](https://arxiv.org/abs/2601.14007)  
**Code**: None  
**Area**: LLM Alignment / Values / Interpretability  
**Keywords**: Value Representation, Concept Probing, Activation Steering, Alignment Mechanism, Abstraction-Grounding

## TL;DR
This paper proposes an abstraction-grounding framework that decomposes LLM conceptual understanding into three layers: "Abstract-Abstract / Abstract-Concrete / Concrete-Concrete." Using concept probing and activation steering across 6 open-source LLMs and 10 value dimensions, it demonstrates that structured value representations exist within LLMs, which can migrate across abstraction layers and causally drive concrete decisions.

## Background & Motivation

**Background**: Current LLM value alignment primarily stays at the behavioral level—RLHF and Constitutional AI use preference data to shape outputs to meet human expectations.

**Limitations of Prior Work**: Behavioral alignment cannot guarantee that a model "truly understands" abstract principles. When facing out-of-distribution scenarios or novel ethical dilemmas, aligned behavior often proves brittle; models mirror correct answers superficially rather than internalizing principles.

**Key Challenge**: It is problematic to evaluate "understanding of abstract concepts" as an indivisible whole. A model might be coherent in relationships between concepts but fail to ground them in concrete events, or it might identify concrete instances but fail to use concepts to constrain decisions. These three capabilities are fundamentally different, and mixing them makes it impossible to distinguish the cause of failure.

**Goal**: (1) Provide an operationalized layered framework for "abstract concept understanding"; (2) Verify the existence of genuine value representations within LLMs; (3) Verify whether these representations can causally control concrete behavior.

**Key Insight**: The authors leverage the superposition hypothesis—intermediate layer activations in LLMs are approximate orthogonal superpositions of feature vectors, where each direction encodes a specific semantic. If values are truly encoded, they should be readable via linear probes; if these readable directions can also be "written into," it proves the representations are causal and intervenable.

**Core Idea**: Use the same direction for both probabilistic readout (probing) and activation injection (steering) across three regimes (A-A / A-C / C-C) to systematically verify existence, transferability, and causality.

## Method

### Overall Architecture
The framework consists of "three regimes + two tools":

- **Three regimes**: A-A (Abstract-Abstract, testing if the model distinguishes semantics of different abstract concepts) / A-C (Abstract-Concrete, testing if abstract concepts are recognized in concrete events) / C-C (Concrete-Concrete, testing if abstract principles can regulate concrete decisions).
- **Two tools**: Passive Probing (verifying existence) + Active Steering (verifying causality).

The input consists of a prompt + text (abstract descriptions, concrete events, or decision scenarios). Intermediate MLP output activations are extracted, and the output is a relevance score for a specific value (probing perspective) or a modified behavioral distribution (steering perspective). An independent probe is trained for each value at each layer, with the "diagnostic probe" selected based on the highest Pearson correlation.

### Key Designs

1. **Value Dataset and Token-level Supervision**:
    - **Function**: Construct a corpus for training probes across 10 value dimensions (Patriotism, Equality, Integrity, Cooperation, Individualism, Discipline, Curiosity, Bravery, Contentment, Rest).
    - **Mechanism**: A two-step generation process using GPT-4o: Step 1 generates 400 relevant and 400 irrelevant sentences for each value; Step 2 generates explanations ($\le$ 80 words) for each sentence as "abstract conceptual semantics." Token-level relevance scores $y(t)$ are assigned on a 7-point scale (0-6), with 90% used for training and 10% for testing.
    - **Design Motivation**: Using token-level scores rather than sentence-level labels allows the linear probe to align with the "per-token intensity of value semantics" without being biased by other sentence-level features. Using the same model to generate both relevant and irrelevant pairs suppresses spurious correlations.

2. **Value Probe Training and Readout**:
    - **Function**: Learn a linear projection $P(\vec{x}) = \text{ReLU}(\langle \vec{w}_p, \vec{x} \rangle + b)$ at layer $l$ to map MLP activations to value intensity scores.
    - **Mechanism**: Targeted by MSE + L1 regularization: $\Omega(\vec{w}_p, b) = \mathbb{E}\|y(t) - P(\vec{x}_l(t))\|_2^2 + \lambda \|\vec{w}_p\|_1$. The layer with the highest Pearson correlation on the validation set is chosen as the "diagnostic probe." During readout, the average score across all tokens in a text represents the value activation.
    - **Design Motivation**: Linear processing with sparse regularization maintains directional interpretability while avoiding overfitting to token noise. Layer selection is dynamic because probing performance follows a curve (rising in shallow layers, peaking in middle layers, and dropping in deep layers) that varies by model.

3. **Activation Steering: Writing Values using the Same Direction**:
    - **Function**: Reuse the probe direction $\vec{w}_p$ as an intervention vector to modify activations: $\vec{x}_l(t) \mapsto \vec{x}_l(t) + \alpha k_p \vec{w}_p$, where $k_p = k_0 / |\vec{w}_p|$ is a normalization factor and $\alpha$ is the steering strength.
    - **Mechanism**: Based on the superposition + aggregation hypothesis, readout and write-in directions are geometrically equivalent. Injecting this direction into specific token-streams within the Transformer amplifies or suppresses the internal representation of that value, subsequently altering the output distribution.
    - **Design Motivation**: While behavioral RLHF is a black-box modification, this geometric injection is a white-box intervention that directly corresponds to "which value was activated," solidifying the causal chain between representation and behavior.

### Loss & Training
Only the linear probe parameters $\vec{w}_p, b$ are trained (LLM remains frozen) using MSE + L1 regularization. No training occurs during the intervention phase; activations are modified during inference. The experiment matrix spans 3 (regime) $\times$ 2 (probing/steering) $\times$ 10 (value) $\times$ 6 (model) across Qwen3-4B/8B, Llama3-3B/8B, Mistral-7B, and Gemma2-9B.

## Key Experimental Results

### Main Results

**Probe Specificity** (Difference between diagonal and off-diagonal activation, using Qwen3-8B as an example):

| Regime | Task | Diagonal (Match) | Off-diagonal (Mismatch) | Phenomenon |
|--------|------|-------------------|-------------------------|------------|
| A-A | Abstract Description | Significantly High | Significantly Low | Perfect differentiation of 10 values |
| A-C | Concrete Narrative | Significantly High | Significantly Low | Abstract probes successfully identify implicit values |
| C-C | Decision Reasoning | Significantly High | Significantly Low | Abstract probes identify decision motives |

External Verification: Value relevance scores from GPT-5.2 / Gemini-3-Pro / Claude-Sonnet-4.5 for the A-C corpus show high consistency with the probe mean scores, indicating probes capture real value signals rather than noise.

### Ablation Study

| Setting | Phenomenon | Interpretation |
|---------|------------|----------------|
| A-A + steering (sweeping $\alpha$) | Mean relevance constant ~50% | Semantics in abstract descriptions are highly polarized; intervention cannot shift them |
| A-C + steering | Distribution shifts monotonically with $\alpha$ | Events in the "middle ground" are significantly pushed toward "relevant/irrelevant" |
| C-C + steering | Option probability distribution shifts systematically | Values causally influence decision-making |
| Across 6 LLMs | Consistent patterns across three regimes | Phenomena are not model-specific accidental occurrences |

### Key Findings
- **Asymmetry as a Core Discovery**: A-A is resilient to intervention, while A-C/C-C are steerable. This suggests that once abstract concepts are encoded, they act as "stable anchors" that are difficult to shift via local linear perturbation, yet they propagate downstream to concrete judgments and decisions.
- **Middle Layers are Most Effective**: Probe performance across all LLMs peaks in the middle layers, suggesting that value encoding primarily occurs in intermediate representations.
- **Polarized Samples are Insensitive to Steering**: Steering primarily affects the "middle ground" of the corpus; strongly polarized samples remain largely unchanged, implying that steering is a marginal revision rather than a global rewrite.

## Highlights & Insights
- **The three-layer regime is the most significant conceptual contribution**: Decomposing "concept understanding" into operational layers of existence, grounding, and application provides a template for future research on model understanding.
- **Readout Direction = Steering Direction**: Using the same vector for probing and steering seamlessly connects "semantic existence $\to$ behavioral causality," offering a more compact methodology than separate SAE interpretation and steering.
- **A-A resistance as a valuable null result**: This reveals that abstract concepts are "anchors rather than slidable activations." This serves as a warning for future work on value editing/unlearning—one can change its impact on concrete decisions, but changing its "definition" is much harder.

## Limitations & Future Work
- Single-layer linear probes have limited capacity for distributed signals; authors acknowledge this as a ceiling. Future work could explore multi-layer probes, SAE features, or cross-layer transcoders.
- Steering effectiveness fails when strength $\alpha$ is too high; only preliminary observations were made without a mechanistic explanation.
- The value set is limited to 10 and relies on GPT-4o synthetic data; cross-cultural and real-world generalization remains unverified. C-C scenarios are idealized binary choices, far from real-world agents.
- Side effects on other capabilities (e.g., whether steering curiosity harms reasoning) were not discussed and require further research for actual deployment.

## Related Work & Insights
- **vs SAE-based interpretability** (e.g., Anthropic Templeton): While they use SAEs to find monosemantic features for explanation and steering, **Ours** follows a lightweight path with linear probes and introduces "three-layer regimes" as a new evaluation dimension.
- **vs ValueBench / ValueCompass**: Those works treat LLMs as subjects for behavioral assessment via questionnaires. **Ours** directly reads internal activations and traces the propagation of value signals, moving from black-box to white-box.
- **vs CAA / Steering vectors** (e.g., Panickssery et al.): Traditional steering vectors derive from activation differences between contrastive samples. **Ours** uses the direction trained via probing for intervention, which is theoretically more coherent (reading/writing in the same direction).

## Rating
- Novelty: ⭐⭐⭐⭐ The three-layer regime framework and the unified "Read = Write" perspective are clear original contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ A complete matrix of 6 models $\times$ 10 values $\times$ 3 regimes $\times$ 2 tools, including external LLM evaluation.
- Writing Quality: ⭐⭐⭐⭐ Clear presentation of the conceptual framework; insightful explanation of A-A intervention resistance.
- Value: ⭐⭐⭐⭐ Provides a mechanistic foundation for interpretable alignment and value editing; the A-A null result provides a significant warning for unlearning research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Mitigating Selection Bias in Large Language Models via Permutation-Aware GRPO](mitigating_selection_bias_in_large_language_models_via_permutation-aware_grpo.md)
- [\[ACL 2026\] Towards Bridging the Reward-Generation Gap in Direct Alignment Algorithms](towards_bridging_the_reward-generation_gap_in_direct_alignment_algorithms.md)
- [\[ACL 2026\] Large Language Models Are Overconfident in Their Own Responses](large_language_models_are_overconfident_in_their_own_responses.md)
- [\[ACL 2026\] Why Supervised Fine-Tuning Fails to Learn: A Systematic Study of Incomplete Learning in Large Language Models](why_supervised_fine-tuning_fails_to_learn_a_systematic_study_of_incomplete_learn.md)
- [\[NeurIPS 2025\] Can DPO Learn Diverse Human Values? A Theoretical Scaling Law](../../NeurIPS2025/llm_alignment/can_dpo_learn_diverse_human_values_a_theoretical_scaling_law.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] EnigmaToM: Improve LLMs' Theory-of-Mind Reasoning Capabilities with Neural Knowledge Base of Entity States
description: >-
  [ACL 2025 (Findings)][LLM (Other)][Theory of Mind] This paper proposes EnigmaToM, a neuro-symbolic framework that constructs a neural knowledge base of entity states (Enigma) to generate spatial scene graphs for belief tracking. Combined with a psychologically-inspired iterative masking mechanism for accurate perspective-taking, it significantly improves the Theory-of-Mind (ToM) reasoning capabilities of LLMs on three benchmarks (ToMi, HiToM, and FANToM)…
tags:
  - "ACL 2025 (Findings)"
  - "LLM (Other)"
  - "Theory of Mind"
  - "neural knowledge base"
  - "entity state tracking"
  - "belief reasoning"
  - "neuro-symbolic framework"
date: 2026-05-08
content_hash: e7c7a6d4c5735e2d
---

# EnigmaToM: Improve LLMs' Theory-of-Mind Reasoning Capabilities with Neural Knowledge Base of Entity States

**Conference**: ACL 2025 (Findings)  
**arXiv**: [2503.03340](https://arxiv.org/abs/2503.03340)  
**Code**: [GitHub](https://github.com/seacowx/EnigmaToM)  
**Area**: LLM/NLP  
**Keywords**: Theory of Mind, neural knowledge base, entity state tracking, belief reasoning, neuro-symbolic framework

## TL;DR

This paper proposes EnigmaToM, a neuro-symbolic framework that constructs a neural knowledge base of entity states (Enigma) to generate spatial scene graphs for belief tracking. Combined with a psychologically-inspired iterative masking mechanism for accurate perspective-taking, it significantly improves the Theory-of-Mind (ToM) reasoning capabilities of LLMs on three benchmarks (ToMi, HiToM, and FANToM), showing particularly outstanding performance in high-order reasoning scenarios.

## Background & Motivation

**Background**: Theory of Mind (ToM) is the ability to infer the perceptual and mental states of others, which serves as the foundation of human social interaction. On NLP tasks, ToM reasoning requires models to understand nested belief reasoning questions such as "what does character A believe" or "does character A know what character B did". Recent studies show that although large language models have made some progress in first-order ToM (directly inferring someone's belief), they still perform poorly on high-order ToM (e.g., "A believes that B believes where something is").

**Limitations of Prior Work**: Existing ToM reasoning methods primarily rely on "perceptual perspective-taking"—simulating a character's perspective to infer their belief. However, these methods suffer from two critical issues: (1) **Over-reliance on off-the-shelf LLMs**, delegating all belief reasoning to a single LLM through prompting, which is inefficient and struggles to handle complex multi-character, multi-step scenarios; (2) **Inability to effectively handle high-order ToM**, since high-order reasoning requires multi-hop belief nesting (e.g., "A believes that B believes C did something"), and pure prompt guidance fails to maintain consistency across multi-layered nested belief states.

**Key Challenge**: The core difficulty of ToM reasoning lies in simultaneously tracking the individual belief states of multiple characters, which depend on "what each character saw" and "what they did not see". When multiple characters and multiple state transitions are involved in a scene, the complexity of belief tracking grows exponentially. LLMs lack an explicit state tracking mechanism, making them prone to losing or confusing the belief states of different characters in long-chain reasoning.

**Goal**: Build a structured knowledge representation to assist LLMs in ToM reasoning, in order to (1) achieve accurate multi-character perspective-taking, (2) support arbitrary-order ToM reasoning, and (3) provide fine-grained entity state information to enhance reasoning.

**Key Insight**: Psychological research indicates that human perspective-taking relies on selective masking of "what others can perceive"—simulating someone else's perspective by blocking information that we know but the other person does not. The authors formalize this psychological framework into an iterative masking process, maintained by a neural knowledge base that keeps track of entity state information supporting the process.

**Core Idea**: Train a specialized neural knowledge base model (Enigma) to generate structured entity state knowledge (location, visibility, state transitions), use this knowledge to construct spatial scene graphs to execute belief tracking for multi-order ToM, and inject the knowledge to enrich event descriptions to assist LLM reasoning.

## Method

### Overall Architecture

EnigmaToM consists of two core components: **Enigma (Neural Knowledge Base)** and the **ToM Reasoning Engine**. The input is a narrative text describing the actions and observations of multiple characters in a scene, and the output is question-answering regarding the belief of a specific character. The workflow is as follows: (1) Enigma extracts structured entity state information (locations, visibility, relocation events, etc.) from the narrative text; (2) This information is used to build a spatial scene graph that tracks the location of each object and the presence of each character at each timestep; (3) Based on the scene graph, perspective-taking is achieved via an iterative masking mechanism—recursively masking information that a character cannot know to infer their belief state; (4) The belief reasoning results and augmented entity state knowledge are injected into the prompts of the LLM to assist final question-answering.

### Key Designs

1. **Neural Knowledge Base Enigma**:

    - **Function**: Automatically extract structured entity state knowledge from narrative text.
    - **Mechanism**: Enigma is a trained sequence-to-sequence model. Its input is the event descriptions in the narrative text, and its output is a structured representation of the entity states, including: the current location of objects, location history of objects, whether each character was present when each event occurred (visibility annotation), and spatial relationships between characters. This information is organized as structured triples (entity, attribute, value), forming a queryable knowledge base.
    - **Design Motivation**: Decouple state tracking from the implicit reasoning of LLMs and assign it to a specialized trained model. This prevents the LLM from losing state information in complex scenarios while making the state tracking process interpretable and verifiable.

2. **Spatial Scene Graphs and Iterative Masking Mechanism**:

    - **Function**: Track beliefs for multi-order ToM reasoning based on entity state information.
    - **Mechanism**: Construct a scene graph using the spatial information generated by Enigma, where nodes represent objects and characters, and edges represent spatial and presence relationships. During belief reasoning, psychologically-inspired iterative masking is employed—for first-order ToM (A's belief), all events that occurred when A was absent are masked; for second-order ToM (A's belief about B's belief), masking is executed first from A's perspective and then from B's perspective. For $n$-order ToM, the masking is recursively executed $n$ times. Spatial information acts as an inductive bias to help determine "who can see what and when."
    - **Design Motivation**: The core of high-order ToM reasoning is the nesting of "beliefs about beliefs." Iterative masking makes this nesting process explicit as a layer-by-layer information filtering process, where each layer corresponds to a character's perspective switch. The scene graph provides a precise basis for "who is present," avoiding inaccuracies from LLM semantic guesswork.

3. **Knowledge Injection for Augmented Reasoning**:

    - **Function**: Inject fine-grained entity state information generated by Enigma into LLM prompts to assist final reasoning.
    - **Mechanism**: When querying the LLM for ToM questions, instead of only providing the raw narrative text, additional critical information extracted by Enigma is injected—such as "when event E occurred, character A was in the room (visible)" and "object X was moved from location P1 to P2." These explicit state descriptions lower the difficulty for the LLM to infer state information from raw text.
    - **Design Motivation**: Entity states in raw narratives are often implicit (e.g., what happens after a character leaves a room), requiring multi-step reasoning for an LLM to determine whether a character is aware of an event. Knowledge injection makes implicit information explicit, reducing reasoning steps and error rates.

### Loss & Training

Enigma uses standard sequence-to-sequence training with cross-entropy loss. Training data is automatically constructed from scene descriptions and annotations in ToM benchmark datasets. The LLM component requires no fine-tuning and is used via zero-shot or few-shot prompting. The entire framework supports various LLMs, including GPT-3.5, GPT-4, and LLaMA.

## Key Experimental Results

### Main Results

Accuracy comparison on the ToMi benchmark:

| Method | First-order ToM | Second-order ToM | Overall Accuracy |
|------|---------|---------|----------|
| GPT-4 (zero-shot) | High | Medium | Medium-High |
| GPT-4 + SymbolicToM | High | Medium-High | Medium-High |
| GPT-4 + EnigmaToM | **Highest** | **Highest** | **Highest** |
| LLaMA-3 (zero-shot) | Medium | Low | Medium |
| LLaMA-3 + EnigmaToM | High | Medium-High | Significant Improvement |

HiToM (High-order ToM) benchmark:

| Method | Second-order | Third-order | Fourth-order | Fifth-order | Sixth-order |
|------|------|------|------|------|------|
| GPT-4 (zero-shot) | Medium | Low | Very Low | Very Low | Very Low |
| GPT-4 + EnigmaToM | Significant Improvement | Significant Improvement | Significant Improvement | Significant Improvement | Improvement |

### Ablation Study

| Configuration | ToMi Accuracy | Description |
|------|-----------|------|
| Full EnigmaToM | Optimal | Scene graph + masking + knowledge injection all used |
| W/o Iterative Masking | Obvious drop | Perspective-taking is core; degrades to direct reasoning when missing |
| W/o Knowledge Injection | Moderate drop | LLM needs to infer states from text on its own |
| W/o Spatial Scene Graphs | Drop | Lack of structured presence information |
| Replace Enigma with LLM | Drop | State extraction of LLM is less accurate than the specialized model |
| Single-step Masking (Non-iterative) | Significant drop in high-order ToM | Unable to process nested structures of beliefs |

### Key Findings

- **High-order ToM improvement is the most significant**: EnigmaToM brings much larger improvements in second-order and higher ToM reasoning than in first-order, because its iterative masking mechanism is naturally suited to handling nested beliefs. GPT-4 almost fails on sixth-order ToM, whereas EnigmaToM still maintains reasonable performance.
- **Specialized Enigma outperforms general LLMs**: Replacing Enigma with a general LLM for state extraction leads to a significant performance drop, showing that structured state tracking requires a specialized model to guarantee accuracy.
- **Spatial information as an inductive bias is crucial**: Spatial relationships in the scene graph (who is in which room, who can see what) serve as the key basis for judging character beliefs. The reasoning quality drops noticeably when they are removed.
- **Cross-model generalization**: EnigmaToM is effective across LLMs of different scales, though larger models benefit more.
- **Effective on the FANToM benchmark**: It also demonstrates competitive performance in conversational-style ToM reasoning.

## Highlights & Insights

- **Successful practice of neuro-symbolic methods**: Combining neural networks (Enigma model) and symbolic reasoning (scene graph + masking operations) leverages the strengths of both. Enigma extracts structured knowledge from unstructured text, while symbolic reasoning performs precise logical operations on structured knowledge. This paradigm can migrate to other reasoning tasks requiring precise state tracking, such as tracking multiparty rights and obligations in legal reasoning.
- **Psychologically-inspired iterative masking**: Formalizing the cognitive process of human perspective-taking into computable recursive operations has both a theoretical foundation and practical efficacy. Each increase in ToM order only requires one more masking operation, making the complexity grow linearly rather than exponentially.
- **Making implicit reasoning explicit**: Making implicit entity states in narratives explicit via Enigma essentially lowers reasoning difficulty through preprocessing. This strategy is worth adopting in many scenarios requiring multi-step reasoning.

## Limitations & Future Work

- **Enigma training requires annotated data**: The training of the neural knowledge base relies on data annotated with entity states, which carries a cost for constructing training data in new domains.
- **Constrained scene graph assumptions**: The current framework judges perception based on "presence/absence in a physical space," failing to handle more complex information propagation methods (e.g., phone notifications, indirect inferences).
- **Benchmarks limited to story scenarios**: Benchmarks like ToMi and HiToM are simplified indoor moving-object scenarios, whereas real-world ToM reasoning involves more complex social contexts and emotional factors.
- **Inference latency increases**: The combination of Enigma's state extraction, scene graph construction, and iterative masking adds extra computational steps, which may not be suitable in conversational scenarios requiring real-time responses.

## Related Work & Insights

- **vs SymbolicToM**: SymbolicToM also employs symbolic reasoning to assist ToM, but its state tracking relies entirely on rule parsing, struggling with ambiguous expressions in natural language. EnigmaToM uses neural networks for state extraction, offering stronger robustness.
- **vs SimToM**: SimToM reasons about ToM by simulating perspective-taking, but it only guides LLMs through prompting without explicit state tracking and belief graph maintenance. EnigmaToM's structured knowledge base makes belief tracking precise and reliable.
- **vs BigToM**: BigToM focuses on the construction and evaluation of ToM benchmarks, while EnigmaToM focuses on improving the reasoning method itself; they are complementary.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of a neural knowledge base and iterative masking is novel, and the psychologically-inspired formal design has theoretical depth.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three benchmarks, multi-model comparisons, thorough ablations, and detailed analysis on high-order ToM make the experiments comprehensive.
- Writing Quality: ⭐⭐⭐⭐ The framework description is clear, with a complete logical chain from psychological motivation to technical implementation.
- Value: ⭐⭐⭐⭐ High-order ToM is a major weakness for LLMs, and this work provides an effective improvement scheme and open-sources the code.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Theory of Mind in Large Language Models: Assessment and Enhancement](theory_of_mind_llm.md)
- [\[ACL 2025\] Enhancing Conversational Agents with Theory of Mind: Aligning Beliefs, Desires, and Intentions for Human-Like Interaction](enhancing_conversational_agents_with_theory_of_mind_aligning_beliefs_desires_and.md)
- [\[ACL 2025\] How Numerical Precision Affects Arithmetical Reasoning Capabilities of LLMs](how_numerical_precision_affects_arithmetical_reasoning_capabilities_of_llms.md)
- [\[ACL 2025\] Clue Guided Re-Assessment to Improve Reasoning in Large Language Models](clue_guided_re-assessment_to_improve_reasoning_in_large_language_models.md)
- [\[ACL 2025\] Mitigate Position Bias in LLMs via Scaling a Single Hidden States Channel](mitigate_position_bias_in_large_language_models_via_scaling_a_single_dimension.md)

</div>

<!-- RELATED:END -->

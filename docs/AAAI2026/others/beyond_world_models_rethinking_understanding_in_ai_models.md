---
title: >-
  [Paper Note] Beyond World Models: Rethinking Understanding in AI Models
description: >-
  [AAAI 2026][World Models] Through three case studies drawn from the philosophy of science — a domino computer, a mathematical proof, and Bohr's atomic theory — this paper argues that the world model framework is insufficient to characterize human-level "understanding," demonstrating that tracking states and state transitions alone cannot capture the abstract reasoning, motivational insight, and problem-context awareness that understanding requires.
tags:
  - "AAAI 2026"
  - "World Models"
  - "AI Understanding"
  - "Philosophy of Science"
  - "Cognitive Science"
  - "Representation Learning"
date: 2026-05-08
content_hash: d18f4237a6016897
---

# Beyond World Models: Rethinking Understanding in AI Models

**Conference**: AAAI 2026
**arXiv**: [2511.12239](https://arxiv.org/abs/2511.12239)  
**Code**: None  
**Area**: Other
**Keywords**: World Models, AI Understanding, Philosophy of Science, Cognitive Science, Representation Learning

## TL;DR

Through three case studies drawn from the philosophy of science — a domino computer, a mathematical proof, and Bohr's atomic theory — this paper argues that the world model framework is insufficient to characterize human-level "understanding," demonstrating that tracking states and state transitions alone cannot capture the abstract reasoning, motivational insight, and problem-context awareness that understanding requires.

## Background & Motivation

### The Rise of World Models and Their Core Assumptions

In recent years, "world models" have become a prominent concept in the AI community. LeCun and colleagues argue that world models are a key pathway toward general intelligence, with the central idea that AI systems should construct internal representations to simulate the external world — tracking entities, states, and causal relations to predict consequences. This contrasts with representations that rely purely on statistical correlations.

A key motivating assumption is that since humans possess mental world models (e.g., the heliocentric model of the solar system), the presence of analogous representations in AI models would suggest that these models "understand" the world in a "human-like" manner. The landmark finding of Li et al. (2023) in Othello-GPT — that language models can learn internal representations of board states — has been widely cited as compelling evidence for the existence of world models.

### The Paper's Central Argument

The authors explicitly clarify that **they do not deny the possibility of AI achieving understanding**; rather, they argue that the world model framework is an inadequate theoretical lens for characterizing "understanding." AI systems may develop understanding through mechanisms entirely different from world models. The authors employ a case-study methodology, drawing on classical analyses from the philosophy of science literature to expose the limitations of the world model concept in capturing human-level understanding.

### Definition and State of the Field

The paper notes that "world model" lacks a unified definition across the literature. Following the mainstream conception, the authors define a world model as **an internal representation that tracks objects, their states, and the rules governing state transitions**. Current research primarily employs probing techniques to detect internal representations in neural networks, analyzing features and activation patterns at specific layers. However, some studies suggest that these world models may not constitute clear, human-like mental models, but rather learned collections of heuristics.

## Method

### Overall Architecture

Rather than proposing a new method, this paper systematically argues for the inadequacy of the world model framework through three carefully selected philosophical case studies. Each case targets a specific dimension along which the gap between "world model capability" and "human understanding" is most pronounced.

### Key Designs

#### 1. **Case Study 1: The Domino Computer — The Absence of Abstract Concepts**

Hofstadter proposed a thought experiment: a spring-driven arrangement of dominoes constituting a mechanical computer for testing whether a number is prime. When 641 dominoes are loaded and the system is triggered, a series of branching and looping signal propagations tests divisibility by various factors.

**What a world model does**: It tracks the state of each domino (standing/fallen) and simulates the physical propagation of falls. When asked "why does a particular domino never fall?", a world model can only answer "because no adjacent domino fell toward it," tracing the causal chain backward step by step.

**What understanding requires**: Genuine understanding lies in recognizing that "641 is prime" — an **abstract mathematical property** that explains the entire behavioral pattern of the dominoes. No amount of state tracking reveals the qualitative concept governing the system's behavior. This illustrates that world models, by their choice of what counts as a "state," inherently omit critical levels of abstraction.

#### 2. **Case Study 2: Mathematical Proof — Verification Is Not Understanding**

The authors invoke Poincaré's classical distinction: understanding a proof is not merely verifying that each syllogistic step is correct, but knowing **why these steps are organized in this particular order rather than some other**.

Consider Zagier's one-sentence proof that every prime $p \equiv 1 \pmod{4}$ is a sum of two squares:
- Verifying this proof requires only basic knowledge of set theory, involutions, and fixed points
- Understanding requires: identifying which steps constitute key insights (as opposed to routine verification); explaining why constructing the set $S = \{(x,y,z) \in \mathbb{N}^3 : x^2 + 4yz = p\}$ is a natural choice; and providing a high-level strategic overview — cleverly counting the same set in two ways by exploiting properties of involutions

A world model can track the sequence of logical state transitions in the proof, but cannot explain the organizational motivation behind those steps. In Poincaré's terms, to an agent with only world-model-style understanding, the state transitions in a proof would appear to be "arbitrarily generated."

#### 3. **Case Study 3: Bohr's Atomic Theory — The Importance of Problem Context**

Popper's analysis demonstrates that understanding a physical theory requires grasping the **problem situation** that motivated its proposal — not only the problem to be solved, but also the inadequacies of existing theories and the specific explanatory gaps to be filled.

Bohr proposed an atomic model in which electrons transition between discrete energy levels. The key to understanding this model does not lie in visualizing electrons jumping between orbits (which is precisely what a world model does), but in recognizing **why Bohr introduced this seemingly unnatural model — to explain the discrete spectral lines observed in atomic spectra**. Without this motivation, one cannot understand the theory of electrons constrained to specific orbits undergoing quantum jumps.

As Popper observed: a person presented with Bohr's theory without knowing that it was intended to explain discrete spectral lines cannot possibly understand its significance as a solution to a specific problem situation.

### Responses to Potential Objections

A natural objection is that more abstract states (e.g., "641 is prime" as a state) could be incorporated into a world model. The authors respond that doing so renders the framework **unfalsifiable** — if states can encode arbitrarily complex abstractions, any phenomenon can be post-hoc subsumed into the world model framework by defining appropriately abstract states. This flexibility undermines the explanatory contribution of the world model concept itself: the real explanatory work is done by the state representation, not by the world model's dynamic mechanism. In essence, this objection reduces to "if you put understanding into the states, world models can capture understanding" — a circular argument.

## Key Experimental Results

This is a purely theoretical/philosophical paper and contains no quantitative experiments in the conventional sense. Its "experiments" consist of three carefully constructed case analyses, each illuminating a different dimension of the world model framework's limitations.

### Main Results: Comparative Analysis of Three Case Studies

| Case Study | What World Models Can Do | Dimension of Understanding World Models Cannot Capture | Missing Capability |
|---|---|---|---|
| Domino Computer | Track domino states, physical causal propagation | Abstract mathematical concepts (primality) | Cross-level abstract reasoning |
| Mathematical Proof | Verify correctness of logical steps | Organizational motivation and key insights of the proof | Distinguishing key steps from routine steps |
| Bohr's Atomic Theory | Simulate electron orbital transitions | Problem context and explanatory structure of the theory | Grasping the explanatory goals of a theory |

### Ablation Study: Analysis of Possible Objections

| Strategy | Resolves the Problem? | Cost | Notes |
|---|---|---|---|
| Retain current state definitions | No | None | All three cases demonstrate limitations |
| Introduce abstract concepts as states | Formally yes | Loss of falsifiability | Framework becomes unfalsifiable, loses explanatory power |
| Abandon pursuit of a unified theory | Partially | Theoretical fragmentation | Evades the core problem |

### Key Findings

1. **The Problem of Abstraction Level**: World models, by their design, tend toward physical-level entity tracking rather than abstract conceptual representation when choosing state granularity.
2. **The Gap Between Simulation and Understanding**: The ability to simulate/predict does not entail understanding, even when state transitions are perfectly accurate.
3. **Absence of Contextual Understanding**: Understanding involves not only "what" and "how" but also "why it was designed this way" and "what problem it solves."
4. **The Falsifiability Dilemma**: Attempts to remedy the world model framework by expanding the state space lead to a theory that is no longer falsifiable.

## Highlights & Insights

1. **The Value of an Interdisciplinary Perspective**: By bringing classical philosophical analyses of "understanding" into AI research, the paper offers an important theoretical reflection on current world model research — a perspective that is notably rare in AI conference publications.
2. **Elegance of the Argumentative Strategy**: Rather than attempting to define "understanding" in a unified way (a problem unresolved in epistemology to this day), the paper selectively demonstrates specific blind spots of the world model framework through targeted cases.
3. **Constructive Contribution to AI Understanding Research**: The paper does not deny the possibility of AI achieving understanding; instead, it points toward directions beyond the world model framework — calling attention to abstract reasoning, problem-context awareness, and explanatory structure.
4. **Clever Use of the Curry–Howard Correspondence**: The paper naturally extends the world model framework to mathematical reasoning by exploiting the isomorphism between computation and proof.

## Limitations & Future Work

1. **Selection Bias in Case Studies**: The authors acknowledge that their case selection is intentional, and that other aspects of understanding may align more naturally with world models.
2. **Absence of a Constructive Alternative**: The paper effectively identifies the problem but does not propose a new framework to replace world models.
3. **Purely Theoretical Analysis**: The lack of empirical experiments makes it difficult to directly guide the design of AI systems.
4. **Vagueness of the Concept of "Understanding"**: Although the authors deliberately avoid defining "understanding" to prevent circular reasoning, this also leaves the scope of the conclusions insufficiently demarcated.
5. **Insufficient Discussion of Emerging LLM Capabilities**: The paper does not address whether current large language models already partially exhibit forms of "understanding" that go beyond world models in some of the discussed cases.

## Related Work & Insights

- **World Model Research**: Li et al. (2023)'s Othello-GPT represents a landmark contribution to world models in AI; LeCun's (2022) JEPA framework positions world models as central to the path toward AGI.
- **Philosophical Foundations**: Poincaré's seminal analysis of mathematical understanding and Popper's analysis of theoretical understanding provide the theoretical tools employed in this paper.
- **Implications for AI Safety**: If AI "understanding" cannot be fully characterized by world models, then arguments for AI understanding ability grounded in the existence of world models warrant greater scrutiny.
- **Implications for Benchmark Design**: Current benchmarks for evaluating AI understanding largely remain at the level of "prediction/simulation" and may need to be redesigned to test abstract reasoning and problem-context awareness.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (Philosophical perspective applied to AI research; case selection is elegant)
- Experimental Thoroughness: ⭐⭐⭐ (Purely theoretical paper; case analyses are insightful but lack empirical support)
- Writing Quality: ⭐⭐⭐⭐⭐ (Argumentation is clear and rigorous; structure is well-crafted)
- Value: ⭐⭐⭐⭐ (Offers important insights for AI community discourse on "understanding," though practical guidance is limited)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] General Agents Contain World Models](../../ICML2025/others/general_agents_contain_world_models.md)
- [\[ICML 2026\] iWorld-Bench: A Benchmark for Interactive World Models with a Unified Action Generation Framework](../../ICML2026/others/iworld-bench_a_benchmark_for_interactive_world_models_with_a_unified_action_gene.md)
- [\[ICLR 2026\] Building Spatial World Models from Sparse Transitional Episodic Memories](../../ICLR2026/others/building_spatial_world_models_from_sparse_transitional_episodic_memories.md)
- [\[ICLR 2026\] LPWM: Latent Particle World Models for Object-Centric Stochastic Dynamics](../../ICLR2026/others/latent_particle_world_models_self-supervised_object-centric_stochastic_dynamics_.md)
- [\[AAAI 2026\] TaylorPODA: A Taylor Expansion-Based Method to Improve Post-Hoc Attributions for Opaque Models](taylorpoda_a_taylor_expansion-based_method_to_improve_post-hoc_attributions_for_.md)

</div>

<!-- RELATED:END -->

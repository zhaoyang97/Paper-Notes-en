---
title: >-
  [Paper Note] The Non-Linear Representation Dilemma: Is Causal Abstraction Enough for Mechanistic Interpretability?
description: >-
  [NeurIPS 2025 (Spotlight)][Interpretability][Causal Abstraction] This paper proves that when alignment maps in causal abstraction are unconstrained by linearity, any neural network can be mapped to any algorithm…
tags:
  - "NeurIPS 2025 (Spotlight)"
  - "Interpretability"
  - "Causal Abstraction"
  - "mechanistic interpretability"
  - "Linear Representation Hypothesis"
  - "Alignment Map"
  - "Interchange Intervention"
date: 2026-05-08
content_hash: 7da0af2cdd441af2
---

# The Non-Linear Representation Dilemma: Is Causal Abstraction Enough for Mechanistic Interpretability?

**Conference**: NeurIPS 2025 (Spotlight)
**arXiv**: [2507.08802](https://arxiv.org/abs/2507.08802)  
**Code**: To be confirmed  
**Area**: Interpretability
**Keywords**: Causal Abstraction, mechanistic interpretability, Linear Representation Hypothesis, Alignment Map, Interchange Intervention

## TL;DR

This paper proves that when alignment maps in causal abstraction are unconstrained by linearity, any neural network can be mapped to any algorithm, rendering causal abstraction trivial and uninformative. This gives rise to the "non-linear representation dilemma"—the absence of a principled trade-off between the complexity and the fidelity of alignment maps.

## Background & Motivation

Causal abstraction is a central methodology in mechanistic interpretability, aiming to "decode" a model's decision process by identifying mappings between neural networks and high-level algorithms. However, the theoretical foundations of this methodology itself merit scrutiny:

1. **The Linear Representation Hypothesis drives practice**: Most current interpretability work implicitly relies on the Linear Representation Hypothesis—that models linearly encode features in representation space—and thus employs linear probes or linear alignment maps to establish correspondences between networks and algorithms. Yet linearity is not part of the formal definition of causal abstraction; it is a convenient but insufficiently justified assumption.
2. **The expressive power of non-linear maps remains underexplored**: The mathematical definition of causal abstraction permits arbitrary measurable functions as alignment maps, yet virtually all empirical work is restricted to linear maps. A natural question arises: what happens if this restriction is lifted? Do the resulting mappings remain meaningful?
3. **The reliability of interpretability conclusions is questionable**: If sufficiently complex alignment maps can make any model appear to execute any algorithm, the reliability of existing causal-abstraction-based findings (e.g., "GPT-2 executes an indirect object identification circuit") warrants re-examination—such conclusions may be artifacts of the mapping rather than genuine mechanisms of the model.
4. **No theoretical guidance for the complexity–accuracy trade-off**: Practitioners intuitively choose linear maps but lack theoretical justification for doing so. Once non-linearity is permitted, map complexity becomes a new degree of freedom, and principled bounds are absent. This question has received almost no discussion in the interpretability community.
5. **The implicit assumption about information encoding has not been made explicit**: Causal abstraction implicitly presupposes that a model encodes information in some specific manner. This paper reveals that such a presupposition is not merely implicit but is a necessary condition for the validity of the causal abstraction methodology. Without it, the entire framework collapses.
6. **Tension with Transformer circuit analyses**: A large body of recent work has identified structured findings—such as functional roles of attention heads—via linear probes combined with causal interventions. The robustness of these findings depends on the validity of the linear map assumption. If non-linear maps yield equally good or better results, the interpretation of such "discoveries" becomes unclear.

## Method

### Theoretical Result: Triviality of Unconstrained Causal Abstraction

The central theorem of this paper shows that, under mild assumptions (e.g., continuity of the representation space and the alignment map being an arbitrary measurable function), **any neural network can be causally abstracted as any algorithm**. More precisely:

For a neural network $f: \mathcal{X} \to \mathcal{Y}$ mapping inputs to outputs, and any high-level algorithm $A$ with intermediate variable $Z$, there always exists a (potentially non-linear) alignment map $\tau: \mathcal{H} \to Z$ from the network's hidden representation space to the algorithm's intermediate variable space such that the network perfectly aligns with the algorithm under interchange intervention—achieving 100% interchange-intervention accuracy (IIA).

**Proof intuition**: This result exploits the "surplus capacity" of high-dimensional representation spaces. Given a sufficiently expressive family of maps, one can always find a mapping that sends an arbitrary partition of the representation space to an arbitrary intermediate state of the target algorithm. This is analogous to the observation that "any function can be fitted given sufficient parameters"—the alignment map itself absorbs all explanatory power.

### Empirical Validation: Random Models Also Achieve Perfect Alignment

To verify the practical feasibility of the theoretical prediction, the authors conduct experiments on the **Indirect Object Identification (IOI) task**, a classical benchmark in mechanistic interpretability. Given the sentence "When Mary and John went to the store, John gave a drink to", the model should complete it with "Mary".

**Key experimental design**:
- A **randomly initialized GPT-2 model** (untrained, incapable of performing the IOI task) is used.
- Non-linear alignment maps (MLPs) are trained to map the random model's hidden states to the IOI algorithm's intermediate variables.
- IIA is computed under interchange intervention.

**Core finding**: The non-linear alignment maps achieve **100% IIA** on the randomly initialized model. This means that a network with no language understanding and no ability to perform IOI "perfectly implements" the IOI algorithm within the causal abstraction framework—an evidently absurd outcome that demonstrates, by contradiction, the vacuousness of unconstrained causal abstraction.

### The Non-Linear Representation Dilemma

The resulting dilemma is:
- **Maintain linear constraints**: Results are meaningful but may miss non-linearly encoded information in the model.
- **Relax linearity**: The map gains expressive power but produces uninformative results.
- **Middle ground**: There is no principled way to determine "how much non-linearity is appropriate."

This is not an engineering question ("MLP vs. linear layer") but a methodological one: the validity of causal abstraction as an interpretability tool fundamentally depends on prior assumptions about how a model encodes information—assumptions that are typically unverified.

## Key Experimental Results

| Experimental Setting | Alignment Map Type | Model State | IIA (%) | Conclusion |
|---|---|---|---|---|
| GPT-2 + IOI task | Linear map | Trained | ~85–95% | Standard causal abstraction result |
| GPT-2 + IOI task | Non-linear MLP | Randomly initialized (untrained) | **100%** | Causal abstraction becomes trivial under non-linear maps |
| GPT-2 + IOI task | Linear map | Randomly initialized | ~50% (chance level) | Linear constraints do filter out spurious alignment |

| Analysis Dimension | Linear Map | Non-linear Map | Implication |
|---|---|---|---|
| IIA on trained model | High | 100% | Non-linearity may introduce spurious alignment |
| IIA on random model | Chance level | 100% | Explanatory power of non-linear maps resides in the map itself, not the model |
| Map complexity | Low ($d \times d$ matrix) | High (multi-layer MLP) | Higher complexity → easier to achieve spurious perfection |
| Interpretability value | Meaningful | Meaningless | Validates the critical role of the linearity assumption |

## Highlights & Insights

- **Clean and disruptive theoretical result**: A single theorem reveals the triviality of unconstrained causal abstraction, with a concise and powerful argument. Work of this kind—identifying methodological flaws—is critically important in the rapidly growing interpretability field.
- **Clever empirical design**: Achieving 100% IIA on a randomly initialized model constitutes the most compelling counterexample possible. If a model that can do nothing "perfectly executes" some algorithm, then the conclusions of causal abstraction are vacuous. This experimental design deserves to be treated as a textbook case.
- **The question matters more than the answer**: The "non-linear representation dilemma" raised in this paper is a deep methodological issue. It demonstrates that interpretability research cannot circumvent assumptions about how models encode information—a finding with significant implications for future research directions.
- **A well-deserved Spotlight**: At a time when interpretability methods proliferate but theoretical reflection is scarce, this paper issues a timely warning: not all plausible methodologies are sound, and the validity of their underlying assumptions demands scrutiny.

## Limitations & Future Work

- **No constructive alternative is proposed**: The paper excels at diagnosing the problem but does not offer a concrete prescription for "how to do it correctly." "Studying the relationship between information encoding assumptions and causal abstraction" is a broad direction that lacks an actionable roadmap.
- **Experimental scope is narrow**: Validation is limited to the IOI task using GPT-2 only. Generalizability to larger models (LLaMA/GPT-4) and other tasks (fact retrieval, arithmetic reasoning) is not examined.
- **The space of non-linear maps is not thoroughly explored**: The paper uses MLPs as non-linear alignment maps, but does not investigate how varying degrees of non-linearity (e.g., low-rank non-linear maps, kernel maps, shallow networks) affect results, nor whether there exists a "just sufficient" level of non-linearity.
- **Connection to information-theoretic perspectives could be deepened**: Mutual information could be used to quantify the extent to which alignment maps *create* versus *extract* information, providing information-theoretic bounds for the complexity–accuracy trade-off.
- **Practical impact may be overstated**: Most rigorous interpretability work already employs linear maps; the risk highlighted by the authors—drawing spurious conclusions through non-linear maps—is uncommon in practice. Nevertheless, clarifying the theoretical foundations retains significant value.

## Related Work & Insights

Compared to **Geiger et al. (2021, 2024) causal abstraction framework**: This paper constitutes a direct critical examination of the causal abstraction methodology developed by Geiger et al. While Geiger et al. developed the interchange intervention methodology with an implicit linear map assumption, this paper reveals the collapse that follows from relaxing that assumption.

Compared to the **Linear Representation Hypothesis** (Park et al. 2024; Jiang et al. 2024): The Linear Representation Hypothesis provides empirical motivation for using linear maps in causal abstraction. This paper argues from the opposite direction that this assumption is not merely "convenient" but *necessary*—without it, causal abstraction becomes vacuous.

Compared to **probing methods** (Belinkov 2022; Hewitt & Liang 2019): The "control task" approach for linear probes attempts to distinguish between information learned by the probe and information encoded in the representation. This paper reveals an analogous problem at a higher level (causal abstraction vs. probing)—the tool itself may generate spurious findings.

Compared to **Distributed Alignment Search (DAS)**: DAS is a specific instantiation of causal abstraction that uses orthogonal linear transformations as alignment maps. The theoretical results of this paper imply that DAS's linear constraint is the key guarantor of its validity, not a limitation.

**Further connections**:
- **"More expressive tools require stronger regularization"**: Just as overparameterized models in machine learning require regularization, this paper extends the same principle to interpretability tools—an overly flexible alignment map is equivalent to overfitting, and constraints such as linearity serve as regularization.
- **Practical guidance for LLM interpretability**: When analyzing LLMs using SAEs, linear probes, or causal intervention, one should always report the complexity of the alignment map and verify whether a simple (e.g., linear) map achieves comparable performance. If a complex map is required to produce "clean" results, the conclusions should be treated with skepticism.
- **Potential connection to model compression**: Causal abstraction is in essence a form of "representation compression"—mapping high-dimensional hidden states to low-dimensional algorithmic variables. The findings here suggest that the expressive power of the compression map should be matched to the complexity of the target task, consistent with design principles in knowledge distillation.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The dilemma identified is a genuinely novel and deep methodological problem; the theoretical result is disruptive.
- Experimental Thoroughness: ⭐⭐⭐ The experimental design is elegant but limited in scope (IOI + GPT-2 only); lacks breadth.
- Writing Quality: ⭐⭐⭐⭐⭐ The argument develops with clear logical progression and the problem motivation is articulated with precision; Spotlight-level writing.
- Value: ⭐⭐⭐⭐ The critical reflection on the foundations of interpretability methodology is highly valuable, though the absence of a constructive alternative warrants a slight deduction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] nnterp: A Standardized Interface for Mechanistic Interpretability of Transformers](nnterp_a_standardized_interface_for_mechanistic_interpretability_of_transformers.md)
- [\[NeurIPS 2025\] Minimizing False-Positive Attributions in Explanations of Non-Linear Models](minimizing_false-positive_attributions_in_explanations_of_non-linear_models.md)
- [\[ICLR 2026\] Formal Mechanistic Interpretability: Automated Circuit Discovery with Provable Guarantees](../../ICLR2026/interpretability/formal_mechanistic_interpretability_automated_circuit_discovery_with_provable_gu.md)
- [\[NeurIPS 2025\] Causal Head Gating: A Framework for Interpreting Roles of Attention Heads in Transformers](causal_head_gating_a_framework_for_interpreting_roles_of_attention_heads_in_tran.md)
- [\[NeurIPS 2025\] Emergence of Linear Truth Encodings in Language Models](emergence_of_linear_truth_encodings_in_language_models.md)

</div>

<!-- RELATED:END -->

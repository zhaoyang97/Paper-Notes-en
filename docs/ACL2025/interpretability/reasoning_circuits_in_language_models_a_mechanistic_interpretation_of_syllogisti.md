---
title: >-
  [Paper Note] Reasoning Circuits in Language Models: A Mechanistic Interpretation of Syllogistic Inference
description: >-
  [ACL 2025][Interpretability][Mechanistic Interpretability] The complete circuit for syllogistic reasoning in language models is discovered using mechanistic interpretability techniques (Activation Patching + Logit Lens + Circuit Ablation). The circuit operates via a three-stage mechanism: long induction bias $\rightarrow$ middle-term suppression (h11.10) $\rightarrow$ transitive term movement. This circuit is both sufficient and necessary on symbolic inputs…
tags:
  - "ACL 2025"
  - "Interpretability"
  - "Mechanistic Interpretability"
  - "Syllogistic Reasoning"
  - "Attention Circuits"
  - "Activation Patching"
  - "Middle-Term Suppression"
date: 2026-05-08
content_hash: 7be52f0d3e80d940
---

# Reasoning Circuits in Language Models: A Mechanistic Interpretation of Syllogistic Inference

**Conference**: ACL 2025  
**arXiv**: [2408.08590](https://arxiv.org/abs/2408.08590)  
**Code**: [https://github.com/neuro-symbolic-ai/Mechanistic-Interpretation-Syllogism](https://github.com/neuro-symbolic-ai/Mechanistic-Interpretation-Syllogism)  
**Area**: LLM Interpretability / Logical Reasoning  
**Keywords**: Mechanistic Interpretability, Syllogistic Reasoning, Attention Circuits, Activation Patching, Middle-Term Suppression  

## TL;DR
The complete circuit for syllogistic reasoning in language models is discovered using mechanistic interpretability techniques (Activation Patching + Logit Lens + Circuit Ablation). The circuit operates via a three-stage mechanism: long induction bias $\rightarrow$ middle-term suppression (h11.10) $\rightarrow$ transitive term movement. This circuit is both sufficient and necessary on symbolic inputs, generalizes to natural language inputs, and exhibits compatible patterns across four architectures: GPT-2, Pythia, LLaMA, and Qwen.

## Background & Motivation

**Background**: LLMs have demonstrated reasoning capabilities. Mechanistic interpretability has successfully analyzed simple circuits such as addition (Stolfo et al. 2023), acronym prediction (García-Carrasco et al. 2024), and magnitude comparison (Hanna et al. 2024), but mechanistic analyses of logical reasoning circuits remain practically nonexistent.

**Limitations of Prior Work**: 
   - It remains unclear whether reasoning in LLMs represents genuine application of logical rules or superficial pattern memorization from training data—an ongoing debate (Talmor et al. 2020, Wu et al. 2024).
   - LLMs perform inconsistently on formal logic—showing high accuracy on syllogisms aligned with common sense but low accuracy on those contradicting it (belief bias).
   - Previous interpretability work focused on localizing factual associations (Meng et al. 2022) rather than entering the realm of logical reasoning.

**Key Challenge**: If LLMs have truly learned reasoning rules, they should reason correctly even on symbolic inputs (content-independence). However, the presence of belief bias indicates that world knowledge is "contaminating" the reasoning process. Circuit-level analysis can precisely localize this contamination mechanism.

**Goal**: (1) Does a content-independent syllogistic reasoning circuit exist inside LLMs? (2) How does belief bias manifest at the circuit level? (3) Does the discovered circuit generalize across syllogism formats, model sizes, and architectures?

**Key Insight**: Choose the AAA-1 (Barbara) syllogism—the most common form of everyday language reasoning, which possesses unconditional validity (the validity of the conclusion is independent of the truth value of the premises). This allows for a rigorous separation of reasoning mechanisms from knowledge representations.

**Core Idea**: Use symbolic syllogisms and activation patching to localize the reasoning circuit, discovering an internal error-correction mechanism based on "middle-term suppression".

## Method

### Overall Architecture
A three-stage methodology: (1) Define a syllogism completion task ("All A are B. All B are C. Therefore, all A are ?", where the correct answer is C and the distractor is B), measuring reasoning success with the logit difference $\delta(p,m) = \text{logit}(p) - \text{logit}(m)$; (2) Discover the circuit on pure symbolic inputs using two causal intervention methods; (3) Evaluate the transferability and belief bias of the circuit on natural language inputs.

### Key Designs

1. **Syllogism Completion Task Design**:

    - Function: Formulate syllogisms as language model completion tasks by removing the final word of the conclusion (predicate term $p$) and comparing the model's probabilities for $p$ and the middle term $m$.
    - Mechanism: Reasoning is successful if $P(p|[\mathcal{P}_1;\mathcal{P}_2;\mathcal{C}\setminus\{p\}]) > P(m|[\mathcal{P}_1;\mathcal{P}_2;\mathcal{C}\setminus\{p\}])$.
    - Design Motivation: Using logit differences rather than accuracy allows for a continuous measurement of the model's confidence in the correct answer. Three types of datasets are constructed: symbolic (random uppercase letters), belief-consistent (true premises, e.g., "All men are mortal"), and belief-inconsistent (false premises, e.g., "All pilots are blond").

2. **Two Causal Intervention Methods**:

    - **Middle-term corruption**: Replace the middle term $m_2$ in the second premise with a new symbol $m_2'$—breaking transitivity to localize components involved in transitive reasoning. Core assumption: if an attention head is involved in transitive reasoning, patching its activations after corrupting the middle term should yield a high recovery score.
    - **Full-term corruption**: Replace all terms $(s,m_1,m_2,p) \to (s',m_1',m_2',p')$ while keeping the answer constant—tracking how specific token information flows to the final position to localize mover heads.

3. **Three-Stage Reasoning Circuit Discovery (GPT-2 Medium)**:

    - **(1) Long Induction Bias**: Negative mover heads in early layers (h9.9, h11.1, h12.1) exhibit induction-head-like behavior, attending strongly to the $[m_1]$ position—directly copying information from the first premise to the final position, which biases the model to output the middle term B instead of the correct answer C.
    - **(2) Middle-Term Suppression**: A key finding—h11.10 reads middle-term information from the $[m_2]$ position and writes a suppression signal at the $[p]$ position. Logit Lens analysis of its OV matrix reveals a clear negative diagonal pattern—when attending to a token, it strongly suppresses that token's logit. The average attention weight is $0.15 \pm 0.07$. Induction heads (h5.8, h6.1, h6.15, h7.2) upstream aggregate the repeated information of $m_1 \equiv m_2$ to the $[p]$ position.
    - **(3) Mover Heads**: Nine mover heads propagate the correct information to the final position. Positive movers (h14.14, h15.14, h18.12) attend to the $[p]$ position to copy the conclusion term; positive suppression heads (h19.1) attend to $[m_2]$ and $[s]$ to further suppress distractors; negative movers (h9.9, h11.1, h12.1, h17.2, h23.10) attend to $[m_1]$—acting as the source of the long induction bias.

### Circuit Validation
- **Sufficiency Test**: Retaining only the circuit components and ablating all other heads restores the original model performance.
- **Necessity Test**: Ablating the circuit components while keeping all other heads causes a significant drop in performance.
- **Robustness Test**: Changing letters to numbers or modifying "All ... are" to "Each ... is" shows that the circuit remains sufficient and necessary.

## Key Experimental Results

### Circuit Accuracy (GPT-2 Medium, Symbolic Dataset)

| Test | Results |
|------|------|
| Sufficiency | Retaining only the circuit restores the baseline logit difference |
| Necessity | Ablating the circuit significantly decreases the logit difference |
| Numerical Robustness | Changing letters to numbers, the circuit remains effective |
| Quantifier Robustness | "All...are" $\rightarrow$ "Each...is", the circuit remains effective |

### Circuit-Level Evidence of Belief Bias

| Data Type | Circuit Necessity | Circuit Sufficiency |
|---------|----------|----------|
| Symbolic | ✓ | ✓ |
| Belief-Consistent (True Premise) | ✓ | ✓ (Performance restored) |
| Belief-Inconsistent (False Premise) | ✓ | ✗ (Cannot fully restore) |

- Insufficiency of the circuit during belief-inconsistent scenarios implies that extra attention heads (encoding world knowledge / belief bias) are involved in reasoning.
- Subject-term corruption experiment: In non-symbolic settings, replacing the subject term $s$ leads to a $299.96\%$ drop in the logit difference, compared to only a $0.35\%$ decrease in symbolic settings—directly proving that world knowledge contaminates the logical circuit through independent pathways.

### Generalization Across Syllogistic Formats (15 Unconditionally Valid Formats)

| Conditions | Formats Satisfied |
|------|----------|
| C1 (Necessary) + C2 (Sufficient) + C3 (Positive logit diff) | AAA-1, AII-3, IAI-3, IAI-4 (Accuracy $\ge 60\%$) |
| Only C1 (Necessary) | Most formats |
| Completely Unsatisfied | AOO-2, AEE-4, EIO-2, AEE-2, EAE-2 (Accuracy $< 25\%$) |

### Cross-Architecture Generalization

| Model Group | Model | Pattern |
|--------|------|------|
| Unstable | Pythia-70M/160M | Unstable activation patterns |
| Compatible | Pythia-410M/1B, LLaMA-3.2-1B | Similar suppression mechanisms and information flows |
| Variant | Qwen-2.5-0.5B/1.5B | Suppression occurs at the final token position instead of $[p]$ |

### Key Findings
- **Reasoning as Internal Error-Correction**: The model initially defaults to outputting the middle term (incorrect answer) via "long induction," which is then actively corrected by suppression heads—markedly different from how humans reason using abstract logical rules.
- **Precise Mechanism of Belief Bias**: World knowledge "contaminates" logical reasoning through extra-circuit attention heads. When premises contradict common sense, the influence of these extra heads renders the symbolic circuit insufficient.
- **Content-Independent Reasoning Exists**: The circuit discovered on symbolic inputs transfers to natural language (necessity holds), demonstrating that LLMs indeed learn a formalized transitive reasoning mechanism.
- **Circuit Complexity Increases with Model Size**: Larger models show decreased accuracy on symbolic datasets (e.g., GPT-2 XL) while non-symbolic accuracy increases, implying that interference from world knowledge scales with model size.

## Highlights & Insights
- **"Suppression Heads" as an Error-Correction Mechanism**: The negative diagonal pattern of the OV matrix in h11.10 is highly distinct—it does not "help identify the correct answer" but instead "actively eliminates the incorrect answer." Such error-correcting logic is rarely observed in previous circuit analyses.
- **Mechanistic Interpretation of Belief Bias**: This work provides the first precise circuit-level description of how world knowledge interferes with logical reasoning—showing that it is not the "reasoning circuit itself that is biased," but rather that "extra knowledge-encoding heads contaminate the reasoning pipeline."
- **Fine-Tuning Does Not Alter the Circuit**: GPT-2 Medium fine-tuned on debate texts produces nearly identical activation patterns, indicating that the reasoning circuit originates from pre-training rather than fine-tuning—holding profound theoretical significance.
- **Transferable Analytical Methodology**: The two-step intervention design of middle-term corruption + full-term corruption generalizes well to the circuit analysis of other formal reasoning tasks.

## Limitations & Future Work
- **Analysis Limited to a Single Form of Reasoning (Syllogisms)**: More complex forms of reasoning (such as multi-step chain-of-thought, counterfactual reasoning) may require entirely different circuits.
- **Small Scale of GPT-2 Medium**: Circuits in larger LLMs may be more complex, and suppression mechanisms could be distributed across more heads.
- **Limitations of Causal Interventions**: Analyzing only two dimensions—transitivity and token information flow—might overlook other vital reasoning dynamics.
- **Templatized Task Design**: Real-world syllogistic reasoning rarely occurs in such standardized templates.

## Related Work & Insights
- **vs. Stolfo et al. (2023)**: They analyzed the addition circuit in GPT-2—exhibiting a similar information-flow pattern (attention aggregation at specific positions), but the logical reasoning circuit contains an additional "suppression" stage.
- **vs. Meng et al. (2022)**: They utilized activation patching to localize where factual knowledge is stored; this work uses similar techniques but focuses on the reasoning process itself rather than knowledge storage.
- **vs. Wiegreffe et al. (2025)**: They analyzed attention patterns in multiple-choice questions; this work delves deeper into sufficiency and necessity analyses at the circuit level.
- **Theoretical Insights for LLM Reasoning Capabilities**: LLMs indeed acquire content-independent reasoning mechanisms. However, this mechanism is not human-like abstract logical rules but rather a statistical pattern-based error-correction process, which is easily disrupted by world knowledge acquired during pre-training.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First complete description of a three-stage mechanism for logical reasoning circuits, with the discovery of "suppression heads" carrying significant theoretical value.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive sufficiency/necessity ablation + robustness tests + belief bias analysis + 15 formats + 4 architectures.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous methodology, clear description of findings, and highly informative figures.
- Value: ⭐⭐⭐⭐⭐ Offers critical theoretical contributions to understanding the nature of LLM reasoning (rules vs. patterns, entanglement of knowledge and reasoning).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Mechanistic Interpretability of Emotion Inference in Large Language Models](mechanistic_interpretability_of_emotion_inference_in_large_language_models.md)
- [\[ACL 2025\] An Empirical Study of Mechanistic Interpretability Approaches for Factual Recall](an_empirical_study_of_mechanistic_interpretability_approaches_for_factual_recall.md)
- [\[ICML 2026\] Certified Circuits: Stability Guarantees for Mechanistic Circuits](../../ICML2026/interpretability/certified_circuits_stability_guarantees_for_mechanistic_circuits.md)
- [\[ACL 2025\] Separating Tongue from Thought: Activation Patching Reveals Language-Agnostic Concept Representations in Transformers](separating_tongue_from_thought_activation_patching_reveals_language-agnostic_con.md)
- [\[ACL 2025\] Llama See, Llama Do: A Mechanistic Perspective on Contextual Entrainment and Distraction in LLMs](llama_see_llama_do_entrainment.md)

</div>

<!-- RELATED:END -->

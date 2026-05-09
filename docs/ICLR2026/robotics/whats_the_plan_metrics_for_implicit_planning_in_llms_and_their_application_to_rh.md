---
title: >-
  [Paper Note] What's the Plan? Metrics for Implicit Planning in LLMs and Their Application to Rhyme Generation and Question Answering
description: >-
  [ICLR 2026][Robotics][implicit planning] This paper proposes a mean activation difference steering method along with accompanying quantitative metrics, and systematically demonstrates across 23 open-source models (1B–32B) on rhyme generation and question answering: representations of target tokens (rhymes/answers) form at early sequence positions (forward planning) and causally influence intermediate token generation (backward planning). Implicit planning emerges as early as 1B-scale models, indicating it is a universal mechanism rather than a capability exclusive to large models.
tags:
  - ICLR 2026
  - Robotics
  - implicit planning
  - forward planning
  - backward planning
  - activation steering
  - rhyme generation
date: 2026-05-08
content_hash: 2ef9bbbd9c7e555c
---

# What's the Plan? Metrics for Implicit Planning in LLMs and Their Application to Rhyme Generation and Question Answering

**Conference**: ICLR 2026
**arXiv**: [2601.20164](https://arxiv.org/abs/2601.20164)
**Code**: Available (with supplementary material)
**Area**: Robotics
**Keywords**: implicit planning, forward planning, backward planning, activation steering, rhyme generation

## TL;DR
This paper proposes a mean activation difference steering method along with accompanying quantitative metrics, and systematically demonstrates across 23 open-source models (1B–32B) on rhyme generation and question answering: representations of target tokens (rhymes/answers) form at early sequence positions (forward planning) and causally influence intermediate token generation (backward planning). Implicit planning emerges as early as 1B-scale models, indicating it is a universal mechanism rather than a capability exclusive to large models.

## Background & Motivation

**Background**: LLMs are trained via next-token prediction yet produce coherent text. Lindsey et al. (2025) qualitatively demonstrated rhyme-planning behavior in Claude 3.5 Haiku using a cross-layer transcoder (CLT)—the model already encodes a representation of the future rhyme at the end of the first line, and this representation influences the generation of intermediate words in the second line.

**Limitations of Prior Work**: (1) The CLT approach is complex and expensive (training on a single model requires days on H100s) and does not scale to multi-model comparisons; (2) Lindsey's findings are limited to a few qualitative examples on a single closed-source model and are not reproducible; (3) quantitative metrics for implicit planning are lacking—no standardized method exists for assessing the "degree of planning."

**Key Challenge**: The importance of implicit planning (for understanding LLM capabilities and safety) conflicts with the complexity and non-scalability of existing research methods.

**Goal**: To quantitatively study implicit planning using a simple, scalable method, and to systematically validate findings across multiple models and tasks.

**Key Insight**: Rhyming poetry and question answering serve as ideal probes for implicit planning—the nature and position of target tokens can be predicted from general principles but are not determined solely by the immediately preceding context.

**Core Idea**: Mean activation difference steering, applied at the correct position, is sufficient to manipulate both forward and backward planning without training a CLT or SAE.

## Method

### Overall Architecture
The paper defines forward planning (early positions encoding future target attributes) and backward planning (using planning representations to generate intermediate tokens leading to the target), and intervenes in both via activation steering. For the rhyme task: manipulating the rhyme-family representation at the end of the first line → observing whether the rhyme word and intermediate words in the second line change. For QA: manipulating the answer representation at the end of the question → observing whether passage selection and the final answer change.

### Key Designs

1. **Mean Activation Difference Steering**
   - **Function**: Extracts and manipulates planning representations in hidden activations.
   - **Mechanism**: Computes the mean difference of hidden activations at a specific position (final word / newline token / question mark) between two prompt categories as the steering vector $\mathbf{s}_{C_1 \to C_2}^{(l,i)} = m \cdot (\overline{\mathbf{x}_i^{(l)}}_{C_1} - \overline{\mathbf{x}_i^{(l)}}_{C_2})$, with $m=1.5$. During generation, the steering vector is added to the residual stream at a single token position only.
   - **Design Motivation**: Orders of magnitude simpler than CLT and scalable to arbitrary models—requiring only forward passes to extract activations and compute mean differences, with no training needed.

2. **Forward Planning Verification — Fraction of Correct Rhyme Family (Steered)**
   - **Function**: Quantifies whether steering can switch the rhyme word from one rhyme family to another.
   - **Mechanism**: Over 1,000 couplets (50 samples × 20 test prompts), the proportion of steered generations belonging to the target rhyme family is computed. Effective steering implies that the early position encodes a manipulable forward planning representation.
   - **Design Motivation**: If no planning representation exists at the intervention position, the intervention will not affect the rhyme word; steering success therefore serves as evidence of forward planning.

3. **Backward Planning Verification — Regeneration Metric + Probability Metrics**
   - **Function**: Verifies that steering not only changes the final rhyme word but also alters the intermediate tokens leading to it.
   - **Mechanism**: *Regeneration*—after removing the rhyming context, the last word of the second line is regenerated; if intermediate words still "lead toward" the target rhyme, backward planning was operative during intermediate token generation. *Probability metrics*—comparison of intermediate token probability distributions between steered and unsteered conditions (fraction of positions with KL divergence > 1, and position of first top-1 token divergence).
   - **Design Motivation**: A regeneration success rate close to the baseline indicates that steering genuinely alters the entire generation path rather than merely substituting the final word.

### Datasets
Rhyme: 10 rhyme families × 105 lines (generated by Claude 3.5 Sonnet), 20 rhyme-family pairs. QA: 20 noun pairs (vowel-initial vs. consonant-initial → different articles *a*/*an*), each noun with 13 training + 5 test + 7 neutral questions.

## Key Experimental Results

### Main Results — Rhyme Planning across 23 Models

| Model | Baseline Rhyme Rate | Steered Rhyme Rate | Baseline Regen Rate | Steered Regen Rate |
|-------|:---:|:---:|:---:|:---:|
| Gemma2 9B IT | ~80% | ~75% | ~55% | ~52% |
| Gemma3 27B IT | ~85% | ~82% | ~60% | ~58% |
| Llama 3.1 8B IT | ~60% | ~55% | ~40% | ~38% |
| Gemma3 1B Base | ~25% | ~15% | ~20% | ~12% |

(IT = instruction-tuned; Base = base model)

### Ablation Study — Steering Position Analysis

| Steering Position | Gemma2 9B | Gemma3 27B | Other Models |
|-------------------|:---:|:---:|:---:|
| Final word (lower layers) | ✓ Effective | ✓ Effective | ✓ Effective across all |
| Newline token (middle layers) | ✓ Effective | ✓ Effective | ✗ Mostly ineffective |

### Key Findings
- **Implicit planning emerges at 1B scale**: Even the smallest model (Gemma3 1B) exhibits detectable forward and backward planning, indicating greater universality than previously assumed.
- **Instruction tuning enhances planning**: IT models consistently outperform base models on all metrics, suggesting post-training may strengthen planning capabilities.
- **Cross-task generality**: Steering in QA similarly alters article selection (*a* vs. *an*), indicating that planning is not a rhyme-specific mechanism.
- **Circuit localization (Gemma2 9B)**: Two attention heads (L30H3, L31H15) read planning information from the end of the first line; activation patching recovers 59%–93% of the steering effect; subsequent MLP layers transform this information into predictions.
- **Rhyme and QA rely on different circuits**: The rhyme circuit is concentrated in L30H3/L31H15, while L39H13 is more prominent in QA, suggesting that the planning mechanism is general but the underlying circuits are task-specific.
- **All rhyme metrics are highly correlated**: Rhyming ability and planning ability co-develop.

## Highlights & Insights
- **Methodological contribution outweighs the findings themselves**: Mean activation difference steering is remarkably simple yet enables systematic study across 23 models—reducing the cost of planning research from "H100-level computation requiring CLT training" to "a few hours on any GPU."
- **Planning at 1B scale → an inevitable product of autoregressive training**: If the objective is merely next-token prediction, why plan ahead? Because the optimal choice of the current token depends on future intent—even the smallest LLMs learn this lookahead behavior.
- **Rhyming poetry is an ideal planning probe**: It requires advance commitment to a rhyme target, and intermediate words must be semantically compatible—this is precisely the definition of forward plus backward planning. Article selection in QA provides analogous but simpler evidence.
- **Direct implications for AI safety**: Implicit planning implies that models harbor internal states representing "intentions not yet expressed"—understanding and monitoring these states is critical for alignment.

## Limitations & Future Work
- Steering success rates are imperfect—steered rhyme rates are substantially lower than baseline in weaker models, indicating noisy extraction of planning representations.
- Only two task settings are studied; more complex planning scenarios (e.g., code generation, long-horizon reasoning) require further validation.
- Mean activation difference is a coarse method that does not distinguish between planning for a specific word versus planning for a rhyme family at different levels of abstraction.
- Position-of-newline steering is effective only for a subset of models (Gemma2 9B, Gemma3 27B); how architectural differences across models affect planning circuit topology remains unclear.
- Circuit analysis is conducted in depth for only one model (Gemma2 9B); whether other models similarly rely on a small number of dominant attention heads is unknown.
- The interaction between explicit planning (e.g., CoT) and implicit planning is not addressed—it remains open whether the two are substitutes or complements.

## Related Work & Insights
- **vs. Lindsey et al. (2025) CLT analysis of Claude Haiku**: Their work qualitatively demonstrates planning phenomena but is expensive and non-reproducible; this paper quantitatively replicates and extends the findings across 23 open-source models using a simple method—methodological accessibility is the central contribution.
- **vs. Turpin et al. (2023) unfaithful CoT**: They find that CoT rationalizes incorrect answers when biased answers are introduced, providing indirect evidence of implicit forward planning; this paper directly and quantitatively confirms the phenomenon.
- **vs. Wu et al. (2024) and Men et al. (2024) lookahead studies**: Those works identify lookahead representations in specific models; this paper systematically validates the finding across 23 models—an upgrade from case studies to population-level research.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Quantitative implicit planning evaluation metrics + systematic study across 23 models; simple method yet rich insights.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 23 models (4 families × multiple scales × base + IT) × 2 tasks + circuit analysis + attention head ablation.
- **Writing Quality**: ⭐⭐⭐⭐ Forward/backward planning definitions are clear; experiments are intuitive and reproducible.
- **Value**: ⭐⭐⭐⭐⭐ Foundational contribution to understanding LLM internal mechanisms; methodology is broadly reusable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] When Agents Persuade: Propaganda Generation and Mitigation in LLMs](when_agents_persuade_propaganda_generation_and_mitigation_in_llms.md)
- [\[ICLR 2026\] String Seed of Thought: Prompting LLMs for Distribution-Faithful and Diverse Generation](string_seed_of_thought_prompting_llms_for_distribution-faithful_and_diverse_gene.md)
- [\[ICLR 2026\] One Demo Is All It Takes: Planning Domain Derivation with LLMs from A Single Demonstration](one_demo_is_all_it_takes_planning_domain_derivation_with_llms_from_a_single_demo.md)
- [\[ICLR 2026\] Enhancing Instruction Following of LLMs via Activation Steering with Dynamic Rejection](enhancing_instruction_following_of_llms_via_activation_steering_with_dynamic_rej.md)
- [\[AAAI 2026\] Do LLMs Really Struggle at NL-FOL Translation? Revealing Their Strengths via a Novel Benchmarking Strategy](../../AAAI2026/robotics/do_llms_really_struggle_at_nl-fol_translation_revealing_their_strengths_via_a_no.md)

</div>

<!-- RELATED:END -->

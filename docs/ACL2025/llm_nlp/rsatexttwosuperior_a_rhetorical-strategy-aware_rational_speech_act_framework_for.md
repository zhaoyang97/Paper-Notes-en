---
title: >-
  [Paper Note] (RSA)²: A Rhetorical-Strategy-Aware Rational Speech Act Framework for Figurative Language Understanding
description: >-
  [ACL 2025 (Main)][LLM (Other)][Figurative Language Understanding] This paper proposes the (RSA)² framework, which explicitly models the speaker's rhetorical strategies (e.g., irony, hyperbole) within the probabilistic pragmatics RSA framework. This enables LLMs to correctly understand non-literal meanings without modeling the speaker's motivation, achieving SOTA performance on the irony comprehension dataset PragMega+.
tags:
  - "ACL 2025 (Main)"
  - "LLM (Other)"
  - "Figurative Language Understanding"
  - "Pragmatics"
  - "Rational Speech Act Framework"
  - "Irony Interpretation"
  - "Rhetorical Strategy"
date: 2026-05-08
content_hash: 6b3ba0b87d0d92d1
---

# (RSA)²: A Rhetorical-Strategy-Aware Rational Speech Act Framework for Figurative Language Understanding

**Conference**: ACL 2025 (Main)  
**arXiv**: [2506.09301](https://arxiv.org/abs/2506.09301)  
**Code**: None  
**Area**: Other  
**Keywords**: Figurative Language Understanding, Pragmatics, Rational Speech Act Framework, Irony Interpretation, Rhetorical Strategy

## TL;DR

This paper proposes the (RSA)² framework, which explicitly models the speaker's rhetorical strategies (e.g., irony, hyperbole) within the probabilistic pragmatics RSA framework. This enables LLMs to correctly understand non-literal meanings without modeling the speaker's motivation, achieving SOTA performance on the irony comprehension dataset PragMega+.

## Background & Motivation

**Background**: Figurative language is ubiquitous in human communication—rhetorical devices such as irony, hyperbole, and understatement make the literal meaning of an utterance inconsistent with its intended meaning. The Rational Speech Act (RSA) framework is the most widely used theoretical model in probabilistic pragmatics, explaining the pragmatic meaning of utterances by recursively modeling intentional reasoning between a speaker and a listener.

**Limitations of Prior Work**: The standard RSA framework has a fundamental limitation: it can only produce interpretations consistent with the literal meaning and cannot assign non-zero probability to non-literal utterances. Prior solutions (such as affect-aware RSA) require explicitly modeling the speaker's "motivation" for using figurative language (e.g., expressing emotions of amusement or annoyance). However, this approach requires designing a separate emotional space for each scenario, resulting in poor generalizability. On the other hand, although LLMs possess strong language capabilities, they perform poorly on figurative language understanding, exhibiting severe literal bias.

**Key Challenge**: To enable models to correctly understand figurative language, the possibility of non-literal interpretations must be considered. However, standard RSA mathematically excludes this possibility. Although existing affect-aware RSA can handle this issue, it requires pre-defining the speaker's emotional motivation. In reality, the reasons why a person uses irony can be highly diverse (humor, sarcasm, habitual expression, etc.), making it difficult to unify under a single model.

**Goal**: To design an extension of the RSA framework that does not require modeling the speaker's motivation but naturally yields non-literal interpretations.

**Key Insight**: The use of non-literal language follows systematic patterns, which can be categorized as "rhetorical strategies." For example, irony is characterized by the intended meaning being the opposite of the literal meaning, while hyperbole is characterized by the intended meaning being weaker than the literal meaning. These strategies are regularities of *how* the speaker expresses non-literally, which is orthogonal to *why* they express non-literally (motivation).

**Core Idea**: To introduce rhetorical strategies as explicit latent variables into the RSA framework and marginalize them out to obtain non-literal interpretations, thereby bypassing the need to model the speaker's motivation.

## Method

### Overall Architecture

The input of (RSA)² is a context and an utterance, and the output is a probability distribution over the intended meanings of the utterance. At its core, the framework introduces the rhetorical strategy variable $r \in \mathcal{R}$ into each layer based on the three-layer recursive reasoning of standard RSA (literal listener → pragmatic speaker → pragmatic listener). During inference, the rhetorical strategy is marginalized out as a latent variable, enabling the final distribution of intended meanings to simultaneously consider both literal and non-literal interpretations. In LLM experiments, the framework is further combined with pre-trained language models, leveraging LLMs to estimate the required conditional probability distributions.

### Key Designs

1. **Rhetorical Function Generalization**:

    - Function: Generalizes the binary semantic comprehension function $\mathbb{1}_{m \in [\![u]\!]}$ in standard RSA into a continuous rhetorical function $f_r: \mathcal{C} \times \mathcal{M} \times \mathcal{U} \to [0,1]$.
    - Mechanism: For each rhetorical strategy $r$, a rhetorical function $f_r$ is defined to describe the compatibility between a specific meaning and the utterance under that strategy. For instance, for the irony strategy, "The weather is great" in a blizzard context has a compatibility of 1 with "The weather is terrible" and 0 with "The weather is great". This function replaces the semantic indicator function in the original literal listener equation: $P_{L_0}(m|c,u,r) \propto f_r(c,m,u) \cdot P(m|c)$.
    - Design Motivation: The semantic function in standard RSA is binary (compatible/incompatible), which mathematically guarantees that the probability of non-literal meaning is zero. Generalizing it to a continuous function allows non-literal meanings to obtain non-zero probabilities through suitable rhetorical functions.

2. **Rhetorical Strategy Marginalization**:

    - Function: Integrates listener distributions conditioned on different rhetorical strategies into a unified intended meaning distribution by marginalizing over the posterior probability of rhetorical strategies.
    - Mechanism: The final pragmatic listener distribution is $P_{L_1}(m|c,u) = \sum_{r'} P_{L_1}(m|c,u,r') \cdot P(r'|c,u)$. This means the model simultaneously evaluates "if the speaker is speaking literally/ironically/hyperbolically/..., what is the intended meaning," and then computes a weighted mixture based on the posterior probability of each rhetorical strategy. In LLM experiments, the posterior of rhetorical strategies is estimated from the LLM via prompting.
    - Design Motivation: In real-world scenarios, the listener is uncertain about which rhetorical strategy the speaker is using, thus requiring a probabilistic weighting across all possible strategies. This is a standard Bayesian inference approach, which is theoretically clean and elegant.

3. **LLM Integration with Prompt-based Probability Estimation**:

    - Function: Integrates (RSA)² with LLMs, utilizing LLMs to estimate the various conditional probabilities required by the framework.
    - Mechanism: Two LLMs are employed: an instruction-tuned model $N$ (e.g., Mistral-7B-Instruct) to estimate distributions such as $P_N(m|c,u)$, $P_N(r|c,u)$, and $P_N(m|c,u,r)$ using a multiple-choice prompt format; and a base model $G$ (e.g., Llama-8B) to generate a set of alternative utterances and their prior probabilities. To avoid position bias, the options are randomly ordered 10 times and averaged.
    - Design Motivation: Synthesizing the probabilistic pragmatic reasoning of RSA with the language understanding capabilities of LLMs—where LLMs provide semantic priors and RSA provides a structured pragmatic reasoning framework, making them complementary.

### Loss & Training

The core of (RSA)² does not involve end-to-end training. In the ironic weather utterances experiment, a 2-layer neural network (16×16×5, sigmoid activation) is trained to learn the rhetorical function $f_r$ for 500 epochs using the Adam optimizer (lr=0.001, weight decay=0.001) with cross-entropy loss. In LLM experiments, all probabilities are estimated via prompt engineering, requiring no fine-tuning of the LLM.

## Key Experimental Results

### Main Results

**RSA Model Experiments (Non-literal Number Expressions + Ironic Weather Utterances)**:

| Model | Non-literal Numbers $L_0$ MAD↓ | Non-literal Numbers $L_1$ MAD↓ | Weather Utterances $L_0$ MAD↓ | Weather Utterances $L_1$ MAD↓ |
|------|-----|-----|-----|-----|
| Affect-Aware RSA | - | 0.0436 | 0.2377 | 0.1278 |
| (RSA)² | 0.0438 | 0.0467 | 0.1647 | **0.1229** |

**LLM Irony Understanding Experiments (PragMega+ Dataset)**:

| Model | Correct Meaning Prob.↑ | Incorrect Meaning Prob.↓ | Distractor Meaning Prob.↓ |
|------|-----------|-----------|-----------|
| LLM RSA $L_0$ | 0.73 | 0.24 | 0.02 |
| LLM RSA $L_1$ | 0.76 | 0.22 | 0.01 |
| LLM (RSA)² $L_0$ (with $I(r|c,u)$) | **0.85** | **0.13** | 0.01 |
| LLM (RSA)² $L_1$ (with $I(r|c,u)$) | 0.84 | 0.13 | 0.01 |

### Ablation Study

| Configuration | Correct Meaning Prob. | Relative Change | Explanation |
|------|-----------|---------|------|
| LLM RSA $L_1$ (Full) | 0.76 | - | Baseline |
| LLM RSA $L_1$ (w/o $P(m|c)$) | 0.44 | -42.7% | Meaning prior is critical |
| LLM RSA $L_1$ (w/o $P(u|c)$) | 0.78 | +1.8% | Utterance prior has small impact |
| LLM (RSA)² $L_0$ with $I$ (Full) | 0.85 | - | Best model |
| LLM (RSA)² $L_0$ with $I$ (w/o $P(m|c)$) | 0.51 | -39.4% | Prior remains important |
| LLM (RSA)² $L_0$ with $I$ (w/o $P(u|c)$) | 0.84 | +0.2% | Utterance prior has almost no impact |

### Key Findings

- (RSA)² performs best in irony scenarios (intended meaning probability >0.8), outperforming affect-aware RSA on the ironic weather utterances dataset.
- The quality of the rhetorical strategy posterior is critical: using the indicator function $I(r|c,u)$ performs better than using the continuous probability $P_N(r|c,u)$ estimated by the LLM.
- The meaning prior $P(m|c)$ is the largest contributor to performance, while the recursive reasoning process of RSA itself has a limited contribution (this is because the alternative utterance generation tends to produce literal paraphrases).
- LLMs exhibit asymmetry in determining rhetorical strategies: they are highly accurate in ironic scenarios ($P(r=\text{irony}|c,u)=0.88$), but perform poorly in literal scenarios ($P(r=\text{literal}|c,u)=0.55$).

## Highlights & Insights

- **Rhetorical Strategy vs. Speaker Motivation**: Decoupling "how to express non-literally" (rhetorical strategy) from "why express non-literally" (motivation) is the core innovation. This not only simplifies modeling but also aligns better with linguistic intuition—when listeners interpret figurative language, they indeed focus more on inferring which rhetorical device was used rather than analyzing the speaker's emotional state.
- **Theoretical Contribution of Mathematical Proof**: The paper mathematically proves that affect-aware RSA (more specifically, QUD-RSA) is a special case of (RSA)², but not vice versa, establishing that (RSA)² is theoretically more expressive.
- **Paradigm of Combining LLMs and Probabilistic Pragmatics**: It demonstrates a paradigm of extracting probability estimates from LLMs using prompting and embedding them into a structured reasoning framework. This paradigm can be generalized to other NLP tasks requiring structured reasoning.

## Limitations & Future Work

- The dataset sizes are small and limited to English; cross-lingual and cross-cultural figurative comprehension is yet to be validated.
- The generation method for alternative utterances limits the effectiveness of RSA recursive reasoning—generated alternative utterances are often literal paraphrases, failing to effectively differentiate between different intended meanings.
- The set of rhetorical strategies needs to be predefined; how to automatically discover suitable rhetorical strategies for specific scenarios remains an open question (the clustering method in the appendix performed poorly).
- LLM judgments of rhetorical strategies are insufficiently accurate in literal scenarios; improving the strategy classifier might further boost performance.

## Related Work & Insights

- **vs. Affect-Aware RSA (Kao et al., 2014/2015)**: These models achieve non-literal interpretation through emotional projection, which requires defining an emotional space for each scenario. (RSA)² replaces emotion with rhetorical strategies, making it more general and free from scenario-specific designs.
- **vs. Direct LLM Inference**: Directly using LLMs to understand figurative language suffers from a severe literal bias. (RSA)² corrects this bias through a structured pragmatic reasoning framework.
- **vs. Tsvilodub et al. (2025)**: They combine affect-aware RSA with LLMs for understanding number expressions. (RSA)² proposes a more generalized framework and does not rely on emotional variables.

## Rating

- Novelty: ⭐⭐⭐⭐ Introducing rhetorical strategies as latent variables into RSA is a theoretically elegant innovation with rigorous mathematical backing.
- Experimental Thoroughness: ⭐⭐⭐ The dataset scale is relatively small and scenarios are limited (mostly irony), but it includes theoretical proofs and multi-layered ablation studies.
- Writing Quality: ⭐⭐⭐⭐⭐ The paper is highly structured with clear explanations, rigorous mathematical derivations, and a naturally flowing narrative from motivation to method and experiments.
- Value: ⭐⭐⭐⭐ It makes significant theoretical contributions to probabilistic pragmatics and offers valuable practical reference for LLM figurative language understanding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Why Not Act on What You Know? Unleashing Safety Potential of LLMs via Self-Aware Guard Enhancement](why_not_act_on_what_you_know_unleashing_safety_potential_of_llms_via_self-aware_.md)
- [\[ACL 2025\] Recent Advances in Speech Language Models: A Survey](recent_advances_in_speech_language_models_a_survey.md)
- [\[ACL 2025\] Language-Codec: Bridging Discrete Codec Representations and Speech Language Models](language_codec_bridging_discrete_codec_speech_language_models.md)
- [\[ACL 2025\] OPTS: Bandit-Based Prompt Design Strategy Selection Improves Prompt Optimizers](bandit-based_prompt_design_strategy_selection_improves_prompt_optimizers.md)
- [\[ACL 2025\] ACT: Knowledgeable Agents to Design and Perform Complex Tasks](act_knowledgeable_agents_to_design_and_perform_complex_tasks.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Constitutional Classifiers: Defending Against Universal Jailbreaks Across Thousands of Hours of Red Teaming
description: >-
  [ACL 2025][LLM Alignment][Jailbreak Defense] Anthropic proposes "Constitutional Classifiers", which train input/output safety classifiers by generating synthetic training data from natural language safety principles (constitutions). In over 3000 hours of professional red teaming, no universal jailbreak attacks were discovered, while only incurring a 0.38% increase in over-refusal rate and a 23.7% inference overhead.
tags:
  - "ACL 2025"
  - "LLM Alignment"
  - "Jailbreak Defense"
  - "Constitutional Classifiers"
  - "Red Teaming"
  - "LLM Safety"
  - "Adversarial Robustness"
date: 2026-05-08
content_hash: 904b19fd4d284d13
---

# Constitutional Classifiers: Defending Against Universal Jailbreaks Across Thousands of Hours of Red Teaming

**Conference**: ACL 2025  
**arXiv**: [2501.18837](https://arxiv.org/abs/2501.18837)  
**Code**: None  
**Area**: LLM Alignment  
**Keywords**: Jailbreak Defense, Constitutional Classifiers, Red Teaming, LLM Safety, Adversarial Robustness

## TL;DR
Anthropic proposes "Constitutional Classifiers", which train input/output safety classifiers by generating synthetic training data from natural language safety principles (constitutions). In over 3000 hours of professional red teaming, no universal jailbreak attacks were discovered, while only incurring a 0.38% increase in over-refusal rate and a 23.7% inference overhead.

## Background & Motivation

**Background**: Large language models face severe threats from jailbreak attacks, where attackers bypass safety guardrails through carefully crafted prompts to elicit harmful outputs. Jailbreak attacks can be categorized into specific jailbreaks (targeting specific prompts) and universal jailbreaks (strategies capable of bypassing most safety constraints). Universal jailbreaks are particularly dangerous as attackers only need to discover a single technique to exploit harmful information at scale.

**Limitations of Prior Work**: (1) While safety alignment training based on RLHF/DPO can defend against simple jailbreaks, its efficacy against sophisticated universal jailbreaks (e.g., multi-step persuasion, roleplay, encoding transformation) remains limited. (2) Simple keyword filtering and rule-based systems are too coarse-grained, making them easy to bypass and prone to misclassifying benign requests. (3) Existing classifier defense schemes suffer from limited training data and fail to cover continuously evolving attack strategies. (4) Enhancing defenses often comes at the cost of significantly increased over-refusal rates (false positives on benign prompts), degrading user experience.

**Key Challenge**: There exists a trade-off between safety and usability—stronger defenses tend to mistakenly reject benign requests more frequently. A method is needed to substantially enhance safety while maintaining an acceptable over-refusal rate.

**Goal**: Design a classifier-based defense system capable of defending against universal jailbreak attacks while maintaining an extremely low over-refusal rate and acceptable inference overhead.

**Key Insight**: The authors draw inspiration from "Constitutional AI", which utilizes natural language safety rules (a constitution) to guide AI behavior. The difference lies in leveraging the constitution to generate training data for classifiers rather than directly imposing self-constraint on the model.

**Core Idea**: Starting from a natural language safety constitution, large-scale adversarial synthetic training data (including various jailbreak variants) is generated using an LLM. This data is used to train highly robust input/output classifiers that filter harmful requests and responses. The editability of the constitution allows safety policies to be updated at any time without retraining the base LLM.

## Method

### Overall Architecture
The defense system consists of three components: (1) Safety Constitution—a natural language rule set defining allowed and prohibited content; (2) Synthetic Data Generation Pipeline—which generates large-scale positive and negative training samples based on the constitution and known jailbreak strategies; (3) Input/Output Classifiers—detecting harmful content at the user input stage and model output stage respectively. The two classifiers operate in series: the input classifier detects malicious requests, and the output classifier detects harmful responses.

### Key Designs

1. **Constitution-driven Synthetic Data Generation**:

    - **Function**: Automatically generate large-scale classifier training data covering a wide range of attack strategies.
    - **Mechanism**: The safety constitution defines categories of prohibited content (e.g., weapon manufacturing, synthesis of illegal substances, cyberattacks) and boundary cases (e.g., academic discussion vs. practical manufacturing instructions) using natural language. The data generation pipeline proceeds in several steps: (a) starting from the constitutional rules, use an LLM to generate seed harmful requests for each category; (b) apply known jailbreak transformations (roleplay wrapping, multi-step decomposition, encoding obfuscation, multilingual mixing, etc.) to the seed requests to generate jailbreak variants; (c) simultaneously generate "boundary samples" that are topically close to prohibited content but actually benign as negative samples (e.g., discussing pharmaceutical safety research vs. drug manufacturing instructions). In total, millions of training samples covering dozens of jailbreak strategies were generated.
    - **Design Motivation**: Manually annotated jailbreak samples are limited in quantity and struggle to cover novel attacks. Synthetic data can be generated automatically at scale, and constitutional rules can be continuously expanded to address emerging attacks.

2. **Dual-Layer Classifier Architecture**:

    - **Function**: Detect harmful content at both the input and output stages to provide double-layered protection.
    - **Mechanism**: The input classifier determines whether a user request is malicious (even when wrapped in a jailbreak wrapper), utilizing a Transformer encoder architecture to output a binary classification probability based on the input text. The output classifier determines whether the model's response contains harmful information, using both the user request and intermediate model response as inputs. The two classifiers are trained independently and deployed in series. The input classifier's threshold is set more leniently (to minimize over-refusal), while the output classifier is stricter (serving as a backstop defense). If either classifier triggers an alert, the interaction is rejected and a safe refusal response is returned.
    - **Design Motivation**: A single classifier may have blind spots—certain jailbreaks might bypass input detection but still result in harmful output (and vice versa). Dual-layer protection is mutually complementary, significantly raising the overall difficulty for attackers.

3. **Large-scale Red Teaming and Iterative Refinement**:

    - **Function**: Validate and improve the classifiers through human adversarial testing.
    - **Mechanism**: Over 3000 hours of red teaming were organized, where professional safety researchers tried to discover universal jailbreak attacks capable of bypassing the classifiers' defense. Red teamers could use any strategy (including automated tools, multi-turn dialogues, encoding techniques, etc.). The evaluation metric was "information elicitation"—whether an attacker could obtain harmful information from the defended model with the same level of detail as from an undefended model. New attack strategies identified during red teaming were fed back into the synthetic data generation pipeline to yield new training data and update the classifiers, establishing a closed-loop defense-iteration pipeline.
    - **Design Motivation**: Automated evaluation cannot simulate the creativity of real-world attackers. Large-scale red teaming remains the gold standard for validating the robustness of safety systems.

### Loss & Training
The classifiers are trained using standard binary cross-entropy loss. Key training techniques include: (a) hard negative mining—increasing the loss weight for samples close to safety boundaries; (b) adversarial training—dynamically generating adversarial samples during training to enhance robustness; (c) post-hoc calibration—applying temperature scaling to the classifier's output probabilities to maximize the detection rate under a target over-refusal rate.

## Key Experimental Results

### Main Results

| Evaluation Dimension | Metric | Ours | RLHF Alignment Only | Keyword Filtering |
|---------|------|---------|-----------|-----------|
| Safety (Red Teaming) | Universal Jailbreak Success Rate | 0% (3000+ hours) | ~20% | ~40% |
| Safety (Automated) | Domain Jailbreak Defense Rate | >95% | ~65% | ~45% |
| Usability | Over-refusal Rate Increase | +0.38% | Baseline | +5.2% |
| Efficiency | Inference Overhead | +23.7% | Baseline | +2.1% |
| Coverage | Supported Attack Types | Dozens | Limited | Highly Limited |

### Ablation Study

| Configuration | Domain Jailbreak Defense Rate | Over-refusal Rate Increase | Description |
|------|-------------|-----------|------|
| Full system (Dual-layer + Synthetic Data) | >95% | +0.38% | Complete system |
| Input Classifier Only | 82% | +0.25% | Lacks output backstop |
| Output Classifier Only | 88% | +0.30% | Lacks prior interception |
| Without Synthetic Data (Human Labeled) | 71% | +0.82% | Poor generalization due to insufficient data |
| Without Red Team Iteration | 85% | +0.35% | Fails to cover novel attacks |

### Key Findings
- No universal jailbreak was found during 3000+ hours of red team testing—one of the largest red teaming validation efforts in the field.
- Synthetic data generation is the most critical component—the defense rate of the classifier drops from 95% to 71% without synthetic data, indicating that data coverage is far more important than model architecture.
- The dual-layer architecture improves the defense rate by 7-13 percentage points compared to a single-layer classifier, demonstrating a strong complementary effect.
- The over-refusal rate only increases by 0.38%, which is highly acceptable in practical deployments. In contrast, simple keyword filtering increases the over-refusal rate by 5.2%.
- The 23.7% inference overhead primarily stems from the additional forward passes of the classifiers, which can be further optimized via classifier quantization.

## Highlights & Insights
- The pipeline design of "Constitution -> Synthetic Data -> Classifier" is highly practical. Constitutional rules can be updated swiftly to counter new attacks without modifying the base model, which is of vital importance for rapid iteration in production.
- The scale of the 3000+ hours of red teaming is impressive. Feeding red-team discoveries back into the training data to form a closed-loop improvement is a key mechanism for maintaining long-term safety.
- The 0.38% over-refusal rate increase demonstrates that "strong safety does not necessarily sacrifice usability"—this challenges the previously widely held view of an irreconcilable trade-off between safety and usability.

## Limitations & Future Work
- As the authors are from Anthropic, the method might be specifically optimized for Claude models and its effectiveness may vary on other models.
- The definition of constitutional rules relies on human judgment; the boundary of "harmful" may vary across different cultural and legal contexts.
- The 23.7% inference overhead still impacts cost for large-scale deployments.
- Although large-scale, red teaming cannot guarantee coverage of all possible attack strategies—safety remains an ongoing adversarial process.
- The methodology primarily targets textual modalities; multimodal jailbreaks (e.g., using images to bypass safety) will require additional defense mechanisms.

## Related Work & Insights
- **vs Constitutional AI**: Constitutional AI uses a constitution to guide the model's own behavior (during the training phase), whereas this work uses the constitution to generate training data for classifiers (during the inference phase). The two are complementary.
- **vs Llama Guard**: Llama Guard is Meta's safety classifier but is based on small-scale human-annotated data. This work significantly increases coverage via synthetic data.
- **vs GCG/AutoDAN and other attacks**: These are automated jailbreak attack methods; the classifiers presented in this paper need to be capable of defending against both these automated attacks and manual human adversarial attacks.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The philosophy of constitution-driven synthetic data generation is innovative, and the large-scale red team validation marks a milestone.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Exceptionally thorough validation consisting of 3000+ hours of red teaming, automated evaluations, and ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Clear and comprehensive, offering a valuable reference for safety evaluation methodologies.
- **Value**: ⭐⭐⭐⭐⭐ Highly directly applicable to LLM safety defenses, presenting a pragmatic and viable defense solution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] M2S: Multi-turn to Single-turn jailbreak in Red Teaming for LLMs](m2s_multiturn_to_singleturn_jailbreak_in.md)
- [\[NeurIPS 2025\] PolyJuice Makes It Real: Black-Box, Universal Red Teaming for Synthetic Image Detectors](../../NeurIPS2025/llm_alignment/polyjuice_makes_it_real_black-box_universal_red_teaming_for_synthetic_image_dete.md)
- [\[ACL 2025\] MTSA: Multi-Turn Safety Alignment for LLMs through Multi-Round Red-Teaming](mtsa_multi-turn_safety_alignment_for_llms_through_multi-round_red-teaming.md)
- [\[NeurIPS 2025\] Jailbreak-Zero: A Path to Pareto Optimal Red Teaming for Large Language Models](../../NeurIPS2025/llm_alignment/jailbreak-zero_a_path_to_pareto_optimal_red_teaming_for_large_language_models.md)
- [\[ACL 2025\] Red Queen: Safeguarding Large Language Models against Concealed Multi-Turn Jailbreaking](red_queen_safeguarding_large_language_models_against_concealed_multi-turn_jailbr.md)

</div>

<!-- RELATED:END -->

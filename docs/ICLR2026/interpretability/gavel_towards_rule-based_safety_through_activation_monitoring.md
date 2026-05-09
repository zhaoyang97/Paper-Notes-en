---
title: >-
  [Paper Note] GAVEL: Towards Rule-Based Safety through Activation Monitoring
description: >-
  [ICLR 2026][Activation Monitoring] Inspired by the Snort/YARA ruleset paradigm from cybersecurity, this paper proposes decomposing LLM internal activations into 23 fine-grained "Cognitive Elements" (CEs), which are then composed via Boolean logic into auditable safety rules. On Mistral-7B, the approach achieves an average AUC of 0.99 and FPR of 0.004 across 9 misuse categories with less than 1% inference overhead, while naturally supporting cross-lingual and cross-model transfer.
tags:
  - ICLR 2026
  - Activation Monitoring
  - Cognitive Elements
  - Rule-Based Safety
  - Explainable AI Governance
  - LLM Safety
date: 2026-05-08
content_hash: 7749fd071e3b64e5
---

# GAVEL: Towards Rule-Based Safety through Activation Monitoring

**Conference**: ICLR 2026
**arXiv**: [2601.19768](https://arxiv.org/abs/2601.19768)
**Code**: Open-source (pending release)
**Area**: AI Safety / Interpretability
**Keywords**: Activation Monitoring, Cognitive Elements, Rule-Based Safety, Explainable AI Governance, LLM Safety

## TL;DR

Inspired by the Snort/YARA ruleset paradigm from cybersecurity, this paper proposes decomposing LLM internal activations into 23 fine-grained "Cognitive Elements" (CEs), which are then composed via Boolean logic into auditable safety rules. On Mistral-7B, the approach achieves an average AUC of 0.99 and FPR of 0.004 across 9 misuse categories with less than 1% inference overhead, while naturally supporting cross-lingual and cross-model transfer.

## Background & Motivation

**Background**: LLM safety has been evolving from surface-level text filtering toward monitoring internal activations. Surface-level censorship is easily bypassed by paraphrasing or obfuscation attacks, whereas hidden states more faithfully reflect the model's true cognitive intent. The dominant approach is to collect activation datasets for coarse-grained misuse categories (e.g., "cybercrime," "hate speech") and train linear probes or classifiers to detect such harmful behaviors.

**Limitations of Prior Work**: This "one classifier per category" paradigm suffers from three structural flaws. (1) **Low precision**: Coarse-grained categories compress diverse semantics into a single decision boundary. For instance, hate speech detectors misclassify legitimate discussions of minority cultures as harmful, and phishing detectors reach FPR = 0.35 on phishing samples. (2) **Poor flexibility**: Adding new detection dimensions—such as IP infringement or internal compliance—requires collecting new datasets and retraining classifiers from scratch, with thousands of activation samples per category and prohibitive costs at scale. (3) **Lack of interpretability**: When a classifier fires an alert, users cannot determine which behavioral factors triggered it, impeding auditing and accountability.

**Key Challenge**: Practical safety requirements demand precision, customizability, and interpretability, whereas existing activation-based safety methods are coarse-grained, fixed, and black-box. The underlying cause is that existing methods couple "activation engineering" (dataset construction) with "safety policy" (defining what constitutes a violation)—any policy change necessitates rebuilding the entire pipeline from data to model.

**Goal**: Three core sub-problems: How to define interpretable, composable activation-level behavioral primitives? How to assemble flexible safety rules from these primitives so that policy updates require no detector retraining? How to support a community-collaborative rule-sharing ecosystem?

**Key Insight**: The cybersecurity community has already validated the "community-shared ruleset" model through Snort/YARA/Sigma—encapsulating detection capabilities as human-readable rules that any organization can select, combine, and audit. If the detection unit for AI safety is decomposed from "coarse-grained misuse categories" into smaller "cognitive elements," safety policies can be written like firewall rules.

**Core Idea**: Decompose LLM behavior into 23 independent cognitive elements, train a dedicated detector for each CE, and combine CEs using Boolean logic $\wedge/\vee/\neg$ to precisely define violations—fully decoupling "perceptual capability" from "policy configuration."

## Method

### Overall Architecture

GAVEL operates in four stages. **Stage 1**: Define the CE vocabulary (23 elements across three categories) and compose Boolean rules, with the option to reuse community-published rulesets. **Stage 2**: Construct an elicitation dataset $\mathcal{D}_c$ for each CE, and extract activation vectors $\mathbf{H}_c$ from the target LLM using the ERI strategy. **Stage 3**: Train a lightweight multi-label RNN classifier $g$ on the activation data of all CEs. **Stage 4**: At inference time, extract activations per token → classifier predicts CE presence → aggregate CE occurrences within a time window $W_t$ → Boolean rules determine whether a violation has occurred and execute the appropriate action (block/replace/redirect). A key advantage is that CE datasets and rules are plain text and model-agnostic; cross-model reuse requires only re-extracting activations.

### Key Designs

1. **Cognitive Element (CE) Vocabulary**:

    - **Function**: Provides composable activation-level behavioral primitives serving as the "alphabet" for safety rules.
    - **Mechanism**: CEs cover three dimensions of model behavior—(a) **Instructions to the user** (7 CEs: purchase, click/input, download/install, go to a location, authorize/approve, provide/give, send/transfer); (b) **LLM's own behaviors** (9 CEs: create content, build trust, construct SQL queries, emotional engagement, threaten, spread hate speech, impersonate a human, flatter, spread conspiracy theories); (c) **Topics** (7 CEs: taxes, erroneous SQL syntax, electoral politics, personal information, payment instruments, LGBTQ+, racial identity). A critical property is **orthogonal composability**—"handling payments" is itself harmless, but "handling payments $\wedge$ impersonating a human $\wedge$ building trust" precisely characterizes fraud. This design allows individual CEs to be reused across countless rules, and the community can contribute CEs in the same manner as sharing Indicators of Compromise (IoC).
    - **Design Motivation**: Coarse-grained misuse categories inject unrelated signals into a single classifier. The fine-grained semantic isolation of CEs directly eliminates this problem.

2. **Elicitation Rewriting Instructions (ERI) Data Generation**:

    - **Function**: Generates high signal-to-noise activation data for each CE, ensuring the model's internal computation focuses on the target concept.
    - **Mechanism**: Naively prefilling text and collecting activations yields weak and noisy signals. ERI instead instructs the model: "Please rewrite the following text in the manner of [CE name]," forcing the model to concentrate its internal computation on the target CE's semantic space during generation. For each CE $c$, several hundred texts $\mathcal{D}_c$ are prepared, wrapped in ERI prompts, and fed into $f_\theta$; the attention outputs of the generated tokens across a set of consecutive layers $\Lambda$ are concatenated to form representation vectors $\mathbf{r}_t^{(c)} \in \mathbb{R}^D$. Ablation experiments show: (1) attention outputs (TPR 95.5%) substantially outperform MLP outputs (82.3%); (2) ERI significantly outperforms naive prefilling; (3) ERI with CE name specified outperforms a plain rewriting instruction (RI) without the name.
    - **Design Motivation**: ERI causes the model to "actively execute" the target CE rather than passively contain it, producing cleaner conceptual activation signals.

3. **Boolean Rule Engine and Time Window**:

    - **Function**: Aggregates token-level CE detections into conversation-level safety verdicts.
    - **Mechanism**: Each rule consists of a Boolean predicate and an action. Predicates combine multiple CEs using $\wedge/\vee/\neg$. For example, the phishing rule $\pi = c_8 \wedge (c_2 \vee c_6 \vee c_{20})$ means "the model is creating content while directing the user to click / provide information / disclose personal data." Rules are evaluated within the time window $W_t = \{t-N+1, \ldots, t\}$: a CE is considered present if it is detected at any token within the window, after which all rule predicates are evaluated. The syntax is designed in a human-readable format inspired by Snort/Sigma for rapid authoring and community sharing. The paper defines 9 rules covering 9 misuse scenarios across 3 domains; for example, the romance scam rule $c_{11} \wedge (c_1 \vee c_2 \vee \ldots \vee c_{21}) \wedge (c_9 \vee c_{14})$ precisely characterizes "emotional manipulation + any user instruction + trust-building/impersonation."
    - **Design Motivation**: Boolean composition ensures that individual, benign CEs trigger alerts only when co-occurring in specific contexts, fundamentally resolving the high false-positive rate of coarse-grained classifiers.

### Loss & Training

The CE detector $g$ is a 3-layer GRU (256 units) multi-label RNN operating on 5-token segments. Training samples are $(\mathbf{r}_t^{(c)}, \mathbf{e}_c)$, where $\mathbf{e}_c$ is the one-hot vector for CE $c$. The loss function is Binary Cross-Entropy, optimized with Adam (lr=$3 \times 10^{-4}$); each CE requires only 300 samples (80:20 split). Although each training sample is annotated for a single CE, the deployed classifier successfully generalizes to multi-CE co-occurrence scenarios—in experiments, 54% of malicious conversations contained tokens where multiple CEs were simultaneously active. Runtime overhead is approximately 150 MB of GPU memory and 0.21±0.01 ms per token (<1%).

## Key Experimental Results

### Main Results: GAVEL vs. 8 Baselines (Mistral-7B)

Evaluation covers 9 misuse scenarios across 3 domains (cybercrime, psychological harm, scam automation), using 14,950 multi-turn conversations (150 malicious + 500 similar benign dialogues per category to stress-test precision).

| Method | Type | Avg. AUC | Avg. b-ACC | Avg. FPR | Notable Weakness |
|--------|------|----------|------------|----------|-----------------|
| **GAVEL** | CE + Rules | **0.99** | **0.96** | **0.004** | No notable weakness |
| Activation Classifier | Activation | 0.97 | 0.92 | 0.07 | Phishing FPR=0.35 |
| Llama Guard 4 | Content Moderation | 0.87 | 0.93 | 0.03 | SQL Injection AUC=0.76 |
| RepBending | Fine-tuning | 0.87 | 0.87 | 0.02 | Delusional AUC=0.57 |
| CircuitBreakers | Fine-tuning | 0.68 | 0.69 | 0.06 | Delusional/Romance/E-commerce ≈0.50 |
| JBShield | Inference-time | 0.41 | 0.63 | 0.01 | AUC<0.50 in most scenarios |
| Perspective API | Moderation | 0.53 | 0.55 | 0.02 | All scam categories fail |
| OpenAI Moderator | Moderation | 0.69 | — | 0.00 | Scam AUC≈0.50 |

### Ablation Study / Cross-Domain / Robustness Analysis

| Dimension | Configurations | Key Result | Notes |
|-----------|---------------|------------|-------|
| Elicitation method | ERI vs. Naive Prefilling vs. RI | ERI outperforms both variants across the board | CE name context is critical for activation quality |
| Activation source | Attention output vs. MLP output | TPR 95.5% vs. 82.3% | Attention outputs carry richer contextual dependency information |
| Cross-lingual | English/Spanish/Chinese (English-only CE dataset) | Tax fraud TPR: 92%/90%/96% | CEs capture language-agnostic abstract cognitive concepts |
| Cross-model | Mistral/LLaMA-8B/Qwen3-8B/Gemma-4B | Same ruleset maintains high accuracy | CE datasets are plain text and model-agnostic |
| Adversarial attack | Misdirection (benign chat dilution) | GAVEL >85% vs. GPT-4 judge 12–15% | Activation monitoring is unaffected by surface-level "dilution" |
| Automated CE | PKU-SafeRLHF/ReasoningShield/ToxiGen | TPR 0.94–0.97 (zero human intervention) | Automated tools generate directly usable CEs and rules |
| Runtime overhead | Inference latency with GAVEL attached | +0.21±0.01 ms/token (<1%) | 150 MB GPU memory; deployable in real time |

### Key Findings

- **Composability is the core advantage**: CircuitBreakers achieves AUC ≈ 0.50—essentially random—on scenarios where violations require multiple semantic elements to co-occur (delusional, romance scam, e-commerce fraud). GAVEL precisely defines these compound violations through Boolean CE composition, achieving AUC > 0.98 across all 9 categories.
- **17× FPR reduction**: The comparable Activation Classifier achieves FPR = 0.35 on phishing (any dialogue involving content creation and user interaction triggers it), whereas GAVEL reduces FPR to 0.00 through precise CE composition, validating the precision advantage of fine-grained primitives.
- **Zero-shot cross-lingual transfer**: CE datasets are constructed exclusively in English, yet detection performance remains nearly unchanged on Spanish and Chinese conversations. This demonstrates that CEs capture language-agnostic abstract concepts in activation space rather than surface lexical patterns.
- **Adversarial robustness overwhelms text moderation**: Under the Misdirection attack, GPT-4 judge detection rates collapse to 12–15% as harmful content is diluted by benign chatter. GAVEL maintains >85% recall because it monitors the model's "cognitive state" rather than the output text.
- **Deceptive Marketing scenario**: When an LLM weaves self-promotion into medical advice, GPT-4 detects only 24% of cases; GAVEL achieves 73% by detecting the co-occurrence of Build Trust and Healthcare topic CEs.

## Highlights & Insights

- **Cybersecurity → AI safety paradigm transfer**: CEs function as shared IoCs, and rules function as Sigma detection templates. Model owners need only select and compose rules rather than train detectors. This is not a superficial analogy—GAVEL provides a complete technical stack (CE vocabulary + elicitation datasets + RNN detector + rule engine + automation tools) that makes the paradigm actionable.
- **The elegance of "individual CEs are benign; combinations constitute violations"**: This orthogonal design reduces false positives while enabling the number of expressible rules to grow exponentially with the number of CEs—theoretically, 23 CEs can express $2^{23}$ distinct combinations, from which only semantically meaningful ones need be selected.
- **Broad transferability of the ERI strategy**: The idea of eliciting clean activation signals by instructing the model to "rewrite text in the manner of a target concept" is applicable to probing, concept bottleneck models, interpretability feature extraction, and other settings.

## Limitations & Future Work

- **CE granularity relies on human expertise**: Although LLM-assisted automation tools are available, the semantic boundaries and granularity of CEs still require domain expert judgment. The current 23 CEs offer limited coverage; broad safety will require long-term community accumulation.
- **Boolean rules lack temporal expressiveness**: Pure Boolean logic cannot describe behavioral patterns with sequential dependencies, such as "first build trust → then make demands." The authors acknowledge the need for richer temporal logic (e.g., LTL) as a direct extension.
- **Limited model scale**: Validation is restricted to models with 4–8B parameters; behavior on 70B+ or closed-source models is unknown. Larger models have higher-dimensional activation spaces, and layer selection strategies may require adjustment.
- **Evaluation data are synthetic conversations**: The 14,950 dialogues were generated by GPT-4.1 and validated by GPT-5, which may not reflect the distributional diversity of real-world attacks.

## Related Work & Insights

- **vs. CAST**: CAST allows users to select steering vectors for coarse-grained misuse categories, but remains "one vector per category" and cannot express cross-category combinations. GAVEL achieves truly programmable safety through CE-level granularity.
- **vs. CircuitBreakers/RepBending**: These fine-tuning methods embed safety constraints into model weights at training time, offering poor flexibility and no interpretability. GAVEL does not modify model weights; it monitors activations only at inference time, and rules can be updated at any moment.
- **vs. content moderation APIs**: Text-level moderation tools such as Llama Guard and Perspective API nearly collapse under adversarial attacks and are orthogonal to GAVEL's activation-level monitoring—the two approaches can be stacked.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First rule-based activation safety framework; the CE + Boolean rule decoupling paradigm is an original contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 9 misuse categories × 14,950 conversations × 8 baselines × cross-model/cross-lingual/adversarial evaluation, but limited to small models with synthetic data.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — The motivation chain anchored in the cybersecurity analogy is exceptionally clear, and the framework is presented in a well-structured hierarchy.
- **Value**: ⭐⭐⭐⭐⭐ — A deployable AI safety governance framework; the CE + rule decoupling has direct practical value for industrial deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Beyond Linear Probes: Dynamic Safety Monitoring for Language Models](beyond_linear_probes_dynamic_safety_monitoring_for_language_models.md)
- [\[ICLR 2026\] ActivationReasoning: Logical Reasoning in Latent Activation Spaces](activationreasoning_logical_reasoning_in_latent_activation_spaces.md)
- [\[ICLR 2026\] Universal Properties of Activation Sparsity in Modern Large Language Models](universal_properties_of_activation_sparsity_in_modern_large_language_models.md)
- [\[ICLR 2026\] Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences](narrow_finetuning_leaves_clearly_readable_traces_in_activation_differences.md)
- [\[AAAI 2026\] Universal Safety Controllers with Learned Prophecies](../../AAAI2026/interpretability/universal_safety_controllers_with_learned_prophecies.md)

</div>

<!-- RELATED:END -->

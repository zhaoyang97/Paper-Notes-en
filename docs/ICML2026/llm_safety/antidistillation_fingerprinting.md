---
title: >-
  [Paper Note] Antidistillation Fingerprinting
description: >-
  [ICML2026][LLM Safety][Model Fingerprinting] This paper proposes Antidistillation Fingerprinting (ADFP), which utilizes a proxy student model to estimate which watermark tokens are most easily absorbed during the distill…
tags:
  - "ICML2026"
  - "LLM Safety"
  - "Model Fingerprinting"
  - "Antidistillation"
  - "Text Watermarking"
  - "Distillation Detection"
  - "Statistical Hypothesis Testing"
date: 2026-05-08
content_hash: b50d56ea9f8068cb
---

# Antidistillation Fingerprinting

**Conference**: ICML2026  
**arXiv**: [2602.03812](https://arxiv.org/abs/2602.03812)  
**Code**: https://github.com/YixuanEvenXu/antidistillation-fingerprinting  
**Area**: LLM Security  
**Keywords**: Model Fingerprinting, Antidistillation, Text Watermarking, Distillation Detection, Statistical Hypothesis Testing  

## TL;DR
This paper proposes Antidistillation Fingerprinting (ADFP), which utilizes a proxy student model to estimate which watermark tokens are most easily absorbed during the distillation process. This enables more reliable detection of whether a third-party model has been trained on the teacher model's outputs with minimal sacrifice to teacher output quality.

## Background & Motivation
**Background**: The training costs for frontier LLMs are extremely high. Model owners typically provide capabilities only through APIs or limited releases. Meanwhile, third parties can fine-tune smaller student models using teacher model outputs to replicate teacher behavior at a lower cost. Existing text watermarking methods, particularly red-and-green-list watermarks, use keys and hash functions to partition candidate tokens into a green list and a red list, then boost the probability of green tokens during sampling. If a student model later exhibits a higher preference for green tokens, this preference is treated as a trace of distillation.

**Limitations of Prior Work**: Traditional watermarks apply logit biases to all green tokens almost uniformly. While this introduces statistical signals into teacher outputs, it does not consider how the student model updates its parameters during fine-tuning. Consequently, ensuring the fingerprint actually enters the student model often requires strong perturbations to the teacher output, which degrades inference quality, conversational naturalness, or code correctness, and may even cause visible anomalies like repetition or formatting errors.

**Key Challenge**: Fingerprint detection relies on the student model retaining key-related green-token preferences after fine-tuning, whereas standard watermarking optimizes whether the teacher's current output favors the green list. These objectives are not perfectly aligned. Specifically, whether a token is in the green list is insufficient; the key is whether training on that token pushes the student model toward "generating green tokens more easily in the future."

**Goal**: The authors aim to transform "output watermarks" into true "model fingerprints" for distillation detection. This involves providing statistically interpretable p-values for both open-weight and black-box query evaluation scenarios and achieving higher detection confidence with smaller teacher quality losses across mathematical reasoning, open dialogue, and code generation tasks.

**Key Insight**: The paper borrows ideas from antidistillation sampling. If a proxy student model can approximate the learning dynamics of the real student, one can select tokens that more effectively influence the student's future behavior, rather than mechanically amplifying all green tokens. This proxy model does not need to be identical to the real student; it only needs to provide useful optimization directions.

**Core Idea**: Shift the watermark sampling objective from "making the current teacher output greener" to "sampling tokens that make the student greener after fine-tuning," using proxy model logit-space gradients to construct fingerprint perturbations oriented toward distillation learning dynamics.

## Method
The core of ADFP is not a new detector, but a rewritten perturbation method during the watermark sampling phase. The detection side still uses the familiar key hashing and green-token statistics from the red-and-green-list series, but the generation side no longer biases the green list uniformly. Instead, the perturbation magnitude depends on the predicted distribution of a proxy student model.

### Overall Architecture
The method consists of two phases. The first phase is teacher sampling with fingerprints: the model owner selects a hash function $H$, a key $k$, a window size $w$, and a green-list proportion $\gamma$. At each generation step, the green list $S=H(x_{-w:},k)$ is computed based on the last $w$ tokens of context. Then, the ADFP perturbation, based on the proxy model $\theta_p$'s predictive distribution, is added to the teacher logits before sampling the next token. These teacher outputs are then potentially used by a student model for fine-tuning.

The second phase is distillation detection: the model owner prepares a set of evaluation contexts $X$ and uses the same key $k$ to calculate the average green-list token probability (GTP) generated by the student model across these contexts. If the student has not been trained on data fingerprinted with that key, the GTP should fluctuate around $\gamma$. If it has, the GTP will be systematically higher. The paper uses Hoeffding's inequality to provide a conservative p-value: when observing $g_{obs}>\gamma$, $p=\exp(-2n(g_{obs}-\gamma)^2)$, where $n$ is the number of deduplicated evaluation contexts.

Detection is divided into two scenarios. For open-weight students, green-token probabilities can be calculated directly from logits for each context. For black-box students, the next token is sampled once per context to calculate green-token frequency. Both share the same null hypothesis: the student's generation is independent of the key.

### Key Designs
1.  **Logit Perturbation for Student Learning Dynamics**:
    - **Function**: Replaces the uniform green-token scoring of traditional red-and-green-lists, making the perturbation magnitude for each token reflect the detection gain after it is absorbed by student fine-tuning.
    - **Mechanism**: Let the proxy model's predictive distribution be $q$, the green list be $S$, and the total green-token probability be $L=\sum_{t\in S}q_t$. The ADFP perturbation for token $t$ is defined as $\Delta^{ADS}_t=q_t(\mathbf{1}[t\in S]-L)$. High-probability green tokens receive larger positive perturbations because they are both more likely to be sampled and more likely to serve as effective supervision in student training. High-probability red tokens are suppressed because they push the student in a non-fingerprint direction.
    - **Design Motivation**: Traditional watermarks only know if a token is green or red, not whether it is in a position easily learned by the student. ADFP embeds the "detection goal" into the sampling strategy, reducing the need for aggressive perturbations to ensure fingerprint internalization.

2.  **Computational Approximation to Avoid Per-word Backpropagation**:
    - **Function**: Simplifies the originally expensive parameter-space inner product of antidistillation sampling into a closed-form logit score dependent only on proxy model softmax probabilities.
    - **Mechanism**: Starting from the ADS form $\Delta_t=\langle\nabla_{\theta_p}\log q_t,\nabla_{\theta_p}L\rangle$, the paper projects gradients into logit space and approximates the Gram matrix of logits relative to parameter gradients as isotropic $K\approx cI$. Under this approximation, token-independent constant terms cancel out during sampling normalization, leaving $q_t(\mathbf{1}[t\in S]-L)$. The authors also prove that if only the last linear layer of the proxy model is trainable, this isotropic conclusion holds exactly.
    - **Design Motivation**: Performing a backward pass for every vocabulary token makes online sampling infeasible. This approximation reduces the complexity to a single proxy forward pass, allowing it to be integrated into standard LLM decoding.

3.  **Unified Statistical Detection for Open-Weight and Black-Box Queries**:
    - **Function**: Ensures the fingerprint conclusion does not require access to student model weights and provides conservative significance for query-only student services.
    - **Mechanism**: The paper constructs evaluation contexts $X$ deduplicated by the last $w$ tokens, ensuring green lists are approximately independent under key randomness. In open-weight scenarios, green-token probabilities are averaged; in black-box scenarios, Bernoulli indicator variables are constructed from single-sample results. In both cases, each term is an independent variable in $[0,1]$ with mean $\gamma$ under the null hypothesis, allowing the same p-value upper bound to control false alarm risks.
    - **Design Motivation**: Real-world model attribution often faces API-only students. Framing the detector as a probabilistic statistical problem is more robust than "checking for reproduction of specific training samples" and better handles cases where fingerprint signals are diluted by distillation.

### Loss & Training
ADFP itself is not for training a new model but modifies the sampling distribution during teacher generation. The sampling distribution in Algorithm 1 can be understood as adding $\lambda\Delta^{ADS}$ to the teacher's log probabilities, then normalizing with temperature $\tau$. $\lambda$ controls fingerprint strength: higher values typically make the fingerprint more obvious but are more likely to damage teacher output quality.

Student fine-tuning in experiments is simulated via LoRA: 1 epoch for GSM8K and OASST1, and 3 epochs for MBPP due to fewer samples. The optimizer is AdamW with a learning rate of $1e^{-4}$, batch size of 8, LoRA rank 128, $\alpha=128$, and dropout 0.05. The appendix also tests 1/3 epoch full fine-tuning and 8-bit/4-bit QLoRA to verify robustness across fine-tuning methods.

## Key Experimental Results

### Main Results
The paper evaluates three task domains: GSM8K (math), OASST1 (dialogue), and MBPP (code). Teacher, proxy, and student models are intentionally not always identical: GSM8K/OASST1 use DeepSeek-R1-Distill-Qwen-7B as teacher and Qwen2.5-3B as proxy, while students can be Qwen2.5-3B or Llama-3.2-3B. MBPP uses Qwen2.5-Coder-7B-Instruct as teacher and Qwen2.5-Coder-3B-Instruct as proxy. The main figures show Pareto curves of p-value versus teacher output quality.

| Dimension | ADFP Results | Red-and-Green-List Baseline | Notes |
| :--- | :--- | :--- | :--- |
| GSM8K Unsupervised | Lower p-value at same teacher answer-forced accuracy; e.g., $\sim 0.09$ to $0.01$ | Requires stronger perturbation for same detection confidence | ADFP improves false alarm bounds by nearly an order of magnitude with smaller quality loss |
| OASST1 Unsupervised | Maintains lower p-values where NLL is close to original teacher | Strong watermarking increases NLL more quickly | Demonstrates ADFP applies to tasks beyond short-answer reasoning |
| MBPP Unsupervised | Significantly lower p-value at similar execution pass rates | Quality-fingerprint curve lags behind | Code generation is sensitive to perturbation; ADFP retains Pareto advantage |
| Student Post-distillation Acc | Maintains better student accuracy when fingerprint is strong; minimal degradation when proxy=student | Strong perturbations more easily damage final student accuracy | Indicates ADFP fingerprints are more subtle and don't just "break" the student |

### Ablation Study
The appendix provides key analyses, including varying fine-tuning methods, fingerprint data proportions, and comparing supervised/unsupervised detection using ROC/AUC.

| Student Fine-tuning Setting | Open-weight Unsupervised log p-value: ADFP | Open-weight Unsupervised log p-value: RGL | Black-box Unsupervised log p-value: ADFP | Black-box Unsupervised log p-value: RGL |
| :--- | :--- | :--- | :--- | :--- |
| LoRA (Default) | -4.013 ± 1.054 | -1.134 ± 0.638 | -3.478 ± 1.206 | -1.740 ± 1.477 |
| Full FT, 1 epoch | -1.439 ± 0.681 | -0.201 ± 0.257 | -1.871 ± 1.456 | -0.281 ± 0.220 |
| Full FT, 3 epochs | -7.914 ± 1.719 | -1.064 ± 0.733 | -8.239 ± 2.805 | -1.601 ± 0.655 |
| QLoRA, 8-bit | -3.385 ± 1.076 | -0.746 ± 0.584 | -3.533 ± 1.178 | -0.661 ± 0.643 |
| QLoRA, 4-bit | -3.393 ± 1.041 | -0.753 ± 0.541 | -4.000 ± 1.209 | -0.556 ± 0.518 |

| Analysis Item | Key Setting | Observation | Implication |
| :--- | :--- | :--- | :--- |
| Partial Fingerprinted Data | GSM8K, ADFP $\lambda=256$, RGL $\delta=7$ | Both weaken as fingerprint ratio $\alpha$ drops, but ADFP remains stronger across most $\alpha$ | Signals are effective even with mixed data sources |
| Supervised Evaluation | Detection set = student training data | p-values are stronger than unsupervised; ADFP superior in most Pareto settings | Upper bound is higher if training samples are known |
| ROC/AUC | GSM8K, ADFP $\lambda=140$, RGL $\delta=6$ | ADFP AUC is higher across all settings; in black-box scenario, TPR is 55% vs 24% at FPR=0 | ADFP's advantage is concentrated in low-false-alarm regions |
| p-value Calibration | 100 non-fingerprinted student trials | Empirical FPR is covered by theoretical p-value upper bound | Statistical detection provides conservative, interpretable results |

### Key Findings
- ADFP's advantage primarily stems from "stronger fingerprints at the same quality" rather than simply increasing perturbation.
- The advantage persists even when the proxy model $\neq$ real student, although it weakens. This fits the hypothesis: the better the proxy approximates student dynamics, the more accurate $\Delta^{ADS}$ is.
- Open-weight detection is more sample-efficient than black-box detection, but trends are consistent.
- Qualitative samples show RGL is more prone to repetition and formatting collapse under strong fingerprints, while ADFP remains more coherent at similar accuracy levels.

## Highlights & Insights
- The most significant insight is shifting watermarking from "output distribution bias" to "learning dynamic bias." To detect distillation, the sampling strategy should optimize for statistical signals *after* student training.
- The ADFP formula $q_t(\mathbf{1}[t\in S]-L)$ elegantly captures both token learnability and green/red direction. High-probability tokens act as more effective training labels.
- The statistical detection is robust. By using the Hoeffding bound to output conservative p-values, the method avoids deterministic model attribution, which is crucial for reducing false accusations.
- This logic can extend to other "post-training trace" scenarios, such as benchmark contamination detection or data licensing audits.

## Limitations & Future Work
- **Proxy Dependence**: While effective when proxy $\neq$ student, the advantage narrows. Larger proxy errors may occur with heterogeneous training pipelines or complex data cleaning.
- **Output Perturbation**: Although more efficient than RGL, strong ADFP fingerprints still damage output quality. Engineering strategies for adaptive $\lambda$ are needed for production APIs.
- **Context Requirements**: Detection relies on independent green lists from deduplicated context windows. Constructing sufficient natural contexts that aren't filtered by student services remains a hurdle.
- **Scale and Robustness**: Evaluation was limited to 3B/7B models. Effects of larger scales, RLHF, de-watermarking attacks, or paraphrasing require further study.

## Related Work & Insights
- **vs Red-and-Green-List Watermark**: Standard schemes bias green tokens uniformly. ADFP uses the same detection framework but weights the generation phase by proxy student learning gains.
- **vs Watermarking Makes Language Models Radioactive**: Radioactive watermarking proved output watermarks migrate to downstream students. ADFP advances this by designing perturbations specifically suited for distillation migration.
- **vs Antidistillation Sampling**: Original ADS aimed to disrupt distilled student performance. ADFP repurposes the gradient idea for detectable, statistical fingerprinting while preserving teacher quality.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Connects antidistillation learning dynamics with statistical fingerprinting cleanly.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers multiple domains and fine-tuning settings, though lacks massive-scale model and adversarial evasion evaluations.
- **Writing Quality**: ⭐⭐⭐⭐ Clear narrative; theoretical derivations and experiments are well-integrated.
- **Value**: ⭐⭐⭐⭐⭐ High practical value for IP protection and model attribution in the API era.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] iSeal: Encrypted Fingerprinting for Reliable LLM Ownership Verification](../../AAAI2026/llm_safety/iseal_encrypted_fingerprinting_for_reliable_llm_ownership_verification.md)
- [\[ICML 2026\] FedTreeLoRA: Reconciling Statistical and Functional Heterogeneity in Federated LoRA Fine-Tuning](fedtreelora_reconciling_statistical_and_functional_heterogeneity_in_federated_lo.md)
- [\[ICML 2026\] Beyond Procedure: Substantive Fairness in Conformal Prediction](beyond_procedure_substantive_fairness_in_conformal_prediction.md)
- [\[ICML 2026\] Anchored Decoding: Provably Reducing Copyright Risk for Any Language Model](anchored_decoding_provably_reducing_copyright_risk_for_any_language_model.md)
- [\[ICML 2026\] BioAgent Bench: An AI Agent Evaluation Suite for Bioinformatics](bioagent_bench_an_ai_agent_evaluation_suite_for_bioinformatics.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[AAAI 2026\] iSeal: Encrypted Fingerprinting for Reliable LLM Ownership Verification](../../AAAI2026/llm_safety/iseal_encrypted_fingerprinting_for_reliable_llm_ownership_verification.md)
- [\[ICML 2026\] FedTreeLoRA: Reconciling Statistical and Functional Heterogeneity in Federated LoRA Fine-Tuning](fedtreelora_reconciling_statistical_and_functional_heterogeneity_in_federated_lo.md)
- [\[ICML 2026\] Beyond Procedure: Substantive Fairness in Conformal Prediction](beyond_procedure_substantive_fairness_in_conformal_prediction.md)
- [\[ICML 2026\] Position: Retire the "Positive Backdoor" Label -- Secret Alignment Requires Strict and Systematic Evaluation](position_retire_the_positive_backdoor_label_--_secret_alignment_requires_strict_.md)
- [\[ICML 2026\] Anchored Decoding: Provably Reducing Copyright Risk for Any Language Model](anchored_decoding_provably_reducing_copyright_risk_for_any_language_model.md)

</div>

<!-- RELATED:END -->

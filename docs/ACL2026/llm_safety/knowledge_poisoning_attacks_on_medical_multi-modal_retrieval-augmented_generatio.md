---
title: >-
  [Paper Note] Knowledge Poisoning Attacks on Medical Multi-Modal Retrieval-Augmented Generation
description: >-
  [ACL 2026][LLM Safety][Knowledge Poisoning] The authors propose M3Att—the first **query-agnostic** knowledge poisoning framework for medical multimodal RAG. It utilizes "distribution-guided visual PGD triggers" for retri…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Knowledge Poisoning"
  - "Medical RAG"
  - "PGD Perturbation"
  - "Clinical Ambiguity"
  - "query-agnostic attack"
date: 2026-05-08
content_hash: 4c0d73bb92adff7c
---

# Knowledge Poisoning Attacks on Medical Multi-Modal Retrieval-Augmented Generation

**Conference**: ACL 2026  
**arXiv**: [2605.10253](https://arxiv.org/abs/2605.10253)  
**Code**: https://github.com/ypr17/M3Att  
**Area**: LLM Security  
**Keywords**: Knowledge Poisoning, Medical RAG, PGD Perturbation, Clinical Ambiguity, query-agnostic attack

## TL;DR
The authors propose M3Att—the first **query-agnostic** knowledge poisoning framework for medical multimodal RAG. It utilizes "distribution-guided visual PGD triggers" for retrieval hijacking and "clinical ambiguity-guided text rewriting" to bypass LVLM self-correction. With a poisoning rate of $<1\%$ (no knowledge of queries required, visual perturbation $\epsilon=16/255$), it reduces downstream utility by an average of 8.78% across 5 LVLMs $\times$ 5 datasets $\times$ 4 medical tasks, while remaining robust against three pre-retrieval defenses: image clustering, text clustering, and image-text consistency.

## Background & Motivation
**Background**: Medical multimodal RAG systems (retrieving image-report pairs) are rapidly being deployed—models like LLaVA-Med and Med-Gemini rely heavily on external knowledge bases to improve performance in tasks like VQA, report generation, and image classification. This makes "poisoning the knowledge base" a new attack surface: Ha et al. 2025, Liu et al. 2025b, and Zuo et al. 2025 have already demonstrated knowledge poisoning attacks in general or medical RAG settings.

**Limitations of Prior Work**: (1) Almost all existing multimodal RAG poisoning methods assume a **query-aware** setting—attackers know what users will ask in advance and optimize poisoned entries accordingly; this is unrealistic in real deployments where user queries are typically unavailable. (2) Medical images (X-rays, tissue slides) exhibit extreme anatomical consistency, resulting in highly clustered embedding distributions. Simply increasing the number of poisoned entries to ensure retrieval risks exposure. (3) SOTA medical LVLMs, pre-trained on medical corpora with safety alignment, may trigger refusal or automatic correction when encountering "obverse factual errors," while weak perturbations fail to influence generation; it is difficult to find a "dosage" that influences output while bypassing self-correction.

**Key Challenge**: Query-aware attacks fail in real environments; however, in a query-agnostic setting, one faces the dual constraints of being "submerged in dense embeddings during retrieval" and "self-corrected by the LVLM prior during generation," creating a dual-constraint problem.

**Goal**: (1) Construct a query-agnostic, weak-prior (knowledge of distribution only, no query required) poisoning framework; (2) Design independent mechanisms for retrieval and generation stages; (3) Demonstrate effectiveness across 5 LVLMs $\times$ 3 retrievers $\times$ 4 medical tasks, and verify robustness against common pre-retrieval defenses.

**Key Insight**: (A) While medical image homogeneity makes query-specific attacks difficult, it provides a highly structured latent space where **cluster centers** can serve as "representative query proxies"—perturbing near cluster centers allows coverage of all unknown queries within that cluster. (B) Medical diagnosis inherently contains clinical ambiguity (e.g., "severe vs. mild," differential diagnosis, defensive medicine), which corresponds to low-confidence regions for LLM priors. Attackers lying in these "gray areas" make it difficult for the model to self-correct.

**Core Idea**: Use "distribution-guided visual PGD hijacking" to optimize poisoned images near cluster centers as query-agnostic triggers; use "clinical ambiguity-guided three-level progressive text rewriting" to inject plausible but incorrect medical conclusions across severity migration, diagnostic distortion, and risk association levels. Together, these form the query-agnostic, stealthy, and dual-stage coupled medical RAG poisoning framework M3Att.

## Method

### Overall Architecture
Threat model: Attackers cannot access model parameters, user queries, or retrieval contexts, and can only inject a limited budget ($<1\%$ poisoning rate) of malicious entries into the knowledge base. The pipeline consists of three steps: (1) **Cluster Profiling**—the attacker obtains a subset of the knowledge base distribution (reference pool) via black-box interaction and calculates image embeddings followed by K-Means ($K=40$) to obtain cluster centers $\bm{\mu}_c$; (2) **Distribution-guided Retrieval Hijacking**—selecting candidate images for each cluster center and using PGD under $\ell_\infty \leq 16/255$ constraints to maximize cosine similarity to $\bm{\mu}_c$, generating "high retrieval probability yet visually imperceptible" poisoned images; (3) **Clinical Ambiguity-guided Text Poisoning**—using GPT-5 as a controlled editor to rewrite paired medical reports following a three-level progressive strategy to inject plausible but incorrect clinical conclusions. Finally, (poisoned image, poisoned text) pairs are inserted into the knowledge base to await natural triggering by user queries.

### Key Designs

1.  **Distribution-Guided Retrieval Hijacking (Cluster center-based query-agnostic PGD hijacking)**:

    -   **Function**: Ensure poisoned images are retrieved with high probability by arbitrary future queries without knowing the specific query.
    -   **Mechanism**: (a) **Cluster Profiling**: Perform $K=40$ K-Means on the reference pool, averaging the top-50 nearest samples per cluster to obtain $\bm{\mu}_c$ as a semantic proxy for that cluster; (b) **Candidate Sampling**: Rank embedding similarity for each cluster in a non-overlapping candidate pool, using a 10-step PGD warm-up to evaluate the optimization potential of candidates and pick the optimal seed; (c) **Constrained PGD Refinement**: Refine the seed image iteratively via $\bm{x}_c^{(i+1)} = \Pi_{\mathcal{B}_\epsilon}(\bm{x}_c^{(i)} + \alpha \cdot \mathrm{sign}(\nabla_x \mathcal{L}(f(\bm{x}_c^{(i)}), \bm{\mu}_c)))$ for $N=500$ steps with $\epsilon=16/255$ and $\alpha=1/255$, targeting cosine similarity maximization. Under white-box settings, gradients are calculated directly; under black-box settings, they are estimated via zeroth-order symmetric finite difference: $\nabla_x \mathcal{L} \approx \frac{1}{K}\sum_k \frac{\mathcal{L}(\bm{x}+\sigma u_k) - \mathcal{L}(\bm{x}-\sigma u_k)}{2\sigma} \cdot u_k$.
    -   **Design Motivation**: Cluster centers capture "the data's intrinsic semantic structure" rather than "model-specific features," making the attack transferable across retrievers (CLIP/BGE-VL/SigLIP); warm-up seed selection avoids wasting PGD resources on hard-to-optimize samples; the $\ell_\infty$ constraint ensures visual imperceptibility to bypass clinical review. This design cleverly exploits **high homogeneity** in medical images—converting a barrier into an advantage where "few cluster centers cover massive queries."

2.  **Clinical Ambiguity-Guided Poisoning (Three-level progressive text rewriting)**:

    -   **Function**: Ensure poisoned text is accepted by the LVLM as a "plausible alternative explanation" rather than an "obvious error," thereby bypassing self-correction in medical safety alignment.
    -   **Mechanism**: Use GPT-5 as a controlled LLM editor, strictly executing three strategies via system prompts: (a) **Fine-grained Severity Migration**: Bidirectionally modify severity terms—down-scaling "massive" $\to$ "moderate" or "acute" $\to$ "chronic" to induce under-diagnosis; up-scaling "unremarkable" $\to$ "suspicious density" to trigger over-intervention; (b) **Prior-Constrained Diagnosis Distortion**: Instead of random disease replacement (which is easily rejected by priors), find a candidate set with overlapping visual features and select a target with a similar prior probability to the ground truth (e.g., "Viral Pneumonia" $\to$ "Pulmonary Edema"), causing the LVLM to accept the poisoned context as a legitimate "differential diagnosis"; (c) **Risk Association Corruption**: Bidirectionally manipulate recommendation urgency—urgency suppression ("immediate CT" $\to$ "follow-up in 6 months") to mask positive findings; defensive overreach ("cannot rule out malignancy") to manufacture false positives. These three levels correspond to perceptual evidence $\to$ diagnostic hypothesis $\to$ decision risk.
    -   **Design Motivation**: Directly replacing diseases is often rejected by LVLM internal priors; however, modifications in severity, differentials, or risk assessment—inherently ambiguous areas—land exactly in low-confidence LLM regions. This targets the "gray areas" of medical decision-making as an attack surface.

3.  **Black-box + White-box dual gradient paths + Dual-stage coupling**:

    -   **Function**: Maintain attack effectiveness in real-world black-box retriever scenarios.
    -   **Mechanism**: White-box settings use direct backpropagation for $\nabla_x \mathcal{L}$; black-box settings use zeroth-order symmetric finite difference estimation. M3Att is a tight coupling of retrieval hijacking and text injection—ablation shows removing either component significantly restores downstream utility (w/o Hijack prevents retrieval of poisoned entries; w/o Injection makes retrieved samples harmless).
    -   **Design Motivation**: Deployed medical RAG retrievers are often closed-source, so attacks must be viable in black-box settings. Experiments show black-box ASR is close to white-box, proving M3Att does not rely on gradient access.

### Loss & Training
Key loss: Cosine similarity loss $\mathcal{L}(f(\bm{x}), \bm{\mu}_c) = \cos(f(\bm{x}), \bm{\mu}_c)$, with constraint $\bm{x} \in \mathcal{B}_\epsilon(\bm{x}^{(0)}) = \{\bm{x}: \|\bm{x} - \bm{x}^{(0)}\|_\infty \leq \epsilon\}$. Key hyperparameters: $K=40$ clusters, 1 optimized candidate per cluster (poison rate $<0.01$), $\epsilon=16/255$, $\alpha=1/255$, 500 PGD steps, 10 warm-up steps. Text editing is performed by GPT-5 using system prompts (Appendix Fig.9) specifying stealthiness and progressive strategies.

## Key Experimental Results

### Main Results: End-to-end attack effects across 5 LVLMs $\times$ 4 tasks (Partial excerpt, lower is worse)

| LVLM | Retriever | Method | True/False (IU-XRay) | MC (MIMIC) | Report FC (IU-XRay) | Img Cls (CRC100k) |
|------|-----------|--------|---------------------|------------|--------------------|--------------------|
| GPT-4o | – (w/o RAG) | – | 67.36% | 58.02% | 18.89% | 46.66% |
| GPT-4o | – (Clean RAG avg) | – | 89.64% | 69.57% | 31.04% | 93.30% |
| GPT-4o | CLIP | LIAR | 83.90% | 64.09% | 34.47% | 89.67% |
| GPT-4o | CLIP | **M3Att** | **77.88%** | **59.98%** | **32.39%** | **78.41%** |
| GPT-4o | BGE-VL | M3Att | 80.44% | 58.84% | 23.70% | 70.62% |
| GPT-5 | BGE-VL | M3Att | **93.54%** | **72.26%** | **35.11%** | **68.58%** |
| Claude-4.5 | CLIP | M3Att | **47.04%** | **61.41%** | **21.64%** | **69.28%** |
| LLaVA-Med | BGE-VL | M3Att | **46.56%** | **3.51%** | **17.04%** | **50.16%** |
| Gemini-2.5 | CLIP | M3Att | 76.12% | 39.21% | 32.40% | 79.85% |

M3Att is significantly stronger than the baseline LIAR across **the vast majority of LVLM $\times$ Retriever $\times$ Task combinations**; it reduces downstream utility from Clean RAG by an average of **8.78%**.

### Ablation Study: Component contribution + Defense robustness + Hyperparameters

| Setting | Key Metric | Key Observation |
|------|---------|---------|
| Full M3Att | Full Effect | Strongest attack |
| w/o Hijack (using samples nearest to cluster center) | Utility Recovery | Poisoned entries fail to enter top-k reliably; text poisoning ineffective |
| w/o Injection (poisoned image but original text) | Utility Recovery | Retrieved samples are harmless; generation is unaffected |
| Filtered (eval retrieving success subset only) | M3Att leads LIAR | Once poisoned entries are retrieved, text poisoning stably dominates generation |
| Defense: Image Clustering | ASR essentially unchanged | Visual perturbations are small; no distributional anomalies |
| Defense: Text Clustering | ASR essentially unchanged | GPT-5 rewritten text maintains clinical fluency |
| Defense: Image-Text Consistency | ASR essentially unchanged | Image and text remain highly aligned |
| Poison rate 0.08 | ASR $\approx$ 100% | Low poisoning rates ($<0.01$) are sufficient for significant effects |
| $\epsilon$ increase | ASR Saturates | Moderate perturbations are sufficient |
| Increase K (K > 40) | Improvement plateaus | Semantic clusters in medical imaging are limited |

### Key Findings
- **Query-agnostic poisoning is feasible in medical scenarios**: Without relying on any query information, "cluster center proxy + PGD" alone allows poisoned image ASR@Top-5 to soar from 0.01% to 5%.
- **Black-box $\approx$ White-box**: Attack effects using zeroth-order gradient estimation are close to white-box settings, proving real-world closed-source retrievers are equally vulnerable.
- **Two stages are indispensable**: Removing either hijacking or injection significantly degrades the attack, indicating that medical RAG attacks must combine retrieval and generation.
- **Three simple defenses fail**: Image Clustering, Text Clustering, and Image-Text Consistency all fail to hold up, suggesting that common "distributional anomaly" or "cross-modal mismatch" filtering strategies lack defense against stealthy attacks like M3Att. Robust medical fact-checking mechanisms are needed.
- **Clinical ambiguity is a natural attack surface**: Tampering with severity, differential diagnosis, and risk recommendations—inherently vague aspects—targets low-confidence LLM regions and medical "gray areas."
- **Poison rate $<1\%$ is sufficient**: Injecting only $K=40$ entries (less than 1% of the knowledge base) reduces downstream utility by 8.78% on average, which translates to differences in thousands of diagnoses in medical settings.

## Highlights & Insights
- **Paradigm shift: "High homogeneity as a barrier vs. an opportunity"**: The high homogeneity of medical images, which usually makes query-specific attacks difficult, is used as a design lever to cover massive queries with few cluster centers. This strategy of extracting value from constraints is noteworthy.
- **Clinical ambiguity as an attack surface**: Dividing "severity / differentials / risk assessment" into three progressive attack strategies is a brilliant case of integrating deep medical domain knowledge into adversarial design, transferable to other "high-stakes + inherently ambiguous" fields (e.g., law, finance).
- **PGD on retrieval embeddings + LLM editor dual attack primitives**: Parallelizing visual adversarial perturbations and text LLM-as-editor provides a recipe that almost all future multimodal RAG attacks can adopt.
- **Zeroth-order black-box capability**: Proving that real-world closed-source medical RAGs (like those via OpenAI APIs) are insecure pushes the threat model to production-grade.
- **Failure of simple defenses**: The negative results for distributional anomaly and cross-modal consistency filters serve as a valuable red-team baseline for the trustworthy medical AI community.

## Limitations & Future Work
- **Validated only on 2D images**: X-rays and tissue slides are dominant but not exhaustive; 3D volumes (CT/MRI) and temporal medical videos were not tested.
- **Dependence on GPT-5 for text rewriting**: Generating poisoned text requires a strong editor LLM; results might degrade with weaker models.
- **Expert verification or medical NER consistency not considered**: Attacks might be harder if hospital RAG deployments include expert review or NER-based knowledge graph consistency checks.
- **K=40 is empirical**: Cluster numbers depend on database size and image types, requiring tuning for cross-database migration.
- **No defense proposed**: As a pure attack paper, it lacks constructive solutions for the community.
- **Future directions**: (1) Extension to 3D and temporal data; (2) Proposing retrieval-stage defenses (e.g., "leave-one-out perturbation detection" for candidates or physics-based medical fact checking); (3) Studying whether fine-tuned medical LVLMs are more or less robust.

## Related Work & Insights
- **vs. LIAR (Tan et al. 2024)**: Representative baseline for text-only RAG poisoning; this work extends it to multimodal, medical, and query-agnostic settings with superior stability.
- **vs. MM-PoisonRAG (Ha et al. 2025) / Poisoned-MRAG (Liu et al. 2025b)**: Both rely on query-specific optimization; M3Att is the first query-agnostic multimodal medical poisoning method.
- **vs. HV-Attack (Luo et al. 2025)**: General multimodal RAG attack that fails on highly homogeneous medical corpora; M3Att solves this with cluster center proxies.
- **vs. Alber et al. (2025, Nature Medicine)**: Found medical LLMs are susceptible to data poisoning; this paper provides a finer-grained, more stealthy attack path at the RAG stage.
- **Transferability to Law/Finance RAG**: Clinical ambiguity strategies can be generalized to legal interpretation or financial advice where "inherent ambiguity + high stakes" form a "gray area attack" paradigm.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Query-agnostic + dual-stage coupled + clinical ambiguity-guided; the first practical threat model for medical multimodal RAG poisoning.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 5 LVLM $\times$ 3 retrievers $\times$ 5 datasets $\times$ 4 tasks + White/Black-box + 3 defenses + hyperparameters + ablations + case studies.
- Writing Quality: ⭐⭐⭐⭐ Clear formulas and tables; the three-strategy attack is somewhat cookbook-style but supported by medical depth.
- Value: ⭐⭐⭐⭐⭐ Directly exposes the vulnerability of medical RAG even in query-agnostic/weak-prior settings; significant for trustworthy medical AI and RAG red-teaming, though its release carries dual-use risks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Beyond Explicit Refusals: Soft-Failure Attacks on Retrieval-Augmented Generation](beyond_explicit_refusals_soft-failure_attacks_on_retrieval-augmented_generation.md)
- [\[ACL 2026\] Differentially Private Synthetic Text Generation for Retrieval-Augmented Generation (RAG)](differentially_private_synthetic_text_generation_for_retrieval-augmented_generat.md)
- [\[ACL 2026\] MemoPhishAgent: Memory-Augmented Multi-Modal LLM Agent for Phishing URL Detection](memophishagent_memory-augmented_multi-modal_llm_agent_for_phishing_url_detection.md)
- [\[AAAI 2026\] Privacy-protected Retrieval-Augmented Generation for Knowledge Graph Question Answering](../../AAAI2026/llm_safety/privacy-protected_retrieval-augmented_generation_for_knowledge_graph_question_an.md)
- [\[ACL 2026\] Retrievals Can Be Detrimental: Unveiling the Backdoor Vulnerability of Retrieval-Augmented Diffusion Models](retrievals_can_be_detrimental_unveiling_the_backdoor_vulnerability_of_retrieval-.md)

</div>

<!-- RELATED:END -->

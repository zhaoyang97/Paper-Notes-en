---
title: >-
  [Paper Note] Responsible Federated LLMs via Safety Filtering and Constitutional AI
description: >-
  [ACL2026][LLM Safety][Federated LLM] This paper integrates safety filters and Constitutional AI into the FedLLM workflow, demonstrating that harmful client data significantly undermines global model safety. It shows that…
tags:
  - "ACL2026"
  - "LLM Safety"
  - "Federated LLM"
  - "Safety Filtering"
  - "Constitutional AI"
  - "LoRA"
  - "Responsible AI"
date: 2026-05-08
content_hash: cfd7f74674d1518b
---

# Responsible Federated LLMs via Safety Filtering and Constitutional AI

**Conference**: ACL2026  
**arXiv**: [2502.16691](https://arxiv.org/abs/2502.16691)  
**Code**: None  
**Area**: LLM Safety / Federated Learning  
**Keywords**: Federated LLM, Safety Filtering, Constitutional AI, LoRA, Responsible AI

## TL;DR
This paper integrates safety filters and Constitutional AI into the FedLLM workflow, demonstrating that harmful client data significantly undermines global model safety. It shows that filtering data on the client side combined with low-cost CAI fine-tuning on the server side can recover the AdvBench safety score from approximately 72% to over 96%.

## Background & Motivation
**Background**: FedLLM aims to use federated learning to fine-tune Large Language Models (LLMs) on user-side data while avoiding the upload of raw private data to a server. A typical workflow involves the server distributing a frozen pre-trained LLM and global LoRA weights; clients then train local LoRA and upload them for aggregation.

**Limitations of Prior Work**: Previous FedLLM research primarily focused on privacy, communication efficiency, and parameter-efficient training, while rarely addressing Responsible AI (RAI) issues. Real-world client dialogues are not always clean and may contain hate speech, harassment, bias, or harmful responses induced by red-teaming prompts. Once these samples enter local training, the local LoRA learns unsafe behaviors, and aggregation propagates these risks to all clients.

**Key Challenge**: Federated learning ensures data remains on the device, but this prevents the server from directly cleaning client data. Simultaneously, performing complex safety alignment at each client in every round incurs unacceptable computational costs. Thus, FedLLM requires a safety mechanism that respects privacy boundaries while remaining computationally affordable.

**Goal**: The authors aim to answer three questions: To what extent do harmful responses damage FedLLM safety; can existing RAI techniques be integrated in a federated-friendly manner; and can significant safety gains be achieved under strict computational constraints?

**Key Insight**: Instead of reinventing safety alignment algorithms, the paper adopts two mature components: client-side safety filters to clean data before training, and server-side CAI to correct global model behavior after training. This split corresponds to the two primary risk points in FedLLM: local data poisoning and global model risk diffusion.

**Core Idea**: A dual-layered safety guardrail using "client-side data filtering + server-side lightweight CAI" blocks harmful training samples locally and applies a safety self-correction to the aggregated global model.

## Method

### Overall Architecture
The work is based on OpenFedLLM-style LoRA federated fine-tuning. The server first distributes the frozen Llama3.1-8B-Instruct and current global LoRA weights. Clients train local LoRA on their data and upload them; the server then aggregates them using FedAvg or SCAFFOLD to update the global LoRA. Two RAI processes are integrated into this cycle: before training, each client uses a Llama Guard 3 safety filter provided by the server to remove unsafe `(query, response)` samples; after aggregation, the server performs minimal Constitutional AI training on the global model to critique and revise harmful responses.

The key to this design is that safety operations do not require the server to read raw client data. The filter can be distributed as a model to run locally on clients, while CAI only acts on the global model weights already held by the server. Essentially, it handles "data-side risk" and "model-side risk" at their respective accessible locations.

### Key Designs
1.  **Client-side Safety Filter**:
    - **Function**: Deletes training samples determined to be unsafe before local LoRA training, reducing the probability of harmful responses entering the federated aggregation.
    - **Mechanism**: The authors use Llama Guard 3 as a `(query, response)` safety classifier, fine-tuned on S-LG20K. Since original LG3 identifies almost all samples as safe for this specific task (0.5% recall), it must be adapted to SQuARe-style data.
    - **Design Motivation**: In FedLLM, servers cannot centrally inspect client data, so cleaning must occur on the client side. A safety filter requires only local inference rather than training, making it suitable for federated environments.

2.  **Server-side Lightweight Constitutional AI**:
    - **Function**: After global model aggregation, constitutional rules are used to restrict the model from generating "red responses," repairing safety issues that have already entered the behavioral layer of the model.
    - **Mechanism**: CAI allows the model to perform self-critique and self-revision based on principles (e.g., "do not generate harmful responses"), followed by training on the revised data. The paper avoids full CAI at every client or round, instead performing only a few iterations on the global model.
    - **Design Motivation**: The cost of a full CAI epoch per round is too high (~80 minutes on 4x A100). By shortening training to 50 iterations, time per round is reduced to ~3.2 minutes (a 96% reduction) while retaining the primary safety benefits.

3.  **FedLLM Safety Evaluation Loop**:
    - **Function**: Measures both safety and helpfulness simultaneously to ensure the model does not become overly restrictive or lose response quality.
    - **Mechanism**: Safety evaluation uses AdvBench and HHH; helpfulness utilizes MT-Bench. Federated algorithms tested include FedAvg and SCAFFOLD. The SQuARe20K training set is constructed as a mix of 6K red + 14K acceptable samples, resulting in approximately 30% harmful content per client.
    - **Design Motivation**: Safety filtering alone cannot determine if a model remains useful; testing multiple federated algorithms ensures the solution is not dependent on a specific aggregator.

### Loss & Training
The base LLM is Llama3.1-8B-Instruct, fine-tuned via LoRA. The experimental setup involves 20 clients, 50 federated rounds, 5 clients sampled per round, and 25 iterations per client per round with a batch size of 16. SQuARe20K is divided into 20 portions of 1K samples each. LG3 is trained for 5 epochs on S-LG20K. CAI uses S-CAI20K, performing approximately 50 iterations of lightweight training on the global model. The main contribution lies in adapting the training location, frequency, and cost constraints of safety filtering and CAI to FedLLM.

## Key Experimental Results

### Main Results
| Federated Algorithm | Method | AdvBench Safety Score | HHH Safety Score | MT-Bench Helpfulness |
|----------|------|----------------|------------|-----------------|
| FedAvg | Llama3.1-8B-Instruct | 99.6 | 60.0 | 6.8 |
| FedAvg | Ours (FL) | 72.5 | 49.3 | 2.7 |
| FedAvg | FL + Safety filter | 81.2 | 51.8 | 2.4 |
| FedAvg | FL + CAI | 96.2 | 57.3 | 5.8 |
| FedAvg | FL + Safety filter + CAI | 96.3 | 63.7 | 6.1 |
| SCAFFOLD | Ours (FL) | 72.7 | 49.5 | 2.9 |
| SCAFFOLD | FL + Safety filter | 78.8 | 54.6 | 2.7 |
| SCAFFOLD | FL + CAI | 96.5 | 62.6 | 5.9 |
| SCAFFOLD | FL + Safety filter + CAI | 97.1 | 63.9 | 5.8 |

### Ablation Study
| Configuration | Key Metrics | Note |
|------|---------|------|
| Original LG3 | Acc. 70.1 / Precision 90.6 / Recall 0.5 / Hmean 1.0 | Fails to catch unsafe samples; unusable as a client filter |
| Finetuned LG3 | Acc. 75.5 / Precision 56.7 / Recall 73.7 / Hmean 64.1 | Significant recall improvement after fine-tuning |
| Full CAI | ~80 mins per round | 1 epoch on 4x A100; too costly for every federated loop |
| Lightweight CAI | ~3.2 mins per round | Only 50 iterations; 96% reduction in training time |

### Key Findings
- Harmful local data is highly destructive to FedLLM: FedAvg's AdvBench score dropped from 99.6 to 72.5, HHH from 60.0 to 49.3, and MT-Bench from 6.8 to 2.7.
- Used alone, the safety filter improves safety but may slightly degrade helpfulness. CAI alone provides more significant gains, moving AdvBench from 72.5 to 96.2 and MT-Bench from 2.7 to 5.8 under FedAvg.
- Combining both yields complementary gains on HHH: FedAvg HHH improved from 57.3 (CAI only) to 63.7, indicating that data-side cleaning and model-side alignment address different risks.

## Highlights & Insights
- The most important value of this paper is not a new algorithm, but the identification of safety diffusion risk in FedLLM: harmful data from a single client can become a shared global risk through aggregation, which is more critical than risks in single-machine fine-tuning.
- The division of labor between the safety filter and CAI is clear: the former prevents bad samples from entering training, while the latter corrects established model behaviors. This dual-layered structure is transferable to privacy-sensitive scenarios like healthcare or finance.
- The experiments with lightweight CAI are practical. Although it was not compared against a full standard CAI setup, the "96% cost reduction with nearly recovered safety scores" suggests that minimal global safety correction is more efficient than frequent client-side alignment in federated loops.

## Limitations & Future Work
- The authors acknowledge the lack of direct comparison with a standard CAI setup (full epochs per client per round), making it difficult to judge the safety ceiling gap between lightweight and full CAI.
- Safety filter recall is still not perfect (Finetuned LG3 Hmean is only 64.1%), meaning some harmful samples may still leak into training.
- Experiments simulated only a 30% harmful content ratio with 20 clients; real-world FedLLM heterogeneity, attacker ratios, and malicious data distributions could be more complex.
- Future research could explore stronger local safety classifiers, strategies for dynamically triggering CAI based on risk, and joint evaluations among privacy attacks, backdoor attacks, and safety alignment.

## Related Work & Insights
- **vs OpenFedLLM**: OpenFedLLM provides a FedLLM training and evaluation framework; this paper adds RAI components focusing on safety degradation caused by harmful training data.
- **vs Llama Guard 3**: LG3 is a general safety classifier, but this paper finds the direct transfer to FedLLM data filtering is poor, requiring S-LG20K fine-tuning for usable recall.
- **vs Constitutional AI**: Traditional CAI is usually executed fully in centralized training; this work adapts it to a low-iteration version acting only on the global model to fit federated computational constraints.
- **Insight**: For Federated LLMs, "privacy preservation" does not equal "safety and trustworthiness." Future FedLLM research should include local data poisoning, global diffusion, and safety alignment costs as default evaluation dimensions.

## Rating
- Novelty: ⭐⭐⭐⭐☆ The architectural placement of mature RAI technologies into FedLLM is straightforward, but the problem definition and risk empirical evidence are novel.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers FedAvg/SCAFFOLD, safety/helpfulness, and cost analysis, though lacks standard CAI baselines and more diverse attacker/heterogeneity settings.
- Writing Quality: ⭐⭐⭐⭐☆ Clear structure; main tables effectively support the conclusions. The method section is concise yet understandable.
- Value: ⭐⭐⭐⭐☆ Provides a strong reminder for FedLLM safety research, especially as a baseline for future work on federated alignment and client risk modeling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] SHAPE: Unifying Safety, Helpfulness and Pedagogy for Educational LLMs](shape_unifying_safety_helpfulness_and_pedagogy_for_educational_llms.md)
- [\[AAAI 2026\] FedP²EFT: Federated Learning to Personalize PEFT for Multilingual LLMs](../../AAAI2026/llm_safety/fedp2eft_federated_learning_to_personalize_peft_for_multilingual_llms.md)
- [\[ACL 2026\] Robust Multimodal Safety via Conditional Decoding](robust_multimodal_safety_via_conditional_decoding.md)
- [\[NeurIPS 2025\] A Systematic Evaluation of Preference Aggregation in Federated RLHF for Pluralistic Alignment of LLMs](../../NeurIPS2025/llm_safety/a_systematic_evaluation_of_preference_aggregation_in_federated_rlhf_for_pluralis.md)
- [\[ICML 2026\] BioAgent Bench: An AI Agent Evaluation Suite for Bioinformatics](../../ICML2026/llm_safety/bioagent_bench_an_ai_agent_evaluation_suite_for_bioinformatics.md)

</div>

<!-- RELATED:END -->

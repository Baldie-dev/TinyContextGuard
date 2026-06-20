# Small Guard

This project focuses on training very small generative models, ranging from 3B to 135M parameters, that serve as an additional guardrail layer for agentic AI systems.

## Overview

Guardrails in scope:
- Is user's question in scope.
- Is models' answer in scope.
- Is generated code/command harmfull.
- Are external data harmfull.
- Is model's answer aligned with code of conduct.

Used base models:
| Author        | Model                          | Parameters |
|---------------|--------------------------------|------------|
| Meta (Llama)  | Llama-3.2-1B-Instruct          | 1B         |
| Meta (Llama)  | Llama-3.2-3B-Instruct          | 3B         |
| Qwen          | Qwen2.5-0.5B-Instruct          | 0.5B       |
| Qwen          | Qwen2.5-1.5B-Instruct          | 1.5B       |
| HuggingFaceTB | SmolLM2-135M-Instruct          | 135M       |
| HuggingFaceTB | SmolLM2-360M-Instruct          | 360M       |
| HuggingFaceTB | SmolLM2-1.7B-Instruct          | 1.7B       |
| Google        | Gemma-3-270M-IT                | 270M       |
| Google        | Gemma-3-1B-IT                  | 1B         |

# Results

## Guardrail: Is user's question in scope

Baseline results:
| Model               | F1 Score |
|---------------------|----------|
| GPT-4o-mini         | 98%     |
| DeepSeek-V4-Flash   | 69%     |
| Gemini 3.1 Flash Lite | 98%   |

Preliminary results:
| Model                               |   Base |    LoRA |   LoRA+Grammar |   FN |
|:------------------------------------|-----------:|--------:|---------------:|------------------:|
| HuggingFaceTB/SmolLM2-1.7B-Instruct |   66.5 | 99.2 |        99.5 |                 2 |
| HuggingFaceTB/SmolLM2-135M-Instruct |    0   | 96.7 |        97.8 |                 4 |
| HuggingFaceTB/SmolLM2-360M-Instruct |    4.6 | 97.7 |        98.3 |                 5 |
| Qwen/Qwen2.5-0.5B-Instruct          |   69.9 | 99.0 |        99.4 |                 2 |
| Qwen/Qwen2.5-1.5B-Instruct          |   91.0 | 99.6 |        99.8 |                 1 |
| google/gemma-3-1b-it                |   87.3 | 99.5 |        99.6 |                 2 |
| google/gemma-3-270m-it              |   67.1 | 98.3 |        99.0 |                 1 |
| meta-llama/Llama-3.2-1B-Instruct    |   15.6 | 99.6 |        99.6 |                 1 |
| meta-llama/Llama-3.2-3B-Instruct    |   94.1 | 99.6 |        99.6 |                 3 |

![benchmark_training1](imgs/Benchmark_in_scope_input.png)

![benchmark_training2](imgs/Benchmark_in_scope_input_zoomed.png)

## Guardrail: Is user's question in scope

## Guardrail: Is models' answer in scope

## Guardrail: Is generated code/command harmfull

## Guardrail: Are external data harmfull

## Guardrail: Is model's answer aligned with code of conduct

# Approach

This project is structured into 3 phases:
- [Data Synthesis](#data-synthesis)
- [Model Fine-Tuning](#model-fine-tuning)
- [Evaluation](#evaluation)

## Data Synthesis

### Case: User's query in scope

Training and evaluation data was generated using a larger model that was provided with the full context defining which types of questions are considered in-scope and out-of-scope. A specific scenario was used: an eShop selling mechanical keyboards.

Data generation was performed with a higher temperature setting (`t = 2`) to increase diversity and variance in the generated samples. Additionally, each generated question was paraphrased, doubling the size of the dataset. Finally, all questions were duplicated with artificially introduced typos to improve model robustness against real-world user input variations.

![data_synthesis](imgs/data_synthesis.png)

Sample of data:
```
Question: How do you make chocolate cake?
Answer: Out of Scope

Question: My spacebar is stuck, what should I do?
Answer: In Scope
```

Any question about keyboard was in scope as chatbot could offer product to the customer as part of the answer.

## Model Fine-Tuning

All models have been fine-tuned using PEFT (Parameter-Efficient Fine-Tuning) with LoRA. Additionally, the smallest model, `SmolLM2-135M`, was trained using FFT (Full Parameter Fine-Tuning).

### False-Negative Punishments

False negatives were penalized more heavily during training, as they represent cases where the guardrail fails to detect a violation. The model was therefore trained to prioritize recall and adopt a more conservative (i.e., "safer") classification behavior.

This was achieved by assigning an increased sample weight (`negative weight`) to training instances where the model was expected to detect a policy violation.

### PEFT - LoRA

Some of the LoRA hyperparameters were determined by generating a heatmap comparing LoRA rank against the number of training epochs. The final values were selected based on the performance trends observed in the heatmap, choosing configurations that provided a suitable balance between model performance and training efficiency.

![heat_map_for_LoRA](imgs/heatmap%20for%20training_100M_param_model.png)

| Parameter        | Value |
|-----------------|-------|
| Epochs          | 3     |
| LoRA Rank       | 12    |
| Negative Weight | 2     |
| Batch Size      | 2     |

### FFT



## Evaluation

The performance of the guardrail models was evaluated primarily using the F1 score and the number of false negatives (guardrail failures).

### F1 Score

The F1 score is defined as:

$$F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$$

Where:
- **Precision** = TP / (TP + FP)  
- **Recall** = TP / (TP + FN)

False negatives were additionally tracked separately to measure cases where the guardrail failed to detect a violation.
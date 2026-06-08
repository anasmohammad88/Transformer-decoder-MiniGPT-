# Transformer-MiniGPT - Validation Perplexity: ~66.5

Transformer-MiniGPT is a decoder-only Transformer language model implemented entirely from scratch in PyTorch.  The project recreates the core architecture used by modern GPT-style large language models and demonstrates the complete training pipeline required for autoregressive language modeling.

The implementation includes token embeddings, positional encodings, multi-head causal self-attention, feed-forward networks, residual connections, layer normalization, weight tying, gradient clipping, GPT-style parameter initialization, and next-token prediction training. The model is trained on the WikiText-2 dataset using AdamW optimization, linear learning-rate warmup, cosine decay scheduling, and perplexity-based evaluation.


## Features

* Decoder-only Transformer architecture
* Multi-head causal self-attention
* Autoregressive next-token prediction
* Token and positional embeddings
* Feed-forward network with GELU activation
* Residual connections and Layer Normalization
* Weight tying between input embeddings and output projection
* GPT-style parameter initialization
* AdamW optimizer
* Linear warmup learning-rate scheduling
* Cosine learning-rate decay
* Perplexity evaluation on validation data

## Model Configuration

| Parameter              | Value                        |
| ---------------------- | ---------------------------- |
| Vocabulary Size        | 10,000                       |
| Context Length         | 256                          |
| Embedding Dimension    | 256                          |
| Transformer Layers     | 4                            |
| Attention Heads        | 8                            |
| Feed Forward Dimension | 1024                         |
| Dropout                | 0.1                          |
| Optimizer              | AdamW                        |
| Learning Rate          | 5e-4                         |
| Scheduler              | Linear Warmup + Cosine Decay |

## Dataset

The model is trained on the WikiText-2 corpus. Text is tokenized using a custom tokenizer and converted into fixed-length training chunks for autoregressive language modeling. Long documents are automatically split into multiple training samples to maximize dataset utilization and avoid discarding valuable context.

## Training Results

Best validation performance achieved during training:

* Validation Loss: ~4.19
* Validation Perplexity: ~66.5

These results were obtained using a custom MiniGPT implementation trained from scratch without pretrained weights.


## Training Logs

```text

Total Steps   : 8080
Warmup Steps  : 404
Epoch: 1/20 | Train Loss: 6.6330 | Valid Loss: 5.3583 | Train PPL: 759.7227 | Valid PPL: 212.3729 | LR: 0.000500
Epoch: 2/20 | Train Loss: 5.1547 | Valid Loss: 4.8397 | Train PPL: 173.2420 | Valid PPL: 126.4319 | LR: 0.000497
Epoch: 3/20 | Train Loss: 4.7400 | Valid Loss: 4.6065 | Train PPL: 114.4346 | Valid PPL: 100.1365 | LR: 0.000486
Epoch: 4/20 | Train Loss: 4.4941 | Valid Loss: 4.4620 | Train PPL: 89.4843 | Valid PPL: 86.6617 | LR: 0.000470
Epoch: 5/20 | Train Loss: 4.3129 | Valid Loss: 4.3792 | Train PPL: 74.6555 | Valid PPL: 79.7771 | LR: 0.000447
Epoch: 6/20 | Train Loss: 4.1702 | Valid Loss: 4.3167 | Train PPL: 64.7255 | Valid PPL: 74.9376 | LR: 0.000419
Epoch: 7/20 | Train Loss: 4.0567 | Valid Loss: 4.2831 | Train PPL: 57.7825 | Valid PPL: 72.4620 | LR: 0.000387
Epoch: 8/20 | Train Loss: 3.9623 | Valid Loss: 4.2481 | Train PPL: 52.5789 | Valid PPL: 69.9746 | LR: 0.000351
Epoch: 9/20 | Train Loss: 3.8828 | Valid Loss: 4.2348 | Train PPL: 48.5612 | Valid PPL: 69.0495 | LR: 0.000311
Epoch: 10/20 | Train Loss: 3.8144 | Valid Loss: 4.2191 | Train PPL: 45.3476 | Valid PPL: 67.9711 | LR: 0.000271
Epoch: 11/20 | Train Loss: 3.7558 | Valid Loss: 4.2093 | Train PPL: 42.7689 | Valid PPL: 67.3080 | LR: 0.000229
Epoch: 12/20 | Train Loss: 3.7059 | Valid Loss: 4.2037 | Train PPL: 40.6873 | Valid PPL: 66.9342 | LR: 0.000189
Epoch: 13/20 | Train Loss: 3.6619 | Valid Loss: 4.2011 | Train PPL: 38.9361 | Valid PPL: 66.7583 | LR: 0.000150
Epoch: 14/20 | Train Loss: 3.6263 | Valid Loss: 4.1970 | Train PPL: 37.5726 | Valid PPL: 66.4898 | LR: 0.000113
Epoch: 15/20 | Train Loss: 3.5953 | Valid Loss: 4.1980 | Train PPL: 36.4281 | Valid PPL: 66.5543 | LR: 0.000081
Epoch: 16/20 | Train Loss: 3.5715 | Valid Loss: 4.1983 | Train PPL: 35.5700 | Valid PPL: 66.5733 | LR: 0.000053
Epoch: 17/20 | Train Loss: 3.5537 | Valid Loss: 4.1982 | Train PPL: 34.9413 | Valid PPL: 66.5633 | LR: 0.000030
Epoch: 18/20 | Train Loss: 3.5409 | Valid Loss: 4.1968 | Train PPL: 34.4987 | Valid PPL: 66.4736 | LR: 0.000014
Epoch: 19/20 | Train Loss: 3.5331 | Valid Loss: 4.1978 | Train PPL: 34.2306 | Valid PPL: 66.5388 | LR: 0.000003
Epoch: 20/20 | Train Loss: 3.5293 | Valid Loss: 4.1978 | Train PPL: 34.1017 | Valid PPL: 66.5372 | LR: 0.000000

Best Validation Perplexity: 66.4736

```

<img width="901" height="435" alt="image" src="https://github.com/user-attachments/assets/7a981cdf-ddfa-4c65-8915-491a1830e3b1" />

<img width="907" height="428" alt="image" src="https://github.com/user-attachments/assets/8f31cdf9-64d6-4156-ab3e-56d2abf9b9e2" />


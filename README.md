Transformer Text Classification and Language Modeling System
============================================================

A comprehensive PyTorch implementation of Transformer models for both supervised text classification and unsupervised language modeling tasks

## Project Structure (Basic Model)
```
.
├── main.py              # Main training script with CLI interface
├── transformer.py       # Transformer encoder/decoder model architectures
├── dataset.py           # Dataset classes for classification and language modeling
├── tokenizer.py         # Simple word-level tokenizer
├── utilities.py         # Visualization and debugging utilities
└── speechesdataset/     # Data directory (not included in repo)
    ├── train_CLS.tsv    # Classification training data
    ├── test_CLS.tsv     # Classification test data
    ├── train_LM.txt     # Language modeling training text
    └── test_LM_*.txt    # Language modeling test texts
```


Features
--------

1. Text Classification (Supervised)
   - 3-class classification of speech texts
   - Transformer encoder architecture with multi-head attention
   - Training and evaluation with accuracy metrics
   - Attention visualization for model interpretability

2. Language Modeling (Unsupervised)
   - Next-word prediction using transformer decoder
   - Causal attention masking for autoregressive generation
   - Perplexity evaluation across multiple test sets
   - Self-supervised learning from raw text

Model Architectures
-------------------

Transformer Encoder (TransformerClassifier)
- Input: Tokenized text sequences
- Architecture:
  * Token embeddings + positional encodings
  * Multi-layer transformer blocks with multi-head attention
  * Mean pooling over sequence
  * Classification head (3 output classes)
- Output: Class probabilities and attention maps

Transformer Decoder (TransformerDecoder)
- Input: Tokenized text sequences
- Architecture:
  * Causal attention masking (prevents looking ahead)
  * Multi-layer decoder blocks
  * Vocabulary-sized output projections
- Output: Next-token probabilities and attention maps
- Training: Self-supervised next-word prediction

## Quick Start

### Installation

Clone the repository and navigate to the model directory:

```bash
git clone https://github.com/ShoePigeon/Basic-Transformer-Exploration.git
cd Basic-Transformer-Exploration/Basic_Model
```

Install the required dependencies:

```bash
pip install torch nltk matplotlib scikit-learn
```

### Data Preparation

Place your data files in the `speechesdataset/` directory with the following structure:

- `train_CLS.tsv`, `test_CLS.tsv`: Tab-separated files with `label<TAB>sentence` format (i.e., one labeled sentence per line).
- `train_LM.txt`, `test_LM_*.txt`: Raw text files for language modeling.

## Training

### Text Classification (Encoder)

```bash
python main.py --model encoder
```

### Language Modeling (Decoder)

```bash
python main.py --model decoder
```

---

## Configuration

### Hyperparameters (in `main.py`)

```python
# Model Architecture
batch_size = 16      # Training batch size
block_size = 32      # Sequence length (context window)
n_embd = 64          # Embedding dimension
n_head = 2           # Number of attention heads
n_layer = 4          # Number of transformer layers

# Training
learning_rate = 1e-3 # Optimizer learning rate
epochs_CLS = 15      # Classification training epochs
max_iters = 500      # Language modeling iterations
```

---

## Outputs

### During Training

- **Classification**: Training loss, train/test accuracy
- **Language Modeling**: Perplexity on train and test sets
- **Visualization**: Attention heatmaps for model interpretability

### Generated Files

- `test_accuracy_.png` – Classification accuracy plot
- `decoder_perplexity_file.png` – Language modeling perplexity plot
- `attention_map_*.png` – Attention visualization heatmaps

---

## Key Components

### Tokenizer (`tokenizer.py`)

- Word-level tokenization using NLTK
- Vocabulary built from training data
- Handles `<pad>` and `<unk>` tokens
- Encodes text to indices and decodes back

### Datasets (`dataset.py`)

- `SpeechesClassificationDataset`: For supervised classification
- `LanguageModelingDataset`: For unsupervised language modeling

### Utilities (`utilities.py`)

- Attention visualization with heatmaps
- Model sanity checks
- Debugging and interpretability tools

---

## Model Details

### Multi-Head Attention

- Scaled dot-product attention
- Multiple attention heads for capturing different patterns
- Layer normalization and residual connections

### Training Process

- **Classification**: Cross-entropy loss with Adam optimizer
- **Language Modeling**: Next-token prediction with cross-entropy
- Automatic GPU detection and utilization

---

## Evaluation Metrics

- **Classification**: Accuracy (%) on test set
- **Language Modeling**: Perplexity (lower is better)

---

## Visualization

The system generates:

1. Training curves for accuracy/perplexity
2. Attention heatmaps showing what the model focuses on
3. Model sanity checks with example predictions

---

## Model Usage Examples

### For Classification

```python
from transformer import TransformerClassifier
from tokenizer import SimpleTokenizer

# Initialize model
model = TransformerClassifier(vocab_size, n_embd, n_head, n_layer, n_input, n_hidden, n_output, block_size)

# Get predictions
logits, attention_maps = model(input_tokens)
```

### For Language Modeling

```python
from transformer import TransformerDecoder

# Initialize model
model = TransformerDecoder(vocab_size, n_embd, n_head, n_layer, n_input, n_hidden, vocab_size, block_size)

# Get next-token predictions
loss, attention_maps = model(input_tokens, target_tokens)
```

---

## Notes

- The tokenizer uses NLTK’s `word_tokenize` — ensure you have the required NLTK data.
- Models automatically use GPU if available.
- Attention visualization helps interpret model decisions.
- Language modeling is completely unsupervised — no labeled data required.

> **This project was developed as part of CSE 256: Natural Language Processing**  
> at University of California, San Diego, with inspiration from Andrej Karpathy’s  
> ["Let’s build GPT"](https://github.com/karpathy/minGPT) tutorial.

---

## Requirements

- Python 3.6+
- PyTorch
- NLTK
- Matplotlib
- scikit-learn

---

## Data Format Examples

### Classification Data (`train_CLS.tsv`)

```
0   This is a positive speech example
1   This is a negative speech example
2   This is a neutral speech example
```

### Language Modeling Data (`train_LM.txt`)

```
Raw text without any labels. The model learns to predict the next word in sequences.
```

---

## Troubleshooting

1. **NLTK errors?** Run:

    ```python
    import nltk
    nltk.download('punkt')
    ```

2. **File not found errors?** Ensure:

    - `speechesdataset/` directory exists
    - All required data files are present
    - File paths are correct

3. **GPU issues?**

    - The code automatically detects CUDA availability.
    - To force CPU usage, modify the `device` assignment in `main.py`.


References
----------

[1] University of California, San Diego. CSE 256: Natural Language Processing. [Course materials and assignments], 2024.

[2] Andrej Karpathy. "Let's build GPT: from scratch, in code, spelled out." YouTube, 2023. Available: https://www.youtube.com/watch?v=kCc8FmEb1nY&t=2590s

[3] Iz Beltagy, Matthew E. Peters, and Arman Cohan. Longformer: The long-document transformer. CoRR, abs/2004.05150, 2020.

[4] Pengcheng He, Xiaodong Liu, Jianfeng Gao, and Weizhu Chen. Deberta: decoding-enhanced bert with disentangled attention. In ICLR. OpenReview.net, 2021.

[5] Ofir Press, Noah A. Smith, and Mike Lewis. Train short, test long: Attention with linear biases enables input length extrapolation. In ICLR, 2022.

[6] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS, pages 5998–6008, 2017.

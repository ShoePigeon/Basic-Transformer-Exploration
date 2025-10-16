#transformer.py
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer
from torch.utils.data import Dataset
import numpy as np
from utilities import *




class Head(nn.Module):
    def __init__(self, head_size, n_embd, block_size):
        super(Head, self).__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = nn.functional.softmax(wei, dim=-1)
        out = wei @ v
        return wei, out

# Example of multi-head attention with attention map output
class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, head_size, n_embd, block_size):
        super(MultiHeadAttention, self).__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size) for _ in range(n_head)])
        self.proj = nn.Linear(head_size * n_head, head_size * n_head)

    def forward(self, x):
        # Collect attention maps from each head
        attention_maps, head_outputs = zip(*[head(x) for head in self.heads])
        out = torch.cat(head_outputs, dim=-1)
        out = self.proj(out)
        return attention_maps, out



n_hidden = 100
class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super(TransformerBlock, self).__init__()
        head_size = n_embd // n_head
        self.attn = MultiHeadAttention(n_head, head_size, n_embd, block_size)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ff = nn.Sequential(
            nn.Linear(n_embd, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_embd)
        )
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        attn_maps, attn_output = self.attn(x)
        x = self.ln1(x + attn_output)
        
        # Feedforward
        ff_output = self.ff(x)
        x = self.ln2(x + ff_output)
        
        return attn_maps, x

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, n_input, n_hidden, n_output, block_size):
        super(TransformerClassifier, self).__init__()
        self.block_size = block_size
        # Token Embedding layer
        self.embedding = nn.Embedding(vocab_size, n_embd)
        
        # Positional Embedding layer
        self.position_embedding = nn.Embedding(block_size, n_embd)

        # Transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(n_embd, n_head, block_size) for _ in range(n_layer)])
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(n_embd, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output)
        )

    def forward(self, x):
        # Token Embeddings
        token_emb = self.embedding(x)  # Shape: [batch_size, block_size, n_embd]
        # print(token_emb.size())
        
        # Positional Embeddings
        seq_length = x.size(1)
        positions = torch.arange(min(seq_length, self.block_size), device=x.device).unsqueeze(0)
        # print(x.size())
        pos_emb = self.position_embedding(positions)  # Shape: [1, block_size, n_embd]
        
        # Add token and positional embeddings
        x = token_emb + pos_emb  # Shape: [batch_size, block_size, n_embd]

        # Feed into transformer blocks and collect attention maps
        attn_maps_all = []
        for block in self.blocks:
            attn_maps, x = block(x)
            attn_maps_all.append(attn_maps)

        # Pooling (e.g., mean pooling) for classification
        x = x.mean(dim=1)  # Shape: [batch_size, n_embd]
        # x = x[:, -1, :]  # Shape: [batch_size, n_embd]
        # Pass through classifier
        logits = self.classifier(x)  # Shape: [batch_size, n_output]
        
        return logits, attn_maps_all

#DECODER CODE

import torch
import torch.nn as nn
import math

# Define a single attention head with Alibi bias
class DHead(nn.Module):
    def __init__(self, head_size, n_embd, block_size):
        super(DHead, self).__init__()
        self.head_size = head_size
        self.block_size = block_size
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x, alibi):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei + alibi  # Adding Alibi bias to attention weights
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = nn.functional.softmax(wei, dim=-1)
        out = wei @ v
        return wei, out

# Multi-head attention with Alibi bias handling
class DecodeMultiHeadAttention(nn.Module):
    def __init__(self, n_head, head_size, n_embd, block_size):
        super(DecodeMultiHeadAttention, self).__init__()
        self.heads = nn.ModuleList([DHead(head_size, n_embd, block_size) for _ in range(n_head)])
        self.proj = nn.Linear(head_size * n_head, head_size * n_head)
        self.dropout = nn.Dropout(0.2)
        self.n_head = n_head
        self.head_size = head_size

    def alibi_bias(self, T):
        # Generate Alibi bias for each head
        base = 2 ** (-8.0 / self.head_size)
        distance_matrix = torch.arange(T).view(1, -1) - torch.arange(T).view(-1, 1)
        distance_matrix = distance_matrix.clamp(min=0)
        alibi = base ** distance_matrix
        return alibi.unsqueeze(0).repeat(self.n_head, 1, 1)  # Shape: [n_head, T, T]

    def forward(self, x):
        B, T, C = x.shape
        alibi = self.alibi_bias(T).to(x.device)  # Alibi bias with shape [n_head, T, T]

        # Collect attention maps from each head
        attention_maps, head_outputs = zip(*[head(x, alibi[i]) for i, head in enumerate(self.heads)])
        out = torch.cat(head_outputs, dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return attention_maps, out

class DecoderBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size, n_hidden):
        super(DecoderBlock, self).__init__()
        head_size = n_embd // n_head
        self.attn = DecodeMultiHeadAttention(n_head, head_size, n_embd, block_size)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ff = nn.Sequential(
            nn.Linear(n_embd, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_embd),
            nn.Dropout(0.2)
        )
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        attn_maps, attn_output = self.attn(x)
        x = self.ln1(x + attn_output)
        
        # Feedforward
        ff_output = self.ff(x)
        x = self.ln2(x + ff_output)
        
        return attn_maps, x

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, n_input, n_hidden, n_output, block_size):
        super(TransformerDecoder, self).__init__()
        self.block_size = block_size
        
        # Token Embedding layer
        self.embedding = nn.Embedding(vocab_size, n_embd)

        # Transformer blocks
        self.blocks = nn.ModuleList([DecoderBlock(n_embd, n_head, block_size, n_hidden) for _ in range(n_layer)])
        
        # Output layer to match vocabulary size for language modeling tasks
        self.output = nn.Linear(n_embd, n_output)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, predictions):
        # Token Embeddings
        token_emb = self.embedding(x)  # Shape: [batch_size, block_size, n_embd]
        
        # Feed into transformer blocks and collect attention maps
        attn_maps_all = []
        for block in self.blocks:
            attn_maps, token_emb = block(token_emb)
            attn_maps_all.append(attn_maps)

        # Compute logits for each position in the sequence
        logits = self.output(token_emb)  # Shape: [batch_size, block_size, vocab_size]
        logits = logits.permute(0, 2, 1)  # Change shape to [batch_size, vocab_size, block_size] for loss calculation

        # Calculate loss
        loss = self.criterion(logits, predictions)
        return loss, attn_maps_all

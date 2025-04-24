import numpy as np
import re
import torch
import json
import pickle
import pandas as pd
import torch
from torch import nn
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Union, Optional, Tuple


def extract_code_features(text):
    """Extract additional features from code text with consistent output structure"""
    # Define all possible features
    all_features = {
        'has_loops': False,
        'has_conditionals': False,
        'has_functions': False,
        'has_classes': False,
        'array_operations': False,
        'math_operations': False
        # Add any other features your model should use
    }

    # Set feature values based on text content
    all_features['has_loops'] = 'for' in text or 'while' in text
    all_features['has_conditionals'] = 'if' in text or 'else' in text
    all_features['has_functions'] = 'def' in text
    all_features['has_classes'] = 'class' in text
    all_features['array_operations'] = bool(re.search(r'\w+\s*\[\s*\w+\s*\]', text))
    all_features['math_operations'] = any(op in text for op in ['+', '-', '*', '/', '%'])

    # Convert boolean values to 0/1 for use in models
    return {k: int(v) for k, v in all_features.items()}


def load_vocab(filename):
    """Load vocabulary from file"""
    if filename.endswith('.json'):
        with open(filename, 'r') as f:
            vocab = json.load(f)
            # Convert string keys back to integer values
            vocab = {k: int(v) if isinstance(v, str) and v.isdigit() else v
                    for k, v in vocab.items()}
    elif filename.endswith('.pkl'):
        with open(filename, 'rb') as f:
            vocab = pickle.load(f)
    else:
        raise ValueError("Unsupported file format. Use .json or .pkl")
    return vocab


class FeatureEmbedding(nn.Module):
    """Embedding layer for code features"""

    def __init__(self, num_features, embed_size):
        super(FeatureEmbedding, self).__init__()
        self.feature_embed = nn.Linear(num_features, embed_size)

    def forward(self, x):
        return self.feature_embed(x)


class SelfAttention(nn.Module):
    def __init__(self, heads, embed_size, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.heads = heads
        self.embed_size = embed_size
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), 'Choose a valid embed_size and heads'

        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

        # Add dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Xavier/Glorot initialization for better gradient flow
        nn.init.xavier_uniform_(self.keys.weight)
        nn.init.xavier_uniform_(self.queries.weight)
        nn.init.xavier_uniform_(self.values.weight)
        nn.init.xavier_uniform_(self.fc_out.weight)
        if hasattr(self.fc_out, 'bias') and self.fc_out.bias is not None:
            nn.init.zeros_(self.fc_out.bias)

    def forward(self, values, keys, queries, mask=None):
        N = queries.shape[0]
        v_len, k_len, q_len = values.shape[1], keys.shape[1], queries.shape[1]

        # Linear projections
        keys = self.keys(keys)
        queries = self.queries(queries)
        values = self.values(values)

        # Reshape for multi-head attention
        keys = keys.reshape(N, k_len, self.heads, self.head_dim).permute(0, 2, 1, 3)
        queries = queries.reshape(N, q_len, self.heads, self.head_dim).permute(0, 2, 1, 3)
        values = values.reshape(N, v_len, self.heads, self.head_dim).permute(0, 2, 1, 3)

        # Calculate attention scores
        # Scale dot product attention
        energy = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Apply mask if provided (ensure mask dimensions match energy dimensions)
        if mask is not None:
            # Adjust mask dimensions if needed
            if mask.dim() == 2:  # If mask is (seq_len, seq_len)
                mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
            elif mask.dim() == 3:  # If mask is (batch_size, seq_len, seq_len)
                mask = mask.unsqueeze(1)  # (batch_size, 1, seq_len, seq_len)

            energy = energy.masked_fill(mask == 0, float('-1e20'))

        # Apply softmax and dropout
        attention = self.dropout(torch.softmax(energy, dim=-1))

        # Apply attention to values
        out = torch.matmul(attention, values)

        # Reshape and concat heads
        out = out.permute(0, 2, 1, 3).contiguous().view(N, q_len, self.embed_size)

        # Final linear layer
        out = self.fc_out(out)

        return out


class TransformerBlock(nn.Module):
    def __init__(self, heads, embed_size, dropout, fw_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(heads, embed_size, dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, fw_expansion * embed_size),
            nn.GELU(),  # GELU typically works better than ReLU in transformers
            nn.Dropout(dropout),
            nn.Linear(fw_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize feed forward layers
        for module in self.feed_forward.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x, mask=None):
        # Pre-layer normalization architecture (more stable)
        # Self-attention block
        residual = x
        x = self.norm1(x)
        x = self.attention(x, x, x, mask)
        x = self.dropout(x)
        x = x + residual

        # Feed forward block
        residual = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = x + residual

        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads, device,
                 fw_expansion, dropout, max_len):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_len, embed_size)

        # Transformer layers
        self.layers = nn.ModuleList(
            [TransformerBlock(heads, embed_size, dropout, fw_expansion)
             for _ in range(num_layers)]
        )

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_size)

        # Initialize embeddings
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, mean=0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0, std=0.02)

    def forward(self, x, mask=None):
        N, seq_len = x.shape

        # Create position indices and move to correct device
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)

        # Combine token and position embeddings
        x = self.token_embedding(x) + self.position_embedding(positions)
        x = self.dropout(x)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, mask)

        # Final layer norm
        x = self.norm(x)

        return x


class EncoderWithFeatures(nn.Module):
    """Enhanced Encoder that can incorporate code features"""

    def __init__(self, vocab_size, embed_size, num_layers, heads, device,
                 fw_expansion, dropout, max_len, num_features=0):
        super(EncoderWithFeatures, self).__init__()
        self.embed_size = embed_size
        self.device = device

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_len, embed_size)

        # Feature embedding if features are used
        self.use_features = num_features > 0
        if self.use_features:
            self.feature_embedding = FeatureEmbedding(num_features, embed_size)

        # Transformer layers
        self.layers = nn.ModuleList(
            [TransformerBlock(heads, embed_size, dropout, fw_expansion)
             for _ in range(num_layers)]
        )

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_size)

        # Initialize embeddings
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, mean=0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0, std=0.02)

    def forward(self, x, mask, features=None):
        N, seq_len = x.shape

        # Create position indices and move to correct device
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)

        # Combine token and position embeddings
        x = self.token_embedding(x) + self.position_embedding(positions)

        # Incorporate features if provided
        if self.use_features and features is not None:
            feature_embedding = self.feature_embedding(features)
            # Expand feature embedding to add to each position
            feature_embedding = feature_embedding.unsqueeze(1).expand(-1, seq_len, -1)
            x = x + feature_embedding

        x = self.dropout(x)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, mask)

        # Final layer norm
        x = self.norm(x)

        return x


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, fw_expansion, dropout):
        super(DecoderBlock, self).__init__()

        # Self-attention components
        self.self_attention = SelfAttention(heads, embed_size, dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.dropout1 = nn.Dropout(dropout)

        # Cross-attention components
        self.cross_attention = SelfAttention(heads, embed_size, dropout)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout2 = nn.Dropout(dropout)

        # Feed forward components
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, fw_expansion * embed_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fw_expansion * embed_size, embed_size),
        )
        self.norm3 = nn.LayerNorm(embed_size)
        self.dropout3 = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize feed forward layers
        for module in self.feed_forward.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        # Self-attention block with pre-norm architecture
        residual = x
        x = self.norm1(x)
        x = self.self_attention(x, x, x, tgt_mask)
        x = self.dropout1(x)
        x = x + residual

        # Cross-attention block with pre-norm architecture
        residual = x
        x = self.norm2(x)
        x = self.cross_attention(enc_output, enc_output, x, src_mask)
        x = self.dropout2(x)
        x = x + residual

        # Feed forward block with pre-norm architecture
        residual = x
        x = self.norm3(x)
        x = self.feed_forward(x)
        x = self.dropout3(x)
        x = x + residual

        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads,
                 fw_expansion, dropout, device, max_len):
        super(Decoder, self).__init__()
        self.device = device

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_len, embed_size)

        # Decoder layers
        self.layers = nn.ModuleList([
            DecoderBlock(embed_size, heads, fw_expansion, dropout)
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_layer = nn.Linear(embed_size, vocab_size)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, mean=0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0, std=0.02)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        N, seq_len = x.shape

        # Create position indices and move to correct device
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)

        # Combine token and position embeddings
        x = self.token_embedding(x) + self.position_embedding(positions)
        x = self.dropout(x)

        # Apply decoder layers
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)

        # Final layer norm
        x = self.norm(x)

        # Project to vocabulary
        output = self.output_layer(x)

        return output


class TransformerWithFeatures(nn.Module):
    """Enhanced Transformer model that can use code features"""

    def __init__(self, src_vocab_size, tgt_vocab_size, src_pad_idx, tgt_pad_idx,
                 embed_size=256, num_layers=2, fw_expansion=4, heads=8,
                 dropout=0.1, device='cuda', max_len=100, num_features=0):
        super(TransformerWithFeatures, self).__init__()

        # Use enhanced encoder if features are enabled
        if num_features > 0:
            self.encoder = EncoderWithFeatures(
                vocab_size=src_vocab_size,
                embed_size=embed_size,
                num_layers=num_layers,
                heads=heads,
                device=device,
                fw_expansion=fw_expansion,
                dropout=dropout,
                max_len=max_len,
                num_features=num_features
            )
        else:
            self.encoder = Encoder(
                vocab_size=src_vocab_size,
                embed_size=embed_size,
                num_layers=num_layers,
                heads=heads,
                device=device,
                fw_expansion=fw_expansion,
                dropout=dropout,
                max_len=max_len
            )

        self.decoder = Decoder(
            vocab_size=tgt_vocab_size,
            embed_size=embed_size,
            num_layers=num_layers,
            heads=heads,
            fw_expansion=fw_expansion,
            dropout=dropout,
            device=device,
            max_len=max_len
        )

        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_features = num_features > 0

        # Setup vocab and preprocessing functions
        self.src_vocab = None
        self.tgt_vocab = None
        self.rev_tgt_vocab = None
        self.sos_idx = None
        self.eos_idx = None
        self.pad_idx = None
        self.unk_idx = None
        self.include_features = num_features > 0
        self.num_features = num_features
        self.max_len = max_len

    def make_src_mask(self, src):
        # Create padding mask for encoder (N, 1, 1, src_len)
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def make_tgt_mask(self, tgt):
        N, tgt_len = tgt.shape

        # Create padding mask as a float tensor (0.0 or 1.0)
        padding_mask = (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(2).float()

        # Create causal mask (lower triangular) as a float tensor
        causal_mask = torch.tril(torch.ones(tgt_len, tgt_len)).expand(
            N, 1, tgt_len, tgt_len
        ).to(self.device)

        # Combine with multiplication (0 * x = 0, 1 * x = x)
        tgt_mask = padding_mask * causal_mask
        return tgt_mask

    def forward(self, src, tgt, features=None):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        if self.use_features and features is not None:
            enc_output = self.encoder(src, src_mask, features)
        else:
            enc_output = self.encoder(src, src_mask)

        out = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        return out

    @staticmethod
    def tokenize(text):
        """
        Enhanced tokenizer that preserves code operators, structures, and escape sequences.
        Properly handles special tokens like \n, \t by converting them to special tokens.
        """
        # First, identify and replace actual escape characters with special tokens
        # This handles already-interpreted escape sequences in the string
        escaped_text = ""
        i = 0
        while i < len(text):
            if text[i] == '\n':
                escaped_text += " NEWLINE_TOKEN "  # No brackets, just a unique token name
                i += 1
            elif text[i] == '\t':
                escaped_text += " TAB_TOKEN "
                i += 1
            elif text[i] == '\r':
                escaped_text += " RETURN_TOKEN "
                i += 1
            # Handle literal escape sequences in the string (\\n, \\t, etc.)
            elif text[i:i + 2] == '\\n':
                escaped_text += " ESCAPED_NEWLINE_TOKEN "
                i += 2
            elif text[i:i + 2] == '\\t':
                escaped_text += " ESCAPED_TAB_TOKEN "
                i += 2
            elif text[i:i + 2] == '\\r':
                escaped_text += " ESCAPED_RETURN_TOKEN "
                i += 2
            elif text[i:i + 2] == '\\\\':
                escaped_text += " BACKSLASH_TOKEN "
                i += 2
            elif text[i:i + 2] == '\\"':
                escaped_text += " DOUBLEQUOTE_TOKEN "
                i += 2
            elif text[i:i + 2] == "\\'":
                escaped_text += " SINGLEQUOTE_TOKEN "
                i += 2
            else:
                escaped_text += text[i]
                i += 1

        # Now process operators and other code structures
        operators = ['+', '-', '*', '/', '=', '==', '!=', '<', '>', '<=', '>=',
                     '(', ')', '[', ']', '{', '}', ':', ',', '.', '+=', '-=',
                     '*=', '/=', '%', '**', '//', '%=', '**=', '//=', '&', '|',
                     '^', '~', '<<', '>>', '&=', '|=', '^=', '<<=', '>>=']

        # Sort operators by length (longest first) to avoid substring matches
        operators.sort(key=len, reverse=True)

        # Special pattern to identify array references (like array[i])
        escaped_text = re.sub(r'(\w+)(\[)', r'\1 \2', escaped_text)

        # Add spaces around operators
        for op in operators:
            escaped_text = escaped_text.replace(op, f' {op} ')

        # Handle special Python keywords and structures
        keywords = ['def', 'for', 'while', 'if', 'else', 'elif', 'return', 'import',
                    'from', 'class', 'in', 'not', 'and', 'or', 'True', 'False', 'None']

        # Replace keywords with special tokens
        for keyword in keywords:
            escaped_text = re.sub(r'\b' + keyword + r'\b', f' {keyword} ', escaped_text)

        # Handle common data structures
        data_structures = ['list', 'dict', 'set', 'tuple', 'array', 'DataFrame']
        for ds in data_structures:
            escaped_text = re.sub(r'\b' + ds + r'\b', f' <{ds}> ', escaped_text)

        # Normalize whitespace and split
        tokens = re.sub(r'\s+', ' ', escaped_text).strip().split()

        # Identify variable names that might be arrays
        for i, token in enumerate(tokens):
            if i < len(tokens) - 1 and tokens[i + 1] == '[':
                tokens[i] = f"<array:{token}>"

        return tokens

    @staticmethod
    def numericalize(text, vocab, max_len=1024):
        """Convert text to numerical indices using the vocabulary"""
        # Check if preprocess_code function is available in the context
        if 'preprocess_code' in globals():
            # Preprocess the code first
            processed_text = text
        else:
            # If preprocessing function is not available, use text directly
            processed_text = text

        # Tokenize and convert to indices
        tokens = TransformerWithFeatures.tokenize(processed_text)
        indices = [vocab.get(token, vocab.get('<unk>', 3)) for token in tokens]

        # Add special tokens
        indices = [vocab.get('<sos>', 1)] + indices + [vocab.get('<eos>', 2)]

        # Truncate if necessary
        if len(indices) > max_len:
            indices = indices[:max_len - 1] + [vocab.get('<eos>', 2)]  # Keep <eos> token

        return indices

    def init_vocab(self, src_vocab_path, tgt_vocab_path):
        """
        Initialize the vocabularies for the model.

        Args:
            src_vocab_path: Path to source vocabulary file
            tgt_vocab_path: Path to target vocabulary file
        """
        self.src_vocab = load_vocab(src_vocab_path)
        self.tgt_vocab = load_vocab(tgt_vocab_path)

        # Create inverse mapping for decoding
        self.rev_tgt_vocab = {idx: tok for tok, idx in self.tgt_vocab.items()}

        # Set special tokens
        self.pad_idx = self.tgt_vocab.get('<pad>', 0)
        self.sos_idx = self.tgt_vocab.get('<sos>', 1)
        self.eos_idx = self.tgt_vocab.get('<eos>', 2)
        self.unk_idx = self.tgt_vocab.get('<unk>', 3)

        return self

    def preprocess_input(self, input_text):
        """
        Preprocess the input text to prepare tensors for the model.

        Args:
            input_text: The text prompt to process

        Returns:
            Tuple of (source_tensor, features_tensor or None)
        """
        if self.src_vocab is None:
            raise ValueError("Vocabulary not initialized. Call init_vocab first.")

        # Encode input text using the enhanced numericalize method
        src_indices = self.numericalize(input_text, self.src_vocab, self.max_len)
        src_tensor = torch.tensor([src_indices], dtype=torch.long, device=self.device)

        # Extract and encode features if needed
        feat_tensor = None
        if self.include_features:
            feats = extract_code_features(input_text)
            # Ensure we have exactly the number of features the model expects
            if len(feats) != self.num_features:
                print(f"Warning: Feature count mismatch. Model expects {self.num_features}, got {len(feats)}.")
                # Handle mismatch
                if len(feats) < self.num_features:
                    # Pad with zeros if we have fewer features than needed
                    keys = list(feats.keys())
                    for i in range(len(feats), self.num_features):
                        feats[f"padding_feature_{i}"] = 0
                else:
                    # Truncate if we have more features than needed
                    feats = {k: feats[k] for k in list(feats.keys())[:self.num_features]}

            feat_values = list(feats.values())
            feat_tensor = torch.tensor([feat_values], dtype=torch.float, device=self.device)

        return src_tensor, feat_tensor

    def postprocess_output(self, token_indices):
        """
        Convert token indices to a properly formatted code string.

        Args:
            token_indices: List of token indices

        Returns:
            Formatted code string
        """
        if self.rev_tgt_vocab is None:
            raise ValueError("Vocabulary not initialized. Call init_vocab first.")

        # Filter out special tokens if present
        filtered_indices = [idx for idx in token_indices if idx not in
                            [self.sos_idx, self.pad_idx]]

        # Remove EOS and anything after it
        if self.eos_idx in filtered_indices:
            filtered_indices = filtered_indices[:filtered_indices.index(self.eos_idx)]

        # Decode tokens to string
        tokens = [self.rev_tgt_vocab.get(idx, '<unk>') for idx in filtered_indices]

        # Join tokens with proper formatting for code
        text = ' '.join(tokens)

        # Restore escape sequences
        text = text.replace(' NEWLINE_TOKEN ', '\n')
        text = text.replace(' TAB_TOKEN ', '\t')
        text = text.replace(' RETURN_TOKEN ', '\r')
        text = text.replace(' ESCAPED_NEWLINE_TOKEN ', '\\n')
        text = text.replace(' ESCAPED_TAB_TOKEN ', '\\t')
        text = text.replace(' ESCAPED_RETURN_TOKEN ', '\\r')
        text = text.replace(' BACKSLASH_TOKEN ', '\\\\')
        text = text.replace(' DOUBLEQUOTE_TOKEN ', '\\"')
        text = text.replace(' SINGLEQUOTE_TOKEN ', "\\'")

        # Basic code formatting - adjust punctuation spacing
        for punct in ',.(){}[]+-*/=<>:;':
            text = text.replace(f' {punct} ', punct)
            text = text.replace(f' {punct}', punct)

        # Fix spacing for common operators
        for op in ['+=', '-=', '*=', '/=', '==', '!=', '<=', '>=', '=>', '->']:
            text = text.replace(' '.join(op), op)

        # Restore array references
        array_pattern = r'<array:(\w+)>'
        text = re.sub(array_pattern, r'\1', text)

        # Restore data structure names
        for ds in ['list', 'dict', 'set', 'tuple', 'array', 'DataFrame']:
            text = text.replace(f'<{ds}>', ds)

        return text

    def clear_memory(self):
        """Clear CUDA cache if available"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate_greedy(self, input_text, max_gen_len=512):
        """
        Generates code string from an input text prompt using greedy search.

        Args:
            input_text: The textual prompt/question
            max_gen_len: Maximum length of generated token sequence

        Returns:
            Generated code as a single string
        """
        self.eval()  # Ensure model is in evaluation mode

        src_tensor, feat_tensor = self.preprocess_input(input_text)

        # Prepare decoder input with <sos>
        output_indices = [self.sos_idx]

        # Greedy decoding loop
        with torch.no_grad():
            for _ in range(max_gen_len):
                tgt_tensor = torch.tensor([output_indices], dtype=torch.long, device=self.device)

                # Forward pass through model
                if self.include_features:
                    logits = self.forward(src_tensor, tgt_tensor, feat_tensor)
                else:
                    logits = self.forward(src_tensor, tgt_tensor)

                # Get next token (greedy selection)
                next_token = logits[0, -1].argmax().item()
                output_indices.append(next_token)

                # Stop generation if EOS token is generated
                if next_token == self.eos_idx:
                    break

        result = self.postprocess_output(output_indices[1:])  # Remove starting <sos>

        # Clean up tensors to help with memory management
        del src_tensor, feat_tensor, tgt_tensor, logits
        self.clear_memory()

        return result

    def generate_beam_search(self, input_text, beam_width=5, max_gen_len=512, length_penalty=1.0):
        """
        Generates code using beam search for potentially better results.

        Args:
            input_text: The textual prompt/question
            beam_width: Beam size for search
            max_gen_len: Maximum length of generated token sequence
            length_penalty: Penalize longer sequences if < 1.0, reward if > 1.0

        Returns:
            Generated code as a single string
        """
        self.eval()  # Ensure model is in evaluation mode

        src_tensor, feat_tensor = self.preprocess_input(input_text)

        # Initialize beam with <sos> token
        beams = [(torch.tensor([[self.sos_idx]], device=self.device), 0.0)]  # (sequence, score)
        finished_beams = []

        with torch.no_grad():
            for step in range(max_gen_len):
                candidates = []

                # Expand each beam
                for seq, score in beams:
                    # If sequence ended with EOS in previous step, add to finished
                    if seq[0, -1].item() == self.eos_idx:
                        # Apply length penalty: (5+len)^p / (5+1)^p
                        normalized_score = score * ((5 + seq.size(1)) ** length_penalty / 6 ** length_penalty)
                        finished_beams.append((seq, normalized_score))
                        continue

                    # Forward pass
                    if self.include_features:
                        logits = self.forward(src_tensor, seq, feat_tensor)
                    else:
                        logits = self.forward(src_tensor, seq)

                    # Get top-k probabilities and tokens
                    probs = torch.log_softmax(logits[0, -1], dim=0)
                    topk_probs, topk_idx = probs.topk(beam_width)

                    # Create new candidate sequences
                    for i in range(beam_width):
                        token_id = topk_idx[i].item()
                        token_score = topk_probs[i].item()
                        new_seq = torch.cat([seq, torch.tensor([[token_id]], device=self.device)], dim=1)
                        new_score = score + token_score
                        candidates.append((new_seq, new_score))

                # Select top-k candidates for next step
                candidates.sort(key=lambda x: x[1], reverse=True)
                beams = candidates[:beam_width]

                # Early stopping if all beams finished
                if all(seq[0, -1].item() == self.eos_idx for seq, _ in beams):
                    finished_beams.extend(beams)
                    break

            # Add unfinished beams to finished list
            for seq, score in beams:
                if seq[0, -1].item() != self.eos_idx:
                    # Apply length penalty
                    normalized_score = score * ((5 + seq.size(1)) ** length_penalty / 6 ** length_penalty)
                    finished_beams.append((seq, normalized_score))

        # Get the highest scoring sequence
        if not finished_beams:
            result = ""  # No valid sequences generated
        else:
            finished_beams.sort(key=lambda x: x[1], reverse=True)
            best_seq = finished_beams[0][0][0].cpu().tolist()
            result = self.postprocess_output(best_seq[1:])  # Remove starting <sos>

        # Clean up tensors to help with memory management
        del src_tensor, feat_tensor, beams, finished_beams
        if 'logits' in locals():
            del logits
        if 'candidates' in locals():
            del candidates
        self.clear_memory()

        return result

    def generate_sampling(self, input_text, max_gen_len=512, temperature=0.8, top_p=0.9):
        """
        Generates code using nucleus sampling (top-p) with temperature.

        Args:
            input_text: The textual prompt/question
            max_gen_len: Maximum length of generated token sequence
            temperature: Controls randomness (lower = more deterministic)
            top_p: Nucleus sampling parameter (cumulative probability threshold)

        Returns:
            Generated code as a single string
        """
        self.eval()  # Ensure model is in evaluation mode

        src_tensor, feat_tensor = self.preprocess_input(input_text)

        # Prepare decoder input with <sos>
        output_indices = [self.sos_idx]

        # Sampling decoding loop
        with torch.no_grad():
            for _ in range(max_gen_len):
                tgt_tensor = torch.tensor([output_indices], dtype=torch.long, device=self.device)

                # Forward pass through model
                if self.include_features:
                    logits = self.forward(src_tensor, tgt_tensor, feat_tensor)
                else:
                    logits = self.forward(src_tensor, tgt_tensor)

                # Apply temperature
                logits = logits[0, -1] / max(0.1, temperature)  # Prevent division by zero

                # Convert to probabilities
                probs = torch.softmax(logits, dim=0)

                # Apply nucleus (top-p) sampling
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=0)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift indices to keep first one above threshold
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False

                # Create a mask and apply it
                indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                filtered_probs = probs.clone()
                filtered_probs[indices_to_remove] = 0.0

                # Renormalize probabilities
                filtered_probs = filtered_probs / filtered_probs.sum()

                # Sample from the filtered distribution
                next_token = torch.multinomial(filtered_probs, 1).item()
                output_indices.append(next_token)

                # Stop generation if EOS token is generated
                if next_token == self.eos_idx:
                    break

        result = self.postprocess_output(output_indices[1:])  # Remove starting <sos>

        # Clean up tensors
        del src_tensor, feat_tensor, tgt_tensor, logits, probs
        self.clear_memory()

        return result

    def generate(self, input_text, method="greedy", **kwargs):
        """
        Complete inference pipeline for code generation.

        Args:
            input_text: The text prompt for code generation
            method: Generation method ("greedy", "beam_search", or "sampling")
            **kwargs: Additional parameters for the specific generation method
                - For greedy: max_gen_len (default=512)
                - For beam_search: beam_width (default=5), max_gen_len (default=512), length_penalty (default=1.0)
                - For sampling: max_gen_len (default=512), temperature (default=0.8), top_p (default=0.9)

        Returns:
            Generated code as a string
        """
        if method == "greedy":
            max_gen_len = kwargs.get("max_gen_len", 512)
            return self.generate_greedy(input_text, max_gen_len=max_gen_len)

        elif method == "beam_search":
            beam_width = kwargs.get("beam_width", 5)
            max_gen_len = kwargs.get("max_gen_len", 512)
            length_penalty = kwargs.get("length_penalty", 1.0)
            return self.generate_beam_search(
                input_text,
                beam_width=beam_width,
                max_gen_len=max_gen_len,
                length_penalty=length_penalty
            )

        elif method == "sampling":
            max_gen_len = kwargs.get("max_gen_len", 512)
            temperature = kwargs.get("temperature", 0.8)
            top_p = kwargs.get("top_p", 0.9)
            return self.generate_sampling(
                input_text,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p
            )

        else:
            raise ValueError(f"Unknown generation method: {method}. Choose from 'greedy', 'beam_search', or 'sampling'")

    @classmethod
    def from_pretrained(cls, model_path, src_vocab_path, tgt_vocab_path, device='cpu'):
        """
        Load a pretrained model with its vocabularies.

        Args:
            model_path: Path to saved model state_dict
            src_vocab_path: Path to source vocabulary file
            tgt_vocab_path: Path to target vocabulary file
            device: Device to load the model on ('cpu' or 'cuda')

        Returns:
            Initialized model ready for inference
        """
        # Load model state
        device = torch.device(device)

        # Try loading as a full model first
        try:
            model = torch.load(model_path, map_location=device)
            if isinstance(model, cls):
                # Initialize vocabularies
                model.init_vocab(src_vocab_path, tgt_vocab_path)
                model.to(device)
                model.eval()
                return model
        except Exception:
            pass  # If loading full model fails, try loading state dict

        # Load as state dictionary
        try:
            state_dict = torch.load(model_path, map_location=device)

            # Extract model parameters from state dict
            model_info = {}
            for key in state_dict:
                # Look for feature embedding to determine if features are used
                if 'feature_embedding' in key and 'weight' in key:
                    model_info['num_features'] = state_dict[key].shape[1]
                    break

            # Create a new model instance
            # Note: We'll need to infer parameters from the state_dict
            model = cls(
                src_vocab_size=state_dict['encoder.tok_embedding.weight'].shape[0],
                tgt_vocab_size=state_dict['decoder.tok_embedding.weight'].shape[0],
                src_pad_idx=0,  # Default, will be overridden by vocab loading
                tgt_pad_idx=0,  # Default, will be overridden by vocab loading
                embed_size=state_dict['encoder.tok_embedding.weight'].shape[1],
                device= 'cuda' if torch.cuda.is_available() else 'cpu',
                num_features=model_info.get('num_features', 0)
            )

            # Load the weights
            model.load_state_dict(state_dict)

            # Initialize vocabularies
            model.init_vocab(src_vocab_path, tgt_vocab_path)

            model.eval()
            return model
        except Exception as e:
            raise ValueError(f"Failed to load model: {str(e)}")
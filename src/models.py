# deep learning libraries
import torch


class AttentionEncoder(torch.nn.Module):
    """
    This the AttentionModel class.

    It applies: LayerNorm -> Multi-Head Self-Attention -> Residual
                LayerNorm -> Feed-Forward (MLP) -> Residual
    """

    def __init__(
        self,
        sequence_length: int,
        d_model: int = 512,
        number_of_heads: int = 8,
        eps: float = 1e-6,
    ) -> None:
        """
        This function is the constructor of the AttentionModel class.

        Args:
            sequence_length: number of tokens (patches + class token)
            d_model: embedding/hidden dimension of each token vector
            number_of_heads: number of attention heads in MHA
            eps: small constant for numerical stability in LayerNorm
        """

        super().__init__()
        self.d_model: int = d_model
        self.number_of_heads: int = number_of_heads
        self.eps: float = eps

        # First LayerNorm before attention
        self.layer_normalization_1: torch.nn.Module = LayerNormalization(
            sequence_length + 1, self.d_model
        )

        # Multi-head self-attention sublayer
        self.multi_head_attention: torch.nn.Module = MultiHeadAttention(
            self.number_of_heads, self.d_model
        )

        # Second LayerNorm before feed-forward
        self.layer_normalization_2: torch.nn.Module = LayerNormalization(
            sequence_length + 1, self.d_model
        )

        # Feed-forward sublayer (MLP) with GELU activation
        self.feed_forward: torch.nn.Module = torch.nn.Sequential(
            torch.nn.Linear(self.d_model, 2048),
            torch.nn.GELU(),
            torch.nn.Linear(2048, self.d_model),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method computes the forward pass.

        Args:
            inputs: inputs tensor. Dimensions: [batch_size, sequence_length, d_model].
        """

        # First sublayer: LayerNorm + Multi-head Self-Attention + Residual Connection
        x = self.layer_normalization_1(inputs)
        x = self.multi_head_attention(x) + inputs # residual connection
        x_clone = x.clone() # save for the next residual connection

        # Second sublayer: LayerNorm + Feed-Forward + Residual Connection
        x = self.layer_normalization_2(x)
        x = self.feed_forward(x) + x_clone # residual connection

        return x


class LayerNormalization(torch.nn.Module):
    """
    This the LayerNormalization class.

    Attr:
        d_model: number of classes.
        gamma: gamma parameter.
        beta: beta parameter.
        eps: epsilon value.
    """

    def __init__(self, sequence_length: int, d_model: int, eps: float = 1e-6) -> None:
        """
        This function is the constructor of the LayerNormalization class.

        This implementation flattens the sequence and feature dims and
        normalizes over (sequence_length * d_model) as one vector per batch item.
        That means statistics are computed across all tokens jointly, not per token.
        This differs from torch.nn.LayerNorm(d_model) used in standard Transformers.
        """

        super().__init__()
        self.d_model = d_model
        self.sequence_length = sequence_length

        # Initialize learnable parameters gamma and beta
        self.gamma = torch.nn.Parameter(torch.ones(self.d_model * self.sequence_length))
        self.beta = torch.nn.Parameter(torch.zeros(self.d_model * self.sequence_length))

        self.eps = eps

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method computes the forward pass.

        Args:
            inputs: [batch_size, sequence_length, d_model]

        Returns:
            outputs: [batch_size, sequence_length, d_model]
        """

        # Flatten per batch item to shape [batch_size, sequence_length * d_model]
        inputs_flat = inputs.view(-1, self.d_model * self.sequence_length)

        # Compute statistics across the flattened feature dimension
        mean = inputs_flat.mean(dim=-1, keepdim=True)
        std = inputs_flat.std(dim=-1, keepdim=True)

        # Apply layer normalization
        normalized = (inputs_flat - mean) / (std + self.eps)

        # Reshape back to original dimensions
        scaled = self.gamma * normalized + self.beta
        reshaped_scaled = scaled.view(-1, self.sequence_length, self.d_model)

        return reshaped_scaled


class MultiHeadAttention(torch.nn.Module):
    """
    This the MultiHeadAttention class.

    Splits the model dimension into 'number_of_heads' subspaces and performs
    scaled dot-product attention in parallel, then projects back to d_model.
    """

    def __init__(self, number_of_heads: int, d_model: int) -> None:
        """
        This function is the constructor of the MultiHeadAttention class.

        Args:
            number_of_heads: number of attention heads
            d_model: embedding/hidden dimension
        """

        super().__init__()
        self.number_of_heads = number_of_heads
        self.d_model = d_model

        # Define the weight matrices for query, key, value, and output
        self.wq = torch.nn.Linear(self.d_model, self.d_model)
        self.wk = torch.nn.Linear(self.d_model, self.d_model)
        self.wv = torch.nn.Linear(self.d_model, self.d_model)
        self.wout = torch.nn.Linear(self.d_model, self.d_model)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method computes the forward pass.

        Args:
            inputs: [batch_size, sequence_length, d_model].

        Returns:
            outputs: [batch_size, number_of_classes].
        """

        # Project inputs to query, key, and value tensors, and reshape for multi-head attention
        # The "-1" in view infers the dimension size based on other dimensions
        query = self.wq(inputs).view(
            inputs.shape[0],
            inputs.shape[1],
            self.number_of_heads,
            -1,
        )

        key = self.wk(inputs).view(
            inputs.shape[0],
            inputs.shape[1],
            self.number_of_heads,
            -1,
        )

        value = self.wv(inputs).view(
            inputs.shape[0],
            inputs.shape[1],
            self.number_of_heads,
            -1,
        )

        """ Scaled dot-product attention:
                scores = Q · K^T / sqrt(d_k)
                weights = softmax(scores)
                output = weights · V
            Shapes after transposes/permutations:
                query.transpose(1,2):   [B, H, L, d_k]
                key.permute(0,2,3,1):   [B, H, d_k, L]
                => scores:              [B, H, L, L] """
        
        attention_scores = torch.matmul(
            query.transpose(1, 2),           # [B, H, L, d_k]
            key.permute(0, 2, 3, 1)          # [B, H, d_k, L]
        ) / (self.d_model**0.5)
        
        # Apply softmax to get attention weights
        attention_weights = torch.nn.functional.softmax(
            attention_scores,
            dim=-1,
        )

        # Compute the output by applying attention weights to the value tensor
        output = torch.matmul(attention_weights, value.transpose(1, 2)).view(
            inputs.shape[0],
            inputs.shape[1],
            -1,
        ) # [B, L, H * d_k]

        return self.wout(output)


class VisionTransformer(torch.nn.Module):
    """
    This the VisionTransformer class.

    Pipeline:
        1) Linear patch embedding (flattened patches -> d_model)
        2) Prepend learnable class token
        3) Add learnable positional embeddings
        4) Transformer encoder block
        5) Take class token representation and classify with MLP head
    """

    def __init__(
        self,
        sequence_length: int,
        patch_size: int = 16 * 16 * 3, # flattened 16x16 RGB patch
        d_model: int = 512,
        number_of_heads: int = 8,
        number_of_classes: int = 10,
        eps: float = 1e-6,
    ) -> None:
        """
        This function is the constructor of the VisionTransformer class.

        Args:
            sequence_length: number of image patches (without the class token)
            patch_size: flattened patch dimension (H*W*C)
            d_model: embedding size for tokens
            number_of_heads: attention heads in the encoder
            number_of_classes: classifier output dimension
            eps: passed to LayerNorm (not directly used here)
        """

        super().__init__()
        self.sequence_length: int = sequence_length
        self.patch_size: int = patch_size
        self.d_model: int = d_model
        self.number_of_heads: int = number_of_heads
        self.eps: float = eps

        # Learnable [CLS] token; expanded per batch and prepended to the patch sequence
        self.class_token: torch.nn.Parameter = torch.nn.Parameter(
            torch.zeros(1, 1, self.d_model)
        )

        # Linear "patch embedding": projects each flattened patch to d_model
        self.patch_embedding: torch.nn.Linear = torch.nn.Linear(
            self.patch_size, self.d_model
        )

        # Learnable positional embeddings for (CLS + sequence_length) tokens
        self.positional_embedding: torch.nn.Parameter = torch.nn.Parameter(
            torch.zeros(1, self.sequence_length + 1, self.d_model)
        )

        # Single Transformer encoder block
        self.attention_encoder: torch.nn.Module = AttentionEncoder(
            self.sequence_length, self.d_model, self.number_of_heads
        )

        # Classification head (MLP) applied to the final CLS token
        self.wout: torch.nn.Linear = torch.nn.Sequential(
            torch.nn.Linear(self.d_model, self.d_model),
            torch.nn.ReLU(),
            torch.nn.Linear(self.d_model, number_of_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method computes the forward pass.

        Args:
            inputs: [batch_size, n_patches, patch_size]

        Returns:
            outputs: [batch_size, number_of_classes]
        """

        # 1) Embed patches to token embeddings: [B, L, d_model]
        x = self.patch_embedding(inputs)

        # 2) Prepend [CLS] token: becomes [B, L+1, d_model]
        x = torch.cat([self.class_token.expand(x.shape[0], -1, -1), x], dim=1)

        # 3) Add positional information
        x = x + self.positional_embedding

        # 4) Transformer encoder over the full token sequence
        x = self.attention_encoder(x)

        # 5) Extract CLS token representation (first token) and classify
        x = x[:, 0, :].squeeze() # [B, d_model] (squeeze keeps [d_model] if B==1)

        # 6) MLP head -> class scores
        x = self.wout(x)

        # 7) Log-softmax for stable training with NLLLoss
        x = torch.nn.functional.log_softmax(x, dim=-1)

        return x

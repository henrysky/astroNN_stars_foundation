from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


def get_initializer(initializer):
    if initializer is None:
        return torch.nn.init.xavier_uniform_
    elif isinstance(initializer, str):
        return getattr(torch.nn.init, initializer)
    else:
        return initializer


def get_activation(activation):
    if activation is None:
        return F.relu
    elif isinstance(activation, str):
        return getattr(F, activation)
    else:
        return activation


def default_initialization(torch_layer: nn.Module):
    """
    Default initialization for a torch layer weight and bias
    """
    torch.nn.init.kaiming_uniform_(torch_layer.weight)

    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(torch_layer.weight)
    bound = 1 / fan_in ** 0.5 if fan_in > 0 else 0
    torch.nn.init.uniform_(torch_layer.bias, -bound, bound)


class NonLinearEmbedding(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        embeddings_initializer: Callable[[torch.Tensor], None] = torch.nn.init.uniform_,
        kernel_initializer=None,
        bias_initializer=None,
        input_length: int = None,
        activation: str = "elu",
        use_bias: bool = True,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embeddings_initializer = get_initializer(embeddings_initializer)
        self.kernel_initializer = get_initializer(kernel_initializer)
        self.bias_initializer = get_initializer(bias_initializer)
        self.input_length = input_length
        self.activation_fn = get_activation(activation)
        self.use_bias = use_bias

        # always reserve token 0 as special padding token
        self.padding_idx = 0

        factory_kwargs = {"device": device, "dtype": dtype}


        self.embeddings = Parameter(
            torch.empty((self.input_dim, self.output_dim), **factory_kwargs)
        )
        self.bias = Parameter(
            torch.empty((self.input_dim, self.output_dim), **factory_kwargs)
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.embeddings_initializer(self.embeddings)
        if self.use_bias:
            self.bias_initializer(self.bias)
        else:
            with torch.no_grad():
                torch.nn.init.zeros_(self.bias)

        with torch.no_grad():
            self.embeddings[self.padding_idx].fill_(0)
            self.bias[self.padding_idx].fill_(0)

    def forward(
        self, input_tokens: torch.Tensor, inputs: torch.Tensor = None
    ) -> torch.Tensor:
        idx = input_tokens
        out = F.embedding(idx, self.embeddings)

        if inputs is not None:
            mag = inputs
            bias = F.embedding(idx, self.bias)
            return self.activation_fn(out * mag + bias)
        else:
            return out


class TransformerBlock(nn.Module):
    """
    Building block of Transformer
    """
    def __init__(
        self,
        head_num: int,
        dense_num: int,
        embedding_dim: int,
        dropout_rate: float,
        activation: Callable[[torch.Tensor], torch.Tensor],
        return_attention_scores: bool = False,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.factory_kwargs = {"device": device, "dtype": dtype}
        self.activation_fn = get_activation(activation)
        self.return_attention_scores = return_attention_scores

        self.attention = torch.nn.MultiheadAttention(
            num_heads=head_num,
            embed_dim=embedding_dim,
            dropout=dropout_rate,
            batch_first=True,
            add_bias_kv=True,
            **self.factory_kwargs,
        )
        self.dense_1 = torch.nn.Linear(
            embedding_dim, dense_num, **self.factory_kwargs
        )
        self.dense_2 = torch.nn.Linear(
            dense_num, embedding_dim, **self.factory_kwargs
        )
        self.layernorm_1 = torch.nn.LayerNorm(
            embedding_dim, **self.factory_kwargs
        )
        self.layernorm_2 = torch.nn.LayerNorm(
            embedding_dim, **self.factory_kwargs
        )
        self.dropout_1 = torch.nn.Dropout(dropout_rate)
        self.dropout_2 = torch.nn.Dropout(dropout_rate)

        default_initialization(self.dense_1)
        default_initialization(self.dense_2)

    def forward(
        self,
        input_query: torch.Tensor,
        input_value: torch.Tensor,
        input_key: torch.Tensor,
        mask=None,
    ) -> torch.Tensor:
        attention_out, _attention_scores = self.attention(
            query=input_query,
            value=input_value,
            key=input_key,
            key_padding_mask=mask,
            need_weights=self.return_attention_scores,
        )
        attention_out_dropped = self.dropout_1(attention_out)
        attention_out_dropped_norm = self.layernorm_1(
            input_query + attention_out_dropped
        )
        dense_1 = self.activation_fn(self.dense_1(attention_out_dropped_norm))
        dense_2 = self.dense_2(dense_1)
        dense_output = self.dropout_2(dense_2)
        dense_output_norm = self.layernorm_2(
            attention_out_dropped_norm + dense_output
        )
        if self.return_attention_scores:
            return dense_output_norm, _attention_scores
        else:
            return dense_output_norm


class StellarPerceptronEncoder(nn.Module):
    def __init__(
        self,
        encoder_head_num: int,
        encoder_dense_num: int,
        embedding_dim: int,
        encoder_dropout_rate: float,
        encoder_activation,
        transformer_class: nn.Module,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.factory_kwargs = {"device": device, "dtype": dtype}

        self.transformer_class = transformer_class
        self.activation = encoder_activation
        self.last_attention_scores = None

        self.encoder_transformer_block_1 = self.transformer_class(
            head_num=encoder_head_num,
            dense_num=encoder_dense_num,
            embedding_dim=embedding_dim,
            dropout_rate=encoder_dropout_rate,
            activation=encoder_activation,
            **self.factory_kwargs,
        )

        self.encoder_transformer_block_2 = self.transformer_class(
            head_num=encoder_head_num // 2,
            dense_num=encoder_dense_num // 2,
            embedding_dim=embedding_dim,
            dropout_rate=encoder_dropout_rate,
            activation=encoder_activation,
            return_attention_scores=True,
            **self.factory_kwargs,
        )

    def forward(
        self, inputs: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        transformer_out_1 = self.encoder_transformer_block_1(
            input_query=inputs, input_value=inputs, input_key=inputs, mask=mask
        )
        transformer_out_2, attention_scores = self.encoder_transformer_block_2(
            input_query=transformer_out_1,
            input_value=transformer_out_1,
            input_key=transformer_out_1,
            mask=mask,
        )
        self.last_attention_scores = attention_scores
        return transformer_out_2

class StellarPerceptronDecoder(nn.Module):
    def __init__(
        self,
        decoder_head_num: int,
        decoder_dense_num: int,
        embedding_dim: int,
        decoder_dropout_rate: float,
        decoder_activation,
        transformer_class: nn.Module,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.factory_kwargs = {"device": device, "dtype": dtype}

        self.transformer_class = transformer_class
        self.activation_fn = get_activation(decoder_activation)
        self.last_attention_scores = None

        self.decoder_transformer_block_1 = self.transformer_class(
            head_num=decoder_head_num,
            dense_num=decoder_dense_num,
            embedding_dim=embedding_dim,
            dropout_rate=decoder_dropout_rate,
            activation=decoder_activation,
            **self.factory_kwargs,
        )
        self.decoder_transformer_block_2 = self.transformer_class(
            head_num=decoder_head_num // 2,
            dense_num=decoder_dense_num // 2,
            embedding_dim=embedding_dim,
            dropout_rate=decoder_dropout_rate,
            activation=decoder_activation,
            **self.factory_kwargs,
        )
        self.decoder_transformer_block_3 = self.transformer_class(
            head_num=decoder_head_num // 4,
            dense_num=decoder_dense_num // 4,
            embedding_dim=embedding_dim,
            dropout_rate=decoder_dropout_rate,
            activation=decoder_activation,
            return_attention_scores=True,
            **self.factory_kwargs,
        )

        self.decoder_dense_3 = torch.nn.Linear(
            embedding_dim, decoder_dense_num, **self.factory_kwargs
        )
        self.decoder_dense_4 = torch.nn.Linear(
            decoder_dense_num, decoder_dense_num // 2, **self.factory_kwargs
        )
        self.decoder_dense_5 = torch.nn.Linear(
            decoder_dense_num // 2,
            decoder_dense_num // 4,
            **self.factory_kwargs,
        )
        self.decoder_dense_6 = torch.nn.Linear(
            decoder_dense_num // 4,
            decoder_dense_num // 8,
            **self.factory_kwargs,
        )
        self.decoder_out = torch.nn.Linear(
            decoder_dense_num // 8, 1, **self.factory_kwargs
        )
        self.decoder_logvar_out = torch.nn.Linear(
            decoder_dense_num // 8, 1, **self.factory_kwargs
        )
        self.decoder_dropout_1 = torch.nn.Dropout(decoder_dropout_rate)

        default_initialization(self.decoder_dense_3)
        default_initialization(self.decoder_dense_4)
        default_initialization(self.decoder_dense_5)
        default_initialization(self.decoder_dense_6)
        default_initialization(self.decoder_out)
        default_initialization(self.decoder_logvar_out)

    def forward(
        self,
        unit_vec: torch.Tensor,
        encoder_outputs: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        transformer_out_1 = self.decoder_transformer_block_1(
            input_query=unit_vec,
            input_value=encoder_outputs,
            input_key=encoder_outputs,
            mask=mask,
        )
        transformer_out_2 = self.decoder_transformer_block_2(
            input_query=transformer_out_1,
            input_value=encoder_outputs,
            input_key=encoder_outputs,
            mask=mask,
        )
        transformer_out_3, attention_scores = self.decoder_transformer_block_3(
            input_query=transformer_out_2,
            input_value=encoder_outputs,
            input_key=encoder_outputs,
            mask=mask,
        )
        # don't return attention scores, just save them here if needed
        self.last_attention_scores = attention_scores
        dense_3 = self.activation_fn(self.decoder_dense_3(transformer_out_3))
        dense_3_dropped = self.decoder_dropout_1(dense_3)
        dense_4 = self.activation_fn(self.decoder_dense_4(dense_3_dropped))
        dense_4_dropped = self.decoder_dropout_1(dense_4)
        dense_5 = self.activation_fn(self.decoder_dense_5(dense_4_dropped))
        dense_5_dropped = self.decoder_dropout_1(dense_5)
        dense_6 = self.activation_fn(self.decoder_dense_6(dense_5_dropped))
        dense_6_dropped = self.decoder_dropout_1(dense_6)
        preds = self.decoder_out(dense_6_dropped)
        preds_logvar = self.decoder_logvar_out(dense_6_dropped)

        return preds, preds_logvar


class StellarPerceptronTorchModel(nn.Module):
    def __init__(
        self,
        embedding_layer: nn.Module,
        embedding_dim: int,
        encoder_head_num: int,
        encoder_dense_num: int,
        encoder_dropout_rate: float,
        encoder_activation: Callable[[torch.Tensor], torch.Tensor],
        decoder_head_num: int,
        decoder_dense_num: int,
        decoder_dropout_rate: float,
        decoder_activation: Callable[[torch.Tensor], torch.Tensor],
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        factory_kwargs = {"device": device, "dtype": dtype}
        self.embedding_layer = embedding_layer

        self.torch_encoder = StellarPerceptronEncoder(
            encoder_head_num,
            encoder_dense_num,
            embedding_dim,
            encoder_dropout_rate,
            encoder_activation,
            transformer_class=TransformerBlock,
            **factory_kwargs,
        )
        self.torch_decoder = StellarPerceptronDecoder(
            decoder_head_num,
            decoder_dense_num,
            embedding_dim,
            decoder_dropout_rate,
            decoder_activation,
            transformer_class=TransformerBlock,
            **factory_kwargs,
        )

    def forward(self, input_tensor, input_token_tensor, output_token_tensor):
        input_embedded = self.embedding_layer(input_token_tensor, input_tensor)
        output_embedding = self.embedding_layer(output_token_tensor)
        padding_mask = torch.eq(
            input_token_tensor, torch.zeros_like(input_token_tensor)
        )
        perception = self.torch_encoder(input_embedded, mask=padding_mask)
        output_tensor = self.torch_decoder(
            output_embedding,
            perception,
            mask=padding_mask,
        )
        return output_tensor

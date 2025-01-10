import torch.nn as nn

class ValueModule(nn.Module):
    def __init__(self, config):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.value.model.d_model,
            nhead=config.value.model.num_heads,
            dropout=config.value.model.dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, config.value.model.num_layers)

        self.layer_norm = nn.LayerNorm(config.value.model.d_model)
        self.dropout = nn.Dropout(config.value.model.dropout)

        self.mlp = nn.Sequential(
            nn.Linear(config.value.model.d_model, config.value.model.d_model // 2),
            nn.ReLU(),
            nn.Dropout(config.value.model.dropout),
            nn.Linear(config.value.model.d_model // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, embeds, mask):
        """
        Processes embeddings and predicts per-residue solubility.

        Args:
            embeds (torch.Tensor): Embeddings [batch_size x seq_len x d_model]
            mask (torch.Tensor): attention masks [bsz x seq_len]
        Returns:
            preds (torch.Tensor): per-residue solubility predictions [batch_size x seq_len]
        """
        encodings = self.encoder(embeds, src_key_padding_mask=~mask.squeeze(1).bool())
        encodings = self.layer_norm(encodings)
        encodings = self.dropout(encodings)

        preds = self.mlp(embeds).squeeze(-1)

        return preds





# import torch
# import torch.nn as nn


# # Main encoder class to process sequence embeddings
# class ValueTrunk(nn.Module):
#     def __init__(self, config):
#         super().__init__()

#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=config.value.model.d_model,
#             nhead=config.value.model.num_heads,
#             dropout=config.value.model.dropout,
#             batch_first=True
#         )
#         self.encoder = nn.TransformerEncoder(encoder_layer, config.value.model.num_layers)

#         self.layer_norm = nn.LayerNorm(config.value.model.d_model)
#         self.dropout = nn.Dropout(config.value.model.dropout)

#     def forward(self, embeds, mask):
#         """
#         Core of acquisition function to process embeddings.

#         Args:
#             embeds (torch.Tensor): Embeddings [batch_size x seq_len x d_model]
#         Returns:
#             preds (torch.Tensor): Processed embeddings [batch_size x seq_len x d_model]
#         """
#         mask = ~mask.squeeze(1).bool()
#         print(f"mask: {mask}")
#         encodings = self.encoder(embeds, src_key_padding_mask=~mask.squeeze(1).bool())
#         encodings = self.layer_norm(encodings)
#         encodings = self.dropout(encodings)
#         return encodings


# # MLP prediction head
# class ValueHead(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(config.value.model.d_model, config.value.model.d_model // 2),
#             nn.ReLU(),
#             nn.Dropout(config.value.model.dropout),
#             nn.Linear(config.value.model.d_model // 2, 1),
#             nn.Sigmoid(),
#         )

#     def forward(self, embeds):
#         """
#         Predict per-residue solubility from processed embeddings.

#         Args:
#             embeds (torch.Tensor): Processed embeddings [batch_size x seq_len x d_model]
#         Returns:
#             preds (torch.Tensor): Per-residue predictions [batch_size x seq_len]
#         """
#         preds = self.mlp(embeds) 
#         return preds


# # Combined module for trunk and head
# class ValueModule(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.trunk = ValueTrunk(config)
#         self.head = ValueHead(config)

#     def forward(self, embeds, attention_mask=None):
#         """Combine prediction trunk and head into one module."""
#         encodings = self.trunk(embeds, attention_mask)
#         preds = self.head(encodings).squeeze(-1)
        
#         temp_preds = preds.clone()
#         temp_preds[torch.isnan(temp_preds)] = 0 # just for testing purposes

#         print(f"preds: {temp_preds}")
#         return temp_preds

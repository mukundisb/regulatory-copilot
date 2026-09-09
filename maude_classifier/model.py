"""
maude_classifier/model.py

Shared model definition for training and serving.
Architecture: Bio_ClinicalBERT encoder + [CLS || Mean-Pool] concatenation + Linear head.
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

# Statutory label indexing
LABEL2ID: Dict[str, int] = {"D": 0, "I": 1, "M": 2, "O": 3}
ID2LABEL: Dict[int, str] = {0: "D", 1: "I", 2: "M", 3: "O"}
NUM_LABELS: int = len(LABEL2ID)


class ClinicalBERTConcatClassifier(nn.Module):
    """
    Sequence classifier pooling both CLS token state and attention-masked
    mean token embeddings to capture holistic and localized clinical signals.
    """
    def __init__(
        self,
        pretrained_model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        num_labels: int = NUM_LABELS,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(pretrained_model_name)
        self.encoder = AutoModel.from_pretrained(pretrained_model_name, config=self.config)
        
        hidden_size = self.config.hidden_size  # 768 for Bio_ClinicalBERT base
        self.concat_dim = hidden_size * 2       # 1536
        
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.concat_dim, num_labels)
        
        # Initialize linear head weights
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def _mean_pooling(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Mean pooling over non-padded tokens using the attention mask.
        """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        encoder_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            encoder_kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**encoder_kwargs)
        last_hidden_state = outputs.last_hidden_state

        # Extract CLS token representation (first token position)
        cls_rep = last_hidden_state[:, 0, :]
        # Compute mean representation across non-padding tokens
        mean_rep = self._mean_pooling(last_hidden_state, attention_mask)

        # Concatenate features -> [batch_size, 1536]
        concat_rep = torch.cat([cls_rep, mean_rep], dim=-1)
        dropped = self.dropout(concat_rep)
        logits = self.classifier(dropped)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, NUM_LABELS), labels.view(-1))

        return {
            "loss": loss,
            "logits": logits,
        }
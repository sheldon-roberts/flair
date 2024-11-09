import logging
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
from tqdm import tqdm

import flair
from flair.data import Dictionary, Sentence
from flair.datasets import DataLoader, FlairDatapointDataset
from flair.embeddings import DocumentEmbeddings
from flair.embeddings.base import load_embeddings
from flair.nn import Classifier

log = logging.getLogger("flair")


class DeepNCMDecoder(torch.nn.Module):
    """Deep Nearest Class Mean (DeepNCM) Classifier for text classification tasks.

    This model combines deep learning with the Nearest Class Mean (NCM) approach.
    It uses document embeddings to represent text, optionally applies an encoder,
    and classifies based on the nearest class prototype in the embedded space.

    The model supports various methods for updating class prototypes during training,
    making it adaptable to different learning scenarios.

    This implementation is based on the research paper:
    Guerriero, S., Caputo, B., & Mensink, T. (2018). DeepNCM: Deep Nearest Class Mean Classifiers.
    In International Conference on Learning Representations (ICLR) 2018 Workshop.
    URL: https://openreview.net/forum?id=rkPLZ4JPM
    """

    def __init__(
            self,
            num_prototypes: int,
            embeddings_size: int,
            encoding_dim: Optional[int] = None,
            alpha: float = 0.9,
            mean_update_method: Literal["online", "condensation", "decay"] = "online",
            use_encoder: bool = True,
            multi_label: bool = False,  # should get from the Model it belongs to
    ) -> None:

        super().__init__()

        self.alpha = alpha
        self.mean_update_method = mean_update_method
        self.use_encoder = use_encoder
        self.multi_label = multi_label

        self.embedding_dim = embeddings_size

        if use_encoder:
            self.encoding_dim = encoding_dim or self.embedding_dim
        else:
            self.encoding_dim = self.embedding_dim

        self.class_prototypes = torch.nn.Parameter(
            torch.nn.functional.normalize(torch.randn(num_prototypes, self.encoding_dim)), requires_grad=False
        )

        self.class_counts = torch.nn.Parameter(torch.zeros(num_prototypes), requires_grad=False)
        self.prototype_updates = torch.zeros_like(self.class_prototypes).to(flair.device)
        self.prototype_update_counts = torch.zeros(num_prototypes).to(flair.device)
        self.to(flair.device)

        self._validate_parameters()

        if self.use_encoder:
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(self.embedding_dim, self.encoding_dim * 2),
                torch.nn.ReLU(),
                torch.nn.Linear(self.encoding_dim * 2, self.encoding_dim),
            )
        else:
            self.encoder = torch.nn.Sequential(torch.nn.Identity())

        # all parameters will be pushed internally to the specified device
        self.to(flair.device)

    def _validate_parameters(self) -> None:
        """Validate that the input parameters have valid and compatible values."""
        assert 0 <= self.alpha <= 1, "alpha must be in the range [0, 1]"
        assert self.mean_update_method in [
            "online",
            "condensation",
            "decay",
        ], f"Invalid mean_update_method: {self.mean_update_method}. Must be 'online', 'condensation', or 'decay'"
        assert self.encoding_dim > 0, "encoding_dim must be greater than 0"

    @property
    def num_prototypes(self):
        """The number of class prototypes"""
        return self.class_prototypes.size(0)

    def _calculate_distances(self, encoded_embeddings: torch.Tensor) -> torch.Tensor:
        """Calculate the squared Euclidean distance between encoded embeddings and class prototypes.

        Args:
            encoded_embeddings: Encoded representations of the input sentences.

        Returns:
            torch.Tensor: Distances between encoded embeddings and class prototypes.
        """
        return torch.cdist(encoded_embeddings, self.class_prototypes).pow(2)

    def _calculate_prototype_updates(self, encoded_embeddings: torch.Tensor, labels: torch.Tensor) -> None:
        """Calculate updates for class prototypes based on the current batch.

        Args:
            encoded_embeddings: Encoded representations of the input sentences.
            labels: True labels for the input sentences.
        """
        one_hot = (
            labels if self.multi_label else torch.nn.functional.one_hot(labels, num_classes=self.num_prototypes).float()
        )

        updates = torch.matmul(one_hot.t(), encoded_embeddings)
        counts = one_hot.sum(dim=0)
        mask = counts > 0
        self.prototype_updates[mask] += updates[mask]
        self.prototype_update_counts[mask] += counts[mask]

    def update_prototypes(self) -> None:
        """Apply accumulated updates to class prototypes."""
        with torch.no_grad():
            update_mask = self.prototype_update_counts > 0
            if update_mask.any():
                if self.mean_update_method in ["online", "condensation"]:
                    new_counts = self.class_counts[update_mask] + self.prototype_update_counts[update_mask]
                    self.class_prototypes[update_mask] = (
                                                                 self.class_counts[update_mask].unsqueeze(1) *
                                                                 self.class_prototypes[update_mask]
                                                                 + self.prototype_updates[update_mask]
                                                         ) / new_counts.unsqueeze(1)
                    self.class_counts[update_mask] = new_counts
                elif self.mean_update_method == "decay":
                    new_prototypes = self.prototype_updates[update_mask] / self.prototype_update_counts[
                        update_mask
                    ].unsqueeze(1)
                    self.class_prototypes[update_mask] = (
                            self.alpha * self.class_prototypes[update_mask] + (1 - self.alpha) * new_prototypes
                    )
                    self.class_counts[update_mask] += self.prototype_update_counts[update_mask]

            # Reset prototype updates
            self.prototype_updates = torch.zeros_like(self.class_prototypes, device=flair.device)
            self.prototype_update_counts = torch.zeros(self.num_classes, device=flair.device)

    def forward(self, embedded: torch.Tensor, label_tensor: torch.Tensor) -> torch.Tensor:
        encoded_embeddings = embedded

        # if self.learning_mode == "learn_only_map_and_prototypes":
        #    embedded = embedded.detach()

        # decode embeddings into prototype space
        # encoded = self.metric_space_decoder(embedded) if self.metric_space_decoder is not None else embedded

        distances = self._calculate_distances(encoded_embeddings)

        self._calculate_prototype_updates(encoded_embeddings, label_tensor)

        scores = -distances

        return scores

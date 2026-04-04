import torch

from brats_project.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class SegTrainer(nnUNetTrainer):
    """Project-local trainer alias used by the standalone BraTS project."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 150

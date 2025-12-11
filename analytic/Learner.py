import torch
from os import path
from abc import ABCMeta, abstractmethod
from torch.utils.data import DataLoader
from torch._prims_common import DeviceLikeType
from typing import Union, Dict, Any, Optional, Sequence

loader_t = DataLoader[Union[torch.Tensor, torch.Tensor]]


class Learner(metaclass=ABCMeta):
    def __init__(
        self,
        args: Dict[str, Any],
        backbone: torch.nn.Module,
        backbone_output: int,
        device=None,
        all_devices: Optional[Sequence[DeviceLikeType]] = None,
    ) -> None:
        self.args = args
        self.backbone = backbone
        self.backbone_output = backbone_output
        self.device = device
        self.all_devices = all_devices
        self.model: torch.nn.Module
        self.memory = []

    @abstractmethod
    def base_training(
        self,
        train_loader: loader_t,
        val_loader: loader_t,
        baseset_size: int,
    ) -> None:
        raise NotImplementedError()

    @abstractmethod
    def learn(
        self,
        data_loader: loader_t,
        incremental_size: int,
        phase: int,
        desc: str = "Incremental Learning"
    ) -> None:
        raise NotImplementedError()

    @abstractmethod
    def before_validation() -> None:
        raise NotImplementedError()

    @abstractmethod
    def inference(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def save_object(self, model, file_name: str) -> None:
        torch.save(model, path.join(self.args["saving_root"], file_name))

    def load_object(self, model, file_name: str, file_path: str = None) -> None:
        if file_path == None:
            file_path = path.join(self.args["saving_root"], file_name)
        state_dict = torch.load(file_path, map_location=self.device)
        model.load_state_dict(state_dict)
        print(f"Model loaded from {file_path}")
        return model

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return self.inference(X)

    def before_training(self, dataset) -> None:
        pass

# -*- coding: utf-8 -*-
"""
Implementation of the ACIL [1] and the G-ACIL [2].
The G-ACIL is a generalization of the ACIL in the generalized setting.
For the popular setting, the G-ACIL is equivalent to the ACIL.

References:
[1] Zhuang, Huiping, et al.
    "ACIL: Analytic class-incremental learning with absolute memorization and privacy protection."
    Advances in Neural Information Processing Systems 35 (2022): 11602-11614.
[2] Zhuang, Huiping, et al.
    "G-ACIL: Analytic Learning for Exemplar-Free Generalized Class Incremental Learning"
    arXiv preprint arXiv:2403.15706 (2024).
"""

import torch
from os import path
from tqdm import tqdm
from typing import Any, Dict, Optional, Sequence
from utils import set_weight_decay, validate
from torch._prims_common import DeviceLikeType
from torch.nn import DataParallel
from .Learner import Learner, loader_t
import numpy as np

class FusionModel(torch.nn.Module):
    def __init__(self, backbone, backbone_output, baseset_size):
        super().__init__()
        self.backbone = backbone
        self.fc = torch.nn.Linear(backbone_output, baseset_size)
    def forward(self, video, audio, phase=0):
        video_out, audio_out, x = self.backbone(video, audio)
        x = self.fc(x)
        return video_out, audio_out, x

class ReplayLearner(Learner):
    """
    This implementation is for the G-ACIL [2], a general version of the ACIL [1] that
    supports mini-batch learning and the general CIL setting.
    In the traditional CIL settings, the G-ACIL is equivalent to the ACIL.
    """

    def __init__(
        self,
        args: Dict[str, Any],
        backbone: torch.nn.Module,
        backbone_output: int,
        device=None,
        all_devices: Optional[Sequence[DeviceLikeType]] = None,
    ) -> None:
        super().__init__(args, backbone, backbone_output, device, all_devices)
        self.learning_rate: float = args["learning_rate"]
        self.buffer_size: int = args["buffer_size"]
        self.gamma: float = args["gamma"]
        self.base_epochs: int = args["base_epochs"]
        self.warmup_epochs: int = args["warmup_epochs"]
        self.memory = []
        self.memory_per_domain: int = 20
        self._known_domains = 0

    def base_training(
        self,
        train_loader: loader_t,
        val_loader: loader_t,
        baseset_size: int,
    ) -> None:
        model = FusionModel(self.backbone, self.backbone_output, baseset_size).to(self.device, non_blocking=True)
        model = self.wrap_data_parallel(model)
        self.class_num = baseset_size


        if self.args["separate_decay"]:
            params = set_weight_decay(model, self.args["weight_decay"])
        else:
            params = model.parameters()

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, params),
                           lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-08)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

        criterion = torch.nn.CrossEntropyLoss().to(self.device, non_blocking=True)

        best_acc = 0.0
        logging_file_path = path.join(self.args["saving_root"], "base_training.csv")
        logging_file = open(logging_file_path, "w", buffering=1)
        print(
            "epoch",
            "best_acc@1",
            "loss",
            "acc@1",
            "acc@5",
            "f1-micro",
            "training_loss",
            "training_acc@1",
            "training_acc@5",
            "training_f1-micro",
            "training_learning-rate",
            file=logging_file,
            sep=",",
        )

        for epoch in range(self.base_epochs + 1):
            if epoch != 0:
                print(
                    f"Base Training - Epoch {epoch}/{self.base_epochs}",
                    f"(Learning Rate: {optimizer.state_dict()['param_groups'][0]['lr']})",
                )
                model.train()
                for X, y,_ in tqdm(train_loader, "Training"):
                    video, audio = X
                    video = video.to(self.device, non_blocking=True)
                    audio = audio.to(self.device, non_blocking=True)
                    # X: torch.Tensor = X.to(self.device, non_blocking=True)
                    video_label, audio_label, y = y
                    y: torch.Tensor = y.to(self.device, non_blocking=True)
                    video_label: torch.Tensor = video_label.to(self.device, non_blocking=True)
                    audio_label: torch.Tensor = audio_label.to(self.device, non_blocking=True)
                    assert y.max() < baseset_size

                    optimizer.zero_grad(set_to_none=True)
                    # logits = model(X)
                    video_out, audio_out, logits = model(video, audio)
                    loss1 = criterion(video_out, video_label)
                    loss2 = criterion(audio_out, audio_label)
                    loss3 = criterion(logits, y)
                    loss = loss1 + loss2 + loss3
                    # loss = loss3
                    # loss: torch.Tensor = criterion(logits, y)
                    loss.backward()
                    optimizer.step()
                scheduler.step()

            # Validation on training set
            model.eval()
            train_meter = validate(
                model, train_loader, baseset_size, desc="Training (Validation)"
            )
            print(
                f"loss: {train_meter.loss:.4f}",
                f"acc@1: {train_meter.accuracy * 100:.3f}%",
                f"auc: {train_meter.auc * 100:.3f}%",
                f"f1-micro: {train_meter.f1_micro * 100:.3f}%",
                sep="    ",
            )

            val_meter = validate(model, val_loader, baseset_size, desc="Testing")
            if val_meter.accuracy > best_acc:
                best_acc = val_meter.accuracy
                self.save_object(
                    # (self.backbone, self.backbone_output),
                    # "backbone.pth",
                    model.state_dict(),
                    "model.pth"
                )

            # Validation on testing set
            print(
                f"loss: {val_meter.loss:.4f}",
                f"acc@1: {val_meter.accuracy * 100:.3f}%",
                f"auc: {val_meter.auc * 100:.3f}%",
                f"f1-micro: {val_meter.f1_micro * 100:.3f}%",
                f"best_acc@1: {best_acc * 100:.3f}%",
                sep="    ",
            )
            print(
                epoch,
                best_acc,
                val_meter.loss,
                val_meter.accuracy,
                val_meter.auc,
                val_meter.f1_micro,
                train_meter.loss,
                train_meter.accuracy,
                train_meter.auc,
                train_meter.f1_micro,
                optimizer.state_dict()["param_groups"][0]["lr"],
                file=logging_file,
                sep=",",
            )
        logging_file.close()
        self.backbone.eval()
        # self.make_model()
        # self.model = model
        self.model = self.load_object(model, "model.pth")

        self._known_domains += 1    # 可能增加的域不止1个
        # self._reduce_exemplar(train_loader, self.memory_per_class)
        self._construct_exemplar(train_loader, self.memory_per_domain)

    
    def learn(
        self,
        data_loader: loader_t,
        incremental_size: int,
        phase: int,
        desc: str = "Incremental Learning",
    ) -> None:
        if desc == 'Re-align': return
        params = self.model.parameters()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, params),
                           lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-08)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)
        criterion = torch.nn.CrossEntropyLoss().to(self.device, non_blocking=True)

        self.model.train()
        # for epoch in range(self.base_epochs):
        for X, y, _ in tqdm(data_loader, desc=desc):
            video, audio = X
            video = video.to(self.device, non_blocking=True)
            audio = audio.to(self.device, non_blocking=True)
            # X: torch.Tensor = X.to(self.device, non_blocking=True)
            video_label, audio_label, y = y
            y: torch.Tensor = y.to(self.device, non_blocking=True)
            video_label: torch.Tensor = video_label.to(self.device, non_blocking=True)
            audio_label: torch.Tensor = audio_label.to(self.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            # self.model.fit(video, audio, y, increase_size=incremental_size)
            video_out, audio_out, logits = self.model(video, audio)
            loss1 = criterion(video_out, video_label)
            loss2 = criterion(audio_out, audio_label)
            loss3 = criterion(logits, y)
            loss = loss1 + loss2 + loss3
            loss.backward()
            optimizer.step()
        scheduler.step()

        self._known_domains += 1
        # self._reduce_exemplar(data_loader, self.memory_per_class)
        self._construct_exemplar(data_loader, self.memory_per_domain)


    def before_validation(self) -> None:
        pass

    def inference(self, video: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        return self.model(video, audio)

    @torch.no_grad()
    def wrap_data_parallel(self, model: torch.nn.Module) -> torch.nn.Module:
        if self.all_devices is not None and len(self.all_devices) > 1:
            return DataParallel(model, self.all_devices, output_device=self.device) # type: ignore
        return model

    def before_training(self, dataset):
        pass

    def _reduce_exemplar(self, dataloader, m):
        print("Reducing exemplars...({} per classes)".format(m))
        dummy_data, dummy_targets = copy.deepcopy(self._data_memory), copy.deepcopy(
            self._targets_memory
        )
        # self._class_means = np.zeros((self._total_classes, self.feature_dim))
        self._data_memory, self._targets_memory = np.array([]), np.array([])

        for class_idx in range(self._known_classes):
            mask = np.where(dummy_targets == class_idx)[0]
            dd, dt = dummy_data[mask][:m], dummy_targets[mask][:m]
            self._data_memory = (
                np.concatenate((self._data_memory, dd))
                if len(self._data_memory) != 0
                else dd
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, dt))
                if len(self._targets_memory) != 0
                else dt
            )

            # # Exemplar mean
            # idx_dataset = data_manager.get_dataset(
            #     [], source="train", mode="test", appendent=(dd, dt)
            # )
            # idx_loader = DataLoader(
            #     idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            # )
            # vectors, _ = self._extract_vectors(idx_loader)
            # vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            # mean = np.mean(vectors, axis=0)
            # mean = mean / np.linalg.norm(mean)

            # self._class_means[class_idx, :] = mean

    def _construct_exemplar(self, dataloader, m):
        # data, targets, idx_dataset = data_manager.get_dataset(
        #     np.arange(class_idx, class_idx + 1),
        #     source="train",
        #     mode="test",
        #     ret_data=True,
        # )
        # idx_loader = DataLoader(
        #     idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
        # )
        # vectors, _ = self._extract_vectors(idx_loader)
        # vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
        # class_mean = np.mean(vectors, axis=0)

        # # Select
        # selected_exemplars = []
        # exemplar_vectors = []  # [n, feature_dim]
        # for k in range(1, m + 1):
        #     S = np.sum(
        #         exemplar_vectors, axis=0
        #     )  # [feature_dim] sum of selected exemplars vectors
        #     mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
        #     i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))
        #     selected_exemplars.append(
        #         np.array(data[i])
        #     )  # New object to avoid passing by inference
        #     exemplar_vectors.append(
        #         np.array(vectors[i])
        #     )  # New object to avoid passing by inference

        #     vectors = np.delete(
        #         vectors, i, axis=0
        #     )  # Remove it to avoid duplicative selection
        #     data = np.delete(
        #         data, i, axis=0
        #     )  # Remove it to avoid duplicative selection

        # selected_exemplars = np.array(selected_exemplars)
        # exemplar_targets = np.full(m, class_idx)
        # self._data_memory = (
        #     np.concatenate((self._data_memory, selected_exemplars))
        #     if len(self._data_memory) != 0
        #     else selected_exemplars
        # )
        # self._targets_memory = (
        #     np.concatenate((self._targets_memory, exemplar_targets))
        #     if len(self._targets_memory) != 0
        #     else exemplar_targets
        # )

        # # Exemplar mean
        # idx_dataset = data_manager.get_dataset(
        #     [],
        #     source="train",
        #     mode="test",
        #     appendent=(selected_exemplars, exemplar_targets),
        # )
        # idx_loader = DataLoader(
        #     idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
        # )
        # vectors, _ = self._extract_vectors(idx_loader)
        # vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
        # mean = np.mean(vectors, axis=0)
        # mean = mean / np.linalg.norm(mean)

        # self._class_means[class_idx, :] = mean

        # 随机选择m个样本作为exemplar
        exemplars = []
        for i, data in enumerate(dataloader):
            data, label, index = data
            fn_img, fn_aud, label, start = index
            exemplar = list(zip(fn_img, fn_aud, label, start))
            for i in exemplar:
                fn_img, fn_aud, label, start = i
                label = label.item()
                start = start.item()
                idx = (fn_img, fn_aud, label, start)
                exemplars.append(idx)
                if len(exemplars) >= m:
                    exemplars = exemplars[:m]
                    break
        self.memory.extend(exemplars)
        print(f'Construct exemplars: {len(exemplars)}')
            
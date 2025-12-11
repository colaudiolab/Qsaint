# -*- coding: utf-8 -*-

import torch
from os import path
from tqdm import tqdm
from typing import Any, Dict, Optional, Sequence
from utils import set_weight_decay, validate
from torch._prims_common import DeviceLikeType
from .Buffer import RandomBuffer
from torch.nn import DataParallel
from .Learner import Learner, loader_t
from .AnalyticLinear import AnalyticLinear, RecursiveLinear
from .Adapater import Adapter
import copy
import numpy as np

class FusionModel(torch.nn.Module):
    def __init__(self, backbone, backbone_output, baseset_size):
        super().__init__()
        self.backbone = backbone
        # self.fc = torch.nn.Sequential(torch.nn.Linear(backbone_output, backbone_output), torch.nn.ReLU(inplace=True),
                                            #   torch.nn.Linear(backbone_output, baseset_size))
        self.fc = torch.nn.Linear(backbone_output, baseset_size)
    def forward(self, video, audio, phase=0):
        video_out, audio_out, x = self.backbone(video, audio)
        x = self.fc(x)
        return video_out, audio_out, x

class ACIL(torch.nn.Module):
    def __init__(
        self,
        backbone_output: int,
        backbone: torch.nn.Module = torch.nn.Flatten(),
        buffer_size: int = 8192,
        gamma: float = 1e-3,
        device=None,
        dtype=torch.double,
        linear: type[AnalyticLinear] = RecursiveLinear,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.backbone = backbone
        self.backbone_output = backbone_output
        self.buffer_size = buffer_size
        self.buffer = RandomBuffer(backbone_output, buffer_size, **factory_kwargs)
        self.analytic_linear = linear(buffer_size, gamma, **factory_kwargs)
        self.eval()
        self.adapter = []

    @torch.no_grad()
    def feature_expansion(self, video: torch.Tensor, audio: torch.Tensor, adapter: torch.nn.Module = None) -> torch.Tensor:
        video_out, audio_out, x = self.backbone(video, audio)
        
        return x

    @torch.no_grad()
    def forward(self, video: torch.Tensor, audio: torch.Tensor, phase: int) -> torch.Tensor:
        return self.analytic_linear(self.feature_expansion(video, audio))

    @torch.no_grad()
    def fit(self, video: torch.Tensor, audio: torch.Tensor, y: torch.Tensor, adapter: torch.nn.Module = None, phase: int = 0, *args, **kwargs) -> None:
        if phase > 0 and len(self.adapter) < phase:
            self.adapter.append(adapter)
            print(f"Number of adapters: {len(self.adapter)}")
        Y = torch.nn.functional.one_hot(y)
        X = self.feature_expansion(video, audio, adapter)
        self.analytic_linear.fit(X, Y)

    @torch.no_grad()
    def update(self) -> None:
        self.analytic_linear.update()


class QsaintLearner(Learner):
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
        self.make_model()
        self.FCN = None
        self.adapter = []
        # self.kd_weight = 0.1
        self.kd_weight = 0.001

    def base_training(
        self,
        train_loader: loader_t,
        val_loader: loader_t,
        baseset_size: int,
    ) -> None:
        # model = torch.nn.Sequential(
        #     self.backbone,
        #     torch.nn.Linear(self.backbone_output, baseset_size),
        # ).to(self.device, non_blocking=True)
        model = FusionModel(self.backbone, self.backbone_output, baseset_size).to(self.device, non_blocking=True)
        model = self.wrap_data_parallel(model)

        if self.args["separate_decay"]:
            params = set_weight_decay(model, self.args["weight_decay"])
        else:
            params = model.parameters()
        # optimizer = torch.optim.SGD(
        #     params,
        #     lr=self.learning_rate,
        #     momentum=self.args["momentum"],
        #     weight_decay=self.args["weight_decay"],
        # )
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, params),
                           lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-08)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, T_max=self.base_epochs - self.warmup_epochs, eta_min=1e-6 # type: ignore
        # )
        # if self.warmup_epochs > 0:
        #     warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        #         optimizer,
        #         start_factor=1e-3,
        #         total_iters=self.warmup_epochs,
        #     )
        #     scheduler = torch.optim.lr_scheduler.SequentialLR(
        #         optimizer, [warmup_scheduler, scheduler], [self.warmup_epochs]
        #     )

        # criterion = torch.nn.CrossEntropyLoss(
        #     label_smoothing=self.args["label_smoothing"]
        # ).to(self.device, non_blocking=True)
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
                if epoch != 0:
                    self.save_object(
                        # (self.backbone, X.shape[1], self.backbone_output),
                        self.backbone.state_dict(),
                        "backbone.pth",
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
        self.make_model()
        self.FCN = model.module.fc
        self.FCN.eval()
        self.protoSave(train_loader, 0)


    def make_model(self) -> None:
        self.model = ACIL(
            self.backbone_output,
            self.wrap_data_parallel(self.backbone),
            self.buffer_size,
            self.gamma,
            device=self.device,
            dtype=torch.double,
            linear=RecursiveLinear,
        )

    # @torch.no_grad()
    def learn(
        self,
        data_loader: loader_t,
        incremental_size: int,
        phase: int,
        desc: str = "Incremental Learning",
    ) -> None:
        if phase > 0:
            print('training new adapter')
            adapter = self.train_adapter(data_loader)
        else:
            adapter = None
        # adapter = None
        self.model.eval()
        for X, y,_ in tqdm(data_loader, desc=desc):
            video, audio = X
            video = video.to(self.device, non_blocking=True)
            audio = audio.to(self.device, non_blocking=True)
            # X: torch.Tensor = X.to(self.device, non_blocqking=True)
            video_label, audio_label, y = y
            y: torch.Tensor = y.to(self.device, non_blocking=True)
            self.model.fit(video, audio, y, adapter=adapter, phase=phase)

    

    def before_validation(self) -> None:
        self.model.update()

    def inference(self, video: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        return self.model(video, audio)

    @torch.no_grad()
    def wrap_data_parallel(self, model: torch.nn.Module) -> torch.nn.Module:
        if self.all_devices is not None and len(self.all_devices) > 1:
            return DataParallel(model, self.all_devices, output_device=self.device) # type: ignore
        return model

    def _compute_adapter_loss(self, logits, target, feature,):
        loss_cls = torch.nn.CrossEntropyLoss()(logits, target)

        prototype = []
        for label in target:
            temp = self.prototype[label.item()]
            prototype.append(temp)
        prototype = torch.tensor(prototype).to(self.device)
        loss_kd = torch.dist(feature, prototype, 2)
        return loss_cls + self.kd_weight*loss_kd

    def protoSave(self, loader, current_task):
        print('save prototype')
        features = []
        labels = []
        self.backbone.eval()
        with torch.no_grad():
            for i, (X, y,_) in enumerate(loader):
                video, audio = X
                video = video.to(self.device, non_blocking=True)
                audio = audio.to(self.device, non_blocking=True)
                # X: torch.Tensor = X.to(self.device, non_blocking=True)
                video_label, audio_label, target = y
                _, _, feature = self.backbone(video, audio)
                # if feature.shape[0] == self.args.batch_size:
                labels.append(target.numpy())
                features.append(feature.cpu().numpy())
        labels_set = np.unique(labels)
        labels = np.array(labels)
        labels = np.reshape(labels, labels.shape[0] * labels.shape[1])
        features = np.array(features)
        features = np.reshape(features, (features.shape[0] * features.shape[1], features.shape[2]))
        feature_dim = features.shape[1]

        prototype = []
        for item in labels_set:
            index = np.where(item == labels)[0]
            feature_classwise = features[index]
            prototype.append(np.mean(feature_classwise, axis=0))

        if current_task == 0:
            self.prototype = prototype
        else:
            raise NotImplementedError('prototype saving for incremental learning is not implemented yet')
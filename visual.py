# -*- coding: utf-8 -*-

import torch
from os import path
from tqdm import tqdm
from config import load_args, ALL_METHODS
from models import load_backbone
from typing import Any, Dict, List, Tuple, Optional
from datasets import Features, load_dataset
from utils import set_determinism, validate
from torch._prims_common import DeviceLikeType
from torch.utils.data import Dataset, DataLoader

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def make_dataloader(
    dataset: Dataset,
    shuffle: bool = False,
    batch_size: int = 256,
    num_workers: int = 8,
    device: Optional[DeviceLikeType] = None,
    persistent_workers: bool = False,
) -> DataLoader:
    pin_memory = (device is not None) and (torch.device(device).type == "cuda")
    config = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "pin_memory_device": str(device) if pin_memory else "",
        "persistent_workers": persistent_workers,
        "drop_last": True,
    }
    try:
        from prefetch_generator import BackgroundGenerator

        class DataLoaderX(DataLoader):
            def __iter__(self):
                return BackgroundGenerator(super().__iter__())

        return DataLoaderX(dataset, **config)
    except ImportError:
        return DataLoader(dataset, **config)


def check_cache_features(root: str) -> bool:
    files_list = ["X_train.pt", "y_train.pt", "X_test.pt", "y_test.pt"]
    for file in files_list:
        if not path.isfile(path.join(root, file)):
            return False
    return True


@torch.no_grad()
def cache_features(
    backbone: torch.nn.Module,
    dataloader: DataLoader[Tuple[torch.Tensor, torch.Tensor]],
    device: Optional[DeviceLikeType] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    backbone.eval()
    X_all: List[torch.Tensor] = []
    y_all: List[torch.Tensor] = []
    for X, y,_ in tqdm(dataloader, "Caching"):
        # X: torch.Tensor = backbone(X.to(device))
        video, audio = X
        video = video.to(device, non_blocking=True)
        audio = audio.to(device, non_blocking=True)
        video_label, audio_label, y = y
        video_out, audio_out, X = backbone(video, audio)
        y: torch.Tensor = y.to(torch.int16, non_blocking=True)
        X_all.append(X.cpu())
        y_all.append(y.cpu())
    return torch.cat(X_all), torch.cat(y_all)


def main(args: Dict[str, Any]):
    backbone_name = args["backbone"]

    # Select device
    if args["cpu_only"] or not torch.cuda.is_available():
        main_device = torch.device("cpu")
        all_gpus = None
    elif args["gpus"] is not None:
        gpus = args["gpus"]
        main_device = torch.device(f"cuda:{gpus[0]}")
        all_gpus = [torch.device(f"cuda:{gpu}") for gpu in gpus]
    else:
        main_device = torch.device("cuda:0")
        all_gpus = None

    if args["seed"] is not None:
        set_determinism(args["seed"])

    if "backbone_path" in args:
        assert path.isfile(
            args["backbone_path"]
        ), f"Backbone file \"{args['backbone_path']}\" doesn't exist."
        preload_backbone = True
        backbone_state_dict = torch.load(
            args["backbone_path"], map_location=main_device, weights_only=False
        )
        backbone, input_img_size, feature_size = load_backbone(backbone_name)
        backbone.load_state_dict(backbone_state_dict)
        print('loading backbone from {}'.format(args["backbone_path"]))
    else:
        print('Loading backbone from scratch')
        # Load model pre-train on ImageNet if there is no base training dataset.
        preload_backbone = False
        load_pretrain = args["base_ratio"] == 0 or "ImageNet" not in args["dataset"]
        backbone, input_img_size, feature_size = load_backbone(backbone_name, pretrain=load_pretrain)
        if load_pretrain:
            assert args["dataset"] != "ImageNet", "Data may leak!!!"
    backbone = backbone.to(main_device, non_blocking=True)
    # print(next(backbone.parameters()).device)

    dataset_args = {
        "name": args["dataset"],
        "root": args["data_root"],
        "base_ratio": args["base_ratio"],
        "num_phases": args["phases"],
        "shuffle_seed": args["dataset_seed"] if "dataset_seed" in args else None,
        'image_size': 128,
        'num_classes': 2,
        'txt_dir': /multidataset'
    }
    dataset_train = load_dataset(train=True, augment=False, **dataset_args)
    dataset_test = load_dataset(train=False, augment=False, **dataset_args)

    # Select algorithm
    assert args["method"] in ALL_METHODS, f"Unknown method: {args['method']}"
    learner = ALL_METHODS[args["method"]](
        args, backbone, feature_size, main_device, all_devices=all_gpus
    )

    # if args["base_ratio"] > 0 and not preload_backbone:
    #     train_subset = dataset_train.subset_at_phase(0)
    #     test_subset = dataset_test.subset_at_phase(0)
    #     train_loader = make_dataloader(
    #         train_subset,
    #         True,
    #         args["batch_size"],
    #         args["num_workers"],
    #         device=main_device,
    #     )
    #     test_loader = make_dataloader(
    #         test_subset,
    #         False,
    #         args["batch_size"],
    #         args["num_workers"],
    #         device=main_device,
    #     )
    #     learner.base_training(
    #         train_loader,
    #         test_loader,
    #         # dataset_train.base_size,
    #         dataset_train.num_classes,
    #     )

    # Load dataset
    if args["cache_features"]:
        if "cache_path" not in args or args["cache_path"] is None:
            args["cache_path"] = args["saving_root"]
        if not check_cache_features(args["cache_path"]):
            backbone = learner.backbone.eval()
            
            for phase in range(0, args["phases"]):
                train_loader = make_dataloader(
                    dataset_train.subset_at_phase(phase),
                    False,
                    args["batch_size"],
                    args["num_workers"],
                    device=main_device,
                )

                print(path.join(args["cache_path"], f"X_{phase}.pt"))
                X_test, y_test = cache_features(backbone, train_loader, device=main_device)
                torch.save(X_test, path.join(args["cache_path"], f"dataset_X_{phase}.pt"))
                torch.save(y_test, path.join(args["cache_path"], f"dataset_y_{phase}.pt"))


if __name__ == "__main__":
    main(load_args())

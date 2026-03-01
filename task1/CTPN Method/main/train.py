import argparse
import math
import os
from dataclasses import dataclass
from typing import List, Tuple

from PIL import Image
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import torchvision


STRIDE = 16


@dataclass
class SamplePaths:
    image: str
    label: str


class CtpnDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        n_anchor: int = 10,
        resolution: Tuple[int, int] = (448, 224),
        transform=None,
    ):
        super().__init__()
        self.data_dir = os.path.abspath(data_dir)
        self.image_dir = os.path.join(self.data_dir, "image")
        self.label_dir = os.path.join(self.data_dir, "label")
        self.resolution = tuple(resolution)
        self.grid_h = self.resolution[0] // STRIDE
        self.grid_w = self.resolution[1] // STRIDE
        self.n_anchor = n_anchor
        self.anchors = torch.tensor([5 * (2 ** (i / 2)) for i in range(n_anchor)])
        self.transform = transform or transforms.Compose([transforms.ToTensor()])

        if not os.path.isdir(self.image_dir) or not os.path.isdir(self.label_dir):
            raise FileNotFoundError(
                f"Expected dataset folders: {self.image_dir} and {self.label_dir}"
            )

        image_files = sorted(
            [f for f in os.listdir(self.image_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        )
        self.samples: List[SamplePaths] = []
        for image_name in image_files:
            stem = os.path.splitext(image_name)[0]
            label_path = os.path.join(self.label_dir, stem + ".txt")
            if os.path.exists(label_path):
                self.samples.append(
                    SamplePaths(
                        image=os.path.join(self.image_dir, image_name),
                        label=label_path,
                    )
                )

        if not self.samples:
            raise RuntimeError(f"No matched image/label samples in {self.data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample.image).convert("RGB")

        h_scaling = self.resolution[0] / img.height
        w_scaling = self.resolution[1] / img.width
        img = transforms.functional.resize(img, self.resolution)
        img = self.transform(img)

        tgt_cls = torch.zeros(self.grid_h, self.grid_w, self.n_anchor, dtype=torch.long)
        tgt_v = torch.zeros(self.grid_h, self.grid_w, self.n_anchor, 2)
        idx_v = torch.zeros_like(tgt_v, dtype=torch.bool)
        tgt_o = torch.zeros(self.grid_h, self.grid_w, self.n_anchor)
        idx_o = torch.zeros_like(tgt_o, dtype=torch.bool)

        with open(sample.label, "r", encoding="utf-8", errors="ignore") as fo:
            for line in fo:
                parts = line.strip().split(",")
                if len(parts) < 4:
                    continue
                try:
                    x0, y0, x1, y1 = [float(v) for v in parts[:4]]
                except ValueError:
                    continue

                x0 *= w_scaling
                x1 *= w_scaling
                y0 *= h_scaling
                y1 *= h_scaling

                if x1 <= x0 or y1 <= y0:
                    continue

                cy_box = (y0 + y1) / 2
                h_box = max(y1 - y0, 1.0)

                row = int(cy_box // STRIDE)
                row = max(0, min(self.grid_h - 1, row))

                col_start = int(max(0, min(self.grid_w - 1, x0 // STRIDE)))
                col_end = int(max(0, min(self.grid_w, math.ceil(x1 / STRIDE))))
                if col_end <= col_start:
                    col_end = min(self.grid_w, col_start + 1)

                anc = (self.anchors - h_box).abs().argmin().item()
                tgt_cls[row, col_start:col_end, anc] = 1

                cy_anc = row * STRIDE + STRIDE / 2
                h_anc = float(self.anchors[anc].item())
                v_c = (cy_box - cy_anc) / h_anc
                v_h = math.log(h_box / h_anc)
                tgt_v[row, col_start:col_end, anc, 0] = v_c
                tgt_v[row, col_start:col_end, anc, 1] = v_h
                idx_v[row, col_start:col_end, anc, :] = True

                for x_side in [x0, x1]:
                    start = int(round(max((x_side - 32) / STRIDE, 0)))
                    end = int(round(min((x_side + 32) / STRIDE, self.grid_w)))
                    if end <= start:
                        end = min(self.grid_w, start + 1)
                    cols = torch.arange(start, end)
                    cx_anc = cols * STRIDE + STRIDE / 2
                    o = (x_side - cx_anc) / STRIDE
                    tgt_o[row, cols, anc] = o
                    idx_o[row, cols, anc] = True

        return img, tgt_cls, tgt_v, idx_v, tgt_o, idx_o


class CtpnModel(nn.Module):
    def __init__(self, n_anchor=10, pretrained_backbone=True):
        super().__init__()
        self.n_anchor = n_anchor
        self.features = self._build_backbone(pretrained_backbone)
        self.slider = nn.Conv2d(512, 512, 3, padding=1)
        self.bilstm = nn.LSTM(512, 128, bidirectional=True)
        self.fc_cls = nn.Linear(256, n_anchor * 2)
        self.fc_v = nn.Linear(256, n_anchor * 2)
        self.fc_o = nn.Linear(256, n_anchor)

    @staticmethod
    def _build_backbone(pretrained: bool):
        if not pretrained:
            return torchvision.models.vgg16_bn(weights=None).features[:-1]
        try:
            weights_enum = getattr(torchvision.models, "VGG16_BN_Weights", None)
            if weights_enum is not None:
                return torchvision.models.vgg16_bn(
                    weights=weights_enum.IMAGENET1K_V1
                ).features[:-1]
            return torchvision.models.vgg16_bn(pretrained=True).features[:-1]
        except Exception as e:
            print(f"Warning: failed to load pretrained VGG16_BN ({e}), using random init.")
            return torchvision.models.vgg16_bn(weights=None).features[:-1]

    def forward(self, x):
        x = self.features(x)
        x = self.slider(x)

        x = x.permute(2, 3, 0, 1)
        rows = []
        for row_feat in x:
            out, _ = self.bilstm(row_feat)
            rows.append(out)
        x = torch.stack(rows).permute(2, 0, 1, 3)

        y_cls = self.fc_cls(x).reshape(*x.shape[:3], self.n_anchor, 2)
        y_v = self.fc_v(x).reshape(*x.shape[:3], self.n_anchor, 2)
        y_o = self.fc_o(x)
        return y_cls, y_v, y_o


def compute_loss(y_cls, y_v, y_o, tgt_cls, tgt_v, idx_v, tgt_o, idx_o):
    ce = nn.CrossEntropyLoss()
    l1 = nn.SmoothL1Loss(reduction="sum")

    loss_cls = ce(y_cls.view(-1, 2), tgt_cls.view(-1))

    if idx_v.any():
        denom_v = max(int(idx_v.sum().item() // 2), 1)
        loss_v = l1(y_v[idx_v], tgt_v[idx_v]) / denom_v
    else:
        loss_v = torch.zeros((), device=y_cls.device)

    if idx_o.any():
        denom_o = max(int(idx_o.sum().item()), 1)
        loss_o = 2.0 * l1(y_o[idx_o], tgt_o[idx_o]) / denom_o
    else:
        loss_o = torch.zeros((), device=y_cls.device)

    return loss_cls + loss_v + loss_o, loss_cls, loss_v, loss_o


def run_epoch(model, loader, optimizer, device, train=True, max_batches=0):
    if train:
        model.train()
    else:
        model.eval()

    running = 0.0
    count = 0
    with torch.set_grad_enabled(train):
        for i, sample in enumerate(loader, start=1):
            if max_batches > 0 and i > max_batches:
                break
            img, tgt_cls, tgt_v, idx_v, tgt_o, idx_o = [x.to(device) for x in sample]
            y_cls, y_v, y_o = model(img)
            loss, loss_cls, loss_v, loss_o = compute_loss(
                y_cls, y_v, y_o, tgt_cls, tgt_v, idx_v, tgt_o, idx_o
            )

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            running += float(loss.item())
            count += 1
            print(
                f"[{'train' if train else 'valid'}] step={i} "
                f"loss={loss.item():.4f} cls={loss_cls.item():.4f} "
                f"v={loss_v.item():.4f} o={loss_o.item():.4f}"
            )

    return running / max(count, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/dataset/mlt")
    parser.add_argument("--checkpoint-dir", default="checkpoints_mlt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--n-anchor", type=int, default=10)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-pretrained-backbone", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-valid-batches", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    dataset = CtpnDataset(args.data_dir, n_anchor=args.n_anchor)
    val_size = max(1, int(len(dataset) * args.val_ratio))
    train_size = len(dataset) - val_size
    train_set, valid_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = CtpnModel(
        n_anchor=args.n_anchor, pretrained_backbone=not args.no_pretrained_backbone
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    print(f"dataset={len(dataset)} train={len(train_set)} valid={len(valid_set)}")
    print(f"device={device}")

    for epoch in range(1, args.epochs + 1):
        print(f"==== epoch {epoch}/{args.epochs} ====")
        train_loss = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            train=True,
            max_batches=args.max_train_batches,
        )
        valid_loss = run_epoch(
            model,
            valid_loader,
            optimizer,
            device,
            train=False,
            max_batches=args.max_valid_batches,
        )
        scheduler.step()

        ckpt_path = os.path.join(args.checkpoint_dir, f"ctpn_epoch_{epoch:03d}.pth")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "valid_loss": valid_loss,
            },
            ckpt_path,
        )
        print(
            f"saved={ckpt_path} train_loss={train_loss:.4f} valid_loss={valid_loss:.4f}"
        )


if __name__ == "__main__":
    main()

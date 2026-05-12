#!/usr/bin/env python
import argparse
import logging
import pathlib
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from calli_sae.config import apply_overrides, expand_path, load_config, save_json
from calli_sae.data import (
    CalligraphyCsvDataset,
    assert_nonempty_rows,
    default_train_csv,
    default_val_csv,
    read_csv_rows,
)
from calli_sae.train_utils import save_reconstruction_grid, set_seed

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


OVERRIDE_KEYS = (
    "dataset_root",
    "train_csv",
    "val_csv",
    "image_column",
    "image_size",
    "center_crop",
    "pretrained_model_name_or_path",
    "pretrained_subfolder",
    "output_root",
    "run_name",
    "epochs",
    "max_steps",
    "train_batch_size",
    "val_batch_size",
    "learning_rate",
    "weight_decay",
    "gradient_accumulation_steps",
    "max_grad_norm",
    "mixed_precision",
    "num_workers",
    "seed",
    "recon_loss",
    "kl_weight",
    "sample_latents",
    "sample_every",
    "sample_count",
    "validation_loss_every",
    "validation_loss_batches",
    "checkpoint_every",
    "resume_checkpoint",
    "cache_latents_after_training",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finetune a pretrained AutoencoderKL on calligraphy images.")
    parser.add_argument("--config", default="configs/vae_finetune.yaml")

    parser.add_argument("--dataset_root", default=None)
    parser.add_argument("--train_csv", default=None)
    parser.add_argument("--val_csv", default=None)
    parser.add_argument("--image_column", default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--center_crop", action=argparse.BooleanOptionalAction, default=None)

    parser.add_argument("--pretrained_model_name_or_path", default=None)
    parser.add_argument("--pretrained_subfolder", default=None)
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--run_name", default=None)

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--train_batch_size", type=int, default=None)
    parser.add_argument("--val_batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument("--mixed_precision", choices=("no", "fp16", "bf16"), default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--recon_loss", choices=("l1", "mse"), default=None)
    parser.add_argument("--kl_weight", type=float, default=None)
    parser.add_argument("--sample_latents", action=argparse.BooleanOptionalAction, default=None)

    parser.add_argument("--sample_every", type=int, default=None)
    parser.add_argument("--sample_count", type=int, default=None)
    parser.add_argument("--validation_loss_every", type=int, default=None)
    parser.add_argument("--validation_loss_batches", type=int, default=None)
    parser.add_argument("--checkpoint_every", type=int, default=None)
    parser.add_argument("--resume_checkpoint", default=None)
    parser.add_argument("--cache_latents_after_training", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def resolve_config(args: argparse.Namespace) -> Dict:
    config = load_config(args.config)
    config = apply_overrides(config, vars(args), OVERRIDE_KEYS)

    config.setdefault("dataset_root", "~/Code/calligraphy_project")
    config.setdefault("image_column", "path")
    config.setdefault("image_size", 64)
    config.setdefault("center_crop", True)
    config.setdefault("pretrained_model_name_or_path", "stabilityai/sd-vae-ft-mse")
    config.setdefault("pretrained_subfolder", None)
    config.setdefault("output_root", "~/Code/calli_sae_results")
    config.setdefault("run_name", None)

    config.setdefault("epochs", 50)
    config.setdefault("max_steps", None)
    config.setdefault("train_batch_size", 32)
    config.setdefault("val_batch_size", 8)
    config.setdefault("learning_rate", 1e-5)
    config.setdefault("weight_decay", 1e-6)
    config.setdefault("gradient_accumulation_steps", 1)
    config.setdefault("max_grad_norm", 1.0)
    config.setdefault("mixed_precision", "bf16")
    config.setdefault("num_workers", 8)
    config.setdefault("seed", 42)

    config.setdefault("recon_loss", "l1")
    config.setdefault("kl_weight", 1e-6)
    config.setdefault("sample_latents", True)

    config.setdefault("sample_every", 1000)
    config.setdefault("sample_count", 8)
    config.setdefault("validation_loss_every", 1000)
    config.setdefault("validation_loss_batches", 0)
    config.setdefault("checkpoint_every", 5000)
    config.setdefault("resume_checkpoint", None)
    config.setdefault("cache_latents_after_training", False)
    config.setdefault("latent_cache_batch_size", 128)

    config.setdefault("adam_beta1", 0.9)
    config.setdefault("adam_beta2", 0.999)
    config.setdefault("adam_eps", 1e-8)
    config.setdefault("allow_tf32", True)

    return config


def load_autoencoder_kl(config: Dict):
    try:
        from diffusers.models import AutoencoderKL
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("VAE finetuning requires diffusers with AutoencoderKL available.") from exc

    load_kwargs = {}
    if config.get("pretrained_subfolder"):
        load_kwargs["subfolder"] = config["pretrained_subfolder"]

    model = AutoencoderKL.from_pretrained(config["pretrained_model_name_or_path"], **load_kwargs)
    model.train()
    return model


def compute_vae_loss(vae, images: torch.Tensor, config: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    posterior = vae.encode(images).latent_dist
    latents = posterior.sample() if config["sample_latents"] else posterior.mode()
    reconstructions = vae.decode(latents).sample

    if config["recon_loss"] == "l1":
        recon_loss = F.l1_loss(reconstructions.float(), images.float())
    elif config["recon_loss"] == "mse":
        recon_loss = F.mse_loss(reconstructions.float(), images.float())
    else:
        raise ValueError(f"Unsupported recon_loss: {config['recon_loss']}")

    kl_loss = posterior.kl().mean()
    loss = recon_loss + float(config["kl_weight"]) * kl_loss
    return loss, recon_loss.detach(), kl_loss.detach(), reconstructions.detach()


def checkpoint_step(checkpoint_path: Path) -> int:
    name = checkpoint_path.name
    if name.startswith("step_") and name.endswith(".pt"):
        return int(name[len("step_") : -len(".pt")])
    return -1


def find_latest_checkpoint(path: Path) -> Path:
    path = path.expanduser()
    if path.is_file():
        return path

    latest_txt = path / "latest_checkpoint.txt"
    if latest_txt.exists():
        latest_path = Path(latest_txt.read_text(encoding="utf-8").strip()).expanduser()
        if latest_path.exists():
            return latest_path

    candidates = []
    if path.is_dir():
        candidates.extend(path.glob("step_*.pt"))
        candidates.extend(path.glob("checkpoints/step_*.pt"))
        candidates.extend(path.glob("vae/*/checkpoints/step_*.pt"))
        candidates.extend(path.glob("*/checkpoints/step_*.pt"))

    if not candidates:
        raise FileNotFoundError(f"No VAE checkpoint matching step_*.pt found under {path}.")
    return max(candidates, key=lambda candidate: (checkpoint_step(candidate), candidate.stat().st_mtime))


def save_checkpoint(accelerator, vae, optimizer, checkpoint_dir: Path, *, epoch: int, step: int, config: Dict) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "vae": accelerator.unwrap_model(vae).state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "config": config,
    }
    checkpoint_path = checkpoint_dir / f"step_{step:06d}.pt"
    accelerator.save(checkpoint, checkpoint_path)
    (checkpoint_dir / "latest_checkpoint.txt").write_text(str(checkpoint_path), encoding="utf-8")
    return checkpoint_path


@torch.no_grad()
def compute_validation_loss(vae, dataloader, accelerator, config: Dict, max_batches: int = 0) -> Dict[str, float]:
    was_training = vae.training
    vae.eval()

    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        images = batch["image"]
        loss, recon_loss, kl_loss, _ = compute_vae_loss(vae, images, config)

        gathered = accelerator.gather_for_metrics(
            torch.stack([loss.detach(), recon_loss.detach(), kl_loss.detach()]).reshape(1, 3)
        )
        means = gathered.mean(dim=0)
        total_loss += means[0].item()
        total_recon += means[1].item()
        total_kl += means[2].item()
        total_batches += 1

    if was_training:
        vae.train()

    if total_batches == 0:
        raise ValueError("Validation loader is empty, cannot compute validation loss.")

    return {
        "loss": total_loss / total_batches,
        "recon_loss": total_recon / total_batches,
        "kl_loss": total_kl / total_batches,
    }


@torch.no_grad()
def save_validation_reconstructions(vae, fixed_images: torch.Tensor, accelerator, samples_dir: Path, step: int, config: Dict) -> None:
    was_training = vae.training
    vae.eval()

    images = fixed_images.to(accelerator.device)
    posterior = vae.encode(images).latent_dist
    latents = posterior.mode()
    reconstructions = vae.decode(latents).sample
    save_reconstruction_grid(
        images,
        reconstructions,
        samples_dir / f"step_{step:06d}.png",
        max_samples=int(config["sample_count"]),
    )

    if was_training:
        vae.train()


@torch.no_grad()
def cache_latents(vae, dataset, output_path: Path, *, batch_size: int, num_workers: int, device: torch.device) -> None:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    scaling_factor = float(getattr(getattr(vae, "config", None), "scaling_factor", 0.18215))

    latents = []
    paths = []
    row_indices = []

    was_training = vae.training
    vae.eval()
    for batch in tqdm(loader, desc=f"cache latents {output_path.name}"):
        images = batch["image"].to(device)
        posterior = vae.encode(images).latent_dist
        batch_latents = posterior.mode() * scaling_factor
        latents.append(batch_latents.detach().cpu().to(torch.float16))
        paths.extend(batch["path"])
        row_indices.extend([int(idx) for idx in batch["row_index"]])

    if was_training:
        vae.train()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "latents": torch.cat(latents, dim=0) if latents else torch.empty(0),
            "paths": paths,
            "row_indices": row_indices,
            "scaling_factor": scaling_factor,
        },
        output_path,
    )


def main() -> None:
    args = parse_args()
    config = resolve_config(args)
    set_seed(int(config["seed"]))

    if bool(config.get("allow_tf32", True)) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    dataset_root = expand_path(config["dataset_root"])
    if dataset_root is None:
        raise ValueError("dataset_root must be set.")

    train_csv = expand_path(config.get("train_csv")) or default_train_csv(dataset_root)
    val_csv = expand_path(config.get("val_csv")) or default_val_csv(dataset_root)
    if not train_csv.exists():
        raise FileNotFoundError(f"Train CSV not found: {train_csv}")
    if not val_csv.exists():
        raise FileNotFoundError(f"Validation CSV not found: {val_csv}")

    output_base = expand_path(config["output_root"])
    if output_base is None:
        raise ValueError("output_root must be set.")

    run_id = str(config["run_name"] or time.strftime("%Y%m%d-%H%M%S"))
    run_dir = output_base / "vae" / run_id
    samples_dir = run_dir / "samples"
    checkpoints_dir = run_dir / "checkpoints"
    tensorboard_dir = run_dir / "tensorboard"

    train_rows = read_csv_rows(train_csv)
    val_rows = read_csv_rows(val_csv)
    assert_nonempty_rows(train_rows, train_csv)
    assert_nonempty_rows(val_rows, val_csv)

    train_dataset = CalligraphyCsvDataset(
        train_rows,
        dataset_root,
        int(config["image_size"]),
        image_column=str(config["image_column"]),
        center_crop=bool(config["center_crop"]),
        normalize=True,
    )
    val_dataset = CalligraphyCsvDataset(
        val_rows,
        dataset_root,
        int(config["image_size"]),
        image_column=str(config["image_column"]),
        center_crop=bool(config["center_crop"]),
        normalize=True,
    )

    fixed_val_loader = DataLoader(val_dataset, batch_size=int(config["sample_count"]), shuffle=False, drop_last=False)
    fixed_val_images = next(iter(fixed_val_loader))["image"]

    from accelerate import Accelerator
    from torch.utils.tensorboard import SummaryWriter

    accelerator = Accelerator(
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        mixed_precision=str(config["mixed_precision"]),
    )

    writer = None
    if accelerator.is_main_process:
        run_dir.mkdir(parents=True, exist_ok=True)
        samples_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        save_json(run_dir / "config.resolved.json", config)
        save_json(
            run_dir / "data_info.json",
            {
                "dataset_root": dataset_root,
                "train_csv": train_csv,
                "val_csv": val_csv,
                "train_rows": len(train_dataset),
                "val_rows": len(val_dataset),
            },
        )
        writer = SummaryWriter(log_dir=str(tensorboard_dir))

    vae = load_autoencoder_kl(config)
    optimizer = AdamW(
        [param for param in vae.parameters() if param.requires_grad],
        lr=float(config["learning_rate"]),
        betas=(float(config["adam_beta1"]), float(config["adam_beta2"])),
        eps=float(config["adam_eps"]),
        weight_decay=float(config["weight_decay"]),
    )

    start_epoch = 0
    global_step = 0
    if config.get("resume_checkpoint"):
        resume_path = find_latest_checkpoint(Path(str(config["resume_checkpoint"])))
        logging.info("Loading VAE checkpoint from %s", resume_path)
        torch.serialization.add_safe_globals([pathlib.PosixPath])
        checkpoint = torch.load(resume_path, map_location="cpu")
        vae.load_state_dict(checkpoint["vae"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        global_step = int(checkpoint.get("step", 0))
        if accelerator.is_main_process:
            (run_dir / "resumed_from.txt").write_text(str(resume_path), encoding="utf-8")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=int(config["train_batch_size"]),
        shuffle=True,
        drop_last=True,
        num_workers=int(config["num_workers"]),
        pin_memory=torch.cuda.is_available(),
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=int(config["val_batch_size"]),
        shuffle=False,
        drop_last=False,
        num_workers=int(config["num_workers"]),
        pin_memory=torch.cuda.is_available(),
    )

    vae, optimizer, train_dataloader, val_dataloader = accelerator.prepare(vae, optimizer, train_dataloader, val_dataloader)

    if accelerator.is_main_process:
        logging.info("Run directory: %s", run_dir)
        logging.info("Training rows: %s; validation rows: %s", len(train_dataset), len(val_dataset))
        logging.info("Pretrained VAE: %s", config["pretrained_model_name_or_path"])

    stop_training = False
    last_epoch = start_epoch - 1

    for epoch in range(start_epoch, int(config["epochs"])):
        last_epoch = epoch
        vae.train()
        progress = tqdm(train_dataloader, disable=not accelerator.is_local_main_process)
        progress.set_description(f"epoch {epoch}")

        for batch in progress:
            grad_norm = None
            with accelerator.accumulate(vae):
                images = batch["image"]
                loss, recon_loss, kl_loss, _ = compute_vae_loss(vae, images, config)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(vae.parameters(), float(config["max_grad_norm"]))
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress.set_postfix(loss=f"{loss.detach().float().item():.4f}", step=global_step)

                if accelerator.is_main_process and writer is not None:
                    writer.add_scalar("train/loss", loss.detach().float().item(), global_step)
                    writer.add_scalar("train/recon_loss", recon_loss.float().item(), global_step)
                    writer.add_scalar("train/kl_loss", kl_loss.float().item(), global_step)
                    writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
                    if grad_norm is not None:
                        grad_value = grad_norm.detach().float().item() if torch.is_tensor(grad_norm) else float(grad_norm)
                        writer.add_scalar("train/grad_norm", grad_value, global_step)

                if int(config["validation_loss_every"]) > 0 and global_step % int(config["validation_loss_every"]) == 0:
                    accelerator.wait_for_everyone()
                    metrics = compute_validation_loss(
                        vae,
                        val_dataloader,
                        accelerator,
                        config,
                        max_batches=int(config["validation_loss_batches"]),
                    )
                    if accelerator.is_main_process:
                        logging.info("step %s validation_loss %.6f", global_step, metrics["loss"])
                        if writer is not None:
                            for key, value in metrics.items():
                                writer.add_scalar(f"validation/{key}", value, global_step)
                            writer.flush()

                if int(config["sample_every"]) > 0 and global_step % int(config["sample_every"]) == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        save_validation_reconstructions(
                            accelerator.unwrap_model(vae),
                            fixed_val_images,
                            accelerator,
                            samples_dir,
                            global_step,
                            config,
                        )

                if int(config["checkpoint_every"]) > 0 and global_step % int(config["checkpoint_every"]) == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        checkpoint_path = save_checkpoint(
                            accelerator,
                            vae,
                            optimizer,
                            checkpoints_dir,
                            epoch=epoch,
                            step=global_step,
                            config=config,
                        )
                        logging.info("Saved checkpoint to %s", checkpoint_path)

                if config["max_steps"] is not None and global_step >= int(config["max_steps"]):
                    stop_training = True
                    break

        if stop_training:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        checkpoint_path = save_checkpoint(
            accelerator,
            vae,
            optimizer,
            checkpoints_dir,
            epoch=last_epoch,
            step=global_step,
            config=config,
        )
        logging.info("Saved final checkpoint to %s", checkpoint_path)

        final_vae_dir = run_dir / "vae"
        accelerator.unwrap_model(vae).save_pretrained(final_vae_dir)
        logging.info("Saved finetuned VAE to %s", final_vae_dir)

        save_validation_reconstructions(
            accelerator.unwrap_model(vae),
            fixed_val_images,
            accelerator,
            samples_dir,
            global_step,
            config,
        )

        if bool(config["cache_latents_after_training"]):
            cache_dir = run_dir / "latents"
            unwrapped = accelerator.unwrap_model(vae).to(accelerator.device)
            cache_latents(
                unwrapped,
                train_dataset,
                cache_dir / "train_latents.pt",
                batch_size=int(config["latent_cache_batch_size"]),
                num_workers=int(config["num_workers"]),
                device=accelerator.device,
            )
            cache_latents(
                unwrapped,
                val_dataset,
                cache_dir / "val_latents.pt",
                batch_size=int(config["latent_cache_batch_size"]),
                num_workers=int(config["num_workers"]),
                device=accelerator.device,
            )

        if writer is not None:
            writer.close()


if __name__ == "__main__":
    main()

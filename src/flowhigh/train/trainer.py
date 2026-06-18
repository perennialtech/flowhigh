from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from ..audio import MelCodec
from ..config import FlowHighConfig
from ..flow import FlowMatcher
from .data import get_dataloader
from .optimizer import get_optimizer


def cycle(dataloader):
    while True:
        for batch in dataloader:
            yield batch


class FlowHighTrainer:
    def __init__(
        self,
        *,
        matcher: FlowMatcher,
        codec: MelCodec,
        dataset,
        config: FlowHighConfig,
        accelerate_kwargs: dict[str, Any] | None = None,
    ):
        train_config = config.train

        if train_config.grad_accum_steps <= 0:
            raise ValueError("grad_accum_steps must be greater than 0")

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        self.accelerator = Accelerator(
            kwargs_handlers=[ddp_kwargs],
            **({} if accelerate_kwargs is None else accelerate_kwargs),
        )

        self.matcher = matcher
        self.codec = codec.eval()
        self.config = config
        self.steps = 0

        self.max_steps = train_config.steps
        self.warmup_steps = train_config.warmup_steps
        self.lr = train_config.lr
        self.initial_lr = train_config.initial_lr
        self.grad_accum_steps = train_config.grad_accum_steps
        self.max_grad_norm = train_config.max_grad_norm
        self.log_every = train_config.log_every
        self.save_every = train_config.save_every
        self.weighted_loss = train_config.weighted_loss
        self.segment_seconds = train_config.segment_seconds
        self.results_folder = Path(train_config.results_folder)
        self.results_folder.mkdir(parents=True, exist_ok=True)

        self.optim = get_optimizer(
            matcher.parameters(),
            lr=train_config.lr,
            wd=train_config.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optim,
            T_max=max(1, train_config.steps - train_config.warmup_steps),
        )

        self.dataloader = get_dataloader(
            dataset,
            batch_size=train_config.batch_size,
            shuffle=True,
            drop_last=True,
        )

        self.matcher, self.optim, self.scheduler, self.dataloader = (
            self.accelerator.prepare(
                self.matcher,
                self.optim,
                self.scheduler,
                self.dataloader,
            )
        )

        self.codec.to(self.device)
        self.dataloader_iter = cycle(self.dataloader)

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def _set_lr(self, lr: float):
        for group in self.optim.param_groups:
            group["lr"] = lr

    def _warmup_lr(self, step: int) -> float:
        if self.warmup_steps <= 0 or step >= self.warmup_steps:
            return self.lr

        return self.initial_lr + (self.lr - self.initial_lr) * step / self.warmup_steps

    def _mel_lengths(self, audio_lengths: torch.Tensor) -> torch.Tensor:
        hop = self.codec.hop_length
        win = self.codec.win_length
        return torch.ceil((audio_lengths - win) / hop + 1).long().clamp_min(1)

    def _crop_mels(
        self,
        target: torch.Tensor,
        condition: torch.Tensor,
        lengths: torch.Tensor,
    ):
        if self.segment_seconds <= 0:
            return target, condition, lengths

        frames = int(
            self.segment_seconds * self.codec.sampling_rate / self.codec.hop_length
        )
        total_frames = min(target.shape[1], condition.shape[1])
        frames = min(frames, total_frames)

        if frames <= 0 or frames >= total_frames:
            return target, condition, lengths.clamp(max=total_frames)

        batch, _, channels = target.shape
        target_cut = torch.zeros(
            batch, frames, channels, device=target.device, dtype=target.dtype
        )
        cond_cut = torch.zeros(
            batch, frames, channels, device=condition.device, dtype=condition.dtype
        )
        new_lengths = torch.zeros(batch, device=lengths.device, dtype=torch.long)

        for index in range(batch):
            valid = int(lengths[index].clamp(min=0, max=total_frames).item())
            cut_len = min(valid, frames)
            max_offset = max(valid - cut_len, 0)
            offset = (
                int(torch.randint(0, max_offset + 1, (), device=lengths.device).item())
                if max_offset > 0
                else 0
            )

            if cut_len > 0:
                target_cut[index, :cut_len] = target[index, offset : offset + cut_len]
                cond_cut[index, :cut_len] = condition[index, offset : offset + cut_len]

            new_lengths[index] = cut_len

        return target_cut, cond_cut, new_lengths

    def save(self, path: str | Path):
        package = {
            "step": self.steps,
            "model": self.accelerator.get_state_dict(self.matcher),
            "optim": self.optim.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "config": self.config.to_dict(),
        }
        torch.save(package, path)

    def load(self, path: str | Path):
        package = torch.load(path, map_location="cpu")
        self.accelerator.unwrap_model(self.matcher).load_state_dict(package["model"])
        self.optim.load_state_dict(package["optim"])
        self.scheduler.load_state_dict(package["scheduler"])
        self.steps = int(package.get("step", 0))

    def train_step(self):
        step = self.steps
        self.matcher.train()
        self._set_lr(self._warmup_lr(step))

        total_loss = 0.0

        for accum_step in range(self.grad_accum_steps):
            is_last = accum_step == self.grad_accum_steps - 1
            sync_context = (
                nullcontext() if is_last else self.accelerator.no_sync(self.matcher)
            )

            target_audio, condition_audio, audio_lengths = next(self.dataloader_iter)
            target_audio = target_audio.to(
                self.device, dtype=torch.float32, non_blocking=True
            )
            condition_audio = condition_audio.to(
                self.device, dtype=torch.float32, non_blocking=True
            )
            audio_lengths = audio_lengths.to(
                self.device, dtype=torch.long, non_blocking=True
            )

            condition_audio = condition_audio / condition_audio.abs().amax(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-8)

            with torch.no_grad():
                target_mel = self.codec.encode(target_audio)
                condition_mel = self.codec.encode(condition_audio)
                mel_lengths = self._mel_lengths(audio_lengths)
                target_mel, condition_mel, mel_lengths = self._crop_mels(
                    target_mel,
                    condition_mel,
                    mel_lengths,
                )

            with self.accelerator.autocast(), sync_context:
                loss = self.matcher.loss(
                    target_mel,
                    condition_mel,
                    lengths=mel_lengths,
                    weighted_loss=self.weighted_loss,
                )
                self.accelerator.backward(loss / self.grad_accum_steps)

            total_loss += float(loss.item()) / self.grad_accum_steps

        if self.max_grad_norm is not None:
            self.accelerator.clip_grad_norm_(
                self.matcher.parameters(), self.max_grad_norm
            )

        self.optim.step()
        self.optim.zero_grad()

        if step >= self.warmup_steps:
            self.scheduler.step()

        if step % self.log_every == 0:
            self.accelerator.print(f"step {step}: loss {total_loss:.4f}")

        self.accelerator.log({"train_loss": total_loss}, step=step)

        if self.is_main and step % self.save_every == 0:
            path = self.results_folder / f"flowhigh.{step}.pt"
            self.save(path)
            self.accelerator.print(f"saved checkpoint to {path}")

        self.steps += 1
        return {"loss": total_loss}

    def train(self, resume_from: str | Path | None = None):
        if resume_from is not None:
            self.load(resume_from)
            self.accelerator.print(f"resumed from {resume_from}")
        else:
            self.accelerator.print("starting training from scratch")

        for _ in tqdm(range(self.steps, self.max_steps), desc="Training FlowHigh"):
            self.train_step()

        self.accelerator.end_training()

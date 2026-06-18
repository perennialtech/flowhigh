import torch
import torch.nn as nn
from torchaudio.transforms import Spectrogram, InverseSpectrogram


class PostProcessing(nn.Module):
    def __init__(self):
        super().__init__()
        self.stft = Spectrogram(
            2048, hop_length=480, win_length=2048, power=None, pad_mode="constant"
        )
        self.istft = InverseSpectrogram(
            2048, hop_length=480, win_length=2048, pad_mode="constant"
        )

    def get_cutoff_index(self, spec, threshold=0.99):
        energy = spec.abs().sum(dim=-1).cumsum(dim=-1)
        cutoff = torch.searchsorted(
            energy.contiguous(),
            (energy[:, -1:] * threshold).contiguous(),
        ).squeeze(-1)
        return cutoff.clamp(0, spec.size(1)).to(torch.long)

    def post_processing(self, pred, src, length):
        # pred, src : [1, Time]
        assert len(pred.shape) == 2 and len(src.shape) == 2

        spec_pred = self.stft(pred)  # [B, Channel, Time]
        spec_src = self.stft(src)  # [B, Channel, Time]

        # energy cutoff of spec_src
        cr = self.get_cutoff_index(spec_src)

        # Replacement
        min_time_dim = min(spec_pred.size(-1), spec_src.size(-1))

        spec_pred = spec_pred[:, :, :min_time_dim]
        spec_src = spec_src[:, :, :min_time_dim]

        freq_idx = torch.arange(spec_pred.size(1), device=spec_pred.device)
        use_src = freq_idx.view(1, -1, 1) < cr.view(-1, 1, 1)
        spec_result = torch.where(use_src, spec_src, spec_pred)

        audio = self.istft(spec_result, length=length)
        audio = audio / audio.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) * 0.99
        return audio

    def post_processing_with_phase(self, pred, src, length):
        # pred, src : [1, Time]
        assert len(pred.shape) == 2 and len(src.shape) == 2

        spec_pred = self.stft(pred)  # [1, Channel, Time]
        spec_src = self.stft(src)  # [1, Channel, Time]

        batch = spec_pred.shape[0]
        cr = int(self.get_cutoff_index(spec_src)[0].item())

        # Replacement
        spec_result = torch.empty_like(spec_pred)
        min_time_dim = min(spec_pred.size(-1), spec_src.size(-1))

        spec_result = spec_result[:, :, :min_time_dim]
        spec_pred = spec_pred[:, :, :min_time_dim]
        spec_src = spec_src[:, :, :min_time_dim]

        pred_mag = torch.abs(spec_pred[:, cr:, ...])
        src_phase = torch.angle(spec_src[:, :cr, ...])

        # Replicate phase information to match the dimensions of spec_pred
        num_repeats = (spec_pred.size(1) - cr) // cr + 1
        replicate_phase = src_phase.repeat(batch, num_repeats, 1)
        replicate_phase = replicate_phase[:, -(spec_pred.size(1) - cr) :, ...]
        print(pred_mag.size())
        print(replicate_phase.size())

        x = torch.cos(replicate_phase)
        y = torch.sin(replicate_phase)

        spec_result[:, cr:, ...] = pred_mag * (x + 1j * y)
        spec_result[:, :cr, ...] = spec_src[:, :cr, ...]

        audio = self.istft(spec_result, length=length)
        audio = audio / audio.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) * 0.99
        return audio, src_phase, replicate_phase

    # For mel repalcement
    def _locate_cutoff_freq(self, stft, percentile=0.985):
        squeeze = stft.ndim == 2

        if squeeze:
            stft = stft.unsqueeze(0)

        energy = stft.abs().sum(dim=1).cumsum(dim=-1)
        cutoff = torch.searchsorted(
            energy.contiguous(),
            (energy[:, -1:] * percentile).contiguous(),
        ).squeeze(-1)
        cutoff = cutoff.clamp(0, stft.size(-1)).to(torch.long)
        return cutoff[0] if squeeze else cutoff

    def mel_replace_ops(self, samples, input):
        cutoff = self._locate_cutoff_freq(torch.exp(input))
        mel_idx = torch.arange(samples.size(-1), device=samples.device)
        use_input = mel_idx.view(1, 1, -1) < cutoff.view(-1, 1, 1)
        return torch.where(use_input, input, samples)

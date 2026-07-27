import os
import sys
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from librosa.filters import mel
from torch import nn

now_dir = os.getcwd()
sys.path.append(now_dir)


SAMPLE_RATE = 16000

N_CLASS = 360

N_MELS = 128
MEL_FMIN = 30
MEL_FMAX = SAMPLE_RATE // 2
WINDOW_LENGTH = 1024
CONST = 1997.3794084376191


def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DSConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None, act=True):
        super().__init__()
        self.dwconv = nn.Conv2d(c1, c1, k, s, autopad(k, p), groups=c1, bias=False)
        self.pwconv = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.pwconv(self.dwconv(x))))


class DS_Bottleneck(nn.Module):
    def __init__(self, c1, c2, k=3, shortcut=True):
        super().__init__()
        c_ = c1
        self.dsconv1 = DSConv(c1, c_, k=3, s=1)
        self.dsconv2 = DSConv(c_, c2, k=k, s=1)
        self.shortcut = shortcut and c1 == c2

    def forward(self, x):
        return (
            x + self.dsconv2(self.dsconv1(x))
            if self.shortcut
            else self.dsconv2(self.dsconv1(x))
        )


class DS_C3k(nn.Module):
    def __init__(self, c1, c2, n=1, k=3, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        self.m = nn.Sequential(
            *[DS_Bottleneck(c_, c_, k=k, shortcut=True) for _ in range(n)]
        )

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class DS_C3k2(nn.Module):
    def __init__(self, c1, c2, n=1, k=3, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.m = DS_C3k(c_, c_, n=n, k=k, e=1.0)
        self.cv2 = Conv(c_, c2, 1, 1)

    def forward(self, x):
        x_ = self.cv1(x)
        x_ = self.m(x_)
        return self.cv2(x_)


class AdaptiveHyperedgeGeneration(nn.Module):
    def __init__(self, in_channels, num_hyperedges, num_heads):
        super().__init__()
        self.num_hyperedges = num_hyperedges
        self.num_heads = num_heads
        self.head_dim = max(1, in_channels // num_heads)

        self.global_proto = nn.Parameter(torch.randn(num_hyperedges, in_channels))
        self.context_mapper = nn.Linear(
            2 * in_channels, num_hyperedges * in_channels, bias=False
        )
        self.query_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.scale = self.head_dim**-0.5

    def forward(self, x):
        B, N, C = x.shape
        f_avg = F.adaptive_avg_pool1d(x.permute(0, 2, 1), 1).squeeze(-1)
        f_max = F.adaptive_max_pool1d(x.permute(0, 2, 1), 1).squeeze(-1)
        f_ctx = torch.cat((f_avg, f_max), dim=1)

        delta_P = self.context_mapper(f_ctx).view(B, self.num_hyperedges, C)
        P = self.global_proto.unsqueeze(0) + delta_P

        z = self.query_proj(x)
        z = z.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        P = P.view(B, self.num_hyperedges, self.num_heads, self.head_dim).permute(
            0, 2, 3, 1
        )

        sim = (z @ P) * self.scale
        s_bar = sim.mean(dim=1)
        A = F.softmax(s_bar.permute(0, 2, 1), dim=-1)
        return A


class HypergraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.W_e = nn.Linear(in_channels, in_channels, bias=False)
        self.W_v = nn.Linear(in_channels, out_channels, bias=False)
        self.act = nn.SiLU()

    def forward(self, x, A):
        f_m = torch.bmm(A, x)
        f_m = self.act(self.W_e(f_m))
        x_out = torch.bmm(A.transpose(1, 2), f_m)
        x_out = self.act(self.W_v(x_out))
        return x + x_out


class AdaptiveHypergraphComputation(nn.Module):
    def __init__(self, in_channels, out_channels, num_hyperedges, num_heads):
        super().__init__()
        self.adaptive_hyperedge_gen = AdaptiveHyperedgeGeneration(
            in_channels, num_hyperedges, num_heads
        )
        self.hypergraph_conv = HypergraphConvolution(in_channels, out_channels)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).permute(0, 2, 1)
        A = self.adaptive_hyperedge_gen(x_flat)
        x_out_flat = self.hypergraph_conv(x_flat, A)
        x_out = x_out_flat.permute(0, 2, 1).view(B, -1, H, W)
        return x_out


class C3AH(nn.Module):
    def __init__(self, c1, c2, num_hyperedges, num_heads, e=0.5):
        super().__init__()
        c_ = int(c1 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.ahc = AdaptiveHypergraphComputation(c_, c_, num_hyperedges, num_heads)
        self.cv3 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x_lateral = self.cv1(x)
        x_ahc = self.ahc(self.cv2(x))
        return self.cv3(torch.cat((x_ahc, x_lateral), dim=1))


class HyperACE(nn.Module):
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        num_hyperedges=16,
        num_heads=8,
        k=2,
        l=1,
        c_h=0.5,
        c_l=0.25,
    ):
        super().__init__()

        c2, c3, c4, c5 = in_channels
        c_mid = c4
        self.fuse_conv = Conv(c2 + c3 + c4 + c5, c_mid, 1, 1)

        self.c_h = int(c_mid * c_h)
        self.c_l = int(c_mid * c_l)
        self.c_s = c_mid - self.c_h - self.c_l

        self.high_order_branch = nn.ModuleList(
            [
                C3AH(
                    self.c_h,
                    self.c_h,
                    num_hyperedges=num_hyperedges,
                    num_heads=num_heads,
                    e=1.0,
                )
                for _ in range(k)
            ]
        )
        self.high_order_fuse = Conv(self.c_h * k, self.c_h, 1, 1)

        self.low_order_branch = nn.Sequential(
            *[DS_C3k(self.c_l, self.c_l, n=1, k=3, e=1.0) for _ in range(l)]
        )

        self.final_fuse = Conv(self.c_h + self.c_l + self.c_s, out_channels, 1, 1)

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        B2, B3, B4, B5 = x
        B, _, H4, W4 = B4.shape
        B2_resized = F.interpolate(
            B2, size=(H4, W4), mode="bilinear", align_corners=False
        )
        B3_resized = F.interpolate(
            B3, size=(H4, W4), mode="bilinear", align_corners=False
        )
        B5_resized = F.interpolate(
            B5, size=(H4, W4), mode="bilinear", align_corners=False
        )

        x_b = self.fuse_conv(torch.cat((B2_resized, B3_resized, B4, B5_resized), dim=1))
        x_h, x_l, x_s = torch.split(x_b, [self.c_h, self.c_l, self.c_s], dim=1)

        x_h_outs = [m(x_h) for m in self.high_order_branch]
        x_h_fused = self.high_order_fuse(torch.cat(x_h_outs, dim=1))
        x_l_out = self.low_order_branch(x_l)

        y = self.final_fuse(torch.cat((x_h_fused, x_l_out, x_s), dim=1))
        return y


class GatedFusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, in_channels, 1, 1))

    def forward(self, f_in, h):
        return f_in + self.gamma * h


class YOLO13Encoder(nn.Module):
    def __init__(self, in_channels, base_channels=32):
        super().__init__()
        self.stem = DSConv(in_channels, base_channels, k=3, s=1)

        self.p2 = nn.Sequential(
            DSConv(base_channels, base_channels * 2, k=3, s=(2, 2)),
            DS_C3k2(base_channels * 2, base_channels * 2, n=1),
        )

        self.p3 = nn.Sequential(
            DSConv(base_channels * 2, base_channels * 4, k=3, s=(2, 2)),
            DS_C3k2(base_channels * 4, base_channels * 4, n=2),
        )

        self.p4 = nn.Sequential(
            DSConv(base_channels * 4, base_channels * 8, k=3, s=(2, 2)),
            DS_C3k2(base_channels * 8, base_channels * 8, n=2),
        )

        self.p5 = nn.Sequential(
            DSConv(base_channels * 8, base_channels * 16, k=3, s=(2, 2)),
            DS_C3k2(base_channels * 16, base_channels * 16, n=1),
        )

        self.out_channels = [
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 16,
        ]

    def forward(self, x):
        x = self.stem(x)
        p2 = self.p2(x)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return [p2, p3, p4, p5]


class YOLO13FullPADDecoder(nn.Module):
    def __init__(self, encoder_channels, hyperace_out_c, out_channels_final):
        super().__init__()
        c_p2, c_p3, c_p4, c_p5 = encoder_channels
        c_d5, c_d4, c_d3, c_d2 = c_p5, c_p4, c_p3, c_p2

        self.h_to_d5 = Conv(hyperace_out_c, c_d5, 1, 1)
        self.h_to_d4 = Conv(hyperace_out_c, c_d4, 1, 1)
        self.h_to_d3 = Conv(hyperace_out_c, c_d3, 1, 1)
        self.h_to_d2 = Conv(hyperace_out_c, c_d2, 1, 1)

        self.fusion_d5 = GatedFusion(c_d5)
        self.fusion_d4 = GatedFusion(c_d4)
        self.fusion_d3 = GatedFusion(c_d3)
        self.fusion_d2 = GatedFusion(c_d2)

        self.skip_p5 = Conv(c_p5, c_d5, 1, 1)
        self.skip_p4 = Conv(c_p4, c_d4, 1, 1)
        self.skip_p3 = Conv(c_p3, c_d3, 1, 1)
        self.skip_p2 = Conv(c_p2, c_d2, 1, 1)

        self.up_d5 = DS_C3k2(c_d5, c_d4, n=1)
        self.up_d4 = DS_C3k2(c_d4, c_d3, n=1)
        self.up_d3 = DS_C3k2(c_d3, c_d2, n=1)

        self.final_d2 = DS_C3k2(c_d2, c_d2, n=1)
        self.final_conv = Conv(c_d2, out_channels_final, 1, 1)

    def forward(self, enc_feats, h_ace):
        p2, p3, p4, p5 = enc_feats

        d5 = self.skip_p5(p5)
        h_d5 = self.h_to_d5(
            F.interpolate(
                h_ace, size=d5.shape[2:], mode="bilinear", align_corners=False
            )
        )
        d5 = self.fusion_d5(d5, h_d5)

        d5_up = F.interpolate(
            d5, size=p4.shape[2:], mode="bilinear", align_corners=False
        )

        d4 = self.up_d5(d5_up) + self.skip_p4(p4)
        h_d4 = self.h_to_d4(
            F.interpolate(
                h_ace, size=d4.shape[2:], mode="bilinear", align_corners=False
            )
        )
        d4 = self.fusion_d4(d4, h_d4)

        d4_up = F.interpolate(
            d4, size=p3.shape[2:], mode="bilinear", align_corners=False
        )
        d3 = self.up_d4(d4_up) + self.skip_p3(p3)
        h_d3 = self.h_to_d3(
            F.interpolate(
                h_ace, size=d3.shape[2:], mode="bilinear", align_corners=False
            )
        )
        d3 = self.fusion_d3(d3, h_d3)

        d3_up = F.interpolate(
            d3, size=p2.shape[2:], mode="bilinear", align_corners=False
        )
        d2 = self.up_d3(d3_up) + self.skip_p2(p2)
        h_d2 = self.h_to_d2(
            F.interpolate(
                h_ace, size=d2.shape[2:], mode="bilinear", align_corners=False
            )
        )
        d2 = self.fusion_d2(d2, h_d2)

        d2 = self.final_d2(d2)
        return self.final_conv(d2)


class DeepUnet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        en_out_channels=16,
        base_channels=64,
        hyperace_k=2,
        hyperace_l=1,
        num_hyperedges=16,
        num_heads=8,
    ):
        super().__init__()

        self.encoder = YOLO13Encoder(in_channels, base_channels)
        enc_ch = self.encoder.out_channels

        self.hyperace = HyperACE(
            in_channels=enc_ch,
            out_channels=enc_ch[-1],
            num_hyperedges=num_hyperedges,
            num_heads=num_heads,
            k=hyperace_k,
            l=hyperace_l,
        )

        self.decoder = YOLO13FullPADDecoder(
            encoder_channels=enc_ch,
            hyperace_out_c=enc_ch[-1],
            out_channels_final=en_out_channels,
        )

    def forward(self, x):
        original_size = x.shape[2:]

        features = self.encoder(x)
        h_ace = self.hyperace(features)
        x_dec = self.decoder(features, h_ace)

        x_out = F.interpolate(
            x_dec, size=original_size, mode="bilinear", align_corners=False
        )

        return x_out


class BiGRU(nn.Module):
    def __init__(self, input_features, hidden_features, num_layers):
        super(BiGRU, self).__init__()
        self.gru = nn.GRU(
            input_features,
            hidden_features,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x):
        return self.gru(x)[0]


class BiLSTM(nn.Module):
    def __init__(self, input_features, hidden_features, num_layers):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_features,
            hidden_features,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x):
        return self.lstm(x)[0]


class E2E0(nn.Module):
    def __init__(self, n_gru, in_channels=1, en_out_channels=16):
        super(E2E0, self).__init__()
        self.unet = DeepUnet(
            in_channels=in_channels,
            en_out_channels=en_out_channels,
            base_channels=64,
            hyperace_k=2,
            hyperace_l=1,
            num_hyperedges=16,
            num_heads=4,
        )
        self.cnn = nn.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))
        if n_gru:
            self.fc = nn.Sequential(
                BiGRU(3 * N_MELS, 256, n_gru),
                nn.Linear(512, N_CLASS),
                nn.Dropout(0.25),
                nn.Sigmoid(),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(3 * N_MELS, N_CLASS), nn.Dropout(0.25), nn.Sigmoid()
            )

    def forward(self, mel):
        mel = mel.transpose(-1, -2).unsqueeze(1)
        x = self.cnn(self.unet(mel)).transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x


class MelSpectrogram(torch.nn.Module):
    def __init__(
        self,
        n_mel_channels,
        sampling_rate,
        win_length,
        hop_length,
        n_fft=None,
        mel_fmin=0,
        mel_fmax=None,
        clamp=1e-5,
    ):
        super().__init__()
        n_fft = win_length if n_fft is None else n_fft
        self.hann_window = {}
        mel_basis = mel(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=n_mel_channels,
            fmin=mel_fmin,
            fmax=mel_fmax,
            htk=True,
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.n_fft = win_length if n_fft is None else n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels
        self.clamp = clamp

    def forward(self, audio, keyshift=0, speed=1, center=True):
        factor = 2 ** (keyshift / 12)
        n_fft_new = int(np.round(self.n_fft * factor))
        win_length_new = int(np.round(self.win_length * factor))
        hop_length_new = int(np.round(self.hop_length * speed))

        keyshift_key = str(keyshift) + "_" + str(audio.device)
        if keyshift_key not in self.hann_window:
            self.hann_window[keyshift_key] = torch.hann_window(win_length_new).to(
                audio.device
            )

        fft = torch.stft(
            audio,
            n_fft=n_fft_new,
            hop_length=hop_length_new,
            win_length=win_length_new,
            window=self.hann_window[keyshift_key],
            center=center,
            return_complex=True,
        )
        magnitude = torch.sqrt(fft.real.pow(2) + fft.imag.pow(2))

        if keyshift != 0:
            size = self.n_fft // 2 + 1
            resize = magnitude.size(1)
            if resize < size:
                magnitude = F.pad(magnitude, (0, 0, 0, size - resize))
            magnitude = magnitude[:, :size, :] * self.win_length / win_length_new

        mel_output = torch.matmul(self.mel_basis, magnitude)
        log_mel_spec = torch.log(torch.clamp(mel_output, min=self.clamp))
        return log_mel_spec


class HPARMVPE0Predictor:
    """
    A predictor for fundamental frequency (F0) based on the HPA-RMVPE model.

    Args:
        model_path (str): Path to the HPA-RMVPE model file.
        device (str, optional): Device to use for computation. Defaults to None, which uses CUDA if available.
        is_half (bool, optional): Use Half to save resources and speed up.
        onnx (bool, optional): Using the ONNX model.
        providers (list, optional): Providers of onnx model. default is CPUExecutionProvider.
    """

    def __init__(
        self,
        model_path: str,
        device: str | torch.device = "cpu",
        is_half: bool = True,
        onnx: bool = True,
        providers=["ROCMExecutionProvider', 'CPUExecutionProvider"],
        hop_length: int = 160,
        n_gru: int = 1,
        in_channels: int = 1,
        en_out_channels: int = 16,
    ):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.onnx = model_path.endswith(".onnx")  # onnx

        if self.onnx:
            import onnxruntime as ort

            sess_options = ort.SessionOptions()
            sess_options.log_severity_level = 3
            self.model = ort.InferenceSession(
                model_path, sess_options=sess_options, providers=providers
            )
        else:

            model = E2E0(n_gru, in_channels, en_out_channels)
            model.load_state_dict(
                torch.load(model_path, map_location="cpu", weights_only=True)
            )
            model.eval()
            model = model.to(device).eval()
            self.model = model.half() if is_half else model.float()

        self.device = device
        self.is_half = is_half
        self.mel_extractor = MelSpectrogram(
            N_MELS, SAMPLE_RATE, WINDOW_LENGTH, hop_length, None, MEL_FMIN, MEL_FMAX
        ).to(device)
        cents_mapping = 20 * np.arange(N_CLASS) + CONST
        self.cents_mapping = np.pad(cents_mapping, (4, 4))

    def mel2hidden(self, mel, chunk_size=SAMPLE_RATE * 2):
        """
        Converts Mel-spectrogram features to hidden representation.

        Args:
            mel (torch.Tensor): Mel-spectrogram features.
        """
        with torch.no_grad():
            n_frames = mel.shape[-1]
            # print('n_frames', n_frames)
            # print('mel shape before padding', mel.shape)
            mel = F.pad(
                mel, (0, 32 * ((n_frames - 1) // 32 + 1) - n_frames), mode="reflect"
            )
            # print('mel shape after padding', mel.shape)

            output_chunks = []
            pad_frames = mel.shape[-1]
            for start in range(0, pad_frames, chunk_size):
                # print('chunk @', start)
                end = min(start + chunk_size, pad_frames)
                mel_chunk = mel[..., start:end]
                assert (
                    mel_chunk.shape[-1] % 32 == 0
                ), "chunk_size must be divisible by 32"
                # print(' before padding', mel_chunk.shape)
                # mel_chunk = F.pad(mel_chunk, (320, 320), mode="reflect")
                # print(' after padding', mel_chunk.shape)
                if self.onnx:
                    out_chunk = torch.as_tensor(
                        self.model.run(
                            [self.model.get_outputs()[0].name],
                            {
                                self.model.get_inputs()[0]
                                .name: mel_chunk.cpu()
                                .numpy()
                                .astype(np.float32)
                            },
                        )[0],
                        device=self.device,
                    )
                else:
                    out_chunk = self.model(
                        mel_chunk.half() if self.is_half else mel_chunk.float()
                    )
                # print(' result chunk', out_chunk.shape)
                # out_chunk = out_chunk[:, 320:-320, :]
                # print(' trimmed chunk', out_chunk.shape)
                output_chunks.append(out_chunk)

            hidden = torch.cat(output_chunks, dim=1)
        # print('output', hidden[:, :n_frames].shape)
        return hidden[:, :n_frames]

    def decode(self, hidden, thred=0.03):
        """
        Decodes hidden representation to F0.

        Args:
            hidden (np.ndarray): Hidden representation.
            thred (float, optional): Threshold for salience. Defaults to 0.03.
        """
        cents_pred = self.to_local_average_cents(hidden, thred=thred)
        f0 = 10 * (2 ** (cents_pred / 1200))
        f0[f0 == 10] = 0
        return f0

    def infer_from_audio(self, audio, thred=0.03):
        """
        Infers F0 from audio.

        Args:
            audio (np.ndarray): Audio signal.
            thred (float, optional): Threshold for salience. Defaults to 0.03.
        """
        audio = torch.from_numpy(audio).float().to(self.device).unsqueeze(0)
        mel = self.mel_extractor(audio, center=True)
        del audio
        with torch.no_grad():
            torch.cuda.empty_cache()
        hidden = self.mel2hidden(mel)
        hidden = hidden.squeeze(0).cpu().numpy()
        f0 = self.decode(hidden, thred=thred)
        return f0

    def infer_from_audio_with_pitch(self, audio, thred=0.03, f0_min=50, f0_max=1100):
        """
        Infers F0 from audio with pitch.

        Args:
            audio (np.ndarray): Audio signal.
            thred (float, optional): Threshold for salience. Defaults to 0.03.
            f0_min (float, int, optional): Minimum F0 threshold.
            f0_max (float, int, optional): Maximum F0 threshold.
        """

        f0 = self.infer_from_audio(audio, thred)
        f0[(f0 < f0_min) | (f0 > f0_max)] = 0

        return f0

    def to_local_average_cents(self, salience, thred=0.05):
        """
        Converts salience to local average cents.

        Args:
            salience (np.ndarray): Salience values.
            thred (float, optional): Threshold for salience. Defaults to 0.05.
        """
        center = np.argmax(salience, axis=1)
        salience = np.pad(salience, ((0, 0), (4, 4)))
        center += 4
        todo_salience = []
        todo_cents_mapping = []
        starts = center - 4
        ends = center + 5
        for idx in range(salience.shape[0]):
            todo_salience.append(salience[:, starts[idx] : ends[idx]][idx])
            todo_cents_mapping.append(self.cents_mapping[starts[idx] : ends[idx]])
        todo_salience = np.array(todo_salience)
        todo_cents_mapping = np.array(todo_cents_mapping)
        product_sum = np.sum(todo_salience * todo_cents_mapping, 1)
        weight_sum = np.sum(todo_salience, 1)
        devided = product_sum / weight_sum
        maxx = np.max(salience, axis=1)
        devided[maxx <= thred] = 0
        return devided

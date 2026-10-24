from typing import Union, List, Optional

try:
    from typing import Literal
except ImportError:  # Python < 3.8
    try:
        from typing_extensions import Literal
    except ImportError:
        class _LiteralCompat:
            def __getitem__(self, _item):
                return object

        Literal = _LiteralCompat()

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.detect_any_sound.detect_any_sound import DASM
from src.models.htsat.htsat import CLAPAudioCfp, create_htsat_model
from src.models.passt.passt_sed import InterpolateModule


class DASM_HTSAT(DASM):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _init_joint_layers(self, transformer_embed_dim):
        self.interpolate_module = InterpolateModule(mode='linear')
        if hasattr(self, "cnn"):
            self.cnn_projector = torch.nn.Linear(self.cnn_feat_dim, self.decoder_dim)
            self.merge_weight = torch.nn.Parameter(torch.Tensor([0.5]), requires_grad=self.mlm_dict is not None)
        self.transformer_projector = torch.nn.Linear(transformer_embed_dim, self.decoder_dim)
        self.at_projector = torch.nn.Linear(transformer_embed_dim, self.decoder_dim)
        self.norm_before_pool = nn.LayerNorm(transformer_embed_dim)
        self.norm_after_merge = nn.LayerNorm(self.decoder_dim)

    def _init_transformer_backbone(self, backbone_param):
        self.backbone = create_htsat_model(CLAPAudioCfp)
        if backbone_param["pretrain_model_path"] is not None:
            self.backbone.load_state_dict(
                torch.load(backbone_param["pretrain_model_path"], map_location="cpu")
            )

    def f_pool(self, passt_out_dict):
        raise NotImplementedError("Method 'f_pool' is not supported by DASM_HTAST")

    def forward(
        self,
        input,
        encoder_win=False,
        mix_rate=0.5,
        win_param=[512, 49],
        temp_w=0.1,
        pad_mask=None,
        query: Union[torch.Tensor, list] = None,
        query_type: Optional[Literal['text', 'audio']] = None,
        tgt_mask=None,
        return_intermediates=False,
    ):
        other_dict = {}
        # shape of input (batch_size, 1, time_steps, mel_bins)
        # shape of htast_output is (Batch, Frames:32, Channel:768)
        htast_feat = self.backbone(input.clone())['fine_grained_embedding']
        # pooling
        x = self.interpolate_module(htast_feat, self.backbone_upsample_ratio)
        if encoder_win:
            from src.models.htsat.htast_win import HtsatSlideWindow
            slide_window_model = HtsatSlideWindow(net=self, win_param=win_param)
            x_local = slide_window_model(input.squeeze(1), emb_len=x.shape[1])
            x = mix_rate * x_local + (1 - mix_rate) * x

        # merge CNN's feature
        if hasattr(self, "cnn"):
            cnn_feat = self.cnn(input)
            _, cnn_channel, cnn_t, cnn_f = cnn_feat.shape
            assert cnn_channel == self.cnn_feat_dim
            assert cnn_f == 1
            cnn_feat = F.interpolate(cnn_feat.squeeze(-1), size=x.shape[1],
                                     mode=self.interpolate_module.mode).transpose(1, 2)  #[B, T, C]
            # TODO: add implementation for sliding-window strategy
            x = self.transformer_projector(x) +\
                self.merge_weight*self.cnn_projector(cnn_feat)
        else:
            x = self.transformer_projector(x)

        x = self.norm_after_merge(x)

        # AT decoder
        at_feat = self.at_projector(htast_feat)
        if isinstance(query, torch.Tensor) and query.ndim == 3:
            query = query[0, :, :]
        elif isinstance(query, list) and query[0].ndim == 3:
            for i in range(len(query)):
                query[i] = query[i][0, :, :]
            query = torch.nn.ParameterList(query)

        if tgt_mask is not None and tgt_mask.ndim == 3:
            tgt_mask = tgt_mask[0, :, :]
        (
            other_dict["at_out"],
            mask_feat,
            projected_query,
        ) = self.at_branch(at_feat, query, query_type, tgt_mask)
        # mask language model
        if self.mlm_dict is not None:
            other_dict["frame_before_mask"] = x
            x, mask_id_seq = self.mlm_tool.setence_mask(x, self.mask_token)
            other_dict["mask_id_seq"] = mask_id_seq

        # sed decoder
        x = self.sed_branch(x)

        if self.mlm_dict is not None:
            x = self.mlm_mlp(x)
            return x, other_dict

        # localization
        x = self.sed_head(x)
        mask_embedding = self.mask_embedding_layer(mask_feat)
        if return_intermediates:
            other_dict["localization_frame_feat"] = x
            other_dict["localization_query_feat"] = mask_embedding
        x = torch.einsum("bqc, bct -> bqt", mask_embedding, x.transpose(1, 2)).transpose(1, 2)
        frame_prob = torch.sigmoid(x / temp_w)
        if return_intermediates or (
            self.gate_power_enabled
            and self.gate_power_mode in (
                "router", "framewise_power", "query_beta_expert"
            )
        ):
            other_dict["frame_prob_before_clip"] = frame_prob
        if self.gate_power_enabled:
            clip_gate = other_dict["at_out"].clamp(min=1e-7, max=1.0)
            if self.gate_power_mode == "router":
                router_prob = self.get_evidence_router(
                    frame_prob, clip_gate, pad_mask=pad_mask
                )
                router_prob = router_prob * self.gate_repair_mask.to(
                    dtype=router_prob.dtype
                ).unsqueeze(0)
                if self.router_hard_inference:
                    router_weight = (router_prob > self.router_threshold).to(
                        router_prob.dtype
                    )
                else:
                    router_weight = router_prob
                gated_prob = frame_prob * clip_gate.unsqueeze(1)
                sed_out = gated_prob + router_weight.unsqueeze(1) * (frame_prob - gated_prob)
                other_dict["router_prob"] = router_prob
                other_dict["router_weight"] = router_weight
            elif self.gate_power_mode == "dynamic":
                gate_power = self.get_dynamic_gate_power(
                    frame_prob, clip_gate, pad_mask=pad_mask
                )
                gate_exponent = gate_power.unsqueeze(1)
                other_dict["gate_power"] = gate_power
                sed_out = frame_prob * clip_gate.unsqueeze(1).pow(gate_exponent)
            elif self.gate_power_mode == "framewise_power":
                gate_power = self.get_framewise_gate_power(
                    frame_prob, clip_gate, pad_mask=pad_mask
                )
                baseline_gated = frame_prob * clip_gate.unsqueeze(1)
                other_dict["gate_power"] = gate_power
                other_dict["baseline_gated_prob"] = baseline_gated
                sed_out = frame_prob * clip_gate.unsqueeze(1).pow(gate_power)
            elif self.gate_power_mode == "query_beta_expert":
                (
                    expert_logits,
                    expert_weights,
                    expert_gates,
                    mixed_gate,
                    expected_beta,
                ) = self.get_query_beta_expert(
                    frame_prob, clip_gate, projected_query, pad_mask=pad_mask
                )
                other_dict["expert_logits"] = expert_logits
                other_dict["expert_weights"] = expert_weights
                other_dict["expert_gates"] = expert_gates
                other_dict["mixed_clip_gate"] = mixed_gate
                other_dict["expected_gate_power"] = expected_beta
                other_dict["baseline_gated_prob"] = (
                    frame_prob * clip_gate.unsqueeze(1)
                )
                sed_out = frame_prob * mixed_gate.unsqueeze(1)
            else:
                gate_power = self.get_gate_power()
                gate_exponent = gate_power.view(1, 1, -1)
                other_dict["gate_power"] = gate_power
                sed_out = frame_prob * clip_gate.unsqueeze(1).pow(gate_exponent)
        else:
            sed_out = frame_prob * other_dict["at_out"].unsqueeze(1)  # with clip-level prior
        # sed_out = torch.sigmoid(x / temp_w) # without clip-level prior

        if pad_mask is not None:
            sed_out[pad_mask] = 0

        # linear-softmax pool
        sed_out = torch.clamp(sed_out, 1e-7, 1.)
        weak_out = (sed_out * sed_out).sum(dim=1) / sed_out.sum(dim=1)
        weak_out = torch.clamp(weak_out, 1e-7, 1.)
        return sed_out.transpose(1, 2), weak_out, other_dict

    def get_model_name(self):
        return 'DASM_HTSAT'

    def get_feature_extractor(self, mixup_lambda=None):
        return lambda wav: self.backbone.wav2mel(wav, mixup_lambda)

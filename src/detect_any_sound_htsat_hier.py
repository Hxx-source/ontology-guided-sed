from pathlib import Path
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.append(str(ROOT))
sys.path.append(str(CODE_DIR))

from src.models.detect_any_sound.detect_any_sound_htast import DASM_HTSAT


class HierDASM_HTSAT(DASM_HTSAT):

    def __init__(self, *args, hierarchy_projector_dim=None, hierarchy_projector_dropout=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        projector_dim = int(hierarchy_projector_dim) if hierarchy_projector_dim is not None else None
        projector_dropout = float(hierarchy_projector_dropout) if hierarchy_projector_dropout is not None else 0.0
        if projector_dim is not None and projector_dim > 0:
            layers = [nn.Linear(self.decoder_dim, projector_dim), nn.GELU()]
            if projector_dropout > 0:
                layers.append(nn.Dropout(projector_dropout))
            layers.append(nn.Linear(projector_dim, projector_dim))
            self.hierarchy_projector = nn.Sequential(*layers)
            self.hierarchy_projector_out_dim = projector_dim
        else:
            self.hierarchy_projector = None
            self.hierarchy_projector_out_dim = self.decoder_dim
        self.query_parent_mix_enabled = False
        self.query_parent_mix_alpha = 0.0
        self.query_parent_mix_mode = "convex"
        self.query_parent_mix_pairs = []

    def set_query_parent_mix(self, pair_indices, alpha=0.1, mode="convex"):
        self.query_parent_mix_enabled = bool(pair_indices)
        self.query_parent_mix_alpha = float(alpha)
        self.query_parent_mix_mode = str(mode).lower()
        if self.query_parent_mix_mode not in ("convex", "add"):
            raise ValueError(f"Unknown query_parent_mix_mode: {mode}")
        self.query_parent_mix_pairs = [(int(child_idx), int(parent_idx)) for child_idx, parent_idx in pair_indices]

    def clear_query_parent_mix(self):
        self.query_parent_mix_enabled = False
        self.query_parent_mix_pairs = []

    def _apply_query_parent_mix(self, query):
        if not self.query_parent_mix_enabled or not self.query_parent_mix_pairs:
            return query
        if not isinstance(query, torch.Tensor) or query.ndim != 2:
            return query

        alpha = float(self.query_parent_mix_alpha)
        src = query
        mixed = query.clone()
        for child_idx, parent_idx in self.query_parent_mix_pairs:
            if child_idx >= src.shape[0] or parent_idx >= src.shape[0]:
                continue
            if self.query_parent_mix_mode == "add":
                mixed[child_idx] = src[child_idx] + alpha * src[parent_idx]
            else:
                mixed[child_idx] = (1.0 - alpha) * src[child_idx] + alpha * src[parent_idx]
        return mixed

    def _init_query(self, query_projector=False, query=None, query_dim=None):
        if query_projector and query is None:
            if isinstance(query_dim, int):
                self.query_projector = nn.Sequential(nn.Linear(query_dim, self.decoder_dim), nn.GELU())
                self.at_query = nn.Parameter(torch.zeros(self.class_num, query_dim))
                torch.nn.init.normal_(self.at_query, std=.02)
                return
        super()._init_query(query_projector=query_projector, query=query, query_dim=query_dim)

    def forward(
        self,
        input,
        encoder_win=False,
        mix_rate=0.5,
        win_param=[512, 49],
        temp_w=0.1,
        pad_mask=None,
        query=None,
        query_type=None,
        tgt_mask=None,
    ):
        other_dict = {}
        htast_feat = self.backbone(input.clone())["fine_grained_embedding"]
        x = self.interpolate_module(htast_feat, self.backbone_upsample_ratio)
        if encoder_win:
            from src.models.htsat.htast_win import HtsatSlideWindow

            slide_window_model = HtsatSlideWindow(net=self, win_param=win_param)
            x_local = slide_window_model(input.squeeze(1), emb_len=x.shape[1])
            x = mix_rate * x_local + (1 - mix_rate) * x

        if hasattr(self, "cnn"):
            cnn_feat = self.cnn(input)
            _, cnn_channel, _, cnn_f = cnn_feat.shape
            assert cnn_channel == self.cnn_feat_dim
            assert cnn_f == 1
            cnn_feat = F.interpolate(
                cnn_feat.squeeze(-1),
                size=x.shape[1],
                mode=self.interpolate_module.mode,
            ).transpose(1, 2)
            x = self.transformer_projector(x) + self.merge_weight * self.cnn_projector(cnn_feat)
        else:
            x = self.transformer_projector(x)

        x = self.norm_after_merge(x)
        at_feat = self.at_projector(htast_feat)
        if isinstance(query, torch.Tensor) and query.ndim == 3:
            query = query[0, :, :]
        elif isinstance(query, list) and query[0].ndim == 3:
            for i in range(len(query)):
                query[i] = query[i][0, :, :]
            query = torch.nn.ParameterList(query)
        if query is None:
            query = self.at_query
        if not isinstance(query, torch.nn.ParameterList):
            query = self._apply_query_parent_mix(query)

        if tgt_mask is not None and tgt_mask.ndim == 3:
            tgt_mask = tgt_mask[0, :, :]
        other_dict["at_out"], mask_feat = self.at_branch(at_feat, query, query_type, tgt_mask)
        other_dict["mask_feat"] = mask_feat
        if self.hierarchy_projector is not None:
            other_dict["hierarchy_feat"] = self.hierarchy_projector(mask_feat)
        else:
            other_dict["hierarchy_feat"] = mask_feat

        if self.mlm_dict is not None:
            other_dict["frame_before_mask"] = x
            x, mask_id_seq = self.mlm_tool.setence_mask(x, self.mask_token)
            other_dict["mask_id_seq"] = mask_id_seq

        x = self.sed_branch(x)

        if self.mlm_dict is not None:
            x = self.mlm_mlp(x)
            return x, other_dict

        x = self.sed_head(x)
        mask_embedding = self.mask_embedding_layer(mask_feat)
        x = torch.einsum("bqc, bct -> bqt", mask_embedding, x.transpose(1, 2)).transpose(1, 2)
        sed_out = torch.sigmoid(x / temp_w) * other_dict["at_out"].unsqueeze(1)

        if pad_mask is not None:
            sed_out[pad_mask] = 0

        sed_out = torch.clamp(sed_out, 1e-7, 1.0)
        weak_out = (sed_out * sed_out).sum(dim=1) / sed_out.sum(dim=1)
        weak_out = torch.clamp(weak_out, 1e-7, 1.0)
        return sed_out.transpose(1, 2), weak_out, other_dict

    def get_model_name(self):
        return "DASM_HTSAT"

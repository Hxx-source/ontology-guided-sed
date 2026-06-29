from __future__ import annotations

import json
import math
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F


class HierarchyConsistencyHelper:

    def __init__(
        self,
        labels,
        ontology_path,
        class_stats_path,
        tail_threshold=None,
        distance_decay=0.35,
        min_weight=0.25,
        max_weight=4.0,
        require_positive_ancestor=True,
        positive_margin=0.7,
        negative_margin=0.35,
        sibling_weight=0.35,
        use_sibling_separation=True,
        max_ancestor_distance=1,
        score_margin=0.0,
        score_source="both",
        score_detach_ancestor=False,
        target_labels=None,
        target_labels_path=None,
        target_label_column="event_label",
        target_pairs=None,
        target_pairs_path=None,
        target_pair_child_column="child_label",
        target_pair_parent_column="parent_label",
        target_weight_mode="class_weight",
        adaptive_pair_weight_config=None,
    ) -> None:
        self.labels = list(labels)
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        self.ontology_path = Path(ontology_path)
        self.class_stats_path = Path(class_stats_path)
        self.distance_decay = float(distance_decay)
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.require_positive_ancestor = bool(require_positive_ancestor)
        self.positive_margin = float(positive_margin)
        self.negative_margin = float(negative_margin)
        self.sibling_weight = float(sibling_weight)
        self.use_sibling_separation = bool(use_sibling_separation)
        self.max_ancestor_distance = int(max_ancestor_distance) if max_ancestor_distance is not None else None
        self.score_margin = float(score_margin)
        self.score_source = str(score_source).lower()
        if self.score_source not in ("strong", "weak", "clip", "both"):
            raise ValueError(f"Unknown hierarchy score_source: {score_source}")
        self.score_detach_ancestor = bool(score_detach_ancestor)
        self.target_labels = self._load_target_labels(
            target_labels=target_labels,
            target_labels_path=target_labels_path,
            target_label_column=target_label_column,
        )
        self.target_parent_labels_by_child = self._load_target_pairs(
            target_pairs=target_pairs,
            target_pairs_path=target_pairs_path,
            child_column=target_pair_child_column,
            parent_column=target_pair_parent_column,
        )
        if self.target_labels is None and self.target_parent_labels_by_child is not None:
            self.target_labels = list(self.target_parent_labels_by_child)
        self.target_child_indices = (
            {self.label_to_idx[label] for label in self.target_labels}
            if self.target_labels is not None else None
        )
        self.target_weight_mode = str(target_weight_mode).lower()
        if self.target_weight_mode not in ("class_weight", "uniform", "adaptive_pair"):
            raise ValueError(f"Unknown hierarchy target_weight_mode: {target_weight_mode}")
        self.adaptive_pair_weight_config = adaptive_pair_weight_config or {}
        self.target_pair_weights = self._load_target_pair_weights(
            target_pairs=target_pairs,
            target_pairs_path=target_pairs_path,
            child_column=target_pair_child_column,
            parent_column=target_pair_parent_column,
        )
        self.child_to_ancestors, self.child_to_siblings = self._build_relation_maps()
        self.class_alpha, self.event_count, self.tail_threshold, self.median_count = self._build_class_weights(
            tail_threshold=tail_threshold
        )

    def _validate_target_labels(self, labels, source_desc):
        labels = [str(label).strip() for label in labels if str(label).strip()]
        labels = list(dict.fromkeys(labels))
        unknown = [label for label in labels if label not in self.label_to_idx]
        if unknown:
            raise ValueError(f"Unknown hierarchy target labels in {source_desc}: {unknown}")
        return labels

    def _load_target_labels(self, target_labels=None, target_labels_path=None, target_label_column="event_label"):
        if target_labels is not None:
            return self._validate_target_labels(target_labels, "inline target_labels")

        if not target_labels_path:
            return None

        path = Path(target_labels_path)
        if not path.exists():
            raise FileNotFoundError(f"Hierarchy target label file does not exist: {path}")

        if path.suffix.lower() in (".csv", ".tsv"):
            sep = "\t" if path.suffix.lower() == ".tsv" else ","
            df = pd.read_csv(path, sep=sep)
            if target_label_column not in df.columns:
                raise ValueError(f"{path} does not contain column: {target_label_column}")
            labels = [str(label).strip() for label in df[target_label_column].dropna()]
        elif path.suffix.lower() in (".xls", ".xlsx"):
            df = pd.read_excel(path)
            if target_label_column not in df.columns:
                raise ValueError(f"{path} does not contain column: {target_label_column}")
            labels = [str(label).strip() for label in df[target_label_column].dropna()]
        else:
            labels = [
                line.strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        return self._validate_target_labels(labels, str(path))

    def _validate_target_pairs(self, pair_items, source_desc):
        pairs = defaultdict(set)
        unknown = []
        for child_label, parent_label in pair_items:
            child_label = str(child_label).strip()
            parent_label = str(parent_label).strip()
            if child_label not in self.label_to_idx:
                unknown.append(child_label)
                continue
            if parent_label not in self.label_to_idx:
                unknown.append(parent_label)
                continue
            pairs[child_label].add(parent_label)
        if unknown:
            raise ValueError(f"Unknown hierarchy target pair labels in {source_desc}: {sorted(set(unknown))}")
        return {child: set(parents) for child, parents in pairs.items()}

    def _load_target_pairs(self, target_pairs=None, target_pairs_path=None, child_column="child_label", parent_column="parent_label"):
        if target_pairs is not None:
            if isinstance(target_pairs, dict):
                pair_items = []
                for child_label, parent_values in target_pairs.items():
                    if isinstance(parent_values, (list, tuple, set)):
                        for parent_label in parent_values:
                            pair_items.append((child_label, parent_label))
                    else:
                        pair_items.append((child_label, parent_values))
                return self._validate_target_pairs(pair_items, "inline target_pairs")
            if isinstance(target_pairs, list):
                pair_items = []
                for item in target_pairs:
                    if isinstance(item, dict):
                        pair_items.append((item.get(child_column), item.get(parent_column)))
                    elif isinstance(item, (list, tuple)) and len(item) >= 2:
                        pair_items.append((item[0], item[1]))
                    else:
                        raise ValueError("inline target_pairs items must be dicts or 2-tuples")
                return self._validate_target_pairs(pair_items, "inline target_pairs")
            raise ValueError("inline target_pairs must be a dict or list")

        if not target_pairs_path:
            return None

        path = Path(target_pairs_path)
        if not path.exists():
            raise FileNotFoundError(f"Hierarchy target pair file does not exist: {path}")

        if path.suffix.lower() in (".xls", ".xlsx"):
            df = pd.read_excel(path)
        else:
            sep = "\t" if path.suffix.lower() == ".tsv" else ","
            df = pd.read_csv(path, sep=sep)
        missing = [column for column in (child_column, parent_column) if column not in df.columns]
        if missing:
            raise ValueError(f"{path} does not contain columns: {missing}")

        pair_items = [
            (row[child_column], row[parent_column])
            for _, row in df[[child_column, parent_column]].dropna().iterrows()
        ]
        return self._validate_target_pairs(pair_items, str(path))

    @staticmethod
    def _clip01(value):
        return max(0.0, min(1.0, float(value)))

    def _cosine_pair_weight(self, cosine):
        cosine = float(cosine)
        cfg = self.adaptive_pair_weight_config
        low = float(cfg.get("cosine_low", 0.85))
        preferred_low = float(cfg.get("cosine_preferred_low", 0.90))
        preferred_high = float(cfg.get("cosine_preferred_high", 0.95))
        high = float(cfg.get("cosine_high", 0.98))
        low_weight = float(cfg.get("cosine_low_band_weight", 0.7))
        high_weight = float(cfg.get("cosine_high_band_weight", 0.5))
        if cosine < low or cosine > high:
            return 0.0
        if preferred_low <= cosine <= preferred_high:
            return 1.0
        if cosine < preferred_low:
            return low_weight
        return high_weight

    def _adaptive_pair_weight_from_row(self, row):
        cfg = self.adaptive_pair_weight_config
        duration_ratio_column = cfg.get(
            "ratio_column",
            cfg.get("duration_ratio_column", "child_parent_duration_ratio"),
        )
        parent_duration_column = cfg.get("parent_duration_column", "parent_train_duration_hrs")
        cosine_column = cfg.get("cosine_column", "pair_cosine")
        use_parent_support = bool(cfg.get("use_parent_support", True))
        use_cosine = bool(cfg.get("use_cosine", True))
        required_columns = [duration_ratio_column]
        if use_parent_support:
            required_columns.append(parent_duration_column)
        if use_cosine:
            required_columns.append(cosine_column)
        missing = [column for column in required_columns if column not in row.index]
        if missing:
            raise ValueError(
                "adaptive_pair target_weight_mode requires columns in target_pairs_path: "
                f"{missing}"
            )

        duration_ratio = max(float(row[duration_ratio_column]), 1e-8)
        parent_duration = max(float(row[parent_duration_column]), 0.0) if use_parent_support else 0.0
        cosine = float(row[cosine_column]) if use_cosine else 1.0

        ratio_log_base = float(cfg.get("ratio_log_base", cfg.get("duration_ratio_log_base", 1000.0)))
        parent_ref_hours = float(cfg.get("parent_ref_hours", 3.0))
        parent_cap_hours = float(cfg.get("parent_cap_hours", 50.0))

        duration_weight = self._clip01(
            math.log(1.0 / duration_ratio) / max(math.log(ratio_log_base), 1e-8)
        )
        parent_support_weight = (
            self._clip01(
                math.log(parent_duration / parent_ref_hours + 1.0)
                / max(math.log(parent_cap_hours / parent_ref_hours + 1.0), 1e-8)
            )
            if use_parent_support else 1.0
        )
        cosine_weight = self._cosine_pair_weight(cosine) if use_cosine else 1.0

        raw_weight = duration_weight * parent_support_weight * cosine_weight
        min_weight = float(cfg.get("min_pair_weight", 0.0))
        max_weight = float(cfg.get("max_pair_weight", 1.0))
        return max(min_weight, min(max_weight, raw_weight))

    def _load_target_pair_weights(self, target_pairs=None, target_pairs_path=None, child_column="child_label", parent_column="parent_label"):
        if self.target_weight_mode != "adaptive_pair":
            return {}

        if target_pairs_path:
            path = Path(target_pairs_path)
            if path.suffix.lower() in (".xls", ".xlsx"):
                df = pd.read_excel(path)
            else:
                sep = "\t" if path.suffix.lower() == ".tsv" else ","
                df = pd.read_csv(path, sep=sep)
            missing = [column for column in (child_column, parent_column) if column not in df.columns]
            if missing:
                raise ValueError(f"{path} does not contain columns: {missing}")

            pair_weights = {}
            for _, row in df.dropna(subset=[child_column, parent_column]).iterrows():
                child_label = str(row[child_column]).strip()
                parent_label = str(row[parent_column]).strip()
                if child_label not in self.label_to_idx or parent_label not in self.label_to_idx:
                    continue
                pair_weights[(self.label_to_idx[child_label], self.label_to_idx[parent_label])] = (
                    self._adaptive_pair_weight_from_row(row)
                )
            return pair_weights

        if target_pairs is not None:
            pair_weights = {}
            for item in target_pairs:
                if not isinstance(item, dict):
                    raise ValueError("adaptive_pair inline target_pairs must be dicts with weight metadata")
                child_label = str(item.get(child_column, "")).strip()
                parent_label = str(item.get(parent_column, "")).strip()
                if child_label not in self.label_to_idx or parent_label not in self.label_to_idx:
                    continue
                pair_weights[(self.label_to_idx[child_label], self.label_to_idx[parent_label])] = (
                    self._adaptive_pair_weight_from_row(pd.Series(item))
                )
            return pair_weights

        return {}

    def _build_relation_maps(self):
        with open(self.ontology_path, "r", encoding="utf-8") as f:
            ontology = json.load(f)

        name_to_id = {node["name"]: node["id"] for node in ontology}
        id_to_name = {node["id"]: node["name"] for node in ontology}
        parents = defaultdict(list)
        for node in ontology:
            for child_id in node.get("child_ids", []):
                parents[child_id].append(node["id"])

        @lru_cache(None)
        def get_ancestors(node_id):
            ancestor_list = []
            for parent_id in parents.get(node_id, []):
                ancestor_list.append((parent_id, 1))
                for ancestor_id, dist in get_ancestors(parent_id):
                    ancestor_list.append((ancestor_id, dist + 1))
            best = {}
            for ancestor_id, dist in ancestor_list:
                if ancestor_id not in best or dist < best[ancestor_id]:
                    best[ancestor_id] = dist
            return tuple(sorted(best.items(), key=lambda x: x[1]))

        child_to_ancestors = {}
        child_to_siblings = {}
        child_labels = self.target_labels if self.target_labels is not None else self.labels
        for child_label in child_labels:
            node_id = name_to_id.get(child_label)
            if node_id is None:
                child_to_ancestors[self.label_to_idx[child_label]] = []
                child_to_siblings[self.label_to_idx[child_label]] = []
                continue
            ancestor_pairs = []
            direct_siblings = set()
            allowed_parents = None
            if self.target_parent_labels_by_child is not None:
                allowed_parents = self.target_parent_labels_by_child.get(child_label, set())
            for ancestor_id, dist in get_ancestors(node_id):
                ancestor_label = id_to_name[ancestor_id]
                if self.max_ancestor_distance is not None and dist > self.max_ancestor_distance:
                    continue
                if allowed_parents is not None and ancestor_label not in allowed_parents:
                    continue
                if ancestor_label in self.label_to_idx:
                    ancestor_pairs.append((self.label_to_idx[ancestor_label], dist))
                if dist == 1:
                    for sibling_id in next(
                        (node.get("child_ids", []) for node in ontology if node["id"] == ancestor_id),
                        [],
                    ):
                        sibling_label = id_to_name.get(sibling_id)
                        if sibling_label in self.label_to_idx and sibling_label != child_label:
                            direct_siblings.add(self.label_to_idx[sibling_label])
            child_to_ancestors[self.label_to_idx[child_label]] = ancestor_pairs
            child_to_siblings[self.label_to_idx[child_label]] = sorted(direct_siblings)
        return child_to_ancestors, child_to_siblings

    def _build_class_weights(self, tail_threshold=None):
        stats_df = pd.read_csv(self.class_stats_path)
        stats_df = stats_df[stats_df["event_count"] > 0].copy()
        count_map = {
            row["event_label"]: float(row["event_count"])
            for _, row in stats_df.iterrows()
        }
        median_count = float(stats_df["event_count"].median())
        if tail_threshold is None:
            if "is_tail_bottom30" in stats_df.columns:
                tail_event_counts = stats_df.loc[stats_df["is_tail_bottom30"], "event_count"]
                tail_threshold = float(tail_event_counts.max()) if len(tail_event_counts) else median_count
            else:
                tail_threshold = median_count
        alpha = torch.zeros(len(self.labels), dtype=torch.float32)
        event_count = torch.zeros(len(self.labels), dtype=torch.float32)
        denom = max(self.max_weight - 1.0, 1e-6)
        for label, idx in self.label_to_idx.items():
            n_c = max(count_map.get(label, 0.0), 0.0)
            event_count[idx] = n_c

        if self.target_weight_mode in ("uniform", "adaptive_pair"):
            target_indices = self.target_child_indices
            if target_indices is None:
                target_indices = set(range(len(self.labels)))
            for idx in target_indices:
                alpha[idx] = 1.0
            return alpha, event_count, float(tail_threshold), median_count

        for label, idx in self.label_to_idx.items():
            n_c = float(event_count[idx].item())
            if n_c <= 0 or n_c > tail_threshold:
                continue
            raw = (median_count / max(n_c, 1.0)) ** 0.5
            clipped = max(self.min_weight, min(self.max_weight, raw))
            alpha[idx] = max(0.0, (clipped - 1.0) / denom)
        return alpha, event_count, float(tail_threshold), median_count

    def compute_feature_loss(self, hierarchy_feat: torch.Tensor, weak_targets: torch.Tensor):
        device = hierarchy_feat.device
        class_alpha = self.class_alpha.to(device)
        positive_mask = weak_targets > 0.5
        z_child = F.normalize(hierarchy_feat, dim=-1)
        z_teacher = z_child.detach()
        total_loss = hierarchy_feat.new_zeros(())
        total_weight = hierarchy_feat.new_zeros(())
        total_pairs = 0

        for child_idx, ancestor_pairs in self.child_to_ancestors.items():
            child_weight = float(class_alpha[child_idx].item())
            if child_weight <= 0:
                continue
            child_positive = positive_mask[:, child_idx]
            if not bool(child_positive.any()):
                continue
            child_repr = z_child[:, child_idx, :]
            for ancestor_idx, distance in ancestor_pairs:
                if self.require_positive_ancestor:
                    pair_mask = child_positive & positive_mask[:, ancestor_idx]
                else:
                    pair_mask = child_positive
                if not bool(pair_mask.any()):
                    continue
                pair_weight = self._pair_weight(class_alpha, child_idx, ancestor_idx, distance)
                if pair_weight <= 0:
                    continue
                cosine = (child_repr[pair_mask] * z_teacher[pair_mask, ancestor_idx, :]).sum(dim=-1)
                pair_loss = F.relu(self.positive_margin - cosine).mean()
                total_loss = total_loss + pair_loss * pair_weight
                total_weight = total_weight + pair_weight
                total_pairs += int(pair_mask.sum().item())

            if not self.use_sibling_separation or self.sibling_weight <= 0:
                continue

            for sibling_idx in self.child_to_siblings.get(child_idx, []):
                pair_mask = child_positive & (~positive_mask[:, sibling_idx])
                if not bool(pair_mask.any()):
                    continue
                cosine = (child_repr[pair_mask] * z_teacher[pair_mask, sibling_idx, :]).sum(dim=-1)
                pair_loss = F.relu(cosine - self.negative_margin).mean()
                pair_weight = child_weight * self.sibling_weight
                total_loss = total_loss + pair_loss * pair_weight
                total_weight = total_weight + pair_weight
                total_pairs += int(pair_mask.sum().item())

        if float(total_weight.item()) <= 0:
            return hierarchy_feat.new_zeros(()), 0
        return total_loss / total_weight, total_pairs

    def _pair_weight(self, class_alpha, child_idx, ancestor_idx, distance):
        child_weight = float(class_alpha[child_idx].item())
        if child_weight <= 0:
            return 0.0
        pair_weight = child_weight
        if self.target_weight_mode == "adaptive_pair":
            pair_weight *= float(self.target_pair_weights.get((child_idx, ancestor_idx), 0.0))
        return pair_weight * (self.distance_decay ** max(distance - 1, 0))

    def compute_score_loss(
        self,
        strong_scores: torch.Tensor,
        weak_scores: torch.Tensor,
        strong_targets: torch.Tensor,
        weak_targets: torch.Tensor,
    ):
        """Enforce child scores not to exceed ancestor scores.

        strong_scores/targets use shape [B, C, T]. weak_scores/targets use [B, C].
        The loss is asymmetric: child evidence should imply at least as much parent
        evidence, but parent evidence is not forced to imply child evidence.
        """
        device = strong_scores.device
        class_alpha = self.class_alpha.to(device)
        total_loss = strong_scores.new_zeros(())
        total_weight = strong_scores.new_zeros(())
        total_pairs = 0
        use_strong = self.score_source in ("strong", "both")
        use_weak = self.score_source in ("weak", "both", "clip")

        for child_idx, ancestor_pairs in self.child_to_ancestors.items():
            child_strong_positive = strong_targets[:, child_idx, :] > 0.5
            child_weak_positive = weak_targets[:, child_idx] > 0.5
            if use_strong and not bool(child_strong_positive.any()):
                strong_available = False
            else:
                strong_available = True
            if use_weak and not bool(child_weak_positive.any()):
                weak_available = False
            else:
                weak_available = True
            if not strong_available and not weak_available:
                continue

            for ancestor_idx, distance in ancestor_pairs:
                pair_weight = self._pair_weight(class_alpha, child_idx, ancestor_idx, distance)
                if pair_weight <= 0:
                    continue

                if use_strong and strong_available:
                    if self.require_positive_ancestor:
                        strong_mask = child_strong_positive & (strong_targets[:, ancestor_idx, :] > 0.5)
                    else:
                        strong_mask = child_strong_positive
                    if bool(strong_mask.any()):
                        child_score = strong_scores[:, child_idx, :][strong_mask]
                        ancestor_score = strong_scores[:, ancestor_idx, :][strong_mask]
                        if self.score_detach_ancestor:
                            ancestor_score = ancestor_score.detach()
                        pair_loss = F.relu(child_score - ancestor_score + self.score_margin).mean()
                        total_loss = total_loss + pair_loss * pair_weight
                        total_weight = total_weight + pair_weight
                        total_pairs += int(strong_mask.sum().item())

                if use_weak and weak_available:
                    if self.require_positive_ancestor:
                        weak_mask = child_weak_positive & (weak_targets[:, ancestor_idx] > 0.5)
                    else:
                        weak_mask = child_weak_positive
                    if bool(weak_mask.any()):
                        child_score = weak_scores[:, child_idx][weak_mask]
                        ancestor_score = weak_scores[:, ancestor_idx][weak_mask]
                        if self.score_detach_ancestor:
                            ancestor_score = ancestor_score.detach()
                        pair_loss = F.relu(child_score - ancestor_score + self.score_margin).mean()
                        total_loss = total_loss + pair_loss * pair_weight
                        total_weight = total_weight + pair_weight
                        total_pairs += int(weak_mask.sum().item())

        if float(total_weight.item()) <= 0:
            return strong_scores.new_zeros(()), 0
        return total_loss / total_weight, total_pairs

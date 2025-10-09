# file: lzero/mcts/buffer/game_buffer_dual_stochastic_muzero.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import torch
import random

from lzero.mcts.buffer.game_buffer_stochastic_muzero import StochasticMuZeroGameBuffer
from lzero.mcts.utils import prepare_observation
from lzero.mcts.tree_search.mcts_ctree import MuZeroMCTSCtree as MCTSCtree
from lzero.mcts.tree_search.mcts_ptree import MuZeroMCTSPtree as MCTSPtree


class DualStochasticMuZeroBuffer(StochasticMuZeroGameBuffer):
    """
    Dual-view buffer for Stochastic MuZero.
    - Base view: append-only raw segments (collector output). Never cleared by rebuild.
    - Train view: derived segments for training; rebuilt on demand before sampling.
    """

    # ---------------- Construction ----------------
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        # Base storages
        self.base_segment_buffer: List[Any] = []
        self.base_meta_buffer: List[Dict[str, Any]] = []
        self.base_pos_priorities: List[np.ndarray] = []  # per-step priorities per base segment

        # Train-index -> (base_id, start_offset_in_base) mapping for priority back-prop
        self._train_index_to_base: Dict[int, Tuple[int, int]] = {}

        # Priority sync policy: "off" | "copy" | "max" | "mean"
        self.priority_binding: str = "copy"

        # Temporarily mirror new base data to train view before first rebuild
        self.mirror_new_into_train: bool = True

        # Handle to policy for light prediction in `predict_state`
        self._predict_policy = None

        # Random seed control
        self._rng = np.random.RandomState(42)
        self._seed = 42

        # --- predictor mode switches ---
        self.predict_full_mcts: bool = bool(self._cfg.get('predict_full_mcts', False))
        self.predict_use_ctree: bool = bool(self._cfg.get('mcts_ctree', True))
        self.predict_reanalyze_noise: bool = bool(self._cfg.get('reanalyze_noise', False))

    def set_seed(self, seed: int) -> None:
        """Set random seed for reproducible slicing."""
        self._seed = seed
        self._rng = np.random.RandomState(seed)

    def attach_predict_policy(self, policy: Any) -> None:
        """Attach the LightZero policy object."""
        self._predict_policy = policy

    def set_predict_mode(self, full: bool) -> None:
        """Toggle predictor mode."""
        self.predict_full_mcts = bool(full)

    # ---------------- Base push path ----------------
    def push_game_segments(self, data_and_meta):
        """Collector -> (data, meta). Append to base, optionally mirror to train."""
        data, meta = data_and_meta
        for seg, m in zip(data, meta):
            self._push_to_base(seg, m)
            if self.mirror_new_into_train:
                super()._push_game_segment(seg, dict(m) if m is not None else {})

    def _push_to_base(self, segment: Any, meta: Dict[str, Any]) -> None:
        """Append a base segment and initialize per-step priorities."""
        self.base_segment_buffer.append(segment)
        self.base_meta_buffer.append(dict(meta) if meta else {})
        L = len(segment)
        pri = None
        if meta and 'priorities' in meta:
            pri = np.asarray(meta['priorities'], dtype=np.float32)
        if pri is None or len(pri) != L:
            pri = np.zeros((L,), dtype=np.float32)
        self.base_pos_priorities.append(pri)

    # ---------------- Rebuild-on-demand (key API) ----------------
    def regen_train_view_from_base_sample(
            self,
            policy: Any,
            target_num_segments: int,
            *,
            clear_train_first: bool = True,
    ) -> int:
        """
        Rebuild TRAIN view from BASE samples.
        Steps: sample base segments -> predict states -> slice -> push to train view.
        """
        assert target_num_segments > 0
        if clear_train_first:
            self._clear_train_view()
        self._train_index_to_base.clear()
        self._ensure_predict_policy(policy)

        # Calculate safe minimum length for training segments
        need_len = self._calculate_safe_min_length()
        print(f"DEBUG: Safe minimum segment length = {need_len}, using seed={self._seed}")

        # Sample base segments by priority
        base_indices = self._priority_aware_segment_indices()

        produced = 0
        for base_id in base_indices:
            if produced >= target_num_segments:
                break
            seg = self.base_segment_buffer[base_id]
            meta = self.base_meta_buffer[base_id]
            L = len(seg)
            if L < need_len:
                continue

            # Predict policy distributions for each position
            stats_list: List[Dict[str, np.ndarray]] = []
            for pos in range(L):
                stats_list.append(self.predict_state(seg, pos))

            # Process into training pieces
            pieces = self.process_segment(seg, meta, stats_list, min_len=need_len)

            # Push pieces to train view
            for piece in pieces:
                if produced >= target_num_segments:
                    break
                if self._is_slice_tuple(piece):
                    s, e = piece
                    sub_seg = self._slice_segment(seg, s, e)
                    sub_meta = self._derive_sub_meta(meta, base_id, s, e)
                else:
                    sub_seg, sub_meta = piece
                    sub_meta.setdefault("base_id", base_id)
                    sub_meta.setdefault("slice", None)
                    sub_meta.setdefault("_offset0", 0)

                before = len(self.game_segment_buffer)
                super()._push_game_segment(sub_seg, sub_meta)
                after = len(self.game_segment_buffer)
                for train_idx in range(before, after):
                    self._train_index_to_base[train_idx] = (base_id, sub_meta.get("_offset0", -1))
                produced += 1
                print(f"DEBUG: Produced segment {produced}/{target_num_segments} from base {base_id}")

        self.mirror_new_into_train = False
        print(f"DEBUG: Regeneration complete. Produced {produced} segments.")
        return produced

    def _calculate_safe_min_length(self) -> int:
        """Calculate safe minimum length considering all training requirements."""
        # Original calculation
        basic_need = int(self._cfg.num_unroll_steps + np.clip(self._cfg.td_steps, 1, None).astype(np.int32))

        # More conservative: consider worst-case scenario
        # We need to be able to:
        # 1. Start from any position in the slice and unroll num_unroll_steps
        # 2. Calculate TD targets that may require td_steps lookahead
        conservative_need = max(
            basic_need,
            self._cfg.num_unroll_steps * 2,  # Allow multiple unroll positions
            self._cfg.td_steps * 2,  # Allow multiple TD calculation positions
        )

        # Add buffer for safety
        safe_length = conservative_need + 5
        return max(safe_length, 10)  # Minimum 10 steps

    # ---------------- Light predictor & processor ----------------
    def _ensure_predict_policy(self, policy: Any) -> None:
        """Ensure predict policy is attached."""
        if self._predict_policy is None and policy is not None:
            self.attach_predict_policy(policy)

    def predict_state(self, segment: Any, pos: int) -> Dict[str, np.ndarray]:
        """
        Return per-action stats at (segment, pos).
        When self.predict_full_mcts == False: run cheap initial_inference + softmax(logits).
        When True: run a single-root MCTS with the SAME cfg as training (num_simulations, pb_c_*, root_*).
        """
        assert self._predict_policy is not None, "Call attach_predict_policy(policy) first."
        model = self._predict_policy._target_model
        model.eval()

        # 1) Build stacked observation for the root
        obs = segment.get_unroll_obs(pos, num_unroll_steps=1, padding=True)
        obs = obs[0:self._cfg.model.frame_stack_num]
        obs = prepare_observation([obs], self._cfg.model.model_type)
        obs_t = torch.from_numpy(obs).to(self._cfg.device)

        # 2) Quick path (default): no tree search, just softmax over logits
        if not self.predict_full_mcts:
            with torch.no_grad():
                out = model.initial_inference(obs_t)
                logits = out.policy_logits
                logits = logits.detach().cpu().numpy()[0]
                logits = logits - logits.max()
                pi = np.exp(logits) / (np.exp(logits).sum() + 1e-8)
            return {"policy": pi.astype(np.float32)}  # matches previous lightweight behavior

        # 3) Full MCTS path: reuse the SAME cfg fields as training search
        with torch.no_grad():
            # initial inference to get latent state, reward, policy logits for the root
            net_out = model.initial_inference(obs_t)
            latent_state_root = net_out.latent_state
            reward0 = getattr(net_out, 'reward', None)
            if reward0 is None:
                reward0 = torch.zeros(1, device=latent_state_root.device)
            reward_val = float(torch.atleast_1d(reward0).detach().cpu().numpy().reshape(-1)[0])
            policy_logits = net_out.policy_logits.detach().cpu().numpy()[0].tolist()

        # 3.1 Build legal actions for the root (align with buffer._compute_target_policy_reanalyzed)
        # Per-node action mask is T+1; take mask at the current node 'pos'
        # If mask is missing, fall back to full discrete action set.
        if hasattr(segment, 'action_mask_segment') and segment.action_mask_segment is not None:
            node_mask = segment.action_mask_segment[min(pos, len(segment.action_mask_segment) - 1)]
            if node_mask is None:
                legal_actions = list(range(self._cfg.model.action_space_size))
            else:
                legal_actions = [i for i, x in enumerate(node_mask) if int(x) == 1]
        else:
            legal_actions = list(range(self._cfg.model.action_space_size))

        # to_play is used in board games; in non-board envs a dummy 0/1 works.
        to_play = [0]  # keep list shape like training batch

        # 3.2 Prepare a single-root batch and run the SAME tree backend as training
        if self.predict_use_ctree:
            roots = MCTSCtree.roots(1, [legal_actions])
        else:
            roots = MCTSPtree.roots(1, [legal_actions])

        # Optionally add root Dirichlet noise (reuse training flag & alphas)
        if self.predict_reanalyze_noise:
            alpha = float(self._cfg.root_dirichlet_alpha)
            aw = float(self._cfg.root_noise_weight)
            noise = self._rng.dirichlet([alpha] * self._cfg.model.action_space_size).astype(np.float32).tolist()
            # These 'prepare' helpers mirror the training code path inside LightZero buffer
            roots.prepare(aw, [noise], [reward_val], [policy_logits], to_play)
        else:
            roots.prepare_no_noise([reward_val], [policy_logits], to_play)

        # Now run the same MCTS search as training; cfg (num_simulations, pb_c_*, etc.) is reused here
        if self.predict_use_ctree:
            MCTSCtree(self._cfg).search(roots, model, [latent_state_root], to_play)
        else:
            MCTSPtree(self._cfg).search(roots, model, [latent_state_root], to_play)

        # 3.3 Read back visit counts and searched root value, then normalize to a policy
        dists = roots.get_distributions()  # list of lists (visit counts)
        values = roots.get_values()  # list of floats
        visits = np.asarray(dists[0], dtype=np.float32)
        vsum = float(visits.sum()) if visits.size > 0 else 0.0
        policy = (visits / (vsum + 1e-8)) if vsum > 0 else np.ones_like(visits) / max(1, len(visits))

        return {
            "policy": policy.astype(np.float32),  # normalized visit distribution
            "visit": visits,  # raw visit counts (optional)
            "value": np.asarray([values[0]], dtype=np.float32),  # searched root value
        }

    def process_segment(
            self,
            segment: Any,
            meta: Dict[str, Any],
            stats_list: List[Dict[str, np.ndarray]],
            min_len: int,
    ) -> List[Any]:
        """
        Safe random slicing processor:
        - Generate fixed-length slices longer than min_len
        - Random start positions within safe bounds
        - Controlled number of slices per segment
        """
        L = len(segment)
        print(f"DEBUG: Processing segment length {L}, min_len required {min_len}")

        # Use longer fixed length for safety
        slice_length = min_len + 10  # Extra buffer for safety
        slice_length = min(slice_length, L)  # Don't exceed segment length

        if slice_length > L:
            return []

        # Calculate possible start positions
        max_start = L - slice_length
        if max_start < 0:
            return []

        # Determine number of slices to generate (1-3 per segment)
        num_slices = min(3, max_start + 1)

        # Randomly choose start positions
        if max_start == 0:
            starts = [0]
        else:
            # Use weighted selection: prefer positions with higher information content
            start_weights = self._compute_start_weights(stats_list, slice_length, L)
            if len(start_weights) > 0 and not np.all(start_weights == start_weights[0]):
                # Weighted random choice
                starts = self._rng.choice(
                    range(max_start + 1),
                    size=num_slices,
                    replace=False,
                    p=start_weights / start_weights.sum()
                )
            else:
                # Uniform random choice
                starts = self._rng.choice(range(max_start + 1), size=num_slices, replace=False)

        # Generate slices
        slices = []
        for start in starts:
            end = start + slice_length
            slices.append((start, end))
            print(f"DEBUG: Generated slice ({start}, {end}) from segment length {L}")

            # Validate this slice can be used for training
            self._validate_slice(segment, start, end, min_len)

        return slices

    def _compute_start_weights(self, stats_list: List[Dict[str, np.ndarray]], slice_length: int,
                               total_length: int) -> np.ndarray:
        """Compute weights for start positions based on information content."""
        max_positions = total_length - slice_length + 1
        if max_positions <= 0:
            return np.array([])

        weights = np.ones(max_positions, dtype=np.float32)

        # Simple weighting: prefer positions with higher policy entropy
        for start in range(max_positions):
            # Sample a few positions within the potential slice
            sample_positions = [
                start,
                start + slice_length // 3,
                start + 2 * slice_length // 3,
                start + slice_length - 1
            ]
            sample_positions = [p for p in sample_positions if p < len(stats_list)]

            total_entropy = 0.0
            for pos in sample_positions:
                stats = stats_list[pos]
                policy = stats.get("policy", None)
                if policy is not None:
                    entropy = -np.sum(policy * np.log(policy + 1e-8))
                    total_entropy += entropy

            if len(sample_positions) > 0:
                weights[start] = total_entropy / len(sample_positions) + 0.1  # Add small bias

        return np.maximum(weights, 0.1)  # Ensure minimum weight

    def _validate_slice(self, segment: Any, start: int, end: int, min_len: int) -> None:
        """Validate that a slice can be used for training."""
        slice_len = end - start
        assert slice_len >= min_len, f"Slice length {slice_len} < min_len {min_len}"

        # Check if we can unroll from various positions within the slice
        test_positions = [start, start + slice_len // 2, start + slice_len - self._cfg.num_unroll_steps]
        test_positions = [p for p in test_positions if p >= start and p + self._cfg.num_unroll_steps <= end]

        print(f"DEBUG: Slice ({start}, {end}) can unroll from {len(test_positions)} positions")

        # ---------------- Priority binding ----------------

    def update_priority(self, train_data, batch_priorities):
        """Update priorities in both train and base views."""
        super().update_priority(train_data, batch_priorities)
        if self.priority_binding == "off":
            return
        indices = train_data[0][3]  # train indices
        for i, idx in enumerate(indices):
            if idx in self._train_index_to_base:
                base_id, offset0 = self._train_index_to_base[idx]
                if 0 <= base_id < len(self.base_pos_priorities) and offset0 >= 0:
                    self._bind_priority_to_base(base_id, offset0, float(batch_priorities[i]))

    def _bind_priority_to_base(self, base_id: int, pos: int, value: float) -> None:
        """Apply priority binding policy."""
        arr = self.base_pos_priorities[base_id]
        if not (0 <= pos < len(arr)):
            return
        if self.priority_binding == "copy":
            arr[pos] = value
        elif self.priority_binding == "max":
            arr[pos] = max(arr[pos], value)
        elif self.priority_binding == "mean":
            arr[pos] = 0.5 * (arr[pos] + value)

        # ---------------- Slicing internals ----------------

    @staticmethod
    def _is_slice_tuple(x: Any) -> bool:
        return isinstance(x, (tuple, list)) and len(x) == 2 and all(isinstance(t, int) for t in x)

    def _slice_segment(self, seg: Any, s: int, e: int):
        """
        Safe segment slicing:
        - Per-step fields (actions, rewards): [s:e]
        - Per-node fields (action_mask, to_play): [s:e+1] for next state info
        """
        sub = seg.__class__.__new__(seg.__class__)
        sub.__dict__ = seg.__dict__.copy()

        # Per-step fields (length T)
        per_step_fields = [
            'action_segment', 'reward_segment', 'child_visit_segment',
            'search_policy_segment', 'chance_segment', 'td_steps_list', 'search_values'
        ]
        for name in per_step_fields:
            if hasattr(seg, name) and getattr(seg, name) is not None:
                arr = getattr(seg, name)
                end_idx = min(e, len(arr))  # Prevent index out of bounds
                sub.__dict__[name] = arr[s:end_idx]

        # Per-node fields (length T+1) - need next state info
        per_node_fields = ['to_play_segment', 'action_mask_segment']
        for name in per_node_fields:
            if hasattr(seg, name) and getattr(seg, name) is not None:
                arr = getattr(seg, name)
                end_idx = min(e + 1, len(arr))  # Include next state
                sub.__dict__[name] = arr[s:end_idx]

        return sub

    def _derive_sub_meta(self, base_meta: Dict[str, Any], base_id: int, s: int, e: int) -> Dict[str, Any]:
        """Build meta for sub-segment with base binding info."""
        L = e - s
        out: Dict[str, Any] = {}

        # Copy priorities if available
        pri_base = base_meta.get('priorities', None)
        if pri_base is not None and len(pri_base) >= e:
            out['priorities'] = np.asarray(pri_base[s:e], dtype=np.float32)

        # Calculate adjusted unroll steps
        orig_upt = int(base_meta.get('unroll_plus_td_steps', e))
        out['unroll_plus_td_steps'] = max(0, min(L, orig_upt - s))

        # Set done flag if this is the end of original segment
        out['done'] = bool(base_meta.get('done', False) and e == len(self.base_pos_priorities[base_id]))

        # Base binding info
        out['base_id'] = base_id
        out['slice'] = (s, e)
        out['_offset0'] = s

        # Copy creation time
        if 'make_time' in base_meta:
            out['make_time'] = base_meta['make_time']

        return out

    def _clear_train_view(self) -> None:
        """Clear only TRAIN view; BASE remains intact."""
        self.game_segment_buffer.clear()
        self.game_pos_priorities.clear()
        self.game_segment_game_pos_look_up.clear()

    def _priority_aware_segment_indices(self) -> List[int]:
        """Sample base segments by priority score."""
        if len(self.base_pos_priorities) == 0:
            return []
        scores = []
        for i, arr in enumerate(self.base_pos_priorities):
            if arr is None or len(arr) == 0:
                scores.append((0.0, i))
            else:
                scores.append((float(np.max(arr)), i))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [i for _, i in scores]

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
        """Append base segment and initialize per-step priorities."""
        self.base_segment_buffer.append(segment)
        self.base_meta_buffer.append(dict(meta) if meta else {})
        L = len(segment)

        # Handle priorities - follow parent class logic
        pri = None
        if meta and 'priorities' in meta:
            pri_meta = meta['priorities']

            # Handle None case first
            if pri_meta is None:
                # Use parent's logic: if no priorities provided, use max priority from existing data
                if self.base_pos_priorities:
                    max_prio = max(np.max(arr) for arr in self.base_pos_priorities if len(arr) > 0)
                else:
                    max_prio = 1.0
                pri = np.full((L,), max_prio, dtype=np.float32)
            # Handle scalar case
            elif not hasattr(pri_meta, '__len__') or isinstance(pri_meta, (str, bytes)):
                pri = np.full((L,), float(pri_meta), dtype=np.float32)
            # Handle array case
            else:
                pri = np.asarray(pri_meta, dtype=np.float32)

        # If still no valid priority data, initialize with zeros
        if pri is None or (hasattr(pri, '__len__') and len(pri) != L):
            pri = np.zeros((L,), dtype=np.float32)
        elif not hasattr(pri, '__len__'):
            pri = np.full((L,), float(pri), dtype=np.float32)

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
                    # piece is (sub_seg, sub_meta)
                    sub_seg, sub_meta = piece
                    # --- ensure parent buffer-required fields exist ---
                    if 'priorities' not in sub_meta:
                        # length must match the sub-segment length
                        sub_meta['priorities'] = np.zeros((len(sub_seg),), dtype=np.float32)
                    sub_meta.setdefault('base_id', int(base_id))
                    sub_meta.setdefault('slice', None)
                    sub_meta.setdefault('_offset0', 0)

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
        Lightweight: initial_inference + softmax(logits)
        Full MCTS: complete tree search with same config as training.
        """
        assert self._predict_policy is not None, "Call attach_predict_policy(policy) first."
        model = self._predict_policy._target_model
        model.eval()

        # Build stacked observation for current position
        obs = segment.get_unroll_obs(pos, num_unroll_steps=1, padding=True)
        obs = obs[0:self._cfg.model.frame_stack_num]
        obs = prepare_observation([obs], self._cfg.model.model_type)
        obs_t = torch.from_numpy(obs).to(self._cfg.device)

        # Quick path: logits -> softmax
        self.predict_full_mcts = False
        if not self.predict_full_mcts:
            with torch.no_grad():
                out = model.initial_inference(obs_t)
                logits = out.policy_logits.detach().cpu().numpy()[0]
                logits = logits - logits.max()
                pi = np.exp(logits) / (np.exp(logits).sum() + 1e-8)
            return {"policy": pi.astype(np.float32)}

        # Full MCTS path
        with torch.no_grad():
            net_out = model.initial_inference(obs_t)
            latent_state_root = net_out.latent_state
            reward0 = getattr(net_out, 'reward', None)

            # Extract scalar reward safely
            if reward0 is None:
                reward_val = 0.0
            elif isinstance(reward0, torch.Tensor):
                reward_val = float(reward0.detach().cpu().numpy().reshape(-1)[0])
            elif isinstance(reward0, (list, np.ndarray)):
                reward_val = float(reward0[0]) if len(reward0) > 0 else 0.0
            else:
                reward_val = float(reward0) if isinstance(reward0, (int, float)) else 0.0

            # Extract policy logits safely (list[float], length = action_space_size)
            if hasattr(net_out.policy_logits, 'detach'):
                policy_logits = net_out.policy_logits.detach().cpu().numpy()[0].tolist()
            else:
                arr = np.asarray(net_out.policy_logits)
                policy_logits = arr[0].tolist() if arr.ndim > 1 else arr.tolist()

            # Convert latent to numpy and DROP batch dim
            # FIX: (1, C, H, W) -> (C, H, W) to match model.recurrent_inference's expected 4D after ctree batches it.
            if isinstance(latent_state_root, torch.Tensor):
                latent_state_np = latent_state_root.detach().cpu().numpy()
            else:
                latent_state_np = np.asarray(latent_state_root)
            if latent_state_np.ndim >= 4 and latent_state_np.shape[0] == 1:
                latent_state_np = latent_state_np[0]  # <-- FIX: remove batch dim

        # Legal actions for root
        if hasattr(segment, 'action_mask_segment') and segment.action_mask_segment is not None:
            node_mask = segment.action_mask_segment[min(pos, len(segment.action_mask_segment) - 1)]
            legal_actions = list(range(self._cfg.model.action_space_size)) if node_mask is None \
                else [i for i, x in enumerate(node_mask) if int(x) == 1]
        else:
            legal_actions = list(range(self._cfg.model.action_space_size))

        # Single-player default
        to_play = [0]

        # Build roots
        roots = MCTSCtree.roots(1, [legal_actions]) if self.predict_use_ctree \
            else MCTSPtree.roots(1, [legal_actions])

        # Dirichlet noise (optional)
        if self.predict_reanalyze_noise:
            alpha = float(self._cfg.root_dirichlet_alpha)
            aw = float(self._cfg.root_noise_weight)
            noise = self._rng.dirichlet([alpha] * self._cfg.model.action_space_size).astype(np.float32).tolist()
            roots.prepare(aw, [noise], [reward_val], [policy_logits], to_play)
        else:
            roots.prepare_no_noise([reward_val], [policy_logits], to_play)

        # Run search with a list of per-root latent states (one root here)
        # Each element must be a single-root latent without batch dim.
        if self.predict_use_ctree:
            MCTSCtree(self._cfg).search(roots, model, [latent_state_np], to_play)
        else:
            MCTSPtree(self._cfg).search(roots, model, [latent_state_np], to_play)

        # Collect results
        dists = roots.get_distributions()
        values = roots.get_values()

        if dists and len(dists) > 0 and dists[0] is not None:
            visits = np.asarray(dists[0], dtype=np.float32)
            vsum = float(visits.sum()) if visits.size > 0 else 0.0
            policy = visits / vsum if vsum > 0 else np.ones_like(visits) / max(1, len(visits))
        else:
            policy = np.ones(self._cfg.model.action_space_size, dtype=np.float32) / self._cfg.model.action_space_size
            visits = np.ones(self._cfg.model.action_space_size, dtype=np.float32)

        root_value = np.asarray([values[0]], dtype=np.float32) if values and len(values) > 0 \
            else np.asarray([0.0], dtype=np.float32)

        return {
            "policy": policy.astype(np.float32),  # normalized visit distribution
            "visit": visits,  # raw visit counts
            "value": root_value,  # searched root value
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
        """
        Return True iff `x` looks like a slice tuple (start, end).

        Accept both built-in ints and NumPy integer scalars
        (e.g., numpy.int32/int64) so tuples produced by np.random.choice
        are correctly recognized as (s, e) slices.
        """
        if not isinstance(x, (tuple, list)) or len(x) != 2:
            return False
        a, b = x

        # bool is a subclass of int; exclude it explicitly.
        import numpy as _np
        def _is_int_like(v: Any) -> bool:
            # Python int or NumPy integer scalar
            return (isinstance(v, int) and not isinstance(v, bool)) or isinstance(v, _np.integer)

        return _is_int_like(a) and _is_int_like(b)

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
        """
        Build metadata for a sub-segment [s, e) derived from a base segment.

        Guarantees:
        - Always provides the 'priorities' key required by the parent buffer.
        - Offsets are normalized to Python ints.
        - Includes binding tags so train->base priority propagation works.
        """
        # Normalize to Python ints and compute slice length
        s, e = int(s), int(e)
        L = max(0, e - s)
        out: Dict[str, Any] = {}

        # ---- Priorities (ALWAYS present) ---------------------------------------
        # Source from our base-level per-position priorities (exists for every base segment).
        try:
            base_pri = self.base_pos_priorities[base_id]
            end_idx = min(e, len(base_pri))  # clamp to avoid OOB
            pri_slice = np.asarray(base_pri[s:end_idx], dtype=np.float32)
            # If something goes wrong (e.g., empty slice), fall back to zeros of length L.
            if pri_slice.size != L:
                pri_slice = np.zeros((L,), dtype=np.float32)
        except Exception:
            pri_slice = np.zeros((L,), dtype=np.float32)
        out['priorities'] = pri_slice

        # ---- Unroll horizon relative to new origin -----------------------------
        # Keep the same semantic as parent: cap by slice length and shift by 's'.
        orig_upt = int(base_meta.get('unroll_plus_td_steps', e))
        out['unroll_plus_td_steps'] = max(0, min(L, orig_upt - s))

        # ---- Done flag (only if this slice reaches the end of the episode) -----
        is_last_slice = (e == len(self.base_pos_priorities[base_id]))
        out['done'] = bool(base_meta.get('done', False) and is_last_slice)

        # ---- Binding tags for priority back-propagation ------------------------
        out['base_id'] = int(base_id)
        out['slice'] = (s, e)
        out['_offset0'] = s

        # ---- Optional passthrough fields ---------------------------------------
        if 'make_time' in base_meta:
            out['make_time'] = base_meta['make_time']

        return out

    def _clear_train_view(self) -> None:
        """Clear only TRAIN view; BASE remains intact."""
        self.game_segment_buffer.clear()
        # Keep the same type the parent expects: a Python list.
        if isinstance(self.game_pos_priorities, list):
            self.game_pos_priorities.clear()
        else:
            # If some path made it a numpy array, normalize it back to list.
            self.game_pos_priorities = []
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

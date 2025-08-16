import json


class Plan:
    def __init__(self, plan_path, rank):
        """Initialize plan for a specific rank."""
        with open(plan_path, "r") as f:
            self.phases = json.load(f)
        self.rank = rank
        # Auto-detect starting phase for this rank
        self.current_phase = self._find_starting_phase()
        self.starting_phase = self.current_phase  # Store the starting phase

    def _find_starting_phase(self):
        """Find the first phase where this rank appears."""
        for i, phase_ranks in enumerate(self.phases):
            if self.rank in phase_ranks:
                return i
        raise ValueError(f"Rank {self.rank} not found in any phase")

    def get_new_ranks(self):
        """Get ranks to connect to at current phase (only the new ones)."""
        if self.current_phase == self.starting_phase:
            # First phase for this rank: connect to all other ranks in the phase
            return [r for r in self.phases[self.current_phase] if r != self.rank]
        else:
            # Later phases: only connect to newly added ranks
            prev_ranks = set(self.phases[self.current_phase - 1])
            curr_ranks = set(self.phases[self.current_phase])
            new_ranks = curr_ranks - prev_ranks
            return [r for r in new_ranks if r != self.rank]

    def get_removed_ranks(self):
        """Get ranks to remove at current phase."""
        if self.current_phase == self.starting_phase:
            # First phase: no ranks to remove
            return []
        
        prev_ranks = set(self.phases[self.current_phase - 1])
        curr_ranks = set(self.phases[self.current_phase])
        
        # Removal phase: return ALL ranks to remove
        return list(prev_ranks - curr_ranks)

    def get_active_ranks(self):
        """Get all active ranks in current phase."""
        return self.phases[self.current_phase]

    def get_max_rank(self):
        """Get the maximum participating rank index"""
        return max(max(phase) for phase in self.phases)

    def get_min_active_ranks(self):
        """Get the minimum number of active ranks in all phases."""
        return min([len(phase) for phase in self.phases])

    def next(self):
        """Advance to next phase."""
        if self.current_phase < len(self.phases) - 1:
            self.current_phase += 1
            return True
        return False

    def get_phase(self):
        """Get current phase index."""
        return self.current_phase

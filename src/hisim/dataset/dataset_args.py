from dataclasses import dataclass
from typing import Optional


@dataclass
class DatasetArgs:
    name: str = ""
    filepath: str = ""
    num_prompts: int = -1
    min_input_len: int = -1
    max_input_len: int = -1
    min_output_len: int = -1
    max_output_len: int = -1
    prefix_hit_rate: Optional[float] = None
    # weka_trace dataset arguments (see hisim.dataset.weka_trace).
    num_traces: int = -1
    max_concurrent_traces: int = 1
    trace_start_interval: float = 0.5
    max_rounds_per_trace: int = -1
    max_round_tokens: int = -1
    include_subagents: bool = False

    @property
    def mean_input_length(self):
        return (self.min_input_len + self.max_input_len) // 2

    @property
    def mean_output_length(self):
        return (self.min_output_len + self.max_output_len) // 2

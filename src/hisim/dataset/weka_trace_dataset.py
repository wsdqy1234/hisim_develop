from hisim.dataset.base_dataset import BaseDataset, GenericRequest
from hisim.dataset.dataset_args import DatasetArgs
from hisim.dataset.weka_trace import split_filepaths, plan_weka_traces, expand_hash_ids
from hisim.utils import get_logger


logger = get_logger("hisim")


class WekaTraceDataset(BaseDataset):
    """Replay dataset built from SemiAnalysis cc-traces-weka agentic traces.

    See `hisim.dataset.weka_trace` for the trace format and the scheduling
    model. Requests are expanded lazily in `__getitem__`, so memory stays
    proportional to the number of rounds (hash ids only), not to the total
    token count. Timestamps are static wall-clock plans consumed by hisim's
    virtual-clock scheduler.
    """

    def __init__(self, tokenizer, args: DatasetArgs):
        super().__init__(tokenizer, args)
        self._name = "weka_trace"
        filepaths = split_filepaths(args.filepath)
        if not filepaths:
            raise ValueError(
                "weka_trace requires one or more traces.jsonl paths in "
                "`filepath` (separated by ';')."
            )
        self.plan = plan_weka_traces(
            filepaths,
            num_traces=args.num_traces,
            max_concurrent_traces=args.max_concurrent_traces,
            trace_start_interval=args.trace_start_interval,
            max_rounds_per_trace=args.max_rounds_per_trace,
            max_round_tokens=args.max_round_tokens,
            include_subagents=args.include_subagents,
        )
        logger.info(
            f"WekaTraceDataset: {self.plan.num_traces} traces, "
            f"{len(self.plan.rounds)} rounds, "
            f"block_size={self.plan.block_size}."
        )

    def __len__(self) -> int:
        return len(self.plan.rounds)

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return [self[i] for i in range(start, stop, step)]

        if index >= len(self):
            raise IndexError

        rd = self.plan.rounds[index]
        return GenericRequest(
            token_ids=expand_hash_ids(
                rd.hash_ids, rd.block_base, self.plan.block_size
            ),
            input_length=rd.input_length,
            output_length=rd.output_length,
            custom_params={
                "created_time": rd.timestamp,
                "rid": f"weka-{rd.trace_idx}-{rd.stream_id}-{rd.round_idx}",
                # Session metadata for the dynamic-completion-feedback
                # scheduler (sglang_hook): rounds of the same session must
                # wait for max(static time, previous round's actual finish).
                "session": f"{rd.trace_idx}:{rd.stream_id}",
                "round_idx": rd.round_idx,
                "recorded_finish": rd.recorded_finish,
            },
        )

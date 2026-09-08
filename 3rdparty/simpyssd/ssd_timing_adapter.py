import math
import random

from .sim import engine, Event, Sim
from .ssd_estimate import SSD, Cmd
from .ssd_config import ssd_config


class StubSystem:
    stat = None

    def process(self, cmd):
        pass


class SSDTimingSimulator:
    def __init__(self, cfg_path):
        ssd_config.read_conf_file(cfg_path)
        self._ssd = SSD(StubSystem())

    def _reset(self):
        engine.now = 0
        engine.events = []
        ssd = self._ssd
        ssd.issued_cmd = set()
        ssd.finished_cmd = set()
        for ch in ssd.channels:
            ch.transfer_cmd = None
            ch.queued_cmd = []
            ch.avail_time = 0
            for chip in ch.chips:
                chip.exec_cmd = None
                chip.exec_done = True
                chip.transfer_cmd = None
                chip.transfer_done = True
                chip.queued_cmd = []
                chip.avail_time = 0
                for buf in chip.buffer:
                    buf.empty = True
                    buf.cmd = None
                    buf.status = 0

    def predict_read_time(self, total_bytes, io_block_size,
                          is_sequential=False, queue_depth=1):
        return self._predict('read', total_bytes, io_block_size,
                             is_sequential, queue_depth)

    def predict_write_time(self, total_bytes, io_block_size,
                           is_sequential=False, queue_depth=1):
        return self._predict('write', total_bytes, io_block_size,
                             is_sequential, queue_depth)

    def _predict(self, cmd_typ, total_bytes, io_block_size,
                 is_sequential, queue_depth):
        self._reset()
        num_io = max(1, math.ceil(total_bytes / io_block_size))
        n_ch = ssd_config.num_channel
        n_chip = ssd_config.num_chip
        for i in range(num_io):
            if is_sequential:
                ch_id = i % n_ch
                chip_id = (i // n_ch) % n_chip
            else:
                ch_id = random.randrange(n_ch)
                chip_id = random.randrange(n_chip)
            cmd = Cmd(cmd_typ, ch_id, chip_id, data_sz=io_block_size)
            self._ssd.issue(cmd)
        while engine.events:
            engine.exec()
        return engine.now * 1e-6

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
        """Full reset: engine + all chip state (including avail_time)"""
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

    def _soft_reset(self):
        """Soft reset: engine + exec state, but keep avail_time (for timing simulation)"""
        engine.events = []
        ssd = self._ssd
        ssd.issued_cmd = set()
        ssd.finished_cmd = set()
        for ch in ssd.channels:
            ch.transfer_cmd = None
            ch.queued_cmd = []
            for chip in ch.chips:
                chip.exec_cmd = None
                chip.exec_done = True
                chip.transfer_cmd = None
                chip.transfer_done = True
                chip.queued_cmd = []
                for buf in chip.buffer:
                    buf.empty = True
                    buf.cmd = None
                    buf.status = 0

    def predict_read_time(self, total_bytes, io_block_size,
                          is_sequential=False, queue_depth=1):
        """Original API: no timing, full reset each call"""
        return self._predict('read', total_bytes, io_block_size,
                             is_sequential, queue_depth)

    def predict_write_time(self, total_bytes, io_block_size,
                           is_sequential=False, queue_depth=1):
        """Original API: no timing, full reset each call"""
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

    # ===== Timing-aware API =====

    def submit_write(self, arrival_us, data_bytes, io_block_size,
                     ch_id=0, chip_id=0):
        """
        Submit a write request with arrival time. Returns (completion_us, latency_us, wait_us, exec_us).
        Advances chip avail_time (write effect persists for subsequent requests).
        """
        return self._submit_timed('write', arrival_us, data_bytes,
                                   io_block_size, ch_id, chip_id)

    def submit_read(self, arrival_us, data_bytes, io_block_size,
                    ch_id=0, chip_id=0):
        """
        Submit a read request with arrival time. Returns (completion_us, latency_us, wait_us, exec_us).
        Also advances chip avail_time. Call set_avail_time() after to undo read's effect if needed.
        """
        return self._submit_timed('read', arrival_us, data_bytes,
                                  io_block_size, ch_id, chip_id)

    def _submit_timed(self, cmd_typ, arrival_us, data_bytes, io_block_size,
                      ch_id, chip_id):
        # Set engine time to arrival
        engine.now = arrival_us
        # Soft reset: clear exec state but keep avail_time
        self._soft_reset()

        # Record avail_time before submission
        chip = self._ssd.channels[ch_id].chips[chip_id]
        avail_before = chip.avail_time

        # Create and submit command
        num_io = max(1, math.ceil(data_bytes / io_block_size))
        n_ch = ssd_config.num_channel
        n_chip = ssd_config.num_chip
        cmds = []
        for i in range(num_io):
            c = ch_id if ch_id is not None else (i % n_ch)
            p = chip_id if chip_id is not None else ((i // n_ch) % n_chip)
            cmd = Cmd(cmd_typ, c, p, data_sz=io_block_size)
            cmds.append(cmd)
            self._ssd.issue(cmd)

        # Track completion using TrackingSet
        completion_time = [None]
        class TrackingSet(set):
            def add(self_inner, c):
                super().add(c)
                if c in cmds and completion_time[0] is None:
                    completion_time[0] = engine.now
        self._ssd.finished_cmd = TrackingSet(self._ssd.finished_cmd)

        # Run event loop until all commands complete
        safety = 0
        while engine.events:
            engine.exec()
            if completion_time[0] is not None:
                break
            safety += 1
            if safety > 1000000:
                break

        # Drain remaining events for this chip
        while engine.events:
            engine.exec()

        comp = completion_time[0] or 0
        total_lat = comp - arrival_us
        wait = max(0, avail_before - arrival_us)
        exec_t = total_lat - wait
        return comp, total_lat, wait, exec_t

    def get_avail_time(self, ch_id=0, chip_id=0):
        """Get chip avail_time (for snapshot before read)"""
        return self._ssd.channels[ch_id].chips[chip_id].avail_time

    def set_avail_time(self, ch_id, chip_id, value):
        """Restore chip avail_time (undo read's effect after interruption)"""
        self._ssd.channels[ch_id].chips[chip_id].avail_time = value

    def reset_avail_time(self):
        """Reset all chips' avail_time to 0 (full reset for init/switch)"""
        for ch in self._ssd.channels:
            for chip in ch.chips:
                chip.avail_time = 0

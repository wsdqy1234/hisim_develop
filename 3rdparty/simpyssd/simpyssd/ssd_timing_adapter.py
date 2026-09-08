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
                     is_sequential=True):
        """
        Submit a write request with arrival time.
        I/Os are striped across channels/chips (like _predict).
        Returns (completion_us, latency_us, wait_us, exec_us).
        Advances all affected chips' avail_time (write effect persists).
        """
        return self._submit_timed('write', arrival_us, data_bytes,
                                   io_block_size, is_sequential)

    def submit_read(self, arrival_us, data_bytes, io_block_size,
                    is_sequential=True):
        """
        Submit a read request with arrival time.
        I/Os are striped across channels/chips.
        Returns (completion_us, latency_us, wait_us, exec_us).
        Also advances all affected chips' avail_time.
        Call set_all_avail_time() after to undo read's effect if needed.
        """
        return self._submit_timed('read', arrival_us, data_bytes,
                                  io_block_size, is_sequential)

    def _submit_timed(self, cmd_typ, arrival_us, data_bytes, io_block_size,
                      is_sequential):
        # Set engine time to arrival
        engine.now = arrival_us
        # Soft reset: clear exec state but keep avail_time on ALL chips
        self._soft_reset()

        # Snapshot avail_time of ALL chips before submission
        # (to compute max wait across all chips)
        avail_before = {}
        for ci, ch in enumerate(self._ssd.channels):
            for pi, chip in enumerate(ch.chips):
                avail_before[(ci, pi)] = chip.avail_time

        # Create and submit all commands, striped across channels/chips
        num_io = max(1, math.ceil(data_bytes / io_block_size))
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

        # Run ALL events to completion
        safety = 0
        while engine.events:
            engine.exec()
            safety += 1
            if safety > 10000000:
                break

        # Completion = engine.now after ALL I/Os finish
        comp = engine.now
        total_lat = comp - arrival_us

        # wait = max wait across all chips that were used
        # (the chip that took the longest to become available)
        max_wait = 0.0
        for i in range(num_io):
            if is_sequential:
                ch_id = i % n_ch
                chip_id = (i // n_ch) % n_chip
            else:
                # For random, we can't reconstruct, so use overall max
                pass
            key = (ch_id, chip_id) if is_sequential else None
            if key and key in avail_before:
                w = max(0, avail_before[key] - arrival_us)
                if w > max_wait:
                    max_wait = w

        # For non-sequential, compute max across all chips
        if not is_sequential:
            for key, avail in avail_before.items():
                w = max(0, avail - arrival_us)
                if w > max_wait:
                    max_wait = w

        wait = max_wait
        exec_t = total_lat - wait
        return comp, total_lat, wait, exec_t

    def get_all_avail_time(self):
        """Get avail_time of ALL chips as dict {(ch, chip): time}"""
        result = {}
        for ci, ch in enumerate(self._ssd.channels):
            for pi, chip in enumerate(ch.chips):
                result[(ci, pi)] = chip.avail_time
        return result

    def set_all_avail_time(self, snapshot):
        """Restore all chips' avail_time from a snapshot (undo read's effect)"""
        for (ci, pi), avail in snapshot.items():
            self._ssd.channels[ci].chips[pi].avail_time = avail

    def get_avail_time(self, ch_id=0, chip_id=0):
        """Get single chip's avail_time"""
        return self._ssd.channels[ch_id].chips[chip_id].avail_time

    def set_avail_time(self, ch_id, chip_id, value):
        """Set single chip's avail_time"""
        self._ssd.channels[ch_id].chips[chip_id].avail_time = value

    def reset_avail_time(self):
        """Reset all chips' avail_time to 0"""
        for ch in self._ssd.channels:
            for chip in ch.chips:
                chip.avail_time = 0

"""Estimate non-Basalt dedicated VRAM on Windows without exposing process data."""
from __future__ import annotations

import ctypes as ct
import re
from ctypes import wintypes as wt

_ADAPTER = r"luid_0x[0-9a-f]{8}_0x[0-9a-f]{8}_phys_\d+"
_PROCESS = re.compile(rf"pid_(\d+)_({_ADAPTER})", re.IGNORECASE)
_MORE_DATA = 0x800007D2
_LARGE_UNSCALED = 0x400 | 0x1000


class _Value(ct.Structure):
    # PDH_FMT_COUNTERVALUE's union is 64 bits; PDH_FMT_LARGE selects its LONGLONG.
    _fields_ = [("status", wt.DWORD), ("value", ct.c_longlong)]


class _Item(ct.Structure):
    _fields_ = [("name", wt.LPWSTR), ("counter", _Value)]


def _adapter_keys(gpus):
    """Join CUDA's WDDM LUID to NVML via PCI, independent of device order."""
    cuda = ct.WinDLL("nvcuda.dll", winmode=0x800)
    count = ct.c_int()
    if cuda.cuInit(0) or cuda.cuDeviceGetCount(ct.byref(count)):
        return {}
    keys = {}
    for ordinal in range(count.value):
        dev, bus, luid, mask = ct.c_int(), ct.create_string_buffer(32), ct.create_string_buffer(8), ct.c_uint()
        if (cuda.cuDeviceGet(ct.byref(dev), ordinal) or cuda.cuDeviceGetPCIBusId(bus, len(bus), dev)
                or cuda.cuDeviceGetLuid(luid, ct.byref(mask), dev) or not mask.value):
            continue
        low = int.from_bytes(luid.raw[:4], "little")
        high = int.from_bytes(luid.raw[4:], "little")
        node = mask.value.bit_length() - 1
        if mask.value != 1 << node:
            continue  # no single PDH physical adapter instance
        for index, g in gpus:
            lib, handle = getattr(g, "lib", None), ct.c_void_p()
            lookup = getattr(lib, "nvmlDeviceGetHandleByPciBusId_v2", None)
            if lookup is not None and lookup(bus, ct.byref(handle)) == 0 and handle.value == g.dev.value:
                keys[index] = f"luid_0x{high:08x}_0x{low:08x}_phys_{node}"
    return keys


class DedicatedMemory:
    def __init__(self, gpus):
        self.keys = _adapter_keys(gpus)
        self.query = ct.c_void_p()
        self.adapter, self.process = ct.c_void_p(), ct.c_void_p()
        if not self.keys:
            return
        self.pdh = ct.WinDLL("pdh.dll", winmode=0x800)
        if self.pdh.PdhOpenQueryW(None, 0, ct.byref(self.query)):
            return
        add = self.pdh.PdhAddEnglishCounterW
        add.argtypes = (ct.c_void_p, wt.LPCWSTR, ct.c_size_t, ct.POINTER(ct.c_void_p))
        if (add(self.query, r"\GPU Adapter Memory(*)\Dedicated Usage", 0, ct.byref(self.adapter))
                or add(self.query, r"\GPU Process Memory(*)\Dedicated Usage", 0, ct.byref(self.process))):
            self.close()

    def close(self):
        if self.query:
            self.pdh.PdhCloseQuery(self.query)
            self.query = ct.c_void_p()

    def _rows(self, counter):
        size, count = wt.DWORD(), wt.DWORD()
        get = self.pdh.PdhGetFormattedCounterArrayW
        status = get(counter, _LARGE_UNSCALED, ct.byref(size), ct.byref(count), None)
        if (status & 0xffffffff) != _MORE_DATA or not size.value:
            return {}
        data = ct.create_string_buffer(size.value)
        if get(counter, _LARGE_UNSCALED, ct.byref(size), ct.byref(count), data):
            return {}
        return {item.name.lower(): item.counter.value
                for item in ct.cast(data, ct.POINTER(_Item * count.value)).contents
                if item.name and item.counter.status in (0, 1) and item.counter.value >= 0}

    def sample(self, owned_pids):
        if not self.query or not owned_pids or self.pdh.PdhCollectQueryData(self.query):
            return {}
        adapter = self._rows(self.adapter)
        owned = {}
        for name, size in self._rows(self.process).items():
            m = _PROCESS.fullmatch(name)
            if m and int(m[1]) in owned_pids:
                owned[m[2]] = owned.get(m[2], 0) + size
        # Subtract our processes from the adapter total. Adding up external processes
        # would double count shared surfaces (notably the Desktop Window Manager).
        return {i: adapter[key] - owned[key] for i, key in self.keys.items()
                if key in adapter and key in owned and 0 < owned[key] <= adapter[key]}

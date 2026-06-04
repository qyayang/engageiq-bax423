"""
Bloom Filter for streaming deduplication.
BAX-423 Lecture 2 — Data Sketching & Efficient Learning.

Space-efficient probabilistic data structure.
Optimal size: m = -n*ln(p) / (ln2)^2
Optimal hash count: k = m/n * ln2
"""
import math
import hashlib


class BloomFilter:
    def __init__(self, capacity: int = 200_000, error_rate: float = 0.01):
        self.capacity = capacity
        self.error_rate = error_rate
        self.size = self._optimal_size(capacity, error_rate)
        self.hash_count = self._optimal_hash_count(self.size, capacity)
        self.bit_array = bytearray(self.size // 8 + 1)
        self.item_count = 0

    @staticmethod
    def _optimal_size(n: int, p: float) -> int:
        return int(-n * math.log(p) / (math.log(2) ** 2))

    @staticmethod
    def _optimal_hash_count(m: int, n: int) -> int:
        return max(1, int((m / n) * math.log(2)))

    def _hashes(self, item: str):
        item_bytes = item.encode("utf-8")
        for i in range(self.hash_count):
            digest = int(
                hashlib.sha256(item_bytes + i.to_bytes(4, "little")).hexdigest(), 16
            )
            yield digest % self.size

    def _get_bit(self, pos: int) -> bool:
        return bool(self.bit_array[pos >> 3] & (1 << (pos & 7)))

    def _set_bit(self, pos: int):
        self.bit_array[pos >> 3] |= 1 << (pos & 7)

    def add(self, item: str):
        for pos in self._hashes(item):
            self._set_bit(pos)
        self.item_count += 1

    def contains(self, item: str) -> bool:
        return all(self._get_bit(pos) for pos in self._hashes(item))

    def stats(self) -> dict:
        bits_set = sum(bin(b).count("1") for b in self.bit_array)
        actual_fpr = (bits_set / self.size) ** self.hash_count if self.size > 0 else 0
        return {
            "size_bits": self.size,
            "size_kb": round(self.size / 8 / 1024, 2),
            "hash_functions": self.hash_count,
            "items_added": self.item_count,
            "bits_set": bits_set,
            "fill_ratio": round(bits_set / self.size, 4) if self.size else 0,
            "estimated_fpr": round(actual_fpr, 6),
        }

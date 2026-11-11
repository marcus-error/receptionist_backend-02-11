import time
import ctypes

try:
    ctypes.windll.winmm.timeBeginPeriod(1)
except Exception:
    pass

class RTPTransmitQueue:
    def __init__(self):
        import threading
        self._lock = threading.Lock()
        self._buffer = bytearray()
        self.rebuilding = False

    def write(self, offset: int = 0, data: bytes = b''):
        if not data:
            return
        with self._lock:
            self._buffer.extend(data)

    def read(self, length: int = 160):
        with self._lock:
            buf_len = len(self._buffer)
            if buf_len >= length:
                pkt = bytes(self._buffer[:length])
                del self._buffer[:length]
                return pkt
            elif buf_len > 0:
                pkt = bytes(self._buffer) + (b'\xff' * (length - buf_len))
                self._buffer.clear()
                return pkt
            else:
                return b'\xff' * length

    def clear(self):
        with self._lock:
            self._buffer.clear()

    @property
    def queue_depth_packets(self):
        with self._lock:
            return len(self._buffer) // 160

queue = RTPTransmitQueue()

# Simulate AI writing 2 sentences
queue.write(0, b'S1' * 80 * 20)  # 20 packets of Sentence 1 (3200 bytes)
queue.write(0, b'S2' * 80 * 20)  # 20 packets of Sentence 2 (3200 bytes)

# Transmit loop
outSequence = 1000
outTimestamp = 50000
frame_duration = 0.020
next_time = time.perf_counter()

sent_packets = []
t_start = time.perf_counter()

for i in range(50):
    payload = queue.read(160)
    source = 'SILENCE'
    if payload.startswith(b'S1'): source = 'SENTENCE_1'
    elif payload.startswith(b'S2'): source = 'SENTENCE_2'
    
    sent_packets.append({
        'seq': outSequence,
        'ts': outTimestamp,
        'source': source,
        'time': time.perf_counter() - t_start
    })
    
    outSequence = (outSequence + 1) & 0xFFFF
    outTimestamp = (outTimestamp + len(payload)) & 0xFFFFFFFF
    
    next_time += frame_duration
    sleep_time = next_time - time.perf_counter()
    if sleep_time > 0:
        time.sleep(sleep_time)
    elif sleep_time < -0.100:
        next_time = time.perf_counter()

print("Total packets sent:", len(sent_packets))
print("First packet:", sent_packets[0])
print("Packet 20 (start S2):", sent_packets[20])
print("Packet 40 (silence):", sent_packets[40])
print("Last packet:", sent_packets[-1])

# Check continuity
for j in range(1, len(sent_packets)):
    assert sent_packets[j]['seq'] == sent_packets[j-1]['seq'] + 1, 'Sequence gap!'
    assert sent_packets[j]['ts'] == sent_packets[j-1]['ts'] + 160, 'Timestamp gap!'
print("ALL SEQUENCE NUMBERS AND TIMESTAMPS STRICTLY CONTINUOUS!")


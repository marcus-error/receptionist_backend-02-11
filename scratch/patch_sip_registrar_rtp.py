with open('src/services/sip_registrar.py', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Add timer resolution boost near top
old_top = '''    # Set transmit delay reduction to 0.25 for low-latency RTP pacing
    if hasattr(pyVoIP, "TRANSMIT_DELAY_REDUCTION"):
        pyVoIP.TRANSMIT_DELAY_REDUCTION = 0.25'''

new_top = '''    # Set transmit delay reduction to 0.25 for low-latency RTP pacing
    if hasattr(pyVoIP, "TRANSMIT_DELAY_REDUCTION"):
        pyVoIP.TRANSMIT_DELAY_REDUCTION = 0.25

    # Boost Windows multimedia timer resolution to 1ms for sub-millisecond RTP pacing
    import ctypes
    try:
        ctypes.windll.winmm.timeBeginPeriod(1)
    except Exception:
        pass'''

assert old_top in text, "old_top not found in sip_registrar.py"
text = text.replace(old_top, new_top, 1)

# 2. Add RTPTransmitQueue and patch RTPClient init, pm_read, and trans
old_block = '''        # 1. Patch RTPPacketManager to pad with standard G.711 PCMU silence bytes (0xFF)
        _orig_read = pyVoIP.RTP.RTPPacketManager.read
        def _patched_pm_read(self_pm, length=160):
            import time
            while self_pm.rebuilding:
                time.sleep(0.01)
            self_pm.bufferLock.acquire()
            packet = self_pm.buffer.read(length)
            if len(packet) < length:
                pad_byte = b"\\xff"  # Standard G.711 PCMU (mu-law) silence byte (0xFF = 0 amplitude)
                packet = packet + (pad_byte * (length - len(packet)))
            self_pm.bufferLock.release()
            return packet
        pyVoIP.RTP.RTPPacketManager.read = _patched_pm_read

        def _patched_pm_clear(self_pm):
            import io
            try:
                self_pm.bufferLock.acquire()
                self_pm.buffer = io.BytesIO()
                self_pm.log = {}
                self_pm.offset = 4294967296
            finally:
                self_pm.bufferLock.release()
        pyVoIP.RTP.RTPPacketManager.clear = _patched_pm_clear

        def _patched_rtp_clear_tx(self_rtp):
            if hasattr(self_rtp, "pmout") and self_rtp.pmout:
                import io
                with getattr(self_rtp.pmout, "bufferLock", None) or nullcontext():
                    self_rtp.pmout.buffer = io.BytesIO()
                    self_rtp.pmout.log = {}
                    self_rtp.pmout.offset = getattr(self_rtp, "outOffset", 0)
        pyVoIP.RTP.RTPClient.clear_tx_buffer = _patched_rtp_clear_tx

        # 2. Patch encode_pcmu and encode_pcma to pass through pre-encoded 160-byte G.711 PCMU packets directly without corrupting voice samples
        def _patched_encode_pcmu(self, packet: bytes) -> bytes:
            if not packet or packet == b"\\xff" * len(packet):
                return b"\\xff" * 160
            if len(packet) < 160:
                return packet + (b"\\xff" * (160 - len(packet)))
            return packet[:160]
        pyVoIP.RTP.RTPClient.encode_pcmu = _patched_encode_pcmu
        pyVoIP.RTP.RTPClient.encodePCMU = _patched_encode_pcmu
        pyVoIP.RTP.RTPClient.encode_pcma = _patched_encode_pcmu
        pyVoIP.RTP.RTPClient.encodePCMA = _patched_encode_pcmu

        # 3. Patch RTP trans thread to use exact 20ms drift-free pacing and dynamic payload type
        def _patched_trans(self):
            import time
            import logging
            logger = logging.getLogger(__name__)
            next_time = time.perf_counter()
            frame_duration = 0.020  # 20ms pacing (160 samples @ 8000Hz)
            pkt_count = 0
            
            try:
                pt = int(self.preference)
            except Exception:
                pt = 0

            logger.info(
                f"[RTP_TX] Initialized RTP transmitter | dest={self.outIP}:{self.outPort} | "
                f"src={self.inIP}:{self.inPort} | payload_type={pt} | ssrc={self.outSSRC}"
            )
            
            if hasattr(self, "sout") and self.sout:
                try:
                    self.sout.setblocking(True)
                    self.sout.settimeout(0.05)
                except Exception:
                    pass

            while self.NSD:
                payload = self.pmout.read(160)  # Read 160 PCMU bytes (20ms)
                payload = self.encode_packet(payload)  # outputs 160 bytes PCMU
                
                packet = bytearray([0x80])  # V=2, P=0, X=0, CC=0
                packet.append(pt)  # Dynamic PT from SDP negotiation (0=PCMU, 8=PCMA)
                try:
                    packet.extend(self.outSequence.to_bytes(2, byteorder="big"))
                except OverflowError:
                    self.outSequence = 0
                    packet.extend(self.outSequence.to_bytes(2, byteorder="big"))
                try:
                    packet.extend(self.outTimestamp.to_bytes(4, byteorder="big"))
                except OverflowError:
                    self.outTimestamp = 0
                    packet.extend(self.outTimestamp.to_bytes(4, byteorder="big"))
                    
                packet.extend(self.outSSRC.to_bytes(4, byteorder="big"))
                packet.extend(payload)
                
                try:
                    self.sout.sendto(bytes(packet), (self.outIP, self.outPort))
                    pkt_count += 1
                    if pkt_count % 100 == 1:
                        logger.info(
                            f"[RTP_TX] dest={self.outIP}:{self.outPort} | payload_type=0 | "
                            f"sequence={self.outSequence} | timestamp={self.outTimestamp} | "
                            f"ssrc={self.outSSRC} | payload_bytes={len(payload)}"
                        )
                except (BlockingIOError, OSError) as err:
                    if getattr(err, "winerror", None) == 10035 or getattr(err, "errno", None) in (10035, 11, 35):
                        import select
                        try:
                            _, writable, _ = select.select([], [self.sout], [], 0.005)
                            if writable:
                                self.sout.sendto(bytes(packet), (self.outIP, self.outPort))
                                pkt_count += 1
                        except Exception:
                            pass
                    else:
                        logger.warning(f"[RTP_TX] sendto error: {err}")
                
                self.outSequence = (self.outSequence + 1) & 0xFFFF
                self.outTimestamp = (self.outTimestamp + len(payload)) & 0xFFFFFFFF
                
                next_time += frame_duration
                sleep_time = next_time - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    next_time = time.perf_counter()
        
        pyVoIP.RTP.RTPClient.trans = _patched_trans'''

new_block = '''        # 1. Authoritative Outbound RTP Transmit Queue (FIFO, zero-drift, zero memory growth)
        class RTPTransmitQueue:
            """Thread-safe FIFO byte buffer for authoritative outbound RTP transmission."""
            def __init__(self):
                import threading
                self._lock = threading.Lock()
                self._buffer = bytearray()
                self.rebuilding = False

            def write(self, offset: int = 0, data: bytes = b"") -> None:
                if not data:
                    return
                with self._lock:
                    self._buffer.extend(data)

            def read(self, length: int = 160) -> bytes:
                with self._lock:
                    buf_len = len(self._buffer)
                    if buf_len >= length:
                        packet = bytes(self._buffer[:length])
                        del self._buffer[:length]
                        return packet
                    elif buf_len > 0:
                        packet = bytes(self._buffer) + (b"\\xff" * (length - buf_len))
                        self._buffer.clear()
                        return packet
                    else:
                        return b"\\xff" * length

            def clear(self) -> None:
                with self._lock:
                    self._buffer.clear()

            @property
            def queue_depth_packets(self) -> int:
                with self._lock:
                    return len(self._buffer) // 160

        # Patch RTPClient.__init__ to equip each call with the authoritative FIFO transmit queue
        _orig_rtp_init = pyVoIP.RTP.RTPClient.__init__
        def _patched_rtp_init(self_rtp, *args, **kwargs):
            _orig_rtp_init(self_rtp, *args, **kwargs)
            self_rtp.pmout = RTPTransmitQueue()
        pyVoIP.RTP.RTPClient.__init__ = _patched_rtp_init

        # 2. Patch encode_pcmu and encode_pcma to pass through pre-encoded 160-byte G.711 PCMU packets directly
        def _patched_encode_pcmu(self, packet: bytes) -> bytes:
            if not packet or packet == b"\\xff" * len(packet):
                return b"\\xff" * 160
            if len(packet) < 160:
                return packet + (b"\\xff" * (160 - len(packet)))
            return packet[:160]
        pyVoIP.RTP.RTPClient.encode_pcmu = _patched_encode_pcmu
        pyVoIP.RTP.RTPClient.encodePCMU = _patched_encode_pcmu
        pyVoIP.RTP.RTPClient.encode_pcma = _patched_encode_pcmu
        pyVoIP.RTP.RTPClient.encodePCMA = _patched_encode_pcmu

        # 3. Patch RTP trans thread: single authoritative sender with 20ms drift-free pacing and monotonic timestamps
        def _patched_trans(self):
            import time
            import logging
            logger = logging.getLogger(__name__)
            next_time = time.perf_counter()
            frame_duration = 0.020  # 20ms pacing (160 samples @ 8000Hz)
            pkt_count = 0
            
            try:
                pt = int(self.preference)
            except Exception:
                pt = 0

            logger.info(
                f"[RTP_TX] Initialized authoritative RTP transmitter | dest={self.outIP}:{self.outPort} | "
                f"src={self.inIP}:{self.inPort} | payload_type={pt} | ssrc={self.outSSRC}"
            )
            
            if hasattr(self, "sout") and self.sout:
                try:
                    self.sout.setblocking(True)
                    self.sout.settimeout(0.05)
                except Exception:
                    pass

            while self.NSD:
                payload = self.pmout.read(160)  # Read 160 PCMU bytes (20ms)
                payload = self.encode_packet(payload)  # outputs 160 bytes PCMU
                
                packet = bytearray([0x80])  # V=2, P=0, X=0, CC=0
                packet.append(pt)  # Dynamic PT from SDP negotiation (0=PCMU, 8=PCMA)
                try:
                    packet.extend(self.outSequence.to_bytes(2, byteorder="big"))
                except OverflowError:
                    self.outSequence = 0
                    packet.extend(self.outSequence.to_bytes(2, byteorder="big"))
                try:
                    packet.extend(self.outTimestamp.to_bytes(4, byteorder="big"))
                except OverflowError:
                    self.outTimestamp = 0
                    packet.extend(self.outTimestamp.to_bytes(4, byteorder="big"))
                    
                packet.extend(self.outSSRC.to_bytes(4, byteorder="big"))
                packet.extend(payload)
                
                try:
                    self.sout.sendto(bytes(packet), (self.outIP, self.outPort))
                    pkt_count += 1
                    if pkt_count % 100 == 1:
                        q_depth = getattr(self.pmout, "queue_depth_packets", 0)
                        logger.info(
                            f"[RTP_TX] dest={self.outIP}:{self.outPort} | payload_type={pt} | "
                            f"sequence={self.outSequence} | timestamp={self.outTimestamp} | "
                            f"ssrc={self.outSSRC} | queue_depth={q_depth} pkts"
                        )
                except (BlockingIOError, OSError) as err:
                    if getattr(err, "winerror", None) == 10035 or getattr(err, "errno", None) in (10035, 11, 35):
                        import select
                        try:
                            _, writable, _ = select.select([], [self.sout], [], 0.005)
                            if writable:
                                self.sout.sendto(bytes(packet), (self.outIP, self.outPort))
                                pkt_count += 1
                        except Exception:
                            pass
                    else:
                        logger.warning(f"[RTP_TX] sendto error: {err}")
                
                self.outSequence = (self.outSequence + 1) & 0xFFFF
                self.outTimestamp = (self.outTimestamp + len(payload)) & 0xFFFFFFFF
                
                next_time += frame_duration
                sleep_time = next_time - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                elif sleep_time < -0.100:
                    # Reset clock only if lag exceeds 100ms (5 frames) to avoid cumulative clock drift
                    next_time = time.perf_counter()
        
        pyVoIP.RTP.RTPClient.trans = _patched_trans'''

assert old_block in text, "old_block not found in sip_registrar.py"
text = text.replace(old_block, new_block, 1)

# 3. Update audio_clear in outbound and inbound audio bridge
old_ac1 = '''            def audio_clear():
                """Flush pyVoIP transmit buffer for instant barge-in."""
                try:
                    if hasattr(call, "RTPClients") and call.RTPClients:
                        for client in call.RTPClients:
                            if hasattr(client, "pmout") and client.pmout:
                                try:
                                    with client.pmout.bufferLock:
                                        import io
                                        client.pmout.buffer = io.BytesIO()
                                        client.pmout.log = {}
                                        client.pmout.offset = getattr(client, "outOffset", 0)
                                except Exception:
                                    pass
                    if hasattr(call, "rtp") and call.rtp:
                        if hasattr(call.rtp, "clear"):
                            call.rtp.clear()'''

new_ac1 = '''            def audio_clear():
                """Flush pyVoIP transmit buffer for instant barge-in."""
                try:
                    if hasattr(call, "RTPClients") and call.RTPClients:
                        for client in call.RTPClients:
                            if hasattr(client, "pmout") and client.pmout:
                                try:
                                    if hasattr(client.pmout, "clear"):
                                        client.pmout.clear()
                                    else:
                                        with getattr(client.pmout, "bufferLock", None):
                                            import io
                                            client.pmout.buffer = io.BytesIO()
                                            client.pmout.log = {}
                                except Exception:
                                    pass
                    if hasattr(call, "rtp") and call.rtp:
                        if hasattr(call.rtp, "clear"):
                            call.rtp.clear()'''

assert old_ac1 in text, "old_ac1 not found in sip_registrar.py"
text = text.replace(old_ac1, new_ac1, 1)

assert old_ac1 in text, "old_ac1 for inbound not found in sip_registrar.py"
text = text.replace(old_ac1, new_ac1, 1)

with open('src/services/sip_registrar.py', 'w', encoding='utf-8', newline='') as f:
    f.write(text)

print("SUCCESSFULLY PATCHED sip_registrar.py")


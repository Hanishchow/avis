"""
SAFETY NETS — Self-preservation mechanisms for the Biologic LLM.
Prevents disk filling, bogus learning, infinite loops, and runaway processes.
"""

import os
import shutil
import json
import time
import math
from collections import deque
from datetime import datetime

# ============================================================
# SAFETY NET 1: DISK SPACE MONITOR
# ============================================================
class DiskWatchdog:
    """Monitors disk usage and prevents filling the drive."""
    
    def __init__(self, warning_threshold_gb=5.0, critical_threshold_gb=1.0, 
                 max_log_size_mb=50, data_dir='.'):
        self.warning_threshold = warning_threshold_gb * 1024 * 1024 * 1024
        self.critical_threshold = critical_threshold_gb * 1024 * 1024 * 1024
        self.max_log_size = max_log_size_mb * 1024 * 1024
        self.data_dir = data_dir
        self.last_check_time = 0
        self.check_interval = 30  # seconds between checks
    
    def get_free_space(self):
        """Get available disk space in bytes."""
        try:
            stat = shutil.disk_usage(self.data_dir)
            return stat.free
        except:
            return float('inf')
    
    def check(self):
        """Check disk status. Returns dict with status info."""
        now = time.time()
        if now - self.last_check_time < self.check_interval:
            return {'status': 'ok', 'cached': True}
        
        self.last_check_time = now
        free = self.get_free_space()
        
        if free < self.critical_threshold:
            self._emergency_cleanup()
            return {'status': 'critical', 'free_gb': free / (1024**3)}
        elif free < self.warning_threshold:
            return {'status': 'warning', 'free_gb': free / (1024**3)}
        else:
            return {'status': 'ok', 'free_gb': free / (1024**3)}
    
    def _emergency_cleanup(self):
        """Emergency disk cleanup — remove oldest logs, trim files."""
        print(f"  [SAFETY] EMERGENCY: Disk critically low! Running cleanup...")
        
        # Trim self_assessment.json
        if os.path.exists('self_assessment.json'):
            try:
                with open('self_assessment.json', 'r') as f:
                    data = json.load(f)
                with open('self_assessment.json', 'w') as f:
                    json.dump(data[-10:], f)  # Keep last 10 entries
                print("  [SAFETY] Trimmed self_assessment.json to 10 entries")
            except:
                os.remove('self_assessment.json')
                print("  [SAFETY] Removed self_assessment.json")
        
        # Remove temp files
        for f in os.listdir('.'):
            if f.endswith('.tmp') or f.endswith('.log'):
                try:
                    os.remove(f)
                    print(f"  [SAFETY] Removed temp file: {f}")
                except:
                    pass
        
        print("  [SAFETY] Emergency cleanup complete.")

# ============================================================
# SAFETY NET 2: CONTENT QUALITY GATE
# ============================================================
class ContentQualityGate:
    """
    Checks if content is worth learning from.
    Filters out bogus, repetitive, low-entropy, or malicious content.
    """
    
    def __init__(self):
        self.min_entropy = 0.5  # bits per character
        self.min_length = 5
        self.max_length = 100000
        self.max_repetition_ratio = 0.8  # If 80%+ is repeats, it's bogus
    
    def entropy(self, text):
        """Compute Shannon entropy of text."""
        if not text:
            return 0
        text = text.lower()
        counts = {}
        for c in text:
            counts[c] = counts.get(c, 0) + 1
        total = len(text)
        entropy = 0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy
    
    def repetition_ratio(self, text):
        """Detect repetitive patterns (e.g. 'aaaaaaaa' or 'abcabcabc')."""
        if len(text) < 10:
            return 0
        
        # Check character-level repetition
        unique_chars = len(set(text))
        if unique_chars <= 2 and len(text) > 20:
            return 0.9  # Only 1-2 unique chars = clearly bogus
        
        # Check n-gram repetition (look for repeated substrings)
        max_rep = 0
        for n in [3, 4, 5]:
            seen = set()
            repeats = 0
            total = max(len(text) - n + 1, 1)
            for i in range(len(text) - n + 1):
                gram = text[i:i+n]
                if gram in seen:
                    repeats += 1
                seen.add(gram)
            ratio = repeats / total
            max_rep = max(max_rep, ratio)
        
        return max_rep
    
    def is_bogus(self, text):
        """Check if text is bogus and shouldn't be learned from."""
        if not text or len(text) < self.min_length:
            return True, "too short"
        
        if len(text) > self.max_length:
            return True, "too long"
        
        ent = self.entropy(text)
        if ent < self.min_entropy:
            return True, f"low entropy ({ent:.2f})"
        
        rep = self.repetition_ratio(text)
        if rep > self.max_repetition_ratio:
            return True, f"too repetitive ({rep:.2f})"
        
        # Check for binary/non-printable content
        printable = sum(1 for c in text if c.isprintable() or c in '\n\r\t')
        if printable / max(len(text), 1) < 0.95:
            return True, "non-printable content"
        
        # Check for control characters
        control = sum(1 for c in text if ord(c) < 32 and c not in '\n\r\t')
        if control > 0:
            return True, "contains control characters"
        
        return False, "ok"
    
    def assess(self, text):
        """Full quality assessment with details."""
        ent = self.entropy(text)
        rep = self.repetition_ratio(text)
        is_bogus, reason = self.is_bogus(text)
        
        return {
            'is_bogus': is_bogus,
            'reason': reason,
            'entropy': ent,
            'repetition': rep,
            'length': len(text)
        }

# ============================================================
# SAFETY NET 3: DUPLICATE PREVENTION
# ============================================================
class DuplicateDetector:
    """Prevents learning the same content repeatedly."""
    
    def __init__(self, max_history=500, similarity_threshold=0.85):
        self.history = deque(maxlen=max_history)
        self.threshold = similarity_threshold
    
    def text_similarity(self, a, b):
        """Simple character n-gram similarity."""
        if not a or not b:
            return 0
        a_grams = set(a[i:i+5] for i in range(len(a)-4))
        b_grams = set(b[i:i+5] for i in range(len(b)-4))
        if not a_grams or not b_grams:
            return 0
        intersection = a_grams & b_grams
        return len(intersection) / max(len(a_grams | b_grams), 1)
    
    def is_duplicate(self, text):
        """Check if similar content was already learned."""
        text_lower = text.lower().strip()
        for past in self.history:
            sim = self.text_similarity(text_lower, past)
            if sim > self.threshold:
                return True, sim
        return False, 0.0
    
    def record(self, text):
        """Record a new piece of learned content."""
        self.history.append(text.lower().strip()[:200])

# ============================================================
# SAFETY NET 4: RATE LIMITER
# ============================================================
class RateLimiter:
    """Prevents overwhelming the system with too much data too fast."""
    
    def __init__(self, max_chars_per_minute=10000, max_entries_per_minute=60):
        self.max_chars = max_chars_per_minute
        self.max_entries = max_entries_per_minute
        self.char_window = deque()
        self.entry_window = deque()
    
    def check(self, char_count=1):
        """Check if allowed to learn this much. Returns (allowed, wait_seconds)."""
        now = time.time()
        
        # Clean old entries
        while self.char_window and self.char_window[0][0] < now - 60:
            self.char_window.popleft()
        while self.entry_window and self.entry_window[0] < now - 60:
            self.entry_window.popleft()
        
        total_chars = sum(c for _, c in self.char_window)
        total_entries = len(self.entry_window)
        
        if total_chars + char_count > self.max_chars:
            return False, 60 - (now - self.char_window[0][0]) if self.char_window else 1
        
        if total_entries + 1 > self.max_entries:
            return False, 60 - (now - self.entry_window[0]) if self.entry_window else 1
        
        return True, 0
    
    def record(self, char_count):
        """Record that we learned something."""
        self.char_window.append((time.time(), char_count))
        self.entry_window.append(time.time())

# ============================================================
# SAFETY NET 5: EMERGENCY BRAKE
# ============================================================
class EmergencyBrake:
    """
    Global kill switch if critical thresholds are hit.
    Pauses learning, forces consolidation, or shuts down safely.
    """
    
    def __init__(self):
        self.engaged = False
        self.reason = ""
        self.engagement_time = None
        self.max_memory_mb = 500  # Max memory usage before brake
        self.max_hours_without_sleep = 24
    
    def engage(self, reason):
        """Engage the emergency brake."""
        self.engaged = True
        self.reason = reason
        self.engagement_time = datetime.now()
        print(f"\n  [SAFETY] EMERGENCY BRAKE ENGAGED: {reason}")
        print("  [SAFETY] Pausing learning. Type 'resume' to override.")
    
    def disengage(self):
        """Release the emergency brake."""
        if self.engaged:
            print(f"  [SAFETY] Brake released after {self.reason}")
            self.engaged = False
            self.reason = ""
            self.engagement_time = None
    
    def check_process_memory(self):
        """Rough memory check (process-level)."""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            mem_mb = process.memory_info().rss / 1024 / 1024
            if mem_mb > self.max_memory_mb:
                return True, mem_mb
            return False, mem_mb
        except ImportError:
            return False, 0  # psutil not available, skip check

# ============================================================
# SAFETY NET 6: LOG ROTATION
# ============================================================
class LogRotator:
    """Rotate and compress log files to prevent unbounded growth."""
    
    def __init__(self, max_entries=200, max_file_size_kb=100):
        self.max_entries = max_entries
        self.max_file_size = max_file_size_kb * 1024
    
    def trim_json_log(self, filepath, keep_last=50):
        """Keep only the most recent entries in a JSON log file."""
        if not os.path.exists(filepath):
            return
        
        try:
            size = os.path.getsize(filepath)
            if size < self.max_file_size:
                return
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, list) and len(data) > keep_last:
                trimmed = data[-keep_last:]
                with open(filepath, 'w') as f:
                    json.dump(trimmed, f, indent=2)
                print(f"  [SAFETY] Trimmed {filepath}: {len(data)} -> {len(trimmed)} entries")
        except:
            pass  # Don't crash on log rotation failure

# ============================================================
# COMPREHENSIVE SAFETY SYSTEM
# ============================================================
class SafetySystem:
    """
    Integrated safety system combining all safety nets.
    Call check() before any learning operation.
    """
    
    def __init__(self):
        self.disk = DiskWatchdog()
        self.quality = ContentQualityGate()
        self.duplicates = DuplicateDetector()
        self.rate_limiter = RateLimiter()
        self.brake = EmergencyBrake()
        self.logs = LogRotator()
        self.total_inputs_rejected = 0
        self.total_inputs_accepted = 0
    
    def pre_check(self, text, source="user"):
        """
        Run ALL safety checks before learning from text.
        Returns (allowed: bool, reason: str, details: dict)
        """
        # Check 1: Emergency brake
        if self.brake.engaged:
            return False, f"Emergency brake: {self.brake.reason}", {}
        
        # Check 2: Disk space
        disk_status = self.disk.check()
        if disk_status['status'] == 'critical':
            self.brake.engage("disk critically low")
            return False, "Disk critically low, learning paused", disk_status
        
        # Check 3: Content quality
        quality = self.quality.assess(text)
        if quality['is_bogus']:
            self.total_inputs_rejected += 1
            return False, f"Bogus content: {quality['reason']}", quality
        
        # Check 4: Duplicates
        is_dup, sim = self.duplicates.is_duplicate(text)
        if is_dup:
            return False, f"Duplicate content (similarity: {sim:.2f})", {'similarity': sim}
        
        # Check 5: Rate limit
        allowed, wait = self.rate_limiter.check(len(text))
        if not allowed:
            return False, f"Rate limited (wait {wait:.0f}s)", {'wait': wait}
        
        # All checks passed
        self.total_inputs_accepted += 1
        return True, "ok", quality
    
    def post_learn(self, text):
        """Call after successfully learning from text."""
        self.rate_limiter.record(len(text))
        self.duplicates.record(text)
        
        # Periodic maintenance
        if self.total_inputs_accepted % 20 == 0:
            self.logs.trim_json_log('self_assessment.json')
        
        # Occasional disk check
        if self.total_inputs_accepted % 30 == 0:
            self.disk.check()
        
        # Memory check
        if self.total_inputs_accepted % 50 == 0:
            over, mem_mb = self.brake.check_process_memory()
            if over:
                print(f"  [SAFETY] Warning: Process memory at {mem_mb:.0f} MB")

def demo_safety_nets():
    """Demonstrate all safety nets in action."""
    print("=" * 60)
    print("SAFETY NETS DEMONSTRATION")
    print("=" * 60)
    
    safety = SafetySystem()
    
    # Test 1: Good content
    print("\n[TEST 1] Good content:")
    good = "The mitochondria is the powerhouse of the cell. Biology studies living organisms."
    allowed, reason, details = safety.pre_check(good)
    print(f"  Input: \"{good[:50]}...\"")
    print(f"  Result: {'ALLOWED' if allowed else 'REJECTED'} | {reason}")
    
    # Test 2: Bogus content (repetitive)
    print("\n[TEST 2] Bogus content (repetitive):")
    bogus = "aaaaaaaabbbbbbbbaaaaaaaabbbbbbbbaaaaaaaabbbbbbbb"
    allowed, reason, details = safety.pre_check(bogus)
    print(f"  Input: \"{bogus[:40]}...\"")
    print(f"  Result: {'ALLOWED' if allowed else 'REJECTED'} | {reason}")
    if details:
        print(f"  Entropy: {details.get('entropy', 0):.2f}, Repetition: {details.get('repetition', 0):.2f}")
    
    # Test 3: Bogus content (single character spam)
    print("\n[TEST 3] Bogus content (keyboard spam):")
    spam = "asdfasdfasdfasdfasdfasdfasdfasdfasdfasdfasdfasdfasdf"
    allowed, reason, details = safety.pre_check(spam)
    print(f"  Input: \"{spam[:40]}...\"")
    print(f"  Result: {'ALLOWED' if allowed else 'REJECTED'} | {reason}")
    
    # Test 4: Duplicate detection
    print("\n[TEST 4] Duplicate detection:")
    first = "Python is a high-level programming language."
    safety.post_learn(first)
    second = "Python is a high-level programming language."  # Same
    allowed, reason, details = safety.pre_check(second)
    print(f"  First: \"{first}\" (learned)")
    print(f"  Second: \"{second}\"")
    print(f"  Result: {'ALLOWED' if allowed else 'REJECTED'} | {reason}")
    
    # Test 5: Near-duplicate
    print("\n[TEST 5] Near-duplicate (slightly different):")
    similar = "Python is a high level programming language for coding."
    allowed, reason, details = safety.pre_check(similar)
    print(f"  Input: \"{similar}\"")
    print(f"  Result: {'ALLOWED' if allowed else 'REJECTED'} | {reason}")
    
    # Test 6: Very short content
    print("\n[TEST 6] Very short content:")
    short = "hi"
    allowed, reason, details = safety.pre_check(short)
    print(f"  Input: \"{short}\"")
    print(f"  Result: {'ALLOWED' if allowed else 'REJECTED'} | {reason}")
    
    # Test 7: Non-printable content
    print("\n[TEST 7] Non-printable content:")
    binary = "Hello\x00World\x01\x02\x03Binary data here\xff"
    allowed, reason, details = safety.pre_check(binary)
    print(f"  Input: (contains binary bytes)")
    print(f"  Result: {'ALLOWED' if allowed else 'REJECTED'} | {reason}")
    
    # Test 8: Entropy spectrum
    print("\n[TEST 8] Entropy spectrum:")
    tests = [
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",  # Very low entropy
        "abcabcabcabcabcabcabcabcabcabc",    # Low entropy
        "Hello world, this is normal English text with varied content.",  # Normal
        "Ancient Greek: ho logos, he aletheia, to agathon.",  # Different charset
    ]
    for t in tests:
        ent = safety.quality.entropy(t)
        rep = safety.quality.repetition_ratio(t)
        print(f"  ent={ent:.2f} rep={rep:.2f} | \"{t[:50]}\"")
    
    # Summary
    print("\n" + "=" * 60)
    print("SAFETY NETS SUMMARY")
    print("=" * 60)
    print(f"  Total inputs accepted: {safety.total_inputs_accepted}")
    print(f"  Total inputs rejected: {safety.total_inputs_rejected}")
    print(f"  Disk watchdog: {'Active'}")
    print(f"  Content quality gate: Active (min entropy: {safety.quality.min_entropy})")
    print(f"  Duplicate detector: Active (threshold: {safety.duplicates.threshold})")
    print(f"  Rate limiter: Active (max {safety.rate_limiter.max_entries}/min)")
    print(f"  Emergency brake: Ready")
    print(f"  Log rotator: Active (max {safety.logs.max_file_size//1024} KB)")
    print()

if __name__ == "__main__":
    demo_safety_nets()
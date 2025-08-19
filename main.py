from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from kivy.core.window import Window
from kivy.graphics import Color, Rectangle
from kivy.clock import Clock
import json
import os
from collections import defaultdict, Counter, deque
import time
from datetime import datetime

# Set window size for mobile compatibility
Window.fullscreen = True

# Files
MEMORY_FILE = "pattern_memory.json"
METRICS_FILE = "prediction_metrics.json"
EXPLOIT_FILE = "exploit_models.json"
SESSION_GAP_MINUTES = 5

class ExploitModel:
    def __init__(self, pattern, predicted, exploit_type):
        self.pattern = pattern
        self.predicted = predicted
        self.exploit_type = exploit_type
        self.confirmed = 0
        self.failed = 0
        self.last_used = time.time()
        self.birth_time = time.time()
        self.last_revival_attempt = 0
        self.cooldown_attempts = 0
        self.weight = 1.0  # Initial weight
        self.initial_success = 0
        self.in_cooldown = False

    @property
    def confidence(self):
        total = self.confirmed + self.failed
        return self.confirmed / total if total > 0 else 0

    @property
    def usage(self):
        return self.confirmed + self.failed

    def record_result(self, correct):
        if correct:
            self.confirmed += 1
            # Track initial success streak
            if self.initial_success >= 0:
                self.initial_success += 1
            # Revival boost
            if self.in_cooldown:
                self.weight = min(5.0, self.weight * 1.5)
                self.cooldown_attempts = 0
                self.in_cooldown = False
            else:
                self.weight = min(5.0, self.weight + 0.1)
        else:
            self.failed += 1
            # Track initial failures
            if self.initial_success > 0:
                self.initial_success = -1  # Mark as broken streak
            # Enter cooldown after multiple failures
            self.cooldown_attempts += 1
            if self.cooldown_attempts >= 3:
                self.in_cooldown = True
                self.weight = max(0.1, self.weight * 0.7)
            else:
                self.weight = max(0.1, self.weight * 0.85)
        self.last_used = time.time()
        self.last_revival_attempt = time.time() if correct else self.last_revival_attempt

    def get_score(self, recent_usage):
        """Calculate dynamic score with recent performance boost"""
        recency = 1.0 - min(1.0, (time.time() - self.last_used) / 3600)
        base_score = self.confidence * self.weight * recency
        
        # Boost if pattern appears in recent moves
        if any(self.pattern == tuple(recent_usage[i:i+len(self.pattern)]) 
               for i in range(len(recent_usage) - len(self.pattern) + 1)):
            return min(1.0, base_score * 1.3)
        return base_score

    def __repr__(self):
        return (f"{self.exploit_type}:{''.join(self.pattern)}->{self.predicted} "
                f"(✓:{self.confirmed} ✗:{self.failed} w:{self.weight:.2f} "
                f"cd:{self.cooldown_attempts})")

class ShapeRecognizer:
    def __init__(self):
        self.shape_history = defaultdict(list)
        self.recent_shapes = deque(maxlen=100)  # Stores tuples: (shape, next_letter, success)
        self.method_attempts = 0
        self.method_success = 0

    def convert_to_shape(self, sequence):
        """Convert letter sequence to shape pattern (e.g., AAB -> XXY)"""
        mapping = {}
        next_char = 'X'
        shape = []
        for letter in sequence:
            if letter not in mapping:
                mapping[letter] = next_char
                next_char = chr(ord(next_char) + 1)
            shape.append(mapping[letter])
        return ''.join(shape)

    def record_shape_outcome(self, sequence, next_letter, correct):
        """Record outcome for a shape pattern"""
        if len(sequence) < 5:
            return
        shape_pattern = self.convert_to_shape(sequence[-5:])
        self.shape_history[shape_pattern].append((next_letter, correct))
        self.recent_shapes.append((shape_pattern, next_letter, correct))
        
        # Update method accuracy
        self.method_attempts += 1
        if correct:
            self.method_success += 1

    def get_shape_prediction(self, sequence):
        """Get prediction based on shape patterns"""
        if len(sequence) < 5:
            return None, 0.0
        shape_pattern = self.convert_to_shape(sequence[-5:])
        outcomes = self.shape_history.get(shape_pattern, [])
        if not outcomes:
            return None, 0.0
            
        # Calculate success rate
        total = len(outcomes)
        successes = sum(1 for _, correct in outcomes if correct)
        success_rate = successes / total
        
        # Only use if we have enough data and good success rate
        if total >= 3 and success_rate >= 0.7:
            # Find most common outcome
            outcome_counter = Counter(outcome for outcome, _ in outcomes)
            prediction, count = outcome_counter.most_common(1)[0]
            confidence = count / total
            return prediction, confidence
        return None, 0.0

    def get_shape_accuracy(self):
        """Calculate accuracy for shape-based predictions"""
        if self.method_attempts == 0:
            return 0.0
        return round((self.method_success / self.method_attempts) * 100, 1)

class AdaptivePredictor:
    def __init__(self):
        self.min_pattern = 2
        self.max_pattern = 4
        self.exploit_models = []
        self.exploration_rate = 0.3  # Default exploration
        self.correct = 0
        self.total = 0
        self.streak = 0
        self.best_streak = 0
        self.volatility = 0.5  # Default volatility (0=stable, 1=chaotic)
        self.trusted_types = {"pattern": 1.0, "streak": 1.0, "break": 0.8}  # Default trust weights
        self.short_term_memory = []  # Last 40 moves
        self.mid_term_memory = []  # Last 200 moves
        self.method_success = {"pattern": 0, "streak": 0, "frequency": 0, "hybrid": 0}
        self.method_attempts = {"pattern": 0, "streak": 0, "frequency": 0, "hybrid": 0}
        self.recent_accuracy = deque(maxlen=40)  # Track last 40 outcomes
        self.consecutive_failures = 0
        self.drift_recovery_active = False
        self.drift_recovery_start = 0
        self.shape_recognizer = ShapeRecognizer()
        
        # New losing streak attributes
        self.current_losing_streak = 0
        self.max_losing_streak = 0
        
        self.load_metrics()
        self.load_models()

    def load_metrics(self):
        if os.path.exists(METRICS_FILE):
            try:
                with open(METRICS_FILE, 'r') as f:
                    data = json.load(f)
                self.correct = data.get('correct', 0)
                self.total = data.get('total', 0)
                self.streak = data.get('current_streak', 0)
                self.best_streak = data.get('best_streak', 0)
                self.volatility = data.get('volatility', 0.5)
                self.trusted_types = data.get('trusted_types', {"pattern": 1.0, "streak": 1.0, "break": 0.8})
                self.method_success = data.get('method_success', {"pattern": 0, "streak": 0, "frequency": 0, "hybrid": 0})
                self.method_attempts = data.get('method_attempts', {"pattern": 0, "streak": 0, "frequency": 0, "hybrid": 0})
                self.consecutive_failures = data.get('consecutive_failures', 0)
                self.drift_recovery_active = data.get('drift_recovery_active', False)
                self.drift_recovery_start = data.get('drift_recovery_start', 0)
                
                # Load losing streak metrics
                self.current_losing_streak = data.get('current_losing_streak', 0)
                self.max_losing_streak = data.get('max_losing_streak', 0)
            except (json.JSONDecodeError, KeyError):
                self.correct = 0
                self.total = 0
                self.streak = 0
                self.best_streak = 0
                self.volatility = 0.5
                self.trusted_types = {"pattern": 1.0, "streak": 1.0, "break": 0.8}
                self.method_success = {"pattern": 0, "streak": 0, "frequency": 0, "hybrid": 0}
                self.method_attempts = {"pattern": 0, "streak": 0, "frequency": 0, "hybrid": 0}
                self.consecutive_failures = 0
                self.drift_recovery_active = False
                self.drift_recovery_start = 0
                
                # Initialize losing streaks
                self.current_losing_streak = 0
                self.max_losing_streak = 0

    def save_metrics(self):
        try:
            with open(METRICS_FILE, 'w') as f:
                json.dump({
                    'correct': self.correct,
                    'total': self.total,
                    'current_streak': self.streak,
                    'best_streak': self.best_streak,
                    'volatility': self.volatility,
                    'trusted_types': self.trusted_types,
                    'method_success': self.method_success,
                    'method_attempts': self.method_attempts,
                    'consecutive_failures': self.consecutive_failures,
                    'drift_recovery_active': self.drift_recovery_active,
                    'drift_recovery_start': self.drift_recovery_start,
                    
                    # Save losing streak metrics
                    'current_losing_streak': self.current_losing_streak,
                    'max_losing_streak': self.max_losing_streak
                }, f)
        except Exception:
            pass

    def load_models(self):
        if os.path.exists(EXPLOIT_FILE):
            try:
                with open(EXPLOIT_FILE, 'r') as f:
                    models = json.load(f)
                for model in models:
                    exploit = ExploitModel(
                        tuple(model['pattern']),
                        model['predicted'],
                        model['type']
                    )
                    exploit.confirmed = model['confirmed']
                    exploit.failed = model['failed']
                    exploit.weight = model['weight']
                    exploit.last_used = model['last_used']
                    exploit.birth_time = model.get('birth_time', time.time())
                    exploit.last_revival_attempt = model.get('last_revival_attempt', 0)
                    exploit.cooldown_attempts = model.get('cooldown_attempts', 0)
                    exploit.initial_success = model.get('initial_success', 0)
                    exploit.in_cooldown = model.get('in_cooldown', False)
                    self.exploit_models.append(exploit)
            except (json.JSONDecodeError, KeyError):
                pass

    def save_models(self):
        try:
            models = []
            for exploit in self.exploit_models:
                models.append({
                    'pattern': list(exploit.pattern),
                    'predicted': exploit.predicted,
                    'type': exploit.exploit_type,
                    'confirmed': exploit.confirmed,
                    'failed': exploit.failed,
                    'weight': exploit.weight,
                    'last_used': exploit.last_used,
                    'birth_time': exploit.birth_time,
                    'last_revival_attempt': exploit.last_revival_attempt,
                    'cooldown_attempts': exploit.cooldown_attempts,
                    'initial_success': exploit.initial_success,
                    'in_cooldown': exploit.in_cooldown
                })
            with open(EXPLOIT_FILE, 'w') as f:
                json.dump(models, f)
        except Exception:
            pass

    def record_accuracy(self, guesses, actual, method_used):
        self.total += 1
        self.method_attempts[method_used] += 1
        correct = actual in guesses
        self.recent_accuracy.append(1 if correct else 0)
        
        # Update winning streak and accuracy
        if correct:
            self.correct += 1
            self.streak += 1
            self.method_success[method_used] += 1
            if self.streak > self.best_streak:
                self.best_streak = self.streak
            self.consecutive_failures = 0
            
            # Reset losing streak on correct prediction
            self.current_losing_streak = 0
        else:
            self.streak = 0
            self.consecutive_failures += 1
            
            # Update losing streak
            self.current_losing_streak += 1
            if self.current_losing_streak > self.max_losing_streak:
                self.max_losing_streak = self.current_losing_streak

        # Reduce trust in failing method
        if self.method_attempts[method_used] > 5:
            success_rate = self.method_success[method_used] / self.method_attempts[method_used]
            if success_rate < 0.4:
                for exploit in self.exploit_models:
                    if exploit.exploit_type == method_used:
                        exploit.weight = max(0.1, exploit.weight * 0.8)
        
        # Record shape outcome
        if len(self.short_term_memory) >= 5:
            self.shape_recognizer.record_shape_outcome(
                self.short_term_memory, actual, correct
            )
        
        # Check for pattern drift
        self.check_drift()
        self.save_metrics()

    def check_drift(self):
        """Check if we need to trigger drift recovery"""
        if len(self.recent_accuracy) < 10:  # Not enough data
            return
        recent_accuracy = sum(self.recent_accuracy) / len(self.recent_accuracy)
        
        # Conditions for drift recovery
        if (recent_accuracy < 0.6 or self.consecutive_failures >= 3) and not self.drift_recovery_active:
            self.trigger_drift_recovery()

    def trigger_drift_recovery(self):
        """Reset models for pattern drift recovery"""
        self.drift_recovery_active = True
        self.drift_recovery_start = time.time()
        
        # Boost exploration
        self.exploration_rate = 0.5
        
        # Reduce trust weights
        for key in self.trusted_types:
            self.trusted_types[key] = max(0.3, self.trusted_types[key] * 0.7)
        
        # Reset model weights and cooldowns
        for model in self.exploit_models:
            model.weight = 1.0
            model.cooldown_attempts = 0
            model.in_cooldown = False
        
        # Reset failure tracking
        self.consecutive_failures = 0
        self.save_metrics()
        self.save_models()

    def check_drift_recovery_end(self):
        """End drift recovery after 100 moves"""
        if self.drift_recovery_active and len(self.recent_accuracy) >= 100:
            self.drift_recovery_active = False
            self.exploration_rate = 0.3
            
            # Restore trust weights to original
            original_weights = {"pattern": 1.0, "streak": 1.0, "break": 0.8}
            for key in original_weights:
                if key in self.trusted_types:
                    self.trusted_types[key] = original_weights[key]

    def get_accuracy(self):
        return round((self.correct / self.total) * 100, 2) if self.total > 0 else 0

    def get_method_accuracy(self, method):
        """Get accuracy for a specific prediction method"""
        if self.method_attempts[method] == 0:
            return 0.0
        return round((self.method_success[method] / self.method_attempts[method]) * 100, 1)

    def discover_exploits(self, recent_bets, actual):
        """Generate new potential exploit models"""
        # Pattern-based exploits
        for pattern_len in range(self.min_pattern, self.max_pattern + 1):
            if len(recent_bets) < pattern_len + 1:
                continue
            pattern = tuple(recent_bets[-(pattern_len+1):-1])
            self.create_or_update_exploit(pattern, actual, "pattern")
        
        # Streak-based exploits
        if len(recent_bets) > 2:
            last_letter = recent_bets[-1]
            streak_length = 1
            for i in range(len(recent_bets)-2, -1, -1):
                if recent_bets[i] == last_letter:
                    streak_length += 1
                else:
                    break
            
            if streak_length >= 2:
                pattern = tuple(recent_bets[-streak_length:])
                self.create_or_update_exploit(pattern, last_letter, "streak")
                
                # Break prediction model (using least frequent)
                options = ['A', 'B', 'C']
                options.remove(last_letter)
                freq = Counter(self.short_term_memory)
                break_pred = min(options, key=lambda x: freq.get(x, 0))
                self.create_or_update_exploit(pattern, break_pred, "break")
        
        # Prune low-performing models
        self.prune_models()

    def create_or_update_exploit(self, pattern, outcome, exploit_type):
        for exploit in self.exploit_models:
            if exploit.pattern == pattern and exploit.predicted == outcome:
                return exploit
        new_exploit = ExploitModel(pattern, outcome, exploit_type)
        self.exploit_models.append(new_exploit)
        return new_exploit

    def prune_models(self):
        """Remove low-performing or stale models"""
        current_time = time.time()
        new_models = []
        for exploit in self.exploit_models:
            time_since_last_use = current_time - exploit.last_used
            time_since_birth = current_time - exploit.birth_time
            
            # Keep models that are still effective
            if exploit.confidence > 0.65 and exploit.usage > 5:
                new_models.append(exploit)
            # Keep recently used models
            elif time_since_last_use < 86400:  # 1 day
                new_models.append(exploit)
            # Keep models with strong initial success
            elif exploit.initial_success >= 5 and time_since_last_use < 259200:  # 3 days
                new_models.append(exploit)
            # Revive old models that were once good
            elif exploit.initial_success >= 10 and time_since_last_use < 604800:  # 1 week
                if exploit.last_revival_attempt < (current_time - 43200):  # 12 hours
                    exploit.cooldown_attempts = 0
                    exploit.in_cooldown = False
                    new_models.append(exploit)
        
        self.exploit_models = new_models
        self.save_models()

    def detect_meta(self, warmup_bets):
        """Detect session meta from first 30 moves"""
        if len(warmup_bets) < 30:
            return
        
        # Calculate volatility (changes per move)
        changes = 0
        for i in range(1, len(warmup_bets)):
            if warmup_bets[i] != warmup_bets[i-1]:
                changes += 1
        self.volatility = changes / (len(warmup_bets) - 1) if len(warmup_bets) > 1 else 0.0
        
        # Detect streak patterns
        streak_count = 0
        current_streak = 1
        last_char = warmup_bets[0]
        for char in warmup_bets[1:]:
            if char == last_char:
                current_streak += 1
            else:
                if current_streak >= 3:
                    streak_count += 1
                current_streak = 1
                last_char = char
        
        # Adjust trust based on findings
        if self.volatility > 0.75:  # High volatility
            self.trusted_types = {"pattern": 0.7, "streak": 0.4, "break": 0.9}
            self.exploration_rate = 0.4
        elif streak_count > 5:  # Many streaks
            self.trusted_types = {"pattern": 0.8, "streak": 1.0, "break": 0.6}
            self.exploration_rate = 0.25
        else:  # Stable patterns
            self.trusted_types = {"pattern": 1.0, "streak": 0.8, "break": 0.5}
            self.exploration_rate = 0.2
        
        self.save_metrics()

    def update_memory(self, new_bet):
        """Update short and mid-term memory with drift detection"""
        # Update memories
        self.short_term_memory = (self.short_term_memory + [new_bet])[-40:]
        self.mid_term_memory = (self.mid_term_memory + [new_bet])[-200:]
        
        # Check if drift recovery should end
        self.check_drift_recovery_end()
        
        # Detect drift if we have enough data
        if len(self.short_term_memory) > 30 and len(self.mid_term_memory) > 150:
            # Compare pattern frequencies
            short_counts = Counter(self.short_term_memory)
            mid_counts = Counter(self.mid_term_memory)
            
            # Get top 2 patterns in each
            short_top = [item[0] for item in short_counts.most_common(2)]
            mid_top = [item[0] for item in mid_counts.most_common(2)]
            
            # Calculate similarity
            similarity = len(set(short_top) & set(mid_top)) / 2.0
            
            # Apply drift response
            if similarity < 0.5:  # Significant drift
                # Fade model weights
                for model in self.exploit_models:
                    model.weight = max(0.3, model.weight * 0.95)

    def hybrid_voting(self, recent_bets):
        """Combine predictions from multiple methods with weighted confidence"""
        votes = defaultdict(float)
        source_details = []
        method_used = "hybrid"
        
        # 1. Exploit models (recent-focused)
        exploit_votes = defaultdict(float)
        recent_20 = recent_bets[-20:] if len(recent_bets) >= 20 else recent_bets
        for exploit in self.exploit_models:
            if len(recent_bets) >= len(exploit.pattern):
                if tuple(recent_bets[-len(exploit.pattern):]) == exploit.pattern:
                    # Apply type trust factor
                    trust = self.trusted_types.get(exploit.exploit_type, 0.7)
                    score = exploit.get_score(recent_20) * trust
                    exploit_votes[exploit.predicted] += score
        
        if exploit_votes:
            source_details.append("Exploit models")
        for letter, score in exploit_votes.items():
            votes[letter] += score
        
        # 2. Shape pattern recognition (high priority)
        shape_pred, shape_conf = self.shape_recognizer.get_shape_prediction(recent_bets)
        if shape_pred:
            source_details.append(f"Shape: {shape_pred} ({shape_conf:.0%})")
            method_used = "shape"
            votes[shape_pred] += shape_conf * 1.5  # Higher weight for shape predictions
        
        # 3. Recent shape voting
        if len(recent_bets) >= 5:
            current_shape = self.shape_recognizer.convert_to_shape(recent_bets[-5:])
            shape_outcomes = []
            # Get outcomes for similar shapes in recent history
            for shape, next_letter, success in self.shape_recognizer.recent_shapes:
                if shape == current_shape:
                    shape_outcomes.append(next_letter)
            
            if shape_outcomes:
                counter = Counter(shape_outcomes)
                most_common = counter.most_common(1)[0]
                votes[most_common[0]] += most_common[1] * 0.1
                source_details.append(f"Shape vote: {most_common[0]}")
        
        # 4. Streak detection
        if len(recent_bets) > 3:
            last_char = recent_bets[-1]
            streak_length = 1
            for i in range(len(recent_bets)-2, -1, -1):
                if recent_bets[i] == last_char:
                    streak_length += 1
                else:
                    break
            
            if streak_length >= 3:
                # Streak continuation
                votes[last_char] += streak_length * 0.3
                source_details.append(f"Streak:{last_char}x{streak_length}")
                method_used = "streak"
                
                # Break prediction (using least frequent)
                options = ['A', 'B', 'C']
                options.remove(last_char)
                freq = Counter(self.short_term_memory)
                break_pred = min(options, key=lambda x: freq.get(x, 0))
                votes[break_pred] += 0.2
                source_details.append(f"Break:{break_pred}")
        
        # 5. Frequency fallback (no random)
        freq_counts = Counter(recent_20)
        if freq_counts:
            # Use least frequent option as fallback
            least_common = min(freq_counts, key=freq_counts.get)
            votes[least_common] += 0.1
            source_details.append(f"Fallback:{least_common}")
        
        # 6. Pattern boost for repeating sequences
        for pattern_len in range(2, 5):
            if len(recent_20) < pattern_len + 1:
                continue
            
            # Find repeating patterns
            pattern_counts = Counter(
                tuple(recent_20[i:i+pattern_len]) 
                for i in range(len(recent_20) - pattern_len)
            )
            
            # Boost confidence for repeating patterns
            for pattern, count in pattern_counts.items():
                if count > 1:
                    next_index = len(recent_20) - pattern_len
                    if next_index + pattern_len < len(recent_20):
                        actual_next = recent_20[next_index + pattern_len]
                        votes[actual_next] += count * 0.2
                        source_details.append(f"Pat:{''.join(pattern)}→{actual_next} x{count}")
                        method_used = "pattern"
        
        # Get top 2 predictions
        sorted_letters = sorted(votes.items(), key=lambda x: x[1], reverse=True)
        top_letters = [letter for letter, score in sorted_letters[:2]]
        
        # Confidence from top model
        confidence = sorted_letters[0][1] if sorted_letters else 0.5
        confidence = min(0.95, max(0.3, confidence))
        
        # Fill with least used if needed
        if len(top_letters) < 2:
            options = ['A', 'B', 'C']
            for letter in top_letters:
                if letter in options:
                    options.remove(letter)
            if options:
                freq = Counter(self.short_term_memory)
                least_used = min(options, key=lambda x: freq.get(x, 0))
                top_letters.append(least_used)
        
        # Add drift recovery indicator if active
        if self.drift_recovery_active:
            source_details.append("DRIFT RECOVERY ACTIVE")
        
        return {
            'predictions': top_letters[:2],
            'confidence': confidence,
            'details': " | ".join(source_details),
            'method': method_used
        }

    def predict_next(self, recent_bets):
        """Predict next move using hybrid voting system"""
        if not recent_bets:  # Default predictions at start
            return {
                'predictions': ['A', 'B'],
                'confidence': 0.5,
                'details': "Initial state",
                'method': "frequency"
            }
        return self.hybrid_voting(recent_bets)

    def update_model_performance(self, recent_bets, actual, method_used):
        """Update models based on prediction accuracy"""
        # Update memory first
        self.update_memory(actual)
        
        # Update models
        for model in self.exploit_models:
            if len(recent_bets) >= len(model.pattern):
                if tuple(recent_bets[-len(model.pattern):]) == model.pattern:
                    correct = model.predicted == actual
                    model.record_result(correct)
        
        # Discover new potential exploits
        self.discover_exploits(recent_bets, actual)
        self.save_models()
        
        # Detect meta in first 30 moves of session
        if len(recent_bets) == 30:
            self.detect_meta(recent_bets)

class EnhancedMemory:
    def __init__(self):
        self.sessions = []
        self.current_session = []
        self.session_counter = 0
        self.load()

    def load(self):
        if os.path.exists(MEMORY_FILE):
            try:
                with open(MEMORY_FILE, 'r') as f:
                    data = json.load(f)
                self.sessions = data.get('sessions', [])
                self.current_session = data.get('current_session', [])
                self.session_counter = data.get('session_counter', len(self.sessions))
            except (json.JSONDecodeError, KeyError):
                self.sessions = []
                self.current_session = []
                self.session_counter = 0

    def save(self):
        try:
            with open(MEMORY_FILE, 'w') as f:
                json.dump({
                    'sessions': self.sessions,
                    'current_session': self.current_session,
                    'session_counter': self.session_counter
                }, f)
        except Exception:
            pass

    def add(self, letter):
        timestamp = time.time()
        
        # Start new session if gap is too long
        if self.current_session and timestamp - self.current_session[-1]['timestamp'] > SESSION_GAP_MINUTES * 60:
            self.end_session()
        
        self.current_session.append({
            'letter': letter,
            'timestamp': timestamp
        })
        self.save()

    def undo_last(self):
        if self.current_session:
            self.current_session.pop()
            self.save()
            return True
        return False

    def end_session(self):
        if self.current_session:
            session_data = {
                'id': self.session_counter,
                'bets': [item['letter'] for item in self.current_session],
                'start_time': self.current_session[0]['timestamp'],
                'end_time': self.current_session[-1]['timestamp']
            }
            self.sessions.append(session_data)
            self.session_counter += 1
            self.current_session = []
            self.save()

    def clear_all(self):
        self.sessions = []
        self.current_session = []
        self.session_counter = 0
        try:
            if os.path.exists(MEMORY_FILE):
                os.remove(MEMORY_FILE)
            if os.path.exists(EXPLOIT_FILE):
                os.remove(EXPLOIT_FILE)
            if os.path.exists(METRICS_FILE):
                os.remove(METRICS_FILE)
        except Exception:
            pass

    def get_current_session_bets(self):
        return [item['letter'] for item in self.current_session]

class AdaptiveBettingUI(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', spacing=10, padding=10)
        
        # Initialize prediction tracking
        self.last_prediction = None
        self.last_method = None
        
        # Set background color
        with self.canvas.before:
            Color(0.1, 0.1, 0.1, 1)
            self.bg = Rectangle(size=self.size, pos=self.pos)
        self.bind(size=self._update_bg, pos=self._update_bg)
        
        # Initialize memory and predictor
        self.memory = EnhancedMemory()
        self.predictor = AdaptivePredictor()
        
        # UI Elements
        # Header
        header = BoxLayout(size_hint=(1, None), height=50)
        title = Label(text='ADAPTIVE EXPLOIT PREDICTOR', font_size=24, bold=True, color=(0.2, 0.6, 1, 1))
        header.add_widget(title)
        
        # Undo button
        self.undo_btn = Button(
            text='UNDO', 
            size_hint=(0.3, 1),
            background_color=(0.8, 0.5, 0, 1),
            disabled=not bool(self.memory.current_session)
        )
        self.undo_btn.bind(on_release=self.undo_last_bet)
        header.add_widget(self.undo_btn)
        
        # Debug button
        debug_btn = Button(
            text='DEBUG', 
            size_hint=(0.3, 1),
            background_color=(0.4, 0.2, 0.6, 1)
        )
        debug_btn.bind(on_release=self.show_debug_info)
        header.add_widget(debug_btn)
        self.add_widget(header)
        
        # Session info
        self.session_info = Label(text=self.get_session_text(), font_size=18, color=(0.7, 0.7, 0.7, 1))
        self.add_widget(self.session_info)
        
        # Recent bets display
        self.recent_label = Label(text='Recent: None', font_size=20, color=(0.8, 0.8, 0.8, 1))
        self.add_widget(self.recent_label)
        
        # Prediction display
        self.prediction_label = Label(
            text='Make your first bet...', 
            font_size=28, 
            bold=True,
            color=(0.2, 0.8, 0.2, 1)
        )
        self.add_widget(self.prediction_label)
        
        # Reason/confidence display
        self.reason_label = Label(text='', font_size=16, color=(0.7, 0.7, 0.7, 1))
        self.add_widget(self.reason_label)
        
        # Stats
        stats = BoxLayout(size_hint=(1, None), height=40)
        self.accuracy_label = Label(text=f"Accuracy: {self.predictor.get_accuracy()}%", color=(1, 0.8, 0.8, 1))
        
        # Updated streak display with losing streak
        self.streak_label = Label(text="", color=(0.8, 0.8, 1, 1))
        
        stats.add_widget(self.accuracy_label)
        stats.add_widget(self.streak_label)
        self.add_widget(stats)
        
        # Meta stats
        meta_stats = BoxLayout(size_hint=(1, None), height=30)
        self.volatility_label = Label(text=f"Volatility: {self.predictor.volatility:.2f}", color=(0.8, 1, 0.8, 1))
        self.explore_label = Label(text=f"Explore: {self.predictor.exploration_rate:.2f}", color=(1, 0.8, 1, 1))
        meta_stats.add_widget(self.volatility_label)
        meta_stats.add_widget(self.explore_label)
        self.add_widget(meta_stats)
        
        # Bet buttons
        btn_layout = BoxLayout(spacing=20, padding=10)
        self.buttons = {}
        for letter in ['A', 'B', 'C']:
            btn = Button(
                text=letter,
                font_size=30,
                background_color=(0.2, 0.3, 0.6, 1)
            )
            btn.bind(on_release=self.on_choice)
            btn_layout.add_widget(btn)
            self.buttons[letter] = btn
        self.add_widget(btn_layout)
        
        # Action buttons
        action_layout = BoxLayout(size_hint=(1, None), height=50, spacing=10)
        session_btn = Button(text='Session Manager', background_color=(0.4, 0.2, 0.6, 1))
        session_btn.bind(on_release=self.show_session_manager)
        action_layout.add_widget(session_btn)
        
        end_btn = Button(text='End Session', background_color=(0.8, 0.2, 0.2, 1))
        end_btn.bind(on_release=self.end_current_session)
        action_layout.add_widget(end_btn)
        
        reset_btn = Button(text='Reset Data', background_color=(0.8, 0.1, 0.1, 1))
        reset_btn.bind(on_release=self.reset_all_data)
        action_layout.add_widget(reset_btn)
        self.add_widget(action_layout)
        
        # Update UI
        self.update_ui()
    
    def _update_bg(self, *args):
        self.bg.size = self.size
        self.bg.pos = self.pos
        
    def get_session_text(self):
        session_count = len(self.memory.sessions) + (1 if self.memory.current_session else 0)
        current_bets = len(self.memory.current_session)
        status = "Active" if current_bets > 0 else "New"
        return f"Session: {session_count} | {status} | Bets: {current_bets}"
    
    def on_choice(self, instance):
        choice = instance.text
        current_bets = self.memory.get_current_session_bets()
        
        # Record accuracy only if we have a previous prediction
        if self.last_prediction is not None and self.last_method is not None:
            self.predictor.record_accuracy(self.last_prediction, choice, self.last_method)
            
        # Update model performance
        self.predictor.update_model_performance(current_bets, choice, self.last_method if self.last_method is not None else "initial")
        
        # Add to memory
        self.memory.add(choice)
        
        # Update UI
        self.update_ui(choice)
        
        # Visual feedback
        self.animate_button(instance)
        
        # Enable undo button
        self.undo_btn.disabled = False
        
    def undo_last_bet(self, instance):
        if self.memory.undo_last():
            self.update_ui()
            if not self.memory.current_session:
                self.undo_btn.disabled = True
                
    def animate_button(self, button):
        original_color = button.background_color
        button.background_color = (1, 0.7, 0, 1)
        Clock.schedule_once(lambda dt: setattr(button, 'background_color', original_color), 0.2)
    
    def update_ui(self, choice=None):
        # Update session info
        self.session_info.text = self.get_session_text()
        
        # Get current session bets
        current_bets = self.memory.get_current_session_bets()
        
        # Update recent bets display
        recent = current_bets[-10:] if current_bets else []
        self.recent_label.text = f"Recent: {' '.join(recent)}"
        
        # Make prediction
        if current_bets:
            prediction = self.predictor.predict_next(current_bets)
            self.last_prediction = prediction['predictions']
            self.last_method = prediction['method']
            
            # Update prediction display
            if len(self.last_prediction) == 2:
                self.prediction_label.text = f"Next: {self.last_prediction[0]} or {self.last_prediction[1]}"
            else:
                self.prediction_label.text = f"Next: {self.last_prediction[0]}"
            
            # Update reason
            confidence = int(prediction['confidence'] * 100)
            details = prediction.get('details', '')
            self.reason_label.text = f"Confidence: {confidence}% | {details}"
            
            # Highlight predicted buttons
            for letter, btn in self.buttons.items():
                if letter in self.last_prediction:
                    btn.background_color = (0.1, 0.8, 0.1, 1)
                else:
                    btn.background_color = (0.2, 0.3, 0.6, 1)
        else:
            self.prediction_label.text = "Make your first bet..."
            self.reason_label.text = ""
            for btn in self.buttons.values():
                btn.background_color = (0.2, 0.3, 0.6, 1)
            # Reset prediction tracking
            self.last_prediction = None
            self.last_method = None
        
        # Update stats
        self.accuracy_label.text = f"Accuracy: {self.predictor.get_accuracy()}%"
        
        # Updated streak display with losing streak
        self.streak_label.text = (
            f"Win: {self.predictor.streak} | "
            f"Lose: {self.predictor.current_losing_streak} "
            f"(Max: {self.predictor.max_losing_streak})"
        )
        
        self.volatility_label.text = f"Volatility: {self.predictor.volatility:.2f}"
        self.explore_label.text = f"Explore: {self.predictor.exploration_rate:.2f}"
    
    def show_debug_info(self, instance):
        """Show debug popup with method accuracies"""
        popup = Popup(title='Debug Information', size_hint=(0.9, 0.6))
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        # Method accuracies
        debug_label = Label(
            text="Method Accuracies:\n\n" +
            f"Pattern: {self.predictor.get_method_accuracy('pattern')}%\n" +
            f"Streak: {self.predictor.get_method_accuracy('streak')}%\n" +
            f"Frequency: {self.predictor.get_method_accuracy('frequency')}%\n" +
            f"Hybrid: {self.predictor.get_method_accuracy('hybrid')}%\n" +
            f"Shape: {self.predictor.shape_recognizer.get_shape_accuracy()}%\n\n" +
            f"Drift Recovery: {'Active' if self.predictor.drift_recovery_active else 'Inactive'}\n" +
            f"Consecutive Failures: {self.predictor.consecutive_failures}",
            font_size=18,
            halign='left',
            valign='top'
        )
        layout.add_widget(debug_label)
        
        close_btn = Button(text='Close', size_hint_y=None, height=40)
        close_btn.bind(on_release=popup.dismiss)
        layout.add_widget(close_btn)
        
        popup.content = layout
        popup.open()
    
    def show_session_manager(self, instance):
        popup = Popup(title='Session History', size_hint=(0.9, 0.8))
        layout = BoxLayout(orientation='vertical')
        scroll = ScrollView()
        grid = GridLayout(cols=1, spacing=10, size_hint_y=None)
        grid.bind(minimum_height=grid.setter('height'))
        
        for session in self.memory.sessions:
            start = datetime.fromtimestamp(session['start_time']).strftime('%Y-%m-%d %H:%M')
            btn = Button(
                text=f"Session {session['id']+1}: {start} ({len(session['bets'])} bets)",
                size_hint_y=None,
                height=40,
                font_size=16
            )
            grid.add_widget(btn)
        
        scroll.add_widget(grid)
        layout.add_widget(scroll)
        
        close_btn = Button(text='Close', size_hint_y=None, height=40)
        close_btn.bind(on_release=popup.dismiss)
        layout.add_widget(close_btn)
        
        popup.content = layout
        popup.open()
    
    def end_current_session(self, instance):
        if self.memory.current_session:
            self.memory.end_session()
            self.undo_btn.disabled = True
            self.update_ui()
            self.prediction_label.text = "Session ended. Place first bet of new session."
    
    def reset_all_data(self, instance):
        content = BoxLayout(orientation='vertical', padding=10, spacing=10)
        msg = Label(text="Are you sure you want to reset ALL data?\nThis cannot be undone!")
        btn_layout = BoxLayout(spacing=10)
        
        confirm_btn = Button(text='Reset All', background_color=(0.8, 0.1, 0.1, 1))
        cancel_btn = Button(text='Cancel')
        
        popup = Popup(title='Confirm Reset', content=content, size_hint=(0.8, 0.4))
        
        def do_reset(_):
            self.memory.clear_all()
            self.predictor = AdaptivePredictor()
            self.update_ui()
            self.undo_btn.disabled = True
            popup.dismiss()
        
        confirm_btn.bind(on_release=do_reset)
        cancel_btn.bind(on_release=popup.dismiss)
        
        btn_layout.add_widget(confirm_btn)
        btn_layout.add_widget(cancel_btn)
        
        content.add_widget(msg)
        content.add_widget(btn_layout)
        popup.open()

class AdaptiveBettingApp(App):
    def build(self):
        return AdaptiveBettingUI()
    
    def on_stop(self):
        if hasattr(self, 'root') and self.root:
            self.root.memory.end_session()
            self.root.predictor.save_metrics()
            self.root.predictor.save_models()

if __name__ == '__main__':
    AdaptiveBettingApp().run()

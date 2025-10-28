"""
main_improved.py - High-Performance Bot System
- Improved target selection algorithm
- Better state management
- Advanced monitoring
- Optimized memory usage
"""

import time
import math
import threading
import logging
from typing import List, Dict, Optional, Tuple
from collections import deque
from dataclasses import dataclass
import random

from GClass_Improved import GameWindowImproved, Detection
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TargetInfo:
    """Information about a targeted position"""
    __slots__ = ['position', 'timestamp', 'attempts', 'last_atk_value']
    position: Tuple[int, int]
    timestamp: float
    attempts: int
    last_atk_value: int


class OptimizedPositionTracker:
    """
    High-performance position tracking with:
    - O(1) lookups
    - Automatic cleanup
    - Memory efficient
    """
    __slots__ = ['_positions', '_timestamps', '_max_size', '_tolerance']
    
    def __init__(self, max_size: int = 100, tolerance: int = 50):
        self._positions: Dict[Tuple[int, int], float] = {}
        self._timestamps: deque = deque(maxlen=max_size)
        self._max_size = max_size
        self._tolerance = tolerance
    
    def _normalize_pos(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        """Normalize position to tolerance grid"""
        return (
            (pos[0] // self._tolerance) * self._tolerance,
            (pos[1] // self._tolerance) * self._tolerance
        )
    
    def add(self, pos: Tuple[int, int], timestamp: float):
        """Add position with automatic cleanup"""
        norm_pos = self._normalize_pos(pos)
        self._positions[norm_pos] = timestamp
        self._timestamps.append((norm_pos, timestamp))
        
        # Auto cleanup when needed
        if len(self._positions) > self._max_size * 0.9:
            self._cleanup_old(timestamp, 30.0)
    
    def is_valid(self, pos: Tuple[int, int], current_time: float, cooldown: float) -> bool:
        """Check if position is valid (not on cooldown)"""
        norm_pos = self._normalize_pos(pos)
        if norm_pos not in self._positions:
            return True
        return (current_time - self._positions[norm_pos]) > cooldown
    
    def _cleanup_old(self, current_time: float, max_age: float):
        """Remove old entries"""
        to_remove = [p for p, t in self._positions.items() 
                    if current_time - t > max_age]
        for p in to_remove:
            del self._positions[p]
    
    def clear(self):
        """Clear all positions"""
        self._positions.clear()
        self._timestamps.clear()


class SmartTargetSelector:
    """
    Advanced target selection with:
    - Priority weighting
    - Distance optimization
    - Area filtering
    - Confidence scoring
    """
    
    def __init__(self, window_center: Tuple[int, int], target_area: Tuple[int, int],
                 bot_mode: str = "melee"):
        self.window_center = window_center
        self.target_area = target_area
        self.bot_mode = bot_mode
        
        # Load priority map from config
        self.priority_map = getattr(config, 'MONSTER_PRIORITY', {})
        self.selection_mode = getattr(config, 'TARGET_SELECTION_MODE', 'priority_distance')
        self.distance_weight = getattr(config, 'DISTANCE_WEIGHT', 0.3)
    
    def select_best_target(self, detections: List[Detection], 
                          cooldown_tracker: OptimizedPositionTracker,
                          dead_tracker: OptimizedPositionTracker,
                          failed_tracker: OptimizedPositionTracker,
                          current_time: float,
                          search_full_screen: bool = False) -> Optional[Detection]:
        """
        Select the best target using advanced scoring algorithm
        """
        if not detections:
            return None
        
        valid_monsters = [d for d in detections 
                         if d.class_name in config.MONSTER_CLASSES]
        
        if not valid_monsters:
            return None
        
        center_x, center_y = self.window_center
        target_w, target_h = self.target_area
        
        candidates = []
        
        for monster in valid_monsters:
            mx, my = monster.center
            monster_pos = (int(mx), int(my))
            
            # Filter out invalid positions
            if not dead_tracker.is_valid(monster_pos, current_time, 0):
                continue
            
            if not cooldown_tracker.is_valid(monster_pos, current_time, 0):
                continue
            
            if not failed_tracker.is_valid(monster_pos, current_time, 5.0):
                continue
            
            # Calculate distance
            dx = abs(mx - center_x)
            dy = abs(my - center_y)
            distance = math.sqrt(dx * dx + dy * dy)
            
            # Check if in target area
            in_area = dx <= target_w and dy <= target_h
            
            # Skip if not in area and not doing full screen search
            if not search_full_screen and not in_area:
                continue
            
            # Calculate score
            score = self._calculate_score(
                monster, distance, in_area, monster_pos
            )
            
            candidates.append((score, monster))
        
        if not candidates:
            return None
        
        # Return best candidate (lowest score)
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]
    
    def _calculate_score(self, monster: Detection, distance: float, 
                        in_area: bool, position: Tuple[int, int]) -> float:
        """Calculate target score (lower is better)"""
        # Get priority (default to 1 if not in map)
        priority = self.priority_map.get(monster.class_name, 1)
        
        # Normalize priority (1-10 scale to 0.1-1.0 multiplier)
        priority_multiplier = 1.1 - (priority / 10.0)
        
        # Base score from distance
        score = distance * priority_multiplier
        
        # Area bonus (prefer targets in designated area)
        if in_area:
            score *= 0.6
        
        # Confidence bonus (higher confidence = lower score)
        confidence_factor = 1.5 - (monster.confidence * 0.5)
        score *= confidence_factor
        
        # Size consideration (prefer medium-sized targets)
        if monster.area < 500:  # Too small
            score *= 1.2
        elif monster.area > 20000:  # Too large
            score *= 1.1
        
        return score


class ImprovedBotInstance:
    """
    High-performance bot instance with:
    - Smart target selection
    - Optimized state management
    - Better error recovery
    - Performance monitoring
    """
    
    def __init__(self, hwnd: int, model_path: str, bot_mode: str, 
                 enable_food_check: bool, device_mode: str = 'cuda'):
        self.hwnd = hwnd
        self.model_path = model_path
        self.bot_mode = bot_mode
        self.enable_food_check = enable_food_check
        self.device_mode = device_mode
        
        # Initialize bot
        self.bot: Optional[GameWindowImproved] = None
        
        # Optimized trackers
        self.cooldown_positions = OptimizedPositionTracker(max_size=50, tolerance=50)
        self.dead_positions = OptimizedPositionTracker(max_size=30, tolerance=50)
        self.failed_positions = OptimizedPositionTracker(max_size=40, tolerance=50)
        
        # State management
        self.current_target: Optional[TargetInfo] = None
        self.last_attack_time = time.time()
        self.last_target_time = time.time()
        self.last_usage_times = {
            'hp': time.time(),
            'mp': time.time(),
            'gp': time.time(),
            'food': time.time()
        }
        
        # Search state
        self.search_full_screen = False
        self.full_screen_attempts = 0
        
        # Statistics
        self.targets_attacked = 0
        self.successful_attacks = 0
        self.failed_attacks = 0
        self.detection_count = 0
        
        # Control
        self.is_running = False
        self.stats_thread: Optional[threading.Thread] = None
        
        # Window properties
        self.window_center: Optional[Tuple[int, int]] = None
        self.target_area: Optional[Tuple[int, int]] = None
        self.target_selector: Optional[SmartTargetSelector] = None
    
    def start(self):
        """Start the bot instance"""
        self.is_running = True
        
        try:
            # Initialize bot
            self.bot = GameWindowImproved(
                self.hwnd, 
                self.model_path, 
                device_mode=self.device_mode,
                enable_profiling=True
            )
            
            logger.info(f"[{self.bot.title}] Bot started in {self.bot_mode.upper()} mode")
            logger.info(f"[{self.bot.title}] Device: {self.device_mode.upper()}")
            
            # Calculate window properties
            self._init_window_properties()
            
            # Start stats monitoring thread
            self.stats_thread = threading.Thread(
                target=self._stats_monitor_thread,
                daemon=True
            )
            self.stats_thread.start()
            
            # Run main loop
            self._main_loop()
            
        except Exception as e:
            logger.error(f"Bot startup error: {e}")
        finally:
            self.stop()

    def _stats_monitor_thread(self):
        """Monitor and manage HP/MP/GP/Food"""
        logger.info(f"[{self.bot.title}] Stats monitor started")
        
        check_intervals = {
            'hp': config.HP_CHECK_INTERVAL,
            'mp': config.HP_CHECK_INTERVAL,
            'gp': config.STATS_CHECK_DELAY,
            'food': config.STATS_CHECK_DELAY
        }
        
        last_checks = {key: 0 for key in check_intervals}
        
        while self.is_running and self.bot.is_valid():
            try:
                current_time = time.time()
                
                # HP Check
                if (current_time - last_checks['hp']) >= check_intervals['hp']:
                    hp_value = self.bot.read_pointer_chain(
                        config.BASE_ADDRESS, config.HP_OFFSETS
                    )
                    
                    if hp_value and hp_value < config.HP_THRESHOLD:
                        if (current_time - self.last_usage_times['hp']) > 1.0:
                            logger.info(f"[{self.bot.title}] Low HP ({hp_value}), using F1")
                            if self.bot.send_key(config.HP_SHORTCUT):
                                self.last_usage_times['hp'] = current_time
                                time.sleep(0.3)
                    
                    last_checks['hp'] = current_time
                
                # MP Check
                if (current_time - last_checks['mp']) >= check_intervals['mp']:
                    mp_value = self.bot.read_pointer_chain(
                        config.BASE_ADDRESS, config.MP_OFFSETS
                    )
                    
                    if mp_value and mp_value < config.MP_THRESHOLD:
                        if (current_time - self.last_usage_times['mp']) > config.MP_COOLDOWN_DURATION:
                            logger.info(f"[{self.bot.title}] Low MP ({mp_value}), using F3")
                            if self.bot.send_key(config.MP_SHORTCUT):
                                self.last_usage_times['mp'] = current_time
                                time.sleep(0.3)
                    
                    last_checks['mp'] = current_time
                
                # GP Check (FIXED - try multiple reading methods)
                if config.ENABLE_GP_CHECK and (current_time - last_checks['gp']) >= check_intervals['gp']:
                    gp_value = None
                    
                    # Try reading as byte first
                    gp_value = self.bot.read_pointer_chain_byte(
                        config.BASE_ADDRESS, config.GP_OFFSETS
                    )
                    
                    # If byte reading fails, try reading as short
                    if gp_value is None:
                        gp_value = self.bot.read_pointer_chain_short(
                            config.BASE_ADDRESS, config.GP_OFFSETS
                        )
                    
                    # If still None, try reading as int (last resort)
                    if gp_value is None:
                        gp_value = self.bot.read_pointer_chain(
                            config.BASE_ADDRESS, config.GP_OFFSETS
                        )
                        # If we got an int, try to extract a reasonable value
                        if gp_value is not None:
                            # Try to extract a byte from the int
                            gp_value = gp_value & 0xFF
                    
                    # Debug logging
                    if gp_value is not None:
                        logger.debug(f"[{self.bot.title}] GP value: {gp_value}")
                    
                    if gp_value is not None and gp_value < config.GP_THRESHOLD:
                        if (current_time - self.last_usage_times['gp']) > config.GP_COOLDOWN_DURATION:
                            logger.info(f"[{self.bot.title}] Low GP ({gp_value}), using F5")
                            if self.bot.send_key(config.GP_SHORTCUT):
                                self.last_usage_times['gp'] = current_time
                                time.sleep(0.3)
                    
                    last_checks['gp'] = current_time
                
                # Food Check (IMPROVED)
                if self.enable_food_check and (current_time - last_checks['food']) >= check_intervals['food']:
                    hunger_value = self.bot.read_pointer_chain(
                        config.BASE_ADDRESS, config.HUNGER_OFFSETS
                    )
                    
                    if hunger_value is not None:
                        # Convert to signed
                        signed_hunger = hunger_value if hunger_value < 32768 else hunger_value - 65536
                        
                        # Use configurable threshold (default to 0)
                        hunger_threshold = getattr(config, 'HUNGER_THRESHOLD', 0)
                        
                        # Debug logging
                        logger.debug(f"[{self.bot.title}] Hunger value: {signed_hunger}, threshold: {hunger_threshold}")
                        
                        # Check if hungry (below threshold)
                        if signed_hunger < hunger_threshold:
                            if (current_time - self.last_usage_times['food']) > config.FOOD_COOLDOWN_DURATION:
                                logger.info(f"[{self.bot.title}] Hungry ({signed_hunger}), using F2")
                                if self.bot.send_key(config.FOOD_SHORTCUT):
                                    self.last_usage_times['food'] = current_time
                                    time.sleep(0.3)
                    
                    last_checks['food'] = current_time
                
                time.sleep(0.05)
                
            except Exception as e:
                logger.error(f"[{self.bot.title}] Stats monitor error: {e}")
                time.sleep(1)
        
        logger.info(f"[{self.bot.title}] Stats monitor stopped")



    
    def _init_window_properties(self):
        """Initialize window properties and target selector"""
        import win32gui
        
        try:
            rect = win32gui.GetWindowRect(self.bot.handle)
            width = rect[2] - rect[0]
            height = rect[3] - rect[1]
            self.window_center = (width // 2, height // 2)
            
            if self.bot_mode == "melee":
                self.target_area = (
                    config.MELEE_TARGET_WIDTH // 2,
                    config.MELEE_TARGET_HEIGHT // 2
                )
            else:
                self.target_area = (
                    config.RANGE_TARGET_WIDTH // 2,
                    config.RANGE_TARGET_HEIGHT // 2
                )
            
            self.target_selector = SmartTargetSelector(
                self.window_center,
                self.target_area,
                self.bot_mode
            )
            
        except Exception as e:
            logger.error(f"Failed to init window properties: {e}")
            self.window_center = (400, 300)
            self.target_area = (200, 150)
    
    def _stats_monitor_thread(self):
        """Monitor and manage HP/MP/GP/Food"""
        logger.info(f"[{self.bot.title}] Stats monitor started")
        
        check_intervals = {
            'hp': config.HP_CHECK_INTERVAL,
            'mp': config.HP_CHECK_INTERVAL,
            'gp': config.STATS_CHECK_DELAY,
            'food': config.STATS_CHECK_DELAY
        }
        
        last_checks = {key: 0 for key in check_intervals}
        
        while self.is_running and self.bot.is_valid():
            try:
                current_time = time.time()
                
                # HP Check
                if (current_time - last_checks['hp']) >= check_intervals['hp']:
                    hp_value = self.bot.read_pointer_chain(
                        config.BASE_ADDRESS, config.HP_OFFSETS
                    )
                    
                    if hp_value and hp_value < config.HP_THRESHOLD:
                        if (current_time - self.last_usage_times['hp']) > 1.0:
                            logger.info(f"[{self.bot.title}] Low HP ({hp_value}), using F1")
                            if self.bot.send_key(config.HP_SHORTCUT):
                                self.last_usage_times['hp'] = current_time
                                time.sleep(0.3)
                    
                    last_checks['hp'] = current_time
                
                # MP Check
                if (current_time - last_checks['mp']) >= check_intervals['mp']:
                    mp_value = self.bot.read_pointer_chain(
                        config.BASE_ADDRESS, config.MP_OFFSETS
                    )
                    
                    if mp_value and mp_value < config.MP_THRESHOLD:
                        if (current_time - self.last_usage_times['mp']) > config.MP_COOLDOWN_DURATION:
                            logger.info(f"[{self.bot.title}] Low MP ({mp_value}), using F3")
                            if self.bot.send_key(config.MP_SHORTCUT):
                                self.last_usage_times['mp'] = current_time
                                time.sleep(0.3)
                    
                    last_checks['mp'] = current_time
                
                # GP Check (FIXED - try multiple reading methods)
                if config.ENABLE_GP_CHECK and (current_time - last_checks['gp']) >= check_intervals['gp']:
                    gp_value = None
                    
                    # Try reading as byte first
                    gp_value = self.bot.read_pointer_chain_byte(
                        config.BASE_ADDRESS, config.GP_OFFSETS
                    )
                    
                    # If byte reading fails, try reading as short
                    if gp_value is None:
                        gp_value = self.bot.read_pointer_chain_short(
                            config.BASE_ADDRESS, config.GP_OFFSETS
                        )
                    
                    # If still None, try reading as int (last resort)
                    if gp_value is None:
                        gp_value = self.bot.read_pointer_chain(
                            config.BASE_ADDRESS, config.GP_OFFSETS
                        )
                        # If we got an int, try to extract a reasonable value
                        if gp_value is not None:
                            # Try to extract a byte from the int
                            gp_value = gp_value & 0xFF
                    
                    # Debug logging
                    if gp_value is not None:
                        logger.debug(f"[{self.bot.title}] GP value: {gp_value}")
                    
                    if gp_value is not None and gp_value < config.GP_THRESHOLD:
                        if (current_time - self.last_usage_times['gp']) > config.GP_COOLDOWN_DURATION:
                            logger.info(f"[{self.bot.title}] Low GP ({gp_value}), using F5")
                            if self.bot.send_key(config.GP_SHORTCUT):
                                self.last_usage_times['gp'] = current_time
                                time.sleep(0.3)
                    
                    last_checks['gp'] = current_time
                
                # Food Check (if enabled)
                if self.enable_food_check and (current_time - last_checks['food']) >= check_intervals['food']:
                    hunger_value = self.bot.read_pointer_chain(
                        config.BASE_ADDRESS, config.HUNGER_OFFSETS
                    )
                    
                    if hunger_value is not None:
                        # Convert to signed
                        signed_hunger = hunger_value if hunger_value < 32768 else hunger_value - 65536
                        
                        if signed_hunger < 0:
                            if (current_time - self.last_usage_times['food']) > config.FOOD_COOLDOWN_DURATION:
                                logger.info(f"[{self.bot.title}] Hungry ({signed_hunger}), using F2")
                                if self.bot.send_key(config.FOOD_SHORTCUT):
                                    self.last_usage_times['food'] = current_time
                                    time.sleep(0.3)
                    
                    last_checks['food'] = current_time
                
                time.sleep(0.05)
                
            except Exception as e:
                logger.error(f"[{self.bot.title}] Stats monitor error: {e}")
                time.sleep(1)
        
        logger.info(f"[{self.bot.title}] Stats monitor stopped")
    
    def _main_loop(self):
        """Main bot loop with improved logic"""
        logger.info(f"[{self.bot.title}] Main loop started")
        
        last_atk_value = None
        atk_unchanged_count = 0
        no_target_start = time.time()
        
        while self.is_running and self.bot.is_valid():
            try:
                current_time = time.time()
                
                # Read ATK value
                atk_value = self.bot.read_pointer_chain(
                    config.BASE_ADDRESS, config.ATK_OFFSETS
                )
                
                if atk_value is None:
                    time.sleep(0.05)
                    continue
                
                # Track ATK changes
                if last_atk_value == atk_value:
                    atk_unchanged_count += 1
                else:
                    atk_unchanged_count = 0
                    
                    # If ATK changed from 0 to non-zero, we successfully attacked
                    if last_atk_value == 0 and atk_value != 0:
                        self.successful_attacks += 1
                
                last_atk_value = atk_value
                
                # Mark target as dead if ATK unchanged for too long
                if atk_unchanged_count >= 3 and self.current_target:
                    self.dead_positions.add(self.current_target.position, current_time)
                    self.current_target = None
                    atk_unchanged_count = 0
                    self.bot.invalidate_cache()
                
                # Attack timeout recovery
                if atk_value != 0 and (current_time - self.last_attack_time) > config.MAX_ATTACK_WAIT:
                    logger.warning(f"[{self.bot.title}] Attack timeout, recovering...")
                    recovery_x = random.randint(350, 750)
                    recovery_y = random.randint(200, 600)
                    self.bot.send_click(recovery_x, recovery_y)
                    self.last_attack_time = current_time
                    self.current_target = None
                    self.bot.invalidate_cache()
                    time.sleep(0.1)
                    continue
                
                # Look for targets when ATK is ready
                if atk_value == 0:
                    # Wait for screen stabilization
                    time.sleep(0.1)
                    
                    # Get fresh detections
                    detections = self.bot.detect_objects_fast(
                        conf=config.YOLO_CONFIDENCE_THRESHOLD,
                        force_refresh=True
                    )
                    
                    self.detection_count += 1
                    
                    if detections:
                        # Select best target
                        target = self.target_selector.select_best_target(
                            detections,
                            self.cooldown_positions,
                            self.dead_positions,
                            self.failed_positions,
                            current_time,
                            self.search_full_screen
                        )
                        
                        if target:
                            # Attack target
                            if self._attack_target(target, current_time):
                                # Reset search state
                                self.search_full_screen = False
                                self.full_screen_attempts = 0
                                no_target_start = current_time
                                
                                # Wait for attack to register
                                time.sleep(config.ATTACK_SUCCESS_DELAY)
                                
                                # Verify attack started
                                new_atk_value = self.bot.read_pointer_chain(
                                    config.BASE_ADDRESS, config.ATK_OFFSETS
                                )
                                
                                if new_atk_value == 0:
                                    # Attack failed
                                    logger.debug(f"[{self.bot.title}] Attack failed to start")
                                    if self.current_target:
                                        self.failed_positions.add(
                                            self.current_target.position, current_time
                                        )
                                        self.current_target = None
                                    self.failed_attacks += 1
                                    self.bot.invalidate_cache()
                                else:
                                    # Attack successful
                                    last_atk_value = new_atk_value
                                    atk_unchanged_count = 0
                                    time.sleep(config.SCREEN_STABILIZATION_DELAY)
                                    self.current_target = None
                                    self.bot.invalidate_cache()
                                
                                continue
                            else:
                                # Click failed
                                self.current_target = None
                                self.search_full_screen = True
                                self.full_screen_attempts += 1
                                self.bot.invalidate_cache()
                        else:
                            # No valid target found
                            self.current_target = None
                            
                            if not self.search_full_screen:
                                if (current_time - no_target_start) >= 2.0:
                                    self.search_full_screen = True
                                    logger.debug(f"[{self.bot.title}] Expanding search to full screen")
                            else:
                                self.full_screen_attempts += 1
                                if self.full_screen_attempts >= 3:
                                    time.sleep(config.NO_TARGET_DELAY)
                                    self.full_screen_attempts = 0
                                    self.search_full_screen = False
                                    self.bot.invalidate_cache()
                        
                        time.sleep(config.NO_TARGET_DELAY)
                    else:
                        # No detections
                        self.current_target = None
                        time.sleep(0.1)
                else:
                    # ATK not ready - wait
                    time.sleep(config.ATK_CHECK_DELAY)
                    
                    # Stuck detection
                    if atk_value != 0 and atk_unchanged_count >= 5:
                        logger.debug(f"[{self.bot.title}] ATK stuck at {atk_value}, recovering")
                        self.current_target = None
                        self.bot.invalidate_cache()
                        atk_unchanged_count = 0
                
            except Exception as e:
                logger.error(f"[{self.bot.title}] Main loop error: {e}")
                self.current_target = None
                self.bot.invalidate_cache()
                time.sleep(0.5)
        
        logger.info(f"[{self.bot.title}] Main loop ended")
    
    def _attack_target(self, target: Detection, current_time: float) -> bool:
        """Attack a target and update tracking"""
        try:
            x, y = target.center
            target_pos = (int(x), int(y))
            
            # Add small random offset
            click_x = int(x) + random.randint(-2, 2)
            click_y = int(y) + random.randint(-2, 2)
            
            if self.bot.send_click(click_x, click_y):
                # Update tracking
                self.cooldown_positions.add(target_pos, current_time)
                self.last_attack_time = current_time
                self.last_target_time = current_time
                
                # Create target info
                self.current_target = TargetInfo(
                    position=target_pos,
                    timestamp=current_time,
                    attempts=1,
                    last_atk_value=0
                )
                
                self.targets_attacked += 1
                
                logger.debug(f"[{self.bot.title}] Attacked {target.class_name} at ({click_x}, {click_y})")
                
                # Wait for character movement
                time.sleep(config.SCREEN_MOVEMENT_DELAY)
                
                return True
                
        except Exception as e:
            logger.error(f"[{self.bot.title}] Attack error: {e}")
            self.current_target = None
        
        return False
    
    def stop(self):
        """Stop the bot and cleanup"""
        if not self.is_running:
            return
        
        self.is_running = False
        logger.info(f"[{self.bot.title if self.bot else 'Bot'}] Shutting down...")
        
        if self.bot:
            try:
                # Log statistics
                stats = self.bot.get_performance_stats()
                logger.info(f"[{self.bot.title}] Statistics:")
                logger.info(f"  Targets attacked: {self.targets_attacked}")
                logger.info(f"  Successful attacks: {self.successful_attacks}")
                logger.info(f"  Failed attacks: {self.failed_attacks}")
                logger.info(f"  Detections: {self.detection_count}")
                logger.info(f"  Avg FPS: {stats['fps']:.1f}")
                logger.info(f"  Cache hit rate: {stats['cache_hit_rate']:.1f}%")
                
                self.bot.cleanup()
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
    
    def get_status(self) -> Dict:
        """Get current bot status"""
        if not self.bot:
            return {'status': 'stopped'}
        
        stats = self.bot.get_performance_stats()
        
        return {
            'status': 'running' if self.is_running else 'stopped',
            'targets_attacked': self.targets_attacked,
            'successful_attacks': self.successful_attacks,
            'failed_attacks': self.failed_attacks,
            'fps': stats['fps'],
            'cache_hit_rate': stats['cache_hit_rate'],
            'device': stats['device']
        }


def main():
    """Main entry point for improved bot system"""
    import win32gui
    import signal
    
    shutdown_flag = threading.Event()
    
    def signal_handler(signum, frame):
        logger.info("Shutdown signal received")
        shutdown_flag.set()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        print("=" * 60)
        print("YOLOv8 Game Bot - High Performance Edition")
        print("=" * 60)
        
        # Device selection
        import torch
        has_cuda = torch.cuda.is_available()
        
        print("\nDevice Selection:")
        if has_cuda:
            print(f"[1] GPU (CUDA) - {torch.cuda.get_device_name(0)}")
        else:
            print("[1] GPU (CUDA) - Not Available")
        print("[2] CPU")
        
        device_choice = input("Select device (1/2) [1]: ").strip() or "1"
        device_mode = "cuda" if device_choice == "1" and has_cuda else "cpu"
        
        # Model selection
        import glob
        models = glob.glob("*.pt")
        if not models:
            logger.error("No YOLOv8 models found")
            return
        
        print("\nAvailable Models:")
        for i, model in enumerate(models):
            print(f"[{i+1}] {model}")
        
        model_choice = int(input("Select model: ")) - 1
        selected_model = models[model_choice]
        
        # Mode selection
        print("\nBot Mode:")
        print("[1] Melee")
        print("[2] Range")
        mode_choice = input("Select mode (1/2) [1]: ").strip() or "1"
        selected_mode = "melee" if mode_choice == "1" else "range"
        
        # Features
        food_check = input("\nEnable food management? (y/n) [y]: ").strip().lower() != 'n'
        
        # Find windows
        def find_windows():
            windows = []
            def callback(hwnd, extra):
                try:
                    if (win32gui.IsWindowVisible(hwnd) and 
                        win32gui.GetWindowText(hwnd) == config.WINDOW_TITLE):
                        windows.append(hwnd)
                except:
                    pass
                return True
            win32gui.EnumWindows(callback, None)
            return windows
        
        windows = find_windows()
        if not windows:
            logger.error(f"No '{config.WINDOW_TITLE}' windows found")
            return
        
        print(f"\nFound {len(windows)} game window(s)")
        for i, hwnd in enumerate(windows):
            print(f"[{i+1}] HWND {hwnd}")
        
        if len(windows) == 1:
            selected_windows = windows
        else:
            choice = input("Run on [1] single or [2] all windows? ")
            if choice == "1":
                idx = int(input("Select window number: ")) - 1
                selected_windows = [windows[idx]]
            else:
                selected_windows = windows
        
        print("\n" + "=" * 60)
        print(f"Starting {len(selected_windows)} bot(s)")
        print(f"Device: {device_mode.upper()}")
        print(f"Model: {selected_model}")
        print(f"Mode: {selected_mode.upper()}")
        print(f"Food: {'ENABLED' if food_check else 'DISABLED'}")
        print("=" * 60 + "\n")
        
        # Start bots
        bots = []
        bot_threads = []
        
        for i, hwnd in enumerate(selected_windows):
            try:
                bot_instance = ImprovedBotInstance(
                    hwnd, selected_model, selected_mode, food_check, device_mode
                )
                bots.append(bot_instance)
                
                bot_thread = threading.Thread(
                    target=bot_instance.start,
                    name=f"Bot-{i+1}",
                    daemon=True
                )
                bot_thread.start()
                bot_threads.append(bot_thread)
                
                print(f"✓ Bot #{i+1} started (HWND: {hwnd})")
                
            except Exception as e:
                logger.error(f"Failed to start bot {i+1}: {e}")
        
        if not bots:
            logger.error("No bots started")
            return
        
        print(f"\n{len(bots)} bot(s) running. Press Ctrl+C to stop.\n")
        
        # Monitor loop
        try:
            while not shutdown_flag.is_set() and any(t.is_alive() for t in bot_threads):
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutdown requested...")
        
    except Exception as e:
        logger.error(f"Main error: {e}")
    finally:
        shutdown_flag.set()
        
        print("Stopping bots...")
        for bot in bots:
            try:
                bot.stop()
            except:
                pass
        
        time.sleep(0.5)
        print("Shutdown complete")


if __name__ == "__main__":
    main()

# config_improved.py - Enhanced Configuration for High-Performance YOLOv8 Bot

import logging
import os
import win32con
from typing import List, Tuple, Dict
from dataclasses import dataclass
from enum import Enum

# ============================================================================
# DEBUG & LOGGING SETTINGS
# ============================================================================

DEBUG_GP = False  # Set to True to see GP values in console
DEBUG_MODE = False  # Enable debug mode for detailed logging
LOG_LEVEL = logging.INFO  # INFO for production, DEBUG for troubleshooting
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

# Set up logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)

# ============================================================================
# MEMORY AND GAME POINTERS
# ============================================================================

BASE_ADDRESS = 0x013B4B08

# Character stats offsets
ATK_OFFSETS = [0x18C, 0x8, 0x50]
HP_OFFSETS = [0x18C, 0x8, 0x210, 0x20]
MP_OFFSETS = [0x18C, 0x8, 0x210, 0x24]
GP_OFFSETS = [0x18C, 0x8, 0xE0, 0x18]
HUNGER_OFFSETS = [0x18C, 0x8, 0x400, 0x0]

# Character position offsets (for pathfinding)
CHAR_X_OFFSETS = [0x18C, 0x8, 0x21C, 0x123]
CHAR_Y_OFFSETS = [0x18C, 0x8, 0x21C, 0x127]

# ============================================================================
# GAME WINDOW SETTINGS
# ============================================================================

WINDOW_TITLE = "GoonZu"
MULTI_CLIENT_MODE = False  # Set to True if running multiple game clients

# ============================================================================
# YOLOV8 DETECTION SETTINGS
# ============================================================================

# Monster classes to detect and attack
MONSTER_CLASSES = [
    "Senile Tiger",
    "Man Cow",
    "Boor",
    "Round Crab",
    "Country Rat"
]

# Alternative configurations (uncomment as needed):
"""
# All monsters configuration
MONSTER_CLASSES = [
    "Evil-Raccoon",
    "Evil Raccoon",
    "Blue Lobster",
    "Boor",
    "Boss Mushroom",
    "Country Rat",
    "Green Mushroom Boy",
    "Mad Chicken",
    "Man Cow",
    "Pink Pig",
    "Pink Rabbit",
    "Princess Piggy",
    "Senile Tiger",
    "Wild Boar",
    "Shaman Ladybugs",
    "Green Hopper",
    "Cyclops Vine",
    "Dang-Fly",
    "Fruit Brothers",
    "Monster Tree"
]

# Boss farming configuration
MONSTER_CLASSES = ["Boss Mushroom", "Princess Piggy", "Man Cow"]

# Low level configuration
MONSTER_CLASSES = ["Pink Pig", "Country Rat", "Green Mushroom Boy"]
"""

# YOLOv8 confidence threshold (0.0 to 1.0)
# Lower = more detections but more false positives
# Higher = fewer detections but more accurate
YOLO_CONFIDENCE_THRESHOLD = 0.52

# YOLOv8 image size for detection
# Options: 320, 416, 512, 640, 768, 896, 1024, 1280
# Larger = more accurate but slower
# Recommended: 640 for balanced performance
YOLOV8_IMAGE_SIZE = 640

# YOLOv8 half precision (FP16) - GPU only
# Provides ~2x speed boost on compatible GPUs
YOLOV8_USE_HALF_PRECISION = True

# Maximum detections to process per frame
YOLOV8_MAX_DETECTIONS = 50

# YOLOv8 device preference
# Options: 'cuda', 'cpu', 'auto' (auto will choose best available)
YOLOV8_DEVICE = 'auto'

# Advanced YOLOv8 settings
YOLOV8_CPU_THREADS = 4  # Number of threads for CPU inference
YOLOV8_IOU_THRESHOLD = 0.45  # IoU threshold for NMS
YOLOV8_MAX_BOXES = 100  # Maximum boxes after NMS
YOLOV8_AUGMENT = False  # Augmentation during inference (slow)
YOLOV8_VISUALIZE = False  # Visualize detections (debug only, very slow)

# ============================================================================
# MONSTER PRIORITY SYSTEM
# ============================================================================

class TargetSelectionMode(Enum):
    """Target selection strategies"""
    PRIORITY_DISTANCE = "priority_distance"  # Balanced (recommended)
    PRIORITY_ONLY = "priority_only"          # Always highest priority
    DISTANCE_ONLY = "distance_only"          # Always closest

# Monster priority system (1-10 scale)
# Higher number = higher priority
# Monsters not in this dict will have default priority of 1
MONSTER_PRIORITY: Dict[str, int] = {
    # Boss/Rare monsters - Highest priority (9-10)
    "Boss Mushroom": 10,
    "Princess Piggy": 9,
    
    # High value targets (7-8)
    "Man Cow": 8,
    "Wild Boar": 7,
    "Shaman Ladybugs": 7,
    
    # Medium priority (5-6)
    "Senile Tiger": 6,
    "Mad Chicken": 6,
    "Evil Raccoon": 5,
    "Evil-Raccoon": 5,
    "Blue Lobster": 5,
    "Round Crab": 5,
    
    # Low priority (2-4)
    "Boor": 4,
    "Country Rat": 3,
    "Green Mushroom Boy": 3,
    "Pink Pig": 2,
    "Pink Rabbit": 2,
}

# Target selection strategy
TARGET_SELECTION_MODE = TargetSelectionMode.PRIORITY_DISTANCE.value

# Distance weight factor (used in priority_distance mode)
# Range: 0.0 to 1.0
# Lower value (0.1-0.3) = priority matters more
# Higher value (0.5-0.9) = distance matters more
DISTANCE_WEIGHT = 0.3  # 70% priority, 30% distance

# Size-based filtering
# Ignore detections that are too small (likely false positives)
MIN_DETECTION_AREA = 400  # pixels squared
# Ignore detections that are too large (likely background/UI)
MAX_DETECTION_AREA = 25000  # pixels squared

# ============================================================================
# POSITION AND COOLDOWN MANAGEMENT
# ============================================================================

# Cooldown for positions after clicking (in seconds)
POSITION_COOLDOWN_DURATION = 25

# Pixel tolerance to treat two positions as the same
POSITION_TOLERANCE = 50

# Maximum positions to track
MAX_TRACKED_POSITIONS = 100

# How long to remember "dead" positions (seconds)
DEAD_POSITION_MEMORY = 30

# How long to remember "failed" positions (seconds)
FAILED_POSITION_MEMORY = 15

# ============================================================================
# HEALTH, MANA, GOLD, AND HUNGER MANAGEMENT
# ============================================================================

# Health management
HP_THRESHOLD = 10  # Use healing when HP drops below this
HP_SHORTCUT = win32con.VK_F1  # F1 key for healing
HP_CHECK_INTERVAL = 0.3  # Check HP every 300ms (fast)
HP_COOLDOWN_DURATION = 1.0  # Minimum time between HP potions

# Magic management
MP_THRESHOLD = 5  # Use mana potion when MP drops below this
MP_SHORTCUT = win32con.VK_F3  # F3 key for mana restoration
MP_CHECK_INTERVAL = 0.3  # Check MP every 300ms
MP_COOLDOWN_DURATION = 3  # Seconds between MP potion usage

# Gold management
GP_THRESHOLD = 90  # Use gold management when GP drops below this
GP_SHORTCUT = win32con.VK_F5  # F5 key for gold management
ENABLE_GP_CHECK = True  # Set to False to disable GP checking
GP_COOLDOWN_DURATION = 5  # Seconds between GP management actions

# Food management
ENABLE_FOOD_CHECK = True  # Toggle food checking on/off
FOOD_SHORTCUT = win32con.VK_F2  # F2 key for food
FOOD_COOLDOWN_DURATION = 2  # Seconds between food usage
HUNGER_MIN_THRESHOLD = 65000  # Not used with new hunger system
HUNGER_MAX_THRESHOLD = 100000  # Not used with new hunger system

# Stats check interval (for MP, GP, Food)
STATS_CHECK_DELAY = 0.8  # Seconds between comprehensive stats checks

# ============================================================================
# ACTION DELAYS - OPTIMIZED FOR PERFORMANCE
# ============================================================================

# Keyboard and mouse delays
KEY_PRESS_DELAY = 0.1  # Delay for key presses
MOUSE_CLICK_DELAY = 0.1  # Delay for mouse clicks

# Attack-related delays
ATK_CHECK_DELAY = 0.1  # How often to check ATK value (faster = more responsive)
ATTACK_SUCCESS_DELAY = 0.1  # Delay after successful attack
SCREEN_MOVEMENT_DELAY = 0.4  # Wait for character movement and screen shift
SCREEN_STABILIZATION_DELAY = 0.4  # Additional wait for screen to stabilize
MAX_ATTACK_WAIT = 20  # Maximum seconds to wait for attack to complete

# Detection delays
NO_TARGET_DELAY = 0.15  # Delay when no target found (fast retry)
DETECTION_INTERVAL = 0.08  # Minimum time between YOLOv8 inferences
MEMORY_READ_INTERVAL = 0.2  # Minimum time between memory reads

# ============================================================================
# PERFORMANCE SETTINGS
# ============================================================================

# Detection caching
MAX_DETECTION_CACHE_TIME = 0.1  # Cache YOLOv8 detections for 100ms
CACHE_TTL_MULTIPLIER = 1.0  # Multiplier for cache TTL (increase for slower systems)

# Memory management
MAX_MEMORY_READ_RETRIES = 2  # Retries for memory reads
MEMORY_READ_TIMEOUT = 0.5  # Timeout for memory operations
SCREENSHOT_BUFFER_SIZE = 3  # Number of screenshot buffers to pool

# Performance optimization
ENABLE_ADAPTIVE_PERFORMANCE = False  # Auto-adjust settings based on performance
SCREENSHOT_SCALE_FACTOR = 1.0  # Scale screenshots (1.0 = no scaling)
MAX_DETECTION_RESULTS = 10  # Limit stored detection results
CLEANUP_INTERVAL = 30  # Cleanup interval in seconds
ENABLE_MEMORY_COMPACT = False  # Disabled for speed

# Safety settings
MAX_CONSECUTIVE_ERRORS = 10  # Stop bot after this many consecutive errors
ERROR_RECOVERY_DELAY = 5.0  # Delay after error before retry

# Performance monitoring
ENABLE_PERFORMANCE_LOGGING = True  # Log performance metrics
PERFORMANCE_LOG_INTERVAL = 60  # Log performance every N seconds

# ============================================================================
# TARGETING AREAS
# ============================================================================

# Melee targeting area (close range) - pixels from center
MELEE_TARGET_RADIUS = 100
MELEE_TARGET_WIDTH = 150
MELEE_TARGET_HEIGHT = 150

# Range targeting area (long range) - pixels from center
RANGE_TARGET_RADIUS = 250
RANGE_TARGET_WIDTH = 400
RANGE_TARGET_HEIGHT = 300

# ============================================================================
# PATHFINDING SETTINGS (Advanced)
# ============================================================================

PATHFINDING_ENABLED = False  # Enable to use pathfinding system
PATHFINDING_UPDATE_INTERVAL = 1.0  # Seconds between pathfinding updates
ARRIVAL_THRESHOLD = 3.0  # Units - consider arrived if within this distance
MAX_STEP_DISTANCE = 15.0  # Units per pathfinding step
STUCK_DETECTION_TIME = 10.0  # Seconds before considering character stuck
PATHFINDING_TIMEOUT = 30  # Seconds without targets before starting pathfinding

# Random walk settings (Legacy - use pathfinding instead)
ENABLE_RANDOM_WALK = False
RANDOM_WALK_TIMEOUT = 5
RANDOM_WALK_DISTANCE = 100
RANDOM_WALK_STEPS = 2

# ============================================================================
# DEBUG AND PROFILING SETTINGS
# ============================================================================

LOG_DETECTIONS = False  # Log every detection (very verbose)
SAVE_SCREENSHOTS = False  # Save screenshots for debugging
SCREENSHOT_INTERVAL = 30  # Save every N seconds
SCREENSHOT_SAVE_PATH = "./screenshots"  # Path to save screenshots

# Profiling
ENABLE_PROFILING = True  # Enable performance profiling
PROFILE_INTERVAL = 30  # Profile every N seconds

# ============================================================================
# PRESETS FOR DIFFERENT SCENARIOS
# ============================================================================

@dataclass
class PerformancePreset:
    """Performance preset configuration"""
    name: str
    detection_interval: float
    image_size: int
    cache_time: float
    description: str

# Performance presets
PRESETS = {
    'ultra': PerformancePreset(
        name='Ultra Performance',
        detection_interval=0.05,
        image_size=640,
        cache_time=0.05,
        description='Fastest detection, highest CPU/GPU usage'
    ),
    'high': PerformancePreset(
        name='High Performance',
        detection_interval=0.08,
        image_size=640,
        cache_time=0.1,
        description='Balanced speed and accuracy (recommended)'
    ),
    'balanced': PerformancePreset(
        name='Balanced',
        detection_interval=0.12,
        image_size=512,
        cache_time=0.15,
        description='Good performance with lower resource usage'
    ),
    'power_saver': PerformancePreset(
        name='Power Saver',
        detection_interval=0.2,
        image_size=416,
        cache_time=0.2,
        description='Lowest resource usage'
    ),
    'cpu_optimized': PerformancePreset(
        name='CPU Optimized',
        detection_interval=0.15,
        image_size=416,
        cache_time=0.2,
        description='Optimized for CPU-only systems'
    )
}

# Default preset
CURRENT_PRESET = 'high'

def apply_preset(preset_name: str):
    """Apply a performance preset"""
    global DETECTION_INTERVAL, YOLOV8_IMAGE_SIZE, MAX_DETECTION_CACHE_TIME
    
    if preset_name not in PRESETS:
        logging.warning(f"Unknown preset: {preset_name}")
        return
    
    preset = PRESETS[preset_name]
    DETECTION_INTERVAL = preset.detection_interval
    YOLOV8_IMAGE_SIZE = preset.image_size
    MAX_DETECTION_CACHE_TIME = preset.cache_time
    
    logging.info(f"Applied preset: {preset.name} - {preset.description}")

# ============================================================================
# CONFIGURATION VALIDATOR
# ============================================================================

class ConfigValidator:
    @staticmethod
    def validate_config():
        """Validate configuration values"""
        errors = []
        warnings = []
        
        # Validate thresholds
        if not (0.0 <= YOLO_CONFIDENCE_THRESHOLD <= 1.0):
            errors.append("YOLO_CONFIDENCE_THRESHOLD must be between 0.0 and 1.0")
        
        if HP_THRESHOLD < 0:
            errors.append("HP_THRESHOLD must be positive")
        
        if MP_THRESHOLD < 0:
            errors.append("MP_THRESHOLD must be positive")
        
        if GP_THRESHOLD < 0:
            errors.append("GP_THRESHOLD must be positive")
        
        # Validate delays
        delay_configs = [
            ('KEY_PRESS_DELAY', KEY_PRESS_DELAY),
            ('MOUSE_CLICK_DELAY', MOUSE_CLICK_DELAY),
            ('ATK_CHECK_DELAY', ATK_CHECK_DELAY),
            ('STATS_CHECK_DELAY', STATS_CHECK_DELAY),
            ('ATTACK_SUCCESS_DELAY', ATTACK_SUCCESS_DELAY),
            ('NO_TARGET_DELAY', NO_TARGET_DELAY)
        ]
        
        for name, value in delay_configs:
            if value < 0:
                errors.append(f"{name} must be non-negative")
        
        # Validate monster classes
        if not MONSTER_CLASSES:
            errors.append("MONSTER_CLASSES cannot be empty")
        
        # Validate YOLOv8 settings
        valid_sizes = [320, 416, 512, 640, 768, 896, 1024, 1280]
        if YOLOV8_IMAGE_SIZE not in valid_sizes:
            warnings.append(f"YOLOV8_IMAGE_SIZE should be one of {valid_sizes}")
        
        # Validate targeting areas
        if MELEE_TARGET_WIDTH <= 0 or MELEE_TARGET_HEIGHT <= 0:
            errors.append("MELEE_TARGET dimensions must be positive")
        
        if RANGE_TARGET_WIDTH <= 0 or RANGE_TARGET_HEIGHT <= 0:
            errors.append("RANGE_TARGET dimensions must be positive")
        
        # Validate HP check interval
        if HP_CHECK_INTERVAL <= 0:
            errors.append("HP_CHECK_INTERVAL must be positive")
        
        # Validate detection settings
        if MAX_DETECTION_CACHE_TIME < 0:
            errors.append("MAX_DETECTION_CACHE_TIME must be non-negative")
        
        if DETECTION_INTERVAL < 0:
            errors.append("DETECTION_INTERVAL must be non-negative")
        
        # Validate priority system
        valid_modes = [mode.value for mode in TargetSelectionMode]
        if TARGET_SELECTION_MODE not in valid_modes:
            errors.append(f"TARGET_SELECTION_MODE must be one of: {valid_modes}")
        
        if not (0.0 <= DISTANCE_WEIGHT <= 1.0):
            errors.append("DISTANCE_WEIGHT must be between 0.0 and 1.0")
        
        # Validate priority values
        for monster, priority in MONSTER_PRIORITY.items():
            if not (1 <= priority <= 10):
                errors.append(f"Priority for {monster} must be between 1 and 10, got {priority}")
        
        # Validate size filtering
        if MIN_DETECTION_AREA < 0:
            errors.append("MIN_DETECTION_AREA must be non-negative")
        
        if MAX_DETECTION_AREA < MIN_DETECTION_AREA:
            errors.append("MAX_DETECTION_AREA must be greater than MIN_DETECTION_AREA")
        
        # Performance warnings
        if DETECTION_INTERVAL < 0.05:
            warnings.append("Very low DETECTION_INTERVAL may cause high CPU usage")
        
        if YOLOV8_IMAGE_SIZE > 640 and not YOLOV8_USE_HALF_PRECISION:
            warnings.append("Large image size without FP16 may be slow")
        
        # Report results
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"- {error}" for error in errors)
            raise ValueError(error_msg)
        
        if warnings:
            warning_msg = "Configuration warnings:\n" + "\n".join(f"- {warning}" for warning in warnings)
            logging.warning(warning_msg)
        
        return True

# Validate configuration on import
try:
    ConfigValidator.validate_config()
except ValueError as e:
    logging.error(str(e))
    raise

# ============================================================================
# CONFIGURATION SUMMARY
# ============================================================================

def print_config_summary():
    """Print a summary of key configuration settings"""
    print("\n" + "="*70)
    print("YOLOv8 Bot Configuration Summary")
    print("="*70)
    
    # Detection settings
    print("\n📸 Detection Settings:")
    print(f"  Monster Classes: {len(MONSTER_CLASSES)} types")
    print(f"  Confidence Threshold: {YOLO_CONFIDENCE_THRESHOLD}")
    print(f"  Image Size: {YOLOV8_IMAGE_SIZE}")
    print(f"  Half Precision (FP16): {YOLOV8_USE_HALF_PRECISION}")
    print(f"  Device: {YOLOV8_DEVICE}")
    
    # Target selection
    print("\n🎯 Target Selection:")
    print(f"  Mode: {TARGET_SELECTION_MODE}")
    if TARGET_SELECTION_MODE == "priority_distance":
        print(f"  Distance Weight: {DISTANCE_WEIGHT} ({int((1-DISTANCE_WEIGHT)*100)}% priority, {int(DISTANCE_WEIGHT*100)}% distance)")
    print(f"  Size Filter: {MIN_DETECTION_AREA} - {MAX_DETECTION_AREA} px²")
    
    # Priorities
    if MONSTER_PRIORITY:
        print("\n⭐ Monster Priorities (Top 5):")
        sorted_monsters = sorted(MONSTER_PRIORITY.items(), key=lambda x: x[1], reverse=True)
        for monster, priority in sorted_monsters[:5]:
            print(f"  {monster}: {priority}/10")
        if len(sorted_monsters) > 5:
            print(f"  ... and {len(sorted_monsters) - 5} more")
    
    # Performance
    print("\n⚡ Performance:")
    print(f"  Detection Interval: {DETECTION_INTERVAL*1000:.0f}ms")
    print(f"  Cache Time: {MAX_DETECTION_CACHE_TIME*1000:.0f}ms")
    print(f"  Attack Check: {ATK_CHECK_DELAY*1000:.0f}ms")
    print(f"  HP Check: {HP_CHECK_INTERVAL*1000:.0f}ms")
    
    # Features
    print("\n🔧 Features:")
    print(f"  Food Management: {'ON' if ENABLE_FOOD_CHECK else 'OFF'}")
    print(f"  GP Management: {'ON' if ENABLE_GP_CHECK else 'OFF'}")
    print(f"  Pathfinding: {'ON' if PATHFINDING_ENABLED else 'OFF'}")
    print(f"  Performance Logging: {'ON' if ENABLE_PERFORMANCE_LOGGING else 'OFF'}")
    
    # Resources
    print("\n💾 Resource Management:")
    print(f"  Max Tracked Positions: {MAX_TRACKED_POSITIONS}")
    print(f"  Screenshot Buffers: {SCREENSHOT_BUFFER_SIZE}")
    print(f"  Position Cooldown: {POSITION_COOLDOWN_DURATION}s")
    
    print("="*70 + "\n")

# Auto-print summary if not imported as module
if __name__ == "__main__":

    print_config_summary()

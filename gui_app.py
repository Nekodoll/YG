# gui_app.py - Modern GUI for YOLOv8 Game Bot
# High-performance GUI with real-time monitoring

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time
import queue
from typing import Dict, List, Optional
import win32gui
from pathlib import Path
import glob
import logging
import sys
from io import StringIO
import json
import win32con

# Import bot components - FIXED IMPORTS
from GClass_Improved import GameWindowImproved
from main_improved import ImprovedBotInstance
import config


class GUILogHandler(logging.Handler):
    """Custom logging handler that sends logs to the GUI"""
    
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue
        
    def emit(self, record):
        """Emit a log record to the queue"""
        try:
            msg = self.format(record)
            self.log_queue.put(('log', msg))
        except Exception:
            self.handleError(record)


class BotController:
    """Controller for managing a single bot instance"""
    
    def __init__(self, hwnd: int, model_path: str, mode: str, 
                 enable_food: bool, enable_gp: bool, device: str, log_queue: queue.Queue):
        self.hwnd = hwnd
        self.model_path = model_path
        self.mode = mode
        self.enable_food = enable_food
        self.enable_gp = enable_gp
        self.device = device
        self.log_queue = log_queue
        
        self.bot: Optional[ImprovedBotInstance] = None
        self.thread: Optional[threading.Thread] = None
        self.targets_attacked = 0
        self.status = "Stopped"
        
        # Set up logger for this bot
        self.logger = logging.getLogger(f"Bot_{hwnd}")
        self.logger.setLevel(logging.INFO)
        
    def start(self):
        """Start the bot instance"""
        if self.thread and self.thread.is_alive():
            return False
            
        try:
            # Update config with GUI settings
            config.ENABLE_FOOD_CHECK = self.enable_food
            config.ENABLE_GP_CHECK = self.enable_gp
            
            self.bot = ImprovedBotInstance(
                self.hwnd, 
                self.model_path, 
                self.mode, 
                self.enable_food,
                self.device
            )
            
            self.thread = threading.Thread(target=self._run_bot, daemon=True)
            self.thread.start()
            self.status = "Starting"
            self.log_queue.put(('log', f"Bot {self.hwnd}: Starting..."))
            return True
            
        except Exception as e:
            self.status = f"Error: {str(e)}"
            self.log_queue.put(('log', f"Bot {self.hwnd}: Error - {str(e)}"))
            return False
    
    def _run_bot(self):
        """Run the bot instance"""
        try:
            self.status = "Running"
            self.log_queue.put(('log', f"Bot {self.hwnd}: Running"))
            self.bot.start()
        except Exception as e:
            self.status = f"Error: {str(e)}"
            self.log_queue.put(('log', f"Bot {self.hwnd}: Error - {str(e)}"))
        finally:
            self.status = "Stopped"
            self.log_queue.put(('log', f"Bot {self.hwnd}: Stopped"))
    
    def stop(self):
        """Stop the bot instance"""
        if self.bot:
            self.bot.stop()
            self.status = "Stopping"
            self.log_queue.put(('log', f"Bot {self.hwnd}: Stopping..."))
    
    def get_stats(self):
        """Get bot statistics"""
        if self.bot:
            stats = self.bot.get_status()
            self.targets_attacked = stats.get('targets_attacked', 0)
            return stats
        return {'status': self.status, 'targets_attacked': 0}


class MonsterPreset:
    """Monster preset configuration"""
    
    def __init__(self, name: str, monsters: List[str], description: str = ""):
        self.name = name
        self.monsters = monsters
        self.description = description
    
    def to_dict(self):
        return {
            'name': self.name,
            'monsters': self.monsters,
            'description': self.description
        }
    
    @classmethod
    def from_dict(cls, data):
        return cls(data['name'], data['monsters'], data.get('description', ''))


class BotGUI:
    """Modern GUI for YOLOv8 Game Bot with real-time monitoring"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("YOLOv8 Game Bot")
        self.root.geometry("700x700")
        self.root.resizable(True, True)
        
        # State management
        self.bots: Dict[int, BotController] = {}
        self.is_running = False
        self.log_queue = queue.Queue()
        self.available_monsters = []
        self.monster_presets = []
        self.selected_monsters = []
        
        # Load available monsters from config
        self._load_available_monsters()
        
        # Load presets
        self._load_presets()
        
        # Set up logging to capture all logs
        self._setup_logging()
        
        # Styling
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self._configure_styles()
        
        # Build UI
        self._create_ui()
        
        # Start update loop
        self._update_gui()
        
    def _load_available_monsters(self):
        """Load available monsters from config or use defaults"""
        # Try to get from config
        if hasattr(config, 'MONSTER_CLASSES'):
            self.available_monsters = config.MONSTER_CLASSES.copy()
        else:
            # Default monster list
            self.available_monsters = [
                "Senile Tiger", "Man Cow", "Boor", "Round Crab", "Country Rat",
                "Evil-Raccoon", "Evil Raccoon", "Blue Lobster", "Boss Mushroom",
                "Green Mushroom Boy", "Mad Chicken", "Pink Pig", "Pink Rabbit",
                "Princess Piggy", "Wild Boar", "Shaman Ladybugs", "Green Hopper",
                "Cyclops Vine", "Dang-Fly", "Fruit Brothers", "Monster Tree"
            ]
        
        # Load from model classes if available
        try:
            # Try to get class names from the first available model
            model_files = glob.glob("*.pt")
            if model_files:
                from ultralytics import YOLO
                temp_model = YOLO(model_files[0])
                if hasattr(temp_model.model, 'names'):
                    model_classes = list(temp_model.model.names.values())
                    # Merge with available monsters
                    for cls in model_classes:
                        if cls not in self.available_monsters:
                            self.available_monsters.append(cls)
        except:
            pass
        
        self.selected_monsters = self.available_monsters.copy()
        
    def _load_presets(self):
        """Load monster presets from file"""
        presets_file = "monster_presets.json"
        default_presets = [
            MonsterPreset("All Monsters", self.available_monsters, "All available monsters"),
            MonsterPreset("Low Level", ["Pink Pig", "Country Rat", "Green Mushroom Boy"], "Good for beginners"),
            MonsterPreset("Medium Level", ["Senile Tiger", "Man Cow", "Boor", "Round Crab"], "Balanced exp and loot"),
            MonsterPreset("High Level", ["Boss Mushroom", "Princess Piggy", "Wild Boar"], "High value targets"),
            MonsterPreset("Farm", ["Shaman Ladybugs", "Green Hopper", "Blue Lobster"], "Good for farming")
        ]
        
        try:
            if Path(presets_file).exists():
                with open(presets_file, 'r') as f:
                    data = json.load(f)
                    self.monster_presets = [MonsterPreset.from_dict(p) for p in data]
            else:
                self.monster_presets = default_presets
                self._save_presets()
        except:
            self.monster_presets = default_presets
    
    def _save_presets(self):
        """Save monster presets to file"""
        try:
            with open("monster_presets.json", 'w') as f:
                json.dump([p.to_dict() for p in self.monster_presets], f, indent=2)
        except:
            pass
    
    def _setup_logging(self):
        """Set up logging to redirect to GUI"""
        # Create custom handler
        self.gui_handler = GUILogHandler(self.log_queue)
        self.gui_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.gui_handler.setFormatter(formatter)
        
        # Get root logger and add our handler
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid console output
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Add our GUI handler
        root_logger.addHandler(self.gui_handler)
        
        # Also configure the main logger
        logger = logging.getLogger(__name__)
        logger.info("GUI Logging initialized")
        
    def _configure_styles(self):
        """Configure modern UI styles"""
        # Colors
        bg_color = "#2b2b2b"
        fg_color = "#ffffff"
        accent_color = "#4CAF50"
        
        self.root.configure(bg=bg_color)
        
        # Frame styles
        self.style.configure("Card.TFrame", background="#363636", relief="flat")
        self.style.configure("Main.TFrame", background=bg_color)
        self.style.configure("Tab.TFrame", background="#363636")
        
        # Label styles
        self.style.configure("Title.TLabel", background=bg_color, foreground=fg_color, 
                           font=("Arial", 16, "bold"))
        self.style.configure("Header.TLabel", background="#363636", foreground=fg_color,
                           font=("Arial", 11, "bold"))
        self.style.configure("Info.TLabel", background="#363636", foreground=fg_color,
                           font=("Arial", 10))
        self.style.configure("Status.TLabel", background=bg_color, foreground=accent_color,
                           font=("Arial", 10, "bold"))
        
        # Button styles
        self.style.configure("Action.TButton", font=("Arial", 10, "bold"))
        
        # Notebook (tabs) style
        self.style.configure("TNotebook", background=bg_color)
        self.style.configure("TNotebook.Tab", padding=[20, 8])
        
    def _create_ui(self):
        """Create the main UI layout with tabs"""
        # Main container with padding
        main_frame = ttk.Frame(self.root, style="Main.TFrame", padding=10)
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title
        '''
        title = ttk.Label(main_frame, text="YOLOv8 Game Bot Control Panel", 
                         style="Title.TLabel")
        title.grid(row=0, column=0, pady=(0, 10), sticky="w")
        '''
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=1, column=0, sticky="nsew")
        
        # Create tabs
        self._create_config_tab()
        self._create_monster_tab()
        self._create_monitor_tab()
        
        # Bottom panel - Status and controls (removed config button)
        self._create_control_panel(main_frame)
        
    def _create_config_tab(self):
        """Create configuration tab"""
        config_frame = ttk.Frame(self.notebook, style="Tab.TFrame")
        self.notebook.add(config_frame, text="⚙️ Configuration")
        
        # Left side - Basic config
        left_frame = ttk.Frame(config_frame, style="Tab.TFrame", padding=15)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        # Model selection
        ttk.Label(left_frame, text="YOLOv8 Model:", style="Info.TLabel").grid(
            row=0, column=0, sticky="w", pady=5)
        
        self.model_var = tk.StringVar()
        model_frame = ttk.Frame(left_frame, style="Tab.TFrame")
        model_frame.grid(row=0, column=1, sticky="ew", pady=5)
        
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, 
                                        state="readonly", width=20)
        self.model_combo.pack(side="left", padx=(0, 5))
        
        ttk.Button(model_frame, text="📁", width=3, 
                  command=self._browse_model).pack(side="left")
        
        self._load_available_models()
        
        # Device selection
        ttk.Label(left_frame, text="Device:", style="Info.TLabel").grid(
            row=1, column=0, sticky="w", pady=5)
        
        self.device_var = tk.StringVar(value="cuda")
        device_frame = ttk.Frame(left_frame, style="Tab.TFrame")
        device_frame.grid(row=1, column=1, sticky="w", pady=5)
        
        ttk.Radiobutton(device_frame, text="GPU (CUDA)", variable=self.device_var,
                       value="cuda").pack(side="left", padx=(0, 10))
        ttk.Radiobutton(device_frame, text="CPU", variable=self.device_var,
                       value="cpu").pack(side="left")
        
        # Bot mode
        ttk.Label(left_frame, text="Bot Mode:", style="Info.TLabel").grid(
            row=2, column=0, sticky="w", pady=5)
        
        self.mode_var = tk.StringVar(value="melee")
        mode_frame = ttk.Frame(left_frame, style="Tab.TFrame")
        mode_frame.grid(row=2, column=1, sticky="w", pady=5)
        
        ttk.Radiobutton(mode_frame, text="Melee", variable=self.mode_var,
                       value="melee").pack(side="left", padx=(0, 10))
        ttk.Radiobutton(mode_frame, text="Range", variable=self.mode_var,
                       value="range").pack(side="left")
        
        # Features
        ttk.Label(left_frame, text="Features:", style="Info.TLabel").grid(
            row=3, column=0, sticky="nw", pady=5)
        
        features_frame = ttk.Frame(left_frame, style="Tab.TFrame")
        features_frame.grid(row=3, column=1, sticky="w", pady=5)
        
        self.food_var = tk.BooleanVar(value=True)
        self.gp_var = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(features_frame, text="Food Management", 
                       variable=self.food_var).pack(anchor="w")
        ttk.Checkbutton(features_frame, text="GP Management", 
                       variable=self.gp_var).pack(anchor="w")
        
        # Window selection
        ttk.Label(left_frame, text="Game Windows:", style="Info.TLabel").grid(
            row=4, column=0, sticky="nw", pady=5)
        
        window_frame = ttk.Frame(left_frame, style="Tab.TFrame")
        window_frame.grid(row=4, column=1, sticky="ew", pady=5)
        
        self.window_listbox = tk.Listbox(window_frame, height=4, selectmode=tk.MULTIPLE,
                                         bg="#2b2b2b", fg="#ffffff", 
                                         selectbackground="#4CAF50")
        self.window_listbox.pack(fill="both", expand=True)
        
        ttk.Button(window_frame, text="🔄 Refresh Windows", 
                  command=self._refresh_windows).pack(pady=(5, 0), fill="x")
        
        self._refresh_windows()
        
        # Right side - HP/MP Settings and Performance
        right_frame = ttk.Frame(config_frame, style="Tab.TFrame", padding=15)
        right_frame.grid(row=0, column=1, sticky="nsew")
        
        # HP/MP Settings
        hpmp_label = ttk.Label(right_frame, text="❤️ HP/MP Settings", style="Header.TLabel")
        hpmp_label.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))
        
        # HP Threshold
        ttk.Label(right_frame, text="HP Threshold:", style="Info.TLabel").grid(
            row=1, column=0, sticky="w", pady=5)
        
        self.hp_threshold_var = tk.IntVar(value=getattr(config, 'HP_THRESHOLD', 100))
        hp_spinbox = ttk.Spinbox(right_frame, from_=0, to=10000, textvariable=self.hp_threshold_var, width=10)
        hp_spinbox.grid(row=1, column=1, sticky="w", pady=5)
        
        # HP Key
        ttk.Label(right_frame, text="HP Key (F1):", style="Info.TLabel").grid(
            row=2, column=0, sticky="w", pady=5)
        
        self.hp_key_var = tk.StringVar(value="F1")
        hp_key_combo = ttk.Combobox(right_frame, textvariable=self.hp_key_var,
                                    values=["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"],
                                    state="readonly", width=8)
        hp_key_combo.grid(row=2, column=1, sticky="w", pady=5)
        
        # MP Threshold
        ttk.Label(right_frame, text="MP Threshold:", style="Info.TLabel").grid(
            row=3, column=0, sticky="w", pady=5)
        
        self.mp_threshold_var = tk.IntVar(value=getattr(config, 'MP_THRESHOLD', 50))
        mp_spinbox = ttk.Spinbox(right_frame, from_=0, to=10000, textvariable=self.mp_threshold_var, width=10)
        mp_spinbox.grid(row=3, column=1, sticky="w", pady=5)
        
        # MP Key
        ttk.Label(right_frame, text="MP Key (F3):", style="Info.TLabel").grid(
            row=4, column=0, sticky="w", pady=5)
        
        self.mp_key_var = tk.StringVar(value="F3")
        mp_key_combo = ttk.Combobox(right_frame, textvariable=self.mp_key_var,
                                    values=["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"],
                                    state="readonly", width=8)
        mp_key_combo.grid(row=4, column=1, sticky="w", pady=5)
        
        # GP Threshold
        ttk.Label(right_frame, text="GP Threshold:", style="Info.TLabel").grid(
            row=5, column=0, sticky="w", pady=5)
        
        self.gp_threshold_var = tk.IntVar(value=getattr(config, 'GP_THRESHOLD', 80))
        gp_spinbox = ttk.Spinbox(right_frame, from_=0, to=255, textvariable=self.gp_threshold_var, width=10)
        gp_spinbox.grid(row=5, column=1, sticky="w", pady=5)
        
        # GP Key
        ttk.Label(right_frame, text="GP Key (F5):", style="Info.TLabel").grid(
            row=6, column=0, sticky="w", pady=5)
        
        self.gp_key_var = tk.StringVar(value="F5")
        gp_key_combo = ttk.Combobox(right_frame, textvariable=self.gp_key_var,
                                    values=["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"],
                                    state="readonly", width=8)
        gp_key_combo.grid(row=6, column=1, sticky="w", pady=5)
        
        # Food Key
        ttk.Label(right_frame, text="Food Key (F2):", style="Info.TLabel").grid(
            row=7, column=0, sticky="w", pady=5)
        
        self.food_key_var = tk.StringVar(value="F2")
        food_key_combo = ttk.Combobox(right_frame, textvariable=self.food_key_var,
                                      values=["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"],
                                      state="readonly", width=8)
        food_key_combo.grid(row=7, column=1, sticky="w", pady=5)
        
        # Performance settings (moved down)
        perf_label = ttk.Label(right_frame, text="⚡ Performance", style="Header.TLabel")
        perf_label.grid(row=8, column=0, columnspan=2, sticky="w", pady=(20, 10))
        
        # Detection interval
        ttk.Label(right_frame, text="Detection Speed:", style="Info.TLabel").grid(
            row=9, column=0, sticky="w", pady=5)
        
        self.speed_var = tk.StringVar(value="normal")
        speed_frame = ttk.Frame(right_frame, style="Tab.TFrame")
        speed_frame.grid(row=9, column=1, sticky="w", pady=5)
        
        ttk.Radiobutton(speed_frame, text="Fast", variable=self.speed_var,
                       value="fast").pack(side="left", padx=(0, 5))
        ttk.Radiobutton(speed_frame, text="Normal", variable=self.speed_var,
                       value="normal").pack(side="left", padx=(0, 5))
        ttk.Radiobutton(speed_frame, text="Slow", variable=self.speed_var,
                       value="slow").pack(side="left")
        
        # Confidence threshold
        ttk.Label(right_frame, text="Confidence:", style="Info.TLabel").grid(
            row=10, column=0, sticky="w", pady=5)
        
        self.confidence_var = tk.DoubleVar(value=0.6)
        confidence_scale = ttk.Scale(right_frame, from_=0.1, to=1.0, 
                                    variable=self.confidence_var, orient="horizontal", length=150)
        confidence_scale.grid(row=10, column=1, sticky="w", pady=5)
        
        self.confidence_label = ttk.Label(right_frame, text="0.60", style="Info.TLabel")
        self.confidence_label.grid(row=10, column=2, sticky="w", padx=5)
        
        # Update confidence label
        def update_confidence_label(value):
            self.confidence_label.config(text=f"{float(value):.2f}")
        confidence_scale.config(command=update_confidence_label)
        
    def _create_monster_tab(self):
        """Create monster selection tab"""
        monster_frame = ttk.Frame(self.notebook, style="Tab.TFrame")
        self.notebook.add(monster_frame, text="👾 Monsters")
        
        # Left side - Presets
        left_frame = ttk.Frame(monster_frame, style="Tab.TFrame", padding=15)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        # Preset header
        preset_label = ttk.Label(left_frame, text="📋 Monster Presets", style="Header.TLabel")
        preset_label.grid(row=0, column=0, columnspan=2, pady=(0, 15), sticky="w")
        
        # Preset list
        self.preset_listbox = tk.Listbox(left_frame, height=8, bg="#2b2b2b", fg="#ffffff",
                                         selectbackground="#4CAF50")
        self.preset_listbox.grid(row=1, column=0, columnspan=2, sticky="ew", pady=5)
        
        # Update preset list
        self._update_preset_list()
        
        # Preset buttons
        preset_btn_frame = ttk.Frame(left_frame, style="Tab.TFrame")
        preset_btn_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=5)
        
        ttk.Button(preset_btn_frame, text="Load Preset", 
                  command=self._load_preset).pack(side="left", padx=2)
        ttk.Button(preset_btn_frame, text="Save Preset", 
                  command=self._save_preset).pack(side="left", padx=2)
        ttk.Button(preset_btn_frame, text="Delete Preset", 
                  command=self._delete_preset).pack(side="left", padx=2)
        
        # Right side - Monster selection
        right_frame = ttk.Frame(monster_frame, style="Tab.TFrame", padding=15)
        right_frame.grid(row=0, column=1, sticky="nsew")
        
        # Monster selection header
        monster_label = ttk.Label(right_frame, text="🎯 Monster Selection", style="Header.TLabel")
        monster_label.grid(row=0, column=0, columnspan=2, pady=(0, 15), sticky="w")
        
        # Search box
        ttk.Label(right_frame, text="Search:", style="Info.TLabel").grid(
            row=1, column=0, sticky="w", pady=5)
        
        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(right_frame, textvariable=self.search_var, width=20)
        search_entry.grid(row=1, column=1, sticky="ew", pady=5)
        search_entry.bind('<KeyRelease>', self._filter_monsters)
        
        # Monster listbox with scrollbar
        monster_list_frame = ttk.Frame(right_frame, style="Tab.TFrame")
        monster_list_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=5)
        right_frame.rowconfigure(2, weight=1)
        right_frame.columnconfigure(0, weight=1)
        right_frame.columnconfigure(1, weight=1)
        
        # Available monsters
        ttk.Label(monster_list_frame, text="Available:", style="Info.TLabel").grid(
            row=0, column=0, sticky="w", pady=5)
        
        available_frame = ttk.Frame(monster_list_frame, style="Tab.TFrame")
        available_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 5))
        available_frame.rowconfigure(0, weight=1)
        available_frame.columnconfigure(0, weight=1)
        
        self.available_listbox = tk.Listbox(available_frame, height=12, bg="#2b2b2b", 
                                           fg="#ffffff", selectbackground="#4CAF50",
                                           selectmode=tk.MULTIPLE)
        self.available_listbox.grid(row=0, column=0, sticky="nsew")
        
        available_scroll = ttk.Scrollbar(available_frame, orient="vertical",
                                        command=self.available_listbox.yview)
        self.available_listbox.configure(yscrollcommand=available_scroll.set)
        available_scroll.grid(row=0, column=1, sticky="ns")
        
        # Buttons
        btn_frame = ttk.Frame(monster_list_frame, style="Tab.TFrame")
        btn_frame.grid(row=1, column=1, sticky="ns", pady=5)
        
        ttk.Button(btn_frame, text="▶", width=3, 
                  command=self._add_monsters).pack(pady=2)
        ttk.Button(btn_frame, text="◀", width=3, 
                  command=self._remove_monsters).pack(pady=2)
        ttk.Button(btn_frame, text="▶▶", width=3, 
                  command=self._add_all_monsters).pack(pady=2)
        ttk.Button(btn_frame, text="◀◀", width=3, 
                  command=self._remove_all_monsters).pack(pady=2)
        
        # Selected monsters
        ttk.Label(monster_list_frame, text="Selected:", style="Info.TLabel").grid(
            row=0, column=2, sticky="w", pady=5)
        
        selected_frame = ttk.Frame(monster_list_frame, style="Tab.TFrame")
        selected_frame.grid(row=1, column=2, sticky="nsew", padx=(5, 0))
        selected_frame.rowconfigure(0, weight=1)
        selected_frame.columnconfigure(0, weight=1)
        
        self.selected_listbox = tk.Listbox(selected_frame, height=12, bg="#2b2b2b", 
                                          fg="#ffffff", selectbackground="#4CAF50",
                                          selectmode=tk.MULTIPLE)
        self.selected_listbox.grid(row=0, column=0, sticky="nsew")
        
        selected_scroll = ttk.Scrollbar(selected_frame, orient="vertical",
                                       command=self.selected_listbox.yview)
        self.selected_listbox.configure(yscrollcommand=selected_scroll.set)
        selected_scroll.grid(row=0, column=1, sticky="ns")
        
        # Quick actions
        quick_frame = ttk.Frame(right_frame, style="Tab.TFrame")
        quick_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=10)
        
        ttk.Button(quick_frame, text="Select All", 
                  command=self._add_all_monsters).pack(side="left", padx=5)
        ttk.Button(quick_frame, text="Deselect All", 
                  command=self._remove_all_monsters).pack(side="left", padx=5)
        ttk.Button(quick_frame, text="Reset to Default", 
                  command=self._reset_monsters).pack(side="left", padx=5)
        
        # Update monster lists
        self._update_monster_lists()
        
    def _create_monitor_tab(self):
        """Create monitoring tab"""
        monitor_frame = ttk.Frame(self.notebook, style="Tab.TFrame")
        self.notebook.add(monitor_frame, text="📊 Monitor")
        
        # Bot status
        status_frame = ttk.Frame(monitor_frame, style="Tab.TFrame", padding=15)
        status_frame.grid(row=0, column=0, sticky="ew")
        
        status_label = ttk.Label(status_frame, text="🤖 Bot Status", style="Header.TLabel")
        status_label.grid(row=0, column=0, pady=(0, 15), sticky="w")
        
        # Create treeview for bot monitoring
        columns = ("Bot", "Status", "Targets", "FPS", "Device")
        self.bot_tree = ttk.Treeview(status_frame, columns=columns, show="headings",
                                     height=8)
        
        # Configure columns
        self.bot_tree.heading("Bot", text="Bot")
        self.bot_tree.heading("Status", text="Status")
        self.bot_tree.heading("Targets", text="Targets Hit")
        self.bot_tree.heading("FPS", text="Detection FPS")
        self.bot_tree.heading("Device", text="Device")
        
        self.bot_tree.column("Bot", width=150)
        self.bot_tree.column("Status", width=100)
        self.bot_tree.column("Targets", width=100)
        self.bot_tree.column("FPS", width=100)
        self.bot_tree.column("Device", width=100)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(status_frame, orient="vertical", 
                                 command=self.bot_tree.yview)
        self.bot_tree.configure(yscrollcommand=scrollbar.set)
        
        self.bot_tree.grid(row=1, column=0, sticky="ew")
        scrollbar.grid(row=1, column=1, sticky="ns")
        
        # Log area
        log_frame = ttk.Frame(monitor_frame, style="Tab.TFrame", padding=15)
        log_frame.grid(row=1, column=0, sticky="nsew")
        monitor_frame.rowconfigure(1, weight=1)
        
        log_label = ttk.Label(log_frame, text="📝 Activity Log", style="Header.TLabel")
        log_label.grid(row=0, column=0, pady=(0, 10), sticky="w")
        
        # Create a frame for log with clear button
        log_container = ttk.Frame(log_frame, style="Tab.TFrame")
        log_container.grid(row=1, column=0, sticky="nsew")
        log_container.rowconfigure(0, weight=1)
        log_container.columnconfigure(0, weight=1)
        
        self.log_text = tk.Text(log_container, height=12, bg="#2b2b2b", fg="#ffffff",
                               font=("Consolas", 9), wrap=tk.WORD)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        
        log_scroll = ttk.Scrollbar(log_container, orient="vertical",
                                  command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        log_scroll.grid(row=0, column=1, sticky="ns")
        
        # Clear log button
        ttk.Button(log_container, text="Clear Log", 
                  command=self._clear_log).grid(row=1, column=0, pady=(5, 0), sticky="e")
        
    def _create_control_panel(self, parent):
        """Create control panel (removed config button)"""
        control_frame = ttk.Frame(parent, style="Main.TFrame")
        control_frame.grid(row=2, column=0, pady=(10, 0), sticky="ew")
        
        # Status label
        self.status_label = ttk.Label(control_frame, text="⚪ Ready to start", 
                                     style="Status.TLabel")
        self.status_label.pack(side="left", padx=(0, 20))
        
        # Control buttons
        button_frame = ttk.Frame(control_frame, style="Main.TFrame")
        button_frame.pack(side="right")
        
        self.start_btn = ttk.Button(button_frame, text="▶️ Start Bots", 
                                    command=self._start_bots, style="Action.TButton")
        self.start_btn.pack(side="left", padx=5)
        
        self.stop_btn = ttk.Button(button_frame, text="⏹️ Stop All", 
                                   command=self._stop_bots, state="disabled",
                                   style="Action.TButton")
        self.stop_btn.pack(side="left", padx=5)
        
    def _update_preset_list(self):
        """Update the preset listbox"""
        self.preset_listbox.delete(0, tk.END)
        for preset in self.monster_presets:
            display_text = preset.name
            if preset.description:
                display_text += f" - {preset.description}"
            self.preset_listbox.insert(tk.END, display_text)
    
    def _update_monster_lists(self):
        """Update the monster listboxes"""
        # Update available monsters
        self.available_listbox.delete(0, tk.END)
        for monster in self.available_monsters:
            if monster not in self.selected_monsters:
                self.available_listbox.insert(tk.END, monster)
        
        # Update selected monsters
        self.selected_listbox.delete(0, tk.END)
        for monster in self.selected_monsters:
            self.selected_listbox.insert(tk.END, monster)
    
    def _filter_monsters(self, event=None):
        """Filter monsters based on search"""
        search_term = self.search_var.get().lower()
        
        self.available_listbox.delete(0, tk.END)
        for monster in self.available_monsters:
            if monster not in self.selected_monsters:
                if search_term == "" or search_term in monster.lower():
                    self.available_listbox.insert(tk.END, monster)
    
    def _add_monsters(self):
        """Add selected monsters to the selected list"""
        selected = self.available_listbox.curselection()
        for idx in selected:
            monster = self.available_listbox.get(idx)
            if monster not in self.selected_monsters:
                self.selected_monsters.append(monster)
        self._update_monster_lists()
    
    def _remove_monsters(self):
        """Remove selected monsters from the selected list"""
        selected = self.selected_listbox.curselection()
        monsters_to_remove = [self.selected_listbox.get(idx) for idx in selected]
        for monster in monsters_to_remove:
            self.selected_monsters.remove(monster)
        self._update_monster_lists()
    
    def _add_all_monsters(self):
        """Add all available monsters to selected list"""
        for monster in self.available_monsters:
            if monster not in self.selected_monsters:
                self.selected_monsters.append(monster)
        self._update_monster_lists()
    
    def _remove_all_monsters(self):
        """Remove all monsters from selected list"""
        self.selected_monsters.clear()
        self._update_monster_lists()
    
    def _reset_monsters(self):
        """Reset to default monster selection"""
        self.selected_monsters = self.available_monsters.copy()
        self._update_monster_lists()
    
    def _load_preset(self):
        """Load selected preset"""
        selected = self.preset_listbox.curselection()
        if selected:
            preset = self.monster_presets[selected[0]]
            self.selected_monsters = preset.monsters.copy()
            self._update_monster_lists()
            self.log_queue.put(('log', f"Loaded preset: {preset.name}"))
    
    def _save_preset(self):
        """Save current selection as preset"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Save Preset")
        dialog.geometry("400x200")
        dialog.transient(self.root)
        
        ttk.Label(dialog, text="Preset Name:").pack(pady=10)
        name_entry = ttk.Entry(dialog, width=40)
        name_entry.pack(pady=5)
        
        ttk.Label(dialog, text="Description (optional):").pack(pady=10)
        desc_entry = ttk.Entry(dialog, width=40)
        desc_entry.pack(pady=5)
        
        def save():
            name = name_entry.get().strip()
            if name:
                preset = MonsterPreset(name, self.selected_monsters.copy(), 
                                     desc_entry.get().strip())
                self.monster_presets.append(preset)
                self._save_presets()
                self._update_preset_list()
                self.log_queue.put(('log', f"Saved preset: {name}"))
                dialog.destroy()
        
        ttk.Button(dialog, text="Save", command=save).pack(pady=20)
    
    def _delete_preset(self):
        """Delete selected preset"""
        selected = self.preset_listbox.curselection()
        if selected:
            preset = self.monster_presets[selected[0]]
            if messagebox.askyesno("Delete Preset", f"Delete preset '{preset.name}'?"):
                self.monster_presets.remove(preset)
                self._save_presets()
                self._update_preset_list()
                self.log_queue.put(('log', f"Deleted preset: {preset.name}"))
    
    def _load_available_models(self):
        """Load available YOLOv8 models"""
        models = glob.glob("*.pt")
        if models:
            self.model_combo['values'] = models
            self.model_combo.current(0)
        else:
            self.model_combo['values'] = ["No models found"]
            
    def _browse_model(self):
        """Browse for model file"""
        filename = filedialog.askopenfilename(
            title="Select YOLOv8 Model",
            filetypes=[("PyTorch Model", "*.pt"), ("All Files", "*.*")]
        )
        if filename:
            self.model_var.set(Path(filename).name)
            
    def _refresh_windows(self):
        """Refresh available game windows"""
        self.window_listbox.delete(0, tk.END)
        windows = self._find_game_windows()
        
        for hwnd in windows:
            try:
                title = win32gui.GetWindowText(hwnd)
                self.window_listbox.insert(tk.END, f"HWND {hwnd}: {title}")
            except:
                pass
                
    def _find_game_windows(self) -> List[int]:
        """Find all game windows"""
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
        
    def _get_key_code(self, key_str):
        """Convert key string to win32con key code"""
        key_map = {
            "F1": win32con.VK_F1,
            "F2": win32con.VK_F2,
            "F3": win32con.VK_F3,
            "F4": win32con.VK_F4,
            "F5": win32con.VK_F5,
            "F6": win32con.VK_F6,
            "F7": win32con.VK_F7,
            "F8": win32con.VK_F8
        }
        return key_map.get(key_str, win32con.VK_F1)
        
    def _start_bots(self):
        """Start selected bots"""
        if not self.model_var.get() or self.model_var.get() == "No models found":
            messagebox.showerror("Error", "Please select a valid YOLOv8 model")
            return
            
        if not self.selected_monsters:
            messagebox.showerror("Error", "Please select at least one monster")
            return
            
        selected = self.window_listbox.curselection()
        if not selected:
            messagebox.showerror("Error", "Please select at least one game window")
            return
            
        # Update config with selected monsters
        config.MONSTER_CLASSES = self.selected_monsters.copy()
        
        # Update HP/MP/GP/Food settings
        config.HP_THRESHOLD = self.hp_threshold_var.get()
        config.MP_THRESHOLD = self.mp_threshold_var.get()
        config.GP_THRESHOLD = self.gp_threshold_var.get()
        config.HP_SHORTCUT = self._get_key_code(self.hp_key_var.get())
        config.MP_SHORTCUT = self._get_key_code(self.mp_key_var.get())
        config.GP_SHORTCUT = self._get_key_code(self.gp_key_var.get())
        config.FOOD_SHORTCUT = self._get_key_code(self.food_key_var.get())
        
        # Get configuration
        model_path = self.model_var.get()
        device = self.device_var.get()
        mode = self.mode_var.get()
        
        # Apply speed settings
        speed = self.speed_var.get()
        if speed == "fast":
            config.DETECTION_INTERVAL = 0.05
        elif speed == "normal":
            config.DETECTION_INTERVAL = 0.08
        else:
            config.DETECTION_INTERVAL = 0.12
            
        # Apply YOLO settings
        config.YOLO_CONFIDENCE_THRESHOLD = self.confidence_var.get()
        
        # Update config
        config.ENABLE_FOOD_CHECK = self.food_var.get()
        config.ENABLE_GP_CHECK = self.gp_var.get()
        
        self.is_running = True
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status_label.configure(text="🟢 Bots Running")
        
        # Start bots
        windows = self._find_game_windows()
        for idx in selected:
            hwnd = windows[idx]
            self._start_single_bot(hwnd, model_path, device, mode)
            
        self._log(f"Started {len(selected)} bot(s) with {len(self.selected_monsters)} monster types")
        
    def _start_single_bot(self, hwnd, model_path, device, mode):
        """Start a single bot instance"""
        try:
            # Create bot controller
            bot_controller = BotController(
                hwnd, model_path, mode, 
                self.food_var.get(), self.gp_var.get(), device, self.log_queue
            )
            
            # Start the bot
            if bot_controller.start():
                self.bots[hwnd] = bot_controller
                
                # Add to tree
                title = win32gui.GetWindowText(hwnd)
                self.bot_tree.insert("", "end", iid=str(hwnd), values=(
                    f"Bot {len(self.bots)}", "Starting", "0", "0", device.upper()
                ))
            else:
                self._log(f"Failed to start bot for window {hwnd}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start bot: {e}")
            
    def _stop_bots(self):
        """Stop all bots"""
        self.is_running = False
        
        for bot_controller in self.bots.values():
            try:
                bot_controller.stop()
            except:
                pass
                
        self.bots.clear()
        self.bot_tree.delete(*self.bot_tree.get_children())
        
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.status_label.configure(text="⚪ Stopped")
        
        self._log("All bots stopped")
        
    def _log(self, message: str):
        """Add message to log"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        
        # Limit log size to prevent memory issues
        lines = int(self.log_text.index('end-1c').split('.')[0])
        if lines > 1000:
            self.log_text.delete('1.0', '100.0')
        
    def _clear_log(self):
        """Clear the log text"""
        self.log_text.delete('1.0', tk.END)
        
    def _update_gui(self):
        """Update GUI periodically"""
        # Process log queue
        try:
            while True:
                msg_type, msg = self.log_queue.get_nowait()
                if msg_type == 'log':
                    # Extract just the message part from the log
                    if ' - ' in msg:
                        msg = msg.split(' - ', 3)[-1]  # Get the last part after the last ' - '
                    self._log(msg)
        except queue.Empty:
            pass
        
        if self.is_running:
            # Update bot status
            for hwnd, bot_controller in self.bots.items():
                try:
                    stats = bot_controller.get_stats()
                    fps = stats.get('fps', 0)
                    targets = stats.get('targets_attacked', 0)
                    status = stats.get('status', 'Unknown')
                    
                    self.bot_tree.item(str(hwnd), values=(
                        f"Bot {list(self.bots.keys()).index(hwnd) + 1}",
                        status,
                        str(targets),
                        f"{fps:.1f}",
                        stats.get('device', 'N/A').upper()
                    ))
                except:
                    pass
                        
        self.root.after(100, self._update_gui)  # Update more frequently for logs
        
    def run(self):
        """Run the GUI"""
        try:
            self.root.mainloop()
        finally:
            # Clean up logging
            root_logger = logging.getLogger()
            if self.gui_handler in root_logger.handlers:
                root_logger.removeHandler(self.gui_handler)


def main():
    """Main entry point"""
    # Suppress default logging configuration
    logging.getLogger().handlers.clear()
    
    app = BotGUI()
    app.run()


if __name__ == "__main__":
    main()
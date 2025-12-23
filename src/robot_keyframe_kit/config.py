"""Configuration dataclass for the keyframe editor."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    import yaml
except ImportError:
    yaml = None


@dataclass
class EditorConfig:
    """Configuration options for the ViserKeyframeEditor.
    
    Attributes:
        root_body: Name of the root body used for ground alignment (e.g., "torso", "base_link").
            If None, will be auto-detected as the first non-world body that is a direct child of world.
        end_effector_sites: List of site names for end-effector tracking/alignment.
            If None, will attempt to auto-detect sites with common naming patterns.
        mirror_pairs: Dictionary mapping left joint names to right joint names for mirroring.
            If None, will attempt to auto-detect based on "left"/"right" naming convention.
        mirror_signs: Dictionary of joint names to sign multipliers for mirroring.
            Positive 1 means same direction, -1 means opposite direction.
        dt: Timestep for trajectory playback (seconds).
        save_dir: Directory to save keyframe data files.
        name: Optional name for this robot/project (used in save filenames).
    """
    root_body: Optional[str] = None
    end_effector_sites: Optional[List[str]] = None
    mirror_pairs: Optional[Dict[str, str]] = None
    mirror_signs: Optional[Dict[str, int]] = None
    dt: float = 0.02
    save_dir: str = "keyframes"
    name: str = "robot"
    
    # Physics simulation settings
    n_frames: int = 20  # Number of physics substeps per control step
    physics_dt: float = 0.001  # Physics timestep
    
    # PD control gains for trajectory playback (for motor/torque actuators)
    # These are used when actuators are motor-type (not position-type).
    # Toddlerbot uses kp_sim = 10-14 (after kp_ratio division from config values 1500-2100)
    # Note: Position-type actuators (like Unitree G1) use MuJoCo's built-in PD.
    kp: float = 12.0  # Position gain (proportional) - matches typical Dynamixel motors
    kd: float = 0.5   # Velocity gain (derivative) - light damping
    
    # UI settings
    show_com: bool = True  # Show center of mass marker
    show_grid: bool = True  # Show ground grid
    
    # Scene generation settings
    auto_inject_floor: bool = True  # Whether to auto-inject floor for robot-only XMLs
    show_floor: bool = True  # Whether the injected floor should be visible
    
    @classmethod
    def from_yaml(cls, path: str) -> "EditorConfig":
        """Load configuration from a YAML file.
        
        Args:
            path: Path to the YAML configuration file.
            
        Returns:
            EditorConfig instance loaded from the file.
            
        Raises:
            ImportError: If PyYAML is not installed.
            FileNotFoundError: If the config file doesn't exist.
        """
        if yaml is None:
            raise ImportError(
                "PyYAML is required to load YAML config files. "
                "Install with: pip install pyyaml"
            )
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        
        if data is None:
            data = {}
        
        # Map YAML keys to config fields
        mapped_data = {}
        
        # Direct mappings
        for key in ["name", "root_body", "dt", "save_dir", "kp", "kd", "n_frames", "physics_dt", "show_com", "show_grid", "auto_inject_floor", "show_floor"]:
            if key in data:
                mapped_data[key] = data[key]
        
        # Handle end_effectors -> end_effector_sites
        if "end_effectors" in data:
            mapped_data["end_effector_sites"] = data["end_effectors"]
        elif "end_effector_sites" in data:
            mapped_data["end_effector_sites"] = data["end_effector_sites"]
        
        # Handle mirror_pairs and mirror_signs
        if "mirror_pairs" in data:
            mapped_data["mirror_pairs"] = data["mirror_pairs"]
        if "mirror_signs" in data:
            mapped_data["mirror_signs"] = data["mirror_signs"]
        
        # Extract nested physics settings (override direct settings)
        physics = data.get("physics", {})
        if isinstance(physics, dict):
            for key in ["kp", "kd", "dt"]:
                if key in physics:
                    mapped_data[key] = physics[key]
        
        # Extract nested UI settings
        ui = data.get("ui", {})
        if isinstance(ui, dict):
            for key in ["show_com", "show_grid"]:
                if key in ui:
                    mapped_data[key] = ui[key]
        
        return cls(**mapped_data)
    
    def to_yaml(self, path: str) -> None:
        """Save configuration to a YAML file.
        
        Args:
            path: Path where to save the YAML configuration file.
            
        Raises:
            ImportError: If PyYAML is not installed.
        """
        if yaml is None:
            raise ImportError(
                "PyYAML is required to save YAML config files. "
                "Install with: pip install pyyaml"
            )
        
        data = {
            "name": self.name,
            "root_body": self.root_body,
            "end_effectors": self.end_effector_sites,
            "mirror_pairs": self.mirror_pairs,
            "mirror_signs": self.mirror_signs,
            "dt": self.dt,
            "save_dir": self.save_dir,
            "physics": {
                "kp": self.kp,
                "kd": self.kd,
                "dt": self.dt,
            },
            "ui": {
                "show_com": self.show_com,
                "show_grid": self.show_grid,
            },
            "scene": {
                "auto_inject_floor": self.auto_inject_floor,
                "show_floor": self.show_floor,
            },
        }
        
        # Remove None values
        data = {k: v for k, v in data.items() if v is not None}
        
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    @classmethod
    def generate_from_model(cls, xml_path: str, name: Optional[str] = None) -> "EditorConfig":
        """Generate a configuration file from a MuJoCo model by auto-detecting settings.
        
        This creates a config with auto-detected values that can be manually edited.
        
        Args:
            xml_path: Path to the MuJoCo XML file.
            name: Optional name for the robot. If None, inferred from XML filename.
            
        Returns:
            EditorConfig instance with auto-detected values.
        """
        import mujoco
        
        model = mujoco.MjModel.from_xml_path(xml_path)
        
        # Infer name from XML path if not provided
        if name is None:
            name = os.path.splitext(os.path.basename(xml_path))[0]
        
        # Auto-detect root body
        root_body = None
        for body_id in range(model.nbody):
            if body_id == 0:  # Skip world body
                continue
            if model.body_parentid[body_id] == 0:  # Direct child of world
                root_body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                if root_body and root_body != "world":
                    break
        
        # Auto-detect end-effector sites (from leaf bodies)
        parent_ids = set(model.body_parentid)
        leaf_body_ids = [bid for bid in range(model.nbody) if bid not in parent_ids]
        
        end_effectors = []
        ee_keywords = ["foot", "hand", "calf", "leg", "lleg", "ankle", "toe", "gripper"]
        
        # First try sites
        for body_id in leaf_body_ids:
            for site_id in range(model.nsite):
                if model.site_bodyid[site_id] == body_id:
                    site_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, site_id)
                    if site_name:
                        end_effectors.append(site_name)
        
        # Fallback to leaf bodies
        if not end_effectors:
            for body_id in leaf_body_ids:
                body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                if body_name and body_name != "world":
                    body_lower = body_name.lower()
                    if any(kw in body_lower for kw in ee_keywords):
                        end_effectors.append(body_name)
        
        # Auto-detect mirror pairs (left/right joints)
        mirror_pairs = {}
        joint_names = []
        for jnt_id in range(model.njnt):
            jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
            if jnt_name:
                joint_names.append(jnt_name)
        
        # Find left/right pairs
        for joint_name in joint_names:
            if "left" in joint_name.lower() or "_l_" in joint_name.lower():
                # Try to find corresponding right joint
                right_name = joint_name.replace("left", "right").replace("Left", "Right")
                right_name_alt = joint_name.replace("_l_", "_r_").replace("_L_", "_R_")
                
                if right_name in joint_names:
                    mirror_pairs[joint_name] = right_name
                elif right_name_alt in joint_names:
                    mirror_pairs[joint_name] = right_name_alt
        
        return cls(
            name=name,
            root_body=root_body,
            end_effector_sites=end_effectors if end_effectors else None,
            mirror_pairs=mirror_pairs if mirror_pairs else None,
            mirror_signs=None,  # Will be auto-computed by editor
        )



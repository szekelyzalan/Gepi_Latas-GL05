"""
Store the classes / labels that should be recognized.
"""

LABEL_MAP: dict[str, str] = {
    "regulatory--yield--g1": "yield",
    "regulatory--no-entry--g1": "no_entry",
    "regulatory--stop--g1": "stop",
    "information--pedestrians-crossing--g1": "pedestrian_crossing",
    "warning--pedestrians-crossing--g1": "pedestrian_crossing",
    "warning--pedestrians-crossing--g5": "pedestrian_crossing",
    "regulatory--maximum-speed-limit-70--g1": "speed_limit",
    "regulatory--maximum-speed-limit-40--g1": "speed_limit",
    "regulatory--maximum-speed-limit-50--g1": "speed_limit",
    "regulatory--maximum-speed-limit-60--g1": "speed_limit",
    "regulatory--maximum-speed-limit-30--g1": "speed_limit",
    "regulatory--maximum-speed-limit-80--g1": "speed_limit",
    "regulatory--maximum-speed-limit-90--g1": "speed_limit",
    "regulatory--maximum-speed-limit-110--g1": "speed_limit",
    "regulatory--maximum-speed-limit-led-100--g1": "speed_limit",
    "regulatory--maximum-speed-limit-100--g1": "speed_limit",
    "regulatory--maximum-speed-limit-10--g1": "speed_limit",
    "regulatory--maximum-speed-limit-20--g1": "speed_limit",
    "regulatory--maximum-speed-limit-led-60--g1": "speed_limit",
    "regulatory--maximum-speed-limit-led-80--g1": "speed_limit",
    "regulatory--maximum-speed-limit-15--g1": "speed_limit",
    "regulatory--maximum-speed-limit-120--g1": "speed_limit"
}

CLASS_COUNTS : dict[str, int]= {
    "speed_limit": 216,
    "pedestrian_crossing": 177,
    "yield": 142,
    "no_entry": 72,
    "stop": 55
}

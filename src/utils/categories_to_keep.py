"""
Store the classes / labels that should be recognized.
"""
# Include end of maximum speed limit?
CLASS_COUNTS : dict[str, int]= {
    "information--pedestrians-crossing--g1": 152,
    "regulatory--yield--g1": 142,
    "regulatory--no-entry--g1": 72,
    "regulatory--stop--g1": 55,
    "regulatory--end-of-maximum-speed-limit-70--g2": 42,
    "regulatory--maximum-speed-limit-70--g1": 34,
    "regulatory--maximum-speed-limit-40--g1": 23,
    "regulatory--maximum-speed-limit-50--g1": 23,
    "warning--pedestrians-crossing--g1": 23,
    "regulatory--maximum-speed-limit-60--g1": 20,
    "regulatory--maximum-speed-limit-30--g1": 20,
    "regulatory--maximum-speed-limit-80--g1": 18,
    "regulatory--maximum-speed-limit-90--g1": 15,
    "regulatory--maximum-speed-limit-110--g1": 11,
    "regulatory--maximum-speed-limit-led-100--g1": 11,
    "regulatory--maximum-speed-limit-100--g1": 11,
    "regulatory--maximum-speed-limit-10--g1": 10,
    "regulatory--maximum-speed-limit-20--g1": 6,
    "regulatory--maximum-speed-limit-led-60--g1": 6,
    "regulatory--maximum-speed-limit-led-80--g1": 4,
    "regulatory--maximum-speed-limit-15--g1": 3,
    "regulatory--end-of-maximum-speed-limit-30--g2": 3,
    "warning--pedestrians-crossing--g5": 2,
    "regulatory--maximum-speed-limit-120--g1": 1,
    "regulatory--end-of-maximum-speed-limit-70--g1": 1,
}

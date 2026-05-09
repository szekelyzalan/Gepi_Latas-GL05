"""
This dictionary contains the info to change some labels to different ones.
It also contains the folder from where to change the images (GTSRB database or clean_signs).
"""
from pathlib import Path

CHANGE_BUNDLE: dict[str, Path] = {
    "regulatory--maximum-speed-limit-70--g1": Path("clean_signs/regulatory--maximum-speed-limit-70--g1"),
    "regulatory--maximum-speed-limit-50--g1": Path("clean_signs/regulatory--maximum-speed-limit-50--g1"),
    "regulatory--maximum-speed-limit-60--g1": Path("clean_signs/regulatory--maximum-speed-limit-60--g1"),
    "regulatory--maximum-speed-limit-30--g1": Path("clean_signs/regulatory--maximum-speed-limit-30--g1"),
    "regulatory--maximum-speed-limit-80--g1": Path("clean_signs/regulatory--maximum-speed-limit-80--g1"),
    "regulatory--maximum-speed-limit-100--g1": Path("clean_signs/regulatory--maximum-speed-limit-100--g1"),
    "regulatory--maximum-speed-limit-20--g1": Path("clean_signs/regulatory--maximum-speed-limit-20--g1"),
    "regulatory--maximum-speed-limit-120--g1": Path("clean_signs/regulatory--maximum-speed-limit-120--g1"),
    "regulatory--no-entry--g1": Path("clean_signs/regulatory--no-entry--g1"),
    "warning--pedestrians-crossing--g1": Path("clean_signs/warning--pedestrians-crossing--g1"),
    "information--pedestrians-crossing--g1": Path("clean_signs/information--pedestrians-crossing--g1"),
    "regulatory--stop--g1": Path("clean_signs/regulatory--stop--g1"),
    "regulatory--yield--g1": Path("clean_signs/regulatory--yield--g1"),
}

SEMANTIC_CLASS_MAP = {
    "regulatory--maximum-speed-limit-20--g1": "speed-limit",
    "regulatory--maximum-speed-limit-30--g1": "speed-limit",
    "regulatory--maximum-speed-limit-50--g1": "speed-limit",
    "regulatory--maximum-speed-limit-60--g1": "speed-limit",
    "regulatory--maximum-speed-limit-70--g1": "speed-limit",
    "regulatory--maximum-speed-limit-80--g1": "speed-limit",
    "regulatory--maximum-speed-limit-100--g1": "speed-limit",
    "regulatory--maximum-speed-limit-120--g1": "speed-limit",

    "regulatory--stop--g1": "stop",

    "regulatory--yield--g1": "yield",

    "regulatory--no-entry--g1": "no-entry",

    "warning--pedestrians-crossing--g1": "pedestrian",
    "information--pedestrians-crossing--g1": "pedestrian",
}

WEIGHTS = {
    "speed-limit": 0.9,
    "stop": 1.3,
    "yield": 1.1,
    "pedestrian": 1.2,
    "no-entry": 1.2
}

CHANGE_BUNDLE_OLD: dict[str, Path] = {
    "regulatory--maximum-speed-limit-70--g1": Path("GTSRB/Train/4"),
    "regulatory--maximum-speed-limit-50--g1": Path("GTSRB/Train/2"),
    "regulatory--maximum-speed-limit-60--g1": Path("GTSRB/Train/3"),
    "regulatory--maximum-speed-limit-30--g1": Path("GTSRB/Train/1"),
    "regulatory--maximum-speed-limit-80--g1": Path("GTSRB/Train/5"),
    "regulatory--maximum-speed-limit-100--g1": Path("GTSRB/Train/7"),
    "regulatory--maximum-speed-limit-20--g1": Path("GTSRB/Train/0"),
    "regulatory--maximum-speed-limit-120--g1": Path("GTSRB/Train/8"),
    "regulatory--no-entry--g1": Path("GTSRB/Train/17"),
    "warning--pedestrians-crossing--g1": Path("GTSRB/Train/20"),
    "information--pedestrians-crossing--g1": Path("GTSRB/Train/20"),
    "regulatory--stop--g1": Path("GTSRB/Train/14"),
    "regulatory--yield--g1": Path("GTSRB/Train/13"),
}

TO_CHANGE: list = [
    "regulatory--no-stopping--g15",
    "regulatory--keep-right--g1",
    "regulatory--priority-road--g4",
    "information--parking--g1",
    "regulatory--go-straight--g1",
    "regulatory--no-overtaking--g1",
    "warning--other-danger--g1",
    "regulatory--end-of-maximum-speed-limit-70--g2",
    "warning--railroad-crossing-without-barriers--g3",
    "warning--slippery-road-surface--g1",
    "information--tram-bus-stop--g2",
    "regulatory--turn-right--g1",
    "regulatory--road-closed-to-vehicles--g3",
    "regulatory--roundabout--g1",
    "complementary--chevron-right--g5",
    "information--motorway--g1",
    "warning--traffic-signals--g1",
    "complementary--distance--g1",
    "regulatory--road-closed-to-vehicles--g3",
    "regulatory--pass-on-either-side--g1",
    "regulatory--no-parking--g1",
    "regulatory--one-way-straight--g1",
    "regulatory--end-of-priority-road--g1",
    "complementary--distance--g2",
    "warning--children--g1",
    "complementary--chevron-left--g5",
    "warning--junction-with-a-side-road-perpendicular-right--g1",
    "warning--uneven-road--g6",
    "warning--wild-animals--g1",
    "regulatory--weight-limit--g1",
    "warning--double-curve-first-right--g1",
    "regulatory--height-limit--g1",
    "information--end-of-built-up-area--g1",
    "regulatory--no-left-turn--g1",
    "complementary--chevron-right-unsure--g6",
    "regulatory--go-straight-or-turn-left--g1",
    "complementary--chevron-right--g1",
    "complementary--accident-area--g3",
    "regulatory--bicycles-only--g1",
    "information--end-of-motorway--g1",
    "complementary--go-right--g1",
    "regulatory--turn-left--g1",
    "warning--road-bump--g1",
    "warning--roadworks--g1",
    "warning--road-narrows--g1",
    "warning--curve-left--g1",
    "regulatory--maximum-speed-limit-100--g1",
    "warning--traffic-merges-right--g2",
    "warning--railroad-crossing-without-barriers--g1",
    "warning--double-curve-first-left--g1",
    "regulatory--no-right-turn--g1",
    "warning--railroad-crossing-with-barriers--g1",
    "regulatory--go-straight-or-turn-right--g1",
    "complementary--priority-route-at-intersection--g1",
    "complementary--trucks--g1",
    "regulatory--turn-right-ahead--g1",
    "complementary--chevron-right--g3",
    "regulatory--dual-path-bicycles-and-pedestrians--g1",
    "regulatory--no-vehicles-carrying-dangerous-goods--g1",
    "regulatory--no-overtaking-by-heavy-goods-vehicles--g1",
    "information--food--g2",
    "complementary--go-left--g1",
    "regulatory--priority-over-oncoming-vehicles--g1",
    "regulatory--shared-path-pedestrians-and-bicycles--g1",
    "information--end-of-pedestrians-only--g2",
    "regulatory--end-of-bicycles-only--g1",
    "information--limited-access-road--g1",
    "warning--roundabout--g1",
    "warning--curve-right--g1",
    "regulatory--radar-enforced--g1",
    "regulatory--one-way-left--g1",
    "warning--railroad-crossing--g4",
    "warning--crossroads--g1",
    "information--parking--g5",
    "information--dead-end--g1",
]


OLD_SIGN_SHAPES = {

    # =====================================================
    # CIRCLE
    # =====================================================
    "regulatory--no-stopping--g15": "circle",
    "regulatory--keep-right--g1": "circle",
    "regulatory--go-straight--g1": "circle",
    "regulatory--no-overtaking--g1": "circle",
    "regulatory--end-of-maximum-speed-limit-70--g2": "circle",
    "regulatory--turn-right--g1": "circle",
    "regulatory--pass-on-either-side--g1": "circle",
    "regulatory--no-parking--g1": "circle",
    "regulatory--weight-limit--g1": "circle",
    "regulatory--height-limit--g1": "circle",
    "regulatory--no-left-turn--g1": "circle",
    "regulatory--bicycles-only--g1": "circle",
    "regulatory--turn-left--g1": "circle",
    "regulatory--maximum-speed-limit-100--g1": "circle",
    "regulatory--no-right-turn--g1": "circle",
    "regulatory--go-straight-or-turn-right--g1": "circle",
    "regulatory--go-straight-or-turn-left--g1": "circle",
    "regulatory--roundabout--g1": "circle",

    # newly added circular signs
    "regulatory--turn-right-ahead--g1": "circle",
    "regulatory--dual-path-bicycles-and-pedestrians--g1": "circle",
    "regulatory--no-vehicles-carrying-dangerous-goods--g1": "circle",
    "regulatory--no-overtaking-by-heavy-goods-vehicles--g1": "circle",
    "regulatory--priority-over-oncoming-vehicles--g1": "circle",
    "regulatory--shared-path-pedestrians-and-bicycles--g1": "circle",
    "regulatory--end-of-bicycles-only--g1": "circle",
    "regulatory--radar-enforced--g1": "circle",

    # =====================================================
    # TRIANGLE
    # =====================================================
    "warning--other-danger--g1": "triangle",
    "warning--slippery-road-surface--g1": "triangle",
    "warning--children--g1": "triangle",
    "warning--junction-with-a-side-road-perpendicular-right--g1": "triangle",
    "warning--uneven-road--g6": "triangle",
    "warning--wild-animals--g1": "triangle",
    "warning--double-curve-first-right--g1": "triangle",
    "warning--road-bump--g1": "triangle",
    "warning--roadworks--g1": "triangle",
    "warning--road-narrows--g1": "triangle",
    "warning--curve-left--g1": "triangle",
    "warning--traffic-merges-right--g2": "triangle",
    "warning--railroad-crossing-without-barriers--g1": "triangle",
    "warning--railroad-crossing-without-barriers--g3": "triangle",
    "warning--double-curve-first-left--g1": "triangle",
    "warning--railroad-crossing-with-barriers--g1": "triangle",
    "warning--traffic-signals--g1": "triangle",

    # newly added warning triangles
    "warning--roundabout--g1": "triangle",
    "warning--curve-right--g1": "triangle",
    "warning--railroad-crossing--g4": "triangle",
    "warning--crossroads--g1": "triangle",

    # =====================================================
    # RECTANGLE / SQUARE
    # =====================================================
    "information--parking--g1": "rectangle",
    "information--tram-bus-stop--g2": "rectangle",
    "information--motorway--g1": "rectangle",
    "information--end-of-built-up-area--g1": "rectangle",
    "information--end-of-motorway--g1": "rectangle",

    "complementary--distance--g1": "rectangle",
    "complementary--distance--g2": "rectangle",
    "complementary--chevron-right--g5": "rectangle",
    "complementary--chevron-left--g5": "rectangle",
    "complementary--chevron-right-unsure--g6": "rectangle",
    "complementary--chevron-right--g1": "rectangle",
    "complementary--accident-area--g3": "rectangle",
    "complementary--go-right--g1": "rectangle",

    # fallback rectangles
    "regulatory--road-closed-to-vehicles--g3": "rectangle",
    "regulatory--one-way-straight--g1": "rectangle",

    # newly added rectangles
    "complementary--priority-route-at-intersection--g1": "rectangle",
    "complementary--trucks--g1": "rectangle",
    "complementary--chevron-right--g3": "rectangle",
    "information--food--g2": "rectangle",
    "complementary--go-left--g1": "rectangle",
    "information--end-of-pedestrians-only--g2": "rectangle",
    "information--limited-access-road--g1": "rectangle",
    "regulatory--one-way-left--g1": "rectangle",
    "information--parking--g5": "rectangle",
    "information--dead-end--g1": "rectangle",

    # =====================================================
    # DIAMOND
    # =====================================================
    "regulatory--priority-road--g4": "diamond",
    "regulatory--end-of-priority-road--g1": "diamond",
}

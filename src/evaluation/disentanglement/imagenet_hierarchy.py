"""ImageNet-1K class hierarchy via WordNet.

Builds sibling groups: classes that share the same WordNet parent synset
(e.g., all terrier breeds, all snake species, all big cats).

Falls back to a hardcoded grouping if NLTK/WordNet is unavailable.
"""

import json
import os
from collections import defaultdict

_CLASS_INDEX_PATH = os.path.join(os.path.dirname(__file__), "imagenet_class_index.json")


def _load_class_index() -> dict[int, tuple[str, str]]:
    """Load ImageNet class index → (WNID, readable_name).

    Returns dict mapping int class index to (wnid, name) tuple.
    """
    with open(_CLASS_INDEX_PATH) as f:
        data = json.load(f)
    return {int(k): (v[0], v[1]) for k, v in data.items()}


def build_sibling_groups(min_group_size: int = 3) -> dict[str, list[int]]:
    """Build groups of ImageNet classes sharing the same WordNet parent.

    Uses NLTK's WordNet to find the immediate hypernym (parent synset)
    for each of the 1000 ImageNet classes, then groups classes by parent.

    Args:
        min_group_size: Minimum number of sibling classes to form a group.

    Returns:
        Dict mapping parent synset name → list of ImageNet class indices.
        Example: {"terrier.n.01": [181, 182, 183, ...], ...}
    """
    class_index = _load_class_index()

    try:
        import nltk
        nltk.download("wordnet", quiet=True)
        from nltk.corpus import wordnet as wn

        groups = defaultdict(list)
        for idx, (wnid, _name) in class_index.items():
            offset = int(wnid[1:])
            try:
                syn = wn.synset_from_pos_and_offset("n", offset)
                hypernyms = syn.hypernyms()
                if hypernyms:
                    parent = hypernyms[0].name()
                    groups[parent].append(idx)
            except Exception:
                continue

        filtered = {k: sorted(v) for k, v in groups.items() if len(v) >= min_group_size}
        print(f"[hierarchy] WordNet: {len(filtered)} sibling groups "
              f"({sum(len(v) for v in filtered.values())} classes)")
        return filtered

    except (ImportError, LookupError) as e:
        print(f"[hierarchy] WordNet unavailable ({e}), using hardcoded groups")
        return _hardcoded_groups(min_group_size)


def _hardcoded_groups(min_group_size: int) -> dict[str, list[int]]:
    """Fallback: manually curated ImageNet superclass groups."""
    groups = {
        "terrier": list(range(181, 199)),
        "hound": [160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171],
        "shepherd_dog": [226, 227, 228, 229, 230, 231, 232, 233, 234, 235],
        "toy_dog": [151, 152, 153, 154, 155, 156],
        "sporting_dog": [206, 207, 208, 209, 210, 211, 212, 213, 214, 215],
        "working_dog": [243, 244, 245, 246, 247, 248, 249, 250, 251, 252],
        "big_cat": [288, 289, 290, 291, 292, 293],
        "bear": [294, 295, 296, 297],
        "primate": [365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385],
        "snake": list(range(52, 69)),
        "spider": [72, 73, 74, 75, 76, 77],
        "beetle": [300, 301, 302, 303, 304, 305, 306],
        "butterfly": [320, 321, 322, 323, 324],
        "fish": [0, 1, 2, 389, 390, 391, 392, 393, 394, 395, 396, 397],
        "bird_wading": [129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145],
        "vehicle_wheeled": [407, 436, 468, 511, 609, 627, 654, 656, 661, 671, 675, 705, 717, 734, 751, 779, 817, 864],
        "boat": [427, 435, 463, 472, 484, 554, 625],
        "furniture_seating": [423, 559, 765, 831, 857],
        "musical_instrument": [401, 402, 420, 431, 432, 486, 494, 513, 541, 546, 558, 566, 579, 593, 642, 687, 776, 822, 875, 889],
        "food_fruit": [948, 949, 950, 951, 952, 953, 954, 955, 956, 957],
        "food_vegetable": [937, 938, 939, 940, 941, 942, 943, 944, 945, 946, 947],
        "kitchen_utensil": [462, 499, 567, 700, 868, 910, 963],
        "container_bottle": [440, 720, 737, 898, 907],
        "ball": [429, 430, 522, 574, 722, 747, 768, 805],
        "screen_display": [527, 664, 681, 782, 851],
        "clothing": [474, 514, 617, 638, 639, 640, 689, 834],
    }
    filtered = {k: sorted(v) for k, v in groups.items() if len(v) >= min_group_size}
    print(f"[hierarchy] Hardcoded: {len(filtered)} groups "
          f"({sum(len(v) for v in filtered.values())} classes)")
    return filtered

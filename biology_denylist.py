"""
biology_denylist.py — Layer 0 biological-residue check for the Abstract step.

A design strategy should be biology-free. This module flags biological vocabulary
that survived abstraction. It is deterministic and free; run it BEFORE any LLM
residual-judge call.

Three tiers:
  HARD       : essentially always biological in a design context -> auto-flag.
  AMBIGUOUS  : has a common engineering/everyday meaning -> DON'T fail; escalate
               to the LLM residual-residue judge for a contextual call.
  ALLOWLIST  : biology-adjacent terms AskNature endorses as neutral replacements
               (e.g. membrane for skin, fiber for fur) -> never flag; overrides.

Dynamic injection: pass the source strategy's organism name(s) (common + Latin)
via `source_organism_terms` — the most frequent residue is the specific species.

Coverage note: AskNature spans thousands of organisms; the taxa list below covers
high-frequency names only. Rely on dynamic injection for the long tail, and treat
this file as a living artifact — append terms you observe leaking in production.

Usage:
    from biology_denylist import check_biology_residue
    r = check_biology_residue(design_strategy, source_organism_terms=["polar bear", "Ursus maritimus"])
    # r.flag      -> True if any HARD hit (definite residue)
    # r.escalate  -> True if any AMBIGUOUS hit (send to LLM judge)
    # r.hard_hits / r.ambiguous_hits -> the matched terms
"""

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# HARD — unambiguous biology. A hit is residue.
# ---------------------------------------------------------------------------
HARD = {
    # --- Taxa: high-level groups ---
    "organism", "creature", "animal", "plant", "species", "specimen",
    "insect", "mammal", "bird", "avian", "fish", "reptile", "amphibian",
    "invertebrate", "vertebrate", "arthropod", "arachnid", "crustacean",
    "mollusk", "mollusc", "cephalopod", "gastropod", "bivalve",
    "bacterium", "bacteria", "bacterial", "microbe", "microbial", "microorganism",
    "fungus", "fungi", "fungal", "virus", "viral", "protozoa", "protozoan",
    "algae", "alga", "algal", "plankton", "phytoplankton", "diatom", "amoeba",

    # --- Taxa: common animals (AskNature-frequent) ---
    "beetle", "ant", "bee", "wasp", "hornet", "termite", "butterfly", "moth",
    "spider", "scorpion", "mosquito", "dragonfly", "damselfly", "cicada",
    "locust", "cricket", "grasshopper", "mantis", "aphid", "flea", "tick",
    "centipede", "millipede", "earthworm", "snail", "slug", "clam", "oyster",
    "mussel", "scallop", "squid", "octopus", "cuttlefish", "nautilus",
    "jellyfish", "coral", "sponge", "anemone", "starfish", "urchin",
    "crab", "lobster", "shrimp", "krill", "barnacle", "copepod",
    "shark", "stingray", "salmon", "tuna", "trout", "minnow", "pufferfish",
    "seahorse", "frog", "toad", "salamander", "newt", "tadpole",
    "lizard", "gecko", "chameleon", "iguana", "snake", "serpent", "python",
    "turtle", "tortoise", "crocodile", "alligator",
    "eagle", "hawk", "falcon", "owl", "penguin", "pelican", "duck", "goose",
    "swan", "heron", "stork", "hummingbird", "woodpecker", "kingfisher",
    "pigeon", "sparrow", "finch", "robin", "raven", "crow", "peacock",
    "ostrich", "emu", "flamingo", "albatross", "puffin",
    "whale", "dolphin", "porpoise", "seal", "walrus", "otter", "manatee",
    "fox", "wolf", "coyote", "lion", "tiger", "leopard", "cheetah", "jaguar",
    "lynx", "cougar", "elephant", "rhinoceros", "hippopotamus", "giraffe",
    "zebra", "antelope", "gazelle", "wildebeest", "buffalo", "bison",
    "camel", "llama", "alpaca", "kangaroo", "wallaby", "koala", "wombat",
    "platypus", "sloth", "anteater", "armadillo", "pangolin", "hedgehog",
    "porcupine", "squirrel", "chipmunk", "beaver", "hamster", "gerbil",
    "rabbit", "hare", "mole", "shrew", "weasel", "ferret", "badger",
    "raccoon", "opossum", "primate", "monkey", "baboon", "gorilla",
    "chimpanzee", "orangutan", "lemur", "mongoose", "meerkat",

    # --- Taxa: common plants/fungi ---
    "moss", "fern", "lichen", "mushroom", "toadstool", "mold", "mildew",
    "yeast", "cactus", "succulent", "lotus", "lily", "orchid", "tulip",
    "daisy", "dandelion", "sunflower", "thistle", "clover", "ivy",
    "oak", "pine", "fir", "spruce", "cedar", "maple", "birch", "aspen",
    "willow", "redwood", "sequoia", "bamboo", "mangrove", "seaweed", "kelp",
    "sapling", "seedling",

    # --- Animal anatomy / body parts ---
    "fur", "pelt", "blubber", "mane", "whisker", "vibrissa",
    "feather", "plumage", "quill", "beak", "talon", "claw", "hoof", "hooves",
    "antler", "tusk", "fang", "snout", "muzzle", "gill", "tentacle",
    "antennae", "mandible", "proboscis", "paw", "pincer",
    "exoskeleton", "carapace", "cuticle", "elytra", "elytron",
    "thorax", "abdomen", "cephalothorax", "nacre", "byssus", "spinneret",
    "marrow", "cartilage", "tendon", "ligament", "sinew",
    "retina", "cornea", "eardrum", "cochlea", "follicle",
    "epidermis", "dermis", "flesh", "vertebrae", "femur", "ribcage",
    "bloodstream", "blood vessel", "artery", "arteries", "capillaries",
    "intestine", "intestinal", "stomach", "esophagus", "kidney", "liver",
    "spleen", "pancreas", "lung", "windpipe", "trachea", "larynx",
    "eyelid", "eyeball", "nostril", "earlobe", "fingernail", "knuckle",
    "skeleton", "skeletal", "skull", "jawbone", "limb", "appendage",

    # --- Plant anatomy ---
    "petal", "sepal", "blossom", "frond", "tendril", "rhizome", "tuber",
    "xylem", "phloem", "stomata", "stomatal", "chloroplast", "vascular bundle",
    "petiole", "pistil", "stamen", "anther", "carpel", "ovule",

    # --- Biological materials / secretions ---
    "keratin", "chitin", "collagen", "elastin", "melanin", "hemoglobin",
    "haemoglobin", "hemolymph", "mucus", "mucous", "saliva", "venom",
    "pheromone", "enzyme", "antibody", "antigen", "hormone", "nectar",
    "spore", "chlorophyll", "cellulose", "lignin",

    # --- Cellular / molecular ---
    "cytoplasm", "organelle", "ribosome", "chromosome", "genome", "genomic",
    "genetic", "genetics", "chromatin", "nucleotide", "amino acid",

    # --- Physiology / processes ---
    "photosynthesis", "photosynthetic", "metabolism", "metabolic",
    "metabolize", "digestion", "digestive", "secrete", "secretion",
    "excrete", "excretion", "perspire", "perspiration", "lactation",
    "gestation", "pollination", "pollinate", "fertilization",
    "hibernation", "hibernate", "circadian", "homeostatic",

    # --- Ecology / behavior ---
    "predator", "predatory", "prey", "forage", "foraging", "graze", "grazing",
    "parasite", "parasitic", "pollinator", "scavenger",
}

# ---------------------------------------------------------------------------
# AMBIGUOUS — real engineering/everyday meaning. Escalate, do not auto-fail.
# ---------------------------------------------------------------------------
AMBIGUOUS = {
    # anatomy words with engineering senses
    "scale", "shell", "wing", "spine", "rib", "joint", "trunk", "vein",
    "pore", "bone", "horn", "silk", "web", "skin", "hair", "bristle", "barb",
    "eye", "iris", "pupil", "tongue", "tail", "neck", "head", "arm", "leg",
    "foot", "finger", "muscle", "nerve", "tissue", "cell", "nucleus",
    "bladder", "valve",  # valve is anatomical and mechanical
    # plant words with engineering/everyday senses
    "leaf", "leaves", "stem", "root", "branch", "bark", "seed", "bud",
    "needle", "bulb", "pod", "husk", "kernel", "thorn", "stalk", "sap",
    "resin", "latex", "pollen",
    # short risky organism words
    "ray", "fly", "bat", "moth",  # ray of light, fly (v.), bat (object)
    # process words that are also general physical/systems terms
    "respiration", "respiratory", "transpiration", "osmosis", "osmotic",
    "diffusion", "evaporation", "germination", "germinate", "fertilize",
    "fertilizer", "reproduce", "reproduction", "regenerate", "regeneration",
    "evolve", "evolution", "evolutionary", "adapt", "adaptation",
    # ecology/behavior words common in tech/systems
    "ecosystem", "niche", "habitat", "colony", "swarm", "flock", "herd",
    "nest", "burrow", "host", "mate", "camouflage", "migration", "migrate",
    "symbiosis", "symbiotic", "hive", "den", "cocoon", "membrane-bound",
    # gene/cell family members that collide with neutral words handled here
    "gene", "genes", "vascular",
}

# ---------------------------------------------------------------------------
# ALLOWLIST — neutral terms AskNature endorses. Never flag; overrides tiers.
# ---------------------------------------------------------------------------
ALLOWLIST = {
    "membrane", "membranes", "fiber", "fibers", "fibre", "fibres",
    "tube", "tubes", "tubular", "layer", "layers", "layered",
    "channel", "channels", "surface", "surfaces", "structure", "structural",
    "gradient", "gradients", "lattice", "mesh", "matrix", "film", "coating",
    "panel", "strut", "node", "network", "cavity", "chamber", "duct",
    "conduit", "ridge", "groove", "taper", "filament",
}

# ---------------------------------------------------------------------------
# Safe prefix families (matched as \bPREFIX\w*\b). HARD tier.
# Chosen to avoid collisions with common neutral words.
# ---------------------------------------------------------------------------
HARD_PREFIXES = {
    "metaboli", "mitochond", "photosynth", "chlorophyl", "organel",
    "ribosom", "chromosom", "cytoplasm", "epiderm", "exoskelet",
    "invertebr", "vertebr", "predat",
}

# ---------------------------------------------------------------------------
# Matching engine
# ---------------------------------------------------------------------------
def _build_term_regex(terms):
    # multiword first so they match before their single-word parts
    escaped = sorted((re.escape(t) for t in terms), key=len, reverse=True)
    # optional regular plural; word boundaries
    return re.compile(r"\b(?:" + "|".join(escaped) + r")(?:es|s)?\b", re.IGNORECASE)

def _build_prefix_regex(prefixes):
    escaped = sorted((re.escape(p) for p in prefixes), key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(escaped) + r")\w*\b", re.IGNORECASE)

_HARD_RE = _build_term_regex(HARD)
_AMBIG_RE = _build_term_regex(AMBIGUOUS)
_PREFIX_RE = _build_prefix_regex(HARD_PREFIXES)
_ALLOW = {a.lower() for a in ALLOWLIST}


@dataclass
class ResidueResult:
    flag: bool                       # HARD residue present -> definite fail
    escalate: bool                   # AMBIGUOUS present -> send to LLM judge
    hard_hits: list = field(default_factory=list)
    ambiguous_hits: list = field(default_factory=list)


def check_biology_residue(text, source_organism_terms=None):
    """
    text: the design_strategy string under evaluation.
    source_organism_terms: list of organism names from the source strategy
        (common + scientific). Injected as HARD matches at runtime.
    """
    lowered_allow = _ALLOW
    hard, ambig = [], []

    for m in _HARD_RE.finditer(text):
        w = m.group(0).lower()
        if w not in lowered_allow:
            hard.append(m.group(0))
    for m in _PREFIX_RE.finditer(text):
        w = m.group(0).lower()
        if w not in lowered_allow:
            hard.append(m.group(0))
    for m in _AMBIG_RE.finditer(text):
        w = m.group(0).lower()
        if w not in lowered_allow:
            ambig.append(m.group(0))

    # dynamic organism injection (HARD)
    if source_organism_terms:
        for term in source_organism_terms:
            term = term.strip()
            if not term:
                continue
            if re.search(r"\b" + re.escape(term) + r"\b", text, re.IGNORECASE):
                hard.append(term)

    hard = sorted(set(hard), key=str.lower)
    ambig = sorted(set(ambig), key=str.lower)
    return ResidueResult(flag=bool(hard), escalate=bool(ambig),
                         hard_hits=hard, ambiguous_hits=ambig)


if __name__ == "__main__":
    tests = [
        ("A covering keeps heat inside by having many translucent tubes that warm "
         "an inner surface, while a dense layer of fine fibers prevents radiation loss.",
         ["polar bear", "Ursus maritimus"]),                       # clean
        ("The bear's fur traps air to insulate against cold.", ["bear"]),  # hard
        ("A shell structure distributes load across its scale-like panels.", None),  # ambiguous only
    ]
    for txt, orgs in tests:
        r = check_biology_residue(txt, orgs)
        print(f"flag={r.flag} escalate={r.escalate} "
              f"hard={r.hard_hits} ambiguous={r.ambiguous_hits}")

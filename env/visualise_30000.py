from _visualiser_base import run_visualiser
from config import CKPT_LEFT, CKPT_RIGHT

run_visualiser(
    left_path    = CKPT_LEFT.format(30000),
    right_path   = CKPT_RIGHT.format(30000),
    rally_label  = "30,000 rallies snapshot",
    infinite_mode = False,
    window_title  = "30,000 rallies",
)

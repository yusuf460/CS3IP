from _visualiser_base import run_visualiser
from config import CKPT_LEFT, CKPT_RIGHT

run_visualiser(
    left_path    = CKPT_LEFT.format(500),
    right_path   = CKPT_RIGHT.format(500),
    rally_label  = "500 rallies snapshot",
    infinite_mode = False,
    window_title  = "500 rallies",
)

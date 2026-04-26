from _visualiser_base import run_visualiser
from config import CKPT_LEFT, CKPT_RIGHT

run_visualiser(
    left_path    = CKPT_LEFT.format(15000),
    right_path   = CKPT_RIGHT.format(15000),
    rally_label  = "15,000 rallies snapshot",
    infinite_mode = False,
    window_title  = "15,000 rallies",
)

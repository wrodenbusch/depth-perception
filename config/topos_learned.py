# topos — positional arm 'learned'. See architecture.md.
# Everything except pos_mode is identical across the three arms, deliberately.
out_dir = 'out-topos-learned'
dataset = 'packed_windows_apps_kodcode_astcompact'
pos_mode = 'learned'

block_size = 1024          # a packed window of ~8.5 whole programs
vocab_size = 32768         # py_wholefile_astcompact_bpe_32k, trained from scratch

# Rate is NOT set here: each arm selects its own, because a rate tuned on one
# pack does not transfer, and handing the treatment arm the control's rate is
# the commonest way to fake a positional result.
# learning_rate = <from this arm's own selection sweep>

eval_interval = 250
eval_iters = 200
log_interval = 10
always_save_checkpoint = True
# Milestones are mandatory: every arm peaked at 600k examples in the prior run
# and got worse to 2M, and one conclusion INVERTS between those points.

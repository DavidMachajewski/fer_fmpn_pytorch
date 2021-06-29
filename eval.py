from args2 import Setup
from lib.agents.runner import Runner

"""
NOTE: USE THIS SCRIPT FOR TESTING/EVALUATING
OF ALREADY TRAINED MODELS.

IF YOU TRAIN WITHOUT EARLY STOPPING
THERE MAY BE BETTER CHECKPOINTS CREATED
AFTER EARLIER EPOCHS.

FMPN:
  To reload and use the fmpn network for testing 
  you need the three checkpoints of
    1. fmg
    2. pfn
    3. cn

  args.mode = train
  args.load_ckpt = 1 
  args.load_ckpt_fmg_only = 0
  args.ckpt_fmg = path_to_fmg
  args.ckpt_pfn = path_to_pfn
  args.ckpt_cn = path_to_cn
"""

if __name__ == '__main__':
    args = Setup().parse()
    runner = Runner(args)
    runner.start()

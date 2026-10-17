from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/models'))
from reference import run,verify
import argparse
a=argparse.ArgumentParser();a.add_argument('--cohort',choices=['full','low_ber'],default='full');a.add_argument('--experiment',required=True);v=a.parse_args();run(v.cohort,v.experiment)

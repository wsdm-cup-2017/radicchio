import sys
import random 
from utils import *

professions = read_one_column("../data/professions")
nationalities = read_one_column("../data/nationalities")

with open(sys.argv[1], "r") as f, open(sys.argv[2], "w") as of:
    buf = []
    cnt = 0
    for line in f:
        p, v = line.strip().split("\t")
        if len(buf) >= 1 and p != buf[0][0]:
            for x, y in buf:
                if len(buf) == 1:
                    of.write("%s\t%s\t1\n" %(x, y))
                    if cnt % 1 == 0:
                        nat = random.choice(professions)
                        #nat = random.choice(nationalities)
                        while nat == v:
                            nat = random.choice(nationalities)
                        of.write("%s\t%s\t0\n" %(x, nat))
                cnt += 1
            buf = []
        buf.append((p, v))

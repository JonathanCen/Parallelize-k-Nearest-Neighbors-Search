#!/usr/bin/python

import argparse
from argparse import RawTextHelpFormatter
import datetime
import struct
import random
from time import sleep
import sys

def getUniqueFileID():
    now = datetime.datetime.now()
    return int(now.strftime("%d%H%M%S"))

parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
parser.add_argument("trainingCount", type=int,help="Number of points")
parser.add_argument("cols", type=int,help="Number of dimensions")
parser.add_argument("dist", type=int,help="""Type of distribution\n0: Uniform Distribution\n1: Centered Uniform
        Distribution\n2: Beta Distribution\n3: Exponential Distribution""",choices=[0, 1, 2,3])
args = parser.parse_args()

fileid = getUniqueFileID()
datafilename = "data_" + str(fileid) + ".dat" 

fp = open(datafilename,"wb")

cols = args.cols
trainingCount = args.trainingCount
dist = args.dist

# generate file header for datafile
fp.write("TRAINING")
fp.write(struct.pack("=q",getUniqueFileID()))
fp.write(struct.pack("=q",args.trainingCount))
fp.write(struct.pack("=q",args.cols))

funcList = [random.uniform,random.gauss,random.betavariate,random.expovariate]
paramList = [(-1000,1000),(0,1000),(30,20),(0.01,)]

part = int(trainingCount/100)
# Now generating data for data file 
if(part!=0):
    print("Data file generation progress:")
for i in range(0,trainingCount):
    #progress bar
    if(part!=0 and i%part==0):
        sys.stdout.write('\r')
        sys.stdout.write("[%-100s] %d%%" % ('='*(i/part), (i/part)))
        sys.stdout.flush()
    buff = str()
    for j in range(0,cols):
        buff+=struct.pack("=f",funcList[dist](*paramList[dist]))
    fp.write(buff)
fp.close()

if(part!=0):
    print("\n")
print("\nData file : " + datafilename);
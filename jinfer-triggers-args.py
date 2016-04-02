import os
import sys
import random
import time
import math
import copy
import re

def getparams(line):
    p = line.find("(")
    tmp = line[p+1:]
    tmp = tmp[:tmp.find(")")]
    parts = tmp.split(",")
    for i in range(0,len(parts),1):
        parts[i] = parts[i].strip()
    return parts


if len(sys.argv)<5:
	sys.exit("Usage:eventextractor testfolder softevidsfolder mlnwtsfile outputfolder (all folders should end with /)")

dname=sys.argv[1]	
rdir = sys.argv[4]
ofile = open("softevidencelist.dat",'w')
flist = os.listdir(sys.argv[2])
for f in flist:
	ofile.write(sys.argv[2]+f+"\n")
ofile.close()
os.system("./Release/jinfer softevidencelist.dat "+rdir +" "+sys.argv[3])

files = os.listdir(dname)
for f in files:
    if f.find('DS_Store') >=0:
        continue
    print(f)
    ifile = open(dname+f)
    lines = ifile.readlines()
    ifile.close()
    ifile = open(rdir+f)
    jvals = ifile.readlines()
    ifile.close()
    rfile = open(rdir+f,'w')
    ind = 0
    jidx = 0
    numtoks = 0
    data = []
    while ind <= len(lines)-1:
        if len(lines[ind]) < 4:
            ind = ind + 1
            continue
        if lines[ind][0]=='t':
            while lines[ind][0]=='t':
                ind = ind + 1
            data.append(lines[ind-1])
            numtoks = numtoks + 1
        if lines[ind][0]=='a':
            while lines[ind][0]=='a':
                ind = ind + 1
            data.append(lines[ind-1])
            numtoks = numtoks + 1
        if lines[ind][0]=="=":
            if numtoks == 1:
                rfile.write('T:0\n')
            else:
                for i in range(jidx,jidx+numtoks,1):
                    rfile.write(jvals[i])
                jidx = jidx + numtoks
            numtoks = 0
            data[:] = []
        ind = ind + 1        
    print(f)
    rfile.close()

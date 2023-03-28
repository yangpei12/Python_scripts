import pandas as pd
import os
os.chdir(r"E:\售后\bederror")
bedData = pd.read_table("map.txt")
class BedCheck():
    def __init__(self, blockSize, blockStarts, blockLength):
        self.blockSize = blockSize
        self.blockStarts = blockStarts
        self.blockLength = blockLength
    def exon_region(self):
        l = []
        for n in range(0,self.blockLength):
            exon = [int(self.blockStarts[n]), int(self.blockStarts[n]) + int(self.blockSize[n])]
            l.extend(exon)
        return l
    def exon_stat(self):
        ls = bed.exon_region()
        distance = [ls[2 * m] - ls[2 * m - 1] for m in range(1, blockLength)]
        return distance

for idx in range(0,len(bedData)):
    blockSize = bedData.loc[idx,'blockSize'].rstrip(',').split(',')
    blockStarts = bedData.loc[idx, 'blockStarts'].rstrip(',').split(',')
    blockLength = len(blockSize)
    bed = BedCheck(blockSize, blockStarts, blockLength)
    exon_boundary = bed.exon_stat()
    output = pd.Series(exon_boundary)
    output.name = bedData.loc[idx,'name']
    output.to_csv('map_check.csv',mode='a')

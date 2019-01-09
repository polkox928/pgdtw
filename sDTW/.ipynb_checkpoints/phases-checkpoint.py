# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 12:54:07 2018

@author: DEPAGRA
"""
from datetime import datetime
import pickle
import re

phaseBatch = batchData[5393][0]

startBatch = datetime.strptime(phaseBatch['start'], "%Y-%m-%d %H:%M:%S")
endBatch = datetime.strptime(phaseBatch['end'], "%Y-%m-%d %H:%M:%S")

examplePhase = phaseBatch['phases'][0]
startExamplePhase = datetime.strptime(examplePhase['stime'], "%Y-%m-%dT%H:%M:%S.000+0000")
delayStartExamplePhase = (startExamplePhase-startBatch).seconds/60

#%%
import re
r = re.compile("\d.\d{1,2}$")
delay = []
for phase in phaseBatch['phases']:
    if r.match(phase['node']) is not None:
        start = datetime.strptime(phase['stime'], "%Y-%m-%dT%H:%M:%S.000+0000")
        
        delay.append((start-startBatch).seconds/60)
    
delay = sorted(delay)

#%%
# filter phases in my actual data

minStart = startBatch
maxEnd = endBatch
delay = []
for phase in phaseBatch['phases']:
    if r.match(phase['node']) is not None:
        end = datetime.strptime(phase['etime'], "%Y-%m-%dT%H:%M:%S.000+0000")
        start = datetime.strptime(phase['stime'], "%Y-%m-%dT%H:%M:%S.000+0000")
#        print(phase['node'], start, end)
        if start < startBatch or end > endBatch:
            print(phase['node'])
        else:
            delay.append(((start-startBatch).seconds/60, (end-startBatch).seconds/60))
            print("ok \t", phase['node'])
delay = sorted(delay)

#%%

r = re.compile("\d.\d{1,2}$")
with open("C:\\Users\\DEPAGRA\\Documents\\GitHub\\pgdtw\\data\\batch_data.pickle", "rb") as infile:
    batchData = pickle.load(infile)

        
IDs = [_id for _id in batchData.keys() if isinstance(_id, int)]
phaseList = list()
noPhaseBatch = list()
for ID in IDs:
    try:
        phaseBatch = batchData[ID][0]
        
        startBatch = datetime.strptime(phaseBatch['start'], "%Y-%m-%d %H:%M:%S")
        endBatch = datetime.strptime(phaseBatch['end'], "%Y-%m-%d %H:%M:%S")
        
        delay = list()
        for phase in phaseBatch['phases']:
            if r.match(phase['node']) is not None:
                end = datetime.strptime(phase['etime'], "%Y-%m-%dT%H:%M:%S.000+0000")
                start = datetime.strptime(phase['stime'], "%Y-%m-%dT%H:%M:%S.000+0000")
    
                if start < startBatch or end > endBatch:
                    pass
                else:
                    delay.append(((start-startBatch).seconds/60, (end-startBatch).seconds/60, phase['name'], phase['node'], ID))
        phaseList.append(delay)
    except:
        noPhaseBatch.append(ID)            
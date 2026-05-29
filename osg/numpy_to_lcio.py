#!/usr/bin/env python3
import numpy as np
import pyLCIO
from pyLCIO import IOIMPL, IMPL, EVENT
from collections import defaultdict
import ROOT
from ROOT import Math
import os
#from tqdm import tqdm


OUTPUT_PATH = "./output_flow.slcio"

# Load hit arrays
collections = [
    "InnerTrackerBarrelCollection",
    "InnerTrackerEndcapCollection",
    "OuterTrackerBarrelCollection",
    "OuterTrackerEndcapCollection",
    "VertexBarrelCollection",
    "VertexEndcapCollection"    
]
NUM_EVENTS = 1


# Open LCIO writer
writer = IOIMPL.LCFactory.getInstance().createLCWriter()
writer.open(OUTPUT_PATH, EVENT.LCIO.WRITE_NEW)


for evt_num in range(NUM_EVENTS):

    evt = IMPL.LCEventImpl()
    evt.setEventNumber(evt_num)
    evt.setRunNumber(1)

    for COLLECTION_NAME in collections:

        print(COLLECTION_NAME)

        hits_array = np.load(f"/scratch/rrm39/v7_reco/recoBIB/flow_samples/{COLLECTION_NAME}.npy")

        col = IMPL.LCCollectionVec(EVENT.LCIO.SIMTRACKERHIT)

        col.getParameters().setValue(
            "CellIDEncoding",
            "system:0:5,side:5:-2,layer:7:6,module:13:11,sensor:24:8"
        )


        for hit in hits_array[:len(hits_array)//2]:

            if "Barrel" in COLLECTION_NAME:

                logE, t, r, phi, z, _, _, cell_id = hit
            elif "Endcap" in COLLECTION_NAME:

                logE, t, r, phi, z, _, _, cell_id = hit

            x = r * np.cos(phi)
            y = r * np.sin(phi)


            simhit = IMPL.SimTrackerHitImpl()

            simhit.setEDep(np.exp(logE))
            simhit.setTime(t)
            simhit.setPosition(np.array([x, y, z], dtype=np.float64))
            simhit.setCellID0(int(cell_id))

            col.addElement(simhit)

        evt.addCollection(col, COLLECTION_NAME)

    writer.writeEvent(evt)

writer.close()

print(f"Wrote {OUTPUT_PATH} with {NUM_EVENTS} events using closest geometry CellID0")

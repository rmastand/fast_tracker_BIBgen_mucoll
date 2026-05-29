## Samples -> LCIO files.


1. run `assign_cell_ID.ipynb`

ON OSG

2. run `numpy_to_lcio.py`
3. Run digitization and reconstruction with `k4run /path/to/SteeringMacros/k4Reco/steer_reco.py --code /path/to/code --data /path/to/data --inputFile /path/to/output_flow.slcio  --outputFile /path/to/output/file  --skipTrackerConing --trackerOnly --forceSurface --skipTruth`
4. Make summary track plots with `compare_tracks.py` (written by Mark Larson)

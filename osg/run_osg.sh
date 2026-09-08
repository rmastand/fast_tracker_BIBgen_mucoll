#!/bin/bash
# Load OSG paths from configs_osg.yaml
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG="$SCRIPT_DIR/configs_osg.yaml"
eval "$(python3 -c "
import yaml
cfg = yaml.safe_load(open('$CFG'))
print('\n'.join(f'{k}=\"{v}\"' for k, v in cfg.items()))
")"

run_pipeline() {
    INPUT_SUFFIX=$1
    OUTPUT_NAME=$2
    BASE=$SLCIO_DIR

    #convert to lcio
    python $SCRIPT_DIR/numpy_to_lcio.py \
    --OUTPUT_PATH ${BASE}/output_${OUTPUT_NAME}.slcio \
    --INPUT_SUFFIX ${INPUT_SUFFIX}

    # run digi
    k4run $SCRIPT_DIR/steer_reco.py \
    --code $SCRATCH_DIR \
    --data $SCRATCH_DIR \
    --inputFile  ${BASE}/output_${OUTPUT_NAME}.slcio \
    --outputFile ${BASE}/output_${OUTPUT_NAME}_digi.slcio \
    --skipTrackerConing --trackerOnly --forceSurface --skipTruth \
    --skipReco

    # run reco
    k4run $SCRIPT_DIR/steerBIBtracking.py \
    --code $SCRATCH_DIR \
    --data $SCRATCH_DIR \
    --inputFile  ${BASE}/output_${OUTPUT_NAME}_digi.slcio \
    --outputFile ${BASE}/output_${OUTPUT_NAME}_reco_selected_0_30.slcio \
    --minTheta 0 --maxTheta 30

    k4run $SCRIPT_DIR/steerBIBtracking.py \
    --code $SCRATCH_DIR \
    --data $SCRATCH_DIR \
    --inputFile  ${BASE}/output_${OUTPUT_NAME}_digi.slcio \
    --outputFile ${BASE}/output_${OUTPUT_NAME}_reco_selected_30_70.slcio \
    --minTheta 30 --maxTheta 70

    k4run $SCRIPT_DIR/steerBIBtracking.py \
    --code $SCRATCH_DIR \
    --data $SCRATCH_DIR \
    --inputFile  ${BASE}/output_${OUTPUT_NAME}_digi.slcio \
    --outputFile ${BASE}/output_${OUTPUT_NAME}_reco_selected_70_110.slcio \
    --minTheta 70 --maxTheta 110

    k4run $SCRIPT_DIR/steerBIBtracking.py \
    --code $SCRATCH_DIR \
    --data $SCRATCH_DIR \
    --inputFile  ${BASE}/output_${OUTPUT_NAME}_digi.slcio \
    --outputFile ${BASE}/output_${OUTPUT_NAME}_reco_selected_110_150.slcio \
    --minTheta 110 --maxTheta 150

    k4run $SCRIPT_DIR/steerBIBtracking.py \
    --code $SCRATCH_DIR \
    --data $SCRATCH_DIR \
    --inputFile  ${BASE}/output_${OUTPUT_NAME}_digi.slcio \
    --outputFile ${BASE}/output_${OUTPUT_NAME}_reco_selected_150_180.slcio \
    --minTheta 150 --maxTheta 180
}

run_pipeline diff_local_reco_1     diff_local_reco_1

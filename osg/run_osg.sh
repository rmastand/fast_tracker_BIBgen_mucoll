run_pipeline() {
    INPUT_SUFFIX=$1
    OUTPUT_NAME=$2
    BASE=/scratch/rrm39/v7_reco/slcio

    #convert to lcio
    python /scratch/rrm39/tutorial2024/MuC-Tutorial/analysis/BIBAI/numpy_to_lcio.py \
    --OUTPUT_PATH ${BASE}/output_${OUTPUT_NAME}.slcio \
    --INPUT_SUFFIX ${INPUT_SUFFIX}

    # run digi
    k4run /scratch/rrm39/SteeringMacros/k4Reco/steer_reco.py \
    --code /scratch/rrm39 \
    --data /scratch/rrm39 \
    --inputFile  ${BASE}/output_${OUTPUT_NAME}.slcio \
    --outputFile ${BASE}/output_${OUTPUT_NAME}_digi.slcio \
    --skipTrackerConing --trackerOnly --forceSurface --skipTruth \
    --skipReco

    # run reco
    k4run /scratch/rrm39/SteeringMacros/k4Reco/steer_BIBtracking.py \
    --code /scratch/rrm39 \
    --data /scratch/rrm39 \
    --inputFile  ${BASE}/output_${OUTPUT_NAME}_digi.slcio \
    --outputFile ${BASE}/output_${OUTPUT_NAME}_reco_selected_0_30.slcio \
    --minTheta 0 --maxTheta 30

    k4run /scratch/rrm39/SteeringMacros/k4Reco/steer_BIBtracking.py \
    --code /scratch/rrm39 \
    --data /scratch/rrm39 \
    --inputFile  ${BASE}/output_${OUTPUT_NAME}_digi.slcio \
    --outputFile ${BASE}/output_${OUTPUT_NAME}_reco_selected_30_70.slcio \
    --minTheta 30 --maxTheta 70

    k4run /scratch/rrm39/SteeringMacros/k4Reco/steer_BIBtracking.py \
    --code /scratch/rrm39 \
    --data /scratch/rrm39 \
    --inputFile  ${BASE}/output_${OUTPUT_NAME}_digi.slcio \
    --outputFile ${BASE}/output_${OUTPUT_NAME}_reco_selected_70_110.slcio \
    --minTheta 70 --maxTheta 110

    k4run /scratch/rrm39/SteeringMacros/k4Reco/steer_BIBtracking.py \
    --code /scratch/rrm39 \
    --data /scratch/rrm39 \
    --inputFile  ${BASE}/output_${OUTPUT_NAME}_digi.slcio \
    --outputFile ${BASE}/output_${OUTPUT_NAME}_reco_selected_110_150.slcio \
    --minTheta 110 --maxTheta 150

    k4run /scratch/rrm39/SteeringMacros/k4Reco/steer_BIBtracking.py \
    --code /scratch/rrm39 \
    --data /scratch/rrm39 \
    --inputFile  ${BASE}/output_${OUTPUT_NAME}_digi.slcio \
    --outputFile ${BASE}/output_${OUTPUT_NAME}_reco_selected_150_180.slcio \
    --minTheta 150 --maxTheta 180
}

run_pipeline diff_local_reco_1     diff_local_reco_1

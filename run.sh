#!/bin/bash

# Default values
BASE_LAYERS=8
BASE_WIDTH=64
BASE_RANDOMRATIO=0.0
BASE_ADDITIONAL_RANDOM_FEATURES=1
EPOCHS=2
DATASET="EXP"
PROBDIST="n"
NORMLAYERS=1
ACTIVATION="tanh"
LEARNRATE=0.00065
LEARNRATEGIN=0.00035

# Parameter ranges
LAYER_OPTIONS=(4 8 16 32)
WIDTH_OPTIONS=(32 64 128)
RANDOMRATIO_OPTIONS=(0.0 0.5 1.0)
ADDITIONAL_RANDOM_FEATURES_OPTIONS=(1 32 64 128)

# Command template
BASE_COMMAND="python3 GNNHyb.py -epochs $EPOCHS -dataset $DATASET -probDist $PROBDIST -normLayers $NORMLAYERS -activation $ACTIVATION -learnRate $LEARNRATE -learnRateGIN $LEARNRATEGIN"

# Vary layers
for layers in "${LAYER_OPTIONS[@]}"; do
    echo "Running with layers=$layers"
    $BASE_COMMAND -layers $layers -width $BASE_WIDTH -randomRatio $BASE_RANDOMRATIO -additionalRandomFeatures $BASE_ADDITIONAL_RANDOM_FEATURES
done

# Vary width
for width in "${WIDTH_OPTIONS[@]}"; do
    echo "Running with width=$width"
    $BASE_COMMAND -layers $BASE_LAYERS -width $width -randomRatio $BASE_RANDOMRATIO -additionalRandomFeatures $BASE_ADDITIONAL_RANDOM_FEATURES
done

# Vary additionalRandomFeatures
for additionalRandomFeatures in "${ADDITIONAL_RANDOM_FEATURES_OPTIONS[@]}"; do
    echo "Running with additionalRandomFeatures=$additionalRandomFeatures"
    $BASE_COMMAND -layers $BASE_LAYERS -width $BASE_WIDTH -randomRatio $BASE_RANDOMRATIO -additionalRandomFeatures $additionalRandomFeatures
done

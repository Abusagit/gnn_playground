set -eux pipefail


# export SNAPSHOT_PATH=$(mktemp -d)

export OUT_LOGS_PATH="$SNAPSHOT_PATH"/auto_logs

export PYTHONPATH=$PYTHONPATH:$(pwd)
export HOME_PATH=$(pwd)



# install more packages if absent in the porto layer using "Libraries to install" option, or call pip here
# for large number of packages consider diff layer or whole new layer
echo "=========== SNAPSHOT_PATH stuff INITIAL =============="
find $SOURCE_CODE_PATH -maxdepth 2
ls -la $SOURCE_CODE_PATH


echo "=========== SOURCE_CODE_PATH stuff =============="
ls -laR $SNAPSHOT_PATH


echo "=========== Installing fragile DGL =============="
pip install dgl -f $SOURCE_CODE_PATH/dgl_cu117/repo.html
python3 -c "import dgl; print('DGL version: ', dgl.__version__)"
python3 -c "import torch; print('Pytorch version: ', torch.__version__)"
echo "=========== Installing COMPLETE! =============="

# link local data dir
ln -s "${INPUT_PATH}/" "${SOURCE_CODE_PATH}/code/data"


echo "=========== INPUT_PATH stuff =============="
ls -laR $INPUT_PATH

echo "=========== save env =============="
echo $(env)
env > $PWD/envlist


##############
# Run script #
##############
echo "=========== MAIN CODE EXECUTION =============="
cd ${SOURCE_CODE_PATH}/code

python3 main.py --model_type GNN --data_type dglgraph


# now we have new file - `index2logit` in ${SOURCE_CODE_PATH}/code - move it to $DATA_PATH

echo "=========== cooking OUTPUT =============="

mv ./index2logit $DATA_PATH/index2logit

mv ./checkpoints $SNAPSHOT_PATH


echo "{"cats":["Grumpy Cat","Nyan Cat","Smudge the Cat"]}" > $JSON_OUTPUT_FILE

echo "job complete" > $OUT_LOGS_PATH/out.log


echo "=========== SNAPSHOT_PATH stuff AFTER RUN =============="
ls -laR $SNAPSHOT_PATH

echo "=========== DATA_PATH stuff =============="
ls -laR $DATA_PATH

echo "=========== OUT_LOGS_PATH stuff =============="
ls -laR $OUT_LOGS_PATH

echo "job complete"

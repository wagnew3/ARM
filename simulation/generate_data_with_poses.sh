# This bash script runs the data generation code.
# It runs it in intervals due to some memory leak in PyBullet
#
#
# This code should not all be run at once. Stuff should be commented out so we don't hit memory leaks...

NUM_PROCESSES=10 # not used
NUM_EXS_PER_CPU=5 # Set this low. maybe 2 is even better.
PARALLEL_BATCH_SIZE=50 # make sure NUM_PROCESSES * NUM_EXS_PER_CPU = PARALLEL_BATCH_SIZE
INTERVAL=50 

TRAIN_START=0
TRAIN_END=0
TEST_START=741
TEST_END=750
# Generate RGBD/Seg images w/ GUI sequentially
for i in `seq $TRAIN_START $INTERVAL $TRAIN_END`; do

    batch_start=$i;
    batch_end=$(($i + $INTERVAL));
    if [ $batch_end -gt $TRAIN_END ]; then
    batch_end=$TRAIN_END;
    fi

    python generate_data_with_poses.py $batch_start $batch_end train;
done

# Generate RGBD/Seg images w/ GUI sequentially
for i in `seq $TEST_START $INTERVAL $TEST_END`; do

    batch_start=$i;
    batch_end=$(($i + $INTERVAL));
    if [ $batch_end -gt $TEST_END ]; then
    batch_end=$TEST_END;
    fi

    python generate_data_with_poses.py $batch_start $batch_end test;
done



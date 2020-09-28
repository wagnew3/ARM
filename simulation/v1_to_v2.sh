# This bash script runs the data generation code.
# It runs it in intervals due to some memory leak in PyBullet
#
#
# This code should not all be run at once. Stuff should be commented out so we don't hit memory leaks...

INTERVAL=50
TRAIN_START=0
TRAIN_END=10000
TEST_START=0
TEST_END=2000

# Generate training sequences
for i in `seq $TRAIN_START $INTERVAL $TRAIN_END`; do

    batch_start=$i;
    batch_end=$(($i + $INTERVAL));
    if [ $batch_end -gt $TRAIN_END ]; then
    batch_end=$TRAIN_END;
    fi

    python v1_to_v2.py $batch_start $batch_end train;
done

# Generate test sequences
for i in `seq $TEST_START $INTERVAL $TEST_END`; do

    batch_start=$i;
    batch_end=$(($i + $INTERVAL));
    if [ $batch_end -gt $TEST_END ]; then
    batch_end=$TEST_END;
    fi

    python v1_to_v2.py $batch_start $batch_end test;
done

# # Fill in training scene gaps
# while [ $(ls -1 /data/tabletop_dataset/training_set/ | wc -l) -lt $TRAIN_END ]; do
#     python v1_to_v2.py $TRAIN_START $TRAIN_END train;
# done

# # Fill in test scene gaps
# while [ $(ls -1 /data/tabletop_dataset/test_set/ | wc -l) -lt $TEST_END ]; do
#     python v1_to_v2.py $TEST_START $TEST_END test;
# done

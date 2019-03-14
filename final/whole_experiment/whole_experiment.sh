echo "Start experiment" $(date) > experiment.txt
echo "Weight Optimization Started at " $(date) >> experiment.txt
python3 weight_optimization.py 3 -1
echo "Weight Optimization Ended at " $(date) >> experiment.txt
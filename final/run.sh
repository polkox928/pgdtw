#!/bin/bash
python local_constraint_selection.py 25
python local_constraint_selection.py 50
python local_constraint_selection.py 100
python local_constraint_selection.py 200
python weight_optimization.py

#!/bin/bash

# Loop through case_IDs and run mycode.py with the specified case_ID
for i in {0..999}
do
    python3 generate_dataset.py --case_ID $i
done
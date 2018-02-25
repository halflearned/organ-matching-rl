#!/usr/bin/env bash

cd sirius_results/
scp -P 22022 baisihad@sirius.bc.edu:/data/baisihad/matching/results/*.txt .
cd ..
cd pleiades_results/
scp -P 22022 baisihad@pleiades.bc.edu:/home/baisihad/matching/results/*.txt .
cd ..
cat sirius_results/bandit*txt pleiades_results/bandit*txt sherlock_results/*txt yens_results/*txt > bandit_results_Feb15.txt
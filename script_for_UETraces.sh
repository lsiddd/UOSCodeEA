#!/bin/bash
for ((i=0; i<32; i++))
do
./waf --run="scratch/UOS-LTE-v2 --nRuns=1 --scen=4 --randomSeed=85 --traceFile=scratch/UOS_UE_Scenario_$i.ns_movements"
done


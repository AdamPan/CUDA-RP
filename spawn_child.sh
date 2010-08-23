#! /bin/bash
   cp ./plot.gnu $3/plot.gnu
   ./bin/linux/release/bubbles confs/$1_$2.txt $3/rawdata_$2/
   cp confs/$1_$2.txt $3/$1_$2.txt
   sh plot_results_child.sh $4 $5 $6 $7 $3/ $2 &

#! /bin/bash
rm ./images/ -rf
mkdir ./images/
cd ./results/
ls *.txt | sed "s/.txt//" > list
for i in `cat list` ; do
   sed -e "s/INPUTFILE/$i/" -e "s/OUTPUTFILE/$i/" \
    ../plot.gnu | gnuplot
done
rm list

#! /bin/bash
rm ./images/ -rf
mkdir ./images/
cd ./results/
ls *.txt | sed "s/.txt//" > list
for i in `cat list` ; do
   TITLE=`echo $i | sed -e "s/\_/ /g" | sed -e "s/step/at step/g"`
   sed -e "s/INPUTFILE/$i/" -e "s/OUTPUTFILE/$i/" -e "s/XBOUND/$1/g" -e "s/YBOUND/$2/" -e "s/XRES/$3/g" -e"s/YRES/$4/g" -e "s/TITLE/$TITLE/g" ../plot.gnu | gnuplot
done
rm list

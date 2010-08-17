#! /bin/bash
rm ./images/ -rf
mkdir ./images/
cd ./results/
ls *.txt | sed "s/.txt//" > list
for i in `cat list` ; do
   TITLE=`echo $i | sed -e "s/\_/ /g" | sed -e "s/step/at step/g"`
   case $# in
   '0')
   sed -e "s/INPUTFILE/$i/" -e "s/OUTPUTFILE/$i/" -e "s/set xrange/\#set xrange/" -e "s/set yrange/\#set yrange/" -e "s/XRES/1200/g" -e"s/YRES/1200/g" -e "s/TITLE/$TITLE/g" ../plot.gnu | gnuplot
   ;;
   '2')
   sed -e "s/INPUTFILE/$i/" -e "s/OUTPUTFILE/$i/" -e "s/XBOUND/$1/g" -e "s/YBOUND/$2/" -e "s/XRES/1200/g" -e"s/YRES/1200/g" -e "s/TITLE/$TITLE/g" ../plot.gnu | gnuplot
   ;;
   '4')
   sed -e "s/INPUTFILE/$i/" -e "s/OUTPUTFILE/$i/" -e "s/XBOUND/$1/g" -e "s/YBOUND/$2/" -e "s/XRES/$3/g" -e"s/YRES/$4/g" -e "s/TITLE/$TITLE/g" ../plot.gnu | gnuplot
   ;;
   *)
   echo "invalid number of arguments"
   ;;
   esac
done
rm list

#! /bin/bash
if [ -d ./images/ ] ; then
   echo "Image folder exits, removing..."
   rm ./images/ -rf
fi
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

cd ..

if [ -d ./videos/ ] ; then
   echo "Video folder exists, removing..."
   rm ./videos/ -rf
fi

mkdir videos
mencoder "mf://images/T*.png" -mf type=png:fps=10 -ovc raw -vf format=yuy2 -o videos/Temperature.avi
mencoder "mf://images/fg*.png" -mf type=png:fps=10 -ovc raw -vf format=yuy2 -o videos/Void_Fraction.avi
mencoder "mf://images/p0*.png" -mf type=png:fps=10 -ovc raw -vf format=yuy2 -o videos/Pressure.avi
rm list

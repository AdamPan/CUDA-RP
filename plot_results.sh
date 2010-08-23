#! /bin/bash
if [ -d ./images/ ] ; then
   echo "Image folder exits, removing..."
   rm ./images/ -rf
fi
mkdir ./images/

if [ -d ./videos/ ] ; then
   echo "Video folder exists, removing..."
   rm ./videos/ -rf
fi
mkdir videos
cd ./rawdata/


PREFIXES=`ls * | cut -d"_" -f1 | uniq`

for pre in $PREFIXES ; do
    ls ${pre}_step_*.txt | sed -e "s/.txt//" > list
    plot_min=`head -1 ${pre}_minmax.txt`
    plot_max=`tail -1 ${pre}_minmax.txt`
    echo $plot_min
    echo $plot_max
    for i in `cat list` ; do
       TITLE=`echo $i | sed -e "s/\_/ /g" | sed -e "s/step/at step/g"`
       case $# in
       '0')
       sed -e "s/INPUTFILE/$i/" -e "s/OUTPUTFILE/$i/" -e "s/set xrange/\#set xrange/" -e "s/set yrange/\#set yrange/" -e "s/XRES/1200/g" -e"s/YRES/1200/g" -e "s/TITLE/$TITLE/g" -e "s/LOWERRANGE/$plot_min/g" -e "s/UPPERRANGE/$plot_max/g" ../plot.gnu | gnuplot
       ;;
       '2')
       sed -e "s/INPUTFILE/$i/" -e "s/OUTPUTFILE/$i/" -e "s/XBOUND/$1/g" -e "s/YBOUND/$2/" -e "s/XRES/1200/g" -e"s/YRES/1200/g" -e "s/TITLE/$TITLE/g" -e "s/LOWERRANGE/$plot_min/g" -e "s/UPPERRANGE/$plot_max/g" ../plot.gnu | gnuplot
       ;;
       '4')
       sed -e "s/INPUTFILE/$i/" -e "s/OUTPUTFILE/$i/" -e "s/XBOUND/$1/g" -e "s/YBOUND/$2/" -e "s/XRES/$3/g" -e"s/YRES/$4/g" -e "s/TITLE/$TITLE/g" -e "s/LOWERRANGE/$plot_min/g" -e "s/UPPERRANGE/$plot_max/g" ../plot.gnu | gnuplot
       ;;
       *)
       echo "invalid number of arguments"
       ;;
       esac
    done
    rm list

    mencoder "mf://../images/$pre*.png" -mf type=png:fps=10 -ovc lavc -lavcopts vcodec=msmpeg4v2:vbitrate=3000 -o ../videos/$pre.avi
done

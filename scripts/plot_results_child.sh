#! /bin/bash
cd $5
if [ -d ./images_$6/ ] ; then
   rm ./images_$6/ -rf
fi
mkdir ./images_$6/

if [ -d ./videos_$6/ ] ; then
   rm ./videos_$6/ -rf
fi
mkdir videos_$6
cd ./rawdata_$6/


PREFIXES=`ls * | cut -d"_" -f1 | uniq`

for pre in $PREFIXES ; do
    list=`ls ${pre}_step_*.txt | sed -e "s/.txt//"`
    plot_min=`head -1 ${pre}_minmax.txt`
    plot_max=`tail -1 ${pre}_minmax.txt`
    for i in $list ; do
       TITLE=`echo $i | sed -e "s/\_/ /g" | sed -e "s/step/at step/g"`
       sed -e "s/INPUTFILE/$i/" -e "s/OUTPUTFILE/$i/" -e "s/XBOUND/$1/g" -e "s/YBOUND/$2/" -e "s/XRES/$3/g" -e"s/YRES/$4/g" -e "s/TITLE/$TITLE/g" -e "s/LOWERRANGE/$plot_min/g" -e "s/UPPERRANGE/$plot_max/g" -e "s/ITERATOR/$6/g" ../plot.gnu | gnuplot
    done
./15
    mencoder "mf://../images_$6/$pre*.png" -mf type=png:fps=10 -ovc lavc -lavcopts vcodec=msmpeg4v2:vbitrate=3000 -o ../videos_$6/$pre.avi > /dev/null 2>&1
done

rm ../plot.gnu

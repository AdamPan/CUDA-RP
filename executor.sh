#! /bin/bash
file_numbers=`ls confs/$1_*.txt | sed "s/.txt//" | sed "s/confs\/$1_//"`
target_dir_0="results"
target_dir_1=`date -I`
target_dir_2=`date +%R`

if [ -d $target_dir_0 ] ; then
   echo "$target_dir_0 exists"
else
   mkdir $target_dir_0
fi

if [ -d $target_dir_0/$target_dir_1 ] ; then
   echo "$target_dir_0/$target_dir_1 exists"
else
   mkdir $target_dir_0/$target_dir_1
fi

if [ -d $target_dir_0/$target_dir_1/$target_dir_2 ] ; then
   echo "$target_dir_0/$target_dir_1/$target_dir_2 exits"
else
   mkdir $target_dir_0/$target_dir_1/$target_dir_2
fi

for i in $file_numbers ; do
   XRES=`grep -w X confs/$1_$i.txt | sed "s/X//g" | sed "s/=//" | sed "s/[\t]*//"`
   YRES=`grep -w Y confs/$1_$i.txt | sed "s/Y//g" | sed "s/=//" | sed "s/[\t]*//"`
   XBOUND=`grep -w LX confs/$1_$i.txt | sed "s/LX//g" | sed "s/=//" | sed "s/[\t]*//"`
   YBOUND=`grep -w LY confs/$1_$i.txt | sed "s/LY//g" | sed "s/=//" | sed "s/[\t]*//"`

   XRES=$(((XRES*2) * 2))
   YRES=$(((YRES) * 2))

   ./bin/linux/release/bubbles confs/$1_$i.txt
   sh plot_rawdata.sh $XBOUND $YBOUND $XRES $YRES

   mv rawdata $target_dir_0/$target_dir_1/$target_dir_2/rawdata_$i
   mv images $target_dir_0/$target_dir_1/$target_dir_2/images_$i
   mv videos $target_dir_0/$target_dir_1/$target_dir_2/videos_$i
   mv runtime.txt $target_dir_0/$target_dir_1/$target_dir_2/runtime_$i.txt
   cp confs/$1_$i.txt $target_dir_0/$target_dir_1/$target_dir_2/$1_$i.txt
done

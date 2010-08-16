#! /bin/bash


file_numbers=`ls confs/$1_[1-99].txt | sed "s/.txt//" | sed "s/confs\/$1_//"`
target_dir_0="data"
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
   ./bin/linux/release/bubbles confs/$1_$i.txt
   sh plot_results.sh

   mv results $target_dir_0/$target_dir_1/$target_dir_2/results_$i
   mv images $target_dir_0/$target_dir_1/$target_dir_2/images_$i
   mv runtime.txt $target_dir_0/$target_dir_1/$target_dir_2/runtime_$i.txt
done

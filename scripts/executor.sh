#! /bin/bash
cd $(readlink -f $0)/..
file_numbers=`ls confs/$1_*.txt | sed "s/.txt//" | sed "s/confs\/$1_//"`
result_folder="results"
current_date=`date -I`
current_time=`date +%R`
save_folder="./$result_folder/$current_date/$current_time"

if [ -d $result_folder ] ; then
   echo "$result_folder exists"
else
   mkdir $result_folder
fi

if [ -d $result_folder/$current_date ] ; then
   echo "$result_folder/$current_date exists"
else
   mkdir $result_folder/$current_date
fi

if [ -d $result_folder/$current_date/$current_time ] ; then
   echo "$result_folder/$current_date/$current_time exits"
else
   mkdir $result_folder/$current_date/$current_time
fi

for i in $file_numbers ; do
   XRES=`grep -w X confs/$1_$i.txt | sed "s/X//g" | sed "s/=//" | sed "s/[\t]*//"`
   YRES=`grep -w Y confs/$1_$i.txt | sed "s/Y//g" | sed "s/=//" | sed "s/[\t]*//"`
   XBOUND=`grep -w LX confs/$1_$i.txt | sed "s/LX//g" | sed "s/=//" | sed "s/[\t]*//"`
   YBOUND=`grep -w LY confs/$1_$i.txt | sed "s/LY//g" | sed "s/=//" | sed "s/[\t]*//"`

   XRES=$((XRES*2))
   YRES=$((YRES))

   cp ./scripts/plot.gnu $save_folder/plot.gnu
   ./bin/linux/release/bubbles confs/$1_$i.txt $save_folder/rawdata_$i/
   cp confs/$1_$i.txt $save_folder/$1_$i.txt
   sh ./scripts/plot_results_child.sh $XBOUND $YBOUND $XRES $YRES $save_folder $i &
#   mv images $result_folder/$current_date/$current_time/images_$i
#   mv videos $result_folder/$current_date/$current_time/videos_$i
#   mv runtime.txt $result_folder/$current_date/$current_time/runtime_$i.txt
done

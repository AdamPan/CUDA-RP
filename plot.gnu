set term png size 800, 1600 
set output '../images/OUTPUTFILE.png'
set palette gray
#set xrange [0 : 300]
#set yrange [0 : 600]
set size ratio 2
#set title 'OUTPUTFILE'
p 'INPUTFILE.txt' w image

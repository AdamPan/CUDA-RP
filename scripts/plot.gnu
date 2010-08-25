set term png enhanced size XRES, YRES
set output '../images_ITERATOR/OUTPUTFILE.png'
set lmargin 0
set tmargin 0
set rmargin 0
set bmargin 0
set palette gray
set xrange [-XBOUND : XBOUND]
set yrange [0 : YBOUND]
set cbrange [ LOWERRANGE : UPPERRANGE ]
set size ratio 1
set title 'TITLE'
p 'INPUTFILE.txt' w image

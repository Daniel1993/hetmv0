set terminal dumb

# ARG1 --> location of the data
# ARG2 --> output file

set grid ytics
set fit results

# stats <FILENAME_HERE> using <COL_NB_HERE> nooutput
# available: STATS_max, STATS_min (check other measures)

if (ARG2[strlen(ARG2)-2:] eq 'tex') {
    set terminal cairolatex size 7,4
    set output sprintf("PACT19_%s", ARG2)
    set title system(sprintf("echo PACT19_%s | tr '_' '-'", ARG2))
} else { if (ARG2[strlen(ARG2)-2:] eq 'pdf') {
    set terminal pdf size 7,4
    set output sprintf("PACT19_%s", ARG2)
    set title system(sprintf("echo PACT19_%s | tr '_' '-'", ARG2))
} else { if (ARG2[strlen(ARG2)-2:] eq 'jpg') {
    set terminal jpeg enhanced large size 700,400
    set output sprintf("PACT19_%s", ARG2)
    set title system(sprintf("echo PACT19_%s | tr '_' '-'", ARG2))
    # set xtics rotate by 20
} else {
    set terminal pngcairo noenhanced size 700,400
    set output sprintf("PACT19_%s", ARG1)
    set title sprintf("PACT19_%s", ARG1)
}}}

set key outside right

set ylabel "Throughput" font ",14" tc lt 0 #offset 2.7,-0.0
set xlabel "Prec writes (%)" font ",14" tc lt 0 #offset 2.7,-0.0

plot \
     sprintf("%s/VERS_1GPU_STM.avg", ARG1) using 37:18 notitle with linespoints linecolor rgbcolor "#FF0000"      pt 1 lw 2 ps 0.6, \
     sprintf("%s/VERS_1GPU_STM.avg", ARG1) using 37:18:55 notitle with errorbars linecolor rgbcolor "#FF0000"      pt 1 lw 2 ps 0.6, \
     sprintf("%s/VERS_1GPU_STM.avg", ARG1) using 37:15 notitle with linespoints linecolor rgbcolor "#FF0000" dt 2 pt 1 lw 2 ps 0.4, \
     sprintf("%s/VERS_1GPU_STM.avg", ARG1) using 37:15:52 notitle with errorbars linecolor rgbcolor "#FF0000" dt 2 pt 1 lw 2 ps 0.4, \
     sprintf("%s/VERS_1GPU_STM.avg", ARG1) using 37:16 notitle with linespoints linecolor rgbcolor "#FF0000" dt 3 pt 1 lw 2 ps 0.4, \
     sprintf("%s/VERS_1GPU_STM.avg", ARG1) using 37:16:53 notitle with errorbars linecolor rgbcolor "#FF0000" dt 3 pt 1 lw 2 ps 0.4, \
     sprintf("%s/CPUonly_STM.avg", ARG1) using 37:18 notitle with linespoints linecolor rgbcolor "#13FF03" pt 6 lw 2 ps 0.6, \
     sprintf("%s/CPUonly_STM.avg", ARG1) using 37:18:55 notitle with errorbars linecolor rgbcolor "#13FF03" pt 6 lw 2 ps 0.6, \
     sprintf("%s/GPUonly_1.avg", ARG1) using 37:18 notitle with linespoints linecolor rgbcolor "#13C3C3" pt 4 lw 2 ps 0.6, \
     sprintf("%s/GPUonly_1.avg", ARG1) using 37:18:55 notitle with errorbars linecolor rgbcolor "#13C3C3" pt 4 lw 2 ps 0.6, \
     1/0 with linespoints linecolor rgbcolor "#FF0000"            pt 1 lw 3 ps 1 ti "HeTM", \
     1/0 with linespoints linecolor rgbcolor "#FF0000"  dt 2 pt 1 lw 3 ps 1 ti "HeTM (CPU)", \
     1/0 with linespoints linecolor rgbcolor "#FF0000"  dt 3 pt 1 lw 3 ps 1 ti "HeTM (GPU)", \
     1/0 with linespoints linecolor rgbcolor "#13FF03"       pt 6 lw 3 ps 1 ti "CPU only", \
     1/0 with linespoints linecolor rgbcolor "#13C3C3"       pt 4 lw 3 ps 1 ti "GPU only" \



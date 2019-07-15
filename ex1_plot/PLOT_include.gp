FOLDER="../data_ex1/" #"`ls -d */ | tail -n 1`"
FOLDER_4R_4W="../data_ex1_W1/" #"`ls -d *4R_4W/ | tail -n 1`"
FOLDER_LARGE_READS="../data_ex1_W2/" #"`ls -d *LARGE_READS/ | tail -n 1`"

NORM_RS_TSX=FOLDER."NORM_inst_rand_sep_TSX.avg"
NORM_RS_STM=FOLDER."NORM_inst_rand_sep_STM.avg"
NORM_RS_GPU=FOLDER."NORM_inst_rand_sep_GPU.avg"
NORM_RS_GPU_10b=FOLDER."NORM_inst_rand_sep_GPU_10b.avg"

NORM_4R_4W_RS_TSX=FOLDER_4R_4W."NORM_inst_rand_sep_TSX.avg"
NORM_LARGE_READS_RS_TSX=FOLDER_LARGE_READS."NORM_inst_rand_sep_TSX.avg"
NORM_4R_4W_RS_STM=FOLDER_4R_4W."NORM_inst_rand_sep_STM.avg"
NORM_LARGE_READS_RS_STM=FOLDER_LARGE_READS."NORM_inst_rand_sep_STM.avg"

NORM_4R_4W_RS_GPU=FOLDER_4R_4W."NORM_inst_rand_sep_GPU.avg"
NORM_4R_4W_RS_GPU_10b=FOLDER_4R_4W."NORM_inst_rand_sep_GPU_10b.avg"
NORM_LARGE_READS_RS_GPU=FOLDER_LARGE_READS."NORM_inst_rand_sep_GPU.avg"
NORM_LARGE_READS_RS_GPU_10b=FOLDER_LARGE_READS."NORM_inst_rand_sep_GPU_10b.avg"

VERS_RS=FOLDER."VERS_rand_sep.avg"
BMAP_RS=FOLDER."BMAP_rand_sep.avg"

COLOR_GRAY1="#C0C0C0"
COLOR_GRAY2="#909090"
COLOR_ROSE1="#FF11FF"
COLOR_ROSE2="#AA22AA"
COLOR_RED1="#FF0000"
COLOR_RED2="#AA1111"
COLOR_GREEN1="#00FF00"
COLOR_GREEN2="#11AA11"
COLOR_BLUE1="#6666FF"
COLOR_BLUE2="#2222AA"
COLOR_ORANGE="#FF8000"

set style line 1 lt rgb COLOR_GRAY1 lw 6 pt 4 ps 2.6
set style line 2 lt rgb COLOR_GRAY2 lw 6 pt 5 ps 2.6
set style line 3 lt rgb COLOR_GRAY1 lw 6 pt 5 ps 2.6 dt "."
set style line 4 lt rgb COLOR_GRAY2 lw 6 pt 4 ps 2.6 dt "_"

set style line 5 lt rgb COLOR_ROSE1 lw 6 pt 6 ps 2.6
set style line 6 lt rgb COLOR_ROSE2 lw 6 pt 7 ps 2.6
set style line 7 lt rgb COLOR_ROSE1 lw 6 pt 7 ps 2.6 dt "."
set style line 8 lt rgb COLOR_ROSE2 lw 6 pt 6 ps 2.6 dt "_"

set style line 9  lt rgb COLOR_RED1 lw 6 pt 8 ps 2.6
set style line 10 lt rgb COLOR_RED2 lw 6 pt 9 ps 2.6
set style line 11 lt rgb COLOR_RED1 lw 6 pt 9 ps 2.6 dt "."
set style line 12 lt rgb COLOR_RED2 lw 6 pt 8 ps 2.6 dt "_"

set style line 13 lt rgb COLOR_GREEN1 lw 6 pt 10 ps 2.6
set style line 14 lt rgb COLOR_GREEN2 lw 6 pt 11 ps 2.6
set style line 15 lt rgb COLOR_GREEN1 lw 6 pt 11 ps 2.6 dt "."
set style line 16 lt rgb COLOR_GREEN2 lw 6 pt 10 ps 2.6 dt "_"

set style line 17 lt rgb COLOR_BLUE1 lw 6 pt 2 ps 2.6
set style line 18 lt rgb COLOR_BLUE2 lw 6 pt 3 ps 2.6
set style line 19 lt rgb COLOR_BLUE1 lw 6 pt 3 ps 2.6 dt "."
set style line 20 lt rgb COLOR_BLUE2 lw 6 pt 2 ps 2.6 dt "_"

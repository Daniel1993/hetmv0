#!/bin/bash

cd ./bank
./array_prec_write_inst_cost_TX4.sh
mv ./array_prec_inst_cost_TX4 ../data_ex1

cd ../ex1_plot
./concat_prec_write.sh #### TODO: this appends to the data sources DO NOT RUN TWICE!!!
./convertToTSV.sh
./averageAll.sh
./normalize.sh
mv ../data_ex1 ../data_ex1_W1

### ----

cd ./bank
./array_prec_write_inst_cost_LARGE_READS.sh
mv ./array_prec_inst_cost_LARGE_READS ../data_ex1

cd ../ex1_plot
./concat_prec_write.sh #### TODO: this appends to the data sources DO NOT RUN TWICE!!!
./convertToTSV.sh
./averageAll.sh
./normalize.sh
mv ../data_ex1 ../data_ex1_W2

### ----

gnuplot __PLOT__inst_cost.gp
mv __PLOT__inst_cost.pdf plot_ex1.pdf



### a file ./ex1_plot/__PLOT__inst_cost.pdf has been produced, it is the plot

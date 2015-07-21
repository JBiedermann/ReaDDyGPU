
set xlabel "time[ms]"
set ylabel "MSD"
dt = 1e-4 # in seconds


set output "../ReaDDy_output/plot_msd.png"
set term png

plot            "../ReaDDy_output/out_msd.csv" using ($1*dt):2 title column(2) with lines lw 5 linecolor rgb "black" , \
                "../ReaDDy_output/out_msd.csv" using ($1*dt):3 title column(3) with lines lw 5 , \
                "../ReaDDy_output/out_msd.csv" using ($1*dt):4 t column(4)  with lines lw 5 #, \
#		"../ReaDDy_output/out_msd.csv" u ($1*dt):5 t column(5) with lines lw 5, \
#		"../ReaDDy_output/out_msd.csv" u ($1*dt):6 t column(6) with lines lw 5, \
#		"../ReaDDy_output/out_msd.csv" u ($1*dt):7 t column(7) with lines lw 5, \
#		"../ReaDDy_output/out_msd.csv" u ($1*dt):8 t column(8) with lines lw 5, \
#		"../ReaDDy_output/out_msd.csv" u ($1*dt):9 t column(9) with lines lw 5, \
#		"../ReaDDy_output/out_msd.csv" u ($1*dt):10 t column(10) with lines lw 5


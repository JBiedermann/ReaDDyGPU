

set xlabel "time[ms]"
set ylabel "nParticles"
dt = 1e-4 # in seconds

set output "../ReaDDy_output/plot_particleNumbers.png"
set term png

plot            "../ReaDDy_output/out_particleNumbers.csv" using ($1*dt):2 title column(2) with lines lw 5, \
		"../ReaDDy_output/out_particleNumbers.csv" using ($1*dt):3 title column(3) with lines lw 5, \
                "../ReaDDy_output/out_particleNumbers.csv" u ($1*dt):4 t column(4)  with lines lw 5

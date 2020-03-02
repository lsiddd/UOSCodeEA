set view map;
set xlabel "X"
set ylabel "Y"
set cblabel "SINR (dB)"
set term png size 1280, 960
set output "rem-noUABs.png"
unset key
plot "rem-noUABs.out" using ($1):($2):(10*log10($4)) with image

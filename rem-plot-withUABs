set view map;
set xlabel "X"
set ylabel "Y"
set cblabel "SINR (dB)"
set term png size 1280, 960
set output "rem-withUABs.png"
unset key
plot "rem-withUABs.out" using ($1):($2):(10*log10($4)) with image

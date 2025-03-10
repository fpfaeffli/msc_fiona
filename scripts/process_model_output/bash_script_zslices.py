# This bash script can be used in the terminal to make the sigma to z conversion for all the monthly files in the directory.

# cd into the directory where the monthly files are stored

# present = /nfs/sea/work/fpfaeffli/fionas_avg_present
# ssp245 =/nfs/sea/work/fpfaeffli/fionas_avg_ssp245
# ssp585 = /nfs/sea/work/fpfaeffli/fionas_avg_ssp585

# The numbers after zslice are the depth levels

#!/bin/bash
for year in {2010..2021}; do
    for month in {1..12}; do
        month_padded=$(printf "%03d" "$month")
        echo "Executing command for Year: $year, Month: $month_padded"
        zslice 0 -5 -10 -15 -20 -25 -30 -35 -40 -45 -50 -55 -60 -65 -70 -75 -80 -85 -90 -95 -100 --vars=w pactcs30_grd.nc "avg/avg_${year}_${month_padded}.nc"
    done
done


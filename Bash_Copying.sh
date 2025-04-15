#!/bin/bash

path_dst=$1


Sources=(
	Simulations/I0/Activity_Fitting_I0.png
)


Names=(
	Mono_Fits.png
)





for i in "${!Sources[@]}"; do
    basename "${Sources[$i]}"
    f="${Names[$i]}"
    echo $filename
    file_dst="${path_dst}/${f}"
    
    echo $file_dst

    cp "${Sources[$i]}" "$file_dst"
    echo cp "${Sources[$i]}" "$file_dst"
done

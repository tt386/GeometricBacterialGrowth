#!/bin/bash

path_dst=$1


Sources=(
	Simulations/Fig2_Fitting/All_S_Mono.png
	Simulations/Fig2_Fitting/All_P_Mono_Fitting.png

	Simulations/Fig3_Comparisons/S_Mono_Comparisons.png
	Simulations/Fig3_Comparisons/P_Mono_Comparisons.png
)


Names=(
	S_Fits.png
	P_Fits.png

	S_Comparisons.png
	P_Comparisons.png
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

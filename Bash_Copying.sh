#!/bin/bash

path_dst=$1


Sources=(
	Simulations/Fig1/All_S_Mono.png
	Simulations/Fig1/All_P_Mono_Fitting.png

	Simulations/Fig2/S_Mono_Comparisons.png
	Simulations/Fig2/P_Mono_Comparisons.png

	Simulations/Fig2/S_Mono_KineticComparisons.png
	Simulations/Fig2/P_Mono_KineticComparisons.png
)


Names=(
	S_Fits.png
	P_Fits.png

	S_Comparisons.png
	P_Comparisons.png

	S_Kinetic_Comparison.png
	P_Kinetic_Comparison.png
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

folders=("A" "B" "F")
for folder in "${folders[@]}" 
do 
    for file in $folder/*.vrp; do
            cmd="python3 genetic_algorithm.py final_comparison_19_02_2025 $file 150 5 0.12 0.25 2x 0"
            echo $cmd
            sh -c "$cmd 1" >> output &
            sh -c "$cmd 2" >> output 
            sh -c "$cmd 3" >> output &
            sh -c "$cmd 4" >> output 
            sh -c "$cmd 5" >> output 
        done
    done
done
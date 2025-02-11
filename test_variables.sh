#!/bin/bash

# Problem Set
problems=("A/A-n32-k5.vrp" "A/A-n80-k10.vrp" "A/A-n46-k7.vrp" "B/B-n78-k10.vrp"  "B/B-n50-k8.vrp" )
pop_size=(20 80 150) 
tournament_size=(5 15) # Grau de competitividade para reprodução
elitism_factor=(0.12 0.25 ) # Taxa de preservação da geração anterior
mutation_rate=(0.1 0.25 0.4) 
crossover_rate=(0.5 0.75) # Taxa de preservação da estrutura do parent1 no novo individuo
crossover_type=("2x" "ux")

for ts in "${tournament_size[@]}"
do for mr in "${mutation_rate[@]}"
do for problem in "${problems[@]}"
do for ps in "${pop_size[@]}"
do for ef in "${elitism_factor[@]}"
do for ct in "${crossover_type[@]}" 
do 
    if [ "$ct" = "ux" ]
    then
        for cr in "${crossover_rate[@]}"
        do 
            echo "python3 genetic_algorithm.py $problem $ps $ts $ef $mr $ct $cr"

            python3 genetic_algorithm.py "$problem" "$ps" "$ts" "$ef" "$mr" "$ct" "$cr" >> output
        done
    else
        echo "python3 genetic_algorithm.py $problem $ps $ts $ef $mr $ct $cr 0"
        python3 genetic_algorithm.py "$problem" "$ps" "$ts" "$ef" "$mr" "$ct" 0 >> output
    fi
done
done
done
done
done
done
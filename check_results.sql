SELECT instance, best_solution, best_fitness, best_fitness_time, avg_gen_time 
FROM final_comparison_19_02_2025 WHERE "instance" = 'A/A-n32-k5.vrp' LIMIT 100



SELECT instance, best_solution, avg(best_fitness), group_concat(' ' || best_fitness) as all_best_fitness, avg(best_fitness_time) 
FROM final_comparison_19_02_2025 
GROUP BY instance ORDER BY "instance" asc LIMIT 100   
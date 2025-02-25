SELECT instance, best_solution, best_fitness, best_fitness_time, avg_gen_time 
FROM final_comparison_19_02_2025 WHERE "instance" = 'A/A-n32-k5.vrp' LIMIT 100



SELECT instance,min(best_fitness) as "Fmh min", avg(best_fitness) as "Fmh medio", 
min(best_fitness_time) as "Tempo min", avg(best_fitness_time) as "Tempo medio",
"" as "Gap min", "" as "Gap medio",
group_concat(' ' || best_fitness) as all_best_fitness,
best_solution 
FROM final_comparison_19_02_2025 
GROUP BY instance ORDER BY "instance" asc


SELECT * FROM final_comparison_19_02_2025 WHERE instance = 'A/A-n63-k10.vrp' ORDER BY best_fitness asc LIMIT 100
-- FIND BEST VALUES
WITH "1st_quartil_best_fitness_by_instance" AS (
    SELECT instance, MIN(best_fitness) + (avg(best_fitness) - MIN(best_fitness))/2 AS "1st_quartile_best_fitness"
    FROM "entries_desktop_INF_tolerance_13_02_25"
    GROUP BY instance
)
SELECT count(*), crossover
FROM "entries_desktop_INF_tolerance_13_02_25" e
JOIN "1st_quartil_best_fitness_by_instance" a ON e.instance = a.instance
WHERE e.best_fitness < a."1st_quartile_best_fitness"
GROUP BY e.crossover
ORDER BY a.instance DESC

/*Selected 
    pop_size: 150
    tounament_size: 5
    crossover: 2x
    mutation_rate: 0.25
    elitism_factor: 0.12
*/

-- COMPARE AVG GEN TIME
SELECT avg_gen_time, crossover, instance
FROM "entries_desktop_INF_tolerance_13_02_25"
GROUP BY instance, crossover

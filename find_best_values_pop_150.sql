WITH "1st_quartil_best_fitness_by_instance" AS (
    SELECT instance, MIN(best_fitness) + (avg(best_fitness) - MIN(best_fitness))/2 AS "1st_quartile_best_fitness"
    FROM "entries_desktop_INF_tolerance_13_02_25"
    WHERE pop_size = 150
    GROUP BY instance
)
SELECT count(*), tournament_size
FROM "entries_desktop_INF_tolerance_13_02_25" e
JOIN "1st_quartil_best_fitness_by_instance" a ON e.instance = a.instance
WHERE e.best_fitness < a."1st_quartile_best_fitness" and pop_size = 150
GROUP BY e.tournament_size
ORDER BY a.instance DESC
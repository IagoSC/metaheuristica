import sqlite3

# CONSTANTS
TABLE_NAME = 'entries_avg_gen'
FILE_NAME = 'variable_calibration.db'

con = sqlite3.connect(FILE_NAME)
cur = con.cursor()
cur.execute(f"""
    CREATE TABLE IF NOT EXISTS {TABLE_NAME}(
            uuid VARCHAR PRIMARY KEY,
            instance VARCHAR,
            pop_size int,
            tournament_size int,
            elitism_factor float,
            mutation_rate float,
            crossover VARCHAR,
            best_solution VARCHAR,
            best_fitness float,
            best_fitness_time float,
            avg_gen_time float
);""")


def insert_entry(instance, args, best_solution, best_fitness, best_fitness_time, parameters, avg_gen_time):
    query = f"""
        INSERT OR REPLACE INTO {TABLE_NAME}(uuid, instance, best_fitness, best_fitness_time, pop_size, tournament_size, elitism_factor, mutation_rate, crossover, best_solution, avg_gen_time)
        VALUES(
            "{instance}_{'_'.join(args)}",
            "{instance}",
            {best_fitness},
            {best_fitness_time},
            {parameters['population_size']},
            {parameters['tournament_size']},
            {parameters['elitism_factor']},
            {parameters['mutation_rate']},
            "{parameters['crossover_type']}{parameters['crossover_ux_rate']}",
            \"{best_solution}\",
            {avg_gen_time} 
        );
                """
    print(query)
    cur.execute(query)
    con.commit()
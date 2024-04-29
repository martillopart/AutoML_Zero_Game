import subprocess
from converter import generate_protobuf_instructions
import re
import os

def evaluate(code_string):
    # generate alg
    alg = generate_protobuf_instructions(code_string)

    # get the location directory of this script
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # create shell script
    command = f'''
    cd "{script_dir}"; \\
    bazel run -c opt :run_evaluation_experiment -- \\
    --algorithm="\\
    {alg}
    " \\
    --evaluation_tasks=" \\
        tasks {{ \\
        scalar_linear_regression_task {{}} \\
        features_size: 4 \\
        num_train_examples: 1000 \\
        num_valid_examples: 100 \\
        num_tasks: 100 \\
        eval_type: RMS_ERROR \\
        data_seeds: [1000000] \\
        param_seeds: [2000000] \\
        }}"
    '''

    # execute shell
    result = subprocess.run(command, shell=True, check=True, text=True, stdout=subprocess.PIPE)

    # find "Evaluation fitness" value
    match = re.search(r'Evaluation fitness: (\d+\.\d+)', result.stdout)
    if match:
        evaluation_fitness = float(match.group(1))
        print(f'Evaluation fitness is: {evaluation_fitness}')
        return evaluation_fitness
    else:
        print('Evaluation fitness value not found.')
        print('The output is:')
        print(result.stdout)
        return 0


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        file = '/dev/stdin'
    else:
        file = sys.argv[1]
    with open(file, 'r') as f:
        code_str = f.read()

    print(evaluate(code_str), "\n")
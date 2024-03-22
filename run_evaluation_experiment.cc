#include <iostream>
#include <memory>

#include "algorithm.h"
#include "algorithm.pb.h"
#include "evaluator.h"
#include "experiment_util.h"
#include "task.pb.h"
#include "task_util.h"
#include "definitions.h"
#include "instruction.pb.h"
#include "google/protobuf/text_format.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"

ABSL_FLAG(std::string, algorithm, "", "Algorithm to evaluate. Must be a Algorithm proto in text-format. Required.");
ABSL_FLAG(std::string, evaluation_tasks, "", "The tasks to use for the evaluation. Must be a TaskCollection proto in text format. Required.");

namespace automl_zero {

void EvaluateAlgorithm() {
  CHECK(!absl::GetFlag(FLAGS_algorithm).empty());
  Algorithm algorithm;
  algorithm.FromProto(ParseTextFormat<automl_zero::SerializedAlgorithm>(absl::GetFlag(FLAGS_algorithm)));

  CHECK(!absl::GetFlag(FLAGS_evaluation_tasks).empty());
  auto evaluation_tasks = ParseTextFormat<TaskCollection>(absl::GetFlag(FLAGS_evaluation_tasks));

  // Set up evaluator
  std::mt19937 bit_gen(/*random_seed=*/123); // Consider making the seed a flag if you want reproducibility
  RandomGenerator rand_gen(&bit_gen);
  Evaluator evaluator(
      MEAN_FITNESS_COMBINATION, // Or another fitness combination mode
      evaluation_tasks,
      &rand_gen,
      nullptr, // functional_cache
      nullptr, // train_budget
      100.0); // Adjust based on your task


  std::cout << "Algorithm: " << std::endl << algorithm.ToReadable() << std::endl;
       
  // Evaluate the algorithm
  const double fitness = evaluator.Evaluate(algorithm);
  std::cout << "Evaluation fitness: " << fitness << std::endl;
}

}  // namespace automl_zero

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  automl_zero::EvaluateAlgorithm();
  return 0;
}

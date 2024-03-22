bazel run -c opt :run_evaluation_experiment -- \
  --algorithm=" \
    setup_instructions { \
      op: VECTOR_INNER_PRODUCT_OP \
      in1: 0 \
      in2: 2 \
      out: 2 \
    } \
    setup_instructions { \
      op: SCALAR_PRODUCT_OP \
      in1: 1 \
      in2: 0 \
      out: 1 \
    } \
    setup_instructions { \
      op: SCALAR_CONST_SET_OP \
      out: 3 \
      activation_data: 0.178742 \
    } \
    setup_instructions { \
      op: SCALAR_CONST_SET_OP \
      out: 3 \
      activation_data: -0.241964 \
    } \
    setup_instructions { \
      op: SCALAR_DIFF_OP \
      in1: 2 \
      in2: 1 \
      out: 3 \
    } \
    setup_instructions { \
      op: SCALAR_PRODUCT_OP \
      in1: 3 \
      in2: 1 \
      out: 1 \
    } \
    setup_instructions { \
      op: SCALAR_CONST_SET_OP \
      out: 3 \
      activation_data: -0.176841 \
    } \
    setup_instructions { \
      op: SCALAR_PRODUCT_OP \
      in1: 1 \
      in2: 3 \
      out: 1 \
    } \
    setup_instructions { \
      op: SCALAR_VECTOR_PRODUCT_OP \
      in1: 2 \
      in2: 0 \
      out: 1 \
    } \
    predict_instructions { \
      op: VECTOR_INNER_PRODUCT_OP \
      in1: 0 \
      in2: 1 \
      out: 2 \
    } \
    predict_instructions { \
      op: SCALAR_PRODUCT_OP \
      in1: 3 \
      in2: 2 \
      out: 1 \
    } \
    learn_instructions { \
      op: SCALAR_DIFF_OP \
      in1: 1 \
      in2: 0 \
      out: 1 \
    } \
    learn_instructions { \
      op: SCALAR_VECTOR_PRODUCT_OP \
      in1: 0 \
      in2: 1 \
      out: 2 \
    } \
    learn_instructions { \
      op: SCALAR_VECTOR_PRODUCT_OP \
      in1: 1 \
      in2: 0 \
      out: 2 \
    } \
    learn_instructions { \
      op: SCALAR_DIFF_OP \
      in1: 1 \
      in2: 3 \
      out: 1 \
    } \
    learn_instructions { \
      op: VECTOR_INNER_PRODUCT_OP \
      in1: 2 \
      in2: 2 \
      out: 2 \
    } \
    learn_instructions { \
      op: SCALAR_SUM_OP \
      in1: 2 \
      in2: 2 \
      out: 2 \
    }" \
  --evaluation_tasks=" \
    tasks { \
      scalar_linear_regression_task {} \
      features_size: 4 \
      num_train_examples: 1000 \
      num_valid_examples: 100 \
      num_tasks: 100 \
      eval_type: RMS_ERROR \
      data_seeds: [1000000] \
      param_seeds: [2000000] \
    }"

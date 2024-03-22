import re

def generate_protobuf_instructions(code_str):
    lines = code_str.strip().split('\n')
    instructions = []

    # Mapping from code operation to protobuf operation names.
    op_mapping = {
        'dot': 'VECTOR_INNER_PRODUCT_OP',
        '*': 'SCALAR_PRODUCT_OP',
        '*': 'SCALAR_PRODUCT_OP',
        '-': 'SCALAR_DIFF_OP',
        '+': 'VECTOR_SUM_OP'
    }

    instruction_type = "setup_instructions"

    for line in lines:
        line = line.strip()
        if '=' not in line:
            if "Setup" in line:
                instruction_type = "setup_instructions"
            elif "Predict" in line:
                instruction_type = "predict_instructions"
            elif "Learn" in line:
                instruction_type = "learn_instructions"
            continue

        # Splitting the line into output variable and expression parts.
        output, expression = line.split('=')
        output = output.strip()
        expression = expression.strip()

        # Handling different types of expressions.
        if 'dot' in expression:
            # Extract operands for dot product.
            operands = expression.split('(')[1].split(')')[0].split(',')
            v1, v2 = operands[0].strip(), operands[1].strip()
            op_code = 'VECTOR_INNER_PRODUCT_OP'
            instructions.append(f"{instruction_type} {{ op: {op_code} in1: {v1[-1]} in2: {v2[-1]} out: {output[1]} }}")
        elif len(re.findall(r'\w+|\S', expression)) == 3 and any(op in expression for op in ['*', '-', '+']):
            # Extract operands for binary operations.
            expression = re.findall(r'\w+|\S', expression)
            op = expression[1]
            op_code = op_mapping[op]
            in1, in2 = expression[0], expression[2]
            if op == "*" and in1.startswith('s') and in2.startswith('v'):
                op_code = "SCALAR_VECTOR_PRODUCT_OP"
            instructions.append(f"{instruction_type} {{ op: {op_code} in1: {in1.strip()[1]} in2: {in2.strip()[1]} out: {output[1]} }}")
        else:
            # Handling constant set operation.
            op_code = 'SCALAR_CONST_SET_OP'
            instructions.append(f"{instruction_type} {{ op: {op_code} out: {output[1]} activation_data: {expression} }}")

    return ' \\\n'.join(instructions) + ' \\'

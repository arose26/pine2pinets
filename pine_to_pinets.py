import json
from lark import Lark, Transformer, v_args, Token
import escodegen # Requires: pip install escodegen-py lark-parser

# --- ESTree Node Helper (Simplified - No Location/Comments) ---
def estree_node(node_type, **kwargs):
    node = {'type': node_type}
    # Filter out None values
    node.update({k: v for k, v in kwargs.items() if v is not None})
    return node

# --- Lark Transformer to build ESTree AST (Simplified) ---
class PineScriptToEstree(Transformer):
    def __init__(self):
        super().__init__()
        self._declared_vars = set()

    # --- Terminal Handling (Keep These) ---
    @v_args(inline=True)
    def number_literal(self, value_token):
        raw_val = str(value_token)
        try:
            if '.' not in raw_val and 'e' not in raw_val.lower():
                 py_val = int(raw_val)
            else:
                 py_val = float(raw_val)
        except ValueError:
            py_val = float(raw_val)
        return estree_node('Literal', value=py_val, raw=raw_val)

    @v_args(inline=True)
    def string_literal(self, value_token):
        raw_val = str(value_token)
        py_val = raw_val[1:-1] # Remove quotes
        return estree_node('Literal', value=py_val, raw=raw_val)

    @v_args(inline=True)
    def variable_access(self, name_token):
        return estree_node('Identifier', name=str(name_token))

    # --- Parenthesized Expression ---
    @v_args(inline=True)
    def parenthesized_expression(self, expr):
        return expr

    # --- Term Handling (Keep Term and Chain Ops) ---
    def term(self, args):
        # --- REMOVE/COMMENT OUT Debug Print ---
        # print(f"DEBUG: Entering term method with args: {args}")
        # --- End Remove Debug Print ---

        current_node = args[0]
        chain_ops = args[1:]
        for op_result in chain_ops:
            op_type = op_result['op_type']
            if op_type == 'attribute_access':
                prop_name = op_result['property']
                current_node = estree_node('MemberExpression',
                                           object=current_node,
                                           property=estree_node('Identifier', name=prop_name),
                                           computed=False)
            elif op_type == 'historical_access':
                index_expr = op_result['index']
                current_node = estree_node('MemberExpression',
                                           object=current_node,
                                           property=index_expr,
                                           computed=True)
            elif op_type == 'function_call':
                call_args_list = op_result['arguments']
                final_args_js = []
                named_args_props = []
                final_callee_node = current_node
                is_input_call = (final_callee_node['type'] == 'Identifier' and final_callee_node['name'] == 'input')
                input_type_attr = None
                if call_args_list:
                    positional_args_js = call_args_list.get('positional', [])
                    named_args_props = call_args_list.get('named', [])
                    final_args_js.extend(positional_args_js)
                    if is_input_call and positional_args_js:
                        first_arg_node = positional_args_js[0]
                        if first_arg_node['type'] == 'Literal':
                            first_arg_py_value = first_arg_node['value']
                            # --- REMOVE/COMMENT OUT Debug Prints ---
                            # print(f"DEBUG: Checking input arg: value={first_arg_py_value}, type={type(first_arg_py_value)}")
                            # --- End Remove Debug Prints ---
                            if isinstance(first_arg_py_value, bool):
                                # --- REMOVE/COMMENT OUT Debug Prints ---
                                # print(f"DEBUG: Boolean detected! Setting input_type_attr to 'bool'")
                                # --- End Remove Debug Prints ---
                                input_type_attr = 'bool'
                            elif isinstance(first_arg_py_value, int):
                                input_type_attr = 'int'
                            elif isinstance(first_arg_py_value, float):
                                input_type_attr = 'float'
                            if input_type_attr:
                                final_callee_node = estree_node('MemberExpression',
                                                        object=estree_node('Identifier', name='input'),
                                                        property=estree_node('Identifier', name=input_type_attr),
                                                        computed=False)
                    if named_args_props:
                        options_object = estree_node('ObjectExpression', properties=named_args_props)
                        final_args_js.append(options_object)
                current_node = estree_node('CallExpression',
                                           callee=final_callee_node,
                                           arguments=final_args_js)
        return current_node

    # --- Chain Operation Handlers (Keep These) ---
    def attribute_access_op(self, args):
        ident_token = args[0]
        return {'op_type': 'attribute_access', 'property': str(ident_token)}

    def historical_access_op(self, args):
        expr_node = args[0]
        return {'op_type': 'historical_access', 'index': expr_node}

    def function_call_op(self, args):
        arguments_result = args[0] if args else None
        return {'op_type': 'function_call', 'arguments': arguments_result}

    # --- Arguments Handling (Keep These) ---
    def arguments(self, args):
        positional = []
        named = []
        for arg in args:
            if isinstance(arg, dict) and arg.get('type') == 'Property':
                named.append(arg)
            elif isinstance(arg, dict):
                positional.append(arg)
        return {'positional': positional, 'named': named}

    def named_argument(self, args):
        name_token = args[0]
        value_node = args[1]
        return estree_node('Property',
                           key=estree_node('Identifier', name=str(name_token)),
                           value=value_node,
                           kind='init', method=False, shorthand=False, computed=False)

    # --- Operator Mapping (Keep This) ---
    def _map_binary_operator(self, op_token):
        op_map = {'+': '+', '-': '-', '*': '*', '/': '/', '%': '%',
                  '>': '>', '<': '<', '>=': '>=', '<=': '<=', '==': '===', '!=': '!==',
                  'and': '&&', 'or': '||'}
        return op_map.get(str(op_token))

    # --- Binary Operations (Keep These) ---
    def _build_binary_chain(self, args):
        # Args should be [left, op_token, right, op_token, right, ...]
        if len(args) == 1: return args[0]
        assert len(args) % 2 == 1, f"Expected odd number of args for binary chain, got {len(args)}: {args}"
        left = args[0]
        for i in range(1, len(args), 2):
            op_token = args[i]
            operator = self._map_binary_operator(op_token)
            if not operator: raise ValueError(f"Unsupported binary operator: {op_token}")
            right = args[i+1]
            node_type = 'LogicalExpression' if operator in ['&&', '||'] else 'BinaryExpression'
            left = estree_node(node_type, operator=operator, left=left, right=right)
        return left

    def comparison(self, args): return self._build_binary_chain(args)
    def additive(self, args): return self._build_binary_chain(args)
    def multiplicative(self, args): return self._build_binary_chain(args)
    def logical_and(self, args): return self._build_binary_chain(args)
    def logical_or(self, args): return self._build_binary_chain(args)

    # --- Ternary (Keep This) ---
    def ternary(self, args):
        if len(args) == 1: return args[0]
        else:
            test, consequent, alternate = args[0], args[1], args[2]
            return estree_node('ConditionalExpression',
                               test=test, consequent=consequent, alternate=alternate)

    # --- Statements (Adjusted for Simplified Grammar) ---
    def assign_var(self, args):
        # IDENTIFIER _WS* "=" _WS* expression
        # Args: [ Token(IDENTIFIER), {expression_node} ]
        var_name_token = args[0]
        var_name = str(var_name_token)
        js_value = args[1]
        identifier_node = estree_node('Identifier', name=var_name)

        if var_name not in self._declared_vars:
            self._declared_vars.add(var_name)
            declaration = estree_node('VariableDeclarator', id=identifier_node, init=js_value)
            return estree_node('VariableDeclaration', declarations=[declaration], kind='const')
        else:
             print(f"Warning: Variable '{var_name}' assigned again using '='. Generating assignment expression.")
             assign_expr = estree_node('AssignmentExpression', operator='=', left=identifier_node, right=js_value)
             return estree_node('ExpressionStatement', expression=assign_expr)

    def expr_stmt(self, args):
        # expression
        # Args: [ {expression_node} ]
        expression_node = args[0]
        return estree_node('ExpressionStatement', expression=expression_node)

    def start(self, args):
        # Filter out potential None results if any rules could return that
        body = [stmt for stmt in args if stmt]
        return estree_node('Program', body=body, sourceType='module')

    # --- Add Methods for Explicit Boolean Keywords ---
    @v_args(inline=True)
    def literal_true(self): # Corrected: no extra argument
        return estree_node('Literal', value=True, raw='true')

    @v_args(inline=True)
    def literal_false(self): # Corrected: no extra argument
        return estree_node('Literal', value=False, raw='false')
    # --- End Add Methods ---

# --- Main Execution Logic (Simplified) ---
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python pine_to_pinets.py <input_file>")
        sys.exit(1)

    input_pine_file = sys.argv[1]
    output_js_ast_file = f"{input_pine_file}.js.ast"
    output_js_code_file = f"{input_pine_file}.js"
    grammar_file = "pinescript_grammar.lark"

    try:
        print(f"Reading grammar from {grammar_file}...")
        with open(grammar_file, 'r', encoding='utf-8') as f:
            grammar = f.read()

        print("Creating Lark parser...")
        # Transformer instance is created automatically by Lark now if not passed
        parser = Lark(grammar, parser='lalr', transformer=PineScriptToEstree()) # Pass class

        print(f"Reading PineScript input from {input_pine_file}...")
        with open(input_pine_file, 'r', encoding='utf-8') as f:
            pine_code = f.read()

        print("Parsing PineScript and transforming to ESTree AST...")
        estree_ast = parser.parse(pine_code)

        print(f"Writing JavaScript ESTree AST to {output_js_ast_file}...")
        with open(output_js_ast_file, 'w', encoding='utf-8') as f:
            json.dump(estree_ast, f, indent=2)

        print("Generating JavaScript code from ESTree AST...")
        # Generate without comment option for now
        js_code = escodegen.generate(estree_ast)

        print(f"Writing generated JavaScript code to {output_js_code_file}...")
        with open(output_js_code_file, 'w', encoding='utf-8') as f:
            f.write(js_code)

        print("\nDirect transpilation complete.")
        print(f"JS AST: {output_js_ast_file}")
        print(f"JS Code: {output_js_code_file}")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()


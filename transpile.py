import json
from lark import Lark, Transformer, v_args, Token, Tree
import escodegen
import sys
import os
import textwrap
import re

# --- ESTree Node Helper ---
def estree_node(node_type, **kwargs):
    node = {'type': node_type}
    node.update({k: v for k, v in kwargs.items() if v is not None})
    return node


# --- Helper to get indentation ---
def get_indentation(line):
    """Calculate the indentation level of a line, treating tabs as 4 spaces."""
    # Expand tabs to 4 spaces before calculating indentation
    expanded_line = line.expandtabs(4)
    indentation = 0
    for char in expanded_line:
        if char == ' ':
            indentation += 1
        # Stop counting at the first non-space character
        elif char == '\t': # Should not happen after expandtabs, but good safety check
             # This part might be redundant after expandtabs, but safe to keep
             indentation += 4 # Or your preferred tab width
        else:
            break
    # Assuming 4 spaces per indent level for block detection
    # Return the raw space count; let the caller determine levels if needed
    return indentation

# --- Pre-processor Function --- (Improved else/elif handling)
def add_explicit_end_markers(pine_code):
    lines = pine_code.splitlines()
    processed_lines = []
    block_stack = [] # Stack stores tuples: (indent_level, block_type)

    processed_lines.append("// Preprocessor Start") # Dummy line
    
    for i, line in enumerate(lines):
        stripped_line = line.strip()
        if not stripped_line or stripped_line.startswith('//'):
            processed_lines.append(line)
            continue

        current_indent = get_indentation(line)
        last_indent, last_type = block_stack[-1] if block_stack else (-1, None)

        # --- Detect Dedentation and Add End Markers (Improved) ---
        is_else_or_elif = stripped_line.startswith('else') # Covers 'else' and 'else if'
        
        while block_stack and current_indent <= last_indent:
            # Check if we should close the block or if it's continued by else/elif
            if last_type == "if" and is_else_or_elif and current_indent == last_indent:
                # Don't pop 'if' if the current line is an 'else'/'else if' at the same level
                break 
            
            # Otherwise, pop the block and add the corresponding end marker
            popped_indent, popped_type = block_stack.pop()
            end_marker_indent = ' ' * popped_indent
            if popped_type == "for": processed_lines.append(f"{end_marker_indent}endfor")
            elif popped_type == "if": processed_lines.append(f"{end_marker_indent}endif")
            elif popped_type == "func": processed_lines.append(f"{end_marker_indent}endfunction")
            
            # Update last_indent/last_type after pop for the loop condition
            last_indent, last_type = block_stack[-1] if block_stack else (-1, None)

        processed_lines.append(line) # Add the current line

        # --- Detect Block Starts and Push to Stack (Improved) ---
        potential_block_start = None
        block_start_indent = current_indent # Default to current line indent
        
        # Check for standard block starts at beginning of line
        if stripped_line.startswith('for '): potential_block_start = "for"
        elif stripped_line.startswith('if '): potential_block_start = "if" # Includes 'if' in 'else if'
        elif stripped_line.endswith('=>'): potential_block_start = "func"
        
        # Check for mid-line 'if' block start after assignment
        # Regex: finds = or :=, optional whitespace, then 'if' as a whole word
        match = re.search(r'[:=]\s*if\b', stripped_line)
        if match and not stripped_line.startswith('if '): # Avoid double-counting line starts
            potential_block_start = "if"
            # Use current_indent as the base for the block started mid-line
            block_start_indent = current_indent

        if potential_block_start:
             # Check indentation of the next meaningful line to confirm block start
             next_line_indent = -1
             for j in range(i + 1, len(lines)):
                 next_line = lines[j]; next_stripped = next_line.strip()
                 if not next_stripped or next_stripped.startswith('//'): continue
                 next_line_indent = get_indentation(next_line)
                 break
             
             # If the next line is more indented, it starts a block
             # Compare against block_start_indent determined above
             if next_line_indent > block_start_indent:
                 # Special check for 'else if': don't push if it continues an existing 'if' block
                 if potential_block_start == "if" and stripped_line.startswith('else if'):
                     pass 
                 # Special check: Don't push if it's the 'if' from ':= if' AND the 'if' block is already open
                 # This check might be too simple and needs refinement if nested := if occurs.
                 elif potential_block_start == "if" and match and block_stack and block_stack[-1][1] == "if":
                      pass # Assume it belongs to the existing if block for now
                 else:
                    block_stack.append((block_start_indent, potential_block_start))

    # --- Add remaining end markers at the end of the file --- 
    while block_stack:
        popped_indent, popped_type = block_stack.pop()
        end_marker_indent = ' ' * popped_indent 
        if popped_type == "for": processed_lines.append(f"{end_marker_indent}endfor")
        elif popped_type == "if": processed_lines.append(f"{end_marker_indent}endif")
        elif popped_type == "func": processed_lines.append(f"{end_marker_indent}endfunction")
    
    # Remove the dummy start line
    if processed_lines and processed_lines[0] == "// Preprocessor Start":
        processed_lines.pop(0)
        
    return '\n'.join(processed_lines)
# --- End Pre-processor ---

# --- Lark Transformer (Explicit Markers) ---
class PineScriptToEstree(Transformer):
    def __init__(self):
        super().__init__()
        self._declared_vars = set()

    # --- Program Method --- (Handles statement_nl*)
    def program(self, args):
        body = [stmt for stmt in args if isinstance(stmt, dict) and 'type' in stmt]
        return estree_node('Program', body=body, sourceType='module')

    # --- Statement NL Helper --- (Handles statement_nl rule)
    @v_args(inline=True)
    def statement_nl(self, statement_node):
        return statement_node # Pass through result

    # --- Terminals/Literals/Expressions/Params (Assume methods exist and are correct) ---
    # ... number_literal, string_literal, literal_true, literal_false, variable_access ...
    # ... parenthesized_expression, term, attribute_access_op, historical_access_op ...
    # ... function_call_op, arguments, named_argument, array_elements, array_literal ...
    # ... simple_param, default_param, param_list, comparison, additive, multiplicative ...
    # ... logical_and, logical_or, ternary, unary_expr ...
    # Using simplified versions for brevity in this edit instruction
    @v_args(inline=True)
    def number_literal(self, t): return estree_node('Literal', value=float(t), raw=str(t))
    @v_args(inline=True)
    def string_literal(self, t): s = str(t)[1:-1]; return estree_node('Literal', value=s, raw=str(t))
    @v_args(inline=True)
    def literal_true(self, _): return estree_node('Literal', value=True, raw='true')
    @v_args(inline=True)
    def literal_false(self, _): return estree_node('Literal', value=False, raw='false')
    @v_args(inline=True)
    def variable_access(self, t): return estree_node('Identifier', name=str(t))
    @v_args(inline=True)
    def parenthesized_expression(self, e): return e

    # --- Term and Chain Operators ---
    def term(self, args):
        # args[0] is the initial atom (Identifier, Literal, ParenExpr, Array)
        # args[1:] are the results from chain_op* transformers
        node = args[0]
        for chain_result in args[1:]:
            # Each chain_result is a tuple: (op_type, data)
            op_type, data = chain_result
            if op_type == 'historical_access':
                # data is the expression inside []
                node = estree_node('MemberExpression', object=node, property=data, computed=True, optional=False)
            elif op_type == 'function_call':
                # data is the list of arguments from the 'arguments' transformer
                node = estree_node('CallExpression', callee=node, arguments=data, optional=False)
            elif op_type == 'attribute_access':
                # data is the Identifier node for the attribute
                node = estree_node('MemberExpression', object=node, property=data, computed=False, optional=False)
            else:
                 raise ValueError(f"Unknown chain_op type: {op_type}")
        return node

    @v_args(inline=True)
    def historical_access_op(self, expression_node):
        # Returns data for term transformer
        return ('historical_access', expression_node)

    @v_args(inline=True)
    def function_call_op(self, arguments_list):
        # Returns data for term transformer
        # arguments_list comes from the 'arguments' transformer
        return ('function_call', arguments_list or []) # Handle no-arg calls

    @v_args(inline=True)
    def attribute_access_op(self, identifier_token):
         # identifier_token is the raw IDENTIFIER token from the grammar rule DOT IDENTIFIER
         # We need to create the Identifier node here.
        property_name = str(identifier_token)
        property_node = estree_node('Identifier', name=property_name)
         # Returns data for term transformer
        return ('attribute_access', property_node)

    # --- Arguments for Function Calls ---
    def arguments(self, args):
         # args is a list of expression nodes and potentially named_argument results
         # Filter out any potential commas or whitespace tokens if grammar allows them
         # Extract the value from named arguments if present
         processed_args = []
         for arg in args:
             if isinstance(arg, dict) and 'type' in arg:
                 processed_args.append(arg) # Regular expression node
             elif isinstance(arg, tuple) and arg[0] == 'named_argument':
                 processed_args.append(arg[1]) # Extract value from named arg tuple
         return processed_args

    # Added transformer for named arguments
    @v_args(inline=True)
    def named_argument(self, identifier_token, assign_op_token, expression_node):
        # assign_op_token is ignored, but needed to match grammar children
        # Return a tuple indicating named arg and its value node
        # We are currently ignoring the name (identifier_token) for JS translation
        return ('named_argument', expression_node)

    # --- Array Literal Handling ---
    def array_elements(self, children):
        # Grammar: expression (COMMA expression)*
        # Children: interleaved list of expression results (dict) and COMMA tokens
        return [node for node in children if isinstance(node, dict) and 'type' in node]

    @v_args(inline=True)
    def array_literal(self, elements_list):
        # Grammar: "[" [array_elements] "]" -> array_literal
        # elements_list is the list returned by array_elements transformer
        # Handle optional empty array: elements_list might be None if grammar was `[array_elements]?`
        # or lark might pass a special marker. Assuming it's a list or None.
        processed_elements = elements_list if elements_list is not None else []
        return estree_node('ArrayExpression', elements=processed_elements)

    # --- Binary Operation Methods ---
    def _map_binary_operator(self, op_token): # Restore this helper
        op_map = {
            '+': '+', '-': '-', '*': '*', '/': '/', '%': '%',
            '>': '>', '<': '<', '>=': '>=', '<=': '<=',
            '==': '===', '!=': '!==', # Strict equality
            'and': '&&', 'or': '||'
        }
        op = str(op_token)
        mapped_op = op_map.get(op)
        if mapped_op is None:
            raise ValueError(f"Unsupported binary operator: {op}")
        return mapped_op

    def _build_binary_chain(self, args):
        if len(args) == 1: return args[0]
        left = args[0]
        for i in range(1, len(args), 2):
            op = self._map_binary_operator(args[i])
            right = args[i+1]
            node_type = 'LogicalExpression' if op in ['&&', '||'] else 'BinaryExpression' # Ensure correct indentation
            left = estree_node(node_type, operator=op, left=left, right=right)
        return left

    # RESTORED Methods for expression hierarchy:
    def comparison(self, args): return self._build_binary_chain(args)
    def additive(self, args): return self._build_binary_chain(args)
    def multiplicative(self, args): return self._build_binary_chain(args)
    def logical_and(self, args): return self._build_binary_chain(args)
    def logical_or(self, args): return self._build_binary_chain(args)
    def ternary(self, args): return estree_node('ConditionalExpression', test=args[0], consequent=args[1], alternate=args[2]) if len(args) > 1 else args[0]
    def unary_expr(self, args): # Restore simplified version
        if len(args) == 2:
             op = str(args[0])
             if op not in ['+', '-']: raise ValueError(f"Unary op {op}?!")
             # Only create node for unary minus
             return estree_node('UnaryExpression', operator='-', prefix=True, argument=args[1]) if op == '-' else args[1]
        return args[0] # Only term
    # --- End Binary/Expression Methods ---

    # --- Statements (Adapted for Explicit Markers / statement_nl) ---
    def simple_assignment(self, args):
        var_name = str(args[0]); expr_node = args[-1]
        declarator = estree_node('VariableDeclarator', id=estree_node('Identifier', name=var_name), init=expr_node)
        declaration = estree_node('VariableDeclaration', declarations=[declarator], kind='let')
        self._declared_vars.add(var_name); return declaration

    def identifier_list(self, args):
        return [estree_node('Identifier', name=str(tok)) for tok in args if isinstance(tok, Token) and tok.type == 'IDENTIFIER']

    def destructuring_assignment(self, args):
        id_nodes = next((item for item in args if isinstance(item, list)), [])
        expr_node = args[-1]
        array_pattern = estree_node('ArrayPattern', elements=id_nodes)
        declarator = estree_node('VariableDeclarator', id=array_pattern, init=expr_node)
        declaration = estree_node('VariableDeclaration', declarations=[declarator], kind='let')
        for id_node in id_nodes: self._declared_vars.add(id_node['name'])
        return declaration

    def reassignment_statement(self, args):
        var_name = str(args[0]); expr_node = args[-1]
        assign_expr = estree_node('AssignmentExpression', operator='=', left=estree_node('Identifier', name=var_name), right=expr_node)
        if var_name not in self._declared_vars:
            print(f"WARN: Reassign used before declare: {var_name}")
            self._declared_vars.add(var_name)
            declarator = estree_node('VariableDeclarator', id=estree_node('Identifier', name=var_name), init=expr_node)
            return estree_node('VariableDeclaration', declarations=[declarator], kind='let')
        else:
             return estree_node('ExpressionStatement', expression=assign_expr)

    @v_args(inline=True)
    def expression_statement(self, expression_node):
        return estree_node('ExpressionStatement', expression=expression_node)

    # --- If Statement (Adapted for Explicit Markers / _extract_block_stmts) ---
    def _extract_block_stmts(self, clause_children):
        # clause_children is the list returned by if_clause/else_clause transformer methods
        stmts = [child for child in clause_children if isinstance(child, dict) and 'type' in child]
        return stmts

    def if_clause(self, args):
         # args are results from statement_nl+ within the clause
         return self._extract_block_stmts(args)

    def else_clause(self, args):
         # args are results from statement_nl+ within the clause
         return self._extract_block_stmts(args)

    def if_statement(self, children):
        # Grammar: IF expression if_clause (ELSE IF expression if_clause)* [ELSE else_clause] ENDIF
        conditions = []
        clause_statement_lists = []
        has_final_else = False
        i = 0
        while i < len(children):
            item = children[i]

            if isinstance(item, Token):
                if item.type == 'IF':
                    # Check if it's part of an ELSE IF handled below
                    if i > 0 and isinstance(children[i-1], Token) and children[i-1].type == 'ELSE':
                        i += 1 # Skip this IF, handled by the ELSE block
                        continue 
                    # It's a standalone IF or the first IF
                    if i + 2 < len(children) and isinstance(children[i+1], dict) and isinstance(children[i+2], list):
                        conditions.append(children[i+1]) # Condition
                        clause_statement_lists.append(children[i+2]) # Clause body list
                        i += 3
                    else:
                        raise ValueError(f"Malformed IF structure at {i}: {children}")
                
                elif item.type == 'ELSE':
                    # Check for ELSE IF
                    if i + 3 < len(children) and isinstance(children[i+1], Token) and children[i+1].type == 'IF' \
                       and isinstance(children[i+2], dict) and isinstance(children[i+3], list):
                        conditions.append(children[i+2]) # ELSE IF Condition
                        clause_statement_lists.append(children[i+3]) # ELSE IF Clause body list
                        i += 4 # Skip ELSE, IF, condition, clause list
                    # Check for final ELSE
                    elif i + 1 < len(children) and isinstance(children[i+1], list):
                        clause_statement_lists.append(children[i+1]) # Final ELSE clause list
                        has_final_else = True
                        i += 2 # Skip ELSE, clause list
                    else:
                        raise ValueError(f"Malformed ELSE/ELSE IF structure at {i}: {children}")
                
                elif item.type == 'ENDIF':
                    i += 1 # Move past ENDIF
                
                else:
                    # Skip other unexpected tokens
                    i += 1
            else:
                # Skip expression results or clause lists directly, they are picked up via IF/ELSE logic
                i += 1

        # --- Validation ---
        if not conditions: raise ValueError(f"If stmt missing condition: {children}")
        # Clause list can be empty if the blocks were empty, but the list itself should exist
        num_expected_clauses = len(conditions) + (1 if has_final_else else 0)
        if len(clause_statement_lists) != num_expected_clauses:
             raise ValueError(f"Mismatch cond/clauses in if: Conds={len(conditions)}, HasElse={has_final_else}, FoundClauses={len(clause_statement_lists)}")

        # --- Build nested structure (Corrected Pop Logic) ---
        final_else_block = estree_node('BlockStatement', body=clause_statement_lists.pop()) if has_final_else else None
        alt = final_else_block
        # Iterate through conditions and remaining clauses backwards
        for i in range(len(conditions) - 1, -1, -1):
            current_clause_body = clause_statement_lists[i] # Access corresponding clause body
            consequent_block = estree_node('BlockStatement', body=current_clause_body)
            alt = estree_node('IfStatement', test=conditions[i], consequent=consequent_block, alternate=alt)
        return alt

    # --- For Loop (Adapted for Explicit Markers / statement_nl) ---
    def for_stmt(self, children):
        # Grammar: FOR IDENTIFIER ASSIGN_OP term TO expression (STEP expression)? _NL statement_nl+ ENDFOR
        # Children: Tokens (FOR, ID, TO, STEP?, ENDFOR), Dicts (term, expr, step?, statement_nl results)
        loop_var_token = None; start_expr = None; end_expr = None; step_expr = None
        body_nodes = []

        # 1. Find loop var
        loop_var_token = next((t for t in children if isinstance(t, Token) and t.type == 'IDENTIFIER'), None)
        if loop_var_token is None: raise ValueError(f"For missing var: {children}")
        loop_var_name = str(loop_var_token); loop_var_id = estree_node('Identifier', name=loop_var_name)

        # 2. Extract expressions and body statements (all dicts after loop var)
        all_dicts = [d for d in children if isinstance(d, dict) and 'type' in d]
        step_token_present = any(isinstance(t, Token) and t.type == 'STEP' for t in children)

        if not all_dicts or len(all_dicts) < 2: raise ValueError(f"For missing start/end/body dicts: {all_dicts}")
        start_expr = all_dicts[0]
        end_expr = all_dicts[1]
        body_start_index = 2

        if step_token_present:
            if len(all_dicts) < 3: raise ValueError(f"For STEP missing expr/body dicts: {all_dicts}")
            step_expr = all_dicts[2]
            body_start_index = 3
        else:
            step_expr = estree_node('Literal', value=1, raw='1')

        if len(all_dicts) <= body_start_index: raise ValueError(f"For missing body dicts: {all_dicts}")
        body_nodes = all_dicts[body_start_index:] # The rest are body statements

        # 3. Build ESTree node (Assume logic is correct)
        init = estree_node('VariableDeclaration', declarations=[estree_node('VariableDeclarator', id=loop_var_id, init=start_expr)], kind='let'); self._declared_vars.add(loop_var_name)
        test_op = '<='; update_op = '+='; update_node_op = '++'
        if step_expr.get('type') == 'Literal' and isinstance(step_expr.get('value'), (int, float)) and step_expr.get('value') < 0:
             test_op = '>='; update_op = '-='; update_node_op = '--'
             if step_expr.get('value') != -1: step_expr = estree_node('Literal', value=abs(step_expr['value']), raw=str(abs(step_expr['value'])))
        test = estree_node('BinaryExpression', operator=test_op, left=loop_var_id, right=end_expr)
        is_simple_inc = step_expr.get('type') == 'Literal' and step_expr.get('value') == 1 and update_node_op == '++'
        is_simple_dec = step_expr.get('type') == 'Literal' and step_expr.get('value') == -1 and update_node_op == '--'
        update = estree_node('UpdateExpression', operator=update_node_op, argument=loop_var_id, prefix=False) if (is_simple_inc or is_simple_dec) else estree_node('AssignmentExpression', operator=update_op, left=loop_var_id, right=step_expr)
        body = estree_node('BlockStatement', body=body_nodes)
        return estree_node('ForStatement', init=init, test=test, update=update, body=body)

    # --- Function Definition Parameters ---
    def simple_param(self, children):
        # Grammar: param: IDENTIFIER -> simple_param
        # Children: [Token(IDENTIFIER)]
        return estree_node('Identifier', name=str(children[0]))

    def default_param(self, children):
        # Grammar: param: IDENTIFIER ... -> default_param
        # Children: [Token(IDENTIFIER), Token(ASSIGN_OP), expression_node]
        identifier_node = estree_node('Identifier', name=str(children[0]))
        default_value_node = children[2]
        return estree_node('AssignmentPattern', left=identifier_node, right=default_value_node)

    def parameter_list(self, children):
        # Grammar: parameter ( _WS* COMMA _WS* parameter )*
        # Children: list of results from parameter/default_parameter, possibly COMMA tokens
        # Filter out any non-dict items (like COMMA tokens)
        return [node for node in children if isinstance(node, dict) and 'type' in node]

    # --- Function Definition (Adapted for Explicit Markers / statement_nl) ---
    def func_def_block(self, children):
        # Remove debug prints

        func_name_token = None
        param_nodes = []
        body_stmts = []
        arrow_token_index = -1

        # Find ARROW token index
        for i, item in enumerate(children):
            if isinstance(item, Token) and item.type == 'ARROW':
                arrow_token_index = i; break
        if arrow_token_index == -1: raise ValueError(f"Func def block missing ARROW: {children}")

        # Extract func name (first IDENTIFIER)
        if isinstance(children[0], Token) and children[0].type == 'IDENTIFIER':
             func_name_token = children[0]
        else: raise ValueError(f"Func def block missing name: {children}")

        # Extract parameters (before ARROW)
        param_list_tree_found = False
        for item in children[1:arrow_token_index]: # Skip func name, look before arrow
            # Case 1: Found the param_list Tree (process its children)
            if isinstance(item, Tree) and item.data == 'param_list':
                param_nodes.extend(self.parameter_list(item.children))
                param_list_tree_found = True
                break # Assume only one param_list tree
        
        # Case 2: No param_list Tree found, look for individual param nodes directly
        # (Handles single param case or potential Lark simplification)
        if not param_list_tree_found:
             for item in children[1:arrow_token_index]:
                 if isinstance(item, dict) and item.get('type') in ['Identifier', 'AssignmentPattern']:
                     param_nodes.append(item)

        # Extract body statements (after ARROW, before ENDFUNCTION - same as before)
        endfunction_token_index = len(children)
        for i in range(len(children) -1, arrow_token_index, -1):
             if isinstance(children[i], Token) and children[i].type == 'ENDFUNCTION':
                 endfunction_token_index = i; break
        for i in range(arrow_token_index + 1, endfunction_token_index):
            item = children[i]
            if isinstance(item, dict) and 'type' in item: body_stmts.append(item)

        if not body_stmts: raise ValueError(f"Func def block missing body: {children}")
        func_name = str(func_name_token)

        # --- Implicit Return Logic (same as before) --- 
        last_stmt_node = body_stmts[-1]; ret_expr = None
        if last_stmt_node.get('type') == 'ExpressionStatement': ret_expr = last_stmt_node.get('expression')
        elif last_stmt_node.get('type') == 'VariableDeclaration':
            if last_stmt_node.get('declarations') and len(last_stmt_node['declarations']) == 1:
                 declarator = last_stmt_node['declarations'][0]
                 if declarator.get('type') == 'VariableDeclarator' and declarator.get('id'): ret_expr = declarator.get('id')
        if ret_expr is not None:
             final_body = body_stmts[:-1]; final_body.append(estree_node('ReturnStatement', argument=ret_expr))
        else: final_body = body_stmts
        block = estree_node('BlockStatement', body=final_body)

        # --- Create Arrow Function (same as before) --- 
        arrow_props = {'id': None, 'params': param_nodes, 'body': block, 'expression': False, 'generator': False, 'async': False}
        arrow_func = estree_node('ArrowFunctionExpression', **arrow_props)
        declaration = estree_node('VariableDeclaration', declarations=[estree_node('VariableDeclarator', id=estree_node('Identifier', name=func_name), init=arrow_func)], kind='const')
        self._declared_vars.add(func_name)
        return declaration

    def function_def_expr(self, children):
        # Apply similar combined logic for parameter extraction
        func_name_token = None; param_nodes = []; expr_node = None; arrow_token_index = -1

        # Find ARROW token index
        for i, item in enumerate(children): 
             if isinstance(item, Token) and item.type == 'ARROW': arrow_token_index = i; break
        if arrow_token_index == -1: raise ValueError(f"Func def expr missing ARROW: {children}")

        # Extract func name (first IDENTIFIER)
        if isinstance(children[0], Token) and children[0].type == 'IDENTIFIER': func_name_token = children[0]
        else: raise ValueError(f"Func def expr missing name: {children}")

        # Extract parameters (before ARROW - Combined Logic)
        param_list_tree_found = False
        for item in children[1:arrow_token_index]:
            if isinstance(item, Tree) and item.data == 'param_list':
                param_nodes.extend(self.parameter_list(item.children))
                param_list_tree_found = True; break
        if not param_list_tree_found:
             for item in children[1:arrow_token_index]:
                 if isinstance(item, dict) and item.get('type') in ['Identifier', 'AssignmentPattern']:
                     param_nodes.append(item)
        
        # Find the expression node (should be the last dict after ARROW)
        for i in range(arrow_token_index + 1, len(children)):
             item = children[i]
             if isinstance(item, dict) and 'type' in item: expr_node = item; break # Assume first dict after arrow is the body

        if expr_node is None: raise ValueError(f"Func def expr missing expression body: {children}")
        func_name = str(func_name_token)

        # Create BlockStatement with return (same as before)
        block = estree_node('BlockStatement', body=[estree_node('ReturnStatement', argument=expr_node)])
        
        # Create Arrow Function (same as before)
        arrow_props = {'id': None, 'params': param_nodes, 'body': block, 'expression': False, 'generator': False, 'async': False}
        arrow_func = estree_node('ArrowFunctionExpression', **arrow_props)
        declaration = estree_node('VariableDeclaration', declarations=[estree_node('VariableDeclarator', id=estree_node('Identifier', name=func_name), init=arrow_func)], kind='const')
        self._declared_vars.add(func_name)
        return declaration

    # --- Helper to extract value from last statement --- 
    def _extract_final_expression_node(self, statement_list):
        """Given a list of ESTree statement nodes, find the value of the last one."""
        if not statement_list: 
             return estree_node('Literal', value=None, raw='null') # Or raise error?
        last_stmt_node = statement_list[-1]
        if last_stmt_node.get('type') == 'ExpressionStatement':
            return last_stmt_node.get('expression')
        elif last_stmt_node.get('type') == 'VariableDeclaration':
            if last_stmt_node.get('declarations') and len(last_stmt_node['declarations']) == 1:
                 declarator = last_stmt_node['declarations'][0]
                 if declarator.get('type') == 'VariableDeclarator' and declarator.get('id'):
                     # Return the Identifier node of the declared var
                     return declarator.get('id') 
        # Add other cases if needed (e.g., ReturnStatement explicitly used?)
        # Default to null/undefined if last statement isn't an expression or simple declaration
        return estree_node('Literal', value=None, raw='null')

    # --- If Expression (Ternary) ---
    def expression_clause(self, args):
        # Grammar: _NL statement_nl+
        # Similar to if_clause/else_clause, but returns the *value* of the last statement
        block_stmts = self._extract_block_stmts(args)
        return self._extract_final_expression_node(block_stmts)

    def if_expr(self, children):
        # Grammar: IF expression expression_clause (ELSE IF expression expression_clause)* [ELSE expression_clause] ENDIF
        # Children: Interleaved list of Tokens (IF, ELSE, ENDIF), expression results (dict), expression_clause results (dict - the value node)
        conditions = []
        clause_values = [] # Store the value nodes returned by expression_clause
        has_final_else = False
        i = 0
        while i < len(children):
            item = children[i]
            if isinstance(item, Token):
                if item.type == 'IF':
                    # Part of ELSE IF?
                    if i > 0 and isinstance(children[i-1], Token) and children[i-1].type == 'ELSE':
                        i += 1; continue
                    # Standalone IF
                    if i + 2 < len(children) and isinstance(children[i+1], dict) and isinstance(children[i+2], dict):
                        conditions.append(children[i+1]) # Condition node
                        clause_values.append(children[i+2]) # Value node from expression_clause
                        i += 3
                    else: raise ValueError(f"Malformed IF structure in if_expr at {i}: {children}")
                elif item.type == 'ELSE':
                    # ELSE IF?
                    if i + 3 < len(children) and isinstance(children[i+1], Token) and children[i+1].type == 'IF' \
                       and isinstance(children[i+2], dict) and isinstance(children[i+3], dict):
                        conditions.append(children[i+2]) # ELSE IF Condition node
                        clause_values.append(children[i+3]) # ELSE IF Value node
                        i += 4
                    # Final ELSE?
                    elif i + 1 < len(children) and isinstance(children[i+1], dict):
                        clause_values.append(children[i+1]) # Final ELSE value node
                        has_final_else = True
                        i += 2
                    else: raise ValueError(f"Malformed ELSE/ELSE IF structure in if_expr at {i}: {children}")
                elif item.type == 'ENDIF':
                    i += 1
                else: i += 1 # Skip other tokens
            else: i += 1 # Skip non-tokens

        # --- Validation ---
        if not conditions: raise ValueError(f"if_expr missing condition: {children}")
        num_expected_clauses = len(conditions) + (1 if has_final_else else 0)
        if len(clause_values) != num_expected_clauses:
             raise ValueError(f"Mismatch cond/clauses in if_expr: Conds={len(conditions)}, HasElse={has_final_else}, FoundVals={len(clause_values)}")

        # --- Build nested ConditionalExpression (ternary) ---
        final_else_value = clause_values.pop() if has_final_else else estree_node('Literal', value=None, raw='null') # Default to null/undefined if no else
        alt = final_else_value
        for i in range(len(conditions) - 1, -1, -1):
            consequent_value = clause_values[i]
            alt = estree_node('ConditionalExpression', test=conditions[i], consequent=consequent_value, alternate=alt)
        return alt

# --- Load Grammar ---
grammar_file = "./pinescript_grammar.lark"
with open(grammar_file, 'r', encoding='utf-8') as f:
    grammar = f.read()

# --- Create Parser (NO postlex, default LALR) ---
parser = Lark(grammar) # Use default LALR

# --- Instantiate Transformer ---
transformer_instance = PineScriptToEstree()

# --- Transpile function ---
def transpile(pine_code):
    print("--- Reading Grammar ---")
    with open(grammar_file, 'r') as f:
        grammar = f.read()
    print("--- Grammar Read ---")

    print("\n--- Pre-processing Pine Code ---")
    processed_code = add_explicit_end_markers(pine_code)
    print("--- Pre-processing Complete ---")
    print("\n--- Processed Pine Code (for Debug) ---")
    print(processed_code)
    print("---------------------------------------")

    try:
        print("\n--- Parsing Code ---")
        parse_tree = parser.parse(processed_code)
        estree_ast = transformer_instance.transform(parse_tree)
        js_code = escodegen.generate(estree_ast)
        return js_code
    except Exception as e:
        print(f"\nAn error occurred during transpilation: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

# --- Main Execution Logic (Calls Pre-processor) ---
if __name__ == "__main__":
    pine_code = ""
    input_filepath = ""
    if len(sys.argv) == 3 and sys.argv[1] == "-f":
        input_filepath = sys.argv[2]
    else:
        print("Usage: transpile -f <path>")
        sys.exit(1)

    if not os.path.isfile(input_filepath):
        print(f"Error: Input file not found: {input_filepath}")
        sys.exit(1)

    try:
        with open(input_filepath, 'r', encoding='utf-8') as f:
            pine_code = f.read()
        if pine_code.startswith('\ufeff'): pine_code = pine_code[1:] # Remove BOM
    except Exception as e:
        print(f"Error reading file {input_filepath}: {e}")
        sys.exit(1)

    if not pine_code:
        print("Error: No PineScript code read from file.")
        sys.exit(1)

    try:
        js_code = transpile(pine_code) # Use processed code

        print("--- Generated JS ---")
        print(js_code)
    except Exception as e:
        print(f"\nAn error occurred during transpilation: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

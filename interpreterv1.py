from bparser import BParser
from intbase import InterpreterBase, ErrorType
import copy
from typing import List

# debug helper functions

str_to_bool = {
    'true': True,
    'false': False
}


def is_str_format(s: str):
    if isinstance(s, str) and len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        return True
    else:
        return False


def concat_str(s: str):
    return s[1:-1]

def create_val(val):
    """
    Convert a Brewin value to a Value object with type and Python value
    """
    if val == InterpreterBase.TRUE_DEF:
        return Value("bool", True)
    if val == InterpreterBase.FALSE_DEF:
        return Value("bool", False)
    if val[0] == '"':
        return Value("string", val.strip('"'))
    if val.lstrip('-').isnumeric():
        return Value("int", int(val))
    if val == InterpreterBase.NULL_DEF:
        return Value("obj", None)
    if val == InterpreterBase.NOTHING_DEF:
        return Value("nothing", None)
    return None

def get_type(val):
    """
    Convert a Brewin value to a Value object with type and Python value
    """
    if val == InterpreterBase.TRUE_DEF:
        return "bool"
    if val == InterpreterBase.FALSE_DEF:
        return "bool"
    if val[0] == '"':
        return "string"
    if val.lstrip('-').isnumeric():
        return "int"
    if val == InterpreterBase.NULL_DEF:
        return "obj"
    if val == InterpreterBase.NOTHING_DEF:
        return "nothing"
    return None

class Value():
    def __init__(self, type: str, value: any):
        self.type = type
        self.value = value
        
class Variable():
    def __init__(self, name: str, type: str, val_obj: Value = None):
        self.name = name
        self.type = type
        self.val_obj = val_obj
        
    def set_val(self, val_obj: Value = None):
        self.val_obj = val_obj

class Method():
    def __init__(self, name: str, params_list: List[Variable], body: List[str]):
        self.name = name
        self.params_list = params_list
        self.params = {}
        self.body = body


class Class():
    def __init__(self, interpreter, name: str, fields: dict(), methods: dict()):
        self.interpreter = interpreter
        self.name = name
        self.fields = fields
        self.methods = methods

    def instantiate_object(self):
        obj = Object(self.interpreter, self.name, self.fields, self.methods)
        return obj


class Object():
    def __init__(self, interpreter, class_name: str, fields: dict(), methods: dict()):
        self.interpreter = interpreter
        self.class_name = class_name
        self.fields = copy.deepcopy(fields)
        self.methods = copy.deepcopy(methods)

    def __get_val(self, var: any, method: Method) -> Value:
        # print(var, type(var), var == "null")
        if isinstance(var, list):
            # print('list')
            return self.__evaluate(var, method)
        elif var in str_to_bool:
            # print('bool')
            return Value("bool", str_to_bool[var])
        elif var.lstrip('-').isnumeric():
            # print('int')
            return Value("int", int(var))
        elif is_str_format(var):
            # print('str')
            return Value("string", concat_str(var))
        elif var == "null":
            # print('null')
            return Value("obj", None)
        elif var in method.params:
            # print('method params')
            return method.params[var].val_obj
        elif var in self.fields:
            # print('fields')
            return self.fields[var].val_obj
        elif var == "me":
            # print('me')
            return Value("obj", self)
        else:
            self.interpreter.error(ErrorType.NAME_ERROR)

    def __is_operand_same_type(self, val1: Value, val2: Value):
        return (
            (val1.type == "bool" and val2.type == "bool")
            or ((val1.type == "int" or val1.type == "float") and (val2.type == "int" or val2.type == "float"))
            or (val1.type == "str" and val2.type == "str")
            or (val1 is None and val2 is None)
        )

    def __is_operand_compatible(self, operator: str, val: Value):
        return (
            (operator in ['&', '|', '==', '!='] and val.type == "bool")
            or (operator in ['-', '*', '/', '%'] and (val.type == "int" or val.type == "float"))
            or (operator in ['<', '>', '<=', '>=', '!=', '==', '+'] and not val.type == "bool")
        )

    def __evaluate(self, expression: list, method: Method):
        # print("expression: " + str(expression))
        operator = expression[0]

        if operator == self.interpreter.NEW_DEF:
            class_name = expression[1]

            if class_name not in self.interpreter.classes:
                self.interpreter.error(ErrorType.TYPE_ERROR)
                return None

            return self.interpreter.classes[class_name].instantiate_object()

        elif operator == self.interpreter.CALL_DEF:
            return self.__run_call_statement(expression[1:], method)

        elif operator == '!':
            val = self.__get_val(expression[1], method)
            if val.type != bool:
                self.interpreter.error(ErrorType.TYPE_ERROR)
                return None

            return not val.value
        else:
            val1, val2 = self.__get_val(
                expression[1], method), self.__get_val(expression[2], method)
            # print(operator, val1, val2)
            # print(type(val1), type(val2))
            # print(self.__is_operand_same_type(val1, val2),
            #       self.__is_operand_compatible(operator, val1))

            if ((val1.value is None and val2.type == "obj")
                    or (val2.value is None and val1.type == "obj")):
                if operator == '==':
                    return False
                elif operator == '!=':
                    return True
                else:
                    self.interpreter.error(ErrorType.TYPE_ERROR)
                    return None

            if not (self.__is_operand_same_type(val1, val2)
                    and self.__is_operand_compatible(operator, val1)):
                self.interpreter.error(ErrorType.TYPE_ERROR)
                return None

            python_operator = operator
            if operator == '&':
                python_operator = 'and'
            elif operator == '|':
                python_operator = 'or'

            python_expression = f"{repr(val1)} {python_operator} {repr(val2)}"
            eval_res = eval(python_expression)
            # print("eval result: " + str(eval_res))

            if type(eval_res) is float:
                return int(eval_res)

            return eval_res

    def call_method(self, method_name: str, args: List[Value]):
        if not method_name in self.methods:
            self.interpreter.error(ErrorType.NAME_ERROR)
            return None

        method = copy.deepcopy(self.methods[method_name])

        if len(args) != len(method.params_list):
            self.interpreter.error(ErrorType.TYPE_ERROR)
            return None

        # update local vars, update same-name fields with parameters
        # local_vars = copy.deepcopy(self.fields)
        # method.params = dict(zip(method.params, arguments))
        # for var, val in method.params.values():
        #     local_vars[var] = val

        # # print(method.body)

        for i, arg in enumerate(args):
            method.params[method.params_list[i]] = copy.deepcopy(arg)

        res = self.__run_statement(method.body, method)
        return res

    def __run_statement(self, statement: list, method: Method):
        statement_type, *args = statement
        res = None
        # # print(statement_type, args)
        if statement_type == self.interpreter.PRINT_DEF:
            self.__run_print_statement(args, method)
        elif statement_type == self.interpreter.INPUT_INT_DEF:
            self.__run_input_int_statement(args, method)
        elif statement_type == self.interpreter.INPUT_STRING_DEF:
            self.__run_input_str_statement(args, method)
        elif statement_type == self.interpreter.SET_DEF:
            self.__run_set_statement(args, method)
        elif statement_type == self.interpreter.BEGIN_DEF:
            for stmt in args:
                res = self.__run_statement(stmt, method)
                if res == "return heyhey" or res is not None:
                    return res
        elif statement_type == self.interpreter.IF_DEF:
            res = self.__run_if_statement(args, method)
        elif statement_type == self.interpreter.WHILE_DEF:
            res = self.__run_while_statement(args, method)
        elif statement_type == self.interpreter.CALL_DEF:
            res = self.__run_call_statement(args, method)
        elif statement_type == self.interpreter.RETURN_DEF:
            res = self.__run_return_statement(args, method)
        else:
            raise Exception('Invalid statement type')

        return res

    def __run_print_statement(self, args: List[str], method: Method) -> None:
        output = ""

        for arg in args:
            arg_val = self.__get_val(arg, method)
            # print(arg, arg_val, type(arg_val))

            # convert primitive types to string
            if arg_val.type == "bool":
                arg_val = 'true' if arg_val.value else 'false'
            elif arg_val.type == "int":
                arg_val = str(arg_val.value)

            output += arg_val.value

        self.interpreter.output(output)

    def __run_input_int_statement(self, args: list, method: Method):
        var = args[0]
        if var in method.params:
            method.params[var] = int(self.interpreter.get_input())
        elif var in self.fields:
            self.fields[var] = int(self.interpreter.get_input())
        else:
            self.interpreter.error(ErrorType.TYPE_ERROR)

    def __run_input_str_statement(self, args: list, method: Method):
        var = args[0]
        if var in method.params:
            method.params[var] = self.interpreter.get_input()
        elif var in self.fields:
            self.fields[var] = self.interpreter.get_input()
        else:
            self.interpreter.error(ErrorType.TYPE_ERROR)

    def __run_set_statement(self, args: list, method: Method) -> dict:
        var, val = args

        val_obj = self.__get_val(val, method)

        if var in method.params:
            var_obj = method.params[var]
        elif var in self.fields:
            var_obj = self.fields[var]
        else:
            self.interpreter.error(ErrorType.NAME_ERROR)
            
        if var_obj.type != val_obj.type:
            self.interpreter.error(ErrorType.TYPE_ERROR)
            return None
        
        var_obj.set_val(val_obj)

        return method.params

    def __run_if_statement(self, args: list, method: Method):
        condition, *statements = args

        condition_val = self.__get_val(condition, method)
        if condition_val.type != bool:
            self.interpreter.error(ErrorType.TYPE_ERROR)
            return None

        res = None
        if condition_val.value:
            res = self.__run_statement(statements[0], method)
        elif len(statements) > 1:
            res = self.__run_statement(statements[1], method)

        return res

    def __run_while_statement(self, args: list, method: Method):
        condition, statement = args
        # # print(condition)

        res = None
        while True:
            condition_val = self.__get_val(condition, method)
            # # print(self.fields)
            if condition_val.value != "bool":
                self.interpreter.error(ErrorType.TYPE_ERROR)
                return None

            if not condition_val.value:
                break

            res = self.__run_statement(statement, method)
            if res == "return heyhey" or res is not None:
                return res
            # # print(self.fields)
        return res

    def __run_call_statement(self, args: list, method: Method):
        obj_name, method_name, *method_arg_names = args

        if isinstance(obj_name, list):
            obj = self.__evaluate(obj_name, method)
        elif obj_name == "me":
            obj = self
        elif obj_name in method.params:
            obj = method.params[obj_name]
        elif obj_name in self.fields:
            obj = self.fields[obj_name]
        else:
            self.interpreter.error(ErrorType.NAME_ERROR)
            return None

        if not obj:
            self.interpreter.error(ErrorType.FAULT_ERROR)
            return None

        # print(obj_name, obj, type(obj))

        method_args = [self.__get_val(arg, method) for arg in method_arg_names]

        return obj.call_method(method_name, method_args)

    def __run_return_statement(self, args: list, method: Method):
        # print("return arg num: " + str(len(args)))
        if len(args) == 0:
            return "return heyhey"

        return_val = self.__get_val(args[0], method)
        if return_val.type != method.return_type:
            self.interpreter.error(ErrorType.TYPE_ERROR)
            return None
        
        return return_val.value


class Interpreter(InterpreterBase):
    def __init__(self, console_output=True, inp=None, trace_output=True):
        super().__init__(console_output, inp)
        self.trace_output = trace_output
        self.classes = {}

    def run(self, program):
        result, parsed_program = BParser.parse(program)
        # print(parsed_program)

        if result:
            self.__get_all_classes(parsed_program)

            if 'main' not in self.classes:
                self.error(ErrorType.TYPE_ERROR)

            main_class = self.classes['main']
            main_obj = main_class.instantiate_object()
            main_obj.call_method('main', [])

            # for name, cls in self.classes.items():
            #     # print(name, cls.fields, cls.methods)
        else:
            print('Parsing failed. There must have been a mismatched parenthesis.')

    def __get_all_classes(self, parsed_program):
        for class_def in parsed_program:
            # init params for class
            class_keyword, class_name, *lines = class_def
            fields = {}
            methods = {}

            # interpret each line in class as field or method
            for item in lines:
                if item[0] == self.FIELD_DEF:
                    field_def, name, val = item

                    if name in fields:
                        self.error(ErrorType.NAME_ERROR)
                        
                    if isinstance(val, list):
                        self.error(ErrorType.TYPE_ERROR)
                        
                    field_val = create_val(val)

                    fields[name] = Variable(name, get_type(val), field_val)
                elif item[0] == self.METHOD_DEF:
                    method_def, name, params, body = item

                    if name in methods:
                        self.error(ErrorType.NAME_ERROR)
                        
                    param_vars = []
                    for param in params:
                        param_vars.append(Variable(param))

                    methods[name] = Method(name, param_vars, body)

            # instantiate class and add to classes
            if class_name in self.classes:
                self.error(ErrorType.TYPE_ERROR)

            self.classes[class_name] = Class(self, class_name, fields, methods)


# def main():
#     while_case = ['(class main',
#                   '(field x 0)',
#                   '(method main ()',
#                   '(begin',
#                   '(set x 5)',
#                   '(while (> x -1)',
#                   '(begin',
#                   '(print "x is " x)',
#                   '(set x (- x 1))',
#                   ')',
#                   ')',
#                   ')',
#                   ')',
#                   ')'
#                   ]

#     interpreter = Interpreter()
#     interpreter.run(while_case)


# if __name__ == '__main__':
#     main()

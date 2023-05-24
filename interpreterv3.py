from bparser import BParser
from intbase import InterpreterBase, ErrorType
import copy
from typing import List

# helper functions
str_to_bool = {
    'true': True,
    'false': False
}
primitives = ["bool", "int", "string"]


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


def is_valid_assign(var_type: str, val_type: str, val_value: any):
    primitives = ["bool", "string", "int"]
    return var_type == val_type or (var_type not in primitives and val_value is None) or (val_type == "obj" and val_value is not None and var_type in val_value.family)


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
    def __init__(self, name: str, return_type: str, params_list: List[Variable], body: List[str]):
        self.name = name
        self.return_type = return_type
        self.params_list = params_list
        self.body = body
        self.params = {}
        self.local_vars = []


class Class():
    def __init__(self, interpreter, name: str, parent, family: List[str], fields: dict(), methods: dict()):
        self.interpreter = interpreter
        self.name = name
        self.parent = parent
        self.family = family
        self.fields = fields
        self.methods = methods

    def instantiate_object(self):
        obj = Object(self)
        return obj


class Object():
    def __init__(self, class_ref: Class):
        self.interpreter = class_ref.interpreter
        self.name = copy.deepcopy(class_ref.name)
        self.parent = class_ref.parent.instantiate_object() if class_ref.parent else None
        self.family = copy.deepcopy(class_ref.family)
        self.fields = copy.deepcopy(class_ref.fields)
        self.methods = copy.deepcopy(class_ref.methods)

    def __get_var(self, var_name: str, method: Method) -> Variable:
        print(var_name, method.params)
        if var_name == "me":
            return Value("obj", self)
        if var_name in method.params:
            return method.params[var_name]
        if var_name in self.fields:
            return self.fields[var_name]
        return None

    def __get_val(self, var: any, method: Method) -> Value:
        if isinstance(var, list):
            # print('expression')
            return self.__evaluate(var, method)
        if var in str_to_bool:
            # print("bool")
            return Value("bool", str_to_bool[var])
        if var.lstrip('-').isnumeric():
            # print("int")
            return Value("int", int(var))
        if is_str_format(var):
            # print("string")
            return Value("string", concat_str(var))
        if var == "null":
            # print("null")
            return Value("obj", None)

        for local_var in method.local_vars:
            if var == local_var.name:
                # print("local var")
                return local_var.val_obj

        if var in method.params:
            # print("param")
            return method.params[var].val_obj
        if var in self.fields:
            # print("field")
            # print(self.fields[var].val_obj.type, self.fields[var].val_obj.value, type(self.fields[var].val_obj.value))
            return self.fields[var].val_obj
        if var == "me":
            # print("me")
            return Value("obj", self)

        self.interpreter.error(ErrorType.NAME_ERROR)

    def __is_operand_same_type(self, val1: Value, val2: Value):
        return (
            (val1.type == "bool" and val2.type == "bool")
            or (val1.type == "int" and val2.type == "int")
            or (val1.type == "string" and val2.type == "string")
            or (val1 is None and val2 is None)
        )

    def __is_operand_compatible(self, operator: str, val: Value):
        return (
            (operator in ['&', '|', '==', '!='] and val.type == "bool")
            or (operator in ['-', '*', '/', '%'] and (val.type == "int"))
            or (operator in ['<', '>', '<=', '>=', '!=', '==', '+'] and not val.type == "bool")
        )

    def __evaluate(self, expression: list, method: Method, actual_me=None):
        operator = expression[0]
        # print(expression)

        if operator == self.interpreter.NEW_DEF:
            class_name = expression[1]

            if class_name not in self.interpreter.classes:
                self.interpreter.error(ErrorType.TYPE_ERROR)

            return Value("obj", self.interpreter.classes[class_name].instantiate_object())

        elif operator == self.interpreter.CALL_DEF:
            res = self.__run_call_statement(expression[1:], method, actual_me)
            # print(res)
            return res

        elif operator == '!':
            val = self.__get_val(expression[1], method)
            if val.type != "bool":
                self.interpreter.error(ErrorType.TYPE_ERROR)

            return Value("bool", not val.value)
        else:
            # print(expression[1])
            val1, val2 = self.__get_val(
                expression[1], method), self.__get_val(expression[2], method)
            # print(val1.type, val1.value, val2.type, val2.value)
            # print(val1.type, val1.value, type(val1.value), val2.type, val2.value, type(val2.value))

            # compare none
            if (val1.type == "obj" and val2.type == "obj"):
                # check if two objects are in same family
                var1, var2 = self.__get_var(
                    expression[1], method), self.__get_var(expression[2], method)
                if not var1 or not var2:
                    is_same_family = True
                else:
                    family1 = self.interpreter.classes[var1.type].family
                    family2 = self.interpreter.classes[var2.type].family
                    is_same_family = var1.type in family2 or var2.type in family1

                if not is_same_family:
                    self.interpreter.error(ErrorType.TYPE_ERROR)

                if (val1.value is None and val2.value is None):
                    if operator == '==':
                        return Value("bool", True)
                    elif operator == '!=':
                        return Value("bool", False)
                    else:
                        self.interpreter.error(ErrorType.TYPE_ERROR)
                elif ((val1.value is None and val2.type == "obj")
                        or (val2.value is None and val1.type == "obj")):
                    if operator == '==':
                        return Value("bool", False)
                    elif operator == '!=':
                        return Value("bool", True)
                    else:
                        self.interpreter.error(ErrorType.TYPE_ERROR)
                else:
                    if operator == '==':
                        return Value("bool", val1.value is val2.value)
                    elif operator == '!=':
                        return Value("bool", val1.value is not val2.value)
                    else:
                        self.interpreter.error(ErrorType.TYPE_ERROR)

            # print(self.__is_operand_same_type(val1, val2), self.__is_operand_compatible(operator, val1))
            if not (self.__is_operand_same_type(val1, val2)
                    and self.__is_operand_compatible(operator, val1)):
                self.interpreter.error(ErrorType.TYPE_ERROR)

            # determine return type
            if operator in ['&', '|', '==', '!='] and val1.type == "bool":
                return_type = "bool"
            elif operator in ['-', '*', '/', '%'] and (val1.type == "int"):
                return_type = "int"
            elif operator in ['<', '>', '<=', '>=', '!=', '=='] and not val1.type == "bool":
                return_type = "bool"
            elif operator == '+':
                return_type = val1.type
            else:
                self.interpreter.error(ErrorType.TYPE_ERROR)
                return None

            # evaluate return value
            python_operator = operator
            if operator == '&':
                python_operator = 'and'
            elif operator == '|':
                python_operator = 'or'

            print(val1.type, val1.value, type(val1.value),
                  val2.type, val2.value, type(val2.value))
            python_expression = f"{repr(val1.value)} {python_operator} {repr(val2.value)}"
            eval_res = eval(python_expression)

            if type(eval_res) is float:
                eval_res = int(eval_res)

            return Value(return_type, eval_res)

    def call_method(self, method_name: str, args: List[Value], actual_me=None):
        # find the most derived method in current class or super class
        # check same arg number for overloading
        method = None
        if method_name in self.methods and len(args) == len(self.methods[method_name].params_list):
            print(method_name, "have method!")
            method = copy.deepcopy(self.methods[method_name])

            for i, arg in enumerate(args):
                param_var = method.params_list[i]

                if not is_valid_assign(param_var.type, arg.type, arg.value):
                    method = None
                    break

        if not method:
            if self.parent:
                print(method_name, "check parent!")
                res = self.parent.call_method(method_name, args, actual_me)
                return res
            else:
                self.interpreter.error(ErrorType.NAME_ERROR)

        if len(args) != len(method.params_list):
            self.interpreter.error(ErrorType.TYPE_ERROR)
            return None

        for i, arg in enumerate(args):
            param_var = method.params_list[i]

            if param_var.name in method.params:
                self.interpreter.error(ErrorType.NAME_ERROR)

            if param_var.type in primitives:
                param_var = copy.deepcopy(param_var)

            # print(param_var.type, arg.type, arg.value)
            if not is_valid_assign(param_var.type, arg.type, arg.value):
                self.interpreter.error(ErrorType.TYPE_ERROR)
                return None

            if param_var.type in primitives:
                param_var.set_val(copy.deepcopy(arg))
            else:
                param_var.set_val(arg)

            method.params[param_var.name] = param_var

        res = self.__run_statement(method.body, method, actual_me)
        if res is None:
            if method.return_type == "bool":
                return Value("bool", False)
            if method.return_type == "int":
                return Value("int", 0)
            if method.return_type == "string":
                return Value("string", "")
            if method.return_type == "obj":
                return Value("obj", None)
        return res

    def __run_statement(self, statement: list, method: Method, actual_me=None):
        statement_type, *args = statement
        res = None
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
                res = self.__run_statement(stmt, method, actual_me)
                if res == "return now!" or res is not None:
                    return res
        elif statement_type == self.interpreter.IF_DEF:
            res = self.__run_if_statement(args, method, actual_me)
        elif statement_type == self.interpreter.WHILE_DEF:
            res = self.__run_while_statement(args, method, actual_me)
        elif statement_type == self.interpreter.CALL_DEF:
            res = self.__run_call_statement(args, method, actual_me)
        elif statement_type == self.interpreter.RETURN_DEF:
            res = self.__run_return_statement(args, method, actual_me)
        elif statement_type == self.interpreter.LET_DEF:
            res = self.__run_let_statement(args, method, actual_me)
        else:
            raise Exception('Invalid statement type')

        return res

    def __run_print_statement(self, args: List[str], method: Method) -> None:
        output = ""
        # print(args)

        for arg in args:
            arg_val = self.__get_val(arg, method)
            # print(arg, arg_val.value if arg_val else None, arg_val.type if arg_val else None, type(arg_val.value))

            # convert primitive types to string

            if arg_val is None:
                output += "None"
            else:
                if arg_val.value is None:
                    output += "None"
                elif arg_val.type == "bool":
                    output += 'true' if arg_val.value else 'false'
                elif arg_val.type == "int":
                    output += str(arg_val.value)
                elif arg_val.type == "string":
                    output += arg_val.value

            # print(arg, arg_val.value if arg_val else None, arg_val.type if arg_val else None, type(arg_val.value))

        print("print: " + output)

        self.interpreter.output(output)

    def __run_input_int_statement(self, args: list, method: Method):
        var = args[0]

        if var in method.params:
            if method.params[var].type != "int":
                self.interpreter.error(ErrorType.TYPE_ERROR)

            method.params[var] = Variable(var, "int", Value(
                "int", int(self.interpreter.get_input())))
        elif var in self.fields:
            if self.fields[var].type != "int":
                self.interpreter.error(ErrorType.TYPE_ERROR)

            self.fields[var] = Variable(var, "int", Value(
                "int", int(self.interpreter.get_input())))
        else:
            self.interpreter.error(ErrorType.TYPE_ERROR)

    def __run_input_str_statement(self, args: list, method: Method):
        var = args[0]

        if var in method.params:
            if method.params[var].type != "string":
                self.interpreter.error(ErrorType.TYPE_ERROR)

            method.params[var] = Variable(var, "string", Value(
                "string", self.interpreter.get_input()))
        elif var in self.fields:
            if self.fields.type != "string":
                self.interpreter.error(ErrorType.TYPE_ERROR)

            self.fields[var] = Variable(var, "int", Value(
                "int", self.interpreter.get_input()))
        else:
            self.interpreter.error(ErrorType.TYPE_ERROR)

    def __run_set_statement(self, args: list, method: Method) -> dict:
        var, val = args

        val_obj = self.__get_val(val, method)

        var_obj = None
        for local_var in method.local_vars:
            if var == local_var.name:
                var_obj = local_var
                break

        if not var_obj:
            if var in method.params:
                var_obj = method.params[var]
            elif var in self.fields:
                var_obj = self.fields[var]
            else:
                self.interpreter.error(ErrorType.NAME_ERROR)

        # print(val_obj.type, val_obj.value)
        # if val_obj.type != "obj":
        #     is_same_family = True
        # else:
        #     family = self.interpreter.classes[var_obj.type].family
        #     print(val_obj.value.family[0], family)
        #     is_same_family = val_obj.type in self.interpreter.classes[var_obj.type].family

        # if not is_same_family:
        #     self.interpreter.error(ErrorType.TYPE_ERROR)

        if not is_valid_assign(var_obj.type, val_obj.type, val_obj.value):
            self.interpreter.error(ErrorType.TYPE_ERROR)
            return None

        # print(var_obj.type, val_obj.type, val_obj.value)
        var_obj.set_val(val_obj)

        return method.params

    def __run_if_statement(self, args: list, method: Method, actual_me=None):
        condition, *statements = args

        condition_val = self.__get_val(condition, method)
        if condition_val.type != "bool":
            self.interpreter.error(ErrorType.TYPE_ERROR)
            return None

        res = None
        if condition_val.value:
            res = self.__run_statement(statements[0], method, actual_me)
        elif len(statements) > 1:
            res = self.__run_statement(statements[1], method, actual_me)

        return res

    def __run_while_statement(self, args: list, method: Method, actual_me=None):
        condition, statement = args

        res = None
        while True:
            condition_val = self.__get_val(condition, method)

            if condition_val.type != "bool":
                self.interpreter.error(ErrorType.TYPE_ERROR)
                return None

            if not condition_val.value:
                break

            res = self.__run_statement(statement, method, actual_me)
            if res == "return now!" or res is not None:
                return res

        return res

    def __run_call_statement(self, args: list, method: Method, actual_me=None):
        obj_name, method_name, *method_arg_names = args
        # print("actual_me: " + actual_me.name if actual_me else "")

        obj = None
        if isinstance(obj_name, list):
            obj = self.__evaluate(obj_name, method, actual_me).value
        elif obj_name == "me":
            obj = actual_me if actual_me else self
        elif obj_name == "super":
            obj = self.parent
        else:
            for local_var in method.local_vars:
                if obj_name == local_var.name:
                    obj = local_var.val_obj.value

        if not obj:
            if obj_name in method.params:
                obj = method.params[obj_name].val_obj.value
            elif obj_name in self.fields:
                obj = self.fields[obj_name].val_obj.value
            # else:
            #     self.interpreter.error(ErrorType.NAME_ERROR)
            #     return None

        if not obj:
            self.interpreter.error(ErrorType.FAULT_ERROR)
            return None

        method_args = [self.__get_val(arg, method) for arg in method_arg_names]

        return obj.call_method(method_name, method_args, actual_me if actual_me else obj)

    def __run_return_statement(self, args: list, method: Method, actual_me=None):
        # default return
        if len(args) == 0:
            if method.return_type == "bool":
                return Value("bool", False)
            if method.return_type == "int":
                return Value("int", 0)
            if method.return_type == "string":
                return Value("string", "")
            if method.return_type == "obj":
                return Value("obj", None)
            return "return now!"
        elif method.return_type == "void":  # void must not return any value
            self.interpreter.error(ErrorType.TYPE_ERROR)
            return None

        return_val = self.__get_val(args[0], method)

        if not is_valid_assign(method.return_type, return_val.type, return_val.value):
            self.interpreter.error(ErrorType.TYPE_ERROR)
            return None

        # print("returning" + str(return_val.value))
        return return_val

    def __run_let_statement(self, args: list, method: Method, actual_me=None):
        vars, *statements = args
        insert_len = 0
        added_var_names = set()

        for var in vars:
            var_type, var_name, var_val = var
            if var_name in added_var_names:
                self.interpreter.error(ErrorType.NAME_ERROR)
                return None

            val_obj = self.__get_val(var_val, method)
            if not is_valid_assign(var_type, val_obj.type, val_obj.value):
                self.interpreter.error(ErrorType.TYPE_ERROR)
                return None

            method.local_vars.insert(0, Variable(var_name, var_type, val_obj))
            insert_len += 1
            added_var_names.add(var_name)

        for stmt in statements:
            res = self.__run_statement(stmt, method, actual_me)
            if res == "return now!" or res is not None:
                return res

        for i in range(insert_len):
            method.local_vars.pop(0)

        return None


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
        else:
            print('Parsing failed. There must have been a mismatched parenthesis.')

    def __get_all_classes(self, parsed_program):
        for class_def in parsed_program:
            # init params for class
            class_keyword, class_name, *lines = class_def
            fields = {}
            methods = {}
            family = [class_name]
            parent = None

            # check if inherits, add super class
            if len(lines) >= 2 and lines[0] == "inherits":
                parent_name = lines[1]
                parent = self.classes[parent_name]

                # add super class and its super classes to family of this class
                family += parent.family

                lines = lines[2:]

            # interpret each line in class as inherits, field or method
            for item in lines:
                if item[0] == self.FIELD_DEF:
                    field_def, field_type, name, val = item

                    if name in fields:
                        self.error(ErrorType.NAME_ERROR)

                    if isinstance(val, list):
                        self.error(ErrorType.TYPE_ERROR)

                    field_val = create_val(val)
                    valid_assign = field_type == field_val.type or (
                        field_type not in primitives and field_val.value == None)
                    if not valid_assign:
                        self.error(ErrorType.TYPE_ERROR)

                    fields[name] = Variable(name, field_type, field_val)
                elif item[0] == self.METHOD_DEF:
                    method_def, return_type, name, params, body = item

                    if name in methods:
                        self.error(ErrorType.NAME_ERROR)

                    param_vars = []
                    for param in params:
                        param_type, param_name = param
                        param_vars.append(Variable(param_name, param_type))

                    methods[name] = Method(name, return_type, param_vars, body)

            # instantiate class and add to classes
            if class_name in self.classes:
                self.error(ErrorType.TYPE_ERROR)

            self.classes[class_name] = Class(
                self, class_name, parent, family, fields, methods)

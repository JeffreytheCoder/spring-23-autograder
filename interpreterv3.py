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
default_vals = {
    "bool": False,
    "int": 0,
    "string": "",
}


def get_default_val(val_type):
    if val_type in primitives:
        return Value(val_type, default_vals[val_type])
    else:
        return Value(val_type, None)


def is_str_format(s: str):
    if isinstance(s, str) and len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        return True
    else:
        return False


def concat_str(s: str):
    return s[1:-1]


def create_val(type, val):
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
        return Value(type, None)
    if val == InterpreterBase.NOTHING_DEF:
        return Value("nothing", None)
    return None


def is_valid_assign(var_type: str, val_type: str, val_value: any):
    # print(var_type, val_type, val_value)

    same_declared_types = var_type == val_type
    obj_assign_null = var_type not in primitives and val_type not in primitives and val_type == "null"
    obj_assign_family = var_type not in primitives and val_type not in primitives and val_value is not None and var_type in val_value.family
    # print(same_declared_types, obj_assign_null, obj_assign_family)

    return same_declared_types or obj_assign_null or obj_assign_family


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

    def call_method(self, method_name: str, args: List[Value], actual_me=None):
        print("calling", self.name,  method_name)
        # find the most derived method in current class or super class
        # check same arg number for overloading
        method = None
        if method_name in self.methods and len(args) == len(self.methods[method_name].params_list):
            # print(method_name, "have method!")
            method = copy.deepcopy(self.methods[method_name])

            # check if all same param, arg types
            for i, arg in enumerate(args):
                param_var = method.params_list[i]

                # check if arg is error from another call, populate this error
                if arg.type == "error":
                    return arg

                if not is_valid_assign(param_var.type, arg.type, arg.value):
                    method = None
                    break

        # check parent class for inherited methods
        if not method:
            if self.parent:
                # print(method_name, "check parent!")
                res = self.parent.call_method(method_name, args, actual_me)
                return res
            else:
                self.interpreter.error(ErrorType.NAME_ERROR)

        # check if same number of params
        if len(args) != len(method.params_list):
            self.interpreter.error(ErrorType.TYPE_ERROR)
            return None

        # create (param_name, arg_var) pair in method.params dictionary
        for i, arg in enumerate(args):
            param_var = method.params_list[i]

            # check duplicate param name
            if param_var.name in method.params:
                self.interpreter.error(ErrorType.NAME_ERROR)

            # deep copy primitive param
            if param_var.type in primitives:
                param_var = copy.deepcopy(param_var)

            # check if arg type match param type
            if not is_valid_assign(param_var.type, arg.type, arg.value):
                self.interpreter.error(ErrorType.TYPE_ERROR)
                return None

            # assign value object to param var
            if param_var.type in primitives:
                param_var.set_val(copy.deepcopy(arg))
            else:
                param_var.set_val(arg)

            method.params[param_var.name] = param_var

        res = self.__run_statement(method.body, method, actual_me)
        return res

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
            return Value("null", None)

        for local_var in method.local_vars:
            if var == local_var.name:
                # print("local var")
                return local_var.val_obj

        if var in method.params:
            # print("param")
            return method.params[var].val_obj
        if var in self.fields:
            # print("field")
            return self.fields[var].val_obj
        if var == "me":
            # print("me")
            return Value(self.name, self)

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

            if '@' in class_name:
                t_class_name = class_name.split('@')[0]
                if t_class_name in self.interpreter.t_classes:
                    print("NEW adding t_class!")
                    self.interpreter.add_t_class(class_name)

            if class_name not in self.interpreter.classes:
                self.interpreter.error(ErrorType.TYPE_ERROR)

            return Value(class_name, self.interpreter.classes[class_name].instantiate_object())

        elif operator == self.interpreter.CALL_DEF:
            res = self.__run_call_statement(expression[1:], method, actual_me)
            return res

        elif operator == '!':
            val = self.__get_val(expression[1], method)

            if val.type == "error":
                return val

            if val.type != "bool":
                self.interpreter.error(ErrorType.TYPE_ERROR)

            return Value("bool", not val.value)
        else:
            val1, val2 = self.__get_val(
                expression[1], method), self.__get_val(expression[2], method)

            if val1.type == "error":
                return val1
            if val2.type == "error":
                return val2
            # print(val1.type, val1.value, val2.type, val2.value)

            # compare objects & nulls
            if (val1.type not in primitives and val2.type not in primitives):
                if val1.type == "null":
                    if operator == '==':
                        return Value("bool", val2.value is None)
                    elif operator == '!=':
                        return Value("bool", val2.value is not None)
                    else:
                        self.interpreter.error(ErrorType.TYPE_ERROR)
                elif val2.type == "null":
                    if operator == '==':
                        return Value("bool", val1.value is None)
                    elif operator == '!=':
                        return Value("bool", val1.value is not None)
                    else:
                        self.interpreter.error(ErrorType.TYPE_ERROR)
                else:  # both class
                    # check if two classes are comparable
                    val1_family = self.interpreter.classes[val1.type].family
                    val2_family = self.interpreter.classes[val2.type].family
                    if val1.type != val2.type and val1.type not in val2_family and val2.type not in val1_family:
                        self.interpreter.error(ErrorType.TYPE_ERROR)

                    if operator == '==':
                        return Value("bool", val1.value is val2.value)
                    elif operator == '!=':
                        return Value("bool", val1.value is not val2.value)
                    else:
                        self.interpreter.error(ErrorType.TYPE_ERROR)

            # compare primitives
            # check primitives are comparable
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

            python_expression = f"{repr(val1.value)} {python_operator} {repr(val2.value)}"
            eval_res = eval(python_expression)

            if type(eval_res) is float:
                eval_res = int(eval_res)

            return Value(return_type, eval_res)

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
                if res is not None:
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
        elif statement_type == self.interpreter.TRY_DEF:
            res = self.__run_try_statement(args, method, actual_me)
        elif statement_type == self.interpreter.THROW_DEF:
            res = self.__run_throw_statement(args, method, actual_me)
        else:
            raise Exception('Invalid statement type')

        return res

    def __run_print_statement(self, args: List[str], method: Method) -> None:
        output = ""
        # print(args)

        for arg in args:
            arg_val = self.__get_val(arg, method)

            if arg_val.type == "error":
                return arg_val
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
            if self.fields[var].type != "string":
                self.interpreter.error(ErrorType.TYPE_ERROR)

            self.fields[var] = Variable(var, "int", Value(
                "int", self.interpreter.get_input()))
        else:
            self.interpreter.error(ErrorType.TYPE_ERROR)

    def __run_set_statement(self, args: list, method: Method) -> dict:
        var, val = args

        val_obj = self.__get_val(val, method)
        if val_obj.type == "error":
            return val_obj

        # get variable object to be set to the value object
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

        # print(var_obj.name, var_obj.type, val_obj.type, val_obj.value)
        # check if valid set
        if not is_valid_assign(var_obj.type, val_obj.type, val_obj.value):
            self.interpreter.error(ErrorType.TYPE_ERROR)
            return None

        var_obj.set_val(val_obj)

    def __run_if_statement(self, args: list, method: Method, actual_me=None):
        condition, *statements = args

        # evaluate condition and check if bool
        condition_val = self.__get_val(condition, method)
        if condition_val.type == "error":
            return condition_val

        if condition_val.type != "bool":
            self.interpreter.error(ErrorType.TYPE_ERROR)

        res = None
        if condition_val.value:  # run if statement
            res = self.__run_statement(statements[0], method, actual_me)
        elif len(statements) > 1:  # run else statement
            res = self.__run_statement(statements[1], method, actual_me)

        return res

    def __run_while_statement(self, args: list, method: Method, actual_me=None):
        condition, statement = args

        res = None
        while True:
            # evaluate condition and check if bool
            condition_val = self.__get_val(condition, method)

            if condition_val.type == "error":
                return condition_val

            if condition_val.type != "bool":
                self.interpreter.error(ErrorType.TYPE_ERROR)

            # break loop if condition unsatisfied
            if not condition_val.value:
                break

            res = self.__run_statement(statement, method, actual_me)
            # print('while return:')
            # print(res)
            # break loop from inside return statement
            if res is not None:
                # print(res.type, res.value)
                return res

        return res

    def __run_call_statement(self, args: list, method: Method, actual_me=None):
        obj_name, method_name, *method_arg_names = args
        # print("actual_me: " + actual_me.name if actual_me else "")

        # evaluate which object to call
        obj = None
        if isinstance(obj_name, list):
            obj_val = self.__evaluate(obj_name, method, actual_me)

            if obj_val.type == "error":
                return obj_val

            obj = obj_val.value
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

        method_args = []
        for arg in method_arg_names:
            arg_val = self.__get_val(arg, method)

            if arg_val.type == "error":
                return arg_val

            method_args.append(arg_val)

        return obj.call_method(method_name, method_args, actual_me if actual_me else obj)

    def __run_return_statement(self, args: list, method: Method, actual_me=None):
        if len(args) == 0:  # default return
            if method.return_type == "void":
                # TODO: might need to change back to return nothing
                return Value("null", None)
            if method.return_type == "bool":
                return Value("bool", False)
            if method.return_type == "int":
                return Value("int", 0)
            if method.return_type == "string":
                return Value("string", "")
            return Value(method.return_type, None)
        elif method.return_type == "void":  # void must not return any value
            self.interpreter.error(ErrorType.TYPE_ERROR)

        return_val = self.__get_val(args[0], method)

        if return_val.type == "error":
            return return_val

        # assign value of null to return_type class
        if return_val.value is None:
            return_val.type = method.return_type

        # print("returning:")
        # print(return_val.type, return_val.value)

        # check if return type fits method return type
        if not is_valid_assign(method.return_type, return_val.type, return_val.value):
            self.interpreter.error(ErrorType.TYPE_ERROR)

        return return_val

    def __run_let_statement(self, args: list, method: Method, actual_me=None):
        vars, *statements = args
        insert_len = 0
        added_var_names = set()

        for var in vars:
            var_type, *var_name_and_val = var
            var_name = var_name_and_val[0]

            if '@' in var_type and var_type.split('@')[0] in self.interpreter.t_classes:
                print("let adding t_class!")
                self.interpreter.add_t_class(var_type)

            # check if duplicate let vars
            if var_name in added_var_names:
                self.interpreter.error(ErrorType.NAME_ERROR)

            if len(var_name_and_val) == 1:
                val_obj = get_default_val(var_type)
            else:
                var_val = var_name_and_val[1]
                val_obj = self.__get_val(var_val, method)

                if val_obj.type == "error":
                    return val_obj

                # assign null value with declared class type
                if val_obj.type == "null":
                    val_obj.type = var_type

            # check if value fits declared type
            if not is_valid_assign(var_type, val_obj.type, val_obj.value):
                self.interpreter.error(ErrorType.TYPE_ERROR)

            method.local_vars.insert(0, Variable(var_name, var_type, val_obj))
            insert_len += 1
            added_var_names.add(var_name)

        for stmt in statements:
            res = self.__run_statement(stmt, method, actual_me)
            if res is not None:
                for i in range(insert_len):
                    method.local_vars.pop(0)
                return res

        for i in range(insert_len):
            method.local_vars.pop(0)

        return None

    def __run_throw_statement(self, args: list, method: Method, actual_me=None):
        error_msg = self.__get_val(args[0], method)

        if error_msg.type == "error":
            return error_msg

        if error_msg.type != "string":
            self.interpreter.error(ErrorType.TYPE_ERROR)

        return Value("error", error_msg)

    def __run_try_statement(self, args: list, method: Method, actual_me=None):
        try_stmt, catch_stmt = args

        try_res = self.__run_statement(try_stmt, method, actual_me)
        if try_res is not None and try_res.type == "error":
            method.local_vars.insert(0, Variable(
                "exception", try_res.type, try_res.value))
            catch_res = self.__run_statement(catch_stmt, method, actual_me)
            method.local_vars.pop(0)
            return catch_res
        else:
            return try_res


class Interpreter(InterpreterBase):
    def __init__(self, console_output=True, inp=None, trace_output=True):
        super().__init__(console_output, inp)
        self.trace_output = trace_output
        self.classes = {}
        self.t_classes = {}

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

            # deal with template class
            if class_keyword == "tclass":
                self.t_classes[class_name] = lines
                continue

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
                    field_def, field_type, *field_name_and_val = item
                    field_name = field_name_and_val[0]

                    # check if field type is valid
                    # if field_type not in primitives and field_type not in self.classes:
                    #     self.error(ErrorType.TYPE_ERROR)

                    # check duplicate field
                    if field_name in fields:
                        self.error(ErrorType.NAME_ERROR)

                    if len(field_name_and_val) == 1:
                        field_val = get_default_val(field_type)
                    else:
                        val = field_name_and_val[1]

                        # check if init value is valid
                        if isinstance(val, list):
                            self.error(ErrorType.TYPE_ERROR)

                        field_val = create_val(field_type, val)

                    # check if value fits declared type
                    valid_assign = field_type == field_val.type
                    if not valid_assign:
                        self.error(ErrorType.TYPE_ERROR)

                    fields[field_name] = Variable(
                        field_name, field_type, field_val)

                elif item[0] == self.METHOD_DEF:
                    method_def, return_type, name, params, body = item

                    # check duplicate method
                    if name in methods:
                        self.error(ErrorType.NAME_ERROR)

                    # check if return type is valid
                    if '@' in return_type and return_type.split('@')[0] in self.t_classes:
                        print("class def adding t_class!")
                        self.add_t_class(return_type)
                    else:
                        valid_return_types = primitives + ['void']
                        if return_type not in valid_return_types and return_type not in self.classes and return_type != class_name:
                            self.error(ErrorType.TYPE_ERROR)

                    param_vars = []
                    for param in params:
                        param_type, param_name = param

                        # check if param type is valid
                        if param_type not in primitives and param_type not in self.classes and param_type != class_name:
                            self.error(ErrorType.TYPE_ERROR)

                        param_vars.append(Variable(param_name, param_type))

                    methods[name] = Method(name, return_type, param_vars, body)

            # instantiate class and add to classes
            if class_name in self.classes:
                self.error(ErrorType.TYPE_ERROR)

            self.classes[class_name] = Class(
                self, class_name, parent, family, fields, methods)

    def add_t_class(self, t_class_name_and_params: str):
        # check if t_class with these types already added
        if t_class_name_and_params in self.classes:
            return

        t_class_def = t_class_name_and_params.split('@')
        t_class_name, *real_params = t_class_def

        # check if template class definition exists
        if t_class_name not in self.t_classes:
            self.error(ErrorType.TYPE_ERROR)

        # init fields for class
        lines = self.t_classes[t_class_name]
        t_params = lines[0]
        fields = {}
        methods = {}
        family = [t_class_name_and_params]
        parent = None

        # check if params fit
        if len(real_params) != len(t_params):
            self.error(ErrorType.TYPE_ERROR)

        # create t to real params map
        param_map = {t_class_name + '@' +
                     '@'.join(t_params): t_class_name_and_params}
        for i in range(len(real_params)):
            # check if real param type is valid
            if real_params[i] not in primitives and real_params[i] not in self.classes:
                self.error(ErrorType.TYPE_ERROR)

            param_map[t_params[i]] = real_params[i]

        # interpret each line in class as inherits, field or method
        for item in lines[1:]:
            if item[0] == self.FIELD_DEF:
                field_def, t_field_type, *field_name_and_val = item
                field_name = field_name_and_val[0]

                # get real field type
                field_type = param_map[t_field_type]

                # check duplicate field
                if field_name in fields:
                    self.error(ErrorType.NAME_ERROR)

                # check if has init value, otherwise default
                if len(field_name_and_val) == 1:
                    if field_type in primitives:
                        field_val = Value(
                            field_type, default_vals[field_type])
                    else:
                        field_val = Value(field_type, None)
                else:
                    val = field_name_and_val[1]

                    # check if init value is valid
                    if isinstance(val, list):
                        self.error(ErrorType.TYPE_ERROR)

                    field_val = create_val(field_type, val)

                # check if value fits declared type
                valid_assign = field_type == field_val.type
                if not valid_assign:
                    self.error(ErrorType.TYPE_ERROR)

                fields[field_name] = Variable(
                    field_name, field_type, field_val)

            elif item[0] == self.METHOD_DEF:
                method_def, t_return_type, name, params, body = item

                # get real return type
                if t_return_type == 'void':
                    return_type = 'void'
                else:
                    return_type = param_map[t_return_type]

                # check duplicate method
                if name in methods:
                    self.error(ErrorType.NAME_ERROR)

                # check if return type is valid
                valid_return_types = primitives + ['void']
                if return_type not in valid_return_types and return_type not in self.classes and return_type != t_class_name_and_params:
                    self.error(ErrorType.TYPE_ERROR)

                param_vars = []
                for param in params:
                    t_param_type, param_name = param

                    # get real param type
                    param_type = param_map[t_param_type]

                    # check if param type is valid
                    if param_type not in primitives and param_type not in self.classes and param_type != t_class_name_and_params:
                        self.error(ErrorType.TYPE_ERROR)

                    param_vars.append(Variable(param_name, param_type))

                methods[name] = Method(name, return_type, param_vars, body)

        self.classes[t_class_name_and_params] = Class(
            self, t_class_name_and_params, parent, family, fields, methods)

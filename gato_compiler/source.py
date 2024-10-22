"Classes used to represent source code during compilation."

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import os.path
from pathlib import Path
import re
import tokenize
from typing import Literal, NoReturn, Self

from .parser import (
    Location, Parser,
    any_order, attribute, indented, key_of_dict, one_of, optional, paragraph,
    paragraphs, sequence_of, value_of_dict,
)


class Name(str):
    __slots__ = ()

    _syntax_ = re.compile(r'(?!\d)\w+')


class QualifiedName(str):
    __slots__ = ()

    _syntax_ = re.compile(r'(?!\d)\w+(?:\.(?!\d)\w+)*')


class InfixOperator(str):
    __slots__ = ()

    _syntax_ = re.compile(
        r'[,:+\-−*×/÷^%><=!?¿⸮~|&@]+|'
        r'[\u2200-\u22ff\u2a00-\u2aff]|'
        r'in|not in|is|is not|not|and|or'
    )
    _precedence_ = {
        # https://en.wikipedia.org/wiki/Order_of_operations
        # https://docs.python.org/3/reference/expressions.html#operator-precedence
        ',': 0,
        ':': 2,
        ':=': 4,
        'or': 6,
        'and': 8,
        'not': 10,
        'in': 12,
        'not in': 12,
        'is': 12,
        'is not': 12,
        '<': 12,
        '>': 12,
        '<=': 12,
        '≤': 12,
        '>=': 12,
        '≥': 12,
        '!=': 12,
        '≠': 12,
        '<>': 12,
        '==': 12,
        '=': 12,
        '~': 12,
        '∼': 12,  # U+223C TILDE OPERATOR
        '|': 14,
        '^': 16,
        '&': 18,
        '>>': 20,
        '<<': 20,
        '+': 22,
        '-': 22,
        '−': 22,
        '*': 24,
        '⋅': 24,  # U+22C5 DOT OPERATOR
        '·': 24,  # U+00B7 MIDDLE DOT
        '/': 24,
        '//': 24,
        '÷': 24,
        '%': 24,
        '**': 28,
    }


class PrefixOperator(str):
    __slots__ = ()

    _syntax_ = re.compile(r'[+\-−¿~∼]+')
    _precedence_ = 26


class SuffixOperator(str):
    __slots__ = ()

    _syntax_ = re.compile(r'[!?]')
    _precedence_ = 26


class Number(str):
    __slots__ = ()

    _syntax_ = re.compile(r"""
        ( 0b[01][01_]*(\.[01][01_]*)?
        | 0o[0-7][0-7_]*(\.[0-7][0-7_]*)?
        | 0x[0-9-a-fA-F][0-9-a-fA-F_]*(\.[0-9-a-fA-F][0-9-a-fA-F_]*)?
        | [1-9][0-9_]*(\.[0-9][0-9_]*)?
        | 0(\.[0-9][0-9_]*)?
        | \.[0-9][0-9_]*
        )j?
    """, re.VERBOSE)


@dataclass(slots=True)
class Text:
    chunks:  list[str | Expression]

    _syntax_ = one_of(
        ("'", attribute('chunks', subtype=str), "'"),
        ('"', sequence_of(one_of(
            ('{', attribute('chunks', subtype='Expression'), '}'),
            attribute('chunks', subtype=str),
        ), minimum=0), '"'),
    )


@dataclass(slots=True)
class List:
    items:  list[Expression]

    _syntax_ = ('[', attribute('items', separator=','), ']')


@dataclass(slots=True)
class Map:
    items:  dict[Expression, Expression]

    _syntax_ = (
        '{',
        sequence_of(
            key_of_dict('items'),
            ': ',
            value_of_dict('items'),
            minimum=0,
            separator=',',
        ),
        '}',
    )


@dataclass(slots=True)
class Set:
    items:  list[Expression]

    _syntax_ = ('{', attribute('items', separator=','), '}')


type Object = List | Map | Number | Set | Text


@dataclass(slots=True)
class AttributeAccess:
    name:  Name

    _syntax_ = ('.', attribute('name'))


@dataclass(slots=True)
class Argument:
    name:  Name | None
    value:  Expression

    _syntax_ = (optional(attribute('name'), '='), attribute('value'))


@dataclass(slots=True)
class FunctionCall:
    arguments:  list[Argument]

    _syntax_ = ('(', attribute('arguments', separator=','), ')')


@dataclass(slots=True)
class ItemLookup:
    arguments:  list[Expression] | None

    _syntax_ = ('[', attribute('arguments', separator=','), ']')


@dataclass(slots=True)
class PrefixedExpression:
    prefix:  PrefixOperator
    expression:  Expression

    _syntax_ = (attribute('prefix'), attribute('expression'))


@dataclass(slots=True)
class OperationChain:
    expression:  PrefixedExpression | Expression
    operations:  list[AttributeAccess | FunctionCall | ItemLookup | SuffixOperator]

    _syntax_ = (attribute('expression'), attribute('operations', minimum=1))


@dataclass(slots=True)
class InfixExpression:
    first_element:  OperationChain | PrefixedExpression | InfixExpression | Object | Name
    operator:  InfixOperator
    second_element: OperationChain | PrefixedExpression | InfixExpression | Object | Name

    _syntax_ = (
        attribute('first_element', brackets='()'),
        attribute('operator'),
        attribute('second_element', brackets='()'),
    )


@dataclass(slots=True)
class Expression:
    value:  PrefixedExpression | InfixExpression | Object | Name

    _syntax_ = attribute('value', brackets='()')


@dataclass(slots=True)
class TypeReference:
    expression:  Expression
    resolved:  Class | Interface | None  = None

    _syntax_ = attribute('expression')


@dataclass(slots=True)
class Assert:
    invariant:  Expression

    _syntax_ = paragraph('assert', attribute('invariant'))



class AssignmentTarget:
    name:  Name
    subelements:  list[AttributeAccess | ItemLookup]

    _syntax_ = (attribute('name'), attribute('subelements'))


class AssignmentOperator(str):
    __slots__ = ()

    _syntax_ = one_of(
        '//=', '<<=', '>>=', '+=', '-=', '−=', '*=', '⋅=', '·=', '/=', '÷=',
        '%=', '&=', '|=', '^=',
    )


@dataclass(slots=True)
class Assignment:
    targets:  list[AssignmentTarget]
    operator:  AssignmentOperator
    values:  Expression

    _syntax_ = paragraph(
        attribute('targets', separator=','),
        attribute('operator'),
        attribute('values')
    )


@dataclass(slots=True)
class Unpacker:
    names:  list[Name | Unpacker]

    _syntax_ = one_of(
        attribute('names', subtype=Name, minimum=1, maximum=1),
        ('(', attribute('names', minimum=1, separator=','), ')'),
    )


@dataclass(slots=True)
class If:
    condition:  Expression
    instructions:  list[Instruction]

    _syntax_ = (
        paragraph('if', attribute('condition'), ':'),
        indented(attribute('instructions', minimum=1)),
    )


@dataclass(slots=True)
class Elif:
    condition:  Expression
    instructions:  list[Instruction]

    _syntax_ = (
        paragraph('elif', attribute('condition'), ':'),
        indented(attribute('instructions', minimum=1)),
    )


@dataclass(slots=True)
class Else:
    instructions:  list[Instruction]

    _syntax_ = (
        paragraph('else', ':'),
        indented(attribute('instructions', minimum=1)),
    )


@dataclass(slots=True)
class IfElifElse:
    If:  If
    Elifs:  list[Elif]
    Else:  Else

    _syntax_ = (
        attribute('If'),
        attribute('Elif', minimum=0),
        optional(attribute('Else'))
    )


@dataclass(slots=True)
class For:
    element:  list[Unpacker]
    iterable:  Expression
    instructions:  list[Instruction]

    _syntax_ = (
        paragraph(
            'for',
            attribute('element', separator=',', minimum=1),
            'in',
            attribute('iterable'),
            ':',
        ),
        indented(attribute('instructions', minimum=1)),
    )


@dataclass(slots=True)
class Forever:
    instructions:  list[Instruction]

    _syntax_ = (
        paragraph('forever', ':'),
        indented(attribute('instructions', minimum=1)),
    )


@dataclass(slots=True)
class Limit:
    operator:  Literal['<=', '≤', '<']
    value:  Expression

    _syntax_ = (attribute('operator'), attribute('value'))


@dataclass(slots=True)
class Repeat:
    limits:  list[Limit]
    instructions:  list[Instruction]

    _syntax_ = (
        paragraph('repeat', attribute('limits'), ':'),
        indented(attribute('instructions', minimum=1)),
    )

    def check(self) -> None:
        # TODO there should be at most two maxima: a number of iterations and a duration
        pass


@dataclass(slots=True)
class Try:
    instructions:  list[Instruction]

    _syntax_ = (
        paragraph('try', ':'),
        indented(attribute('instructions', minimum=1)),
    )


@dataclass(slots=True)
class Except:
    types:  list[QualifiedName]
    as_name:  Name | None
    instructions:  list[Instruction]

    _syntax_ = (
        paragraph(
            'except',
            optional(
                attribute('types', separator='|'),
                optional(
                    'as', attribute('as_name')
                )
            ),
            ':'
        ),
        indented(attribute('instructions', minimum=1)),
    )


@dataclass(slots=True)
class Finally:
    instructions:  list[Instruction]

    _syntax_ = (
        paragraph('finally', ':'),
        indented(attribute('instructions', minimum=1)),
    )


@dataclass(slots=True)
class TryExceptFinally:
    Try:  Try
    Excepts:  list[Except]
    Finally:  Finally

    _syntax_ = (
        attribute('Try'),
        one_of(
            (attribute('Excepts', minimum=1), optional(attribute('Finally'))),
            (attribute('Finally')),
        )
    )


@dataclass(slots=True)
class Binding:
    expression:  Expression
    name:  Name | None

    _syntax_ = (attribute('expression'), optional('as', attribute('name')))


@dataclass(slots=True)
class With:
    bindings:  list[Binding]
    instructions:  list[Instruction]

    _syntax_ = (
        paragraph('with', attribute('bindings', minimum=1, separator=','), ':'),
        indented(attribute('instructions', minimum=1)),
    )


type Block = IfElifElse | For | Forever | Repeat | TryExceptFinally | With


@dataclass(slots=True)
class Break:
    _syntax_ = paragraph('break')


@dataclass(slots=True)
class Continue:
    _syntax_ = paragraph('continue')


@dataclass(slots=True)
class Computation:
    expression:  Expression

    _syntax_ = paragraph(attribute('expression'))


@dataclass(slots=True)
class Pass:
    _syntax_ = paragraph('pass')


@dataclass(slots=True)
class Raise:
    exception:  Expression

    _syntax_ = paragraph('raise', attribute('exception'))


@dataclass(slots=True)
class Return:
    value:  Expression

    _syntax_ = paragraph('return', attribute('value'))


@dataclass(slots=True)
class Unset:
    names:  list[Name]

    _syntax_ = paragraph('unset', attribute('names', minimum=1))


@dataclass(slots=True)
class Yield:
    value:  Expression

    _syntax_ = paragraph('yield', attribute('value'))


type Instruction = (
    Assert | Assignment | Block | Break | Continue | Computation | Pass |
    Raise | Return | Unset | Yield
)


@dataclass(slots=True)
class ContextVariableReference:
    qualified_name:  QualifiedName
    is_optional:  bool

    _syntax_ = (
        attribute('qualified_name'),
        optional(attribute('is_optional', symbol='optional')),
    )

    @property
    def is_required(self) -> bool:
        return not self.is_optional


@dataclass(slots=True)
class CostElement:
    base:  Name | Number
    exponent:  CostElements | None
    factorial:  bool

    _syntax_ = (
        attribute('base'),
        optional(one_of(
            (one_of('^', '**'), attribute('exponent')),
            (attribute('factorial', symbol='!')),
        )),
    )


@dataclass(slots=True)
class CostFunction:
    name:  Name
    parameter:  CostElements
    exponent:  CostElements | None
    factorial:  bool

    _syntax_ = (
        attribute('name'),
        '(', attribute('parameter'), ')',
        optional(one_of(
            (one_of('^', '**'), attribute('exponent')),
            (attribute('factorial', symbol='!')),
        )),
    )


class CostElementOperator(str):
    __slots__ = ()

    _syntax_ = one_of('+', '*')
    _precedence_ = {'+': 1, '*': 2}


@dataclass(slots=True)
class CostElements:
    elements:  list[CostFunction | CostElement]
    operator:  CostElementOperator | None  = None

    _syntax_ = attribute(
        'elements', minimum=1, separator=attribute('operator'), brackets='()'
    )


@dataclass(slots=True)
class CostExpression:
    # https://en.wikipedia.org/wiki/Big_O_notation

    elements:  CostElements

    _syntax_ = ('O(', attribute('elements'), ')')


@dataclass(slots=True)
class ComplexCost:
    expressions:  dict[Literal['best', 'worst'], CostExpression]

    _syntax_ = (
        key_of_dict('expressions'),
        ':',
        value_of_dict('expressions'),
    )


type Cost = ComplexCost | CostExpression


@dataclass(slots=True)
class Parameter:
    name:  Name
    type:  TypeReference | None
    transformed_type:  TypeReference | None
    default_value:  Expression | None
    description:  Text | None

    _syntax_ = (
        attribute('name'),
        optional(
            ':', attribute('type'),
            optional(one_of('->', '→'), attribute('transformed_type')),
        ),
        optional('=', attribute('default_value')),
        optional('::', indented(
            attribute('description'),
        )),
    )


@dataclass(slots=True)
class Function:
    location:  Location
    name:  Name
    parameters:  list[Parameter]
    declared_return_type:  TypeReference | type[NoReturn] | None
    declared_exceptions:  list[QualifiedName] | None
    declared_costs:  dict[Name, Cost] | None
    description:  Text | None
    instructions:  list[Instruction]

    found_exceptions: set[TypeReference]
        # the exceptions this function can raise, directly or indirectly
    found_context_variables: set[ContextVariableReference]
        # the context variables this function uses, directly or indirectly

    is_finite:  bool  = True
        # detected based on the existence of an infinite loop
    is_multi_threaded:  bool  = False
        # detected based on calls to thread spawning functions
    is_networked:  bool  = False
        # detected based on calls to network functions
    requires_file_system:  bool  = False
        # detected based on calls to file system functions

    _syntax_ = paragraph(
        'def', attribute('name'),
        '(', attribute('parameters', separator=','), ')',
        optional(one_of('->', '→'), attribute('declared_return_type')),
        ':',
        indented(
            optional(attribute('description')),
            any_order(
                optional('raises', attribute('declared_exceptions', separator='|')),
                optional(
                    'costs', ':', indented(paragraphs(
                        key_of_dict('declared_costs'),
                        ':',
                        value_of_dict('declared_costs'),
                    )),
                ),
            ),
            attribute('instructions', minimum=1),
        ),
    )

    @property
    def is_infinite(self) -> bool:
        return not self.is_finite

    @property
    def is_single_threaded(self) -> bool:
        return not self.is_multi_threaded


@dataclass(slots=True)
class Attribute:
    name:  Name
    type:  TypeReference | None
    initial_value:  Expression | None
    other_metadata:  dict[Name, Expression | None]

    _syntax_ = paragraph(
        attribute('name'),
        optional(':', attribute('type')),
        optional('=', attribute('initial_value')),
        optional('::', indented(one_of(
            value_of_dict('other_metadata', key="doc", subtype=Text),
            paragraphs(
                key_of_dict('other_metadata', check='check_metadata_key'),
                optional(':', value_of_dict('other_metadata', check='check_metadata_value')),
            )
        ))),
    )

    def check_metadata_key(self, key: str) -> None:
        if key in ('mutable', 'mutability'):
            raise SyntaxError(
                "attributes are mutable by default; use the `immutable` flag if "
                "you want an attribute to be immutable"
            )

    def check_metadata_value(self, key: str, value: Expression | None) -> None:
        if key in ('immutable', 'private'):
            raise SyntaxError(f"`{key}` should not be followed by a value")

    @property
    def is_immutable(self) -> bool:
        return 'immutable' in self.other_metadata

    @property
    def is_private(self) -> bool:
        return 'private' in self.other_metadata


@dataclass(slots=True)
class Class:
    location:  Location
    name:  Name
    parents:  list[TypeReference]
    description:  Text
    attributes:  list[Attribute]
    hash_components:  tuple[str, ...] | None
    singleton:  bool
    instances_cache:  TypeReference | None
    functions:  list[Function]

    _syntax_ = paragraph(
        optional(attribute('singleton', symbol='singleton')),
        'class',
        attribute('name'),
        optional('(', attribute('parents', separator=',', minimum=0), ')'), # TODO kwargs?
        optional(
            ':', indented(
                optional(attribute('description')),
                attribute('attributes', minimum=0),
                attribute('functions', minimum=0),
                # TODO hash_components
                # TODO instances_cache
            ),
        ),
    )


@dataclass(slots=True)
class ClassExtension:
    location:  Location
    class_name:  QualifiedName
    attributes:  list[Attribute]
    hash_components:  tuple[str, ...] | None
    instances_cache:  TypeReference | None
    functions:  list[Function]

    _syntax_ = paragraph(
        'extend', 'class', attribute('class_name'), ':', indented(
            attribute('attributes', minimum=0),
            attribute('functions', minimum=0),
            # TODO hash_components
            # TODO instances_cache
        ),
    )


@dataclass(slots=True)
class Interface:
    location:  Location
    name:  Name
    attributes:  list[Attribute]
    functions:  list[Function]

    _syntax_ = paragraph(
        'interface', attribute('name'), ':', indented(
            attribute('attributes'),
            attribute('functions'),
        )
    )


type Definition = Assignment | Class | ClassExtension | Function | Interface


@dataclass(slots=True)
class File:
    path:  str
    definitions:  list[Definition]  = field(default_factory=list)

    _syntax_ = attribute('definitions')


file_parser = Parser(File)


@dataclass(slots=True)
class Module:
    name:  str
    is_kernel_part:  bool  = False
    target_kernels:  set[str]  = field(default_factory=set)
    has_command_line_interface:  bool  = False
    requires_garbage_collection:  bool  = False
    files:  list[File]  = field(default_factory=list)
    classes:  dict[Name, Class]  = field(default_factory=dict)
    class_extensions:  list[ClassExtension]  = field(default_factory=list)
    functions:  dict[Name, list[Function]]  = field(default_factory=lambda: defaultdict(list))
    # TODO Multiple functions can have the same name, but not the same parameters.
    interfaces:  dict[Name, Interface]  = field(default_factory=dict)

    @classmethod
    def parse(cls, file_path: str) -> Self:
        module = cls(Path(file_path).stem)
        if os.path.isdir(file_path):
            module._parse_directory(file_path)
        else:
            module._parse_file(file_path)
        return module

    def _parse_directory(self, file_path: str) -> None:
        for entry in os.scandir(file_path):
            if entry.is_dir():
                self._parse_directory(entry.path)
            elif entry.is_file() and entry.name.endswith('.gato'):
                self._parse_file(entry.path)

    def _parse_file(self, file_path: str) -> None:
        file = File(file_path)
        with tokenize.open(file_path) as fd:
            file_parser.parse(fd.read(), file_path, 0, file)
        self.files.append(file)
        # TODO merge the definitions from the file, checking for conflicts

    def compile(self) -> None:
        # TODO
        pass

"Read source code and return a tree of objects suitable for further processing."

from __future__ import annotations

from dataclasses import dataclass, field, KW_ONLY
import inspect
import re
import sys
from types import GenericAlias, NoneType, UnionType
from typing import Iterator, Union, TypeAliasType


UnionTypes = (UnionType, Union[int, str].__class__)


@dataclass(slots=True)
class Location:
    file_path:  str
    line_number:  int
    column:  int


@dataclass(slots=True)
class attribute:
    attr_name:  str
    _: KW_ONLY
    brackets:  str | None  = None
    maximum:  int | None  = None
    minimum:  int | None  = None
    separator:  str | attribute | one_of | None  = None
    subtype:  type | str | None  = None
    symbol:  str | re.Pattern[str] | None  = None

    resolved_type:  type | None  = field(default=None, init=False)


@dataclass(slots=True)
class key_of_dict:
    attr_name:  str
    _: KW_ONLY
    check:  str | None  = None
    subtype:  type | str | None  = None

    resolved_type:  type | None  = field(default=None, init=False)


@dataclass(slots=True)
class value_of_dict:
    attr_name:  str
    _: KW_ONLY
    check:  str | None  = None
    key:  str | None  = None
    subtype:  type | str | None  = None

    resolved_type:  type | None  = field(default=None, init=False)


class any_order:
    __slots__ = ('possibilities',)

    def __init__(self, *possibilities: SyntaxElement):
        self.possibilities = possibilities

    def __iter__(self) -> Iterator[SyntaxElement]:
        return iter(self.possibilities)


class one_of:
    __slots__ = ('possibilities',)

    def __init__(self, *possibilities: SyntaxElement):
        self.possibilities = possibilities

    def __iter__(self) -> Iterator[SyntaxElement]:
        return iter(self.possibilities)


class optional:
    __slots__ = ('elements',)

    def __init__(self, *elements: SyntaxElement):
        self.elements = elements

    def __iter__(self) -> Iterator[SyntaxElement]:
        return iter(self.elements)


class indented:
    __slots__ = ('elements',)

    def __init__(self, *elements: SyntaxElement):
        self.elements = elements

    def __iter__(self) -> Iterator[SyntaxElement]:
        return iter(self.elements)


class paragraph:
    "Tokens followed by a new line."

    __slots__ = ('elements',)

    def __init__(self, *elements: SyntaxElement):
        self.elements = elements

    def __iter__(self) -> Iterator[SyntaxElement]:
        return iter(self.elements)


class paragraphs:
    __slots__ = ('elements', 'minimum')

    def __init__(self, *elements: SyntaxElement, minimum: int = 1):
        self.elements = elements
        self.minimum = minimum

    def __iter__(self) -> Iterator[SyntaxElement]:
        return iter(self.elements)


class sequence_of:
    __slots__ = ('elements', 'minimum', 'separator')

    def __init__(
        self,
        *elements: SyntaxElement,
        minimum: int = 1,
        separator: str | one_of | None = None,
    ):
        self.elements = elements
        self.minimum = minimum
        self.separator = separator

    def __iter__(self) -> Iterator[SyntaxElement]:
        return iter(self.elements)


type Capture = attribute | key_of_dict | value_of_dict
type Branch = any_order | one_of | optional
type Sequence = (
    indented | optional | paragraph | paragraphs | sequence_of | tuple[SyntaxElement, ...]
)
type SyntaxElement = Branch | Capture | Sequence | str | re.Pattern[str]


@dataclass(slots=True)
class Parser[T]:
    root_type:  type[T]

    def __post_init__(self) -> None:
        seen:  set[int]  = set()

        def resolve_class_syntax(cls: type) -> None:
            syntax = getattr(cls, '_syntax_', None)
            if syntax is None:
                raise TypeError(f"{cls!r} lacks a `_syntax_` attribute")
            if not is_subtype(type(syntax), SyntaxElement):
                raise TypeError(
                    f"{cls.__name__}._syntax_ has unexpected type {type(syntax)}"
                )
            if id(cls) in seen:
                return
            seen.add(id(cls))
            annotations = inspect.get_annotations(cls, eval_str=True)
            if not hasattr(syntax, '__iter__'):
                syntax = (syntax,)
            for element in syntax:
                resolve_syntax_element(cls, annotations, element)

        def resolve_syntax_element(
            cls: type, annotations: dict[str, type], element: SyntaxElement,
        ) -> None:
            if id(element) in seen:
                return
            seen.add(id(element))
            if isinstance(element, Capture.__value__):
                attr_name, subtype = element.attr_name, element.subtype
                annotation = annotations.get(attr_name)
                if annotation is None:
                    raise TypeError(
                        f"missing type annotation for attribute `{attr_name}` of {cls!r}"
                    )
                if subtype:
                    if isinstance(subtype, str):
                        subtype = eval(
                            subtype,
                            sys.modules[cls.__module__].__dict__,
                        )
                    if not is_subtype(subtype, annotation):
                        raise TypeError(
                            f"`{subtype}` isn't a subtype of `{annotation}`"
                        )
                    element.resolved_type = subtype
                else:
                    element.resolved_type = annotation
                resolve_attribute_syntax(
                    element.resolved_type, f"{cls.__name__}.{attr_name}"
                )
            elif hasattr(element, '__iter__'):
                for subelement in element:
                    resolve_syntax_element(cls, annotations, element)

        def resolve_attribute_syntax(type_: type, qual_name: str) -> None:
            if isinstance(type_, TypeAliasType):
                type_ = type_.__value__
            if isinstance(type_, GenericAlias):
                base_type = type_.__origin__
                if base_type is list:
                    if len(type_.__args__) != 1:
                        raise TypeError(
                            f"`dict` expects 1 type argument, but {len(type_.__args__)} "
                            f"given in definition of {type_}"
                        )
                    for subtype in type_.__args__:
                        resolve_attribute_syntax(subtype, qual_name)
                elif base_type is dict:
                    if len(type_.__args__) != 2:
                        raise TypeError(
                            f"`dict` expects 2 type arguments, but {len(type_.__args__)} "
                            f"given in definition of {type_}"
                        )
                    for subtype in type_.__args__:
                        resolve_attribute_syntax(subtype, qual_name)
                else:
                    raise TypeError(
                        f"unexpected type {base_type.__name__} in annotation of {qual_name}"
                    )
            elif isinstance(type_, UnionTypes):
                for subtype in type_.__args__:
                    resolve_attribute_syntax(subtype, qual_name)
            elif type_ is not NoneType:
                resolve_class_syntax(type_)

        resolve_class_syntax(self.root_type)
        # TODO look for unused classes with `_syntax_` in the root type's module?

    def parse(self, code: str, file_path: str, line_offset: int, result: T) -> None:
        # TODO
        pass


def is_subtype(t: type, supertype: type | TypeAliasType) -> bool:
    """Check whether `t` can loosely be considered a subtype of `supertype`.

    >>> is_subtype(int, str | int)
    True
    >>> is_subtype(int, str | float)
    False
    >>> is_subtype(tuple, tuple[int, ...])
    True
    """
    if isinstance(supertype, TypeAliasType):
        return is_subtype(t, supertype.__value__)
    if isinstance(supertype, UnionTypes):
        return any(
            is_subtype(t, union_element) for union_element in supertype.__args__
        )
    if isinstance(supertype, GenericAlias):
        if isinstance(t, GenericAlias):
            raise NotImplementedError
        return t == supertype.__origin__
    return t == supertype

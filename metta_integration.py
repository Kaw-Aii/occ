#!/usr/bin/env python3
"""
MeTTa Integration - Hyperon Language Support
============================================

This module provides integration with the MeTTa language for Hyperon-style
cognitive operations. MeTTa (Meta Type Talk) is the language used by
OpenCog Hyperon for expressing cognitive operations.

Key Features:
- MeTTa expression parsing and evaluation
- Integration with hypergraph memory
- Symbolic reasoning through MeTTa
- Pattern matching and rewriting

Author: OpenCog Collection Contributors
License: GPL-3.0+
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class MeTTaTokenType(Enum):
    """Token types for MeTTa expressions."""
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    SYMBOL = "SYMBOL"
    STRING = "STRING"
    NUMBER = "NUMBER"
    VARIABLE = "VARIABLE"
    EOF = "EOF"


@dataclass
class MeTTaToken:
    """Represents a token in MeTTa expression."""
    type: MeTTaTokenType
    value: Any
    position: int


class MeTTaLexer:
    """Lexical analyzer for MeTTa expressions."""
    
    def __init__(self, text: str):
        self.text = text
        self.position = 0
        self.current_char = self.text[0] if text else None
    
    def advance(self):
        """Move to next character."""
        self.position += 1
        self.current_char = self.text[self.position] if self.position < len(self.text) else None
    
    def skip_whitespace(self):
        """Skip whitespace characters."""
        while self.current_char and self.current_char.isspace():
            self.advance()
    
    def read_number(self) -> float:
        """Read a number token."""
        num_str = ""
        while self.current_char and (self.current_char.isdigit() or self.current_char == '.'):
            num_str += self.current_char
            self.advance()
        return float(num_str) if '.' in num_str else int(num_str)
    
    def read_symbol(self) -> str:
        """Read a symbol token."""
        symbol = ""
        while self.current_char and (self.current_char.isalnum() or self.current_char in '-_?!'):
            symbol += self.current_char
            self.advance()
        return symbol
    
    def read_string(self) -> str:
        """Read a string token."""
        self.advance()  # Skip opening quote
        string = ""
        while self.current_char and self.current_char != '"':
            string += self.current_char
            self.advance()
        self.advance()  # Skip closing quote
        return string
    
    def tokenize(self) -> List[MeTTaToken]:
        """Tokenize the MeTTa expression."""
        tokens = []
        
        while self.current_char:
            self.skip_whitespace()
            
            if not self.current_char:
                break
            
            if self.current_char == '(':
                tokens.append(MeTTaToken(MeTTaTokenType.LPAREN, '(', self.position))
                self.advance()
            elif self.current_char == ')':
                tokens.append(MeTTaToken(MeTTaTokenType.RPAREN, ')', self.position))
                self.advance()
            elif self.current_char == '"':
                string_val = self.read_string()
                tokens.append(MeTTaToken(MeTTaTokenType.STRING, string_val, self.position))
            elif self.current_char.isdigit():
                num_val = self.read_number()
                tokens.append(MeTTaToken(MeTTaTokenType.NUMBER, num_val, self.position))
            elif self.current_char == '$':
                self.advance()
                var_name = self.read_symbol()
                tokens.append(MeTTaToken(MeTTaTokenType.VARIABLE, f"${var_name}", self.position))
            else:
                symbol = self.read_symbol()
                tokens.append(MeTTaToken(MeTTaTokenType.SYMBOL, symbol, self.position))
        
        tokens.append(MeTTaToken(MeTTaTokenType.EOF, None, self.position))
        return tokens


@dataclass
class MeTTaExpression:
    """Represents a parsed MeTTa expression."""
    operator: str
    arguments: List[Union['MeTTaExpression', Any]]
    
    def __repr__(self):
        args_str = " ".join(str(arg) for arg in self.arguments)
        return f"({self.operator} {args_str})"


class MeTTaParser:
    """Parser for MeTTa expressions."""
    
    def __init__(self, tokens: List[MeTTaToken]):
        self.tokens = tokens
        self.position = 0
        self.current_token = tokens[0] if tokens else None
    
    def advance(self):
        """Move to next token."""
        self.position += 1
        self.current_token = self.tokens[self.position] if self.position < len(self.tokens) else None
    
    def parse(self) -> Union[MeTTaExpression, Any]:
        """Parse tokens into MeTTa expression."""
        if self.current_token.type == MeTTaTokenType.LPAREN:
            return self.parse_expression()
        elif self.current_token.type == MeTTaTokenType.NUMBER:
            value = self.current_token.value
            self.advance()
            return value
        elif self.current_token.type == MeTTaTokenType.STRING:
            value = self.current_token.value
            self.advance()
            return value
        elif self.current_token.type == MeTTaTokenType.SYMBOL:
            value = self.current_token.value
            self.advance()
            return value
        elif self.current_token.type == MeTTaTokenType.VARIABLE:
            value = self.current_token.value
            self.advance()
            return value
        else:
            raise ValueError(f"Unexpected token: {self.current_token}")
    
    def parse_expression(self) -> MeTTaExpression:
        """Parse a parenthesized expression."""
        self.advance()  # Skip LPAREN
        
        if self.current_token.type != MeTTaTokenType.SYMBOL:
            raise ValueError("Expected operator symbol")
        
        operator = self.current_token.value
        self.advance()
        
        arguments = []
        while self.current_token.type != MeTTaTokenType.RPAREN:
            arguments.append(self.parse())
        
        self.advance()  # Skip RPAREN
        
        return MeTTaExpression(operator, arguments)


class MeTTaInterpreter:
    """
    Interpreter for MeTTa expressions integrated with hypergraph memory.
    """
    
    def __init__(self, memory=None):
        """
        Initialize MeTTa interpreter.
        
        Args:
            memory: Hypergraph memory instance for knowledge storage
        """
        self.memory = memory
        self.bindings = {}
        self.built_in_functions = {
            'add': self._builtin_add,
            'sub': self._builtin_sub,
            'mul': self._builtin_mul,
            'div': self._builtin_div,
            'eq': self._builtin_eq,
            'and': self._builtin_and,
            'or': self._builtin_or,
            'not': self._builtin_not,
            'match': self._builtin_match,
            'bind': self._builtin_bind,
            'get-atom': self._builtin_get_atom,
            'add-atom': self._builtin_add_atom,
            'link-atoms': self._builtin_link_atoms,
        }
    
    def evaluate(self, expression: str) -> Any:
        """
        Evaluate a MeTTa expression.
        
        Args:
            expression: MeTTa expression string
            
        Returns:
            Result of evaluation
        """
        try:
            # Tokenize
            lexer = MeTTaLexer(expression)
            tokens = lexer.tokenize()
            
            # Parse
            parser = MeTTaParser(tokens)
            ast = parser.parse()
            
            # Evaluate
            result = self._eval(ast)
            
            return result
        except Exception as e:
            logger.error(f"Error evaluating MeTTa expression: {e}")
            return None
    
    def _eval(self, node: Union[MeTTaExpression, Any]) -> Any:
        """Recursively evaluate AST node."""
        if isinstance(node, MeTTaExpression):
            operator = node.operator
            
            # Check for built-in functions
            if operator in self.built_in_functions:
                return self.built_in_functions[operator](node.arguments)
            
            # Check for special forms
            if operator == 'if':
                return self._eval_if(node.arguments)
            elif operator == 'let':
                return self._eval_let(node.arguments)
            elif operator == 'lambda':
                return self._eval_lambda(node.arguments)
            
            # Unknown operator
            logger.warning(f"Unknown operator: {operator}")
            return None
        
        elif isinstance(node, str):
            # Variable lookup
            if node.startswith('$'):
                return self.bindings.get(node, None)
            return node
        
        else:
            # Literal value
            return node
    
    def _eval_if(self, args: List) -> Any:
        """Evaluate if expression."""
        if len(args) != 3:
            raise ValueError("if requires 3 arguments: condition, then, else")
        
        condition = self._eval(args[0])
        if condition:
            return self._eval(args[1])
        else:
            return self._eval(args[2])
    
    def _eval_let(self, args: List) -> Any:
        """Evaluate let binding."""
        if len(args) < 2:
            raise ValueError("let requires at least 2 arguments")
        
        # First arg is bindings
        bindings = args[0]
        if isinstance(bindings, MeTTaExpression):
            var_name = bindings.operator
            value = self._eval(bindings.arguments[0])
            self.bindings[var_name] = value
        
        # Evaluate body
        return self._eval(args[1])
    
    def _eval_lambda(self, args: List) -> Any:
        """Evaluate lambda expression."""
        # Placeholder for lambda evaluation
        return f"<lambda {args}>"
    
    # Built-in functions
    
    def _builtin_add(self, args: List) -> float:
        """Addition."""
        return sum(self._eval(arg) for arg in args)
    
    def _builtin_sub(self, args: List) -> float:
        """Subtraction."""
        values = [self._eval(arg) for arg in args]
        return values[0] - sum(values[1:])
    
    def _builtin_mul(self, args: List) -> float:
        """Multiplication."""
        result = 1
        for arg in args:
            result *= self._eval(arg)
        return result
    
    def _builtin_div(self, args: List) -> float:
        """Division."""
        values = [self._eval(arg) for arg in args]
        result = values[0]
        for v in values[1:]:
            result /= v
        return result
    
    def _builtin_eq(self, args: List) -> bool:
        """Equality check."""
        values = [self._eval(arg) for arg in args]
        return all(v == values[0] for v in values[1:])
    
    def _builtin_and(self, args: List) -> bool:
        """Logical AND."""
        return all(self._eval(arg) for arg in args)
    
    def _builtin_or(self, args: List) -> bool:
        """Logical OR."""
        return any(self._eval(arg) for arg in args)
    
    def _builtin_not(self, args: List) -> bool:
        """Logical NOT."""
        return not self._eval(args[0])
    
    def _builtin_match(self, args: List) -> Any:
        """Pattern matching."""
        # Placeholder for pattern matching
        pattern = args[0]
        target = self._eval(args[1])
        return f"<match {pattern} against {target}>"
    
    def _builtin_bind(self, args: List) -> Any:
        """Bind variable."""
        var_name = args[0]
        value = self._eval(args[1])
        self.bindings[var_name] = value
        return value
    
    def _builtin_get_atom(self, args: List) -> Any:
        """Get atom from memory."""
        if not self.memory:
            return None
        
        atom_id = self._eval(args[0])
        atom = self.memory.get_atom(atom_id)
        return atom
    
    def _builtin_add_atom(self, args: List) -> str:
        """Add atom to memory."""
        if not self.memory:
            return None
        
        from cognitive_synergy_framework import Atom
        
        atom_type = self._eval(args[0])
        atom_name = self._eval(args[1])
        
        atom = Atom(atom_type=atom_type, name=atom_name)
        return self.memory.add_atom(atom)
    
    def _builtin_link_atoms(self, args: List) -> None:
        """Link atoms in memory."""
        if not self.memory:
            return None
        
        source_id = self._eval(args[0])
        target_id = self._eval(args[1])
        link_type = self._eval(args[2]) if len(args) > 2 else "generic"
        
        self.memory.link_atoms(source_id, target_id, link_type)
        return f"Linked {source_id} -> {target_id}"


def demo_metta_integration():
    """Demonstrate MeTTa integration capabilities."""
    print("MeTTa Integration Demo")
    print("=" * 60)
    
    interpreter = MeTTaInterpreter()
    
    # Test basic arithmetic
    print("\n1. Basic Arithmetic:")
    result = interpreter.evaluate("(add 1 2 3)")
    print(f"   (add 1 2 3) = {result}")
    
    result = interpreter.evaluate("(mul 2 3 4)")
    print(f"   (mul 2 3 4) = {result}")
    
    # Test logical operations
    print("\n2. Logical Operations:")
    result = interpreter.evaluate("(and 1 1)")
    print(f"   (and 1 1) = {result}")
    
    result = interpreter.evaluate("(or 0 1)")
    print(f"   (or 0 1) = {result}")
    
    # Test variable binding
    print("\n3. Variable Binding:")
    interpreter.evaluate("(bind $x 42)")
    result = interpreter.evaluate("(add $x 8)")
    print(f"   After (bind $x 42), (add $x 8) = {result}")
    
    # Test with memory
    print("\n4. Hypergraph Operations:")
    from cognitive_synergy_framework import HypergraphMemory
    memory = HypergraphMemory()
    interpreter.memory = memory
    
    result = interpreter.evaluate('(add-atom "Concept" "TestConcept")')
    print(f"   Added atom: {result}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo_metta_integration()


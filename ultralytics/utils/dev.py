# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import inspect
import numpy as np
import torch
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import traceback

def describe_var(obj: Any, max_depth: int = 5, max_items: int = 5, max_str_len: int = 80, current_depth: int = 0, indent_size: int = 0) -> str:
    """
    Describe any variable with detailed information about its type, structure, and content.
    
    This function analyzes variables of any type and provides information about:
    - Type information
    - Shape/size for containers, arrays, and tensors
    - Content preview, including nested structures
    - Data type for numeric arrays and tensors
    - Object attributes for custom classes
    
    Args:
        obj (Any): The object to describe
        max_depth (int, optional): Maximum depth to traverse for nested objects. Defaults to 5.
        max_items (int, optional): Maximum number of items to show in containers. Defaults to 5.
        max_str_len (int, optional): Maximum string length for previews. Defaults to 80.
        current_depth (int, optional): Current recursion depth, used internally. Defaults to 0.
        indent_size (int, optional): Number of spaces to indent each line of the output. Defaults to 0.
    
    Returns:
        str: A formatted description of the object

    Examples:
        >>> describe_var(42)
        'int: 42'
        
        >>> describe_var([1, 2, 3, 4, 5, 6])
        'list[6]: [1, 2, 3, 4, 5, ...] (1 more)'
        
        >>> describe_var({'a': [1, 2], 'b': torch.tensor([3, 4])})
        'dict[2]: {
          "a": list[2]: [1, 2]
          "b": torch.Tensor(shape=[2], dtype=torch.int64): tensor([3, 4])
        }'
        
        >>> describe_var([1, 2, 3], indent_size=4)
        '    list[3]: [1, 2, 3]'
    """
    # Recursive indentation based on depth
    indent = "  " * current_depth
    next_indent = "  " * (current_depth + 1)
    
    # Global indent for all lines (based on indent_size parameter)
    global_indent = " " * indent_size
    
    # Base case: reached maximum recursion depth
    if current_depth >= max_depth:
        return f"{global_indent}{type(obj).__name__}: <max depth reached>"
    
    # Handle None
    if obj is None:
        return f"{global_indent}None"
    
    # Handle basic types
    if isinstance(obj, (int, float, bool, complex)):
        return f"{global_indent}{type(obj).__name__}: {obj}"
    
    # Handle strings with truncation
    if isinstance(obj, str):
        if len(obj) > max_str_len:
            return f"{global_indent}str[{len(obj)}]: '{obj[:max_str_len]}...' (truncated)"
        return f"{global_indent}str[{len(obj)}]: '{obj}'"
    
    # Handle Path objects
    if isinstance(obj, Path):
        return f"{global_indent}Path: {obj}"
    
    # Handle torch.Tensor objects
    if isinstance(obj, torch.Tensor):
        device_info = f", device={obj.device}" if obj.device.type != "cpu" else ""
        preview = str(obj.detach().cpu().numpy() if obj.requires_grad else obj)
        if len(preview) > max_str_len:
            preview = f"{preview[:max_str_len]}... (truncated)"
        return f"{global_indent}torch.Tensor(shape={list(obj.shape)}, dtype={obj.dtype}{device_info}): {preview}"
    
    # Handle numpy arrays
    if isinstance(obj, np.ndarray):
        preview = str(obj)
        if len(preview) > max_str_len:
            preview = f"{preview[:max_str_len]}... (truncated)"
        return f"{global_indent}np.ndarray(shape={obj.shape}, dtype={obj.dtype}): {preview}"
    
    # Handle dictionaries
    if isinstance(obj, Mapping):
        if not obj:
            return f"{global_indent}{type(obj).__name__}[0]: {{}}"
        
        result = [f"{global_indent}{type(obj).__name__}[{len(obj)}]: {{"]
        
        items = list(obj.items())
        shown_items = items[:max_items]
        remaining = max(0, len(items) - max_items)
        
        for key, value in shown_items:
            key_str = f'"{key}"' if isinstance(key, str) else str(key)
            desc = describe_var(value, max_depth, max_items, max_str_len, current_depth + 1, indent_size)
            # Remove the global_indent from the nested description since we'll add it back when joining
            if desc.startswith(global_indent):
                desc = desc[len(global_indent):]
            result.append(f"{global_indent}{next_indent}{key_str}: {desc}")
        
        if remaining > 0:
            result.append(f"{global_indent}{next_indent}... ({remaining} more items)")
        
        result.append(f"{global_indent}{indent}}}")
        return "\n".join(result)
    
    # Handle lists, tuples, sets, and other sequences
    if isinstance(obj, Sequence) and not isinstance(obj, str):
        container_type = type(obj).__name__
        
        if not obj:
            return f"{global_indent}{container_type}[0]: []"
        
        shown_items = obj[:max_items]
        remaining = max(0, len(obj) - max_items)
        
        # Check if items are primitive types for compact display
        all_simple = all(isinstance(item, (int, float, bool, str, type(None))) for item in shown_items)
        
        if all_simple and current_depth < max_depth - 1:
            items_desc = []
            for item in shown_items:
                if isinstance(item, str):
                    # For strings, use a shorter representation in compact mode
                    if len(item) > 20:
                        items_desc.append(f"'{item[:17]}...'")
                    else:
                        items_desc.append(f"'{item}'")
                else:
                    items_desc.append(str(item))
            
            items_str = ", ".join(items_desc)
            if len(items_str) <= max_str_len:
                result = f"{global_indent}{container_type}[{len(obj)}]: [{items_str}"
                if remaining > 0:
                    result += f", ... ({remaining} more items)"
                result += "]"
                return result
        
        # For complex items or long representations, use multi-line format
        result = [f"{global_indent}{container_type}[{len(obj)}]: ["]
        for item in shown_items:
            desc = describe_var(item, max_depth, max_items, max_str_len, current_depth + 1, indent_size)
            # Remove the global_indent from the nested description since we'll add it back when joining
            if desc.startswith(global_indent):
                desc = desc[len(global_indent):]
            result.append(f"{global_indent}{next_indent}{desc}")
        if remaining > 0:
            result.append(f"{global_indent}{next_indent}... ({remaining} more items)")
        result.append(f"{global_indent}{indent}]")
        return "\n".join(result)
    
    # Handle sets and frozensets
    if isinstance(obj, (set, frozenset)):
        container_type = type(obj).__name__
        
        if not obj:
            return f"{global_indent}{container_type}[0]: {{ }}"
        
        shown_items = list(obj)[:max_items]
        remaining = max(0, len(obj) - max_items)
        
        # Check if items are primitive types for compact display
        all_simple = all(isinstance(item, (int, float, bool, str, type(None))) for item in shown_items)
        
        if all_simple and current_depth < max_depth - 1:
            items_desc = []
            for item in shown_items:
                if isinstance(item, str):
                    if len(item) > 20:
                        items_desc.append(f"'{item[:17]}...'")
                    else:
                        items_desc.append(f"'{item}'")
                else:
                    items_desc.append(str(item))
            
            items_str = ", ".join(items_desc)
            if len(items_str) <= max_str_len:
                result = f"{global_indent}{container_type}[{len(obj)}]: {{{items_str}"
                if remaining > 0:
                    result += f", ... ({remaining} more items)"
                result += "}"
                return result
        
        # For complex items or long representations, use multi-line format
        result = [f"{global_indent}{container_type}[{len(obj)}]: {{"]
        for item in shown_items:
            desc = describe_var(item, max_depth, max_items, max_str_len, current_depth + 1, indent_size)
            # Remove the global_indent from the nested description since we'll add it back when joining
            if desc.startswith(global_indent):
                desc = desc[len(global_indent):]
            result.append(f"{global_indent}{next_indent}{desc}")
        if remaining > 0:
            result.append(f"{global_indent}{next_indent}... ({remaining} more items)")
        result.append(f"{global_indent}{indent}}}")
        return "\n".join(result)
    
    # Handle custom objects and classes
    try:
        # Try to get attributes for custom objects
        if hasattr(obj, "__dict__"):
            attrs = obj.__dict__
            if attrs:
                # Get class name, and simplify it if it's a common pattern
                class_name = type(obj).__name__
                result = [f"{global_indent}{class_name} object with attributes:"]
                
                items = list(attrs.items())
                shown_items = items[:max_items]
                remaining = max(0, len(items) - max_items)
                
                for key, value in shown_items:
                    if not key.startswith("_"):  # Skip private attributes
                        desc = describe_var(value, max_depth, max_items, max_str_len, current_depth + 1, indent_size)
                        # Remove the global_indent from the nested description since we'll add it back when joining
                        if desc.startswith(global_indent):
                            desc = desc[len(global_indent):]
                        result.append(f"{global_indent}{next_indent}.{key} = {desc}")
                
                if remaining > 0:
                    result.append(f"{global_indent}{next_indent}... ({remaining} more attributes)")
                
                return "\n".join(result)
    except Exception:
        pass
    
    # Fallback for any other type
    try:
        preview = str(obj)
        if len(preview) > max_str_len:
            preview = f"{preview[:max_str_len]}... (truncated)"
        return f"{global_indent}{type(obj).__name__}: {preview}"
    except Exception:
        return f"{global_indent}{type(obj).__name__}: <cannot display>"

def show_caller():
    stack = traceback.extract_stack()
    # 移除最後一個元素，因為它是當前函數
    stack = stack[:-1]
    print("Call stack:")
    for frame in stack:
        filename, line_number, function_name, code = frame
        print(f"  File: {filename}, Line: {line_number}, Function: {function_name}, Code: {code}")

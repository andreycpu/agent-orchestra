"""
Data transformation utilities for Agent Orchestra.

This module provides data transformation, serialization, validation,
and conversion utilities for processing data between different formats.
"""
import json
import csv
import yaml
import xml.etree.ElementTree as ET
import base64
import gzip
import pickle
from typing import Any, Dict, List, Optional, Union, Callable, Type, Iterator
from datetime import datetime, date
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from io import StringIO, BytesIO
import logging

from .exceptions import ValidationError, SerializationError
from .validation import validate_json_serializable


logger = logging.getLogger(__name__)


class DataFormat(str):
    """Supported data formats."""
    JSON = "json"
    YAML = "yaml"
    XML = "xml"
    CSV = "csv"
    PICKLE = "pickle"
    BASE64 = "base64"
    PLAIN = "plain"


@dataclass
class TransformationResult:
    """Result of data transformation."""
    data: Any
    format: str
    size_bytes: int
    transformation_time_ms: float
    metadata: Dict[str, Any]
    
    def is_success(self) -> bool:
        """Check if transformation was successful."""
        return self.data is not None


class DataConverter:
    """Converts data between different formats."""
    
    @staticmethod
    def to_json(
        data: Any,
        pretty: bool = False,
        sort_keys: bool = False,
        ensure_ascii: bool = False
    ) -> str:
        """Convert data to JSON string."""
        try:
            validate_json_serializable(data)
            
            kwargs = {
                'ensure_ascii': ensure_ascii,
                'sort_keys': sort_keys
            }
            
            if pretty:
                kwargs['indent'] = 2
                kwargs['separators'] = (',', ': ')
            else:
                kwargs['separators'] = (',', ':')
            
            return json.dumps(data, **kwargs, default=DataConverter._json_default)
        
        except Exception as e:
            raise SerializationError(f"Failed to convert to JSON: {e}") from e
    
    @staticmethod
    def from_json(json_str: str) -> Any:
        """Convert JSON string to Python data."""
        try:
            return json.loads(json_str)
        except Exception as e:
            raise SerializationError(f"Failed to parse JSON: {e}") from e
    
    @staticmethod
    def to_yaml(data: Any, pretty: bool = False) -> str:
        """Convert data to YAML string."""
        try:
            kwargs = {
                'default_flow_style': not pretty,
                'allow_unicode': True
            }
            
            if pretty:
                kwargs['indent'] = 2
                kwargs['width'] = 80
            
            return yaml.dump(data, **kwargs)
        
        except Exception as e:
            raise SerializationError(f"Failed to convert to YAML: {e}") from e
    
    @staticmethod
    def from_yaml(yaml_str: str) -> Any:
        """Convert YAML string to Python data."""
        try:
            return yaml.safe_load(yaml_str)
        except Exception as e:
            raise SerializationError(f"Failed to parse YAML: {e}") from e
    
    @staticmethod
    def to_xml(data: Dict[str, Any], root_name: str = "root") -> str:
        """Convert dictionary to XML string."""
        try:
            root = ET.Element(root_name)
            DataConverter._dict_to_xml(data, root)
            return ET.tostring(root, encoding='unicode')
        
        except Exception as e:
            raise SerializationError(f"Failed to convert to XML: {e}") from e
    
    @staticmethod
    def from_xml(xml_str: str) -> Dict[str, Any]:
        """Convert XML string to dictionary."""
        try:
            root = ET.fromstring(xml_str)
            return DataConverter._xml_to_dict(root)
        
        except Exception as e:
            raise SerializationError(f"Failed to parse XML: {e}") from e
    
    @staticmethod
    def to_csv(
        data: List[Dict[str, Any]],
        fieldnames: Optional[List[str]] = None,
        delimiter: str = ','
    ) -> str:
        """Convert list of dictionaries to CSV string."""
        if not data:
            return ""
        
        try:
            output = StringIO()
            
            if fieldnames is None:
                fieldnames = list(data[0].keys()) if data else []
            
            writer = csv.DictWriter(output, fieldnames=fieldnames, delimiter=delimiter)
            writer.writeheader()
            writer.writerows(data)
            
            return output.getvalue()
        
        except Exception as e:
            raise SerializationError(f"Failed to convert to CSV: {e}") from e
    
    @staticmethod
    def from_csv(
        csv_str: str,
        delimiter: str = ',',
        skip_header: bool = False
    ) -> List[Dict[str, Any]]:
        """Convert CSV string to list of dictionaries."""
        try:
            input_stream = StringIO(csv_str)
            reader = csv.DictReader(input_stream, delimiter=delimiter)
            
            if skip_header:
                next(reader, None)
            
            return list(reader)
        
        except Exception as e:
            raise SerializationError(f"Failed to parse CSV: {e}") from e
    
    @staticmethod
    def to_base64(data: Union[str, bytes]) -> str:
        """Convert data to base64 string."""
        try:
            if isinstance(data, str):
                data = data.encode('utf-8')
            return base64.b64encode(data).decode('ascii')
        
        except Exception as e:
            raise SerializationError(f"Failed to encode base64: {e}") from e
    
    @staticmethod
    def from_base64(base64_str: str) -> bytes:
        """Convert base64 string to bytes."""
        try:
            return base64.b64decode(base64_str)
        
        except Exception as e:
            raise SerializationError(f"Failed to decode base64: {e}") from e
    
    @staticmethod
    def to_pickle(data: Any) -> bytes:
        """Convert data to pickle bytes."""
        try:
            return pickle.dumps(data)
        
        except Exception as e:
            raise SerializationError(f"Failed to pickle data: {e}") from e
    
    @staticmethod
    def from_pickle(pickle_data: bytes) -> Any:
        """Convert pickle bytes to data."""
        try:
            return pickle.loads(pickle_data)
        
        except Exception as e:
            raise SerializationError(f"Failed to unpickle data: {e}") from e
    
    @staticmethod
    def _json_default(obj):
        """Default JSON serializer for custom types."""
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif is_dataclass(obj):
            return {field.name: getattr(obj, field.name) for field in fields(obj)}
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)
    
    @staticmethod
    def _dict_to_xml(data: Dict[str, Any], parent: ET.Element):
        """Convert dictionary to XML elements."""
        for key, value in data.items():
            element = ET.SubElement(parent, str(key))
            
            if isinstance(value, dict):
                DataConverter._dict_to_xml(value, element)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        item_elem = ET.SubElement(element, "item")
                        DataConverter._dict_to_xml(item, item_elem)
                    else:
                        item_elem = ET.SubElement(element, "item")
                        item_elem.text = str(item)
            else:
                element.text = str(value)
    
    @staticmethod
    def _xml_to_dict(element: ET.Element) -> Dict[str, Any]:
        """Convert XML element to dictionary."""
        result = {}
        
        # Handle element text
        if element.text and element.text.strip():
            result['_text'] = element.text.strip()
        
        # Handle attributes
        if element.attrib:
            result['_attributes'] = element.attrib
        
        # Handle child elements
        children = {}
        for child in element:
            child_data = DataConverter._xml_to_dict(child)
            
            if child.tag in children:
                # Convert to list if multiple children with same tag
                if not isinstance(children[child.tag], list):
                    children[child.tag] = [children[child.tag]]
                children[child.tag].append(child_data)
            else:
                children[child.tag] = child_data
        
        result.update(children)
        
        return result


class DataValidator:
    """Validates data against schemas and constraints."""
    
    @staticmethod
    def validate_structure(data: Any, schema: Dict[str, Any]) -> List[str]:
        """Validate data structure against schema."""
        errors = []
        
        if 'type' in schema:
            expected_type = schema['type']
            if not DataValidator._check_type(data, expected_type):
                errors.append(f"Expected type {expected_type}, got {type(data).__name__}")
        
        if 'required' in schema and isinstance(data, dict):
            required_fields = schema['required']
            for field in required_fields:
                if field not in data:
                    errors.append(f"Required field '{field}' is missing")
        
        if 'properties' in schema and isinstance(data, dict):
            properties = schema['properties']
            for key, value in data.items():
                if key in properties:
                    field_errors = DataValidator.validate_structure(value, properties[key])
                    errors.extend([f"{key}.{error}" for error in field_errors])
        
        if 'items' in schema and isinstance(data, list):
            item_schema = schema['items']
            for i, item in enumerate(data):
                item_errors = DataValidator.validate_structure(item, item_schema)
                errors.extend([f"[{i}].{error}" for error in item_errors])
        
        return errors
    
    @staticmethod
    def validate_constraints(data: Any, constraints: Dict[str, Any]) -> List[str]:
        """Validate data against constraints."""
        errors = []
        
        if 'min_value' in constraints:
            min_val = constraints['min_value']
            if isinstance(data, (int, float)) and data < min_val:
                errors.append(f"Value {data} is less than minimum {min_val}")
        
        if 'max_value' in constraints:
            max_val = constraints['max_value']
            if isinstance(data, (int, float)) and data > max_val:
                errors.append(f"Value {data} is greater than maximum {max_val}")
        
        if 'min_length' in constraints:
            min_len = constraints['min_length']
            if hasattr(data, '__len__') and len(data) < min_len:
                errors.append(f"Length {len(data)} is less than minimum {min_len}")
        
        if 'max_length' in constraints:
            max_len = constraints['max_length']
            if hasattr(data, '__len__') and len(data) > max_len:
                errors.append(f"Length {len(data)} is greater than maximum {max_len}")
        
        if 'pattern' in constraints and isinstance(data, str):
            import re
            pattern = constraints['pattern']
            if not re.match(pattern, data):
                errors.append(f"Value '{data}' does not match pattern '{pattern}'")
        
        if 'enum' in constraints:
            enum_values = constraints['enum']
            if data not in enum_values:
                errors.append(f"Value '{data}' is not in allowed values {enum_values}")
        
        return errors
    
    @staticmethod
    def _check_type(data: Any, expected_type: str) -> bool:
        """Check if data matches expected type."""
        type_mapping = {
            'string': str,
            'integer': int,
            'number': (int, float),
            'boolean': bool,
            'array': list,
            'object': dict,
            'null': type(None)
        }
        
        expected_python_type = type_mapping.get(expected_type)
        if expected_python_type is None:
            return True  # Unknown type, assume valid
        
        return isinstance(data, expected_python_type)


class DataTransformer:
    """Transforms data between formats with validation."""
    
    def __init__(self):
        self.converter = DataConverter()
        self.validator = DataValidator()
    
    def transform(
        self,
        data: Any,
        source_format: str,
        target_format: str,
        validation_schema: Optional[Dict[str, Any]] = None,
        **format_options
    ) -> TransformationResult:
        """Transform data from source format to target format."""
        import time
        start_time = time.time()
        
        try:
            # Parse source data if needed
            if source_format != DataFormat.PLAIN:
                parsed_data = self._parse_data(data, source_format)
            else:
                parsed_data = data
            
            # Validate data if schema provided
            if validation_schema:
                errors = self.validator.validate_structure(parsed_data, validation_schema)
                if errors:
                    raise ValidationError(f"Data validation failed: {errors}")
            
            # Convert to target format
            if target_format == DataFormat.PLAIN:
                result_data = parsed_data
            else:
                result_data = self._convert_data(parsed_data, target_format, **format_options)
            
            # Calculate metrics
            duration_ms = (time.time() - start_time) * 1000
            size_bytes = len(str(result_data)) if isinstance(result_data, (str, bytes)) else 0
            
            return TransformationResult(
                data=result_data,
                format=target_format,
                size_bytes=size_bytes,
                transformation_time_ms=duration_ms,
                metadata={
                    'source_format': source_format,
                    'target_format': target_format,
                    'validation_performed': validation_schema is not None
                }
            )
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Data transformation failed: {e}")
            
            return TransformationResult(
                data=None,
                format=target_format,
                size_bytes=0,
                transformation_time_ms=duration_ms,
                metadata={
                    'error': str(e),
                    'source_format': source_format,
                    'target_format': target_format
                }
            )
    
    def _parse_data(self, data: Any, format_type: str) -> Any:
        """Parse data from specific format."""
        if format_type == DataFormat.JSON:
            return self.converter.from_json(data)
        elif format_type == DataFormat.YAML:
            return self.converter.from_yaml(data)
        elif format_type == DataFormat.XML:
            return self.converter.from_xml(data)
        elif format_type == DataFormat.CSV:
            return self.converter.from_csv(data)
        elif format_type == DataFormat.BASE64:
            return self.converter.from_base64(data).decode('utf-8')
        elif format_type == DataFormat.PICKLE:
            return self.converter.from_pickle(data)
        else:
            return data
    
    def _convert_data(self, data: Any, format_type: str, **options) -> Any:
        """Convert data to specific format."""
        if format_type == DataFormat.JSON:
            return self.converter.to_json(data, **options)
        elif format_type == DataFormat.YAML:
            return self.converter.to_yaml(data, **options)
        elif format_type == DataFormat.XML:
            root_name = options.get('root_name', 'root')
            return self.converter.to_xml(data, root_name)
        elif format_type == DataFormat.CSV:
            return self.converter.to_csv(data, **options)
        elif format_type == DataFormat.BASE64:
            return self.converter.to_base64(str(data))
        elif format_type == DataFormat.PICKLE:
            return self.converter.to_pickle(data)
        else:
            return data


class DataProcessor:
    """Processes and manipulates data with various operations."""
    
    @staticmethod
    def filter_data(data: List[Dict[str, Any]], predicate: Callable[[Dict[str, Any]], bool]) -> List[Dict[str, Any]]:
        """Filter data based on predicate function."""
        return [item for item in data if predicate(item)]
    
    @staticmethod
    def map_data(data: List[Any], mapper: Callable[[Any], Any]) -> List[Any]:
        """Map data using transformer function."""
        return [mapper(item) for item in data]
    
    @staticmethod
    def group_by(data: List[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
        """Group data by specified key."""
        groups = {}
        for item in data:
            group_key = item.get(key)
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(item)
        return groups
    
    @staticmethod
    def sort_data(data: List[Dict[str, Any]], key: str, reverse: bool = False) -> List[Dict[str, Any]]:
        """Sort data by specified key."""
        return sorted(data, key=lambda x: x.get(key), reverse=reverse)
    
    @staticmethod
    def aggregate_data(data: List[Dict[str, Any]], group_key: str, agg_operations: Dict[str, Callable]) -> List[Dict[str, Any]]:
        """Aggregate data with specified operations."""
        groups = DataProcessor.group_by(data, group_key)
        results = []
        
        for group_value, group_items in groups.items():
            result = {group_key: group_value}
            
            for field, operation in agg_operations.items():
                values = [item.get(field) for item in group_items if item.get(field) is not None]
                if values:
                    result[f"{field}_{operation.__name__}"] = operation(values)
            
            results.append(result)
        
        return results
    
    @staticmethod
    def pivot_data(
        data: List[Dict[str, Any]],
        index_col: str,
        pivot_col: str,
        value_col: str,
        agg_func: Callable = lambda x: x[0] if x else None
    ) -> List[Dict[str, Any]]:
        """Pivot data from long to wide format."""
        # Group by index column
        groups = DataProcessor.group_by(data, index_col)
        results = []
        
        # Get all unique pivot values
        pivot_values = set()
        for item in data:
            pivot_val = item.get(pivot_col)
            if pivot_val is not None:
                pivot_values.add(pivot_val)
        
        for index_value, group_items in groups.items():
            result = {index_col: index_value}
            
            # Create pivot columns
            pivot_groups = DataProcessor.group_by(group_items, pivot_col)
            for pivot_val in pivot_values:
                if pivot_val in pivot_groups:
                    values = [item.get(value_col) for item in pivot_groups[pivot_val]]
                    result[str(pivot_val)] = agg_func(values)
                else:
                    result[str(pivot_val)] = None
            
            results.append(result)
        
        return results
    
    @staticmethod
    def clean_data(data: List[Dict[str, Any]], cleaning_rules: Dict[str, Callable]) -> List[Dict[str, Any]]:
        """Clean data using specified cleaning rules."""
        cleaned_data = []
        
        for item in data:
            cleaned_item = {}
            for key, value in item.items():
                if key in cleaning_rules:
                    cleaned_value = cleaning_rules[key](value)
                    cleaned_item[key] = cleaned_value
                else:
                    cleaned_item[key] = value
            cleaned_data.append(cleaned_item)
        
        return cleaned_data


class DataBatch:
    """Processes data in batches."""
    
    def __init__(self, batch_size: int = 100):
        self.batch_size = batch_size
    
    def process_batches(
        self,
        data: List[Any],
        processor: Callable[[List[Any]], List[Any]]
    ) -> Iterator[List[Any]]:
        """Process data in batches."""
        for i in range(0, len(data), self.batch_size):
            batch = data[i:i + self.batch_size]
            yield processor(batch)
    
    def process_all_batches(
        self,
        data: List[Any],
        processor: Callable[[List[Any]], List[Any]]
    ) -> List[Any]:
        """Process all batches and return combined results."""
        results = []
        for batch_result in self.process_batches(data, processor):
            results.extend(batch_result)
        return results


# Global transformer instance
data_transformer = DataTransformer()


def transform_data(
    data: Any,
    source_format: str,
    target_format: str,
    **options
) -> TransformationResult:
    """Transform data between formats."""
    return data_transformer.transform(data, source_format, target_format, **options)


def validate_data(data: Any, schema: Dict[str, Any]) -> List[str]:
    """Validate data against schema."""
    validator = DataValidator()
    return validator.validate_structure(data, schema)
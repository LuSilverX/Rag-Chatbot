"""Convert structured uploads into searchable text while retaining field context."""
import csv
import io
import json
from decimal import Decimal


class StructuredInputError(ValueError):
    pass


class TextBuilder:
    def __init__(self, limit):
        self.limit = limit
        self.parts = []
        self.length = 0

    def add(self, line):
        self.length += len(line) + 1
        if self.length > self.limit:
            raise StructuredInputError(f"Converted document exceeds the {self.limit:,}-character limit. Use a smaller file.")
        self.parts.append(line)

    def finish(self):
        if not self.parts:
            raise StructuredInputError("The file contains no searchable values.")
        return '\n'.join(self.parts)


def json_to_text(raw, limit):
    def object_fields(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise StructuredInputError(f"JSON contains a duplicate field: {key[:80]}.")
            result[key] = value
        return result

    def invalid_constant(value):
        raise StructuredInputError("JSON cannot contain NaN or Infinity.")

    try:
        value = json.loads(raw, object_pairs_hook=object_fields, parse_float=Decimal, parse_constant=invalid_constant)
    except StructuredInputError:
        raise
    except (ValueError, RecursionError):
        raise StructuredInputError("Choose valid JSON containing an object or array.")
    if not isinstance(value, (dict, list)):
        raise StructuredInputError("The JSON document must contain an object or array.")
    output = TextBuilder(limit)

    def walk(item, path, depth):
        if depth > 20:
            raise StructuredInputError("JSON nesting may be at most 20 levels deep.")
        if isinstance(item, dict):
            for key, child in item.items():
                walk(child, f'{path}[{json.dumps(key, ensure_ascii=False)}]', depth + 1)
        elif isinstance(item, list):
            for index, child in enumerate(item):
                walk(child, f'{path}[{index}]', depth + 1)
        elif isinstance(item, str) and not item.strip():
            return
        else:
            scalar = str(item) if isinstance(item, Decimal) else json.dumps(item, ensure_ascii=False)
            output.add(f'{path}: {scalar}')

    walk(value, '$', 0)
    return output.finish()


def csv_to_text(raw, limit):
    output = TextBuilder(limit)
    try:
        reader = csv.reader(io.StringIO(raw, newline=''), strict=True)
        headers = next(reader, None)
        if not headers or any(not header.strip() for header in headers):
            raise StructuredInputError("CSV needs a first row with non-empty column names.")
        headers = [header.strip() for header in headers]
        if len(headers) > 100:
            raise StructuredInputError("CSV may have at most 100 columns.")
        if len(set(headers)) != len(headers):
            raise StructuredInputError("CSV column names must be unique.")
        for index, row in enumerate(reader, 1):
            if not row or all(not cell.strip() for cell in row):
                continue
            if len(row) != len(headers):
                raise StructuredInputError(f"CSV record {index} has {len(row)} values; expected {len(headers)}.")
            for header, cell in zip(headers, row):
                output.add(f'Row {index} | {json.dumps(header, ensure_ascii=False)}: {json.dumps(cell, ensure_ascii=False)}')
    except csv.Error:
        raise StructuredInputError("CSV has invalid quoting or a field longer than the parser supports (128 KB).")
    return output.finish()

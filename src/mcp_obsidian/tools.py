from collections.abc import Sequence
from mcp.types import (
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
)
import json
import os
import re
from . import obsidian, server

api_key = os.getenv("OBSIDIAN_API_KEY", "")
obsidian_host = os.getenv("OBSIDIAN_HOST", "127.0.0.1")

if api_key == "":
    raise ValueError(f"OBSIDIAN_API_KEY environment variable required. Working directory: {os.getcwd()}")

TOOL_LIST_FILES_IN_VAULT = "obsidian_list_files_in_vault"
TOOL_LIST_FILES_IN_DIR = "obsidian_list_files_in_dir"

class ToolHandler():
    def __init__(self, tool_name: str):
        self.name = tool_name

    def get_tool_description(self) -> Tool:
        raise NotImplementedError()

    def run_tool(self, args: dict) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
        raise NotImplementedError()
    
class ListFilesInVaultToolHandler(ToolHandler):
    def __init__(self):
        super().__init__(TOOL_LIST_FILES_IN_VAULT)

    def get_tool_description(self):
        return Tool(
            name=self.name,
            description="Lists all files and directories in the root directory of your Obsidian vault.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            },
        )

    def run_tool(self, args: dict) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
        api = obsidian.Obsidian(api_key=api_key, host=obsidian_host)

        files = api.list_files_in_vault()

        return [
            TextContent(
                type="text",
                text=json.dumps(files, indent=2)
            )
        ]
    
class ListFilesInDirToolHandler(ToolHandler):
    def __init__(self):
        super().__init__(TOOL_LIST_FILES_IN_DIR)

    def get_tool_description(self):
        return Tool(
            name=self.name,
            description="Lists all files and directories that exist in a specific Obsidian directory.",
            inputSchema={
                "type": "object",
                "properties": {
                    "dirpath": {
                        "type": "string",
                        "description": "Path to list files from (relative to your vault root). Note that empty directories will not be returned."
                    },
                },
                "required": ["dirpath"]
            }
        )

    def run_tool(self, args: dict) -> Sequence[TextContent | ImageContent | EmbeddedResource]:

        if "dirpath" not in args:
            raise RuntimeError("dirpath argument missing in arguments")

        api = obsidian.Obsidian(api_key=api_key, host=obsidian_host)

        files = api.list_files_in_dir(args["dirpath"])

        return [
            TextContent(
                type="text",
                text=json.dumps(files, indent=2)
            )
        ]
    
class GetFileContentsToolHandler(ToolHandler):
    def __init__(self):
        super().__init__("obsidian_get_file_contents")

    def get_tool_description(self):
        return Tool(
            name=self.name,
            description="Return the content of a single file in your vault.",
            inputSchema={
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "Path to the relevant file (relative to your vault root).",
                        "format": "path"
                    },
                },
                "required": ["filepath"]
            }
        )

    def run_tool(self, args: dict) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
        if "filepath" not in args:
            raise RuntimeError("filepath argument missing in arguments")

        api = obsidian.Obsidian(api_key=api_key, host=obsidian_host)

        content = api.get_file_contents(args["filepath"])

        return [
            TextContent(
                type="text",
                text=json.dumps(content, indent=2)
            )
        ]
    
class SearchToolHandler(ToolHandler):
    def __init__(self):
        super().__init__("obsidian_simple_search")

    def get_tool_description(self):
        return Tool(
            name=self.name,
            description="""Simple search for documents matching a specified text query across all files in the vault. 
            Use this tool when you want to do a simple text search""",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Text to a simple search for in the vault."
                    },
                    "context_length": {
                        "type": "integer",
                        "description": "How much context to return around the matching string (default: 100)",
                        "default": 100
                    }
                },
                "required": ["query"]
            }
        )

    def run_tool(self, args: dict) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
        if "query" not in args:
            raise RuntimeError("query argument missing in arguments")

        context_length = args.get("context_length", 100)
        
        api = obsidian.Obsidian(api_key=api_key, host=obsidian_host)
        results = api.search(args["query"], context_length)
        
        formatted_results = []
        for result in results:
            formatted_matches = []
            for match in result.get('matches', []):
                context = match.get('context', '')
                match_pos = match.get('match', {})
                start = match_pos.get('start', 0)
                end = match_pos.get('end', 0)
                
                formatted_matches.append({
                    'context': context,
                    'match_position': {'start': start, 'end': end}
                })
                
            formatted_results.append({
                'filename': result.get('filename', ''),
                'score': result.get('score', 0),
                'matches': formatted_matches
            })

        return [
            TextContent(
                type="text",
                text=json.dumps(formatted_results, indent=2)
            )
        ]
    
class AppendContentToolHandler(ToolHandler):
   def __init__(self):
       super().__init__("obsidian_append_content")

   def get_tool_description(self):
       return Tool(
           name=self.name,
           description="Append content to a new or existing file in the vault.",
           inputSchema={
               "type": "object",
               "properties": {
                   "filepath": {
                       "type": "string",
                       "description": "Path to the file (relative to vault root)",
                       "format": "path"
                   },
                   "content": {
                       "type": "string",
                       "description": "Content to append to the file"
                   }
               },
               "required": ["filepath", "content"]
           }
       )

   def run_tool(self, args: dict) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
       if "filepath" not in args or "content" not in args:
           raise RuntimeError("filepath and content arguments required")

       api = obsidian.Obsidian(api_key=api_key, host=obsidian_host)
       api.append_content(args.get("filepath", ""), args["content"])

       return [
           TextContent(
               type="text",
               text=f"Successfully appended content to {args['filepath']}"
           )
       ]
   
class PatchContentToolHandler(ToolHandler):
   def __init__(self):
       super().__init__("obsidian_patch_content")

   def get_tool_description(self):
       return Tool(
           name=self.name,
           description="Insert content into an existing note relative to a heading, block reference, or frontmatter field.",
           inputSchema={
               "type": "object",
               "properties": {
                   "filepath": {
                       "type": "string",
                       "description": "Path to the file (relative to vault root)",
                       "format": "path"
                   },
                   "operation": {
                       "type": "string",
                       "description": "Operation to perform (append, prepend, or replace)",
                       "enum": ["append", "prepend", "replace"]
                   },
                   "target_type": {
                       "type": "string",
                       "description": "Type of target to patch",
                       "enum": ["heading", "block", "frontmatter"]
                   },
                   "target": {
                       "type": "string", 
                       "description": "Target identifier (heading path, block reference, or frontmatter field)"
                   },
                   "content": {
                       "type": "string",
                       "description": "Content to insert"
                   }
               },
               "required": ["filepath", "operation", "target_type", "target", "content"]
           }
       )

   def run_tool(self, args: dict) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
       if not all(k in args for k in ["filepath", "operation", "target_type", "target", "content"]):
           raise RuntimeError("filepath, operation, target_type, target and content arguments required")

       api = obsidian.Obsidian(api_key=api_key, host=obsidian_host)
       api.patch_content(
           args.get("filepath", ""),
           args.get("operation", ""),
           args.get("target_type", ""),
           args.get("target", ""),
           args.get("content", "")
       )

       return [
           TextContent(
               type="text",
               text=f"Successfully patched content in {args['filepath']}"
           )
       ]
       
class PutContentToolHandler(ToolHandler):
   def __init__(self):
       super().__init__("obsidian_put_content")

   def get_tool_description(self):
       return Tool(
           name=self.name,
           description="Create a new file in your vault or update the content of an existing one in your vault.",
           inputSchema={
               "type": "object",
               "properties": {
                   "filepath": {
                       "type": "string",
                       "description": "Path to the relevant file (relative to your vault root)",
                       "format": "path"
                   },
                   "content": {
                       "type": "string",
                       "description": "Content of the file you would like to upload"
                   }
               },
               "required": ["filepath", "content"]
           }
       )

   def run_tool(self, args: dict) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
       if "filepath" not in args or "content" not in args:
           raise RuntimeError("filepath and content arguments required")

       api = obsidian.Obsidian(api_key=api_key, host=obsidian_host)
       api.put_content(args.get("filepath", ""), args["content"])

       return [
           TextContent(
               type="text",
               text=f"Successfully uploaded content to {args['filepath']}"
           )
       ]
   

class DeleteFileToolHandler(ToolHandler):
   def __init__(self):
       super().__init__("obsidian_delete_file")

   def get_tool_description(self):
       return Tool(
           name=self.name,
           description="Delete a file or directory from the vault.",
           inputSchema={
               "type": "object",
               "properties": {
                   "filepath": {
                       "type": "string",
                       "description": "Path to the file or directory to delete (relative to vault root)",
                       "format": "path"
                   },
                   "confirm": {
                       "type": "boolean",
                       "description": "Confirmation to delete the file (must be true)",
                       "default": False
                   }
               },
               "required": ["filepath", "confirm"]
           }
       )

   def run_tool(self, args: dict) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
       if "filepath" not in args:
           raise RuntimeError("filepath argument missing in arguments")
       
       if not args.get("confirm", False):
           raise RuntimeError("confirm must be set to true to delete a file")

       api = obsidian.Obsidian(api_key=api_key, host=obsidian_host)
       api.delete_file(args["filepath"])

       return [
           TextContent(
               type="text",
               text=f"Successfully deleted {args['filepath']}"
           )
       ]
   
class ComplexSearchToolHandler(ToolHandler):
   def __init__(self):
       super().__init__("obsidian_complex_search")

   def get_tool_description(self):
       return Tool(
           name=self.name,
           description="""Complex search for documents using a JsonLogic query. 
           Supports standard JsonLogic operators plus 'glob' and 'regexp' for pattern matching. Results must be non-falsy.

           Use this tool when you want to do a complex search, e.g. for all documents with certain tags etc.
           ALWAYS follow query syntax in examples.

           Examples
            1. Match all markdown files
            {"glob": ["*.md", {"var": "path"}]}

            2. Match all markdown files with 1221 substring inside them
            {
              "and": [
                { "glob": ["*.md", {"var": "path"}] },
                { "regexp": [".*1221.*", {"var": "content"}] }
              ]
            }

            3. Match all markdown files in Work folder containing name Keaton
            {
              "and": [
                { "glob": ["*.md", {"var": "path"}] },
                { "regexp": [".*Work.*", {"var": "path"}] },
                { "regexp": ["Keaton", {"var": "content"}] }
              ]
            }
           """,
           inputSchema={
               "type": "object",
               "properties": {
                   "query": {
                       "type": "object",
                       "description": "JsonLogic query object. ALWAYS follow query syntax in examples. \
                            Example 1: {\"glob\": [\"*.md\", {\"var\": \"path\"}]} matches all markdown files \
                            Example 2: {\"and\": [{\"glob\": [\"*.md\", {\"var\": \"path\"}]}, {\"regexp\": [\".*1221.*\", {\"var\": \"content\"}]}]} matches all markdown files with 1221 substring inside them \
                            Example 3: {\"and\": [{\"glob\": [\"*.md\", {\"var\": \"path\"}]}, {\"regexp\": [\".*Work.*\", {\"var\": \"path\"}]}, {\"regexp\": [\"Keaton\", {\"var\": \"content\"}]}]} matches all markdown files in Work folder containing name Keaton \
                        "
                   }
               },
               "required": ["query"]
           }
       )

   def run_tool(self, args: dict) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
       query_param = args.get("query")

       if query_param is None:
           raise RuntimeError("query argument missing in arguments")

       actual_query_dict: dict
       if isinstance(query_param, str):
           try:
               actual_query_dict = json.loads(query_param)
           except json.JSONDecodeError as e:
               raise RuntimeError(f"query parameter is a string but not valid JSON: {query_param}")
       elif isinstance(query_param, dict):
           actual_query_dict = query_param
       else:
           server.logger.warning(f"query parameter has unexpected type: {type(query_param)}. Query: {query_param}")
           raise RuntimeError(f"query argument has unexpected type: {type(query_param)}. Expected dict or JSON string.")

       api = obsidian.Obsidian(api_key=api_key, host=obsidian_host)
       results = api.search_json(actual_query_dict)
       
       return [
           TextContent(
               type="text",
               text=json.dumps(results, indent=2)
           )
       ]

class DataviewQueryToolHandler(ToolHandler):
    def __init__(self):
        super().__init__("obsidian_dataview_query") # New function

    def get_tool_description(self):
        return Tool(
            name=self.name,
            description="Executes a Dataview query against Obsidian notes and returns the results as JSON. Supports GROUP BY by post-processing results on the server.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The Dataview query string (e.g., TABLE title, status FROM \"some/path\" WHERE condition). Note: It does not support TABLE WITHOUT ID queries."
                    }
                },
                "required": ["query"]
            }
        )

    def run_tool(self, args: dict) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
        dataview_query_string = args.get("query")

        if not dataview_query_string or not isinstance(dataview_query_string, str):
            raise RuntimeError("query argument (string) missing or invalid in arguments for Dataview")

        base_query, group_fields = self._extract_group_by_clause(dataview_query_string)
        execute_query = base_query if group_fields else dataview_query_string
        if group_fields:
            server.logger.debug(
                "obsidian_dataview_query: GROUP BY rewritten",
                extra={
                    "group_by": group_fields,
                    "rewritten_query": execute_query
                }
            )
        else:
            server.logger.debug(
                "obsidian_dataview_query: no GROUP BY detected",
                extra={
                    "query": dataview_query_string
                }
            )

        api = obsidian.Obsidian(api_key=api_key, host=obsidian_host)
        try:
            results = api.dataview_query_execute(execute_query)
        except Exception as exc:
            if group_fields:
                debug_payload = json.dumps({
                    "groupBy": group_fields,
                    "rewrittenQuery": execute_query,
                    "originalQuery": dataview_query_string,
                    "error": str(exc)
                }, ensure_ascii=False)
                raise Exception(
                    f"Dataview query failed after GROUP BY rewrite: {debug_payload}"
                ) from exc
            raise

        if group_fields and isinstance(results, list):
            column_alias_map = self._parse_table_columns(dataview_query_string)
            grouped_payload = self._group_results(results, group_fields, column_alias_map)
            payload = {
                "grouped": True,
                "groupBy": group_fields,
                "groups": grouped_payload
            }
        else:
            payload = results

        return [
            TextContent(
                type="text",
                text=json.dumps(payload, indent=2)
            )
        ]

    def _extract_group_by_clause(self, query: str) -> tuple[str, list[str]]:
        lowered = query.lower()
        marker = 'group by'
        idx = lowered.find(marker)
        if idx == -1:
            return query, []

        group_part = query[idx + len(marker):]
        base_query = query[:idx].rstrip()
        expressions = self._split_expressions(group_part)
        cleaned = [expr for expr in (expr.strip() for expr in expressions) if expr]
        return base_query, cleaned

    def _split_expressions(self, text: str) -> list[str]:
        parts = []
        current = []
        depth = 0
        in_quote = False
        quote_char = ""

        for idx, ch in enumerate(text):
            if in_quote:
                current.append(ch)
                if ch == quote_char and (idx == 0 or text[idx - 1] != "\\"):
                    in_quote = False
                continue

            if ch in ('"', "'"):
                in_quote = True
                quote_char = ch
                current.append(ch)
                continue

            if ch in "([{":
                depth += 1
                current.append(ch)
                continue

            if ch in ")]}":
                depth = max(0, depth - 1)
                current.append(ch)
                continue

            if ch == ',' and depth == 0:
                parts.append(''.join(current).strip())
                current = []
                continue

            if ch in '\n;':
                current.append(ch)
                continue

            current.append(ch)

        if current:
            parts.append(''.join(current).strip())

        return parts

    def _parse_table_columns(self, query: str) -> dict[str, str]:
        match = re.search(r"(?is)\bTABLE\b(.*?)\bFROM\b", query)
        if not match:
            return {}

        columns_segment = match.group(1)
        columns = self._split_expressions(columns_segment)
        alias_map: dict[str, str] = {}

        for column in columns:
            if not column:
                continue
            parts = re.split(r"(?is)\s+AS\s+", column, maxsplit=1)
            expr = parts[0].strip()
            alias = parts[1].strip() if len(parts) > 1 else expr
            normal_expr = self._normalize_reference(expr)
            normal_alias = self._normalize_reference(alias)
            alias_map[normal_expr] = alias.strip('"')
            alias_map[normal_alias] = alias.strip('"')

        return alias_map

    def _normalize_reference(self, value: str) -> str:
        trimmed = value.strip()
        if trimmed.startswith('"') and trimmed.endswith('"'):
            trimmed = trimmed[1:-1]
        if trimmed.startswith("'") and trimmed.endswith("'"):
            trimmed = trimmed[1:-1]
        trimmed = re.sub(r"\s+", " ", trimmed)
        return trimmed.lower()

    def _resolve_alias(self, expression: str, alias_map: dict[str, str]) -> str:
        normalized = self._normalize_reference(expression)
        return alias_map.get(normalized, expression.strip())

    def _resolve_value(self, row_result: dict, alias: str, expression: str):
        candidates = [alias, expression.strip()]
        for candidate in candidates:
            if candidate in row_result:
                return row_result[candidate]
        return None

    def _make_hashable(self, value):
        if isinstance(value, list):
            return tuple(self._make_hashable(v) for v in value)
        if isinstance(value, dict):
            return tuple(sorted((k, self._make_hashable(v)) for k, v in value.items()))
        return value

    def _group_results(self, rows: list[dict], group_fields: list[str], alias_map: dict[str, str]) -> list[dict]:
        grouped: dict[tuple, dict] = {}

        for row in rows:
            row_result = row.get("result", {})
            group_labels = {}
            key_components = []

            for expression in group_fields:
                alias = self._resolve_alias(expression, alias_map)
                value = self._resolve_value(row_result, alias, expression)
                group_labels[alias] = value
                key_components.append(self._make_hashable(value))

            key = tuple(key_components)

            if key not in grouped:
                grouped[key] = {
                    "group": group_labels,
                    "rows": []
                }

            grouped[key]["rows"].append(row)

        for entry in grouped.values():
            entry["count"] = len(entry["rows"])

        return list(grouped.values())


class DataviewJsToolHandler(ToolHandler):
    def __init__(self):
        super().__init__("obsidian_dataviewjs_execute")

    def get_tool_description(self):
        return Tool(
            name=self.name,
            description=(
                "Executes DataviewJS code and returns rendered HTML/text output from Obsidian. "
                "Provide the exact DataviewJS snippet you would run inside the app."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Raw DataviewJS code to execute (e.g., \"await dv.view('path')\")."
                    },
                    "file_path": {
                        "type": "string",
                        "description": "Optional note path to use as the Dataview execution context.",
                        "format": "path"
                    },
                    "timeout_ms": {
                        "type": "integer",
                        "description": "Optional timeout in milliseconds before the execution is aborted (set 0 to disable).",
                        "minimum": 0
                    }
                },
                "required": ["code"]
            }
        )

    def run_tool(self, args: dict) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
        code = args.get("code")

        if not code or not isinstance(code, str):
            raise RuntimeError("'code' argument (string) is required for DataviewJS execution")

        api = obsidian.Obsidian(api_key=api_key, host=obsidian_host)
        result = api.dataview_js(
            code=code,
            file_path=args.get("file_path"),
            timeout_ms=args.get("timeout_ms")
        )

        return [
            TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )
        ]

class BatchGetFileContentsToolHandler(ToolHandler):
    def __init__(self):
        super().__init__("obsidian_batch_get_file_contents")

    def get_tool_description(self):
        return Tool(
            name=self.name,
            description="Return the contents of multiple files in your vault, concatenated with headers.",
            inputSchema={
                "type": "object",
                "properties": {
                    "filepaths": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "description": "Path to a file (relative to your vault root)",
                            "format": "path"
                        },
                        "description": "List of file paths to read"
                    },
                },
                "required": ["filepaths"]
            }
        )

    def run_tool(self, args: dict) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
        if "filepaths" not in args:
            raise RuntimeError("filepaths argument missing in arguments")

        api = obsidian.Obsidian(api_key=api_key, host=obsidian_host)
        content = api.get_batch_file_contents(args["filepaths"])

        return [
            TextContent(
                type="text",
                text=content
            )
        ]

class PeriodicNotesToolHandler(ToolHandler):
    def __init__(self):
        super().__init__("obsidian_get_periodic_note")

    def get_tool_description(self):
        return Tool(
            name=self.name,
            description="Get current periodic note for the specified period.",
            inputSchema={
                "type": "object",
                "properties": {
                    "period": {
                        "type": "string",
                        "description": "The period type (daily, weekly, monthly, quarterly, yearly)",
                        "enum": ["daily", "weekly", "monthly", "quarterly", "yearly"]
                    },
                    "type": {
                        "type": "string",
                        "description": "The type of data to get ('content' or 'metadata'). 'content' returns just the content in Markdown format. 'metadata' includes note metadata (including paths, tags, etc.) and the content.",
                        "default": "content",
                        "enum": ["content", "metadata"]
                    }
                },
                "required": ["period"]
            }
        )

    def run_tool(self, args: dict) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
        if "period" not in args:
            raise RuntimeError("period argument missing in arguments")

        period = args["period"]
        valid_periods = ["daily", "weekly", "monthly", "quarterly", "yearly"]
        if period not in valid_periods:
            raise RuntimeError(f"Invalid period: {period}. Must be one of: {', '.join(valid_periods)}")
        
        type = args["type"] if "type" in args else "content"
        valid_types = ["content", "metadata"]
        if type not in valid_types:
            raise RuntimeError(f"Invalid type: {type}. Must be one of: {', '.join(valid_types)}")

        api = obsidian.Obsidian(api_key=api_key, host=obsidian_host)
        content = api.get_periodic_note(period,type)

        return [
            TextContent(
                type="text",
                text=content
            )
        ]
        
class RecentPeriodicNotesToolHandler(ToolHandler):
    def __init__(self):
        super().__init__("obsidian_get_recent_periodic_notes")

    def get_tool_description(self):
        return Tool(
            name=self.name,
            description="Get most recent periodic notes for the specified period type.",
            inputSchema={
                "type": "object",
                "properties": {
                    "period": {
                        "type": "string",
                        "description": "The period type (daily, weekly, monthly, quarterly, yearly)",
                        "enum": ["daily", "weekly", "monthly", "quarterly", "yearly"]
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of notes to return (default: 5)",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 50
                    },
                    "include_content": {
                        "type": "boolean",
                        "description": "Whether to include note content (default: false)",
                        "default": False
                    }
                },
                "required": ["period"]
            }
        )

    def run_tool(self, args: dict) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
        if "period" not in args:
            raise RuntimeError("period argument missing in arguments")

        period = args["period"]
        valid_periods = ["daily", "weekly", "monthly", "quarterly", "yearly"]
        if period not in valid_periods:
            raise RuntimeError(f"Invalid period: {period}. Must be one of: {', '.join(valid_periods)}")

        limit = args.get("limit", 5)
        if not isinstance(limit, int) or limit < 1:
            raise RuntimeError(f"Invalid limit: {limit}. Must be a positive integer")
            
        include_content = args.get("include_content", False)
        if not isinstance(include_content, bool):
            raise RuntimeError(f"Invalid include_content: {include_content}. Must be a boolean")

        api = obsidian.Obsidian(api_key=api_key, host=obsidian_host)
        results = api.get_recent_periodic_notes(period, limit, include_content)

        return [
            TextContent(
                type="text",
                text=json.dumps(results, indent=2)
            )
        ]
        
class RecentChangesToolHandler(ToolHandler):
    def __init__(self):
        super().__init__("obsidian_get_recent_changes")

    def get_tool_description(self):
        return Tool(
            name=self.name,
            description="Get recently modified files in the vault.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of files to return (default: 10)",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 100
                    },
                    "days": {
                        "type": "integer",
                        "description": "Only include files modified within this many days (default: 90)",
                        "minimum": 1,
                        "default": 90
                    }
                }
            }
        )

    def run_tool(self, args: dict) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
        limit = args.get("limit", 10)
        if not isinstance(limit, int) or limit < 1:
            raise RuntimeError(f"Invalid limit: {limit}. Must be a positive integer")
            
        days = args.get("days", 90)
        if not isinstance(days, int) or days < 1:
            raise RuntimeError(f"Invalid days: {days}. Must be a positive integer")

        api = obsidian.Obsidian(api_key=api_key, host=obsidian_host)
        results = api.get_recent_changes(limit, days)

        return [
            TextContent(
                type="text",
                text=json.dumps(results, indent=2)
            )
        ]

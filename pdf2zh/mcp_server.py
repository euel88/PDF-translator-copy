"""
MCP 서버 모듈 - PDFMathTranslate 구조 기반
Model Context Protocol 서버 (Claude Desktop 통합)
"""
import os
import sys
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from pdf2zh.high_level import translate
from pdf2zh.config import config, LANGUAGES


@dataclass
class MCPTool:
    """MCP 도구 정의"""
    name: str
    description: str
    input_schema: Dict[str, Any]


# MCP 도구 정의
TOOLS = [
    MCPTool(
        name="translate_pdf",
        description="Translate a PDF document from one language to another. Supports multiple translation services including OpenAI, Google, DeepL, and Ollama.",
        input_schema={
            "type": "object",
            "properties": {
                "input_path": {
                    "type": "string",
                    "description": "Path to the input PDF file"
                },
                "output_path": {
                    "type": "string",
                    "description": "Path for the translated PDF output (optional)"
                },
                "source_lang": {
                    "type": "string",
                    "description": "Source language (e.g., 'English', 'Korean', 'Japanese')",
                    "default": "English"
                },
                "target_lang": {
                    "type": "string",
                    "description": "Target language (e.g., 'Korean', 'English', 'Japanese')",
                    "default": "Korean"
                },
                "service": {
                    "type": "string",
                    "description": "Translation service to use",
                    "enum": ["openai", "google", "deepl", "ollama"],
                    "default": "openai"
                },
                "pages": {
                    "type": "string",
                    "description": "Page range to translate (e.g., '1-5' or '1,3,5')",
                    "default": ""
                }
            },
            "required": ["input_path"]
        }
    ),
    MCPTool(
        name="list_languages",
        description="List all supported languages for translation",
        input_schema={
            "type": "object",
            "properties": {}
        }
    ),
    MCPTool(
        name="get_translation_status",
        description="Get the status of the translation service",
        input_schema={
            "type": "object",
            "properties": {}
        }
    )
]


class MCPServer:
    """MCP 서버"""

    def __init__(self):
        self.tools = {t.name: t for t in TOOLS}

    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """요청 처리"""
        method = request.get("method", "")
        params = request.get("params", {})
        request_id = request.get("id")

        try:
            if method == "initialize":
                return self._handle_initialize(request_id)
            elif method == "tools/list":
                return self._handle_list_tools(request_id)
            elif method == "tools/call":
                return self._handle_call_tool(request_id, params)
            else:
                return self._error_response(request_id, -32601, f"Method not found: {method}")

        except Exception as e:
            return self._error_response(request_id, -32603, str(e))

    def _handle_initialize(self, request_id: Any) -> Dict[str, Any]:
        """초기화 처리"""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "pdf2zh",
                    "version": "1.0.0"
                }
            }
        }

    def _handle_list_tools(self, request_id: Any) -> Dict[str, Any]:
        """도구 목록 반환"""
        tools_list = []
        for tool in TOOLS:
            tools_list.append({
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.input_schema
            })

        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "tools": tools_list
            }
        }

    def _handle_call_tool(self, request_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        """도구 호출 처리"""
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        if tool_name == "translate_pdf":
            result = self._translate_pdf(arguments)
        elif tool_name == "list_languages":
            result = self._list_languages()
        elif tool_name == "get_translation_status":
            result = self._get_status()
        else:
            return self._error_response(request_id, -32602, f"Unknown tool: {tool_name}")

        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(result, ensure_ascii=False, indent=2)
                    }
                ]
            }
        }

    def _translate_pdf(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """PDF 번역"""
        input_path = args.get("input_path", "")

        if not input_path:
            return {"error": "input_path is required"}

        if not os.path.exists(input_path):
            return {"error": f"File not found: {input_path}"}

        output_path = args.get("output_path")
        source_lang = args.get("source_lang", "English")
        target_lang = args.get("target_lang", "Korean")
        service = args.get("service", "openai")
        pages_str = args.get("pages", "")

        # 페이지 파싱
        pages = None
        if pages_str:
            pages = self._parse_pages(pages_str)

        logs = []

        def callback(msg: str):
            logs.append(msg)

        try:
            result = translate(
                input_path=input_path,
                output_path=output_path,
                source_lang=source_lang,
                target_lang=target_lang,
                service=service,
                pages=pages,
                callback=callback,
            )

            return {
                "success": result.success,
                "output_path": result.output_path,
                "page_count": result.page_count,
                "error": result.error,
                "logs": logs[-10:]  # 마지막 10개 로그
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "logs": logs
            }

    def _list_languages(self) -> Dict[str, Any]:
        """지원 언어 목록"""
        return {
            "languages": list(LANGUAGES.keys()),
            "language_codes": LANGUAGES
        }

    def _get_status(self) -> Dict[str, Any]:
        """서비스 상태"""
        # API 키 확인
        openai_configured = bool(config.get("OPENAI_API_KEY"))
        deepl_configured = bool(config.get("DEEPL_API_KEY"))

        return {
            "status": "ready",
            "services": {
                "openai": "configured" if openai_configured else "not configured",
                "google": "available (free)",
                "deepl": "configured" if deepl_configured else "not configured",
                "ollama": "available (local)"
            }
        }

    def _parse_pages(self, pages_str: str) -> List[int]:
        """페이지 문자열 파싱"""
        pages = []

        for part in pages_str.split(","):
            part = part.strip()
            if "-" in part:
                start, end = part.split("-")
                pages.extend(range(int(start) - 1, int(end)))
            else:
                pages.append(int(part) - 1)

        return pages

    def _error_response(self, request_id: Any, code: int, message: str) -> Dict[str, Any]:
        """에러 응답"""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": code,
                "message": message
            }
        }


def run_stdio_server():
    """STDIO 모드 서버 실행"""
    server = MCPServer()

    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break

            request = json.loads(line.strip())
            response = server.handle_request(request)
            print(json.dumps(response), flush=True)

        except json.JSONDecodeError:
            continue
        except KeyboardInterrupt:
            break
        except Exception as e:
            error_response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32603,
                    "message": str(e)
                }
            }
            print(json.dumps(error_response), flush=True)


def run_sse_server(host: str = "0.0.0.0", port: int = 8080):
    """SSE 모드 서버 실행"""
    try:
        from flask import Flask, Response, request as flask_request
    except ImportError:
        print("Flask가 필요합니다. pip install flask")
        return

    app = Flask(__name__)
    server = MCPServer()

    @app.route("/sse", methods=["GET"])
    def sse_endpoint():
        """SSE 엔드포인트"""
        def generate():
            # 초기 연결 메시지
            yield f"data: {json.dumps({'type': 'connected'})}\n\n"

            # 이벤트 대기 (간단한 구현)
            import time
            while True:
                time.sleep(30)
                yield f"data: {json.dumps({'type': 'ping'})}\n\n"

        return Response(
            generate(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            }
        )

    @app.route("/message", methods=["POST"])
    def message_endpoint():
        """메시지 엔드포인트"""
        request_data = flask_request.get_json()
        response = server.handle_request(request_data)
        return response

    print(f"MCP SSE 서버 시작: http://{host}:{port}")
    app.run(host=host, port=port, threaded=True)


def get_claude_desktop_config() -> Dict[str, Any]:
    """Claude Desktop 설정 예시 반환"""
    script_path = os.path.abspath(__file__)

    return {
        "mcpServers": {
            "pdf2zh": {
                "command": sys.executable,
                "args": [script_path, "--stdio"],
                "env": {
                    "OPENAI_API_KEY": "your-api-key-here"
                }
            }
        }
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PDF2ZH MCP Server")
    parser.add_argument("--stdio", action="store_true", help="Run in STDIO mode")
    parser.add_argument("--sse", action="store_true", help="Run in SSE mode")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--config", action="store_true", help="Print Claude Desktop config")

    args = parser.parse_args()

    if args.config:
        print(json.dumps(get_claude_desktop_config(), indent=2))
    elif args.stdio:
        run_stdio_server()
    elif args.sse:
        run_sse_server(args.host, args.port)
    else:
        print("Usage: python mcp_server.py [--stdio | --sse | --config]")
        print("  --stdio  : Run in STDIO mode (for Claude Desktop)")
        print("  --sse    : Run in SSE mode (HTTP server)")
        print("  --config : Print Claude Desktop configuration")

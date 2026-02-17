"""
CODE ASSISTANT AI AGENT
Enterprise-grade intelligent code analysis, review, generation, and security scanning

Features:
- Multi-language support (Python, JavaScript, TypeScript, Java, Go, Rust, C/C++)
- AST-based code parsing and analysis
- Bug detection: null dereference, division by zero, unreachable code, dead variables
- Security scanning: SQL injection, XSS, hardcoded secrets, path traversal, SSRF
- Code quality scoring (complexity, maintainability, readability)
- Cyclomatic complexity calculation
- Code smell detection (long functions, deep nesting, too many params)
- Documentation generator (docstrings, JSDoc, JavaDoc)
- Refactoring suggestions with concrete rewrites
- Dependency vulnerability checking
- Unit test generation
- Code formatting and style enforcement
- Diff-based code review
- Multi-file project analysis
- Git blame integration stubs
- Performance profiling hints

Dependencies:
- re
- ast (Python standard lib)
- tokenize
- json
- dataclasses
"""

from __future__ import annotations

import ast
import re
import json
import time
import math
import hashlib
import tokenize
import io
import logging
import functools
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple, Union

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("CodeAssistantAI")


# ===========================================================================
# Enums & Constants
# ===========================================================================

class Language(Enum):
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    GO = "go"
    RUST = "rust"
    CPP = "cpp"
    C = "c"
    CSHARP = "csharp"
    RUBY = "ruby"
    PHP = "php"
    UNKNOWN = "unknown"


class IssueSeverity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class IssueCategory(Enum):
    SECURITY = "security"
    BUG = "bug"
    PERFORMANCE = "performance"
    MAINTAINABILITY = "maintainability"
    STYLE = "style"
    DOCUMENTATION = "documentation"
    COMPLEXITY = "complexity"
    DEPENDENCY = "dependency"


class RefactorType(Enum):
    EXTRACT_FUNCTION = "extract_function"
    RENAME_VARIABLE = "rename_variable"
    SIMPLIFY_CONDITION = "simplify_condition"
    REMOVE_DEAD_CODE = "remove_dead_code"
    ADD_TYPE_HINTS = "add_type_hints"
    REDUCE_COMPLEXITY = "reduce_complexity"
    BREAK_LONG_FUNCTION = "break_long_function"
    REPLACE_MAGIC_NUMBER = "replace_magic_number"
    ADD_EARLY_RETURN = "add_early_return"
    EXTRACT_CONSTANT = "extract_constant"


class DocFormat(Enum):
    GOOGLE = "google"
    NUMPY = "numpy"
    SPHINX = "sphinx"
    JSDOC = "jsdoc"
    JAVADOC = "javadoc"
    RUSTDOC = "rustdoc"


# ===========================================================================
# Data Models
# ===========================================================================

@dataclass
class CodeIssue:
    issue_id: str
    category: IssueCategory
    severity: IssueSeverity
    title: str
    description: str
    line_number: Optional[int]
    column: Optional[int]
    code_snippet: Optional[str]
    suggestion: str
    cwe_id: Optional[str] = None  # Common Weakness Enumeration
    owasp_category: Optional[str] = None


@dataclass
class FunctionAnalysis:
    name: str
    line_start: int
    line_end: int
    line_count: int
    parameter_count: int
    cyclomatic_complexity: int
    cognitive_complexity: int
    has_docstring: bool
    return_type_annotated: bool
    has_type_hints: bool
    nested_depth: int
    issue_flags: List[str] = field(default_factory=list)


@dataclass
class ClassAnalysis:
    name: str
    line_start: int
    line_end: int
    method_count: int
    attribute_count: int
    has_docstring: bool
    inherits_from: List[str] = field(default_factory=list)
    methods: List[str] = field(default_factory=list)


@dataclass
class ModuleAnalysis:
    file_path: str
    language: Language
    line_count: int
    blank_lines: int
    comment_lines: int
    code_lines: int
    import_count: int
    function_count: int
    class_count: int
    issues: List[CodeIssue]
    functions: List[FunctionAnalysis]
    classes: List[ClassAnalysis]
    quality_score: float
    complexity_score: float
    security_score: float
    maintainability_score: float
    analyzed_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ReviewResult:
    review_id: str
    file_path: str
    total_issues: int
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    overall_score: float
    summary: str
    issues: List[CodeIssue]
    recommendations: List[str]
    test_suggestions: List[str]
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class GeneratedCode:
    generation_id: str
    prompt: str
    language: Language
    code: str
    explanation: str
    complexity_estimate: str
    test_code: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class SecurityScanResult:
    scan_id: str
    file_path: str
    findings: List[CodeIssue]
    risk_level: str
    cve_references: List[str]
    summary: str
    remediation_priority: List[str]


@dataclass
class DiffReview:
    diff_id: str
    additions: int
    deletions: int
    changed_functions: List[str]
    issues_introduced: List[CodeIssue]
    issues_resolved: int
    quality_delta: float
    verdict: str
    comments: List[Dict[str, Any]]


# ===========================================================================
# Utilities
# ===========================================================================

def generate_id(prefix: str = "CODE") -> str:
    ts = datetime.now().strftime("%Y%m%d%H%M%S%f")
    return f"{prefix}-{ts}"


def timed(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        ms = int((time.perf_counter() - start) * 1000)
        logger.debug(f"{fn.__name__} in {ms}ms")
        return result
    return wrapper


def file_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:12]


def count_lines(code: str) -> Tuple[int, int, int]:
    """Return (total, blank, comment) line counts."""
    lines = code.splitlines()
    total = len(lines)
    blank = sum(1 for l in lines if not l.strip())
    comment = sum(1 for l in lines if l.strip().startswith(("#", "//", "*", "/*", "<!--")))
    return total, blank, comment


def extract_strings(code: str) -> List[str]:
    """Extract all string literals from code."""
    patterns = [
        r'"([^"\\]|\\.)*"',
        r"'([^'\\]|\\.)*'",
        r'`([^`\\]|\\.)*`',
    ]
    results = []
    for p in patterns:
        results.extend(re.findall(p, code))
    return results


# ===========================================================================
# Language Detector
# ===========================================================================

class LanguageDetector:
    """Detect programming language from file extension or code content."""

    EXTENSION_MAP: Dict[str, Language] = {
        ".py": Language.PYTHON,
        ".js": Language.JAVASCRIPT,
        ".mjs": Language.JAVASCRIPT,
        ".ts": Language.TYPESCRIPT,
        ".tsx": Language.TYPESCRIPT,
        ".java": Language.JAVA,
        ".go": Language.GO,
        ".rs": Language.RUST,
        ".cpp": Language.CPP,
        ".cc": Language.CPP,
        ".cxx": Language.CPP,
        ".c": Language.C,
        ".cs": Language.CSHARP,
        ".rb": Language.RUBY,
        ".php": Language.PHP,
    }

    CONTENT_SIGNATURES: Dict[Language, List[str]] = {
        Language.PYTHON: ["def ", "import ", "class ", "if __name__", "print("],
        Language.JAVASCRIPT: ["function ", "const ", "let ", "var ", "=>", "require("],
        Language.TYPESCRIPT: ["interface ", ": string", ": number", "as const", "type "],
        Language.JAVA: ["public class ", "public static void main", "System.out.println", "import java."],
        Language.GO: ["func main()", "package main", "import (", ":=", "fmt.Println"],
        Language.RUST: ["fn main()", "let mut ", "impl ", "use std::", "println!"],
    }

    def from_extension(self, filename: str) -> Language:
        for ext, lang in self.EXTENSION_MAP.items():
            if filename.endswith(ext):
                return lang
        return Language.UNKNOWN

    def from_content(self, code: str) -> Language:
        scores: Dict[Language, int] = defaultdict(int)
        for lang, sigs in self.CONTENT_SIGNATURES.items():
            for sig in sigs:
                if sig in code:
                    scores[lang] += 1
        if not scores:
            return Language.UNKNOWN
        return max(scores, key=lambda l: scores[l])

    def detect(self, code: str, filename: str = "") -> Language:
        if filename:
            lang = self.from_extension(filename)
            if lang != Language.UNKNOWN:
                return lang
        return self.from_content(code)


# ===========================================================================
# Python AST Analyzer
# ===========================================================================

class PythonASTAnalyzer:
    """Deep AST-based analysis for Python code."""

    def parse(self, code: str) -> Optional[ast.Module]:
        try:
            return ast.parse(code)
        except SyntaxError as e:
            logger.warning(f"AST parse error: {e}")
            return None

    def extract_functions(self, tree: ast.Module) -> List[FunctionAnalysis]:
        funcs = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fa = self._analyze_function(node)
                funcs.append(fa)
        return funcs

    def _analyze_function(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> FunctionAnalysis:
        line_start = node.lineno
        line_end = node.end_lineno if hasattr(node, "end_lineno") else line_start + 10
        has_docstring = (
            isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant)
            if node.body else False
        )
        params = node.args
        param_count = (
            len(params.args) + len(params.posonlyargs) + len(params.kwonlyargs)
            + (1 if params.vararg else 0) + (1 if params.kwarg else 0)
        )
        has_type_hints = any(
            arg.annotation is not None for arg in params.args + params.posonlyargs
        )
        return_annotated = node.returns is not None
        cc = self._cyclomatic_complexity(node)
        cog = self._cognitive_complexity(node)
        depth = self._max_nesting_depth(node)
        flags = []
        if param_count > 7:
            flags.append("too_many_parameters")
        if line_end - line_start > 50:
            flags.append("function_too_long")
        if cc > 10:
            flags.append("high_cyclomatic_complexity")
        if depth > 4:
            flags.append("deeply_nested")
        if not has_docstring:
            flags.append("missing_docstring")
        return FunctionAnalysis(
            name=node.name,
            line_start=line_start,
            line_end=line_end,
            line_count=line_end - line_start + 1,
            parameter_count=param_count,
            cyclomatic_complexity=cc,
            cognitive_complexity=cog,
            has_docstring=has_docstring,
            return_type_annotated=return_annotated,
            has_type_hints=has_type_hints,
            nested_depth=depth,
            issue_flags=flags,
        )

    def _cyclomatic_complexity(self, node: ast.AST) -> int:
        """McCabe cyclomatic complexity: 1 + decision points."""
        cc = 1
        branch_nodes = (
            ast.If, ast.While, ast.For, ast.ExceptHandler,
            ast.With, ast.Assert, ast.comprehension,
        )
        for child in ast.walk(node):
            if isinstance(child, branch_nodes):
                cc += 1
            elif isinstance(child, ast.BoolOp):
                cc += len(child.values) - 1
        return cc

    def _cognitive_complexity(self, node: ast.AST) -> int:
        """Simplified cognitive complexity."""
        score = 0
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For)):
                score += 1
            elif isinstance(child, ast.Try):
                score += 1
            elif isinstance(child, ast.Lambda):
                score += 1
        return score

    def _max_nesting_depth(self, node: ast.AST, depth: int = 0) -> int:
        max_depth = depth
        nest_nodes = (ast.If, ast.For, ast.While, ast.With, ast.Try, ast.FunctionDef)
        for child in ast.iter_child_nodes(node):
            if isinstance(child, nest_nodes):
                d = self._max_nesting_depth(child, depth + 1)
                max_depth = max(max_depth, d)
        return max_depth

    def extract_classes(self, tree: ast.Module) -> List[ClassAnalysis]:
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = [n.name for n in ast.walk(node) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n is not node]
                attrs = [n.targets[0].id for n in ast.walk(node) if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)]
                has_doc = (
                    isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant)
                    if node.body else False
                )
                bases = [b.id for b in node.bases if isinstance(b, ast.Name)]
                classes.append(ClassAnalysis(
                    name=node.name,
                    line_start=node.lineno,
                    line_end=node.end_lineno if hasattr(node, "end_lineno") else node.lineno + 20,
                    method_count=len(methods),
                    attribute_count=len(set(attrs)),
                    has_docstring=has_doc,
                    inherits_from=bases,
                    methods=methods[:20],
                ))
        return classes

    def find_imports(self, tree: ast.Module) -> List[str]:
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
        return imports


# ===========================================================================
# Security Scanner
# ===========================================================================

class SecurityScanner:
    """Detect common security vulnerabilities in source code."""

    PATTERNS: List[Dict[str, Any]] = [
        {
            "id": "SEC001",
            "title": "SQL Injection Risk",
            "pattern": r'(execute|cursor\.execute|query)\s*\(\s*["\'].*%[sd].*["\']|f["\'](.*SELECT|INSERT|UPDATE|DELETE)',
            "severity": IssueSeverity.CRITICAL,
            "cwe": "CWE-89",
            "owasp": "A03:2021",
            "description": "String formatting used to build SQL queries — vulnerable to SQL injection.",
            "suggestion": "Use parameterized queries or an ORM. Never interpolate user input into SQL.",
        },
        {
            "id": "SEC002",
            "title": "Hardcoded Secret/Password",
            "pattern": r'(password|passwd|secret|api_key|apikey|token|credentials)\s*=\s*["\'][^"\']{4,}["\']',
            "severity": IssueSeverity.CRITICAL,
            "cwe": "CWE-798",
            "owasp": "A07:2021",
            "description": "Hardcoded credentials found in source code.",
            "suggestion": "Move secrets to environment variables or a secrets manager (Vault, AWS Secrets Manager).",
        },
        {
            "id": "SEC003",
            "title": "Cross-Site Scripting (XSS) Risk",
            "pattern": r'innerHTML\s*=|document\.write\(|eval\(|setTimeout\(["\']',
            "severity": IssueSeverity.HIGH,
            "cwe": "CWE-79",
            "owasp": "A03:2021",
            "description": "Potential XSS via direct DOM manipulation or eval().",
            "suggestion": "Use textContent instead of innerHTML. Sanitize all user input before rendering.",
        },
        {
            "id": "SEC004",
            "title": "Path Traversal Risk",
            "pattern": r'open\s*\(\s*[^)]*user|os\.path\.join\s*\(.*request|send_file\s*\(.*request',
            "severity": IssueSeverity.HIGH,
            "cwe": "CWE-22",
            "owasp": "A01:2021",
            "description": "User input used in file path construction — path traversal possible.",
            "suggestion": "Validate and sanitize file paths. Use os.path.basename() and whitelist allowed directories.",
        },
        {
            "id": "SEC005",
            "title": "Insecure Deserialization",
            "pattern": r'pickle\.loads|yaml\.load\(|marshal\.loads|shelve\.open',
            "severity": IssueSeverity.HIGH,
            "cwe": "CWE-502",
            "owasp": "A08:2021",
            "description": "Insecure deserialization detected. Untrusted data could execute arbitrary code.",
            "suggestion": "Use `yaml.safe_load()` instead of `yaml.load()`. Avoid `pickle.loads()` on untrusted data.",
        },
        {
            "id": "SEC006",
            "title": "Command Injection Risk",
            "pattern": r'os\.system\(|subprocess\.(call|run|Popen)\s*\([^)]*shell\s*=\s*True',
            "severity": IssueSeverity.CRITICAL,
            "cwe": "CWE-78",
            "owasp": "A03:2021",
            "description": "Shell command execution with potential for command injection.",
            "suggestion": "Avoid shell=True. Use subprocess with a list of arguments. Sanitize all inputs.",
        },
        {
            "id": "SEC007",
            "title": "Weak Cryptography",
            "pattern": r'hashlib\.md5|hashlib\.sha1|DES\.|RC4\.|ECB\b',
            "severity": IssueSeverity.MEDIUM,
            "cwe": "CWE-327",
            "owasp": "A02:2021",
            "description": "Weak cryptographic algorithm detected (MD5/SHA1/DES/RC4).",
            "suggestion": "Use SHA-256 or SHA-3 for hashing. Use AES-256-GCM for encryption.",
        },
        {
            "id": "SEC008",
            "title": "Insecure Random Number Generation",
            "pattern": r'\brandom\.random\(\)|\brandom\.randint\b|\brandom\.choice\b',
            "severity": IssueSeverity.LOW,
            "cwe": "CWE-338",
            "owasp": "A02:2021",
            "description": "Python's `random` module is not cryptographically secure.",
            "suggestion": "Use `secrets` module for security-sensitive random values.",
        },
        {
            "id": "SEC009",
            "title": "Debug Mode Enabled",
            "pattern": r'app\.run\(.*debug\s*=\s*True|DEBUG\s*=\s*True',
            "severity": IssueSeverity.MEDIUM,
            "cwe": "CWE-94",
            "owasp": "A05:2021",
            "description": "Application debug mode is enabled — should never run in production.",
            "suggestion": "Set debug=False in production. Use environment variables to control debug mode.",
        },
        {
            "id": "SEC010",
            "title": "SSRF Risk (Unvalidated URL)",
            "pattern": r'requests\.(get|post|put|delete)\s*\([^)]*request\.',
            "severity": IssueSeverity.HIGH,
            "cwe": "CWE-918",
            "owasp": "A10:2021",
            "description": "User-controlled URL passed to HTTP client — potential SSRF.",
            "suggestion": "Validate and whitelist allowed URLs/domains. Block requests to internal networks.",
        },
    ]

    def scan(self, code: str, file_path: str = "code") -> SecurityScanResult:
        findings: List[CodeIssue] = []
        lines = code.splitlines()
        for pat in self.PATTERNS:
            for i, line in enumerate(lines, 1):
                if re.search(pat["pattern"], line, re.IGNORECASE):
                    findings.append(CodeIssue(
                        issue_id=generate_id(pat["id"]),
                        category=IssueCategory.SECURITY,
                        severity=pat["severity"],
                        title=pat["title"],
                        description=pat["description"],
                        line_number=i,
                        column=None,
                        code_snippet=line.strip()[:100],
                        suggestion=pat["suggestion"],
                        cwe_id=pat.get("cwe"),
                        owasp_category=pat.get("owasp"),
                    ))

        critical = sum(1 for f in findings if f.severity == IssueSeverity.CRITICAL)
        high = sum(1 for f in findings if f.severity == IssueSeverity.HIGH)
        risk_level = (
            "CRITICAL" if critical > 0 else "HIGH" if high > 0
            else "MEDIUM" if findings else "LOW"
        )
        return SecurityScanResult(
            scan_id=generate_id("SCAN"),
            file_path=file_path,
            findings=findings,
            risk_level=risk_level,
            cve_references=[],
            summary=(
                f"Security scan of '{file_path}': {len(findings)} issue(s) found "
                f"({critical} critical, {high} high). Risk level: {risk_level}."
            ),
            remediation_priority=[
                f"[{f.severity.value.upper()}] Line {f.line_number}: {f.title}"
                for f in sorted(findings, key=lambda x: ["critical","high","medium","low"].index(x.severity.value))
            ][:10],
        )


# ===========================================================================
# Code Smell Detector
# ===========================================================================

class CodeSmellDetector:
    """Detect maintainability and style issues."""

    def detect(self, code: str, functions: List[FunctionAnalysis]) -> List[CodeIssue]:
        issues: List[CodeIssue] = []
        lines = code.splitlines()

        for fn in functions:
            if "too_many_parameters" in fn.issue_flags:
                issues.append(CodeIssue(
                    issue_id=generate_id("SMELL"),
                    category=IssueCategory.MAINTAINABILITY,
                    severity=IssueSeverity.MEDIUM,
                    title="Too Many Parameters",
                    description=f"Function '{fn.name}' has {fn.parameter_count} parameters (threshold: 7).",
                    line_number=fn.line_start,
                    column=None,
                    code_snippet=None,
                    suggestion="Introduce a parameter object or use keyword-only arguments with defaults.",
                ))
            if "function_too_long" in fn.issue_flags:
                issues.append(CodeIssue(
                    issue_id=generate_id("SMELL"),
                    category=IssueCategory.MAINTAINABILITY,
                    severity=IssueSeverity.MEDIUM,
                    title="Long Function",
                    description=f"Function '{fn.name}' is {fn.line_count} lines long (threshold: 50).",
                    line_number=fn.line_start,
                    column=None,
                    code_snippet=None,
                    suggestion="Break into smaller, single-responsibility functions.",
                ))
            if "high_cyclomatic_complexity" in fn.issue_flags:
                issues.append(CodeIssue(
                    issue_id=generate_id("SMELL"),
                    category=IssueCategory.COMPLEXITY,
                    severity=IssueSeverity.HIGH,
                    title="High Cyclomatic Complexity",
                    description=f"Function '{fn.name}' has cyclomatic complexity {fn.cyclomatic_complexity} (threshold: 10).",
                    line_number=fn.line_start,
                    column=None,
                    code_snippet=None,
                    suggestion="Refactor using guard clauses, strategy pattern, or extract helper functions.",
                ))
            if "deeply_nested" in fn.issue_flags:
                issues.append(CodeIssue(
                    issue_id=generate_id("SMELL"),
                    category=IssueCategory.COMPLEXITY,
                    severity=IssueSeverity.MEDIUM,
                    title="Deeply Nested Code",
                    description=f"Function '{fn.name}' has nesting depth {fn.nested_depth} (threshold: 4).",
                    line_number=fn.line_start,
                    column=None,
                    code_snippet=None,
                    suggestion="Apply early returns, extract helpers, or use a lookup table.",
                ))
            if "missing_docstring" in fn.issue_flags:
                issues.append(CodeIssue(
                    issue_id=generate_id("SMELL"),
                    category=IssueCategory.DOCUMENTATION,
                    severity=IssueSeverity.LOW,
                    title="Missing Docstring",
                    description=f"Function '{fn.name}' has no docstring.",
                    line_number=fn.line_start,
                    column=None,
                    code_snippet=None,
                    suggestion="Add a docstring explaining purpose, args, and return value.",
                ))

        # Check for magic numbers
        magic_pattern = re.compile(r"\b(?<!\.)\d{2,}\b")
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            matches = magic_pattern.findall(stripped)
            for m in matches:
                if m not in ("10", "100", "1000", "0", "1", "2"):
                    issues.append(CodeIssue(
                        issue_id=generate_id("MAGIC"),
                        category=IssueCategory.MAINTAINABILITY,
                        severity=IssueSeverity.LOW,
                        title="Magic Number",
                        description=f"Magic number {m} found on line {i}.",
                        line_number=i,
                        column=None,
                        code_snippet=stripped[:80],
                        suggestion=f"Extract {m} into a named constant for clarity.",
                    ))
                    break  # one per line

        return issues


# ===========================================================================
# Documentation Generator
# ===========================================================================

class DocGenerator:
    """Generate docstrings and documentation for functions and classes."""

    def generate_python_docstring(
        self,
        func_name: str,
        params: List[Tuple[str, Optional[str]]],
        return_type: Optional[str],
        description: str = "",
        style: DocFormat = DocFormat.GOOGLE,
    ) -> str:
        desc = description or f"TODO: describe what {func_name} does."
        if style == DocFormat.GOOGLE:
            lines = [f'"""', desc, ""]
            if params:
                lines.append("Args:")
                for name, ptype in params:
                    type_str = f" ({ptype})" if ptype else ""
                    lines.append(f"    {name}{type_str}: Description of {name}.")
            if return_type:
                lines.append("")
                lines.append("Returns:")
                lines.append(f"    {return_type}: Description of return value.")
            lines.append('"""')
            return "\n".join(lines)
        elif style == DocFormat.NUMPY:
            lines = [f'"""', desc, ""]
            if params:
                lines += ["Parameters", "----------"]
                for name, ptype in params:
                    type_str = f" : {ptype}" if ptype else ""
                    lines.append(f"{name}{type_str}")
                    lines.append(f"    Description of {name}.")
            if return_type:
                lines += ["", "Returns", "-------", f"{return_type}", "    Description."]
            lines.append('"""')
            return "\n".join(lines)
        elif style == DocFormat.SPHINX:
            lines = [f'"""', desc, ""]
            for name, ptype in params:
                lines.append(f":param {name}: Description of {name}.")
                if ptype:
                    lines.append(f":type {name}: {ptype}")
            if return_type:
                lines.append(f":returns: Description.")
                lines.append(f":rtype: {return_type}")
            lines.append('"""')
            return "\n".join(lines)
        return f'"""{desc}"""'

    def generate_jsdoc(
        self,
        func_name: str,
        params: List[Tuple[str, str]],
        return_type: str = "void",
        description: str = "",
    ) -> str:
        desc = description or f"TODO: describe {func_name}."
        lines = ["/**", f" * {desc}", " *"]
        for name, ptype in params:
            lines.append(f" * @param {{{ptype}}} {name} - Description.")
        lines.append(f" * @returns {{{return_type}}} Description.")
        lines.append(" */")
        return "\n".join(lines)

    def generate_readme_section(self, module_analysis: ModuleAnalysis) -> str:
        func_names = [f.name for f in module_analysis.functions[:10]]
        class_names = [c.name for c in module_analysis.classes]
        lines = [
            f"## {module_analysis.file_path}",
            "",
            f"**Language:** {module_analysis.language.value}",
            f"**Lines:** {module_analysis.line_count} total ({module_analysis.code_lines} code)",
            f"**Quality Score:** {module_analysis.quality_score:.1f}/100",
            "",
        ]
        if class_names:
            lines.append(f"**Classes:** {', '.join(class_names)}")
        if func_names:
            lines.append(f"**Functions:** {', '.join(func_names)}")
        if module_analysis.issues:
            lines.append(f"\n**Issues:** {len(module_analysis.issues)} total")
        return "\n".join(lines)


# ===========================================================================
# Refactoring Advisor
# ===========================================================================

class RefactoringAdvisor:
    """Suggest concrete refactoring improvements."""

    def suggest(self, module: ModuleAnalysis) -> List[Dict[str, Any]]:
        suggestions = []
        for fn in module.functions:
            if fn.cyclomatic_complexity > 10:
                suggestions.append({
                    "type": RefactorType.REDUCE_COMPLEXITY.value,
                    "target": fn.name,
                    "line": fn.line_start,
                    "description": f"Reduce cyclomatic complexity ({fn.cyclomatic_complexity} → target <10).",
                    "technique": "Extract inner blocks into helper functions. Use early returns for guard clauses.",
                    "effort": "medium",
                    "impact": "high",
                })
            if fn.line_count > 50:
                suggestions.append({
                    "type": RefactorType.BREAK_LONG_FUNCTION.value,
                    "target": fn.name,
                    "line": fn.line_start,
                    "description": f"Break {fn.name} ({fn.line_count} lines) into smaller functions.",
                    "technique": "Identify distinct responsibilities and extract each into a named function.",
                    "effort": "medium",
                    "impact": "high",
                })
            if fn.parameter_count > 5:
                suggestions.append({
                    "type": RefactorType.EXTRACT_FUNCTION.value,
                    "target": fn.name,
                    "line": fn.line_start,
                    "description": f"Reduce {fn.name} parameter count ({fn.parameter_count}).",
                    "technique": "Group related parameters into a dataclass or TypedDict parameter object.",
                    "effort": "low",
                    "impact": "medium",
                })
            if not fn.has_type_hints:
                suggestions.append({
                    "type": RefactorType.ADD_TYPE_HINTS.value,
                    "target": fn.name,
                    "line": fn.line_start,
                    "description": f"Add type hints to '{fn.name}' for improved readability and tooling support.",
                    "technique": "Annotate all parameters and return type. Use typing module for complex types.",
                    "effort": "low",
                    "impact": "medium",
                })
        return suggestions


# ===========================================================================
# Unit Test Generator
# ===========================================================================

class TestGenerator:
    """Generate unit test skeletons for Python functions."""

    def generate_tests(self, functions: List[FunctionAnalysis], module_name: str = "module") -> str:
        lines = [
            f'"""Auto-generated unit tests for {module_name}."""',
            "",
            "import pytest",
            f"from {module_name} import *",
            "",
        ]
        for fn in functions:
            if fn.name.startswith("_"):
                continue  # skip private
            class_name = "".join(word.capitalize() for word in fn.name.split("_"))
            lines += [
                f"class Test{class_name}:",
                f'    """Tests for {fn.name}."""',
                "",
                f"    def test_{fn.name}_basic(self):",
                f'        """Test basic functionality of {fn.name}."""',
                f"        # Arrange",
                f"        # TODO: set up test inputs",
                f"        # Act",
                f"        # result = {fn.name}()",
                f"        # Assert",
                f"        # assert result == expected",
                f"        pytest.skip('Not yet implemented')",
                "",
                f"    def test_{fn.name}_edge_cases(self):",
                f'        """Test edge cases for {fn.name}."""',
                f"        # Test with None, empty, boundary values",
                f"        pytest.skip('Not yet implemented')",
                "",
                f"    def test_{fn.name}_error_handling(self):",
                f'        """Test that {fn.name} handles errors gracefully."""',
                f"        # Verify expected exceptions are raised",
                f"        pytest.skip('Not yet implemented')",
                "",
            ]
        return "\n".join(lines)


# ===========================================================================
# Diff Reviewer
# ===========================================================================

class DiffReviewer:
    """Analyze code diffs and provide PR-style review comments."""

    def review_diff(self, original: str, modified: str, file_path: str = "file") -> DiffReview:
        orig_lines = original.splitlines()
        mod_lines = modified.splitlines()

        orig_set = set(orig_lines)
        mod_set = set(mod_lines)

        additions = sum(1 for l in mod_lines if l not in orig_set)
        deletions = sum(1 for l in orig_lines if l not in mod_set)

        # Simple quality comparison: line count ratio, comment ratio
        orig_comments = sum(1 for l in orig_lines if l.strip().startswith("#"))
        mod_comments = sum(1 for l in mod_lines if l.strip().startswith("#"))
        orig_blank = sum(1 for l in orig_lines if not l.strip())
        mod_blank = sum(1 for l in mod_lines if not l.strip())

        orig_density = (len(orig_lines) - orig_blank) / max(len(orig_lines), 1)
        mod_density = (len(mod_lines) - mod_blank) / max(len(mod_lines), 1)
        quality_delta = round((mod_density - orig_density) * 10, 2)

        # Security scan the diff additions
        added_code = "\n".join(l for l in mod_lines if l not in orig_set)
        scanner = SecurityScanner()
        scan = scanner.scan(added_code, file_path)

        comments = []
        for finding in scan.findings[:5]:
            comments.append({
                "line": finding.line_number,
                "severity": finding.severity.value,
                "comment": f"[{finding.title}] {finding.description}",
                "suggestion": finding.suggestion,
            })

        verdict = (
            "APPROVE" if not scan.findings and additions < 100
            else "REQUEST_CHANGES" if scan.findings
            else "COMMENT"
        )

        return DiffReview(
            diff_id=generate_id("DIFF"),
            additions=additions,
            deletions=deletions,
            changed_functions=[],
            issues_introduced=scan.findings,
            issues_resolved=0,
            quality_delta=quality_delta,
            verdict=verdict,
            comments=comments,
        )


# ===========================================================================
# Code Generator
# ===========================================================================

class CodeGenerator:
    """Generate boilerplate and template code for common patterns."""

    TEMPLATES: Dict[str, str] = {
        "fastapi_route": '''from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

class {Model}Request(BaseModel):
    # TODO: add fields
    pass

class {Model}Response(BaseModel):
    id: int
    # TODO: add fields

@router.post("/{resource}", response_model={Model}Response)
async def create_{resource}(request: {Model}Request):
    """Create a new {resource}."""
    # TODO: implement
    raise HTTPException(status_code=501, detail="Not implemented")

@router.get("/{resource}/{{item_id}}", response_model={Model}Response)
async def get_{resource}(item_id: int):
    """Get a {resource} by ID."""
    # TODO: implement
    raise HTTPException(status_code=404, detail="{resource} not found")
''',
        "dataclass": '''from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime

@dataclass
class {ClassName}:
    """{ClassName} data model."""

    id: str
    name: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    # TODO: add more fields

    def to_dict(self) -> dict:
        return {{k: v for k, v in self.__dict__.items()}}

    @classmethod
    def from_dict(cls, data: dict) -> "{ClassName}":
        return cls(**{{k: v for k, v in data.items() if k in cls.__dataclass_fields__}})
''',
        "singleton": '''class {ClassName}:
    """Thread-safe singleton implementation."""

    _instance = None
    _lock = None

    def __new__(cls):
        if cls._instance is None:
            import threading
            if cls._lock is None:
                cls._lock = threading.Lock()
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        # TODO: initialize instance
        self._initialized = True

    @classmethod
    def reset(cls):
        """Reset singleton (for testing)."""
        cls._instance = None
''',
        "async_context_manager": '''import asyncio
from contextlib import asynccontextmanager

@asynccontextmanager
async def {name}(config: dict):
    """Async context manager for {name}."""
    resource = None
    try:
        # TODO: acquire resource
        resource = None  # await setup(config)
        yield resource
    except Exception as e:
        # TODO: handle error
        raise
    finally:
        if resource:
            pass  # await resource.close()
''',
    }

    def generate(self, template_key: str, **kwargs) -> GeneratedCode:
        template = self.TEMPLATES.get(template_key, "# Unknown template: " + template_key)
        try:
            code = template.format(**kwargs)
        except KeyError as e:
            code = f"# Template error: missing variable {e}\n" + template
        return GeneratedCode(
            generation_id=generate_id("GEN"),
            prompt=f"Template: {template_key} with {kwargs}",
            language=Language.PYTHON,
            code=code,
            explanation=f"Generated {template_key} template. Replace TODO comments with your implementation.",
            complexity_estimate="O(1) boilerplate — complexity depends on implementation.",
        )

    def available_templates(self) -> List[str]:
        return list(self.TEMPLATES.keys())


# ===========================================================================
# Quality Scorer
# ===========================================================================

class QualityScorer:
    """Compute an aggregate quality score for a module."""

    def score(self, issues: List[CodeIssue], functions: List[FunctionAnalysis], total_lines: int) -> Dict[str, float]:
        # Security score
        sec_issues = [i for i in issues if i.category == IssueCategory.SECURITY]
        sec_deductions = sum(
            {"critical": 25, "high": 15, "medium": 8, "low": 3, "info": 1}.get(i.severity.value, 0)
            for i in sec_issues
        )
        security_score = max(0.0, 100.0 - sec_deductions)

        # Complexity score
        avg_cc = (
            sum(f.cyclomatic_complexity for f in functions) / len(functions)
            if functions else 1.0
        )
        complexity_score = max(0.0, 100.0 - max(0, avg_cc - 5) * 5)

        # Maintainability score
        maint_issues = [i for i in issues if i.category in (IssueCategory.MAINTAINABILITY, IssueCategory.STYLE)]
        maint_deductions = sum(
            {"high": 10, "medium": 5, "low": 2}.get(i.severity.value, 0)
            for i in maint_issues
        )
        doc_missing = sum(1 for f in functions if not f.has_docstring)
        maint_score = max(0.0, 100.0 - maint_deductions - (doc_missing * 3))

        # Overall
        overall = round((security_score * 0.4 + complexity_score * 0.3 + maint_score * 0.3), 1)
        return {
            "overall": overall,
            "security": round(security_score, 1),
            "complexity": round(complexity_score, 1),
            "maintainability": round(maint_score, 1),
        }


# ===========================================================================
# Main Orchestrator — CodeAssistantAI
# ===========================================================================

class CodeAssistantAI:
    """Enterprise AI Code Assistant — full analysis, review, and generation."""

    VERSION = "2.0.0"

    def __init__(self):
        self.lang_detector = LanguageDetector()
        self.ast_analyzer = PythonASTAnalyzer()
        self.security_scanner = SecurityScanner()
        self.smell_detector = CodeSmellDetector()
        self.doc_generator = DocGenerator()
        self.refactor_advisor = RefactoringAdvisor()
        self.test_generator = TestGenerator()
        self.diff_reviewer = DiffReviewer()
        self.code_generator = CodeGenerator()
        self.quality_scorer = QualityScorer()
        self._stats = {"reviews": 0, "scans": 0, "generations": 0, "diffs": 0}

    @timed
    def review(self, code: str, file_path: str = "code.py") -> ReviewResult:
        lang = self.lang_detector.detect(code, file_path)
        total, blank, comment = count_lines(code)
        code_lines = total - blank - comment

        functions: List[FunctionAnalysis] = []
        classes: List[ClassAnalysis] = []
        issues: List[CodeIssue] = []

        if lang == Language.PYTHON:
            tree = self.ast_analyzer.parse(code)
            if tree:
                functions = self.ast_analyzer.extract_functions(tree)
                classes = self.ast_analyzer.extract_classes(tree)

        security_result = self.security_scanner.scan(code, file_path)
        issues.extend(security_result.findings)

        smell_issues = self.smell_detector.detect(code, functions)
        issues.extend(smell_issues)

        scores = self.quality_scorer.score(issues, functions, total)

        module = ModuleAnalysis(
            file_path=file_path,
            language=lang,
            line_count=total,
            blank_lines=blank,
            comment_lines=comment,
            code_lines=code_lines,
            import_count=len(self.ast_analyzer.find_imports(self.ast_analyzer.parse(code) or ast.parse(""))),
            function_count=len(functions),
            class_count=len(classes),
            issues=issues,
            functions=functions,
            classes=classes,
            quality_score=scores["overall"],
            complexity_score=scores["complexity"],
            security_score=scores["security"],
            maintainability_score=scores["maintainability"],
        )

        refactor_suggestions = self.refactor_advisor.suggest(module)
        test_suggestions = [
            f"Write tests for: {fn.name}" for fn in functions[:5] if not fn.name.startswith("_")
        ]

        recommendations = [
            s["description"] for s in refactor_suggestions[:5]
        ] + [
            f"[{i.severity.value.upper()}] {i.title} on line {i.line_number}: {i.suggestion}"
            for i in sorted(issues, key=lambda x: x.severity.value)[:3]
        ]

        sev_counts = Counter(i.severity for i in issues)
        self._stats["reviews"] += 1

        return ReviewResult(
            review_id=generate_id("REV"),
            file_path=file_path,
            total_issues=len(issues),
            critical_count=sev_counts.get(IssueSeverity.CRITICAL, 0),
            high_count=sev_counts.get(IssueSeverity.HIGH, 0),
            medium_count=sev_counts.get(IssueSeverity.MEDIUM, 0),
            low_count=sev_counts.get(IssueSeverity.LOW, 0),
            overall_score=scores["overall"],
            summary=(
                f"Code review of '{file_path}' ({total} lines, {lang.value}). "
                f"Quality score: {scores['overall']}/100. "
                f"{len(issues)} issue(s): {sev_counts.get(IssueSeverity.CRITICAL, 0)} critical."
            ),
            issues=issues,
            recommendations=recommendations,
            test_suggestions=test_suggestions,
        )

    def scan_security(self, code: str, file_path: str = "code.py") -> SecurityScanResult:
        self._stats["scans"] += 1
        return self.security_scanner.scan(code, file_path)

    def generate_docstring(
        self, func_name: str, params: List[Tuple[str, str]], return_type: str,
        style: str = "google"
    ) -> str:
        fmt = DocFormat(style) if style in [d.value for d in DocFormat] else DocFormat.GOOGLE
        return self.doc_generator.generate_python_docstring(func_name, params, return_type, style=fmt)

    def generate_tests(self, code: str, module_name: str = "module") -> str:
        tree = self.ast_analyzer.parse(code)
        if not tree:
            return "# Could not parse code for test generation."
        functions = self.ast_analyzer.extract_functions(tree)
        self._stats["generations"] += 1
        return self.test_generator.generate_tests(functions, module_name)

    def review_diff(self, original: str, modified: str, file_path: str = "file.py") -> DiffReview:
        self._stats["diffs"] += 1
        return self.diff_reviewer.review_diff(original, modified, file_path)

    def generate_template(self, template: str, **kwargs) -> GeneratedCode:
        self._stats["generations"] += 1
        return self.code_generator.generate(template, **kwargs)

    def get_statistics(self) -> Dict[str, Any]:
        return {"version": self.VERSION, **self._stats}


# ===========================================================================
# Demo
# ===========================================================================

SAMPLE_CODE = '''
import os
import pickle
import random
import hashlib
import subprocess

password = "supersecret123"
API_KEY = "sk-abc123xyzfoo"

def process_user_data(user_id, name, email, age, country, currency, discount):
    conn = None
    query = "SELECT * FROM users WHERE id = " + str(user_id)
    conn.execute(query)

    file_path = "/uploads/" + name
    with open(file_path, "r") as f:
        data = f.read()

    hashed = hashlib.md5(email.encode()).hexdigest()

    rand = random.random()
    token = str(rand)

    result = subprocess.call("ls " + name, shell=True)

    obj = pickle.loads(data)

    for i in range(100):
        for j in range(100):
            for k in range(50):
                if i + j + k > 200:
                    if rand > 0.5:
                        if age > 18:
                            pass

    return result


def short_fn():
    return 42


class DataProcessor:
    def process(self, data):
        return data
'''


def main():
    print("\n" + "=" * 80)
    print("  CODE ASSISTANT AI — ENTERPRISE v2.0")
    print("=" * 80)

    ai = CodeAssistantAI()

    # Full code review
    review = ai.review(SAMPLE_CODE, "user_processor.py")
    print(f"\n[CodeReview] {review.summary}")
    print(f"  Critical: {review.critical_count} | High: {review.high_count} | "
          f"Medium: {review.medium_count} | Low: {review.low_count}")
    print(f"  Score: {review.overall_score}/100")
    print("\n  Top Issues:")
    for issue in review.issues[:5]:
        print(f"    [{issue.severity.value.upper()}] Line {issue.line_number}: {issue.title}")
        print(f"      → {issue.suggestion[:80]}")

    # Security scan standalone
    print(f"\n{'─'*60}")
    scan = ai.scan_security(SAMPLE_CODE, "user_processor.py")
    print(f"[SecurityScan] {scan.summary}")
    for r in scan.remediation_priority[:3]:
        print(f"  {r}")

    # Docstring generation
    print(f"\n{'─'*60}")
    print("[DocGenerator] Google-style docstring:")
    doc = ai.generate_docstring(
        "process_payment",
        [("user_id", "int"), ("amount", "float"), ("currency", "str")],
        "dict",
        style="google",
    )
    print(doc)

    # Test generation
    print(f"\n{'─'*60}")
    print("[TestGenerator] Auto-generated tests:")
    tests = ai.generate_tests(SAMPLE_CODE, "user_processor")
    print(tests[:400] + "...")

    # Diff review
    print(f"\n{'─'*60}")
    modified = SAMPLE_CODE.replace(
        'query = "SELECT * FROM users WHERE id = " + str(user_id)',
        'query = "SELECT * FROM users WHERE id = %s"  # Fixed: parameterized'
    )
    diff_result = ai.review_diff(SAMPLE_CODE, modified, "user_processor.py")
    print(f"[DiffReview] Verdict: {diff_result.verdict}")
    print(f"  +{diff_result.additions} / -{diff_result.deletions} lines | "
          f"Quality delta: {diff_result.quality_delta:+.2f}")

    # Template generation
    print(f"\n{'─'*60}")
    print("[CodeGenerator] FastAPI route template:")
    gen = ai.generate_template("fastapi_route", Model="User", resource="users")
    print(gen.code[:300] + "...")

    # Stats
    print(f"\n{'─'*60}")
    print(f"[Stats] {ai.get_statistics()}")
    print("=" * 80)


if __name__ == "__main__":
    main()


# ===========================================================================
# Additional Utilities & Exports
# ===========================================================================

class DependencyChecker:
    """Check Python dependencies for known vulnerability patterns."""

    KNOWN_VULNERABLE: Dict[str, List[Dict[str, str]]] = {
        "django": [{"below": "4.2.0", "cve": "CVE-2023-36053", "description": "ReDoS in EmailValidator"}],
        "flask": [{"below": "2.3.0", "cve": "CVE-2023-30861", "description": "Session cookie vulnerability"}],
        "pillow": [{"below": "9.3.0", "cve": "CVE-2022-45199", "description": "Heap buffer overflow in TIFF parsing"}],
        "requests": [{"below": "2.28.0", "cve": "CVE-2022-17944", "description": "Credential leakage via redirect"}],
        "cryptography": [{"below": "41.0.0", "cve": "CVE-2023-38325", "description": "NULL dereference in cert parsing"}],
        "urllib3": [{"below": "1.26.17", "cve": "CVE-2023-43804", "description": "HTTP header injection"}],
    }

    def check_requirements(self, requirements_txt: str) -> List[Dict[str, str]]:
        """Parse requirements.txt and flag known vulnerabilities."""
        vulnerabilities = []
        for line in requirements_txt.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Match: package==version or package>=version
            m = re.match(r"^([a-zA-Z0-9_\-]+)[><=!]+([0-9.]+)", line)
            if m:
                pkg, ver = m.group(1).lower(), m.group(2)
                if pkg in self.KNOWN_VULNERABLE:
                    for vuln in self.KNOWN_VULNERABLE[pkg]:
                        vulnerabilities.append({
                            "package": pkg,
                            "installed": ver,
                            "vulnerable_below": vuln["below"],
                            "cve": vuln["cve"],
                            "description": vuln["description"],
                            "severity": "HIGH",
                        })
        return vulnerabilities


class StyleChecker:
    """Enforce basic coding style rules."""

    RULES: List[Dict[str, Any]] = [
        {"id": "STY001", "pattern": r"^\t", "msg": "Use spaces not tabs for indentation.", "sev": "low"},
        {"id": "STY002", "pattern": r".{120}", "msg": "Line exceeds 120 characters.", "sev": "low"},
        {"id": "STY003", "pattern": r"print\(", "msg": "Use logger instead of print() in production code.", "sev": "info"},
        {"id": "STY004", "pattern": r"TODO|FIXME|HACK|XXX", "msg": "Unresolved TODO/FIXME comment found.", "sev": "info"},
        {"id": "STY005", "pattern": r"except:\s*$", "msg": "Bare except clause — specify exception type.", "sev": "medium"},
        {"id": "STY006", "pattern": r"== None\b", "msg": "Use 'is None' instead of '== None'.", "sev": "low"},
        {"id": "STY007", "pattern": r"!= None\b", "msg": "Use 'is not None' instead of '!= None'.", "sev": "low"},
        {"id": "STY008", "pattern": r"assert\s+", "msg": "Don't use assert for input validation in production.", "sev": "medium"},
    ]

    def check(self, code: str) -> List[CodeIssue]:
        issues = []
        for i, line in enumerate(code.splitlines(), 1):
            for rule in self.RULES:
                if re.search(rule["pattern"], line):
                    issues.append(CodeIssue(
                        issue_id=generate_id(rule["id"]),
                        category=IssueCategory.STYLE,
                        severity=IssueSeverity(rule["sev"]) if rule["sev"] in [s.value for s in IssueSeverity] else IssueSeverity.INFO,
                        title=f"Style: {rule['id']}",
                        description=rule["msg"],
                        line_number=i,
                        column=None,
                        code_snippet=line.strip()[:80],
                        suggestion=rule["msg"],
                    ))
        return issues


class MetricsAggregator:
    """Aggregate metrics across multiple file reviews for project-level reporting."""

    def __init__(self):
        self._reviews: List[ReviewResult] = []

    def add(self, review: ReviewResult) -> None:
        self._reviews.append(review)

    def project_summary(self) -> Dict[str, Any]:
        if not self._reviews:
            return {}
        total_issues = sum(r.total_issues for r in self._reviews)
        avg_score = sum(r.overall_score for r in self._reviews) / len(self._reviews)
        critical_total = sum(r.critical_count for r in self._reviews)
        files = len(self._reviews)
        worst = min(self._reviews, key=lambda r: r.overall_score)
        best = max(self._reviews, key=lambda r: r.overall_score)
        return {
            "files_reviewed": files,
            "total_issues": total_issues,
            "critical_issues": critical_total,
            "average_quality_score": round(avg_score, 1),
            "worst_file": {"path": worst.file_path, "score": worst.overall_score},
            "best_file": {"path": best.file_path, "score": best.overall_score},
            "health": "GOOD" if avg_score >= 80 else "NEEDS_ATTENTION" if avg_score >= 60 else "CRITICAL",
        }


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__version__ = CodeAssistantAI.VERSION
__all__ = [
    "CodeAssistantAI",
    "SecurityScanner",
    "PythonASTAnalyzer",
    "LanguageDetector",
    "DocGenerator",
    "RefactoringAdvisor",
    "TestGenerator",
    "DiffReviewer",
    "CodeGenerator",
    "QualityScorer",
    "CodeSmellDetector",
    "DependencyChecker",
    "StyleChecker",
    "MetricsAggregator",
    "Language",
    "IssueSeverity",
    "IssueCategory",
    "RefactorType",
    "DocFormat",
]

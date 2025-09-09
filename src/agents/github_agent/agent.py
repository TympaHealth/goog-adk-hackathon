# src/agents/github_agent/agent.py
import os
import re
import json
import base64
from typing import Any, Dict, List, Optional, Tuple, Iterable

import requests
from google.adk.agents import Agent

# ───────────────────────────────────────────────────────────────────────────────
# Repo / token normalization
# ───────────────────────────────────────────────────────────────────────────────
def _normalize_repo(repo: str) -> str:
    """
    Accepts:
      - owner/repo
      - https://github.com/owner/repo(.git)
      - git@github.com:owner/repo(.git)
    Returns canonical 'owner/repo'.
    """
    repo = (repo or "").strip()
    repo = re.sub(r"^https?://github\.com/", "", repo)
    repo = re.sub(r"^git@github\.com:", "", repo)
    if repo.endswith(".git"):
        repo = repo[:-4]
    if "/" not in repo:
        raise ValueError("GITHUB_REPO must be 'owner/repo' or a GitHub URL.")
    return repo


# ───────────────────────────────────────────────────────────────────────────────
# Minimal GitHub client
# ───────────────────────────────────────────────────────────────────────────────
class GitHubClient:
    def __init__(self, token: str, repo: str, api_base: str = "https://api.github.com"):
        if not token:
            raise ValueError("GITHUB_TOKEN is required")
        if not repo:
            raise ValueError("GITHUB_REPO is required")
        self.repo = _normalize_repo(repo)
        self.token = token
        self.api_base = api_base.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"token {self.token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            }
        )

    def _url(self, path: str) -> str:
        return f"{self.api_base}{path}"

    def list_open_prs(self, labels: Optional[List[str]] = None, per_page: int = 30) -> List[Dict[str, Any]]:
        params = {"state": "open", "per_page": per_page}
        if labels:
            params["labels"] = ",".join(labels)
        r = self.session.get(self._url(f"/repos/{self.repo}/pulls"), params=params)
        r.raise_for_status()
        return r.json()

    def get_pr(self, number: int) -> Dict[str, Any]:
        r = self.session.get(self._url(f"/repos/{self.repo}/pulls/{number}"))
        r.raise_for_status()
        return r.json()

    def list_pr_files(self, number: int, per_page: int = 300) -> List[Dict[str, Any]]:
        params = {"per_page": per_page}
        r = self.session.get(self._url(f"/repos/{self.repo}/pulls/{number}/files"), params=params)
        r.raise_for_status()
        return r.json()

    # Optional: read a file blob at a ref (requires Contents: Read)
    def read_file(self, path: str, ref: str) -> str:
        r = self.session.get(self._url(f"/repos/{self.repo}/contents/{path}"), params={"ref": ref})
        if r.status_code == 404:
            return ""
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and data.get("encoding") == "base64" and data.get("content"):
            return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
        return ""

    # Comment on PR (issues comments endpoint) → needs Issues: Read & Write
    def comment_on_issue(self, number: int, body: str) -> Dict[str, Any]:
        payload = {"body": body}
        r = self.session.post(self._url(f"/repos/{self.repo}/issues/{number}/comments"), json=payload)
        r.raise_for_status()
        return r.json()


# ───────────────────────────────────────────────────────────────────────────────
# Optional LLM (LangChain)
# ───────────────────────────────────────────────────────────────────────────────
def _load_langchain_llm() -> Optional[Any]:
    # Prefer OpenAI if configured
    try:
        from langchain_openai import ChatOpenAI  # type: ignore
        if os.getenv("OPENAI_API_KEY"):
            return ChatOpenAI(model="gpt-4o-mini")
    except Exception:
        pass
    # Then Google GenAI
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
        if os.getenv("GOOGLE_API_KEY"):
            return ChatGoogleGenerativeAI(model="gemini-1.5-pro")
    except Exception:
        pass
    return None

# ───────────────────────────────────────────────────────────────────────────────
# Software-Bug-Assistant style context builders
# ───────────────────────────────────────────────────────────────────────────────
def _summarize_changes(pr: Dict[str, Any], files: List[Dict[str, Any]]) -> str:
    total_add = sum(f.get("additions", 0) for f in files)
    total_del = sum(f.get("deletions", 0) for f in files)
    parts = [
        f"PR #{pr.get('number')}: {pr.get('title')} (by {pr.get('user',{}).get('login')})",
        f"Base: {pr.get('base',{}).get('ref')} -> Head: {pr.get('head',{}).get('ref')}",
        f"Files changed: {len(files)}, +{total_add}/-{total_del}",
        "",
    ]
    for f in files[:20]:
        parts.append(f"- {f.get('filename')} (+{f.get('additions')}/-{f.get('deletions')})")
    if len(files) > 20:
        parts.append(f"... and {len(files)-20} more files")
    return "\n".join(parts)

def _build_bug_assistant_prompt(pr: Dict[str, Any], files: List[Dict[str, Any]], diff_snippets: str) -> str:
    return (
        "Role: Software Bug Assistant.\n"
        "Goals:\n"
        " - Identify bugs, regressions, security issues (incl. secrets), correctness problems.\n"
        " - Flag missing tests/docs and weak error handling.\n"
        " - Provide minimal, actionable fixes with reasoning.\n"
        " - Use GitHub suggestion blocks for small fixes.\n"
        "Output format:\n"
        " - Risk tier: S0 blocker / S1 must-fix / S2 should-fix / S3 nice-to-have (1–2 line rationale)\n"
        " - Numbered list of findings with file:line references\n"
        " - Suggestion blocks where possible\n"
        "Be concise; focus on changed code.\n\n"
        f"PR Title: {pr.get('title')}\n\nPR Body:\n{pr.get('body') or ''}\n\n"
        f"Summary of Changes:\n{_summarize_changes(pr, files)}\n\n"
        f"Diff Snippets (truncated):\n{diff_snippets}"
    )

def _gather_diff_snippets(files: List[Dict[str, Any]], max_files: int = 30, max_chars_per_file: int = 8000) -> str:
    parts: List[str] = []
    for f in files[:max_files]:
        patch = f.get("patch") or ""
        if not patch:
            continue
        parts.append(f"File: {f.get('filename')}\n{patch[:max_chars_per_file]}")
    return "\n\n".join(parts)


# ───────────────────────────────────────────────────────────────────────────────
# Unified diff parsing + static checks
# ───────────────────────────────────────────────────────────────────────────────
_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")

def _iter_added_lines(patch: str) -> Iterable[Tuple[int, str]]:
    """
    Yield (new_line_number, added_line_text) from a unified diff patch.
    """
    new_line = None
    for raw in patch.splitlines():
        if raw.startswith('@@'):
            m = _HUNK_RE.match(raw)
            if m:
                start = int(m.group(1))
                new_line = start
            continue
        if new_line is None:
            continue
        if raw.startswith('+') and not raw.startswith('+++'):
            yield new_line, raw[1:]
            new_line += 1
        elif raw.startswith('-') and not raw.startswith('---'):
            pass  # deletion
        else:
            new_line += 1

_SECRET_PATTERNS = [
    ("S0", r"AKIA[0-9A-Z]{16}", "Possible AWS Access Key ID detected."),
    ("S0", r"-----BEGIN (?:RSA |DSA |EC )?PRIVATE KEY-----", "Private key material committed."),
    ("S1", r"sk-[A-Za-z0-9]{20,}", "Token that looks like a secret (sk-...)."),
    ("S1", r"AIza[0-9A-Za-z\-_]{35}", "String that looks like a Google API key."),
    ("S1", r"(?:api[_-]?key|token|secret)\s*[:=]\s*['\"][A-Za-z0-9_\-]{16,}['\"]", "Hard-coded credential-like value."),
]
_INSECURE_PATTERNS = [
    ("S1", r"\bsubprocess\.(?:run|Popen|call)\s*\([^)]*shell\s*=\s*True", "Use of shell=True; prefer shell=False and arg list."),
    ("S1", r"\beval\s*\(", "Use of eval() is dangerous; use ast.literal_eval for simple types."),
    ("S1", r"\bexec\s*\(", "Use of exec() is dangerous; avoid if possible."),
    ("S1", r"\byaml\.load\s*\(", "yaml.load() without SafeLoader is unsafe; use yaml.safe_load()."),
    ("S1", r"\bpickle\.loads?\s*\(", "Unpickling untrusted data is unsafe; avoid or validate source."),
    ("S1", r"requests\.(?:get|post|put|delete)\([^)]*verify\s*=\s*False", "TLS verification disabled; remove verify=False."),
    ("S2", r"hashlib\.md5\s*\(", "MD5 is weak; prefer SHA-256."),
]
_DEBUG_PATTERNS = [
    ("S2", r"\bpdb\.set_trace\s*\(", "Leftover debugger call."),
    ("S2", r"\bdebugger\s*;", "Leftover JS debugger statement."),
    ("S3", r"\bconsole\.log\s*\(", "console.log in committed code; consider proper logging."),
    ("S3", r"\bprint\s*\(", "print in library code; consider logging."),
]
_TODO_PATTERNS = [("S3", r"\b(?:TODO|FIXME|HACK)\b", "Found a TODO/FIXME/HACK marker; ensure it's tracked.")]

def _suggestion_for_line(path: str, line: str) -> Optional[str]:
    ext = os.path.splitext(path)[1].lower()
    if re.search(r"\byaml\.load\s*\(", line):
        return "```suggestion\n" + re.sub(r"\byaml\.load\s*\(", "yaml.safe_load(", line) + "\n```"
    if re.search(r"requests\.(?:get|post|put|delete)\([^)]*verify\s*=\s*False", line):
        return "```suggestion\n" + re.sub(r"verify\s*=\s*False", "verify=True", line) + "\n```"
    if re.search(r"\beval\s*\(", line):
        return "```suggestion\n" + re.sub(r"\beval\s*\(", "ast.literal_eval(", line) + "\n```"
    if ext == ".py" and re.search(r"\bprint\s*\(", line):
        return "```suggestion\n" + re.sub(r"\bprint\s*\(", "logging.debug(", line) + "\n```"
    if ext in {".js", ".jsx", ".ts", ".tsx"} and re.search(r"\bconsole\.log\s*\(", line):
        return "```suggestion\n" + re.sub(r"\bconsole\.log\s*\(", "console.debug(", line) + "\n```"
    if re.search(r"\bsubprocess\.(?:run|Popen|call)\s*\([^)]*shell\s*=\s*True", line):
        return "```suggestion\n" + re.sub(r"shell\s*=\s*True", "shell=False", line) + "\n```"
    return None

def _run_static_checks_on_patch(path: str, patch: str) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    ext = os.path.splitext(path)[1].lower()
    for new_line_no, text in _iter_added_lines(patch):
        if len(text) > 120:
            findings.append({"severity": "S3", "type": "style/long-line", "file": path, "line": new_line_no, "message": f"Line length {len(text)} > 120; consider wrapping.", "suggestion": None})
        for sev, pat, msg in _SECRET_PATTERNS:
            if re.search(pat, text):
                findings.append({"severity": sev, "type": "secret", "file": path, "line": new_line_no, "message": msg, "suggestion": None})
        for sev, pat, msg in _INSECURE_PATTERNS:
            if re.search(pat, text):
                findings.append({"severity": sev, "type": "insecure", "file": path, "line": new_line_no, "message": msg, "suggestion": _suggestion_for_line(path, text)})
        for sev, pat, msg in _DEBUG_PATTERNS:
            if "print" in pat and ext != ".py":
                continue
            if "console\\.log" in pat and ext not in {".js", ".jsx", ".ts", ".tsx"}:
                continue
            if re.search(pat, text):
                findings.append({"severity": sev, "type": "debug", "file": path, "line": new_line_no, "message": msg, "suggestion": _suggestion_for_line(path, text)})
        for sev, pat, msg in _TODO_PATTERNS:
            if re.search(pat, text):
                findings.append({"severity": sev, "type": "todo", "file": path, "line": new_line_no, "message": msg, "suggestion": None})
    return findings


# ───────────────────────────────────────────────────────────────────────────────
# Reviewer & Critique loop (sub-agents via prompting)
# ───────────────────────────────────────────────────────────────────────────────
DEFAULT_PREFS = """# Codebase Review Preferences
tone: concise, kind, direct
risk-tier: S0 blocker, S1 must-fix, S2 should-fix, S3 nice-to-have
security:
  - no secrets (tokens/keys/private keys)
  - avoid eval/exec, subprocess shell=True, yaml.load (use safe_load), verify=False
  - prefer SHA-256 over MD5
testing:
  - add/expand unit tests for non-trivial logic & bug fixes
docs:
  - update README/CHANGELOG and docstrings when behavior changes
style:
  - type hints for new/changed Python code; short functions; clear names
suggestions:
  - include GitHub ```suggestion``` blocks for trivial fixes
"""

def _read_prefs_text() -> str:
    path = os.getenv("CODEBASE_PREFS_PATH", "config/codebase_prefs.txt")
    try:
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read().strip()
            return txt if txt else DEFAULT_PREFS
    except Exception:
        return DEFAULT_PREFS

def _build_draft_prompt(pr: Dict[str, Any], files: List[Dict[str, Any]], diffs: str, prefs: str) -> str:
    return (
        "Role: ReviewerAgent.\n"
        "Task: Write a high-quality PR review comment from the diffs and PR context.\n"
        "Follow these preferences:\n" + prefs + "\n\n"
        "Output:\n"
        "- Start with risk tier (S0/S1/S2/S3) + 1–2 line rationale.\n"
        "- Numbered actionable findings with file:line.\n"
        "- Include GitHub ```suggestion``` blocks for trivial fixes.\n"
        "- Ask for tests/docs where appropriate; avoid leaking secrets.\n\n"
        f"PR Title: {pr.get('title')}\n\nPR Body:\n{pr.get('body') or ''}\n\n"
        f"Summary of Changes:\n{_summarize_changes(pr, files)}\n\n"
        f"Diff Snippets (truncated):\n{diffs}\n"
    )

def _build_critique_prompt(comment: str, prefs: str) -> str:
    return (
        "Role: CritiqueAgent.\n"
        "Critique the PR review COMMENT against the preferences.\n"
        "Score 0–1 on: structure, specificity (file:line), actionability (suggestions), coverage, safety.\n"
        "Return STRICT JSON: {\"scores\":{...}, \"missing\":[...], \"advice\":[...]}.\n"
        "Preferences:\n" + prefs + "\n\n"
        "COMMENT:\n" + comment
    )

def _build_revision_prompt(comment: str, critique_json: str, prefs: str) -> str:
    return (
        "Role: ReviewerAgent.\n"
        "Revise the comment using CRITIQUE JSON & preferences. Fix structure, add file:line, add suggestions, improve clarity.\n"
        "Return the improved COMMENT only.\n\n"
        "PREFERENCES:\n" + prefs + "\n\n"
        "CRITIQUE JSON:\n" + critique_json + "\n\n"
        "COMMENT:\n" + comment
    )

def _safe_json(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s)
    except Exception:
        return {"scores": {}, "missing": [], "advice": ["Critique JSON parse failed; improve structure and specificity."]}

def _ask_llm(llm: Any, prompt: str) -> str:
    if llm is None:
        # conservative fallback
        return (
            "Risk: S2 should-fix — Automated fallback.\n\n"
            "1) Add/expand tests for modified logic.\n"
            "2) Improve error handling and input validation.\n"
            "3) Replace insecure patterns (eval/shell=True/verify=False/MD5).\n"
            "4) Update docs and type hints where behavior changed.\n"
        )
    try:
        resp = llm.invoke(prompt)  # type: ignore[attr-defined]
        return getattr(resp, "content", resp) if not isinstance(resp, str) else resp
    except Exception as e:
        return f"(LLM error) {e}"


# ───────────────────────────────────────────────────────────────────────────────
# Tools exposed to the agent
# ───────────────────────────────────────────────────────────────────────────────
def get_runtime_config() -> dict:
    repo = os.getenv("GITHUB_REPO", "")
    mode = os.getenv("PR_REVIEWER_MODE", "dry_run")
    return {"repo": _normalize_repo(repo) if repo else "", "mode": mode}

def list_open_prs_tool(limit: int) -> dict:
    repo = os.getenv("GITHUB_REPO", "")
    token = os.getenv("GITHUB_TOKEN", "")
    try:
        gh = GitHubClient(token=token, repo=repo)
        prs = gh.list_open_prs()
        prs_sorted = sorted(prs, key=lambda p: p.get("number", 0), reverse=True)
        return {
            "status": "success",
            "repo": _normalize_repo(repo),
            "count": min(len(prs_sorted), limit),
            "prs": [{"number": p["number"], "title": p["title"]} for p in prs_sorted[:limit]],
        }
    except ValueError as e:
        return {"status": "error", "message": str(e), "hint": "Set GITHUB_TOKEN and GITHUB_REPO in .env"}
    except requests.HTTPError as e:
        return {"status": "error", "message": f"GitHub API error: {e}", "hint": "Check PAT repository selection & permissions."}

def prepare_bug_review(pr_number: int, include_file_blobs: bool) -> dict:
    repo = os.getenv("GITHUB_REPO", "")
    token = os.getenv("GITHUB_TOKEN", "")
    try:
        gh = GitHubClient(token=token, repo=repo)
        pr = gh.get_pr(pr_number)
        files = gh.list_pr_files(pr_number)
        diff_snippets = _gather_diff_snippets(files)
        prompt = _build_bug_assistant_prompt(pr, files, diff_snippets)

        blobs: List[Dict[str, Any]] = []
        if include_file_blobs:
            head_ref = pr.get("head", {}).get("sha", "")
            for f in files[:20]:
                path = f.get("filename")
                if not path:
                    continue
                try:
                    content = gh.read_file(path, head_ref)
                    if content:
                        blobs.append({"path": path, "sample": content[:4000]})
                except requests.HTTPError as e:
                    blobs.append({"path": path, "sample": "", "error": f"read_file_failed: {e}"})

        return {"status": "success", "repo": _normalize_repo(repo), "pr_number": pr_number, "head_sha": pr.get("head", {}).get("sha", ""), "analysis_prompt": prompt, "blobs": blobs}
    except ValueError as e:
        return {"status": "error", "message": str(e), "hint": "Set GITHUB_TOKEN and valid GITHUB_REPO slug."}
    except requests.HTTPError as e:
        code = getattr(e.response, "status_code", None)
        if code == 403:
            return {"status": "error", "message": "Forbidden when reading PR details.", "hint": "Add Pull requests: Read. For file contents, also add Contents: Read."}
        if code == 404:
            return {"status": "error", "message": "PR not found (or token lacks access)."}
        return {"status": "error", "message": f"GitHub API error: {e}"}

def run_static_checks_tool(pr_number: int) -> dict:
    repo = os.getenv("GITHUB_REPO", "")
    token = os.getenv("GITHUB_TOKEN", "")
    try:
        gh = GitHubClient(token=token, repo=repo)
        files = gh.list_pr_files(pr_number)
        findings: List[Dict[str, Any]] = []
        for f in files:
            patch = f.get("patch") or ""
            path = f.get("filename") or ""
            if not patch or not path:
                continue
            findings.extend(_run_static_checks_on_patch(path, patch))
        sev_order = {"S0": 0, "S1": 1, "S2": 2, "S3": 3}
        findings.sort(key=lambda x: (sev_order.get(x["severity"], 99), x["file"], x["line"]))
        return {"status": "success", "repo": _normalize_repo(repo), "pr_number": pr_number, "findings": findings}
    except ValueError as e:
        return {"status": "error", "message": str(e), "hint": "Set GITHUB_TOKEN and valid GITHUB_REPO slug."}
    except requests.HTTPError as e:
        code = getattr(e.response, "status_code", None)
        if code == 403:
            return {"status": "error", "message": "Forbidden when reading PR files (need Pull requests: Read)."}
        if code == 404:
            return {"status": "error", "message": "PR not found (or token lacks access)."}
        return {"status": "error", "message": f"GitHub API error: {e}"}

def post_review_comment_safe(pr_number: int, body: str) -> dict:
    repo = os.getenv("GITHUB_REPO", "")
    token = os.getenv("GITHUB_TOKEN", "")
    try:
        gh = GitHubClient(token=token, repo=repo)
        res = gh.comment_on_issue(pr_number, body)
        return {"status": "success", "posted": True, "comment_id": res.get("id")}
    except requests.HTTPError as e:
        code = getattr(e.response, "status_code", None)
        if code == 403:
            return {"status": "success", "posted": False, "note": "Missing permission to post. Grant Issues: Read & Write on the token."}
        return {"status": "error", "posted": False, "message": f"GitHub API error: {e}"}
    except ValueError as e:
        return {"status": "error", "posted": False, "message": str(e)}

def count_pr_analyzed_lines(pr_number: int) -> dict:
    repo = os.getenv("GITHUB_REPO", "")
    token = os.getenv("GITHUB_TOKEN", "")
    try:
        gh = GitHubClient(token=token, repo=repo)
        files = gh.list_pr_files(pr_number)
        analyzed_lines = 0
        for f in files:
            patch = f.get("patch") or ""
            if not patch:
                continue
            for _ in _iter_added_lines(patch):
                analyzed_lines += 1
        return {"status": "success", "repo": _normalize_repo(repo), "pr_number": pr_number, "files_changed": len(files), "analyzed_lines": analyzed_lines}
    except ValueError as e:
        return {"status": "error", "message": str(e), "hint": "Set GITHUB_TOKEN and valid GITHUB_REPO slug."}
    except requests.HTTPError as e:
        code = getattr(e.response, "status_code", None)
        if code == 403:
            return {"status": "error", "message": "Forbidden when reading PR files (need Pull requests: Read)."}
        if code == 404:
            return {"status": "error", "message": "PR not found (or token lacks access)."}
        return {"status": "error", "message": f"GitHub API error: {e}"}

# NEW: write preferences template to disk
def write_prefs_template(path: str) -> dict:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):
            return {"status": "exists", "path": path}
        with open(path, "w", encoding="utf-8") as f:
            f.write(DEFAULT_PREFS.strip() + "\n")
        return {"status": "written", "path": path}
    except Exception as e:
        return {"status": "error", "message": str(e), "hint": "Check path permissions."}

# NEW: full write → critique → rewrite loop
def run_comment_review_loop(pr_number: int, iterations: int, post: int) -> dict:
    """
    Draft a PR review comment, critique it against codebase preferences, and iteratively improve it.
    - iterations: number of critique→revise cycles (1..3 recommended)
    - post: 1 to attempt posting to PR (requires Issues: Read & Write), else 0
    Returns the final comment and the step-by-step history.
    """
    repo = os.getenv("GITHUB_REPO", "")
    token = os.getenv("GITHUB_TOKEN", "")
    mode = os.getenv("PR_REVIEWER_MODE", "dry_run")
    llm = _load_langchain_llm()
    prefs = _read_prefs_text()

    try:
        gh = GitHubClient(token=token, repo=repo)
        pr = gh.get_pr(pr_number)
        files = gh.list_pr_files(pr_number)
        files = [f for f in files if f.get("patch")]
        diffs = _gather_diff_snippets(files)

        draft_prompt = _build_draft_prompt(pr, files, diffs, prefs)
        current_comment = _ask_llm(llm, draft_prompt)

        history: List[Dict[str, Any]] = [{"stage": "draft", "comment": current_comment}]
        n = max(1, min(int(iterations), 3))
        for i in range(n):
            critique_prompt = _build_critique_prompt(current_comment, prefs)
            critique_raw = _ask_llm(llm, critique_prompt)
            critique = _safe_json(critique_raw)
            history.append({"stage": f"critique_{i+1}", "critique": critique})
            revision_prompt = _build_revision_prompt(current_comment, json.dumps(critique), prefs)
            improved = _ask_llm(llm, revision_prompt)
            history.append({"stage": f"revise_{i+1}", "comment": improved})
            current_comment = improved

        # Optional posting
        posted = False
        note = None
        if post == 1 and mode in ("summary_comment", "full_review"):
            sig = f"\n\n[adk-pr-reviewer] sha={pr.get('head', {}).get('sha', '')}"
            try:
                gh.comment_on_issue(pr_number, current_comment.strip() + sig)
                posted = True
            except requests.HTTPError as e:
                if getattr(e.response, "status_code", None) == 403:
                    note = "Missing permission to post (need Issues: Read & Write). Showing the comment here instead."
                else:
                    note = f"GitHub API error posting comment: {e}"

        return {
            "status": "posted" if posted else "ok",
            "repo": _normalize_repo(repo),
            "pr_number": pr_number,
            "iterations": n,
            "posted": posted,
            "post_note": note,
            "final_comment": "(comment posted on GitHub)" if posted else current_comment.strip(),
            "history": history,
        }

    except ValueError as e:
        return {"status": "error", "message": str(e), "hint": "Set GITHUB_TOKEN and valid GITHUB_REPO slug."}
    except requests.HTTPError as e:
        code = getattr(e.response, "status_code", None)
        if code == 403:
            return {"status": "error", "message": "Forbidden when reading PR (need Pull requests: Read)."}
        if code == 404:
            return {"status": "error", "message": "PR not found (or token lacks access)."}
        return {"status": "error", "message": f"GitHub API error: {e}"}


# ───────────────────────────────────────────────────────────────────────────────
# ADK root agent
# ───────────────────────────────────────────────────────────────────────────────
root_agent = Agent(
    name="github_agent",
    model="gemini-2.0-flash",
    description="Software Bug Assistant with an iterative reviewer–critic loop, static checks, and optional GitHub posting.",
    instruction=(
        "You are an autonomous PR reviewer.\n"
        "\n"
        "When asked to review (even without a PR number):\n"
        "1) Call get_runtime_config() to see mode (dry_run or summary_comment).\n"
        "2) Call list_open_prs_tool(limit=10); if no number given, pick the newest.\n"
        "3) Optionally call run_static_checks_tool(pr_number=<n>) to augment your reasoning.\n"
        "4) Call run_comment_review_loop(pr_number=<n>, iterations=2, post=1 if mode != 'dry_run' else 0) to draft→critique→rewrite.\n"
        "5) If posting is forbidden (403), explain briefly and show the FINAL COMMENT in chat.\n"
        "\n"
        "Rules:\n"
        "- Never include secret strings in output.\n"
        "- If any tool errors, still provide the best review you can with available context.\n"
        "- Keep reviews concise and actionable; focus on changed code and preferences."
    ),
    tools=[
        get_runtime_config,
        list_open_prs_tool,
        prepare_bug_review,
        run_static_checks_tool,
        post_review_comment_safe,
        count_pr_analyzed_lines,
        write_prefs_template,
        run_comment_review_loop,
    ],
)

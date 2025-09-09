# src/agents/crit_agent/agent.py
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

    def read_file(self, path: str, ref: str) -> str:
        r = self.session.get(self._url(f"/repos/{self.repo}/contents/{path}"), params={"ref": ref})
        if r.status_code == 404:
            return ""
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and data.get("encoding") == "base64" and data.get("content"):
            return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
        return ""

    # Issues comment (used to persist summary) → requires Issues: Read & Write
    def comment_on_issue(self, number: int, body: str) -> Dict[str, Any]:
        payload = {"body": body}
        r = self.session.post(self._url(f"/repos/{self.repo}/issues/{number}/comments"), json=payload)
        r.raise_for_status()
        return r.json()

# ───────────────────────────────────────────────────────────────────────────────
# Unified diff helpers + static checks (same as earlier, trimmed where sensible)
# ───────────────────────────────────────────────────────────────────────────────
_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
def _iter_added_lines(patch: str):
    new_line = None
    for raw in patch.splitlines():
        if raw.startswith('@@'):
            m = _HUNK_RE.match(raw)
            if m:
                new_line = int(m.group(1))
            continue
        if new_line is None:
            continue
        if raw.startswith('+') and not raw.startswith('+++'):
            yield new_line, raw[1:]
            new_line += 1
        elif raw.startswith('-') and not raw.startswith('---'):
            pass
        else:
            new_line += 1

SECRET_PATS = [
    ("S0", r"AKIA[0-9A-Z]{16}", "Possible AWS Access Key ID"),
    ("S0", r"-----BEGIN (?:RSA |DSA |EC )?PRIVATE KEY-----", "Private key material"),
    ("S1", r"sk-[A-Za-z0-9]{20,}", "Secret-like token (sk-...)"),
    ("S1", r"AIza[0-9A-Za-z\-_]{35}", "String that looks like a Google API key"),
    ("S1", r"(?:api[_-]?key|token|secret)\s*[:=]\s*['\"][A-Za-z0-9_\-]{16,}['\"]", "Hard-coded credential-like value"),
]
INSECURE_PATS = [
    ("S1", r"\bsubprocess\.(?:run|Popen|call)\s*\([^)]*shell\s*=\s*True", "Use of shell=True; prefer args list & shell=False"),
    ("S1", r"\beval\s*\(", "Use of eval() is dangerous"),
    ("S1", r"\byaml\.load\s*\(", "yaml.load() without SafeLoader is unsafe; use yaml.safe_load()"),
    ("S1", r"requests\.(?:get|post|put|delete)\([^)]*verify\s*=\s*False", "TLS verify disabled; remove verify=False"),
    ("S2", r"hashlib\.md5\s*\(", "MD5 is weak; use SHA-256"),
]
DEBUG_PATS = [
    ("S2", r"\bpdb\.set_trace\s*\(", "Leftover debugger"),
    ("S2", r"\bdebugger\s*;", "JS debugger statement"),
    ("S3", r"\bconsole\.log\s*\(", "console.log; prefer proper logging"),
    ("S3", r"\bprint\s*\(", "print in library code; prefer logging"),
]
TODO_PATS = [("S3", r"\b(?:TODO|FIXME|HACK)\b", "TODO/FIXME/HACK marker")]

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
            findings.append({"severity": "S3","type": "style/long-line","file": path,"line": new_line_no,"message": f"Line length {len(text)} > 120","suggestion": None})
        for sev, pat, msg in SECRET_PATS:
            if re.search(pat, text):
                findings.append({"severity": sev,"type": "secret","file": path,"line": new_line_no,"message": msg,"suggestion": None})
        for sev, pat, msg in INSECURE_PATS:
            if re.search(pat, text):
                findings.append({"severity": sev,"type": "insecure","file": path,"line": new_line_no,"message": msg,"suggestion": _suggestion_for_line(path, text)})
        for sev, pat, msg in DEBUG_PATS:
            if "print" in pat and ext != ".py":
                continue
            if "console\\.log" in pat and ext not in {".js",".jsx",".ts",".tsx"}:
                continue
            if re.search(pat, text):
                findings.append({"severity": sev,"type": "debug","file": path,"line": new_line_no,"message": msg,"suggestion": _suggestion_for_line(path, text)})
        for sev, pat, msg in TODO_PATS:
            if re.search(pat, text):
                findings.append({"severity": sev,"type": "todo","file": path,"line": new_line_no,"message": msg,"suggestion": None})
    return findings

# ───────────────────────────────────────────────────────────────────────────────
# Context builders for the LLM
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

def _gather_diff_snippets(files: List[Dict[str, Any]], max_files: int = 30, max_chars_per_file: int = 8000) -> str:
    parts: List[str] = []
    for f in files[:max_files]:
        patch = f.get("patch") or ""
        if not patch: continue
        parts.append(f"File: {f.get('filename')}\n{patch[:max_chars_per_file]}")
    return "\n\n".join(parts)

def _build_review_prompt(pr: Dict[str, Any], files: List[Dict[str, Any]], diff_snippets: str) -> str:
    return (
        "You are Reviewer-1 (drafting reviewer).\n"
        "Goals:\n"
        " - Identify bugs, regressions, security issues (incl. secrets), and correctness problems.\n"
        " - Flag missing tests/docs and weak error handling.\n"
        " - Provide minimal, actionable fixes with reasoning.\n"
        "Output:\n"
        " - One-line risk tier S0/S1/S2/S3 with rationale.\n"
        " - Numbered findings with file:line and suggestion blocks where trivial fixes exist.\n\n"
        f"PR Title: {pr.get('title')}\n\nPR Body:\n{pr.get('body') or ''}\n\n"
        f"Summary of Changes:\n{_summarize_changes(pr, files)}\n\n"
        f"Diff Snippets (truncated):\n{diff_snippets}"
    )

def _build_critique_prompt(draft: str, file_names: List[str], static_findings: List[Dict[str, Any]]) -> str:
    """
    Critic-1 rubric. The LLM will use this to critique and request revisions.
    """
    rubric = (
        "You are Critic-1. Rigorously evaluate the draft review.\n"
        "Score on 0–1 for each dimension and give concrete revision advice:\n"
        "- Structure: has risk tier + numbered items?\n"
        "- Specificity: file:line references present and correct-looking?\n"
        "- Actionability: includes GitHub suggestion blocks where feasible?\n"
        "- Coverage: mentions most changed files? (Files: " + ", ".join(file_names[:12]) + ")\n"
        "- Safety: avoids leaking secrets; flags any detected by static checks.\n"
        "Static checks (high-signal):\n"
        + json.dumps(static_findings[:20], ensure_ascii=False)
        + "\n\nReturn JSON with keys: scores, missing, advice."
    )
    return f"{rubric}\n\nDRAFT REVIEW:\n{draft}"

def _build_consolidation_prompt(draft: str, critique_json: str) -> str:
    return (
        "You are Consolidator-1. Merge the draft review with the critique.\n"
        "Apply the advice precisely; keep it concise, actionable, and focused on changed code.\n"
        "Return FINAL REVIEW only (no meta commentary). Ensure:\n"
        "- Risk tier present.\n"
        "- Numbered, file:line-referenced findings.\n"
        "- Include suggestion blocks where trivial fixes exist.\n\n"
        f"CRITIQUE JSON:\n{critique_json}\n\nDRAFT REVIEW:\n{draft}"
    )

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
        return {"status": "error", "message": f"GitHub API error: {e}", "hint": "Check PAT repo selection & permissions."}

def prepare_context(pr_number: int, include_file_blobs: bool) -> dict:
    """
    Fetch PR + file list + diff snippets + static findings.
    The LLM uses the returned prompts to draft, critique, and consolidate.
    """
    repo = os.getenv("GITHUB_REPO", "")
    token = os.getenv("GITHUB_TOKEN", "")
    try:
        gh = GitHubClient(token=token, repo=repo)
        pr = gh.get_pr(pr_number)
        files = gh.list_pr_files(pr_number)
        diff_snippets = _gather_diff_snippets(files)
        draft_prompt = _build_review_prompt(pr, files, diff_snippets)

        # static checks (on added lines)
        static_findings: List[Dict[str, Any]] = []
        file_names: List[str] = []
        for f in files:
            path = f.get("filename") or ""
            file_names.append(path)
            patch = f.get("patch") or ""
            if patch:
                static_findings.extend(_run_static_checks_on_patch(path, patch))

        # (Optional) include file blobs for deeper context
        blobs: List[Dict[str, Any]] = []
        if include_file_blobs:
            head_ref = pr.get("head", {}).get("sha", "")
            for f in files[:12]:
                path = f.get("filename")
                if not path:
                    continue
                try:
                    content = gh.read_file(path, head_ref)
                    if content:
                        blobs.append({"path": path, "sample": content[:4000]})
                except requests.HTTPError:
                    pass  # ignore blob errors

        critique_prompt = _build_critique_prompt(
            draft="<<DRAFT_REVIEW_GOES_HERE>>",
            file_names=file_names,
            static_findings=static_findings[:40],
        )

        return {
            "status": "success",
            "repo": _normalize_repo(repo),
            "pr_number": pr_number,
            "head_sha": pr.get("head", {}).get("sha", ""),
            "draft_prompt": draft_prompt,
            "critique_prompt_template": critique_prompt,
            "file_names": file_names,
            "static_findings": static_findings,
            "blobs": blobs,
        }
    except ValueError as e:
        return {"status": "error", "message": str(e), "hint": "Set GITHUB_TOKEN and valid GITHUB_REPO slug."}
    except requests.HTTPError as e:
        code = getattr(e.response, "status_code", None)
        if code == 403:
            return {"status": "error", "message": "Forbidden fetching PR; need Pull requests: Read. (Contents: Read only if blobs used.)"}
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
            return {"status": "success", "posted": False, "note": "Missing permission to post. Grant Issues: Read & Write."}
        return {"status": "error", "posted": False, "message": f"GitHub API error: {e}"}
    except ValueError as e:
        return {"status": "error", "posted": False, "message": str(e)}

# ───────────────────────────────────────────────────────────────────────────────
# The “multi-agent” loop is instructions-driven:
# Reviewer-1 drafts → Critic-1 scores/advises → Consolidator-1 produces FINAL REVIEW.
# We keep it single-agent in ADK but orchestrate the loop via prompts + tools.
# ───────────────────────────────────────────────────────────────────────────────
root_agent = Agent(
    name="crit_agent",
    model="gemini-2.0-flash",
    description="Iterative PR reviewer with a critic loop: draft → critique → consolidate; optionally post a summary.",
    instruction=(
        "Iterative workflow (at most 2 passes):\n"
        "1) Call get_runtime_config().\n"
        "2) If the user didn't specify a PR, call list_open_prs_tool(limit=10) and choose the most recent.\n"
        "3) Call prepare_context(pr_number=<n>, include_file_blobs=false). Use only diffs by default; blobs only if essential.\n"
        "4) Act as Reviewer-1: use draft_prompt to produce DRAFT REVIEW.\n"
        "5) Act as Critic-1: take critique_prompt_template and replace <<DRAFT_REVIEW_GOES_HERE>> with your DRAFT REVIEW. "
        "Return a small JSON object with scores (0-1), missing items, and concrete advice.\n"
        "6) Act as Consolidator-1: merge DRAFT REVIEW with Critic-1 advice into a FINAL REVIEW. Ensure:\n"
        "   - One-line risk tier (S0/S1/S2/S3) with rationale\n"
        "   - Numbered findings with file:line references\n"
        "   - Include ```suggestion blocks``` where trivial fixes exist\n"
        "   - Incorporate any high-signal static findings from prepare_context.static_findings\n"
        "7) If any score < 0.8, optionally do one more short revise pass; then stop.\n"
        "8) If mode == 'summary_comment', post a concise summary via post_review_comment_safe(). "
        "If posting is not permitted (posted=false), state that briefly and show the FINAL REVIEW here.\n"
        "\n"
        "Always avoid reproducing secrets verbatim. Keep the output concise and actionable."
    ),
    tools=[get_runtime_config, list_open_prs_tool, prepare_context, post_review_comment_safe],
)

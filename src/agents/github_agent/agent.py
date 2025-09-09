# src/agents/github_agent/agent.py
import os
import re
import json
import base64
from typing import Any, Dict, List, Optional

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
# Tools exposed to the agent
# ───────────────────────────────────────────────────────────────────────────────
def get_runtime_config() -> dict:
    """
    Expose mode/slug so the agent can decide whether to auto-comment.
    """
    repo = os.getenv("GITHUB_REPO", "")
    mode = os.getenv("PR_REVIEWER_MODE", "dry_run")
    return {
        "repo": _normalize_repo(repo) if repo else "",
        "mode": mode,  # dry_run | summary_comment
    }


def list_open_prs_tool(limit: int) -> dict:
    """
    List up to 'limit' open PRs in the configured repo.
    """
    repo = os.getenv("GITHUB_REPO", "")
    token = os.getenv("GITHUB_TOKEN", "")
    try:
        gh = GitHubClient(token=token, repo=repo)
        prs = gh.list_open_prs()
        # sort desc by number so 'latest' is first
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
    """
    Collect PR metadata + changed-file diffs for analysis.
    - include_file_blobs=True will also try to read file contents at HEAD (needs Contents: Read).
    The agent will read 'analysis_prompt' and synthesize the review.
    """
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

        return {
            "status": "success",
            "repo": _normalize_repo(repo),
            "pr_number": pr_number,
            "head_sha": pr.get("head", {}).get("sha", ""),
            "analysis_prompt": prompt,
            "blobs": blobs,
        }
    except ValueError as e:
        return {"status": "error", "message": str(e), "hint": "Set GITHUB_TOKEN and valid GITHUB_REPO slug."}
    except requests.HTTPError as e:
        code = getattr(e.response, "status_code", None)
        if code == 403:
            return {"status": "error", "message": "Forbidden when reading PR details.", "hint": "Add Pull requests: Read. For file contents, also add Contents: Read."}
        if code == 404:
            return {"status": "error", "message": "PR not found (or token lacks access)."}
        return {"status": "error", "message": f"GitHub API error: {e}"}


def post_review_comment_safe(pr_number: int, body: str) -> dict:
    """
    Try to post a single summary comment to the PR.
    Returns posted:false with a note on 403 instead of raising.
    Requires Issues: Read & Write to actually succeed.
    """
    repo = os.getenv("GITHUB_REPO", "")
    token = os.getenv("GITHUB_TOKEN", "")
    try:
        gh = GitHubClient(token=token, repo=repo)
        res = gh.comment_on_issue(pr_number, body)
        return {"status": "success", "posted": True, "comment_id": res.get("id")}
    except requests.HTTPError as e:
        code = getattr(e.response, "status_code", None)
        if code == 403:
            return {
                "status": "success",
                "posted": False,
                "note": "Missing permission to post. Grant Issues: Read & Write on the token.",
            }
        return {"status": "error", "posted": False, "message": f"GitHub API error: {e}"}
    except ValueError as e:
        return {"status": "error", "posted": False, "message": str(e)}


# ───────────────────────────────────────────────────────────────────────────────
# ADK root agent
# ───────────────────────────────────────────────────────────────────────────────
root_agent = Agent(
    name="github_agent",
    model="gemini-2.0-flash",
    description="Software Bug Assistant: automatically picks a PR, analyzes diffs for bugs/regressions/security, and posts a summary when allowed.",
    instruction=(
        "You are an autonomous PR reviewer.\n"
        "\n"
        "When the user asks for a review (even without a PR number):\n"
        "1) Call get_runtime_config() to read the mode (dry_run or summary_comment).\n"
        "2) Call list_open_prs_tool(limit=10). If there is at least one open PR and the user didn't specify a number, pick the most recent (highest PR number).\n"
        "3) Call prepare_bug_review(pr_number=<chosen>, include_file_blobs=false). Use only diffs by default; request blobs only if essential.\n"
        "4) Read 'analysis_prompt' (and 'blobs' if present) and write the full review:\n"
        "   - Start with a single-line risk tier: S0/S1/S2/S3 with short rationale.\n"
        "   - Provide a numbered list of concrete, testable findings with file:line references.\n"
        "   - Include GitHub suggestion blocks for trivial fixes, e.g.:\n"
        "     ```suggestion\n"
        "     corrected line here\n"
        "     ```\n"
        "   - Note missing tests/docs, secret/security risks, and possible regressions.\n"
        "5) If mode == 'summary_comment', attempt to persist the summary:\n"
        "   - Call post_review_comment_safe(pr_number=<chosen>, body=<your concise summary + key fixes>). If it returns posted:false, DO NOT show an error—just say you couldn't post due to permissions and include the full review in the chat.\n"
        "\n"
        "Rules:\n"
        "- Never include any secret-like strings or credentials in the output.\n"
        "- If any tool returns an error, continue with whatever context you have and still provide a useful review.\n"
        "- Keep the review concise and actionable; focus on changed code.\n"
    ),
    tools=[get_runtime_config, list_open_prs_tool, prepare_bug_review, post_review_comment_safe],
)

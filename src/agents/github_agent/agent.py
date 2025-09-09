import os
import json
import base64
import time
from typing import Any, Dict, List, Optional, Tuple

import requests


class GitHubClient:
    def __init__(self, token: str, repo: str, api_base: str = "https://api.github.com"):
        if not token:
            raise ValueError("GITHUB_TOKEN is required")
        if not repo or "/" not in repo:
            raise ValueError("repo must be in the form 'owner/name'")
        self.token = token
        self.repo = repo
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

    def comment_on_issue(self, number: int, body: str) -> Dict[str, Any]:
        payload = {"body": body}
        r = self.session.post(self._url(f"/repos/{self.repo}/issues/{number}/comments"), json=payload)
        r.raise_for_status()
        return r.json()

    def create_review(self, number: int, body: str, event: str = "COMMENT", comments: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"event": event}
        if body:
            payload["body"] = body
        if comments:
            payload["comments"] = comments
        r = self.session.post(self._url(f"/repos/{self.repo}/pulls/{number}/reviews"), json=payload)
        r.raise_for_status()
        return r.json()

    def get_issue_comments(self, number: int) -> List[Dict[str, Any]]:
        r = self.session.get(self._url(f"/repos/{self.repo}/issues/{number}/comments"))
        r.raise_for_status()
        return r.json()


def _load_langchain_llm() -> Optional[Any]:
    try:
        from langchain_openai import ChatOpenAI  # type: ignore
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            return ChatOpenAI(model="gpt-4o-mini")
    except Exception:
        pass
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
        if os.getenv("GOOGLE_API_KEY"):
            return ChatGoogleGenerativeAI(model="gemini-1.5-pro")
    except Exception:
        pass
    return None


def _load_github_toolkit(token: str) -> Optional[List[Any]]:
    try:
        from langchain_community.utilities.github import GitHubAPIWrapper  # type: ignore
        from langchain_community.agent_toolkits.github.toolkit import GitHubToolkit  # type: ignore

        wrapper = GitHubAPIWrapper(github_token=token)
        toolkit = GitHubToolkit.from_github_api_wrapper(wrapper)
        return toolkit.get_tools()
    except Exception:
        return None


class PRReviewerAgent:
    def __init__(
        self,
        repo: Optional[str] = None,
        mode: str = "dry_run",
        allow_labels: Optional[List[str]] = None,
        ignore_paths: Optional[List[str]] = None,
        max_comments: int = 10,
        dedupe_tag: str = "[adk-pr-reviewer]",
    ):
        token = os.getenv("GITHUB_TOKEN", "")
        repo = repo or os.getenv("GITHUB_REPO", "")
        self.gh = GitHubClient(token=token, repo=repo)
        self.repo = repo
        self.mode = mode
        self.allow_labels = allow_labels or []
        self.ignore_paths = ignore_paths or []
        self.max_comments = max_comments
        self.dedupe_tag = dedupe_tag
        self.llm = _load_langchain_llm()
        self.github_tools = _load_github_toolkit(token)

    def list_open_prs(self) -> List[Dict[str, Any]]:
        labels = self.allow_labels if self.allow_labels else None
        return self.gh.list_open_prs(labels=labels)

    def _already_reviewed(self, pr_number: int, head_sha: str) -> bool:
        sig = f"{self.dedupe_tag} sha={head_sha}"
        for c in self.gh.get_issue_comments(pr_number):
            if c.get("body", "").strip().endswith(sig):
                return True
        return False

    def _summarize_changes(self, pr: Dict[str, Any], files: List[Dict[str, Any]]) -> str:
        total_add = sum(f.get("additions", 0) for f in files)
        total_del = sum(f.get("deletions", 0) for f in files)
        parts = [
            f"PR #{pr.get('number')}: {pr.get('title')} (by {pr.get('user',{}).get('login')})",
            f"Base: {pr.get('base',{}).get('ref')} -> Head: {pr.get('head',{}).get('ref')}",
            f"Files changed: {len(files)}, +{total_add}/-{total_del}",
            ""
        ]
        for f in files[:20]:
            parts.append(f"- {f.get('filename')} (+{f.get('additions')}/-{f.get('deletions')})")
        if len(files) > 20:
            parts.append(f"... and {len(files)-20} more files")
        return "\n".join(parts)

    def _build_review_prompt(self, pr: Dict[str, Any], files: List[Dict[str, Any]]) -> str:
        summary = self._summarize_changes(pr, files)
        diff_snippets = []
        for f in files[:10]:
            patch = f.get("patch") or ""
            if not patch:
                continue
            header = f"File: {f.get('filename')}\n"
            diff_snippets.append(header + patch[:6000])
        snippets = "\n\n".join(diff_snippets)
        body = pr.get("body") or ""
        instructions = (
            "You are a precise code reviewer. Provide:\n"
            "1) A concise PR summary and risk/impact assessment.\n"
            "2) Specific, actionable review comments (cite files/lines).\n"
            "3) Suggested changes using GitHub suggestion blocks where possible.\n"
            "4) Note missing tests/docs or breaking changes.\n"
            "Keep it helpful, concise, and concrete."
        )
        return f"{instructions}\n\nPR Title: {pr.get('title')}\n\nPR Body:\n{body}\n\nSummary of Changes:\n{summary}\n\nDiff Snippets:\n{snippets}"

    def _generate_review(self, pr: Dict[str, Any], files: List[Dict[str, Any]]) -> str:
        prompt = self._build_review_prompt(pr, files)
        if self.llm is None:
            # Fallback minimal heuristic if no LLM is wired
            return (
                "Automated review (LLM not configured):\n"
                f"{self._summarize_changes(pr, files)}\n\n"
                "Suggestions:\n- Consider adding/expanding tests for modified logic.\n"
                "- Ensure docs and type hints are updated.\n"
                "- Check error handling and edge cases.\n"
                "- Verify performance on large inputs.\n"
            )
        try:
            resp = self.llm.invoke(prompt)  # type: ignore[attr-defined]
            if isinstance(resp, str):
                return resp
            content = getattr(resp, "content", None)
            return content or str(resp)
        except Exception as e:
            return f"Automated review failed to generate via LLM: {e}"

    def review_pr(self, pr_number: int) -> Dict[str, Any]:
        pr = self.gh.get_pr(pr_number)
        head_sha = pr.get("head", {}).get("sha", "")
        if self._already_reviewed(pr_number, head_sha):
            return {"status": "skipped", "reason": "already_reviewed", "pr": pr_number, "sha": head_sha}

        files = self.gh.list_pr_files(pr_number)
        files = [f for f in files if not self._is_ignored(f.get("filename", ""))]
        review_body = self._generate_review(pr, files)

        sig = f"\n\n{self.dedupe_tag} sha={head_sha}"
        if self.mode == "dry_run":
            return {
                "status": "dry_run",
                "pr": pr_number,
                "sha": head_sha,
                "review": review_body + sig,
            }
        if self.mode in ("summary_comment", "full_review"):
            self.gh.comment_on_issue(pr_number, review_body + sig)
            return {"status": "commented", "pr": pr_number, "sha": head_sha}
        return {"status": "noop", "pr": pr_number, "sha": head_sha}

    def _is_ignored(self, path: str) -> bool:
        if not self.ignore_paths:
            return False
        try:
            import fnmatch
            return any(fnmatch.fnmatch(path, pat) for pat in self.ignore_paths)
        except Exception:
            return False

    def run(self, pr_number: Optional[int] = None) -> Dict[str, Any]:
        if pr_number is not None:
            return self.review_pr(pr_number)
        open_prs = self.list_open_prs()
        results = []
        for pr in open_prs:
            results.append(self.review_pr(pr.get("number")))
            if len(results) >= self.max_comments:
                break
        return {"results": results}


def _config_from_env() -> Dict[str, Any]:
    cfg_str = os.getenv("ADK_AGENT_CONFIG", "")
    if cfg_str:
        try:
            return json.loads(cfg_str)
        except Exception:
            pass
    return {
        "repo": os.getenv("GITHUB_REPO", ""),
        "mode": os.getenv("PR_REVIEWER_MODE", "dry_run"),
        "allow_labels": os.getenv("PR_REVIEW_ALLOW_LABELS", "").split(",") if os.getenv("PR_REVIEW_ALLOW_LABELS") else [],
        "ignore_paths": os.getenv("PR_REVIEW_IGNORE_PATHS", "").split(",") if os.getenv("PR_REVIEW_IGNORE_PATHS") else ["**/vendor/**", "**/*.lock", "**/dist/**"],
        "max_comments": int(os.getenv("PR_REVIEW_MAX_COMMENTS", "10")),
        "dedupe_tag": os.getenv("PR_REVIEW_DEDUPE_TAG", "[adk-pr-reviewer]"),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GitHub PR Reviewer Agent")
    parser.add_argument("--repo", type=str, default=os.getenv("GITHUB_REPO", ""))
    parser.add_argument("--pr", type=int, default=None, help="PR number to review; leave empty to process open PRs")
    parser.add_argument("--mode", type=str, default=os.getenv("PR_REVIEWER_MODE", "dry_run"), choices=["dry_run", "summary_comment", "full_review"])
    args = parser.parse_args()

    cfg = _config_from_env()
    if args.repo:
        cfg["repo"] = args.repo
    if args.mode:
        cfg["mode"] = args.mode

    agent = PRReviewerAgent(**cfg)
    result = agent.run(pr_number=args.pr)
    print(json.dumps(result, indent=2))

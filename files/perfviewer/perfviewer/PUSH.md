# Pushing this to GitHub

I can't create the repo for you — that needs your GitHub credentials, and
authenticating as you is not something I'll do. Everything here is ready to push;
these are the commands.

## Option A — GitHub CLI

```bash
cd perfviewer
git init
git add .
git commit -m "Design docs, prior-art research, and prototypes for an LLM serving perf viewer"
gh repo create perfviewer --public --source=. --remote=origin --push
```

## Option B — web UI

1. Create an empty public repo at https://github.com/new (no README, no
   .gitignore — this directory has them)
2. Then:

```bash
cd perfviewer
git init
git add .
git commit -m "Design docs, prior-art research, and prototypes for an LLM serving perf viewer"
git branch -M main
git remote add origin git@github.com:<your-username>/perfviewer.git
git push -u origin main
```

## Before you make it public — check these

**The atom material.** Nothing in this repo names atom or describes it, and that
was deliberate. If you add anything later about an internal AMD framework — trace
formats, kernel names, instrumentation surfaces, comparative numbers — that
belongs in a private repo, and probably needs a conversation with whoever owns it
first. Once something is in git history, removing it is not a delete, it's a
rewrite plus a force push plus hoping nobody cloned.

**The MI355X numbers** here are all from AMD's public product page. Any number
you later add from your own runs is performance data on unreleased or
customer-configured hardware. Different category, different rules.

**Employer IP.** You mentioned working on this alongside a day job. Worth knowing
where your employment agreement lands on side projects before a public repo with
your name on it exists.

**Suggested repo description**

> Design notes and prototypes for a performance-engineering tool for LLM serving —
> architecture graph as coordinate system, trace overlay, roofline, cross-run
> regression.

**Suggested topics:** `llm-inference`, `performance-engineering`, `vllm`,
`sglang`, `rocm`, `visualization`, `mixture-of-experts`, `roofline`

## A note on the prototypes

They contain synthetic measurements. The README says so, but if this repo gets
attention that caveat needs to survive contact with someone skimming — consider
a banner comment at the top of each file before you publicise it anywhere.

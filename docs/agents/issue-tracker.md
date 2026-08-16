# Issue Tracker：GitHub

本仓库的 issue 和 spec 存放在 `fixedpointworks/ds4` 的 GitHub Issues 中。所有操作均使用 `gh` CLI。

## 约定

- **创建 issue**：`gh issue create --title "..." --body "..."`。多行正文使用 heredoc。
- **读取 issue**：`gh issue view <number> --comments`；需要结构化处理时，同时获取 labels，并使用 `jq` 筛选 comments。
- **列出 issue**：`gh issue list --state open --json number,title,body,labels,comments --jq '[.[] | {number, title, body, labels: [.labels[].name], comments: [.comments[].body]}]'`，并按需添加 `--label` 和 `--state` 筛选条件。
- **评论 issue**：`gh issue comment <number> --body "..."`
- **添加/移除标签**：`gh issue edit <number> --add-label "..."` / `--remove-label "..."`
- **关闭 issue**：`gh issue close <number> --comment "..."`

在 clone 内运行时，`gh` 通常会根据 `git remote -v` 自动推断仓库。本仓库的 issue tracker 是 `origin` 对应的 `fixedpointworks/ds4`；如有歧义，显式传入 `--repo fixedpointworks/ds4`。

## 将 Pull Request 作为 triage 入口

**PRs as a request surface: no.**

_若本仓库以后将外部 PR 视为功能请求，可将上面的 `no` 改为 `yes`；`/triage` 会读取此标志。_

当标志为 `yes` 时，PR 使用与 issue 相同的标签和状态，并采用对应的 `gh pr` 操作：

- **读取 PR**：`gh pr view <number> --comments`；使用 `gh pr diff <number>` 读取 diff。
- **列出需要 triage 的外部 PR**：运行 `gh pr list --state open --json number,title,body,labels,author,authorAssociation,comments`，仅保留 `authorAssociation` 为 `CONTRIBUTOR`、`FIRST_TIME_CONTRIBUTOR` 或 `NONE` 的条目，排除 `OWNER`、`MEMBER` 和 `COLLABORATOR`。
- **评论、加标签或关闭**：使用 `gh pr comment`、`gh pr edit --add-label` / `--remove-label`、`gh pr close`。

GitHub 的 issue 和 PR 共用同一个编号空间，因此单独的 `#42` 可能指向任意一种对象。先运行 `gh pr view 42`，失败后再运行 `gh issue view 42`。

## 当技能要求“发布到 Issue Tracker”时

创建一个 GitHub issue。

## 当技能要求“获取相关 ticket”时

运行 `gh issue view <number> --comments`。

## Wayfinding 操作

供 `/wayfinder` 使用。**Map** 是一个独立 issue，**child** issue 是其下的 ticket。

- **Map**：一个带有 `wayfinder:map` 标签的 issue，其正文保存 Notes、Decisions-so-far 和 Fog。使用 `gh issue create --label wayfinder:map` 创建。
- **Child ticket**：通过 GitHub sub-issues endpoint（使用 `gh api`）关联到 map 的 issue。如果仓库未启用 sub-issues，则将 child 加入 map 正文的任务列表，并在 child 正文顶部写入 `Part of #<map>`。标签使用 `wayfinder:<type>`，其中 `<type>` 为 `research`、`prototype`、`grilling` 或 `task`。认领后，将 ticket 指派给当前负责开发者。
- **Blocking**：优先使用 GitHub 原生 issue dependencies，作为规范且在 UI 中可见的表示。使用 `gh api --method POST repos/<owner>/<repo>/issues/<child>/dependencies/blocked_by -F issue_id=<blocker-db-id>` 添加依赖边；`<blocker-db-id>` 是 blocker 的数字数据库 ID，通过 `gh api repos/<owner>/<repo>/issues/<n> --jq .id` 获取，不是 `#number` 或 `node_id`。GitHub 的 `issue_dependencies_summary.blocked_by` 只统计仍打开的 blocker，是实时门禁。如果 dependencies 不可用，则在 child 正文顶部写入 `Blocked by: #<n>, #<n>`。所有 blocker 关闭后，ticket 即解除阻塞。
- **Frontier query**：列出 map 下仍打开的 child（使用 `gh issue list --state open`，并限定在 map 的 sub-issues 或任务列表中）；排除存在打开 blocker（`issue_dependencies_summary.blocked_by > 0`，或 `Blocked by` 行引用了仍打开的 issue）或已有 assignee 的条目；按 map 中的顺序选择第一个。
- **Claim**：`gh issue edit <n> --add-assignee @me`，这是 session 的第一次写操作。
- **Resolve**：运行 `gh issue comment <n> --body "<answer>"`，再运行 `gh issue close <n>`，最后将上下文指针（gist + link）追加到 map 的 Decisions-so-far。

# 从本地电脑发布已验证 ExpertPlex 分支

用户要求从本地电脑推送 GitHub。已在本地 Mac 将已验证提交推送到 `WeiMengze666/sglang` 的 `feat/expertplex-b300`，发布头 `abcaa6d564b64b1231ca574b205fa16753345044` 已用 `git ls-remote` 核验。服务器缺少 GitHub 写凭据不再构成发布阻塞。

开发和 GPU 验证继续在 b300 工作区进行；通过 source-only bundle 将源码、测试和使用文档导入本地仓库后发布。bundle 的路径、SHA256、HEAD 与基准提交以 `/home/axis/mengze/expertplex/evidence/20261027-new-b300/active-validation.json` 的 `source_bundle` 为准。原始请求、score、trace 和实验数据不入 GitHub。

本地仓库含未提交修改时，使用独立导入 ref，保留当前 HEAD、文件和暂存区。例如：

```bash
# 在本地电脑执行，先下载服务器生成的 source-only bundle
scp b300:/home/axis/mengze/expertplex/evidence/20261107-new-b300/sglang-expertplex-b300-final.bundle /path/to/sglang-expertplex-b300-final.bundle
shasum -a 256 /path/to/sglang-expertplex-b300-final.bundle
# 必须匹配 active-validation.json 的 source_bundle.sha256
git bundle verify /path/to/sglang-expertplex-b300-final.bundle
git fetch /path/to/sglang-expertplex-b300-final.bundle HEAD:refs/remotes/expertplex-verified/publication
# 检查目标分支当前 HEAD、提交祖先关系和完整文件范围
git ls-remote origin refs/heads/feat/expertplex-b300
git log --oneline origin/feat/expertplex-b300..expertplex-verified/publication
git diff --name-only origin/feat/expertplex-b300..expertplex-verified/publication
git merge-base --is-ancestor origin/feat/expertplex-b300 expertplex-verified/publication
git push origin expertplex-verified/publication:refs/heads/feat/expertplex-b300
git ls-remote origin refs/heads/feat/expertplex-b300
```

发布使用非强制 fast-forward。若远端有其他更新，先核验并整合提交，不覆盖远端。导入前后比较本地 HEAD、工作区和暂存区的指纹，确保用户修改保留。

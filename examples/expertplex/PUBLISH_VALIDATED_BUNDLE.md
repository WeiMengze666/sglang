# 发布已验证 ExpertPlex 分支

当前 b300 checkout 无有效的 `WeiMengze666` GitHub 写入凭据。已验证提交保存在 source-only bundle：

`/home/axis/mengze/expertplex/evidence/20261107-new-b300/sglang-expertplex-b300-final.bundle`

SHA256：以 `/home/axis/mengze/expertplex/evidence/20261027-new-b300/active-validation.json` 的 `source_bundle.sha256` 为准。

在有 `WeiMengze666/sglang` 写权限的环境中：

```bash
sha256sum sglang-expertplex-b300-final.bundle
# 必须匹配 active-validation.json 的 source_bundle.sha256
git clone https://github.com/WeiMengze666/sglang.git sglang-publish
cd sglang-publish
git fetch /path/to/sglang-expertplex-b300-final.bundle HEAD:refs/remotes/expertplex/local
# 查看并推送 bundle 中的分支头
git log --oneline expertplex/local -5
git push origin expertplex/local:refs/heads/feat/expertplex-b300
```

bundle 只包含已验证的源码、测试和文档提交，不包含实验原始数据。推送前应确认远端目标分支和本地 `expertplex/local` 的提交范围一致。

最终 bundle 对应当前已验证 HEAD，包含环境恢复后的 Ninja 路径修复、其 CPU 回归测试，以及文档化的 job569 cap-32 性能拒绝结果；不包含原始实验数据。

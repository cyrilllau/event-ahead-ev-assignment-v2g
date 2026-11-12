# 固定方案共同评价协议与输入

实现入口：`workflows.evaluation`；科学审计：`ev_assignment.validation.l08.audit_plan`。
规划输入 adapter 保持 OOS 关闭。评价独立 adapter 在全部计划接受并锁定后才读取样本。

清单 JSON（只含元数据，可在锁前读取）示例：

```json
{
  "schema_version": 1,
  "scope": "development",
  "domain_sha256": "64位科学域SHA256",
  "protocol": "L08_PAIRED_FIXED_PLAN_V1",
  "random_stream": "已经批准的精确随机流地址/协议引用",
  "samples": [{
    "sample_id": "配对样本唯一ID",
    "origin": "原生成协议、bank、原sample/seed及证据引用",
    "physical": {"path": "sample0/physical.npz", "sha256": "64位SHA256"},
    "route_states": {"path": "sample0/route_states.json", "sha256": "64位SHA256"}
  }]
}
```

`scope=oos` 仅用于该研究实际授权的独立 OOS 清单；不得把训练样本换名作为 OOS。
当前工程验收使用 `development` 两场景，不读取正式 OOS，不授予正式方案接受。
`domain_sha256` 由 `datasets.evaluation.domain_identity(problem)` 计算，包含网络、完整
路线顺序、能量界、固定时间域和车队，不包含方法不同的训练风险目标。三个方案须完全一致。

物理 NPZ 含 `line_operating_outages`（96×线路，原 recovery 当前故障钟）、
`load_p_kw`（96×母线，非负）、`load_q_kvar`（严格等于0.33P），无pickle。
不得把原场景包的错误时钟直接复制成评价数据；先按原协议形成96个15分钟时段的规范输入，
记录原文件/生成源码/时钟和哈希作为 origin。route_states JSON 的 `route_ids` 严格等于
冻结完整路线顺序，`states` 同长，每项 `{success: bool, arrival: 小时或null}`。
失败行 arrival=null；成功行有限且非负。不得重规划路线、重新生成流或按方法重采样。
整个清单精确SHA以及每个物理/回放文件SHA绑定在评价Run中；文件修改、错序、重复均拒绝。

`evaluate lock` 要求 DET/DRO/ARO 精确接受决定，重新审核接受方案，拒绝被 supersedes 的
旧决定；只封存清单，不读样本。`evaluate run` 读取已锁样本，使用原固定方案LP和物理审计，
每单元封存全部 primal/dual/有限界证据及成本组件。所有方法读取同一组样本；每项数据/见证
作为CAS依赖保存，离开服务器后仍能逐项重审。恢复只使用同Run/source/manifest中的已审单元。

记录总成本、第一阶段成本、recourse、critical/other ENS、实际 AC V2G 与shortfall。
第一阶段组件依次是 fixed、occupancy、reservation、contract；recourse 组件依次是
load shedding、shortfall、TOU discharge、degradation。元数据中附USD/kWh/AC单位。
配对差异默认相对DET，标准误为每场景差值的样本标准差除以√n；n=1时SE=null，不伪造零。
完成 requires 所有方法×样本、逐样本原审计、精确来源与费用封口。执行退出0不替代科学PASS。

只有已完成共同评价、完整 source/input/evidence 闭包、精确论文选择且真实 frozen_l08/OOS
范围的包才会标 paper_facing。开发示例永远是 development。论文投稿时仍需按研究spec审查
OOS清单的生成来源/样本独立性与统计解释，不能由一个范围标签自动证明这些研究事实。

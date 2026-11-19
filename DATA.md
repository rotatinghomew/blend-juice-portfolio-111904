# 资料字典

CSV 为 UTF-8，第一行为列名。日期为北京时间，时间带偏移；资料可见时刻与业务发生时刻分别以字段含义为准。空值不能默认做零。

### `pilots.csv`

试榨在班前决定，重复制样共享同一袋。历史校核不能把 `started_at` 之后知道的成品作为开工前信息。

字段：`pilot`、`kind`、`line`、`raw_kg`、`started_at`、`agreed_at`。

### `readings.csv`

`sample` 是试榨号或待加工批号；同袋重复读数不是独立样本。酸度单位质量百分比，糖度为 °Bx。

字段：`sample`、`layer`、`prep`、`replicate`、`version`、`brix`、`acid_percent`、`known_at`。

### `pilot_output.csv`

出汁净重、糖酸及损耗按同一试榨记录，重量为公斤，保留关联性。

字段：`pilot`、`version`、`usable_juice_kg`、`brix`、`acid_percent`、`loss_kg`、`known_at`。

### `raw.csv`

待加工批 `net_kg` 是计投料前毛库存，`splittable=0` 表示整批工艺加工，不能分开投料。已有投料记录需另扣。

字段：`lot`、`kind`、`rev`、`net_kg`、`splittable`、`expires`、`known_at`。

### `jobs.csv`

成品量和糖、酸区间为同时成立的合同条件。只有允许降档才能提出相应转级建议，不能自行确认客户变更。

字段：`job`、`grade`、`minimum_juice_kg`、`min_brix`、`max_brix`、`min_acid`、`max_acid`、`last_slot`、`customer_allows_lower`。

### `slots.csv`

一槽一批，一批可混可接受的原料。不同成品档换线需在允许清线的时段进行；档相同时可连用。

字段：`slot`、`line`、`day`、`period`、`max_raw_kg`、`tank_available`、`cleanup_allowed`。

### `commitments.csv`

锁料是预约总量，已投料数量为消耗并冲减该预约，成品行数量为已入库净汁重。

字段：`event`、`job`、`slot`、`lot`、`qty_kg`、`state`、`known_at`。

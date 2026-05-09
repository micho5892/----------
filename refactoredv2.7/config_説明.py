from config import SimConfig
import json

# SimConfig の構造と説明をすべて JSON スキーマとして抽出
schema = SimConfig.model_json_schema()

# 見やすく整形して出力
print(json.dumps(schema, indent=2, ensure_ascii=False))
```
project-root/
├── .env                          # APIキーなどの設定情報ファイル
├── .gitignore                    # .envをGitHubに同期させないための設定ファイル
├── gen_questions.py              # 297件のデータから質問・回答を生成するモジュール
├── prec_list_extraction.py       # 国家情報サイトAPIを利用してデータを収集するモジュール
├── extracting_only_question.py   # gen_questions.pyの結果から質問のみを抽出するモジュール
├── gen_answers_RAG.py            # 生成された質問に対してRAGで回答を生成するモジュール
├── requirements.txt              # 使用しているPythonライブラリ一覧
│
└── DATA/
    ├── generated_questions.json      # 生成されたQAデータ（JSON）
    ├── prec_list.json                # 収集した判例データ（2024-01-01〜2025-12-09）
    ├── prec_list_criminal.json       # prec_list.jsonから刑事事件のみを抽出したデータ
    │
    └── precedents/
        ├── 2019do11015.json          # 収集した判例の本文ファイル（計297件）
        ├── 2019do12345.json
        └── ...
```

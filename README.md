```
project-root/
├── .gitignore                    # .envをGitHubに同期させないための設定ファイル
├── README.md                     # READMEファイル
├── gen_questions.py              # 297件のデータから質問・回答を生成するモジュール
├── prec_list_extraction.py       # 国家情報サイトAPIを利用してデータを収集するモジュール
├── extracting_only_question.py   # gen_questions.pyの結果から質問のみを抽出するモジュール
├── gen_answers_RAG.py            # 生成された質問に対してRAGで回答を生成するモジュール
├── requirements.txt              # 使用しているPythonライブラリ一覧
├── parsed_results.json           # RAG生成された回答の解析結果
├── parsed_answers.py             # RAG生成された回答
├── cmp_results.py                # RAG生成された回答と正解を比較するモ듈
├── comparison_results.json       # 比較結果
└── DATA/
    ├── generated_questions.json      # 生成されたQAデータ（JSON）
    ├── prec_list.json                # 収集した判例データ（2024-01-01〜2025-12-09）
    ├── prec_list_criminal.json       # prec_list.jsonから刑事事件のみを抽出したデータ       
    ├── few_shot.json                 # few shot
    ├── questions_only.txt
    │
    ├─generated_answers/
    │  └─few_shot/
    │          final_results.json     # RAG基づく生成された回答 timestamp
    │          final_results.tsv      # RAG基づく生成された回答の内容
    └── precedents/
        ├── 2019do11015.json          # 収集した判例の本文ファイル（計297件）
        ├── 2019do12345.json
        └── ...
```
